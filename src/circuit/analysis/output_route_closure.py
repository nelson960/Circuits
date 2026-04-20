from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    _build_route_competition_pairs,
    _checkpoint_step_from_path,
    _head_contributions_for_layer,
    _holdout_pair_set,
    _validate_single_query_batch,
)
from circuit.analysis.route_to_margin_closure import fit_route_to_margin_closure
from circuit.analysis.route_to_scalar_closure import _switch_bucket_matches
from circuit.analysis.shared_feature_dynamics import _import_matplotlib
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import iter_jsonl, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


OUTPUT_ROUTE_CLOSURE_SCHEMA_VERSION = 1
OUTPUT_ROUTE_SCALARS = [
    "moving_answer_margin",
    "fixed_source_competitor_margin",
    "fixed_target_competitor_margin",
    "correct_value_logit",
    "source_best_wrong_logit",
    "target_best_wrong_logit",
    "negative_answer_loss",
]
OUTPUT_ROUTE_SWITCH_BUCKETS = ["all", "same_competitor", "competitor_switch"]
OUTPUT_ROUTE_MARGIN_SIDES = ["clean", "corrupted"]


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"Cannot compute mean for empty values: {label}")
    return float(sum(values) / len(values))


def _resolve_unique_values(
    *,
    values: list[str] | None,
    default_values: list[str],
    allowed_values: list[str],
    label: str,
) -> list[str]:
    raw_values = default_values if values is None else values
    if not raw_values:
        raise ValueError(f"{label} must not be empty.")
    resolved: list[str] = []
    for value in raw_values:
        if value not in allowed_values:
            raise ValueError(f"Unsupported {label} {value!r}; expected one of {allowed_values}.")
        if value not in resolved:
            resolved.append(value)
    return resolved


def _load_scalar_pair_rows(path: Path) -> list[dict[str, Any]]:
    rows = [row for row in iter_jsonl(path)]
    if not rows:
        raise RuntimeError(f"Scalar pair rows file is empty: {path}")
    return rows


def _filter_scalar_pair_rows(
    *,
    rows: list[dict[str, Any]],
    margin_sides: list[str],
    pair_types: list[str] | None,
    scalar_names: list[str],
) -> list[dict[str, Any]]:
    pair_type_filter = None if pair_types is None else set(pair_types)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if str(row["margin_side"]) not in margin_sides:
            continue
        if pair_type_filter is not None and str(row["pair_type"]) not in pair_type_filter:
            continue
        scalars = row.get("scalars")
        if not isinstance(scalars, dict):
            raise RuntimeError(f"Scalar pair row is missing scalars: {row.get('interval_pair_id')}")
        missing = [name for name in scalar_names if name not in scalars]
        if missing:
            raise KeyError(f"Scalar pair row {row.get('interval_pair_id')} is missing scalar names: {missing}")
        filtered.append(row)
    if not filtered:
        raise RuntimeError("No scalar pair rows survived output-route-closure filters.")
    return filtered


def _checkpoint_paths_by_step(rows: list[dict[str, Any]]) -> dict[int, Path]:
    paths_by_step: dict[int, Path] = {}
    for row in rows:
        for step_key, path_key in (("source_step", "source_checkpoint"), ("target_step", "target_checkpoint")):
            step = int(row[step_key])
            path = Path(str(row[path_key]))
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint path from scalar pair rows does not exist: {path}")
            if step in paths_by_step and paths_by_step[step] != path:
                raise RuntimeError(
                    f"Multiple checkpoint paths for step {step}: {paths_by_step[step]} and {path}"
                )
            paths_by_step[step] = path
    if not paths_by_step:
        raise RuntimeError("No checkpoint paths found in scalar pair rows.")
    return paths_by_step


def _selected_pairs_by_id(
    *,
    config_path: Path,
    probe_set_path: Path,
    pair_types: list[str],
    split_filter: list[str] | None,
    max_pairs_per_type: int,
    min_pairs_per_type: int,
    required_pair_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    pairs, pair_construction = _build_route_competition_pairs(
        probe_set_path=probe_set_path,
        spec=spec,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    pairs_by_id: dict[str, dict[str, Any]] = {str(pair["pair_id"]): pair for pair in pairs}
    missing = sorted(pair_id for pair_id in required_pair_ids if pair_id not in pairs_by_id)
    if missing:
        raise RuntimeError(
            "Reconstructed causal pairs do not cover scalar-pair rows. "
            f"Missing {len(missing)} pair ids; first missing ids: {missing[:10]}"
        )
    selected = {pair_id: pairs_by_id[pair_id] for pair_id in sorted(required_pair_ids)}
    if len(selected) != len(required_pair_ids):
        raise RuntimeError("Selected pair id count mismatch after reconstruction.")
    return selected, pair_construction


def _component_labels(num_layers: int, num_heads: int) -> list[str]:
    labels = ["embedding"]
    for layer_index in range(num_layers):
        for head_index in range(num_heads):
            labels.append(f"L{layer_index}H{head_index}")
        labels.append(f"L{layer_index}MLP")
    return labels


def _filter_component_labels(
    *,
    requested_components: list[str] | None,
    available_components: list[str],
) -> list[str]:
    if requested_components is None:
        return list(available_components)
    resolved: list[str] = []
    available = set(available_components)
    for component in requested_components:
        if component not in available:
            raise ValueError(f"Unsupported component {component!r}; expected one of {available_components}.")
        if component not in resolved:
            resolved.append(component)
    if not resolved:
        raise ValueError("component filter resolved to an empty component list.")
    return resolved


def _scalar_gradient_vectors(
    *,
    model: torch.nn.Module,
    final_residual_vectors: torch.Tensor,
    scalar_name: str,
    correct_token_ids: torch.Tensor,
    source_wrong_token_ids: torch.Tensor,
    target_wrong_token_ids: torch.Tensor,
    endpoint_kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if endpoint_kind not in {"source", "target"}:
        raise ValueError(f"Unsupported endpoint kind {endpoint_kind!r}.")
    residual = final_residual_vectors.detach().clone().requires_grad_(True)
    logits = model.lm_head(model.final_norm(residual))
    row_index = torch.arange(residual.size(0), device=residual.device)
    correct_logits = logits[row_index, correct_token_ids]
    source_wrong_logits = logits[row_index, source_wrong_token_ids]
    target_wrong_logits = logits[row_index, target_wrong_token_ids]
    if scalar_name == "moving_answer_margin":
        wrong_logits = source_wrong_logits if endpoint_kind == "source" else target_wrong_logits
        scalar_values = correct_logits - wrong_logits
    elif scalar_name == "fixed_source_competitor_margin":
        scalar_values = correct_logits - source_wrong_logits
    elif scalar_name == "fixed_target_competitor_margin":
        scalar_values = correct_logits - target_wrong_logits
    elif scalar_name == "correct_value_logit":
        scalar_values = correct_logits
    elif scalar_name == "source_best_wrong_logit":
        scalar_values = source_wrong_logits
    elif scalar_name == "target_best_wrong_logit":
        scalar_values = target_wrong_logits
    elif scalar_name == "negative_answer_loss":
        scalar_values = torch.log_softmax(logits, dim=-1)[row_index, correct_token_ids]
    else:
        raise ValueError(f"Unsupported output-route scalar {scalar_name!r}; expected one of {OUTPUT_ROUTE_SCALARS}.")
    gradients = torch.autograd.grad(scalar_values.sum(), residual)[0]
    return gradients.detach(), scalar_values.detach()


def _endpoint_key(*, step: int, margin_side: str) -> tuple[int, str]:
    return (int(step), str(margin_side))


def _build_endpoint_requests(
    *,
    scalar_pair_rows: list[dict[str, Any]],
    scalar_names: list[str],
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for row in scalar_pair_rows:
        for scalar_name in scalar_names:
            for endpoint_kind in ("source", "target"):
                requests.append(
                    {
                        "request_id": (
                            int(row["source_step"]),
                            int(row["target_step"]),
                            str(row["pair_id"]),
                            str(row["margin_side"]),
                            scalar_name,
                            endpoint_kind,
                        ),
                        "step": int(row[f"{endpoint_kind}_step"]),
                        "checkpoint": str(row[f"{endpoint_kind}_checkpoint"]),
                        "pair_id": str(row["pair_id"]),
                        "margin_side": str(row["margin_side"]),
                        "scalar_name": scalar_name,
                        "endpoint_kind": endpoint_kind,
                        "answer_target_id": int(row["answer_target_id"]),
                        "source_best_wrong_token_id": int(row["source_best_wrong_token_id"]),
                        "target_best_wrong_token_id": int(row["target_best_wrong_token_id"]),
                    }
                )
    if not requests:
        raise RuntimeError("No endpoint requests were built for output-route closure.")
    return requests


def _compute_endpoint_payloads(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    endpoint_keys: set[tuple[int, str]],
    component_labels: list[str],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[tuple[int, str, str], dict[str, Any]]:
    if not endpoint_keys:
        raise ValueError("endpoint_keys must not be empty.")
    available_components = set(component_labels)
    num_layers = len(model.blocks)
    payloads: dict[tuple[int, str, str], dict[str, Any]] = {}
    for step, margin_side in sorted(endpoint_keys):
        if margin_side not in OUTPUT_ROUTE_MARGIN_SIDES:
            raise ValueError(f"Unsupported margin side {margin_side!r}; expected one of {OUTPUT_ROUTE_MARGIN_SIDES}.")
        if step not in checkpoint_paths_by_step:
            raise KeyError(f"No checkpoint path for step {step}.")
        checkpoint_path = checkpoint_paths_by_step[step]
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        payload_step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if payload_step != step or path_step != step:
            raise RuntimeError(
                f"Checkpoint step mismatch for {checkpoint_path}: requested={step} payload={payload_step} path={path_step}"
            )
        side_key = f"{margin_side}_record"
        pair_ids = sorted(pairs_by_id)
        with torch.no_grad():
            for start_index in range(0, len(pair_ids), batch_size):
                batch_pair_ids = pair_ids[start_index : start_index + batch_size]
                pair_batch = [pairs_by_id[pair_id] for pair_id in batch_pair_ids]
                records = [pair[side_key] for pair in pair_batch]
                batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_residual_streams=True,
                )
                if outputs.residual_streams is None:
                    raise RuntimeError("Output-route closure requires residual streams.")
                answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
                _validate_single_query_batch(batch=batch, metadata=metadata, label="output-route closure")
                rows = metadata["rows"]
                prediction_positions = metadata["prediction_positions"]
                final_pre_stage = f"layer_{num_layers - 1}_post_mlp"
                final_pre_vectors = outputs.residual_streams[final_pre_stage][rows, prediction_positions, :]
                component_vectors: dict[str, torch.Tensor] = {}
                if "embedding" in available_components:
                    component_vectors["embedding"] = outputs.residual_streams["embedding"][rows, prediction_positions, :]
                pre_block_states = [outputs.residual_streams["embedding"]]
                for layer_index in range(1, num_layers):
                    pre_block_states.append(outputs.residual_streams[f"layer_{layer_index - 1}_post_mlp"])
                for layer_index, block in enumerate(model.blocks):
                    requested_heads = [
                        head_index
                        for head_index in range(block.attn.n_heads)
                        if f"L{layer_index}H{head_index}" in available_components
                    ]
                    if requested_heads:
                        head_contributions = _head_contributions_for_layer(
                            block=block,
                            pre_state=pre_block_states[layer_index],
                            attention_mask=batch["attention_mask"],
                        )
                        for head_index in requested_heads:
                            component_vectors[f"L{layer_index}H{head_index}"] = head_contributions[head_index][
                                rows,
                                prediction_positions,
                                :,
                            ]
                    mlp_label = f"L{layer_index}MLP"
                    if mlp_label in available_components:
                        component_vectors[mlp_label] = (
                            outputs.residual_streams[f"layer_{layer_index}_post_mlp"][rows, prediction_positions, :]
                            - outputs.residual_streams[f"layer_{layer_index}_post_attn"][rows, prediction_positions, :]
                        )
                missing = [component for component in component_labels if component not in component_vectors]
                if missing:
                    raise RuntimeError(f"Failed to compute requested component vectors: {missing}")
                predictions = answer_logits.argmax(dim=-1)
                for item_index, pair_id in enumerate(batch_pair_ids):
                    payloads[(step, margin_side, pair_id)] = {
                        "step": step,
                        "checkpoint": str(checkpoint_path),
                        "margin_side": margin_side,
                        "pair_id": pair_id,
                        "answer_target_id": int(answer_targets[item_index].detach().cpu().item()),
                        "answer_prediction_id": int(predictions[item_index].detach().cpu().item()),
                        "final_pre_vector": final_pre_vectors[item_index].detach(),
                        "component_vectors": {
                            component: component_vectors[component][item_index].detach()
                            for component in component_labels
                        },
                    }
    expected_count = len(endpoint_keys) * len(pairs_by_id)
    if len(payloads) != expected_count:
        raise RuntimeError(f"Endpoint payload count mismatch: expected={expected_count} got={len(payloads)}")
    return payloads


def _compute_endpoint_component_values(
    *,
    model: torch.nn.Module,
    requests: list[dict[str, Any]],
    endpoint_payloads: dict[tuple[int, str, str], dict[str, Any]],
    component_labels: list[str],
    device: torch.device,
) -> dict[tuple[int, int, str, str, str, str], dict[str, Any]]:
    values_by_request: dict[tuple[int, int, str, str, str, str], dict[str, Any]] = {}
    grouped_requests: dict[tuple[int, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        grouped_requests[
            (
                int(request["step"]),
                str(request["checkpoint"]),
                str(request["margin_side"]),
                str(request["scalar_name"]),
                str(request["endpoint_kind"]),
            )
        ].append(request)
    for (step, checkpoint_text, _, scalar_name, endpoint_kind), group in sorted(grouped_requests.items()):
        checkpoint_path = Path(checkpoint_text)
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        payload_step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if payload_step != step or path_step != step:
            raise RuntimeError(
                f"Endpoint gradient checkpoint step mismatch for {checkpoint_path}: "
                f"requested={step} payload={payload_step} path={path_step}"
            )
        final_residual_vectors: list[torch.Tensor] = []
        component_batches: dict[str, list[torch.Tensor]] = {component: [] for component in component_labels}
        correct_ids: list[int] = []
        source_wrong_ids: list[int] = []
        target_wrong_ids: list[int] = []
        for request in group:
            payload_key = (
                int(request["step"]),
                str(request["margin_side"]),
                str(request["pair_id"]),
            )
            if payload_key not in endpoint_payloads:
                raise KeyError(f"Missing endpoint payload for {payload_key}.")
            payload = endpoint_payloads[payload_key]
            expected_answer = int(request["answer_target_id"])
            if int(payload["answer_target_id"]) != expected_answer:
                raise RuntimeError(
                    f"Answer target mismatch for {payload_key}: scalar row={expected_answer} "
                    f"payload={payload['answer_target_id']}"
                )
            final_residual_vectors.append(payload["final_pre_vector"])
            for component in component_labels:
                component_batches[component].append(payload["component_vectors"][component])
            correct_ids.append(expected_answer)
            source_wrong_ids.append(int(request["source_best_wrong_token_id"]))
            target_wrong_ids.append(int(request["target_best_wrong_token_id"]))
        final_residual_batch = torch.stack(final_residual_vectors, dim=0).to(device)
        gradients, scalar_values = _scalar_gradient_vectors(
            model=model,
            final_residual_vectors=final_residual_batch,
            scalar_name=scalar_name,
            correct_token_ids=torch.tensor(correct_ids, device=device, dtype=torch.long),
            source_wrong_token_ids=torch.tensor(source_wrong_ids, device=device, dtype=torch.long),
            target_wrong_token_ids=torch.tensor(target_wrong_ids, device=device, dtype=torch.long),
            endpoint_kind=endpoint_kind,
        )
        component_value_tensors: dict[str, torch.Tensor] = {}
        for component in component_labels:
            component_batch = torch.stack(component_batches[component], dim=0).to(device)
            component_value_tensors[component] = (component_batch.float() * gradients.float()).sum(dim=-1)
        for item_index, request in enumerate(group):
            request_id = request["request_id"]
            values_by_request[request_id] = {
                "step": int(request["step"]),
                "checkpoint": str(request["checkpoint"]),
                "pair_id": str(request["pair_id"]),
                "margin_side": str(request["margin_side"]),
                "scalar_name": scalar_name,
                "endpoint_kind": endpoint_kind,
                "scalar_value_recomputed": float(scalar_values[item_index].detach().float().cpu().item()),
                "component_values": {
                    component: float(component_value_tensors[component][item_index].detach().float().cpu().item())
                    for component in component_labels
                },
            }
    if len(values_by_request) != len(requests):
        raise RuntimeError(f"Endpoint request value count mismatch: expected={len(requests)} got={len(values_by_request)}")
    return values_by_request


def _build_output_route_observations(
    *,
    scalar_pair_rows: list[dict[str, Any]],
    component_values_by_request: dict[tuple[int, int, str, str, str, str], dict[str, Any]],
    component_labels: list[str],
    scalar_names: list[str],
    switch_buckets: list[str],
    scalar_value_tolerance: float,
) -> list[dict[str, Any]]:
    if scalar_value_tolerance < 0.0:
        raise ValueError("scalar_value_tolerance must be non-negative.")
    observations: list[dict[str, Any]] = []
    for row in scalar_pair_rows:
        for scalar_name in scalar_names:
            scalar_payload = row["scalars"][scalar_name]
            source_request_id = (
                int(row["source_step"]),
                int(row["target_step"]),
                str(row["pair_id"]),
                str(row["margin_side"]),
                scalar_name,
                "source",
            )
            target_request_id = (
                int(row["source_step"]),
                int(row["target_step"]),
                str(row["pair_id"]),
                str(row["margin_side"]),
                scalar_name,
                "target",
            )
            if source_request_id not in component_values_by_request:
                raise KeyError(f"Missing source output-route request values for {source_request_id}.")
            if target_request_id not in component_values_by_request:
                raise KeyError(f"Missing target output-route request values for {target_request_id}.")
            source_values = component_values_by_request[source_request_id]
            target_values = component_values_by_request[target_request_id]
            source_scalar_delta = abs(float(source_values["scalar_value_recomputed"]) - float(scalar_payload["source"]))
            target_scalar_delta = abs(float(target_values["scalar_value_recomputed"]) - float(scalar_payload["target"]))
            if source_scalar_delta > scalar_value_tolerance or target_scalar_delta > scalar_value_tolerance:
                raise RuntimeError(
                    f"Scalar recomputation mismatch for {row['interval_pair_id']} scalar={scalar_name}: "
                    f"source_delta={source_scalar_delta:.6g} target_delta={target_scalar_delta:.6g} "
                    f"tolerance={scalar_value_tolerance:.6g}"
                )
            component_deltas = {
                component: float(target_values["component_values"][component]) - float(source_values["component_values"][component])
                for component in component_labels
            }
            for switch_bucket in switch_buckets:
                if not _switch_bucket_matches(row, switch_bucket):
                    continue
                observations.append(
                    {
                        "source_step": int(row["source_step"]),
                        "target_step": int(row["target_step"]),
                        "step_gap": int(row["step_gap"]),
                        "pair_id": str(row["pair_id"]),
                        "interval_pair_id": str(row["interval_pair_id"]),
                        "split": str(row["split"]),
                        "pair_type": str(row["pair_type"]),
                        "margin_side": str(row["margin_side"]),
                        "switch_bucket": switch_bucket,
                        "competitor_switched": bool(row["competitor_switched"]),
                        "scalar_name": scalar_name,
                        "source_scalar": float(scalar_payload["source"]),
                        "target_scalar": float(scalar_payload["target"]),
                        "actual_scalar_delta": float(scalar_payload["delta"]),
                        "component_deltas": component_deltas,
                    }
                )
    if not observations:
        raise RuntimeError("No output-route closure observations survived switch-bucket filters.")
    return observations


def _fit_output_route_bucket(
    *,
    observations: list[dict[str, Any]],
    component_labels: list[str],
    scalar_name: str,
    switch_bucket: str,
    fit_intercept: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        row
        for row in observations
        if str(row["scalar_name"]) == scalar_name and str(row["switch_bucket"]) == switch_bucket
    ]
    if not rows:
        raise RuntimeError(f"No output-route observations for scalar={scalar_name} switch_bucket={switch_bucket}.")
    component_delta_columns = {
        component: [float(row["component_deltas"][component]) for row in rows]
        for component in component_labels
    }
    scalar_deltas = [float(row["actual_scalar_delta"]) for row in rows]
    fit = fit_route_to_margin_closure(
        route_delta_columns=component_delta_columns,
        margin_deltas=scalar_deltas,
        fit_intercept=fit_intercept,
    )
    closure_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        fitted = dict(row)
        fitted["predicted_scalar_delta"] = float(fit["predicted_values"][index])
        fitted["closure_residual"] = float(fit["residual_values"][index])
        closure_rows.append(fitted)
    coefficient_rows: list[dict[str, Any]] = []
    for component in component_labels:
        mean_delta = _mean(component_delta_columns[component], label=f"{scalar_name}/{switch_bucket}/{component}")
        coefficient = float(fit["coefficients"][component])
        coefficient_rows.append(
            {
                "scalar_name": scalar_name,
                "switch_bucket": switch_bucket,
                "component": component,
                "coefficient": coefficient,
                "mean_component_delta": mean_delta,
                "mean_contribution": coefficient * mean_delta,
                "fit_intercept": fit_intercept,
                "num_observations": int(fit["num_observations"]),
                "matrix_rank": int(fit["matrix_rank"]),
                "num_parameters": int(fit["num_parameters"]),
                "rank_deficient": bool(fit["rank_deficient"]),
                "r_squared": fit["r_squared"],
                "mean_abs_residual": float(fit["mean_abs_residual"]),
            }
        )
    return fit, closure_rows, coefficient_rows


def _safe_r_squared(y_values: list[float], predicted_values: list[float]) -> float | None:
    if len(y_values) != len(predicted_values):
        raise ValueError("y_values and predicted_values must have same length.")
    mean_y = _mean(y_values, label="r2 y")
    sst = sum((value - mean_y) ** 2 for value in y_values)
    if sst <= 1.0e-12:
        return None
    sse = sum((actual - predicted) ** 2 for actual, predicted in zip(y_values, predicted_values, strict=True))
    return float(1.0 - (sse / sst))


def _summarize_output_route_closure(
    *,
    closure_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    fits_by_bucket: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    if not closure_rows:
        raise ValueError("Cannot summarize output-route closure without closure rows.")
    summary_rows: list[dict[str, Any]] = []
    for (scalar_name, switch_bucket), fit in sorted(fits_by_bucket.items()):
        rows = [
            row
            for row in closure_rows
            if str(row["scalar_name"]) == scalar_name and str(row["switch_bucket"]) == switch_bucket
        ]
        if not rows:
            raise RuntimeError(f"Missing closure rows for scalar={scalar_name} switch_bucket={switch_bucket}.")
        actual = [float(row["actual_scalar_delta"]) for row in rows]
        predicted = [float(row["predicted_scalar_delta"]) for row in rows]
        residuals = [float(row["closure_residual"]) for row in rows]
        summary_rows.append(
            {
                "scalar_name": scalar_name,
                "switch_bucket": switch_bucket,
                "num_observations": len(rows),
                "r_squared": _safe_r_squared(y_values=actual, predicted_values=predicted),
                "fit_r_squared": fit["r_squared"],
                "mean_actual_delta": _mean(actual, label=f"{scalar_name}/{switch_bucket}/actual"),
                "mean_predicted_delta": _mean(predicted, label=f"{scalar_name}/{switch_bucket}/predicted"),
                "mean_residual": _mean(residuals, label=f"{scalar_name}/{switch_bucket}/residual"),
                "mean_abs_residual": _mean(
                    [abs(value) for value in residuals],
                    label=f"{scalar_name}/{switch_bucket}/abs residual",
                ),
                "matrix_rank": int(fit["matrix_rank"]),
                "num_parameters": int(fit["num_parameters"]),
                "rank_deficient": bool(fit["rank_deficient"]),
            }
        )
    return {
        "num_observations": len(closure_rows),
        "scalar_summary_rows": summary_rows,
        "top_component_contributions": sorted(
            coefficient_rows,
            key=lambda row: (
                str(row["scalar_name"]),
                str(row["switch_bucket"]),
                -abs(float(row["mean_contribution"])),
            ),
        ),
    }


def _plot_output_route_actual_vs_predicted(
    *,
    closure_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not closure_rows:
        return None
    _, plt = _import_matplotlib()
    scalars = sorted({str(row["scalar_name"]) for row in closure_rows})
    fig, axes = plt.subplots(len(scalars), 1, figsize=(7, max(5, 4 * len(scalars))))
    if len(scalars) == 1:
        axes = [axes]
    for ax, scalar_name in zip(axes, scalars, strict=True):
        rows = [
            row
            for row in closure_rows
            if str(row["scalar_name"]) == scalar_name and str(row["switch_bucket"]) == "all"
        ]
        if not rows:
            continue
        actual = [float(row["actual_scalar_delta"]) for row in rows]
        predicted = [float(row["predicted_scalar_delta"]) for row in rows]
        ax.scatter(predicted, actual, alpha=0.55, color="#376f8f")
        values = actual + predicted
        min_value = min(values)
        max_value = max(values)
        if min_value == max_value:
            min_value -= 1.0
            max_value += 1.0
        ax.plot([min_value, max_value], [min_value, max_value], color="#777777", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#999999", linewidth=0.8)
        ax.axvline(0.0, color="#999999", linewidth=0.8)
        ax.set_title(f"{scalar_name}: output-route predicted vs actual")
        ax.set_xlabel("predicted scalar delta")
        ax.set_ylabel("actual scalar delta")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_output_route_r_squared(
    *,
    summary_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    rows = [row for row in summary_rows if str(row["switch_bucket"]) == "all"]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(rows)), 5))
    labels = [str(row["scalar_name"]) for row in rows]
    values = [0.0 if row["r_squared"] is None else float(row["r_squared"]) for row in rows]
    ax.bar(labels, values, color="#376f8f")
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_title("Output-route closure R squared")
    ax.set_ylabel("R squared")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_output_route_component_contributions(
    *,
    coefficient_rows: list[dict[str, Any]],
    output_path: Path,
    top_k_components: int,
) -> Path | None:
    rows = [
        row
        for row in coefficient_rows
        if str(row["switch_bucket"]) == "all"
    ]
    if not rows:
        return None
    if top_k_components <= 0:
        raise ValueError("top_k_components must be positive.")
    selected: list[dict[str, Any]] = []
    for scalar_name in sorted({str(row["scalar_name"]) for row in rows}):
        scalar_rows = [row for row in rows if str(row["scalar_name"]) == scalar_name]
        selected.extend(sorted(scalar_rows, key=lambda row: abs(float(row["mean_contribution"])), reverse=True)[:top_k_components])
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(selected)), 6))
    labels = [f"{row['scalar_name']}:{row['component']}" for row in selected]
    values = [float(row["mean_contribution"]) for row in selected]
    colors = ["#376f8f" if value >= 0.0 else "#8f374a" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_title("Top fitted output-route contributions")
    ax.set_ylabel("coefficient * mean component-DLA delta")
    ax.tick_params(axis="x", rotation=50)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_output_route_closure_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    lines = [
        "# Output-Route Closure",
        "",
        "## Calculation",
        "",
        "This report is not the earlier patch-transfer route closure. It tests whether output-space component DLA movement explains scalar movement.",
        "",
        "```text",
        "g_s(theta, x) = d scalar_s / d final_pre_layernorm_residual",
        "DLA_{c,s}(theta, x) = component_write_c(theta, x) dot g_s(theta, x)",
        "Delta DLA_{c,s} = DLA_{c,s}(theta_target, x) - DLA_{c,s}(theta_source, x)",
        "Delta scalar_s ~= sum_c beta_c Delta DLA_{c,s} + residual",
        "```",
        "",
        "The readout gradient goes through final layernorm. Attention rows decompose the head-dependent value paths; attention output projection bias is not assigned to individual heads.",
        "",
        "## Inputs",
        "",
        f"- scalar pair rows: `{report['scalar_pair_rows_path']}`",
        f"- probe set: `{report['probe_set_path']}`",
        f"- margin sides: `{report['margin_sides']}`",
        f"- pair types: `{report['pair_types']}`",
        f"- scalars: `{report['scalar_names']}`",
        f"- components: `{report['component_labels']}`",
        f"- fit intercept: `{bool(report['fit_intercept'])}`",
        "",
        "## Closure Summary",
        "",
        "| scalar | switch bucket | observations | R squared | mean actual | mean predicted | mean abs residual | rank |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["summary"]["scalar_summary_rows"]:
        r2 = row["r_squared"]
        lines.append(
            "| {scalar} | {bucket} | {n} | {r2} | {actual:.6g} | {pred:.6g} | {resid:.6g} | {rank}/{params} |".format(
                scalar=row["scalar_name"],
                bucket=row["switch_bucket"],
                n=int(row["num_observations"]),
                r2="" if r2 is None else f"{float(r2):.6f}",
                actual=float(row["mean_actual_delta"]),
                pred=float(row["mean_predicted_delta"]),
                resid=float(row["mean_abs_residual"]),
                rank=int(row["matrix_rank"]),
                params=int(row["num_parameters"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Component Contributions",
            "",
            "| scalar | bucket | component | coefficient | mean delta | mean contribution |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in report["summary"]["top_component_contributions"]:
        grouped[(str(row["scalar_name"]), str(row["switch_bucket"]))].append(row)
    for key in sorted(grouped):
        for row in sorted(grouped[key], key=lambda item: abs(float(item["mean_contribution"])), reverse=True)[: report["top_k_components"]]:
            lines.append(
                "| {scalar} | {bucket} | `{component}` | {coef:.6g} | {delta:.6g} | {contribution:.6g} |".format(
                    scalar=row["scalar_name"],
                    bucket=row["switch_bucket"],
                    component=row["component"],
                    coef=float(row["coefficient"]),
                    delta=float(row["mean_component_delta"]),
                    contribution=float(row["mean_contribution"]),
                )
            )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- closure rows: `{report['closure_rows_path']}`",
            f"- endpoint component rows: `{report['endpoint_component_rows_path']}`",
            f"- coefficient rows: `{report['coefficient_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
            "",
            "## Plots",
            "",
        ]
    )
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_output_route_closure(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    pair_types: list[str],
    device_name: str = "mps",
    scalar_names: list[str] | None = None,
    margin_sides: list[str] | None = None,
    switch_buckets: list[str] | None = None,
    component_labels: list[str] | None = None,
    split_filter: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    fit_intercept: bool = False,
    top_k_components: int = 8,
    scalar_value_tolerance: float = 1.0e-4,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if not pair_types:
        raise ValueError("output-route-closure requires at least one pair type.")
    if max_pairs_per_type <= 0:
        raise ValueError("max_pairs_per_type must be positive.")
    if min_pairs_per_type <= 0:
        raise ValueError("min_pairs_per_type must be positive.")
    if top_k_components <= 0:
        raise ValueError("top_k_components must be positive.")
    if scalar_value_tolerance < 0.0:
        raise ValueError("scalar_value_tolerance must be non-negative.")
    resolved_scalars = _resolve_unique_values(
        values=scalar_names,
        default_values=[
            "moving_answer_margin",
            "fixed_source_competitor_margin",
            "fixed_target_competitor_margin",
            "correct_value_logit",
            "source_best_wrong_logit",
            "target_best_wrong_logit",
            "negative_answer_loss",
        ],
        allowed_values=OUTPUT_ROUTE_SCALARS,
        label="scalar",
    )
    resolved_margin_sides = _resolve_unique_values(
        values=margin_sides,
        default_values=["clean"],
        allowed_values=OUTPUT_ROUTE_MARGIN_SIDES,
        label="margin side",
    )
    resolved_switch_buckets = _resolve_unique_values(
        values=switch_buckets,
        default_values=["all", "same_competitor", "competitor_switch"],
        allowed_values=OUTPUT_ROUTE_SWITCH_BUCKETS,
        label="switch bucket",
    )
    pair_types = sorted(set(pair_types), key=pair_types.index)
    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(device_name)
    model = build_model(spec.model, len(vocab.tokens), device)
    available_components = _component_labels(num_layers=len(model.blocks), num_heads=model.spec.n_heads)
    resolved_components = _filter_component_labels(
        requested_components=component_labels,
        available_components=available_components,
    )
    scalar_pair_rows = _filter_scalar_pair_rows(
        rows=_load_scalar_pair_rows(scalar_pair_rows_path),
        margin_sides=resolved_margin_sides,
        pair_types=pair_types,
        scalar_names=resolved_scalars,
    )
    checkpoint_paths_by_step = _checkpoint_paths_by_step(scalar_pair_rows)
    required_pair_ids = {str(row["pair_id"]) for row in scalar_pair_rows}
    pairs_by_id, pair_construction = _selected_pairs_by_id(
        config_path=config_path,
        probe_set_path=probe_set_path,
        pair_types=pair_types,
        split_filter=split_filter,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        required_pair_ids=required_pair_ids,
    )
    endpoint_keys = {
        _endpoint_key(step=int(row["source_step"]), margin_side=str(row["margin_side"]))
        for row in scalar_pair_rows
    }
    endpoint_keys.update(
        {
            _endpoint_key(step=int(row["target_step"]), margin_side=str(row["margin_side"]))
            for row in scalar_pair_rows
        }
    )
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[output-route-closure] "
        f"pairs={len(pairs_by_id)} scalar_rows={len(scalar_pair_rows)} endpoints={len(endpoint_keys)} "
        f"components={len(resolved_components)} scalars={resolved_scalars} device={device_name}",
        flush=True,
    )
    endpoint_payloads = _compute_endpoint_payloads(
        model=model,
        checkpoint_paths_by_step=checkpoint_paths_by_step,
        pairs_by_id=pairs_by_id,
        endpoint_keys=endpoint_keys,
        component_labels=resolved_components,
        batch_size=spec.evaluation.batch_size,
        pad_token_id=vocab.pad_token_id,
        device=device,
    )
    endpoint_requests = _build_endpoint_requests(
        scalar_pair_rows=scalar_pair_rows,
        scalar_names=resolved_scalars,
    )
    endpoint_component_values = _compute_endpoint_component_values(
        model=model,
        requests=endpoint_requests,
        endpoint_payloads=endpoint_payloads,
        component_labels=resolved_components,
        device=device,
    )
    endpoint_component_rows = [
        {
            **{
                "source_step": int(request_id[0]),
                "target_step": int(request_id[1]),
                "pair_id": str(request_id[2]),
                "margin_side": str(request_id[3]),
                "scalar_name": str(request_id[4]),
                "endpoint_kind": str(request_id[5]),
            },
            **value,
        }
        for request_id, value in sorted(endpoint_component_values.items())
    ]
    observations = _build_output_route_observations(
        scalar_pair_rows=scalar_pair_rows,
        component_values_by_request=endpoint_component_values,
        component_labels=resolved_components,
        scalar_names=resolved_scalars,
        switch_buckets=resolved_switch_buckets,
        scalar_value_tolerance=scalar_value_tolerance,
    )
    all_closure_rows: list[dict[str, Any]] = []
    all_coefficient_rows: list[dict[str, Any]] = []
    fits_by_bucket: dict[tuple[str, str], dict[str, Any]] = {}
    for scalar_name in resolved_scalars:
        for switch_bucket in resolved_switch_buckets:
            fit, closure_rows, coefficient_rows = _fit_output_route_bucket(
                observations=observations,
                component_labels=resolved_components,
                scalar_name=scalar_name,
                switch_bucket=switch_bucket,
                fit_intercept=fit_intercept,
            )
            fits_by_bucket[(scalar_name, switch_bucket)] = {
                key: value
                for key, value in fit.items()
                if key not in {"predicted_values", "residual_values"}
            }
            all_closure_rows.extend(closure_rows)
            all_coefficient_rows.extend(coefficient_rows)
            print(
                "[output-route-closure] fitted "
                f"scalar={scalar_name} switch_bucket={switch_bucket} "
                f"observations={fit['num_observations']} r_squared={fit['r_squared']}",
                flush=True,
            )
    summary = _summarize_output_route_closure(
        closure_rows=all_closure_rows,
        coefficient_rows=all_coefficient_rows,
        fits_by_bucket=fits_by_bucket,
    )
    closure_rows_path = output_dir / "output_route_closure_rows.jsonl"
    endpoint_component_rows_path = output_dir / "output_route_closure_endpoint_component_rows.jsonl"
    coefficient_rows_path = output_dir / "output_route_closure_coefficients.jsonl"
    pair_rows_path = output_dir / "output_route_closure_pairs.jsonl"
    report_path = output_dir / "output_route_closure_report.json"
    markdown_path = output_dir / "output_route_closure_report.md"
    write_jsonl(closure_rows_path, all_closure_rows)
    write_jsonl(endpoint_component_rows_path, endpoint_component_rows)
    write_jsonl(coefficient_rows_path, all_coefficient_rows)
    write_jsonl(pair_rows_path, [{key: value for key, value in pair.items() if key not in {"clean_record", "corrupted_record"}} for pair in pairs_by_id.values()])
    plot_paths: dict[str, Path] = {}
    actual_vs_predicted_path = _plot_output_route_actual_vs_predicted(
        closure_rows=all_closure_rows,
        output_path=output_dir / "output_route_closure_actual_vs_predicted.svg",
    )
    if actual_vs_predicted_path is not None:
        plot_paths["actual_vs_predicted"] = actual_vs_predicted_path
    r_squared_path = _plot_output_route_r_squared(
        summary_rows=summary["scalar_summary_rows"],
        output_path=output_dir / "output_route_closure_r_squared.svg",
    )
    if r_squared_path is not None:
        plot_paths["r_squared"] = r_squared_path
    component_contributions_path = _plot_output_route_component_contributions(
        coefficient_rows=all_coefficient_rows,
        output_path=output_dir / "output_route_closure_component_contributions.svg",
        top_k_components=top_k_components,
    )
    if component_contributions_path is not None:
        plot_paths["component_contributions"] = component_contributions_path
    report = {
        "schema_version": OUTPUT_ROUTE_CLOSURE_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "device": device_name,
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names": resolved_scalars,
        "switch_buckets": resolved_switch_buckets,
        "component_labels": resolved_components,
        "fit_intercept": fit_intercept,
        "top_k_components": top_k_components,
        "scalar_value_tolerance": scalar_value_tolerance,
        "checkpoint_paths_by_step": {str(step): str(path) for step, path in checkpoint_paths_by_step.items()},
        "pair_construction": pair_construction,
        "calculation": {
            "readout_gradient": "g_s = d scalar_s / d final_pre_layernorm_residual",
            "component_score": "DLA_{component,s} = component_residual_write dot g_s",
            "component_delta": "target endpoint DLA - source endpoint DLA",
            "closure": "actual scalar delta ~= fitted linear combination of component DLA deltas",
            "scope_note": (
                "This is an output-space DLA closure, not a causal patch-transfer route closure. "
                "It tests whether final-readout component movements explain scalar movements."
            ),
        },
        "closure_rows_path": str(closure_rows_path),
        "endpoint_component_rows_path": str(endpoint_component_rows_path),
        "coefficient_rows_path": str(coefficient_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_output_route_closure_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    print(
        f"[output-route-closure] complete report={report_path} rows={closure_rows_path}",
        flush=True,
    )
    return (
        report_path,
        markdown_path,
        closure_rows_path,
        endpoint_component_rows_path,
        coefficient_rows_path,
        pair_rows_path,
        plot_paths,
    )
