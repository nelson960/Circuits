from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.actual_batch_route_attribution import (
    _load_trace_batch_rows,
    _load_trace_step_rows,
    _records_by_sample_id,
    _records_for_trace_batch,
)
from circuit.analysis.bilinear_qk_rank_adam_state_attribution import (
    _adamw_component_updates,
    _optimizer_trace_metadata,
    _sub_tensors,
    _tensor_l2_norm,
)
from circuit.analysis.bilinear_qk_rank_data_attribution import (
    LOSS_SCOPES,
    _compute_loss_gradient_for_records_by_scope,
)
from circuit.analysis.bilinear_qk_rank_update_attribution import _assert_finite_gradients
from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.component_output_rescue import (
    _build_patch_groups,
    _component_groups_by_order,
    _component_order,
    _component_patch_stage,
    _component_write,
    _parse_local_component,
    _validate_downstream_components,
)
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    _build_route_gradient_decomposition_groups,
    _checkpoint_step_from_path,
    _gradient_dot_summary,
    _gradient_dot_summary_for_group,
    _group_metadata,
    _model_parameter_snapshot,
    _pair_metadata,
    _parameter_delta,
    _parameter_gradients,
    _resolve_checkpoint_paths,
    _safe_ratio,
    _sign_match,
    _validate_single_query_batch,
)
from circuit.analysis.output_component_causal_validation import _component_mask_kwargs
from circuit.analysis.output_route_closure import (
    OUTPUT_ROUTE_MARGIN_SIDES,
    OUTPUT_ROUTE_SCALARS,
    _build_endpoint_requests,
    _component_labels,
    _filter_component_labels,
    _filter_scalar_pair_rows,
    _load_scalar_pair_rows,
    _mean,
    _resolve_unique_values,
    _selected_pairs_by_id,
)
from circuit.analysis.residual_state_rescue import _validate_maskable_components
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.train import _resume_training_state
from circuit.vocab import Vocabulary


COMPONENT_OUTPUT_RESCUE_ADAM_STATE_ATTRIBUTION_SCHEMA_VERSION = 1
ADAM_COMPONENT_NAMES = [
    "raw_sgd",
    "clipped_sgd",
    "adam_current_gradient",
    "adam_historical_momentum",
    "adam_preconditioned",
    "weight_decay",
    "reconstructed_adamw_update",
]


def _default_parameter_group_ids(*, source_components: list[str], patch_components: list[str]) -> list[str]:
    ids = ["global:all_named_parameters"]
    for component in [*source_components, *patch_components]:
        kind, layer, head = _parse_local_component(component)
        if kind == "head":
            if head is None:
                raise RuntimeError(f"Parsed head component without head index: {component}")
            module_id = f"module:L{layer}.attention"
            head_id = f"attention_head:L{layer}H{head}.qkvo"
            for group_id in (module_id, head_id):
                if group_id not in ids:
                    ids.append(group_id)
        elif kind == "mlp":
            group_id = f"module:L{layer}.mlp"
            if group_id not in ids:
                ids.append(group_id)
        else:
            raise RuntimeError(f"Unsupported parsed component kind {kind!r} for {component!r}.")
    return ids


def _resolve_parameter_groups(
    *,
    model: torch.nn.Module,
    source_components: list[str],
    patch_components: list[str],
    parameter_group_ids: list[str] | None,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    groups, decomposition_summary = _build_route_gradient_decomposition_groups(
        model=model,
        decomposition_modes=["module_blocks", "attention_heads", "mlp_neurons"],
    )
    groups_by_id = {group.group_id: group for group in groups}
    requested_ids = (
        _default_parameter_group_ids(source_components=source_components, patch_components=patch_components)
        if parameter_group_ids is None
        else list(parameter_group_ids)
    )
    if not requested_ids:
        raise ValueError("At least one parameter group is required.")
    duplicate_ids = sorted({group_id for group_id in requested_ids if requested_ids.count(group_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate parameter group ids: {duplicate_ids}")
    missing = [group_id for group_id in requested_ids if group_id not in groups_by_id]
    if missing:
        raise ValueError(f"Unknown parameter group id(s) {missing}; available groups include {sorted(groups_by_id)}.")
    selected_groups = [groups_by_id[group_id] for group_id in requested_ids]
    model_parameters = dict(model.named_parameters(remove_duplicate=False))
    group_rows = [_group_metadata(model_parameters=model_parameters, group=group) for group in selected_groups]
    summary = {
        **decomposition_summary,
        "selected_group_ids": requested_ids,
        "selected_num_groups": len(selected_groups),
        "selected_group_note": (
            "Default groups include global parameters plus module/head groups for source and patch components. "
            "Module and head groups overlap; compare rows within the same partition_name."
        ),
    }
    return selected_groups, group_rows, summary


def _scalar_tensor_from_logits(
    *,
    logits: torch.Tensor,
    scalar_name: str,
    answer_target_id: int,
    source_best_wrong_token_id: int,
    target_best_wrong_token_id: int,
    endpoint_kind: str,
) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError(f"Expected rank-1 logits for scalar calculation, got shape {tuple(logits.shape)}.")
    if endpoint_kind not in {"source", "target"}:
        raise ValueError(f"Unsupported endpoint kind {endpoint_kind!r}; expected source or target.")
    correct = logits[int(answer_target_id)]
    source_wrong = logits[int(source_best_wrong_token_id)]
    target_wrong = logits[int(target_best_wrong_token_id)]
    if scalar_name == "moving_answer_margin":
        wrong = source_wrong if endpoint_kind == "source" else target_wrong
        return correct - wrong
    if scalar_name == "fixed_source_competitor_margin":
        return correct - source_wrong
    if scalar_name == "fixed_target_competitor_margin":
        return correct - target_wrong
    if scalar_name == "correct_value_logit":
        return correct
    if scalar_name == "source_best_wrong_logit":
        return source_wrong
    if scalar_name == "target_best_wrong_logit":
        return target_wrong
    if scalar_name == "negative_answer_loss":
        return torch.log_softmax(logits, dim=-1)[int(answer_target_id)]
    raise ValueError(f"Unsupported scalar {scalar_name!r}; expected one of {OUTPUT_ROUTE_SCALARS}.")


def _scalar_tensor_payloads(
    *,
    answer_logits: torch.Tensor,
    answer_targets: torch.Tensor,
    metadata: dict[str, Any],
    batch_pair_ids: list[str],
    request_specs_by_pair_id: dict[str, list[dict[str, Any]]],
    label: str,
) -> dict[tuple[int, int, str, str, str, str], torch.Tensor]:
    if len(metadata["rows"]) != len(batch_pair_ids):
        raise RuntimeError(
            f"{label} expected one answer row per pair: pairs={len(batch_pair_ids)} rows={len(metadata['rows'])}"
        )
    values: dict[tuple[int, int, str, str, str, str], torch.Tensor] = {}
    for item_index, pair_id in enumerate(batch_pair_ids):
        if pair_id not in request_specs_by_pair_id:
            continue
        target_id = int(answer_targets[item_index].detach().cpu().item())
        logits = answer_logits[item_index]
        for request in request_specs_by_pair_id[pair_id]:
            expected_target = int(request["answer_target_id"])
            if target_id != expected_target:
                raise RuntimeError(
                    f"{label} answer target mismatch for pair={pair_id}: expected={expected_target} got={target_id}"
                )
            request_id = request["request_id"]
            values[request_id] = _scalar_tensor_from_logits(
                logits=logits,
                scalar_name=str(request["scalar_name"]),
                answer_target_id=expected_target,
                source_best_wrong_token_id=int(request["source_best_wrong_token_id"]),
                target_best_wrong_token_id=int(request["target_best_wrong_token_id"]),
                endpoint_kind=str(request["endpoint_kind"]),
            )
    return values


def _request_maps(
    *,
    scalar_pair_rows: list[dict[str, Any]],
    scalar_names: list[str],
    endpoint_kind: str,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str, str, str, str], dict[str, Any]]]:
    if endpoint_kind not in {"source", "target"}:
        raise ValueError(f"Unsupported endpoint kind {endpoint_kind!r}.")
    requests: list[dict[str, Any]] = []
    for request in _build_endpoint_requests(scalar_pair_rows=scalar_pair_rows, scalar_names=scalar_names):
        if str(request["endpoint_kind"]) != endpoint_kind:
            continue
        source_row = next(
            row
            for row in scalar_pair_rows
            if int(row["source_step"]) == int(request["request_id"][0])
            and int(row["target_step"]) == int(request["request_id"][1])
            and str(row["pair_id"]) == str(request["request_id"][2])
            and str(row["margin_side"]) == str(request["request_id"][3])
        )
        request["scalar_payload"] = source_row["scalars"][str(request["scalar_name"])]
        request["pair_type"] = str(source_row["pair_type"])
        requests.append(request)
    if not requests:
        raise RuntimeError(f"No {endpoint_kind} endpoint requests were built.")
    return requests, {request["request_id"]: request for request in requests}


def _component_output_rescue_score_payloads(
    *,
    model: torch.nn.Module,
    pairs_by_id: dict[str, dict[str, Any]],
    requests: list[dict[str, Any]],
    request_by_id: dict[tuple[int, int, str, str, str, str], dict[str, Any]],
    source_component: str,
    patch_groups: list[dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
    scalar_value_tolerance: float,
    device: torch.device,
    track_grad: bool,
    validate_clean_scalar: bool = True,
) -> dict[str, dict[str, Any]]:
    num_layers = len(model.blocks)
    num_heads = int(model.spec.n_heads)
    source_mask_kwargs = _component_mask_kwargs(
        component=source_component,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
    )
    request_specs_by_pair_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        request_specs_by_pair_id[str(request["pair_id"])].append(request)
    pair_ids = sorted(pairs_by_id)
    rescue_sums: dict[tuple[str, str], torch.Tensor | None] = {}
    clean_sums: dict[tuple[str, str], torch.Tensor | None] = {}
    ablated_sums: dict[tuple[str, str], torch.Tensor | None] = {}
    patched_sums: dict[tuple[str, str], torch.Tensor | None] = {}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    rescue_values: dict[tuple[str, str], list[float]] = defaultdict(list)

    grad_context = torch.enable_grad() if track_grad else torch.no_grad()
    with grad_context:
        for start_index in range(0, len(pair_ids), batch_size):
            batch_pair_ids = pair_ids[start_index : start_index + batch_size]
            records = [
                pairs_by_id[pair_id][f"{str(request_specs_by_pair_id[pair_id][0]['margin_side'])}_record"]
                for pair_id in batch_pair_ids
                if pair_id in request_specs_by_pair_id
            ]
            requested_batch_pair_ids = [pair_id for pair_id in batch_pair_ids if pair_id in request_specs_by_pair_id]
            if not requested_batch_pair_ids:
                continue
            if len(records) != len(requested_batch_pair_ids):
                raise RuntimeError("Record/pair id batch construction mismatch.")
            batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
            clean_outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
            )
            source_ablated_outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
                **source_mask_kwargs,
            )
            if clean_outputs.residual_streams is None or source_ablated_outputs.residual_streams is None:
                raise RuntimeError("component-output-rescue Adam attribution requires residual streams.")
            clean_logits, clean_targets, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            ablated_logits, ablated_targets, ablated_metadata = extract_answer_logits(source_ablated_outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=clean_metadata, label="component-output-rescue-adam clean")
            _validate_single_query_batch(
                batch=batch,
                metadata=ablated_metadata,
                label="component-output-rescue-adam ablated",
            )
            batch_request_specs = {
                pair_id: request_specs_by_pair_id[pair_id]
                for pair_id in requested_batch_pair_ids
            }
            clean_tensors = _scalar_tensor_payloads(
                answer_logits=clean_logits,
                answer_targets=clean_targets,
                metadata=clean_metadata,
                batch_pair_ids=requested_batch_pair_ids,
                request_specs_by_pair_id=batch_request_specs,
                label="component-output-rescue-adam clean",
            )
            ablated_tensors = _scalar_tensor_payloads(
                answer_logits=ablated_logits,
                answer_targets=ablated_targets,
                metadata=ablated_metadata,
                batch_pair_ids=requested_batch_pair_ids,
                request_specs_by_pair_id=batch_request_specs,
                label="component-output-rescue-adam ablated",
            )

            patched_tensors_by_group: dict[str, dict[tuple[int, int, str, str, str, str], torch.Tensor]] = {}
            for patch_group in patch_groups:
                patch_group_id = str(patch_group["patch_group_id"])
                patch_components = [str(component) for component in patch_group["patch_components"]]
                residual_patch: dict[str, torch.Tensor] = {}
                for _, ordered_components in _component_groups_by_order(patch_components):
                    current_outputs = model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_residual_streams=True,
                        residual_patch=residual_patch,
                        **source_mask_kwargs,
                    )
                    if current_outputs.residual_streams is None:
                        raise RuntimeError("component-output-rescue Adam attribution requires current residual streams.")
                    stage_deltas: dict[str, torch.Tensor] = {}
                    for patch_component in ordered_components:
                        patch_stage = _component_patch_stage(patch_component)
                        clean_write = _component_write(
                            model=model,
                            component=patch_component,
                            residual_streams=clean_outputs.residual_streams,
                            attention_mask=batch["attention_mask"],
                        )
                        current_write = _component_write(
                            model=model,
                            component=patch_component,
                            residual_streams=current_outputs.residual_streams,
                            attention_mask=batch["attention_mask"],
                        )
                        delta = clean_write - current_write
                        if patch_stage in stage_deltas:
                            stage_deltas[patch_stage] = stage_deltas[patch_stage] + delta
                        else:
                            stage_deltas[patch_stage] = delta
                    for patch_stage, delta in stage_deltas.items():
                        residual_patch[patch_stage] = current_outputs.residual_streams[patch_stage] + delta
                patched_outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    residual_patch=residual_patch,
                    **source_mask_kwargs,
                )
                patched_logits, patched_targets, patched_metadata = extract_answer_logits(patched_outputs.logits, batch)
                _validate_single_query_batch(
                    batch=batch,
                    metadata=patched_metadata,
                    label=f"component-output-rescue-adam patched {patch_group_id}",
                )
                patched_tensors_by_group[patch_group_id] = _scalar_tensor_payloads(
                    answer_logits=patched_logits,
                    answer_targets=patched_targets,
                    metadata=patched_metadata,
                    batch_pair_ids=requested_batch_pair_ids,
                    request_specs_by_pair_id=batch_request_specs,
                    label=f"component-output-rescue-adam patched {patch_group_id}",
                )

            for request_id, clean_scalar in clean_tensors.items():
                request = request_by_id[request_id]
                scalar_payload = request.get("scalar_payload")
                if scalar_payload is None:
                    raise KeyError(f"Request is missing scalar_payload: {request_id}")
                if validate_clean_scalar:
                    expected = float(scalar_payload[str(request["endpoint_kind"])])
                    actual_clean = float(clean_scalar.detach().float().cpu().item())
                    delta = abs(actual_clean - expected)
                    if delta > scalar_value_tolerance:
                        raise RuntimeError(
                            f"Clean scalar mismatch for {request_id}: expected={expected:.6g} clean={actual_clean:.6g} "
                            f"delta={delta:.6g} tolerance={scalar_value_tolerance:.6g}"
                        )
                if request_id not in ablated_tensors:
                    raise KeyError(f"Missing ablated scalar for request {request_id}.")
                ablated_scalar = ablated_tensors[request_id]
                for patch_group in patch_groups:
                    patch_group_id = str(patch_group["patch_group_id"])
                    patched_tensors = patched_tensors_by_group[patch_group_id]
                    if request_id not in patched_tensors:
                        raise KeyError(f"Missing patched scalar for request {request_id} group={patch_group_id}.")
                    patched_scalar = patched_tensors[request_id]
                    rescue = patched_scalar - ablated_scalar
                    key = (patch_group_id, str(request["scalar_name"]))
                    rescue_sums[key] = rescue if rescue_sums.get(key) is None else rescue_sums[key] + rescue
                    clean_sums[key] = clean_scalar if clean_sums.get(key) is None else clean_sums[key] + clean_scalar
                    ablated_sums[key] = (
                        ablated_scalar if ablated_sums.get(key) is None else ablated_sums[key] + ablated_scalar
                    )
                    patched_sums[key] = (
                        patched_scalar if patched_sums.get(key) is None else patched_sums[key] + patched_scalar
                    )
                    counts[key] += 1
                    rescue_values[key].append(float(rescue.detach().float().cpu().item()))

    payloads: dict[str, dict[str, Any]] = {}
    for scalar_index, key in enumerate(sorted(rescue_sums)):
        patch_group_id, scalar_name = key
        total_rescue = rescue_sums[key]
        total_clean = clean_sums[key]
        total_ablated = ablated_sums[key]
        total_patched = patched_sums[key]
        count = counts[key]
        if total_rescue is None or total_clean is None or total_ablated is None or total_patched is None or count <= 0:
            raise RuntimeError(f"Component-output rescue score produced no values for {key}.")
        mean_rescue = total_rescue / float(count)
        score_name = f"{patch_group_id}:{scalar_name}:component_output_rescue"
        payload: dict[str, Any] = {
            "score_name": score_name,
            "patch_group_id": patch_group_id,
            "scalar_name": scalar_name,
            "score_value": float(mean_rescue.detach().float().cpu().item()),
            "clean_score_value": float((total_clean / float(count)).detach().float().cpu().item()),
            "ablated_score_value": float((total_ablated / float(count)).detach().float().cpu().item()),
            "patched_score_value": float((total_patched / float(count)).detach().float().cpu().item()),
            "score_value_abs_mean": _mean([abs(value) for value in rescue_values[key]], label=f"{score_name} abs"),
            "num_entries": count,
            "num_pairs": len(pairs_by_id),
        }
        if track_grad:
            if not mean_rescue.requires_grad:
                raise RuntimeError(f"Component-output rescue scalar does not require grad: {score_name}")
            model.zero_grad(set_to_none=True)
            mean_rescue.backward(retain_graph=scalar_index < len(rescue_sums) - 1)
            gradients, zero_gradient_parameter_names = _parameter_gradients(model=model, require_all=False)
            model.zero_grad(set_to_none=True)
            payload["gradients"] = gradients
            payload["zero_gradient_parameter_names"] = zero_gradient_parameter_names
        payloads[score_name] = payload
    return payloads


def _component_group_row(
    *,
    metric_row: dict[str, Any],
    component_name: str,
    parameter_group: Any,
    component_tensors: dict[str, torch.Tensor],
    scalar_gradients: dict[str, torch.Tensor],
    actual_delta_parameters: dict[str, torch.Tensor],
) -> dict[str, Any]:
    component_dot = _gradient_dot_summary_for_group(
        left_gradients=component_tensors,
        right_gradients=scalar_gradients,
        group=parameter_group,
        label=(
            f"component-output rescue Adam component {component_name} {parameter_group.group_id} "
            f"{metric_row['source_step']}->{metric_row['target_step']} {metric_row['score_name']}"
        ),
    )
    actual_dot = _gradient_dot_summary_for_group(
        left_gradients=actual_delta_parameters,
        right_gradients=scalar_gradients,
        group=parameter_group,
        label=(
            f"component-output rescue actual update {parameter_group.group_id} "
            f"{metric_row['source_step']}->{metric_row['target_step']} {metric_row['score_name']}"
        ),
    )
    component_delta = float(component_dot["dot"])
    actual_delta = float(actual_dot["dot"])
    return {
        "source_step": int(metric_row["source_step"]),
        "target_step": int(metric_row["target_step"]),
        "step_gap": int(metric_row["step_gap"]),
        "source_checkpoint": metric_row["source_checkpoint"],
        "target_checkpoint": metric_row["target_checkpoint"],
        "optimizer_trace_dir": metric_row["optimizer_trace_dir"],
        "learning_rate": float(metric_row["learning_rate"]),
        "source_component": metric_row["source_component"],
        "patch_group_id": metric_row["patch_group_id"],
        "patch_components": list(metric_row["patch_components"]),
        "patch_stages": list(metric_row["patch_stages"]),
        "score_name": metric_row["score_name"],
        "scalar_name": metric_row["scalar_name"],
        "margin_side": metric_row["margin_side"],
        "pair_types": list(metric_row["pair_types"]),
        "component": component_name,
        "parameter_group_id": parameter_group.group_id,
        "parameter_group_kind": parameter_group.group_kind,
        "component_type": parameter_group.component_type,
        "partition_name": parameter_group.partition_name,
        "group_layer": parameter_group.layer,
        "group_head": parameter_group.head,
        "group_projection": parameter_group.projection,
        "selection_count": len(parameter_group.selections),
        "num_selected_parameters": int(component_dot["num_parameters"]),
        "component_scalar_delta": component_delta,
        "actual_update_predicted_scalar_delta_for_group": actual_delta,
        "component_l2_norm": float(component_dot["left_l2_norm"]),
        "scalar_gradient_l2_norm": float(component_dot["right_l2_norm"]),
        "component_scalar_gradient_cosine": component_dot["cosine"],
        "component_fraction_of_actual_update_prediction_for_group": _safe_ratio(component_delta, actual_delta),
        "component_fraction_of_global_actual_update_prediction": _safe_ratio(
            component_delta,
            float(metric_row["actual_update_predicted_scalar_delta"]),
        ),
        "notes": list(parameter_group.notes),
    }


def _summarize(
    *,
    metric_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not metric_rows:
        raise RuntimeError("Cannot summarize empty component-output rescue Adam metric rows.")
    scalar_rows: list[dict[str, Any]] = []
    scalar_keys = sorted({(str(row["score_name"]), str(row["patch_group_id"]), str(row["scalar_name"])) for row in metric_rows})
    for score_name, patch_group_id, scalar_name in scalar_keys:
        rows = [
            row
            for row in metric_rows
            if str(row["score_name"]) == score_name
            and str(row["patch_group_id"]) == patch_group_id
            and str(row["scalar_name"]) == scalar_name
        ]
        scalar_rows.append(
            {
                "score_name": score_name,
                "patch_group_id": patch_group_id,
                "scalar_name": scalar_name,
                "num_intervals": len(rows),
                "sum_actual_score_delta": sum(float(row["actual_score_delta"]) for row in rows),
                "sum_actual_update_predicted_scalar_delta": sum(
                    float(row["actual_update_predicted_scalar_delta"]) for row in rows
                ),
                "sum_raw_sgd_scalar_delta": sum(float(row["raw_sgd_scalar_delta"]) for row in rows),
                "sum_clipped_sgd_scalar_delta": sum(float(row["clipped_sgd_scalar_delta"]) for row in rows),
                "sum_adam_current_gradient_scalar_delta": sum(
                    float(row["adam_current_gradient_scalar_delta"]) for row in rows
                ),
                "sum_adam_historical_momentum_scalar_delta": sum(
                    float(row["adam_historical_momentum_scalar_delta"]) for row in rows
                ),
                "sum_adam_preconditioned_scalar_delta": sum(float(row["adam_preconditioned_scalar_delta"]) for row in rows),
                "sum_weight_decay_scalar_delta": sum(float(row["weight_decay_scalar_delta"]) for row in rows),
                "sum_reconstructed_adamw_scalar_delta": sum(
                    float(row["reconstructed_adamw_scalar_delta"]) for row in rows
                ),
                "actual_update_sign_match_rate": _mean(
                    [1.0 if bool(row["actual_update_scalar_sign_match"]) else 0.0 for row in rows],
                    label=f"{score_name} sign match",
                ),
                "reconstructed_sign_match_rate": _mean(
                    [1.0 if bool(row["reconstructed_adamw_sign_match"]) else 0.0 for row in rows],
                    label=f"{score_name} reconstructed sign match",
                ),
                "mean_reconstructed_adamw_relative_error": _mean(
                    [float(row["reconstructed_adamw_relative_error"]) for row in rows],
                    label=f"{score_name} reconstructed relative error",
                ),
            }
        )

    component_summary_rows: list[dict[str, Any]] = []
    component_keys = sorted(
        {
            (
                str(row["score_name"]),
                str(row["patch_group_id"]),
                str(row["scalar_name"]),
                str(row["parameter_group_id"]),
                str(row["component"]),
            )
            for row in component_rows
        }
    )
    for score_name, patch_group_id, scalar_name, parameter_group_id, component_name in component_keys:
        rows = [
            row
            for row in component_rows
            if str(row["score_name"]) == score_name
            and str(row["patch_group_id"]) == patch_group_id
            and str(row["scalar_name"]) == scalar_name
            and str(row["parameter_group_id"]) == parameter_group_id
            and str(row["component"]) == component_name
        ]
        component_summary_rows.append(
            {
                "score_name": score_name,
                "patch_group_id": patch_group_id,
                "scalar_name": scalar_name,
                "parameter_group_id": parameter_group_id,
                "component": component_name,
                "num_intervals": len(rows),
                "sum_component_scalar_delta": sum(float(row["component_scalar_delta"]) for row in rows),
                "sum_actual_update_predicted_scalar_delta_for_group": sum(
                    float(row["actual_update_predicted_scalar_delta_for_group"]) for row in rows
                ),
            }
        )
    return {
        "scalar_rows": scalar_rows,
        "component_summary_rows": component_summary_rows,
    }


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Component Output Rescue Adam State Attribution",
        "",
        "This report decomposes the exact AdamW update for a behavioral component-output rescue scalar.",
        "",
        "The scalar is:",
        "",
        "`C = mean[scalar(source ablated + clean downstream component write patch) - scalar(source ablated)]`",
        "",
        "## Summary",
        "",
        "| score | intervals | actual delta | actual-update pred | raw SGD | Adam current | Adam momentum | weight decay | reconstructed | recon rel err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]["scalar_rows"]:
        lines.append(
            "| `{score}` | {n} | {actual:.6g} | {pred:.6g} | {raw:.6g} | {current:.6g} | {momentum:.6g} | {decay:.6g} | {recon:.6g} | {err:.6g} |".format(
                score=row["score_name"],
                n=int(row["num_intervals"]),
                actual=float(row["sum_actual_score_delta"]),
                pred=float(row["sum_actual_update_predicted_scalar_delta"]),
                raw=float(row["sum_raw_sgd_scalar_delta"]),
                current=float(row["sum_adam_current_gradient_scalar_delta"]),
                momentum=float(row["sum_adam_historical_momentum_scalar_delta"]),
                decay=float(row["sum_weight_decay_scalar_delta"]),
                recon=float(row["sum_reconstructed_adamw_scalar_delta"]),
                err=float(row["mean_reconstructed_adamw_relative_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- metric rows: `{report['metric_rows_path']}`",
            f"- component rows: `{report['component_rows_path']}`",
            f"- group rows: `{report['group_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_component_output_rescue_adam_state_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    optimizer_trace_dir: Path,
    output_dir: Path,
    pair_types: list[str],
    source_component: str,
    patch_components: list[str] | None,
    patch_groups: list[str] | None = None,
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    scalar_names: list[str] | None = None,
    margin_sides: list[str] | None = None,
    split_filter: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    batch_size: int | None = None,
    loss_scope: str = "full_lm",
    loss_match_tolerance: float = 1.0e-4,
    grad_norm_match_tolerance: float = 1.0e-4,
    scalar_value_tolerance: float = 1.0e-4,
    min_error_denominator: float = 1.0e-9,
    parameter_group_ids: list[str] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    if not pair_types:
        raise ValueError("component-output-rescue-adam-state-attribution requires at least one pair type.")
    if max_pairs_per_type <= 0:
        raise ValueError("max_pairs_per_type must be positive.")
    if min_pairs_per_type <= 0:
        raise ValueError("min_pairs_per_type must be positive.")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    if loss_scope not in LOSS_SCOPES:
        raise ValueError(f"Unsupported loss_scope {loss_scope!r}; expected one of {LOSS_SCOPES}.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    if scalar_value_tolerance < 0.0:
        raise ValueError("scalar_value_tolerance must be non-negative.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_scalars = _resolve_unique_values(
        values=scalar_names,
        default_values=["correct_value_logit", "fixed_source_competitor_margin", "fixed_target_competitor_margin"],
        allowed_values=OUTPUT_ROUTE_SCALARS,
        label="scalar",
    )
    resolved_margin_sides = _resolve_unique_values(
        values=margin_sides,
        default_values=["clean"],
        allowed_values=OUTPUT_ROUTE_MARGIN_SIDES,
        label="margin side",
    )
    pair_types = sorted(set(pair_types), key=pair_types.index)
    spec = TrainSpec.from_path(config_path)
    if device_name is not None:
        spec = replace(spec, device=device_name)
    if float(spec.model.dropout) != 0.0:
        raise RuntimeError("Component-output rescue Adam state attribution requires dropout=0.0.")
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(spec.device)
    analysis_batch_size = int(spec.evaluation.batch_size) if batch_size is None else int(batch_size)

    checkpoint_dir = optimizer_trace_dir / "checkpoints"
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if start_step is not None:
        checkpoints = [path for path in checkpoints if _checkpoint_step_from_path(path) >= start_step]
    if end_step is not None:
        checkpoints = [path for path in checkpoints if _checkpoint_step_from_path(path) <= end_step]
    if len(checkpoints) < 2:
        raise ValueError("component-output-rescue-adam-state-attribution requires at least two trace checkpoints.")
    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    for source_checkpoint_path, target_checkpoint_path in intervals:
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        if target_step - source_step != 1:
            raise RuntimeError(
                "Component-output rescue Adam attribution requires one-step checkpoint intervals. "
                f"Got {source_step}->{target_step}."
            )

    scalar_pair_rows = _filter_scalar_pair_rows(
        rows=_load_scalar_pair_rows(scalar_pair_rows_path),
        margin_sides=resolved_margin_sides,
        pair_types=pair_types,
        scalar_names=resolved_scalars,
    )
    min_source_step = _checkpoint_step_from_path(checkpoints[0])
    max_target_step = _checkpoint_step_from_path(checkpoints[-1])
    scalar_pair_rows = [
        row
        for row in scalar_pair_rows
        if int(row["source_step"]) >= min_source_step and int(row["target_step"]) <= max_target_step
    ]
    if not scalar_pair_rows:
        raise RuntimeError(
            "No scalar pair rows survived checkpoint interval filters: "
            f"start={min_source_step} end={max_target_step}."
        )
    rows_by_interval: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in scalar_pair_rows:
        source_step = int(row["source_step"])
        target_step = int(row["target_step"])
        if target_step - source_step != 1:
            raise RuntimeError(f"Scalar pair row is not one-step: {source_step}->{target_step}")
        rows_by_interval[(source_step, target_step)].append(row)
    missing_interval_rows = [
        (int(_checkpoint_step_from_path(source)), int(_checkpoint_step_from_path(target)))
        for source, target in intervals
        if (int(_checkpoint_step_from_path(source)), int(_checkpoint_step_from_path(target))) not in rows_by_interval
    ]
    if missing_interval_rows:
        raise RuntimeError(f"Missing scalar pair rows for interval(s): {missing_interval_rows[:10]}")

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

    trace_batch_rows = _load_trace_batch_rows(optimizer_trace_dir / "optimizer_update_trace_batches.jsonl")
    trace_step_rows = _load_trace_step_rows(optimizer_trace_dir / "optimizer_update_trace_steps.jsonl")
    records_by_id = _records_by_sample_id(benchmark_dir=spec.benchmark_dir, split_name="train")
    optimizer_trace_status, optimizer_trace_blocker = _optimizer_trace_metadata(optimizer_trace_dir)

    context = _resume_training_state(spec=spec, resume_checkpoint=checkpoints[0])
    model: torch.nn.Module = context["model"]
    available_components = _component_labels(num_layers=len(model.blocks), num_heads=int(model.spec.n_heads))
    resolved_sources = _filter_component_labels(
        requested_components=[source_component],
        available_components=available_components,
    )
    if len(resolved_sources) != 1:
        raise RuntimeError(f"Expected exactly one source component, got {resolved_sources}.")
    resolved_source = resolved_sources[0]
    resolved_patch_groups = _build_patch_groups(
        patch_components=patch_components,
        patch_groups=patch_groups,
        available_components=available_components,
    )
    resolved_patch_components: list[str] = []
    for patch_group in resolved_patch_groups:
        for component in patch_group["patch_components"]:
            if component not in resolved_patch_components:
                resolved_patch_components.append(str(component))
    overlap = sorted({resolved_source} & set(resolved_patch_components))
    if overlap:
        raise ValueError(f"Source and patch component sets must be disjoint; overlap={overlap}")
    _validate_maskable_components(
        components=[resolved_source],
        num_layers=len(model.blocks),
        num_heads=int(model.spec.n_heads),
        device=device,
    )
    _validate_maskable_components(
        components=resolved_patch_components,
        num_layers=len(model.blocks),
        num_heads=int(model.spec.n_heads),
        device=device,
    )
    _validate_downstream_components(source_components=[resolved_source], patch_components=resolved_patch_components)

    selected_groups, group_rows, decomposition_summary = _resolve_parameter_groups(
        model=model,
        source_components=[resolved_source],
        patch_components=resolved_patch_components,
        parameter_group_ids=parameter_group_ids,
    )

    metric_rows_path = output_dir / "component_output_rescue_adam_state_attribution_rows.jsonl"
    component_rows_path = output_dir / "component_output_rescue_adam_state_attribution_components.jsonl"
    group_rows_path = output_dir / "component_output_rescue_adam_state_attribution_groups.jsonl"
    pair_rows_path = output_dir / "component_output_rescue_adam_state_attribution_pairs.jsonl"
    report_path = output_dir / "component_output_rescue_adam_state_attribution_report.json"
    markdown_path = output_dir / "component_output_rescue_adam_state_attribution_report.md"
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs_by_id.values()])
    write_jsonl(group_rows_path, group_rows)

    print(
        "[component-output-rescue-adam-state-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(pairs_by_id)} "
        f"source={resolved_source} patch_groups={resolved_patch_groups} scalars={resolved_scalars} "
        f"device={spec.device} loss_scope={loss_scope}",
        flush=True,
    )

    metric_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        if target_step not in trace_batch_rows:
            raise KeyError(f"No optimizer trace batch row found for target step {target_step}.")
        if target_step not in trace_step_rows:
            raise KeyError(f"No optimizer trace step row found for target step {target_step}.")
        batch_row = trace_batch_rows[target_step]
        step_row = trace_step_rows[target_step]
        learning_rate = float(step_row["learning_rate"])
        actual_batch_records = _records_for_trace_batch(batch_row=batch_row, records_by_id=records_by_id)
        print(
            "[component-output-rescue-adam-state-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )

        context = _resume_training_state(spec=spec, resume_checkpoint=source_checkpoint_path)
        model = context["model"]
        optimizer: torch.optim.Optimizer = context["optimizer"]
        source_checkpoint = context["checkpoint"]
        if int(source_checkpoint["step"]) != source_step:
            raise RuntimeError(f"Source checkpoint step mismatch: payload={source_checkpoint['step']} path={source_step}")
        source_parameters = _model_parameter_snapshot(model)

        loss_payload = _compute_loss_gradient_for_records_by_scope(
            model=model,
            records=actual_batch_records,
            batch_size=spec.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            loss_scope=loss_scope,
        )
        if loss_scope == "full_lm":
            loss_delta = float(loss_payload["loss"]) - float(step_row["loss"])
            if abs(loss_delta) > loss_match_tolerance:
                raise RuntimeError(
                    f"Actual-batch loss mismatch at step {target_step}: recomputed={loss_payload['loss']:.8g} "
                    f"trace={float(step_row['loss']):.8g} delta={loss_delta:.8g} tolerance={loss_match_tolerance:.8g}"
                )
        else:
            loss_delta = None
        raw_loss_gradients = loss_payload["gradients"]
        if not isinstance(raw_loss_gradients, dict):
            raise TypeError("Loss payload gradients must be a dictionary.")

        interval_rows = rows_by_interval[(source_step, target_step)]
        source_requests, source_request_by_id = _request_maps(
            scalar_pair_rows=interval_rows,
            scalar_names=resolved_scalars,
            endpoint_kind="source",
        )
        target_requests, target_request_by_id = _request_maps(
            scalar_pair_rows=interval_rows,
            scalar_names=resolved_scalars,
            endpoint_kind="target",
        )
        source_payloads = _component_output_rescue_score_payloads(
            model=model,
            pairs_by_id=pairs_by_id,
            requests=source_requests,
            request_by_id=source_request_by_id,
            source_component=resolved_source,
            patch_groups=resolved_patch_groups,
            batch_size=analysis_batch_size,
            pad_token_id=vocab.pad_token_id,
            scalar_value_tolerance=scalar_value_tolerance,
            device=device,
            track_grad=True,
        )
        for payload in source_payloads.values():
            gradients = payload.get("gradients")
            if not isinstance(gradients, dict):
                raise TypeError(f"Component-output rescue source payload gradients must be a dictionary: {payload['score_name']}")
            _assert_finite_gradients(gradients, label=f"component-output rescue Adam scalar {payload['score_name']}")

        adam_components, adam_metadata = _adamw_component_updates(
            model=model,
            optimizer=optimizer,
            source_step=source_step,
            target_step=target_step,
            learning_rate=learning_rate,
            raw_loss_gradients=raw_loss_gradients,
            grad_clip_norm=float(spec.optimization.grad_clip_norm),
            trace_pre_clip_grad_norm=float(step_row["pre_clip_grad_norm"]),
            grad_norm_match_tolerance=grad_norm_match_tolerance,
        )

        target_checkpoint = load_checkpoint(target_checkpoint_path, device)
        if int(target_checkpoint["step"]) != target_step:
            raise RuntimeError(f"Target checkpoint step mismatch: payload={target_checkpoint['step']} path={target_step}")
        load_model_state(model, target_checkpoint["model_state"])
        target_parameters = _model_parameter_snapshot(model)
        actual_delta_parameters = _parameter_delta(
            source_parameters=source_parameters,
            target_parameters=target_parameters,
            label=f"component-output rescue Adam actual delta {source_step}->{target_step}",
        )
        reconstruction_error = _sub_tensors(
            adam_components["reconstructed_adamw_update"],
            actual_delta_parameters,
            label=f"component-output rescue AdamW reconstruction error {source_step}->{target_step}",
        )
        reconstruction_error_l2 = _tensor_l2_norm(reconstruction_error, label="component-output rescue AdamW reconstruction error")
        actual_delta_l2 = _tensor_l2_norm(actual_delta_parameters, label="component-output rescue actual parameter delta")
        target_payloads = _component_output_rescue_score_payloads(
            model=model,
            pairs_by_id=pairs_by_id,
            requests=target_requests,
            request_by_id=target_request_by_id,
            source_component=resolved_source,
            patch_groups=resolved_patch_groups,
            batch_size=analysis_batch_size,
            pad_token_id=vocab.pad_token_id,
            scalar_value_tolerance=scalar_value_tolerance,
            device=device,
            track_grad=False,
        )

        for score_name, source_payload in sorted(source_payloads.items()):
            if score_name not in target_payloads:
                raise KeyError(f"Missing target payload for score {score_name}.")
            gradients = source_payload["gradients"]
            actual_update_dot = _gradient_dot_summary(
                left_gradients=actual_delta_parameters,
                right_gradients=gradients,
                label=f"component-output rescue actual update {score_name} {source_step}->{target_step}",
            )
            component_dots: dict[str, float] = {}
            for component_name, component_tensors in adam_components.items():
                component_dot = _gradient_dot_summary(
                    left_gradients=component_tensors,
                    right_gradients=gradients,
                    label=f"component-output rescue {component_name} {score_name} {source_step}->{target_step}",
                )
                component_dots[component_name] = float(component_dot["dot"])
            target_payload = target_payloads[score_name]
            actual_score_delta = float(target_payload["score_value"]) - float(source_payload["score_value"])
            actual_update_predicted = float(actual_update_dot["dot"])
            reconstructed = component_dots["reconstructed_adamw_update"]
            reconstructed_residual = actual_update_predicted - reconstructed
            denominator = max(abs(actual_update_predicted), min_error_denominator)
            patch_group_id = str(source_payload["patch_group_id"])
            patch_group = next(
                group for group in resolved_patch_groups if str(group["patch_group_id"]) == patch_group_id
            )
            metric_row = {
                "source_step": source_step,
                "target_step": target_step,
                "step_gap": target_step - source_step,
                "source_checkpoint": str(source_checkpoint_path),
                "target_checkpoint": str(target_checkpoint_path),
                "optimizer_trace_dir": str(optimizer_trace_dir),
                "optimizer_trace_batch_step": target_step,
                "learning_rate": learning_rate,
                "source_component": resolved_source,
                "patch_group_id": patch_group_id,
                "patch_components": list(patch_group["patch_components"]),
                "patch_stages": list(patch_group["patch_stages"]),
                "score_name": score_name,
                "scalar_name": str(source_payload["scalar_name"]),
                "margin_side": ",".join(resolved_margin_sides),
                "pair_types": pair_types,
                "loss_scope": loss_scope,
                "loss": float(loss_payload["loss"]),
                "loss_delta_vs_optimizer_trace": loss_delta,
                "loss_num_records": int(loss_payload["num_records"]),
                "loss_num_tokens": int(loss_payload["num_tokens"]),
                "actual_batch_sample_count": len(actual_batch_records),
                "actual_batch_query_event_count": int(batch_row["query_event_count"]),
                "source_score_value": float(source_payload["score_value"]),
                "target_score_value": float(target_payload["score_value"]),
                "source_clean_score_value": float(source_payload["clean_score_value"]),
                "source_ablated_score_value": float(source_payload["ablated_score_value"]),
                "source_patched_score_value": float(source_payload["patched_score_value"]),
                "target_clean_score_value": float(target_payload["clean_score_value"]),
                "target_ablated_score_value": float(target_payload["ablated_score_value"]),
                "target_patched_score_value": float(target_payload["patched_score_value"]),
                "actual_score_delta": actual_score_delta,
                "actual_update_predicted_scalar_delta": actual_update_predicted,
                "actual_update_scalar_sign_match": _sign_match(actual_score_delta, actual_update_predicted),
                "reconstructed_adamw_scalar_delta": reconstructed,
                "reconstructed_adamw_residual": reconstructed_residual,
                "reconstructed_adamw_relative_error": abs(reconstructed_residual) / denominator,
                "reconstructed_adamw_sign_match": _sign_match(actual_update_predicted, reconstructed),
                "raw_sgd_scalar_delta": component_dots["raw_sgd"],
                "clipped_sgd_scalar_delta": component_dots["clipped_sgd"],
                "adam_current_gradient_scalar_delta": component_dots["adam_current_gradient"],
                "adam_historical_momentum_scalar_delta": component_dots["adam_historical_momentum"],
                "adam_preconditioned_scalar_delta": component_dots["adam_preconditioned"],
                "weight_decay_scalar_delta": component_dots["weight_decay"],
                "parameter_delta_l2_norm": actual_delta_l2,
                "reconstructed_adamw_l2_error": reconstruction_error_l2,
                "reconstructed_adamw_l2_relative_error": _safe_ratio(reconstruction_error_l2, actual_delta_l2),
                "zero_scalar_gradient_parameter_count": len(source_payload["zero_gradient_parameter_names"]),
                "zero_scalar_gradient_parameter_names": source_payload["zero_gradient_parameter_names"],
                **adam_metadata,
            }
            metric_rows.append(metric_row)
            for component_name, component_tensors in adam_components.items():
                for parameter_group in selected_groups:
                    component_rows.append(
                        _component_group_row(
                            metric_row=metric_row,
                            component_name=component_name,
                            parameter_group=parameter_group,
                            component_tensors=component_tensors,
                            scalar_gradients=gradients,
                            actual_delta_parameters=actual_delta_parameters,
                        )
                    )

        primary = max(
            [row for row in metric_rows if int(row["source_step"]) == source_step and int(row["target_step"]) == target_step],
            key=lambda row: abs(float(row["source_score_value"])),
        )
        print(
            "[component-output-rescue-adam-state-attribution] finished "
            f"{source_step}->{target_step} score={primary['score_name']} "
            f"actual={float(primary['actual_score_delta']):.6g} "
            f"recon={float(primary['reconstructed_adamw_scalar_delta']):.6g} "
            f"raw_sgd={float(primary['raw_sgd_scalar_delta']):.6g} "
            f"momentum={float(primary['adam_historical_momentum_scalar_delta']):.6g}",
            flush=True,
        )

    write_jsonl(metric_rows_path, metric_rows)
    write_jsonl(component_rows_path, component_rows)
    summary = _summarize(metric_rows=metric_rows, component_rows=component_rows)
    report = {
        "schema_version": COMPONENT_OUTPUT_RESCUE_ADAM_STATE_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "optimizer_trace_dir": str(optimizer_trace_dir),
        "optimizer_trace_status": optimizer_trace_status,
        "optimizer_trace_blocker": optimizer_trace_blocker,
        "output_dir": str(output_dir),
        "device": spec.device,
        "checkpoint_paths": [str(path) for path in checkpoints],
        "checkpoint_dir": str(checkpoint_dir),
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names": resolved_scalars,
        "source_component": resolved_source,
        "patch_groups": resolved_patch_groups,
        "patch_components": resolved_patch_components,
        "batch_size": analysis_batch_size,
        "loss_scope": loss_scope,
        "loss_match_tolerance": loss_match_tolerance,
        "grad_norm_match_tolerance": grad_norm_match_tolerance,
        "scalar_value_tolerance": scalar_value_tolerance,
        "min_error_denominator": min_error_denominator,
        "parameter_group_ids": decomposition_summary["selected_group_ids"],
        "decomposition_summary": decomposition_summary,
        "pair_construction": pair_construction,
        "calculation": {
            "score": "mean scalar(source ablated + clean downstream component write patch) - scalar(source ablated)",
            "actual_delta": "score(theta_{t+1}, target endpoint competitors) - score(theta_t, source endpoint competitors)",
            "first_order": "grad_theta score(theta_t) dot actual parameter update",
            "adam_current_gradient": "AdamW update contribution from (1-beta1) * clipped_gradient using the full Adam denominator",
            "adam_historical_momentum": "AdamW update contribution from beta1 * exp_avg_old using the full Adam denominator",
            "adam_preconditioned": "adam_current_gradient + adam_historical_momentum",
            "weight_decay": "decoupled AdamW weight decay, -learning_rate * weight_decay * theta_source",
            "reconstructed_adamw_update": "weight_decay + adam_preconditioned",
        },
        "metric_rows_path": str(metric_rows_path),
        "component_rows_path": str(component_rows_path),
        "group_rows_path": str(group_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(f"[component-output-rescue-adam-state-attribution] complete report={report_path} rows={metric_rows_path}", flush=True)
    return report_path, markdown_path, metric_rows_path, component_rows_path, group_rows_path, pair_rows_path
