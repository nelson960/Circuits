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
from circuit.analysis.bilinear_qk_rank_data_attribution import LOSS_SCOPES, _compute_loss_gradient_for_records_by_scope
from circuit.analysis.bilinear_qk_rank_update_attribution import _assert_finite_gradients
from circuit.analysis.component_output_rescue import (
    _component_order,
    _component_patch_stage,
    _parse_local_component,
    _validate_downstream_components,
)
from circuit.analysis.contextual_svd_alignment import CONTEXTUAL_GROUP_BY_OPTIONS
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    ATTENTION_SCORE_RECORD_SIDES,
    GEOMETRY_POSITION_ROLES,
    _attention_key_positions,
    _build_route_gradient_decomposition_groups,
    _checkpoint_step_from_path,
    _gradient_dot_summary,
    _gradient_dot_summary_for_group,
    _group_metadata,
    _model_parameter_snapshot,
    _parameter_delta,
    _parameter_gradients,
    _resolve_checkpoint_paths,
    _safe_ratio,
    _sign_match,
    _validate_single_query_batch,
)
from circuit.analysis.mlp_input_functional_subspace_report import _compute_functional_rows
from circuit.analysis.mlp_local_write_map_report import _mlp_input_stage, _validate_mlp_components
from circuit.analysis.output_route_closure import (
    OUTPUT_ROUTE_MARGIN_SIDES,
    OUTPUT_ROUTE_SCALARS,
    _filter_component_labels,
    _load_scalar_pair_rows,
    _mean,
    _resolve_unique_values,
    _selected_pairs_by_id,
)
from circuit.analysis.residual_delta_vector_report import _endpoint_pair_ids, _filter_scalar_pair_rows_for_delta
from circuit.analysis.residual_state_rescue import RESIDUAL_STATE_RESCUE_ENDPOINT_ROLES, _validate_maskable_components
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.train import _resume_training_state
from circuit.vocab import Vocabulary


MLP_FUNCTIONAL_WRITE_ADAM_STATE_ATTRIBUTION_SCHEMA_VERSION = 1
REFERENCE_VECTOR_KINDS = ["input_gradient", "post_mlp_gradient"]
TARGET_VECTOR_KINDS = ["input_delta", "mlp_output_delta", "post_mlp_total_delta"]
_REFERENCE_FIELD_BY_KIND = {
    "input_gradient": "_input_gradient_mean",
    "post_mlp_gradient": "_post_mlp_gradient_mean",
}


def _target_tensor(
    *,
    target_vector_kind: str,
    input_delta: torch.Tensor,
    mlp_output_delta: torch.Tensor,
    post_mlp_total_delta: torch.Tensor,
) -> torch.Tensor:
    if target_vector_kind == "input_delta":
        return input_delta
    if target_vector_kind == "mlp_output_delta":
        return mlp_output_delta
    if target_vector_kind == "post_mlp_total_delta":
        return post_mlp_total_delta
    raise ValueError(f"Unsupported target vector kind {target_vector_kind!r}; expected one of {TARGET_VECTOR_KINDS}.")


def _reference_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row["pair_id"]),
        str(row["margin_side"]),
        str(row["endpoint_kind"]),
        str(row["scalar_name"]),
        str(row["position_role"]),
        str(row["group_by"]),
    )


def _build_reference_vectors(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    scalar_pair_rows_by_endpoint: dict[tuple[int, str, str], list[dict[str, Any]]],
    endpoint_pair_ids: dict[tuple[int, str, str], set[str]],
    source_component: str,
    mlp_component: str,
    position_roles: list[str],
    group_by_values: list[str],
    scalar_names: list[str],
    batch_size: int,
    pad_token_id: int,
    vocab: Vocabulary,
    device: torch.device,
    reference_vector_kind: str,
) -> tuple[dict[tuple[str, str, str, str, str, str], torch.Tensor], list[dict[str, Any]]]:
    rows = _compute_functional_rows(
        model=model,
        checkpoint_paths_by_step=checkpoint_paths_by_step,
        pairs_by_id=pairs_by_id,
        scalar_pair_rows_by_endpoint=scalar_pair_rows_by_endpoint,
        endpoint_pair_ids=endpoint_pair_ids,
        source_component=source_component,
        mlp_component=mlp_component,
        position_roles=position_roles,
        group_by_values=group_by_values,
        scalar_names=scalar_names,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        vocab=vocab,
        device=device,
    )
    if reference_vector_kind not in _REFERENCE_FIELD_BY_KIND:
        raise ValueError(
            f"Unsupported reference vector kind {reference_vector_kind!r}; expected one of {REFERENCE_VECTOR_KINDS}."
        )
    field = _REFERENCE_FIELD_BY_KIND[reference_vector_kind]
    vectors: dict[tuple[str, str, str, str, str, str], torch.Tensor] = {}
    metadata_rows: list[dict[str, Any]] = []
    for row in rows:
        key = _reference_key(row)
        if key in vectors:
            raise RuntimeError(f"Duplicate reference vector key: {key}")
        vector = row[field].detach().float().cpu()
        if vector.ndim != 1:
            raise RuntimeError(f"Reference vector for {key} must be rank-1, got shape {tuple(vector.shape)}.")
        if not torch.isfinite(vector).all():
            raise RuntimeError(f"Non-finite reference vector for {key}.")
        if float(vector.norm().item()) == 0.0:
            raise RuntimeError(f"Zero reference vector for {key}.")
        vectors[key] = vector
        metadata_rows.append(
            {
                "pair_id": row["pair_id"],
                "source_component": row["source_component"],
                "mlp_component": row["mlp_component"],
                "reference_step": int(row["step"]),
                "endpoint_kind": row["endpoint_kind"],
                "margin_side": row["margin_side"],
                "scalar_name": row["scalar_name"],
                "position_role": row["position_role"],
                "group_by": row["group_by"],
                "reference_vector_kind": reference_vector_kind,
                "reference_vector_norm": float(vector.norm().item()),
                "input_stage": row["input_stage"],
                "output_stage": row["output_stage"],
            }
        )
    if not vectors:
        raise RuntimeError("No reference vectors were built.")
    return vectors, metadata_rows


def _default_parameter_group_ids(*, source_component: str, mlp_component: str) -> list[str]:
    group_ids = ["global:all_named_parameters"]
    source_kind, source_layer, source_head = _parse_local_component(source_component)
    if source_kind == "head":
        if source_head is None:
            raise RuntimeError(f"Parsed attention-head source without head index: {source_component}")
        group_ids.append(f"attention_head:L{source_layer}H{source_head}.qkvo")
        group_ids.append(f"module:L{source_layer}.attention")
    elif source_kind == "mlp":
        group_ids.append(f"module:L{source_layer}.mlp")
    else:
        raise ValueError(f"Unsupported source component kind {source_kind!r}.")
    mlp_kind, mlp_layer, _ = _parse_local_component(mlp_component)
    if mlp_kind != "mlp":
        raise ValueError(f"Expected MLP component, got {mlp_component!r}.")
    mlp_group = f"module:L{mlp_layer}.mlp"
    if mlp_group not in group_ids:
        group_ids.append(mlp_group)
    return group_ids


def _resolve_parameter_groups(
    *,
    model: torch.nn.Module,
    source_component: str,
    mlp_component: str,
    parameter_group_ids: list[str] | None,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    groups, decomposition_summary = _build_route_gradient_decomposition_groups(
        model=model,
        decomposition_modes=["module_blocks", "attention_heads", "attention_projections", "mlp_neurons"],
    )
    groups_by_id = {group.group_id: group for group in groups}
    requested_ids = (
        _default_parameter_group_ids(source_component=source_component, mlp_component=mlp_component)
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
        "selected_group_note": "Default groups isolate the source head/attention block, selected MLP block, and global update.",
    }
    return selected_groups, group_rows, summary


def _score_name(
    *,
    mlp_component: str,
    position_role: str,
    group_by: str,
    endpoint_kind: str,
    scalar_name: str,
    reference_vector_kind: str,
    target_vector_kind: str,
) -> str:
    return (
        f"{mlp_component}:{position_role}:{group_by}:{endpoint_kind}:"
        f"{scalar_name}:{reference_vector_kind}_dot_{target_vector_kind}"
    )


def _fixed_reference_score_payload(
    *,
    model: torch.nn.Module,
    pairs_by_id: dict[str, dict[str, Any]],
    reference_vectors: dict[tuple[str, str, str, str, str, str], torch.Tensor],
    source_component: str,
    mlp_component: str,
    position_roles: list[str],
    group_by_values: list[str],
    scalar_names: list[str],
    endpoint_roles: list[str],
    margin_sides: list[str],
    record_side: str,
    reference_vector_kind: str,
    target_vector_kind: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    track_grad: bool,
) -> dict[str, dict[str, Any]]:
    side_key = f"{record_side}_record"
    num_layers = len(model.blocks)
    num_heads = int(model.spec.n_heads)
    from circuit.analysis.output_component_causal_validation import _component_mask_kwargs

    source_mask_kwargs = _component_mask_kwargs(
        component=source_component,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
    )
    score_sums: dict[str, torch.Tensor | None] = {}
    score_values: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    score_metadata: dict[str, dict[str, Any]] = {}
    pair_ids = sorted(pairs_by_id)

    context = torch.enable_grad() if track_grad else torch.no_grad()
    with context:
        for start_index in range(0, len(pair_ids), batch_size):
            batch_pair_ids = pair_ids[start_index : start_index + batch_size]
            records = [pairs_by_id[pair_id][side_key] for pair_id in batch_pair_ids]
            batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
            clean_outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
            )
            ablated_outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
                **source_mask_kwargs,
            )
            if clean_outputs.residual_streams is None or ablated_outputs.residual_streams is None:
                raise RuntimeError("Functional write Adam score requires residual streams.")
            _, _, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            _, _, ablated_metadata = extract_answer_logits(ablated_outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=clean_metadata, label="functional-write-adam clean")
            _validate_single_query_batch(batch=batch, metadata=ablated_metadata, label="functional-write-adam ablated")
            input_stage = _mlp_input_stage(mlp_component)
            output_stage = _component_patch_stage(mlp_component)
            clean_input_stage = clean_outputs.residual_streams[input_stage]
            ablated_input_stage = ablated_outputs.residual_streams[input_stage]
            clean_output_stage = clean_outputs.residual_streams[output_stage]
            ablated_output_stage = ablated_outputs.residual_streams[output_stage]
            clean_mlp_write = clean_output_stage - clean_input_stage
            ablated_mlp_write = ablated_output_stage - ablated_input_stage
            input_delta_stage = clean_input_stage - ablated_input_stage
            mlp_output_delta_stage = clean_mlp_write - ablated_mlp_write
            post_mlp_total_delta_stage = clean_output_stage - ablated_output_stage
            for flat_index in range(int(clean_metadata["rows"].numel())):
                query_batch_row = int(clean_metadata["rows"][flat_index].item())
                pair_id = str(batch_pair_ids[query_batch_row])
                prediction_position = int(clean_metadata["prediction_positions"][flat_index].item())
                for position_role in position_roles:
                    context_batch_row, positions_tensor = _attention_key_positions(
                        batch=batch,
                        metadata=clean_metadata,
                        flat_index=flat_index,
                        position_role=position_role,
                        max_position=prediction_position,
                    )
                    positions = [int(position) for position in positions_tensor]
                    if not positions:
                        raise RuntimeError(f"No positions selected for pair={pair_id} role={position_role}.")
                    input_delta = torch.stack(
                        [input_delta_stage[context_batch_row, position, :] for position in positions],
                        dim=0,
                    ).mean(dim=0)
                    mlp_output_delta = torch.stack(
                        [mlp_output_delta_stage[context_batch_row, position, :] for position in positions],
                        dim=0,
                    ).mean(dim=0)
                    post_mlp_total_delta = torch.stack(
                        [post_mlp_total_delta_stage[context_batch_row, position, :] for position in positions],
                        dim=0,
                    ).mean(dim=0)
                    target_vector = _target_tensor(
                        target_vector_kind=target_vector_kind,
                        input_delta=input_delta,
                        mlp_output_delta=mlp_output_delta,
                        post_mlp_total_delta=post_mlp_total_delta,
                    )
                    for margin_side in margin_sides:
                        for endpoint_kind in endpoint_roles:
                            for scalar_name in scalar_names:
                                for group_by in group_by_values:
                                    reference_key = (
                                        pair_id,
                                        margin_side,
                                        endpoint_kind,
                                        scalar_name,
                                        position_role,
                                        group_by,
                                    )
                                    if reference_key not in reference_vectors:
                                        raise KeyError(f"Missing reference vector for {reference_key}.")
                                    reference_vector = reference_vectors[reference_key].to(
                                        target_vector.device,
                                        dtype=target_vector.dtype,
                                    )
                                    score_tensor = torch.dot(reference_vector, target_vector)
                                    score_name = _score_name(
                                        mlp_component=mlp_component,
                                        position_role=position_role,
                                        group_by=group_by,
                                        endpoint_kind=endpoint_kind,
                                        scalar_name=scalar_name,
                                        reference_vector_kind=reference_vector_kind,
                                        target_vector_kind=target_vector_kind,
                                    )
                                    score_sums[score_name] = (
                                        score_tensor
                                        if score_sums.get(score_name) is None
                                        else score_sums[score_name] + score_tensor
                                    )
                                    score_values.setdefault(score_name, []).append(
                                        float(score_tensor.detach().float().cpu().item())
                                    )
                                    counts[score_name] = counts.get(score_name, 0) + 1
                                    score_metadata.setdefault(
                                        score_name,
                                        {
                                            "scalar_name": score_name,
                                            "base_scalar_name": scalar_name,
                                            "mlp_component": mlp_component,
                                            "source_component": source_component,
                                            "position_role": position_role,
                                            "group_by": group_by,
                                            "endpoint_kind": endpoint_kind,
                                            "margin_side": margin_side,
                                            "reference_vector_kind": reference_vector_kind,
                                            "target_vector_kind": target_vector_kind,
                                            "input_stage": input_stage,
                                            "output_stage": output_stage,
                                        },
                                    )

    payloads: dict[str, dict[str, Any]] = {}
    for score_index, score_name in enumerate(sorted(score_sums)):
        total_score = score_sums[score_name]
        count = counts.get(score_name, 0)
        if total_score is None or count <= 0:
            raise RuntimeError(f"Functional write score produced no values for {score_name}.")
        mean_score = total_score / float(count)
        values = score_values[score_name]
        payload: dict[str, Any] = {
            **score_metadata[score_name],
            "score_value": float(mean_score.detach().float().cpu().item()),
            "score_value_abs_mean": _mean([abs(value) for value in values], label=f"{score_name} abs values"),
            "num_entries": count,
            "num_pairs": len(pairs_by_id),
        }
        if track_grad:
            if not mean_score.requires_grad:
                raise RuntimeError(f"Functional write scalar does not require grad: {score_name}")
            model.zero_grad(set_to_none=True)
            mean_score.backward(retain_graph=score_index < len(score_sums) - 1)
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
            f"Functional write Adam component {component_name} {parameter_group.group_id} "
            f"{metric_row['source_step']}->{metric_row['target_step']} {metric_row['scalar_name']}"
        ),
    )
    actual_dot = _gradient_dot_summary_for_group(
        left_gradients=actual_delta_parameters,
        right_gradients=scalar_gradients,
        group=parameter_group,
        label=(
            f"Functional write actual update {parameter_group.group_id} "
            f"{metric_row['source_step']}->{metric_row['target_step']} {metric_row['scalar_name']}"
        ),
    )
    component_delta = float(component_dot["dot"])
    actual_delta = float(actual_dot["dot"])
    return {
        "source_step": int(metric_row["source_step"]),
        "target_step": int(metric_row["target_step"]),
        "step_gap": int(metric_row["step_gap"]),
        "scalar_name": metric_row["scalar_name"],
        "base_scalar_name": metric_row["base_scalar_name"],
        "mlp_component": metric_row["mlp_component"],
        "source_component": metric_row["source_component"],
        "position_role": metric_row["position_role"],
        "group_by": metric_row["group_by"],
        "endpoint_kind": metric_row["endpoint_kind"],
        "margin_side": metric_row["margin_side"],
        "reference_vector_kind": metric_row["reference_vector_kind"],
        "target_vector_kind": metric_row["target_vector_kind"],
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


def _summarize(*, metric_rows: list[dict[str, Any]], component_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_rows:
        raise RuntimeError("Cannot summarize functional write Adam attribution without metric rows.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[str(row["scalar_name"])].append(row)
    scalar_rows: list[dict[str, Any]] = []
    for scalar_name, rows in sorted(grouped.items()):
        scalar_rows.append(
            {
                "scalar_name": scalar_name,
                "base_scalar_name": rows[0]["base_scalar_name"],
                "mlp_component": rows[0]["mlp_component"],
                "source_component": rows[0]["source_component"],
                "position_role": rows[0]["position_role"],
                "group_by": rows[0]["group_by"],
                "endpoint_kind": rows[0]["endpoint_kind"],
                "margin_side": rows[0]["margin_side"],
                "reference_vector_kind": rows[0]["reference_vector_kind"],
                "target_vector_kind": rows[0]["target_vector_kind"],
                "num_intervals": len(rows),
                "sum_actual_score_delta": sum(float(row["actual_score_delta"]) for row in rows),
                "sum_actual_update_predicted_scalar_delta": sum(
                    float(row["actual_update_predicted_scalar_delta"]) for row in rows
                ),
                "sum_reconstructed_adamw_scalar_delta": sum(
                    float(row["reconstructed_adamw_scalar_delta"]) for row in rows
                ),
                "sum_raw_sgd_scalar_delta": sum(float(row["raw_sgd_scalar_delta"]) for row in rows),
                "sum_clipped_sgd_scalar_delta": sum(float(row["clipped_sgd_scalar_delta"]) for row in rows),
                "sum_adam_current_gradient_scalar_delta": sum(
                    float(row["adam_current_gradient_scalar_delta"]) for row in rows
                ),
                "sum_adam_historical_momentum_scalar_delta": sum(
                    float(row["adam_historical_momentum_scalar_delta"]) for row in rows
                ),
                "sum_adam_preconditioned_scalar_delta": sum(
                    float(row["adam_preconditioned_scalar_delta"]) for row in rows
                ),
                "sum_weight_decay_scalar_delta": sum(float(row["weight_decay_scalar_delta"]) for row in rows),
                "mean_reconstructed_adamw_relative_error": _mean(
                    [float(row["reconstructed_adamw_relative_error"]) for row in rows],
                    label=f"{scalar_name} reconstructed errors",
                ),
                "actual_update_sign_match_count": sum(
                    1 for row in rows if bool(row["actual_update_scalar_sign_match"])
                ),
                "reconstructed_adamw_sign_match_count": sum(
                    1 for row in rows if bool(row["reconstructed_adamw_sign_match"])
                ),
            }
        )
    grouped_components: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        grouped_components[(str(row["scalar_name"]), str(row["component"]), str(row["parameter_group_id"]))].append(row)
    component_summary_rows = [
        {
            "scalar_name": scalar_name,
            "component": component_name,
            "parameter_group_id": parameter_group_id,
            "num_intervals": len(rows),
            "sum_component_scalar_delta": sum(float(row["component_scalar_delta"]) for row in rows),
            "sum_actual_update_predicted_scalar_delta_for_group": sum(
                float(row["actual_update_predicted_scalar_delta_for_group"]) for row in rows
            ),
        }
        for (scalar_name, component_name, parameter_group_id), rows in sorted(grouped_components.items())
    ]
    return {"scalar_rows": scalar_rows, "component_summary_rows": component_summary_rows}


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# MLP Functional Write Adam State Attribution",
        "",
        "This report decomposes AdamW updates for a fixed-readout write scalar.",
        "",
        "`C = mean reference_readout(pair, scalar, endpoint) dot current_write_delta(pair)`",
        "",
        "The reference readout is built at the requested reference step; the current write delta is evaluated at each optimizer-trace checkpoint.",
        "",
        "## Summary",
        "",
    ]
    rows = report["summary"]["scalar_rows"]
    if rows:
        lines.extend(
            [
                "| scalar | intervals | actual delta | actual-update pred | raw SGD | Adam current | Adam momentum | weight decay | reconstructed | recon rel err |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['scalar_name']}` | {row['num_intervals']} | "
                f"{row['sum_actual_score_delta']:.6g} | "
                f"{row['sum_actual_update_predicted_scalar_delta']:.6g} | "
                f"{row['sum_raw_sgd_scalar_delta']:.6g} | "
                f"{row['sum_adam_current_gradient_scalar_delta']:.6g} | "
                f"{row['sum_adam_historical_momentum_scalar_delta']:.6g} | "
                f"{row['sum_weight_decay_scalar_delta']:.6g} | "
                f"{row['sum_reconstructed_adamw_scalar_delta']:.6g} | "
                f"{row['mean_reconstructed_adamw_relative_error']:.6g} |"
            )
    else:
        lines.append("No summary rows produced.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- metric rows: `{report['metric_rows_path']}`",
            f"- component rows: `{report['component_rows_path']}`",
            f"- group rows: `{report['group_rows_path']}`",
            f"- reference rows: `{report['reference_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_mlp_functional_write_adam_state_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    optimizer_trace_dir: Path,
    output_dir: Path,
    source_component: str,
    mlp_component: str,
    position_roles: list[str],
    group_by_values: list[str],
    scalar_names: list[str] | None,
    margin_sides: list[str] | None,
    endpoint_roles: list[str] | None,
    pair_types: list[str],
    reference_step: int,
    reference_vector_kind: str,
    target_vector_kind: str,
    device_name: str = "cpu",
    checkpoint_paths: list[Path] | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    record_side: str = "clean",
    split_filter: list[str] | None = None,
    train_split: str = "train",
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    batch_size: int | None = None,
    loss_scope: str = "full_lm",
    loss_match_tolerance: float = 1.0e-4,
    grad_norm_match_tolerance: float = 1.0e-4,
    min_error_denominator: float = 1.0e-9,
    parameter_group_ids: list[str] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    if record_side not in ATTENTION_SCORE_RECORD_SIDES:
        raise ValueError(f"Unsupported record_side {record_side!r}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    if reference_vector_kind not in REFERENCE_VECTOR_KINDS:
        raise ValueError(f"Unsupported reference vector kind {reference_vector_kind!r}; expected one of {REFERENCE_VECTOR_KINDS}.")
    if target_vector_kind not in TARGET_VECTOR_KINDS:
        raise ValueError(f"Unsupported target vector kind {target_vector_kind!r}; expected one of {TARGET_VECTOR_KINDS}.")
    if loss_scope not in LOSS_SCOPES:
        raise ValueError(f"Unsupported loss_scope {loss_scope!r}; expected one of {LOSS_SCOPES}.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    _validate_mlp_components(components=[mlp_component])
    resolved_position_roles = _resolve_unique_values(
        values=position_roles,
        default_values=[],
        allowed_values=GEOMETRY_POSITION_ROLES,
        label="position role",
    )
    resolved_group_by = _resolve_unique_values(
        values=group_by_values,
        default_values=[],
        allowed_values=CONTEXTUAL_GROUP_BY_OPTIONS,
        label="group by",
    )
    resolved_scalars = _resolve_unique_values(
        values=scalar_names,
        default_values=["fixed_source_competitor_margin", "fixed_target_competitor_margin"],
        allowed_values=OUTPUT_ROUTE_SCALARS,
        label="scalar",
    )
    resolved_margin_sides = _resolve_unique_values(
        values=margin_sides,
        default_values=["clean"],
        allowed_values=OUTPUT_ROUTE_MARGIN_SIDES,
        label="margin side",
    )
    resolved_endpoint_roles = _resolve_unique_values(
        values=endpoint_roles,
        default_values=["source"],
        allowed_values=RESIDUAL_STATE_RESCUE_ENDPOINT_ROLES,
        label="endpoint role",
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = TrainSpec.from_path(config_path)
    if device_name is not None:
        spec = replace(spec, device=device_name)
    if float(spec.model.dropout) != 0.0:
        raise RuntimeError("Functional write Adam attribution requires dropout=0.0.")
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(spec.device)
    reference_model = build_model(spec.model, len(vocab.tokens), device)
    available_components = ["embedding"]
    for layer_index in range(len(reference_model.blocks)):
        for head_index in range(int(reference_model.spec.n_heads)):
            available_components.append(f"L{layer_index}H{head_index}")
        available_components.append(f"L{layer_index}MLP")
    resolved_source = _filter_component_labels(
        requested_components=[source_component],
        available_components=available_components,
    )[0]
    resolved_mlp = _filter_component_labels(
        requested_components=[mlp_component],
        available_components=available_components,
    )[0]
    _validate_maskable_components(
        components=[resolved_source],
        num_layers=len(reference_model.blocks),
        num_heads=reference_model.spec.n_heads,
        device=device,
    )
    _validate_downstream_components(source_components=[resolved_source], patch_components=[resolved_mlp])

    checkpoint_dir = optimizer_trace_dir / "checkpoints"
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    checkpoints_by_step = {_checkpoint_step_from_path(path): path for path in checkpoints}
    if reference_step not in checkpoints_by_step:
        raise KeyError(f"Reference step {reference_step} is not present in optimizer trace checkpoints.")
    if start_step is not None:
        checkpoints = [path for path in checkpoints if _checkpoint_step_from_path(path) >= start_step]
    if end_step is not None:
        checkpoints = [path for path in checkpoints if _checkpoint_step_from_path(path) <= end_step]
    if len(checkpoints) < 2:
        raise ValueError("mlp-functional-write-adam-state-attribution requires at least two trace checkpoints.")

    scalar_pair_rows = _filter_scalar_pair_rows_for_delta(
        rows=_load_scalar_pair_rows(scalar_pair_rows_path),
        margin_sides=resolved_margin_sides,
        pair_types=pair_types,
        scalar_names=resolved_scalars,
    )
    all_endpoint_pair_ids = _endpoint_pair_ids(
        scalar_pair_rows=scalar_pair_rows,
        endpoint_roles=resolved_endpoint_roles,
    )
    reference_endpoint_pair_ids = {
        key: value
        for key, value in all_endpoint_pair_ids.items()
        if int(key[0]) == int(reference_step)
    }
    if not reference_endpoint_pair_ids:
        raise RuntimeError(f"No scalar-pair endpoints found for reference step {reference_step}.")
    scalar_pair_rows_by_endpoint: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    allowed_reference_keys = set(reference_endpoint_pair_ids)
    for row in scalar_pair_rows:
        margin_side = str(row["margin_side"])
        for endpoint_kind in resolved_endpoint_roles:
            key = (int(row[f"{endpoint_kind}_step"]), margin_side, endpoint_kind)
            if key in allowed_reference_keys:
                scalar_pair_rows_by_endpoint[key].append(row)
    missing_reference_keys = sorted(allowed_reference_keys - set(scalar_pair_rows_by_endpoint))
    if missing_reference_keys:
        raise RuntimeError(f"Missing scalar rows for reference endpoint(s): {missing_reference_keys}")
    required_pair_ids = {pair_id for pair_ids in reference_endpoint_pair_ids.values() for pair_id in pair_ids}
    pairs_by_id, pair_construction = _selected_pairs_by_id(
        config_path=config_path,
        probe_set_path=probe_set_path,
        pair_types=pair_types,
        split_filter=split_filter,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        required_pair_ids=required_pair_ids,
    )
    analysis_batch_size = int(spec.evaluation.batch_size) if batch_size is None else int(batch_size)
    reference_vectors, reference_rows = _build_reference_vectors(
        model=reference_model,
        checkpoint_paths_by_step={reference_step: checkpoints_by_step[reference_step]},
        pairs_by_id=pairs_by_id,
        scalar_pair_rows_by_endpoint=scalar_pair_rows_by_endpoint,
        endpoint_pair_ids=reference_endpoint_pair_ids,
        source_component=resolved_source,
        mlp_component=resolved_mlp,
        position_roles=resolved_position_roles,
        group_by_values=resolved_group_by,
        scalar_names=resolved_scalars,
        batch_size=analysis_batch_size,
        pad_token_id=vocab.pad_token_id,
        vocab=vocab,
        device=device,
        reference_vector_kind=reference_vector_kind,
    )

    trace_batch_rows = _load_trace_batch_rows(optimizer_trace_dir / "optimizer_update_trace_batches.jsonl")
    trace_step_rows = _load_trace_step_rows(optimizer_trace_dir / "optimizer_update_trace_steps.jsonl")
    records_by_id = _records_by_sample_id(benchmark_dir=spec.benchmark_dir, split_name=train_split)
    optimizer_trace_status, optimizer_trace_blocker = _optimizer_trace_metadata(optimizer_trace_dir)

    metric_rows_path = output_dir / "mlp_functional_write_adam_state_attribution_rows.jsonl"
    component_rows_path = output_dir / "mlp_functional_write_adam_state_attribution_components.jsonl"
    group_rows_path = output_dir / "mlp_functional_write_adam_state_attribution_groups.jsonl"
    reference_rows_path = output_dir / "mlp_functional_write_adam_state_attribution_reference_rows.jsonl"
    pair_rows_path = output_dir / "mlp_functional_write_adam_state_attribution_pairs.jsonl"
    report_path = output_dir / "mlp_functional_write_adam_state_attribution_report.json"
    markdown_path = output_dir / "mlp_functional_write_adam_state_attribution_report.md"
    write_jsonl(reference_rows_path, reference_rows)
    write_jsonl(pair_rows_path, [pairs_by_id[pair_id] for pair_id in sorted(pairs_by_id)])

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[mlp-functional-write-adam-state-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(pairs_by_id)} "
        f"source={resolved_source} mlp={resolved_mlp} reference_step={reference_step} "
        f"reference={reference_vector_kind} target={target_vector_kind} device={spec.device} loss_scope={loss_scope}",
        flush=True,
    )

    metric_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] | None = None
    decomposition_summary: dict[str, Any] | None = None
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        if target_step - source_step != 1:
            raise RuntimeError(
                "Functional write Adam attribution requires one-step checkpoint intervals. "
                f"Got {source_step}->{target_step}."
            )
        if target_step not in trace_batch_rows:
            raise KeyError(f"No optimizer trace batch row found for target step {target_step}.")
        if target_step not in trace_step_rows:
            raise KeyError(f"No optimizer trace step row found for target step {target_step}.")
        batch_row = trace_batch_rows[target_step]
        step_row = trace_step_rows[target_step]
        learning_rate = float(step_row["learning_rate"])
        actual_batch_records = _records_for_trace_batch(batch_row=batch_row, records_by_id=records_by_id)
        print(
            "[mlp-functional-write-adam-state-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )

        context = _resume_training_state(spec=spec, resume_checkpoint=source_checkpoint_path)
        model: torch.nn.Module = context["model"]
        optimizer: torch.optim.Optimizer = context["optimizer"]
        source_checkpoint = context["checkpoint"]
        if int(source_checkpoint["step"]) != source_step:
            raise RuntimeError(f"Source checkpoint step mismatch: payload={source_checkpoint['step']} path={source_step}")
        source_parameters = _model_parameter_snapshot(model)
        selected_groups, selected_group_rows, selected_decomposition_summary = _resolve_parameter_groups(
            model=model,
            source_component=resolved_source,
            mlp_component=resolved_mlp,
            parameter_group_ids=parameter_group_ids,
        )
        if group_rows is None:
            group_rows = selected_group_rows
            decomposition_summary = selected_decomposition_summary

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

        source_payloads = _fixed_reference_score_payload(
            model=model,
            pairs_by_id=pairs_by_id,
            reference_vectors=reference_vectors,
            source_component=resolved_source,
            mlp_component=resolved_mlp,
            position_roles=resolved_position_roles,
            group_by_values=resolved_group_by,
            scalar_names=resolved_scalars,
            endpoint_roles=resolved_endpoint_roles,
            margin_sides=resolved_margin_sides,
            record_side=record_side,
            reference_vector_kind=reference_vector_kind,
            target_vector_kind=target_vector_kind,
            batch_size=analysis_batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            track_grad=True,
        )
        for payload in source_payloads.values():
            gradients = payload.get("gradients")
            if not isinstance(gradients, dict):
                raise TypeError(f"Functional write source payload gradients must be a dictionary: {payload['scalar_name']}")
            _assert_finite_gradients(gradients, label=f"Functional write Adam scalar {payload['scalar_name']}")

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
            label=f"Functional write Adam actual delta {source_step}->{target_step}",
        )
        reconstruction_error = _sub_tensors(
            adam_components["reconstructed_adamw_update"],
            actual_delta_parameters,
            label=f"Functional write AdamW reconstruction error {source_step}->{target_step}",
        )
        reconstruction_error_l2 = _tensor_l2_norm(reconstruction_error, label="Functional write AdamW reconstruction error")
        actual_delta_l2 = _tensor_l2_norm(actual_delta_parameters, label="Functional write actual parameter delta")
        target_payloads = _fixed_reference_score_payload(
            model=model,
            pairs_by_id=pairs_by_id,
            reference_vectors=reference_vectors,
            source_component=resolved_source,
            mlp_component=resolved_mlp,
            position_roles=resolved_position_roles,
            group_by_values=resolved_group_by,
            scalar_names=resolved_scalars,
            endpoint_roles=resolved_endpoint_roles,
            margin_sides=resolved_margin_sides,
            record_side=record_side,
            reference_vector_kind=reference_vector_kind,
            target_vector_kind=target_vector_kind,
            batch_size=analysis_batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            track_grad=False,
        )

        interval_rows: list[dict[str, Any]] = []
        for score_name, source_payload in sorted(source_payloads.items()):
            if score_name not in target_payloads:
                raise KeyError(f"Missing target payload for scalar {score_name}.")
            gradients = source_payload["gradients"]
            actual_update_dot = _gradient_dot_summary(
                left_gradients=actual_delta_parameters,
                right_gradients=gradients,
                label=f"Functional write actual update {score_name} {source_step}->{target_step}",
            )
            component_dots: dict[str, float] = {}
            for component_name, component_tensors in adam_components.items():
                component_dot = _gradient_dot_summary(
                    left_gradients=component_tensors,
                    right_gradients=gradients,
                    label=f"Functional write {component_name} {score_name} {source_step}->{target_step}",
                )
                component_dots[component_name] = float(component_dot["dot"])
            target_payload = target_payloads[score_name]
            actual_score_delta = float(target_payload["score_value"]) - float(source_payload["score_value"])
            actual_update_predicted = float(actual_update_dot["dot"])
            reconstructed = component_dots["reconstructed_adamw_update"]
            reconstructed_residual = actual_update_predicted - reconstructed
            denominator = max(abs(actual_update_predicted), min_error_denominator)
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
                "mlp_component": source_payload["mlp_component"],
                "component_order": _component_order(source_payload["mlp_component"]),
                "position_role": source_payload["position_role"],
                "group_by": source_payload["group_by"],
                "endpoint_kind": source_payload["endpoint_kind"],
                "margin_side": source_payload["margin_side"],
                "base_scalar_name": source_payload["base_scalar_name"],
                "scalar_name": score_name,
                "reference_step": reference_step,
                "reference_vector_kind": source_payload["reference_vector_kind"],
                "target_vector_kind": source_payload["target_vector_kind"],
                "record_side": record_side,
                "loss_scope": loss_scope,
                "loss": float(loss_payload["loss"]),
                "loss_delta_vs_optimizer_trace": loss_delta,
                "loss_num_records": int(loss_payload["num_records"]),
                "loss_num_tokens": int(loss_payload["num_tokens"]),
                "actual_batch_sample_count": len(actual_batch_records),
                "actual_batch_query_event_count": int(batch_row["query_event_count"]),
                "source_score_value": float(source_payload["score_value"]),
                "target_score_value": float(target_payload["score_value"]),
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
            interval_rows.append(metric_row)
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

        primary = max(interval_rows, key=lambda row: abs(float(row["source_score_value"])))
        print(
            "[mlp-functional-write-adam-state-attribution] finished "
            f"{source_step}->{target_step} scalar={primary['scalar_name']} "
            f"actual={float(primary['actual_score_delta']):.6g} "
            f"recon={float(primary['reconstructed_adamw_scalar_delta']):.6g} "
            f"raw_sgd={float(primary['raw_sgd_scalar_delta']):.6g} "
            f"momentum={float(primary['adam_historical_momentum_scalar_delta']):.6g}",
            flush=True,
        )

    if group_rows is None or decomposition_summary is None:
        raise RuntimeError("Functional write Adam attribution did not resolve parameter groups.")
    write_jsonl(metric_rows_path, metric_rows)
    write_jsonl(component_rows_path, component_rows)
    write_jsonl(group_rows_path, group_rows)
    summary = _summarize(metric_rows=metric_rows, component_rows=component_rows)
    report = {
        "schema_version": MLP_FUNCTIONAL_WRITE_ADAM_STATE_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "optimizer_trace_dir": str(optimizer_trace_dir),
        "optimizer_trace_status": optimizer_trace_status,
        "optimizer_trace_blocker": optimizer_trace_blocker,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "output_dir": str(output_dir),
        "device": spec.device,
        "train_split": train_split,
        "source_component": resolved_source,
        "mlp_component": resolved_mlp,
        "position_roles": resolved_position_roles,
        "group_by": resolved_group_by,
        "scalar_names": resolved_scalars,
        "margin_sides": resolved_margin_sides,
        "endpoint_roles": resolved_endpoint_roles,
        "pair_types": pair_types,
        "record_side": record_side,
        "reference_step": reference_step,
        "reference_vector_kind": reference_vector_kind,
        "target_vector_kind": target_vector_kind,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "loss_scope": loss_scope,
        "loss_match_tolerance": loss_match_tolerance,
        "grad_norm_match_tolerance": grad_norm_match_tolerance,
        "min_error_denominator": min_error_denominator,
        "parameter_group_ids": decomposition_summary["selected_group_ids"],
        "calculation": {
            "score": "mean fixed reference scalar-gradient vector dot current source-component-caused write delta",
            "reference": "reference vector comes from mlp-input-functional-subspace rows at reference_step",
            "raw_sgd": "-learning_rate * raw_batch_gradient",
            "clipped_sgd": "-learning_rate * clipped_batch_gradient",
            "adam_current_gradient": "AdamW update contribution from (1-beta1) * clipped_gradient using the full Adam denominator",
            "adam_historical_momentum": "AdamW update contribution from beta1 * exp_avg_old using the full Adam denominator",
            "adam_preconditioned": "adam_current_gradient + adam_historical_momentum",
            "weight_decay": "decoupled AdamW weight decay, -learning_rate * weight_decay * theta_source",
            "reconstructed_adamw_update": "weight_decay + adam_preconditioned",
        },
        "decomposition_summary": decomposition_summary,
        "pair_construction": pair_construction,
        "num_pairs": len(pairs_by_id),
        "num_reference_vectors": len(reference_vectors),
        "metric_rows_path": str(metric_rows_path),
        "component_rows_path": str(component_rows_path),
        "group_rows_path": str(group_rows_path),
        "reference_rows_path": str(reference_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(
        f"[mlp-functional-write-adam-state-attribution] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, metric_rows_path, component_rows_path, group_rows_path, reference_rows_path, pair_rows_path
