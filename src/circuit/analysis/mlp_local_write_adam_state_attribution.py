from __future__ import annotations

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
    _component_order,
    _component_patch_stage,
    _parse_local_component,
    _validate_downstream_components,
)
from circuit.analysis.contextual_svd_alignment import (
    CONTEXTUAL_GROUP_BY_OPTIONS,
    _role_subspace,
)
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    ATTENTION_SCORE_RECORD_SIDES,
    GEOMETRY_POSITION_ROLES,
    _attention_key_positions,
    _build_causal_patch_pairs,
    _build_route_gradient_decomposition_groups,
    _checkpoint_step_from_path,
    _gradient_dot_summary,
    _gradient_dot_summary_for_group,
    _group_metadata,
    _holdout_pair_set,
    _model_parameter_snapshot,
    _pair_metadata,
    _parameter_delta,
    _parameter_gradients,
    _resolve_checkpoint_paths,
    _route_objective_pairs,
    _safe_ratio,
    _sign_match,
)
from circuit.analysis.mlp_local_write_map_report import _mlp_input_stage, _mlp_local_function, _validate_mlp_components
from circuit.analysis.residual_delta_vector_report import _group_token_id
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.train import _resume_training_state
from circuit.vocab import Vocabulary


MLP_LOCAL_WRITE_ADAM_STATE_ATTRIBUTION_SCHEMA_VERSION = 1
ADAM_COMPONENT_NAMES = [
    "raw_sgd",
    "clipped_sgd",
    "adam_current_gradient",
    "adam_historical_momentum",
    "adam_preconditioned",
    "weight_decay",
    "reconstructed_adamw_update",
]


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise RuntimeError(f"Cannot compute mean for empty values: {label}")
    return sum(values) / float(len(values))


def _default_parameter_group_ids(*, components: list[str]) -> list[str]:
    ids = ["global:all_named_parameters"]
    for component in components:
        kind, layer, _ = _parse_local_component(component)
        if kind != "mlp":
            raise ValueError(f"Expected MLP component, got {component!r}.")
        group_id = f"module:L{layer}.mlp"
        if group_id not in ids:
            ids.append(group_id)
    return ids


def _resolve_parameter_groups(
    *,
    model: torch.nn.Module,
    components: list[str],
    parameter_group_ids: list[str] | None,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    groups, decomposition_summary = _build_route_gradient_decomposition_groups(
        model=model,
        decomposition_modes=["module_blocks", "mlp_neurons"],
    )
    groups_by_id = {group.group_id: group for group in groups}
    requested_ids = _default_parameter_group_ids(components=components) if parameter_group_ids is None else list(parameter_group_ids)
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
        "selected_group_note": "Default groups isolate the selected MLP module blocks plus the global update.",
    }
    return selected_groups, group_rows, summary


def _projected_norm(value: torch.Tensor, basis: torch.Tensor, *, label: str) -> torch.Tensor:
    if value.ndim != 1:
        raise ValueError(f"{label} value must be rank-1, got shape {tuple(value.shape)}.")
    if basis.ndim != 2:
        raise ValueError(f"{label} basis must be rank-2, got shape {tuple(basis.shape)}.")
    if value.shape[0] != basis.shape[0]:
        raise ValueError(f"{label} value/basis dim mismatch: {tuple(value.shape)} vs {tuple(basis.shape)}.")
    projection = basis.to(value.device, dtype=value.dtype).matmul(basis.to(value.device, dtype=value.dtype).T.matmul(value))
    return projection.norm()


def _build_source_subspaces(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    components: list[str],
    position_roles: list[str],
    group_by_values: list[str],
    record_side: str,
    pca_rank: int,
    batch_size: int,
    pad_token_id: int,
    vocab: Vocabulary,
    device: torch.device,
) -> dict[tuple[str, str, str], torch.Tensor]:
    vectors_by_key: dict[tuple[str, str, str], dict[int, list[torch.Tensor]]] = {
        (_component_patch_stage(component), position_role, group_by): {}
        for component in components
        for position_role in position_roles
        for group_by in group_by_values
    }
    vectors_by_key = {key: {} for key in vectors_by_key}
    side_key = f"{record_side}_record"
    for start_index in range(0, len(pairs), batch_size):
        pair_batch = pairs[start_index : start_index + batch_size]
        records = [pair[side_key] for pair in pair_batch]
        batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
        with torch.no_grad():
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
            )
        if outputs.residual_streams is None:
            raise RuntimeError("MLP local Adam source subspace construction requires residual streams.")
        _, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        _validate_single_query_batch_for_mlp_adam(batch=batch, metadata=metadata, label="mlp-local-adam subspace")
        for flat_index in range(int(metadata["rows"].numel())):
            prediction_position = int(metadata["prediction_positions"][flat_index].item())
            for position_role in position_roles:
                context_batch_row, positions = _attention_key_positions(
                    batch=batch,
                    metadata=metadata,
                    flat_index=flat_index,
                    position_role=position_role,
                    max_position=prediction_position,
                )
                for component in components:
                    output_stage = _component_patch_stage(component)
                    stage_vectors = outputs.residual_streams[output_stage].detach().float().cpu()
                    for position in positions:
                        clean_vector = stage_vectors[context_batch_row, int(position), :].clone()
                        for group_by in group_by_values:
                            group_token_id = _group_token_id(
                                group_by=group_by,
                                batch=batch,
                                metadata=metadata,
                                answer_targets=answer_targets,
                                flat_index=flat_index,
                                context_batch_row=context_batch_row,
                                context_position=int(position),
                            )
                            bucket = vectors_by_key[(output_stage, position_role, group_by)].setdefault(group_token_id, [])
                            bucket.append(clean_vector)
    subspaces: dict[tuple[str, str, str], torch.Tensor] = {}
    for key, vectors_by_token in sorted(vectors_by_key.items()):
        output_stage, position_role, group_by = key
        subspace, _ = _role_subspace(
            role_label=f"{output_stage}:{position_role}:{group_by}",
            context_role=position_role,
            group_by=group_by,
            vectors_by_token=vectors_by_token,
            vocab=vocab,
            pca_rank=pca_rank,
        )
        subspaces[key] = subspace["identity_basis"].detach().float().cpu()
    return subspaces


def _validate_single_query_batch_for_mlp_adam(*, batch: dict[str, Any], metadata: dict[str, torch.Tensor], label: str) -> None:
    # Local import would be cleaner semantically, but this wrapper keeps the Adam tool's error label specific.
    from circuit.analysis.geometric_mechanisms import _validate_single_query_batch

    _validate_single_query_batch(batch=batch, metadata=metadata, label=label)


def _mlp_write_score_payload(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    source_component: str,
    components: list[str],
    position_roles: list[str],
    group_by_values: list[str],
    subspaces: dict[tuple[str, str, str], torch.Tensor],
    record_side: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    track_grad: bool,
) -> dict[str, Any]:
    side_key = f"{record_side}_record"
    num_layers = len(model.blocks)
    num_heads = int(model.spec.n_heads)
    source_mask_kwargs = _component_mask_kwargs_for_mlp_adam(
        component=source_component,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
    )
    scalar_sums: dict[tuple[str, str, str], torch.Tensor | None] = {
        (component, position_role, group_by): None
        for component in components
        for position_role in position_roles
        for group_by in group_by_values
    }
    scalar_values: dict[tuple[str, str, str], list[float]] = {key: [] for key in scalar_sums}
    counts: dict[tuple[str, str, str], int] = {key: 0 for key in scalar_sums}

    context = torch.enable_grad() if track_grad else torch.no_grad()
    with context:
        for start_index in range(0, len(pairs), batch_size):
            pair_batch = pairs[start_index : start_index + batch_size]
            records = [pair[side_key] for pair in pair_batch]
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
                raise RuntimeError("MLP local write Adam payload requires residual streams.")
            _, answer_targets, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            _, _, ablated_metadata = extract_answer_logits(ablated_outputs.logits, batch)
            _validate_single_query_batch_for_mlp_adam(batch=batch, metadata=clean_metadata, label="mlp-local-adam clean")
            _validate_single_query_batch_for_mlp_adam(batch=batch, metadata=ablated_metadata, label="mlp-local-adam ablated")
            for flat_index in range(int(clean_metadata["rows"].numel())):
                prediction_position = int(clean_metadata["prediction_positions"][flat_index].item())
                for position_role in position_roles:
                    context_batch_row, positions = _attention_key_positions(
                        batch=batch,
                        metadata=clean_metadata,
                        flat_index=flat_index,
                        position_role=position_role,
                        max_position=prediction_position,
                    )
                    for component in components:
                        input_stage = _mlp_input_stage(component)
                        output_stage = _component_patch_stage(component)
                        clean_inputs = torch.stack(
                            [
                                clean_outputs.residual_streams[input_stage][context_batch_row, int(position), :]
                                for position in positions
                            ],
                            dim=0,
                        )
                        ablated_inputs = torch.stack(
                            [
                                ablated_outputs.residual_streams[input_stage][context_batch_row, int(position), :]
                                for position in positions
                            ],
                            dim=0,
                        )
                        clean_input_mean = clean_inputs.mean(dim=0)
                        ablated_input_mean = ablated_inputs.mean(dim=0)
                        clean_output = _mlp_local_function(
                            model=model,
                            component=component,
                            residual_vectors=clean_input_mean.unsqueeze(0),
                        ).squeeze(0)
                        ablated_output = _mlp_local_function(
                            model=model,
                            component=component,
                            residual_vectors=ablated_input_mean.unsqueeze(0),
                        ).squeeze(0)
                        actual_delta_out = clean_output - ablated_output
                        for group_by in group_by_values:
                            subspace_key = (output_stage, position_role, group_by)
                            if subspace_key not in subspaces:
                                raise KeyError(f"Missing source-fixed subspace for {subspace_key}.")
                            projected = _projected_norm(
                                actual_delta_out,
                                subspaces[subspace_key],
                                label=f"{component}/{position_role}/{group_by}",
                            )
                            scalar_key = (component, position_role, group_by)
                            scalar_sums[scalar_key] = projected if scalar_sums[scalar_key] is None else scalar_sums[scalar_key] + projected
                            scalar_values[scalar_key].append(float(projected.detach().float().cpu().item()))
                            counts[scalar_key] += 1

    payloads: dict[str, dict[str, Any]] = {}
    for scalar_index, scalar_key in enumerate(sorted(scalar_sums)):
        total_scalar = scalar_sums[scalar_key]
        count = counts[scalar_key]
        if total_scalar is None or count <= 0:
            raise RuntimeError(f"MLP local write score produced no values for {scalar_key}.")
        mean_scalar = total_scalar / float(count)
        component, position_role, group_by = scalar_key
        scalar_name = _scalar_name(component=component, position_role=position_role, group_by=group_by)
        values = scalar_values[scalar_key]
        payload: dict[str, Any] = {
            "scalar_name": scalar_name,
            "component": component,
            "position_role": position_role,
            "group_by": group_by,
            "score_value": float(mean_scalar.detach().float().cpu().item()),
            "score_value_abs_mean": _mean([abs(value) for value in values], label=f"{scalar_name} abs values"),
            "num_entries": count,
            "num_pairs": len(pairs),
        }
        if track_grad:
            if not mean_scalar.requires_grad:
                raise RuntimeError(f"MLP local write scalar does not require grad: {scalar_name}")
            model.zero_grad(set_to_none=True)
            mean_scalar.backward(retain_graph=scalar_index < len(scalar_sums) - 1)
            gradients, zero_gradient_parameter_names = _parameter_gradients(model=model, require_all=False)
            model.zero_grad(set_to_none=True)
            payload["gradients"] = gradients
            payload["zero_gradient_parameter_names"] = zero_gradient_parameter_names
        payloads[scalar_name] = payload
    return payloads


def _component_mask_kwargs_for_mlp_adam(
    *,
    component: str,
    num_layers: int,
    num_heads: int,
    device: torch.device,
) -> dict[str, Any]:
    from circuit.analysis.output_component_causal_validation import _component_mask_kwargs

    return _component_mask_kwargs(component=component, num_layers=num_layers, num_heads=num_heads, device=device)


def _scalar_name(*, component: str, position_role: str, group_by: str) -> str:
    return f"{component}:{position_role}:{group_by}:projected_output_delta"


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
            f"MLP local Adam component {component_name} {parameter_group.group_id} "
            f"{metric_row['source_step']}->{metric_row['target_step']} {metric_row['scalar_name']}"
        ),
    )
    actual_dot = _gradient_dot_summary_for_group(
        left_gradients=actual_delta_parameters,
        right_gradients=scalar_gradients,
        group=parameter_group,
        label=(
            f"MLP local actual update {parameter_group.group_id} "
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
        "mlp_component": metric_row["mlp_component"],
        "source_component": metric_row["source_component"],
        "position_role": metric_row["position_role"],
        "group_by": metric_row["group_by"],
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
        raise RuntimeError("Cannot summarize MLP local Adam attribution without metric rows.")
    if not component_rows:
        raise RuntimeError("Cannot summarize MLP local Adam attribution without component rows.")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        grouped.setdefault(str(row["scalar_name"]), []).append(row)
    scalar_rows: list[dict[str, Any]] = []
    for scalar_name, rows in sorted(grouped.items()):
        scalar_rows.append(
            {
                "scalar_name": scalar_name,
                "mlp_component": rows[0]["mlp_component"],
                "position_role": rows[0]["position_role"],
                "group_by": rows[0]["group_by"],
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
    grouped_components: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in component_rows:
        key = (str(row["scalar_name"]), str(row["component"]), str(row["parameter_group_id"]))
        grouped_components.setdefault(key, []).append(row)
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
    return {
        "scalar_rows": scalar_rows,
        "component_summary_rows": component_summary_rows,
    }


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    rows = report["summary"]["scalar_rows"]
    lines = [
        "# MLP Local Write Adam State Attribution",
        "",
        "This report decomposes the exact AdamW update for fixed-basis MLP local write scores.",
        "",
        "The scalar is the projected magnitude of the `L0H0`-caused MLP output delta into a source-fixed contextual identity subspace:",
        "",
        "`C = mean || P_identity [MLP_i(LN(z_clean)) - MLP_i(LN(z_L0H0_ablated))] ||`",
        "",
        "## Summary",
        "",
    ]
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
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- metric rows: `{report['metric_rows_path']}`")
    lines.append(f"- component rows: `{report['component_rows_path']}`")
    lines.append(f"- group rows: `{report['group_rows_path']}`")
    lines.append(f"- route pair rows: `{report['route_pair_rows_path']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_mlp_local_write_adam_state_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    optimizer_trace_dir: Path,
    output_dir: Path,
    source_component: str,
    components: list[str],
    position_roles: list[str],
    group_by_values: list[str],
    route_pair_types: list[str],
    route_pair_type: str,
    device_name: str = "cpu",
    checkpoint_paths: list[Path] | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    record_side: str = "clean",
    route_split_filter: list[str] | None = None,
    route_split: str = "__all__",
    train_split: str = "train",
    max_route_pairs_per_type: int = 64,
    min_route_pairs_per_type: int = 1,
    pca_rank: int = 4,
    loss_scope: str = "full_lm",
    loss_match_tolerance: float = 1.0e-4,
    grad_norm_match_tolerance: float = 1.0e-4,
    min_error_denominator: float = 1.0e-9,
    parameter_group_ids: list[str] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    if record_side not in ATTENTION_SCORE_RECORD_SIDES:
        raise ValueError(f"Unsupported record_side {record_side!r}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    if not components:
        raise ValueError("At least one MLP component is required.")
    _validate_mlp_components(components=components)
    if not position_roles:
        raise ValueError("At least one position role is required.")
    unsupported_roles = [role for role in position_roles if role not in GEOMETRY_POSITION_ROLES]
    if unsupported_roles:
        raise ValueError(f"Unsupported position roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if not group_by_values:
        raise ValueError("At least one group-by value is required.")
    unsupported_group_by = [group_by for group_by in group_by_values if group_by not in CONTEXTUAL_GROUP_BY_OPTIONS]
    if unsupported_group_by:
        raise ValueError(f"Unsupported group-by values {unsupported_group_by}; expected one of {CONTEXTUAL_GROUP_BY_OPTIONS}.")
    if pca_rank <= 0:
        raise ValueError(f"pca_rank must be positive, got {pca_rank}.")
    if loss_scope not in LOSS_SCOPES:
        raise ValueError(f"Unsupported loss_scope {loss_scope!r}; expected one of {LOSS_SCOPES}.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = TrainSpec.from_path(config_path)
    if device_name is not None:
        spec = replace(spec, device=device_name)
    if float(spec.model.dropout) != 0.0:
        raise RuntimeError("MLP local Adam state attribution requires dropout=0.0.")
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(spec.device)
    checkpoint_dir = optimizer_trace_dir / "checkpoints"
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if start_step is not None:
        checkpoints = [path for path in checkpoints if _checkpoint_step_from_path(path) >= start_step]
    if end_step is not None:
        checkpoints = [path for path in checkpoints if _checkpoint_step_from_path(path) <= end_step]
    if len(checkpoints) < 2:
        raise ValueError("mlp-local-write-adam-state-attribution requires at least two trace checkpoints.")
    trace_batch_rows = _load_trace_batch_rows(optimizer_trace_dir / "optimizer_update_trace_batches.jsonl")
    trace_step_rows = _load_trace_step_rows(optimizer_trace_dir / "optimizer_update_trace_steps.jsonl")
    records_by_id = _records_by_sample_id(benchmark_dir=spec.benchmark_dir, split_name=train_split)
    optimizer_trace_status, optimizer_trace_blocker = _optimizer_trace_metadata(optimizer_trace_dir)

    route_pair_types = sorted(set(route_pair_types), key=route_pair_types.index)
    route_pairs_all, route_pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=route_pair_types,
        max_pairs_per_type=max_route_pairs_per_type,
        min_pairs_per_type=min_route_pairs_per_type,
        split_filter=route_split_filter,
    )
    route_pairs = _route_objective_pairs(
        pairs=route_pairs_all,
        route_split=route_split,
        route_pair_type=route_pair_type,
    )

    metric_rows_path = output_dir / "mlp_local_write_adam_state_attribution_rows.jsonl"
    component_rows_path = output_dir / "mlp_local_write_adam_state_attribution_components.jsonl"
    group_rows_path = output_dir / "mlp_local_write_adam_state_attribution_groups.jsonl"
    route_pair_rows_path = output_dir / "mlp_local_write_adam_state_attribution_pairs.jsonl"
    report_path = output_dir / "mlp_local_write_adam_state_attribution_report.json"
    markdown_path = output_dir / "mlp_local_write_adam_state_attribution_report.md"
    write_jsonl(route_pair_rows_path, [_pair_metadata(pair) for pair in route_pairs])

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[mlp-local-write-adam-state-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} route_pairs={len(route_pairs)} "
        f"sources={source_component} components={components} group_by={group_by_values} device={spec.device} "
        f"loss_scope={loss_scope}",
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
                "MLP local Adam state attribution requires one-step checkpoint intervals. "
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
            "[mlp-local-write-adam-state-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )

        context = _resume_training_state(spec=spec, resume_checkpoint=source_checkpoint_path)
        model: torch.nn.Module = context["model"]
        optimizer: torch.optim.Optimizer = context["optimizer"]
        _validate_downstream_components(source_components=[source_component], patch_components=components)
        source_checkpoint = context["checkpoint"]
        if int(source_checkpoint["step"]) != source_step:
            raise RuntimeError(f"Source checkpoint step mismatch: payload={source_checkpoint['step']} path={source_step}")
        source_parameters = _model_parameter_snapshot(model)
        selected_groups, selected_group_rows, selected_decomposition_summary = _resolve_parameter_groups(
            model=model,
            components=components,
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

        source_subspaces = _build_source_subspaces(
            model=model,
            pairs=route_pairs,
            components=components,
            position_roles=position_roles,
            group_by_values=group_by_values,
            record_side=record_side,
            pca_rank=pca_rank,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            vocab=vocab,
            device=device,
        )
        source_payloads = _mlp_write_score_payload(
            model=model,
            pairs=route_pairs,
            source_component=source_component,
            components=components,
            position_roles=position_roles,
            group_by_values=group_by_values,
            subspaces=source_subspaces,
            record_side=record_side,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            track_grad=True,
        )
        for payload in source_payloads.values():
            gradients = payload.get("gradients")
            if not isinstance(gradients, dict):
                raise TypeError(f"MLP local source payload gradients must be a dictionary: {payload['scalar_name']}")
            _assert_finite_gradients(gradients, label=f"MLP local Adam scalar {payload['scalar_name']}")

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
            label=f"MLP local Adam actual delta {source_step}->{target_step}",
        )
        reconstruction_error = _sub_tensors(
            adam_components["reconstructed_adamw_update"],
            actual_delta_parameters,
            label=f"MLP local AdamW reconstruction error {source_step}->{target_step}",
        )
        reconstruction_error_l2 = _tensor_l2_norm(reconstruction_error, label="MLP local AdamW reconstruction error")
        actual_delta_l2 = _tensor_l2_norm(actual_delta_parameters, label="MLP local actual parameter delta")
        target_payloads = _mlp_write_score_payload(
            model=model,
            pairs=route_pairs,
            source_component=source_component,
            components=components,
            position_roles=position_roles,
            group_by_values=group_by_values,
            subspaces=source_subspaces,
            record_side=record_side,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            track_grad=False,
        )

        for scalar_name, source_payload in sorted(source_payloads.items()):
            if scalar_name not in target_payloads:
                raise KeyError(f"Missing target payload for scalar {scalar_name}.")
            gradients = source_payload["gradients"]
            actual_update_dot = _gradient_dot_summary(
                left_gradients=actual_delta_parameters,
                right_gradients=gradients,
                label=f"MLP local actual update {scalar_name} {source_step}->{target_step}",
            )
            component_dots: dict[str, float] = {}
            for component_name, component_tensors in adam_components.items():
                component_dot = _gradient_dot_summary(
                    left_gradients=component_tensors,
                    right_gradients=gradients,
                    label=f"MLP local {component_name} {scalar_name} {source_step}->{target_step}",
                )
                component_dots[component_name] = float(component_dot["dot"])
            target_payload = target_payloads[scalar_name]
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
                "source_component": source_component,
                "mlp_component": source_payload["component"],
                "component_order": _component_order(source_payload["component"]),
                "position_role": source_payload["position_role"],
                "group_by": source_payload["group_by"],
                "scalar_name": scalar_name,
                "record_side": record_side,
                "route_pair_type": route_pair_type,
                "route_split": route_split,
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
            "[mlp-local-write-adam-state-attribution] finished "
            f"{source_step}->{target_step} scalar={primary['scalar_name']} "
            f"actual={float(primary['actual_score_delta']):.6g} "
            f"recon={float(primary['reconstructed_adamw_scalar_delta']):.6g} "
            f"raw_sgd={float(primary['raw_sgd_scalar_delta']):.6g} "
            f"momentum={float(primary['adam_historical_momentum_scalar_delta']):.6g}",
            flush=True,
        )

    if group_rows is None or decomposition_summary is None:
        raise RuntimeError("MLP local Adam state attribution did not resolve parameter groups.")
    write_jsonl(metric_rows_path, metric_rows)
    write_jsonl(component_rows_path, component_rows)
    write_jsonl(group_rows_path, group_rows)
    summary = _summarize(metric_rows=metric_rows, component_rows=component_rows)
    report = {
        "schema_version": MLP_LOCAL_WRITE_ADAM_STATE_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "optimizer_trace_dir": str(optimizer_trace_dir),
        "optimizer_trace_status": optimizer_trace_status,
        "optimizer_trace_blocker": optimizer_trace_blocker,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "output_dir": str(output_dir),
        "device": spec.device,
        "train_split": train_split,
        "source_component": source_component,
        "components": components,
        "position_roles": position_roles,
        "group_by": group_by_values,
        "record_side": record_side,
        "pca_rank": pca_rank,
        "route_pair_types": route_pair_types,
        "route_pair_type": route_pair_type,
        "route_split": route_split,
        "route_split_filter": route_split_filter,
        "max_route_pairs_per_type": max_route_pairs_per_type,
        "min_route_pairs_per_type": min_route_pairs_per_type,
        "loss_scope": loss_scope,
        "loss_match_tolerance": loss_match_tolerance,
        "grad_norm_match_tolerance": grad_norm_match_tolerance,
        "min_error_denominator": min_error_denominator,
        "parameter_group_ids": decomposition_summary["selected_group_ids"],
        "calculation": {
            "score": "mean source-fixed identity-subspace projected norm of F_i(z_clean) - F_i(z_source_component_ablated)",
            "mlp_function": "F_i(z) = MLP_i(LN_2(z))",
            "raw_sgd": "-learning_rate * raw_batch_gradient",
            "clipped_sgd": "-learning_rate * clipped_batch_gradient",
            "adam_current_gradient": "AdamW update contribution from (1-beta1) * clipped_gradient using the full Adam denominator",
            "adam_historical_momentum": "AdamW update contribution from beta1 * exp_avg_old using the full Adam denominator",
            "adam_preconditioned": "adam_current_gradient + adam_historical_momentum",
            "weight_decay": "decoupled AdamW weight decay, -learning_rate * weight_decay * theta_source",
            "reconstructed_adamw_update": "weight_decay + adam_preconditioned",
        },
        "decomposition_summary": decomposition_summary,
        "route_pair_construction": route_pair_construction,
        "route_num_pairs": len(route_pairs),
        "metric_rows_path": str(metric_rows_path),
        "component_rows_path": str(component_rows_path),
        "group_rows_path": str(group_rows_path),
        "route_pair_rows_path": str(route_pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(
        f"[mlp-local-write-adam-state-attribution] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, metric_rows_path, component_rows_path, group_rows_path, route_pair_rows_path
