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
from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.geometric_mechanisms import (
    ATTENTION_SCORE_RECORD_SIDES,
    ATTENTION_DOWNSTREAM_UPDATE_SCALARS,
    GEOMETRY_POSITION_ROLES,
    _attention_downstream_actual_summary,
    _build_causal_patch_pairs,
    _build_route_gradient_decomposition_groups,
    _checkpoint_step_from_path,
    _compute_attention_downstream_actual_rows,
    _compute_attention_downstream_scalar_gradients_for_pairs,
    _gradient_dot_summary,
    _gradient_dot_summary_for_group,
    _group_metadata,
    _head_label,
    _holdout_pair_set,
    _model_parameter_snapshot,
    _pair_metadata,
    _parameter_delta,
    _resolve_checkpoint_paths,
    _route_objective_pairs,
    _safe_ratio,
    _sign_match,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import read_symbolic_kv_stream_metadata
from circuit.io import read_json, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, require_device
from circuit.train import _resume_training_state
from circuit.vocab import Vocabulary


ATTENTION_DOWNSTREAM_ADAM_STATE_ATTRIBUTION_SCHEMA_VERSION = 1
DEFAULT_WRITE_SCALARS = [
    "attended_support_ov_value_margin",
    "head_value_margin_dla",
    "head_margin_dla_fixed_readout",
]
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


def _resolve_scalars(scalars: list[str] | None) -> list[str]:
    requested = list(DEFAULT_WRITE_SCALARS) if scalars is None else list(scalars)
    if not requested:
        raise ValueError("At least one downstream scalar is required.")
    unsupported = [scalar for scalar in requested if scalar not in ATTENTION_DOWNSTREAM_UPDATE_SCALARS]
    if unsupported:
        raise ValueError(
            f"Unsupported downstream scalar(s) {unsupported}; expected one of "
            f"{ATTENTION_DOWNSTREAM_UPDATE_SCALARS}."
        )
    resolved: list[str] = []
    for scalar in requested:
        if scalar not in resolved:
            resolved.append(scalar)
    return resolved


def _default_parameter_group_ids(*, head_layer: int, head: int) -> list[str]:
    head_label = _head_label(head_layer, head)
    return [
        "global:all_named_parameters",
        f"attention_head_projection:{head_label}.q_proj",
        f"attention_head_projection:{head_label}.k_proj",
        f"attention_head_projection:{head_label}.v_proj",
        f"attention_head_projection:{head_label}.out_proj",
        f"attention_head:{head_label}.qkvo",
    ]


def _resolve_parameter_groups(
    *,
    model: torch.nn.Module,
    head_layer: int,
    head: int,
    parameter_group_ids: list[str] | None,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    groups, decomposition_summary = _build_route_gradient_decomposition_groups(
        model=model,
        decomposition_modes=["attention_heads", "module_blocks"],
    )
    groups_by_id = {group.group_id: group for group in groups}
    requested_ids = (
        _default_parameter_group_ids(head_layer=head_layer, head=head)
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
        available = sorted(groups_by_id)
        raise ValueError(f"Unknown parameter group id(s) {missing}; available groups include {available}.")
    selected_groups = [groups_by_id[group_id] for group_id in requested_ids]
    model_parameters = dict(model.named_parameters(remove_duplicate=False))
    group_rows = [_group_metadata(model_parameters=model_parameters, group=group) for group in selected_groups]
    summary = {
        **decomposition_summary,
        "selected_group_ids": requested_ids,
        "selected_num_groups": len(selected_groups),
        "selected_group_note": (
            "Default groups isolate the traced head's Q/K/V/O slices. "
            "The out_proj group excludes out_proj.bias because that bias is shared after head concatenation."
        ),
    }
    return selected_groups, group_rows, summary


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
            f"downstream Adam component {component_name} {parameter_group.group_id} "
            f"{metric_row['source_step']}->{metric_row['target_step']} {metric_row['scalar_name']}"
        ),
    )
    actual_dot = _gradient_dot_summary_for_group(
        left_gradients=actual_delta_parameters,
        right_gradients=scalar_gradients,
        group=parameter_group,
        label=(
            f"downstream actual update {parameter_group.group_id} "
            f"{metric_row['source_step']}->{metric_row['target_step']} {metric_row['scalar_name']}"
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
        "objective_pair_type": metric_row["objective_pair_type"],
        "objective_route_split": metric_row["objective_route_split"],
        "record_side": metric_row["record_side"],
        "scalar_name": metric_row["scalar_name"],
        "head_layer": int(metric_row["head_layer"]),
        "head": int(metric_row["head"]),
        "head_label": metric_row["head_label"],
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


def _metric_row(
    *,
    source_step: int,
    target_step: int,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    optimizer_trace_dir: Path,
    learning_rate: float,
    loss_payload: dict[str, Any],
    loss_delta: float | None,
    batch_row: dict[str, Any],
    actual_summary: dict[str, Any],
    source_payload: dict[str, Any],
    actual_update_dot: dict[str, Any],
    component_dots: dict[str, float],
    adam_metadata: dict[str, Any],
    reconstruction_error_l2: float,
    actual_delta_l2: float,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    record_side: str,
    scalar_name: str,
    objective_pair_type: str,
    objective_route_split: str,
    loss_scope: str,
    min_error_denominator: float,
) -> dict[str, Any]:
    actual_scalar_delta = float(actual_summary["actual_delta"])
    actual_update_predicted = float(actual_update_dot["dot"])
    reconstructed = component_dots["reconstructed_adamw_update"]
    reconstruction_residual = actual_update_predicted - reconstructed
    denominator = max(abs(actual_update_predicted), min_error_denominator)
    return {
        "source_step": source_step,
        "target_step": target_step,
        "step_gap": target_step - source_step,
        "source_checkpoint": str(source_checkpoint_path),
        "target_checkpoint": str(target_checkpoint_path),
        "optimizer_trace_dir": str(optimizer_trace_dir),
        "optimizer_trace_batch_step": target_step,
        "learning_rate": learning_rate,
        "head_layer": head_layer,
        "head": head,
        "head_label": _head_label(head_layer, head),
        "score_query_role": score_query_role,
        "support_key_role": support_key_role,
        "distractor_key_role": distractor_key_role,
        "record_side": record_side,
        "scalar_name": scalar_name,
        "objective_pair_type": objective_pair_type,
        "objective_route_split": objective_route_split,
        "loss_scope": loss_scope,
        "loss": float(loss_payload["loss"]),
        "loss_delta_vs_optimizer_trace": loss_delta,
        "loss_num_records": int(loss_payload["num_records"]),
        "loss_num_tokens": int(loss_payload["num_tokens"]),
        "actual_batch_sample_count": int(len(batch_row["sample_ids"])),
        "actual_batch_query_event_count": int(batch_row["query_event_count"]),
        "num_pairs": int(source_payload["num_pairs"]),
        "num_entries": int(actual_summary["num_entries"]),
        "num_unique_pairs": int(actual_summary["num_unique_pairs"]),
        "source_value": float(actual_summary["source_value"]),
        "target_value": float(actual_summary["target_value"]),
        "actual_scalar_delta": actual_scalar_delta,
        "source_objective_value": float(source_payload["scalar_value"]),
        "source_objective_value_abs_mean": float(source_payload["scalar_value_abs_mean"]),
        "source_objective_value_std": float(source_payload["scalar_value_std"]),
        "actual_update_predicted_scalar_delta": actual_update_predicted,
        "actual_update_scalar_sign_match": _sign_match(actual_scalar_delta, actual_update_predicted),
        "raw_sgd_scalar_delta": component_dots["raw_sgd"],
        "clipped_sgd_scalar_delta": component_dots["clipped_sgd"],
        "adam_current_gradient_scalar_delta": component_dots["adam_current_gradient"],
        "adam_historical_momentum_scalar_delta": component_dots["adam_historical_momentum"],
        "adam_preconditioned_scalar_delta": component_dots["adam_preconditioned"],
        "weight_decay_scalar_delta": component_dots["weight_decay"],
        "reconstructed_adamw_scalar_delta": reconstructed,
        "reconstructed_adamw_residual": reconstruction_residual,
        "reconstructed_adamw_relative_error": abs(reconstruction_residual) / denominator,
        "reconstructed_adamw_sign_match": _sign_match(actual_update_predicted, reconstructed),
        "parameter_delta_l2_norm": actual_delta_l2,
        "scalar_gradient_l2_norm": float(actual_update_dot["right_l2_norm"]),
        "actual_update_scalar_gradient_cosine": actual_update_dot["cosine"],
        "reconstructed_adamw_l2_error": reconstruction_error_l2,
        "reconstructed_adamw_l2_relative_error": _safe_ratio(reconstruction_error_l2, actual_delta_l2),
        "zero_scalar_gradient_parameter_count": len(source_payload["zero_gradient_parameter_names"]),
        "zero_scalar_gradient_parameter_names": source_payload["zero_gradient_parameter_names"],
        **adam_metadata,
    }


def _summarize(
    *,
    metric_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not metric_rows:
        raise RuntimeError("Cannot summarize downstream Adam attribution without metric rows.")
    if not component_rows:
        raise RuntimeError("Cannot summarize downstream Adam attribution without component rows.")
    by_scalar: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        by_scalar.setdefault(str(row["scalar_name"]), []).append(row)
    scalar_summaries: list[dict[str, Any]] = []
    for scalar_name, rows in sorted(by_scalar.items()):
        rows = sorted(rows, key=lambda row: int(row["source_step"]))
        scalar_summaries.append(
            {
                "scalar_name": scalar_name,
                "num_intervals": len(rows),
                "source_step": int(rows[0]["source_step"]),
                "target_step": int(rows[-1]["target_step"]),
                "sum_actual_scalar_delta": sum(float(row["actual_scalar_delta"]) for row in rows),
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
                    label=f"{scalar_name} reconstruction relative error",
                ),
                "actual_update_sign_match_count": sum(
                    1 for row in rows if bool(row["actual_update_scalar_sign_match"])
                ),
                "reconstructed_adamw_sign_match_count": sum(
                    1 for row in rows if bool(row["reconstructed_adamw_sign_match"])
                ),
                "sign_match_total": len(rows),
                "mean_clip_coefficient": _mean(
                    [float(row["clip_coefficient"]) for row in rows],
                    label=f"{scalar_name} clip coefficient",
                ),
            }
        )

    grouped_components: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in component_rows:
        key = (str(row["scalar_name"]), str(row["component"]), str(row["parameter_group_id"]))
        grouped_components.setdefault(key, []).append(row)
    component_summaries: list[dict[str, Any]] = []
    for (scalar_name, component, group_id), rows in sorted(grouped_components.items()):
        component_summaries.append(
            {
                "scalar_name": scalar_name,
                "component": component,
                "parameter_group_id": group_id,
                "num_intervals": len(rows),
                "sum_component_scalar_delta": sum(float(row["component_scalar_delta"]) for row in rows),
                "mean_component_scalar_gradient_cosine": _mean(
                    [
                        float(row["component_scalar_gradient_cosine"] or 0.0)
                        for row in rows
                    ],
                    label=f"{scalar_name}/{component}/{group_id} cosine",
                ),
                "mean_fraction_of_global_actual_update_prediction": _mean(
                    [
                        float(row["component_fraction_of_global_actual_update_prediction"] or 0.0)
                        for row in rows
                    ],
                    label=f"{scalar_name}/{component}/{group_id} global fraction",
                ),
            }
        )
    return {
        "num_intervals": len({(int(row["source_step"]), int(row["target_step"])) for row in metric_rows}),
        "source_step": min(int(row["source_step"]) for row in metric_rows),
        "target_step": max(int(row["target_step"]) for row in metric_rows),
        "scalar_summaries": scalar_summaries,
        "component_summaries": component_summaries,
        "top_abs_component_summaries": sorted(
            component_summaries,
            key=lambda row: abs(float(row["sum_component_scalar_delta"])),
            reverse=True,
        )[:24],
    }


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Attention Downstream Adam State Attribution",
        "",
        "This report decomposes the exact AdamW update into optimizer pieces for downstream write-side scalars.",
        "",
        "```text",
        "write_scalar_delta ~= grad_theta(write_scalar at theta_t) . Delta theta_actual",
        "Delta theta_actual = AdamW(current gradient + historical momentum + weight decay)",
        "```",
        "",
        "The default parameter groups isolate the traced head's Q/K/V/O slices, so the same scalar can be read as a",
        "global optimizer effect and as separate pressure on `W_V` and `W_O`.",
        "",
        "## Run",
        "",
        f"- head: `{report['head_label']}`",
        f"- optimizer trace: `{report['optimizer_trace_dir']}`",
        f"- objective pair type: `{report['objective_pair_type']}`",
        f"- objective route split: `{report['objective_route_split']}`",
        f"- record side: `{report['record_side']}`",
        f"- scalars: `{report['scalar_names']}`",
        f"- selected parameter groups: `{report['parameter_group_ids']}`",
        "",
        "## Scalar Summary",
        "",
        "| scalar | actual | actual-update pred | raw SGD | Adam current | Adam momentum | weight decay | reconstructed | recon rel err |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scalar_summaries"]:
        lines.append(
            "| `{scalar}` | {actual:.6g} | {pred:.6g} | {raw:.6g} | {current:.6g} | {momentum:.6g} | {decay:.6g} | {recon:.6g} | {err:.6g} |".format(
                scalar=row["scalar_name"],
                actual=float(row["sum_actual_scalar_delta"]),
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
            "## Top Component/Parameter-Group Contributions",
            "",
            "| scalar | component | parameter group | summed contribution | mean cosine |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in summary["top_abs_component_summaries"]:
        lines.append(
            "| `{scalar}` | `{component}` | `{group}` | {value:.6g} | {cosine:.6g} |".format(
                scalar=row["scalar_name"],
                component=row["component"],
                group=row["parameter_group_id"],
                value=float(row["sum_component_scalar_delta"]),
                cosine=float(row["mean_component_scalar_gradient_cosine"]),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- metric rows: `{report['metric_rows_path']}`",
            f"- component rows: `{report['component_rows_path']}`",
            f"- group rows: `{report['group_rows_path']}`",
            f"- route pair rows: `{report['route_pair_rows_path']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_attention_downstream_adam_state_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    optimizer_trace_dir: Path,
    output_dir: Path,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    route_pair_types: list[str],
    objective_pair_type: str,
    device_name: str = "cpu",
    checkpoint_paths: list[Path] | None = None,
    record_side: str = "clean",
    scalar_names: list[str] | None = None,
    objective_route_split: str = "__all__",
    route_split_filter: list[str] | None = None,
    train_split: str = "train",
    max_route_pairs_per_type: int = 64,
    min_route_pairs_per_type: int = 1,
    loss_scope: str = "full_lm",
    loss_match_tolerance: float = 1.0e-4,
    grad_norm_match_tolerance: float = 1.0e-4,
    min_error_denominator: float = 1.0e-9,
    parameter_group_ids: list[str] | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    if record_side not in ATTENTION_SCORE_RECORD_SIDES:
        raise ValueError(f"Unsupported record_side {record_side!r}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    unsupported_roles = [
        role
        for role in [score_query_role, support_key_role, distractor_key_role]
        if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported position roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if support_key_role == distractor_key_role:
        raise ValueError("support_key_role and distractor_key_role must be different.")
    if loss_scope not in LOSS_SCOPES:
        raise ValueError(f"Unsupported loss_scope {loss_scope!r}; expected one of {LOSS_SCOPES}.")
    if loss_match_tolerance < 0.0:
        raise ValueError("loss_match_tolerance must be non-negative.")
    if grad_norm_match_tolerance < 0.0:
        raise ValueError("grad_norm_match_tolerance must be non-negative.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_scalar_names = _resolve_scalars(scalar_names)
    spec = TrainSpec.from_path(config_path)
    if device_name is not None:
        spec = replace(spec, device=device_name)
    if float(spec.model.dropout) != 0.0:
        raise RuntimeError("Downstream Adam state attribution requires dropout=0.0.")
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
    if len(checkpoints) < 2:
        raise ValueError("attention-downstream-adam-state-attribution requires at least two trace checkpoints.")
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
        route_split=objective_route_split,
        route_pair_type=objective_pair_type,
    )
    if not route_pairs:
        raise RuntimeError("Downstream Adam state attribution constructed no objective pairs.")

    target_model = build_model(spec.model, len(vocab.tokens), device)
    if head_layer < 0 or head_layer >= len(target_model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(target_model.blocks) - 1}.")
    if head < 0 or head >= target_model.blocks[head_layer].attn.n_heads:
        raise ValueError(f"head {head} outside model range 0..{target_model.blocks[head_layer].attn.n_heads - 1}.")
    parameter_groups, group_rows, decomposition_summary = _resolve_parameter_groups(
        model=target_model,
        head_layer=head_layer,
        head=head,
        parameter_group_ids=parameter_group_ids,
    )

    metric_rows_path = output_dir / "attention_downstream_adam_state_attribution_rows.jsonl"
    component_rows_path = output_dir / "attention_downstream_adam_state_attribution_components.jsonl"
    group_rows_path = output_dir / "attention_downstream_adam_state_attribution_groups.jsonl"
    route_pair_rows_path = output_dir / "attention_downstream_adam_state_attribution_pairs.jsonl"
    report_path = output_dir / "attention_downstream_adam_state_attribution_report.json"
    markdown_path = output_dir / "attention_downstream_adam_state_attribution_report.md"
    write_jsonl(route_pair_rows_path, [_pair_metadata(pair) for pair in route_pairs])
    write_jsonl(group_rows_path, group_rows)

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[attention-downstream-adam-state-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(route_pairs)} "
        f"scalars={resolved_scalar_names} device={spec.device} head={_head_label(head_layer, head)} "
        f"loss_scope={loss_scope} parameter_groups={len(parameter_groups)}",
        flush=True,
    )

    metric_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        if target_step - source_step != 1:
            raise RuntimeError(
                "Downstream Adam state attribution requires one-step checkpoint intervals. "
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
            "[attention-downstream-adam-state-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )

        context = _resume_training_state(spec=spec, resume_checkpoint=source_checkpoint_path)
        source_model: torch.nn.Module = context["model"]
        optimizer: torch.optim.Optimizer = context["optimizer"]
        source_checkpoint = context["checkpoint"]
        if int(source_checkpoint["step"]) != source_step:
            raise RuntimeError(f"Source checkpoint step mismatch: payload={source_checkpoint['step']} path={source_step}")
        source_parameters = _model_parameter_snapshot(source_model)

        loss_payload = _compute_loss_gradient_for_records_by_scope(
            model=source_model,
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
                    f"trace={float(step_row['loss']):.8g} delta={loss_delta:.8g} "
                    f"tolerance={loss_match_tolerance:.8g}"
                )
        else:
            loss_delta = None
        raw_loss_gradients = loss_payload["gradients"]
        if not isinstance(raw_loss_gradients, dict):
            raise TypeError("Loss payload gradients must be a dictionary.")

        scalar_payloads = _compute_attention_downstream_scalar_gradients_for_pairs(
            model=source_model,
            pairs=route_pairs,
            vocab=vocab,
            head_layer=head_layer,
            head=head,
            score_query_role=score_query_role,
            support_key_role=support_key_role,
            distractor_key_role=distractor_key_role,
            record_side=record_side,
            scalar_names=resolved_scalar_names,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
        )

        adam_components, adam_metadata = _adamw_component_updates(
            model=source_model,
            optimizer=optimizer,
            source_step=source_step,
            target_step=target_step,
            learning_rate=learning_rate,
            raw_loss_gradients=raw_loss_gradients,
            grad_clip_norm=float(spec.optimization.grad_clip_norm),
            trace_pre_clip_grad_norm=float(step_row["pre_clip_grad_norm"]),
            grad_norm_match_tolerance=grad_norm_match_tolerance,
        )

        actual_rows = _compute_attention_downstream_actual_rows(
            source_model=source_model,
            target_model=target_model,
            source_checkpoint_path=source_checkpoint_path,
            target_checkpoint_path=target_checkpoint_path,
            pairs=route_pairs,
            vocab=vocab,
            head_layer=head_layer,
            head=head,
            score_query_role=score_query_role,
            support_key_role=support_key_role,
            distractor_key_role=distractor_key_role,
            record_sides=[record_side],
            scalar_names=resolved_scalar_names,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
        )

        target_checkpoint = load_checkpoint(target_checkpoint_path, device)
        if int(target_checkpoint["step"]) != target_step:
            raise RuntimeError(f"Target checkpoint step mismatch: payload={target_checkpoint['step']} path={target_step}")
        load_model_state(target_model, target_checkpoint["model_state"])
        target_parameters = _model_parameter_snapshot(target_model)
        actual_delta_parameters = _parameter_delta(
            source_parameters=source_parameters,
            target_parameters=target_parameters,
            label=f"downstream Adam state actual delta {source_step}->{target_step}",
        )
        reconstruction_error = _sub_tensors(
            adam_components["reconstructed_adamw_update"],
            actual_delta_parameters,
            label=f"downstream AdamW reconstruction error {source_step}->{target_step}",
        )
        reconstruction_error_l2 = _tensor_l2_norm(reconstruction_error, label="downstream AdamW reconstruction error")
        actual_delta_l2 = _tensor_l2_norm(actual_delta_parameters, label="downstream actual parameter delta")

        for scalar_name in resolved_scalar_names:
            source_payload = scalar_payloads[scalar_name]
            scalar_gradients = source_payload["gradients"]
            if not isinstance(scalar_gradients, dict):
                raise TypeError(f"Scalar payload gradients must be a dictionary for {scalar_name}.")
            actual_summary = _attention_downstream_actual_summary(
                actual_rows=actual_rows,
                split="__all__",
                pair_type="__all__",
                record_side=record_side,
                scalar_name=scalar_name,
            )
            actual_update_dot = _gradient_dot_summary(
                left_gradients=actual_delta_parameters,
                right_gradients=scalar_gradients,
                label=f"downstream actual update {source_step}->{target_step} {scalar_name}",
            )
            component_dots: dict[str, float] = {}
            for component_name in ADAM_COMPONENT_NAMES:
                component_dot = _gradient_dot_summary(
                    left_gradients=adam_components[component_name],
                    right_gradients=scalar_gradients,
                    label=f"downstream {component_name} {source_step}->{target_step} {scalar_name}",
                )
                component_dots[component_name] = float(component_dot["dot"])
            row = _metric_row(
                source_step=source_step,
                target_step=target_step,
                source_checkpoint_path=source_checkpoint_path,
                target_checkpoint_path=target_checkpoint_path,
                optimizer_trace_dir=optimizer_trace_dir,
                learning_rate=learning_rate,
                loss_payload=loss_payload,
                loss_delta=loss_delta,
                batch_row=batch_row,
                actual_summary=actual_summary,
                source_payload=source_payload,
                actual_update_dot=actual_update_dot,
                component_dots=component_dots,
                adam_metadata=adam_metadata,
                reconstruction_error_l2=reconstruction_error_l2,
                actual_delta_l2=actual_delta_l2,
                head_layer=head_layer,
                head=head,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
                record_side=record_side,
                scalar_name=scalar_name,
                objective_pair_type=objective_pair_type,
                objective_route_split=objective_route_split,
                loss_scope=loss_scope,
                min_error_denominator=min_error_denominator,
            )
            metric_rows.append(row)
            for component_name in ADAM_COMPONENT_NAMES:
                for parameter_group in parameter_groups:
                    component_rows.append(
                        _component_group_row(
                            metric_row=row,
                            component_name=component_name,
                            parameter_group=parameter_group,
                            component_tensors=adam_components[component_name],
                            scalar_gradients=scalar_gradients,
                            actual_delta_parameters=actual_delta_parameters,
                        )
                    )

        primary = next(
            row
            for row in metric_rows
            if int(row["source_step"]) == source_step
            and int(row["target_step"]) == target_step
            and str(row["scalar_name"]) == resolved_scalar_names[0]
        )
        print(
            "[attention-downstream-adam-state-attribution] finished "
            f"{source_step}->{target_step} scalar={primary['scalar_name']} "
            f"actual={float(primary['actual_scalar_delta']):.6g} "
            f"recon={float(primary['reconstructed_adamw_scalar_delta']):.6g} "
            f"raw_sgd={float(primary['raw_sgd_scalar_delta']):.6g} "
            f"momentum={float(primary['adam_historical_momentum_scalar_delta']):.6g}",
            flush=True,
        )

    write_jsonl(metric_rows_path, metric_rows)
    write_jsonl(component_rows_path, component_rows)
    summary = _summarize(metric_rows=metric_rows, component_rows=component_rows)
    report = {
        "schema_version": ATTENTION_DOWNSTREAM_ADAM_STATE_ATTRIBUTION_SCHEMA_VERSION,
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
        "head_layer": head_layer,
        "head": head,
        "head_label": _head_label(head_layer, head),
        "score_query_role": score_query_role,
        "support_key_role": support_key_role,
        "distractor_key_role": distractor_key_role,
        "record_side": record_side,
        "scalar_names": resolved_scalar_names,
        "route_pair_types": route_pair_types,
        "objective_pair_type": objective_pair_type,
        "objective_route_split": objective_route_split,
        "route_split_filter": route_split_filter,
        "max_route_pairs_per_type": max_route_pairs_per_type,
        "min_route_pairs_per_type": min_route_pairs_per_type,
        "loss_scope": loss_scope,
        "loss_match_tolerance": loss_match_tolerance,
        "grad_norm_match_tolerance": grad_norm_match_tolerance,
        "min_error_denominator": min_error_denominator,
        "parameter_group_ids": [row["group_id"] for row in group_rows],
        "decomposition": decomposition_summary,
        "calculation": {
            "actual_scalar_delta": "scalar(theta_target) - scalar(theta_source)",
            "actual_update_predicted_scalar_delta": "grad_theta scalar(theta_source) . (theta_target - theta_source)",
            "raw_sgd": "-learning_rate * raw_batch_gradient",
            "clipped_sgd": "-learning_rate * clipped_batch_gradient",
            "adam_current_gradient": "AdamW update contribution from (1-beta1) * clipped_gradient using the full Adam denominator",
            "adam_historical_momentum": "AdamW update contribution from beta1 * exp_avg_old using the full Adam denominator",
            "adam_preconditioned": "adam_current_gradient + adam_historical_momentum",
            "weight_decay": "decoupled AdamW weight decay, -learning_rate * weight_decay * theta_source",
            "reconstructed_adamw_update": "weight_decay + adam_preconditioned",
            "attended_support_ov_value_margin": (
                "mean support attention mass times the value-token margin of the support-position OV write"
            ),
            "support_mass_ov_value_margin": (
                "total support attention mass times the value-token margin of the support-position OV write"
            ),
            "qk_ov_product": (
                "QK support-minus-distractor score separation times the value-token margin of the support-position OV write"
            ),
            "head_value_margin_dla": "head write at the query position projected through the value-token unembedding margin",
            "head_margin_dla_fixed_readout": (
                "head write at the query position dotted with the source checkpoint answer-margin readout"
            ),
        },
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
        f"[attention-downstream-adam-state-attribution] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, metric_rows_path, component_rows_path, group_rows_path, route_pair_rows_path
