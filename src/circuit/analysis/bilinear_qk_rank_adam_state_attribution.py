from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import shutil
from typing import Any

import torch

from circuit.analysis.actual_batch_route_attribution import (
    _load_trace_batch_rows,
    _load_trace_step_rows,
    _records_by_sample_id,
    _records_for_trace_batch,
)
from circuit.analysis.bilinear_qk_rank_data_attribution import (
    LOSS_SCOPES,
    _compute_loss_gradient_for_records_by_scope,
)
from circuit.analysis.bilinear_qk_rank_update_attribution import (
    LAYER_NORM_MODES,
    _assert_finite_gradients,
    _rank_match_payload_for_pairs,
    _rank_qk_source_basis,
)
from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.geometric_mechanisms import (
    ATTENTION_SCORE_RECORD_SIDES,
    GEOMETRY_POSITION_ROLES,
    _build_causal_patch_pairs,
    _checkpoint_step_from_path,
    _gradient_dot_summary,
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
from circuit.runtime import load_checkpoint, load_model_state, require_device
from circuit.train import _resume_training_state
from circuit.vocab import Vocabulary


BILINEAR_QK_RANK_ADAM_STATE_ATTRIBUTION_SCHEMA_VERSION = 1


def _tensor_l2_norm(values: dict[str, torch.Tensor], *, label: str) -> float:
    total = 0.0
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite tensor in {label}: {name}")
        flat = value.float().reshape(-1)
        total += float(torch.dot(flat, flat).item())
    return math.sqrt(total)


def _scale_tensors(values: dict[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    return {name: value.float() * float(scale) for name, value in values.items()}


def _sub_tensors(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], *, label: str) -> dict[str, torch.Tensor]:
    if set(left) != set(right):
        raise ValueError(f"Tensor keys differ for {label}: left_only={sorted(set(left)-set(right))} right_only={sorted(set(right)-set(left))}")
    result: dict[str, torch.Tensor] = {}
    for name in sorted(left):
        if left[name].shape != right[name].shape:
            raise ValueError(f"Tensor shape mismatch for {label} key {name}: {tuple(left[name].shape)} vs {tuple(right[name].shape)}")
        result[name] = left[name].float() - right[name].float()
    return result


def _add_many_tensors(parts: list[dict[str, torch.Tensor]], *, label: str) -> dict[str, torch.Tensor]:
    if not parts:
        raise ValueError(f"No tensor parts for {label}.")
    keys = set(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        if set(part) != keys:
            raise ValueError(f"Tensor keys differ for {label} part {index}.")
    result: dict[str, torch.Tensor] = {}
    for name in sorted(keys):
        total = torch.zeros_like(parts[0][name].float())
        for part in parts:
            if part[name].shape != total.shape:
                raise ValueError(f"Tensor shape mismatch for {label} key {name}.")
            total = total + part[name].float()
        result[name] = total
    return result


def _optimizer_trace_metadata(optimizer_trace_dir: Path) -> tuple[str, str]:
    path = optimizer_trace_dir / "optimizer_update_trace_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing optimizer trace config: {path}")
    payload = read_json(path)
    return (
        str(payload.get("historical_replay_status", "unknown")),
        str(payload.get("historical_replay_blocker", "")),
    )


def _adam_state_step_value(value: Any, *, parameter_name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"Adam state step for {parameter_name} is not scalar: shape={tuple(value.shape)}")
        return float(value.detach().cpu().item())
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"Unsupported Adam state step type for {parameter_name}: {type(value).__name__}")


def _adamw_component_updates(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source_step: int,
    target_step: int,
    learning_rate: float,
    raw_loss_gradients: dict[str, torch.Tensor],
    grad_clip_norm: float,
    trace_pre_clip_grad_norm: float,
    grad_norm_match_tolerance: float,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    if target_step - source_step != 1:
        raise RuntimeError(f"AdamW component decomposition requires one-step intervals, got {source_step}->{target_step}.")
    param_to_names: dict[torch.nn.Parameter, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        param_to_names.setdefault(parameter, []).append(name)
    all_names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    raw_names = set(raw_loss_gradients)
    if raw_names != all_names:
        raise ValueError(f"Raw gradient keys do not match model parameters: missing={sorted(all_names-raw_names)} extra={sorted(raw_names-all_names)}")
    unique_raw_gradients = {
        names[0]: raw_loss_gradients[names[0]].float()
        for names in param_to_names.values()
    }
    raw_grad_norm = _tensor_l2_norm(unique_raw_gradients, label="unique raw loss gradients")
    if abs(raw_grad_norm - trace_pre_clip_grad_norm) > grad_norm_match_tolerance:
        raise RuntimeError(
            f"Recomputed gradient norm mismatch at {source_step}->{target_step}: "
            f"raw={raw_grad_norm:.8g} trace={trace_pre_clip_grad_norm:.8g} "
            f"delta={abs(raw_grad_norm - trace_pre_clip_grad_norm):.8g} tolerance={grad_norm_match_tolerance:.8g}"
        )
    clip_coefficient = min(1.0, grad_clip_norm / (raw_grad_norm + 1.0e-6))
    clipped_gradients = _scale_tensors(raw_loss_gradients, clip_coefficient)

    raw_sgd: dict[str, torch.Tensor] = {}
    clipped_sgd: dict[str, torch.Tensor] = {}
    adam_current: dict[str, torch.Tensor] = {}
    adam_momentum: dict[str, torch.Tensor] = {}
    adam_preconditioned: dict[str, torch.Tensor] = {}
    weight_decay: dict[str, torch.Tensor] = {}
    reconstructed: dict[str, torch.Tensor] = {}

    seen_names: set[str] = set()
    seen_parameters: set[torch.nn.Parameter] = set()
    state_steps: list[float] = []
    eps_values: set[float] = set()
    beta_values: set[tuple[float, float]] = set()
    weight_decay_values: set[float] = set()
    stored_group_lr_values: set[float] = set()
    for group in optimizer.param_groups:
        if bool(group.get("amsgrad", False)):
            raise RuntimeError("AdamW component decomposition does not support amsgrad=True.")
        if bool(group.get("maximize", False)):
            raise RuntimeError("AdamW component decomposition does not support maximize=True.")
        beta1, beta2 = group["betas"]
        beta1 = float(beta1)
        beta2 = float(beta2)
        eps = float(group["eps"])
        group_weight_decay = float(group["weight_decay"])
        group_lr = float(group["lr"])
        stored_group_lr_values.add(group_lr)
        eps_values.add(eps)
        beta_values.add((beta1, beta2))
        weight_decay_values.add(group_weight_decay)
        for parameter in group["params"]:
            if parameter in seen_parameters:
                raise RuntimeError("Parameter object appears in multiple optimizer groups.")
            seen_parameters.add(parameter)
            if parameter not in param_to_names:
                raise KeyError("Optimizer contains a parameter object not present in model.named_parameters.")
            names = param_to_names[parameter]
            overlapping_names = [name for name in names if name in seen_names]
            if overlapping_names:
                raise RuntimeError(f"Parameter names appear in multiple optimizer groups: {overlapping_names}")
            seen_names.update(names)
            state = optimizer.state.get(parameter)
            if not state:
                if source_step != 0:
                    raise RuntimeError(f"Missing Adam state for parameter names {names} at source step {source_step}.")
                exp_avg = torch.zeros_like(parameter.detach().cpu().float())
                exp_avg_sq = torch.zeros_like(parameter.detach().cpu().float())
                old_step = 0.0
            else:
                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                state_step = state.get("step")
                if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
                    raise RuntimeError(f"Adam state for {names} is missing exp_avg/exp_avg_sq tensors.")
                old_step = _adam_state_step_value(state_step, parameter_name=names[0])
            if abs(old_step - float(source_step)) > 1.0e-4:
                raise RuntimeError(f"Adam state step mismatch for {names}: state={old_step} source_step={source_step}")
            next_step = old_step + 1.0
            if abs(next_step - float(target_step)) > 1.0e-4:
                raise RuntimeError(f"Adam next step mismatch for {names}: next={next_step} target_step={target_step}")
            state_steps.append(old_step)

            source_parameter = parameter.detach().cpu().float()
            raw_gradient = raw_loss_gradients[names[0]].float()
            clipped_gradient = clipped_gradients[names[0]].float()
            for alias in names[1:]:
                if raw_loss_gradients[alias].shape != raw_gradient.shape:
                    raise ValueError(f"Tied parameter gradient shape mismatch for {names[0]} vs {alias}.")
                if not torch.equal(raw_loss_gradients[alias].float(), raw_gradient):
                    raise RuntimeError(f"Tied parameter gradients differ for {names[0]} vs {alias}.")
            exp_avg_old = exp_avg.detach().cpu().float()
            exp_avg_sq_old = exp_avg_sq.detach().cpu().float()
            exp_avg_next = beta1 * exp_avg_old + (1.0 - beta1) * clipped_gradient
            exp_avg_sq_next = beta2 * exp_avg_sq_old + (1.0 - beta2) * clipped_gradient * clipped_gradient
            bias_correction1 = 1.0 - beta1**next_step
            bias_correction2 = 1.0 - beta2**next_step
            if bias_correction1 <= 0.0 or bias_correction2 <= 0.0:
                raise RuntimeError(f"Non-positive Adam bias correction for {name}: bc1={bias_correction1} bc2={bias_correction2}")
            denominator = exp_avg_sq_next.sqrt() / math.sqrt(bias_correction2) + eps
            if not torch.isfinite(denominator).all():
                raise RuntimeError(f"Non-finite Adam denominator for {names}.")
            step_size = learning_rate / bias_correction1
            raw_sgd_value = -learning_rate * raw_gradient
            clipped_sgd_value = -learning_rate * clipped_gradient
            adam_current_value = -step_size * ((1.0 - beta1) * clipped_gradient) / denominator
            adam_momentum_value = -step_size * (beta1 * exp_avg_old) / denominator
            adam_preconditioned_value = -step_size * exp_avg_next / denominator
            weight_decay_value = -learning_rate * group_weight_decay * source_parameter
            reconstructed_value = weight_decay_value + adam_preconditioned_value
            for name in names:
                raw_sgd[name] = raw_sgd_value.clone()
                clipped_sgd[name] = clipped_sgd_value.clone()
                adam_current[name] = adam_current_value.clone()
                adam_momentum[name] = adam_momentum_value.clone()
                adam_preconditioned[name] = adam_preconditioned_value.clone()
                weight_decay[name] = weight_decay_value.clone()
                reconstructed[name] = reconstructed_value.clone()

    if seen_names != all_names:
        raise RuntimeError(f"Optimizer did not cover all model parameters: missing={sorted(all_names-seen_names)}")
    metadata = {
        "raw_gradient_l2_norm": raw_grad_norm,
        "trace_pre_clip_grad_norm": trace_pre_clip_grad_norm,
        "grad_clip_norm": grad_clip_norm,
        "clip_coefficient": clip_coefficient,
        "clipped_gradient_l2_norm": _tensor_l2_norm(clipped_gradients, label="clipped loss gradients"),
        "adam_state_step_min": min(state_steps),
        "adam_state_step_max": max(state_steps),
        "adam_eps_values": sorted(eps_values),
        "adam_beta_values": sorted(beta_values),
        "adam_weight_decay_values": sorted(weight_decay_values),
        "trace_learning_rate": learning_rate,
        "stored_optimizer_group_lr_values": sorted(stored_group_lr_values),
        "stored_optimizer_group_lr_matches_trace": all(
            abs(group_lr - learning_rate) <= 1.0e-12 for group_lr in stored_group_lr_values
        ),
    }
    return (
        {
            "raw_sgd": raw_sgd,
            "clipped_sgd": clipped_sgd,
            "adam_current_gradient": adam_current,
            "adam_historical_momentum": adam_momentum,
            "adam_preconditioned": adam_preconditioned,
            "weight_decay": weight_decay,
            "reconstructed_adamw_update": reconstructed,
        },
        metadata,
    )


def _component_row(
    *,
    base_row: dict[str, Any],
    component_name: str,
    component_tensors: dict[str, torch.Tensor],
    route_gradients: dict[str, torch.Tensor],
    actual_delta_parameters: dict[str, torch.Tensor],
) -> dict[str, Any]:
    dot = _gradient_dot_summary(
        left_gradients=component_tensors,
        right_gradients=route_gradients,
        label=f"Adam component {component_name} {base_row['source_step']}->{base_row['target_step']}",
    )
    actual = _gradient_dot_summary(
        left_gradients=actual_delta_parameters,
        right_gradients=route_gradients,
        label=f"Adam component actual {base_row['source_step']}->{base_row['target_step']}",
    )
    component_delta = float(dot["dot"])
    actual_delta = float(actual["dot"])
    return {
        **base_row,
        "component": component_name,
        "component_rank_delta": component_delta,
        "component_l2_norm": float(dot["left_l2_norm"]),
        "rank_gradient_l2_norm": float(dot["right_l2_norm"]),
        "component_rank_gradient_cosine": dot["cosine"],
        "component_fraction_of_actual_update_prediction": _safe_ratio(component_delta, actual_delta),
    }


def _summarize(
    *,
    metric_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not metric_rows:
        raise RuntimeError("Cannot summarize Adam state attribution without metric rows.")
    if not component_rows:
        raise RuntimeError("Cannot summarize Adam state attribution without component rows.")
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for row in metric_rows:
        by_rank.setdefault(int(row["rank"]), []).append(row)
    rank_summaries: list[dict[str, Any]] = []
    for rank, rows in sorted(by_rank.items()):
        rows = sorted(rows, key=lambda row: int(row["source_step"]))
        rank_summaries.append(
            {
                "rank": rank,
                "num_intervals": len(rows),
                "source_step": int(rows[0]["source_step"]),
                "target_step": int(rows[-1]["target_step"]),
                "sum_actual_rank_match_delta": sum(float(row["actual_rank_match_delta"]) for row in rows),
                "sum_actual_update_predicted_rank_match_delta": sum(
                    float(row["actual_update_predicted_rank_match_delta"]) for row in rows
                ),
                "sum_reconstructed_adamw_rank_delta": sum(float(row["reconstructed_adamw_rank_delta"]) for row in rows),
                "sum_reconstruction_residual": sum(float(row["reconstructed_adamw_residual"]) for row in rows),
                "sum_raw_sgd_rank_delta": sum(float(row["raw_sgd_rank_delta"]) for row in rows),
                "sum_clipped_sgd_rank_delta": sum(float(row["clipped_sgd_rank_delta"]) for row in rows),
                "sum_adam_current_gradient_rank_delta": sum(
                    float(row["adam_current_gradient_rank_delta"]) for row in rows
                ),
                "sum_adam_historical_momentum_rank_delta": sum(
                    float(row["adam_historical_momentum_rank_delta"]) for row in rows
                ),
                "sum_adam_preconditioned_rank_delta": sum(float(row["adam_preconditioned_rank_delta"]) for row in rows),
                "sum_weight_decay_rank_delta": sum(float(row["weight_decay_rank_delta"]) for row in rows),
                "mean_reconstructed_adamw_relative_error": sum(
                    float(row["reconstructed_adamw_relative_error"]) for row in rows
                )
                / len(rows),
                "actual_update_sign_match_count": sum(
                    1 for row in rows if bool(row["actual_update_rank_match_sign_match"])
                ),
                "reconstructed_adamw_sign_match_count": sum(
                    1 for row in rows if bool(row["reconstructed_adamw_sign_match"])
                ),
                "sign_match_total": len(rows),
                "mean_clip_coefficient": sum(float(row["clip_coefficient"]) for row in rows) / len(rows),
            }
        )
    component_by_rank_name: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in component_rows:
        component_by_rank_name.setdefault((int(row["rank"]), str(row["component"])), []).append(row)
    component_summaries: list[dict[str, Any]] = []
    for (rank, component), rows in sorted(component_by_rank_name.items()):
        component_summaries.append(
            {
                "rank": rank,
                "component": component,
                "num_intervals": len(rows),
                "sum_component_rank_delta": sum(float(row["component_rank_delta"]) for row in rows),
                "mean_component_rank_gradient_cosine": sum(
                    float(row["component_rank_gradient_cosine"] or 0.0) for row in rows
                )
                / len(rows),
            }
        )
    return {
        "intervals": [f"{row['source_step']}->{row['target_step']}" for row in sorted(metric_rows, key=lambda row: int(row["source_step"]))],
        "rank_summaries": rank_summaries,
        "component_summaries": sorted(
            component_summaries,
            key=lambda row: (int(row["rank"]), -abs(float(row["sum_component_rank_delta"]))),
        ),
    }


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# Bilinear QK Rank Adam State Attribution",
        "",
        "This report decomposes the real one-step AdamW update into components and measures which component grows the fixed-basis low-rank QK matcher.",
        "",
        "```text",
        "actual_update = weight_decay + Adam(preconditioned exp_avg)",
        "Adam(preconditioned exp_avg) = current_gradient_component + historical_momentum_component",
        "component_rank_delta = component_update . grad C_rank(theta_source)",
        "```",
        "",
        "## Replay Status",
        "",
        f"- optimizer trace: `{report['optimizer_trace_dir']}`",
        f"- trace status: `{report['optimizer_trace_status']}`",
        f"- trace blocker: {report['optimizer_trace_blocker']}",
        "",
        "## Rank Summary",
        "",
        "| rank | actual delta | actual-update pred | reconstructed AdamW | raw SGD | clipped SGD | Adam current grad | Adam historical momentum | weight decay | recon sign | recon rel err |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rank_summaries"]:
        lines.append(
            "| {rank} | {actual:.6g} | {actual_pred:.6g} | {recon:.6g} | {raw:.6g} | {clipped:.6g} | {current:.6g} | {momentum:.6g} | {decay:.6g} | {sign}/{total} | {err:.6g} |".format(
                rank=int(row["rank"]),
                actual=float(row["sum_actual_rank_match_delta"]),
                actual_pred=float(row["sum_actual_update_predicted_rank_match_delta"]),
                recon=float(row["sum_reconstructed_adamw_rank_delta"]),
                raw=float(row["sum_raw_sgd_rank_delta"]),
                clipped=float(row["sum_clipped_sgd_rank_delta"]),
                current=float(row["sum_adam_current_gradient_rank_delta"]),
                momentum=float(row["sum_adam_historical_momentum_rank_delta"]),
                decay=float(row["sum_weight_decay_rank_delta"]),
                sign=int(row["reconstructed_adamw_sign_match_count"]),
                total=int(row["sign_match_total"]),
                err=float(row["mean_reconstructed_adamw_relative_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Component Summary",
            "",
            "| rank | component | cumulative rank delta | mean cosine |",
            "|---:|---|---:|---:|",
        ]
    )
    for row in summary["component_summaries"]:
        lines.append(
            "| {rank} | `{component}` | {delta:.6g} | {cosine:.6g} |".format(
                rank=int(row["rank"]),
                component=str(row["component"]),
                delta=float(row["sum_component_rank_delta"]),
                cosine=float(row["mean_component_rank_gradient_cosine"]),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- metric rows: `{report['metric_rows_path']}`",
            f"- component rows: `{report['component_rows_path']}`",
            f"- route pair rows: `{report['route_pair_rows_path']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bilinear_qk_rank_adam_state_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    optimizer_trace_dir: Path,
    output_dir: Path,
    head_layer: int,
    head: int,
    ranks: list[int],
    context_stage: str,
    layernorm_mode: str,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    route_pair_types: list[str],
    route_pair_type: str,
    device_name: str = "cpu",
    checkpoint_paths: list[Path] | None = None,
    record_side: str = "clean",
    route_split_filter: list[str] | None = None,
    route_split: str = "__all__",
    train_split: str = "train",
    max_route_pairs_per_type: int = 64,
    min_route_pairs_per_type: int = 1,
    loss_scope: str = "full_lm",
    loss_match_tolerance: float = 1.0e-4,
    grad_norm_match_tolerance: float = 1.0e-4,
    min_error_denominator: float = 1.0e-9,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    if not ranks:
        raise ValueError("At least one --rank is required.")
    resolved_ranks = sorted(set(ranks))
    if any(rank <= 0 for rank in resolved_ranks):
        raise ValueError(f"Ranks must be positive integers, got {resolved_ranks}.")
    if record_side not in ATTENTION_SCORE_RECORD_SIDES:
        raise ValueError(f"Unsupported record_side {record_side!r}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    if layernorm_mode not in LAYER_NORM_MODES:
        raise ValueError(f"Unsupported layernorm_mode {layernorm_mode!r}; expected one of {LAYER_NORM_MODES}.")
    unsupported_roles = [
        role
        for role in [score_query_role, support_key_role, distractor_key_role]
        if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported position roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
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

    spec = TrainSpec.from_path(config_path)
    if device_name is not None:
        spec = replace(spec, device=device_name)
    if float(spec.model.dropout) != 0.0:
        raise RuntimeError("Adam state attribution requires dropout=0.0.")
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
        raise ValueError("bilinear-qk-rank-adam-state-attribution requires at least two trace checkpoints.")
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

    metric_rows_path = output_dir / "bilinear_qk_rank_adam_state_attribution_rows.jsonl"
    component_rows_path = output_dir / "bilinear_qk_rank_adam_state_attribution_components.jsonl"
    route_pair_rows_path = output_dir / "bilinear_qk_rank_adam_state_attribution_route_pairs.jsonl"
    report_path = output_dir / "bilinear_qk_rank_adam_state_attribution_report.json"
    markdown_path = output_dir / "bilinear_qk_rank_adam_state_attribution_report.md"
    write_jsonl(route_pair_rows_path, [_pair_metadata(pair) for pair in route_pairs])

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[bilinear-qk-rank-adam-state-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} route_pairs={len(route_pairs)} "
        f"ranks={resolved_ranks} device={spec.device} head=L{head_layer}H{head} loss_scope={loss_scope}",
        flush=True,
    )

    metric_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        if target_step - source_step != 1:
            raise RuntimeError(
                "Adam state attribution requires one-step checkpoint intervals. "
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
            "[bilinear-qk-rank-adam-state-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )

        context = _resume_training_state(spec=spec, resume_checkpoint=source_checkpoint_path)
        model: torch.nn.Module = context["model"]
        optimizer: torch.optim.Optimizer = context["optimizer"]
        if head_layer < 0 or head_layer >= len(model.blocks):
            raise ValueError(f"head_layer {head_layer} outside model range 0..{len(model.blocks) - 1}.")
        if head < 0 or head >= model.blocks[head_layer].attn.n_heads:
            raise ValueError(f"head {head} outside model range 0..{model.blocks[head_layer].attn.n_heads - 1}.")
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

        source_payloads: dict[int, dict[str, Any]] = {}
        route_gradients_by_rank: dict[int, dict[str, torch.Tensor]] = {}
        basis_by_rank: dict[int, tuple[torch.Tensor, torch.Tensor, dict[str, float]]] = {}
        for rank in resolved_ranks:
            left_basis, right_basis, singular_summary = _rank_qk_source_basis(
                model=model,
                head_layer=head_layer,
                head=head,
                rank=rank,
            )
            basis_by_rank[rank] = (left_basis, right_basis, singular_summary)
            source_payload = _rank_match_payload_for_pairs(
                model=model,
                pairs=route_pairs,
                head_layer=head_layer,
                head=head,
                rank=rank,
                left_basis=left_basis,
                right_basis=right_basis,
                singular_summary=singular_summary,
                context_stage=context_stage,
                layernorm_mode=layernorm_mode,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
                record_side=record_side,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
                track_grad=True,
            )
            gradients = source_payload["gradients"]
            if not isinstance(gradients, dict):
                raise TypeError("Rank QK source payload gradients must be a dictionary.")
            _assert_finite_gradients(gradients, label=f"rank-{rank} Adam-state route objective")
            source_payloads[rank] = source_payload
            route_gradients_by_rank[rank] = gradients

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
            label=f"Adam state attribution actual delta {source_step}->{target_step}",
        )
        reconstruction_error = _sub_tensors(
            adam_components["reconstructed_adamw_update"],
            actual_delta_parameters,
            label=f"AdamW reconstruction error {source_step}->{target_step}",
        )
        reconstruction_error_l2 = _tensor_l2_norm(reconstruction_error, label="AdamW reconstruction error")
        actual_delta_l2 = _tensor_l2_norm(actual_delta_parameters, label="actual parameter delta")

        for rank in resolved_ranks:
            left_basis, right_basis, singular_summary = basis_by_rank[rank]
            target_payload = _rank_match_payload_for_pairs(
                model=model,
                pairs=route_pairs,
                head_layer=head_layer,
                head=head,
                rank=rank,
                left_basis=left_basis,
                right_basis=right_basis,
                singular_summary=singular_summary,
                context_stage=context_stage,
                layernorm_mode=layernorm_mode,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
                record_side=record_side,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
                track_grad=False,
            )
            source_payload = source_payloads[rank]
            route_gradients = route_gradients_by_rank[rank]
            actual_update_dot = _gradient_dot_summary(
                left_gradients=actual_delta_parameters,
                right_gradients=route_gradients,
                label=f"actual update rank {rank} {source_step}->{target_step}",
            )
            component_dots: dict[str, float] = {}
            for component_name, component_tensors in adam_components.items():
                component_dot = _gradient_dot_summary(
                    left_gradients=component_tensors,
                    right_gradients=route_gradients,
                    label=f"{component_name} rank {rank} {source_step}->{target_step}",
                )
                component_dots[component_name] = float(component_dot["dot"])
            actual_rank_delta = float(target_payload["score_value"]) - float(source_payload["score_value"])
            actual_update_predicted = float(actual_update_dot["dot"])
            reconstructed_rank_delta = component_dots["reconstructed_adamw_update"]
            reconstructed_residual = actual_update_predicted - reconstructed_rank_delta
            reconstructed_relative_error = abs(reconstructed_residual) / max(abs(actual_update_predicted), min_error_denominator)
            metric_row = {
                "source_step": source_step,
                "target_step": target_step,
                "step_gap": target_step - source_step,
                "source_checkpoint": str(source_checkpoint_path),
                "target_checkpoint": str(target_checkpoint_path),
                "optimizer_trace_dir": str(optimizer_trace_dir),
                "optimizer_trace_batch_step": target_step,
                "learning_rate": learning_rate,
                "rank": rank,
                "context_stage": context_stage,
                "layernorm_mode": layernorm_mode,
                "head_layer": head_layer,
                "head": head,
                "head_label": f"L{head_layer}H{head}",
                "score_query_role": score_query_role,
                "support_key_role": support_key_role,
                "distractor_key_role": distractor_key_role,
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
                "source_rank_match_score": float(source_payload["score_value"]),
                "target_rank_match_score": float(target_payload["score_value"]),
                "actual_rank_match_delta": actual_rank_delta,
                "actual_update_predicted_rank_match_delta": actual_update_predicted,
                "actual_update_rank_match_sign_match": _sign_match(actual_rank_delta, actual_update_predicted),
                "reconstructed_adamw_rank_delta": reconstructed_rank_delta,
                "reconstructed_adamw_residual": reconstructed_residual,
                "reconstructed_adamw_relative_error": reconstructed_relative_error,
                "reconstructed_adamw_sign_match": _sign_match(actual_update_predicted, reconstructed_rank_delta),
                "raw_sgd_rank_delta": component_dots["raw_sgd"],
                "clipped_sgd_rank_delta": component_dots["clipped_sgd"],
                "adam_current_gradient_rank_delta": component_dots["adam_current_gradient"],
                "adam_historical_momentum_rank_delta": component_dots["adam_historical_momentum"],
                "adam_preconditioned_rank_delta": component_dots["adam_preconditioned"],
                "weight_decay_rank_delta": component_dots["weight_decay"],
                "parameter_delta_l2_norm": actual_delta_l2,
                "reconstructed_adamw_l2_error": reconstruction_error_l2,
                "reconstructed_adamw_l2_relative_error": _safe_ratio(reconstruction_error_l2, actual_delta_l2),
                **adam_metadata,
                **source_payload["singular_summary"],
            }
            metric_rows.append(metric_row)
            for component_name, component_tensors in adam_components.items():
                component_rows.append(
                    _component_row(
                        base_row=metric_row,
                        component_name=component_name,
                        component_tensors=component_tensors,
                        route_gradients=route_gradients,
                        actual_delta_parameters=actual_delta_parameters,
                    )
                )

        primary = next(
            row
            for row in metric_rows
            if int(row["source_step"]) == source_step
            and int(row["target_step"]) == target_step
            and int(row["rank"]) == resolved_ranks[-1]
        )
        print(
            "[bilinear-qk-rank-adam-state-attribution] finished "
            f"{source_step}->{target_step} rank={primary['rank']} "
            f"actual={float(primary['actual_rank_match_delta']):.6g} "
            f"recon={float(primary['reconstructed_adamw_rank_delta']):.6g} "
            f"raw_sgd={float(primary['raw_sgd_rank_delta']):.6g} "
            f"momentum={float(primary['adam_historical_momentum_rank_delta']):.6g}",
            flush=True,
        )

    write_jsonl(metric_rows_path, metric_rows)
    write_jsonl(component_rows_path, component_rows)
    summary = _summarize(metric_rows=metric_rows, component_rows=component_rows)
    report = {
        "schema_version": BILINEAR_QK_RANK_ADAM_STATE_ATTRIBUTION_SCHEMA_VERSION,
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
        "ranks": resolved_ranks,
        "context_stage": context_stage,
        "layernorm_mode": layernorm_mode,
        "score_query_role": score_query_role,
        "support_key_role": support_key_role,
        "distractor_key_role": distractor_key_role,
        "record_side": record_side,
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
        "calculation": {
            "rank_match_score": "mean score_rank(prediction, support_value) - mean score_rank(prediction, value_distractors)",
            "rank_basis": "source checkpoint fixed SVD basis for each interval and rank",
            "raw_sgd": "-learning_rate * raw_batch_gradient",
            "clipped_sgd": "-learning_rate * clipped_batch_gradient",
            "adam_current_gradient": "AdamW update contribution from (1-beta1) * clipped_gradient using the full Adam denominator",
            "adam_historical_momentum": "AdamW update contribution from beta1 * exp_avg_old using the full Adam denominator",
            "adam_preconditioned": "adam_current_gradient + adam_historical_momentum",
            "weight_decay": "decoupled AdamW weight decay, -learning_rate * weight_decay * theta_source",
            "reconstructed_adamw_update": "weight_decay + adam_preconditioned",
        },
        "route_pair_construction": route_pair_construction,
        "route_num_pairs": len(route_pairs),
        "metric_rows_path": str(metric_rows_path),
        "component_rows_path": str(component_rows_path),
        "route_pair_rows_path": str(route_pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(
        f"[bilinear-qk-rank-adam-state-attribution] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, metric_rows_path, component_rows_path, route_pair_rows_path
