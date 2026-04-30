from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import re
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
    _adam_state_step_value,
    _adamw_component_updates,
    _optimizer_trace_metadata,
    _sub_tensors,
    _tensor_l2_norm,
)
from circuit.analysis.bilinear_qk_rank_data_attribution import (
    LOSS_SCOPES,
    _compute_loss_gradient_for_records_by_scope,
)
from circuit.analysis.geometric_mechanisms import (
    _checkpoint_step_from_path,
    _model_parameter_snapshot,
    _parameter_delta,
    _resolve_checkpoint_paths,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import load_checkpoint, load_model_state, require_device
from circuit.train import _resume_training_state
from circuit.vocab import Vocabulary


ADAM_STATE_GEOMETRY_REPORT_SCHEMA_VERSION = 1

_ATTENTION_TARGET_RE = re.compile(
    r"^L(?P<layer>\d+)H(?P<head>\d+)\.(?P<matrix>q_proj|k_proj|v_proj|out_proj|W_QK|W_OV)$"
)
_MLP_TARGET_RE = re.compile(r"^L(?P<layer>\d+)MLP\.(?P<matrix>fc_in|fc_out)$")

DEFAULT_SOURCE_KINDS = [
    "weight",
    "exp_avg",
    "bias_corrected_exp_avg",
    "sqrt_exp_avg_sq",
    "adam_denominator",
    "raw_sgd",
    "clipped_sgd",
    "adam_current_gradient",
    "adam_historical_momentum",
    "adam_preconditioned",
    "weight_decay",
    "reconstructed_adamw_update",
    "actual_parameter_delta",
    "reconstruction_error",
]


def _head_slice(*, model: torch.nn.Module, layer: int, head: int) -> slice:
    if layer < 0 or layer >= len(model.blocks):
        raise ValueError(f"Layer {layer} outside model range 0..{len(model.blocks) - 1}.")
    attn = model.blocks[layer].attn
    if head < 0 or head >= attn.n_heads:
        raise ValueError(f"Head {head} outside layer {layer} head range 0..{attn.n_heads - 1}.")
    return slice(head * attn.head_dim, (head + 1) * attn.head_dim)


def _parse_target(raw: str) -> dict[str, Any]:
    match = _ATTENTION_TARGET_RE.match(raw)
    if match is not None:
        return {
            "target_id": raw,
            "target_type": "attention",
            "layer": int(match.group("layer")),
            "head": int(match.group("head")),
            "matrix": match.group("matrix"),
        }
    match = _MLP_TARGET_RE.match(raw)
    if match is not None:
        return {
            "target_id": raw,
            "target_type": "mlp",
            "layer": int(match.group("layer")),
            "head": None,
            "matrix": match.group("matrix"),
        }
    raise ValueError(
        f"Unsupported target spec {raw!r}. Expected examples: L2H1.q_proj, L2H1.W_QK, L1MLP.fc_in."
    )


def _attention_weight_name(*, layer: int, projection: str) -> str:
    return f"blocks.{layer}.attn.{projection}.weight"


def _mlp_weight_name(*, layer: int, matrix: str) -> str:
    return f"blocks.{layer}.ff.{matrix}.weight"


def _select_attention_projection(
    *,
    values: dict[str, torch.Tensor],
    model: torch.nn.Module,
    layer: int,
    head: int,
    projection: str,
) -> torch.Tensor:
    name = _attention_weight_name(layer=layer, projection=projection)
    if name not in values:
        raise KeyError(f"Missing parameter tensor {name!r}.")
    matrix = values[name].detach().cpu().float()
    head_slice = _head_slice(model=model, layer=layer, head=head)
    if projection in {"q_proj", "k_proj", "v_proj"}:
        return matrix[head_slice, :]
    if projection == "out_proj":
        return matrix[:, head_slice]
    raise ValueError(f"Unsupported attention projection {projection!r}.")


def _select_mlp_matrix(*, values: dict[str, torch.Tensor], layer: int, matrix: str) -> torch.Tensor:
    name = _mlp_weight_name(layer=layer, matrix=matrix)
    if name not in values:
        raise KeyError(f"Missing parameter tensor {name!r}.")
    return values[name].detach().cpu().float()


def _target_parameter_names(*, target: dict[str, Any]) -> list[str]:
    if target["target_type"] == "mlp":
        return [_mlp_weight_name(layer=int(target["layer"]), matrix=str(target["matrix"]))]
    layer = int(target["layer"])
    matrix = str(target["matrix"])
    if matrix in {"q_proj", "k_proj", "v_proj", "out_proj"}:
        return [_attention_weight_name(layer=layer, projection=matrix)]
    if matrix == "W_QK":
        return [
            _attention_weight_name(layer=layer, projection="q_proj"),
            _attention_weight_name(layer=layer, projection="k_proj"),
        ]
    if matrix == "W_OV":
        return [
            _attention_weight_name(layer=layer, projection="v_proj"),
            _attention_weight_name(layer=layer, projection="out_proj"),
        ]
    raise ValueError(f"Unsupported attention matrix {matrix!r}.")


def _target_matrix(
    *,
    target: dict[str, Any],
    source_kind: str,
    tensors: dict[str, torch.Tensor],
    source_parameters: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> tuple[torch.Tensor, str]:
    if target["target_type"] == "mlp":
        if source_kind == "weight":
            values = source_parameters
            mode = "exact_weight"
        else:
            values = tensors
            mode = "parameter_field"
        return _select_mlp_matrix(values=values, layer=int(target["layer"]), matrix=str(target["matrix"])), mode

    layer = int(target["layer"])
    head = int(target["head"])
    matrix = str(target["matrix"])
    if matrix in {"q_proj", "k_proj", "v_proj", "out_proj"}:
        if source_kind == "weight":
            values = source_parameters
            mode = "exact_weight"
        else:
            values = tensors
            mode = "parameter_field"
        return (
            _select_attention_projection(
                values=values,
                model=model,
                layer=layer,
                head=head,
                projection=matrix,
            ),
            mode,
        )
    if matrix == "W_QK":
        q = _select_attention_projection(
            values=source_parameters,
            model=model,
            layer=layer,
            head=head,
            projection="q_proj",
        )
        k = _select_attention_projection(
            values=source_parameters,
            model=model,
            layer=layer,
            head=head,
            projection="k_proj",
        )
        if source_kind == "weight":
            return q.T.matmul(k), "exact_weight"
        dq = _select_attention_projection(
            values=tensors,
            model=model,
            layer=layer,
            head=head,
            projection="q_proj",
        )
        dk = _select_attention_projection(
            values=tensors,
            model=model,
            layer=layer,
            head=head,
            projection="k_proj",
        )
        return dq.T.matmul(k) + q.T.matmul(dk), "first_order_functional_field"
    if matrix == "W_OV":
        v = _select_attention_projection(
            values=source_parameters,
            model=model,
            layer=layer,
            head=head,
            projection="v_proj",
        )
        out = _select_attention_projection(
            values=source_parameters,
            model=model,
            layer=layer,
            head=head,
            projection="out_proj",
        )
        if source_kind == "weight":
            return v.T.matmul(out.T), "exact_weight"
        dv = _select_attention_projection(
            values=tensors,
            model=model,
            layer=layer,
            head=head,
            projection="v_proj",
        )
        dout = _select_attention_projection(
            values=tensors,
            model=model,
            layer=layer,
            head=head,
            projection="out_proj",
        )
        return dv.T.matmul(out.T) + v.T.matmul(dout.T), "first_order_functional_field"
    raise ValueError(f"Unsupported attention matrix {matrix!r}.")


def _adam_state_tensors(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    source_step: int,
) -> dict[str, dict[str, torch.Tensor]]:
    param_to_names: dict[torch.nn.Parameter, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        param_to_names.setdefault(parameter, []).append(name)
    all_names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    exp_avg_values: dict[str, torch.Tensor] = {}
    bias_corrected_exp_avg_values: dict[str, torch.Tensor] = {}
    sqrt_exp_avg_sq_values: dict[str, torch.Tensor] = {}
    adam_denominator_values: dict[str, torch.Tensor] = {}
    seen_names: set[str] = set()
    seen_parameters: set[torch.nn.Parameter] = set()

    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        beta1 = float(beta1)
        beta2 = float(beta2)
        eps = float(group["eps"])
        for parameter in group["params"]:
            if parameter in seen_parameters:
                raise RuntimeError("Parameter object appears in multiple optimizer groups.")
            seen_parameters.add(parameter)
            if parameter not in param_to_names:
                raise KeyError("Optimizer contains a parameter object not present in model.named_parameters.")
            names = param_to_names[parameter]
            overlapping = [name for name in names if name in seen_names]
            if overlapping:
                raise RuntimeError(f"Parameter names appear in multiple optimizer groups: {overlapping}")
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
            exp_avg_cpu = exp_avg.detach().cpu().float()
            exp_avg_sq_cpu = exp_avg_sq.detach().cpu().float()
            sqrt_exp_avg_sq = exp_avg_sq_cpu.sqrt()
            if source_step == 0:
                bias_corrected_exp_avg = torch.zeros_like(exp_avg_cpu)
                denominator = torch.full_like(exp_avg_sq_cpu, eps)
            else:
                bias_correction1 = 1.0 - beta1**float(source_step)
                bias_correction2 = 1.0 - beta2**float(source_step)
                if bias_correction1 <= 0.0 or bias_correction2 <= 0.0:
                    raise RuntimeError(
                        f"Non-positive Adam source bias correction for {names}: "
                        f"bc1={bias_correction1} bc2={bias_correction2}"
                    )
                bias_corrected_exp_avg = exp_avg_cpu / bias_correction1
                denominator = (exp_avg_sq_cpu / bias_correction2).sqrt() + eps
            for name in names:
                exp_avg_values[name] = exp_avg_cpu.clone()
                bias_corrected_exp_avg_values[name] = bias_corrected_exp_avg.clone()
                sqrt_exp_avg_sq_values[name] = sqrt_exp_avg_sq.clone()
                adam_denominator_values[name] = denominator.clone()

    if seen_names != all_names:
        raise RuntimeError(f"Optimizer did not cover all model parameters: missing={sorted(all_names-seen_names)}")
    return {
        "exp_avg": exp_avg_values,
        "bias_corrected_exp_avg": bias_corrected_exp_avg_values,
        "sqrt_exp_avg_sq": sqrt_exp_avg_sq_values,
        "adam_denominator": adam_denominator_values,
    }


def _svd_rows_for_matrix(
    *,
    matrix: torch.Tensor,
    base_row: dict[str, Any],
    top_k: int,
    top_vector_ranks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if matrix.ndim != 2:
        raise ValueError(f"Expected rank-2 matrix for {base_row['target_id']} {base_row['source_kind']}, got {tuple(matrix.shape)}")
    matrix_cpu = matrix.detach().cpu().float()
    if not torch.isfinite(matrix_cpu).all():
        raise RuntimeError(f"Non-finite value in matrix for {base_row['target_id']} {base_row['source_kind']}.")
    u, singular_values, vh = torch.linalg.svd(matrix_cpu.to(dtype=torch.float64), full_matrices=False)
    available = int(singular_values.numel())
    if available <= 0:
        raise RuntimeError(f"SVD produced no singular values for {base_row['target_id']} {base_row['source_kind']}.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if top_vector_ranks < 0:
        raise ValueError("top_vector_ranks must be non-negative.")
    singular_sum = float(singular_values.sum().item())
    singular_sq_sum = float(singular_values.square().sum().item())
    if singular_sq_sum <= 0.0:
        effective_rank = 0.0
    else:
        effective_rank = (singular_sum * singular_sum) / singular_sq_sum
    spectral_mass_top3 = 0.0 if singular_sum == 0.0 else float((singular_values[: min(3, available)].sum() / singular_values.sum()).item())
    fro_norm = float(matrix_cpu.norm().item())
    max_abs = float(matrix_cpu.abs().max().item()) if matrix_cpu.numel() else 0.0
    stats = {
        **base_row,
        "matrix_rows": int(matrix_cpu.shape[0]),
        "matrix_cols": int(matrix_cpu.shape[1]),
        "singular_value_count": available,
        "singular_value_sum": singular_sum,
        "singular_value_sq_sum": singular_sq_sum,
        "effective_rank": effective_rank,
        "spectral_mass_top3": spectral_mass_top3,
        "fro_norm": fro_norm,
        "max_abs": max_abs,
        "top_singular_value": float(singular_values[0].item()),
    }
    singular_rows = [
        {
            **stats,
            "singular_value_rank": rank_index,
            "singular_value": float(value),
        }
        for rank_index, value in enumerate(singular_values[: min(top_k, available)].tolist(), start=1)
    ]
    vector_rows: list[dict[str, Any]] = []
    for rank_index in range(1, min(top_vector_ranks, available) + 1):
        vector_rows.append(
            {
                **stats,
                "singular_value_rank": rank_index,
                "singular_value": float(singular_values[rank_index - 1].item()),
                "vector_side": "left",
                "vector_dim": int(u.shape[0]),
                "vector": [float(value) for value in u[:, rank_index - 1].tolist()],
            }
        )
        vector_rows.append(
            {
                **stats,
                "singular_value_rank": rank_index,
                "singular_value": float(singular_values[rank_index - 1].item()),
                "vector_side": "right",
                "vector_dim": int(vh.shape[1]),
                "vector": [float(value) for value in vh[rank_index - 1, :].tolist()],
            }
        )
    return singular_rows, vector_rows, stats


def _filter_intervals(
    *,
    checkpoints: list[Path],
    start_step: int | None,
    end_step: int | None,
    checkpoint_stride: int,
) -> list[tuple[Path, Path]]:
    if checkpoint_stride <= 0:
        raise ValueError("checkpoint_stride must be positive.")
    raw_intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    if not raw_intervals:
        raise ValueError("At least two checkpoints are required.")
    base_step = start_step if start_step is not None else _checkpoint_step_from_path(raw_intervals[0][0])
    intervals: list[tuple[Path, Path]] = []
    for source_path, target_path in raw_intervals:
        source_step = _checkpoint_step_from_path(source_path)
        target_step = _checkpoint_step_from_path(target_path)
        if target_step - source_step != 1:
            raise RuntimeError(f"Adam geometry report requires one-step intervals, got {source_step}->{target_step}.")
        if start_step is not None and source_step < start_step:
            continue
        if end_step is not None and target_step > end_step:
            continue
        if (source_step - base_step) % checkpoint_stride != 0:
            continue
        intervals.append((source_path, target_path))
    if not intervals:
        raise ValueError(
            f"No one-step intervals selected with start_step={start_step}, end_step={end_step}, "
            f"checkpoint_stride={checkpoint_stride}."
        )
    return intervals


def _summarize_stats(stats_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in stats_rows:
        grouped.setdefault((str(row["target_id"]), str(row["source_kind"])), []).append(row)
    summary_rows: list[dict[str, Any]] = []
    for (target_id, source_kind), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["source_step"]))
        first = ordered[0]
        last = ordered[-1]
        summary_rows.append(
            {
                "target_id": target_id,
                "source_kind": source_kind,
                "num_intervals": len(ordered),
                "first_source_step": int(first["source_step"]),
                "last_source_step": int(last["source_step"]),
                "first_top_singular_value": float(first["top_singular_value"]),
                "last_top_singular_value": float(last["top_singular_value"]),
                "delta_top_singular_value": float(last["top_singular_value"]) - float(first["top_singular_value"]),
                "first_effective_rank": float(first["effective_rank"]),
                "last_effective_rank": float(last["effective_rank"]),
                "delta_effective_rank": float(last["effective_rank"]) - float(first["effective_rank"]),
                "first_spectral_mass_top3": float(first["spectral_mass_top3"]),
                "last_spectral_mass_top3": float(last["spectral_mass_top3"]),
                "delta_spectral_mass_top3": float(last["spectral_mass_top3"]) - float(first["spectral_mass_top3"]),
                "first_fro_norm": float(first["fro_norm"]),
                "last_fro_norm": float(last["fro_norm"]),
                "delta_fro_norm": float(last["fro_norm"]) - float(first["fro_norm"]),
            }
        )
    return sorted(
        summary_rows,
        key=lambda row: (str(row["target_id"]), -abs(float(row["delta_top_singular_value"]))),
    )


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Adam State Geometry Report",
        "",
        "This report tracks whether AdamW's internal state and update components become geometrically structured in the same matrices where circuit formation is being studied.",
        "",
        "For parameter matrices the SVD is taken directly. For `W_QK` and `W_OV`, non-weight source kinds are first-order functional fields, e.g. `d(W_QK) = dW_Q^T W_K + W_Q^T dW_K`.",
        "",
        "## Trace",
        "",
        f"- optimizer trace: `{report['optimizer_trace_dir']}`",
        f"- trace status: `{report['optimizer_trace_status']}`",
        f"- trace blocker: {report['optimizer_trace_blocker']}",
        f"- intervals: {report['num_intervals']}",
        f"- targets: {', '.join(report['targets'])}",
        f"- source kinds: {', '.join(report['source_kinds'])}",
        "",
        "## Largest Top-Singular-Value Changes",
        "",
        "| target | source kind | steps | top sv delta | effective-rank delta | top3 mass delta | fro norm delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary_rows"][: int(report["markdown_top_k_rows"])]:
        lines.append(
            "| `{target}` | `{source}` | {first}->{last} | {top:.6g} | {rank:.6g} | {mass:.6g} | {fro:.6g} |".format(
                target=str(row["target_id"]),
                source=str(row["source_kind"]),
                first=int(row["first_source_step"]),
                last=int(row["last_source_step"]),
                top=float(row["delta_top_singular_value"]),
                rank=float(row["delta_effective_rank"]),
                mass=float(row["delta_spectral_mass_top3"]),
                fro=float(row["delta_fro_norm"]),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- singular value rows: `{report['singular_value_rows_path']}`",
            f"- top vector rows: `{report['top_vector_rows_path']}`",
            f"- summary rows: `{report['summary_rows_path']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_adam_state_geometry_report(
    *,
    config_path: Path,
    optimizer_trace_dir: Path,
    output_dir: Path,
    targets: list[str],
    device_name: str = "cpu",
    checkpoint_paths: list[Path] | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    checkpoint_stride: int = 1,
    source_kinds: list[str] | None = None,
    train_split: str = "train",
    loss_scope: str = "full_lm",
    top_k: int = 8,
    top_vector_ranks: int = 2,
    loss_match_tolerance: float = 1.0e-4,
    grad_norm_match_tolerance: float = 1.0e-4,
    markdown_top_k_rows: int = 80,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    if not targets:
        raise ValueError("At least one --target is required.")
    parsed_targets = [_parse_target(target) for target in targets]
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if top_vector_ranks < 0:
        raise ValueError("top_vector_ranks must be non-negative.")
    if loss_scope not in LOSS_SCOPES:
        raise ValueError(f"Unsupported loss_scope {loss_scope!r}; expected one of {LOSS_SCOPES}.")
    selected_source_kinds = source_kinds if source_kinds is not None else list(DEFAULT_SOURCE_KINDS)
    unsupported_source_kinds = sorted(set(selected_source_kinds) - set(DEFAULT_SOURCE_KINDS))
    if unsupported_source_kinds:
        raise ValueError(f"Unsupported source kinds {unsupported_source_kinds}; expected one of {DEFAULT_SOURCE_KINDS}.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = TrainSpec.from_path(config_path)
    if device_name is not None:
        spec = replace(spec, device=device_name)
    if float(spec.model.dropout) != 0.0:
        raise RuntimeError("Adam state geometry report requires dropout=0.0.")
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(spec.device)
    checkpoint_dir = optimizer_trace_dir / "checkpoints"
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    intervals = _filter_intervals(
        checkpoints=checkpoints,
        start_step=start_step,
        end_step=end_step,
        checkpoint_stride=checkpoint_stride,
    )
    trace_batch_rows = _load_trace_batch_rows(optimizer_trace_dir / "optimizer_update_trace_batches.jsonl")
    trace_step_rows = _load_trace_step_rows(optimizer_trace_dir / "optimizer_update_trace_steps.jsonl")
    records_by_id = _records_by_sample_id(benchmark_dir=spec.benchmark_dir, split_name=train_split)
    optimizer_trace_status, optimizer_trace_blocker = _optimizer_trace_metadata(optimizer_trace_dir)

    singular_rows_path = output_dir / "adam_state_geometry_singular_values.jsonl"
    top_vectors_path = output_dir / "adam_state_geometry_top_vectors.jsonl"
    summary_rows_path = output_dir / "adam_state_geometry_summary_rows.jsonl"
    report_path = output_dir / "adam_state_geometry_report.json"
    markdown_path = output_dir / "adam_state_geometry_report.md"

    print(
        "[adam-state-geometry-report] "
        f"intervals={len(intervals)} targets={targets} source_kinds={selected_source_kinds} "
        f"stride={checkpoint_stride} device={spec.device} loss_scope={loss_scope}",
        flush=True,
    )

    singular_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        if target_step not in trace_batch_rows:
            raise KeyError(f"No optimizer trace batch row found for target step {target_step}.")
        if target_step not in trace_step_rows:
            raise KeyError(f"No optimizer trace step row found for target step {target_step}.")
        print(
            "[adam-state-geometry-report] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )
        batch_row = trace_batch_rows[target_step]
        step_row = trace_step_rows[target_step]
        learning_rate = float(step_row["learning_rate"])
        actual_batch_records = _records_for_trace_batch(batch_row=batch_row, records_by_id=records_by_id)

        context = _resume_training_state(spec=spec, resume_checkpoint=source_checkpoint_path)
        model: torch.nn.Module = context["model"]
        optimizer: torch.optim.Optimizer = context["optimizer"]
        source_checkpoint = context["checkpoint"]
        if int(source_checkpoint["step"]) != source_step:
            raise RuntimeError(f"Source checkpoint step mismatch: payload={source_checkpoint['step']} path={source_step}")
        source_parameters = _model_parameter_snapshot(model)
        adam_state_fields = _adam_state_tensors(model=model, optimizer=optimizer, source_step=source_step)

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
        raw_loss_gradients = loss_payload["gradients"]
        if not isinstance(raw_loss_gradients, dict):
            raise TypeError("Loss payload gradients must be a dictionary.")

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
            label=f"Adam state geometry actual delta {source_step}->{target_step}",
        )
        reconstruction_error = _sub_tensors(
            adam_components["reconstructed_adamw_update"],
            actual_delta_parameters,
            label=f"Adam state geometry reconstruction error {source_step}->{target_step}",
        )
        source_fields: dict[str, dict[str, torch.Tensor]] = {
            "weight": source_parameters,
            **adam_state_fields,
            **adam_components,
            "actual_parameter_delta": actual_delta_parameters,
            "reconstruction_error": reconstruction_error,
        }
        actual_delta_l2 = _tensor_l2_norm(actual_delta_parameters, label="actual parameter delta")
        reconstruction_error_l2 = _tensor_l2_norm(reconstruction_error, label="reconstruction error")

        for target in parsed_targets:
            parameter_names = _target_parameter_names(target=target)
            for parameter_name in parameter_names:
                if parameter_name not in source_parameters:
                    raise KeyError(f"Target {target['target_id']} references missing parameter {parameter_name!r}.")
            for source_kind in selected_source_kinds:
                matrix, functional_mode = _target_matrix(
                    target=target,
                    source_kind=source_kind,
                    tensors=source_fields[source_kind],
                    source_parameters=source_parameters,
                    model=model,
                )
                base_row = {
                    "schema_version": ADAM_STATE_GEOMETRY_REPORT_SCHEMA_VERSION,
                    "optimizer_trace_dir": str(optimizer_trace_dir),
                    "source_checkpoint": str(source_checkpoint_path),
                    "target_checkpoint": str(target_checkpoint_path),
                    "source_step": int(source_step),
                    "target_step": int(target_step),
                    "step_gap": int(target_step - source_step),
                    "target_id": str(target["target_id"]),
                    "target_type": str(target["target_type"]),
                    "layer": int(target["layer"]),
                    "head": target["head"],
                    "matrix_name": str(target["matrix"]),
                    "parameter_names": parameter_names,
                    "source_kind": source_kind,
                    "functional_matrix_mode": functional_mode,
                    "learning_rate": learning_rate,
                    "clip_coefficient": float(adam_metadata["clip_coefficient"]),
                    "actual_parameter_delta_l2": actual_delta_l2,
                    "reconstruction_error_l2": reconstruction_error_l2,
                }
                current_singular_rows, current_vector_rows, stats = _svd_rows_for_matrix(
                    matrix=matrix,
                    base_row=base_row,
                    top_k=top_k,
                    top_vector_ranks=top_vector_ranks,
                )
                singular_rows.extend(current_singular_rows)
                vector_rows.extend(current_vector_rows)
                stats_rows.append(stats)

    summary_rows = _summarize_stats(stats_rows)
    write_jsonl(singular_rows_path, singular_rows)
    write_jsonl(top_vectors_path, vector_rows)
    write_jsonl(summary_rows_path, summary_rows)
    report = {
        "schema_version": ADAM_STATE_GEOMETRY_REPORT_SCHEMA_VERSION,
        "config_path": str(config_path),
        "optimizer_trace_dir": str(optimizer_trace_dir),
        "optimizer_trace_status": optimizer_trace_status,
        "optimizer_trace_blocker": optimizer_trace_blocker,
        "num_intervals": len(intervals),
        "targets": targets,
        "source_kinds": selected_source_kinds,
        "top_k": top_k,
        "top_vector_ranks": top_vector_ranks,
        "checkpoint_stride": checkpoint_stride,
        "start_step": start_step,
        "end_step": end_step,
        "loss_scope": loss_scope,
        "singular_value_rows_path": str(singular_rows_path),
        "top_vector_rows_path": str(top_vectors_path),
        "summary_rows_path": str(summary_rows_path),
        "markdown_top_k_rows": markdown_top_k_rows,
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(
        f"[adam-state-geometry-report] complete report={report_path} rows={singular_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, singular_rows_path, top_vectors_path, summary_rows_path
