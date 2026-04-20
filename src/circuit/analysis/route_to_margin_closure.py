from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    GEOMETRY_POSITION_ROLES,
    _build_route_competition_pairs,
    _checkpoint_step_from_path,
    _contrast_margin,
    _holdout_pair_set,
    _intervention_positions_for_query,
    _pair_metadata,
    _parse_route_competition_specs,
    _resolve_causal_patch_basis,
    _resolve_checkpoint_paths,
    _route_competition_route_metadata,
    _route_group_metrics_from_logits,
    _route_objective_pairs,
    _token_ids_for_values,
    _transfer_stage_tensor,
    _validate_geometry_stage,
    _validate_query_metadata_match,
    _validate_single_query_batch,
    _value_margin,
)
from circuit.analysis.shared_feature_dynamics import _import_matplotlib
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


ROUTE_TO_MARGIN_CLOSURE_SCHEMA_VERSION = 1
ROUTE_TO_MARGIN_TARGETS = ["answer_margin", "clean_vs_corrupted_contrast"]
ROUTE_TO_MARGIN_SIDES = ["clean", "corrupted"]


def fit_route_to_margin_closure(
    *,
    route_delta_columns: dict[str, list[float]],
    margin_deltas: list[float],
    fit_intercept: bool,
    min_variance: float = 1.0e-12,
) -> dict[str, Any]:
    if not route_delta_columns:
        raise ValueError("route_delta_columns must not be empty.")
    route_labels = sorted(route_delta_columns)
    num_observations = len(margin_deltas)
    if num_observations == 0:
        raise ValueError("margin_deltas must not be empty.")
    for label in route_labels:
        if len(route_delta_columns[label]) != num_observations:
            raise ValueError(
                f"Route column {label!r} has {len(route_delta_columns[label])} rows, "
                f"but margin_deltas has {num_observations}."
            )

    x_columns = [route_delta_columns[label] for label in route_labels]
    if fit_intercept:
        x_columns = [[1.0] * num_observations, *x_columns]
    x = torch.tensor(list(zip(*x_columns, strict=True)), dtype=torch.float64)
    y = torch.tensor(margin_deltas, dtype=torch.float64).unsqueeze(1)
    if x.size(0) < x.size(1):
        raise ValueError(
            f"Underdetermined closure fit: observations={x.size(0)} parameters={x.size(1)}. "
            "Use more intervals/pairs or fewer routes."
        )
    solution = torch.linalg.lstsq(x, y).solution.squeeze(1)
    predicted = (x @ solution.unsqueeze(1)).squeeze(1)
    residual = y.squeeze(1) - predicted
    sse = float(torch.sum(residual * residual).item())
    centered = y.squeeze(1) - torch.mean(y.squeeze(1))
    sst = float(torch.sum(centered * centered).item())
    r_squared = None if sst <= min_variance else 1.0 - (sse / sst)
    rank = int(torch.linalg.matrix_rank(x).item())
    coefficients: dict[str, float] = {}
    offset = 0
    intercept = 0.0
    if fit_intercept:
        intercept = float(solution[0].item())
        offset = 1
    for index, label in enumerate(route_labels):
        coefficients[label] = float(solution[index + offset].item())
    return {
        "route_labels": route_labels,
        "fit_intercept": fit_intercept,
        "intercept": intercept,
        "coefficients": coefficients,
        "num_observations": num_observations,
        "num_parameters": int(x.size(1)),
        "matrix_rank": rank,
        "rank_deficient": rank < int(x.size(1)),
        "sse": sse,
        "sst": sst,
        "r_squared": r_squared,
        "mean_actual_margin_delta": float(torch.mean(y.squeeze(1)).item()),
        "mean_predicted_margin_delta": float(torch.mean(predicted).item()),
        "mean_residual": float(torch.mean(residual).item()),
        "mean_abs_residual": float(torch.mean(torch.abs(residual)).item()),
        "predicted_values": [float(value) for value in predicted.tolist()],
        "residual_values": [float(value) for value in residual.tolist()],
    }


def _compute_answer_scalar_rows_for_pairs(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    record_side: str,
    target_scalar: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if record_side not in ROUTE_TO_MARGIN_SIDES:
        raise ValueError(f"Unsupported margin side {record_side!r}; expected one of {ROUTE_TO_MARGIN_SIDES}.")
    if target_scalar not in ROUTE_TO_MARGIN_TARGETS:
        raise ValueError(f"Unsupported target scalar {target_scalar!r}; expected one of {ROUTE_TO_MARGIN_TARGETS}.")
    if not pairs:
        raise ValueError("pairs must not be empty for answer-scalar computation.")
    checkpoint = load_checkpoint(checkpoint_path, device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()
    step = int(checkpoint["step"])
    path_step = _checkpoint_step_from_path(checkpoint_path)
    if step != path_step:
        raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={step} path={path_step}")
    value_token_ids = torch.tensor(vocab.value_token_ids, device=device, dtype=torch.long)
    rows: list[dict[str, Any]] = []
    side_key = f"{record_side}_record"
    with torch.no_grad():
        for start_index in range(0, len(pairs), batch_size):
            pair_batch = pairs[start_index : start_index + batch_size]
            records = [pair[side_key] for pair in pair_batch]
            batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=metadata, label="route-to-margin closure")
            answer_margins = _value_margin(answer_logits, answer_targets, value_token_ids)
            answer_losses = torch.nn.functional.cross_entropy(answer_logits, answer_targets, reduction="none")
            clean_transfer_ids = _token_ids_for_values(
                values=[str(pair["clean_transfer_token"]) for pair in pair_batch],
                vocab=vocab,
                device=device,
            )
            corrupted_transfer_ids = _token_ids_for_values(
                values=[str(pair["corrupted_transfer_token"]) for pair in pair_batch],
                vocab=vocab,
                device=device,
            )
            contrast = _contrast_margin(
                answer_logits=answer_logits,
                positive_token_ids=clean_transfer_ids,
                negative_token_ids=corrupted_transfer_ids,
            )
            if target_scalar == "answer_margin":
                scalar_values = answer_margins
            elif target_scalar == "clean_vs_corrupted_contrast":
                scalar_values = contrast
            else:
                raise RuntimeError(f"Unhandled target scalar: {target_scalar}")
            predictions = answer_logits.argmax(dim=-1)
            for pair_index, pair in enumerate(pair_batch):
                rows.append(
                    {
                        "step": step,
                        "checkpoint": str(checkpoint_path),
                        "pair_id": str(pair["pair_id"]),
                        "split": str(pair["split"]),
                        "pair_type": str(pair["pair_type"]),
                        "record_side": record_side,
                        "target_scalar": target_scalar,
                        "source_sample_id": str(pair["source_sample_id"]),
                        "source_query_index": int(pair["source_query_index"]),
                        "clean_transfer_token": str(pair["clean_transfer_token"]),
                        "corrupted_transfer_token": str(pair["corrupted_transfer_token"]),
                        "answer_target_id": int(answer_targets[pair_index].detach().cpu().item()),
                        "answer_prediction_id": int(predictions[pair_index].detach().cpu().item()),
                        "answer_correct": bool((predictions[pair_index] == answer_targets[pair_index]).detach().cpu().item()),
                        "answer_margin": float(answer_margins[pair_index].detach().float().cpu().item()),
                        "answer_loss": float(answer_losses[pair_index].detach().float().cpu().item()),
                        "clean_vs_corrupted_contrast": float(contrast[pair_index].detach().float().cpu().item()),
                        "value": float(scalar_values[pair_index].detach().float().cpu().item()),
                    }
                )
    return rows


def _compute_route_score_rows_for_pairs(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    basis: torch.Tensor | None,
    stage_name: str,
    position_role: str,
    route_label: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if not pairs:
        raise ValueError("pairs must not be empty for route score row computation.")
    checkpoint = load_checkpoint(checkpoint_path, device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()
    step = int(checkpoint["step"])
    path_step = _checkpoint_step_from_path(checkpoint_path)
    if step != path_step:
        raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={step} path={path_step}")
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start_index in range(0, len(pairs), batch_size):
            pair_batch = pairs[start_index : start_index + batch_size]
            clean_records = [pair["clean_record"] for pair in pair_batch]
            corrupted_records = [pair["corrupted_record"] for pair in pair_batch]
            clean_batch = move_batch_to_device(collate_symbolic_kv(clean_records, pad_token_id), device)
            corrupted_batch = move_batch_to_device(collate_symbolic_kv(corrupted_records, pad_token_id), device)
            clean_outputs = model(
                clean_batch["input_ids"],
                attention_mask=clean_batch["attention_mask"],
                return_residual_streams=True,
            )
            corrupted_outputs = model(
                corrupted_batch["input_ids"],
                attention_mask=corrupted_batch["attention_mask"],
                return_residual_streams=True,
            )
            if clean_outputs.residual_streams is None or corrupted_outputs.residual_streams is None:
                raise RuntimeError("Route-to-margin closure requires residual streams for route scores.")
            if stage_name not in clean_outputs.residual_streams:
                raise KeyError(f"Stage {stage_name!r} not found in clean residual streams.")
            if stage_name not in corrupted_outputs.residual_streams:
                raise KeyError(f"Stage {stage_name!r} not found in corrupted residual streams.")
            clean_answer_logits, _, clean_metadata = extract_answer_logits(clean_outputs.logits, clean_batch)
            corrupted_answer_logits, _, corrupted_metadata = extract_answer_logits(corrupted_outputs.logits, corrupted_batch)
            _validate_single_query_batch(batch=clean_batch, metadata=clean_metadata, label="closure route clean")
            _validate_single_query_batch(batch=corrupted_batch, metadata=corrupted_metadata, label="closure route corrupted")
            clean_selected = [
                _intervention_positions_for_query(
                    batch=clean_batch,
                    metadata=clean_metadata,
                    flat_index=flat_index,
                    position_role=position_role,
                )
                for flat_index in range(len(pair_batch))
            ]
            corrupted_selected = [
                _intervention_positions_for_query(
                    batch=corrupted_batch,
                    metadata=corrupted_metadata,
                    flat_index=flat_index,
                    position_role=position_role,
                )
                for flat_index in range(len(pair_batch))
            ]
            patched_stage = _transfer_stage_tensor(
                clean_stage=clean_outputs.residual_streams[stage_name],
                corrupted_stage=corrupted_outputs.residual_streams[stage_name],
                clean_selected=clean_selected,
                corrupted_selected=corrupted_selected,
                basis=basis,
            )
            patched_outputs = model(
                corrupted_batch["input_ids"],
                attention_mask=corrupted_batch["attention_mask"],
                residual_patch={stage_name: patched_stage},
            )
            patched_answer_logits, _, patched_metadata = extract_answer_logits(patched_outputs.logits, corrupted_batch)
            _validate_query_metadata_match(
                baseline_metadata=corrupted_metadata,
                patched_metadata=patched_metadata,
            )
            metric_tensors = _route_group_metrics_from_logits(
                clean_answer_logits=clean_answer_logits,
                corrupted_answer_logits=corrupted_answer_logits,
                patched_answer_logits=patched_answer_logits,
                pairs=pair_batch,
                vocab=vocab,
                device=device,
            )
            for pair_index, pair in enumerate(pair_batch):
                rows.append(
                    {
                        "step": step,
                        "checkpoint": str(checkpoint_path),
                        "route_label": route_label,
                        "pair_id": str(pair["pair_id"]),
                        "split": str(pair["split"]),
                        "pair_type": str(pair["pair_type"]),
                        "source_sample_id": str(pair["source_sample_id"]),
                        "source_query_index": int(pair["source_query_index"]),
                        "transfer_margin_clean": float(
                            metric_tensors["transfer_margin_clean"][pair_index].detach().float().cpu().item()
                        ),
                        "transfer_margin_corrupted": float(
                            metric_tensors["transfer_margin_corrupted"][pair_index].detach().float().cpu().item()
                        ),
                        "transfer_margin_patched": float(
                            metric_tensors["transfer_margin_patched"][pair_index].detach().float().cpu().item()
                        ),
                        "route_score": float(metric_tensors["route_delta"][pair_index].detach().float().cpu().item()),
                    }
                )
    return rows


def _rows_by_pair_id(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        pair_id = str(row["pair_id"])
        if pair_id in by_pair:
            raise RuntimeError(f"Duplicate pair_id {pair_id!r} in {label}.")
        by_pair[pair_id] = row
    return by_pair


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"Cannot compute mean for empty values: {label}")
    return float(sum(values) / len(values))


def _safe_r_squared(y_values: list[float], predicted_values: list[float]) -> float | None:
    if len(y_values) != len(predicted_values):
        raise ValueError("y_values and predicted_values must have the same length.")
    mean_y = _mean(y_values, label="r2 y")
    sst = sum((value - mean_y) ** 2 for value in y_values)
    if sst <= 1.0e-12:
        return None
    sse = sum((actual - predicted) ** 2 for actual, predicted in zip(y_values, predicted_values, strict=True))
    return float(1.0 - (sse / sst))


def _summarize_route_to_margin_closure(
    *,
    closure_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    fit: dict[str, Any],
) -> dict[str, Any]:
    if not closure_rows:
        raise ValueError("Cannot summarize route-to-margin closure without closure rows.")
    intervals = sorted({(int(row["source_step"]), int(row["target_step"])) for row in closure_rows})
    route_labels = [str(row["route_label"]) for row in coefficient_rows]
    actual_values = [float(row["actual_margin_delta"]) for row in closure_rows]
    predicted_values = [float(row["predicted_margin_delta"]) for row in closure_rows]
    residual_values = [float(row["closure_residual"]) for row in closure_rows]
    interval_rows: list[dict[str, Any]] = []
    for source_step, target_step in intervals:
        rows = [
            row
            for row in closure_rows
            if int(row["source_step"]) == source_step and int(row["target_step"]) == target_step
        ]
        interval_rows.append(
            {
                "source_step": source_step,
                "target_step": target_step,
                "num_observations": len(rows),
                "actual_margin_delta_mean": _mean(
                    [float(row["actual_margin_delta"]) for row in rows],
                    label=f"actual {source_step}->{target_step}",
                ),
                "predicted_margin_delta_mean": _mean(
                    [float(row["predicted_margin_delta"]) for row in rows],
                    label=f"predicted {source_step}->{target_step}",
                ),
                "closure_residual_mean": _mean(
                    [float(row["closure_residual"]) for row in rows],
                    label=f"residual {source_step}->{target_step}",
                ),
                "closure_abs_residual_mean": _mean(
                    [abs(float(row["closure_residual"])) for row in rows],
                    label=f"abs residual {source_step}->{target_step}",
                ),
                "r_squared": _safe_r_squared(
                    y_values=[float(row["actual_margin_delta"]) for row in rows],
                    predicted_values=[float(row["predicted_margin_delta"]) for row in rows],
                ),
            }
        )
    return {
        "num_observations": len(closure_rows),
        "num_intervals": len(intervals),
        "route_labels": route_labels,
        "fit": {
            key: value
            for key, value in fit.items()
            if key not in {"predicted_values", "residual_values"}
        },
        "actual_margin_delta_mean": _mean(actual_values, label="summary actual"),
        "predicted_margin_delta_mean": _mean(predicted_values, label="summary predicted"),
        "closure_residual_mean": _mean(residual_values, label="summary residual"),
        "closure_abs_residual_mean": _mean([abs(value) for value in residual_values], label="summary abs residual"),
        "r_squared": _safe_r_squared(y_values=actual_values, predicted_values=predicted_values),
        "interval_rows": interval_rows,
        "route_contributions": sorted(
            coefficient_rows,
            key=lambda row: abs(float(row["mean_contribution"])),
            reverse=True,
        ),
    }


def _plot_closure_actual_vs_predicted(
    *,
    closure_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not closure_rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 7))
    actual = [float(row["actual_margin_delta"]) for row in closure_rows]
    predicted = [float(row["predicted_margin_delta"]) for row in closure_rows]
    ax.scatter(predicted, actual, color="#376f8f", alpha=0.65)
    values = actual + predicted
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    ax.plot([min_value, max_value], [min_value, max_value], color="#777777", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.axvline(0.0, color="#999999", linewidth=0.8)
    ax.set_title("Route-to-margin closure: predicted vs actual")
    ax.set_xlabel("predicted margin delta from route deltas")
    ax.set_ylabel("actual margin delta")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_closure_interval_trajectory(
    *,
    interval_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not interval_rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{row['source_step']}->{row['target_step']}" for row in interval_rows]
    x_values = list(range(len(interval_rows)))
    actual = [float(row["actual_margin_delta_mean"]) for row in interval_rows]
    predicted = [float(row["predicted_margin_delta_mean"]) for row in interval_rows]
    residual = [float(row["closure_residual_mean"]) for row in interval_rows]
    ax.plot(x_values, actual, marker="o", label="actual margin delta", color="#376f8f")
    ax.plot(x_values, predicted, marker="o", label="route-predicted margin delta", color="#6c8f37")
    ax.plot(x_values, residual, marker="o", label="residual", color="#8f374a")
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Route-to-margin closure by interval")
    ax.set_ylabel("mean delta")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_route_contributions(
    *,
    coefficient_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not coefficient_rows:
        return None
    rows = sorted(coefficient_rows, key=lambda row: abs(float(row["mean_contribution"])), reverse=True)
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(rows)), 5))
    labels = [str(row["route_label"]) for row in rows]
    values = [float(row["mean_contribution"]) for row in rows]
    colors = ["#376f8f" if value >= 0.0 else "#8f374a" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title("Mean fitted route contribution")
    ax.set_ylabel("coefficient * mean route delta")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_route_to_margin_closure_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    fit = summary["fit"]
    r2_value = summary["r_squared"]
    r2_text = "" if r2_value is None else f"{float(r2_value):.6f}"
    lines = [
        "# Route-To-Margin Closure",
        "",
        "## Calculation",
        "",
        "This report tests whether measured route-score growth explains answer-margin growth on the same causal pairs.",
        "",
        "```text",
        "Delta m_t(pair) = target_scalar(theta_target, pair) - target_scalar(theta_source, pair)",
        "Delta C_{P,t}(pair) = route_score_P(theta_target; source_basis, pair) - route_score_P(theta_source; source_basis, pair)",
        "Delta m_t(pair) ~= sum_P beta_P Delta C_{P,t}(pair) + residual_t(pair)",
        "```",
        "",
        "The subspace basis for every route is fixed at the source checkpoint of each interval.",
        "",
        "## Inputs",
        "",
        f"- target scalar: `{report['target_scalar']}`",
        f"- margin side: `{report['margin_side']}`",
        f"- route pair type: `{report['route_pair_type']}`",
        f"- fit intercept: `{bool(report['fit_intercept'])}`",
        f"- observations: `{int(summary['num_observations'])}`",
        f"- intervals: `{int(summary['num_intervals'])}`",
        "",
        "## Closure Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| mean actual margin delta | {float(summary['actual_margin_delta_mean']):.6g} |",
        f"| mean predicted margin delta | {float(summary['predicted_margin_delta_mean']):.6g} |",
        f"| mean residual | {float(summary['closure_residual_mean']):.6g} |",
        f"| mean absolute residual | {float(summary['closure_abs_residual_mean']):.6g} |",
        f"| R squared | {r2_text} |",
        f"| design rank | {int(fit['matrix_rank'])} / {int(fit['num_parameters'])} |",
        f"| rank deficient | {bool(fit['rank_deficient'])} |",
        "",
        "## Route Coefficients",
        "",
        "| route | coefficient | mean route delta | mean contribution |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["route_contributions"]:
        lines.append(
            "| {route} | {coef:.6g} | {delta:.6g} | {contribution:.6g} |".format(
                route=row["route_label"],
                coef=float(row["coefficient"]),
                delta=float(row["mean_route_score_delta"]),
                contribution=float(row["mean_contribution"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interval Closure",
            "",
            "| interval | observations | actual mean | predicted mean | residual mean | abs residual mean | R squared |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["interval_rows"]:
        interval_r2 = row["r_squared"]
        lines.append(
            "| {source}->{target} | {n} | {actual:.6g} | {pred:.6g} | {resid:.6g} | {abs_resid:.6g} | {r2} |".format(
                source=int(row["source_step"]),
                target=int(row["target_step"]),
                n=int(row["num_observations"]),
                actual=float(row["actual_margin_delta_mean"]),
                pred=float(row["predicted_margin_delta_mean"]),
                resid=float(row["closure_residual_mean"]),
                abs_resid=float(row["closure_abs_residual_mean"]),
                r2="" if interval_r2 is None else f"{float(interval_r2):.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Route Specs",
            "",
        ]
    )
    for route in report["routes"]:
        lines.append(f"- `{route['route_label']}`: `{route}`")
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- closure rows: `{report['closure_rows_path']}`",
            f"- margin rows: `{report['margin_rows_path']}`",
            f"- route rows: `{report['route_rows_path']}`",
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


def run_route_to_margin_closure(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    raw_route_specs: list[str],
    route_pair_type: str,
    pair_types: list[str],
    target_scalar: str = "answer_margin",
    margin_side: str = "clean",
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    split_filter: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    fit_intercept: bool = False,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if target_scalar not in ROUTE_TO_MARGIN_TARGETS:
        raise ValueError(f"Unsupported target scalar {target_scalar!r}; expected one of {ROUTE_TO_MARGIN_TARGETS}.")
    if margin_side not in ROUTE_TO_MARGIN_SIDES:
        raise ValueError(f"Unsupported margin side {margin_side!r}; expected one of {ROUTE_TO_MARGIN_SIDES}.")
    routes = _parse_route_competition_specs(raw_route_specs)
    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("route-to-margin-closure requires at least two checkpoints.")
    model = build_model(spec.model, len(vocab.tokens), device)
    for route in routes:
        _validate_geometry_stage(model=model, stage_name=route.stage_name)
        if route.position_role not in GEOMETRY_POSITION_ROLES:
            raise ValueError(f"Unsupported position role {route.position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")

    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    pair_types = sorted(set(pair_types), key=pair_types.index)
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
    route_pairs = _route_objective_pairs(
        pairs=pairs,
        route_split="__all__",
        route_pair_type=route_pair_type,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows_path = output_dir / "route_to_margin_closure_pairs.jsonl"
    margin_rows_path = output_dir / "route_to_margin_closure_margin_rows.jsonl"
    route_rows_path = output_dir / "route_to_margin_closure_route_rows.jsonl"
    closure_rows_path = output_dir / "route_to_margin_closure_rows.jsonl"
    coefficient_rows_path = output_dir / "route_to_margin_closure_coefficients.jsonl"
    report_path = output_dir / "route_to_margin_closure_report.json"
    markdown_path = output_dir / "route_to_margin_closure_report.md"
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in route_pairs])

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[route-to-margin-closure] "
        f"routes={len(routes)} intervals={len(intervals)} pairs={len(route_pairs)} "
        f"route_pair_type={route_pair_type} target={target_scalar} margin_side={margin_side} device={device_name}",
        flush=True,
    )

    all_margin_rows: list[dict[str, Any]] = []
    all_route_rows: list[dict[str, Any]] = []
    closure_rows_without_fit: list[dict[str, Any]] = []
    final_subspace_summaries: dict[str, dict[str, Any]] = {}
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        print(
            "[route-to-margin-closure] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )
        source_margin_rows = _compute_answer_scalar_rows_for_pairs(
            model=model,
            checkpoint_path=source_checkpoint_path,
            pairs=route_pairs,
            vocab=vocab,
            record_side=margin_side,
            target_scalar=target_scalar,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
        )
        target_margin_rows = _compute_answer_scalar_rows_for_pairs(
            model=model,
            checkpoint_path=target_checkpoint_path,
            pairs=route_pairs,
            vocab=vocab,
            record_side=margin_side,
            target_scalar=target_scalar,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
        )
        all_margin_rows.extend(source_margin_rows)
        all_margin_rows.extend(target_margin_rows)
        source_margin_by_pair = _rows_by_pair_id(source_margin_rows, label=f"source margin {source_step}")
        target_margin_by_pair = _rows_by_pair_id(target_margin_rows, label=f"target margin {target_step}")
        if set(source_margin_by_pair) != set(target_margin_by_pair):
            raise RuntimeError(f"Margin pair ids changed across interval {source_step}->{target_step}.")

        route_delta_by_label: dict[str, dict[str, float]] = {}
        for route in routes:
            source_checkpoint = load_checkpoint(source_checkpoint_path, device)
            load_model_state(model, source_checkpoint["model_state"])
            basis, subspace_summary = _resolve_causal_patch_basis(
                model=model,
                vocab=vocab,
                subspace_name=route.subspace_name,
                rank=route.rank,
                head_layer=route.head_layer,
                head=route.head,
                device=device,
            )
            final_subspace_summaries[route.label] = subspace_summary
            source_route_rows = _compute_route_score_rows_for_pairs(
                model=model,
                checkpoint_path=source_checkpoint_path,
                pairs=route_pairs,
                vocab=vocab,
                basis=basis,
                stage_name=route.stage_name,
                position_role=route.position_role,
                route_label=route.label,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
            )
            target_route_rows = _compute_route_score_rows_for_pairs(
                model=model,
                checkpoint_path=target_checkpoint_path,
                pairs=route_pairs,
                vocab=vocab,
                basis=basis,
                stage_name=route.stage_name,
                position_role=route.position_role,
                route_label=route.label,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
            )
            all_route_rows.extend(source_route_rows)
            all_route_rows.extend(target_route_rows)
            source_route_by_pair = _rows_by_pair_id(source_route_rows, label=f"{route.label} source route {source_step}")
            target_route_by_pair = _rows_by_pair_id(target_route_rows, label=f"{route.label} target route {target_step}")
            if set(source_route_by_pair) != set(target_route_by_pair):
                raise RuntimeError(f"Route pair ids changed for {route.label} across {source_step}->{target_step}.")
            route_delta_by_label[route.label] = {
                pair_id: float(target_route_by_pair[pair_id]["route_score"])
                - float(source_route_by_pair[pair_id]["route_score"])
                for pair_id in source_route_by_pair
            }

        for pair in route_pairs:
            pair_id = str(pair["pair_id"])
            route_deltas = {
                label: float(delta_by_pair[pair_id])
                for label, delta_by_pair in route_delta_by_label.items()
            }
            closure_rows_without_fit.append(
                {
                    "source_step": source_step,
                    "target_step": target_step,
                    "step_gap": target_step - source_step,
                    "source_checkpoint": str(source_checkpoint_path),
                    "target_checkpoint": str(target_checkpoint_path),
                    "pair_id": pair_id,
                    "split": str(pair["split"]),
                    "pair_type": str(pair["pair_type"]),
                    "record_side": margin_side,
                    "target_scalar": target_scalar,
                    "source_margin": float(source_margin_by_pair[pair_id]["value"]),
                    "target_margin": float(target_margin_by_pair[pair_id]["value"]),
                    "actual_margin_delta": float(target_margin_by_pair[pair_id]["value"])
                    - float(source_margin_by_pair[pair_id]["value"]),
                    "route_score_deltas": route_deltas,
                }
            )
        print(
            "[route-to-margin-closure] finished "
            f"{source_step}->{target_step} mean_margin_delta="
            f"{_mean([float(row['actual_margin_delta']) for row in closure_rows_without_fit if int(row['source_step']) == source_step and int(row['target_step']) == target_step], label='interval margin delta'):.6g}",
            flush=True,
        )

    route_delta_columns = {
        route.label: [float(row["route_score_deltas"][route.label]) for row in closure_rows_without_fit]
        for route in routes
    }
    margin_deltas = [float(row["actual_margin_delta"]) for row in closure_rows_without_fit]
    fit = fit_route_to_margin_closure(
        route_delta_columns=route_delta_columns,
        margin_deltas=margin_deltas,
        fit_intercept=fit_intercept,
    )
    coefficient_rows: list[dict[str, Any]] = []
    for route in routes:
        label = route.label
        mean_route_delta = _mean(route_delta_columns[label], label=f"{label} route delta")
        coefficient = float(fit["coefficients"][label])
        coefficient_rows.append(
            {
                "route_label": label,
                "coefficient": coefficient,
                "mean_route_score_delta": mean_route_delta,
                "mean_contribution": coefficient * mean_route_delta,
                **_route_competition_route_metadata(route),
            }
        )
    closure_rows: list[dict[str, Any]] = []
    predicted_values = fit["predicted_values"]
    residual_values = fit["residual_values"]
    for index, row in enumerate(closure_rows_without_fit):
        fitted_row = dict(row)
        fitted_row["predicted_margin_delta"] = float(predicted_values[index])
        fitted_row["closure_residual"] = float(residual_values[index])
        closure_rows.append(fitted_row)

    write_jsonl(margin_rows_path, all_margin_rows)
    write_jsonl(route_rows_path, all_route_rows)
    write_jsonl(closure_rows_path, closure_rows)
    write_jsonl(coefficient_rows_path, coefficient_rows)
    summary = _summarize_route_to_margin_closure(
        closure_rows=closure_rows,
        coefficient_rows=coefficient_rows,
        fit=fit,
    )
    plot_paths: dict[str, Path] = {}
    actual_vs_predicted_plot = _plot_closure_actual_vs_predicted(
        closure_rows=closure_rows,
        output_path=output_dir / "route_to_margin_closure_actual_vs_predicted.svg",
    )
    if actual_vs_predicted_plot is not None:
        plot_paths["actual_vs_predicted"] = actual_vs_predicted_plot
    trajectory_plot = _plot_closure_interval_trajectory(
        interval_rows=summary["interval_rows"],
        output_path=output_dir / "route_to_margin_closure_interval_trajectory.svg",
    )
    if trajectory_plot is not None:
        plot_paths["interval_trajectory"] = trajectory_plot
    contribution_plot = _plot_route_contributions(
        coefficient_rows=coefficient_rows,
        output_path=output_dir / "route_to_margin_closure_route_contributions.svg",
    )
    if contribution_plot is not None:
        plot_paths["route_contributions"] = contribution_plot

    report = {
        "schema_version": ROUTE_TO_MARGIN_CLOSURE_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "device": device_name,
        "routes": [_route_competition_route_metadata(route) for route in routes],
        "subspaces": final_subspace_summaries,
        "pair_types": pair_types,
        "route_pair_type": route_pair_type,
        "target_scalar": target_scalar,
        "margin_side": margin_side,
        "split_filter": split_filter,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "fit_intercept": fit_intercept,
        "basis_mode": "source_checkpoint_per_interval",
        "pair_construction": pair_construction,
        "calculation": {
            "margin_delta": "target_scalar(theta_target, pair) - target_scalar(theta_source, pair)",
            "route_score": "patched_transfer_margin - corrupted_transfer_margin",
            "route_delta": "route_score(theta_target; source_basis) - route_score(theta_source; source_basis)",
            "closure": "margin_delta ~= sum_route beta_route * route_delta + residual",
            "basis_warning": "Each route basis is fixed from the source checkpoint for that interval.",
        },
        "pair_rows_path": str(pair_rows_path),
        "margin_rows_path": str(margin_rows_path),
        "route_rows_path": str(route_rows_path),
        "closure_rows_path": str(closure_rows_path),
        "coefficient_rows_path": str(coefficient_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_route_to_margin_closure_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    print(
        f"[route-to-margin-closure] complete report={report_path} rows={closure_rows_path}",
        flush=True,
    )
    return (
        report_path,
        markdown_path,
        closure_rows_path,
        margin_rows_path,
        route_rows_path,
        coefficient_rows_path,
        pair_rows_path,
        plot_paths,
    )
