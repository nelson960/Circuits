from __future__ import annotations

import csv
import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.contextual_svd_alignment import CONTEXTUAL_GROUP_BY_OPTIONS
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import GEOMETRY_POSITION_ROLES, _intervention_positions_for_query
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import iter_jsonl, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


CONTEXTUAL_KEY_SEPARABILITY_SCHEMA_VERSION = 1


def _checkpoint_step_from_path(path: Path) -> int:
    stem = path.stem
    prefix = "step_"
    if not stem.startswith(prefix):
        raise ValueError(f"Checkpoint filename must start with '{prefix}': {path}")
    return int(stem[len(prefix) :])


def _resolve_checkpoint_paths(*, checkpoint_dir: Path, checkpoint_paths: list[Path] | None) -> list[Path]:
    if checkpoint_paths is None:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        resolved = sorted(checkpoint_dir.glob("step_*.pt"), key=_checkpoint_step_from_path)
    else:
        resolved = [Path(path) for path in checkpoint_paths]
    if not resolved:
        raise FileNotFoundError(f"No checkpoints provided or found in {checkpoint_dir}")
    missing = [path for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint path(s): {[str(path) for path in missing]}")
    return sorted(resolved, key=_checkpoint_step_from_path)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / "contextual_key_separability_report.json",
        output_dir / "contextual_key_separability_report.md",
        output_dir / "contextual_key_separability_rows.jsonl",
        output_dir / "contextual_key_separability_rows.csv",
        output_dir / "contextual_key_separability_group_rows.jsonl",
        output_dir / "contextual_key_separability_trajectory.svg",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing contextual key-separability outputs without --overwrite: "
            f"{[str(path) for path in existing]}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _make_probe_loader(*, probe_records: list[dict[str, Any]], batch_size: int, pad_token_id: int) -> DataLoader[Any]:
    if not probe_records:
        raise ValueError("probe_records must not be empty.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    return DataLoader(
        probe_records,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_symbolic_kv(batch, pad_token_id),
    )


def _load_behavior_by_step(
    *,
    behavior_rows_path: Path | None,
    behavior_split: str,
    margin_field: str,
    accuracy_field: str,
) -> dict[int, dict[str, float]]:
    if behavior_rows_path is None:
        return {}
    if not behavior_rows_path.exists():
        raise FileNotFoundError(f"Behavior rows file not found: {behavior_rows_path}")
    rows_by_step: dict[int, dict[str, float]] = {}
    for row in iter_jsonl(behavior_rows_path):
        if str(row.get("split")) != behavior_split:
            continue
        if margin_field not in row:
            raise KeyError(f"Missing behavior margin field '{margin_field}' in row from {behavior_rows_path}: {row}")
        step = int(row["step"])
        if step in rows_by_step:
            raise RuntimeError(f"Duplicate behavior row for split={behavior_split} step={step} in {behavior_rows_path}")
        behavior = {"answer_margin": float(row[margin_field])}
        if accuracy_field in row:
            behavior["answer_accuracy"] = float(row[accuracy_field])
        rows_by_step[step] = behavior
    if not rows_by_step:
        raise RuntimeError(f"No behavior rows found for split={behavior_split} in {behavior_rows_path}")
    return rows_by_step


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise RuntimeError(f"Pearson inputs have different lengths: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / (denom_x * denom_y)


def _import_matplotlib() -> Any:
    cache_dir = Path(tempfile.gettempdir()) / "circuit_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _orthonormal_span(basis: torch.Tensor, *, label: str) -> torch.Tensor:
    if basis.ndim != 2:
        raise ValueError(f"{label} basis must be rank-2, got shape {tuple(basis.shape)}.")
    rank = int(torch.linalg.matrix_rank(basis.float()).item())
    if rank <= 0:
        raise RuntimeError(f"{label} basis has zero rank.")
    u, _, _ = torch.linalg.svd(basis.float(), full_matrices=False)
    return u[:, :rank].contiguous()


def _projection_bases(
    *,
    model: torch.nn.Module,
    head_layer: int,
    head: int,
    projection_rank: int,
    include_full_residual: bool,
) -> tuple[dict[str, torch.Tensor | None], dict[str, float]]:
    block = model.blocks[head_layer]
    head_dim = int(block.attn.head_dim)
    if projection_rank <= 0:
        raise ValueError(f"projection_rank must be positive, got {projection_rank}.")
    if projection_rank > int(model.spec.d_model):
        raise ValueError(f"projection_rank {projection_rank} exceeds d_model={int(model.spec.d_model)}.")
    head_slice = slice(head * head_dim, (head + 1) * head_dim)
    q_rows = block.attn.q_proj.weight.detach().float().cpu()[head_slice, :]
    k_rows = block.attn.k_proj.weight.detach().float().cpu()[head_slice, :]
    qk_matrix = q_rows.T.matmul(k_rows)
    u, singular_values, vh = torch.linalg.svd(qk_matrix, full_matrices=False)
    keep = min(projection_rank, int(singular_values.numel()))
    if keep < projection_rank:
        raise RuntimeError(f"Requested projection_rank={projection_rank}, but W_QK only has {keep} singular values.")
    left = u[:, :keep].contiguous()
    right = vh[:keep, :].T.contiguous()
    both = _orthonormal_span(torch.cat([left, right], dim=1), label="qk_both")
    bases: dict[str, torch.Tensor | None] = {
        "qk_left": left,
        "qk_right": right,
        "qk_both": both,
    }
    if include_full_residual:
        bases["full_residual"] = None
    singular_summary = {
        "qk_singular_value_top": float(singular_values[0].item()),
        "qk_singular_value_sum": float(singular_values[:keep].sum().item()),
    }
    return bases, singular_summary


def _resolve_group_token_id(
    *,
    group_by: str,
    batch: dict[str, Any],
    metadata: dict[str, torch.Tensor],
    answer_targets: torch.Tensor,
    flat_index: int,
    context_batch_row: int,
    context_position: int,
) -> int:
    query_batch_row = int(metadata["rows"][flat_index].item())
    query_index = int(metadata["query_indices"][flat_index].item())
    record = batch["records"][query_batch_row]
    if group_by == "position_token":
        return int(batch["input_ids"][context_batch_row, context_position].item())
    if group_by == "query_key":
        return int(batch["input_ids"][query_batch_row, int(metadata["query_key_positions"][flat_index].item())].item())
    if group_by == "support_value":
        return int(batch["input_ids"][query_batch_row, int(metadata["support_value_positions"][flat_index].item())].item())
    if group_by == "answer_value":
        return int(answer_targets[flat_index].item())
    if group_by == "support_key":
        support_batch_row, support_positions = _intervention_positions_for_query(
            batch=batch,
            metadata=metadata,
            flat_index=flat_index,
            position_role="support_key",
        )
        if support_batch_row != query_batch_row or len(support_positions) != 1:
            raise RuntimeError(
                f"Expected exactly one support_key position for {record['sample_id']} query {query_index}, "
                f"got row={support_batch_row} positions={support_positions}."
            )
        return int(batch["input_ids"][support_batch_row, support_positions[0]].item())
    raise ValueError(f"Unhandled group_by mode: {group_by}")


def _collect_vectors_by_stage(
    *,
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    context_stages: list[str],
    context_role: str,
    group_by: str,
) -> tuple[dict[str, dict[int, list[torch.Tensor]]], list[dict[str, Any]]]:
    vectors_by_stage: dict[str, dict[int, list[torch.Tensor]]] = {
        stage: defaultdict(list) for stage in context_stages
    }
    group_rows: list[dict[str, Any]] = []
    for batch_index, raw_batch in enumerate(loader):
        batch = move_batch_to_device(raw_batch, device)
        with torch.no_grad():
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
            )
        if outputs.residual_streams is None:
            raise RuntimeError("Contextual key separability requires residual streams.")
        missing_stages = [stage for stage in context_stages if stage not in outputs.residual_streams]
        if missing_stages:
            raise KeyError(
                f"Residual stage(s) {missing_stages} not found. Available stages: {sorted(outputs.residual_streams)}"
            )
        _, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        for flat_index in range(int(metadata["rows"].numel())):
            query_batch_row = int(metadata["rows"][flat_index].item())
            query_index = int(metadata["query_indices"][flat_index].item())
            record = batch["records"][query_batch_row]
            sample_id = str(record["sample_id"])
            split = str(record["split"])
            context_batch_row, positions = _intervention_positions_for_query(
                batch=batch,
                metadata=metadata,
                flat_index=flat_index,
                position_role=context_role,
            )
            for position in positions:
                group_token_id = _resolve_group_token_id(
                    group_by=group_by,
                    batch=batch,
                    metadata=metadata,
                    answer_targets=answer_targets,
                    flat_index=flat_index,
                    context_batch_row=context_batch_row,
                    context_position=position,
                )
                position_token_id = int(batch["input_ids"][context_batch_row, position].item())
                for stage in context_stages:
                    vector = outputs.residual_streams[stage].detach().float().cpu()[context_batch_row, position, :].clone()
                    vectors_by_stage[stage][group_token_id].append(vector)
                    group_rows.append(
                        {
                            "stage": stage,
                            "batch_index": batch_index,
                            "sample_id": sample_id,
                            "split": split,
                            "query_index": query_index,
                            "context_role": context_role,
                            "context_position": int(position),
                            "group_by": group_by,
                            "group_token_id": group_token_id,
                            "position_token_id": position_token_id,
                        }
                    )
    return vectors_by_stage, group_rows


def _project(vectors: torch.Tensor, basis: torch.Tensor | None) -> torch.Tensor:
    if basis is None:
        return vectors.float()
    return vectors.float().matmul(basis.float())


def _compute_separability_metrics(
    *,
    vectors_by_token: dict[int, list[torch.Tensor]],
    basis: torch.Tensor | None,
    vocab: Vocabulary,
    label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(vectors_by_token) < 2:
        raise RuntimeError(f"{label} requires at least two groups, got {len(vectors_by_token)}.")
    projected_by_token: dict[int, torch.Tensor] = {}
    for token_id, vectors in sorted(vectors_by_token.items()):
        if len(vectors) < 2:
            raise RuntimeError(f"{label} group {vocab.tokens[token_id]} has fewer than two vectors: {len(vectors)}.")
        projected_by_token[token_id] = _project(torch.stack(vectors, dim=0), basis)
    token_ids = sorted(projected_by_token)
    all_projected = torch.cat([projected_by_token[token_id] for token_id in token_ids], dim=0)
    global_centroid = all_projected.mean(dim=0)

    total_count = 0
    within_sum = 0.0
    between_sum = 0.0
    centroid_norm_sum = 0.0
    group_rows: list[dict[str, Any]] = []
    centroids: dict[int, torch.Tensor] = {}
    for token_id in token_ids:
        values = projected_by_token[token_id]
        centroid = values.mean(dim=0)
        centroids[token_id] = centroid
        count = int(values.size(0))
        centered = values - centroid
        within = float(centered.pow(2).sum(dim=1).mean().item())
        between = float((centroid - global_centroid).pow(2).sum().item())
        within_sum += within * count
        between_sum += between * count
        centroid_norm_sum += float(centroid.norm().item()) * count
        total_count += count
        group_rows.append(
            {
                "group_token_id": token_id,
                "group_token": vocab.tokens[token_id],
                "group_count": count,
                "within_variance": within,
                "centroid_global_squared_distance": between,
                "centroid_norm": float(centroid.norm().item()),
            }
        )
    within_mean = within_sum / total_count
    between_mean = between_sum / total_count
    if within_mean <= 0.0:
        raise RuntimeError(f"{label} has non-positive within-class variance: {within_mean}.")
    separation_ratio = between_mean / within_mean

    correct = 0
    examples = 0
    margin_sum = 0.0
    for token_id in token_ids:
        values = projected_by_token[token_id]
        count = int(values.size(0))
        for row_index in range(count):
            value = values[row_index]
            candidate_distances: list[tuple[int, float]] = []
            for candidate_id in token_ids:
                if candidate_id == token_id:
                    own_centroid = (values.sum(dim=0) - value) / (count - 1)
                    centroid = own_centroid
                else:
                    centroid = centroids[candidate_id]
                distance = float((value - centroid).pow(2).sum().item())
                candidate_distances.append((candidate_id, distance))
            candidate_distances.sort(key=lambda item: (item[1], item[0]))
            predicted_id = candidate_distances[0][0]
            correct_distance = next(distance for candidate_id, distance in candidate_distances if candidate_id == token_id)
            nearest_wrong_distance = next(distance for candidate_id, distance in candidate_distances if candidate_id != token_id)
            margin_sum += nearest_wrong_distance - correct_distance
            correct += int(predicted_id == token_id)
            examples += 1
    nearest_centroid_accuracy = correct / examples
    nearest_centroid_margin = margin_sum / examples
    metrics = {
        "num_groups": len(token_ids),
        "num_vectors": total_count,
        "min_group_count": min(len(projected_by_token[token_id]) for token_id in token_ids),
        "max_group_count": max(len(projected_by_token[token_id]) for token_id in token_ids),
        "projected_dim": int(all_projected.size(1)),
        "within_variance": within_mean,
        "between_variance": between_mean,
        "separation_ratio": separation_ratio,
        "nearest_centroid_accuracy": nearest_centroid_accuracy,
        "nearest_centroid_margin": nearest_centroid_margin,
        "mean_centroid_norm": centroid_norm_sum / total_count,
    }
    return metrics, group_rows


def _summarize_metric_rows(
    *,
    metric_rows: list[dict[str, Any]],
    include_margin: bool,
    window_start: int | None,
    window_end: int | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["context_stage"]), str(row["projection"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (stage, projection), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["checkpoint_step"]))
        separation_values = [float(row["separation_ratio"]) for row in rows]
        accuracy_values = [float(row["nearest_centroid_accuracy"]) for row in rows]
        singular_values = [float(row["qk_singular_value_top"]) for row in rows]
        summary = {
            "context_stage": stage,
            "projection": projection,
            "num_checkpoints": len(rows),
            "start_step": int(rows[0]["checkpoint_step"]),
            "end_step": int(rows[-1]["checkpoint_step"]),
            "start_separation_ratio": separation_values[0],
            "end_separation_ratio": separation_values[-1],
            "delta_separation_ratio": separation_values[-1] - separation_values[0],
            "start_nearest_centroid_accuracy": accuracy_values[0],
            "end_nearest_centroid_accuracy": accuracy_values[-1],
            "delta_nearest_centroid_accuracy": accuracy_values[-1] - accuracy_values[0],
            "separation_ratio_vs_qk_singular_value_top": _pearson(separation_values, singular_values),
            "nearest_centroid_accuracy_vs_qk_singular_value_top": _pearson(accuracy_values, singular_values),
        }
        if include_margin:
            margins = [float(row["answer_margin"]) for row in rows]
            summary["separation_ratio_vs_answer_margin"] = _pearson(separation_values, margins)
            summary["nearest_centroid_accuracy_vs_answer_margin"] = _pearson(accuracy_values, margins)
        if window_start is not None and window_end is not None:
            window_rows = [
                row
                for row in rows
                if window_start <= int(row["checkpoint_step"]) <= window_end
            ]
            if len(window_rows) >= 2:
                window_sep = [float(row["separation_ratio"]) for row in window_rows]
                window_acc = [float(row["nearest_centroid_accuracy"]) for row in window_rows]
                window_singular = [float(row["qk_singular_value_top"]) for row in window_rows]
                summary["window_start"] = int(window_rows[0]["checkpoint_step"])
                summary["window_end"] = int(window_rows[-1]["checkpoint_step"])
                summary["window_start_separation_ratio"] = window_sep[0]
                summary["window_end_separation_ratio"] = window_sep[-1]
                summary["window_delta_separation_ratio"] = window_sep[-1] - window_sep[0]
                summary["window_start_nearest_centroid_accuracy"] = window_acc[0]
                summary["window_end_nearest_centroid_accuracy"] = window_acc[-1]
                summary["window_delta_nearest_centroid_accuracy"] = window_acc[-1] - window_acc[0]
                summary["window_separation_ratio_vs_qk_singular_value_top"] = _pearson(window_sep, window_singular)
                summary["window_nearest_centroid_accuracy_vs_qk_singular_value_top"] = _pearson(window_acc, window_singular)
                if include_margin:
                    window_margins = [float(row["answer_margin"]) for row in window_rows]
                    summary["window_separation_ratio_vs_answer_margin"] = _pearson(window_sep, window_margins)
                    summary["window_nearest_centroid_accuracy_vs_answer_margin"] = _pearson(window_acc, window_margins)
        summary_rows.append(summary)
    return summary_rows


def _plot_trajectory(*, metric_rows: list[dict[str, Any]], output_path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["context_stage"]), str(row["projection"]))].append(row)
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(13, 7))
    for (stage, projection), rows in sorted(grouped.items()):
        if projection == "full_residual":
            continue
        rows = sorted(rows, key=lambda row: int(row["checkpoint_step"]))
        steps = [int(row["checkpoint_step"]) for row in rows]
        values = [float(row["separation_ratio"]) for row in rows]
        ax.plot(steps, values, marker="o", linewidth=2, label=f"{stage}/{projection}")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("between-key / within-key variance")
    ax.set_title("Contextual key separability inside L2H1 QK projections")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _write_markdown_report(path: Path, report: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Contextual Key Separability",
        "",
        "This report tests whether contextual residual states become separable by the requested grouping variable after projection into L2H1 QK singular-vector geometry.",
        "",
        "## Scope",
        "",
        f"- head: `L{report['head_layer']}H{report['head']}`",
        f"- context role: `{report['context_role']}`",
        f"- group by: `{report['group_by']}`",
        f"- stages: `{', '.join(report['context_stages'])}`",
        f"- projection rank: `{report['projection_rank']}`",
        f"- checkpoints: `{report['num_checkpoints']}`",
        f"- probe records: `{report['num_probe_records']}`",
        f"- behavior rows loaded: `{report['behavior_rows_loaded']}`",
        "",
        "## Outputs",
        "",
        f"- metric rows JSONL: `{report['metric_rows_path']}`",
        f"- metric rows CSV: `{report['metric_csv_path']}`",
        f"- group rows: `{report['group_rows_path']}`",
        f"- trajectory plot: `{report['trajectory_plot_path']}`",
        "",
        "## Summary",
        "",
        "| stage | projection | start sep | end sep | delta sep | sep vs sv | sep vs margin | start acc | end acc | delta acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| `{stage}` | `{projection}` | {start_sep:.6g} | {end_sep:.6g} | {delta_sep:.6g} | {sep_sv} | {sep_margin} | {start_acc:.6g} | {end_acc:.6g} | {delta_acc:.6g} |".format(
                stage=row["context_stage"],
                projection=row["projection"],
                start_sep=float(row["start_separation_ratio"]),
                end_sep=float(row["end_separation_ratio"]),
                delta_sep=float(row["delta_separation_ratio"]),
                sep_sv="n/a"
                if row["separation_ratio_vs_qk_singular_value_top"] is None
                else f"{float(row['separation_ratio_vs_qk_singular_value_top']):.4f}",
                sep_margin="n/a"
                if row.get("separation_ratio_vs_answer_margin") is None
                else f"{float(row['separation_ratio_vs_answer_margin']):.4f}",
                start_acc=float(row["start_nearest_centroid_accuracy"]),
                end_acc=float(row["end_nearest_centroid_accuracy"]),
                delta_acc=float(row["delta_nearest_centroid_accuracy"]),
            )
        )
    if report["window_start"] is not None and report["window_end"] is not None:
        lines.extend(
            [
                "",
                "## Window Summary",
                "",
                f"Requested window: `{report['window_start']}..{report['window_end']}`.",
                "",
                "| stage | projection | window start | window end | delta sep | sep vs sv | sep vs margin | delta acc |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary_rows:
            if "window_delta_separation_ratio" not in row:
                continue
            lines.append(
                "| `{stage}` | `{projection}` | {start} | {end} | {delta_sep:.6g} | {sep_sv} | {sep_margin} | {delta_acc:.6g} |".format(
                    stage=row["context_stage"],
                    projection=row["projection"],
                    start=int(row["window_start"]),
                    end=int(row["window_end"]),
                    delta_sep=float(row["window_delta_separation_ratio"]),
                    sep_sv="n/a"
                    if row["window_separation_ratio_vs_qk_singular_value_top"] is None
                    else f"{float(row['window_separation_ratio_vs_qk_singular_value_top']):.4f}",
                    sep_margin="n/a"
                    if row.get("window_separation_ratio_vs_answer_margin") is None
                    else f"{float(row['window_separation_ratio_vs_answer_margin']):.4f}",
                    delta_acc=float(row["window_delta_nearest_centroid_accuracy"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "A rising separation ratio shows that grouped contextual states become easier to separate inside the chosen projection. It does not by itself prove SGD selected this geometry; that requires an update-attribution run using this separability scalar.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def run_contextual_key_separability(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    checkpoint_paths: list[Path] | None,
    device_name: str,
    head_layer: int,
    head: int,
    context_stages: list[str],
    context_role: str,
    group_by: str,
    projection_rank: int,
    batch_size: int,
    split_filter: list[str] | None,
    include_full_residual: bool,
    behavior_rows_path: Path | None,
    behavior_split: str,
    behavior_margin_field: str,
    behavior_accuracy_field: str,
    window_start: int | None,
    window_end: int | None,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Path]]:
    if not context_stages:
        raise ValueError("At least one --context-stage is required.")
    if context_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported context_role {context_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if group_by not in CONTEXTUAL_GROUP_BY_OPTIONS:
        raise ValueError(f"Unsupported group_by {group_by!r}; expected one of {CONTEXTUAL_GROUP_BY_OPTIONS}.")
    if window_start is None and window_end is not None:
        raise ValueError("--window-end requires --window-start.")
    if window_start is not None and window_end is None:
        raise ValueError("--window-start requires --window-end.")
    if window_start is not None and window_end is not None and window_start > window_end:
        raise ValueError(f"window_start must be <= window_end, got {window_start}>{window_end}.")
    _prepare_output_dir(output_dir, overwrite=overwrite)
    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if split_filter is not None:
        split_set = set(split_filter)
        probe_records = [record for record in probe_records if str(record["split"]) in split_set]
        if not probe_records:
            raise RuntimeError(f"Split filter {sorted(split_set)} matched no probe records in {probe_set_path}.")
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    behavior_by_step = _load_behavior_by_step(
        behavior_rows_path=behavior_rows_path,
        behavior_split=behavior_split,
        margin_field=behavior_margin_field,
        accuracy_field=behavior_accuracy_field,
    )
    if behavior_rows_path is not None:
        missing_behavior = [path.name for path in checkpoints if _checkpoint_step_from_path(path) not in behavior_by_step]
        if missing_behavior:
            raise RuntimeError(f"Behavior rows are missing checkpoint(s): {missing_behavior}")
    device = require_device(device_name)
    model = build_model(spec.model, vocab_size=len(vocab.tokens), device=device)
    if head_layer < 0 or head_layer >= len(model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model block range 0..{len(model.blocks) - 1}.")
    if head < 0 or head >= int(model.blocks[head_layer].attn.n_heads):
        raise ValueError(f"head {head} outside head range 0..{int(model.blocks[head_layer].attn.n_heads) - 1}.")
    valid_stages = ["embedding"]
    for layer_index in range(len(model.blocks)):
        valid_stages.extend([f"layer_{layer_index}_post_attn", f"layer_{layer_index}_post_mlp"])
    valid_stages.append("final_norm")
    unsupported_stages = sorted(set(context_stages) - set(valid_stages))
    if unsupported_stages:
        raise ValueError(f"Unsupported context stages {unsupported_stages}; expected one of {valid_stages}.")
    loader = _make_probe_loader(probe_records=probe_records, batch_size=batch_size, pad_token_id=vocab.pad_token_id)

    metric_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    print(
        "[contextual-key-separability] "
        f"checkpoints={len(checkpoints)} records={len(probe_records)} head=L{head_layer}H{head} "
        f"stages={context_stages} role={context_role} group_by={group_by} rank={projection_rank} device={device_name}",
        flush=True,
    )
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint_step = _checkpoint_step_from_path(checkpoint_path)
        print(
            f"[contextual-key-separability] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        projection_bases, singular_summary = _projection_bases(
            model=model,
            head_layer=head_layer,
            head=head,
            projection_rank=projection_rank,
            include_full_residual=include_full_residual,
        )
        vectors_by_stage, checkpoint_group_rows = _collect_vectors_by_stage(
            model=model,
            loader=loader,
            device=device,
            context_stages=context_stages,
            context_role=context_role,
            group_by=group_by,
        )
        behavior = behavior_by_step.get(checkpoint_step)
        for raw_group_row in checkpoint_group_rows:
            group_rows.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_step": checkpoint_step,
                    **raw_group_row,
                    "group_token": vocab.tokens[int(raw_group_row["group_token_id"])],
                    "position_token": vocab.tokens[int(raw_group_row["position_token_id"])],
                }
            )
        for stage in context_stages:
            for projection, basis in projection_bases.items():
                metrics, per_group_rows = _compute_separability_metrics(
                    vectors_by_token=dict(vectors_by_stage[stage]),
                    basis=basis,
                    vocab=vocab,
                    label=f"{checkpoint_path.name}/{stage}/{projection}",
                )
                row = {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_step": checkpoint_step,
                    "context_stage": stage,
                    "context_role": context_role,
                    "group_by": group_by,
                    "head_layer": head_layer,
                    "head": head,
                    "projection": projection,
                    "projection_rank": projection_rank,
                    **singular_summary,
                    **metrics,
                    "answer_margin": None if behavior is None else behavior["answer_margin"],
                    "answer_accuracy": None if behavior is None else behavior.get("answer_accuracy"),
                }
                metric_rows.append(row)
                for group_row in per_group_rows:
                    group_rows.append(
                        {
                            "checkpoint": str(checkpoint_path),
                            "checkpoint_name": checkpoint_path.name,
                            "checkpoint_step": checkpoint_step,
                            "context_stage": stage,
                            "context_role": context_role,
                            "group_by": group_by,
                            "projection": projection,
                            **group_row,
                        }
                    )
        print(
            "[contextual-key-separability] finished "
            f"step={checkpoint_step} sv1={singular_summary['qk_singular_value_top']:.6g}",
            flush=True,
        )

    include_margin = behavior_rows_path is not None
    summary_rows = _summarize_metric_rows(
        metric_rows=metric_rows,
        include_margin=include_margin,
        window_start=window_start,
        window_end=window_end,
    )
    report_path = output_dir / "contextual_key_separability_report.json"
    markdown_path = output_dir / "contextual_key_separability_report.md"
    metric_rows_path = output_dir / "contextual_key_separability_rows.jsonl"
    metric_csv_path = output_dir / "contextual_key_separability_rows.csv"
    group_rows_path = output_dir / "contextual_key_separability_group_rows.jsonl"
    trajectory_plot_path = output_dir / "contextual_key_separability_trajectory.svg"

    write_jsonl(metric_rows_path, metric_rows)
    _write_csv(
        metric_csv_path,
        metric_rows,
        fieldnames=[
            "checkpoint",
            "checkpoint_name",
            "checkpoint_step",
            "context_stage",
            "context_role",
            "group_by",
            "head_layer",
            "head",
            "projection",
            "projection_rank",
            "qk_singular_value_top",
            "qk_singular_value_sum",
            "num_groups",
            "num_vectors",
            "min_group_count",
            "max_group_count",
            "projected_dim",
            "within_variance",
            "between_variance",
            "separation_ratio",
            "nearest_centroid_accuracy",
            "nearest_centroid_margin",
            "mean_centroid_norm",
            "answer_margin",
            "answer_accuracy",
        ],
    )
    write_jsonl(group_rows_path, group_rows)
    _plot_trajectory(metric_rows=metric_rows, output_path=trajectory_plot_path)
    report = {
        "schema_version": CONTEXTUAL_KEY_SEPARABILITY_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "probe_metadata": probe_metadata,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "output_dir": str(output_dir),
        "device": device_name,
        "head_layer": head_layer,
        "head": head,
        "context_stages": context_stages,
        "context_role": context_role,
        "group_by": group_by,
        "projection_rank": projection_rank,
        "include_full_residual": include_full_residual,
        "batch_size": batch_size,
        "split_filter": split_filter,
        "window_start": window_start,
        "window_end": window_end,
        "num_probe_records": len(probe_records),
        "num_checkpoints": len(checkpoints),
        "num_metric_rows": len(metric_rows),
        "num_group_rows": len(group_rows),
        "behavior_rows_loaded": behavior_rows_path is not None,
        "behavior_rows_path": None if behavior_rows_path is None else str(behavior_rows_path),
        "behavior_split": behavior_split,
        "behavior_margin_field": behavior_margin_field,
        "behavior_accuracy_field": behavior_accuracy_field,
        "matrix_convention": "W_QK = q_rows.T @ k_rows; qk_left uses left singular vectors, qk_right uses right singular vectors, qk_both uses the span of both.",
        "metric_definitions": {
            "between_variance": "weighted mean squared distance from each group centroid to the global centroid",
            "within_variance": "weighted mean squared distance from vectors to their group centroid",
            "separation_ratio": "between_variance / within_variance",
            "nearest_centroid_accuracy": "leave-one-out nearest-centroid classification accuracy",
            "nearest_centroid_margin": "mean nearest-wrong-centroid distance minus correct-centroid distance",
        },
        "summary_rows": summary_rows,
        "metric_rows_path": str(metric_rows_path),
        "metric_csv_path": str(metric_csv_path),
        "group_rows_path": str(group_rows_path),
        "trajectory_plot_path": str(trajectory_plot_path),
    }
    write_json(report_path, report)
    _write_markdown_report(path=markdown_path, report=report, summary_rows=summary_rows)
    print(f"[contextual-key-separability] complete report={report_path} rows={metric_rows_path}", flush=True)
    return (
        report_path,
        markdown_path,
        metric_rows_path,
        metric_csv_path,
        group_rows_path,
        {"trajectory": trajectory_plot_path},
    )
