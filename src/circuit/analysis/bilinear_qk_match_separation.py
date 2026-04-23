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
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    GEOMETRY_POSITION_ROLES,
    _attention_key_positions,
    _single_attention_position,
    _value_margin,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


BILINEAR_QK_MATCH_SEPARATION_SCHEMA_VERSION = 1
LAYER_NORM_MODES = ["none", "head_ln1"]
SCORE_MODES = ["full_with_bias", "weight_full"]
GROUP_BY_OPTIONS = ["query_key", "support_key", "support_value", "answer_value", "position_token"]


def _checkpoint_step_from_path(path: Path) -> int:
    prefix = "step_"
    stem = path.stem
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
        output_dir / "bilinear_qk_match_separation_report.json",
        output_dir / "bilinear_qk_match_separation_report.md",
        output_dir / "bilinear_qk_match_separation_rows.jsonl",
        output_dir / "bilinear_qk_match_separation_rows.csv",
        output_dir / "bilinear_qk_match_separation_event_rows.jsonl",
        output_dir / "bilinear_qk_match_separation_group_rows.jsonl",
        output_dir / "bilinear_qk_match_separation_trajectory.svg",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing bilinear QK match-separation outputs without --overwrite: "
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


def _import_matplotlib() -> Any:
    cache_dir = Path(tempfile.gettempdir()) / "circuit_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _valid_residual_stages(model: torch.nn.Module) -> list[str]:
    stages = ["embedding"]
    for layer_index in range(len(model.blocks)):
        stages.extend([f"layer_{layer_index}_post_attn", f"layer_{layer_index}_post_mlp"])
    stages.append("final_norm")
    return stages


def _head_qk_matrices(
    *,
    model: torch.nn.Module,
    head_layer: int,
    head: int,
    ranks: list[int],
) -> dict[str, Any]:
    block = model.blocks[head_layer]
    head_dim = int(block.attn.head_dim)
    head_slice = slice(head * head_dim, (head + 1) * head_dim)
    q_rows = block.attn.q_proj.weight.detach().float()[head_slice, :]
    k_rows = block.attn.k_proj.weight.detach().float()[head_slice, :]
    q_bias = block.attn.q_proj.bias.detach().float()[head_slice]
    k_bias = block.attn.k_proj.bias.detach().float()[head_slice]
    qk_matrix = q_rows.T.matmul(k_rows)
    u, singular_values, vh = torch.linalg.svd(qk_matrix.cpu(), full_matrices=False)
    max_rank = int(singular_values.numel())
    unsupported = [rank for rank in ranks if rank <= 0 or rank > max_rank]
    if unsupported:
        raise ValueError(f"Unsupported rank(s) {unsupported}; expected integer rank in 1..{max_rank}.")
    rank_matrices: dict[int, torch.Tensor] = {}
    for rank in ranks:
        rank_matrices[rank] = ((u[:, :rank] * singular_values[:rank]).matmul(vh[:rank, :])).to(qk_matrix.device)
    return {
        "q_rows": q_rows,
        "k_rows": k_rows,
        "q_bias": q_bias,
        "k_bias": k_bias,
        "qk_matrix": qk_matrix,
        "rank_matrices": rank_matrices,
        "head_dim": head_dim,
        "singular_values": singular_values,
    }


def _score_vectors(
    *,
    query_vector: torch.Tensor,
    key_vectors: torch.Tensor,
    qk_payload: dict[str, Any],
    projection: str,
    scale: float,
) -> torch.Tensor:
    if projection == "full_with_bias":
        q = torch.matmul(query_vector, qk_payload["q_rows"].T) + qk_payload["q_bias"]
        k = torch.matmul(key_vectors, qk_payload["k_rows"].T) + qk_payload["k_bias"]
        return torch.matmul(k, q) / scale
    if projection == "weight_full":
        return torch.matmul(torch.matmul(query_vector, qk_payload["qk_matrix"]), key_vectors.T) / scale
    rank_prefix = "rank_"
    if projection.startswith(rank_prefix):
        rank = int(projection[len(rank_prefix) :])
        if rank not in qk_payload["rank_matrices"]:
            raise KeyError(f"Projection {projection!r} requested rank {rank}, but no rank matrix was built.")
        return torch.matmul(torch.matmul(query_vector, qk_payload["rank_matrices"][rank]), key_vectors.T) / scale
    raise ValueError(f"Unsupported projection {projection!r}.")


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
    if group_by == "position_token":
        return int(batch["input_ids"][context_batch_row, context_position].item())
    if group_by == "query_key":
        return int(batch["input_ids"][query_batch_row, int(metadata["query_key_positions"][flat_index].item())].item())
    if group_by == "support_value":
        return int(batch["input_ids"][query_batch_row, int(metadata["support_value_positions"][flat_index].item())].item())
    if group_by == "answer_value":
        return int(answer_targets[flat_index].item())
    if group_by == "support_key":
        support_batch_row, support_positions = _attention_key_positions(
            batch=batch,
            metadata=metadata,
            flat_index=flat_index,
            position_role="support_key",
            max_position=int(metadata["prediction_positions"][flat_index].item()),
        )
        if support_batch_row != query_batch_row or len(support_positions) != 1:
            raise RuntimeError(
                f"Expected exactly one support_key position for group_by=support_key, "
                f"got row={support_batch_row} positions={support_positions}."
            )
        return int(batch["input_ids"][support_batch_row, support_positions[0]].item())
    raise ValueError(f"Unhandled group_by mode: {group_by}")


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise RuntimeError(f"Cannot compute mean for empty values: {label}")
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) * (value - mean) for value in values) / len(values))


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise RuntimeError(f"Pearson inputs have different lengths: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    norm_x = math.sqrt(sum(value * value for value in centered_x))
    norm_y = math.sqrt(sum(value * value for value in centered_y))
    if norm_x <= 0.0 or norm_y <= 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / (norm_x * norm_y)


def _aggregate_rows(
    *,
    event_rows: list[dict[str, Any]],
    singular_by_step: dict[int, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not event_rows:
        raise RuntimeError("Cannot aggregate empty bilinear QK event rows.")
    metric_groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    semantic_groups: dict[tuple[int, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        metric_groups[(int(row["checkpoint_step"]), str(row["context_stage"]), str(row["projection"]))].append(row)
        semantic_groups[
            (
                int(row["checkpoint_step"]),
                str(row["context_stage"]),
                str(row["projection"]),
                int(row["group_token_id"]),
            )
        ].append(row)

    metric_rows: list[dict[str, Any]] = []
    for (step, stage, projection), rows in sorted(metric_groups.items()):
        separations = [float(row["qk_match_separation"]) for row in rows]
        max_margins = [float(row["qk_match_max_margin"]) for row in rows]
        support_scores = [float(row["support_score_mean"]) for row in rows]
        distractor_scores = [float(row["distractor_score_mean"]) for row in rows]
        answer_margins = [float(row["answer_margin"]) for row in rows]
        answer_correct = [bool(row["answer_correct"]) for row in rows]
        support_beats_all = [bool(row["support_beats_all_distractors"]) for row in rows]
        singular_summary = singular_by_step[step]
        metric_rows.append(
            {
                "checkpoint_step": step,
                "checkpoint": rows[0]["checkpoint"],
                "checkpoint_name": rows[0]["checkpoint_name"],
                "context_stage": stage,
                "projection": projection,
                "head_layer": int(rows[0]["head_layer"]),
                "head": int(rows[0]["head"]),
                "score_query_role": rows[0]["score_query_role"],
                "support_role": rows[0]["support_role"],
                "distractor_role": rows[0]["distractor_role"],
                "layernorm_mode": rows[0]["layernorm_mode"],
                "group_by": rows[0]["group_by"],
                "num_events": len(rows),
                "qk_match_separation_mean": _mean(separations, label="qk_match_separation"),
                "qk_match_separation_std": _std(separations),
                "qk_match_max_margin_mean": _mean(max_margins, label="qk_match_max_margin"),
                "qk_match_max_margin_std": _std(max_margins),
                "support_score_mean": _mean(support_scores, label="support_score"),
                "distractor_score_mean": _mean(distractor_scores, label="distractor_score"),
                "support_beats_all_rate": sum(1 for value in support_beats_all if value) / len(support_beats_all),
                "answer_margin_mean": _mean(answer_margins, label="answer_margin"),
                "answer_accuracy": sum(1 for value in answer_correct if value) / len(answer_correct),
                **singular_summary,
            }
        )

    group_rows: list[dict[str, Any]] = []
    for (step, stage, projection, group_token_id), rows in sorted(semantic_groups.items()):
        separations = [float(row["qk_match_separation"]) for row in rows]
        max_margins = [float(row["qk_match_max_margin"]) for row in rows]
        group_rows.append(
            {
                "checkpoint_step": step,
                "checkpoint": rows[0]["checkpoint"],
                "checkpoint_name": rows[0]["checkpoint_name"],
                "context_stage": stage,
                "projection": projection,
                "group_by": rows[0]["group_by"],
                "group_token_id": group_token_id,
                "group_token": rows[0]["group_token"],
                "num_events": len(rows),
                "qk_match_separation_mean": _mean(separations, label="group_qk_match_separation"),
                "qk_match_separation_std": _std(separations),
                "qk_match_max_margin_mean": _mean(max_margins, label="group_qk_match_max_margin"),
                "support_beats_all_rate": sum(1 for row in rows if bool(row["support_beats_all_distractors"]))
                / len(rows),
            }
        )
    return metric_rows, group_rows


def _summarize_metric_rows(
    *,
    metric_rows: list[dict[str, Any]],
    window_start: int | None,
    window_end: int | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["context_stage"]), str(row["projection"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (stage, projection), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["checkpoint_step"]))
        separations = [float(row["qk_match_separation_mean"]) for row in rows]
        max_margins = [float(row["qk_match_max_margin_mean"]) for row in rows]
        singular_values = [float(row["qk_singular_value_top"]) for row in rows]
        answer_margins = [float(row["answer_margin_mean"]) for row in rows]
        support_rates = [float(row["support_beats_all_rate"]) for row in rows]
        summary: dict[str, Any] = {
            "context_stage": stage,
            "projection": projection,
            "num_checkpoints": len(rows),
            "start_step": int(rows[0]["checkpoint_step"]),
            "end_step": int(rows[-1]["checkpoint_step"]),
            "start_qk_match_separation": separations[0],
            "end_qk_match_separation": separations[-1],
            "delta_qk_match_separation": separations[-1] - separations[0],
            "start_qk_match_max_margin": max_margins[0],
            "end_qk_match_max_margin": max_margins[-1],
            "delta_qk_match_max_margin": max_margins[-1] - max_margins[0],
            "start_support_beats_all_rate": support_rates[0],
            "end_support_beats_all_rate": support_rates[-1],
            "delta_support_beats_all_rate": support_rates[-1] - support_rates[0],
            "qk_match_separation_vs_qk_singular_value_top": _pearson(separations, singular_values),
            "qk_match_separation_vs_answer_margin": _pearson(separations, answer_margins),
        }
        if window_start is not None and window_end is not None:
            window_rows = [
                row
                for row in rows
                if window_start <= int(row["checkpoint_step"]) <= window_end
            ]
            if len(window_rows) >= 2:
                window_sep = [float(row["qk_match_separation_mean"]) for row in window_rows]
                window_max = [float(row["qk_match_max_margin_mean"]) for row in window_rows]
                window_sv = [float(row["qk_singular_value_top"]) for row in window_rows]
                window_margin = [float(row["answer_margin_mean"]) for row in window_rows]
                window_rates = [float(row["support_beats_all_rate"]) for row in window_rows]
                summary["window_start"] = int(window_rows[0]["checkpoint_step"])
                summary["window_end"] = int(window_rows[-1]["checkpoint_step"])
                summary["window_start_qk_match_separation"] = window_sep[0]
                summary["window_end_qk_match_separation"] = window_sep[-1]
                summary["window_delta_qk_match_separation"] = window_sep[-1] - window_sep[0]
                summary["window_start_qk_match_max_margin"] = window_max[0]
                summary["window_end_qk_match_max_margin"] = window_max[-1]
                summary["window_delta_qk_match_max_margin"] = window_max[-1] - window_max[0]
                summary["window_start_support_beats_all_rate"] = window_rates[0]
                summary["window_end_support_beats_all_rate"] = window_rates[-1]
                summary["window_delta_support_beats_all_rate"] = window_rates[-1] - window_rates[0]
                summary["window_qk_match_separation_vs_qk_singular_value_top"] = _pearson(window_sep, window_sv)
                summary["window_qk_match_separation_vs_answer_margin"] = _pearson(window_sep, window_margin)
        summary_rows.append(summary)
    return summary_rows


def _plot_trajectory(*, metric_rows: list[dict[str, Any]], output_path: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[(str(row["context_stage"]), str(row["projection"]))].append(row)
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(13, 7))
    for (stage, projection), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["checkpoint_step"]))
        steps = [int(row["checkpoint_step"]) for row in rows]
        values = [float(row["qk_match_separation_mean"]) for row in rows]
        ax.plot(steps, values, marker="o", linewidth=2, label=f"{stage}/{projection}")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("support score - distractor score")
    ax.set_title("Bilinear QK match separation")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _write_markdown_report(path: Path, report: dict[str, Any], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Bilinear QK Match Separation",
        "",
        "This report tests whether a head's QK bilinear form scores the correct support role above distractor roles.",
        "",
        "## Scope",
        "",
        f"- head: `L{report['head_layer']}H{report['head']}`",
        f"- query role: `{report['score_query_role']}`",
        f"- support role: `{report['support_role']}`",
        f"- distractor role: `{report['distractor_role']}`",
        f"- context stages: `{', '.join(report['context_stages'])}`",
        f"- layernorm mode: `{report['layernorm_mode']}`",
        f"- score modes: `{', '.join(report['score_modes'])}`",
        f"- ranks: `{report['ranks']}`",
        f"- group by: `{report['group_by']}`",
        f"- checkpoints: `{report['num_checkpoints']}`",
        f"- probe records: `{report['num_probe_records']}`",
        "",
        "## Outputs",
        "",
        f"- metric rows JSONL: `{report['metric_rows_path']}`",
        f"- metric rows CSV: `{report['metric_csv_path']}`",
        f"- event rows: `{report['event_rows_path']}`",
        f"- group rows: `{report['group_rows_path']}`",
        f"- trajectory plot: `{report['trajectory_plot_path']}`",
        "",
        "## Summary",
        "",
        "| stage | projection | start sep | end sep | delta sep | sep vs sv | sep vs margin | start win | end win | delta win |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| `{stage}` | `{projection}` | {start_sep:.6g} | {end_sep:.6g} | {delta_sep:.6g} | {sep_sv} | {sep_margin} | {start_win:.6g} | {end_win:.6g} | {delta_win:.6g} |".format(
                stage=row["context_stage"],
                projection=row["projection"],
                start_sep=float(row["start_qk_match_separation"]),
                end_sep=float(row["end_qk_match_separation"]),
                delta_sep=float(row["delta_qk_match_separation"]),
                sep_sv="n/a"
                if row["qk_match_separation_vs_qk_singular_value_top"] is None
                else f"{float(row['qk_match_separation_vs_qk_singular_value_top']):.4f}",
                sep_margin="n/a"
                if row["qk_match_separation_vs_answer_margin"] is None
                else f"{float(row['qk_match_separation_vs_answer_margin']):.4f}",
                start_win=float(row["start_support_beats_all_rate"]),
                end_win=float(row["end_support_beats_all_rate"]),
                delta_win=float(row["delta_support_beats_all_rate"]),
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
                "| stage | projection | window start | window end | delta sep | sep vs sv | sep vs margin | delta win |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary_rows:
            if "window_delta_qk_match_separation" not in row:
                continue
            lines.append(
                "| `{stage}` | `{projection}` | {start} | {end} | {delta_sep:.6g} | {sep_sv} | {sep_margin} | {delta_win:.6g} |".format(
                    stage=row["context_stage"],
                    projection=row["projection"],
                    start=int(row["window_start"]),
                    end=int(row["window_end"]),
                    delta_sep=float(row["window_delta_qk_match_separation"]),
                    sep_sv="n/a"
                    if row["window_qk_match_separation_vs_qk_singular_value_top"] is None
                    else f"{float(row['window_qk_match_separation_vs_qk_singular_value_top']):.4f}",
                    sep_margin="n/a"
                    if row["window_qk_match_separation_vs_answer_margin"] is None
                    else f"{float(row['window_qk_match_separation_vs_answer_margin']):.4f}",
                    delta_win=float(row["window_delta_support_beats_all_rate"]),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "A rising match separation means the bilinear QK form increasingly scores the selected support role above distractors. It does not by itself prove SGD caused that geometry; update attribution should use this scalar only if this semantic test passes.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bilinear_qk_match_separation(
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
    score_query_role: str,
    support_role: str,
    distractor_role: str,
    layernorm_mode: str,
    score_modes: list[str],
    ranks: list[int],
    group_by: str,
    batch_size: int,
    split_filter: list[str] | None,
    window_start: int | None,
    window_end: int | None,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if not context_stages:
        raise ValueError("At least one --context-stage is required.")
    unsupported_roles = [
        role for role in [score_query_role, support_role, distractor_role] if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported role(s) {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if support_role == distractor_role:
        raise ValueError("support_role and distractor_role must be different.")
    if layernorm_mode not in LAYER_NORM_MODES:
        raise ValueError(f"Unsupported layernorm_mode {layernorm_mode!r}; expected one of {LAYER_NORM_MODES}.")
    if not score_modes and not ranks:
        raise ValueError("At least one --score-mode or --rank must be provided.")
    unsupported_score_modes = [mode for mode in score_modes if mode not in SCORE_MODES]
    if unsupported_score_modes:
        raise ValueError(f"Unsupported score mode(s) {unsupported_score_modes}; expected one of {SCORE_MODES}.")
    if group_by not in GROUP_BY_OPTIONS:
        raise ValueError(f"Unsupported group_by {group_by!r}; expected one of {GROUP_BY_OPTIONS}.")
    if window_start is None and window_end is not None:
        raise ValueError("--window-end requires --window-start.")
    if window_start is not None and window_end is None:
        raise ValueError("--window-start requires --window-end.")
    if window_start is not None and window_end is not None and window_start > window_end:
        raise ValueError(f"window_start must be <= window_end, got {window_start}>{window_end}.")
    _prepare_output_dir(output_dir, overwrite=overwrite)

    spec = TrainSpec.from_path(config_path)
    benchmark_metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(benchmark_metadata["vocabulary"])
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    if split_filter is not None:
        split_set = set(split_filter)
        probe_records = [record for record in probe_records if str(record["split"]) in split_set]
        if not probe_records:
            raise RuntimeError(f"Split filter {sorted(split_set)} matched no probe records in {probe_set_path}.")
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    device = require_device(device_name)
    model = build_model(spec.model, vocab_size=len(vocab.tokens), device=device)
    if head_layer < 0 or head_layer >= len(model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model block range 0..{len(model.blocks) - 1}.")
    if head < 0 or head >= int(model.blocks[head_layer].attn.n_heads):
        raise ValueError(f"head {head} outside head range 0..{int(model.blocks[head_layer].attn.n_heads) - 1}.")
    valid_stages = _valid_residual_stages(model)
    unsupported_stages = sorted(set(context_stages) - set(valid_stages))
    if unsupported_stages:
        raise ValueError(f"Unsupported context stages {unsupported_stages}; expected one of {valid_stages}.")

    deduped_score_modes = sorted(set(score_modes), key=score_modes.index)
    deduped_ranks = sorted(set(ranks))
    projections = deduped_score_modes + [f"rank_{rank}" for rank in deduped_ranks]
    loader = _make_probe_loader(probe_records=probe_records, batch_size=batch_size, pad_token_id=vocab.pad_token_id)
    value_token_ids = torch.tensor(vocab.value_token_ids, device=device, dtype=torch.long)

    event_rows: list[dict[str, Any]] = []
    singular_by_step: dict[int, dict[str, float]] = {}
    print(
        "[bilinear-qk-match-separation] "
        f"checkpoints={len(checkpoints)} records={len(probe_records)} head=L{head_layer}H{head} "
        f"stages={context_stages} query={score_query_role} support={support_role} "
        f"distractor={distractor_role} projections={projections} layernorm={layernorm_mode} device={device_name}",
        flush=True,
    )
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint_step = _checkpoint_step_from_path(checkpoint_path)
        print(
            f"[bilinear-qk-match-separation] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        checkpoint = load_checkpoint(checkpoint_path, device)
        payload_step = int(checkpoint["step"])
        if payload_step != checkpoint_step:
            raise RuntimeError(f"Checkpoint step mismatch: payload={payload_step} path={checkpoint_step}")
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        qk_payload = _head_qk_matrices(model=model, head_layer=head_layer, head=head, ranks=deduped_ranks)
        singular_values = qk_payload["singular_values"]
        singular_value_sum = singular_values.sum()
        if float(singular_value_sum.item()) <= 0.0:
            raise RuntimeError(f"Non-positive QK singular-value sum at checkpoint {checkpoint_path}: {singular_value_sum}")
        singular_by_step[checkpoint_step] = {
            "qk_singular_value_top": float(singular_values[0].item()),
            "qk_singular_value_sum": float(singular_value_sum.item()),
            "qk_singular_value_top3_fraction": float(
                (singular_values[: min(3, int(singular_values.numel()))].sum() / singular_value_sum).item()
            ),
        }
        block = model.blocks[head_layer]
        scale = math.sqrt(float(qk_payload["head_dim"]))
        for batch_index, raw_batch in enumerate(loader):
            batch = move_batch_to_device(raw_batch, device)
            with torch.no_grad():
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_residual_streams=True,
                )
            if outputs.residual_streams is None:
                raise RuntimeError("Bilinear QK match separation requires residual streams.")
            missing_stages = [stage for stage in context_stages if stage not in outputs.residual_streams]
            if missing_stages:
                raise KeyError(
                    f"Residual stage(s) {missing_stages} not found. Available stages: {sorted(outputs.residual_streams)}"
                )
            answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
            answer_margins = _value_margin(answer_logits, answer_targets, value_token_ids)
            answer_correct = answer_logits.argmax(dim=-1) == answer_targets
            states_by_stage: dict[str, torch.Tensor] = {}
            for stage in context_stages:
                stage_state = outputs.residual_streams[stage].detach().float()
                if layernorm_mode == "head_ln1":
                    stage_state = block.ln_1(stage_state)
                states_by_stage[stage] = stage_state
            for flat_index in range(int(metadata["rows"].numel())):
                batch_row, query_position = _single_attention_position(
                    batch=batch,
                    metadata=metadata,
                    flat_index=flat_index,
                    position_role=score_query_role,
                    label="bilinear QK query",
                )
                support_batch_row, support_positions = _attention_key_positions(
                    batch=batch,
                    metadata=metadata,
                    flat_index=flat_index,
                    position_role=support_role,
                    max_position=query_position,
                )
                distractor_batch_row, distractor_positions = _attention_key_positions(
                    batch=batch,
                    metadata=metadata,
                    flat_index=flat_index,
                    position_role=distractor_role,
                    max_position=query_position,
                )
                if support_batch_row != batch_row:
                    raise RuntimeError(
                        f"Support role {support_role!r} selected row {support_batch_row}, "
                        f"but query role {score_query_role!r} selected row {batch_row}."
                    )
                if distractor_batch_row != batch_row:
                    raise RuntimeError(
                        f"Distractor role {distractor_role!r} selected row {distractor_batch_row}, "
                        f"but query role {score_query_role!r} selected row {batch_row}."
                    )
                group_token_id = _resolve_group_token_id(
                    group_by=group_by,
                    batch=batch,
                    metadata=metadata,
                    answer_targets=answer_targets,
                    flat_index=flat_index,
                    context_batch_row=batch_row,
                    context_position=query_position,
                )
                record = batch["records"][batch_row]
                for stage in context_stages:
                    stage_state = states_by_stage[stage]
                    query_vector = stage_state[batch_row, query_position, :]
                    support_vectors = stage_state[batch_row, support_positions, :]
                    distractor_vectors = stage_state[batch_row, distractor_positions, :]
                    for projection in projections:
                        support_scores = _score_vectors(
                            query_vector=query_vector,
                            key_vectors=support_vectors,
                            qk_payload=qk_payload,
                            projection=projection,
                            scale=scale,
                        )
                        distractor_scores = _score_vectors(
                            query_vector=query_vector,
                            key_vectors=distractor_vectors,
                            qk_payload=qk_payload,
                            projection=projection,
                            scale=scale,
                        )
                        support_mean = support_scores.mean()
                        distractor_mean = distractor_scores.mean()
                        max_distractor = distractor_scores.max()
                        max_margin = support_scores.mean() - max_distractor
                        event_rows.append(
                            {
                                "checkpoint": str(checkpoint_path),
                                "checkpoint_name": checkpoint_path.name,
                                "checkpoint_step": checkpoint_step,
                                "batch_index": batch_index,
                                "sample_id": str(record["sample_id"]),
                                "split": str(record["split"]),
                                "query_index": int(metadata["query_indices"][flat_index].item()),
                                "head_layer": head_layer,
                                "head": head,
                                "context_stage": stage,
                                "layernorm_mode": layernorm_mode,
                                "score_query_role": score_query_role,
                                "support_role": support_role,
                                "distractor_role": distractor_role,
                                "projection": projection,
                                "query_position": int(query_position),
                                "support_positions": [int(position) for position in support_positions],
                                "distractor_positions": [int(position) for position in distractor_positions],
                                "num_support_positions": len(support_positions),
                                "num_distractor_positions": len(distractor_positions),
                                "support_score_mean": float(support_mean.detach().cpu().item()),
                                "support_score_max": float(support_scores.max().detach().cpu().item()),
                                "distractor_score_mean": float(distractor_mean.detach().cpu().item()),
                                "distractor_score_max": float(max_distractor.detach().cpu().item()),
                                "qk_match_separation": float((support_mean - distractor_mean).detach().cpu().item()),
                                "qk_match_max_margin": float(max_margin.detach().cpu().item()),
                                "support_beats_all_distractors": bool((support_scores.max() > max_distractor).detach().cpu().item()),
                                "group_by": group_by,
                                "group_token_id": int(group_token_id),
                                "group_token": vocab.tokens[int(group_token_id)],
                                "answer_token_id": int(answer_targets[flat_index].item()),
                                "answer_token": vocab.tokens[int(answer_targets[flat_index].item())],
                                "answer_margin": float(answer_margins[flat_index].detach().cpu().item()),
                                "answer_correct": bool(answer_correct[flat_index].detach().cpu().item()),
                            }
                        )
        print(
            "[bilinear-qk-match-separation] finished "
            f"step={checkpoint_step} sv1={singular_by_step[checkpoint_step]['qk_singular_value_top']:.6g}",
            flush=True,
        )

    metric_rows, group_rows = _aggregate_rows(event_rows=event_rows, singular_by_step=singular_by_step)
    summary_rows = _summarize_metric_rows(
        metric_rows=metric_rows,
        window_start=window_start,
        window_end=window_end,
    )
    report_path = output_dir / "bilinear_qk_match_separation_report.json"
    markdown_path = output_dir / "bilinear_qk_match_separation_report.md"
    metric_rows_path = output_dir / "bilinear_qk_match_separation_rows.jsonl"
    metric_csv_path = output_dir / "bilinear_qk_match_separation_rows.csv"
    event_rows_path = output_dir / "bilinear_qk_match_separation_event_rows.jsonl"
    group_rows_path = output_dir / "bilinear_qk_match_separation_group_rows.jsonl"
    trajectory_plot_path = output_dir / "bilinear_qk_match_separation_trajectory.svg"
    write_jsonl(metric_rows_path, metric_rows)
    _write_csv(
        metric_csv_path,
        metric_rows,
        fieldnames=[
            "checkpoint_step",
            "checkpoint",
            "checkpoint_name",
            "context_stage",
            "projection",
            "head_layer",
            "head",
            "score_query_role",
            "support_role",
            "distractor_role",
            "layernorm_mode",
            "group_by",
            "num_events",
            "qk_match_separation_mean",
            "qk_match_separation_std",
            "qk_match_max_margin_mean",
            "qk_match_max_margin_std",
            "support_score_mean",
            "distractor_score_mean",
            "support_beats_all_rate",
            "answer_margin_mean",
            "answer_accuracy",
            "qk_singular_value_top",
            "qk_singular_value_sum",
            "qk_singular_value_top3_fraction",
        ],
    )
    write_jsonl(event_rows_path, event_rows)
    write_jsonl(group_rows_path, group_rows)
    _plot_trajectory(metric_rows=metric_rows, output_path=trajectory_plot_path)
    report = {
        "schema_version": BILINEAR_QK_MATCH_SEPARATION_SCHEMA_VERSION,
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
        "score_query_role": score_query_role,
        "support_role": support_role,
        "distractor_role": distractor_role,
        "layernorm_mode": layernorm_mode,
        "score_modes": deduped_score_modes,
        "ranks": deduped_ranks,
        "group_by": group_by,
        "batch_size": batch_size,
        "split_filter": split_filter,
        "window_start": window_start,
        "window_end": window_end,
        "num_probe_records": len(probe_records),
        "num_checkpoints": len(checkpoints),
        "num_event_rows": len(event_rows),
        "num_metric_rows": len(metric_rows),
        "num_group_rows": len(group_rows),
        "calculation": {
            "full_with_bias": "(W_Q x_query + b_Q) dot (W_K x_key + b_K) / sqrt(head_dim)",
            "weight_full": "x_query.T @ W_Q.T @ W_K @ x_key / sqrt(head_dim)",
            "rank_k": "x_query.T @ rank-k-SVD(W_Q.T @ W_K) @ x_key / sqrt(head_dim)",
            "qk_match_separation": "mean support score - mean distractor score",
            "qk_match_max_margin": "mean support score - max distractor score",
            "support_beats_all_rate": "fraction of query events where max support score is strictly greater than max distractor score",
        },
        "metric_rows_path": str(metric_rows_path),
        "metric_csv_path": str(metric_csv_path),
        "event_rows_path": str(event_rows_path),
        "group_rows_path": str(group_rows_path),
        "trajectory_plot_path": str(trajectory_plot_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown_report(path=markdown_path, report=report, summary_rows=summary_rows)
    print(f"[bilinear-qk-match-separation] complete report={report_path} rows={metric_rows_path}", flush=True)
    return (
        report_path,
        markdown_path,
        metric_rows_path,
        metric_csv_path,
        event_rows_path,
        group_rows_path,
        {"trajectory": trajectory_plot_path},
    )
