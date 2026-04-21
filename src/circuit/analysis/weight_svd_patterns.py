from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from circuit.io import iter_jsonl, write_json, write_jsonl


MatrixKey = tuple[int, int | None, str, str]
VectorKey = tuple[int, int | None, str, str, int, str]


def _matrix_label(*, layer: int, head: int | None, matrix_name: str) -> str:
    if head is None:
        return f"L{layer} {matrix_name}"
    return f"L{layer}H{head} {matrix_name}"


def _vector_label(*, key: VectorKey) -> str:
    layer, head, _component_type, matrix_name, rank, side = key
    return f"{_matrix_label(layer=layer, head=head, matrix_name=matrix_name)} r{rank} {side}"


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / "weight_svd_pattern_report.json",
        output_dir / "weight_svd_pattern_report.md",
        output_dir / "weight_svd_matrix_summary.jsonl",
        output_dir / "weight_svd_matrix_summary.csv",
        output_dir / "weight_svd_vector_alignment.jsonl",
        output_dir / "weight_svd_interval_events.jsonl",
        output_dir / "weight_svd_coordination_windows.jsonl",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing weight SVD pattern outputs without --overwrite: "
            f"{[str(path) for path in existing]}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _load_rank1_singular_rows(path: Path) -> dict[MatrixKey, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Singular-values file not found: {path}")
    rows_by_key: dict[MatrixKey, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(path):
        if int(row["singular_value_rank"]) != 1:
            continue
        head = None if row["head"] is None else int(row["head"])
        key = (int(row["layer"]), head, str(row["component_type"]), str(row["matrix_name"]))
        rows_by_key[key].append(
            {
                "checkpoint": str(row["checkpoint"]),
                "checkpoint_name": str(row["checkpoint_name"]),
                "step": int(row["step"]),
                "layer": int(row["layer"]),
                "head": head,
                "component_type": str(row["component_type"]),
                "matrix_name": str(row["matrix_name"]),
                "singular_value": float(row["singular_value"]),
                "effective_rank": float(row["effective_rank"]),
                "spectral_mass_top3": float(row["spectral_mass_top3"]),
                "singular_value_count": int(row["singular_value_count"]),
            }
        )
    if not rows_by_key:
        raise RuntimeError(f"No rank-1 singular value rows found in {path}")
    for key, rows in rows_by_key.items():
        rows.sort(key=lambda item: int(item["step"]))
        steps = [int(row["step"]) for row in rows]
        if len(set(steps)) != len(steps):
            raise RuntimeError(f"Duplicate checkpoint steps for matrix {_matrix_label(layer=key[0], head=key[1], matrix_name=key[3])}: {steps}")
    return dict(rows_by_key)


def _load_vector_rows(path: Path, *, max_rank: int) -> dict[VectorKey, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Top singular vector file not found: {path}")
    if max_rank <= 0:
        raise ValueError("--max-vector-rank must be positive.")
    rows_by_key: dict[VectorKey, list[dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(path):
        rank = int(row["singular_value_rank"])
        if rank > max_rank:
            continue
        head = None if row["head"] is None else int(row["head"])
        vector = [float(value) for value in row["vector"]]
        if not vector:
            raise RuntimeError(f"Empty vector encountered in {path} for row: {row}")
        key = (
            int(row["layer"]),
            head,
            str(row["component_type"]),
            str(row["matrix_name"]),
            rank,
            str(row["vector_side"]),
        )
        rows_by_key[key].append(
            {
                "checkpoint": str(row["checkpoint"]),
                "checkpoint_name": str(row["checkpoint_name"]),
                "step": int(row["step"]),
                "vector_label": str(row["vector_label"]),
                "singular_value": float(row["singular_value"]),
                "vector": vector,
            }
        )
    if not rows_by_key:
        raise RuntimeError(f"No vector rows at rank <= {max_rank} found in {path}")
    for key, rows in rows_by_key.items():
        rows.sort(key=lambda item: int(item["step"]))
        steps = [int(row["step"]) for row in rows]
        if len(set(steps)) != len(steps):
            raise RuntimeError(f"Duplicate checkpoint steps for vector {_vector_label(key=key)}: {steps}")
    return dict(rows_by_key)


def _abs_cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError(f"Cannot compare vectors with different dimensions: {len(left)} vs {len(right)}")
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for left_value, right_value in zip(left, right):
        dot += left_value * right_value
        left_sq += left_value * left_value
        right_sq += right_value * right_value
    if left_sq <= 0.0 or right_sq <= 0.0:
        raise RuntimeError("Cannot compute cosine for a zero vector.")
    return abs(dot / ((left_sq**0.5) * (right_sq**0.5)))


def _first_step_final_aligned(rows: list[dict[str, Any]], *, threshold: float) -> int | None:
    for index, row in enumerate(rows):
        if all(float(later["final_abs_cosine"]) >= threshold for later in rows[index:]):
            return int(row["step"])
    return None


def _first_step_adjacent_stable(rows: list[dict[str, Any]], *, threshold: float, patience: int) -> int | None:
    if patience <= 0:
        raise ValueError("--stability-patience must be positive.")
    for index, row in enumerate(rows):
        if row["adjacent_abs_cosine"] is None:
            continue
        tail = [later for later in rows[index:] if later["adjacent_abs_cosine"] is not None]
        if len(tail) < patience:
            continue
        if all(float(later["adjacent_abs_cosine"]) >= threshold for later in tail):
            return int(row["step"])
    return None


def _build_vector_alignment_rows(
    vector_rows_by_key: dict[VectorKey, list[dict[str, Any]]],
    *,
    final_alignment_threshold: float,
    adjacent_stability_threshold: float,
    stability_patience: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alignment_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for key, rows in vector_rows_by_key.items():
        final_vector = rows[-1]["vector"]
        key_rows: list[dict[str, Any]] = []
        previous_vector: list[float] | None = None
        previous_step: int | None = None
        for row in rows:
            current_vector = row["vector"]
            adjacent_abs_cosine = None if previous_vector is None else _abs_cosine(previous_vector, current_vector)
            alignment_row = {
                "step": int(row["step"]),
                "checkpoint_name": str(row["checkpoint_name"]),
                "layer": int(key[0]),
                "head": key[1],
                "component_type": key[2],
                "matrix_name": key[3],
                "matrix_label": _matrix_label(layer=key[0], head=key[1], matrix_name=key[3]),
                "singular_value_rank": int(key[4]),
                "vector_side": key[5],
                "vector_label": str(row["vector_label"]),
                "previous_step": previous_step,
                "adjacent_abs_cosine": adjacent_abs_cosine,
                "adjacent_rotation": None if adjacent_abs_cosine is None else 1.0 - adjacent_abs_cosine,
                "final_abs_cosine": _abs_cosine(current_vector, final_vector),
                "final_rotation_remaining": 1.0 - _abs_cosine(current_vector, final_vector),
                "singular_value": float(row["singular_value"]),
            }
            key_rows.append(alignment_row)
            previous_vector = current_vector
            previous_step = int(row["step"])
        alignment_rows.extend(key_rows)
        adjacent_rows = [row for row in key_rows if row["adjacent_abs_cosine"] is not None]
        min_adjacent_row = min(adjacent_rows, key=lambda item: float(item["adjacent_abs_cosine"])) if adjacent_rows else None
        summary_rows.append(
            {
                "layer": int(key[0]),
                "head": key[1],
                "component_type": key[2],
                "matrix_name": key[3],
                "matrix_label": _matrix_label(layer=key[0], head=key[1], matrix_name=key[3]),
                "singular_value_rank": int(key[4]),
                "vector_side": key[5],
                "vector_label": str(rows[0]["vector_label"]),
                "start_step": int(rows[0]["step"]),
                "end_step": int(rows[-1]["step"]),
                "start_final_abs_cosine": float(key_rows[0]["final_abs_cosine"]),
                "end_final_abs_cosine": float(key_rows[-1]["final_abs_cosine"]),
                "first_final_alignment_step": _first_step_final_aligned(key_rows, threshold=final_alignment_threshold),
                "first_adjacent_stable_step": _first_step_adjacent_stable(
                    key_rows,
                    threshold=adjacent_stability_threshold,
                    patience=stability_patience,
                ),
                "min_adjacent_abs_cosine": None if min_adjacent_row is None else float(min_adjacent_row["adjacent_abs_cosine"]),
                "min_adjacent_interval_start": None if min_adjacent_row is None else int(min_adjacent_row["previous_step"]),
                "min_adjacent_interval_end": None if min_adjacent_row is None else int(min_adjacent_row["step"]),
                "total_rotation_to_final": 1.0 - float(key_rows[0]["final_abs_cosine"]),
                "max_final_abs_cosine_before_end": max(float(row["final_abs_cosine"]) for row in key_rows[:-1]) if len(key_rows) > 1 else float(key_rows[-1]["final_abs_cosine"]),
            }
        )
    return alignment_rows, summary_rows


def _build_matrix_summary_rows(rows_by_key: dict[MatrixKey, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for key, rows in rows_by_key.items():
        start = rows[0]
        end = rows[-1]
        sv_start = float(start["singular_value"])
        sv_end = float(end["singular_value"])
        if sv_start == 0.0:
            relative_growth = None
        else:
            relative_growth = (sv_end - sv_start) / abs(sv_start)
        effective_rank_delta = float(end["effective_rank"]) - float(start["effective_rank"])
        spectral_mass_top3_delta = float(end["spectral_mass_top3"]) - float(start["spectral_mass_top3"])
        summary_rows.append(
            {
                "layer": int(key[0]),
                "head": key[1],
                "component_type": key[2],
                "matrix_name": key[3],
                "matrix_label": _matrix_label(layer=key[0], head=key[1], matrix_name=key[3]),
                "start_step": int(start["step"]),
                "end_step": int(end["step"]),
                "start_singular_value_rank1": sv_start,
                "end_singular_value_rank1": sv_end,
                "singular_value_rank1_delta": sv_end - sv_start,
                "singular_value_rank1_relative_delta": relative_growth,
                "start_effective_rank": float(start["effective_rank"]),
                "end_effective_rank": float(end["effective_rank"]),
                "effective_rank_delta": effective_rank_delta,
                "start_spectral_mass_top3": float(start["spectral_mass_top3"]),
                "end_spectral_mass_top3": float(end["spectral_mass_top3"]),
                "spectral_mass_top3_delta": spectral_mass_top3_delta,
                "selective_concentration_score": (sv_end - sv_start) * max(0.0, spectral_mass_top3_delta),
                "singular_value_count": int(end["singular_value_count"]),
            }
        )
    summary_rows.sort(key=lambda row: float(row["singular_value_rank1_delta"]), reverse=True)
    return summary_rows


def _build_interval_event_rows(
    rows_by_key: dict[MatrixKey, list[dict[str, Any]]],
    vector_alignment_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rank1_rotation_by_interval: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in vector_alignment_rows:
        if int(row["singular_value_rank"]) != 1:
            continue
        if row["previous_step"] is None:
            continue
        interval_key = (str(row["matrix_label"]), int(row["previous_step"]), int(row["step"]))
        rank1_rotation_by_interval[interval_key].append(float(row["adjacent_rotation"]))

    interval_rows: list[dict[str, Any]] = []
    for key, rows in rows_by_key.items():
        label = _matrix_label(layer=key[0], head=key[1], matrix_name=key[3])
        for previous, current in zip(rows, rows[1:]):
            previous_sv = float(previous["singular_value"])
            current_sv = float(current["singular_value"])
            sv_delta = current_sv - previous_sv
            rotations = rank1_rotation_by_interval.get((label, int(previous["step"]), int(current["step"])), [])
            rank1_mean_rotation = None if not rotations else sum(rotations) / len(rotations)
            interval_rows.append(
                {
                    "interval_start_step": int(previous["step"]),
                    "interval_end_step": int(current["step"]),
                    "layer": int(key[0]),
                    "head": key[1],
                    "component_type": key[2],
                    "matrix_name": key[3],
                    "matrix_label": label,
                    "singular_value_rank1_start": previous_sv,
                    "singular_value_rank1_end": current_sv,
                    "singular_value_rank1_delta": sv_delta,
                    "singular_value_rank1_relative_delta": None if previous_sv == 0.0 else sv_delta / abs(previous_sv),
                    "effective_rank_delta": float(current["effective_rank"]) - float(previous["effective_rank"]),
                    "spectral_mass_top3_delta": float(current["spectral_mass_top3"]) - float(previous["spectral_mass_top3"]),
                    "rank1_mean_adjacent_rotation": rank1_mean_rotation,
                    "rank1_mean_adjacent_abs_cosine": None if rank1_mean_rotation is None else 1.0 - rank1_mean_rotation,
                    "positive_concentration_delta": sv_delta * max(0.0, float(current["spectral_mass_top3"]) - float(previous["spectral_mass_top3"])),
                }
            )

    rows_by_interval: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in interval_rows:
        rows_by_interval[(int(row["interval_start_step"]), int(row["interval_end_step"]))].append(row)
    coordination_rows: list[dict[str, Any]] = []
    for (start_step, end_step), rows in rows_by_interval.items():
        positive_growth_rows = [row for row in rows if float(row["singular_value_rank1_delta"]) > 0.0]
        concentration_rows = [row for row in rows if float(row["spectral_mass_top3_delta"]) > 0.0]
        rotation_rows = [row for row in rows if row["rank1_mean_adjacent_rotation"] is not None]
        top_growth = sorted(rows, key=lambda row: float(row["singular_value_rank1_delta"]), reverse=True)[:8]
        top_rotation = sorted(
            [row for row in rows if row["rank1_mean_adjacent_rotation"] is not None],
            key=lambda row: float(row["rank1_mean_adjacent_rotation"]),
            reverse=True,
        )[:8]
        coordination_rows.append(
            {
                "interval_start_step": start_step,
                "interval_end_step": end_step,
                "num_matrices": len(rows),
                "num_positive_growth_matrices": len(positive_growth_rows),
                "num_positive_mass_matrices": len(concentration_rows),
                "total_positive_singular_value_rank1_delta": sum(float(row["singular_value_rank1_delta"]) for row in positive_growth_rows),
                "total_abs_singular_value_rank1_delta": sum(abs(float(row["singular_value_rank1_delta"])) for row in rows),
                "total_positive_spectral_mass_top3_delta": sum(float(row["spectral_mass_top3_delta"]) for row in concentration_rows),
                "mean_rank1_adjacent_rotation": None
                if not rotation_rows
                else sum(float(row["rank1_mean_adjacent_rotation"]) for row in rotation_rows) / len(rotation_rows),
                "top_growth_matrices": [
                    {
                        "matrix_label": str(row["matrix_label"]),
                        "delta": float(row["singular_value_rank1_delta"]),
                        "mass_delta": float(row["spectral_mass_top3_delta"]),
                    }
                    for row in top_growth
                ],
                "top_rotation_matrices": [
                    {
                        "matrix_label": str(row["matrix_label"]),
                        "rank1_mean_adjacent_rotation": float(row["rank1_mean_adjacent_rotation"]),
                        "delta": float(row["singular_value_rank1_delta"]),
                    }
                    for row in top_rotation
                ],
            }
        )
    coordination_rows.sort(key=lambda row: int(row["interval_start_step"]))
    interval_rows.sort(
        key=lambda row: (int(row["interval_start_step"]), -float(row["singular_value_rank1_delta"]), str(row["matrix_label"]))
    )
    return interval_rows, coordination_rows


def _format_float(value: Any, *, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}f}"


def _write_markdown_report(
    path: Path,
    *,
    report: dict[str, Any],
    matrix_summary_rows: list[dict[str, Any]],
    vector_summary_rows: list[dict[str, Any]],
    coordination_rows: list[dict[str, Any]],
    top_k: int,
) -> None:
    top_growth = sorted(matrix_summary_rows, key=lambda row: float(row["singular_value_rank1_delta"]), reverse=True)[:top_k]
    top_concentration = sorted(matrix_summary_rows, key=lambda row: float(row["selective_concentration_score"]), reverse=True)[:top_k]
    top_direction_birth = sorted(vector_summary_rows, key=lambda row: float(row["total_rotation_to_final"]), reverse=True)[:top_k]
    top_windows = sorted(
        coordination_rows,
        key=lambda row: float(row["total_positive_singular_value_rank1_delta"]),
        reverse=True,
    )[:top_k]

    lines = [
        "# Weight SVD Pattern Report",
        "",
        "This report reads existing `weight-svd-trace` outputs. It does not rerun model checkpoints or use activations.",
        "",
        "## Files",
        "",
        f"- singular values: `{report['singular_values_path']}`",
        f"- top singular vectors: `{report['top_singular_vectors_path']}`",
        f"- matrix summary rows: `{report['matrix_summary_rows_path']}`",
        f"- vector alignment rows: `{report['vector_alignment_rows_path']}`",
        f"- interval event rows: `{report['interval_event_rows_path']}`",
        f"- coordination window rows: `{report['coordination_window_rows_path']}`",
        "",
        "## Top Rank-1 Singular Value Growth",
        "",
        "| matrix | start -> end | sv1 delta | rel delta | eff rank delta | top3 mass delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_growth:
        lines.append(
            "| "
            f"{row['matrix_label']} | {row['start_step']} -> {row['end_step']} | "
            f"{_format_float(row['singular_value_rank1_delta'])} | "
            f"{_format_float(row['singular_value_rank1_relative_delta'])} | "
            f"{_format_float(row['effective_rank_delta'])} | "
            f"{_format_float(row['spectral_mass_top3_delta'])} |"
        )

    lines.extend(
        [
            "",
            "## Top Selective Concentration",
            "",
            "This ranks matrices where the top singular value grew while top-3 spectral mass also increased.",
            "",
            "| matrix | sv1 delta | top3 mass delta | score |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in top_concentration:
        lines.append(
            "| "
            f"{row['matrix_label']} | "
            f"{_format_float(row['singular_value_rank1_delta'])} | "
            f"{_format_float(row['spectral_mass_top3_delta'])} | "
            f"{_format_float(row['selective_concentration_score'])} |"
        )

    lines.extend(
        [
            "",
            "## Largest Direction-Birth Candidates",
            "",
            "These vectors started least aligned with their final direction, then ended at that final direction by construction.",
            "",
            "| vector | start final cosine | first final-aligned step | first stable step | worst adjacent interval |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in top_direction_birth:
        worst_interval = (
            "n/a"
            if row["min_adjacent_interval_start"] is None
            else f"{row['min_adjacent_interval_start']}->{row['min_adjacent_interval_end']} ({_format_float(row['min_adjacent_abs_cosine'])})"
        )
        lines.append(
            "| "
            f"{row['matrix_label']} r{row['singular_value_rank']} {row['vector_side']} | "
            f"{_format_float(row['start_final_abs_cosine'])} | "
            f"{row['first_final_alignment_step']} | "
            f"{row['first_adjacent_stable_step']} | "
            f"{worst_interval} |"
        )

    lines.extend(
        [
            "",
            "## Most Coordinated Growth Windows",
            "",
            "| interval | total positive sv1 delta | positive-growth matrices | positive-mass matrices | top growth matrices |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in top_windows:
        top_labels = ", ".join(
            f"{item['matrix_label']} ({_format_float(item['delta'], precision=3)})"
            for item in row["top_growth_matrices"][:5]
        )
        lines.append(
            "| "
            f"{row['interval_start_step']} -> {row['interval_end_step']} | "
            f"{_format_float(row['total_positive_singular_value_rank1_delta'])} | "
            f"{row['num_positive_growth_matrices']} | "
            f"{row['num_positive_mass_matrices']} | "
            f"{top_labels} |"
        )

    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- `sv1 delta`: change in the top singular value from first to last checkpoint.",
            "- `effective rank`: `(sum singular values)^2 / sum(singular values^2)`.",
            "- `top3 mass`: fraction of the singular value mass in the top three singular values.",
            "- `first final-aligned step`: first checkpoint after which final-direction cosine stays above threshold.",
            "- `first stable step`: first checkpoint after which adjacent-checkpoint cosine stays above threshold.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def run_weight_svd_patterns(
    *,
    singular_values_path: Path,
    top_singular_vectors_path: Path,
    output_dir: Path,
    max_vector_rank: int = 1,
    final_alignment_threshold: float = 0.95,
    adjacent_stability_threshold: float = 0.99,
    stability_patience: int = 3,
    markdown_top_k: int = 16,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    _prepare_output_dir(output_dir, overwrite=overwrite)

    rows_by_key = _load_rank1_singular_rows(singular_values_path)
    vector_rows_by_key = _load_vector_rows(top_singular_vectors_path, max_rank=max_vector_rank)
    matrix_summary_rows = _build_matrix_summary_rows(rows_by_key)
    vector_alignment_rows, vector_summary_rows = _build_vector_alignment_rows(
        vector_rows_by_key,
        final_alignment_threshold=final_alignment_threshold,
        adjacent_stability_threshold=adjacent_stability_threshold,
        stability_patience=stability_patience,
    )
    interval_event_rows, coordination_window_rows = _build_interval_event_rows(rows_by_key, vector_alignment_rows)

    report_path = output_dir / "weight_svd_pattern_report.json"
    markdown_path = output_dir / "weight_svd_pattern_report.md"
    matrix_summary_rows_path = output_dir / "weight_svd_matrix_summary.jsonl"
    matrix_summary_csv_path = output_dir / "weight_svd_matrix_summary.csv"
    vector_alignment_rows_path = output_dir / "weight_svd_vector_alignment.jsonl"
    interval_event_rows_path = output_dir / "weight_svd_interval_events.jsonl"
    coordination_window_rows_path = output_dir / "weight_svd_coordination_windows.jsonl"

    write_jsonl(matrix_summary_rows_path, matrix_summary_rows)
    _write_csv(
        matrix_summary_csv_path,
        matrix_summary_rows,
        fieldnames=[
            "layer",
            "head",
            "component_type",
            "matrix_name",
            "matrix_label",
            "start_step",
            "end_step",
            "start_singular_value_rank1",
            "end_singular_value_rank1",
            "singular_value_rank1_delta",
            "singular_value_rank1_relative_delta",
            "start_effective_rank",
            "end_effective_rank",
            "effective_rank_delta",
            "start_spectral_mass_top3",
            "end_spectral_mass_top3",
            "spectral_mass_top3_delta",
            "selective_concentration_score",
            "singular_value_count",
        ],
    )
    write_jsonl(vector_alignment_rows_path, vector_alignment_rows)
    write_jsonl(interval_event_rows_path, interval_event_rows)
    write_jsonl(coordination_window_rows_path, coordination_window_rows)

    steps = sorted({int(row["step"]) for rows in rows_by_key.values() for row in rows})
    report = {
        "singular_values_path": str(singular_values_path),
        "top_singular_vectors_path": str(top_singular_vectors_path),
        "output_dir": str(output_dir),
        "num_steps": len(steps),
        "steps": steps,
        "num_matrices": len(rows_by_key),
        "num_vector_series": len(vector_rows_by_key),
        "max_vector_rank": max_vector_rank,
        "final_alignment_threshold": final_alignment_threshold,
        "adjacent_stability_threshold": adjacent_stability_threshold,
        "stability_patience": stability_patience,
        "matrix_summary_rows_path": str(matrix_summary_rows_path),
        "matrix_summary_csv_path": str(matrix_summary_csv_path),
        "vector_alignment_rows_path": str(vector_alignment_rows_path),
        "interval_event_rows_path": str(interval_event_rows_path),
        "coordination_window_rows_path": str(coordination_window_rows_path),
        "top_rank1_growth": matrix_summary_rows[: min(markdown_top_k, len(matrix_summary_rows))],
        "top_selective_concentration": sorted(
            matrix_summary_rows,
            key=lambda row: float(row["selective_concentration_score"]),
            reverse=True,
        )[: min(markdown_top_k, len(matrix_summary_rows))],
        "top_direction_birth_candidates": sorted(
            vector_summary_rows,
            key=lambda row: float(row["total_rotation_to_final"]),
            reverse=True,
        )[: min(markdown_top_k, len(vector_summary_rows))],
        "top_coordinated_growth_windows": sorted(
            coordination_window_rows,
            key=lambda row: float(row["total_positive_singular_value_rank1_delta"]),
            reverse=True,
        )[: min(markdown_top_k, len(coordination_window_rows))],
    }
    write_json(report_path, report)
    _write_markdown_report(
        markdown_path,
        report=report,
        matrix_summary_rows=matrix_summary_rows,
        vector_summary_rows=vector_summary_rows,
        coordination_rows=coordination_window_rows,
        top_k=markdown_top_k,
    )
    print(
        "[weight-svd-patterns] complete "
        f"report={report_path} matrices={len(rows_by_key)} vector_series={len(vector_rows_by_key)} "
        f"steps={len(steps)}",
        flush=True,
    )
    return (
        report_path,
        markdown_path,
        matrix_summary_rows_path,
        matrix_summary_csv_path,
        vector_alignment_rows_path,
        interval_event_rows_path,
        coordination_window_rows_path,
    )
