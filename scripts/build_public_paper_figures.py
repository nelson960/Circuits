#!/usr/bin/env python3
"""Build public-paper SVG figures from existing analysis artifacts.

The script is intentionally strict: expected artifact files must exist and
expected metrics must be present. Missing data is an error, not a fallback.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "assets" / "figures"

WEIGHT_SVD_CSV = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/weight_svd_trace/"
    "phase1_000250_5500_top16/weight_svd_singular_values.csv"
)
QK_MATCH_CSV = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/bilinear_qk_match_separation/"
    "l2h1_support_value_vs_distractors_000250_005500_stage_sweep/"
    "bilinear_qk_match_separation_rows.csv"
)
ADAM_REPORT = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/"
    "bilinear_qk_rank_adam_state_attribution/"
    "from_init_l2h1_rank8_support_value_0000_6000_stepwise/"
    "bilinear_qk_rank_adam_state_attribution_report.json"
)
STATIC_ALIGN_REPORT = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/svd_task_alignment/"
    "l2h1_qk_key_geometry_000250_005500/svd_task_alignment_report.json"
)
CONTEXT_ALIGN_REPORT = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/contextual_svd_alignment/"
    "l2h1_prediction_grouped_by_query_key_layer1_post_mlp_000250_005500/"
    "contextual_svd_alignment_report.json"
)
KEY_SEPARABILITY_REPORT = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/contextual_key_separability/"
    "l2h1_prediction_query_key_stage_sweep_000250_005500/"
    "contextual_key_separability_report.json"
)
WRITE_TRAJECTORY_ROWS = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_functional_subspace_trajectory/"
    "l0h0_to_l0mlp_prediction_ref2500_0750_3500/"
    "mlp_functional_subspace_trajectory_functional_summary_rows.jsonl"
)
REFERENCE_WRITE_ADAM_REPORT = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_functional_write_adam_state_attribution/"
    "l0h0_l0mlp_prediction_ref2500_postgrad_total_1500_2500/"
    "mlp_functional_write_adam_state_attribution_report.json"
)
ROUTE_TO_SCALAR_CLOSURE_REPORT = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/route_to_scalar_closure/"
    "qk_ov_output_routes_1500_2500_formation/route_to_scalar_closure_report.json"
)
OUTPUT_ROUTE_CLOSURE_REPORT = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/output_route_closure/"
    "qk_ov_output_routes_1500_2500_formation/output_route_closure_report.json"
)
LINE_INTEGRAL_REPORT = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/component_output_rescue_line_integral/"
    "l0h0_full_converter_1500_2500_stride10/component_output_rescue_line_integral_report.json"
)
BRANCH_DECOMPOSITION_REPORT = ROOT / (
    "artifacts/runs/symbolic_kv_reference_formation/analysis/answer_margin_branch_decomposition/"
    "query_key_support_value_5500_5550_stepwise/answer_margin_branch_decomposition_report.json"
)
CROSS_SEED_ROOT = ROOT / "artifacts/runs/symbolic_kv_cross_seed_adam"


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> dict:
    return json.loads(require_path(path).read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(require_path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise RuntimeError(f"Empty JSONL line in {path} at {line_number}")
        rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def write_svg(name: str, width: int, height: int, body: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        "<defs>\n"
        "<marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"8\" refY=\"3\" "
        "orient=\"auto\" markerUnits=\"strokeWidth\">\n"
        "<path d=\"M0,0 L0,6 L9,3 z\" fill=\"#343434\" />\n"
        "</marker>\n"
        "<style>\n"
        ".title{font:700 22px Georgia,serif;fill:#171615}.subtitle{font:15px Georgia,serif;fill:#4a4741}"
        ".label{font:14px Georgia,serif;fill:#2a2927}.small{font:12px Georgia,serif;fill:#5a5247}"
        ".tiny{font:10.5px Georgia,serif;fill:#5a5247}.axis{stroke:#4a4741;stroke-width:1}"
        ".grid{stroke:#ddd6c8;stroke-width:1}.box{fill:#fffdf8;stroke:#9b8f7e;stroke-width:1.4;rx:8}"
        ".ok{fill:#e7f3ed;stroke:#4f8f6f}.warn{fill:#fff4dc;stroke:#b78b3b}.open{fill:#f8e8e6;stroke:#aa615b}"
        "</style>\n"
        "</defs>\n"
        f"{body}\n</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")
    print(path.relative_to(ROOT))


def text(x: float, y: float, value: str, cls: str = "label", anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(value)}</text>'


def rect(x: float, y: float, w: float, h: float, cls: str = "box") -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" class="{cls}" />'


def line(x1: float, y1: float, x2: float, y2: float, cls: str = "", color: str = "#343434", width: float = 1.5, arrow: bool = False) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    class_attr = f' class="{cls}"' if cls else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"'
        f'{class_attr} stroke="{color}" stroke-width="{width}"{marker}/>'
    )


def polyline(points: list[tuple[float, float]], color: str, width: float = 2.5) -> str:
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{point_text}" />'


def normalize(values: list[float]) -> list[float]:
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        raise ValueError(f"Cannot normalize constant series: {values[:5]}")
    return [(v - lo) / (hi - lo) for v in values]


def plot_series(
    x_values: list[float],
    series: list[tuple[str, list[float], str]],
    width: int = 980,
    height: int = 430,
    title: str = "",
    subtitle: str = "",
) -> str:
    left, right, top, bottom = 74, width - 32, 82, height - 62
    x_min, x_max = min(x_values), max(x_values)
    parts = [text(32, 34, title, "title"), text(32, 58, subtitle, "subtitle")]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = bottom - frac * (bottom - top)
        parts.append(line(left, y, right, y, "grid", "#ddd6c8", 1))
        parts.append(text(54, y + 4, f"{frac:.2f}", "tiny", "end"))
    parts.append(line(left, bottom, right, bottom, "axis", "#4a4741", 1.2))
    parts.append(line(left, top, left, bottom, "axis", "#4a4741", 1.2))
    for step in [min(x_values), 750, 2500, 3500, max(x_values)]:
        if step < x_min or step > x_max:
            continue
        x = left + (step - x_min) / (x_max - x_min) * (right - left)
        parts.append(line(x, bottom, x, bottom + 6, color="#4a4741", width=1))
        parts.append(text(x, bottom + 22, str(int(step)), "tiny", "middle"))
    parts.append(text((left + right) / 2, height - 18, "training step", "small", "middle"))
    parts.append(text(20, (top + bottom) / 2, "min-max normalized", "small", "middle"))
    parts.append(line(
        left + (750 - x_min) / (x_max - x_min) * (right - left),
        top,
        left + (750 - x_min) / (x_max - x_min) * (right - left),
        bottom,
        color="#a67c00",
        width=1.2,
    ))
    parts.append(line(
        left + (3500 - x_min) / (x_max - x_min) * (right - left),
        top,
        left + (3500 - x_min) / (x_max - x_min) * (right - left),
        bottom,
        color="#a67c00",
        width=1.2,
    ))
    parts.append(text(left + 150, top - 10, "main formation window: 750 -> 3500", "small"))
    legend_x, legend_y = right - 260, top + 8
    for i, (label, values, color) in enumerate(series):
        ys = normalize(values)
        pts = []
        for step, y_norm in zip(x_values, ys):
            x = left + (step - x_min) / (x_max - x_min) * (right - left)
            y = bottom - y_norm * (bottom - top)
            pts.append((x, y))
        parts.append(polyline(pts, color))
        ly = legend_y + i * 22
        parts.append(line(legend_x, ly - 4, legend_x + 32, ly - 4, color=color, width=3))
        parts.append(text(legend_x + 40, ly, label, "small"))
    return "\n".join(parts)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with require_path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_one(rows: list[dict], label: str) -> dict:
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row for {label}, found {len(rows)}")
    return rows[0]


def build_updated_chain() -> None:
    labels = [
        ("Data relation", ("latest write", "answers read")),
        ("Loss", ("next-token CE", "on real batches")),
        ("AdamW state", ("preconditioned current", "+ momentum")),
        ("Weight geometry", ("rank-8 W_QK", "support matcher")),
        ("Route", ("support value", "> distractors")),
        ("Behavior", ("answer margin", "improves")),
    ]
    x0, y, w, h, gap = 26, 108, 150, 92, 24
    parts = [text(30, 38, "From loss to lookup: the measured chain", "title"),
             text(30, 63, "Current paper claim: a support-value retrieval role is built by actual AdamW updates.", "subtitle")]
    for i, (a, b_lines) in enumerate(labels):
        x = x0 + i * (w + gap)
        parts.append(rect(x, y, w, h))
        parts.append(text(x + w / 2, y + 29, a, "label", "middle"))
        parts.append(text(x + w / 2, y + 57, b_lines[0], "small", "middle"))
        parts.append(text(x + w / 2, y + 75, b_lines[1], "small", "middle"))
        if i < len(labels) - 1:
            parts.append(line(x + w + 5, y + h / 2, x + w + gap - 8, y + h / 2, arrow=True))
    parts.append(text(30, 250, "The old question was \"which neuron matters?\" The current proof object is a differentiable route scalar C(theta).", "small"))
    parts.append(text(30, 273, "Supported: route growth, low-rank QK birth, exact AdamW update reconstruction, and 5-seed role replication.", "small"))
    parts.append(text(30, 296, "Open: full answer-margin closure and whether the same method scales beyond this symbolic model.", "small"))
    write_svg("updated_loss_to_lookup_chain.svg", 1080, 335, "\n".join(parts))


def build_weight_birth() -> None:
    rows = read_csv_rows(WEIGHT_SVD_CSV)
    sv_rows = [
        r for r in rows
        if r["head"] != "" and int(r["layer"]) == 2 and int(r["head"]) == 1
        and r["matrix_name"] == "W_QK" and int(r["singular_value_rank"]) == 1
    ]
    if not sv_rows:
        raise RuntimeError("No L2H1 W_QK rank-1 SVD rows found.")
    sv_rows.sort(key=lambda r: int(r["step"]))
    steps = [int(r["step"]) for r in sv_rows]
    singular = [float(r["singular_value"]) for r in sv_rows]
    eff_rank_compression = [-float(r["effective_rank"]) for r in sv_rows]
    top3_mass = [float(r["spectral_mass_top3"]) for r in sv_rows]

    qk_rows = read_csv_rows(QK_MATCH_CSV)
    qk_rows = [
        r for r in qk_rows
        if r["context_stage"] == "layer_1_post_mlp" and r["projection"] == "rank_8"
    ]
    by_step = {int(r["checkpoint_step"]): float(r["qk_match_separation_mean"]) for r in qk_rows}
    retrieval = [by_step[s] for s in steps if s in by_step]
    retrieval_steps = [s for s in steps if s in by_step]
    if retrieval_steps != steps:
        raise RuntimeError("QK match rows do not align with SVD steps.")

    body = plot_series(
        steps,
        [
            ("W_QK top singular value", singular, "#245f73"),
            ("effective-rank compression", eff_rank_compression, "#8f5a24"),
            ("top-3 spectral mass", top3_mass, "#4f7f54"),
            ("rank-8 support separation", retrieval, "#8b3f58"),
        ],
        title="L2H1 W_QK forms a concentrated support-value matcher",
        subtitle="All curves are artifact-backed and min-max normalized for shape comparison.",
    )
    write_svg("weight_qk_birth_timeline.svg", 980, 430, body)


def build_contextual_alignment() -> None:
    static = load_json(STATIC_ALIGN_REPORT)
    contextual = load_json(CONTEXT_ALIGN_REPORT)
    sep = load_json(KEY_SEPARABILITY_REPORT)
    metrics = [
        ("static key cosine vs margin", static["rank1_correlations"]["right_key_cosine_vs_answer_margin"]),
        ("static PCA overlap vs margin", static["rank1_correlations"]["right_key_pca_overlap_vs_answer_margin"]),
        ("contextual support cosine vs margin", contextual["rank1_correlations"]["support_value.right_mean_cosine_vs_answer_margin"]),
        ("singular value vs contextual support", contextual["rank1_correlations"]["support_value.singular_value_vs_right_mean_cosine"]),
    ]
    full_rows = [r for r in sep["summary_rows"] if r["context_stage"] == "layer_1_post_mlp" and r["projection"] == "qk_both"]
    if len(full_rows) != 1:
        raise RuntimeError("Expected one layer_1_post_mlp qk_both separability summary row.")
    full = full_rows[0]
    width, height = 960, 390
    left, top, bar_w, gap = 82, 120, 145, 42
    parts = [text(32, 36, "Semantic target: contextual residual geometry", "title"),
             text(32, 61, "The useful QK direction should be read against the representations produced by earlier layers, not only raw token embeddings.", "subtitle")]
    axis_y = 285
    parts.append(line(left - 20, axis_y, 850, axis_y, color="#4a4741", width=1.2))
    for i, (label, val) in enumerate(metrics):
        x = left + i * (bar_w + gap)
        y0 = axis_y
        bar_h = abs(val) * 130
        y = y0 - bar_h if val >= 0 else y0
        color = "#4f7f54" if val >= 0 else "#b95f56"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity="0.88"/>')
        parts.append(text(x + bar_w / 2, y - 8 if val >= 0 else y + bar_h + 18, f"{val:.3f}", "small", "middle"))
        words = label.split()
        parts.append(text(x + bar_w / 2, axis_y + 28, " ".join(words[:2]), "tiny", "middle"))
        parts.append(text(x + bar_w / 2, axis_y + 43, " ".join(words[2:]), "tiny", "middle"))
    parts.append(line(left - 20, axis_y - 130, 850, axis_y - 130, "grid", "#ddd6c8", 1))
    parts.append(line(left - 20, axis_y + 130, 850, axis_y + 130, "grid", "#ddd6c8", 1))
    parts.append(text(842, axis_y - 128, "+1", "tiny"))
    parts.append(text(842, axis_y + 134, "-1", "tiny"))
    parts.append(rect(660, 86, 250, 78, "box"))
    parts.append(text(785, 113, "Layer-1 contextual separability", "small", "middle"))
    parts.append(text(785, 137, f"qk_both window delta: {full['window_delta_separation_ratio']:.3f}", "tiny", "middle"))
    parts.append(text(785, 153, f"end separation ratio: {full['end_separation_ratio']:.3f}", "tiny", "middle"))
    parts.append(text(32, 358, "Interpretation: static embeddings give some signal, but the route is best treated as reading contextual residual states built by earlier layers.", "small"))
    write_svg("contextual_semantic_alignment.svg", width, height, "\n".join(parts))


def sum_qk_phase(rows: list[dict], start: int, end: int) -> dict[str, float]:
    selected = [
        r for r in rows
        if int(r["source_step"]) >= start and int(r["target_step"]) <= end
    ]
    expected = end - start
    if len(selected) != expected:
        raise RuntimeError(f"Expected {expected} QK rows for {start}->{end}, found {len(selected)}")
    return {
        "actual": sum(float(r["actual_rank_match_delta"]) for r in selected),
        "predicted": sum(float(r["reconstructed_adamw_rank_delta"]) for r in selected),
        "raw_sgd": sum(float(r["raw_sgd_rank_delta"]) for r in selected),
        "current": sum(float(r["adam_current_gradient_rank_delta"]) for r in selected),
        "momentum": sum(float(r["adam_historical_momentum_rank_delta"]) for r in selected),
        "decay": sum(float(r["weight_decay_rank_delta"]) for r in selected),
    }


def build_qk_optimizer_phase_structure() -> None:
    report = load_json(ADAM_REPORT)
    metric_rows = load_jsonl(ROOT / report["metric_rows_path"])
    phases = [
        ("0 -> 750", 0, 750, "setup"),
        ("750 -> 2500", 750, 2500, "momentum birth"),
        ("2500 -> 3500", 2500, 3500, "fresh gradients join"),
        ("3500 -> 6000", 3500, 6000, "saturation"),
    ]
    phase_rows = [(label, tag, sum_qk_phase(metric_rows, start, end)) for label, start, end, tag in phases]
    width, height = 1040, 520
    parts = [
        text(32, 36, "QK route formation has phases", "title"),
        text(32, 61, "Rank-8 L2H1 support-value route, summed from every one-step AdamW attribution row.", "subtitle"),
    ]
    left, right, top, bottom = 78, width - 42, 112, 332
    max_abs = max(abs(v) for _, _, row in phase_rows for v in row.values())
    zero_y = top + max_abs / (2 * max_abs) * (bottom - top)
    parts.append(line(left, zero_y, right, zero_y, color="#4a4741", width=1.2))
    colors = {
        "actual": "#245f73",
        "predicted": "#4f7f54",
        "raw_sgd": "#b95f56",
        "current": "#7b5ea7",
        "momentum": "#8f5a24",
        "decay": "#777777",
    }
    keys = ["actual", "predicted", "raw_sgd", "current", "momentum", "decay"]
    group_w = (right - left) / len(phase_rows)
    bw = 18
    for group_i, (label, tag, values) in enumerate(phase_rows):
        group_x = left + group_i * group_w + 18
        parts.append(text(group_x + 72, 94, label, "small", "middle"))
        parts.append(text(group_x + 72, 356, tag, "tiny", "middle"))
        for i, key in enumerate(keys):
            val = values[key]
            h = abs(val) / max_abs * 96
            x = group_x + i * (bw + 8)
            y = zero_y - h if val >= 0 else zero_y
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{colors[key]}" opacity="0.9"/>')
        parts.append(text(group_x + 72, 381, f"actual {values['actual']:+.3f}", "tiny", "middle"))
        parts.append(text(group_x + 72, 398, f"raw {values['raw_sgd']:+.3f}", "tiny", "middle"))
        parts.append(text(group_x + 72, 415, f"cur {values['current']:+.3f} / mom {values['momentum']:+.3f}", "tiny", "middle"))
    legend_x, legend_y = 140, 455
    for i, key in enumerate(keys):
        x = legend_x + i * 135
        parts.append(f'<rect x="{x:.1f}" y="{legend_y - 11:.1f}" width="18" height="10" fill="{colors[key]}" opacity="0.9"/>')
        parts.append(text(x + 26, legend_y, key.replace("_", " "), "tiny"))
    parts.append(text(32, 493, "The cleanest birth window is 750 -> 2500: actual route growth is positive while raw SGD-equivalent movement is slightly negative.", "small"))
    write_svg("qk_optimizer_phase_structure.svg", width, height, "\n".join(parts))


def build_reference_write_optimizer_split() -> None:
    rows = load_json(REFERENCE_WRITE_ADAM_REPORT)["summary"]["scalar_rows"]
    if not rows:
        raise RuntimeError(f"No scalar rows in {REFERENCE_WRITE_ADAM_REPORT}")
    values = [
        ("actual", sum(float(r["sum_actual_score_delta"]) for r in rows), "#245f73"),
        ("reconstructed", sum(float(r["sum_reconstructed_adamw_scalar_delta"]) for r in rows), "#4f7f54"),
        ("raw SGD-eq", sum(float(r["sum_raw_sgd_scalar_delta"]) for r in rows), "#b95f56"),
        ("Adam current", sum(float(r["sum_adam_current_gradient_scalar_delta"]) for r in rows), "#7b5ea7"),
        ("Adam momentum", sum(float(r["sum_adam_historical_momentum_scalar_delta"]) for r in rows), "#8f5a24"),
        ("weight decay", sum(float(r["sum_weight_decay_scalar_delta"]) for r in rows), "#777777"),
    ]
    actual = values[0][1]
    width, height = 900, 405
    left, top, bottom = 76, 96, 280
    max_abs = max(abs(v) for _, v, _ in values)
    zero_y = bottom
    parts = [
        text(32, 36, "Reference-seed write scalar is momentum-heavy", "title"),
        text(32, 61, "L0H0 -> L0MLP, prediction position, fixed step-2500 readout, 1500 -> 2500.", "subtitle"),
        line(left, zero_y, 844, zero_y, color="#4a4741", width=1.2),
    ]
    bw, gap = 92, 28
    for i, (label, val, color) in enumerate(values):
        x = left + i * (bw + gap)
        h = abs(val) / max_abs * 158
        y = zero_y - h if val >= 0 else zero_y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{color}" opacity="0.9"/>')
        parts.append(text(x + bw / 2, y - 8 if val >= 0 else y + h + 18, f"{val:+.3f}", "small", "middle"))
        words = label.split()
        parts.append(text(x + bw / 2, 315, words[0], "tiny", "middle"))
        if len(words) > 1:
            parts.append(text(x + bw / 2, 330, " ".join(words[1:]), "tiny", "middle"))
    parts.append(text(32, 370, f"Relative to actual growth: raw SGD-eq {values[2][1] / actual * 100:.2f}%, Adam current {values[3][1] / actual * 100:.1f}%, Adam momentum {values[4][1] / actual * 100:.1f}%.", "small"))
    write_svg("reference_write_optimizer_split.svg", width, height, "\n".join(parts))


def build_write_side_mechanism() -> None:
    parts = [
        text(32, 36, "Write side: the useful object is a residual coupling", "title"),
        text(32, 61, "QK chooses a source. The write side asks whether the resulting residual change is readable by downstream answer directions.", "subtitle"),
    ]
    boxes = [
        (38, 126, 150, 86, "L0H0 source", "attends to value-bearing state"),
        (240, 126, 185, 86, "delta_write(x)", "change at prediction position"),
        (477, 126, 155, 86, "L0MLP boundary", "residual + nonlinear correction"),
        (684, 126, 180, 86, "g_ref(x)", "mature answer-readout direction"),
    ]
    for i, (x, y, w, h, a, b) in enumerate(boxes):
        parts.append(rect(x, y, w, h))
        parts.append(text(x + w / 2, y + 34, a, "label", "middle"))
        parts.append(text(x + w / 2, y + 58, b, "tiny", "middle"))
        if i < len(boxes) - 1:
            parts.append(line(x + w + 10, y + h / 2, boxes[i + 1][0] - 10, y + h / 2, arrow=True))
    parts.append(rect(156, 270, 668, 70, "warn"))
    parts.append(text(490, 298, "C_write(theta) = E_x [ g_ref(x) . delta_write_theta(x) ]", "label", "middle"))
    parts.append(text(490, 320, "This scalar asks whether the write is useful to the finished readout, not whether W_OV directly points at a token embedding.", "small", "middle"))
    parts.append(text(32, 385, "Interpretation: static W_OV was the wrong proof object. The measured write is contextual, position-specific, and read through later computation.", "small"))
    write_svg("write_side_mechanism.svg", 980, 430, "\n".join(parts))


def build_write_functional_birth() -> None:
    rows = [
        r for r in load_jsonl(WRITE_TRAJECTORY_ROWS)
        if r["group_by"] == "answer_value"
        and r["position_role"] == "prediction"
        and r["mlp_component"] == "L0MLP"
    ]
    if not rows:
        raise RuntimeError("No L0H0 -> L0MLP prediction write-trajectory rows found.")
    by_step: dict[int, dict[str, float]] = {}
    for step in sorted({int(r["step"]) for r in rows}):
        step_rows = [r for r in rows if int(r["step"]) == step]
        if not step_rows:
            raise RuntimeError(f"No write trajectory rows for step {step}")
        count = len(step_rows)
        by_step[step] = {
            "total": sum(float(r["sum_post_mlp_gradient_dot_total_delta"]) for r in step_rows) / count,
            "skip": sum(float(r["sum_post_mlp_gradient_dot_skip_delta"]) for r in step_rows) / count,
            "mlp": sum(float(r["sum_post_mlp_gradient_dot_mlp_output_delta"]) for r in step_rows) / count,
            "count": count,
        }
    steps = sorted(by_step)
    if len(steps) < 5:
        raise RuntimeError(f"Expected a trajectory, found only {len(steps)} steps in {WRITE_TRAJECTORY_ROWS}")

    width, height = 980, 460
    left, right, top, bottom = 78, width - 34, 88, height - 70
    y_values = [by_step[s][k] for s in steps for k in ("total", "skip", "mlp")]
    y_min = min(-5.0, min(y_values))
    y_max = max(110.0, max(y_values))
    x_min, x_max = min(steps), max(steps)

    def x_for(step: int) -> float:
        return left + (step - x_min) / (x_max - x_min) * (right - left)

    def y_for(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    parts = [
        text(32, 36, "Functional write coupling turns on before the late behavior plateau", "title"),
        text(32, 61, "Reference seed L0H0 -> L0MLP, prediction position, grouped by answer value.", "subtitle"),
    ]
    for tick in [0, 25, 50, 75, 100]:
        y = y_for(tick)
        parts.append(line(left, y, right, y, "grid", "#ddd6c8", 1))
        parts.append(text(left - 10, y + 4, str(tick), "tiny", "end"))
    parts.append(line(left, bottom, right, bottom, "axis", "#4a4741", 1.2))
    parts.append(line(left, top, left, bottom, "axis", "#4a4741", 1.2))
    for step in [750, 1500, 1750, 2500, 3500]:
        if step < x_min or step > x_max:
            continue
        x = x_for(step)
        parts.append(line(x, bottom, x, bottom + 6, color="#4a4741", width=1))
        parts.append(text(x, bottom + 24, str(step), "tiny", "middle"))
    for step, label in [(1500, "coupling starts"), (1750, "sharp jump")]:
        x = x_for(step)
        parts.append(line(x, top, x, bottom, color="#a67c00", width=1.2))
        parts.append(text(x + 6, top + (16 if step == 1500 else 34), label, "tiny"))
    series = [
        ("total functional write", "total", "#245f73"),
        ("residual skip part", "skip", "#4f7f54"),
        ("L0MLP nonlinear part", "mlp", "#8f5a24"),
    ]
    for label, key, color in series:
        pts = [(x_for(s), y_for(by_step[s][key])) for s in steps]
        parts.append(polyline(pts, color, 3))
    legend_x, legend_y = 670, 102
    for i, (label, _, color) in enumerate(series):
        y = legend_y + i * 22
        parts.append(line(legend_x, y - 5, legend_x + 36, y - 5, color=color, width=3))
        parts.append(text(legend_x + 45, y, label, "small"))
    parts.append(text(32, 432, "The direction is not born from zero. What changes sharply is its coupling to the mature answer-readout direction.", "small"))
    write_svg("write_functional_birth.svg", width, height, "\n".join(parts))


def build_cross_seed_qk_write_role_map() -> None:
    seed_rows = []
    for selection_path in sorted(CROSS_SEED_ROOT.glob("seed_*/analysis/cross_seed_head_selection.json")):
        seed_match = re.search(r"seed_(\d+)", str(selection_path))
        if not seed_match:
            raise RuntimeError(f"Could not parse seed from {selection_path}")
        seed = int(seed_match.group(1))
        selection = load_json(selection_path)
        qk_winner = selection["candidates"][0]["head_label"]
        write_reports = sorted(
            selection_path.parent.glob(
                "mlp_functional_write_adam_state_attribution/winner_*/"
                "mlp_functional_write_adam_state_attribution_report.json"
            )
        )
        if len(write_reports) != 1:
            raise RuntimeError(f"Expected one write winner report for seed {seed}, found {len(write_reports)}")
        report_name = write_reports[0].parent.name
        match = re.match(r"winner_(?P<source>.+)_to_(?P<mlp>.+)_prediction_ref2500_postgrad_total_1500_2500", report_name)
        if not match:
            raise RuntimeError(f"Could not parse write winner report name: {report_name}")
        write_report = load_json(write_reports[0])
        scalar_rows = write_report["summary"]["scalar_rows"]
        if not scalar_rows:
            raise RuntimeError(f"No scalar rows in {write_reports[0]}")
        actual_mean = sum(float(r["sum_actual_score_delta"]) for r in scalar_rows) / len(scalar_rows)
        raw_pct = (
            sum(float(r["sum_raw_sgd_scalar_delta"]) for r in scalar_rows)
            / sum(float(r["sum_reconstructed_adamw_scalar_delta"]) for r in scalar_rows)
            * 100.0
        )
        seed_rows.append({
            "seed": seed,
            "qk_winner": qk_winner,
            "write_source": match.group("source"),
            "write_mlp": match.group("mlp"),
            "write_actual_mean": actual_mean,
            "write_raw_pct": raw_pct,
        })
    if len(seed_rows) != 5:
        raise RuntimeError(f"Expected five cross-seed rows, found {len(seed_rows)}")

    parts = [
        text(32, 36, "Role repeats, address changes on both sides of the circuit", "title"),
        text(32, 61, "QK winners and write/readout paths are selected from independent cross-seed artifacts.", "subtitle"),
    ]
    x_seed, x_qk, x_write, x_effect = 60, 150, 360, 670
    y0, row_h = 110, 56
    parts.extend([
        text(x_seed, 90, "seed", "small"),
        text(x_qk, 90, "QK retrieval winner", "small"),
        text(x_write, 90, "write/readout winner", "small"),
        text(x_effect, 90, "mean write scalar", "small"),
    ])
    max_effect = max(abs(r["write_actual_mean"]) for r in seed_rows)
    for i, row in enumerate(seed_rows):
        y = y0 + i * row_h
        parts.append(rect(42, y - 28, 880, 42, "box"))
        parts.append(text(x_seed, y, str(row["seed"]), "label"))
        parts.append(text(x_qk, y, row["qk_winner"], "label"))
        parts.append(text(x_write, y, f"{row['write_source']} -> {row['write_mlp']}", "label"))
        bar_w = 160 * abs(row["write_actual_mean"]) / max_effect
        parts.append(f'<rect x="{x_effect:.1f}" y="{y - 16:.1f}" width="{bar_w:.1f}" height="14" fill="#4f7f54" opacity="0.9"/>')
        parts.append(text(x_effect + bar_w + 8, y - 4, f"{row['write_actual_mean']:.2f}", "tiny"))
        parts.append(text(x_effect, y + 14, f"raw SGD-eq / predicted {row['write_raw_pct']:.2f}%", "tiny"))
    parts.append(text(42, 415, "Reading this figure: a named head is not the invariant. The invariant is the role: retrieve support value, then create a readout-useful residual write.", "small"))
    write_svg("cross_seed_qk_write_role_map.svg", 980, 455, "\n".join(parts))


def build_closure_boundary() -> None:
    route_summary = load_json(ROUTE_TO_SCALAR_CLOSURE_REPORT)["summary"]["scalar_bucket_summaries"]
    output_summary = load_json(OUTPUT_ROUTE_CLOSURE_REPORT)["summary"]["scalar_summary_rows"]
    route_by_scalar = {r["scalar_name"]: float(r["r_squared"]) for r in route_summary}
    output_by_scalar = {r["scalar_name"]: float(r["r_squared"]) for r in output_summary}
    scalars = [
        ("correct logit", "correct_value_logit"),
        ("fixed source", "fixed_source_competitor_margin"),
        ("fixed target", "fixed_target_competitor_margin"),
        ("moving margin", "moving_answer_margin"),
        ("neg loss", "negative_answer_loss"),
    ]
    missing = [name for _, name in scalars if name not in route_by_scalar or name not in output_by_scalar]
    if missing:
        raise RuntimeError(f"Missing closure scalar summaries for: {missing}")
    line_rows = load_json(LINE_INTEGRAL_REPORT)["summary"]["score_rows"]
    fixed_source_line = require_one([
        r for r in line_rows
        if r["patch_group_id"] == "L0MLP+L1H3+L1MLP+L2MLP"
        and r["scalar_name"] == "fixed_source_competitor_margin"
    ], "full-converter fixed-source line integral")
    neg_loss_line = require_one([
        r for r in line_rows
        if r["patch_group_id"] == "L0MLP+L1H3+L1MLP+L2MLP"
        and r["scalar_name"] == "negative_answer_loss"
    ], "full-converter negative-loss line integral")
    branch_summary = load_json(BRANCH_DECOMPOSITION_REPORT)["summary"]
    branch_all = require_one([
        r for r in branch_summary["branch_aware_closure_summary_rows"]
        if r["pair_type"] == "__all__" and r["switch_bucket"] == "all"
    ], "branch-aware closure all rows")
    branch_switch = require_one([
        r for r in branch_summary["branch_aware_closure_summary_rows"]
        if r["pair_type"] == "__all__" and r["switch_bucket"] == "competitor_switch"
    ], "branch-aware closure switch rows")
    branch_energy = require_one([
        r for r in branch_summary["branch_summary_rows"]
        if r["pair_type"] == "__all__" and r["switch_bucket"] == "competitor_switch"
    ], "branch-energy switch rows")

    parts = [
        text(32, 36, "Closure boundary: routes help, output space helps more, nonlinear paths still matter", "title"),
        text(32, 61, "Formation-window 1500 -> 2500 closure on 512 observations.", "subtitle"),
    ]
    left, top, bottom = 86, 105, 310
    group_w = 160
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = bottom - tick * (bottom - top)
        parts.append(line(left - 20, y, 910, y, "grid", "#ddd6c8", 1))
        parts.append(text(left - 28, y + 4, f"{tick:.2f}", "tiny", "end"))
    parts.append(line(left - 20, bottom, 910, bottom, color="#4a4741", width=1.2))
    for i, (label, scalar) in enumerate(scalars):
        x = left + i * group_w
        route_r2 = route_by_scalar[scalar]
        output_r2 = output_by_scalar[scalar]
        for j, (value, color) in enumerate([(route_r2, "#d69b3a"), (output_r2, "#245f73")]):
            bw = 42
            bar_h = value * (bottom - top)
            bx = x + j * 48
            by = bottom - bar_h
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw}" height="{bar_h:.1f}" fill="{color}" opacity="0.9"/>')
            parts.append(text(bx + bw / 2, by - 7, f"{value:.2f}", "tiny", "middle"))
        parts.append(text(x + 45, bottom + 24, label, "tiny", "middle"))
    parts.append(line(650, 350, 686, 350, color="#d69b3a", width=4))
    parts.append(text(695, 355, "route/write scalar closure", "small"))
    parts.append(line(650, 374, 686, 374, color="#245f73", width=4))
    parts.append(text(695, 379, "output-space closure", "small"))
    parts.append(rect(42, 386, 875, 58, "warn"))
    parts.append(text(58, 412, f"Branch accounting: moving-margin direct R2 {branch_all['direct_moving_r_squared']:.3f}; fixed-source+branch {branch_all['source_fixed_branch_r_squared']:.3f}; fixed-target+branch {branch_all['target_fixed_branch_r_squared']:.3f}.", "small"))
    parts.append(text(58, 434, f"On switch rows: direct {branch_switch['direct_moving_r_squared']:.3f} -> {branch_switch['source_fixed_branch_r_squared']:.3f}/{branch_switch['target_fixed_branch_r_squared']:.3f}; target branch energy fraction {branch_energy['target_branch_energy_fraction_of_moving']:.3f}.", "small"))
    parts.append(rect(42, 454, 875, 58, "warn"))
    parts.append(text(58, 480, f"Line integral: fixed-source actual {fixed_source_line['sum_actual_endpoint_delta']:.3f}, endpoint first-order {fixed_source_line['sum_source_endpoint_first_order_delta']:.3f}, line integral {fixed_source_line['sum_source_endpoint_line_integral_delta']:.3f}.", "small"))
    parts.append(text(58, 502, f"Negative loss: endpoint first-order {neg_loss_line['sum_source_endpoint_first_order_delta']:.3f}, line integral {neg_loss_line['sum_source_endpoint_line_integral_delta']:.3f}, actual {neg_loss_line['sum_actual_endpoint_delta']:.3f}.", "small"))
    write_svg("closure_boundary.svg", 980, 535, "\n".join(parts))


def build_proof_status() -> None:
    rows = [
        ("Behavior learns lookup", "supported", "heldout-pair accuracy is high"),
        ("Feature families reveal structure", "supported as diagnostic", "shared infrastructure, not atoms"),
        ("QK route scalar", "supported", "support-value separation"),
        ("Weight-level QK birth", "supported", "low-rank W_QK growth"),
        ("QK optimizer cause", "supported for AdamW", "raw SGD-eq tiny, Adam state large"),
        ("Cross-seed QK role", "supported", "same role, moving head address"),
        ("Write functional subspace", "supported", "contextual prediction-position residual coupling"),
        ("Write optimizer cause", "supported for AdamW", "raw SGD-eq tiny, AdamW-preconditioned growth"),
        ("Full answer-margin closure", "partial", "output closure stronger than route closure"),
        ("Plain-SGD sufficiency", "open", "SGD-only training not tested"),
        ("Scaling to LLMs", "open", "requires candidate filtering"),
    ]
    parts = [text(32, 36, "Proof status after the current experiments", "title"),
             text(32, 61, "The result is no longer only a trained-model story, but it is not a universal theorem.", "subtitle")]
    x0, y0, w, h = 55, 92, 840, 31
    for i, (claim, status, note) in enumerate(rows):
        y = y0 + i * 36
        cls = "ok" if status.startswith("supported") else "open"
        if status == "supported as diagnostic":
            cls = "warn"
        if status == "partial":
            cls = "warn"
        parts.append(rect(x0, y - 22, w, h, cls))
        parts.append(text(x0 + 18, y, claim, "label"))
        parts.append(text(x0 + 390, y, status, "small"))
        parts.append(text(x0 + 585, y, note, "small"))
    write_svg("proof_status_ladder_updated.svg", 980, 475, "\n".join(parts))


def main() -> None:
    for path in [
        WEIGHT_SVD_CSV,
        QK_MATCH_CSV,
        ADAM_REPORT,
        STATIC_ALIGN_REPORT,
        CONTEXT_ALIGN_REPORT,
        KEY_SEPARABILITY_REPORT,
        WRITE_TRAJECTORY_ROWS,
        REFERENCE_WRITE_ADAM_REPORT,
        ROUTE_TO_SCALAR_CLOSURE_REPORT,
        OUTPUT_ROUTE_CLOSURE_REPORT,
        LINE_INTEGRAL_REPORT,
        BRANCH_DECOMPOSITION_REPORT,
        CROSS_SEED_ROOT,
    ]:
        require_path(path)
    build_updated_chain()
    build_weight_birth()
    build_contextual_alignment()
    build_qk_optimizer_phase_structure()
    build_reference_write_optimizer_split()
    build_write_side_mechanism()
    build_write_functional_birth()
    build_cross_seed_qk_write_role_map()
    build_closure_boundary()
    build_proof_status()


if __name__ == "__main__":
    main()
