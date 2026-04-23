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
CROSS_SEED_ROOT = ROOT / "artifacts/runs/symbolic_kv_cross_seed_adam"


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> dict:
    return json.loads(require_path(path).read_text(encoding="utf-8"))


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


def build_updated_chain() -> None:
    labels = [
        ("Data relation", "latest write answers read"),
        ("Loss", "next-token CE on real batches"),
        ("AdamW state", "preconditioned current + momentum"),
        ("Weight geometry", "rank-8 W_QK support matcher"),
        ("Route", "support value > distractors"),
        ("Behavior", "answer margin improves"),
    ]
    x0, y, w, h, gap = 30, 110, 135, 76, 24
    parts = [text(30, 38, "From loss to lookup: the measured chain", "title"),
             text(30, 63, "Current paper claim: a support-value retrieval role is built by actual AdamW updates.", "subtitle")]
    for i, (a, b) in enumerate(labels):
        x = x0 + i * (w + gap)
        parts.append(rect(x, y, w, h))
        parts.append(text(x + w / 2, y + 30, a, "label", "middle"))
        parts.append(text(x + w / 2, y + 52, b, "tiny", "middle"))
        if i < len(labels) - 1:
            parts.append(line(x + w + 4, y + h / 2, x + w + gap - 6, y + h / 2, arrow=True))
    parts.append(text(30, 235, "The old question was \"which neuron matters?\" The current proof object is a differentiable route scalar C(theta).", "small"))
    parts.append(text(30, 258, "Supported: route growth, low-rank QK birth, exact AdamW update reconstruction, and 5-seed role replication.", "small"))
    parts.append(text(30, 281, "Open: full answer-margin closure and whether the same method scales beyond this symbolic model.", "small"))
    write_svg("updated_loss_to_lookup_chain.svg", 1000, 320, "\n".join(parts))


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


def build_adam_decomposition() -> None:
    report = load_json(ADAM_REPORT)
    rank_summary = report["summary"]["rank_summaries"][0]
    values = [
        ("actual route growth", rank_summary["sum_actual_rank_match_delta"], "#245f73"),
        ("reconstructed AdamW", rank_summary["sum_reconstructed_adamw_rank_delta"], "#4f7f54"),
        ("raw SGD", rank_summary["sum_raw_sgd_rank_delta"], "#b95f56"),
        ("clipped SGD", rank_summary["sum_clipped_sgd_rank_delta"], "#d69b3a"),
        ("Adam current", rank_summary["sum_adam_current_gradient_rank_delta"], "#7b5ea7"),
        ("Adam momentum", rank_summary["sum_adam_historical_momentum_rank_delta"], "#8f5a24"),
        ("weight decay", rank_summary["sum_weight_decay_rank_delta"], "#777777"),
    ]
    width, height = 980, 430
    left, top, bottom = 70, 96, 330
    max_abs = max(abs(v) for _, v, _ in values)
    zero = bottom - (0 - (-0.5)) / (max_abs + 0.5) * (bottom - top)
    parts = [text(32, 36, "Exact AdamW update decomposition for seed 7", "title"),
             text(32, 61, "Rank-8 L2H1 support-value route growth over 0 -> 6000.", "subtitle"),
             line(left, zero, 930, zero, color="#4a4741", width=1.2)]
    bw, gap = 92, 28
    for i, (label, val, color) in enumerate(values):
        x = left + i * (bw + gap)
        h = abs(val) / max_abs * 190
        y = zero - h if val >= 0 else zero
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{color}" opacity="0.9"/>')
        parts.append(text(x + bw / 2, y - 8 if val >= 0 else y + h + 18, f"{val:+.3f}", "small", "middle"))
        label_words = label.split()
        parts.append(text(x + bw / 2, 362, " ".join(label_words[:2]), "tiny", "middle"))
        if len(label_words) > 2:
            parts.append(text(x + bw / 2, 377, " ".join(label_words[2:]), "tiny", "middle"))
    actual = rank_summary["sum_actual_rank_match_delta"]
    raw_pct = rank_summary["sum_raw_sgd_rank_delta"] / actual * 100
    cur_pct = rank_summary["sum_adam_current_gradient_rank_delta"] / actual * 100
    mom_pct = rank_summary["sum_adam_historical_momentum_rank_delta"] / actual * 100
    wd_pct = rank_summary["sum_weight_decay_rank_delta"] / actual * 100
    parts.append(text(32, 407, f"Fractions of actual growth: raw SGD {raw_pct:.2f}%, Adam current {cur_pct:.1f}%, Adam momentum {mom_pct:.1f}%, weight decay {wd_pct:.1f}%.", "small"))
    write_svg("adam_route_growth_decomposition.svg", width, height, "\n".join(parts))


def classify_adam_report(path: Path) -> str:
    name = path.parent.name
    if name.startswith("runner_up_"):
        return "runner-up"
    if name.startswith("bottom_"):
        return "bottom"
    return "winner"


def build_cross_seed() -> None:
    selections = []
    for path in sorted(CROSS_SEED_ROOT.glob("seed_*/analysis/cross_seed_head_selection.json")):
        data = load_json(path)
        candidates = data["candidates"]
        if len(candidates) < 2:
            raise RuntimeError(f"Need at least two candidates in {path}")
        winner = candidates[0]
        runner = candidates[1]
        bottom = candidates[-1]
        selections.append({
            "seed": int(winner["seed"]),
            "winner": winner["head_label"],
            "runner": runner["head_label"],
            "bottom": bottom["head_label"],
            "score": float(winner["score"]),
            "sv_corr": float(winner["window_qk_match_separation_vs_qk_singular_value_top"]),
            "margin_corr": float(winner["window_qk_match_separation_vs_answer_margin"]),
        })
    if len(selections) != 5:
        raise RuntimeError(f"Expected 5 cross-seed selections, found {len(selections)}")

    actuals: dict[tuple[int, str], float] = {}
    raw_pct: list[float] = []
    sign_rates: list[float] = []
    for path in sorted(CROSS_SEED_ROOT.glob("seed_*/analysis/bilinear_qk_rank_adam_state_attribution/*/bilinear_qk_rank_adam_state_attribution_report.json")):
        seed_match = re.search(r"seed_(\d+)", str(path))
        if not seed_match:
            raise RuntimeError(f"Could not parse seed from {path}")
        seed = int(seed_match.group(1))
        role = classify_adam_report(path)
        report = load_json(path)
        row = report["summary"]["rank_summaries"][0]
        actuals[(seed, role)] = float(row["sum_actual_rank_match_delta"])
        if role == "winner":
            actual = float(row["sum_actual_rank_match_delta"])
            raw_pct.append(float(row["sum_raw_sgd_rank_delta"]) / actual * 100.0)
            sign_rates.append(float(row["reconstructed_adamw_sign_match_count"]) / float(row["sign_match_total"]) * 100.0)
    width, height = 980, 470
    parts = [text(32, 36, "Cross-seed validation: role repeats, head address changes", "title"),
             text(32, 61, "Each seed scans all heads, then checks winner / runner-up / bottom with exact Adam-state attribution.", "subtitle")]
    x_seed, x_win, x_run, x_bot, x_bar = 55, 135, 295, 455, 620
    y0, row_h = 108, 52
    parts.extend([
        text(x_seed, 92, "seed", "small"),
        text(x_win, 92, "winner", "small"),
        text(x_run, 92, "runner-up", "small"),
        text(x_bot, 92, "bottom", "small"),
        text(x_bar, 92, "actual route growth", "small"),
    ])
    scale = 55
    zero_x = x_bar + 120
    parts.append(line(zero_x, 92, zero_x, y0 + row_h * len(selections), color="#4a4741", width=1))
    for i, row in enumerate(selections):
        y = y0 + i * row_h
        seed = row["seed"]
        win_actual = actuals[(seed, "winner")]
        run_actual = actuals[(seed, "runner-up")]
        bot_actual = actuals[(seed, "bottom")]
        parts.append(text(x_seed, y, str(seed), "label"))
        parts.append(text(x_win, y, f"{row['winner']} ({win_actual:+.2f})", "small"))
        parts.append(text(x_run, y, f"{row['runner']} ({run_actual:+.2f})", "small"))
        parts.append(text(x_bot, y, f"{row['bottom']} ({bot_actual:+.2f})", "small"))
        for j, (val, color) in enumerate([(win_actual, "#4f7f54"), (run_actual, "#d69b3a"), (bot_actual, "#b95f56")]):
            bar_y = y - 14 + j * 12
            if val >= 0:
                parts.append(f'<rect x="{zero_x:.1f}" y="{bar_y:.1f}" width="{abs(val)*scale:.1f}" height="8" fill="{color}" opacity="0.9"/>')
            else:
                parts.append(f'<rect x="{zero_x + val*scale:.1f}" y="{bar_y:.1f}" width="{abs(val)*scale:.1f}" height="8" fill="{color}" opacity="0.9"/>')
    parts.append(text(620, 398, f"winner raw-SGD / actual mean: {sum(raw_pct)/len(raw_pct):.2f}%", "small"))
    parts.append(text(620, 421, f"Adam reconstruction sign match: {min(sign_rates):.1f}% -> {max(sign_rates):.1f}%", "small"))
    parts.append(text(32, 448, "Conclusion: the support-value retrieval role is stable; the named head that implements it is seed-dependent.", "small"))
    write_svg("cross_seed_role_replication.svg", width, height, "\n".join(parts))


def build_proof_status() -> None:
    rows = [
        ("Behavior learns lookup", "supported", "heldout-pair accuracy is high"),
        ("Feature families reveal structure", "supported as diagnostic", "not natural atoms"),
        ("Route scalar C(theta)", "supported", "support-value separation"),
        ("Weight-level QK birth", "supported", "low-rank W_QK growth"),
        ("Optimizer-level why", "supported for AdamW", "raw SGD tiny, Adam state large"),
        ("Cross-seed role replication", "supported", "5 seeds, address changes"),
        ("Full answer-margin closure", "open", "route family / OV side incomplete"),
        ("Scaling to LLMs", "open", "requires candidate filtering"),
    ]
    parts = [text(32, 36, "Proof status after the current experiments", "title"),
             text(32, 61, "The result is no longer only a trained-model story, but it is not a universal theorem.", "subtitle")]
    x0, y0, w, h = 55, 95, 840, 34
    for i, (claim, status, note) in enumerate(rows):
        y = y0 + i * 42
        cls = "ok" if status.startswith("supported") else "open"
        if status == "supported as diagnostic":
            cls = "warn"
        parts.append(rect(x0, y - 22, w, h, cls))
        parts.append(text(x0 + 18, y, claim, "label"))
        parts.append(text(x0 + 390, y, status, "small"))
        parts.append(text(x0 + 585, y, note, "small"))
    parts.append(text(55, 455, "The paper should claim a detailed mechanistic accounting for this task, not a theorem about all transformers.", "small"))
    write_svg("proof_status_ladder_updated.svg", 980, 490, "\n".join(parts))


def main() -> None:
    for path in [
        WEIGHT_SVD_CSV,
        QK_MATCH_CSV,
        ADAM_REPORT,
        STATIC_ALIGN_REPORT,
        CONTEXT_ALIGN_REPORT,
        KEY_SEPARABILITY_REPORT,
        CROSS_SEED_ROOT,
    ]:
        require_path(path)
    build_updated_chain()
    build_weight_birth()
    build_contextual_alignment()
    build_adam_decomposition()
    build_cross_seed()
    build_proof_status()


if __name__ == "__main__":
    main()
