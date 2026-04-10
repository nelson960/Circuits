from __future__ import annotations

import html
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from circut.io import ensure_parent_dir, iter_jsonl, read_json, write_json


def _import_matplotlib() -> tuple[Any, Any]:
    from circut.analysis.shared_feature_dynamics import _import_matplotlib as _shared_import_matplotlib

    return _shared_import_matplotlib()


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required analysis artifact not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, found something else: {path}")
    return path


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"No rows found in JSONL file: {path}")
    return rows


def _format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _relative_href(*, source_dir: Path, target_path: Path) -> str:
    return os.path.relpath(target_path, source_dir).replace(os.sep, "/")


def _render_link(*, source_dir: Path, target_path: Path, label: str | None = None) -> str:
    href = _relative_href(source_dir=source_dir, target_path=target_path)
    text = html.escape(label or target_path.name)
    return f'<a href="{html.escape(href)}">{text}</a>'


def _find_single(paths: list[Path], label: str, stage_dir: Path) -> Path | None:
    if not paths:
        return None
    if len(paths) > 1:
        raise ValueError(f"Expected at most one {label} file in {stage_dir}, found {len(paths)}.")
    return paths[0]


def _summarize_best_checkpoint(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    best_row = max(rows, key=lambda row: float(row[metric_name]))
    return {
        "step": int(best_row["step"]),
        "value": float(best_row[metric_name]),
        "checkpoint_path": str(best_row["checkpoint_path"]),
    }


def _make_timeline_plot(
    *,
    checkpoint_rows: list[dict[str, Any]],
    birth_windows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    _, plt = _import_matplotlib()
    steps = [int(row["step"]) for row in checkpoint_rows]
    answer = [float(row["answer_accuracy"]) for row in checkpoint_rows]
    heldout = [float(row["heldout_answer_accuracy"]) for row in checkpoint_rows]
    q_values = [float(row["q"]) for row in checkpoint_rows]
    r_values = [float(row["r"]) for row in checkpoint_rows]
    w_values = [float(row["w"]) for row in checkpoint_rows]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(steps, answer, label="Probe answer", linewidth=2.0)
    axes[0].plot(steps, heldout, label="Probe heldout", linewidth=2.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Checkpoint behavior over training")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, q_values, label="Q", linewidth=1.8)
    axes[1].plot(steps, r_values, label="R", linewidth=1.8)
    axes[1].plot(steps, w_values, label="W", linewidth=1.8)
    axes[1].set_xlabel("Checkpoint step")
    axes[1].set_ylabel("Metric value")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    for window in birth_windows:
        window_spec = window["window"]
        start_step = int(window_spec["start_step"])
        end_step = int(window_spec["end_step"])
        for axis in axes:
            axis.axvspan(start_step, end_step, color="#dbeafe", alpha=0.35)

    ensure_parent_dir(output_path)
    fig.tight_layout()
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _render_card(title: str, value: str, subtitle: str | None = None) -> str:
    subtitle_html = f'<div class="card-subtitle">{html.escape(subtitle)}</div>' if subtitle is not None else ""
    return (
        '<div class="card">'
        f'<div class="card-title">{html.escape(title)}</div>'
        f'<div class="card-value">{html.escape(value)}</div>'
        f"{subtitle_html}"
        "</div>"
    )


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return '<p class="empty">No rows.</p>'
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>"


def _summarize_compare_report(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    source_summaries: list[dict[str, Any]] = []
    for source in payload["source_comparisons"]:
        residual_rows = source["residual_patch_results"]
        if not residual_rows:
            raise ValueError(f"No residual_patch_results found in compare report: {path}")
        best_residual = max(
            residual_rows,
            key=lambda row: float(row["delta_vs_target_answer_accuracy"]) + float(row["delta_vs_target_heldout_answer_accuracy"]),
        )
        source_summaries.append(
            {
                "source_step": int(source["source_step"]),
                "best_residual_stage": str(best_residual["stage"]),
                "best_residual_answer_delta": float(best_residual["delta_vs_target_answer_accuracy"]),
                "best_residual_heldout_delta": float(best_residual["delta_vs_target_heldout_answer_accuracy"]),
            }
        )
    return {
        "path": str(path),
        "target_step": int(payload["target_step"]),
        "source_summaries": source_summaries,
    }


def _stage_plot_map(stage_dir: Path) -> dict[str, Path]:
    stage_name = stage_dir.name
    plot_paths = {
        "trajectory_plot": stage_dir / "trajectories" / "plots" / f"{stage_name}_feature_trajectory_topk.svg",
        "heatmap_plot": stage_dir / "trajectories" / "plots" / f"{stage_name}_feature_heatmap.svg",
        "birth_plot": stage_dir / "births" / "plots" / "feature_birth_raster.svg",
        "family_similarity_plot": stage_dir / "families" / "plots" / "feature_family_similarity_heatmap.svg",
        "family_trajectory_plot": stage_dir / "families" / "plots" / "feature_family_trajectories.svg",
    }
    compare_plots = sorted(stage_dir.glob("feature_compare_*_bar.svg"))
    family_compare_plots = sorted((stage_dir / "families").glob("feature_family_compare_*_bar.svg"))
    if compare_plots:
        plot_paths["feature_compare_plot"] = compare_plots[0]
    if family_compare_plots:
        plot_paths["family_compare_plot"] = family_compare_plots[0]
    return plot_paths


def _summarize_stage(stage_dir: Path) -> dict[str, Any]:
    basis_manifest_path = _require_file(stage_dir / "shared_feature_basis.json")
    basis_features_path = _require_file(stage_dir / "shared_feature_basis_features.json")
    trajectory_summary_path = _require_file(stage_dir / "trajectories" / "feature_checkpoint_summary.json")
    split_profiles_path = _require_file(stage_dir / "trajectories" / "feature_split_profiles.json")
    births_summary_path = _require_file(stage_dir / "births" / "feature_birth_summary.json")
    births_path = _require_file(stage_dir / "births" / "feature_births.json")
    families_path = _require_file(stage_dir / "families" / "feature_families.json")
    family_trajectories_path = _require_file(stage_dir / "families" / "feature_family_trajectories.jsonl")

    basis_manifest = read_json(basis_manifest_path)
    basis_features = read_json(basis_features_path)
    trajectory_summary = read_json(trajectory_summary_path)
    split_profiles = read_json(split_profiles_path)
    births_summary = read_json(births_summary_path)
    births_payload = read_json(births_path)
    families_payload = read_json(families_path)

    feature_compare_paths = sorted(path for path in stage_dir.glob("feature_compare_*.json"))
    feature_patch_paths = sorted(path for path in stage_dir.glob("feature_patch_*.json"))
    feature_lineage_paths = sorted(
        path
        for path in stage_dir.glob("feature_lineage_*.json")
        if not path.name.endswith("_graph.json")
    )
    family_compare_paths = sorted(path for path in (stage_dir / "families").glob("feature_family_compare_*.json"))
    family_patch_paths = sorted(path for path in (stage_dir / "families").glob("feature_family_patch_*.json"))

    missing_optional: list[str] = []
    if not feature_compare_paths:
        missing_optional.append("feature_compare")
    if not feature_patch_paths:
        missing_optional.append("feature_patch")
    if not feature_lineage_paths:
        missing_optional.append("feature_lineage")
    if not family_compare_paths:
        missing_optional.append("feature_family_compare")
    if not family_patch_paths:
        missing_optional.append("feature_family_patch")

    family_compare_payload = None
    family_compare_path = _find_single(family_compare_paths, "family compare", stage_dir / "families")
    if family_compare_path is not None:
        family_compare_payload = read_json(family_compare_path)

    family_patch_payloads = [read_json(path) for path in family_patch_paths]
    feature_patch_payloads = [read_json(path) for path in feature_patch_paths]
    feature_lineage_payloads = [read_json(path) for path in feature_lineage_paths]

    plots = {
        name: str(path)
        for name, path in _stage_plot_map(stage_dir).items()
        if path.exists()
    }
    missing_plot_names = sorted(set(_stage_plot_map(stage_dir)) - set(plots))
    if missing_plot_names:
        missing_optional.extend(f"plot:{name}" for name in missing_plot_names)

    top_families_by_useful_delta = []
    if family_compare_payload is not None:
        top_families_by_useful_delta = list(family_compare_payload["top_families"]["by_useful_delta"][:5])

    top_family_members = [row for row in families_payload["families"] if int(row["size"]) > 1][:5]

    top_lineage = None
    if feature_lineage_payloads:
        top_lineage = feature_lineage_payloads[0]

    return {
        "stage_name": str(basis_manifest["stage_name"]),
        "stage_dir": str(stage_dir),
        "basis_manifest_path": str(basis_manifest_path),
        "basis_features_path": str(basis_features_path),
        "trajectory_summary_path": str(trajectory_summary_path),
        "split_profiles_path": str(split_profiles_path),
        "births_summary_path": str(births_summary_path),
        "births_path": str(births_path),
        "families_path": str(families_path),
        "family_trajectories_path": str(family_trajectories_path),
        "basis_manifest": basis_manifest,
        "basis_features": basis_features,
        "trajectory_summary": trajectory_summary,
        "split_profiles": split_profiles,
        "births_summary": births_summary,
        "births_payload": births_payload,
        "families_payload": families_payload,
        "top_families_by_useful_delta": top_families_by_useful_delta,
        "top_multi_feature_families": top_family_members,
        "feature_compare_paths": [str(path) for path in feature_compare_paths],
        "feature_patch_paths": [str(path) for path in feature_patch_paths],
        "feature_lineage_paths": [str(path) for path in feature_lineage_paths],
        "family_compare_paths": [str(path) for path in family_compare_paths],
        "family_patch_paths": [str(path) for path in family_patch_paths],
        "feature_patch_payloads": feature_patch_payloads,
        "family_patch_payloads": family_patch_payloads,
        "feature_lineage_payloads": feature_lineage_payloads,
        "top_lineage": top_lineage,
        "plots": plots,
        "missing_optional_artifacts": sorted(set(missing_optional)),
    }


def _render_stage_section(*, report_dir: Path, stage_summary: dict[str, Any]) -> str:
    stage_name = stage_summary["stage_name"]
    basis_manifest = stage_summary["basis_manifest"]
    births_summary = stage_summary["births_summary"]
    plots = stage_summary["plots"]
    cards = "".join(
        [
            _render_card("Basis ID", str(basis_manifest["basis_id"])),
            _render_card("Features", str(basis_manifest["num_features"])),
            _render_card("Input Dim", str(basis_manifest["input_dim"])),
            _render_card("Explained Variance", _format_float(float(basis_manifest["fit_metrics"]["explained_variance"]))),
            _render_card("Active Fraction", _format_float(float(basis_manifest["fit_metrics"]["active_fraction"]))),
            _render_card(
                "Earliest Useful Birth",
                str(births_summary["earliest_useful_births"][0]["birth_step"]) if births_summary["earliest_useful_births"] else "none",
            ),
        ]
    )

    plot_blocks: list[str] = []
    for label, plot_key in [
        ("Feature trajectories", "trajectory_plot"),
        ("Feature heatmap", "heatmap_plot"),
        ("Birth raster", "birth_plot"),
        ("Family similarity", "family_similarity_plot"),
        ("Family trajectories", "family_trajectory_plot"),
        ("Feature compare", "feature_compare_plot"),
        ("Family compare", "family_compare_plot"),
    ]:
        if plot_key in plots:
            plot_path = Path(plots[plot_key])
            plot_blocks.append(
                '<div class="plot-card">'
                f'<div class="plot-title">{html.escape(label)}</div>'
                f'<img src="{html.escape(_relative_href(source_dir=report_dir, target_path=plot_path))}" alt="{html.escape(label)}" />'
                f'<div class="artifact-link">{_render_link(source_dir=report_dir, target_path=plot_path)}</div>'
                '</div>'
            )

    top_family_rows = []
    for family in stage_summary["top_families_by_useful_delta"]:
        top_family_rows.append(
            [
                html.escape(str(family["family_id"])),
                html.escape(str(family["representative_feature_id"])),
                html.escape(str(family["size"])),
                html.escape(_format_float(float(family["useful_delta"]))),
                html.escape(_format_float(float(family["heldout_gap_mean_delta"]))),
                html.escape(_format_float(float(family["correctness_gap_mean_delta"]))),
                html.escape(_format_float(float(family["structural_ood_gap_mean_delta"]))),
            ]
        )

    family_patch_rows = []
    for payload in stage_summary["family_patch_payloads"]:
        family_patch_rows.append(
            [
                html.escape(",".join(str(item) for item in payload["family_ids"])),
                html.escape(",".join(str(item) for item in payload["resolved_feature_ids"])),
                html.escape(_format_float(float(payload["deltas"]["answer_accuracy"]))),
                html.escape(_format_float(float(payload["deltas"]["heldout_answer_accuracy"]))),
                html.escape(_format_float(float(payload["deltas"]["structural_ood_answer_accuracy"]))),
            ]
        )

    feature_patch_rows = []
    for payload in stage_summary["feature_patch_payloads"]:
        feature_patch_rows.append(
            [
                html.escape(",".join(str(item) for item in payload["feature_ids"])),
                html.escape(_format_float(float(payload["deltas"]["answer_accuracy"]))),
                html.escape(_format_float(float(payload["deltas"]["heldout_answer_accuracy"]))),
                html.escape(_format_float(float(payload["deltas"]["structural_ood_answer_accuracy"]))),
            ]
        )

    lineage_html = '<p class="empty">No lineage output found.</p>'
    top_lineage = stage_summary["top_lineage"]
    if top_lineage is not None:
        head_rows = [
            [
                html.escape(str(row["feature_id"])),
                html.escape(f"L{row['layer']}H{row['head']}"),
                html.escape(_format_float(float(row["mean_abs_feature_shift"]))),
                html.escape(_format_float(float(row["mean_activation_delta"]))),
            ]
            for row in top_lineage["head_effects"][:6]
        ]
        mlp_rows = [
            [
                html.escape(str(row["feature_id"])),
                html.escape(f"L{row['layer']}"),
                html.escape(_format_float(float(row["mean_abs_feature_shift"]))),
                html.escape(_format_float(float(row["mean_activation_delta"]))),
            ]
            for row in top_lineage["mlp_effects"][:6]
        ]
        neuron_rows = [
            [
                html.escape(str(row["feature_id"])),
                html.escape(f"L{row['layer']}"),
                html.escape(",".join(str(item) for item in row["neurons"])),
                html.escape(_format_float(float(row["mean_abs_feature_shift"]))),
            ]
            for row in top_lineage["neuron_group_effects"][:6]
        ]
        lineage_html = (
            '<div class="subsection"><h4>Lineage: top heads</h4>'
            f'{_render_table(["Feature", "Head", "Abs shift", "Activation delta"], head_rows)}</div>'
            '<div class="subsection"><h4>Lineage: top MLPs</h4>'
            f'{_render_table(["Feature", "Layer", "Abs shift", "Activation delta"], mlp_rows)}</div>'
            '<div class="subsection"><h4>Lineage: top neuron groups</h4>'
            f'{_render_table(["Feature", "Layer", "Neurons", "Abs shift"], neuron_rows)}</div>'
        )

    artifact_links = [
        _render_link(source_dir=report_dir, target_path=Path(stage_summary["basis_manifest_path"]), label="basis manifest"),
        _render_link(source_dir=report_dir, target_path=Path(stage_summary["basis_features_path"]), label="basis features"),
        _render_link(source_dir=report_dir, target_path=Path(stage_summary["trajectory_summary_path"]), label="trajectory summary"),
        _render_link(source_dir=report_dir, target_path=Path(stage_summary["births_summary_path"]), label="birth summary"),
        _render_link(source_dir=report_dir, target_path=Path(stage_summary["families_path"]), label="families"),
    ]
    artifact_links.extend(
        _render_link(source_dir=report_dir, target_path=Path(path), label=Path(path).name)
        for path in stage_summary["feature_compare_paths"] + stage_summary["feature_patch_paths"] + stage_summary["family_compare_paths"] + stage_summary["family_patch_paths"] + stage_summary["feature_lineage_paths"]
    )

    missing_html = ""
    if stage_summary["missing_optional_artifacts"]:
        missing_items = "".join(f"<li>{html.escape(item)}</li>" for item in stage_summary["missing_optional_artifacts"])
        missing_html = f'<div class="warning"><strong>Missing optional artifacts</strong><ul>{missing_items}</ul></div>'

    return (
        f'<section class="stage-section" id="{html.escape(stage_name)}">'
        f'<h2>Stage: {html.escape(stage_name)}</h2>'
        f'<div class="card-grid">{cards}</div>'
        f"{missing_html}"
        '<div class="plot-grid">' + "".join(plot_blocks) + '</div>'
        '<div class="subsection"><h3>Top families by useful delta</h3>'
        f'{_render_table(["Family", "Representative", "Size", "Useful delta", "Heldout delta", "Correctness delta", "Structural OOD delta"], top_family_rows)}</div>'
        '<div class="subsection"><h3>Family patches</h3>'
        f'{_render_table(["Family IDs", "Features", "Answer Δ", "Heldout Δ", "Structural OOD Δ"], family_patch_rows)}</div>'
        '<div class="subsection"><h3>Feature patches</h3>'
        f'{_render_table(["Features", "Answer Δ", "Heldout Δ", "Structural OOD Δ"], feature_patch_rows)}</div>'
        f"{lineage_html}"
        '<div class="subsection"><h3>Artifacts</h3><p>' + " | ".join(artifact_links) + "</p></div>"
        '</section>'
    )


def build_analysis_report(
    *,
    analysis_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")
    if not analysis_dir.is_dir():
        raise ValueError(f"Expected analysis_dir to be a directory: {analysis_dir}")

    report_path = output_dir / "report.html"
    manifest_path = output_dir / "report_manifest.json"
    timeline_plot_path = output_dir / "plots" / "checkpoint_timeline.svg"
    for path in [report_path, manifest_path, timeline_plot_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing report artifact without --overwrite: {path}")

    metrics_path = _require_file(analysis_dir / "checkpoint_metrics.jsonl")
    metrics_summary_path = _require_file(analysis_dir / "checkpoint_metrics_summary.json")
    birth_analysis_path = _require_file(analysis_dir / "birth_window_analysis.json")

    checkpoint_rows = _read_rows(metrics_path)
    metrics_summary = read_json(metrics_summary_path)
    birth_analysis = read_json(birth_analysis_path)

    shared_features_dir = analysis_dir / "shared_features"
    if not shared_features_dir.exists():
        raise FileNotFoundError(f"shared_features directory not found: {shared_features_dir}")
    stage_dirs = sorted(path for path in shared_features_dir.iterdir() if path.is_dir())
    if not stage_dirs:
        raise ValueError(f"No stage directories found in {shared_features_dir}")

    birth_windows = birth_analysis["birth_windows"]
    timeline_plot_path = _make_timeline_plot(
        checkpoint_rows=checkpoint_rows,
        birth_windows=birth_windows,
        output_path=timeline_plot_path,
    )

    stage_summaries = [_summarize_stage(stage_dir) for stage_dir in stage_dirs]
    compare_reports = [
        _summarize_compare_report(path)
        for path in sorted(analysis_dir.glob("compare_*.json"))
    ]

    best_answer = _summarize_best_checkpoint(checkpoint_rows, "answer_accuracy")
    best_heldout = _summarize_best_checkpoint(checkpoint_rows, "heldout_answer_accuracy")
    final_row = checkpoint_rows[-1]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_dir": str(analysis_dir),
        "metrics_path": str(metrics_path),
        "metrics_summary_path": str(metrics_summary_path),
        "birth_analysis_path": str(birth_analysis_path),
        "timeline_plot_path": str(timeline_plot_path),
        "num_checkpoints": int(metrics_summary["num_checkpoints"]),
        "best_probe_answer": best_answer,
        "best_probe_heldout": best_heldout,
        "final_checkpoint": {
            "step": int(final_row["step"]),
            "answer_accuracy": float(final_row["answer_accuracy"]),
            "heldout_answer_accuracy": float(final_row["heldout_answer_accuracy"]),
            "q": float(final_row["q"]),
            "r": float(final_row["r"]),
            "w": float(final_row["w"]),
        },
        "birth_windows": birth_windows,
        "compare_reports": compare_reports,
        "stages": [
            {
                "stage_name": stage_summary["stage_name"],
                "stage_dir": stage_summary["stage_dir"],
                "basis_manifest_path": stage_summary["basis_manifest_path"],
                "trajectory_summary_path": stage_summary["trajectory_summary_path"],
                "births_summary_path": stage_summary["births_summary_path"],
                "families_path": stage_summary["families_path"],
                "missing_optional_artifacts": stage_summary["missing_optional_artifacts"],
            }
            for stage_summary in stage_summaries
        ],
    }
    write_json(manifest_path, manifest)

    overview_cards = "".join(
        [
            _render_card("Checkpoints", str(metrics_summary["num_checkpoints"])),
            _render_card("Best probe answer", _format_percent(best_answer["value"]), f"step {best_answer['step']}"),
            _render_card("Best probe heldout", _format_percent(best_heldout["value"]), f"step {best_heldout['step']}"),
            _render_card("Final step", str(final_row["step"]), f"answer {_format_percent(float(final_row['answer_accuracy']))}"),
            _render_card("Final Q/R/W", f"{_format_float(float(final_row['q']))} / {_format_float(float(final_row['r']))} / {_format_float(float(final_row['w']))}"),
        ]
    )

    birth_window_rows = []
    for item in birth_windows:
        window = item["window"]
        first = item["behavior"]["first"]
        last = item["behavior"]["last"]
        birth_window_rows.append(
            [
                html.escape(f"{window['start_step']} -> {window['end_step']}"),
                html.escape(_format_percent(float(first["answer_accuracy"]))),
                html.escape(_format_percent(float(last["answer_accuracy"]))),
                html.escape(_format_percent(float(first["heldout_answer_accuracy"]))),
                html.escape(_format_percent(float(last["heldout_answer_accuracy"]))),
                html.escape(_format_float(float(last["r"]))),
                html.escape(_format_float(float(last["w"]))),
            ]
        )

    compare_report_sections = []
    for report in compare_reports:
        rows = [
            [
                html.escape(str(item["source_step"])),
                html.escape(str(item["best_residual_stage"])),
                html.escape(_format_float(float(item["best_residual_answer_delta"]))),
                html.escape(_format_float(float(item["best_residual_heldout_delta"]))),
            ]
            for item in report["source_summaries"]
        ]
        compare_report_sections.append(
            '<div class="subsection">'
            f"<h3>Target step {report['target_step']}</h3>"
            f"<p>{_render_link(source_dir=output_dir, target_path=Path(report['path']), label=Path(report['path']).name)}</p>"
            f'{_render_table(["Source step", "Best residual stage", "Answer Δ vs target", "Heldout Δ vs target"], rows)}'
            "</div>"
        )

    nav_links = "".join(
        [
            '<a href="#overview">Overview</a>',
            '<a href="#birth-windows">Birth windows</a>',
            '<a href="#checkpoint-comparisons">Checkpoint comparisons</a>',
            *[
                f'<a href="#{html.escape(stage_summary["stage_name"])}">{html.escape(stage_summary["stage_name"])}</a>'
                for stage_summary in stage_summaries
            ],
        ]
    )

    html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>circut analysis report</title>
  <style>
    body {{
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      margin: 0;
      background: #f7f7f2;
      color: #111827;
    }}
    .page {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      background: linear-gradient(140deg, #fffef7 0%, #eef5ff 100%);
      border: 1px solid #d1d5db;
      border-radius: 18px;
      padding: 24px;
      margin-bottom: 24px;
    }}
    .hero h1 {{
      margin: 0 0 10px 0;
      font-size: 34px;
      letter-spacing: -0.03em;
    }}
    .hero p {{
      margin: 8px 0;
      max-width: 1000px;
      line-height: 1.45;
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 18px;
    }}
    nav a {{
      text-decoration: none;
      color: #0f4c81;
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 14px;
    }}
    section {{
      background: white;
      border: 1px solid #d1d5db;
      border-radius: 18px;
      padding: 22px;
      margin-bottom: 24px;
    }}
    h2 {{
      margin-top: 0;
      margin-bottom: 14px;
      font-size: 26px;
      letter-spacing: -0.02em;
    }}
    h3 {{
      margin-bottom: 10px;
      font-size: 20px;
    }}
    h4 {{
      margin-bottom: 8px;
      font-size: 17px;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card {{
      border: 1px solid #d1d5db;
      border-radius: 14px;
      padding: 14px;
      background: #fcfcfa;
    }}
    .card-title {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #6b7280;
      margin-bottom: 6px;
    }}
    .card-value {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .card-subtitle {{
      margin-top: 6px;
      color: #374151;
      font-size: 14px;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .plot-card {{
      border: 1px solid #d1d5db;
      border-radius: 14px;
      padding: 12px;
      background: #fcfcfa;
    }}
    .plot-title {{
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .plot-card img {{
      width: 100%;
      height: auto;
      display: block;
      background: white;
      border-radius: 8px;
    }}
    .artifact-link {{
      margin-top: 10px;
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 12px;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #f8fafc;
      font-weight: 700;
    }}
    .empty {{
      color: #6b7280;
      font-style: italic;
    }}
    .warning {{
      border: 1px solid #f59e0b;
      background: #fffbeb;
      border-radius: 12px;
      padding: 10px 14px;
      margin-bottom: 16px;
    }}
    .subsection {{
      margin-top: 18px;
    }}
    code {{
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
      background: #f3f4f6;
      padding: 2px 5px;
      border-radius: 6px;
      font-size: 0.95em;
    }}
    a {{
      color: #0f4c81;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>circut analysis report</h1>
      <p>
        This report is a read-only interpretation surface over the existing CLI artifacts in
        <code>{html.escape(str(analysis_dir))}</code>. It does not recompute the science. It assembles the checkpoint sweep,
        birth-window analysis, shared-feature trajectories, family comparisons, patches, and lineage outputs into one human-readable view.
      </p>
      <p>
        Manifest: {_render_link(source_dir=output_dir, target_path=manifest_path, label='report_manifest.json')} |
        Sweep metrics: {_render_link(source_dir=output_dir, target_path=metrics_path, label='checkpoint_metrics.jsonl')} |
        Birth analysis: {_render_link(source_dir=output_dir, target_path=birth_analysis_path, label='birth_window_analysis.json')}
      </p>
      <nav>{nav_links}</nav>
    </div>

    <section id="overview">
      <h2>Overview</h2>
      <div class="card-grid">{overview_cards}</div>
      <div class="plot-grid">
        <div class="plot-card">
          <div class="plot-title">Checkpoint timeline</div>
          <img src="{html.escape(_relative_href(source_dir=output_dir, target_path=timeline_plot_path))}" alt="Checkpoint timeline" />
          <div class="artifact-link">{_render_link(source_dir=output_dir, target_path=timeline_plot_path)}</div>
        </div>
      </div>
    </section>

    <section id="birth-windows">
      <h2>Birth windows</h2>
      {_render_table(["Window", "Answer start", "Answer end", "Heldout start", "Heldout end", "R end", "W end"], birth_window_rows)}
    </section>

    <section id="checkpoint-comparisons">
      <h2>Checkpoint comparisons</h2>
      {''.join(compare_report_sections) if compare_report_sections else '<p class="empty">No birth-window compare reports found.</p>'}
    </section>

    {''.join(_render_stage_section(report_dir=output_dir, stage_summary=stage_summary) for stage_summary in stage_summaries)}
  </div>
</body>
</html>
"""

    ensure_parent_dir(report_path)
    report_path.write_text(html_output, encoding="utf-8")
    return report_path, manifest_path, timeline_plot_path
