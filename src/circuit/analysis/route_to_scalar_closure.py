from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.route_to_margin_closure import fit_route_to_margin_closure
from circuit.analysis.shared_feature_dynamics import _import_matplotlib
from circuit.io import iter_jsonl, write_json, write_jsonl


ROUTE_TO_SCALAR_CLOSURE_SCHEMA_VERSION = 1
ROUTE_TO_SCALAR_SWITCH_BUCKETS = ["all", "same_competitor", "competitor_switch"]
DEFAULT_ROUTE_TO_SCALAR_SCALARS = [
    "moving_answer_margin",
    "fixed_source_competitor_margin",
    "fixed_target_competitor_margin",
    "correct_value_logit",
    "source_best_wrong_logit",
    "target_best_wrong_logit",
    "negative_answer_loss",
]


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"Cannot compute mean for empty values: {label}")
    return sum(values) / float(len(values))


def _resolve_unique(
    *,
    values: list[str] | None,
    default_values: list[str],
    allowed_values: list[str] | None,
    label: str,
) -> list[str]:
    raw_values = default_values if values is None else values
    if not raw_values:
        raise ValueError(f"{label} must not be empty.")
    resolved: list[str] = []
    for value in raw_values:
        if allowed_values is not None and value not in allowed_values:
            raise ValueError(f"Unsupported {label} {value!r}; expected one of {allowed_values}.")
        if value not in resolved:
            resolved.append(value)
    return resolved


def _switch_bucket_matches(row: dict[str, Any], switch_bucket: str) -> bool:
    if switch_bucket == "all":
        return True
    if switch_bucket == "same_competitor":
        return not bool(row["competitor_switched"])
    if switch_bucket == "competitor_switch":
        return bool(row["competitor_switched"])
    raise ValueError(f"Unsupported switch bucket {switch_bucket!r}; expected one of {ROUTE_TO_SCALAR_SWITCH_BUCKETS}.")


def _load_route_closure_rows(
    *,
    route_closure_rows_path: Path,
    value_tolerance: float,
) -> tuple[dict[tuple[int, int, str], dict[str, float]], list[str]]:
    if value_tolerance < 0.0:
        raise ValueError("value_tolerance must be non-negative.")
    rows_by_key: dict[tuple[int, int, str], dict[str, float]] = {}
    route_labels: set[str] = set()
    row_count = 0
    for row in iter_jsonl(route_closure_rows_path):
        row_count += 1
        key = (int(row["source_step"]), int(row["target_step"]), str(row["pair_id"]))
        deltas = row.get("route_score_deltas")
        if not isinstance(deltas, dict) or not deltas:
            raise RuntimeError(f"Route closure row is missing route_score_deltas: {key}")
        numeric_deltas = {str(label): float(value) for label, value in deltas.items()}
        if key in rows_by_key:
            previous = rows_by_key[key]
            if set(previous) != set(numeric_deltas):
                raise RuntimeError(f"Route closure duplicate label mismatch for {key}.")
            mismatches = [
                label
                for label in previous
                if abs(float(previous[label]) - float(numeric_deltas[label])) > value_tolerance
            ]
            if mismatches:
                raise RuntimeError(
                    f"Route closure duplicate delta mismatch for {key}: {mismatches[:10]} "
                    f"(tolerance={value_tolerance})."
                )
        rows_by_key[key] = numeric_deltas
        route_labels.update(numeric_deltas)
    if row_count == 0:
        raise RuntimeError(f"Route closure rows file is empty: {route_closure_rows_path}")
    if not route_labels:
        raise RuntimeError(f"No route labels found in route closure rows: {route_closure_rows_path}")
    return rows_by_key, sorted(route_labels)


def _load_scalar_pair_rows(path: Path) -> list[dict[str, Any]]:
    rows = [row for row in iter_jsonl(path)]
    if not rows:
        raise RuntimeError(f"Scalar pair rows file is empty: {path}")
    return rows


def _route_deltas_for_scalar_row(
    *,
    row: dict[str, Any],
    route_closure_rows: dict[tuple[int, int, str], dict[str, float]],
    route_labels: list[str],
) -> dict[str, float]:
    source_step = int(row["source_step"])
    target_step = int(row["target_step"])
    pair_id = str(row["pair_id"])
    key = (source_step, target_step, pair_id)
    if key not in route_closure_rows:
        raise KeyError(f"Missing route closure deltas for scalar row {source_step}->{target_step} pair={pair_id}.")
    row_deltas = route_closure_rows[key]
    missing = [route_label for route_label in route_labels if route_label not in row_deltas]
    if missing:
        raise KeyError(f"Missing route labels for scalar row {source_step}->{target_step} pair={pair_id}: {missing[:10]}")
    return {route_label: float(row_deltas[route_label]) for route_label in route_labels}


def _build_scalar_observations(
    *,
    scalar_pair_rows: list[dict[str, Any]],
    route_closure_rows: dict[tuple[int, int, str], dict[str, float]],
    route_labels: list[str],
    scalar_names: list[str],
    switch_buckets: list[str],
    margin_side: str | None,
    pair_types: list[str] | None,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    pair_type_filter = None if pair_types is None else set(pair_types)
    for row in scalar_pair_rows:
        if margin_side is not None and str(row["margin_side"]) != margin_side:
            continue
        if pair_type_filter is not None and str(row["pair_type"]) not in pair_type_filter:
            continue
        scalars = row.get("scalars")
        if not isinstance(scalars, dict):
            raise RuntimeError(f"Scalar pair row is missing a scalars object: {row.get('interval_pair_id')}")
        route_deltas = _route_deltas_for_scalar_row(
            row=row,
            route_closure_rows=route_closure_rows,
            route_labels=route_labels,
        )
        for scalar_name in scalar_names:
            if scalar_name not in scalars:
                raise KeyError(f"Scalar {scalar_name!r} not found in pair row {row.get('interval_pair_id')}.")
            scalar_payload = scalars[scalar_name]
            actual_delta = float(scalar_payload["delta"])
            for switch_bucket in switch_buckets:
                if not _switch_bucket_matches(row, switch_bucket):
                    continue
                observations.append(
                    {
                        "source_step": int(row["source_step"]),
                        "target_step": int(row["target_step"]),
                        "step_gap": int(row["step_gap"]),
                        "pair_id": str(row["pair_id"]),
                        "interval_pair_id": str(row["interval_pair_id"]),
                        "split": str(row["split"]),
                        "pair_type": str(row["pair_type"]),
                        "margin_side": str(row["margin_side"]),
                        "switch_bucket": switch_bucket,
                        "competitor_switched": bool(row["competitor_switched"]),
                        "scalar_name": scalar_name,
                        "source_scalar": float(scalar_payload["source"]),
                        "target_scalar": float(scalar_payload["target"]),
                        "actual_scalar_delta": actual_delta,
                        "route_score_deltas": dict(route_deltas),
                    }
                )
    if not observations:
        raise RuntimeError("No route-to-scalar closure observations survived the filters.")
    return observations


def _fit_scalar_bucket(
    *,
    observations: list[dict[str, Any]],
    route_labels: list[str],
    scalar_name: str,
    switch_bucket: str,
    fit_intercept: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        row
        for row in observations
        if str(row["scalar_name"]) == scalar_name and str(row["switch_bucket"]) == switch_bucket
    ]
    if not rows:
        raise RuntimeError(f"No observations for scalar={scalar_name} switch_bucket={switch_bucket}.")
    route_delta_columns = {
        label: [float(row["route_score_deltas"][label]) for row in rows]
        for label in route_labels
    }
    scalar_deltas = [float(row["actual_scalar_delta"]) for row in rows]
    fit = fit_route_to_margin_closure(
        route_delta_columns=route_delta_columns,
        margin_deltas=scalar_deltas,
        fit_intercept=fit_intercept,
    )
    closure_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        fitted = dict(row)
        fitted["predicted_scalar_delta"] = float(fit["predicted_values"][index])
        fitted["closure_residual"] = float(fit["residual_values"][index])
        closure_rows.append(fitted)
    coefficient_rows: list[dict[str, Any]] = []
    for route_label in route_labels:
        mean_delta = _mean(route_delta_columns[route_label], label=f"{scalar_name}/{switch_bucket}/{route_label}")
        coefficient = float(fit["coefficients"][route_label])
        coefficient_rows.append(
            {
                "scalar_name": scalar_name,
                "switch_bucket": switch_bucket,
                "route_label": route_label,
                "coefficient": coefficient,
                "mean_route_score_delta": mean_delta,
                "mean_contribution": coefficient * mean_delta,
            }
        )
    fit_summary = {
        key: value for key, value in fit.items() if key not in {"predicted_values", "residual_values"}
    }
    return fit_summary, closure_rows, coefficient_rows


def _r_squared(*, actual_values: list[float], predicted_values: list[float]) -> float | None:
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual_values and predicted_values must have the same length.")
    mean_actual = _mean(actual_values, label="actual r squared")
    sst = sum((value - mean_actual) ** 2 for value in actual_values)
    if sst <= 1.0e-12:
        return None
    sse = sum((actual - predicted) ** 2 for actual, predicted in zip(actual_values, predicted_values, strict=True))
    return 1.0 - (sse / sst)


def _summarize_route_to_scalar_closure(
    *,
    closure_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not closure_rows:
        raise ValueError("Cannot summarize route-to-scalar closure without closure rows.")
    summary_rows: list[dict[str, Any]] = []
    for fit_row in fit_rows:
        scalar_name = str(fit_row["scalar_name"])
        switch_bucket = str(fit_row["switch_bucket"])
        rows = [
            row
            for row in closure_rows
            if str(row["scalar_name"]) == scalar_name and str(row["switch_bucket"]) == switch_bucket
        ]
        actual_values = [float(row["actual_scalar_delta"]) for row in rows]
        predicted_values = [float(row["predicted_scalar_delta"]) for row in rows]
        residual_values = [float(row["closure_residual"]) for row in rows]
        summary_rows.append(
            {
                "scalar_name": scalar_name,
                "switch_bucket": switch_bucket,
                "num_observations": len(rows),
                "competitor_switch_fraction": _mean(
                    [1.0 if bool(row["competitor_switched"]) else 0.0 for row in rows],
                    label=f"{scalar_name}/{switch_bucket}/switch_fraction",
                ),
                "mean_actual_scalar_delta": _mean(actual_values, label=f"{scalar_name}/{switch_bucket}/actual"),
                "mean_predicted_scalar_delta": _mean(
                    predicted_values,
                    label=f"{scalar_name}/{switch_bucket}/predicted",
                ),
                "mean_residual": _mean(residual_values, label=f"{scalar_name}/{switch_bucket}/residual"),
                "mean_abs_residual": _mean(
                    [abs(value) for value in residual_values],
                    label=f"{scalar_name}/{switch_bucket}/abs_residual",
                ),
                "r_squared": _r_squared(actual_values=actual_values, predicted_values=predicted_values),
                "fit": {
                    key: value
                    for key, value in fit_row.items()
                    if key not in {"scalar_name", "switch_bucket"}
                },
            }
        )
    return {
        "num_observations": len(closure_rows),
        "num_fits": len(summary_rows),
        "scalar_bucket_summaries": sorted(
            summary_rows,
            key=lambda row: (
                -1.0 if row["r_squared"] is None else -float(row["r_squared"]),
                str(row["scalar_name"]),
                str(row["switch_bucket"]),
            ),
        ),
        "ranked_by_mean_abs_residual": sorted(
            summary_rows,
            key=lambda row: float(row["mean_abs_residual"]),
        ),
        "route_contributions": sorted(
            coefficient_rows,
            key=lambda row: abs(float(row["mean_contribution"])),
            reverse=True,
        ),
    }


def _plot_scalar_r_squared(
    *,
    summary_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    rows = [row for row in summary_rows if str(row["switch_bucket"]) == "all" and row["r_squared"] is not None]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    labels = [str(row["scalar_name"]) for row in rows]
    values = [float(row["r_squared"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(rows)), 5))
    ax.bar(range(len(rows)), values, color="#376f8f")
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Route-to-scalar closure R squared")
    ax.set_ylabel("R squared")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_scalar_abs_residual(
    *,
    summary_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    rows = [row for row in summary_rows if str(row["switch_bucket"]) == "all"]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    labels = [str(row["scalar_name"]) for row in rows]
    values = [float(row["mean_abs_residual"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(rows)), 5))
    ax.bar(range(len(rows)), values, color="#8f6237")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Route-to-scalar closure mean absolute residual")
    ax.set_ylabel("mean |actual - predicted|")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_route_to_scalar_closure_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    lines = [
        "# Route-To-Scalar Closure",
        "",
        "## Calculation",
        "",
        "This report tests whether existing route-score growth explains scalar movement from the scalar residual diagnosis artifact.",
        "",
        "```text",
        "Delta scalar_t(pair) = scalar(theta_target, pair) - scalar(theta_source, pair)",
        "Delta route_P,t(pair) = route_score_P(theta_target, pair) - route_score_P(theta_source, pair)",
        "Delta scalar_t(pair) ~= sum_P beta_P Delta route_P,t(pair) + residual_t(pair)",
        "```",
        "",
        "This is an artifact join: it does not recompute model activations.",
        "",
        "## Inputs",
        "",
        f"- route closure rows: `{report['route_closure_rows_path']}`",
        f"- scalar pair rows: `{report['scalar_pair_rows_path']}`",
        f"- route labels: `{report['route_labels']}`",
        f"- scalar names: `{report['scalar_names']}`",
        f"- switch buckets: `{report['switch_buckets']}`",
        f"- fit intercept: `{bool(report['fit_intercept'])}`",
        "",
        "## Scalar Closure Summary",
        "",
        "| scalar | bucket | observations | switch fraction | R squared | mean actual | mean predicted | mean abs residual | rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scalar_bucket_summaries"]:
        r2 = row["r_squared"]
        lines.append(
            "| `{scalar}` | `{bucket}` | {n} | {switch:.3f} | {r2} | {actual:.6g} | {pred:.6g} | {abs_resid:.6g} | {rank}/{params} |".format(
                scalar=row["scalar_name"],
                bucket=row["switch_bucket"],
                n=int(row["num_observations"]),
                switch=float(row["competitor_switch_fraction"]),
                r2="" if r2 is None else f"{float(r2):.6f}",
                actual=float(row["mean_actual_scalar_delta"]),
                pred=float(row["mean_predicted_scalar_delta"]),
                abs_resid=float(row["mean_abs_residual"]),
                rank=int(row["fit"]["matrix_rank"]),
                params=int(row["fit"]["num_parameters"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Route Contributions",
            "",
            "| scalar | bucket | route | coefficient | mean route delta | mean contribution |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["route_contributions"][: min(80, len(summary["route_contributions"]))]:
        lines.append(
            "| `{scalar}` | `{bucket}` | `{route}` | {coef:.6g} | {delta:.6g} | {contribution:.6g} |".format(
                scalar=row["scalar_name"],
                bucket=row["switch_bucket"],
                route=row["route_label"],
                coef=float(row["coefficient"]),
                delta=float(row["mean_route_score_delta"]),
                contribution=float(row["mean_contribution"]),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- closure rows: `{report['closure_rows_path']}`",
            f"- coefficient rows: `{report['coefficient_rows_path']}`",
        ]
    )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for label, plot_path in plot_paths.items():
            lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_route_to_scalar_closure(
    *,
    route_closure_rows_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    scalar_names: list[str] | None = None,
    switch_buckets: list[str] | None = None,
    route_labels: list[str] | None = None,
    margin_side: str | None = None,
    pair_types: list[str] | None = None,
    fit_intercept: bool = False,
    duplicate_tolerance: float = 1.0e-6,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, dict[str, Path]]:
    resolved_scalars = _resolve_unique(
        values=scalar_names,
        default_values=list(DEFAULT_ROUTE_TO_SCALAR_SCALARS),
        allowed_values=None,
        label="scalar",
    )
    resolved_switch_buckets = _resolve_unique(
        values=switch_buckets,
        default_values=["all"],
        allowed_values=ROUTE_TO_SCALAR_SWITCH_BUCKETS,
        label="switch bucket",
    )
    route_closure_rows, available_route_labels = _load_route_closure_rows(
        route_closure_rows_path=route_closure_rows_path,
        value_tolerance=duplicate_tolerance,
    )
    if route_labels is None:
        resolved_route_labels = available_route_labels
    else:
        resolved_route_labels = _resolve_unique(
            values=route_labels,
            default_values=[],
            allowed_values=available_route_labels,
            label="route label",
        )
    scalar_pair_rows = _load_scalar_pair_rows(scalar_pair_rows_path)
    observations = _build_scalar_observations(
        scalar_pair_rows=scalar_pair_rows,
        route_closure_rows=route_closure_rows,
        route_labels=resolved_route_labels,
        scalar_names=resolved_scalars,
        switch_buckets=resolved_switch_buckets,
        margin_side=margin_side,
        pair_types=pair_types,
    )
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    closure_rows_path = output_dir / "route_to_scalar_closure_rows.jsonl"
    coefficient_rows_path = output_dir / "route_to_scalar_closure_coefficients.jsonl"
    report_path = output_dir / "route_to_scalar_closure_report.json"
    markdown_path = output_dir / "route_to_scalar_closure_report.md"

    print(
        "[route-to-scalar-closure] "
        f"observations={len(observations)} routes={len(resolved_route_labels)} "
        f"scalars={resolved_scalars} switch_buckets={resolved_switch_buckets}",
        flush=True,
    )
    all_closure_rows: list[dict[str, Any]] = []
    all_coefficient_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for scalar_name in resolved_scalars:
        for switch_bucket in resolved_switch_buckets:
            fit, closure_rows, coefficient_rows = _fit_scalar_bucket(
                observations=observations,
                route_labels=resolved_route_labels,
                scalar_name=scalar_name,
                switch_bucket=switch_bucket,
                fit_intercept=fit_intercept,
            )
            fit_rows.append(
                {
                    "scalar_name": scalar_name,
                    "switch_bucket": switch_bucket,
                    **fit,
                }
            )
            all_closure_rows.extend(closure_rows)
            all_coefficient_rows.extend(coefficient_rows)
    write_jsonl(closure_rows_path, all_closure_rows)
    write_jsonl(coefficient_rows_path, all_coefficient_rows)
    summary = _summarize_route_to_scalar_closure(
        closure_rows=all_closure_rows,
        coefficient_rows=all_coefficient_rows,
        fit_rows=fit_rows,
    )
    plot_paths: dict[str, Path] = {}
    r2_plot = _plot_scalar_r_squared(
        summary_rows=summary["scalar_bucket_summaries"],
        output_path=output_dir / "route_to_scalar_closure_r_squared.svg",
    )
    if r2_plot is not None:
        plot_paths["r_squared"] = r2_plot
    residual_plot = _plot_scalar_abs_residual(
        summary_rows=summary["scalar_bucket_summaries"],
        output_path=output_dir / "route_to_scalar_closure_abs_residual.svg",
    )
    if residual_plot is not None:
        plot_paths["abs_residual"] = residual_plot

    report = {
        "schema_version": ROUTE_TO_SCALAR_CLOSURE_SCHEMA_VERSION,
        "route_closure_rows_path": str(route_closure_rows_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "output_dir": str(output_dir),
        "scalar_names": resolved_scalars,
        "switch_buckets": resolved_switch_buckets,
        "route_labels": resolved_route_labels,
        "margin_side": margin_side,
        "pair_types": pair_types,
        "fit_intercept": fit_intercept,
        "duplicate_tolerance": duplicate_tolerance,
        "calculation": {
            "route_delta": "route_score(theta_target) - route_score(theta_source), read from route closure rows",
            "scalar_delta": "scalar(theta_target) - scalar(theta_source), read from scalar pair rows",
            "closure": "scalar_delta ~= sum_route beta_route * route_delta + residual",
        },
        "closure_rows_path": str(closure_rows_path),
        "coefficient_rows_path": str(coefficient_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_route_to_scalar_closure_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    print(
        f"[route-to-scalar-closure] complete report={report_path} rows={closure_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, closure_rows_path, coefficient_rows_path, plot_paths
