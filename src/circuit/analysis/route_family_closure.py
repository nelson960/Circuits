from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from circuit.analysis.route_to_margin_closure import fit_route_to_margin_closure
from circuit.analysis.route_to_scalar_closure import _r_squared
from circuit.analysis.shared_feature_dynamics import _import_matplotlib
from circuit.io import iter_jsonl, write_json, write_jsonl


ROUTE_FAMILY_CLOSURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RouteFamilySpec:
    label: str
    route_labels: tuple[str, ...]


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"Cannot compute mean for empty values: {label}")
    return sum(values) / float(len(values))


def _parse_family_spec(raw_spec: str) -> RouteFamilySpec:
    fields: dict[str, list[str]] = defaultdict(list)
    for item in raw_spec.split(","):
        if not item:
            raise ValueError(f"Empty family field in {raw_spec!r}.")
        if "=" not in item:
            raise ValueError(f"Family field {item!r} is missing '=' in {raw_spec!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in {"label", "route"}:
            raise ValueError(f"Unsupported family key {key!r}; expected 'label' or 'route'.")
        if not value:
            raise ValueError(f"Empty value for family key {key!r} in {raw_spec!r}.")
        fields[key].append(value)
    labels = fields.get("label", [])
    if len(labels) != 1:
        raise ValueError(f"Family spec must contain exactly one label field: {raw_spec!r}.")
    route_labels = fields.get("route", [])
    if not route_labels:
        raise ValueError(f"Family spec must contain at least one route field: {raw_spec!r}.")
    unique_routes: list[str] = []
    for route_label in route_labels:
        if route_label in unique_routes:
            raise ValueError(f"Duplicate route {route_label!r} in family spec {raw_spec!r}.")
        unique_routes.append(route_label)
    return RouteFamilySpec(label=labels[0], route_labels=tuple(unique_routes))


def _parse_family_specs(raw_specs: list[str]) -> list[RouteFamilySpec]:
    if not raw_specs:
        raise ValueError("At least one --family spec is required.")
    families = [_parse_family_spec(raw_spec) for raw_spec in raw_specs]
    seen_labels: set[str] = set()
    for family in families:
        if family.label in seen_labels:
            raise ValueError(f"Duplicate family label {family.label!r}.")
        seen_labels.add(family.label)
    return families


def _load_route_closure_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = [dict(row) for row in iter_jsonl(path)]
    if not rows:
        raise RuntimeError(f"Route closure rows file is empty: {path}")
    route_labels: set[str] = set()
    for row in rows:
        if "actual_margin_delta" not in row:
            raise RuntimeError(f"Route closure row is missing actual_margin_delta: {row.get('pair_id')}")
        route_deltas = row.get("route_score_deltas")
        if not isinstance(route_deltas, dict) or not route_deltas:
            raise RuntimeError(f"Route closure row is missing route_score_deltas: {row.get('pair_id')}")
        route_labels.update(str(label) for label in route_deltas)
    if not route_labels:
        raise RuntimeError(f"No route labels found in route closure rows: {path}")
    return rows, sorted(route_labels)


def _filter_rows(
    *,
    rows: list[dict[str, Any]],
    pair_types: list[str] | None,
    splits: list[str] | None,
    target_scalar: str | None,
    record_side: str | None,
) -> list[dict[str, Any]]:
    pair_type_filter = None if pair_types is None else set(pair_types)
    split_filter = None if splits is None else set(splits)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if pair_type_filter is not None and str(row["pair_type"]) not in pair_type_filter:
            continue
        if split_filter is not None and str(row["split"]) not in split_filter:
            continue
        if target_scalar is not None and str(row["target_scalar"]) != target_scalar:
            continue
        if record_side is not None and str(row["record_side"]) != record_side:
            continue
        filtered.append(row)
    if not filtered:
        raise RuntimeError("No route-family closure rows survived filters.")
    return filtered


def _validate_families(*, families: list[RouteFamilySpec], available_route_labels: list[str]) -> None:
    available = set(available_route_labels)
    for family in families:
        missing = [route_label for route_label in family.route_labels if route_label not in available]
        if missing:
            raise KeyError(
                f"Family {family.label!r} references missing route labels: {missing}. "
                f"Available labels: {available_route_labels}"
            )


def _fit_family(
    *,
    rows: list[dict[str, Any]],
    family: RouteFamilySpec,
    fit_intercept: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    route_delta_columns = {
        route_label: [float(row["route_score_deltas"][route_label]) for row in rows]
        for route_label in family.route_labels
    }
    actual_values = [float(row["actual_margin_delta"]) for row in rows]
    fit = fit_route_to_margin_closure(
        route_delta_columns=route_delta_columns,
        margin_deltas=actual_values,
        fit_intercept=fit_intercept,
    )
    predicted_values = fit["predicted_values"]
    residual_values = fit["residual_values"]
    closure_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        route_deltas = {
            route_label: float(row["route_score_deltas"][route_label])
            for route_label in family.route_labels
        }
        closure_rows.append(
            {
                "family_label": family.label,
                "family_route_labels": list(family.route_labels),
                "source_step": int(row["source_step"]),
                "target_step": int(row["target_step"]),
                "step_gap": int(row["step_gap"]),
                "source_checkpoint": str(row["source_checkpoint"]),
                "target_checkpoint": str(row["target_checkpoint"]),
                "pair_id": str(row["pair_id"]),
                "split": str(row["split"]),
                "pair_type": str(row["pair_type"]),
                "record_side": str(row["record_side"]),
                "target_scalar": str(row["target_scalar"]),
                "source_margin": float(row["source_margin"]),
                "target_margin": float(row["target_margin"]),
                "actual_margin_delta": float(row["actual_margin_delta"]),
                "predicted_margin_delta": float(predicted_values[index]),
                "closure_residual": float(residual_values[index]),
                "route_score_deltas": route_deltas,
            }
        )
    coefficient_rows: list[dict[str, Any]] = []
    for route_label in family.route_labels:
        mean_delta = _mean(route_delta_columns[route_label], label=f"{family.label}/{route_label}/delta")
        coefficient = float(fit["coefficients"][route_label])
        coefficient_rows.append(
            {
                "family_label": family.label,
                "route_label": route_label,
                "coefficient": coefficient,
                "mean_route_score_delta": mean_delta,
                "mean_contribution": coefficient * mean_delta,
            }
        )
    fit_summary = {
        key: value
        for key, value in fit.items()
        if key not in {"predicted_values", "residual_values"}
    }
    fit_summary["family_label"] = family.label
    fit_summary["route_labels"] = list(family.route_labels)
    return fit_summary, closure_rows, coefficient_rows


def _summarize_family_rows(
    *,
    closure_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not closure_rows:
        raise ValueError("Cannot summarize route-family closure without closure rows.")
    family_summaries: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    for fit_row in fit_rows:
        family_label = str(fit_row["family_label"])
        family_rows = [row for row in closure_rows if str(row["family_label"]) == family_label]
        actual_values = [float(row["actual_margin_delta"]) for row in family_rows]
        predicted_values = [float(row["predicted_margin_delta"]) for row in family_rows]
        residual_values = [float(row["closure_residual"]) for row in family_rows]
        family_summaries.append(
            {
                "family_label": family_label,
                "route_labels": list(fit_row["route_labels"]),
                "num_routes": len(fit_row["route_labels"]),
                "num_observations": len(family_rows),
                "mean_actual_margin_delta": _mean(actual_values, label=f"{family_label}/actual"),
                "mean_predicted_margin_delta": _mean(predicted_values, label=f"{family_label}/predicted"),
                "mean_residual": _mean(residual_values, label=f"{family_label}/residual"),
                "mean_abs_residual": _mean([abs(value) for value in residual_values], label=f"{family_label}/abs_residual"),
                "r_squared": _r_squared(actual_values=actual_values, predicted_values=predicted_values),
                "fit": {key: value for key, value in fit_row.items() if key not in {"family_label", "route_labels"}},
            }
        )
        by_interval: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in family_rows:
            by_interval[(int(row["source_step"]), int(row["target_step"]))].append(row)
        for (source_step, target_step), rows in sorted(by_interval.items()):
            interval_actual = [float(row["actual_margin_delta"]) for row in rows]
            interval_predicted = [float(row["predicted_margin_delta"]) for row in rows]
            interval_residual = [float(row["closure_residual"]) for row in rows]
            interval_rows.append(
                {
                    "family_label": family_label,
                    "source_step": source_step,
                    "target_step": target_step,
                    "num_observations": len(rows),
                    "actual_mean": _mean(interval_actual, label=f"{family_label}/{source_step}->{target_step}/actual"),
                    "predicted_mean": _mean(
                        interval_predicted,
                        label=f"{family_label}/{source_step}->{target_step}/predicted",
                    ),
                    "residual_mean": _mean(
                        interval_residual,
                        label=f"{family_label}/{source_step}->{target_step}/residual",
                    ),
                    "abs_residual_mean": _mean(
                        [abs(value) for value in interval_residual],
                        label=f"{family_label}/{source_step}->{target_step}/abs_residual",
                    ),
                    "r_squared": _r_squared(actual_values=interval_actual, predicted_values=interval_predicted),
                }
            )
    return {
        "num_observations": len(closure_rows),
        "num_families": len(family_summaries),
        "family_summaries": family_summaries,
        "ranked_by_r_squared": sorted(
            family_summaries,
            key=lambda row: (-1.0 if row["r_squared"] is None else -float(row["r_squared"]), str(row["family_label"])),
        ),
        "ranked_by_mean_abs_residual": sorted(
            family_summaries,
            key=lambda row: (float(row["mean_abs_residual"]), str(row["family_label"])),
        ),
        "interval_rows": interval_rows,
        "route_contributions": sorted(
            coefficient_rows,
            key=lambda row: (str(row["family_label"]), -abs(float(row["mean_contribution"]))),
        ),
    }


def _plot_family_metric(
    *,
    summary_rows: list[dict[str, Any]],
    metric_name: str,
    output_path: Path,
) -> Path | None:
    rows = [row for row in summary_rows if row.get(metric_name) is not None]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    labels = [str(row["family_label"]) for row in rows]
    values = [float(row[metric_name]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(rows)), 5))
    ax.bar(range(len(rows)), values, color="#376f8f")
    if metric_name == "r_squared":
        ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
        ax.set_ylabel("R squared")
        ax.set_title("Route-family closure R squared")
    else:
        ax.set_ylabel(metric_name.replace("_", " "))
        ax.set_title("Route-family closure mean absolute residual")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_actual_vs_predicted(
    *,
    closure_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not closure_rows:
        return None
    _, plt = _import_matplotlib()
    families = sorted({str(row["family_label"]) for row in closure_rows})
    fig, ax = plt.subplots(figsize=(7, 7))
    for family in families:
        rows = [row for row in closure_rows if str(row["family_label"]) == family]
        ax.scatter(
            [float(row["actual_margin_delta"]) for row in rows],
            [float(row["predicted_margin_delta"]) for row in rows],
            s=10,
            alpha=0.35,
            label=family,
        )
    values = [float(row["actual_margin_delta"]) for row in closure_rows]
    values.extend(float(row["predicted_margin_delta"]) for row in closure_rows)
    low = min(values)
    high = max(values)
    ax.plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xlabel("actual margin delta")
    ax.set_ylabel("predicted margin delta")
    ax.set_title("Route-family closure: predicted vs actual")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_markdown(*, path: Path, report: dict[str, Any], plot_paths: dict[str, Path]) -> None:
    summary = report["summary"]
    lines = [
        "# Route-Family Closure Report",
        "",
        "## Calculation",
        "",
        "This report compares named route families using existing route-to-margin closure rows.",
        "",
        "```text",
        "Delta m_t(pair) ~= sum_{route in family} beta_route Delta C_route,t(pair) + residual_t(pair)",
        "```",
        "",
        "This is an artifact join and refit. It does not recompute model activations.",
        "",
        "## Inputs",
        "",
        f"- route closure rows: `{report['route_closure_rows_path']}`",
        f"- fit intercept: `{bool(report['fit_intercept'])}`",
        f"- pair type filters: `{report['pair_types']}`",
        f"- split filters: `{report['splits']}`",
        f"- target scalar filter: `{report['target_scalar']}`",
        f"- record side filter: `{report['record_side']}`",
        "",
        "## Family Summary",
        "",
        "| family | routes | observations | R squared | mean actual | mean predicted | mean abs residual | rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["family_summaries"]:
        r2 = row["r_squared"]
        fit = row["fit"]
        lines.append(
            "| `{family}` | {routes} | {obs} | {r2} | {actual:.6g} | {pred:.6g} | {abs_resid:.6g} | {rank}/{params} |".format(
                family=row["family_label"],
                routes=int(row["num_routes"]),
                obs=int(row["num_observations"]),
                r2="" if r2 is None else f"{float(r2):.6f}",
                actual=float(row["mean_actual_margin_delta"]),
                pred=float(row["mean_predicted_margin_delta"]),
                abs_resid=float(row["mean_abs_residual"]),
                rank=int(fit["matrix_rank"]),
                params=int(fit["num_parameters"]),
            )
        )
    lines.extend(["", "## Family Routes", ""])
    for family in report["families"]:
        lines.append(f"- `{family['label']}`: `{family['route_labels']}`")
    lines.extend(
        [
            "",
            "## Top Contributions",
            "",
            "| family | route | coefficient | mean route delta | mean contribution |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summary["route_contributions"][: min(120, len(summary["route_contributions"]))]:
        lines.append(
            "| `{family}` | `{route}` | {coef:.6g} | {delta:.6g} | {contribution:.6g} |".format(
                family=row["family_label"],
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
            f"- family summary rows: `{report['family_summary_rows_path']}`",
            f"- interval rows: `{report['interval_rows_path']}`",
        ]
    )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for label, plot_path in plot_paths.items():
            lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_route_family_closure_report(
    *,
    route_closure_rows_path: Path,
    output_dir: Path,
    raw_family_specs: list[str],
    pair_types: list[str] | None = None,
    splits: list[str] | None = None,
    target_scalar: str | None = None,
    record_side: str | None = None,
    fit_intercept: bool = False,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    families = _parse_family_specs(raw_family_specs)
    rows, available_route_labels = _load_route_closure_rows(route_closure_rows_path)
    _validate_families(families=families, available_route_labels=available_route_labels)
    filtered_rows = _filter_rows(
        rows=rows,
        pair_types=pair_types,
        splits=splits,
        target_scalar=target_scalar,
        record_side=record_side,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[route-family-closure-report] "
        f"families={len(families)} observations={len(filtered_rows)} source={route_closure_rows_path}",
        flush=True,
    )
    all_closure_rows: list[dict[str, Any]] = []
    all_coefficient_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for family in families:
        fit_summary, closure_rows, coefficient_rows = _fit_family(
            rows=filtered_rows,
            family=family,
            fit_intercept=fit_intercept,
        )
        fit_rows.append(fit_summary)
        all_closure_rows.extend(closure_rows)
        all_coefficient_rows.extend(coefficient_rows)
        r_squared = fit_summary["r_squared"]
        r_squared_text = "" if r_squared is None else f"{float(r_squared):.6g}"
        print(
            "[route-family-closure-report] fitted "
            f"family={family.label} routes={len(family.route_labels)} "
            f"r_squared={r_squared_text}",
            flush=True,
        )
    summary = _summarize_family_rows(
        closure_rows=all_closure_rows,
        fit_rows=fit_rows,
        coefficient_rows=all_coefficient_rows,
    )
    closure_rows_path = output_dir / "route_family_closure_rows.jsonl"
    coefficient_rows_path = output_dir / "route_family_closure_coefficients.jsonl"
    family_summary_rows_path = output_dir / "route_family_closure_family_summary.jsonl"
    interval_rows_path = output_dir / "route_family_closure_interval_rows.jsonl"
    report_path = output_dir / "route_family_closure_report.json"
    markdown_path = output_dir / "route_family_closure_report.md"
    write_jsonl(closure_rows_path, all_closure_rows)
    write_jsonl(coefficient_rows_path, all_coefficient_rows)
    write_jsonl(family_summary_rows_path, summary["family_summaries"])
    write_jsonl(interval_rows_path, summary["interval_rows"])
    plot_paths: dict[str, Path] = {}
    r2_plot = _plot_family_metric(
        summary_rows=summary["family_summaries"],
        metric_name="r_squared",
        output_path=output_dir / "route_family_closure_r_squared.svg",
    )
    if r2_plot is not None:
        plot_paths["r_squared"] = r2_plot
    residual_plot = _plot_family_metric(
        summary_rows=summary["family_summaries"],
        metric_name="mean_abs_residual",
        output_path=output_dir / "route_family_closure_abs_residual.svg",
    )
    if residual_plot is not None:
        plot_paths["abs_residual"] = residual_plot
    actual_vs_predicted_plot = _plot_actual_vs_predicted(
        closure_rows=all_closure_rows,
        output_path=output_dir / "route_family_closure_actual_vs_predicted.svg",
    )
    if actual_vs_predicted_plot is not None:
        plot_paths["actual_vs_predicted"] = actual_vs_predicted_plot
    report = {
        "schema_version": ROUTE_FAMILY_CLOSURE_SCHEMA_VERSION,
        "route_closure_rows_path": str(route_closure_rows_path),
        "output_dir": str(output_dir),
        "families": [
            {"label": family.label, "route_labels": list(family.route_labels)}
            for family in families
        ],
        "available_route_labels": available_route_labels,
        "pair_types": pair_types,
        "splits": splits,
        "target_scalar": target_scalar,
        "record_side": record_side,
        "fit_intercept": fit_intercept,
        "calculation": {
            "source": "existing route-to-margin closure rows",
            "closure": "margin_delta ~= sum_route beta_route * route_delta + residual, fitted separately per family",
        },
        "closure_rows_path": str(closure_rows_path),
        "coefficient_rows_path": str(coefficient_rows_path),
        "family_summary_rows_path": str(family_summary_rows_path),
        "interval_rows_path": str(interval_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    print(
        f"[route-family-closure-report] complete report={report_path} rows={closure_rows_path}",
        flush=True,
    )
    return (
        report_path,
        markdown_path,
        closure_rows_path,
        coefficient_rows_path,
        family_summary_rows_path,
        interval_rows_path,
        plot_paths,
    )
