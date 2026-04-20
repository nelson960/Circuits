from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
from typing import Any

from circuit.analysis.route_to_scalar_closure import _switch_bucket_matches
from circuit.analysis.shared_feature_dynamics import _import_matplotlib
from circuit.io import iter_jsonl, write_json, write_jsonl


ANSWER_MARGIN_BRANCH_DECOMPOSITION_SCHEMA_VERSION = 1
ANSWER_MARGIN_BRANCH_SWITCH_BUCKETS = ["all", "same_competitor", "competitor_switch"]
ANSWER_MARGIN_BRANCH_REQUIRED_SCALARS = [
    "moving_answer_margin",
    "fixed_source_competitor_margin",
    "fixed_target_competitor_margin",
    "source_best_wrong_logit",
    "target_best_wrong_logit",
]


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"Cannot compute mean for empty values: {label}")
    return float(sum(values) / len(values))


def _sum_squares(values: list[float]) -> float:
    return float(sum(value * value for value in values))


def _r_squared(*, actual_values: list[float], predicted_values: list[float]) -> float | None:
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual_values and predicted_values must have the same length.")
    mean_actual = _mean(actual_values, label="r squared actual")
    total = sum((value - mean_actual) ** 2 for value in actual_values)
    if total <= 1.0e-12:
        return None
    residual = sum((actual - predicted) ** 2 for actual, predicted in zip(actual_values, predicted_values, strict=True))
    return float(1.0 - (residual / total))


def _resolve_unique_values(
    *,
    values: list[str] | None,
    default_values: list[str],
    allowed_values: list[str],
    label: str,
) -> list[str]:
    raw_values = default_values if values is None else values
    if not raw_values:
        raise ValueError(f"{label} must not be empty.")
    resolved: list[str] = []
    for value in raw_values:
        if value not in allowed_values:
            raise ValueError(f"Unsupported {label} {value!r}; expected one of {allowed_values}.")
        if value not in resolved:
            resolved.append(value)
    return resolved


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows = [row for row in iter_jsonl(path)]
    if not rows:
        raise RuntimeError(f"{label} file is empty: {path}")
    return rows


def _validate_scalar_payload(row: dict[str, Any]) -> dict[str, Any]:
    scalars = row.get("scalars")
    if not isinstance(scalars, dict):
        raise RuntimeError(f"Scalar pair row is missing scalars object: {row.get('interval_pair_id')}")
    missing = [scalar for scalar in ANSWER_MARGIN_BRANCH_REQUIRED_SCALARS if scalar not in scalars]
    if missing:
        raise KeyError(f"Scalar pair row {row.get('interval_pair_id')} is missing required scalars: {missing}")
    return scalars


def build_branch_decomposition_rows(
    *,
    scalar_pair_rows: list[dict[str, Any]],
    margin_side: str | None,
    pair_types: list[str] | None,
    switch_buckets: list[str],
    reconstruction_tolerance: float,
) -> list[dict[str, Any]]:
    if reconstruction_tolerance < 0.0:
        raise ValueError("reconstruction_tolerance must be non-negative.")
    pair_type_filter = None if pair_types is None else set(pair_types)
    rows: list[dict[str, Any]] = []
    for source_row in scalar_pair_rows:
        if margin_side is not None and str(source_row["margin_side"]) != margin_side:
            continue
        if pair_type_filter is not None and str(source_row["pair_type"]) not in pair_type_filter:
            continue
        scalars = _validate_scalar_payload(source_row)
        moving_delta = float(scalars["moving_answer_margin"]["delta"])
        fixed_source_delta = float(scalars["fixed_source_competitor_margin"]["delta"])
        fixed_target_delta = float(scalars["fixed_target_competitor_margin"]["delta"])
        source_wrong_at_source = float(scalars["source_best_wrong_logit"]["source"])
        source_wrong_at_target = float(scalars["source_best_wrong_logit"]["target"])
        target_wrong_at_source = float(scalars["target_best_wrong_logit"]["source"])
        target_wrong_at_target = float(scalars["target_best_wrong_logit"]["target"])
        target_branch_correction = source_wrong_at_target - target_wrong_at_target
        source_branch_correction = source_wrong_at_source - target_wrong_at_source
        reconstructed_from_source = fixed_source_delta + target_branch_correction
        reconstructed_from_target = fixed_target_delta + source_branch_correction
        source_reconstruction_error = moving_delta - reconstructed_from_source
        target_reconstruction_error = moving_delta - reconstructed_from_target
        if abs(source_reconstruction_error) > reconstruction_tolerance:
            raise RuntimeError(
                f"Source-branch reconstruction mismatch for {source_row.get('interval_pair_id')}: "
                f"error={source_reconstruction_error:.6g} tolerance={reconstruction_tolerance:.6g}"
            )
        if abs(target_reconstruction_error) > reconstruction_tolerance:
            raise RuntimeError(
                f"Target-branch reconstruction mismatch for {source_row.get('interval_pair_id')}: "
                f"error={target_reconstruction_error:.6g} tolerance={reconstruction_tolerance:.6g}"
            )
        for switch_bucket in switch_buckets:
            if not _switch_bucket_matches(source_row, switch_bucket):
                continue
            rows.append(
                {
                    "source_step": int(source_row["source_step"]),
                    "target_step": int(source_row["target_step"]),
                    "step_gap": int(source_row["step_gap"]),
                    "pair_id": str(source_row["pair_id"]),
                    "interval_pair_id": str(source_row["interval_pair_id"]),
                    "split": str(source_row["split"]),
                    "pair_type": str(source_row["pair_type"]),
                    "margin_side": str(source_row["margin_side"]),
                    "switch_bucket": switch_bucket,
                    "competitor_switched": bool(source_row["competitor_switched"]),
                    "answer_target_id": int(source_row["answer_target_id"]),
                    "source_best_wrong_token_id": int(source_row["source_best_wrong_token_id"]),
                    "target_best_wrong_token_id": int(source_row["target_best_wrong_token_id"]),
                    "moving_margin_delta": moving_delta,
                    "fixed_source_margin_delta": fixed_source_delta,
                    "fixed_target_margin_delta": fixed_target_delta,
                    "target_branch_correction": target_branch_correction,
                    "source_branch_correction": source_branch_correction,
                    "reconstructed_from_source_fixed": reconstructed_from_source,
                    "reconstructed_from_target_fixed": reconstructed_from_target,
                    "source_reconstruction_error": source_reconstruction_error,
                    "target_reconstruction_error": target_reconstruction_error,
                    "source_wrong_at_source": source_wrong_at_source,
                    "source_wrong_at_target": source_wrong_at_target,
                    "target_wrong_at_source": target_wrong_at_source,
                    "target_wrong_at_target": target_wrong_at_target,
                }
            )
    if not rows:
        raise RuntimeError("No answer-margin branch rows survived the filters.")
    return rows


def summarize_branch_decomposition(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot summarize empty branch rows.")
    summary_rows: list[dict[str, Any]] = []
    buckets = sorted({str(row["switch_bucket"]) for row in rows})
    pair_types = ["__all__", *sorted({str(row["pair_type"]) for row in rows})]
    for switch_bucket in buckets:
        for pair_type in pair_types:
            selected = [
                row
                for row in rows
                if str(row["switch_bucket"]) == switch_bucket
                and (pair_type == "__all__" or str(row["pair_type"]) == pair_type)
            ]
            if not selected:
                continue
            moving = [float(row["moving_margin_delta"]) for row in selected]
            fixed_source = [float(row["fixed_source_margin_delta"]) for row in selected]
            fixed_target = [float(row["fixed_target_margin_delta"]) for row in selected]
            target_correction = [float(row["target_branch_correction"]) for row in selected]
            source_correction = [float(row["source_branch_correction"]) for row in selected]
            source_errors = [float(row["source_reconstruction_error"]) for row in selected]
            target_errors = [float(row["target_reconstruction_error"]) for row in selected]
            moving_energy = _sum_squares(moving)
            target_branch_energy = _sum_squares(target_correction)
            source_branch_energy = _sum_squares(source_correction)
            switch_count = sum(1 for row in selected if bool(row["competitor_switched"]))
            summary_rows.append(
                {
                    "switch_bucket": switch_bucket,
                    "pair_type": pair_type,
                    "num_observations": len(selected),
                    "competitor_switch_count": switch_count,
                    "competitor_switch_fraction": switch_count / float(len(selected)),
                    "moving_delta_mean": _mean(moving, label=f"{switch_bucket}/{pair_type}/moving"),
                    "moving_delta_abs_mean": _mean(
                        [abs(value) for value in moving],
                        label=f"{switch_bucket}/{pair_type}/abs moving",
                    ),
                    "fixed_source_delta_mean": _mean(fixed_source, label=f"{switch_bucket}/{pair_type}/fixed source"),
                    "fixed_source_delta_abs_mean": _mean(
                        [abs(value) for value in fixed_source],
                        label=f"{switch_bucket}/{pair_type}/abs fixed source",
                    ),
                    "fixed_target_delta_mean": _mean(fixed_target, label=f"{switch_bucket}/{pair_type}/fixed target"),
                    "fixed_target_delta_abs_mean": _mean(
                        [abs(value) for value in fixed_target],
                        label=f"{switch_bucket}/{pair_type}/abs fixed target",
                    ),
                    "target_branch_correction_mean": _mean(
                        target_correction,
                        label=f"{switch_bucket}/{pair_type}/target correction",
                    ),
                    "target_branch_correction_abs_mean": _mean(
                        [abs(value) for value in target_correction],
                        label=f"{switch_bucket}/{pair_type}/abs target correction",
                    ),
                    "source_branch_correction_mean": _mean(
                        source_correction,
                        label=f"{switch_bucket}/{pair_type}/source correction",
                    ),
                    "source_branch_correction_abs_mean": _mean(
                        [abs(value) for value in source_correction],
                        label=f"{switch_bucket}/{pair_type}/abs source correction",
                    ),
                    "target_branch_energy_fraction_of_moving": None
                    if moving_energy <= 1.0e-12
                    else target_branch_energy / moving_energy,
                    "source_branch_energy_fraction_of_moving": None
                    if moving_energy <= 1.0e-12
                    else source_branch_energy / moving_energy,
                    "source_reconstruction_abs_error_max": max(abs(value) for value in source_errors),
                    "target_reconstruction_abs_error_max": max(abs(value) for value in target_errors),
                }
            )
    return summary_rows


def _closure_key(row: dict[str, Any]) -> tuple[int, int, str, str, str, str]:
    return (
        int(row["source_step"]),
        int(row["target_step"]),
        str(row["interval_pair_id"]),
        str(row["margin_side"]),
        str(row["switch_bucket"]),
        str(row["scalar_name"]),
    )


def _load_output_closure_rows(path: Path) -> dict[tuple[int, int, str, str, str, str], dict[str, Any]]:
    rows_by_key: dict[tuple[int, int, str, str, str, str], dict[str, Any]] = {}
    for row in _load_jsonl(path, label="output closure rows"):
        key = _closure_key(row)
        if key in rows_by_key:
            raise RuntimeError(f"Duplicate output closure row for key: {key}")
        rows_by_key[key] = row
    return rows_by_key


def build_branch_aware_closure_rows(
    *,
    branch_rows: list[dict[str, Any]],
    output_closure_rows: dict[tuple[int, int, str, str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for branch_row in branch_rows:
        base_key = (
            int(branch_row["source_step"]),
            int(branch_row["target_step"]),
            str(branch_row["interval_pair_id"]),
            str(branch_row["margin_side"]),
            str(branch_row["switch_bucket"]),
        )
        moving_key = (*base_key, "moving_answer_margin")
        fixed_source_key = (*base_key, "fixed_source_competitor_margin")
        fixed_target_key = (*base_key, "fixed_target_competitor_margin")
        missing = [
            key
            for key in (moving_key, fixed_source_key, fixed_target_key)
            if key not in output_closure_rows
        ]
        if missing:
            raise KeyError(f"Missing output closure rows for branch-aware closure: {missing[:3]}")
        moving_closure = output_closure_rows[moving_key]
        fixed_source_closure = output_closure_rows[fixed_source_key]
        fixed_target_closure = output_closure_rows[fixed_target_key]
        actual_moving = float(branch_row["moving_margin_delta"])
        direct_predicted = float(moving_closure["predicted_scalar_delta"])
        source_branch_predicted = (
            float(fixed_source_closure["predicted_scalar_delta"])
            + float(branch_row["target_branch_correction"])
        )
        target_branch_predicted = (
            float(fixed_target_closure["predicted_scalar_delta"])
            + float(branch_row["source_branch_correction"])
        )
        rows.append(
            {
                "source_step": int(branch_row["source_step"]),
                "target_step": int(branch_row["target_step"]),
                "step_gap": int(branch_row["step_gap"]),
                "pair_id": str(branch_row["pair_id"]),
                "interval_pair_id": str(branch_row["interval_pair_id"]),
                "split": str(branch_row["split"]),
                "pair_type": str(branch_row["pair_type"]),
                "margin_side": str(branch_row["margin_side"]),
                "switch_bucket": str(branch_row["switch_bucket"]),
                "competitor_switched": bool(branch_row["competitor_switched"]),
                "actual_moving_margin_delta": actual_moving,
                "direct_moving_predicted_delta": direct_predicted,
                "direct_moving_residual": actual_moving - direct_predicted,
                "source_fixed_branch_predicted_delta": source_branch_predicted,
                "source_fixed_branch_residual": actual_moving - source_branch_predicted,
                "target_fixed_branch_predicted_delta": target_branch_predicted,
                "target_fixed_branch_residual": actual_moving - target_branch_predicted,
                "fixed_source_predicted_delta": float(fixed_source_closure["predicted_scalar_delta"]),
                "fixed_target_predicted_delta": float(fixed_target_closure["predicted_scalar_delta"]),
                "target_branch_correction": float(branch_row["target_branch_correction"]),
                "source_branch_correction": float(branch_row["source_branch_correction"]),
            }
        )
    if not rows:
        raise RuntimeError("No branch-aware closure rows were built.")
    return rows


def summarize_branch_aware_closure(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot summarize empty branch-aware closure rows.")
    summary_rows: list[dict[str, Any]] = []
    buckets = sorted({str(row["switch_bucket"]) for row in rows})
    pair_types = ["__all__", *sorted({str(row["pair_type"]) for row in rows})]
    for switch_bucket in buckets:
        for pair_type in pair_types:
            selected = [
                row
                for row in rows
                if str(row["switch_bucket"]) == switch_bucket
                and (pair_type == "__all__" or str(row["pair_type"]) == pair_type)
            ]
            if not selected:
                continue
            actual = [float(row["actual_moving_margin_delta"]) for row in selected]
            direct_predicted = [float(row["direct_moving_predicted_delta"]) for row in selected]
            source_branch_predicted = [float(row["source_fixed_branch_predicted_delta"]) for row in selected]
            target_branch_predicted = [float(row["target_fixed_branch_predicted_delta"]) for row in selected]
            direct_residuals = [float(row["direct_moving_residual"]) for row in selected]
            source_branch_residuals = [float(row["source_fixed_branch_residual"]) for row in selected]
            target_branch_residuals = [float(row["target_fixed_branch_residual"]) for row in selected]
            summary_rows.append(
                {
                    "switch_bucket": switch_bucket,
                    "pair_type": pair_type,
                    "num_observations": len(selected),
                    "direct_moving_r_squared": _r_squared(
                        actual_values=actual,
                        predicted_values=direct_predicted,
                    ),
                    "source_fixed_branch_r_squared": _r_squared(
                        actual_values=actual,
                        predicted_values=source_branch_predicted,
                    ),
                    "target_fixed_branch_r_squared": _r_squared(
                        actual_values=actual,
                        predicted_values=target_branch_predicted,
                    ),
                    "direct_moving_abs_residual_mean": _mean(
                        [abs(value) for value in direct_residuals],
                        label=f"{switch_bucket}/{pair_type}/direct residual",
                    ),
                    "source_fixed_branch_abs_residual_mean": _mean(
                        [abs(value) for value in source_branch_residuals],
                        label=f"{switch_bucket}/{pair_type}/source branch residual",
                    ),
                    "target_fixed_branch_abs_residual_mean": _mean(
                        [abs(value) for value in target_branch_residuals],
                        label=f"{switch_bucket}/{pair_type}/target branch residual",
                    ),
                }
            )
    return summary_rows


def _plot_branch_energy(
    *,
    summary_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    rows = [row for row in summary_rows if str(row["pair_type"]) == "__all__"]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    labels = [str(row["switch_bucket"]) for row in rows]
    target_values = [
        0.0 if row["target_branch_energy_fraction_of_moving"] is None else float(row["target_branch_energy_fraction_of_moving"])
        for row in rows
    ]
    source_values = [
        0.0 if row["source_branch_energy_fraction_of_moving"] is None else float(row["source_branch_energy_fraction_of_moving"])
        for row in rows
    ]
    x_values = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([x - width / 2 for x in x_values], target_values, width=width, label="target branch correction")
    ax.bar([x + width / 2 for x in x_values], source_values, width=width, label="source branch correction")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_ylabel("correction energy / moving-margin energy")
    ax.set_title("Branch correction size")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_branch_aware_closure(
    *,
    summary_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    rows = [row for row in summary_rows if str(row["pair_type"]) == "__all__"]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    labels = [str(row["switch_bucket"]) for row in rows]
    direct = [0.0 if row["direct_moving_r_squared"] is None else float(row["direct_moving_r_squared"]) for row in rows]
    source = [0.0 if row["source_fixed_branch_r_squared"] is None else float(row["source_fixed_branch_r_squared"]) for row in rows]
    target = [0.0 if row["target_fixed_branch_r_squared"] is None else float(row["target_fixed_branch_r_squared"]) for row in rows]
    x_values = list(range(len(labels)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([x - width for x in x_values], direct, width=width, label="direct moving prediction")
    ax.bar(x_values, source, width=width, label="fixed-source + branch")
    ax.bar([x + width for x in x_values], target, width=width, label="fixed-target + branch")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R squared")
    ax.set_title("Branch-aware moving-margin closure")
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_branch_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    lines = [
        "# Answer-Margin Branch Decomposition",
        "",
        "## Calculation",
        "",
        "This report checks whether moving answer margin is hard to close because the best wrong token changes.",
        "",
        "```text",
        "moving_margin = logit(correct) - logit(best_wrong_at_checkpoint)",
        "",
        "Delta moving_margin",
        "  = Delta fixed_source_margin",
        "    + [target_logit(source_wrong) - target_logit(target_wrong)]",
        "",
        "Delta moving_margin",
        "  = Delta fixed_target_margin",
        "    + [source_logit(source_wrong) - source_logit(target_wrong)]",
        "```",
        "",
        "## Inputs",
        "",
        f"- scalar pair rows: `{report['scalar_pair_rows_path']}`",
        f"- output closure rows: `{report['output_closure_rows_path']}`",
        f"- margin side: `{report['margin_side']}`",
        f"- pair types: `{report['pair_types']}`",
        "",
        "## Branch Size",
        "",
        "| bucket | pair type | observations | switch fraction | moving abs mean | target branch abs mean | source branch abs mean | target branch energy / moving | source branch energy / moving |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["summary"]["branch_summary_rows"]:
        target_energy = row["target_branch_energy_fraction_of_moving"]
        source_energy = row["source_branch_energy_fraction_of_moving"]
        lines.append(
            "| {bucket} | {pair_type} | {n} | {switch:.4f} | {moving:.6g} | {target_abs:.6g} | {source_abs:.6g} | {target_energy} | {source_energy} |".format(
                bucket=row["switch_bucket"],
                pair_type=row["pair_type"],
                n=int(row["num_observations"]),
                switch=float(row["competitor_switch_fraction"]),
                moving=float(row["moving_delta_abs_mean"]),
                target_abs=float(row["target_branch_correction_abs_mean"]),
                source_abs=float(row["source_branch_correction_abs_mean"]),
                target_energy="" if target_energy is None else f"{float(target_energy):.6g}",
                source_energy="" if source_energy is None else f"{float(source_energy):.6g}",
            )
        )
    if report["summary"].get("branch_aware_closure_summary_rows"):
        lines.extend(
            [
                "",
                "## Branch-Aware Closure",
                "",
                "| bucket | pair type | observations | direct moving R2 | fixed-source + branch R2 | fixed-target + branch R2 | direct abs residual | source-branch abs residual | target-branch abs residual |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["summary"]["branch_aware_closure_summary_rows"]:
            direct = row["direct_moving_r_squared"]
            source = row["source_fixed_branch_r_squared"]
            target = row["target_fixed_branch_r_squared"]
            lines.append(
                "| {bucket} | {pair_type} | {n} | {direct} | {source} | {target} | {direct_abs:.6g} | {source_abs:.6g} | {target_abs:.6g} |".format(
                    bucket=row["switch_bucket"],
                    pair_type=row["pair_type"],
                    n=int(row["num_observations"]),
                    direct="" if direct is None else f"{float(direct):.6f}",
                    source="" if source is None else f"{float(source):.6f}",
                    target="" if target is None else f"{float(target):.6f}",
                    direct_abs=float(row["direct_moving_abs_residual_mean"]),
                    source_abs=float(row["source_fixed_branch_abs_residual_mean"]),
                    target_abs=float(row["target_fixed_branch_abs_residual_mean"]),
                )
            )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- branch rows: `{report['branch_rows_path']}`",
            f"- branch-aware closure rows: `{report['branch_aware_closure_rows_path']}`",
            "",
            "## Plots",
            "",
        ]
    )
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_answer_margin_branch_decomposition(
    *,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    output_closure_rows_path: Path | None = None,
    margin_side: str | None = "clean",
    pair_types: list[str] | None = None,
    switch_buckets: list[str] | None = None,
    reconstruction_tolerance: float = 1.0e-5,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path | None, dict[str, Path]]:
    if reconstruction_tolerance < 0.0:
        raise ValueError("reconstruction_tolerance must be non-negative.")
    resolved_switch_buckets = _resolve_unique_values(
        values=switch_buckets,
        default_values=["all", "same_competitor", "competitor_switch"],
        allowed_values=ANSWER_MARGIN_BRANCH_SWITCH_BUCKETS,
        label="switch bucket",
    )
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scalar_pair_rows = _load_jsonl(scalar_pair_rows_path, label="scalar pair rows")
    branch_rows = build_branch_decomposition_rows(
        scalar_pair_rows=scalar_pair_rows,
        margin_side=margin_side,
        pair_types=pair_types,
        switch_buckets=resolved_switch_buckets,
        reconstruction_tolerance=reconstruction_tolerance,
    )
    branch_summary_rows = summarize_branch_decomposition(branch_rows)
    branch_aware_closure_rows: list[dict[str, Any]] = []
    branch_aware_closure_summary_rows: list[dict[str, Any]] = []
    if output_closure_rows_path is not None:
        output_closure_rows = _load_output_closure_rows(output_closure_rows_path)
        branch_aware_closure_rows = build_branch_aware_closure_rows(
            branch_rows=branch_rows,
            output_closure_rows=output_closure_rows,
        )
        branch_aware_closure_summary_rows = summarize_branch_aware_closure(branch_aware_closure_rows)
    branch_rows_path = output_dir / "answer_margin_branch_decomposition_rows.jsonl"
    branch_aware_rows_path = output_dir / "answer_margin_branch_aware_closure_rows.jsonl"
    report_path = output_dir / "answer_margin_branch_decomposition_report.json"
    markdown_path = output_dir / "answer_margin_branch_decomposition_report.md"
    write_jsonl(branch_rows_path, branch_rows)
    branch_aware_path: Path | None = None
    if branch_aware_closure_rows:
        write_jsonl(branch_aware_rows_path, branch_aware_closure_rows)
        branch_aware_path = branch_aware_rows_path
    plot_paths: dict[str, Path] = {}
    branch_energy_plot = _plot_branch_energy(
        summary_rows=branch_summary_rows,
        output_path=output_dir / "answer_margin_branch_energy.svg",
    )
    if branch_energy_plot is not None:
        plot_paths["branch_energy"] = branch_energy_plot
    if branch_aware_closure_summary_rows:
        branch_aware_plot = _plot_branch_aware_closure(
            summary_rows=branch_aware_closure_summary_rows,
            output_path=output_dir / "answer_margin_branch_aware_closure.svg",
        )
        if branch_aware_plot is not None:
            plot_paths["branch_aware_closure"] = branch_aware_plot
    report = {
        "schema_version": ANSWER_MARGIN_BRANCH_DECOMPOSITION_SCHEMA_VERSION,
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "output_closure_rows_path": None if output_closure_rows_path is None else str(output_closure_rows_path),
        "margin_side": margin_side,
        "pair_types": pair_types,
        "switch_buckets": resolved_switch_buckets,
        "reconstruction_tolerance": reconstruction_tolerance,
        "calculation": {
            "moving_margin": "correct logit - best wrong value logit at each checkpoint",
            "source_identity": "Delta moving = Delta fixed_source_margin + target_logit(source_wrong) - target_logit(target_wrong)",
            "target_identity": "Delta moving = Delta fixed_target_margin + source_logit(source_wrong) - source_logit(target_wrong)",
            "branch_aware_closure": (
                "If output closure rows are provided, compare direct moving-margin prediction with fixed-branch "
                "prediction plus exact branch correction."
            ),
        },
        "branch_rows_path": str(branch_rows_path),
        "branch_aware_closure_rows_path": None if branch_aware_path is None else str(branch_aware_path),
        "summary": {
            "branch_summary_rows": branch_summary_rows,
            "branch_aware_closure_summary_rows": branch_aware_closure_summary_rows,
        },
    }
    write_json(report_path, report)
    _write_branch_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    print(
        f"[answer-margin-branch-decomposition] complete report={report_path} rows={branch_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, branch_rows_path, branch_aware_path, plot_paths
