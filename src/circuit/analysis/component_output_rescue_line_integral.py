from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.component_output_rescue import (
    _build_patch_groups,
    _validate_downstream_components,
)
from circuit.analysis.component_output_rescue_adam_state_attribution import (
    _component_output_rescue_score_payloads,
    _request_maps,
)
from circuit.analysis.geometric_mechanisms import (
    _build_route_gradient_decomposition_groups,
    _checkpoint_step_from_path,
    _gradient_dot_summary,
    _group_metadata,
    _model_parameter_snapshot,
    _pair_metadata,
    _parameter_delta,
    _resolve_checkpoint_paths,
    _sign_match,
)
from circuit.analysis.output_route_closure import (
    OUTPUT_ROUTE_MARGIN_SIDES,
    OUTPUT_ROUTE_SCALARS,
    _component_labels,
    _filter_component_labels,
    _filter_scalar_pair_rows,
    _load_scalar_pair_rows,
    _mean,
    _resolve_unique_values,
    _selected_pairs_by_id,
)
from circuit.analysis.residual_state_rescue import _validate_maskable_components
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import load_checkpoint, load_model_state, require_device
from circuit.train import _resume_training_state
from circuit.vocab import Vocabulary


COMPONENT_OUTPUT_RESCUE_LINE_INTEGRAL_SCHEMA_VERSION = 1


def _alpha_key(alpha: float) -> float:
    return round(float(alpha), 12)


def _resolve_alpha_values(*, alpha_values: list[float] | None, alpha_grid_points: int) -> list[float]:
    if alpha_values is not None:
        if not alpha_values:
            raise ValueError("alpha_values must not be empty when provided.")
        values = sorted({_alpha_key(value) for value in alpha_values})
    else:
        if alpha_grid_points < 2:
            raise ValueError("alpha_grid_points must be at least 2.")
        values = [_alpha_key(index / float(alpha_grid_points - 1)) for index in range(alpha_grid_points)]
    if values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError(f"Alpha grid must include 0.0 and 1.0; got {values}.")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Alpha values must be in [0, 1], got {value}.")
    return values


def _resolve_gradient_alpha_values(
    *,
    alpha_values: list[float],
    gradient_alpha_values: list[float] | None,
    gradient_all_alphas: bool,
) -> list[float]:
    if gradient_all_alphas:
        if gradient_alpha_values:
            raise ValueError("Use either --gradient-all-alphas or --gradient-alpha, not both.")
        return list(alpha_values)
    if gradient_alpha_values is None:
        return []
    values = sorted({_alpha_key(value) for value in gradient_alpha_values})
    unknown = [value for value in values if value not in set(alpha_values)]
    if unknown:
        raise ValueError(f"Gradient alpha values must be present in the alpha grid. Missing: {unknown}")
    if values and (0.0 not in values or 1.0 not in values):
        raise ValueError("Gradient alpha values must include 0.0 and 1.0 for line-integral estimates.")
    return values


def _resolve_interpolation_groups(
    *,
    model: torch.nn.Module,
    interpolation_group_specs: list[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups, decomposition_summary = _build_route_gradient_decomposition_groups(
        model=model,
        decomposition_modes=["module_blocks", "attention_heads", "attention_projections"],
    )
    groups_by_id = {group.group_id: group for group in groups}
    raw_specs = ["global:all_named_parameters"] if interpolation_group_specs is None else list(interpolation_group_specs)
    if not raw_specs:
        raise ValueError("interpolation_group_specs must not be empty when provided.")
    resolved_specs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_spec in raw_specs:
        group_ids = [part.strip() for part in raw_spec.split(",") if part.strip()]
        if not group_ids:
            raise ValueError(f"Interpolation group is empty: {raw_spec!r}")
        duplicate_ids = sorted({group_id for group_id in group_ids if group_ids.count(group_id) > 1})
        if duplicate_ids:
            raise ValueError(f"Duplicate group id(s) in interpolation group {raw_spec!r}: {duplicate_ids}")
        missing = [group_id for group_id in group_ids if group_id not in groups_by_id]
        if missing:
            raise ValueError(f"Unknown interpolation group id(s) {missing}; available groups include {sorted(groups_by_id)}.")
        interpolation_group_id = "+".join(group_ids)
        if interpolation_group_id in seen_ids:
            raise ValueError(f"Duplicate interpolation group id: {interpolation_group_id}")
        seen_ids.add(interpolation_group_id)
        resolved_specs.append(
            {
                "interpolation_group_id": interpolation_group_id,
                "parameter_group_ids": group_ids,
                "parameter_groups": [groups_by_id[group_id] for group_id in group_ids],
            }
        )
    model_parameters = dict(model.named_parameters(remove_duplicate=False))
    group_rows: list[dict[str, Any]] = []
    for spec in resolved_specs:
        for group in spec["parameter_groups"]:
            row = _group_metadata(model_parameters=model_parameters, group=group)
            row["interpolation_group_id"] = spec["interpolation_group_id"]
            row["interpolation_parameter_group_ids"] = list(spec["parameter_group_ids"])
            group_rows.append(row)
    summary = {
        **decomposition_summary,
        "selected_interpolation_group_ids": [
            str(spec["interpolation_group_id"])
            for spec in resolved_specs
        ],
        "selected_interpolation_group_count": len(resolved_specs),
        "selected_interpolation_group_note": (
            "Each interpolation group is evaluated independently. Comma-separated specs interpolate the union of "
            "their selected parameter slices while all other parameters stay at the source checkpoint."
        ),
    }
    return resolved_specs, group_rows, summary


def _set_interpolated_parameters(
    *,
    model: torch.nn.Module,
    source_parameters: dict[str, torch.Tensor],
    delta_parameters: dict[str, torch.Tensor],
    alpha: float,
    parameter_groups: list[Any],
) -> None:
    selected_by_parameter: dict[str, list[tuple[Any, ...] | None]] = defaultdict(list)
    for group in parameter_groups:
        for selection in group.selections:
            selected_by_parameter[selection.parameter_name].append(selection.selector)
    seen_parameters: set[torch.nn.Parameter] = set()
    with torch.no_grad():
        for name, parameter in model.named_parameters(remove_duplicate=False):
            if parameter in seen_parameters:
                continue
            if name not in source_parameters:
                raise KeyError(f"Missing source parameter for interpolation: {name}")
            if name not in delta_parameters:
                raise KeyError(f"Missing delta parameter for interpolation: {name}")
            source = source_parameters[name]
            delta = delta_parameters[name]
            if source.shape != parameter.shape or delta.shape != parameter.shape:
                raise ValueError(
                    f"Interpolation shape mismatch for {name}: parameter={tuple(parameter.shape)} "
                    f"source={tuple(source.shape)} delta={tuple(delta.shape)}"
                )
            interpolated = source.clone()
            for selector in selected_by_parameter.get(name, []):
                if selector is None:
                    interpolated = source + float(alpha) * delta
                    break
                interpolated[selector] = source[selector] + float(alpha) * delta[selector]
            parameter.copy_(interpolated.to(device=parameter.device, dtype=parameter.dtype))
            seen_parameters.add(parameter)


def _scoped_delta_parameters(
    *,
    delta_parameters: dict[str, torch.Tensor],
    parameter_groups: list[Any],
) -> dict[str, torch.Tensor]:
    scoped = {name: torch.zeros_like(delta.float()) for name, delta in delta_parameters.items()}
    for group in parameter_groups:
        for selection in group.selections:
            name = selection.parameter_name
            if name not in scoped:
                raise KeyError(f"Selected interpolation group references unknown parameter: {name}")
            if selection.selector is None:
                scoped[name] = delta_parameters[name].float().clone()
            else:
                scoped[name][selection.selector] = delta_parameters[name].float()[selection.selector]
    return scoped


def _trapezoid_integral(points: list[tuple[float, float]]) -> float | None:
    if not points:
        return None
    sorted_points = sorted((_alpha_key(alpha), float(value)) for alpha, value in points)
    if sorted_points[0][0] != 0.0 or sorted_points[-1][0] != 1.0:
        raise ValueError(f"Line-integral derivative points must include alpha 0 and 1, got {sorted_points}.")
    total = 0.0
    for (left_alpha, left_value), (right_alpha, right_value) in zip(sorted_points[:-1], sorted_points[1:], strict=True):
        if right_alpha <= left_alpha:
            raise ValueError(f"Gradient alpha values must be strictly increasing, got {sorted_points}.")
        total += 0.5 * (left_value + right_value) * (right_alpha - left_alpha)
    return total


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def _summarize(
    *,
    curve_rows: list[dict[str, Any]],
    interval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not curve_rows:
        raise RuntimeError("Cannot summarize component-output-rescue line integral without curve rows.")
    if not interval_rows:
        raise RuntimeError("Cannot summarize component-output-rescue line integral without interval rows.")
    score_rows: list[dict[str, Any]] = []
    summary_keys = sorted(
        {
            (str(row["interpolation_group_id"]), str(row["score_name"]))
            for row in interval_rows
        }
    )
    for interpolation_group_id, score_name in summary_keys:
        rows = [
            row
            for row in interval_rows
            if str(row["interpolation_group_id"]) == interpolation_group_id
            and str(row["score_name"]) == score_name
        ]
        line_integral_values = [row["source_endpoint_line_integral_delta"] for row in rows]
        line_integral_complete = all(value is not None for value in line_integral_values)
        source_first_order_values = [row["source_endpoint_first_order_delta"] for row in rows]
        first_order_complete = all(value is not None for value in source_first_order_values)
        score_rows.append(
            {
                "score_name": score_name,
                "interpolation_group_id": interpolation_group_id,
                "interpolation_parameter_group_ids": list(rows[0]["interpolation_parameter_group_ids"]),
                "patch_group_id": str(rows[0]["patch_group_id"]),
                "scalar_name": str(rows[0]["scalar_name"]),
                "num_intervals": len(rows),
                "sum_actual_endpoint_delta": sum(float(row["actual_endpoint_delta"]) for row in rows),
                "sum_source_fixed_curve_delta": sum(float(row["source_fixed_curve_delta"]) for row in rows),
                "sum_target_fixed_curve_delta": sum(float(row["target_fixed_curve_delta"]) for row in rows),
                "sum_branch_gap_at_target": sum(float(row["branch_gap_at_target"]) for row in rows),
                "sum_branch_gap_at_source": sum(float(row["branch_gap_at_source"]) for row in rows),
                "sum_source_endpoint_first_order_delta": (
                    sum(float(value) for value in source_first_order_values)
                    if first_order_complete
                    else None
                ),
                "sum_source_endpoint_line_integral_delta": (
                    sum(float(value) for value in line_integral_values)
                    if line_integral_complete
                    else None
                ),
                "sum_source_curvature_gap_vs_first_order": (
                    sum(float(row["source_curvature_gap_vs_first_order"]) for row in rows)
                    if first_order_complete
                    else None
                ),
                "sum_source_line_integral_residual": (
                    sum(float(row["source_line_integral_residual"]) for row in rows)
                    if line_integral_complete
                    else None
                ),
                "mean_source_max_abs_chord_residual": _mean(
                    [float(row["source_max_abs_chord_residual"]) for row in rows],
                    label=f"{score_name} source chord residual",
                ),
                "mean_target_max_abs_chord_residual": _mean(
                    [float(row["target_max_abs_chord_residual"]) for row in rows],
                    label=f"{score_name} target chord residual",
                ),
                "source_first_order_sign_match_rate": (
                    _mean(
                        [
                            1.0 if bool(row["source_first_order_sign_match"]) else 0.0
                            for row in rows
                        ],
                        label=f"{score_name} source first-order sign match",
                    )
                    if first_order_complete
                    else None
                ),
                "source_line_integral_sign_match_rate": (
                    _mean(
                        [
                            1.0 if bool(row["source_line_integral_sign_match"]) else 0.0
                            for row in rows
                        ],
                        label=f"{score_name} source line-integral sign match",
                    )
                    if line_integral_complete
                    else None
                ),
            }
        )
    return {"score_rows": score_rows}


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Component Output Rescue Line Integral",
        "",
        "This report evaluates the behavioral component-output rescue scalar along the actual parameter update path.",
        "",
        "The scalar is fixed to the same rescue definition:",
        "",
        "`C = mean[scalar(source ablated + clean downstream component write patch) - scalar(source ablated)]`",
        "",
        "For each interval, the tool evaluates `theta(alpha) = theta_t + alpha * (theta_{t+1} - theta_t)`.",
        "",
        "The source-endpoint curve measures parameter-space curvature with the scalar branch held fixed.",
        "The target-vs-source endpoint gap measures scalar/competitor branch effects.",
        "",
        "## Summary",
        "",
        "| interpolation group | score | intervals | actual endpoint delta | source fixed curve | branch gap at target | source first-order | source line integral | first-order curvature gap | line residual | source chord residual |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]["score_rows"]:
        lines.append(
            "| `{group}` | `{score}` | {n} | {actual:.6g} | {source:.6g} | {branch:.6g} | {first} | {integral} | {curvature} | {residual} | {chord:.6g} |".format(
                group=row["interpolation_group_id"],
                score=row["score_name"],
                n=int(row["num_intervals"]),
                actual=float(row["sum_actual_endpoint_delta"]),
                source=float(row["sum_source_fixed_curve_delta"]),
                branch=float(row["sum_branch_gap_at_target"]),
                first=_format_optional(row["sum_source_endpoint_first_order_delta"]),
                integral=_format_optional(row["sum_source_endpoint_line_integral_delta"]),
                curvature=_format_optional(row["sum_source_curvature_gap_vs_first_order"]),
                residual=_format_optional(row["sum_source_line_integral_residual"]),
                chord=float(row["mean_source_max_abs_chord_residual"]),
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- curve rows: `{report['curve_rows_path']}`",
            f"- interval rows: `{report['interval_rows_path']}`",
            f"- group rows: `{report['group_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_component_output_rescue_line_integral(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    optimizer_trace_dir: Path,
    output_dir: Path,
    pair_types: list[str],
    source_component: str,
    patch_components: list[str] | None,
    patch_groups: list[str] | None = None,
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
    interval_stride: int = 1,
    alpha_values: list[float] | None = None,
    alpha_grid_points: int = 5,
    gradient_alpha_values: list[float] | None = None,
    gradient_all_alphas: bool = False,
    interpolation_group_specs: list[str] | None = None,
    scalar_names: list[str] | None = None,
    margin_sides: list[str] | None = None,
    split_filter: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    batch_size: int | None = None,
    scalar_value_tolerance: float = 1.0e-4,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    if not pair_types:
        raise ValueError("component-output-rescue-line-integral requires at least one pair type.")
    if max_pairs_per_type <= 0:
        raise ValueError("max_pairs_per_type must be positive.")
    if min_pairs_per_type <= 0:
        raise ValueError("min_pairs_per_type must be positive.")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    if interval_stride <= 0:
        raise ValueError("interval_stride must be positive.")
    if scalar_value_tolerance < 0.0:
        raise ValueError("scalar_value_tolerance must be non-negative.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = _resolve_alpha_values(alpha_values=alpha_values, alpha_grid_points=alpha_grid_points)
    gradient_alphas = _resolve_gradient_alpha_values(
        alpha_values=alphas,
        gradient_alpha_values=gradient_alpha_values,
        gradient_all_alphas=gradient_all_alphas,
    )
    gradient_alpha_set = set(gradient_alphas)

    resolved_scalars = _resolve_unique_values(
        values=scalar_names,
        default_values=["correct_value_logit", "fixed_source_competitor_margin", "fixed_target_competitor_margin"],
        allowed_values=OUTPUT_ROUTE_SCALARS,
        label="scalar",
    )
    resolved_margin_sides = _resolve_unique_values(
        values=margin_sides,
        default_values=["clean"],
        allowed_values=OUTPUT_ROUTE_MARGIN_SIDES,
        label="margin side",
    )
    pair_types = sorted(set(pair_types), key=pair_types.index)
    spec = TrainSpec.from_path(config_path)
    if device_name is not None:
        spec = replace(spec, device=device_name)
    if float(spec.model.dropout) != 0.0:
        raise RuntimeError("Component-output rescue line integral requires dropout=0.0.")
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(spec.device)
    analysis_batch_size = int(spec.evaluation.batch_size) if batch_size is None else int(batch_size)

    checkpoint_dir = optimizer_trace_dir / "checkpoints"
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if start_step is not None:
        checkpoints = [path for path in checkpoints if _checkpoint_step_from_path(path) >= start_step]
    if end_step is not None:
        checkpoints = [path for path in checkpoints if _checkpoint_step_from_path(path) <= end_step]
    if len(checkpoints) < 2:
        raise ValueError("component-output-rescue-line-integral requires at least two trace checkpoints.")
    all_intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    for source_checkpoint_path, target_checkpoint_path in all_intervals:
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        if target_step - source_step != 1:
            raise RuntimeError(
                "Component-output rescue line integral requires one-step checkpoint intervals. "
                f"Got {source_step}->{target_step}."
            )
    intervals = [interval for index, interval in enumerate(all_intervals) if index % interval_stride == 0]
    if not intervals:
        raise RuntimeError("Interval stride selected no intervals.")

    selected_interval_keys = {
        (_checkpoint_step_from_path(source), _checkpoint_step_from_path(target))
        for source, target in intervals
    }
    scalar_pair_rows = _filter_scalar_pair_rows(
        rows=_load_scalar_pair_rows(scalar_pair_rows_path),
        margin_sides=resolved_margin_sides,
        pair_types=pair_types,
        scalar_names=resolved_scalars,
    )
    scalar_pair_rows = [
        row
        for row in scalar_pair_rows
        if (int(row["source_step"]), int(row["target_step"])) in selected_interval_keys
    ]
    if not scalar_pair_rows:
        raise RuntimeError("No scalar pair rows survived selected interval filters.")
    rows_by_interval: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in scalar_pair_rows:
        source_step = int(row["source_step"])
        target_step = int(row["target_step"])
        if target_step - source_step != 1:
            raise RuntimeError(f"Scalar pair row is not one-step: {source_step}->{target_step}")
        rows_by_interval[(source_step, target_step)].append(row)
    missing_interval_rows = [
        (int(_checkpoint_step_from_path(source)), int(_checkpoint_step_from_path(target)))
        for source, target in intervals
        if (int(_checkpoint_step_from_path(source)), int(_checkpoint_step_from_path(target))) not in rows_by_interval
    ]
    if missing_interval_rows:
        raise RuntimeError(f"Missing scalar pair rows for interval(s): {missing_interval_rows[:10]}")

    required_pair_ids = {str(row["pair_id"]) for row in scalar_pair_rows}
    pairs_by_id, pair_construction = _selected_pairs_by_id(
        config_path=config_path,
        probe_set_path=probe_set_path,
        pair_types=pair_types,
        split_filter=split_filter,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        required_pair_ids=required_pair_ids,
    )

    context = _resume_training_state(spec=spec, resume_checkpoint=intervals[0][0])
    model: torch.nn.Module = context["model"]
    model.eval()
    available_components = _component_labels(num_layers=len(model.blocks), num_heads=int(model.spec.n_heads))
    resolved_sources = _filter_component_labels(
        requested_components=[source_component],
        available_components=available_components,
    )
    if len(resolved_sources) != 1:
        raise RuntimeError(f"Expected exactly one source component, got {resolved_sources}.")
    resolved_source = resolved_sources[0]
    resolved_patch_groups = _build_patch_groups(
        patch_components=patch_components,
        patch_groups=patch_groups,
        available_components=available_components,
    )
    resolved_patch_components: list[str] = []
    for patch_group in resolved_patch_groups:
        for component in patch_group["patch_components"]:
            if component not in resolved_patch_components:
                resolved_patch_components.append(str(component))
    overlap = sorted({resolved_source} & set(resolved_patch_components))
    if overlap:
        raise ValueError(f"Source and patch component sets must be disjoint; overlap={overlap}")
    _validate_maskable_components(
        components=[resolved_source],
        num_layers=len(model.blocks),
        num_heads=int(model.spec.n_heads),
        device=device,
    )
    _validate_maskable_components(
        components=resolved_patch_components,
        num_layers=len(model.blocks),
        num_heads=int(model.spec.n_heads),
        device=device,
    )
    _validate_downstream_components(source_components=[resolved_source], patch_components=resolved_patch_components)
    interpolation_specs, group_rows, interpolation_summary = _resolve_interpolation_groups(
        model=model,
        interpolation_group_specs=interpolation_group_specs,
    )

    curve_rows_path = output_dir / "component_output_rescue_line_integral_rows.jsonl"
    interval_rows_path = output_dir / "component_output_rescue_line_integral_interval_rows.jsonl"
    group_rows_path = output_dir / "component_output_rescue_line_integral_groups.jsonl"
    pair_rows_path = output_dir / "component_output_rescue_line_integral_pairs.jsonl"
    report_path = output_dir / "component_output_rescue_line_integral_report.json"
    markdown_path = output_dir / "component_output_rescue_line_integral_report.md"
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs_by_id.values()])
    write_jsonl(group_rows_path, group_rows)

    print(
        "[component-output-rescue-line-integral] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} stride={interval_stride} "
        f"pairs={len(pairs_by_id)} source={resolved_source} patch_groups={resolved_patch_groups} "
        f"scalars={resolved_scalars} interpolation_groups={interpolation_summary['selected_interpolation_group_ids']} "
        f"alphas={alphas} gradient_alphas={gradient_alphas} device={spec.device}",
        flush=True,
    )

    curve_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        print(
            "[component-output-rescue-line-integral] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )

        source_checkpoint = load_checkpoint(source_checkpoint_path, device)
        target_checkpoint = load_checkpoint(target_checkpoint_path, device)
        if int(source_checkpoint["step"]) != source_step:
            raise RuntimeError(f"Source checkpoint step mismatch: payload={source_checkpoint['step']} path={source_step}")
        if int(target_checkpoint["step"]) != target_step:
            raise RuntimeError(f"Target checkpoint step mismatch: payload={target_checkpoint['step']} path={target_step}")
        load_model_state(model, source_checkpoint["model_state"])
        source_parameters = _model_parameter_snapshot(model)
        load_model_state(model, target_checkpoint["model_state"])
        target_parameters = _model_parameter_snapshot(model)
        delta_parameters = _parameter_delta(
            source_parameters=source_parameters,
            target_parameters=target_parameters,
            label=f"component-output rescue line integral {source_step}->{target_step}",
        )
        interval_scalar_rows = rows_by_interval[(source_step, target_step)]
        source_requests, source_request_by_id = _request_maps(
            scalar_pair_rows=interval_scalar_rows,
            scalar_names=resolved_scalars,
            endpoint_kind="source",
        )
        target_requests, target_request_by_id = _request_maps(
            scalar_pair_rows=interval_scalar_rows,
            scalar_names=resolved_scalars,
            endpoint_kind="target",
        )

        for interpolation_spec in interpolation_specs:
            interpolation_group_id = str(interpolation_spec["interpolation_group_id"])
            interpolation_parameter_group_ids = list(interpolation_spec["parameter_group_ids"])
            interpolation_parameter_groups = list(interpolation_spec["parameter_groups"])
            scoped_delta_parameters = _scoped_delta_parameters(
                delta_parameters=delta_parameters,
                parameter_groups=interpolation_parameter_groups,
            )
            for alpha in alphas:
                _set_interpolated_parameters(
                    model=model,
                    source_parameters=source_parameters,
                    delta_parameters=delta_parameters,
                    alpha=alpha,
                    parameter_groups=interpolation_parameter_groups,
                )
                for endpoint_kind, requests, request_by_id in (
                    ("source", source_requests, source_request_by_id),
                    ("target", target_requests, target_request_by_id),
                ):
                    should_track_grad = alpha in gradient_alpha_set
                    validate_clean_scalar = (
                        interpolation_group_id == "global:all_named_parameters"
                        and (
                            (endpoint_kind == "source" and alpha == 0.0)
                            or (endpoint_kind == "target" and alpha == 1.0)
                        )
                    )
                    payloads = _component_output_rescue_score_payloads(
                        model=model,
                        pairs_by_id=pairs_by_id,
                        requests=requests,
                        request_by_id=request_by_id,
                        source_component=resolved_source,
                        patch_groups=resolved_patch_groups,
                        batch_size=analysis_batch_size,
                        pad_token_id=vocab.pad_token_id,
                        scalar_value_tolerance=scalar_value_tolerance,
                        device=device,
                        track_grad=should_track_grad,
                        validate_clean_scalar=validate_clean_scalar,
                    )
                    for score_name, payload in sorted(payloads.items()):
                        gradient_dot_delta = None
                        gradient_l2_norm = None
                        delta_l2_norm = None
                        gradient_delta_cosine = None
                        zero_gradient_parameter_names: list[str] = []
                        if should_track_grad:
                            gradients = payload.get("gradients")
                            if not isinstance(gradients, dict):
                                raise TypeError(f"Missing gradients for {endpoint_kind} alpha={alpha} score={score_name}.")
                            dot_summary = _gradient_dot_summary(
                                left_gradients=scoped_delta_parameters,
                                right_gradients=gradients,
                                label=(
                                    f"component-output rescue line integral {interpolation_group_id} "
                                    f"{endpoint_kind} alpha={alpha} {score_name} {source_step}->{target_step}"
                                ),
                            )
                            gradient_dot_delta = float(dot_summary["dot"])
                            delta_l2_norm = float(dot_summary["left_l2_norm"])
                            gradient_l2_norm = float(dot_summary["right_l2_norm"])
                            gradient_delta_cosine = dot_summary["cosine"]
                            zero_gradient_parameter_names = list(payload["zero_gradient_parameter_names"])
                        curve_rows.append(
                            {
                                "source_step": source_step,
                                "target_step": target_step,
                                "step_gap": target_step - source_step,
                                "source_checkpoint": str(source_checkpoint_path),
                                "target_checkpoint": str(target_checkpoint_path),
                                "optimizer_trace_dir": str(optimizer_trace_dir),
                                "interpolation_group_id": interpolation_group_id,
                                "interpolation_parameter_group_ids": interpolation_parameter_group_ids,
                                "source_component": resolved_source,
                                "patch_group_id": str(payload["patch_group_id"]),
                                "patch_components": next(
                                    list(group["patch_components"])
                                    for group in resolved_patch_groups
                                    if str(group["patch_group_id"]) == str(payload["patch_group_id"])
                                ),
                                "score_name": score_name,
                                "scalar_name": str(payload["scalar_name"]),
                                "endpoint_kind": endpoint_kind,
                                "alpha": alpha,
                                "score_value": float(payload["score_value"]),
                                "clean_score_value": float(payload["clean_score_value"]),
                                "ablated_score_value": float(payload["ablated_score_value"]),
                                "patched_score_value": float(payload["patched_score_value"]),
                                "num_entries": int(payload["num_entries"]),
                                "num_pairs": int(payload["num_pairs"]),
                                "gradient_dot_delta": gradient_dot_delta,
                                "gradient_l2_norm": gradient_l2_norm,
                                "delta_l2_norm": delta_l2_norm,
                                "gradient_delta_cosine": gradient_delta_cosine,
                                "zero_gradient_parameter_count": len(zero_gradient_parameter_names),
                                "zero_gradient_parameter_names": zero_gradient_parameter_names,
                            }
                        )

        current_rows = [
            row
            for row in curve_rows
            if int(row["source_step"]) == source_step and int(row["target_step"]) == target_step
        ]
        values_by_key = {
            (
                str(row["interpolation_group_id"]),
                str(row["endpoint_kind"]),
                str(row["score_name"]),
                _alpha_key(float(row["alpha"])),
            ): row
            for row in current_rows
        }
        interval_score_keys = sorted(
            {
                (str(row["interpolation_group_id"]), str(row["score_name"]))
                for row in current_rows
            }
        )
        for interpolation_group_id, score_name in interval_score_keys:
            source_zero = values_by_key[(interpolation_group_id, "source", score_name, 0.0)]
            source_one = values_by_key[(interpolation_group_id, "source", score_name, 1.0)]
            target_zero = values_by_key[(interpolation_group_id, "target", score_name, 0.0)]
            target_one = values_by_key[(interpolation_group_id, "target", score_name, 1.0)]
            source_fixed_delta = float(source_one["score_value"]) - float(source_zero["score_value"])
            target_fixed_delta = float(target_one["score_value"]) - float(target_zero["score_value"])
            actual_endpoint_delta = float(target_one["score_value"]) - float(source_zero["score_value"])
            branch_gap_at_target = float(target_one["score_value"]) - float(source_one["score_value"])
            branch_gap_at_source = float(target_zero["score_value"]) - float(source_zero["score_value"])

            def chord_stats(endpoint_kind: str) -> tuple[float, float]:
                zero = values_by_key[(interpolation_group_id, endpoint_kind, score_name, 0.0)]
                one = values_by_key[(interpolation_group_id, endpoint_kind, score_name, 1.0)]
                start_value = float(zero["score_value"])
                end_value = float(one["score_value"])
                residuals = []
                for alpha in alphas:
                    row = values_by_key[(interpolation_group_id, endpoint_kind, score_name, alpha)]
                    expected = start_value + alpha * (end_value - start_value)
                    residuals.append(abs(float(row["score_value"]) - expected))
                return max(residuals), _mean(residuals, label=f"{endpoint_kind} {score_name} chord residual")

            source_max_chord, source_mean_chord = chord_stats("source")
            target_max_chord, target_mean_chord = chord_stats("target")

            source_gradient_points = [
                (float(row["alpha"]), float(row["gradient_dot_delta"]))
                for row in current_rows
                if str(row["interpolation_group_id"]) == interpolation_group_id
                and str(row["endpoint_kind"]) == "source"
                and str(row["score_name"]) == score_name
                and row["gradient_dot_delta"] is not None
            ]
            source_line_integral = _trapezoid_integral(source_gradient_points) if source_gradient_points else None
            source_first_order = None
            source_endpoint_gradient = None
            for alpha, value in source_gradient_points:
                if _alpha_key(alpha) == 0.0:
                    source_first_order = value
                if _alpha_key(alpha) == 1.0:
                    source_endpoint_gradient = value
            source_curvature_gap = (
                source_fixed_delta - source_first_order if source_first_order is not None else None
            )
            source_line_residual = (
                source_fixed_delta - source_line_integral if source_line_integral is not None else None
            )
            interval_rows.append(
                {
                    "source_step": source_step,
                    "target_step": target_step,
                    "step_gap": target_step - source_step,
                    "source_checkpoint": str(source_checkpoint_path),
                    "target_checkpoint": str(target_checkpoint_path),
                    "optimizer_trace_dir": str(optimizer_trace_dir),
                    "interpolation_group_id": interpolation_group_id,
                    "interpolation_parameter_group_ids": list(source_zero["interpolation_parameter_group_ids"]),
                    "source_component": resolved_source,
                    "patch_group_id": str(source_zero["patch_group_id"]),
                    "patch_components": list(source_zero["patch_components"]),
                    "score_name": score_name,
                    "scalar_name": str(source_zero["scalar_name"]),
                    "actual_endpoint_delta": actual_endpoint_delta,
                    "source_fixed_curve_delta": source_fixed_delta,
                    "target_fixed_curve_delta": target_fixed_delta,
                    "branch_gap_at_target": branch_gap_at_target,
                    "branch_gap_at_source": branch_gap_at_source,
                    "source_endpoint_first_order_delta": source_first_order,
                    "source_endpoint_gradient_delta": source_endpoint_gradient,
                    "source_endpoint_line_integral_delta": source_line_integral,
                    "source_curvature_gap_vs_first_order": source_curvature_gap,
                    "source_line_integral_residual": source_line_residual,
                    "source_first_order_sign_match": (
                        _sign_match(source_fixed_delta, source_first_order)
                        if source_first_order is not None
                        else None
                    ),
                    "source_line_integral_sign_match": (
                        _sign_match(source_fixed_delta, source_line_integral)
                        if source_line_integral is not None
                        else None
                    ),
                    "source_max_abs_chord_residual": source_max_chord,
                    "source_mean_abs_chord_residual": source_mean_chord,
                    "target_max_abs_chord_residual": target_max_chord,
                    "target_mean_abs_chord_residual": target_mean_chord,
                }
            )

        primary = max(
            [row for row in interval_rows if int(row["source_step"]) == source_step and int(row["target_step"]) == target_step],
            key=lambda row: abs(float(row["actual_endpoint_delta"])),
        )
        print(
            "[component-output-rescue-line-integral] finished "
            f"{source_step}->{target_step} group={primary['interpolation_group_id']} score={primary['score_name']} "
            f"actual={float(primary['actual_endpoint_delta']):.6g} "
            f"source_curve={float(primary['source_fixed_curve_delta']):.6g} "
            f"branch_gap={float(primary['branch_gap_at_target']):.6g}",
            flush=True,
        )

    write_jsonl(curve_rows_path, curve_rows)
    write_jsonl(interval_rows_path, interval_rows)
    summary = _summarize(curve_rows=curve_rows, interval_rows=interval_rows)
    report = {
        "schema_version": COMPONENT_OUTPUT_RESCUE_LINE_INTEGRAL_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "optimizer_trace_dir": str(optimizer_trace_dir),
        "output_dir": str(output_dir),
        "device": spec.device,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "selected_intervals": [
            f"{_checkpoint_step_from_path(source)}->{_checkpoint_step_from_path(target)}"
            for source, target in intervals
        ],
        "interval_stride": interval_stride,
        "alpha_values": alphas,
        "gradient_alpha_values": gradient_alphas,
        "interpolation_summary": interpolation_summary,
        "interpolation_groups": [
            {
                "interpolation_group_id": str(spec["interpolation_group_id"]),
                "parameter_group_ids": list(spec["parameter_group_ids"]),
            }
            for spec in interpolation_specs
        ],
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names": resolved_scalars,
        "source_component": resolved_source,
        "patch_groups": resolved_patch_groups,
        "patch_components": resolved_patch_components,
        "batch_size": analysis_batch_size,
        "scalar_value_tolerance": scalar_value_tolerance,
        "pair_construction": pair_construction,
        "calculation": {
            "theta_alpha": "theta_t + alpha * (theta_{t+1} - theta_t)",
            "source_fixed_curve_delta": "score(theta_1, source endpoint scalar branch) - score(theta_0, source endpoint scalar branch)",
            "actual_endpoint_delta": "score(theta_1, target endpoint scalar branch) - score(theta_0, source endpoint scalar branch)",
            "branch_gap_at_target": "score(theta_1, target endpoint scalar branch) - score(theta_1, source endpoint scalar branch)",
            "source_endpoint_first_order_delta": "grad_theta score(theta_0, source endpoint scalar branch) dot (theta_1 - theta_0)",
            "source_endpoint_line_integral_delta": "trapezoid integral over alpha of grad_theta score(theta_alpha, source endpoint scalar branch) dot (theta_1 - theta_0)",
        },
        "curve_rows_path": str(curve_rows_path),
        "interval_rows_path": str(interval_rows_path),
        "group_rows_path": str(group_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(f"[component-output-rescue-line-integral] complete report={report_path} rows={curve_rows_path}", flush=True)
    return report_path, markdown_path, curve_rows_path, interval_rows_path, group_rows_path, pair_rows_path
