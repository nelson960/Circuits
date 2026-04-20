from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import shutil
from typing import Any

import torch

from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import _checkpoint_step_from_path, _validate_single_query_batch
from circuit.analysis.output_component_causal_validation import _component_mask_kwargs, _scalar_from_logits
from circuit.analysis.output_route_closure import (
    OUTPUT_ROUTE_MARGIN_SIDES,
    OUTPUT_ROUTE_SCALARS,
    _build_endpoint_requests,
    _checkpoint_paths_by_step,
    _component_labels,
    _filter_component_labels,
    _filter_scalar_pair_rows,
    _load_scalar_pair_rows,
    _mean,
    _resolve_unique_values,
    _safe_r_squared,
    _selected_pairs_by_id,
)
from circuit.analysis.shared_feature_dynamics import _import_matplotlib
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


RESIDUAL_STATE_RESCUE_SCHEMA_VERSION = 1
RESIDUAL_STATE_RESCUE_ENDPOINT_ROLES = ["source", "target"]


def _available_residual_stages(num_layers: int) -> list[str]:
    stages = ["embedding"]
    for layer_index in range(num_layers):
        stages.append(f"layer_{layer_index}_post_attn")
        stages.append(f"layer_{layer_index}_post_mlp")
    stages.append("final_norm")
    return stages


def _resolve_patch_stages(*, patch_stages: list[str], num_layers: int) -> list[str]:
    if not patch_stages:
        raise ValueError("patch_stages must not be empty.")
    available = _available_residual_stages(num_layers)
    resolved: list[str] = []
    for stage in patch_stages:
        if stage not in available:
            raise ValueError(f"Unsupported patch stage {stage!r}; expected one of {available}.")
        if stage not in resolved:
            resolved.append(stage)
    return resolved


def _safe_correlation(x_values: list[float], y_values: list[float]) -> float | None:
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have same length.")
    if len(x_values) < 2:
        return None
    mean_x = _mean(x_values, label="correlation x")
    mean_y = _mean(y_values, label="correlation y")
    x_centered = [value - mean_x for value in x_values]
    y_centered = [value - mean_y for value in y_values]
    denom_x = math.sqrt(sum(value * value for value in x_centered))
    denom_y = math.sqrt(sum(value * value for value in y_centered))
    if denom_x <= 1.0e-12 or denom_y <= 1.0e-12:
        return None
    return float(sum(x * y for x, y in zip(x_centered, y_centered, strict=True)) / (denom_x * denom_y))


def _validate_maskable_components(
    *,
    components: list[str],
    num_layers: int,
    num_heads: int,
    device: torch.device,
) -> None:
    for component in components:
        _component_mask_kwargs(component=component, num_layers=num_layers, num_heads=num_heads, device=device)


def _compute_scalar_payloads(
    *,
    answer_logits: torch.Tensor,
    answer_targets: torch.Tensor,
    metadata: dict[str, Any],
    batch_pair_ids: list[str],
    request_specs_by_pair_id: dict[str, list[dict[str, Any]]],
    label: str,
) -> dict[tuple[int, int, str, str, str, str], float]:
    if len(metadata["rows"]) != len(batch_pair_ids):
        raise RuntimeError(
            f"{label} expected one answer row per pair: pairs={len(batch_pair_ids)} rows={len(metadata['rows'])}"
        )
    values: dict[tuple[int, int, str, str, str, str], float] = {}
    for item_index, pair_id in enumerate(batch_pair_ids):
        if pair_id not in request_specs_by_pair_id:
            continue
        target_id = int(answer_targets[item_index].detach().cpu().item())
        logits = answer_logits[item_index].detach().float().cpu()
        for request in request_specs_by_pair_id[pair_id]:
            expected_target = int(request["answer_target_id"])
            if target_id != expected_target:
                raise RuntimeError(
                    f"{label} answer target mismatch for pair={pair_id}: expected={expected_target} got={target_id}"
                )
            request_id = request["request_id"]
            values[request_id] = _scalar_from_logits(
                logits=logits,
                scalar_name=str(request["scalar_name"]),
                answer_target_id=expected_target,
                source_best_wrong_token_id=int(request["source_best_wrong_token_id"]),
                target_best_wrong_token_id=int(request["target_best_wrong_token_id"]),
                endpoint_kind=str(request["endpoint_kind"]),
            )
    return values


def _compute_rescue_rows_for_source(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    endpoint_keys: set[tuple[int, str]],
    endpoint_requests: list[dict[str, Any]],
    source_component: str,
    patch_stages: list[str],
    batch_size: int,
    pad_token_id: int,
    scalar_value_tolerance: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    num_layers = len(model.blocks)
    num_heads = model.spec.n_heads
    source_mask_kwargs = _component_mask_kwargs(
        component=source_component,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
    )
    request_specs: dict[tuple[int, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    request_by_id: dict[tuple[int, int, str, str, str, str], dict[str, Any]] = {}
    for request in endpoint_requests:
        key = (int(request["step"]), str(request["margin_side"]))
        request_specs[key][str(request["pair_id"])].append(request)
        request_by_id[request["request_id"]] = request
    pair_ids = sorted(pairs_by_id)
    rows: list[dict[str, Any]] = []
    for step, margin_side in sorted(endpoint_keys):
        if step not in checkpoint_paths_by_step:
            raise KeyError(f"No checkpoint path for step {step}.")
        if margin_side not in OUTPUT_ROUTE_MARGIN_SIDES:
            raise ValueError(f"Unsupported margin side {margin_side!r}; expected one of {OUTPUT_ROUTE_MARGIN_SIDES}.")
        checkpoint_path = checkpoint_paths_by_step[step]
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        payload_step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if payload_step != step or path_step != step:
            raise RuntimeError(
                f"Checkpoint step mismatch for {checkpoint_path}: requested={step} payload={payload_step} path={path_step}"
            )
        side_key = f"{margin_side}_record"
        specs_for_endpoint = request_specs[(step, margin_side)]
        for start_index in range(0, len(pair_ids), batch_size):
            batch_pair_ids = pair_ids[start_index : start_index + batch_size]
            records = [pairs_by_id[pair_id][side_key] for pair_id in batch_pair_ids]
            batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
            with torch.no_grad():
                clean_outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_residual_streams=True,
                )
                ablated_outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_residual_streams=True,
                    **source_mask_kwargs,
                )
            if clean_outputs.residual_streams is None or ablated_outputs.residual_streams is None:
                raise RuntimeError("Residual-state rescue requires residual streams.")
            clean_logits, clean_targets, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            ablated_logits, ablated_targets, ablated_metadata = extract_answer_logits(ablated_outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=clean_metadata, label="residual-state rescue clean")
            _validate_single_query_batch(batch=batch, metadata=ablated_metadata, label="residual-state rescue ablated")
            request_specs_by_pair_id = {
                pair_id: specs_for_endpoint[pair_id]
                for pair_id in batch_pair_ids
                if pair_id in specs_for_endpoint
            }
            if not request_specs_by_pair_id:
                continue
            clean_values = _compute_scalar_payloads(
                answer_logits=clean_logits,
                answer_targets=clean_targets,
                metadata=clean_metadata,
                batch_pair_ids=batch_pair_ids,
                request_specs_by_pair_id=request_specs_by_pair_id,
                label="residual-state rescue clean",
            )
            ablated_values = _compute_scalar_payloads(
                answer_logits=ablated_logits,
                answer_targets=ablated_targets,
                metadata=ablated_metadata,
                batch_pair_ids=batch_pair_ids,
                request_specs_by_pair_id=request_specs_by_pair_id,
                label="residual-state rescue ablated",
            )
            patched_values_by_stage: dict[str, dict[tuple[int, int, str, str, str, str], float]] = {}
            for patch_stage in patch_stages:
                clean_patch = clean_outputs.residual_streams[patch_stage].detach()
                with torch.no_grad():
                    patched_outputs = model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        residual_patch={patch_stage: clean_patch},
                        **source_mask_kwargs,
                    )
                patched_logits, patched_targets, patched_metadata = extract_answer_logits(patched_outputs.logits, batch)
                _validate_single_query_batch(
                    batch=batch,
                    metadata=patched_metadata,
                    label=f"residual-state rescue patched {patch_stage}",
                )
                patched_values_by_stage[patch_stage] = _compute_scalar_payloads(
                    answer_logits=patched_logits,
                    answer_targets=patched_targets,
                    metadata=patched_metadata,
                    batch_pair_ids=batch_pair_ids,
                    request_specs_by_pair_id=request_specs_by_pair_id,
                    label=f"residual-state rescue patched {patch_stage}",
                )
            for request_id, clean_scalar in clean_values.items():
                request = request_by_id[request_id]
                scalar_payload = request.get("scalar_payload")
                if scalar_payload is not None:
                    expected = float(scalar_payload[str(request["endpoint_kind"])])
                    delta = abs(clean_scalar - expected)
                    if delta > scalar_value_tolerance:
                        raise RuntimeError(
                            f"Clean scalar mismatch for {request_id}: expected={expected:.6g} clean={clean_scalar:.6g} "
                            f"delta={delta:.6g} tolerance={scalar_value_tolerance:.6g}"
                        )
                if request_id not in ablated_values:
                    raise KeyError(f"Missing ablated scalar for request {request_id}.")
                ablated_scalar = ablated_values[request_id]
                total_drop = clean_scalar - ablated_scalar
                for patch_stage in patch_stages:
                    patched_values = patched_values_by_stage[patch_stage]
                    if request_id not in patched_values:
                        raise KeyError(f"Missing patched scalar for request {request_id} stage={patch_stage}.")
                    patched_scalar = patched_values[request_id]
                    rescue = patched_scalar - ablated_scalar
                    unrecovered = clean_scalar - patched_scalar
                    rows.append(
                        {
                            "source_step": int(request_id[0]),
                            "target_step": int(request_id[1]),
                            "pair_id": str(request_id[2]),
                            "margin_side": str(request_id[3]),
                            "scalar_name": str(request_id[4]),
                            "endpoint_kind": str(request_id[5]),
                            "step": int(request["step"]),
                            "checkpoint": str(request["checkpoint"]),
                            "source_component": source_component,
                            "patch_stage": patch_stage,
                            "clean_scalar": clean_scalar,
                            "source_ablated_scalar": ablated_scalar,
                            "patched_scalar": patched_scalar,
                            "total_drop": total_drop,
                            "rescue": rescue,
                            "unrecovered": unrecovered,
                            "absolute_unrecovered": abs(unrecovered),
                            "improved_by_patch": abs(unrecovered) < abs(total_drop),
                        }
                    )
    if not rows:
        raise RuntimeError(f"No residual-state rescue rows produced for source component {source_component}.")
    return rows


def summarize_rescue_rows(*, rows: list[dict[str, Any]], denominator_threshold: float) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot summarize empty residual-state rescue rows.")
    if denominator_threshold < 0.0:
        raise ValueError("denominator_threshold must be non-negative.")
    summaries: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (
                str(row["scalar_name"]),
                str(row["endpoint_kind"]),
                str(row["source_component"]),
                str(row["patch_stage"]),
            )
            for row in rows
        }
    )
    for scalar_name, endpoint_kind, source_component, patch_stage in group_keys:
        group = [
            row
            for row in rows
            if str(row["scalar_name"]) == scalar_name
            and str(row["endpoint_kind"]) == endpoint_kind
            and str(row["source_component"]) == source_component
            and str(row["patch_stage"]) == patch_stage
        ]
        total_drops = [float(row["total_drop"]) for row in group]
        rescues = [float(row["rescue"]) for row in group]
        unrecovered = [float(row["unrecovered"]) for row in group]
        recovery_fraction_rows = [
            float(row["rescue"]) / float(row["total_drop"])
            for row in group
            if abs(float(row["total_drop"])) > denominator_threshold
        ]
        mean_total_drop = _mean(total_drops, label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_stage}/drop")
        mean_rescue = _mean(rescues, label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_stage}/rescue")
        summaries.append(
            {
                "scalar_name": scalar_name,
                "endpoint_kind": endpoint_kind,
                "source_component": source_component,
                "patch_stage": patch_stage,
                "num_observations": len(group),
                "mean_clean_scalar": _mean(
                    [float(row["clean_scalar"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_stage}/clean",
                ),
                "mean_source_ablated_scalar": _mean(
                    [float(row["source_ablated_scalar"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_stage}/ablated",
                ),
                "mean_patched_scalar": _mean(
                    [float(row["patched_scalar"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_stage}/patched",
                ),
                "mean_total_drop": mean_total_drop,
                "mean_abs_total_drop": _mean(
                    [abs(value) for value in total_drops],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_stage}/abs drop",
                ),
                "mean_rescue": mean_rescue,
                "mean_abs_rescue": _mean(
                    [abs(value) for value in rescues],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_stage}/abs rescue",
                ),
                "mean_unrecovered": _mean(
                    unrecovered,
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_stage}/unrecovered",
                ),
                "mean_abs_unrecovered": _mean(
                    [abs(value) for value in unrecovered],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_stage}/abs unrecovered",
                ),
                "mean_rescue_fraction_from_means": None
                if abs(mean_total_drop) <= denominator_threshold
                else float(mean_rescue / mean_total_drop),
                "num_recovery_fraction_rows": len(recovery_fraction_rows),
                "mean_recovery_fraction_per_row": None
                if not recovery_fraction_rows
                else _mean(
                    recovery_fraction_rows,
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_stage}/fraction rows",
                ),
                "improved_fraction": _mean(
                    [1.0 if bool(row["improved_by_patch"]) else 0.0 for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_stage}/improved",
                ),
                "rescue_vs_drop_r_squared": _safe_r_squared(
                    y_values=total_drops,
                    predicted_values=rescues,
                ),
                "rescue_vs_drop_correlation": _safe_correlation(
                    x_values=rescues,
                    y_values=total_drops,
                ),
            }
        )
    return summaries


def _plot_rescue_fraction(
    *,
    summary_rows: list[dict[str, Any]],
    output_path: Path,
    top_k_rows: int,
) -> Path | None:
    rows = [row for row in summary_rows if str(row["endpoint_kind"]) == "target"]
    if not rows:
        return None
    if top_k_rows <= 0:
        raise ValueError("top_k_rows must be positive.")
    rows = sorted(
        rows,
        key=lambda row: (
            str(row["scalar_name"]),
            str(row["source_component"]),
            -abs(0.0 if row["mean_rescue_fraction_from_means"] is None else float(row["mean_rescue_fraction_from_means"])),
        ),
    )[:top_k_rows]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(rows)), 6))
    labels = [f"{row['scalar_name']}:{row['source_component']}:{row['patch_stage']}" for row in rows]
    values = [
        0.0 if row["mean_rescue_fraction_from_means"] is None else float(row["mean_rescue_fraction_from_means"])
        for row in rows
    ]
    colors = ["#376f8f" if value >= 0.0 else "#8f374a" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.axhline(1.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_ylabel("mean rescue / mean source-ablation drop")
    ax.set_title("Residual-state rescue fraction at target endpoints")
    ax.tick_params(axis="x", rotation=50)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_unrecovered(
    *,
    summary_rows: list[dict[str, Any]],
    output_path: Path,
    top_k_rows: int,
) -> Path | None:
    rows = [row for row in summary_rows if str(row["endpoint_kind"]) == "target"]
    if not rows:
        return None
    if top_k_rows <= 0:
        raise ValueError("top_k_rows must be positive.")
    rows = sorted(rows, key=lambda row: float(row["mean_abs_unrecovered"]), reverse=True)[:top_k_rows]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(rows)), 6))
    labels = [f"{row['scalar_name']}:{row['source_component']}:{row['patch_stage']}" for row in rows]
    values = [float(row["mean_abs_unrecovered"]) for row in rows]
    ax.bar(labels, values, color="#8f6b37")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_ylabel("mean abs clean - patched scalar")
    ax.set_title("Residual-state rescue unrecovered effect at target endpoints")
    ax.tick_params(axis="x", rotation=50)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    lines = [
        "# Residual-State Rescue",
        "",
        "## Calculation",
        "",
        "This report tests whether a clean residual state can rescue the model after a source component is ablated.",
        "",
        "```text",
        "damage = scalar(clean) - scalar(source ablated)",
        "rescue = scalar(source ablated + clean residual patch at stage S) - scalar(source ablated)",
        "unrecovered = scalar(clean) - scalar(source ablated + clean residual patch at stage S)",
        "rescue_fraction = rescue / damage",
        "```",
        "",
        "A high rescue fraction means the missing causal information is present at that residual stage. A low rescue fraction means the source component's effect is not restored by that stage patch.",
        "",
        "## Inputs",
        "",
        f"- scalar pair rows: `{report['scalar_pair_rows_path']}`",
        f"- probe set: `{report['probe_set_path']}`",
        f"- pair types: `{report['pair_types']}`",
        f"- margin sides: `{report['margin_sides']}`",
        f"- scalars: `{report['scalar_names']}`",
        f"- endpoint roles: `{report['endpoint_roles']}`",
        f"- source components: `{report['source_components']}`",
        f"- patch stages: `{report['patch_stages']}`",
        "",
        "## Summary",
        "",
        "| scalar | endpoint | source | patch stage | observations | damage | rescue | rescue fraction | abs unrecovered | improved fraction | corr | R squared |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        report["summary_rows"],
        key=lambda item: (
            str(item["scalar_name"]),
            str(item["endpoint_kind"]),
            str(item["source_component"]),
            str(item["patch_stage"]),
        ),
    )[: report["markdown_top_k_rows"]]:
        fraction = row["mean_rescue_fraction_from_means"]
        corr = row["rescue_vs_drop_correlation"]
        r2 = row["rescue_vs_drop_r_squared"]
        lines.append(
            "| {scalar} | {endpoint} | `{source}` | `{stage}` | {n} | {damage:.6g} | {rescue:.6g} | {fraction} | {unrecovered:.6g} | {improved:.3f} | {corr} | {r2} |".format(
                scalar=row["scalar_name"],
                endpoint=row["endpoint_kind"],
                source=row["source_component"],
                stage=row["patch_stage"],
                n=int(row["num_observations"]),
                damage=float(row["mean_total_drop"]),
                rescue=float(row["mean_rescue"]),
                fraction="" if fraction is None else f"{float(fraction):.4f}",
                unrecovered=float(row["mean_abs_unrecovered"]),
                improved=float(row["improved_fraction"]),
                corr="" if corr is None else f"{float(corr):.4f}",
                r2="" if r2 is None else f"{float(r2):.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- rescue rows: `{report['rescue_rows_path']}`",
            f"- summary rows: `{report['summary_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
            "",
            "## Plots",
            "",
        ]
    )
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_residual_state_rescue(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    pair_types: list[str],
    source_components: list[str],
    patch_stages: list[str],
    device_name: str = "mps",
    scalar_names: list[str] | None = None,
    margin_sides: list[str] | None = None,
    endpoint_roles: list[str] | None = None,
    split_filter: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    scalar_value_tolerance: float = 1.0e-4,
    denominator_threshold: float = 1.0e-6,
    markdown_top_k_rows: int = 120,
    plot_top_k_rows: int = 48,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Path]]:
    if not pair_types:
        raise ValueError("residual-state-rescue requires at least one pair type.")
    if not source_components:
        raise ValueError("source_components must not be empty.")
    if not patch_stages:
        raise ValueError("patch_stages must not be empty.")
    if max_pairs_per_type <= 0:
        raise ValueError("max_pairs_per_type must be positive.")
    if min_pairs_per_type <= 0:
        raise ValueError("min_pairs_per_type must be positive.")
    if scalar_value_tolerance < 0.0:
        raise ValueError("scalar_value_tolerance must be non-negative.")
    if denominator_threshold < 0.0:
        raise ValueError("denominator_threshold must be non-negative.")
    if markdown_top_k_rows <= 0:
        raise ValueError("markdown_top_k_rows must be positive.")
    if plot_top_k_rows <= 0:
        raise ValueError("plot_top_k_rows must be positive.")

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
    resolved_endpoint_roles = _resolve_unique_values(
        values=endpoint_roles,
        default_values=["source", "target"],
        allowed_values=RESIDUAL_STATE_RESCUE_ENDPOINT_ROLES,
        label="endpoint role",
    )
    pair_types = sorted(set(pair_types), key=pair_types.index)

    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(device_name)
    model = build_model(spec.model, len(vocab.tokens), device)
    available_components = _component_labels(num_layers=len(model.blocks), num_heads=model.spec.n_heads)
    resolved_sources = _filter_component_labels(
        requested_components=source_components,
        available_components=available_components,
    )
    _validate_maskable_components(
        components=resolved_sources,
        num_layers=len(model.blocks),
        num_heads=model.spec.n_heads,
        device=device,
    )
    resolved_patch_stages = _resolve_patch_stages(patch_stages=patch_stages, num_layers=len(model.blocks))

    scalar_pair_rows = _filter_scalar_pair_rows(
        rows=_load_scalar_pair_rows(scalar_pair_rows_path),
        margin_sides=resolved_margin_sides,
        pair_types=pair_types,
        scalar_names=resolved_scalars,
    )
    checkpoint_paths_by_step = _checkpoint_paths_by_step(scalar_pair_rows)
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
    endpoint_keys = {
        (int(row[f"{endpoint_kind}_step"]), str(row["margin_side"]))
        for row in scalar_pair_rows
        for endpoint_kind in resolved_endpoint_roles
    }
    endpoint_requests = []
    for request in _build_endpoint_requests(scalar_pair_rows=scalar_pair_rows, scalar_names=resolved_scalars):
        if str(request["endpoint_kind"]) not in set(resolved_endpoint_roles):
            continue
        source_row = next(
            row
            for row in scalar_pair_rows
            if int(row["source_step"]) == int(request["request_id"][0])
            and int(row["target_step"]) == int(request["request_id"][1])
            and str(row["pair_id"]) == str(request["request_id"][2])
            and str(row["margin_side"]) == str(request["request_id"][3])
        )
        request["scalar_payload"] = source_row["scalars"][str(request["scalar_name"])]
        endpoint_requests.append(request)
    if not endpoint_requests:
        raise RuntimeError("No endpoint requests survived endpoint-role filters.")

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[residual-state-rescue] "
        f"pairs={len(pairs_by_id)} scalar_rows={len(scalar_pair_rows)} endpoints={len(endpoint_keys)} "
        f"sources={resolved_sources} patch_stages={resolved_patch_stages} scalars={resolved_scalars} "
        f"endpoint_roles={resolved_endpoint_roles} device={device_name}",
        flush=True,
    )
    rescue_rows: list[dict[str, Any]] = []
    for source_component in resolved_sources:
        print(f"[residual-state-rescue] source={source_component} running clean/ablated/patched forwards", flush=True)
        rescue_rows.extend(
            _compute_rescue_rows_for_source(
                model=model,
                checkpoint_paths_by_step=checkpoint_paths_by_step,
                pairs_by_id=pairs_by_id,
                endpoint_keys=endpoint_keys,
                endpoint_requests=endpoint_requests,
                source_component=source_component,
                patch_stages=resolved_patch_stages,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                scalar_value_tolerance=scalar_value_tolerance,
                device=device,
            )
        )
    summary_rows = summarize_rescue_rows(rows=rescue_rows, denominator_threshold=denominator_threshold)

    rescue_rows_path = output_dir / "residual_state_rescue_rows.jsonl"
    summary_rows_path = output_dir / "residual_state_rescue_summary_rows.jsonl"
    pair_rows_path = output_dir / "residual_state_rescue_pairs.jsonl"
    report_path = output_dir / "residual_state_rescue_report.json"
    markdown_path = output_dir / "residual_state_rescue_report.md"
    write_jsonl(rescue_rows_path, rescue_rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(
        pair_rows_path,
        [{key: value for key, value in pair.items() if key not in {"clean_record", "corrupted_record"}} for pair in pairs_by_id.values()],
    )

    plot_paths: dict[str, Path] = {}
    fraction_plot = _plot_rescue_fraction(
        summary_rows=summary_rows,
        output_path=output_dir / "residual_state_rescue_fraction.svg",
        top_k_rows=plot_top_k_rows,
    )
    if fraction_plot is not None:
        plot_paths["rescue_fraction"] = fraction_plot
    unrecovered_plot = _plot_unrecovered(
        summary_rows=summary_rows,
        output_path=output_dir / "residual_state_rescue_unrecovered.svg",
        top_k_rows=plot_top_k_rows,
    )
    if unrecovered_plot is not None:
        plot_paths["unrecovered"] = unrecovered_plot

    report = {
        "schema_version": RESIDUAL_STATE_RESCUE_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "device": device_name,
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names": resolved_scalars,
        "endpoint_roles": resolved_endpoint_roles,
        "source_components": resolved_sources,
        "patch_stages": resolved_patch_stages,
        "scalar_value_tolerance": scalar_value_tolerance,
        "denominator_threshold": denominator_threshold,
        "markdown_top_k_rows": markdown_top_k_rows,
        "plot_top_k_rows": plot_top_k_rows,
        "checkpoint_paths_by_step": {str(step): str(path) for step, path in checkpoint_paths_by_step.items()},
        "pair_construction": pair_construction,
        "calculation": {
            "damage": "scalar(clean) - scalar(source ablated)",
            "rescue": "scalar(source ablated + clean residual patch) - scalar(source ablated)",
            "unrecovered": "scalar(clean) - scalar(source ablated + clean residual patch)",
            "rescue_fraction": "rescue / damage where |damage| > denominator_threshold",
            "patch_scope": "full residual stream tensor at the requested stage",
        },
        "rescue_rows_path": str(rescue_rows_path),
        "summary_rows_path": str(summary_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    print(f"[residual-state-rescue] complete report={report_path} rows={rescue_rows_path}", flush=True)
    return report_path, markdown_path, rescue_rows_path, summary_rows_path, pair_rows_path, plot_paths
