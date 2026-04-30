from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import re
import shutil
from typing import Any

import torch

from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import _checkpoint_step_from_path, _validate_single_query_batch
from circuit.analysis.output_component_causal_validation import _component_mask_kwargs
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
from circuit.analysis.residual_state_rescue import (
    RESIDUAL_STATE_RESCUE_ENDPOINT_ROLES,
    _compute_scalar_payloads,
    _safe_correlation,
    _validate_maskable_components,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


COMPONENT_OUTPUT_RESCUE_SCHEMA_VERSION = 1


_HEAD_COMPONENT_RE = re.compile(r"^L(?P<layer>\d+)H(?P<head>\d+)$")
_MLP_COMPONENT_RE = re.compile(r"^L(?P<layer>\d+)MLP$")


def _parse_local_component(component: str) -> tuple[str, int, int | None]:
    head_match = _HEAD_COMPONENT_RE.match(component)
    if head_match is not None:
        return "head", int(head_match.group("layer")), int(head_match.group("head"))
    mlp_match = _MLP_COMPONENT_RE.match(component)
    if mlp_match is not None:
        return "mlp", int(mlp_match.group("layer")), None
    raise ValueError(f"Unsupported component-output-rescue component {component!r}; expected attention head or MLP label.")


def _component_order(component: str) -> int:
    kind, layer, _ = _parse_local_component(component)
    offset = 0 if kind == "head" else 1
    return layer * 2 + offset


def _component_patch_stage(component: str) -> str:
    kind, layer, _ = _parse_local_component(component)
    return f"layer_{layer}_post_attn" if kind == "head" else f"layer_{layer}_post_mlp"


def _pre_attention_stage(layer: int) -> str:
    if layer < 0:
        raise ValueError(f"Layer must be non-negative, got {layer}.")
    return "embedding" if layer == 0 else f"layer_{layer - 1}_post_mlp"


def _attention_head_write(
    *,
    model: torch.nn.Module,
    layer: int,
    head: int,
    pre_layer_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if layer < 0 or layer >= len(model.blocks):
        raise ValueError(f"Head layer out of range: layer={layer} num_layers={len(model.blocks)}.")
    num_heads = int(model.spec.n_heads)
    if head < 0 or head >= num_heads:
        raise ValueError(f"Head index out of range: head={head} num_heads={num_heads}.")
    block = model.blocks[layer]
    one_hot = torch.zeros(num_heads, device=pre_layer_state.device, dtype=pre_layer_state.dtype)
    one_hot[head] = 1.0
    zero = torch.zeros(num_heads, device=pre_layer_state.device, dtype=pre_layer_state.dtype)
    normalized = block.ln_1(pre_layer_state)
    head_only, _ = block.attn(
        normalized,
        attention_mask=attention_mask,
        head_mask=one_hot,
        return_attention=False,
    )
    no_heads, _ = block.attn(
        normalized,
        attention_mask=attention_mask,
        head_mask=zero,
        return_attention=False,
    )
    return head_only - no_heads


def _component_write(
    *,
    model: torch.nn.Module,
    component: str,
    residual_streams: dict[str, torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    kind, layer, head = _parse_local_component(component)
    if kind == "mlp":
        post_attn = residual_streams[f"layer_{layer}_post_attn"]
        post_mlp = residual_streams[f"layer_{layer}_post_mlp"]
        return post_mlp - post_attn
    if head is None:
        raise RuntimeError(f"Parsed head component without head index: {component}")
    pre_stage = _pre_attention_stage(layer)
    return _attention_head_write(
        model=model,
        layer=layer,
        head=head,
        pre_layer_state=residual_streams[pre_stage],
        attention_mask=attention_mask,
    )


def _validate_downstream_components(*, source_components: list[str], patch_components: list[str]) -> None:
    for source_component in source_components:
        source_order = _component_order(source_component)
        for patch_component in patch_components:
            patch_order = _component_order(patch_component)
            if patch_order <= source_order:
                raise ValueError(
                    "component-output-rescue patches only downstream component writes. "
                    f"source={source_component} order={source_order} patch={patch_component} order={patch_order}"
                )


def _parse_patch_group(group_spec: str) -> list[str]:
    components = [component.strip() for component in group_spec.split(",") if component.strip()]
    if not components:
        raise ValueError(f"Patch group is empty: {group_spec!r}")
    unique_components: list[str] = []
    for component in components:
        if component not in unique_components:
            unique_components.append(component)
    return unique_components


def _patch_group_id(components: list[str]) -> str:
    if not components:
        raise ValueError("Cannot build patch group id for an empty component list.")
    return "+".join(components)


def _build_patch_groups(
    *,
    patch_components: list[str] | None,
    patch_groups: list[str] | None,
    available_components: list[str],
) -> list[dict[str, Any]]:
    groups: list[list[str]] = []
    if patch_components is not None:
        for component in patch_components:
            groups.append([component])
    if patch_groups is not None:
        for group_spec in patch_groups:
            groups.append(_parse_patch_group(group_spec))
    if not groups:
        raise ValueError("At least one --patch-component or --patch-group is required.")

    resolved_groups: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    for group in groups:
        resolved = _filter_component_labels(
            requested_components=group,
            available_components=available_components,
        )
        group_id = _patch_group_id(resolved)
        if group_id in seen_group_ids:
            continue
        seen_group_ids.add(group_id)
        resolved_groups.append(
            {
                "patch_group_id": group_id,
                "patch_components": resolved,
                "patch_stages": sorted({_component_patch_stage(component) for component in resolved}),
            }
        )
    if not resolved_groups:
        raise RuntimeError("Patch group resolution produced no groups.")
    return resolved_groups


def _component_groups_by_order(components: list[str]) -> list[tuple[int, list[str]]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for component in components:
        grouped[_component_order(component)].append(component)
    return [(order, grouped[order]) for order in sorted(grouped)]


def _compute_component_output_rescue_rows_for_source(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    endpoint_keys: set[tuple[int, str]],
    endpoint_requests: list[dict[str, Any]],
    source_component: str,
    patch_groups: list[dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
    scalar_value_tolerance: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    num_layers = len(model.blocks)
    num_heads = int(model.spec.n_heads)
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
                source_ablated_outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_residual_streams=True,
                    **source_mask_kwargs,
                )
            if clean_outputs.residual_streams is None or source_ablated_outputs.residual_streams is None:
                raise RuntimeError("component-output-rescue requires residual streams.")

            clean_logits, clean_targets, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            ablated_logits, ablated_targets, ablated_metadata = extract_answer_logits(source_ablated_outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=clean_metadata, label="component-output rescue clean")
            _validate_single_query_batch(batch=batch, metadata=ablated_metadata, label="component-output rescue ablated")
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
                label="component-output rescue clean",
            )
            ablated_values = _compute_scalar_payloads(
                answer_logits=ablated_logits,
                answer_targets=ablated_targets,
                metadata=ablated_metadata,
                batch_pair_ids=batch_pair_ids,
                request_specs_by_pair_id=request_specs_by_pair_id,
                label="component-output rescue ablated",
            )

            patched_values_by_group: dict[str, dict[tuple[int, int, str, str, str, str], float]] = {}
            for patch_group in patch_groups:
                patch_group_id = str(patch_group["patch_group_id"])
                patch_components = [str(component) for component in patch_group["patch_components"]]
                residual_patch: dict[str, torch.Tensor] = {}
                for _, ordered_components in _component_groups_by_order(patch_components):
                    with torch.no_grad():
                        current_outputs = model(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            return_residual_streams=True,
                            residual_patch=residual_patch,
                            **source_mask_kwargs,
                        )
                    if current_outputs.residual_streams is None:
                        raise RuntimeError("component-output-rescue requires residual streams for current patched run.")
                    stage_deltas: dict[str, torch.Tensor] = {}
                    for patch_component in ordered_components:
                        patch_stage = _component_patch_stage(patch_component)
                        clean_write = _component_write(
                            model=model,
                            component=patch_component,
                            residual_streams=clean_outputs.residual_streams,
                            attention_mask=batch["attention_mask"],
                        ).detach()
                        current_write = _component_write(
                            model=model,
                            component=patch_component,
                            residual_streams=current_outputs.residual_streams,
                            attention_mask=batch["attention_mask"],
                        ).detach()
                        delta = clean_write - current_write
                        if patch_stage in stage_deltas:
                            stage_deltas[patch_stage] = stage_deltas[patch_stage] + delta
                        else:
                            stage_deltas[patch_stage] = delta
                    for patch_stage, delta in stage_deltas.items():
                        residual_patch[patch_stage] = current_outputs.residual_streams[patch_stage].detach() + delta
                with torch.no_grad():
                    patched_outputs = model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        residual_patch=residual_patch,
                        **source_mask_kwargs,
                    )
                patched_logits, patched_targets, patched_metadata = extract_answer_logits(patched_outputs.logits, batch)
                _validate_single_query_batch(
                    batch=batch,
                    metadata=patched_metadata,
                    label=f"component-output rescue patched {patch_group_id}",
                )
                patched_values_by_group[patch_group_id] = _compute_scalar_payloads(
                    answer_logits=patched_logits,
                    answer_targets=patched_targets,
                    metadata=patched_metadata,
                    batch_pair_ids=batch_pair_ids,
                    request_specs_by_pair_id=request_specs_by_pair_id,
                    label=f"component-output rescue patched {patch_group_id}",
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
                    raise KeyError(f"Missing source-ablated scalar for request {request_id}.")
                ablated_scalar = ablated_values[request_id]
                total_drop = clean_scalar - ablated_scalar
                for patch_group in patch_groups:
                    patch_group_id = str(patch_group["patch_group_id"])
                    patched_values = patched_values_by_group[patch_group_id]
                    if request_id not in patched_values:
                        raise KeyError(f"Missing patched scalar for request {request_id} group={patch_group_id}.")
                    patched_scalar = patched_values[request_id]
                    rescue = patched_scalar - ablated_scalar
                    unrecovered = clean_scalar - patched_scalar
                    rows.append(
                        {
                            "source_step": int(request_id[0]),
                            "target_step": int(request_id[1]),
                            "endpoint_kind": str(request["endpoint_kind"]),
                            "pair_id": str(request["pair_id"]),
                            "pair_type": str(request["pair_type"]),
                            "margin_side": str(request["margin_side"]),
                            "scalar_name": str(request["scalar_name"]),
                            "source_component": source_component,
                            "patch_component": patch_group_id,
                            "patch_group_id": patch_group_id,
                            "patch_components": list(patch_group["patch_components"]),
                            "patch_stage": "+".join(str(stage) for stage in patch_group["patch_stages"]),
                            "patch_stages": list(patch_group["patch_stages"]),
                            "clean_scalar": clean_scalar,
                            "source_ablated_scalar": ablated_scalar,
                            "patched_scalar": patched_scalar,
                            "total_drop": total_drop,
                            "rescue": rescue,
                            "unrecovered": unrecovered,
                            "rescue_fraction": None if abs(total_drop) <= 1.0e-12 else rescue / total_drop,
                            "improved_by_patch": abs(unrecovered) < abs(total_drop),
                        }
                    )
    return rows


def summarize_component_output_rescue_rows(
    *,
    rows: list[dict[str, Any]],
    denominator_threshold: float,
) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError("No component-output rescue rows to summarize.")
    if denominator_threshold < 0.0:
        raise ValueError("denominator_threshold must be non-negative.")
    summaries: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (
                str(row["scalar_name"]),
                str(row["endpoint_kind"]),
                str(row["source_component"]),
                str(row["patch_component"]),
                str(row["patch_stage"]),
            )
            for row in rows
        }
    )
    for scalar_name, endpoint_kind, source_component, patch_component, patch_stage in group_keys:
        group = [
            row
            for row in rows
            if str(row["scalar_name"]) == scalar_name
            and str(row["endpoint_kind"]) == endpoint_kind
            and str(row["source_component"]) == source_component
            and str(row["patch_component"]) == patch_component
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
        mean_total_drop = _mean(
            total_drops,
            label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_component}/drop",
        )
        mean_rescue = _mean(
            rescues,
            label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_component}/rescue",
        )
        summaries.append(
            {
                "scalar_name": scalar_name,
                "endpoint_kind": endpoint_kind,
                "source_component": source_component,
                "patch_component": patch_component,
                "patch_stage": patch_stage,
                "num_observations": len(group),
                "mean_clean_scalar": _mean(
                    [float(row["clean_scalar"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_component}/clean",
                ),
                "mean_source_ablated_scalar": _mean(
                    [float(row["source_ablated_scalar"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_component}/ablated",
                ),
                "mean_patched_scalar": _mean(
                    [float(row["patched_scalar"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_component}/patched",
                ),
                "mean_total_drop": mean_total_drop,
                "mean_abs_total_drop": _mean(
                    [abs(value) for value in total_drops],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_component}/abs drop",
                ),
                "mean_rescue": mean_rescue,
                "mean_abs_rescue": _mean(
                    [abs(value) for value in rescues],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_component}/abs rescue",
                ),
                "mean_unrecovered": _mean(
                    unrecovered,
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_component}/unrecovered",
                ),
                "mean_abs_unrecovered": _mean(
                    [abs(value) for value in unrecovered],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_component}/abs unrecovered",
                ),
                "mean_rescue_fraction_from_means": None
                if abs(mean_total_drop) <= denominator_threshold
                else float(mean_rescue / mean_total_drop),
                "num_recovery_fraction_rows": len(recovery_fraction_rows),
                "mean_recovery_fraction_per_row": None
                if not recovery_fraction_rows
                else _mean(
                    recovery_fraction_rows,
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_component}/fraction rows",
                ),
                "improved_fraction": _mean(
                    [1.0 if bool(row["improved_by_patch"]) else 0.0 for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{patch_component}/improved",
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


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Component Output Rescue",
        "",
        "This report tests whether clean downstream component writes rescue a source-component ablation.",
        "",
        "Definitions:",
        "",
        "- damage = scalar(clean) - scalar(source ablated)",
        "- rescue = scalar(source ablated + clean downstream component write) - scalar(source ablated)",
        "- unrecovered = scalar(clean) - scalar(source ablated + clean downstream component write)",
        "",
        "A patch may contain one component or an ordered component group. For attention heads, the patched write is the single-head residual contribution. It is computed as only-that-head attention output minus all-heads-off attention output, so the shared output bias cancels.",
        "",
        "| scalar | endpoint | source | patch component | patch stage | observations | damage | rescue | rescue fraction | abs unrecovered | improved fraction | corr | R squared |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = sorted(
        report["summary_rows"],
        key=lambda row: (
            -abs(float(row["mean_rescue"])),
            str(row["scalar_name"]),
            str(row["endpoint_kind"]),
            str(row["patch_component"]),
        ),
    )[: int(report["markdown_top_k_rows"])]
    for row in rows:
        rescue_fraction = row["mean_rescue_fraction_from_means"]
        corr = row["rescue_vs_drop_correlation"]
        r_squared = row["rescue_vs_drop_r_squared"]
        lines.append(
            "| {scalar} | {endpoint} | `{source}` | `{patch}` | `{stage}` | {n} | {damage:.6g} | {rescue:.6g} | {frac} | {unrec:.6g} | {improved:.3f} | {corr} | {r2} |".format(
                scalar=row["scalar_name"],
                endpoint=row["endpoint_kind"],
                source=row["source_component"],
                patch=row["patch_component"],
                stage=row["patch_stage"],
                n=int(row["num_observations"]),
                damage=float(row["mean_total_drop"]),
                rescue=float(row["mean_rescue"]),
                frac="" if rescue_fraction is None else f"{float(rescue_fraction):.3f}",
                unrec=float(row["mean_abs_unrecovered"]),
                improved=float(row["improved_fraction"]),
                corr="" if corr is None else f"{float(corr):.3f}",
                r2="" if r_squared is None or not math.isfinite(float(r_squared)) else f"{float(r_squared):.3f}",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_component_output_rescue(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    pair_types: list[str],
    source_components: list[str],
    patch_components: list[str] | None,
    patch_groups: list[str] | None = None,
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
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    if not pair_types:
        raise ValueError("component-output-rescue requires at least one pair type.")
    if not source_components:
        raise ValueError("source_components must not be empty.")
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
    overlap = sorted(set(resolved_sources) & set(resolved_patch_components))
    if overlap:
        raise ValueError(f"Source and patch component sets must be disjoint; overlap={overlap}")
    _validate_maskable_components(
        components=resolved_sources,
        num_layers=len(model.blocks),
        num_heads=model.spec.n_heads,
        device=device,
    )
    _validate_maskable_components(
        components=resolved_patch_components,
        num_layers=len(model.blocks),
        num_heads=model.spec.n_heads,
        device=device,
    )
    _validate_downstream_components(source_components=resolved_sources, patch_components=resolved_patch_components)

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
    endpoint_requests: list[dict[str, Any]] = []
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
        request["pair_type"] = str(source_row["pair_type"])
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
        "[component-output-rescue] "
        f"pairs={len(pairs_by_id)} scalar_rows={len(scalar_pair_rows)} endpoints={len(endpoint_keys)} "
        f"sources={resolved_sources} patch_groups={resolved_patch_groups} scalars={resolved_scalars} "
        f"endpoint_roles={resolved_endpoint_roles} device={device_name}",
        flush=True,
    )
    rescue_rows: list[dict[str, Any]] = []
    for source_component in resolved_sources:
        print(f"[component-output-rescue] source={source_component} running clean/ablated/component-patched forwards", flush=True)
        rescue_rows.extend(
            _compute_component_output_rescue_rows_for_source(
                model=model,
                checkpoint_paths_by_step=checkpoint_paths_by_step,
                pairs_by_id=pairs_by_id,
                endpoint_keys=endpoint_keys,
                endpoint_requests=endpoint_requests,
                source_component=source_component,
                patch_groups=resolved_patch_groups,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                scalar_value_tolerance=scalar_value_tolerance,
                device=device,
            )
        )
    summary_rows = summarize_component_output_rescue_rows(
        rows=rescue_rows,
        denominator_threshold=denominator_threshold,
    )

    rescue_rows_path = output_dir / "component_output_rescue_rows.jsonl"
    summary_rows_path = output_dir / "component_output_rescue_summary_rows.jsonl"
    pair_rows_path = output_dir / "component_output_rescue_pairs.jsonl"
    report_path = output_dir / "component_output_rescue_report.json"
    markdown_path = output_dir / "component_output_rescue_report.md"
    write_jsonl(rescue_rows_path, rescue_rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(
        pair_rows_path,
        [{key: value for key, value in pair.items() if key not in {"clean_record", "corrupted_record"}} for pair in pairs_by_id.values()],
    )

    report = {
        "schema_version": COMPONENT_OUTPUT_RESCUE_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "device": device_name,
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names": resolved_scalars,
        "endpoint_roles": resolved_endpoint_roles,
        "source_components": resolved_sources,
        "patch_components": resolved_patch_components,
        "patch_groups": resolved_patch_groups,
        "scalar_value_tolerance": scalar_value_tolerance,
        "denominator_threshold": denominator_threshold,
        "markdown_top_k_rows": markdown_top_k_rows,
        "checkpoint_paths_by_step": {str(step): str(path) for step, path in checkpoint_paths_by_step.items()},
        "pair_construction": pair_construction,
        "calculation": {
            "damage": "scalar(clean) - scalar(source ablated)",
            "rescue": "scalar(source ablated + clean downstream component write) - scalar(source ablated)",
            "unrecovered": "scalar(clean) - scalar(source ablated + clean downstream component write)",
            "patch_scope": "single downstream component write or ordered downstream component-write group; for attention heads, output-bias-cancelled single-head write",
        },
        "rescue_rows_path": str(rescue_rows_path),
        "summary_rows_path": str(summary_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(f"[component-output-rescue] complete report={report_path} rows={rescue_rows_path}", flush=True)
    return report_path, markdown_path, rescue_rows_path, summary_rows_path, pair_rows_path
