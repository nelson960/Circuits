from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import shutil
from typing import Any

import torch

from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    _checkpoint_step_from_path,
    _head_contributions_for_layer,
    _validate_single_query_batch,
)
from circuit.analysis.output_component_causal_validation import _component_mask_kwargs
from circuit.analysis.output_route_closure import (
    OUTPUT_ROUTE_MARGIN_SIDES,
    OUTPUT_ROUTE_SCALARS,
    _build_endpoint_requests,
    _checkpoint_paths_by_step,
    _component_labels,
    _compute_endpoint_component_values,
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


OUTPUT_MEDIATED_CAUSAL_DECOMPOSITION_SCHEMA_VERSION = 1
OUTPUT_MEDIATED_ENDPOINT_ROLES = ["source", "target"]


def _validate_maskable_components(
    *,
    components: list[str],
    num_layers: int,
    num_heads: int,
    device: torch.device,
) -> None:
    for component in components:
        _component_mask_kwargs(component=component, num_layers=num_layers, num_heads=num_heads, device=device)


def _compute_endpoint_payloads_with_source_ablation(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    endpoint_keys: set[tuple[int, str]],
    component_labels: list[str],
    source_component: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[tuple[int, str, str], dict[str, Any]]:
    if not endpoint_keys:
        raise ValueError("endpoint_keys must not be empty.")
    if not component_labels:
        raise ValueError("component_labels must not be empty.")
    num_layers = len(model.blocks)
    num_heads = model.spec.n_heads
    source_mask_kwargs = _component_mask_kwargs(
        component=source_component,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
    )
    available_components = set(component_labels)
    payloads: dict[tuple[int, str, str], dict[str, Any]] = {}
    pair_ids = sorted(pairs_by_id)
    for step, margin_side in sorted(endpoint_keys):
        if margin_side not in OUTPUT_ROUTE_MARGIN_SIDES:
            raise ValueError(f"Unsupported margin side {margin_side!r}; expected one of {OUTPUT_ROUTE_MARGIN_SIDES}.")
        if step not in checkpoint_paths_by_step:
            raise KeyError(f"No checkpoint path for step {step}.")
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
        with torch.no_grad():
            for start_index in range(0, len(pair_ids), batch_size):
                batch_pair_ids = pair_ids[start_index : start_index + batch_size]
                pair_batch = [pairs_by_id[pair_id] for pair_id in batch_pair_ids]
                records = [pair[side_key] for pair in pair_batch]
                batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_residual_streams=True,
                    **source_mask_kwargs,
                )
                if outputs.residual_streams is None:
                    raise RuntimeError("Mediated causal decomposition requires residual streams.")
                answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
                _validate_single_query_batch(batch=batch, metadata=metadata, label="output mediated causal decomposition")
                rows = metadata["rows"]
                prediction_positions = metadata["prediction_positions"]
                final_pre_stage = f"layer_{num_layers - 1}_post_mlp"
                final_pre_vectors = outputs.residual_streams[final_pre_stage][rows, prediction_positions, :]
                component_vectors: dict[str, torch.Tensor] = {}
                if "embedding" in available_components:
                    component_vectors["embedding"] = outputs.residual_streams["embedding"][rows, prediction_positions, :]
                pre_block_states = [outputs.residual_streams["embedding"]]
                for layer_index in range(1, num_layers):
                    pre_block_states.append(outputs.residual_streams[f"layer_{layer_index - 1}_post_mlp"])
                for layer_index, block in enumerate(model.blocks):
                    requested_heads = [
                        head_index
                        for head_index in range(block.attn.n_heads)
                        if f"L{layer_index}H{head_index}" in available_components
                    ]
                    if requested_heads:
                        head_contributions = _head_contributions_for_layer(
                            block=block,
                            pre_state=pre_block_states[layer_index],
                            attention_mask=batch["attention_mask"],
                        )
                        for head_index in requested_heads:
                            component_vectors[f"L{layer_index}H{head_index}"] = head_contributions[head_index][
                                rows,
                                prediction_positions,
                                :,
                            ]
                    mlp_label = f"L{layer_index}MLP"
                    if mlp_label in available_components:
                        component_vectors[mlp_label] = (
                            outputs.residual_streams[f"layer_{layer_index}_post_mlp"][rows, prediction_positions, :]
                            - outputs.residual_streams[f"layer_{layer_index}_post_attn"][rows, prediction_positions, :]
                        )
                missing = [component for component in component_labels if component not in component_vectors]
                if missing:
                    raise RuntimeError(
                        f"Failed to compute requested component vectors under {source_component} ablation: {missing}"
                    )
                predictions = answer_logits.argmax(dim=-1)
                for item_index, pair_id in enumerate(batch_pair_ids):
                    payloads[(step, margin_side, pair_id)] = {
                        "step": step,
                        "checkpoint": str(checkpoint_path),
                        "margin_side": margin_side,
                        "pair_id": pair_id,
                        "source_component_ablated": source_component,
                        "answer_target_id": int(answer_targets[item_index].detach().cpu().item()),
                        "answer_prediction_id": int(predictions[item_index].detach().cpu().item()),
                        "final_pre_vector": final_pre_vectors[item_index].detach(),
                        "component_vectors": {
                            component: component_vectors[component][item_index].detach()
                            for component in component_labels
                        },
                    }
    expected_count = len(endpoint_keys) * len(pairs_by_id)
    if len(payloads) != expected_count:
        raise RuntimeError(f"Ablated endpoint payload count mismatch: expected={expected_count} got={len(payloads)}")
    return payloads


def _validate_baseline_scalar_values(
    *,
    scalar_pair_rows: list[dict[str, Any]],
    baseline_values: dict[tuple[int, int, str, str, str, str], dict[str, Any]],
    scalar_names: list[str],
    endpoint_roles: list[str],
    tolerance: float,
) -> None:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")
    for row in scalar_pair_rows:
        for scalar_name in scalar_names:
            scalar_payload = row["scalars"][scalar_name]
            for endpoint_kind in endpoint_roles:
                request_id = (
                    int(row["source_step"]),
                    int(row["target_step"]),
                    str(row["pair_id"]),
                    str(row["margin_side"]),
                    scalar_name,
                    endpoint_kind,
                )
                if request_id not in baseline_values:
                    raise KeyError(f"Missing baseline endpoint values for {request_id}.")
                recomputed = float(baseline_values[request_id]["scalar_value_recomputed"])
                expected = float(scalar_payload[endpoint_kind])
                delta = abs(recomputed - expected)
                if delta > tolerance:
                    raise RuntimeError(
                        f"Baseline scalar recomputation mismatch for {request_id}: "
                        f"expected={expected:.6g} recomputed={recomputed:.6g} "
                        f"delta={delta:.6g} tolerance={tolerance:.6g}"
                    )


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


def _summarize_source_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot summarize empty source rows.")
    summaries: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (
                str(row["scalar_name"]),
                str(row["endpoint_kind"]),
                str(row["source_component"]),
            )
            for row in rows
        }
    )
    for scalar_name, endpoint_kind, source_component in group_keys:
        group = [
            row
            for row in rows
            if str(row["scalar_name"]) == scalar_name
            and str(row["endpoint_kind"]) == endpoint_kind
            and str(row["source_component"]) == source_component
        ]
        total = [float(row["total_causal_effect"]) for row in group]
        direct = [float(row["direct_source_dla"]) for row in group]
        mediated = [float(row["mediated_downstream_sum"]) for row in group]
        explained = [float(row["direct_plus_mediated"]) for row in group]
        residuals = [float(row["mediation_residual"]) for row in group]
        mean_total = _mean(total, label=f"{scalar_name}/{endpoint_kind}/{source_component}/total")
        mean_explained = _mean(explained, label=f"{scalar_name}/{endpoint_kind}/{source_component}/explained")
        summaries.append(
            {
                "scalar_name": scalar_name,
                "endpoint_kind": endpoint_kind,
                "source_component": source_component,
                "num_observations": len(group),
                "mean_total_causal_effect": mean_total,
                "mean_abs_total_causal_effect": _mean(
                    [abs(value) for value in total],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/abs total",
                ),
                "mean_direct_source_dla": _mean(
                    direct,
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/direct",
                ),
                "mean_mediated_downstream_sum": _mean(
                    mediated,
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/mediated",
                ),
                "mean_direct_plus_mediated": mean_explained,
                "mean_mediation_residual": _mean(
                    residuals,
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/residual",
                ),
                "mean_abs_mediation_residual": _mean(
                    [abs(value) for value in residuals],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/abs residual",
                ),
                "mean_explained_fraction": None
                if abs(mean_total) <= 1.0e-12
                else float(mean_explained / mean_total),
                "sign_match_fraction": _mean(
                    [
                        1.0
                        if (float(row["total_causal_effect"]) == 0.0 and float(row["direct_plus_mediated"]) == 0.0)
                        or (float(row["total_causal_effect"]) > 0.0 and float(row["direct_plus_mediated"]) > 0.0)
                        or (float(row["total_causal_effect"]) < 0.0 and float(row["direct_plus_mediated"]) < 0.0)
                        else 0.0
                        for row in group
                    ],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/sign",
                ),
                "total_vs_direct_plus_mediated_r_squared": _safe_r_squared(
                    y_values=total,
                    predicted_values=explained,
                ),
                "total_vs_direct_plus_mediated_correlation": _safe_correlation(
                    x_values=explained,
                    y_values=total,
                ),
            }
        )
    return summaries


def _summarize_downstream_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot summarize empty downstream rows.")
    summaries: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (
                str(row["scalar_name"]),
                str(row["endpoint_kind"]),
                str(row["source_component"]),
                str(row["downstream_component"]),
            )
            for row in rows
        }
    )
    for scalar_name, endpoint_kind, source_component, downstream_component in group_keys:
        group = [
            row
            for row in rows
            if str(row["scalar_name"]) == scalar_name
            and str(row["endpoint_kind"]) == endpoint_kind
            and str(row["source_component"]) == source_component
            and str(row["downstream_component"]) == downstream_component
        ]
        effects = [float(row["mediated_effect"]) for row in group]
        summaries.append(
            {
                "scalar_name": scalar_name,
                "endpoint_kind": endpoint_kind,
                "source_component": source_component,
                "downstream_component": downstream_component,
                "num_observations": len(group),
                "mean_baseline_downstream_dla": _mean(
                    [float(row["baseline_downstream_dla"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{downstream_component}/base",
                ),
                "mean_ablated_downstream_dla": _mean(
                    [float(row["ablated_downstream_dla"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{downstream_component}/ablated",
                ),
                "mean_mediated_effect": _mean(
                    effects,
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{downstream_component}/mediated",
                ),
                "mean_abs_mediated_effect": _mean(
                    [abs(value) for value in effects],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{downstream_component}/abs mediated",
                ),
            }
        )
    return summaries


def _plot_source_mediation(
    *,
    source_summary_rows: list[dict[str, Any]],
    output_path: Path,
    top_k_rows: int,
) -> Path | None:
    rows = [
        row
        for row in source_summary_rows
        if str(row["endpoint_kind"]) == "target"
    ]
    if not rows:
        return None
    if top_k_rows <= 0:
        raise ValueError("top_k_rows must be positive.")
    rows = sorted(rows, key=lambda row: abs(float(row["mean_total_causal_effect"])), reverse=True)[:top_k_rows]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(rows)), 6))
    labels = [f"{row['scalar_name']}:{row['source_component']}" for row in rows]
    total = [float(row["mean_total_causal_effect"]) for row in rows]
    direct = [float(row["mean_direct_source_dla"]) for row in rows]
    mediated = [float(row["mean_mediated_downstream_sum"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.25
    ax.bar([value - width for value in x], total, width=width, label="total causal", color="#376f8f")
    ax.bar(x, direct, width=width, label="direct source DLA", color="#8f6b37")
    ax.bar([value + width for value in x], mediated, width=width, label="downstream mediated", color="#4a8f37")
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("mean scalar effect")
    ax.set_title("Output mediation: total vs direct vs downstream")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_downstream_mediation(
    *,
    downstream_summary_rows: list[dict[str, Any]],
    output_path: Path,
    top_k_rows: int,
) -> Path | None:
    rows = [
        row
        for row in downstream_summary_rows
        if str(row["endpoint_kind"]) == "target"
    ]
    if not rows:
        return None
    if top_k_rows <= 0:
        raise ValueError("top_k_rows must be positive.")
    rows = sorted(rows, key=lambda row: abs(float(row["mean_mediated_effect"])), reverse=True)[:top_k_rows]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(rows)), 6))
    labels = [f"{row['scalar_name']}:{row['source_component']}->{row['downstream_component']}" for row in rows]
    values = [float(row["mean_mediated_effect"]) for row in rows]
    colors = ["#4a8f37" if value >= 0.0 else "#8f374a" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#777777", linewidth=1.0)
    ax.set_ylabel("mean downstream DLA loss after source ablation")
    ax.set_title("Top mediated downstream effects")
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
        "# Output Mediated Causal Decomposition",
        "",
        "## Calculation",
        "",
        "This report tests whether an upstream component's causal effect is mediated through later output-writing components.",
        "",
        "```text",
        "total_effect(A) = scalar(theta) - scalar(theta with A ablated)",
        "direct_effect(A) = DLA_A(theta)",
        "mediated_effect(A -> B) = DLA_B(theta) - DLA_B(theta with A ablated)",
        "residual = total_effect(A) - direct_effect(A) - sum_B mediated_effect(A -> B)",
        "```",
        "",
        "A small residual means the selected downstream components explain the upstream component's causal effect. A large residual means the effect flows through unmeasured components, nonlinear interactions, or a scalar branch not represented by this downstream set.",
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
        f"- downstream components: `{report['downstream_components']}`",
        "",
        "## Source Mediation Summary",
        "",
        "| scalar | endpoint | source | observations | total causal | direct DLA | downstream mediated | residual abs | explained fraction | corr | R squared |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        report["source_summary_rows"],
        key=lambda item: (
            str(item["scalar_name"]),
            str(item["endpoint_kind"]),
            -abs(float(item["mean_total_causal_effect"])),
        ),
    )[: report["markdown_top_k_rows"]]:
        fraction = row["mean_explained_fraction"]
        corr = row["total_vs_direct_plus_mediated_correlation"]
        r2 = row["total_vs_direct_plus_mediated_r_squared"]
        lines.append(
            "| {scalar} | {endpoint} | `{source}` | {n} | {total:.6g} | {direct:.6g} | {mediated:.6g} | {residual:.6g} | {fraction} | {corr} | {r2} |".format(
                scalar=row["scalar_name"],
                endpoint=row["endpoint_kind"],
                source=row["source_component"],
                n=int(row["num_observations"]),
                total=float(row["mean_total_causal_effect"]),
                direct=float(row["mean_direct_source_dla"]),
                mediated=float(row["mean_mediated_downstream_sum"]),
                residual=float(row["mean_abs_mediation_residual"]),
                fraction="" if fraction is None else f"{float(fraction):.4f}",
                corr="" if corr is None else f"{float(corr):.4f}",
                r2="" if r2 is None else f"{float(r2):.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## Top Downstream Mediated Effects",
            "",
            "| scalar | endpoint | source | downstream | observations | baseline DLA | ablated DLA | mediated effect |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        report["downstream_summary_rows"],
        key=lambda item: -abs(float(item["mean_mediated_effect"])),
    )[: report["markdown_top_k_rows"]]:
        lines.append(
            "| {scalar} | {endpoint} | `{source}` | `{downstream}` | {n} | {base:.6g} | {ablated:.6g} | {mediated:.6g} |".format(
                scalar=row["scalar_name"],
                endpoint=row["endpoint_kind"],
                source=row["source_component"],
                downstream=row["downstream_component"],
                n=int(row["num_observations"]),
                base=float(row["mean_baseline_downstream_dla"]),
                ablated=float(row["mean_ablated_downstream_dla"]),
                mediated=float(row["mean_mediated_effect"]),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- source rows: `{report['source_rows_path']}`",
            f"- downstream rows: `{report['downstream_rows_path']}`",
            f"- source summary rows: `{report['source_summary_rows_path']}`",
            f"- downstream summary rows: `{report['downstream_summary_rows_path']}`",
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


def run_output_mediated_causal_decomposition(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    pair_types: list[str],
    source_components: list[str],
    downstream_components: list[str],
    device_name: str = "mps",
    scalar_names: list[str] | None = None,
    margin_sides: list[str] | None = None,
    endpoint_roles: list[str] | None = None,
    split_filter: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    scalar_value_tolerance: float = 1.0e-4,
    markdown_top_k_rows: int = 48,
    plot_top_k_rows: int = 24,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if not pair_types:
        raise ValueError("output-mediated-causal-decomposition requires at least one pair type.")
    if not source_components:
        raise ValueError("source_components must not be empty.")
    if not downstream_components:
        raise ValueError("downstream_components must not be empty.")
    if set(source_components) & set(downstream_components):
        overlap = sorted(set(source_components) & set(downstream_components))
        raise ValueError(f"Source and downstream component sets must be disjoint; overlap={overlap}")
    if max_pairs_per_type <= 0:
        raise ValueError("max_pairs_per_type must be positive.")
    if min_pairs_per_type <= 0:
        raise ValueError("min_pairs_per_type must be positive.")
    if scalar_value_tolerance < 0.0:
        raise ValueError("scalar_value_tolerance must be non-negative.")
    if markdown_top_k_rows <= 0:
        raise ValueError("markdown_top_k_rows must be positive.")
    if plot_top_k_rows <= 0:
        raise ValueError("plot_top_k_rows must be positive.")

    resolved_scalars = _resolve_unique_values(
        values=scalar_names,
        default_values=["negative_answer_loss", "correct_value_logit", "fixed_source_competitor_margin", "fixed_target_competitor_margin"],
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
        allowed_values=OUTPUT_MEDIATED_ENDPOINT_ROLES,
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
    resolved_downstream = _filter_component_labels(
        requested_components=downstream_components,
        available_components=available_components,
    )
    _validate_maskable_components(
        components=resolved_sources,
        num_layers=len(model.blocks),
        num_heads=model.spec.n_heads,
        device=device,
    )
    _validate_maskable_components(
        components=resolved_downstream,
        num_layers=len(model.blocks),
        num_heads=model.spec.n_heads,
        device=device,
    )

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
    endpoint_requests = [
        request
        for request in _build_endpoint_requests(scalar_pair_rows=scalar_pair_rows, scalar_names=resolved_scalars)
        if str(request["endpoint_kind"]) in set(resolved_endpoint_roles)
    ]
    if not endpoint_requests:
        raise RuntimeError("No endpoint requests survived endpoint-role filters.")

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_components = list(dict.fromkeys(resolved_sources + resolved_downstream))
    print(
        "[output-mediated-causal-decomposition] "
        f"pairs={len(pairs_by_id)} scalar_rows={len(scalar_pair_rows)} endpoints={len(endpoint_keys)} "
        f"sources={resolved_sources} downstream={resolved_downstream} scalars={resolved_scalars} "
        f"endpoint_roles={resolved_endpoint_roles} device={device_name}",
        flush=True,
    )
    from circuit.analysis.output_route_closure import _compute_endpoint_payloads

    baseline_payloads = _compute_endpoint_payloads(
        model=model,
        checkpoint_paths_by_step=checkpoint_paths_by_step,
        pairs_by_id=pairs_by_id,
        endpoint_keys=endpoint_keys,
        component_labels=baseline_components,
        batch_size=spec.evaluation.batch_size,
        pad_token_id=vocab.pad_token_id,
        device=device,
    )
    baseline_values = _compute_endpoint_component_values(
        model=model,
        requests=endpoint_requests,
        endpoint_payloads=baseline_payloads,
        component_labels=baseline_components,
        device=device,
    )
    _validate_baseline_scalar_values(
        scalar_pair_rows=scalar_pair_rows,
        baseline_values=baseline_values,
        scalar_names=resolved_scalars,
        endpoint_roles=resolved_endpoint_roles,
        tolerance=scalar_value_tolerance,
    )

    source_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    request_by_id = {request["request_id"]: request for request in endpoint_requests}
    for source_component in resolved_sources:
        print(
            f"[output-mediated-causal-decomposition] source={source_component} computing ablated downstream DLA",
            flush=True,
        )
        ablated_payloads = _compute_endpoint_payloads_with_source_ablation(
            model=model,
            checkpoint_paths_by_step=checkpoint_paths_by_step,
            pairs_by_id=pairs_by_id,
            endpoint_keys=endpoint_keys,
            component_labels=resolved_downstream,
            source_component=source_component,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
        )
        ablated_values = _compute_endpoint_component_values(
            model=model,
            requests=endpoint_requests,
            endpoint_payloads=ablated_payloads,
            component_labels=resolved_downstream,
            device=device,
        )
        for request_id, request in sorted(request_by_id.items()):
            baseline = baseline_values.get(request_id)
            ablated = ablated_values.get(request_id)
            if baseline is None:
                raise KeyError(f"Missing baseline values for request {request_id}.")
            if ablated is None:
                raise KeyError(f"Missing ablated values for request {request_id} source={source_component}.")
            scalar_name = str(request["scalar_name"])
            baseline_scalar = float(baseline["scalar_value_recomputed"])
            ablated_scalar = float(ablated["scalar_value_recomputed"])
            total_effect = baseline_scalar - ablated_scalar
            direct_source_dla = float(baseline["component_values"][source_component])
            mediated_by_component: dict[str, float] = {}
            for downstream_component in resolved_downstream:
                baseline_downstream_dla = float(baseline["component_values"][downstream_component])
                ablated_downstream_dla = float(ablated["component_values"][downstream_component])
                mediated_effect = baseline_downstream_dla - ablated_downstream_dla
                mediated_by_component[downstream_component] = mediated_effect
                downstream_rows.append(
                    {
                        "source_step": int(request_id[0]),
                        "target_step": int(request_id[1]),
                        "pair_id": str(request_id[2]),
                        "margin_side": str(request_id[3]),
                        "scalar_name": scalar_name,
                        "endpoint_kind": str(request_id[5]),
                        "step": int(request["step"]),
                        "checkpoint": str(request["checkpoint"]),
                        "source_component": source_component,
                        "downstream_component": downstream_component,
                        "baseline_downstream_dla": baseline_downstream_dla,
                        "ablated_downstream_dla": ablated_downstream_dla,
                        "mediated_effect": mediated_effect,
                    }
                )
            mediated_sum = sum(mediated_by_component.values())
            direct_plus_mediated = direct_source_dla + mediated_sum
            source_rows.append(
                {
                    "source_step": int(request_id[0]),
                    "target_step": int(request_id[1]),
                    "pair_id": str(request_id[2]),
                    "margin_side": str(request_id[3]),
                    "scalar_name": scalar_name,
                    "endpoint_kind": str(request_id[5]),
                    "step": int(request["step"]),
                    "checkpoint": str(request["checkpoint"]),
                    "source_component": source_component,
                    "baseline_scalar": baseline_scalar,
                    "source_ablated_scalar": ablated_scalar,
                    "total_causal_effect": total_effect,
                    "direct_source_dla": direct_source_dla,
                    "mediated_downstream_sum": mediated_sum,
                    "direct_plus_mediated": direct_plus_mediated,
                    "mediation_residual": total_effect - direct_plus_mediated,
                    "mediated_by_component": mediated_by_component,
                }
            )
    if not source_rows:
        raise RuntimeError("No source mediation rows were produced.")
    if not downstream_rows:
        raise RuntimeError("No downstream mediation rows were produced.")

    source_summary_rows = _summarize_source_rows(source_rows)
    downstream_summary_rows = _summarize_downstream_rows(downstream_rows)

    source_rows_path = output_dir / "output_mediated_causal_decomposition_source_rows.jsonl"
    downstream_rows_path = output_dir / "output_mediated_causal_decomposition_downstream_rows.jsonl"
    source_summary_rows_path = output_dir / "output_mediated_causal_decomposition_source_summary_rows.jsonl"
    downstream_summary_rows_path = output_dir / "output_mediated_causal_decomposition_downstream_summary_rows.jsonl"
    pair_rows_path = output_dir / "output_mediated_causal_decomposition_pairs.jsonl"
    report_path = output_dir / "output_mediated_causal_decomposition_report.json"
    markdown_path = output_dir / "output_mediated_causal_decomposition_report.md"
    write_jsonl(source_rows_path, source_rows)
    write_jsonl(downstream_rows_path, downstream_rows)
    write_jsonl(source_summary_rows_path, source_summary_rows)
    write_jsonl(downstream_summary_rows_path, downstream_summary_rows)
    write_jsonl(
        pair_rows_path,
        [{key: value for key, value in pair.items() if key not in {"clean_record", "corrupted_record"}} for pair in pairs_by_id.values()],
    )

    plot_paths: dict[str, Path] = {}
    source_plot = _plot_source_mediation(
        source_summary_rows=source_summary_rows,
        output_path=output_dir / "output_mediated_causal_decomposition_source_mediation.svg",
        top_k_rows=plot_top_k_rows,
    )
    if source_plot is not None:
        plot_paths["source_mediation"] = source_plot
    downstream_plot = _plot_downstream_mediation(
        downstream_summary_rows=downstream_summary_rows,
        output_path=output_dir / "output_mediated_causal_decomposition_downstream_mediation.svg",
        top_k_rows=plot_top_k_rows,
    )
    if downstream_plot is not None:
        plot_paths["downstream_mediation"] = downstream_plot

    report = {
        "schema_version": OUTPUT_MEDIATED_CAUSAL_DECOMPOSITION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "device": device_name,
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names": resolved_scalars,
        "endpoint_roles": resolved_endpoint_roles,
        "source_components": resolved_sources,
        "downstream_components": resolved_downstream,
        "scalar_value_tolerance": scalar_value_tolerance,
        "markdown_top_k_rows": markdown_top_k_rows,
        "plot_top_k_rows": plot_top_k_rows,
        "checkpoint_paths_by_step": {str(step): str(path) for step, path in checkpoint_paths_by_step.items()},
        "pair_construction": pair_construction,
        "calculation": {
            "total_effect": "scalar(theta) - scalar(theta with source component ablated)",
            "direct_effect": "baseline DLA of the source component",
            "mediated_effect": "baseline downstream DLA - downstream DLA after source component ablation",
            "residual": "total_effect - direct_effect - sum(downstream mediated effects)",
        },
        "source_rows_path": str(source_rows_path),
        "downstream_rows_path": str(downstream_rows_path),
        "source_summary_rows_path": str(source_summary_rows_path),
        "downstream_summary_rows_path": str(downstream_summary_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "source_summary_rows": source_summary_rows,
        "downstream_summary_rows": downstream_summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    print(
        f"[output-mediated-causal-decomposition] complete report={report_path} rows={source_rows_path}",
        flush=True,
    )
    return (
        report_path,
        markdown_path,
        source_rows_path,
        downstream_rows_path,
        source_summary_rows_path,
        downstream_summary_rows_path,
        pair_rows_path,
        plot_paths,
    )
