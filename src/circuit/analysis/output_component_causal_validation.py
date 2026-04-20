from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import re
import shutil
from typing import Any

import torch

from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.output_route_closure import (
    OUTPUT_ROUTE_MARGIN_SIDES,
    OUTPUT_ROUTE_SCALARS,
    _build_endpoint_requests,
    _checkpoint_paths_by_step,
    _component_labels,
    _compute_endpoint_component_values,
    _compute_endpoint_payloads,
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
from circuit.io import iter_jsonl, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


OUTPUT_COMPONENT_CAUSAL_VALIDATION_SCHEMA_VERSION = 1
OUTPUT_COMPONENT_ENDPOINT_ROLES = ["source", "target"]


_HEAD_COMPONENT_RE = re.compile(r"^L(?P<layer>\d+)H(?P<head>\d+)$")
_MLP_COMPONENT_RE = re.compile(r"^L(?P<layer>\d+)MLP$")


def _component_mask_kwargs(
    *,
    component: str,
    num_layers: int,
    num_heads: int,
    device: torch.device,
) -> dict[str, Any]:
    head_match = _HEAD_COMPONENT_RE.match(component)
    if head_match is not None:
        layer = int(head_match.group("layer"))
        head = int(head_match.group("head"))
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"Head component layer out of range for {component!r}: num_layers={num_layers}.")
        if head < 0 or head >= num_heads:
            raise ValueError(f"Head component head out of range for {component!r}: num_heads={num_heads}.")
        head_mask = {layer_index: torch.ones(num_heads, device=device) for layer_index in range(num_layers)}
        head_mask[layer][head] = 0.0
        return {"head_mask": head_mask}

    mlp_match = _MLP_COMPONENT_RE.match(component)
    if mlp_match is not None:
        layer = int(mlp_match.group("layer"))
        if layer < 0 or layer >= num_layers:
            raise ValueError(f"MLP component layer out of range for {component!r}: num_layers={num_layers}.")
        mlp_mask = {layer_index: 1.0 for layer_index in range(num_layers)}
        mlp_mask[layer] = 0.0
        return {"mlp_mask": mlp_mask}

    if component == "embedding":
        raise ValueError(
            "Embedding DLA cannot be causally validated by this tool. "
            "It is not a locally maskable residual write like an attention head or MLP block."
        )
    raise ValueError(f"Unsupported causal-validation component label: {component!r}.")


def _validate_causally_maskable_components(
    *,
    components: list[str],
    num_layers: int,
    num_heads: int,
    device: torch.device,
) -> None:
    for component in components:
        _component_mask_kwargs(component=component, num_layers=num_layers, num_heads=num_heads, device=device)


def _load_coefficient_components(
    *,
    coefficient_rows_path: Path,
    scalar_names: list[str],
    switch_buckets: list[str],
    top_k_components: int,
) -> list[str]:
    if top_k_components <= 0:
        raise ValueError("top_k_components must be positive when selecting from coefficient rows.")
    rows = [row for row in iter_jsonl(coefficient_rows_path)]
    if not rows:
        raise RuntimeError(f"Coefficient rows file is empty: {coefficient_rows_path}")
    scalar_filter = set(scalar_names)
    bucket_filter = set(switch_buckets)
    candidates = [
        row
        for row in rows
        if str(row.get("scalar_name")) in scalar_filter and str(row.get("switch_bucket")) in bucket_filter
    ]
    if not candidates:
        raise RuntimeError(
            "No coefficient rows survived scalar/switch filters: "
            f"scalars={scalar_names} switch_buckets={switch_buckets}"
        )
    ranked = sorted(candidates, key=lambda row: abs(float(row["mean_contribution"])), reverse=True)
    selected: list[str] = []
    for row in ranked:
        component = str(row["component"])
        if component not in selected:
            selected.append(component)
        if len(selected) >= top_k_components:
            break
    if not selected:
        raise RuntimeError("Coefficient-based component selection produced an empty component list.")
    return selected


def _scalar_from_logits(
    *,
    logits: torch.Tensor,
    scalar_name: str,
    answer_target_id: int,
    source_best_wrong_token_id: int,
    target_best_wrong_token_id: int,
    endpoint_kind: str,
) -> float:
    if logits.ndim != 1:
        raise ValueError(f"Expected rank-1 logits for scalar calculation, got shape {tuple(logits.shape)}.")
    if endpoint_kind not in OUTPUT_COMPONENT_ENDPOINT_ROLES:
        raise ValueError(f"Unsupported endpoint kind {endpoint_kind!r}.")
    correct = logits[int(answer_target_id)]
    source_wrong = logits[int(source_best_wrong_token_id)]
    target_wrong = logits[int(target_best_wrong_token_id)]
    if scalar_name == "moving_answer_margin":
        wrong = source_wrong if endpoint_kind == "source" else target_wrong
        value = correct - wrong
    elif scalar_name == "fixed_source_competitor_margin":
        value = correct - source_wrong
    elif scalar_name == "fixed_target_competitor_margin":
        value = correct - target_wrong
    elif scalar_name == "correct_value_logit":
        value = correct
    elif scalar_name == "source_best_wrong_logit":
        value = source_wrong
    elif scalar_name == "target_best_wrong_logit":
        value = target_wrong
    elif scalar_name == "negative_answer_loss":
        value = torch.log_softmax(logits, dim=-1)[int(answer_target_id)]
    else:
        raise ValueError(f"Unsupported scalar {scalar_name!r}; expected one of {OUTPUT_ROUTE_SCALARS}.")
    return float(value.detach().float().cpu().item())


def _compute_endpoint_logits(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    endpoint_keys: set[tuple[int, str]],
    component_labels: list[str],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    if not endpoint_keys:
        raise ValueError("endpoint_keys must not be empty.")
    logits_by_key: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    pair_ids = sorted(pairs_by_id)
    num_layers = len(model.blocks)
    num_heads = model.spec.n_heads
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
        if payload_step != int(step):
            raise RuntimeError(f"Checkpoint step mismatch: requested={step} payload={payload_step} path={checkpoint_path}")
        side_key = f"{margin_side}_record"
        for start_index in range(0, len(pair_ids), batch_size):
            batch_pair_ids = pair_ids[start_index : start_index + batch_size]
            records = [pairs_by_id[pair_id][side_key] for pair_id in batch_pair_ids]
            batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
            model_kwargs_by_component: list[tuple[str, dict[str, Any]]] = [("__baseline__", {})]
            for component in component_labels:
                model_kwargs_by_component.append(
                    (
                        component,
                        _component_mask_kwargs(
                            component=component,
                            num_layers=num_layers,
                            num_heads=num_heads,
                            device=device,
                        ),
                    )
                )
            for component_key, model_kwargs in model_kwargs_by_component:
                with torch.no_grad():
                    outputs = model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        **model_kwargs,
                    )
                answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
                if len(metadata["rows"]) != len(batch_pair_ids):
                    raise RuntimeError(
                        "Output component causal validation expected one answer row per pair: "
                        f"pairs={len(batch_pair_ids)} rows={len(metadata['rows'])}"
                    )
                for item_index, pair_id in enumerate(batch_pair_ids):
                    logits_by_key[(step, margin_side, pair_id, component_key)] = {
                        "answer_target_id": int(answer_targets[item_index].detach().cpu().item()),
                        "answer_logits": answer_logits[item_index].detach().float().cpu(),
                    }
    expected_count = len(endpoint_keys) * len(pairs_by_id) * (1 + len(component_labels))
    if len(logits_by_key) != expected_count:
        raise RuntimeError(f"Endpoint logit count mismatch: expected={expected_count} got={len(logits_by_key)}")
    return logits_by_key


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


def _summarize_validation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot summarize empty validation rows.")
    summaries: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (
                str(row["scalar_name"]),
                str(row["endpoint_kind"]),
                str(row["component"]),
            )
            for row in rows
        }
    )
    for scalar_name, endpoint_kind, component in group_keys:
        group = [
            row
            for row in rows
            if str(row["scalar_name"]) == scalar_name
            and str(row["endpoint_kind"]) == endpoint_kind
            and str(row["component"]) == component
        ]
        causal_effects = [float(row["causal_effect"]) for row in group]
        dla_values = [float(row["dla_contribution"]) for row in group]
        residuals = [float(row["causal_minus_dla"]) for row in group]
        summaries.append(
            {
                "scalar_name": scalar_name,
                "endpoint_kind": endpoint_kind,
                "component": component,
                "num_observations": len(group),
                "mean_baseline_scalar": _mean(
                    [float(row["baseline_scalar"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{component}/baseline",
                ),
                "mean_ablated_scalar": _mean(
                    [float(row["ablated_scalar"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{component}/ablated",
                ),
                "mean_causal_effect": _mean(
                    causal_effects,
                    label=f"{scalar_name}/{endpoint_kind}/{component}/causal",
                ),
                "mean_abs_causal_effect": _mean(
                    [abs(value) for value in causal_effects],
                    label=f"{scalar_name}/{endpoint_kind}/{component}/abs causal",
                ),
                "mean_dla_contribution": _mean(
                    dla_values,
                    label=f"{scalar_name}/{endpoint_kind}/{component}/dla",
                ),
                "mean_abs_dla_contribution": _mean(
                    [abs(value) for value in dla_values],
                    label=f"{scalar_name}/{endpoint_kind}/{component}/abs dla",
                ),
                "mean_causal_minus_dla": _mean(
                    residuals,
                    label=f"{scalar_name}/{endpoint_kind}/{component}/residual",
                ),
                "mean_abs_causal_minus_dla": _mean(
                    [abs(value) for value in residuals],
                    label=f"{scalar_name}/{endpoint_kind}/{component}/abs residual",
                ),
                "sign_match_fraction": _mean(
                    [1.0 if bool(row["sign_match"]) else 0.0 for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{component}/sign",
                ),
                "causal_vs_dla_r_squared": _safe_r_squared(
                    y_values=causal_effects,
                    predicted_values=dla_values,
                ),
                "causal_vs_dla_correlation": _safe_correlation(
                    x_values=dla_values,
                    y_values=causal_effects,
                ),
            }
        )
    return summaries


def _plot_causal_vs_dla(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not rows:
        return None
    _, plt = _import_matplotlib()
    scalars = sorted({str(row["scalar_name"]) for row in rows})
    fig, axes = plt.subplots(len(scalars), 1, figsize=(7, max(5, 4 * len(scalars))))
    if len(scalars) == 1:
        axes = [axes]
    for ax, scalar_name in zip(axes, scalars, strict=True):
        scalar_rows = [row for row in rows if str(row["scalar_name"]) == scalar_name]
        x_values = [float(row["dla_contribution"]) for row in scalar_rows]
        y_values = [float(row["causal_effect"]) for row in scalar_rows]
        ax.scatter(x_values, y_values, alpha=0.35, color="#376f8f")
        values = x_values + y_values
        min_value = min(values)
        max_value = max(values)
        if min_value == max_value:
            min_value -= 1.0
            max_value += 1.0
        ax.plot([min_value, max_value], [min_value, max_value], color="#777777", linestyle="--", linewidth=1.0)
        ax.axhline(0.0, color="#999999", linewidth=0.8)
        ax.axvline(0.0, color="#999999", linewidth=0.8)
        ax.set_title(f"{scalar_name}: causal effect vs endpoint DLA")
        ax.set_xlabel("DLA contribution")
        ax.set_ylabel("baseline scalar - ablated scalar")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_top_effects(
    *,
    summary_rows: list[dict[str, Any]],
    output_path: Path,
    top_k_components: int,
) -> Path | None:
    if not summary_rows:
        return None
    if top_k_components <= 0:
        raise ValueError("top_k_components must be positive.")
    selected: list[dict[str, Any]] = []
    for scalar_name in sorted({str(row["scalar_name"]) for row in summary_rows}):
        scalar_rows = [
            row
            for row in summary_rows
            if str(row["scalar_name"]) == scalar_name and str(row["endpoint_kind"]) == "target"
        ]
        selected.extend(
            sorted(
                scalar_rows,
                key=lambda row: abs(float(row["mean_causal_effect"])),
                reverse=True,
            )[:top_k_components]
        )
    if not selected:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(selected)), 6))
    labels = [f"{row['scalar_name']}:{row['component']}" for row in selected]
    values = [float(row["mean_causal_effect"]) for row in selected]
    colors = ["#376f8f" if value >= 0.0 else "#8f374a" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_title("Top output-component causal effects at target endpoints")
    ax.set_ylabel("mean baseline scalar - ablated scalar")
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
        "# Output-Component Causal Validation",
        "",
        "## Calculation",
        "",
        "This report tests whether component DLA is only a readout accounting term or whether the same component is causally load-bearing.",
        "",
        "```text",
        "DLA_{c,s}(theta, x) = component_write_c(theta, x) dot d scalar_s / d final_pre_layernorm_residual",
        "causal_effect_{c,s}(theta, x) = scalar_s(theta, x) - scalar_s(theta with component c ablated, x)",
        "validation gap = causal_effect_{c,s} - DLA_{c,s}",
        "```",
        "",
        "Attention components are ablated with head masks. MLP components are ablated with MLP block masks. Embedding is intentionally unsupported because it is not a locally maskable residual write.",
        "",
        "## Inputs",
        "",
        f"- scalar pair rows: `{report['scalar_pair_rows_path']}`",
        f"- probe set: `{report['probe_set_path']}`",
        f"- margin sides: `{report['margin_sides']}`",
        f"- pair types: `{report['pair_types']}`",
        f"- scalars: `{report['scalar_names']}`",
        f"- endpoint roles: `{report['endpoint_roles']}`",
        f"- components: `{report['component_labels']}`",
        "",
        "## Summary",
        "",
        "| scalar | endpoint | component | observations | mean causal effect | mean DLA | mean abs gap | sign match | corr | R squared |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        report["summary_rows"],
        key=lambda item: (
            str(item["scalar_name"]),
            str(item["endpoint_kind"]),
            -abs(float(item["mean_causal_effect"])),
        ),
    )[: report["markdown_top_k_rows"]]:
        corr = row["causal_vs_dla_correlation"]
        r2 = row["causal_vs_dla_r_squared"]
        lines.append(
            "| {scalar} | {endpoint} | `{component}` | {n} | {causal:.6g} | {dla:.6g} | {gap:.6g} | {sign:.3f} | {corr} | {r2} |".format(
                scalar=row["scalar_name"],
                endpoint=row["endpoint_kind"],
                component=row["component"],
                n=int(row["num_observations"]),
                causal=float(row["mean_causal_effect"]),
                dla=float(row["mean_dla_contribution"]),
                gap=float(row["mean_abs_causal_minus_dla"]),
                sign=float(row["sign_match_fraction"]),
                corr="" if corr is None else f"{float(corr):.4f}",
                r2="" if r2 is None else f"{float(r2):.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- validation rows: `{report['validation_rows_path']}`",
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


def run_output_component_causal_validation(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    pair_types: list[str],
    device_name: str = "mps",
    scalar_names: list[str] | None = None,
    margin_sides: list[str] | None = None,
    endpoint_roles: list[str] | None = None,
    component_labels: list[str] | None = None,
    coefficient_rows_path: Path | None = None,
    coefficient_switch_buckets: list[str] | None = None,
    split_filter: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    top_k_components: int = 8,
    scalar_value_tolerance: float = 1.0e-4,
    markdown_top_k_rows: int = 48,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Path]]:
    if not pair_types:
        raise ValueError("output-component-causal-validation requires at least one pair type.")
    if max_pairs_per_type <= 0:
        raise ValueError("max_pairs_per_type must be positive.")
    if min_pairs_per_type <= 0:
        raise ValueError("min_pairs_per_type must be positive.")
    if top_k_components <= 0:
        raise ValueError("top_k_components must be positive.")
    if scalar_value_tolerance < 0.0:
        raise ValueError("scalar_value_tolerance must be non-negative.")
    if markdown_top_k_rows <= 0:
        raise ValueError("markdown_top_k_rows must be positive.")

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
        allowed_values=OUTPUT_COMPONENT_ENDPOINT_ROLES,
        label="endpoint role",
    )
    resolved_coefficient_buckets = _resolve_unique_values(
        values=coefficient_switch_buckets,
        default_values=["all"],
        allowed_values=["all", "same_competitor", "competitor_switch"],
        label="coefficient switch bucket",
    )
    pair_types = sorted(set(pair_types), key=pair_types.index)

    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(device_name)
    model = build_model(spec.model, len(vocab.tokens), device)
    available_components = _component_labels(num_layers=len(model.blocks), num_heads=model.spec.n_heads)
    if component_labels is None:
        if coefficient_rows_path is None:
            raise ValueError("Provide --component values or --coefficient-rows for component selection.")
        requested_components = _load_coefficient_components(
            coefficient_rows_path=coefficient_rows_path,
            scalar_names=resolved_scalars,
            switch_buckets=resolved_coefficient_buckets,
            top_k_components=top_k_components,
        )
    else:
        requested_components = component_labels
    resolved_components = _filter_component_labels(
        requested_components=requested_components,
        available_components=available_components,
    )
    _validate_causally_maskable_components(
        components=resolved_components,
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

    print(
        "[output-component-causal-validation] "
        f"pairs={len(pairs_by_id)} scalar_rows={len(scalar_pair_rows)} endpoints={len(endpoint_keys)} "
        f"components={resolved_components} scalars={resolved_scalars} endpoint_roles={resolved_endpoint_roles} "
        f"device={device_name}",
        flush=True,
    )
    endpoint_payloads = _compute_endpoint_payloads(
        model=model,
        checkpoint_paths_by_step=checkpoint_paths_by_step,
        pairs_by_id=pairs_by_id,
        endpoint_keys=endpoint_keys,
        component_labels=resolved_components,
        batch_size=spec.evaluation.batch_size,
        pad_token_id=vocab.pad_token_id,
        device=device,
    )
    endpoint_component_values = _compute_endpoint_component_values(
        model=model,
        requests=endpoint_requests,
        endpoint_payloads=endpoint_payloads,
        component_labels=resolved_components,
        device=device,
    )
    endpoint_logits = _compute_endpoint_logits(
        model=model,
        checkpoint_paths_by_step=checkpoint_paths_by_step,
        pairs_by_id=pairs_by_id,
        endpoint_keys=endpoint_keys,
        component_labels=resolved_components,
        batch_size=spec.evaluation.batch_size,
        pad_token_id=vocab.pad_token_id,
        device=device,
    )

    validation_rows: list[dict[str, Any]] = []
    request_by_id = {request["request_id"]: request for request in endpoint_requests}
    for request_id, request in sorted(request_by_id.items()):
        values = endpoint_component_values.get(request_id)
        if values is None:
            raise KeyError(f"Missing DLA endpoint values for request {request_id}.")
        scalar_name = str(request["scalar_name"])
        endpoint_kind = str(request["endpoint_kind"])
        pair_id = str(request["pair_id"])
        step = int(request["step"])
        margin_side = str(request["margin_side"])
        baseline_key = (step, margin_side, pair_id, "__baseline__")
        baseline_payload = endpoint_logits.get(baseline_key)
        if baseline_payload is None:
            raise KeyError(f"Missing baseline logits for {baseline_key}.")
        expected_answer = int(request["answer_target_id"])
        if int(baseline_payload["answer_target_id"]) != expected_answer:
            raise RuntimeError(
                f"Baseline answer target mismatch for {baseline_key}: "
                f"expected={expected_answer} got={baseline_payload['answer_target_id']}"
            )
        baseline_scalar = _scalar_from_logits(
            logits=baseline_payload["answer_logits"],
            scalar_name=scalar_name,
            answer_target_id=expected_answer,
            source_best_wrong_token_id=int(request["source_best_wrong_token_id"]),
            target_best_wrong_token_id=int(request["target_best_wrong_token_id"]),
            endpoint_kind=endpoint_kind,
        )
        recomputed_delta = abs(float(values["scalar_value_recomputed"]) - baseline_scalar)
        if recomputed_delta > scalar_value_tolerance:
            raise RuntimeError(
                f"Endpoint scalar recomputation mismatch for {request_id}: "
                f"dla_payload={values['scalar_value_recomputed']:.6g} baseline={baseline_scalar:.6g} "
                f"delta={recomputed_delta:.6g} tolerance={scalar_value_tolerance:.6g}"
            )
        for component in resolved_components:
            ablated_key = (step, margin_side, pair_id, component)
            ablated_payload = endpoint_logits.get(ablated_key)
            if ablated_payload is None:
                raise KeyError(f"Missing ablated logits for {ablated_key}.")
            if int(ablated_payload["answer_target_id"]) != expected_answer:
                raise RuntimeError(
                    f"Ablated answer target mismatch for {ablated_key}: "
                    f"expected={expected_answer} got={ablated_payload['answer_target_id']}"
                )
            ablated_scalar = _scalar_from_logits(
                logits=ablated_payload["answer_logits"],
                scalar_name=scalar_name,
                answer_target_id=expected_answer,
                source_best_wrong_token_id=int(request["source_best_wrong_token_id"]),
                target_best_wrong_token_id=int(request["target_best_wrong_token_id"]),
                endpoint_kind=endpoint_kind,
            )
            causal_effect = baseline_scalar - ablated_scalar
            dla_contribution = float(values["component_values"][component])
            validation_rows.append(
                {
                    "source_step": int(request_id[0]),
                    "target_step": int(request_id[1]),
                    "pair_id": pair_id,
                    "margin_side": margin_side,
                    "scalar_name": scalar_name,
                    "endpoint_kind": endpoint_kind,
                    "step": step,
                    "checkpoint": str(request["checkpoint"]),
                    "component": component,
                    "answer_target_id": expected_answer,
                    "source_best_wrong_token_id": int(request["source_best_wrong_token_id"]),
                    "target_best_wrong_token_id": int(request["target_best_wrong_token_id"]),
                    "baseline_scalar": baseline_scalar,
                    "ablated_scalar": ablated_scalar,
                    "causal_effect": causal_effect,
                    "dla_contribution": dla_contribution,
                    "causal_minus_dla": causal_effect - dla_contribution,
                    "sign_match": (causal_effect == 0.0 and dla_contribution == 0.0)
                    or (causal_effect > 0.0 and dla_contribution > 0.0)
                    or (causal_effect < 0.0 and dla_contribution < 0.0),
                }
            )
    if not validation_rows:
        raise RuntimeError("No output-component causal validation rows were produced.")
    summary_rows = _summarize_validation_rows(validation_rows)

    validation_rows_path = output_dir / "output_component_causal_validation_rows.jsonl"
    summary_rows_path = output_dir / "output_component_causal_validation_summary_rows.jsonl"
    pair_rows_path = output_dir / "output_component_causal_validation_pairs.jsonl"
    report_path = output_dir / "output_component_causal_validation_report.json"
    markdown_path = output_dir / "output_component_causal_validation_report.md"
    write_jsonl(validation_rows_path, validation_rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(
        pair_rows_path,
        [{key: value for key, value in pair.items() if key not in {"clean_record", "corrupted_record"}} for pair in pairs_by_id.values()],
    )
    plot_paths: dict[str, Path] = {}
    causal_vs_dla_path = _plot_causal_vs_dla(
        rows=validation_rows,
        output_path=output_dir / "output_component_causal_validation_causal_vs_dla.svg",
    )
    if causal_vs_dla_path is not None:
        plot_paths["causal_vs_dla"] = causal_vs_dla_path
    top_effects_path = _plot_top_effects(
        summary_rows=summary_rows,
        output_path=output_dir / "output_component_causal_validation_top_effects.svg",
        top_k_components=top_k_components,
    )
    if top_effects_path is not None:
        plot_paths["top_effects"] = top_effects_path

    report = {
        "schema_version": OUTPUT_COMPONENT_CAUSAL_VALIDATION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "coefficient_rows_path": None if coefficient_rows_path is None else str(coefficient_rows_path),
        "device": device_name,
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names": resolved_scalars,
        "endpoint_roles": resolved_endpoint_roles,
        "coefficient_switch_buckets": resolved_coefficient_buckets,
        "component_labels": resolved_components,
        "top_k_components": top_k_components,
        "scalar_value_tolerance": scalar_value_tolerance,
        "markdown_top_k_rows": markdown_top_k_rows,
        "checkpoint_paths_by_step": {str(step): str(path) for step, path in checkpoint_paths_by_step.items()},
        "pair_construction": pair_construction,
        "calculation": {
            "dla": "DLA_{c,s} = component_write_c dot d scalar_s / d final_pre_layernorm_residual",
            "causal_effect": "scalar_s(theta, x) - scalar_s(theta with component c ablated, x)",
            "validation_gap": "causal_effect - DLA",
            "supported_components": "attention heads via head masks and MLP blocks via MLP masks",
            "unsupported_components": "embedding is not a locally maskable residual write and is rejected",
        },
        "validation_rows_path": str(validation_rows_path),
        "summary_rows_path": str(summary_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    print(
        f"[output-component-causal-validation] complete report={report_path} rows={validation_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, validation_rows_path, summary_rows_path, pair_rows_path, plot_paths
