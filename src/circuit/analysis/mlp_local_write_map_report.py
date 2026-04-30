from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.component_output_rescue import (
    _component_order,
    _component_patch_stage,
    _parse_local_component,
    _validate_downstream_components,
)
from circuit.analysis.contextual_svd_alignment import (
    CONTEXTUAL_GROUP_BY_OPTIONS,
    _abs_cosine,
    _role_subspace,
    _subspace_overlap,
)
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    GEOMETRY_POSITION_ROLES,
    _attention_key_positions,
    _checkpoint_step_from_path,
    _validate_single_query_batch,
)
from circuit.analysis.output_component_causal_validation import _component_mask_kwargs
from circuit.analysis.output_route_closure import (
    OUTPUT_ROUTE_MARGIN_SIDES,
    OUTPUT_ROUTE_SCALARS,
    _checkpoint_paths_by_step,
    _component_labels,
    _filter_component_labels,
    _load_scalar_pair_rows,
    _mean,
    _resolve_unique_values,
    _safe_r_squared,
    _selected_pairs_by_id,
)
from circuit.analysis.residual_delta_vector_report import (
    _endpoint_pair_ids,
    _filter_scalar_pair_rows_for_delta,
    _group_token_id,
    _token_label,
)
from circuit.analysis.residual_state_rescue import RESIDUAL_STATE_RESCUE_ENDPOINT_ROLES, _validate_maskable_components
from circuit.analysis.shared_feature_dynamics import _import_matplotlib
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


MLP_LOCAL_WRITE_MAP_SCHEMA_VERSION = 1


def _validate_mlp_components(*, components: list[str]) -> None:
    for component in components:
        kind, _, _ = _parse_local_component(component)
        if kind != "mlp":
            raise ValueError(f"mlp-local-write-map-report only accepts MLP components, got {component!r}.")


def _mlp_input_stage(component: str) -> str:
    kind, layer, _ = _parse_local_component(component)
    if kind != "mlp":
        raise ValueError(f"Expected an MLP component, got {component!r}.")
    return f"layer_{layer}_post_attn"


def _mlp_local_function(*, model: torch.nn.Module, component: str, residual_vectors: torch.Tensor) -> torch.Tensor:
    kind, layer, _ = _parse_local_component(component)
    if kind != "mlp":
        raise ValueError(f"Expected an MLP component, got {component!r}.")
    block = model.blocks[layer]
    normalized = block.ln_2(residual_vectors)
    output, _ = block.ff(normalized, return_hidden_state=False)
    return output


def _safe_norm_relative_error(*, actual: torch.Tensor, predicted: torch.Tensor, label: str) -> float:
    actual_norm = float(actual.norm().item())
    if actual_norm == 0.0:
        predicted_norm = float(predicted.norm().item())
        if predicted_norm != 0.0:
            raise RuntimeError(f"Cannot compute relative error for zero actual norm and nonzero prediction: {label}")
        return 0.0
    return float((actual - predicted).norm().item()) / actual_norm


def _summarize_local_rows(*, rows: list[dict[str, Any]], group_by_values: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for group_by in group_by_values:
            grouped[
                (
                    str(row["source_component"]),
                    str(row["component"]),
                    int(row["step"]),
                    str(row["endpoint_kind"]),
                    str(row["margin_side"]),
                    str(row["position_role"]),
                    group_by,
                )
            ].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (
        source_component,
        component,
        step,
        endpoint_kind,
        margin_side,
        position_role,
        group_by,
    ), group_rows in sorted(grouped.items()):
        input_norms = [float(row["input_delta_norm"]) for row in group_rows]
        actual_norms = [float(row["actual_output_delta_norm"]) for row in group_rows]
        jvp_ablated_norms = [float(row["jvp_ablated_norm"]) for row in group_rows]
        jvp_clean_norms = [float(row["jvp_clean_norm"]) for row in group_rows]
        jvp_ablated_cosines = [float(row["jvp_ablated_cosine_to_actual"]) for row in group_rows]
        jvp_clean_cosines = [float(row["jvp_clean_cosine_to_actual"]) for row in group_rows]
        jvp_ablated_errors = [float(row["jvp_ablated_relative_error"]) for row in group_rows]
        jvp_clean_errors = [float(row["jvp_clean_relative_error"]) for row in group_rows]
        actual_identity = [float(row[f"actual_{group_by}_identity_overlap"]) for row in group_rows]
        jvp_ablated_identity = [float(row[f"jvp_ablated_{group_by}_identity_overlap"]) for row in group_rows]
        jvp_clean_identity = [float(row[f"jvp_clean_{group_by}_identity_overlap"]) for row in group_rows]
        summary_rows.append(
            {
                "source_component": source_component,
                "component": component,
                "component_order": int(group_rows[0]["component_order"]),
                "step": step,
                "endpoint_kind": endpoint_kind,
                "margin_side": margin_side,
                "input_stage": str(group_rows[0]["input_stage"]),
                "output_stage": str(group_rows[0]["output_stage"]),
                "position_role": position_role,
                "group_by": group_by,
                "num_rows": len(group_rows),
                "mean_input_delta_norm": _mean(input_norms, label=f"{component}/{step}/{group_by} input norm"),
                "mean_actual_output_delta_norm": _mean(
                    actual_norms,
                    label=f"{component}/{step}/{group_by} actual output norm",
                ),
                "mean_jvp_ablated_norm": _mean(jvp_ablated_norms, label=f"{component}/{step}/{group_by} ablated jvp norm"),
                "mean_jvp_clean_norm": _mean(jvp_clean_norms, label=f"{component}/{step}/{group_by} clean jvp norm"),
                "mean_jvp_ablated_cosine_to_actual": _mean(
                    jvp_ablated_cosines,
                    label=f"{component}/{step}/{group_by} ablated jvp cosine",
                ),
                "mean_jvp_clean_cosine_to_actual": _mean(
                    jvp_clean_cosines,
                    label=f"{component}/{step}/{group_by} clean jvp cosine",
                ),
                "mean_jvp_ablated_relative_error": _mean(
                    jvp_ablated_errors,
                    label=f"{component}/{step}/{group_by} ablated jvp error",
                ),
                "mean_jvp_clean_relative_error": _mean(
                    jvp_clean_errors,
                    label=f"{component}/{step}/{group_by} clean jvp error",
                ),
                "mean_actual_identity_overlap": _mean(
                    actual_identity,
                    label=f"{component}/{step}/{group_by} actual identity",
                ),
                "mean_jvp_ablated_identity_overlap": _mean(
                    jvp_ablated_identity,
                    label=f"{component}/{step}/{group_by} ablated jvp identity",
                ),
                "mean_jvp_clean_identity_overlap": _mean(
                    jvp_clean_identity,
                    label=f"{component}/{step}/{group_by} clean jvp identity",
                ),
                "actual_identity_overlap_r_squared_vs_actual_norm": _safe_r_squared(actual_norms, actual_identity),
            }
        )
    return summary_rows


def _plot_local_summary(
    *,
    summary_rows: list[dict[str, Any]],
    output_path: Path,
    top_k_rows: int,
) -> Path | None:
    if not summary_rows:
        return None
    _, plt = _import_matplotlib()
    rows = sorted(
        summary_rows,
        key=lambda row: abs(float(row["mean_actual_identity_overlap"])) * float(row["mean_actual_output_delta_norm"]),
        reverse=True,
    )[:top_k_rows]
    if not rows:
        return None
    labels = [
        f"{row['component']} {row['step']} {row['endpoint_kind']} {row['group_by']}"
        for row in rows
    ]
    actual_values = [float(row["mean_actual_identity_overlap"]) for row in rows]
    jvp_values = [float(row["mean_jvp_ablated_identity_overlap"]) for row in rows]
    height = max(4.0, 0.4 * len(rows))
    fig, ax = plt.subplots(figsize=(12, height))
    y_positions = list(range(len(rows)))
    ax.barh([position - 0.18 for position in y_positions], actual_values, height=0.34, color="#356ca5", label="actual")
    ax.barh([position + 0.18 for position in y_positions], jvp_values, height=0.34, color="#2f7d67", label="JVP at ablated")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean overlap with contextual identity subspace")
    ax.set_title("MLP Local Write Map Semantic Alignment")
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _write_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary_rows = sorted(
        report["summary_rows"],
        key=lambda row: abs(float(row["mean_actual_identity_overlap"])) * float(row["mean_actual_output_delta_norm"]),
        reverse=True,
    )[: int(report["markdown_top_k_rows"])]
    lines = [
        "# MLP Local Write Map Report",
        "",
        "This report asks whether an MLP locally maps a source-component residual perturbation into semantic answer/value directions.",
        "",
        "## Scope",
        "",
        f"- sources: `{', '.join(report['source_components'])}`",
        f"- components: `{', '.join(report['components'])}`",
        f"- position roles: `{', '.join(report['position_roles'])}`",
        f"- group-by labels: `{', '.join(report['group_by'])}`",
        f"- pair types: `{', '.join(report['pair_types'])}`",
        f"- endpoint roles: `{', '.join(report['endpoint_roles'])}`",
        f"- PCA rank: `{report['pca_rank']}`",
        "",
        "## Calculation",
        "",
        "For an MLP component, define:",
        "",
        "`F_i(z) = MLP_i(LN_2(z))`",
        "",
        "For a source component ablation, at the selected position:",
        "",
        "`delta_in = z_clean - z_source_ablated`",
        "",
        "`actual_delta_out = F_i(z_clean) - F_i(z_source_ablated)`",
        "",
        "The weight-level local prediction is a Jacobian-vector product:",
        "",
        "`jvp_ablated = J_F_i(z_source_ablated) delta_in`",
        "",
        "The report compares `actual_delta_out` and the JVP with contextual clean residual subspaces at the MLP output stage.",
        "",
        "## Top Local Write Maps",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| source | component | step | endpoint | group | rows | input norm | actual norm | actual overlap | JVP overlap | JVP cosine | JVP rel err |",
                "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary_rows:
            lines.append(
                f"| `{row['source_component']}` | `{row['component']}` | {row['step']} | "
                f"`{row['endpoint_kind']}` | `{row['group_by']}` | {row['num_rows']} | "
                f"{row['mean_input_delta_norm']:.6g} | {row['mean_actual_output_delta_norm']:.6g} | "
                f"{row['mean_actual_identity_overlap']:.6g} | {row['mean_jvp_ablated_identity_overlap']:.6g} | "
                f"{row['mean_jvp_ablated_cosine_to_actual']:.6g} | {row['mean_jvp_ablated_relative_error']:.6g} |"
            )
    else:
        lines.append("No summary rows were produced.")
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- local rows: `{report['local_rows_path']}`")
    lines.append(f"- summary rows: `{report['summary_rows_path']}`")
    lines.append(f"- subspace rows: `{report['subspace_rows_path']}`")
    lines.append(f"- pair rows: `{report['pair_rows_path']}`")
    for label, plot_path in sorted(plot_paths.items()):
        lines.append(f"- {label}: `{plot_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _compute_mlp_local_rows_for_source(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    endpoint_pair_ids: dict[tuple[int, str, str], set[str]],
    source_component: str,
    components: list[str],
    position_roles: list[str],
    group_by_values: list[str],
    pca_rank: int,
    batch_size: int,
    pad_token_id: int,
    vocab: Vocabulary,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    num_layers = len(model.blocks)
    num_heads = model.spec.n_heads
    source_mask_kwargs = _component_mask_kwargs(
        component=source_component,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
    )
    local_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    pair_ids = sorted(pairs_by_id)
    input_stage_by_component = {component: _mlp_input_stage(component) for component in components}
    output_stage_by_component = {component: _component_patch_stage(component) for component in components}
    component_order = {component: _component_order(component) for component in components}

    for step, margin_side, endpoint_kind in sorted(endpoint_pair_ids):
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
        endpoint_required_pair_ids = endpoint_pair_ids[(step, margin_side, endpoint_kind)]
        vectors_by_key: dict[tuple[str, str, str], dict[int, list[torch.Tensor]]] = {
            (stage, position_role, group_by): defaultdict(list)
            for stage in sorted(set(output_stage_by_component.values()))
            for position_role in position_roles
            for group_by in group_by_values
        }
        endpoint_payloads: list[dict[str, Any]] = []
        for start_index in range(0, len(pair_ids), batch_size):
            batch_pair_ids = pair_ids[start_index : start_index + batch_size]
            active_pair_ids = [pair_id for pair_id in batch_pair_ids if pair_id in endpoint_required_pair_ids]
            if not active_pair_ids:
                continue
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
                raise RuntimeError("MLP local write map report requires residual streams.")
            _, answer_targets, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            _, _, ablated_metadata = extract_answer_logits(ablated_outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=clean_metadata, label="mlp-local-write-map clean")
            _validate_single_query_batch(batch=batch, metadata=ablated_metadata, label="mlp-local-write-map ablated")
            for flat_index in range(int(clean_metadata["rows"].numel())):
                query_batch_row = int(clean_metadata["rows"][flat_index].item())
                pair_id = str(batch_pair_ids[query_batch_row])
                if pair_id not in endpoint_required_pair_ids:
                    continue
                query_index = int(clean_metadata["query_indices"][flat_index].item())
                record = batch["records"][query_batch_row]
                prediction_position = int(clean_metadata["prediction_positions"][flat_index].item())
                query_key_token_id = int(
                    batch["input_ids"][
                        query_batch_row,
                        int(clean_metadata["query_key_positions"][flat_index].item()),
                    ].item()
                )
                support_value_token_id = int(
                    batch["input_ids"][
                        query_batch_row,
                        int(clean_metadata["support_value_positions"][flat_index].item()),
                    ].item()
                )
                answer_value_token_id = int(answer_targets[flat_index].item())
                for position_role in position_roles:
                    context_batch_row, positions = _attention_key_positions(
                        batch=batch,
                        metadata=clean_metadata,
                        flat_index=flat_index,
                        position_role=position_role,
                        max_position=prediction_position,
                    )
                    for component in components:
                        input_stage = input_stage_by_component[component]
                        output_stage = output_stage_by_component[component]
                        clean_input_stage = clean_outputs.residual_streams[input_stage].detach()
                        ablated_input_stage = ablated_outputs.residual_streams[input_stage].detach()
                        clean_output_stage = clean_outputs.residual_streams[output_stage].detach().float().cpu()
                        clean_input_vectors = torch.stack(
                            [
                                clean_input_stage[context_batch_row, int(position), :].clone()
                                for position in positions
                            ],
                            dim=0,
                        )
                        ablated_input_vectors = torch.stack(
                            [
                                ablated_input_stage[context_batch_row, int(position), :].clone()
                                for position in positions
                            ],
                            dim=0,
                        )
                        for position in positions:
                            clean_output_vector = clean_output_stage[context_batch_row, int(position), :].clone()
                            for group_by in group_by_values:
                                group_token_id = _group_token_id(
                                    group_by=group_by,
                                    batch=batch,
                                    metadata=clean_metadata,
                                    answer_targets=answer_targets,
                                    flat_index=flat_index,
                                    context_batch_row=context_batch_row,
                                    context_position=int(position),
                                )
                                vectors_by_key[(output_stage, position_role, group_by)][group_token_id].append(
                                    clean_output_vector
                                )
                        clean_input_mean = clean_input_vectors.mean(dim=0)
                        ablated_input_mean = ablated_input_vectors.mean(dim=0)
                        delta_input = clean_input_mean - ablated_input_mean
                        clean_base = clean_input_mean.detach()
                        ablated_base = ablated_input_mean.detach()
                        with torch.no_grad():
                            clean_output = _mlp_local_function(
                                model=model,
                                component=component,
                                residual_vectors=clean_base.unsqueeze(0),
                            ).squeeze(0)
                            ablated_output = _mlp_local_function(
                                model=model,
                                component=component,
                                residual_vectors=ablated_base.unsqueeze(0),
                            ).squeeze(0)
                        actual_output_delta = clean_output.detach() - ablated_output.detach()

                        _, jvp_ablated = torch.autograd.functional.jvp(
                            lambda residual: _mlp_local_function(
                                model=model,
                                component=component,
                                residual_vectors=residual,
                            ),
                            (ablated_base.unsqueeze(0),),
                            (delta_input.unsqueeze(0),),
                            create_graph=False,
                            strict=True,
                        )
                        _, jvp_clean = torch.autograd.functional.jvp(
                            lambda residual: _mlp_local_function(
                                model=model,
                                component=component,
                                residual_vectors=residual,
                            ),
                            (clean_base.unsqueeze(0),),
                            (delta_input.unsqueeze(0),),
                            create_graph=False,
                            strict=True,
                        )
                        jvp_ablated_vector = jvp_ablated.squeeze(0).detach()
                        jvp_clean_vector = jvp_clean.squeeze(0).detach()
                        endpoint_payloads.append(
                            {
                                "source_component": source_component,
                                "component": component,
                                "component_order": component_order[component],
                                "step": step,
                                "checkpoint": str(checkpoint_path),
                                "endpoint_kind": endpoint_kind,
                                "margin_side": margin_side,
                                "pair_id": pair_id,
                                "pair_type": str(pairs_by_id[pair_id]["pair_type"]),
                                "sample_id": str(record["sample_id"]),
                                "split": str(record["split"]),
                                "query_index": query_index,
                                "input_stage": input_stage,
                                "output_stage": output_stage,
                                "position_role": position_role,
                                "selected_positions": [int(position) for position in positions],
                                "num_positions": len(positions),
                                "query_key_token_id": query_key_token_id,
                                "query_key_token": _token_label(vocab, query_key_token_id),
                                "support_value_token_id": support_value_token_id,
                                "support_value_token": _token_label(vocab, support_value_token_id),
                                "answer_value_token_id": answer_value_token_id,
                                "answer_value_token": _token_label(vocab, answer_value_token_id),
                                "clean_input_norm": float(clean_input_mean.norm().item()),
                                "ablated_input_norm": float(ablated_input_mean.norm().item()),
                                "input_delta_norm": float(delta_input.norm().item()),
                                "clean_output_norm": float(clean_output.norm().item()),
                                "ablated_output_norm": float(ablated_output.norm().item()),
                                "actual_output_delta_norm": float(actual_output_delta.norm().item()),
                                "jvp_ablated_norm": float(jvp_ablated_vector.norm().item()),
                                "jvp_clean_norm": float(jvp_clean_vector.norm().item()),
                                "jvp_ablated_cosine_to_actual": _abs_cosine(
                                    jvp_ablated_vector.cpu(),
                                    actual_output_delta.cpu(),
                                    label=f"{source_component}/{component}/{step}/{pair_id}/jvp_ablated_actual",
                                ),
                                "jvp_clean_cosine_to_actual": _abs_cosine(
                                    jvp_clean_vector.cpu(),
                                    actual_output_delta.cpu(),
                                    label=f"{source_component}/{component}/{step}/{pair_id}/jvp_clean_actual",
                                ),
                                "jvp_ablated_relative_error": _safe_norm_relative_error(
                                    actual=actual_output_delta.cpu(),
                                    predicted=jvp_ablated_vector.cpu(),
                                    label=f"{source_component}/{component}/{step}/{pair_id}/jvp_ablated",
                                ),
                                "jvp_clean_relative_error": _safe_norm_relative_error(
                                    actual=actual_output_delta.cpu(),
                                    predicted=jvp_clean_vector.cpu(),
                                    label=f"{source_component}/{component}/{step}/{pair_id}/jvp_clean",
                                ),
                                "_actual_output_delta": actual_output_delta.detach().float().cpu(),
                                "_jvp_ablated": jvp_ablated_vector.detach().float().cpu(),
                                "_jvp_clean": jvp_clean_vector.detach().float().cpu(),
                            }
                        )
        if not endpoint_payloads:
            raise RuntimeError(
                f"No MLP local write-map payloads built for source={source_component} endpoint="
                f"{step}/{margin_side}/{endpoint_kind}."
            )
        subspaces: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
        for key, vectors_by_token in sorted(vectors_by_key.items()):
            output_stage, position_role, group_by = key
            subspace, summary = _role_subspace(
                role_label=f"{output_stage}:{position_role}:{group_by}",
                context_role=position_role,
                group_by=group_by,
                vectors_by_token=vectors_by_token,
                vocab=vocab,
                pca_rank=pca_rank,
            )
            subspaces[key] = subspace
            subspace_rows.append(
                {
                    "source_component": source_component,
                    "step": step,
                    "checkpoint": str(checkpoint_path),
                    "endpoint_kind": endpoint_kind,
                    "margin_side": margin_side,
                    "output_stage": output_stage,
                    "position_role": position_role,
                    "group_by": group_by,
                    **summary,
                }
            )
        for payload in endpoint_payloads:
            actual_output_delta = payload.pop("_actual_output_delta")
            jvp_ablated = payload.pop("_jvp_ablated")
            jvp_clean = payload.pop("_jvp_clean")
            row = dict(payload)
            for group_by in group_by_values:
                subspace_key = (str(row["output_stage"]), str(row["position_role"]), group_by)
                if subspace_key not in subspaces:
                    raise KeyError(f"Missing contextual subspace for {subspace_key}.")
                subspace = subspaces[subspace_key]
                for vector_label, vector in [
                    ("actual", actual_output_delta),
                    ("jvp_ablated", jvp_ablated),
                    ("jvp_clean", jvp_clean),
                ]:
                    row[f"{vector_label}_{group_by}_mean_abs_cosine"] = _abs_cosine(
                        vector,
                        subspace["mean_direction"],
                        label=f"{source_component}/{row['component']}/{step}/{row['pair_id']}/{vector_label}/{group_by}.mean",
                    )
                    row[f"{vector_label}_{group_by}_identity_overlap"] = _subspace_overlap(
                        vector,
                        subspace["identity_basis"],
                        label=f"{source_component}/{row['component']}/{step}/{row['pair_id']}/{vector_label}/{group_by}.identity",
                    )
                    row[f"{vector_label}_{group_by}_all_vector_overlap"] = _subspace_overlap(
                        vector,
                        subspace["all_vector_basis"],
                        label=f"{source_component}/{row['component']}/{step}/{row['pair_id']}/{vector_label}/{group_by}.all_vector",
                    )
            local_rows.append(row)
        expected = len(endpoint_required_pair_ids) * len(components) * len(position_roles)
        actual = len(endpoint_payloads)
        if actual != expected:
            raise RuntimeError(
                f"MLP local write-map payload count mismatch for {source_component} "
                f"endpoint={step}/{margin_side}/{endpoint_kind}: expected={expected} got={actual}"
            )
    return local_rows, subspace_rows


def run_mlp_local_write_map_report(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    device_name: str,
    pair_types: list[str],
    source_components: list[str],
    components: list[str],
    position_roles: list[str],
    group_by_values: list[str],
    scalar_names: list[str] | None,
    margin_sides: list[str] | None,
    endpoint_roles: list[str] | None,
    split_filter: list[str] | None,
    max_pairs_per_type: int,
    min_pairs_per_type: int,
    pca_rank: int,
    markdown_top_k_rows: int,
    plot_top_k_rows: int,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if pca_rank <= 0:
        raise ValueError(f"pca_rank must be positive, got {pca_rank}.")
    resolved_margin_sides = _resolve_unique_values(
        values=margin_sides,
        default_values=["clean"],
        allowed_values=OUTPUT_ROUTE_MARGIN_SIDES,
        label="margin side",
    )
    resolved_scalars = _resolve_unique_values(
        values=scalar_names,
        default_values=["correct_value_logit"],
        allowed_values=OUTPUT_ROUTE_SCALARS,
        label="scalar",
    )
    resolved_endpoint_roles = _resolve_unique_values(
        values=endpoint_roles,
        default_values=["source", "target"],
        allowed_values=RESIDUAL_STATE_RESCUE_ENDPOINT_ROLES,
        label="endpoint role",
    )
    resolved_position_roles = _resolve_unique_values(
        values=position_roles,
        default_values=[],
        allowed_values=GEOMETRY_POSITION_ROLES,
        label="position role",
    )
    resolved_group_by = _resolve_unique_values(
        values=group_by_values,
        default_values=[],
        allowed_values=CONTEXTUAL_GROUP_BY_OPTIONS,
        label="group by",
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
    resolved_components = _filter_component_labels(
        requested_components=components,
        available_components=available_components,
    )
    _validate_mlp_components(components=resolved_components)
    _validate_maskable_components(
        components=resolved_sources,
        num_layers=len(model.blocks),
        num_heads=model.spec.n_heads,
        device=device,
    )
    _validate_downstream_components(source_components=resolved_sources, patch_components=resolved_components)

    scalar_pair_rows = _filter_scalar_pair_rows_for_delta(
        rows=_load_scalar_pair_rows(scalar_pair_rows_path),
        margin_sides=resolved_margin_sides,
        pair_types=pair_types,
        scalar_names=resolved_scalars,
    )
    checkpoint_paths_by_step = _checkpoint_paths_by_step(scalar_pair_rows)
    endpoint_pair_ids = _endpoint_pair_ids(
        scalar_pair_rows=scalar_pair_rows,
        endpoint_roles=resolved_endpoint_roles,
    )
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

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[mlp-local-write-map-report] "
        f"pairs={len(pairs_by_id)} scalar_rows={len(scalar_pair_rows)} endpoints={len(endpoint_pair_ids)} "
        f"sources={resolved_sources} components={resolved_components} position_roles={resolved_position_roles} "
        f"group_by={resolved_group_by} pca_rank={pca_rank} device={device_name}",
        flush=True,
    )
    local_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    for source_component in resolved_sources:
        print(
            f"[mlp-local-write-map-report] source={source_component} computing MLP local Jacobian-vector maps",
            flush=True,
        )
        source_local_rows, source_subspace_rows = _compute_mlp_local_rows_for_source(
            model=model,
            checkpoint_paths_by_step=checkpoint_paths_by_step,
            pairs_by_id=pairs_by_id,
            endpoint_pair_ids=endpoint_pair_ids,
            source_component=source_component,
            components=resolved_components,
            position_roles=resolved_position_roles,
            group_by_values=resolved_group_by,
            pca_rank=pca_rank,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            vocab=vocab,
            device=device,
        )
        local_rows.extend(source_local_rows)
        subspace_rows.extend(source_subspace_rows)
    summary_rows = _summarize_local_rows(rows=local_rows, group_by_values=resolved_group_by)

    local_rows_path = output_dir / "mlp_local_write_map_rows.jsonl"
    summary_rows_path = output_dir / "mlp_local_write_map_summary_rows.jsonl"
    subspace_rows_path = output_dir / "mlp_local_write_map_subspaces.jsonl"
    pair_rows_path = output_dir / "mlp_local_write_map_pairs.jsonl"
    report_path = output_dir / "mlp_local_write_map_report.json"
    markdown_path = output_dir / "mlp_local_write_map_report.md"
    write_jsonl(local_rows_path, local_rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(subspace_rows_path, subspace_rows)
    write_jsonl(
        pair_rows_path,
        [{key: value for key, value in pair.items() if key not in {"clean_record", "corrupted_record"}} for pair in pairs_by_id.values()],
    )
    plot_paths: dict[str, Path] = {}
    summary_plot = _plot_local_summary(
        summary_rows=summary_rows,
        output_path=output_dir / "mlp_local_write_map_summary.svg",
        top_k_rows=plot_top_k_rows,
    )
    if summary_plot is not None:
        plot_paths["summary"] = summary_plot

    report = {
        "schema_version": MLP_LOCAL_WRITE_MAP_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "device": device_name,
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names_required_in_rows": resolved_scalars,
        "endpoint_roles": resolved_endpoint_roles,
        "source_components": resolved_sources,
        "components": resolved_components,
        "input_stages": {component: _mlp_input_stage(component) for component in resolved_components},
        "output_stages": {component: _component_patch_stage(component) for component in resolved_components},
        "position_roles": resolved_position_roles,
        "group_by": resolved_group_by,
        "pca_rank": pca_rank,
        "markdown_top_k_rows": markdown_top_k_rows,
        "plot_top_k_rows": plot_top_k_rows,
        "checkpoint_paths_by_step": {str(step): str(path) for step, path in checkpoint_paths_by_step.items()},
        "pair_construction": pair_construction,
        "calculation": {
            "mlp_function": "F_i(z) = MLP_i(LN_2(z))",
            "input_delta": "z_clean - z_source_component_ablated at the selected MLP input residual stage",
            "actual_output_delta": "F_i(z_clean) - F_i(z_source_component_ablated)",
            "jvp_ablated": "J_F_i(z_source_component_ablated) @ input_delta",
            "jvp_clean": "J_F_i(z_clean) @ input_delta",
            "identity_overlap": "norm of vector projected into PCA subspace of per-token clean contextual means at the MLP output stage",
        },
        "local_rows_path": str(local_rows_path),
        "summary_rows_path": str(summary_rows_path),
        "subspace_rows_path": str(subspace_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    print(f"[mlp-local-write-map-report] complete report={report_path} rows={local_rows_path}", flush=True)
    return report_path, markdown_path, local_rows_path, summary_rows_path, subspace_rows_path, pair_rows_path, plot_paths
