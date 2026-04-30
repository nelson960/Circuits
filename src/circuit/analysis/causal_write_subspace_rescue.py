from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import shutil
from typing import Any

import torch

from circuit.analysis.component_output_rescue import (
    _component_order,
    _component_groups_by_order,
    _component_patch_stage,
    _component_write,
    _parse_patch_group,
    _patch_group_id,
    _validate_downstream_components,
)
from circuit.analysis.contextual_svd_alignment import CONTEXTUAL_GROUP_BY_OPTIONS
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
from circuit.analysis.residual_delta_vector_report import (
    _group_token_id,
    _token_label,
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


CAUSAL_WRITE_SUBSPACE_RESCUE_SCHEMA_VERSION = 1
CAUSAL_WRITE_BASIS_KINDS = ["all_delta_pca", "identity_delta_pca"]


def _parse_position_group(group_spec: str) -> list[str]:
    roles = [role.strip() for role in group_spec.split(",") if role.strip()]
    if not roles:
        raise ValueError(f"Position group is empty: {group_spec!r}")
    unique_roles: list[str] = []
    for role in roles:
        if role not in unique_roles:
            unique_roles.append(role)
    return unique_roles


def _position_group_id(position_roles: list[str]) -> str:
    if not position_roles:
        raise ValueError("Cannot build position group id for an empty role list.")
    return "+".join(position_roles)


def _build_component_groups(
    *,
    components: list[str] | None,
    component_groups: list[str] | None,
    available_components: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    raw_groups: list[list[str]] = []
    if components is not None:
        for component in components:
            raw_groups.append([component])
    if component_groups is not None:
        for group_spec in component_groups:
            raw_groups.append(_parse_patch_group(group_spec))
    if not raw_groups:
        raise ValueError("At least one --component or --component-group is required.")

    resolved_groups: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    resolved_components: list[str] = []
    for raw_group in raw_groups:
        group_components = _filter_component_labels(
            requested_components=raw_group,
            available_components=available_components,
        )
        group_id = _patch_group_id(group_components)
        if group_id in seen_group_ids:
            continue
        seen_group_ids.add(group_id)
        for component in group_components:
            if component not in resolved_components:
                resolved_components.append(component)
        resolved_groups.append(
            {
                "component_group_id": group_id,
                "components": group_components,
                "component_stages": sorted({_component_patch_stage(component) for component in group_components}),
            }
        )
    if not resolved_groups:
        raise RuntimeError("Component group resolution produced no groups.")
    return resolved_components, resolved_groups


def _build_position_groups(
    *,
    position_roles: list[str] | None,
    position_groups: list[str] | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    raw_groups: list[list[str]] = []
    if position_roles is not None:
        for role in position_roles:
            raw_groups.append([role])
    if position_groups is not None:
        for group_spec in position_groups:
            raw_groups.append(_parse_position_group(group_spec))
    if not raw_groups:
        raise ValueError("At least one --position-role or --position-group is required.")

    resolved_roles: list[str] = []
    resolved_groups: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    for raw_group in raw_groups:
        unsupported_roles = [role for role in raw_group if role not in GEOMETRY_POSITION_ROLES]
        if unsupported_roles:
            raise ValueError(f"Unsupported position roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
        group_roles: list[str] = []
        for role in raw_group:
            if role not in group_roles:
                group_roles.append(role)
            if role not in resolved_roles:
                resolved_roles.append(role)
        group_id = _position_group_id(group_roles)
        if group_id in seen_group_ids:
            continue
        seen_group_ids.add(group_id)
        resolved_groups.append({"position_group_id": group_id, "position_roles": group_roles})
    if not resolved_groups:
        raise RuntimeError("Position group resolution produced no groups.")
    return resolved_roles, resolved_groups


def _pca_basis(
    *,
    vectors: torch.Tensor,
    rank: int,
    label: str,
) -> tuple[torch.Tensor, list[float], int]:
    if rank <= 0:
        raise ValueError(f"rank must be positive for {label}, got {rank}.")
    if vectors.ndim != 2:
        raise ValueError(f"{label} vectors must be rank-2, got shape {tuple(vectors.shape)}.")
    if vectors.size(0) < rank + 1:
        raise ValueError(f"{label} PCA rank {rank} requires at least {rank + 1} vectors, got {vectors.size(0)}.")
    centered = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    matrix_rank = int(torch.linalg.matrix_rank(centered).item())
    if matrix_rank < rank:
        raise RuntimeError(f"{label} centered rank {matrix_rank} is below requested PCA rank {rank}.")
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:rank, :].T.contiguous(), [float(value) for value in singular_values[:rank].tolist()], matrix_rank


def _project_vector(*, vector: torch.Tensor, basis: torch.Tensor, label: str) -> torch.Tensor:
    if vector.ndim != 1:
        raise ValueError(f"{label} vector must be rank-1, got shape {tuple(vector.shape)}.")
    if basis.ndim != 2:
        raise ValueError(f"{label} basis must be rank-2, got shape {tuple(basis.shape)}.")
    if vector.numel() != basis.size(0):
        raise ValueError(f"{label} dimension mismatch: vector={vector.numel()} basis_rows={basis.size(0)}.")
    return basis.matmul(basis.T.matmul(vector.float()))


def _build_basis_payloads(
    *,
    delta_vectors: list[torch.Tensor],
    delta_vectors_by_token: dict[int, list[torch.Tensor]],
    ranks: list[int],
    basis_kinds: list[str],
    label: str,
) -> tuple[dict[tuple[str, int], torch.Tensor], list[dict[str, Any]]]:
    if not delta_vectors:
        raise RuntimeError(f"No delta vectors collected for {label}.")
    all_vectors = torch.stack(delta_vectors, dim=0).float()
    basis_by_key: dict[tuple[str, int], torch.Tensor] = {}
    rows: list[dict[str, Any]] = []
    for basis_kind in basis_kinds:
        if basis_kind == "all_delta_pca":
            source_vectors = all_vectors
            num_groups = None
        elif basis_kind == "identity_delta_pca":
            if not delta_vectors_by_token:
                raise RuntimeError(f"No token-grouped delta vectors collected for {label}.")
            token_ids = sorted(delta_vectors_by_token)
            source_vectors = torch.stack(
                [
                    torch.stack(delta_vectors_by_token[token_id], dim=0).float().mean(dim=0)
                    for token_id in token_ids
                ],
                dim=0,
            )
            num_groups = len(token_ids)
        else:
            raise ValueError(f"Unsupported basis kind {basis_kind!r}; expected one of {CAUSAL_WRITE_BASIS_KINDS}.")
        for rank in ranks:
            basis, singular_values, centered_rank = _pca_basis(
                vectors=source_vectors,
                rank=rank,
                label=f"{label}/{basis_kind}/rank{rank}",
            )
            basis_by_key[(basis_kind, rank)] = basis
            rows.append(
                {
                    "basis_kind": basis_kind,
                    "subspace_rank": rank,
                    "num_vectors": int(source_vectors.size(0)),
                    "num_token_groups": num_groups,
                    "centered_rank": centered_rank,
                    "singular_values": singular_values,
                    "top_singular_value": singular_values[0],
                    "singular_value_sum": float(sum(singular_values)),
                }
            )
    return basis_by_key, rows


def _summarize_rows(
    *,
    rows: list[dict[str, Any]],
    denominator_threshold: float,
) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError("No causal write subspace rescue rows to summarize.")
    if denominator_threshold < 0.0:
        raise ValueError("denominator_threshold must be non-negative.")
    grouped: dict[tuple[str, str, str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["scalar_name"]),
                str(row["endpoint_kind"]),
                str(row["source_component"]),
                str(row["component_group_id"]),
                str(row["position_group_id"]),
                str(row["basis_kind"]),
                int(row["subspace_rank"]),
            )
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (
        scalar_name,
        endpoint_kind,
        source_component,
        component_group_id,
        position_group_id,
        basis_kind,
        subspace_rank,
    ), group in sorted(grouped.items()):
        total_drops = [float(row["total_drop"]) for row in group]
        rescues = [float(row["rescue"]) for row in group]
        unrecovered = [float(row["unrecovered"]) for row in group]
        projection_fractions = [
            float(row["projection_fraction"])
            for row in group
            if row["projection_fraction"] is not None
        ]
        recovery_fraction_rows = [
            float(row["rescue"]) / float(row["total_drop"])
            for row in group
            if abs(float(row["total_drop"])) > denominator_threshold
        ]
        label_id = f"{component_group_id}/{position_group_id}"
        mean_total_drop = _mean(
            total_drops,
            label=f"{scalar_name}/{endpoint_kind}/{source_component}/{label_id}/{basis_kind}/drop",
        )
        mean_rescue = _mean(
            rescues,
            label=f"{scalar_name}/{endpoint_kind}/{source_component}/{label_id}/{basis_kind}/rescue",
        )
        summary_rows.append(
            {
                "scalar_name": scalar_name,
                "endpoint_kind": endpoint_kind,
                "source_component": source_component,
                "component_group_id": component_group_id,
                "components": list(group[0]["components"]),
                "component_stages": list(group[0]["component_stages"]),
                "position_group_id": position_group_id,
                "position_roles": list(group[0]["position_roles"]),
                "group_by": str(group[0]["group_by"]),
                "basis_kind": basis_kind,
                "subspace_rank": subspace_rank,
                "num_observations": len(group),
                "mean_clean_scalar": _mean(
                    [float(row["clean_scalar"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{label_id}/{basis_kind}/clean",
                ),
                "mean_source_ablated_scalar": _mean(
                    [float(row["source_ablated_scalar"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{label_id}/{basis_kind}/ablated",
                ),
                "mean_patched_scalar": _mean(
                    [float(row["patched_scalar"]) for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{label_id}/{basis_kind}/patched",
                ),
                "mean_total_drop": mean_total_drop,
                "mean_abs_total_drop": _mean(
                    [abs(value) for value in total_drops],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{label_id}/{basis_kind}/abs_drop",
                ),
                "mean_rescue": mean_rescue,
                "mean_abs_rescue": _mean(
                    [abs(value) for value in rescues],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{label_id}/{basis_kind}/abs_rescue",
                ),
                "mean_unrecovered": _mean(
                    unrecovered,
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{label_id}/{basis_kind}/unrecovered",
                ),
                "mean_abs_unrecovered": _mean(
                    [abs(value) for value in unrecovered],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{label_id}/{basis_kind}/abs_unrecovered",
                ),
                "mean_projection_fraction": None
                if not projection_fractions
                else _mean(
                    projection_fractions,
                    label=f"{source_component}/{label_id}/{basis_kind}/projection_fraction",
                ),
                "mean_rescue_fraction_from_means": None
                if abs(mean_total_drop) <= denominator_threshold
                else float(mean_rescue / mean_total_drop),
                "num_recovery_fraction_rows": len(recovery_fraction_rows),
                "mean_recovery_fraction_per_row": None
                if not recovery_fraction_rows
                else _mean(
                    recovery_fraction_rows,
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{label_id}/{basis_kind}/fraction_rows",
                ),
                "improved_fraction": _mean(
                    [1.0 if bool(row["improved_by_patch"]) else 0.0 for row in group],
                    label=f"{scalar_name}/{endpoint_kind}/{source_component}/{label_id}/{basis_kind}/improved",
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
    return summary_rows


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    rows = sorted(
        report["summary_rows"],
        key=lambda row: (
            str(row["scalar_name"]),
            str(row["endpoint_kind"]),
            str(row["component_group_id"]),
            str(row["position_group_id"]),
            str(row["basis_kind"]),
            int(row["subspace_rank"]),
        ),
    )[: int(report["markdown_top_k_rows"])]
    lines = [
        "# Causal Write Subspace Rescue",
        "",
        "This report tests whether a low-dimensional subspace of a downstream component's source-caused write delta is sufficient to rescue a source-component ablation.",
        "",
        "Definitions:",
        "",
        "- `component_delta = component_write(clean) - component_write(source_ablated)`",
        "- `all_delta_pca`: PCA basis over all component delta vectors in the selected endpoint slice",
        "- `identity_delta_pca`: PCA basis over token-mean component delta vectors grouped by the selected semantic label",
        "- `projected_delta = P_subspace component_delta`",
        "- component groups and position groups are patched together in one intervention",
        "- `rescue = scalar(source ablated + projected_delta patch) - scalar(source ablated)`",
        "",
        "| scalar | endpoint | source | component group | position group | basis | rank | observations | damage | rescue | rescue fraction | projection fraction | abs unrecovered | improved | corr | R squared |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        rescue_fraction = row["mean_rescue_fraction_from_means"]
        projection_fraction = row["mean_projection_fraction"]
        corr = row["rescue_vs_drop_correlation"]
        r_squared = row["rescue_vs_drop_r_squared"]
        lines.append(
            "| {scalar} | {endpoint} | `{source}` | `{component}` | `{position}` | `{basis}` | {rank} | {n} | {damage:.6g} | {rescue:.6g} | {fraction} | {projection} | {unrecovered:.6g} | {improved:.3f} | {corr} | {r2} |".format(
                scalar=row["scalar_name"],
                endpoint=row["endpoint_kind"],
                source=row["source_component"],
                component=row["component_group_id"],
                position=row["position_group_id"],
                basis=row["basis_kind"],
                rank=int(row["subspace_rank"]),
                n=int(row["num_observations"]),
                damage=float(row["mean_total_drop"]),
                rescue=float(row["mean_rescue"]),
                fraction="" if rescue_fraction is None else f"{float(rescue_fraction):.3f}",
                projection="" if projection_fraction is None else f"{float(projection_fraction):.3f}",
                unrecovered=float(row["mean_abs_unrecovered"]),
                improved=float(row["improved_fraction"]),
                corr="" if corr is None else f"{float(corr):.3f}",
                r2="" if r_squared is None or not math.isfinite(float(r_squared)) else f"{float(r_squared):.3f}",
            )
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- rescue rows: `{report['rescue_rows_path']}`",
            f"- summary rows: `{report['summary_rows_path']}`",
            f"- subspace rows: `{report['subspace_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_causal_write_subspace_rows_for_source(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    endpoint_keys: set[tuple[int, str]],
    endpoint_requests: list[dict[str, Any]],
    source_component: str,
    components: list[str],
    component_groups: list[dict[str, Any]],
    position_roles: list[str],
    position_groups: list[dict[str, Any]],
    group_by: str,
    basis_kinds: list[str],
    subspace_ranks: list[int],
    batch_size: int,
    pad_token_id: int,
    vocab: Vocabulary,
    scalar_value_tolerance: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    num_layers = len(model.blocks)
    num_heads = int(model.spec.n_heads)
    source_mask_kwargs = _component_mask_kwargs(
        component=source_component,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
    )
    component_stage = {component: _component_patch_stage(component) for component in components}
    request_specs: dict[tuple[int, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    request_by_id: dict[tuple[int, int, str, str, str, str], dict[str, Any]] = {}
    for request in endpoint_requests:
        key = (int(request["step"]), str(request["margin_side"]))
        request_specs[key][str(request["pair_id"])].append(request)
        request_by_id[request["request_id"]] = request

    pair_ids = sorted(pairs_by_id)
    rescue_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    for step, margin_side in sorted(endpoint_keys):
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
        specs_for_endpoint = request_specs[(step, margin_side)]
        required_pair_ids = {pair_id for pair_id in pair_ids if pair_id in specs_for_endpoint}
        if not required_pair_ids:
            continue

        delta_vectors: dict[tuple[str, str], list[torch.Tensor]] = {
            (component, position_role): []
            for component in components
            for position_role in position_roles
        }
        delta_vectors_by_token: dict[tuple[str, str], dict[int, list[torch.Tensor]]] = {
            (component, position_role): defaultdict(list)
            for component in components
            for position_role in position_roles
        }
        for start_index in range(0, len(pair_ids), batch_size):
            batch_pair_ids = pair_ids[start_index : start_index + batch_size]
            active_pair_ids = [pair_id for pair_id in batch_pair_ids if pair_id in required_pair_ids]
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
                raise RuntimeError("causal-write-subspace-rescue requires residual streams.")
            _, answer_targets, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            _, _, ablated_metadata = extract_answer_logits(ablated_outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=clean_metadata, label="causal write subspace clean")
            _validate_single_query_batch(batch=batch, metadata=ablated_metadata, label="causal write subspace ablated")
            clean_component_writes = {
                component: _component_write(
                    model=model,
                    component=component,
                    residual_streams=clean_outputs.residual_streams,
                    attention_mask=batch["attention_mask"],
                ).detach().float().cpu()
                for component in components
            }
            ablated_component_writes = {
                component: _component_write(
                    model=model,
                    component=component,
                    residual_streams=ablated_outputs.residual_streams,
                    attention_mask=batch["attention_mask"],
                ).detach().float().cpu()
                for component in components
            }
            for flat_index in range(int(clean_metadata["rows"].numel())):
                query_batch_row = int(clean_metadata["rows"][flat_index].item())
                pair_id = str(batch_pair_ids[query_batch_row])
                if pair_id not in required_pair_ids:
                    continue
                prediction_position = int(clean_metadata["prediction_positions"][flat_index].item())
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
                        for position in positions:
                            clean_vector = clean_component_writes[component][context_batch_row, int(position), :].clone()
                            ablated_vector = ablated_component_writes[component][context_batch_row, int(position), :].clone()
                            delta = clean_vector - ablated_vector
                            group_token_id = _group_token_id(
                                group_by=group_by,
                                batch=batch,
                                metadata=clean_metadata,
                                answer_targets=answer_targets,
                                flat_index=flat_index,
                                context_batch_row=context_batch_row,
                                context_position=int(position),
                            )
                            delta_vectors[(component, position_role)].append(delta)
                            delta_vectors_by_token[(component, position_role)][group_token_id].append(delta)
                            if group_by == "answer_value" and group_token_id != answer_value_token_id:
                                raise RuntimeError(
                                    f"answer_value group mismatch for pair={pair_id}: group={group_token_id} answer={answer_value_token_id}"
                                )

        basis_by_component_role: dict[tuple[str, str], dict[tuple[str, int], torch.Tensor]] = {}
        for component in components:
            for position_role in position_roles:
                key = (component, position_role)
                basis_payloads, basis_rows = _build_basis_payloads(
                    delta_vectors=delta_vectors[key],
                    delta_vectors_by_token=delta_vectors_by_token[key],
                    ranks=subspace_ranks,
                    basis_kinds=basis_kinds,
                    label=f"{source_component}/{component}/{step}/{margin_side}/{position_role}/{group_by}",
                )
                basis_by_component_role[key] = basis_payloads
                for row in basis_rows:
                    subspace_rows.append(
                        {
                            "source_component": source_component,
                            "component": component,
                            "component_order": _component_order(component),
                            "component_stage": component_stage[component],
                            "step": step,
                            "checkpoint": str(checkpoint_path),
                            "margin_side": margin_side,
                            "position_role": position_role,
                            "group_by": group_by,
                            **row,
                        }
                    )

        for start_index in range(0, len(pair_ids), batch_size):
            batch_pair_ids = pair_ids[start_index : start_index + batch_size]
            active_pair_ids = [pair_id for pair_id in batch_pair_ids if pair_id in required_pair_ids]
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
                raise RuntimeError("causal-write-subspace-rescue requires residual streams for patch phase.")
            clean_logits, clean_targets, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            ablated_logits, ablated_targets, ablated_metadata = extract_answer_logits(ablated_outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=clean_metadata, label="causal write subspace patch clean")
            _validate_single_query_batch(batch=batch, metadata=ablated_metadata, label="causal write subspace patch ablated")
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
                label="causal write subspace clean",
            )
            ablated_values = _compute_scalar_payloads(
                answer_logits=ablated_logits,
                answer_targets=ablated_targets,
                metadata=ablated_metadata,
                batch_pair_ids=batch_pair_ids,
                request_specs_by_pair_id=request_specs_by_pair_id,
                label="causal write subspace ablated",
            )
            clean_component_writes = {
                component: _component_write(
                    model=model,
                    component=component,
                    residual_streams=clean_outputs.residual_streams,
                    attention_mask=batch["attention_mask"],
                ).detach().float().cpu()
                for component in components
            }
            patched_values: dict[tuple[str, str, str, int], dict[tuple[int, int, str, str, str, str], float]] = {}
            projection_stats: dict[tuple[str, str, str, int, str], tuple[float, float, float | None]] = {}
            for component_group in component_groups:
                component_group_id = str(component_group["component_group_id"])
                group_components = [str(component) for component in component_group["components"]]
                for position_group in position_groups:
                    position_group_id = str(position_group["position_group_id"])
                    group_position_roles = [str(role) for role in position_group["position_roles"]]
                    for basis_kind in basis_kinds:
                        for subspace_rank in subspace_ranks:
                            residual_patch: dict[str, torch.Tensor] = {}
                            pair_delta_norms: dict[str, list[float]] = defaultdict(list)
                            pair_projected_norms: dict[str, list[float]] = defaultdict(list)
                            for _, ordered_components in _component_groups_by_order(group_components):
                                with torch.no_grad():
                                    current_outputs = model(
                                        batch["input_ids"],
                                        attention_mask=batch["attention_mask"],
                                        return_residual_streams=True,
                                        residual_patch=residual_patch,
                                        **source_mask_kwargs,
                                    )
                                if current_outputs.residual_streams is None:
                                    raise RuntimeError("causal-write-subspace-rescue requires residual streams for grouped patch phase.")
                                current_component_writes = {
                                    component: _component_write(
                                        model=model,
                                        component=component,
                                        residual_streams=current_outputs.residual_streams,
                                        attention_mask=batch["attention_mask"],
                                    ).detach().float().cpu()
                                    for component in ordered_components
                                }
                                stage_deltas: dict[str, torch.Tensor] = {}
                                for component in ordered_components:
                                    stage = component_stage[component]
                                    if stage not in stage_deltas:
                                        stage_deltas[stage] = torch.zeros_like(
                                            current_outputs.residual_streams[stage]
                                        ).detach().float().cpu()
                                    seen_positions: set[tuple[int, int]] = set()
                                    for position_role in group_position_roles:
                                        basis = basis_by_component_role[(component, position_role)][(basis_kind, subspace_rank)]
                                        for flat_index in range(int(clean_metadata["rows"].numel())):
                                            query_batch_row = int(clean_metadata["rows"][flat_index].item())
                                            pair_id = str(batch_pair_ids[query_batch_row])
                                            if pair_id not in request_specs_by_pair_id:
                                                continue
                                            prediction_position = int(clean_metadata["prediction_positions"][flat_index].item())
                                            context_batch_row, positions = _attention_key_positions(
                                                batch=batch,
                                                metadata=clean_metadata,
                                                flat_index=flat_index,
                                                position_role=position_role,
                                                max_position=prediction_position,
                                            )
                                            for position in positions:
                                                position_key = (context_batch_row, int(position))
                                                if position_key in seen_positions:
                                                    raise RuntimeError(
                                                        "Position group contains overlapping positions for "
                                                        f"source={source_component} component={component} "
                                                        f"group={position_group_id} pair={pair_id} position={position_key}."
                                                    )
                                                seen_positions.add(position_key)
                                                delta = (
                                                    clean_component_writes[component][context_batch_row, int(position), :]
                                                    - current_component_writes[component][context_batch_row, int(position), :]
                                                )
                                                projected = _project_vector(
                                                    vector=delta,
                                                    basis=basis,
                                                    label=(
                                                        f"{source_component}/{component_group_id}/{position_group_id}/"
                                                        f"{component}/{step}/{pair_id}/{basis_kind}/rank{subspace_rank}"
                                                    ),
                                                )
                                                stage_deltas[stage][context_batch_row, int(position), :] += projected
                                                pair_delta_norms[pair_id].append(float(delta.norm().item()))
                                                pair_projected_norms[pair_id].append(float(projected.norm().item()))
                                for stage, stage_delta in stage_deltas.items():
                                    residual_patch[stage] = current_outputs.residual_streams[stage].detach().float().cpu().to(device) + stage_delta.to(device)
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
                                label=(
                                    f"causal write subspace patched {component_group_id}/"
                                    f"{position_group_id}/{basis_kind}/rank{subspace_rank}"
                                ),
                            )
                            patched_values[(component_group_id, position_group_id, basis_kind, subspace_rank)] = _compute_scalar_payloads(
                                answer_logits=patched_logits,
                                answer_targets=patched_targets,
                                metadata=patched_metadata,
                                batch_pair_ids=batch_pair_ids,
                                request_specs_by_pair_id=request_specs_by_pair_id,
                                label=(
                                    f"causal write subspace patched {component_group_id}/"
                                    f"{position_group_id}/{basis_kind}/rank{subspace_rank}"
                                ),
                            )
                            for pair_id in request_specs_by_pair_id:
                                delta_norm = _mean(
                                    pair_delta_norms[pair_id],
                                    label=f"{pair_id}/{component_group_id}/{position_group_id}/{basis_kind}/rank{subspace_rank}/delta_norm",
                                )
                                projected_norm = _mean(
                                    pair_projected_norms[pair_id],
                                    label=f"{pair_id}/{component_group_id}/{position_group_id}/{basis_kind}/rank{subspace_rank}/projected_norm",
                                )
                                projection_stats[(component_group_id, position_group_id, basis_kind, subspace_rank, pair_id)] = (
                                    delta_norm,
                                    projected_norm,
                                    None if delta_norm == 0.0 else projected_norm / delta_norm,
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
                pair_id = str(request["pair_id"])
                for component_group in component_groups:
                    component_group_id = str(component_group["component_group_id"])
                    group_components = [str(component) for component in component_group["components"]]
                    group_stages = [str(stage) for stage in component_group["component_stages"]]
                    for position_group in position_groups:
                        position_group_id = str(position_group["position_group_id"])
                        group_position_roles = [str(role) for role in position_group["position_roles"]]
                        for basis_kind in basis_kinds:
                            for subspace_rank in subspace_ranks:
                                patched_by_request = patched_values[(component_group_id, position_group_id, basis_kind, subspace_rank)]
                                if request_id not in patched_by_request:
                                    raise KeyError(
                                        f"Missing patched scalar for request {request_id}: "
                                        f"component_group={component_group_id} position_group={position_group_id} "
                                        f"basis={basis_kind} rank={subspace_rank}"
                                    )
                                patched_scalar = patched_by_request[request_id]
                                rescue = patched_scalar - ablated_scalar
                                unrecovered = clean_scalar - patched_scalar
                                delta_norm, projected_norm, projection_fraction = projection_stats[
                                    (component_group_id, position_group_id, basis_kind, subspace_rank, pair_id)
                                ]
                                rescue_rows.append(
                                    {
                                        "source_step": int(request_id[0]),
                                        "target_step": int(request_id[1]),
                                        "endpoint_kind": str(request["endpoint_kind"]),
                                        "pair_id": pair_id,
                                        "pair_type": str(request["pair_type"]),
                                        "margin_side": str(request["margin_side"]),
                                        "scalar_name": str(request["scalar_name"]),
                                        "source_component": source_component,
                                        "component_group_id": component_group_id,
                                        "components": group_components,
                                        "component_stages": group_stages,
                                        "position_group_id": position_group_id,
                                        "position_roles": group_position_roles,
                                        "group_by": group_by,
                                        "basis_kind": basis_kind,
                                        "subspace_rank": subspace_rank,
                                        "clean_scalar": clean_scalar,
                                        "source_ablated_scalar": ablated_scalar,
                                        "patched_scalar": patched_scalar,
                                        "total_drop": total_drop,
                                        "rescue": rescue,
                                        "unrecovered": unrecovered,
                                        "rescue_fraction": None if abs(total_drop) <= 1.0e-12 else rescue / total_drop,
                                        "improved_by_patch": abs(unrecovered) < abs(total_drop),
                                        "mean_component_delta_norm": delta_norm,
                                        "mean_projected_delta_norm": projected_norm,
                                        "projection_fraction": projection_fraction,
                                    }
                                )
    return rescue_rows, subspace_rows


def run_causal_write_subspace_rescue(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    pair_types: list[str],
    source_components: list[str],
    components: list[str] | None,
    group_by: str,
    component_groups: list[str] | None = None,
    position_roles: list[str] | None = None,
    position_groups: list[str] | None = None,
    basis_kinds: list[str] | None = None,
    subspace_ranks: list[int] | None = None,
    device_name: str = "mps",
    scalar_names: list[str] | None = None,
    margin_sides: list[str] | None = None,
    endpoint_roles: list[str] | None = None,
    split_filter: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    scalar_value_tolerance: float = 1.0e-4,
    denominator_threshold: float = 1.0e-6,
    markdown_top_k_rows: int = 160,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    if not pair_types:
        raise ValueError("causal-write-subspace-rescue requires at least one pair type.")
    if not source_components:
        raise ValueError("source_components must not be empty.")
    resolved_basis_kinds = _resolve_unique_values(
        values=basis_kinds,
        default_values=CAUSAL_WRITE_BASIS_KINDS,
        allowed_values=CAUSAL_WRITE_BASIS_KINDS,
        label="basis kind",
    )
    resolved_ranks = sorted(set(subspace_ranks if subspace_ranks is not None else [1, 2, 4, 8]))
    if any(rank <= 0 for rank in resolved_ranks):
        raise ValueError(f"subspace ranks must be positive, got {resolved_ranks}.")
    if group_by not in CONTEXTUAL_GROUP_BY_OPTIONS:
        raise ValueError(f"Unsupported group_by {group_by!r}; expected one of {CONTEXTUAL_GROUP_BY_OPTIONS}.")
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
    resolved_components, resolved_component_groups = _build_component_groups(
        components=components,
        component_groups=component_groups,
        available_components=available_components,
    )
    resolved_position_roles, resolved_position_groups = _build_position_groups(
        position_roles=position_roles,
        position_groups=position_groups,
    )
    overlap = sorted(set(resolved_sources) & set(resolved_components))
    if overlap:
        raise ValueError(f"Source and downstream component sets must be disjoint; overlap={overlap}")
    _validate_maskable_components(
        components=resolved_sources,
        num_layers=len(model.blocks),
        num_heads=model.spec.n_heads,
        device=device,
    )
    _validate_maskable_components(
        components=resolved_components,
        num_layers=len(model.blocks),
        num_heads=model.spec.n_heads,
        device=device,
    )
    _validate_downstream_components(source_components=resolved_sources, patch_components=resolved_components)

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
        "[causal-write-subspace-rescue] "
        f"pairs={len(pairs_by_id)} scalar_rows={len(scalar_pair_rows)} endpoints={len(endpoint_keys)} "
        f"sources={resolved_sources} component_groups={resolved_component_groups} "
        f"position_groups={resolved_position_groups} "
        f"group_by={group_by} basis_kinds={resolved_basis_kinds} ranks={resolved_ranks} "
        f"scalars={resolved_scalars} endpoint_roles={resolved_endpoint_roles} device={device_name}",
        flush=True,
    )
    rescue_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    for source_component in resolved_sources:
        print(
            f"[causal-write-subspace-rescue] source={source_component} building subspaces and projected patches",
            flush=True,
        )
        source_rescue_rows, source_subspace_rows = _compute_causal_write_subspace_rows_for_source(
            model=model,
            checkpoint_paths_by_step=checkpoint_paths_by_step,
            pairs_by_id=pairs_by_id,
            endpoint_keys=endpoint_keys,
            endpoint_requests=endpoint_requests,
            source_component=source_component,
            components=resolved_components,
            component_groups=resolved_component_groups,
            position_roles=resolved_position_roles,
            position_groups=resolved_position_groups,
            group_by=group_by,
            basis_kinds=resolved_basis_kinds,
            subspace_ranks=resolved_ranks,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            vocab=vocab,
            scalar_value_tolerance=scalar_value_tolerance,
            device=device,
        )
        rescue_rows.extend(source_rescue_rows)
        subspace_rows.extend(source_subspace_rows)
    summary_rows = _summarize_rows(rows=rescue_rows, denominator_threshold=denominator_threshold)

    rescue_rows_path = output_dir / "causal_write_subspace_rescue_rows.jsonl"
    summary_rows_path = output_dir / "causal_write_subspace_rescue_summary_rows.jsonl"
    subspace_rows_path = output_dir / "causal_write_subspace_rescue_subspaces.jsonl"
    pair_rows_path = output_dir / "causal_write_subspace_rescue_pairs.jsonl"
    report_path = output_dir / "causal_write_subspace_rescue_report.json"
    markdown_path = output_dir / "causal_write_subspace_rescue_report.md"
    write_jsonl(rescue_rows_path, rescue_rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(subspace_rows_path, subspace_rows)
    write_jsonl(
        pair_rows_path,
        [{key: value for key, value in pair.items() if key not in {"clean_record", "corrupted_record"}} for pair in pairs_by_id.values()],
    )
    report = {
        "schema_version": CAUSAL_WRITE_SUBSPACE_RESCUE_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "device": device_name,
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names": resolved_scalars,
        "endpoint_roles": resolved_endpoint_roles,
        "source_components": resolved_sources,
        "components": resolved_components,
        "component_groups": resolved_component_groups,
        "position_roles": resolved_position_roles,
        "position_groups": resolved_position_groups,
        "group_by": group_by,
        "basis_kinds": resolved_basis_kinds,
        "subspace_ranks": resolved_ranks,
        "scalar_value_tolerance": scalar_value_tolerance,
        "denominator_threshold": denominator_threshold,
        "markdown_top_k_rows": markdown_top_k_rows,
        "checkpoint_paths_by_step": {str(step): str(path) for step, path in checkpoint_paths_by_step.items()},
        "pair_construction": pair_construction,
        "calculation": {
            "component_delta": "component_write(clean) - component_write(source_ablated)",
            "all_delta_pca": "PCA basis over all component_delta vectors in the selected endpoint slice",
            "identity_delta_pca": "PCA basis over token-mean component_delta vectors grouped by group_by",
            "projected_delta": "orthogonal projection of component_delta into the selected basis",
            "grouped_patch": "component groups are patched in component order; position groups are patched together",
            "rescue": "scalar(source ablated + grouped projected_delta patch) - scalar(source ablated)",
        },
        "rescue_rows_path": str(rescue_rows_path),
        "summary_rows_path": str(summary_rows_path),
        "subspace_rows_path": str(subspace_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(f"[causal-write-subspace-rescue] complete report={report_path} rows={rescue_rows_path}", flush=True)
    return report_path, markdown_path, rescue_rows_path, summary_rows_path, subspace_rows_path, pair_rows_path
