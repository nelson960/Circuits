from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import shutil
from typing import Any

import torch

from circuit.analysis.causal_write_subspace_rescue import (
    _build_component_groups,
    _build_position_groups,
    _project_vector,
)
from circuit.analysis.component_output_rescue import (
    _component_groups_by_order,
    _component_patch_stage,
    _component_write,
    _validate_downstream_components,
)
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import _attention_key_positions, _checkpoint_step_from_path, _validate_single_query_batch
from circuit.analysis.output_component_causal_validation import _component_mask_kwargs, _scalar_from_logits
from circuit.analysis.output_route_closure import (
    OUTPUT_ROUTE_MARGIN_SIDES,
    OUTPUT_ROUTE_SCALARS,
    _build_endpoint_requests,
    _checkpoint_paths_by_step,
    _component_labels,
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


CAUSAL_WRITE_GRADIENT_SUBSPACE_RESCUE_SCHEMA_VERSION = 1
CAUSAL_WRITE_GRADIENT_BASIS_KINDS = ["scalar_gradient_svd"]


def _gradient_svd_basis(
    *,
    vectors: torch.Tensor,
    rank: int,
    label: str,
) -> tuple[torch.Tensor, list[float], int, int, str, float]:
    if rank <= 0:
        raise ValueError(f"rank must be positive for {label}, got {rank}.")
    if vectors.ndim != 2:
        raise ValueError(f"{label} vectors must be rank-2, got shape {tuple(vectors.shape)}.")
    if vectors.size(0) == 0:
        raise ValueError(f"{label} requires at least one vector.")
    matrix = vectors.float()
    gradient_norms = matrix.norm(dim=1)
    mean_gradient_norm = float(gradient_norms.mean().item())
    matrix_rank = int(torch.linalg.matrix_rank(matrix).item())
    if matrix_rank == 0:
        basis = torch.zeros(matrix.size(1), rank, dtype=matrix.dtype)
        return basis, [0.0 for _ in range(rank)], matrix_rank, 0, "zero_gradient", mean_gradient_norm
    _, singular_values_tensor, vh = torch.linalg.svd(matrix, full_matrices=False)
    available_rank = min(rank, int(vh.size(0)), matrix_rank)
    if available_rank <= 0:
        raise RuntimeError(f"{label} produced no available singular vectors despite matrix_rank={matrix_rank}.")
    basis = vh[:available_rank, :].T.contiguous()
    singular_values = [float(value) for value in singular_values_tensor[:available_rank].tolist()]
    status = "full_rank"
    if available_rank < rank:
        padding = torch.zeros(matrix.size(1), rank - available_rank, dtype=basis.dtype)
        basis = torch.cat([basis, padding], dim=1)
        singular_values.extend([0.0 for _ in range(rank - available_rank)])
        status = "rank_deficient"
    return basis, singular_values, matrix_rank, available_rank, status, mean_gradient_norm


def _differentiable_scalar_from_logits(
    *,
    logits: torch.Tensor,
    scalar_name: str,
    answer_target_id: int,
    source_best_wrong_token_id: int,
    target_best_wrong_token_id: int,
    endpoint_kind: str,
) -> torch.Tensor:
    correct_logit = logits[answer_target_id]
    source_wrong_logit = logits[source_best_wrong_token_id]
    target_wrong_logit = logits[target_best_wrong_token_id]
    if scalar_name == "moving_answer_margin":
        wrong_logit = source_wrong_logit if endpoint_kind == "source" else target_wrong_logit
        return correct_logit - wrong_logit
    if scalar_name == "fixed_source_competitor_margin":
        return correct_logit - source_wrong_logit
    if scalar_name == "fixed_target_competitor_margin":
        return correct_logit - target_wrong_logit
    if scalar_name == "correct_value_logit":
        return correct_logit
    if scalar_name == "source_best_wrong_logit":
        return source_wrong_logit
    if scalar_name == "target_best_wrong_logit":
        return target_wrong_logit
    if scalar_name == "negative_answer_loss":
        return torch.log_softmax(logits, dim=-1)[answer_target_id]
    raise ValueError(f"Unsupported scalar {scalar_name!r}; expected one of {OUTPUT_ROUTE_SCALARS}.")


def _scalar_requests_for_pair(
    *,
    request_specs_by_pair_id: dict[str, list[dict[str, Any]]],
    pair_id: str,
    scalar_name: str,
    endpoint_kind: str,
) -> list[dict[str, Any]]:
    return [
        request
        for request in request_specs_by_pair_id.get(pair_id, [])
        if str(request["scalar_name"]) == scalar_name and str(request["endpoint_kind"]) == endpoint_kind
    ]


def _scalar_values_for_combo(
    *,
    answer_logits: torch.Tensor,
    answer_targets: torch.Tensor,
    batch_pair_ids: list[str],
    request_specs_by_pair_id: dict[str, list[dict[str, Any]]],
    scalar_name: str,
    endpoint_kind: str,
    label: str,
) -> tuple[torch.Tensor, list[int]]:
    scalar_values: list[torch.Tensor] = []
    item_indices: list[int] = []
    for item_index, pair_id in enumerate(batch_pair_ids):
        requests = _scalar_requests_for_pair(
            request_specs_by_pair_id=request_specs_by_pair_id,
            pair_id=pair_id,
            scalar_name=scalar_name,
            endpoint_kind=endpoint_kind,
        )
        if not requests:
            continue
        if len(requests) != 1:
            raise RuntimeError(
                f"{label} expected one request for pair={pair_id} scalar={scalar_name} endpoint={endpoint_kind}, "
                f"got {len(requests)}."
            )
        request = requests[0]
        expected_target = int(request["answer_target_id"])
        target_id = int(answer_targets[item_index].detach().cpu().item())
        if target_id != expected_target:
            raise RuntimeError(
                f"{label} answer target mismatch for pair={pair_id}: expected={expected_target} got={target_id}"
            )
        scalar_values.append(
            _differentiable_scalar_from_logits(
                logits=answer_logits[item_index],
                scalar_name=scalar_name,
                answer_target_id=expected_target,
                source_best_wrong_token_id=int(request["source_best_wrong_token_id"]),
                target_best_wrong_token_id=int(request["target_best_wrong_token_id"]),
                endpoint_kind=endpoint_kind,
            )
        )
        item_indices.append(item_index)
    if not scalar_values:
        return torch.empty(0, device=answer_logits.device), []
    return torch.stack(scalar_values), item_indices


def _request_specs_for_combo(
    *,
    specs_for_endpoint: dict[str, list[dict[str, Any]]],
    batch_pair_ids: list[str],
    scalar_name: str,
    endpoint_kind: str,
) -> dict[str, list[dict[str, Any]]]:
    filtered: dict[str, list[dict[str, Any]]] = {}
    for pair_id in batch_pair_ids:
        requests = _scalar_requests_for_pair(
            request_specs_by_pair_id=specs_for_endpoint,
            pair_id=pair_id,
            scalar_name=scalar_name,
            endpoint_kind=endpoint_kind,
        )
        if requests:
            filtered[pair_id] = requests
    return filtered


def _summarize_rows(*, rows: list[dict[str, Any]], denominator_threshold: float) -> list[dict[str, Any]]:
    if not rows:
        raise RuntimeError("No causal write gradient subspace rescue rows to summarize.")
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
        projection_fractions = [float(row["projection_fraction"]) for row in group if row["projection_fraction"] is not None]
        gradient_norms = [float(row["mean_gradient_norm"]) for row in group if row["mean_gradient_norm"] is not None]
        recovery_fraction_rows = [
            float(row["rescue"]) / float(row["total_drop"])
            for row in group
            if abs(float(row["total_drop"])) > denominator_threshold
        ]
        label = f"{scalar_name}/{endpoint_kind}/{source_component}/{component_group_id}/{position_group_id}/{basis_kind}"
        mean_total_drop = _mean(total_drops, label=f"{label}/drop")
        mean_rescue = _mean(rescues, label=f"{label}/rescue")
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
                "basis_kind": basis_kind,
                "subspace_rank": subspace_rank,
                "num_observations": len(group),
                "mean_clean_scalar": _mean([float(row["clean_scalar"]) for row in group], label=f"{label}/clean"),
                "mean_source_ablated_scalar": _mean(
                    [float(row["source_ablated_scalar"]) for row in group],
                    label=f"{label}/ablated",
                ),
                "mean_patched_scalar": _mean([float(row["patched_scalar"]) for row in group], label=f"{label}/patched"),
                "mean_total_drop": mean_total_drop,
                "mean_abs_total_drop": _mean([abs(value) for value in total_drops], label=f"{label}/abs_drop"),
                "mean_rescue": mean_rescue,
                "mean_abs_rescue": _mean([abs(value) for value in rescues], label=f"{label}/abs_rescue"),
                "mean_unrecovered": _mean(unrecovered, label=f"{label}/unrecovered"),
                "mean_abs_unrecovered": _mean([abs(value) for value in unrecovered], label=f"{label}/abs_unrecovered"),
                "mean_projection_fraction": None
                if not projection_fractions
                else _mean(projection_fractions, label=f"{label}/projection_fraction"),
                "mean_gradient_norm": None if not gradient_norms else _mean(gradient_norms, label=f"{label}/gradient_norm"),
                "mean_rescue_fraction_from_means": None
                if abs(mean_total_drop) <= denominator_threshold
                else float(mean_rescue / mean_total_drop),
                "num_recovery_fraction_rows": len(recovery_fraction_rows),
                "mean_recovery_fraction_per_row": None
                if not recovery_fraction_rows
                else _mean(recovery_fraction_rows, label=f"{label}/fraction_rows"),
                "improved_fraction": _mean(
                    [1.0 if bool(row["improved_by_patch"]) else 0.0 for row in group],
                    label=f"{label}/improved",
                ),
                "rescue_vs_drop_r_squared": _safe_r_squared(y_values=total_drops, predicted_values=rescues),
                "rescue_vs_drop_correlation": _safe_correlation(x_values=rescues, y_values=total_drops),
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
            int(row["subspace_rank"]),
        ),
    )[: int(report["markdown_top_k_rows"])]
    lines = [
        "# Causal Write Gradient Subspace Rescue",
        "",
        "This report tests whether scalar-gradient-selected write directions rescue a source-component ablation.",
        "",
        "Definitions:",
        "",
        "- `component_delta = component_write(clean) - component_write(current_source_ablated_state)`",
        "- `scalar_gradient_svd`: uncentered SVD basis over `d scalar / d residual_stage[position]` vectors",
        "- `projected_delta = P_gradient_subspace component_delta`",
        "- component groups are patched in component order; position groups are patched together",
        "",
        "| scalar | endpoint | source | component group | position group | rank | observations | damage | rescue | rescue fraction | projection fraction | grad norm | abs unrecovered | improved | corr | R squared |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        rescue_fraction = row["mean_rescue_fraction_from_means"]
        projection_fraction = row["mean_projection_fraction"]
        gradient_norm = row["mean_gradient_norm"]
        corr = row["rescue_vs_drop_correlation"]
        r_squared = row["rescue_vs_drop_r_squared"]
        lines.append(
            "| {scalar} | {endpoint} | `{source}` | `{component}` | `{position}` | {rank} | {n} | {damage:.6g} | {rescue:.6g} | {fraction} | {projection} | {grad_norm} | {unrecovered:.6g} | {improved:.3f} | {corr} | {r2} |".format(
                scalar=row["scalar_name"],
                endpoint=row["endpoint_kind"],
                source=row["source_component"],
                component=row["component_group_id"],
                position=row["position_group_id"],
                rank=int(row["subspace_rank"]),
                n=int(row["num_observations"]),
                damage=float(row["mean_total_drop"]),
                rescue=float(row["mean_rescue"]),
                fraction="" if rescue_fraction is None else f"{float(rescue_fraction):.3f}",
                projection="" if projection_fraction is None else f"{float(projection_fraction):.3f}",
                grad_norm="" if gradient_norm is None else f"{float(gradient_norm):.6g}",
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
            f"- basis rows: `{report['basis_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_gradient_bases_for_source(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    endpoint_keys: set[tuple[int, str]],
    endpoint_requests: list[dict[str, Any]],
    source_component: str,
    components: list[str],
    position_roles: list[str],
    scalar_names: list[str],
    endpoint_roles: list[str],
    subspace_ranks: list[int],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> tuple[dict[tuple[int, str, str, str, str, str, int], torch.Tensor], list[dict[str, Any]]]:
    num_layers = len(model.blocks)
    num_heads = int(model.spec.n_heads)
    source_mask_kwargs = _component_mask_kwargs(
        component=source_component,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
    )
    component_stage = {component: _component_patch_stage(component) for component in components}
    stages = sorted(set(component_stage.values()))
    request_specs: dict[tuple[int, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for request in endpoint_requests:
        request_specs[(int(request["step"]), str(request["margin_side"]))][str(request["pair_id"])].append(request)

    pair_ids = sorted(pairs_by_id)
    basis_by_key: dict[tuple[int, str, str, str, str, str, int], torch.Tensor] = {}
    basis_rows: list[dict[str, Any]] = []
    for step, margin_side in sorted(endpoint_keys):
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
        vectors: dict[tuple[str, str, str, str], list[torch.Tensor]] = {
            (component, position_role, scalar_name, endpoint_kind): []
            for component in components
            for position_role in position_roles
            for scalar_name in scalar_names
            for endpoint_kind in endpoint_roles
        }
        for start_index in range(0, len(pair_ids), batch_size):
            batch_pair_ids = pair_ids[start_index : start_index + batch_size]
            active_pair_ids = [pair_id for pair_id in batch_pair_ids if pair_id in required_pair_ids]
            if not active_pair_ids:
                continue
            records = [pairs_by_id[pair_id][side_key] for pair_id in batch_pair_ids]
            batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
            request_specs_by_pair_id = {
                pair_id: specs_for_endpoint[pair_id]
                for pair_id in batch_pair_ids
                if pair_id in specs_for_endpoint
            }
            for scalar_name in scalar_names:
                for endpoint_kind in endpoint_roles:
                    combo_specs = _request_specs_for_combo(
                        specs_for_endpoint=request_specs_by_pair_id,
                        batch_pair_ids=batch_pair_ids,
                        scalar_name=scalar_name,
                        endpoint_kind=endpoint_kind,
                    )
                    if not combo_specs:
                        continue
                    model.zero_grad(set_to_none=True)
                    outputs = model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        return_residual_streams=True,
                        **source_mask_kwargs,
                    )
                    if outputs.residual_streams is None:
                        raise RuntimeError("causal-write-gradient-subspace-rescue requires residual streams.")
                    answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
                    _validate_single_query_batch(batch=batch, metadata=metadata, label="causal write gradient basis")
                    scalar_values, item_indices = _scalar_values_for_combo(
                        answer_logits=answer_logits,
                        answer_targets=answer_targets,
                        batch_pair_ids=batch_pair_ids,
                        request_specs_by_pair_id=combo_specs,
                        scalar_name=scalar_name,
                        endpoint_kind=endpoint_kind,
                        label="causal write gradient basis",
                    )
                    if not item_indices:
                        continue
                    stage_tensors = [outputs.residual_streams[stage] for stage in stages]
                    gradients = torch.autograd.grad(scalar_values.sum(), stage_tensors)
                    gradients_by_stage = {stage: gradient.detach().float().cpu() for stage, gradient in zip(stages, gradients, strict=True)}
                    for item_index in item_indices:
                        prediction_position = int(metadata["prediction_positions"][item_index].item())
                        for position_role in position_roles:
                            context_batch_row, positions = _attention_key_positions(
                                batch=batch,
                                metadata=metadata,
                                flat_index=item_index,
                                position_role=position_role,
                                max_position=prediction_position,
                            )
                            for component in components:
                                gradient_stage = gradients_by_stage[component_stage[component]]
                                for position in positions:
                                    vectors[(component, position_role, scalar_name, endpoint_kind)].append(
                                        gradient_stage[context_batch_row, int(position), :].clone()
                                    )
                    model.zero_grad(set_to_none=True)
        for component in components:
            for position_role in position_roles:
                for scalar_name in scalar_names:
                    for endpoint_kind in endpoint_roles:
                        key = (component, position_role, scalar_name, endpoint_kind)
                        if not vectors[key]:
                            continue
                        for rank in subspace_ranks:
                            (
                                basis,
                                singular_values,
                                matrix_rank,
                                basis_effective_rank,
                                basis_status,
                                mean_gradient_norm,
                            ) = _gradient_svd_basis(
                                vectors=torch.stack(vectors[key], dim=0),
                                rank=rank,
                                label=f"{source_component}/{component}/{position_role}/{scalar_name}/{endpoint_kind}/{step}/gradient_svd",
                            )
                            basis_key = (step, margin_side, component, position_role, scalar_name, endpoint_kind, rank)
                            basis_by_key[basis_key] = basis
                            basis_rows.append(
                                {
                                    "source_component": source_component,
                                    "component": component,
                                    "component_stage": component_stage[component],
                                    "position_role": position_role,
                                    "scalar_name": scalar_name,
                                    "endpoint_kind": endpoint_kind,
                                    "step": step,
                                    "checkpoint": str(checkpoint_path),
                                    "margin_side": margin_side,
                                    "basis_kind": "scalar_gradient_svd",
                                    "subspace_rank": rank,
                                    "num_vectors": len(vectors[key]),
                                    "matrix_rank": matrix_rank,
                                    "basis_effective_rank": basis_effective_rank,
                                    "basis_status": basis_status,
                                    "singular_values": singular_values,
                                    "top_singular_value": singular_values[0],
                                    "singular_value_sum": float(sum(singular_values)),
                                    "mean_gradient_norm": mean_gradient_norm,
                                }
                            )
    return basis_by_key, basis_rows


def _compute_rescue_rows_for_source(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    endpoint_keys: set[tuple[int, str]],
    endpoint_requests: list[dict[str, Any]],
    source_component: str,
    components: list[str],
    component_groups: list[dict[str, Any]],
    position_groups: list[dict[str, Any]],
    basis_by_key: dict[tuple[int, str, str, str, str, str, int], torch.Tensor],
    subspace_ranks: list[int],
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
    component_stage = {component: _component_patch_stage(component) for component in components}
    request_specs: dict[tuple[int, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    request_by_id: dict[tuple[int, int, str, str, str, str], dict[str, Any]] = {}
    for request in endpoint_requests:
        request_specs[(int(request["step"]), str(request["margin_side"]))][str(request["pair_id"])].append(request)
        request_by_id[request["request_id"]] = request

    pair_ids = sorted(pairs_by_id)
    rows: list[dict[str, Any]] = []
    for step, margin_side in sorted(endpoint_keys):
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
        combos = sorted(
            {
                (str(request["scalar_name"]), str(request["endpoint_kind"]))
                for requests in specs_for_endpoint.values()
                for request in requests
            }
        )
        for start_index in range(0, len(pair_ids), batch_size):
            batch_pair_ids = pair_ids[start_index : start_index + batch_size]
            active_pair_ids = [pair_id for pair_id in batch_pair_ids if pair_id in required_pair_ids]
            if not active_pair_ids:
                continue
            records = [pairs_by_id[pair_id][side_key] for pair_id in batch_pair_ids]
            batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
            request_specs_by_pair_id = {
                pair_id: specs_for_endpoint[pair_id]
                for pair_id in batch_pair_ids
                if pair_id in specs_for_endpoint
            }
            if not request_specs_by_pair_id:
                continue
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
                raise RuntimeError("causal-write-gradient-subspace-rescue requires residual streams for patch phase.")
            clean_logits, clean_targets, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            ablated_logits, ablated_targets, ablated_metadata = extract_answer_logits(ablated_outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=clean_metadata, label="causal write gradient clean")
            _validate_single_query_batch(batch=batch, metadata=ablated_metadata, label="causal write gradient ablated")
            clean_values = _compute_scalar_payloads(
                answer_logits=clean_logits,
                answer_targets=clean_targets,
                metadata=clean_metadata,
                batch_pair_ids=batch_pair_ids,
                request_specs_by_pair_id=request_specs_by_pair_id,
                label="causal write gradient clean",
            )
            ablated_values = _compute_scalar_payloads(
                answer_logits=ablated_logits,
                answer_targets=ablated_targets,
                metadata=ablated_metadata,
                batch_pair_ids=batch_pair_ids,
                request_specs_by_pair_id=request_specs_by_pair_id,
                label="causal write gradient ablated",
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
            for scalar_name, endpoint_kind in combos:
                combo_specs = _request_specs_for_combo(
                    specs_for_endpoint=request_specs_by_pair_id,
                    batch_pair_ids=batch_pair_ids,
                    scalar_name=scalar_name,
                    endpoint_kind=endpoint_kind,
                )
                if not combo_specs:
                    continue
                for component_group in component_groups:
                    component_group_id = str(component_group["component_group_id"])
                    group_components = [str(component) for component in component_group["components"]]
                    group_stages = [str(stage) for stage in component_group["component_stages"]]
                    for position_group in position_groups:
                        position_group_id = str(position_group["position_group_id"])
                        group_position_roles = [str(role) for role in position_group["position_roles"]]
                        for subspace_rank in subspace_ranks:
                            residual_patch: dict[str, torch.Tensor] = {}
                            pair_delta_norms: dict[str, list[float]] = defaultdict(list)
                            pair_projected_norms: dict[str, list[float]] = defaultdict(list)
                            pair_gradient_norms: dict[str, list[float]] = defaultdict(list)
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
                                    raise RuntimeError("causal-write-gradient-subspace-rescue requires current residual streams.")
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
                                        stage_deltas[stage] = torch.zeros_like(current_outputs.residual_streams[stage]).detach().float().cpu()
                                    seen_positions: set[tuple[int, int]] = set()
                                    for position_role in group_position_roles:
                                        basis_key = (step, margin_side, component, position_role, scalar_name, endpoint_kind, subspace_rank)
                                        if basis_key not in basis_by_key:
                                            raise KeyError(f"Missing gradient basis for {basis_key}.")
                                        basis = basis_by_key[basis_key]
                                        for flat_index in range(int(clean_metadata["rows"].numel())):
                                            query_batch_row = int(clean_metadata["rows"][flat_index].item())
                                            pair_id = str(batch_pair_ids[query_batch_row])
                                            if pair_id not in combo_specs:
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
                                                        f"{component}/{step}/{pair_id}/{scalar_name}/{endpoint_kind}/rank{subspace_rank}"
                                                    ),
                                                )
                                                stage_deltas[stage][context_batch_row, int(position), :] += projected
                                                pair_delta_norms[pair_id].append(float(delta.norm().item()))
                                                pair_projected_norms[pair_id].append(float(projected.norm().item()))
                                                pair_gradient_norms[pair_id].append(float(basis.norm(dim=0).mean().item()))
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
                                    f"causal write gradient patched {component_group_id}/"
                                    f"{position_group_id}/{scalar_name}/{endpoint_kind}/rank{subspace_rank}"
                                ),
                            )
                            patched_values = _compute_scalar_payloads(
                                answer_logits=patched_logits,
                                answer_targets=patched_targets,
                                metadata=patched_metadata,
                                batch_pair_ids=batch_pair_ids,
                                request_specs_by_pair_id=combo_specs,
                                label=(
                                    f"causal write gradient patched {component_group_id}/"
                                    f"{position_group_id}/{scalar_name}/{endpoint_kind}/rank{subspace_rank}"
                                ),
                            )
                            for request_id, patched_scalar in patched_values.items():
                                request = request_by_id[request_id]
                                clean_scalar = clean_values[request_id]
                                scalar_payload = request.get("scalar_payload")
                                if scalar_payload is not None:
                                    expected = float(scalar_payload[str(request["endpoint_kind"])])
                                    delta = abs(clean_scalar - expected)
                                    if delta > scalar_value_tolerance:
                                        raise RuntimeError(
                                            f"Clean scalar mismatch for {request_id}: expected={expected:.6g} "
                                            f"clean={clean_scalar:.6g} delta={delta:.6g} "
                                            f"tolerance={scalar_value_tolerance:.6g}"
                                        )
                                ablated_scalar = ablated_values[request_id]
                                total_drop = clean_scalar - ablated_scalar
                                rescue = patched_scalar - ablated_scalar
                                unrecovered = clean_scalar - patched_scalar
                                pair_id = str(request["pair_id"])
                                delta_norm = _mean(
                                    pair_delta_norms[pair_id],
                                    label=f"{pair_id}/{component_group_id}/{position_group_id}/{scalar_name}/delta_norm",
                                )
                                projected_norm = _mean(
                                    pair_projected_norms[pair_id],
                                    label=f"{pair_id}/{component_group_id}/{position_group_id}/{scalar_name}/projected_norm",
                                )
                                gradient_norm = _mean(
                                    pair_gradient_norms[pair_id],
                                    label=f"{pair_id}/{component_group_id}/{position_group_id}/{scalar_name}/gradient_norm",
                                )
                                rows.append(
                                    {
                                        "source_step": int(request_id[0]),
                                        "target_step": int(request_id[1]),
                                        "endpoint_kind": endpoint_kind,
                                        "pair_id": pair_id,
                                        "pair_type": str(request["pair_type"]),
                                        "margin_side": str(request["margin_side"]),
                                        "scalar_name": scalar_name,
                                        "source_component": source_component,
                                        "component_group_id": component_group_id,
                                        "components": group_components,
                                        "component_stages": group_stages,
                                        "position_group_id": position_group_id,
                                        "position_roles": group_position_roles,
                                        "basis_kind": "scalar_gradient_svd",
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
                                        "projection_fraction": None if delta_norm == 0.0 else projected_norm / delta_norm,
                                        "mean_gradient_norm": gradient_norm,
                                    }
                                )
    return rows


def run_causal_write_gradient_subspace_rescue(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    pair_types: list[str],
    source_components: list[str],
    scalar_names: list[str] | None = None,
    components: list[str] | None = None,
    component_groups: list[str] | None = None,
    position_roles: list[str] | None = None,
    position_groups: list[str] | None = None,
    subspace_ranks: list[int] | None = None,
    device_name: str = "mps",
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
        raise ValueError("causal-write-gradient-subspace-rescue requires at least one pair type.")
    if not source_components:
        raise ValueError("source_components must not be empty.")
    resolved_ranks = sorted(set(subspace_ranks if subspace_ranks is not None else [1, 2, 4, 8]))
    if any(rank <= 0 for rank in resolved_ranks):
        raise ValueError(f"subspace ranks must be positive, got {resolved_ranks}.")
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
    resolved_sources = [
        component for component in source_components if component in set(available_components)
    ]
    if len(resolved_sources) != len(dict.fromkeys(source_components)):
        missing = [component for component in source_components if component not in set(available_components)]
        raise ValueError(f"Unsupported source components {missing}; expected one of {available_components}.")
    resolved_sources = list(dict.fromkeys(resolved_sources))
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
        "[causal-write-gradient-subspace-rescue] "
        f"pairs={len(pairs_by_id)} scalar_rows={len(scalar_pair_rows)} endpoints={len(endpoint_keys)} "
        f"sources={resolved_sources} component_groups={resolved_component_groups} "
        f"position_groups={resolved_position_groups} ranks={resolved_ranks} "
        f"scalars={resolved_scalars} endpoint_roles={resolved_endpoint_roles} device={device_name}",
        flush=True,
    )
    rescue_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    for source_component in resolved_sources:
        print(
            f"[causal-write-gradient-subspace-rescue] source={source_component} building scalar-gradient bases",
            flush=True,
        )
        source_basis_by_key, source_basis_rows = _collect_gradient_bases_for_source(
            model=model,
            checkpoint_paths_by_step=checkpoint_paths_by_step,
            pairs_by_id=pairs_by_id,
            endpoint_keys=endpoint_keys,
            endpoint_requests=endpoint_requests,
            source_component=source_component,
            components=resolved_components,
            position_roles=resolved_position_roles,
            scalar_names=resolved_scalars,
            endpoint_roles=resolved_endpoint_roles,
            subspace_ranks=resolved_ranks,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
        )
        print(
            f"[causal-write-gradient-subspace-rescue] source={source_component} running projected rescues",
            flush=True,
        )
        source_rescue_rows = _compute_rescue_rows_for_source(
            model=model,
            checkpoint_paths_by_step=checkpoint_paths_by_step,
            pairs_by_id=pairs_by_id,
            endpoint_keys=endpoint_keys,
            endpoint_requests=endpoint_requests,
            source_component=source_component,
            components=resolved_components,
            component_groups=resolved_component_groups,
            position_groups=resolved_position_groups,
            basis_by_key=source_basis_by_key,
            subspace_ranks=resolved_ranks,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            scalar_value_tolerance=scalar_value_tolerance,
            device=device,
        )
        rescue_rows.extend(source_rescue_rows)
        basis_rows.extend(source_basis_rows)
    summary_rows = _summarize_rows(rows=rescue_rows, denominator_threshold=denominator_threshold)

    rescue_rows_path = output_dir / "causal_write_gradient_subspace_rescue_rows.jsonl"
    summary_rows_path = output_dir / "causal_write_gradient_subspace_rescue_summary_rows.jsonl"
    basis_rows_path = output_dir / "causal_write_gradient_subspace_rescue_bases.jsonl"
    pair_rows_path = output_dir / "causal_write_gradient_subspace_rescue_pairs.jsonl"
    report_path = output_dir / "causal_write_gradient_subspace_rescue_report.json"
    markdown_path = output_dir / "causal_write_gradient_subspace_rescue_report.md"
    write_jsonl(rescue_rows_path, rescue_rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(basis_rows_path, basis_rows)
    write_jsonl(
        pair_rows_path,
        [{key: value for key, value in pair.items() if key not in {"clean_record", "corrupted_record"}} for pair in pairs_by_id.values()],
    )
    report = {
        "schema_version": CAUSAL_WRITE_GRADIENT_SUBSPACE_RESCUE_SCHEMA_VERSION,
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
        "basis_kinds": CAUSAL_WRITE_GRADIENT_BASIS_KINDS,
        "subspace_ranks": resolved_ranks,
        "scalar_value_tolerance": scalar_value_tolerance,
        "denominator_threshold": denominator_threshold,
        "markdown_top_k_rows": markdown_top_k_rows,
        "checkpoint_paths_by_step": {str(step): str(path) for step, path in checkpoint_paths_by_step.items()},
        "pair_construction": pair_construction,
        "calculation": {
            "component_delta": "component_write(clean) - component_write(current_source_ablated_state)",
            "scalar_gradient_svd": "uncentered SVD basis over d scalar / d residual_stage[position] vectors",
            "projected_delta": "orthogonal projection of component_delta into the scalar-gradient SVD basis",
            "grouped_patch": "component groups are patched in component order; position groups are patched together",
            "rescue": "scalar(source ablated + grouped projected_delta patch) - scalar(source ablated)",
        },
        "rescue_rows_path": str(rescue_rows_path),
        "summary_rows_path": str(summary_rows_path),
        "basis_rows_path": str(basis_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(f"[causal-write-gradient-subspace-rescue] complete report={report_path} rows={rescue_rows_path}", flush=True)
    return report_path, markdown_path, rescue_rows_path, summary_rows_path, basis_rows_path, pair_rows_path
