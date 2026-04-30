from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.component_output_rescue import _component_patch_stage, _parse_local_component
from circuit.analysis.component_output_rescue_adam_state_attribution import _scalar_tensor_from_logits
from circuit.analysis.contextual_svd_alignment import CONTEXTUAL_GROUP_BY_OPTIONS, _abs_cosine, _subspace_overlap
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    GEOMETRY_POSITION_ROLES,
    _attention_key_positions,
    _checkpoint_step_from_path,
    _validate_single_query_batch,
)
from circuit.analysis.mlp_local_write_map_report import _mlp_input_stage
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
    _selected_pairs_by_id,
)
from circuit.analysis.residual_delta_vector_report import (
    _endpoint_pair_ids,
    _filter_scalar_pair_rows_for_delta,
    _group_token_id,
    _token_label,
)
from circuit.analysis.residual_state_rescue import RESIDUAL_STATE_RESCUE_ENDPOINT_ROLES, _validate_maskable_components
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


MLP_INPUT_FUNCTIONAL_SUBSPACE_SCHEMA_VERSION = 1


def _validate_single_mlp_component(component: str) -> None:
    kind, _, _ = _parse_local_component(component)
    if kind != "mlp":
        raise ValueError(f"Expected an MLP component, got {component!r}.")


def _vector_norm(value: torch.Tensor) -> float:
    return float(value.float().norm().item())


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _safe_signed_fraction(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1.0e-12:
        return None
    return numerator / denominator


def _pca_basis(vectors: list[torch.Tensor], *, rank: int, label: str) -> tuple[torch.Tensor, dict[str, Any]]:
    if rank <= 0:
        raise ValueError("rank must be positive.")
    if not vectors:
        raise ValueError(f"Cannot build PCA basis from empty vectors: {label}")
    matrix = torch.stack([vector.float().cpu() for vector in vectors], dim=0)
    if matrix.ndim != 2:
        raise RuntimeError(f"Expected PCA matrix rank 2 for {label}, got shape {tuple(matrix.shape)}.")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    matrix_rank = int(torch.linalg.matrix_rank(centered).item())
    if matrix_rank <= 0:
        raise RuntimeError(f"{label} centered rank is zero; cannot build a functional subspace.")
    effective_rank = min(rank, matrix_rank, int(centered.size(0)), int(centered.size(1)))
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:effective_rank, :].T.contiguous()
    return basis, {
        "num_vectors": int(matrix.size(0)),
        "dim": int(matrix.size(1)),
        "requested_rank": rank,
        "centered_rank": matrix_rank,
        "basis_rank": effective_rank,
        "singular_values": [float(value) for value in singular_values[:effective_rank].tolist()],
        "top_singular_value": float(singular_values[0].item()),
        "spectral_mass_top_basis": float(singular_values[:effective_rank].sum().item() / singular_values.sum().item()),
    }


def _flatten_selected(stage_tensor: torch.Tensor, *, row: int, positions: list[int]) -> torch.Tensor:
    if not positions:
        raise ValueError("positions must not be empty.")
    return torch.cat([stage_tensor[row, int(position), :].float().reshape(-1) for position in positions], dim=0)


def _mean_selected(stage_tensor: torch.Tensor, *, row: int, positions: list[int]) -> torch.Tensor:
    if not positions:
        raise ValueError("positions must not be empty.")
    return torch.stack([stage_tensor[row, int(position), :].float() for position in positions], dim=0).mean(dim=0)


def _grouped_summary(*, rows: list[dict[str, Any]], subspace_rank: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["source_component"]),
                str(row["mlp_component"]),
                int(row["step"]),
                str(row["endpoint_kind"]),
                str(row["margin_side"]),
                str(row["scalar_name"]),
                str(row["position_role"]),
                str(row["group_by"]),
            )
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        (
            source_component,
            mlp_component,
            step,
            endpoint_kind,
            margin_side,
            scalar_name,
            position_role,
            group_by,
        ) = key
        input_delta_vectors = [row["_input_delta_mean"] for row in group_rows]
        input_gradient_vectors = [row["_input_gradient_mean"] for row in group_rows]
        post_mlp_gradient_vectors = [row["_post_mlp_gradient_mean"] for row in group_rows]
        mlp_output_delta_vectors = [row["_mlp_output_delta_mean"] for row in group_rows]
        input_gradient_basis, input_gradient_summary = _pca_basis(
            input_gradient_vectors,
            rank=subspace_rank,
            label=f"{key}/input_gradient",
        )
        input_delta_basis, input_delta_summary = _pca_basis(
            input_delta_vectors,
            rank=subspace_rank,
            label=f"{key}/input_delta",
        )
        post_mlp_gradient_basis, post_mlp_gradient_summary = _pca_basis(
            post_mlp_gradient_vectors,
            rank=subspace_rank,
            label=f"{key}/post_mlp_gradient",
        )
        mlp_output_delta_basis, mlp_output_delta_summary = _pca_basis(
            mlp_output_delta_vectors,
            rank=subspace_rank,
            label=f"{key}/mlp_output_delta",
        )
        for basis_name, basis_summary in [
            ("input_gradient_pca", input_gradient_summary),
            ("input_delta_pca", input_delta_summary),
            ("post_mlp_gradient_pca", post_mlp_gradient_summary),
            ("mlp_output_delta_pca", mlp_output_delta_summary),
        ]:
            subspace_rows.append(
                {
                    "source_component": source_component,
                    "mlp_component": mlp_component,
                    "step": step,
                    "endpoint_kind": endpoint_kind,
                    "margin_side": margin_side,
                    "scalar_name": scalar_name,
                    "position_role": position_role,
                    "group_by": group_by,
                    "basis_name": basis_name,
                    **basis_summary,
                }
            )
        input_dot_values = [float(row["input_stage_gradient_dot_input_delta"]) for row in group_rows]
        skip_dot_values = [float(row["post_mlp_gradient_dot_skip_delta"]) for row in group_rows]
        mlp_dot_values = [float(row["post_mlp_gradient_dot_mlp_output_delta"]) for row in group_rows]
        total_dot_values = [float(row["post_mlp_gradient_dot_total_delta"]) for row in group_rows]
        input_delta_norms = [float(row["input_delta_norm"]) for row in group_rows]
        input_gradient_norms = [float(row["input_gradient_norm"]) for row in group_rows]
        mlp_output_delta_norms = [float(row["mlp_output_delta_norm"]) for row in group_rows]
        input_cosines = [float(row["input_delta_input_gradient_abs_cosine"]) for row in group_rows]
        post_cosines = [float(row["mlp_output_delta_post_mlp_gradient_abs_cosine"]) for row in group_rows]
        input_delta_in_gradient_basis = [
            _subspace_overlap(
                row["_input_delta_mean"],
                input_gradient_basis,
                label=f"{key}/{row['pair_id']}/input_delta_in_input_gradient_basis",
            )
            for row in group_rows
        ]
        input_gradient_in_delta_basis = [
            _subspace_overlap(
                row["_input_gradient_mean"],
                input_delta_basis,
                label=f"{key}/{row['pair_id']}/input_gradient_in_delta_basis",
            )
            for row in group_rows
        ]
        mlp_delta_in_post_gradient_basis = [
            _subspace_overlap(
                row["_mlp_output_delta_mean"],
                post_mlp_gradient_basis,
                label=f"{key}/{row['pair_id']}/mlp_delta_in_post_gradient_basis",
            )
            for row in group_rows
        ]
        post_gradient_in_mlp_delta_basis = [
            _subspace_overlap(
                row["_post_mlp_gradient_mean"],
                mlp_output_delta_basis,
                label=f"{key}/{row['pair_id']}/post_gradient_in_mlp_delta_basis",
            )
            for row in group_rows
        ]
        total_dot = sum(total_dot_values)
        summary_rows.append(
            {
                "source_component": source_component,
                "mlp_component": mlp_component,
                "step": step,
                "endpoint_kind": endpoint_kind,
                "margin_side": margin_side,
                "scalar_name": scalar_name,
                "input_stage": str(group_rows[0]["input_stage"]),
                "output_stage": str(group_rows[0]["output_stage"]),
                "position_role": position_role,
                "group_by": group_by,
                "num_rows": len(group_rows),
                "sum_input_stage_gradient_dot_input_delta": sum(input_dot_values),
                "sum_post_mlp_gradient_dot_skip_delta": sum(skip_dot_values),
                "sum_post_mlp_gradient_dot_mlp_output_delta": sum(mlp_dot_values),
                "sum_post_mlp_gradient_dot_total_delta": total_dot,
                "mlp_output_fraction_of_post_mlp_total": _safe_signed_fraction(sum(mlp_dot_values), total_dot),
                "skip_fraction_of_post_mlp_total": _safe_signed_fraction(sum(skip_dot_values), total_dot),
                "mean_input_delta_norm": _mean(input_delta_norms, label=f"{key}/input_delta_norm"),
                "mean_input_gradient_norm": _mean(input_gradient_norms, label=f"{key}/input_gradient_norm"),
                "mean_mlp_output_delta_norm": _mean(mlp_output_delta_norms, label=f"{key}/mlp_output_delta_norm"),
                "mean_input_delta_input_gradient_abs_cosine": _mean(input_cosines, label=f"{key}/input_cosine"),
                "mean_mlp_output_delta_post_mlp_gradient_abs_cosine": _mean(post_cosines, label=f"{key}/post_cosine"),
                "mean_input_delta_overlap_with_input_gradient_basis": _mean(
                    input_delta_in_gradient_basis,
                    label=f"{key}/input_delta_in_gradient_basis",
                ),
                "mean_input_gradient_overlap_with_input_delta_basis": _mean(
                    input_gradient_in_delta_basis,
                    label=f"{key}/input_gradient_in_delta_basis",
                ),
                "mean_mlp_output_delta_overlap_with_post_mlp_gradient_basis": _mean(
                    mlp_delta_in_post_gradient_basis,
                    label=f"{key}/mlp_delta_in_post_gradient_basis",
                ),
                "mean_post_mlp_gradient_overlap_with_mlp_output_delta_basis": _mean(
                    post_gradient_in_mlp_delta_basis,
                    label=f"{key}/post_gradient_in_mlp_delta_basis",
                ),
            }
        )
    return summary_rows, subspace_rows


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    summary_rows = sorted(
        report["summary_rows"],
        key=lambda row: abs(float(row["sum_post_mlp_gradient_dot_total_delta"])),
        reverse=True,
    )[: int(report["markdown_top_k_rows"])]
    lines = [
        "# MLP Input Functional Subspace Report",
        "",
        "This report asks which L0H0-written directions are read at the selected MLP input/output stages.",
        "",
        "## Calculation",
        "",
        "`delta_in = z_clean[input_stage] - z_source_ablated[input_stage]`",
        "",
        "`mlp_output_delta = MLP(z_clean) - MLP(z_source_ablated)`",
        "",
        "`input_score = grad_scalar(input_stage) dot delta_in`",
        "",
        "`mlp_score = grad_scalar(output_stage) dot mlp_output_delta`",
        "",
        "The PCA overlap columns ask whether the L0H0 write deltas lie in the same low-dimensional subspace as the scalar gradients.",
        "",
        "## Top Functional Subspaces",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| source | mlp | step | endpoint | scalar | position | group | rows | input dot | mlp dot | total dot | mlp frac | delta/read overlap | output/read overlap |",
                "|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary_rows:
            mlp_fraction = row["mlp_output_fraction_of_post_mlp_total"]
            lines.append(
                f"| `{row['source_component']}` | `{row['mlp_component']}` | {row['step']} | "
                f"`{row['endpoint_kind']}` | `{row['scalar_name']}` | `{row['position_role']}` | `{row['group_by']}` | "
                f"{row['num_rows']} | {row['sum_input_stage_gradient_dot_input_delta']:.6g} | "
                f"{row['sum_post_mlp_gradient_dot_mlp_output_delta']:.6g} | "
                f"{row['sum_post_mlp_gradient_dot_total_delta']:.6g} | "
                f"{'' if mlp_fraction is None else f'{mlp_fraction:.6g}'} | "
                f"{row['mean_input_delta_overlap_with_input_gradient_basis']:.6g} | "
                f"{row['mean_mlp_output_delta_overlap_with_post_mlp_gradient_basis']:.6g} |"
            )
    else:
        lines.append("No summary rows were produced.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- functional rows: `{report['functional_rows_path']}`",
            f"- summary rows: `{report['summary_rows_path']}`",
            f"- subspace rows: `{report['subspace_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compute_functional_rows(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    scalar_pair_rows_by_endpoint: dict[tuple[int, str, str], list[dict[str, Any]]],
    endpoint_pair_ids: dict[tuple[int, str, str], set[str]],
    source_component: str,
    mlp_component: str,
    position_roles: list[str],
    group_by_values: list[str],
    scalar_names: list[str],
    batch_size: int,
    pad_token_id: int,
    vocab: Vocabulary,
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
    input_stage = _mlp_input_stage(mlp_component)
    output_stage = _component_patch_stage(mlp_component)
    rows: list[dict[str, Any]] = []
    pair_ids = sorted(pairs_by_id)
    for step, margin_side, endpoint_kind in sorted(endpoint_pair_ids):
        if step not in checkpoint_paths_by_step:
            raise KeyError(f"No checkpoint path for step {step}.")
        checkpoint_path = checkpoint_paths_by_step[step]
        checkpoint = load_checkpoint(checkpoint_path, device)
        if int(checkpoint["step"]) != step:
            raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={checkpoint['step']} step={step}")
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        endpoint_required_pair_ids = endpoint_pair_ids[(step, margin_side, endpoint_kind)]
        endpoint_scalar_rows = scalar_pair_rows_by_endpoint[(step, margin_side, endpoint_kind)]
        scalar_rows_by_pair_id = {str(row["pair_id"]): row for row in endpoint_scalar_rows}
        side_key = f"{margin_side}_record"
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
                raise RuntimeError("MLP input functional subspace report requires residual streams.")
            clean_logits, clean_targets, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            _, _, ablated_metadata = extract_answer_logits(ablated_outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=clean_metadata, label="mlp-input-functional clean")
            _validate_single_query_batch(batch=batch, metadata=ablated_metadata, label="mlp-input-functional ablated")

            input_patch = ablated_outputs.residual_streams[input_stage].detach().clone().requires_grad_(True)
            input_patch_outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                residual_patch={input_stage: input_patch},
            )
            input_patch_logits, input_patch_targets, input_patch_metadata = extract_answer_logits(
                input_patch_outputs.logits,
                batch,
            )
            _validate_single_query_batch(batch=batch, metadata=input_patch_metadata, label="mlp-input-functional input patch")
            output_patch = ablated_outputs.residual_streams[output_stage].detach().clone().requires_grad_(True)
            output_patch_outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                residual_patch={output_stage: output_patch},
            )
            output_patch_logits, output_patch_targets, output_patch_metadata = extract_answer_logits(
                output_patch_outputs.logits,
                batch,
            )
            _validate_single_query_batch(batch=batch, metadata=output_patch_metadata, label="mlp-input-functional output patch")
            for flat_index in range(int(clean_metadata["rows"].numel())):
                query_batch_row = int(clean_metadata["rows"][flat_index].item())
                pair_id = str(batch_pair_ids[query_batch_row])
                if pair_id not in endpoint_required_pair_ids:
                    continue
                if pair_id not in scalar_rows_by_pair_id:
                    raise KeyError(f"Missing scalar row for pair={pair_id} endpoint={step}/{margin_side}/{endpoint_kind}.")
                scalar_row = scalar_rows_by_pair_id[pair_id]
                if int(input_patch_targets[flat_index].item()) != int(clean_targets[flat_index].item()):
                    raise RuntimeError(f"Input patch target mismatch for pair={pair_id}.")
                if int(output_patch_targets[flat_index].item()) != int(clean_targets[flat_index].item()):
                    raise RuntimeError(f"Output patch target mismatch for pair={pair_id}.")
                query_index = int(clean_metadata["query_indices"][flat_index].item())
                prediction_position = int(clean_metadata["prediction_positions"][flat_index].item())
                record = batch["records"][query_batch_row]
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
                answer_value_token_id = int(clean_targets[flat_index].item())
                source_wrong_token_id = int(scalar_row["source_best_wrong_token_id"])
                target_wrong_token_id = int(scalar_row["target_best_wrong_token_id"])
                clean_input_stage = clean_outputs.residual_streams[input_stage].detach()
                ablated_input_stage = ablated_outputs.residual_streams[input_stage].detach()
                clean_output_stage = clean_outputs.residual_streams[output_stage].detach()
                ablated_output_stage = ablated_outputs.residual_streams[output_stage].detach()
                clean_mlp_write = clean_output_stage - clean_input_stage
                ablated_mlp_write = ablated_output_stage - ablated_input_stage
                for scalar_name in scalar_names:
                    input_scalar = _scalar_tensor_from_logits(
                        logits=input_patch_logits[flat_index],
                        scalar_name=scalar_name,
                        answer_target_id=answer_value_token_id,
                        source_best_wrong_token_id=source_wrong_token_id,
                        target_best_wrong_token_id=target_wrong_token_id,
                        endpoint_kind=endpoint_kind,
                    )
                    output_scalar = _scalar_tensor_from_logits(
                        logits=output_patch_logits[flat_index],
                        scalar_name=scalar_name,
                        answer_target_id=answer_value_token_id,
                        source_best_wrong_token_id=source_wrong_token_id,
                        target_best_wrong_token_id=target_wrong_token_id,
                        endpoint_kind=endpoint_kind,
                    )
                    input_grad = torch.autograd.grad(
                        input_scalar,
                        input_patch,
                        retain_graph=True,
                        allow_unused=False,
                    )[0].detach()
                    output_grad = torch.autograd.grad(
                        output_scalar,
                        output_patch,
                        retain_graph=True,
                        allow_unused=False,
                    )[0].detach()
                    for position_role in position_roles:
                        context_batch_row, positions_tensor = _attention_key_positions(
                            batch=batch,
                            metadata=clean_metadata,
                            flat_index=flat_index,
                            position_role=position_role,
                            max_position=prediction_position,
                        )
                        positions = [int(position) for position in positions_tensor]
                        input_delta_flat = _flatten_selected(
                            clean_input_stage - ablated_input_stage,
                            row=context_batch_row,
                            positions=positions,
                        )
                        post_mlp_delta_flat = _flatten_selected(
                            clean_output_stage - ablated_output_stage,
                            row=context_batch_row,
                            positions=positions,
                        )
                        mlp_output_delta_flat = _flatten_selected(
                            clean_mlp_write - ablated_mlp_write,
                            row=context_batch_row,
                            positions=positions,
                        )
                        input_gradient_flat = _flatten_selected(input_grad, row=context_batch_row, positions=positions)
                        post_mlp_gradient_flat = _flatten_selected(output_grad, row=context_batch_row, positions=positions)
                        input_delta_mean = _mean_selected(
                            clean_input_stage - ablated_input_stage,
                            row=context_batch_row,
                            positions=positions,
                        ).detach().cpu()
                        input_gradient_mean = _mean_selected(input_grad, row=context_batch_row, positions=positions).detach().cpu()
                        post_mlp_gradient_mean = _mean_selected(
                            output_grad,
                            row=context_batch_row,
                            positions=positions,
                        ).detach().cpu()
                        mlp_output_delta_mean = _mean_selected(
                            clean_mlp_write - ablated_mlp_write,
                            row=context_batch_row,
                            positions=positions,
                        ).detach().cpu()
                        post_mlp_total_mean = _mean_selected(
                            clean_output_stage - ablated_output_stage,
                            row=context_batch_row,
                            positions=positions,
                        ).detach().cpu()
                        group_tokens = {
                            group_by: _group_token_id(
                                group_by=group_by,
                                batch=batch,
                                metadata=clean_metadata,
                                answer_targets=clean_targets,
                                flat_index=flat_index,
                                context_batch_row=context_batch_row,
                                context_position=positions[0],
                            )
                            for group_by in group_by_values
                        }
                        input_dot = float(torch.dot(input_gradient_flat.float(), input_delta_flat.float()).item())
                        skip_dot = float(torch.dot(post_mlp_gradient_flat.float(), input_delta_flat.float()).item())
                        mlp_dot = float(torch.dot(post_mlp_gradient_flat.float(), mlp_output_delta_flat.float()).item())
                        total_dot = float(torch.dot(post_mlp_gradient_flat.float(), post_mlp_delta_flat.float()).item())
                        base_row: dict[str, Any] = {
                            "source_component": source_component,
                            "mlp_component": mlp_component,
                            "step": step,
                            "checkpoint": str(checkpoint_path),
                            "endpoint_kind": endpoint_kind,
                            "margin_side": margin_side,
                            "scalar_name": scalar_name,
                            "pair_id": pair_id,
                            "pair_type": str(pairs_by_id[pair_id]["pair_type"]),
                            "sample_id": str(record["sample_id"]),
                            "split": str(record["split"]),
                            "query_index": query_index,
                            "input_stage": input_stage,
                            "output_stage": output_stage,
                            "position_role": position_role,
                            "selected_positions": positions,
                            "num_positions": len(positions),
                            "query_key_token_id": query_key_token_id,
                            "query_key_token": _token_label(vocab, query_key_token_id),
                            "support_value_token_id": support_value_token_id,
                            "support_value_token": _token_label(vocab, support_value_token_id),
                            "answer_value_token_id": answer_value_token_id,
                            "answer_value_token": _token_label(vocab, answer_value_token_id),
                            "input_delta_norm": _vector_norm(input_delta_flat),
                            "input_gradient_norm": _vector_norm(input_gradient_flat),
                            "post_mlp_gradient_norm": _vector_norm(post_mlp_gradient_flat),
                            "mlp_output_delta_norm": _vector_norm(mlp_output_delta_flat),
                            "post_mlp_total_delta_norm": _vector_norm(post_mlp_delta_flat),
                            "input_stage_gradient_dot_input_delta": input_dot,
                            "post_mlp_gradient_dot_skip_delta": skip_dot,
                            "post_mlp_gradient_dot_mlp_output_delta": mlp_dot,
                            "post_mlp_gradient_dot_total_delta": total_dot,
                            "mlp_output_fraction_of_post_mlp_total": _safe_ratio(mlp_dot, total_dot),
                            "skip_fraction_of_post_mlp_total": _safe_ratio(skip_dot, total_dot),
                            "input_delta_input_gradient_abs_cosine": _abs_cosine(
                                input_delta_flat.cpu(),
                                input_gradient_flat.cpu(),
                                label=f"{pair_id}/{scalar_name}/{position_role}/input_delta_input_gradient",
                            ),
                            "mlp_output_delta_post_mlp_gradient_abs_cosine": _abs_cosine(
                                mlp_output_delta_flat.cpu(),
                                post_mlp_gradient_flat.cpu(),
                                label=f"{pair_id}/{scalar_name}/{position_role}/mlp_output_post_gradient",
                            ),
                            "_input_delta_mean": input_delta_mean,
                            "_input_gradient_mean": input_gradient_mean,
                            "_post_mlp_gradient_mean": post_mlp_gradient_mean,
                            "_mlp_output_delta_mean": mlp_output_delta_mean,
                            "_post_mlp_total_mean": post_mlp_total_mean,
                        }
                        for group_by, token_id in group_tokens.items():
                            base_row[f"{group_by}_token_id"] = int(token_id)
                            base_row[f"{group_by}_token"] = _token_label(vocab, int(token_id))
                        for group_by in group_by_values:
                            row = dict(base_row)
                            row["group_by"] = group_by
                            rows.append(row)
    return rows


def run_mlp_input_functional_subspace_report(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    device_name: str,
    pair_types: list[str],
    source_component: str,
    mlp_component: str,
    position_roles: list[str],
    group_by_values: list[str],
    scalar_names: list[str] | None,
    margin_sides: list[str] | None,
    endpoint_roles: list[str] | None,
    endpoint_steps: list[int] | None,
    split_filter: list[str] | None,
    max_pairs_per_type: int,
    min_pairs_per_type: int,
    batch_size: int | None,
    subspace_rank: int,
    markdown_top_k_rows: int,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    if subspace_rank <= 0:
        raise ValueError("subspace_rank must be positive.")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    resolved_margin_sides = _resolve_unique_values(
        values=margin_sides,
        default_values=["clean"],
        allowed_values=OUTPUT_ROUTE_MARGIN_SIDES,
        label="margin side",
    )
    resolved_scalars = _resolve_unique_values(
        values=scalar_names,
        default_values=["fixed_source_competitor_margin", "fixed_target_competitor_margin"],
        allowed_values=OUTPUT_ROUTE_SCALARS,
        label="scalar",
    )
    resolved_endpoint_roles = _resolve_unique_values(
        values=endpoint_roles,
        default_values=["source"],
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
        requested_components=[source_component],
        available_components=available_components,
    )
    if len(resolved_sources) != 1:
        raise RuntimeError(f"Expected exactly one source component, got {resolved_sources}.")
    resolved_mlps = _filter_component_labels(
        requested_components=[mlp_component],
        available_components=available_components,
    )
    if len(resolved_mlps) != 1:
        raise RuntimeError(f"Expected exactly one MLP component, got {resolved_mlps}.")
    resolved_source = resolved_sources[0]
    resolved_mlp = resolved_mlps[0]
    _validate_single_mlp_component(resolved_mlp)
    _validate_maskable_components(
        components=[resolved_source],
        num_layers=len(model.blocks),
        num_heads=model.spec.n_heads,
        device=device,
    )

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
    if endpoint_steps is not None:
        endpoint_step_set = set(int(step) for step in endpoint_steps)
        endpoint_pair_ids = {
            key: value
            for key, value in endpoint_pair_ids.items()
            if int(key[0]) in endpoint_step_set
        }
        if not endpoint_pair_ids:
            raise RuntimeError(f"Endpoint step filter matched no endpoints: {sorted(endpoint_step_set)}")
    scalar_pair_rows_by_endpoint: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    allowed_endpoint_keys = set(endpoint_pair_ids)
    for row in scalar_pair_rows:
        margin_side = str(row["margin_side"])
        for endpoint_kind in resolved_endpoint_roles:
            key = (int(row[f"{endpoint_kind}_step"]), margin_side, endpoint_kind)
            if key in allowed_endpoint_keys:
                scalar_pair_rows_by_endpoint[key].append(row)
    missing_scalar_endpoints = sorted(allowed_endpoint_keys - set(scalar_pair_rows_by_endpoint))
    if missing_scalar_endpoints:
        raise RuntimeError(f"Missing scalar rows for endpoint(s): {missing_scalar_endpoints}")
    required_pair_ids = {pair_id for pair_ids in endpoint_pair_ids.values() for pair_id in pair_ids}
    pairs_by_id, pair_construction = _selected_pairs_by_id(
        config_path=config_path,
        probe_set_path=probe_set_path,
        pair_types=pair_types,
        split_filter=split_filter,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        required_pair_ids=required_pair_ids,
    )
    analysis_batch_size = int(spec.evaluation.batch_size) if batch_size is None else int(batch_size)

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[mlp-input-functional-subspace-report] "
        f"pairs={len(pairs_by_id)} endpoints={len(endpoint_pair_ids)} source={resolved_source} mlp={resolved_mlp} "
        f"position_roles={resolved_position_roles} group_by={resolved_group_by} scalars={resolved_scalars} "
        f"subspace_rank={subspace_rank} device={device_name}",
        flush=True,
    )
    functional_rows = _compute_functional_rows(
        model=model,
        checkpoint_paths_by_step=checkpoint_paths_by_step,
        pairs_by_id=pairs_by_id,
        scalar_pair_rows_by_endpoint=scalar_pair_rows_by_endpoint,
        endpoint_pair_ids=endpoint_pair_ids,
        source_component=resolved_source,
        mlp_component=resolved_mlp,
        position_roles=resolved_position_roles,
        group_by_values=resolved_group_by,
        scalar_names=resolved_scalars,
        batch_size=analysis_batch_size,
        pad_token_id=vocab.pad_token_id,
        vocab=vocab,
        device=device,
    )
    summary_rows, subspace_rows = _grouped_summary(rows=functional_rows, subspace_rank=subspace_rank)
    serializable_functional_rows: list[dict[str, Any]] = []
    for row in functional_rows:
        serializable_functional_rows.append(
            {
                key: value
                for key, value in row.items()
                if not key.startswith("_")
            }
        )

    functional_rows_path = output_dir / "mlp_input_functional_subspace_rows.jsonl"
    summary_rows_path = output_dir / "mlp_input_functional_subspace_summary_rows.jsonl"
    subspace_rows_path = output_dir / "mlp_input_functional_subspace_subspaces.jsonl"
    pair_rows_path = output_dir / "mlp_input_functional_subspace_pairs.jsonl"
    report_path = output_dir / "mlp_input_functional_subspace_report.json"
    markdown_path = output_dir / "mlp_input_functional_subspace_report.md"
    write_jsonl(functional_rows_path, serializable_functional_rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(subspace_rows_path, subspace_rows)
    write_jsonl(
        pair_rows_path,
        [{key: value for key, value in pair.items() if key not in {"clean_record", "corrupted_record"}} for pair in pairs_by_id.values()],
    )

    report = {
        "schema_version": MLP_INPUT_FUNCTIONAL_SUBSPACE_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "device": device_name,
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names": resolved_scalars,
        "endpoint_roles": resolved_endpoint_roles,
        "endpoint_steps": None if endpoint_steps is None else sorted(set(int(step) for step in endpoint_steps)),
        "source_component": resolved_source,
        "mlp_component": resolved_mlp,
        "input_stage": _mlp_input_stage(resolved_mlp),
        "output_stage": _component_patch_stage(resolved_mlp),
        "position_roles": resolved_position_roles,
        "group_by": resolved_group_by,
        "subspace_rank": subspace_rank,
        "batch_size": analysis_batch_size,
        "markdown_top_k_rows": markdown_top_k_rows,
        "checkpoint_paths_by_step": {str(step): str(path) for step, path in checkpoint_paths_by_step.items()},
        "pair_construction": pair_construction,
        "calculation": {
            "input_stage_gradient_dot_input_delta": "grad scalar wrt selected MLP input residual dot L0H0 clean-minus-ablated input delta",
            "post_mlp_gradient_dot_mlp_output_delta": "grad scalar wrt selected MLP output residual dot the MLP's local transformation of the L0H0 input delta",
            "post_mlp_gradient_dot_skip_delta": "same output-stage gradient dot the residual skip part of the L0H0 input delta",
            "functional_subspace": "PCA basis of scalar gradients compared with PCA basis of L0H0 deltas and MLP output deltas",
        },
        "functional_rows_path": str(functional_rows_path),
        "summary_rows_path": str(summary_rows_path),
        "subspace_rows_path": str(subspace_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(f"[mlp-input-functional-subspace-report] complete report={report_path} rows={functional_rows_path}", flush=True)
    return report_path, markdown_path, functional_rows_path, summary_rows_path, subspace_rows_path, pair_rows_path
