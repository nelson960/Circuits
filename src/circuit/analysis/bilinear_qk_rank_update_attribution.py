from __future__ import annotations

import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    ATTENTION_SCORE_RECORD_SIDES,
    GEOMETRY_POSITION_ROLES,
    ROUTE_GRADIENT_DECOMPOSITION_MODES,
    _attention_key_positions,
    _build_causal_patch_pairs,
    _build_route_gradient_decomposition_groups,
    _checkpoint_step_from_path,
    _gradient_dot_summary,
    _gradient_dot_summary_for_group,
    _group_metadata,
    _holdout_pair_set,
    _model_parameter_snapshot,
    _pair_metadata,
    _parameter_delta,
    _parameter_gradients,
    _resolve_checkpoint_paths,
    _resolve_route_gradient_decomposition_modes,
    _route_gradient_groups,
    _safe_ratio,
    _sign_match,
    _single_attention_position,
    _validate_single_query_batch,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import append_jsonl, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.train import _compute_learning_rate
from circuit.vocab import Vocabulary


BILINEAR_QK_RANK_UPDATE_ATTRIBUTION_SCHEMA_VERSION = 1
LAYER_NORM_MODES = ["none", "head_ln1"]


def _import_matplotlib() -> Any:
    cache_dir = Path(tempfile.gettempdir()) / "circuit_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _rank_qk_source_basis(
    *,
    model: torch.nn.Module,
    head_layer: int,
    head: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    block = model.blocks[head_layer]
    head_dim = int(block.attn.head_dim)
    head_slice = slice(head * head_dim, (head + 1) * head_dim)
    with torch.no_grad():
        q_rows = block.attn.q_proj.weight.detach()[head_slice, :].float()
        k_rows = block.attn.k_proj.weight.detach()[head_slice, :].float()
        qk_matrix = q_rows.T.matmul(k_rows)
        u, singular_values, vh = torch.linalg.svd(qk_matrix.cpu(), full_matrices=False)
    if rank <= 0 or rank > int(singular_values.numel()):
        raise ValueError(f"rank must be in 1..{int(singular_values.numel())}, got {rank}.")
    left_basis = u[:, :rank].to(device=block.attn.q_proj.weight.device, dtype=block.attn.q_proj.weight.dtype).detach()
    right_basis = vh[:rank, :].T.to(device=block.attn.q_proj.weight.device, dtype=block.attn.q_proj.weight.dtype).detach()
    gaps = (singular_values[:-1] - singular_values[1:]).abs()
    min_gap = None if gaps.numel() == 0 else float(gaps.min().detach().cpu().item())
    singular_sum = singular_values.sum()
    if float(singular_sum.detach().cpu().item()) <= 0.0:
        raise RuntimeError(f"Non-positive QK singular-value sum for L{head_layer}H{head}.")
    return left_basis, right_basis, {
        "qk_singular_value_top": float(singular_values[0].detach().cpu().item()),
        "qk_singular_value_sum": float(singular_sum.detach().cpu().item()),
        "qk_singular_value_rank_sum": float(singular_values[:rank].sum().detach().cpu().item()),
        "qk_singular_value_rank_fraction": float((singular_values[:rank].sum() / singular_sum).detach().cpu().item()),
        "qk_singular_value_min_gap": min_gap,
    }


def _fixed_basis_rank_qk_matrix(
    *,
    model: torch.nn.Module,
    head_layer: int,
    head: int,
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
) -> torch.Tensor:
    block = model.blocks[head_layer]
    head_dim = int(block.attn.head_dim)
    head_slice = slice(head * head_dim, (head + 1) * head_dim)
    q_rows = block.attn.q_proj.weight[head_slice, :]
    k_rows = block.attn.k_proj.weight[head_slice, :]
    qk_matrix = q_rows.T.matmul(k_rows)
    projected = left_basis.matmul(left_basis.T.matmul(qk_matrix).matmul(right_basis)).matmul(right_basis.T)
    if not torch.isfinite(projected).all():
        raise RuntimeError(f"Non-finite fixed-basis rank QK matrix for L{head_layer}H{head}.")
    return projected


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise RuntimeError(f"Cannot compute mean for empty values: {label}")
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) * (value - mean) for value in values) / len(values))


def _parameter_l2_norm(parameters: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, value in parameters.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite parameter tensor while computing norm: {name}")
        flat = value.float().reshape(-1)
        total += float(torch.dot(flat, flat).item())
    return math.sqrt(total)


def _assert_finite_gradients(gradients: dict[str, torch.Tensor], *, label: str) -> None:
    bad = [name for name, gradient in gradients.items() if not torch.isfinite(gradient).all()]
    if bad:
        raise RuntimeError(f"Non-finite gradients for {label}: {bad[:20]}")


def _rank_match_payload_for_pairs(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    head_layer: int,
    head: int,
    rank: int,
    left_basis: torch.Tensor,
    right_basis: torch.Tensor,
    singular_summary: dict[str, float],
    context_stage: str,
    layernorm_mode: str,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    record_side: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    track_grad: bool,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pairs must not be empty for rank QK match computation.")
    if record_side not in ATTENTION_SCORE_RECORD_SIDES:
        raise ValueError(f"Unsupported record side {record_side!r}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    if layernorm_mode not in LAYER_NORM_MODES:
        raise ValueError(f"Unsupported layernorm_mode {layernorm_mode!r}; expected one of {LAYER_NORM_MODES}.")

    model.eval()
    if track_grad:
        model.zero_grad(set_to_none=True)

    rank_matrix = _fixed_basis_rank_qk_matrix(
        model=model,
        head_layer=head_layer,
        head=head,
        left_basis=left_basis,
        right_basis=right_basis,
    )
    scale = math.sqrt(float(model.blocks[head_layer].attn.head_dim))
    side_key = f"{record_side}_record"
    objective_terms: list[torch.Tensor] = []
    pair_rows: list[dict[str, Any]] = []
    support_score_values: list[float] = []
    distractor_score_values: list[float] = []
    num_support_scores = 0
    num_distractor_scores = 0

    for start_index in range(0, len(pairs), batch_size):
        pair_batch = pairs[start_index : start_index + batch_size]
        records = [pair[side_key] for pair in pair_batch]
        batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
        if track_grad:
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
            )
        else:
            with torch.no_grad():
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_residual_streams=True,
                )
        if outputs.residual_streams is None:
            raise RuntimeError("Rank QK update attribution requires residual streams.")
        if context_stage not in outputs.residual_streams:
            raise KeyError(
                f"Residual stage {context_stage!r} not found. Available stages: {sorted(outputs.residual_streams)}"
            )
        _, _, metadata = extract_answer_logits(outputs.logits, batch)
        _validate_single_query_batch(batch=batch, metadata=metadata, label="rank QK update attribution")
        states = outputs.residual_streams[context_stage].float()
        if layernorm_mode == "head_ln1":
            states = model.blocks[head_layer].ln_1(states)

        for pair_index, pair in enumerate(pair_batch):
            batch_row, query_position = _single_attention_position(
                batch=batch,
                metadata=metadata,
                flat_index=pair_index,
                position_role=score_query_role,
                label="rank QK query",
            )
            support_batch_row, support_positions = _attention_key_positions(
                batch=batch,
                metadata=metadata,
                flat_index=pair_index,
                position_role=support_key_role,
                max_position=query_position,
            )
            distractor_batch_row, distractor_positions = _attention_key_positions(
                batch=batch,
                metadata=metadata,
                flat_index=pair_index,
                position_role=distractor_key_role,
                max_position=query_position,
            )
            if support_batch_row != batch_row:
                raise RuntimeError(
                    f"Support role {support_key_role!r} selected batch row {support_batch_row}, "
                    f"but query role {score_query_role!r} selected batch row {batch_row} for pair {pair['pair_id']}."
                )
            if distractor_batch_row != batch_row:
                raise RuntimeError(
                    f"Distractor role {distractor_key_role!r} selected batch row {distractor_batch_row}, "
                    f"but query role {score_query_role!r} selected batch row {batch_row} for pair {pair['pair_id']}."
                )
            query_vector = states[batch_row, query_position, :]
            support_vectors = states[batch_row, support_positions, :]
            distractor_vectors = states[batch_row, distractor_positions, :]
            query_projected = torch.matmul(query_vector, rank_matrix)
            support_scores = torch.matmul(support_vectors, query_projected) / scale
            distractor_scores = torch.matmul(distractor_vectors, query_projected) / scale
            if support_scores.numel() <= 0 or distractor_scores.numel() <= 0:
                raise RuntimeError(f"Empty support/distractor scores for pair {pair['pair_id']}.")
            support_mean = support_scores.mean()
            distractor_mean = distractor_scores.mean()
            separation = support_mean - distractor_mean
            objective_terms.append(separation)
            support_values = [float(value.detach().cpu().item()) for value in support_scores.reshape(-1)]
            distractor_values = [float(value.detach().cpu().item()) for value in distractor_scores.reshape(-1)]
            support_score_values.extend(support_values)
            distractor_score_values.extend(distractor_values)
            num_support_scores += len(support_values)
            num_distractor_scores += len(distractor_values)
            pair_rows.append(
                {
                    "pair_id": str(pair["pair_id"]),
                    "split": str(pair["split"]),
                    "pair_type": str(pair["pair_type"]),
                    "record_side": record_side,
                    "source_sample_id": str(pair["source_sample_id"]),
                    "source_query_index": int(pair["source_query_index"]),
                    "query_position": int(query_position),
                    "support_positions": [int(position) for position in support_positions],
                    "distractor_positions": [int(position) for position in distractor_positions],
                    "num_support_positions": len(support_positions),
                    "num_distractor_positions": len(distractor_positions),
                    "support_score_mean": float(support_mean.detach().cpu().item()),
                    "distractor_score_mean": float(distractor_mean.detach().cpu().item()),
                    "rank_match_separation": float(separation.detach().cpu().item()),
                    "rank_match_max_margin": float((support_scores.mean() - distractor_scores.max()).detach().cpu().item()),
                    "support_beats_all_distractors": bool((support_scores.max() > distractor_scores.max()).detach().cpu().item()),
                }
            )

    if not objective_terms:
        raise RuntimeError("Rank QK update attribution produced no objective terms.")
    objective = torch.stack(objective_terms).mean()
    payload: dict[str, Any] = {
        "score_value": float(objective.detach().cpu().item()),
        "score_value_abs_mean": _mean([abs(float(row["rank_match_separation"])) for row in pair_rows], label="abs rank match separation"),
        "score_value_std": _std([float(row["rank_match_separation"]) for row in pair_rows]),
        "support_score_value_mean": _mean(support_score_values, label="support rank scores"),
        "distractor_score_value_mean": _mean(distractor_score_values, label="distractor rank scores"),
        "num_scores": num_support_scores + num_distractor_scores,
        "num_support_scores": num_support_scores,
        "num_distractor_scores": num_distractor_scores,
        "num_pairs": len(pairs),
        "pair_rows": pair_rows,
        "singular_summary": dict(singular_summary),
    }
    if track_grad:
        objective.backward()
        gradients, zero_gradient_parameter_names = _parameter_gradients(model=model, require_all=False)
        _assert_finite_gradients(gradients, label=f"rank-{rank} QK objective")
        model.zero_grad(set_to_none=True)
        payload["gradients"] = gradients
        payload["zero_gradient_parameter_names"] = zero_gradient_parameter_names
    return payload


def _actual_summary(
    *,
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    split: str,
    pair_type: str,
) -> dict[str, Any]:
    source_by_id = {str(row["pair_id"]): row for row in source_rows}
    target_by_id = {str(row["pair_id"]): row for row in target_rows}
    if set(source_by_id) != set(target_by_id):
        raise RuntimeError("Source/target rank pair row ids differ.")
    source_values: list[float] = []
    target_values: list[float] = []
    delta_values: list[float] = []
    support_delta_values: list[float] = []
    distractor_delta_values: list[float] = []
    max_margin_delta_values: list[float] = []
    win_values: list[float] = []
    num_support_scores = 0
    num_distractor_scores = 0
    for pair_id in sorted(source_by_id):
        source = source_by_id[pair_id]
        target = target_by_id[pair_id]
        if split != "__all__" and str(source["split"]) != split:
            continue
        if pair_type != "__all__" and str(source["pair_type"]) != pair_type:
            continue
        source_value = float(source["rank_match_separation"])
        target_value = float(target["rank_match_separation"])
        source_values.append(source_value)
        target_values.append(target_value)
        delta_values.append(target_value - source_value)
        support_delta_values.append(float(target["support_score_mean"]) - float(source["support_score_mean"]))
        distractor_delta_values.append(float(target["distractor_score_mean"]) - float(source["distractor_score_mean"]))
        max_margin_delta_values.append(float(target["rank_match_max_margin"]) - float(source["rank_match_max_margin"]))
        win_values.append(float(bool(target["support_beats_all_distractors"])) - float(bool(source["support_beats_all_distractors"])))
        num_support_scores += int(source["num_support_positions"])
        num_distractor_scores += int(source["num_distractor_positions"])
    if not delta_values:
        raise RuntimeError(f"No rank QK rows matched split={split!r} pair_type={pair_type!r}.")
    return {
        "num_pairs": len(delta_values),
        "num_scores": num_support_scores + num_distractor_scores,
        "num_support_scores": num_support_scores,
        "num_distractor_scores": num_distractor_scores,
        "source_value": _mean(source_values, label="source rank match separation"),
        "target_value": _mean(target_values, label="target rank match separation"),
        "actual_delta": _mean(delta_values, label="rank match delta"),
        "actual_delta_abs_mean": _mean([abs(value) for value in delta_values], label="abs rank match delta"),
        "actual_delta_std": _std(delta_values),
        "support_delta_mean": _mean(support_delta_values, label="support rank score delta"),
        "distractor_delta_mean": _mean(distractor_delta_values, label="distractor rank score delta"),
        "max_margin_delta_mean": _mean(max_margin_delta_values, label="rank max margin delta"),
        "support_beats_all_delta_mean": _mean(win_values, label="support beats all delta"),
    }


def _metric_row(
    *,
    source_step: int,
    target_step: int,
    source_checkpoint: Path,
    target_checkpoint: Path,
    learning_rate: float,
    split: str,
    pair_type: str,
    record_side: str,
    rank: int,
    context_stage: str,
    layernorm_mode: str,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    actual: dict[str, Any],
    source_payload: dict[str, Any],
    dot_summary: dict[str, Any],
    parameter_delta_l2_norm: float,
    min_error_denominator: float,
) -> dict[str, Any]:
    actual_delta = float(actual["actual_delta"])
    predicted_delta = float(dot_summary["dot"])
    residual = actual_delta - predicted_delta
    relative_error_denominator = max(abs(actual_delta), min_error_denominator)
    predicted_relative_error_denominator = max(abs(predicted_delta), min_error_denominator)
    return {
        "source_step": source_step,
        "target_step": target_step,
        "step_gap": target_step - source_step,
        "source_checkpoint": str(source_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "learning_rate": learning_rate,
        "split": split,
        "pair_type": pair_type,
        "record_side": record_side,
        "objective": "rank_bilinear_qk_retrieval_separation",
        "rank": rank,
        "context_stage": context_stage,
        "layernorm_mode": layernorm_mode,
        "head_layer": head_layer,
        "head": head,
        "head_label": f"L{head_layer}H{head}",
        "score_query_role": score_query_role,
        "support_key_role": support_key_role,
        "distractor_key_role": distractor_key_role,
        "source_value": float(actual["source_value"]),
        "target_value": float(actual["target_value"]),
        "actual_delta": actual_delta,
        "actual_delta_abs_mean": float(actual["actual_delta_abs_mean"]),
        "actual_delta_std": float(actual["actual_delta_std"]),
        "predicted_delta": predicted_delta,
        "residual": residual,
        "absolute_error": abs(residual),
        "relative_error": abs(residual) / relative_error_denominator,
        "predicted_relative_error": abs(residual) / predicted_relative_error_denominator,
        "relative_error_denominator": relative_error_denominator,
        "predicted_relative_error_denominator": predicted_relative_error_denominator,
        "sign_match": _sign_match(actual_delta, predicted_delta),
        "support_delta_mean": float(actual["support_delta_mean"]),
        "distractor_delta_mean": float(actual["distractor_delta_mean"]),
        "max_margin_delta_mean": float(actual["max_margin_delta_mean"]),
        "support_beats_all_delta_mean": float(actual["support_beats_all_delta_mean"]),
        "num_pairs": int(actual["num_pairs"]),
        "num_scores": int(actual["num_scores"]),
        "num_support_scores": int(actual["num_support_scores"]),
        "num_distractor_scores": int(actual["num_distractor_scores"]),
        "parameter_delta_l2_norm": parameter_delta_l2_norm,
        "score_gradient_l2_norm": float(dot_summary["right_l2_norm"]),
        "update_score_gradient_cosine": dot_summary["cosine"],
        "num_parameters": int(dot_summary["num_parameters"]),
        "zero_score_gradient_parameter_count": len(source_payload["zero_gradient_parameter_names"]),
        "zero_score_gradient_parameter_names": source_payload["zero_gradient_parameter_names"],
        **source_payload["singular_summary"],
    }


def _decomposition_row(
    *,
    metric_row: dict[str, Any],
    group: Any,
    dot_summary: dict[str, Any],
) -> dict[str, Any]:
    contribution = float(dot_summary["dot"])
    return {
        "source_step": metric_row["source_step"],
        "target_step": metric_row["target_step"],
        "step_gap": metric_row["step_gap"],
        "source_checkpoint": metric_row["source_checkpoint"],
        "target_checkpoint": metric_row["target_checkpoint"],
        "learning_rate": metric_row["learning_rate"],
        "split": metric_row["split"],
        "pair_type": metric_row["pair_type"],
        "record_side": metric_row["record_side"],
        "objective": metric_row["objective"],
        "rank": metric_row["rank"],
        "context_stage": metric_row["context_stage"],
        "layernorm_mode": metric_row["layernorm_mode"],
        "head_layer": metric_row["head_layer"],
        "head": metric_row["head"],
        "head_label": metric_row["head_label"],
        "score_query_role": metric_row["score_query_role"],
        "support_key_role": metric_row["support_key_role"],
        "distractor_key_role": metric_row["distractor_key_role"],
        "group_id": group.group_id,
        "group_kind": group.group_kind,
        "component_type": group.component_type,
        "partition_name": group.partition_name,
        "group_layer": group.layer,
        "group_head": group.head,
        "group_projection": group.projection,
        "group_neuron": group.neuron,
        "selection_count": len(group.selections),
        "num_selected_parameters": int(dot_summary["num_parameters"]),
        "predicted_delta_contribution": contribution,
        "contribution_per_parameter": contribution / float(dot_summary["num_parameters"]),
        "update_score_gradient_cosine": dot_summary["cosine"],
        "parameter_delta_l2_norm": float(dot_summary["left_l2_norm"]),
        "score_gradient_l2_norm": float(dot_summary["right_l2_norm"]),
        "actual_delta": metric_row["actual_delta"],
        "global_predicted_delta": metric_row["predicted_delta"],
        "global_residual": metric_row["residual"],
        "global_relative_error": metric_row["relative_error"],
        "notes": list(group.notes),
    }


def _summarize(
    *,
    metric_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    top_k_groups: int,
) -> dict[str, Any]:
    if not metric_rows:
        raise RuntimeError("Cannot summarize empty rank QK update metric rows.")
    interval_rows = [
        row for row in metric_rows if row["split"] == "__all__" and row["pair_type"] == "__all__"
    ]
    by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in interval_rows:
        by_rank[int(row["rank"])].append(row)
    rank_summaries: list[dict[str, Any]] = []
    for rank, rows in sorted(by_rank.items()):
        rows = sorted(rows, key=lambda row: int(row["source_step"]))
        rank_summaries.append(
            {
                "rank": rank,
                "num_intervals": len(rows),
                "source_step": int(rows[0]["source_step"]),
                "target_step": int(rows[-1]["target_step"]),
                "sum_actual_delta": sum(float(row["actual_delta"]) for row in rows),
                "sum_predicted_delta": sum(float(row["predicted_delta"]) for row in rows),
                "mean_relative_error": _mean([float(row["relative_error"]) for row in rows], label="rank relative error"),
                "sign_match_count": sum(1 for row in rows if bool(row["sign_match"])),
                "sign_match_total": len(rows),
                "sum_support_delta_mean": sum(float(row["support_delta_mean"]) for row in rows),
                "sum_distractor_delta_mean": sum(float(row["distractor_delta_mean"]) for row in rows),
            }
        )
    final_target_step = max(int(row["target_step"]) for row in metric_rows)
    final_metric_rows = [
        row
        for row in metric_rows
        if int(row["target_step"]) == final_target_step
        and row["split"] == "__all__"
        and row["pair_type"] == "__all__"
    ]
    aggregate_contribs: dict[tuple[int, str, str], float] = defaultdict(float)
    for row in decomposition_rows:
        if row["split"] == "__all__" and row["pair_type"] == "__all__":
            aggregate_contribs[(int(row["rank"]), str(row["partition_name"]), str(row["group_id"]))] += float(
                row["predicted_delta_contribution"]
            )
    top_contributions: list[dict[str, Any]] = []
    for (rank, partition, group_id), contribution in sorted(
        aggregate_contribs.items(), key=lambda item: abs(item[1]), reverse=True
    )[:top_k_groups]:
        top_contributions.append(
            {
                "rank": rank,
                "partition_name": partition,
                "group_id": group_id,
                "predicted_delta_contribution": contribution,
            }
        )
    return {
        "intervals": sorted(
            {f"{row['source_step']}->{row['target_step']}" for row in metric_rows},
            key=lambda label: int(label.split("->")[0]),
        ),
        "rank_summaries": rank_summaries,
        "final_target_step": final_target_step,
        "final_metric_rows": final_metric_rows,
        "top_aggregate_contributions_abs": top_contributions,
    }


def _plot_actual_vs_predicted(*, metric_rows: list[dict[str, Any]], output_path: Path) -> Path | None:
    rows = [
        row for row in metric_rows if row["split"] == "__all__" and row["pair_type"] == "__all__"
    ]
    if not rows:
        return None
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6))
    for rank in sorted({int(row["rank"]) for row in rows}):
        rank_rows = [row for row in rows if int(row["rank"]) == rank]
        ax.scatter(
            [float(row["actual_delta"]) for row in rank_rows],
            [float(row["predicted_delta"]) for row in rank_rows],
            label=f"rank {rank}",
        )
    all_values = [float(row["actual_delta"]) for row in rows] + [float(row["predicted_delta"]) for row in rows]
    lo = min(all_values)
    hi = max(all_values)
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("actual delta")
    ax.set_ylabel("predicted first-order delta")
    ax.set_title("Rank QK retrieval-separation update attribution")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
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
    lines = [
        "# Bilinear QK Rank Update Attribution",
        "",
        "This report tests whether actual checkpoint-to-checkpoint parameter movement explains growth of a rank-truncated QK retrieval matcher.",
        "",
        "```text",
        "C_rank(theta) = mean score_rank(prediction, support_value) - mean score_rank(prediction, value_distractors)",
        "score_rank(x, y) = x.T @ P_U @ W_Q.T @ W_K @ P_V @ y / sqrt(head_dim)",
        "P_U and P_V are fixed from the source checkpoint rank-k SVD basis for each interval.",
        "predicted_delta = grad_theta C_rank(theta_source) . (theta_target - theta_source)",
        "```",
        "",
        "## Run",
        "",
        f"- head: `L{report['head_layer']}H{report['head']}`",
        f"- ranks: `{report['ranks']}`",
        f"- context stage: `{report['context_stage']}`",
        f"- layernorm mode: `{report['layernorm_mode']}`",
        f"- query role: `{report['score_query_role']}`",
        f"- support role: `{report['support_key_role']}`",
        f"- distractor role: `{report['distractor_key_role']}`",
        f"- record sides: `{report['record_sides']}`",
        f"- intervals: `{report['summary']['intervals']}`",
        "",
        "## Cumulative Summary",
        "",
        "| rank | actual delta | predicted delta | sign match | mean relative error | support delta | distractor delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]["rank_summaries"]:
        lines.append(
            "| {rank} | {actual:.6g} | {predicted:.6g} | {sign}/{total} | {error:.6g} | {support:.6g} | {distractor:.6g} |".format(
                rank=int(row["rank"]),
                actual=float(row["sum_actual_delta"]),
                predicted=float(row["sum_predicted_delta"]),
                sign=int(row["sign_match_count"]),
                total=int(row["sign_match_total"]),
                error=float(row["mean_relative_error"]),
                support=float(row["sum_support_delta_mean"]),
                distractor=float(row["sum_distractor_delta_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Final Interval Rows",
            "",
            "| rank | actual delta | predicted delta | residual | relative error | sign match |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["summary"]["final_metric_rows"]:
        lines.append(
            "| {rank} | {actual:.6g} | {predicted:.6g} | {residual:.6g} | {error:.6g} | `{sign}` |".format(
                rank=int(row["rank"]),
                actual=float(row["actual_delta"]),
                predicted=float(row["predicted_delta"]),
                residual=float(row["residual"]),
                error=float(row["relative_error"]),
                sign=bool(row["sign_match"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Aggregate Contributions By Absolute Value",
            "",
            "| rank | partition | group | contribution |",
            "|---:|---|---|---:|",
        ]
    )
    for row in report["summary"]["top_aggregate_contributions_abs"]:
        lines.append(
            "| {rank} | `{partition}` | `{group}` | {contribution:.6g} |".format(
                rank=int(row["rank"]),
                partition=row["partition_name"],
                group=row["group_id"],
                contribution=float(row["predicted_delta_contribution"]),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- metric rows: `{report['metric_rows_path']}`",
            f"- decomposition rows: `{report['decomposition_rows_path']}`",
            f"- group rows: `{report['group_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
            f"- score rows: `{report['score_rows_path']}`",
        ]
    )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for label, plot_path in plot_paths.items():
            lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bilinear_qk_rank_update_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    head_layer: int,
    head: int,
    ranks: list[int],
    context_stage: str,
    layernorm_mode: str,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    pair_types: list[str],
    device_name: str = "cpu",
    checkpoint_paths: list[Path] | None = None,
    record_sides: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    decomposition_modes: list[str] | None = None,
    top_k_groups: int = 40,
    min_error_denominator: float = 1.0e-9,
) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if not ranks:
        raise ValueError("At least one --rank is required.")
    resolved_ranks = sorted(set(ranks))
    if any(rank <= 0 for rank in resolved_ranks):
        raise ValueError(f"Ranks must be positive integers, got {resolved_ranks}.")
    unsupported_roles = [
        role
        for role in [score_query_role, support_key_role, distractor_key_role]
        if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported attention roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if support_key_role == distractor_key_role:
        raise ValueError("support_key_role and distractor_key_role must be different.")
    if layernorm_mode not in LAYER_NORM_MODES:
        raise ValueError(f"Unsupported layernorm_mode {layernorm_mode!r}; expected one of {LAYER_NORM_MODES}.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")

    resolved_record_sides = list(ATTENTION_SCORE_RECORD_SIDES) if record_sides is None else sorted(set(record_sides), key=record_sides.index)
    unsupported_sides = [side for side in resolved_record_sides if side not in ATTENTION_SCORE_RECORD_SIDES]
    if unsupported_sides:
        raise ValueError(f"Unsupported record sides {unsupported_sides}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    resolved_decomposition_modes = _resolve_route_gradient_decomposition_modes(decomposition_modes)
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("bilinear-qk-rank-update-attribution requires at least two checkpoints.")
    device = require_device(device_name)
    source_model = build_model(spec.model, len(vocab.tokens), device)
    target_model = build_model(spec.model, len(vocab.tokens), device)
    if head_layer < 0 or head_layer >= len(source_model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(source_model.blocks) - 1}.")
    if head < 0 or head >= source_model.blocks[head_layer].attn.n_heads:
        raise ValueError(
            f"head {head} outside model range 0..{source_model.blocks[head_layer].attn.n_heads - 1} for layer {head_layer}."
        )
    valid_stages = ["embedding"]
    for layer_index in range(len(source_model.blocks)):
        valid_stages.extend([f"layer_{layer_index}_post_attn", f"layer_{layer_index}_post_mlp"])
    valid_stages.append("final_norm")
    if context_stage not in valid_stages:
        raise ValueError(f"Unsupported context_stage {context_stage!r}; expected one of {valid_stages}.")
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Rank QK update attribution constructed no pairs.")
    groups, decomposition_summary = _build_route_gradient_decomposition_groups(
        model=source_model,
        decomposition_modes=resolved_decomposition_modes,
    )
    group_rows = [
        _group_metadata(
            model_parameters=dict(source_model.named_parameters(remove_duplicate=False)),
            group=group,
        )
        for group in groups
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "bilinear_qk_rank_update_attribution_report.json"
    markdown_path = output_dir / "bilinear_qk_rank_update_attribution_report.md"
    metric_rows_path = output_dir / "bilinear_qk_rank_update_attribution_rows.jsonl"
    decomposition_rows_path = output_dir / "bilinear_qk_rank_update_attribution_decomposition_rows.jsonl"
    group_rows_path = output_dir / "bilinear_qk_rank_update_attribution_groups.jsonl"
    score_rows_path = output_dir / "bilinear_qk_rank_update_attribution_score_rows.jsonl"
    pair_rows_path = output_dir / "bilinear_qk_rank_update_attribution_pairs.jsonl"
    progress_path = output_dir / "bilinear_qk_rank_update_attribution_progress.json"
    for partial_path in (
        metric_rows_path,
        decomposition_rows_path,
        group_rows_path,
        score_rows_path,
        pair_rows_path,
        progress_path,
    ):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])
    write_jsonl(group_rows_path, group_rows)

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[bilinear-qk-rank-update-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(pairs)} ranks={resolved_ranks} "
        f"pair_types={pair_types} device={device_name} head=L{head_layer}H{head} stage={context_stage} "
        f"query={score_query_role} support={support_key_role} distractor={distractor_key_role} "
        f"record_sides={resolved_record_sides} groups={len(groups)}",
        flush=True,
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_decomposition_rows: list[dict[str, Any]] = []
    pair_groups = _route_gradient_groups(pairs)
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        learning_rate = _compute_learning_rate(spec.optimization, source_step)
        print(
            "[bilinear-qk-rank-update-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )
        source_checkpoint = load_checkpoint(source_checkpoint_path, device)
        target_checkpoint = load_checkpoint(target_checkpoint_path, device)
        if int(source_checkpoint["step"]) != source_step:
            raise RuntimeError(f"Source checkpoint step mismatch: payload={source_checkpoint['step']} path={source_step}")
        if int(target_checkpoint["step"]) != target_step:
            raise RuntimeError(f"Target checkpoint step mismatch: payload={target_checkpoint['step']} path={target_step}")
        load_model_state(source_model, source_checkpoint["model_state"])
        load_model_state(target_model, target_checkpoint["model_state"])
        source_parameters = _model_parameter_snapshot(source_model)
        target_parameters = _model_parameter_snapshot(target_model)
        delta_parameters = _parameter_delta(
            source_parameters=source_parameters,
            target_parameters=target_parameters,
            label=f"rank QK update {source_step}->{target_step}",
        )
        parameter_delta_l2_norm = _parameter_l2_norm(delta_parameters)

        for rank in resolved_ranks:
            left_basis, right_basis, singular_summary = _rank_qk_source_basis(
                model=source_model,
                head_layer=head_layer,
                head=head,
                rank=rank,
            )
            for record_side in resolved_record_sides:
                source_all = _rank_match_payload_for_pairs(
                    model=source_model,
                    pairs=pairs,
                    head_layer=head_layer,
                    head=head,
                    rank=rank,
                    left_basis=left_basis,
                    right_basis=right_basis,
                    singular_summary=singular_summary,
                    context_stage=context_stage,
                    layernorm_mode=layernorm_mode,
                    score_query_role=score_query_role,
                    support_key_role=support_key_role,
                    distractor_key_role=distractor_key_role,
                    record_side=record_side,
                    batch_size=spec.evaluation.batch_size,
                    pad_token_id=vocab.pad_token_id,
                    device=device,
                    track_grad=False,
                )
                target_all = _rank_match_payload_for_pairs(
                    model=target_model,
                    pairs=pairs,
                    head_layer=head_layer,
                    head=head,
                    rank=rank,
                    left_basis=left_basis,
                    right_basis=right_basis,
                    singular_summary=singular_summary,
                    context_stage=context_stage,
                    layernorm_mode=layernorm_mode,
                    score_query_role=score_query_role,
                    support_key_role=support_key_role,
                    distractor_key_role=distractor_key_role,
                    record_side=record_side,
                    batch_size=spec.evaluation.batch_size,
                    pad_token_id=vocab.pad_token_id,
                    device=device,
                    track_grad=False,
                )
                for source_pair_row, target_pair_row in zip(
                    source_all["pair_rows"], target_all["pair_rows"], strict=True
                ):
                    if source_pair_row["pair_id"] != target_pair_row["pair_id"]:
                        raise RuntimeError(
                            f"Rank QK pair row mismatch: {source_pair_row['pair_id']} vs {target_pair_row['pair_id']}"
                        )
                    append_jsonl(
                        score_rows_path,
                        {
                            "source_step": source_step,
                            "target_step": target_step,
                            "rank": rank,
                            "context_stage": context_stage,
                            "layernorm_mode": layernorm_mode,
                            "head_layer": head_layer,
                            "head": head,
                            "record_side": record_side,
                            "pair_id": source_pair_row["pair_id"],
                            "split": source_pair_row["split"],
                            "pair_type": source_pair_row["pair_type"],
                            "source_value": source_pair_row["rank_match_separation"],
                            "target_value": target_pair_row["rank_match_separation"],
                            "actual_delta": float(target_pair_row["rank_match_separation"])
                            - float(source_pair_row["rank_match_separation"]),
                            "source_support_score_mean": source_pair_row["support_score_mean"],
                            "target_support_score_mean": target_pair_row["support_score_mean"],
                            "source_distractor_score_mean": source_pair_row["distractor_score_mean"],
                            "target_distractor_score_mean": target_pair_row["distractor_score_mean"],
                            "source_rank_match_max_margin": source_pair_row["rank_match_max_margin"],
                            "target_rank_match_max_margin": target_pair_row["rank_match_max_margin"],
                            "source_support_beats_all_distractors": source_pair_row["support_beats_all_distractors"],
                            "target_support_beats_all_distractors": target_pair_row["support_beats_all_distractors"],
                        },
                    )
                for (split, pair_type), group_pairs in sorted(pair_groups.items()):
                    actual = _actual_summary(
                        source_rows=source_all["pair_rows"],
                        target_rows=target_all["pair_rows"],
                        split=split,
                        pair_type=pair_type,
                    )
                    source_payload = _rank_match_payload_for_pairs(
                        model=source_model,
                        pairs=group_pairs,
                        head_layer=head_layer,
                        head=head,
                        rank=rank,
                        left_basis=left_basis,
                        right_basis=right_basis,
                        singular_summary=singular_summary,
                        context_stage=context_stage,
                        layernorm_mode=layernorm_mode,
                        score_query_role=score_query_role,
                        support_key_role=support_key_role,
                        distractor_key_role=distractor_key_role,
                        record_side=record_side,
                        batch_size=spec.evaluation.batch_size,
                        pad_token_id=vocab.pad_token_id,
                        device=device,
                        track_grad=True,
                    )
                    if abs(float(source_payload["score_value"]) - float(actual["source_value"])) > 1.0e-4:
                        raise RuntimeError(
                            "Rank QK source objective mismatch for "
                            f"{source_step}->{target_step} rank={rank} split={split} pair_type={pair_type}: "
                            f"gradient={source_payload['score_value']} actual={actual['source_value']}"
                        )
                    gradients = source_payload["gradients"]
                    if not isinstance(gradients, dict):
                        raise TypeError("Rank QK payload gradients must be a dictionary.")
                    dot_summary = _gradient_dot_summary(
                        left_gradients=delta_parameters,
                        right_gradients=gradients,
                        label=f"rank QK update {source_step}->{target_step} rank={rank}/{split}/{pair_type}/{record_side}",
                    )
                    metric_row = _metric_row(
                        source_step=source_step,
                        target_step=target_step,
                        source_checkpoint=source_checkpoint_path,
                        target_checkpoint=target_checkpoint_path,
                        learning_rate=learning_rate,
                        split=split,
                        pair_type=pair_type,
                        record_side=record_side,
                        rank=rank,
                        context_stage=context_stage,
                        layernorm_mode=layernorm_mode,
                        head_layer=head_layer,
                        head=head,
                        score_query_role=score_query_role,
                        support_key_role=support_key_role,
                        distractor_key_role=distractor_key_role,
                        actual=actual,
                        source_payload=source_payload,
                        dot_summary=dot_summary,
                        parameter_delta_l2_norm=parameter_delta_l2_norm,
                        min_error_denominator=min_error_denominator,
                    )
                    append_jsonl(metric_rows_path, metric_row)
                    all_metric_rows.append(metric_row)
                    for group in groups:
                        group_dot_summary = _gradient_dot_summary_for_group(
                            left_gradients=delta_parameters,
                            right_gradients=gradients,
                            group=group,
                            label=(
                                f"rank QK update {source_step}->{target_step} "
                                f"rank={rank}/{split}/{pair_type}/{record_side}/{group.group_id}"
                            ),
                        )
                        decomposition_row = _decomposition_row(
                            metric_row=metric_row,
                            group=group,
                            dot_summary=group_dot_summary,
                        )
                        append_jsonl(decomposition_rows_path, decomposition_row)
                        all_decomposition_rows.append(decomposition_row)
        primary = next(
            row
            for row in all_metric_rows
            if int(row["source_step"]) == source_step
            and int(row["target_step"]) == target_step
            and int(row["rank"]) == resolved_ranks[-1]
            and row["split"] == "__all__"
            and row["pair_type"] == "__all__"
            and row["record_side"] == resolved_record_sides[0]
        )
        print(
            "[bilinear-qk-rank-update-attribution] finished "
            f"{source_step}->{target_step} rank={primary['rank']} "
            f"actual_delta={float(primary['actual_delta']):.6g} "
            f"predicted_delta={float(primary['predicted_delta']):.6g} "
            f"relative_error={float(primary['relative_error']):.6g} "
            f"sign_match={primary['sign_match']}",
            flush=True,
        )
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_intervals": interval_index,
                "total_intervals": len(intervals),
                "last_source_step": source_step,
                "last_target_step": target_step,
                "metric_rows_path": str(metric_rows_path),
                "decomposition_rows_path": str(decomposition_rows_path),
                "score_rows_path": str(score_rows_path),
            },
        )

    summary = _summarize(
        metric_rows=all_metric_rows,
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
    )
    plot_paths: dict[str, Path] = {}
    plot_path = _plot_actual_vs_predicted(
        metric_rows=all_metric_rows,
        output_path=output_dir / "bilinear_qk_rank_update_actual_vs_predicted.svg",
    )
    if plot_path is not None:
        plot_paths["actual_vs_predicted"] = plot_path
    report = {
        "schema_version": BILINEAR_QK_RANK_UPDATE_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "probe_metadata": probe_metadata,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "output_dir": str(output_dir),
        "device": device_name,
        "head_layer": head_layer,
        "head": head,
        "ranks": resolved_ranks,
        "context_stage": context_stage,
        "layernorm_mode": layernorm_mode,
        "score_query_role": score_query_role,
        "support_key_role": support_key_role,
        "distractor_key_role": distractor_key_role,
        "record_sides": resolved_record_sides,
        "pair_types": pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "decomposition": decomposition_summary,
        "decomposition_modes": resolved_decomposition_modes,
        "top_k_groups": top_k_groups,
        "min_error_denominator": min_error_denominator,
        "pair_construction": pair_construction,
        "calculation": {
            "rank_qk_matrix": "source-checkpoint fixed-basis projection P_U @ W_Q.T @ W_K @ P_V for the selected head",
            "score": "x_query.T @ source_fixed_rank_qk_matrix @ x_key / sqrt(head_dim)",
            "retrieval_separation": "mean support score minus mean distractor score, computed per pair then averaged",
            "predicted_delta": "grad_theta retrieval_separation(theta_source) . (theta_target - theta_source)",
            "basis_mode": "source_checkpoint_fixed_svd_basis",
            "basis_reason": "Moving-SVD gradients are ill-conditioned when singular values are close; this measures local growth inside the source checkpoint rank subspace.",
        },
        "metric_rows_path": str(metric_rows_path),
        "decomposition_rows_path": str(decomposition_rows_path),
        "group_rows_path": str(group_rows_path),
        "score_rows_path": str(score_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_intervals": len(intervals),
            "total_intervals": len(intervals),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "metric_rows_path": str(metric_rows_path),
            "decomposition_rows_path": str(decomposition_rows_path),
            "score_rows_path": str(score_rows_path),
        },
    )
    print(f"[bilinear-qk-rank-update-attribution] complete report={report_path} rows={metric_rows_path}", flush=True)
    return (
        report_path,
        markdown_path,
        metric_rows_path,
        decomposition_rows_path,
        group_rows_path,
        score_rows_path,
        pair_rows_path,
        plot_paths,
    )
