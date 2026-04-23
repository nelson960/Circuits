from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from circuit.analysis.bilinear_qk_rank_update_attribution import (
    LAYER_NORM_MODES,
    _assert_finite_gradients,
    _rank_match_payload_for_pairs,
    _rank_qk_source_basis,
)
from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    ATTENTION_SCORE_RECORD_SIDES,
    GEOMETRY_POSITION_ROLES,
    ROUTE_GRADIENT_LOSS_SIDES,
    _build_causal_patch_pairs,
    _checkpoint_step_from_path,
    _compute_loss_gradient_for_records,
    _gradient_dot_summary,
    _group_pairs_for_data_update,
    _holdout_pair_set,
    _loss_records_for_pairs,
    _model_parameter_snapshot,
    _pair_metadata,
    _parameter_delta,
    _parameter_gradients,
    _resolve_checkpoint_paths,
    _route_objective_pairs,
    _safe_ratio,
    _sign_match,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import append_jsonl, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.train import _compute_learning_rate
from circuit.vocab import Vocabulary


BILINEAR_QK_RANK_DATA_ATTRIBUTION_SCHEMA_VERSION = 1
LOSS_SCOPES = ["full_lm", "answer"]


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise RuntimeError(f"Cannot compute mean for empty values: {label}")
    return sum(values) / len(values)


def _rank_route_row(
    *,
    source_step: int,
    target_step: int,
    source_checkpoint: Path,
    target_checkpoint: Path,
    learning_rate: float,
    route_split: str,
    route_pair_type: str,
    record_side: str,
    rank: int,
    context_stage: str,
    layernorm_mode: str,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    source_payload: dict[str, Any],
    target_payload: dict[str, Any],
    actual_update_dot_summary: dict[str, Any],
    min_error_denominator: float,
) -> dict[str, Any]:
    source_value = float(source_payload["score_value"])
    target_value = float(target_payload["score_value"])
    actual_delta = target_value - source_value
    predicted_delta = float(actual_update_dot_summary["dot"])
    residual = actual_delta - predicted_delta
    relative_error_denominator = max(abs(actual_delta), min_error_denominator)
    return {
        "source_step": source_step,
        "target_step": target_step,
        "step_gap": target_step - source_step,
        "source_checkpoint": str(source_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "learning_rate": learning_rate,
        "route_split": route_split,
        "route_pair_type": route_pair_type,
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
        "source_rank_match_score": source_value,
        "target_rank_match_score": target_value,
        "actual_rank_match_delta": actual_delta,
        "actual_update_predicted_rank_match_delta": predicted_delta,
        "actual_update_rank_match_residual": residual,
        "actual_update_rank_match_relative_error": abs(residual) / relative_error_denominator,
        "actual_update_rank_match_sign_match": _sign_match(actual_delta, predicted_delta),
        "source_support_score_mean": float(source_payload["support_score_value_mean"]),
        "source_distractor_score_mean": float(source_payload["distractor_score_value_mean"]),
        "target_support_score_mean": float(target_payload["support_score_value_mean"]),
        "target_distractor_score_mean": float(target_payload["distractor_score_value_mean"]),
        "support_score_delta_mean": float(target_payload["support_score_value_mean"])
        - float(source_payload["support_score_value_mean"]),
        "distractor_score_delta_mean": float(target_payload["distractor_score_value_mean"])
        - float(source_payload["distractor_score_value_mean"]),
        "num_route_pairs": int(source_payload["num_pairs"]),
        "num_route_scores": int(source_payload["num_scores"]),
        "parameter_delta_l2_norm": float(actual_update_dot_summary["left_l2_norm"]),
        "rank_gradient_l2_norm": float(actual_update_dot_summary["right_l2_norm"]),
        "actual_update_rank_gradient_cosine": actual_update_dot_summary["cosine"],
        "num_parameters": int(actual_update_dot_summary["num_parameters"]),
        "zero_rank_gradient_parameter_count": len(source_payload["zero_gradient_parameter_names"]),
        "zero_rank_gradient_parameter_names": source_payload["zero_gradient_parameter_names"],
        **source_payload["singular_summary"],
    }


def _compute_answer_loss_gradient_for_records(
    *,
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty for answer-loss-gradient computation.")
    model.eval()
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_answers = 0
    num_batches = 0
    for start_index in range(0, len(records), batch_size):
        batch_records = records[start_index : start_index + batch_size]
        batch = move_batch_to_device(collate_symbolic_kv(batch_records, pad_token_id), device)
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        answer_logits, answer_targets, _ = extract_answer_logits(outputs.logits, batch)
        if answer_targets.numel() <= 0:
            raise RuntimeError("Answer-loss-gradient batch produced no answer targets.")
        loss_sum = torch.nn.functional.cross_entropy(answer_logits, answer_targets, reduction="sum")
        loss_sum.backward()
        total_loss += float(loss_sum.detach().cpu().item())
        total_answers += int(answer_targets.numel())
        num_batches += 1
    if total_answers <= 0:
        raise RuntimeError("Answer-loss-gradient records produced no answer targets.")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(float(total_answers))
    gradients, zero_gradient_parameter_names = _parameter_gradients(model=model, require_all=True)
    if zero_gradient_parameter_names:
        raise RuntimeError(f"Answer loss gradient unexpectedly had zero-gradient parameters: {zero_gradient_parameter_names}")
    _assert_finite_gradients(gradients, label="answer loss")
    model.zero_grad(set_to_none=True)
    return {
        "loss": total_loss / float(total_answers),
        "num_tokens": total_answers,
        "num_records": len(records),
        "num_batches": num_batches,
        "loss_scope": "answer",
        "gradients": gradients,
    }


def _compute_loss_gradient_for_records_by_scope(
    *,
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    loss_scope: str,
) -> dict[str, Any]:
    if loss_scope == "full_lm":
        payload = _compute_loss_gradient_for_records(
            model=model,
            records=records,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            device=device,
        )
        payload["loss_scope"] = "full_lm"
        return payload
    if loss_scope == "answer":
        return _compute_answer_loss_gradient_for_records(
            model=model,
            records=records,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            device=device,
        )
    raise ValueError(f"Unsupported loss_scope {loss_scope!r}; expected one of {LOSS_SCOPES}.")


def _data_row(
    *,
    route_row: dict[str, Any],
    data_group_id: str,
    data_group_values: dict[str, str],
    loss_side: str,
    loss_scope: str,
    loss_payload: dict[str, Any],
    loss_rank_dot_summary: dict[str, Any],
    loss_update_dot_summary: dict[str, Any],
) -> dict[str, Any]:
    loss_dot_rank = float(loss_rank_dot_summary["dot"])
    negative_loss_dot_rank = -loss_dot_rank
    loss_dot_update = float(loss_update_dot_summary["dot"])
    negative_loss_dot_update = -loss_dot_update
    loss_gradient_l2_norm = float(loss_rank_dot_summary["left_l2_norm"])
    rank_gradient_l2_norm = float(loss_rank_dot_summary["right_l2_norm"])
    parameter_delta_l2_norm = float(loss_update_dot_summary["right_l2_norm"])
    learning_rate = float(route_row["learning_rate"])
    return {
        "source_step": int(route_row["source_step"]),
        "target_step": int(route_row["target_step"]),
        "step_gap": int(route_row["step_gap"]),
        "source_checkpoint": route_row["source_checkpoint"],
        "target_checkpoint": route_row["target_checkpoint"],
        "learning_rate": learning_rate,
        "route_split": route_row["route_split"],
        "route_pair_type": route_row["route_pair_type"],
        "record_side": route_row["record_side"],
        "objective": route_row["objective"],
        "rank": int(route_row["rank"]),
        "context_stage": route_row["context_stage"],
        "layernorm_mode": route_row["layernorm_mode"],
        "head_layer": int(route_row["head_layer"]),
        "head": int(route_row["head"]),
        "head_label": route_row["head_label"],
        "score_query_role": route_row["score_query_role"],
        "support_key_role": route_row["support_key_role"],
        "distractor_key_role": route_row["distractor_key_role"],
        "data_group_id": data_group_id,
        "data_group_values": data_group_values,
        "loss_side": loss_side,
        "loss_scope": loss_scope,
        "loss": float(loss_payload["loss"]),
        "loss_num_records": int(loss_payload["num_records"]),
        "loss_num_tokens": int(loss_payload["num_tokens"]),
        "source_rank_match_score": float(route_row["source_rank_match_score"]),
        "target_rank_match_score": float(route_row["target_rank_match_score"]),
        "actual_rank_match_delta": float(route_row["actual_rank_match_delta"]),
        "actual_update_predicted_rank_match_delta": float(route_row["actual_update_predicted_rank_match_delta"]),
        "actual_update_rank_match_residual": float(route_row["actual_update_rank_match_residual"]),
        "actual_update_rank_match_relative_error": float(route_row["actual_update_rank_match_relative_error"]),
        "actual_update_rank_match_sign_match": bool(route_row["actual_update_rank_match_sign_match"]),
        "loss_gradient_l2_norm": loss_gradient_l2_norm,
        "rank_gradient_l2_norm": rank_gradient_l2_norm,
        "parameter_delta_l2_norm": parameter_delta_l2_norm,
        "loss_dot_rank_gradient": loss_dot_rank,
        "negative_loss_dot_rank_gradient": negative_loss_dot_rank,
        "loss_negative_rank_gradient_cosine": _safe_ratio(
            negative_loss_dot_rank,
            loss_gradient_l2_norm * rank_gradient_l2_norm,
        ),
        "sgd_equivalent_rank_match_delta_linearized": learning_rate * negative_loss_dot_rank,
        "loss_dot_actual_update": loss_dot_update,
        "negative_loss_dot_actual_update": negative_loss_dot_update,
        "loss_reduction_under_actual_update_linearized": negative_loss_dot_update,
        "loss_negative_actual_update_cosine": _safe_ratio(
            negative_loss_dot_update,
            loss_gradient_l2_norm * parameter_delta_l2_norm,
        ),
        "loss_rank_gradient_cosine": loss_rank_dot_summary["cosine"],
        "loss_actual_update_cosine": loss_update_dot_summary["cosine"],
    }


def _summarize(
    *,
    route_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    top_k_data_groups: int,
) -> dict[str, Any]:
    if not route_rows:
        raise RuntimeError("Cannot summarize bilinear QK rank data attribution without route rows.")
    if not data_rows:
        raise RuntimeError("Cannot summarize bilinear QK rank data attribution without data rows.")
    route_by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in route_rows:
        route_by_rank[int(row["rank"])].append(row)
    route_summaries: list[dict[str, Any]] = []
    for rank, rows in sorted(route_by_rank.items()):
        rows = sorted(rows, key=lambda row: int(row["source_step"]))
        route_summaries.append(
            {
                "rank": rank,
                "num_intervals": len(rows),
                "source_step": int(rows[0]["source_step"]),
                "target_step": int(rows[-1]["target_step"]),
                "sum_actual_rank_match_delta": sum(float(row["actual_rank_match_delta"]) for row in rows),
                "sum_actual_update_predicted_rank_match_delta": sum(
                    float(row["actual_update_predicted_rank_match_delta"]) for row in rows
                ),
                "mean_actual_update_rank_match_relative_error": _mean(
                    [float(row["actual_update_rank_match_relative_error"]) for row in rows],
                    label="actual update rank match relative error",
                ),
                "actual_update_sign_match_count": sum(
                    1 for row in rows if bool(row["actual_update_rank_match_sign_match"])
                ),
                "actual_update_sign_match_total": len(rows),
            }
        )
    data_by_rank_group: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in data_rows:
        data_by_rank_group[(int(row["rank"]), str(row["data_group_id"]))].append(row)
    cumulative_data_rows: list[dict[str, Any]] = []
    for (rank, data_group_id), rows in sorted(data_by_rank_group.items()):
        rows = sorted(rows, key=lambda row: int(row["source_step"]))
        first_values = rows[0]["data_group_values"]
        for row in rows[1:]:
            if row["data_group_values"] != first_values:
                raise RuntimeError(f"Data group values changed across intervals for {rank}/{data_group_id}.")
        cumulative_data_rows.append(
            {
                "rank": rank,
                "data_group_id": data_group_id,
                "data_group_values": first_values,
                "num_intervals": len(rows),
                "sum_negative_loss_dot_rank_gradient": sum(
                    float(row["negative_loss_dot_rank_gradient"]) for row in rows
                ),
                "sum_sgd_equivalent_rank_match_delta_linearized": sum(
                    float(row["sgd_equivalent_rank_match_delta_linearized"]) for row in rows
                ),
                "sum_loss_reduction_under_actual_update_linearized": sum(
                    float(row["loss_reduction_under_actual_update_linearized"]) for row in rows
                ),
                "mean_loss_negative_rank_gradient_cosine": _mean(
                    [
                        float(row["loss_negative_rank_gradient_cosine"])
                        for row in rows
                        if row["loss_negative_rank_gradient_cosine"] is not None
                    ],
                    label=f"loss negative rank gradient cosine {rank}/{data_group_id}",
                ),
                "mean_loss_negative_actual_update_cosine": _mean(
                    [
                        float(row["loss_negative_actual_update_cosine"])
                        for row in rows
                        if row["loss_negative_actual_update_cosine"] is not None
                    ],
                    label=f"loss negative actual update cosine {rank}/{data_group_id}",
                ),
                "mean_loss": _mean([float(row["loss"]) for row in rows], label=f"loss {rank}/{data_group_id}"),
                "mean_loss_num_records": _mean(
                    [float(row["loss_num_records"]) for row in rows],
                    label=f"loss records {rank}/{data_group_id}",
                ),
            }
        )
    non_all = [row for row in cumulative_data_rows if row["data_group_id"] != "__all__"]
    return {
        "intervals": sorted(
            {f"{row['source_step']}->{row['target_step']}" for row in route_rows},
            key=lambda label: int(label.split("->")[0]),
        ),
        "route_summaries": route_summaries,
        "cumulative_data_rows": cumulative_data_rows,
        "top_cumulative_rank_support": sorted(
            non_all,
            key=lambda row: float(row["sum_negative_loss_dot_rank_gradient"]),
            reverse=True,
        )[:top_k_data_groups],
        "top_cumulative_rank_conflict": sorted(
            non_all,
            key=lambda row: float(row["sum_negative_loss_dot_rank_gradient"]),
        )[:top_k_data_groups],
        "top_cumulative_actual_update_alignment": sorted(
            non_all,
            key=lambda row: float(row["sum_loss_reduction_under_actual_update_linearized"]),
            reverse=True,
        )[:top_k_data_groups],
    }


def _write_markdown(
    *,
    path: Path,
    report: dict[str, Any],
) -> None:
    summary = report["summary"]
    lines = [
        "# Bilinear QK Rank Data Attribution",
        "",
        "This report asks which data-group loss gradients support the fixed-basis low-rank QK matcher.",
        "",
        "```text",
        "C_rank(theta) = mean score_rank(prediction, support_value) - mean score_rank(prediction, value_distractors)",
        "data_rank_support_g = < -grad loss_g(theta_source), grad C_rank(theta_source) >",
        "sgd_equivalent_delta_g = learning_rate * data_rank_support_g",
        "actual_update_alignment_g = < -grad loss_g(theta_source), Delta theta >",
        "```",
        "",
        "This is a source-checkpoint diagnostic, not an exact replay of historical optimizer batches.",
        "",
        "## Run",
        "",
        f"- head: `L{report['head_layer']}H{report['head']}`",
        f"- ranks: `{report['ranks']}`",
        f"- context stage: `{report['context_stage']}`",
        f"- layernorm mode: `{report['layernorm_mode']}`",
        f"- route pair type: `{report['route_pair_type']}`",
        f"- data pair types: `{report['data_pair_types']}`",
        f"- data group fields: `{report['data_group_fields']}`",
        f"- loss side: `{report['loss_side']}`",
        f"- loss scope: `{report['loss_scope']}`",
        f"- intervals: `{summary['intervals']}`",
        "",
        "## Route Summary",
        "",
        "| rank | actual delta | update-predicted delta | sign match | mean relative error |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["route_summaries"]:
        lines.append(
            "| {rank} | {actual:.6g} | {predicted:.6g} | {sign}/{total} | {error:.6g} |".format(
                rank=int(row["rank"]),
                actual=float(row["sum_actual_rank_match_delta"]),
                predicted=float(row["sum_actual_update_predicted_rank_match_delta"]),
                sign=int(row["actual_update_sign_match_count"]),
                total=int(row["actual_update_sign_match_total"]),
                error=float(row["mean_actual_update_rank_match_relative_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Cumulative Data Groups",
            "",
            "| rank | data group | route support | SGD-equivalent delta | actual-update loss reduction | mean route cosine |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["cumulative_data_rows"]:
        lines.append(
            "| {rank} | `{group}` | {support:.6g} | {delta:.6g} | {update:.6g} | {cosine:.6g} |".format(
                rank=int(row["rank"]),
                group=row["data_group_id"],
                support=float(row["sum_negative_loss_dot_rank_gradient"]),
                delta=float(row["sum_sgd_equivalent_rank_match_delta_linearized"]),
                update=float(row["sum_loss_reduction_under_actual_update_linearized"]),
                cosine=float(row["mean_loss_negative_rank_gradient_cosine"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Cumulative Rank Support",
            "",
            "| rank | data group | route support | SGD-equivalent delta | actual-update loss reduction |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in summary["top_cumulative_rank_support"]:
        lines.append(
            "| {rank} | `{group}` | {support:.6g} | {delta:.6g} | {update:.6g} |".format(
                rank=int(row["rank"]),
                group=row["data_group_id"],
                support=float(row["sum_negative_loss_dot_rank_gradient"]),
                delta=float(row["sum_sgd_equivalent_rank_match_delta_linearized"]),
                update=float(row["sum_loss_reduction_under_actual_update_linearized"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Cumulative Rank Conflict",
            "",
            "| rank | data group | route support | SGD-equivalent delta | actual-update loss reduction |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in summary["top_cumulative_rank_conflict"]:
        lines.append(
            "| {rank} | `{group}` | {support:.6g} | {delta:.6g} | {update:.6g} |".format(
                rank=int(row["rank"]),
                group=row["data_group_id"],
                support=float(row["sum_negative_loss_dot_rank_gradient"]),
                delta=float(row["sum_sgd_equivalent_rank_match_delta_linearized"]),
                update=float(row["sum_loss_reduction_under_actual_update_linearized"]),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- route rows: `{report['route_rows_path']}`",
            f"- data rows: `{report['data_rows_path']}`",
            f"- route pair rows: `{report['route_pair_rows_path']}`",
            f"- data pair rows: `{report['data_pair_rows_path']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_bilinear_qk_rank_data_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    data_probe_set_path: Path,
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
    route_pair_types: list[str],
    route_pair_type: str,
    data_pair_types: list[str],
    data_group_fields: list[str],
    device_name: str = "cpu",
    checkpoint_paths: list[Path] | None = None,
    record_side: str = "clean",
    route_split_filter: list[str] | None = None,
    data_split_filter: list[str] | None = None,
    route_split: str = "__all__",
    max_route_pairs_per_type: int = 64,
    min_route_pairs_per_type: int = 1,
    max_data_pairs_per_type: int = 64,
    min_data_pairs_per_type: int = 1,
    loss_side: str = "clean",
    loss_scope: str = "full_lm",
    top_k_data_groups: int = 24,
    min_error_denominator: float = 1.0e-9,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    if not ranks:
        raise ValueError("At least one --rank is required.")
    resolved_ranks = sorted(set(ranks))
    if any(rank <= 0 for rank in resolved_ranks):
        raise ValueError(f"Ranks must be positive integers, got {resolved_ranks}.")
    if record_side not in ATTENTION_SCORE_RECORD_SIDES:
        raise ValueError(f"Unsupported record_side {record_side!r}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    if layernorm_mode not in LAYER_NORM_MODES:
        raise ValueError(f"Unsupported layernorm_mode {layernorm_mode!r}; expected one of {LAYER_NORM_MODES}.")
    unsupported_roles = [
        role
        for role in [score_query_role, support_key_role, distractor_key_role]
        if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported position roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if loss_side not in ROUTE_GRADIENT_LOSS_SIDES:
        raise ValueError(f"Unsupported loss_side {loss_side!r}; expected one of {ROUTE_GRADIENT_LOSS_SIDES}.")
    if loss_scope not in LOSS_SCOPES:
        raise ValueError(f"Unsupported loss_scope {loss_scope!r}; expected one of {LOSS_SCOPES}.")
    if not data_group_fields:
        raise ValueError("At least one --data-group-field is required.")
    if top_k_data_groups <= 0:
        raise ValueError("top_k_data_groups must be positive.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")

    spec = TrainSpec.from_path(config_path)
    route_probe_records, route_probe_metadata = load_probe_set(probe_set_path)
    data_probe_records, data_probe_metadata = load_probe_set(data_probe_set_path)
    if str(route_probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Route probe set benchmark mismatch: probe={route_probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    if str(data_probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Data probe set benchmark mismatch: probe={data_probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("bilinear-qk-rank-data-attribution requires at least two checkpoints.")
    device = require_device(device_name)
    model = build_model(spec.model, len(vocab.tokens), device)
    if head_layer < 0 or head_layer >= len(model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(model.blocks) - 1}.")
    if head < 0 or head >= model.blocks[head_layer].attn.n_heads:
        raise ValueError(f"head {head} outside model range 0..{model.blocks[head_layer].attn.n_heads - 1}.")
    valid_stages = ["embedding"]
    for layer_index in range(len(model.blocks)):
        valid_stages.extend([f"layer_{layer_index}_post_attn", f"layer_{layer_index}_post_mlp"])
    valid_stages.append("final_norm")
    if context_stage not in valid_stages:
        raise ValueError(f"Unsupported context_stage {context_stage!r}; expected one of {valid_stages}.")

    route_pair_types = sorted(set(route_pair_types), key=route_pair_types.index)
    data_pair_types = sorted(set(data_pair_types), key=data_pair_types.index)
    route_pairs_all, route_pair_construction = _build_causal_patch_pairs(
        probe_records=route_probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=route_pair_types,
        max_pairs_per_type=max_route_pairs_per_type,
        min_pairs_per_type=min_route_pairs_per_type,
        split_filter=route_split_filter,
    )
    route_pairs = _route_objective_pairs(
        pairs=route_pairs_all,
        route_split=route_split,
        route_pair_type=route_pair_type,
    )
    data_pairs, data_pair_construction = _build_causal_patch_pairs(
        probe_records=data_probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=data_pair_types,
        max_pairs_per_type=max_data_pairs_per_type,
        min_pairs_per_type=min_data_pairs_per_type,
        split_filter=data_split_filter,
    )
    data_groups = _group_pairs_for_data_update(
        pairs=data_pairs,
        data_group_fields=data_group_fields,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "bilinear_qk_rank_data_attribution_report.json"
    markdown_path = output_dir / "bilinear_qk_rank_data_attribution_report.md"
    route_rows_path = output_dir / "bilinear_qk_rank_data_attribution_route_rows.jsonl"
    data_rows_path = output_dir / "bilinear_qk_rank_data_attribution_rows.jsonl"
    route_pair_rows_path = output_dir / "bilinear_qk_rank_data_attribution_route_pairs.jsonl"
    data_pair_rows_path = output_dir / "bilinear_qk_rank_data_attribution_data_pairs.jsonl"
    progress_path = output_dir / "bilinear_qk_rank_data_attribution_progress.json"
    for partial_path in (
        route_rows_path,
        data_rows_path,
        route_pair_rows_path,
        data_pair_rows_path,
        progress_path,
    ):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(route_pair_rows_path, [_pair_metadata(pair) for pair in route_pairs])
    write_jsonl(data_pair_rows_path, [_pair_metadata(pair) for pair in data_pairs])

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[bilinear-qk-rank-data-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} route_pairs={len(route_pairs)} "
        f"data_pairs={len(data_pairs)} data_groups={len(data_groups)} ranks={resolved_ranks} "
        f"device={device_name} head=L{head_layer}H{head} route={route_split}/{route_pair_type} "
        f"data_fields={data_group_fields} loss_side={loss_side} loss_scope={loss_scope}",
        flush=True,
    )

    all_route_rows: list[dict[str, Any]] = []
    all_data_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        learning_rate = _compute_learning_rate(spec.optimization, source_step)
        print(
            "[bilinear-qk-rank-data-attribution] starting "
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
        model.eval()
        source_parameters = _model_parameter_snapshot(model)

        loss_payloads: dict[str, dict[str, Any]] = {}
        for group in data_groups:
            loss_records = _loss_records_for_pairs(pairs=group["pairs"], loss_side=loss_side)
            loss_payloads[str(group["data_group_id"])] = _compute_loss_gradient_for_records_by_scope(
                model=model,
                records=loss_records,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
                loss_scope=loss_scope,
            )

        source_payloads: dict[int, dict[str, Any]] = {}
        route_gradients_by_rank: dict[int, dict[str, torch.Tensor]] = {}
        basis_by_rank: dict[int, tuple[torch.Tensor, torch.Tensor, dict[str, float]]] = {}
        for rank in resolved_ranks:
            left_basis, right_basis, singular_summary = _rank_qk_source_basis(
                model=model,
                head_layer=head_layer,
                head=head,
                rank=rank,
            )
            basis_by_rank[rank] = (left_basis, right_basis, singular_summary)
            source_payload = _rank_match_payload_for_pairs(
                model=model,
                pairs=route_pairs,
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
            gradients = source_payload["gradients"]
            if not isinstance(gradients, dict):
                raise TypeError("Rank QK source payload gradients must be a dictionary.")
            _assert_finite_gradients(gradients, label=f"rank-{rank} data attribution route objective")
            source_payloads[rank] = source_payload
            route_gradients_by_rank[rank] = gradients

        load_model_state(model, target_checkpoint["model_state"])
        model.eval()
        target_parameters = _model_parameter_snapshot(model)
        delta_parameters = _parameter_delta(
            source_parameters=source_parameters,
            target_parameters=target_parameters,
            label=f"bilinear QK rank data attribution {source_step}->{target_step}",
        )

        for rank in resolved_ranks:
            left_basis, right_basis, singular_summary = basis_by_rank[rank]
            target_payload = _rank_match_payload_for_pairs(
                model=model,
                pairs=route_pairs,
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
            source_payload = source_payloads[rank]
            route_gradients = route_gradients_by_rank[rank]
            actual_update_dot_summary = _gradient_dot_summary(
                left_gradients=delta_parameters,
                right_gradients=route_gradients,
                label=f"rank data actual update {source_step}->{target_step} rank={rank}",
            )
            route_row = _rank_route_row(
                source_step=source_step,
                target_step=target_step,
                source_checkpoint=source_checkpoint_path,
                target_checkpoint=target_checkpoint_path,
                learning_rate=learning_rate,
                route_split=route_split,
                route_pair_type=route_pair_type,
                record_side=record_side,
                rank=rank,
                context_stage=context_stage,
                layernorm_mode=layernorm_mode,
                head_layer=head_layer,
                head=head,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
                source_payload=source_payload,
                target_payload=target_payload,
                actual_update_dot_summary=actual_update_dot_summary,
                min_error_denominator=min_error_denominator,
            )
            append_jsonl(route_rows_path, route_row)
            all_route_rows.append(route_row)

            for group in data_groups:
                group_id = str(group["data_group_id"])
                loss_payload = loss_payloads[group_id]
                loss_gradients = loss_payload["gradients"]
                if not isinstance(loss_gradients, dict):
                    raise TypeError("Loss payload gradients must be a dictionary.")
                loss_rank_dot_summary = _gradient_dot_summary(
                    left_gradients=loss_gradients,
                    right_gradients=route_gradients,
                    label=f"rank data loss-route {source_step}->{target_step} rank={rank} group={group_id}",
                )
                loss_update_dot_summary = _gradient_dot_summary(
                    left_gradients=loss_gradients,
                    right_gradients=delta_parameters,
                    label=f"rank data loss-actual-update {source_step}->{target_step} rank={rank} group={group_id}",
                )
                data_row = _data_row(
                    route_row=route_row,
                    data_group_id=group_id,
                    data_group_values=group["data_group_values"],
                    loss_side=loss_side,
                    loss_scope=loss_scope,
                    loss_payload=loss_payload,
                    loss_rank_dot_summary=loss_rank_dot_summary,
                    loss_update_dot_summary=loss_update_dot_summary,
                )
                append_jsonl(data_rows_path, data_row)
                all_data_rows.append(data_row)

        all_rank_rows = [
            row
            for row in all_route_rows
            if int(row["source_step"]) == source_step and int(row["target_step"]) == target_step
        ]
        primary = next(row for row in all_rank_rows if int(row["rank"]) == resolved_ranks[-1])
        all_group = next(
            row
            for row in all_data_rows
            if int(row["source_step"]) == source_step
            and int(row["target_step"]) == target_step
            and int(row["rank"]) == resolved_ranks[-1]
            and str(row["data_group_id"]) == "__all__"
        )
        print(
            "[bilinear-qk-rank-data-attribution] finished "
            f"{source_step}->{target_step} rank={primary['rank']} "
            f"actual_delta={float(primary['actual_rank_match_delta']):.6g} "
            f"update_predicted={float(primary['actual_update_predicted_rank_match_delta']):.6g} "
            f"all_data_rank_support={float(all_group['negative_loss_dot_rank_gradient']):.6g} "
            f"sgd_equiv={float(all_group['sgd_equivalent_rank_match_delta_linearized']):.6g}",
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
                "route_rows_path": str(route_rows_path),
                "data_rows_path": str(data_rows_path),
            },
        )

    summary = _summarize(
        route_rows=all_route_rows,
        data_rows=all_data_rows,
        top_k_data_groups=top_k_data_groups,
    )
    report = {
        "schema_version": BILINEAR_QK_RANK_DATA_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "data_probe_set_path": str(data_probe_set_path),
        "route_probe_metadata": route_probe_metadata,
        "data_probe_metadata": data_probe_metadata,
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
        "record_side": record_side,
        "route_pair_types": route_pair_types,
        "route_pair_type": route_pair_type,
        "route_split": route_split,
        "route_split_filter": route_split_filter,
        "data_pair_types": data_pair_types,
        "data_split_filter": data_split_filter,
        "data_group_fields": data_group_fields,
        "loss_side": loss_side,
        "loss_scope": loss_scope,
        "max_route_pairs_per_type": max_route_pairs_per_type,
        "min_route_pairs_per_type": min_route_pairs_per_type,
        "max_data_pairs_per_type": max_data_pairs_per_type,
        "min_data_pairs_per_type": min_data_pairs_per_type,
        "top_k_data_groups": top_k_data_groups,
        "min_error_denominator": min_error_denominator,
        "calculation": {
            "rank_match_score": "mean score_rank(prediction, support_value) - mean score_rank(prediction, value_distractors)",
            "rank_basis": "source checkpoint fixed SVD basis for each interval and rank",
            "data_rank_support": "< -grad loss_data_group(theta_source), grad rank_match_score(theta_source) >",
            "loss_scope": "full_lm uses every next-token prediction; answer uses only answer-position cross entropy from extract_answer_logits.",
            "sgd_equivalent_delta": "learning_rate * data_rank_support",
            "actual_update_alignment": "< -grad loss_data_group(theta_source), theta_target - theta_source >",
            "caveat": "Data gradients are recomputed at the source checkpoint over the selected probe records; this is not an exact replay of historical mini-batches.",
        },
        "route_pair_construction": route_pair_construction,
        "data_pair_construction": data_pair_construction,
        "route_num_pairs": len(route_pairs),
        "data_num_pairs": len(data_pairs),
        "data_num_groups": len(data_groups),
        "route_rows_path": str(route_rows_path),
        "data_rows_path": str(data_rows_path),
        "route_pair_rows_path": str(route_pair_rows_path),
        "data_pair_rows_path": str(data_pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_intervals": len(intervals),
            "total_intervals": len(intervals),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "route_rows_path": str(route_rows_path),
            "data_rows_path": str(data_rows_path),
        },
    )
    print(f"[bilinear-qk-rank-data-attribution] complete report={report_path} rows={data_rows_path}", flush=True)
    return (
        report_path,
        markdown_path,
        route_rows_path,
        data_rows_path,
        route_pair_rows_path,
        data_pair_rows_path,
    )
