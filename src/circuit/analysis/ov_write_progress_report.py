from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import shutil
from typing import Any

import torch

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    ATTENTION_SCORE_RECORD_SIDES,
    GEOMETRY_POSITION_ROLES,
    _attention_key_positions,
    _best_wrong_value_token_ids,
    _build_causal_patch_pairs,
    _checkpoint_step_from_path,
    _head_label,
    _holdout_pair_set,
    _margin_gradient_vectors,
    _pair_metadata,
    _resolve_attention_score_record_sides,
    _resolve_checkpoint_paths,
    _single_attention_position,
    _single_vector_value_margin,
    _validate_single_query_batch,
    _value_margin,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


OV_WRITE_PROGRESS_REPORT_SCHEMA_VERSION = 1

OV_WRITE_CONDITIONS = [
    "real_attention_real_values",
    "correct_support_attention_real_values",
    "real_attention_shuffled_values",
    "correct_support_attention_shuffled_values",
]

CHECKPOINT_METRICS = [
    "qk_separation",
    "attention_support_mean",
    "attention_support_mass",
    "attention_mass_separation",
    "ov_map_value_margin",
    "head_value_margin_dla",
    "head_answer_logit_dla",
    "head_margin_dla",
    "attended_support_ov_value_margin",
    "support_mass_ov_value_margin",
    "qk_ov_product",
    "attention_ov_product",
    "correct_value_logit",
    "best_wrong_value_logit",
    "moving_answer_margin",
    "negative_answer_loss",
]

OUTPUT_DELTA_METRICS = [
    "delta_correct_value_logit",
    "delta_negative_answer_loss",
    "delta_fixed_source_competitor_margin",
    "delta_fixed_target_competitor_margin",
    "delta_moving_answer_margin",
]

WRITE_DELTA_METRICS = [
    "delta_ov_map_value_margin",
    "delta_head_value_margin_dla",
    "delta_head_answer_logit_dla",
    "delta_head_margin_dla",
    "delta_attended_support_ov_value_margin",
    "delta_support_mass_ov_value_margin",
    "delta_qk_ov_product",
    "delta_attention_ov_product",
]


@dataclass(frozen=True)
class HeadSpec:
    layer: int
    head: int

    @property
    def label(self) -> str:
        return _head_label(self.layer, self.head)


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise RuntimeError(f"Cannot compute mean for empty values: {label}")
    return sum(values) / float(len(values))


def _std(values: list[float], *, label: str) -> float:
    if not values:
        raise RuntimeError(f"Cannot compute std for empty values: {label}")
    mean = _mean(values, label=label)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / float(len(values)))


def _fraction(numerator: int, denominator: int, *, label: str) -> float:
    if denominator <= 0:
        raise RuntimeError(f"Cannot compute fraction for {label}: denominator={denominator}")
    return numerator / float(denominator)


def _pearson(x_values: list[float], y_values: list[float], *, label: str) -> dict[str, Any]:
    if len(x_values) != len(y_values):
        raise ValueError(f"Correlation input length mismatch for {label}: {len(x_values)} vs {len(y_values)}.")
    if len(x_values) < 2:
        return {
            "label": label,
            "value": None,
            "status": "not_computed",
            "reason": "requires_at_least_two_intervals",
            "num_points": len(x_values),
        }
    x_mean = _mean(x_values, label=f"{label} x")
    y_mean = _mean(y_values, label=f"{label} y")
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    x_norm = math.sqrt(sum(value * value for value in x_centered))
    y_norm = math.sqrt(sum(value * value for value in y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return {
            "label": label,
            "value": None,
            "status": "not_computed",
            "reason": "zero_variance",
            "num_points": len(x_values),
        }
    value = sum(x * y for x, y in zip(x_centered, y_centered, strict=True)) / (x_norm * y_norm)
    return {
        "label": label,
        "value": float(value),
        "status": "computed",
        "reason": None,
        "num_points": len(x_values),
    }


def _parse_head_spec(raw: str) -> HeadSpec:
    match = re.fullmatch(r"L(?P<layer>\d+)H(?P<head>\d+)", raw)
    if match is None:
        raise ValueError(f"Head spec must have form L<layer>H<head>, got {raw!r}.")
    return HeadSpec(layer=int(match.group("layer")), head=int(match.group("head")))


def _resolve_head_specs(raw_heads: list[str]) -> list[HeadSpec]:
    if not raw_heads:
        raise ValueError("At least one --head is required.")
    heads = [_parse_head_spec(raw) for raw in raw_heads]
    seen: set[tuple[int, int]] = set()
    resolved: list[HeadSpec] = []
    for head in heads:
        key = (head.layer, head.head)
        if key in seen:
            raise ValueError(f"Duplicate head spec: {head.label}")
        seen.add(key)
        resolved.append(head)
    return resolved


def _value_logits_by_token_id(
    *,
    answer_logits: torch.Tensor,
    row_index: int,
    value_token_ids: torch.Tensor,
) -> dict[str, float]:
    values = answer_logits[row_index].index_select(0, value_token_ids)
    return {
        str(int(token_id)): float(value.detach().float().cpu().item())
        for token_id, value in zip(value_token_ids.detach().cpu().tolist(), values, strict=True)
    }


def _payload_for_records(
    *,
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    head_spec: HeadSpec,
    vocab: Vocabulary,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty for OV write progress report.")
    if head_spec.layer < 0 or head_spec.layer >= len(model.blocks):
        raise ValueError(f"head layer {head_spec.layer} outside model range 0..{len(model.blocks) - 1}.")
    block = model.blocks[head_spec.layer]
    if head_spec.head < 0 or head_spec.head >= block.attn.n_heads:
        raise ValueError(
            f"head {head_spec.head} outside model range 0..{block.attn.n_heads - 1} "
            f"for layer {head_spec.layer}."
        )

    batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
    with torch.no_grad():
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_attentions=True,
            return_residual_streams=True,
        )
    if outputs.attentions is None:
        raise RuntimeError("OV write progress report requires attention probabilities.")
    if outputs.residual_streams is None:
        raise RuntimeError("OV write progress report requires residual streams.")

    answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
    _validate_single_query_batch(batch=batch, metadata=metadata, label="OV write progress report")
    value_token_ids = torch.tensor(vocab.value_token_ids, device=device, dtype=torch.long)
    answer_margins = _value_margin(answer_logits, answer_targets, value_token_ids)
    answer_losses = torch.nn.functional.cross_entropy(answer_logits, answer_targets, reduction="none")
    wrong_token_ids = _best_wrong_value_token_ids(
        logits=answer_logits,
        answer_targets=answer_targets,
        value_token_ids=value_token_ids,
    )
    final_pre_stage = f"layer_{len(model.blocks) - 1}_post_mlp"
    final_pre_vectors = outputs.residual_streams[final_pre_stage][
        metadata["rows"],
        metadata["prediction_positions"],
        :,
    ]
    margin_gradients, recomputed_margins = _margin_gradient_vectors(
        model=model,
        final_residual_vectors=final_pre_vectors,
        correct_token_ids=answer_targets,
        wrong_token_ids=wrong_token_ids,
    )
    if not torch.allclose(recomputed_margins, answer_margins.detach(), atol=1.0e-4, rtol=1.0e-4):
        max_delta = float((recomputed_margins - answer_margins.detach()).abs().max().item())
        raise RuntimeError(f"OV write progress margin-gradient check failed: max_delta={max_delta:.6g}")

    pre_state = (
        outputs.residual_streams["embedding"]
        if head_spec.layer == 0
        else outputs.residual_streams[f"layer_{head_spec.layer - 1}_post_mlp"]
    )
    attention_input = block.ln_1(pre_state)
    batch_size, seq_len, _ = attention_input.shape
    head_dim = int(block.attn.head_dim)
    q_all = block.attn.q_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    k_all = block.attn.k_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    v_all = block.attn.v_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    q_head = q_all[:, :, head_spec.head, :]
    k_head = k_all[:, :, head_spec.head, :]
    v_head = v_all[:, :, head_spec.head, :]
    scores = torch.matmul(q_head, k_head.transpose(-2, -1)) / math.sqrt(head_dim)
    attention = outputs.attentions[head_spec.layer][:, head_spec.head, :, :]
    head_slice = slice(head_spec.head * head_dim, (head_spec.head + 1) * head_dim)
    out_head = block.attn.out_proj.weight[:, head_slice]

    return {
        "batch": batch,
        "metadata": metadata,
        "answer_logits": answer_logits,
        "answer_targets": answer_targets,
        "answer_margins": answer_margins,
        "answer_losses": answer_losses,
        "answer_correct": answer_logits.argmax(dim=-1) == answer_targets,
        "wrong_token_ids": wrong_token_ids,
        "margin_gradients": margin_gradients,
        "value_token_ids": value_token_ids,
        "scores": scores,
        "attention": attention,
        "v": v_head,
        "out_head": out_head,
        "unembed": model.lm_head.weight,
    }


def _condition_head_outputs(
    *,
    real_head_output: torch.Tensor,
    support_v_mean: torch.Tensor,
    support_v_contribution: torch.Tensor,
    support_attention_mass: torch.Tensor,
    shuffled_support_v_mean: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    real_attention_shuffled = real_head_output - support_v_contribution + support_attention_mass * shuffled_support_v_mean
    return {
        "real_attention_real_values": (real_head_output, support_v_mean),
        "correct_support_attention_real_values": (support_v_mean, support_v_mean),
        "real_attention_shuffled_values": (real_attention_shuffled, shuffled_support_v_mean),
        "correct_support_attention_shuffled_values": (shuffled_support_v_mean, shuffled_support_v_mean),
    }


def _pair_rows_for_payload(
    *,
    payload: dict[str, Any],
    pairs: list[dict[str, Any]],
    checkpoint_path: Path,
    step: int,
    head_spec: HeadSpec,
    record_side: str,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
) -> list[dict[str, Any]]:
    if len(pairs) < 2:
        raise ValueError("OV write progress shuffled-value controls require at least two pairs.")
    device = payload["answer_targets"].device
    support_summaries: list[dict[str, Any]] = []
    support_v_means: list[torch.Tensor] = []
    for pair_index, pair in enumerate(pairs):
        batch_row, query_position = _single_attention_position(
            batch=payload["batch"],
            metadata=payload["metadata"],
            flat_index=pair_index,
            position_role=score_query_role,
            label="OV write query",
        )
        support_batch_row, support_positions = _attention_key_positions(
            batch=payload["batch"],
            metadata=payload["metadata"],
            flat_index=pair_index,
            position_role=support_key_role,
            max_position=query_position,
        )
        distractor_batch_row, distractor_positions = _attention_key_positions(
            batch=payload["batch"],
            metadata=payload["metadata"],
            flat_index=pair_index,
            position_role=distractor_key_role,
            max_position=query_position,
        )
        if support_batch_row != batch_row:
            raise RuntimeError(
                f"Support role {support_key_role!r} selected row {support_batch_row}, "
                f"but query role selected row {batch_row} for pair {pair['pair_id']}."
            )
        if distractor_batch_row != batch_row:
            raise RuntimeError(
                f"Distractor role {distractor_key_role!r} selected row {distractor_batch_row}, "
                f"but query role selected row {batch_row} for pair {pair['pair_id']}."
            )
        support_position_tensor = torch.tensor(support_positions, device=device, dtype=torch.long)
        distractor_position_tensor = torch.tensor(distractor_positions, device=device, dtype=torch.long)
        attention_row = payload["attention"][batch_row, query_position, :]
        score_row = payload["scores"][batch_row, query_position, :]
        support_attention = attention_row.index_select(0, support_position_tensor)
        distractor_attention = attention_row.index_select(0, distractor_position_tensor)
        support_v = payload["v"][batch_row].index_select(0, support_position_tensor)
        support_v_mean = support_v.mean(dim=0)
        support_v_contribution = torch.matmul(support_attention, support_v)
        real_head_output = torch.matmul(attention_row, payload["v"][batch_row])
        support_scores = score_row.index_select(0, support_position_tensor)
        distractor_scores = score_row.index_select(0, distractor_position_tensor)
        support_summaries.append(
            {
                "pair_index": pair_index,
                "pair": pair,
                "batch_row": batch_row,
                "query_position": query_position,
                "support_positions": support_positions,
                "distractor_positions": distractor_positions,
                "support_attention_mean": support_attention.mean(),
                "support_attention_mass": support_attention.sum(),
                "distractor_attention_mass": distractor_attention.sum(),
                "qk_support_score": support_scores.mean(),
                "qk_distractor_score": distractor_scores.mean(),
                "real_head_output": real_head_output,
                "support_v_contribution": support_v_contribution,
                "support_v_mean": support_v_mean,
            }
        )
        support_v_means.append(support_v_mean)

    support_v_matrix = torch.stack(support_v_means, dim=0)
    shuffled_support_v_matrix = torch.roll(support_v_matrix, shifts=1, dims=0)
    rows: list[dict[str, Any]] = []
    for summary in support_summaries:
        pair_index = int(summary["pair_index"])
        pair = summary["pair"]
        answer_token_id = int(payload["answer_targets"][pair_index].item())
        wrong_token_id = int(payload["wrong_token_ids"][pair_index].item())
        qk_support_score = summary["qk_support_score"]
        qk_distractor_score = summary["qk_distractor_score"]
        qk_separation = qk_support_score - qk_distractor_score
        attention_support_mean = summary["support_attention_mean"]
        attention_support_mass = summary["support_attention_mass"]
        attention_mass_separation = attention_support_mass - summary["distractor_attention_mass"]
        conditions = _condition_head_outputs(
            real_head_output=summary["real_head_output"],
            support_v_mean=summary["support_v_mean"],
            support_v_contribution=summary["support_v_contribution"],
            support_attention_mass=attention_support_mass,
            shuffled_support_v_mean=shuffled_support_v_matrix[pair_index],
        )
        correct_value_logit = float(payload["answer_logits"][pair_index, answer_token_id].detach().float().cpu().item())
        best_wrong_value_logit = float(payload["answer_logits"][pair_index, wrong_token_id].detach().float().cpu().item())
        value_logits = _value_logits_by_token_id(
            answer_logits=payload["answer_logits"],
            row_index=pair_index,
            value_token_ids=payload["value_token_ids"],
        )
        for condition_name, (condition_head_output, condition_source_v) in conditions.items():
            head_write = torch.matmul(condition_head_output, payload["out_head"].T)
            ov_map_write = torch.matmul(condition_source_v, payload["out_head"].T)
            ov_map_value_margin = _single_vector_value_margin(
                residual_vector=ov_map_write,
                correct_token_id=answer_token_id,
                value_token_ids=payload["value_token_ids"],
                unembed=payload["unembed"],
            )
            head_value_margin_dla = _single_vector_value_margin(
                residual_vector=head_write,
                correct_token_id=answer_token_id,
                value_token_ids=payload["value_token_ids"],
                unembed=payload["unembed"],
            )
            head_answer_logit_dla = torch.dot(head_write, payload["unembed"][answer_token_id])
            head_margin_dla = torch.dot(head_write, payload["margin_gradients"][pair_index])
            attended_support_ov_value_margin = attention_support_mean * ov_map_value_margin
            support_mass_ov_value_margin = attention_support_mass * ov_map_value_margin
            qk_ov_product = qk_separation * ov_map_value_margin
            attention_ov_product = attention_mass_separation * ov_map_value_margin
            rows.append(
                {
                    "step": step,
                    "checkpoint": str(checkpoint_path),
                    "split": str(pair["split"]),
                    "pair_type": str(pair["pair_type"]),
                    "record_side": record_side,
                    "pair_id": str(pair["pair_id"]),
                    "source_sample_id": str(pair["source_sample_id"]),
                    "source_query_index": int(pair["source_query_index"]),
                    "head_layer": head_spec.layer,
                    "head": head_spec.head,
                    "head_label": head_spec.label,
                    "condition": condition_name,
                    "score_query_role": score_query_role,
                    "support_key_role": support_key_role,
                    "distractor_key_role": distractor_key_role,
                    "query_position": int(summary["query_position"]),
                    "support_positions": [int(position) for position in summary["support_positions"]],
                    "distractor_positions": [int(position) for position in summary["distractor_positions"]],
                    "answer_token_id": answer_token_id,
                    "best_wrong_token_id": wrong_token_id,
                    "value_logits_by_token_id": value_logits,
                    "answer_correct": bool(payload["answer_correct"][pair_index].detach().cpu().item()),
                    "qk_support_score": float(qk_support_score.detach().float().cpu().item()),
                    "qk_distractor_score": float(qk_distractor_score.detach().float().cpu().item()),
                    "qk_separation": float(qk_separation.detach().float().cpu().item()),
                    "attention_support_mean": float(attention_support_mean.detach().float().cpu().item()),
                    "attention_support_mass": float(attention_support_mass.detach().float().cpu().item()),
                    "attention_mass_separation": float(attention_mass_separation.detach().float().cpu().item()),
                    "ov_map_value_margin": float(ov_map_value_margin.detach().float().cpu().item()),
                    "head_value_margin_dla": float(head_value_margin_dla.detach().float().cpu().item()),
                    "head_answer_logit_dla": float(head_answer_logit_dla.detach().float().cpu().item()),
                    "head_margin_dla": float(head_margin_dla.detach().float().cpu().item()),
                    "attended_support_ov_value_margin": float(
                        attended_support_ov_value_margin.detach().float().cpu().item()
                    ),
                    "support_mass_ov_value_margin": float(support_mass_ov_value_margin.detach().float().cpu().item()),
                    "qk_ov_product": float(qk_ov_product.detach().float().cpu().item()),
                    "attention_ov_product": float(attention_ov_product.detach().float().cpu().item()),
                    "correct_value_logit": correct_value_logit,
                    "best_wrong_value_logit": best_wrong_value_logit,
                    "moving_answer_margin": float(payload["answer_margins"][pair_index].detach().float().cpu().item()),
                    "negative_answer_loss": -float(payload["answer_losses"][pair_index].detach().float().cpu().item()),
                }
            )
    return rows


def _group_keys_for_row(row: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(row["split"]), str(row["pair_type"])),
        ("__all__", str(row["pair_type"])),
        (str(row["split"]), "__all__"),
        ("__all__", "__all__"),
    ]


def _aggregate_checkpoint_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in pair_rows:
        for split, pair_type in _group_keys_for_row(row):
            key = (
                int(row["step"]),
                int(row["head_layer"]),
                int(row["head"]),
                str(row["condition"]),
                split,
                pair_type,
                str(row["record_side"]),
            )
            groups.setdefault(key, []).append(row)
    checkpoint_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        step, head_layer, head, condition, split, pair_type, record_side = key
        first = rows[0]
        row: dict[str, Any] = {
            "step": int(step),
            "checkpoint": str(first["checkpoint"]),
            "head_layer": int(head_layer),
            "head": int(head),
            "head_label": _head_label(int(head_layer), int(head)),
            "condition": condition,
            "split": split,
            "pair_type": pair_type,
            "record_side": record_side,
            "num_entries": len(rows),
            "num_unique_pairs": len({str(item["pair_id"]) for item in rows}),
            "answer_accuracy": _fraction(
                sum(1 for item in rows if bool(item["answer_correct"])),
                len(rows),
                label="OV write checkpoint answer accuracy",
            ),
        }
        for metric_name in CHECKPOINT_METRICS:
            values = [float(item[metric_name]) for item in rows]
            row[f"{metric_name}_mean"] = _mean(values, label=f"{metric_name} mean")
            row[f"{metric_name}_std"] = _std(values, label=f"{metric_name} std")
        checkpoint_rows.append(row)
    if not checkpoint_rows:
        raise RuntimeError("OV write progress report produced no checkpoint rows.")
    return checkpoint_rows


def _fixed_margin_delta(
    *,
    source_row: dict[str, Any],
    target_row: dict[str, Any],
    wrong_token_id: int,
) -> float:
    token_key = str(wrong_token_id)
    source_logits = source_row["value_logits_by_token_id"]
    target_logits = target_row["value_logits_by_token_id"]
    if token_key not in source_logits or token_key not in target_logits:
        raise RuntimeError(f"Missing fixed competitor token {token_key} in value logits.")
    source_margin = float(source_row["correct_value_logit"]) - float(source_logits[token_key])
    target_margin = float(target_row["correct_value_logit"]) - float(target_logits[token_key])
    return target_margin - source_margin


def _build_pair_delta_rows(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in pair_rows:
        key = (
            int(row["head_layer"]),
            int(row["head"]),
            str(row["condition"]),
            str(row["record_side"]),
            str(row["pair_id"]),
        )
        by_key.setdefault(key, []).append(row)
    delta_rows: list[dict[str, Any]] = []
    for key, rows in sorted(by_key.items()):
        sorted_rows = sorted(rows, key=lambda item: int(item["step"]))
        for source_row, target_row in zip(sorted_rows[:-1], sorted_rows[1:], strict=True):
            source_step = int(source_row["step"])
            target_step = int(target_row["step"])
            if target_step <= source_step:
                raise RuntimeError(f"Non-increasing OV write pair delta interval: {source_step}->{target_step}")
            if str(source_row["answer_token_id"]) != str(target_row["answer_token_id"]):
                raise RuntimeError(
                    f"Answer token changed for pair {source_row['pair_id']} "
                    f"{source_step}->{target_step}: {source_row['answer_token_id']} vs {target_row['answer_token_id']}"
                )
            row: dict[str, Any] = {
                "source_step": source_step,
                "target_step": target_step,
                "step_gap": target_step - source_step,
                "source_checkpoint": str(source_row["checkpoint"]),
                "target_checkpoint": str(target_row["checkpoint"]),
                "split": str(source_row["split"]),
                "pair_type": str(source_row["pair_type"]),
                "record_side": str(source_row["record_side"]),
                "pair_id": str(source_row["pair_id"]),
                "head_layer": int(source_row["head_layer"]),
                "head": int(source_row["head"]),
                "head_label": str(source_row["head_label"]),
                "condition": str(source_row["condition"]),
                "answer_token_id": int(source_row["answer_token_id"]),
                "source_best_wrong_token_id": int(source_row["best_wrong_token_id"]),
                "target_best_wrong_token_id": int(target_row["best_wrong_token_id"]),
                "competitor_switched": int(source_row["best_wrong_token_id"]) != int(target_row["best_wrong_token_id"]),
                "source_answer_correct": bool(source_row["answer_correct"]),
                "target_answer_correct": bool(target_row["answer_correct"]),
                "delta_fixed_source_competitor_margin": _fixed_margin_delta(
                    source_row=source_row,
                    target_row=target_row,
                    wrong_token_id=int(source_row["best_wrong_token_id"]),
                ),
                "delta_fixed_target_competitor_margin": _fixed_margin_delta(
                    source_row=source_row,
                    target_row=target_row,
                    wrong_token_id=int(target_row["best_wrong_token_id"]),
                ),
            }
            for metric_name in CHECKPOINT_METRICS:
                row[f"delta_{metric_name}"] = float(target_row[metric_name]) - float(source_row[metric_name])
            delta_rows.append(row)
    if not delta_rows:
        raise RuntimeError("OV write progress report produced no pair delta rows.")
    return delta_rows


def _aggregate_delta_rows(pair_delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in pair_delta_rows:
        for split, pair_type in _group_keys_for_row(row):
            key = (
                int(row["source_step"]),
                int(row["target_step"]),
                int(row["head_layer"]),
                int(row["head"]),
                str(row["condition"]),
                split,
                pair_type,
                str(row["record_side"]),
            )
            groups.setdefault(key, []).append(row)
    aggregate_rows: list[dict[str, Any]] = []
    delta_metrics = [f"delta_{metric_name}" for metric_name in CHECKPOINT_METRICS]
    delta_metrics.extend(["delta_fixed_source_competitor_margin", "delta_fixed_target_competitor_margin"])
    for key, rows in sorted(groups.items()):
        source_step, target_step, head_layer, head, condition, split, pair_type, record_side = key
        row: dict[str, Any] = {
            "source_step": int(source_step),
            "target_step": int(target_step),
            "step_gap": int(target_step) - int(source_step),
            "head_layer": int(head_layer),
            "head": int(head),
            "head_label": _head_label(int(head_layer), int(head)),
            "condition": condition,
            "split": split,
            "pair_type": pair_type,
            "record_side": record_side,
            "num_entries": len(rows),
            "num_unique_pairs": len({str(item["pair_id"]) for item in rows}),
            "competitor_switch_fraction": _fraction(
                sum(1 for item in rows if bool(item["competitor_switched"])),
                len(rows),
                label="OV write competitor switch fraction",
            ),
        }
        for metric_name in delta_metrics:
            values = [float(item[metric_name]) for item in rows]
            row[f"{metric_name}_mean"] = _mean(values, label=f"{metric_name} mean")
            row[f"{metric_name}_std"] = _std(values, label=f"{metric_name} std")
        aggregate_rows.append(row)
    if not aggregate_rows:
        raise RuntimeError("OV write progress report produced no aggregate delta rows.")
    return aggregate_rows


def _build_correlation_rows(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary_rows = [
        row
        for row in delta_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    groups: dict[tuple[int, int, str, str], list[dict[str, Any]]] = {}
    for row in primary_rows:
        key = (
            int(row["head_layer"]),
            int(row["head"]),
            str(row["condition"]),
            str(row["record_side"]),
        )
        groups.setdefault(key, []).append(row)
    correlation_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        head_layer, head, condition, record_side = key
        rows = sorted(rows, key=lambda item: int(item["source_step"]))
        for write_metric in WRITE_DELTA_METRICS:
            for output_metric in OUTPUT_DELTA_METRICS:
                x_values = [float(row[f"{write_metric}_mean"]) for row in rows]
                y_values = [float(row[f"{output_metric}_mean"]) for row in rows]
                correlation = _pearson(
                    x_values,
                    y_values,
                    label=f"{_head_label(head_layer, head)} {condition} {write_metric} vs {output_metric}",
                )
                correlation_rows.append(
                    {
                        "head_layer": head_layer,
                        "head": head,
                        "head_label": _head_label(head_layer, head),
                        "condition": condition,
                        "record_side": record_side,
                        "write_delta_metric": write_metric,
                        "output_delta_metric": output_metric,
                        "correlation": correlation["value"],
                        "status": correlation["status"],
                        "reason": correlation["reason"],
                        "num_points": correlation["num_points"],
                    }
                )
    if not correlation_rows:
        raise RuntimeError("OV write progress report produced no correlation rows.")
    return correlation_rows


def _summarize(
    *,
    checkpoint_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    primary_checkpoint_rows = [
        row
        for row in checkpoint_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    final_step = max(int(row["step"]) for row in primary_checkpoint_rows)
    final_rows = [row for row in primary_checkpoint_rows if int(row["step"]) == final_step]
    primary_delta_rows = [
        row
        for row in delta_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    computed_correlations = [row for row in correlation_rows if row["status"] == "computed"]
    return {
        "num_checkpoints": len({int(row["step"]) for row in checkpoint_rows}),
        "steps": sorted({int(row["step"]) for row in checkpoint_rows}),
        "num_delta_intervals": len({(int(row["source_step"]), int(row["target_step"])) for row in delta_rows}),
        "final_step": final_step,
        "final_primary_rows": sorted(
            final_rows,
            key=lambda row: (
                str(row["head_label"]),
                str(row["condition"]),
            ),
        ),
        "top_abs_correlations": sorted(
            computed_correlations,
            key=lambda row: abs(float(row["correlation"])),
            reverse=True,
        )[:top_k],
        "strongest_positive_correlations": sorted(
            computed_correlations,
            key=lambda row: float(row["correlation"]),
            reverse=True,
        )[:top_k],
        "strongest_negative_correlations": sorted(
            computed_correlations,
            key=lambda row: float(row["correlation"]),
        )[:top_k],
        "primary_delta_rows": primary_delta_rows,
    }


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# OV Write Progress Report",
        "",
        "This report audits candidate OV/write scalars before optimizer attribution.",
        "",
        "It does not claim full causal patching. The attention-frozen conditions are readout/DLA-level controls:",
        "",
        "```text",
        "real_attention_real_values: actual head output",
        "correct_support_attention_real_values: force the head output to the support-value V vector",
        "real_attention_shuffled_values: replace only the support-value contribution with another pair's support V",
        "correct_support_attention_shuffled_values: force the head output to a shuffled support V",
        "```",
        "",
        "## Run",
        "",
        f"- heads: `{report['heads']}`",
        f"- checkpoints: `{len(report['checkpoint_paths'])}`",
        f"- record sides: `{report['record_sides']}`",
        f"- pair types: `{report['pair_types']}`",
        "",
        "## Final Primary Rows",
        "",
        "| head | condition | OV map | head value DLA | head margin DLA | attended OV | correct logit | -loss | fixed note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["final_primary_rows"]:
        lines.append(
            "| `{head}` | `{condition}` | {ov:.6g} | {value:.6g} | {margin:.6g} | {attn:.6g} | {logit:.6g} | {loss:.6g} | `{note}` |".format(
                head=row["head_label"],
                condition=row["condition"],
                ov=float(row["ov_map_value_margin_mean"]),
                value=float(row["head_value_margin_dla_mean"]),
                margin=float(row["head_margin_dla_mean"]),
                attn=float(row["attended_support_ov_value_margin_mean"]),
                logit=float(row["correct_value_logit_mean"]),
                loss=float(row["negative_answer_loss_mean"]),
                note="output scalars are real-model outputs, repeated across conditions",
            )
        )
    lines.extend(
        [
            "",
            "## Top Correlations",
            "",
            "| head | condition | write delta | output delta | corr | points |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for row in summary["top_abs_correlations"]:
        lines.append(
            "| `{head}` | `{condition}` | `{write}` | `{output}` | {corr:.6f} | {points} |".format(
                head=row["head_label"],
                condition=row["condition"],
                write=row["write_delta_metric"],
                output=row["output_delta_metric"],
                corr=float(row["correlation"]),
                points=int(row["num_points"]),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- pair rows: `{report['pair_rows_path']}`",
            f"- checkpoint rows: `{report['checkpoint_rows_path']}`",
            f"- pair delta rows: `{report['pair_delta_rows_path']}`",
            f"- delta rows: `{report['delta_rows_path']}`",
            f"- correlation rows: `{report['correlation_rows_path']}`",
            f"- pair metadata: `{report['pair_metadata_rows_path']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ov_write_progress_report(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    heads: list[str],
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    pair_types: list[str],
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    record_sides: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 2,
    split_filter: list[str] | None = None,
    top_k_correlations: int = 24,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    head_specs = _resolve_head_specs(heads)
    unsupported_roles = [
        role
        for role in [score_query_role, support_key_role, distractor_key_role]
        if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported position roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if support_key_role == distractor_key_role:
        raise ValueError("support_key_role and distractor_key_role must be different.")
    if max_pairs_per_type < 2 or min_pairs_per_type < 2:
        raise ValueError("OV write shuffled controls require at least two pairs per requested pair type.")
    if top_k_correlations <= 0:
        raise ValueError("top_k_correlations must be positive.")
    resolved_record_sides = _resolve_attention_score_record_sides(record_sides)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("ov-write-progress-report requires at least two checkpoints.")
    model = build_model(spec.model, len(vocab.tokens), device)
    for head_spec in head_specs:
        if head_spec.layer < 0 or head_spec.layer >= len(model.blocks):
            raise ValueError(f"head layer {head_spec.layer} outside model range 0..{len(model.blocks) - 1}.")
        if head_spec.head < 0 or head_spec.head >= model.blocks[head_spec.layer].attn.n_heads:
            raise ValueError(
                f"head {head_spec.head} outside model range 0..{model.blocks[head_spec.layer].attn.n_heads - 1} "
                f"for layer {head_spec.layer}."
            )
    resolved_pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=resolved_pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if len(pairs) < 2:
        raise RuntimeError(f"OV write progress requires at least two constructed pairs, got {len(pairs)}.")

    pair_rows_path = output_dir / "ov_write_progress_pair_rows.jsonl"
    checkpoint_rows_path = output_dir / "ov_write_progress_checkpoint_rows.jsonl"
    pair_delta_rows_path = output_dir / "ov_write_progress_pair_delta_rows.jsonl"
    delta_rows_path = output_dir / "ov_write_progress_delta_rows.jsonl"
    correlation_rows_path = output_dir / "ov_write_progress_correlations.jsonl"
    pair_metadata_rows_path = output_dir / "ov_write_progress_pairs.jsonl"
    report_path = output_dir / "ov_write_progress_report.json"
    markdown_path = output_dir / "ov_write_progress_report.md"
    write_jsonl(pair_metadata_rows_path, [_pair_metadata(pair) for pair in pairs])

    print(
        "[ov-write-progress-report] "
        f"checkpoints={len(checkpoints)} heads={[head.label for head in head_specs]} pairs={len(pairs)} "
        f"pair_types={resolved_pair_types} device={device_name} record_sides={resolved_record_sides}",
        flush=True,
    )
    all_pair_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = load_checkpoint(checkpoint_path, device)
        step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if step != path_step:
            raise RuntimeError(f"Checkpoint step mismatch: payload={step} path={path_step}")
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        print(
            "[ov-write-progress-report] starting "
            f"{checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        checkpoint_pair_rows: list[dict[str, Any]] = []
        for head_spec in head_specs:
            for start_index in range(0, len(pairs), spec.evaluation.batch_size):
                pair_batch = pairs[start_index : start_index + spec.evaluation.batch_size]
                if len(pair_batch) < 2:
                    raise RuntimeError(
                        "OV write progress shuffled controls require each evaluation batch to contain at least two pairs; "
                        f"last batch had {len(pair_batch)}. Increase max pairs or evaluation batch size."
                    )
                for record_side in resolved_record_sides:
                    side_key = f"{record_side}_record"
                    if record_side not in ATTENTION_SCORE_RECORD_SIDES:
                        raise ValueError(
                            f"Unsupported record side {record_side!r}; expected one of {ATTENTION_SCORE_RECORD_SIDES}."
                        )
                    records = [pair[side_key] for pair in pair_batch]
                    payload = _payload_for_records(
                        model=model,
                        records=records,
                        head_spec=head_spec,
                        vocab=vocab,
                        pad_token_id=vocab.pad_token_id,
                        device=device,
                    )
                    rows = _pair_rows_for_payload(
                        payload=payload,
                        pairs=pair_batch,
                        checkpoint_path=checkpoint_path,
                        step=step,
                        head_spec=head_spec,
                        record_side=record_side,
                        score_query_role=score_query_role,
                        support_key_role=support_key_role,
                        distractor_key_role=distractor_key_role,
                    )
                    checkpoint_pair_rows.extend(rows)
        all_pair_rows.extend(checkpoint_pair_rows)
        primary_rows = [
            row
            for row in checkpoint_pair_rows
            if row["condition"] == "real_attention_real_values"
            and row["record_side"] == resolved_record_sides[0]
        ]
        mean_dla = _mean(
            [float(row["head_value_margin_dla"]) for row in primary_rows],
            label=f"checkpoint {step} primary head_value_margin_dla",
        )
        print(
            "[ov-write-progress-report] finished "
            f"step={step} primary_head_value_dla={mean_dla:.6g}",
            flush=True,
        )

    write_jsonl(pair_rows_path, all_pair_rows)
    checkpoint_rows = _aggregate_checkpoint_rows(all_pair_rows)
    pair_delta_rows = _build_pair_delta_rows(all_pair_rows)
    delta_rows = _aggregate_delta_rows(pair_delta_rows)
    correlation_rows = _build_correlation_rows(delta_rows)
    write_jsonl(checkpoint_rows_path, checkpoint_rows)
    write_jsonl(pair_delta_rows_path, pair_delta_rows)
    write_jsonl(delta_rows_path, delta_rows)
    write_jsonl(correlation_rows_path, correlation_rows)
    summary = _summarize(
        checkpoint_rows=checkpoint_rows,
        delta_rows=delta_rows,
        correlation_rows=correlation_rows,
        top_k=top_k_correlations,
    )
    report = {
        "schema_version": OV_WRITE_PROGRESS_REPORT_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "output_dir": str(output_dir),
        "device": device_name,
        "heads": [head.label for head in head_specs],
        "score_query_role": score_query_role,
        "support_key_role": support_key_role,
        "distractor_key_role": distractor_key_role,
        "record_sides": resolved_record_sides,
        "pair_types": resolved_pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "conditions": OV_WRITE_CONDITIONS,
        "checkpoint_metrics": CHECKPOINT_METRICS,
        "write_delta_metrics": WRITE_DELTA_METRICS,
        "output_delta_metrics": OUTPUT_DELTA_METRICS,
        "calculation": {
            "ov_map_value_margin": "support-position V vector through the head O slice, scored by correct value minus best wrong value unembedding",
            "head_value_margin_dla": "conditioned head output through O, scored by correct value minus best wrong value unembedding",
            "head_margin_dla": "conditioned head write dotted with the final answer-margin gradient at the same checkpoint",
            "attended_support_ov_value_margin": "mean support attention times ov_map_value_margin, matching earlier downstream-update scalar semantics",
            "support_mass_ov_value_margin": "total support attention mass times ov_map_value_margin",
            "qk_ov_product": "QK support-minus-distractor score separation times ov_map_value_margin",
            "attention_ov_product": "attention support-minus-distractor mass separation times ov_map_value_margin",
            "real_attention_shuffled_values": "actual head output with the support-value V contribution replaced by a deterministic one-pair roll",
            "correct_support_attention_shuffled_values": "forced support read using the deterministic one-pair rolled support V vector",
        },
        "pair_construction": pair_construction,
        "pair_rows_path": str(pair_rows_path),
        "checkpoint_rows_path": str(checkpoint_rows_path),
        "pair_delta_rows_path": str(pair_delta_rows_path),
        "delta_rows_path": str(delta_rows_path),
        "correlation_rows_path": str(correlation_rows_path),
        "pair_metadata_rows_path": str(pair_metadata_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(
        f"[ov-write-progress-report] complete report={report_path} rows={checkpoint_rows_path}",
        flush=True,
    )
    return (
        report_path,
        markdown_path,
        pair_rows_path,
        checkpoint_rows_path,
        pair_delta_rows_path,
        delta_rows_path,
        correlation_rows_path,
        pair_metadata_rows_path,
    )
