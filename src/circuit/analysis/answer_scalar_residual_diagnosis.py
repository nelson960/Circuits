from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    _build_route_competition_pairs,
    _checkpoint_step_from_path,
    _gradient_dot_summary,
    _holdout_pair_set,
    _model_parameter_snapshot,
    _pair_metadata,
    _parameter_delta,
    _parameter_gradients,
    _resolve_checkpoint_paths,
    _safe_ratio,
    _sign_match,
    _validate_single_query_batch,
)
from circuit.analysis.shared_feature_dynamics import _import_matplotlib
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import append_jsonl, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.train import _compute_learning_rate
from circuit.vocab import Vocabulary


ANSWER_SCALAR_RESIDUAL_DIAGNOSIS_SCHEMA_VERSION = 1

ANSWER_SCALAR_NAMES = [
    "moving_answer_margin",
    "fixed_source_competitor_margin",
    "fixed_target_competitor_margin",
    "correct_value_logit",
    "source_best_wrong_logit",
    "target_best_wrong_logit",
    "negative_answer_loss",
]
ANSWER_SCALAR_MARGIN_SIDES = ["clean", "corrupted"]
ANSWER_SCALAR_SWITCH_BUCKETS = ["all", "same_competitor", "competitor_switch"]
ANSWER_SCALAR_METRIC_SCOPES = ["aggregate", "pair_type", "split", "split_pair_type"]
ANSWER_SCALAR_SECOND_ORDER_MODES = ["none", "endpoint_gradient"]


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"Cannot compute mean for empty values: {label}")
    return sum(values) / float(len(values))


def _resolve_unique_values(
    *,
    values: list[str] | None,
    default_values: list[str],
    allowed_values: list[str],
    label: str,
) -> list[str]:
    raw_values = default_values if values is None else values
    if not raw_values:
        raise ValueError(f"{label} must not be empty.")
    resolved: list[str] = []
    for value in raw_values:
        if value not in allowed_values:
            raise ValueError(f"Unsupported {label} {value!r}; expected one of {allowed_values}.")
        if value not in resolved:
            resolved.append(value)
    return resolved


def _value_target_indices(answer_targets: torch.Tensor, value_token_ids: torch.Tensor) -> torch.Tensor:
    matches = (value_token_ids.unsqueeze(0) == answer_targets.unsqueeze(1)).nonzero(as_tuple=False)
    if matches.size(0) != answer_targets.size(0):
        raise RuntimeError("Failed to locate every answer target in the value-token set.")
    return matches[:, 1]


def _best_wrong_value_tokens(
    *,
    answer_logits: torch.Tensor,
    answer_targets: torch.Tensor,
    value_token_ids: torch.Tensor,
    top_k_wrong: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if top_k_wrong <= 0:
        raise ValueError("top_k_wrong must be positive.")
    value_logits = answer_logits.index_select(dim=-1, index=value_token_ids)
    target_indices = _value_target_indices(answer_targets, value_token_ids)
    row_index = torch.arange(answer_targets.size(0), device=answer_targets.device)
    correct_logits = value_logits[row_index, target_indices]
    masked = value_logits.clone()
    masked[row_index, target_indices] = torch.finfo(masked.dtype).min
    best_wrong_values, best_wrong_indices = masked.max(dim=-1)
    best_wrong_token_ids = value_token_ids[best_wrong_indices]
    top_k = min(top_k_wrong, max(1, int(value_logits.size(1)) - 1))
    top_wrong_values, top_wrong_indices = torch.topk(masked, k=top_k, dim=-1)
    top_wrong_token_ids = value_token_ids[top_wrong_indices]
    return correct_logits, best_wrong_values, best_wrong_token_ids, top_wrong_values, top_wrong_token_ids


def _pair_id_for_pair(pair: dict[str, Any], margin_side: str) -> str:
    return f"{pair['pair_id']}::{margin_side}"


def _records_for_margin_side(*, pairs: list[dict[str, Any]], margin_side: str) -> list[dict[str, Any]]:
    if margin_side not in ANSWER_SCALAR_MARGIN_SIDES:
        raise ValueError(f"Unsupported margin side {margin_side!r}; expected one of {ANSWER_SCALAR_MARGIN_SIDES}.")
    if not pairs:
        raise ValueError("Cannot build records from an empty pair list.")
    return [pair[f"{margin_side}_record"] for pair in pairs]


def _compute_checkpoint_value_payload(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    margin_side: str,
    batch_size: int,
    pad_token_id: int,
    top_k_wrong: int,
    device: torch.device,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pairs must not be empty for answer scalar payload computation.")
    if top_k_wrong <= 0:
        raise ValueError("top_k_wrong must be positive.")
    checkpoint_step = _checkpoint_step_from_path(checkpoint_path)
    model.eval()
    rows: list[dict[str, Any]] = []
    value_logits_by_pair_id: dict[str, torch.Tensor] = {}
    correct_logit_by_pair_id: dict[str, float] = {}
    correct_log_prob_by_pair_id: dict[str, float] = {}
    correct_token_id_by_pair_id: dict[str, int] = {}
    best_wrong_token_id_by_pair_id: dict[str, int] = {}
    best_wrong_logit_by_pair_id: dict[str, float] = {}
    value_token_ids = torch.tensor(vocab.value_token_ids, device=device, dtype=torch.long)
    value_token_ids_cpu = [int(token_id) for token_id in value_token_ids.detach().cpu().tolist()]
    with torch.no_grad():
        for start_index in range(0, len(pairs), batch_size):
            pair_batch = pairs[start_index : start_index + batch_size]
            records = _records_for_margin_side(pairs=pair_batch, margin_side=margin_side)
            batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=metadata, label="answer scalar residual diagnosis")
            (
                correct_logits,
                best_wrong_logits,
                best_wrong_token_ids,
                top_wrong_logits,
                top_wrong_token_ids,
            ) = _best_wrong_value_tokens(
                answer_logits=answer_logits,
                answer_targets=answer_targets,
                value_token_ids=value_token_ids,
                top_k_wrong=top_k_wrong,
            )
            log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
            correct_log_probs = log_probs.gather(dim=-1, index=answer_targets.unsqueeze(1)).squeeze(1)
            predictions = answer_logits.argmax(dim=-1)
            value_logits = answer_logits.index_select(dim=-1, index=value_token_ids).detach().float().cpu()
            for pair_index, pair in enumerate(pair_batch):
                interval_pair_id = _pair_id_for_pair(pair, margin_side)
                if interval_pair_id in value_logits_by_pair_id:
                    raise RuntimeError(f"Duplicate interval pair id while building scalar payload: {interval_pair_id}")
                target_id = int(answer_targets[pair_index].detach().cpu().item())
                prediction_id = int(predictions[pair_index].detach().cpu().item())
                correct_logit = float(correct_logits[pair_index].detach().float().cpu().item())
                correct_log_prob = float(correct_log_probs[pair_index].detach().float().cpu().item())
                best_wrong_token_id = int(best_wrong_token_ids[pair_index].detach().cpu().item())
                best_wrong_logit = float(best_wrong_logits[pair_index].detach().float().cpu().item())
                top_token_ids = [int(token_id) for token_id in top_wrong_token_ids[pair_index].detach().cpu().tolist()]
                top_logits = [float(logit) for logit in top_wrong_logits[pair_index].detach().float().cpu().tolist()]
                value_logits_by_pair_id[interval_pair_id] = value_logits[pair_index]
                correct_logit_by_pair_id[interval_pair_id] = correct_logit
                correct_log_prob_by_pair_id[interval_pair_id] = correct_log_prob
                correct_token_id_by_pair_id[interval_pair_id] = target_id
                best_wrong_token_id_by_pair_id[interval_pair_id] = best_wrong_token_id
                best_wrong_logit_by_pair_id[interval_pair_id] = best_wrong_logit
                rows.append(
                    {
                        "step": checkpoint_step,
                        "checkpoint": str(checkpoint_path),
                        "pair_id": str(pair["pair_id"]),
                        "interval_pair_id": interval_pair_id,
                        "split": str(pair["split"]),
                        "pair_type": str(pair["pair_type"]),
                        "margin_side": margin_side,
                        "source_sample_id": str(pair["source_sample_id"]),
                        "source_query_index": int(pair["source_query_index"]),
                        "answer_target_id": target_id,
                        "answer_prediction_id": prediction_id,
                        "answer_correct": bool(prediction_id == target_id),
                        "correct_value_logit": correct_logit,
                        "correct_log_prob": correct_log_prob,
                        "negative_answer_loss": correct_log_prob,
                        "best_wrong_token_id": best_wrong_token_id,
                        "best_wrong_logit": best_wrong_logit,
                        "moving_answer_margin": correct_logit - best_wrong_logit,
                        "top_wrong_token_ids": top_token_ids,
                        "top_wrong_logits": top_logits,
                    }
                )
    if len(value_logits_by_pair_id) != len(pairs):
        raise RuntimeError(
            f"Expected one scalar payload row per pair, got payload={len(value_logits_by_pair_id)} pairs={len(pairs)}."
        )
    return {
        "step": checkpoint_step,
        "checkpoint": str(checkpoint_path),
        "margin_side": margin_side,
        "rows": rows,
        "value_token_ids": value_token_ids_cpu,
        "value_index_by_token_id": {token_id: index for index, token_id in enumerate(value_token_ids_cpu)},
        "value_logits_by_pair_id": value_logits_by_pair_id,
        "correct_logit_by_pair_id": correct_logit_by_pair_id,
        "correct_log_prob_by_pair_id": correct_log_prob_by_pair_id,
        "correct_token_id_by_pair_id": correct_token_id_by_pair_id,
        "best_wrong_token_id_by_pair_id": best_wrong_token_id_by_pair_id,
        "best_wrong_logit_by_pair_id": best_wrong_logit_by_pair_id,
    }


def _logit_for_token(
    *,
    payload: dict[str, Any],
    interval_pair_id: str,
    token_id: int,
) -> float:
    index = payload["value_index_by_token_id"].get(int(token_id))
    if index is None:
        raise KeyError(f"Token id {token_id} is not in value token set for {interval_pair_id}.")
    return float(payload["value_logits_by_pair_id"][interval_pair_id][index].item())


def _scalar_value_for_pair(
    *,
    scalar_name: str,
    payload: dict[str, Any],
    interval_pair_id: str,
    source_best_wrong_token_id: int,
    target_best_wrong_token_id: int,
) -> float:
    correct_logit = float(payload["correct_logit_by_pair_id"][interval_pair_id])
    if scalar_name == "moving_answer_margin":
        wrong_token_id = int(payload["best_wrong_token_id_by_pair_id"][interval_pair_id])
        return correct_logit - _logit_for_token(payload=payload, interval_pair_id=interval_pair_id, token_id=wrong_token_id)
    if scalar_name == "fixed_source_competitor_margin":
        return correct_logit - _logit_for_token(
            payload=payload,
            interval_pair_id=interval_pair_id,
            token_id=source_best_wrong_token_id,
        )
    if scalar_name == "fixed_target_competitor_margin":
        return correct_logit - _logit_for_token(
            payload=payload,
            interval_pair_id=interval_pair_id,
            token_id=target_best_wrong_token_id,
        )
    if scalar_name == "correct_value_logit":
        return correct_logit
    if scalar_name == "source_best_wrong_logit":
        return _logit_for_token(
            payload=payload,
            interval_pair_id=interval_pair_id,
            token_id=source_best_wrong_token_id,
        )
    if scalar_name == "target_best_wrong_logit":
        return _logit_for_token(
            payload=payload,
            interval_pair_id=interval_pair_id,
            token_id=target_best_wrong_token_id,
        )
    if scalar_name == "negative_answer_loss":
        return float(payload["correct_log_prob_by_pair_id"][interval_pair_id])
    raise ValueError(f"Unsupported answer scalar {scalar_name!r}; expected one of {ANSWER_SCALAR_NAMES}.")


def _combine_interval_pair_rows(
    *,
    source_payload: dict[str, Any],
    target_payload: dict[str, Any],
    pairs: list[dict[str, Any]],
    margin_side: str,
    scalar_names: list[str],
) -> list[dict[str, Any]]:
    source_rows_by_id = {str(row["interval_pair_id"]): row for row in source_payload["rows"]}
    target_rows_by_id = {str(row["interval_pair_id"]): row for row in target_payload["rows"]}
    if set(source_rows_by_id) != set(target_rows_by_id):
        raise RuntimeError("Source and target scalar payload pair ids do not match.")
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        interval_pair_id = _pair_id_for_pair(pair, margin_side)
        if interval_pair_id not in source_rows_by_id:
            raise KeyError(f"Missing source scalar payload row for {interval_pair_id}.")
        if interval_pair_id not in target_rows_by_id:
            raise KeyError(f"Missing target scalar payload row for {interval_pair_id}.")
        source_row = source_rows_by_id[interval_pair_id]
        target_row = target_rows_by_id[interval_pair_id]
        if int(source_row["answer_target_id"]) != int(target_row["answer_target_id"]):
            raise RuntimeError(
                f"Answer target changed across checkpoints for {interval_pair_id}: "
                f"{source_row['answer_target_id']} vs {target_row['answer_target_id']}"
            )
        source_best_wrong_token_id = int(source_payload["best_wrong_token_id_by_pair_id"][interval_pair_id])
        target_best_wrong_token_id = int(target_payload["best_wrong_token_id_by_pair_id"][interval_pair_id])
        competitor_switched = source_best_wrong_token_id != target_best_wrong_token_id
        scalar_values: dict[str, dict[str, float]] = {}
        for scalar_name in scalar_names:
            source_value = _scalar_value_for_pair(
                scalar_name=scalar_name,
                payload=source_payload,
                interval_pair_id=interval_pair_id,
                source_best_wrong_token_id=source_best_wrong_token_id,
                target_best_wrong_token_id=target_best_wrong_token_id,
            )
            target_value = _scalar_value_for_pair(
                scalar_name=scalar_name,
                payload=target_payload,
                interval_pair_id=interval_pair_id,
                source_best_wrong_token_id=source_best_wrong_token_id,
                target_best_wrong_token_id=target_best_wrong_token_id,
            )
            scalar_values[scalar_name] = {
                "source": source_value,
                "target": target_value,
                "delta": target_value - source_value,
            }
        rows.append(
            {
                "source_step": int(source_payload["step"]),
                "target_step": int(target_payload["step"]),
                "step_gap": int(target_payload["step"]) - int(source_payload["step"]),
                "source_checkpoint": source_payload["checkpoint"],
                "target_checkpoint": target_payload["checkpoint"],
                "pair_id": str(pair["pair_id"]),
                "interval_pair_id": interval_pair_id,
                "split": str(pair["split"]),
                "pair_type": str(pair["pair_type"]),
                "margin_side": margin_side,
                "source_sample_id": str(pair["source_sample_id"]),
                "source_query_index": int(pair["source_query_index"]),
                "clean_transfer_token": str(pair["clean_transfer_token"]),
                "corrupted_transfer_token": str(pair["corrupted_transfer_token"]),
                "answer_target_id": int(source_row["answer_target_id"]),
                "source_answer_prediction_id": int(source_row["answer_prediction_id"]),
                "target_answer_prediction_id": int(target_row["answer_prediction_id"]),
                "source_answer_correct": bool(source_row["answer_correct"]),
                "target_answer_correct": bool(target_row["answer_correct"]),
                "source_correct_value_logit": float(source_row["correct_value_logit"]),
                "target_correct_value_logit": float(target_row["correct_value_logit"]),
                "source_correct_log_prob": float(source_row["correct_log_prob"]),
                "target_correct_log_prob": float(target_row["correct_log_prob"]),
                "source_best_wrong_token_id": source_best_wrong_token_id,
                "target_best_wrong_token_id": target_best_wrong_token_id,
                "source_best_wrong_logit": float(source_row["best_wrong_logit"]),
                "target_best_wrong_logit": float(target_row["best_wrong_logit"]),
                "competitor_switched": competitor_switched,
                "source_top_wrong_token_ids": source_row["top_wrong_token_ids"],
                "source_top_wrong_logits": source_row["top_wrong_logits"],
                "target_top_wrong_token_ids": target_row["top_wrong_token_ids"],
                "target_top_wrong_logits": target_row["top_wrong_logits"],
                "scalars": scalar_values,
            }
        )
    return rows


def _pair_in_switch_bucket(*, row: dict[str, Any], switch_bucket: str) -> bool:
    if switch_bucket == "all":
        return True
    if switch_bucket == "same_competitor":
        return not bool(row["competitor_switched"])
    if switch_bucket == "competitor_switch":
        return bool(row["competitor_switched"])
    raise ValueError(f"Unsupported switch bucket {switch_bucket!r}; expected one of {ANSWER_SCALAR_SWITCH_BUCKETS}.")


def _build_metric_pair_groups(
    *,
    pairs: list[dict[str, Any]],
    metric_scopes: list[str],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if not pairs:
        raise ValueError("Cannot build metric pair groups from no pairs.")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if "aggregate" in metric_scopes:
        groups[("__all__", "__all__")] = list(pairs)
    if "pair_type" in metric_scopes:
        pair_types = sorted({str(pair["pair_type"]) for pair in pairs})
        for pair_type in pair_types:
            groups[("__all__", pair_type)] = [pair for pair in pairs if str(pair["pair_type"]) == pair_type]
    if "split" in metric_scopes:
        splits = sorted({str(pair["split"]) for pair in pairs})
        for split in splits:
            groups[(split, "__all__")] = [pair for pair in pairs if str(pair["split"]) == split]
    if "split_pair_type" in metric_scopes:
        splits = sorted({str(pair["split"]) for pair in pairs})
        pair_types = sorted({str(pair["pair_type"]) for pair in pairs})
        for split in splits:
            for pair_type in pair_types:
                selected = [
                    pair for pair in pairs if str(pair["split"]) == split and str(pair["pair_type"]) == pair_type
                ]
                if selected:
                    groups[(split, pair_type)] = selected
    return groups


def _rows_for_group_and_bucket(
    *,
    interval_pair_rows: list[dict[str, Any]],
    selected_pairs: list[dict[str, Any]],
    margin_side: str,
    switch_bucket: str,
) -> list[dict[str, Any]]:
    selected_ids = {_pair_id_for_pair(pair, margin_side) for pair in selected_pairs}
    rows = [
        row
        for row in interval_pair_rows
        if str(row["interval_pair_id"]) in selected_ids and _pair_in_switch_bucket(row=row, switch_bucket=switch_bucket)
    ]
    if len(rows) > len(selected_ids):
        raise RuntimeError("Metric row selection produced more rows than selected pairs.")
    return rows


def _actual_scalar_summary(
    *,
    interval_pair_rows: list[dict[str, Any]],
    selected_pairs: list[dict[str, Any]],
    margin_side: str,
    switch_bucket: str,
    scalar_name: str,
) -> dict[str, Any] | None:
    rows = _rows_for_group_and_bucket(
        interval_pair_rows=interval_pair_rows,
        selected_pairs=selected_pairs,
        margin_side=margin_side,
        switch_bucket=switch_bucket,
    )
    if not rows:
        return None
    source_values = [float(row["scalars"][scalar_name]["source"]) for row in rows]
    target_values = [float(row["scalars"][scalar_name]["target"]) for row in rows]
    delta_values = [float(row["scalars"][scalar_name]["delta"]) for row in rows]
    switch_count = sum(1 for row in rows if bool(row["competitor_switched"]))
    return {
        "num_pairs": len(rows),
        "source_value": _mean(source_values, label=f"{scalar_name} source"),
        "target_value": _mean(target_values, label=f"{scalar_name} target"),
        "actual_delta": _mean(delta_values, label=f"{scalar_name} delta"),
        "competitor_switch_count": switch_count,
        "competitor_switch_fraction": switch_count / float(len(rows)),
        "source_correct_logit": _mean(
            [float(row["source_correct_value_logit"]) for row in rows],
            label="source_correct_logit",
        ),
        "target_correct_logit": _mean(
            [float(row["target_correct_value_logit"]) for row in rows],
            label="target_correct_logit",
        ),
        "source_correct_log_prob": _mean(
            [float(row["source_correct_log_prob"]) for row in rows],
            label="source_correct_log_prob",
        ),
        "target_correct_log_prob": _mean(
            [float(row["target_correct_log_prob"]) for row in rows],
            label="target_correct_log_prob",
        ),
    }


def _wrong_token_ids_for_gradient(
    *,
    scalar_name: str,
    interval_pair_rows_by_id: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    margin_side: str,
    gradient_checkpoint_role: str,
) -> list[int] | None:
    if scalar_name in {"correct_value_logit", "negative_answer_loss"}:
        return None
    token_ids: list[int] = []
    for pair in pairs:
        row = interval_pair_rows_by_id[_pair_id_for_pair(pair, margin_side)]
        if scalar_name == "moving_answer_margin":
            if gradient_checkpoint_role == "source":
                token_ids.append(int(row["source_best_wrong_token_id"]))
            elif gradient_checkpoint_role == "target":
                token_ids.append(int(row["target_best_wrong_token_id"]))
            else:
                raise ValueError(f"Unsupported gradient checkpoint role: {gradient_checkpoint_role}")
        elif scalar_name in {"fixed_source_competitor_margin", "source_best_wrong_logit"}:
            token_ids.append(int(row["source_best_wrong_token_id"]))
        elif scalar_name in {"fixed_target_competitor_margin", "target_best_wrong_logit"}:
            token_ids.append(int(row["target_best_wrong_token_id"]))
        else:
            raise ValueError(f"Unsupported answer scalar {scalar_name!r}; expected one of {ANSWER_SCALAR_NAMES}.")
    return token_ids


def _compute_scalar_gradient_for_pairs(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    interval_pair_rows_by_id: dict[str, dict[str, Any]],
    margin_side: str,
    scalar_name: str,
    gradient_checkpoint_role: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pairs must not be empty for answer scalar gradient computation.")
    if gradient_checkpoint_role not in {"source", "target"}:
        raise ValueError("gradient_checkpoint_role must be 'source' or 'target'.")
    if scalar_name not in ANSWER_SCALAR_NAMES:
        raise ValueError(f"Unsupported answer scalar {scalar_name!r}; expected one of {ANSWER_SCALAR_NAMES}.")
    wrong_token_ids = _wrong_token_ids_for_gradient(
        scalar_name=scalar_name,
        interval_pair_rows_by_id=interval_pair_rows_by_id,
        pairs=pairs,
        margin_side=margin_side,
        gradient_checkpoint_role=gradient_checkpoint_role,
    )
    records = _records_for_margin_side(pairs=pairs, margin_side=margin_side)
    model.eval()
    model.zero_grad(set_to_none=True)
    total_value = 0.0
    total_entries = 0
    total_batches = 0
    for start_index in range(0, len(records), batch_size):
        batch_records = records[start_index : start_index + batch_size]
        batch = move_batch_to_device(collate_symbolic_kv(batch_records, pad_token_id), device)
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        _validate_single_query_batch(batch=batch, metadata=metadata, label="answer scalar gradient")
        correct_logits = answer_logits.gather(dim=-1, index=answer_targets.unsqueeze(1)).squeeze(1)
        if scalar_name == "correct_value_logit":
            scalar_values = correct_logits
        elif scalar_name == "negative_answer_loss":
            log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
            scalar_values = log_probs.gather(dim=-1, index=answer_targets.unsqueeze(1)).squeeze(1)
        else:
            if wrong_token_ids is None:
                raise RuntimeError(f"Scalar {scalar_name} requires wrong token ids.")
            batch_wrong_ids = torch.tensor(
                wrong_token_ids[start_index : start_index + len(batch_records)],
                device=device,
                dtype=torch.long,
            )
            wrong_logits = answer_logits.gather(dim=-1, index=batch_wrong_ids.unsqueeze(1)).squeeze(1)
            if scalar_name in {"moving_answer_margin", "fixed_source_competitor_margin", "fixed_target_competitor_margin"}:
                scalar_values = correct_logits - wrong_logits
            elif scalar_name in {"source_best_wrong_logit", "target_best_wrong_logit"}:
                scalar_values = wrong_logits
            else:
                raise RuntimeError(f"Unhandled answer scalar: {scalar_name}")
        scalar_sum = scalar_values.sum()
        scalar_sum.backward()
        total_value += float(scalar_sum.detach().float().cpu().item())
        total_entries += int(scalar_values.numel())
        total_batches += 1
    if total_entries <= 0:
        raise ValueError("Answer scalar gradient records produced no answer entries.")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(float(total_entries))
    gradients, zero_gradient_parameter_names = _parameter_gradients(model=model, require_all=True)
    if zero_gradient_parameter_names:
        raise RuntimeError(
            f"Answer scalar gradient unexpectedly had zero-gradient parameters: {zero_gradient_parameter_names}"
        )
    model.zero_grad(set_to_none=True)
    return {
        "scalar_value": total_value / float(total_entries),
        "num_entries": total_entries,
        "num_pairs": len(pairs),
        "num_batches": total_batches,
        "gradients": gradients,
        "zero_gradient_parameter_names": zero_gradient_parameter_names,
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
    margin_side: str,
    switch_bucket: str,
    scalar_name: str,
    actual_summary: dict[str, Any],
    source_payload: dict[str, Any],
    source_dot_summary: dict[str, float | int | None],
    min_error_denominator: float,
    endpoint_payload: dict[str, Any] | None,
    endpoint_dot_summary: dict[str, float | int | None] | None,
) -> dict[str, Any]:
    actual_delta = float(actual_summary["actual_delta"])
    predicted_delta = float(source_dot_summary["dot"])
    residual = actual_delta - predicted_delta
    denominator = max(abs(actual_delta), min_error_denominator)
    endpoint_predicted_delta = None
    endpoint_second_order_correction = None
    endpoint_residual = None
    endpoint_relative_error = None
    endpoint_sign_match = None
    endpoint_target_gradient_dot_delta = None
    if endpoint_payload is not None or endpoint_dot_summary is not None:
        if endpoint_payload is None or endpoint_dot_summary is None:
            raise RuntimeError("Endpoint payload and dot summary must either both exist or both be None.")
        endpoint_target_gradient_dot_delta = float(endpoint_dot_summary["dot"])
        endpoint_second_order_correction = 0.5 * (endpoint_target_gradient_dot_delta - predicted_delta)
        endpoint_predicted_delta = predicted_delta + endpoint_second_order_correction
        endpoint_residual = actual_delta - endpoint_predicted_delta
        endpoint_relative_error = abs(endpoint_residual) / denominator
        endpoint_sign_match = _sign_match(actual_delta, endpoint_predicted_delta)
    return {
        "source_step": source_step,
        "target_step": target_step,
        "step_gap": target_step - source_step,
        "source_checkpoint": str(source_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "learning_rate": learning_rate,
        "split": split,
        "pair_type": pair_type,
        "margin_side": margin_side,
        "switch_bucket": switch_bucket,
        "scalar_name": scalar_name,
        "num_pairs": int(actual_summary["num_pairs"]),
        "competitor_switch_count": int(actual_summary["competitor_switch_count"]),
        "competitor_switch_fraction": float(actual_summary["competitor_switch_fraction"]),
        "source_value": float(actual_summary["source_value"]),
        "target_value": float(actual_summary["target_value"]),
        "actual_delta": actual_delta,
        "first_order_predicted_delta": predicted_delta,
        "first_order_residual": residual,
        "first_order_absolute_error": abs(residual),
        "first_order_relative_error": abs(residual) / denominator,
        "first_order_sign_match": _sign_match(actual_delta, predicted_delta),
        "source_gradient_scalar_value": float(source_payload["scalar_value"]),
        "source_gradient_num_batches": int(source_payload["num_batches"]),
        "source_gradient_num_entries": int(source_payload["num_entries"]),
        "source_gradient_l2_norm": float(source_dot_summary["right_l2_norm"]),
        "parameter_delta_l2_norm": float(source_dot_summary["left_l2_norm"]),
        "source_update_gradient_cosine": source_dot_summary["cosine"],
        "endpoint_target_gradient_dot_delta": endpoint_target_gradient_dot_delta,
        "endpoint_second_order_correction": endpoint_second_order_correction,
        "endpoint_predicted_delta": endpoint_predicted_delta,
        "endpoint_residual": endpoint_residual,
        "endpoint_relative_error": endpoint_relative_error,
        "endpoint_sign_match": endpoint_sign_match,
        "endpoint_gradient_scalar_value": None if endpoint_payload is None else float(endpoint_payload["scalar_value"]),
        "source_correct_logit": float(actual_summary["source_correct_logit"]),
        "target_correct_logit": float(actual_summary["target_correct_logit"]),
        "source_correct_log_prob": float(actual_summary["source_correct_log_prob"]),
        "target_correct_log_prob": float(actual_summary["target_correct_log_prob"]),
        "zero_source_gradient_parameter_count": len(source_payload["zero_gradient_parameter_names"]),
        "zero_source_gradient_parameter_names": source_payload["zero_gradient_parameter_names"],
        "zero_endpoint_gradient_parameter_count": None
        if endpoint_payload is None
        else len(endpoint_payload["zero_gradient_parameter_names"]),
        "zero_endpoint_gradient_parameter_names": None
        if endpoint_payload is None
        else endpoint_payload["zero_gradient_parameter_names"],
    }


def _compute_answer_scalar_residual_interval(
    *,
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    learning_rate: float,
    margin_sides: list[str],
    scalar_names: list[str],
    switch_buckets: list[str],
    metric_scopes: list[str],
    second_order_mode: str,
    batch_size: int,
    pad_token_id: int,
    top_k_wrong: int,
    device: torch.device,
    min_error_denominator: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_checkpoint = load_checkpoint(source_checkpoint_path, device)
    target_checkpoint = load_checkpoint(target_checkpoint_path, device)
    load_model_state(source_model, source_checkpoint["model_state"])
    load_model_state(target_model, target_checkpoint["model_state"])
    source_step = int(source_checkpoint["step"])
    target_step = int(target_checkpoint["step"])
    source_path_step = _checkpoint_step_from_path(source_checkpoint_path)
    target_path_step = _checkpoint_step_from_path(target_checkpoint_path)
    if source_step != source_path_step:
        raise RuntimeError(f"Source checkpoint step mismatch for {source_checkpoint_path}: payload={source_step} path={source_path_step}")
    if target_step != target_path_step:
        raise RuntimeError(f"Target checkpoint step mismatch for {target_checkpoint_path}: payload={target_step} path={target_path_step}")
    source_parameters = _model_parameter_snapshot(source_model)
    target_parameters = _model_parameter_snapshot(target_model)
    delta_parameters = _parameter_delta(
        source_parameters=source_parameters,
        target_parameters=target_parameters,
        label=f"answer scalar residual {source_step}->{target_step}",
    )

    interval_pair_rows: list[dict[str, Any]] = []
    for margin_side in margin_sides:
        source_payload = _compute_checkpoint_value_payload(
            model=source_model,
            checkpoint_path=source_checkpoint_path,
            pairs=pairs,
            vocab=vocab,
            margin_side=margin_side,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            top_k_wrong=top_k_wrong,
            device=device,
        )
        target_payload = _compute_checkpoint_value_payload(
            model=target_model,
            checkpoint_path=target_checkpoint_path,
            pairs=pairs,
            vocab=vocab,
            margin_side=margin_side,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            top_k_wrong=top_k_wrong,
            device=device,
        )
        interval_pair_rows.extend(
            _combine_interval_pair_rows(
                source_payload=source_payload,
                target_payload=target_payload,
                pairs=pairs,
                margin_side=margin_side,
                scalar_names=scalar_names,
            )
        )

    rows_by_id = {str(row["interval_pair_id"]): row for row in interval_pair_rows}
    metric_groups = _build_metric_pair_groups(pairs=pairs, metric_scopes=metric_scopes)
    metric_rows: list[dict[str, Any]] = []
    for (split, pair_type), group_pairs in sorted(metric_groups.items()):
        for margin_side in margin_sides:
            for switch_bucket in switch_buckets:
                bucket_pair_rows = _rows_for_group_and_bucket(
                    interval_pair_rows=interval_pair_rows,
                    selected_pairs=group_pairs,
                    margin_side=margin_side,
                    switch_bucket=switch_bucket,
                )
                if not bucket_pair_rows:
                    continue
                bucket_pair_ids = {str(row["interval_pair_id"]) for row in bucket_pair_rows}
                bucket_pairs = [pair for pair in group_pairs if _pair_id_for_pair(pair, margin_side) in bucket_pair_ids]
                if not bucket_pairs:
                    raise RuntimeError(
                        f"Switch bucket {switch_bucket} produced rows but no pairs for {split}/{pair_type}/{margin_side}."
                    )
                for scalar_name in scalar_names:
                    actual_summary = _actual_scalar_summary(
                        interval_pair_rows=interval_pair_rows,
                        selected_pairs=group_pairs,
                        margin_side=margin_side,
                        switch_bucket=switch_bucket,
                        scalar_name=scalar_name,
                    )
                    if actual_summary is None:
                        raise RuntimeError(
                            f"Missing actual scalar summary for non-empty bucket {split}/{pair_type}/{margin_side}/{switch_bucket}."
                        )
                    source_scalar_payload = _compute_scalar_gradient_for_pairs(
                        model=source_model,
                        pairs=bucket_pairs,
                        vocab=vocab,
                        interval_pair_rows_by_id=rows_by_id,
                        margin_side=margin_side,
                        scalar_name=scalar_name,
                        gradient_checkpoint_role="source",
                        batch_size=batch_size,
                        pad_token_id=pad_token_id,
                        device=device,
                    )
                    source_dot_summary = _gradient_dot_summary(
                        left_gradients=delta_parameters,
                        right_gradients=source_scalar_payload["gradients"],
                        label=f"answer scalar first-order {source_step}->{target_step} {split}/{pair_type}/{margin_side}/{switch_bucket}/{scalar_name}",
                    )
                    endpoint_payload = None
                    endpoint_dot_summary = None
                    if second_order_mode == "endpoint_gradient":
                        endpoint_payload = _compute_scalar_gradient_for_pairs(
                            model=target_model,
                            pairs=bucket_pairs,
                            vocab=vocab,
                            interval_pair_rows_by_id=rows_by_id,
                            margin_side=margin_side,
                            scalar_name=scalar_name,
                            gradient_checkpoint_role="target",
                            batch_size=batch_size,
                            pad_token_id=pad_token_id,
                            device=device,
                        )
                        endpoint_dot_summary = _gradient_dot_summary(
                            left_gradients=delta_parameters,
                            right_gradients=endpoint_payload["gradients"],
                            label=f"answer scalar endpoint-gradient {source_step}->{target_step} {split}/{pair_type}/{margin_side}/{switch_bucket}/{scalar_name}",
                        )
                    elif second_order_mode != "none":
                        raise ValueError(
                            f"Unsupported second_order_mode {second_order_mode!r}; expected one of {ANSWER_SCALAR_SECOND_ORDER_MODES}."
                        )
                    metric_rows.append(
                        _metric_row(
                            source_step=source_step,
                            target_step=target_step,
                            source_checkpoint=source_checkpoint_path,
                            target_checkpoint=target_checkpoint_path,
                            learning_rate=learning_rate,
                            split=split,
                            pair_type=pair_type,
                            margin_side=margin_side,
                            switch_bucket=switch_bucket,
                            scalar_name=scalar_name,
                            actual_summary=actual_summary,
                            source_payload=source_scalar_payload,
                            source_dot_summary=source_dot_summary,
                            min_error_denominator=min_error_denominator,
                            endpoint_payload=endpoint_payload,
                            endpoint_dot_summary=endpoint_dot_summary,
                        )
                    )
    return metric_rows, interval_pair_rows


def _summarize_answer_scalar_residual_diagnosis(
    *,
    metric_rows: list[dict[str, Any]],
    top_k_rows: int,
) -> dict[str, Any]:
    if top_k_rows <= 0:
        raise ValueError("top_k_rows must be positive.")
    if not metric_rows:
        raise ValueError("Cannot summarize answer scalar residual diagnosis without metric rows.")
    aggregate_rows = [
        row
        for row in metric_rows
        if str(row["split"]) == "__all__"
        and str(row["pair_type"]) == "__all__"
        and str(row["switch_bucket"]) in {"all", "same_competitor", "competitor_switch"}
    ]
    if not aggregate_rows:
        raise RuntimeError("Answer scalar residual diagnosis has no aggregate metric rows.")
    scalar_summaries: list[dict[str, Any]] = []
    for scalar_name in sorted({str(row["scalar_name"]) for row in aggregate_rows}):
        for switch_bucket in sorted({str(row["switch_bucket"]) for row in aggregate_rows}):
            rows = [
                row
                for row in aggregate_rows
                if str(row["scalar_name"]) == scalar_name and str(row["switch_bucket"]) == switch_bucket
            ]
            if not rows:
                continue
            actual_sum = sum(float(row["actual_delta"]) for row in rows)
            first_order_sum = sum(float(row["first_order_predicted_delta"]) for row in rows)
            first_order_residual_sum = actual_sum - first_order_sum
            endpoint_rows = [row for row in rows if row["endpoint_predicted_delta"] is not None]
            endpoint_predicted_sum = None
            endpoint_residual_sum = None
            endpoint_mean_abs_error = None
            if endpoint_rows:
                if len(endpoint_rows) != len(rows):
                    raise RuntimeError("Some but not all rows have endpoint second-order estimates.")
                endpoint_predicted_sum = sum(float(row["endpoint_predicted_delta"]) for row in endpoint_rows)
                endpoint_residual_sum = actual_sum - endpoint_predicted_sum
                endpoint_mean_abs_error = _mean(
                    [abs(float(row["endpoint_residual"])) for row in endpoint_rows],
                    label=f"{scalar_name}/{switch_bucket} endpoint_abs_error",
                )
            scalar_summaries.append(
                {
                    "scalar_name": scalar_name,
                    "switch_bucket": switch_bucket,
                    "num_rows": len(rows),
                    "num_pairs_mean": _mean([float(row["num_pairs"]) for row in rows], label="num_pairs_mean"),
                    "competitor_switch_fraction_mean": _mean(
                        [float(row["competitor_switch_fraction"]) for row in rows],
                        label="competitor_switch_fraction_mean",
                    ),
                    "actual_delta_sum": actual_sum,
                    "first_order_predicted_delta_sum": first_order_sum,
                    "first_order_residual_sum": first_order_residual_sum,
                    "first_order_mean_abs_error": _mean(
                        [abs(float(row["first_order_residual"])) for row in rows],
                        label=f"{scalar_name}/{switch_bucket} first_order_abs_error",
                    ),
                    "first_order_sign_match_fraction": _mean(
                        [1.0 if bool(row["first_order_sign_match"]) else 0.0 for row in rows],
                        label=f"{scalar_name}/{switch_bucket} first_order_sign_match",
                    ),
                    "endpoint_predicted_delta_sum": endpoint_predicted_sum,
                    "endpoint_residual_sum": endpoint_residual_sum,
                    "endpoint_mean_abs_error": endpoint_mean_abs_error,
                    "endpoint_sign_match_fraction": None
                    if not endpoint_rows
                    else _mean(
                        [1.0 if bool(row["endpoint_sign_match"]) else 0.0 for row in endpoint_rows],
                        label=f"{scalar_name}/{switch_bucket} endpoint_sign_match",
                    ),
                }
            )
    all_bucket_rows = [row for row in aggregate_rows if str(row["switch_bucket"]) == "all"]
    return {
        "num_intervals": len({(int(row["source_step"]), int(row["target_step"])) for row in metric_rows}),
        "intervals": sorted({f"{int(row['source_step'])}->{int(row['target_step'])}" for row in metric_rows}),
        "target_steps": sorted({int(row["target_step"]) for row in metric_rows}),
        "scalar_summaries": sorted(
            scalar_summaries,
            key=lambda row: (str(row["scalar_name"]), str(row["switch_bucket"])),
        ),
        "ranked_by_first_order_mean_abs_error": sorted(
            [row for row in scalar_summaries if str(row["switch_bucket"]) == "all"],
            key=lambda row: float(row["first_order_mean_abs_error"]),
        ),
        "ranked_by_endpoint_mean_abs_error": sorted(
            [
                row
                for row in scalar_summaries
                if str(row["switch_bucket"]) == "all" and row["endpoint_mean_abs_error"] is not None
            ],
            key=lambda row: float(row["endpoint_mean_abs_error"]),
        ),
        "worst_first_order_rows": sorted(
            all_bucket_rows,
            key=lambda row: abs(float(row["first_order_residual"])),
            reverse=True,
        )[:top_k_rows],
        "worst_endpoint_rows": sorted(
            [row for row in all_bucket_rows if row["endpoint_residual"] is not None],
            key=lambda row: abs(float(row["endpoint_residual"])),
            reverse=True,
        )[:top_k_rows],
    }


def _plot_scalar_residual_summary(
    *,
    summary: dict[str, Any],
    output_path: Path,
) -> Path | None:
    rows = [row for row in summary["scalar_summaries"] if str(row["switch_bucket"]) == "all"]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    labels = [str(row["scalar_name"]) for row in rows]
    first_order_errors = [float(row["first_order_mean_abs_error"]) for row in rows]
    endpoint_errors = [
        None if row["endpoint_mean_abs_error"] is None else float(row["endpoint_mean_abs_error"])
        for row in rows
    ]
    x_positions = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(10, 1.25 * len(rows)), 5))
    ax.bar([x - 0.2 for x in x_positions], first_order_errors, width=0.4, label="first-order", color="#376f8f")
    if any(value is not None for value in endpoint_errors):
        ax.bar(
            [x + 0.2 for x in x_positions],
            [0.0 if value is None else value for value in endpoint_errors],
            width=0.4,
            label="endpoint-gradient",
            color="#6f8f37",
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Scalar residual diagnosis: mean absolute error")
    ax.set_ylabel("mean |actual - predicted|")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_switch_bucket_errors(
    *,
    summary: dict[str, Any],
    output_path: Path,
) -> Path | None:
    rows = [
        row
        for row in summary["scalar_summaries"]
        if str(row["scalar_name"]) in {"moving_answer_margin", "fixed_source_competitor_margin", "negative_answer_loss"}
    ]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    labels = [f"{row['scalar_name']}\\n{row['switch_bucket']}" for row in rows]
    values = [float(row["first_order_mean_abs_error"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(rows)), 5))
    ax.bar(range(len(rows)), values, color="#8f6237")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("First-order error by competitor-switch bucket")
    ax.set_ylabel("mean |actual - first-order|")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_answer_scalar_residual_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    lines = [
        "# Answer Scalar Residual Diagnosis",
        "",
        "## Calculation",
        "",
        "This report diagnoses why first-order answer-margin attribution leaves residual error.",
        "",
        "```text",
        "first_order = grad_theta scalar(theta_source) . (theta_target - theta_source)",
        "endpoint_gradient = 0.5 * (grad scalar(theta_source) + grad scalar(theta_target)) . Delta theta",
        "residual = actual_delta - predicted_delta",
        "```",
        "",
        "The endpoint-gradient term is a directional curvature diagnostic, not a full Hessian calculation.",
        "",
        "## Run",
        "",
        f"- scalar names: `{report['scalar_names']}`",
        f"- margin sides: `{report['margin_sides']}`",
        f"- switch buckets: `{report['switch_buckets']}`",
        f"- metric scopes: `{report['metric_scopes']}`",
        f"- second-order mode: `{report['second_order_mode']}`",
        f"- intervals: `{summary['intervals']}`",
        "",
        "## Scalar Error Ranking",
        "",
        "| rank | scalar | first-order mean abs error | first-order sign match | endpoint mean abs error | endpoint sign match |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summary["ranked_by_first_order_mean_abs_error"], start=1):
        endpoint_error = row["endpoint_mean_abs_error"]
        endpoint_error_text = "" if endpoint_error is None else f"{float(endpoint_error):.6g}"
        endpoint_sign = row["endpoint_sign_match_fraction"]
        endpoint_sign_text = "" if endpoint_sign is None else f"{float(endpoint_sign):.3f}"
        lines.append(
            "| {rank} | `{scalar}` | {first_error:.6g} | {first_sign:.3f} | {endpoint_error} | {endpoint_sign} |".format(
                rank=rank,
                scalar=row["scalar_name"],
                first_error=float(row["first_order_mean_abs_error"]),
                first_sign=float(row["first_order_sign_match_fraction"]),
                endpoint_error=endpoint_error_text,
                endpoint_sign=endpoint_sign_text,
            )
        )
    lines.extend(
        [
            "",
            "## Branch Stability",
            "",
            "| scalar | bucket | rows | switch fraction | actual sum | first-order sum | first residual sum | endpoint sum | endpoint residual sum |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["scalar_summaries"]:
        endpoint_sum = row["endpoint_predicted_delta_sum"]
        endpoint_residual = row["endpoint_residual_sum"]
        endpoint_sum_text = "" if endpoint_sum is None else f"{float(endpoint_sum):.6g}"
        endpoint_residual_text = "" if endpoint_residual is None else f"{float(endpoint_residual):.6g}"
        lines.append(
            "| `{scalar}` | `{bucket}` | {rows} | {switch:.3f} | {actual:.6g} | {first:.6g} | {residual:.6g} | {endpoint} | {endpoint_residual} |".format(
                scalar=row["scalar_name"],
                bucket=row["switch_bucket"],
                rows=int(row["num_rows"]),
                switch=float(row["competitor_switch_fraction_mean"]),
                actual=float(row["actual_delta_sum"]),
                first=float(row["first_order_predicted_delta_sum"]),
                residual=float(row["first_order_residual_sum"]),
                endpoint=endpoint_sum_text,
                endpoint_residual=endpoint_residual_text,
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- metric rows: `{report['metric_rows_path']}`",
            f"- interval pair rows: `{report['interval_pair_rows_path']}`",
            f"- pair metadata rows: `{report['pair_metadata_rows_path']}`",
        ]
    )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for label, plot_path in plot_paths.items():
            lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_answer_scalar_residual_diagnosis(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    pair_types: list[str],
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    margin_sides: list[str] | None = None,
    scalar_names: list[str] | None = None,
    switch_buckets: list[str] | None = None,
    metric_scopes: list[str] | None = None,
    second_order_mode: str = "none",
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    top_k_wrong: int = 5,
    top_k_rows: int = 24,
    min_error_denominator: float = 1.0e-9,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Path]]:
    if not pair_types:
        raise ValueError("answer-scalar-residual-diagnosis requires at least one pair type.")
    if max_pairs_per_type <= 0:
        raise ValueError("max_pairs_per_type must be positive.")
    if min_pairs_per_type <= 0:
        raise ValueError("min_pairs_per_type must be positive.")
    if top_k_wrong <= 0:
        raise ValueError("top_k_wrong must be positive.")
    if top_k_rows <= 0:
        raise ValueError("top_k_rows must be positive.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    resolved_margin_sides = _resolve_unique_values(
        values=margin_sides,
        default_values=["clean"],
        allowed_values=ANSWER_SCALAR_MARGIN_SIDES,
        label="margin side",
    )
    resolved_scalar_names = _resolve_unique_values(
        values=scalar_names,
        default_values=list(ANSWER_SCALAR_NAMES),
        allowed_values=ANSWER_SCALAR_NAMES,
        label="answer scalar",
    )
    resolved_switch_buckets = _resolve_unique_values(
        values=switch_buckets,
        default_values=list(ANSWER_SCALAR_SWITCH_BUCKETS),
        allowed_values=ANSWER_SCALAR_SWITCH_BUCKETS,
        label="switch bucket",
    )
    resolved_metric_scopes = _resolve_unique_values(
        values=metric_scopes,
        default_values=["aggregate"],
        allowed_values=ANSWER_SCALAR_METRIC_SCOPES,
        label="metric scope",
    )
    if second_order_mode not in ANSWER_SCALAR_SECOND_ORDER_MODES:
        raise ValueError(
            f"Unsupported second_order_mode {second_order_mode!r}; expected one of {ANSWER_SCALAR_SECOND_ORDER_MODES}."
        )
    pair_types = sorted(set(pair_types), key=pair_types.index)

    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("answer-scalar-residual-diagnosis requires at least two checkpoints.")
    pairs, pair_construction = _build_route_competition_pairs(
        probe_set_path=probe_set_path,
        spec=spec,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Answer scalar residual diagnosis constructed no pairs.")

    source_model = build_model(spec.model, len(vocab.tokens), device)
    target_model = build_model(spec.model, len(vocab.tokens), device)

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows_path = output_dir / "answer_scalar_residual_diagnosis_rows.jsonl"
    interval_pair_rows_path = output_dir / "answer_scalar_residual_diagnosis_pair_rows.jsonl"
    pair_metadata_rows_path = output_dir / "answer_scalar_residual_diagnosis_pair_metadata.jsonl"
    progress_path = output_dir / "answer_scalar_residual_diagnosis_progress.json"
    write_jsonl(pair_metadata_rows_path, [_pair_metadata(pair) for pair in pairs])

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[answer-scalar-residual-diagnosis] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(pairs)} "
        f"pair_types={pair_types} scalars={resolved_scalar_names} switch_buckets={resolved_switch_buckets} "
        f"second_order={second_order_mode} device={device_name}",
        flush=True,
    )

    all_metric_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        learning_rate = _compute_learning_rate(spec.optimization, source_step)
        print(
            "[answer-scalar-residual-diagnosis] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )
        metric_rows, interval_pair_rows = _compute_answer_scalar_residual_interval(
            source_model=source_model,
            target_model=target_model,
            source_checkpoint_path=source_checkpoint_path,
            target_checkpoint_path=target_checkpoint_path,
            pairs=pairs,
            vocab=vocab,
            learning_rate=learning_rate,
            margin_sides=resolved_margin_sides,
            scalar_names=resolved_scalar_names,
            switch_buckets=resolved_switch_buckets,
            metric_scopes=resolved_metric_scopes,
            second_order_mode=second_order_mode,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            top_k_wrong=top_k_wrong,
            device=device,
            min_error_denominator=min_error_denominator,
        )
        for row in metric_rows:
            append_jsonl(metric_rows_path, row)
        for row in interval_pair_rows:
            append_jsonl(interval_pair_rows_path, row)
        all_metric_rows.extend(metric_rows)
        main_row = next(
            row
            for row in metric_rows
            if str(row["split"]) == "__all__"
            and str(row["pair_type"]) == "__all__"
            and str(row["switch_bucket"]) == "all"
            and str(row["scalar_name"]) == resolved_scalar_names[0]
            and str(row["margin_side"]) == resolved_margin_sides[0]
        )
        endpoint_error = main_row["endpoint_relative_error"]
        endpoint_text = "" if endpoint_error is None else f" endpoint_relative_error={float(endpoint_error):.6g}"
        print(
            "[answer-scalar-residual-diagnosis] finished "
            f"{source_step}->{target_step} scalar={main_row['scalar_name']} "
            f"actual_delta={float(main_row['actual_delta']):.6g} "
            f"first_order={float(main_row['first_order_predicted_delta']):.6g} "
            f"first_relative_error={float(main_row['first_order_relative_error']):.6g}"
            f"{endpoint_text}",
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
                "interval_pair_rows_path": str(interval_pair_rows_path),
                "pair_metadata_rows_path": str(pair_metadata_rows_path),
            },
        )

    summary = _summarize_answer_scalar_residual_diagnosis(
        metric_rows=all_metric_rows,
        top_k_rows=top_k_rows,
    )
    plot_paths: dict[str, Path] = {}
    scalar_residual_plot = _plot_scalar_residual_summary(
        summary=summary,
        output_path=output_dir / "answer_scalar_residual_summary.svg",
    )
    if scalar_residual_plot is not None:
        plot_paths["scalar_residual_summary"] = scalar_residual_plot
    switch_bucket_plot = _plot_switch_bucket_errors(
        summary=summary,
        output_path=output_dir / "answer_scalar_switch_bucket_errors.svg",
    )
    if switch_bucket_plot is not None:
        plot_paths["switch_bucket_errors"] = switch_bucket_plot

    report_path = output_dir / "answer_scalar_residual_diagnosis_report.json"
    markdown_path = output_dir / "answer_scalar_residual_diagnosis_report.md"
    report = {
        "schema_version": ANSWER_SCALAR_RESIDUAL_DIAGNOSIS_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names": resolved_scalar_names,
        "switch_buckets": resolved_switch_buckets,
        "metric_scopes": resolved_metric_scopes,
        "second_order_mode": second_order_mode,
        "top_k_wrong": top_k_wrong,
        "top_k_rows": top_k_rows,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "min_error_denominator": min_error_denominator,
        "calculation": {
            "moving_answer_margin": "logit(correct) - max_wrong_value_logit at each checkpoint",
            "fixed_source_competitor_margin": "logit(correct) - logit(source checkpoint best wrong value)",
            "fixed_target_competitor_margin": "logit(correct) - logit(target checkpoint best wrong value)",
            "negative_answer_loss": "log softmax probability assigned to the correct answer token",
            "first_order": "grad_theta scalar(theta_source) . (theta_target - theta_source)",
            "endpoint_gradient": (
                "0.5 * (grad scalar(theta_source) + grad scalar(theta_target)) . Delta theta; "
                "a directional curvature diagnostic, not a full Hessian artifact"
            ),
        },
        "pair_construction": pair_construction,
        "metric_rows_path": str(metric_rows_path),
        "interval_pair_rows_path": str(interval_pair_rows_path),
        "pair_metadata_rows_path": str(pair_metadata_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_answer_scalar_residual_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_intervals": len(intervals),
            "total_intervals": len(intervals),
            "last_target_step": int(summary["target_steps"][-1]),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "metric_rows_path": str(metric_rows_path),
            "interval_pair_rows_path": str(interval_pair_rows_path),
            "pair_metadata_rows_path": str(pair_metadata_rows_path),
        },
    )
    print(
        f"[answer-scalar-residual-diagnosis] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, metric_rows_path, interval_pair_rows_path, pair_metadata_rows_path, plot_paths
