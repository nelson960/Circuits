from __future__ import annotations

from typing import Any

import torch

from circuit.runtime import move_batch_to_device


def extract_answer_logits(
    logits: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    valid_queries = batch["query_mask"].nonzero(as_tuple=False)
    if valid_queries.numel() == 0:
        raise RuntimeError("Batch contains no valid query events.")
    rows = valid_queries[:, 0]
    query_indices = valid_queries[:, 1]
    answer_positions = batch["answer_token_positions"][rows, query_indices]
    prediction_positions = answer_positions - 1
    answer_logits = logits[rows, prediction_positions, :]
    answer_targets = batch["input_ids"][rows, answer_positions]
    metadata = {
        "rows": rows,
        "query_indices": query_indices,
        "answer_positions": answer_positions,
        "prediction_positions": prediction_positions,
        "query_key_positions": batch["query_key_positions"][rows, query_indices],
        "support_value_positions": batch["support_value_positions"][rows, query_indices],
    }
    return answer_logits, answer_targets, metadata


def compute_qrw_batch(
    *,
    logits: torch.Tensor,
    attentions: list[torch.Tensor],
    batch: dict[str, Any],
    value_token_ids: torch.Tensor,
) -> dict[str, float]:
    if not attentions:
        raise ValueError("compute_qrw_batch requires attention tensors.")
    answer_logits, answer_targets, metadata = extract_answer_logits(logits, batch)
    q_scores: list[torch.Tensor] = []
    r_scores: list[torch.Tensor] = []
    for batch_index in range(answer_logits.size(0)):
        per_layer_head_q: list[torch.Tensor] = []
        per_layer_head_r: list[torch.Tensor] = []
        attention_row = int(metadata["rows"][batch_index].item())
        prediction_position = int(metadata["prediction_positions"][batch_index].item())
        query_position = int(metadata["query_key_positions"][batch_index].item())
        support_position = int(metadata["support_value_positions"][batch_index].item())
        for attention in attentions:
            per_layer_head_q.append(attention[attention_row, :, prediction_position, query_position])
            per_layer_head_r.append(attention[attention_row, :, prediction_position, support_position])
        q_scores.append(torch.cat(per_layer_head_q).max())
        r_scores.append(torch.cat(per_layer_head_r).max())
    q = torch.stack(q_scores).mean().item()
    r = torch.stack(r_scores).mean().item()

    value_logits = answer_logits.index_select(dim=-1, index=value_token_ids)
    row_index = torch.arange(answer_logits.size(0), device=answer_logits.device)
    target_in_value_space = (value_token_ids.unsqueeze(0) == answer_targets.unsqueeze(1)).nonzero(as_tuple=False)
    if target_in_value_space.size(0) != answer_targets.size(0):
        raise RuntimeError("Failed to locate answer targets inside the value-token subset.")
    correct_logits = value_logits[row_index, target_in_value_space[:, 1]]
    masked_value_logits = value_logits.clone()
    masked_value_logits[row_index, target_in_value_space[:, 1]] = torch.finfo(masked_value_logits.dtype).min
    best_incorrect = masked_value_logits.max(dim=-1).values
    w = (correct_logits - best_incorrect).mean().item()

    predictions = answer_logits.argmax(dim=-1)
    answer_accuracy = (predictions == answer_targets).float().mean().item()
    return {
        "q": q,
        "r": r,
        "w": w,
        "answer_accuracy": answer_accuracy,
    }


@torch.no_grad()
def collect_analysis_batches(
    data_loader: torch.utils.data.DataLoader[Any],
    *,
    device: torch.device,
    max_batches: int,
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(data_loader):
        if batch_index >= max_batches:
            break
        collected.append(move_batch_to_device(batch, device))
    if not collected:
        raise RuntimeError("No batches collected for analysis.")
    return collected


@torch.no_grad()
def compute_head_localization(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
) -> list[dict[str, float | int]]:
    head_metrics: dict[tuple[int, int], dict[str, float | int]] = {}
    for batch in batches:
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], return_attentions=True)
        if outputs.attentions is None:
            raise RuntimeError("Model did not return attentions.")
        _, _, metadata = extract_answer_logits(outputs.logits, batch)
        for layer_index, attention in enumerate(outputs.attentions):
            n_heads = attention.size(1)
            for head_index in range(n_heads):
                key = (layer_index, head_index)
                if key not in head_metrics:
                    head_metrics[key] = {
                        "layer": layer_index,
                        "head": head_index,
                        "query_attention_sum": 0.0,
                        "support_attention_sum": 0.0,
                        "num_examples": 0,
                    }
                for query_event_index in range(metadata["rows"].size(0)):
                    attention_row = int(metadata["rows"][query_event_index].item())
                    prediction_position = int(metadata["prediction_positions"][query_event_index].item())
                    query_position = int(metadata["query_key_positions"][query_event_index].item())
                    support_position = int(metadata["support_value_positions"][query_event_index].item())
                    head_metrics[key]["query_attention_sum"] += float(
                        attention[attention_row, head_index, prediction_position, query_position].item()
                    )
                    head_metrics[key]["support_attention_sum"] += float(
                        attention[attention_row, head_index, prediction_position, support_position].item()
                    )
                    head_metrics[key]["num_examples"] += 1
    summary: list[dict[str, float | int]] = []
    for key in sorted(head_metrics):
        metric = head_metrics[key]
        num_examples = int(metric["num_examples"])
        summary.append(
            {
                "layer": int(metric["layer"]),
                "head": int(metric["head"]),
                "query_attention_mean": float(metric["query_attention_sum"]) / num_examples,
                "support_attention_mean": float(metric["support_attention_sum"]) / num_examples,
                "num_examples": num_examples,
            }
        )
    return summary


@torch.no_grad()
def compute_head_ablation_importance(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
) -> list[dict[str, float | int]]:
    baseline_correct = 0
    baseline_total = 0
    for batch in batches:
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        answer_logits, answer_targets, _ = extract_answer_logits(outputs.logits, batch)
        baseline_correct += (answer_logits.argmax(dim=-1) == answer_targets).sum().item()
        baseline_total += answer_targets.numel()
    if baseline_total == 0:
        raise RuntimeError("No answer targets available for ablation analysis.")
    baseline_accuracy = baseline_correct / baseline_total

    n_layers = len(model.blocks)
    n_heads = model.spec.n_heads
    summary: list[dict[str, float | int]] = []
    for layer_index in range(n_layers):
        for head_index in range(n_heads):
            ablated_correct = 0
            ablated_total = 0
            head_mask = {
                layer: torch.ones(n_heads, device=batches[0]["input_ids"].device)
                for layer in range(n_layers)
            }
            head_mask[layer_index][head_index] = 0.0
            for batch in batches:
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    head_mask=head_mask,
                )
                answer_logits, answer_targets, _ = extract_answer_logits(outputs.logits, batch)
                ablated_correct += (answer_logits.argmax(dim=-1) == answer_targets).sum().item()
                ablated_total += answer_targets.numel()
            ablated_accuracy = ablated_correct / ablated_total
            summary.append(
                {
                    "layer": layer_index,
                    "head": head_index,
                    "baseline_accuracy": baseline_accuracy,
                    "ablated_accuracy": ablated_accuracy,
                    "accuracy_drop": baseline_accuracy - ablated_accuracy,
                }
            )
    return summary


def summarize_formation_trace(
    *,
    rows: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    births: dict[str, int | None] = {}
    for metric_name, threshold in thresholds.items():
        birth_step = None
        for row in rows:
            if float(row[metric_name]) >= threshold:
                birth_step = int(row["step"])
                break
        births[metric_name] = birth_step
    return {
        "birth_steps": births,
        "num_checkpoints": len(rows),
    }
