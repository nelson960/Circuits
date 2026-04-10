from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from circut.analysis.formation import compute_qrw_batch, extract_answer_logits
from circut.runtime import compute_lm_loss, move_batch_to_device


def _build_token_role_map(record: dict[str, Any]) -> dict[int, str]:
    role_by_position: dict[int, str] = {}
    for step in record["steps"]:
        positions = step["positions"]
        op = step["op"]
        if op == "write":
            role_by_position[int(positions["op"])] = "op_write"
            role_by_position[int(positions["key"])] = "key_write"
            role_by_position[int(positions["value"])] = "value_write"
            continue
        if op == "read":
            role_by_position[int(positions["op"])] = "op_read"
            role_by_position[int(positions["key"])] = "key_read"
            role_by_position[int(positions["answer"])] = "value_answer"
            continue
        raise RuntimeError(f"Unknown step op: {op}")
    eos_position = len(record["tokens"]) - 1
    role_by_position[eos_position] = "eos"
    return role_by_position


@torch.no_grad()
def evaluate_split(
    *,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader[Any],
    device: torch.device,
    pad_token_id: int,
    value_token_ids: list[int],
    max_batches: int | None,
    include_analysis: bool,
) -> dict[str, Any]:
    value_token_ids_tensor = torch.tensor(value_token_ids, device=device, dtype=torch.long)
    total_loss = 0.0
    total_token_accuracy = 0.0
    total_answer_correct = 0
    total_examples = 0
    num_batches = 0
    analysis_accumulator = defaultdict(float)
    slice_totals: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    token_role_correct: dict[str, int] = defaultdict(int)
    token_role_total: dict[str, int] = defaultdict(int)

    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_attentions=include_analysis,
        )
        loss, token_accuracy = compute_lm_loss(
            logits=outputs.logits,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pad_token_id=pad_token_id,
        )
        answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        answer_predictions = answer_logits.argmax(dim=-1)
        answer_correct = answer_predictions == answer_targets
        token_predictions = outputs.logits[:, :-1, :].argmax(dim=-1)
        token_targets = batch["input_ids"][:, 1:]
        token_mask = batch["attention_mask"][:, 1:]

        total_loss += loss.item()
        total_token_accuracy += token_accuracy.item()
        total_answer_correct += answer_correct.sum().item()
        total_examples += answer_targets.numel()
        num_batches += 1

        for row_index, record in enumerate(batch["records"]):
            role_by_position = _build_token_role_map(record)
            sequence_length = len(record["token_ids"])
            for shifted_index in range(sequence_length - 1):
                if not bool(token_mask[row_index, shifted_index].item()):
                    continue
                token_position = shifted_index + 1
                if token_position not in role_by_position:
                    raise RuntimeError(
                        f"Missing token role annotation for token position {token_position} in sample {record['sample_id']}"
                    )
                role = role_by_position[token_position]
                token_role_total[role] += 1
                if int(token_predictions[row_index, shifted_index].item()) == int(token_targets[row_index, shifted_index].item()):
                    token_role_correct[role] += 1

        rows = metadata["rows"]
        query_indices = metadata["query_indices"]
        for axis_name, axis_values in batch["axes"].items():
            for event_index, row in enumerate(rows.tolist()):
                axis_value = int(axis_values[row].item())
                bucket = slice_totals[axis_name][axis_value]
                bucket[0] += int(answer_correct[event_index].item())
                bucket[1] += 1
        if "query_axes" in batch:
            for axis_name, axis_values in batch["query_axes"].items():
                for event_index, (row, query_index) in enumerate(zip(rows.tolist(), query_indices.tolist(), strict=True)):
                    axis_value = int(axis_values[row, query_index].item())
                    bucket = slice_totals[axis_name][axis_value]
                    bucket[0] += int(answer_correct[event_index].item())
                    bucket[1] += 1

        if include_analysis:
            if outputs.attentions is None:
                raise RuntimeError("Requested analysis metrics, but model did not return attentions.")
            qrw = compute_qrw_batch(
                logits=outputs.logits,
                attentions=outputs.attentions,
                batch=batch,
                value_token_ids=value_token_ids_tensor,
            )
            for key, value in qrw.items():
                analysis_accumulator[key] += value

    if num_batches == 0:
        raise RuntimeError("Evaluation produced zero batches.")
    token_role_accuracy = {
        role: token_role_correct[role] / token_role_total[role]
        for role in sorted(token_role_total)
    }
    token_role_fraction = {
        role: token_role_total[role] / sum(token_role_total.values())
        for role in sorted(token_role_total)
    }
    slice_accuracy = {
        axis_name: {
            str(axis_value): correct / total
            for axis_value, (correct, total) in sorted(axis_buckets.items())
        }
        for axis_name, axis_buckets in slice_totals.items()
    }
    result = {
        "loss": total_loss / num_batches,
        "token_accuracy": total_token_accuracy / num_batches,
        "answer_accuracy": total_answer_correct / total_examples,
        "num_examples": total_examples,
        "write_key_accuracy": token_role_accuracy["key_write"],
        "read_key_accuracy": token_role_accuracy["key_read"],
        "write_value_accuracy": token_role_accuracy["value_write"],
        "token_role_accuracy": token_role_accuracy,
        "token_role_fraction": token_role_fraction,
        "slice_accuracy": slice_accuracy,
    }
    if include_analysis:
        result.update({key: value / num_batches for key, value in analysis_accumulator.items()})
    return result
