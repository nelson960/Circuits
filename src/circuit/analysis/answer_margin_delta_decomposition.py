from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    _build_route_competition_pairs,
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
    _validate_single_query_batch,
    _value_margin,
)
from circuit.analysis.shared_feature_dynamics import _import_matplotlib
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import append_jsonl, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.train import _compute_learning_rate
from circuit.vocab import Vocabulary


ANSWER_MARGIN_DELTA_DECOMPOSITION_SCHEMA_VERSION = 1
ANSWER_MARGIN_SIDES = ["clean", "corrupted"]


def _mean(values: list[float], *, label: str) -> float:
    if not values:
        raise ValueError(f"Cannot compute mean for empty values: {label}")
    return sum(values) / float(len(values))


def _resolve_margin_sides(margin_sides: list[str] | None) -> list[str]:
    if margin_sides is None:
        return ["clean"]
    if not margin_sides:
        raise ValueError("margin_sides must not be empty when provided.")
    resolved: list[str] = []
    for side in margin_sides:
        if side not in ANSWER_MARGIN_SIDES:
            raise ValueError(f"Unsupported margin side {side!r}; expected one of {ANSWER_MARGIN_SIDES}.")
        if side not in resolved:
            resolved.append(side)
    return resolved


def _records_for_margin_side(*, pairs: list[dict[str, Any]], margin_side: str) -> list[dict[str, Any]]:
    if margin_side not in ANSWER_MARGIN_SIDES:
        raise ValueError(f"Unsupported margin side {margin_side!r}; expected one of {ANSWER_MARGIN_SIDES}.")
    if not pairs:
        raise ValueError("Cannot build margin records from an empty pair list.")
    record_key = f"{margin_side}_record"
    return [pair[record_key] for pair in pairs]


def _row_matches_group(*, row: dict[str, Any], split: str, pair_type: str, margin_side: str) -> bool:
    split_matches = split == "__all__" or str(row["split"]) == split
    pair_type_matches = pair_type == "__all__" or str(row["pair_type"]) == pair_type
    return split_matches and pair_type_matches and str(row["margin_side"]) == margin_side


def _compute_answer_margin_rows_for_loaded_model(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    checkpoint_step: int,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    margin_side: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if margin_side not in ANSWER_MARGIN_SIDES:
        raise ValueError(f"Unsupported margin side {margin_side!r}; expected one of {ANSWER_MARGIN_SIDES}.")
    if not pairs:
        raise ValueError("pairs must not be empty for answer-margin row computation.")
    model.eval()
    rows: list[dict[str, Any]] = []
    value_token_ids = torch.tensor(vocab.value_token_ids, device=device, dtype=torch.long)
    record_key = f"{margin_side}_record"
    with torch.no_grad():
        for start_index in range(0, len(pairs), batch_size):
            pair_batch = pairs[start_index : start_index + batch_size]
            records = [pair[record_key] for pair in pair_batch]
            batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=metadata, label="answer-margin delta decomposition")
            answer_margins = _value_margin(answer_logits, answer_targets, value_token_ids)
            answer_losses = torch.nn.functional.cross_entropy(answer_logits, answer_targets, reduction="none")
            predictions = answer_logits.argmax(dim=-1)
            for pair_index, pair in enumerate(pair_batch):
                rows.append(
                    {
                        "step": checkpoint_step,
                        "checkpoint": str(checkpoint_path),
                        "pair_id": str(pair["pair_id"]),
                        "split": str(pair["split"]),
                        "pair_type": str(pair["pair_type"]),
                        "margin_side": margin_side,
                        "source_sample_id": str(pair["source_sample_id"]),
                        "source_query_index": int(pair["source_query_index"]),
                        "clean_transfer_token": str(pair["clean_transfer_token"]),
                        "corrupted_transfer_token": str(pair["corrupted_transfer_token"]),
                        "answer_target_id": int(answer_targets[pair_index].detach().cpu().item()),
                        "answer_prediction_id": int(predictions[pair_index].detach().cpu().item()),
                        "answer_correct": bool(
                            (predictions[pair_index] == answer_targets[pair_index]).detach().cpu().item()
                        ),
                        "answer_margin": float(answer_margins[pair_index].detach().float().cpu().item()),
                        "answer_loss": float(answer_losses[pair_index].detach().float().cpu().item()),
                    }
                )
    return rows


def _compute_answer_margin_gradient_for_pairs(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    margin_side: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if margin_side not in ANSWER_MARGIN_SIDES:
        raise ValueError(f"Unsupported margin side {margin_side!r}; expected one of {ANSWER_MARGIN_SIDES}.")
    if not pairs:
        raise ValueError("pairs must not be empty for answer-margin gradient computation.")
    model.eval()
    model.zero_grad(set_to_none=True)
    value_token_ids = torch.tensor(vocab.value_token_ids, device=device, dtype=torch.long)
    records = _records_for_margin_side(pairs=pairs, margin_side=margin_side)
    total_margin = 0.0
    total_loss = 0.0
    total_correct = 0
    total_entries = 0
    num_batches = 0
    for start_index in range(0, len(records), batch_size):
        batch_records = records[start_index : start_index + batch_size]
        batch = move_batch_to_device(collate_symbolic_kv(batch_records, pad_token_id), device)
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        _validate_single_query_batch(batch=batch, metadata=metadata, label="answer-margin gradient")
        answer_margins = _value_margin(answer_logits, answer_targets, value_token_ids)
        answer_losses = torch.nn.functional.cross_entropy(answer_logits, answer_targets, reduction="none")
        predictions = answer_logits.argmax(dim=-1)
        margin_sum = answer_margins.sum()
        margin_sum.backward()
        total_margin += float(margin_sum.detach().float().cpu().item())
        total_loss += float(answer_losses.sum().detach().float().cpu().item())
        total_correct += int((predictions == answer_targets).detach().sum().cpu().item())
        total_entries += int(answer_margins.numel())
        num_batches += 1
    if total_entries <= 0:
        raise ValueError("Answer-margin gradient records produced no answer entries.")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(float(total_entries))
    gradients, zero_gradient_parameter_names = _parameter_gradients(model=model, require_all=True)
    if zero_gradient_parameter_names:
        raise RuntimeError(
            f"Answer-margin gradient unexpectedly had zero-gradient parameters: {zero_gradient_parameter_names}"
        )
    model.zero_grad(set_to_none=True)
    return {
        "answer_margin": total_margin / float(total_entries),
        "answer_loss": total_loss / float(total_entries),
        "answer_accuracy": total_correct / float(total_entries),
        "num_pairs": len(pairs),
        "num_entries": total_entries,
        "num_batches": num_batches,
        "gradients": gradients,
        "zero_gradient_parameter_names": zero_gradient_parameter_names,
    }


def _actual_summary(
    *,
    actual_rows: list[dict[str, Any]],
    step: int,
    split: str,
    pair_type: str,
    margin_side: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in actual_rows
        if int(row["step"]) == step and _row_matches_group(row=row, split=split, pair_type=pair_type, margin_side=margin_side)
    ]
    if not rows:
        raise RuntimeError(
            f"No answer-margin actual rows for step={step} split={split} pair_type={pair_type} margin_side={margin_side}."
        )
    return {
        "step": step,
        "split": split,
        "pair_type": pair_type,
        "margin_side": margin_side,
        "num_pairs": len({str(row["pair_id"]) for row in rows}),
        "num_entries": len(rows),
        "answer_margin": _mean([float(row["answer_margin"]) for row in rows], label="answer_margin"),
        "answer_loss": _mean([float(row["answer_loss"]) for row in rows], label="answer_loss"),
        "answer_accuracy": _mean(
            [1.0 if bool(row["answer_correct"]) else 0.0 for row in rows],
            label="answer_accuracy",
        ),
    }


def _answer_margin_delta_metric_row(
    *,
    source_step: int,
    target_step: int,
    source_checkpoint: Path,
    target_checkpoint: Path,
    learning_rate: float,
    split: str,
    pair_type: str,
    margin_side: str,
    source_actual: dict[str, Any],
    target_actual: dict[str, Any],
    source_payload: dict[str, Any],
    dot_summary: dict[str, float | int | None],
    min_error_denominator: float,
) -> dict[str, Any]:
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    actual_delta = float(target_actual["answer_margin"]) - float(source_actual["answer_margin"])
    predicted_delta = float(dot_summary["dot"])
    residual = actual_delta - predicted_delta
    denominator = max(abs(actual_delta), min_error_denominator)
    if int(source_actual["num_entries"]) != int(source_payload["num_entries"]):
        raise RuntimeError(
            "Source actual row count does not match source gradient payload: "
            f"{source_actual['num_entries']} vs {source_payload['num_entries']}"
        )
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
        "num_pairs": int(source_actual["num_pairs"]),
        "num_entries": int(source_actual["num_entries"]),
        "source_answer_margin": float(source_actual["answer_margin"]),
        "target_answer_margin": float(target_actual["answer_margin"]),
        "actual_delta": actual_delta,
        "predicted_delta": predicted_delta,
        "residual": residual,
        "absolute_error": abs(residual),
        "relative_error": abs(residual) / denominator,
        "sign_match": _sign_match(actual_delta, predicted_delta),
        "source_answer_loss": float(source_actual["answer_loss"]),
        "target_answer_loss": float(target_actual["answer_loss"]),
        "source_answer_accuracy": float(source_actual["answer_accuracy"]),
        "target_answer_accuracy": float(target_actual["answer_accuracy"]),
        "answer_margin_gradient_l2_norm": float(dot_summary["right_l2_norm"]),
        "parameter_delta_l2_norm": float(dot_summary["left_l2_norm"]),
        "update_answer_margin_gradient_cosine": dot_summary["cosine"],
        "num_parameters": int(dot_summary["num_parameters"]),
        "source_gradient_answer_margin": float(source_payload["answer_margin"]),
        "source_gradient_answer_loss": float(source_payload["answer_loss"]),
        "source_gradient_answer_accuracy": float(source_payload["answer_accuracy"]),
        "source_gradient_num_batches": int(source_payload["num_batches"]),
        "zero_answer_margin_gradient_parameter_count": len(source_payload["zero_gradient_parameter_names"]),
        "zero_answer_margin_gradient_parameter_names": source_payload["zero_gradient_parameter_names"],
    }


def _answer_margin_delta_decomposition_row(
    *,
    metric_row: dict[str, Any],
    group: Any,
    dot_summary: dict[str, float | int | None],
) -> dict[str, Any]:
    predicted_delta = float(dot_summary["dot"])
    answer_margin_gradient_l2_norm = float(dot_summary["right_l2_norm"])
    parameter_delta_l2_norm = float(dot_summary["left_l2_norm"])
    num_selected_parameters = int(dot_summary["num_parameters"])
    return {
        "source_step": int(metric_row["source_step"]),
        "target_step": int(metric_row["target_step"]),
        "step_gap": int(metric_row["step_gap"]),
        "source_checkpoint": metric_row["source_checkpoint"],
        "target_checkpoint": metric_row["target_checkpoint"],
        "learning_rate": float(metric_row["learning_rate"]),
        "split": metric_row["split"],
        "pair_type": metric_row["pair_type"],
        "margin_side": metric_row["margin_side"],
        "num_pairs": int(metric_row["num_pairs"]),
        "num_entries": int(metric_row["num_entries"]),
        "source_answer_margin": float(metric_row["source_answer_margin"]),
        "target_answer_margin": float(metric_row["target_answer_margin"]),
        "actual_delta": float(metric_row["actual_delta"]),
        "global_predicted_delta": float(metric_row["predicted_delta"]),
        "global_residual": float(metric_row["residual"]),
        "global_relative_error": float(metric_row["relative_error"]),
        "group_id": group.group_id,
        "group_kind": group.group_kind,
        "component_type": group.component_type,
        "partition_name": group.partition_name,
        "group_layer": group.layer,
        "group_head": group.head,
        "group_projection": group.projection,
        "group_neuron": group.neuron,
        "selection_count": len(group.selections),
        "num_selected_parameters": num_selected_parameters,
        "predicted_delta_contribution": predicted_delta,
        "parameter_delta_l2_norm": parameter_delta_l2_norm,
        "answer_margin_gradient_l2_norm": answer_margin_gradient_l2_norm,
        "update_answer_margin_gradient_cosine": dot_summary["cosine"],
        "contribution_per_parameter": predicted_delta / float(num_selected_parameters),
        "notes": list(group.notes),
    }


def _compute_answer_margin_delta_interval(
    *,
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    learning_rate: float,
    margin_sides: list[str],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    groups: list[Any],
    min_error_denominator: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not groups:
        raise ValueError("Answer-margin delta decomposition requires at least one decomposition group.")
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
        label=f"answer-margin delta {source_step}->{target_step}",
    )

    actual_rows: list[dict[str, Any]] = []
    for margin_side in margin_sides:
        actual_rows.extend(
            _compute_answer_margin_rows_for_loaded_model(
                model=source_model,
                checkpoint_path=source_checkpoint_path,
                checkpoint_step=source_step,
                pairs=pairs,
                vocab=vocab,
                margin_side=margin_side,
                batch_size=batch_size,
                pad_token_id=pad_token_id,
                device=device,
            )
        )
        actual_rows.extend(
            _compute_answer_margin_rows_for_loaded_model(
                model=target_model,
                checkpoint_path=target_checkpoint_path,
                checkpoint_step=target_step,
                pairs=pairs,
                vocab=vocab,
                margin_side=margin_side,
                batch_size=batch_size,
                pad_token_id=pad_token_id,
                device=device,
            )
        )

    pair_groups = _route_gradient_groups(pairs)
    metric_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    for (split, pair_type), group_pairs in sorted(pair_groups.items()):
        for margin_side in margin_sides:
            source_payload = _compute_answer_margin_gradient_for_pairs(
                model=source_model,
                pairs=group_pairs,
                vocab=vocab,
                margin_side=margin_side,
                batch_size=batch_size,
                pad_token_id=pad_token_id,
                device=device,
            )
            answer_margin_gradients = source_payload["gradients"]
            if not isinstance(answer_margin_gradients, dict):
                raise TypeError("Answer-margin gradient payload must contain a gradients dictionary.")
            dot_summary = _gradient_dot_summary(
                left_gradients=delta_parameters,
                right_gradients=answer_margin_gradients,
                label=f"answer-margin delta {source_step}->{target_step} {split}/{pair_type}/{margin_side}",
            )
            metric_row = _answer_margin_delta_metric_row(
                source_step=source_step,
                target_step=target_step,
                source_checkpoint=source_checkpoint_path,
                target_checkpoint=target_checkpoint_path,
                learning_rate=learning_rate,
                split=split,
                pair_type=pair_type,
                margin_side=margin_side,
                source_actual=_actual_summary(
                    actual_rows=actual_rows,
                    step=source_step,
                    split=split,
                    pair_type=pair_type,
                    margin_side=margin_side,
                ),
                target_actual=_actual_summary(
                    actual_rows=actual_rows,
                    step=target_step,
                    split=split,
                    pair_type=pair_type,
                    margin_side=margin_side,
                ),
                source_payload=source_payload,
                dot_summary=dot_summary,
                min_error_denominator=min_error_denominator,
            )
            metric_rows.append(metric_row)
            for group in groups:
                group_dot_summary = _gradient_dot_summary_for_group(
                    left_gradients=delta_parameters,
                    right_gradients=answer_margin_gradients,
                    group=group,
                    label=f"answer-margin delta {source_step}->{target_step} {split}/{pair_type}/{margin_side}/{group.group_id}",
                )
                decomposition_rows.append(
                    _answer_margin_delta_decomposition_row(
                        metric_row=metric_row,
                        group=group,
                        dot_summary=group_dot_summary,
                    )
                )
    return metric_rows, decomposition_rows, actual_rows


def _summarize_answer_margin_delta_decomposition(
    *,
    metric_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    top_k_groups: int,
    min_error_denominator: float,
) -> dict[str, Any]:
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    if not metric_rows:
        raise ValueError("Cannot summarize answer-margin delta decomposition without metric rows.")
    if not decomposition_rows:
        raise ValueError("Cannot summarize answer-margin delta decomposition without decomposition rows.")
    all_rows = [
        row
        for row in metric_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    if not all_rows:
        raise RuntimeError("Answer-margin delta decomposition has no __all__/__all__ metric rows.")
    final_target_step = max(int(row["target_step"]) for row in metric_rows)
    final_rows = [row for row in all_rows if int(row["target_step"]) == final_target_step]
    cumulative_rows: list[dict[str, Any]] = []
    for margin_side in sorted({str(row["margin_side"]) for row in all_rows}):
        side_rows = sorted(
            [row for row in all_rows if str(row["margin_side"]) == margin_side],
            key=lambda row: (int(row["source_step"]), int(row["target_step"])),
        )
        actual_sum = sum(float(row["actual_delta"]) for row in side_rows)
        predicted_sum = sum(float(row["predicted_delta"]) for row in side_rows)
        residual_sum = actual_sum - predicted_sum
        cumulative_rows.append(
            {
                "margin_side": margin_side,
                "num_intervals": len(side_rows),
                "source_step_first": int(side_rows[0]["source_step"]),
                "target_step_last": int(side_rows[-1]["target_step"]),
                "actual_delta_sum": actual_sum,
                "predicted_delta_sum": predicted_sum,
                "residual_sum": residual_sum,
                "absolute_error_sum": abs(residual_sum),
                "relative_error_sum": abs(residual_sum) / max(abs(actual_sum), min_error_denominator),
                "sign_match_fraction": _mean(
                    [1.0 if bool(row["sign_match"]) else 0.0 for row in side_rows],
                    label=f"{margin_side} sign_match_fraction",
                ),
            }
        )
    final_decomposition_rows = [
        row
        for row in decomposition_rows
        if int(row["target_step"]) == final_target_step
        and str(row["split"]) == "__all__"
        and str(row["pair_type"]) == "__all__"
        and str(row["group_kind"]) not in {"global_all", "parameter_tensor"}
    ]
    return {
        "num_intervals": len({(int(row["source_step"]), int(row["target_step"])) for row in metric_rows}),
        "intervals": sorted({f"{int(row['source_step'])}->{int(row['target_step'])}" for row in all_rows}),
        "target_steps": sorted({int(row["target_step"]) for row in metric_rows}),
        "final_target_step": final_target_step,
        "final_metric_rows": sorted(final_rows, key=lambda row: str(row["margin_side"])),
        "cumulative_rows": cumulative_rows,
        "all_all_sign_match_fraction": _mean(
            [1.0 if bool(row["sign_match"]) else 0.0 for row in all_rows],
            label="answer-margin delta sign_match_fraction",
        ),
        "all_all_mean_absolute_error": _mean(
            [float(row["absolute_error"]) for row in all_rows],
            label="answer-margin delta absolute_error",
        ),
        "all_all_mean_relative_error": _mean(
            [float(row["relative_error"]) for row in all_rows],
            label="answer-margin delta relative_error",
        ),
        "all_all_worst_relative_error": max(all_rows, key=lambda row: float(row["relative_error"])),
        "final_top_positive_contributions": sorted(
            final_decomposition_rows,
            key=lambda row: float(row["predicted_delta_contribution"]),
            reverse=True,
        )[:top_k_groups],
        "final_top_negative_contributions": sorted(
            final_decomposition_rows,
            key=lambda row: float(row["predicted_delta_contribution"]),
        )[:top_k_groups],
        "final_top_abs_contributions": sorted(
            final_decomposition_rows,
            key=lambda row: abs(float(row["predicted_delta_contribution"])),
            reverse=True,
        )[:top_k_groups],
    }


def _plot_answer_margin_actual_vs_predicted(
    *,
    metric_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    rows = [
        row
        for row in metric_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 7))
    actual = [float(row["actual_delta"]) for row in rows]
    predicted = [float(row["predicted_delta"]) for row in rows]
    ax.scatter(predicted, actual, color="#376f8f")
    for row in rows:
        ax.annotate(
            f"{row['source_step']}->{row['target_step']} {row['margin_side']}",
            (float(row["predicted_delta"]), float(row["actual_delta"])),
            fontsize=7,
            alpha=0.7,
        )
    min_value = min(actual + predicted)
    max_value = max(actual + predicted)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    ax.plot([min_value, max_value], [min_value, max_value], color="#777777", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.axvline(0.0, color="#999999", linewidth=0.8)
    ax.set_title("Answer-margin delta: actual vs first-order prediction")
    ax.set_xlabel("grad(answer_margin) . Delta theta")
    ax.set_ylabel("answer_margin(theta_target) - answer_margin(theta_source)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_answer_margin_interval_trajectory(
    *,
    metric_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    rows = sorted(
        [
            row
            for row in metric_rows
            if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
        ],
        key=lambda row: (str(row["margin_side"]), int(row["target_step"])),
    )
    if not rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 5))
    for margin_side in sorted({str(row["margin_side"]) for row in rows}):
        side_rows = [row for row in rows if str(row["margin_side"]) == margin_side]
        target_steps = [int(row["target_step"]) for row in side_rows]
        actual_cumulative: list[float] = []
        predicted_cumulative: list[float] = []
        running_actual = 0.0
        running_predicted = 0.0
        for row in side_rows:
            running_actual += float(row["actual_delta"])
            running_predicted += float(row["predicted_delta"])
            actual_cumulative.append(running_actual)
            predicted_cumulative.append(running_predicted)
        ax.plot(target_steps, actual_cumulative, marker="o", label=f"{margin_side} actual")
        ax.plot(target_steps, predicted_cumulative, marker="x", linestyle="--", label=f"{margin_side} predicted")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_title("Cumulative answer-margin delta over intervals")
    ax.set_xlabel("target checkpoint step")
    ax.set_ylabel("cumulative delta")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_answer_margin_top_contributions(
    *,
    decomposition_rows: list[dict[str, Any]],
    top_k_groups: int,
    output_path: Path,
) -> Path | None:
    if not decomposition_rows:
        return None
    final_target_step = max(int(row["target_step"]) for row in decomposition_rows)
    rows = [
        row
        for row in decomposition_rows
        if int(row["target_step"]) == final_target_step
        and str(row["split"]) == "__all__"
        and str(row["pair_type"]) == "__all__"
        and str(row["group_kind"]) not in {"global_all", "parameter_tensor"}
    ]
    if not rows:
        return None
    top_rows = sorted(rows, key=lambda row: abs(float(row["predicted_delta_contribution"])), reverse=True)[:top_k_groups]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(13, max(5, 0.38 * len(top_rows))))
    y_positions = list(range(len(top_rows)))
    values = [float(row["predicted_delta_contribution"]) for row in top_rows]
    labels = [f"{row['margin_side']} {row['group_id']}" for row in top_rows]
    colors = ["#376f8f" if value >= 0.0 else "#8f374a" for value in values]
    ax.barh(y_positions, values, color=colors)
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"Top answer-margin update contributions ending at step {final_target_step}")
    ax.set_xlabel("grad(answer_margin) . Delta theta contribution")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_answer_margin_delta_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    lines = [
        "# Answer-Margin Delta Decomposition",
        "",
        "## Calculation",
        "",
        "This report tests the direct local explanation of answer-margin movement.",
        "",
        "```text",
        "m(theta) = mean_x [ logit(correct_value | x) - max_wrong_value logit(wrong_value | x) ]",
        "actual_delta = m(theta_target) - m(theta_source)",
        "predicted_delta = grad_theta m(theta_source) . (theta_target - theta_source)",
        "residual = actual_delta - predicted_delta",
        "group_contribution = grad_group m(theta_source) . Delta theta_group",
        "```",
        "",
        "Unlike route-to-margin closure, this does not fit route scores to margin. It differentiates the answer margin itself.",
        "",
        "## Run",
        "",
        f"- margin sides: `{report['margin_sides']}`",
        f"- intervals: `{summary['intervals']}`",
        f"- pair types: `{report['pair_types']}`",
        f"- decomposition modes: `{report['decomposition']['decomposition_modes']}`",
        "",
        "## Cumulative Closure",
        "",
        "| side | intervals | actual delta sum | predicted delta sum | residual sum | relative error | sign match |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cumulative_rows"]:
        lines.append(
            "| `{side}` | {intervals} | {actual:.6g} | {predicted:.6g} | {residual:.6g} | {error:.6g} | {sign:.3f} |".format(
                side=row["margin_side"],
                intervals=int(row["num_intervals"]),
                actual=float(row["actual_delta_sum"]),
                predicted=float(row["predicted_delta_sum"]),
                residual=float(row["residual_sum"]),
                error=float(row["relative_error_sum"]),
                sign=float(row["sign_match_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "## Final Interval Metrics",
            "",
            "| side | actual delta | predicted delta | residual | relative error | sign match | source margin | target margin |",
            "|---|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in summary["final_metric_rows"]:
        lines.append(
            "| `{side}` | {actual:.6g} | {predicted:.6g} | {residual:.6g} | {error:.6g} | `{sign}` | {source:.6g} | {target:.6g} |".format(
                side=row["margin_side"],
                actual=float(row["actual_delta"]),
                predicted=float(row["predicted_delta"]),
                residual=float(row["residual"]),
                error=float(row["relative_error"]),
                sign=bool(row["sign_match"]),
                source=float(row["source_answer_margin"]),
                target=float(row["target_answer_margin"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Positive Contributions",
            "",
            "| group | side | kind | contribution | cosine |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in summary["final_top_positive_contributions"]:
        cosine = row["update_answer_margin_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| `{group}` | `{side}` | `{kind}` | {contribution:.6g} | {cosine} |".format(
                group=row["group_id"],
                side=row["margin_side"],
                kind=row["group_kind"],
                contribution=float(row["predicted_delta_contribution"]),
                cosine=cosine_text,
            )
        )
    lines.extend(
        [
            "",
            "## Top Negative Contributions",
            "",
            "| group | side | kind | contribution | cosine |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in summary["final_top_negative_contributions"]:
        cosine = row["update_answer_margin_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| `{group}` | `{side}` | `{kind}` | {contribution:.6g} | {cosine} |".format(
                group=row["group_id"],
                side=row["margin_side"],
                kind=row["group_kind"],
                contribution=float(row["predicted_delta_contribution"]),
                cosine=cosine_text,
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
            f"- margin rows: `{report['margin_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
        ]
    )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for label, plot_path in plot_paths.items():
            lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_answer_margin_delta_decomposition(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    pair_types: list[str],
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    margin_sides: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    decomposition_modes: list[str] | None = None,
    top_k_groups: int = 24,
    min_error_denominator: float = 1.0e-9,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if not pair_types:
        raise ValueError("answer-margin-delta-decomposition requires at least one pair type.")
    if max_pairs_per_type <= 0:
        raise ValueError("max_pairs_per_type must be positive.")
    if min_pairs_per_type <= 0:
        raise ValueError("min_pairs_per_type must be positive.")
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    resolved_margin_sides = _resolve_margin_sides(margin_sides)
    resolved_decomposition_modes = _resolve_route_gradient_decomposition_modes(decomposition_modes)
    pair_types = sorted(set(pair_types), key=pair_types.index)

    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("answer-margin-delta-decomposition requires at least two checkpoints.")
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
        raise RuntimeError("Answer-margin delta decomposition constructed no pairs.")

    source_model = build_model(spec.model, len(vocab.tokens), device)
    target_model = build_model(spec.model, len(vocab.tokens), device)
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

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows_path = output_dir / "answer_margin_delta_decomposition_rows.jsonl"
    decomposition_rows_path = output_dir / "answer_margin_delta_decomposition_components.jsonl"
    group_rows_path = output_dir / "answer_margin_delta_decomposition_groups.jsonl"
    margin_rows_path = output_dir / "answer_margin_delta_decomposition_margin_rows.jsonl"
    pair_rows_path = output_dir / "answer_margin_delta_decomposition_pairs.jsonl"
    progress_path = output_dir / "answer_margin_delta_decomposition_progress.json"
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])
    write_jsonl(group_rows_path, group_rows)

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[answer-margin-delta-decomposition] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(pairs)} "
        f"pair_types={pair_types} margin_sides={resolved_margin_sides} device={device_name} groups={len(groups)}",
        flush=True,
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_decomposition_rows: list[dict[str, Any]] = []
    all_margin_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        learning_rate = _compute_learning_rate(spec.optimization, source_step)
        print(
            "[answer-margin-delta-decomposition] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )
        metric_rows, decomposition_rows, margin_rows = _compute_answer_margin_delta_interval(
            source_model=source_model,
            target_model=target_model,
            source_checkpoint_path=source_checkpoint_path,
            target_checkpoint_path=target_checkpoint_path,
            pairs=pairs,
            vocab=vocab,
            learning_rate=learning_rate,
            margin_sides=resolved_margin_sides,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            groups=groups,
            min_error_denominator=min_error_denominator,
        )
        for row in metric_rows:
            append_jsonl(metric_rows_path, row)
        for row in decomposition_rows:
            append_jsonl(decomposition_rows_path, row)
        for row in margin_rows:
            append_jsonl(margin_rows_path, row)
        all_metric_rows.extend(metric_rows)
        all_decomposition_rows.extend(decomposition_rows)
        all_margin_rows.extend(margin_rows)
        all_row = next(
            row
            for row in metric_rows
            if str(row["split"]) == "__all__"
            and str(row["pair_type"]) == "__all__"
            and str(row["margin_side"]) == resolved_margin_sides[0]
        )
        print(
            "[answer-margin-delta-decomposition] finished "
            f"{source_step}->{target_step} side={all_row['margin_side']} "
            f"actual_delta={float(all_row['actual_delta']):.6g} "
            f"predicted_delta={float(all_row['predicted_delta']):.6g} "
            f"relative_error={float(all_row['relative_error']):.6g} "
            f"sign_match={all_row['sign_match']}",
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
                "margin_rows_path": str(margin_rows_path),
            },
        )

    summary = _summarize_answer_margin_delta_decomposition(
        metric_rows=all_metric_rows,
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
        min_error_denominator=min_error_denominator,
    )
    plot_paths: dict[str, Path] = {}
    actual_vs_predicted_plot = _plot_answer_margin_actual_vs_predicted(
        metric_rows=all_metric_rows,
        output_path=output_dir / "answer_margin_delta_actual_vs_predicted.svg",
    )
    if actual_vs_predicted_plot is not None:
        plot_paths["actual_vs_predicted"] = actual_vs_predicted_plot
    interval_trajectory_plot = _plot_answer_margin_interval_trajectory(
        metric_rows=all_metric_rows,
        output_path=output_dir / "answer_margin_delta_interval_trajectory.svg",
    )
    if interval_trajectory_plot is not None:
        plot_paths["interval_trajectory"] = interval_trajectory_plot
    top_contributions_plot = _plot_answer_margin_top_contributions(
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
        output_path=output_dir / "answer_margin_delta_top_contributions.svg",
    )
    if top_contributions_plot is not None:
        plot_paths["top_contributions"] = top_contributions_plot

    report_path = output_dir / "answer_margin_delta_decomposition_report.json"
    markdown_path = output_dir / "answer_margin_delta_decomposition_report.md"
    report = {
        "schema_version": ANSWER_MARGIN_DELTA_DECOMPOSITION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "margin_sides": resolved_margin_sides,
        "pair_types": pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "decomposition": decomposition_summary,
        "top_k_groups": top_k_groups,
        "min_error_denominator": min_error_denominator,
        "calculation": {
            "answer_margin": "logit(correct_value) - max_wrong_value logit(wrong_value)",
            "actual_delta": "mean_answer_margin(theta_target) - mean_answer_margin(theta_source)",
            "predicted_delta": "grad_theta mean_answer_margin(theta_source) . (theta_target - theta_source)",
            "residual": "actual_delta - predicted_delta",
            "group_contribution": "grad_group mean_answer_margin(theta_source) . Delta theta_group",
        },
        "pair_construction": pair_construction,
        "metric_rows_path": str(metric_rows_path),
        "decomposition_rows_path": str(decomposition_rows_path),
        "group_rows_path": str(group_rows_path),
        "margin_rows_path": str(margin_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_answer_margin_delta_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_intervals": len(intervals),
            "total_intervals": len(intervals),
            "last_target_step": int(summary["final_target_step"]),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "metric_rows_path": str(metric_rows_path),
            "decomposition_rows_path": str(decomposition_rows_path),
            "group_rows_path": str(group_rows_path),
            "margin_rows_path": str(margin_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[answer-margin-delta-decomposition] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return (
        report_path,
        markdown_path,
        metric_rows_path,
        decomposition_rows_path,
        group_rows_path,
        margin_rows_path,
        pair_rows_path,
        plot_paths,
    )
