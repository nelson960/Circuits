from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from circuit.config import TrainSpec
from circuit.io import append_jsonl, write_json
from circuit.runtime import compute_lm_loss, save_checkpoint, set_seed
from circuit.train import build_run_context, _resume_training_state, _set_learning_rate, _to_jsonable, make_data_loader
from circuit.vocab import Vocabulary


_HISTORICAL_REPLAY_BLOCKER = (
    "Existing checkpoints save model and optimizer state, but they do not save the DataLoader sampler order, "
    "the active iterator offset, or RNG state at the saved step. This trace is therefore an instrumented "
    "continuation from the checkpoint unless the batch stream was recorded by this tool."
)


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device)
        if isinstance(value, torch.Tensor)
        else {inner_key: inner_value.to(device) for inner_key, inner_value in value.items()}
        if isinstance(value, dict)
        else value
        for key, value in batch.items()
    }


def _parameter_snapshots(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _optimizer_state_summary(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    exp_avg_sq_sum = 0.0
    exp_avg_sq_count = 0
    exp_avg_sum = 0.0
    exp_avg_count = 0
    max_state_step = 0.0
    state_entries = 0
    for state in optimizer.state.values():
        if not state:
            continue
        state_entries += 1
        step_value = state.get("step")
        if isinstance(step_value, torch.Tensor):
            max_state_step = max(max_state_step, float(step_value.detach().max().item()))
        elif isinstance(step_value, int | float):
            max_state_step = max(max_state_step, float(step_value))
        exp_avg = state.get("exp_avg")
        if isinstance(exp_avg, torch.Tensor):
            exp_avg_sum += float(torch.sum(exp_avg.detach().float() ** 2).item())
            exp_avg_count += exp_avg.numel()
        exp_avg_sq = state.get("exp_avg_sq")
        if isinstance(exp_avg_sq, torch.Tensor):
            exp_avg_sq_sum += float(torch.sum(exp_avg_sq.detach().float() ** 2).item())
            exp_avg_sq_count += exp_avg_sq.numel()
    return {
        "optimizer_state_entries": state_entries,
        "optimizer_state_step_max": max_state_step,
        "adam_exp_avg_l2": exp_avg_sum**0.5,
        "adam_exp_avg_sq_l2": exp_avg_sq_sum**0.5,
        "adam_exp_avg_count": exp_avg_count,
        "adam_exp_avg_sq_count": exp_avg_sq_count,
    }


def _batch_trace_row(*, step: int, batch: dict[str, Any]) -> dict[str, Any]:
    records = batch["records"]
    sample_ids = [str(record["sample_id"]) for record in records]
    splits = sorted({str(record["split"]) for record in records})
    query_events = [event for record in records for event in record["query_events"]]
    query_keys = [str(event["key"]) for event in query_events]
    answer_values = [str(event["answer_value"]) for event in query_events]
    return {
        "step": step,
        "batch_size": len(records),
        "query_event_count": len(query_events),
        "splits": splits,
        "sample_ids": sample_ids,
        "query_keys": query_keys,
        "answer_values": answer_values,
    }


def _parameter_update_rows(
    *,
    step: int,
    before_parameters: dict[str, torch.Tensor],
    model: torch.nn.Module,
    top_k_parameters: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_delta_sq = 0.0
    total_parameter_sq = 0.0
    total_gradient_sq = 0.0
    update_dot_negative_gradient = 0.0
    max_abs_delta = 0.0
    rows: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name not in before_parameters:
            raise KeyError(f"Missing parameter snapshot for {name}.")
        before = before_parameters[name]
        after = parameter.detach()
        delta = after - before
        delta_sq = float(torch.sum(delta.float() ** 2).item())
        parameter_sq = float(torch.sum(before.float() ** 2).item())
        gradient = parameter.grad.detach() if parameter.grad is not None else None
        gradient_sq = float(torch.sum(gradient.float() ** 2).item()) if gradient is not None else 0.0
        dot = float(torch.sum(delta.float() * (-gradient.float())).item()) if gradient is not None else 0.0
        delta_max_abs = float(torch.max(torch.abs(delta)).item()) if delta.numel() else 0.0
        total_delta_sq += delta_sq
        total_parameter_sq += parameter_sq
        total_gradient_sq += gradient_sq
        update_dot_negative_gradient += dot
        max_abs_delta = max(max_abs_delta, delta_max_abs)
        rows.append(
            {
                "step": step,
                "parameter": name,
                "numel": parameter.numel(),
                "delta_l2": delta_sq**0.5,
                "delta_max_abs": delta_max_abs,
                "parameter_l2_before": parameter_sq**0.5,
                "clipped_gradient_l2": gradient_sq**0.5,
                "update_dot_negative_clipped_gradient": dot,
            }
        )
    rows.sort(key=lambda row: abs(float(row["delta_l2"])), reverse=True)
    summary = {
        "parameter_delta_l2": total_delta_sq**0.5,
        "parameter_delta_max_abs": max_abs_delta,
        "parameter_l2_before": total_parameter_sq**0.5,
        "clipped_gradient_l2": total_gradient_sq**0.5,
        "update_dot_negative_clipped_gradient": update_dot_negative_gradient,
    }
    return summary, rows[:top_k_parameters]


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    start_source_label = "resume checkpoint" if report["trace_start_mode"] == "resume_checkpoint" else "start source"
    lines = [
        "# Optimizer Update Trace",
        "",
        "This artifact records real optimizer steps from an instrumented training run.",
        "",
        "## Replay Status",
        "",
        f"- status: `{report['historical_replay_status']}`",
        f"- blocker: {report['historical_replay_blocker']}",
        "",
        "## Trace",
        "",
        f"- start mode: `{report['trace_start_mode']}`",
        f"- {start_source_label}: `{report['resume_checkpoint'] if report['trace_start_mode'] == 'resume_checkpoint' else 'initialization'}`",
        f"- checkpoint directory: `{report['checkpoint_dir']}`",
        f"- traced steps: `{report['trace_start_step']} -> {report['trace_end_step']}`",
        f"- checkpoint interval: `{report['checkpoint_every_steps']}`",
        f"- checkpoint start step: `{report['checkpoint_start_step']}`",
        f"- step rows: `{report['step_rows_path']}`",
        f"- batch rows: `{report['batch_rows_path']}`",
        f"- parameter update rows: `{report['parameter_update_rows_path']}`",
        "",
        "## Summary",
        "",
        f"- steps traced: `{report['summary']['num_traced_steps']}`",
        f"- mean loss: `{report['summary']['loss_mean']:.6g}`",
        f"- mean token accuracy: `{report['summary']['token_accuracy_mean']:.6g}`",
        f"- mean parameter update L2: `{report['summary']['parameter_delta_l2_mean']:.6g}`",
        f"- saved checkpoints: `{report['summary']['num_saved_checkpoints']}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_optimizer_update_trace(
    *,
    config_path: Path,
    resume_checkpoint: Path | None,
    from_initialization: bool = False,
    output_dir: Path,
    end_step: int,
    device_name: str | None,
    train_split: str,
    checkpoint_every_steps: int,
    checkpoint_start_step: int | None = None,
    progress_every_steps: int,
    top_k_parameters: int,
    overwrite: bool = False,
    require_historical_replay: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    if from_initialization == (resume_checkpoint is not None):
        raise ValueError("Expected exactly one of resume_checkpoint or from_initialization=True.")
    if require_historical_replay and not from_initialization:
        raise RuntimeError(_HISTORICAL_REPLAY_BLOCKER)
    if checkpoint_every_steps <= 0:
        raise ValueError("checkpoint_every_steps must be positive.")
    if progress_every_steps < 0:
        raise ValueError("progress_every_steps must be non-negative.")
    if top_k_parameters < 0:
        raise ValueError("top_k_parameters must be non-negative.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = TrainSpec.from_path(config_path)
    if device_name is not None:
        spec = replace(spec, device=device_name)
    set_seed(spec.seed)
    if from_initialization:
        context = build_run_context(spec)
        start_step = 0
        latest_eval: dict[str, Any] = {}
        trace_start_mode = "from_initialization"
        replay_status = "instrumented_from_initialization_exact_for_this_trace"
        replay_blocker = (
            "This trace starts at initialization and records its own batch stream and optimizer steps. "
            "It is exact for this traced run; it is not a replay of an older run whose batch/RNG state was not saved."
        )
    else:
        if resume_checkpoint is None:
            raise ValueError("resume_checkpoint is required unless from_initialization=True.")
        context = _resume_training_state(spec=spec, resume_checkpoint=resume_checkpoint)
        start_step = int(context["start_step"])
        latest_eval = context.get("latest_eval", {})
        trace_start_mode = "resume_checkpoint"
        replay_status = "instrumented_continuation_not_historical_replay"
        replay_blocker = _HISTORICAL_REPLAY_BLOCKER
    metadata = context["metadata"]
    vocab: Vocabulary = context["vocab"]
    device: torch.device = context["device"]
    model: torch.nn.Module = context["model"]
    optimizer: torch.optim.Optimizer = context["optimizer"]
    if end_step <= start_step:
        raise ValueError(f"end_step must be greater than trace start step {start_step}, got {end_step}.")
    if end_step > spec.num_steps:
        raise ValueError(f"end_step={end_step} exceeds config num_steps={spec.num_steps}.")
    if checkpoint_start_step is None:
        resolved_checkpoint_start_step = start_step
    else:
        resolved_checkpoint_start_step = int(checkpoint_start_step)
    if resolved_checkpoint_start_step < start_step:
        raise ValueError(
            f"checkpoint_start_step={resolved_checkpoint_start_step} is before trace start step {start_step}."
        )
    if resolved_checkpoint_start_step > end_step:
        raise ValueError(
            f"checkpoint_start_step={resolved_checkpoint_start_step} is after trace end step {end_step}."
        )

    train_loader = make_data_loader(
        benchmark_dir=spec.benchmark_dir,
        split_name=train_split,
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=spec.num_workers,
        pad_token_id=vocab.pad_token_id,
    )
    train_iterator = iter(train_loader)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    step_rows_path = output_dir / "optimizer_update_trace_steps.jsonl"
    batch_rows_path = output_dir / "optimizer_update_trace_batches.jsonl"
    parameter_update_rows_path = output_dir / "optimizer_update_trace_parameter_updates.jsonl"
    report_path = output_dir / "optimizer_update_trace_report.json"
    markdown_path = output_dir / "optimizer_update_trace_report.md"
    trace_config_path = output_dir / "optimizer_update_trace_config.json"

    trace_config = {
        "config_path": str(config_path),
        "trace_start_mode": trace_start_mode,
        "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
        "from_initialization": from_initialization,
        "output_dir": str(output_dir),
        "device": spec.device,
        "train_split": train_split,
        "trace_start_step": start_step,
        "trace_end_step": end_step,
        "checkpoint_every_steps": checkpoint_every_steps,
        "checkpoint_start_step": resolved_checkpoint_start_step,
        "progress_every_steps": progress_every_steps,
        "top_k_parameters": top_k_parameters,
        "historical_replay_status": replay_status,
        "historical_replay_blocker": replay_blocker,
    }
    write_json(trace_config_path, trace_config)

    saved_checkpoint_paths: list[str] = []

    def save_trace_checkpoint(step: int) -> None:
        checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            step=step,
            metrics=latest_eval,
            config={
                "train_spec": _to_jsonable(spec),
                "benchmark_metadata": metadata,
                "model_parameter_count": model.count_parameters(),
                "optimizer_update_trace": trace_config,
            },
        )
        saved_checkpoint_paths.append(str(checkpoint_path))

    if start_step >= resolved_checkpoint_start_step:
        save_trace_checkpoint(start_step)
    print(
        "[optimizer-update-trace] "
        f"start={start_step} end={end_step} split={train_split} device={device} "
        f"checkpoint_every={checkpoint_every_steps}"
    )

    step_rows: list[dict[str, Any]] = []
    for step in range(start_step + 1, end_step + 1):
        model.train()
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        batch_row = _batch_trace_row(step=step, batch=batch)
        append_jsonl(batch_rows_path, batch_row)
        batch = _move_batch_to_device(batch, device)
        learning_rate = _set_learning_rate(optimizer, spec.optimization, step)
        optimizer.zero_grad(set_to_none=True)
        before_parameters = _parameter_snapshots(model)
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss, token_accuracy = compute_lm_loss(
            logits=outputs.logits,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pad_token_id=vocab.pad_token_id,
        )
        loss.backward()
        pre_clip_grad_norm = float(clip_grad_norm_(model.parameters(), spec.optimization.grad_clip_norm).item())
        optimizer.step()

        update_summary, parameter_rows = _parameter_update_rows(
            step=step,
            before_parameters=before_parameters,
            model=model,
            top_k_parameters=top_k_parameters,
        )
        for parameter_row in parameter_rows:
            append_jsonl(parameter_update_rows_path, parameter_row)
        step_row = {
            "step": step,
            "source_step": step - 1,
            "target_step": step,
            "loss": float(loss.item()),
            "token_accuracy": float(token_accuracy.item()),
            "learning_rate": learning_rate,
            "pre_clip_grad_norm": pre_clip_grad_norm,
            "grad_clip_norm": spec.optimization.grad_clip_norm,
            "batch_size": batch_row["batch_size"],
            "query_event_count": batch_row["query_event_count"],
            **update_summary,
            **_optimizer_state_summary(optimizer),
        }
        append_jsonl(step_rows_path, step_row)
        step_rows.append(step_row)

        if step >= resolved_checkpoint_start_step and (step % checkpoint_every_steps == 0 or step == end_step):
            save_trace_checkpoint(step)
        if progress_every_steps and (step == end_step or step % progress_every_steps == 0):
            print(
                "[optimizer-update-trace] "
                f"step={step} loss={float(loss.item()):.6g} tok_acc={float(token_accuracy.item()):.6g} "
                f"grad_norm={pre_clip_grad_norm:.6g} update_l2={update_summary['parameter_delta_l2']:.6g}"
            )

    num_steps = len(step_rows)
    loss_mean = sum(float(row["loss"]) for row in step_rows) / num_steps
    token_accuracy_mean = sum(float(row["token_accuracy"]) for row in step_rows) / num_steps
    update_l2_mean = sum(float(row["parameter_delta_l2"]) for row in step_rows) / num_steps
    report = {
        "trace_config_path": str(trace_config_path),
        "config_path": str(config_path),
        "trace_start_mode": trace_start_mode,
        "resume_checkpoint": None if resume_checkpoint is None else str(resume_checkpoint),
        "from_initialization": from_initialization,
        "checkpoint_dir": str(checkpoint_dir),
        "step_rows_path": str(step_rows_path),
        "batch_rows_path": str(batch_rows_path),
        "parameter_update_rows_path": str(parameter_update_rows_path),
        "historical_replay_status": replay_status,
        "historical_replay_blocker": replay_blocker,
        "device": spec.device,
        "train_split": train_split,
        "trace_start_step": start_step,
        "trace_end_step": end_step,
        "checkpoint_every_steps": checkpoint_every_steps,
        "checkpoint_start_step": resolved_checkpoint_start_step,
        "saved_checkpoints": saved_checkpoint_paths,
        "summary": {
            "num_traced_steps": num_steps,
            "loss_mean": loss_mean,
            "token_accuracy_mean": token_accuracy_mean,
            "parameter_delta_l2_mean": update_l2_mean,
            "num_saved_checkpoints": len(saved_checkpoint_paths),
            "first_step_loss": float(step_rows[0]["loss"]),
            "last_step_loss": float(step_rows[-1]["loss"]),
        },
    }
    write_json(report_path, report)
    _write_markdown_report(markdown_path, report)
    print(
        "[optimizer-update-trace] "
        f"complete report={report_path} rows={step_rows_path} checkpoints={checkpoint_dir}"
    )
    return report_path, markdown_path, step_rows_path, batch_rows_path, parameter_update_rows_path, checkpoint_dir
