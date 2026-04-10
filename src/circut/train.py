from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import partial
import math
from pathlib import Path
import shutil
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from circut.config import OptimizationSpec, TrainSpec
from circut.data.symbolic_kv_stream import (
    SymbolicKVDataset,
    collate_symbolic_kv,
    read_symbolic_kv_stream_metadata,
)
from circut.eval import evaluate_split
from circut.io import append_jsonl, read_json, write_json
from circut.runtime import build_model, compute_lm_loss, load_checkpoint, load_model_state, require_device, save_checkpoint, set_seed
from circut.vocab import Vocabulary


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def make_data_loader(
    *,
    benchmark_dir: Path,
    split_name: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pad_token_id: int,
) -> DataLoader[Any]:
    dataset = SymbolicKVDataset(benchmark_dir, split_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_symbolic_kv, pad_token_id=pad_token_id),
    )


def build_run_context(spec: TrainSpec) -> dict[str, Any]:
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(spec.device)
    model = build_model(spec.model, len(vocab.tokens), device)
    optimizer = AdamW(
        model.parameters(),
        lr=spec.optimization.learning_rate,
        betas=(spec.optimization.beta1, spec.optimization.beta2),
        weight_decay=spec.optimization.weight_decay,
    )
    return {
        "metadata": metadata,
        "vocab": vocab,
        "device": device,
        "model": model,
        "optimizer": optimizer,
    }


def _compute_learning_rate(optimization: OptimizationSpec, step: int) -> float:
    base_lr = optimization.learning_rate
    if optimization.warmup_steps > 0 and step <= optimization.warmup_steps:
        return base_lr * (step / optimization.warmup_steps)

    schedule = optimization.schedule
    if schedule.kind == "constant":
        return base_lr
    if schedule.kind == "cosine_decay":
        if schedule.decay_start_step is None or schedule.decay_end_step is None or schedule.min_learning_rate is None:
            raise RuntimeError("cosine_decay schedule is missing required parameters.")
        if step <= schedule.decay_start_step:
            return base_lr
        if step >= schedule.decay_end_step:
            return schedule.min_learning_rate
        progress = (step - schedule.decay_start_step) / (schedule.decay_end_step - schedule.decay_start_step)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return schedule.min_learning_rate + (base_lr - schedule.min_learning_rate) * cosine_factor
    raise RuntimeError(f"Unhandled learning-rate schedule kind: {schedule.kind}")


def _set_learning_rate(optimizer: torch.optim.Optimizer, optimization: OptimizationSpec, step: int) -> float:
    lr = _compute_learning_rate(optimization, step)
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = lr
    return lr


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _format_metric_line(prefix: str, metrics: dict[str, Any], ordered_keys: list[str]) -> str:
    parts = [prefix]
    for key in ordered_keys:
        if key in metrics:
            parts.append(f"{key}={_format_metric_value(metrics[key])}")
    return " | ".join(parts)


def run_evaluation_suite(
    *,
    model: torch.nn.Module,
    spec: TrainSpec,
    vocab: Vocabulary,
    device: torch.device,
) -> dict[str, Any]:
    eval_results: dict[str, Any] = {}
    for split_name in spec.evaluation.tracked_splits:
        loader = make_data_loader(
            benchmark_dir=spec.benchmark_dir,
            split_name=split_name,
            batch_size=spec.evaluation.batch_size,
            shuffle=False,
            num_workers=spec.num_workers,
            pad_token_id=vocab.pad_token_id,
        )
        eval_results[split_name] = evaluate_split(
            model=model,
            data_loader=loader,
            device=device,
            pad_token_id=vocab.pad_token_id,
            value_token_ids=vocab.value_token_ids,
            max_batches=spec.evaluation.max_eval_batches,
            include_analysis=(split_name in spec.evaluation.analysis_splits),
        )
    return eval_results


def _load_existing_run_config(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def _load_best_checkpoint_record(path: Path, checkpoint_path: Path) -> dict[str, Any] | None:
    if path.exists() != checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint state is inconsistent: metadata={path.exists()} checkpoint={checkpoint_path.exists()}."
        )
    if not path.exists():
        return None
    return read_json(path)


def _coerce_metric_float(value: Any, label: str) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric, but found {type(value).__name__}.")
    return float(value)


def _is_better_checkpoint(
    *,
    step: int,
    split_metrics: dict[str, Any],
    best_record: dict[str, Any] | None,
    metric_name: str,
    maximize: bool,
) -> tuple[bool, float, float]:
    metric_value = _coerce_metric_float(split_metrics[metric_name], metric_name)
    loss_value = _coerce_metric_float(split_metrics["loss"], "loss")
    if best_record is None:
        return True, metric_value, loss_value

    best_metric_value = _coerce_metric_float(best_record["metric_value"], "best metric_value")
    best_loss_value = _coerce_metric_float(best_record["loss"], "best loss")
    best_step = int(best_record["step"])

    if maximize:
        if metric_value > best_metric_value:
            return True, metric_value, loss_value
        if metric_value < best_metric_value:
            return False, metric_value, loss_value
    else:
        if metric_value < best_metric_value:
            return True, metric_value, loss_value
        if metric_value > best_metric_value:
            return False, metric_value, loss_value

    if loss_value < best_loss_value:
        return True, metric_value, loss_value
    if loss_value > best_loss_value:
        return False, metric_value, loss_value
    return step < best_step, metric_value, loss_value


def _resume_training_state(
    *,
    spec: TrainSpec,
    resume_checkpoint: Path,
) -> dict[str, Any]:
    context = build_run_context(spec)
    model: torch.nn.Module = context["model"]
    optimizer: torch.optim.Optimizer = context["optimizer"]
    device: torch.device = context["device"]
    checkpoint = load_checkpoint(resume_checkpoint, device)
    checkpoint_step = int(checkpoint["step"])
    if checkpoint_step >= spec.num_steps:
        raise ValueError(
            f"Resume checkpoint step {checkpoint_step} is not below target num_steps={spec.num_steps}."
        )
    load_model_state(model, checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return {
        **context,
        "checkpoint": checkpoint,
        "start_step": checkpoint_step,
        "latest_eval": checkpoint.get("metrics", {}),
    }


def train_from_config(
    config_path: Path,
    *,
    overwrite: bool = False,
    resume_checkpoint: Path | None = None,
) -> Path:
    spec = TrainSpec.from_path(config_path)
    if resume_checkpoint is None:
        if spec.output_dir.exists() and any(spec.output_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {spec.output_dir}")
        if spec.output_dir.exists() and overwrite:
            shutil.rmtree(spec.output_dir)
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(spec.output_dir / "run_config.json", _to_jsonable(spec))
    else:
        if overwrite:
            raise ValueError("overwrite=True is incompatible with resume_checkpoint.")
        if not spec.output_dir.exists():
            raise FileNotFoundError(f"Cannot resume because output_dir does not exist: {spec.output_dir}")
        existing_run_config = _load_existing_run_config(spec.output_dir / "run_config.json")
        if existing_run_config is None:
            raise FileNotFoundError(f"Missing existing run_config.json in {spec.output_dir}")
        if str(existing_run_config.get("benchmark_dir")) != str(spec.benchmark_dir):
            raise ValueError(
                f"Resume benchmark_dir mismatch: existing={existing_run_config.get('benchmark_dir')} "
                f"new={spec.benchmark_dir}"
            )
        write_json(spec.output_dir / f"resume_config_step_{resume_checkpoint.stem}.json", _to_jsonable(spec))

    set_seed(spec.seed)
    context = _resume_training_state(spec=spec, resume_checkpoint=resume_checkpoint) if resume_checkpoint else build_run_context(spec)
    metadata = context["metadata"]
    vocab: Vocabulary = context["vocab"]
    device: torch.device = context["device"]
    model: torch.nn.Module = context["model"]
    optimizer: torch.optim.Optimizer = context["optimizer"]

    train_loader = make_data_loader(
        benchmark_dir=spec.benchmark_dir,
        split_name="train",
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=spec.num_workers,
        pad_token_id=vocab.pad_token_id,
    )
    train_iterator = iter(train_loader)
    metrics_path = spec.output_dir / "metrics.jsonl"
    checkpoint_dir = spec.output_dir / "checkpoints"
    best_checkpoint_path = checkpoint_dir / "best.pt"
    best_checkpoint_record_path = spec.output_dir / "best_checkpoint.json"
    start_step = int(context.get("start_step", 0))
    best_checkpoint_record = _load_best_checkpoint_record(best_checkpoint_record_path, best_checkpoint_path)

    progress_bar = tqdm(
        total=spec.num_steps - start_step,
        desc=f"train:{spec.run_name}",
        dynamic_ncols=True,
        leave=True,
    )
    progress_bar.write(
        _format_metric_line(
            "run",
            {
                "device": spec.device,
                "seed": spec.seed,
                "params": model.count_parameters(),
                "train_samples": len(train_loader.dataset),
                "batch_size": spec.batch_size,
            },
            ["device", "seed", "params", "train_samples", "batch_size"],
        )
    )

    latest_eval: dict[str, Any] = dict(context.get("latest_eval", {}))
    if resume_checkpoint is not None:
        append_jsonl(
            metrics_path,
            {
                "phase": "resume",
                "step": start_step,
                "resume_checkpoint": str(resume_checkpoint),
                "target_num_steps": spec.num_steps,
            },
        )
        progress_bar.write(
            f"resume step={start_step} from={resume_checkpoint} target_num_steps={spec.num_steps}"
        )
    try:
        for step in range(start_step + 1, spec.num_steps + 1):
            model.train()
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else {
                    inner_key: inner_value.to(device) for inner_key, inner_value in value.items()
                } if isinstance(value, dict) else value
                for key, value in batch.items()
            }
            lr = _set_learning_rate(
                optimizer,
                spec.optimization,
                step,
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            loss, token_accuracy = compute_lm_loss(
                logits=outputs.logits,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pad_token_id=vocab.pad_token_id,
            )
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), spec.optimization.grad_clip_norm).item()
            optimizer.step()

            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                tok_acc=f"{token_accuracy.item():.4f}",
                lr=f"{lr:.2e}",
            )

            if step % spec.log_every_steps == 0 or step == 1:
                train_metrics = {
                    "phase": "train",
                    "step": step,
                    "loss": loss.item(),
                    "token_accuracy": token_accuracy.item(),
                    "grad_norm": grad_norm,
                    "learning_rate": lr,
                }
                append_jsonl(metrics_path, train_metrics)
                progress_bar.write(
                    _format_metric_line(
                        f"train step={step}",
                        train_metrics,
                        ["loss", "token_accuracy", "grad_norm", "learning_rate"],
                    )
                )

            if step % spec.eval_every_steps == 0 or step == spec.num_steps:
                model.eval()
                latest_eval = run_evaluation_suite(
                    model=model,
                    spec=spec,
                    vocab=vocab,
                    device=device,
                )
                for split_name, split_metrics in latest_eval.items():
                    append_jsonl(
                        metrics_path,
                        {
                            "phase": "eval",
                            "step": step,
                            "split": split_name,
                            **split_metrics,
                        },
                    )
                    ordered_keys = [
                        "loss",
                        "answer_accuracy",
                        "token_accuracy",
                        "read_key_accuracy",
                        "write_key_accuracy",
                        "write_value_accuracy",
                    ]
                    if split_name == "validation_iid":
                        ordered_keys.extend(["q", "r", "w"])
                    progress_bar.write(
                        _format_metric_line(
                            f"eval step={step} split={split_name}",
                            split_metrics,
                            ordered_keys,
                        )
                    )
                if spec.checkpointing.save_best_checkpoint:
                    if spec.checkpointing.best_checkpoint_split not in latest_eval:
                        raise KeyError(
                            f"best_checkpoint_split '{spec.checkpointing.best_checkpoint_split}' "
                            f"was not evaluated. tracked_splits={spec.evaluation.tracked_splits}"
                        )
                    best_split_metrics = latest_eval[spec.checkpointing.best_checkpoint_split]
                    if spec.checkpointing.best_checkpoint_metric not in best_split_metrics:
                        raise KeyError(
                            f"best_checkpoint_metric '{spec.checkpointing.best_checkpoint_metric}' "
                            f"is missing from split '{spec.checkpointing.best_checkpoint_split}'."
                        )
                    should_save_best, best_metric_value, best_loss_value = _is_better_checkpoint(
                        step=step,
                        split_metrics=best_split_metrics,
                        best_record=best_checkpoint_record,
                        metric_name=spec.checkpointing.best_checkpoint_metric,
                        maximize=spec.checkpointing.best_checkpoint_maximize,
                    )
                    if should_save_best:
                        save_checkpoint(
                            path=best_checkpoint_path,
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            metrics=latest_eval,
                            config={
                                "train_spec": _to_jsonable(spec),
                                "benchmark_metadata": metadata,
                                "model_parameter_count": model.count_parameters(),
                            },
                        )
                        best_checkpoint_record = {
                            "path": str(best_checkpoint_path),
                            "step": step,
                            "split": spec.checkpointing.best_checkpoint_split,
                            "metric": spec.checkpointing.best_checkpoint_metric,
                            "maximize": spec.checkpointing.best_checkpoint_maximize,
                            "metric_value": best_metric_value,
                            "loss": best_loss_value,
                        }
                        write_json(best_checkpoint_record_path, best_checkpoint_record)
                        progress_bar.write(
                            _format_metric_line(
                                f"best checkpoint step={step}",
                                {
                                    "split": spec.checkpointing.best_checkpoint_split,
                                    "metric": spec.checkpointing.best_checkpoint_metric,
                                    "value": best_metric_value,
                                    "loss": best_loss_value,
                                    "path": str(best_checkpoint_path),
                                },
                                ["split", "metric", "value", "loss", "path"],
                            )
                        )

            if spec.checkpointing.save_step_checkpoints and (
                step % spec.checkpoint_every_steps == 0 or step == spec.num_steps
            ):
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
                    },
                )
                progress_bar.write(f"checkpoint step={step} path={checkpoint_path}")
    finally:
        progress_bar.close()

    summary = {
        "run_name": spec.run_name,
        "num_steps": spec.num_steps,
        "device": spec.device,
        "model_parameter_count": model.count_parameters(),
        "latest_eval": latest_eval,
        "best_checkpoint": best_checkpoint_record,
    }
    write_json(spec.output_dir / "summary.json", summary)
    return spec.output_dir


def load_model_from_checkpoint(
    *,
    config_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    spec = TrainSpec.from_path(config_path)
    context = build_run_context(spec)
    model: torch.nn.Module = context["model"]
    optimizer: torch.optim.Optimizer = context["optimizer"]
    device: torch.device = context["device"]
    checkpoint = load_checkpoint(checkpoint_path, device)
    load_model_state(model, checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return {
        "spec": spec,
        "metadata": context["metadata"],
        "vocab": context["vocab"],
        "device": device,
        "model": model,
        "optimizer": optimizer,
        "checkpoint": checkpoint,
    }
