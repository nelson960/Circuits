from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from circuit.io import read_json


def _pop_required(payload: dict[str, Any], key: str, context: str) -> Any:
    if key not in payload:
        raise KeyError(f"Missing key '{key}' in {context}.")
    return payload.pop(key)


def _ensure_empty(payload: dict[str, Any], context: str) -> None:
    if payload:
        raise ValueError(f"Unexpected keys in {context}: {sorted(payload)}")


@dataclass(frozen=True)
class AxisRange:
    min: int
    max: int

    def __post_init__(self) -> None:
        if self.min < 0:
            raise ValueError("AxisRange min must be non-negative.")
        if self.max < self.min:
            raise ValueError("AxisRange max must be >= min.")

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "AxisRange":
        payload = dict(data)
        instance = cls(
            min=int(_pop_required(payload, "min", context)),
            max=int(_pop_required(payload, "max", context)),
        )
        _ensure_empty(payload, context)
        return instance


@dataclass(frozen=True)
class SplitSpec:
    num_samples: int
    num_pairs: AxisRange
    distractor_count: AxisRange
    overwrite_count: AxisRange

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("SplitSpec num_samples must be positive.")
        if self.num_pairs.min < 2:
            raise ValueError("SplitSpec requires at least 2 keys per sample.")

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "SplitSpec":
        payload = dict(data)
        instance = cls(
            num_samples=int(_pop_required(payload, "num_samples", context)),
            num_pairs=AxisRange.from_dict(_pop_required(payload, "num_pairs", context), f"{context}.num_pairs"),
            distractor_count=AxisRange.from_dict(
                _pop_required(payload, "distractor_count", context),
                f"{context}.distractor_count",
            ),
            overwrite_count=AxisRange.from_dict(
                _pop_required(payload, "overwrite_count", context),
                f"{context}.overwrite_count",
            ),
        )
        _ensure_empty(payload, context)
        return instance


@dataclass(frozen=True)
class CounterfactualSpec:
    num_samples: int
    source_split: str

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("CounterfactualSpec num_samples must be positive.")

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "CounterfactualSpec":
        payload = dict(data)
        instance = cls(
            num_samples=int(_pop_required(payload, "num_samples", context)),
            source_split=str(_pop_required(payload, "source_split", context)),
        )
        _ensure_empty(payload, context)
        return instance


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    output_dir: Path
    seed: int
    num_keys: int
    num_values: int
    holdout_answer_pair_fraction: float
    heuristic_max_accuracy: float
    train: SplitSpec
    validation_iid: SplitSpec
    test_iid: SplitSpec
    heldout_pairs: SplitSpec
    structural_ood: SplitSpec
    counterfactual: CounterfactualSpec

    def __post_init__(self) -> None:
        if self.num_keys < 2:
            raise ValueError("BenchmarkSpec num_keys must be at least 2.")
        if self.num_values < self.num_keys:
            raise ValueError("BenchmarkSpec num_values must be >= num_keys.")
        if not 0.0 < self.holdout_answer_pair_fraction < 1.0:
            raise ValueError("holdout_answer_pair_fraction must be between 0 and 1.")
        if not 0.0 <= self.heuristic_max_accuracy <= 1.0:
            raise ValueError("heuristic_max_accuracy must be between 0 and 1.")

    @classmethod
    def from_path(cls, path: Path) -> "BenchmarkSpec":
        payload = dict(read_json(path))
        instance = cls(
            name=str(_pop_required(payload, "name", "benchmark config")),
            output_dir=Path(_pop_required(payload, "output_dir", "benchmark config")),
            seed=int(_pop_required(payload, "seed", "benchmark config")),
            num_keys=int(_pop_required(payload, "num_keys", "benchmark config")),
            num_values=int(_pop_required(payload, "num_values", "benchmark config")),
            holdout_answer_pair_fraction=float(
                _pop_required(payload, "holdout_answer_pair_fraction", "benchmark config")
            ),
            heuristic_max_accuracy=float(_pop_required(payload, "heuristic_max_accuracy", "benchmark config")),
            train=SplitSpec.from_dict(_pop_required(payload, "train", "benchmark config"), "benchmark config.train"),
            validation_iid=SplitSpec.from_dict(
                _pop_required(payload, "validation_iid", "benchmark config"),
                "benchmark config.validation_iid",
            ),
            test_iid=SplitSpec.from_dict(
                _pop_required(payload, "test_iid", "benchmark config"),
                "benchmark config.test_iid",
            ),
            heldout_pairs=SplitSpec.from_dict(
                _pop_required(payload, "heldout_pairs", "benchmark config"),
                "benchmark config.heldout_pairs",
            ),
            structural_ood=SplitSpec.from_dict(
                _pop_required(payload, "structural_ood", "benchmark config"),
                "benchmark config.structural_ood",
            ),
            counterfactual=CounterfactualSpec.from_dict(
                _pop_required(payload, "counterfactual", "benchmark config"),
                "benchmark config.counterfactual",
            ),
        )
        _ensure_empty(payload, "benchmark config")
        return instance


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    dropout: float
    max_seq_len: int

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0 or self.d_ff <= 0:
            raise ValueError("ModelSpec dimensions must be positive.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("ModelSpec d_model must be divisible by n_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("ModelSpec dropout must be in [0, 1).")

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "ModelSpec":
        payload = dict(data)
        instance = cls(
            d_model=int(_pop_required(payload, "d_model", context)),
            n_layers=int(_pop_required(payload, "n_layers", context)),
            n_heads=int(_pop_required(payload, "n_heads", context)),
            d_ff=int(_pop_required(payload, "d_ff", context)),
            dropout=float(_pop_required(payload, "dropout", context)),
            max_seq_len=int(_pop_required(payload, "max_seq_len", context)),
        )
        _ensure_empty(payload, context)
        return instance


@dataclass(frozen=True)
class LearningRateScheduleSpec:
    kind: str
    decay_start_step: int | None
    decay_end_step: int | None
    min_learning_rate: float | None

    def __post_init__(self) -> None:
        if self.kind not in {"constant", "cosine_decay"}:
            raise ValueError(f"Unsupported learning-rate schedule kind: {self.kind}")
        if self.kind == "constant":
            if self.decay_start_step is not None or self.decay_end_step is not None or self.min_learning_rate is not None:
                raise ValueError("Constant schedule must not set decay_start_step, decay_end_step, or min_learning_rate.")
            return
        if self.decay_start_step is None or self.decay_end_step is None or self.min_learning_rate is None:
            raise ValueError("cosine_decay schedule requires decay_start_step, decay_end_step, and min_learning_rate.")
        if self.decay_start_step < 0:
            raise ValueError("decay_start_step must be non-negative.")
        if self.decay_end_step <= self.decay_start_step:
            raise ValueError("decay_end_step must be greater than decay_start_step.")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate must be non-negative.")

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "LearningRateScheduleSpec":
        payload = dict(data)
        kind = str(_pop_required(payload, "kind", context))
        instance = cls(
            kind=kind,
            decay_start_step=None if "decay_start_step" not in payload else int(payload.pop("decay_start_step")),
            decay_end_step=None if "decay_end_step" not in payload else int(payload.pop("decay_end_step")),
            min_learning_rate=None
            if "min_learning_rate" not in payload
            else float(payload.pop("min_learning_rate")),
        )
        _ensure_empty(payload, context)
        return instance


@dataclass(frozen=True)
class OptimizationSpec:
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip_norm: float
    warmup_steps: int
    schedule: LearningRateScheduleSpec

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0.0 < self.beta1 < 1.0 or not 0.0 < self.beta2 < 1.0:
            raise ValueError("Adam betas must be in (0, 1).")
        if self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if self.schedule.kind == "cosine_decay":
            if self.schedule.min_learning_rate is None:
                raise ValueError("cosine_decay schedule requires min_learning_rate.")
            if self.schedule.min_learning_rate > self.learning_rate:
                raise ValueError("min_learning_rate must be <= learning_rate.")
            if self.schedule.decay_start_step is None or self.schedule.decay_end_step is None:
                raise ValueError("cosine_decay schedule requires decay_start_step and decay_end_step.")
            if self.schedule.decay_start_step < self.warmup_steps:
                raise ValueError("decay_start_step must be >= warmup_steps.")

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "OptimizationSpec":
        payload = dict(data)
        instance = cls(
            learning_rate=float(_pop_required(payload, "learning_rate", context)),
            weight_decay=float(_pop_required(payload, "weight_decay", context)),
            beta1=float(_pop_required(payload, "beta1", context)),
            beta2=float(_pop_required(payload, "beta2", context)),
            grad_clip_norm=float(_pop_required(payload, "grad_clip_norm", context)),
            warmup_steps=int(_pop_required(payload, "warmup_steps", context)),
            schedule=LearningRateScheduleSpec.from_dict(
                _pop_required(payload, "schedule", context),
                f"{context}.schedule",
            ),
        )
        _ensure_empty(payload, context)
        return instance


@dataclass(frozen=True)
class EvaluationSpec:
    batch_size: int
    max_eval_batches: int | None
    max_analysis_batches: int
    tracked_splits: list[str]
    analysis_splits: list[str]
    birth_thresholds: dict[str, float]

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("Evaluation batch_size must be positive.")
        if self.max_eval_batches is not None and self.max_eval_batches <= 0:
            raise ValueError("max_eval_batches must be positive when provided.")
        if self.max_analysis_batches <= 0:
            raise ValueError("max_analysis_batches must be positive.")
        if not self.tracked_splits:
            raise ValueError("tracked_splits must not be empty.")
        if not self.analysis_splits:
            raise ValueError("analysis_splits must not be empty.")
        if not self.birth_thresholds:
            raise ValueError("birth_thresholds must not be empty.")

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "EvaluationSpec":
        payload = dict(data)
        thresholds = dict(_pop_required(payload, "birth_thresholds", context))
        raw_max_eval_batches = payload.pop("max_eval_batches", None)
        if "max_eval_batches" not in data:
            raise KeyError(f"Missing key 'max_eval_batches' in {context}.")
        instance = cls(
            batch_size=int(_pop_required(payload, "batch_size", context)),
            max_eval_batches=None if raw_max_eval_batches is None else int(raw_max_eval_batches),
            max_analysis_batches=int(_pop_required(payload, "max_analysis_batches", context)),
            tracked_splits=[str(item) for item in _pop_required(payload, "tracked_splits", context)],
            analysis_splits=[str(item) for item in _pop_required(payload, "analysis_splits", context)],
            birth_thresholds={str(key): float(value) for key, value in thresholds.items()},
        )
        _ensure_empty(payload, context)
        return instance


@dataclass(frozen=True)
class CheckpointSpec:
    save_step_checkpoints: bool
    save_best_checkpoint: bool
    best_checkpoint_split: str
    best_checkpoint_metric: str
    best_checkpoint_maximize: bool

    def __post_init__(self) -> None:
        if not self.save_step_checkpoints and not self.save_best_checkpoint:
            raise ValueError("CheckpointSpec must enable at least one checkpoint mode.")
        if not self.best_checkpoint_split:
            raise ValueError("best_checkpoint_split must not be empty.")
        if not self.best_checkpoint_metric:
            raise ValueError("best_checkpoint_metric must not be empty.")

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "CheckpointSpec":
        payload = dict(data)
        instance = cls(
            save_step_checkpoints=bool(_pop_required(payload, "save_step_checkpoints", context)),
            save_best_checkpoint=bool(_pop_required(payload, "save_best_checkpoint", context)),
            best_checkpoint_split=str(_pop_required(payload, "best_checkpoint_split", context)),
            best_checkpoint_metric=str(_pop_required(payload, "best_checkpoint_metric", context)),
            best_checkpoint_maximize=bool(_pop_required(payload, "best_checkpoint_maximize", context)),
        )
        _ensure_empty(payload, context)
        return instance


@dataclass(frozen=True)
class TrainSpec:
    run_name: str
    seed: int
    device: str
    benchmark_dir: Path
    output_dir: Path
    batch_size: int
    num_steps: int
    log_every_steps: int
    eval_every_steps: int
    checkpoint_every_steps: int
    num_workers: int
    model: ModelSpec
    optimization: OptimizationSpec
    evaluation: EvaluationSpec
    checkpointing: CheckpointSpec

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.num_steps <= 0:
            raise ValueError("TrainSpec batch_size and num_steps must be positive.")
        if self.log_every_steps <= 0 or self.eval_every_steps <= 0 or self.checkpoint_every_steps <= 0:
            raise ValueError("TrainSpec step intervals must be positive.")
        if self.num_workers < 0:
            raise ValueError("TrainSpec num_workers must be non-negative.")

    @classmethod
    def from_path(cls, path: Path) -> "TrainSpec":
        payload = dict(read_json(path))
        instance = cls(
            run_name=str(_pop_required(payload, "run_name", "train config")),
            seed=int(_pop_required(payload, "seed", "train config")),
            device=str(_pop_required(payload, "device", "train config")),
            benchmark_dir=Path(_pop_required(payload, "benchmark_dir", "train config")),
            output_dir=Path(_pop_required(payload, "output_dir", "train config")),
            batch_size=int(_pop_required(payload, "batch_size", "train config")),
            num_steps=int(_pop_required(payload, "num_steps", "train config")),
            log_every_steps=int(_pop_required(payload, "log_every_steps", "train config")),
            eval_every_steps=int(_pop_required(payload, "eval_every_steps", "train config")),
            checkpoint_every_steps=int(_pop_required(payload, "checkpoint_every_steps", "train config")),
            num_workers=int(_pop_required(payload, "num_workers", "train config")),
            model=ModelSpec.from_dict(_pop_required(payload, "model", "train config"), "train config.model"),
            optimization=OptimizationSpec.from_dict(
                _pop_required(payload, "optimization", "train config"),
                "train config.optimization",
            ),
            evaluation=EvaluationSpec.from_dict(
                _pop_required(payload, "evaluation", "train config"),
                "train config.evaluation",
            ),
            checkpointing=CheckpointSpec.from_dict(
                _pop_required(payload, "checkpointing", "train config"),
                "train config.checkpointing",
            ),
        )
        _ensure_empty(payload, "train config")
        return instance
