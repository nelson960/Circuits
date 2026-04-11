from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import read_symbolic_kv_stream_metadata
from circuit.eval import evaluate_split
from circuit.io import read_json
from circuit.runtime import build_model, load_checkpoint, load_model_state
from circuit.train import make_data_loader
from circuit.vocab import Vocabulary

REFERENCE_SPLITS = [
    "validation_iid",
    "test_iid",
    "heldout_pairs",
    "structural_ood",
    "counterfactual",
]


def _require_best_checkpoint_path(run_dir: Path) -> tuple[dict[str, Any], Path]:
    best_record_path = run_dir / "best_checkpoint.json"
    if not best_record_path.exists():
        raise FileNotFoundError(f"Missing best_checkpoint.json in run directory: {run_dir}")
    best_record = read_json(best_record_path)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best checkpoint file: {checkpoint_path}")
    if Path(str(best_record["path"])).name != checkpoint_path.name:
        raise ValueError(
            f"best_checkpoint.json path does not reference {checkpoint_path.name}: {best_record['path']}"
        )
    return best_record, checkpoint_path


def _compact_split_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss": metrics["loss"],
        "answer_accuracy": metrics["answer_accuracy"],
        "token_accuracy": metrics["token_accuracy"],
        "read_key_accuracy": metrics["read_key_accuracy"],
        "write_key_accuracy": metrics["write_key_accuracy"],
        "write_value_accuracy": metrics["write_value_accuracy"],
    }


def evaluate_reference_candidate(run_dir: Path, *, device_name: str = "cpu") -> dict[str, Any]:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json in run directory: {run_dir}")
    spec = TrainSpec.from_path(run_config_path)
    best_record, checkpoint_path = _require_best_checkpoint_path(run_dir)

    device = torch.device(device_name)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    model = build_model(spec.model, len(vocab.tokens), device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()

    split_metrics: dict[str, dict[str, Any]] = {}
    for split_name in REFERENCE_SPLITS:
        loader = make_data_loader(
            benchmark_dir=spec.benchmark_dir,
            split_name=split_name,
            batch_size=spec.evaluation.batch_size,
            shuffle=False,
            num_workers=spec.num_workers,
            pad_token_id=vocab.pad_token_id,
        )
        metrics = evaluate_split(
            model=model,
            data_loader=loader,
            device=device,
            pad_token_id=vocab.pad_token_id,
            value_token_ids=vocab.value_token_ids,
            max_batches=None,
            include_analysis=(split_name == "validation_iid"),
        )
        split_metrics[split_name] = _compact_split_metrics(metrics)

    return {
        "run_dir": str(run_dir),
        "run_name": spec.run_name,
        "run_config_path": str(run_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "selection_checkpoint_metric": str(best_record["metric"]),
        "selection_checkpoint_split": str(best_record["split"]),
        "model_parameter_count": model.count_parameters(),
        "metrics": split_metrics,
    }


def rank_reference_candidates(
    candidates: list[dict[str, Any]],
    *,
    min_validation_answer_accuracy: float,
) -> list[dict[str, Any]]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["metrics"]["validation_iid"]["answer_accuracy"] >= min_validation_answer_accuracy
    ]
    if not eligible:
        raise RuntimeError(
            f"No candidate meets min_validation_answer_accuracy={min_validation_answer_accuracy:.4f}."
        )

    def sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, int]:
        metrics = candidate["metrics"]
        return (
            metrics["heldout_pairs"]["answer_accuracy"],
            metrics["validation_iid"]["answer_accuracy"],
            metrics["structural_ood"]["answer_accuracy"],
            metrics["test_iid"]["answer_accuracy"],
            metrics["counterfactual"]["answer_accuracy"],
            -int(candidate["model_parameter_count"]),
        )

    ranked = sorted(eligible, key=sort_key, reverse=True)
    return ranked


def select_reference_configuration(
    run_dirs: list[Path],
    *,
    device_name: str = "cpu",
    min_validation_answer_accuracy: float = 0.9,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("run_dirs must not be empty.")
    candidates = [evaluate_reference_candidate(run_dir, device_name=device_name) for run_dir in run_dirs]
    ranked = rank_reference_candidates(
        candidates,
        min_validation_answer_accuracy=min_validation_answer_accuracy,
    )
    return {
        "selection_policy": {
            "device": device_name,
            "minimum_validation_answer_accuracy": min_validation_answer_accuracy,
            "ranking_order": [
                "heldout_pairs.answer_accuracy",
                "validation_iid.answer_accuracy",
                "structural_ood.answer_accuracy",
                "test_iid.answer_accuracy",
                "counterfactual.answer_accuracy",
                "smaller_model_parameter_count",
            ],
        },
        "selected": ranked[0],
        "ranking": ranked,
    }
