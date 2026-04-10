from __future__ import annotations

from pathlib import Path

from circut.io import write_json


def write_small_benchmark_config(path: Path) -> Path:
    write_json(
        path,
        {
            "benchmark_type": "symbolic_kv_stream",
            "name": "test_symbolic_kv_stream",
            "output_dir": str(path.parent / "benchmark"),
            "seed": 11,
            "num_keys": 6,
            "num_values": 64,
            "holdout_answer_pair_fraction": 0.15,
            "heuristic_max_accuracy": 0.8,
            "train": {
                "num_samples": 64,
                "active_keys": {"min": 3, "max": 4},
                "overwrite_count": {"min": 5, "max": 6},
                "num_queries": {"min": 4, "max": 5},
                "query_lag": {"min": 1, "max": 2},
            },
            "validation_iid": {
                "num_samples": 16,
                "active_keys": {"min": 3, "max": 4},
                "overwrite_count": {"min": 5, "max": 6},
                "num_queries": {"min": 4, "max": 5},
                "query_lag": {"min": 1, "max": 2},
            },
            "test_iid": {
                "num_samples": 16,
                "active_keys": {"min": 3, "max": 4},
                "overwrite_count": {"min": 5, "max": 6},
                "num_queries": {"min": 4, "max": 5},
                "query_lag": {"min": 1, "max": 2},
            },
            "heldout_pairs": {
                "num_samples": 16,
                "active_keys": {"min": 3, "max": 4},
                "overwrite_count": {"min": 5, "max": 6},
                "num_queries": {"min": 4, "max": 5},
                "query_lag": {"min": 1, "max": 2},
            },
            "structural_ood": {
                "num_samples": 16,
                "active_keys": {"min": 5, "max": 5},
                "overwrite_count": {"min": 8, "max": 8},
                "num_queries": {"min": 6, "max": 6},
                "query_lag": {"min": 2, "max": 3},
            },
            "counterfactual": {
                "num_samples": 8,
                "source_split": "test_iid",
            },
        },
    )
    return path


def write_small_train_config(
    path: Path,
    benchmark_dir: Path,
    *,
    num_steps: int = 4,
    max_eval_batches: int | None = 2,
    save_step_checkpoints: bool = True,
    save_best_checkpoint: bool = True,
) -> Path:
    write_json(
        path,
        {
            "run_name": "test_train",
            "seed": 5,
            "device": "cpu",
            "benchmark_dir": str(benchmark_dir),
            "output_dir": str(path.parent / "run"),
            "batch_size": 8,
            "num_steps": num_steps,
            "log_every_steps": 1,
            "eval_every_steps": 2,
            "checkpoint_every_steps": 2,
            "num_workers": 0,
            "model": {
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 2,
                "d_ff": 64,
                "dropout": 0.0,
                "max_seq_len": 96,
            },
            "optimization": {
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.95,
                "grad_clip_norm": 1.0,
                "warmup_steps": 1,
                "schedule": {
                    "kind": "constant"
                }
            },
            "evaluation": {
                "batch_size": 8,
                "max_eval_batches": max_eval_batches,
                "max_analysis_batches": 1,
                "tracked_splits": ["validation_iid"],
                "analysis_splits": ["validation_iid"],
                "birth_thresholds": {
                    "answer_accuracy": 0.5,
                    "q": 0.0,
                    "r": 0.0,
                    "w": -10.0,
                },
            },
            "checkpointing": {
                "save_step_checkpoints": save_step_checkpoints,
                "save_best_checkpoint": save_best_checkpoint,
                "best_checkpoint_split": "validation_iid",
                "best_checkpoint_metric": "answer_accuracy",
                "best_checkpoint_maximize": True,
            },
        },
    )
    return path
