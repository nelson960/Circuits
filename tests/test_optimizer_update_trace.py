from __future__ import annotations

from pathlib import Path

import pytest

from circuit.analysis.actual_batch_route_attribution import run_actual_batch_route_attribution
from circuit.analysis.checkpoint_sweep import generate_probe_set
from circuit.analysis.optimizer_update_trace import run_optimizer_update_trace
from circuit.data.symbolic_kv_stream import generate_symbolic_kv_stream_benchmark
from circuit.io import iter_jsonl, read_json
from circuit.train import train_from_config

from .helpers import write_small_train_config


def test_optimizer_update_trace_records_dense_update_window(
    tmp_path: Path,
    benchmark_config_path: Path,
) -> None:
    benchmark_dir = generate_symbolic_kv_stream_benchmark(benchmark_config_path)
    train_config = write_small_train_config(tmp_path / "train_config.json", benchmark_dir, num_steps=4)
    run_dir = train_from_config(train_config)
    resume_checkpoint = run_dir / "checkpoints" / "step_000002.pt"

    report_path, markdown_path, step_rows_path, batch_rows_path, parameter_rows_path, checkpoint_dir = (
        run_optimizer_update_trace(
            config_path=train_config,
            resume_checkpoint=resume_checkpoint,
            output_dir=tmp_path / "optimizer_update_trace",
            end_step=4,
            device_name="cpu",
            train_split="train",
            checkpoint_every_steps=1,
            progress_every_steps=0,
            top_k_parameters=3,
        )
    )

    report = read_json(report_path)
    step_rows = list(iter_jsonl(step_rows_path))
    batch_rows = list(iter_jsonl(batch_rows_path))
    parameter_rows = list(iter_jsonl(parameter_rows_path))

    assert report_path.exists()
    assert markdown_path.exists()
    assert report["historical_replay_status"] == "instrumented_continuation_not_historical_replay"
    assert report["trace_start_step"] == 2
    assert report["trace_end_step"] == 4
    assert report["summary"]["num_traced_steps"] == 2
    assert len(step_rows) == 2
    assert len(batch_rows) == 2
    assert len(parameter_rows) <= 6
    assert all(row["parameter_delta_l2"] > 0.0 for row in step_rows)
    assert all(row["sample_ids"] for row in batch_rows)
    assert (checkpoint_dir / "step_000002.pt").exists()
    assert (checkpoint_dir / "step_000003.pt").exists()
    assert (checkpoint_dir / "step_000004.pt").exists()

    probe_set_path, _ = generate_probe_set(
        benchmark_dir=benchmark_dir,
        output_path=tmp_path / "probe_set.jsonl",
        examples_per_split=2,
        seed=17,
    )
    actual_report_path, actual_markdown_path, actual_rows_path, actual_pair_rows_path = (
        run_actual_batch_route_attribution(
            config_path=train_config,
            probe_set_path=probe_set_path,
            optimizer_trace_dir=tmp_path / "optimizer_update_trace",
            output_dir=tmp_path / "actual_batch_route_attribution",
            raw_route_specs=[
                "label=embedding_key_identity,stage=embedding,subspace=embedding_key_identity,rank=1,position_role=query_key",
            ],
            route_pair_type="query_key",
            pair_types=["query_key"],
            device_name="cpu",
            max_pairs_per_type=2,
            min_pairs_per_type=1,
        )
    )
    actual_report = read_json(actual_report_path)
    actual_rows = list(iter_jsonl(actual_rows_path))

    assert actual_report_path.exists()
    assert actual_markdown_path.exists()
    assert actual_pair_rows_path.exists()
    assert actual_report["summary"]["num_routes"] == 1
    assert actual_report["summary"]["num_intervals"] == 2
    assert len(actual_rows) == 2
    assert all(row["data_group_id"] == "actual_batch" for row in actual_rows)
    assert all(row["actual_batch_sample_count"] == 8 for row in actual_rows)
    assert actual_report["summary_max_abs_loss_mismatch"] <= actual_report["loss_match_tolerance"]


def test_optimizer_update_trace_rejects_historical_replay_claim(
    tmp_path: Path,
    benchmark_config_path: Path,
) -> None:
    benchmark_dir = generate_symbolic_kv_stream_benchmark(benchmark_config_path)
    train_config = write_small_train_config(tmp_path / "train_config.json", benchmark_dir, num_steps=4)
    run_dir = train_from_config(train_config)

    with pytest.raises(RuntimeError, match="DataLoader sampler order"):
        run_optimizer_update_trace(
            config_path=train_config,
            resume_checkpoint=run_dir / "checkpoints" / "step_000002.pt",
            output_dir=tmp_path / "optimizer_update_trace",
            end_step=4,
            device_name="cpu",
            train_split="train",
            checkpoint_every_steps=1,
            progress_every_steps=0,
            top_k_parameters=3,
            require_historical_replay=True,
        )
