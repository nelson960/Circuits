from __future__ import annotations

from pathlib import Path

from circut.data.symbolic_kv_stream import generate_symbolic_kv_stream_benchmark, oracle_query_answers
from circut.io import iter_jsonl, read_json


def test_generate_symbolic_kv_benchmark(benchmark_config_path: Path) -> None:
    benchmark_dir = generate_symbolic_kv_stream_benchmark(benchmark_config_path)
    metadata = read_json(benchmark_dir / "metadata.json")
    train_records = list(iter_jsonl(benchmark_dir / "train.jsonl"))
    counterfactual_records = list(iter_jsonl(benchmark_dir / "counterfactual.jsonl"))
    test_records = {record["sample_id"]: record for record in iter_jsonl(benchmark_dir / "test_iid.jsonl")}

    assert metadata["splits"]["train"]["num_samples"] == 64
    assert metadata["diagnostics"]["holdout_leakage"]["train"]["heldout_pair_uses"] == 0
    assert oracle_query_answers(train_records[0]) == [event["answer_value"] for event in train_records[0]["query_events"]]

    for counterfactual in counterfactual_records:
        source = test_records[counterfactual["counterfactual_of"]]
        changed_writes = [
            index
            for index, (source_write, cf_write) in enumerate(zip(source["writes"], counterfactual["writes"], strict=True))
            if source_write["value"] != cf_write["value"]
        ]
        assert len(changed_writes) == 1
        assert any(
            source_event["answer_value"] != cf_event["answer_value"]
            for source_event, cf_event in zip(source["query_events"], counterfactual["query_events"], strict=True)
        )
