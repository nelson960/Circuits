from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from circut.config import BenchmarkSpec, SplitSpec
from circut.io import iter_jsonl, write_json, write_jsonl
from circut.vocab import Vocabulary


WRITE_TOKENS = 5
QUERY_SUFFIX_TOKENS = 7


@dataclass(frozen=True)
class SymbolicKVRecord:
    sample_id: str
    split: str
    tokens: list[str]
    token_ids: list[int]
    axes: dict[str, int]
    query_key: str
    answer_value: str
    answer_token_index: int
    query_positions: dict[str, int]
    support_positions: dict[str, int]
    writes: list[dict[str, Any]]
    answer_pair_type: str
    latent_program_signature: str
    exact_sequence_signature: str
    counterfactual_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "axes": self.axes,
            "query_key": self.query_key,
            "answer_value": self.answer_value,
            "answer_token_index": self.answer_token_index,
            "query_positions": self.query_positions,
            "support_positions": self.support_positions,
            "writes": self.writes,
            "answer_pair_type": self.answer_pair_type,
            "latent_program_signature": self.latent_program_signature,
            "exact_sequence_signature": self.exact_sequence_signature,
            "counterfactual_of": self.counterfactual_of,
        }


def _split_spec_items(spec: BenchmarkSpec) -> list[tuple[str, SplitSpec]]:
    return [
        ("train", spec.train),
        ("validation_iid", spec.validation_iid),
        ("test_iid", spec.test_iid),
        ("heldout_pairs", spec.heldout_pairs),
        ("structural_ood", spec.structural_ood),
    ]


def _sample_axis_value(axis: Any, rng: random.Random) -> int:
    return rng.randint(axis.min, axis.max)


def _build_axis_schedule(split_spec: SplitSpec, num_samples: int, rng: random.Random) -> list[dict[str, int]]:
    combinations: list[dict[str, int]] = []
    for num_pairs in range(split_spec.num_pairs.min, split_spec.num_pairs.max + 1):
        for distractor_count in range(split_spec.distractor_count.min, split_spec.distractor_count.max + 1):
            for overwrite_count in range(split_spec.overwrite_count.min, split_spec.overwrite_count.max + 1):
                combinations.append(
                    {
                        "num_pairs": num_pairs,
                        "distractor_count": distractor_count,
                        "overwrite_count": overwrite_count,
                    }
                )
    if not combinations:
        raise RuntimeError("Axis schedule construction produced no combinations.")
    schedule: list[dict[str, int]] = []
    while len(schedule) < num_samples:
        block = list(combinations)
        rng.shuffle(block)
        schedule.extend(block)
    return schedule[:num_samples]


def _build_holdout_pairs(vocab: Vocabulary, spec: BenchmarkSpec, rng: random.Random) -> set[tuple[str, str]]:
    all_pairs = [(key, value) for key in vocab.key_tokens for value in vocab.value_tokens]
    rng.shuffle(all_pairs)
    num_holdout = max(1, int(len(all_pairs) * spec.holdout_answer_pair_fraction))
    holdout_pairs = set(all_pairs[:num_holdout])
    if len(holdout_pairs) != num_holdout:
        raise RuntimeError("Holdout pair construction produced duplicates.")
    return holdout_pairs


def _validate_spec(spec: BenchmarkSpec, vocab: Vocabulary) -> None:
    for split_name, split_spec in _split_spec_items(spec):
        if split_spec.num_pairs.max > spec.num_keys:
            raise ValueError(f"{split_name} num_pairs.max exceeds num_keys.")
        max_total_writes = split_spec.num_pairs.max + split_spec.overwrite_count.max + split_spec.distractor_count.max
        if max_total_writes > spec.num_values:
            raise ValueError(
                f"{split_name} requires {max_total_writes} distinct values, which exceeds num_values={spec.num_values}."
            )
    if spec.counterfactual.source_split not in {"train", "validation_iid", "test_iid", "heldout_pairs", "structural_ood"}:
        raise ValueError(f"Unsupported counterfactual source split: {spec.counterfactual.source_split}")
    if spec.counterfactual.num_samples <= 0:
        raise ValueError("counterfactual.num_samples must be positive.")
    if len(vocab.value_tokens) != spec.num_values:
        raise RuntimeError("Vocabulary and benchmark spec disagree on num_values.")


def _choose_value(
    *,
    key: str,
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
    used_values: set[str],
    require_holdout: bool,
    rng: random.Random,
) -> str:
    candidates = [
        value
        for value in vocab.value_tokens
        if value not in used_values and (((key, value) in holdout_pairs) == require_holdout)
    ]
    if not candidates:
        raise RuntimeError(
            f"No available values for key={key}, require_holdout={require_holdout}, used_values={len(used_values)}."
        )
    return rng.choice(candidates)


def _hash_signature(parts: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update("||".join(parts).encode("utf-8"))
    return digest.hexdigest()


def _assemble_record(
    *,
    sample_id: str,
    split_name: str,
    query_key: str,
    writes: list[dict[str, str | int]],
    vocab: Vocabulary,
    answer_pair_type: str,
    counterfactual_of: str | None = None,
) -> SymbolicKVRecord:
    tokens = ["<bos>"]
    positioned_writes: list[dict[str, Any]] = []
    for write in writes:
        statement_start = len(tokens)
        tokens.extend(["SET", str(write["key"]), "=", str(write["value"]), ";"])
        positioned_writes.append(
            {
                "write_index": int(write["write_index"]),
                "phase": str(write["phase"]),
                "key": str(write["key"]),
                "value": str(write["value"]),
                "positions": {
                    "statement_start": statement_start,
                    "key": statement_start + 1,
                    "equals": statement_start + 2,
                    "value": statement_start + 3,
                    "statement_end": statement_start + 4,
                },
            }
        )

    support_write = None
    for write in positioned_writes:
        if write["key"] == query_key:
            support_write = write
    if support_write is None:
        raise RuntimeError(f"Failed to identify support write for sample {sample_id}.")

    answer_value = str(support_write["value"])
    query_start = len(tokens)
    tokens.extend(["QRY", query_key, ";", "ANS", answer_value, ";", "<eos>"])
    query_positions = {
        "query_start": query_start,
        "query_key": query_start + 1,
        "query_end": query_start + 2,
        "answer_marker": query_start + 3,
        "answer_value": query_start + 4,
        "answer_end": query_start + 5,
        "eos": query_start + 6,
    }
    answer_token_index = query_positions["answer_value"]
    axes = {
        "num_pairs": len({write["key"] for write in positioned_writes}),
        "overwrite_count": sum(1 for write in positioned_writes if write["key"] == query_key) - 1,
        "distractor_count": sum(1 for write in positioned_writes if write["phase"] == "distractor"),
        "context_tokens": len(tokens),
        "total_writes": len(positioned_writes),
    }
    exact_sequence_signature = " ".join(tokens)
    latent_parts = [query_key, answer_value]
    latent_parts.extend(f'{write["key"]}:{write["value"]}:{write["phase"]}' for write in positioned_writes)
    latent_program_signature = _hash_signature(latent_parts)
    return SymbolicKVRecord(
        sample_id=sample_id,
        split=split_name,
        tokens=tokens,
        token_ids=vocab.encode(tokens),
        axes=axes,
        query_key=query_key,
        answer_value=answer_value,
        answer_token_index=answer_token_index,
        query_positions=query_positions,
        support_positions=dict(support_write["positions"]),
        writes=positioned_writes,
        answer_pair_type=answer_pair_type,
        latent_program_signature=latent_program_signature,
        exact_sequence_signature=exact_sequence_signature,
        counterfactual_of=counterfactual_of,
    )


def _sample_record(
    *,
    split_name: str,
    axes: dict[str, int],
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
    rng: random.Random,
    sample_id: str,
) -> SymbolicKVRecord:
    num_pairs = int(axes["num_pairs"])
    overwrite_count = int(axes["overwrite_count"])
    distractor_count = int(axes["distractor_count"])
    total_writes = num_pairs + overwrite_count + distractor_count
    if total_writes > len(vocab.value_tokens):
        raise RuntimeError("Distinct-value design violated by sampled axes.")

    context_keys = rng.sample(vocab.key_tokens, num_pairs)
    query_key = rng.choice(context_keys)
    non_query_keys = [key for key in context_keys if key != query_key]
    if not non_query_keys:
        raise RuntimeError("Expected at least one non-query key.")

    require_holdout_answer = split_name == "heldout_pairs"
    used_values: set[str] = set()

    final_query_value = _choose_value(
        key=query_key,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        used_values=used_values,
        require_holdout=require_holdout_answer,
        rng=rng,
    )
    used_values.add(final_query_value)

    prefix_writes: list[dict[str, str | int]] = []
    for overwrite_index in range(overwrite_count):
        value = _choose_value(
            key=query_key,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
            used_values=used_values,
            require_holdout=False,
            rng=rng,
        )
        used_values.add(value)
        prefix_writes.append(
            {
                "write_index": overwrite_index,
                "key": query_key,
                "value": value,
                "phase": "core",
            }
        )

    for key in non_query_keys:
        value = _choose_value(
            key=key,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
            used_values=used_values,
            require_holdout=False,
            rng=rng,
        )
        used_values.add(value)
        prefix_writes.append(
            {
                "write_index": len(prefix_writes),
                "key": key,
                "value": value,
                "phase": "core",
            }
        )

    rng.shuffle(prefix_writes)
    writes: list[dict[str, str | int]] = []
    for index, write in enumerate(prefix_writes):
        writes.append(
            {
                "write_index": index,
                "key": str(write["key"]),
                "value": str(write["value"]),
                "phase": str(write["phase"]),
            }
        )

    writes.append(
        {
            "write_index": len(writes),
            "key": query_key,
            "value": final_query_value,
            "phase": "core",
        }
    )

    for _ in range(distractor_count):
        distractor_key = rng.choice(non_query_keys)
        distractor_value = _choose_value(
            key=distractor_key,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
            used_values=used_values,
            require_holdout=False,
            rng=rng,
        )
        used_values.add(distractor_value)
        writes.append(
            {
                "write_index": len(writes),
                "key": distractor_key,
                "value": distractor_value,
                "phase": "distractor",
            }
        )

    expected_total_writes = num_pairs + overwrite_count + distractor_count
    if len(writes) != expected_total_writes:
        raise RuntimeError(f"Write count mismatch: {len(writes)} vs {expected_total_writes}")

    return _assemble_record(
        sample_id=sample_id,
        split_name=split_name,
        query_key=query_key,
        writes=writes,
        vocab=vocab,
        answer_pair_type="heldout" if require_holdout_answer else "seen",
    )


def oracle_answer(record: dict[str, Any]) -> str:
    query_key = str(record["query_key"])
    latest_value = None
    for write in record["writes"]:
        if write["key"] == query_key:
            latest_value = str(write["value"])
    if latest_value is None:
        raise RuntimeError(f"Sample {record['sample_id']} does not contain a query key write.")
    return latest_value


def _heuristic_predictions(record: dict[str, Any]) -> dict[str, str]:
    values = [str(write["value"]) for write in record["writes"]]
    query_key = str(record["query_key"])
    first_query_value = None
    for write in record["writes"]:
        if write["key"] == query_key:
            first_query_value = str(write["value"])
            break
    if first_query_value is None:
        raise RuntimeError(f"Sample {record['sample_id']} has no query-key write.")
    frequencies = Counter(values)
    most_frequent_value = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return {
        "last_write_value": values[-1],
        "first_query_value": first_query_value,
        "most_frequent_value": most_frequent_value,
    }


def _compute_heuristic_report(records_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for split_name, records in records_by_split.items():
        split_report: dict[str, float] = {}
        for heuristic_name in ["last_write_value", "first_query_value", "most_frequent_value"]:
            correct = 0
            for record in records:
                prediction = _heuristic_predictions(record)[heuristic_name]
                if prediction == record["answer_value"]:
                    correct += 1
            split_report[heuristic_name] = correct / len(records)
        report[split_name] = split_report
    return report


def _compute_overlap_report(records_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    exact_signatures = {
        split_name: {record["exact_sequence_signature"] for record in records}
        for split_name, records in records_by_split.items()
    }
    latent_signatures = {
        split_name: {record["latent_program_signature"] for record in records}
        for split_name, records in records_by_split.items()
    }
    split_names = list(records_by_split)
    report: dict[str, Any] = {"exact_sequence": {}, "latent_program": {}}
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            key = f"{left_name}__{right_name}"
            report["exact_sequence"][key] = len(exact_signatures[left_name] & exact_signatures[right_name])
            report["latent_program"][key] = len(latent_signatures[left_name] & latent_signatures[right_name])
    return report


def _compute_holdout_leakage_report(
    *,
    records_by_split: dict[str, list[dict[str, Any]]],
    holdout_pairs: set[tuple[str, str]],
) -> dict[str, Any]:
    leakage: dict[str, Any] = {}
    for split_name, records in records_by_split.items():
        heldout_pair_uses = 0
        for record in records:
            for write in record["writes"]:
                if (write["key"], write["value"]) in holdout_pairs:
                    heldout_pair_uses += 1
        answer_heldout_count = sum(
            1 for record in records if (record["query_key"], record["answer_value"]) in holdout_pairs
        )
        leakage[split_name] = {
            "heldout_pair_uses": heldout_pair_uses,
            "answer_heldout_count": answer_heldout_count,
        }
    return leakage


def _enforce_checks(
    *,
    records_by_split: dict[str, list[dict[str, Any]]],
    holdout_pairs: set[tuple[str, str]],
    heuristic_max_accuracy: float,
) -> dict[str, Any]:
    for split_name, records in records_by_split.items():
        sample_ids = {record["sample_id"] for record in records}
        if len(sample_ids) != len(records):
            raise RuntimeError(f"Duplicate sample ids detected in split {split_name}.")
        for record in records:
            oracle = oracle_answer(record)
            if oracle != record["answer_value"]:
                raise RuntimeError(f"Oracle mismatch in sample {record['sample_id']}: {oracle} vs {record['answer_value']}")
            if record["tokens"][record["answer_token_index"]] != record["answer_value"]:
                raise RuntimeError(f"Answer token index mismatch in sample {record['sample_id']}.")
            if record["support_positions"]["value"] >= record["answer_token_index"]:
                raise RuntimeError(f"Support value position must precede answer token in sample {record['sample_id']}.")
        signatures = [record["latent_program_signature"] for record in records]
        if len(set(signatures)) != len(signatures):
            raise RuntimeError(f"Duplicate latent programs detected in split {split_name}.")

    heuristic_report = _compute_heuristic_report(records_by_split)
    for split_name, split_report in heuristic_report.items():
        strongest = max(split_report.values())
        if strongest > heuristic_max_accuracy:
            raise RuntimeError(
                f"Heuristic baseline too strong in split {split_name}: {strongest:.4f} > {heuristic_max_accuracy:.4f}"
            )

    overlap_report = _compute_overlap_report(records_by_split)
    if any(overlap_report["exact_sequence"].values()):
        raise RuntimeError(f"Exact sequence overlap detected across splits: {overlap_report['exact_sequence']}")
    if any(overlap_report["latent_program"].values()):
        raise RuntimeError(f"Latent program overlap detected across splits: {overlap_report['latent_program']}")

    leakage_report = _compute_holdout_leakage_report(records_by_split=records_by_split, holdout_pairs=holdout_pairs)
    for split_name, split_report in leakage_report.items():
        if split_name == "heldout_pairs":
            if split_report["answer_heldout_count"] != len(records_by_split[split_name]):
                raise RuntimeError("Heldout split contains non-heldout answers.")
        elif split_report["heldout_pair_uses"] != 0 or split_report["answer_heldout_count"] != 0:
            raise RuntimeError(f"Holdout leakage detected in split {split_name}: {split_report}")

    return {
        "heuristics": heuristic_report,
        "overlap": overlap_report,
        "holdout_leakage": leakage_report,
    }


def _generate_counterfactual_records(
    *,
    source_records: list[dict[str, Any]],
    num_samples: int,
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
    rng: random.Random,
    used_signatures: set[str],
) -> list[dict[str, Any]]:
    if num_samples > len(source_records):
        raise ValueError("Counterfactual num_samples cannot exceed source split size.")
    chosen_sources = rng.sample(source_records, num_samples)
    counterfactuals: list[dict[str, Any]] = []
    for index, source in enumerate(chosen_sources):
        query_key = str(source["query_key"])
        used_values = {str(write["value"]) for write in source["writes"]}
        support_write_index = max(
            int(write["write_index"]) for write in source["writes"] if str(write["key"]) == query_key
        )
        replacement_value = _choose_value(
            key=query_key,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
            used_values=used_values,
            require_holdout=False,
            rng=rng,
        )
        writes: list[dict[str, str | int]] = []
        support_replaced = False
        for write in source["writes"]:
            updated_write = {
                "write_index": int(write["write_index"]),
                "key": str(write["key"]),
                "value": str(write["value"]),
                "phase": str(write["phase"]),
            }
            if int(updated_write["write_index"]) == support_write_index:
                updated_write["value"] = replacement_value
                support_replaced = True
            writes.append(updated_write)
        if not support_replaced:
            raise RuntimeError(f"Counterfactual source missing support write: {source['sample_id']}")

        record = _assemble_record(
            sample_id=f"counterfactual_{index:06d}",
            split_name="counterfactual",
            query_key=query_key,
            writes=writes,
            vocab=vocab,
            answer_pair_type="seen",
            counterfactual_of=str(source["sample_id"]),
        )
        if record.latent_program_signature in used_signatures:
            raise RuntimeError(f"Counterfactual signature collided with existing sample: {record.sample_id}")
        used_signatures.add(record.latent_program_signature)
        counterfactuals.append(record.to_dict())
    return counterfactuals


def generate_symbolic_kv_benchmark(config_path: Path, *, overwrite: bool = False) -> Path:
    spec = BenchmarkSpec.from_path(config_path)
    vocab = Vocabulary.build(spec.num_keys, spec.num_values)
    _validate_spec(spec, vocab)

    output_dir = spec.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(spec.seed)
    holdout_pairs = _build_holdout_pairs(vocab, spec, rng)

    records_by_split: dict[str, list[dict[str, Any]]] = {}
    used_signatures: set[str] = set()

    for split_name, split_spec in _split_spec_items(spec):
        split_records: list[dict[str, Any]] = []
        axis_schedule = _build_axis_schedule(split_spec, split_spec.num_samples, rng)
        max_attempts = split_spec.num_samples * 100
        attempts = 0
        while len(split_records) < split_spec.num_samples:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Exceeded max attempts while generating split {split_name}.")
            record = _sample_record(
                split_name=split_name,
                axes=axis_schedule[len(split_records)],
                vocab=vocab,
                holdout_pairs=holdout_pairs,
                rng=rng,
                sample_id=f"{split_name}_{len(split_records):06d}",
            )
            if record.latent_program_signature in used_signatures:
                continue
            used_signatures.add(record.latent_program_signature)
            split_records.append(record.to_dict())
        records_by_split[split_name] = split_records

    source_records = records_by_split[spec.counterfactual.source_split]
    records_by_split["counterfactual"] = _generate_counterfactual_records(
        source_records=source_records,
        num_samples=spec.counterfactual.num_samples,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        rng=rng,
        used_signatures=used_signatures,
    )

    diagnostics = _enforce_checks(
        records_by_split=records_by_split,
        holdout_pairs=holdout_pairs,
        heuristic_max_accuracy=spec.heuristic_max_accuracy,
    )

    split_summaries = {
        split_name: {
            "num_samples": len(records),
            "max_context_tokens": max(record["axes"]["context_tokens"] for record in records),
            "min_context_tokens": min(record["axes"]["context_tokens"] for record in records),
        }
        for split_name, records in records_by_split.items()
    }

    for split_name, records in records_by_split.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", records)

    metadata = {
        "name": spec.name,
        "seed": spec.seed,
        "vocabulary": vocab.to_metadata(),
        "holdout_pairs": sorted(f"{key}:{value}" for key, value in holdout_pairs),
        "splits": split_summaries,
        "diagnostics": diagnostics,
        "config": {
            "num_keys": spec.num_keys,
            "num_values": spec.num_values,
            "holdout_answer_pair_fraction": spec.holdout_answer_pair_fraction,
            "heuristic_max_accuracy": spec.heuristic_max_accuracy,
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    return output_dir


class SymbolicKVDataset(Dataset[dict[str, Any]]):
    def __init__(self, benchmark_dir: Path, split_name: str) -> None:
        self.benchmark_dir = benchmark_dir
        self.split_name = split_name
        self.metadata = read_symbolic_kv_metadata(benchmark_dir)
        self.vocab = Vocabulary.from_metadata(self.metadata["vocabulary"])
        split_path = benchmark_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        self.records = list(iter_jsonl(split_path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def read_symbolic_kv_metadata(benchmark_dir: Path) -> dict[str, Any]:
    metadata_path = benchmark_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Benchmark metadata not found: {metadata_path}")
    from circut.io import read_json

    return read_json(metadata_path)


def collate_symbolic_kv(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    max_len = max(len(item["token_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    answer_token_index = torch.empty(len(batch), dtype=torch.long)
    query_key_position = torch.empty(len(batch), dtype=torch.long)
    support_value_position = torch.empty(len(batch), dtype=torch.long)
    axes: dict[str, torch.Tensor] = {}
    axis_names = sorted(batch[0]["axes"])
    for axis_name in axis_names:
        axes[axis_name] = torch.empty(len(batch), dtype=torch.long)

    for row_index, item in enumerate(batch):
        token_ids = torch.tensor(item["token_ids"], dtype=torch.long)
        seq_len = token_ids.numel()
        input_ids[row_index, :seq_len] = token_ids
        attention_mask[row_index, :seq_len] = True
        answer_token_index[row_index] = int(item["answer_token_index"])
        query_key_position[row_index] = int(item["query_positions"]["query_key"])
        support_value_position[row_index] = int(item["support_positions"]["value"])
        for axis_name in axis_names:
            axes[axis_name][row_index] = int(item["axes"][axis_name])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_token_index": answer_token_index,
        "query_key_position": query_key_position,
        "support_value_position": support_value_position,
        "axes": axes,
        "records": batch,
    }
