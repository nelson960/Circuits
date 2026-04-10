from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from circut.io import iter_jsonl, read_json, write_json, write_jsonl
from circut.vocab import Vocabulary


def _pop_required(payload: dict[str, Any], key: str, context: str) -> Any:
    if key not in payload:
        raise KeyError(f"Missing key '{key}' in {context}.")
    return payload.pop(key)


def _ensure_empty(payload: dict[str, Any], context: str) -> None:
    if payload:
        raise ValueError(f"Unexpected keys in {context}: {sorted(payload)}")


@dataclass(frozen=True)
class IntRange:
    min: int
    max: int

    def __post_init__(self) -> None:
        if self.min < 0:
            raise ValueError("IntRange min must be non-negative.")
        if self.max < self.min:
            raise ValueError("IntRange max must be >= min.")

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "IntRange":
        payload = dict(data)
        instance = cls(
            min=int(_pop_required(payload, "min", context)),
            max=int(_pop_required(payload, "max", context)),
        )
        _ensure_empty(payload, context)
        return instance


@dataclass(frozen=True)
class StreamSplitSpec:
    num_samples: int
    active_keys: IntRange
    overwrite_count: IntRange
    num_queries: IntRange
    query_lag: IntRange

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("StreamSplitSpec num_samples must be positive.")
        if self.active_keys.min < 2:
            raise ValueError("StreamSplitSpec requires at least 2 active keys.")
        if self.num_queries.min <= 0:
            raise ValueError("StreamSplitSpec requires at least one query.")
        if self.query_lag.min < 1:
            raise ValueError("StreamSplitSpec query_lag.min must be at least 1.")

    @classmethod
    def from_dict(cls, data: dict[str, Any], context: str) -> "StreamSplitSpec":
        payload = dict(data)
        instance = cls(
            num_samples=int(_pop_required(payload, "num_samples", context)),
            active_keys=IntRange.from_dict(_pop_required(payload, "active_keys", context), f"{context}.active_keys"),
            overwrite_count=IntRange.from_dict(
                _pop_required(payload, "overwrite_count", context),
                f"{context}.overwrite_count",
            ),
            num_queries=IntRange.from_dict(_pop_required(payload, "num_queries", context), f"{context}.num_queries"),
            query_lag=IntRange.from_dict(_pop_required(payload, "query_lag", context), f"{context}.query_lag"),
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
class StreamBenchmarkSpec:
    benchmark_type: str
    name: str
    output_dir: Path
    seed: int
    num_keys: int
    num_values: int
    holdout_answer_pair_fraction: float
    heuristic_max_accuracy: float
    train: StreamSplitSpec
    validation_iid: StreamSplitSpec
    test_iid: StreamSplitSpec
    heldout_pairs: StreamSplitSpec
    structural_ood: StreamSplitSpec
    counterfactual: CounterfactualSpec

    def __post_init__(self) -> None:
        if self.benchmark_type != "symbolic_kv_stream":
            raise ValueError(f"Unsupported benchmark_type: {self.benchmark_type}")
        if self.num_keys < 2:
            raise ValueError("num_keys must be at least 2.")
        if self.num_values <= 0:
            raise ValueError("num_values must be positive.")
        if not 0.0 < self.holdout_answer_pair_fraction < 1.0:
            raise ValueError("holdout_answer_pair_fraction must be between 0 and 1.")
        if not 0.0 <= self.heuristic_max_accuracy <= 1.0:
            raise ValueError("heuristic_max_accuracy must be between 0 and 1.")

    @classmethod
    def from_path(cls, path: Path) -> "StreamBenchmarkSpec":
        payload = dict(read_json(path))
        instance = cls(
            benchmark_type=str(_pop_required(payload, "benchmark_type", "benchmark config")),
            name=str(_pop_required(payload, "name", "benchmark config")),
            output_dir=Path(_pop_required(payload, "output_dir", "benchmark config")),
            seed=int(_pop_required(payload, "seed", "benchmark config")),
            num_keys=int(_pop_required(payload, "num_keys", "benchmark config")),
            num_values=int(_pop_required(payload, "num_values", "benchmark config")),
            holdout_answer_pair_fraction=float(
                _pop_required(payload, "holdout_answer_pair_fraction", "benchmark config")
            ),
            heuristic_max_accuracy=float(_pop_required(payload, "heuristic_max_accuracy", "benchmark config")),
            train=StreamSplitSpec.from_dict(_pop_required(payload, "train", "benchmark config"), "benchmark config.train"),
            validation_iid=StreamSplitSpec.from_dict(
                _pop_required(payload, "validation_iid", "benchmark config"),
                "benchmark config.validation_iid",
            ),
            test_iid=StreamSplitSpec.from_dict(
                _pop_required(payload, "test_iid", "benchmark config"),
                "benchmark config.test_iid",
            ),
            heldout_pairs=StreamSplitSpec.from_dict(
                _pop_required(payload, "heldout_pairs", "benchmark config"),
                "benchmark config.heldout_pairs",
            ),
            structural_ood=StreamSplitSpec.from_dict(
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
class StreamQueryPlan:
    slot_after_write: int
    key: str
    support_write_index: int
    writes_since_support: int


def _split_spec_items(spec: StreamBenchmarkSpec) -> list[tuple[str, StreamSplitSpec]]:
    return [
        ("train", spec.train),
        ("validation_iid", spec.validation_iid),
        ("test_iid", spec.test_iid),
        ("heldout_pairs", spec.heldout_pairs),
        ("structural_ood", spec.structural_ood),
    ]


def _range_value(axis: IntRange, rng: random.Random) -> int:
    return rng.randint(axis.min, axis.max)


def _build_holdout_pairs(vocab: Vocabulary, spec: StreamBenchmarkSpec, rng: random.Random) -> set[tuple[str, str]]:
    all_pairs = [(key, value) for key in vocab.key_tokens for value in vocab.value_tokens]
    rng.shuffle(all_pairs)
    num_holdout = max(1, int(len(all_pairs) * spec.holdout_answer_pair_fraction))
    holdout_pairs = set(all_pairs[:num_holdout])
    if len(holdout_pairs) != num_holdout:
        raise RuntimeError("Holdout pair construction produced duplicates.")
    return holdout_pairs


def _validate_spec(spec: StreamBenchmarkSpec, vocab: Vocabulary) -> None:
    for split_name, split_spec in _split_spec_items(spec):
        if split_spec.active_keys.max > spec.num_keys:
            raise ValueError(f"{split_name} active_keys.max exceeds num_keys.")
        max_total_writes = split_spec.active_keys.max + split_spec.overwrite_count.max
        if max_total_writes > spec.num_values:
            raise ValueError(f"{split_name} requires more unique writes than num_values permits.")
        if split_spec.query_lag.max >= max_total_writes:
            raise ValueError(f"{split_name} query_lag.max must be smaller than the maximum number of writes.")
    if spec.counterfactual.source_split not in {"train", "validation_iid", "test_iid", "heldout_pairs", "structural_ood"}:
        raise ValueError(f"Unsupported counterfactual source split: {spec.counterfactual.source_split}")
    if len(vocab.value_tokens) != spec.num_values:
        raise RuntimeError("Vocabulary and spec disagree on num_values.")


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
            f"No value candidates for key={key}, require_holdout={require_holdout}, used_values={len(used_values)}."
        )
    return rng.choice(candidates)


def _hash_signature(parts: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update("||".join(parts).encode("utf-8"))
    return digest.hexdigest()


def _sample_write_plan(
    *,
    active_keys: list[str],
    overwrite_count: int,
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
    require_holdout: bool,
    rng: random.Random,
) -> list[dict[str, Any]]:
    used_values: set[str] = set()
    writes: list[dict[str, Any]] = []
    overwrite_tally = {key: 0 for key in active_keys}
    initial_order = list(active_keys)
    rng.shuffle(initial_order)
    for key in initial_order:
        value = _choose_value(
            key=key,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
            used_values=used_values,
            require_holdout=require_holdout,
            rng=rng,
        )
        used_values.add(value)
        writes.append({"write_index": len(writes), "key": key, "value": value})
    for _ in range(overwrite_count):
        candidate_keys = [key for key in active_keys if key != str(writes[-1]["key"])]
        if not candidate_keys:
            candidate_keys = list(active_keys)
        min_overwrites = min(overwrite_tally[key] for key in candidate_keys)
        least_overwritten = [key for key in candidate_keys if overwrite_tally[key] == min_overwrites]
        key = rng.choice(least_overwritten)
        value = _choose_value(
            key=key,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
            used_values=used_values,
            require_holdout=require_holdout,
            rng=rng,
        )
        used_values.add(value)
        writes.append({"write_index": len(writes), "key": key, "value": value})
        overwrite_tally[key] += 1
    return writes


def _candidate_query_plan(
    *,
    writes: list[dict[str, Any]],
    query_lag: IntRange,
    min_support_write_index: int,
) -> list[StreamQueryPlan]:
    current_value: dict[str, str] = {}
    last_write_index: dict[str, int] = {}
    candidates: list[StreamQueryPlan] = []
    for write_index, write in enumerate(writes):
        key = str(write["key"])
        current_value[key] = str(write["value"])
        last_write_index[key] = write_index
        for candidate_key, support_write_index in last_write_index.items():
            lag = write_index - support_write_index
            if support_write_index >= min_support_write_index and query_lag.min <= lag <= query_lag.max:
                candidates.append(
                    StreamQueryPlan(
                        slot_after_write=write_index,
                        key=candidate_key,
                        support_write_index=support_write_index,
                        writes_since_support=lag,
                    )
                )
    return candidates


def _sample_query_plan(
    *,
    writes: list[dict[str, Any]],
    num_queries: int,
    query_lag: IntRange,
    min_support_write_index: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates = _candidate_query_plan(
        writes=writes,
        query_lag=query_lag,
        min_support_write_index=min_support_write_index,
    )
    if len(candidates) < num_queries:
        raise RuntimeError(f"Not enough query candidates: required {num_queries}, found {len(candidates)}.")
    selected = rng.sample(candidates, num_queries)
    grouped: dict[int, list[StreamQueryPlan]] = defaultdict(list)
    for candidate in selected:
        grouped[candidate.slot_after_write].append(candidate)
    plan: list[dict[str, Any]] = []
    for slot_after_write in sorted(grouped):
        slot_queries = grouped[slot_after_write]
        rng.shuffle(slot_queries)
        for candidate in slot_queries:
            plan.append(
                {
                    "slot_after_write": candidate.slot_after_write,
                    "key": candidate.key,
                    "support_write_index": candidate.support_write_index,
                    "writes_since_support": candidate.writes_since_support,
                }
            )
    return plan


def _assemble_record(
    *,
    sample_id: str,
    split_name: str,
    writes: list[dict[str, Any]],
    query_plan: list[dict[str, Any]],
    vocab: Vocabulary,
    answer_pair_type: str,
    counterfactual_of: str | None = None,
    changed_write_index: int | None = None,
) -> dict[str, Any]:
    tokens = ["<bos>"]
    steps: list[dict[str, Any]] = []
    write_positions: dict[int, dict[str, int]] = {}
    query_events: list[dict[str, Any]] = []
    last_write_for_key: dict[str, int] = {}
    grouped_queries: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for query in query_plan:
        grouped_queries[int(query["slot_after_write"])].append(query)

    query_index = 0
    for write in writes:
        write_index = int(write["write_index"])
        step_index = len(steps)
        start = len(tokens)
        tokens.extend(["W", str(write["key"]), str(write["value"])])
        positions = {"op": start, "key": start + 1, "value": start + 2}
        write_positions[write_index] = positions
        steps.append(
            {
                "step_index": step_index,
                "op": "write",
                "write_index": write_index,
                "key": str(write["key"]),
                "value": str(write["value"]),
                "positions": positions,
            }
        )
        last_write_for_key[str(write["key"])] = write_index

        for query in grouped_queries.get(write_index, []):
            key = str(query["key"])
            support_write_index = int(query["support_write_index"])
            if last_write_for_key.get(key) != support_write_index:
                raise RuntimeError(
                    f"Query plan references stale support write for key={key}: "
                    f"expected {last_write_for_key.get(key)}, got {support_write_index}"
                )
            support_write = writes[support_write_index]
            answer_value = str(support_write["value"])
            query_start = len(tokens)
            tokens.extend(["R", key, answer_value])
            query_positions = {"op": query_start, "key": query_start + 1, "answer": query_start + 2}
            query_step_index = len(steps)
            steps.append(
                {
                    "step_index": query_step_index,
                    "op": "read",
                    "query_index": query_index,
                    "key": key,
                    "value": answer_value,
                    "positions": query_positions,
                    "support_write_index": support_write_index,
                }
            )
            query_events.append(
                {
                    "query_index": query_index,
                    "step_index": query_step_index,
                    "slot_after_write": write_index,
                    "key": key,
                    "answer_value": answer_value,
                    "support_write_index": support_write_index,
                    "writes_since_support": int(query["writes_since_support"]),
                    "tokens_since_support": query_positions["answer"] - write_positions[support_write_index]["value"],
                    "positions": query_positions,
                    "support_positions": dict(write_positions[support_write_index]),
                    "answer_pair_type": answer_pair_type,
                }
            )
            query_index += 1

    tokens.append("<eos>")
    if not query_events:
        raise RuntimeError(f"Record {sample_id} has no query events.")

    query_lags = [int(event["writes_since_support"]) for event in query_events]
    axes = {
        "active_keys": len({str(write["key"]) for write in writes}),
        "overwrite_count": len(writes) - len({str(write["key"]) for write in writes}),
        "num_queries": len(query_events),
        "total_writes": len(writes),
        "mean_query_lag": sum(query_lags) // len(query_lags),
        "max_query_lag": max(query_lags),
        "context_tokens": len(tokens),
    }
    exact_sequence_signature = " ".join(tokens)
    latent_parts = [f"W:{write['key']}:{write['value']}" for write in writes]
    latent_parts.extend(
        f"Q:{query['slot_after_write']}:{query['key']}:{query['support_write_index']}" for query in query_plan
    )
    if changed_write_index is not None:
        latent_parts.append(f"CF:{changed_write_index}")
    return {
        "sample_id": sample_id,
        "split": split_name,
        "tokens": tokens,
        "token_ids": vocab.encode(tokens),
        "axes": axes,
        "writes": writes,
        "query_plan": query_plan,
        "steps": steps,
        "query_events": query_events,
        "latent_program_signature": _hash_signature(latent_parts),
        "exact_sequence_signature": exact_sequence_signature,
        "counterfactual_of": counterfactual_of,
        "changed_write_index": changed_write_index,
    }


def oracle_query_answers(record: dict[str, Any]) -> list[str]:
    store: dict[str, str] = {}
    answers: list[str] = []
    for step in record["steps"]:
        if step["op"] == "write":
            store[str(step["key"])] = str(step["value"])
            continue
        if step["op"] != "read":
            raise RuntimeError(f"Unknown step op: {step['op']}")
        key = str(step["key"])
        if key not in store:
            raise RuntimeError(f"Query before initialization for key={key} in sample {record['sample_id']}")
        answers.append(store[key])
    if len(answers) != len(record["query_events"]):
        raise RuntimeError(f"Oracle/query-event mismatch in sample {record['sample_id']}")
    return answers


def _query_heuristics(record: dict[str, Any]) -> list[dict[str, str]]:
    write_history_by_key: dict[str, list[str]] = defaultdict(list)
    seen_values: list[str] = []
    heuristics: list[dict[str, str]] = []
    query_event_by_step = {int(event["step_index"]): event for event in record["query_events"]}
    for step in record["steps"]:
        if step["op"] == "write":
            key = str(step["key"])
            value = str(step["value"])
            write_history_by_key[key].append(value)
            seen_values.append(value)
            continue
        query = query_event_by_step[int(step["step_index"])]
        key = str(query["key"])
        if key not in write_history_by_key:
            raise RuntimeError(f"Missing write history for query key {key} in sample {record['sample_id']}")
        most_frequent_value = sorted(Counter(seen_values).items(), key=lambda item: (-item[1], item[0]))[0][0]
        heuristics.append(
            {
                "last_value_before_query": seen_values[-1],
                "first_value_for_key": write_history_by_key[key][0],
                "most_frequent_value_before_query": most_frequent_value,
            }
        )
    return heuristics


def _compute_heuristic_report(records_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    heuristic_names = ["last_value_before_query", "first_value_for_key", "most_frequent_value_before_query"]
    for split_name, records in records_by_split.items():
        correct_by_heuristic = {name: 0 for name in heuristic_names}
        total_queries = 0
        for record in records:
            heuristics = _query_heuristics(record)
            total_queries += len(heuristics)
            for heuristic, query_event in zip(heuristics, record["query_events"], strict=True):
                for heuristic_name in heuristic_names:
                    if heuristic[heuristic_name] == query_event["answer_value"]:
                        correct_by_heuristic[heuristic_name] += 1
        report[split_name] = {
            heuristic_name: correct_by_heuristic[heuristic_name] / total_queries for heuristic_name in heuristic_names
        }
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
    report: dict[str, Any] = {}
    for split_name, records in records_by_split.items():
        heldout_pair_uses = 0
        answer_heldout_count = 0
        total_query_events = 0
        for record in records:
            for write in record["writes"]:
                if (str(write["key"]), str(write["value"])) in holdout_pairs:
                    heldout_pair_uses += 1
            for query_event in record["query_events"]:
                total_query_events += 1
                if (str(query_event["key"]), str(query_event["answer_value"])) in holdout_pairs:
                    answer_heldout_count += 1
        report[split_name] = {
            "heldout_pair_uses": heldout_pair_uses,
            "answer_heldout_count": answer_heldout_count,
            "total_query_events": total_query_events,
        }
    return report


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
        signatures = [record["latent_program_signature"] for record in records]
        if len(set(signatures)) != len(signatures):
            raise RuntimeError(f"Duplicate latent programs detected in split {split_name}.")
        for record in records:
            oracle_answers = oracle_query_answers(record)
            query_answers = [event["answer_value"] for event in record["query_events"]]
            if oracle_answers != query_answers:
                raise RuntimeError(f"Oracle mismatch in sample {record['sample_id']}.")
            for event in record["query_events"]:
                if int(event["support_positions"]["value"]) >= int(event["positions"]["answer"]):
                    raise RuntimeError(f"Support must precede query answer in sample {record['sample_id']}.")

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
            if split_report["answer_heldout_count"] != split_report["total_query_events"]:
                raise RuntimeError("Heldout split contains non-heldout query answers.")
        elif split_report["heldout_pair_uses"] != 0 or split_report["answer_heldout_count"] != 0:
            raise RuntimeError(f"Holdout leakage detected in split {split_name}: {split_report}")

    return {
        "heuristics": heuristic_report,
        "overlap": overlap_report,
        "holdout_leakage": leakage_report,
    }


def _sample_record(
    *,
    split_name: str,
    split_spec: StreamSplitSpec,
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
    rng: random.Random,
    sample_id: str,
) -> dict[str, Any]:
    active_key_count = _range_value(split_spec.active_keys, rng)
    overwrite_count = _range_value(split_spec.overwrite_count, rng)
    num_queries = _range_value(split_spec.num_queries, rng)
    active_keys = rng.sample(vocab.key_tokens, active_key_count)
    require_holdout = split_name == "heldout_pairs"

    max_attempts = 256
    for _ in range(max_attempts):
        writes = _sample_write_plan(
            active_keys=active_keys,
            overwrite_count=overwrite_count,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
            require_holdout=require_holdout,
            rng=rng,
        )
        try:
            query_plan = _sample_query_plan(
                writes=writes,
                num_queries=num_queries,
                query_lag=split_spec.query_lag,
                min_support_write_index=active_key_count,
                rng=rng,
            )
        except RuntimeError:
            continue
        return _assemble_record(
            sample_id=sample_id,
            split_name=split_name,
            writes=writes,
            query_plan=query_plan,
            vocab=vocab,
            answer_pair_type="heldout" if require_holdout else "seen",
        )
    raise RuntimeError(f"Failed to sample a valid record for split {split_name} after {max_attempts} attempts.")


def _choose_counterfactual_write_index(record: dict[str, Any], rng: random.Random) -> int:
    supported_write_indices = sorted({int(query["support_write_index"]) for query in record["query_events"]})
    if not supported_write_indices:
        raise RuntimeError(f"No supported write index available for counterfactual in {record['sample_id']}")
    return rng.choice(supported_write_indices)


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
        changed_write_index = _choose_counterfactual_write_index(source, rng)
        writes = [
            {
                "write_index": int(write["write_index"]),
                "key": str(write["key"]),
                "value": str(write["value"]),
            }
            for write in source["writes"]
        ]
        key = str(writes[changed_write_index]["key"])
        used_values = {str(write["value"]) for write in writes}
        replacement_value = _choose_value(
            key=key,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
            used_values=used_values,
            require_holdout=False,
            rng=rng,
        )
        writes[changed_write_index]["value"] = replacement_value
        record = _assemble_record(
            sample_id=f"counterfactual_{index:06d}",
            split_name="counterfactual",
            writes=writes,
            query_plan=[
                {
                    "slot_after_write": int(query["slot_after_write"]),
                    "key": str(query["key"]),
                    "support_write_index": int(query["support_write_index"]),
                    "writes_since_support": int(query["writes_since_support"]),
                }
                for query in source["query_plan"]
            ],
            vocab=vocab,
            answer_pair_type="seen",
            counterfactual_of=str(source["sample_id"]),
            changed_write_index=changed_write_index,
        )
        if record["latent_program_signature"] in used_signatures:
            raise RuntimeError(f"Counterfactual signature collided with existing sample: {record['sample_id']}")
        used_signatures.add(record["latent_program_signature"])
        counterfactuals.append(record)
    return counterfactuals


def generate_symbolic_kv_stream_benchmark(config_path: Path, *, overwrite: bool = False) -> Path:
    spec = StreamBenchmarkSpec.from_path(config_path)
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
        max_attempts = split_spec.num_samples * 200
        attempts = 0
        while len(split_records) < split_spec.num_samples:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"Exceeded max attempts while generating split {split_name}.")
            record = _sample_record(
                split_name=split_name,
                split_spec=split_spec,
                vocab=vocab,
                holdout_pairs=holdout_pairs,
                rng=rng,
                sample_id=f"{split_name}_{len(split_records):06d}",
            )
            if record["latent_program_signature"] in used_signatures:
                continue
            used_signatures.add(record["latent_program_signature"])
            split_records.append(record)
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
            "num_query_events": sum(len(record["query_events"]) for record in records),
            "max_context_tokens": max(record["axes"]["context_tokens"] for record in records),
            "min_context_tokens": min(record["axes"]["context_tokens"] for record in records),
        }
        for split_name, records in records_by_split.items()
    }
    for split_name, records in records_by_split.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", records)
    metadata = {
        "benchmark_type": spec.benchmark_type,
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
        self.metadata = read_symbolic_kv_stream_metadata(benchmark_dir)
        self.vocab = Vocabulary.from_metadata(self.metadata["vocabulary"])
        split_path = benchmark_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        self.records = list(iter_jsonl(split_path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def read_symbolic_kv_stream_metadata(benchmark_dir: Path) -> dict[str, Any]:
    metadata_path = benchmark_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Benchmark metadata not found: {metadata_path}")
    return read_json(metadata_path)


def collate_symbolic_kv(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    max_len = max(len(item["token_ids"]) for item in batch)
    max_queries = max(len(item["query_events"]) for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    answer_token_positions = torch.full((len(batch), max_queries), -1, dtype=torch.long)
    query_key_positions = torch.full((len(batch), max_queries), -1, dtype=torch.long)
    support_value_positions = torch.full((len(batch), max_queries), -1, dtype=torch.long)
    query_mask = torch.zeros((len(batch), max_queries), dtype=torch.bool)

    axis_names = sorted(batch[0]["axes"])
    axes = {axis_name: torch.empty(len(batch), dtype=torch.long) for axis_name in axis_names}
    query_axis_names = ["writes_since_support", "tokens_since_support", "slot_after_write"]
    query_axes = {
        axis_name: torch.full((len(batch), max_queries), -1, dtype=torch.long) for axis_name in query_axis_names
    }

    for row_index, item in enumerate(batch):
        token_ids = torch.tensor(item["token_ids"], dtype=torch.long)
        seq_len = token_ids.numel()
        input_ids[row_index, :seq_len] = token_ids
        attention_mask[row_index, :seq_len] = True
        for axis_name in axis_names:
            axes[axis_name][row_index] = int(item["axes"][axis_name])
        for query_index, query_event in enumerate(item["query_events"]):
            query_mask[row_index, query_index] = True
            answer_token_positions[row_index, query_index] = int(query_event["positions"]["answer"])
            query_key_positions[row_index, query_index] = int(query_event["positions"]["key"])
            support_value_positions[row_index, query_index] = int(query_event["support_positions"]["value"])
            for axis_name in query_axis_names:
                query_axes[axis_name][row_index, query_index] = int(query_event[axis_name])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_token_positions": answer_token_positions,
        "query_key_positions": query_key_positions,
        "support_value_positions": support_value_positions,
        "query_mask": query_mask,
        "axes": axes,
        "query_axes": query_axes,
        "records": batch,
    }
