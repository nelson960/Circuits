from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.shared_feature_dynamics import _import_matplotlib
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from circuit.runtime import build_model, compute_lm_loss, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.train import _compute_learning_rate
from circuit.vocab import Vocabulary


DATASET_GEOMETRY_SCHEMA_VERSION = 1
ATTENTION_GEOMETRY_SCHEMA_VERSION = 1
PATH_LOGIT_DECOMPOSITION_SCHEMA_VERSION = 1
PROMPT_NEURON_TRACE_SCHEMA_VERSION = 1
GEOMETRY_SUBSPACE_INTERVENTION_SCHEMA_VERSION = 1
CAUSAL_VARIABLE_PATCH_SCHEMA_VERSION = 1
CANDIDATE_ROUTE_GRADIENT_SELECTION_SCHEMA_VERSION = 1
ROUTE_GRADIENT_DECOMPOSITION_SCHEMA_VERSION = 1
CHECKPOINT_UPDATE_ATTRIBUTION_SCHEMA_VERSION = 1
ATTENTION_SCORE_DELTA_DECOMPOSITION_SCHEMA_VERSION = 1
ATTENTION_SCORE_UPDATE_ATTRIBUTION_SCHEMA_VERSION = 1
ATTENTION_RETRIEVAL_SEPARATION_UPDATE_ATTRIBUTION_SCHEMA_VERSION = 1
ATTENTION_RETRIEVAL_CHAIN_REPORT_SCHEMA_VERSION = 1
ATTENTION_DOWNSTREAM_UPDATE_ATTRIBUTION_SCHEMA_VERSION = 1
DATA_UPDATE_ATTRIBUTION_SCHEMA_VERSION = 1
ROUTE_COMPETITION_REPORT_SCHEMA_VERSION = 1
DATASET_GEOMETRY_SPLIT_ORDER = [
    "train",
    "validation_iid",
    "test_iid",
    "heldout_pairs",
    "structural_ood",
    "counterfactual",
]
ROLE_NAMES = [
    "bos",
    "support_write_op",
    "support_write_key",
    "support_write_value",
    "same_key_stale_write_op",
    "same_key_stale_write_key",
    "same_key_stale_write_value",
    "other_key_write_op",
    "other_key_write_key",
    "other_key_write_value",
    "prior_read_op",
    "prior_read_key",
    "prior_read_answer",
    "current_read_op",
    "current_read_key_self",
    "other",
]


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute mean of an empty list.")
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute std of an empty list.")
    mean = _mean(values)
    return float(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)))


def _histogram(values: list[int]) -> list[dict[str, int]]:
    counts = Counter(values)
    return [{"value": int(value), "count": int(counts[value])} for value in sorted(counts)]


def _fraction(numerator: int, denominator: int, label: str) -> float:
    if denominator <= 0:
        raise ValueError(f"Cannot compute fraction for {label}: denominator must be positive.")
    return float(numerator / denominator)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _ordered_split_names(metadata: dict[str, Any]) -> list[str]:
    splits = metadata.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("Benchmark metadata must contain a non-empty splits object.")
    known = [split_name for split_name in DATASET_GEOMETRY_SPLIT_ORDER if split_name in splits]
    extras = sorted(split_name for split_name in splits if split_name not in DATASET_GEOMETRY_SPLIT_ORDER)
    return [*known, *extras]


def _load_records_by_split(benchmark_dir: Path, split_names: list[str]) -> dict[str, list[dict[str, Any]]]:
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name in split_names:
        split_path = benchmark_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found for dataset geometry report: {split_path}")
        records = list(iter_jsonl(split_path))
        if not records:
            raise ValueError(f"Split contains no records: {split_path}")
        records_by_split[split_name] = records
    return records_by_split


def _pair_key(key: str, value: str) -> str:
    return f"{key}:{value}"


def _query_distractor_counts(record: dict[str, Any], query_event: dict[str, Any]) -> dict[str, int | bool]:
    query_step_index = int(query_event["step_index"])
    support_write_index = int(query_event["support_write_index"])
    query_key = str(query_event["key"])
    latest_write_by_other_key: dict[str, int] = {}
    same_key_stale_writes = 0
    other_key_prior_writes = 0
    support_is_immediate_previous_write = False
    answer_seen_elsewhere_prior = False
    answer_value = str(query_event["answer_value"])
    previous_write_index: int | None = None

    for step in record["steps"]:
        step_index = int(step["step_index"])
        if step_index >= query_step_index:
            break
        if step["op"] != "write":
            continue
        write_index = int(step["write_index"])
        key = str(step["key"])
        value = str(step["value"])
        previous_write_index = write_index
        if key == query_key:
            if write_index != support_write_index:
                same_key_stale_writes += 1
        else:
            other_key_prior_writes += 1
            latest_write_by_other_key[key] = write_index
            if value == answer_value:
                answer_seen_elsewhere_prior = True

    if previous_write_index is not None:
        support_is_immediate_previous_write = previous_write_index == support_write_index

    return {
        "same_key_stale_writes": same_key_stale_writes,
        "other_key_prior_writes": other_key_prior_writes,
        "other_key_current_values": len(latest_write_by_other_key),
        "support_is_immediate_previous_write": support_is_immediate_previous_write,
        "answer_seen_elsewhere_prior": answer_seen_elsewhere_prior,
    }


def _summarize_dataset_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    active_keys: list[int] = []
    overwrite_counts: list[int] = []
    num_queries: list[int] = []
    total_writes: list[int] = []
    context_tokens: list[int] = []
    query_lags: list[int] = []
    token_lags: list[int] = []
    support_slots: list[int] = []
    pair_counts: Counter[str] = Counter()
    write_pair_counts: Counter[str] = Counter()
    query_key_counts: Counter[str] = Counter()
    answer_value_counts: Counter[str] = Counter()
    answer_pair_type_counts: Counter[str] = Counter()
    same_key_stale_counts: list[int] = []
    other_key_prior_counts: list[int] = []
    other_key_current_counts: list[int] = []
    immediate_support_count = 0
    answer_seen_elsewhere_prior_count = 0
    total_queries = 0

    for record in records:
        axes = record["axes"]
        active_keys.append(int(axes["active_keys"]))
        overwrite_counts.append(int(axes["overwrite_count"]))
        num_queries.append(int(axes["num_queries"]))
        total_writes.append(int(axes["total_writes"]))
        context_tokens.append(int(axes["context_tokens"]))
        for write in record["writes"]:
            write_pair_counts[_pair_key(str(write["key"]), str(write["value"]))] += 1
        for query_event in record["query_events"]:
            query_key = str(query_event["key"])
            answer_value = str(query_event["answer_value"])
            query_key_counts[query_key] += 1
            answer_value_counts[answer_value] += 1
            pair_counts[_pair_key(query_key, answer_value)] += 1
            answer_pair_type_counts[str(query_event["answer_pair_type"])] += 1
            query_lags.append(int(query_event["writes_since_support"]))
            token_lags.append(int(query_event["tokens_since_support"]))
            support_slots.append(int(query_event["slot_after_write"]))
            distractors = _query_distractor_counts(record, query_event)
            same_key_stale_counts.append(int(distractors["same_key_stale_writes"]))
            other_key_prior_counts.append(int(distractors["other_key_prior_writes"]))
            other_key_current_counts.append(int(distractors["other_key_current_values"]))
            if bool(distractors["support_is_immediate_previous_write"]):
                immediate_support_count += 1
            if bool(distractors["answer_seen_elsewhere_prior"]):
                answer_seen_elsewhere_prior_count += 1
            total_queries += 1

    if total_queries <= 0:
        raise ValueError("Dataset geometry split summary requires at least one query event.")

    return {
        "num_records": len(records),
        "num_query_events": total_queries,
        "axes": {
            "active_keys": {
                "mean": _mean([float(value) for value in active_keys]),
                "std": _std([float(value) for value in active_keys]),
                "histogram": _histogram(active_keys),
            },
            "overwrite_count": {
                "mean": _mean([float(value) for value in overwrite_counts]),
                "std": _std([float(value) for value in overwrite_counts]),
                "histogram": _histogram(overwrite_counts),
            },
            "num_queries": {
                "mean": _mean([float(value) for value in num_queries]),
                "std": _std([float(value) for value in num_queries]),
                "histogram": _histogram(num_queries),
            },
            "total_writes": {
                "mean": _mean([float(value) for value in total_writes]),
                "std": _std([float(value) for value in total_writes]),
                "histogram": _histogram(total_writes),
            },
            "context_tokens": {
                "mean": _mean([float(value) for value in context_tokens]),
                "std": _std([float(value) for value in context_tokens]),
                "histogram": _histogram(context_tokens),
            },
        },
        "query_geometry": {
            "writes_since_support": {
                "mean": _mean([float(value) for value in query_lags]),
                "std": _std([float(value) for value in query_lags]),
                "histogram": _histogram(query_lags),
            },
            "tokens_since_support": {
                "mean": _mean([float(value) for value in token_lags]),
                "std": _std([float(value) for value in token_lags]),
                "histogram": _histogram(token_lags),
            },
            "slot_after_write": {
                "mean": _mean([float(value) for value in support_slots]),
                "std": _std([float(value) for value in support_slots]),
                "histogram": _histogram(support_slots),
            },
        },
        "distractor_geometry": {
            "same_key_stale_writes_mean": _mean([float(value) for value in same_key_stale_counts]),
            "other_key_prior_writes_mean": _mean([float(value) for value in other_key_prior_counts]),
            "other_key_current_values_mean": _mean([float(value) for value in other_key_current_counts]),
            "support_is_immediate_previous_write_fraction": _fraction(
                immediate_support_count,
                total_queries,
                "support_is_immediate_previous_write_fraction",
            ),
            "answer_seen_elsewhere_prior_fraction": _fraction(
                answer_seen_elsewhere_prior_count,
                total_queries,
                "answer_seen_elsewhere_prior_fraction",
            ),
        },
        "relation_counts": {
            "query_key_counts": dict(sorted(query_key_counts.items())),
            "answer_value_counts": dict(sorted(answer_value_counts.items())),
            "answer_pair_counts": dict(sorted(pair_counts.items())),
            "write_pair_counts": dict(sorted(write_pair_counts.items())),
            "answer_pair_type_counts": dict(sorted(answer_pair_type_counts.items())),
        },
    }


def _pair_overlap_report(split_pair_sets: dict[str, set[str]]) -> dict[str, int]:
    split_names = sorted(split_pair_sets)
    overlaps: dict[str, int] = {}
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            overlaps[f"{left_name}__{right_name}"] = len(split_pair_sets[left_name] & split_pair_sets[right_name])
    return overlaps


def _build_answer_pair_matrix(
    *,
    split_summaries: dict[str, dict[str, Any]],
    key_tokens: list[str],
    value_tokens: list[str],
) -> dict[str, list[list[int]]]:
    matrix_by_split: dict[str, list[list[int]]] = {}
    for split_name, summary in split_summaries.items():
        pair_counts = summary["relation_counts"]["answer_pair_counts"]
        rows: list[list[int]] = []
        for key in key_tokens:
            row: list[int] = []
            for value in value_tokens:
                row.append(int(pair_counts.get(_pair_key(key, value), 0)))
            rows.append(row)
        matrix_by_split[split_name] = rows
    return matrix_by_split


def _top_counter_rows(counter_payload: dict[str, int], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    return [
        {"item": item, "count": int(count)}
        for item, count in sorted(counter_payload.items(), key=lambda pair: (-int(pair[1]), pair[0]))[:limit]
    ]


def _plot_dataset_split_axes(
    *,
    split_summaries: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    _, plt = _import_matplotlib()
    split_names = list(split_summaries)
    metrics = ["active_keys", "overwrite_count", "num_queries", "total_writes", "context_tokens"]
    x_positions = list(range(len(split_names)))
    width = 0.14
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for metric_index, metric in enumerate(metrics):
        offset = (metric_index - (len(metrics) - 1) / 2) * width
        values = [float(split_summaries[split]["axes"][metric]["mean"]) for split in split_names]
        ax.bar([position + offset for position in x_positions], values, width=width, label=metric)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(split_names, rotation=20, ha="right")
    ax.set_ylabel("mean")
    ax.set_title("Dataset split geometry")
    ax.legend(loc="upper left", ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_dataset_distractors(
    *,
    split_summaries: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    _, plt = _import_matplotlib()
    split_names = list(split_summaries)
    metrics = [
        "same_key_stale_writes_mean",
        "other_key_prior_writes_mean",
        "other_key_current_values_mean",
        "support_is_immediate_previous_write_fraction",
    ]
    x_positions = list(range(len(split_names)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for metric_index, metric in enumerate(metrics):
        offset = (metric_index - (len(metrics) - 1) / 2) * width
        values = [float(split_summaries[split]["distractor_geometry"][metric]) for split in split_names]
        ax.bar([position + offset for position in x_positions], values, width=width, label=metric)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(split_names, rotation=20, ha="right")
    ax.set_ylabel("mean or fraction")
    ax.set_title("Query distractor geometry")
    ax.legend(loc="upper left", ncols=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_answer_pair_matrix(
    *,
    answer_pair_matrix: dict[str, list[list[int]]],
    key_tokens: list[str],
    value_tokens: list[str],
    output_path: Path,
) -> None:
    _, plt = _import_matplotlib()
    split_names = list(answer_pair_matrix)
    num_splits = len(split_names)
    fig, axes = plt.subplots(num_splits, 1, figsize=(13, max(2.4 * num_splits, 4)), squeeze=False)
    for axis_index, split_name in enumerate(split_names):
        ax = axes[axis_index][0]
        matrix = answer_pair_matrix[split_name]
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
        ax.set_title(f"{split_name} answer-pair counts")
        ax.set_ylabel("key")
        ax.set_yticks(range(len(key_tokens)))
        ax.set_yticklabels(key_tokens)
        if axis_index == num_splits - 1:
            step = max(1, len(value_tokens) // 16)
            ticks = list(range(0, len(value_tokens), step))
            ax.set_xticks(ticks)
            ax.set_xticklabels([value_tokens[index] for index in ticks], rotation=45, ha="right")
            ax.set_xlabel("value")
        else:
            ax.set_xticks([])
        fig.colorbar(image, ax=ax, fraction=0.015, pad=0.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _write_dataset_geometry_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    lines = [
        "# Dataset Geometry Report",
        "",
        f"Benchmark: `{report['benchmark_dir']}`",
        "",
        "## Relation",
        "",
        "The symbolic relation is:",
        "",
        "```text",
        "query_key(x) -> latest matching write for that key -> associated value y",
        "```",
        "",
        "The minimal algorithm is to maintain a key-indexed store, update it on every `W K V`, and emit the current store value on every `R K`.",
        "",
        "## Split Geometry",
        "",
        "| split | records | queries | active keys | writes | overwrites | query lag | support-immediate frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name, summary in report["splits"].items():
        lines.append(
            "| {split} | {records} | {queries} | {active:.3f} | {writes:.3f} | {overwrites:.3f} | {lag:.3f} | {immediate:.3f} |".format(
                split=split_name,
                records=int(summary["num_records"]),
                queries=int(summary["num_query_events"]),
                active=float(summary["axes"]["active_keys"]["mean"]),
                writes=float(summary["axes"]["total_writes"]["mean"]),
                overwrites=float(summary["axes"]["overwrite_count"]["mean"]),
                lag=float(summary["query_geometry"]["writes_since_support"]["mean"]),
                immediate=float(summary["distractor_geometry"]["support_is_immediate_previous_write_fraction"]),
            )
        )
    lines.extend(["", "## Plots", ""])
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    lines.extend(["", "## Pair Overlap", ""])
    for name, count in report["split_answer_pair_overlap"].items():
        lines.append(f"- `{name}`: `{count}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset_geometry_report(
    *,
    benchmark_dir: Path,
    output_dir: Path,
    top_k_pairs: int = 20,
) -> tuple[Path, Path, dict[str, Path]]:
    metadata = read_symbolic_kv_stream_metadata(benchmark_dir)
    benchmark_type = metadata.get("benchmark_type")
    if benchmark_type != "symbolic_kv_stream":
        raise ValueError(f"dataset-geometry-report requires symbolic_kv_stream metadata, got {benchmark_type!r}.")
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    split_names = _ordered_split_names(metadata)
    records_by_split = _load_records_by_split(benchmark_dir, split_names)

    split_summaries = {
        split_name: _summarize_dataset_split(records)
        for split_name, records in records_by_split.items()
    }
    split_answer_pair_sets = {
        split_name: set(summary["relation_counts"]["answer_pair_counts"])
        for split_name, summary in split_summaries.items()
    }
    answer_pair_matrix = _build_answer_pair_matrix(
        split_summaries=split_summaries,
        key_tokens=vocab.key_tokens,
        value_tokens=vocab.value_tokens,
    )
    top_answer_pairs = {
        split_name: _top_counter_rows(summary["relation_counts"]["answer_pair_counts"], top_k_pairs)
        for split_name, summary in split_summaries.items()
    }
    report = {
        "schema_version": DATASET_GEOMETRY_SCHEMA_VERSION,
        "benchmark_dir": str(benchmark_dir),
        "benchmark_name": metadata.get("name"),
        "vocabulary": {
            "num_tokens": len(vocab.tokens),
            "num_keys": len(vocab.key_tokens),
            "num_values": len(vocab.value_tokens),
            "key_tokens": vocab.key_tokens,
            "value_tokens": vocab.value_tokens,
        },
        "task_relation": {
            "input_object": "context containing key-value writes plus read queries",
            "target_object": "value currently bound to the read key",
            "key_identity_classes": vocab.key_tokens,
            "value_identity_classes": vocab.value_tokens,
            "relation_matrix_axes": {
                "rows": "key tokens",
                "columns": "value tokens",
            },
            "minimal_symbolic_algorithm": [
                "initialize an empty key-value store",
                "for every W K V event, set store[K] = V",
                "for every R K event, emit store[K]",
                "ignore stale writes for the queried key and all current values for other keys",
            ],
        },
        "metadata_diagnostics": metadata.get("diagnostics"),
        "splits": split_summaries,
        "split_answer_pair_overlap": _pair_overlap_report(split_answer_pair_sets),
        "answer_pair_matrix": answer_pair_matrix,
        "top_answer_pairs": top_answer_pairs,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dataset_geometry_report.json"
    markdown_path = output_dir / "dataset_geometry_report.md"
    plot_paths = {
        "split_axes": output_dir / "dataset_geometry_split_axes.svg",
        "distractors": output_dir / "dataset_geometry_distractors.svg",
        "answer_pair_matrix": output_dir / "dataset_geometry_answer_pair_matrix.svg",
    }
    write_json(report_path, report)
    _plot_dataset_split_axes(split_summaries=split_summaries, output_path=plot_paths["split_axes"])
    _plot_dataset_distractors(split_summaries=split_summaries, output_path=plot_paths["distractors"])
    _plot_answer_pair_matrix(
        answer_pair_matrix=answer_pair_matrix,
        key_tokens=vocab.key_tokens,
        value_tokens=vocab.value_tokens,
        output_path=plot_paths["answer_pair_matrix"],
    )
    _write_dataset_geometry_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    return report_path, markdown_path, plot_paths


def _checkpoint_step_from_path(path: Path) -> int:
    if path.stem == "best":
        raise ValueError("Attention geometry trace requires step checkpoints, not best.pt.")
    prefix = "step_"
    if not path.stem.startswith(prefix):
        raise ValueError(f"Checkpoint file name does not start with {prefix!r}: {path}")
    return int(path.stem[len(prefix) :])


def _resolve_checkpoint_paths(
    *,
    checkpoint_dir: Path,
    checkpoint_paths: list[Path] | None,
) -> list[Path]:
    if checkpoint_paths is not None:
        if not checkpoint_paths:
            raise ValueError("checkpoint_paths must not be empty when provided.")
        paths = list(checkpoint_paths)
    else:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        paths = sorted(checkpoint_dir.glob("step_*.pt"), key=_checkpoint_step_from_path)
    if not paths:
        raise FileNotFoundError(f"No step checkpoints found in {checkpoint_dir}")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    return sorted(paths, key=_checkpoint_step_from_path)


def _make_probe_loader(
    *,
    probe_records: list[dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
) -> DataLoader[Any]:
    if not probe_records:
        raise ValueError("probe_records must not be empty.")
    return DataLoader(
        probe_records,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_symbolic_kv(batch, pad_token_id),
    )


def _value_target_indices(answer_targets: torch.Tensor, value_token_ids: torch.Tensor) -> torch.Tensor:
    matches = (value_token_ids.unsqueeze(0) == answer_targets.unsqueeze(1)).nonzero(as_tuple=False)
    if matches.size(0) != answer_targets.size(0):
        raise RuntimeError("Failed to locate every answer target in the value-token set.")
    return matches[:, 1]


def _value_margin(logits: torch.Tensor, answer_targets: torch.Tensor, value_token_ids: torch.Tensor) -> torch.Tensor:
    value_logits = logits.index_select(dim=-1, index=value_token_ids)
    target_indices = _value_target_indices(answer_targets, value_token_ids)
    row_index = torch.arange(answer_targets.size(0), device=answer_targets.device)
    correct = value_logits[row_index, target_indices]
    masked = value_logits.clone()
    masked[row_index, target_indices] = torch.finfo(masked.dtype).min
    return correct - masked.max(dim=-1).values


def _positions_for_query(record: dict[str, Any], query_index: int, prediction_position: int) -> dict[str, Any]:
    event = record["query_events"][query_index]
    if int(event["positions"]["answer"]) - 1 != prediction_position:
        raise RuntimeError(
            f"Query metadata mismatch for {record['sample_id']} query {query_index}: "
            f"event predicts at {int(event['positions']['answer']) - 1}, metadata predicts at {prediction_position}."
        )
    support_write_index = int(event["support_write_index"])
    query_key = str(event["key"])
    support_positions = event["support_positions"]
    support_key_position = int(support_positions["key"])
    support_value_position = int(support_positions["value"])
    support_op_position = int(support_positions["op"])
    key_distractors: list[int] = []
    value_distractors: list[int] = []
    role_positions: dict[str, list[int]] = {role: [] for role in ROLE_NAMES}
    role_positions["bos"].append(0)

    for step in record["steps"]:
        positions = step["positions"]
        if step["op"] == "write":
            write_index = int(step["write_index"])
            key = str(step["key"])
            if write_index == support_write_index:
                role_prefix = "support_write"
            elif key == query_key:
                role_prefix = "same_key_stale_write"
                key_distractors.append(int(positions["key"]))
                value_distractors.append(int(positions["value"]))
            else:
                role_prefix = "other_key_write"
                key_distractors.append(int(positions["key"]))
                value_distractors.append(int(positions["value"]))
            for position_name, role_suffix in (("op", "op"), ("key", "key"), ("value", "value")):
                position = int(positions[position_name])
                if position <= prediction_position:
                    role_positions[f"{role_prefix}_{role_suffix}"].append(position)
            continue
        if step["op"] == "read":
            is_current_read = int(step["step_index"]) == int(event["step_index"])
            if is_current_read:
                if int(positions["op"]) <= prediction_position:
                    role_positions["current_read_op"].append(int(positions["op"]))
                if int(positions["key"]) <= prediction_position:
                    role_positions["current_read_key_self"].append(int(positions["key"]))
            else:
                for position_name, role_suffix in (("op", "op"), ("key", "key"), ("answer", "answer")):
                    position = int(positions[position_name])
                    if position <= prediction_position:
                        role_positions[f"prior_read_{role_suffix}"].append(position)
            continue
        raise RuntimeError(f"Unsupported step op in record {record['sample_id']}: {step['op']}")

    if support_key_position > prediction_position:
        raise RuntimeError(f"Support key position occurs after prediction position in {record['sample_id']}.")
    if support_value_position > prediction_position:
        raise RuntimeError(f"Support value position occurs after prediction position in {record['sample_id']}.")
    if support_op_position > prediction_position:
        raise RuntimeError(f"Support op position occurs after prediction position in {record['sample_id']}.")
    if not key_distractors:
        raise RuntimeError(f"No key distractor positions for {record['sample_id']} query {query_index}.")
    if not value_distractors:
        raise RuntimeError(f"No value distractor positions for {record['sample_id']} query {query_index}.")
    return {
        "query_event": event,
        "support_key_position": support_key_position,
        "support_value_position": support_value_position,
        "support_op_position": support_op_position,
        "key_distractors": key_distractors,
        "value_distractors": value_distractors,
        "role_positions": role_positions,
    }


def _head_label(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def _top_token_alignments(
    *,
    direction: torch.Tensor,
    embedding_weight: torch.Tensor,
    vocab: Vocabulary,
    top_k: int,
) -> list[dict[str, Any]]:
    scores = embedding_weight.float().matmul(direction.float())
    top_values, top_indices = torch.topk(scores.abs(), k=min(top_k, scores.numel()))
    alignments: list[dict[str, Any]] = []
    for _, token_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
        alignments.append(
            {
                "token": vocab.tokens[int(token_index)],
                "token_id": int(token_index),
                "signed_score": float(scores[int(token_index)].item()),
                "abs_score": float(scores[int(token_index)].abs().item()),
            }
        )
    return alignments


def _token_subspace(
    *,
    embedding_weight: torch.Tensor,
    token_ids: list[int],
    label: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if len(token_ids) < 2:
        raise ValueError(f"{label} subspace requires at least two token ids.")
    token_id_tensor = torch.tensor(token_ids, dtype=torch.long)
    vectors = embedding_weight.index_select(0, token_id_tensor).float()
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    rank = int(torch.linalg.matrix_rank(centered).item())
    if rank <= 0:
        raise RuntimeError(f"{label} token embeddings produced zero-rank identity subspace.")
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:rank, :].T.contiguous()
    return basis, {
        "rank": rank,
        "ambient_dim": int(embedding_weight.size(1)),
        "num_tokens": len(token_ids),
        "singular_values": [float(value) for value in singular_values[:rank].tolist()],
    }


def _subspace_alignment(direction: torch.Tensor, basis: torch.Tensor, label: str) -> float:
    direction = direction.float()
    norm = direction.norm()
    if float(norm.item()) <= 0.0:
        raise RuntimeError(f"Cannot compute subspace alignment for zero-norm direction: {label}")
    projection = basis.matmul(basis.T.matmul(direction))
    return float((projection.norm() / norm).item())


def _compute_head_svd_summary(
    *,
    model: torch.nn.Module,
    vocab: Vocabulary,
    top_k_tokens: int,
    singular_value_count: int = 5,
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[str, Any]]:
    if top_k_tokens <= 0:
        raise ValueError("top_k_tokens must be positive.")
    embedding_weight = model.token_embedding.weight.detach().float().cpu()
    key_token_ids = [vocab.token_to_id[token] for token in vocab.key_tokens]
    value_token_ids = [vocab.token_to_id[token] for token in vocab.value_tokens]
    key_basis, key_subspace_summary = _token_subspace(
        embedding_weight=embedding_weight,
        token_ids=key_token_ids,
        label="key identity",
    )
    value_basis, value_subspace_summary = _token_subspace(
        embedding_weight=embedding_weight,
        token_ids=value_token_ids,
        label="value identity",
    )
    summaries: dict[tuple[int, int], dict[str, Any]] = {}
    for layer_index, block in enumerate(model.blocks):
        n_heads = block.attn.n_heads
        head_dim = block.attn.head_dim
        q_weight = block.attn.q_proj.weight.detach().float().cpu()
        k_weight = block.attn.k_proj.weight.detach().float().cpu()
        v_weight = block.attn.v_proj.weight.detach().float().cpu()
        out_weight = block.attn.out_proj.weight.detach().float().cpu()
        for head_index in range(n_heads):
            head_slice = slice(head_index * head_dim, (head_index + 1) * head_dim)
            q_rows = q_weight[head_slice, :]
            k_rows = k_weight[head_slice, :]
            v_rows = v_weight[head_slice, :]
            out_head = out_weight[:, head_slice]
            qk_matrix = q_rows.T.matmul(k_rows)
            ov_matrix = v_rows.T.matmul(out_head.T)
            qk_u, qk_s, qk_vh = torch.linalg.svd(qk_matrix, full_matrices=False)
            ov_u, ov_s, ov_vh = torch.linalg.svd(ov_matrix, full_matrices=False)
            keep = min(singular_value_count, qk_s.numel(), ov_s.numel())
            summaries[(layer_index, head_index)] = {
                "qk_singular_values": [float(value) for value in qk_s[:keep].tolist()],
                "ov_singular_values": [float(value) for value in ov_s[:keep].tolist()],
                "subspace_alignment": {
                    "qk_query_key_identity_fraction": _subspace_alignment(
                        qk_u[:, 0],
                        key_basis,
                        "qk_query_key_identity_fraction",
                    ),
                    "qk_key_key_identity_fraction": _subspace_alignment(
                        qk_vh[0, :],
                        key_basis,
                        "qk_key_key_identity_fraction",
                    ),
                    "ov_input_value_identity_fraction": _subspace_alignment(
                        ov_u[:, 0],
                        value_basis,
                        "ov_input_value_identity_fraction",
                    ),
                    "ov_output_value_identity_fraction": _subspace_alignment(
                        ov_vh[0, :],
                        value_basis,
                        "ov_output_value_identity_fraction",
                    ),
                },
                "qk_query_direction_top_tokens": _top_token_alignments(
                    direction=qk_u[:, 0],
                    embedding_weight=embedding_weight,
                    vocab=vocab,
                    top_k=top_k_tokens,
                ),
                "qk_key_direction_top_tokens": _top_token_alignments(
                    direction=qk_vh[0, :],
                    embedding_weight=embedding_weight,
                    vocab=vocab,
                    top_k=top_k_tokens,
                ),
                "ov_input_direction_top_tokens": _top_token_alignments(
                    direction=ov_u[:, 0],
                    embedding_weight=embedding_weight,
                    vocab=vocab,
                    top_k=top_k_tokens,
                ),
                "ov_output_direction_top_tokens": _top_token_alignments(
                    direction=ov_vh[0, :],
                    embedding_weight=embedding_weight,
                    vocab=vocab,
                    top_k=top_k_tokens,
                ),
            }
    return summaries, {
        "key_identity_subspace": key_subspace_summary,
        "value_identity_subspace": value_subspace_summary,
    }


def _new_head_accumulator(layer: int, head: int) -> dict[str, Any]:
    return {
        "layer": layer,
        "head": head,
        "num_examples": 0,
        "support_key_attention_sum": 0.0,
        "support_value_attention_sum": 0.0,
        "support_op_attention_sum": 0.0,
        "key_distractor_attention_max_sum": 0.0,
        "value_distractor_attention_max_sum": 0.0,
        "support_key_qk_margin_sum": 0.0,
        "support_value_qk_margin_sum": 0.0,
        "support_key_attention_margin_sum": 0.0,
        "support_value_attention_margin_sum": 0.0,
        "attention_entropy_sum": 0.0,
        "ov_value_score_sum": 0.0,
        "ov_value_margin_sum": 0.0,
        "attended_ov_value_score_sum": 0.0,
        "attended_ov_value_margin_sum": 0.0,
        "role_attention_sums": {role: 0.0 for role in ROLE_NAMES},
    }


def _compute_attention_checkpoint_rows(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    vocab: Vocabulary,
    checkpoint_step: int,
    top_k_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    value_token_ids = torch.tensor(vocab.value_token_ids, device=batches[0]["input_ids"].device, dtype=torch.long)
    head_metrics: dict[tuple[int, int], dict[str, Any]] = {}
    total_correct = 0
    total_queries = 0
    margin_sum = 0.0
    lossless_query_count = 0

    for batch in batches:
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_attentions=True,
            return_residual_streams=True,
        )
        if outputs.attentions is None:
            raise RuntimeError("Attention geometry trace requires attention tensors.")
        if outputs.residual_streams is None:
            raise RuntimeError("Attention geometry trace requires residual streams.")

        answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        margins = _value_margin(answer_logits, answer_targets, value_token_ids)
        target_value_indices = _value_target_indices(answer_targets, value_token_ids)
        margin_sum += float(margins.sum().item())
        predictions = answer_logits.argmax(dim=-1)
        total_correct += int((predictions == answer_targets).sum().item())
        total_queries += int(answer_targets.numel())

        pre_block_states = [outputs.residual_streams["embedding"]]
        for layer_index in range(1, len(model.blocks)):
            pre_block_states.append(outputs.residual_streams[f"layer_{layer_index - 1}_post_mlp"])

        query_geometries: list[dict[str, Any]] = []
        for flat_index in range(metadata["rows"].size(0)):
            batch_row = int(metadata["rows"][flat_index].item())
            query_index = int(metadata["query_indices"][flat_index].item())
            prediction_position = int(metadata["prediction_positions"][flat_index].item())
            record = batch["records"][batch_row]
            query_geometry = _positions_for_query(record, query_index, prediction_position)
            query_geometry["answer_token_id"] = int(answer_targets[flat_index].item())
            query_geometry["flat_index"] = flat_index
            query_geometry["batch_row"] = batch_row
            query_geometry["target_value_index"] = int(target_value_indices[flat_index].item())
            query_geometry["prediction_position"] = prediction_position
            query_geometry["query_index"] = query_index
            lossless_query_count += 1
            query_geometries.append(query_geometry)

        for layer_index, block in enumerate(model.blocks):
            pre_state = pre_block_states[layer_index]
            attention_input = block.ln_1(pre_state)
            batch_size, seq_len, _ = attention_input.shape
            n_heads = block.attn.n_heads
            head_dim = block.attn.head_dim
            q = block.attn.q_proj(attention_input).view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
            k = block.attn.k_proj(attention_input).view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
            v = block.attn.v_proj(attention_input).view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
            qk_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
            attention = outputs.attentions[layer_index]
            if attention.shape != qk_scores.shape:
                raise RuntimeError(
                    f"Attention and QK score shapes disagree at layer {layer_index}: "
                    f"{tuple(attention.shape)} vs {tuple(qk_scores.shape)}"
                )
            out_weight = block.attn.out_proj.weight
            unembed = model.lm_head.weight
            for head_index in range(n_heads):
                key = (layer_index, head_index)
                if key not in head_metrics:
                    head_metrics[key] = _new_head_accumulator(layer_index, head_index)
                accumulator = head_metrics[key]
                head_slice = slice(head_index * head_dim, (head_index + 1) * head_dim)
                out_head = out_weight[:, head_slice]
                for query_geometry in query_geometries:
                    flat_index = int(query_geometry["flat_index"])
                    batch_row = int(query_geometry["batch_row"])
                    prediction_position = int(query_geometry["prediction_position"])
                    support_key_position = int(query_geometry["support_key_position"])
                    support_value_position = int(query_geometry["support_value_position"])
                    support_op_position = int(query_geometry["support_op_position"])
                    key_distractors = torch.tensor(
                        query_geometry["key_distractors"],
                        device=attention.device,
                        dtype=torch.long,
                    )
                    value_distractors = torch.tensor(
                        query_geometry["value_distractors"],
                        device=attention.device,
                        dtype=torch.long,
                    )

                    attention_row = attention[batch_row, head_index, prediction_position, :]
                    qk_row = qk_scores[batch_row, head_index, prediction_position, :]
                    support_key_attention = attention_row[support_key_position]
                    support_value_attention = attention_row[support_value_position]
                    support_op_attention = attention_row[support_op_position]
                    key_distractor_attention_max = attention_row.index_select(0, key_distractors).max()
                    value_distractor_attention_max = attention_row.index_select(0, value_distractors).max()
                    support_key_qk_margin = qk_row[support_key_position] - qk_row.index_select(0, key_distractors).max()
                    support_value_qk_margin = qk_row[support_value_position] - qk_row.index_select(0, value_distractors).max()
                    support_key_attention_margin = support_key_attention - key_distractor_attention_max
                    support_value_attention_margin = support_value_attention - value_distractor_attention_max
                    causal_probs = attention_row[: prediction_position + 1]
                    positive_probs = causal_probs[causal_probs > 0]
                    entropy = -(positive_probs * positive_probs.log()).sum()

                    support_v = v[batch_row, head_index, support_value_position, :]
                    head_write = torch.matmul(support_v, out_head.T)
                    answer_token_id = int(answer_targets[flat_index].item())
                    answer_score = torch.dot(head_write, unembed[answer_token_id])
                    value_scores = torch.matmul(head_write, unembed.index_select(0, value_token_ids).T)
                    target_index = int(query_geometry["target_value_index"])
                    masked_value_scores = value_scores.clone()
                    masked_value_scores[target_index] = torch.finfo(masked_value_scores.dtype).min
                    ov_value_margin = answer_score - masked_value_scores.max()
                    attended_ov_value_score = support_value_attention * answer_score
                    attended_ov_value_margin = support_value_attention * ov_value_margin

                    accumulator["support_key_attention_sum"] += float(support_key_attention.item())
                    accumulator["support_value_attention_sum"] += float(support_value_attention.item())
                    accumulator["support_op_attention_sum"] += float(support_op_attention.item())
                    accumulator["key_distractor_attention_max_sum"] += float(key_distractor_attention_max.item())
                    accumulator["value_distractor_attention_max_sum"] += float(value_distractor_attention_max.item())
                    accumulator["support_key_qk_margin_sum"] += float(support_key_qk_margin.item())
                    accumulator["support_value_qk_margin_sum"] += float(support_value_qk_margin.item())
                    accumulator["support_key_attention_margin_sum"] += float(support_key_attention_margin.item())
                    accumulator["support_value_attention_margin_sum"] += float(support_value_attention_margin.item())
                    accumulator["attention_entropy_sum"] += float(entropy.item())
                    accumulator["ov_value_score_sum"] += float(answer_score.item())
                    accumulator["ov_value_margin_sum"] += float(ov_value_margin.item())
                    accumulator["attended_ov_value_score_sum"] += float(attended_ov_value_score.item())
                    accumulator["attended_ov_value_margin_sum"] += float(attended_ov_value_margin.item())
                    accumulator["num_examples"] += 1
                    role_positions = query_geometry["role_positions"]
                    for role_name, positions in role_positions.items():
                        if not positions:
                            continue
                        position_tensor = torch.tensor(positions, device=attention.device, dtype=torch.long)
                        role_mass = attention_row.index_select(0, position_tensor).sum()
                        accumulator["role_attention_sums"][role_name] += float(role_mass.item())

    if total_queries <= 0:
        raise RuntimeError("Attention geometry trace collected no query targets.")
    if lossless_query_count != total_queries:
        raise RuntimeError(
            f"Query position validation count {lossless_query_count} does not match total query count {total_queries}."
        )

    svd_summaries, embedding_subspaces = _compute_head_svd_summary(
        model=model,
        vocab=vocab,
        top_k_tokens=top_k_tokens,
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(head_metrics):
        accumulator = head_metrics[key]
        num_examples = int(accumulator["num_examples"])
        if num_examples <= 0:
            raise RuntimeError(f"Head accumulator has no examples: {key}")
        role_attention = {
            role_name: float(value) / num_examples
            for role_name, value in sorted(accumulator["role_attention_sums"].items())
        }
        row = {
            "step": checkpoint_step,
            "layer": int(accumulator["layer"]),
            "head": int(accumulator["head"]),
            "head_label": _head_label(int(accumulator["layer"]), int(accumulator["head"])),
            "num_examples": num_examples,
            "support_key_attention_mean": float(accumulator["support_key_attention_sum"]) / num_examples,
            "support_value_attention_mean": float(accumulator["support_value_attention_sum"]) / num_examples,
            "support_op_attention_mean": float(accumulator["support_op_attention_sum"]) / num_examples,
            "key_distractor_attention_max_mean": float(accumulator["key_distractor_attention_max_sum"]) / num_examples,
            "value_distractor_attention_max_mean": float(accumulator["value_distractor_attention_max_sum"]) / num_examples,
            "support_key_qk_margin_mean": float(accumulator["support_key_qk_margin_sum"]) / num_examples,
            "support_value_qk_margin_mean": float(accumulator["support_value_qk_margin_sum"]) / num_examples,
            "support_key_attention_margin_mean": float(accumulator["support_key_attention_margin_sum"]) / num_examples,
            "support_value_attention_margin_mean": float(accumulator["support_value_attention_margin_sum"]) / num_examples,
            "attention_entropy_mean": float(accumulator["attention_entropy_sum"]) / num_examples,
            "ov_value_score_mean": float(accumulator["ov_value_score_sum"]) / num_examples,
            "ov_value_margin_mean": float(accumulator["ov_value_margin_sum"]) / num_examples,
            "attended_ov_value_score_mean": float(accumulator["attended_ov_value_score_sum"]) / num_examples,
            "attended_ov_value_margin_mean": float(accumulator["attended_ov_value_margin_sum"]) / num_examples,
            "qk_query_key_subspace_alignment": float(
                svd_summaries[key]["subspace_alignment"]["qk_query_key_identity_fraction"]
            ),
            "qk_key_key_subspace_alignment": float(
                svd_summaries[key]["subspace_alignment"]["qk_key_key_identity_fraction"]
            ),
            "ov_input_value_subspace_alignment": float(
                svd_summaries[key]["subspace_alignment"]["ov_input_value_identity_fraction"]
            ),
            "ov_output_value_subspace_alignment": float(
                svd_summaries[key]["subspace_alignment"]["ov_output_value_identity_fraction"]
            ),
            "role_attention_mean": role_attention,
            "svd": svd_summaries[key],
        }
        rows.append(row)
    checkpoint_summary = {
        "step": checkpoint_step,
        "num_query_events": total_queries,
        "answer_accuracy": _fraction(total_correct, total_queries, "checkpoint answer accuracy"),
        "answer_value_margin_mean": margin_sum / total_queries,
        "embedding_subspaces": embedding_subspaces,
    }
    return rows, checkpoint_summary


def _summarize_attention_rows(rows: list[dict[str, Any]], checkpoint_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty attention geometry rows.")
    final_step = max(int(row["step"]) for row in rows)
    final_rows = [row for row in rows if int(row["step"]) == final_step]
    if not final_rows:
        raise RuntimeError("Final attention geometry rows are empty.")

    def top_rows(metric: str, limit: int = 8) -> list[dict[str, Any]]:
        return [
            {
                "step": int(row["step"]),
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "head_label": str(row["head_label"]),
                metric: float(row[metric]),
            }
            for row in sorted(final_rows, key=lambda item: float(item[metric]), reverse=True)[:limit]
        ]

    first_positive_joint: list[dict[str, Any]] = []
    head_keys = sorted({(int(row["layer"]), int(row["head"])) for row in rows})
    for layer, head in head_keys:
        head_rows = sorted(
            [row for row in rows if int(row["layer"]) == layer and int(row["head"]) == head],
            key=lambda item: int(item["step"]),
        )
        for row in head_rows:
            if (
                float(row["support_value_qk_margin_mean"]) > 0.0
                and float(row["support_value_attention_margin_mean"]) > 0.0
                and float(row["attended_ov_value_margin_mean"]) > 0.0
            ):
                first_positive_joint.append(
                    {
                        "layer": layer,
                        "head": head,
                        "head_label": _head_label(layer, head),
                        "first_step": int(row["step"]),
                        "support_value_qk_margin_mean": float(row["support_value_qk_margin_mean"]),
                        "support_value_attention_margin_mean": float(row["support_value_attention_margin_mean"]),
                        "attended_ov_value_margin_mean": float(row["attended_ov_value_margin_mean"]),
                    }
                )
                break
    return {
        "num_checkpoints": len(checkpoint_rows),
        "steps": [int(row["step"]) for row in checkpoint_rows],
        "final_step": final_step,
        "top_final_heads": {
            "by_support_value_attention": top_rows("support_value_attention_mean"),
            "by_support_value_qk_margin": top_rows("support_value_qk_margin_mean"),
            "by_qk_key_subspace_alignment": top_rows("qk_key_key_subspace_alignment"),
            "by_attended_ov_value_margin": top_rows("attended_ov_value_margin_mean"),
            "by_ov_output_value_subspace_alignment": top_rows("ov_output_value_subspace_alignment"),
            "by_low_entropy": [
                {
                    "step": int(row["step"]),
                    "layer": int(row["layer"]),
                    "head": int(row["head"]),
                    "head_label": str(row["head_label"]),
                    "attention_entropy_mean": float(row["attention_entropy_mean"]),
                }
                for row in sorted(final_rows, key=lambda item: float(item["attention_entropy_mean"]))[:8]
            ],
        },
        "first_positive_joint_geometry": first_positive_joint,
    }


def _plot_attention_metric_trajectory(
    *,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    output_path: Path,
    top_k_heads: int,
) -> None:
    if top_k_heads <= 0:
        raise ValueError("top_k_heads must be positive.")
    _, plt = _import_matplotlib()
    final_step = max(int(row["step"]) for row in rows)
    final_rows = [row for row in rows if int(row["step"]) == final_step]
    selected = sorted(final_rows, key=lambda row: float(row[metric]), reverse=True)[:top_k_heads]
    selected_keys = [(int(row["layer"]), int(row["head"])) for row in selected]
    fig, ax = plt.subplots(figsize=(12, 6))
    for layer, head in selected_keys:
        head_rows = sorted(
            [row for row in rows if int(row["layer"]) == layer and int(row["head"]) == head],
            key=lambda row: int(row["step"]),
        )
        ax.plot(
            [int(row["step"]) for row in head_rows],
            [float(row[metric]) for row in head_rows],
            marker="o",
            label=_head_label(layer, head),
        )
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_role_attention_heatmap(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    _, plt = _import_matplotlib()
    final_step = max(int(row["step"]) for row in rows)
    final_rows = sorted(
        [row for row in rows if int(row["step"]) == final_step],
        key=lambda row: (int(row["layer"]), int(row["head"])),
    )
    labels = [str(row["head_label"]) for row in final_rows]
    matrix = [
        [float(row["role_attention_mean"].get(role, 0.0)) for role in ROLE_NAMES]
        for row in final_rows
    ]
    fig, ax = plt.subplots(figsize=(max(12, 0.55 * len(ROLE_NAMES)), max(5, 0.36 * len(labels))))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_title(f"Role-conditioned attention at step {final_step}")
    ax.set_ylabel("head")
    ax.set_xlabel("source role")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(ROLE_NAMES)))
    ax.set_xticklabels(ROLE_NAMES, rotation=45, ha="right")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_attention_checkpoint_summary(
    *,
    checkpoint_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    _, plt = _import_matplotlib()
    steps = [int(row["step"]) for row in checkpoint_rows]
    best_qk: list[float] = []
    best_ov: list[float] = []
    for step in steps:
        step_rows = [row for row in rows if int(row["step"]) == step]
        if not step_rows:
            raise RuntimeError(f"No attention rows found for checkpoint step {step}.")
        best_qk.append(max(float(row["support_value_qk_margin_mean"]) for row in step_rows))
        best_ov.append(max(float(row["attended_ov_value_margin_mean"]) for row in step_rows))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(steps, [float(row["answer_accuracy"]) for row in checkpoint_rows], marker="o")
    axes[0].set_ylabel("answer accuracy")
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, [float(row["answer_value_margin_mean"]) for row in checkpoint_rows], marker="o")
    axes[1].set_ylabel("answer margin")
    axes[1].grid(alpha=0.25)
    axes[2].plot(steps, best_qk, marker="o", label="best support value QK margin")
    axes[2].plot(steps, best_ov, marker="o", label="best attended OV margin")
    axes[2].axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    axes[2].set_ylabel("best head geometry")
    axes[2].set_xlabel("checkpoint step")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _write_attention_geometry_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    final_step = int(summary["final_step"])
    lines = [
        "# Attention Geometry Trace",
        "",
        f"Config: `{report['config_path']}`",
        f"Probe set: `{report['probe_set_path']}`",
        f"Final traced step: `{final_step}`",
        "",
        "## Top Final Heads",
        "",
    ]
    for metric_name, rows in summary["top_final_heads"].items():
        lines.extend([f"### {metric_name}", ""])
        lines.append("| head | value |")
        lines.append("| --- | ---: |")
        for row in rows[:8]:
            metric_keys = [key for key in row if key not in {"step", "layer", "head", "head_label"}]
            if len(metric_keys) != 1:
                raise RuntimeError(f"Unexpected top-head row shape: {row}")
            metric_key = metric_keys[0]
            lines.append(f"| `{row['head_label']}` | {float(row[metric_key]):.6f} |")
        lines.append("")
    lines.extend(["## First Positive Joint Geometry", ""])
    if summary["first_positive_joint_geometry"]:
        lines.append("| head | first step | QK value margin | attention value margin | attended OV margin |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in summary["first_positive_joint_geometry"]:
            lines.append(
                "| `{head}` | {step} | {qk:.6f} | {attn:.6f} | {ov:.6f} |".format(
                    head=row["head_label"],
                    step=int(row["first_step"]),
                    qk=float(row["support_value_qk_margin_mean"]),
                    attn=float(row["support_value_attention_margin_mean"]),
                    ov=float(row["attended_ov_value_margin_mean"]),
                )
            )
    else:
        lines.append("No head had positive support-value QK margin, support-value attention margin, and attended OV margin in the traced checkpoints.")
    lines.extend(["", "## Plots", ""])
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_attention_geometry_trace(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    top_k_tokens: int = 8,
    top_k_plot_heads: int = 6,
) -> tuple[Path, Path, Path, dict[str, Path]]:
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    model = build_model(spec.model, len(vocab.tokens), device)
    probe_loader = _make_probe_loader(
        probe_records=probe_records,
        batch_size=spec.evaluation.batch_size,
        pad_token_id=vocab.pad_token_id,
    )
    batches = [move_batch_to_device(batch, device) for batch in probe_loader]
    if not batches:
        raise RuntimeError("Attention geometry trace produced no batches.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "attention_geometry_trace_rows.jsonl"
    checkpoint_rows_path = output_dir / "attention_geometry_checkpoint_rows.jsonl"
    progress_path = output_dir / "attention_geometry_trace_progress.json"
    report_path = output_dir / "attention_geometry_trace_report.json"
    markdown_path = output_dir / "attention_geometry_trace_report.md"
    for partial_path in (rows_path, checkpoint_rows_path, progress_path):
        if partial_path.exists():
            partial_path.unlink()

    total_probe_queries = sum(len(record["query_events"]) for record in probe_records)
    print(
        "[attention-geometry-trace] "
        f"checkpoints={len(checkpoints)} records={len(probe_records)} query_events={total_probe_queries} "
        f"device={device_name}",
        flush=True,
    )

    all_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        print(
            f"[attention-geometry-trace] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        checkpoint_step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if checkpoint_step != path_step:
            raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={checkpoint_step} path={path_step}")
        rows, checkpoint_summary = _compute_attention_checkpoint_rows(
            model=model,
            batches=batches,
            vocab=vocab,
            checkpoint_step=checkpoint_step,
            top_k_tokens=top_k_tokens,
        )
        all_rows.extend(rows)
        checkpoint_rows.append(checkpoint_summary)
        for row in rows:
            append_jsonl(rows_path, row)
        append_jsonl(checkpoint_rows_path, checkpoint_summary)
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_checkpoints": checkpoint_index,
                "total_checkpoints": len(checkpoints),
                "last_completed_step": checkpoint_step,
                "rows_path": str(rows_path),
                "checkpoint_rows_path": str(checkpoint_rows_path),
            },
        )
        print(
            "[attention-geometry-trace] finished "
            f"step={checkpoint_step} answer_accuracy={checkpoint_summary['answer_accuracy']:.6f} "
            f"answer_margin={checkpoint_summary['answer_value_margin_mean']:.6f}",
            flush=True,
        )

    summary = _summarize_attention_rows(all_rows, checkpoint_rows)
    plot_paths = {
        "support_value_qk_margin": output_dir / "attention_geometry_support_value_qk_margin.svg",
        "qk_key_subspace_alignment": output_dir / "attention_geometry_qk_key_subspace_alignment.svg",
        "attended_ov_value_margin": output_dir / "attention_geometry_attended_ov_value_margin.svg",
        "ov_output_value_subspace_alignment": output_dir / "attention_geometry_ov_output_value_subspace_alignment.svg",
        "role_attention": output_dir / "attention_geometry_role_attention.svg",
        "checkpoint_summary": output_dir / "attention_geometry_checkpoint_summary.svg",
    }
    report = {
        "schema_version": ATTENTION_GEOMETRY_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "checkpoint_rows": checkpoint_rows,
        "summary": summary,
        "rows_path": str(rows_path),
        "checkpoint_rows_path": str(checkpoint_rows_path),
        "rows": all_rows,
    }
    write_json(report_path, report)
    _plot_attention_metric_trajectory(
        rows=all_rows,
        metric="support_value_qk_margin_mean",
        title="Support value QK margin by checkpoint",
        output_path=plot_paths["support_value_qk_margin"],
        top_k_heads=top_k_plot_heads,
    )
    _plot_attention_metric_trajectory(
        rows=all_rows,
        metric="qk_key_key_subspace_alignment",
        title="QK key-side alignment with key identity subspace",
        output_path=plot_paths["qk_key_subspace_alignment"],
        top_k_heads=top_k_plot_heads,
    )
    _plot_attention_metric_trajectory(
        rows=all_rows,
        metric="attended_ov_value_margin_mean",
        title="Attended OV value margin by checkpoint",
        output_path=plot_paths["attended_ov_value_margin"],
        top_k_heads=top_k_plot_heads,
    )
    _plot_attention_metric_trajectory(
        rows=all_rows,
        metric="ov_output_value_subspace_alignment",
        title="OV output alignment with value identity subspace",
        output_path=plot_paths["ov_output_value_subspace_alignment"],
        top_k_heads=top_k_plot_heads,
    )
    _plot_role_attention_heatmap(rows=all_rows, output_path=plot_paths["role_attention"])
    _plot_attention_checkpoint_summary(
        checkpoint_rows=checkpoint_rows,
        rows=all_rows,
        output_path=plot_paths["checkpoint_summary"],
    )
    _write_attention_geometry_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_checkpoints": len(checkpoints),
            "total_checkpoints": len(checkpoints),
            "last_completed_step": int(checkpoint_rows[-1]["step"]),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "rows_path": str(rows_path),
            "checkpoint_rows_path": str(checkpoint_rows_path),
        },
    )
    print(
        f"[attention-geometry-trace] complete report={report_path} rows={rows_path}",
        flush=True,
    )
    return report_path, markdown_path, rows_path, plot_paths


def _stage_order(num_layers: int) -> list[str]:
    stages = ["embedding"]
    for layer_index in range(num_layers):
        stages.append(f"layer_{layer_index}_post_attn")
        stages.append(f"layer_{layer_index}_post_mlp")
    stages.append("final_norm")
    return stages


def _new_scalar_accumulator() -> dict[str, float]:
    return {
        "sum": 0.0,
        "abs_sum": 0.0,
        "positive_count": 0.0,
        "negative_count": 0.0,
        "total": 0.0,
    }


def _accumulate_scalar(accumulator: dict[str, float], value: float) -> None:
    accumulator["sum"] += float(value)
    accumulator["abs_sum"] += abs(float(value))
    if value > 0.0:
        accumulator["positive_count"] += 1.0
    if value < 0.0:
        accumulator["negative_count"] += 1.0
    accumulator["total"] += 1.0


def _scalar_summary(accumulator: dict[str, float], prefix: str) -> dict[str, float | int]:
    total = int(accumulator["total"])
    if total <= 0:
        raise RuntimeError(f"Cannot summarize empty accumulator for {prefix}.")
    return {
        f"{prefix}_mean": float(accumulator["sum"]) / total,
        f"{prefix}_abs_mean": float(accumulator["abs_sum"]) / total,
        f"{prefix}_positive_fraction": float(accumulator["positive_count"]) / total,
        f"{prefix}_negative_fraction": float(accumulator["negative_count"]) / total,
        "num_query_events": total,
    }


def _new_behavior_accumulator() -> dict[str, float]:
    return {
        "margin_sum": 0.0,
        "correct_count": 0.0,
        "total": 0.0,
    }


def _accumulate_behavior(
    accumulator: dict[str, float],
    *,
    margin: float,
    correct: bool,
) -> None:
    accumulator["margin_sum"] += float(margin)
    if correct:
        accumulator["correct_count"] += 1.0
    accumulator["total"] += 1.0


def _behavior_summary(accumulator: dict[str, float]) -> dict[str, float | int]:
    total = int(accumulator["total"])
    if total <= 0:
        raise RuntimeError("Cannot summarize empty behavior accumulator.")
    return {
        "margin_mean": float(accumulator["margin_sum"]) / total,
        "accuracy": float(accumulator["correct_count"]) / total,
        "num_query_events": total,
    }


def _best_wrong_value_token_ids(
    *,
    logits: torch.Tensor,
    answer_targets: torch.Tensor,
    value_token_ids: torch.Tensor,
) -> torch.Tensor:
    value_logits = logits.index_select(dim=-1, index=value_token_ids)
    target_indices = _value_target_indices(answer_targets, value_token_ids)
    row_index = torch.arange(answer_targets.size(0), device=answer_targets.device)
    masked = value_logits.clone()
    masked[row_index, target_indices] = torch.finfo(masked.dtype).min
    best_wrong_indices = masked.argmax(dim=-1)
    return value_token_ids.index_select(0, best_wrong_indices)


def _margin_gradient_vectors(
    *,
    model: torch.nn.Module,
    final_residual_vectors: torch.Tensor,
    correct_token_ids: torch.Tensor,
    wrong_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = final_residual_vectors.detach().clone().requires_grad_(True)
    normalized = model.final_norm(residual)
    logits = model.lm_head(normalized)
    row_index = torch.arange(residual.size(0), device=residual.device)
    margins = logits[row_index, correct_token_ids] - logits[row_index, wrong_token_ids]
    gradients = torch.autograd.grad(margins.sum(), residual)[0]
    return gradients.detach(), margins.detach()


def _component_id(kind: str, layer: int, head: int | None = None) -> str:
    if kind == "attention_head":
        if head is None:
            raise ValueError("attention_head component ids require head.")
        return f"L{layer}H{head}"
    if kind == "mlp_block":
        return f"L{layer}MLP"
    raise ValueError(f"Unsupported component kind: {kind}")


def _component_label(row: dict[str, Any]) -> str:
    return str(row["component_id"])


def _update_component_accumulators(
    *,
    accumulators: dict[tuple[str, int, int | None, str], dict[str, float]],
    kind: str,
    layer: int,
    head: int | None,
    split_names: list[str],
    values: torch.Tensor,
) -> None:
    if len(split_names) != values.numel():
        raise RuntimeError("split_names and component attribution values have different lengths.")
    for split_name, value in zip(split_names, values.detach().float().cpu().tolist(), strict=True):
        for aggregate_split in (split_name, "__all__"):
            key = (kind, layer, head, aggregate_split)
            if key not in accumulators:
                accumulators[key] = _new_scalar_accumulator()
            _accumulate_scalar(accumulators[key], float(value))


def _update_behavior_accumulators(
    *,
    accumulators: dict[tuple[str, str], dict[str, float]],
    stage_or_scope: str,
    split_names: list[str],
    margins: torch.Tensor,
    correct: torch.Tensor,
) -> None:
    if len(split_names) != margins.numel() or len(split_names) != correct.numel():
        raise RuntimeError("split_names, margins, and correct tensors have different lengths.")
    for split_name, margin, is_correct in zip(
        split_names,
        margins.detach().float().cpu().tolist(),
        correct.detach().cpu().tolist(),
        strict=True,
    ):
        for aggregate_split in (split_name, "__all__"):
            key = (stage_or_scope, aggregate_split)
            if key not in accumulators:
                accumulators[key] = _new_behavior_accumulator()
            _accumulate_behavior(accumulators[key], margin=float(margin), correct=bool(is_correct))


def _head_contributions_for_layer(
    *,
    block: torch.nn.Module,
    pre_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[torch.Tensor]:
    batch_size, seq_len, _ = pre_state.shape
    attention_input = block.ln_1(pre_state)
    n_heads = block.attn.n_heads
    head_dim = block.attn.head_dim
    q = block.attn.q_proj(attention_input).view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    k = block.attn.k_proj(attention_input).view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    v = block.attn.v_proj(attention_input).view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=pre_state.device), diagonal=1)
    scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
    if attention_mask.shape != (batch_size, seq_len):
        raise ValueError(f"Expected attention_mask shape {(batch_size, seq_len)}, got {tuple(attention_mask.shape)}")
    key_mask = attention_mask[:, None, None, :]
    scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    head_outputs = torch.matmul(probs, v)
    out_weight = block.attn.out_proj.weight
    contributions: list[torch.Tensor] = []
    for head_index in range(n_heads):
        head_slice = slice(head_index * head_dim, (head_index + 1) * head_dim)
        out_head = out_weight[:, head_slice]
        contributions.append(torch.matmul(head_outputs[:, head_index, :, :], out_head.T))
    return contributions


def _logits_from_stage_for_path_report(
    *,
    model: torch.nn.Module,
    stage_name: str,
    residual_state: torch.Tensor,
) -> torch.Tensor:
    if stage_name == "final_norm":
        return model.lm_head(residual_state)
    return model.lm_head(model.final_norm(residual_state))


def _stage_readout_metrics(
    *,
    model: torch.nn.Module,
    stage_name: str,
    residual_state: torch.Tensor,
    rows: torch.Tensor,
    prediction_positions: torch.Tensor,
    answer_targets: torch.Tensor,
    value_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _logits_from_stage_for_path_report(
        model=model,
        stage_name=stage_name,
        residual_state=residual_state,
    )[rows, prediction_positions, :]
    margins = _value_margin(logits, answer_targets, value_token_ids)
    correct = logits.argmax(dim=-1) == answer_targets
    return margins, correct


def _summarize_component_rows(
    *,
    step: int,
    component_accumulators: dict[tuple[str, int, int | None, str], dict[str, float]],
    baseline_by_split: dict[str, dict[str, float | int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(component_accumulators, key=lambda item: (item[3], item[0], item[1], -1 if item[2] is None else item[2])):
        kind, layer, head, split_name = key
        summary = _scalar_summary(component_accumulators[key], "direct_margin_contribution")
        baseline = baseline_by_split.get(split_name)
        if baseline is None:
            raise RuntimeError(f"Missing baseline behavior for split {split_name}.")
        row: dict[str, Any] = {
            "step": step,
            "split": split_name,
            "component_type": kind,
            "component_id": _component_id(kind, layer, head),
            "layer": layer,
            **summary,
            "baseline_margin_mean": float(baseline["margin_mean"]),
            "baseline_accuracy": float(baseline["accuracy"]),
        }
        if head is not None:
            row["head"] = head
        rows.append(row)
    return rows


def _summarize_stage_rows(
    *,
    step: int,
    stage_accumulators: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_name, split_name in sorted(stage_accumulators, key=lambda item: (item[1], item[0])):
        summary = _behavior_summary(stage_accumulators[(stage_name, split_name)])
        rows.append(
            {
                "step": step,
                "split": split_name,
                "stage": stage_name,
                "readout_margin_mean": float(summary["margin_mean"]),
                "readout_accuracy": float(summary["accuracy"]),
                "num_query_events": int(summary["num_query_events"]),
            }
        )
    return rows


def _compute_path_decomposition_checkpoint(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    vocab: Vocabulary,
    checkpoint_step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    value_token_ids = torch.tensor(vocab.value_token_ids, device=batches[0]["input_ids"].device, dtype=torch.long)
    component_accumulators: dict[tuple[str, int, int | None, str], dict[str, float]] = {}
    baseline_accumulators: dict[tuple[str, str], dict[str, float]] = {}
    stage_accumulators: dict[tuple[str, str], dict[str, float]] = {}
    num_layers = len(model.blocks)

    for batch in batches:
        with torch.no_grad():
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
            )
        if outputs.residual_streams is None:
            raise RuntimeError("Path logit decomposition requires residual streams.")
        answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        rows = metadata["rows"]
        prediction_positions = metadata["prediction_positions"]
        baseline_margins = _value_margin(answer_logits, answer_targets, value_token_ids)
        baseline_correct = answer_logits.argmax(dim=-1) == answer_targets
        wrong_token_ids = _best_wrong_value_token_ids(
            logits=answer_logits,
            answer_targets=answer_targets,
            value_token_ids=value_token_ids,
        )
        final_pre_stage = f"layer_{num_layers - 1}_post_mlp"
        final_pre_vectors = outputs.residual_streams[final_pre_stage][rows, prediction_positions, :]
        margin_gradients, recomputed_margins = _margin_gradient_vectors(
            model=model,
            final_residual_vectors=final_pre_vectors,
            correct_token_ids=answer_targets,
            wrong_token_ids=wrong_token_ids,
        )
        if not torch.allclose(recomputed_margins, baseline_margins, atol=1e-4, rtol=1e-4):
            max_delta = (recomputed_margins - baseline_margins).abs().max().item()
            raise RuntimeError(f"Recomputed final-norm margins disagree with model logits: max_delta={max_delta:.6g}")

        split_names = [str(batch["records"][int(row.item())]["split"]) for row in rows]
        _update_behavior_accumulators(
            accumulators=baseline_accumulators,
            stage_or_scope="baseline",
            split_names=split_names,
            margins=baseline_margins,
            correct=baseline_correct,
        )

        for stage_name in _stage_order(num_layers):
            stage_margins, stage_correct = _stage_readout_metrics(
                model=model,
                stage_name=stage_name,
                residual_state=outputs.residual_streams[stage_name],
                rows=rows,
                prediction_positions=prediction_positions,
                answer_targets=answer_targets,
                value_token_ids=value_token_ids,
            )
            _update_behavior_accumulators(
                accumulators=stage_accumulators,
                stage_or_scope=stage_name,
                split_names=split_names,
                margins=stage_margins,
                correct=stage_correct,
            )

        pre_block_states = [outputs.residual_streams["embedding"]]
        for layer_index in range(1, num_layers):
            pre_block_states.append(outputs.residual_streams[f"layer_{layer_index - 1}_post_mlp"])
        with torch.no_grad():
            for layer_index, block in enumerate(model.blocks):
                head_contributions = _head_contributions_for_layer(
                    block=block,
                    pre_state=pre_block_states[layer_index],
                    attention_mask=batch["attention_mask"],
                )
                for head_index, contribution in enumerate(head_contributions):
                    component_vectors = contribution[rows, prediction_positions, :]
                    values = (component_vectors.float() * margin_gradients.float()).sum(dim=-1)
                    _update_component_accumulators(
                        accumulators=component_accumulators,
                        kind="attention_head",
                        layer=layer_index,
                        head=head_index,
                        split_names=split_names,
                        values=values,
                    )
                mlp_vectors = (
                    outputs.residual_streams[f"layer_{layer_index}_post_mlp"][rows, prediction_positions, :]
                    - outputs.residual_streams[f"layer_{layer_index}_post_attn"][rows, prediction_positions, :]
                )
                mlp_values = (mlp_vectors.float() * margin_gradients.float()).sum(dim=-1)
                _update_component_accumulators(
                    accumulators=component_accumulators,
                    kind="mlp_block",
                    layer=layer_index,
                    head=None,
                    split_names=split_names,
                    values=mlp_values,
                )

    baseline_by_split = {
        split_name: _behavior_summary(accumulator)
        for (_, split_name), accumulator in baseline_accumulators.items()
    }
    checkpoint_rows = [
        {
            "step": checkpoint_step,
            "split": split_name,
            "baseline_margin_mean": float(summary["margin_mean"]),
            "baseline_accuracy": float(summary["accuracy"]),
            "num_query_events": int(summary["num_query_events"]),
        }
        for split_name, summary in sorted(baseline_by_split.items())
    ]
    component_rows = _summarize_component_rows(
        step=checkpoint_step,
        component_accumulators=component_accumulators,
        baseline_by_split=baseline_by_split,
    )
    stage_rows = _summarize_stage_rows(step=checkpoint_step, stage_accumulators=stage_accumulators)
    return component_rows, stage_rows, checkpoint_rows


def _evaluate_component_ablation(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    vocab: Vocabulary,
    component_row: dict[str, Any],
    baseline_by_split: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    value_token_ids = torch.tensor(vocab.value_token_ids, device=batches[0]["input_ids"].device, dtype=torch.long)
    kind = str(component_row["component_type"])
    layer = int(component_row["layer"])
    head = int(component_row["head"]) if "head" in component_row else None
    ablated_accumulators: dict[tuple[str, str], dict[str, float]] = {}
    n_layers = len(model.blocks)
    n_heads = model.spec.n_heads
    for batch in batches:
        with torch.no_grad():
            if kind == "attention_head":
                if head is None:
                    raise RuntimeError("Head ablation row is missing head index.")
                head_mask = {
                    layer_index: torch.ones(n_heads, device=batch["input_ids"].device)
                    for layer_index in range(n_layers)
                }
                head_mask[layer][head] = 0.0
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    head_mask=head_mask,
                )
            elif kind == "mlp_block":
                mlp_mask = {layer_index: 1.0 for layer_index in range(n_layers)}
                mlp_mask[layer] = 0.0
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    mlp_mask=mlp_mask,
                )
            else:
                raise ValueError(f"Unsupported ablation component kind: {kind}")
        answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        margins = _value_margin(answer_logits, answer_targets, value_token_ids)
        correct = answer_logits.argmax(dim=-1) == answer_targets
        split_names = [str(batch["records"][int(row.item())]["split"]) for row in metadata["rows"]]
        _update_behavior_accumulators(
            accumulators=ablated_accumulators,
            stage_or_scope="ablated",
            split_names=split_names,
            margins=margins,
            correct=correct,
        )

    rows: list[dict[str, Any]] = []
    for (_, split_name), accumulator in sorted(ablated_accumulators.items()):
        ablated = _behavior_summary(accumulator)
        baseline = baseline_by_split.get(split_name)
        if baseline is None:
            raise RuntimeError(f"Missing baseline row for split {split_name} during ablation.")
        rows.append(
            {
                "step": int(component_row["step"]),
                "split": split_name,
                "component_type": kind,
                "component_id": str(component_row["component_id"]),
                "layer": layer,
                **({"head": head} if head is not None else {}),
                "direct_margin_contribution_mean": float(component_row["direct_margin_contribution_mean"]),
                "baseline_margin_mean": float(baseline["baseline_margin_mean"]),
                "ablated_margin_mean": float(ablated["margin_mean"]),
                "margin_drop": float(baseline["baseline_margin_mean"]) - float(ablated["margin_mean"]),
                "baseline_accuracy": float(baseline["baseline_accuracy"]),
                "ablated_accuracy": float(ablated["accuracy"]),
                "accuracy_drop": float(baseline["baseline_accuracy"]) - float(ablated["accuracy"]),
                "num_query_events": int(ablated["num_query_events"]),
            }
        )
    return rows


def _select_ablation_components(
    *,
    component_rows: list[dict[str, Any]],
    ablation_top_k: int,
) -> list[dict[str, Any]]:
    if ablation_top_k < 0:
        raise ValueError("ablation_top_k must be non-negative.")
    if ablation_top_k == 0:
        return []
    all_rows = [row for row in component_rows if row["split"] == "__all__"]
    positive = sorted(all_rows, key=lambda row: float(row["direct_margin_contribution_mean"]), reverse=True)[
        :ablation_top_k
    ]
    negative = sorted(all_rows, key=lambda row: float(row["direct_margin_contribution_mean"]))[:ablation_top_k]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*positive, *negative]:
        key = str(row["component_id"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
    return selected


def _summarize_path_report(
    *,
    component_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not checkpoint_rows:
        raise ValueError("Cannot summarize path report without checkpoint rows.")
    final_step = max(int(row["step"]) for row in checkpoint_rows)
    final_components = [
        row for row in component_rows if int(row["step"]) == final_step and str(row["split"]) == "__all__"
    ]
    final_stages = [row for row in stage_rows if int(row["step"]) == final_step and str(row["split"]) == "__all__"]
    top_positive = [
        row
        for row in sorted(final_components, key=lambda row: float(row["direct_margin_contribution_mean"]), reverse=True)
        if float(row["direct_margin_contribution_mean"]) > 0.0
    ][:8]
    top_negative = [
        row
        for row in sorted(final_components, key=lambda row: float(row["direct_margin_contribution_mean"]))
        if float(row["direct_margin_contribution_mean"]) < 0.0
    ][:8]
    final_ablation = [
        row for row in ablation_rows if int(row["step"]) == final_step and str(row["split"]) == "__all__"
    ]
    return {
        "num_checkpoints": len({int(row["step"]) for row in checkpoint_rows}),
        "steps": sorted({int(row["step"]) for row in checkpoint_rows}),
        "final_step": final_step,
        "top_final_positive_direct_components": top_positive,
        "top_final_negative_direct_components": top_negative,
        "final_stage_readout": final_stages,
        "final_ablation_rows": sorted(final_ablation, key=lambda row: float(row["margin_drop"]), reverse=True),
    }


def _plot_path_component_trajectory(
    *,
    component_rows: list[dict[str, Any]],
    output_path: Path,
    top_k: int,
) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    _, plt = _import_matplotlib()
    all_rows = [row for row in component_rows if row["split"] == "__all__"]
    final_step = max(int(row["step"]) for row in all_rows)
    final_rows = [row for row in all_rows if int(row["step"]) == final_step]
    selected = sorted(final_rows, key=lambda row: abs(float(row["direct_margin_contribution_mean"])), reverse=True)[:top_k]
    fig, ax = plt.subplots(figsize=(12, 6))
    for selected_row in selected:
        component_id = str(selected_row["component_id"])
        rows = sorted(
            [row for row in all_rows if str(row["component_id"]) == component_id],
            key=lambda row: int(row["step"]),
        )
        ax.plot(
            [int(row["step"]) for row in rows],
            [float(row["direct_margin_contribution_mean"]) for row in rows],
            marker="o",
            label=component_id,
        )
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title("Direct margin contribution trajectory")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("first-order direct margin contribution")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_stage_readout_trajectory(
    *,
    stage_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    _, plt = _import_matplotlib()
    all_rows = [row for row in stage_rows if row["split"] == "__all__"]
    stages = _stage_order(max(int(row["stage"].split("_")[1]) for row in all_rows if row["stage"].startswith("layer_")) + 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    for stage in stages:
        rows = sorted([row for row in all_rows if row["stage"] == stage], key=lambda row: int(row["step"]))
        if not rows:
            continue
        ax.plot(
            [int(row["step"]) for row in rows],
            [float(row["readout_margin_mean"]) for row in rows],
            marker="o",
            label=stage,
        )
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title("Exact readout margin by residual stage")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("readout margin")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_path_ablation_scatter(
    *,
    ablation_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not ablation_rows:
        return None
    _, plt = _import_matplotlib()
    rows = [row for row in ablation_rows if row["split"] == "__all__"]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        [float(row["direct_margin_contribution_mean"]) for row in rows],
        [float(row["margin_drop"]) for row in rows],
    )
    for row in rows:
        ax.annotate(str(row["component_id"]), (float(row["direct_margin_contribution_mean"]), float(row["margin_drop"])))
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title("Direct attribution vs causal ablation")
    ax.set_xlabel("direct margin contribution")
    ax.set_ylabel("actual margin drop when ablated")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_path_logit_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    lines = [
        "# Path Logit Decomposition",
        "",
        "## Calculation",
        "",
        "For each query example, define the value-margin:",
        "",
        "```text",
        "m(x, y) = logit(y | x) - max_{z in values, z != y} logit(z | x)",
        "```",
        "",
        "Let `z_final` be the final pre-layernorm residual vector at the answer prediction position. The local readout gradient is:",
        "",
        "```text",
        "g_margin = d m / d z_final",
        "```",
        "",
        "For a component residual write `r_c`, this report computes first-order direct logit attribution:",
        "",
        "```text",
        "DLA_c(x, y) = r_c(x) dot g_margin(x, y)",
        "```",
        "",
        "A component is not treated as causal from DLA alone. For selected components, the report also computes:",
        "",
        "```text",
        "ablation_drop_c = m_baseline - m_without_c",
        "```",
        "",
        "## Final Positive Direct Components",
        "",
        "| component | type | DLA mean | positive fraction |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in summary["top_final_positive_direct_components"][:8]:
        lines.append(
            "| `{component}` | {kind} | {value:.6f} | {frac:.3f} |".format(
                component=row["component_id"],
                kind=row["component_type"],
                value=float(row["direct_margin_contribution_mean"]),
                frac=float(row["direct_margin_contribution_positive_fraction"]),
            )
        )
    lines.extend(["", "## Final Negative Direct Components", ""])
    lines.append("| component | type | DLA mean | negative fraction |")
    lines.append("| --- | --- | ---: | ---: |")
    for row in summary["top_final_negative_direct_components"][:8]:
        lines.append(
            "| `{component}` | {kind} | {value:.6f} | {frac:.3f} |".format(
                component=row["component_id"],
                kind=row["component_type"],
                value=float(row["direct_margin_contribution_mean"]),
                frac=float(row["direct_margin_contribution_negative_fraction"]),
            )
        )
    if summary["final_ablation_rows"]:
        lines.extend(["", "## Final Causal Ablations", ""])
        lines.append("| component | DLA mean | margin drop | accuracy drop |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in summary["final_ablation_rows"]:
            lines.append(
                "| `{component}` | {dla:.6f} | {drop:.6f} | {acc:.6f} |".format(
                    component=row["component_id"],
                    dla=float(row["direct_margin_contribution_mean"]),
                    drop=float(row["margin_drop"]),
                    acc=float(row["accuracy_drop"]),
                )
            )
    lines.extend(["", "## Plots", ""])
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_path_logit_decomposition(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    ablation_top_k: int = 3,
    ablation_steps: list[int] | None = None,
    top_k_plot_components: int = 8,
) -> tuple[Path, Path, dict[str, Path]]:
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    resolved_ablation_steps = set(ablation_steps) if ablation_steps is not None else {int(_checkpoint_step_from_path(checkpoints[-1]))}
    model = build_model(spec.model, len(vocab.tokens), device)
    loader = _make_probe_loader(
        probe_records=probe_records,
        batch_size=spec.evaluation.batch_size,
        pad_token_id=vocab.pad_token_id,
    )
    batches = [move_batch_to_device(batch, device) for batch in loader]
    if not batches:
        raise RuntimeError("Path logit decomposition produced no batches.")

    output_dir.mkdir(parents=True, exist_ok=True)
    component_rows_path = output_dir / "path_logit_component_rows.jsonl"
    stage_rows_path = output_dir / "path_logit_stage_rows.jsonl"
    checkpoint_rows_path = output_dir / "path_logit_checkpoint_rows.jsonl"
    ablation_rows_path = output_dir / "path_logit_ablation_rows.jsonl"
    progress_path = output_dir / "path_logit_decomposition_progress.json"
    for partial_path in (component_rows_path, stage_rows_path, checkpoint_rows_path, ablation_rows_path, progress_path):
        if partial_path.exists():
            partial_path.unlink()

    print(
        "[path-logit-decomposition] "
        f"checkpoints={len(checkpoints)} records={len(probe_records)} device={device_name} "
        f"ablation_steps={sorted(resolved_ablation_steps)}",
        flush=True,
    )
    all_component_rows: list[dict[str, Any]] = []
    all_stage_rows: list[dict[str, Any]] = []
    all_checkpoint_rows: list[dict[str, Any]] = []
    all_ablation_rows: list[dict[str, Any]] = []

    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if step != path_step:
            raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={step} path={path_step}")
        print(f"[path-logit-decomposition] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}", flush=True)
        component_rows, stage_rows, checkpoint_rows = _compute_path_decomposition_checkpoint(
            model=model,
            batches=batches,
            vocab=vocab,
            checkpoint_step=step,
        )
        for row in component_rows:
            append_jsonl(component_rows_path, row)
        for row in stage_rows:
            append_jsonl(stage_rows_path, row)
        for row in checkpoint_rows:
            append_jsonl(checkpoint_rows_path, row)
        all_component_rows.extend(component_rows)
        all_stage_rows.extend(stage_rows)
        all_checkpoint_rows.extend(checkpoint_rows)

        if step in resolved_ablation_steps:
            baseline_by_split = {
                str(row["split"]): row
                for row in checkpoint_rows
            }
            selected = _select_ablation_components(component_rows=component_rows, ablation_top_k=ablation_top_k)
            for component_row in selected:
                ablation_rows = _evaluate_component_ablation(
                    model=model,
                    batches=batches,
                    vocab=vocab,
                    component_row=component_row,
                    baseline_by_split=baseline_by_split,
                )
                for row in ablation_rows:
                    append_jsonl(ablation_rows_path, row)
                all_ablation_rows.extend(ablation_rows)

        write_json(
            progress_path,
            {
                "status": "running",
                "completed_checkpoints": checkpoint_index,
                "total_checkpoints": len(checkpoints),
                "last_completed_step": step,
                "component_rows_path": str(component_rows_path),
                "stage_rows_path": str(stage_rows_path),
                "checkpoint_rows_path": str(checkpoint_rows_path),
                "ablation_rows_path": str(ablation_rows_path),
            },
        )
        all_split_row = next(row for row in checkpoint_rows if row["split"] == "__all__")
        print(
            "[path-logit-decomposition] finished "
            f"step={step} margin={all_split_row['baseline_margin_mean']:.6f} "
            f"accuracy={all_split_row['baseline_accuracy']:.6f}",
            flush=True,
        )

    summary = _summarize_path_report(
        component_rows=all_component_rows,
        stage_rows=all_stage_rows,
        checkpoint_rows=all_checkpoint_rows,
        ablation_rows=all_ablation_rows,
    )
    report_path = output_dir / "path_logit_decomposition_report.json"
    markdown_path = output_dir / "path_logit_decomposition_report.md"
    plot_paths: dict[str, Path] = {
        "component_trajectory": output_dir / "path_logit_component_trajectory.svg",
        "stage_readout": output_dir / "path_logit_stage_readout.svg",
    }
    _plot_path_component_trajectory(
        component_rows=all_component_rows,
        output_path=plot_paths["component_trajectory"],
        top_k=top_k_plot_components,
    )
    _plot_stage_readout_trajectory(stage_rows=all_stage_rows, output_path=plot_paths["stage_readout"])
    ablation_plot_path = _plot_path_ablation_scatter(
        ablation_rows=all_ablation_rows,
        output_path=output_dir / "path_logit_ablation_vs_dla.svg",
    )
    if ablation_plot_path is not None:
        plot_paths["ablation_vs_dla"] = ablation_plot_path
    report = {
        "schema_version": PATH_LOGIT_DECOMPOSITION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "calculation": {
            "margin": "logit(correct_value) - max_{wrong value token} logit(wrong_value)",
            "gradient": "g_margin = d margin / d final_pre_layernorm_residual",
            "direct_component_attribution": "component_residual_write dot g_margin",
            "causal_ablation": "baseline margin - margin after zeroing selected component",
            "scope_note": (
                "Direct attribution is a first-order direct-to-readout calculation through final layernorm. "
                "It is not treated as causal unless the ablation rows agree."
            ),
            "attention_bias_note": (
                "Attention output projection bias is not assigned to individual heads; per-head rows only decompose "
                "the head-dependent value paths."
            ),
        },
        "component_rows_path": str(component_rows_path),
        "stage_rows_path": str(stage_rows_path),
        "checkpoint_rows_path": str(checkpoint_rows_path),
        "ablation_rows_path": str(ablation_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_path_logit_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_checkpoints": len(checkpoints),
            "total_checkpoints": len(checkpoints),
            "last_completed_step": int(all_checkpoint_rows[-1]["step"]),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "component_rows_path": str(component_rows_path),
            "stage_rows_path": str(stage_rows_path),
            "checkpoint_rows_path": str(checkpoint_rows_path),
            "ablation_rows_path": str(ablation_rows_path),
        },
    )
    print(f"[path-logit-decomposition] complete report={report_path}", flush=True)
    return report_path, markdown_path, plot_paths


def _parse_neuron_id(neuron_id: str) -> tuple[int, int]:
    if not neuron_id.startswith("L") or "N" not in neuron_id:
        raise ValueError(f"Neuron id must have form L<layer>N<neuron>, got: {neuron_id}")
    layer_text, neuron_text = neuron_id[1:].split("N", maxsplit=1)
    if layer_text == "" or neuron_text == "":
        raise ValueError(f"Neuron id must have form L<layer>N<neuron>, got: {neuron_id}")
    return int(layer_text), int(neuron_text)


def _neuron_id(layer: int, neuron: int) -> str:
    return f"L{layer}N{neuron}"


def _resolve_mlp_layers(model: torch.nn.Module, mlp_layers: list[int] | None) -> list[int]:
    n_layers = len(model.blocks)
    if mlp_layers is None:
        return list(range(n_layers))
    resolved = sorted(set(int(layer) for layer in mlp_layers))
    if not resolved:
        raise ValueError("mlp_layers must not be empty when provided.")
    invalid = [layer for layer in resolved if layer < 0 or layer >= n_layers]
    if invalid:
        raise ValueError(f"Requested MLP layer(s) outside model range 0..{n_layers - 1}: {invalid}")
    return resolved


def _query_info_rows(
    *,
    batch: dict[str, Any],
    vocab: Vocabulary,
    metadata: dict[str, torch.Tensor],
    answer_targets: torch.Tensor,
    margins: torch.Tensor,
    correct: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = metadata["rows"]
    query_indices = metadata["query_indices"]
    answer_positions = metadata["answer_positions"]
    prediction_positions = metadata["prediction_positions"]
    query_key_positions = metadata["query_key_positions"]
    support_value_positions = metadata["support_value_positions"]
    if not (
        rows.numel()
        == query_indices.numel()
        == answer_targets.numel()
        == margins.numel()
        == correct.numel()
    ):
        raise RuntimeError("Query metadata tensors have inconsistent lengths.")
    decoded_targets = vocab.decode([int(token_id) for token_id in answer_targets.detach().cpu().tolist()])
    result: list[dict[str, Any]] = []
    for item_index in range(rows.numel()):
        batch_row = int(rows[item_index].item())
        query_index = int(query_indices[item_index].item())
        record = batch["records"][batch_row]
        query_event = record["query_events"][query_index]
        axes = record["axes"]
        result.append(
            {
                "split": str(record["split"]),
                "sample_id": str(record["sample_id"]),
                "query_index": query_index,
                "query_id": f"{record['split']}:{record['sample_id']}:{query_index}",
                "query_key": str(query_event["key"]),
                "answer_value": str(query_event["answer_value"]),
                "answer_token": decoded_targets[item_index],
                "answer_pair_type": str(query_event["answer_pair_type"]),
                "answer_position": int(answer_positions[item_index].item()),
                "prediction_position": int(prediction_positions[item_index].item()),
                "query_key_position": int(query_key_positions[item_index].item()),
                "support_value_position": int(support_value_positions[item_index].item()),
                "support_write_index": int(query_event["support_write_index"]),
                "writes_since_support": int(query_event["writes_since_support"]),
                "tokens_since_support": int(query_event["tokens_since_support"]),
                "slot_after_write": int(query_event["slot_after_write"]),
                "active_keys": int(axes["active_keys"]),
                "overwrite_count": int(axes["overwrite_count"]),
                "num_queries": int(axes["num_queries"]),
                "total_writes": int(axes["total_writes"]),
                "context_tokens": int(axes["context_tokens"]),
                "baseline_margin": float(margins[item_index].detach().float().cpu().item()),
                "baseline_correct": bool(correct[item_index].detach().cpu().item()),
            }
        )
    return result


def _group_query_indices_by_split(query_infos: list[dict[str, Any]]) -> dict[str, list[int]]:
    by_split: dict[str, list[int]] = defaultdict(list)
    for index, info in enumerate(query_infos):
        by_split[str(info["split"])].append(index)
        by_split["__all__"].append(index)
    return dict(by_split)


def _neuron_aggregate_rows_for_layer(
    *,
    step: int,
    layer: int,
    hidden: torch.Tensor,
    readout_coefficients: torch.Tensor,
    direct_contributions: torch.Tensor,
    write_norms: torch.Tensor,
    query_infos: list[dict[str, Any]],
    activation_threshold: float,
) -> list[dict[str, Any]]:
    if hidden.shape != readout_coefficients.shape or hidden.shape != direct_contributions.shape:
        raise RuntimeError(
            "Neuron trace tensors must share shape: "
            f"hidden={tuple(hidden.shape)} readout={tuple(readout_coefficients.shape)} "
            f"dla={tuple(direct_contributions.shape)}"
        )
    if hidden.size(0) != len(query_infos):
        raise RuntimeError("Hidden activations and query info rows have different query counts.")
    if write_norms.numel() != hidden.size(1):
        raise RuntimeError("write_norms length does not match hidden width.")

    rows: list[dict[str, Any]] = []
    indices_by_split = _group_query_indices_by_split(query_infos)
    for split_name, indices in sorted(indices_by_split.items()):
        if not indices:
            raise RuntimeError(f"Empty split index group for {split_name}.")
        index_tensor = torch.tensor(indices, device=hidden.device, dtype=torch.long)
        split_hidden = hidden.index_select(0, index_tensor).float()
        split_readout = readout_coefficients.index_select(0, index_tensor).float()
        split_dla = direct_contributions.index_select(0, index_tensor).float()
        activation_mean = split_hidden.mean(dim=0).detach().cpu().tolist()
        activation_abs_mean = split_hidden.abs().mean(dim=0).detach().cpu().tolist()
        activation_active_fraction = (
            (split_hidden.abs() > activation_threshold).float().mean(dim=0).detach().cpu().tolist()
        )
        readout_mean = split_readout.mean(dim=0).detach().cpu().tolist()
        readout_abs_mean = split_readout.abs().mean(dim=0).detach().cpu().tolist()
        dla_mean = split_dla.mean(dim=0).detach().cpu().tolist()
        dla_abs_mean = split_dla.abs().mean(dim=0).detach().cpu().tolist()
        dla_positive_fraction = (split_dla > 0.0).float().mean(dim=0).detach().cpu().tolist()
        dla_negative_fraction = (split_dla < 0.0).float().mean(dim=0).detach().cpu().tolist()
        write_norm_values = write_norms.detach().float().cpu().tolist()
        for neuron_index in range(hidden.size(1)):
            rows.append(
                {
                    "step": step,
                    "split": split_name,
                    "layer": layer,
                    "neuron": neuron_index,
                    "neuron_id": _neuron_id(layer, neuron_index),
                    "activation_mean": float(activation_mean[neuron_index]),
                    "activation_abs_mean": float(activation_abs_mean[neuron_index]),
                    "activation_active_fraction": float(activation_active_fraction[neuron_index]),
                    "readout_coefficient_mean": float(readout_mean[neuron_index]),
                    "readout_coefficient_abs_mean": float(readout_abs_mean[neuron_index]),
                    "direct_margin_contribution_mean": float(dla_mean[neuron_index]),
                    "direct_margin_contribution_abs_mean": float(dla_abs_mean[neuron_index]),
                    "direct_margin_contribution_positive_fraction": float(dla_positive_fraction[neuron_index]),
                    "direct_margin_contribution_negative_fraction": float(dla_negative_fraction[neuron_index]),
                    "write_vector_norm": float(write_norm_values[neuron_index]),
                    "num_query_events": len(indices),
                }
            )
    return rows


def _prompt_top_neuron_rows_for_layer(
    *,
    step: int,
    layer: int,
    hidden: torch.Tensor,
    readout_coefficients: torch.Tensor,
    direct_contributions: torch.Tensor,
    write_norms: torch.Tensor,
    query_infos: list[dict[str, Any]],
    top_k_per_query: int,
) -> tuple[list[dict[str, Any]], dict[str, list[set[int]]]]:
    if top_k_per_query < 0:
        raise ValueError("top_k_per_query must be non-negative.")
    if top_k_per_query == 0:
        return [], {}
    k = min(top_k_per_query, hidden.size(1))
    selection_specs = {
        "abs_dla": direct_contributions.abs(),
        "abs_activation": hidden.abs(),
    }
    rows: list[dict[str, Any]] = []
    selected_sets: dict[str, list[set[int]]] = {}
    for selection_kind, score_tensor in selection_specs.items():
        top_values, top_indices = torch.topk(score_tensor.float(), k=k, dim=1)
        selected_sets[selection_kind] = [set(int(item) for item in row.tolist()) for row in top_indices.detach().cpu()]
        for query_row_index, query_info in enumerate(query_infos):
            for rank_index in range(k):
                neuron_index = int(top_indices[query_row_index, rank_index].detach().cpu().item())
                rows.append(
                    {
                        "step": step,
                        **query_info,
                        "layer": layer,
                        "neuron": neuron_index,
                        "neuron_id": _neuron_id(layer, neuron_index),
                        "selection_kind": selection_kind,
                        "rank": rank_index + 1,
                        "selection_metric_value": float(top_values[query_row_index, rank_index].detach().cpu().item()),
                        "hidden_activation": float(hidden[query_row_index, neuron_index].detach().float().cpu().item()),
                        "readout_coefficient": float(
                            readout_coefficients[query_row_index, neuron_index].detach().float().cpu().item()
                        ),
                        "direct_margin_contribution": float(
                            direct_contributions[query_row_index, neuron_index].detach().float().cpu().item()
                        ),
                        "write_vector_norm": float(write_norms[neuron_index].detach().float().cpu().item()),
                    }
                )
    return rows, selected_sets


def _overlap_summary_rows(
    *,
    step: int,
    layer: int,
    selected_sets_by_kind: dict[str, list[set[int]]],
    query_infos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(query_infos) < 2:
        return rows
    for selection_kind, selected_sets in sorted(selected_sets_by_kind.items()):
        if len(selected_sets) != len(query_infos):
            raise RuntimeError("Selected neuron sets and query infos have different lengths.")
        accumulators: dict[str, dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "count": 0.0})
        for left_index in range(len(query_infos)):
            left_set = selected_sets[left_index]
            for right_index in range(left_index + 1, len(query_infos)):
                right_set = selected_sets[right_index]
                union = left_set | right_set
                if not union:
                    raise RuntimeError("Cannot compute overlap for two empty selected-neuron sets.")
                jaccard = len(left_set & right_set) / len(union)
                left = query_infos[left_index]
                right = query_infos[right_index]
                categories = ["all_pairs"]
                categories.append("same_query_key" if left["query_key"] == right["query_key"] else "different_query_key")
                categories.append(
                    "same_answer_value" if left["answer_value"] == right["answer_value"] else "different_answer_value"
                )
                same_pair = left["query_key"] == right["query_key"] and left["answer_value"] == right["answer_value"]
                categories.append("same_key_value_pair" if same_pair else "different_key_value_pair")
                categories.append("same_split" if left["split"] == right["split"] else "cross_split")
                for category in categories:
                    accumulators[category]["sum"] += float(jaccard)
                    accumulators[category]["count"] += 1.0
        for category, accumulator in sorted(accumulators.items()):
            count = int(accumulator["count"])
            if count <= 0:
                raise RuntimeError(f"Overlap category has no pairs: {category}")
            rows.append(
                {
                    "step": step,
                    "layer": layer,
                    "selection_kind": selection_kind,
                    "pair_category": category,
                    "mean_jaccard": float(accumulator["sum"]) / count,
                    "num_pairs": count,
                }
            )
    return rows


def _baseline_by_split_from_query_infos(query_infos: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    accumulators: dict[str, dict[str, float]] = {}
    for info in query_infos:
        for split_name in (str(info["split"]), "__all__"):
            if split_name not in accumulators:
                accumulators[split_name] = _new_behavior_accumulator()
            _accumulate_behavior(
                accumulators[split_name],
                margin=float(info["baseline_margin"]),
                correct=bool(info["baseline_correct"]),
            )
    return {
        split_name: {
            "baseline_margin_mean": float(summary["margin_mean"]),
            "baseline_accuracy": float(summary["accuracy"]),
            "num_query_events": int(summary["num_query_events"]),
        }
        for split_name, summary in (
            (split_name, _behavior_summary(accumulator)) for split_name, accumulator in accumulators.items()
        )
    }


def _select_ablation_neurons(
    *,
    neuron_rows: list[dict[str, Any]],
    mlp_layers: list[int],
    ablation_top_k_per_layer: int,
    explicit_neurons: list[str] | None,
) -> list[dict[str, Any]]:
    if ablation_top_k_per_layer < 0:
        raise ValueError("ablation_top_k_per_layer must be non-negative.")
    all_rows = [row for row in neuron_rows if str(row["split"]) == "__all__"]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for layer in mlp_layers:
        layer_rows = [row for row in all_rows if int(row["layer"]) == layer]
        ranked = sorted(
            layer_rows,
            key=lambda row: float(row["direct_margin_contribution_abs_mean"]),
            reverse=True,
        )
        for row in ranked[:ablation_top_k_per_layer]:
            key = (int(row["layer"]), int(row["neuron"]))
            if key not in seen:
                selected.append(row)
                seen.add(key)
    if explicit_neurons is not None:
        row_by_key = {(int(row["layer"]), int(row["neuron"])): row for row in all_rows}
        for neuron_id in explicit_neurons:
            key = _parse_neuron_id(neuron_id)
            row = row_by_key.get(key)
            if row is None:
                raise ValueError(f"Explicit neuron {neuron_id} is not present in the aggregate neuron rows.")
            if key not in seen:
                selected.append(row)
                seen.add(key)
    return selected


def _evaluate_neuron_ablation(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    vocab: Vocabulary,
    selected_row: dict[str, Any],
    baseline_query_infos: list[dict[str, Any]],
    baseline_by_split: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer = int(selected_row["layer"])
    neuron = int(selected_row["neuron"])
    d_ff = model.blocks[layer].ff.d_ff
    if neuron < 0 or neuron >= d_ff:
        raise ValueError(f"Neuron index {neuron} outside layer {layer} d_ff={d_ff}.")
    value_token_ids = torch.tensor(vocab.value_token_ids, device=batches[0]["input_ids"].device, dtype=torch.long)
    neuron_mask = {layer: torch.ones(d_ff, device=batches[0]["input_ids"].device)}
    neuron_mask[layer][neuron] = 0.0
    accumulators: dict[str, dict[str, float]] = {}
    query_rows: list[dict[str, Any]] = []
    query_cursor = 0
    with torch.no_grad():
        for batch in batches:
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                neuron_mask=neuron_mask,
            )
            answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
            margins = _value_margin(answer_logits, answer_targets, value_token_ids)
            correct = answer_logits.argmax(dim=-1) == answer_targets
            ablated_infos = _query_info_rows(
                batch=batch,
                vocab=vocab,
                metadata=metadata,
                answer_targets=answer_targets,
                margins=margins,
                correct=correct,
            )
            for local_info in ablated_infos:
                if query_cursor >= len(baseline_query_infos):
                    raise RuntimeError("Ablation produced more query rows than the baseline trace.")
                baseline_info = baseline_query_infos[query_cursor]
                for key_name in ("query_id", "split", "sample_id", "query_index"):
                    if local_info[key_name] != baseline_info[key_name]:
                        raise RuntimeError(
                            "Ablation query order mismatch for "
                            f"{key_name}: baseline={baseline_info[key_name]} ablated={local_info[key_name]}"
                        )
                for split_name in (str(local_info["split"]), "__all__"):
                    if split_name not in accumulators:
                        accumulators[split_name] = _new_behavior_accumulator()
                    _accumulate_behavior(
                        accumulators[split_name],
                        margin=float(local_info["baseline_margin"]),
                        correct=bool(local_info["baseline_correct"]),
                    )
                query_rows.append(
                    {
                        "step": int(selected_row["step"]),
                        "layer": layer,
                        "neuron": neuron,
                        "neuron_id": _neuron_id(layer, neuron),
                        "query_id": baseline_info["query_id"],
                        "split": baseline_info["split"],
                        "sample_id": baseline_info["sample_id"],
                        "query_index": baseline_info["query_index"],
                        "query_key": baseline_info["query_key"],
                        "answer_value": baseline_info["answer_value"],
                        "answer_pair_type": baseline_info["answer_pair_type"],
                        "baseline_margin": float(baseline_info["baseline_margin"]),
                        "ablated_margin": float(local_info["baseline_margin"]),
                        "margin_drop": float(baseline_info["baseline_margin"]) - float(local_info["baseline_margin"]),
                        "baseline_correct": bool(baseline_info["baseline_correct"]),
                        "ablated_correct": bool(local_info["baseline_correct"]),
                        "accuracy_drop_indicator": int(bool(baseline_info["baseline_correct"]))
                        - int(bool(local_info["baseline_correct"])),
                    }
                )
                query_cursor += 1
    if query_cursor != len(baseline_query_infos):
        raise RuntimeError(
            f"Ablation produced {query_cursor} query rows, expected {len(baseline_query_infos)} from baseline."
        )
    aggregate_rows: list[dict[str, Any]] = []
    for split_name, accumulator in sorted(accumulators.items()):
        ablated = _behavior_summary(accumulator)
        baseline = baseline_by_split.get(split_name)
        if baseline is None:
            raise RuntimeError(f"Missing baseline split summary for neuron ablation split {split_name}.")
        aggregate_rows.append(
            {
                "step": int(selected_row["step"]),
                "split": split_name,
                "layer": layer,
                "neuron": neuron,
                "neuron_id": _neuron_id(layer, neuron),
                "selection_direct_margin_contribution_mean": float(
                    selected_row["direct_margin_contribution_mean"]
                ),
                "selection_direct_margin_contribution_abs_mean": float(
                    selected_row["direct_margin_contribution_abs_mean"]
                ),
                "selection_activation_abs_mean": float(selected_row["activation_abs_mean"]),
                "selection_write_vector_norm": float(selected_row["write_vector_norm"]),
                "baseline_margin_mean": float(baseline["baseline_margin_mean"]),
                "ablated_margin_mean": float(ablated["margin_mean"]),
                "margin_drop": float(baseline["baseline_margin_mean"]) - float(ablated["margin_mean"]),
                "baseline_accuracy": float(baseline["baseline_accuracy"]),
                "ablated_accuracy": float(ablated["accuracy"]),
                "accuracy_drop": float(baseline["baseline_accuracy"]) - float(ablated["accuracy"]),
                "num_query_events": int(ablated["num_query_events"]),
            }
        )
    return aggregate_rows, query_rows


def _compute_prompt_neuron_checkpoint(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    vocab: Vocabulary,
    checkpoint_step: int,
    mlp_layers: list[int],
    activation_threshold: float,
    top_k_per_query: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    value_token_ids = torch.tensor(vocab.value_token_ids, device=batches[0]["input_ids"].device, dtype=torch.long)
    num_layers = len(model.blocks)
    query_infos: list[dict[str, Any]] = []
    layer_hidden_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in mlp_layers}
    layer_readout_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in mlp_layers}
    layer_dla_chunks: dict[int, list[torch.Tensor]] = {layer: [] for layer in mlp_layers}

    for batch in batches:
        with torch.no_grad():
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
                return_mlp_states=True,
            )
        if outputs.residual_streams is None:
            raise RuntimeError("Prompt neuron trace requires residual streams.")
        if outputs.mlp_states is None:
            raise RuntimeError("Prompt neuron trace requires MLP states.")
        answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        rows = metadata["rows"]
        prediction_positions = metadata["prediction_positions"]
        baseline_margins = _value_margin(answer_logits, answer_targets, value_token_ids)
        baseline_correct = answer_logits.argmax(dim=-1) == answer_targets
        wrong_token_ids = _best_wrong_value_token_ids(
            logits=answer_logits,
            answer_targets=answer_targets,
            value_token_ids=value_token_ids,
        )
        final_pre_stage = f"layer_{num_layers - 1}_post_mlp"
        final_pre_vectors = outputs.residual_streams[final_pre_stage][rows, prediction_positions, :]
        margin_gradients, recomputed_margins = _margin_gradient_vectors(
            model=model,
            final_residual_vectors=final_pre_vectors,
            correct_token_ids=answer_targets,
            wrong_token_ids=wrong_token_ids,
        )
        if not torch.allclose(recomputed_margins, baseline_margins, atol=1e-4, rtol=1e-4):
            max_delta = (recomputed_margins - baseline_margins).abs().max().item()
            raise RuntimeError(f"Prompt neuron trace margin-gradient check failed: max_delta={max_delta:.6g}")
        query_infos.extend(
            _query_info_rows(
                batch=batch,
                vocab=vocab,
                metadata=metadata,
                answer_targets=answer_targets,
                margins=baseline_margins,
                correct=baseline_correct,
            )
        )
        for layer in mlp_layers:
            state_key = f"layer_{layer}_hidden"
            if state_key not in outputs.mlp_states:
                raise KeyError(f"MLP state {state_key} not present in model output.")
            hidden = outputs.mlp_states[state_key][rows, prediction_positions, :].detach()
            fc_out_weight = model.blocks[layer].ff.fc_out.weight.detach().float()
            readout_coefficients = torch.matmul(margin_gradients.float(), fc_out_weight)
            direct_contributions = hidden.float() * readout_coefficients
            layer_hidden_chunks[layer].append(hidden)
            layer_readout_chunks[layer].append(readout_coefficients.detach())
            layer_dla_chunks[layer].append(direct_contributions.detach())

    if not query_infos:
        raise RuntimeError("Prompt neuron trace found no query events.")

    baseline_rows = [
        {
            "step": checkpoint_step,
            **info,
        }
        for info in query_infos
    ]
    aggregate_rows: list[dict[str, Any]] = []
    top_query_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for layer in mlp_layers:
        hidden = torch.cat(layer_hidden_chunks[layer], dim=0)
        readout_coefficients = torch.cat(layer_readout_chunks[layer], dim=0)
        direct_contributions = torch.cat(layer_dla_chunks[layer], dim=0)
        write_norms = model.blocks[layer].ff.fc_out.weight.detach().float().norm(dim=0)
        if hidden.size(0) != len(query_infos):
            raise RuntimeError(f"Layer {layer} hidden query count does not match query metadata.")
        aggregate_rows.extend(
            _neuron_aggregate_rows_for_layer(
                step=checkpoint_step,
                layer=layer,
                hidden=hidden,
                readout_coefficients=readout_coefficients,
                direct_contributions=direct_contributions,
                write_norms=write_norms,
                query_infos=query_infos,
                activation_threshold=activation_threshold,
            )
        )
        layer_top_rows, selected_sets = _prompt_top_neuron_rows_for_layer(
            step=checkpoint_step,
            layer=layer,
            hidden=hidden,
            readout_coefficients=readout_coefficients,
            direct_contributions=direct_contributions,
            write_norms=write_norms,
            query_infos=query_infos,
            top_k_per_query=top_k_per_query,
        )
        top_query_rows.extend(layer_top_rows)
        overlap_rows.extend(
            _overlap_summary_rows(
                step=checkpoint_step,
                layer=layer,
                selected_sets_by_kind=selected_sets,
                query_infos=query_infos,
            )
        )
    return baseline_rows, aggregate_rows, top_query_rows, overlap_rows, query_infos


def _summarize_prompt_neuron_report(
    *,
    baseline_rows: list[dict[str, Any]],
    neuron_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not baseline_rows:
        raise ValueError("Cannot summarize prompt neuron trace without baseline rows.")
    final_step = max(int(row["step"]) for row in baseline_rows)
    final_neurons = [row for row in neuron_rows if int(row["step"]) == final_step and str(row["split"]) == "__all__"]
    final_ablations = [
        row for row in ablation_rows if int(row["step"]) == final_step and str(row["split"]) == "__all__"
    ]
    final_overlap = [row for row in overlap_rows if int(row["step"]) == final_step]
    return {
        "num_checkpoints": len({int(row["step"]) for row in baseline_rows}),
        "steps": sorted({int(row["step"]) for row in baseline_rows}),
        "final_step": final_step,
        "num_baseline_query_rows": len(baseline_rows),
        "num_neuron_aggregate_rows": len(neuron_rows),
        "num_ablation_rows": len(ablation_rows),
        "top_final_abs_dla_neurons": sorted(
            final_neurons,
            key=lambda row: float(row["direct_margin_contribution_abs_mean"]),
            reverse=True,
        )[:16],
        "top_final_positive_dla_neurons": [
            row
            for row in sorted(
                final_neurons,
                key=lambda row: float(row["direct_margin_contribution_mean"]),
                reverse=True,
            )
            if float(row["direct_margin_contribution_mean"]) > 0.0
        ][:16],
        "top_final_negative_dla_neurons": [
            row
            for row in sorted(final_neurons, key=lambda row: float(row["direct_margin_contribution_mean"]))
            if float(row["direct_margin_contribution_mean"]) < 0.0
        ][:16],
        "final_ablation_rows_by_margin_drop": sorted(
            final_ablations,
            key=lambda row: float(row["margin_drop"]),
            reverse=True,
        ),
        "final_overlap_rows": sorted(
            final_overlap,
            key=lambda row: (int(row["layer"]), str(row["selection_kind"]), str(row["pair_category"])),
        ),
    }


def _plot_prompt_neuron_dla_trajectory(
    *,
    neuron_rows: list[dict[str, Any]],
    output_path: Path,
    top_k: int,
) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    _, plt = _import_matplotlib()
    all_rows = [row for row in neuron_rows if str(row["split"]) == "__all__"]
    if not all_rows:
        raise ValueError("Cannot plot prompt neuron DLA trajectory without __all__ rows.")
    final_step = max(int(row["step"]) for row in all_rows)
    selected = sorted(
        [row for row in all_rows if int(row["step"]) == final_step],
        key=lambda row: float(row["direct_margin_contribution_abs_mean"]),
        reverse=True,
    )[:top_k]
    fig, ax = plt.subplots(figsize=(12, 6))
    for selected_row in selected:
        neuron_id = str(selected_row["neuron_id"])
        rows = sorted(
            [row for row in all_rows if str(row["neuron_id"]) == neuron_id],
            key=lambda row: int(row["step"]),
        )
        ax.plot(
            [int(row["step"]) for row in rows],
            [float(row["direct_margin_contribution_mean"]) for row in rows],
            marker="o",
            label=neuron_id,
        )
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title("Neuron direct margin contribution trajectory")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("mean neuron DLA")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_prompt_neuron_ablation_scatter(
    *,
    ablation_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    rows = [row for row in ablation_rows if str(row["split"]) == "__all__"]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        [float(row["selection_direct_margin_contribution_mean"]) for row in rows],
        [float(row["margin_drop"]) for row in rows],
    )
    for row in rows:
        ax.annotate(str(row["neuron_id"]), (float(row["selection_direct_margin_contribution_mean"]), float(row["margin_drop"])))
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title("Neuron DLA vs causal ablation")
    ax.set_xlabel("mean neuron DLA")
    ax.set_ylabel("margin drop when neuron is ablated")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_prompt_neuron_overlap(
    *,
    overlap_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not overlap_rows:
        return None
    _, plt = _import_matplotlib()
    final_step = max(int(row["step"]) for row in overlap_rows)
    selected_categories = [
        "all_pairs",
        "same_query_key",
        "different_query_key",
        "same_answer_value",
        "different_answer_value",
        "same_split",
        "cross_split",
    ]
    rows = [
        row
        for row in overlap_rows
        if int(row["step"]) == final_step
        and str(row["selection_kind"]) == "abs_dla"
        and str(row["pair_category"]) in selected_categories
    ]
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: (int(row["layer"]), selected_categories.index(str(row["pair_category"]))))
    labels = [f"L{row['layer']} {row['pair_category']}" for row in rows]
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(labels)), 5))
    ax.bar(range(len(rows)), [float(row["mean_jaccard"]) for row in rows])
    ax.set_title(f"Prompt top-neuron overlap at step {final_step}")
    ax.set_ylabel("mean Jaccard overlap")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_prompt_neuron_trace_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    lines = [
        "# Prompt-Conditioned Neuron Trace",
        "",
        "## Calculation",
        "",
        "For each query event, this report computes the answer-value margin:",
        "",
        "```text",
        "m(x, y) = logit(correct value) - max_{wrong value token} logit(wrong value)",
        "```",
        "",
        "For an MLP neuron `n`, with hidden activation `a_n(x)` and output vector `W_out[:, n]`:",
        "",
        "```text",
        "write_n(x) = a_n(x) * W_out[:, n]",
        "DLA_n(x, y) = write_n(x) dot d m / d z_final",
        "```",
        "",
        "For selected neurons, causal ablation is:",
        "",
        "```text",
        "ablation_drop_n = margin_baseline - margin_with_neuron_n_zeroed",
        "```",
        "",
        "## Raw Outputs",
        "",
        f"- baseline query rows: `{report['baseline_query_rows_path']}`",
        f"- neuron aggregate rows: `{report['neuron_rows_path']}`",
        f"- prompt top-neuron rows: `{report['top_query_neuron_rows_path']}`",
        f"- overlap rows: `{report['overlap_rows_path']}`",
        f"- ablation rows: `{report['ablation_rows_path']}`",
        f"- ablation query rows: `{report['ablation_query_rows_path']}`",
        "",
        "## Final Top Absolute-DLA Neurons",
        "",
        "| neuron | DLA mean | abs DLA mean | activation abs mean | active fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["top_final_abs_dla_neurons"][:12]:
        lines.append(
            "| `{neuron}` | {dla:.6f} | {abs_dla:.6f} | {act:.6f} | {frac:.3f} |".format(
                neuron=row["neuron_id"],
                dla=float(row["direct_margin_contribution_mean"]),
                abs_dla=float(row["direct_margin_contribution_abs_mean"]),
                act=float(row["activation_abs_mean"]),
                frac=float(row["activation_active_fraction"]),
            )
        )
    if summary["final_ablation_rows_by_margin_drop"]:
        lines.extend(["", "## Final Causal Neuron Ablations", ""])
        lines.append("| neuron | DLA mean | margin drop | accuracy drop |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in summary["final_ablation_rows_by_margin_drop"]:
            lines.append(
                "| `{neuron}` | {dla:.6f} | {drop:.6f} | {acc:.6f} |".format(
                    neuron=row["neuron_id"],
                    dla=float(row["selection_direct_margin_contribution_mean"]),
                    drop=float(row["margin_drop"]),
                    acc=float(row["accuracy_drop"]),
                )
            )
    lines.extend(["", "## Plots", ""])
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_prompt_neuron_trace(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    mlp_layers: list[int] | None = None,
    activation_threshold: float = 0.0,
    top_k_per_query: int = 8,
    ablation_top_k_per_layer: int = 4,
    ablation_steps: list[int] | None = None,
    ablation_neurons: list[str] | None = None,
    top_k_plot_neurons: int = 12,
) -> tuple[Path, Path, dict[str, Path]]:
    if activation_threshold < 0.0:
        raise ValueError("activation_threshold must be non-negative.")
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    resolved_ablation_steps = set(ablation_steps) if ablation_steps is not None else {int(_checkpoint_step_from_path(checkpoints[-1]))}
    model = build_model(spec.model, len(vocab.tokens), device)
    resolved_mlp_layers = _resolve_mlp_layers(model, mlp_layers)
    loader = _make_probe_loader(
        probe_records=probe_records,
        batch_size=spec.evaluation.batch_size,
        pad_token_id=vocab.pad_token_id,
    )
    batches = [move_batch_to_device(batch, device) for batch in loader]
    if not batches:
        raise RuntimeError("Prompt neuron trace produced no batches.")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_query_rows_path = output_dir / "prompt_neuron_baseline_query_rows.jsonl"
    neuron_rows_path = output_dir / "prompt_neuron_rows.jsonl"
    top_query_neuron_rows_path = output_dir / "prompt_neuron_top_query_rows.jsonl"
    overlap_rows_path = output_dir / "prompt_neuron_overlap_rows.jsonl"
    ablation_rows_path = output_dir / "prompt_neuron_ablation_rows.jsonl"
    ablation_query_rows_path = output_dir / "prompt_neuron_ablation_query_rows.jsonl"
    progress_path = output_dir / "prompt_neuron_trace_progress.json"
    for partial_path in (
        baseline_query_rows_path,
        neuron_rows_path,
        top_query_neuron_rows_path,
        overlap_rows_path,
        ablation_rows_path,
        ablation_query_rows_path,
        progress_path,
    ):
        if partial_path.exists():
            partial_path.unlink()

    total_probe_queries = sum(len(record["query_events"]) for record in probe_records)
    print(
        "[prompt-neuron-trace] "
        f"checkpoints={len(checkpoints)} records={len(probe_records)} query_events={total_probe_queries} "
        f"layers={resolved_mlp_layers} device={device_name} ablation_steps={sorted(resolved_ablation_steps)}",
        flush=True,
    )
    all_baseline_rows: list[dict[str, Any]] = []
    all_neuron_rows: list[dict[str, Any]] = []
    all_top_query_rows: list[dict[str, Any]] = []
    all_overlap_rows: list[dict[str, Any]] = []
    all_ablation_rows: list[dict[str, Any]] = []
    all_ablation_query_rows: list[dict[str, Any]] = []

    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if step != path_step:
            raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={step} path={path_step}")
        print(f"[prompt-neuron-trace] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}", flush=True)
        baseline_rows, neuron_rows, top_query_rows, overlap_rows, query_infos = _compute_prompt_neuron_checkpoint(
            model=model,
            batches=batches,
            vocab=vocab,
            checkpoint_step=step,
            mlp_layers=resolved_mlp_layers,
            activation_threshold=activation_threshold,
            top_k_per_query=top_k_per_query,
        )
        for row in baseline_rows:
            append_jsonl(baseline_query_rows_path, row)
        for row in neuron_rows:
            append_jsonl(neuron_rows_path, row)
        for row in top_query_rows:
            append_jsonl(top_query_neuron_rows_path, row)
        for row in overlap_rows:
            append_jsonl(overlap_rows_path, row)
        all_baseline_rows.extend(baseline_rows)
        all_neuron_rows.extend(neuron_rows)
        all_top_query_rows.extend(top_query_rows)
        all_overlap_rows.extend(overlap_rows)

        if step in resolved_ablation_steps:
            selected_neurons = _select_ablation_neurons(
                neuron_rows=neuron_rows,
                mlp_layers=resolved_mlp_layers,
                ablation_top_k_per_layer=ablation_top_k_per_layer,
                explicit_neurons=ablation_neurons,
            )
            baseline_by_split = _baseline_by_split_from_query_infos(query_infos)
            for selected_row in selected_neurons:
                ablation_rows, ablation_query_rows = _evaluate_neuron_ablation(
                    model=model,
                    batches=batches,
                    vocab=vocab,
                    selected_row=selected_row,
                    baseline_query_infos=query_infos,
                    baseline_by_split=baseline_by_split,
                )
                for row in ablation_rows:
                    append_jsonl(ablation_rows_path, row)
                for row in ablation_query_rows:
                    append_jsonl(ablation_query_rows_path, row)
                all_ablation_rows.extend(ablation_rows)
                all_ablation_query_rows.extend(ablation_query_rows)

        write_json(
            progress_path,
            {
                "status": "running",
                "completed_checkpoints": checkpoint_index,
                "total_checkpoints": len(checkpoints),
                "last_completed_step": step,
                "baseline_query_rows_path": str(baseline_query_rows_path),
                "neuron_rows_path": str(neuron_rows_path),
                "top_query_neuron_rows_path": str(top_query_neuron_rows_path),
                "overlap_rows_path": str(overlap_rows_path),
                "ablation_rows_path": str(ablation_rows_path),
                "ablation_query_rows_path": str(ablation_query_rows_path),
            },
        )
        all_split_baseline = _baseline_by_split_from_query_infos(query_infos)["__all__"]
        print(
            "[prompt-neuron-trace] finished "
            f"step={step} margin={all_split_baseline['baseline_margin_mean']:.6f} "
            f"accuracy={all_split_baseline['baseline_accuracy']:.6f} "
            f"neuron_rows={len(neuron_rows)} ablation_rows_total={len(all_ablation_rows)}",
            flush=True,
        )

    summary = _summarize_prompt_neuron_report(
        baseline_rows=all_baseline_rows,
        neuron_rows=all_neuron_rows,
        ablation_rows=all_ablation_rows,
        overlap_rows=all_overlap_rows,
    )
    plot_paths: dict[str, Path] = {
        "neuron_dla_trajectory": output_dir / "prompt_neuron_dla_trajectory.svg",
    }
    _plot_prompt_neuron_dla_trajectory(
        neuron_rows=all_neuron_rows,
        output_path=plot_paths["neuron_dla_trajectory"],
        top_k=top_k_plot_neurons,
    )
    ablation_plot_path = _plot_prompt_neuron_ablation_scatter(
        ablation_rows=all_ablation_rows,
        output_path=output_dir / "prompt_neuron_ablation_vs_dla.svg",
    )
    if ablation_plot_path is not None:
        plot_paths["ablation_vs_dla"] = ablation_plot_path
    overlap_plot_path = _plot_prompt_neuron_overlap(
        overlap_rows=all_overlap_rows,
        output_path=output_dir / "prompt_neuron_overlap.svg",
    )
    if overlap_plot_path is not None:
        plot_paths["overlap"] = overlap_plot_path

    report_path = output_dir / "prompt_neuron_trace_report.json"
    markdown_path = output_dir / "prompt_neuron_trace_report.md"
    report = {
        "schema_version": PROMPT_NEURON_TRACE_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "mlp_layers": resolved_mlp_layers,
        "activation_threshold": activation_threshold,
        "top_k_per_query": top_k_per_query,
        "ablation_top_k_per_layer": ablation_top_k_per_layer,
        "ablation_steps": sorted(resolved_ablation_steps),
        "ablation_neurons": ablation_neurons or [],
        "calculation": {
            "margin": "logit(correct_value) - max_{wrong value token} logit(wrong_value)",
            "neuron_write": "hidden_activation_n * W_out[:, n]",
            "neuron_direct_logit_attribution": "neuron_write dot d margin / d final_pre_layernorm_residual",
            "causal_ablation_drop": "baseline margin - margin after zeroing the selected MLP hidden neuron",
            "overlap": "Jaccard overlap between per-query top-neuron sets",
        },
        "baseline_query_rows_path": str(baseline_query_rows_path),
        "neuron_rows_path": str(neuron_rows_path),
        "top_query_neuron_rows_path": str(top_query_neuron_rows_path),
        "overlap_rows_path": str(overlap_rows_path),
        "ablation_rows_path": str(ablation_rows_path),
        "ablation_query_rows_path": str(ablation_query_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_prompt_neuron_trace_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_checkpoints": len(checkpoints),
            "total_checkpoints": len(checkpoints),
            "last_completed_step": int(all_baseline_rows[-1]["step"]),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "baseline_query_rows_path": str(baseline_query_rows_path),
            "neuron_rows_path": str(neuron_rows_path),
            "top_query_neuron_rows_path": str(top_query_neuron_rows_path),
            "overlap_rows_path": str(overlap_rows_path),
            "ablation_rows_path": str(ablation_rows_path),
            "ablation_query_rows_path": str(ablation_query_rows_path),
        },
    )
    print(f"[prompt-neuron-trace] complete report={report_path}", flush=True)
    return report_path, markdown_path, plot_paths


GEOMETRY_SUBSPACE_NAMES = [
    "embedding_key_identity",
    "embedding_value_identity",
    "head_qk_query",
    "head_qk_key",
    "head_ov_input",
    "head_ov_output",
]
GEOMETRY_POSITION_ROLES = [
    "prediction",
    "query_key",
    "current_read_key",
    "support_key",
    "support_value",
    "support_op",
    "support_write",
    "key_distractors",
    "value_distractors",
    "all_query_relation",
    "causal_prefix",
]
GEOMETRY_INTERVENTION_OPERATIONS = ["remove", "keep"]
GEOMETRY_QUERY_MODES = ["single_query", "batch_union"]


def _validate_geometry_rank(*, rank: int, max_rank: int, label: str) -> None:
    if rank <= 0:
        raise ValueError(f"{label} rank must be positive, got {rank}.")
    if rank > max_rank:
        raise ValueError(f"{label} rank {rank} exceeds available rank {max_rank}.")


def _embedding_identity_basis(
    *,
    model: torch.nn.Module,
    vocab: Vocabulary,
    subspace_name: str,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    embedding_weight = model.token_embedding.weight.detach().float().cpu()
    if subspace_name == "embedding_key_identity":
        token_ids = [vocab.token_to_id[token] for token in vocab.key_tokens]
        token_label = "key"
    elif subspace_name == "embedding_value_identity":
        token_ids = [vocab.token_to_id[token] for token in vocab.value_tokens]
        token_label = "value"
    else:
        raise ValueError(f"Unsupported embedding identity subspace: {subspace_name}")
    token_id_tensor = torch.tensor(token_ids, dtype=torch.long)
    vectors = embedding_weight.index_select(0, token_id_tensor)
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    available_rank = int(torch.linalg.matrix_rank(centered).item())
    _validate_geometry_rank(rank=rank, max_rank=available_rank, label=subspace_name)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:rank, :].T.contiguous().to(device=device)
    return basis, {
        "subspace_name": subspace_name,
        "subspace_type": "embedding_identity",
        "token_class": token_label,
        "num_tokens": len(token_ids),
        "available_rank": available_rank,
        "selected_rank": rank,
        "ambient_dim": int(embedding_weight.size(1)),
        "singular_values": [float(value) for value in singular_values[:rank].tolist()],
        "basis_svd_device": "cpu",
    }


def _head_geometry_basis(
    *,
    model: torch.nn.Module,
    subspace_name: str,
    head_layer: int | None,
    head: int | None,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if head_layer is None or head is None:
        raise ValueError(f"{subspace_name} requires --head-layer and --head.")
    if head_layer < 0 or head_layer >= len(model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(model.blocks) - 1}.")
    block = model.blocks[head_layer]
    if head < 0 or head >= block.attn.n_heads:
        raise ValueError(f"head {head} outside model range 0..{block.attn.n_heads - 1} for layer {head_layer}.")

    head_dim = block.attn.head_dim
    head_slice = slice(head * head_dim, (head + 1) * head_dim)
    q_rows = block.attn.q_proj.weight.detach().float().cpu()[head_slice, :]
    k_rows = block.attn.k_proj.weight.detach().float().cpu()[head_slice, :]
    v_rows = block.attn.v_proj.weight.detach().float().cpu()[head_slice, :]
    out_head = block.attn.out_proj.weight.detach().float().cpu()[:, head_slice]
    qk_matrix = q_rows.T.matmul(k_rows)
    ov_matrix = v_rows.T.matmul(out_head.T)
    qk_u, qk_s, qk_vh = torch.linalg.svd(qk_matrix, full_matrices=False)
    ov_u, ov_s, ov_vh = torch.linalg.svd(ov_matrix, full_matrices=False)

    if subspace_name == "head_qk_query":
        singular_values = qk_s
        basis = qk_u[:, :rank]
        matrix_name = "QK"
        side = "query"
    elif subspace_name == "head_qk_key":
        singular_values = qk_s
        basis = qk_vh[:rank, :].T.contiguous()
        matrix_name = "QK"
        side = "key"
    elif subspace_name == "head_ov_input":
        singular_values = ov_s
        basis = ov_u[:, :rank]
        matrix_name = "OV"
        side = "input"
    elif subspace_name == "head_ov_output":
        singular_values = ov_s
        basis = ov_vh[:rank, :].T.contiguous()
        matrix_name = "OV"
        side = "output"
    else:
        raise ValueError(f"Unsupported head geometry subspace: {subspace_name}")
    _validate_geometry_rank(rank=rank, max_rank=int(singular_values.numel()), label=subspace_name)
    return basis.to(device=device), {
        "subspace_name": subspace_name,
        "subspace_type": "head_singular_vectors",
        "matrix": matrix_name,
        "side": side,
        "head_layer": head_layer,
        "head": head,
        "head_label": _head_label(head_layer, head),
        "selected_rank": rank,
        "available_rank": int(singular_values.numel()),
        "ambient_dim": int(qk_matrix.size(0)),
        "head_dim": int(head_dim),
        "singular_values": [float(value) for value in singular_values[:rank].tolist()],
        "basis_svd_device": "cpu",
    }


def _resolve_geometry_subspace_basis(
    *,
    model: torch.nn.Module,
    vocab: Vocabulary,
    subspace_name: str,
    rank: int,
    head_layer: int | None,
    head: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if subspace_name not in GEOMETRY_SUBSPACE_NAMES:
        raise ValueError(f"Unsupported subspace {subspace_name!r}; expected one of {GEOMETRY_SUBSPACE_NAMES}.")
    if subspace_name.startswith("embedding_"):
        if head_layer is not None or head is not None:
            raise ValueError(f"{subspace_name} does not use --head-layer or --head.")
        return _embedding_identity_basis(
            model=model,
            vocab=vocab,
            subspace_name=subspace_name,
            rank=rank,
            device=device,
        )
    return _head_geometry_basis(
        model=model,
        subspace_name=subspace_name,
        head_layer=head_layer,
        head=head,
        rank=rank,
        device=device,
    )


def _validate_geometry_stage(*, model: torch.nn.Module, stage_name: str) -> None:
    valid_stages = _stage_order(len(model.blocks))
    if stage_name not in valid_stages:
        raise ValueError(f"Unsupported stage {stage_name!r}; expected one of {valid_stages}.")


def _intervention_positions_for_query(
    *,
    batch: dict[str, Any],
    metadata: dict[str, torch.Tensor],
    flat_index: int,
    position_role: str,
) -> tuple[int, list[int]]:
    if position_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported position role {position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    batch_row = int(metadata["rows"][flat_index].item())
    query_index = int(metadata["query_indices"][flat_index].item())
    prediction_position = int(metadata["prediction_positions"][flat_index].item())
    query_key_position = int(metadata["query_key_positions"][flat_index].item())
    record = batch["records"][batch_row]
    query_geometry = _positions_for_query(record, query_index, prediction_position)
    if position_role == "prediction":
        positions = [prediction_position]
    elif position_role in {"query_key", "current_read_key"}:
        positions = [query_key_position]
    elif position_role == "support_key":
        positions = [int(query_geometry["support_key_position"])]
    elif position_role == "support_value":
        positions = [int(query_geometry["support_value_position"])]
    elif position_role == "support_op":
        positions = [int(query_geometry["support_op_position"])]
    elif position_role == "support_write":
        positions = [
            int(query_geometry["support_op_position"]),
            int(query_geometry["support_key_position"]),
            int(query_geometry["support_value_position"]),
        ]
    elif position_role == "key_distractors":
        positions = [int(position) for position in query_geometry["key_distractors"]]
    elif position_role == "value_distractors":
        positions = [int(position) for position in query_geometry["value_distractors"]]
    elif position_role == "all_query_relation":
        positions = [
            int(query_geometry["support_key_position"]),
            int(query_geometry["support_value_position"]),
            query_key_position,
        ]
    elif position_role == "causal_prefix":
        positions = list(range(prediction_position + 1))
    else:
        raise ValueError(f"Unhandled position role: {position_role}")
    positions = sorted(set(positions))
    if not positions:
        raise RuntimeError(f"No positions selected for {position_role} in {record['sample_id']} query {query_index}.")
    invalid = [position for position in positions if position < 0 or position > prediction_position]
    if invalid:
        raise RuntimeError(
            f"Intervention positions must be in the causal prefix for {record['sample_id']} query {query_index}: {invalid}"
        )
    return batch_row, positions


def _apply_geometry_operation(
    *,
    vectors: torch.Tensor,
    basis: torch.Tensor,
    operation: str,
) -> torch.Tensor:
    if operation not in GEOMETRY_INTERVENTION_OPERATIONS:
        raise ValueError(f"Unsupported operation {operation!r}; expected one of {GEOMETRY_INTERVENTION_OPERATIONS}.")
    if vectors.ndim != 2:
        raise ValueError(f"Expected rank-2 vectors, got shape {tuple(vectors.shape)}.")
    if basis.ndim != 2:
        raise ValueError(f"Expected rank-2 basis, got shape {tuple(basis.shape)}.")
    if vectors.size(-1) != basis.size(0):
        raise ValueError(f"Vector dim {vectors.size(-1)} does not match basis dim {basis.size(0)}.")
    basis = basis.to(device=vectors.device, dtype=vectors.dtype)
    projection = vectors.matmul(basis).matmul(basis.T)
    if operation == "remove":
        return vectors - projection
    if operation == "keep":
        return projection
    raise ValueError(f"Unhandled operation: {operation}")


def _patched_stage_tensor(
    *,
    stage_state: torch.Tensor,
    selected_positions: list[tuple[int, int]],
    basis: torch.Tensor,
    operation: str,
) -> torch.Tensor:
    if not selected_positions:
        raise ValueError("selected_positions must not be empty.")
    unique_positions = sorted(set(selected_positions))
    batch_indices = torch.tensor([item[0] for item in unique_positions], device=stage_state.device, dtype=torch.long)
    position_indices = torch.tensor([item[1] for item in unique_positions], device=stage_state.device, dtype=torch.long)
    patched = stage_state.clone()
    selected = patched[batch_indices, position_indices, :]
    patched_selected = _apply_geometry_operation(vectors=selected, basis=basis, operation=operation)
    patched[batch_indices, position_indices, :] = patched_selected
    return patched


def _validate_query_metadata_match(
    *,
    baseline_metadata: dict[str, torch.Tensor],
    patched_metadata: dict[str, torch.Tensor],
) -> None:
    for key in ("rows", "query_indices", "answer_positions", "prediction_positions"):
        if key not in baseline_metadata or key not in patched_metadata:
            raise KeyError(f"Missing query metadata key during intervention validation: {key}")
        if not torch.equal(baseline_metadata[key].detach().cpu(), patched_metadata[key].detach().cpu()):
            raise RuntimeError(f"Patched forward changed query metadata ordering for key {key}.")


def _new_geometry_intervention_accumulator() -> dict[str, float]:
    return {
        "baseline_margin_sum": 0.0,
        "intervened_margin_sum": 0.0,
        "margin_drop_sum": 0.0,
        "margin_drop_abs_sum": 0.0,
        "positive_margin_drop_count": 0.0,
        "negative_margin_drop_count": 0.0,
        "baseline_correct_count": 0.0,
        "intervened_correct_count": 0.0,
        "total": 0.0,
    }


def _accumulate_geometry_intervention(
    accumulator: dict[str, float],
    *,
    baseline_margin: float,
    intervened_margin: float,
    baseline_correct: bool,
    intervened_correct: bool,
) -> None:
    margin_drop = float(baseline_margin) - float(intervened_margin)
    accumulator["baseline_margin_sum"] += float(baseline_margin)
    accumulator["intervened_margin_sum"] += float(intervened_margin)
    accumulator["margin_drop_sum"] += margin_drop
    accumulator["margin_drop_abs_sum"] += abs(margin_drop)
    if margin_drop > 0.0:
        accumulator["positive_margin_drop_count"] += 1.0
    if margin_drop < 0.0:
        accumulator["negative_margin_drop_count"] += 1.0
    if baseline_correct:
        accumulator["baseline_correct_count"] += 1.0
    if intervened_correct:
        accumulator["intervened_correct_count"] += 1.0
    accumulator["total"] += 1.0


def _geometry_intervention_summary(accumulator: dict[str, float]) -> dict[str, float | int]:
    total = int(accumulator["total"])
    if total <= 0:
        raise RuntimeError("Cannot summarize empty geometry intervention accumulator.")
    baseline_accuracy = float(accumulator["baseline_correct_count"]) / total
    intervened_accuracy = float(accumulator["intervened_correct_count"]) / total
    return {
        "baseline_margin_mean": float(accumulator["baseline_margin_sum"]) / total,
        "intervened_margin_mean": float(accumulator["intervened_margin_sum"]) / total,
        "margin_drop_mean": float(accumulator["margin_drop_sum"]) / total,
        "margin_drop_abs_mean": float(accumulator["margin_drop_abs_sum"]) / total,
        "margin_drop_positive_fraction": float(accumulator["positive_margin_drop_count"]) / total,
        "margin_drop_negative_fraction": float(accumulator["negative_margin_drop_count"]) / total,
        "baseline_accuracy": baseline_accuracy,
        "intervened_accuracy": intervened_accuracy,
        "accuracy_drop": baseline_accuracy - intervened_accuracy,
        "num_query_events": total,
    }


def _geometry_query_row(
    *,
    step: int,
    subspace_name: str,
    stage_name: str,
    operation: str,
    position_role: str,
    query_mode: str,
    rank: int,
    subspace_summary: dict[str, Any],
    baseline_info: dict[str, Any],
    intervened_margin: float,
    intervened_correct: bool,
    selected_positions: list[int],
) -> dict[str, Any]:
    return {
        "step": step,
        "subspace_name": subspace_name,
        "stage": stage_name,
        "operation": operation,
        "position_role": position_role,
        "query_mode": query_mode,
        "rank": rank,
        "subspace_type": str(subspace_summary["subspace_type"]),
        "head_label": subspace_summary.get("head_label"),
        "split": baseline_info["split"],
        "sample_id": baseline_info["sample_id"],
        "query_index": baseline_info["query_index"],
        "query_id": baseline_info["query_id"],
        "query_key": baseline_info["query_key"],
        "answer_value": baseline_info["answer_value"],
        "answer_pair_type": baseline_info["answer_pair_type"],
        "baseline_margin": float(baseline_info["baseline_margin"]),
        "intervened_margin": float(intervened_margin),
        "margin_drop": float(baseline_info["baseline_margin"]) - float(intervened_margin),
        "baseline_correct": bool(baseline_info["baseline_correct"]),
        "intervened_correct": bool(intervened_correct),
        "accuracy_drop_indicator": int(bool(baseline_info["baseline_correct"])) - int(bool(intervened_correct)),
        "patched_position_count": len(selected_positions),
        "patched_positions": selected_positions,
    }


def _aggregate_geometry_query_rows(
    *,
    query_rows: list[dict[str, Any]],
    step: int,
    subspace_name: str,
    stage_name: str,
    operation: str,
    position_role: str,
    query_mode: str,
    rank: int,
    subspace_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    accumulators: dict[str, dict[str, float]] = {}
    for row in query_rows:
        for split_name in (str(row["split"]), "__all__"):
            if split_name not in accumulators:
                accumulators[split_name] = _new_geometry_intervention_accumulator()
            _accumulate_geometry_intervention(
                accumulators[split_name],
                baseline_margin=float(row["baseline_margin"]),
                intervened_margin=float(row["intervened_margin"]),
                baseline_correct=bool(row["baseline_correct"]),
                intervened_correct=bool(row["intervened_correct"]),
            )
    rows: list[dict[str, Any]] = []
    for split_name, accumulator in sorted(accumulators.items()):
        rows.append(
            {
                "step": step,
                "split": split_name,
                "subspace_name": subspace_name,
                "stage": stage_name,
                "operation": operation,
                "position_role": position_role,
                "query_mode": query_mode,
                "rank": rank,
                "subspace_type": str(subspace_summary["subspace_type"]),
                "head_label": subspace_summary.get("head_label"),
                **_geometry_intervention_summary(accumulator),
            }
        )
    return rows


def _patched_query_margins_from_outputs(
    *,
    outputs: Any,
    batch: dict[str, Any],
    value_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
    margins = _value_margin(answer_logits, answer_targets, value_token_ids)
    correct = answer_logits.argmax(dim=-1) == answer_targets
    return margins, correct, metadata


def _compute_geometry_intervention_checkpoint(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    vocab: Vocabulary,
    checkpoint_step: int,
    basis: torch.Tensor,
    subspace_summary: dict[str, Any],
    subspace_name: str,
    rank: int,
    stage_name: str,
    operation: str,
    position_role: str,
    query_mode: str,
    progress_every_queries: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if query_mode not in GEOMETRY_QUERY_MODES:
        raise ValueError(f"Unsupported query mode {query_mode!r}; expected one of {GEOMETRY_QUERY_MODES}.")
    if progress_every_queries < 0:
        raise ValueError("progress_every_queries must be non-negative.")
    value_token_ids = torch.tensor(vocab.value_token_ids, device=batches[0]["input_ids"].device, dtype=torch.long)
    checkpoint_query_rows: list[dict[str, Any]] = []
    processed_queries = 0
    total_queries = sum(len(record["query_events"]) for batch in batches for record in batch["records"])

    for batch in batches:
        with torch.no_grad():
            baseline_outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
            )
        if baseline_outputs.residual_streams is None:
            raise RuntimeError("Geometry subspace intervention requires residual streams.")
        if stage_name not in baseline_outputs.residual_streams:
            raise KeyError(f"Stage {stage_name!r} not found in residual streams.")
        baseline_margins, baseline_correct, baseline_metadata = _patched_query_margins_from_outputs(
            outputs=baseline_outputs,
            batch=batch,
            value_token_ids=value_token_ids,
        )
        answer_logits, answer_targets, _ = extract_answer_logits(baseline_outputs.logits, batch)
        baseline_infos = _query_info_rows(
            batch=batch,
            vocab=vocab,
            metadata=baseline_metadata,
            answer_targets=answer_targets,
            margins=baseline_margins,
            correct=baseline_correct,
        )
        stage_state = baseline_outputs.residual_streams[stage_name]
        selected_by_flat_index: list[tuple[int, list[int]]] = [
            _intervention_positions_for_query(
                batch=batch,
                metadata=baseline_metadata,
                flat_index=flat_index,
                position_role=position_role,
            )
            for flat_index in range(baseline_margins.numel())
        ]

        if query_mode == "batch_union":
            union_positions: list[tuple[int, int]] = []
            for batch_row, positions in selected_by_flat_index:
                union_positions.extend((batch_row, position) for position in positions)
            patched_stage = _patched_stage_tensor(
                stage_state=stage_state,
                selected_positions=union_positions,
                basis=basis,
                operation=operation,
            )
            with torch.no_grad():
                patched_outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    residual_patch={stage_name: patched_stage},
                )
            intervened_margins, intervened_correct, patched_metadata = _patched_query_margins_from_outputs(
                outputs=patched_outputs,
                batch=batch,
                value_token_ids=value_token_ids,
            )
            _validate_query_metadata_match(
                baseline_metadata=baseline_metadata,
                patched_metadata=patched_metadata,
            )
            for flat_index, baseline_info in enumerate(baseline_infos):
                _, selected_positions = selected_by_flat_index[flat_index]
                checkpoint_query_rows.append(
                    _geometry_query_row(
                        step=checkpoint_step,
                        subspace_name=subspace_name,
                        stage_name=stage_name,
                        operation=operation,
                        position_role=position_role,
                        query_mode=query_mode,
                        rank=rank,
                        subspace_summary=subspace_summary,
                        baseline_info=baseline_info,
                        intervened_margin=float(intervened_margins[flat_index].detach().float().cpu().item()),
                        intervened_correct=bool(intervened_correct[flat_index].detach().cpu().item()),
                        selected_positions=selected_positions,
                    )
                )
            processed_queries += len(baseline_infos)
        elif query_mode == "single_query":
            for flat_index, baseline_info in enumerate(baseline_infos):
                batch_row, selected_positions = selected_by_flat_index[flat_index]
                patched_stage = _patched_stage_tensor(
                    stage_state=stage_state,
                    selected_positions=[(batch_row, position) for position in selected_positions],
                    basis=basis,
                    operation=operation,
                )
                with torch.no_grad():
                    patched_outputs = model(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        residual_patch={stage_name: patched_stage},
                    )
                intervened_margins, intervened_correct, patched_metadata = _patched_query_margins_from_outputs(
                    outputs=patched_outputs,
                    batch=batch,
                    value_token_ids=value_token_ids,
                )
                _validate_query_metadata_match(
                    baseline_metadata=baseline_metadata,
                    patched_metadata=patched_metadata,
                )
                checkpoint_query_rows.append(
                    _geometry_query_row(
                        step=checkpoint_step,
                        subspace_name=subspace_name,
                        stage_name=stage_name,
                        operation=operation,
                        position_role=position_role,
                        query_mode=query_mode,
                        rank=rank,
                        subspace_summary=subspace_summary,
                        baseline_info=baseline_info,
                        intervened_margin=float(intervened_margins[flat_index].detach().float().cpu().item()),
                        intervened_correct=bool(intervened_correct[flat_index].detach().cpu().item()),
                        selected_positions=selected_positions,
                    )
                )
                processed_queries += 1
                if progress_every_queries and processed_queries % progress_every_queries == 0:
                    print(
                        "[geometry-subspace-intervention] "
                        f"step={checkpoint_step} processed_queries={processed_queries}/{total_queries}",
                        flush=True,
                    )
        else:
            raise ValueError(f"Unhandled query mode: {query_mode}")

    if len(checkpoint_query_rows) != total_queries:
        raise RuntimeError(
            f"Geometry intervention produced {len(checkpoint_query_rows)} query rows, expected {total_queries}."
        )
    aggregate_rows = _aggregate_geometry_query_rows(
        query_rows=checkpoint_query_rows,
        step=checkpoint_step,
        subspace_name=subspace_name,
        stage_name=stage_name,
        operation=operation,
        position_role=position_role,
        query_mode=query_mode,
        rank=rank,
        subspace_summary=subspace_summary,
    )
    return aggregate_rows, checkpoint_query_rows


def _summarize_geometry_intervention_report(
    *,
    aggregate_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not aggregate_rows:
        raise ValueError("Cannot summarize geometry intervention without aggregate rows.")
    final_step = max(int(row["step"]) for row in aggregate_rows)
    final_aggregate = [row for row in aggregate_rows if int(row["step"]) == final_step]
    final_query_rows = [row for row in query_rows if int(row["step"]) == final_step]
    all_rows = [row for row in final_aggregate if str(row["split"]) == "__all__"]
    if len(all_rows) != 1:
        raise RuntimeError(f"Expected exactly one final __all__ row, got {len(all_rows)}.")
    top_positive_queries = [
        row
        for row in sorted(final_query_rows, key=lambda item: float(item["margin_drop"]), reverse=True)
        if float(row["margin_drop"]) > 0.0
    ][:16]
    top_negative_queries = [
        row
        for row in sorted(final_query_rows, key=lambda item: float(item["margin_drop"]))
        if float(row["margin_drop"]) < 0.0
    ][:16]
    return {
        "num_checkpoints": len({int(row["step"]) for row in aggregate_rows}),
        "steps": sorted({int(row["step"]) for row in aggregate_rows}),
        "final_step": final_step,
        "final_all": all_rows[0],
        "final_by_split": sorted(final_aggregate, key=lambda row: str(row["split"])),
        "top_final_positive_margin_drop_queries": top_positive_queries,
        "top_final_negative_margin_drop_queries": top_negative_queries,
    }


def _plot_geometry_margin_drop_trajectory(
    *,
    aggregate_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    _, plt = _import_matplotlib()
    split_names = sorted({str(row["split"]) for row in aggregate_rows})
    fig, ax = plt.subplots(figsize=(12, 6))
    for split_name in split_names:
        rows = sorted(
            [row for row in aggregate_rows if str(row["split"]) == split_name],
            key=lambda row: int(row["step"]),
        )
        ax.plot(
            [int(row["step"]) for row in rows],
            [float(row["margin_drop_mean"]) for row in rows],
            marker="o",
            label=split_name,
        )
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title("Causal subspace margin drop")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("baseline margin - intervened margin")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_geometry_accuracy_drop_trajectory(
    *,
    aggregate_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    _, plt = _import_matplotlib()
    split_names = sorted({str(row["split"]) for row in aggregate_rows})
    fig, ax = plt.subplots(figsize=(12, 6))
    for split_name in split_names:
        rows = sorted(
            [row for row in aggregate_rows if str(row["split"]) == split_name],
            key=lambda row: int(row["step"]),
        )
        ax.plot(
            [int(row["step"]) for row in rows],
            [float(row["accuracy_drop"]) for row in rows],
            marker="o",
            label=split_name,
        )
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title("Causal subspace accuracy drop")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("baseline accuracy - intervened accuracy")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncols=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)


def _plot_geometry_query_drop_histogram(
    *,
    query_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not query_rows:
        return None
    _, plt = _import_matplotlib()
    final_step = max(int(row["step"]) for row in query_rows)
    rows = [row for row in query_rows if int(row["step"]) == final_step]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist([float(row["margin_drop"]) for row in rows], bins=40)
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title(f"Per-query subspace margin drops at step {final_step}")
    ax.set_xlabel("baseline margin - intervened margin")
    ax.set_ylabel("query count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_geometry_intervention_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    final_all = summary["final_all"]
    lines = [
        "# Geometry Subspace Intervention",
        "",
        "## Calculation",
        "",
        "For an orthonormal basis `B` of the selected subspace and a residual vector `z` at the selected stage/positions:",
        "",
        "```text",
        "remove: z' = z - (z B) B^T",
        "keep:   z' = (z B) B^T",
        "```",
        "",
        "The causal effect is measured by rerunning the model with the patched residual stream and computing:",
        "",
        "```text",
        "margin_drop = margin_baseline - margin_intervened",
        "margin = logit(correct value) - max_{wrong value token} logit(wrong value)",
        "```",
        "",
        "This is a geometry intervention, not a feature-family or neuron ranking: the selected vector subspace itself is removed or isolated.",
        "",
        "## Intervention",
        "",
        f"- subspace: `{report['subspace']['subspace_name']}`",
        f"- rank: `{report['rank']}`",
        f"- stage: `{report['stage']}`",
        f"- position role: `{report['position_role']}`",
        f"- operation: `{report['operation']}`",
        f"- query mode: `{report['query_mode']}`",
        f"- basis SVD device: `{report['subspace']['basis_svd_device']}`",
        "",
        "## Final Aggregate",
        "",
        "| split | baseline margin | intervened margin | margin drop | baseline acc | intervened acc | acc drop | positive drop frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["final_by_split"]:
        lines.append(
            "| {split} | {base:.6f} | {inter:.6f} | {drop:.6f} | {bacc:.6f} | {iacc:.6f} | {adrop:.6f} | {pfrac:.3f} |".format(
                split=row["split"],
                base=float(row["baseline_margin_mean"]),
                inter=float(row["intervened_margin_mean"]),
                drop=float(row["margin_drop_mean"]),
                bacc=float(row["baseline_accuracy"]),
                iacc=float(row["intervened_accuracy"]),
                adrop=float(row["accuracy_drop"]),
                pfrac=float(row["margin_drop_positive_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "## Final All-Query Result",
            "",
            f"- margin drop: `{float(final_all['margin_drop_mean']):.6f}`",
            f"- accuracy drop: `{float(final_all['accuracy_drop']):.6f}`",
            f"- positive margin-drop fraction: `{float(final_all['margin_drop_positive_fraction']):.3f}`",
            f"- negative margin-drop fraction: `{float(final_all['margin_drop_negative_fraction']):.3f}`",
            "",
            "## Raw Outputs",
            "",
            f"- aggregate rows: `{report['aggregate_rows_path']}`",
            f"- query rows: `{report['query_rows_path']}`",
            "",
            "## Plots",
            "",
        ]
    )
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_geometry_subspace_intervention(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    stage_name: str,
    subspace_name: str,
    rank: int,
    operation: str,
    position_role: str,
    query_mode: str,
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    head_layer: int | None = None,
    head: int | None = None,
    progress_every_queries: int = 100,
) -> tuple[Path, Path, Path, Path, dict[str, Path]]:
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    model = build_model(spec.model, len(vocab.tokens), device)
    _validate_geometry_stage(model=model, stage_name=stage_name)
    if operation not in GEOMETRY_INTERVENTION_OPERATIONS:
        raise ValueError(f"Unsupported operation {operation!r}; expected one of {GEOMETRY_INTERVENTION_OPERATIONS}.")
    if position_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported position role {position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if query_mode not in GEOMETRY_QUERY_MODES:
        raise ValueError(f"Unsupported query mode {query_mode!r}; expected one of {GEOMETRY_QUERY_MODES}.")
    loader = _make_probe_loader(
        probe_records=probe_records,
        batch_size=spec.evaluation.batch_size,
        pad_token_id=vocab.pad_token_id,
    )
    batches = [move_batch_to_device(batch, device) for batch in loader]
    if not batches:
        raise RuntimeError("Geometry subspace intervention produced no batches.")

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows_path = output_dir / "geometry_subspace_intervention_rows.jsonl"
    query_rows_path = output_dir / "geometry_subspace_intervention_query_rows.jsonl"
    progress_path = output_dir / "geometry_subspace_intervention_progress.json"
    for partial_path in (aggregate_rows_path, query_rows_path, progress_path):
        if partial_path.exists():
            partial_path.unlink()

    total_probe_queries = sum(len(record["query_events"]) for record in probe_records)
    print(
        "[geometry-subspace-intervention] "
        f"checkpoints={len(checkpoints)} records={len(probe_records)} query_events={total_probe_queries} "
        f"device={device_name} subspace={subspace_name} rank={rank} stage={stage_name} "
        f"role={position_role} operation={operation} query_mode={query_mode}",
        flush=True,
    )

    all_aggregate_rows: list[dict[str, Any]] = []
    all_query_rows: list[dict[str, Any]] = []
    final_subspace_summary: dict[str, Any] | None = None
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if step != path_step:
            raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={step} path={path_step}")
        basis, subspace_summary = _resolve_geometry_subspace_basis(
            model=model,
            vocab=vocab,
            subspace_name=subspace_name,
            rank=rank,
            head_layer=head_layer,
            head=head,
            device=device,
        )
        final_subspace_summary = subspace_summary
        print(
            f"[geometry-subspace-intervention] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        aggregate_rows, query_rows = _compute_geometry_intervention_checkpoint(
            model=model,
            batches=batches,
            vocab=vocab,
            checkpoint_step=step,
            basis=basis,
            subspace_summary=subspace_summary,
            subspace_name=subspace_name,
            rank=rank,
            stage_name=stage_name,
            operation=operation,
            position_role=position_role,
            query_mode=query_mode,
            progress_every_queries=progress_every_queries,
        )
        for row in aggregate_rows:
            append_jsonl(aggregate_rows_path, row)
        for row in query_rows:
            append_jsonl(query_rows_path, row)
        all_aggregate_rows.extend(aggregate_rows)
        all_query_rows.extend(query_rows)
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_checkpoints": checkpoint_index,
                "total_checkpoints": len(checkpoints),
                "last_completed_step": step,
                "aggregate_rows_path": str(aggregate_rows_path),
                "query_rows_path": str(query_rows_path),
            },
        )
        all_row = next(row for row in aggregate_rows if str(row["split"]) == "__all__")
        print(
            "[geometry-subspace-intervention] finished "
            f"step={step} margin_drop={float(all_row['margin_drop_mean']):.6f} "
            f"accuracy_drop={float(all_row['accuracy_drop']):.6f} "
            f"positive_drop_fraction={float(all_row['margin_drop_positive_fraction']):.3f}",
            flush=True,
        )

    if final_subspace_summary is None:
        raise RuntimeError("No checkpoints were processed for geometry subspace intervention.")
    summary = _summarize_geometry_intervention_report(
        aggregate_rows=all_aggregate_rows,
        query_rows=all_query_rows,
    )
    report_path = output_dir / "geometry_subspace_intervention_report.json"
    markdown_path = output_dir / "geometry_subspace_intervention_report.md"
    plot_paths: dict[str, Path] = {
        "margin_drop": output_dir / "geometry_subspace_margin_drop.svg",
        "accuracy_drop": output_dir / "geometry_subspace_accuracy_drop.svg",
    }
    _plot_geometry_margin_drop_trajectory(aggregate_rows=all_aggregate_rows, output_path=plot_paths["margin_drop"])
    _plot_geometry_accuracy_drop_trajectory(aggregate_rows=all_aggregate_rows, output_path=plot_paths["accuracy_drop"])
    histogram_path = _plot_geometry_query_drop_histogram(
        query_rows=all_query_rows,
        output_path=output_dir / "geometry_subspace_query_margin_drop_histogram.svg",
    )
    if histogram_path is not None:
        plot_paths["query_margin_drop_histogram"] = histogram_path
    report = {
        "schema_version": GEOMETRY_SUBSPACE_INTERVENTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "stage": stage_name,
        "subspace": final_subspace_summary,
        "subspace_name": subspace_name,
        "rank": rank,
        "operation": operation,
        "position_role": position_role,
        "query_mode": query_mode,
        "progress_every_queries": progress_every_queries,
        "calculation": {
            "basis": "B is an orthonormal basis from token-identity PCA or head QK/OV singular vectors.",
            "remove": "z' = z - (z B) B^T at selected residual positions",
            "keep": "z' = (z B) B^T at selected residual positions",
            "margin": "logit(correct_value) - max_{wrong value token} logit(wrong_value)",
            "causal_effect": "margin_drop = margin_baseline - margin_intervened after rerunning the model",
            "single_query_mode": "patches only the selected positions for one query event per forward pass",
            "batch_union_mode": "patches the union of selected positions for all query events in a batch in one forward pass",
        },
        "aggregate_rows_path": str(aggregate_rows_path),
        "query_rows_path": str(query_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_geometry_intervention_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_checkpoints": len(checkpoints),
            "total_checkpoints": len(checkpoints),
            "last_completed_step": int(summary["final_step"]),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "aggregate_rows_path": str(aggregate_rows_path),
            "query_rows_path": str(query_rows_path),
        },
    )
    print(
        f"[geometry-subspace-intervention] complete report={report_path} rows={aggregate_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, aggregate_rows_path, query_rows_path, plot_paths


CAUSAL_VARIABLE_PAIR_TYPES = ["query_key", "support_value", "recency", "distractor"]
CAUSAL_VARIABLE_PATCH_SUBSPACES = ["full_residual", *GEOMETRY_SUBSPACE_NAMES]


def _holdout_pair_set(metadata: dict[str, Any]) -> set[tuple[str, str]]:
    raw_pairs = metadata.get("holdout_pairs")
    if raw_pairs is None:
        raise KeyError("Benchmark metadata is missing holdout_pairs.")
    if not isinstance(raw_pairs, list):
        raise TypeError("Benchmark metadata holdout_pairs must be a list.")
    pairs: set[tuple[str, str]] = set()
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, str) or ":" not in raw_pair:
            raise ValueError(f"Invalid holdout pair entry: {raw_pair!r}")
        key, value = raw_pair.split(":", maxsplit=1)
        if not key or not value:
            raise ValueError(f"Invalid holdout pair entry: {raw_pair!r}")
        pairs.add((key, value))
    return pairs


def _answer_pair_type_for_value(*, key: str, value: str, holdout_pairs: set[tuple[str, str]]) -> str:
    return "heldout" if (key, value) in holdout_pairs else "seen"


def _find_write(record: dict[str, Any], write_index: int) -> dict[str, Any]:
    matches = [write for write in record["writes"] if int(write["write_index"]) == write_index]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one write index {write_index} in {record['sample_id']}, got {len(matches)}.")
    return matches[0]


def _find_write_step(record: dict[str, Any], write_index: int) -> dict[str, Any]:
    matches = [
        step
        for step in record["steps"]
        if step["op"] == "write" and int(step["write_index"]) == write_index
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one write step {write_index} in {record['sample_id']}, got {len(matches)}.")
    return matches[0]


def _find_read_step(record: dict[str, Any], step_index: int) -> dict[str, Any]:
    matches = [
        step
        for step in record["steps"]
        if step["op"] == "read" and int(step["step_index"]) == step_index
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one read step {step_index} in {record['sample_id']}, got {len(matches)}.")
    return matches[0]


def _latest_writes_before_event(record: dict[str, Any], event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    query_step_index = int(event["step_index"])
    latest: dict[str, dict[str, Any]] = {}
    for step in record["steps"]:
        step_index = int(step["step_index"])
        if step_index >= query_step_index:
            break
        if step["op"] != "write":
            continue
        write_index = int(step["write_index"])
        latest[str(step["key"])] = {
            "write_index": write_index,
            "key": str(step["key"]),
            "value": str(step["value"]),
            "positions": dict(step["positions"]),
        }
    return latest


def _prior_read_uses_write(record: dict[str, Any], event: dict[str, Any], write_index: int) -> bool:
    query_step_index = int(event["step_index"])
    for step in record["steps"]:
        step_index = int(step["step_index"])
        if step_index >= query_step_index:
            break
        if step["op"] == "read" and int(step["support_write_index"]) == write_index:
            return True
    return False


def _used_write_values(record: dict[str, Any]) -> set[str]:
    return {str(write["value"]) for write in record["writes"]}


def _replacement_value(
    *,
    vocab: Vocabulary,
    key: str,
    holdout_pairs: set[tuple[str, str]],
    reference_pair_type: str,
    excluded_values: set[str],
) -> str | None:
    if reference_pair_type not in {"seen", "heldout"}:
        raise ValueError(f"Unsupported answer pair type for replacement: {reference_pair_type!r}")
    require_holdout = reference_pair_type == "heldout"
    candidates = [
        value
        for value in vocab.value_tokens
        if value not in excluded_values and (((key, value) in holdout_pairs) == require_holdout)
    ]
    if not candidates:
        return None
    return candidates[0]


def _make_single_query_record(
    *,
    source_record: dict[str, Any],
    source_query_index: int,
    sample_id: str,
    vocab: Vocabulary,
) -> dict[str, Any]:
    if source_query_index < 0 or source_query_index >= len(source_record["query_events"]):
        raise IndexError(
            f"source_query_index {source_query_index} outside query range for {source_record['sample_id']}."
        )
    record = copy.deepcopy(source_record)
    event = copy.deepcopy(source_record["query_events"][source_query_index])
    current_read_step = _find_read_step(record, int(event["step_index"]))
    event["query_index"] = 0
    current_read_step["query_index"] = 0
    record["sample_id"] = sample_id
    record["source_sample_id"] = str(source_record["sample_id"])
    record["source_query_index"] = source_query_index
    record["query_events"] = [event]
    record["query_plan"] = [
        {
            "slot_after_write": int(event["slot_after_write"]),
            "key": str(event["key"]),
            "support_write_index": int(event["support_write_index"]),
            "writes_since_support": int(event["writes_since_support"]),
        }
    ]
    record["token_ids"] = vocab.encode([str(token) for token in record["tokens"]])
    _validate_pair_record(record=record, vocab=vocab)
    return record


def _set_write_value(*, record: dict[str, Any], write_index: int, value: str) -> None:
    write = _find_write(record, write_index)
    step = _find_write_step(record, write_index)
    value_position = int(step["positions"]["value"])
    write["value"] = value
    step["value"] = value
    record["tokens"][value_position] = value


def _set_current_query_event(
    *,
    record: dict[str, Any],
    key: str,
    answer_value: str,
    support_write_index: int,
    holdout_pairs: set[tuple[str, str]],
) -> None:
    if len(record["query_events"]) != 1:
        raise RuntimeError(f"Pair record must contain exactly one query event: {record['sample_id']}")
    event = record["query_events"][0]
    read_step = _find_read_step(record, int(event["step_index"]))
    support_step = _find_write_step(record, support_write_index)
    support_positions = dict(support_step["positions"])
    query_key_position = int(event["positions"]["key"])
    answer_position = int(event["positions"]["answer"])
    if int(read_step["positions"]["key"]) != query_key_position:
        raise RuntimeError(f"Read step/query event key position mismatch in {record['sample_id']}.")
    if int(read_step["positions"]["answer"]) != answer_position:
        raise RuntimeError(f"Read step/query event answer position mismatch in {record['sample_id']}.")

    event["key"] = key
    event["answer_value"] = answer_value
    event["support_write_index"] = support_write_index
    event["support_positions"] = support_positions
    event["writes_since_support"] = int(event["slot_after_write"]) - support_write_index
    event["tokens_since_support"] = answer_position - int(support_positions["value"])
    event["answer_pair_type"] = _answer_pair_type_for_value(
        key=key,
        value=answer_value,
        holdout_pairs=holdout_pairs,
    )
    read_step["key"] = key
    read_step["value"] = answer_value
    read_step["support_write_index"] = support_write_index
    record["tokens"][query_key_position] = key
    record["tokens"][answer_position] = answer_value
    record["query_plan"] = [
        {
            "slot_after_write": int(event["slot_after_write"]),
            "key": key,
            "support_write_index": support_write_index,
            "writes_since_support": int(event["writes_since_support"]),
        }
    ]


def _validate_pair_record(*, record: dict[str, Any], vocab: Vocabulary) -> None:
    if len(record["query_events"]) != 1:
        raise RuntimeError(f"Causal patch pair record must contain exactly one query event: {record['sample_id']}")
    event = record["query_events"][0]
    read_step = _find_read_step(record, int(event["step_index"]))
    if read_step["op"] != "read":
        raise RuntimeError(f"Current event step is not a read step in {record['sample_id']}.")
    if str(read_step["key"]) != str(event["key"]):
        raise RuntimeError(f"Read step key disagrees with event key in {record['sample_id']}.")
    if str(read_step["value"]) != str(event["answer_value"]):
        raise RuntimeError(f"Read step answer disagrees with event answer in {record['sample_id']}.")
    query_key_position = int(event["positions"]["key"])
    answer_position = int(event["positions"]["answer"])
    if str(record["tokens"][query_key_position]) != str(event["key"]):
        raise RuntimeError(f"Query key token disagrees with event in {record['sample_id']}.")
    if str(record["tokens"][answer_position]) != str(event["answer_value"]):
        raise RuntimeError(f"Answer token disagrees with event in {record['sample_id']}.")
    support_write_index = int(event["support_write_index"])
    support_step = _find_write_step(record, support_write_index)
    support_positions = dict(event["support_positions"])
    if int(support_step["positions"]["value"]) != int(support_positions["value"]):
        raise RuntimeError(f"Support value position mismatch in {record['sample_id']}.")
    if str(support_step["value"]) != str(event["answer_value"]):
        raise RuntimeError(f"Support write value disagrees with event answer in {record['sample_id']}.")
    for step in record["steps"]:
        positions = step["positions"]
        if step["op"] == "write":
            if str(record["tokens"][int(positions["key"])]) != str(step["key"]):
                raise RuntimeError(f"Write key token mismatch in {record['sample_id']}.")
            if str(record["tokens"][int(positions["value"])]) != str(step["value"]):
                raise RuntimeError(f"Write value token mismatch in {record['sample_id']}.")
        elif step["op"] == "read":
            if str(record["tokens"][int(positions["key"])]) != str(step["key"]):
                raise RuntimeError(f"Read key token mismatch in {record['sample_id']}.")
            if str(record["tokens"][int(positions["answer"])]) != str(step["value"]):
                raise RuntimeError(f"Read answer token mismatch in {record['sample_id']}.")
        else:
            raise RuntimeError(f"Unsupported step op in pair record {record['sample_id']}: {step['op']}")
    token_ids = vocab.encode([str(token) for token in record["tokens"]])
    if token_ids != list(record["token_ids"]):
        raise RuntimeError(f"token_ids are stale in pair record {record['sample_id']}.")


def _finalize_pair_record(*, record: dict[str, Any], vocab: Vocabulary) -> None:
    record["token_ids"] = vocab.encode([str(token) for token in record["tokens"]])
    _validate_pair_record(record=record, vocab=vocab)


def _pair_metadata(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in pair.items()
        if key not in {"clean_record", "corrupted_record"}
    }


def _build_query_key_pair(
    *,
    source_record: dict[str, Any],
    source_query_index: int,
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
) -> tuple[dict[str, Any] | None, str | None]:
    event = source_record["query_events"][source_query_index]
    latest = _latest_writes_before_event(source_record, event)
    query_key = str(event["key"])
    clean_answer = str(event["answer_value"])
    candidates = [
        item
        for key, item in sorted(latest.items())
        if key != query_key and str(item["value"]) != clean_answer
    ]
    if not candidates:
        return None, "no_alternative_current_key"
    alternative = candidates[0]
    pair_id = f"query_key:{source_record['sample_id']}:{source_query_index}:{alternative['key']}"
    clean_record = _make_single_query_record(
        source_record=source_record,
        source_query_index=source_query_index,
        sample_id=f"{pair_id}:clean",
        vocab=vocab,
    )
    corrupted_record = _make_single_query_record(
        source_record=source_record,
        source_query_index=source_query_index,
        sample_id=f"{pair_id}:corrupted",
        vocab=vocab,
    )
    _set_current_query_event(
        record=corrupted_record,
        key=str(alternative["key"]),
        answer_value=str(alternative["value"]),
        support_write_index=int(alternative["write_index"]),
        holdout_pairs=holdout_pairs,
    )
    _finalize_pair_record(record=corrupted_record, vocab=vocab)
    return {
        "pair_id": pair_id,
        "pair_type": "query_key",
        "split": str(source_record["split"]),
        "source_sample_id": str(source_record["sample_id"]),
        "source_query_index": source_query_index,
        "clean_answer_value": clean_answer,
        "corrupted_answer_value": str(alternative["value"]),
        "clean_transfer_token": clean_answer,
        "corrupted_transfer_token": str(alternative["value"]),
        "mutation": {
            "changed_role": "current_read_key",
            "clean_query_key": query_key,
            "corrupted_query_key": str(alternative["key"]),
            "clean_support_write_index": int(event["support_write_index"]),
            "corrupted_support_write_index": int(alternative["write_index"]),
        },
        "clean_record": clean_record,
        "corrupted_record": corrupted_record,
    }, None


def _build_support_value_pair(
    *,
    source_record: dict[str, Any],
    source_query_index: int,
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
) -> tuple[dict[str, Any] | None, str | None]:
    event = source_record["query_events"][source_query_index]
    support_write_index = int(event["support_write_index"])
    if _prior_read_uses_write(source_record, event, support_write_index):
        return None, "support_write_used_by_prior_read"
    query_key = str(event["key"])
    clean_answer = str(event["answer_value"])
    replacement = _replacement_value(
        vocab=vocab,
        key=query_key,
        holdout_pairs=holdout_pairs,
        reference_pair_type=str(event["answer_pair_type"]),
        excluded_values={*_used_write_values(source_record), clean_answer},
    )
    if replacement is None:
        return None, "no_replacement_value_for_support_pair_type"
    pair_id = f"support_value:{source_record['sample_id']}:{source_query_index}:{replacement}"
    clean_record = _make_single_query_record(
        source_record=source_record,
        source_query_index=source_query_index,
        sample_id=f"{pair_id}:clean",
        vocab=vocab,
    )
    corrupted_record = _make_single_query_record(
        source_record=source_record,
        source_query_index=source_query_index,
        sample_id=f"{pair_id}:corrupted",
        vocab=vocab,
    )
    _set_write_value(record=corrupted_record, write_index=support_write_index, value=replacement)
    _set_current_query_event(
        record=corrupted_record,
        key=query_key,
        answer_value=replacement,
        support_write_index=support_write_index,
        holdout_pairs=holdout_pairs,
    )
    _finalize_pair_record(record=corrupted_record, vocab=vocab)
    return {
        "pair_id": pair_id,
        "pair_type": "support_value",
        "split": str(source_record["split"]),
        "source_sample_id": str(source_record["sample_id"]),
        "source_query_index": source_query_index,
        "clean_answer_value": clean_answer,
        "corrupted_answer_value": replacement,
        "clean_transfer_token": clean_answer,
        "corrupted_transfer_token": replacement,
        "mutation": {
            "changed_role": "support_write_value",
            "query_key": query_key,
            "support_write_index": support_write_index,
            "clean_support_value": clean_answer,
            "corrupted_support_value": replacement,
        },
        "clean_record": clean_record,
        "corrupted_record": corrupted_record,
    }, None


def _build_recency_pair(
    *,
    source_record: dict[str, Any],
    source_query_index: int,
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
) -> tuple[dict[str, Any] | None, str | None]:
    event = source_record["query_events"][source_query_index]
    support_write_index = int(event["support_write_index"])
    query_key = str(event["key"])
    stale_writes = [
        write
        for write in source_record["writes"]
        if str(write["key"]) == query_key and int(write["write_index"]) < support_write_index
    ]
    stale_writes = sorted(stale_writes, key=lambda item: int(item["write_index"]), reverse=True)
    if not stale_writes:
        return None, "no_same_key_stale_write"
    if _prior_read_uses_write(source_record, event, support_write_index):
        return None, "support_write_used_by_prior_read"
    clean_answer = str(event["answer_value"])
    for stale_write in stale_writes:
        stale_write_index = int(stale_write["write_index"])
        if _prior_read_uses_write(source_record, event, stale_write_index):
            continue
        stale_value = str(stale_write["value"])
        if stale_value == clean_answer:
            continue
        pair_id = f"recency:{source_record['sample_id']}:{source_query_index}:{stale_write_index}"
        clean_record = _make_single_query_record(
            source_record=source_record,
            source_query_index=source_query_index,
            sample_id=f"{pair_id}:clean",
            vocab=vocab,
        )
        corrupted_record = _make_single_query_record(
            source_record=source_record,
            source_query_index=source_query_index,
            sample_id=f"{pair_id}:corrupted",
            vocab=vocab,
        )
        _set_write_value(record=corrupted_record, write_index=stale_write_index, value=clean_answer)
        _set_write_value(record=corrupted_record, write_index=support_write_index, value=stale_value)
        _set_current_query_event(
            record=corrupted_record,
            key=query_key,
            answer_value=stale_value,
            support_write_index=support_write_index,
            holdout_pairs=holdout_pairs,
        )
        _finalize_pair_record(record=corrupted_record, vocab=vocab)
        return {
            "pair_id": pair_id,
            "pair_type": "recency",
            "split": str(source_record["split"]),
            "source_sample_id": str(source_record["sample_id"]),
            "source_query_index": source_query_index,
            "clean_answer_value": clean_answer,
            "corrupted_answer_value": stale_value,
            "clean_transfer_token": clean_answer,
            "corrupted_transfer_token": stale_value,
            "mutation": {
                "changed_role": "same_key_write_order_values",
                "query_key": query_key,
                "stale_write_index": stale_write_index,
                "support_write_index": support_write_index,
                "clean_latest_value": clean_answer,
                "corrupted_latest_value": stale_value,
            },
            "clean_record": clean_record,
            "corrupted_record": corrupted_record,
        }, None
    return None, "same_key_stale_writes_used_by_prior_reads"


def _build_distractor_pair(
    *,
    source_record: dict[str, Any],
    source_query_index: int,
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
) -> tuple[dict[str, Any] | None, str | None]:
    event = source_record["query_events"][source_query_index]
    query_key = str(event["key"])
    clean_answer = str(event["answer_value"])
    candidate_writes = [
        write
        for write in source_record["writes"]
        if str(write["key"]) != query_key and int(write["write_index"]) < int(event["slot_after_write"]) + 1
    ]
    candidate_writes = sorted(candidate_writes, key=lambda item: int(item["write_index"]), reverse=True)
    if not candidate_writes:
        return None, "no_distractor_write"
    for distractor_write in candidate_writes:
        write_index = int(distractor_write["write_index"])
        if _prior_read_uses_write(source_record, event, write_index):
            continue
        distractor_key = str(distractor_write["key"])
        clean_distractor_value = str(distractor_write["value"])
        reference_pair_type = _answer_pair_type_for_value(
            key=distractor_key,
            value=clean_distractor_value,
            holdout_pairs=holdout_pairs,
        )
        replacement = _replacement_value(
            vocab=vocab,
            key=distractor_key,
            holdout_pairs=holdout_pairs,
            reference_pair_type=reference_pair_type,
            excluded_values={*_used_write_values(source_record), clean_answer, clean_distractor_value},
        )
        if replacement is None:
            continue
        pair_id = f"distractor:{source_record['sample_id']}:{source_query_index}:{write_index}:{replacement}"
        clean_record = _make_single_query_record(
            source_record=source_record,
            source_query_index=source_query_index,
            sample_id=f"{pair_id}:clean",
            vocab=vocab,
        )
        corrupted_record = _make_single_query_record(
            source_record=source_record,
            source_query_index=source_query_index,
            sample_id=f"{pair_id}:corrupted",
            vocab=vocab,
        )
        _set_write_value(record=corrupted_record, write_index=write_index, value=replacement)
        _finalize_pair_record(record=corrupted_record, vocab=vocab)
        return {
            "pair_id": pair_id,
            "pair_type": "distractor",
            "split": str(source_record["split"]),
            "source_sample_id": str(source_record["sample_id"]),
            "source_query_index": source_query_index,
            "clean_answer_value": clean_answer,
            "corrupted_answer_value": clean_answer,
            "clean_transfer_token": clean_distractor_value,
            "corrupted_transfer_token": replacement,
            "mutation": {
                "changed_role": "distractor_write_value",
                "distractor_key": distractor_key,
                "distractor_write_index": write_index,
                "clean_distractor_value": clean_distractor_value,
                "corrupted_distractor_value": replacement,
                "query_key": query_key,
            },
            "clean_record": clean_record,
            "corrupted_record": corrupted_record,
        }, None
    return None, "no_eligible_distractor_replacement"


def _build_causal_patch_pair(
    *,
    pair_type: str,
    source_record: dict[str, Any],
    source_query_index: int,
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
) -> tuple[dict[str, Any] | None, str | None]:
    if pair_type == "query_key":
        return _build_query_key_pair(
            source_record=source_record,
            source_query_index=source_query_index,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
        )
    if pair_type == "support_value":
        return _build_support_value_pair(
            source_record=source_record,
            source_query_index=source_query_index,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
        )
    if pair_type == "recency":
        return _build_recency_pair(
            source_record=source_record,
            source_query_index=source_query_index,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
        )
    if pair_type == "distractor":
        return _build_distractor_pair(
            source_record=source_record,
            source_query_index=source_query_index,
            vocab=vocab,
            holdout_pairs=holdout_pairs,
        )
    raise ValueError(f"Unsupported causal variable pair type {pair_type!r}; expected one of {CAUSAL_VARIABLE_PAIR_TYPES}.")


def _build_causal_patch_pairs(
    *,
    probe_records: list[dict[str, Any]],
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
    pair_types: list[str],
    max_pairs_per_type: int,
    min_pairs_per_type: int,
    split_filter: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not pair_types:
        raise ValueError("At least one pair type is required.")
    unsupported = [pair_type for pair_type in pair_types if pair_type not in CAUSAL_VARIABLE_PAIR_TYPES]
    if unsupported:
        raise ValueError(f"Unsupported pair types {unsupported}; expected one of {CAUSAL_VARIABLE_PAIR_TYPES}.")
    if max_pairs_per_type <= 0:
        raise ValueError("max_pairs_per_type must be positive.")
    if min_pairs_per_type <= 0:
        raise ValueError("min_pairs_per_type must be positive.")
    if min_pairs_per_type > max_pairs_per_type:
        raise ValueError("min_pairs_per_type cannot exceed max_pairs_per_type.")
    split_set = set(split_filter) if split_filter is not None else None
    pairs: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    skip_reasons: dict[str, Counter[str]] = {pair_type: Counter() for pair_type in pair_types}
    scanned_queries: Counter[str] = Counter()
    scanned_records = 0

    for record in probe_records:
        split = str(record["split"])
        if split_set is not None and split not in split_set:
            continue
        scanned_records += 1
        for source_query_index in range(len(record["query_events"])):
            for pair_type in pair_types:
                if counts[pair_type] >= max_pairs_per_type:
                    continue
                scanned_queries[pair_type] += 1
                pair, skip_reason = _build_causal_patch_pair(
                    pair_type=pair_type,
                    source_record=record,
                    source_query_index=source_query_index,
                    vocab=vocab,
                    holdout_pairs=holdout_pairs,
                )
                if pair is None:
                    if skip_reason is None:
                        raise RuntimeError(f"Pair builder returned no pair and no skip reason for {pair_type}.")
                    skip_reasons[pair_type][skip_reason] += 1
                    continue
                pairs.append(pair)
                counts[pair_type] += 1
            if all(counts[pair_type] >= max_pairs_per_type for pair_type in pair_types):
                break
        if all(counts[pair_type] >= max_pairs_per_type for pair_type in pair_types):
            break

    too_few = {
        pair_type: int(counts[pair_type])
        for pair_type in pair_types
        if counts[pair_type] < min_pairs_per_type
    }
    if too_few:
        reason_summary = {
            pair_type: dict(skip_reasons[pair_type].most_common())
            for pair_type in pair_types
        }
        if split_set is not None and scanned_records == 0:
            available_splits = sorted({str(record["split"]) for record in probe_records})
            raise RuntimeError(
                "Failed to construct causal patch pairs because split_filter matched no probe records: "
                f"split_filter={sorted(split_set)} available_splits={available_splits}"
            )
        raise RuntimeError(
            "Failed to construct the requested minimum causal patch pairs: "
            f"counts={too_few} min_pairs_per_type={min_pairs_per_type} skip_reasons={reason_summary}"
        )

    return pairs, {
        "requested_pair_types": pair_types,
        "split_filter": sorted(split_set) if split_set is not None else None,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "scanned_records": scanned_records,
        "constructed_counts": {pair_type: int(counts[pair_type]) for pair_type in pair_types},
        "scanned_queries": {pair_type: int(scanned_queries[pair_type]) for pair_type in pair_types},
        "skip_reasons": {
            pair_type: [
                {"reason": reason, "count": int(count)}
                for reason, count in skip_reasons[pair_type].most_common()
            ]
            for pair_type in pair_types
        },
    }


def _resolve_causal_patch_basis(
    *,
    model: torch.nn.Module,
    vocab: Vocabulary,
    subspace_name: str,
    rank: int | None,
    head_layer: int | None,
    head: int | None,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if subspace_name not in CAUSAL_VARIABLE_PATCH_SUBSPACES:
        raise ValueError(
            f"Unsupported causal patch subspace {subspace_name!r}; expected one of {CAUSAL_VARIABLE_PATCH_SUBSPACES}."
        )
    if subspace_name == "full_residual":
        if rank is not None:
            raise ValueError("full_residual patching does not use --rank.")
        if head_layer is not None or head is not None:
            raise ValueError("full_residual patching does not use --head-layer or --head.")
        return None, {
            "subspace_name": subspace_name,
            "subspace_type": "full_residual",
            "selected_rank": None,
            "basis_svd_device": None,
        }
    if rank is None:
        raise ValueError(f"{subspace_name} requires --rank.")
    return _resolve_geometry_subspace_basis(
        model=model,
        vocab=vocab,
        subspace_name=subspace_name,
        rank=rank,
        head_layer=head_layer,
        head=head,
        device=device,
    )


def _replace_projection_from_clean(
    *,
    clean_vectors: torch.Tensor,
    corrupted_vectors: torch.Tensor,
    basis: torch.Tensor | None,
) -> torch.Tensor:
    if clean_vectors.shape != corrupted_vectors.shape:
        raise ValueError(
            f"Clean/corrupted selected vector shapes differ: {tuple(clean_vectors.shape)} vs {tuple(corrupted_vectors.shape)}"
        )
    if clean_vectors.ndim != 2:
        raise ValueError(f"Expected selected vectors to be rank-2, got shape {tuple(clean_vectors.shape)}.")
    if basis is None:
        return clean_vectors
    basis = basis.to(device=corrupted_vectors.device, dtype=corrupted_vectors.dtype)
    clean_projection = clean_vectors.matmul(basis).matmul(basis.T)
    corrupted_projection = corrupted_vectors.matmul(basis).matmul(basis.T)
    return corrupted_vectors - corrupted_projection + clean_projection


def _transfer_stage_tensor(
    *,
    clean_stage: torch.Tensor,
    corrupted_stage: torch.Tensor,
    clean_selected: list[tuple[int, list[int]]],
    corrupted_selected: list[tuple[int, list[int]]],
    basis: torch.Tensor | None,
) -> torch.Tensor:
    if clean_stage.shape != corrupted_stage.shape:
        raise ValueError(
            f"Clean/corrupted stage shapes differ: {tuple(clean_stage.shape)} vs {tuple(corrupted_stage.shape)}"
        )
    if len(clean_selected) != len(corrupted_selected):
        raise ValueError("Clean/corrupted selected position lists have different lengths.")
    patched = corrupted_stage.clone()
    for pair_index, ((clean_row, clean_positions), (corrupted_row, corrupted_positions)) in enumerate(
        zip(clean_selected, corrupted_selected, strict=True)
    ):
        if len(clean_positions) != len(corrupted_positions):
            raise RuntimeError(
                f"Pair {pair_index} has different clean/corrupted position counts: "
                f"{len(clean_positions)} vs {len(corrupted_positions)}."
            )
        if not clean_positions:
            raise RuntimeError(f"Pair {pair_index} has no selected positions.")
        clean_position_tensor = torch.tensor(clean_positions, device=clean_stage.device, dtype=torch.long)
        corrupted_position_tensor = torch.tensor(corrupted_positions, device=corrupted_stage.device, dtype=torch.long)
        clean_vectors = clean_stage[clean_row, clean_position_tensor, :]
        corrupted_vectors = corrupted_stage[corrupted_row, corrupted_position_tensor, :]
        patched_vectors = _replace_projection_from_clean(
            clean_vectors=clean_vectors,
            corrupted_vectors=corrupted_vectors,
            basis=basis,
        )
        patched[corrupted_row, corrupted_position_tensor, :] = patched_vectors
    return patched


def _validate_single_query_batch(
    *,
    batch: dict[str, Any],
    metadata: dict[str, torch.Tensor],
    label: str,
) -> None:
    expected = len(batch["records"])
    if int(metadata["rows"].numel()) != expected:
        raise RuntimeError(f"{label} batch produced {metadata['rows'].numel()} query rows, expected {expected}.")
    expected_rows = torch.arange(expected, device=metadata["rows"].device, dtype=metadata["rows"].dtype)
    if not torch.equal(metadata["rows"], expected_rows):
        raise RuntimeError(f"{label} batch query rows are not one query per record in order.")
    if not torch.equal(metadata["query_indices"], torch.zeros_like(metadata["query_indices"])):
        raise RuntimeError(f"{label} batch query indices are not all zero.")


def _value_predictions(
    *,
    answer_logits: torch.Tensor,
    value_token_ids: torch.Tensor,
    vocab: Vocabulary,
) -> list[str]:
    value_logits = answer_logits.index_select(dim=-1, index=value_token_ids)
    predicted_value_offsets = value_logits.argmax(dim=-1)
    predicted_token_ids = value_token_ids.index_select(0, predicted_value_offsets)
    return vocab.decode([int(token_id) for token_id in predicted_token_ids.detach().cpu().tolist()])


def _margin_for_targets(
    *,
    answer_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    value_token_ids: torch.Tensor,
) -> torch.Tensor:
    return _value_margin(answer_logits, target_token_ids, value_token_ids)


def _contrast_margin(
    *,
    answer_logits: torch.Tensor,
    positive_token_ids: torch.Tensor,
    negative_token_ids: torch.Tensor,
) -> torch.Tensor:
    if positive_token_ids.shape != negative_token_ids.shape:
        raise ValueError("positive_token_ids and negative_token_ids must have the same shape.")
    row_index = torch.arange(answer_logits.size(0), device=answer_logits.device)
    return answer_logits[row_index, positive_token_ids] - answer_logits[row_index, negative_token_ids]


def _token_ids_for_values(*, values: list[str], vocab: Vocabulary, device: torch.device) -> torch.Tensor:
    token_ids: list[int] = []
    for value in values:
        if value not in vocab.token_to_id:
            raise KeyError(f"Value token {value!r} is missing from the vocabulary.")
        if value not in vocab.value_tokens:
            raise ValueError(f"Transfer token {value!r} is not a value token.")
        token_ids.append(int(vocab.token_to_id[value]))
    return torch.tensor(token_ids, device=device, dtype=torch.long)


def _float_item(tensor: torch.Tensor, index: int) -> float:
    return float(tensor[index].detach().float().cpu().item())


def _compute_causal_patch_checkpoint(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    checkpoint_step: int,
    basis: torch.Tensor | None,
    subspace_summary: dict[str, Any],
    subspace_name: str,
    rank: int | None,
    stage_name: str,
    position_role: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    min_recovery_denominator: float,
    progress_every_pairs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if progress_every_pairs < 0:
        raise ValueError("progress_every_pairs must be non-negative.")
    if min_recovery_denominator < 0.0:
        raise ValueError("min_recovery_denominator must be non-negative.")
    value_token_ids = torch.tensor(vocab.value_token_ids, device=device, dtype=torch.long)
    query_rows: list[dict[str, Any]] = []
    processed_pairs = 0

    for start_index in range(0, len(pairs), batch_size):
        pair_batch = pairs[start_index : start_index + batch_size]
        clean_records = [pair["clean_record"] for pair in pair_batch]
        corrupted_records = [pair["corrupted_record"] for pair in pair_batch]
        clean_batch = move_batch_to_device(collate_symbolic_kv(clean_records, pad_token_id), device)
        corrupted_batch = move_batch_to_device(collate_symbolic_kv(corrupted_records, pad_token_id), device)

        with torch.no_grad():
            clean_outputs = model(
                clean_batch["input_ids"],
                attention_mask=clean_batch["attention_mask"],
                return_residual_streams=True,
            )
            corrupted_outputs = model(
                corrupted_batch["input_ids"],
                attention_mask=corrupted_batch["attention_mask"],
                return_residual_streams=True,
            )
        if clean_outputs.residual_streams is None or corrupted_outputs.residual_streams is None:
            raise RuntimeError("Causal variable patching requires residual streams.")
        if stage_name not in clean_outputs.residual_streams:
            raise KeyError(f"Stage {stage_name!r} not found in clean residual streams.")
        if stage_name not in corrupted_outputs.residual_streams:
            raise KeyError(f"Stage {stage_name!r} not found in corrupted residual streams.")

        clean_answer_logits, clean_answer_targets, clean_metadata = extract_answer_logits(
            clean_outputs.logits,
            clean_batch,
        )
        corrupted_answer_logits, corrupted_answer_targets, corrupted_metadata = extract_answer_logits(
            corrupted_outputs.logits,
            corrupted_batch,
        )
        _validate_single_query_batch(batch=clean_batch, metadata=clean_metadata, label="clean")
        _validate_single_query_batch(batch=corrupted_batch, metadata=corrupted_metadata, label="corrupted")

        clean_selected = [
            _intervention_positions_for_query(
                batch=clean_batch,
                metadata=clean_metadata,
                flat_index=flat_index,
                position_role=position_role,
            )
            for flat_index in range(len(pair_batch))
        ]
        corrupted_selected = [
            _intervention_positions_for_query(
                batch=corrupted_batch,
                metadata=corrupted_metadata,
                flat_index=flat_index,
                position_role=position_role,
            )
            for flat_index in range(len(pair_batch))
        ]
        patched_stage = _transfer_stage_tensor(
            clean_stage=clean_outputs.residual_streams[stage_name],
            corrupted_stage=corrupted_outputs.residual_streams[stage_name],
            clean_selected=clean_selected,
            corrupted_selected=corrupted_selected,
            basis=basis,
        )
        with torch.no_grad():
            patched_outputs = model(
                corrupted_batch["input_ids"],
                attention_mask=corrupted_batch["attention_mask"],
                residual_patch={stage_name: patched_stage},
            )
        patched_answer_logits, patched_corrupted_targets, patched_metadata = extract_answer_logits(
            patched_outputs.logits,
            corrupted_batch,
        )
        _validate_query_metadata_match(
            baseline_metadata=corrupted_metadata,
            patched_metadata=patched_metadata,
        )
        if not torch.equal(corrupted_answer_targets.detach().cpu(), patched_corrupted_targets.detach().cpu()):
            raise RuntimeError("Patched forward changed corrupted answer targets.")

        clean_transfer_ids = _token_ids_for_values(
            values=[str(pair["clean_transfer_token"]) for pair in pair_batch],
            vocab=vocab,
            device=device,
        )
        corrupted_transfer_ids = _token_ids_for_values(
            values=[str(pair["corrupted_transfer_token"]) for pair in pair_batch],
            vocab=vocab,
            device=device,
        )
        clean_answer_ids = _token_ids_for_values(
            values=[str(pair["clean_answer_value"]) for pair in pair_batch],
            vocab=vocab,
            device=device,
        )
        corrupted_answer_ids = _token_ids_for_values(
            values=[str(pair["corrupted_answer_value"]) for pair in pair_batch],
            vocab=vocab,
            device=device,
        )
        if not torch.equal(clean_answer_targets, clean_answer_ids):
            raise RuntimeError("Clean batch answer targets disagree with pair metadata.")
        if not torch.equal(corrupted_answer_targets, corrupted_answer_ids):
            raise RuntimeError("Corrupted batch answer targets disagree with pair metadata.")

        clean_value_margin = _margin_for_targets(
            answer_logits=clean_answer_logits,
            target_token_ids=clean_answer_ids,
            value_token_ids=value_token_ids,
        )
        corrupted_value_margin = _margin_for_targets(
            answer_logits=corrupted_answer_logits,
            target_token_ids=corrupted_answer_ids,
            value_token_ids=value_token_ids,
        )
        corrupted_clean_answer_margin = _margin_for_targets(
            answer_logits=corrupted_answer_logits,
            target_token_ids=clean_answer_ids,
            value_token_ids=value_token_ids,
        )
        patched_clean_answer_margin = _margin_for_targets(
            answer_logits=patched_answer_logits,
            target_token_ids=clean_answer_ids,
            value_token_ids=value_token_ids,
        )
        patched_corrupted_answer_margin = _margin_for_targets(
            answer_logits=patched_answer_logits,
            target_token_ids=corrupted_answer_ids,
            value_token_ids=value_token_ids,
        )
        transfer_margin_clean = _contrast_margin(
            answer_logits=clean_answer_logits,
            positive_token_ids=clean_transfer_ids,
            negative_token_ids=corrupted_transfer_ids,
        )
        transfer_margin_corrupted = _contrast_margin(
            answer_logits=corrupted_answer_logits,
            positive_token_ids=clean_transfer_ids,
            negative_token_ids=corrupted_transfer_ids,
        )
        transfer_margin_patched = _contrast_margin(
            answer_logits=patched_answer_logits,
            positive_token_ids=clean_transfer_ids,
            negative_token_ids=corrupted_transfer_ids,
        )

        clean_predictions = _value_predictions(
            answer_logits=clean_answer_logits,
            value_token_ids=value_token_ids,
            vocab=vocab,
        )
        corrupted_predictions = _value_predictions(
            answer_logits=corrupted_answer_logits,
            value_token_ids=value_token_ids,
            vocab=vocab,
        )
        patched_predictions = _value_predictions(
            answer_logits=patched_answer_logits,
            value_token_ids=value_token_ids,
            vocab=vocab,
        )

        for pair_index, pair in enumerate(pair_batch):
            clean_position_list = [int(position) for position in clean_selected[pair_index][1]]
            corrupted_position_list = [int(position) for position in corrupted_selected[pair_index][1]]
            recovery_denominator = _float_item(transfer_margin_clean, pair_index) - _float_item(
                transfer_margin_corrupted,
                pair_index,
            )
            transfer_delta = _float_item(transfer_margin_patched, pair_index) - _float_item(
                transfer_margin_corrupted,
                pair_index,
            )
            recovery_defined = abs(recovery_denominator) >= min_recovery_denominator
            transfer_recovery = transfer_delta / recovery_denominator if recovery_defined else None
            clean_answer_value = str(pair["clean_answer_value"])
            corrupted_answer_value = str(pair["corrupted_answer_value"])
            query_rows.append(
                {
                    "step": checkpoint_step,
                    "pair_id": str(pair["pair_id"]),
                    "pair_type": str(pair["pair_type"]),
                    "split": str(pair["split"]),
                    "source_sample_id": str(pair["source_sample_id"]),
                    "source_query_index": int(pair["source_query_index"]),
                    "stage": stage_name,
                    "subspace_name": subspace_name,
                    "subspace_type": str(subspace_summary["subspace_type"]),
                    "head_label": subspace_summary.get("head_label"),
                    "rank": rank,
                    "position_role": position_role,
                    "clean_answer_value": clean_answer_value,
                    "corrupted_answer_value": corrupted_answer_value,
                    "clean_transfer_token": str(pair["clean_transfer_token"]),
                    "corrupted_transfer_token": str(pair["corrupted_transfer_token"]),
                    "mutation": pair["mutation"],
                    "clean_value_margin": _float_item(clean_value_margin, pair_index),
                    "corrupted_value_margin": _float_item(corrupted_value_margin, pair_index),
                    "corrupted_clean_answer_margin": _float_item(corrupted_clean_answer_margin, pair_index),
                    "patched_clean_answer_margin": _float_item(patched_clean_answer_margin, pair_index),
                    "patched_corrupted_answer_margin": _float_item(patched_corrupted_answer_margin, pair_index),
                    "transfer_margin_clean": _float_item(transfer_margin_clean, pair_index),
                    "transfer_margin_corrupted": _float_item(transfer_margin_corrupted, pair_index),
                    "transfer_margin_patched": _float_item(transfer_margin_patched, pair_index),
                    "transfer_margin_delta": transfer_delta,
                    "recovery_denominator": recovery_denominator,
                    "transfer_recovery": transfer_recovery,
                    "recovery_defined": recovery_defined,
                    "clean_prediction": clean_predictions[pair_index],
                    "corrupted_prediction": corrupted_predictions[pair_index],
                    "patched_prediction": patched_predictions[pair_index],
                    "clean_predicts_clean_answer": clean_predictions[pair_index] == clean_answer_value,
                    "corrupted_predicts_corrupted_answer": corrupted_predictions[pair_index] == corrupted_answer_value,
                    "patched_predicts_clean_answer": patched_predictions[pair_index] == clean_answer_value,
                    "patched_predicts_corrupted_answer": patched_predictions[pair_index] == corrupted_answer_value,
                    "clean_selected_positions": clean_position_list,
                    "corrupted_selected_positions": corrupted_position_list,
                    "selected_position_count": len(corrupted_position_list),
                }
            )
        processed_pairs += len(pair_batch)
        if progress_every_pairs and processed_pairs % progress_every_pairs == 0:
            print(
                "[causal-variable-patch] "
                f"step={checkpoint_step} processed_pairs={processed_pairs}/{len(pairs)}",
                flush=True,
            )

    aggregate_rows = _aggregate_causal_patch_rows(
        query_rows=query_rows,
        step=checkpoint_step,
        subspace_summary=subspace_summary,
        subspace_name=subspace_name,
        rank=rank,
        stage_name=stage_name,
        position_role=position_role,
    )
    return aggregate_rows, query_rows


def _new_causal_patch_accumulator() -> dict[str, float]:
    return {
        "clean_value_margin_sum": 0.0,
        "corrupted_value_margin_sum": 0.0,
        "corrupted_clean_answer_margin_sum": 0.0,
        "patched_clean_answer_margin_sum": 0.0,
        "patched_corrupted_answer_margin_sum": 0.0,
        "transfer_margin_clean_sum": 0.0,
        "transfer_margin_corrupted_sum": 0.0,
        "transfer_margin_patched_sum": 0.0,
        "transfer_margin_delta_sum": 0.0,
        "transfer_margin_delta_abs_sum": 0.0,
        "recovery_sum": 0.0,
        "recovery_abs_sum": 0.0,
        "recovery_positive_count": 0.0,
        "recovery_negative_count": 0.0,
        "recovery_defined_count": 0.0,
        "clean_predicts_clean_answer_count": 0.0,
        "corrupted_predicts_corrupted_answer_count": 0.0,
        "patched_predicts_clean_answer_count": 0.0,
        "patched_predicts_corrupted_answer_count": 0.0,
        "total": 0.0,
    }


def _accumulate_causal_patch_row(accumulator: dict[str, float], row: dict[str, Any]) -> None:
    accumulator["clean_value_margin_sum"] += float(row["clean_value_margin"])
    accumulator["corrupted_value_margin_sum"] += float(row["corrupted_value_margin"])
    accumulator["corrupted_clean_answer_margin_sum"] += float(row["corrupted_clean_answer_margin"])
    accumulator["patched_clean_answer_margin_sum"] += float(row["patched_clean_answer_margin"])
    accumulator["patched_corrupted_answer_margin_sum"] += float(row["patched_corrupted_answer_margin"])
    accumulator["transfer_margin_clean_sum"] += float(row["transfer_margin_clean"])
    accumulator["transfer_margin_corrupted_sum"] += float(row["transfer_margin_corrupted"])
    accumulator["transfer_margin_patched_sum"] += float(row["transfer_margin_patched"])
    accumulator["transfer_margin_delta_sum"] += float(row["transfer_margin_delta"])
    accumulator["transfer_margin_delta_abs_sum"] += abs(float(row["transfer_margin_delta"]))
    if bool(row["recovery_defined"]):
        recovery = float(row["transfer_recovery"])
        accumulator["recovery_sum"] += recovery
        accumulator["recovery_abs_sum"] += abs(recovery)
        if recovery > 0.0:
            accumulator["recovery_positive_count"] += 1.0
        if recovery < 0.0:
            accumulator["recovery_negative_count"] += 1.0
        accumulator["recovery_defined_count"] += 1.0
    if bool(row["clean_predicts_clean_answer"]):
        accumulator["clean_predicts_clean_answer_count"] += 1.0
    if bool(row["corrupted_predicts_corrupted_answer"]):
        accumulator["corrupted_predicts_corrupted_answer_count"] += 1.0
    if bool(row["patched_predicts_clean_answer"]):
        accumulator["patched_predicts_clean_answer_count"] += 1.0
    if bool(row["patched_predicts_corrupted_answer"]):
        accumulator["patched_predicts_corrupted_answer_count"] += 1.0
    accumulator["total"] += 1.0


def _causal_patch_summary(accumulator: dict[str, float]) -> dict[str, float | int | None]:
    total = int(accumulator["total"])
    if total <= 0:
        raise RuntimeError("Cannot summarize empty causal patch accumulator.")
    recovery_defined_count = int(accumulator["recovery_defined_count"])
    recovery_mean = None
    recovery_abs_mean = None
    recovery_positive_fraction = None
    recovery_negative_fraction = None
    if recovery_defined_count > 0:
        recovery_mean = float(accumulator["recovery_sum"]) / recovery_defined_count
        recovery_abs_mean = float(accumulator["recovery_abs_sum"]) / recovery_defined_count
        recovery_positive_fraction = float(accumulator["recovery_positive_count"]) / recovery_defined_count
        recovery_negative_fraction = float(accumulator["recovery_negative_count"]) / recovery_defined_count
    return {
        "num_pairs": total,
        "clean_value_margin_mean": float(accumulator["clean_value_margin_sum"]) / total,
        "corrupted_value_margin_mean": float(accumulator["corrupted_value_margin_sum"]) / total,
        "corrupted_clean_answer_margin_mean": float(accumulator["corrupted_clean_answer_margin_sum"]) / total,
        "patched_clean_answer_margin_mean": float(accumulator["patched_clean_answer_margin_sum"]) / total,
        "patched_corrupted_answer_margin_mean": float(accumulator["patched_corrupted_answer_margin_sum"]) / total,
        "transfer_margin_clean_mean": float(accumulator["transfer_margin_clean_sum"]) / total,
        "transfer_margin_corrupted_mean": float(accumulator["transfer_margin_corrupted_sum"]) / total,
        "transfer_margin_patched_mean": float(accumulator["transfer_margin_patched_sum"]) / total,
        "transfer_margin_delta_mean": float(accumulator["transfer_margin_delta_sum"]) / total,
        "transfer_margin_delta_abs_mean": float(accumulator["transfer_margin_delta_abs_sum"]) / total,
        "transfer_recovery_mean": recovery_mean,
        "transfer_recovery_abs_mean": recovery_abs_mean,
        "transfer_recovery_positive_fraction": recovery_positive_fraction,
        "transfer_recovery_negative_fraction": recovery_negative_fraction,
        "recovery_defined_fraction": recovery_defined_count / total,
        "clean_predicts_clean_answer_fraction": float(accumulator["clean_predicts_clean_answer_count"]) / total,
        "corrupted_predicts_corrupted_answer_fraction": (
            float(accumulator["corrupted_predicts_corrupted_answer_count"]) / total
        ),
        "patched_predicts_clean_answer_fraction": float(accumulator["patched_predicts_clean_answer_count"]) / total,
        "patched_predicts_corrupted_answer_fraction": (
            float(accumulator["patched_predicts_corrupted_answer_count"]) / total
        ),
    }


def _aggregate_causal_patch_rows(
    *,
    query_rows: list[dict[str, Any]],
    step: int,
    subspace_summary: dict[str, Any],
    subspace_name: str,
    rank: int | None,
    stage_name: str,
    position_role: str,
) -> list[dict[str, Any]]:
    accumulators: dict[tuple[str, str], dict[str, float]] = {}
    for row in query_rows:
        keys = [
            (str(row["split"]), str(row["pair_type"])),
            ("__all__", str(row["pair_type"])),
            (str(row["split"]), "__all__"),
            ("__all__", "__all__"),
        ]
        for key in keys:
            if key not in accumulators:
                accumulators[key] = _new_causal_patch_accumulator()
            _accumulate_causal_patch_row(accumulators[key], row)
    rows: list[dict[str, Any]] = []
    for (split_name, pair_type), accumulator in sorted(accumulators.items()):
        rows.append(
            {
                "step": step,
                "split": split_name,
                "pair_type": pair_type,
                "subspace_name": subspace_name,
                "subspace_type": str(subspace_summary["subspace_type"]),
                "head_label": subspace_summary.get("head_label"),
                "rank": rank,
                "stage": stage_name,
                "position_role": position_role,
                **_causal_patch_summary(accumulator),
            }
        )
    return rows


def _summarize_causal_patch_report(
    *,
    aggregate_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not aggregate_rows:
        raise ValueError("Cannot summarize causal variable patch without aggregate rows.")
    final_step = max(int(row["step"]) for row in aggregate_rows)
    final_aggregate = [row for row in aggregate_rows if int(row["step"]) == final_step]
    final_query_rows = [row for row in query_rows if int(row["step"]) == final_step]
    final_all = [
        row
        for row in final_aggregate
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    if len(final_all) != 1:
        raise RuntimeError(f"Expected one final all/all aggregate row, got {len(final_all)}.")
    top_recoveries = [
        row
        for row in sorted(
            final_query_rows,
            key=lambda item: float(item["transfer_recovery"]) if item["transfer_recovery"] is not None else -math.inf,
            reverse=True,
        )
        if row["transfer_recovery"] is not None
    ][:16]
    bottom_recoveries = [
        row
        for row in sorted(
            final_query_rows,
            key=lambda item: float(item["transfer_recovery"]) if item["transfer_recovery"] is not None else math.inf,
        )
        if row["transfer_recovery"] is not None
    ][:16]
    return {
        "num_checkpoints": len({int(row["step"]) for row in aggregate_rows}),
        "steps": sorted({int(row["step"]) for row in aggregate_rows}),
        "final_step": final_step,
        "final_all": final_all[0],
        "final_by_pair_type": sorted(
            [
                row
                for row in final_aggregate
                if str(row["split"]) == "__all__" and str(row["pair_type"]) != "__all__"
            ],
            key=lambda row: str(row["pair_type"]),
        ),
        "final_by_split_and_pair_type": sorted(
            final_aggregate,
            key=lambda row: (str(row["split"]), str(row["pair_type"])),
        ),
        "top_final_patch_recoveries": top_recoveries,
        "bottom_final_patch_recoveries": bottom_recoveries,
    }


def _plot_causal_patch_recovery_by_pair_type(
    *,
    aggregate_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    final_step = max(int(row["step"]) for row in aggregate_rows)
    rows = [
        row
        for row in aggregate_rows
        if int(row["step"]) == final_step
        and str(row["split"]) == "__all__"
        and str(row["pair_type"]) != "__all__"
        and row["transfer_recovery_mean"] is not None
    ]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    rows = sorted(rows, key=lambda row: str(row["pair_type"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [str(row["pair_type"]) for row in rows],
        [float(row["transfer_recovery_mean"]) for row in rows],
        color="#376f8f",
    )
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.axhline(1.0, color="#777777", linewidth=1.0, linestyle=":")
    ax.set_title(f"Patch recovery by pair type at step {final_step}")
    ax.set_xlabel("pair type")
    ax.set_ylabel("mean transfer recovery")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_causal_patch_recovery_histogram(
    *,
    query_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    final_step = max(int(row["step"]) for row in query_rows)
    rows = [
        row
        for row in query_rows
        if int(row["step"]) == final_step and row["transfer_recovery"] is not None
    ]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist([float(row["transfer_recovery"]) for row in rows], bins=40, color="#6f8f37")
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.axvline(1.0, color="#777777", linewidth=1.0, linestyle=":")
    ax.set_title(f"Patch recovery distribution at step {final_step}")
    ax.set_xlabel("transfer recovery")
    ax.set_ylabel("pair count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_causal_patch_margin_scatter(
    *,
    query_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    final_step = max(int(row["step"]) for row in query_rows)
    rows = [row for row in query_rows if int(row["step"]) == final_step]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    pair_types = sorted({str(row["pair_type"]) for row in rows})
    colors = {
        pair_type: color
        for pair_type, color in zip(pair_types, ["#376f8f", "#8f6237", "#6f8f37", "#8f374a"], strict=False)
    }
    fig, ax = plt.subplots(figsize=(7, 7))
    for pair_type in pair_types:
        typed_rows = [row for row in rows if str(row["pair_type"]) == pair_type]
        ax.scatter(
            [float(row["transfer_margin_corrupted"]) for row in typed_rows],
            [float(row["transfer_margin_patched"]) for row in typed_rows],
            s=18,
            alpha=0.7,
            label=pair_type,
            color=colors[pair_type],
        )
    all_values = [
        float(row["transfer_margin_corrupted"])
        for row in rows
    ] + [
        float(row["transfer_margin_patched"])
        for row in rows
    ]
    lower = min(all_values)
    upper = max(all_values)
    ax.plot([lower, upper], [lower, upper], color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title(f"Corrupted vs patched transfer margin at step {final_step}")
    ax.set_xlabel("corrupted transfer margin")
    ax.set_ylabel("patched transfer margin")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_causal_patch_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    final_all = summary["final_all"]
    lines = [
        "# Causal Variable Patch",
        "",
        "## Calculation",
        "",
        "For a clean/corrupted controlled pair, the tool patches the selected residual content from clean into corrupted.",
        "",
        "For full residual patching:",
        "",
        "```text",
        "z_patched = z_clean",
        "```",
        "",
        "For a selected subspace with orthonormal basis `B`:",
        "",
        "```text",
        "z_patched = z_corrupted - (z_corrupted B) B^T + (z_clean B) B^T",
        "```",
        "",
        "Transfer recovery is measured with the clean-vs-corrupted transfer-token contrast:",
        "",
        "```text",
        "recovery = (margin_patched - margin_corrupted) / (margin_clean - margin_corrupted)",
        "margin = logit(clean_transfer_token) - logit(corrupted_transfer_token)",
        "```",
        "",
        "For query_key/support_value/recency pairs, the transfer tokens are the competing answer values. For distractor pairs, the correct answer is fixed and the transfer tokens are the changed distractor values.",
        "",
        "## Patch",
        "",
        f"- subspace: `{report['subspace']['subspace_name']}`",
        f"- rank: `{report['rank']}`",
        f"- stage: `{report['stage']}`",
        f"- position role: `{report['position_role']}`",
        f"- device: `{report['device']}`",
        "",
        "## Pair Construction",
        "",
    ]
    pair_construction = report["pair_construction"]
    for pair_type, count in pair_construction["constructed_counts"].items():
        lines.append(f"- {pair_type}: `{count}` pairs")
    lines.extend(
        [
            "",
            "## Final Aggregate By Pair Type",
            "",
            "| pair type | pairs | clean margin | corrupted margin | patched margin | recovery | patched clean-answer frac | recovery defined frac |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["final_by_pair_type"]:
        recovery = row["transfer_recovery_mean"]
        recovery_text = "" if recovery is None else f"{float(recovery):.6f}"
        lines.append(
            "| {pair_type} | {pairs} | {clean:.6f} | {corrupt:.6f} | {patched:.6f} | {recovery} | {pclean:.3f} | {defined:.3f} |".format(
                pair_type=row["pair_type"],
                pairs=int(row["num_pairs"]),
                clean=float(row["transfer_margin_clean_mean"]),
                corrupt=float(row["transfer_margin_corrupted_mean"]),
                patched=float(row["transfer_margin_patched_mean"]),
                recovery=recovery_text,
                pclean=float(row["patched_predicts_clean_answer_fraction"]),
                defined=float(row["recovery_defined_fraction"]),
            )
        )
    final_recovery = final_all["transfer_recovery_mean"]
    final_recovery_text = "undefined" if final_recovery is None else f"{float(final_recovery):.6f}"
    lines.extend(
        [
            "",
            "## Final All-Pair Result",
            "",
            f"- transfer recovery: `{final_recovery_text}`",
            f"- transfer-margin delta: `{float(final_all['transfer_margin_delta_mean']):.6f}`",
            f"- patched predicts clean answer fraction: `{float(final_all['patched_predicts_clean_answer_fraction']):.3f}`",
            f"- recovery defined fraction: `{float(final_all['recovery_defined_fraction']):.3f}`",
            "",
            "## Raw Outputs",
            "",
            f"- aggregate rows: `{report['aggregate_rows_path']}`",
            f"- query rows: `{report['query_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
            "",
            "## Plots",
            "",
        ]
    )
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_causal_variable_patch(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    stage_name: str,
    subspace_name: str,
    position_role: str,
    pair_types: list[str],
    rank: int | None = None,
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    head_layer: int | None = None,
    head: int | None = None,
    max_pairs_per_type: int = 128,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    min_recovery_denominator: float = 1.0e-6,
    progress_every_pairs: int = 64,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Path]]:
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    model = build_model(spec.model, len(vocab.tokens), device)
    _validate_geometry_stage(model=model, stage_name=stage_name)
    if position_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported position role {position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if not pair_types:
        raise ValueError("pair_types must not be empty.")
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Causal variable patch constructed no pairs.")

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows_path = output_dir / "causal_variable_patch_rows.jsonl"
    query_rows_path = output_dir / "causal_variable_patch_query_rows.jsonl"
    pair_rows_path = output_dir / "causal_variable_patch_pairs.jsonl"
    progress_path = output_dir / "causal_variable_patch_progress.json"
    for partial_path in (aggregate_rows_path, query_rows_path, pair_rows_path, progress_path):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])

    print(
        "[causal-variable-patch] "
        f"checkpoints={len(checkpoints)} pairs={len(pairs)} pair_types={pair_types} "
        f"device={device_name} subspace={subspace_name} rank={rank} stage={stage_name} role={position_role}",
        flush=True,
    )

    all_aggregate_rows: list[dict[str, Any]] = []
    all_query_rows: list[dict[str, Any]] = []
    final_subspace_summary: dict[str, Any] | None = None
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if step != path_step:
            raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={step} path={path_step}")
        basis, subspace_summary = _resolve_causal_patch_basis(
            model=model,
            vocab=vocab,
            subspace_name=subspace_name,
            rank=rank,
            head_layer=head_layer,
            head=head,
            device=device,
        )
        final_subspace_summary = subspace_summary
        print(
            f"[causal-variable-patch] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        aggregate_rows, query_rows = _compute_causal_patch_checkpoint(
            model=model,
            pairs=pairs,
            vocab=vocab,
            checkpoint_step=step,
            basis=basis,
            subspace_summary=subspace_summary,
            subspace_name=subspace_name,
            rank=rank,
            stage_name=stage_name,
            position_role=position_role,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            min_recovery_denominator=min_recovery_denominator,
            progress_every_pairs=progress_every_pairs,
        )
        for row in aggregate_rows:
            append_jsonl(aggregate_rows_path, row)
        for row in query_rows:
            append_jsonl(query_rows_path, row)
        all_aggregate_rows.extend(aggregate_rows)
        all_query_rows.extend(query_rows)
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_checkpoints": checkpoint_index,
                "total_checkpoints": len(checkpoints),
                "last_completed_step": step,
                "aggregate_rows_path": str(aggregate_rows_path),
                "query_rows_path": str(query_rows_path),
                "pair_rows_path": str(pair_rows_path),
            },
        )
        all_row = next(
            row
            for row in aggregate_rows
            if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
        )
        recovery = all_row["transfer_recovery_mean"]
        recovery_text = "undefined" if recovery is None else f"{float(recovery):.6f}"
        print(
            "[causal-variable-patch] finished "
            f"step={step} transfer_delta={float(all_row['transfer_margin_delta_mean']):.6f} "
            f"recovery={recovery_text} patched_clean_answer_fraction="
            f"{float(all_row['patched_predicts_clean_answer_fraction']):.3f}",
            flush=True,
        )

    if final_subspace_summary is None:
        raise RuntimeError("No checkpoints were processed for causal variable patch.")
    summary = _summarize_causal_patch_report(
        aggregate_rows=all_aggregate_rows,
        query_rows=all_query_rows,
    )
    report_path = output_dir / "causal_variable_patch_report.json"
    markdown_path = output_dir / "causal_variable_patch_report.md"
    plot_paths: dict[str, Path] = {}
    recovery_plot = _plot_causal_patch_recovery_by_pair_type(
        aggregate_rows=all_aggregate_rows,
        output_path=output_dir / "causal_variable_patch_recovery_by_pair_type.svg",
    )
    if recovery_plot is not None:
        plot_paths["recovery_by_pair_type"] = recovery_plot
    histogram_plot = _plot_causal_patch_recovery_histogram(
        query_rows=all_query_rows,
        output_path=output_dir / "causal_variable_patch_recovery_histogram.svg",
    )
    if histogram_plot is not None:
        plot_paths["recovery_histogram"] = histogram_plot
    scatter_plot = _plot_causal_patch_margin_scatter(
        query_rows=all_query_rows,
        output_path=output_dir / "causal_variable_patch_margin_scatter.svg",
    )
    if scatter_plot is not None:
        plot_paths["margin_scatter"] = scatter_plot

    report = {
        "schema_version": CAUSAL_VARIABLE_PATCH_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "stage": stage_name,
        "subspace": final_subspace_summary,
        "subspace_name": subspace_name,
        "rank": rank,
        "position_role": position_role,
        "pair_types": pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "min_recovery_denominator": min_recovery_denominator,
        "progress_every_pairs": progress_every_pairs,
        "calculation": {
            "full_residual_patch": "z_patched = z_clean at selected stage/positions",
            "subspace_patch": "z_patched = z_corrupted - (z_corrupted B) B^T + (z_clean B) B^T",
            "transfer_margin": "logit(clean_transfer_token) - logit(corrupted_transfer_token)",
            "transfer_recovery": "(transfer_margin_patched - transfer_margin_corrupted) / (transfer_margin_clean - transfer_margin_corrupted)",
            "distractor_control": "correct answer is fixed; transfer tokens are changed distractor values",
        },
        "pair_construction": pair_construction,
        "aggregate_rows_path": str(aggregate_rows_path),
        "query_rows_path": str(query_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_causal_patch_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_checkpoints": len(checkpoints),
            "total_checkpoints": len(checkpoints),
            "last_completed_step": int(summary["final_step"]),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "aggregate_rows_path": str(aggregate_rows_path),
            "query_rows_path": str(query_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[causal-variable-patch] complete report={report_path} rows={aggregate_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, aggregate_rows_path, query_rows_path, pair_rows_path, plot_paths


ROUTE_GRADIENT_LOSS_SIDES = ["clean", "corrupted", "both"]


def _parameter_gradients(
    *,
    model: torch.nn.Module,
    require_all: bool,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    gradients: dict[str, torch.Tensor] = {}
    zero_gradient_parameter_names: list[str] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter.grad is None:
            if require_all:
                raise RuntimeError(f"Parameter has no gradient after backward: {name}")
            gradients[name] = torch.zeros_like(parameter.detach(), device=torch.device("cpu"), dtype=torch.float32)
            zero_gradient_parameter_names.append(name)
            continue
        gradients[name] = parameter.grad.detach().cpu().float()
    return gradients, zero_gradient_parameter_names


def _gradient_dot_summary(
    *,
    left_gradients: dict[str, torch.Tensor],
    right_gradients: dict[str, torch.Tensor],
    label: str,
) -> dict[str, float | int | None]:
    left_keys = set(left_gradients)
    right_keys = set(right_gradients)
    if left_keys != right_keys:
        missing_right = sorted(left_keys - right_keys)
        extra_right = sorted(right_keys - left_keys)
        raise ValueError(
            f"Gradient keys differ for {label}: missing_right={missing_right} extra_right={extra_right}"
        )
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    num_parameters = 0
    for key in sorted(left_keys):
        left = left_gradients[key].float().reshape(-1)
        right = right_gradients[key].float().reshape(-1)
        if left.shape != right.shape:
            raise ValueError(f"Gradient shape mismatch for {label} key {key}: {tuple(left.shape)} vs {tuple(right.shape)}")
        dot += float(torch.dot(left, right).item())
        left_sq += float(torch.dot(left, left).item())
        right_sq += float(torch.dot(right, right).item())
        num_parameters += int(left.numel())
    left_norm = left_sq ** 0.5
    right_norm = right_sq ** 0.5
    return {
        "num_parameters": num_parameters,
        "dot": dot,
        "left_l2_norm": left_norm,
        "right_l2_norm": right_norm,
        "cosine": _safe_ratio(dot, left_norm * right_norm),
    }


ROUTE_GRADIENT_DECOMPOSITION_MODES = [
    "parameter_tensors",
    "module_blocks",
    "attention_projections",
    "attention_heads",
    "mlp_neurons",
]


@dataclass(frozen=True)
class _GradientSelection:
    parameter_name: str
    selector: tuple[Any, ...] | None
    selector_label: str


@dataclass(frozen=True)
class _RouteGradientDecompositionGroup:
    group_id: str
    group_kind: str
    component_type: str
    partition_name: str
    layer: int | None
    head: int | None
    projection: str | None
    neuron: int | None
    selections: tuple[_GradientSelection, ...]
    notes: tuple[str, ...] = ()


def _resolve_route_gradient_decomposition_modes(modes: list[str] | None) -> list[str]:
    if modes is None:
        return list(ROUTE_GRADIENT_DECOMPOSITION_MODES)
    if not modes:
        raise ValueError("decomposition_modes must not be empty when provided.")
    resolved: list[str] = []
    for mode in modes:
        if mode not in ROUTE_GRADIENT_DECOMPOSITION_MODES:
            raise ValueError(
                f"Unsupported route gradient decomposition mode {mode!r}; "
                f"expected one of {ROUTE_GRADIENT_DECOMPOSITION_MODES}."
            )
        if mode not in resolved:
            resolved.append(mode)
    return resolved


def _selector_numel(parameter: torch.nn.Parameter, selector: tuple[Any, ...] | None) -> int:
    selected = parameter.detach() if selector is None else parameter.detach()[selector]
    numel = int(selected.numel())
    if numel <= 0:
        raise RuntimeError("Gradient decomposition selector produced an empty parameter slice.")
    return numel


def _selection_summary(
    *,
    model_parameters: dict[str, torch.nn.Parameter],
    selection: _GradientSelection,
) -> dict[str, Any]:
    if selection.parameter_name not in model_parameters:
        raise KeyError(f"Gradient decomposition parameter not found: {selection.parameter_name}")
    parameter = model_parameters[selection.parameter_name]
    return {
        "parameter_name": selection.parameter_name,
        "parameter_shape": [int(dim) for dim in parameter.shape],
        "selector": selection.selector_label,
        "num_parameters": _selector_numel(parameter, selection.selector),
    }


def _group_num_parameters(
    *,
    model_parameters: dict[str, torch.nn.Parameter],
    group: _RouteGradientDecompositionGroup,
) -> int:
    if not group.selections:
        raise RuntimeError(f"Gradient decomposition group has no selections: {group.group_id}")
    return sum(
        _selector_numel(model_parameters[selection.parameter_name], selection.selector)
        for selection in group.selections
    )


def _group_metadata(
    *,
    model_parameters: dict[str, torch.nn.Parameter],
    group: _RouteGradientDecompositionGroup,
) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "group_kind": group.group_kind,
        "component_type": group.component_type,
        "partition_name": group.partition_name,
        "layer": group.layer,
        "head": group.head,
        "projection": group.projection,
        "neuron": group.neuron,
        "num_selected_parameters": _group_num_parameters(model_parameters=model_parameters, group=group),
        "selection_count": len(group.selections),
        "selections": [
            _selection_summary(model_parameters=model_parameters, selection=selection)
            for selection in group.selections
        ],
        "notes": list(group.notes),
    }


def _whole_parameter_selection(parameter_name: str) -> _GradientSelection:
    return _GradientSelection(
        parameter_name=parameter_name,
        selector=None,
        selector_label=":",
    )


def _row_block_selection(parameter_name: str, start: int, end: int) -> _GradientSelection:
    return _GradientSelection(
        parameter_name=parameter_name,
        selector=(slice(start, end), slice(None)),
        selector_label=f"[{start}:{end}, :]",
    )


def _vector_block_selection(parameter_name: str, start: int, end: int) -> _GradientSelection:
    return _GradientSelection(
        parameter_name=parameter_name,
        selector=(slice(start, end),),
        selector_label=f"[{start}:{end}]",
    )


def _column_block_selection(parameter_name: str, start: int, end: int) -> _GradientSelection:
    return _GradientSelection(
        parameter_name=parameter_name,
        selector=(slice(None), slice(start, end)),
        selector_label=f"[:, {start}:{end}]",
    )


def _single_row_selection(parameter_name: str, index: int) -> _GradientSelection:
    return _GradientSelection(
        parameter_name=parameter_name,
        selector=(index, slice(None)),
        selector_label=f"[{index}, :]",
    )


def _single_vector_selection(parameter_name: str, index: int) -> _GradientSelection:
    return _GradientSelection(
        parameter_name=parameter_name,
        selector=(index,),
        selector_label=f"[{index}]",
    )


def _single_column_selection(parameter_name: str, index: int) -> _GradientSelection:
    return _GradientSelection(
        parameter_name=parameter_name,
        selector=(slice(None), index),
        selector_label=f"[:, {index}]",
    )


def _require_model_parameter(
    *,
    model_parameters: dict[str, torch.nn.Parameter],
    parameter_name: str,
) -> None:
    if parameter_name not in model_parameters:
        raise KeyError(f"Expected model parameter is missing: {parameter_name}")


def _require_model_parameters(
    *,
    model_parameters: dict[str, torch.nn.Parameter],
    parameter_names: list[str],
) -> None:
    for parameter_name in parameter_names:
        _require_model_parameter(model_parameters=model_parameters, parameter_name=parameter_name)


def _shared_parameter_name_sets(model: torch.nn.Module) -> list[list[str]]:
    names_by_identity: dict[int, list[str]] = defaultdict(list)
    for name, parameter in model.named_parameters(remove_duplicate=False):
        names_by_identity[id(parameter)].append(name)
    return [names for names in names_by_identity.values() if len(names) > 1]


def _build_route_gradient_decomposition_groups(
    *,
    model: torch.nn.Module,
    decomposition_modes: list[str],
) -> tuple[list[_RouteGradientDecompositionGroup], dict[str, Any]]:
    model_parameters = dict(model.named_parameters(remove_duplicate=False))
    if not model_parameters:
        raise RuntimeError("Model exposes no named parameters for route-gradient decomposition.")

    groups: list[_RouteGradientDecompositionGroup] = []
    group_ids: set[str] = set()

    def add_group(group: _RouteGradientDecompositionGroup) -> None:
        if group.group_id in group_ids:
            raise RuntimeError(f"Duplicate route-gradient decomposition group id: {group.group_id}")
        _group_num_parameters(model_parameters=model_parameters, group=group)
        group_ids.add(group.group_id)
        groups.append(group)

    add_group(
        _RouteGradientDecompositionGroup(
            group_id="global:all_named_parameters",
            group_kind="global_all",
            component_type="global",
            partition_name="global_all",
            layer=None,
            head=None,
            projection=None,
            neuron=None,
            selections=tuple(_whole_parameter_selection(name) for name in model_parameters),
            notes=("Uses model.named_parameters(remove_duplicate=False), matching candidate-route-gradient-selection.",),
        )
    )

    if "parameter_tensors" in decomposition_modes:
        for name in sorted(model_parameters):
            add_group(
                _RouteGradientDecompositionGroup(
                    group_id=f"parameter:{name}",
                    group_kind="parameter_tensor",
                    component_type="parameter",
                    partition_name="parameter_tensors",
                    layer=None,
                    head=None,
                    projection=None,
                    neuron=None,
                    selections=(_whole_parameter_selection(name),),
                )
            )

    if "module_blocks" in decomposition_modes:
        for group_id, component_type, names in [
            ("module:token_embedding", "embedding", ["token_embedding.weight"]),
            ("module:position_embedding", "embedding", ["position_embedding.weight"]),
            ("module:final_norm", "layernorm", ["final_norm.weight", "final_norm.bias"]),
            ("module:lm_head", "unembedding", ["lm_head.weight"]),
        ]:
            _require_model_parameters(model_parameters=model_parameters, parameter_names=names)
            add_group(
                _RouteGradientDecompositionGroup(
                    group_id=group_id,
                    group_kind="module_block",
                    component_type=component_type,
                    partition_name="module_blocks",
                    layer=None,
                    head=None,
                    projection=None,
                    neuron=None,
                    selections=tuple(_whole_parameter_selection(name) for name in names),
                )
            )
        for layer_index, _ in enumerate(model.blocks):
            ln_1_names = [f"blocks.{layer_index}.ln_1.weight", f"blocks.{layer_index}.ln_1.bias"]
            attention_names = [
                f"blocks.{layer_index}.attn.q_proj.weight",
                f"blocks.{layer_index}.attn.q_proj.bias",
                f"blocks.{layer_index}.attn.k_proj.weight",
                f"blocks.{layer_index}.attn.k_proj.bias",
                f"blocks.{layer_index}.attn.v_proj.weight",
                f"blocks.{layer_index}.attn.v_proj.bias",
                f"blocks.{layer_index}.attn.out_proj.weight",
                f"blocks.{layer_index}.attn.out_proj.bias",
            ]
            ln_2_names = [f"blocks.{layer_index}.ln_2.weight", f"blocks.{layer_index}.ln_2.bias"]
            mlp_names = [
                f"blocks.{layer_index}.ff.fc_in.weight",
                f"blocks.{layer_index}.ff.fc_in.bias",
                f"blocks.{layer_index}.ff.fc_out.weight",
                f"blocks.{layer_index}.ff.fc_out.bias",
            ]
            for group_id, component_type, names in [
                (f"module:L{layer_index}.ln_1", "layernorm", ln_1_names),
                (f"module:L{layer_index}.attention", "attention", attention_names),
                (f"module:L{layer_index}.ln_2", "layernorm", ln_2_names),
                (f"module:L{layer_index}.mlp", "mlp", mlp_names),
            ]:
                _require_model_parameters(model_parameters=model_parameters, parameter_names=names)
                add_group(
                    _RouteGradientDecompositionGroup(
                        group_id=group_id,
                        group_kind="module_block",
                        component_type=component_type,
                        partition_name="module_blocks",
                        layer=layer_index,
                        head=None,
                        projection=None,
                        neuron=None,
                        selections=tuple(_whole_parameter_selection(name) for name in names),
                    )
                )

    if "attention_projections" in decomposition_modes:
        for layer_index, _ in enumerate(model.blocks):
            for projection_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                names = [
                    f"blocks.{layer_index}.attn.{projection_name}.weight",
                    f"blocks.{layer_index}.attn.{projection_name}.bias",
                ]
                _require_model_parameters(model_parameters=model_parameters, parameter_names=names)
                add_group(
                    _RouteGradientDecompositionGroup(
                        group_id=f"attention_projection:L{layer_index}.{projection_name}",
                        group_kind="attention_projection",
                        component_type="attention",
                        partition_name="attention_projections",
                        layer=layer_index,
                        head=None,
                        projection=projection_name,
                        neuron=None,
                        selections=tuple(_whole_parameter_selection(name) for name in names),
                    )
                )

    if "attention_heads" in decomposition_modes:
        for layer_index, block in enumerate(model.blocks):
            n_heads = int(block.attn.n_heads)
            head_dim = int(block.attn.head_dim)
            for head_index in range(n_heads):
                start = head_index * head_dim
                end = (head_index + 1) * head_dim
                q_weight = f"blocks.{layer_index}.attn.q_proj.weight"
                q_bias = f"blocks.{layer_index}.attn.q_proj.bias"
                k_weight = f"blocks.{layer_index}.attn.k_proj.weight"
                k_bias = f"blocks.{layer_index}.attn.k_proj.bias"
                v_weight = f"blocks.{layer_index}.attn.v_proj.weight"
                v_bias = f"blocks.{layer_index}.attn.v_proj.bias"
                out_weight = f"blocks.{layer_index}.attn.out_proj.weight"
                _require_model_parameters(
                    model_parameters=model_parameters,
                    parameter_names=[q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight],
                )
                projection_groups = [
                    (
                        "q_proj",
                        (
                            _row_block_selection(q_weight, start, end),
                            _vector_block_selection(q_bias, start, end),
                        ),
                        (),
                    ),
                    (
                        "k_proj",
                        (
                            _row_block_selection(k_weight, start, end),
                            _vector_block_selection(k_bias, start, end),
                        ),
                        (),
                    ),
                    (
                        "v_proj",
                        (
                            _row_block_selection(v_weight, start, end),
                            _vector_block_selection(v_bias, start, end),
                        ),
                        (),
                    ),
                    (
                        "out_proj",
                        (_column_block_selection(out_weight, start, end),),
                        ("out_proj.bias is not assigned to a head because it is shared after head concatenation.",),
                    ),
                ]
                qkvo_selections: list[_GradientSelection] = []
                qkvo_notes: list[str] = []
                for projection_name, selections, notes in projection_groups:
                    qkvo_selections.extend(selections)
                    qkvo_notes.extend(notes)
                    add_group(
                        _RouteGradientDecompositionGroup(
                            group_id=f"attention_head_projection:L{layer_index}H{head_index}.{projection_name}",
                            group_kind="attention_head_projection",
                            component_type="attention",
                            partition_name="attention_head_projections",
                            layer=layer_index,
                            head=head_index,
                            projection=projection_name,
                            neuron=None,
                            selections=selections,
                            notes=notes,
                        )
                    )
                add_group(
                    _RouteGradientDecompositionGroup(
                        group_id=f"attention_head:L{layer_index}H{head_index}.qkvo",
                        group_kind="attention_head",
                        component_type="attention",
                        partition_name="attention_heads",
                        layer=layer_index,
                        head=head_index,
                        projection="qkvo",
                        neuron=None,
                        selections=tuple(qkvo_selections),
                        notes=tuple(sorted(set(qkvo_notes))),
                    )
                )

    if "mlp_neurons" in decomposition_modes:
        for layer_index, block in enumerate(model.blocks):
            d_ff = int(block.ff.d_ff)
            fc_in_weight = f"blocks.{layer_index}.ff.fc_in.weight"
            fc_in_bias = f"blocks.{layer_index}.ff.fc_in.bias"
            fc_out_weight = f"blocks.{layer_index}.ff.fc_out.weight"
            _require_model_parameters(
                model_parameters=model_parameters,
                parameter_names=[fc_in_weight, fc_in_bias, fc_out_weight],
            )
            for neuron_index in range(d_ff):
                add_group(
                    _RouteGradientDecompositionGroup(
                        group_id=f"mlp_neuron:L{layer_index}N{neuron_index}",
                        group_kind="mlp_neuron",
                        component_type="mlp",
                        partition_name="mlp_neurons",
                        layer=layer_index,
                        head=None,
                        projection=None,
                        neuron=neuron_index,
                        selections=(
                            _single_row_selection(fc_in_weight, neuron_index),
                            _single_vector_selection(fc_in_bias, neuron_index),
                            _single_column_selection(fc_out_weight, neuron_index),
                        ),
                        notes=("fc_out.bias is a block-level bias and is not assigned to individual neurons.",),
                    )
                )

    if len(groups) != len(group_ids):
        raise RuntimeError("Route-gradient decomposition group ids are not unique.")
    group_counts = Counter(group.group_kind for group in groups)
    return groups, {
        "decomposition_modes": decomposition_modes,
        "num_groups": len(groups),
        "group_counts_by_kind": {key: int(value) for key, value in sorted(group_counts.items())},
        "shared_parameter_name_sets": _shared_parameter_name_sets(model),
        "global_group_semantics": "The global group uses named parameters with remove_duplicate=False to match candidate-route-gradient-selection.",
        "overlap_warning": (
            "Rows are not one mutually exclusive partition across all group kinds. "
            "Compare supports within a group_kind or partition_name."
        ),
    }


def _gradient_dot_summary_for_group(
    *,
    left_gradients: dict[str, torch.Tensor],
    right_gradients: dict[str, torch.Tensor],
    group: _RouteGradientDecompositionGroup,
    label: str,
) -> dict[str, float | int | None]:
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    num_parameters = 0
    for selection in group.selections:
        parameter_name = selection.parameter_name
        if parameter_name not in left_gradients:
            raise KeyError(f"Left gradient missing parameter for {label}: {parameter_name}")
        if parameter_name not in right_gradients:
            raise KeyError(f"Right gradient missing parameter for {label}: {parameter_name}")
        left_tensor = left_gradients[parameter_name]
        right_tensor = right_gradients[parameter_name]
        if left_tensor.shape != right_tensor.shape:
            raise ValueError(
                f"Gradient shape mismatch for {label} key {parameter_name}: "
                f"{tuple(left_tensor.shape)} vs {tuple(right_tensor.shape)}"
            )
        left_selected = left_tensor if selection.selector is None else left_tensor[selection.selector]
        right_selected = right_tensor if selection.selector is None else right_tensor[selection.selector]
        if left_selected.shape != right_selected.shape:
            raise ValueError(
                f"Selected gradient shape mismatch for {label} key {parameter_name} "
                f"selector {selection.selector_label}: {tuple(left_selected.shape)} vs {tuple(right_selected.shape)}"
            )
        left_flat = left_selected.float().reshape(-1)
        right_flat = right_selected.float().reshape(-1)
        if left_flat.numel() <= 0:
            raise RuntimeError(
                f"Selected gradient slice is empty for {label} key {parameter_name} selector {selection.selector_label}"
            )
        dot += float(torch.dot(left_flat, right_flat).item())
        left_sq += float(torch.dot(left_flat, left_flat).item())
        right_sq += float(torch.dot(right_flat, right_flat).item())
        num_parameters += int(left_flat.numel())
    left_norm = left_sq ** 0.5
    right_norm = right_sq ** 0.5
    return {
        "num_parameters": num_parameters,
        "dot": dot,
        "left_l2_norm": left_norm,
        "right_l2_norm": right_norm,
        "cosine": _safe_ratio(dot, left_norm * right_norm),
    }


def _loss_records_for_pairs(*, pairs: list[dict[str, Any]], loss_side: str) -> list[dict[str, Any]]:
    if loss_side not in ROUTE_GRADIENT_LOSS_SIDES:
        raise ValueError(f"Unsupported loss_side {loss_side!r}; expected one of {ROUTE_GRADIENT_LOSS_SIDES}.")
    if not pairs:
        raise ValueError("Cannot build loss records from an empty pair list.")
    if loss_side == "clean":
        return [pair["clean_record"] for pair in pairs]
    if loss_side == "corrupted":
        return [pair["corrupted_record"] for pair in pairs]
    if loss_side == "both":
        records: list[dict[str, Any]] = []
        for pair in pairs:
            records.append(pair["clean_record"])
            records.append(pair["corrupted_record"])
        return records
    raise ValueError(f"Unhandled loss_side: {loss_side}")


def _compute_loss_gradient_for_records(
    *,
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty for loss-gradient computation.")
    model.eval()
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    for start_index in range(0, len(records), batch_size):
        batch_records = records[start_index : start_index + batch_size]
        batch = move_batch_to_device(collate_symbolic_kv(batch_records, pad_token_id), device)
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss, _ = compute_lm_loss(
            logits=outputs.logits,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pad_token_id=pad_token_id,
        )
        token_count = int(batch["attention_mask"][:, 1:].sum().item())
        if token_count <= 0:
            raise ValueError("Loss-gradient batch has no non-padding next-token targets.")
        (loss * token_count).backward()
        total_loss += float(loss.detach().cpu().item()) * token_count
        total_tokens += token_count
        num_batches += 1
    if total_tokens <= 0:
        raise ValueError("Loss-gradient records have no non-padding next-token targets.")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(float(total_tokens))
    gradients, zero_gradient_parameter_names = _parameter_gradients(model=model, require_all=True)
    if zero_gradient_parameter_names:
        raise RuntimeError(f"Loss gradient unexpectedly had zero-gradient parameters: {zero_gradient_parameter_names}")
    model.zero_grad(set_to_none=True)
    return {
        "loss": total_loss / total_tokens,
        "num_tokens": total_tokens,
        "num_records": len(records),
        "num_batches": num_batches,
        "gradients": gradients,
    }


def _route_group_metrics_from_logits(
    *,
    clean_answer_logits: torch.Tensor,
    corrupted_answer_logits: torch.Tensor,
    patched_answer_logits: torch.Tensor,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    clean_transfer_ids = _token_ids_for_values(
        values=[str(pair["clean_transfer_token"]) for pair in pairs],
        vocab=vocab,
        device=device,
    )
    corrupted_transfer_ids = _token_ids_for_values(
        values=[str(pair["corrupted_transfer_token"]) for pair in pairs],
        vocab=vocab,
        device=device,
    )
    transfer_margin_clean = _contrast_margin(
        answer_logits=clean_answer_logits,
        positive_token_ids=clean_transfer_ids,
        negative_token_ids=corrupted_transfer_ids,
    )
    transfer_margin_corrupted = _contrast_margin(
        answer_logits=corrupted_answer_logits,
        positive_token_ids=clean_transfer_ids,
        negative_token_ids=corrupted_transfer_ids,
    )
    transfer_margin_patched = _contrast_margin(
        answer_logits=patched_answer_logits,
        positive_token_ids=clean_transfer_ids,
        negative_token_ids=corrupted_transfer_ids,
    )
    route_delta = transfer_margin_patched - transfer_margin_corrupted
    return {
        "transfer_margin_clean": transfer_margin_clean,
        "transfer_margin_corrupted": transfer_margin_corrupted,
        "transfer_margin_patched": transfer_margin_patched,
        "route_delta": route_delta,
    }


def _compute_route_score_gradient_for_pairs(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    basis: torch.Tensor | None,
    stage_name: str,
    position_role: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pairs must not be empty for route-score gradient computation.")
    model.eval()
    model.zero_grad(set_to_none=True)
    total_route_score: torch.Tensor | None = None
    transfer_margin_clean_values: list[float] = []
    transfer_margin_corrupted_values: list[float] = []
    transfer_margin_patched_values: list[float] = []
    route_delta_values: list[float] = []
    num_batches = 0

    for start_index in range(0, len(pairs), batch_size):
        pair_batch = pairs[start_index : start_index + batch_size]
        clean_records = [pair["clean_record"] for pair in pair_batch]
        corrupted_records = [pair["corrupted_record"] for pair in pair_batch]
        clean_batch = move_batch_to_device(collate_symbolic_kv(clean_records, pad_token_id), device)
        corrupted_batch = move_batch_to_device(collate_symbolic_kv(corrupted_records, pad_token_id), device)
        clean_outputs = model(
            clean_batch["input_ids"],
            attention_mask=clean_batch["attention_mask"],
            return_residual_streams=True,
        )
        corrupted_outputs = model(
            corrupted_batch["input_ids"],
            attention_mask=corrupted_batch["attention_mask"],
            return_residual_streams=True,
        )
        if clean_outputs.residual_streams is None or corrupted_outputs.residual_streams is None:
            raise RuntimeError("Route-score gradient requires residual streams.")
        if stage_name not in clean_outputs.residual_streams:
            raise KeyError(f"Stage {stage_name!r} not found in clean residual streams.")
        if stage_name not in corrupted_outputs.residual_streams:
            raise KeyError(f"Stage {stage_name!r} not found in corrupted residual streams.")
        clean_answer_logits, _, clean_metadata = extract_answer_logits(clean_outputs.logits, clean_batch)
        corrupted_answer_logits, _, corrupted_metadata = extract_answer_logits(corrupted_outputs.logits, corrupted_batch)
        _validate_single_query_batch(batch=clean_batch, metadata=clean_metadata, label="route clean")
        _validate_single_query_batch(batch=corrupted_batch, metadata=corrupted_metadata, label="route corrupted")
        clean_selected = [
            _intervention_positions_for_query(
                batch=clean_batch,
                metadata=clean_metadata,
                flat_index=flat_index,
                position_role=position_role,
            )
            for flat_index in range(len(pair_batch))
        ]
        corrupted_selected = [
            _intervention_positions_for_query(
                batch=corrupted_batch,
                metadata=corrupted_metadata,
                flat_index=flat_index,
                position_role=position_role,
            )
            for flat_index in range(len(pair_batch))
        ]
        patched_stage = _transfer_stage_tensor(
            clean_stage=clean_outputs.residual_streams[stage_name],
            corrupted_stage=corrupted_outputs.residual_streams[stage_name],
            clean_selected=clean_selected,
            corrupted_selected=corrupted_selected,
            basis=basis,
        )
        patched_outputs = model(
            corrupted_batch["input_ids"],
            attention_mask=corrupted_batch["attention_mask"],
            residual_patch={stage_name: patched_stage},
        )
        patched_answer_logits, _, patched_metadata = extract_answer_logits(patched_outputs.logits, corrupted_batch)
        _validate_query_metadata_match(
            baseline_metadata=corrupted_metadata,
            patched_metadata=patched_metadata,
        )
        metric_tensors = _route_group_metrics_from_logits(
            clean_answer_logits=clean_answer_logits,
            corrupted_answer_logits=corrupted_answer_logits,
            patched_answer_logits=patched_answer_logits,
            pairs=pair_batch,
            vocab=vocab,
            device=device,
        )
        batch_route_score = metric_tensors["route_delta"].sum()
        total_route_score = batch_route_score if total_route_score is None else total_route_score + batch_route_score
        transfer_margin_clean_values.extend(
            float(value) for value in metric_tensors["transfer_margin_clean"].detach().float().cpu().tolist()
        )
        transfer_margin_corrupted_values.extend(
            float(value) for value in metric_tensors["transfer_margin_corrupted"].detach().float().cpu().tolist()
        )
        transfer_margin_patched_values.extend(
            float(value) for value in metric_tensors["transfer_margin_patched"].detach().float().cpu().tolist()
        )
        route_delta_values.extend(
            float(value) for value in metric_tensors["route_delta"].detach().float().cpu().tolist()
        )
        num_batches += 1

    if total_route_score is None:
        raise RuntimeError("Route-score gradient produced no score tensor.")
    mean_route_score = total_route_score / float(len(pairs))
    mean_route_score.backward()
    gradients, zero_gradient_parameter_names = _parameter_gradients(model=model, require_all=False)
    model.zero_grad(set_to_none=True)
    return {
        "route_score": float(mean_route_score.detach().float().cpu().item()),
        "num_pairs": len(pairs),
        "num_batches": num_batches,
        "transfer_margin_clean_mean": _mean(transfer_margin_clean_values),
        "transfer_margin_corrupted_mean": _mean(transfer_margin_corrupted_values),
        "transfer_margin_patched_mean": _mean(transfer_margin_patched_values),
        "route_delta_mean": _mean(route_delta_values),
        "route_delta_abs_mean": _mean([abs(value) for value in route_delta_values]),
        "route_delta_positive_fraction": _fraction(
            sum(1 for value in route_delta_values if value > 0.0),
            len(route_delta_values),
            "route_delta_positive_fraction",
        ),
        "route_delta_negative_fraction": _fraction(
            sum(1 for value in route_delta_values if value < 0.0),
            len(route_delta_values),
            "route_delta_negative_fraction",
        ),
        "zero_gradient_parameter_names": zero_gradient_parameter_names,
        "gradients": gradients,
    }


def _compute_route_score_for_pairs(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    basis: torch.Tensor | None,
    stage_name: str,
    position_role: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pairs must not be empty for route-score computation.")
    model.eval()
    transfer_margin_clean_values: list[float] = []
    transfer_margin_corrupted_values: list[float] = []
    transfer_margin_patched_values: list[float] = []
    route_delta_values: list[float] = []
    num_batches = 0

    with torch.no_grad():
        for start_index in range(0, len(pairs), batch_size):
            pair_batch = pairs[start_index : start_index + batch_size]
            clean_records = [pair["clean_record"] for pair in pair_batch]
            corrupted_records = [pair["corrupted_record"] for pair in pair_batch]
            clean_batch = move_batch_to_device(collate_symbolic_kv(clean_records, pad_token_id), device)
            corrupted_batch = move_batch_to_device(collate_symbolic_kv(corrupted_records, pad_token_id), device)
            clean_outputs = model(
                clean_batch["input_ids"],
                attention_mask=clean_batch["attention_mask"],
                return_residual_streams=True,
            )
            corrupted_outputs = model(
                corrupted_batch["input_ids"],
                attention_mask=corrupted_batch["attention_mask"],
                return_residual_streams=True,
            )
            if clean_outputs.residual_streams is None or corrupted_outputs.residual_streams is None:
                raise RuntimeError("Route-score computation requires residual streams.")
            if stage_name not in clean_outputs.residual_streams:
                raise KeyError(f"Stage {stage_name!r} not found in clean residual streams.")
            if stage_name not in corrupted_outputs.residual_streams:
                raise KeyError(f"Stage {stage_name!r} not found in corrupted residual streams.")
            clean_answer_logits, _, clean_metadata = extract_answer_logits(clean_outputs.logits, clean_batch)
            corrupted_answer_logits, _, corrupted_metadata = extract_answer_logits(
                corrupted_outputs.logits,
                corrupted_batch,
            )
            _validate_single_query_batch(batch=clean_batch, metadata=clean_metadata, label="route clean")
            _validate_single_query_batch(batch=corrupted_batch, metadata=corrupted_metadata, label="route corrupted")
            clean_selected = [
                _intervention_positions_for_query(
                    batch=clean_batch,
                    metadata=clean_metadata,
                    flat_index=flat_index,
                    position_role=position_role,
                )
                for flat_index in range(len(pair_batch))
            ]
            corrupted_selected = [
                _intervention_positions_for_query(
                    batch=corrupted_batch,
                    metadata=corrupted_metadata,
                    flat_index=flat_index,
                    position_role=position_role,
                )
                for flat_index in range(len(pair_batch))
            ]
            patched_stage = _transfer_stage_tensor(
                clean_stage=clean_outputs.residual_streams[stage_name],
                corrupted_stage=corrupted_outputs.residual_streams[stage_name],
                clean_selected=clean_selected,
                corrupted_selected=corrupted_selected,
                basis=basis,
            )
            patched_outputs = model(
                corrupted_batch["input_ids"],
                attention_mask=corrupted_batch["attention_mask"],
                residual_patch={stage_name: patched_stage},
            )
            patched_answer_logits, _, patched_metadata = extract_answer_logits(
                patched_outputs.logits,
                corrupted_batch,
            )
            _validate_query_metadata_match(
                baseline_metadata=corrupted_metadata,
                patched_metadata=patched_metadata,
            )
            metric_tensors = _route_group_metrics_from_logits(
                clean_answer_logits=clean_answer_logits,
                corrupted_answer_logits=corrupted_answer_logits,
                patched_answer_logits=patched_answer_logits,
                pairs=pair_batch,
                vocab=vocab,
                device=device,
            )
            transfer_margin_clean_values.extend(
                float(value) for value in metric_tensors["transfer_margin_clean"].detach().float().cpu().tolist()
            )
            transfer_margin_corrupted_values.extend(
                float(value)
                for value in metric_tensors["transfer_margin_corrupted"].detach().float().cpu().tolist()
            )
            transfer_margin_patched_values.extend(
                float(value) for value in metric_tensors["transfer_margin_patched"].detach().float().cpu().tolist()
            )
            route_delta_values.extend(
                float(value) for value in metric_tensors["route_delta"].detach().float().cpu().tolist()
            )
            num_batches += 1

    return {
        "route_score": _mean(route_delta_values),
        "num_pairs": len(pairs),
        "num_batches": num_batches,
        "transfer_margin_clean_mean": _mean(transfer_margin_clean_values),
        "transfer_margin_corrupted_mean": _mean(transfer_margin_corrupted_values),
        "transfer_margin_patched_mean": _mean(transfer_margin_patched_values),
        "route_delta_mean": _mean(route_delta_values),
        "route_delta_abs_mean": _mean([abs(value) for value in route_delta_values]),
        "route_delta_positive_fraction": _fraction(
            sum(1 for value in route_delta_values if value > 0.0),
            len(route_delta_values),
            "route_delta_positive_fraction",
        ),
        "route_delta_negative_fraction": _fraction(
            sum(1 for value in route_delta_values if value < 0.0),
            len(route_delta_values),
            "route_delta_negative_fraction",
        ),
    }


def _route_gradient_groups(pairs: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if not pairs:
        raise ValueError("Cannot build route-gradient groups from no pairs.")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        split = str(pair["split"])
        pair_type = str(pair["pair_type"])
        groups[(split, pair_type)].append(pair)
        groups[("__all__", pair_type)].append(pair)
        groups[(split, "__all__")].append(pair)
        groups[("__all__", "__all__")].append(pair)
    return dict(groups)


def _compute_route_gradient_selection_checkpoint(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    checkpoint_step: int,
    learning_rate: float,
    basis: torch.Tensor | None,
    subspace_summary: dict[str, Any],
    subspace_name: str,
    rank: int | None,
    stage_name: str,
    position_role: str,
    loss_side: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups = _route_gradient_groups(pairs)
    metric_rows: list[dict[str, Any]] = []
    route_gradients_by_group: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for (split, pair_type), group_pairs in sorted(groups.items()):
        loss_records = _loss_records_for_pairs(pairs=group_pairs, loss_side=loss_side)
        loss_payload = _compute_loss_gradient_for_records(
            model=model,
            records=loss_records,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            device=device,
        )
        route_payload = _compute_route_score_gradient_for_pairs(
            model=model,
            pairs=group_pairs,
            vocab=vocab,
            basis=basis,
            stage_name=stage_name,
            position_role=position_role,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            device=device,
        )
        loss_gradients = loss_payload["gradients"]
        route_gradients = route_payload["gradients"]
        if not isinstance(loss_gradients, dict) or not isinstance(route_gradients, dict):
            raise TypeError("Gradient payloads must be dictionaries.")
        dot_summary = _gradient_dot_summary(
            left_gradients=loss_gradients,
            right_gradients=route_gradients,
            label=f"{split}/{pair_type}",
        )
        loss_dot_route = float(dot_summary["dot"])
        negative_loss_dot_route = -loss_dot_route
        route_gradient_l2_norm = float(dot_summary["right_l2_norm"])
        metric_rows.append(
            {
                "step": checkpoint_step,
                "learning_rate": learning_rate,
                "split": split,
                "pair_type": pair_type,
                "loss_side": loss_side,
                "stage": stage_name,
                "subspace_name": subspace_name,
                "subspace_type": str(subspace_summary["subspace_type"]),
                "head_label": subspace_summary.get("head_label"),
                "rank": rank,
                "position_role": position_role,
                "num_pairs": int(route_payload["num_pairs"]),
                "loss_num_records": int(loss_payload["num_records"]),
                "loss_num_tokens": int(loss_payload["num_tokens"]),
                "loss": float(loss_payload["loss"]),
                "route_score": float(route_payload["route_score"]),
                "transfer_margin_clean_mean": float(route_payload["transfer_margin_clean_mean"]),
                "transfer_margin_corrupted_mean": float(route_payload["transfer_margin_corrupted_mean"]),
                "transfer_margin_patched_mean": float(route_payload["transfer_margin_patched_mean"]),
                "route_delta_mean": float(route_payload["route_delta_mean"]),
                "route_delta_abs_mean": float(route_payload["route_delta_abs_mean"]),
                "route_delta_positive_fraction": float(route_payload["route_delta_positive_fraction"]),
                "route_delta_negative_fraction": float(route_payload["route_delta_negative_fraction"]),
                "loss_gradient_l2_norm": float(dot_summary["left_l2_norm"]),
                "route_gradient_l2_norm": route_gradient_l2_norm,
                "loss_dot_route_gradient": loss_dot_route,
                "negative_loss_dot_route_gradient": negative_loss_dot_route,
                "loss_negative_route_gradient_cosine": _safe_ratio(
                    negative_loss_dot_route,
                    float(dot_summary["left_l2_norm"]) * route_gradient_l2_norm,
                ),
                "sgd_route_score_delta_linearized": learning_rate * negative_loss_dot_route,
                "projected_step_size_on_route_gradient": _safe_ratio(
                    negative_loss_dot_route,
                    route_gradient_l2_norm * route_gradient_l2_norm,
                ),
                "zero_route_gradient_parameter_count": len(route_payload["zero_gradient_parameter_names"]),
                "zero_route_gradient_parameter_names": route_payload["zero_gradient_parameter_names"],
            }
        )
        route_gradients_by_group[(split, pair_type)] = route_gradients

    pairwise_rows: list[dict[str, Any]] = []
    pair_type_groups = sorted(
        key
        for key in route_gradients_by_group
        if key[0] == "__all__" and key[1] != "__all__"
    )
    for left_index, left_key in enumerate(pair_type_groups):
        for right_key in pair_type_groups[left_index + 1 :]:
            dot_summary = _gradient_dot_summary(
                left_gradients=route_gradients_by_group[left_key],
                right_gradients=route_gradients_by_group[right_key],
                label=f"pairwise {left_key[1]} vs {right_key[1]}",
            )
            pairwise_rows.append(
                {
                    "step": checkpoint_step,
                    "left_split": left_key[0],
                    "left_pair_type": left_key[1],
                    "right_split": right_key[0],
                    "right_pair_type": right_key[1],
                    "stage": stage_name,
                    "subspace_name": subspace_name,
                    "subspace_type": str(subspace_summary["subspace_type"]),
                    "head_label": subspace_summary.get("head_label"),
                    "rank": rank,
                    "position_role": position_role,
                    "route_gradient_dot": float(dot_summary["dot"]),
                    "left_route_gradient_l2_norm": float(dot_summary["left_l2_norm"]),
                    "right_route_gradient_l2_norm": float(dot_summary["right_l2_norm"]),
                    "route_gradient_cosine": dot_summary["cosine"],
                }
            )
    return metric_rows, pairwise_rows


def _summarize_route_gradient_selection_report(
    *,
    metric_rows: list[dict[str, Any]],
    pairwise_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not metric_rows:
        raise ValueError("Cannot summarize route-gradient selection without metric rows.")
    final_step = max(int(row["step"]) for row in metric_rows)
    final_rows = [row for row in metric_rows if int(row["step"]) == final_step]
    final_all_rows = [
        row
        for row in final_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    if len(final_all_rows) != 1:
        raise RuntimeError(f"Expected one final all/all row, got {len(final_all_rows)}.")
    by_pair_type = sorted(
        [
            row
            for row in final_rows
            if str(row["split"]) == "__all__" and str(row["pair_type"]) != "__all__"
        ],
        key=lambda row: float(row["sgd_route_score_delta_linearized"]),
        reverse=True,
    )
    return {
        "num_checkpoints": len({int(row["step"]) for row in metric_rows}),
        "steps": sorted({int(row["step"]) for row in metric_rows}),
        "final_step": final_step,
        "final_all": final_all_rows[0],
        "final_by_pair_type_ranked_by_sgd_delta": by_pair_type,
        "final_by_split_and_pair_type": sorted(
            final_rows,
            key=lambda row: (str(row["split"]), str(row["pair_type"])),
        ),
        "final_pairwise_route_gradient_rows": [
            row for row in pairwise_rows if int(row["step"]) == final_step
        ],
    }


def _plot_route_gradient_support_by_pair_type(
    *,
    metric_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    final_step = max(int(row["step"]) for row in metric_rows)
    rows = [
        row
        for row in metric_rows
        if int(row["step"]) == final_step
        and str(row["split"]) == "__all__"
        and str(row["pair_type"]) != "__all__"
    ]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    rows = sorted(rows, key=lambda row: str(row["pair_type"]))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [str(row["pair_type"]) for row in rows],
        [float(row["negative_loss_dot_route_gradient"]) for row in rows],
        color="#376f8f",
    )
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title(f"SGD route support at step {final_step}")
    ax.set_xlabel("pair type")
    ax.set_ylabel("-grad(loss) . grad(route score)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_route_score_vs_gradient_support(
    *,
    metric_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    rows = [
        row
        for row in metric_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) != "__all__"
    ]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    pair_types = sorted({str(row["pair_type"]) for row in rows})
    fig, ax = plt.subplots(figsize=(8, 6))
    for pair_type in pair_types:
        typed = sorted([row for row in rows if str(row["pair_type"]) == pair_type], key=lambda row: int(row["step"]))
        ax.plot(
            [float(row["route_score"]) for row in typed],
            [float(row["negative_loss_dot_route_gradient"]) for row in typed],
            marker="o",
            label=pair_type,
        )
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle=":")
    ax.set_title("Route score vs SGD support")
    ax.set_xlabel("route score")
    ax.set_ylabel("-grad(loss) . grad(route score)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_route_pairwise_gradient_cosine(
    *,
    pairwise_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not pairwise_rows:
        return None
    final_step = max(int(row["step"]) for row in pairwise_rows)
    rows = [row for row in pairwise_rows if int(row["step"]) == final_step]
    if not rows:
        return None
    labels = sorted(
        {
            *(str(row["left_pair_type"]) for row in rows),
            *(str(row["right_pair_type"]) for row in rows),
        }
    )
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = torch.eye(len(labels), dtype=torch.float32)
    for row in rows:
        value = row["route_gradient_cosine"]
        if value is None:
            continue
        left = index[str(row["left_pair_type"])]
        right = index[str(row["right_pair_type"])]
        matrix[left, right] = float(value)
        matrix[right, left] = float(value)
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title(f"Route gradient cosine at step {final_step}")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for row_index in range(len(labels)):
        for col_index in range(len(labels)):
            ax.text(col_index, row_index, f"{float(matrix[row_index, col_index]):.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_route_gradient_selection_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    lines = [
        "# Candidate Route Gradient Selection",
        "",
        "## Calculation",
        "",
        "The route score is the causal patch shift:",
        "",
        "```text",
        "route_score = patched_transfer_margin - corrupted_transfer_margin",
        "```",
        "",
        "The SGD support score is:",
        "",
        "```text",
        "sgd_support = < -grad_theta loss, grad_theta route_score >",
        "linearized_delta = learning_rate * sgd_support",
        "```",
        "",
        "Positive support means the local SGD loss step would increase the route score. Negative support means the loss step would weaken it.",
        "",
        "## Route",
        "",
        f"- subspace: `{report['subspace']['subspace_name']}`",
        f"- rank: `{report['rank']}`",
        f"- stage: `{report['stage']}`",
        f"- position role: `{report['position_role']}`",
        f"- loss side: `{report['loss_side']}`",
        f"- device: `{report['device']}`",
        "",
        "## Pair Construction",
        "",
    ]
    for pair_type, count in report["pair_construction"]["constructed_counts"].items():
        lines.append(f"- {pair_type}: `{count}` pairs")
    lines.extend(
        [
            "",
            "## Final By Pair Type",
            "",
            "| pair type | pairs | route score | SGD support | linearized delta | cosine | loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["final_by_pair_type_ranked_by_sgd_delta"]:
        cosine = row["loss_negative_route_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| {pair_type} | {pairs} | {score:.6f} | {support:.6g} | {delta:.6g} | {cosine} | {loss:.6f} |".format(
                pair_type=row["pair_type"],
                pairs=int(row["num_pairs"]),
                score=float(row["route_score"]),
                support=float(row["negative_loss_dot_route_gradient"]),
                delta=float(row["sgd_route_score_delta_linearized"]),
                cosine=cosine_text,
                loss=float(row["loss"]),
            )
        )
    final_all = summary["final_all"]
    lines.extend(
        [
            "",
            "## Final All-Pair Result",
            "",
            f"- route score: `{float(final_all['route_score']):.6f}`",
            f"- SGD support: `{float(final_all['negative_loss_dot_route_gradient']):.6g}`",
            f"- linearized delta: `{float(final_all['sgd_route_score_delta_linearized']):.6g}`",
            "",
            "## Raw Outputs",
            "",
            f"- metric rows: `{report['metric_rows_path']}`",
            f"- pairwise rows: `{report['pairwise_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
            "",
            "## Plots",
            "",
        ]
    )
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_candidate_route_gradient_selection(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    stage_name: str,
    subspace_name: str,
    position_role: str,
    pair_types: list[str],
    rank: int | None = None,
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    head_layer: int | None = None,
    head: int | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    loss_side: str = "both",
) -> tuple[Path, Path, Path, Path, Path, dict[str, Path]]:
    if loss_side not in ROUTE_GRADIENT_LOSS_SIDES:
        raise ValueError(f"Unsupported loss_side {loss_side!r}; expected one of {ROUTE_GRADIENT_LOSS_SIDES}.")
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    model = build_model(spec.model, len(vocab.tokens), device)
    _validate_geometry_stage(model=model, stage_name=stage_name)
    if position_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported position role {position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Candidate route gradient selection constructed no pairs.")

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows_path = output_dir / "candidate_route_gradient_selection_rows.jsonl"
    pairwise_rows_path = output_dir / "candidate_route_gradient_selection_pairwise_rows.jsonl"
    pair_rows_path = output_dir / "candidate_route_gradient_selection_pairs.jsonl"
    progress_path = output_dir / "candidate_route_gradient_selection_progress.json"
    for partial_path in (metric_rows_path, pairwise_rows_path, pair_rows_path, progress_path):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])

    print(
        "[candidate-route-gradient-selection] "
        f"checkpoints={len(checkpoints)} pairs={len(pairs)} pair_types={pair_types} "
        f"device={device_name} subspace={subspace_name} rank={rank} stage={stage_name} "
        f"role={position_role} loss_side={loss_side}",
        flush=True,
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_pairwise_rows: list[dict[str, Any]] = []
    final_subspace_summary: dict[str, Any] | None = None
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if step != path_step:
            raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={step} path={path_step}")
        basis, subspace_summary = _resolve_causal_patch_basis(
            model=model,
            vocab=vocab,
            subspace_name=subspace_name,
            rank=rank,
            head_layer=head_layer,
            head=head,
            device=device,
        )
        final_subspace_summary = subspace_summary
        learning_rate = _compute_learning_rate(spec.optimization, step)
        print(
            f"[candidate-route-gradient-selection] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        metric_rows, pairwise_rows = _compute_route_gradient_selection_checkpoint(
            model=model,
            pairs=pairs,
            vocab=vocab,
            checkpoint_step=step,
            learning_rate=learning_rate,
            basis=basis,
            subspace_summary=subspace_summary,
            subspace_name=subspace_name,
            rank=rank,
            stage_name=stage_name,
            position_role=position_role,
            loss_side=loss_side,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
        )
        for row in metric_rows:
            append_jsonl(metric_rows_path, row)
        for row in pairwise_rows:
            append_jsonl(pairwise_rows_path, row)
        all_metric_rows.extend(metric_rows)
        all_pairwise_rows.extend(pairwise_rows)
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_checkpoints": checkpoint_index,
                "total_checkpoints": len(checkpoints),
                "last_completed_step": step,
                "metric_rows_path": str(metric_rows_path),
                "pairwise_rows_path": str(pairwise_rows_path),
                "pair_rows_path": str(pair_rows_path),
            },
        )
        all_row = next(
            row
            for row in metric_rows
            if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
        )
        print(
            "[candidate-route-gradient-selection] finished "
            f"step={step} route_score={float(all_row['route_score']):.6f} "
            f"sgd_support={float(all_row['negative_loss_dot_route_gradient']):.6g} "
            f"linearized_delta={float(all_row['sgd_route_score_delta_linearized']):.6g}",
            flush=True,
        )

    if final_subspace_summary is None:
        raise RuntimeError("No checkpoints were processed for candidate route gradient selection.")
    summary = _summarize_route_gradient_selection_report(
        metric_rows=all_metric_rows,
        pairwise_rows=all_pairwise_rows,
    )
    report_path = output_dir / "candidate_route_gradient_selection_report.json"
    markdown_path = output_dir / "candidate_route_gradient_selection_report.md"
    plot_paths: dict[str, Path] = {}
    support_plot = _plot_route_gradient_support_by_pair_type(
        metric_rows=all_metric_rows,
        output_path=output_dir / "candidate_route_gradient_support_by_pair_type.svg",
    )
    if support_plot is not None:
        plot_paths["support_by_pair_type"] = support_plot
    score_support_plot = _plot_route_score_vs_gradient_support(
        metric_rows=all_metric_rows,
        output_path=output_dir / "candidate_route_score_vs_gradient_support.svg",
    )
    if score_support_plot is not None:
        plot_paths["score_vs_support"] = score_support_plot
    pairwise_plot = _plot_route_pairwise_gradient_cosine(
        pairwise_rows=all_pairwise_rows,
        output_path=output_dir / "candidate_route_pairwise_gradient_cosine.svg",
    )
    if pairwise_plot is not None:
        plot_paths["pairwise_gradient_cosine"] = pairwise_plot

    report = {
        "schema_version": CANDIDATE_ROUTE_GRADIENT_SELECTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "stage": stage_name,
        "subspace": final_subspace_summary,
        "subspace_name": subspace_name,
        "rank": rank,
        "position_role": position_role,
        "pair_types": pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "loss_side": loss_side,
        "calculation": {
            "route_score": "patched_transfer_margin - corrupted_transfer_margin",
            "sgd_support": "< -grad_theta loss, grad_theta route_score >",
            "linearized_delta": "learning_rate * sgd_support",
            "positive_support": "local SGD step is predicted to increase the route score",
        },
        "pair_construction": pair_construction,
        "metric_rows_path": str(metric_rows_path),
        "pairwise_rows_path": str(pairwise_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_route_gradient_selection_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_checkpoints": len(checkpoints),
            "total_checkpoints": len(checkpoints),
            "last_completed_step": int(summary["final_step"]),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "metric_rows_path": str(metric_rows_path),
            "pairwise_rows_path": str(pairwise_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[candidate-route-gradient-selection] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, metric_rows_path, pairwise_rows_path, pair_rows_path, plot_paths


def _route_gradient_decomposition_row(
    *,
    checkpoint_step: int,
    learning_rate: float,
    split: str,
    pair_type: str,
    loss_side: str,
    stage_name: str,
    subspace_name: str,
    subspace_summary: dict[str, Any],
    rank: int | None,
    position_role: str,
    group: _RouteGradientDecompositionGroup,
    dot_summary: dict[str, float | int | None],
    route_payload: dict[str, Any],
    loss_payload: dict[str, Any],
) -> dict[str, Any]:
    loss_dot_route = float(dot_summary["dot"])
    negative_loss_dot_route = -loss_dot_route
    route_gradient_l2_norm = float(dot_summary["right_l2_norm"])
    loss_gradient_l2_norm = float(dot_summary["left_l2_norm"])
    num_selected_parameters = int(dot_summary["num_parameters"])
    return {
        "step": checkpoint_step,
        "learning_rate": learning_rate,
        "split": split,
        "pair_type": pair_type,
        "loss_side": loss_side,
        "stage": stage_name,
        "subspace_name": subspace_name,
        "subspace_type": str(subspace_summary["subspace_type"]),
        "head_label": subspace_summary.get("head_label"),
        "rank": rank,
        "position_role": position_role,
        "num_pairs": int(route_payload["num_pairs"]),
        "loss_num_records": int(loss_payload["num_records"]),
        "loss_num_tokens": int(loss_payload["num_tokens"]),
        "loss": float(loss_payload["loss"]),
        "route_score": float(route_payload["route_score"]),
        "transfer_margin_clean_mean": float(route_payload["transfer_margin_clean_mean"]),
        "transfer_margin_corrupted_mean": float(route_payload["transfer_margin_corrupted_mean"]),
        "transfer_margin_patched_mean": float(route_payload["transfer_margin_patched_mean"]),
        "route_delta_mean": float(route_payload["route_delta_mean"]),
        "route_delta_abs_mean": float(route_payload["route_delta_abs_mean"]),
        "route_delta_positive_fraction": float(route_payload["route_delta_positive_fraction"]),
        "route_delta_negative_fraction": float(route_payload["route_delta_negative_fraction"]),
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
        "loss_gradient_l2_norm": loss_gradient_l2_norm,
        "route_gradient_l2_norm": route_gradient_l2_norm,
        "loss_dot_route_gradient": loss_dot_route,
        "negative_loss_dot_route_gradient": negative_loss_dot_route,
        "loss_negative_route_gradient_cosine": _safe_ratio(
            negative_loss_dot_route,
            loss_gradient_l2_norm * route_gradient_l2_norm,
        ),
        "sgd_route_score_delta_linearized": learning_rate * negative_loss_dot_route,
        "support_per_parameter": negative_loss_dot_route / num_selected_parameters,
        "projected_step_size_on_route_gradient": _safe_ratio(
            negative_loss_dot_route,
            route_gradient_l2_norm * route_gradient_l2_norm,
        ),
        "notes": list(group.notes),
    }


def _compute_route_gradient_decomposition_checkpoint(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    checkpoint_step: int,
    learning_rate: float,
    basis: torch.Tensor | None,
    subspace_summary: dict[str, Any],
    subspace_name: str,
    rank: int | None,
    stage_name: str,
    position_role: str,
    loss_side: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    groups: list[_RouteGradientDecompositionGroup],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not groups:
        raise ValueError("Route-gradient decomposition requires at least one group.")
    route_groups = _route_gradient_groups(pairs)
    metric_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    global_group = next((group for group in groups if group.group_kind == "global_all"), None)
    if global_group is None:
        raise RuntimeError("Route-gradient decomposition groups are missing the global_all group.")

    for (split, pair_type), group_pairs in sorted(route_groups.items()):
        loss_records = _loss_records_for_pairs(pairs=group_pairs, loss_side=loss_side)
        loss_payload = _compute_loss_gradient_for_records(
            model=model,
            records=loss_records,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            device=device,
        )
        route_payload = _compute_route_score_gradient_for_pairs(
            model=model,
            pairs=group_pairs,
            vocab=vocab,
            basis=basis,
            stage_name=stage_name,
            position_role=position_role,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            device=device,
        )
        loss_gradients = loss_payload["gradients"]
        route_gradients = route_payload["gradients"]
        if not isinstance(loss_gradients, dict) or not isinstance(route_gradients, dict):
            raise TypeError("Gradient payloads must be dictionaries.")

        global_summary = _gradient_dot_summary_for_group(
            left_gradients=loss_gradients,
            right_gradients=route_gradients,
            group=global_group,
            label=f"{split}/{pair_type}/global",
        )
        metric_rows.append(
            _route_gradient_decomposition_row(
                checkpoint_step=checkpoint_step,
                learning_rate=learning_rate,
                split=split,
                pair_type=pair_type,
                loss_side=loss_side,
                stage_name=stage_name,
                subspace_name=subspace_name,
                subspace_summary=subspace_summary,
                rank=rank,
                position_role=position_role,
                group=global_group,
                dot_summary=global_summary,
                route_payload=route_payload,
                loss_payload=loss_payload,
            )
        )
        for group in groups:
            dot_summary = global_summary if group.group_kind == "global_all" else _gradient_dot_summary_for_group(
                left_gradients=loss_gradients,
                right_gradients=route_gradients,
                group=group,
                label=f"{split}/{pair_type}/{group.group_id}",
            )
            decomposition_rows.append(
                _route_gradient_decomposition_row(
                    checkpoint_step=checkpoint_step,
                    learning_rate=learning_rate,
                    split=split,
                    pair_type=pair_type,
                    loss_side=loss_side,
                    stage_name=stage_name,
                    subspace_name=subspace_name,
                    subspace_summary=subspace_summary,
                    rank=rank,
                    position_role=position_role,
                    group=group,
                    dot_summary=dot_summary,
                    route_payload=route_payload,
                    loss_payload=loss_payload,
                )
            )
    return metric_rows, decomposition_rows


def _summarize_route_gradient_decomposition_report(
    *,
    metric_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    top_k_groups: int,
) -> dict[str, Any]:
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")
    if not metric_rows:
        raise ValueError("Cannot summarize route-gradient decomposition without metric rows.")
    if not decomposition_rows:
        raise ValueError("Cannot summarize route-gradient decomposition without decomposition rows.")
    final_step = max(int(row["step"]) for row in metric_rows)
    final_metric_rows = [row for row in metric_rows if int(row["step"]) == final_step]
    final_all_rows = [
        row
        for row in final_metric_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    if len(final_all_rows) != 1:
        raise RuntimeError(f"Expected one final all/all metric row, got {len(final_all_rows)}.")
    final_decomposition_rows = [row for row in decomposition_rows if int(row["step"]) == final_step]
    final_all_decomposition_rows = [
        row
        for row in final_decomposition_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    non_global_final = [
        row
        for row in final_all_decomposition_rows
        if str(row["group_kind"]) != "global_all"
    ]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in non_global_final:
        by_kind.setdefault(str(row["group_kind"]), []).append(row)
    top_by_kind = {
        group_kind: {
            "top_positive_support": sorted(
                rows,
                key=lambda row: float(row["negative_loss_dot_route_gradient"]),
                reverse=True,
            )[:top_k_groups],
            "top_negative_support": sorted(
                rows,
                key=lambda row: float(row["negative_loss_dot_route_gradient"]),
            )[:top_k_groups],
            "top_abs_support": sorted(
                rows,
                key=lambda row: abs(float(row["negative_loss_dot_route_gradient"])),
                reverse=True,
            )[:top_k_groups],
        }
        for group_kind, rows in sorted(by_kind.items())
    }
    return {
        "num_checkpoints": len({int(row["step"]) for row in metric_rows}),
        "steps": sorted({int(row["step"]) for row in metric_rows}),
        "final_step": final_step,
        "final_all": final_all_rows[0],
        "final_metric_by_split_and_pair_type": sorted(
            final_metric_rows,
            key=lambda row: (str(row["split"]), str(row["pair_type"])),
        ),
        "final_top_positive_support": sorted(
            non_global_final,
            key=lambda row: float(row["negative_loss_dot_route_gradient"]),
            reverse=True,
        )[:top_k_groups],
        "final_top_negative_support": sorted(
            non_global_final,
            key=lambda row: float(row["negative_loss_dot_route_gradient"]),
        )[:top_k_groups],
        "final_top_abs_support": sorted(
            non_global_final,
            key=lambda row: abs(float(row["negative_loss_dot_route_gradient"])),
            reverse=True,
        )[:top_k_groups],
        "final_top_by_group_kind": top_by_kind,
    }


def _plot_route_decomposition_top_support(
    *,
    decomposition_rows: list[dict[str, Any]],
    top_k_groups: int,
    output_path: Path,
) -> Path | None:
    if not decomposition_rows:
        return None
    final_step = max(int(row["step"]) for row in decomposition_rows)
    rows = [
        row
        for row in decomposition_rows
        if int(row["step"]) == final_step
        and str(row["split"]) == "__all__"
        and str(row["pair_type"]) == "__all__"
        and str(row["group_kind"]) not in {"global_all", "parameter_tensor"}
    ]
    if not rows:
        return None
    top_rows = sorted(
        rows,
        key=lambda row: abs(float(row["negative_loss_dot_route_gradient"])),
        reverse=True,
    )[:top_k_groups]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(top_rows))))
    y_positions = list(range(len(top_rows)))
    values = [float(row["negative_loss_dot_route_gradient"]) for row in top_rows]
    labels = [str(row["group_id"]) for row in top_rows]
    colors = ["#376f8f" if value >= 0.0 else "#8f374a" for value in values]
    ax.barh(y_positions, values, color=colors)
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f"Top route-gradient supports at step {final_step}")
    ax.set_xlabel("-grad(loss) . grad(route score)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_route_decomposition_attention_qkvo_timeline(
    *,
    decomposition_rows: list[dict[str, Any]],
    head_layer: int | None,
    head: int | None,
    output_path: Path,
) -> Path | None:
    if head_layer is None or head is None:
        return None
    rows = [
        row
        for row in decomposition_rows
        if str(row["split"]) == "__all__"
        and str(row["pair_type"]) != "__all__"
        and str(row["group_kind"]) == "attention_head_projection"
        and int(row["group_layer"]) == head_layer
        and int(row["group_head"]) == head
    ]
    if not rows:
        return None
    pair_types = sorted({str(row["pair_type"]) for row in rows})
    selected_pair_type = "query_key" if "query_key" in pair_types else pair_types[0]
    selected_rows = [row for row in rows if str(row["pair_type"]) == selected_pair_type]
    projections = ["q_proj", "k_proj", "v_proj", "out_proj"]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    for projection in projections:
        projection_rows = sorted(
            [row for row in selected_rows if str(row["group_projection"]) == projection],
            key=lambda row: int(row["step"]),
        )
        if not projection_rows:
            continue
        ax.plot(
            [int(row["step"]) for row in projection_rows],
            [float(row["negative_loss_dot_route_gradient"]) for row in projection_rows],
            marker="o",
            label=projection,
        )
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title(f"L{head_layer}H{head} Q/K/V/O SGD support on {selected_pair_type}")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("-grad(loss) . grad(route score)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_route_decomposition_mlp_neuron_histogram(
    *,
    decomposition_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not decomposition_rows:
        return None
    final_step = max(int(row["step"]) for row in decomposition_rows)
    rows = [
        row
        for row in decomposition_rows
        if int(row["step"]) == final_step
        and str(row["split"]) == "__all__"
        and str(row["pair_type"]) == "__all__"
        and str(row["group_kind"]) == "mlp_neuron"
    ]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist([float(row["negative_loss_dot_route_gradient"]) for row in rows], bins=50, color="#6f8f37")
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title(f"MLP neuron route-gradient support distribution at step {final_step}")
    ax.set_xlabel("-grad(loss) . grad(route score)")
    ax.set_ylabel("neuron count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_route_gradient_decomposition_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    final_all = summary["final_all"]
    lines = [
        "# Route Gradient Decomposition",
        "",
        "## Calculation",
        "",
        "The route score is unchanged from candidate-route-gradient-selection:",
        "",
        "```text",
        "route_score = patched_transfer_margin - corrupted_transfer_margin",
        "```",
        "",
        "For each parameter group `g`, the decomposed support is:",
        "",
        "```text",
        "support_g = < -grad_g loss, grad_g route_score >",
        "linearized_delta_g = learning_rate * support_g",
        "```",
        "",
        "Positive support means the local SGD step would increase this route through that parameter group. Negative support means that group would weaken or rebalance the route.",
        "",
        "## Route",
        "",
        f"- subspace: `{report['subspace']['subspace_name']}`",
        f"- rank: `{report['rank']}`",
        f"- stage: `{report['stage']}`",
        f"- position role: `{report['position_role']}`",
        f"- loss side: `{report['loss_side']}`",
        f"- device: `{report['device']}`",
        "",
        "## Decomposition",
        "",
        f"- modes: `{', '.join(report['decomposition']['decomposition_modes'])}`",
        f"- groups: `{report['decomposition']['num_groups']}`",
        f"- group counts: `{report['decomposition']['group_counts_by_kind']}`",
        "",
        "Rows are not one global partition across all group kinds. Compare rows within the same `group_kind` or `partition_name`.",
        "",
        "## Final All-Pair Result",
        "",
        f"- route score: `{float(final_all['route_score']):.6f}`",
        f"- total SGD support: `{float(final_all['negative_loss_dot_route_gradient']):.6g}`",
        f"- total linearized delta: `{float(final_all['sgd_route_score_delta_linearized']):.6g}`",
        "",
        "## Final Top Positive Groups",
        "",
        "| group | kind | params | support | linearized delta | cosine |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["final_top_positive_support"]:
        cosine = row["loss_negative_route_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| {group} | {kind} | {params} | {support:.6g} | {delta:.6g} | {cosine} |".format(
                group=row["group_id"],
                kind=row["group_kind"],
                params=int(row["num_selected_parameters"]),
                support=float(row["negative_loss_dot_route_gradient"]),
                delta=float(row["sgd_route_score_delta_linearized"]),
                cosine=cosine_text,
            )
        )
    lines.extend(
        [
            "",
            "## Final Top Negative Groups",
            "",
            "| group | kind | params | support | linearized delta | cosine |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["final_top_negative_support"]:
        cosine = row["loss_negative_route_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| {group} | {kind} | {params} | {support:.6g} | {delta:.6g} | {cosine} |".format(
                group=row["group_id"],
                kind=row["group_kind"],
                params=int(row["num_selected_parameters"]),
                support=float(row["negative_loss_dot_route_gradient"]),
                delta=float(row["sgd_route_score_delta_linearized"]),
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
            f"- pair rows: `{report['pair_rows_path']}`",
            "",
            "## Plots",
            "",
        ]
    )
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_route_gradient_decomposition(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    stage_name: str,
    subspace_name: str,
    position_role: str,
    pair_types: list[str],
    rank: int | None = None,
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    head_layer: int | None = None,
    head: int | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    loss_side: str = "both",
    decomposition_modes: list[str] | None = None,
    top_k_groups: int = 24,
) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if loss_side not in ROUTE_GRADIENT_LOSS_SIDES:
        raise ValueError(f"Unsupported loss_side {loss_side!r}; expected one of {ROUTE_GRADIENT_LOSS_SIDES}.")
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")
    resolved_decomposition_modes = _resolve_route_gradient_decomposition_modes(decomposition_modes)
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    model = build_model(spec.model, len(vocab.tokens), device)
    _validate_geometry_stage(model=model, stage_name=stage_name)
    if position_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported position role {position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Route gradient decomposition constructed no pairs.")

    groups, decomposition_summary = _build_route_gradient_decomposition_groups(
        model=model,
        decomposition_modes=resolved_decomposition_modes,
    )
    group_rows = [
        _group_metadata(
            model_parameters=dict(model.named_parameters(remove_duplicate=False)),
            group=group,
        )
        for group in groups
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows_path = output_dir / "route_gradient_decomposition_metric_rows.jsonl"
    decomposition_rows_path = output_dir / "route_gradient_decomposition_rows.jsonl"
    group_rows_path = output_dir / "route_gradient_decomposition_groups.jsonl"
    pair_rows_path = output_dir / "route_gradient_decomposition_pairs.jsonl"
    progress_path = output_dir / "route_gradient_decomposition_progress.json"
    for partial_path in (metric_rows_path, decomposition_rows_path, group_rows_path, pair_rows_path, progress_path):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])
    write_jsonl(group_rows_path, group_rows)

    print(
        "[route-gradient-decomposition] "
        f"checkpoints={len(checkpoints)} pairs={len(pairs)} pair_types={pair_types} "
        f"device={device_name} subspace={subspace_name} rank={rank} stage={stage_name} "
        f"role={position_role} loss_side={loss_side} groups={len(groups)}",
        flush=True,
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_decomposition_rows: list[dict[str, Any]] = []
    final_subspace_summary: dict[str, Any] | None = None
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if step != path_step:
            raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={step} path={path_step}")
        basis, subspace_summary = _resolve_causal_patch_basis(
            model=model,
            vocab=vocab,
            subspace_name=subspace_name,
            rank=rank,
            head_layer=head_layer,
            head=head,
            device=device,
        )
        final_subspace_summary = subspace_summary
        learning_rate = _compute_learning_rate(spec.optimization, step)
        print(
            f"[route-gradient-decomposition] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        metric_rows, decomposition_rows = _compute_route_gradient_decomposition_checkpoint(
            model=model,
            pairs=pairs,
            vocab=vocab,
            checkpoint_step=step,
            learning_rate=learning_rate,
            basis=basis,
            subspace_summary=subspace_summary,
            subspace_name=subspace_name,
            rank=rank,
            stage_name=stage_name,
            position_role=position_role,
            loss_side=loss_side,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            groups=groups,
        )
        for row in metric_rows:
            append_jsonl(metric_rows_path, row)
        for row in decomposition_rows:
            append_jsonl(decomposition_rows_path, row)
        all_metric_rows.extend(metric_rows)
        all_decomposition_rows.extend(decomposition_rows)
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_checkpoints": checkpoint_index,
                "total_checkpoints": len(checkpoints),
                "last_completed_step": step,
                "metric_rows_path": str(metric_rows_path),
                "decomposition_rows_path": str(decomposition_rows_path),
                "group_rows_path": str(group_rows_path),
                "pair_rows_path": str(pair_rows_path),
            },
        )
        all_row = next(
            row
            for row in metric_rows
            if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
        )
        final_checkpoint_rows = [
            row
            for row in decomposition_rows
            if str(row["split"]) == "__all__"
            and str(row["pair_type"]) == "__all__"
            and str(row["group_kind"]) != "global_all"
        ]
        top_abs_row = max(
            final_checkpoint_rows,
            key=lambda row: abs(float(row["negative_loss_dot_route_gradient"])),
        )
        print(
            "[route-gradient-decomposition] finished "
            f"step={step} route_score={float(all_row['route_score']):.6f} "
            f"sgd_support={float(all_row['negative_loss_dot_route_gradient']):.6g} "
            f"top_group={top_abs_row['group_id']} top_support={float(top_abs_row['negative_loss_dot_route_gradient']):.6g}",
            flush=True,
        )

    if final_subspace_summary is None:
        raise RuntimeError("No checkpoints were processed for route gradient decomposition.")
    summary = _summarize_route_gradient_decomposition_report(
        metric_rows=all_metric_rows,
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
    )
    report_path = output_dir / "route_gradient_decomposition_report.json"
    markdown_path = output_dir / "route_gradient_decomposition_report.md"
    plot_paths: dict[str, Path] = {}
    top_support_plot = _plot_route_decomposition_top_support(
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
        output_path=output_dir / "route_gradient_decomposition_top_support.svg",
    )
    if top_support_plot is not None:
        plot_paths["top_support"] = top_support_plot
    qkvo_plot = _plot_route_decomposition_attention_qkvo_timeline(
        decomposition_rows=all_decomposition_rows,
        head_layer=head_layer,
        head=head,
        output_path=output_dir / "route_gradient_decomposition_attention_qkvo_timeline.svg",
    )
    if qkvo_plot is not None:
        plot_paths["attention_qkvo_timeline"] = qkvo_plot
    mlp_histogram = _plot_route_decomposition_mlp_neuron_histogram(
        decomposition_rows=all_decomposition_rows,
        output_path=output_dir / "route_gradient_decomposition_mlp_neuron_histogram.svg",
    )
    if mlp_histogram is not None:
        plot_paths["mlp_neuron_histogram"] = mlp_histogram

    report = {
        "schema_version": ROUTE_GRADIENT_DECOMPOSITION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "stage": stage_name,
        "subspace": final_subspace_summary,
        "subspace_name": subspace_name,
        "rank": rank,
        "position_role": position_role,
        "pair_types": pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "loss_side": loss_side,
        "decomposition": decomposition_summary,
        "top_k_groups": top_k_groups,
        "calculation": {
            "route_score": "patched_transfer_margin - corrupted_transfer_margin",
            "group_support": "< -grad_group loss, grad_group route_score >",
            "linearized_delta": "learning_rate * group_support",
            "positive_support": "local SGD step is predicted to increase the route score through this group",
            "attention_head_projection_units": (
                "q/k/v groups use output-channel rows plus bias; out_proj uses input-channel columns and excludes shared out_proj.bias."
            ),
            "mlp_neuron_unit": "fc_in row + fc_in bias element + fc_out column; fc_out.bias remains a block parameter.",
        },
        "pair_construction": pair_construction,
        "metric_rows_path": str(metric_rows_path),
        "decomposition_rows_path": str(decomposition_rows_path),
        "group_rows_path": str(group_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_route_gradient_decomposition_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_checkpoints": len(checkpoints),
            "total_checkpoints": len(checkpoints),
            "last_completed_step": int(summary["final_step"]),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "metric_rows_path": str(metric_rows_path),
            "decomposition_rows_path": str(decomposition_rows_path),
            "group_rows_path": str(group_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[route-gradient-decomposition] complete report={report_path} rows={decomposition_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, metric_rows_path, decomposition_rows_path, group_rows_path, pair_rows_path, plot_paths


def _model_parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters(remove_duplicate=False)
    }


def _parameter_delta(
    *,
    source_parameters: dict[str, torch.Tensor],
    target_parameters: dict[str, torch.Tensor],
    label: str,
) -> dict[str, torch.Tensor]:
    source_keys = set(source_parameters)
    target_keys = set(target_parameters)
    if source_keys != target_keys:
        missing_target = sorted(source_keys - target_keys)
        extra_target = sorted(target_keys - source_keys)
        raise ValueError(
            f"Parameter keys differ for {label}: missing_target={missing_target} extra_target={extra_target}"
        )
    deltas: dict[str, torch.Tensor] = {}
    for key in sorted(source_keys):
        source = source_parameters[key]
        target = target_parameters[key]
        if source.shape != target.shape:
            raise ValueError(
                f"Parameter shape mismatch for {label} key {key}: {tuple(source.shape)} vs {tuple(target.shape)}"
            )
        delta = target - source
        if not torch.isfinite(delta).all():
            raise RuntimeError(f"Non-finite parameter delta for {label} key {key}.")
        deltas[key] = delta
    return deltas


def _sign_match(actual: float, predicted: float) -> bool:
    if actual == 0.0 and predicted == 0.0:
        return True
    return actual * predicted > 0.0


def _checkpoint_update_metric_row(
    *,
    source_step: int,
    target_step: int,
    source_checkpoint: Path,
    target_checkpoint: Path,
    learning_rate: float,
    split: str,
    pair_type: str,
    stage_name: str,
    subspace_name: str,
    subspace_summary: dict[str, Any],
    rank: int | None,
    position_role: str,
    source_payload: dict[str, Any],
    target_payload: dict[str, Any],
    dot_summary: dict[str, float | int | None],
    min_error_denominator: float,
) -> dict[str, Any]:
    actual_delta = float(target_payload["route_score"]) - float(source_payload["route_score"])
    predicted_delta = float(dot_summary["dot"])
    residual = actual_delta - predicted_delta
    relative_error_denominator = max(abs(actual_delta), min_error_denominator)
    predicted_relative_error_denominator = max(abs(predicted_delta), min_error_denominator)
    return {
        "source_step": source_step,
        "target_step": target_step,
        "step_gap": target_step - source_step,
        "source_checkpoint": str(source_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "learning_rate": learning_rate,
        "split": split,
        "pair_type": pair_type,
        "stage": stage_name,
        "subspace_name": subspace_name,
        "subspace_type": str(subspace_summary["subspace_type"]),
        "head_label": subspace_summary.get("head_label"),
        "rank": rank,
        "position_role": position_role,
        "num_pairs": int(source_payload["num_pairs"]),
        "source_route_score": float(source_payload["route_score"]),
        "target_route_score": float(target_payload["route_score"]),
        "actual_delta": actual_delta,
        "predicted_delta": predicted_delta,
        "residual": residual,
        "absolute_error": abs(residual),
        "relative_error": abs(residual) / relative_error_denominator,
        "relative_error_denominator": relative_error_denominator,
        "predicted_relative_error": abs(residual) / predicted_relative_error_denominator,
        "predicted_relative_error_denominator": predicted_relative_error_denominator,
        "sign_match": _sign_match(actual=actual_delta, predicted=predicted_delta),
        "source_transfer_margin_clean_mean": float(source_payload["transfer_margin_clean_mean"]),
        "source_transfer_margin_corrupted_mean": float(source_payload["transfer_margin_corrupted_mean"]),
        "source_transfer_margin_patched_mean": float(source_payload["transfer_margin_patched_mean"]),
        "target_transfer_margin_clean_mean": float(target_payload["transfer_margin_clean_mean"]),
        "target_transfer_margin_corrupted_mean": float(target_payload["transfer_margin_corrupted_mean"]),
        "target_transfer_margin_patched_mean": float(target_payload["transfer_margin_patched_mean"]),
        "source_route_delta_abs_mean": float(source_payload["route_delta_abs_mean"]),
        "target_route_delta_abs_mean": float(target_payload["route_delta_abs_mean"]),
        "source_route_delta_positive_fraction": float(source_payload["route_delta_positive_fraction"]),
        "target_route_delta_positive_fraction": float(target_payload["route_delta_positive_fraction"]),
        "parameter_delta_l2_norm": float(dot_summary["left_l2_norm"]),
        "route_gradient_l2_norm": float(dot_summary["right_l2_norm"]),
        "update_route_gradient_cosine": dot_summary["cosine"],
        "num_parameters": int(dot_summary["num_parameters"]),
        "zero_route_gradient_parameter_count": len(source_payload["zero_gradient_parameter_names"]),
        "zero_route_gradient_parameter_names": source_payload["zero_gradient_parameter_names"],
    }


def _checkpoint_update_decomposition_row(
    *,
    metric_row: dict[str, Any],
    group: _RouteGradientDecompositionGroup,
    dot_summary: dict[str, float | int | None],
) -> dict[str, Any]:
    predicted_delta = float(dot_summary["dot"])
    route_gradient_l2_norm = float(dot_summary["right_l2_norm"])
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
        "stage": metric_row["stage"],
        "subspace_name": metric_row["subspace_name"],
        "subspace_type": metric_row["subspace_type"],
        "head_label": metric_row["head_label"],
        "rank": metric_row["rank"],
        "position_role": metric_row["position_role"],
        "num_pairs": int(metric_row["num_pairs"]),
        "source_route_score": float(metric_row["source_route_score"]),
        "target_route_score": float(metric_row["target_route_score"]),
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
        "route_gradient_l2_norm": route_gradient_l2_norm,
        "update_route_gradient_cosine": dot_summary["cosine"],
        "contribution_per_parameter": predicted_delta / num_selected_parameters,
        "notes": list(group.notes),
    }


def _compute_checkpoint_update_attribution_interval(
    *,
    model: torch.nn.Module,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    learning_rate: float,
    subspace_name: str,
    rank: int | None,
    head_layer: int | None,
    head: int | None,
    stage_name: str,
    position_role: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    groups: list[_RouteGradientDecompositionGroup],
    min_error_denominator: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not groups:
        raise ValueError("Checkpoint update attribution requires at least one decomposition group.")

    source_checkpoint = load_checkpoint(source_checkpoint_path, device)
    load_model_state(model, source_checkpoint["model_state"])
    model.eval()
    source_step = int(source_checkpoint["step"])
    source_path_step = _checkpoint_step_from_path(source_checkpoint_path)
    if source_step != source_path_step:
        raise RuntimeError(
            f"Source checkpoint step mismatch for {source_checkpoint_path}: payload={source_step} path={source_path_step}"
        )
    source_parameters = _model_parameter_snapshot(model)
    basis, subspace_summary = _resolve_causal_patch_basis(
        model=model,
        vocab=vocab,
        subspace_name=subspace_name,
        rank=rank,
        head_layer=head_layer,
        head=head,
        device=device,
    )

    route_groups = _route_gradient_groups(pairs)
    source_payloads: dict[tuple[str, str], dict[str, Any]] = {}
    for group_key, group_pairs in sorted(route_groups.items()):
        source_payloads[group_key] = _compute_route_score_gradient_for_pairs(
            model=model,
            pairs=group_pairs,
            vocab=vocab,
            basis=basis,
            stage_name=stage_name,
            position_role=position_role,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            device=device,
        )

    target_checkpoint = load_checkpoint(target_checkpoint_path, device)
    load_model_state(model, target_checkpoint["model_state"])
    model.eval()
    target_step = int(target_checkpoint["step"])
    target_path_step = _checkpoint_step_from_path(target_checkpoint_path)
    if target_step != target_path_step:
        raise RuntimeError(
            f"Target checkpoint step mismatch for {target_checkpoint_path}: payload={target_step} path={target_path_step}"
        )
    if target_step <= source_step:
        raise ValueError(
            f"Checkpoint update attribution requires increasing steps, got source={source_step} target={target_step}."
        )
    target_parameters = _model_parameter_snapshot(model)
    delta_parameters = _parameter_delta(
        source_parameters=source_parameters,
        target_parameters=target_parameters,
        label=f"{source_step}->{target_step}",
    )

    metric_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    for group_key, group_pairs in sorted(route_groups.items()):
        split, pair_type = group_key
        source_payload = source_payloads[group_key]
        target_payload = _compute_route_score_for_pairs(
            model=model,
            pairs=group_pairs,
            vocab=vocab,
            basis=basis,
            stage_name=stage_name,
            position_role=position_role,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            device=device,
        )
        route_gradients = source_payload["gradients"]
        if not isinstance(route_gradients, dict):
            raise TypeError("Route gradient payload must contain a gradients dictionary.")
        dot_summary = _gradient_dot_summary(
            left_gradients=delta_parameters,
            right_gradients=route_gradients,
            label=f"checkpoint update {source_step}->{target_step} {split}/{pair_type}",
        )
        metric_row = _checkpoint_update_metric_row(
            source_step=source_step,
            target_step=target_step,
            source_checkpoint=source_checkpoint_path,
            target_checkpoint=target_checkpoint_path,
            learning_rate=learning_rate,
            split=split,
            pair_type=pair_type,
            stage_name=stage_name,
            subspace_name=subspace_name,
            subspace_summary=subspace_summary,
            rank=rank,
            position_role=position_role,
            source_payload=source_payload,
            target_payload=target_payload,
            dot_summary=dot_summary,
            min_error_denominator=min_error_denominator,
        )
        metric_rows.append(metric_row)
        for group in groups:
            group_dot_summary = _gradient_dot_summary_for_group(
                left_gradients=delta_parameters,
                right_gradients=route_gradients,
                group=group,
                label=f"checkpoint update {source_step}->{target_step} {split}/{pair_type}/{group.group_id}",
            )
            decomposition_rows.append(
                _checkpoint_update_decomposition_row(
                    metric_row=metric_row,
                    group=group,
                    dot_summary=group_dot_summary,
                )
            )

    return metric_rows, decomposition_rows, subspace_summary


def _summarize_checkpoint_update_attribution(
    *,
    metric_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    top_k_groups: int,
) -> dict[str, Any]:
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")
    if not metric_rows:
        raise ValueError("Cannot summarize checkpoint update attribution without metric rows.")
    if not decomposition_rows:
        raise ValueError("Cannot summarize checkpoint update attribution without decomposition rows.")
    all_all_rows = [
        row
        for row in metric_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    if not all_all_rows:
        raise RuntimeError("Checkpoint update attribution has no __all__/__all__ metric rows.")
    final_target_step = max(int(row["target_step"]) for row in metric_rows)
    final_all_rows = [
        row
        for row in all_all_rows
        if int(row["target_step"]) == final_target_step
    ]
    if len(final_all_rows) != 1:
        raise RuntimeError(
            f"Expected one final __all__/__all__ metric row at target step {final_target_step}, got {len(final_all_rows)}."
        )
    final_decomposition_rows = [
        row
        for row in decomposition_rows
        if int(row["target_step"]) == final_target_step
        and str(row["split"]) == "__all__"
        and str(row["pair_type"]) == "__all__"
        and str(row["group_kind"]) not in {"global_all"}
    ]
    non_parameter_final = [
        row for row in final_decomposition_rows if str(row["group_kind"]) != "parameter_tensor"
    ]
    sign_match_fraction = _fraction(
        sum(1 for row in all_all_rows if bool(row["sign_match"])),
        len(all_all_rows),
        "checkpoint update sign_match_fraction",
    )
    return {
        "num_intervals": len({(int(row["source_step"]), int(row["target_step"])) for row in metric_rows}),
        "intervals": sorted(
            {
                f"{int(row['source_step'])}->{int(row['target_step'])}"
                for row in metric_rows
                if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
            }
        ),
        "target_steps": sorted({int(row["target_step"]) for row in metric_rows}),
        "final_target_step": final_target_step,
        "final_all": final_all_rows[0],
        "final_metric_by_split_and_pair_type": sorted(
            [row for row in metric_rows if int(row["target_step"]) == final_target_step],
            key=lambda row: (str(row["split"]), str(row["pair_type"])),
        ),
        "all_all_sign_match_fraction": sign_match_fraction,
        "all_all_mean_absolute_error": _mean([float(row["absolute_error"]) for row in all_all_rows]),
        "all_all_mean_relative_error": _mean([float(row["relative_error"]) for row in all_all_rows]),
        "all_all_worst_relative_error": max(all_all_rows, key=lambda row: float(row["relative_error"])),
        "final_top_positive_contributions": sorted(
            non_parameter_final,
            key=lambda row: float(row["predicted_delta_contribution"]),
            reverse=True,
        )[:top_k_groups],
        "final_top_negative_contributions": sorted(
            non_parameter_final,
            key=lambda row: float(row["predicted_delta_contribution"]),
        )[:top_k_groups],
        "final_top_abs_contributions": sorted(
            non_parameter_final,
            key=lambda row: abs(float(row["predicted_delta_contribution"])),
            reverse=True,
        )[:top_k_groups],
    }


def _plot_checkpoint_update_actual_vs_predicted(
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
    min_value = min(actual + predicted)
    max_value = max(actual + predicted)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    ax.plot([min_value, max_value], [min_value, max_value], color="#777777", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.axvline(0.0, color="#999999", linewidth=0.8)
    ax.set_title("Checkpoint update attribution: actual vs predicted")
    ax.set_xlabel("grad(route) . Delta theta")
    ax.set_ylabel("route(theta_target) - route(theta_source)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_checkpoint_update_relative_error(
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
        key=lambda row: int(row["target_step"]),
    )
    if not rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        [int(row["target_step"]) for row in rows],
        [float(row["relative_error"]) for row in rows],
        marker="o",
        color="#8f6237",
    )
    ax.set_title("Checkpoint update attribution relative error")
    ax.set_xlabel("target checkpoint step")
    ax.set_ylabel("|actual - predicted| / max(|actual|, epsilon)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_checkpoint_update_top_contributions(
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
    top_rows = sorted(
        rows,
        key=lambda row: abs(float(row["predicted_delta_contribution"])),
        reverse=True,
    )[:top_k_groups]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(top_rows))))
    y_positions = list(range(len(top_rows)))
    values = [float(row["predicted_delta_contribution"]) for row in top_rows]
    labels = [str(row["group_id"]) for row in top_rows]
    colors = ["#376f8f" if value >= 0.0 else "#8f374a" for value in values]
    ax.barh(y_positions, values, color=colors)
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f"Top actual-update route contributions ending at step {final_target_step}")
    ax.set_xlabel("grad(route) . Delta theta contribution")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_checkpoint_update_attribution_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    final_all = summary["final_all"]
    lines = [
        "# Checkpoint Update Attribution",
        "",
        "## Calculation",
        "",
        "This report tests whether actual checkpoint-to-checkpoint parameter movement explains route growth.",
        "",
        "```text",
        "actual_delta = route_score(theta_target; fixed_source_basis) - route_score(theta_source; fixed_source_basis)",
        "predicted_delta = grad_theta route_score(theta_source; fixed_source_basis) . (theta_target - theta_source)",
        "residual = actual_delta - predicted_delta",
        "```",
        "",
        "The basis is fixed from the source checkpoint for each interval. This avoids measuring route growth in a moving singular-vector coordinate system.",
        "",
        "## Route",
        "",
        f"- subspace: `{report['subspace_name']}`",
        f"- basis mode: `{report['basis_mode']}`",
        f"- rank: `{report['rank']}`",
        f"- stage: `{report['stage']}`",
        f"- position role: `{report['position_role']}`",
        f"- device: `{report['device']}`",
        "",
        "## Final Interval",
        "",
        f"- interval: `{final_all['source_step']} -> {final_all['target_step']}`",
        f"- source route score: `{float(final_all['source_route_score']):.6f}`",
        f"- target route score: `{float(final_all['target_route_score']):.6f}`",
        f"- actual delta: `{float(final_all['actual_delta']):.6g}`",
        f"- predicted delta: `{float(final_all['predicted_delta']):.6g}`",
        f"- residual: `{float(final_all['residual']):.6g}`",
        f"- relative error: `{float(final_all['relative_error']):.6g}`",
        f"- sign match: `{final_all['sign_match']}`",
        "",
        "## Reliability Summary",
        "",
        f"- all/all sign-match fraction: `{float(summary['all_all_sign_match_fraction']):.6f}`",
        f"- all/all mean absolute error: `{float(summary['all_all_mean_absolute_error']):.6g}`",
        f"- all/all mean relative error: `{float(summary['all_all_mean_relative_error']):.6g}`",
        "",
        "## Final Top Positive Contributions",
        "",
        "| group | kind | params | predicted contribution | cosine |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["final_top_positive_contributions"]:
        cosine = row["update_route_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| {group} | {kind} | {params} | {contribution:.6g} | {cosine} |".format(
                group=row["group_id"],
                kind=row["group_kind"],
                params=int(row["num_selected_parameters"]),
                contribution=float(row["predicted_delta_contribution"]),
                cosine=cosine_text,
            )
        )
    lines.extend(
        [
            "",
            "## Final Top Negative Contributions",
            "",
            "| group | kind | params | predicted contribution | cosine |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["final_top_negative_contributions"]:
        cosine = row["update_route_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| {group} | {kind} | {params} | {contribution:.6g} | {cosine} |".format(
                group=row["group_id"],
                kind=row["group_kind"],
                params=int(row["num_selected_parameters"]),
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
            f"- pair rows: `{report['pair_rows_path']}`",
            "",
            "## Plots",
            "",
        ]
    )
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


ATTENTION_SCORE_RECORD_SIDES = ["clean", "corrupted"]
ATTENTION_SCORE_COMPONENT_FIELDS = {
    "q_input_source_key": "q_input_source_key",
    "q_weight_source_key": "q_weight_source_key",
    "q_bias_source_key": "q_bias_source_key",
    "q_weight_input_cross_source_key": "q_weight_input_cross_source_key",
    "source_query_k_input": "source_query_k_input",
    "source_query_k_weight": "source_query_k_weight",
    "source_query_k_bias": "source_query_k_bias",
    "source_query_k_weight_input_cross": "source_query_k_weight_input_cross",
    "qk_vector_cross": "qk_vector_cross",
}


def _resolve_attention_score_record_sides(record_sides: list[str] | None) -> list[str]:
    if record_sides is None:
        return list(ATTENTION_SCORE_RECORD_SIDES)
    if not record_sides:
        raise ValueError("record_sides must not be empty when provided.")
    unsupported = [side for side in record_sides if side not in ATTENTION_SCORE_RECORD_SIDES]
    if unsupported:
        raise ValueError(f"Unsupported record sides {unsupported}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    return sorted(set(record_sides), key=record_sides.index)


def _attention_payload_for_records(
    *,
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    head_layer: int,
    head: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty for attention score decomposition.")
    if head_layer < 0 or head_layer >= len(model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(model.blocks) - 1}.")
    block = model.blocks[head_layer]
    if head < 0 or head >= block.attn.n_heads:
        raise ValueError(f"head {head} outside model range 0..{block.attn.n_heads - 1} for layer {head_layer}.")

    batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
    with torch.no_grad():
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_attentions=True,
            return_residual_streams=True,
        )
    if outputs.attentions is None:
        raise RuntimeError("Attention score decomposition requires attention tensors.")
    if outputs.residual_streams is None:
        raise RuntimeError("Attention score decomposition requires residual streams.")
    _, _, metadata = extract_answer_logits(outputs.logits, batch)
    _validate_single_query_batch(batch=batch, metadata=metadata, label="attention score")

    if head_layer == 0:
        pre_state = outputs.residual_streams["embedding"]
    else:
        pre_state = outputs.residual_streams[f"layer_{head_layer - 1}_post_mlp"]
    attention_input = block.ln_1(pre_state)
    batch_size, seq_len, _ = attention_input.shape
    head_dim = block.attn.head_dim
    head_slice = slice(head * head_dim, (head + 1) * head_dim)
    q_all = block.attn.q_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    k_all = block.attn.k_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    q_head = q_all[:, :, head, :].detach().float().cpu()
    k_head = k_all[:, :, head, :].detach().float().cpu()
    q_weight = block.attn.q_proj.weight.detach().float().cpu()[head_slice, :]
    k_weight = block.attn.k_proj.weight.detach().float().cpu()[head_slice, :]
    q_bias = block.attn.q_proj.bias.detach().float().cpu()[head_slice]
    k_bias = block.attn.k_proj.bias.detach().float().cpu()[head_slice]
    scores = torch.matmul(q_head, k_head.transpose(-2, -1)) / math.sqrt(head_dim)
    attention = outputs.attentions[head_layer][:, head, :, :].detach().float().cpu()
    return {
        "batch": batch,
        "metadata": metadata,
        "attention_input": attention_input.detach().float().cpu(),
        "q": q_head,
        "k": k_head,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "q_bias": q_bias,
        "k_bias": k_bias,
        "scores": scores,
        "attention": attention,
        "head_dim": int(head_dim),
    }


def _single_attention_position(
    *,
    batch: dict[str, Any],
    metadata: dict[str, torch.Tensor],
    flat_index: int,
    position_role: str,
    label: str,
) -> tuple[int, int]:
    batch_row, positions = _intervention_positions_for_query(
        batch=batch,
        metadata=metadata,
        flat_index=flat_index,
        position_role=position_role,
    )
    if len(positions) != 1:
        raise ValueError(
            f"{label} role {position_role!r} selected {len(positions)} positions; expected exactly one."
        )
    return batch_row, int(positions[0])


def _attention_key_positions(
    *,
    batch: dict[str, Any],
    metadata: dict[str, torch.Tensor],
    flat_index: int,
    position_role: str,
    max_position: int,
) -> tuple[int, list[int]]:
    if position_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported position role {position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if max_position < 0:
        raise ValueError(f"max_position must be non-negative, got {max_position}.")
    batch_row = int(metadata["rows"][flat_index].item())
    query_index = int(metadata["query_indices"][flat_index].item())
    prediction_position = int(metadata["prediction_positions"][flat_index].item())
    query_key_position = int(metadata["query_key_positions"][flat_index].item())
    record = batch["records"][batch_row]
    query_geometry = _positions_for_query(record, query_index, prediction_position)
    if position_role == "prediction":
        raw_positions = [prediction_position]
    elif position_role in {"query_key", "current_read_key"}:
        raw_positions = [query_key_position]
    elif position_role == "support_key":
        raw_positions = [int(query_geometry["support_key_position"])]
    elif position_role == "support_value":
        raw_positions = [int(query_geometry["support_value_position"])]
    elif position_role == "support_op":
        raw_positions = [int(query_geometry["support_op_position"])]
    elif position_role == "support_write":
        raw_positions = [
            int(query_geometry["support_op_position"]),
            int(query_geometry["support_key_position"]),
            int(query_geometry["support_value_position"]),
        ]
    elif position_role == "key_distractors":
        raw_positions = [int(position) for position in query_geometry["key_distractors"]]
    elif position_role == "value_distractors":
        raw_positions = [int(position) for position in query_geometry["value_distractors"]]
    elif position_role == "all_query_relation":
        raw_positions = [
            int(query_geometry["support_key_position"]),
            int(query_geometry["support_value_position"]),
            query_key_position,
        ]
    elif position_role == "causal_prefix":
        raw_positions = list(range(max_position + 1))
    else:
        raise ValueError(f"Unhandled position role: {position_role}")
    positions = sorted({position for position in raw_positions if 0 <= position <= max_position})
    if not positions:
        raise RuntimeError(
            f"Key role {position_role!r} selected no positions in the attention causal prefix "
            f"0..{max_position} for {record['sample_id']} query {query_index}."
        )
    return batch_row, positions


def _softmax_first_order_delta(
    *,
    source_attention_row: torch.Tensor,
    score_delta_row: torch.Tensor,
    key_position: int,
    query_position: int,
) -> float:
    if key_position > query_position:
        return 0.0
    causal_positions = slice(0, query_position + 1)
    source_probs = source_attention_row[causal_positions].double()
    score_delta = score_delta_row[causal_positions].double()
    centered_score_delta = score_delta[key_position] - torch.dot(source_probs, score_delta)
    return float((source_probs[key_position] * centered_score_delta).item())


def _decompose_qk_score_delta(
    *,
    source_payload: dict[str, Any],
    target_payload: dict[str, Any],
    batch_row: int,
    query_position: int,
    key_position: int,
    reconstruction_tolerance: float,
) -> dict[str, float]:
    head_dim = int(source_payload["head_dim"])
    if head_dim != int(target_payload["head_dim"]):
        raise RuntimeError(f"Head dim changed across checkpoints: {head_dim} vs {target_payload['head_dim']}")
    scale = math.sqrt(head_dim)
    q0 = source_payload["q"][batch_row, query_position, :].double()
    q1 = target_payload["q"][batch_row, query_position, :].double()
    k0 = source_payload["k"][batch_row, key_position, :].double()
    k1 = target_payload["k"][batch_row, key_position, :].double()
    xq0 = source_payload["attention_input"][batch_row, query_position, :].double()
    xq1 = target_payload["attention_input"][batch_row, query_position, :].double()
    xk0 = source_payload["attention_input"][batch_row, key_position, :].double()
    xk1 = target_payload["attention_input"][batch_row, key_position, :].double()
    wq0 = source_payload["q_weight"].double()
    wq1 = target_payload["q_weight"].double()
    wk0 = source_payload["k_weight"].double()
    wk1 = target_payload["k_weight"].double()
    bq0 = source_payload["q_bias"].double()
    bq1 = target_payload["q_bias"].double()
    bk0 = source_payload["k_bias"].double()
    bk1 = target_payload["k_bias"].double()

    dxq = xq1 - xq0
    dxk = xk1 - xk0
    dwq = wq1 - wq0
    dwk = wk1 - wk0
    dbq = bq1 - bq0
    dbk = bk1 - bk0

    q_components = {
        "input": torch.matmul(dxq, wq0.T),
        "weight": torch.matmul(xq0, dwq.T),
        "bias": dbq,
        "weight_input_cross": torch.matmul(dxq, dwq.T),
    }
    k_components = {
        "input": torch.matmul(dxk, wk0.T),
        "weight": torch.matmul(xk0, dwk.T),
        "bias": dbk,
        "weight_input_cross": torch.matmul(dxk, dwk.T),
    }
    dq = q1 - q0
    dk = k1 - k0
    dq_reconstructed = sum(q_components.values(), torch.zeros_like(dq))
    dk_reconstructed = sum(k_components.values(), torch.zeros_like(dk))
    q_reconstruction_error = float((dq - dq_reconstructed).abs().max().item())
    k_reconstruction_error = float((dk - dk_reconstructed).abs().max().item())
    if q_reconstruction_error > reconstruction_tolerance:
        raise RuntimeError(
            f"Q delta reconstruction error {q_reconstruction_error} exceeds tolerance {reconstruction_tolerance}."
        )
    if k_reconstruction_error > reconstruction_tolerance:
        raise RuntimeError(
            f"K delta reconstruction error {k_reconstruction_error} exceeds tolerance {reconstruction_tolerance}."
        )

    source_score = torch.dot(q0, k0) / scale
    target_score = torch.dot(q1, k1) / scale
    q_vector_delta_source_key = torch.dot(dq, k0) / scale
    source_query_k_vector_delta = torch.dot(q0, dk) / scale
    qk_vector_cross = torch.dot(dq, dk) / scale
    component_values = {
        "q_input_source_key": torch.dot(q_components["input"], k0) / scale,
        "q_weight_source_key": torch.dot(q_components["weight"], k0) / scale,
        "q_bias_source_key": torch.dot(q_components["bias"], k0) / scale,
        "q_weight_input_cross_source_key": torch.dot(q_components["weight_input_cross"], k0) / scale,
        "source_query_k_input": torch.dot(q0, k_components["input"]) / scale,
        "source_query_k_weight": torch.dot(q0, k_components["weight"]) / scale,
        "source_query_k_bias": torch.dot(q0, k_components["bias"]) / scale,
        "source_query_k_weight_input_cross": torch.dot(q0, k_components["weight_input_cross"]) / scale,
        "qk_vector_cross": qk_vector_cross,
    }
    reconstructed_delta = q_vector_delta_source_key + source_query_k_vector_delta + qk_vector_cross
    actual_delta = target_score - source_score
    score_reconstruction_error = float(abs((actual_delta - reconstructed_delta).item()))
    if score_reconstruction_error > reconstruction_tolerance:
        raise RuntimeError(
            f"Score delta reconstruction error {score_reconstruction_error} exceeds tolerance {reconstruction_tolerance}."
        )
    linear_score_delta = (
        component_values["q_input_source_key"]
        + component_values["q_weight_source_key"]
        + component_values["q_bias_source_key"]
        + component_values["source_query_k_input"]
        + component_values["source_query_k_weight"]
        + component_values["source_query_k_bias"]
    )
    internal_cross_delta = (
        component_values["q_weight_input_cross_source_key"]
        + component_values["source_query_k_weight_input_cross"]
    )
    return {
        "source_score": float(source_score.item()),
        "target_score": float(target_score.item()),
        "actual_score_delta": float(actual_delta.item()),
        "reconstructed_score_delta": float(reconstructed_delta.item()),
        "score_reconstruction_error": score_reconstruction_error,
        "q_delta_reconstruction_error": q_reconstruction_error,
        "k_delta_reconstruction_error": k_reconstruction_error,
        "q_vector_delta_source_key": float(q_vector_delta_source_key.item()),
        "source_query_k_vector_delta": float(source_query_k_vector_delta.item()),
        "qk_vector_cross": float(qk_vector_cross.item()),
        "linear_score_delta": float(linear_score_delta.item()),
        "internal_weight_input_cross_delta": float(internal_cross_delta.item()),
        **{field_name: float(value.item()) for field_name, value in component_values.items()},
    }


def _compute_attention_score_delta_interval(
    *,
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    head_layer: int,
    head: int,
    score_query_role: str,
    score_key_roles: list[str],
    record_sides: list[str],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    reconstruction_tolerance: float,
) -> list[dict[str, Any]]:
    source_checkpoint = load_checkpoint(source_checkpoint_path, device)
    target_checkpoint = load_checkpoint(target_checkpoint_path, device)
    load_model_state(source_model, source_checkpoint["model_state"])
    load_model_state(target_model, target_checkpoint["model_state"])
    source_model.eval()
    target_model.eval()
    source_step = int(source_checkpoint["step"])
    target_step = int(target_checkpoint["step"])
    source_path_step = _checkpoint_step_from_path(source_checkpoint_path)
    target_path_step = _checkpoint_step_from_path(target_checkpoint_path)
    if source_step != source_path_step:
        raise RuntimeError(f"Source checkpoint step mismatch: payload={source_step} path={source_path_step}")
    if target_step != target_path_step:
        raise RuntimeError(f"Target checkpoint step mismatch: payload={target_step} path={target_path_step}")
    if target_step <= source_step:
        raise ValueError(f"Attention score delta requires increasing steps, got {source_step}->{target_step}.")

    score_rows: list[dict[str, Any]] = []
    for start_index in range(0, len(pairs), batch_size):
        pair_batch = pairs[start_index : start_index + batch_size]
        for record_side in record_sides:
            side_key = f"{record_side}_record"
            records = [pair[side_key] for pair in pair_batch]
            source_payload = _attention_payload_for_records(
                model=source_model,
                records=records,
                head_layer=head_layer,
                head=head,
                pad_token_id=pad_token_id,
                device=device,
            )
            target_payload = _attention_payload_for_records(
                model=target_model,
                records=records,
                head_layer=head_layer,
                head=head,
                pad_token_id=pad_token_id,
                device=device,
            )
            for pair_index, pair in enumerate(pair_batch):
                batch_row, query_position = _single_attention_position(
                    batch=source_payload["batch"],
                    metadata=source_payload["metadata"],
                    flat_index=pair_index,
                    position_role=score_query_role,
                    label="score query",
                )
                target_batch_row, target_query_position = _single_attention_position(
                    batch=target_payload["batch"],
                    metadata=target_payload["metadata"],
                    flat_index=pair_index,
                    position_role=score_query_role,
                    label="target score query",
                )
                if batch_row != target_batch_row or query_position != target_query_position:
                    raise RuntimeError("Source/target query position selection disagrees.")
                source_score_row = source_payload["scores"][batch_row, query_position, :].double()
                target_score_row = target_payload["scores"][batch_row, query_position, :].double()
                score_delta_row = target_score_row - source_score_row
                for score_key_role in score_key_roles:
                    key_batch_row, key_positions = _attention_key_positions(
                        batch=source_payload["batch"],
                        metadata=source_payload["metadata"],
                        flat_index=pair_index,
                        position_role=score_key_role,
                        max_position=query_position,
                    )
                    target_key_batch_row, target_key_positions = _attention_key_positions(
                        batch=target_payload["batch"],
                        metadata=target_payload["metadata"],
                        flat_index=pair_index,
                        position_role=score_key_role,
                        max_position=query_position,
                    )
                    if key_batch_row != batch_row or target_key_batch_row != batch_row:
                        raise RuntimeError("Source/target key position batch rows disagree with query batch row.")
                    if key_positions != target_key_positions:
                        raise RuntimeError("Source/target key position selection disagrees.")
                    for role_position_index, key_position in enumerate(key_positions):
                        if key_position > query_position:
                            raise RuntimeError(
                                f"Key position {key_position} is after query position {query_position} "
                                f"for role {score_key_role!r} in pair {pair['pair_id']}."
                            )
                        decomposition = _decompose_qk_score_delta(
                            source_payload=source_payload,
                            target_payload=target_payload,
                            batch_row=batch_row,
                            query_position=query_position,
                            key_position=key_position,
                            reconstruction_tolerance=reconstruction_tolerance,
                        )
                        source_attention = float(source_payload["attention"][batch_row, query_position, key_position].item())
                        target_attention = float(target_payload["attention"][batch_row, query_position, key_position].item())
                        softmax_first_order = _softmax_first_order_delta(
                            source_attention_row=source_payload["attention"][batch_row, query_position, :],
                            score_delta_row=score_delta_row,
                            key_position=key_position,
                            query_position=query_position,
                        )
                        actual_attention_delta = target_attention - source_attention
                        score_rows.append(
                            {
                                "source_step": source_step,
                                "target_step": target_step,
                                "step_gap": target_step - source_step,
                                "source_checkpoint": str(source_checkpoint_path),
                                "target_checkpoint": str(target_checkpoint_path),
                                "head_layer": head_layer,
                                "head": head,
                                "head_label": _head_label(head_layer, head),
                                "score_query_role": score_query_role,
                                "score_key_role": score_key_role,
                                "record_side": record_side,
                                "split": str(pair["split"]),
                                "pair_type": str(pair["pair_type"]),
                                "pair_id": str(pair["pair_id"]),
                                "source_sample_id": str(pair["source_sample_id"]),
                                "source_query_index": int(pair["source_query_index"]),
                                "role_position_index": role_position_index,
                                "query_position": query_position,
                                "key_position": key_position,
                                "source_attention": source_attention,
                                "target_attention": target_attention,
                                "actual_attention_delta": actual_attention_delta,
                                "softmax_first_order_attention_delta": softmax_first_order,
                                "softmax_residual": actual_attention_delta - softmax_first_order,
                                **decomposition,
                            }
                        )
    if not score_rows:
        raise RuntimeError("Attention score delta interval produced no score rows.")
    return score_rows


def _aggregate_attention_score_delta_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not score_rows:
        raise ValueError("Cannot aggregate empty attention score rows.")
    metric_names = [
        "source_score",
        "target_score",
        "actual_score_delta",
        "reconstructed_score_delta",
        "score_reconstruction_error",
        "q_delta_reconstruction_error",
        "k_delta_reconstruction_error",
        "q_vector_delta_source_key",
        "source_query_k_vector_delta",
        "qk_vector_cross",
        "linear_score_delta",
        "internal_weight_input_cross_delta",
        "source_attention",
        "target_attention",
        "actual_attention_delta",
        "softmax_first_order_attention_delta",
        "softmax_residual",
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        group_keys = [
            (row["split"], row["pair_type"]),
            ("__all__", row["pair_type"]),
            (row["split"], "__all__"),
            ("__all__", "__all__"),
        ]
        for split, pair_type in group_keys:
            groups[
                (
                    row["source_step"],
                    row["target_step"],
                    split,
                    pair_type,
                    row["record_side"],
                    row["score_key_role"],
                )
            ].append(row)
    metric_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        source_step, target_step, split, pair_type, record_side, score_key_role = key
        first = rows[0]
        metric_row: dict[str, Any] = {
            "source_step": int(source_step),
            "target_step": int(target_step),
            "step_gap": int(first["step_gap"]),
            "split": split,
            "pair_type": pair_type,
            "record_side": record_side,
            "score_key_role": score_key_role,
            "score_query_role": first["score_query_role"],
            "head_layer": int(first["head_layer"]),
            "head": int(first["head"]),
            "head_label": first["head_label"],
            "num_scores": len(rows),
            "num_unique_pairs": len({row["pair_id"] for row in rows}),
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in rows]
            metric_row[f"{metric_name}_mean"] = _mean(values)
            metric_row[f"{metric_name}_abs_mean"] = _mean([abs(value) for value in values])
            metric_row[f"{metric_name}_std"] = _std(values)
        metric_rows.append(metric_row)
    return metric_rows


def _aggregate_attention_score_component_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not score_rows:
        raise ValueError("Cannot aggregate empty attention score rows.")
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    first_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in score_rows:
        group_keys = [
            (row["split"], row["pair_type"]),
            ("__all__", row["pair_type"]),
            (row["split"], "__all__"),
            ("__all__", "__all__"),
        ]
        for component_name, field_name in ATTENTION_SCORE_COMPONENT_FIELDS.items():
            for split, pair_type in group_keys:
                key = (
                    row["source_step"],
                    row["target_step"],
                    split,
                    pair_type,
                    row["record_side"],
                    row["score_key_role"],
                    component_name,
                )
                groups[key].append(float(row[field_name]))
                first_rows[key] = row
    component_rows: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        source_step, target_step, split, pair_type, record_side, score_key_role, component_name = key
        first = first_rows[key]
        component_rows.append(
            {
                "source_step": int(source_step),
                "target_step": int(target_step),
                "step_gap": int(first["step_gap"]),
                "split": split,
                "pair_type": pair_type,
                "record_side": record_side,
                "score_key_role": score_key_role,
                "score_query_role": first["score_query_role"],
                "head_layer": int(first["head_layer"]),
                "head": int(first["head"]),
                "head_label": first["head_label"],
                "component": component_name,
                "num_scores": len(values),
                "contribution_mean": _mean(values),
                "contribution_abs_mean": _mean([abs(value) for value in values]),
                "contribution_std": _std(values),
            }
        )
    return component_rows


def _summarize_attention_score_delta(
    *,
    metric_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    top_k_components: int,
) -> dict[str, Any]:
    if not metric_rows:
        raise ValueError("Cannot summarize attention score delta without metric rows.")
    if not component_rows:
        raise ValueError("Cannot summarize attention score delta without component rows.")
    intervals = sorted({(int(row["source_step"]), int(row["target_step"])) for row in metric_rows})
    final_interval = intervals[-1]
    final_rows = [
        row
        for row in metric_rows
        if (int(row["source_step"]), int(row["target_step"])) == final_interval
    ]
    final_component_rows = [
        row
        for row in component_rows
        if (int(row["source_step"]), int(row["target_step"])) == final_interval
    ]
    top_components = sorted(
        [
            row
            for row in final_component_rows
            if row["split"] == "__all__" and row["pair_type"] == "query_key"
        ],
        key=lambda row: abs(float(row["contribution_mean"])),
        reverse=True,
    )[:top_k_components]
    return {
        "num_intervals": len(intervals),
        "intervals": [f"{source}->{target}" for source, target in intervals],
        "num_metric_rows": len(metric_rows),
        "num_component_rows": len(component_rows),
        "final_interval": f"{final_interval[0]}->{final_interval[1]}",
        "final_rows": final_rows,
        "top_final_query_key_components": top_components,
    }


def _plot_attention_score_delta_components(
    *,
    component_rows: list[dict[str, Any]],
    top_k_components: int,
    output_path: Path,
) -> Path | None:
    if not component_rows:
        return None
    intervals = sorted({(int(row["source_step"]), int(row["target_step"])) for row in component_rows})
    if not intervals:
        return None
    final_interval = intervals[-1]
    rows = [
        row
        for row in component_rows
        if (int(row["source_step"]), int(row["target_step"])) == final_interval
        and row["split"] == "__all__"
        and row["pair_type"] == "query_key"
    ]
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: abs(float(row["contribution_mean"])), reverse=True)[:top_k_components]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.36 * len(rows))))
    labels = [
        f"{row['record_side']} {row['score_key_role']} {row['component']}"
        for row in rows
    ]
    values = [float(row["contribution_mean"]) for row in rows]
    colors = ["#2f7d59" if value >= 0.0 else "#9b3f37" for value in values]
    positions = list(range(len(rows)))
    ax.barh(positions, values, color=colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("mean contribution to QK score delta")
    ax.set_title(f"Top attention score delta terms {final_interval[0]}->{final_interval[1]}")
    ax.invert_yaxis()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _plot_attention_softmax_residual(
    *,
    metric_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    rows = [
        row
        for row in metric_rows
        if row["split"] == "__all__" and row["pair_type"] in {"query_key", "distractor"}
    ]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [
        f"{row['source_step']}->{row['target_step']} {row['pair_type']} {row['record_side']} {row['score_key_role']}"
        for row in rows
    ]
    values = [float(row["softmax_residual_mean"]) for row in rows]
    colors = ["#2f7d59" if value >= 0.0 else "#9b3f37" for value in values]
    positions = list(range(len(rows)))
    ax.bar(positions, values, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("mean softmax residual")
    ax.set_title("Attention probability delta not explained by source-softmax linearization")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _write_attention_score_delta_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    lines = [
        "# Attention Score Delta Decomposition",
        "",
        "## Calculation",
        "",
        "For one head and one attention score:",
        "",
        "`score = q_i . k_j / sqrt(d_head)`",
        "",
        "Across a checkpoint interval the exact score delta is decomposed as:",
        "",
        "`Delta score = Delta q . k0 + q0 . Delta k + Delta q . Delta k`",
        "",
        "`Delta q` and `Delta k` are also split into attention-input, projection-weight, projection-bias, and weight-input cross terms.",
        "",
        "## Run",
        "",
        f"- head: `{report['head_label']}`",
        f"- query role: `{report['score_query_role']}`",
        f"- key roles: `{report['score_key_roles']}`",
        f"- record sides: `{report['record_sides']}`",
        f"- intervals: `{summary['intervals']}`",
        "",
        "## Artifacts",
        "",
        f"- metric rows: `{report['metric_rows_path']}`",
        f"- score rows: `{report['score_rows_path']}`",
        f"- component rows: `{report['component_rows_path']}`",
        f"- pair rows: `{report['pair_rows_path']}`",
        "",
        "## Top Final Query-Key Components",
        "",
        "| component | record side | key role | contribution mean | abs mean |",
        "|---|---|---|---:|---:|",
    ]
    for row in summary["top_final_query_key_components"]:
        lines.append(
            "| `{component}` | `{side}` | `{role}` | {mean:.6f} | {abs_mean:.6f} |".format(
                component=row["component"],
                side=row["record_side"],
                role=row["score_key_role"],
                mean=float(row["contribution_mean"]),
                abs_mean=float(row["contribution_abs_mean"]),
            )
        )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for label, plot_path in plot_paths.items():
            lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_attention_score_delta_decomposition(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    head_layer: int,
    head: int,
    score_query_role: str,
    score_key_roles: list[str],
    pair_types: list[str],
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    record_sides: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    reconstruction_tolerance: float = 1.0e-3,
    top_k_components: int = 16,
) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if not score_key_roles:
        raise ValueError("At least one --score-key-role is required.")
    unsupported_roles = [
        role
        for role in [score_query_role, *score_key_roles]
        if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported attention score roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if reconstruction_tolerance <= 0.0:
        raise ValueError("reconstruction_tolerance must be positive.")
    if top_k_components <= 0:
        raise ValueError("top_k_components must be positive.")

    resolved_record_sides = _resolve_attention_score_record_sides(record_sides)
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("attention-score-delta-decomposition requires at least two checkpoints.")
    source_model = build_model(spec.model, len(vocab.tokens), device)
    target_model = build_model(spec.model, len(vocab.tokens), device)
    if head_layer < 0 or head_layer >= len(source_model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(source_model.blocks) - 1}.")
    if head < 0 or head >= source_model.blocks[head_layer].attn.n_heads:
        raise ValueError(
            f"head {head} outside model range 0..{source_model.blocks[head_layer].attn.n_heads - 1} for layer {head_layer}."
        )
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Attention score delta decomposition constructed no pairs.")

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows_path = output_dir / "attention_score_delta_rows.jsonl"
    score_rows_path = output_dir / "attention_score_delta_score_rows.jsonl"
    component_rows_path = output_dir / "attention_score_delta_components.jsonl"
    pair_rows_path = output_dir / "attention_score_delta_pairs.jsonl"
    progress_path = output_dir / "attention_score_delta_progress.json"
    for partial_path in (metric_rows_path, score_rows_path, component_rows_path, pair_rows_path, progress_path):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[attention-score-delta-decomposition] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(pairs)} "
        f"pair_types={pair_types} device={device_name} head={_head_label(head_layer, head)} "
        f"query_role={score_query_role} key_roles={score_key_roles} record_sides={resolved_record_sides}",
        flush=True,
    )
    all_score_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        print(
            "[attention-score-delta-decomposition] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )
        score_rows = _compute_attention_score_delta_interval(
            source_model=source_model,
            target_model=target_model,
            source_checkpoint_path=source_checkpoint_path,
            target_checkpoint_path=target_checkpoint_path,
            pairs=pairs,
            head_layer=head_layer,
            head=head,
            score_query_role=score_query_role,
            score_key_roles=score_key_roles,
            record_sides=resolved_record_sides,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            reconstruction_tolerance=reconstruction_tolerance,
        )
        for row in score_rows:
            append_jsonl(score_rows_path, row)
        all_score_rows.extend(score_rows)
        interval_metric_rows = _aggregate_attention_score_delta_rows(score_rows)
        all_row = next(
            row
            for row in interval_metric_rows
            if row["split"] == "__all__"
            and row["pair_type"] == "__all__"
            and row["record_side"] == resolved_record_sides[0]
            and row["score_key_role"] == score_key_roles[0]
        )
        print(
            "[attention-score-delta-decomposition] finished "
            f"{source_step}->{target_step} score_delta={float(all_row['actual_score_delta_mean']):.6g} "
            f"linear={float(all_row['linear_score_delta_mean']):.6g} "
            f"qk_cross={float(all_row['qk_vector_cross_mean']):.6g} "
            f"softmax_residual={float(all_row['softmax_residual_mean']):.6g}",
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
                "score_rows_path": str(score_rows_path),
            },
        )

    metric_rows = _aggregate_attention_score_delta_rows(all_score_rows)
    component_rows = _aggregate_attention_score_component_rows(all_score_rows)
    write_jsonl(metric_rows_path, metric_rows)
    write_jsonl(component_rows_path, component_rows)
    summary = _summarize_attention_score_delta(
        metric_rows=metric_rows,
        component_rows=component_rows,
        top_k_components=top_k_components,
    )
    plot_paths: dict[str, Path] = {}
    component_plot = _plot_attention_score_delta_components(
        component_rows=component_rows,
        top_k_components=top_k_components,
        output_path=output_dir / "attention_score_delta_components.svg",
    )
    if component_plot is not None:
        plot_paths["components"] = component_plot
    softmax_plot = _plot_attention_softmax_residual(
        metric_rows=metric_rows,
        output_path=output_dir / "attention_score_delta_softmax_residual.svg",
    )
    if softmax_plot is not None:
        plot_paths["softmax_residual"] = softmax_plot

    report_path = output_dir / "attention_score_delta_report.json"
    markdown_path = output_dir / "attention_score_delta_report.md"
    report = {
        "schema_version": ATTENTION_SCORE_DELTA_DECOMPOSITION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "head_layer": head_layer,
        "head": head,
        "head_label": _head_label(head_layer, head),
        "score_query_role": score_query_role,
        "score_key_roles": score_key_roles,
        "record_sides": resolved_record_sides,
        "pair_types": pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "reconstruction_tolerance": reconstruction_tolerance,
        "top_k_components": top_k_components,
        "calculation": {
            "score": "q_i . k_j / sqrt(d_head)",
            "score_delta": "Delta q . k0 + q0 . Delta k + Delta q . Delta k",
            "q_delta": "W_Q0 Delta x_q + Delta W_Q x_q0 + Delta b_Q + Delta W_Q Delta x_q",
            "k_delta": "W_K0 Delta x_k + Delta W_K x_k0 + Delta b_K + Delta W_K Delta x_k",
            "softmax_first_order": "Delta p_j ~= p_j * (Delta score_j - E_p[Delta score]) at the source checkpoint",
        },
        "pair_construction": pair_construction,
        "metric_rows_path": str(metric_rows_path),
        "score_rows_path": str(score_rows_path),
        "component_rows_path": str(component_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_attention_score_delta_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_intervals": len(intervals),
            "total_intervals": len(intervals),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "metric_rows_path": str(metric_rows_path),
            "score_rows_path": str(score_rows_path),
            "component_rows_path": str(component_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[attention-score-delta-decomposition] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, metric_rows_path, score_rows_path, component_rows_path, pair_rows_path, plot_paths


ATTENTION_SCORE_UPDATE_COMPONENTS = ["score", "q_side", "k_side"]


def _resolve_attention_score_update_components(components: list[str] | None) -> list[str]:
    if components is None:
        return list(ATTENTION_SCORE_UPDATE_COMPONENTS)
    if not components:
        raise ValueError("score update components must not be empty when provided.")
    unsupported = [component for component in components if component not in ATTENTION_SCORE_UPDATE_COMPONENTS]
    if unsupported:
        raise ValueError(
            f"Unsupported attention score update components {unsupported}; "
            f"expected one of {ATTENTION_SCORE_UPDATE_COMPONENTS}."
        )
    return sorted(set(components), key=components.index)


def _attention_score_tensor_payload_for_records(
    *,
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    head_layer: int,
    head: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty for attention score update attribution.")
    if head_layer < 0 or head_layer >= len(model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(model.blocks) - 1}.")
    block = model.blocks[head_layer]
    if head < 0 or head >= block.attn.n_heads:
        raise ValueError(f"head {head} outside model range 0..{block.attn.n_heads - 1} for layer {head_layer}.")

    batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
    outputs = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"],
        return_residual_streams=True,
    )
    if outputs.residual_streams is None:
        raise RuntimeError("Attention score update attribution requires residual streams.")
    _, _, metadata = extract_answer_logits(outputs.logits, batch)
    _validate_single_query_batch(batch=batch, metadata=metadata, label="attention score update")

    if head_layer == 0:
        pre_state = outputs.residual_streams["embedding"]
    else:
        pre_state = outputs.residual_streams[f"layer_{head_layer - 1}_post_mlp"]
    attention_input = block.ln_1(pre_state)
    batch_size, seq_len, _ = attention_input.shape
    head_dim = block.attn.head_dim
    q_all = block.attn.q_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    k_all = block.attn.k_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    return {
        "batch": batch,
        "metadata": metadata,
        "q": q_all[:, :, head, :],
        "k": k_all[:, :, head, :],
        "head_dim": int(head_dim),
    }


def _attention_score_component_tensor(
    *,
    q_vector: torch.Tensor,
    k_vector: torch.Tensor,
    score_component: str,
    scale: float,
) -> torch.Tensor:
    if score_component == "score":
        return torch.dot(q_vector, k_vector) / scale
    if score_component == "q_side":
        return torch.dot(q_vector, k_vector.detach()) / scale
    if score_component == "k_side":
        return torch.dot(q_vector.detach(), k_vector) / scale
    raise ValueError(
        f"Unsupported attention score update component {score_component!r}; "
        f"expected one of {ATTENTION_SCORE_UPDATE_COMPONENTS}."
    )


def _compute_attention_score_component_gradient_for_pairs(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    head_layer: int,
    head: int,
    score_query_role: str,
    score_key_role: str,
    record_side: str,
    score_component: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pairs must not be empty for attention score component gradient computation.")
    if record_side not in ATTENTION_SCORE_RECORD_SIDES:
        raise ValueError(f"Unsupported record side {record_side!r}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    if score_component not in ATTENTION_SCORE_UPDATE_COMPONENTS:
        raise ValueError(
            f"Unsupported attention score update component {score_component!r}; "
            f"expected one of {ATTENTION_SCORE_UPDATE_COMPONENTS}."
        )

    model.eval()
    model.zero_grad(set_to_none=True)
    total_score: torch.Tensor | None = None
    score_values: list[float] = []
    num_scores = 0
    num_batches = 0
    side_key = f"{record_side}_record"

    for start_index in range(0, len(pairs), batch_size):
        pair_batch = pairs[start_index : start_index + batch_size]
        records = [pair[side_key] for pair in pair_batch]
        payload = _attention_score_tensor_payload_for_records(
            model=model,
            records=records,
            head_layer=head_layer,
            head=head,
            pad_token_id=pad_token_id,
            device=device,
        )
        batch_terms: list[torch.Tensor] = []
        scale = math.sqrt(int(payload["head_dim"]))
        for pair_index, pair in enumerate(pair_batch):
            batch_row, query_position = _single_attention_position(
                batch=payload["batch"],
                metadata=payload["metadata"],
                flat_index=pair_index,
                position_role=score_query_role,
                label="score update query",
            )
            key_batch_row, key_positions = _attention_key_positions(
                batch=payload["batch"],
                metadata=payload["metadata"],
                flat_index=pair_index,
                position_role=score_key_role,
                max_position=query_position,
            )
            if key_batch_row != batch_row:
                raise RuntimeError(
                    f"Key role {score_key_role!r} selected batch row {key_batch_row}, "
                    f"but query role {score_query_role!r} selected batch row {batch_row} for pair {pair['pair_id']}."
                )
            for key_position in key_positions:
                if key_position > query_position:
                    raise RuntimeError(
                        f"Key position {key_position} is after query position {query_position} "
                        f"for role {score_key_role!r} in pair {pair['pair_id']}."
                    )
                q_vector = payload["q"][batch_row, query_position, :]
                k_vector = payload["k"][batch_row, key_position, :]
                score = _attention_score_component_tensor(
                    q_vector=q_vector,
                    k_vector=k_vector,
                    score_component=score_component,
                    scale=scale,
                )
                batch_terms.append(score)
                score_values.append(float(score.detach().float().cpu().item()))
        if not batch_terms:
            raise RuntimeError(
                f"Attention score update batch produced no scores for record_side={record_side!r} "
                f"score_key_role={score_key_role!r}."
            )
        batch_score = torch.stack(batch_terms).sum()
        total_score = batch_score if total_score is None else total_score + batch_score
        num_scores += len(batch_terms)
        num_batches += 1

    if total_score is None or num_scores <= 0:
        raise RuntimeError("Attention score component gradient produced no score tensor.")
    mean_score = total_score / float(num_scores)
    mean_score.backward()
    gradients, zero_gradient_parameter_names = _parameter_gradients(model=model, require_all=False)
    model.zero_grad(set_to_none=True)
    return {
        "score_value": float(mean_score.detach().float().cpu().item()),
        "score_value_abs_mean": _mean([abs(value) for value in score_values]),
        "score_value_std": _std(score_values),
        "num_scores": num_scores,
        "num_pairs": len(pairs),
        "num_batches": num_batches,
        "zero_gradient_parameter_names": zero_gradient_parameter_names,
        "gradients": gradients,
    }


def _attention_score_update_actual_summary(
    *,
    score_rows: list[dict[str, Any]],
    split: str,
    pair_type: str,
    record_side: str,
    score_key_role: str,
    score_component: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in score_rows
        if (split == "__all__" or str(row["split"]) == split)
        and (pair_type == "__all__" or str(row["pair_type"]) == pair_type)
        and str(row["record_side"]) == record_side
        and str(row["score_key_role"]) == score_key_role
    ]
    if not rows:
        raise RuntimeError(
            f"No attention score rows for split={split!r} pair_type={pair_type!r} "
            f"record_side={record_side!r} score_key_role={score_key_role!r}."
        )
    source_values = [float(row["source_score"]) for row in rows]
    total_delta_values = [float(row["actual_score_delta"]) for row in rows]
    q_side_delta_values = [float(row["q_vector_delta_source_key"]) for row in rows]
    k_side_delta_values = [float(row["source_query_k_vector_delta"]) for row in rows]
    qk_cross_values = [float(row["qk_vector_cross"]) for row in rows]
    if score_component == "score":
        actual_delta_values = total_delta_values
        target_values = [float(row["target_score"]) for row in rows]
    elif score_component == "q_side":
        actual_delta_values = q_side_delta_values
        target_values = [
            float(row["source_score"]) + float(row["q_vector_delta_source_key"])
            for row in rows
        ]
    elif score_component == "k_side":
        actual_delta_values = k_side_delta_values
        target_values = [
            float(row["source_score"]) + float(row["source_query_k_vector_delta"])
            for row in rows
        ]
    else:
        raise ValueError(
            f"Unsupported attention score update component {score_component!r}; "
            f"expected one of {ATTENTION_SCORE_UPDATE_COMPONENTS}."
        )
    return {
        "num_scores": len(rows),
        "num_unique_pairs": len({str(row["pair_id"]) for row in rows}),
        "source_value": _mean(source_values),
        "target_value": _mean(target_values),
        "actual_delta": _mean(actual_delta_values),
        "actual_delta_abs_mean": _mean([abs(value) for value in actual_delta_values]),
        "actual_delta_std": _std(actual_delta_values),
        "actual_total_score_delta_mean": _mean(total_delta_values),
        "actual_q_side_delta_mean": _mean(q_side_delta_values),
        "actual_k_side_delta_mean": _mean(k_side_delta_values),
        "actual_qk_cross_delta_mean": _mean(qk_cross_values),
        "source_attention_mean": _mean([float(row["source_attention"]) for row in rows]),
        "target_attention_mean": _mean([float(row["target_attention"]) for row in rows]),
        "actual_attention_delta_mean": _mean([float(row["actual_attention_delta"]) for row in rows]),
    }


def _attention_score_update_metric_row(
    *,
    source_step: int,
    target_step: int,
    source_checkpoint: Path,
    target_checkpoint: Path,
    learning_rate: float,
    split: str,
    pair_type: str,
    head_layer: int,
    head: int,
    score_query_role: str,
    score_key_role: str,
    record_side: str,
    score_component: str,
    actual_summary: dict[str, Any],
    source_payload: dict[str, Any],
    dot_summary: dict[str, float | int | None],
    min_error_denominator: float,
) -> dict[str, Any]:
    actual_delta = float(actual_summary["actual_delta"])
    predicted_delta = float(dot_summary["dot"])
    residual = actual_delta - predicted_delta
    relative_error_denominator = max(abs(actual_delta), min_error_denominator)
    predicted_relative_error_denominator = max(abs(predicted_delta), min_error_denominator)
    return {
        "source_step": source_step,
        "target_step": target_step,
        "step_gap": target_step - source_step,
        "source_checkpoint": str(source_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "learning_rate": learning_rate,
        "split": split,
        "pair_type": pair_type,
        "head_layer": head_layer,
        "head": head,
        "head_label": _head_label(head_layer, head),
        "score_query_role": score_query_role,
        "score_key_role": score_key_role,
        "record_side": record_side,
        "score_component": score_component,
        "num_pairs": int(source_payload["num_pairs"]),
        "num_scores": int(actual_summary["num_scores"]),
        "num_unique_pairs": int(actual_summary["num_unique_pairs"]),
        "source_value": float(actual_summary["source_value"]),
        "target_value": float(actual_summary["target_value"]),
        "source_objective_value": float(source_payload["score_value"]),
        "source_objective_value_abs_mean": float(source_payload["score_value_abs_mean"]),
        "source_objective_value_std": float(source_payload["score_value_std"]),
        "actual_delta": actual_delta,
        "actual_delta_abs_mean": float(actual_summary["actual_delta_abs_mean"]),
        "actual_delta_std": float(actual_summary["actual_delta_std"]),
        "predicted_delta": predicted_delta,
        "residual": residual,
        "absolute_error": abs(residual),
        "relative_error": abs(residual) / relative_error_denominator,
        "relative_error_denominator": relative_error_denominator,
        "predicted_relative_error": abs(residual) / predicted_relative_error_denominator,
        "predicted_relative_error_denominator": predicted_relative_error_denominator,
        "sign_match": _sign_match(actual=actual_delta, predicted=predicted_delta),
        "actual_total_score_delta_mean": float(actual_summary["actual_total_score_delta_mean"]),
        "actual_q_side_delta_mean": float(actual_summary["actual_q_side_delta_mean"]),
        "actual_k_side_delta_mean": float(actual_summary["actual_k_side_delta_mean"]),
        "actual_qk_cross_delta_mean": float(actual_summary["actual_qk_cross_delta_mean"]),
        "source_attention_mean": float(actual_summary["source_attention_mean"]),
        "target_attention_mean": float(actual_summary["target_attention_mean"]),
        "actual_attention_delta_mean": float(actual_summary["actual_attention_delta_mean"]),
        "parameter_delta_l2_norm": float(dot_summary["left_l2_norm"]),
        "score_gradient_l2_norm": float(dot_summary["right_l2_norm"]),
        "update_score_gradient_cosine": dot_summary["cosine"],
        "num_parameters": int(dot_summary["num_parameters"]),
        "zero_score_gradient_parameter_count": len(source_payload["zero_gradient_parameter_names"]),
        "zero_score_gradient_parameter_names": source_payload["zero_gradient_parameter_names"],
    }


def _attention_score_update_decomposition_row(
    *,
    metric_row: dict[str, Any],
    group: _RouteGradientDecompositionGroup,
    dot_summary: dict[str, float | int | None],
) -> dict[str, Any]:
    predicted_delta = float(dot_summary["dot"])
    score_gradient_l2_norm = float(dot_summary["right_l2_norm"])
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
        "head_label": metric_row["head_label"],
        "head_layer": int(metric_row["head_layer"]),
        "head": int(metric_row["head"]),
        "score_query_role": metric_row["score_query_role"],
        "score_key_role": metric_row["score_key_role"],
        "record_side": metric_row["record_side"],
        "score_component": metric_row["score_component"],
        "num_pairs": int(metric_row["num_pairs"]),
        "num_scores": int(metric_row["num_scores"]),
        "source_value": float(metric_row["source_value"]),
        "target_value": float(metric_row["target_value"]),
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
        "score_gradient_l2_norm": score_gradient_l2_norm,
        "update_score_gradient_cosine": dot_summary["cosine"],
        "contribution_per_parameter": predicted_delta / num_selected_parameters,
        "notes": list(group.notes),
    }


def _compute_attention_score_update_attribution_interval(
    *,
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    learning_rate: float,
    head_layer: int,
    head: int,
    score_query_role: str,
    score_key_roles: list[str],
    record_sides: list[str],
    score_components: list[str],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    groups: list[_RouteGradientDecompositionGroup],
    reconstruction_tolerance: float,
    min_error_denominator: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not groups:
        raise ValueError("Attention score update attribution requires at least one decomposition group.")

    score_rows = _compute_attention_score_delta_interval(
        source_model=source_model,
        target_model=target_model,
        source_checkpoint_path=source_checkpoint_path,
        target_checkpoint_path=target_checkpoint_path,
        pairs=pairs,
        head_layer=head_layer,
        head=head,
        score_query_role=score_query_role,
        score_key_roles=score_key_roles,
        record_sides=record_sides,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        device=device,
        reconstruction_tolerance=reconstruction_tolerance,
    )
    source_step = _checkpoint_step_from_path(source_checkpoint_path)
    target_step = _checkpoint_step_from_path(target_checkpoint_path)
    source_parameters = _model_parameter_snapshot(source_model)
    target_parameters = _model_parameter_snapshot(target_model)
    delta_parameters = _parameter_delta(
        source_parameters=source_parameters,
        target_parameters=target_parameters,
        label=f"attention score update {source_step}->{target_step}",
    )

    pair_groups = _route_gradient_groups(pairs)
    metric_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    for (split, pair_type), group_pairs in sorted(pair_groups.items()):
        for record_side in record_sides:
            for score_key_role in score_key_roles:
                for score_component in score_components:
                    actual_summary = _attention_score_update_actual_summary(
                        score_rows=score_rows,
                        split=split,
                        pair_type=pair_type,
                        record_side=record_side,
                        score_key_role=score_key_role,
                        score_component=score_component,
                    )
                    source_payload = _compute_attention_score_component_gradient_for_pairs(
                        model=source_model,
                        pairs=group_pairs,
                        head_layer=head_layer,
                        head=head,
                        score_query_role=score_query_role,
                        score_key_role=score_key_role,
                        record_side=record_side,
                        score_component=score_component,
                        batch_size=batch_size,
                        pad_token_id=pad_token_id,
                        device=device,
                    )
                    score_gradients = source_payload["gradients"]
                    if not isinstance(score_gradients, dict):
                        raise TypeError("Attention score gradient payload must contain a gradients dictionary.")
                    dot_summary = _gradient_dot_summary(
                        left_gradients=delta_parameters,
                        right_gradients=score_gradients,
                        label=(
                            f"attention score update {source_step}->{target_step} "
                            f"{split}/{pair_type}/{record_side}/{score_key_role}/{score_component}"
                        ),
                    )
                    metric_row = _attention_score_update_metric_row(
                        source_step=source_step,
                        target_step=target_step,
                        source_checkpoint=source_checkpoint_path,
                        target_checkpoint=target_checkpoint_path,
                        learning_rate=learning_rate,
                        split=split,
                        pair_type=pair_type,
                        head_layer=head_layer,
                        head=head,
                        score_query_role=score_query_role,
                        score_key_role=score_key_role,
                        record_side=record_side,
                        score_component=score_component,
                        actual_summary=actual_summary,
                        source_payload=source_payload,
                        dot_summary=dot_summary,
                        min_error_denominator=min_error_denominator,
                    )
                    metric_rows.append(metric_row)
                    for group in groups:
                        group_dot_summary = _gradient_dot_summary_for_group(
                            left_gradients=delta_parameters,
                            right_gradients=score_gradients,
                            group=group,
                            label=(
                                f"attention score update {source_step}->{target_step} "
                                f"{split}/{pair_type}/{record_side}/{score_key_role}/{score_component}/{group.group_id}"
                            ),
                        )
                        decomposition_rows.append(
                            _attention_score_update_decomposition_row(
                                metric_row=metric_row,
                                group=group,
                                dot_summary=group_dot_summary,
                            )
                        )

    return metric_rows, decomposition_rows, score_rows


def _summarize_attention_score_update_attribution(
    *,
    metric_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    top_k_groups: int,
) -> dict[str, Any]:
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")
    if not metric_rows:
        raise ValueError("Cannot summarize attention score update attribution without metric rows.")
    if not decomposition_rows:
        raise ValueError("Cannot summarize attention score update attribution without decomposition rows.")
    all_rows = [
        row
        for row in metric_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    if not all_rows:
        raise RuntimeError("Attention score update attribution has no __all__/__all__ metric rows.")
    final_target_step = max(int(row["target_step"]) for row in metric_rows)
    final_rows = [
        row for row in all_rows if int(row["target_step"]) == final_target_step
    ]
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
        "intervals": sorted(
            {
                f"{int(row['source_step'])}->{int(row['target_step'])}"
                for row in all_rows
            }
        ),
        "target_steps": sorted({int(row["target_step"]) for row in metric_rows}),
        "final_target_step": final_target_step,
        "final_metric_rows": sorted(
            final_rows,
            key=lambda row: (
                str(row["record_side"]),
                str(row["score_key_role"]),
                str(row["score_component"]),
            ),
        ),
        "all_all_sign_match_fraction": _fraction(
            sum(1 for row in all_rows if bool(row["sign_match"])),
            len(all_rows),
            "attention score update sign_match_fraction",
        ),
        "all_all_mean_absolute_error": _mean([float(row["absolute_error"]) for row in all_rows]),
        "all_all_mean_relative_error": _mean([float(row["relative_error"]) for row in all_rows]),
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


def _plot_attention_score_update_actual_vs_predicted(
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
    min_value = min(actual + predicted)
    max_value = max(actual + predicted)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    ax.plot([min_value, max_value], [min_value, max_value], color="#777777", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.axvline(0.0, color="#999999", linewidth=0.8)
    ax.set_title("Attention score update attribution: actual vs predicted")
    ax.set_xlabel("grad(score component) . Delta theta")
    ax.set_ylabel("actual score-component delta")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_attention_score_update_top_contributions(
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
    top_rows = sorted(
        rows,
        key=lambda row: abs(float(row["predicted_delta_contribution"])),
        reverse=True,
    )[:top_k_groups]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(13, max(5, 0.38 * len(top_rows))))
    y_positions = list(range(len(top_rows)))
    values = [float(row["predicted_delta_contribution"]) for row in top_rows]
    labels = [
        f"{row['score_component']} {row['record_side']} {row['score_key_role']} {row['group_id']}"
        for row in top_rows
    ]
    colors = ["#376f8f" if value >= 0.0 else "#8f374a" for value in values]
    ax.barh(y_positions, values, color=colors)
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"Top actual-update attention score contributions ending at step {final_target_step}")
    ax.set_xlabel("grad(score component) . Delta theta contribution")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_attention_score_update_attribution_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    lines = [
        "# Attention Score Update Attribution",
        "",
        "## Calculation",
        "",
        "This report tests whether actual checkpoint-to-checkpoint parameter movement explains raw QK score geometry changes.",
        "",
        "```text",
        "score = q_i . k_j / sqrt(d_head)",
        "predicted_delta = grad_theta score_component(theta_source) . (theta_target - theta_source)",
        "residual = actual_delta - predicted_delta",
        "```",
        "",
        "`q_side` holds the source key vector fixed and differentiates only through the query side.",
        "`k_side` holds the source query vector fixed and differentiates only through the key side.",
        "",
        "## Run",
        "",
        f"- head: `{report['head_label']}`",
        f"- query role: `{report['score_query_role']}`",
        f"- key roles: `{report['score_key_roles']}`",
        f"- record sides: `{report['record_sides']}`",
        f"- score components: `{report['score_components']}`",
        f"- intervals: `{summary['intervals']}`",
        "",
        "## Final Metrics",
        "",
        "| record side | key role | component | actual delta | predicted delta | residual | relative error | sign match |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["final_metric_rows"]:
        lines.append(
            "| `{side}` | `{role}` | `{component}` | {actual:.6g} | {predicted:.6g} | {residual:.6g} | {error:.6g} | `{sign}` |".format(
                side=row["record_side"],
                role=row["score_key_role"],
                component=row["score_component"],
                actual=float(row["actual_delta"]),
                predicted=float(row["predicted_delta"]),
                residual=float(row["residual"]),
                error=float(row["relative_error"]),
                sign=bool(row["sign_match"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Positive Contributions",
            "",
            "| group | component | kind | contribution | cosine |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in summary["final_top_positive_contributions"]:
        cosine = row["update_score_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| `{group}` | `{component}` | `{kind}` | {contribution:.6g} | {cosine} |".format(
                group=row["group_id"],
                component=row["score_component"],
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
            "| group | component | kind | contribution | cosine |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in summary["final_top_negative_contributions"]:
        cosine = row["update_score_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| `{group}` | `{component}` | `{kind}` | {contribution:.6g} | {cosine} |".format(
                group=row["group_id"],
                component=row["score_component"],
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
            f"- score rows: `{report['score_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
        ]
    )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for label, plot_path in plot_paths.items():
            lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_attention_score_update_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    head_layer: int,
    head: int,
    score_query_role: str,
    score_key_roles: list[str],
    pair_types: list[str],
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    record_sides: list[str] | None = None,
    score_components: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    decomposition_modes: list[str] | None = None,
    reconstruction_tolerance: float = 1.0e-3,
    top_k_groups: int = 24,
    min_error_denominator: float = 1.0e-9,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if not score_key_roles:
        raise ValueError("At least one --score-key-role is required.")
    unsupported_roles = [
        role
        for role in [score_query_role, *score_key_roles]
        if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported attention score roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if reconstruction_tolerance <= 0.0:
        raise ValueError("reconstruction_tolerance must be positive.")
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")

    resolved_record_sides = _resolve_attention_score_record_sides(record_sides)
    resolved_score_components = _resolve_attention_score_update_components(score_components)
    resolved_decomposition_modes = _resolve_route_gradient_decomposition_modes(decomposition_modes)
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("attention-score-update-attribution requires at least two checkpoints.")
    source_model = build_model(spec.model, len(vocab.tokens), device)
    target_model = build_model(spec.model, len(vocab.tokens), device)
    if head_layer < 0 or head_layer >= len(source_model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(source_model.blocks) - 1}.")
    if head < 0 or head >= source_model.blocks[head_layer].attn.n_heads:
        raise ValueError(
            f"head {head} outside model range 0..{source_model.blocks[head_layer].attn.n_heads - 1} for layer {head_layer}."
        )
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Attention score update attribution constructed no pairs.")

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

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows_path = output_dir / "attention_score_update_attribution_rows.jsonl"
    decomposition_rows_path = output_dir / "attention_score_update_attribution_decomposition_rows.jsonl"
    group_rows_path = output_dir / "attention_score_update_attribution_groups.jsonl"
    score_rows_path = output_dir / "attention_score_update_attribution_score_rows.jsonl"
    pair_rows_path = output_dir / "attention_score_update_attribution_pairs.jsonl"
    progress_path = output_dir / "attention_score_update_attribution_progress.json"
    for partial_path in (
        metric_rows_path,
        decomposition_rows_path,
        group_rows_path,
        score_rows_path,
        pair_rows_path,
        progress_path,
    ):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])
    write_jsonl(group_rows_path, group_rows)

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[attention-score-update-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(pairs)} "
        f"pair_types={pair_types} device={device_name} head={_head_label(head_layer, head)} "
        f"query_role={score_query_role} key_roles={score_key_roles} record_sides={resolved_record_sides} "
        f"components={resolved_score_components} groups={len(groups)}",
        flush=True,
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_decomposition_rows: list[dict[str, Any]] = []
    all_score_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        learning_rate = _compute_learning_rate(spec.optimization, source_step)
        print(
            "[attention-score-update-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )
        metric_rows, decomposition_rows, score_rows = _compute_attention_score_update_attribution_interval(
            source_model=source_model,
            target_model=target_model,
            source_checkpoint_path=source_checkpoint_path,
            target_checkpoint_path=target_checkpoint_path,
            pairs=pairs,
            learning_rate=learning_rate,
            head_layer=head_layer,
            head=head,
            score_query_role=score_query_role,
            score_key_roles=score_key_roles,
            record_sides=resolved_record_sides,
            score_components=resolved_score_components,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            groups=groups,
            reconstruction_tolerance=reconstruction_tolerance,
            min_error_denominator=min_error_denominator,
        )
        for row in metric_rows:
            append_jsonl(metric_rows_path, row)
        for row in decomposition_rows:
            append_jsonl(decomposition_rows_path, row)
        for row in score_rows:
            append_jsonl(score_rows_path, row)
        all_metric_rows.extend(metric_rows)
        all_decomposition_rows.extend(decomposition_rows)
        all_score_rows.extend(score_rows)
        all_row = next(
            row
            for row in metric_rows
            if str(row["split"]) == "__all__"
            and str(row["pair_type"]) == "__all__"
            and str(row["record_side"]) == resolved_record_sides[0]
            and str(row["score_key_role"]) == score_key_roles[0]
            and str(row["score_component"]) == resolved_score_components[0]
        )
        print(
            "[attention-score-update-attribution] finished "
            f"{source_step}->{target_step} component={all_row['score_component']} "
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
                "score_rows_path": str(score_rows_path),
            },
        )

    summary = _summarize_attention_score_update_attribution(
        metric_rows=all_metric_rows,
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
    )
    plot_paths: dict[str, Path] = {}
    actual_vs_predicted_plot = _plot_attention_score_update_actual_vs_predicted(
        metric_rows=all_metric_rows,
        output_path=output_dir / "attention_score_update_actual_vs_predicted.svg",
    )
    if actual_vs_predicted_plot is not None:
        plot_paths["actual_vs_predicted"] = actual_vs_predicted_plot
    top_contributions_plot = _plot_attention_score_update_top_contributions(
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
        output_path=output_dir / "attention_score_update_top_contributions.svg",
    )
    if top_contributions_plot is not None:
        plot_paths["top_contributions"] = top_contributions_plot

    report_path = output_dir / "attention_score_update_attribution_report.json"
    markdown_path = output_dir / "attention_score_update_attribution_report.md"
    report = {
        "schema_version": ATTENTION_SCORE_UPDATE_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "head_layer": head_layer,
        "head": head,
        "head_label": _head_label(head_layer, head),
        "score_query_role": score_query_role,
        "score_key_roles": score_key_roles,
        "record_sides": resolved_record_sides,
        "score_components": resolved_score_components,
        "pair_types": pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "decomposition": decomposition_summary,
        "reconstruction_tolerance": reconstruction_tolerance,
        "top_k_groups": top_k_groups,
        "min_error_denominator": min_error_denominator,
        "calculation": {
            "score": "q_i . k_j / sqrt(d_head)",
            "score_component": "score differentiates through q and k; q_side differentiates only through q; k_side differentiates only through k",
            "actual_delta": (
                "score uses target_score-source_score; q_side uses Delta q . k0 / sqrt(d_head); "
                "k_side uses q0 . Delta k / sqrt(d_head)"
            ),
            "predicted_delta": "grad_theta score_component(theta_source) . (theta_target - theta_source)",
            "residual": "actual_delta - predicted_delta",
            "group_contribution": "grad_group score_component(theta_source) . Delta theta_group",
            "cross_term_warning": "The qk cross term Delta q . Delta k is measured in score_rows but has no first-order gradient attribution row.",
        },
        "pair_construction": pair_construction,
        "metric_rows_path": str(metric_rows_path),
        "decomposition_rows_path": str(decomposition_rows_path),
        "group_rows_path": str(group_rows_path),
        "score_rows_path": str(score_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_attention_score_update_attribution_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
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
            "score_rows_path": str(score_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[attention-score-update-attribution] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return (
        report_path,
        markdown_path,
        metric_rows_path,
        decomposition_rows_path,
        group_rows_path,
        score_rows_path,
        pair_rows_path,
        plot_paths,
    )


def _attention_retrieval_separation_key_label(*, support_key_role: str, distractor_key_role: str) -> str:
    return f"{support_key_role}-minus-mean-{distractor_key_role}"


def _attention_retrieval_separation_component_gradient_for_pairs(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    record_side: str,
    score_component: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not pairs:
        raise ValueError("pairs must not be empty for retrieval-separation gradient computation.")
    if record_side not in ATTENTION_SCORE_RECORD_SIDES:
        raise ValueError(f"Unsupported record side {record_side!r}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    if score_component not in ATTENTION_SCORE_UPDATE_COMPONENTS:
        raise ValueError(
            f"Unsupported attention score update component {score_component!r}; "
            f"expected one of {ATTENTION_SCORE_UPDATE_COMPONENTS}."
        )

    model.eval()
    model.zero_grad(set_to_none=True)
    total_separation: torch.Tensor | None = None
    separation_values: list[float] = []
    support_score_values: list[float] = []
    distractor_score_values: list[float] = []
    num_support_scores = 0
    num_distractor_scores = 0
    num_batches = 0
    side_key = f"{record_side}_record"

    for start_index in range(0, len(pairs), batch_size):
        pair_batch = pairs[start_index : start_index + batch_size]
        records = [pair[side_key] for pair in pair_batch]
        payload = _attention_score_tensor_payload_for_records(
            model=model,
            records=records,
            head_layer=head_layer,
            head=head,
            pad_token_id=pad_token_id,
            device=device,
        )
        scale = math.sqrt(int(payload["head_dim"]))
        pair_terms: list[torch.Tensor] = []
        for pair_index, pair in enumerate(pair_batch):
            batch_row, query_position = _single_attention_position(
                batch=payload["batch"],
                metadata=payload["metadata"],
                flat_index=pair_index,
                position_role=score_query_role,
                label="retrieval separation query",
            )
            support_batch_row, support_positions = _attention_key_positions(
                batch=payload["batch"],
                metadata=payload["metadata"],
                flat_index=pair_index,
                position_role=support_key_role,
                max_position=query_position,
            )
            distractor_batch_row, distractor_positions = _attention_key_positions(
                batch=payload["batch"],
                metadata=payload["metadata"],
                flat_index=pair_index,
                position_role=distractor_key_role,
                max_position=query_position,
            )
            if support_batch_row != batch_row:
                raise RuntimeError(
                    f"Support role {support_key_role!r} selected batch row {support_batch_row}, "
                    f"but query role {score_query_role!r} selected batch row {batch_row} for pair {pair['pair_id']}."
                )
            if distractor_batch_row != batch_row:
                raise RuntimeError(
                    f"Distractor role {distractor_key_role!r} selected batch row {distractor_batch_row}, "
                    f"but query role {score_query_role!r} selected batch row {batch_row} for pair {pair['pair_id']}."
                )
            support_terms: list[torch.Tensor] = []
            distractor_terms: list[torch.Tensor] = []
            q_vector = payload["q"][batch_row, query_position, :]
            for key_position in support_positions:
                if key_position > query_position:
                    raise RuntimeError(
                        f"Support key position {key_position} is after query position {query_position} "
                        f"for pair {pair['pair_id']}."
                    )
                k_vector = payload["k"][batch_row, key_position, :]
                score = _attention_score_component_tensor(
                    q_vector=q_vector,
                    k_vector=k_vector,
                    score_component=score_component,
                    scale=scale,
                )
                support_terms.append(score)
                support_score_values.append(float(score.detach().float().cpu().item()))
            for key_position in distractor_positions:
                if key_position > query_position:
                    raise RuntimeError(
                        f"Distractor key position {key_position} is after query position {query_position} "
                        f"for pair {pair['pair_id']}."
                    )
                k_vector = payload["k"][batch_row, key_position, :]
                score = _attention_score_component_tensor(
                    q_vector=q_vector,
                    k_vector=k_vector,
                    score_component=score_component,
                    scale=scale,
                )
                distractor_terms.append(score)
                distractor_score_values.append(float(score.detach().float().cpu().item()))
            if not support_terms:
                raise RuntimeError(f"No support scores for pair {pair['pair_id']} and role {support_key_role!r}.")
            if not distractor_terms:
                raise RuntimeError(f"No distractor scores for pair {pair['pair_id']} and role {distractor_key_role!r}.")
            support_mean = torch.stack(support_terms).mean()
            distractor_mean = torch.stack(distractor_terms).mean()
            pair_separation = support_mean - distractor_mean
            pair_terms.append(pair_separation)
            separation_values.append(float(pair_separation.detach().float().cpu().item()))
            num_support_scores += len(support_terms)
            num_distractor_scores += len(distractor_terms)
        if not pair_terms:
            raise RuntimeError("Retrieval separation batch produced no pair terms.")
        batch_separation = torch.stack(pair_terms).sum()
        total_separation = batch_separation if total_separation is None else total_separation + batch_separation
        num_batches += 1

    if total_separation is None:
        raise RuntimeError("Retrieval separation gradient produced no objective tensor.")
    mean_separation = total_separation / float(len(pairs))
    mean_separation.backward()
    gradients, zero_gradient_parameter_names = _parameter_gradients(model=model, require_all=False)
    model.zero_grad(set_to_none=True)
    return {
        "score_value": float(mean_separation.detach().float().cpu().item()),
        "score_value_abs_mean": _mean([abs(value) for value in separation_values]),
        "score_value_std": _std(separation_values),
        "support_score_value_mean": _mean(support_score_values),
        "distractor_score_value_mean": _mean(distractor_score_values),
        "num_scores": num_support_scores + num_distractor_scores,
        "num_support_scores": num_support_scores,
        "num_distractor_scores": num_distractor_scores,
        "num_pairs": len(pairs),
        "num_batches": num_batches,
        "zero_gradient_parameter_names": zero_gradient_parameter_names,
        "gradients": gradients,
    }


def _attention_score_row_component_values(row: dict[str, Any], score_component: str) -> dict[str, float]:
    source_score = float(row["source_score"])
    if score_component == "score":
        target_value = float(row["target_score"])
        delta_value = float(row["actual_score_delta"])
    elif score_component == "q_side":
        delta_value = float(row["q_vector_delta_source_key"])
        target_value = source_score + delta_value
    elif score_component == "k_side":
        delta_value = float(row["source_query_k_vector_delta"])
        target_value = source_score + delta_value
    else:
        raise ValueError(
            f"Unsupported attention score update component {score_component!r}; "
            f"expected one of {ATTENTION_SCORE_UPDATE_COMPONENTS}."
        )
    return {
        "source": source_score,
        "target": target_value,
        "delta": delta_value,
        "total_delta": float(row["actual_score_delta"]),
        "q_side_delta": float(row["q_vector_delta_source_key"]),
        "k_side_delta": float(row["source_query_k_vector_delta"]),
        "qk_cross_delta": float(row["qk_vector_cross"]),
        "source_attention": float(row["source_attention"]),
        "target_attention": float(row["target_attention"]),
        "attention_delta": float(row["actual_attention_delta"]),
    }


def _mean_field(values: list[dict[str, float]], field: str) -> float:
    return _mean([float(value[field]) for value in values])


def _attention_retrieval_separation_actual_summary(
    *,
    score_rows: list[dict[str, Any]],
    split: str,
    pair_type: str,
    record_side: str,
    support_key_role: str,
    distractor_key_role: str,
    score_component: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in score_rows
        if (split == "__all__" or str(row["split"]) == split)
        and (pair_type == "__all__" or str(row["pair_type"]) == pair_type)
        and str(row["record_side"]) == record_side
        and str(row["score_key_role"]) in {support_key_role, distractor_key_role}
    ]
    if not rows:
        raise RuntimeError(
            f"No attention score rows for retrieval separation split={split!r} pair_type={pair_type!r} "
            f"record_side={record_side!r} support={support_key_role!r} distractor={distractor_key_role!r}."
        )
    by_pair: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_pair[str(row["pair_id"])][str(row["score_key_role"])].append(row)
    missing_support = sorted(pair_id for pair_id, role_rows in by_pair.items() if not role_rows.get(support_key_role))
    missing_distractor = sorted(pair_id for pair_id, role_rows in by_pair.items() if not role_rows.get(distractor_key_role))
    if missing_support or missing_distractor:
        raise RuntimeError(
            "Retrieval separation requires both support and distractor rows for every pair; "
            f"missing_support={missing_support[:10]} missing_distractor={missing_distractor[:10]}."
        )

    source_values: list[float] = []
    target_values: list[float] = []
    actual_delta_values: list[float] = []
    support_delta_values: list[float] = []
    distractor_delta_values: list[float] = []
    q_side_delta_values: list[float] = []
    k_side_delta_values: list[float] = []
    qk_cross_delta_values: list[float] = []
    total_score_delta_values: list[float] = []
    source_attention_values: list[float] = []
    target_attention_values: list[float] = []
    attention_delta_values: list[float] = []
    support_score_counts: list[int] = []
    distractor_score_counts: list[int] = []
    for pair_id, role_rows in sorted(by_pair.items()):
        support_values = [
            _attention_score_row_component_values(row, score_component)
            for row in role_rows[support_key_role]
        ]
        distractor_values = [
            _attention_score_row_component_values(row, score_component)
            for row in role_rows[distractor_key_role]
        ]
        support_source = _mean_field(support_values, "source")
        distractor_source = _mean_field(distractor_values, "source")
        support_target = _mean_field(support_values, "target")
        distractor_target = _mean_field(distractor_values, "target")
        support_delta = _mean_field(support_values, "delta")
        distractor_delta = _mean_field(distractor_values, "delta")
        source_values.append(support_source - distractor_source)
        target_values.append(support_target - distractor_target)
        actual_delta_values.append(support_delta - distractor_delta)
        support_delta_values.append(support_delta)
        distractor_delta_values.append(distractor_delta)
        q_side_delta_values.append(
            _mean_field(support_values, "q_side_delta") - _mean_field(distractor_values, "q_side_delta")
        )
        k_side_delta_values.append(
            _mean_field(support_values, "k_side_delta") - _mean_field(distractor_values, "k_side_delta")
        )
        qk_cross_delta_values.append(
            _mean_field(support_values, "qk_cross_delta") - _mean_field(distractor_values, "qk_cross_delta")
        )
        total_score_delta_values.append(
            _mean_field(support_values, "total_delta") - _mean_field(distractor_values, "total_delta")
        )
        source_attention_values.append(
            _mean_field(support_values, "source_attention") - _mean_field(distractor_values, "source_attention")
        )
        target_attention_values.append(
            _mean_field(support_values, "target_attention") - _mean_field(distractor_values, "target_attention")
        )
        attention_delta_values.append(
            _mean_field(support_values, "attention_delta") - _mean_field(distractor_values, "attention_delta")
        )
        support_score_counts.append(len(support_values))
        distractor_score_counts.append(len(distractor_values))

    return {
        "num_scores": sum(support_score_counts) + sum(distractor_score_counts),
        "num_support_scores": sum(support_score_counts),
        "num_distractor_scores": sum(distractor_score_counts),
        "num_unique_pairs": len(by_pair),
        "support_scores_per_pair_mean": _mean([float(value) for value in support_score_counts]),
        "distractor_scores_per_pair_mean": _mean([float(value) for value in distractor_score_counts]),
        "source_value": _mean(source_values),
        "target_value": _mean(target_values),
        "actual_delta": _mean(actual_delta_values),
        "actual_delta_abs_mean": _mean([abs(value) for value in actual_delta_values]),
        "actual_delta_std": _std(actual_delta_values),
        "support_delta_mean": _mean(support_delta_values),
        "distractor_delta_mean": _mean(distractor_delta_values),
        "actual_total_score_delta_mean": _mean(total_score_delta_values),
        "actual_q_side_delta_mean": _mean(q_side_delta_values),
        "actual_k_side_delta_mean": _mean(k_side_delta_values),
        "actual_qk_cross_delta_mean": _mean(qk_cross_delta_values),
        "source_attention_mean": _mean(source_attention_values),
        "target_attention_mean": _mean(target_attention_values),
        "actual_attention_delta_mean": _mean(attention_delta_values),
    }


def _compute_attention_retrieval_separation_update_attribution_interval(
    *,
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    learning_rate: float,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    record_sides: list[str],
    score_components: list[str],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    groups: list[_RouteGradientDecompositionGroup],
    reconstruction_tolerance: float,
    min_error_denominator: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    score_rows = _compute_attention_score_delta_interval(
        source_model=source_model,
        target_model=target_model,
        source_checkpoint_path=source_checkpoint_path,
        target_checkpoint_path=target_checkpoint_path,
        pairs=pairs,
        head_layer=head_layer,
        head=head,
        score_query_role=score_query_role,
        score_key_roles=[support_key_role, distractor_key_role],
        record_sides=record_sides,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        device=device,
        reconstruction_tolerance=reconstruction_tolerance,
    )
    source_step = _checkpoint_step_from_path(source_checkpoint_path)
    target_step = _checkpoint_step_from_path(target_checkpoint_path)
    source_parameters = _model_parameter_snapshot(source_model)
    target_parameters = _model_parameter_snapshot(target_model)
    delta_parameters = _parameter_delta(
        source_parameters=source_parameters,
        target_parameters=target_parameters,
        label=f"attention retrieval separation update {source_step}->{target_step}",
    )

    score_key_label = _attention_retrieval_separation_key_label(
        support_key_role=support_key_role,
        distractor_key_role=distractor_key_role,
    )
    pair_groups = _route_gradient_groups(pairs)
    metric_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    for (split, pair_type), group_pairs in sorted(pair_groups.items()):
        for record_side in record_sides:
            for score_component in score_components:
                actual_summary = _attention_retrieval_separation_actual_summary(
                    score_rows=score_rows,
                    split=split,
                    pair_type=pair_type,
                    record_side=record_side,
                    support_key_role=support_key_role,
                    distractor_key_role=distractor_key_role,
                    score_component=score_component,
                )
                source_payload = _attention_retrieval_separation_component_gradient_for_pairs(
                    model=source_model,
                    pairs=group_pairs,
                    head_layer=head_layer,
                    head=head,
                    score_query_role=score_query_role,
                    support_key_role=support_key_role,
                    distractor_key_role=distractor_key_role,
                    record_side=record_side,
                    score_component=score_component,
                    batch_size=batch_size,
                    pad_token_id=pad_token_id,
                    device=device,
                )
                score_gradients = source_payload["gradients"]
                if not isinstance(score_gradients, dict):
                    raise TypeError("Retrieval-separation gradient payload must contain a gradients dictionary.")
                dot_summary = _gradient_dot_summary(
                    left_gradients=delta_parameters,
                    right_gradients=score_gradients,
                    label=(
                        f"attention retrieval separation update {source_step}->{target_step} "
                        f"{split}/{pair_type}/{record_side}/{score_component}"
                    ),
                )
                metric_row = _attention_score_update_metric_row(
                    source_step=source_step,
                    target_step=target_step,
                    source_checkpoint=source_checkpoint_path,
                    target_checkpoint=target_checkpoint_path,
                    learning_rate=learning_rate,
                    split=split,
                    pair_type=pair_type,
                    head_layer=head_layer,
                    head=head,
                    score_query_role=score_query_role,
                    score_key_role=score_key_label,
                    record_side=record_side,
                    score_component=score_component,
                    actual_summary=actual_summary,
                    source_payload=source_payload,
                    dot_summary=dot_summary,
                    min_error_denominator=min_error_denominator,
                )
                metric_row.update(
                    {
                        "objective": "retrieval_separation",
                        "support_key_role": support_key_role,
                        "distractor_key_role": distractor_key_role,
                        "support_delta_mean": float(actual_summary["support_delta_mean"]),
                        "distractor_delta_mean": float(actual_summary["distractor_delta_mean"]),
                        "num_support_scores": int(actual_summary["num_support_scores"]),
                        "num_distractor_scores": int(actual_summary["num_distractor_scores"]),
                        "support_scores_per_pair_mean": float(actual_summary["support_scores_per_pair_mean"]),
                        "distractor_scores_per_pair_mean": float(actual_summary["distractor_scores_per_pair_mean"]),
                    }
                )
                metric_rows.append(metric_row)
                for group in groups:
                    group_dot_summary = _gradient_dot_summary_for_group(
                        left_gradients=delta_parameters,
                        right_gradients=score_gradients,
                        group=group,
                        label=(
                            f"attention retrieval separation update {source_step}->{target_step} "
                            f"{split}/{pair_type}/{record_side}/{score_component}/{group.group_id}"
                        ),
                    )
                    decomposition_row = _attention_score_update_decomposition_row(
                        metric_row=metric_row,
                        group=group,
                        dot_summary=group_dot_summary,
                    )
                    decomposition_row.update(
                        {
                            "objective": "retrieval_separation",
                            "support_key_role": support_key_role,
                            "distractor_key_role": distractor_key_role,
                        }
                    )
                    decomposition_rows.append(decomposition_row)
    return metric_rows, decomposition_rows, score_rows


def _write_attention_retrieval_separation_update_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    lines = [
        "# Attention Retrieval Separation Update Attribution",
        "",
        "## Calculation",
        "",
        "This report tests whether actual checkpoint-to-checkpoint parameter movement explains relative QK retrieval geometry.",
        "",
        "```text",
        "support_mean = mean_j score(prediction, support_value_j)",
        "distractor_mean = mean_j score(prediction, value_distractor_j)",
        "retrieval_separation = support_mean - distractor_mean",
        "predicted_delta = grad_theta retrieval_separation_component(theta_source) . (theta_target - theta_source)",
        "```",
        "",
        "`q_side` holds source key vectors fixed and differentiates only through the query side.",
        "`k_side` holds the source query vector fixed and differentiates only through the key side.",
        "",
        "## Run",
        "",
        f"- head: `{report['head_label']}`",
        f"- query role: `{report['score_query_role']}`",
        f"- support role: `{report['support_key_role']}`",
        f"- distractor role: `{report['distractor_key_role']}`",
        f"- record sides: `{report['record_sides']}`",
        f"- score components: `{report['score_components']}`",
        f"- intervals: `{summary['intervals']}`",
        "",
        "## Final Metrics",
        "",
        "| record side | component | actual delta | predicted delta | residual | relative error | sign match | support delta | distractor delta |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in summary["final_metric_rows"]:
        lines.append(
            "| `{side}` | `{component}` | {actual:.6g} | {predicted:.6g} | {residual:.6g} | {error:.6g} | `{sign}` | {support:.6g} | {distractor:.6g} |".format(
                side=row["record_side"],
                component=row["score_component"],
                actual=float(row["actual_delta"]),
                predicted=float(row["predicted_delta"]),
                residual=float(row["residual"]),
                error=float(row["relative_error"]),
                sign=bool(row["sign_match"]),
                support=float(row["support_delta_mean"]),
                distractor=float(row["distractor_delta_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Positive Contributions",
            "",
            "| group | component | kind | contribution | cosine |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in summary["final_top_positive_contributions"]:
        cosine = row["update_score_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| `{group}` | `{component}` | `{kind}` | {contribution:.6g} | {cosine} |".format(
                group=row["group_id"],
                component=row["score_component"],
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
            "| group | component | kind | contribution | cosine |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in summary["final_top_negative_contributions"]:
        cosine = row["update_score_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| `{group}` | `{component}` | `{kind}` | {contribution:.6g} | {cosine} |".format(
                group=row["group_id"],
                component=row["score_component"],
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
            f"- score rows: `{report['score_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
        ]
    )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for label, plot_path in plot_paths.items():
            lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_attention_retrieval_separation_update_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    pair_types: list[str],
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    record_sides: list[str] | None = None,
    score_components: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    decomposition_modes: list[str] | None = None,
    reconstruction_tolerance: float = 1.0e-3,
    top_k_groups: int = 24,
    min_error_denominator: float = 1.0e-9,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    unsupported_roles = [
        role
        for role in [score_query_role, support_key_role, distractor_key_role]
        if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported attention roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if support_key_role == distractor_key_role:
        raise ValueError("support_key_role and distractor_key_role must be different.")
    if reconstruction_tolerance <= 0.0:
        raise ValueError("reconstruction_tolerance must be positive.")
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")

    resolved_record_sides = _resolve_attention_score_record_sides(record_sides)
    resolved_score_components = _resolve_attention_score_update_components(score_components)
    resolved_decomposition_modes = _resolve_route_gradient_decomposition_modes(decomposition_modes)
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("attention-retrieval-separation-update-attribution requires at least two checkpoints.")
    source_model = build_model(spec.model, len(vocab.tokens), device)
    target_model = build_model(spec.model, len(vocab.tokens), device)
    if head_layer < 0 or head_layer >= len(source_model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(source_model.blocks) - 1}.")
    if head < 0 or head >= source_model.blocks[head_layer].attn.n_heads:
        raise ValueError(
            f"head {head} outside model range 0..{source_model.blocks[head_layer].attn.n_heads - 1} for layer {head_layer}."
        )
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Attention retrieval separation update attribution constructed no pairs.")

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

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows_path = output_dir / "attention_retrieval_separation_update_attribution_rows.jsonl"
    decomposition_rows_path = output_dir / "attention_retrieval_separation_update_attribution_decomposition_rows.jsonl"
    group_rows_path = output_dir / "attention_retrieval_separation_update_attribution_groups.jsonl"
    score_rows_path = output_dir / "attention_retrieval_separation_update_attribution_score_rows.jsonl"
    pair_rows_path = output_dir / "attention_retrieval_separation_update_attribution_pairs.jsonl"
    progress_path = output_dir / "attention_retrieval_separation_update_attribution_progress.json"
    for partial_path in (
        metric_rows_path,
        decomposition_rows_path,
        group_rows_path,
        score_rows_path,
        pair_rows_path,
        progress_path,
    ):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])
    write_jsonl(group_rows_path, group_rows)

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[attention-retrieval-separation-update-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(pairs)} "
        f"pair_types={pair_types} device={device_name} head={_head_label(head_layer, head)} "
        f"query_role={score_query_role} support={support_key_role} distractor={distractor_key_role} "
        f"record_sides={resolved_record_sides} components={resolved_score_components} groups={len(groups)}",
        flush=True,
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_decomposition_rows: list[dict[str, Any]] = []
    all_score_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        learning_rate = _compute_learning_rate(spec.optimization, source_step)
        print(
            "[attention-retrieval-separation-update-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )
        metric_rows, decomposition_rows, score_rows = _compute_attention_retrieval_separation_update_attribution_interval(
            source_model=source_model,
            target_model=target_model,
            source_checkpoint_path=source_checkpoint_path,
            target_checkpoint_path=target_checkpoint_path,
            pairs=pairs,
            learning_rate=learning_rate,
            head_layer=head_layer,
            head=head,
            score_query_role=score_query_role,
            support_key_role=support_key_role,
            distractor_key_role=distractor_key_role,
            record_sides=resolved_record_sides,
            score_components=resolved_score_components,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            groups=groups,
            reconstruction_tolerance=reconstruction_tolerance,
            min_error_denominator=min_error_denominator,
        )
        for row in metric_rows:
            append_jsonl(metric_rows_path, row)
        for row in decomposition_rows:
            append_jsonl(decomposition_rows_path, row)
        for row in score_rows:
            append_jsonl(score_rows_path, row)
        all_metric_rows.extend(metric_rows)
        all_decomposition_rows.extend(decomposition_rows)
        all_score_rows.extend(score_rows)
        all_row = next(
            row
            for row in metric_rows
            if str(row["split"]) == "__all__"
            and str(row["pair_type"]) == "__all__"
            and str(row["record_side"]) == resolved_record_sides[0]
            and str(row["score_component"]) == resolved_score_components[0]
        )
        print(
            "[attention-retrieval-separation-update-attribution] finished "
            f"{source_step}->{target_step} component={all_row['score_component']} "
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
                "score_rows_path": str(score_rows_path),
            },
        )

    summary = _summarize_attention_score_update_attribution(
        metric_rows=all_metric_rows,
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
    )
    plot_paths: dict[str, Path] = {}
    actual_vs_predicted_plot = _plot_attention_score_update_actual_vs_predicted(
        metric_rows=all_metric_rows,
        output_path=output_dir / "attention_retrieval_separation_update_actual_vs_predicted.svg",
    )
    if actual_vs_predicted_plot is not None:
        plot_paths["actual_vs_predicted"] = actual_vs_predicted_plot
    top_contributions_plot = _plot_attention_score_update_top_contributions(
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
        output_path=output_dir / "attention_retrieval_separation_update_top_contributions.svg",
    )
    if top_contributions_plot is not None:
        plot_paths["top_contributions"] = top_contributions_plot

    report_path = output_dir / "attention_retrieval_separation_update_attribution_report.json"
    markdown_path = output_dir / "attention_retrieval_separation_update_attribution_report.md"
    report = {
        "schema_version": ATTENTION_RETRIEVAL_SEPARATION_UPDATE_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "head_layer": head_layer,
        "head": head,
        "head_label": _head_label(head_layer, head),
        "score_query_role": score_query_role,
        "support_key_role": support_key_role,
        "distractor_key_role": distractor_key_role,
        "score_key_role": _attention_retrieval_separation_key_label(
            support_key_role=support_key_role,
            distractor_key_role=distractor_key_role,
        ),
        "record_sides": resolved_record_sides,
        "score_components": resolved_score_components,
        "pair_types": pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "decomposition": decomposition_summary,
        "reconstruction_tolerance": reconstruction_tolerance,
        "top_k_groups": top_k_groups,
        "min_error_denominator": min_error_denominator,
        "calculation": {
            "score": "q_i . k_j / sqrt(d_head)",
            "retrieval_separation": "mean support score minus mean distractor score, computed per pair then averaged",
            "score_component": "score differentiates through q and k; q_side differentiates only through q; k_side differentiates only through k",
            "actual_delta": "target retrieval_separation minus source retrieval_separation for the selected component",
            "predicted_delta": "grad_theta retrieval_separation_component(theta_source) . (theta_target - theta_source)",
            "residual": "actual_delta - predicted_delta",
            "group_contribution": "grad_group retrieval_separation_component(theta_source) . Delta theta_group",
        },
        "pair_construction": pair_construction,
        "metric_rows_path": str(metric_rows_path),
        "decomposition_rows_path": str(decomposition_rows_path),
        "group_rows_path": str(group_rows_path),
        "score_rows_path": str(score_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_attention_retrieval_separation_update_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
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
            "score_rows_path": str(score_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[attention-retrieval-separation-update-attribution] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return (
        report_path,
        markdown_path,
        metric_rows_path,
        decomposition_rows_path,
        group_rows_path,
        score_rows_path,
        pair_rows_path,
        plot_paths,
    )


ATTENTION_RETRIEVAL_CHAIN_METRICS = [
    "qk_support_score",
    "qk_distractor_score",
    "qk_separation",
    "attention_support_mean",
    "attention_distractor_mean",
    "attention_separation",
    "attention_support_mass",
    "attention_distractor_mass",
    "attention_mass_separation",
    "head_margin_dla",
    "head_answer_logit_dla",
    "head_value_margin_dla",
    "support_ov_value_margin",
    "attended_support_ov_value_margin",
    "answer_margin",
    "answer_loss",
]

ATTENTION_DOWNSTREAM_UPDATE_SCALARS = [
    "attention_separation",
    "attention_mass_separation",
    "head_answer_logit_dla",
    "head_value_margin_dla",
    "support_ov_value_margin",
    "attended_support_ov_value_margin",
    "head_margin_dla_fixed_readout",
    "answer_margin",
    "negative_answer_loss",
]


def _single_vector_value_margin(
    *,
    residual_vector: torch.Tensor,
    correct_token_id: int,
    value_token_ids: torch.Tensor,
    unembed: torch.Tensor,
) -> torch.Tensor:
    matches = (value_token_ids == int(correct_token_id)).nonzero(as_tuple=False).flatten()
    if matches.numel() != 1:
        raise RuntimeError(f"Expected exactly one value-token match for token id {correct_token_id}, got {matches.numel()}.")
    value_scores = torch.matmul(unembed.index_select(0, value_token_ids), residual_vector)
    target_index = int(matches[0].item())
    correct = value_scores[target_index]
    masked = value_scores.clone()
    masked[target_index] = torch.finfo(masked.dtype).min
    return correct - masked.max()


def _attention_retrieval_chain_payload_for_records(
    *,
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    head_layer: int,
    head: int,
    vocab: Vocabulary,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty for attention retrieval chain report.")
    if head_layer < 0 or head_layer >= len(model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(model.blocks) - 1}.")
    block = model.blocks[head_layer]
    if head < 0 or head >= block.attn.n_heads:
        raise ValueError(f"head {head} outside model range 0..{block.attn.n_heads - 1} for layer {head_layer}.")

    batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
    with torch.no_grad():
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_attentions=True,
            return_residual_streams=True,
        )
    if outputs.attentions is None:
        raise RuntimeError("Attention retrieval chain report requires attention probabilities.")
    if outputs.residual_streams is None:
        raise RuntimeError("Attention retrieval chain report requires residual streams.")
    answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
    _validate_single_query_batch(batch=batch, metadata=metadata, label="attention retrieval chain")

    value_token_ids = torch.tensor(vocab.value_token_ids, device=device, dtype=torch.long)
    answer_margins = _value_margin(answer_logits, answer_targets, value_token_ids)
    answer_losses = torch.nn.functional.cross_entropy(answer_logits, answer_targets, reduction="none")
    wrong_token_ids = _best_wrong_value_token_ids(
        logits=answer_logits,
        answer_targets=answer_targets,
        value_token_ids=value_token_ids,
    )
    final_pre_stage = f"layer_{len(model.blocks) - 1}_post_mlp"
    final_pre_vectors = outputs.residual_streams[final_pre_stage][
        metadata["rows"],
        metadata["prediction_positions"],
        :,
    ]
    margin_gradients, recomputed_margins = _margin_gradient_vectors(
        model=model,
        final_residual_vectors=final_pre_vectors,
        correct_token_ids=answer_targets,
        wrong_token_ids=wrong_token_ids,
    )
    if not torch.allclose(recomputed_margins, answer_margins, atol=1e-4, rtol=1e-4):
        max_delta = (recomputed_margins - answer_margins).abs().max().item()
        raise RuntimeError(f"Attention chain margin-gradient check failed: max_delta={max_delta:.6g}")

    if head_layer == 0:
        pre_state = outputs.residual_streams["embedding"]
    else:
        pre_state = outputs.residual_streams[f"layer_{head_layer - 1}_post_mlp"]
    attention_input = block.ln_1(pre_state)
    batch_size, seq_len, _ = attention_input.shape
    head_dim = block.attn.head_dim
    q_all = block.attn.q_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    k_all = block.attn.k_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    v_all = block.attn.v_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    q_head = q_all[:, :, head, :]
    k_head = k_all[:, :, head, :]
    v_head = v_all[:, :, head, :]
    scores = torch.matmul(q_head, k_head.transpose(-2, -1)) / math.sqrt(head_dim)
    attention = outputs.attentions[head_layer][:, head, :, :]
    head_slice = slice(head * head_dim, (head + 1) * head_dim)
    out_head = block.attn.out_proj.weight[:, head_slice]

    return {
        "batch": batch,
        "metadata": metadata,
        "answer_logits": answer_logits,
        "answer_targets": answer_targets,
        "answer_margins": answer_margins,
        "answer_losses": answer_losses,
        "answer_correct": answer_logits.argmax(dim=-1) == answer_targets,
        "margin_gradients": margin_gradients,
        "value_token_ids": value_token_ids,
        "scores": scores,
        "attention": attention,
        "v": v_head,
        "out_head": out_head,
        "unembed": model.lm_head.weight,
    }


def _compute_attention_retrieval_chain_checkpoint(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    record_sides: list[str],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()
    step = int(checkpoint["step"])
    path_step = _checkpoint_step_from_path(checkpoint_path)
    if step != path_step:
        raise RuntimeError(f"Checkpoint step mismatch: payload={step} path={path_step}")

    pair_metric_rows: list[dict[str, Any]] = []
    for start_index in range(0, len(pairs), batch_size):
        pair_batch = pairs[start_index : start_index + batch_size]
        for record_side in record_sides:
            side_key = f"{record_side}_record"
            records = [pair[side_key] for pair in pair_batch]
            payload = _attention_retrieval_chain_payload_for_records(
                model=model,
                records=records,
                head_layer=head_layer,
                head=head,
                vocab=vocab,
                pad_token_id=pad_token_id,
                device=device,
            )
            for pair_index, pair in enumerate(pair_batch):
                batch_row, query_position = _single_attention_position(
                    batch=payload["batch"],
                    metadata=payload["metadata"],
                    flat_index=pair_index,
                    position_role=score_query_role,
                    label="retrieval chain query",
                )
                support_batch_row, support_positions = _attention_key_positions(
                    batch=payload["batch"],
                    metadata=payload["metadata"],
                    flat_index=pair_index,
                    position_role=support_key_role,
                    max_position=query_position,
                )
                distractor_batch_row, distractor_positions = _attention_key_positions(
                    batch=payload["batch"],
                    metadata=payload["metadata"],
                    flat_index=pair_index,
                    position_role=distractor_key_role,
                    max_position=query_position,
                )
                if support_batch_row != batch_row:
                    raise RuntimeError(
                        f"Support role {support_key_role!r} selected batch row {support_batch_row}, "
                        f"but query role {score_query_role!r} selected row {batch_row} for pair {pair['pair_id']}."
                    )
                if distractor_batch_row != batch_row:
                    raise RuntimeError(
                        f"Distractor role {distractor_key_role!r} selected batch row {distractor_batch_row}, "
                        f"but query role {score_query_role!r} selected row {batch_row} for pair {pair['pair_id']}."
                    )

                score_row = payload["scores"][batch_row, query_position, :]
                attention_row = payload["attention"][batch_row, query_position, :]
                support_position_tensor = torch.tensor(support_positions, device=device, dtype=torch.long)
                distractor_position_tensor = torch.tensor(distractor_positions, device=device, dtype=torch.long)
                support_scores = score_row.index_select(0, support_position_tensor)
                distractor_scores = score_row.index_select(0, distractor_position_tensor)
                support_attention = attention_row.index_select(0, support_position_tensor)
                distractor_attention = attention_row.index_select(0, distractor_position_tensor)

                head_output = torch.matmul(attention_row, payload["v"][batch_row])
                head_write = torch.matmul(head_output, payload["out_head"].T)
                support_v = payload["v"][batch_row].index_select(0, support_position_tensor).mean(dim=0)
                support_head_write = torch.matmul(support_v, payload["out_head"].T)
                answer_token_id = int(payload["answer_targets"][pair_index].item())
                support_ov_value_margin = _single_vector_value_margin(
                    residual_vector=support_head_write,
                    correct_token_id=answer_token_id,
                    value_token_ids=payload["value_token_ids"],
                    unembed=payload["unembed"],
                )
                head_value_margin_dla = _single_vector_value_margin(
                    residual_vector=head_write,
                    correct_token_id=answer_token_id,
                    value_token_ids=payload["value_token_ids"],
                    unembed=payload["unembed"],
                )
                head_answer_logit_dla = torch.dot(head_write, payload["unembed"][answer_token_id])
                head_margin_dla = torch.dot(head_write, payload["margin_gradients"][pair_index])

                support_attention_mean = support_attention.mean()
                distractor_attention_mean = distractor_attention.mean()
                support_attention_mass = support_attention.sum()
                distractor_attention_mass = distractor_attention.sum()
                qk_support_score = support_scores.mean()
                qk_distractor_score = distractor_scores.mean()
                pair_metric_rows.append(
                    {
                        "step": step,
                        "checkpoint": str(checkpoint_path),
                        "split": str(pair["split"]),
                        "pair_type": str(pair["pair_type"]),
                        "record_side": record_side,
                        "pair_id": str(pair["pair_id"]),
                        "source_sample_id": str(pair["source_sample_id"]),
                        "source_query_index": int(pair["source_query_index"]),
                        "head_layer": head_layer,
                        "head": head,
                        "head_label": _head_label(head_layer, head),
                        "score_query_role": score_query_role,
                        "support_key_role": support_key_role,
                        "distractor_key_role": distractor_key_role,
                        "query_position": int(query_position),
                        "support_positions": [int(position) for position in support_positions],
                        "distractor_positions": [int(position) for position in distractor_positions],
                        "num_support_positions": int(len(support_positions)),
                        "num_distractor_positions": int(len(distractor_positions)),
                        "qk_support_score": float(qk_support_score.detach().float().cpu().item()),
                        "qk_distractor_score": float(qk_distractor_score.detach().float().cpu().item()),
                        "qk_separation": float((qk_support_score - qk_distractor_score).detach().float().cpu().item()),
                        "attention_support_mean": float(support_attention_mean.detach().float().cpu().item()),
                        "attention_distractor_mean": float(distractor_attention_mean.detach().float().cpu().item()),
                        "attention_separation": float(
                            (support_attention_mean - distractor_attention_mean).detach().float().cpu().item()
                        ),
                        "attention_support_mass": float(support_attention_mass.detach().float().cpu().item()),
                        "attention_distractor_mass": float(distractor_attention_mass.detach().float().cpu().item()),
                        "attention_mass_separation": float(
                            (support_attention_mass - distractor_attention_mass).detach().float().cpu().item()
                        ),
                        "head_margin_dla": float(head_margin_dla.detach().float().cpu().item()),
                        "head_answer_logit_dla": float(head_answer_logit_dla.detach().float().cpu().item()),
                        "head_value_margin_dla": float(head_value_margin_dla.detach().float().cpu().item()),
                        "support_ov_value_margin": float(support_ov_value_margin.detach().float().cpu().item()),
                        "attended_support_ov_value_margin": float(
                            (support_attention_mean * support_ov_value_margin).detach().float().cpu().item()
                        ),
                        "answer_margin": float(payload["answer_margins"][pair_index].detach().float().cpu().item()),
                        "answer_loss": float(payload["answer_losses"][pair_index].detach().float().cpu().item()),
                        "answer_correct": bool(payload["answer_correct"][pair_index].detach().cpu().item()),
                    }
                )
    if not pair_metric_rows:
        raise RuntimeError("Attention retrieval chain checkpoint produced no pair metric rows.")
    return pair_metric_rows


def _aggregate_attention_retrieval_chain_checkpoint_rows(pair_metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not pair_metric_rows:
        raise ValueError("Cannot aggregate empty attention retrieval chain pair rows.")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_metric_rows:
        group_keys = [
            (row["split"], row["pair_type"]),
            ("__all__", row["pair_type"]),
            (row["split"], "__all__"),
            ("__all__", "__all__"),
        ]
        for split, pair_type in group_keys:
            groups[(row["step"], split, pair_type, row["record_side"])].append(row)

    checkpoint_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        step, split, pair_type, record_side = key
        first = rows[0]
        checkpoint_row: dict[str, Any] = {
            "step": int(step),
            "checkpoint": str(first["checkpoint"]),
            "split": split,
            "pair_type": pair_type,
            "record_side": record_side,
            "head_layer": int(first["head_layer"]),
            "head": int(first["head"]),
            "head_label": first["head_label"],
            "score_query_role": first["score_query_role"],
            "support_key_role": first["support_key_role"],
            "distractor_key_role": first["distractor_key_role"],
            "num_pairs": len(rows),
            "answer_accuracy": _fraction(
                sum(1 for row in rows if bool(row["answer_correct"])),
                len(rows),
                "attention retrieval chain answer accuracy",
            ),
        }
        for metric_name in ATTENTION_RETRIEVAL_CHAIN_METRICS:
            values = [float(row[metric_name]) for row in rows]
            checkpoint_row[f"{metric_name}_mean"] = _mean(values)
            checkpoint_row[f"{metric_name}_std"] = _std(values)
        checkpoint_rows.append(checkpoint_row)
    return checkpoint_rows


def _build_attention_retrieval_chain_delta_rows(checkpoint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not checkpoint_rows:
        raise ValueError("Cannot build attention retrieval chain deltas from empty checkpoint rows.")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoint_rows:
        groups[(str(row["split"]), str(row["pair_type"]), str(row["record_side"]))].append(row)

    delta_rows: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        split, pair_type, record_side = key
        sorted_rows = sorted(rows, key=lambda row: int(row["step"]))
        for source_row, target_row in zip(sorted_rows[:-1], sorted_rows[1:], strict=True):
            source_step = int(source_row["step"])
            target_step = int(target_row["step"])
            if target_step <= source_step:
                raise RuntimeError(f"Non-increasing chain checkpoint steps for {key}: {source_step}->{target_step}")
            if int(source_row["num_pairs"]) != int(target_row["num_pairs"]):
                raise RuntimeError(
                    f"Pair count changed across chain interval {source_step}->{target_step} for {key}: "
                    f"{source_row['num_pairs']} vs {target_row['num_pairs']}"
                )
            delta_row: dict[str, Any] = {
                "source_step": source_step,
                "target_step": target_step,
                "step_gap": target_step - source_step,
                "split": split,
                "pair_type": pair_type,
                "record_side": record_side,
                "head_layer": int(source_row["head_layer"]),
                "head": int(source_row["head"]),
                "head_label": source_row["head_label"],
                "score_query_role": source_row["score_query_role"],
                "support_key_role": source_row["support_key_role"],
                "distractor_key_role": source_row["distractor_key_role"],
                "num_pairs": int(source_row["num_pairs"]),
                "delta_answer_accuracy": float(target_row["answer_accuracy"]) - float(source_row["answer_accuracy"]),
            }
            for metric_name in ATTENTION_RETRIEVAL_CHAIN_METRICS:
                delta_row[f"delta_{metric_name}_mean"] = (
                    float(target_row[f"{metric_name}_mean"]) - float(source_row[f"{metric_name}_mean"])
                )
            delta_row["negative_delta_answer_loss_mean"] = -float(delta_row["delta_answer_loss_mean"])
            delta_rows.append(delta_row)
    return delta_rows


def _pearson_correlation_report(
    *,
    x_values: list[float],
    y_values: list[float],
    label: str,
) -> dict[str, Any]:
    if len(x_values) != len(y_values):
        raise ValueError(f"Correlation input length mismatch for {label}: {len(x_values)} vs {len(y_values)}.")
    if len(x_values) < 2:
        return {
            "label": label,
            "value": None,
            "status": "not_computed",
            "reason": "requires_at_least_two_intervals",
            "num_points": len(x_values),
        }
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    x_centered = [value - x_mean for value in x_values]
    y_centered = [value - y_mean for value in y_values]
    x_norm = math.sqrt(sum(value * value for value in x_centered))
    y_norm = math.sqrt(sum(value * value for value in y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return {
            "label": label,
            "value": None,
            "status": "not_computed",
            "reason": "zero_variance",
            "num_points": len(x_values),
        }
    value = sum(x * y for x, y in zip(x_centered, y_centered, strict=True)) / (x_norm * y_norm)
    return {
        "label": label,
        "value": float(value),
        "status": "computed",
        "reason": None,
        "num_points": len(x_values),
    }


def _summarize_attention_retrieval_chain(
    *,
    checkpoint_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    record_sides: list[str],
) -> dict[str, Any]:
    if not checkpoint_rows:
        raise ValueError("Cannot summarize attention retrieval chain without checkpoint rows.")
    steps = sorted({int(row["step"]) for row in checkpoint_rows})
    final_step = steps[-1]
    primary_record_side = record_sides[0]
    primary_rows = [
        row
        for row in checkpoint_rows
        if row["split"] == "__all__" and row["pair_type"] == "__all__" and row["record_side"] == primary_record_side
    ]
    if len(primary_rows) != len(steps):
        raise RuntimeError(
            f"Expected one primary checkpoint row per step for record_side={primary_record_side}, "
            f"got {len(primary_rows)} rows for {len(steps)} steps."
        )
    final_primary = next(row for row in primary_rows if int(row["step"]) == final_step)
    primary_delta_rows = [
        row
        for row in delta_rows
        if row["split"] == "__all__" and row["pair_type"] == "__all__" and row["record_side"] == primary_record_side
    ]
    total_delta: dict[str, float] = {}
    if len(primary_rows) >= 2:
        first_primary = sorted(primary_rows, key=lambda row: int(row["step"]))[0]
        for metric_name in ATTENTION_RETRIEVAL_CHAIN_METRICS:
            total_delta[f"delta_{metric_name}_mean"] = (
                float(final_primary[f"{metric_name}_mean"]) - float(first_primary[f"{metric_name}_mean"])
            )
        total_delta["negative_delta_answer_loss_mean"] = -float(total_delta["delta_answer_loss_mean"])

    correlations = {
        "qk_to_attention": _pearson_correlation_report(
            x_values=[float(row["delta_qk_separation_mean"]) for row in primary_delta_rows],
            y_values=[float(row["delta_attention_separation_mean"]) for row in primary_delta_rows],
            label="delta_qk_separation_mean vs delta_attention_separation_mean",
        ),
        "attention_to_head_margin_dla": _pearson_correlation_report(
            x_values=[float(row["delta_attention_separation_mean"]) for row in primary_delta_rows],
            y_values=[float(row["delta_head_margin_dla_mean"]) for row in primary_delta_rows],
            label="delta_attention_separation_mean vs delta_head_margin_dla_mean",
        ),
        "head_margin_dla_to_answer_margin": _pearson_correlation_report(
            x_values=[float(row["delta_head_margin_dla_mean"]) for row in primary_delta_rows],
            y_values=[float(row["delta_answer_margin_mean"]) for row in primary_delta_rows],
            label="delta_head_margin_dla_mean vs delta_answer_margin_mean",
        ),
        "answer_margin_to_loss_reduction": _pearson_correlation_report(
            x_values=[float(row["delta_answer_margin_mean"]) for row in primary_delta_rows],
            y_values=[float(row["negative_delta_answer_loss_mean"]) for row in primary_delta_rows],
            label="delta_answer_margin_mean vs -delta_answer_loss_mean",
        ),
    }
    return {
        "num_checkpoints": len(steps),
        "steps": steps,
        "num_checkpoint_rows": len(checkpoint_rows),
        "num_delta_rows": len(delta_rows),
        "primary_record_side": primary_record_side,
        "final_primary_row": final_primary,
        "primary_delta_rows": primary_delta_rows,
        "total_primary_delta": total_delta,
        "correlations": correlations,
    }


def _plot_attention_retrieval_chain_trajectory(
    *,
    checkpoint_rows: list[dict[str, Any]],
    record_side: str,
    output_path: Path,
) -> Path | None:
    rows = sorted(
        [
            row
            for row in checkpoint_rows
            if row["split"] == "__all__" and row["pair_type"] == "__all__" and row["record_side"] == record_side
        ],
        key=lambda row: int(row["step"]),
    )
    if not rows:
        return None
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    steps = [int(row["step"]) for row in rows]
    plotted = [
        ("qk_separation_mean", "QK support - distractor score"),
        ("attention_separation_mean", "attention support - distractor prob"),
        ("head_margin_dla_mean", "head direct margin attribution"),
        ("answer_margin_mean", "answer margin"),
        ("answer_loss_mean", "answer loss"),
    ]
    for axis, (field_name, label) in zip(axes, plotted, strict=True):
        axis.plot(steps, [float(row[field_name]) for row in rows], marker="o")
        axis.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("checkpoint step")
    fig.suptitle(f"Attention retrieval chain for {record_side} records")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_attention_retrieval_chain_deltas(
    *,
    delta_rows: list[dict[str, Any]],
    record_side: str,
    output_path: Path,
) -> Path | None:
    rows = [
        row
        for row in delta_rows
        if row["split"] == "__all__" and row["pair_type"] == "__all__" and row["record_side"] == record_side
    ]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plots = [
        ("delta_qk_separation_mean", "delta_attention_separation_mean", "QK -> attention"),
        ("delta_attention_separation_mean", "delta_head_margin_dla_mean", "attention -> head DLA"),
        ("delta_head_margin_dla_mean", "delta_answer_margin_mean", "head DLA -> margin"),
        ("delta_answer_margin_mean", "negative_delta_answer_loss_mean", "margin -> loss reduction"),
    ]
    for axis, (x_field, y_field, title) in zip(axes.flatten(), plots, strict=True):
        axis.scatter([float(row[x_field]) for row in rows], [float(row[y_field]) for row in rows])
        for row in rows:
            axis.annotate(
                f"{row['source_step']}->{row['target_step']}",
                (float(row[x_field]), float(row[y_field])),
                fontsize=7,
                alpha=0.75,
            )
        axis.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        axis.axvline(0.0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_xlabel(x_field)
        axis.set_ylabel(y_field)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_attention_retrieval_chain_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    final_row = summary["final_primary_row"]
    lines = [
        "# Attention Retrieval Chain Report",
        "",
        "## Question",
        "",
        "This report checks whether the measured QK retrieval geometry is connected to the model's answer behavior on the same causal pairs.",
        "",
        "## Calculation",
        "",
        "`QK separation = mean score(prediction, support) - mean score(prediction, distractors)`",
        "",
        "`attention separation = mean attention(prediction, support) - mean attention(prediction, distractors)`",
        "",
        "`head margin DLA = L2H1 output at the prediction position dotted with the final answer-margin gradient`",
        "",
        "`answer margin = correct value logit - best wrong value logit`",
        "",
        "The expected chain is:",
        "",
        "`Delta QK separation -> Delta attention separation -> Delta head DLA -> Delta answer margin -> -Delta loss`",
        "",
        "## Run",
        "",
        f"- head: `{report['head_label']}`",
        f"- query role: `{report['score_query_role']}`",
        f"- support role: `{report['support_key_role']}`",
        f"- distractor role: `{report['distractor_key_role']}`",
        f"- record sides: `{report['record_sides']}`",
        f"- checkpoints: `{summary['steps']}`",
        "",
        "## Final Primary Row",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| QK separation | {float(final_row['qk_separation_mean']):.6f} |",
        f"| attention separation | {float(final_row['attention_separation_mean']):.6f} |",
        f"| head margin DLA | {float(final_row['head_margin_dla_mean']):.6f} |",
        f"| answer margin | {float(final_row['answer_margin_mean']):.6f} |",
        f"| answer loss | {float(final_row['answer_loss_mean']):.6f} |",
        f"| answer accuracy | {float(final_row['answer_accuracy']):.6f} |",
        "",
        "## Primary Interval Deltas",
        "",
        "| interval | d QK sep | d attention sep | d head DLA | d answer margin | -d loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["primary_delta_rows"]:
        lines.append(
            "| {source}->{target} | {qk:.6f} | {attn:.6f} | {dla:.6f} | {margin:.6f} | {loss:.6f} |".format(
                source=int(row["source_step"]),
                target=int(row["target_step"]),
                qk=float(row["delta_qk_separation_mean"]),
                attn=float(row["delta_attention_separation_mean"]),
                dla=float(row["delta_head_margin_dla_mean"]),
                margin=float(row["delta_answer_margin_mean"]),
                loss=float(row["negative_delta_answer_loss_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- checkpoint rows: `{report['checkpoint_rows_path']}`",
            f"- delta rows: `{report['delta_rows_path']}`",
            f"- pair metric rows: `{report['pair_metric_rows_path']}`",
            f"- pair metadata: `{report['pair_rows_path']}`",
        ]
    )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for label, plot_path in plot_paths.items():
            lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_attention_retrieval_chain_report(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    pair_types: list[str],
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    record_sides: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    unsupported_roles = [
        role
        for role in [score_query_role, support_key_role, distractor_key_role]
        if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported attention roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if support_key_role == distractor_key_role:
        raise ValueError("support_key_role and distractor_key_role must be different.")

    resolved_record_sides = _resolve_attention_score_record_sides(record_sides)
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("attention-retrieval-chain-report requires at least two checkpoints.")
    model = build_model(spec.model, len(vocab.tokens), device)
    if head_layer < 0 or head_layer >= len(model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(model.blocks) - 1}.")
    if head < 0 or head >= model.blocks[head_layer].attn.n_heads:
        raise ValueError(
            f"head {head} outside model range 0..{model.blocks[head_layer].attn.n_heads - 1} for layer {head_layer}."
        )
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Attention retrieval chain report constructed no pairs.")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_rows_path = output_dir / "attention_retrieval_chain_checkpoint_rows.jsonl"
    delta_rows_path = output_dir / "attention_retrieval_chain_delta_rows.jsonl"
    pair_metric_rows_path = output_dir / "attention_retrieval_chain_pair_rows.jsonl"
    pair_rows_path = output_dir / "attention_retrieval_chain_pairs.jsonl"
    progress_path = output_dir / "attention_retrieval_chain_progress.json"
    for partial_path in (checkpoint_rows_path, delta_rows_path, pair_metric_rows_path, pair_rows_path, progress_path):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])

    print(
        "[attention-retrieval-chain-report] "
        f"checkpoints={len(checkpoints)} pairs={len(pairs)} pair_types={pair_types} device={device_name} "
        f"head={_head_label(head_layer, head)} query_role={score_query_role} support={support_key_role} "
        f"distractor={distractor_key_role} record_sides={resolved_record_sides}",
        flush=True,
    )
    all_pair_metric_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        step = _checkpoint_step_from_path(checkpoint_path)
        print(
            "[attention-retrieval-chain-report] starting "
            f"{checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        pair_metric_rows = _compute_attention_retrieval_chain_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            pairs=pairs,
            vocab=vocab,
            head_layer=head_layer,
            head=head,
            score_query_role=score_query_role,
            support_key_role=support_key_role,
            distractor_key_role=distractor_key_role,
            record_sides=resolved_record_sides,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
        )
        for row in pair_metric_rows:
            append_jsonl(pair_metric_rows_path, row)
        all_pair_metric_rows.extend(pair_metric_rows)
        checkpoint_metric_rows = _aggregate_attention_retrieval_chain_checkpoint_rows(pair_metric_rows)
        primary = next(
            row
            for row in checkpoint_metric_rows
            if row["split"] == "__all__"
            and row["pair_type"] == "__all__"
            and row["record_side"] == resolved_record_sides[0]
        )
        print(
            "[attention-retrieval-chain-report] finished "
            f"step={step} qk_sep={float(primary['qk_separation_mean']):.6g} "
            f"attention_sep={float(primary['attention_separation_mean']):.6g} "
            f"head_dla={float(primary['head_margin_dla_mean']):.6g} "
            f"margin={float(primary['answer_margin_mean']):.6g} "
            f"loss={float(primary['answer_loss_mean']):.6g}",
            flush=True,
        )
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_checkpoints": checkpoint_index,
                "total_checkpoints": len(checkpoints),
                "last_step": step,
                "pair_metric_rows_path": str(pair_metric_rows_path),
            },
        )

    checkpoint_rows = _aggregate_attention_retrieval_chain_checkpoint_rows(all_pair_metric_rows)
    delta_rows = _build_attention_retrieval_chain_delta_rows(checkpoint_rows)
    write_jsonl(checkpoint_rows_path, checkpoint_rows)
    write_jsonl(delta_rows_path, delta_rows)
    summary = _summarize_attention_retrieval_chain(
        checkpoint_rows=checkpoint_rows,
        delta_rows=delta_rows,
        record_sides=resolved_record_sides,
    )

    plot_paths: dict[str, Path] = {}
    trajectory_plot = _plot_attention_retrieval_chain_trajectory(
        checkpoint_rows=checkpoint_rows,
        record_side=resolved_record_sides[0],
        output_path=output_dir / "attention_retrieval_chain_trajectory.svg",
    )
    if trajectory_plot is not None:
        plot_paths["trajectory"] = trajectory_plot
    delta_plot = _plot_attention_retrieval_chain_deltas(
        delta_rows=delta_rows,
        record_side=resolved_record_sides[0],
        output_path=output_dir / "attention_retrieval_chain_deltas.svg",
    )
    if delta_plot is not None:
        plot_paths["deltas"] = delta_plot

    report_path = output_dir / "attention_retrieval_chain_report.json"
    markdown_path = output_dir / "attention_retrieval_chain_report.md"
    report = {
        "schema_version": ATTENTION_RETRIEVAL_CHAIN_REPORT_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "head_layer": head_layer,
        "head": head,
        "head_label": _head_label(head_layer, head),
        "score_query_role": score_query_role,
        "support_key_role": support_key_role,
        "distractor_key_role": distractor_key_role,
        "record_sides": resolved_record_sides,
        "pair_types": pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "calculation": {
            "qk_separation": "mean score(query, support) - mean score(query, distractors)",
            "attention_separation": "mean attention(query, support) - mean attention(query, distractors)",
            "head_margin_dla": "head output at query position dotted with final answer-margin gradient",
            "answer_margin": "correct value logit minus best wrong value logit",
            "chain": "Delta QK separation -> Delta attention separation -> Delta head DLA -> Delta answer margin -> -Delta loss",
        },
        "pair_construction": pair_construction,
        "checkpoint_rows_path": str(checkpoint_rows_path),
        "delta_rows_path": str(delta_rows_path),
        "pair_metric_rows_path": str(pair_metric_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_attention_retrieval_chain_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_checkpoints": len(checkpoints),
            "total_checkpoints": len(checkpoints),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "checkpoint_rows_path": str(checkpoint_rows_path),
            "delta_rows_path": str(delta_rows_path),
            "pair_metric_rows_path": str(pair_metric_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[attention-retrieval-chain-report] complete report={report_path} rows={checkpoint_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, checkpoint_rows_path, delta_rows_path, pair_metric_rows_path, pair_rows_path, plot_paths


def _resolve_attention_downstream_update_scalars(scalars: list[str] | None) -> list[str]:
    if scalars is None:
        return list(ATTENTION_DOWNSTREAM_UPDATE_SCALARS)
    if not scalars:
        raise ValueError("attention downstream scalars must not be empty when provided.")
    unsupported = [scalar for scalar in scalars if scalar not in ATTENTION_DOWNSTREAM_UPDATE_SCALARS]
    if unsupported:
        raise ValueError(
            f"Unsupported attention downstream scalar(s) {unsupported}; "
            f"expected one of {ATTENTION_DOWNSTREAM_UPDATE_SCALARS}."
        )
    return sorted(set(scalars), key=scalars.index)


def _attention_downstream_payload_for_records(
    *,
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    head_layer: int,
    head: int,
    vocab: Vocabulary,
    pad_token_id: int,
    device: torch.device,
    track_grad: bool,
) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty for attention downstream update attribution.")
    if head_layer < 0 or head_layer >= len(model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(model.blocks) - 1}.")
    block = model.blocks[head_layer]
    if head < 0 or head >= block.attn.n_heads:
        raise ValueError(f"head {head} outside model range 0..{block.attn.n_heads - 1} for layer {head_layer}.")

    batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
    if track_grad:
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_attentions=True,
            return_residual_streams=True,
        )
    else:
        with torch.no_grad():
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_attentions=True,
                return_residual_streams=True,
            )
    if outputs.attentions is None:
        raise RuntimeError("Attention downstream update attribution requires attention probabilities.")
    if outputs.residual_streams is None:
        raise RuntimeError("Attention downstream update attribution requires residual streams.")

    answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
    _validate_single_query_batch(batch=batch, metadata=metadata, label="attention downstream update attribution")
    value_token_ids = torch.tensor(vocab.value_token_ids, device=device, dtype=torch.long)
    answer_margins = _value_margin(answer_logits, answer_targets, value_token_ids)
    answer_losses = torch.nn.functional.cross_entropy(answer_logits, answer_targets, reduction="none")
    wrong_token_ids = _best_wrong_value_token_ids(
        logits=answer_logits,
        answer_targets=answer_targets,
        value_token_ids=value_token_ids,
    )
    final_pre_stage = f"layer_{len(model.blocks) - 1}_post_mlp"
    final_pre_vectors = outputs.residual_streams[final_pre_stage][
        metadata["rows"],
        metadata["prediction_positions"],
        :,
    ]
    margin_gradients, recomputed_margins = _margin_gradient_vectors(
        model=model,
        final_residual_vectors=final_pre_vectors,
        correct_token_ids=answer_targets,
        wrong_token_ids=wrong_token_ids,
    )
    if not torch.allclose(recomputed_margins, answer_margins.detach(), atol=1.0e-4, rtol=1.0e-4):
        max_delta = (recomputed_margins - answer_margins.detach()).abs().max().item()
        raise RuntimeError(f"Attention downstream margin-gradient check failed: max_delta={max_delta:.6g}")

    if head_layer == 0:
        pre_state = outputs.residual_streams["embedding"]
    else:
        pre_state = outputs.residual_streams[f"layer_{head_layer - 1}_post_mlp"]
    attention_input = block.ln_1(pre_state)
    batch_size, seq_len, _ = attention_input.shape
    head_dim = block.attn.head_dim
    q_all = block.attn.q_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    k_all = block.attn.k_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    v_all = block.attn.v_proj(attention_input).view(batch_size, seq_len, block.attn.n_heads, head_dim)
    q_head = q_all[:, :, head, :]
    k_head = k_all[:, :, head, :]
    v_head = v_all[:, :, head, :]
    scores = torch.matmul(q_head, k_head.transpose(-2, -1)) / math.sqrt(head_dim)
    attention = outputs.attentions[head_layer][:, head, :, :]
    head_slice = slice(head * head_dim, (head + 1) * head_dim)
    out_head = block.attn.out_proj.weight[:, head_slice]

    return {
        "batch": batch,
        "metadata": metadata,
        "answer_logits": answer_logits,
        "answer_targets": answer_targets,
        "answer_margins": answer_margins,
        "answer_losses": answer_losses,
        "answer_correct": answer_logits.argmax(dim=-1) == answer_targets,
        "margin_gradients": margin_gradients,
        "value_token_ids": value_token_ids,
        "scores": scores,
        "attention": attention,
        "v": v_head,
        "out_head": out_head,
        "unembed": model.lm_head.weight,
    }


def _attention_downstream_scalar_entries_for_payload(
    *,
    payload: dict[str, Any],
    pairs: list[dict[str, Any]],
    record_side: str,
    scalar_names: list[str],
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    fixed_margin_gradients: torch.Tensor | None = None,
) -> list[dict[str, Any]]:
    unsupported = [scalar for scalar in scalar_names if scalar not in ATTENTION_DOWNSTREAM_UPDATE_SCALARS]
    if unsupported:
        raise ValueError(
            f"Unsupported attention downstream scalar(s) {unsupported}; "
            f"expected one of {ATTENTION_DOWNSTREAM_UPDATE_SCALARS}."
        )
    if fixed_margin_gradients is not None and fixed_margin_gradients.shape != payload["margin_gradients"].shape:
        raise ValueError(
            "fixed_margin_gradients shape mismatch: "
            f"{tuple(fixed_margin_gradients.shape)} vs {tuple(payload['margin_gradients'].shape)}"
        )

    entries: list[dict[str, Any]] = []
    device = payload["answer_targets"].device
    for pair_index, pair in enumerate(pairs):
        batch_row, query_position = _single_attention_position(
            batch=payload["batch"],
            metadata=payload["metadata"],
            flat_index=pair_index,
            position_role=score_query_role,
            label="attention downstream query",
        )
        support_batch_row, support_positions = _attention_key_positions(
            batch=payload["batch"],
            metadata=payload["metadata"],
            flat_index=pair_index,
            position_role=support_key_role,
            max_position=query_position,
        )
        distractor_batch_row, distractor_positions = _attention_key_positions(
            batch=payload["batch"],
            metadata=payload["metadata"],
            flat_index=pair_index,
            position_role=distractor_key_role,
            max_position=query_position,
        )
        if support_batch_row != batch_row:
            raise RuntimeError(
                f"Support role {support_key_role!r} selected batch row {support_batch_row}, "
                f"but query role {score_query_role!r} selected row {batch_row} for pair {pair['pair_id']}."
            )
        if distractor_batch_row != batch_row:
            raise RuntimeError(
                f"Distractor role {distractor_key_role!r} selected batch row {distractor_batch_row}, "
                f"but query role {score_query_role!r} selected row {batch_row} for pair {pair['pair_id']}."
            )
        support_position_tensor = torch.tensor(support_positions, device=device, dtype=torch.long)
        distractor_position_tensor = torch.tensor(distractor_positions, device=device, dtype=torch.long)
        attention_row = payload["attention"][batch_row, query_position, :]
        support_attention = attention_row.index_select(0, support_position_tensor)
        distractor_attention = attention_row.index_select(0, distractor_position_tensor)
        support_attention_mean = support_attention.mean()
        distractor_attention_mean = distractor_attention.mean()
        attention_separation = support_attention_mean - distractor_attention_mean
        attention_mass_separation = support_attention.sum() - distractor_attention.sum()

        head_output = torch.matmul(attention_row, payload["v"][batch_row])
        head_write = torch.matmul(head_output, payload["out_head"].T)
        support_v = payload["v"][batch_row].index_select(0, support_position_tensor).mean(dim=0)
        support_head_write = torch.matmul(support_v, payload["out_head"].T)

        answer_token_id = int(payload["answer_targets"][pair_index].item())
        support_ov_value_margin = _single_vector_value_margin(
            residual_vector=support_head_write,
            correct_token_id=answer_token_id,
            value_token_ids=payload["value_token_ids"],
            unembed=payload["unembed"],
        )
        head_answer_logit_dla = torch.dot(head_write, payload["unembed"][answer_token_id])
        head_value_margin_dla = _single_vector_value_margin(
            residual_vector=head_write,
            correct_token_id=answer_token_id,
            value_token_ids=payload["value_token_ids"],
            unembed=payload["unembed"],
        )
        margin_readout = (
            fixed_margin_gradients[pair_index]
            if fixed_margin_gradients is not None
            else payload["margin_gradients"][pair_index]
        )
        head_margin_dla_fixed_readout = torch.dot(head_write, margin_readout.detach())
        scalar_tensors = {
            "attention_separation": attention_separation,
            "attention_mass_separation": attention_mass_separation,
            "head_answer_logit_dla": head_answer_logit_dla,
            "head_value_margin_dla": head_value_margin_dla,
            "support_ov_value_margin": support_ov_value_margin,
            "attended_support_ov_value_margin": support_attention_mean * support_ov_value_margin,
            "head_margin_dla_fixed_readout": head_margin_dla_fixed_readout,
            "answer_margin": payload["answer_margins"][pair_index],
            "negative_answer_loss": -payload["answer_losses"][pair_index],
        }
        for scalar_name in scalar_names:
            entries.append(
                {
                    "pair_id": str(pair["pair_id"]),
                    "split": str(pair["split"]),
                    "pair_type": str(pair["pair_type"]),
                    "record_side": record_side,
                    "source_sample_id": str(pair["source_sample_id"]),
                    "source_query_index": int(pair["source_query_index"]),
                    "query_position": int(query_position),
                    "support_positions": [int(position) for position in support_positions],
                    "distractor_positions": [int(position) for position in distractor_positions],
                    "num_support_positions": int(len(support_positions)),
                    "num_distractor_positions": int(len(distractor_positions)),
                    "answer_token_id": answer_token_id,
                    "answer_correct": bool(payload["answer_correct"][pair_index].detach().cpu().item()),
                    "scalar_name": scalar_name,
                    "value_tensor": scalar_tensors[scalar_name],
                }
            )
    return entries


def _attention_downstream_entries_to_rows(
    *,
    entries: list[dict[str, Any]],
    step: int,
    checkpoint_path: Path,
    head_layer: int,
    head: int,
    head_label: str,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        value_tensor = entry["value_tensor"]
        if not isinstance(value_tensor, torch.Tensor):
            raise TypeError(f"Downstream scalar entry must contain a tensor value: {entry['scalar_name']}")
        rows.append(
            {
                "step": step,
                "checkpoint": str(checkpoint_path),
                "split": entry["split"],
                "pair_type": entry["pair_type"],
                "record_side": entry["record_side"],
                "pair_id": entry["pair_id"],
                "source_sample_id": entry["source_sample_id"],
                "source_query_index": int(entry["source_query_index"]),
                "head_layer": head_layer,
                "head": head,
                "head_label": head_label,
                "score_query_role": score_query_role,
                "support_key_role": support_key_role,
                "distractor_key_role": distractor_key_role,
                "query_position": int(entry["query_position"]),
                "support_positions": entry["support_positions"],
                "distractor_positions": entry["distractor_positions"],
                "num_support_positions": int(entry["num_support_positions"]),
                "num_distractor_positions": int(entry["num_distractor_positions"]),
                "answer_token_id": int(entry["answer_token_id"]),
                "answer_correct": bool(entry["answer_correct"]),
                "scalar_name": entry["scalar_name"],
                "value": float(value_tensor.detach().float().cpu().item()),
            }
        )
    return rows


def _compute_attention_downstream_actual_rows(
    *,
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    record_sides: list[str],
    scalar_names: list[str],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    source_checkpoint = load_checkpoint(source_checkpoint_path, device)
    target_checkpoint = load_checkpoint(target_checkpoint_path, device)
    load_model_state(source_model, source_checkpoint["model_state"])
    load_model_state(target_model, target_checkpoint["model_state"])
    source_step = int(source_checkpoint["step"])
    target_step = int(target_checkpoint["step"])
    if source_step != _checkpoint_step_from_path(source_checkpoint_path):
        raise RuntimeError(f"Source checkpoint step mismatch: payload={source_step} path={source_checkpoint_path}")
    if target_step != _checkpoint_step_from_path(target_checkpoint_path):
        raise RuntimeError(f"Target checkpoint step mismatch: payload={target_step} path={target_checkpoint_path}")

    source_model.eval()
    target_model.eval()
    actual_rows: list[dict[str, Any]] = []
    head_label = _head_label(head_layer, head)
    for start_index in range(0, len(pairs), batch_size):
        pair_batch = pairs[start_index : start_index + batch_size]
        for record_side in record_sides:
            side_key = f"{record_side}_record"
            records = [pair[side_key] for pair in pair_batch]
            source_payload = _attention_downstream_payload_for_records(
                model=source_model,
                records=records,
                head_layer=head_layer,
                head=head,
                vocab=vocab,
                pad_token_id=pad_token_id,
                device=device,
                track_grad=False,
            )
            target_payload = _attention_downstream_payload_for_records(
                model=target_model,
                records=records,
                head_layer=head_layer,
                head=head,
                vocab=vocab,
                pad_token_id=pad_token_id,
                device=device,
                track_grad=False,
            )
            source_entries = _attention_downstream_scalar_entries_for_payload(
                payload=source_payload,
                pairs=pair_batch,
                record_side=record_side,
                scalar_names=scalar_names,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
                fixed_margin_gradients=source_payload["margin_gradients"],
            )
            target_entries = _attention_downstream_scalar_entries_for_payload(
                payload=target_payload,
                pairs=pair_batch,
                record_side=record_side,
                scalar_names=scalar_names,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
                fixed_margin_gradients=source_payload["margin_gradients"],
            )
            source_rows = _attention_downstream_entries_to_rows(
                entries=source_entries,
                step=source_step,
                checkpoint_path=source_checkpoint_path,
                head_layer=head_layer,
                head=head,
                head_label=head_label,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
            )
            target_rows = _attention_downstream_entries_to_rows(
                entries=target_entries,
                step=target_step,
                checkpoint_path=target_checkpoint_path,
                head_layer=head_layer,
                head=head,
                head_label=head_label,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
            )
            source_by_key = {
                (row["pair_id"], row["record_side"], row["scalar_name"]): row
                for row in source_rows
            }
            target_by_key = {
                (row["pair_id"], row["record_side"], row["scalar_name"]): row
                for row in target_rows
            }
            if set(source_by_key) != set(target_by_key):
                missing_target = sorted(set(source_by_key) - set(target_by_key))
                extra_target = sorted(set(target_by_key) - set(source_by_key))
                raise RuntimeError(
                    "Source/target downstream scalar row keys differ: "
                    f"missing_target={missing_target} extra_target={extra_target}"
                )
            for key in sorted(source_by_key):
                source_row = source_by_key[key]
                target_row = target_by_key[key]
                actual_rows.append(
                    {
                        "source_step": source_step,
                        "target_step": target_step,
                        "step_gap": target_step - source_step,
                        "source_checkpoint": str(source_checkpoint_path),
                        "target_checkpoint": str(target_checkpoint_path),
                        "split": source_row["split"],
                        "pair_type": source_row["pair_type"],
                        "record_side": source_row["record_side"],
                        "pair_id": source_row["pair_id"],
                        "source_sample_id": source_row["source_sample_id"],
                        "source_query_index": int(source_row["source_query_index"]),
                        "head_layer": head_layer,
                        "head": head,
                        "head_label": head_label,
                        "score_query_role": score_query_role,
                        "support_key_role": support_key_role,
                        "distractor_key_role": distractor_key_role,
                        "query_position": int(source_row["query_position"]),
                        "support_positions": source_row["support_positions"],
                        "distractor_positions": source_row["distractor_positions"],
                        "num_support_positions": int(source_row["num_support_positions"]),
                        "num_distractor_positions": int(source_row["num_distractor_positions"]),
                        "answer_token_id": int(source_row["answer_token_id"]),
                        "source_answer_correct": bool(source_row["answer_correct"]),
                        "target_answer_correct": bool(target_row["answer_correct"]),
                        "scalar_name": source_row["scalar_name"],
                        "source_value": float(source_row["value"]),
                        "target_value": float(target_row["value"]),
                        "actual_delta": float(target_row["value"]) - float(source_row["value"]),
                    }
                )
    if not actual_rows:
        raise RuntimeError("Attention downstream update attribution produced no actual scalar rows.")
    return actual_rows


def _attention_downstream_actual_summary(
    *,
    actual_rows: list[dict[str, Any]],
    split: str,
    pair_type: str,
    record_side: str,
    scalar_name: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in actual_rows
        if (split == "__all__" or str(row["split"]) == split)
        and (pair_type == "__all__" or str(row["pair_type"]) == pair_type)
        and str(row["record_side"]) == record_side
        and str(row["scalar_name"]) == scalar_name
    ]
    if not rows:
        raise RuntimeError(
            f"No downstream actual rows for split={split!r} pair_type={pair_type!r} "
            f"record_side={record_side!r} scalar={scalar_name!r}."
        )
    source_values = [float(row["source_value"]) for row in rows]
    target_values = [float(row["target_value"]) for row in rows]
    actual_delta_values = [float(row["actual_delta"]) for row in rows]
    return {
        "num_entries": len(rows),
        "num_unique_pairs": len({str(row["pair_id"]) for row in rows}),
        "source_value": _mean(source_values),
        "source_value_std": _std(source_values),
        "target_value": _mean(target_values),
        "target_value_std": _std(target_values),
        "actual_delta": _mean(actual_delta_values),
        "actual_delta_abs_mean": _mean([abs(value) for value in actual_delta_values]),
        "actual_delta_std": _std(actual_delta_values),
        "source_accuracy": _fraction(
            sum(1 for row in rows if bool(row["source_answer_correct"])),
            len(rows),
            "downstream source accuracy",
        ),
        "target_accuracy": _fraction(
            sum(1 for row in rows if bool(row["target_answer_correct"])),
            len(rows),
            "downstream target accuracy",
        ),
    }


def _compute_attention_downstream_scalar_gradients_for_pairs(
    *,
    model: torch.nn.Module,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    record_side: str,
    scalar_names: list[str],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    if not pairs:
        raise ValueError("pairs must not be empty for downstream scalar gradient computation.")
    if record_side not in ATTENTION_SCORE_RECORD_SIDES:
        raise ValueError(f"Unsupported record side {record_side!r}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    model.eval()
    model.zero_grad(set_to_none=True)
    scalar_sums: dict[str, torch.Tensor | None] = {scalar_name: None for scalar_name in scalar_names}
    scalar_values: dict[str, list[float]] = {scalar_name: [] for scalar_name in scalar_names}
    scalar_entry_counts: dict[str, int] = {scalar_name: 0 for scalar_name in scalar_names}
    side_key = f"{record_side}_record"
    num_batches = 0

    for start_index in range(0, len(pairs), batch_size):
        pair_batch = pairs[start_index : start_index + batch_size]
        records = [pair[side_key] for pair in pair_batch]
        payload = _attention_downstream_payload_for_records(
            model=model,
            records=records,
            head_layer=head_layer,
            head=head,
            vocab=vocab,
            pad_token_id=pad_token_id,
            device=device,
            track_grad=True,
        )
        entries = _attention_downstream_scalar_entries_for_payload(
            payload=payload,
            pairs=pair_batch,
            record_side=record_side,
            scalar_names=scalar_names,
            score_query_role=score_query_role,
            support_key_role=support_key_role,
            distractor_key_role=distractor_key_role,
            fixed_margin_gradients=payload["margin_gradients"],
        )
        for entry in entries:
            scalar_name = str(entry["scalar_name"])
            value_tensor = entry["value_tensor"]
            if not isinstance(value_tensor, torch.Tensor):
                raise TypeError(f"Downstream scalar value must be a tensor for {scalar_name}.")
            scalar_sums[scalar_name] = (
                value_tensor if scalar_sums[scalar_name] is None else scalar_sums[scalar_name] + value_tensor
            )
            scalar_values[scalar_name].append(float(value_tensor.detach().float().cpu().item()))
            scalar_entry_counts[scalar_name] += 1
        num_batches += 1

    payloads: dict[str, dict[str, Any]] = {}
    for scalar_index, scalar_name in enumerate(scalar_names):
        total_scalar = scalar_sums[scalar_name]
        entry_count = scalar_entry_counts[scalar_name]
        if total_scalar is None or entry_count <= 0:
            raise RuntimeError(f"Downstream scalar gradient produced no values for scalar={scalar_name!r}.")
        mean_scalar = total_scalar / float(entry_count)
        if not mean_scalar.requires_grad:
            raise RuntimeError(f"Downstream scalar {scalar_name!r} does not require grad.")
        model.zero_grad(set_to_none=True)
        mean_scalar.backward(retain_graph=scalar_index < len(scalar_names) - 1)
        gradients, zero_gradient_parameter_names = _parameter_gradients(model=model, require_all=False)
        model.zero_grad(set_to_none=True)
        values = scalar_values[scalar_name]
        payloads[scalar_name] = {
            "scalar_value": float(mean_scalar.detach().float().cpu().item()),
            "scalar_value_abs_mean": _mean([abs(value) for value in values]),
            "scalar_value_std": _std(values),
            "num_entries": entry_count,
            "num_pairs": len(pairs),
            "num_batches": num_batches,
            "zero_gradient_parameter_names": zero_gradient_parameter_names,
            "gradients": gradients,
        }
    return payloads


def _attention_downstream_update_metric_row(
    *,
    source_step: int,
    target_step: int,
    source_checkpoint: Path,
    target_checkpoint: Path,
    learning_rate: float,
    split: str,
    pair_type: str,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    record_side: str,
    scalar_name: str,
    actual_summary: dict[str, Any],
    source_payload: dict[str, Any],
    dot_summary: dict[str, float | int | None],
    min_error_denominator: float,
) -> dict[str, Any]:
    actual_delta = float(actual_summary["actual_delta"])
    predicted_delta = float(dot_summary["dot"])
    residual = actual_delta - predicted_delta
    relative_error_denominator = max(abs(actual_delta), min_error_denominator)
    predicted_relative_error_denominator = max(abs(predicted_delta), min_error_denominator)
    return {
        "source_step": source_step,
        "target_step": target_step,
        "step_gap": target_step - source_step,
        "source_checkpoint": str(source_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "learning_rate": learning_rate,
        "split": split,
        "pair_type": pair_type,
        "head_layer": head_layer,
        "head": head,
        "head_label": _head_label(head_layer, head),
        "score_query_role": score_query_role,
        "support_key_role": support_key_role,
        "distractor_key_role": distractor_key_role,
        "record_side": record_side,
        "scalar_name": scalar_name,
        "num_pairs": int(source_payload["num_pairs"]),
        "num_entries": int(actual_summary["num_entries"]),
        "num_unique_pairs": int(actual_summary["num_unique_pairs"]),
        "source_value": float(actual_summary["source_value"]),
        "source_value_std": float(actual_summary["source_value_std"]),
        "target_value": float(actual_summary["target_value"]),
        "target_value_std": float(actual_summary["target_value_std"]),
        "source_objective_value": float(source_payload["scalar_value"]),
        "source_objective_value_abs_mean": float(source_payload["scalar_value_abs_mean"]),
        "source_objective_value_std": float(source_payload["scalar_value_std"]),
        "actual_delta": actual_delta,
        "actual_delta_abs_mean": float(actual_summary["actual_delta_abs_mean"]),
        "actual_delta_std": float(actual_summary["actual_delta_std"]),
        "predicted_delta": predicted_delta,
        "residual": residual,
        "absolute_error": abs(residual),
        "relative_error": abs(residual) / relative_error_denominator,
        "relative_error_denominator": relative_error_denominator,
        "predicted_relative_error": abs(residual) / predicted_relative_error_denominator,
        "predicted_relative_error_denominator": predicted_relative_error_denominator,
        "sign_match": _sign_match(actual=actual_delta, predicted=predicted_delta),
        "source_accuracy": float(actual_summary["source_accuracy"]),
        "target_accuracy": float(actual_summary["target_accuracy"]),
        "parameter_delta_l2_norm": float(dot_summary["left_l2_norm"]),
        "scalar_gradient_l2_norm": float(dot_summary["right_l2_norm"]),
        "update_scalar_gradient_cosine": dot_summary["cosine"],
        "num_parameters": int(dot_summary["num_parameters"]),
        "zero_scalar_gradient_parameter_count": len(source_payload["zero_gradient_parameter_names"]),
        "zero_scalar_gradient_parameter_names": source_payload["zero_gradient_parameter_names"],
    }


def _attention_downstream_update_decomposition_row(
    *,
    metric_row: dict[str, Any],
    group: _RouteGradientDecompositionGroup,
    dot_summary: dict[str, float | int | None],
) -> dict[str, Any]:
    predicted_delta = float(dot_summary["dot"])
    scalar_gradient_l2_norm = float(dot_summary["right_l2_norm"])
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
        "head_label": metric_row["head_label"],
        "head_layer": int(metric_row["head_layer"]),
        "head": int(metric_row["head"]),
        "score_query_role": metric_row["score_query_role"],
        "support_key_role": metric_row["support_key_role"],
        "distractor_key_role": metric_row["distractor_key_role"],
        "record_side": metric_row["record_side"],
        "scalar_name": metric_row["scalar_name"],
        "num_pairs": int(metric_row["num_pairs"]),
        "num_entries": int(metric_row["num_entries"]),
        "source_value": float(metric_row["source_value"]),
        "target_value": float(metric_row["target_value"]),
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
        "scalar_gradient_l2_norm": scalar_gradient_l2_norm,
        "update_scalar_gradient_cosine": dot_summary["cosine"],
        "contribution_per_parameter": predicted_delta / num_selected_parameters,
        "notes": list(group.notes),
    }


def _compute_attention_downstream_update_attribution_interval(
    *,
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    vocab: Vocabulary,
    learning_rate: float,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    record_sides: list[str],
    scalar_names: list[str],
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    groups: list[_RouteGradientDecompositionGroup],
    min_error_denominator: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not groups:
        raise ValueError("Attention downstream update attribution requires at least one decomposition group.")
    actual_rows = _compute_attention_downstream_actual_rows(
        source_model=source_model,
        target_model=target_model,
        source_checkpoint_path=source_checkpoint_path,
        target_checkpoint_path=target_checkpoint_path,
        pairs=pairs,
        vocab=vocab,
        head_layer=head_layer,
        head=head,
        score_query_role=score_query_role,
        support_key_role=support_key_role,
        distractor_key_role=distractor_key_role,
        record_sides=record_sides,
        scalar_names=scalar_names,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        device=device,
    )
    source_step = _checkpoint_step_from_path(source_checkpoint_path)
    target_step = _checkpoint_step_from_path(target_checkpoint_path)
    source_parameters = _model_parameter_snapshot(source_model)
    target_parameters = _model_parameter_snapshot(target_model)
    delta_parameters = _parameter_delta(
        source_parameters=source_parameters,
        target_parameters=target_parameters,
        label=f"attention downstream update {source_step}->{target_step}",
    )

    pair_groups = _route_gradient_groups(pairs)
    metric_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    for (split, pair_type), group_pairs in sorted(pair_groups.items()):
        for record_side in record_sides:
            gradient_payloads = _compute_attention_downstream_scalar_gradients_for_pairs(
                model=source_model,
                pairs=group_pairs,
                vocab=vocab,
                head_layer=head_layer,
                head=head,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
                record_side=record_side,
                scalar_names=scalar_names,
                batch_size=batch_size,
                pad_token_id=pad_token_id,
                device=device,
            )
            for scalar_name in scalar_names:
                actual_summary = _attention_downstream_actual_summary(
                    actual_rows=actual_rows,
                    split=split,
                    pair_type=pair_type,
                    record_side=record_side,
                    scalar_name=scalar_name,
                )
                source_payload = gradient_payloads[scalar_name]
                scalar_gradients = source_payload["gradients"]
                if not isinstance(scalar_gradients, dict):
                    raise TypeError("Attention downstream gradient payload must contain a gradients dictionary.")
                dot_summary = _gradient_dot_summary(
                    left_gradients=delta_parameters,
                    right_gradients=scalar_gradients,
                    label=(
                        f"attention downstream update {source_step}->{target_step} "
                        f"{split}/{pair_type}/{record_side}/{scalar_name}"
                    ),
                )
                metric_row = _attention_downstream_update_metric_row(
                    source_step=source_step,
                    target_step=target_step,
                    source_checkpoint=source_checkpoint_path,
                    target_checkpoint=target_checkpoint_path,
                    learning_rate=learning_rate,
                    split=split,
                    pair_type=pair_type,
                    head_layer=head_layer,
                    head=head,
                    score_query_role=score_query_role,
                    support_key_role=support_key_role,
                    distractor_key_role=distractor_key_role,
                    record_side=record_side,
                    scalar_name=scalar_name,
                    actual_summary=actual_summary,
                    source_payload=source_payload,
                    dot_summary=dot_summary,
                    min_error_denominator=min_error_denominator,
                )
                metric_rows.append(metric_row)
                for group in groups:
                    group_dot_summary = _gradient_dot_summary_for_group(
                        left_gradients=delta_parameters,
                        right_gradients=scalar_gradients,
                        group=group,
                        label=(
                            f"attention downstream update {source_step}->{target_step} "
                            f"{split}/{pair_type}/{record_side}/{scalar_name}/{group.group_id}"
                        ),
                    )
                    decomposition_rows.append(
                        _attention_downstream_update_decomposition_row(
                            metric_row=metric_row,
                            group=group,
                            dot_summary=group_dot_summary,
                        )
                    )
    return metric_rows, decomposition_rows, actual_rows


def _summarize_attention_downstream_update_attribution(
    *,
    metric_rows: list[dict[str, Any]],
    decomposition_rows: list[dict[str, Any]],
    top_k_groups: int,
) -> dict[str, Any]:
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")
    if not metric_rows:
        raise ValueError("Cannot summarize attention downstream update attribution without metric rows.")
    if not decomposition_rows:
        raise ValueError("Cannot summarize attention downstream update attribution without decomposition rows.")
    all_rows = [
        row
        for row in metric_rows
        if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
    ]
    if not all_rows:
        raise RuntimeError("Attention downstream update attribution has no __all__/__all__ metric rows.")
    final_target_step = max(int(row["target_step"]) for row in metric_rows)
    final_rows = [row for row in all_rows if int(row["target_step"]) == final_target_step]
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
        "final_metric_rows": sorted(
            final_rows,
            key=lambda row: (str(row["record_side"]), str(row["scalar_name"])),
        ),
        "all_all_sign_match_fraction": _fraction(
            sum(1 for row in all_rows if bool(row["sign_match"])),
            len(all_rows),
            "attention downstream update sign_match_fraction",
        ),
        "all_all_mean_absolute_error": _mean([float(row["absolute_error"]) for row in all_rows]),
        "all_all_mean_relative_error": _mean([float(row["relative_error"]) for row in all_rows]),
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


def _plot_attention_downstream_update_actual_vs_predicted(
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
            str(row["scalar_name"]),
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
    ax.set_title("Attention downstream update attribution: actual vs predicted")
    ax.set_xlabel("grad(scalar) . Delta theta")
    ax.set_ylabel("scalar(theta_target) - scalar(theta_source)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_attention_downstream_update_top_contributions(
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
    labels = [f"{row['scalar_name']} {row['record_side']} {row['group_id']}" for row in top_rows]
    colors = ["#376f8f" if value >= 0.0 else "#8f374a" for value in values]
    ax.barh(y_positions, values, color=colors)
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f"Top downstream scalar update contributions ending at step {final_target_step}")
    ax.set_xlabel("grad(scalar) . Delta theta contribution")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_attention_downstream_update_attribution_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    lines = [
        "# Attention Downstream Update Attribution",
        "",
        "## Calculation",
        "",
        "This report tests whether actual checkpoint-to-checkpoint parameter movement explains downstream answer-write scalars.",
        "",
        "```text",
        "actual_delta = scalar(theta_target) - scalar(theta_source)",
        "predicted_delta = grad_theta scalar(theta_source) . (theta_target - theta_source)",
        "residual = actual_delta - predicted_delta",
        "```",
        "",
        "`head_margin_dla_fixed_readout` uses the source checkpoint answer-margin readout vector for both source and target.",
        "That makes it a fixed-readout first-order scalar, not a moving-readout DLA claim.",
        "",
        "## Run",
        "",
        f"- head: `{report['head_label']}`",
        f"- query role: `{report['score_query_role']}`",
        f"- support role: `{report['support_key_role']}`",
        f"- distractor role: `{report['distractor_key_role']}`",
        f"- record sides: `{report['record_sides']}`",
        f"- scalars: `{report['scalar_names']}`",
        f"- intervals: `{summary['intervals']}`",
        "",
        "## Final Metrics",
        "",
        "| record side | scalar | actual delta | predicted delta | residual | relative error | sign match |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["final_metric_rows"]:
        lines.append(
            "| `{side}` | `{scalar}` | {actual:.6g} | {predicted:.6g} | {residual:.6g} | {error:.6g} | `{sign}` |".format(
                side=row["record_side"],
                scalar=row["scalar_name"],
                actual=float(row["actual_delta"]),
                predicted=float(row["predicted_delta"]),
                residual=float(row["residual"]),
                error=float(row["relative_error"]),
                sign=bool(row["sign_match"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Positive Contributions",
            "",
            "| group | scalar | kind | contribution | cosine |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in summary["final_top_positive_contributions"]:
        cosine = row["update_scalar_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| `{group}` | `{scalar}` | `{kind}` | {contribution:.6g} | {cosine} |".format(
                group=row["group_id"],
                scalar=row["scalar_name"],
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
            "| group | scalar | kind | contribution | cosine |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in summary["final_top_negative_contributions"]:
        cosine = row["update_scalar_gradient_cosine"]
        cosine_text = "" if cosine is None else f"{float(cosine):.6f}"
        lines.append(
            "| `{group}` | `{scalar}` | `{kind}` | {contribution:.6g} | {cosine} |".format(
                group=row["group_id"],
                scalar=row["scalar_name"],
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
            f"- scalar rows: `{report['scalar_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
        ]
    )
    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for label, plot_path in plot_paths.items():
            lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_attention_downstream_update_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    head_layer: int,
    head: int,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    pair_types: list[str],
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    record_sides: list[str] | None = None,
    scalar_names: list[str] | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    decomposition_modes: list[str] | None = None,
    top_k_groups: int = 24,
    min_error_denominator: float = 1.0e-9,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    unsupported_roles = [
        role
        for role in [score_query_role, support_key_role, distractor_key_role]
        if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported attention roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if support_key_role == distractor_key_role:
        raise ValueError("support_key_role and distractor_key_role must be different.")
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")

    resolved_record_sides = _resolve_attention_score_record_sides(record_sides)
    resolved_scalar_names = _resolve_attention_downstream_update_scalars(scalar_names)
    resolved_decomposition_modes = _resolve_route_gradient_decomposition_modes(decomposition_modes)
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("attention-downstream-update-attribution requires at least two checkpoints.")
    source_model = build_model(spec.model, len(vocab.tokens), device)
    target_model = build_model(spec.model, len(vocab.tokens), device)
    if head_layer < 0 or head_layer >= len(source_model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(source_model.blocks) - 1}.")
    if head < 0 or head >= source_model.blocks[head_layer].attn.n_heads:
        raise ValueError(
            f"head {head} outside model range 0..{source_model.blocks[head_layer].attn.n_heads - 1} for layer {head_layer}."
        )
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Attention downstream update attribution constructed no pairs.")

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

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows_path = output_dir / "attention_downstream_update_attribution_rows.jsonl"
    decomposition_rows_path = output_dir / "attention_downstream_update_attribution_decomposition_rows.jsonl"
    group_rows_path = output_dir / "attention_downstream_update_attribution_groups.jsonl"
    scalar_rows_path = output_dir / "attention_downstream_update_attribution_scalar_rows.jsonl"
    pair_rows_path = output_dir / "attention_downstream_update_attribution_pairs.jsonl"
    progress_path = output_dir / "attention_downstream_update_attribution_progress.json"
    for partial_path in (
        metric_rows_path,
        decomposition_rows_path,
        group_rows_path,
        scalar_rows_path,
        pair_rows_path,
        progress_path,
    ):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])
    write_jsonl(group_rows_path, group_rows)

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[attention-downstream-update-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(pairs)} "
        f"pair_types={pair_types} device={device_name} head={_head_label(head_layer, head)} "
        f"query_role={score_query_role} support={support_key_role} distractor={distractor_key_role} "
        f"record_sides={resolved_record_sides} scalars={resolved_scalar_names} groups={len(groups)}",
        flush=True,
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_decomposition_rows: list[dict[str, Any]] = []
    all_scalar_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        learning_rate = _compute_learning_rate(spec.optimization, source_step)
        print(
            "[attention-downstream-update-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )
        metric_rows, decomposition_rows, scalar_rows = _compute_attention_downstream_update_attribution_interval(
            source_model=source_model,
            target_model=target_model,
            source_checkpoint_path=source_checkpoint_path,
            target_checkpoint_path=target_checkpoint_path,
            pairs=pairs,
            vocab=vocab,
            learning_rate=learning_rate,
            head_layer=head_layer,
            head=head,
            score_query_role=score_query_role,
            support_key_role=support_key_role,
            distractor_key_role=distractor_key_role,
            record_sides=resolved_record_sides,
            scalar_names=resolved_scalar_names,
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
        for row in scalar_rows:
            append_jsonl(scalar_rows_path, row)
        all_metric_rows.extend(metric_rows)
        all_decomposition_rows.extend(decomposition_rows)
        all_scalar_rows.extend(scalar_rows)
        all_row = next(
            row
            for row in metric_rows
            if str(row["split"]) == "__all__"
            and str(row["pair_type"]) == "__all__"
            and str(row["record_side"]) == resolved_record_sides[0]
            and str(row["scalar_name"]) == resolved_scalar_names[0]
        )
        print(
            "[attention-downstream-update-attribution] finished "
            f"{source_step}->{target_step} scalar={all_row['scalar_name']} "
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
                "scalar_rows_path": str(scalar_rows_path),
            },
        )

    summary = _summarize_attention_downstream_update_attribution(
        metric_rows=all_metric_rows,
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
    )
    plot_paths: dict[str, Path] = {}
    actual_vs_predicted_plot = _plot_attention_downstream_update_actual_vs_predicted(
        metric_rows=all_metric_rows,
        output_path=output_dir / "attention_downstream_update_actual_vs_predicted.svg",
    )
    if actual_vs_predicted_plot is not None:
        plot_paths["actual_vs_predicted"] = actual_vs_predicted_plot
    top_contributions_plot = _plot_attention_downstream_update_top_contributions(
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
        output_path=output_dir / "attention_downstream_update_top_contributions.svg",
    )
    if top_contributions_plot is not None:
        plot_paths["top_contributions"] = top_contributions_plot

    report_path = output_dir / "attention_downstream_update_attribution_report.json"
    markdown_path = output_dir / "attention_downstream_update_attribution_report.md"
    report = {
        "schema_version": ATTENTION_DOWNSTREAM_UPDATE_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "head_layer": head_layer,
        "head": head,
        "head_label": _head_label(head_layer, head),
        "score_query_role": score_query_role,
        "support_key_role": support_key_role,
        "distractor_key_role": distractor_key_role,
        "record_sides": resolved_record_sides,
        "scalar_names": resolved_scalar_names,
        "pair_types": pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "decomposition": decomposition_summary,
        "top_k_groups": top_k_groups,
        "min_error_denominator": min_error_denominator,
        "calculation": {
            "actual_delta": "scalar(theta_target) - scalar(theta_source)",
            "predicted_delta": "grad_theta scalar(theta_source) . (theta_target - theta_source)",
            "head_margin_dla_fixed_readout": (
                "head output at the query role dotted with the source checkpoint answer-margin readout vector"
            ),
            "negative_answer_loss": "-cross_entropy(answer_logits, answer_target)",
            "group_contribution": "grad_group scalar(theta_source) . Delta theta_group",
        },
        "pair_construction": pair_construction,
        "metric_rows_path": str(metric_rows_path),
        "decomposition_rows_path": str(decomposition_rows_path),
        "group_rows_path": str(group_rows_path),
        "scalar_rows_path": str(scalar_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_attention_downstream_update_attribution_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
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
            "scalar_rows_path": str(scalar_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[attention-downstream-update-attribution] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return (
        report_path,
        markdown_path,
        metric_rows_path,
        decomposition_rows_path,
        group_rows_path,
        scalar_rows_path,
        pair_rows_path,
        plot_paths,
    )


def run_checkpoint_update_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    stage_name: str,
    subspace_name: str,
    position_role: str,
    pair_types: list[str],
    rank: int | None = None,
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    head_layer: int | None = None,
    head: int | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    decomposition_modes: list[str] | None = None,
    top_k_groups: int = 24,
    min_error_denominator: float = 1.0e-9,
) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if top_k_groups <= 0:
        raise ValueError("top_k_groups must be positive.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    resolved_decomposition_modes = _resolve_route_gradient_decomposition_modes(decomposition_modes)
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("checkpoint-update-attribution requires at least two checkpoints.")
    model = build_model(spec.model, len(vocab.tokens), device)
    _validate_geometry_stage(model=model, stage_name=stage_name)
    if position_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported position role {position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Checkpoint update attribution constructed no pairs.")

    groups, decomposition_summary = _build_route_gradient_decomposition_groups(
        model=model,
        decomposition_modes=resolved_decomposition_modes,
    )
    group_rows = [
        _group_metadata(
            model_parameters=dict(model.named_parameters(remove_duplicate=False)),
            group=group,
        )
        for group in groups
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows_path = output_dir / "checkpoint_update_attribution_rows.jsonl"
    decomposition_rows_path = output_dir / "checkpoint_update_attribution_decomposition_rows.jsonl"
    group_rows_path = output_dir / "checkpoint_update_attribution_groups.jsonl"
    pair_rows_path = output_dir / "checkpoint_update_attribution_pairs.jsonl"
    progress_path = output_dir / "checkpoint_update_attribution_progress.json"
    for partial_path in (metric_rows_path, decomposition_rows_path, group_rows_path, pair_rows_path, progress_path):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])
    write_jsonl(group_rows_path, group_rows)

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[checkpoint-update-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(pairs)} pair_types={pair_types} "
        f"device={device_name} subspace={subspace_name} rank={rank} stage={stage_name} "
        f"role={position_role} groups={len(groups)} basis_mode=source_checkpoint",
        flush=True,
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_decomposition_rows: list[dict[str, Any]] = []
    final_subspace_summary: dict[str, Any] | None = None
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        learning_rate = _compute_learning_rate(spec.optimization, source_step)
        print(
            "[checkpoint-update-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )
        metric_rows, decomposition_rows, subspace_summary = _compute_checkpoint_update_attribution_interval(
            model=model,
            source_checkpoint_path=source_checkpoint_path,
            target_checkpoint_path=target_checkpoint_path,
            pairs=pairs,
            vocab=vocab,
            learning_rate=learning_rate,
            subspace_name=subspace_name,
            rank=rank,
            head_layer=head_layer,
            head=head,
            stage_name=stage_name,
            position_role=position_role,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            groups=groups,
            min_error_denominator=min_error_denominator,
        )
        final_subspace_summary = subspace_summary
        for row in metric_rows:
            append_jsonl(metric_rows_path, row)
        for row in decomposition_rows:
            append_jsonl(decomposition_rows_path, row)
        all_metric_rows.extend(metric_rows)
        all_decomposition_rows.extend(decomposition_rows)
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
                "group_rows_path": str(group_rows_path),
                "pair_rows_path": str(pair_rows_path),
            },
        )
        all_row = next(
            row
            for row in metric_rows
            if str(row["split"]) == "__all__" and str(row["pair_type"]) == "__all__"
        )
        print(
            "[checkpoint-update-attribution] finished "
            f"{source_step}->{target_step} actual_delta={float(all_row['actual_delta']):.6g} "
            f"predicted_delta={float(all_row['predicted_delta']):.6g} "
            f"relative_error={float(all_row['relative_error']):.6g} "
            f"sign_match={all_row['sign_match']}",
            flush=True,
        )

    if final_subspace_summary is None:
        raise RuntimeError("No checkpoint intervals were processed for checkpoint update attribution.")
    summary = _summarize_checkpoint_update_attribution(
        metric_rows=all_metric_rows,
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
    )
    report_path = output_dir / "checkpoint_update_attribution_report.json"
    markdown_path = output_dir / "checkpoint_update_attribution_report.md"
    plot_paths: dict[str, Path] = {}
    actual_vs_predicted_plot = _plot_checkpoint_update_actual_vs_predicted(
        metric_rows=all_metric_rows,
        output_path=output_dir / "checkpoint_update_actual_vs_predicted.svg",
    )
    if actual_vs_predicted_plot is not None:
        plot_paths["actual_vs_predicted"] = actual_vs_predicted_plot
    relative_error_plot = _plot_checkpoint_update_relative_error(
        metric_rows=all_metric_rows,
        output_path=output_dir / "checkpoint_update_relative_error.svg",
    )
    if relative_error_plot is not None:
        plot_paths["relative_error"] = relative_error_plot
    top_contributions_plot = _plot_checkpoint_update_top_contributions(
        decomposition_rows=all_decomposition_rows,
        top_k_groups=top_k_groups,
        output_path=output_dir / "checkpoint_update_top_contributions.svg",
    )
    if top_contributions_plot is not None:
        plot_paths["top_contributions"] = top_contributions_plot

    report = {
        "schema_version": CHECKPOINT_UPDATE_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "stage": stage_name,
        "subspace": final_subspace_summary,
        "subspace_name": subspace_name,
        "rank": rank,
        "position_role": position_role,
        "pair_types": pair_types,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "basis_mode": "source_checkpoint_per_interval",
        "decomposition": decomposition_summary,
        "top_k_groups": top_k_groups,
        "min_error_denominator": min_error_denominator,
        "calculation": {
            "route_score": "patched_transfer_margin - corrupted_transfer_margin",
            "actual_delta": "route_score(theta_target; source_basis) - route_score(theta_source; source_basis)",
            "predicted_delta": "grad_theta route_score(theta_source; source_basis) . (theta_target - theta_source)",
            "residual": "actual_delta - predicted_delta",
            "relative_error": "abs(residual) / max(abs(actual_delta), min_error_denominator)",
            "basis_warning": "Head/embedding subspace basis is fixed from the source checkpoint for each interval.",
            "group_contribution": "grad_group route_score(theta_source; source_basis) . Delta theta_group",
        },
        "pair_construction": pair_construction,
        "metric_rows_path": str(metric_rows_path),
        "decomposition_rows_path": str(decomposition_rows_path),
        "group_rows_path": str(group_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_checkpoint_update_attribution_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_intervals": len(intervals),
            "total_intervals": len(intervals),
            "last_source_step": int(summary["final_all"]["source_step"]),
            "last_target_step": int(summary["final_all"]["target_step"]),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "metric_rows_path": str(metric_rows_path),
            "decomposition_rows_path": str(decomposition_rows_path),
            "group_rows_path": str(group_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[checkpoint-update-attribution] complete report={report_path} rows={metric_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, metric_rows_path, decomposition_rows_path, group_rows_path, pair_rows_path, plot_paths


def _data_update_group_value(*, pair: dict[str, Any], field_name: str) -> str:
    if not field_name:
        raise ValueError("Data update group field names must be non-empty.")
    current: Any = _pair_metadata(pair)
    traversed: list[str] = []
    for part in field_name.split("."):
        traversed.append(part)
        if not isinstance(current, dict):
            path = ".".join(traversed[:-1])
            raise ValueError(
                f"Cannot read data group field {field_name!r}: {path!r} is not an object."
            )
        if part not in current:
            available = sorted(str(key) for key in current)
            raise KeyError(
                f"Data group field {field_name!r} is missing at {'.'.join(traversed)!r}; "
                f"available keys: {available}"
            )
        current = current[part]
    if isinstance(current, (dict, list, tuple)):
        raise ValueError(
            f"Data group field {field_name!r} resolved to a non-scalar value of type {type(current).__name__}."
        )
    if current is None:
        raise ValueError(f"Data group field {field_name!r} resolved to None.")
    return str(current)


def _data_update_group_id(*, fields: list[str], values: tuple[str, ...]) -> str:
    if len(fields) != len(values):
        raise RuntimeError("Data update group fields and values have different lengths.")
    return "|".join(f"{field}={value}" for field, value in zip(fields, values, strict=True))


def _group_pairs_for_data_update(
    *,
    pairs: list[dict[str, Any]],
    data_group_fields: list[str],
) -> list[dict[str, Any]]:
    if not pairs:
        raise ValueError("Cannot build data update groups from no pairs.")
    if not data_group_fields:
        raise ValueError("At least one data group field is required.")
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        values = tuple(
            _data_update_group_value(pair=pair, field_name=field_name)
            for field_name in data_group_fields
        )
        groups[values].append(pair)
    rows = [
        {
            "data_group_id": "__all__",
            "data_group_values": {},
            "pairs": list(pairs),
        }
    ]
    for values, group_pairs in sorted(groups.items(), key=lambda item: item[0]):
        rows.append(
            {
                "data_group_id": _data_update_group_id(fields=data_group_fields, values=values),
                "data_group_values": dict(zip(data_group_fields, values, strict=True)),
                "pairs": group_pairs,
            }
        )
    return rows


def _route_objective_pairs(
    *,
    pairs: list[dict[str, Any]],
    route_split: str,
    route_pair_type: str,
) -> list[dict[str, Any]]:
    if not pairs:
        raise ValueError("Cannot select a route objective from no pairs.")
    available_splits = sorted({str(pair["split"]) for pair in pairs})
    available_pair_types = sorted({str(pair["pair_type"]) for pair in pairs})
    if route_split != "__all__" and route_split not in available_splits:
        raise ValueError(f"route_split {route_split!r} is not available; expected one of {available_splits} or '__all__'.")
    if route_pair_type != "__all__" and route_pair_type not in available_pair_types:
        raise ValueError(
            f"route_pair_type {route_pair_type!r} is not available; expected one of {available_pair_types} or '__all__'."
        )
    selected = [
        pair
        for pair in pairs
        if (route_split == "__all__" or str(pair["split"]) == route_split)
        and (route_pair_type == "__all__" or str(pair["pair_type"]) == route_pair_type)
    ]
    if not selected:
        raise RuntimeError(
            f"Route objective selected no pairs for route_split={route_split!r} route_pair_type={route_pair_type!r}."
        )
    return selected


def _data_update_route_metric_row(
    *,
    source_step: int,
    target_step: int,
    source_checkpoint: Path,
    target_checkpoint: Path,
    learning_rate: float,
    route_split: str,
    route_pair_type: str,
    stage_name: str,
    subspace_name: str,
    subspace_summary: dict[str, Any],
    rank: int | None,
    position_role: str,
    source_payload: dict[str, Any],
    target_payload: dict[str, Any],
    dot_summary: dict[str, float | int | None],
    min_error_denominator: float,
) -> dict[str, Any]:
    return _checkpoint_update_metric_row(
        source_step=source_step,
        target_step=target_step,
        source_checkpoint=source_checkpoint,
        target_checkpoint=target_checkpoint,
        learning_rate=learning_rate,
        split=route_split,
        pair_type=route_pair_type,
        stage_name=stage_name,
        subspace_name=subspace_name,
        subspace_summary=subspace_summary,
        rank=rank,
        position_role=position_role,
        source_payload=source_payload,
        target_payload=target_payload,
        dot_summary=dot_summary,
        min_error_denominator=min_error_denominator,
    )


def _data_update_group_row(
    *,
    route_metric_row: dict[str, Any],
    data_group_id: str,
    data_group_values: dict[str, str],
    loss_side: str,
    loss_payload: dict[str, Any],
    loss_route_dot_summary: dict[str, float | int | None],
    loss_update_dot_summary: dict[str, float | int | None],
) -> dict[str, Any]:
    loss_dot_route = float(loss_route_dot_summary["dot"])
    negative_loss_dot_route = -loss_dot_route
    loss_dot_update = float(loss_update_dot_summary["dot"])
    negative_loss_dot_update = -loss_dot_update
    loss_gradient_l2_norm = float(loss_route_dot_summary["left_l2_norm"])
    route_gradient_l2_norm = float(loss_route_dot_summary["right_l2_norm"])
    parameter_delta_l2_norm = float(loss_update_dot_summary["right_l2_norm"])
    return {
        "source_step": int(route_metric_row["source_step"]),
        "target_step": int(route_metric_row["target_step"]),
        "step_gap": int(route_metric_row["step_gap"]),
        "source_checkpoint": route_metric_row["source_checkpoint"],
        "target_checkpoint": route_metric_row["target_checkpoint"],
        "learning_rate": float(route_metric_row["learning_rate"]),
        "route_split": route_metric_row["split"],
        "route_pair_type": route_metric_row["pair_type"],
        "stage": route_metric_row["stage"],
        "subspace_name": route_metric_row["subspace_name"],
        "subspace_type": route_metric_row["subspace_type"],
        "head_label": route_metric_row["head_label"],
        "rank": route_metric_row["rank"],
        "position_role": route_metric_row["position_role"],
        "data_group_id": data_group_id,
        "data_group_values": data_group_values,
        "loss_side": loss_side,
        "loss": float(loss_payload["loss"]),
        "loss_num_records": int(loss_payload["num_records"]),
        "loss_num_tokens": int(loss_payload["num_tokens"]),
        "source_route_score": float(route_metric_row["source_route_score"]),
        "target_route_score": float(route_metric_row["target_route_score"]),
        "actual_route_delta": float(route_metric_row["actual_delta"]),
        "actual_update_predicted_route_delta": float(route_metric_row["predicted_delta"]),
        "actual_update_route_residual": float(route_metric_row["residual"]),
        "actual_update_route_relative_error": float(route_metric_row["relative_error"]),
        "actual_update_route_sign_match": bool(route_metric_row["sign_match"]),
        "loss_gradient_l2_norm": loss_gradient_l2_norm,
        "route_gradient_l2_norm": route_gradient_l2_norm,
        "parameter_delta_l2_norm": parameter_delta_l2_norm,
        "loss_dot_route_gradient": loss_dot_route,
        "negative_loss_dot_route_gradient": negative_loss_dot_route,
        "loss_negative_route_gradient_cosine": _safe_ratio(
            negative_loss_dot_route,
            loss_gradient_l2_norm * route_gradient_l2_norm,
        ),
        "local_sgd_route_delta_linearized": float(route_metric_row["learning_rate"]) * negative_loss_dot_route,
        "loss_dot_actual_update": loss_dot_update,
        "negative_loss_dot_actual_update": negative_loss_dot_update,
        "loss_delta_under_actual_update_linearized": loss_dot_update,
        "loss_reduction_under_actual_update_linearized": negative_loss_dot_update,
        "loss_negative_actual_update_cosine": _safe_ratio(
            negative_loss_dot_update,
            loss_gradient_l2_norm * parameter_delta_l2_norm,
        ),
        "loss_route_gradient_cosine": loss_route_dot_summary["cosine"],
        "loss_actual_update_cosine": loss_update_dot_summary["cosine"],
    }


def _compute_data_update_attribution_interval(
    *,
    model: torch.nn.Module,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    pairs: list[dict[str, Any]],
    route_pairs: list[dict[str, Any]],
    data_groups: list[dict[str, Any]],
    vocab: Vocabulary,
    learning_rate: float,
    route_split: str,
    route_pair_type: str,
    subspace_name: str,
    rank: int | None,
    head_layer: int | None,
    head: int | None,
    stage_name: str,
    position_role: str,
    loss_side: str,
    batch_size: int,
    pad_token_id: int,
    device: torch.device,
    min_error_denominator: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if not pairs:
        raise ValueError("Data update attribution requires at least one pair.")
    if not route_pairs:
        raise ValueError("Data update attribution requires at least one route pair.")
    if not data_groups:
        raise ValueError("Data update attribution requires at least one data group.")

    source_checkpoint = load_checkpoint(source_checkpoint_path, device)
    load_model_state(model, source_checkpoint["model_state"])
    model.eval()
    source_step = int(source_checkpoint["step"])
    source_path_step = _checkpoint_step_from_path(source_checkpoint_path)
    if source_step != source_path_step:
        raise RuntimeError(
            f"Source checkpoint step mismatch for {source_checkpoint_path}: payload={source_step} path={source_path_step}"
        )
    source_parameters = _model_parameter_snapshot(model)
    basis, subspace_summary = _resolve_causal_patch_basis(
        model=model,
        vocab=vocab,
        subspace_name=subspace_name,
        rank=rank,
        head_layer=head_layer,
        head=head,
        device=device,
    )
    source_route_payload = _compute_route_score_gradient_for_pairs(
        model=model,
        pairs=route_pairs,
        vocab=vocab,
        basis=basis,
        stage_name=stage_name,
        position_role=position_role,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        device=device,
    )
    route_gradients = source_route_payload["gradients"]
    if not isinstance(route_gradients, dict):
        raise TypeError("Route gradient payload must contain a gradients dictionary.")

    loss_payloads: dict[str, dict[str, Any]] = {}
    for group in data_groups:
        loss_records = _loss_records_for_pairs(pairs=group["pairs"], loss_side=loss_side)
        loss_payloads[str(group["data_group_id"])] = _compute_loss_gradient_for_records(
            model=model,
            records=loss_records,
            batch_size=batch_size,
            pad_token_id=pad_token_id,
            device=device,
        )

    target_checkpoint = load_checkpoint(target_checkpoint_path, device)
    load_model_state(model, target_checkpoint["model_state"])
    model.eval()
    target_step = int(target_checkpoint["step"])
    target_path_step = _checkpoint_step_from_path(target_checkpoint_path)
    if target_step != target_path_step:
        raise RuntimeError(
            f"Target checkpoint step mismatch for {target_checkpoint_path}: payload={target_step} path={target_path_step}"
        )
    if target_step <= source_step:
        raise ValueError(f"Data update attribution requires increasing steps, got source={source_step} target={target_step}.")
    target_parameters = _model_parameter_snapshot(model)
    delta_parameters = _parameter_delta(
        source_parameters=source_parameters,
        target_parameters=target_parameters,
        label=f"{source_step}->{target_step}",
    )
    target_route_payload = _compute_route_score_for_pairs(
        model=model,
        pairs=route_pairs,
        vocab=vocab,
        basis=basis,
        stage_name=stage_name,
        position_role=position_role,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        device=device,
    )
    update_route_dot_summary = _gradient_dot_summary(
        left_gradients=delta_parameters,
        right_gradients=route_gradients,
        label=f"data update route objective {source_step}->{target_step} {route_split}/{route_pair_type}",
    )
    route_metric_row = _data_update_route_metric_row(
        source_step=source_step,
        target_step=target_step,
        source_checkpoint=source_checkpoint_path,
        target_checkpoint=target_checkpoint_path,
        learning_rate=learning_rate,
        route_split=route_split,
        route_pair_type=route_pair_type,
        stage_name=stage_name,
        subspace_name=subspace_name,
        subspace_summary=subspace_summary,
        rank=rank,
        position_role=position_role,
        source_payload=source_route_payload,
        target_payload=target_route_payload,
        dot_summary=update_route_dot_summary,
        min_error_denominator=min_error_denominator,
    )

    data_rows: list[dict[str, Any]] = []
    for group in data_groups:
        group_id = str(group["data_group_id"])
        loss_payload = loss_payloads[group_id]
        loss_gradients = loss_payload["gradients"]
        if not isinstance(loss_gradients, dict):
            raise TypeError("Loss gradient payload must contain a gradients dictionary.")
        loss_route_dot_summary = _gradient_dot_summary(
            left_gradients=loss_gradients,
            right_gradients=route_gradients,
            label=f"data update loss-route {source_step}->{target_step} {group_id}",
        )
        loss_update_dot_summary = _gradient_dot_summary(
            left_gradients=loss_gradients,
            right_gradients=delta_parameters,
            label=f"data update loss-actual-update {source_step}->{target_step} {group_id}",
        )
        data_rows.append(
            _data_update_group_row(
                route_metric_row=route_metric_row,
                data_group_id=group_id,
                data_group_values=group["data_group_values"],
                loss_side=loss_side,
                loss_payload=loss_payload,
                loss_route_dot_summary=loss_route_dot_summary,
                loss_update_dot_summary=loss_update_dot_summary,
            )
        )

    return route_metric_row, data_rows, subspace_summary


def _summarize_data_update_attribution(
    *,
    route_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
    top_k_data_groups: int,
) -> dict[str, Any]:
    if top_k_data_groups <= 0:
        raise ValueError("top_k_data_groups must be positive.")
    if not route_rows:
        raise ValueError("Cannot summarize data update attribution without route rows.")
    if not data_rows:
        raise ValueError("Cannot summarize data update attribution without data rows.")
    final_target_step = max(int(row["target_step"]) for row in route_rows)
    final_route_rows = [row for row in route_rows if int(row["target_step"]) == final_target_step]
    if len(final_route_rows) != 1:
        raise RuntimeError(f"Expected one final route row at target step {final_target_step}, got {len(final_route_rows)}.")
    final_data_rows = [
        row
        for row in data_rows
        if int(row["target_step"]) == final_target_step
    ]
    non_all_final = [row for row in final_data_rows if str(row["data_group_id"]) != "__all__"]
    return {
        "num_intervals": len({(int(row["source_step"]), int(row["target_step"])) for row in route_rows}),
        "intervals": sorted({f"{int(row['source_step'])}->{int(row['target_step'])}" for row in route_rows}),
        "final_target_step": final_target_step,
        "final_route": final_route_rows[0],
        "final_data_rows": sorted(final_data_rows, key=lambda row: str(row["data_group_id"])),
        "final_top_actual_update_loss_reduction": sorted(
            non_all_final,
            key=lambda row: float(row["loss_reduction_under_actual_update_linearized"]),
            reverse=True,
        )[:top_k_data_groups],
        "final_top_route_support": sorted(
            non_all_final,
            key=lambda row: float(row["negative_loss_dot_route_gradient"]),
            reverse=True,
        )[:top_k_data_groups],
        "final_top_route_conflict": sorted(
            non_all_final,
            key=lambda row: float(row["negative_loss_dot_route_gradient"]),
        )[:top_k_data_groups],
    }


def _plot_data_update_group_bars(
    *,
    data_rows: list[dict[str, Any]],
    value_field: str,
    title: str,
    xlabel: str,
    top_k_data_groups: int,
    output_path: Path,
) -> Path | None:
    if not data_rows:
        return None
    final_target_step = max(int(row["target_step"]) for row in data_rows)
    rows = [
        row
        for row in data_rows
        if int(row["target_step"]) == final_target_step and str(row["data_group_id"]) != "__all__"
    ]
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: abs(float(row[value_field])), reverse=True)[:top_k_data_groups]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(rows))))
    y_positions = list(range(len(rows)))
    values = [float(row[value_field]) for row in rows]
    labels = [str(row["data_group_id"]) for row in rows]
    colors = ["#376f8f" if value >= 0.0 else "#8f374a" for value in values]
    ax.barh(y_positions, values, color=colors)
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_data_update_vs_route_support(
    *,
    data_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not data_rows:
        return None
    final_target_step = max(int(row["target_step"]) for row in data_rows)
    rows = [
        row
        for row in data_rows
        if int(row["target_step"]) == final_target_step and str(row["data_group_id"]) != "__all__"
    ]
    if not rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 6))
    x_values = [float(row["loss_reduction_under_actual_update_linearized"]) for row in rows]
    y_values = [float(row["negative_loss_dot_route_gradient"]) for row in rows]
    ax.scatter(x_values, y_values, color="#376f8f")
    for row, x_value, y_value in zip(rows, x_values, y_values, strict=True):
        ax.annotate(str(row["data_group_id"]), (x_value, y_value), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title(f"Data loss update alignment vs route support ending at step {final_target_step}")
    ax.set_xlabel("<-grad loss_group, actual Delta theta>")
    ax.set_ylabel("<-grad loss_group, grad route>")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_data_update_attribution_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    final_route = summary["final_route"]
    lines = [
        "# Data Update Attribution",
        "",
        "## Calculation",
        "",
        "This report links data-group loss gradients to both the actual checkpoint update and the candidate route gradient.",
        "",
        "```text",
        "actual_route_delta = route(theta_target; source_basis) - route(theta_source; source_basis)",
        "actual_update_predicted_route_delta = grad route(theta_source; source_basis) . Delta theta",
        "data_route_support_g = < -grad loss_g(theta_source), grad route(theta_source; source_basis) >",
        "data_actual_update_alignment_g = < -grad loss_g(theta_source), Delta theta >",
        "```",
        "",
        "The last quantity is not a replay of the original optimizer batches. It is a source-checkpoint diagnostic: it asks whether a data group's current loss gradient points in the same direction as the actual checkpoint movement.",
        "",
        "## Route Objective",
        "",
        f"- route split: `{report['route_split']}`",
        f"- route pair type: `{report['route_pair_type']}`",
        f"- subspace: `{report['subspace_name']}`",
        f"- basis mode: `{report['basis_mode']}`",
        f"- rank: `{report['rank']}`",
        f"- stage: `{report['stage']}`",
        f"- position role: `{report['position_role']}`",
        f"- loss side: `{report['loss_side']}`",
        "",
        "## Final Interval",
        "",
        f"- interval: `{final_route['source_step']} -> {final_route['target_step']}`",
        f"- source route score: `{float(final_route['source_route_score']):.6f}`",
        f"- target route score: `{float(final_route['target_route_score']):.6f}`",
        f"- actual route delta: `{float(final_route['actual_delta']):.6g}`",
        f"- actual-update predicted route delta: `{float(final_route['predicted_delta']):.6g}`",
        f"- relative error: `{float(final_route['relative_error']):.6g}`",
        "",
        "## Final Data Groups",
        "",
        "| data group | records | loss | actual update loss reduction | route support | local SGD route delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["final_data_rows"]:
        lines.append(
            "| {group} | {records} | {loss:.6f} | {update:.6g} | {support:.6g} | {delta:.6g} |".format(
                group=row["data_group_id"],
                records=int(row["loss_num_records"]),
                loss=float(row["loss"]),
                update=float(row["loss_reduction_under_actual_update_linearized"]),
                support=float(row["negative_loss_dot_route_gradient"]),
                delta=float(row["local_sgd_route_delta_linearized"]),
            )
        )
    lines.extend(
        [
            "",
            "## Top Route-Supporting Data Groups",
            "",
            "| data group | route support | actual update loss reduction | cosine to route | cosine to actual update |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["final_top_route_support"]:
        route_cosine = row["loss_negative_route_gradient_cosine"]
        update_cosine = row["loss_negative_actual_update_cosine"]
        lines.append(
            "| {group} | {support:.6g} | {update:.6g} | {route_cosine} | {update_cosine} |".format(
                group=row["data_group_id"],
                support=float(row["negative_loss_dot_route_gradient"]),
                update=float(row["loss_reduction_under_actual_update_linearized"]),
                route_cosine="" if route_cosine is None else f"{float(route_cosine):.6f}",
                update_cosine="" if update_cosine is None else f"{float(update_cosine):.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- route rows: `{report['route_rows_path']}`",
            f"- data rows: `{report['data_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
            "",
            "## Plots",
            "",
        ]
    )
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_data_update_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    stage_name: str,
    subspace_name: str,
    position_role: str,
    pair_types: list[str],
    route_pair_type: str,
    data_group_fields: list[str],
    rank: int | None = None,
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    head_layer: int | None = None,
    head: int | None = None,
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    split_filter: list[str] | None = None,
    route_split: str = "__all__",
    loss_side: str = "both",
    top_k_data_groups: int = 24,
    min_error_denominator: float = 1.0e-9,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Path]]:
    if loss_side not in ROUTE_GRADIENT_LOSS_SIDES:
        raise ValueError(f"Unsupported loss_side {loss_side!r}; expected one of {ROUTE_GRADIENT_LOSS_SIDES}.")
    if top_k_data_groups <= 0:
        raise ValueError("top_k_data_groups must be positive.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    if not data_group_fields:
        raise ValueError("data_group_fields must not be empty.")
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("data-update-attribution requires at least two checkpoints.")
    model = build_model(spec.model, len(vocab.tokens), device)
    _validate_geometry_stage(model=model, stage_name=stage_name)
    if position_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported position role {position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    if not pairs:
        raise RuntimeError("Data update attribution constructed no pairs.")
    route_pairs = _route_objective_pairs(
        pairs=pairs,
        route_split=route_split,
        route_pair_type=route_pair_type,
    )
    data_groups = _group_pairs_for_data_update(
        pairs=pairs,
        data_group_fields=data_group_fields,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    route_rows_path = output_dir / "data_update_attribution_route_rows.jsonl"
    data_rows_path = output_dir / "data_update_attribution_rows.jsonl"
    pair_rows_path = output_dir / "data_update_attribution_pairs.jsonl"
    progress_path = output_dir / "data_update_attribution_progress.json"
    for partial_path in (route_rows_path, data_rows_path, pair_rows_path, progress_path):
        if partial_path.exists():
            partial_path.unlink()
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[data-update-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} pairs={len(pairs)} route_pairs={len(route_pairs)} "
        f"pair_types={pair_types} route={route_split}/{route_pair_type} data_groups={len(data_groups)} "
        f"fields={data_group_fields} device={device_name} subspace={subspace_name} rank={rank} "
        f"stage={stage_name} role={position_role} loss_side={loss_side}",
        flush=True,
    )

    all_route_rows: list[dict[str, Any]] = []
    all_data_rows: list[dict[str, Any]] = []
    final_subspace_summary: dict[str, Any] | None = None
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        learning_rate = _compute_learning_rate(spec.optimization, source_step)
        print(
            "[data-update-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )
        route_row, data_rows, subspace_summary = _compute_data_update_attribution_interval(
            model=model,
            source_checkpoint_path=source_checkpoint_path,
            target_checkpoint_path=target_checkpoint_path,
            pairs=pairs,
            route_pairs=route_pairs,
            data_groups=data_groups,
            vocab=vocab,
            learning_rate=learning_rate,
            route_split=route_split,
            route_pair_type=route_pair_type,
            subspace_name=subspace_name,
            rank=rank,
            head_layer=head_layer,
            head=head,
            stage_name=stage_name,
            position_role=position_role,
            loss_side=loss_side,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            min_error_denominator=min_error_denominator,
        )
        final_subspace_summary = subspace_summary
        append_jsonl(route_rows_path, route_row)
        for row in data_rows:
            append_jsonl(data_rows_path, row)
        all_route_rows.append(route_row)
        all_data_rows.extend(data_rows)
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_intervals": interval_index,
                "total_intervals": len(intervals),
                "last_source_step": source_step,
                "last_target_step": target_step,
                "route_rows_path": str(route_rows_path),
                "data_rows_path": str(data_rows_path),
                "pair_rows_path": str(pair_rows_path),
            },
        )
        all_data_row = next(row for row in data_rows if str(row["data_group_id"]) == "__all__")
        print(
            "[data-update-attribution] finished "
            f"{source_step}->{target_step} actual_route_delta={float(route_row['actual_delta']):.6g} "
            f"predicted_route_delta={float(route_row['predicted_delta']):.6g} "
            f"all_data_route_support={float(all_data_row['negative_loss_dot_route_gradient']):.6g} "
            f"all_data_update_alignment={float(all_data_row['negative_loss_dot_actual_update']):.6g}",
            flush=True,
        )

    if final_subspace_summary is None:
        raise RuntimeError("No checkpoint intervals were processed for data update attribution.")
    summary = _summarize_data_update_attribution(
        route_rows=all_route_rows,
        data_rows=all_data_rows,
        top_k_data_groups=top_k_data_groups,
    )
    report_path = output_dir / "data_update_attribution_report.json"
    markdown_path = output_dir / "data_update_attribution_report.md"
    plot_paths: dict[str, Path] = {}
    update_plot = _plot_data_update_group_bars(
        data_rows=all_data_rows,
        value_field="loss_reduction_under_actual_update_linearized",
        title="Data groups aligned with actual checkpoint update",
        xlabel="<-grad loss_group, actual Delta theta>",
        top_k_data_groups=top_k_data_groups,
        output_path=output_dir / "data_update_actual_update_alignment.svg",
    )
    if update_plot is not None:
        plot_paths["actual_update_alignment"] = update_plot
    route_plot = _plot_data_update_group_bars(
        data_rows=all_data_rows,
        value_field="negative_loss_dot_route_gradient",
        title="Data groups supporting route gradient",
        xlabel="<-grad loss_group, grad route>",
        top_k_data_groups=top_k_data_groups,
        output_path=output_dir / "data_update_route_support.svg",
    )
    if route_plot is not None:
        plot_paths["route_support"] = route_plot
    scatter_plot = _plot_data_update_vs_route_support(
        data_rows=all_data_rows,
        output_path=output_dir / "data_update_alignment_vs_route_support.svg",
    )
    if scatter_plot is not None:
        plot_paths["alignment_vs_route_support"] = scatter_plot

    report = {
        "schema_version": DATA_UPDATE_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "stage": stage_name,
        "subspace": final_subspace_summary,
        "subspace_name": subspace_name,
        "rank": rank,
        "position_role": position_role,
        "pair_types": pair_types,
        "route_split": route_split,
        "route_pair_type": route_pair_type,
        "data_group_fields": data_group_fields,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "split_filter": split_filter,
        "loss_side": loss_side,
        "top_k_data_groups": top_k_data_groups,
        "min_error_denominator": min_error_denominator,
        "basis_mode": "source_checkpoint_per_interval",
        "calculation": {
            "route_score": "patched_transfer_margin - corrupted_transfer_margin",
            "actual_route_delta": "route_score(theta_target; source_basis) - route_score(theta_source; source_basis)",
            "actual_update_predicted_route_delta": "grad route(theta_source; source_basis) . Delta theta",
            "data_route_support": "< -grad loss_data_group(theta_source), grad route(theta_source; source_basis) >",
            "data_actual_update_alignment": "< -grad loss_data_group(theta_source), Delta theta >",
            "interpretation_caveat": (
                "This is a source-checkpoint diagnostic against the observed checkpoint delta, "
                "not replayed historical optimizer batches."
            ),
        },
        "pair_construction": pair_construction,
        "route_num_pairs": len(route_pairs),
        "data_num_groups": len(data_groups),
        "route_rows_path": str(route_rows_path),
        "data_rows_path": str(data_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_data_update_attribution_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_intervals": len(intervals),
            "total_intervals": len(intervals),
            "last_source_step": int(summary["final_route"]["source_step"]),
            "last_target_step": int(summary["final_route"]["target_step"]),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "route_rows_path": str(route_rows_path),
            "data_rows_path": str(data_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[data-update-attribution] complete report={report_path} rows={data_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, route_rows_path, data_rows_path, pair_rows_path, plot_paths


@dataclass(frozen=True)
class _RouteCompetitionSpec:
    label: str
    stage_name: str
    subspace_name: str
    position_role: str
    rank: int | None = None
    head_layer: int | None = None
    head: int | None = None


ROUTE_COMPETITION_SPEC_KEYS = {
    "label",
    "stage",
    "subspace",
    "position_role",
    "rank",
    "head_layer",
    "head",
}


def _parse_optional_int_field(*, fields: dict[str, str], key: str) -> int | None:
    if key not in fields:
        return None
    raw_value = fields[key]
    if raw_value.lower() in {"none", "null"}:
        return None
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Route competition field {key!r} must be an integer or none, got {raw_value!r}.") from exc


def _parse_route_competition_spec(raw_spec: str) -> _RouteCompetitionSpec:
    if not raw_spec.strip():
        raise ValueError("Route competition spec must not be empty.")
    fields: dict[str, str] = {}
    for item in raw_spec.split(","):
        if "=" not in item:
            raise ValueError(f"Route competition spec item must be key=value: {item!r} in {raw_spec!r}")
        key, value = item.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip()
        if key not in ROUTE_COMPETITION_SPEC_KEYS:
            raise ValueError(
                f"Unsupported route competition spec key {key!r}; expected one of {sorted(ROUTE_COMPETITION_SPEC_KEYS)}."
            )
        if key in fields:
            raise ValueError(f"Duplicate route competition spec key {key!r} in {raw_spec!r}.")
        if not value:
            raise ValueError(f"Route competition spec key {key!r} has an empty value in {raw_spec!r}.")
        fields[key] = value
    required = ["label", "stage", "subspace", "position_role"]
    missing = [key for key in required if key not in fields]
    if missing:
        raise ValueError(f"Route competition spec is missing required keys {missing}: {raw_spec!r}")
    return _RouteCompetitionSpec(
        label=fields["label"],
        stage_name=fields["stage"],
        subspace_name=fields["subspace"],
        position_role=fields["position_role"],
        rank=_parse_optional_int_field(fields=fields, key="rank"),
        head_layer=_parse_optional_int_field(fields=fields, key="head_layer"),
        head=_parse_optional_int_field(fields=fields, key="head"),
    )


def _parse_route_competition_specs(raw_specs: list[str]) -> list[_RouteCompetitionSpec]:
    if not raw_specs:
        raise ValueError("At least one route spec is required.")
    specs = [_parse_route_competition_spec(raw_spec) for raw_spec in raw_specs]
    labels = [spec.label for spec in specs]
    duplicate_labels = sorted(label for label in set(labels) if labels.count(label) > 1)
    if duplicate_labels:
        raise ValueError(f"Route competition labels must be unique; duplicates={duplicate_labels}")
    return specs


def _route_competition_route_metadata(route_spec: _RouteCompetitionSpec) -> dict[str, Any]:
    return {
        "route_label": route_spec.label,
        "stage": route_spec.stage_name,
        "subspace_name": route_spec.subspace_name,
        "position_role": route_spec.position_role,
        "rank": route_spec.rank,
        "head_layer": route_spec.head_layer,
        "head": route_spec.head,
    }


def _annotate_route_competition_row(
    *,
    row: dict[str, Any],
    route_spec: _RouteCompetitionSpec,
    domain: str,
) -> dict[str, Any]:
    annotated = dict(row)
    annotated.update(_route_competition_route_metadata(route_spec))
    annotated["domain"] = domain
    return annotated


def _build_route_competition_pairs(
    *,
    probe_set_path: Path,
    spec: TrainSpec,
    vocab: Vocabulary,
    holdout_pairs: set[tuple[str, str]],
    pair_types: list[str],
    max_pairs_per_type: int,
    min_pairs_per_type: int,
    split_filter: list[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    return _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )


def _summarize_route_competition_report(
    *,
    route_rows: list[dict[str, Any]],
    data_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not route_rows:
        raise ValueError("Cannot summarize route competition without route rows.")
    if not data_rows:
        raise ValueError("Cannot summarize route competition without data rows.")
    final_target_step = max(int(row["target_step"]) for row in route_rows)
    final_route_rows = [row for row in route_rows if int(row["target_step"]) == final_target_step]
    final_data_rows = [row for row in data_rows if int(row["target_step"]) == final_target_step]
    route_labels = sorted({str(row["route_label"]) for row in final_route_rows})
    combined_rows: list[dict[str, Any]] = []
    for label in route_labels:
        eval_rows = [
            row for row in final_route_rows if str(row["route_label"]) == label and str(row["domain"]) == "eval"
        ]
        train_rows = [
            row for row in final_route_rows if str(row["route_label"]) == label and str(row["domain"]) == "train"
        ]
        if len(eval_rows) != 1 or len(train_rows) != 1:
            raise RuntimeError(
                f"Expected one eval and one train route row for {label}, got eval={len(eval_rows)} train={len(train_rows)}."
            )
        eval_data_all = [
            row
            for row in final_data_rows
            if str(row["route_label"]) == label and str(row["domain"]) == "eval" and str(row["data_group_id"]) == "__all__"
        ]
        train_data_all = [
            row
            for row in final_data_rows
            if str(row["route_label"]) == label and str(row["domain"]) == "train" and str(row["data_group_id"]) == "__all__"
        ]
        if len(eval_data_all) != 1 or len(train_data_all) != 1:
            raise RuntimeError(
                f"Expected one eval and one train __all__ data row for {label}, "
                f"got eval={len(eval_data_all)} train={len(train_data_all)}."
            )
        eval_row = eval_rows[0]
        train_row = train_rows[0]
        eval_data = eval_data_all[0]
        train_data = train_data_all[0]
        combined_rows.append(
            {
                "route_label": label,
                "stage": train_row["stage"],
                "subspace_name": train_row["subspace_name"],
                "rank": train_row["rank"],
                "head_label": train_row["head_label"],
                "position_role": train_row["position_role"],
                "eval_source_route_score": float(eval_row["source_route_score"]),
                "eval_target_route_score": float(eval_row["target_route_score"]),
                "eval_actual_delta": float(eval_row["actual_delta"]),
                "eval_predicted_delta": float(eval_row["predicted_delta"]),
                "eval_relative_error": float(eval_row["relative_error"]),
                "eval_sign_match": bool(eval_row["sign_match"]),
                "eval_route_support": float(eval_data["negative_loss_dot_route_gradient"]),
                "eval_actual_update_loss_reduction": float(eval_data["loss_reduction_under_actual_update_linearized"]),
                "train_source_route_score": float(train_row["source_route_score"]),
                "train_target_route_score": float(train_row["target_route_score"]),
                "train_actual_delta": float(train_row["actual_delta"]),
                "train_predicted_delta": float(train_row["predicted_delta"]),
                "train_relative_error": float(train_row["relative_error"]),
                "train_sign_match": bool(train_row["sign_match"]),
                "train_route_support": float(train_data["negative_loss_dot_route_gradient"]),
                "train_actual_update_loss_reduction": float(train_data["loss_reduction_under_actual_update_linearized"]),
                "train_local_sgd_route_delta": float(train_data["local_sgd_route_delta_linearized"]),
            }
        )
    return {
        "num_routes": len(route_labels),
        "final_target_step": final_target_step,
        "combined_rows": sorted(combined_rows, key=lambda row: str(row["route_label"])),
        "ranked_by_train_route_support": sorted(
            combined_rows,
            key=lambda row: float(row["train_route_support"]),
            reverse=True,
        ),
        "ranked_by_eval_actual_delta": sorted(
            combined_rows,
            key=lambda row: float(row["eval_actual_delta"]),
            reverse=True,
        ),
        "ranked_by_train_actual_delta": sorted(
            combined_rows,
            key=lambda row: float(row["train_actual_delta"]),
            reverse=True,
        ),
    }


def _plot_route_competition_bars(
    *,
    combined_rows: list[dict[str, Any]],
    value_field: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path | None:
    if not combined_rows:
        return None
    rows = sorted(combined_rows, key=lambda row: float(row[value_field]), reverse=True)
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(rows)), 5))
    labels = [str(row["route_label"]) for row in rows]
    values = [float(row[value_field]) for row in rows]
    colors = ["#376f8f" if value >= 0.0 else "#8f374a" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _plot_route_competition_predicted_vs_actual(
    *,
    combined_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    if not combined_rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 7))
    for row in combined_rows:
        ax.scatter(float(row["eval_predicted_delta"]), float(row["eval_actual_delta"]), color="#376f8f")
        ax.annotate(str(row["route_label"]), (float(row["eval_predicted_delta"]), float(row["eval_actual_delta"])), fontsize=8)
    values = [float(row["eval_predicted_delta"]) for row in combined_rows] + [
        float(row["eval_actual_delta"]) for row in combined_rows
    ]
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0
    ax.plot([min_value, max_value], [min_value, max_value], color="#777777", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.axvline(0.0, color="#999999", linewidth=0.8)
    ax.set_title("Route competition: eval actual vs predicted delta")
    ax.set_xlabel("grad(route) . Delta theta")
    ax.set_ylabel("route(theta_target) - route(theta_source)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg")
    plt.close(fig)
    return output_path


def _write_route_competition_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary = report["summary"]
    lines = [
        "# Route Competition Report",
        "",
        "## Calculation",
        "",
        "For every candidate route, this report computes:",
        "",
        "```text",
        "actual_delta = route(theta_target; source_basis) - route(theta_source; source_basis)",
        "predicted_delta = grad route(theta_source; source_basis) . Delta theta",
        "data_route_support = < -grad loss_data(theta_source), grad route(theta_source; source_basis) >",
        "data_actual_update_alignment = < -grad loss_data(theta_source), Delta theta >",
        "```",
        "",
        "The route basis is fixed at the source checkpoint for each interval.",
        "",
        "## Final Competition Table",
        "",
        "| route | stage | subspace | eval actual delta | eval predicted delta | eval rel err | train support | eval support | train actual delta | train predicted delta |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["ranked_by_train_route_support"]:
        lines.append(
            "| {route} | {stage} | {subspace} | {eval_actual:.6g} | {eval_pred:.6g} | {eval_err:.6g} | {train_support:.6g} | {eval_support:.6g} | {train_actual:.6g} | {train_pred:.6g} |".format(
                route=row["route_label"],
                stage=row["stage"],
                subspace=row["subspace_name"],
                eval_actual=float(row["eval_actual_delta"]),
                eval_pred=float(row["eval_predicted_delta"]),
                eval_err=float(row["eval_relative_error"]),
                train_support=float(row["train_route_support"]),
                eval_support=float(row["eval_route_support"]),
                train_actual=float(row["train_actual_delta"]),
                train_pred=float(row["train_predicted_delta"]),
            )
        )
    lines.extend(
        [
            "",
            "## Ranked By Train Route Support",
            "",
            "| rank | route | train route support | train update loss reduction | train local SGD route delta |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for index, row in enumerate(summary["ranked_by_train_route_support"], start=1):
        lines.append(
            "| {rank} | {route} | {support:.6g} | {update:.6g} | {delta:.6g} |".format(
                rank=index,
                route=row["route_label"],
                support=float(row["train_route_support"]),
                update=float(row["train_actual_update_loss_reduction"]),
                delta=float(row["train_local_sgd_route_delta"]),
            )
        )
    lines.extend(
        [
            "",
            "## Route Specs",
            "",
        ]
    )
    for route in report["routes"]:
        lines.append(f"- `{route['route_label']}`: `{route}`")
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- route rows: `{report['route_rows_path']}`",
            f"- data rows: `{report['data_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
            "",
            "## Plots",
            "",
        ]
    )
    for label, plot_path in plot_paths.items():
        lines.append(f"- {label}: `{plot_path}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_route_competition_report(
    *,
    config_path: Path,
    probe_set_path: Path,
    train_probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    raw_route_specs: list[str],
    route_pair_type: str,
    eval_pair_types: list[str],
    train_pair_types: list[str],
    data_group_fields: list[str],
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    eval_split_filter: list[str] | None = None,
    train_split_filter: list[str] | None = None,
    eval_loss_side: str = "both",
    train_loss_side: str = "clean",
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    min_error_denominator: float = 1.0e-9,
) -> tuple[Path, Path, Path, Path, Path, dict[str, Path]]:
    if eval_loss_side not in ROUTE_GRADIENT_LOSS_SIDES:
        raise ValueError(f"Unsupported eval_loss_side {eval_loss_side!r}; expected one of {ROUTE_GRADIENT_LOSS_SIDES}.")
    if train_loss_side not in ROUTE_GRADIENT_LOSS_SIDES:
        raise ValueError(f"Unsupported train_loss_side {train_loss_side!r}; expected one of {ROUTE_GRADIENT_LOSS_SIDES}.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    routes = _parse_route_competition_specs(raw_route_specs)
    if not data_group_fields:
        raise ValueError("data_group_fields must not be empty.")
    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("route-competition-report requires at least two checkpoints.")
    model = build_model(spec.model, len(vocab.tokens), device)
    for route in routes:
        _validate_geometry_stage(model=model, stage_name=route.stage_name)
        if route.position_role not in GEOMETRY_POSITION_ROLES:
            raise ValueError(f"Unsupported position role {route.position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")

    eval_pair_types = sorted(set(eval_pair_types), key=eval_pair_types.index)
    train_pair_types = sorted(set(train_pair_types), key=train_pair_types.index)
    eval_pairs, eval_pair_construction = _build_route_competition_pairs(
        probe_set_path=probe_set_path,
        spec=spec,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=eval_pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=eval_split_filter,
    )
    train_pairs, train_pair_construction = _build_route_competition_pairs(
        probe_set_path=train_probe_set_path,
        spec=spec,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=train_pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=train_split_filter,
    )
    eval_route_pairs = _route_objective_pairs(
        pairs=eval_pairs,
        route_split="__all__",
        route_pair_type=route_pair_type,
    )
    train_route_pairs = _route_objective_pairs(
        pairs=train_pairs,
        route_split="__all__",
        route_pair_type=route_pair_type,
    )
    eval_data_groups = _group_pairs_for_data_update(pairs=eval_pairs, data_group_fields=data_group_fields)
    train_data_groups = _group_pairs_for_data_update(pairs=train_pairs, data_group_fields=data_group_fields)

    output_dir.mkdir(parents=True, exist_ok=True)
    route_rows_path = output_dir / "route_competition_route_rows.jsonl"
    data_rows_path = output_dir / "route_competition_data_rows.jsonl"
    pair_rows_path = output_dir / "route_competition_pairs.jsonl"
    progress_path = output_dir / "route_competition_progress.json"
    for partial_path in (route_rows_path, data_rows_path, pair_rows_path, progress_path):
        if partial_path.exists():
            partial_path.unlink()
    pair_rows = [
        {"domain": "eval", **_pair_metadata(pair)}
        for pair in eval_pairs
    ] + [
        {"domain": "train", **_pair_metadata(pair)}
        for pair in train_pairs
    ]
    write_jsonl(pair_rows_path, pair_rows)

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[route-competition-report] "
        f"routes={len(routes)} intervals={len(intervals)} eval_pairs={len(eval_pairs)} train_pairs={len(train_pairs)} "
        f"route_pair_type={route_pair_type} device={device_name}",
        flush=True,
    )

    all_route_rows: list[dict[str, Any]] = []
    all_data_rows: list[dict[str, Any]] = []
    final_subspace_summaries: dict[str, dict[str, Any]] = {}
    total_runs = len(routes) * len(intervals) * 2
    completed_runs = 0
    for route in routes:
        for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
            source_step = _checkpoint_step_from_path(source_checkpoint_path)
            target_step = _checkpoint_step_from_path(target_checkpoint_path)
            learning_rate = _compute_learning_rate(spec.optimization, source_step)
            print(
                "[route-competition-report] starting "
                f"route={route.label} interval={interval_index}/{len(intervals)} "
                f"{source_checkpoint_path.name}->{target_checkpoint_path.name}",
                flush=True,
            )
            eval_route_row, eval_data_rows, eval_subspace_summary = _compute_data_update_attribution_interval(
                model=model,
                source_checkpoint_path=source_checkpoint_path,
                target_checkpoint_path=target_checkpoint_path,
                pairs=eval_pairs,
                route_pairs=eval_route_pairs,
                data_groups=eval_data_groups,
                vocab=vocab,
                learning_rate=learning_rate,
                route_split="__all__",
                route_pair_type=route_pair_type,
                subspace_name=route.subspace_name,
                rank=route.rank,
                head_layer=route.head_layer,
                head=route.head,
                stage_name=route.stage_name,
                position_role=route.position_role,
                loss_side=eval_loss_side,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
                min_error_denominator=min_error_denominator,
            )
            completed_runs += 1
            train_route_row, train_data_rows, train_subspace_summary = _compute_data_update_attribution_interval(
                model=model,
                source_checkpoint_path=source_checkpoint_path,
                target_checkpoint_path=target_checkpoint_path,
                pairs=train_pairs,
                route_pairs=train_route_pairs,
                data_groups=train_data_groups,
                vocab=vocab,
                learning_rate=learning_rate,
                route_split="__all__",
                route_pair_type=route_pair_type,
                subspace_name=route.subspace_name,
                rank=route.rank,
                head_layer=route.head_layer,
                head=route.head,
                stage_name=route.stage_name,
                position_role=route.position_role,
                loss_side=train_loss_side,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
                min_error_denominator=min_error_denominator,
            )
            completed_runs += 1
            final_subspace_summaries[f"{route.label}:eval"] = eval_subspace_summary
            final_subspace_summaries[f"{route.label}:train"] = train_subspace_summary

            annotated_route_rows = [
                _annotate_route_competition_row(row=eval_route_row, route_spec=route, domain="eval"),
                _annotate_route_competition_row(row=train_route_row, route_spec=route, domain="train"),
            ]
            annotated_data_rows = [
                _annotate_route_competition_row(row=row, route_spec=route, domain="eval")
                for row in eval_data_rows
            ] + [
                _annotate_route_competition_row(row=row, route_spec=route, domain="train")
                for row in train_data_rows
            ]
            for row in annotated_route_rows:
                append_jsonl(route_rows_path, row)
            for row in annotated_data_rows:
                append_jsonl(data_rows_path, row)
            all_route_rows.extend(annotated_route_rows)
            all_data_rows.extend(annotated_data_rows)
            write_json(
                progress_path,
                {
                    "status": "running",
                    "completed_runs": completed_runs,
                    "total_runs": total_runs,
                    "last_route_label": route.label,
                    "last_source_step": source_step,
                    "last_target_step": target_step,
                    "route_rows_path": str(route_rows_path),
                    "data_rows_path": str(data_rows_path),
                    "pair_rows_path": str(pair_rows_path),
                },
            )
            eval_all = next(row for row in eval_data_rows if str(row["data_group_id"]) == "__all__")
            train_all = next(row for row in train_data_rows if str(row["data_group_id"]) == "__all__")
            print(
                "[route-competition-report] finished "
                f"route={route.label} {source_step}->{target_step} "
                f"eval_actual_delta={float(eval_route_row['actual_delta']):.6g} "
                f"eval_predicted_delta={float(eval_route_row['predicted_delta']):.6g} "
                f"train_support={float(train_all['negative_loss_dot_route_gradient']):.6g} "
                f"eval_support={float(eval_all['negative_loss_dot_route_gradient']):.6g}",
                flush=True,
            )

    summary = _summarize_route_competition_report(
        route_rows=all_route_rows,
        data_rows=all_data_rows,
    )
    report_path = output_dir / "route_competition_report.json"
    markdown_path = output_dir / "route_competition_report.md"
    plot_paths: dict[str, Path] = {}
    train_support_plot = _plot_route_competition_bars(
        combined_rows=summary["combined_rows"],
        value_field="train_route_support",
        title="Route competition: train route support",
        ylabel="<-grad train loss, grad route>",
        output_path=output_dir / "route_competition_train_support.svg",
    )
    if train_support_plot is not None:
        plot_paths["train_support"] = train_support_plot
    eval_delta_plot = _plot_route_competition_bars(
        combined_rows=summary["combined_rows"],
        value_field="eval_actual_delta",
        title="Route competition: eval actual route delta",
        ylabel="route(theta_target) - route(theta_source)",
        output_path=output_dir / "route_competition_eval_actual_delta.svg",
    )
    if eval_delta_plot is not None:
        plot_paths["eval_actual_delta"] = eval_delta_plot
    predicted_plot = _plot_route_competition_predicted_vs_actual(
        combined_rows=summary["combined_rows"],
        output_path=output_dir / "route_competition_eval_predicted_vs_actual.svg",
    )
    if predicted_plot is not None:
        plot_paths["eval_predicted_vs_actual"] = predicted_plot

    report = {
        "schema_version": ROUTE_COMPETITION_REPORT_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "train_probe_set_path": str(train_probe_set_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "routes": [_route_competition_route_metadata(route) for route in routes],
        "subspaces": final_subspace_summaries,
        "route_pair_type": route_pair_type,
        "eval_pair_types": eval_pair_types,
        "train_pair_types": train_pair_types,
        "data_group_fields": data_group_fields,
        "eval_split_filter": eval_split_filter,
        "train_split_filter": train_split_filter,
        "eval_loss_side": eval_loss_side,
        "train_loss_side": train_loss_side,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "min_error_denominator": min_error_denominator,
        "basis_mode": "source_checkpoint_per_interval",
        "calculation": {
            "actual_delta": "route(theta_target; source_basis) - route(theta_source; source_basis)",
            "predicted_delta": "grad route(theta_source; source_basis) . Delta theta",
            "data_route_support": "< -grad loss_data(theta_source), grad route(theta_source; source_basis) >",
            "data_actual_update_alignment": "< -grad loss_data(theta_source), Delta theta >",
        },
        "pair_construction": {
            "eval": eval_pair_construction,
            "train": train_pair_construction,
        },
        "route_rows_path": str(route_rows_path),
        "data_rows_path": str(data_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_route_competition_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_runs": total_runs,
            "total_runs": total_runs,
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "route_rows_path": str(route_rows_path),
            "data_rows_path": str(data_rows_path),
            "pair_rows_path": str(pair_rows_path),
        },
    )
    print(
        f"[route-competition-report] complete report={report_path} rows={route_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, route_rows_path, data_rows_path, pair_rows_path, plot_paths
