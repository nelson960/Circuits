from __future__ import annotations

import math
from collections import Counter, defaultdict
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
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


DATASET_GEOMETRY_SCHEMA_VERSION = 1
ATTENTION_GEOMETRY_SCHEMA_VERSION = 1
PATH_LOGIT_DECOMPOSITION_SCHEMA_VERSION = 1
PROMPT_NEURON_TRACE_SCHEMA_VERSION = 1
GEOMETRY_SUBSPACE_INTERVENTION_SCHEMA_VERSION = 1
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
