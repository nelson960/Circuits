from __future__ import annotations

import csv
import math
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import GEOMETRY_POSITION_ROLES, _intervention_positions_for_query
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import iter_jsonl, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


CONTEXTUAL_SVD_ALIGNMENT_SCHEMA_VERSION = 1
CONTEXTUAL_GROUP_BY_OPTIONS = ["position_token", "query_key", "support_key", "support_value", "answer_value"]


@dataclass(frozen=True)
class ContextualRoleSpec:
    label: str
    context_role: str
    group_by: str


def _checkpoint_step_from_path(path: Path) -> int:
    stem = path.stem
    prefix = "step_"
    if not stem.startswith(prefix):
        raise ValueError(f"Checkpoint filename must start with '{prefix}': {path}")
    return int(stem[len(prefix) :])


def _resolve_checkpoint_paths(*, checkpoint_dir: Path, checkpoint_paths: list[Path] | None) -> list[Path]:
    if checkpoint_paths is None:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        resolved = sorted(checkpoint_dir.glob("step_*.pt"), key=_checkpoint_step_from_path)
    else:
        resolved = [Path(path) for path in checkpoint_paths]
    if not resolved:
        raise FileNotFoundError(f"No checkpoints provided or found in {checkpoint_dir}")
    missing = [path for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint path(s): {[str(path) for path in missing]}")
    return sorted(resolved, key=_checkpoint_step_from_path)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / "contextual_svd_alignment_report.json",
        output_dir / "contextual_svd_alignment_report.md",
        output_dir / "contextual_svd_alignment_rows.jsonl",
        output_dir / "contextual_svd_alignment_rows.csv",
        output_dir / "contextual_svd_rank_aggregate_rows.jsonl",
        output_dir / "contextual_svd_rank_aggregate_rows.csv",
        output_dir / "contextual_svd_subspace_rows.jsonl",
        output_dir / "contextual_svd_role_vector_rows.jsonl",
        output_dir / "contextual_svd_alignment_trajectory.svg",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing contextual SVD-alignment outputs without --overwrite: "
            f"{[str(path) for path in existing]}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _make_probe_loader(*, probe_records: list[dict[str, Any]], batch_size: int, pad_token_id: int) -> DataLoader[Any]:
    if not probe_records:
        raise ValueError("probe_records must not be empty.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    return DataLoader(
        probe_records,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_symbolic_kv(batch, pad_token_id),
    )


def _parse_role_spec(spec_text: str) -> ContextualRoleSpec:
    fields: dict[str, str] = {}
    for chunk in spec_text.split(","):
        if "=" not in chunk:
            raise ValueError(
                f"Role spec must be comma-separated key=value pairs, got {spec_text!r} with malformed chunk {chunk!r}."
            )
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Role spec has empty key or value: {spec_text!r}")
        if key in fields:
            raise ValueError(f"Duplicate field {key!r} in role spec {spec_text!r}.")
        fields[key] = value
    required = ["label", "context_role", "group_by"]
    missing = [field for field in required if field not in fields]
    if missing:
        raise ValueError(f"Role spec {spec_text!r} is missing required fields {missing}.")
    context_role = fields["context_role"]
    group_by = fields["group_by"]
    if context_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported context_role {context_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if group_by not in CONTEXTUAL_GROUP_BY_OPTIONS:
        raise ValueError(f"Unsupported group_by {group_by!r}; expected one of {CONTEXTUAL_GROUP_BY_OPTIONS}.")
    return ContextualRoleSpec(label=fields["label"], context_role=context_role, group_by=group_by)


def _resolve_role_specs(*, roles: list[str] | None, role_specs_text: list[str] | None) -> list[ContextualRoleSpec]:
    resolved: list[ContextualRoleSpec] = []
    if roles is not None:
        for role in roles:
            if role not in GEOMETRY_POSITION_ROLES:
                raise ValueError(f"Unsupported role {role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
            resolved.append(ContextualRoleSpec(label=role, context_role=role, group_by="position_token"))
    if role_specs_text is not None:
        for spec_text in role_specs_text:
            resolved.append(_parse_role_spec(spec_text))
    if not resolved:
        raise ValueError("At least one contextual role must be provided via --role or --role-spec.")
    labels = [spec.label for spec in resolved]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Role labels must be unique; duplicate labels found: {duplicates}")
    return resolved


def _normalize_vector(vector: torch.Tensor, *, label: str) -> torch.Tensor:
    vector = vector.float()
    norm = vector.norm()
    if float(norm.item()) <= 0.0:
        raise RuntimeError(f"Cannot normalize zero vector: {label}")
    return vector / norm


def _abs_cosine(left: torch.Tensor, right: torch.Tensor, *, label: str) -> float:
    left_unit = _normalize_vector(left, label=f"{label}.left")
    right_unit = _normalize_vector(right, label=f"{label}.right")
    return float(left_unit.dot(right_unit).abs().item())


def _subspace_overlap(direction: torch.Tensor, basis: torch.Tensor, *, label: str) -> float:
    direction_unit = _normalize_vector(direction, label=label)
    basis = basis.float()
    projection = basis.matmul(basis.T.matmul(direction_unit))
    return float(projection.norm().item())


def _pca_basis(vectors: torch.Tensor, *, pca_rank: int, label: str) -> tuple[torch.Tensor, list[float], int]:
    if pca_rank <= 0:
        raise ValueError(f"pca_rank must be positive, got {pca_rank}.")
    if vectors.ndim != 2:
        raise ValueError(f"{label} vectors must be rank-2, got shape {tuple(vectors.shape)}.")
    if vectors.size(0) < pca_rank + 1:
        raise ValueError(f"{label} PCA rank {pca_rank} requires at least {pca_rank + 1} vectors, got {vectors.size(0)}.")
    centered = vectors.float() - vectors.float().mean(dim=0, keepdim=True)
    rank = int(torch.linalg.matrix_rank(centered).item())
    if rank < pca_rank:
        raise RuntimeError(f"{label} centered rank {rank} is below requested PCA rank {pca_rank}.")
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    return vh[:pca_rank, :].T.contiguous(), [float(value) for value in singular_values[:pca_rank].tolist()], rank


def _role_subspace(
    *,
    role_label: str,
    context_role: str,
    group_by: str,
    vectors_by_token: dict[int, list[torch.Tensor]],
    vocab: Vocabulary,
    pca_rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not vectors_by_token:
        raise RuntimeError(f"No contextual vectors collected for role {role_label!r}.")
    all_vectors = torch.stack([vector for vectors in vectors_by_token.values() for vector in vectors], dim=0).float()
    token_ids = sorted(vectors_by_token)
    token_mean_vectors = torch.stack(
        [torch.stack(vectors_by_token[token_id], dim=0).float().mean(dim=0) for token_id in token_ids],
        dim=0,
    )
    mean_direction = _normalize_vector(all_vectors.mean(dim=0), label=f"{role_label}.mean_direction")
    identity_basis, identity_singular_values, identity_rank = _pca_basis(
        token_mean_vectors,
        pca_rank=pca_rank,
        label=f"{role_label}.token_mean_identity",
    )
    all_vector_basis, all_vector_singular_values, all_vector_rank = _pca_basis(
        all_vectors,
        pca_rank=pca_rank,
        label=f"{role_label}.all_vectors",
    )
    subspace = {
        "mean_direction": mean_direction,
        "identity_basis": identity_basis,
        "all_vector_basis": all_vector_basis,
    }
    summary = {
        "role": role_label,
        "context_role": context_role,
        "group_by": group_by,
        "num_vectors": int(all_vectors.size(0)),
        "num_unique_tokens": len(token_ids),
        "tokens": [vocab.tokens[token_id] for token_id in token_ids],
        "ambient_dim": int(all_vectors.size(1)),
        "pca_rank": pca_rank,
        "identity_centered_rank": identity_rank,
        "identity_singular_values": identity_singular_values,
        "all_vector_centered_rank": all_vector_rank,
        "all_vector_singular_values": all_vector_singular_values,
    }
    return subspace, summary


def _load_behavior_by_step(
    *,
    behavior_rows_path: Path | None,
    behavior_split: str,
    margin_field: str,
    accuracy_field: str,
) -> dict[int, dict[str, float]]:
    if behavior_rows_path is None:
        return {}
    if not behavior_rows_path.exists():
        raise FileNotFoundError(f"Behavior rows file not found: {behavior_rows_path}")
    rows_by_step: dict[int, dict[str, float]] = {}
    for row in iter_jsonl(behavior_rows_path):
        if str(row.get("split")) != behavior_split:
            continue
        if margin_field not in row:
            raise KeyError(f"Missing behavior margin field '{margin_field}' in row from {behavior_rows_path}: {row}")
        step = int(row["step"])
        if step in rows_by_step:
            raise RuntimeError(f"Duplicate behavior row for split={behavior_split} step={step} in {behavior_rows_path}")
        behavior = {"answer_margin": float(row[margin_field])}
        if accuracy_field in row:
            behavior["answer_accuracy"] = float(row[accuracy_field])
        rows_by_step[step] = behavior
    if not rows_by_step:
        raise RuntimeError(f"No behavior rows found for split={behavior_split} in {behavior_rows_path}")
    return rows_by_step


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys):
        raise RuntimeError(f"Pearson inputs have different lengths: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y, strict=True)) / (denom_x * denom_y)


def _minmax(values: list[float], *, label: str) -> list[float]:
    if not values:
        raise RuntimeError(f"Cannot min-max normalize empty series: {label}")
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        raise RuntimeError(f"Cannot min-max normalize constant series: {label}")
    return [(value - min_value) / (max_value - min_value) for value in values]


def _import_matplotlib() -> Any:
    cache_dir = Path(tempfile.gettempdir()) / "circuit_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_trajectory(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    plot_left_role: str,
    plot_right_role: str,
    include_margin: bool,
) -> None:
    rank1 = [row for row in rows if int(row["singular_rank"]) == 1]
    left_rows = sorted([row for row in rank1 if row["role"] == plot_left_role], key=lambda row: int(row["checkpoint_step"]))
    right_rows = sorted([row for row in rank1 if row["role"] == plot_right_role], key=lambda row: int(row["checkpoint_step"]))
    if not left_rows:
        raise RuntimeError(f"No rank-1 rows found for plot_left_role={plot_left_role!r}.")
    if not right_rows:
        raise RuntimeError(f"No rank-1 rows found for plot_right_role={plot_right_role!r}.")
    left_steps = [int(row["checkpoint_step"]) for row in left_rows]
    right_steps = [int(row["checkpoint_step"]) for row in right_rows]
    if left_steps != right_steps:
        raise RuntimeError(f"Plot roles have different checkpoint steps: left={left_steps}, right={right_steps}")
    series: list[tuple[str, list[float]]] = [
        ("W_QK singular value", [float(row["singular_value"]) for row in left_rows]),
        (
            f"left {plot_left_role} identity overlap",
            [float(row["left_identity_overlap"]) for row in left_rows],
        ),
        (
            f"right {plot_right_role} identity overlap",
            [float(row["right_identity_overlap"]) for row in right_rows],
        ),
        (
            f"left {plot_left_role} mean cosine",
            [float(row["left_mean_cosine"]) for row in left_rows],
        ),
        (
            f"right {plot_right_role} mean cosine",
            [float(row["right_mean_cosine"]) for row in right_rows],
        ),
    ]
    if include_margin:
        series.append(("answer margin", [float(row["answer_margin"]) for row in left_rows]))
    normalized_series = [(label, _minmax(values, label=label)) for label, values in series]
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(13, 7))
    for label, values in normalized_series:
        ax.plot(left_steps, values, marker="o", linewidth=2, label=label)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("min-max normalized value")
    ax.set_title("Contextual residual alignment of W_QK singular directions")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _collect_contextual_vectors(
    *,
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    stage_name: str,
    role_specs: list[ContextualRoleSpec],
) -> tuple[dict[str, dict[int, list[torch.Tensor]]], list[dict[str, Any]]]:
    vectors_by_role: dict[str, dict[int, list[torch.Tensor]]] = {
        spec.label: defaultdict(list) for spec in role_specs
    }
    vector_rows: list[dict[str, Any]] = []
    for batch_index, raw_batch in enumerate(loader):
        batch = move_batch_to_device(raw_batch, device)
        with torch.no_grad():
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
            )
        if outputs.residual_streams is None:
            raise RuntimeError("Contextual SVD alignment requires residual streams.")
        if stage_name not in outputs.residual_streams:
            raise KeyError(f"Residual stage {stage_name!r} not found. Available: {sorted(outputs.residual_streams)}")
        residual = outputs.residual_streams[stage_name].detach().float().cpu()
        _, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        for flat_index in range(int(metadata["rows"].numel())):
            query_batch_row = int(metadata["rows"][flat_index].item())
            query_index = int(metadata["query_indices"][flat_index].item())
            record = batch["records"][query_batch_row]
            sample_id = str(record["sample_id"])
            split = str(record["split"])
            query_key_token_id = int(batch["input_ids"][query_batch_row, int(metadata["query_key_positions"][flat_index].item())].item())
            support_value_token_id = int(
                batch["input_ids"][query_batch_row, int(metadata["support_value_positions"][flat_index].item())].item()
            )
            answer_value_token_id = int(answer_targets[flat_index].item())
            support_key_batch_row, support_key_positions = _intervention_positions_for_query(
                batch=batch,
                metadata=metadata,
                flat_index=flat_index,
                position_role="support_key",
            )
            if support_key_batch_row != query_batch_row or len(support_key_positions) != 1:
                raise RuntimeError(
                    f"Expected exactly one support_key position for {sample_id} query {query_index}, got {support_key_positions}."
                )
            support_key_token_id = int(batch["input_ids"][support_key_batch_row, support_key_positions[0]].item())
            for spec in role_specs:
                batch_row, positions = _intervention_positions_for_query(
                    batch=batch,
                    metadata=metadata,
                    flat_index=flat_index,
                    position_role=spec.context_role,
                )
                for position in positions:
                    if spec.group_by == "position_token":
                        token_id = int(batch["input_ids"][batch_row, position].item())
                    elif spec.group_by == "query_key":
                        token_id = query_key_token_id
                    elif spec.group_by == "support_key":
                        token_id = support_key_token_id
                    elif spec.group_by == "support_value":
                        token_id = support_value_token_id
                    elif spec.group_by == "answer_value":
                        token_id = answer_value_token_id
                    else:
                        raise ValueError(f"Unhandled group_by mode: {spec.group_by}")
                    vector = residual[batch_row, position, :].clone()
                    vectors_by_role[spec.label][token_id].append(vector)
                    vector_rows.append(
                        {
                            "batch_index": batch_index,
                            "sample_id": sample_id,
                            "split": split,
                            "query_index": query_index,
                            "role": spec.label,
                            "context_role": spec.context_role,
                            "group_by": spec.group_by,
                            "position": int(position),
                            "group_token_id": token_id,
                            "position_token_id": int(batch["input_ids"][batch_row, position].item()),
                        }
                    )
    return vectors_by_role, vector_rows


def _write_markdown_report(path: Path, report: dict[str, Any], plot_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Contextual SVD Alignment",
        "",
        "This report measures whether QK singular directions align with contextual residual subspaces.",
        "",
        "## Scope",
        "",
        f"- head: `L{report['head_layer']}H{report['head']}`",
        f"- contextual stage: `{report['context_stage']}`",
        f"- checkpoints: `{report['num_checkpoints']}`",
        f"- singular ranks: `1..{report['top_ranks']}`",
        f"- PCA rank: `{report['pca_rank']}`",
        f"- roles: `{', '.join(report['roles'])}`",
        f"- probe records: `{report['num_probe_records']}`",
        f"- behavior rows loaded: `{report['behavior_rows_loaded']}`",
        "",
        "## Outputs",
        "",
        f"- alignment rows JSONL: `{report['alignment_rows_path']}`",
        f"- alignment rows CSV: `{report['alignment_csv_path']}`",
        f"- rank aggregate rows JSONL: `{report['rank_aggregate_rows_path']}`",
        f"- rank aggregate rows CSV: `{report['rank_aggregate_csv_path']}`",
        f"- subspace rows: `{report['subspace_rows_path']}`",
        f"- role vector rows: `{report['role_vector_rows_path']}`",
        f"- trajectory plot: `{report['trajectory_plot_path']}`",
        "",
        "## Rank-1 Plot Rows",
        "",
        "| step | role | singular value | left mean | left identity | left all-vector | right mean | right identity | right all-vector | answer margin |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in plot_rows:
        answer_margin = row.get("answer_margin")
        lines.append(
            "| "
            f"{row['checkpoint_step']} | "
            f"`{row['role']}` | "
            f"{float(row['singular_value']):.4f} | "
            f"{float(row['left_mean_cosine']):.4f} | "
            f"{float(row['left_identity_overlap']):.4f} | "
            f"{float(row['left_all_vector_overlap']):.4f} | "
            f"{float(row['right_mean_cosine']):.4f} | "
            f"{float(row['right_identity_overlap']):.4f} | "
            f"{float(row['right_all_vector_overlap']):.4f} | "
            f"{'n/a' if answer_margin is None else f'{float(answer_margin):.4f}'} |"
        )
    lines.extend(
        [
            "",
            "## Rank-1 Correlations",
            "",
            "Correlations are Pearson correlations over checkpoint-level rank-1 rows.",
            "",
        ]
    )
    for key, value in report["rank1_correlations"].items():
        lines.append(f"- `{key}`: `{'n/a' if value is None else f'{float(value):.4f}'}`")
    lines.extend(
        [
            "",
            "## Top-Rank Aggregate Correlations",
            "",
            "Weighted correlations use singular-value-weighted averages over the retained ranks. Max correlations use the maximum alignment found among the retained ranks.",
            "",
        ]
    )
    for key, value in report["rank_aggregate_correlations"].items():
        lines.append(f"- `{key}`: `{'n/a' if value is None else f'{float(value):.4f}'}`")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This tool tests whether the learned QK directions point at contextual residual subspaces. It does not by itself prove causal use; the result must be compared to route intervention and answer-margin evidence.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def run_contextual_svd_alignment(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    checkpoint_paths: list[Path] | None,
    device_name: str,
    head_layer: int,
    head: int,
    context_stage: str,
    roles: list[str] | None,
    role_specs_text: list[str] | None,
    plot_left_role: str,
    plot_right_role: str,
    top_ranks: int,
    pca_rank: int,
    batch_size: int,
    split_filter: list[str] | None,
    behavior_rows_path: Path | None,
    behavior_split: str,
    behavior_margin_field: str,
    behavior_accuracy_field: str,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if top_ranks <= 0:
        raise ValueError(f"top_ranks must be positive, got {top_ranks}.")
    if pca_rank <= 0:
        raise ValueError(f"pca_rank must be positive, got {pca_rank}.")
    role_specs = _resolve_role_specs(roles=roles, role_specs_text=role_specs_text)
    role_labels = [spec.label for spec in role_specs]
    if plot_left_role not in role_labels:
        raise ValueError(f"plot_left_role={plot_left_role!r} must be included in role labels={role_labels}.")
    if plot_right_role not in role_labels:
        raise ValueError(f"plot_right_role={plot_right_role!r} must be included in role labels={role_labels}.")
    _prepare_output_dir(output_dir, overwrite=overwrite)
    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if split_filter is not None:
        split_set = set(split_filter)
        probe_records = [record for record in probe_records if str(record["split"]) in split_set]
        if not probe_records:
            raise RuntimeError(f"Split filter {sorted(split_set)} matched no probe records in {probe_set_path}.")
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    behavior_by_step = _load_behavior_by_step(
        behavior_rows_path=behavior_rows_path,
        behavior_split=behavior_split,
        margin_field=behavior_margin_field,
        accuracy_field=behavior_accuracy_field,
    )
    if behavior_rows_path is not None:
        missing_behavior = [path.name for path in checkpoints if _checkpoint_step_from_path(path) not in behavior_by_step]
        if missing_behavior:
            raise RuntimeError(f"Behavior rows are missing checkpoint(s): {missing_behavior}")
    device = require_device(device_name)
    model = build_model(spec.model, vocab_size=len(vocab.tokens), device=device)
    if head_layer < 0 or head_layer >= len(model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model block range 0..{len(model.blocks) - 1}.")
    if head < 0 or head >= int(model.blocks[head_layer].attn.n_heads):
        raise ValueError(f"head {head} outside head range 0..{int(model.blocks[head_layer].attn.n_heads) - 1}.")
    valid_stages = ["embedding"]
    for layer_index in range(len(model.blocks)):
        valid_stages.extend([f"layer_{layer_index}_post_attn", f"layer_{layer_index}_post_mlp"])
    valid_stages.append("final_norm")
    if context_stage not in valid_stages:
        raise ValueError(f"Unsupported context_stage {context_stage!r}; expected one of {valid_stages}.")

    loader = _make_probe_loader(probe_records=probe_records, batch_size=batch_size, pad_token_id=vocab.pad_token_id)
    alignment_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    role_vector_rows: list[dict[str, Any]] = []

    print(
        "[contextual-svd-alignment] "
        f"checkpoints={len(checkpoints)} records={len(probe_records)} head=L{head_layer}H{head} "
        f"stage={context_stage} roles={role_labels} device={device_name}",
        flush=True,
    )
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint_step = _checkpoint_step_from_path(checkpoint_path)
        print(
            f"[contextual-svd-alignment] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        vectors_by_role, checkpoint_vector_rows = _collect_contextual_vectors(
            model=model,
            loader=loader,
            device=device,
            stage_name=context_stage,
            role_specs=role_specs,
        )
        subspaces: dict[str, dict[str, torch.Tensor]] = {}
        role_spec_by_label = {spec.label: spec for spec in role_specs}
        for role in role_labels:
            spec = role_spec_by_label[role]
            subspace, summary = _role_subspace(
                role_label=role,
                context_role=spec.context_role,
                group_by=spec.group_by,
                vectors_by_token=dict(vectors_by_role[role]),
                vocab=vocab,
                pca_rank=pca_rank,
            )
            subspaces[role] = subspace
            subspace_rows.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_step": checkpoint_step,
                    "context_stage": context_stage,
                    **summary,
                }
            )
        for vector_row in checkpoint_vector_rows:
            role_vector_rows.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_step": checkpoint_step,
                    "context_stage": context_stage,
                    **vector_row,
                    "group_token": vocab.tokens[int(vector_row["group_token_id"])],
                    "position_token": vocab.tokens[int(vector_row["position_token_id"])],
                }
            )

        block = model.blocks[head_layer]
        head_dim = int(block.attn.head_dim)
        head_slice = slice(head * head_dim, (head + 1) * head_dim)
        q_rows = block.attn.q_proj.weight.detach().float().cpu()[head_slice, :]
        k_rows = block.attn.k_proj.weight.detach().float().cpu()[head_slice, :]
        qk_matrix = q_rows.T.matmul(k_rows)
        u, singular_values, vh = torch.linalg.svd(qk_matrix, full_matrices=False)
        keep = min(top_ranks, int(singular_values.numel()))
        behavior = behavior_by_step.get(checkpoint_step)
        for rank_index in range(1, keep + 1):
            left = u[:, rank_index - 1]
            right = vh[rank_index - 1, :]
            for role in role_labels:
                subspace = subspaces[role]
                row = {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_step": checkpoint_step,
                    "context_stage": context_stage,
                    "head_layer": head_layer,
                    "head": head,
                    "singular_rank": rank_index,
                    "singular_value": float(singular_values[rank_index - 1].item()),
                    "role": role,
                    "role_num_vectors": int(sum(len(vectors) for vectors in vectors_by_role[role].values())),
                    "role_num_unique_tokens": int(len(vectors_by_role[role])),
                    "left_mean_cosine": _abs_cosine(left, subspace["mean_direction"], label=f"left.{role}.mean"),
                    "left_identity_overlap": _subspace_overlap(
                        left,
                        subspace["identity_basis"],
                        label=f"left.{role}.identity",
                    ),
                    "left_all_vector_overlap": _subspace_overlap(
                        left,
                        subspace["all_vector_basis"],
                        label=f"left.{role}.all_vector",
                    ),
                    "right_mean_cosine": _abs_cosine(right, subspace["mean_direction"], label=f"right.{role}.mean"),
                    "right_identity_overlap": _subspace_overlap(
                        right,
                        subspace["identity_basis"],
                        label=f"right.{role}.identity",
                    ),
                    "right_all_vector_overlap": _subspace_overlap(
                        right,
                        subspace["all_vector_basis"],
                        label=f"right.{role}.all_vector",
                    ),
                }
                if behavior is not None:
                    row["answer_margin"] = behavior["answer_margin"]
                    row["answer_accuracy"] = behavior.get("answer_accuracy")
                else:
                    row["answer_margin"] = None
                    row["answer_accuracy"] = None
                alignment_rows.append(row)
        print(
            "[contextual-svd-alignment] finished "
            f"step={checkpoint_step} sv1={float(singular_values[0].item()):.6g}",
            flush=True,
        )

    rank1_rows = [row for row in alignment_rows if int(row["singular_rank"]) == 1]
    if not rank1_rows:
        raise RuntimeError("No rank-1 contextual alignment rows were produced.")
    aggregate_rows: list[dict[str, Any]] = []
    grouped_rows: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in alignment_rows:
        grouped_rows[(int(row["checkpoint_step"]), str(row["role"]))].append(row)
    for (checkpoint_step, role), rows in sorted(grouped_rows.items()):
        rows = sorted(rows, key=lambda row: int(row["singular_rank"]))
        singular_values = [float(row["singular_value"]) for row in rows]
        singular_value_sum = sum(singular_values)
        if singular_value_sum <= 0.0:
            raise RuntimeError(f"Non-positive singular value sum for step={checkpoint_step} role={role}.")

        def weighted(metric: str) -> float:
            return sum(float(row[metric]) * float(row["singular_value"]) for row in rows) / singular_value_sum

        def max_metric(metric: str) -> float:
            return max(float(row[metric]) for row in rows)

        first = rows[0]
        aggregate_rows.append(
            {
                "checkpoint": first["checkpoint"],
                "checkpoint_name": first["checkpoint_name"],
                "checkpoint_step": checkpoint_step,
                "context_stage": context_stage,
                "head_layer": head_layer,
                "head": head,
                "role": role,
                "rank_count": len(rows),
                "singular_value_top": singular_values[0],
                "singular_value_sum": singular_value_sum,
                "left_mean_cosine_weighted": weighted("left_mean_cosine"),
                "left_identity_overlap_weighted": weighted("left_identity_overlap"),
                "left_all_vector_overlap_weighted": weighted("left_all_vector_overlap"),
                "right_mean_cosine_weighted": weighted("right_mean_cosine"),
                "right_identity_overlap_weighted": weighted("right_identity_overlap"),
                "right_all_vector_overlap_weighted": weighted("right_all_vector_overlap"),
                "left_mean_cosine_max": max_metric("left_mean_cosine"),
                "left_identity_overlap_max": max_metric("left_identity_overlap"),
                "left_all_vector_overlap_max": max_metric("left_all_vector_overlap"),
                "right_mean_cosine_max": max_metric("right_mean_cosine"),
                "right_identity_overlap_max": max_metric("right_identity_overlap"),
                "right_all_vector_overlap_max": max_metric("right_all_vector_overlap"),
                "answer_margin": first["answer_margin"],
                "answer_accuracy": first["answer_accuracy"],
            }
        )
    include_margin = behavior_rows_path is not None
    correlations: dict[str, float | None] = {}
    by_role = {
        role: sorted([row for row in rank1_rows if row["role"] == role], key=lambda row: int(row["checkpoint_step"]))
        for role in role_labels
    }
    for role, role_rows in by_role.items():
        singular_values_for_role = [float(row["singular_value"]) for row in role_rows]
        correlations[f"{role}.singular_value_vs_left_identity_overlap"] = _pearson(
            singular_values_for_role,
            [float(row["left_identity_overlap"]) for row in role_rows],
        )
        correlations[f"{role}.singular_value_vs_right_identity_overlap"] = _pearson(
            singular_values_for_role,
            [float(row["right_identity_overlap"]) for row in role_rows],
        )
        correlations[f"{role}.singular_value_vs_left_mean_cosine"] = _pearson(
            singular_values_for_role,
            [float(row["left_mean_cosine"]) for row in role_rows],
        )
        correlations[f"{role}.singular_value_vs_right_mean_cosine"] = _pearson(
            singular_values_for_role,
            [float(row["right_mean_cosine"]) for row in role_rows],
        )
        if include_margin:
            margins = [float(row["answer_margin"]) for row in role_rows]
            correlations[f"{role}.left_identity_overlap_vs_answer_margin"] = _pearson(
                [float(row["left_identity_overlap"]) for row in role_rows],
                margins,
            )
            correlations[f"{role}.right_identity_overlap_vs_answer_margin"] = _pearson(
                [float(row["right_identity_overlap"]) for row in role_rows],
                margins,
            )
            correlations[f"{role}.left_mean_cosine_vs_answer_margin"] = _pearson(
                [float(row["left_mean_cosine"]) for row in role_rows],
                margins,
            )
            correlations[f"{role}.right_mean_cosine_vs_answer_margin"] = _pearson(
                [float(row["right_mean_cosine"]) for row in role_rows],
                margins,
            )

    aggregate_correlations: dict[str, float | None] = {}
    by_role_aggregate = {
        role: sorted([row for row in aggregate_rows if row["role"] == role], key=lambda row: int(row["checkpoint_step"]))
        for role in role_labels
    }
    for role, role_rows in by_role_aggregate.items():
        singular_values_for_role = [float(row["singular_value_top"]) for row in role_rows]
        for metric in [
            "left_identity_overlap_weighted",
            "right_identity_overlap_weighted",
            "left_mean_cosine_weighted",
            "right_mean_cosine_weighted",
            "left_all_vector_overlap_weighted",
            "right_all_vector_overlap_weighted",
            "left_identity_overlap_max",
            "right_identity_overlap_max",
            "left_mean_cosine_max",
            "right_mean_cosine_max",
        ]:
            aggregate_correlations[f"{role}.singular_value_vs_{metric}"] = _pearson(
                singular_values_for_role,
                [float(row[metric]) for row in role_rows],
            )
        if include_margin:
            margins = [float(row["answer_margin"]) for row in role_rows]
            for metric in [
                "left_identity_overlap_weighted",
                "right_identity_overlap_weighted",
                "left_mean_cosine_weighted",
                "right_mean_cosine_weighted",
                "left_all_vector_overlap_weighted",
                "right_all_vector_overlap_weighted",
                "left_identity_overlap_max",
                "right_identity_overlap_max",
                "left_mean_cosine_max",
                "right_mean_cosine_max",
            ]:
                aggregate_correlations[f"{role}.{metric}_vs_answer_margin"] = _pearson(
                    [float(row[metric]) for row in role_rows],
                    margins,
                )

    report_path = output_dir / "contextual_svd_alignment_report.json"
    markdown_path = output_dir / "contextual_svd_alignment_report.md"
    alignment_rows_path = output_dir / "contextual_svd_alignment_rows.jsonl"
    alignment_csv_path = output_dir / "contextual_svd_alignment_rows.csv"
    rank_aggregate_rows_path = output_dir / "contextual_svd_rank_aggregate_rows.jsonl"
    rank_aggregate_csv_path = output_dir / "contextual_svd_rank_aggregate_rows.csv"
    subspace_rows_path = output_dir / "contextual_svd_subspace_rows.jsonl"
    role_vector_rows_path = output_dir / "contextual_svd_role_vector_rows.jsonl"
    trajectory_plot_path = output_dir / "contextual_svd_alignment_trajectory.svg"

    write_jsonl(alignment_rows_path, alignment_rows)
    _write_csv(
        alignment_csv_path,
        alignment_rows,
        fieldnames=[
            "checkpoint",
            "checkpoint_name",
            "checkpoint_step",
            "context_stage",
            "head_layer",
            "head",
            "singular_rank",
            "singular_value",
            "role",
            "role_num_vectors",
            "role_num_unique_tokens",
            "left_mean_cosine",
            "left_identity_overlap",
            "left_all_vector_overlap",
            "right_mean_cosine",
            "right_identity_overlap",
            "right_all_vector_overlap",
            "answer_margin",
            "answer_accuracy",
        ],
    )
    write_jsonl(rank_aggregate_rows_path, aggregate_rows)
    _write_csv(
        rank_aggregate_csv_path,
        aggregate_rows,
        fieldnames=[
            "checkpoint",
            "checkpoint_name",
            "checkpoint_step",
            "context_stage",
            "head_layer",
            "head",
            "role",
            "rank_count",
            "singular_value_top",
            "singular_value_sum",
            "left_mean_cosine_weighted",
            "left_identity_overlap_weighted",
            "left_all_vector_overlap_weighted",
            "right_mean_cosine_weighted",
            "right_identity_overlap_weighted",
            "right_all_vector_overlap_weighted",
            "left_mean_cosine_max",
            "left_identity_overlap_max",
            "left_all_vector_overlap_max",
            "right_mean_cosine_max",
            "right_identity_overlap_max",
            "right_all_vector_overlap_max",
            "answer_margin",
            "answer_accuracy",
        ],
    )
    write_jsonl(subspace_rows_path, subspace_rows)
    write_jsonl(role_vector_rows_path, role_vector_rows)
    _plot_trajectory(
        rows=alignment_rows,
        output_path=trajectory_plot_path,
        plot_left_role=plot_left_role,
        plot_right_role=plot_right_role,
        include_margin=include_margin,
    )
    plot_rows = [
        row
        for row in sorted(rank1_rows, key=lambda item: (int(item["checkpoint_step"]), str(item["role"])))
        if row["role"] in {plot_left_role, plot_right_role}
    ]
    report = {
        "schema_version": CONTEXTUAL_SVD_ALIGNMENT_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "probe_metadata": probe_metadata,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "output_dir": str(output_dir),
        "device": device_name,
        "head_layer": head_layer,
        "head": head,
        "context_stage": context_stage,
        "roles": role_labels,
        "role_specs": [
            {"label": spec.label, "context_role": spec.context_role, "group_by": spec.group_by}
            for spec in role_specs
        ],
        "plot_left_role": plot_left_role,
        "plot_right_role": plot_right_role,
        "top_ranks": top_ranks,
        "pca_rank": pca_rank,
        "batch_size": batch_size,
        "split_filter": split_filter,
        "num_probe_records": len(probe_records),
        "num_checkpoints": len(checkpoints),
        "num_alignment_rows": len(alignment_rows),
        "num_rank_aggregate_rows": len(aggregate_rows),
        "num_subspace_rows": len(subspace_rows),
        "num_role_vector_rows": len(role_vector_rows),
        "behavior_rows_loaded": behavior_rows_path is not None,
        "behavior_rows_path": None if behavior_rows_path is None else str(behavior_rows_path),
        "behavior_split": behavior_split,
        "behavior_margin_field": behavior_margin_field,
        "behavior_accuracy_field": behavior_accuracy_field,
        "matrix_convention": "W_QK = q_rows.T @ k_rows; left singular vectors are query residual directions, right singular vectors are key residual directions.",
        "subspace_convention": "identity_basis is PCA over per-token mean contextual residual vectors for each role; all_vector_basis is PCA over all collected contextual residual vectors for that role.",
        "plot_convention": "The trajectory SVG min-max normalizes each plotted series to put singular value, alignment, and answer margin on one axis.",
        "rank1_correlations": correlations,
        "rank_aggregate_correlations": aggregate_correlations,
        "alignment_rows_path": str(alignment_rows_path),
        "alignment_csv_path": str(alignment_csv_path),
        "rank_aggregate_rows_path": str(rank_aggregate_rows_path),
        "rank_aggregate_csv_path": str(rank_aggregate_csv_path),
        "subspace_rows_path": str(subspace_rows_path),
        "role_vector_rows_path": str(role_vector_rows_path),
        "trajectory_plot_path": str(trajectory_plot_path),
    }
    write_json(report_path, report)
    _write_markdown_report(path=markdown_path, report=report, plot_rows=plot_rows)
    print(f"[contextual-svd-alignment] complete report={report_path} rows={alignment_rows_path}", flush=True)
    return (
        report_path,
        markdown_path,
        alignment_rows_path,
        alignment_csv_path,
        rank_aggregate_rows_path,
        rank_aggregate_csv_path,
        subspace_rows_path,
        role_vector_rows_path,
        {"trajectory": trajectory_plot_path},
    )
