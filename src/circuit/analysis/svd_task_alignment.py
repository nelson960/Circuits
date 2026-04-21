from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path
from typing import Any

import torch

from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import read_symbolic_kv_stream_metadata
from circuit.io import iter_jsonl, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, require_device
from circuit.vocab import Vocabulary


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
    return resolved


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / "svd_task_alignment_report.json",
        output_dir / "svd_task_alignment_report.md",
        output_dir / "svd_task_alignment_rows.jsonl",
        output_dir / "svd_task_alignment_rows.csv",
        output_dir / "svd_task_subspace_rows.jsonl",
        output_dir / "svd_task_token_alignment_rows.jsonl",
        output_dir / "svd_task_alignment_trajectory.svg",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing SVD task-alignment outputs without --overwrite: "
            f"{[str(path) for path in existing]}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _normalize_vector(vector: torch.Tensor, *, label: str) -> torch.Tensor:
    norm = vector.norm()
    if float(norm.item()) <= 0.0:
        raise RuntimeError(f"Cannot normalize zero vector: {label}")
    return vector / norm


def _abs_cosine(left: torch.Tensor, right: torch.Tensor, *, label: str) -> float:
    left_unit = _normalize_vector(left.float(), label=f"{label}.left")
    right_unit = _normalize_vector(right.float(), label=f"{label}.right")
    return float(left_unit.dot(right_unit).abs().item())


def _subspace_overlap(direction: torch.Tensor, basis: torch.Tensor, *, label: str) -> float:
    direction_unit = _normalize_vector(direction.float(), label=label)
    projection = basis.float().matmul(basis.float().T.matmul(direction_unit))
    return float(projection.norm().item())


def _token_subspace(
    *,
    embedding_weight: torch.Tensor,
    token_ids: list[int],
    label: str,
    pca_rank: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if pca_rank <= 0:
        raise ValueError("--pca-rank must be positive.")
    if len(token_ids) < pca_rank + 1:
        raise ValueError(f"{label} PCA rank {pca_rank} requires at least {pca_rank + 1} token ids, got {len(token_ids)}.")
    token_id_tensor = torch.tensor(token_ids, dtype=torch.long)
    vectors = embedding_weight.index_select(0, token_id_tensor).float()
    mean_direction = _normalize_vector(vectors.mean(dim=0), label=f"{label}.mean_direction")
    centered = vectors - vectors.mean(dim=0, keepdim=True)
    rank = int(torch.linalg.matrix_rank(centered).item())
    if rank < pca_rank:
        raise RuntimeError(f"{label} token embeddings have rank {rank}, below requested PCA rank {pca_rank}.")
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[:pca_rank, :].T.contiguous()
    return mean_direction, basis, {
        "group": label,
        "num_tokens": len(token_ids),
        "ambient_dim": int(embedding_weight.size(1)),
        "pca_rank": pca_rank,
        "centered_rank": rank,
        "pca_singular_values": [float(value) for value in singular_values[:pca_rank].tolist()],
    }


def _top_token_alignments(
    *,
    direction: torch.Tensor,
    embedding_weight: torch.Tensor,
    vocab: Vocabulary,
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        raise ValueError("--top-k-tokens must be positive.")
    direction_unit = _normalize_vector(direction.float(), label="top_token_direction")
    embedding_norms = embedding_weight.float().norm(dim=1)
    if bool((embedding_norms <= 0.0).any().item()):
        bad_ids = torch.nonzero(embedding_norms <= 0.0, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"Cannot compute token cosine with zero-norm embedding rows: {bad_ids}")
    normalized_embeddings = embedding_weight.float() / embedding_norms[:, None]
    scores = normalized_embeddings.matmul(direction_unit)
    top_values, top_indices = torch.topk(scores.abs(), k=min(top_k, scores.numel()))
    rows: list[dict[str, Any]] = []
    for abs_score, token_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
        token_id = int(token_index)
        token = vocab.tokens[token_id]
        if token in vocab.key_tokens:
            token_group = "key"
        elif token in vocab.value_tokens:
            token_group = "value"
        else:
            token_group = "other"
        rows.append(
            {
                "token_id": token_id,
                "token": token,
                "token_group": token_group,
                "signed_cosine": float(scores[token_id].item()),
                "abs_cosine": float(abs_score),
            }
        )
    return rows


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


def _plot_rank1_trajectory(*, rows: list[dict[str, Any]], output_path: Path, include_margin: bool) -> None:
    rank1_rows = sorted([row for row in rows if int(row["singular_rank"]) == 1], key=lambda row: int(row["checkpoint_step"]))
    if not rank1_rows:
        raise RuntimeError("No rank-1 rows available for trajectory plot.")
    steps = [int(row["checkpoint_step"]) for row in rank1_rows]
    series: list[tuple[str, list[float]]] = [
        ("W_QK singular value", [float(row["singular_value"]) for row in rank1_rows]),
        ("left key mean cosine", [float(row["left_key_cosine"]) for row in rank1_rows]),
        ("right key mean cosine", [float(row["right_key_cosine"]) for row in rank1_rows]),
    ]
    if include_margin:
        series.append(("answer margin", [float(row["answer_margin"]) for row in rank1_rows]))
    normalized_series = [(label, _minmax(values, label=label)) for label, values in series]
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, values in normalized_series:
        ax.plot(steps, values, marker="o", linewidth=2, label=label)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("min-max normalized value")
    ax.set_title("L2H1 W_QK semantic alignment trajectory")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _write_markdown_report(path: Path, report: dict[str, Any], rank1_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# SVD Task Alignment",
        "",
        "This report measures whether QK singular directions align with task-token embedding subspaces.",
        "",
        "## Scope",
        "",
        f"- head: `L{report['head_layer']}H{report['head']}`",
        f"- checkpoints: `{report['num_checkpoints']}`",
        f"- singular ranks: `1..{report['top_ranks']}`",
        f"- PCA rank: `{report['pca_rank']}`",
        f"- behavior rows loaded: `{report['behavior_rows_loaded']}`",
        "",
        "## Outputs",
        "",
        f"- alignment rows JSONL: `{report['alignment_rows_path']}`",
        f"- alignment rows CSV: `{report['alignment_csv_path']}`",
        f"- subspace rows: `{report['subspace_rows_path']}`",
        f"- token alignment rows: `{report['token_alignment_rows_path']}`",
        f"- trajectory plot: `{report['trajectory_plot_path']}`",
        "",
        "## Rank-1 Trajectory",
        "",
        "| step | singular value | left key mean | right key mean | left key PCA | right key PCA | answer margin |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rank1_rows:
        answer_margin = row.get("answer_margin")
        lines.append(
            "| "
            f"{row['checkpoint_step']} | "
            f"{float(row['singular_value']):.4f} | "
            f"{float(row['left_key_cosine']):.4f} | "
            f"{float(row['right_key_cosine']):.4f} | "
            f"{float(row['left_key_pca_overlap']):.4f} | "
            f"{float(row['right_key_pca_overlap']):.4f} | "
            f"{'n/a' if answer_margin is None else f'{float(answer_margin):.4f}'} |"
        )
    lines.extend(
        [
            "",
            "## Correlations",
            "",
            "Correlations are Pearson correlations over the checkpoints in this run.",
            "",
        ]
    )
    for key, value in report["rank1_correlations"].items():
        lines.append(f"- `{key}`: `{'n/a' if value is None else f'{float(value):.4f}'}`")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This tool proves alignment between singular directions and embedding-token subspaces. It does not by itself prove causal route use; that still requires linking these rows to route scores and answer-margin movement.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def run_svd_task_alignment(
    *,
    config_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    checkpoint_paths: list[Path] | None = None,
    device_name: str = "cpu",
    head_layer: int = 2,
    head: int = 1,
    top_ranks: int = 4,
    pca_rank: int = 2,
    behavior_rows_path: Path | None = None,
    behavior_split: str = "__all__",
    behavior_margin_field: str = "baseline_margin_mean",
    behavior_accuracy_field: str = "baseline_accuracy",
    top_k_tokens: int = 8,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if top_ranks <= 0:
        raise ValueError("--top-ranks must be positive.")
    if head_layer < 0:
        raise ValueError("--head-layer must be non-negative.")
    if head < 0:
        raise ValueError("--head must be non-negative.")
    _prepare_output_dir(output_dir, overwrite=overwrite)
    device = require_device(device_name)
    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    model = build_model(spec.model, len(vocab.tokens), device)
    model.eval()
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    behavior_by_step = _load_behavior_by_step(
        behavior_rows_path=behavior_rows_path,
        behavior_split=behavior_split,
        margin_field=behavior_margin_field,
        accuracy_field=behavior_accuracy_field,
    )

    key_token_ids = [vocab.token_to_id[token] for token in vocab.key_tokens]
    value_token_ids = [vocab.token_to_id[token] for token in vocab.value_tokens]
    task_token_ids = set(key_token_ids) | set(value_token_ids)
    other_token_ids = [index for index in range(len(vocab.tokens)) if index not in task_token_ids]
    if not other_token_ids:
        raise RuntimeError("No other tokens available after excluding key and value tokens.")

    alignment_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    token_alignment_rows: list[dict[str, Any]] = []
    print(
        "[svd-task-alignment] "
        f"checkpoints={len(checkpoints)} head=L{head_layer}H{head} top_ranks={top_ranks} "
        f"pca_rank={pca_rank} behavior_rows={behavior_rows_path is not None}",
        flush=True,
    )
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        print(f"[svd-task-alignment] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}", flush=True)
        checkpoint = load_checkpoint(checkpoint_path, device)
        if "step" not in checkpoint:
            raise KeyError(f"Checkpoint payload is missing required key 'step': {checkpoint_path}")
        checkpoint_step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if checkpoint_step != path_step:
            raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={checkpoint_step} path={path_step}")
        if behavior_rows_path is not None and checkpoint_step not in behavior_by_step:
            raise RuntimeError(
                f"Behavior rows for split={behavior_split} do not contain required checkpoint step {checkpoint_step}"
            )
        load_model_state(model, checkpoint["model_state"])
        if head_layer >= len(model.blocks):
            raise ValueError(f"Requested head layer {head_layer}, but model has {len(model.blocks)} layers.")
        block = model.blocks[head_layer]
        if head >= int(block.attn.n_heads):
            raise ValueError(f"Requested head {head}, but layer {head_layer} has {block.attn.n_heads} heads.")
        embedding_weight = model.token_embedding.weight.detach().float().cpu()
        key_mean, key_basis, key_summary = _token_subspace(
            embedding_weight=embedding_weight,
            token_ids=key_token_ids,
            label="key",
            pca_rank=pca_rank,
        )
        value_mean, value_basis, value_summary = _token_subspace(
            embedding_weight=embedding_weight,
            token_ids=value_token_ids,
            label="value",
            pca_rank=pca_rank,
        )
        other_mean, other_basis, other_summary = _token_subspace(
            embedding_weight=embedding_weight,
            token_ids=other_token_ids,
            label="other",
            pca_rank=pca_rank,
        )
        for summary in [key_summary, value_summary, other_summary]:
            subspace_rows.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_step": checkpoint_step,
                    **summary,
                }
            )

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
            row = {
                "checkpoint": str(checkpoint_path),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_step": checkpoint_step,
                "head_layer": head_layer,
                "head": head,
                "singular_rank": rank_index,
                "singular_value": float(singular_values[rank_index - 1].item()),
                "left_key_cosine": _abs_cosine(left, key_mean, label="left.key_mean"),
                "left_key_pca_overlap": _subspace_overlap(left, key_basis, label="left.key_pca"),
                "left_value_cosine": _abs_cosine(left, value_mean, label="left.value_mean"),
                "left_value_pca_overlap": _subspace_overlap(left, value_basis, label="left.value_pca"),
                "left_other_cosine": _abs_cosine(left, other_mean, label="left.other_mean"),
                "left_other_pca_overlap": _subspace_overlap(left, other_basis, label="left.other_pca"),
                "right_key_cosine": _abs_cosine(right, key_mean, label="right.key_mean"),
                "right_key_pca_overlap": _subspace_overlap(right, key_basis, label="right.key_pca"),
                "right_value_cosine": _abs_cosine(right, value_mean, label="right.value_mean"),
                "right_value_pca_overlap": _subspace_overlap(right, value_basis, label="right.value_pca"),
                "right_other_cosine": _abs_cosine(right, other_mean, label="right.other_mean"),
                "right_other_pca_overlap": _subspace_overlap(right, other_basis, label="right.other_pca"),
            }
            row["left_key_minus_value_pca_overlap"] = float(row["left_key_pca_overlap"]) - float(row["left_value_pca_overlap"])
            row["right_key_minus_value_pca_overlap"] = float(row["right_key_pca_overlap"]) - float(row["right_value_pca_overlap"])
            if behavior is not None:
                row["answer_margin"] = behavior["answer_margin"]
                row["answer_accuracy"] = behavior.get("answer_accuracy")
            else:
                row["answer_margin"] = None
                row["answer_accuracy"] = None
            alignment_rows.append(row)
            for side, direction in [("left", left), ("right", right)]:
                top_token_rows = _top_token_alignments(
                    direction=direction,
                    embedding_weight=embedding_weight,
                    vocab=vocab,
                    top_k=top_k_tokens,
                )
                for token_rank, token_row in enumerate(top_token_rows, start=1):
                    token_alignment_rows.append(
                        {
                            "checkpoint": str(checkpoint_path),
                            "checkpoint_name": checkpoint_path.name,
                            "checkpoint_step": checkpoint_step,
                            "head_layer": head_layer,
                            "head": head,
                            "singular_rank": rank_index,
                            "vector_side": side,
                            "token_rank": token_rank,
                            **token_row,
                        }
                    )
        print(
            "[svd-task-alignment] finished "
            f"step={checkpoint_step} sv1={float(singular_values[0].item()):.6g}",
            flush=True,
        )

    alignment_rows.sort(key=lambda row: (int(row["checkpoint_step"]), int(row["singular_rank"])))
    subspace_rows.sort(key=lambda row: (int(row["checkpoint_step"]), str(row["group"])))
    token_alignment_rows.sort(
        key=lambda row: (
            int(row["checkpoint_step"]),
            int(row["singular_rank"]),
            str(row["vector_side"]),
            int(row["token_rank"]),
        )
    )
    rank1_rows = sorted([row for row in alignment_rows if int(row["singular_rank"]) == 1], key=lambda row: int(row["checkpoint_step"]))
    if not rank1_rows:
        raise RuntimeError("No rank-1 alignment rows were produced.")
    include_margin = behavior_rows_path is not None
    rank1_correlations: dict[str, float | None] = {
        "singular_value_vs_left_key_cosine": _pearson(
            [float(row["singular_value"]) for row in rank1_rows],
            [float(row["left_key_cosine"]) for row in rank1_rows],
        ),
        "singular_value_vs_right_key_cosine": _pearson(
            [float(row["singular_value"]) for row in rank1_rows],
            [float(row["right_key_cosine"]) for row in rank1_rows],
        ),
        "singular_value_vs_left_key_pca_overlap": _pearson(
            [float(row["singular_value"]) for row in rank1_rows],
            [float(row["left_key_pca_overlap"]) for row in rank1_rows],
        ),
        "singular_value_vs_right_key_pca_overlap": _pearson(
            [float(row["singular_value"]) for row in rank1_rows],
            [float(row["right_key_pca_overlap"]) for row in rank1_rows],
        ),
        "left_key_pca_overlap_vs_right_key_pca_overlap": _pearson(
            [float(row["left_key_pca_overlap"]) for row in rank1_rows],
            [float(row["right_key_pca_overlap"]) for row in rank1_rows],
        ),
    }
    if include_margin:
        rank1_correlations.update(
            {
                "singular_value_vs_answer_margin": _pearson(
                    [float(row["singular_value"]) for row in rank1_rows],
                    [float(row["answer_margin"]) for row in rank1_rows],
                ),
                "left_key_cosine_vs_answer_margin": _pearson(
                    [float(row["left_key_cosine"]) for row in rank1_rows],
                    [float(row["answer_margin"]) for row in rank1_rows],
                ),
                "right_key_cosine_vs_answer_margin": _pearson(
                    [float(row["right_key_cosine"]) for row in rank1_rows],
                    [float(row["answer_margin"]) for row in rank1_rows],
                ),
                "left_key_pca_overlap_vs_answer_margin": _pearson(
                    [float(row["left_key_pca_overlap"]) for row in rank1_rows],
                    [float(row["answer_margin"]) for row in rank1_rows],
                ),
                "right_key_pca_overlap_vs_answer_margin": _pearson(
                    [float(row["right_key_pca_overlap"]) for row in rank1_rows],
                    [float(row["answer_margin"]) for row in rank1_rows],
                ),
            }
        )

    report_path = output_dir / "svd_task_alignment_report.json"
    markdown_path = output_dir / "svd_task_alignment_report.md"
    alignment_rows_path = output_dir / "svd_task_alignment_rows.jsonl"
    alignment_csv_path = output_dir / "svd_task_alignment_rows.csv"
    subspace_rows_path = output_dir / "svd_task_subspace_rows.jsonl"
    token_alignment_rows_path = output_dir / "svd_task_token_alignment_rows.jsonl"
    trajectory_plot_path = output_dir / "svd_task_alignment_trajectory.svg"

    write_jsonl(alignment_rows_path, alignment_rows)
    _write_csv(
        alignment_csv_path,
        alignment_rows,
        fieldnames=[
            "checkpoint",
            "checkpoint_name",
            "checkpoint_step",
            "head_layer",
            "head",
            "singular_rank",
            "singular_value",
            "left_key_cosine",
            "left_key_pca_overlap",
            "left_value_cosine",
            "left_value_pca_overlap",
            "left_other_cosine",
            "left_other_pca_overlap",
            "left_key_minus_value_pca_overlap",
            "right_key_cosine",
            "right_key_pca_overlap",
            "right_value_cosine",
            "right_value_pca_overlap",
            "right_other_cosine",
            "right_other_pca_overlap",
            "right_key_minus_value_pca_overlap",
            "answer_margin",
            "answer_accuracy",
        ],
    )
    write_jsonl(subspace_rows_path, subspace_rows)
    write_jsonl(token_alignment_rows_path, token_alignment_rows)
    _plot_rank1_trajectory(rows=alignment_rows, output_path=trajectory_plot_path, include_margin=include_margin)

    report = {
        "config_path": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "output_dir": str(output_dir),
        "device": device_name,
        "head_layer": head_layer,
        "head": head,
        "top_ranks": top_ranks,
        "pca_rank": pca_rank,
        "behavior_rows_path": None if behavior_rows_path is None else str(behavior_rows_path),
        "behavior_split": behavior_split,
        "behavior_margin_field": behavior_margin_field,
        "behavior_accuracy_field": behavior_accuracy_field,
        "behavior_rows_loaded": behavior_rows_path is not None,
        "num_checkpoints": len(checkpoints),
        "num_alignment_rows": len(alignment_rows),
        "num_subspace_rows": len(subspace_rows),
        "num_token_alignment_rows": len(token_alignment_rows),
        "vocab": {
            "num_tokens": len(vocab.tokens),
            "num_key_tokens": len(vocab.key_tokens),
            "num_value_tokens": len(vocab.value_tokens),
            "num_other_tokens": len(other_token_ids),
            "key_tokens": vocab.key_tokens,
            "value_tokens": vocab.value_tokens,
            "other_tokens": [vocab.tokens[token_id] for token_id in other_token_ids],
        },
        "rank1_correlations": rank1_correlations,
        "rank1_start": rank1_rows[0],
        "rank1_end": rank1_rows[-1],
        "alignment_rows_path": str(alignment_rows_path),
        "alignment_csv_path": str(alignment_csv_path),
        "subspace_rows_path": str(subspace_rows_path),
        "token_alignment_rows_path": str(token_alignment_rows_path),
        "trajectory_plot_path": str(trajectory_plot_path),
        "matrix_convention": "W_QK = q_rows.T @ k_rows; left singular vectors are query residual directions, right singular vectors are key residual directions.",
        "plot_convention": "The trajectory SVG min-max normalizes each plotted series to put singular value, alignment, and answer margin on one axis.",
    }
    write_json(report_path, report)
    _write_markdown_report(markdown_path, report, rank1_rows=rank1_rows)
    print(f"[svd-task-alignment] complete report={report_path} rows={alignment_rows_path}", flush=True)
    return (
        report_path,
        markdown_path,
        alignment_rows_path,
        alignment_csv_path,
        subspace_rows_path,
        token_alignment_rows_path,
        {"trajectory": trajectory_plot_path},
    )
