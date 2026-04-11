from __future__ import annotations

import hashlib
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from circuit.analysis.feature_analysis import (
    SparseAutoencoder,
    StageFeatureDataset,
    _encode_features,
    _make_probe_loader,
    collect_stage_feature_dataset,
    fit_sparse_autoencoder,
    summarize_features,
)
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import read_symbolic_kv_stream_metadata
from circuit.io import ensure_parent_dir, iter_jsonl, read_json, write_json, write_jsonl
from circuit.runtime import (
    _migrate_legacy_feedforward_state_dict,
    build_model,
    load_checkpoint,
    load_model_state,
    move_batch_to_device,
)
from circuit.vocab import Vocabulary

DEFAULT_FAMILY_CLUSTER_METRICS = (
    "mean_activation",
    "active_fraction",
    "correctness_gap",
    "heldout_gap",
    "structural_ood_gap",
)


def _resolve_checkpoints(
    *,
    checkpoint_paths: list[Path] | None,
    checkpoint_dir: Path | None,
) -> list[Path]:
    explicit = [] if checkpoint_paths is None else list(checkpoint_paths)
    if explicit and checkpoint_dir is not None:
        raise ValueError("Provide either checkpoint_paths or checkpoint_dir, not both.")
    if explicit:
        missing = [path for path in explicit if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoints: {[str(path) for path in missing]}")
        return sorted(explicit)
    if checkpoint_dir is None:
        raise ValueError("One of checkpoint_paths or checkpoint_dir must be provided.")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints


def _load_probe_batches(
    *,
    config_path: Path,
    probe_set_path: Path,
    device_name: str,
) -> tuple[TrainSpec, Vocabulary, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], torch.device]:
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    device = torch.device(device_name)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    probe_loader = _make_probe_loader(
        probe_records=probe_records,
        batch_size=spec.evaluation.batch_size,
        pad_token_id=vocab.pad_token_id,
    )
    analysis_batches = [move_batch_to_device(batch, device) for batch in probe_loader]
    if not analysis_batches:
        raise ValueError("Probe loader produced no analysis batches.")
    return spec, vocab, analysis_batches, probe_records, probe_metadata, device


def _merge_stage_feature_datasets(datasets: list[StageFeatureDataset]) -> StageFeatureDataset:
    if not datasets:
        raise ValueError("datasets must not be empty.")
    return StageFeatureDataset(
        activations=torch.cat([dataset.activations for dataset in datasets], dim=0),
        correct=torch.cat([dataset.correct for dataset in datasets], dim=0),
        stage_margin=torch.cat([dataset.stage_margin for dataset in datasets], dim=0),
        answer_direction=torch.cat([dataset.answer_direction for dataset in datasets], dim=0),
        splits=[split for dataset in datasets for split in dataset.splits],
        sample_ids=[sample_id for dataset in datasets for sample_id in dataset.sample_ids],
        query_indices=[query_index for dataset in datasets for query_index in dataset.query_indices],
        answer_tokens=[answer_token for dataset in datasets for answer_token in dataset.answer_tokens],
    )


def _compute_normalization(activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if activations.ndim != 2:
        raise ValueError(f"Expected activations to have shape [n, d], got {tuple(activations.shape)}")
    mean = activations.mean(dim=0)
    std = activations.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def _normalize_activations(
    activations: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return (activations - mean) / std


def _denormalize_activations(
    activations: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return activations * std + mean


def _make_basis_id(
    *,
    stage_name: str,
    checkpoint_steps: list[int],
    probe_set_path: Path,
    num_features: int,
    input_dim: int,
) -> str:
    raw = "|".join(
        [
            stage_name,
            str(probe_set_path),
            ",".join(str(step) for step in checkpoint_steps),
            str(num_features),
            str(input_dim),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _load_shared_basis(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Shared feature basis not found: {path}")
    payload = torch.load(path, map_location=device)
    required = [
        "basis_id",
        "stage_name",
        "probe_set_path",
        "num_features",
        "input_dim",
        "normalization_mean",
        "normalization_std",
        "sae_state",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Shared feature basis is missing required keys: {missing}")
    sae = SparseAutoencoder(int(payload["input_dim"]), int(payload["num_features"])).to(device)
    sae.load_state_dict(payload["sae_state"])
    sae.eval()
    return {
        **payload,
        "sae": sae,
        "normalization_mean": payload["normalization_mean"].to(device),
        "normalization_std": payload["normalization_std"].to(device),
    }


def _compute_feature_statistics(
    *,
    dataset: StageFeatureDataset,
    feature_activations: torch.Tensor,
    decoder_vectors: torch.Tensor,
) -> list[dict[str, Any]]:
    if feature_activations.ndim != 2:
        raise ValueError("feature_activations must have rank 2.")
    if decoder_vectors.shape != (feature_activations.size(1), dataset.activations.size(1)):
        raise ValueError(
            "decoder_vectors shape does not match feature_activations/input_dim: "
            f"{tuple(decoder_vectors.shape)} vs {(feature_activations.size(1), dataset.activations.size(1))}"
        )
    mean_answer_direction = F.normalize(dataset.answer_direction.mean(dim=0), dim=0)
    correct_mask = dataset.correct.bool()
    heldout_mask = torch.tensor([split == "heldout_pairs" for split in dataset.splits], device=feature_activations.device)
    structural_ood_mask = torch.tensor(
        [split == "structural_ood" for split in dataset.splits],
        device=feature_activations.device,
    )

    rows: list[dict[str, Any]] = []
    for feature_index in range(feature_activations.size(1)):
        values = feature_activations[:, feature_index]
        centered_values = values - values.mean()
        centered_margin = dataset.stage_margin - dataset.stage_margin.mean()
        margin_correlation = float(
            F.cosine_similarity(centered_values.unsqueeze(0), centered_margin.unsqueeze(0)).item()
        )
        alignment = float(
            F.cosine_similarity(
                F.normalize(decoder_vectors[feature_index].unsqueeze(0), dim=-1),
                mean_answer_direction.unsqueeze(0),
            ).item()
        )
        rows.append(
            {
                "feature_id": feature_index,
                "mean_activation": float(values.mean().item()),
                "active_fraction": float((values > 1e-4).float().mean().item()),
                "correctness_gap": float(
                    (values[correct_mask].mean() - values[~correct_mask].mean()).item()
                ) if correct_mask.any() and (~correct_mask).any() else 0.0,
                "heldout_gap": float(
                    (values[heldout_mask].mean() - values[~heldout_mask].mean()).item()
                ) if heldout_mask.any() and (~heldout_mask).any() else 0.0,
                "structural_ood_gap": float(
                    (values[structural_ood_mask].mean() - values[~structural_ood_mask].mean()).item()
                ) if structural_ood_mask.any() and (~structural_ood_mask).any() else 0.0,
                "margin_correlation": margin_correlation,
                "answer_direction_alignment": alignment,
                "split_mean_activation": {
                    split_name: float(
                        values[[index for index, item in enumerate(dataset.splits) if item == split_name]].mean().item()
                    )
                    for split_name in sorted(set(dataset.splits))
                },
            }
        )
    return rows


def _safe_topk_feature_ids(
    feature_rows: list[dict[str, Any]],
    *,
    key: str,
    top_k: int,
    absolute: bool = False,
) -> list[int]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    ranked = sorted(
        feature_rows,
        key=lambda row: abs(float(row[key])) if absolute else float(row[key]),
        reverse=True,
    )
    return [int(row["feature_id"]) for row in ranked[:top_k]]


def _import_matplotlib() -> tuple[Any, Any]:
    cache_root = Path(tempfile.gettempdir()) / "circuit-mpl-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_root)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _render_feature_trajectory_plots(
    *,
    trajectory_rows: list[dict[str, Any]],
    output_dir: Path,
    stage_name: str,
    top_k: int = 8,
) -> dict[str, str]:
    if not trajectory_rows:
        raise ValueError("trajectory_rows must not be empty.")
    _, plt = _import_matplotlib()
    ensure_parent_dir(output_dir / "placeholder")

    by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        by_feature[int(row["feature_id"])].append(row)
    for rows in by_feature.values():
        rows.sort(key=lambda item: int(item["step"]))

    ranked_ids = sorted(
        by_feature,
        key=lambda feature_id: max(float(item["heldout_gap"]) for item in by_feature[feature_id]),
        reverse=True,
    )[:top_k]

    trajectory_plot_path = output_dir / f"{stage_name}_feature_trajectory_topk.svg"
    fig, ax = plt.subplots(figsize=(9, 5))
    for feature_id in ranked_ids:
        xs = [int(item["step"]) for item in by_feature[feature_id]]
        ys = [float(item["mean_activation"]) for item in by_feature[feature_id]]
        ax.plot(xs, ys, label=f"F{feature_id}")
    ax.set_title(f"{stage_name} top feature trajectories")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean activation")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(trajectory_plot_path)
    plt.close(fig)

    sorted_steps = sorted({int(row["step"]) for row in trajectory_rows})
    sorted_features = sorted(by_feature)
    heatmap = torch.empty((len(sorted_features), len(sorted_steps)), dtype=torch.float32)
    for feature_index, feature_id in enumerate(sorted_features):
        values_by_step = {int(item["step"]): float(item["mean_activation"]) for item in by_feature[feature_id]}
        for step_index, step in enumerate(sorted_steps):
            heatmap[feature_index, step_index] = values_by_step[step]
    heatmap_plot_path = output_dir / f"{stage_name}_feature_heatmap.svg"
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(heatmap.numpy(), aspect="auto", interpolation="nearest")
    ax.set_title(f"{stage_name} feature heatmap")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Feature id")
    ax.set_xticks(range(len(sorted_steps)))
    ax.set_xticklabels(sorted_steps, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(heatmap_plot_path)
    plt.close(fig)

    return {
        "trajectory_plot": str(trajectory_plot_path),
        "heatmap_plot": str(heatmap_plot_path),
    }


def _render_birth_plot(
    *,
    birth_rows: list[dict[str, Any]],
    output_path: Path,
    entity_id_key: str = "feature_id",
    title: str = "Feature birth raster",
    ylabel: str = "Entity rows",
) -> Path:
    _, plt = _import_matplotlib()
    metric_order = ["mean_activation", "active_fraction", "correctness_gap", "heldout_gap"]
    y_positions: list[float] = []
    x_positions: list[float] = []
    colors: list[str] = []
    metric_to_color = {
        "mean_activation": "#1f77b4",
        "active_fraction": "#ff7f0e",
        "correctness_gap": "#2ca02c",
        "heldout_gap": "#d62728",
    }
    sorted_rows = sorted(
        birth_rows,
        key=lambda row: (
            int(row["birth_step"]) if row["birth_step"] is not None else 10**9,
            int(row[entity_id_key]),
        ),
    )
    for index, row in enumerate(sorted_rows):
        for metric_index, metric_name in enumerate(metric_order):
            metric_birth = row["births"].get(metric_name)
            if metric_birth is None:
                continue
            y_positions.append(index + metric_index * 0.12)
            x_positions.append(float(metric_birth["birth_step"]))
            colors.append(metric_to_color[metric_name])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_positions, y_positions, c=colors, s=24)
    ax.set_title(title)
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_compare_plot(
    *,
    diff_rows: list[dict[str, Any]],
    output_path: Path,
    top_k: int = 12,
) -> Path:
    _, plt = _import_matplotlib()
    ranked = sorted(diff_rows, key=lambda row: abs(float(row["mean_activation_delta"])), reverse=True)[:top_k]
    labels = [f"F{row['feature_id']}" for row in ranked]
    values = [float(row["mean_activation_delta"]) for row in ranked]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title("Feature mean-activation deltas")
    ax.set_ylabel("Target - source")
    ax.set_xlabel("Feature")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def shared_feature_fit(
    *,
    config_path: Path,
    probe_set_path: Path,
    stage_name: str,
    output_dir: Path,
    checkpoint_paths: list[Path] | None = None,
    checkpoint_dir: Path | None = None,
    device_name: str = "cpu",
    num_features: int = 64,
    train_steps: int = 400,
    learning_rate: float = 1e-3,
    l1_coefficient: float = 1e-3,
    batch_size: int = 256,
) -> tuple[Path, Path, Path]:
    checkpoints = _resolve_checkpoints(checkpoint_paths=checkpoint_paths, checkpoint_dir=checkpoint_dir)
    spec, vocab, analysis_batches, _, probe_metadata, device = _load_probe_batches(
        config_path=config_path,
        probe_set_path=probe_set_path,
        device_name=device_name,
    )

    datasets: list[StageFeatureDataset] = []
    checkpoint_steps: list[int] = []
    checkpoint_payloads: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        checkpoint = load_checkpoint(checkpoint_path, device)
        model = build_model(spec.model, len(vocab.tokens), device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        dataset = collect_stage_feature_dataset(
            model=model,
            batches=analysis_batches,
            vocab=vocab,
            stage_name=stage_name,
        )
        datasets.append(dataset)
        checkpoint_steps.append(int(checkpoint["step"]))
        checkpoint_payloads.append({"step": int(checkpoint["step"]), "checkpoint_path": str(checkpoint_path)})

    pooled_dataset = _merge_stage_feature_datasets(datasets)
    normalization_mean, normalization_std = _compute_normalization(pooled_dataset.activations)
    normalized_activations = _normalize_activations(pooled_dataset.activations, normalization_mean, normalization_std)
    sae, fit_metrics = fit_sparse_autoencoder(
        activations=normalized_activations,
        num_features=num_features,
        train_steps=train_steps,
        learning_rate=learning_rate,
        l1_coefficient=l1_coefficient,
        batch_size=batch_size,
    )
    pooled_features = _encode_features(sae, normalized_activations)
    feature_rows, ranked = summarize_features(
        sae=sae,
        target_dataset=pooled_dataset,
        target_feature_activations=pooled_features,
        source_dataset=None,
        source_feature_activations=None,
        top_k_examples=4,
    )

    basis_id = _make_basis_id(
        stage_name=stage_name,
        checkpoint_steps=checkpoint_steps,
        probe_set_path=probe_set_path,
        num_features=num_features,
        input_dim=int(pooled_dataset.activations.size(1)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    basis_path = output_dir / "shared_feature_basis.pt"
    manifest_path = output_dir / "shared_feature_basis.json"
    feature_summary_path = output_dir / "shared_feature_basis_features.json"

    torch.save(
        {
            "basis_id": basis_id,
            "stage_name": stage_name,
            "probe_set_path": str(probe_set_path),
            "checkpoint_steps_used_for_fit": checkpoint_steps,
            "checkpoint_paths_used_for_fit": [str(path) for path in checkpoints],
            "num_features": num_features,
            "input_dim": int(pooled_dataset.activations.size(1)),
            "normalization_mean": normalization_mean.detach().cpu(),
            "normalization_std": normalization_std.detach().cpu(),
            "sae_state": sae.state_dict(),
            "fit_hyperparameters": {
                "train_steps": train_steps,
                "learning_rate": learning_rate,
                "l1_coefficient": l1_coefficient,
                "batch_size": batch_size,
            },
            "fit_metrics": fit_metrics,
        },
        basis_path,
    )
    write_json(
        manifest_path,
        {
            "basis_id": basis_id,
            "stage_name": stage_name,
            "probe_set_path": str(probe_set_path),
            "probe_set_benchmark_dir": str(probe_metadata["benchmark_dir"]),
            "checkpoint_steps_used_for_fit": checkpoint_steps,
            "checkpoints_used_for_fit": checkpoint_payloads,
            "num_features": num_features,
            "input_dim": int(pooled_dataset.activations.size(1)),
            "fit_hyperparameters": {
                "train_steps": train_steps,
                "learning_rate": learning_rate,
                "l1_coefficient": l1_coefficient,
                "batch_size": batch_size,
            },
            "fit_metrics": fit_metrics,
        },
    )
    write_json(
        feature_summary_path,
        {
            "basis_id": basis_id,
            "stage_name": stage_name,
            "top_features": {key: value[:12] for key, value in ranked.items()},
            "feature_rows": feature_rows,
        },
    )
    return basis_path, manifest_path, feature_summary_path


def feature_trajectory_sweep(
    *,
    config_path: Path,
    probe_set_path: Path,
    basis_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    device_name: str = "cpu",
) -> tuple[Path, Path, Path, dict[str, Path]]:
    checkpoints = _resolve_checkpoints(checkpoint_paths=None, checkpoint_dir=checkpoint_dir)
    spec, vocab, analysis_batches, _, _, device = _load_probe_batches(
        config_path=config_path,
        probe_set_path=probe_set_path,
        device_name=device_name,
    )
    basis = _load_shared_basis(basis_path, device)
    stage_name = str(basis["stage_name"])
    normalization_mean = basis["normalization_mean"]
    normalization_std = basis["normalization_std"]
    sae: SparseAutoencoder = basis["sae"]
    decoder_vectors = sae.decoder.weight.detach().transpose(0, 1)

    trajectory_rows: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    split_profiles: dict[int, dict[str, dict[str, float]]] = {}

    for checkpoint_path in checkpoints:
        checkpoint = load_checkpoint(checkpoint_path, device)
        model = build_model(spec.model, len(vocab.tokens), device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        dataset = collect_stage_feature_dataset(
            model=model,
            batches=analysis_batches,
            vocab=vocab,
            stage_name=stage_name,
        )
        normalized = _normalize_activations(dataset.activations, normalization_mean, normalization_std)
        feature_activations = _encode_features(sae, normalized)
        feature_rows = _compute_feature_statistics(
            dataset=dataset,
            feature_activations=feature_activations,
            decoder_vectors=decoder_vectors,
        )
        current_step = int(checkpoint["step"])
        split_profiles[current_step] = {
            str(row["feature_id"]): {key: float(value) for key, value in row["split_mean_activation"].items()}
            for row in feature_rows
        }
        top_mean_features = _safe_topk_feature_ids(feature_rows, key="mean_activation", top_k=min(8, len(feature_rows)))
        top_heldout_features = _safe_topk_feature_ids(feature_rows, key="heldout_gap", top_k=min(8, len(feature_rows)))
        checkpoint_summaries.append(
            {
                "step": current_step,
                "checkpoint_path": str(checkpoint_path),
                "top_features_by_mean_activation": top_mean_features,
                "top_features_by_heldout_gap": top_heldout_features,
            }
        )
        for row in feature_rows:
            trajectory_rows.append(
                {
                    "basis_id": str(basis["basis_id"]),
                    "stage_name": stage_name,
                    "checkpoint_path": str(checkpoint_path),
                    "step": current_step,
                    **row,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_path = output_dir / "feature_trajectories.jsonl"
    summary_path = output_dir / "feature_checkpoint_summary.json"
    split_profiles_path = output_dir / "feature_split_profiles.json"
    write_jsonl(trajectories_path, trajectory_rows)
    write_json(
        summary_path,
        {
            "basis_id": str(basis["basis_id"]),
            "stage_name": stage_name,
            "num_rows": len(trajectory_rows),
            "num_checkpoints": len(checkpoints),
            "checkpoint_summaries": checkpoint_summaries,
        },
    )
    write_json(
        split_profiles_path,
        {
            "basis_id": str(basis["basis_id"]),
            "stage_name": stage_name,
            "split_profiles": split_profiles,
        },
    )
    plots_dir = output_dir / "plots"
    plot_paths = _render_feature_trajectory_plots(
        trajectory_rows=trajectory_rows,
        output_dir=plots_dir,
        stage_name=stage_name,
    )
    return trajectories_path, summary_path, split_profiles_path, {key: Path(value) for key, value in plot_paths.items()}


def _load_feature_trajectory_tables(
    *,
    trajectories_path: Path,
    metrics: list[str],
) -> tuple[list[dict[str, Any]], list[int], list[int], dict[str, torch.Tensor], dict[int, list[dict[str, Any]]]]:
    rows = list(iter_jsonl(trajectories_path))
    if not rows:
        raise ValueError(f"No rows found in trajectories file: {trajectories_path}")
    if not metrics:
        raise ValueError("metrics must not be empty.")

    rows_by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_feature[int(row["feature_id"])].append(row)
    for feature_rows in rows_by_feature.values():
        feature_rows.sort(key=lambda item: int(item["step"]))

    feature_ids = sorted(rows_by_feature)
    sorted_steps = sorted({int(row["step"]) for row in rows})
    if not sorted_steps:
        raise ValueError(f"No checkpoint steps found in trajectories file: {trajectories_path}")

    for metric_name in metrics:
        if metric_name not in rows[0]:
            raise KeyError(f"Metric {metric_name} not found in trajectory rows.")

    metric_tables: dict[str, torch.Tensor] = {}
    for metric_name in metrics:
        table = torch.empty((len(feature_ids), len(sorted_steps)), dtype=torch.float32)
        for feature_index, feature_id in enumerate(feature_ids):
            feature_rows = rows_by_feature[feature_id]
            steps_for_feature = [int(row["step"]) for row in feature_rows]
            if steps_for_feature != sorted_steps:
                raise ValueError(
                    f"Feature {feature_id} does not have a complete trajectory over all steps. "
                    f"Expected {sorted_steps}, got {steps_for_feature}."
                )
            for step_index, row in enumerate(feature_rows):
                table[feature_index, step_index] = float(row[metric_name])
        metric_tables[metric_name] = table
    return rows, feature_ids, sorted_steps, metric_tables, rows_by_feature


def _build_family_similarity_vectors(
    *,
    metric_tables: dict[str, torch.Tensor],
    metrics: list[str],
) -> torch.Tensor:
    if not metrics:
        raise ValueError("metrics must not be empty.")
    parts: list[torch.Tensor] = []
    for metric_name in metrics:
        table = metric_tables[metric_name]
        centered = table - table.mean()
        scale = table.std(unbiased=False).clamp_min(1e-6)
        normalized = centered / scale
        deltas = torch.zeros_like(table)
        if table.size(1) > 1:
            deltas[:, 1:] = table[:, 1:] - table[:, :-1]
        delta_centered = deltas - deltas.mean()
        delta_scale = deltas.std(unbiased=False).clamp_min(1e-6)
        normalized_delta = delta_centered / delta_scale
        parts.append(normalized)
        parts.append(normalized_delta)
    vectors = torch.cat([part.reshape(part.size(0), -1) for part in parts], dim=1)
    norms = vectors.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return vectors / norms


def _cluster_similarity_components(
    *,
    similarity: torch.Tensor,
    threshold: float,
) -> list[list[int]]:
    if similarity.ndim != 2 or similarity.size(0) != similarity.size(1):
        raise ValueError(f"similarity must be a square matrix, got {tuple(similarity.shape)}")
    if threshold < -1.0 or threshold > 1.0:
        raise ValueError(f"similarity threshold must be within [-1, 1], got {threshold}")
    clusters: list[list[int]] = [[index] for index in range(similarity.size(0))]
    while True:
        best_pair: tuple[int, int] | None = None
        best_score = float("-inf")
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                left = clusters[left_index]
                right = clusters[right_index]
                inter_similarity = similarity[left][:, right]
                linkage_score = float(inter_similarity.min().item())
                if linkage_score < threshold:
                    continue
                if linkage_score > best_score:
                    best_score = linkage_score
                    best_pair = (left_index, right_index)
        if best_pair is None:
            break
        left_index, right_index = best_pair
        merged = sorted(clusters[left_index] + clusters[right_index])
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
    clusters.sort(key=lambda indices: (-len(indices), indices[0]))
    return clusters


def _load_birth_rows_by_feature(feature_births_path: Path | None) -> dict[int, dict[str, Any]]:
    if feature_births_path is None:
        return {}
    payload = read_json(feature_births_path)
    feature_rows = payload.get("features")
    if not isinstance(feature_rows, list):
        raise ValueError(f"feature_births file must contain a 'features' list: {feature_births_path}")
    by_feature: dict[int, dict[str, Any]] = {}
    for row in feature_rows:
        feature_id = row.get("feature_id")
        if feature_id is None:
            raise KeyError(f"feature_births row missing feature_id: {row}")
        by_feature[int(feature_id)] = row
    return by_feature


def _compute_family_birth_summary(
    *,
    member_feature_ids: list[int],
    birth_rows_by_feature: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    if not birth_rows_by_feature:
        return None
    correctness_steps: list[int] = []
    heldout_steps: list[int] = []
    useful_steps: list[int] = []
    for feature_id in member_feature_ids:
        row = birth_rows_by_feature.get(feature_id)
        if row is None:
            continue
        births = row.get("births", {})
        final_metrics = row.get("final_metrics", {})
        correctness = births.get("correctness_gap")
        heldout = births.get("heldout_gap")
        if correctness is not None:
            correctness_steps.append(int(correctness["birth_step"]))
        if heldout is not None:
            heldout_steps.append(int(heldout["birth_step"]))
        if (
            correctness is not None
            and heldout is not None
            and float(final_metrics.get("correctness_gap", 0.0)) > 0.0
            and float(final_metrics.get("heldout_gap", 0.0)) > 0.0
        ):
            useful_steps.append(max(int(correctness["birth_step"]), int(heldout["birth_step"])))
    return {
        "num_members_with_correctness_birth": len(correctness_steps),
        "num_members_with_heldout_birth": len(heldout_steps),
        "num_members_with_useful_birth": len(useful_steps),
        "earliest_correctness_birth_step": min(correctness_steps) if correctness_steps else None,
        "earliest_heldout_birth_step": min(heldout_steps) if heldout_steps else None,
        "earliest_useful_birth_step": min(useful_steps) if useful_steps else None,
    }


def _render_family_similarity_heatmap(
    *,
    similarity: torch.Tensor,
    ordered_feature_ids: list[int],
    family_assignments_by_feature: dict[int, int],
    output_path: Path,
) -> Path:
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(similarity.numpy(), aspect="auto", interpolation="nearest", vmin=-1.0, vmax=1.0)
    ax.set_title("Feature-family similarity heatmap")
    ax.set_xlabel("Feature id")
    ax.set_ylabel("Feature id")
    labels = [f"F{feature_id}\nG{family_assignments_by_feature[feature_id]}" for feature_id in ordered_feature_ids]
    ax.set_xticks(range(len(ordered_feature_ids)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(ordered_feature_ids)))
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_family_trajectory_plot(
    *,
    family_trajectory_rows: list[dict[str, Any]],
    family_summaries: list[dict[str, Any]],
    metrics: list[str],
    output_path: Path,
    top_k_families: int,
) -> Path:
    if not family_trajectory_rows:
        raise ValueError("family_trajectory_rows must not be empty.")
    tracked_metrics = [metric for metric in ["heldout_gap", "correctness_gap", "mean_activation"] if metric in metrics]
    if not tracked_metrics:
        tracked_metrics = metrics[: min(3, len(metrics))]
    _, plt = _import_matplotlib()
    ranked_family_ids = [
        int(row["family_id"])
        for row in sorted(
            family_summaries,
            key=lambda row: (
                float(row["final_metric_means"].get("heldout_gap", 0.0))
                + float(row["final_metric_means"].get("correctness_gap", 0.0)),
                float(row["mean_pairwise_similarity"]),
                int(row["size"]),
            ),
            reverse=True,
        )[:top_k_families]
    ]
    by_family: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in family_trajectory_rows:
        by_family[int(row["family_id"])].append(row)
    for rows in by_family.values():
        rows.sort(key=lambda item: int(item["step"]))

    fig, axes = plt.subplots(len(tracked_metrics), 1, figsize=(10, 3.5 * len(tracked_metrics)), sharex=True)
    if len(tracked_metrics) == 1:
        axes = [axes]
    for axis, metric_name in zip(axes, tracked_metrics, strict=True):
        for family_id in ranked_family_ids:
            xs = [int(row["step"]) for row in by_family[family_id]]
            ys = [float(row[f"{metric_name}_mean"]) for row in by_family[family_id]]
            axis.plot(xs, ys, label=f"G{family_id}")
        axis.set_ylabel(metric_name)
        axis.legend(loc="best", fontsize=8)
    axes[0].set_title("Feature-family trajectories")
    axes[-1].set_xlabel("Checkpoint step")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def feature_family_cluster(
    *,
    trajectories_path: Path,
    output_dir: Path,
    metrics: list[str] | None = None,
    similarity_threshold: float = 0.85,
    feature_births_path: Path | None = None,
    top_k_families: int = 8,
) -> tuple[Path, Path, Path, dict[str, Path]]:
    selected_metrics = list(DEFAULT_FAMILY_CLUSTER_METRICS if metrics is None else metrics)
    rows, feature_ids, sorted_steps, metric_tables, rows_by_feature = _load_feature_trajectory_tables(
        trajectories_path=trajectories_path,
        metrics=selected_metrics,
    )
    vectors = _build_family_similarity_vectors(metric_tables=metric_tables, metrics=selected_metrics)
    similarity = vectors @ vectors.transpose(0, 1)
    components = _cluster_similarity_components(similarity=similarity, threshold=similarity_threshold)
    birth_rows_by_feature = _load_birth_rows_by_feature(feature_births_path)

    ordered_feature_ids: list[int] = []
    family_assignments_by_feature: dict[int, int] = {}
    family_summaries: list[dict[str, Any]] = []
    family_trajectory_rows: list[dict[str, Any]] = []
    feature_assignment_rows: list[dict[str, Any]] = []

    for family_id, member_indices in enumerate(components):
        member_feature_ids = [feature_ids[index] for index in member_indices]
        ordered_feature_ids.extend(member_feature_ids)
        for feature_id in member_feature_ids:
            family_assignments_by_feature[feature_id] = family_id

        family_similarity = similarity[member_indices][:, member_indices]
        if len(member_indices) == 1:
            mean_pairwise_similarity = 1.0
            min_pairwise_similarity = 1.0
            representative_index = member_indices[0]
        else:
            mask = ~torch.eye(len(member_indices), dtype=torch.bool)
            pairwise_values = family_similarity[mask]
            mean_pairwise_similarity = float(pairwise_values.mean().item())
            min_pairwise_similarity = float(pairwise_values.min().item())
            similarity_means = family_similarity.mean(dim=1)
            representative_index = member_indices[int(similarity_means.argmax().item())]

        representative_feature_id = feature_ids[representative_index]
        final_metric_means = {
            metric_name: float(metric_tables[metric_name][member_indices, -1].mean().item())
            for metric_name in selected_metrics
        }
        peak_metrics = {}
        for metric_name in selected_metrics:
            mean_curve = metric_tables[metric_name][member_indices].mean(dim=0)
            peak_index = int(mean_curve.argmax().item())
            peak_metrics[metric_name] = {
                "step": int(sorted_steps[peak_index]),
                "value": float(mean_curve[peak_index].item()),
            }
        birth_summary = _compute_family_birth_summary(
            member_feature_ids=member_feature_ids,
            birth_rows_by_feature=birth_rows_by_feature,
        )
        family_summaries.append(
            {
                "family_id": family_id,
                "size": len(member_feature_ids),
                "member_feature_ids": member_feature_ids,
                "representative_feature_id": representative_feature_id,
                "mean_pairwise_similarity": mean_pairwise_similarity,
                "min_pairwise_similarity": min_pairwise_similarity,
                "final_metric_means": final_metric_means,
                "peak_metrics": peak_metrics,
                "birth_summary": birth_summary,
            }
        )

        for step_index, step in enumerate(sorted_steps):
            row: dict[str, Any] = {
                "family_id": family_id,
                "step": int(step),
                "size": len(member_feature_ids),
                "member_feature_ids": member_feature_ids,
            }
            for metric_name in selected_metrics:
                values = metric_tables[metric_name][member_indices, step_index]
                row[f"{metric_name}_mean"] = float(values.mean().item())
                row[f"{metric_name}_max"] = float(values.max().item())
                row[f"{metric_name}_min"] = float(values.min().item())
            family_trajectory_rows.append(row)

    for feature_index, feature_id in enumerate(feature_ids):
        neighbor_indices = sorted(
            [index for index in range(len(feature_ids)) if index != feature_index],
            key=lambda index: float(similarity[feature_index, index]),
            reverse=True,
        )[:5]
        feature_assignment_rows.append(
            {
                "feature_id": feature_id,
                "family_id": family_assignments_by_feature[feature_id],
                "top_similarity_neighbors": [
                    {
                        "feature_id": feature_ids[index],
                        "family_id": family_assignments_by_feature[feature_ids[index]],
                        "similarity": float(similarity[feature_index, index].item()),
                    }
                    for index in neighbor_indices
                ],
            }
        )

    similarity_edges: list[dict[str, Any]] = []
    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    for family_summary in family_summaries:
        family_id = int(family_summary["family_id"])
        graph_nodes.append(
            {
                "id": f"family:{family_id}",
                "type": "feature_family",
                "label": f"G{family_id} (n={family_summary['size']})",
            }
        )
    for feature_id in feature_ids:
        graph_nodes.append(
            {
                "id": f"feature:{feature_id}",
                "type": "feature",
                "label": f"F{feature_id}",
            }
        )
        family_id = family_assignments_by_feature[feature_id]
        graph_edges.append(
            {
                "source": f"feature:{feature_id}",
                "target": f"family:{family_id}",
                "type": "member_of",
                "score": 1.0,
                "score_type": "membership",
            }
        )
    for left_index in range(len(feature_ids)):
        for right_index in range(left_index + 1, len(feature_ids)):
            score = float(similarity[left_index, right_index].item())
            if score < similarity_threshold:
                continue
            left_feature = feature_ids[left_index]
            right_feature = feature_ids[right_index]
            similarity_edges.append(
                {
                    "source_feature_id": left_feature,
                    "target_feature_id": right_feature,
                    "similarity": score,
                    "same_family": family_assignments_by_feature[left_feature] == family_assignments_by_feature[right_feature],
                }
            )
            graph_edges.append(
                {
                    "source": f"feature:{left_feature}",
                    "target": f"feature:{right_feature}",
                    "type": "similarity",
                    "score": score,
                    "score_type": "cosine_similarity",
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    families_path = output_dir / "feature_families.json"
    family_trajectories_path = output_dir / "feature_family_trajectories.jsonl"
    graph_path = output_dir / "feature_family_graph.json"

    write_json(
        families_path,
        {
            "trajectories_path": str(trajectories_path),
            "feature_births_path": None if feature_births_path is None else str(feature_births_path),
            "metrics_used": selected_metrics,
            "cluster_method": "complete_link_threshold",
            "similarity_threshold": similarity_threshold,
            "steps": sorted_steps,
            "num_features": len(feature_ids),
            "num_families": len(family_summaries),
            "num_multi_feature_families": sum(1 for row in family_summaries if int(row["size"]) > 1),
            "feature_assignments": feature_assignment_rows,
            "families": family_summaries,
            "similarity_edges": similarity_edges,
        },
    )
    write_jsonl(family_trajectories_path, family_trajectory_rows)
    write_json(graph_path, {"nodes": graph_nodes, "edges": graph_edges})

    order_indices = [feature_ids.index(feature_id) for feature_id in ordered_feature_ids]
    ordered_similarity = similarity[order_indices][:, order_indices]
    plots_dir = output_dir / "plots"
    similarity_heatmap_path = _render_family_similarity_heatmap(
        similarity=ordered_similarity,
        ordered_feature_ids=ordered_feature_ids,
        family_assignments_by_feature=family_assignments_by_feature,
        output_path=plots_dir / "feature_family_similarity_heatmap.svg",
    )
    family_trajectory_plot_path = _render_family_trajectory_plot(
        family_trajectory_rows=family_trajectory_rows,
        family_summaries=family_summaries,
        metrics=selected_metrics,
        output_path=plots_dir / "feature_family_trajectories.svg",
        top_k_families=top_k_families,
    )
    return families_path, family_trajectories_path, graph_path, {
        "similarity_heatmap": similarity_heatmap_path,
        "family_trajectory_plot": family_trajectory_plot_path,
    }


def _load_feature_family_payload(families_path: Path) -> dict[str, Any]:
    payload = read_json(families_path)
    if not isinstance(payload.get("families"), list):
        raise ValueError(f"feature families file must contain a 'families' list: {families_path}")
    if not isinstance(payload.get("feature_assignments"), list):
        raise ValueError(f"feature families file must contain a 'feature_assignments' list: {families_path}")
    return payload


def _load_feature_compare_payload(feature_compare_path: Path) -> dict[str, Any]:
    payload = read_json(feature_compare_path)
    if not isinstance(payload.get("diff_rows"), list):
        raise ValueError(f"feature compare file must contain a 'diff_rows' list: {feature_compare_path}")
    return payload


def _resolve_family_feature_ids(
    *,
    families_payload: dict[str, Any],
    family_ids: list[int],
) -> tuple[list[int], list[dict[str, Any]]]:
    if not family_ids:
        raise ValueError("family_ids must not be empty.")
    families_by_id = {
        int(row["family_id"]): row
        for row in families_payload["families"]
    }
    missing = [family_id for family_id in family_ids if family_id not in families_by_id]
    if missing:
        raise KeyError(f"Unknown family ids {missing} in feature families payload.")
    selected_families = [families_by_id[family_id] for family_id in family_ids]
    feature_ids = sorted(
        {
            int(feature_id)
            for family in selected_families
            for feature_id in family["member_feature_ids"]
        }
    )
    return feature_ids, selected_families


def _render_labeled_bar_plot(
    *,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    xlabel: str,
    output_path: Path,
) -> Path:
    if not labels or not values or len(labels) != len(values):
        raise ValueError("labels and values must be non-empty and have matching lengths.")
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_feature_family_rank_scatter(
    *,
    member_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not member_rows:
        raise ValueError("member_rows must not be empty.")
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    x_values = [float(row["heldout_gap_delta"]) for row in member_rows]
    y_values = [float(row["correctness_gap_delta"]) for row in member_rows]
    colors = [float(row["structural_ood_gap_delta"]) for row in member_rows]
    scatter = ax.scatter(x_values, y_values, c=colors, cmap="coolwarm", s=80, edgecolors="black", linewidths=0.4)
    for row, x_value, y_value in zip(member_rows, x_values, y_values, strict=True):
        ax.annotate(
            f"F{row['feature_id']}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=9,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Feature-family member tradeoffs")
    ax.set_xlabel("Heldout-gap delta")
    ax.set_ylabel("Correctness-gap delta")
    fig.colorbar(scatter, ax=ax, label="Structural-OOD-gap delta")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _resolve_feature_family_rows(
    *,
    families_payload: dict[str, Any],
    feature_compare_payload: dict[str, Any],
    family_id: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    families_by_id = {
        int(row["family_id"]): row
        for row in families_payload["families"]
    }
    family_summary = families_by_id.get(int(family_id))
    if family_summary is None:
        raise KeyError(f"Unknown family id {family_id} in feature families payload.")
    compare_rows_by_feature = {
        int(row["feature_id"]): row
        for row in feature_compare_payload["diff_rows"]
    }
    member_rows: list[dict[str, Any]] = []
    for feature_id in family_summary["member_feature_ids"]:
        compare_row = compare_rows_by_feature.get(int(feature_id))
        if compare_row is None:
            raise KeyError(
                f"Family {family_id} references feature {feature_id}, but it is missing from feature compare payload."
            )
        combined = {
            **compare_row,
            "family_id": int(family_summary["family_id"]),
            "family_size": int(family_summary["size"]),
            "representative_feature_id": int(family_summary["representative_feature_id"]),
            "mean_pairwise_similarity": float(family_summary["mean_pairwise_similarity"]),
            "useful_delta": float(compare_row["heldout_gap_delta"] + compare_row["correctness_gap_delta"]),
        }
        member_rows.append(combined)
    member_rows.sort(key=lambda row: int(row["feature_id"]))
    return family_summary, member_rows


def _make_ranked_family_subsets(
    *,
    ranked_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not ranked_rows:
        raise ValueError("ranked_rows must not be empty.")
    suggestions: list[dict[str, Any]] = []
    running_mean_activation = 0.0
    running_active_fraction = 0.0
    running_correctness = 0.0
    running_heldout = 0.0
    running_structural = 0.0
    for subset_size, row in enumerate(ranked_rows, start=1):
        running_mean_activation += float(row["mean_activation_delta"])
        running_active_fraction += float(row["active_fraction_delta"])
        running_correctness += float(row["correctness_gap_delta"])
        running_heldout += float(row["heldout_gap_delta"])
        running_structural += float(row["structural_ood_gap_delta"])
        suggestions.append(
            {
                "subset_size": subset_size,
                "feature_ids": [int(item["feature_id"]) for item in ranked_rows[:subset_size]],
                "aggregate_compare_proxy": {
                    "mean_activation_delta": running_mean_activation,
                    "active_fraction_delta": running_active_fraction,
                    "correctness_gap_delta": running_correctness,
                    "heldout_gap_delta": running_heldout,
                    "structural_ood_gap_delta": running_structural,
                    "useful_delta": running_correctness + running_heldout,
                },
            }
        )
    return suggestions


def _birth_event(
    *,
    rows: list[dict[str, Any]],
    metric_name: str,
    threshold: float,
    delta_threshold: float,
    window: int,
) -> dict[str, Any] | None:
    if window <= 0:
        raise ValueError("window must be positive.")
    values = [float(row[metric_name]) for row in rows]
    steps = [int(row["step"]) for row in rows]
    if len(values) < (2 * window):
        return None
    for index in range(window, len(values) - window + 1):
        previous_window = values[index - window:index]
        next_window = values[index:index + window]
        previous_mean = sum(previous_window) / len(previous_window)
        next_mean = sum(next_window) / len(next_window)
        if next_mean >= threshold and (next_mean - previous_mean) >= delta_threshold:
            peak_offset = max(range(len(next_window)), key=lambda offset: next_window[offset])
            return {
                "birth_step": steps[index],
                "stabilization_step": steps[index + window - 1],
                "peak_step": steps[index + peak_offset],
                "peak_value": next_window[peak_offset],
                "pre_birth_mean": previous_mean,
                "post_birth_mean": next_mean,
                "birth_window": [steps[index], steps[index + window - 1]],
            }
    return None


def feature_birth_analyze(
    *,
    trajectories_path: Path,
    output_dir: Path,
    thresholds: dict[str, float],
    delta_threshold: float,
    window: int,
) -> tuple[Path, Path, Path]:
    rows = list(iter_jsonl(trajectories_path))
    if not rows:
        raise ValueError(f"No rows found in trajectories file: {trajectories_path}")
    rows_by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_feature[int(row["feature_id"])].append(row)
    for feature_rows in rows_by_feature.values():
        feature_rows.sort(key=lambda item: int(item["step"]))

    birth_rows: list[dict[str, Any]] = []
    for feature_id, feature_rows in sorted(rows_by_feature.items()):
        births: dict[str, Any] = {}
        for metric_name, threshold in thresholds.items():
            if metric_name not in feature_rows[0]:
                raise KeyError(f"Metric {metric_name} not found in trajectory rows.")
            birth = _birth_event(
                rows=feature_rows,
                metric_name=metric_name,
                threshold=float(threshold),
                delta_threshold=delta_threshold,
                window=window,
            )
            if birth is not None:
                births[metric_name] = birth
        first_birth_step = None
        if births:
            first_birth_step = min(int(payload["birth_step"]) for payload in births.values())
        birth_rows.append(
            {
                "feature_id": feature_id,
                "birth_step": first_birth_step,
                "births": births,
                "final_metrics": {
                    metric_name: float(feature_rows[-1][metric_name])
                    for metric_name in thresholds
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    births_path = output_dir / "feature_births.json"
    summary_path = output_dir / "feature_birth_summary.json"
    plot_path = output_dir / "plots" / "feature_birth_raster.svg"
    write_json(
        births_path,
        {
            "trajectories_path": str(trajectories_path),
            "thresholds": thresholds,
            "delta_threshold": delta_threshold,
            "window": window,
            "features": birth_rows,
        },
    )
    sorted_births = sorted(
        [row for row in birth_rows if row["birth_step"] is not None],
        key=lambda row: int(row["birth_step"]),
    )
    births_by_metric: dict[str, list[dict[str, Any]]] = {}
    for metric_name in thresholds:
        metric_rows = [
            {
                "feature_id": row["feature_id"],
                "birth_step": int(row["births"][metric_name]["birth_step"]),
                "birth": row["births"][metric_name],
                "final_metric_value": float(row["final_metrics"][metric_name]),
            }
            for row in birth_rows
            if metric_name in row["births"]
        ]
        births_by_metric[metric_name] = sorted(metric_rows, key=lambda item: int(item["birth_step"]))

    useful_metric_names = {"correctness_gap", "heldout_gap"}
    useful_births = sorted(
        [
            row
            for row in birth_rows
            if all(metric_name in row["births"] for metric_name in useful_metric_names)
            and all(float(row["final_metrics"][metric_name]) > 0.0 for metric_name in useful_metric_names)
        ],
        key=lambda row: max(int(row["births"][metric_name]["birth_step"]) for metric_name in useful_metric_names),
    )
    write_json(
        summary_path,
        {
            "trajectories_path": str(trajectories_path),
            "num_features": len(birth_rows),
            "num_features_with_birth": len(sorted_births),
            "earliest_births": sorted_births[:12],
            "earliest_births_by_metric": {
                metric_name: rows_for_metric[:12] for metric_name, rows_for_metric in births_by_metric.items()
            },
            "num_features_with_birth_by_metric": {
                metric_name: len(rows_for_metric) for metric_name, rows_for_metric in births_by_metric.items()
            },
            "earliest_useful_births": useful_births[:12],
        },
    )
    _render_birth_plot(birth_rows=sorted_births, output_path=plot_path)
    return births_path, summary_path, plot_path


def feature_family_birth_analyze(
    *,
    family_trajectories_path: Path,
    families_path: Path,
    output_dir: Path,
    thresholds: dict[str, float],
    delta_threshold: float,
    window: int,
) -> tuple[Path, Path, Path]:
    family_rows = list(iter_jsonl(family_trajectories_path))
    if not family_rows:
        raise ValueError(f"No rows found in family trajectories file: {family_trajectories_path}")
    families_payload = _load_feature_family_payload(families_path)
    families_by_id = {
        int(row["family_id"]): row
        for row in families_payload["families"]
    }
    rows_by_family: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in family_rows:
        rows_by_family[int(row["family_id"])].append(row)
    for grouped_rows in rows_by_family.values():
        grouped_rows.sort(key=lambda item: int(item["step"]))

    birth_rows: list[dict[str, Any]] = []
    useful_metric_names = {"correctness_gap", "heldout_gap"}
    for family_id, grouped_rows in sorted(rows_by_family.items()):
        family_summary = families_by_id.get(family_id)
        if family_summary is None:
            raise KeyError(
                f"Family {family_id} present in family trajectories but missing from families payload: {families_path}"
            )
        births: dict[str, Any] = {}
        final_metrics: dict[str, float] = {}
        for metric_name, threshold in thresholds.items():
            curve_metric_name = f"{metric_name}_mean"
            if curve_metric_name not in grouped_rows[0]:
                raise KeyError(
                    f"Metric {curve_metric_name} not found in family trajectories rows: {family_trajectories_path}"
                )
            birth = _birth_event(
                rows=grouped_rows,
                metric_name=curve_metric_name,
                threshold=float(threshold),
                delta_threshold=delta_threshold,
                window=window,
            )
            if birth is not None:
                births[metric_name] = birth
            final_metrics[metric_name] = float(grouped_rows[-1][curve_metric_name])
        birth_step = None if not births else min(int(payload["birth_step"]) for payload in births.values())
        useful_birth_step = None
        if (
            all(metric_name in births for metric_name in useful_metric_names)
            and all(final_metrics[metric_name] > 0.0 for metric_name in useful_metric_names)
        ):
            useful_birth_step = max(int(births[metric_name]["birth_step"]) for metric_name in useful_metric_names)
        birth_rows.append(
            {
                "family_id": family_id,
                "size": int(family_summary["size"]),
                "representative_feature_id": int(family_summary["representative_feature_id"]),
                "member_feature_ids": [int(feature_id) for feature_id in family_summary["member_feature_ids"]],
                "mean_pairwise_similarity": float(family_summary["mean_pairwise_similarity"]),
                "birth_step": birth_step,
                "useful_birth_step": useful_birth_step,
                "births": births,
                "final_metrics": final_metrics,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    births_path = output_dir / "feature_family_births.json"
    summary_path = output_dir / "feature_family_birth_summary.json"
    plot_path = output_dir / "plots" / "feature_family_birth_raster.svg"
    write_json(
        births_path,
        {
            "family_trajectories_path": str(family_trajectories_path),
            "families_path": str(families_path),
            "thresholds": thresholds,
            "delta_threshold": delta_threshold,
            "window": window,
            "families": birth_rows,
        },
    )

    sorted_births = sorted(
        [row for row in birth_rows if row["birth_step"] is not None],
        key=lambda row: int(row["birth_step"]),
    )
    births_by_metric: dict[str, list[dict[str, Any]]] = {}
    for metric_name in thresholds:
        metric_rows = [
            {
                "family_id": int(row["family_id"]),
                "size": int(row["size"]),
                "representative_feature_id": int(row["representative_feature_id"]),
                "birth_step": int(row["births"][metric_name]["birth_step"]),
                "birth": row["births"][metric_name],
                "final_metric_value": float(row["final_metrics"][metric_name]),
            }
            for row in birth_rows
            if metric_name in row["births"]
        ]
        births_by_metric[metric_name] = sorted(metric_rows, key=lambda item: int(item["birth_step"]))
    useful_births = sorted(
        [row for row in birth_rows if row["useful_birth_step"] is not None],
        key=lambda row: int(row["useful_birth_step"]),
    )
    write_json(
        summary_path,
        {
            "family_trajectories_path": str(family_trajectories_path),
            "families_path": str(families_path),
            "num_families": len(birth_rows),
            "num_families_with_birth": len(sorted_births),
            "earliest_births": sorted_births[:12],
            "earliest_births_by_metric": {
                metric_name: rows_for_metric[:12]
                for metric_name, rows_for_metric in births_by_metric.items()
            },
            "num_families_with_birth_by_metric": {
                metric_name: len(rows_for_metric)
                for metric_name, rows_for_metric in births_by_metric.items()
            },
            "earliest_useful_births": useful_births[:12],
        },
    )
    _render_birth_plot(
        birth_rows=sorted_births,
        output_path=plot_path,
        entity_id_key="family_id",
        title="Feature-family birth raster",
        ylabel="Family rows",
    )
    return births_path, summary_path, plot_path


def feature_compare(
    *,
    trajectories_path: Path,
    source_step: int,
    target_step: int,
    output_path: Path,
    top_k: int = 12,
) -> tuple[Path, Path]:
    rows = list(iter_jsonl(trajectories_path))
    if not rows:
        raise ValueError(f"No rows found in trajectories file: {trajectories_path}")
    by_step_feature: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        by_step_feature[(int(row["step"]), int(row["feature_id"]))] = row
    feature_ids = sorted({int(row["feature_id"]) for row in rows})
    diff_rows: list[dict[str, Any]] = []
    for feature_id in feature_ids:
        source_row = by_step_feature.get((source_step, feature_id))
        target_row = by_step_feature.get((target_step, feature_id))
        if source_row is None or target_row is None:
            raise KeyError(f"Missing trajectory row for feature {feature_id} at source={source_step} or target={target_step}.")
        diff_rows.append(
            {
                "feature_id": feature_id,
                "source_step": source_step,
                "target_step": target_step,
                "mean_activation_delta": float(target_row["mean_activation"] - source_row["mean_activation"]),
                "active_fraction_delta": float(target_row["active_fraction"] - source_row["active_fraction"]),
                "correctness_gap_delta": float(target_row["correctness_gap"] - source_row["correctness_gap"]),
                "heldout_gap_delta": float(target_row["heldout_gap"] - source_row["heldout_gap"]),
                "structural_ood_gap_delta": float(target_row["structural_ood_gap"] - source_row["structural_ood_gap"]),
                "margin_correlation_delta": float(target_row["margin_correlation"] - source_row["margin_correlation"]),
                "answer_direction_alignment_delta": float(
                    target_row["answer_direction_alignment"] - source_row["answer_direction_alignment"]
                ),
            }
        )
    payload = {
        "trajectories_path": str(trajectories_path),
        "source_step": source_step,
        "target_step": target_step,
        "top_features": {
            "by_abs_mean_activation_delta": sorted(diff_rows, key=lambda row: abs(float(row["mean_activation_delta"])), reverse=True)[:top_k],
            "by_heldout_gap_delta": sorted(diff_rows, key=lambda row: float(row["heldout_gap_delta"]), reverse=True)[:top_k],
            "by_correctness_gap_delta": sorted(diff_rows, key=lambda row: float(row["correctness_gap_delta"]), reverse=True)[:top_k],
        },
        "diff_rows": diff_rows,
    }
    write_json(output_path, payload)
    plot_path = output_path.with_name(f"{output_path.stem}_bar.svg")
    _render_compare_plot(diff_rows=diff_rows, output_path=plot_path, top_k=top_k)
    return output_path, plot_path


def feature_family_compare(
    *,
    family_trajectories_path: Path,
    families_path: Path,
    source_step: int,
    target_step: int,
    output_path: Path,
    top_k: int = 12,
) -> tuple[Path, Path]:
    family_rows = list(iter_jsonl(family_trajectories_path))
    if not family_rows:
        raise ValueError(f"No rows found in family trajectories file: {family_trajectories_path}")
    families_payload = _load_feature_family_payload(families_path)
    families_by_id = {
        int(row["family_id"]): row
        for row in families_payload["families"]
    }
    by_step_family: dict[tuple[int, int], dict[str, Any]] = {}
    for row in family_rows:
        by_step_family[(int(row["step"]), int(row["family_id"]))] = row
    family_ids = sorted({int(row["family_id"]) for row in family_rows})
    metric_names = sorted(
        key[:-5]
        for key in family_rows[0]
        if key.endswith("_mean")
    )
    if not metric_names:
        raise ValueError(f"No family mean metrics found in {family_trajectories_path}")

    diff_rows: list[dict[str, Any]] = []
    for family_id in family_ids:
        source_row = by_step_family.get((source_step, family_id))
        target_row = by_step_family.get((target_step, family_id))
        if source_row is None or target_row is None:
            raise KeyError(
                f"Missing family trajectory row for family {family_id} at source={source_step} or target={target_step}."
            )
        family_summary = families_by_id.get(family_id)
        if family_summary is None:
            raise KeyError(f"Family {family_id} present in trajectory rows but missing from families payload.")
        delta_row: dict[str, Any] = {
            "family_id": family_id,
            "source_step": source_step,
            "target_step": target_step,
            "size": int(family_summary["size"]),
            "representative_feature_id": int(family_summary["representative_feature_id"]),
            "member_feature_ids": [int(feature_id) for feature_id in family_summary["member_feature_ids"]],
            "mean_pairwise_similarity": float(family_summary["mean_pairwise_similarity"]),
        }
        for metric_name in metric_names:
            delta_row[f"{metric_name}_mean_delta"] = float(
                target_row[f"{metric_name}_mean"] - source_row[f"{metric_name}_mean"]
            )
        delta_row["useful_delta"] = float(
            delta_row.get("heldout_gap_mean_delta", 0.0) + delta_row.get("correctness_gap_mean_delta", 0.0)
        )
        diff_rows.append(delta_row)

    payload = {
        "family_trajectories_path": str(family_trajectories_path),
        "families_path": str(families_path),
        "source_step": source_step,
        "target_step": target_step,
        "top_families": {
            "by_useful_delta": sorted(diff_rows, key=lambda row: float(row["useful_delta"]), reverse=True)[:top_k],
            "by_heldout_gap_mean_delta": sorted(
                diff_rows,
                key=lambda row: float(row.get("heldout_gap_mean_delta", 0.0)),
                reverse=True,
            )[:top_k],
            "by_correctness_gap_mean_delta": sorted(
                diff_rows,
                key=lambda row: float(row.get("correctness_gap_mean_delta", 0.0)),
                reverse=True,
            )[:top_k],
            "by_abs_mean_activation_mean_delta": sorted(
                diff_rows,
                key=lambda row: abs(float(row.get("mean_activation_mean_delta", 0.0))),
                reverse=True,
            )[:top_k],
        },
        "diff_rows": diff_rows,
    }
    write_json(output_path, payload)
    ranked = payload["top_families"]["by_useful_delta"][:top_k]
    plot_path = output_path.with_name(f"{output_path.stem}_bar.svg")
    _render_labeled_bar_plot(
        labels=[f"G{row['family_id']}" for row in ranked],
        values=[float(row["useful_delta"]) for row in ranked],
        title="Feature-family useful deltas",
        ylabel="Heldout delta + correctness delta",
        xlabel="Family",
        output_path=plot_path,
    )
    return output_path, plot_path


def feature_family_rank(
    *,
    families_path: Path,
    feature_compare_path: Path,
    family_id: int,
    output_path: Path,
) -> tuple[Path, dict[str, Path]]:
    families_payload = _load_feature_family_payload(families_path)
    feature_compare_payload = _load_feature_compare_payload(feature_compare_path)
    family_summary, member_rows = _resolve_feature_family_rows(
        families_payload=families_payload,
        feature_compare_payload=feature_compare_payload,
        family_id=family_id,
    )
    rankings = {
        "by_useful_delta": sorted(member_rows, key=lambda row: float(row["useful_delta"]), reverse=True),
        "by_heldout_gap_delta": sorted(member_rows, key=lambda row: float(row["heldout_gap_delta"]), reverse=True),
        "by_correctness_gap_delta": sorted(member_rows, key=lambda row: float(row["correctness_gap_delta"]), reverse=True),
        "by_abs_mean_activation_delta": sorted(
            member_rows,
            key=lambda row: abs(float(row["mean_activation_delta"])),
            reverse=True,
        ),
    }
    payload = {
        "families_path": str(families_path),
        "feature_compare_path": str(feature_compare_path),
        "family_id": int(family_summary["family_id"]),
        "source_step": int(feature_compare_payload["source_step"]),
        "target_step": int(feature_compare_payload["target_step"]),
        "family_summary": family_summary,
        "member_rows": member_rows,
        "rankings": rankings,
        "suggested_subsets": {
            ranking_name: _make_ranked_family_subsets(ranked_rows=ranked_rows)
            for ranking_name, ranked_rows in rankings.items()
        },
    }
    write_json(output_path, payload)
    ranked_useful = rankings["by_useful_delta"]
    plot_paths = {
        "useful_delta_bar": _render_labeled_bar_plot(
            labels=[f"F{row['feature_id']}" for row in ranked_useful],
            values=[float(row["useful_delta"]) for row in ranked_useful],
            title=f"Feature-family {family_id} useful deltas",
            ylabel="Heldout delta + correctness delta",
            xlabel="Feature",
            output_path=output_path.with_name(f"{output_path.stem}_useful_delta_bar.svg"),
        ),
        "tradeoff_scatter": _render_feature_family_rank_scatter(
            member_rows=member_rows,
            output_path=output_path.with_name(f"{output_path.stem}_tradeoff_scatter.svg"),
        ),
    }
    return output_path, plot_paths


def _compute_answer_metrics_from_logits(
    *,
    logits: torch.Tensor,
    batch: dict[str, Any],
) -> dict[str, Any]:
    answer_logits, answer_targets, metadata = extract_answer_logits(logits, batch)
    predictions = answer_logits.argmax(dim=-1)
    correct = predictions == answer_targets
    split_correct: dict[str, int] = defaultdict(int)
    split_total: dict[str, int] = defaultdict(int)
    for answer_index, row_index in enumerate(metadata["rows"].tolist()):
        split_name = str(batch["records"][row_index]["split"])
        split_total[split_name] += 1
        if bool(correct[answer_index].item()):
            split_correct[split_name] += 1
    return {
        "answer_correct": int(correct.sum().item()),
        "answer_total": int(answer_targets.numel()),
        "split_correct": dict(split_correct),
        "split_total": dict(split_total),
    }


def _finalize_answer_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    answer_accuracy = metrics["answer_correct"] / metrics["answer_total"] if metrics["answer_total"] else 0.0
    split_accuracy = {
        split_name: metrics["split_correct"].get(split_name, 0) / total
        for split_name, total in metrics["split_total"].items()
    }
    return {
        "answer_accuracy": answer_accuracy,
        "heldout_answer_accuracy": split_accuracy.get("heldout_pairs", 0.0),
        "structural_ood_answer_accuracy": split_accuracy.get("structural_ood", 0.0),
        "split_answer_accuracy": split_accuracy,
        "num_answers": metrics["answer_total"],
    }


def _empty_answer_metrics() -> dict[str, Any]:
    return {"answer_correct": 0, "answer_total": 0, "split_correct": {}, "split_total": {}}


def _accumulate_answer_metrics(accumulator: dict[str, Any], batch_metrics: dict[str, Any]) -> None:
    accumulator["answer_correct"] += batch_metrics["answer_correct"]
    accumulator["answer_total"] += batch_metrics["answer_total"]
    for split_name, value in batch_metrics["split_correct"].items():
        accumulator["split_correct"][split_name] = accumulator["split_correct"].get(split_name, 0) + value
    for split_name, value in batch_metrics["split_total"].items():
        accumulator["split_total"][split_name] = accumulator["split_total"].get(split_name, 0) + value


def _run_feature_patch(
    *,
    config_path: Path,
    probe_set_path: Path,
    basis_path: Path,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    stage_name: str,
    feature_ids: list[int],
    output_path: Path,
    device_name: str = "cpu",
    patch_mode: str = "replace",
) -> dict[str, Any]:
    if patch_mode not in {"replace", "ablate", "additive_delta"}:
        raise ValueError(f"Unsupported patch_mode: {patch_mode}")
    if not feature_ids:
        raise ValueError("feature_ids must not be empty.")
    spec, vocab, analysis_batches, _, _, device = _load_probe_batches(
        config_path=config_path,
        probe_set_path=probe_set_path,
        device_name=device_name,
    )
    basis = _load_shared_basis(basis_path, device)
    if str(basis["stage_name"]) != stage_name:
        raise ValueError(f"Basis stage {basis['stage_name']} does not match requested stage {stage_name}.")
    normalization_mean = basis["normalization_mean"]
    normalization_std = basis["normalization_std"]
    sae: SparseAutoencoder = basis["sae"]
    if max(feature_ids) >= int(basis["num_features"]) or min(feature_ids) < 0:
        raise ValueError(f"feature_ids must be within [0, {int(basis['num_features']) - 1}]")

    source_model = build_model(spec.model, len(vocab.tokens), device)
    source_checkpoint = load_checkpoint(source_checkpoint_path, device)
    load_model_state(source_model, source_checkpoint["model_state"])
    source_model.eval()
    target_model = build_model(spec.model, len(vocab.tokens), device)
    target_checkpoint = load_checkpoint(target_checkpoint_path, device)
    load_model_state(target_model, target_checkpoint["model_state"])
    target_model.eval()

    baseline_metrics = _empty_answer_metrics()
    patched_metrics = _empty_answer_metrics()
    target_reconstruction_errors: list[float] = []
    source_reconstruction_errors: list[float] = []

    for batch in analysis_batches:
        target_outputs = target_model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_residual_streams=True,
        )
        source_outputs = source_model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_residual_streams=True,
        )
        if target_outputs.residual_streams is None or source_outputs.residual_streams is None:
            raise RuntimeError("Feature patch requires residual_streams in model output.")
        if stage_name not in target_outputs.residual_streams or stage_name not in source_outputs.residual_streams:
            raise KeyError(f"Residual stage {stage_name} not found in source/target outputs.")

        batch_baseline_metrics = _compute_answer_metrics_from_logits(logits=target_outputs.logits, batch=batch)
        _accumulate_answer_metrics(baseline_metrics, batch_baseline_metrics)

        _, _, metadata = extract_answer_logits(target_outputs.logits, batch)
        rows = metadata["rows"]
        prediction_positions = metadata["prediction_positions"]

        target_stage = target_outputs.residual_streams[stage_name]
        source_stage = source_outputs.residual_streams[stage_name]
        target_selected = target_stage[rows, prediction_positions, :]
        source_selected = source_stage[rows, prediction_positions, :]

        target_norm = _normalize_activations(target_selected, normalization_mean, normalization_std)
        source_norm = _normalize_activations(source_selected, normalization_mean, normalization_std)
        target_features = _encode_features(sae, target_norm)
        source_features = _encode_features(sae, source_norm)

        patched_features = target_features.clone()
        feature_index = torch.tensor(feature_ids, device=target_features.device, dtype=torch.long)
        if patch_mode == "replace":
            patched_features.index_copy_(1, feature_index, source_features.index_select(1, feature_index))
        elif patch_mode == "ablate":
            patched_features.index_fill_(1, feature_index, 0.0)
        elif patch_mode == "additive_delta":
            patched_features.index_copy_(
                1,
                feature_index,
                target_features.index_select(1, feature_index) + (
                    source_features.index_select(1, feature_index) - target_features.index_select(1, feature_index)
                ),
            )

        target_reconstruction = sae.decoder(target_features)
        source_reconstruction = sae.decoder(source_features)
        patched_reconstruction = sae.decoder(patched_features)
        target_reconstruction_errors.append(float(F.mse_loss(target_reconstruction, target_norm).item()))
        source_reconstruction_errors.append(float(F.mse_loss(source_reconstruction, source_norm).item()))
        patched_selected = target_selected + _denormalize_activations(
            patched_reconstruction - target_reconstruction,
            torch.zeros_like(normalization_mean),
            normalization_std,
        )

        full_patch = target_stage.clone()
        full_patch[rows, prediction_positions, :] = patched_selected
        patched_outputs = target_model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            residual_patch={stage_name: full_patch},
        )
        batch_patched_metrics = _compute_answer_metrics_from_logits(logits=patched_outputs.logits, batch=batch)
        _accumulate_answer_metrics(patched_metrics, batch_patched_metrics)

    baseline = _finalize_answer_metrics(baseline_metrics)
    patched = _finalize_answer_metrics(patched_metrics)
    payload = {
        "basis_id": str(basis["basis_id"]),
        "stage_name": stage_name,
        "patch_mode": patch_mode,
        "feature_ids": feature_ids,
        "source_checkpoint_path": str(source_checkpoint_path),
        "source_checkpoint_step": int(source_checkpoint["step"]),
        "target_checkpoint_path": str(target_checkpoint_path),
        "target_checkpoint_step": int(target_checkpoint["step"]),
        "baseline": baseline,
        "patched": patched,
        "deltas": {
            "answer_accuracy": patched["answer_accuracy"] - baseline["answer_accuracy"],
            "heldout_answer_accuracy": patched["heldout_answer_accuracy"] - baseline["heldout_answer_accuracy"],
            "structural_ood_answer_accuracy": patched["structural_ood_answer_accuracy"] - baseline["structural_ood_answer_accuracy"],
        },
        "reconstruction": {
            "target_reconstruction_mse_mean": sum(target_reconstruction_errors) / len(target_reconstruction_errors),
            "source_reconstruction_mse_mean": sum(source_reconstruction_errors) / len(source_reconstruction_errors),
        },
    }
    return payload


def feature_patch(
    *,
    config_path: Path,
    probe_set_path: Path,
    basis_path: Path,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    stage_name: str,
    feature_ids: list[int],
    output_path: Path,
    device_name: str = "cpu",
    patch_mode: str = "replace",
) -> Path:
    payload = _run_feature_patch(
        config_path=config_path,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        source_checkpoint_path=source_checkpoint_path,
        target_checkpoint_path=target_checkpoint_path,
        stage_name=stage_name,
        feature_ids=feature_ids,
        output_path=output_path,
        device_name=device_name,
        patch_mode=patch_mode,
    )
    write_json(output_path, payload)
    return output_path


def feature_family_patch(
    *,
    config_path: Path,
    probe_set_path: Path,
    basis_path: Path,
    families_path: Path,
    family_ids: list[int],
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    stage_name: str,
    output_path: Path,
    device_name: str = "cpu",
    patch_mode: str = "replace",
) -> Path:
    families_payload = _load_feature_family_payload(families_path)
    resolved_feature_ids, selected_families = _resolve_family_feature_ids(
        families_payload=families_payload,
        family_ids=family_ids,
    )
    payload = _run_feature_patch(
        config_path=config_path,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        source_checkpoint_path=source_checkpoint_path,
        target_checkpoint_path=target_checkpoint_path,
        stage_name=stage_name,
        feature_ids=resolved_feature_ids,
        output_path=output_path,
        device_name=device_name,
        patch_mode=patch_mode,
    )
    payload.update(
        {
            "families_path": str(families_path),
            "family_ids": family_ids,
            "selected_families": selected_families,
            "resolved_feature_ids": resolved_feature_ids,
        }
    )
    write_json(output_path, payload)
    return output_path


def _load_feature_family_rank_payload(feature_family_rank_path: Path) -> dict[str, Any]:
    payload = read_json(feature_family_rank_path)
    required = ["family_id", "member_rows", "rankings", "suggested_subsets"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"feature family rank payload is missing required keys {missing}: {feature_family_rank_path}")
    return payload


def _load_feature_family_subpatch_payload(feature_family_subpatch_path: Path) -> dict[str, Any]:
    payload = read_json(feature_family_subpatch_path)
    required = [
        "family_id",
        "ranking_name",
        "stage_name",
        "subset_results",
        "best_subset_by_heldout",
        "source_checkpoint_path",
        "target_checkpoint_path",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(
            f"feature family subpatch payload is missing required keys {missing}: {feature_family_subpatch_path}"
        )
    return payload


def _load_feature_family_lineage_payload(feature_family_lineage_path: Path) -> dict[str, Any]:
    payload = read_json(feature_family_lineage_path)
    required = [
        "family_id",
        "ranking_name",
        "subset_size",
        "selected_feature_ids",
        "selected_feature_rows",
        "stage_name",
        "aggregated_head_effects",
        "aggregated_mlp_effects",
        "aggregated_neuron_group_effects",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(
            f"feature family lineage payload is missing required keys {missing}: {feature_family_lineage_path}"
        )
    return payload


def _load_feature_family_births_payload(feature_family_births_path: Path) -> dict[str, Any]:
    payload = read_json(feature_family_births_path)
    if not isinstance(payload.get("families"), list):
        raise ValueError(
            f"feature family births file must contain a 'families' list: {feature_family_births_path}"
        )
    return payload


def _load_feature_family_trace_payload(feature_family_trace_path: Path) -> dict[str, Any]:
    payload = read_json(feature_family_trace_path)
    required = ["family_id", "stage_name", "trace_subset", "trace_summary"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(
            f"feature family trace payload is missing required keys {missing}: {feature_family_trace_path}"
        )
    trace_subset = payload["trace_subset"]
    if not isinstance(trace_subset.get("feature_ids"), list) or not trace_subset["feature_ids"]:
        raise ValueError(
            f"feature family trace payload must contain a non-empty trace_subset.feature_ids list: {feature_family_trace_path}"
        )
    return payload


def _load_subset_trajectory_payload(subset_trajectory_path: Path) -> dict[str, Any]:
    payload = read_json(subset_trajectory_path)
    required = ["feature_ids", "rows", "subset_spec", "subset_size"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(
            f"subset trajectory payload is missing required keys {missing}: {subset_trajectory_path}"
        )
    if not isinstance(payload["rows"], list) or not payload["rows"]:
        raise ValueError(f"subset trajectory payload must contain a non-empty rows list: {subset_trajectory_path}")
    return payload


def _resolve_ranked_family_subset(
    *,
    rank_payload: dict[str, Any],
    ranking_name: str,
    subset_size: int,
) -> list[dict[str, Any]]:
    rankings = rank_payload["rankings"]
    if ranking_name not in rankings:
        raise KeyError(f"Unknown ranking_name {ranking_name} in feature family rank payload.")
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    ranked_rows = rankings[ranking_name]
    if subset_size > len(ranked_rows):
        raise ValueError(
            f"Requested subset size {subset_size} exceeds ranked family size {len(ranked_rows)}."
        )
    return ranked_rows[:subset_size]


def _render_feature_family_subpatch_plot(
    *,
    subset_results: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not subset_results:
        raise ValueError("subset_results must not be empty.")
    _, plt = _import_matplotlib()
    ordered = sorted(subset_results, key=lambda row: int(row["subset_size"]))
    x_values = [int(row["subset_size"]) for row in ordered]
    answer_values = [float(row["deltas"]["answer_accuracy"]) for row in ordered]
    heldout_values = [float(row["deltas"]["heldout_answer_accuracy"]) for row in ordered]
    structural_values = [float(row["deltas"]["structural_ood_answer_accuracy"]) for row in ordered]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_values, answer_values, marker="o", linewidth=2.0, label="answer Δ")
    ax.plot(x_values, heldout_values, marker="o", linewidth=2.0, label="heldout Δ")
    ax.plot(x_values, structural_values, marker="o", linewidth=2.0, label="structural OOD Δ")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Feature-family subset patch effects")
    ax.set_xlabel("Subset size")
    ax.set_ylabel("Patch delta")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _resolve_feature_family_birth_row(
    *,
    family_births_payload: dict[str, Any],
    family_id: int,
) -> dict[str, Any]:
    rows = [row for row in family_births_payload["families"] if int(row["family_id"]) == int(family_id)]
    if not rows:
        raise KeyError(f"Family {family_id} not found in feature family births payload.")
    if len(rows) != 1:
        raise RuntimeError(f"Expected one family-birth row for family {family_id}, found {len(rows)}.")
    return rows[0]


def _resolve_matching_family_subset_result(
    *,
    subpatch_payload: dict[str, Any],
    feature_ids: list[int],
) -> dict[str, Any]:
    target = sorted(int(feature_id) for feature_id in feature_ids)
    matches = [
        row
        for row in subpatch_payload["subset_results"]
        if sorted(int(feature_id) for feature_id in row["feature_ids"]) == target
    ]
    if not matches:
        raise KeyError(
            f"No subset result found for feature ids {feature_ids} in subpatch payload."
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one subset result for feature ids {feature_ids}, found {len(matches)}."
        )
    return matches[0]


def _render_feature_family_trace_plot(
    *,
    output_path: Path,
    family_id: int,
    birth_row: dict[str, Any],
    rank_payload: dict[str, Any],
    subpatch_payload: dict[str, Any],
    lineage_payload: dict[str, Any],
) -> Path:
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    metric_order = ["mean_activation", "active_fraction", "correctness_gap", "heldout_gap"]
    metric_labels = {
        "mean_activation": "Mean activation",
        "active_fraction": "Active fraction",
        "correctness_gap": "Correctness gap",
        "heldout_gap": "Heldout gap",
    }
    metric_colors = {
        "mean_activation": "#1f77b4",
        "active_fraction": "#ff7f0e",
        "correctness_gap": "#2ca02c",
        "heldout_gap": "#d62728",
    }
    birth_axis = axes[0, 0]
    birth_metrics = [metric_name for metric_name in metric_order if metric_name in birth_row["births"]]
    birth_steps = [float(birth_row["births"][metric_name]["birth_step"]) for metric_name in birth_metrics]
    birth_axis.scatter(
        birth_steps,
        list(range(len(birth_metrics))),
        c=[metric_colors[metric_name] for metric_name in birth_metrics],
        s=56,
    )
    birth_axis.set_yticks(range(len(birth_metrics)))
    birth_axis.set_yticklabels([metric_labels[metric_name] for metric_name in birth_metrics], fontsize=9)
    birth_axis.set_xlabel("Checkpoint step")
    birth_axis.set_title(f"Family {family_id} birth events")
    birth_axis.grid(alpha=0.25, axis="x")
    useful_birth_step = birth_row["useful_birth_step"]
    if useful_birth_step is not None:
        birth_axis.axvline(float(useful_birth_step), color="#111827", linewidth=1.4, linestyle="--")

    rank_axis = axes[0, 1]
    ranked_rows = rank_payload["rankings"]["by_useful_delta"][: min(6, len(rank_payload["rankings"]["by_useful_delta"]))]
    rank_axis.bar(
        [f"F{row['feature_id']}" for row in ranked_rows],
        [float(row["useful_delta"]) for row in ranked_rows],
        color="#2563eb",
    )
    rank_axis.set_title(f"Family {family_id} member useful deltas")
    rank_axis.set_ylabel("Heldout delta + correctness delta")
    rank_axis.set_xlabel("Feature")
    rank_axis.tick_params(axis="x", labelrotation=0)

    subpatch_axis = axes[1, 0]
    ordered_subset_rows = sorted(subpatch_payload["subset_results"], key=lambda row: int(row["subset_size"]))
    subset_sizes = [int(row["subset_size"]) for row in ordered_subset_rows]
    subpatch_axis.plot(
        subset_sizes,
        [float(row["deltas"]["answer_accuracy"]) for row in ordered_subset_rows],
        marker="o",
        linewidth=2.0,
        label="answer Δ",
    )
    subpatch_axis.plot(
        subset_sizes,
        [float(row["deltas"]["heldout_answer_accuracy"]) for row in ordered_subset_rows],
        marker="o",
        linewidth=2.0,
        label="heldout Δ",
    )
    subpatch_axis.plot(
        subset_sizes,
        [float(row["deltas"]["structural_ood_answer_accuracy"]) for row in ordered_subset_rows],
        marker="o",
        linewidth=2.0,
        label="structural OOD Δ",
    )
    subpatch_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    subpatch_axis.set_title("Subset patch effects")
    subpatch_axis.set_xlabel("Subset size")
    subpatch_axis.set_ylabel("Patch delta")
    subpatch_axis.grid(alpha=0.25)
    subpatch_axis.legend(fontsize=8)

    lineage_axis = axes[1, 1]
    lineage_items: list[tuple[str, float, str]] = []
    for row in lineage_payload["aggregated_head_effects"][:3]:
        lineage_items.append((f"H L{row['layer']}H{row['head']}", float(row["mean_abs_feature_shift_sum"]), "#8b5cf6"))
    for row in lineage_payload["aggregated_mlp_effects"][:3]:
        lineage_items.append((f"M L{row['layer']}", float(row["mean_abs_feature_shift_sum"]), "#059669"))
    for row in lineage_payload["aggregated_neuron_group_effects"][:3]:
        lineage_items.append(
            (
                f"N L{row['layer']}[{','.join(str(item) for item in row['neurons'])}]",
                float(row["mean_abs_feature_shift_sum"]),
                "#ea580c",
            )
        )
    if not lineage_items:
        raise ValueError("No lineage items available for feature-family trace plot.")
    labels = [item[0] for item in lineage_items]
    values = [item[1] for item in lineage_items]
    colors = [item[2] for item in lineage_items]
    lineage_axis.barh(labels, values, color=colors)
    lineage_axis.invert_yaxis()
    lineage_axis.set_title("Selected subset lineage")
    lineage_axis.set_xlabel("Summed abs feature shift")

    fig.suptitle(
        f"Feature-family trace | G{family_id} | stage={lineage_payload['stage_name']} | subset={lineage_payload['selected_feature_ids']}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def feature_family_subpatch(
    *,
    config_path: Path,
    probe_set_path: Path,
    basis_path: Path,
    feature_family_rank_path: Path,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    stage_name: str,
    ranking_name: str,
    subset_sizes: list[int],
    output_path: Path,
    device_name: str = "cpu",
    patch_mode: str = "replace",
) -> tuple[Path, Path]:
    rank_payload = _load_feature_family_rank_payload(feature_family_rank_path)
    if not subset_sizes:
        raise ValueError("subset_sizes must not be empty.")
    ordered_subset_sizes = sorted({int(value) for value in subset_sizes})
    if min(ordered_subset_sizes) <= 0:
        raise ValueError("subset_sizes must be positive.")

    subset_results: list[dict[str, Any]] = []
    for subset_size in ordered_subset_sizes:
        ranked_subset = _resolve_ranked_family_subset(
            rank_payload=rank_payload,
            ranking_name=ranking_name,
            subset_size=subset_size,
        )
        feature_ids = [int(row["feature_id"]) for row in ranked_subset]
        payload = _run_feature_patch(
            config_path=config_path,
            probe_set_path=probe_set_path,
            basis_path=basis_path,
            source_checkpoint_path=source_checkpoint_path,
            target_checkpoint_path=target_checkpoint_path,
            stage_name=stage_name,
            feature_ids=feature_ids,
            output_path=output_path,
            device_name=device_name,
            patch_mode=patch_mode,
        )
        subset_results.append(
            {
                "subset_size": subset_size,
                "feature_ids": feature_ids,
                "deltas": payload["deltas"],
                "baseline": payload["baseline"],
                "patched": payload["patched"],
                "reconstruction": payload["reconstruction"],
            }
        )

    payload = {
        "feature_family_rank_path": str(feature_family_rank_path),
        "family_id": int(rank_payload["family_id"]),
        "ranking_name": ranking_name,
        "stage_name": stage_name,
        "source_checkpoint_path": str(source_checkpoint_path),
        "target_checkpoint_path": str(target_checkpoint_path),
        "patch_mode": patch_mode,
        "subset_results": subset_results,
        "best_subset_by_heldout": max(
            subset_results,
            key=lambda row: (
                float(row["deltas"]["heldout_answer_accuracy"]),
                float(row["deltas"]["answer_accuracy"]),
                float(row["deltas"]["structural_ood_answer_accuracy"]),
            ),
        ),
        "best_subset_by_answer": max(
            subset_results,
            key=lambda row: (
                float(row["deltas"]["answer_accuracy"]),
                float(row["deltas"]["heldout_answer_accuracy"]),
                float(row["deltas"]["structural_ood_answer_accuracy"]),
            ),
        ),
        "best_subset_by_structural_ood": max(
            subset_results,
            key=lambda row: (
                float(row["deltas"]["structural_ood_answer_accuracy"]),
                float(row["deltas"]["heldout_answer_accuracy"]),
                float(row["deltas"]["answer_accuracy"]),
            ),
        ),
    }
    write_json(output_path, payload)
    plot_path = _render_feature_family_subpatch_plot(
        subset_results=subset_results,
        output_path=output_path.with_name(f"{output_path.stem}_deltas.svg"),
    )
    return output_path, plot_path


def _compute_lineage_effect_rows(
    *,
    spec: TrainSpec,
    model: torch.nn.Module,
    analysis_batches: list[dict[str, Any]],
    stage_name: str,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    sae: SparseAutoencoder,
    feature_ids: list[int],
    sweep_row: dict[str, Any] | None,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    baseline_features = _compute_feature_activations_for_model(
        model=model,
        batches=analysis_batches,
        stage_name=stage_name,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        sae=sae,
    )

    head_rows: list[dict[str, Any]] = []
    for layer_index in range(len(model.blocks)):
        for head_index in range(spec.model.n_heads):
            mask = torch.ones(spec.model.n_heads, device=device)
            mask[head_index] = 0.0
            ablated = _compute_feature_activations_for_model(
                model=model,
                batches=analysis_batches,
                stage_name=stage_name,
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
                sae=sae,
                head_mask={layer_index: mask},
            )
            for feature_id in feature_ids:
                head_rows.append(
                    {
                        "feature_id": feature_id,
                        "layer": layer_index,
                        "head": head_index,
                        "mean_activation_delta": float(
                            baseline_features[:, feature_id].mean().item() - ablated[:, feature_id].mean().item()
                        ),
                        "mean_abs_feature_shift": float(
                            (baseline_features[:, feature_id] - ablated[:, feature_id]).abs().mean().item()
                        ),
                    }
                )

    mlp_rows: list[dict[str, Any]] = []
    for layer_index in range(len(model.blocks)):
        ablated = _compute_feature_activations_for_model(
            model=model,
            batches=analysis_batches,
            stage_name=stage_name,
            normalization_mean=normalization_mean,
            normalization_std=normalization_std,
            sae=sae,
            mlp_mask={layer_index: 0.0},
        )
        for feature_id in feature_ids:
            mlp_rows.append(
                {
                    "feature_id": feature_id,
                    "layer": layer_index,
                    "mean_activation_delta": float(
                        baseline_features[:, feature_id].mean().item() - ablated[:, feature_id].mean().item()
                    ),
                    "mean_abs_feature_shift": float(
                        (baseline_features[:, feature_id] - ablated[:, feature_id]).abs().mean().item()
                    ),
                }
            )

    neuron_group_rows: list[dict[str, Any]] = []
    neuron_group_status = "not_requested"
    if sweep_row is not None:
        neuron_group_status = "computed_from_sweep_candidates"
        grouped: dict[int, list[int]] = defaultdict(list)
        for item in sweep_row["top_neurons_by_ablation"] + sweep_row["top_neurons_by_write"]:
            layer_index = int(item["layer"])
            neuron_index = int(item["neuron"])
            if neuron_index not in grouped[layer_index]:
                grouped[layer_index].append(neuron_index)
        for layer_index, neuron_indices in sorted(grouped.items()):
            mask = torch.ones(spec.model.d_ff, device=device)
            mask[torch.tensor(neuron_indices, device=device, dtype=torch.long)] = 0.0
            ablated = _compute_feature_activations_for_model(
                model=model,
                batches=analysis_batches,
                stage_name=stage_name,
                normalization_mean=normalization_mean,
                normalization_std=normalization_std,
                sae=sae,
                neuron_mask={layer_index: mask},
            )
            for feature_id in feature_ids:
                neuron_group_rows.append(
                    {
                        "feature_id": feature_id,
                        "layer": layer_index,
                        "neurons": neuron_indices,
                        "mean_activation_delta": float(
                            baseline_features[:, feature_id].mean().item() - ablated[:, feature_id].mean().item()
                        ),
                        "mean_abs_feature_shift": float(
                            (baseline_features[:, feature_id] - ablated[:, feature_id]).abs().mean().item()
                        ),
                    }
                )

    return head_rows, mlp_rows, neuron_group_rows, neuron_group_status


def _make_feature_lineage_graph(
    *,
    feature_ids: list[int],
    head_rows: list[dict[str, Any]],
    mlp_rows: list[dict[str, Any]],
    neuron_group_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph_nodes: list[dict[str, Any]] = []
    graph_edges: list[dict[str, Any]] = []
    for feature_id in feature_ids:
        graph_nodes.append({"id": f"feature:{feature_id}", "type": "feature", "label": f"F{feature_id}"})
    for row in head_rows:
        node_id = f"head:L{row['layer']}H{row['head']}"
        graph_nodes.append({"id": node_id, "type": "head", "label": f"L{row['layer']}H{row['head']}"})
        graph_edges.append(
            {
                "source": node_id,
                "target": f"feature:{row['feature_id']}",
                "type": "ablates",
                "score": row["mean_abs_feature_shift"],
                "score_type": "mean_abs_feature_shift",
            }
        )
    for row in mlp_rows:
        node_id = f"mlp:{row['layer']}"
        graph_nodes.append({"id": node_id, "type": "mlp_block", "label": f"MLP{row['layer']}"})
        graph_edges.append(
            {
                "source": node_id,
                "target": f"feature:{row['feature_id']}",
                "type": "ablates",
                "score": row["mean_abs_feature_shift"],
                "score_type": "mean_abs_feature_shift",
            }
        )
    for row in neuron_group_rows:
        node_id = f"neurons:{row['layer']}:{','.join(str(neuron) for neuron in row['neurons'])}"
        graph_nodes.append(
            {
                "id": node_id,
                "type": "neuron_group",
                "label": f"L{row['layer']}[{','.join(str(neuron) for neuron in row['neurons'])}]",
            }
        )
        graph_edges.append(
            {
                "source": node_id,
                "target": f"feature:{row['feature_id']}",
                "type": "ablates",
                "score": row["mean_abs_feature_shift"],
                "score_type": "mean_abs_feature_shift",
            }
        )
    return graph_nodes, graph_edges


def _aggregate_lineage_rows(
    *,
    rows: list[dict[str, Any]],
    key_builder: Any,
) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_builder(row)].append(row)
    aggregated_rows: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        feature_ids = sorted({int(row["feature_id"]) for row in group_rows})
        template = group_rows[0]
        aggregated = {
            "feature_ids": feature_ids,
            "feature_count": len(feature_ids),
            "mean_abs_feature_shift_sum": float(sum(float(row["mean_abs_feature_shift"]) for row in group_rows)),
            "mean_abs_feature_shift_mean": float(
                sum(float(row["mean_abs_feature_shift"]) for row in group_rows) / len(group_rows)
            ),
            "mean_activation_delta_sum": float(sum(float(row["mean_activation_delta"]) for row in group_rows)),
            "mean_activation_delta_mean": float(
                sum(float(row["mean_activation_delta"]) for row in group_rows) / len(group_rows)
            ),
        }
        if "head" in template:
            aggregated["layer"] = int(template["layer"])
            aggregated["head"] = int(template["head"])
        elif "neurons" in template:
            aggregated["layer"] = int(template["layer"])
            aggregated["neurons"] = list(template["neurons"])
        else:
            aggregated["layer"] = int(template["layer"])
        aggregated_rows.append(aggregated)
    aggregated_rows.sort(key=lambda row: float(row["mean_abs_feature_shift_sum"]), reverse=True)
    return aggregated_rows


def _render_feature_family_lineage_plots(
    *,
    output_path: Path,
    aggregated_head_rows: list[dict[str, Any]],
    aggregated_mlp_rows: list[dict[str, Any]],
    aggregated_neuron_group_rows: list[dict[str, Any]],
    top_k: int = 8,
) -> dict[str, Path]:
    plot_paths: dict[str, Path] = {}
    if aggregated_head_rows:
        ranked = aggregated_head_rows[:top_k]
        plot_paths["head_bar"] = _render_labeled_bar_plot(
            labels=[f"L{row['layer']}H{row['head']}" for row in ranked],
            values=[float(row["mean_abs_feature_shift_sum"]) for row in ranked],
            title="Family lineage: top heads",
            ylabel="Summed abs feature shift",
            xlabel="Head",
            output_path=output_path.with_name(f"{output_path.stem}_heads.svg"),
        )
    if aggregated_mlp_rows:
        ranked = aggregated_mlp_rows[:top_k]
        plot_paths["mlp_bar"] = _render_labeled_bar_plot(
            labels=[f"L{row['layer']}" for row in ranked],
            values=[float(row["mean_abs_feature_shift_sum"]) for row in ranked],
            title="Family lineage: top MLPs",
            ylabel="Summed abs feature shift",
            xlabel="MLP block",
            output_path=output_path.with_name(f"{output_path.stem}_mlps.svg"),
        )
    if aggregated_neuron_group_rows:
        ranked = aggregated_neuron_group_rows[:top_k]
        plot_paths["neuron_group_bar"] = _render_labeled_bar_plot(
            labels=[f"L{row['layer']}[{','.join(str(item) for item in row['neurons'])}]" for row in ranked],
            values=[float(row["mean_abs_feature_shift_sum"]) for row in ranked],
            title="Family lineage: top neuron groups",
            ylabel="Summed abs feature shift",
            xlabel="Neuron group",
            output_path=output_path.with_name(f"{output_path.stem}_neuron_groups.svg"),
        )
    return plot_paths


def _load_sweep_row_by_step(sweep_metrics_path: Path, step: int) -> dict[str, Any]:
    rows = [row for row in iter_jsonl(sweep_metrics_path) if int(row["step"]) == step]
    if not rows:
        raise KeyError(f"No sweep row found for step {step} in {sweep_metrics_path}")
    if len(rows) != 1:
        raise RuntimeError(f"Expected one sweep row for step {step}, found {len(rows)}")
    return rows[0]


def _compute_feature_activations_for_model(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    stage_name: str,
    normalization_mean: torch.Tensor,
    normalization_std: torch.Tensor,
    sae: SparseAutoencoder,
    head_mask: dict[int, torch.Tensor] | None = None,
    mlp_mask: dict[int, float | torch.Tensor] | None = None,
    neuron_mask: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    encoded_rows: list[torch.Tensor] = []
    for batch in batches:
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_residual_streams=True,
            head_mask=head_mask,
            mlp_mask=mlp_mask,
            neuron_mask=neuron_mask,
        )
        if outputs.residual_streams is None:
            raise RuntimeError("Feature lineage requires residual_streams in model output.")
        stage_state = outputs.residual_streams[stage_name]
        _, _, metadata = extract_answer_logits(outputs.logits, batch)
        rows = metadata["rows"]
        prediction_positions = metadata["prediction_positions"]
        selected = stage_state[rows, prediction_positions, :]
        normalized = _normalize_activations(selected, normalization_mean, normalization_std)
        encoded_rows.append(_encode_features(sae, normalized))
    return torch.cat(encoded_rows, dim=0)


def feature_lineage(
    *,
    config_path: Path,
    probe_set_path: Path,
    basis_path: Path,
    checkpoint_path: Path,
    feature_ids: list[int],
    output_path: Path,
    device_name: str = "cpu",
    sweep_metrics_path: Path | None = None,
) -> tuple[Path, Path]:
    if not feature_ids:
        raise ValueError("feature_ids must not be empty.")
    spec, vocab, analysis_batches, _, _, device = _load_probe_batches(
        config_path=config_path,
        probe_set_path=probe_set_path,
        device_name=device_name,
    )
    basis = _load_shared_basis(basis_path, device)
    stage_name = str(basis["stage_name"])
    normalization_mean = basis["normalization_mean"]
    normalization_std = basis["normalization_std"]
    sae: SparseAutoencoder = basis["sae"]

    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(spec.model, len(vocab.tokens), device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()
    sweep_row = None if sweep_metrics_path is None else _load_sweep_row_by_step(sweep_metrics_path, int(checkpoint["step"]))
    head_rows, mlp_rows, neuron_group_rows, neuron_group_status = _compute_lineage_effect_rows(
        spec=spec,
        model=model,
        analysis_batches=analysis_batches,
        stage_name=stage_name,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        sae=sae,
        feature_ids=feature_ids,
        sweep_row=sweep_row,
        device=device,
    )
    graph_nodes, graph_edges = _make_feature_lineage_graph(
        feature_ids=feature_ids,
        head_rows=head_rows,
        mlp_rows=mlp_rows,
        neuron_group_rows=neuron_group_rows,
    )
    payload = {
        "basis_id": str(basis["basis_id"]),
        "stage_name": stage_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "feature_ids": feature_ids,
        "neuron_group_status": neuron_group_status,
        "head_effects": sorted(head_rows, key=lambda row: float(row["mean_abs_feature_shift"]), reverse=True),
        "mlp_effects": sorted(mlp_rows, key=lambda row: float(row["mean_abs_feature_shift"]), reverse=True),
        "neuron_group_effects": sorted(
            neuron_group_rows,
            key=lambda row: float(row["mean_abs_feature_shift"]),
            reverse=True,
        ),
    }
    graph_path = output_path.with_name(f"{output_path.stem}_graph.json")
    write_json(output_path, payload)
    write_json(graph_path, {"nodes": graph_nodes, "edges": graph_edges})
    return output_path, graph_path


def feature_family_lineage(
    *,
    config_path: Path,
    probe_set_path: Path,
    basis_path: Path,
    feature_family_rank_path: Path,
    checkpoint_path: Path,
    ranking_name: str,
    subset_size: int,
    output_path: Path,
    device_name: str = "cpu",
    sweep_metrics_path: Path,
) -> tuple[Path, Path, dict[str, Path]]:
    rank_payload = _load_feature_family_rank_payload(feature_family_rank_path)
    ranked_subset = _resolve_ranked_family_subset(
        rank_payload=rank_payload,
        ranking_name=ranking_name,
        subset_size=subset_size,
    )
    feature_ids = [int(row["feature_id"]) for row in ranked_subset]

    spec, vocab, analysis_batches, _, _, device = _load_probe_batches(
        config_path=config_path,
        probe_set_path=probe_set_path,
        device_name=device_name,
    )
    basis = _load_shared_basis(basis_path, device)
    stage_name = str(basis["stage_name"])
    normalization_mean = basis["normalization_mean"]
    normalization_std = basis["normalization_std"]
    sae: SparseAutoencoder = basis["sae"]

    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(spec.model, len(vocab.tokens), device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()
    sweep_row = _load_sweep_row_by_step(sweep_metrics_path, int(checkpoint["step"]))
    head_rows, mlp_rows, neuron_group_rows, neuron_group_status = _compute_lineage_effect_rows(
        spec=spec,
        model=model,
        analysis_batches=analysis_batches,
        stage_name=stage_name,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        sae=sae,
        feature_ids=feature_ids,
        sweep_row=sweep_row,
        device=device,
    )
    aggregated_head_rows = _aggregate_lineage_rows(
        rows=head_rows,
        key_builder=lambda row: (int(row["layer"]), int(row["head"])),
    )
    aggregated_mlp_rows = _aggregate_lineage_rows(
        rows=mlp_rows,
        key_builder=lambda row: int(row["layer"]),
    )
    aggregated_neuron_group_rows = _aggregate_lineage_rows(
        rows=neuron_group_rows,
        key_builder=lambda row: (int(row["layer"]), tuple(int(item) for item in row["neurons"])),
    )

    feature_graph_nodes, feature_graph_edges = _make_feature_lineage_graph(
        feature_ids=feature_ids,
        head_rows=head_rows,
        mlp_rows=mlp_rows,
        neuron_group_rows=neuron_group_rows,
    )
    family_node_id = f"family_subset:{rank_payload['family_id']}:{ranking_name}:{subset_size}"
    graph_nodes = feature_graph_nodes + [
        {
            "id": family_node_id,
            "type": "feature_family_subset",
            "label": f"G{rank_payload['family_id']} {ranking_name} top {subset_size}",
        }
    ]
    graph_edges = list(feature_graph_edges)
    for feature_id in feature_ids:
        graph_edges.append(
            {
                "source": family_node_id,
                "target": f"feature:{feature_id}",
                "type": "contains",
                "score": 1.0,
                "score_type": "membership",
            }
        )
    for row in aggregated_head_rows:
        graph_edges.append(
            {
                "source": f"head:L{row['layer']}H{row['head']}",
                "target": family_node_id,
                "type": "aggregated_ablation",
                "score": row["mean_abs_feature_shift_sum"],
                "score_type": "mean_abs_feature_shift_sum",
            }
        )
    for row in aggregated_mlp_rows:
        graph_edges.append(
            {
                "source": f"mlp:{row['layer']}",
                "target": family_node_id,
                "type": "aggregated_ablation",
                "score": row["mean_abs_feature_shift_sum"],
                "score_type": "mean_abs_feature_shift_sum",
            }
        )
    for row in aggregated_neuron_group_rows:
        graph_edges.append(
            {
                "source": f"neurons:{row['layer']}:{','.join(str(neuron) for neuron in row['neurons'])}",
                "target": family_node_id,
                "type": "aggregated_ablation",
                "score": row["mean_abs_feature_shift_sum"],
                "score_type": "mean_abs_feature_shift_sum",
            }
        )

    payload = {
        "feature_family_rank_path": str(feature_family_rank_path),
        "family_id": int(rank_payload["family_id"]),
        "ranking_name": ranking_name,
        "subset_size": subset_size,
        "selected_feature_rows": ranked_subset,
        "selected_feature_ids": feature_ids,
        "basis_id": str(basis["basis_id"]),
        "stage_name": stage_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "sweep_metrics_path": str(sweep_metrics_path),
        "neuron_group_status": neuron_group_status,
        "head_effects": sorted(head_rows, key=lambda row: float(row["mean_abs_feature_shift"]), reverse=True),
        "mlp_effects": sorted(mlp_rows, key=lambda row: float(row["mean_abs_feature_shift"]), reverse=True),
        "neuron_group_effects": sorted(neuron_group_rows, key=lambda row: float(row["mean_abs_feature_shift"]), reverse=True),
        "aggregated_head_effects": aggregated_head_rows,
        "aggregated_mlp_effects": aggregated_mlp_rows,
        "aggregated_neuron_group_effects": aggregated_neuron_group_rows,
    }
    graph_path = output_path.with_name(f"{output_path.stem}_graph.json")
    write_json(output_path, payload)
    write_json(graph_path, {"nodes": graph_nodes, "edges": graph_edges})
    plot_paths = _render_feature_family_lineage_plots(
        output_path=output_path,
        aggregated_head_rows=aggregated_head_rows,
        aggregated_mlp_rows=aggregated_mlp_rows,
        aggregated_neuron_group_rows=aggregated_neuron_group_rows,
    )
    return output_path, graph_path, plot_paths


def feature_family_trace(
    *,
    feature_family_births_path: Path,
    feature_family_rank_path: Path,
    feature_family_subpatch_path: Path,
    feature_family_lineage_path: Path,
    output_path: Path,
) -> tuple[Path, dict[str, Path]]:
    family_births_payload = _load_feature_family_births_payload(feature_family_births_path)
    rank_payload = _load_feature_family_rank_payload(feature_family_rank_path)
    subpatch_payload = _load_feature_family_subpatch_payload(feature_family_subpatch_path)
    lineage_payload = _load_feature_family_lineage_payload(feature_family_lineage_path)

    family_id = int(rank_payload["family_id"])
    if int(subpatch_payload["family_id"]) != family_id:
        raise ValueError(
            f"Family mismatch between rank ({family_id}) and subpatch ({subpatch_payload['family_id']})."
        )
    if int(lineage_payload["family_id"]) != family_id:
        raise ValueError(
            f"Family mismatch between rank ({family_id}) and lineage ({lineage_payload['family_id']})."
        )
    if str(subpatch_payload["ranking_name"]) != str(lineage_payload["ranking_name"]):
        raise ValueError(
            "Subpatch ranking_name does not match lineage ranking_name: "
            f"{subpatch_payload['ranking_name']} vs {lineage_payload['ranking_name']}"
        )
    if str(subpatch_payload["stage_name"]) != str(lineage_payload["stage_name"]):
        raise ValueError(
            f"Stage mismatch between subpatch ({subpatch_payload['stage_name']}) and lineage ({lineage_payload['stage_name']})."
        )

    family_birth_row = _resolve_feature_family_birth_row(
        family_births_payload=family_births_payload,
        family_id=family_id,
    )
    family_summary = rank_payload["family_summary"]
    if sorted(int(feature_id) for feature_id in family_summary["member_feature_ids"]) != sorted(
        int(feature_id) for feature_id in family_birth_row["member_feature_ids"]
    ):
        raise ValueError("Family membership differs between rank payload and family births payload.")

    trace_feature_ids = [int(feature_id) for feature_id in lineage_payload["selected_feature_ids"]]
    matched_subset_result = _resolve_matching_family_subset_result(
        subpatch_payload=subpatch_payload,
        feature_ids=trace_feature_ids,
    )
    if int(matched_subset_result["subset_size"]) != int(lineage_payload["subset_size"]):
        raise ValueError(
            f"Subset size mismatch between lineage ({lineage_payload['subset_size']}) and subpatch result ({matched_subset_result['subset_size']})."
        )

    member_rows_by_feature = {
        int(row["feature_id"]): row
        for row in rank_payload["member_rows"]
    }
    selected_member_rows: list[dict[str, Any]] = []
    for feature_id in trace_feature_ids:
        member_row = member_rows_by_feature.get(feature_id)
        if member_row is None:
            raise KeyError(f"Selected feature {feature_id} missing from family rank payload member_rows.")
        selected_member_rows.append(member_row)

    payload = {
        "feature_family_births_path": str(feature_family_births_path),
        "feature_family_rank_path": str(feature_family_rank_path),
        "feature_family_subpatch_path": str(feature_family_subpatch_path),
        "feature_family_lineage_path": str(feature_family_lineage_path),
        "family_id": family_id,
        "stage_name": str(lineage_payload["stage_name"]),
        "ranking_name": str(lineage_payload["ranking_name"]),
        "source_step": int(rank_payload["source_step"]),
        "target_step": int(rank_payload["target_step"]),
        "family_summary": family_summary,
        "family_birth": family_birth_row,
        "trace_subset": {
            "subset_size": int(lineage_payload["subset_size"]),
            "feature_ids": trace_feature_ids,
            "member_rows": selected_member_rows,
            "patch_result": matched_subset_result,
            "lineage": {
                "aggregated_head_effects": lineage_payload["aggregated_head_effects"],
                "aggregated_mlp_effects": lineage_payload["aggregated_mlp_effects"],
                "aggregated_neuron_group_effects": lineage_payload["aggregated_neuron_group_effects"],
                "neuron_group_status": lineage_payload["neuron_group_status"],
                "checkpoint_step": int(lineage_payload["checkpoint_step"]),
                "checkpoint_path": str(lineage_payload["checkpoint_path"]),
            },
        },
        "family_rankings": {
            "top_by_useful_delta": rank_payload["rankings"]["by_useful_delta"][:8],
            "top_by_heldout_gap_delta": rank_payload["rankings"]["by_heldout_gap_delta"][:8],
            "top_by_correctness_gap_delta": rank_payload["rankings"]["by_correctness_gap_delta"][:8],
        },
        "subpatch_summary": {
            "subset_results": subpatch_payload["subset_results"],
            "best_subset_by_heldout": subpatch_payload["best_subset_by_heldout"],
            "best_subset_by_answer": subpatch_payload["best_subset_by_answer"],
            "best_subset_by_structural_ood": subpatch_payload["best_subset_by_structural_ood"],
        },
        "trace_summary": {
            "family_birth_step": family_birth_row["birth_step"],
            "family_useful_birth_step": family_birth_row["useful_birth_step"],
            "selected_subset_size": int(lineage_payload["subset_size"]),
            "selected_subset_feature_ids": trace_feature_ids,
            "selected_subset_answer_delta": float(matched_subset_result["deltas"]["answer_accuracy"]),
            "selected_subset_heldout_delta": float(matched_subset_result["deltas"]["heldout_answer_accuracy"]),
            "selected_subset_structural_ood_delta": float(
                matched_subset_result["deltas"]["structural_ood_answer_accuracy"]
            ),
            "top_head": None
            if not lineage_payload["aggregated_head_effects"]
            else lineage_payload["aggregated_head_effects"][0],
            "top_mlp": None
            if not lineage_payload["aggregated_mlp_effects"]
            else lineage_payload["aggregated_mlp_effects"][0],
            "top_neuron_group": None
            if not lineage_payload["aggregated_neuron_group_effects"]
            else lineage_payload["aggregated_neuron_group_effects"][0],
        },
    }
    write_json(output_path, payload)
    plot_paths = {
        "trace_plot": _render_feature_family_trace_plot(
            output_path=output_path.with_name(f"{output_path.stem}_summary.svg"),
            family_id=family_id,
            birth_row=family_birth_row,
            rank_payload=rank_payload,
            subpatch_payload=subpatch_payload,
            lineage_payload=lineage_payload,
        )
    }
    return output_path, plot_paths


def _resolve_subset_feature_ids(
    *,
    feature_ids: list[int] | None = None,
    feature_family_rank_path: Path | None = None,
    ranking_name: str | None = None,
    subset_size: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    if feature_ids is not None and feature_family_rank_path is not None:
        raise ValueError("Provide either feature_ids or feature_family_rank_path, not both.")
    if feature_ids is not None:
        resolved = sorted({int(feature_id) for feature_id in feature_ids})
        if not resolved:
            raise ValueError("feature_ids must not be empty.")
        return resolved, {
            "source": "explicit_features",
            "feature_ids": resolved,
        }
    if feature_family_rank_path is None:
        raise ValueError("One of feature_ids or feature_family_rank_path must be provided.")
    if ranking_name is None or subset_size is None:
        raise ValueError("ranking_name and subset_size are required when using feature_family_rank_path.")
    rank_payload = _load_feature_family_rank_payload(feature_family_rank_path)
    ranked_subset = _resolve_ranked_family_subset(
        rank_payload=rank_payload,
        ranking_name=ranking_name,
        subset_size=subset_size,
    )
    resolved = [int(row["feature_id"]) for row in ranked_subset]
    return resolved, {
        "source": "feature_family_rank",
        "feature_family_rank_path": str(feature_family_rank_path),
        "family_id": int(rank_payload["family_id"]),
        "ranking_name": ranking_name,
        "subset_size": int(subset_size),
        "selected_feature_rows": ranked_subset,
        "feature_ids": resolved,
    }


def _render_subset_trajectory_plot(
    *,
    subset_rows: list[dict[str, Any]],
    output_path: Path,
    subset_label: str,
) -> Path:
    if not subset_rows:
        raise ValueError("subset_rows must not be empty.")
    _, plt = _import_matplotlib()
    ordered = sorted(subset_rows, key=lambda row: int(row["step"]))
    steps = [int(row["step"]) for row in ordered]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(steps, [float(row["mean_activation_mean"]) for row in ordered], linewidth=2.0, label="Mean activation")
    axes[0].plot(steps, [float(row["active_fraction_mean"]) for row in ordered], linewidth=2.0, label="Active fraction")
    axes[0].set_ylabel("Mean value")
    axes[0].set_title(f"Subset trajectory | {subset_label}")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(steps, [float(row["correctness_gap_mean"]) for row in ordered], linewidth=2.0, label="Correctness gap")
    axes[1].plot(steps, [float(row["heldout_gap_mean"]) for row in ordered], linewidth=2.0, label="Heldout gap")
    axes[1].plot(steps, [float(row["structural_ood_gap_mean"]) for row in ordered], linewidth=2.0, label="Structural OOD gap")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[1].set_xlabel("Checkpoint step")
    axes[1].set_ylabel("Gap")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def subset_trajectory(
    *,
    trajectories_path: Path,
    output_path: Path,
    feature_ids: list[int] | None = None,
    feature_family_rank_path: Path | None = None,
    ranking_name: str | None = None,
    subset_size: int | None = None,
) -> tuple[Path, Path]:
    resolved_feature_ids, subset_spec = _resolve_subset_feature_ids(
        feature_ids=feature_ids,
        feature_family_rank_path=feature_family_rank_path,
        ranking_name=ranking_name,
        subset_size=subset_size,
    )
    rows = list(iter_jsonl(trajectories_path))
    if not rows:
        raise ValueError(f"No rows found in trajectories file: {trajectories_path}")
    rows_by_feature: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_feature[int(row["feature_id"])].append(row)
    missing = [feature_id for feature_id in resolved_feature_ids if feature_id not in rows_by_feature]
    if missing:
        raise KeyError(f"Subset features {missing} are missing from trajectories file: {trajectories_path}")
    for feature_rows in rows_by_feature.values():
        feature_rows.sort(key=lambda item: int(item["step"]))

    ordered_feature_rows = [rows_by_feature[feature_id] for feature_id in resolved_feature_ids]
    sorted_steps = [int(row["step"]) for row in ordered_feature_rows[0]]
    for feature_id, feature_rows in zip(resolved_feature_ids, ordered_feature_rows, strict=True):
        feature_steps = [int(row["step"]) for row in feature_rows]
        if feature_steps != sorted_steps:
            raise ValueError(
                f"Feature {feature_id} does not have a complete trajectory over the same steps as the subset."
            )

    subset_rows: list[dict[str, Any]] = []
    for step_index, step in enumerate(sorted_steps):
        current_rows = [feature_rows[step_index] for feature_rows in ordered_feature_rows]
        metric_names = [
            "mean_activation",
            "active_fraction",
            "correctness_gap",
            "heldout_gap",
            "structural_ood_gap",
            "margin_correlation",
            "answer_direction_alignment",
        ]
        row: dict[str, Any] = {
            "step": int(step),
            "feature_ids": resolved_feature_ids,
            "subset_size": len(resolved_feature_ids),
        }
        for metric_name in metric_names:
            values = [float(item[metric_name]) for item in current_rows]
            row[f"{metric_name}_mean"] = sum(values) / len(values)
            row[f"{metric_name}_max"] = max(values)
            row[f"{metric_name}_min"] = min(values)
        split_names = sorted(
            {
                split_name
                for item in current_rows
                for split_name in item["split_mean_activation"]
            }
        )
        row["split_mean_activation_mean"] = {
            split_name: sum(float(item["split_mean_activation"].get(split_name, 0.0)) for item in current_rows) / len(current_rows)
            for split_name in split_names
        }
        subset_rows.append(row)

    payload = {
        "trajectories_path": str(trajectories_path),
        "subset_spec": subset_spec,
        "feature_ids": resolved_feature_ids,
        "subset_size": len(resolved_feature_ids),
        "rows": subset_rows,
    }
    write_json(output_path, payload)
    subset_label = ",".join(f"F{feature_id}" for feature_id in resolved_feature_ids)
    plot_path = _render_subset_trajectory_plot(
        subset_rows=subset_rows,
        output_path=output_path.with_name(f"{output_path.stem}_summary.svg"),
        subset_label=subset_label,
    )
    return output_path, plot_path


def subset_birth_analyze(
    *,
    subset_trajectory_path: Path,
    output_path: Path,
    thresholds: dict[str, float],
    delta_threshold: float,
    window: int,
) -> tuple[Path, Path]:
    payload = read_json(subset_trajectory_path)
    subset_rows = payload.get("rows")
    if not isinstance(subset_rows, list) or not subset_rows:
        raise ValueError(f"subset trajectory payload must contain a non-empty 'rows' list: {subset_trajectory_path}")
    births: dict[str, Any] = {}
    final_metrics: dict[str, float] = {}
    for metric_name, threshold in thresholds.items():
        curve_metric_name = f"{metric_name}_mean"
        if curve_metric_name not in subset_rows[0]:
            raise KeyError(f"Metric {curve_metric_name} not found in subset trajectory rows: {subset_trajectory_path}")
        birth = _birth_event(
            rows=subset_rows,
            metric_name=curve_metric_name,
            threshold=float(threshold),
            delta_threshold=delta_threshold,
            window=window,
        )
        if birth is not None:
            births[metric_name] = birth
        final_metrics[metric_name] = float(subset_rows[-1][curve_metric_name])
    useful_metric_names = {"correctness_gap", "heldout_gap"}
    useful_birth_step = None
    if (
        all(metric_name in births for metric_name in useful_metric_names)
        and all(final_metrics[metric_name] > 0.0 for metric_name in useful_metric_names)
    ):
        useful_birth_step = max(int(births[metric_name]["birth_step"]) for metric_name in useful_metric_names)
    birth_payload = {
        "subset_trajectory_path": str(subset_trajectory_path),
        "subset_spec": payload["subset_spec"],
        "feature_ids": payload["feature_ids"],
        "subset_size": int(payload["subset_size"]),
        "birth_step": None if not births else min(int(item["birth_step"]) for item in births.values()),
        "useful_birth_step": useful_birth_step,
        "births": births,
        "final_metrics": final_metrics,
        "thresholds": thresholds,
        "delta_threshold": delta_threshold,
        "window": window,
    }
    write_json(output_path, birth_payload)
    plot_path = _render_birth_plot(
        birth_rows=[birth_payload] if birth_payload["birth_step"] is not None else [],
        output_path=output_path.with_name(f"{output_path.stem}_raster.svg"),
        entity_id_key="subset_size",
        title="Subset birth raster",
        ylabel="Subset",
    )
    return output_path, plot_path


def _render_subset_competition_plot(
    *,
    output_path: Path,
    payload: dict[str, Any],
) -> Path:
    _, plt = _import_matplotlib()
    labels = ["A", "B", "A∪B", "Interaction"]
    answer_values = [
        float(payload["subset_a"]["patch"]["deltas"]["answer_accuracy"]),
        float(payload["subset_b"]["patch"]["deltas"]["answer_accuracy"]),
        float(payload["union_subset"]["patch"]["deltas"]["answer_accuracy"]),
        float(payload["interaction"]["answer_accuracy"]),
    ]
    heldout_values = [
        float(payload["subset_a"]["patch"]["deltas"]["heldout_answer_accuracy"]),
        float(payload["subset_b"]["patch"]["deltas"]["heldout_answer_accuracy"]),
        float(payload["union_subset"]["patch"]["deltas"]["heldout_answer_accuracy"]),
        float(payload["interaction"]["heldout_answer_accuracy"]),
    ]
    structural_values = [
        float(payload["subset_a"]["patch"]["deltas"]["structural_ood_answer_accuracy"]),
        float(payload["subset_b"]["patch"]["deltas"]["structural_ood_answer_accuracy"]),
        float(payload["union_subset"]["patch"]["deltas"]["structural_ood_answer_accuracy"]),
        float(payload["interaction"]["structural_ood_answer_accuracy"]),
    ]
    x = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([value - width for value in x], answer_values, width=width, label="answer Δ")
    ax.bar(x, heldout_values, width=width, label="heldout Δ")
    ax.bar([value + width for value in x], structural_values, width=width, label="structural OOD Δ")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Patch delta")
    ax.set_title("Subset competition")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def subset_competition(
    *,
    config_path: Path,
    probe_set_path: Path,
    basis_path: Path,
    source_checkpoint_path: Path,
    target_checkpoint_path: Path,
    stage_name: str,
    output_path: Path,
    subset_a_feature_ids: list[int] | None = None,
    subset_a_feature_family_rank_path: Path | None = None,
    subset_a_ranking_name: str | None = None,
    subset_a_subset_size: int | None = None,
    subset_b_feature_ids: list[int] | None = None,
    subset_b_feature_family_rank_path: Path | None = None,
    subset_b_ranking_name: str | None = None,
    subset_b_subset_size: int | None = None,
    device_name: str = "cpu",
    patch_mode: str = "replace",
) -> tuple[Path, Path]:
    subset_a_ids, subset_a_spec = _resolve_subset_feature_ids(
        feature_ids=subset_a_feature_ids,
        feature_family_rank_path=subset_a_feature_family_rank_path,
        ranking_name=subset_a_ranking_name,
        subset_size=subset_a_subset_size,
    )
    subset_b_ids, subset_b_spec = _resolve_subset_feature_ids(
        feature_ids=subset_b_feature_ids,
        feature_family_rank_path=subset_b_feature_family_rank_path,
        ranking_name=subset_b_ranking_name,
        subset_size=subset_b_subset_size,
    )
    union_ids = sorted({*subset_a_ids, *subset_b_ids})

    patch_a = _run_feature_patch(
        config_path=config_path,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        source_checkpoint_path=source_checkpoint_path,
        target_checkpoint_path=target_checkpoint_path,
        stage_name=stage_name,
        feature_ids=subset_a_ids,
        output_path=output_path,
        device_name=device_name,
        patch_mode=patch_mode,
    )
    patch_b = _run_feature_patch(
        config_path=config_path,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        source_checkpoint_path=source_checkpoint_path,
        target_checkpoint_path=target_checkpoint_path,
        stage_name=stage_name,
        feature_ids=subset_b_ids,
        output_path=output_path,
        device_name=device_name,
        patch_mode=patch_mode,
    )
    patch_union = _run_feature_patch(
        config_path=config_path,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        source_checkpoint_path=source_checkpoint_path,
        target_checkpoint_path=target_checkpoint_path,
        stage_name=stage_name,
        feature_ids=union_ids,
        output_path=output_path,
        device_name=device_name,
        patch_mode=patch_mode,
    )

    interaction = {
        "answer_accuracy": float(patch_union["deltas"]["answer_accuracy"] - patch_a["deltas"]["answer_accuracy"] - patch_b["deltas"]["answer_accuracy"]),
        "heldout_answer_accuracy": float(
            patch_union["deltas"]["heldout_answer_accuracy"]
            - patch_a["deltas"]["heldout_answer_accuracy"]
            - patch_b["deltas"]["heldout_answer_accuracy"]
        ),
        "structural_ood_answer_accuracy": float(
            patch_union["deltas"]["structural_ood_answer_accuracy"]
            - patch_a["deltas"]["structural_ood_answer_accuracy"]
            - patch_b["deltas"]["structural_ood_answer_accuracy"]
        ),
    }

    payload = {
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "basis_path": str(basis_path),
        "stage_name": stage_name,
        "source_checkpoint_path": str(source_checkpoint_path),
        "target_checkpoint_path": str(target_checkpoint_path),
        "patch_mode": patch_mode,
        "subset_a": {
            "spec": subset_a_spec,
            "feature_ids": subset_a_ids,
            "patch": patch_a,
        },
        "subset_b": {
            "spec": subset_b_spec,
            "feature_ids": subset_b_ids,
            "patch": patch_b,
        },
        "union_subset": {
            "feature_ids": union_ids,
            "patch": patch_union,
        },
        "interaction": interaction,
    }
    write_json(output_path, payload)
    plot_path = _render_subset_competition_plot(
        output_path=output_path.with_name(f"{output_path.stem}_summary.svg"),
        payload=payload,
    )
    return output_path, plot_path


def _resolve_checkpoint_paths_by_step(checkpoint_dir: Path) -> dict[int, Path]:
    checkpoint_paths = _resolve_checkpoints(checkpoint_paths=None, checkpoint_dir=checkpoint_dir)
    paths_by_step: dict[int, Path] = {}
    for checkpoint_path in checkpoint_paths:
        step_token = checkpoint_path.stem.split("_")[-1]
        if not step_token.isdigit():
            raise ValueError(f"Checkpoint filename does not end with an integer step: {checkpoint_path}")
        step = int(step_token)
        if step in paths_by_step:
            raise ValueError(f"Duplicate checkpoint step {step} in {checkpoint_dir}")
        paths_by_step[step] = checkpoint_path
    return paths_by_step


def _load_sweep_rows_by_step(sweep_metrics_path: Path) -> dict[int, dict[str, Any]]:
    rows = list(iter_jsonl(sweep_metrics_path))
    if not rows:
        raise ValueError(f"No rows found in sweep metrics file: {sweep_metrics_path}")
    rows_by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = int(row["step"])
        if step in rows_by_step:
            raise ValueError(f"Duplicate sweep step {step} in {sweep_metrics_path}")
        rows_by_step[step] = row
    return rows_by_step


def _extract_checkpoint_n_heads(checkpoint_payload: dict[str, Any]) -> int:
    config = checkpoint_payload.get("config")
    if not isinstance(config, dict):
        raise KeyError("Checkpoint payload is missing dict config.")
    model_config = config.get("model")
    if not isinstance(model_config, dict):
        train_spec = config.get("train_spec")
        if isinstance(train_spec, dict):
            model_config = train_spec.get("model")
    if not isinstance(model_config, dict):
        raise KeyError("Checkpoint payload config is missing dict model config.")
    n_heads = model_config.get("n_heads")
    if not isinstance(n_heads, int) or n_heads <= 0:
        raise ValueError(f"Checkpoint model config has invalid n_heads: {n_heads}")
    return n_heads


def _load_checkpoint_state_for_update(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_payload = load_checkpoint(checkpoint_path, torch.device("cpu"))
    step = int(checkpoint_payload["step"])
    model_state = _migrate_legacy_feedforward_state_dict(checkpoint_payload["model_state"])
    return {
        "path": checkpoint_path,
        "step": step,
        "checkpoint": checkpoint_payload,
        "model_state": model_state,
    }


def _require_matching_state_keys(
    previous_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
) -> list[str]:
    previous_keys = sorted(previous_state)
    current_keys = sorted(current_state)
    if previous_keys != current_keys:
        raise ValueError("Checkpoint model_state keys do not match for update comparison.")
    return previous_keys


def _compute_delta_l2_norm_for_keys(
    *,
    previous_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    keys: list[str],
) -> float:
    total = 0.0
    for key in keys:
        delta = current_state[key].float() - previous_state[key].float()
        total += float(torch.sum(delta * delta).item())
    return total ** 0.5


def _compute_global_update_stats(
    *,
    previous_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    previous_previous_state: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    keys = _require_matching_state_keys(previous_state, current_state)
    update_sq_sum = 0.0
    previous_param_sq_sum = 0.0
    cosine_dot = 0.0
    previous_update_sq_sum = 0.0
    for key in keys:
        previous_tensor = previous_state[key].float()
        current_tensor = current_state[key].float()
        delta = current_tensor - previous_tensor
        update_sq_sum += float(torch.sum(delta * delta).item())
        previous_param_sq_sum += float(torch.sum(previous_tensor * previous_tensor).item())
        if previous_previous_state is not None:
            previous_delta = previous_tensor - previous_previous_state[key].float()
            cosine_dot += float(torch.sum(delta * previous_delta).item())
            previous_update_sq_sum += float(torch.sum(previous_delta * previous_delta).item())
    update_l2_norm = update_sq_sum ** 0.5
    previous_param_l2_norm = previous_param_sq_sum ** 0.5
    relative_update_l2_norm = update_l2_norm / previous_param_l2_norm if previous_param_l2_norm > 0.0 else None
    cosine_with_previous_update = None
    if previous_previous_state is not None and update_sq_sum > 0.0 and previous_update_sq_sum > 0.0:
        cosine_with_previous_update = cosine_dot / ((update_sq_sum ** 0.5) * (previous_update_sq_sum ** 0.5))
    return {
        "global_update_l2_norm": update_l2_norm,
        "previous_param_l2_norm": previous_param_l2_norm,
        "relative_update_l2_norm": relative_update_l2_norm,
        "global_update_cosine_with_previous": cosine_with_previous_update,
    }


def _compute_prefix_update_norm(
    *,
    previous_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    prefix: str,
) -> float:
    keys = [key for key in sorted(previous_state) if key.startswith(prefix)]
    if not keys:
        raise KeyError(f"No model_state keys found for prefix {prefix}")
    return _compute_delta_l2_norm_for_keys(
        previous_state=previous_state,
        current_state=current_state,
        keys=keys,
    )


def _compute_top_head_update_norm(
    *,
    previous_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    layer: int,
    head: int,
    n_heads: int,
) -> float:
    weight_key = f"blocks.{layer}.attn.q_proj.weight"
    if weight_key not in previous_state:
        raise KeyError(f"Missing attention projection key for layer {layer}: {weight_key}")
    d_model = int(previous_state[weight_key].shape[0])
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads} for layer {layer}")
    head_dim = d_model // n_heads
    head_start = head * head_dim
    head_end = head_start + head_dim
    if head_start < 0 or head_end > d_model:
        raise ValueError(f"Head index {head} is out of range for n_heads={n_heads}")

    total = 0.0
    projection_keys = [
        ("q_proj.weight", (slice(head_start, head_end), slice(None))),
        ("q_proj.bias", (slice(head_start, head_end),)),
        ("k_proj.weight", (slice(head_start, head_end), slice(None))),
        ("k_proj.bias", (slice(head_start, head_end),)),
        ("v_proj.weight", (slice(head_start, head_end), slice(None))),
        ("v_proj.bias", (slice(head_start, head_end),)),
        ("out_proj.weight", (slice(None), slice(head_start, head_end))),
    ]
    for suffix, index in projection_keys:
        key = f"blocks.{layer}.attn.{suffix}"
        previous_tensor = previous_state[key].float()[index]
        current_tensor = current_state[key].float()[index]
        delta = current_tensor - previous_tensor
        total += float(torch.sum(delta * delta).item())
    return total ** 0.5


def _compute_top_neuron_group_update_norm(
    *,
    previous_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    layer: int,
    neurons: list[int],
) -> float:
    if not neurons:
        raise ValueError("neurons must not be empty.")
    row_index = torch.tensor(sorted(int(neuron) for neuron in neurons), dtype=torch.long)
    fc_in_weight_key = f"blocks.{layer}.ff.fc_in.weight"
    fc_in_bias_key = f"blocks.{layer}.ff.fc_in.bias"
    fc_out_weight_key = f"blocks.{layer}.ff.fc_out.weight"
    required = [fc_in_weight_key, fc_in_bias_key, fc_out_weight_key]
    missing = [key for key in required if key not in previous_state]
    if missing:
        raise KeyError(f"Missing MLP neuron-group parameter keys for layer {layer}: {missing}")
    total = 0.0
    delta_fc_in_weight = current_state[fc_in_weight_key].float().index_select(0, row_index) - previous_state[
        fc_in_weight_key
    ].float().index_select(0, row_index)
    total += float(torch.sum(delta_fc_in_weight * delta_fc_in_weight).item())
    delta_fc_in_bias = current_state[fc_in_bias_key].float().index_select(0, row_index) - previous_state[
        fc_in_bias_key
    ].float().index_select(0, row_index)
    total += float(torch.sum(delta_fc_in_bias * delta_fc_in_bias).item())
    delta_fc_out_weight = current_state[fc_out_weight_key].float().index_select(1, row_index) - previous_state[
        fc_out_weight_key
    ].float().index_select(1, row_index)
    total += float(torch.sum(delta_fc_out_weight * delta_fc_out_weight).item())
    return total ** 0.5


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _compute_pearson_correlation(x_values: list[float], y_values: list[float]) -> dict[str, Any]:
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")
    if len(x_values) < 2:
        return {"status": "insufficient_points", "value": None, "num_points": len(x_values)}
    x_tensor = torch.tensor(x_values, dtype=torch.float32)
    y_tensor = torch.tensor(y_values, dtype=torch.float32)
    x_centered = x_tensor - x_tensor.mean()
    y_centered = y_tensor - y_tensor.mean()
    x_norm = float(torch.linalg.norm(x_centered).item())
    y_norm = float(torch.linalg.norm(y_centered).item())
    if x_norm == 0.0 or y_norm == 0.0:
        return {"status": "constant_series", "value": None, "num_points": len(x_values)}
    value = float(torch.dot(x_centered, y_centered).item() / (x_norm * y_norm))
    return {"status": "ok", "value": value, "num_points": len(x_values)}


def _render_family_update_link_interval_plot(
    *,
    interval_rows: list[dict[str, Any]],
    family_id: int,
    output_path: Path,
) -> Path:
    if not interval_rows:
        raise ValueError("interval_rows must not be empty.")
    _, plt = _import_matplotlib()
    steps = [int(row["step"]) for row in interval_rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(steps, [float(row["subset_deltas"]["useful_delta"]) for row in interval_rows], linewidth=2.0, label="Subset useful Δ")
    axes[0].plot(
        steps,
        [float(row["sweep_deltas"]["heldout_answer_accuracy_delta"]) for row in interval_rows],
        linewidth=2.0,
        label="Heldout accuracy Δ",
    )
    axes[0].plot(
        steps,
        [float(row["sweep_deltas"]["answer_accuracy_delta"]) for row in interval_rows],
        linewidth=2.0,
        label="Answer accuracy Δ",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[0].set_ylabel("Delta")
    axes[0].set_title(f"Family update link | family {family_id}")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        steps,
        [float(row["update_metrics"]["top_head_share_global"] or 0.0) for row in interval_rows],
        linewidth=2.0,
        label="Top-head share",
    )
    axes[1].plot(
        steps,
        [float(row["update_metrics"]["top_mlp_share_global"] or 0.0) for row in interval_rows],
        linewidth=2.0,
        label="Top-MLP share",
    )
    axes[1].plot(
        steps,
        [float(row["update_metrics"]["top_neuron_group_share_global"] or 0.0) for row in interval_rows],
        linewidth=2.0,
        label="Top-neuron-group share",
    )
    axes[1].plot(
        steps,
        [float(row["update_metrics"]["relative_update_l2_norm"] or 0.0) for row in interval_rows],
        linewidth=2.0,
        label="Relative update norm",
    )
    axes[1].set_xlabel("Checkpoint step")
    axes[1].set_ylabel("Share / relative norm")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_family_update_link_correlation_plot(
    *,
    correlations: dict[str, dict[str, Any]],
    title: str,
    output_path: Path,
) -> Path:
    _, plt = _import_matplotlib()
    valid_items = [
        (signal_name, float(payload["value"]))
        for signal_name, payload in correlations.items()
        if payload["status"] == "ok"
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    if valid_items:
        labels = [label for label, _ in valid_items]
        values = [value for _, value in valid_items]
        ax.bar(labels, values)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        ax.set_ylabel("Pearson correlation")
    else:
        ax.text(
            0.5,
            0.5,
            "No valid correlations",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def family_update_link(
    *,
    feature_family_trace_path: Path,
    subset_trajectory_path: Path,
    sweep_metrics_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
) -> tuple[Path, dict[str, Path]]:
    trace_payload = _load_feature_family_trace_payload(feature_family_trace_path)
    subset_payload = _load_subset_trajectory_payload(subset_trajectory_path)
    selected_feature_ids = sorted(int(feature_id) for feature_id in trace_payload["trace_subset"]["feature_ids"])
    subset_feature_ids = sorted(int(feature_id) for feature_id in subset_payload["feature_ids"])
    if selected_feature_ids != subset_feature_ids:
        raise ValueError(
            f"Trace selected feature ids {selected_feature_ids} do not match subset trajectory feature ids {subset_feature_ids}."
        )

    top_head = trace_payload["trace_summary"].get("top_head")
    top_mlp = trace_payload["trace_summary"].get("top_mlp")
    top_neuron_group = trace_payload["trace_summary"].get("top_neuron_group")
    if top_head is None or top_mlp is None or top_neuron_group is None:
        raise ValueError("feature family trace must include top_head, top_mlp, and top_neuron_group.")

    subset_rows = sorted(subset_payload["rows"], key=lambda row: int(row["step"]))
    if len(subset_rows) < 2:
        raise ValueError("subset trajectory must contain at least two checkpoint rows for update linking.")

    checkpoint_paths_by_step = _resolve_checkpoint_paths_by_step(checkpoint_dir)
    steps = [int(row["step"]) for row in subset_rows]
    missing_steps = [step for step in steps if step not in checkpoint_paths_by_step]
    if missing_steps:
        raise FileNotFoundError(f"Checkpoint files are missing for steps {missing_steps} in {checkpoint_dir}")
    sweep_rows_by_step = _load_sweep_rows_by_step(sweep_metrics_path)
    missing_sweep_steps = [step for step in steps if step not in sweep_rows_by_step]
    if missing_sweep_steps:
        raise KeyError(f"Sweep metrics are missing steps {missing_sweep_steps}: {sweep_metrics_path}")

    checkpoint_cache: dict[int, dict[str, Any]] = {}

    def load_step(step: int) -> dict[str, Any]:
        cached = checkpoint_cache.get(step)
        if cached is not None:
            return cached
        checkpoint_path = checkpoint_paths_by_step[step]
        payload = _load_checkpoint_state_for_update(checkpoint_path)
        if int(payload["step"]) != step:
            raise ValueError(
                f"Checkpoint filename step {step} does not match payload step {payload['step']}: {checkpoint_path}"
            )
        checkpoint_cache[step] = payload
        return payload

    n_heads = _extract_checkpoint_n_heads(load_step(steps[0])["checkpoint"])
    interval_rows: list[dict[str, Any]] = []
    for index in range(1, len(subset_rows)):
        previous_row = subset_rows[index - 1]
        current_row = subset_rows[index]
        previous_step = int(previous_row["step"])
        current_step = int(current_row["step"])
        previous_payload = load_step(previous_step)
        current_payload = load_step(current_step)
        previous_previous_state = None
        if index >= 2:
            previous_previous_state = load_step(int(subset_rows[index - 2]["step"]))["model_state"]

        global_update = _compute_global_update_stats(
            previous_state=previous_payload["model_state"],
            current_state=current_payload["model_state"],
            previous_previous_state=previous_previous_state,
        )
        top_head_update_l2_norm = _compute_top_head_update_norm(
            previous_state=previous_payload["model_state"],
            current_state=current_payload["model_state"],
            layer=int(top_head["layer"]),
            head=int(top_head["head"]),
            n_heads=n_heads,
        )
        top_head_attention_update_l2_norm = _compute_prefix_update_norm(
            previous_state=previous_payload["model_state"],
            current_state=current_payload["model_state"],
            prefix=f"blocks.{int(top_head['layer'])}.attn.",
        )
        top_mlp_update_l2_norm = _compute_prefix_update_norm(
            previous_state=previous_payload["model_state"],
            current_state=current_payload["model_state"],
            prefix=f"blocks.{int(top_mlp['layer'])}.ff.",
        )
        top_neuron_group_update_l2_norm = _compute_top_neuron_group_update_norm(
            previous_state=previous_payload["model_state"],
            current_state=current_payload["model_state"],
            layer=int(top_neuron_group["layer"]),
            neurons=[int(item) for item in top_neuron_group["neurons"]],
        )
        top_neuron_group_mlp_update_l2_norm = _compute_prefix_update_norm(
            previous_state=previous_payload["model_state"],
            current_state=current_payload["model_state"],
            prefix=f"blocks.{int(top_neuron_group['layer'])}.ff.",
        )
        sweep_row = sweep_rows_by_step[current_step]
        required_sweep_keys = [
            "delta_answer_accuracy",
            "delta_heldout_answer_accuracy",
            "delta_q",
            "delta_r",
            "delta_w",
        ]
        missing_sweep_keys = [key for key in required_sweep_keys if key not in sweep_row]
        if missing_sweep_keys:
            raise KeyError(f"Sweep row for step {current_step} is missing keys {missing_sweep_keys}")
        subset_deltas = {
            "mean_activation_delta": float(current_row["mean_activation_mean"] - previous_row["mean_activation_mean"]),
            "active_fraction_delta": float(current_row["active_fraction_mean"] - previous_row["active_fraction_mean"]),
            "correctness_gap_delta": float(current_row["correctness_gap_mean"] - previous_row["correctness_gap_mean"]),
            "heldout_gap_delta": float(current_row["heldout_gap_mean"] - previous_row["heldout_gap_mean"]),
            "structural_ood_gap_delta": float(
                current_row["structural_ood_gap_mean"] - previous_row["structural_ood_gap_mean"]
            ),
        }
        subset_deltas["useful_delta"] = float(
            subset_deltas["correctness_gap_delta"] + subset_deltas["heldout_gap_delta"]
        )
        interval_rows.append(
            {
                "previous_step": previous_step,
                "step": current_step,
                "checkpoint_path": str(current_payload["path"]),
                "subset_metrics": {
                    "previous": previous_row,
                    "current": current_row,
                },
                "subset_deltas": subset_deltas,
                "sweep_deltas": {
                    "answer_accuracy_delta": float(sweep_row["delta_answer_accuracy"]),
                    "heldout_answer_accuracy_delta": float(sweep_row["delta_heldout_answer_accuracy"]),
                    "delta_q": float(sweep_row["delta_q"]),
                    "delta_r": float(sweep_row["delta_r"]),
                    "delta_w": float(sweep_row["delta_w"]),
                },
                "update_metrics": {
                    **global_update,
                    "top_head_update_l2_norm": top_head_update_l2_norm,
                    "top_head_attention_update_l2_norm": top_head_attention_update_l2_norm,
                    "top_head_share_global": _safe_ratio(top_head_update_l2_norm, global_update["global_update_l2_norm"]),
                    "top_head_share_attention": _safe_ratio(top_head_update_l2_norm, top_head_attention_update_l2_norm),
                    "top_mlp_update_l2_norm": top_mlp_update_l2_norm,
                    "top_mlp_share_global": _safe_ratio(top_mlp_update_l2_norm, global_update["global_update_l2_norm"]),
                    "top_neuron_group_update_l2_norm": top_neuron_group_update_l2_norm,
                    "top_neuron_group_mlp_update_l2_norm": top_neuron_group_mlp_update_l2_norm,
                    "top_neuron_group_share_global": _safe_ratio(
                        top_neuron_group_update_l2_norm,
                        global_update["global_update_l2_norm"],
                    ),
                    "top_neuron_group_share_ff": _safe_ratio(
                        top_neuron_group_update_l2_norm,
                        top_neuron_group_mlp_update_l2_norm,
                    ),
                },
            }
        )

    response_series = {
        "subset_useful_delta": [float(row["subset_deltas"]["useful_delta"]) for row in interval_rows],
        "subset_correctness_gap_delta": [float(row["subset_deltas"]["correctness_gap_delta"]) for row in interval_rows],
        "subset_heldout_gap_delta": [float(row["subset_deltas"]["heldout_gap_delta"]) for row in interval_rows],
        "sweep_answer_accuracy_delta": [float(row["sweep_deltas"]["answer_accuracy_delta"]) for row in interval_rows],
        "sweep_heldout_answer_accuracy_delta": [
            float(row["sweep_deltas"]["heldout_answer_accuracy_delta"]) for row in interval_rows
        ],
    }
    aligned_signal_extractors = {
        "global_relative_update_l2_norm": lambda row: row["update_metrics"]["relative_update_l2_norm"],
        "global_update_cosine_with_previous": lambda row: row["update_metrics"]["global_update_cosine_with_previous"],
        "top_head_share_global": lambda row: row["update_metrics"]["top_head_share_global"],
        "top_head_share_attention": lambda row: row["update_metrics"]["top_head_share_attention"],
        "top_mlp_share_global": lambda row: row["update_metrics"]["top_mlp_share_global"],
        "top_neuron_group_share_global": lambda row: row["update_metrics"]["top_neuron_group_share_global"],
        "top_neuron_group_share_ff": lambda row: row["update_metrics"]["top_neuron_group_share_ff"],
        "delta_q": lambda row: row["sweep_deltas"]["delta_q"],
        "delta_r": lambda row: row["sweep_deltas"]["delta_r"],
        "delta_w": lambda row: row["sweep_deltas"]["delta_w"],
    }
    correlation_matrix: dict[str, dict[str, Any]] = {}
    valid_correlations: list[dict[str, Any]] = []
    for response_name, response_values in response_series.items():
        response_correlations: dict[str, Any] = {}
        for signal_name, extractor in aligned_signal_extractors.items():
            paired_values = [
                (float(row_response), extractor(interval_row))
                for row_response, interval_row in zip(response_values, interval_rows, strict=True)
                if extractor(interval_row) is not None
            ]
            x_values = [left for left, _ in paired_values]
            y_values = [float(right) for _, right in paired_values]
            correlation_payload = _compute_pearson_correlation(x_values, y_values)
            response_correlations[signal_name] = correlation_payload
            if correlation_payload["status"] == "ok":
                valid_correlations.append(
                    {
                        "response": response_name,
                        "signal": signal_name,
                        "value": float(correlation_payload["value"]),
                        "abs_value": abs(float(correlation_payload["value"])),
                    }
                )
        correlation_matrix[response_name] = response_correlations
    valid_correlations.sort(key=lambda row: float(row["abs_value"]), reverse=True)

    payload = {
        "feature_family_trace_path": str(feature_family_trace_path),
        "subset_trajectory_path": str(subset_trajectory_path),
        "sweep_metrics_path": str(sweep_metrics_path),
        "checkpoint_dir": str(checkpoint_dir),
        "family_id": int(trace_payload["family_id"]),
        "stage_name": str(trace_payload["stage_name"]),
        "selected_feature_ids": selected_feature_ids,
        "top_components": {
            "top_head": top_head,
            "top_mlp": top_mlp,
            "top_neuron_group": top_neuron_group,
        },
        "interval_rows": interval_rows,
        "correlation_summary": {
            "num_intervals": len(interval_rows),
            "correlations": correlation_matrix,
            "valid_correlations_sorted": valid_correlations[:16],
        },
        "top_intervals": {
            "by_subset_useful_delta": sorted(
                interval_rows,
                key=lambda row: float(row["subset_deltas"]["useful_delta"]),
                reverse=True,
            )[:8],
            "by_heldout_answer_accuracy_delta": sorted(
                interval_rows,
                key=lambda row: float(row["sweep_deltas"]["heldout_answer_accuracy_delta"]),
                reverse=True,
            )[:8],
            "by_top_mlp_share_global": sorted(
                interval_rows,
                key=lambda row: float(row["update_metrics"]["top_mlp_share_global"] or 0.0),
                reverse=True,
            )[:8],
        },
    }
    write_json(output_path, payload)
    plot_paths = {
        "interval_plot": _render_family_update_link_interval_plot(
            interval_rows=interval_rows,
            family_id=int(trace_payload["family_id"]),
            output_path=output_path.with_name(f"{output_path.stem}_intervals.svg"),
        ),
        "useful_correlation_plot": _render_family_update_link_correlation_plot(
            correlations=correlation_matrix["subset_useful_delta"],
            title=f"Family {trace_payload['family_id']} update-link correlations | subset useful Δ",
            output_path=output_path.with_name(f"{output_path.stem}_useful_correlations.svg"),
        ),
    }
    return output_path, plot_paths
