from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device
from circuit.vocab import Vocabulary


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


def _logits_from_stage(
    *,
    model: torch.nn.Module,
    residual_state: torch.Tensor,
    stage_name: str,
) -> torch.Tensor:
    if stage_name == "final_norm":
        hidden = residual_state
    else:
        hidden = model.final_norm(residual_state)
    return model.lm_head(hidden)


@dataclass(frozen=True)
class StageFeatureDataset:
    activations: torch.Tensor
    correct: torch.Tensor
    stage_margin: torch.Tensor
    answer_direction: torch.Tensor
    splits: list[str]
    sample_ids: list[str]
    query_indices: list[int]
    answer_tokens: list[str]


def _validate_matching_probe_order(
    *,
    target_dataset: StageFeatureDataset,
    source_dataset: StageFeatureDataset,
) -> None:
    if target_dataset.sample_ids != source_dataset.sample_ids:
        raise ValueError("Target and source feature datasets do not share the same sample_id order.")
    if target_dataset.query_indices != source_dataset.query_indices:
        raise ValueError("Target and source feature datasets do not share the same query_index order.")
    if target_dataset.splits != source_dataset.splits:
        raise ValueError("Target and source feature datasets do not share the same split order.")


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, num_features: int) -> None:
        super().__init__()
        if input_dim <= 0 or num_features <= 0:
            raise ValueError("SparseAutoencoder dimensions must be positive.")
        self.input_dim = input_dim
        self.num_features = num_features
        self.encoder = nn.Linear(input_dim, num_features)
        self.decoder = nn.Linear(num_features, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_activations = torch.relu(self.encoder(x))
        reconstruction = self.decoder(feature_activations)
        return reconstruction, feature_activations

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        weights = self.decoder.weight.data
        norms = weights.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.data = weights / norms


@torch.no_grad()
def collect_stage_feature_dataset(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    vocab: Vocabulary,
    stage_name: str,
) -> StageFeatureDataset:
    activation_rows: list[torch.Tensor] = []
    correct_rows: list[torch.Tensor] = []
    margin_rows: list[torch.Tensor] = []
    answer_direction_rows: list[torch.Tensor] = []
    splits: list[str] = []
    sample_ids: list[str] = []
    query_indices: list[int] = []
    answer_tokens: list[str] = []
    value_token_ids = torch.tensor(vocab.value_token_ids, device=batches[0]["input_ids"].device, dtype=torch.long)
    lm_head_weight = model.lm_head.weight.detach()

    for batch in batches:
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_residual_streams=True,
        )
        if outputs.residual_streams is None:
            raise RuntimeError("Feature analysis requires residual_streams in model output.")
        if stage_name not in outputs.residual_streams:
            raise KeyError(f"Residual stage {stage_name} not found in model output.")
        stage_residual = outputs.residual_streams[stage_name]
        stage_logits = _logits_from_stage(model=model, residual_state=stage_residual, stage_name=stage_name)
        answer_logits, answer_targets, metadata = extract_answer_logits(stage_logits, batch)
        rows = metadata["rows"]
        query_event_indices = metadata["query_indices"]
        prediction_positions = metadata["prediction_positions"]
        selected_activations = stage_residual[rows, prediction_positions, :].detach()
        activation_rows.append(selected_activations)

        predictions = answer_logits.argmax(dim=-1)
        correct = (predictions == answer_targets).detach()
        correct_rows.append(correct)

        answer_value_logits = answer_logits.index_select(dim=-1, index=value_token_ids)
        row_index = torch.arange(answer_targets.size(0), device=answer_targets.device)
        target_in_value_space = (value_token_ids.unsqueeze(0) == answer_targets.unsqueeze(1)).nonzero(as_tuple=False)
        if target_in_value_space.size(0) != answer_targets.size(0):
            raise RuntimeError("Failed to locate answer targets in value-token subset.")
        correct_logits = answer_value_logits[row_index, target_in_value_space[:, 1]]
        masked = answer_value_logits.clone()
        masked[row_index, target_in_value_space[:, 1]] = torch.finfo(masked.dtype).min
        best_incorrect_indices = masked.argmax(dim=-1)
        best_incorrect_token_ids = value_token_ids[best_incorrect_indices]
        margin = (correct_logits - masked.max(dim=-1).values).detach()
        margin_rows.append(margin)

        answer_direction = lm_head_weight[answer_targets] - lm_head_weight[best_incorrect_token_ids]
        answer_direction_rows.append(answer_direction.detach())

        for answer_index, row_index_value in enumerate(rows.tolist()):
            record = batch["records"][row_index_value]
            splits.append(str(record["split"]))
            sample_ids.append(str(record["sample_id"]))
            query_index = int(query_event_indices[answer_index].item())
            query_indices.append(query_index)
            answer_tokens.append(str(record["query_events"][query_index]["answer_value"]))

    return StageFeatureDataset(
        activations=torch.cat(activation_rows, dim=0),
        correct=torch.cat(correct_rows, dim=0),
        stage_margin=torch.cat(margin_rows, dim=0),
        answer_direction=torch.cat(answer_direction_rows, dim=0),
        splits=splits,
        sample_ids=sample_ids,
        query_indices=query_indices,
        answer_tokens=answer_tokens,
    )


def _shuffle_batches(x: torch.Tensor, batch_size: int) -> list[torch.Tensor]:
    permutation = torch.randperm(x.size(0), device=x.device)
    shuffled = x[permutation]
    return list(shuffled.split(batch_size))


def fit_sparse_autoencoder(
    *,
    activations: torch.Tensor,
    num_features: int,
    train_steps: int,
    learning_rate: float,
    l1_coefficient: float,
    batch_size: int,
) -> tuple[SparseAutoencoder, dict[str, float]]:
    if activations.ndim != 2:
        raise ValueError(f"Expected activations to have shape [num_examples, d_model], got {tuple(activations.shape)}")
    model = SparseAutoencoder(activations.size(1), num_features).to(activations.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if train_steps <= 0:
        raise ValueError("train_steps must be positive.")

    for step in range(train_steps):
        batches = _shuffle_batches(activations, batch_size=min(batch_size, activations.size(0)))
        batch = batches[step % len(batches)]
        reconstruction, feature_activations = model(batch)
        reconstruction_loss = F.mse_loss(reconstruction, batch)
        sparsity_loss = feature_activations.abs().mean()
        loss = reconstruction_loss + l1_coefficient * sparsity_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize_decoder()

    model.eval()
    with torch.no_grad():
        full_reconstruction, full_features = model(activations)
        reconstruction_loss = F.mse_loss(full_reconstruction, activations).item()
        total_variance = torch.var(activations, dim=0, unbiased=False).sum().item()
        residual_variance = torch.var(activations - full_reconstruction, dim=0, unbiased=False).sum().item()
        explained_variance = 1.0 - (residual_variance / total_variance if total_variance > 0 else 0.0)
        active_fraction = (full_features > 1e-4).float().mean().item()
        mean_feature_activation = full_features.mean().item()
    return model, {
        "reconstruction_loss": reconstruction_loss,
        "explained_variance": explained_variance,
        "active_fraction": active_fraction,
        "mean_feature_activation": mean_feature_activation,
    }


@torch.no_grad()
def _encode_features(model: SparseAutoencoder, activations: torch.Tensor) -> torch.Tensor:
    _, feature_activations = model(activations)
    return feature_activations


def _safe_gap(values: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.any() and (~mask).any():
        return float(values[mask].mean().item() - values[~mask].mean().item())
    return 0.0


def _top_example_rows(
    *,
    feature_values: torch.Tensor,
    dataset: StageFeatureDataset,
    top_k: int,
) -> list[dict[str, Any]]:
    top_values, top_indices = torch.topk(feature_values, k=min(top_k, feature_values.numel()))
    rows: list[dict[str, Any]] = []
    for value, feature_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
        rows.append(
            {
                "activation": float(value),
                "sample_id": dataset.sample_ids[feature_index],
                "split": dataset.splits[feature_index],
                "query_index": dataset.query_indices[feature_index],
                "answer_token": dataset.answer_tokens[feature_index],
                "correct": bool(dataset.correct[feature_index].item()),
                "stage_margin": float(dataset.stage_margin[feature_index].item()),
            }
        )
    return rows


def _mean_split_activation(feature_values: torch.Tensor, splits: list[str], split_name: str) -> float:
    selected_indices = [index for index, item in enumerate(splits) if item == split_name]
    if not selected_indices:
        return 0.0
    return float(feature_values[selected_indices].mean().item())


@torch.no_grad()
def summarize_features(
    *,
    sae: SparseAutoencoder,
    target_dataset: StageFeatureDataset,
    target_feature_activations: torch.Tensor,
    source_dataset: StageFeatureDataset | None,
    source_feature_activations: torch.Tensor | None,
    top_k_examples: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if target_feature_activations.ndim != 2:
        raise ValueError("target_feature_activations must have rank 2.")
    if source_dataset is None and source_feature_activations is not None:
        raise ValueError("source_feature_activations provided without source_dataset.")
    if source_dataset is not None and source_feature_activations is None:
        raise ValueError("source_dataset provided without source_feature_activations.")

    decoder_vectors = sae.decoder.weight.detach().transpose(0, 1)
    mean_answer_direction = F.normalize(target_dataset.answer_direction.mean(dim=0), dim=0)
    correct_mask = target_dataset.correct.bool()
    heldout_mask = torch.tensor([split == "heldout_pairs" for split in target_dataset.splits], device=target_feature_activations.device)
    structural_ood_mask = torch.tensor([split == "structural_ood" for split in target_dataset.splits], device=target_feature_activations.device)

    feature_rows: list[dict[str, Any]] = []
    for feature_index in range(target_feature_activations.size(1)):
        target_values = target_feature_activations[:, feature_index]
        decoder_vector = decoder_vectors[feature_index]
        alignment = F.cosine_similarity(
            F.normalize(decoder_vector.unsqueeze(0), dim=-1),
            mean_answer_direction.unsqueeze(0),
        ).item()
        row = {
            "feature": feature_index,
            "mean_activation": float(target_values.mean().item()),
            "active_fraction": float((target_values > 1e-4).float().mean().item()),
            "correctness_gap": _safe_gap(target_values, correct_mask),
            "heldout_gap": _safe_gap(target_values, heldout_mask),
            "structural_ood_gap": _safe_gap(target_values, structural_ood_mask),
            "margin_correlation": float(
                F.cosine_similarity(
                    (target_values - target_values.mean()).unsqueeze(0),
                    (target_dataset.stage_margin - target_dataset.stage_margin.mean()).unsqueeze(0),
                ).item()
            ),
            "answer_direction_alignment": float(alignment),
            "split_mean_activation": {
                split_name: _mean_split_activation(target_values, target_dataset.splits, split_name)
                for split_name in sorted(set(target_dataset.splits))
            },
            "top_examples": _top_example_rows(
                feature_values=target_values,
                dataset=target_dataset,
                top_k=top_k_examples,
            ),
        }
        if source_feature_activations is not None and source_dataset is not None:
            if source_feature_activations.shape[1] != target_feature_activations.shape[1]:
                raise ValueError("Source and target feature activations must have the same number of features.")
            source_values = source_feature_activations[:, feature_index]
            row["source_mean_activation"] = float(source_values.mean().item())
            row["source_active_fraction"] = float((source_values > 1e-4).float().mean().item())
            row["mean_activation_delta_vs_source"] = float(target_values.mean().item() - source_values.mean().item())
            row["heldout_gap_vs_source"] = float(
                row["heldout_gap"] - _safe_gap(
                    source_values,
                    torch.tensor([split == "heldout_pairs" for split in source_dataset.splits], device=source_values.device),
                )
            )
        feature_rows.append(row)

    ranked = {
        "by_mean_activation": sorted(feature_rows, key=lambda item: float(item["mean_activation"]), reverse=True),
        "by_correctness_gap": sorted(feature_rows, key=lambda item: float(item["correctness_gap"]), reverse=True),
        "by_heldout_gap": sorted(feature_rows, key=lambda item: float(item["heldout_gap"]), reverse=True),
        "by_margin_correlation": sorted(feature_rows, key=lambda item: abs(float(item["margin_correlation"])), reverse=True),
        "by_answer_direction_alignment": sorted(
            feature_rows,
            key=lambda item: float(item["answer_direction_alignment"]),
            reverse=True,
        ),
    }
    if source_feature_activations is not None:
        ranked["by_mean_activation_delta_vs_source"] = sorted(
            feature_rows,
            key=lambda item: float(item["mean_activation_delta_vs_source"]),
            reverse=True,
        )
        ranked["by_abs_mean_activation_delta_vs_source"] = sorted(
            feature_rows,
            key=lambda item: abs(float(item["mean_activation_delta_vs_source"])),
            reverse=True,
        )
    return feature_rows, ranked


def analyze_checkpoint_features(
    *,
    config_path: Path,
    checkpoint_path: Path,
    probe_set_path: Path,
    stage_name: str,
    output_path: Path,
    source_checkpoint_path: Path | None = None,
    device_name: str = "cpu",
    num_features: int = 64,
    train_steps: int = 400,
    learning_rate: float = 1e-3,
    l1_coefficient: float = 1e-3,
    sae_batch_size: int = 256,
    top_k_features: int = 12,
    top_k_examples: int = 6,
) -> tuple[Path, Path]:
    if top_k_features <= 0 or top_k_examples <= 0:
        raise ValueError("top_k_features and top_k_examples must be positive.")

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

    target_model = build_model(spec.model, len(vocab.tokens), device)
    target_checkpoint = load_checkpoint(checkpoint_path, device)
    load_model_state(target_model, target_checkpoint["model_state"])
    target_model.eval()
    target_dataset = collect_stage_feature_dataset(
        model=target_model,
        batches=analysis_batches,
        vocab=vocab,
        stage_name=stage_name,
    )
    sae, sae_metrics = fit_sparse_autoencoder(
        activations=target_dataset.activations,
        num_features=num_features,
        train_steps=train_steps,
        learning_rate=learning_rate,
        l1_coefficient=l1_coefficient,
        batch_size=sae_batch_size,
    )
    target_feature_activations = _encode_features(sae, target_dataset.activations)

    source_dataset = None
    source_feature_activations = None
    source_checkpoint = None
    if source_checkpoint_path is not None:
        source_model = build_model(spec.model, len(vocab.tokens), device)
        source_checkpoint = load_checkpoint(source_checkpoint_path, device)
        load_model_state(source_model, source_checkpoint["model_state"])
        source_model.eval()
        source_dataset = collect_stage_feature_dataset(
            model=source_model,
            batches=analysis_batches,
            vocab=vocab,
            stage_name=stage_name,
        )
        _validate_matching_probe_order(target_dataset=target_dataset, source_dataset=source_dataset)
        source_feature_activations = _encode_features(sae, source_dataset.activations)

    feature_rows, ranked = summarize_features(
        sae=sae,
        target_dataset=target_dataset,
        target_feature_activations=target_feature_activations,
        source_dataset=source_dataset,
        source_feature_activations=source_feature_activations,
        top_k_examples=top_k_examples,
    )

    sae_state_path = output_path.with_name(f"{output_path.stem}_sae.pt")
    torch.save(
        {
            "stage_name": stage_name,
            "checkpoint_path": str(checkpoint_path),
            "source_checkpoint_path": None if source_checkpoint_path is None else str(source_checkpoint_path),
            "sae_state": sae.state_dict(),
            "num_features": num_features,
            "input_dim": int(target_dataset.activations.size(1)),
        },
        sae_state_path,
    )

    payload = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": int(target_checkpoint["step"]),
        "source_checkpoint_path": None if source_checkpoint_path is None else str(source_checkpoint_path),
        "source_checkpoint_step": None,
        "probe_set_path": str(probe_set_path),
        "stage_name": stage_name,
        "device": device_name,
        "sae_weights_path": str(sae_state_path),
        "sae": {
            "num_features": num_features,
            "train_steps": train_steps,
            "learning_rate": learning_rate,
            "l1_coefficient": l1_coefficient,
            "batch_size": sae_batch_size,
            **sae_metrics,
        },
        "dataset": {
            "num_examples": int(target_dataset.activations.size(0)),
            "input_dim": int(target_dataset.activations.size(1)),
            "split_counts": {
                split_name: sum(1 for split in target_dataset.splits if split == split_name)
                for split_name in sorted(set(target_dataset.splits))
            },
        },
        "top_features": {
            ranking_name: features[:top_k_features]
            for ranking_name, features in ranked.items()
        },
        "feature_rows": feature_rows,
    }
    if source_checkpoint is not None:
        payload["source_checkpoint_step"] = int(source_checkpoint["step"])
    write_json(output_path, payload)
    return output_path, sae_state_path
