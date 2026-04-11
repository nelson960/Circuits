from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from circuit.analysis.formation import (
    compute_head_ablation_importance,
    compute_head_localization,
    compute_qrw_batch,
    extract_answer_logits,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import SymbolicKVDataset, collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.eval import evaluate_split
from circuit.io import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device
from circuit.vocab import Vocabulary

PROBE_SPLITS = ["validation_iid", "heldout_pairs", "structural_ood", "counterfactual"]
TOP_MLP_LAYERS_FOR_NEURON_SCREEN = 2
TOP_NEURONS_PER_LAYER = 4


def _axis_signature(record: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    return tuple((axis_name, int(record["axes"][axis_name])) for axis_name in sorted(record["axes"]))


def _select_probe_records(
    records: list[dict[str, Any]],
    *,
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("limit must be positive.")
    if len(records) <= limit:
        return list(records)

    grouped: dict[tuple[tuple[str, int], ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_axis_signature(record)].append(record)

    ordered_groups: list[list[dict[str, Any]]] = []
    for signature, group_records in sorted(grouped.items(), key=lambda item: (len(item[1]), item[0])):
        shuffled = list(group_records)
        rng.shuffle(shuffled)
        ordered_groups.append(shuffled)

    selected: list[dict[str, Any]] = []
    group_index = 0
    while len(selected) < limit:
        if not ordered_groups:
            break
        current_group = ordered_groups[group_index]
        selected.append(current_group.pop())
        if not current_group:
            ordered_groups.pop(group_index)
            if not ordered_groups:
                break
            group_index %= len(ordered_groups)
            continue
        group_index = (group_index + 1) % len(ordered_groups)
    return selected


def generate_probe_set(
    *,
    benchmark_dir: Path,
    output_path: Path,
    examples_per_split: int,
    seed: int,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Probe-set output already exists: {output_path}")
    metadata_path = output_path.with_suffix(".metadata.json")
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Probe-set metadata already exists: {metadata_path}")

    rng = random.Random(seed)
    selected_records: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    split_axis_coverage: dict[str, list[dict[str, int]]] = {}

    for split_name in PROBE_SPLITS:
        dataset = SymbolicKVDataset(benchmark_dir, split_name)
        chosen = _select_probe_records(dataset.records, limit=examples_per_split, rng=rng)
        selected_records.extend(chosen)
        split_counts[split_name] = len(chosen)
        split_axis_coverage[split_name] = [
            {axis_name: int(axis_value) for axis_name, axis_value in signature}
            for signature in sorted({_axis_signature(record) for record in chosen})
        ]

    write_jsonl(output_path, selected_records)
    write_json(
        metadata_path,
        {
            "benchmark_dir": str(benchmark_dir),
            "examples_per_split": examples_per_split,
            "seed": seed,
            "split_counts": split_counts,
            "split_axis_coverage": split_axis_coverage,
            "num_records": len(selected_records),
            "splits": PROBE_SPLITS,
        },
    )
    return output_path, metadata_path


def load_probe_set(probe_set_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = probe_set_path.with_suffix(".metadata.json")
    if not probe_set_path.exists():
        raise FileNotFoundError(f"Probe-set file not found: {probe_set_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Probe-set metadata not found: {metadata_path}")
    return list(iter_jsonl(probe_set_path)), read_json(metadata_path)


def ensure_probe_set(
    *,
    benchmark_dir: Path,
    probe_set_path: Path,
    create_if_missing: bool,
    examples_per_split: int,
    seed: int,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    metadata_path = probe_set_path.with_suffix(".metadata.json")
    if probe_set_path.exists() and metadata_path.exists():
        return probe_set_path, metadata_path
    if probe_set_path.exists() != metadata_path.exists():
        raise FileNotFoundError(
            f"Probe-set state is inconsistent: file={probe_set_path.exists()} metadata={metadata_path.exists()} "
            f"for path {probe_set_path}."
        )
    if not create_if_missing:
        raise FileNotFoundError(
            f"Probe-set file not found: {probe_set_path}. "
            f"Generate it first with `generate-probe-set` or rerun `checkpoint-sweep` with `--create-probe-set`."
        )
    return generate_probe_set(
        benchmark_dir=benchmark_dir,
        output_path=probe_set_path,
        examples_per_split=examples_per_split,
        seed=seed,
        overwrite=overwrite,
    )


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


def _logits_from_residual_state(
    *,
    model: torch.nn.Module,
    residual_state: torch.Tensor,
    stage_name: str,
) -> torch.Tensor:
    if stage_name == "final_norm":
        normalized = residual_state
    else:
        normalized = model.final_norm(residual_state)
    return model.lm_head(normalized)


def _subset_accuracy(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    subset_token_ids: torch.Tensor,
) -> tuple[int, int]:
    subset_logits = logits.index_select(dim=-1, index=subset_token_ids)
    predictions = subset_logits.argmax(dim=-1)
    target_in_subset = (subset_token_ids.unsqueeze(0) == targets.unsqueeze(1)).nonzero(as_tuple=False)
    if target_in_subset.size(0) != targets.size(0):
        raise RuntimeError("Failed to locate targets inside the requested token subset.")
    correct = (predictions == target_in_subset[:, 1]).sum().item()
    return int(correct), int(targets.numel())


@torch.no_grad()
def compute_residual_probe_metrics(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    vocab: Vocabulary,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    key_token_ids = torch.tensor(vocab.encode(vocab.key_tokens), device=batches[0]["input_ids"].device, dtype=torch.long)
    value_token_ids = torch.tensor(vocab.value_token_ids, device=batches[0]["input_ids"].device, dtype=torch.long)
    per_stage_counts: dict[str, dict[str, float]] = {}
    stage_mean_vectors: dict[str, torch.Tensor] = {}
    stage_mean_counts: dict[str, int] = defaultdict(int)

    for batch in batches:
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_attentions=True,
            return_residual_streams=True,
        )
        if outputs.residual_streams is None:
            raise RuntimeError("Residual probe metrics require residual_streams in model output.")
        if outputs.attentions is None:
            raise RuntimeError("Residual probe metrics require attentions in model output.")
        answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        rows = metadata["rows"]
        prediction_positions = metadata["prediction_positions"]
        query_key_positions = metadata["query_key_positions"]
        support_value_positions = metadata["support_value_positions"]
        query_key_targets = batch["input_ids"][rows, query_key_positions]
        support_value_targets = batch["input_ids"][rows, support_value_positions]

        for stage_name, residual_state in outputs.residual_streams.items():
            stage_logits = _logits_from_residual_state(model=model, residual_state=residual_state, stage_name=stage_name)
            prediction_logits = stage_logits[rows, prediction_positions, :]
            query_logits = stage_logits[rows, query_key_positions, :]
            support_logits = stage_logits[rows, support_value_positions, :]

            if stage_name not in per_stage_counts:
                per_stage_counts[stage_name] = {
                    "answer_correct": 0.0,
                    "answer_total": 0.0,
                    "query_correct": 0.0,
                    "query_total": 0.0,
                    "support_correct": 0.0,
                    "support_total": 0.0,
                    "answer_margin_sum": 0.0,
                }

            answer_correct, answer_total = _subset_accuracy(
                logits=prediction_logits,
                targets=answer_targets,
                subset_token_ids=value_token_ids,
            )
            query_correct, query_total = _subset_accuracy(
                logits=query_logits,
                targets=query_key_targets,
                subset_token_ids=key_token_ids,
            )
            support_correct, support_total = _subset_accuracy(
                logits=support_logits,
                targets=support_value_targets,
                subset_token_ids=value_token_ids,
            )

            subset_answer_logits = prediction_logits.index_select(dim=-1, index=value_token_ids)
            target_in_value_space = (value_token_ids.unsqueeze(0) == answer_targets.unsqueeze(1)).nonzero(as_tuple=False)
            row_index = torch.arange(prediction_logits.size(0), device=prediction_logits.device)
            correct_logits = subset_answer_logits[row_index, target_in_value_space[:, 1]]
            masked_answer_logits = subset_answer_logits.clone()
            masked_answer_logits[row_index, target_in_value_space[:, 1]] = torch.finfo(masked_answer_logits.dtype).min
            answer_margin = (correct_logits - masked_answer_logits.max(dim=-1).values).sum().item()

            per_stage_counts[stage_name]["answer_correct"] += answer_correct
            per_stage_counts[stage_name]["answer_total"] += answer_total
            per_stage_counts[stage_name]["query_correct"] += query_correct
            per_stage_counts[stage_name]["query_total"] += query_total
            per_stage_counts[stage_name]["support_correct"] += support_correct
            per_stage_counts[stage_name]["support_total"] += support_total
            per_stage_counts[stage_name]["answer_margin_sum"] += answer_margin

            prediction_state = residual_state[rows, prediction_positions, :]
            stage_mean = prediction_state.sum(dim=0)
            if stage_name not in stage_mean_vectors:
                stage_mean_vectors[stage_name] = stage_mean
            else:
                stage_mean_vectors[stage_name] += stage_mean
            stage_mean_counts[stage_name] += prediction_state.size(0)

    probe_metrics: dict[str, Any] = {
        "answer_probe_accuracy_by_stage": {},
        "query_probe_accuracy_by_stage": {},
        "support_probe_accuracy_by_stage": {},
        "answer_margin_by_stage": {},
    }
    mean_vectors: dict[str, torch.Tensor] = {}
    for stage_name in sorted(per_stage_counts):
        counts = per_stage_counts[stage_name]
        probe_metrics["answer_probe_accuracy_by_stage"][stage_name] = counts["answer_correct"] / counts["answer_total"]
        probe_metrics["query_probe_accuracy_by_stage"][stage_name] = counts["query_correct"] / counts["query_total"]
        probe_metrics["support_probe_accuracy_by_stage"][stage_name] = counts["support_correct"] / counts["support_total"]
        probe_metrics["answer_margin_by_stage"][stage_name] = counts["answer_margin_sum"] / counts["answer_total"]
        mean_vectors[stage_name] = stage_mean_vectors[stage_name] / stage_mean_counts[stage_name]
    return probe_metrics, mean_vectors


@torch.no_grad()
def compute_mlp_block_metrics(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    vocab: Vocabulary,
) -> list[dict[str, float | int]]:
    value_token_ids = torch.tensor(vocab.value_token_ids, device=batches[0]["input_ids"].device, dtype=torch.long)
    results: list[dict[str, float | int]] = []
    for layer_index in range(len(model.blocks)):
        write_norm_sum = 0.0
        answer_margin_delta_sum = 0.0
        num_examples = 0
        for batch in batches:
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
            )
            if outputs.residual_streams is None:
                raise RuntimeError("MLP block metrics require residual_streams in model output.")
            _, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
            rows = metadata["rows"]
            prediction_positions = metadata["prediction_positions"]
            post_attn = outputs.residual_streams[f"layer_{layer_index}_post_attn"][rows, prediction_positions, :]
            post_mlp = outputs.residual_streams[f"layer_{layer_index}_post_mlp"][rows, prediction_positions, :]
            write_norm_sum += (post_mlp - post_attn).norm(dim=-1).sum().item()
            num_examples += post_attn.size(0)

            attn_logits = _logits_from_residual_state(
                model=model,
                residual_state=outputs.residual_streams[f"layer_{layer_index}_post_attn"],
                stage_name=f"layer_{layer_index}_post_attn",
            )[rows, prediction_positions, :]
            mlp_logits = _logits_from_residual_state(
                model=model,
                residual_state=outputs.residual_streams[f"layer_{layer_index}_post_mlp"],
                stage_name=f"layer_{layer_index}_post_mlp",
            )[rows, prediction_positions, :]
            attn_value_logits = attn_logits.index_select(dim=-1, index=value_token_ids)
            mlp_value_logits = mlp_logits.index_select(dim=-1, index=value_token_ids)
            target_in_value_space = (value_token_ids.unsqueeze(0) == answer_targets.unsqueeze(1)).nonzero(as_tuple=False)
            row_index = torch.arange(answer_targets.size(0), device=answer_targets.device)
            attn_correct = attn_value_logits[row_index, target_in_value_space[:, 1]]
            mlp_correct = mlp_value_logits[row_index, target_in_value_space[:, 1]]
            masked_attn = attn_value_logits.clone()
            masked_attn[row_index, target_in_value_space[:, 1]] = torch.finfo(masked_attn.dtype).min
            masked_mlp = mlp_value_logits.clone()
            masked_mlp[row_index, target_in_value_space[:, 1]] = torch.finfo(masked_mlp.dtype).min
            attn_margin = attn_correct - masked_attn.max(dim=-1).values
            mlp_margin = mlp_correct - masked_mlp.max(dim=-1).values
            answer_margin_delta_sum += (mlp_margin - attn_margin).sum().item()

        results.append(
            {
                "layer": layer_index,
                "write_norm_mean": write_norm_sum / num_examples,
                "answer_margin_delta_mean": answer_margin_delta_sum / num_examples,
            }
        )
    return results


@torch.no_grad()
def compute_mlp_ablation_importance(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
) -> list[dict[str, float | int]]:
    baseline_correct = 0
    baseline_total = 0
    for batch in batches:
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        answer_logits, answer_targets, _ = extract_answer_logits(outputs.logits, batch)
        baseline_correct += (answer_logits.argmax(dim=-1) == answer_targets).sum().item()
        baseline_total += answer_targets.numel()
    if baseline_total == 0:
        raise RuntimeError("No answer targets available for MLP ablation analysis.")
    baseline_accuracy = baseline_correct / baseline_total

    n_layers = len(model.blocks)
    summary: list[dict[str, float | int]] = []
    for layer_index in range(n_layers):
        ablated_correct = 0
        ablated_total = 0
        mlp_mask = {layer: 1.0 for layer in range(n_layers)}
        mlp_mask[layer_index] = 0.0
        for batch in batches:
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                mlp_mask=mlp_mask,
            )
            answer_logits, answer_targets, _ = extract_answer_logits(outputs.logits, batch)
            ablated_correct += (answer_logits.argmax(dim=-1) == answer_targets).sum().item()
            ablated_total += answer_targets.numel()
        ablated_accuracy = ablated_correct / ablated_total
        summary.append(
            {
                "layer": layer_index,
                "baseline_accuracy": baseline_accuracy,
                "ablated_accuracy": ablated_accuracy,
                "accuracy_drop": baseline_accuracy - ablated_accuracy,
            }
        )
    return summary


def _select_candidate_mlp_layers(
    *,
    mlp_ablation: list[dict[str, float | int]],
    mlp_block_metrics: list[dict[str, float | int]],
    top_layers: int = TOP_MLP_LAYERS_FOR_NEURON_SCREEN,
) -> list[int]:
    ordered_layers: list[int] = []
    for items, metric_name in (
        (mlp_ablation, "accuracy_drop"),
        (mlp_block_metrics, "answer_margin_delta_mean"),
    ):
        for item in _top_k_by_metric(items, metric_name, top_layers):
            layer_index = int(item["layer"])
            if layer_index not in ordered_layers:
                ordered_layers.append(layer_index)
    return ordered_layers


@torch.no_grad()
def compute_mlp_neuron_write_metrics(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    candidate_layers: list[int],
    top_neurons_per_layer: int = TOP_NEURONS_PER_LAYER,
) -> list[dict[str, float | int]]:
    if not candidate_layers:
        return []
    neuron_scores: list[dict[str, float | int]] = []
    for layer_index in candidate_layers:
        ff = model.blocks[layer_index].ff
        write_norm = ff.fc_out.weight.detach().float().norm(dim=0)
        activation_abs_sum = torch.zeros(ff.d_ff, device=write_norm.device)
        num_examples = 0
        for batch in batches:
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_mlp_states=True,
            )
            if outputs.mlp_states is None:
                raise RuntimeError("Neuron write metrics require mlp_states in model output.")
            _, _, metadata = extract_answer_logits(outputs.logits, batch)
            rows = metadata["rows"]
            prediction_positions = metadata["prediction_positions"]
            hidden = outputs.mlp_states[f"layer_{layer_index}_hidden"][rows, prediction_positions, :]
            activation_abs_sum += hidden.detach().abs().sum(dim=0)
            num_examples += hidden.size(0)
        if num_examples == 0:
            raise RuntimeError("No query examples available for neuron screening.")
        activation_abs_mean = activation_abs_sum / num_examples
        write_strength = activation_abs_mean * write_norm
        top_values, top_indices = torch.topk(
            write_strength,
            k=min(top_neurons_per_layer, write_strength.numel()),
        )
        for score, neuron_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
            neuron_scores.append(
                {
                    "layer": layer_index,
                    "neuron": int(neuron_index),
                    "write_strength": float(score),
                    "activation_abs_mean": float(activation_abs_mean[neuron_index].item()),
                    "write_vector_norm": float(write_norm[neuron_index].item()),
                }
            )
    return neuron_scores


@torch.no_grad()
def compute_mlp_neuron_ablation_importance(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    candidate_neurons: list[dict[str, float | int]],
    baseline_accuracy: float,
) -> list[dict[str, float | int]]:
    if not candidate_neurons:
        return []
    summary: list[dict[str, float | int]] = []
    for candidate in candidate_neurons:
        layer_index = int(candidate["layer"])
        neuron_index = int(candidate["neuron"])
        d_ff = model.blocks[layer_index].ff.d_ff
        ablated_correct = 0
        ablated_total = 0
        neuron_mask = {layer_index: torch.ones(d_ff, device=batches[0]["input_ids"].device)}
        neuron_mask[layer_index][neuron_index] = 0.0
        for batch in batches:
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                neuron_mask=neuron_mask,
            )
            answer_logits, answer_targets, _ = extract_answer_logits(outputs.logits, batch)
            ablated_correct += (answer_logits.argmax(dim=-1) == answer_targets).sum().item()
            ablated_total += answer_targets.numel()
        ablated_accuracy = ablated_correct / ablated_total
        summary.append(
            {
                "layer": layer_index,
                "neuron": neuron_index,
                "baseline_accuracy": baseline_accuracy,
                "ablated_accuracy": ablated_accuracy,
                "accuracy_drop": baseline_accuracy - ablated_accuracy,
                "write_strength": float(candidate["write_strength"]),
                "activation_abs_mean": float(candidate["activation_abs_mean"]),
                "write_vector_norm": float(candidate["write_vector_norm"]),
            }
        )
    return summary


def _top_k_by_metric(items: list[dict[str, Any]], metric_name: str, k: int) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: float(item[metric_name]), reverse=True)[:k]


def analyze_checkpoint_on_probe_set(
    *,
    config_path: Path,
    checkpoint_path: Path,
    probe_set_path: Path,
    device_name: str = "cpu",
    top_k: int = 4,
) -> dict[str, Any]:
    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )

    device = torch.device(device_name)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    model = build_model(spec.model, len(vocab.tokens), device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()

    probe_loader = _make_probe_loader(
        probe_records=probe_records,
        batch_size=spec.evaluation.batch_size,
        pad_token_id=vocab.pad_token_id,
    )
    probe_metrics = evaluate_split(
        model=model,
        data_loader=probe_loader,
        device=device,
        pad_token_id=vocab.pad_token_id,
        value_token_ids=vocab.value_token_ids,
        max_batches=None,
        include_analysis=True,
    )
    split_behavior: dict[str, dict[str, Any]] = {}
    for split_name in sorted({str(record["split"]) for record in probe_records}):
        split_records = [record for record in probe_records if str(record["split"]) == split_name]
        split_loader = _make_probe_loader(
            probe_records=split_records,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
        )
        split_behavior[split_name] = evaluate_split(
            model=model,
            data_loader=split_loader,
            device=device,
            pad_token_id=vocab.pad_token_id,
            value_token_ids=vocab.value_token_ids,
            max_batches=None,
            include_analysis=False,
        )
    analysis_batches = [
        move_batch_to_device(batch, device)
        for batch in _make_probe_loader(
            probe_records=probe_records,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
        )
    ]
    residual_probe_metrics, stage_mean_vectors = compute_residual_probe_metrics(
        model=model,
        batches=analysis_batches,
        vocab=vocab,
    )
    head_localization = compute_head_localization(model=model, batches=analysis_batches)
    head_ablation = compute_head_ablation_importance(model=model, batches=analysis_batches)
    mlp_ablation = compute_mlp_ablation_importance(model=model, batches=analysis_batches)
    mlp_block_metrics = compute_mlp_block_metrics(model=model, batches=analysis_batches, vocab=vocab)
    candidate_layers = _select_candidate_mlp_layers(
        mlp_ablation=mlp_ablation,
        mlp_block_metrics=mlp_block_metrics,
    )
    neuron_write_metrics = compute_mlp_neuron_write_metrics(
        model=model,
        batches=analysis_batches,
        candidate_layers=candidate_layers,
    )
    neuron_ablation = compute_mlp_neuron_ablation_importance(
        model=model,
        batches=analysis_batches,
        candidate_neurons=neuron_write_metrics,
        baseline_accuracy=float(probe_metrics["answer_accuracy"]),
    )

    return {
        "step": int(checkpoint["step"]),
        "checkpoint_path": str(checkpoint_path),
        "probe_set_path": str(probe_set_path),
        "behavior": probe_metrics,
        "split_behavior": split_behavior,
        "residual_probes": residual_probe_metrics,
        "head_localization_top": _top_k_by_metric(head_localization, "support_attention_mean", top_k),
        "head_ablation_top": _top_k_by_metric(head_ablation, "accuracy_drop", top_k),
        "mlp_ablation_top": _top_k_by_metric(mlp_ablation, "accuracy_drop", top_k),
        "mlp_write_top": _top_k_by_metric(mlp_block_metrics, "answer_margin_delta_mean", top_k),
        "candidate_mlp_layers_for_neurons": candidate_layers,
        "neuron_write_top": _top_k_by_metric(neuron_write_metrics, "write_strength", top_k),
        "neuron_ablation_top": _top_k_by_metric(neuron_ablation, "accuracy_drop", top_k),
        "_stage_mean_vectors": stage_mean_vectors,
    }


def _stage_drift(
    current_vectors: dict[str, torch.Tensor],
    previous_vectors: dict[str, torch.Tensor] | None,
) -> dict[str, float]:
    if previous_vectors is None:
        return {stage_name: 0.0 for stage_name in sorted(current_vectors)}
    drift: dict[str, float] = {}
    for stage_name in sorted(current_vectors):
        if stage_name not in previous_vectors:
            drift[stage_name] = 0.0
            continue
        current_vector = current_vectors[stage_name].detach().float().cpu()
        previous_vector = previous_vectors[stage_name].detach().float().cpu()
        cosine = torch.nn.functional.cosine_similarity(
            current_vector.unsqueeze(0),
            previous_vector.unsqueeze(0),
        ).item()
        drift[stage_name] = 1.0 - cosine
    return drift


def summarize_birth_windows(
    *,
    rows: list[dict[str, Any]],
    window_radius: int = 250,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty.")
    answer_gain_steps = sorted(rows, key=lambda row: float(row["delta_answer_accuracy"]), reverse=True)
    heldout_gain_steps = sorted(rows, key=lambda row: float(row["delta_heldout_answer_accuracy"]), reverse=True)
    q_gain_steps = sorted(rows, key=lambda row: float(row["delta_q"]), reverse=True)
    candidate_steps = {
        int(answer_gain_steps[0]["step"]),
        int(heldout_gain_steps[0]["step"]),
        int(q_gain_steps[0]["step"]),
    }
    windows = [
        {
            "center_step": step,
            "start_step": max(0, step - window_radius),
            "end_step": step + window_radius,
        }
        for step in sorted(candidate_steps)
    ]
    return {
        "top_answer_gain_step": int(answer_gain_steps[0]["step"]),
        "top_heldout_gain_step": int(heldout_gain_steps[0]["step"]),
        "top_q_gain_step": int(q_gain_steps[0]["step"]),
        "windows": windows,
    }


def run_checkpoint_sweep(
    *,
    config_path: Path,
    probe_set_path: Path,
    output_path: Path,
    checkpoint_dir: Path | None = None,
    device_name: str = "cpu",
    create_probe_set_if_missing: bool = False,
    probe_examples_per_split: int = 24,
    probe_seed: int = 17,
    overwrite_probe_set: bool = False,
) -> tuple[Path, Path]:
    spec = TrainSpec.from_path(config_path)
    ensure_probe_set(
        benchmark_dir=spec.benchmark_dir,
        probe_set_path=probe_set_path,
        create_if_missing=create_probe_set_if_missing,
        examples_per_split=probe_examples_per_split,
        seed=probe_seed,
        overwrite=overwrite_probe_set,
    )
    effective_checkpoint_dir = checkpoint_dir or (spec.output_dir / "checkpoints")
    checkpoints = sorted(effective_checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No step checkpoints found in {effective_checkpoint_dir}")
    if output_path.exists():
        output_path.unlink()
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    if summary_path.exists():
        summary_path.unlink()

    previous_vectors: dict[str, torch.Tensor] | None = None
    previous_behavior: dict[str, float] | None = None
    rows: list[dict[str, Any]] = []
    progress_bar = tqdm(checkpoints, desc="checkpoint-sweep", dynamic_ncols=True, leave=True)
    for checkpoint_path in progress_bar:
        analysis = analyze_checkpoint_on_probe_set(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            probe_set_path=probe_set_path,
            device_name=device_name,
        )
        behavior = analysis["behavior"]
        split_behavior = analysis["split_behavior"]
        current_vectors = analysis.pop("_stage_mean_vectors")
        heldout_answer_accuracy = float(split_behavior.get("heldout_pairs", {}).get("answer_accuracy", 0.0))
        stage_drift = _stage_drift(current_vectors, previous_vectors)
        row = {
            "step": analysis["step"],
            "checkpoint_path": analysis["checkpoint_path"],
            "probe_set_path": analysis["probe_set_path"],
            "answer_accuracy": behavior["answer_accuracy"],
            "heldout_answer_accuracy": heldout_answer_accuracy,
            "token_accuracy": behavior["token_accuracy"],
            "read_key_accuracy": behavior["read_key_accuracy"],
            "write_key_accuracy": behavior["write_key_accuracy"],
            "write_value_accuracy": behavior["write_value_accuracy"],
            "q": behavior["q"],
            "r": behavior["r"],
            "w": behavior["w"],
            "answer_probe_accuracy_by_stage": analysis["residual_probes"]["answer_probe_accuracy_by_stage"],
            "query_probe_accuracy_by_stage": analysis["residual_probes"]["query_probe_accuracy_by_stage"],
            "support_probe_accuracy_by_stage": analysis["residual_probes"]["support_probe_accuracy_by_stage"],
            "answer_margin_by_stage": analysis["residual_probes"]["answer_margin_by_stage"],
            "prediction_state_drift_by_stage": stage_drift,
            "top_heads_by_ablation": analysis["head_ablation_top"],
            "top_heads_by_localization": analysis["head_localization_top"],
            "top_mlps_by_ablation": analysis["mlp_ablation_top"],
            "top_mlps_by_write": analysis["mlp_write_top"],
            "candidate_mlp_layers_for_neurons": analysis["candidate_mlp_layers_for_neurons"],
            "top_neurons_by_write": analysis["neuron_write_top"],
            "top_neurons_by_ablation": analysis["neuron_ablation_top"],
            "delta_answer_accuracy": 0.0 if previous_behavior is None else behavior["answer_accuracy"] - previous_behavior["answer_accuracy"],
            "delta_heldout_answer_accuracy": 0.0
            if previous_behavior is None
            else heldout_answer_accuracy - previous_behavior["heldout_answer_accuracy"],
            "delta_q": 0.0 if previous_behavior is None else behavior["q"] - previous_behavior["q"],
            "delta_r": 0.0 if previous_behavior is None else behavior["r"] - previous_behavior["r"],
            "delta_w": 0.0 if previous_behavior is None else behavior["w"] - previous_behavior["w"],
        }
        append_jsonl(output_path, row)
        rows.append(row)
        previous_vectors = {stage_name: tensor.detach().cpu() for stage_name, tensor in current_vectors.items()}
        previous_behavior = {
            "answer_accuracy": behavior["answer_accuracy"],
            "heldout_answer_accuracy": row["heldout_answer_accuracy"],
            "q": behavior["q"],
            "r": behavior["r"],
            "w": behavior["w"],
        }
        progress_bar.set_postfix(
            step=int(row["step"]),
            ans=f"{row['answer_accuracy']:.4f}",
            heldout=f"{row['heldout_answer_accuracy']:.4f}",
        )
    progress_bar.close()

    write_json(
        summary_path,
        {
            "num_checkpoints": len(rows),
            "probe_set_path": str(probe_set_path),
            "checkpoint_dir": str(effective_checkpoint_dir),
            "device": device_name,
            "birth_windows": summarize_birth_windows(rows=rows),
        },
    )
    return output_path, summary_path
