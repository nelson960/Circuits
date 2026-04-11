from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import iter_jsonl, write_json
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


def _read_sweep_rows_by_step(sweep_metrics_path: Path) -> dict[int, dict[str, Any]]:
    rows = {int(row["step"]): row for row in iter_jsonl(sweep_metrics_path)}
    if not rows:
        raise ValueError(f"No rows found in sweep metrics file: {sweep_metrics_path}")
    return rows


def _require_sweep_row(rows_by_step: dict[int, dict[str, Any]], *, step: int) -> dict[str, Any]:
    if step not in rows_by_step:
        raise KeyError(f"Step {step} not found in sweep metrics.")
    return rows_by_step[step]


def _checkpoint_path_for_step(checkpoint_dir: Path, step: int) -> Path:
    path = checkpoint_dir / f"step_{step:06d}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint for step {step} not found: {path}")
    return path


def _load_model_for_checkpoint(
    *,
    spec: TrainSpec,
    vocab_size: int,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = build_model(spec.model, vocab_size, device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def _evaluate_answer_metrics(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    residual_patch_batches: list[dict[str, torch.Tensor]] | None = None,
    head_mask: dict[int, torch.Tensor] | None = None,
    mlp_mask: dict[int, float | torch.Tensor] | None = None,
    neuron_mask: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if residual_patch_batches is not None and len(residual_patch_batches) != len(batches):
        raise ValueError(
            f"residual_patch_batches length {len(residual_patch_batches)} does not match number of batches {len(batches)}."
        )
    total_correct = 0
    total_answers = 0
    split_correct: dict[str, int] = defaultdict(int)
    split_total: dict[str, int] = defaultdict(int)
    for batch_index, batch in enumerate(batches):
        model_kwargs: dict[str, Any] = {
            "attention_mask": batch["attention_mask"],
        }
        if residual_patch_batches is not None:
            model_kwargs["residual_patch"] = residual_patch_batches[batch_index]
        if head_mask is not None:
            model_kwargs["head_mask"] = head_mask
        if mlp_mask is not None:
            model_kwargs["mlp_mask"] = mlp_mask
        if neuron_mask is not None:
            model_kwargs["neuron_mask"] = neuron_mask

        outputs = model(batch["input_ids"], **model_kwargs)
        answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        predictions = answer_logits.argmax(dim=-1)
        matches = predictions == answer_targets
        total_correct += int(matches.sum().item())
        total_answers += int(answer_targets.numel())
        for answer_index, row_index in enumerate(metadata["rows"].tolist()):
            split_name = str(batch["records"][row_index]["split"])
            split_total[split_name] += 1
            if bool(matches[answer_index].item()):
                split_correct[split_name] += 1

    if total_answers == 0:
        raise RuntimeError("No answer targets available during comparison evaluation.")
    split_answer_accuracy = {
        split_name: split_correct[split_name] / split_total[split_name]
        for split_name in sorted(split_total)
    }
    return {
        "answer_accuracy": total_correct / total_answers,
        "heldout_answer_accuracy": split_answer_accuracy.get("heldout_pairs", 0.0),
        "split_answer_accuracy": split_answer_accuracy,
        "num_answers": total_answers,
    }


@torch.no_grad()
def _collect_residual_patches(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    stage_names: list[str],
) -> list[dict[str, torch.Tensor]]:
    collected: list[dict[str, torch.Tensor]] = []
    for batch in batches:
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_residual_streams=True,
        )
        if outputs.residual_streams is None:
            raise RuntimeError("Residual patch collection requires residual_streams in model output.")
        batch_patches: dict[str, torch.Tensor] = {}
        for stage_name in stage_names:
            if stage_name not in outputs.residual_streams:
                raise KeyError(f"Residual stage {stage_name} not found in model output.")
            batch_patches[stage_name] = outputs.residual_streams[stage_name].detach()
        collected.append(batch_patches)
    return collected


def _default_stage_names(spec: TrainSpec) -> list[str]:
    stage_names = ["embedding"]
    for layer_index in range(spec.model.n_layers):
        stage_names.append(f"layer_{layer_index}_post_attn")
        stage_names.append(f"layer_{layer_index}_post_mlp")
    stage_names.append("final_norm")
    return stage_names


def _ordered_unique_heads(rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, int]]:
    selected: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        for field_name in ("top_heads_by_ablation", "top_heads_by_localization"):
            for item in row[field_name]:
                key = (int(item["layer"]), int(item["head"]))
                if key in seen:
                    continue
                seen.add(key)
                selected.append({"layer": key[0], "head": key[1]})
                if len(selected) >= top_k:
                    return selected
    return selected


def _ordered_unique_mlp_layers(rows: list[dict[str, Any]], *, top_k: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for row in rows:
        for field_name in ("top_mlps_by_ablation", "top_mlps_by_write"):
            for item in row[field_name]:
                layer_index = int(item["layer"])
                if layer_index in seen:
                    continue
                seen.add(layer_index)
                selected.append(layer_index)
                if len(selected) >= top_k:
                    return selected
    return selected


def _ordered_unique_neurons_by_layer(
    rows: list[dict[str, Any]],
    *,
    max_neurons_per_layer: int,
) -> dict[int, list[int]]:
    selected: dict[int, list[int]] = defaultdict(list)
    seen: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        for field_name in ("top_neurons_by_ablation", "top_neurons_by_write"):
            for item in row[field_name]:
                layer_index = int(item["layer"])
                neuron_index = int(item["neuron"])
                if neuron_index in seen[layer_index]:
                    continue
                if len(selected[layer_index]) >= max_neurons_per_layer:
                    continue
                seen[layer_index].add(neuron_index)
                selected[layer_index].append(neuron_index)
    return dict(sorted(selected.items()))


def _head_mask_for_single(
    *,
    spec: TrainSpec,
    layer_index: int,
    head_index: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    mask = {
        layer: torch.ones(spec.model.n_heads, device=device)
        for layer in range(spec.model.n_layers)
    }
    mask[layer_index][head_index] = 0.0
    return mask


def _mlp_mask_for_single(
    *,
    spec: TrainSpec,
    layer_index: int,
) -> dict[int, float]:
    mask = {layer: 1.0 for layer in range(spec.model.n_layers)}
    mask[layer_index] = 0.0
    return mask


def _neuron_mask_for_group(
    *,
    spec: TrainSpec,
    layer_index: int,
    neuron_indices: list[int],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    mask = torch.ones(spec.model.d_ff, device=device)
    for neuron_index in neuron_indices:
        if neuron_index < 0 or neuron_index >= spec.model.d_ff:
            raise ValueError(f"Neuron index {neuron_index} is out of range for d_ff={spec.model.d_ff}.")
        mask[neuron_index] = 0.0
    return {layer_index: mask}


def _accuracy_drop(baseline: dict[str, Any], ablated: dict[str, Any]) -> dict[str, float]:
    return {
        "answer_accuracy_drop": float(baseline["answer_accuracy"]) - float(ablated["answer_accuracy"]),
        "heldout_answer_accuracy_drop": float(baseline["heldout_answer_accuracy"]) - float(ablated["heldout_answer_accuracy"]),
    }


def compare_birth_window_checkpoints(
    *,
    config_path: Path,
    probe_set_path: Path,
    sweep_metrics_path: Path,
    target_step: int,
    source_steps: list[int],
    output_path: Path,
    checkpoint_dir: Path | None = None,
    device_name: str = "cpu",
    top_k_components: int = 6,
    max_neurons_per_layer: int = 4,
    stage_names: list[str] | None = None,
) -> Path:
    if not source_steps:
        raise ValueError("source_steps must not be empty.")
    if len(set(source_steps)) != len(source_steps):
        raise ValueError(f"source_steps contains duplicates: {source_steps}")

    spec = TrainSpec.from_path(config_path)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    effective_checkpoint_dir = checkpoint_dir or (spec.output_dir / "checkpoints")
    device = torch.device(device_name)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    probe_loader = _make_probe_loader(
        probe_records=probe_records,
        batch_size=spec.evaluation.batch_size,
        pad_token_id=vocab.pad_token_id,
    )
    analysis_batches = [move_batch_to_device(batch, device) for batch in probe_loader]
    rows_by_step = _read_sweep_rows_by_step(sweep_metrics_path)
    target_row = _require_sweep_row(rows_by_step, step=target_step)
    source_rows = [_require_sweep_row(rows_by_step, step=step) for step in source_steps]

    comparison_stage_names = stage_names or _default_stage_names(spec)
    target_checkpoint_path = _checkpoint_path_for_step(effective_checkpoint_dir, target_step)
    target_model, target_checkpoint = _load_model_for_checkpoint(
        spec=spec,
        vocab_size=len(vocab.tokens),
        checkpoint_path=target_checkpoint_path,
        device=device,
    )
    target_baseline = _evaluate_answer_metrics(model=target_model, batches=analysis_batches)

    candidate_rows = [target_row, *source_rows]
    candidate_heads = _ordered_unique_heads(candidate_rows, top_k=top_k_components)
    candidate_mlp_layers = _ordered_unique_mlp_layers(candidate_rows, top_k=min(top_k_components, spec.model.n_layers))
    candidate_neurons = _ordered_unique_neurons_by_layer(candidate_rows, max_neurons_per_layer=max_neurons_per_layer)

    comparisons: list[dict[str, Any]] = []
    for source_step, source_row in zip(source_steps, source_rows, strict=True):
        source_checkpoint_path = _checkpoint_path_for_step(effective_checkpoint_dir, source_step)
        source_model, source_checkpoint = _load_model_for_checkpoint(
            spec=spec,
            vocab_size=len(vocab.tokens),
            checkpoint_path=source_checkpoint_path,
            device=device,
        )
        source_baseline = _evaluate_answer_metrics(model=source_model, batches=analysis_batches)
        source_patches = _collect_residual_patches(
            model=source_model,
            batches=analysis_batches,
            stage_names=comparison_stage_names,
        )

        residual_patch_results: list[dict[str, Any]] = []
        for stage_name in comparison_stage_names:
            patched_metrics = _evaluate_answer_metrics(
                model=target_model,
                batches=analysis_batches,
                residual_patch_batches=[{stage_name: batch_patch[stage_name]} for batch_patch in source_patches],
            )
            residual_patch_results.append(
                {
                    "stage": stage_name,
                    "patched_answer_accuracy": float(patched_metrics["answer_accuracy"]),
                    "patched_heldout_answer_accuracy": float(patched_metrics["heldout_answer_accuracy"]),
                    "delta_vs_target_answer_accuracy": float(patched_metrics["answer_accuracy"]) - float(target_baseline["answer_accuracy"]),
                    "delta_vs_target_heldout_answer_accuracy": float(patched_metrics["heldout_answer_accuracy"]) - float(target_baseline["heldout_answer_accuracy"]),
                    "delta_vs_source_answer_accuracy": float(patched_metrics["answer_accuracy"]) - float(source_baseline["answer_accuracy"]),
                    "delta_vs_source_heldout_answer_accuracy": float(patched_metrics["heldout_answer_accuracy"]) - float(source_baseline["heldout_answer_accuracy"]),
                }
            )

        head_comparisons: list[dict[str, Any]] = []
        for candidate in candidate_heads:
            target_ablated = _evaluate_answer_metrics(
                model=target_model,
                batches=analysis_batches,
                head_mask=_head_mask_for_single(
                    spec=spec,
                    layer_index=int(candidate["layer"]),
                    head_index=int(candidate["head"]),
                    device=device,
                ),
            )
            source_ablated = _evaluate_answer_metrics(
                model=source_model,
                batches=analysis_batches,
                head_mask=_head_mask_for_single(
                    spec=spec,
                    layer_index=int(candidate["layer"]),
                    head_index=int(candidate["head"]),
                    device=device,
                ),
            )
            head_comparisons.append(
                {
                    "layer": int(candidate["layer"]),
                    "head": int(candidate["head"]),
                    "target": _accuracy_drop(target_baseline, target_ablated),
                    "source": _accuracy_drop(source_baseline, source_ablated),
                }
            )

        mlp_comparisons: list[dict[str, Any]] = []
        for layer_index in candidate_mlp_layers:
            target_ablated = _evaluate_answer_metrics(
                model=target_model,
                batches=analysis_batches,
                mlp_mask=_mlp_mask_for_single(spec=spec, layer_index=layer_index),
            )
            source_ablated = _evaluate_answer_metrics(
                model=source_model,
                batches=analysis_batches,
                mlp_mask=_mlp_mask_for_single(spec=spec, layer_index=layer_index),
            )
            mlp_comparisons.append(
                {
                    "layer": layer_index,
                    "target": _accuracy_drop(target_baseline, target_ablated),
                    "source": _accuracy_drop(source_baseline, source_ablated),
                }
            )

        neuron_group_comparisons: list[dict[str, Any]] = []
        neuron_single_comparisons: list[dict[str, Any]] = []
        for layer_index, neuron_indices in candidate_neurons.items():
            target_group_ablated = _evaluate_answer_metrics(
                model=target_model,
                batches=analysis_batches,
                neuron_mask=_neuron_mask_for_group(
                    spec=spec,
                    layer_index=layer_index,
                    neuron_indices=neuron_indices,
                    device=device,
                ),
            )
            source_group_ablated = _evaluate_answer_metrics(
                model=source_model,
                batches=analysis_batches,
                neuron_mask=_neuron_mask_for_group(
                    spec=spec,
                    layer_index=layer_index,
                    neuron_indices=neuron_indices,
                    device=device,
                ),
            )
            neuron_group_comparisons.append(
                {
                    "layer": layer_index,
                    "neurons": neuron_indices,
                    "target": _accuracy_drop(target_baseline, target_group_ablated),
                    "source": _accuracy_drop(source_baseline, source_group_ablated),
                }
            )
            for neuron_index in neuron_indices:
                target_single_ablated = _evaluate_answer_metrics(
                    model=target_model,
                    batches=analysis_batches,
                    neuron_mask=_neuron_mask_for_group(
                        spec=spec,
                        layer_index=layer_index,
                        neuron_indices=[neuron_index],
                        device=device,
                    ),
                )
                source_single_ablated = _evaluate_answer_metrics(
                    model=source_model,
                    batches=analysis_batches,
                    neuron_mask=_neuron_mask_for_group(
                        spec=spec,
                        layer_index=layer_index,
                        neuron_indices=[neuron_index],
                        device=device,
                    ),
                )
                neuron_single_comparisons.append(
                    {
                        "layer": layer_index,
                        "neuron": neuron_index,
                        "target": _accuracy_drop(target_baseline, target_single_ablated),
                        "source": _accuracy_drop(source_baseline, source_single_ablated),
                    }
                )

        comparisons.append(
            {
                "source_step": source_step,
                "source_checkpoint_path": str(source_checkpoint_path),
                "source_checkpoint_recorded_step": int(source_checkpoint["step"]),
                "source_sweep_row": source_row,
                "source_baseline": source_baseline,
                "residual_patch_results": residual_patch_results,
                "head_ablation_comparisons": head_comparisons,
                "mlp_ablation_comparisons": mlp_comparisons,
                "neuron_group_ablation_comparisons": neuron_group_comparisons,
                "neuron_single_ablation_comparisons": neuron_single_comparisons,
            }
        )

    payload = {
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "sweep_metrics_path": str(sweep_metrics_path),
        "checkpoint_dir": str(effective_checkpoint_dir),
        "device": device_name,
        "target_step": target_step,
        "target_checkpoint_path": str(target_checkpoint_path),
        "target_checkpoint_recorded_step": int(target_checkpoint["step"]),
        "target_sweep_row": target_row,
        "target_baseline": target_baseline,
        "comparison_stage_names": comparison_stage_names,
        "candidate_heads": candidate_heads,
        "candidate_mlp_layers": candidate_mlp_layers,
        "candidate_neurons_by_layer": candidate_neurons,
        "source_comparisons": comparisons,
    }
    write_json(output_path, payload)
    return output_path
