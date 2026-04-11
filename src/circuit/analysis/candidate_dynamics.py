from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.shared_feature_dynamics import _load_shared_basis, _normalize_activations
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import iter_jsonl, read_json, write_json
from circuit.runtime import (
    _migrate_legacy_feedforward_state_dict,
    build_model,
    compute_lm_loss,
    load_checkpoint,
    load_model_state,
    move_batch_to_device,
    require_device,
)
from circuit.train import _compute_learning_rate
from circuit.vocab import Vocabulary


CANDIDATE_REGISTRY_SCHEMA_VERSION = 1
GRADIENT_LINK_SCHEMA_VERSION = 1
MECHANISM_REPORT_SCHEMA_VERSION = 1
BIRTH_MODEL_SCHEMA_VERSION = 1
COALITION_MAP_SCHEMA_VERSION = 1
NEURON_INTERVENTION_SCHEMA_VERSION = 1
SUBSET_DELTA_KEYS = [
    "mean_activation_mean",
    "active_fraction_mean",
    "correctness_gap_mean",
    "heldout_gap_mean",
    "structural_ood_gap_mean",
]
BIRTH_MODEL_FACTOR_SPECS = [
    {
        "name": "feature_score_drive",
        "orientation": "positive",
        "description": "Cumulative projected update in the candidate feature-score direction.",
    },
    {
        "name": "gradient_alignment",
        "orientation": "positive",
        "description": "Mean cosine between checkpoint update and the candidate feature-score gradient.",
    },
    {
        "name": "loss_utility",
        "orientation": "positive",
        "description": "Cumulative probe-loss reduction attributed to the candidate parameter scope.",
    },
    {
        "name": "component_accessibility",
        "orientation": "positive",
        "description": "Mean candidate update and loss-gradient share relative to the global update.",
    },
    {
        "name": "activation_support",
        "orientation": "positive",
        "description": "Candidate mean activation plus active fraction at the prediction cutoff.",
    },
    {
        "name": "amplification",
        "orientation": "positive",
        "description": "Positive pre-birth movement in mean activation and active fraction.",
    },
    {
        "name": "interference_cost",
        "orientation": "negative",
        "description": "Negative pre-birth feature-score and useful-movement pressure.",
    },
]
BIRTH_MODEL_FACTOR_ORIENTATION = {
    str(spec["name"]): str(spec["orientation"])
    for spec in BIRTH_MODEL_FACTOR_SPECS
}
NEURON_PARAMETER_SUFFIXES = ("fc_in.weight", "fc_in.bias", "fc_out.weight")


def _require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON list.")
    return value


def _require_non_empty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _require_int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} must be an integer.")
    return value


def _coerce_int_list(value: Any, label: str) -> list[int]:
    raw_items = _require_list(value, label)
    result: list[int] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, int):
            raise TypeError(f"{label}[{index}] must be an integer.")
        result.append(item)
    if not result:
        raise ValueError(f"{label} must not be empty.")
    return result


def _load_json_dict(path: Path, label: str) -> dict[str, Any]:
    payload = read_json(path)
    return _require_dict(payload, f"{label}: {path}")


def _validate_same_length(paths: list[Path] | None, expected: int, label: str) -> list[Path | None]:
    if paths is None:
        return [None] * expected
    if len(paths) != expected:
        raise ValueError(f"{label} count {len(paths)} does not match candidate count {expected}.")
    return list(paths)


def _sanitize_candidate_id(value: str) -> str:
    candidate_id = value.strip()
    if any(character.isspace() for character in candidate_id):
        raise ValueError(f"candidate_id must not contain whitespace: {candidate_id!r}")
    return candidate_id


def _default_candidate_id(*, family_id: int, stage_name: str, feature_ids: list[int]) -> str:
    feature_token = "_".join(str(feature_id) for feature_id in sorted(feature_ids))
    stage_token = stage_name.replace("/", "_")
    return _sanitize_candidate_id(f"family_{family_id}_{stage_token}_features_{feature_token}")


def _load_optional_payload(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    return _load_json_dict(path, label)


def _component_from_trace_summary(trace_summary: dict[str, Any], key: str) -> dict[str, Any]:
    component = trace_summary.get(key)
    if component is None:
        raise KeyError(f"feature-family trace is missing trace_summary.{key}.")
    return _require_dict(component, f"trace_summary.{key}")


def _build_parameter_groups(trace_summary: dict[str, Any]) -> list[dict[str, Any]]:
    top_head = _component_from_trace_summary(trace_summary, "top_head")
    top_mlp = _component_from_trace_summary(trace_summary, "top_mlp")
    top_neuron_group = _component_from_trace_summary(trace_summary, "top_neuron_group")
    neurons = _coerce_int_list(top_neuron_group.get("neurons"), "trace_summary.top_neuron_group.neurons")
    return [
        {
            "name": "top_head",
            "kind": "attention_head",
            "layer": _require_int(top_head.get("layer"), "trace_summary.top_head.layer"),
            "head": _require_int(top_head.get("head"), "trace_summary.top_head.head"),
        },
        {
            "name": "top_mlp",
            "kind": "mlp_block",
            "layer": _require_int(top_mlp.get("layer"), "trace_summary.top_mlp.layer"),
        },
        {
            "name": "top_neuron_group",
            "kind": "mlp_neuron_group",
            "layer": _require_int(top_neuron_group.get("layer"), "trace_summary.top_neuron_group.layer"),
            "neurons": neurons,
        },
    ]


def build_candidate_circuit_registry(
    *,
    feature_family_trace_paths: list[Path],
    subset_trajectory_paths: list[Path],
    output_path: Path,
    candidate_ids: list[str] | None = None,
    basis_paths: list[Path] | None = None,
    subset_birth_paths: list[Path] | None = None,
    family_update_link_paths: list[Path] | None = None,
) -> Path:
    if not feature_family_trace_paths:
        raise ValueError("feature_family_trace_paths must not be empty.")
    if len(feature_family_trace_paths) != len(subset_trajectory_paths):
        raise ValueError(
            "feature_family_trace_paths and subset_trajectory_paths must define the same number of candidates."
        )
    if candidate_ids is not None and len(candidate_ids) != len(feature_family_trace_paths):
        raise ValueError(f"candidate_ids count {len(candidate_ids)} does not match candidate count.")

    resolved_subset_birth_paths = _validate_same_length(
        subset_birth_paths,
        len(feature_family_trace_paths),
        "subset_birth_paths",
    )
    resolved_basis_paths = _validate_same_length(
        basis_paths,
        len(feature_family_trace_paths),
        "basis_paths",
    )
    resolved_update_link_paths = _validate_same_length(
        family_update_link_paths,
        len(feature_family_trace_paths),
        "family_update_link_paths",
    )

    candidates: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    for index, (trace_path, subset_trajectory_path) in enumerate(
        zip(feature_family_trace_paths, subset_trajectory_paths, strict=True)
    ):
        trace_payload = _load_json_dict(trace_path, "feature family trace")
        subset_payload = _load_json_dict(subset_trajectory_path, "subset trajectory")
        trace_subset = _require_dict(trace_payload.get("trace_subset"), "feature family trace trace_subset")
        trace_summary = _require_dict(trace_payload.get("trace_summary"), "feature family trace trace_summary")

        trace_feature_ids = _coerce_int_list(trace_subset.get("feature_ids"), "trace_subset.feature_ids")
        subset_feature_ids = _coerce_int_list(subset_payload.get("feature_ids"), "subset_trajectory.feature_ids")
        if sorted(trace_feature_ids) != sorted(subset_feature_ids):
            raise ValueError(
                f"Trace feature ids {trace_feature_ids} do not match subset trajectory ids {subset_feature_ids}: "
                f"{trace_path} vs {subset_trajectory_path}"
            )

        family_id = _require_int(trace_payload.get("family_id"), "feature_family_trace.family_id")
        stage_name = _require_non_empty_str(trace_payload.get("stage_name"), "feature_family_trace.stage_name")
        candidate_id = (
            _sanitize_candidate_id(candidate_ids[index])
            if candidate_ids is not None
            else _default_candidate_id(family_id=family_id, stage_name=stage_name, feature_ids=trace_feature_ids)
        )
        if candidate_id in seen_candidate_ids:
            raise ValueError(f"Duplicate candidate_id: {candidate_id}")
        seen_candidate_ids.add(candidate_id)

        subset_birth_path = resolved_subset_birth_paths[index]
        basis_path = resolved_basis_paths[index]
        subset_birth_payload = _load_optional_payload(subset_birth_path, "subset birth")
        family_update_link_path = resolved_update_link_paths[index]
        family_update_link_payload = _load_optional_payload(family_update_link_path, "family update link")

        summary_fields = {
            "family_birth_step": trace_summary.get("family_birth_step"),
            "family_useful_birth_step": trace_summary.get("family_useful_birth_step"),
            "selected_subset_answer_delta": trace_summary.get("selected_subset_answer_delta"),
            "selected_subset_heldout_delta": trace_summary.get("selected_subset_heldout_delta"),
            "selected_subset_structural_ood_delta": trace_summary.get("selected_subset_structural_ood_delta"),
        }
        candidate = {
            "candidate_id": candidate_id,
            "kind": "feature_family_subset",
            "stage_name": stage_name,
            "family_id": family_id,
            "feature_ids": trace_feature_ids,
            "feature_ids_sorted": sorted(trace_feature_ids),
            "subset_size": len(trace_feature_ids),
            "parameter_groups": _build_parameter_groups(trace_summary),
            "top_components": {
                "top_head": _component_from_trace_summary(trace_summary, "top_head"),
                "top_mlp": _component_from_trace_summary(trace_summary, "top_mlp"),
                "top_neuron_group": _component_from_trace_summary(trace_summary, "top_neuron_group"),
            },
            "summary": summary_fields,
            "source_artifacts": {
                "feature_family_trace": str(trace_path),
                "subset_trajectory": str(subset_trajectory_path),
                "shared_feature_basis": None if basis_path is None else str(basis_path),
                "subset_birth": None if subset_birth_path is None else str(subset_birth_path),
                "family_update_link": None if family_update_link_path is None else str(family_update_link_path),
            },
        }
        if subset_birth_payload is not None:
            candidate["subset_birth"] = {
                "birth_step": subset_birth_payload.get("birth_step"),
                "useful_birth_step": subset_birth_payload.get("useful_birth_step"),
                "births": subset_birth_payload.get("births"),
            }
        if family_update_link_payload is not None:
            correlation_summary = _require_dict(
                family_update_link_payload.get("correlation_summary"),
                "family_update_link.correlation_summary",
            )
            candidate["existing_update_link_summary"] = {
                "num_intervals": correlation_summary.get("num_intervals"),
                "valid_correlations_sorted": correlation_summary.get("valid_correlations_sorted"),
            }
        candidates.append(candidate)

    write_json(
        output_path,
        {
            "schema_version": CANDIDATE_REGISTRY_SCHEMA_VERSION,
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
    )
    return output_path


def _load_candidate_registry(registry_path: Path) -> dict[str, Any]:
    registry = _load_json_dict(registry_path, "candidate registry")
    schema_version = registry.get("schema_version")
    if schema_version != CANDIDATE_REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported candidate registry schema_version {schema_version}; "
            f"expected {CANDIDATE_REGISTRY_SCHEMA_VERSION}."
        )
    candidates = _require_list(registry.get("candidates"), "candidate_registry.candidates")
    if not candidates:
        raise ValueError(f"Candidate registry contains no candidates: {registry_path}")
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        candidate_dict = _require_dict(candidate, f"candidate_registry.candidates[{index}]")
        candidate_id = _sanitize_candidate_id(
            _require_non_empty_str(candidate_dict.get("candidate_id"), f"candidate_registry.candidates[{index}].candidate_id")
        )
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate_id in registry: {candidate_id}")
        seen.add(candidate_id)
    return registry


def _resolve_checkpoint_paths_by_step(checkpoint_dir: Path) -> dict[int, Path]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoint_paths = sorted(checkpoint_dir.glob("step_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No step checkpoints found in {checkpoint_dir}")
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


def _load_checkpoint_state_for_analysis(checkpoint_path: Path) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint_path, torch.device("cpu"))
    model_state = _migrate_legacy_feedforward_state_dict(payload["model_state"])
    return {
        "step": int(payload["step"]),
        "model_state": {key: value.detach().cpu() for key, value in model_state.items()},
        "checkpoint": payload,
    }


def _load_probe_batches(
    *,
    spec: TrainSpec,
    probe_set_path: Path,
    vocab: Vocabulary,
    device: torch.device,
) -> list[dict[str, Any]]:
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    loader: DataLoader[Any] = DataLoader(
        probe_records,
        batch_size=spec.evaluation.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_symbolic_kv(batch, vocab.pad_token_id),
    )
    batches = [move_batch_to_device(batch, device) for batch in loader]
    if not batches:
        raise ValueError(f"Probe set produced no batches: {probe_set_path}")
    return batches


def _compute_probe_loss(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in batches:
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            loss, token_accuracy = compute_lm_loss(
                logits=outputs.logits,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pad_token_id=pad_token_id,
            )
            token_count = int(batch["attention_mask"][:, 1:].sum().item())
            if token_count <= 0:
                raise ValueError("Probe batch has no non-padding next-token targets.")
            total_loss += float(loss.item()) * token_count
            total_tokens += token_count
    if total_tokens <= 0:
        raise ValueError("Probe set has no non-padding next-token targets.")
    return {
        "loss": total_loss / total_tokens,
        "num_tokens": total_tokens,
        "num_batches": len(batches),
    }


def _compute_probe_loss_and_gradients(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, Any]:
    model.eval()
    model.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_tokens = 0
    for batch in batches:
        outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
        loss, token_accuracy = compute_lm_loss(
            logits=outputs.logits,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pad_token_id=pad_token_id,
        )
        token_count = int(batch["attention_mask"][:, 1:].sum().item())
        if token_count <= 0:
            raise ValueError("Probe batch has no non-padding next-token targets.")
        (loss * token_count).backward()
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count
    if total_tokens <= 0:
        raise ValueError("Probe set has no non-padding next-token targets.")

    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter.grad is None:
            raise RuntimeError(f"Parameter has no gradient after probe loss backward: {name}")
        gradients[name] = parameter.grad.detach().cpu().float() / float(total_tokens)
    model.zero_grad(set_to_none=True)
    return {
        "loss": total_loss / total_tokens,
        "num_tokens": total_tokens,
        "num_batches": len(batches),
        "gradients": gradients,
    }


def _compute_feature_score_and_gradients(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    basis: dict[str, Any],
    stage_name: str,
    feature_ids: list[int],
) -> dict[str, Any]:
    if str(basis["stage_name"]) != stage_name:
        raise ValueError(f"Basis stage {basis['stage_name']} does not match candidate stage {stage_name}.")
    if not feature_ids:
        raise ValueError("feature_ids must not be empty.")
    num_features = int(basis["num_features"])
    invalid_feature_ids = [feature_id for feature_id in feature_ids if feature_id < 0 or feature_id >= num_features]
    if invalid_feature_ids:
        raise ValueError(f"Feature ids out of range for basis with {num_features} features: {invalid_feature_ids}")

    sae = basis["sae"]
    for parameter in sae.parameters():
        parameter.requires_grad_(False)
    normalization_mean = basis["normalization_mean"]
    normalization_std = basis["normalization_std"]
    feature_index = torch.tensor(sorted(feature_ids), device=normalization_mean.device, dtype=torch.long)

    model.eval()
    model.zero_grad(set_to_none=True)
    total_score: torch.Tensor | None = None
    total_feature_values = 0
    total_answers = 0
    for batch in batches:
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_residual_streams=True,
        )
        if outputs.residual_streams is None:
            raise RuntimeError("Feature-score gradient requires residual streams.")
        if stage_name not in outputs.residual_streams:
            raise KeyError(f"Residual stage {stage_name} not found in model outputs.")
        _, _, metadata = extract_answer_logits(outputs.logits, batch)
        rows = metadata["rows"]
        prediction_positions = metadata["prediction_positions"]
        selected_stage = outputs.residual_streams[stage_name][rows, prediction_positions, :]
        normalized = _normalize_activations(selected_stage, normalization_mean, normalization_std)
        features = torch.relu(sae.encoder(normalized))
        selected_features = features.index_select(1, feature_index)
        batch_score = selected_features.sum()
        total_score = batch_score if total_score is None else total_score + batch_score
        total_feature_values += int(selected_features.numel())
        total_answers += int(selected_features.size(0))

    if total_score is None or total_feature_values <= 0:
        raise ValueError("Feature-score gradient had no feature activations to score.")
    mean_score = total_score / float(total_feature_values)
    mean_score.backward()
    gradients: dict[str, torch.Tensor] = {}
    zero_gradient_parameter_names: list[str] = []
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter.grad is None:
            gradients[name] = torch.zeros_like(parameter.detach(), device=torch.device("cpu"), dtype=torch.float32)
            zero_gradient_parameter_names.append(name)
        else:
            gradients[name] = parameter.grad.detach().cpu().float()
    model.zero_grad(set_to_none=True)
    return {
        "score": float(mean_score.detach().cpu().item()),
        "num_answers": total_answers,
        "num_feature_values": total_feature_values,
        "zero_gradient_parameter_names": zero_gradient_parameter_names,
        "gradients": gradients,
    }


def _extract_n_heads(checkpoint_payload: dict[str, Any]) -> int:
    config = checkpoint_payload.get("config")
    if not isinstance(config, dict):
        raise KeyError("Checkpoint payload is missing dict config.")
    train_spec = config.get("train_spec")
    if isinstance(train_spec, dict):
        model_config = train_spec.get("model")
    else:
        model_config = config.get("model")
    if not isinstance(model_config, dict):
        raise KeyError("Checkpoint payload config is missing model config.")
    n_heads = model_config.get("n_heads")
    if not isinstance(n_heads, int) or n_heads <= 0:
        raise ValueError(f"Checkpoint model config has invalid n_heads: {n_heads}")
    return n_heads


def _require_state_key(state: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state:
        raise KeyError(f"Missing model_state key: {key}")
    return state[key]


def _new_false_mask(tensor: torch.Tensor) -> torch.Tensor:
    return torch.zeros(tuple(tensor.shape), dtype=torch.bool)


def _mask_all_parameters(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.ones(tuple(value.shape), dtype=torch.bool) for key, value in state.items()}


def _mask_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    keys = [key for key in sorted(state) if key.startswith(prefix)]
    if not keys:
        raise KeyError(f"No model_state keys found for prefix {prefix}")
    return {key: torch.ones(tuple(state[key].shape), dtype=torch.bool) for key in keys}


def _mask_attention_head(
    *,
    state: dict[str, torch.Tensor],
    layer: int,
    head: int,
    n_heads: int,
) -> dict[str, torch.Tensor]:
    q_weight_key = f"blocks.{layer}.attn.q_proj.weight"
    q_weight = _require_state_key(state, q_weight_key)
    d_model = int(q_weight.shape[0])
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads} for layer {layer}.")
    head_dim = d_model // n_heads
    head_start = head * head_dim
    head_end = head_start + head_dim
    if head < 0 or head_end > d_model:
        raise ValueError(f"Head {head} is out of range for n_heads={n_heads}.")

    masks: dict[str, torch.Tensor] = {}
    for suffix in ["q_proj", "k_proj", "v_proj"]:
        weight_key = f"blocks.{layer}.attn.{suffix}.weight"
        bias_key = f"blocks.{layer}.attn.{suffix}.bias"
        weight = _require_state_key(state, weight_key)
        bias = _require_state_key(state, bias_key)
        weight_mask = _new_false_mask(weight)
        bias_mask = _new_false_mask(bias)
        weight_mask[head_start:head_end, :] = True
        bias_mask[head_start:head_end] = True
        masks[weight_key] = weight_mask
        masks[bias_key] = bias_mask

    out_weight_key = f"blocks.{layer}.attn.out_proj.weight"
    out_weight = _require_state_key(state, out_weight_key)
    out_mask = _new_false_mask(out_weight)
    out_mask[:, head_start:head_end] = True
    masks[out_weight_key] = out_mask
    return masks


def _mask_neuron_group(
    *,
    state: dict[str, torch.Tensor],
    layer: int,
    neurons: list[int],
) -> dict[str, torch.Tensor]:
    if not neurons:
        raise ValueError("neurons must not be empty.")
    fc_in_weight_key = f"blocks.{layer}.ff.fc_in.weight"
    fc_in_bias_key = f"blocks.{layer}.ff.fc_in.bias"
    fc_out_weight_key = f"blocks.{layer}.ff.fc_out.weight"
    fc_in_weight = _require_state_key(state, fc_in_weight_key)
    fc_in_bias = _require_state_key(state, fc_in_bias_key)
    fc_out_weight = _require_state_key(state, fc_out_weight_key)
    d_ff = int(fc_in_weight.shape[0])
    invalid_neurons = [neuron for neuron in neurons if neuron < 0 or neuron >= d_ff]
    if invalid_neurons:
        raise ValueError(f"Neuron ids out of range for layer {layer}, d_ff={d_ff}: {invalid_neurons}")
    row_index = torch.tensor(sorted(set(neurons)), dtype=torch.long)

    fc_in_weight_mask = _new_false_mask(fc_in_weight)
    fc_in_bias_mask = _new_false_mask(fc_in_bias)
    fc_out_weight_mask = _new_false_mask(fc_out_weight)
    fc_in_weight_mask.index_fill_(0, row_index, True)
    fc_in_bias_mask.index_fill_(0, row_index, True)
    fc_out_weight_mask.index_fill_(1, row_index, True)
    return {
        fc_in_weight_key: fc_in_weight_mask,
        fc_in_bias_key: fc_in_bias_mask,
        fc_out_weight_key: fc_out_weight_mask,
    }


def _merge_masks(mask_payloads: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for masks in mask_payloads:
        for key, mask in masks.items():
            if key not in merged:
                merged[key] = mask.clone()
                continue
            if tuple(merged[key].shape) != tuple(mask.shape):
                raise ValueError(f"Mask shape mismatch for key {key}: {tuple(merged[key].shape)} vs {tuple(mask.shape)}")
            merged[key] |= mask
    if not merged:
        raise ValueError("Cannot merge an empty set of parameter masks.")
    return merged


def _build_component_masks(
    *,
    state: dict[str, torch.Tensor],
    candidate: dict[str, Any],
    n_heads: int,
) -> dict[str, dict[str, torch.Tensor]]:
    parameter_groups = _require_list(candidate.get("parameter_groups", []), "candidate.parameter_groups")
    if not parameter_groups:
        return {}
    masks_by_name: dict[str, dict[str, torch.Tensor]] = {}
    for index, group_raw in enumerate(parameter_groups):
        group = _require_dict(group_raw, f"candidate.parameter_groups[{index}]")
        name = _sanitize_candidate_id(
            _require_non_empty_str(group.get("name"), f"candidate.parameter_groups[{index}].name")
        )
        kind = _require_non_empty_str(group.get("kind"), f"candidate.parameter_groups[{index}].kind")
        if name in masks_by_name:
            raise ValueError(f"Duplicate parameter group name in candidate {candidate['candidate_id']}: {name}")
        if kind == "attention_head":
            masks_by_name[name] = _mask_attention_head(
                state=state,
                layer=_require_int(group.get("layer"), f"candidate.parameter_groups[{index}].layer"),
                head=_require_int(group.get("head"), f"candidate.parameter_groups[{index}].head"),
                n_heads=n_heads,
            )
        elif kind == "mlp_block":
            layer = _require_int(group.get("layer"), f"candidate.parameter_groups[{index}].layer")
            masks_by_name[name] = _mask_prefix(state, f"blocks.{layer}.ff.")
        elif kind == "mlp_neuron_group":
            masks_by_name[name] = _mask_neuron_group(
                state=state,
                layer=_require_int(group.get("layer"), f"candidate.parameter_groups[{index}].layer"),
                neurons=_coerce_int_list(group.get("neurons"), f"candidate.parameter_groups[{index}].neurons"),
            )
        else:
            raise ValueError(f"Unsupported parameter group kind: {kind}")

    masks_by_name["candidate_union"] = _merge_masks(list(masks_by_name.values()))
    return masks_by_name


def _validate_state_and_gradient_keys(
    *,
    previous_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
) -> None:
    previous_keys = set(previous_state)
    current_keys = set(current_state)
    gradient_keys = set(gradients)
    if previous_keys != current_keys:
        missing_current = sorted(previous_keys - current_keys)
        extra_current = sorted(current_keys - previous_keys)
        raise ValueError(f"Checkpoint state keys differ. missing_current={missing_current} extra_current={extra_current}")
    if previous_keys != gradient_keys:
        missing_gradients = sorted(previous_keys - gradient_keys)
        extra_gradients = sorted(gradient_keys - previous_keys)
        raise ValueError(
            f"Gradient keys do not match checkpoint state keys. "
            f"missing_gradients={missing_gradients} extra_gradients={extra_gradients}"
        )


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _projection_stats(
    *,
    previous_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    weight_decay: float,
) -> dict[str, Any]:
    update_sq_sum = 0.0
    grad_sq_sum = 0.0
    param_sq_sum = 0.0
    grad_dot_update = 0.0
    param_dot_update = 0.0
    num_parameters = 0
    for key, mask in masks.items():
        if key not in previous_state or key not in current_state or key not in gradients:
            raise KeyError(f"Projection mask references key not present in state/gradients: {key}")
        if tuple(mask.shape) != tuple(previous_state[key].shape):
            raise ValueError(f"Mask shape mismatch for {key}: {tuple(mask.shape)} vs {tuple(previous_state[key].shape)}")
        selected = mask.bool()
        count = int(selected.sum().item())
        if count == 0:
            continue
        previous = previous_state[key].float()[selected]
        current = current_state[key].float()[selected]
        gradient = gradients[key].float()[selected]
        update = current - previous
        update_sq_sum += float(torch.dot(update, update).item())
        grad_sq_sum += float(torch.dot(gradient, gradient).item())
        param_sq_sum += float(torch.dot(previous, previous).item())
        grad_dot_update += float(torch.dot(gradient, update).item())
        param_dot_update += float(torch.dot(previous, update).item())
        num_parameters += count
    if num_parameters == 0:
        raise ValueError("Projection mask selected zero parameters.")

    update_l2_norm = update_sq_sum ** 0.5
    gradient_l2_norm = grad_sq_sum ** 0.5
    previous_param_l2_norm = param_sq_sum ** 0.5
    negative_gradient_dot_update = -grad_dot_update
    regularization_linearized_delta = weight_decay * param_dot_update
    combined_linearized_delta = grad_dot_update + regularization_linearized_delta
    return {
        "num_parameters": num_parameters,
        "update_l2_norm": update_l2_norm,
        "gradient_l2_norm": gradient_l2_norm,
        "previous_param_l2_norm": previous_param_l2_norm,
        "grad_dot_update": grad_dot_update,
        "negative_grad_dot_update": negative_gradient_dot_update,
        "loss_linearized_delta": grad_dot_update,
        "loss_reduction_linearized": -grad_dot_update,
        "update_negative_gradient_cosine": _safe_ratio(
            negative_gradient_dot_update,
            update_l2_norm * gradient_l2_norm,
        ),
        "regularization_linearized_delta": regularization_linearized_delta,
        "regularization_reduction_linearized": -regularization_linearized_delta,
        "combined_linearized_delta": combined_linearized_delta,
        "combined_reduction_linearized": -combined_linearized_delta,
        "param_dot_update": param_dot_update,
        "projected_step_size_on_negative_gradient": _safe_ratio(
            negative_gradient_dot_update,
            grad_sq_sum,
        ),
    }


def _score_projection_stats(
    *,
    previous_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    score_gradients: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> dict[str, Any]:
    raw = _projection_stats(
        previous_state=previous_state,
        current_state=current_state,
        gradients=score_gradients,
        masks=masks,
        weight_decay=0.0,
    )
    update_l2_norm = float(raw["update_l2_norm"])
    score_gradient_l2_norm = float(raw["gradient_l2_norm"])
    score_grad_dot_update = float(raw["grad_dot_update"])
    return {
        "num_parameters": raw["num_parameters"],
        "update_l2_norm": update_l2_norm,
        "score_gradient_l2_norm": score_gradient_l2_norm,
        "previous_param_l2_norm": raw["previous_param_l2_norm"],
        "score_grad_dot_update": score_grad_dot_update,
        "score_linearized_delta": score_grad_dot_update,
        "update_score_gradient_cosine": _safe_ratio(
            score_grad_dot_update,
            update_l2_norm * score_gradient_l2_norm,
        ),
        "projected_step_size_on_score_gradient": _safe_ratio(
            score_grad_dot_update,
            score_gradient_l2_norm * score_gradient_l2_norm,
        ),
    }


def _add_global_shares(group_stats: dict[str, dict[str, Any]]) -> None:
    global_stats = group_stats["global"]
    for group_name, stats in group_stats.items():
        if group_name == "global":
            continue
        stats["update_l2_share_global"] = _safe_ratio(
            float(stats["update_l2_norm"]),
            float(global_stats["update_l2_norm"]),
        )
        stats["gradient_l2_share_global"] = _safe_ratio(
            float(stats["gradient_l2_norm"]),
            float(global_stats["gradient_l2_norm"]),
        )
        stats["loss_reduction_share_global"] = _safe_ratio(
            float(stats["loss_reduction_linearized"]),
            float(global_stats["loss_reduction_linearized"]),
        )
        stats["combined_reduction_share_global"] = _safe_ratio(
            float(stats["combined_reduction_linearized"]),
            float(global_stats["combined_reduction_linearized"]),
        )


def _add_score_global_shares(group_stats: dict[str, dict[str, Any]]) -> None:
    global_stats = group_stats["global"]
    for group_name, stats in group_stats.items():
        if group_name == "global":
            continue
        stats["update_l2_share_global"] = _safe_ratio(
            float(stats["update_l2_norm"]),
            float(global_stats["update_l2_norm"]),
        )
        stats["score_gradient_l2_share_global"] = _safe_ratio(
            float(stats["score_gradient_l2_norm"]),
            float(global_stats["score_gradient_l2_norm"]),
        )
        stats["score_linearized_delta_share_global"] = _safe_ratio(
            float(stats["score_linearized_delta"]),
            float(global_stats["score_linearized_delta"]),
        )


def _load_sweep_rows_by_step(sweep_metrics_path: Path | None) -> dict[int, dict[str, Any]]:
    if sweep_metrics_path is None:
        return {}
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


def _candidate_rows_by_step(candidate: dict[str, Any]) -> dict[int, dict[str, Any]]:
    artifacts = _require_dict(candidate.get("source_artifacts"), "candidate.source_artifacts")
    subset_trajectory_path = Path(
        _require_non_empty_str(artifacts.get("subset_trajectory"), "candidate.source_artifacts.subset_trajectory")
    )
    payload = _load_json_dict(subset_trajectory_path, "subset trajectory")
    rows = _require_list(payload.get("rows"), "subset_trajectory.rows")
    rows_by_step: dict[int, dict[str, Any]] = {}
    expected_feature_ids = sorted(_coerce_int_list(candidate.get("feature_ids"), "candidate.feature_ids"))
    for index, raw_row in enumerate(rows):
        row = _require_dict(raw_row, f"subset_trajectory.rows[{index}]")
        row_step = _require_int(row.get("step"), f"subset_trajectory.rows[{index}].step")
        row_feature_ids = sorted(_coerce_int_list(row.get("feature_ids"), f"subset_trajectory.rows[{index}].feature_ids"))
        if row_feature_ids != expected_feature_ids:
            raise ValueError(
                f"Subset trajectory row feature ids {row_feature_ids} do not match candidate ids {expected_feature_ids} "
                f"at step {row_step}."
            )
        if row_step in rows_by_step:
            raise ValueError(f"Duplicate subset trajectory step {row_step}: {subset_trajectory_path}")
        rows_by_step[row_step] = row
    if len(rows_by_step) < 2:
        raise ValueError(f"Subset trajectory must contain at least two steps: {subset_trajectory_path}")
    return rows_by_step


def _candidate_state_deltas(previous_row: dict[str, Any], current_row: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in SUBSET_DELTA_KEYS:
        if key not in previous_row or key not in current_row:
            raise KeyError(f"Subset trajectory rows are missing required metric: {key}")
        delta_name = key.removesuffix("_mean") + "_delta"
        deltas[delta_name] = float(current_row[key]) - float(previous_row[key])
    deltas["useful_delta"] = deltas["correctness_gap_delta"] + deltas["heldout_gap_delta"]
    return deltas


def _sweep_deltas_for_step(rows_by_step: dict[int, dict[str, Any]], step: int) -> dict[str, float] | None:
    if not rows_by_step:
        return None
    if step not in rows_by_step:
        raise KeyError(f"Sweep metrics are missing step {step}.")
    row = rows_by_step[step]
    required_keys = [
        "delta_answer_accuracy",
        "delta_heldout_answer_accuracy",
        "delta_q",
        "delta_r",
        "delta_w",
    ]
    missing = [key for key in required_keys if key not in row]
    if missing:
        raise KeyError(f"Sweep metrics row for step {step} is missing keys {missing}.")
    return {key: float(row[key]) for key in required_keys}


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


def _import_matplotlib() -> tuple[Any, Any]:
    cache_dir = Path(tempfile.gettempdir()) / "circuit_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return matplotlib, plt


def _candidate_summary(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_rows:
        raise ValueError("candidate_rows must not be empty.")
    signals = {
        "candidate_union_loss_reduction": lambda row: row["parameter_projections"]["candidate_union"][
            "loss_reduction_linearized"
        ],
        "candidate_union_combined_reduction": lambda row: row["parameter_projections"]["candidate_union"][
            "combined_reduction_linearized"
        ],
        "candidate_union_update_negative_gradient_cosine": lambda row: row["parameter_projections"]["candidate_union"][
            "update_negative_gradient_cosine"
        ],
        "candidate_union_update_l2_share_global": lambda row: row["parameter_projections"]["candidate_union"][
            "update_l2_share_global"
        ],
        "candidate_union_gradient_l2_share_global": lambda row: row["parameter_projections"]["candidate_union"][
            "gradient_l2_share_global"
        ],
    }
    responses = {
        "candidate_useful_delta": lambda row: row["candidate_state_deltas"]["useful_delta"],
        "candidate_correctness_gap_delta": lambda row: row["candidate_state_deltas"]["correctness_gap_delta"],
        "candidate_heldout_gap_delta": lambda row: row["candidate_state_deltas"]["heldout_gap_delta"],
    }
    correlations: dict[str, dict[str, Any]] = {}
    valid: list[dict[str, Any]] = []
    for response_name, response_getter in responses.items():
        response_correlations: dict[str, Any] = {}
        for signal_name, signal_getter in signals.items():
            pairs = [
                (response_getter(row), signal_getter(row))
                for row in candidate_rows
                if signal_getter(row) is not None
            ]
            correlation = _compute_pearson_correlation(
                [float(left) for left, _ in pairs],
                [float(right) for _, right in pairs],
            )
            response_correlations[signal_name] = correlation
            if correlation["status"] == "ok":
                valid.append(
                    {
                        "response": response_name,
                        "signal": signal_name,
                        "value": float(correlation["value"]),
                        "abs_value": abs(float(correlation["value"])),
                    }
                )
        correlations[response_name] = response_correlations
    valid.sort(key=lambda item: float(item["abs_value"]), reverse=True)
    summary = {
        "num_intervals": len(candidate_rows),
        "sum_useful_delta": sum(float(row["candidate_state_deltas"]["useful_delta"]) for row in candidate_rows),
        "sum_correctness_gap_delta": sum(
            float(row["candidate_state_deltas"]["correctness_gap_delta"]) for row in candidate_rows
        ),
        "sum_heldout_gap_delta": sum(
            float(row["candidate_state_deltas"]["heldout_gap_delta"]) for row in candidate_rows
        ),
        "sum_mean_activation_delta": sum(
            float(row["candidate_state_deltas"]["mean_activation_delta"]) for row in candidate_rows
        ),
        "sum_active_fraction_delta": sum(
            float(row["candidate_state_deltas"]["active_fraction_delta"]) for row in candidate_rows
        ),
        "sum_loss_reduction_linearized": sum(
            float(row["parameter_projections"]["candidate_union"]["loss_reduction_linearized"])
            for row in candidate_rows
        ),
        "sum_combined_reduction_linearized": sum(
            float(row["parameter_projections"]["candidate_union"]["combined_reduction_linearized"])
            for row in candidate_rows
        ),
        "mean_update_negative_gradient_cosine": sum(
            float(row["parameter_projections"]["candidate_union"]["update_negative_gradient_cosine"] or 0.0)
            for row in candidate_rows
        )
        / len(candidate_rows),
        "correlations": correlations,
        "valid_correlations_sorted": valid[:16],
    }
    feature_score_rows = [
        row
        for row in candidate_rows
        if row.get("feature_score_projection", {}).get("status") == "computed"
    ]
    if feature_score_rows:
        summary["feature_score_summary"] = {
            "num_intervals": len(feature_score_rows),
            "sum_observed_mean_activation_delta": sum(
                float(row["feature_score_projection"]["observed_mean_activation_delta"])
                for row in feature_score_rows
            ),
            "sum_candidate_union_score_linearized_delta": sum(
                float(
                    row["feature_score_projection"]["parameter_projections"]["candidate_union"][
                        "score_linearized_delta"
                    ]
                )
                for row in feature_score_rows
            ),
            "mean_candidate_union_update_score_gradient_cosine": sum(
                float(
                    row["feature_score_projection"]["parameter_projections"]["candidate_union"][
                        "update_score_gradient_cosine"
                    ]
                    or 0.0
                )
                for row in feature_score_rows
            )
            / len(feature_score_rows),
        }
    return summary


def _group_rows_by_candidate(interval_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in interval_rows:
        grouped[str(row["candidate_id"])].append(row)
    return {candidate_id: sorted(rows, key=lambda row: int(row["step"])) for candidate_id, rows in sorted(grouped.items())}


def _cumulative(values: list[float]) -> list[float]:
    total = 0.0
    cumulative: list[float] = []
    for value in values:
        total += value
        cumulative.append(total)
    return cumulative


def _render_gradient_link_cumulative_plot(
    *,
    interval_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    grouped = _group_rows_by_candidate(interval_rows)
    if not grouped:
        raise ValueError("Cannot render cumulative plot without interval rows.")
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for candidate_id, rows in grouped.items():
        steps = [int(row["step"]) for row in rows]
        useful = _cumulative([float(row["candidate_state_deltas"]["useful_delta"]) for row in rows])
        axes[0].plot(steps, useful, linewidth=2.0, marker="o", label=candidate_id)
        feature_rows = [row for row in rows if row["feature_score_projection"]["status"] == "computed"]
        if feature_rows:
            feature_steps = [int(row["step"]) for row in feature_rows]
            score_delta = _cumulative(
                [
                    float(
                        row["feature_score_projection"]["parameter_projections"]["candidate_union"][
                            "score_linearized_delta"
                        ]
                    )
                    for row in feature_rows
                ]
            )
            axes[1].plot(feature_steps, score_delta, linewidth=2.0, marker="o", label=candidate_id)
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[0].set_title("Candidate useful-delta accumulation")
    axes[0].set_ylabel("Cumulative useful delta")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[1].set_title("Feature-score linearized update accumulation")
    axes[1].set_xlabel("Checkpoint step")
    axes[1].set_ylabel("Cumulative score dot update")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_gradient_link_interval_plot(
    *,
    interval_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    grouped = _group_rows_by_candidate(interval_rows)
    if not grouped:
        raise ValueError("Cannot render interval plot without interval rows.")
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for candidate_id, rows in grouped.items():
        steps = [int(row["step"]) for row in rows]
        useful = [float(row["candidate_state_deltas"]["useful_delta"]) for row in rows]
        loss_reduction = [
            float(row["parameter_projections"]["candidate_union"]["loss_reduction_linearized"])
            for row in rows
        ]
        feature_score = [
            None
            if row["feature_score_projection"]["status"] != "computed"
            else float(
                row["feature_score_projection"]["parameter_projections"]["candidate_union"][
                    "score_linearized_delta"
                ]
            )
            for row in rows
        ]
        feature_steps = [step for step, value in zip(steps, feature_score, strict=True) if value is not None]
        feature_values = [float(value) for value in feature_score if value is not None]
        axes[0].plot(steps, useful, linewidth=1.7, marker="o", label=candidate_id)
        axes[1].plot(steps, loss_reduction, linewidth=1.7, marker="o", label=candidate_id)
        if feature_values:
            axes[2].plot(feature_steps, feature_values, linewidth=1.7, marker="o", label=candidate_id)
    axes[0].set_title("Candidate state movement per interval")
    axes[0].set_ylabel("Useful delta")
    axes[1].set_title("Probe-loss gradient link per interval")
    axes[1].set_ylabel("Loss reduction")
    axes[2].set_title("Feature-score gradient link per interval")
    axes[2].set_ylabel("Score dot update")
    axes[2].set_xlabel("Checkpoint step")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_gradient_link_alignment_plot(
    *,
    interval_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    grouped = _group_rows_by_candidate(interval_rows)
    if not grouped:
        raise ValueError("Cannot render alignment plot without interval rows.")
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for candidate_id, rows in grouped.items():
        steps = [int(row["step"]) for row in rows]
        loss_cosine = [
            float(row["parameter_projections"]["candidate_union"]["update_negative_gradient_cosine"] or 0.0)
            for row in rows
        ]
        score_pairs = [
            (
                int(row["step"]),
                row["feature_score_projection"]["parameter_projections"]["candidate_union"][
                    "update_score_gradient_cosine"
                ],
            )
            for row in rows
            if row["feature_score_projection"]["status"] == "computed"
        ]
        axes[0].plot(steps, loss_cosine, linewidth=1.7, marker="o", label=candidate_id)
        if score_pairs:
            score_steps = [step for step, _ in score_pairs]
            score_cosine = [float(value or 0.0) for _, value in score_pairs]
            axes[1].plot(score_steps, score_cosine, linewidth=1.7, marker="o", label=candidate_id)
    axes[0].set_title("Update alignment with probe-loss reduction")
    axes[0].set_ylabel("cos(update, -grad L)")
    axes[1].set_title("Update alignment with feature-score increase")
    axes[1].set_ylabel("cos(update, grad score)")
    axes[1].set_xlabel("Checkpoint step")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_gradient_link_scatter_plot(
    *,
    interval_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    grouped = _group_rows_by_candidate(interval_rows)
    if not grouped:
        raise ValueError("Cannot render scatter plot without interval rows.")
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for candidate_id, rows in grouped.items():
        feature_rows = [row for row in rows if row["feature_score_projection"]["status"] == "computed"]
        if feature_rows:
            observed = [
                float(row["feature_score_projection"]["observed_mean_activation_delta"])
                for row in feature_rows
            ]
            predicted = [
                float(
                    row["feature_score_projection"]["parameter_projections"]["candidate_union"][
                        "score_linearized_delta"
                    ]
                )
                for row in feature_rows
            ]
            axes[0].scatter(observed, predicted, label=candidate_id, alpha=0.75)
        useful = [float(row["candidate_state_deltas"]["useful_delta"]) for row in rows]
        loss_reduction = [
            float(row["parameter_projections"]["candidate_union"]["loss_reduction_linearized"])
            for row in rows
        ]
        axes[1].scatter(useful, loss_reduction, label=candidate_id, alpha=0.75)
    axes[0].set_title("Feature score: observed vs linearized")
    axes[0].set_xlabel("Observed mean-activation delta")
    axes[0].set_ylabel("Score dot update")
    axes[1].set_title("Usefulness vs loss-gradient link")
    axes[1].set_xlabel("Useful delta")
    axes[1].set_ylabel("Loss reduction")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_gradient_link_plots(
    *,
    interval_rows: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Path]:
    stem = output_path.stem
    return {
        "cumulative": _render_gradient_link_cumulative_plot(
            interval_rows=interval_rows,
            output_path=output_path.with_name(f"{stem}_cumulative.svg"),
        ),
        "intervals": _render_gradient_link_interval_plot(
            interval_rows=interval_rows,
            output_path=output_path.with_name(f"{stem}_intervals.svg"),
        ),
        "alignment": _render_gradient_link_alignment_plot(
            interval_rows=interval_rows,
            output_path=output_path.with_name(f"{stem}_alignment.svg"),
        ),
        "scatter": _render_gradient_link_scatter_plot(
            interval_rows=interval_rows,
            output_path=output_path.with_name(f"{stem}_scatter.svg"),
        ),
    }


def _feature_compare_rows_by_feature(feature_compare_path: Path) -> dict[int, dict[str, Any]]:
    payload = _load_json_dict(feature_compare_path, "feature compare")
    rows = _require_list(payload.get("diff_rows"), "feature_compare.diff_rows")
    rows_by_feature: dict[int, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        row = _require_dict(raw_row, f"feature_compare.diff_rows[{index}]")
        feature_id = _require_int(row.get("feature_id"), f"feature_compare.diff_rows[{index}].feature_id")
        if feature_id in rows_by_feature:
            raise ValueError(f"Duplicate feature_id {feature_id} in feature compare: {feature_compare_path}")
        required = [
            "mean_activation_delta",
            "active_fraction_delta",
            "correctness_gap_delta",
            "heldout_gap_delta",
            "structural_ood_gap_delta",
        ]
        missing = [key for key in required if key not in row]
        if missing:
            raise KeyError(f"Feature compare row for feature {feature_id} is missing keys {missing}: {feature_compare_path}")
        rows_by_feature[feature_id] = {
            **row,
            "useful_delta": float(row["correctness_gap_delta"]) + float(row["heldout_gap_delta"]),
        }
    if not rows_by_feature:
        raise ValueError(f"Feature compare has no rows: {feature_compare_path}")
    return rows_by_feature


def _rank_family_member_rows(member_rows: list[dict[str, Any]], ranking_name: str) -> list[dict[str, Any]]:
    if ranking_name == "by_useful_delta":
        return sorted(member_rows, key=lambda row: float(row["useful_delta"]), reverse=True)
    if ranking_name == "by_heldout_gap_delta":
        return sorted(member_rows, key=lambda row: float(row["heldout_gap_delta"]), reverse=True)
    if ranking_name == "by_correctness_gap_delta":
        return sorted(member_rows, key=lambda row: float(row["correctness_gap_delta"]), reverse=True)
    if ranking_name == "by_abs_mean_activation_delta":
        return sorted(member_rows, key=lambda row: abs(float(row["mean_activation_delta"])), reverse=True)
    raise ValueError(f"Unsupported candidate sweep ranking_name: {ranking_name}")


def _aggregate_compare_proxy(selected_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not selected_rows:
        raise ValueError("selected_rows must not be empty.")
    correctness = sum(float(row["correctness_gap_delta"]) for row in selected_rows)
    heldout = sum(float(row["heldout_gap_delta"]) for row in selected_rows)
    return {
        "mean_activation_delta": sum(float(row["mean_activation_delta"]) for row in selected_rows),
        "active_fraction_delta": sum(float(row["active_fraction_delta"]) for row in selected_rows),
        "correctness_gap_delta": correctness,
        "heldout_gap_delta": heldout,
        "structural_ood_gap_delta": sum(float(row["structural_ood_gap_delta"]) for row in selected_rows),
        "useful_delta": correctness + heldout,
    }


def _load_feature_trajectory_rows_by_step(
    *,
    trajectories_path: Path,
    feature_ids: list[int],
    stage_name: str,
) -> dict[int, list[dict[str, Any]]]:
    requested = set(feature_ids)
    rows_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen_pairs: set[tuple[int, int]] = set()
    for index, raw_row in enumerate(iter_jsonl(trajectories_path)):
        row = _require_dict(raw_row, f"feature_trajectories[{index}]")
        row_stage = _require_non_empty_str(row.get("stage_name"), f"feature_trajectories[{index}].stage_name")
        if row_stage != stage_name:
            raise ValueError(f"Trajectory row stage {row_stage} does not match requested stage {stage_name}: {trajectories_path}")
        feature_id = _require_int(row.get("feature_id"), f"feature_trajectories[{index}].feature_id")
        if feature_id not in requested:
            continue
        step = _require_int(row.get("step"), f"feature_trajectories[{index}].step")
        pair = (feature_id, step)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate feature trajectory row for feature {feature_id}, step {step}: {trajectories_path}")
        seen_pairs.add(pair)
        rows_by_step[step].append(row)
    if not rows_by_step:
        raise ValueError(f"No trajectory rows found for features {sorted(feature_ids)} in {trajectories_path}")
    missing_by_step = {
        step: sorted(requested - {int(row["feature_id"]) for row in rows})
        for step, rows in rows_by_step.items()
        if {int(row["feature_id"]) for row in rows} != requested
    }
    if missing_by_step:
        raise ValueError(f"Feature trajectory is incomplete for selected features: {missing_by_step}")
    return dict(sorted(rows_by_step.items()))


def _aggregate_subset_trajectory_rows(
    *,
    trajectories_path: Path,
    feature_ids: list[int],
    stage_name: str,
) -> list[dict[str, Any]]:
    rows_by_step = _load_feature_trajectory_rows_by_step(
        trajectories_path=trajectories_path,
        feature_ids=feature_ids,
        stage_name=stage_name,
    )
    metric_names = [
        "mean_activation",
        "active_fraction",
        "correctness_gap",
        "heldout_gap",
        "structural_ood_gap",
        "answer_direction_alignment",
        "margin_correlation",
    ]
    subset_rows: list[dict[str, Any]] = []
    for step, rows in rows_by_step.items():
        row_payload: dict[str, Any] = {
            "step": int(step),
            "feature_ids": list(feature_ids),
            "subset_size": len(feature_ids),
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in rows if metric_name in row]
            if len(values) != len(rows):
                raise KeyError(f"Metric {metric_name} is missing for at least one selected feature at step {step}.")
            row_payload[f"{metric_name}_mean"] = sum(values) / len(values)
            row_payload[f"{metric_name}_min"] = min(values)
            row_payload[f"{metric_name}_max"] = max(values)
        split_names = sorted({split_name for row in rows for split_name in row.get("split_mean_activation", {})})
        row_payload["split_mean_activation_mean"] = {}
        for split_name in split_names:
            split_values = []
            for row in rows:
                split_payload = _require_dict(row.get("split_mean_activation"), "feature trajectory split_mean_activation")
                if split_name not in split_payload:
                    raise KeyError(f"Missing split_mean_activation.{split_name} for step {step}.")
                split_values.append(float(split_payload[split_name]))
            row_payload["split_mean_activation_mean"][split_name] = sum(split_values) / len(split_values)
        subset_rows.append(row_payload)
    return subset_rows


def _write_sweep_subset_trajectory(
    *,
    trajectories_path: Path,
    output_path: Path,
    feature_ids: list[int],
    stage_name: str,
    subset_spec: dict[str, Any],
) -> Path:
    rows = _aggregate_subset_trajectory_rows(
        trajectories_path=trajectories_path,
        feature_ids=feature_ids,
        stage_name=stage_name,
    )
    write_json(
        output_path,
        {
            "trajectories_path": str(trajectories_path),
            "subset_spec": subset_spec,
            "feature_ids": feature_ids,
            "subset_size": len(feature_ids),
            "rows": rows,
        },
    )
    return output_path


def _make_sweep_candidate_id(
    *,
    stage_name: str,
    family_id: int,
    ranking_name: str,
    feature_ids: list[int],
) -> str:
    feature_token = "_".join(str(feature_id) for feature_id in feature_ids)
    return _sanitize_candidate_id(
        f"{stage_name}_family{family_id}_{ranking_name}_features_{feature_token}".replace("/", "_")
    )


def _build_sweep_registry(
    *,
    stage_names: list[str],
    families_paths: list[Path],
    feature_compare_paths: list[Path],
    trajectories_paths: list[Path],
    basis_paths: list[Path],
    output_dir: Path,
    ranking_name: str,
    subset_size: int,
    min_family_size: int,
    top_k_families: int | None,
) -> tuple[Path, list[dict[str, Any]]]:
    if subset_size <= 0:
        raise ValueError("subset_size must be positive.")
    if min_family_size <= 1:
        raise ValueError("min_family_size must be greater than 1 for candidate sweeps.")
    counts = {
        "stage_names": len(stage_names),
        "families_paths": len(families_paths),
        "feature_compare_paths": len(feature_compare_paths),
        "trajectories_paths": len(trajectories_paths),
        "basis_paths": len(basis_paths),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"Candidate sweep stage argument counts do not match: {counts}")

    candidates_dir = output_dir / "candidate_subsets"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for stage_name, families_path, feature_compare_path, trajectories_path, basis_path in zip(
        stage_names,
        families_paths,
        feature_compare_paths,
        trajectories_paths,
        basis_paths,
        strict=True,
    ):
        families_payload = _load_json_dict(families_path, "feature families")
        feature_compare_rows = _feature_compare_rows_by_feature(feature_compare_path)
        families = _require_list(families_payload.get("families"), "feature_families.families")
        stage_candidates: list[dict[str, Any]] = []
        for index, raw_family in enumerate(families):
            family = _require_dict(raw_family, f"feature_families.families[{index}]")
            family_id = _require_int(family.get("family_id"), f"feature_families.families[{index}].family_id")
            member_feature_ids = _coerce_int_list(
                family.get("member_feature_ids"),
                f"feature_families.families[{index}].member_feature_ids",
            )
            family_size = _require_int(family.get("size"), f"feature_families.families[{index}].size")
            if family_size != len(member_feature_ids):
                raise ValueError(f"Family {family_id} size does not match member_feature_ids length.")
            if family_size < min_family_size:
                continue
            missing_compare = [feature_id for feature_id in member_feature_ids if feature_id not in feature_compare_rows]
            if missing_compare:
                raise KeyError(f"Family {family_id} has features missing from feature compare: {missing_compare}")
            member_rows = [{**feature_compare_rows[feature_id]} for feature_id in member_feature_ids]
            ranked_rows = _rank_family_member_rows(member_rows, ranking_name)
            selected_rows = ranked_rows[: min(subset_size, len(ranked_rows))]
            selected_feature_ids = [int(row["feature_id"]) for row in selected_rows]
            aggregate_proxy = _aggregate_compare_proxy(selected_rows)
            candidate_id = _make_sweep_candidate_id(
                stage_name=stage_name,
                family_id=family_id,
                ranking_name=ranking_name,
                feature_ids=selected_feature_ids,
            )
            subset_path = candidates_dir / f"{candidate_id}_subset_trajectory.json"
            subset_spec = {
                "source": "candidate_sweep",
                "stage_name": stage_name,
                "family_id": family_id,
                "ranking_name": ranking_name,
                "subset_size": len(selected_feature_ids),
                "selected_feature_rows": selected_rows,
                "aggregate_compare_proxy": aggregate_proxy,
            }
            _write_sweep_subset_trajectory(
                trajectories_path=trajectories_path,
                output_path=subset_path,
                feature_ids=selected_feature_ids,
                stage_name=stage_name,
                subset_spec=subset_spec,
            )
            stage_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "kind": "feature_family_subset",
                    "stage_name": stage_name,
                    "family_id": family_id,
                    "feature_ids": selected_feature_ids,
                    "feature_ids_sorted": sorted(selected_feature_ids),
                    "subset_size": len(selected_feature_ids),
                    "parameter_groups": [],
                    "top_components": {},
                    "selection": {
                        "ranking_name": ranking_name,
                        "family_size": family_size,
                        "representative_feature_id": family.get("representative_feature_id"),
                        "mean_pairwise_similarity": family.get("mean_pairwise_similarity"),
                        "member_feature_ids": member_feature_ids,
                        "selected_feature_rows": selected_rows,
                        "aggregate_compare_proxy": aggregate_proxy,
                    },
                    "source_artifacts": {
                        "families": str(families_path),
                        "feature_compare": str(feature_compare_path),
                        "feature_trajectories": str(trajectories_path),
                        "shared_feature_basis": str(basis_path),
                        "subset_trajectory": str(subset_path),
                        "feature_family_trace": None,
                        "subset_birth": None,
                        "family_update_link": None,
                    },
                }
            )
        stage_candidates.sort(
            key=lambda candidate: float(candidate["selection"]["aggregate_compare_proxy"]["useful_delta"]),
            reverse=True,
        )
        if top_k_families is not None:
            if top_k_families <= 0:
                raise ValueError("top_k_families must be positive when provided.")
            stage_candidates = stage_candidates[:top_k_families]
        candidates.extend(stage_candidates)
    if not candidates:
        raise ValueError("Candidate sweep produced no candidates.")
    registry_path = output_dir / "candidate_sweep_registry.json"
    write_json(
        registry_path,
        {
            "schema_version": CANDIDATE_REGISTRY_SCHEMA_VERSION,
            "candidate_count": len(candidates),
            "sweep_selection": {
                "ranking_name": ranking_name,
                "subset_size": subset_size,
                "min_family_size": min_family_size,
                "top_k_families": top_k_families,
            },
            "candidates": candidates,
        },
    )
    return registry_path, candidates


def _candidate_sweep_ranking_rows(
    *,
    registry_candidates: list[dict[str, Any]],
    gradient_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries = _require_dict(gradient_payload.get("candidate_summaries"), "gradient_link.candidate_summaries")
    candidate_by_id = {
        str(candidate["candidate_id"]): candidate
        for candidate in registry_candidates
    }
    ranking_rows: list[dict[str, Any]] = []
    for candidate_id, candidate in candidate_by_id.items():
        if candidate_id not in summaries:
            raise KeyError(f"Gradient-link output is missing candidate summary for {candidate_id}")
        summary = _require_dict(summaries[candidate_id], f"candidate_summaries.{candidate_id}")
        feature_score = _require_dict(summary.get("feature_score_summary"), f"candidate_summaries.{candidate_id}.feature_score_summary")
        selection = _require_dict(candidate.get("selection"), f"candidate {candidate_id}.selection")
        aggregate_proxy = _require_dict(selection.get("aggregate_compare_proxy"), f"candidate {candidate_id}.selection.aggregate_compare_proxy")
        ranking_rows.append(
            {
                "candidate_id": candidate_id,
                "stage_name": candidate["stage_name"],
                "family_id": candidate["family_id"],
                "feature_ids": candidate["feature_ids"],
                "subset_size": candidate["subset_size"],
                "family_size": selection["family_size"],
                "selection_useful_delta_proxy": aggregate_proxy["useful_delta"],
                "sum_useful_delta": summary["sum_useful_delta"],
                "sum_correctness_gap_delta": summary["sum_correctness_gap_delta"],
                "sum_heldout_gap_delta": summary["sum_heldout_gap_delta"],
                "sum_mean_activation_delta": summary["sum_mean_activation_delta"],
                "sum_loss_reduction_linearized": summary["sum_loss_reduction_linearized"],
                "sum_combined_reduction_linearized": summary["sum_combined_reduction_linearized"],
                "mean_update_negative_gradient_cosine": summary["mean_update_negative_gradient_cosine"],
                "feature_sum_observed_mean_activation_delta": feature_score["sum_observed_mean_activation_delta"],
                "feature_sum_score_linearized_delta": feature_score["sum_candidate_union_score_linearized_delta"],
                "feature_mean_update_score_gradient_cosine": feature_score[
                    "mean_candidate_union_update_score_gradient_cosine"
                ],
            }
        )
    ranking_rows.sort(key=lambda row: float(row["sum_useful_delta"]), reverse=True)
    return ranking_rows


def _render_candidate_sweep_bar_plot(
    *,
    rows: list[dict[str, Any]],
    metric_name: str,
    title: str,
    output_path: Path,
    top_k: int = 16,
) -> Path:
    if not rows:
        raise ValueError("Cannot render candidate sweep bar plot without rows.")
    ranked = sorted(rows, key=lambda row: float(row[metric_name]), reverse=True)[:top_k]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{row['stage_name']} F{row['family_id']}" for row in ranked]
    values = [float(row[metric_name]) for row in ranked]
    ax.bar(labels, values)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title(title)
    ax.set_ylabel(metric_name)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_candidate_sweep_scatter_plot(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not rows:
        raise ValueError("Cannot render candidate sweep scatter plot without rows.")
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 7))
    stages = sorted({str(row["stage_name"]) for row in rows})
    for stage_name in stages:
        stage_rows = [row for row in rows if str(row["stage_name"]) == stage_name]
        ax.scatter(
            [float(row["sum_useful_delta"]) for row in stage_rows],
            [float(row["feature_sum_score_linearized_delta"]) for row in stage_rows],
            label=stage_name,
            alpha=0.75,
        )
        for row in stage_rows:
            ax.annotate(
                f"F{row['family_id']}",
                (
                    float(row["sum_useful_delta"]),
                    float(row["feature_sum_score_linearized_delta"]),
                ),
                fontsize=8,
                alpha=0.8,
            )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Candidate usefulness vs feature-score update drive")
    ax.set_xlabel("Cumulative useful delta")
    ax.set_ylabel("Cumulative score dot update")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_candidate_sweep_plots(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    return {
        "top_useful_delta": _render_candidate_sweep_bar_plot(
            rows=rows,
            metric_name="sum_useful_delta",
            title="Top candidates by cumulative useful delta",
            output_path=output_dir / "candidate_sweep_top_useful_delta.svg",
        ),
        "top_heldout_gap_delta": _render_candidate_sweep_bar_plot(
            rows=rows,
            metric_name="sum_heldout_gap_delta",
            title="Top candidates by cumulative heldout-gap delta",
            output_path=output_dir / "candidate_sweep_top_heldout_gap_delta.svg",
        ),
        "top_feature_score_delta": _render_candidate_sweep_bar_plot(
            rows=rows,
            metric_name="feature_sum_score_linearized_delta",
            title="Top candidates by feature-score update drive",
            output_path=output_dir / "candidate_sweep_top_feature_score_delta.svg",
        ),
        "useful_vs_feature_score": _render_candidate_sweep_scatter_plot(
            rows=rows,
            output_path=output_dir / "candidate_sweep_useful_vs_feature_score.svg",
        ),
    }


def run_candidate_sweep(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    stage_names: list[str],
    families_paths: list[Path],
    feature_compare_paths: list[Path],
    trajectories_paths: list[Path],
    basis_paths: list[Path],
    sweep_metrics_path: Path | None = None,
    device_name: str = "cpu",
    ranking_name: str = "by_useful_delta",
    subset_size: int = 3,
    min_family_size: int = 2,
    top_k_families: int | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
) -> tuple[Path, dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_path, registry_candidates = _build_sweep_registry(
        stage_names=stage_names,
        families_paths=families_paths,
        feature_compare_paths=feature_compare_paths,
        trajectories_paths=trajectories_paths,
        basis_paths=basis_paths,
        output_dir=output_dir,
        ranking_name=ranking_name,
        subset_size=subset_size,
        min_family_size=min_family_size,
        top_k_families=top_k_families,
    )
    gradient_link_path = output_dir / "candidate_sweep_gradient_link.json"
    run_circuit_gradient_link(
        config_path=config_path,
        probe_set_path=probe_set_path,
        registry_path=registry_path,
        checkpoint_dir=checkpoint_dir,
        output_path=gradient_link_path,
        device_name=device_name,
        sweep_metrics_path=sweep_metrics_path,
        start_step=start_step,
        end_step=end_step,
    )
    gradient_payload = _load_json_dict(gradient_link_path, "candidate sweep gradient-link")
    ranking_rows = _candidate_sweep_ranking_rows(
        registry_candidates=registry_candidates,
        gradient_payload=gradient_payload,
    )
    plot_paths = _render_candidate_sweep_plots(rows=ranking_rows, output_dir=output_dir)
    summary_path = output_dir / "candidate_sweep_summary.json"
    write_json(
        summary_path,
        {
            "config_path": str(config_path),
            "probe_set_path": str(probe_set_path),
            "checkpoint_dir": str(checkpoint_dir),
            "sweep_metrics_path": None if sweep_metrics_path is None else str(sweep_metrics_path),
            "registry_path": str(registry_path),
            "gradient_link_path": str(gradient_link_path),
            "candidate_count": len(ranking_rows),
            "selection": {
                "stage_names": stage_names,
                "ranking_name": ranking_name,
                "subset_size": subset_size,
                "min_family_size": min_family_size,
                "top_k_families": top_k_families,
                "start_step": start_step,
                "end_step": end_step,
            },
            "plots": {name: str(path) for name, path in plot_paths.items()},
            "rankings": {
                "by_sum_useful_delta": sorted(
                    ranking_rows,
                    key=lambda row: float(row["sum_useful_delta"]),
                    reverse=True,
                ),
                "by_sum_heldout_gap_delta": sorted(
                    ranking_rows,
                    key=lambda row: float(row["sum_heldout_gap_delta"]),
                    reverse=True,
                ),
                "by_feature_sum_score_linearized_delta": sorted(
                    ranking_rows,
                    key=lambda row: float(row["feature_sum_score_linearized_delta"]),
                    reverse=True,
                ),
                "by_feature_mean_update_score_gradient_cosine": sorted(
                    ranking_rows,
                    key=lambda row: float(row["feature_mean_update_score_gradient_cosine"]),
                    reverse=True,
                ),
            },
            "ranking_rows": ranking_rows,
        },
    )
    return summary_path, plot_paths


def run_circuit_gradient_link(
    *,
    config_path: Path,
    probe_set_path: Path,
    registry_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    device_name: str = "cpu",
    sweep_metrics_path: Path | None = None,
    start_step: int | None = None,
    end_step: int | None = None,
) -> Path:
    spec = TrainSpec.from_path(config_path)
    device = require_device(device_name)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    batches = _load_probe_batches(spec=spec, probe_set_path=probe_set_path, vocab=vocab, device=device)
    registry = _load_candidate_registry(registry_path)
    candidates = [_require_dict(candidate, "candidate registry candidate") for candidate in registry["candidates"]]
    checkpoint_paths_by_step = _resolve_checkpoint_paths_by_step(checkpoint_dir)
    sweep_rows_by_step = _load_sweep_rows_by_step(sweep_metrics_path)

    state_cache: dict[int, dict[str, Any]] = {}
    loss_cache: dict[int, dict[str, Any]] = {}
    gradient_cache: dict[int, dict[str, Any]] = {}
    basis_cache: dict[str, dict[str, Any]] = {}
    feature_score_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def load_state(step: int) -> dict[str, Any]:
        cached = state_cache.get(step)
        if cached is not None:
            return cached
        if step not in checkpoint_paths_by_step:
            raise FileNotFoundError(f"Checkpoint for step {step} not found in {checkpoint_dir}.")
        payload = _load_checkpoint_state_for_analysis(checkpoint_paths_by_step[step])
        if int(payload["step"]) != step:
            raise ValueError(
                f"Checkpoint filename step {step} does not match payload step {payload['step']}: "
                f"{checkpoint_paths_by_step[step]}"
            )
        state_cache[step] = payload
        return payload

    def load_model_at_step(step: int) -> torch.nn.Module:
        if step not in checkpoint_paths_by_step:
            raise FileNotFoundError(f"Checkpoint for step {step} not found in {checkpoint_dir}.")
        checkpoint = load_checkpoint(checkpoint_paths_by_step[step], device)
        model = build_model(spec.model, len(vocab.tokens), device)
        load_model_state(model, checkpoint["model_state"])
        return model

    def loss_at_step(step: int) -> dict[str, Any]:
        cached = loss_cache.get(step)
        if cached is not None:
            return cached
        model = load_model_at_step(step)
        payload = _compute_probe_loss(model=model, batches=batches, pad_token_id=vocab.pad_token_id)
        loss_cache[step] = payload
        return payload

    def gradients_at_step(step: int) -> dict[str, Any]:
        cached = gradient_cache.get(step)
        if cached is not None:
            return cached
        model = load_model_at_step(step)
        payload = _compute_probe_loss_and_gradients(model=model, batches=batches, pad_token_id=vocab.pad_token_id)
        gradient_cache[step] = payload
        loss_cache[step] = {
            "loss": payload["loss"],
            "num_tokens": payload["num_tokens"],
            "num_batches": payload["num_batches"],
        }
        return payload

    def basis_for_candidate(candidate: dict[str, Any]) -> dict[str, Any] | None:
        artifacts = _require_dict(candidate.get("source_artifacts"), "candidate.source_artifacts")
        raw_basis_path = artifacts.get("shared_feature_basis")
        if raw_basis_path is None:
            return None
        basis_path = Path(_require_non_empty_str(raw_basis_path, "candidate.source_artifacts.shared_feature_basis"))
        cache_key = str(basis_path)
        cached = basis_cache.get(cache_key)
        if cached is not None:
            return cached
        basis = _load_shared_basis(basis_path, device)
        basis_cache[cache_key] = basis
        return basis

    def feature_score_at_step(candidate: dict[str, Any], step: int) -> dict[str, Any] | None:
        candidate_id = _sanitize_candidate_id(_require_non_empty_str(candidate["candidate_id"], "candidate_id"))
        basis = basis_for_candidate(candidate)
        if basis is None:
            return None
        cache_key = (candidate_id, step)
        cached = feature_score_cache.get(cache_key)
        if cached is not None:
            return cached
        model = load_model_at_step(step)
        payload = _compute_feature_score_and_gradients(
            model=model,
            batches=batches,
            basis=basis,
            stage_name=_require_non_empty_str(candidate.get("stage_name"), "candidate.stage_name"),
            feature_ids=_coerce_int_list(candidate.get("feature_ids"), "candidate.feature_ids"),
        )
        feature_score_cache[cache_key] = payload
        return payload

    interval_rows: list[dict[str, Any]] = []
    rows_by_candidate_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidate_id = _sanitize_candidate_id(_require_non_empty_str(candidate["candidate_id"], "candidate_id"))
        candidate_rows_by_step = _candidate_rows_by_step(candidate)
        ordered_steps = sorted(candidate_rows_by_step)
        for previous_step, current_step in zip(ordered_steps[:-1], ordered_steps[1:], strict=True):
            if start_step is not None and current_step < start_step:
                continue
            if end_step is not None and current_step > end_step:
                continue
            previous_payload = load_state(previous_step)
            current_payload = load_state(current_step)
            gradient_payload = gradients_at_step(previous_step)
            current_loss = loss_at_step(current_step)
            gradients = _require_dict(gradient_payload["gradients"], "gradient_payload.gradients")
            previous_state = previous_payload["model_state"]
            current_state = current_payload["model_state"]
            _validate_state_and_gradient_keys(
                previous_state=previous_state,
                current_state=current_state,
                gradients=gradients,
            )
            n_heads = _extract_n_heads(previous_payload["checkpoint"])
            component_masks = _build_component_masks(
                state=previous_state,
                candidate=candidate,
                n_heads=n_heads,
            )
            group_stats = {
                "global": _projection_stats(
                    previous_state=previous_state,
                    current_state=current_state,
                    gradients=gradients,
                    masks=_mask_all_parameters(previous_state),
                    weight_decay=spec.optimization.weight_decay,
                )
            }
            for group_name, masks in component_masks.items():
                group_stats[group_name] = _projection_stats(
                    previous_state=previous_state,
                    current_state=current_state,
                    gradients=gradients,
                    masks=masks,
                    weight_decay=spec.optimization.weight_decay,
                )
            if "candidate_union" not in group_stats:
                group_stats["candidate_union"] = {
                    **group_stats["global"],
                    "projection_scope": "global_no_component_groups",
                }
            _add_global_shares(group_stats)

            previous_subset_row = candidate_rows_by_step[previous_step]
            current_subset_row = candidate_rows_by_step[current_step]
            candidate_state_deltas = _candidate_state_deltas(previous_subset_row, current_subset_row)
            feature_score_payload = feature_score_at_step(candidate, previous_step)
            feature_score_projection: dict[str, Any]
            if feature_score_payload is None:
                feature_score_projection = {"status": "not_requested"}
            else:
                score_gradients = _require_dict(feature_score_payload["gradients"], "feature_score_payload.gradients")
                _validate_state_and_gradient_keys(
                    previous_state=previous_state,
                    current_state=current_state,
                    gradients=score_gradients,
                )
                score_group_stats = {
                    "global": _score_projection_stats(
                        previous_state=previous_state,
                        current_state=current_state,
                        score_gradients=score_gradients,
                        masks=_mask_all_parameters(previous_state),
                    )
                }
                for group_name, masks in component_masks.items():
                    score_group_stats[group_name] = _score_projection_stats(
                        previous_state=previous_state,
                        current_state=current_state,
                        score_gradients=score_gradients,
                        masks=masks,
                    )
                if "candidate_union" not in score_group_stats:
                    score_group_stats["candidate_union"] = {
                        **score_group_stats["global"],
                        "projection_scope": "global_no_component_groups",
                    }
                _add_score_global_shares(score_group_stats)
                feature_score_projection = {
                    "status": "computed",
                    "previous_score": feature_score_payload["score"],
                    "observed_mean_activation_delta": candidate_state_deltas["mean_activation_delta"],
                    "num_answers": feature_score_payload["num_answers"],
                    "num_feature_values": feature_score_payload["num_feature_values"],
                    "zero_gradient_parameter_names": feature_score_payload["zero_gradient_parameter_names"],
                    "parameter_projections": score_group_stats,
                }
            row = {
                "candidate_id": candidate_id,
                "family_id": candidate.get("family_id"),
                "stage_name": candidate.get("stage_name"),
                "feature_ids": candidate.get("feature_ids"),
                "previous_step": previous_step,
                "step": current_step,
                "previous_checkpoint_path": str(checkpoint_paths_by_step[previous_step]),
                "checkpoint_path": str(checkpoint_paths_by_step[current_step]),
                "learning_rate_previous_step": _compute_learning_rate(spec.optimization, previous_step),
                "learning_rate_current_step": _compute_learning_rate(spec.optimization, current_step),
                "probe_loss": {
                    "previous": gradient_payload["loss"],
                    "current": current_loss["loss"],
                    "delta": float(current_loss["loss"]) - float(gradient_payload["loss"]),
                    "num_tokens": gradient_payload["num_tokens"],
                    "num_batches": gradient_payload["num_batches"],
                },
                "candidate_state_deltas": candidate_state_deltas,
                "sweep_deltas": _sweep_deltas_for_step(sweep_rows_by_step, current_step),
                "parameter_projections": group_stats,
                "feature_score_projection": feature_score_projection,
            }
            interval_rows.append(row)
            rows_by_candidate_id[candidate_id].append(row)

    if not interval_rows:
        raise ValueError("No candidate intervals were selected for gradient-link analysis.")

    candidate_summaries = {
        candidate_id: _candidate_summary(rows)
        for candidate_id, rows in sorted(rows_by_candidate_id.items())
    }
    plot_paths = _render_gradient_link_plots(interval_rows=interval_rows, output_path=output_path)
    payload = {
        "schema_version": GRADIENT_LINK_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "registry_path": str(registry_path),
        "checkpoint_dir": str(checkpoint_dir),
        "sweep_metrics_path": None if sweep_metrics_path is None else str(sweep_metrics_path),
        "device": device_name,
        "gradient_source": {
            "kind": "probe_set_next_token_loss",
            "mode": "eval",
            "num_probe_batches": len(batches),
        },
        "interval_filter": {
            "start_step": start_step,
            "end_step": end_step,
        },
        "candidate_count": len(candidates),
        "interval_count": len(interval_rows),
        "candidate_summaries": candidate_summaries,
        "plots": {name: str(path) for name, path in plot_paths.items()},
        "interval_rows": interval_rows,
    }
    write_json(output_path, payload)
    return output_path


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric.")
    return float(value)


def _load_gradient_link_payload(gradient_link_path: Path) -> dict[str, Any]:
    payload = _load_json_dict(gradient_link_path, "gradient link")
    schema_version = payload.get("schema_version")
    if schema_version != GRADIENT_LINK_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported gradient-link schema_version {schema_version}; expected {GRADIENT_LINK_SCHEMA_VERSION}."
        )
    interval_rows = _require_list(payload.get("interval_rows"), "gradient_link.interval_rows")
    if not interval_rows:
        raise ValueError(f"Gradient-link output contains no interval rows: {gradient_link_path}")
    _require_dict(payload.get("candidate_summaries"), "gradient_link.candidate_summaries")
    return payload


def _candidate_lookup_from_registry(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = _require_list(registry.get("candidates"), "candidate_registry.candidates")
    lookup: dict[str, dict[str, Any]] = {}
    for index, raw_candidate in enumerate(candidates):
        candidate = _require_dict(raw_candidate, f"candidate_registry.candidates[{index}]")
        candidate_id = _sanitize_candidate_id(
            _require_non_empty_str(candidate.get("candidate_id"), f"candidate_registry.candidates[{index}].candidate_id")
        )
        if candidate_id in lookup:
            raise ValueError(f"Duplicate candidate_id in registry: {candidate_id}")
        lookup[candidate_id] = candidate
    if not lookup:
        raise ValueError("Candidate registry has no candidates.")
    return lookup


def _candidate_summary_metric(summary: dict[str, Any], metric_name: str) -> float:
    direct_metrics = {
        "sum_useful_delta",
        "sum_correctness_gap_delta",
        "sum_heldout_gap_delta",
        "sum_mean_activation_delta",
        "sum_active_fraction_delta",
        "sum_loss_reduction_linearized",
        "sum_combined_reduction_linearized",
        "mean_update_negative_gradient_cosine",
    }
    if metric_name in direct_metrics:
        if metric_name not in summary:
            raise KeyError(f"Candidate summary is missing metric {metric_name}.")
        return _require_number(summary[metric_name], f"candidate_summary.{metric_name}")

    feature_metric_map = {
        "feature_sum_observed_mean_activation_delta": "sum_observed_mean_activation_delta",
        "feature_sum_score_linearized_delta": "sum_candidate_union_score_linearized_delta",
        "feature_mean_update_score_gradient_cosine": "mean_candidate_union_update_score_gradient_cosine",
    }
    if metric_name in feature_metric_map:
        feature_summary = _require_dict(
            summary.get("feature_score_summary"),
            f"candidate_summary.feature_score_summary for metric {metric_name}",
        )
        nested_metric = feature_metric_map[metric_name]
        if nested_metric not in feature_summary:
            raise KeyError(f"Candidate feature summary is missing metric {nested_metric}.")
        return _require_number(feature_summary[nested_metric], f"candidate_summary.feature_score_summary.{nested_metric}")

    raise ValueError(f"Unsupported mechanism report ranking metric: {metric_name}")


def _select_mechanism_candidate_ids(
    *,
    summaries: dict[str, Any],
    registry_lookup: dict[str, dict[str, Any]],
    candidate_ids: list[str] | None,
    top_k: int,
    ranking_metric: str,
) -> list[str]:
    summary_ids = set(summaries)
    registry_ids = set(registry_lookup)
    missing_registry = sorted(summary_ids - registry_ids)
    if missing_registry:
        raise KeyError(f"Gradient-link summaries reference candidates missing from registry: {missing_registry}")
    if candidate_ids is not None:
        selected: list[str] = []
        seen: set[str] = set()
        for candidate_id_raw in candidate_ids:
            candidate_id = _sanitize_candidate_id(candidate_id_raw)
            if candidate_id in seen:
                raise ValueError(f"Duplicate selected candidate_id: {candidate_id}")
            if candidate_id not in summaries:
                raise KeyError(f"Selected candidate_id is missing from gradient-link summaries: {candidate_id}")
            if candidate_id not in registry_lookup:
                raise KeyError(f"Selected candidate_id is missing from registry: {candidate_id}")
            selected.append(candidate_id)
            seen.add(candidate_id)
        if not selected:
            raise ValueError("candidate_ids must not be empty when provided.")
        return selected

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    ranked = sorted(
        summaries,
        key=lambda candidate_id: _candidate_summary_metric(
            _require_dict(summaries[candidate_id], f"candidate_summaries.{candidate_id}"),
            ranking_metric,
        ),
        reverse=True,
    )
    return ranked[:top_k]


def _feature_score_union_projection(row: dict[str, Any]) -> dict[str, Any] | None:
    projection = _require_dict(row.get("feature_score_projection"), "interval_row.feature_score_projection")
    status = _require_non_empty_str(projection.get("status"), "interval_row.feature_score_projection.status")
    if status == "computed":
        projections = _require_dict(
            projection.get("parameter_projections"),
            "interval_row.feature_score_projection.parameter_projections",
        )
        return _require_dict(projections.get("candidate_union"), "feature_score_projection.parameter_projections.candidate_union")
    if status in {"not_requested"}:
        return None
    raise ValueError(f"Unsupported feature_score_projection status: {status}")


def _feature_score_delta(row: dict[str, Any]) -> float | None:
    projection = _feature_score_union_projection(row)
    if projection is None:
        return None
    return _require_number(projection.get("score_linearized_delta"), "feature_score_projection.score_linearized_delta")


def _candidate_interval_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    deltas = _require_dict(row.get("candidate_state_deltas"), "interval_row.candidate_state_deltas")
    parameter_projections = _require_dict(row.get("parameter_projections"), "interval_row.parameter_projections")
    union_projection = _require_dict(parameter_projections.get("candidate_union"), "parameter_projections.candidate_union")
    feature_projection = _require_dict(row.get("feature_score_projection"), "interval_row.feature_score_projection")
    observed_feature_delta: float | None = None
    if feature_projection.get("status") == "computed":
        observed_feature_delta = _require_number(
            feature_projection.get("observed_mean_activation_delta"),
            "feature_score_projection.observed_mean_activation_delta",
        )
    return {
        "useful_delta": _require_number(deltas.get("useful_delta"), "candidate_state_deltas.useful_delta"),
        "correctness_gap_delta": _require_number(
            deltas.get("correctness_gap_delta"),
            "candidate_state_deltas.correctness_gap_delta",
        ),
        "heldout_gap_delta": _require_number(deltas.get("heldout_gap_delta"), "candidate_state_deltas.heldout_gap_delta"),
        "mean_activation_delta": _require_number(
            deltas.get("mean_activation_delta"),
            "candidate_state_deltas.mean_activation_delta",
        ),
        "active_fraction_delta": _require_number(
            deltas.get("active_fraction_delta"),
            "candidate_state_deltas.active_fraction_delta",
        ),
        "loss_reduction_linearized": _require_number(
            union_projection.get("loss_reduction_linearized"),
            "parameter_projections.candidate_union.loss_reduction_linearized",
        ),
        "combined_reduction_linearized": _require_number(
            union_projection.get("combined_reduction_linearized"),
            "parameter_projections.candidate_union.combined_reduction_linearized",
        ),
        "update_negative_gradient_cosine": None
        if union_projection.get("update_negative_gradient_cosine") is None
        else _require_number(
            union_projection.get("update_negative_gradient_cosine"),
            "parameter_projections.candidate_union.update_negative_gradient_cosine",
        ),
        "feature_score_linearized_delta": _feature_score_delta(row),
        "observed_feature_mean_activation_delta": observed_feature_delta,
    }


def _sign_name(value: float | None, epsilon: float) -> str:
    if value is None:
        return "unavailable"
    if value > epsilon:
        return "positive"
    if value < -epsilon:
        return "negative"
    return "flat"


def _phase_label(row: dict[str, Any], epsilon: float) -> str:
    metrics = _candidate_interval_metrics(row)
    useful_sign = _sign_name(metrics["useful_delta"], epsilon)
    heldout_sign = _sign_name(metrics["heldout_gap_delta"], epsilon)
    score_sign = _sign_name(metrics["feature_score_linearized_delta"], epsilon)

    if score_sign == "unavailable":
        if useful_sign == "positive" and heldout_sign == "positive":
            return "observed_generalizing_gain_without_score_gradient"
        if useful_sign == "positive":
            return "observed_probe_gain_without_score_gradient"
        if useful_sign == "negative":
            return "observed_candidate_decline_without_score_gradient"
        return "observed_flat_without_score_gradient"
    if score_sign == "positive" and useful_sign == "positive" and heldout_sign == "positive":
        return "sgd_supported_generalizing_gain"
    if score_sign == "positive" and useful_sign == "positive":
        return "sgd_supported_probe_gain"
    if score_sign == "positive" and useful_sign == "negative":
        return "feature_amplified_without_useful_gain"
    if score_sign == "negative" and useful_sign == "negative":
        return "sgd_suppression"
    if score_sign == "negative" and useful_sign == "positive":
        return "useful_gain_against_score_gradient"
    if score_sign == "flat" and useful_sign == "positive":
        return "useful_gain_with_flat_score_gradient"
    if score_sign == "flat" and useful_sign == "negative":
        return "decline_with_flat_score_gradient"
    return "flat_or_reorganizing"


def _sum_interval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty.")
    metric_rows = [_candidate_interval_metrics(row) for row in rows]
    feature_rows = [metrics for metrics in metric_rows if metrics["feature_score_linearized_delta"] is not None]
    cosine_values = [
        float(metrics["update_negative_gradient_cosine"])
        for metrics in metric_rows
        if metrics["update_negative_gradient_cosine"] is not None
    ]
    payload: dict[str, Any] = {
        "interval_count": len(rows),
        "sum_useful_delta": sum(float(metrics["useful_delta"]) for metrics in metric_rows),
        "sum_correctness_gap_delta": sum(float(metrics["correctness_gap_delta"]) for metrics in metric_rows),
        "sum_heldout_gap_delta": sum(float(metrics["heldout_gap_delta"]) for metrics in metric_rows),
        "sum_mean_activation_delta": sum(float(metrics["mean_activation_delta"]) for metrics in metric_rows),
        "sum_active_fraction_delta": sum(float(metrics["active_fraction_delta"]) for metrics in metric_rows),
        "sum_loss_reduction_linearized": sum(
            float(metrics["loss_reduction_linearized"]) for metrics in metric_rows
        ),
        "sum_combined_reduction_linearized": sum(
            float(metrics["combined_reduction_linearized"]) for metrics in metric_rows
        ),
        "mean_update_negative_gradient_cosine": None
        if not cosine_values
        else sum(cosine_values) / len(cosine_values),
        "feature_score_interval_count": len(feature_rows),
        "feature_sum_score_linearized_delta": None
        if not feature_rows
        else sum(float(metrics["feature_score_linearized_delta"]) for metrics in feature_rows),
        "feature_sum_observed_mean_activation_delta": None
        if not feature_rows
        else sum(float(metrics["observed_feature_mean_activation_delta"]) for metrics in feature_rows),
    }
    return payload


def _phase_windows(rows: list[dict[str, Any]], epsilon: float) -> list[dict[str, Any]]:
    if epsilon < 0.0:
        raise ValueError("phase_epsilon must be non-negative.")
    sorted_rows = sorted(rows, key=lambda row: int(row["step"]))
    windows: list[dict[str, Any]] = []
    current_label: str | None = None
    current_rows: list[dict[str, Any]] = []
    for row in sorted_rows:
        label = _phase_label(row, epsilon)
        if current_label is None or label == current_label:
            current_label = label
            current_rows.append(row)
            continue
        windows.append(_build_phase_window(current_label, current_rows))
        current_label = label
        current_rows = [row]
    if current_label is not None:
        windows.append(_build_phase_window(current_label, current_rows))
    return windows


def _build_phase_window(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot build a phase window with no rows.")
    first = rows[0]
    last = rows[-1]
    return {
        "label": label,
        "start_step": _require_int(first.get("previous_step"), "interval_row.previous_step"),
        "end_step": _require_int(last.get("step"), "interval_row.step"),
        **_sum_interval_metrics(rows),
    }


def _rank_interval_extremes(
    *,
    rows: list[dict[str, Any]],
    metric_name: str,
    top_k: int,
    phase_epsilon: float,
) -> dict[str, list[dict[str, Any]]]:
    if top_k <= 0:
        raise ValueError("top_interval_k must be positive.")

    def metric_value(row: dict[str, Any]) -> float | None:
        metrics = _candidate_interval_metrics(row)
        if metric_name not in metrics:
            raise ValueError(f"Unsupported interval metric: {metric_name}")
        value = metrics[metric_name]
        return None if value is None else float(value)

    valued_rows = [
        {
            "previous_step": _require_int(row.get("previous_step"), "interval_row.previous_step"),
            "step": _require_int(row.get("step"), "interval_row.step"),
            "value": value,
            "phase_label": _phase_label(row, phase_epsilon),
        }
        for row in rows
        for value in [metric_value(row)]
        if value is not None
    ]
    ranked_positive = sorted(valued_rows, key=lambda item: float(item["value"]), reverse=True)[:top_k]
    ranked_negative = sorted(valued_rows, key=lambda item: float(item["value"]))[:top_k]
    return {
        "top_positive": ranked_positive,
        "top_negative": ranked_negative,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _aggregate_component_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty.")
    excluded = {"global", "candidate_union"}
    component_names = sorted(
        {
            group_name
            for row in rows
            for group_name in _require_dict(row.get("parameter_projections"), "interval_row.parameter_projections")
            if group_name not in excluded
        }
    )
    if not component_names:
        projection = _require_dict(rows[0].get("parameter_projections"), "interval_row.parameter_projections")
        union = _require_dict(projection.get("candidate_union"), "parameter_projections.candidate_union")
        return {
            "status": "no_component_groups",
            "reason": "candidate registry did not define component parameter groups",
            "projection_scope": union.get("projection_scope"),
            "groups": [],
        }

    group_rows: list[dict[str, Any]] = []
    for group_name in component_names:
        projections: list[dict[str, Any]] = []
        score_projections: list[dict[str, Any]] = []
        for row in rows:
            parameter_projections = _require_dict(row.get("parameter_projections"), "interval_row.parameter_projections")
            if group_name not in parameter_projections:
                raise KeyError(f"Interval row is missing component projection {group_name}.")
            projections.append(_require_dict(parameter_projections[group_name], f"parameter_projections.{group_name}"))
            feature_projection = _require_dict(row.get("feature_score_projection"), "interval_row.feature_score_projection")
            status = feature_projection.get("status")
            if status == "computed":
                feature_parameter_projections = _require_dict(
                    feature_projection.get("parameter_projections"),
                    "feature_score_projection.parameter_projections",
                )
                if group_name not in feature_parameter_projections:
                    raise KeyError(f"Feature-score projection is missing component {group_name}.")
                score_projections.append(
                    _require_dict(feature_parameter_projections[group_name], f"feature_score_projection.{group_name}")
                )
            elif status != "not_requested":
                raise ValueError(f"Unsupported feature_score_projection status: {status}")

        num_parameter_values = {
            _require_int(projection.get("num_parameters"), f"parameter_projections.{group_name}.num_parameters")
            for projection in projections
        }
        if len(num_parameter_values) != 1:
            raise ValueError(f"Component {group_name} has inconsistent num_parameters values: {sorted(num_parameter_values)}")
        cosine_values = [
            _require_number(projection.get("update_negative_gradient_cosine"), f"{group_name}.update_negative_gradient_cosine")
            for projection in projections
            if projection.get("update_negative_gradient_cosine") is not None
        ]
        update_share_values = [
            _require_number(projection.get("update_l2_share_global"), f"{group_name}.update_l2_share_global")
            for projection in projections
            if projection.get("update_l2_share_global") is not None
        ]
        score_cosine_values = [
            _require_number(projection.get("update_score_gradient_cosine"), f"{group_name}.update_score_gradient_cosine")
            for projection in score_projections
            if projection.get("update_score_gradient_cosine") is not None
        ]
        score_share_values = [
            _require_number(projection.get("score_linearized_delta_share_global"), f"{group_name}.score_linearized_delta_share_global")
            for projection in score_projections
            if projection.get("score_linearized_delta_share_global") is not None
        ]
        group_rows.append(
            {
                "group_name": group_name,
                "num_parameters": next(iter(num_parameter_values)),
                "interval_count": len(projections),
                "sum_update_l2_norm": sum(
                    _require_number(projection.get("update_l2_norm"), f"{group_name}.update_l2_norm")
                    for projection in projections
                ),
                "sum_gradient_l2_norm": sum(
                    _require_number(projection.get("gradient_l2_norm"), f"{group_name}.gradient_l2_norm")
                    for projection in projections
                ),
                "sum_loss_reduction_linearized": sum(
                    _require_number(
                        projection.get("loss_reduction_linearized"),
                        f"{group_name}.loss_reduction_linearized",
                    )
                    for projection in projections
                ),
                "sum_combined_reduction_linearized": sum(
                    _require_number(
                        projection.get("combined_reduction_linearized"),
                        f"{group_name}.combined_reduction_linearized",
                    )
                    for projection in projections
                ),
                "mean_update_negative_gradient_cosine": _mean(cosine_values),
                "mean_update_l2_share_global": _mean(update_share_values),
                "score_interval_count": len(score_projections),
                "sum_score_linearized_delta": None
                if not score_projections
                else sum(
                    _require_number(
                        projection.get("score_linearized_delta"),
                        f"{group_name}.score_linearized_delta",
                    )
                    for projection in score_projections
                ),
                "mean_update_score_gradient_cosine": _mean(score_cosine_values),
                "mean_score_linearized_delta_share_global": _mean(score_share_values),
            }
        )
    group_rows.sort(key=lambda row: abs(float(row["sum_loss_reduction_linearized"])), reverse=True)
    return {
        "status": "computed",
        "groups": group_rows,
    }


def _candidate_status_label(candidate_totals: dict[str, Any], epsilon: float) -> str:
    useful = _require_number(candidate_totals.get("sum_useful_delta"), "candidate_totals.sum_useful_delta")
    heldout = _require_number(candidate_totals.get("sum_heldout_gap_delta"), "candidate_totals.sum_heldout_gap_delta")
    score_value = candidate_totals.get("feature_sum_score_linearized_delta")
    if score_value is None:
        if useful > epsilon and heldout > epsilon:
            return "observed_generalizing_candidate_without_score_gradient"
        if useful > epsilon:
            return "observed_probe_candidate_without_score_gradient"
        return "not_sgd_supported_by_available_artifacts"
    score = _require_number(score_value, "candidate_totals.feature_sum_score_linearized_delta")
    if useful > epsilon and heldout > epsilon and score > epsilon:
        return "sgd_supported_generalizing_candidate"
    if useful > epsilon and score > epsilon:
        return "sgd_supported_probe_candidate"
    if useful > epsilon and score < -epsilon:
        return "usefulness_not_explained_by_feature_score_gradient"
    if useful <= epsilon and score > epsilon:
        return "sgd_amplifies_feature_without_measured_usefulness"
    return "not_currently_supported"


def _build_candidate_mechanism_entry(
    *,
    candidate_id: str,
    candidate: dict[str, Any],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    phase_epsilon: float,
    top_interval_k: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"No interval rows found for selected candidate {candidate_id}.")
    rows_sorted = sorted(rows, key=lambda row: int(row["step"]))
    totals = _sum_interval_metrics(rows_sorted)
    component_summary = _aggregate_component_summary(rows_sorted)
    unsupported: list[str] = []
    if totals["feature_score_interval_count"] != totals["interval_count"]:
        unsupported.append("feature_score_gradient_link_for_all_intervals")
    if component_summary["status"] != "computed":
        unsupported.append("component_level_parameter_responsibility")
    if any(row.get("sweep_deltas") is None for row in rows_sorted):
        unsupported.append("global_checkpoint_metric_delta_alignment")
    artifacts = _require_dict(candidate.get("source_artifacts"), f"candidate {candidate_id}.source_artifacts")
    if artifacts.get("feature_family_trace") is None:
        unsupported.append("hand_traced_head_mlp_neuron_lineage")
    return {
        "candidate_id": candidate_id,
        "status_label": _candidate_status_label(totals, phase_epsilon),
        "stage_name": _require_non_empty_str(candidate.get("stage_name"), f"candidate {candidate_id}.stage_name"),
        "family_id": candidate.get("family_id"),
        "feature_ids": _coerce_int_list(candidate.get("feature_ids"), f"candidate {candidate_id}.feature_ids"),
        "subset_size": _require_int(candidate.get("subset_size"), f"candidate {candidate_id}.subset_size"),
        "checkpoint_window": {
            "start_step": _require_int(rows_sorted[0].get("previous_step"), "interval_row.previous_step"),
            "end_step": _require_int(rows_sorted[-1].get("step"), "interval_row.step"),
            "interval_count": len(rows_sorted),
        },
        "registry_summary": candidate.get("summary"),
        "selection": candidate.get("selection"),
        "top_components": candidate.get("top_components"),
        "source_artifacts": artifacts,
        "gradient_summary": summary,
        "totals": totals,
        "component_summary": component_summary,
        "phase_windows": _phase_windows(rows_sorted, phase_epsilon),
        "dominant_intervals": {
            "useful_delta": _rank_interval_extremes(
                rows=rows_sorted,
                metric_name="useful_delta",
                top_k=top_interval_k,
                phase_epsilon=phase_epsilon,
            ),
            "heldout_gap_delta": _rank_interval_extremes(
                rows=rows_sorted,
                metric_name="heldout_gap_delta",
                top_k=top_interval_k,
                phase_epsilon=phase_epsilon,
            ),
            "feature_score_linearized_delta": _rank_interval_extremes(
                rows=rows_sorted,
                metric_name="feature_score_linearized_delta",
                top_k=top_interval_k,
                phase_epsilon=phase_epsilon,
            ),
        },
        "unsupported_claims": unsupported,
    }


def _series_by_step(rows: list[dict[str, Any]], metric_name: str) -> dict[int, float]:
    series: dict[int, float] = {}
    for row in rows:
        step = _require_int(row.get("step"), "interval_row.step")
        if step in series:
            raise ValueError(f"Duplicate interval step in candidate rows: {step}")
        metrics = _candidate_interval_metrics(row)
        if metric_name not in metrics:
            raise ValueError(f"Unsupported series metric: {metric_name}")
        value = metrics[metric_name]
        if value is None:
            continue
        series[step] = float(value)
    return series


def _pairwise_candidate_relationships(
    *,
    selected_candidate_ids: list[str],
    rows_by_candidate: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(selected_candidate_ids):
        for right_id in selected_candidate_ids[left_index + 1:]:
            left_score = _series_by_step(rows_by_candidate[left_id], "feature_score_linearized_delta")
            right_score = _series_by_step(rows_by_candidate[right_id], "feature_score_linearized_delta")
            left_useful = _series_by_step(rows_by_candidate[left_id], "useful_delta")
            right_useful = _series_by_step(rows_by_candidate[right_id], "useful_delta")
            shared_score_steps = sorted(set(left_score) & set(right_score))
            shared_useful_steps = sorted(set(left_useful) & set(right_useful))
            relationship: dict[str, Any] = {
                "candidate_a": left_id,
                "candidate_b": right_id,
                "shared_score_interval_count": len(shared_score_steps),
                "shared_useful_interval_count": len(shared_useful_steps),
            }
            if shared_score_steps:
                left_score_values = [left_score[step] for step in shared_score_steps]
                right_score_values = [right_score[step] for step in shared_score_steps]
                conflict_count = sum(
                    1
                    for left_value, right_value in zip(left_score_values, right_score_values, strict=True)
                    if left_value * right_value < 0.0
                )
                relationship["score_correlation"] = _compute_pearson_correlation(left_score_values, right_score_values)
                relationship["score_sign_conflict_fraction"] = conflict_count / len(shared_score_steps)
                relationship["candidate_a_score_win_fraction"] = sum(
                    1
                    for left_value, right_value in zip(left_score_values, right_score_values, strict=True)
                    if left_value > right_value
                ) / len(shared_score_steps)
            else:
                relationship["score_correlation"] = {"status": "no_shared_feature_score_intervals", "value": None, "num_points": 0}
                relationship["score_sign_conflict_fraction"] = None
                relationship["candidate_a_score_win_fraction"] = None
            if shared_useful_steps:
                left_useful_values = [left_useful[step] for step in shared_useful_steps]
                right_useful_values = [right_useful[step] for step in shared_useful_steps]
                relationship["useful_correlation"] = _compute_pearson_correlation(left_useful_values, right_useful_values)
                relationship["simultaneous_useful_gain_fraction"] = sum(
                    1
                    for left_value, right_value in zip(left_useful_values, right_useful_values, strict=True)
                    if left_value > 0.0 and right_value > 0.0
                ) / len(shared_useful_steps)
                relationship["candidate_a_useful_win_fraction"] = sum(
                    1
                    for left_value, right_value in zip(left_useful_values, right_useful_values, strict=True)
                    if left_value > right_value
                ) / len(shared_useful_steps)
            else:
                relationship["useful_correlation"] = {"status": "no_shared_useful_intervals", "value": None, "num_points": 0}
                relationship["simultaneous_useful_gain_fraction"] = None
                relationship["candidate_a_useful_win_fraction"] = None
            relationships.append(relationship)
    return relationships


def _render_mechanism_scoreboard_plot(
    *,
    candidate_entries: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not candidate_entries:
        raise ValueError("Cannot render mechanism scoreboard without candidates.")
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [str(entry["candidate_id"]) for entry in candidate_entries]
    x_positions = list(range(len(labels)))
    metrics = [
        ("sum_useful_delta", "useful"),
        ("sum_heldout_gap_delta", "heldout"),
        ("feature_sum_score_linearized_delta", "score drive"),
    ]
    bar_width = 0.24
    for metric_index, (metric_name, label) in enumerate(metrics):
        values = [
            float("nan") if entry["totals"][metric_name] is None else float(entry["totals"][metric_name])
            for entry in candidate_entries
        ]
        offsets = [position + (metric_index - 1) * bar_width for position in x_positions]
        ax.bar(offsets, values, width=bar_width, label=label)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Candidate mechanism scoreboard")
    ax.set_ylabel("Cumulative delta")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_mechanism_cumulative_plot(
    *,
    selected_candidate_ids: list[str],
    rows_by_candidate: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> Path:
    if not selected_candidate_ids:
        raise ValueError("Cannot render cumulative plot without selected candidates.")
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for candidate_id in selected_candidate_ids:
        rows = sorted(rows_by_candidate[candidate_id], key=lambda row: int(row["step"]))
        steps = [int(row["step"]) for row in rows]
        useful = _cumulative([float(_candidate_interval_metrics(row)["useful_delta"]) for row in rows])
        heldout = _cumulative([float(_candidate_interval_metrics(row)["heldout_gap_delta"]) for row in rows])
        score_pairs = [
            (int(row["step"]), _candidate_interval_metrics(row)["feature_score_linearized_delta"])
            for row in rows
            if _candidate_interval_metrics(row)["feature_score_linearized_delta"] is not None
        ]
        axes[0].plot(steps, useful, marker="o", linewidth=1.8, label=candidate_id)
        axes[1].plot(steps, heldout, marker="o", linewidth=1.8, label=candidate_id)
        if score_pairs:
            score_steps = [step for step, _ in score_pairs]
            score_values = _cumulative([float(value) for _, value in score_pairs if value is not None])
            axes[2].plot(score_steps, score_values, marker="o", linewidth=1.8, label=candidate_id)
    axes[0].set_title("Cumulative useful movement")
    axes[0].set_ylabel("Useful delta")
    axes[1].set_title("Cumulative heldout movement")
    axes[1].set_ylabel("Heldout-gap delta")
    axes[2].set_title("Cumulative feature-score update drive")
    axes[2].set_ylabel("Score dot update")
    axes[2].set_xlabel("Checkpoint step")
    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_mechanism_component_plot(
    *,
    candidate_entries: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    component_rows: list[dict[str, Any]] = []
    for entry in candidate_entries:
        component_summary = _require_dict(entry.get("component_summary"), "candidate_entry.component_summary")
        if component_summary.get("status") != "computed":
            continue
        for raw_group in _require_list(component_summary.get("groups"), "component_summary.groups"):
            group = _require_dict(raw_group, "component_summary.groups[]")
            component_rows.append(
                {
                    "label": f"{entry['candidate_id']}:{group['group_name']}",
                    "loss": _require_number(
                        group.get("sum_loss_reduction_linearized"),
                        "component_summary.group.sum_loss_reduction_linearized",
                    ),
                    "score": group.get("sum_score_linearized_delta"),
                }
            )
    if not component_rows:
        return None
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [row["label"] for row in component_rows]
    x_positions = list(range(len(labels)))
    bar_width = 0.34
    ax.bar(
        [position - bar_width / 2 for position in x_positions],
        [float(row["loss"]) for row in component_rows],
        width=bar_width,
        label="loss reduction",
    )
    ax.bar(
        [position + bar_width / 2 for position in x_positions],
        [float("nan") if row["score"] is None else float(row["score"]) for row in component_rows],
        width=bar_width,
        label="score drive",
    )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Component-level projected update contributions")
    ax.set_ylabel("Cumulative dot update")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_competition_matrix_plot(
    *,
    selected_candidate_ids: list[str],
    pairwise_relationships: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not selected_candidate_ids:
        raise ValueError("Cannot render competition matrix without selected candidates.")
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(selected_candidate_ids)}
    matrix = [[0.0 for _ in selected_candidate_ids] for _ in selected_candidate_ids]
    for relationship in pairwise_relationships:
        left_id = str(relationship["candidate_a"])
        right_id = str(relationship["candidate_b"])
        value = relationship.get("score_sign_conflict_fraction")
        plotted_value = float("nan") if value is None else float(value)
        left_index = index_by_id[left_id]
        right_index = index_by_id[right_id]
        matrix[left_index][right_index] = plotted_value
        matrix[right_index][left_index] = plotted_value
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("Pairwise score-drive sign conflict")
    ax.set_xticks(list(range(len(selected_candidate_ids))))
    ax.set_xticklabels(selected_candidate_ids, rotation=35, ha="right")
    ax.set_yticks(list(range(len(selected_candidate_ids))))
    ax.set_yticklabels(selected_candidate_ids)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            label = "n/a" if value != value else f"{value:.2f}"
            ax.text(column_index, row_index, label, ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="conflict fraction")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_mechanism_plots(
    *,
    candidate_entries: list[dict[str, Any]],
    selected_candidate_ids: list[str],
    rows_by_candidate: dict[str, list[dict[str, Any]]],
    pairwise_relationships: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    plot_paths: dict[str, Path] = {
        "scoreboard": _render_mechanism_scoreboard_plot(
            candidate_entries=candidate_entries,
            output_path=output_dir / "candidate_mechanism_scoreboard.svg",
        ),
        "cumulative": _render_mechanism_cumulative_plot(
            selected_candidate_ids=selected_candidate_ids,
            rows_by_candidate=rows_by_candidate,
            output_path=output_dir / "candidate_mechanism_cumulative.svg",
        ),
        "competition_matrix": _render_competition_matrix_plot(
            selected_candidate_ids=selected_candidate_ids,
            pairwise_relationships=pairwise_relationships,
            output_path=output_dir / "candidate_mechanism_competition_matrix.svg",
        ),
    }
    component_plot = _render_mechanism_component_plot(
        candidate_entries=candidate_entries,
        output_path=output_dir / "candidate_mechanism_components.svg",
    )
    if component_plot is not None:
        plot_paths["components"] = component_plot
    return plot_paths


def _markdown_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    return f"{float(value):.6g}"


def _write_mechanism_markdown(
    *,
    output_path: Path,
    report_payload: dict[str, Any],
) -> Path:
    candidates = _require_list(report_payload.get("candidate_reports"), "mechanism_report.candidate_reports")
    lines = [
        "# Candidate Mechanism Report",
        "",
        "This report connects candidate feature-family movement to the observed checkpoint update:",
        "",
        "`Delta score_i ~= grad_theta score_i · Delta theta`",
        "",
        "`Delta loss ~= grad_theta loss · Delta theta`",
        "",
        "It is a checkpoint-interval report, not a per-minibatch SGD trace.",
        "",
        "## Inputs",
        "",
        f"- Registry: `{report_payload['registry_path']}`",
        f"- Gradient link: `{report_payload['gradient_link_path']}`",
        f"- Selected candidates: {', '.join(report_payload['selected_candidate_ids'])}",
        "",
        "## Candidate Scoreboard",
        "",
        "| candidate | status | useful | heldout | score drive | intervals |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for raw_candidate in candidates:
        candidate = _require_dict(raw_candidate, "mechanism_report.candidate_reports[]")
        totals = _require_dict(candidate.get("totals"), "candidate_report.totals")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate["candidate_id"]),
                    str(candidate["status_label"]),
                    _markdown_number(totals["sum_useful_delta"]),
                    _markdown_number(totals["sum_heldout_gap_delta"]),
                    _markdown_number(totals["feature_sum_score_linearized_delta"]),
                    str(totals["interval_count"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Phase Windows", ""])
    for raw_candidate in candidates:
        candidate = _require_dict(raw_candidate, "mechanism_report.candidate_reports[]")
        lines.append(f"### {candidate['candidate_id']}")
        lines.append("")
        lines.append("| window | steps | useful | heldout | score drive |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for raw_window in _require_list(candidate.get("phase_windows"), "candidate_report.phase_windows"):
            window = _require_dict(raw_window, "candidate_report.phase_windows[]")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(window["label"]),
                        f"{window['start_step']} to {window['end_step']}",
                        _markdown_number(window["sum_useful_delta"]),
                        _markdown_number(window["sum_heldout_gap_delta"]),
                        _markdown_number(window["feature_sum_score_linearized_delta"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(["## Component Responsibility", ""])
    for raw_candidate in candidates:
        candidate = _require_dict(raw_candidate, "mechanism_report.candidate_reports[]")
        component_summary = _require_dict(candidate.get("component_summary"), "candidate_report.component_summary")
        lines.append(f"### {candidate['candidate_id']}")
        lines.append("")
        if component_summary["status"] != "computed":
            lines.append(f"- Status: `{component_summary['status']}`")
            lines.append(f"- Reason: {component_summary['reason']}")
            lines.append("")
            continue
        lines.append("| group | params | loss reduction | score drive | mean loss cosine | mean score cosine |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for raw_group in _require_list(component_summary.get("groups"), "component_summary.groups"):
            group = _require_dict(raw_group, "component_summary.groups[]")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(group["group_name"]),
                        str(group["num_parameters"]),
                        _markdown_number(group["sum_loss_reduction_linearized"]),
                        _markdown_number(group["sum_score_linearized_delta"]),
                        _markdown_number(group["mean_update_negative_gradient_cosine"]),
                        _markdown_number(group["mean_update_score_gradient_cosine"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(["## Pairwise Competition", ""])
    relationships = _require_list(report_payload.get("pairwise_relationships"), "mechanism_report.pairwise_relationships")
    if relationships:
        lines.append("| candidate A | candidate B | score conflict | useful co-gain | score corr | useful corr |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for raw_relationship in relationships:
            relationship = _require_dict(raw_relationship, "mechanism_report.pairwise_relationships[]")
            score_corr = _require_dict(relationship.get("score_correlation"), "relationship.score_correlation")
            useful_corr = _require_dict(relationship.get("useful_correlation"), "relationship.useful_correlation")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(relationship["candidate_a"]),
                        str(relationship["candidate_b"]),
                        _markdown_number(relationship["score_sign_conflict_fraction"]),
                        _markdown_number(relationship["simultaneous_useful_gain_fraction"]),
                        _markdown_number(score_corr["value"]),
                        _markdown_number(useful_corr["value"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("Only one candidate was selected, so no pairwise competition table was generated.")
    lines.extend(["", "## Unsupported Claims", ""])
    for item in _require_list(report_payload.get("unsupported_claims"), "mechanism_report.unsupported_claims"):
        lines.append(f"- {item}")
    lines.extend(["", "## Plots", ""])
    for name, plot_path in _require_dict(report_payload.get("plots"), "mechanism_report.plots").items():
        lines.append(f"- {name}: `{plot_path}`")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_candidate_mechanism_report(
    *,
    registry_path: Path,
    gradient_link_path: Path,
    output_dir: Path,
    candidate_ids: list[str] | None = None,
    top_k: int = 4,
    ranking_metric: str = "sum_useful_delta",
    phase_epsilon: float = 0.0,
    top_interval_k: int = 5,
) -> tuple[Path, Path, dict[str, Path]]:
    registry = _load_candidate_registry(registry_path)
    gradient_payload = _load_gradient_link_payload(gradient_link_path)
    summaries = _require_dict(gradient_payload.get("candidate_summaries"), "gradient_link.candidate_summaries")
    registry_lookup = _candidate_lookup_from_registry(registry)
    selected_candidate_ids = _select_mechanism_candidate_ids(
        summaries=summaries,
        registry_lookup=registry_lookup,
        candidate_ids=candidate_ids,
        top_k=top_k,
        ranking_metric=ranking_metric,
    )
    interval_rows = [
        _require_dict(row, f"gradient_link.interval_rows[{index}]")
        for index, row in enumerate(_require_list(gradient_payload.get("interval_rows"), "gradient_link.interval_rows"))
    ]
    rows_by_candidate = _group_rows_by_candidate(interval_rows)
    missing_rows = [candidate_id for candidate_id in selected_candidate_ids if candidate_id not in rows_by_candidate]
    if missing_rows:
        raise KeyError(f"Selected candidates have no interval rows: {missing_rows}")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_entries = [
        _build_candidate_mechanism_entry(
            candidate_id=candidate_id,
            candidate=registry_lookup[candidate_id],
            summary=_require_dict(summaries[candidate_id], f"candidate_summaries.{candidate_id}"),
            rows=rows_by_candidate[candidate_id],
            phase_epsilon=phase_epsilon,
            top_interval_k=top_interval_k,
        )
        for candidate_id in selected_candidate_ids
    ]
    pairwise_relationships = _pairwise_candidate_relationships(
        selected_candidate_ids=selected_candidate_ids,
        rows_by_candidate=rows_by_candidate,
    )
    unsupported_claims = sorted(
        {
            claim
            for entry in candidate_entries
            for claim in _require_list(entry.get("unsupported_claims"), "candidate_report.unsupported_claims")
        }
        | {
            "per_minibatch_sgd_update_trace",
            "cross_seed_reproducibility",
            "complete_dense_circuit_decomposition",
        }
    )
    plot_paths = _render_mechanism_plots(
        candidate_entries=candidate_entries,
        selected_candidate_ids=selected_candidate_ids,
        rows_by_candidate=rows_by_candidate,
        pairwise_relationships=pairwise_relationships,
        output_dir=output_dir,
    )
    report_path = output_dir / "candidate_mechanism_report.json"
    markdown_path = output_dir / "candidate_mechanism_report.md"
    report_payload = {
        "schema_version": MECHANISM_REPORT_SCHEMA_VERSION,
        "registry_path": str(registry_path),
        "gradient_link_path": str(gradient_link_path),
        "selected_candidate_ids": selected_candidate_ids,
        "selection": {
            "candidate_ids": candidate_ids,
            "top_k": top_k,
            "ranking_metric": ranking_metric,
            "phase_epsilon": phase_epsilon,
            "top_interval_k": top_interval_k,
        },
        "checkpoint_window": {
            "start_step": min(int(rows_by_candidate[candidate_id][0]["previous_step"]) for candidate_id in selected_candidate_ids),
            "end_step": max(int(rows_by_candidate[candidate_id][-1]["step"]) for candidate_id in selected_candidate_ids),
        },
        "candidate_reports": candidate_entries,
        "pairwise_relationships": pairwise_relationships,
        "unsupported_claims": unsupported_claims,
        "plots": {name: str(path) for name, path in plot_paths.items()},
    }
    write_json(report_path, report_payload)
    _write_mechanism_markdown(output_path=markdown_path, report_payload=report_payload)
    return report_path, markdown_path, plot_paths


def _select_birth_model_candidate_ids(
    *,
    registry_lookup: dict[str, dict[str, Any]],
    rows_by_candidate: dict[str, list[dict[str, Any]]],
    candidate_ids: list[str] | None,
) -> list[str]:
    available_ids = sorted(set(registry_lookup) & set(rows_by_candidate))
    if not available_ids:
        raise ValueError("No candidates are shared between the registry and gradient-link interval rows.")
    if candidate_ids is None:
        return available_ids
    selected: list[str] = []
    seen: set[str] = set()
    for raw_candidate_id in candidate_ids:
        candidate_id = _sanitize_candidate_id(raw_candidate_id)
        if candidate_id in seen:
            raise ValueError(f"Duplicate selected candidate_id: {candidate_id}")
        if candidate_id not in registry_lookup:
            raise KeyError(f"Selected candidate_id is missing from registry: {candidate_id}")
        if candidate_id not in rows_by_candidate:
            raise KeyError(f"Selected candidate_id is missing from gradient-link interval rows: {candidate_id}")
        selected.append(candidate_id)
        seen.add(candidate_id)
    if not selected:
        raise ValueError("candidate_ids must not be empty when provided.")
    return selected


def _candidate_actual_birth_step(
    *,
    candidate: dict[str, Any],
    candidate_id: str,
    birth_metric: str,
) -> int:
    if birth_metric not in {"birth_step", "useful_birth_step"}:
        raise ValueError(f"Unsupported birth_metric: {birth_metric}")
    subset_birth = _require_dict(candidate.get("subset_birth"), f"candidate {candidate_id}.subset_birth")
    raw_step = subset_birth.get(birth_metric)
    if raw_step is None:
        raise ValueError(f"candidate {candidate_id} has null subset_birth.{birth_metric}.")
    return _require_int(raw_step, f"candidate {candidate_id}.subset_birth.{birth_metric}")


def _latest_subset_state_at_or_before(
    *,
    candidate: dict[str, Any],
    max_step: int,
    candidate_id: str,
) -> dict[str, Any]:
    rows_by_step = _candidate_rows_by_step(candidate)
    eligible_steps = [step for step in rows_by_step if step <= max_step]
    if not eligible_steps:
        raise ValueError(f"Candidate {candidate_id} has no subset trajectory rows at or before step {max_step}.")
    return rows_by_step[max(eligible_steps)]


def _birth_model_interval_rows(
    *,
    rows: list[dict[str, Any]],
    actual_birth_step: int,
    candidate_id: str,
    prediction_cutoff_step: int | None,
    lookback_intervals: int | None,
) -> tuple[list[dict[str, Any]], int, bool, str]:
    if lookback_intervals is not None and lookback_intervals <= 0:
        raise ValueError("lookback_intervals must be positive when provided.")
    sorted_rows = sorted(rows, key=lambda row: int(row["step"]))
    if prediction_cutoff_step is None:
        selected = [row for row in sorted_rows if _require_int(row.get("step"), "interval_row.step") < actual_birth_step]
        mode = "strict_prebirth"
        uses_postbirth_information = False
    else:
        selected = [
            row
            for row in sorted_rows
            if _require_int(row.get("step"), "interval_row.step") <= prediction_cutoff_step
        ]
        mode = "fixed_cutoff"
        uses_postbirth_information = prediction_cutoff_step >= actual_birth_step
    if lookback_intervals is not None:
        selected = selected[-lookback_intervals:]
    if not selected:
        raise ValueError(
            f"Candidate {candidate_id} has no eligible intervals for birth-model prediction "
            f"with actual_birth_step={actual_birth_step}, prediction_cutoff_step={prediction_cutoff_step}, "
            f"lookback_intervals={lookback_intervals}."
        )
    cutoff_step = _require_int(selected[-1].get("step"), "selected interval_row.step")
    return selected, cutoff_step, uses_postbirth_information, mode


def _positive_sum(values: list[float]) -> float:
    return sum(value for value in values if value > 0.0)


def _negative_abs_sum(values: list[float]) -> float:
    return sum(-value for value in values if value < 0.0)


def _required_non_null_number(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is null.")
    return _require_number(value, label)


def _birth_interval_raw_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    interval_metrics = _candidate_interval_metrics(row)
    parameter_projections = _require_dict(row.get("parameter_projections"), "interval_row.parameter_projections")
    union_projection = _require_dict(parameter_projections.get("candidate_union"), "parameter_projections.candidate_union")
    score_projection = _feature_score_union_projection(row)
    if score_projection is None:
        raise ValueError(
            f"Candidate {row.get('candidate_id')} interval {row.get('previous_step')}->{row.get('step')} "
            "does not contain computed feature-score gradients."
        )
    return {
        "feature_score_linearized_delta": _require_number(
            score_projection.get("score_linearized_delta"),
            "feature_score_projection.candidate_union.score_linearized_delta",
        ),
        "update_score_gradient_cosine": None
        if score_projection.get("update_score_gradient_cosine") is None
        else _require_number(
            score_projection.get("update_score_gradient_cosine"),
            "feature_score_projection.candidate_union.update_score_gradient_cosine",
        ),
        "loss_reduction_linearized": _require_number(
            union_projection.get("loss_reduction_linearized"),
            "parameter_projections.candidate_union.loss_reduction_linearized",
        ),
        "update_negative_gradient_cosine": None
        if union_projection.get("update_negative_gradient_cosine") is None
        else _require_number(
            union_projection.get("update_negative_gradient_cosine"),
            "parameter_projections.candidate_union.update_negative_gradient_cosine",
        ),
        "update_l2_share_global": _required_non_null_number(
            union_projection.get("update_l2_share_global"),
            "parameter_projections.candidate_union.update_l2_share_global",
        ),
        "gradient_l2_share_global": _required_non_null_number(
            union_projection.get("gradient_l2_share_global"),
            "parameter_projections.candidate_union.gradient_l2_share_global",
        ),
        "mean_activation_delta": _require_number(
            interval_metrics["mean_activation_delta"],
            "candidate_state_deltas.mean_activation_delta",
        ),
        "active_fraction_delta": _require_number(
            interval_metrics["active_fraction_delta"],
            "candidate_state_deltas.active_fraction_delta",
        ),
        "useful_delta": _require_number(interval_metrics["useful_delta"], "candidate_state_deltas.useful_delta"),
        "heldout_gap_delta": _require_number(
            interval_metrics["heldout_gap_delta"],
            "candidate_state_deltas.heldout_gap_delta",
        ),
    }


def _build_birth_factor_row(
    *,
    candidate_id: str,
    candidate: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    cutoff_step: int,
    actual_birth_step: int,
    uses_postbirth_information: bool,
    prediction_mode: str,
) -> dict[str, Any]:
    if not selected_rows:
        raise ValueError(f"Candidate {candidate_id} has no selected rows for birth-factor computation.")
    interval_metrics = [_birth_interval_raw_metrics(row) for row in selected_rows]
    latest_subset_state = _latest_subset_state_at_or_before(
        candidate=candidate,
        max_step=cutoff_step,
        candidate_id=candidate_id,
    )
    feature_score_deltas = [
        _required_non_null_number(metrics["feature_score_linearized_delta"], "feature_score_linearized_delta")
        for metrics in interval_metrics
    ]
    useful_deltas = [
        _required_non_null_number(metrics["useful_delta"], "useful_delta")
        for metrics in interval_metrics
    ]
    mean_activation_deltas = [
        _required_non_null_number(metrics["mean_activation_delta"], "mean_activation_delta")
        for metrics in interval_metrics
    ]
    active_fraction_deltas = [
        _required_non_null_number(metrics["active_fraction_delta"], "active_fraction_delta")
        for metrics in interval_metrics
    ]
    update_shares = [
        _required_non_null_number(metrics["update_l2_share_global"], "update_l2_share_global")
        for metrics in interval_metrics
    ]
    gradient_shares = [
        _required_non_null_number(metrics["gradient_l2_share_global"], "gradient_l2_share_global")
        for metrics in interval_metrics
    ]
    score_cosines = [
        _require_number(metrics["update_score_gradient_cosine"], "update_score_gradient_cosine")
        for metrics in interval_metrics
        if metrics["update_score_gradient_cosine"] is not None
    ]
    loss_cosines = [
        _require_number(metrics["update_negative_gradient_cosine"], "update_negative_gradient_cosine")
        for metrics in interval_metrics
        if metrics["update_negative_gradient_cosine"] is not None
    ]
    if not score_cosines:
        raise ValueError(f"Candidate {candidate_id} has no non-null feature-score cosine values in the prediction window.")
    if not loss_cosines:
        raise ValueError(f"Candidate {candidate_id} has no non-null loss-gradient cosine values in the prediction window.")
    activation_support = _require_number(
        latest_subset_state.get("mean_activation_mean"),
        f"candidate {candidate_id} latest_subset_state.mean_activation_mean",
    ) + _require_number(
        latest_subset_state.get("active_fraction_mean"),
        f"candidate {candidate_id} latest_subset_state.active_fraction_mean",
    )
    raw_factors = {
        "feature_score_drive": sum(feature_score_deltas),
        "gradient_alignment": sum(score_cosines) / len(score_cosines),
        "loss_utility": sum(metrics["loss_reduction_linearized"] for metrics in interval_metrics),
        "component_accessibility": (sum(update_shares) / len(update_shares)) + (sum(gradient_shares) / len(gradient_shares)),
        "activation_support": activation_support,
        "amplification": _positive_sum(mean_activation_deltas) + _positive_sum(active_fraction_deltas),
        "interference_cost": _negative_abs_sum(feature_score_deltas) + _negative_abs_sum(useful_deltas),
    }
    return {
        "candidate_id": candidate_id,
        "stage_name": _require_non_empty_str(candidate.get("stage_name"), f"candidate {candidate_id}.stage_name"),
        "family_id": candidate.get("family_id"),
        "feature_ids": _coerce_int_list(candidate.get("feature_ids"), f"candidate {candidate_id}.feature_ids"),
        "actual_birth_step": actual_birth_step,
        "prediction_cutoff_step": cutoff_step,
        "prediction_mode": prediction_mode,
        "uses_postbirth_information": uses_postbirth_information,
        "interval_count": len(selected_rows),
        "first_interval_start_step": _require_int(selected_rows[0].get("previous_step"), "selected interval previous_step"),
        "last_interval_end_step": _require_int(selected_rows[-1].get("step"), "selected interval step"),
        "raw_factors": raw_factors,
        "raw_submetrics": {
            "feature_score_positive_sum": _positive_sum(feature_score_deltas),
            "feature_score_negative_abs_sum": _negative_abs_sum(feature_score_deltas),
            "useful_positive_sum": _positive_sum(useful_deltas),
            "useful_negative_abs_sum": _negative_abs_sum(useful_deltas),
            "heldout_sum": sum(metrics["heldout_gap_delta"] for metrics in interval_metrics),
            "mean_loss_alignment": sum(loss_cosines) / len(loss_cosines),
            "score_alignment_interval_count": len(score_cosines),
            "score_alignment_null_interval_count": len(interval_metrics) - len(score_cosines),
            "loss_alignment_interval_count": len(loss_cosines),
            "loss_alignment_null_interval_count": len(interval_metrics) - len(loss_cosines),
            "mean_update_l2_share_global": sum(update_shares) / len(update_shares),
            "mean_gradient_l2_share_global": sum(gradient_shares) / len(gradient_shares),
            "latest_mean_activation": _require_number(
                latest_subset_state.get("mean_activation_mean"),
                f"candidate {candidate_id} latest_subset_state.mean_activation_mean",
            ),
            "latest_active_fraction": _require_number(
                latest_subset_state.get("active_fraction_mean"),
                f"candidate {candidate_id} latest_subset_state.active_fraction_mean",
            ),
            "positive_mean_activation_delta_sum": _positive_sum(mean_activation_deltas),
            "positive_active_fraction_delta_sum": _positive_sum(active_fraction_deltas),
        },
    }


def _add_birth_model_scores(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidate_rows:
        raise ValueError("candidate_rows must not be empty.")
    factor_stats: dict[str, dict[str, Any]] = {}
    for spec in BIRTH_MODEL_FACTOR_SPECS:
        factor_name = str(spec["name"])
        values = [
            _require_number(row["raw_factors"].get(factor_name), f"birth_model.raw_factors.{factor_name}")
            for row in candidate_rows
        ]
        minimum = min(values)
        maximum = max(values)
        factor_stats[factor_name] = {
            "min": minimum,
            "max": maximum,
            "status": "constant_across_candidates" if maximum == minimum else "normalized",
        }

    scored_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        contributions: dict[str, Any] = {}
        birth_model_score = 0.0
        for spec in BIRTH_MODEL_FACTOR_SPECS:
            factor_name = str(spec["name"])
            orientation = str(spec["orientation"])
            raw_value = _require_number(row["raw_factors"].get(factor_name), f"birth_model.raw_factors.{factor_name}")
            stats = factor_stats[factor_name]
            if stats["status"] == "constant_across_candidates":
                normalized_value = 0.0
            else:
                normalized_value = (raw_value - float(stats["min"])) / (float(stats["max"]) - float(stats["min"]))
            contribution = normalized_value if orientation == "positive" else -normalized_value
            birth_model_score += contribution
            contributions[factor_name] = {
                "raw_value": raw_value,
                "normalized_value": normalized_value,
                "orientation": orientation,
                "weight": 1.0,
                "contribution": contribution,
                "normalization_status": stats["status"],
            }
        scored_rows.append(
            {
                **row,
                "factor_contributions": contributions,
                "birth_model_score": birth_model_score,
            }
        )
    predicted_order = sorted(scored_rows, key=lambda item: (-float(item["birth_model_score"]), str(item["candidate_id"])))
    actual_order = sorted(scored_rows, key=lambda item: (int(item["actual_birth_step"]), str(item["candidate_id"])))
    predicted_rank_by_id = {
        str(row["candidate_id"]): rank
        for rank, row in enumerate(predicted_order, start=1)
    }
    actual_rank_by_id = {
        str(row["candidate_id"]): rank
        for rank, row in enumerate(actual_order, start=1)
    }
    return [
        {
            **row,
            "predicted_birth_rank": predicted_rank_by_id[str(row["candidate_id"])],
            "actual_birth_rank": actual_rank_by_id[str(row["candidate_id"])],
            "birth_rank_error": predicted_rank_by_id[str(row["candidate_id"])] - actual_rank_by_id[str(row["candidate_id"])],
        }
        for row in scored_rows
    ]


def _first_feature_score_crossing_step(
    *,
    rows: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    cumulative_score_drive = 0.0
    for row in sorted(rows, key=lambda item: int(item["step"])):
        metrics = _birth_interval_raw_metrics(row)
        cumulative_score_drive += metrics["feature_score_linearized_delta"]
        step = _require_int(row.get("step"), "interval_row.step")
        if cumulative_score_drive >= threshold:
            return {
                "status": "crossed",
                "step": step,
                "threshold": threshold,
                "cumulative_feature_score_drive": cumulative_score_drive,
            }
    return {
        "status": "not_crossed",
        "step": None,
        "threshold": threshold,
        "cumulative_feature_score_drive": cumulative_score_drive,
    }


def _rank_correlation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < 2:
        return {"status": "insufficient_candidates", "value": None, "num_candidates": len(rows)}
    predicted = [float(row["predicted_birth_rank"]) for row in rows]
    actual = [float(row["actual_birth_rank"]) for row in rows]
    correlation = _compute_pearson_correlation(predicted, actual)
    return {
        "status": correlation["status"],
        "value": correlation["value"],
        "num_candidates": len(rows),
    }


def _render_birth_model_scoreboard_plot(
    *,
    candidate_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not candidate_rows:
        raise ValueError("Cannot render birth model scoreboard without candidates.")
    ordered = sorted(candidate_rows, key=lambda row: int(row["predicted_birth_rank"]))
    _, plt = _import_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    labels = [str(row["candidate_id"]) for row in ordered]
    x_positions = list(range(len(labels)))
    axes[0].bar(x_positions, [float(row["birth_model_score"]) for row in ordered])
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    axes[0].set_title("Candidate birth-model score")
    axes[0].set_ylabel("score")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x_positions, [int(row["actual_birth_step"]) for row in ordered])
    axes[1].set_title("Actual birth step")
    axes[1].set_ylabel("checkpoint step")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_birth_model_factor_plot(
    *,
    candidate_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not candidate_rows:
        raise ValueError("Cannot render birth model factor plot without candidates.")
    ordered = sorted(candidate_rows, key=lambda row: int(row["predicted_birth_rank"]))
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [str(row["candidate_id"]) for row in ordered]
    x_positions = list(range(len(labels)))
    positive_bottoms = [0.0 for _ in ordered]
    negative_bottoms = [0.0 for _ in ordered]
    for spec in BIRTH_MODEL_FACTOR_SPECS:
        factor_name = str(spec["name"])
        values = [
            float(row["factor_contributions"][factor_name]["contribution"])
            for row in ordered
        ]
        bottoms = [
            positive_bottoms[index] if value >= 0.0 else negative_bottoms[index]
            for index, value in enumerate(values)
        ]
        ax.bar(x_positions, values, bottom=bottoms, label=factor_name)
        for index, value in enumerate(values):
            if value >= 0.0:
                positive_bottoms[index] += value
            else:
                negative_bottoms[index] += value
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Birth-model factor contributions")
    ax.set_ylabel("normalized contribution")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_birth_model_order_plot(
    *,
    candidate_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not candidate_rows:
        raise ValueError("Cannot render birth model order plot without candidates.")
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 7))
    xs = [int(row["actual_birth_step"]) for row in candidate_rows]
    ys = [float(row["birth_model_score"]) for row in candidate_rows]
    ax.scatter(xs, ys, alpha=0.8)
    for row in candidate_rows:
        ax.annotate(
            str(row["candidate_id"]),
            (int(row["actual_birth_step"]), float(row["birth_model_score"])),
            fontsize=8,
            alpha=0.85,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Birth-model score vs actual birth step")
    ax.set_xlabel("Actual birth step")
    ax.set_ylabel("Birth-model score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_birth_model_plots(
    *,
    candidate_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Path]:
    return {
        "scoreboard": _render_birth_model_scoreboard_plot(
            candidate_rows=candidate_rows,
            output_path=output_dir / "candidate_birth_model_scoreboard.svg",
        ),
        "factors": _render_birth_model_factor_plot(
            candidate_rows=candidate_rows,
            output_path=output_dir / "candidate_birth_model_factors.svg",
        ),
        "birth_order": _render_birth_model_order_plot(
            candidate_rows=candidate_rows,
            output_path=output_dir / "candidate_birth_model_birth_order.svg",
        ),
    }


def _write_birth_model_markdown(
    *,
    output_path: Path,
    report_payload: dict[str, Any],
) -> Path:
    rows = _require_list(report_payload.get("candidate_rows"), "birth_model.candidate_rows")
    lines = [
        "# Candidate Birth Model",
        "",
        "This report tests whether candidate formation is predictable before, or at a declared cutoff around, birth.",
        "",
        "The factor model is transparent and intentionally simple:",
        "",
        "`birth_score(c) = sum(positive normalized factors) - sum(normalized cost factors)`",
        "",
        "It is a candidate ranking model, not a closed-form theory of SGD.",
        "",
        "## Inputs",
        "",
        f"- Registry: `{report_payload['registry_path']}`",
        f"- Gradient link: `{report_payload['gradient_link_path']}`",
        f"- Birth metric: `{report_payload['selection']['birth_metric']}`",
        f"- Prediction mode: `{report_payload['selection']['prediction_mode']}`",
        f"- Effective prediction cutoff: `{report_payload['selection']['effective_prediction_cutoff_step']}`",
        "",
        "## Candidate Ranking",
        "",
        "| candidate | score | predicted rank | actual birth | actual rank | cutoff | leakage | first score crossing |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for raw_row in sorted(rows, key=lambda row: int(row["predicted_birth_rank"])):
        row = _require_dict(raw_row, "birth_model.candidate_rows[]")
        crossing = _require_dict(row.get("first_feature_score_crossing"), "candidate_row.first_feature_score_crossing")
        crossing_step = "n/a" if crossing["step"] is None else str(crossing["step"])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["candidate_id"]),
                    _markdown_number(row["birth_model_score"]),
                    str(row["predicted_birth_rank"]),
                    str(row["actual_birth_step"]),
                    str(row["actual_birth_rank"]),
                    str(row["prediction_cutoff_step"]),
                    str(row["uses_postbirth_information"]),
                    crossing_step,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Factor Decomposition", ""])
    factor_names = [str(spec["name"]) for spec in BIRTH_MODEL_FACTOR_SPECS]
    lines.append("| candidate | " + " | ".join(factor_names) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in factor_names) + " |")
    for raw_row in sorted(rows, key=lambda row: int(row["predicted_birth_rank"])):
        row = _require_dict(raw_row, "birth_model.candidate_rows[]")
        contributions = _require_dict(row.get("factor_contributions"), "candidate_row.factor_contributions")
        lines.append(
            "| "
            + " | ".join(
                [str(row["candidate_id"])]
                + [
                    _markdown_number(
                        _require_dict(contributions.get(factor_name), f"factor_contributions.{factor_name}")[
                            "contribution"
                        ]
                    )
                    for factor_name in factor_names
                ]
            )
            + " |"
        )
    lines.extend(["", "## Factor Definitions", ""])
    for spec in BIRTH_MODEL_FACTOR_SPECS:
        lines.append(f"- `{spec['name']}` ({spec['orientation']}): {spec['description']}")
    lines.extend(["", "## Unsupported Claims", ""])
    for claim in _require_list(report_payload.get("unsupported_claims"), "birth_model.unsupported_claims"):
        lines.append(f"- {claim}")
    lines.extend(["", "## Plots", ""])
    for name, plot_path in _require_dict(report_payload.get("plots"), "birth_model.plots").items():
        lines.append(f"- {name}: `{plot_path}`")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_candidate_birth_model(
    *,
    registry_path: Path,
    gradient_link_path: Path,
    output_dir: Path,
    candidate_ids: list[str] | None = None,
    birth_metric: str = "useful_birth_step",
    prediction_cutoff_step: int | None = None,
    lookback_intervals: int | None = None,
    birth_score_threshold: float = 0.0,
) -> tuple[Path, Path, dict[str, Path]]:
    registry = _load_candidate_registry(registry_path)
    gradient_payload = _load_gradient_link_payload(gradient_link_path)
    registry_lookup = _candidate_lookup_from_registry(registry)
    interval_rows = [
        _require_dict(row, f"gradient_link.interval_rows[{index}]")
        for index, row in enumerate(_require_list(gradient_payload.get("interval_rows"), "gradient_link.interval_rows"))
    ]
    rows_by_candidate = _group_rows_by_candidate(interval_rows)
    selected_candidate_ids = _select_birth_model_candidate_ids(
        registry_lookup=registry_lookup,
        rows_by_candidate=rows_by_candidate,
        candidate_ids=candidate_ids,
    )
    actual_birth_steps = {
        candidate_id: _candidate_actual_birth_step(
            candidate=registry_lookup[candidate_id],
            candidate_id=candidate_id,
            birth_metric=birth_metric,
        )
        for candidate_id in selected_candidate_ids
    }
    if prediction_cutoff_step is None:
        earliest_birth_step = min(actual_birth_steps.values())
        eligible_shared_cutoff_steps = sorted(
            {
                _require_int(row.get("step"), "interval_row.step")
                for candidate_id in selected_candidate_ids
                for row in rows_by_candidate[candidate_id]
                if _require_int(row.get("step"), "interval_row.step") < earliest_birth_step
            }
        )
        if not eligible_shared_cutoff_steps:
            raise ValueError(
                f"No checkpoint interval ends before the earliest selected {birth_metric}={earliest_birth_step}."
            )
        effective_prediction_cutoff_step = eligible_shared_cutoff_steps[-1]
        selection_prediction_mode = "shared_strict_prebirth"
    else:
        effective_prediction_cutoff_step = prediction_cutoff_step
        selection_prediction_mode = "fixed_cutoff"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_candidate_rows: list[dict[str, Any]] = []
    for candidate_id in selected_candidate_ids:
        candidate = registry_lookup[candidate_id]
        actual_birth_step = actual_birth_steps[candidate_id]
        selected_rows, cutoff_step, uses_postbirth_information, _prediction_mode = _birth_model_interval_rows(
            rows=rows_by_candidate[candidate_id],
            actual_birth_step=actual_birth_step,
            candidate_id=candidate_id,
            prediction_cutoff_step=effective_prediction_cutoff_step,
            lookback_intervals=lookback_intervals,
        )
        raw_row = _build_birth_factor_row(
            candidate_id=candidate_id,
            candidate=candidate,
            selected_rows=selected_rows,
            cutoff_step=cutoff_step,
            actual_birth_step=actual_birth_step,
            uses_postbirth_information=uses_postbirth_information,
            prediction_mode=selection_prediction_mode,
        )
        raw_row["first_feature_score_crossing"] = _first_feature_score_crossing_step(
            rows=rows_by_candidate[candidate_id],
            threshold=birth_score_threshold,
        )
        raw_candidate_rows.append(raw_row)

    candidate_rows = _add_birth_model_scores(raw_candidate_rows)
    candidate_rows.sort(key=lambda row: int(row["predicted_birth_rank"]))
    plots = _render_birth_model_plots(candidate_rows=candidate_rows, output_dir=output_dir)
    unsupported_claims = [
        "calibrated_absolute_birth_step_prediction",
        "cross_seed_reproducibility",
        "per_minibatch_sgd_update_trace",
        "causal_intervention_that_changes_selected_circuit",
    ]
    if any(bool(row["uses_postbirth_information"]) for row in candidate_rows):
        unsupported_claims.append("strict_prebirth_prediction_for_all_candidates")
    report_path = output_dir / "candidate_birth_model_report.json"
    markdown_path = output_dir / "candidate_birth_model_report.md"
    report_payload = {
        "schema_version": BIRTH_MODEL_SCHEMA_VERSION,
        "registry_path": str(registry_path),
        "gradient_link_path": str(gradient_link_path),
        "selected_candidate_ids": selected_candidate_ids,
        "selection": {
            "candidate_ids": candidate_ids,
            "birth_metric": birth_metric,
            "requested_prediction_cutoff_step": prediction_cutoff_step,
            "effective_prediction_cutoff_step": effective_prediction_cutoff_step,
            "lookback_intervals": lookback_intervals,
            "birth_score_threshold": birth_score_threshold,
            "prediction_mode": selection_prediction_mode,
            "factor_specs": BIRTH_MODEL_FACTOR_SPECS,
        },
        "rank_correlation": _rank_correlation(candidate_rows),
        "candidate_rows": candidate_rows,
        "unsupported_claims": unsupported_claims,
        "plots": {name: str(path) for name, path in plots.items()},
    }
    write_json(report_path, report_payload)
    _write_birth_model_markdown(output_path=markdown_path, report_payload=report_payload)
    return report_path, markdown_path, plots


def _candidate_source_basis_path(candidate: dict[str, Any], candidate_id: str) -> Path:
    artifacts = _require_dict(candidate.get("source_artifacts"), f"candidate {candidate_id}.source_artifacts")
    raw_basis_path = artifacts.get("shared_feature_basis")
    if raw_basis_path is None:
        raise ValueError(f"candidate {candidate_id} is missing source_artifacts.shared_feature_basis.")
    return Path(_require_non_empty_str(raw_basis_path, f"candidate {candidate_id}.source_artifacts.shared_feature_basis"))


def _coalition_target_specs(
    *,
    registry_lookup: dict[str, dict[str, Any]],
    selected_candidate_ids: list[str],
    include_individual_features: bool,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    feature_targets: dict[tuple[str, str, int], dict[str, Any]] = {}
    for candidate_id in selected_candidate_ids:
        candidate = registry_lookup[candidate_id]
        stage_name = _require_non_empty_str(candidate.get("stage_name"), f"candidate {candidate_id}.stage_name")
        feature_ids = _coerce_int_list(candidate.get("feature_ids"), f"candidate {candidate_id}.feature_ids")
        basis_path = _candidate_source_basis_path(candidate, candidate_id)
        targets.append(
            {
                "target_id": candidate_id,
                "target_kind": "candidate",
                "candidate_id": candidate_id,
                "family_id": candidate.get("family_id"),
                "stage_name": stage_name,
                "feature_ids": feature_ids,
                "basis_path": str(basis_path),
            }
        )
        if include_individual_features:
            for feature_id in feature_ids:
                key = (stage_name, str(basis_path), int(feature_id))
                if key not in feature_targets:
                    feature_targets[key] = {
                        "target_id": _sanitize_candidate_id(f"{stage_name}_feature_{feature_id}"),
                        "target_kind": "feature",
                        "candidate_ids": [candidate_id],
                        "family_ids": [candidate.get("family_id")],
                        "stage_name": stage_name,
                        "feature_ids": [int(feature_id)],
                        "basis_path": str(basis_path),
                    }
                else:
                    feature_targets[key]["candidate_ids"].append(candidate_id)
                    feature_targets[key]["family_ids"].append(candidate.get("family_id"))
    targets.extend(feature_targets[key] for key in sorted(feature_targets))
    seen_target_ids: set[str] = set()
    for target in targets:
        target_id = _sanitize_candidate_id(_require_non_empty_str(target.get("target_id"), "coalition target target_id"))
        if target_id in seen_target_ids:
            raise ValueError(f"Duplicate coalition target_id: {target_id}")
        seen_target_ids.add(target_id)
    return targets


def _coalition_neuron_layers(
    *,
    registry_lookup: dict[str, dict[str, Any]],
    selected_candidate_ids: list[str],
    extra_layers: list[int] | None,
) -> list[int]:
    layers: set[int] = set()
    for candidate_id in selected_candidate_ids:
        candidate = registry_lookup[candidate_id]
        for index, raw_group in enumerate(_require_list(candidate.get("parameter_groups"), f"candidate {candidate_id}.parameter_groups")):
            group = _require_dict(raw_group, f"candidate {candidate_id}.parameter_groups[{index}]")
            kind = _require_non_empty_str(group.get("kind"), f"candidate {candidate_id}.parameter_groups[{index}].kind")
            if kind in {"mlp_block", "mlp_neuron_group"}:
                layers.add(_require_int(group.get("layer"), f"candidate {candidate_id}.parameter_groups[{index}].layer"))
    if extra_layers is not None:
        for layer in extra_layers:
            if layer < 0:
                raise ValueError(f"Neuron layer must be non-negative: {layer}")
            layers.add(int(layer))
    if not layers:
        raise ValueError("No MLP neuron layers were found in selected candidates or provided with --neuron-layer.")
    return sorted(layers)


def _common_interval_pairs_for_candidates(
    *,
    rows_by_candidate: dict[str, list[dict[str, Any]]],
    selected_candidate_ids: list[str],
    start_step: int | None,
    end_step: int | None,
) -> list[tuple[int, int]]:
    common_pairs: set[tuple[int, int]] | None = None
    for candidate_id in selected_candidate_ids:
        if candidate_id not in rows_by_candidate:
            raise KeyError(f"Selected candidate has no gradient-link interval rows: {candidate_id}")
        pairs = {
            (
                _require_int(row.get("previous_step"), f"{candidate_id}.interval.previous_step"),
                _require_int(row.get("step"), f"{candidate_id}.interval.step"),
            )
            for row in rows_by_candidate[candidate_id]
            if (start_step is None or _require_int(row.get("step"), f"{candidate_id}.interval.step") >= start_step)
            and (end_step is None or _require_int(row.get("step"), f"{candidate_id}.interval.step") <= end_step)
        }
        common_pairs = pairs if common_pairs is None else common_pairs & pairs
    if not common_pairs:
        raise ValueError(
            f"No common gradient-link intervals remain after filtering start_step={start_step}, end_step={end_step}."
        )
    return sorted(common_pairs, key=lambda pair: (pair[1], pair[0]))


def _validate_neuron_layer_keys(state: dict[str, torch.Tensor], layer: int) -> dict[str, str]:
    keys = {
        "fc_in_weight": f"blocks.{layer}.ff.fc_in.weight",
        "fc_in_bias": f"blocks.{layer}.ff.fc_in.bias",
        "fc_out_weight": f"blocks.{layer}.ff.fc_out.weight",
    }
    missing = [key for key in keys.values() if key not in state]
    if missing:
        raise KeyError(f"Missing MLP neuron parameter keys for layer {layer}: {missing}")
    return keys


def _per_neuron_projection_arrays(
    *,
    previous_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
    layer: int,
) -> dict[str, torch.Tensor]:
    keys = _validate_neuron_layer_keys(previous_state, layer)
    for key in keys.values():
        if key not in current_state:
            raise KeyError(f"Current checkpoint is missing neuron parameter key: {key}")
        if key not in gradients:
            raise KeyError(f"Gradient payload is missing neuron parameter key: {key}")
        if tuple(previous_state[key].shape) != tuple(current_state[key].shape):
            raise ValueError(f"Checkpoint tensor shape mismatch for {key}.")
        if tuple(previous_state[key].shape) != tuple(gradients[key].shape):
            raise ValueError(f"Gradient tensor shape mismatch for {key}.")

    update_in_weight = current_state[keys["fc_in_weight"]].float() - previous_state[keys["fc_in_weight"]].float()
    update_in_bias = current_state[keys["fc_in_bias"]].float() - previous_state[keys["fc_in_bias"]].float()
    update_out_weight = current_state[keys["fc_out_weight"]].float() - previous_state[keys["fc_out_weight"]].float()
    grad_in_weight = gradients[keys["fc_in_weight"]].float()
    grad_in_bias = gradients[keys["fc_in_bias"]].float()
    grad_out_weight = gradients[keys["fc_out_weight"]].float()
    return {
        "dot": (update_in_weight * grad_in_weight).sum(dim=1)
        + (update_in_bias * grad_in_bias)
        + (update_out_weight * grad_out_weight).sum(dim=0),
        "update_sq": (update_in_weight * update_in_weight).sum(dim=1)
        + (update_in_bias * update_in_bias)
        + (update_out_weight * update_out_weight).sum(dim=0),
        "gradient_sq": (grad_in_weight * grad_in_weight).sum(dim=1)
        + (grad_in_bias * grad_in_bias)
        + (grad_out_weight * grad_out_weight).sum(dim=0),
    }


def _add_neuron_projection_accumulators(
    *,
    accumulator: dict[tuple[str, int, int], dict[str, Any]],
    target: dict[str, Any],
    layer: int,
    score_arrays: dict[str, torch.Tensor],
    loss_arrays: dict[str, torch.Tensor],
) -> None:
    score_dot = score_arrays["dot"].detach().cpu()
    loss_dot = loss_arrays["dot"].detach().cpu()
    update_sq = score_arrays["update_sq"].detach().cpu()
    score_grad_sq = score_arrays["gradient_sq"].detach().cpu()
    loss_grad_sq = loss_arrays["gradient_sq"].detach().cpu()
    if score_dot.ndim != 1:
        raise ValueError(f"Expected per-neuron score dot vector for layer {layer}, got {tuple(score_dot.shape)}")
    d_ff = int(score_dot.shape[0])
    if tuple(loss_dot.shape) != (d_ff,) or tuple(update_sq.shape) != (d_ff,):
        raise ValueError(f"Per-neuron projection shape mismatch for layer {layer}.")
    target_id = _sanitize_candidate_id(_require_non_empty_str(target.get("target_id"), "coalition target target_id"))
    for neuron in range(d_ff):
        key = (target_id, layer, neuron)
        row = accumulator.setdefault(
            key,
            {
                "target_id": target_id,
                "target_kind": target["target_kind"],
                "candidate_id": target.get("candidate_id"),
                "candidate_ids": target.get("candidate_ids"),
                "family_id": target.get("family_id"),
                "family_ids": target.get("family_ids"),
                "stage_name": target["stage_name"],
                "feature_ids": target["feature_ids"],
                "layer": layer,
                "neuron": neuron,
                "interval_count": 0,
                "sum_score_linearized_delta": 0.0,
                "positive_score_drive": 0.0,
                "negative_score_drive_abs": 0.0,
                "sum_loss_reduction_linearized": 0.0,
                "positive_loss_reduction": 0.0,
                "negative_loss_reduction_abs": 0.0,
                "sum_update_sq": 0.0,
                "sum_score_gradient_sq": 0.0,
                "sum_loss_gradient_sq": 0.0,
                "sum_score_dot": 0.0,
                "sum_negative_loss_dot": 0.0,
            },
        )
        current_score_dot = float(score_dot[neuron].item())
        current_loss_reduction = -float(loss_dot[neuron].item())
        row["interval_count"] += 1
        row["sum_score_linearized_delta"] += current_score_dot
        row["positive_score_drive"] += max(current_score_dot, 0.0)
        row["negative_score_drive_abs"] += max(-current_score_dot, 0.0)
        row["sum_loss_reduction_linearized"] += current_loss_reduction
        row["positive_loss_reduction"] += max(current_loss_reduction, 0.0)
        row["negative_loss_reduction_abs"] += max(-current_loss_reduction, 0.0)
        row["sum_update_sq"] += float(update_sq[neuron].item())
        row["sum_score_gradient_sq"] += float(score_grad_sq[neuron].item())
        row["sum_loss_gradient_sq"] += float(loss_grad_sq[neuron].item())
        row["sum_score_dot"] += current_score_dot
        row["sum_negative_loss_dot"] += current_loss_reduction


def _finalize_neuron_projection_rows(accumulator: dict[tuple[str, int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in accumulator.values():
        update_norm = float(row["sum_update_sq"]) ** 0.5
        score_grad_norm = float(row["sum_score_gradient_sq"]) ** 0.5
        loss_grad_norm = float(row["sum_loss_gradient_sq"]) ** 0.5
        rows.append(
            {
                **row,
                "update_l2_norm": update_norm,
                "score_gradient_l2_norm": score_grad_norm,
                "loss_gradient_l2_norm": loss_grad_norm,
                "update_score_gradient_cosine": _safe_ratio(
                    float(row["sum_score_dot"]),
                    update_norm * score_grad_norm,
                ),
                "update_negative_loss_gradient_cosine": _safe_ratio(
                    float(row["sum_negative_loss_dot"]),
                    update_norm * loss_grad_norm,
                ),
            }
        )
    rows.sort(key=lambda item: abs(float(item["sum_score_linearized_delta"])), reverse=True)
    return rows


def _gradient_vector_dot_for_neuron_layers(
    *,
    left_gradients: dict[str, torch.Tensor],
    right_gradients: dict[str, torch.Tensor],
    layers: list[int],
) -> dict[str, float]:
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    for layer in layers:
        keys = _validate_neuron_layer_keys(left_gradients, layer)
        for key in keys.values():
            if key not in right_gradients:
                raise KeyError(f"Right gradient payload is missing key: {key}")
            left = left_gradients[key].float()
            right = right_gradients[key].float()
            if tuple(left.shape) != tuple(right.shape):
                raise ValueError(f"Gradient shape mismatch for {key}: {tuple(left.shape)} vs {tuple(right.shape)}")
            dot += float(torch.sum(left * right).item())
            left_sq += float(torch.sum(left * left).item())
            right_sq += float(torch.sum(right * right).item())
    return {
        "dot": dot,
        "left_l2_norm": left_sq ** 0.5,
        "right_l2_norm": right_sq ** 0.5,
        "cosine": _safe_ratio(dot, (left_sq ** 0.5) * (right_sq ** 0.5)),
    }


def _target_feature_score_at_step(
    *,
    target: dict[str, Any],
    step: int,
    basis_cache: dict[str, dict[str, Any]],
    feature_score_cache: dict[tuple[str, int], dict[str, Any]],
    model_loader: Any,
    batches: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    target_id = _sanitize_candidate_id(_require_non_empty_str(target.get("target_id"), "coalition target target_id"))
    cache_key = (target_id, step)
    cached = feature_score_cache.get(cache_key)
    if cached is not None:
        return cached
    basis_path = Path(_require_non_empty_str(target.get("basis_path"), f"target {target_id}.basis_path"))
    basis_cache_key = str(basis_path)
    basis = basis_cache.get(basis_cache_key)
    if basis is None:
        basis = _load_shared_basis(basis_path, device)
        basis_cache[basis_cache_key] = basis
    model = model_loader(step)
    payload = _compute_feature_score_and_gradients(
        model=model,
        batches=batches,
        basis=basis,
        stage_name=_require_non_empty_str(target.get("stage_name"), f"target {target_id}.stage_name"),
        feature_ids=_coerce_int_list(target.get("feature_ids"), f"target {target_id}.feature_ids"),
    )
    feature_score_cache[cache_key] = payload
    return payload


def _coalition_pairwise_gradient_rows(
    *,
    target_specs: list[dict[str, Any]],
    score_gradients_by_target: dict[str, dict[str, torch.Tensor]],
    neuron_layers: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left_target in enumerate(target_specs):
        left_id = str(left_target["target_id"])
        for right_target in target_specs[left_index + 1:]:
            right_id = str(right_target["target_id"])
            stats = _gradient_vector_dot_for_neuron_layers(
                left_gradients=score_gradients_by_target[left_id],
                right_gradients=score_gradients_by_target[right_id],
                layers=neuron_layers,
            )
            rows.append(
                {
                    "target_a": left_id,
                    "target_b": right_id,
                    "score_gradient_dot": stats["dot"],
                    "target_a_score_gradient_l2_norm": stats["left_l2_norm"],
                    "target_b_score_gradient_l2_norm": stats["right_l2_norm"],
                    "score_gradient_cosine": stats["cosine"],
                }
            )
    return rows


def _summarize_pairwise_gradient_rows(pairwise_interval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairwise_interval_rows:
        grouped[(str(row["target_a"]), str(row["target_b"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (left_id, right_id), rows in sorted(grouped.items()):
        cosine_values = [
            _require_number(row["score_gradient_cosine"], "pairwise.score_gradient_cosine")
            for row in rows
            if row["score_gradient_cosine"] is not None
        ]
        summary_rows.append(
            {
                "target_a": left_id,
                "target_b": right_id,
                "interval_count": len(rows),
                "cosine_interval_count": len(cosine_values),
                "cosine_null_interval_count": len(rows) - len(cosine_values),
                "mean_score_gradient_cosine": None if not cosine_values else sum(cosine_values) / len(cosine_values),
                "min_score_gradient_cosine": None if not cosine_values else min(cosine_values),
                "max_score_gradient_cosine": None if not cosine_values else max(cosine_values),
            }
        )
    return summary_rows


def _target_neuron_score_lookup(neuron_rows: list[dict[str, Any]], target_ids: list[str]) -> dict[tuple[int, int], dict[str, float]]:
    lookup: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    target_set = set(target_ids)
    for row in neuron_rows:
        target_id = str(row["target_id"])
        if target_id not in target_set:
            continue
        key = (_require_int(row.get("layer"), "neuron_row.layer"), _require_int(row.get("neuron"), "neuron_row.neuron"))
        lookup[key][target_id] = _require_number(row.get("sum_score_linearized_delta"), "neuron_row.sum_score_linearized_delta")
    return lookup


def _coalition_category_summary(
    *,
    neuron_rows: list[dict[str, Any]],
    candidate_target_ids: list[str],
    sign_epsilon: float,
) -> dict[str, Any]:
    if sign_epsilon < 0.0:
        raise ValueError("sign_epsilon must be non-negative.")
    score_lookup = _target_neuron_score_lookup(neuron_rows, candidate_target_ids)
    category_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for (layer, neuron), scores in sorted(score_lookup.items()):
        missing = sorted(set(candidate_target_ids) - set(scores))
        if missing:
            raise KeyError(f"Neuron L{layer}N{neuron} is missing candidate scores for {missing}")
        positive_targets = [target_id for target_id, value in scores.items() if value > sign_epsilon]
        negative_targets = [target_id for target_id, value in scores.items() if value < -sign_epsilon]
        if len(positive_targets) == len(candidate_target_ids):
            category = "shared_positive"
        elif len(negative_targets) == len(candidate_target_ids):
            category = "shared_negative"
        elif len(positive_targets) == 1 and not negative_targets:
            category = f"specific_positive:{positive_targets[0]}"
        elif positive_targets and negative_targets:
            category = "conflict"
        else:
            category = "flat_or_mixed"
        positive_sum = sum(max(value, 0.0) for value in scores.values())
        negative_abs_sum = sum(max(-value, 0.0) for value in scores.values())
        conflict_magnitude = min(
            sum(max(value, 0.0) for value in scores.values()),
            sum(max(-value, 0.0) for value in scores.values()),
        )
        category_rows.append(
            {
                "layer": layer,
                "neuron": neuron,
                "category": category,
                "scores_by_candidate": scores,
                "positive_sum": positive_sum,
                "negative_abs_sum": negative_abs_sum,
                "conflict_magnitude": conflict_magnitude,
                "min_candidate_score": min(scores.values()),
                "max_candidate_score": max(scores.values()),
            }
        )
        summary = summaries.setdefault(
            category,
            {
                "category": category,
                "neuron_count": 0,
                "positive_score_sum": 0.0,
                "negative_score_abs_sum": 0.0,
                "conflict_magnitude_sum": 0.0,
            },
        )
        summary["neuron_count"] += 1
        summary["positive_score_sum"] += positive_sum
        summary["negative_score_abs_sum"] += negative_abs_sum
        summary["conflict_magnitude_sum"] += conflict_magnitude
    category_rows.sort(
        key=lambda row: (
            str(row["category"]) != "shared_positive",
            -abs(float(row["positive_sum"]) + float(row["negative_abs_sum"])),
            int(row["layer"]),
            int(row["neuron"]),
        )
    )
    summary_rows = sorted(
        summaries.values(),
        key=lambda row: abs(float(row["positive_score_sum"])) + abs(float(row["negative_score_abs_sum"])),
        reverse=True,
    )
    return {
        "candidate_target_ids": candidate_target_ids,
        "sign_epsilon": sign_epsilon,
        "category_summaries": summary_rows,
        "category_rows": category_rows,
    }


def _coalition_top_neuron_sets(
    *,
    neuron_rows: list[dict[str, Any]],
    candidate_target_ids: list[str],
    top_k: int,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k_neurons must be positive.")
    rows_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in neuron_rows:
        target_id = str(row["target_id"])
        if target_id in candidate_target_ids:
            rows_by_target[target_id].append(row)
    top_sets: dict[str, set[tuple[int, int]]] = {}
    top_rows_by_target: dict[str, list[dict[str, Any]]] = {}
    for target_id in candidate_target_ids:
        ranked = sorted(
            rows_by_target[target_id],
            key=lambda row: float(row["sum_score_linearized_delta"]),
            reverse=True,
        )[:top_k]
        top_rows_by_target[target_id] = ranked
        top_sets[target_id] = {
            (_require_int(row.get("layer"), "neuron_row.layer"), _require_int(row.get("neuron"), "neuron_row.neuron"))
            for row in ranked
        }
    overlap_rows: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(candidate_target_ids):
        for right_id in candidate_target_ids[left_index + 1:]:
            left_set = top_sets[left_id]
            right_set = top_sets[right_id]
            union = left_set | right_set
            overlap_rows.append(
                {
                    "target_a": left_id,
                    "target_b": right_id,
                    "top_k": top_k,
                    "intersection_count": len(left_set & right_set),
                    "union_count": len(union),
                    "jaccard": None if not union else len(left_set & right_set) / len(union),
                    "shared_neurons": [
                        {"layer": layer, "neuron": neuron}
                        for layer, neuron in sorted(left_set & right_set)
                    ],
                }
            )
    return {
        "top_rows_by_target": top_rows_by_target,
        "overlap_rows": overlap_rows,
    }


def _select_neurons_for_trajectory(
    *,
    coalition_summary: dict[str, Any],
    neuron_rows: list[dict[str, Any]],
    candidate_target_ids: list[str],
    top_k: int,
) -> list[dict[str, int]]:
    if top_k <= 0:
        raise ValueError("trajectory_top_k must be positive.")
    selected: list[tuple[int, int]] = []
    for row in _require_list(coalition_summary.get("category_rows"), "coalition_summary.category_rows"):
        category_row = _require_dict(row, "coalition_summary.category_rows[]")
        if category_row["category"] != "shared_positive":
            continue
        selected.append(
            (
                _require_int(category_row.get("layer"), "coalition_category.layer"),
                _require_int(category_row.get("neuron"), "coalition_category.neuron"),
            )
        )
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        score_lookup = _target_neuron_score_lookup(neuron_rows, candidate_target_ids)
        ranked = sorted(
            score_lookup,
            key=lambda key: max(abs(value) for value in score_lookup[key].values()),
            reverse=True,
        )
        for key in ranked:
            if key not in selected:
                selected.append(key)
            if len(selected) >= top_k:
                break
    return [{"layer": layer, "neuron": neuron} for layer, neuron in selected[:top_k]]


def _compute_neuron_activation_trajectory(
    *,
    model_loader: Any,
    batches: list[dict[str, Any]],
    steps: list[int],
    selected_neurons: list[dict[str, int]],
) -> list[dict[str, Any]]:
    if not selected_neurons:
        raise ValueError("selected_neurons must not be empty.")
    by_layer: dict[int, list[int]] = defaultdict(list)
    for item in selected_neurons:
        layer = _require_int(item.get("layer"), "selected_neuron.layer")
        neuron = _require_int(item.get("neuron"), "selected_neuron.neuron")
        if neuron not in by_layer[layer]:
            by_layer[layer].append(neuron)
    trajectory_rows: list[dict[str, Any]] = []
    for step in steps:
        model = model_loader(step)
        sums = {(layer, neuron): 0.0 for layer, neurons in by_layer.items() for neuron in neurons}
        counts = {(layer, neuron): 0 for layer, neurons in by_layer.items() for neuron in neurons}
        with torch.no_grad():
            for batch in batches:
                outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_mlp_states=True,
                )
                if outputs.mlp_states is None:
                    raise RuntimeError("Neuron activation trajectory requires mlp_states.")
                _, _, metadata = extract_answer_logits(outputs.logits, batch)
                rows = metadata["rows"]
                prediction_positions = metadata["prediction_positions"]
                for layer, neurons in by_layer.items():
                    state_key = f"layer_{layer}_hidden"
                    if state_key not in outputs.mlp_states:
                        raise KeyError(f"MLP state {state_key} not present in model output.")
                    selected = outputs.mlp_states[state_key][rows, prediction_positions, :]
                    for neuron in neurons:
                        values = selected[:, neuron].float()
                        sums[(layer, neuron)] += float(values.sum().item())
                        counts[(layer, neuron)] += int(values.numel())
        for (layer, neuron), total in sorted(sums.items()):
            count = counts[(layer, neuron)]
            if count <= 0:
                raise ValueError(f"No activation values collected for L{layer}N{neuron} at step {step}.")
            trajectory_rows.append(
                {
                    "step": int(step),
                    "layer": layer,
                    "neuron": neuron,
                    "mean_activation": total / count,
                    "num_values": count,
                }
            )
    return trajectory_rows


def _render_coalition_heatmap(
    *,
    neuron_rows: list[dict[str, Any]],
    target_ids: list[str],
    output_path: Path,
    top_k: int,
) -> Path:
    if not neuron_rows:
        raise ValueError("Cannot render coalition heatmap without neuron rows.")
    if top_k <= 0:
        raise ValueError("top_k_neurons must be positive.")
    score_lookup = _target_neuron_score_lookup(neuron_rows, target_ids)
    ranked_neurons = sorted(
        score_lookup,
        key=lambda key: max(abs(score_lookup[key].get(target_id, 0.0)) for target_id in target_ids),
        reverse=True,
    )[:top_k]
    if not ranked_neurons:
        raise ValueError("No ranked neurons available for coalition heatmap.")
    matrix = [
        [score_lookup[(layer, neuron)].get(target_id, 0.0) for target_id in target_ids]
        for layer, neuron in ranked_neurons
    ]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(target_ids)), max(6, 0.28 * len(ranked_neurons))))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_title("Neuron x candidate feature-score update drive")
    ax.set_xticks(list(range(len(target_ids))))
    ax.set_xticklabels(target_ids, rotation=35, ha="right")
    ax.set_yticks(list(range(len(ranked_neurons))))
    ax.set_yticklabels([f"L{layer}N{neuron}" for layer, neuron in ranked_neurons])
    fig.colorbar(image, ax=ax, label="sum score dot update")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_coalition_category_plot(
    *,
    coalition_summary: dict[str, Any],
    output_path: Path,
) -> Path:
    summaries = _require_list(coalition_summary.get("category_summaries"), "coalition_summary.category_summaries")
    if not summaries:
        raise ValueError("Cannot render coalition category plot without category summaries.")
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [str(row["category"]) for row in summaries]
    x_positions = list(range(len(labels)))
    ax.bar(
        [position - 0.18 for position in x_positions],
        [_require_number(row["positive_score_sum"], "category.positive_score_sum") for row in summaries],
        width=0.36,
        label="positive score drive",
    )
    ax.bar(
        [position + 0.18 for position in x_positions],
        [_require_number(row["negative_score_abs_sum"], "category.negative_score_abs_sum") for row in summaries],
        width=0.36,
        label="negative score drive abs",
    )
    ax.set_title("Shared vs specific neuron score-drive categories")
    ax.set_ylabel("summed score drive")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_coalition_gradient_conflict_matrix(
    *,
    target_ids: list[str],
    pairwise_summary_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not target_ids:
        raise ValueError("Cannot render gradient conflict matrix without targets.")
    index_by_id = {target_id: index for index, target_id in enumerate(target_ids)}
    matrix = [[1.0 if left == right else float("nan") for right in target_ids] for left in target_ids]
    for row in pairwise_summary_rows:
        left_id = str(row["target_a"])
        right_id = str(row["target_b"])
        value = row["mean_score_gradient_cosine"]
        plotted = float("nan") if value is None else float(value)
        left_index = index_by_id[left_id]
        right_index = index_by_id[right_id]
        matrix[left_index][right_index] = plotted
        matrix[right_index][left_index] = plotted
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_title("Feature-score gradient cosine on selected MLP-neuron parameters")
    ax.set_xticks(list(range(len(target_ids))))
    ax.set_xticklabels(target_ids, rotation=35, ha="right")
    ax.set_yticks(list(range(len(target_ids))))
    ax.set_yticklabels(target_ids)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            label = "n/a" if value != value else f"{value:.2f}"
            ax.text(column_index, row_index, label, ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(image, ax=ax, label="cosine")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_neuron_activation_trajectory_plot(
    *,
    trajectory_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not trajectory_rows:
        raise ValueError("Cannot render neuron trajectory plot without trajectory rows.")
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[
            (
                _require_int(row.get("layer"), "trajectory_row.layer"),
                _require_int(row.get("neuron"), "trajectory_row.neuron"),
            )
        ].append(row)
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 6))
    for (layer, neuron), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: int(item["step"]))
        ax.plot(
            [int(row["step"]) for row in ordered],
            [_require_number(row["mean_activation"], "trajectory_row.mean_activation") for row in ordered],
            linewidth=1.8,
            marker="o",
            label=f"L{layer}N{neuron}",
        )
    ax.set_title("Top coalition-neuron activation trajectories")
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("mean answer-position MLP activation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_coalition_plots(
    *,
    output_dir: Path,
    neuron_rows: list[dict[str, Any]],
    target_ids: list[str],
    candidate_target_ids: list[str],
    coalition_summary: dict[str, Any],
    pairwise_summary_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
    top_k_neurons: int,
) -> dict[str, Path]:
    return {
        "neuron_candidate_heatmap": _render_coalition_heatmap(
            neuron_rows=neuron_rows,
            target_ids=target_ids,
            output_path=output_dir / "candidate_coalition_neuron_heatmap.svg",
            top_k=top_k_neurons,
        ),
        "shared_specific": _render_coalition_category_plot(
            coalition_summary=coalition_summary,
            output_path=output_dir / "candidate_coalition_shared_specific.svg",
        ),
        "gradient_conflict_matrix": _render_coalition_gradient_conflict_matrix(
            target_ids=candidate_target_ids,
            pairwise_summary_rows=[
                row
                for row in pairwise_summary_rows
                if row["target_a"] in candidate_target_ids and row["target_b"] in candidate_target_ids
            ],
            output_path=output_dir / "candidate_coalition_gradient_conflict_matrix.svg",
        ),
        "neuron_activation_trajectory": _render_neuron_activation_trajectory_plot(
            trajectory_rows=trajectory_rows,
            output_path=output_dir / "candidate_coalition_neuron_trajectories.svg",
        ),
    }


def _write_coalition_markdown(
    *,
    output_path: Path,
    report_payload: dict[str, Any],
) -> Path:
    target_rows = _require_list(report_payload.get("target_summaries"), "coalition_report.target_summaries")
    coalition_summary = _require_dict(report_payload.get("coalition_summary"), "coalition_report.coalition_summary")
    category_summaries = _require_list(coalition_summary.get("category_summaries"), "coalition_summary.category_summaries")
    top_shared = [
        row
        for row in _require_list(coalition_summary.get("category_rows"), "coalition_summary.category_rows")
        if _require_dict(row, "coalition_summary.category_rows[]")["category"] == "shared_positive"
    ][:12]
    lines = [
        "# Candidate Coalition Map",
        "",
        "This report asks whether selected candidate families are supported by the same MLP neurons.",
        "",
        "For each selected checkpoint interval, it computes per-neuron projected update contributions:",
        "",
        "`Delta score_c,n ~= grad_theta_n score_c . Delta theta_n`",
        "",
        "where `theta_n` is the neuron-specific parameter slice: `fc_in` row, `fc_in` bias, and `fc_out` column.",
        "",
        "## Inputs",
        "",
        f"- Registry: `{report_payload['registry_path']}`",
        f"- Gradient link: `{report_payload['gradient_link_path']}`",
        f"- Checkpoint dir: `{report_payload['checkpoint_dir']}`",
        f"- Selected candidates: {', '.join(report_payload['selected_candidate_ids'])}",
        f"- Neuron layers: {', '.join(str(item) for item in report_payload['selection']['neuron_layers'])}",
        f"- Intervals: {report_payload['interval_count']}",
        "",
        "## Target Summary",
        "",
        "| target | kind | features | top positive neuron | score drive | loss reduction |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for raw_target in target_rows:
        target = _require_dict(raw_target, "coalition_report.target_summaries[]")
        top_positive = target.get("top_positive_neuron")
        top_label = "n/a" if top_positive is None else f"L{top_positive['layer']}N{top_positive['neuron']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(target["target_id"]),
                    str(target["target_kind"]),
                    ",".join(str(item) for item in target["feature_ids"]),
                    top_label,
                    _markdown_number(target["sum_score_linearized_delta"]),
                    _markdown_number(target["sum_loss_reduction_linearized"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Shared vs Specific Categories", ""])
    lines.append("| category | neurons | positive score | negative score abs | conflict magnitude |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for raw_category in category_summaries:
        category = _require_dict(raw_category, "coalition_summary.category_summaries[]")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(category["category"]),
                    str(category["neuron_count"]),
                    _markdown_number(category["positive_score_sum"]),
                    _markdown_number(category["negative_score_abs_sum"]),
                    _markdown_number(category["conflict_magnitude_sum"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Top Shared Positive Neurons", ""])
    if top_shared:
        lines.append("| neuron | scores by candidate |")
        lines.append("| --- | --- |")
        for raw_row in top_shared:
            row = _require_dict(raw_row, "coalition_summary.category_rows[]")
            scores = _require_dict(row.get("scores_by_candidate"), "coalition category scores_by_candidate")
            score_text = ", ".join(f"{target}: {_markdown_number(value)}" for target, value in sorted(scores.items()))
            lines.append(f"| L{row['layer']}N{row['neuron']} | {score_text} |")
    else:
        lines.append("No shared-positive neurons were found under the current sign threshold.")
    lines.extend(["", "## Pairwise Candidate Gradient Cosines", ""])
    pairwise_rows = _require_list(report_payload.get("pairwise_gradient_summary"), "coalition_report.pairwise_gradient_summary")
    lines.append("| target A | target B | mean cosine | intervals | null intervals |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for raw_pair in pairwise_rows:
        pair = _require_dict(raw_pair, "coalition_report.pairwise_gradient_summary[]")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(pair["target_a"]),
                    str(pair["target_b"]),
                    _markdown_number(pair["mean_score_gradient_cosine"]),
                    str(pair["cosine_interval_count"]),
                    str(pair["cosine_null_interval_count"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Unsupported Claims", ""])
    for claim in _require_list(report_payload.get("unsupported_claims"), "coalition_report.unsupported_claims"):
        lines.append(f"- {claim}")
    lines.extend(["", "## Plots", ""])
    for name, plot_path in _require_dict(report_payload.get("plots"), "coalition_report.plots").items():
        lines.append(f"- {name}: `{plot_path}`")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_candidate_coalition_map(
    *,
    config_path: Path,
    probe_set_path: Path,
    registry_path: Path,
    gradient_link_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    candidate_ids: list[str] | None = None,
    device_name: str = "cpu",
    start_step: int | None = None,
    end_step: int | None = None,
    neuron_layers: list[int] | None = None,
    include_individual_features: bool = True,
    top_k_neurons: int = 24,
    trajectory_top_k: int = 8,
    sign_epsilon: float = 0.0,
) -> tuple[Path, Path, dict[str, Path]]:
    if top_k_neurons <= 0:
        raise ValueError("top_k_neurons must be positive.")
    if trajectory_top_k <= 0:
        raise ValueError("trajectory_top_k must be positive.")
    spec = TrainSpec.from_path(config_path)
    device = require_device(device_name)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    batches = _load_probe_batches(spec=spec, probe_set_path=probe_set_path, vocab=vocab, device=device)
    registry = _load_candidate_registry(registry_path)
    gradient_payload = _load_gradient_link_payload(gradient_link_path)
    registry_lookup = _candidate_lookup_from_registry(registry)
    interval_rows = [
        _require_dict(row, f"gradient_link.interval_rows[{index}]")
        for index, row in enumerate(_require_list(gradient_payload.get("interval_rows"), "gradient_link.interval_rows"))
    ]
    rows_by_candidate = _group_rows_by_candidate(interval_rows)
    selected_candidate_ids = _select_birth_model_candidate_ids(
        registry_lookup=registry_lookup,
        rows_by_candidate=rows_by_candidate,
        candidate_ids=candidate_ids,
    )
    selected_neuron_layers = _coalition_neuron_layers(
        registry_lookup=registry_lookup,
        selected_candidate_ids=selected_candidate_ids,
        extra_layers=neuron_layers,
    )
    target_specs = _coalition_target_specs(
        registry_lookup=registry_lookup,
        selected_candidate_ids=selected_candidate_ids,
        include_individual_features=include_individual_features,
    )
    target_ids = [str(target["target_id"]) for target in target_specs]
    candidate_target_ids = [
        str(target["target_id"])
        for target in target_specs
        if str(target["target_kind"]) == "candidate"
    ]
    if len(candidate_target_ids) < 2:
        raise ValueError("candidate-coalition-map requires at least two selected candidate targets.")
    interval_pairs = _common_interval_pairs_for_candidates(
        rows_by_candidate=rows_by_candidate,
        selected_candidate_ids=selected_candidate_ids,
        start_step=start_step,
        end_step=end_step,
    )
    checkpoint_paths_by_step = _resolve_checkpoint_paths_by_step(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_cache: dict[int, dict[str, Any]] = {}
    loss_gradient_cache: dict[int, dict[str, Any]] = {}
    model_cache: dict[int, torch.nn.Module] = {}
    basis_cache: dict[str, dict[str, Any]] = {}
    feature_score_cache: dict[tuple[str, int], dict[str, Any]] = {}

    def load_state(step: int) -> dict[str, Any]:
        cached = state_cache.get(step)
        if cached is not None:
            return cached
        if step not in checkpoint_paths_by_step:
            raise FileNotFoundError(f"Checkpoint for step {step} not found in {checkpoint_dir}.")
        payload = _load_checkpoint_state_for_analysis(checkpoint_paths_by_step[step])
        if int(payload["step"]) != step:
            raise ValueError(f"Checkpoint payload step mismatch for {checkpoint_paths_by_step[step]}.")
        state_cache[step] = payload
        return payload

    def load_model_at_step(step: int) -> torch.nn.Module:
        cached = model_cache.get(step)
        if cached is not None:
            return cached
        if step not in checkpoint_paths_by_step:
            raise FileNotFoundError(f"Checkpoint for step {step} not found in {checkpoint_dir}.")
        checkpoint = load_checkpoint(checkpoint_paths_by_step[step], device)
        model = build_model(spec.model, len(vocab.tokens), device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        model_cache[step] = model
        return model

    def loss_gradients_at_step(step: int) -> dict[str, Any]:
        cached = loss_gradient_cache.get(step)
        if cached is not None:
            return cached
        model = load_model_at_step(step)
        payload = _compute_probe_loss_and_gradients(model=model, batches=batches, pad_token_id=vocab.pad_token_id)
        loss_gradient_cache[step] = payload
        return payload

    neuron_accumulator: dict[tuple[str, int, int], dict[str, Any]] = {}
    pairwise_interval_rows: list[dict[str, Any]] = []
    for previous_step, current_step in interval_pairs:
        previous_payload = load_state(previous_step)
        current_payload = load_state(current_step)
        previous_state = _require_dict(previous_payload["model_state"], "previous_payload.model_state")
        current_state = _require_dict(current_payload["model_state"], "current_payload.model_state")
        loss_gradient_payload = loss_gradients_at_step(previous_step)
        loss_gradients = _require_dict(loss_gradient_payload["gradients"], "loss_gradient_payload.gradients")
        _validate_state_and_gradient_keys(
            previous_state=previous_state,
            current_state=current_state,
            gradients=loss_gradients,
        )
        score_gradients_by_target: dict[str, dict[str, torch.Tensor]] = {}
        for target in target_specs:
            target_id = str(target["target_id"])
            feature_score_payload = _target_feature_score_at_step(
                target=target,
                step=previous_step,
                basis_cache=basis_cache,
                feature_score_cache=feature_score_cache,
                model_loader=load_model_at_step,
                batches=batches,
                device=device,
            )
            score_gradients = _require_dict(feature_score_payload["gradients"], f"feature_score_payload[{target_id}].gradients")
            _validate_state_and_gradient_keys(
                previous_state=previous_state,
                current_state=current_state,
                gradients=score_gradients,
            )
            score_gradients_by_target[target_id] = score_gradients
            for layer in selected_neuron_layers:
                score_arrays = _per_neuron_projection_arrays(
                    previous_state=previous_state,
                    current_state=current_state,
                    gradients=score_gradients,
                    layer=layer,
                )
                loss_arrays = _per_neuron_projection_arrays(
                    previous_state=previous_state,
                    current_state=current_state,
                    gradients=loss_gradients,
                    layer=layer,
                )
                _add_neuron_projection_accumulators(
                    accumulator=neuron_accumulator,
                    target=target,
                    layer=layer,
                    score_arrays=score_arrays,
                    loss_arrays=loss_arrays,
                )
        pairwise_rows = _coalition_pairwise_gradient_rows(
            target_specs=target_specs,
            score_gradients_by_target=score_gradients_by_target,
            neuron_layers=selected_neuron_layers,
        )
        for row in pairwise_rows:
            pairwise_interval_rows.append(
                {
                    **row,
                    "previous_step": previous_step,
                    "step": current_step,
                }
            )

    neuron_rows = _finalize_neuron_projection_rows(neuron_accumulator)
    coalition_summary = _coalition_category_summary(
        neuron_rows=neuron_rows,
        candidate_target_ids=candidate_target_ids,
        sign_epsilon=sign_epsilon,
    )
    top_sets = _coalition_top_neuron_sets(
        neuron_rows=neuron_rows,
        candidate_target_ids=candidate_target_ids,
        top_k=top_k_neurons,
    )
    pairwise_summary_rows = _summarize_pairwise_gradient_rows(pairwise_interval_rows)
    selected_trajectory_neurons = _select_neurons_for_trajectory(
        coalition_summary=coalition_summary,
        neuron_rows=neuron_rows,
        candidate_target_ids=candidate_target_ids,
        top_k=trajectory_top_k,
    )
    trajectory_steps = sorted({interval_pairs[0][0]} | {current_step for _, current_step in interval_pairs})
    trajectory_rows = _compute_neuron_activation_trajectory(
        model_loader=load_model_at_step,
        batches=batches,
        steps=trajectory_steps,
        selected_neurons=selected_trajectory_neurons,
    )

    target_summaries: list[dict[str, Any]] = []
    for target in target_specs:
        target_id = str(target["target_id"])
        rows = [row for row in neuron_rows if str(row["target_id"]) == target_id]
        if not rows:
            raise ValueError(f"No neuron rows were computed for target {target_id}.")
        positive_rows = [row for row in rows if float(row["sum_score_linearized_delta"]) > sign_epsilon]
        top_positive = None
        if positive_rows:
            best = max(positive_rows, key=lambda row: float(row["sum_score_linearized_delta"]))
            top_positive = {
                "layer": int(best["layer"]),
                "neuron": int(best["neuron"]),
                "sum_score_linearized_delta": float(best["sum_score_linearized_delta"]),
            }
        target_summaries.append(
            {
                "target_id": target_id,
                "target_kind": target["target_kind"],
                "candidate_id": target.get("candidate_id"),
                "candidate_ids": target.get("candidate_ids"),
                "family_id": target.get("family_id"),
                "family_ids": target.get("family_ids"),
                "stage_name": target["stage_name"],
                "feature_ids": target["feature_ids"],
                "neuron_count": len(rows),
                "sum_score_linearized_delta": sum(float(row["sum_score_linearized_delta"]) for row in rows),
                "positive_score_drive": sum(float(row["positive_score_drive"]) for row in rows),
                "negative_score_drive_abs": sum(float(row["negative_score_drive_abs"]) for row in rows),
                "sum_loss_reduction_linearized": sum(float(row["sum_loss_reduction_linearized"]) for row in rows),
                "top_positive_neuron": top_positive,
            }
        )
    target_summaries.sort(key=lambda row: abs(float(row["sum_score_linearized_delta"])), reverse=True)

    plot_paths = _render_coalition_plots(
        output_dir=output_dir,
        neuron_rows=neuron_rows,
        target_ids=target_ids,
        candidate_target_ids=candidate_target_ids,
        coalition_summary=coalition_summary,
        pairwise_summary_rows=pairwise_summary_rows,
        trajectory_rows=trajectory_rows,
        top_k_neurons=top_k_neurons,
    )
    unsupported_claims = [
        "complete_dense_circuit_decomposition",
        "causal_necessity_of_shared_neurons",
        "cross_seed_coalition_stability",
        "per_minibatch_neuron_update_trace",
    ]
    report_path = output_dir / "candidate_coalition_map_report.json"
    markdown_path = output_dir / "candidate_coalition_map_report.md"
    report_payload = {
        "schema_version": COALITION_MAP_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "registry_path": str(registry_path),
        "gradient_link_path": str(gradient_link_path),
        "checkpoint_dir": str(checkpoint_dir),
        "device": device_name,
        "selected_candidate_ids": selected_candidate_ids,
        "target_specs": target_specs,
        "selection": {
            "candidate_ids": candidate_ids,
            "start_step": start_step,
            "end_step": end_step,
            "neuron_layers": selected_neuron_layers,
            "include_individual_features": include_individual_features,
            "top_k_neurons": top_k_neurons,
            "trajectory_top_k": trajectory_top_k,
            "sign_epsilon": sign_epsilon,
        },
        "interval_count": len(interval_pairs),
        "interval_pairs": [{"previous_step": previous, "step": current} for previous, current in interval_pairs],
        "target_summaries": target_summaries,
        "neuron_rows": neuron_rows,
        "coalition_summary": coalition_summary,
        "top_neuron_sets": top_sets,
        "pairwise_gradient_summary": pairwise_summary_rows,
        "pairwise_gradient_interval_rows": pairwise_interval_rows,
        "selected_trajectory_neurons": selected_trajectory_neurons,
        "neuron_activation_trajectory_rows": trajectory_rows,
        "unsupported_claims": unsupported_claims,
        "plots": {name: str(path) for name, path in plot_paths.items()},
    }
    write_json(report_path, report_payload)
    _write_coalition_markdown(output_path=markdown_path, report_payload=report_payload)
    return report_path, markdown_path, plot_paths


def _load_coalition_map_payload(coalition_map_path: Path) -> dict[str, Any]:
    payload = _load_json_dict(coalition_map_path, "candidate coalition map")
    schema_version = payload.get("schema_version")
    if schema_version != COALITION_MAP_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported coalition-map schema_version {schema_version}; expected {COALITION_MAP_SCHEMA_VERSION}."
        )
    _require_list(payload.get("target_specs"), "coalition_map.target_specs")
    coalition_summary = _require_dict(payload.get("coalition_summary"), "coalition_map.coalition_summary")
    _require_list(coalition_summary.get("category_rows"), "coalition_map.coalition_summary.category_rows")
    _require_dict(payload.get("top_neuron_sets"), "coalition_map.top_neuron_sets")
    return payload


def _neuron_key_from_row(row: dict[str, Any], label: str) -> tuple[int, int]:
    return (
        _require_int(row.get("layer"), f"{label}.layer"),
        _require_int(row.get("neuron"), f"{label}.neuron"),
    )


def _candidate_intervention_targets(
    *,
    coalition_payload: dict[str, Any],
    score_individual_features: bool,
) -> list[dict[str, Any]]:
    raw_targets = _require_list(coalition_payload.get("target_specs"), "coalition_map.target_specs")
    targets: list[dict[str, Any]] = []
    for index, raw_target in enumerate(raw_targets):
        target = _require_dict(raw_target, f"coalition_map.target_specs[{index}]")
        target_kind = _require_non_empty_str(target.get("target_kind"), f"coalition_map.target_specs[{index}].target_kind")
        if target_kind != "candidate" and not score_individual_features:
            continue
        targets.append(target)
    if not targets:
        raise ValueError("No feature-score targets selected for candidate-neuron-intervention.")
    return targets


def _coalition_candidate_target_ids(coalition_payload: dict[str, Any]) -> list[str]:
    coalition_summary = _require_dict(coalition_payload.get("coalition_summary"), "coalition_map.coalition_summary")
    candidate_target_ids = [
        _sanitize_candidate_id(_require_non_empty_str(item, "coalition_map.coalition_summary.candidate_target_ids[]"))
        for item in _require_list(coalition_summary.get("candidate_target_ids"), "coalition_map.coalition_summary.candidate_target_ids")
    ]
    if len(candidate_target_ids) < 2:
        raise ValueError("candidate-neuron-intervention requires at least two coalition candidate targets.")
    return candidate_target_ids


def _category_lookup_by_neuron(coalition_payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    coalition_summary = _require_dict(coalition_payload.get("coalition_summary"), "coalition_map.coalition_summary")
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for index, raw_row in enumerate(_require_list(coalition_summary.get("category_rows"), "coalition_summary.category_rows")):
        row = _require_dict(raw_row, f"coalition_summary.category_rows[{index}]")
        key = _neuron_key_from_row(row, f"coalition_summary.category_rows[{index}]")
        if key in lookup:
            raise ValueError(f"Duplicate coalition category row for L{key[0]}N{key[1]}.")
        lookup[key] = row
    if not lookup:
        raise ValueError("Coalition map contains no category rows.")
    return lookup


def _intervention_neuron_record(
    *,
    layer: int,
    neuron: int,
    category_lookup: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, Any]:
    key = (layer, neuron)
    if key not in category_lookup:
        raise KeyError(f"Coalition category row not found for L{layer}N{neuron}.")
    category_row = category_lookup[key]
    return {
        "layer": layer,
        "neuron": neuron,
        "label": f"L{layer}N{neuron}",
        "category": category_row["category"],
        "scores_by_candidate": category_row["scores_by_candidate"],
        "positive_sum": category_row["positive_sum"],
        "negative_abs_sum": category_row["negative_abs_sum"],
        "conflict_magnitude": category_row["conflict_magnitude"],
    }


def _make_intervention_set_row(
    *,
    neuron_set_id: str,
    set_kind: str,
    source: str,
    selected_keys: list[tuple[int, int]],
    category_lookup: dict[tuple[int, int], dict[str, Any]],
    candidate_target_ids: list[str],
    selection_metric: str,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    seen: set[tuple[int, int]] = set()
    unique_keys: list[tuple[int, int]] = []
    for key in selected_keys:
        if key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)
    if not unique_keys:
        raise ValueError(f"Cannot build empty neuron intervention set: {neuron_set_id}")
    neurons = [
        _intervention_neuron_record(layer=layer, neuron=neuron, category_lookup=category_lookup)
        for layer, neuron in unique_keys
    ]
    candidate_score_sums = {candidate_id: 0.0 for candidate_id in candidate_target_ids}
    for neuron_row in neurons:
        scores = _require_dict(neuron_row.get("scores_by_candidate"), f"{neuron_set_id}.scores_by_candidate")
        for target_id in candidate_target_ids:
            if target_id not in scores:
                raise KeyError(f"Neuron {neuron_row['label']} is missing score for candidate {target_id}.")
            candidate_score_sums[target_id] += _require_number(scores[target_id], f"{neuron_set_id}.{target_id}.score")
    return {
        "neuron_set_id": _sanitize_candidate_id(neuron_set_id),
        "set_kind": set_kind,
        "candidate_id": candidate_id,
        "source": source,
        "selection_metric": selection_metric,
        "neuron_count": len(neurons),
        "neurons": neurons,
        "category_score_sums": {
            "positive_sum": sum(_require_number(row["positive_sum"], f"{neuron_set_id}.positive_sum") for row in neurons),
            "negative_abs_sum": sum(_require_number(row["negative_abs_sum"], f"{neuron_set_id}.negative_abs_sum") for row in neurons),
            "conflict_magnitude": sum(_require_number(row["conflict_magnitude"], f"{neuron_set_id}.conflict_magnitude") for row in neurons),
        },
        "candidate_score_sums": candidate_score_sums,
    }


def _build_candidate_intervention_sets(
    *,
    coalition_payload: dict[str, Any],
    top_k_per_set: int,
) -> dict[str, Any]:
    if top_k_per_set <= 0:
        raise ValueError("top_k_per_set must be positive.")
    candidate_target_ids = _coalition_candidate_target_ids(coalition_payload)
    category_lookup = _category_lookup_by_neuron(coalition_payload)
    category_rows = list(category_lookup.values())
    intervention_sets: list[dict[str, Any]] = []
    omitted_empty_set_ids: list[str] = []

    def add_category_set(neuron_set_id: str, category: str, metric_name: str) -> None:
        ranked = sorted(
            [row for row in category_rows if str(row["category"]) == category],
            key=lambda row: _require_number(row.get(metric_name), f"{category}.{metric_name}"),
            reverse=True,
        )
        if not ranked:
            omitted_empty_set_ids.append(neuron_set_id)
            return
        intervention_sets.append(
            _make_intervention_set_row(
                neuron_set_id=neuron_set_id,
                set_kind=category,
                source="coalition_category",
                selected_keys=[_neuron_key_from_row(row, f"{category}.row") for row in ranked[:top_k_per_set]],
                category_lookup=category_lookup,
                candidate_target_ids=candidate_target_ids,
                selection_metric=metric_name,
            )
        )

    add_category_set("shared_positive", "shared_positive", "positive_sum")
    add_category_set("conflict", "conflict", "conflict_magnitude")
    add_category_set("shared_negative", "shared_negative", "negative_abs_sum")

    top_neuron_sets = _require_dict(coalition_payload.get("top_neuron_sets"), "coalition_map.top_neuron_sets")
    top_rows_by_target = _require_dict(top_neuron_sets.get("top_rows_by_target"), "coalition_map.top_neuron_sets.top_rows_by_target")
    top_keys_by_target: dict[str, list[tuple[int, int]]] = {}
    for target_id in candidate_target_ids:
        rows = [
            _require_dict(row, f"top_rows_by_target.{target_id}[]")
            for row in _require_list(top_rows_by_target.get(target_id), f"top_rows_by_target.{target_id}")
        ]
        if not rows:
            raise ValueError(f"Coalition map has no top neuron rows for candidate target {target_id}.")
        top_keys_by_target[target_id] = [_neuron_key_from_row(row, f"top_rows_by_target.{target_id}[]") for row in rows]

    shared_top_keys = set(top_keys_by_target[candidate_target_ids[0]])
    for target_id in candidate_target_ids[1:]:
        shared_top_keys &= set(top_keys_by_target[target_id])
    if shared_top_keys:
        ranked_shared_top = sorted(
            shared_top_keys,
            key=lambda key: _require_number(category_lookup[key].get("positive_sum"), "top_overlap.positive_sum"),
            reverse=True,
        )
        intervention_sets.append(
            _make_intervention_set_row(
                neuron_set_id="top_overlap",
                set_kind="top_overlap",
                source="top_neuron_set_intersection",
                selected_keys=ranked_shared_top[:top_k_per_set],
                category_lookup=category_lookup,
                candidate_target_ids=candidate_target_ids,
                selection_metric="positive_sum",
            )
        )
    else:
        omitted_empty_set_ids.append("top_overlap")

    for target_id in candidate_target_ids:
        other_keys: set[tuple[int, int]] = set()
        for other_target_id, keys in top_keys_by_target.items():
            if other_target_id != target_id:
                other_keys.update(keys)
        specific_keys = [key for key in top_keys_by_target[target_id] if key not in other_keys]
        if not specific_keys:
            omitted_empty_set_ids.append(f"candidate_specific:{target_id}")
            continue
        ranked_specific = sorted(
            specific_keys,
            key=lambda key: _require_number(
                category_lookup[key]["scores_by_candidate"][target_id],
                f"candidate_specific:{target_id}.score",
            ),
            reverse=True,
        )
        intervention_sets.append(
            _make_intervention_set_row(
                neuron_set_id=f"candidate_specific:{target_id}",
                set_kind="candidate_specific_top",
                source="candidate_top_neurons_minus_other_candidate_top_neurons",
                selected_keys=ranked_specific[:top_k_per_set],
                category_lookup=category_lookup,
                candidate_target_ids=candidate_target_ids,
                selection_metric=f"scores_by_candidate.{target_id}",
                candidate_id=target_id,
            )
        )

    if not intervention_sets:
        raise ValueError("No non-empty neuron intervention sets could be built from the coalition map.")
    return {
        "candidate_target_ids": candidate_target_ids,
        "intervention_sets": intervention_sets,
        "omitted_empty_set_ids": omitted_empty_set_ids,
    }


def _neuron_mask_for_records(
    *,
    spec: TrainSpec,
    neurons: list[dict[str, Any]],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    if not neurons:
        raise ValueError("neurons must not be empty.")
    neurons_by_layer: dict[int, set[int]] = defaultdict(set)
    for index, raw_neuron in enumerate(neurons):
        neuron = _require_dict(raw_neuron, f"intervention_set.neurons[{index}]")
        layer = _require_int(neuron.get("layer"), f"intervention_set.neurons[{index}].layer")
        neuron_id = _require_int(neuron.get("neuron"), f"intervention_set.neurons[{index}].neuron")
        if layer < 0 or layer >= spec.model.n_layers:
            raise ValueError(f"Layer {layer} is out of range for n_layers={spec.model.n_layers}.")
        if neuron_id < 0 or neuron_id >= spec.model.d_ff:
            raise ValueError(f"Neuron {neuron_id} is out of range for d_ff={spec.model.d_ff}.")
        neurons_by_layer[layer].add(neuron_id)
    masks: dict[int, torch.Tensor] = {}
    for layer, neuron_ids in sorted(neurons_by_layer.items()):
        mask = torch.ones(spec.model.d_ff, device=device)
        index = torch.tensor(sorted(neuron_ids), device=device, dtype=torch.long)
        mask.index_fill_(0, index, 0.0)
        masks[layer] = mask
    return masks


@torch.no_grad()
def _evaluate_intervention_probe_metrics(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    pad_token_id: int,
    neuron_mask: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_token_correct = 0.0
    total_tokens = 0
    answer_correct = 0
    answer_total = 0
    split_correct: dict[str, int] = defaultdict(int)
    split_total: dict[str, int] = defaultdict(int)
    for batch in batches:
        token_count = int(batch["attention_mask"][:, 1:].sum().item())
        if token_count <= 0:
            raise ValueError("Probe batch has no non-padding next-token targets.")
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            neuron_mask=neuron_mask,
        )
        loss, token_accuracy = compute_lm_loss(
            logits=outputs.logits,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pad_token_id=pad_token_id,
        )
        total_loss += float(loss.item()) * token_count
        total_token_correct += float(token_accuracy.detach().cpu().item()) * token_count
        total_tokens += token_count

        answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
        predictions = answer_logits.argmax(dim=-1)
        matches = predictions == answer_targets
        answer_correct += int(matches.sum().item())
        answer_total += int(answer_targets.numel())
        for answer_index, row_index in enumerate(metadata["rows"].tolist()):
            split_name = str(batch["records"][row_index]["split"])
            split_total[split_name] += 1
            if bool(matches[answer_index].item()):
                split_correct[split_name] += 1
    if total_tokens <= 0:
        raise ValueError("Probe set has no non-padding next-token targets.")
    if answer_total <= 0:
        raise ValueError("Probe set has no answer targets.")
    split_accuracy = {
        split_name: split_correct[split_name] / split_total[split_name]
        for split_name in sorted(split_total)
    }
    return {
        "loss": total_loss / total_tokens,
        "token_accuracy": total_token_correct / total_tokens,
        "answer_accuracy": answer_correct / answer_total,
        "heldout_answer_accuracy": split_accuracy.get("heldout_pairs", 0.0),
        "structural_ood_answer_accuracy": split_accuracy.get("structural_ood", 0.0),
        "split_answer_accuracy": split_accuracy,
        "num_tokens": total_tokens,
        "num_answers": answer_total,
        "num_batches": len(batches),
    }


@torch.no_grad()
def _compute_feature_score_without_gradients(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    basis: dict[str, Any],
    stage_name: str,
    feature_ids: list[int],
    neuron_mask: dict[int, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if str(basis["stage_name"]) != stage_name:
        raise ValueError(f"Basis stage {basis['stage_name']} does not match target stage {stage_name}.")
    if not feature_ids:
        raise ValueError("feature_ids must not be empty.")
    num_features = int(basis["num_features"])
    invalid_feature_ids = [feature_id for feature_id in feature_ids if feature_id < 0 or feature_id >= num_features]
    if invalid_feature_ids:
        raise ValueError(f"Feature ids out of range for basis with {num_features} features: {invalid_feature_ids}")
    sae = basis["sae"]
    normalization_mean = basis["normalization_mean"]
    normalization_std = basis["normalization_std"]
    feature_index = torch.tensor(sorted(feature_ids), device=normalization_mean.device, dtype=torch.long)

    model.eval()
    total_score = 0.0
    total_feature_values = 0
    total_answers = 0
    for batch in batches:
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_residual_streams=True,
            neuron_mask=neuron_mask,
        )
        if outputs.residual_streams is None:
            raise RuntimeError("Feature-score intervention requires residual streams.")
        if stage_name not in outputs.residual_streams:
            raise KeyError(f"Residual stage {stage_name} not found in model outputs.")
        _, _, metadata = extract_answer_logits(outputs.logits, batch)
        rows = metadata["rows"]
        prediction_positions = metadata["prediction_positions"]
        selected_stage = outputs.residual_streams[stage_name][rows, prediction_positions, :]
        normalized = _normalize_activations(selected_stage, normalization_mean, normalization_std)
        features = torch.relu(sae.encoder(normalized))
        selected_features = features.index_select(1, feature_index)
        total_score += float(selected_features.sum().item())
        total_feature_values += int(selected_features.numel())
        total_answers += int(selected_features.size(0))
    if total_feature_values <= 0:
        raise ValueError("Feature-score intervention had no feature activations to score.")
    return {
        "score": total_score / float(total_feature_values),
        "num_answers": total_answers,
        "num_feature_values": total_feature_values,
    }


def _feature_scores_for_intervention_targets(
    *,
    model: torch.nn.Module,
    batches: list[dict[str, Any]],
    target_specs: list[dict[str, Any]],
    basis_cache: dict[str, dict[str, Any]],
    device: torch.device,
    neuron_mask: dict[int, torch.Tensor] | None = None,
) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for index, raw_target in enumerate(target_specs):
        target = _require_dict(raw_target, f"intervention.target_specs[{index}]")
        target_id = _sanitize_candidate_id(
            _require_non_empty_str(target.get("target_id"), f"intervention.target_specs[{index}].target_id")
        )
        basis_path = Path(_require_non_empty_str(target.get("basis_path"), f"target {target_id}.basis_path"))
        basis_key = str(basis_path)
        basis = basis_cache.get(basis_key)
        if basis is None:
            basis = _load_shared_basis(basis_path, device)
            basis_cache[basis_key] = basis
        scores[target_id] = _compute_feature_score_without_gradients(
            model=model,
            batches=batches,
            basis=basis,
            stage_name=_require_non_empty_str(target.get("stage_name"), f"target {target_id}.stage_name"),
            feature_ids=_coerce_int_list(target.get("feature_ids"), f"target {target_id}.feature_ids"),
            neuron_mask=neuron_mask,
        )
    return scores


def _target_score_rows(
    *,
    baseline_scores: dict[str, dict[str, Any]],
    intervened_scores: dict[str, dict[str, Any]],
    candidate_target_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_score_drops: list[float] = []
    for target_id in sorted(baseline_scores):
        if target_id not in intervened_scores:
            raise KeyError(f"Intervened feature scores are missing target {target_id}.")
        baseline_score = _require_number(baseline_scores[target_id]["score"], f"{target_id}.baseline_score")
        intervened_score = _require_number(intervened_scores[target_id]["score"], f"{target_id}.intervened_score")
        score_delta = intervened_score - baseline_score
        score_drop = baseline_score - intervened_score
        is_candidate_target = target_id in candidate_target_ids
        if is_candidate_target:
            candidate_score_drops.append(score_drop)
        rows.append(
            {
                "target_id": target_id,
                "is_candidate_target": is_candidate_target,
                "baseline_score": baseline_score,
                "intervened_score": intervened_score,
                "score_delta": score_delta,
                "score_drop": score_drop,
                "num_answers": intervened_scores[target_id]["num_answers"],
                "num_feature_values": intervened_scores[target_id]["num_feature_values"],
            }
        )
    if not candidate_score_drops:
        raise ValueError("No candidate target score drops were computed.")
    return rows, {
        "candidate_target_count": len(candidate_score_drops),
        "candidate_score_drop_sum": sum(candidate_score_drops),
        "candidate_score_drop_mean": sum(candidate_score_drops) / len(candidate_score_drops),
        "candidate_score_drop_min": min(candidate_score_drops),
        "candidate_score_drop_max": max(candidate_score_drops),
        "all_candidate_scores_drop": all(value > 0.0 for value in candidate_score_drops),
        "any_candidate_score_increases": any(value < 0.0 for value in candidate_score_drops),
    }


def _intervention_metric_deltas(
    *,
    baseline: dict[str, Any],
    intervened: dict[str, Any],
) -> dict[str, Any]:
    keys = [
        "loss",
        "token_accuracy",
        "answer_accuracy",
        "heldout_answer_accuracy",
        "structural_ood_answer_accuracy",
    ]
    deltas = {
        f"{key}_delta": _require_number(intervened[key], f"intervened.{key}") - _require_number(baseline[key], f"baseline.{key}")
        for key in keys
    }
    drops = {
        f"{key}_drop": _require_number(baseline[key], f"baseline.{key}") - _require_number(intervened[key], f"intervened.{key}")
        for key in keys
        if key != "loss"
    }
    drops["loss_increase"] = _require_number(intervened["loss"], "intervened.loss") - _require_number(baseline["loss"], "baseline.loss")
    return {
        "deltas": deltas,
        "drops": drops,
    }


def _render_neuron_intervention_behavior_plot(
    *,
    intervention_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not intervention_rows:
        raise ValueError("Cannot render behavior plot without intervention rows.")
    labels = [str(row["neuron_set_id"]) for row in intervention_rows]
    metrics = ["answer_accuracy_drop", "heldout_answer_accuracy_drop", "structural_ood_answer_accuracy_drop"]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(labels)), 6))
    x_positions = list(range(len(labels)))
    width = 0.24
    for metric_index, metric_name in enumerate(metrics):
        offset = (metric_index - 1) * width
        values = [
            _require_number(row["behavior_effect"]["drops"][metric_name], f"{row['neuron_set_id']}.{metric_name}")
            for row in intervention_rows
        ]
        ax.bar([position + offset for position in x_positions], values, width=width, label=metric_name)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Neuron-set ablation behavior effect")
    ax.set_ylabel("baseline - ablated accuracy")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_neuron_intervention_score_plot(
    *,
    intervention_rows: list[dict[str, Any]],
    target_ids: list[str],
    output_path: Path,
) -> Path:
    if not intervention_rows:
        raise ValueError("Cannot render score plot without intervention rows.")
    if not target_ids:
        raise ValueError("Cannot render score plot without target ids.")
    matrix: list[list[float]] = []
    for row in intervention_rows:
        scores_by_target = {
            str(score_row["target_id"]): _require_number(score_row["score_drop"], "target_score.score_drop")
            for score_row in _require_list(row.get("target_score_rows"), f"{row['neuron_set_id']}.target_score_rows")
        }
        matrix.append([scores_by_target[target_id] for target_id in target_ids])
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(target_ids)), max(5, 0.55 * len(intervention_rows))))
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_title("Feature-score drop after neuron-set ablation")
    ax.set_xlabel("target")
    ax.set_ylabel("neuron set")
    ax.set_xticks(list(range(len(target_ids))))
    ax.set_xticklabels(target_ids, rotation=35, ha="right")
    ax.set_yticks(list(range(len(intervention_rows))))
    ax.set_yticklabels([str(row["neuron_set_id"]) for row in intervention_rows])
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ax.text(column_index, row_index, f"{value:.3g}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="baseline score - ablated score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_neuron_intervention_set_size_plot(
    *,
    intervention_sets: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not intervention_sets:
        raise ValueError("Cannot render set-size plot without intervention sets.")
    labels = [str(row["neuron_set_id"]) for row in intervention_sets]
    counts = [_require_int(row.get("neuron_count"), f"{row['neuron_set_id']}.neuron_count") for row in intervention_sets]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(labels)), 5))
    ax.bar(list(range(len(labels))), counts)
    ax.set_title("Selected intervention neuron-set sizes")
    ax.set_ylabel("neurons")
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_single_neuron_intervention_plot(
    *,
    single_neuron_rows: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    if not single_neuron_rows:
        raise ValueError("Cannot render single-neuron plot without rows.")
    labels = [str(row["neuron_label"]) for row in single_neuron_rows]
    values = [
        _require_number(row["candidate_score_drop_summary"]["candidate_score_drop_mean"], f"{row['neuron_label']}.score_drop_mean")
        for row in single_neuron_rows
    ]
    _, plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(labels)), 5))
    ax.bar(list(range(len(labels))), values)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Single-neuron candidate feature-score drop")
    ax.set_ylabel("mean candidate score drop")
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _render_neuron_intervention_plots(
    *,
    output_dir: Path,
    intervention_sets: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    target_ids: list[str],
    single_neuron_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    plot_paths = {
        "behavior": _render_neuron_intervention_behavior_plot(
            intervention_rows=intervention_rows,
            output_path=output_dir / "candidate_neuron_intervention_behavior.svg",
        ),
        "feature_scores": _render_neuron_intervention_score_plot(
            intervention_rows=intervention_rows,
            target_ids=target_ids,
            output_path=output_dir / "candidate_neuron_intervention_feature_scores.svg",
        ),
        "set_sizes": _render_neuron_intervention_set_size_plot(
            intervention_sets=intervention_sets,
            output_path=output_dir / "candidate_neuron_intervention_set_sizes.svg",
        ),
    }
    if single_neuron_rows:
        plot_paths["single_neurons"] = _render_single_neuron_intervention_plot(
            single_neuron_rows=single_neuron_rows,
            output_path=output_dir / "candidate_neuron_intervention_single_neurons.svg",
        )
    return plot_paths


def _write_neuron_intervention_markdown(
    *,
    output_path: Path,
    report_payload: dict[str, Any],
) -> Path:
    baseline = _require_dict(report_payload.get("baseline"), "neuron_intervention.baseline")
    intervention_rows = _require_list(report_payload.get("intervention_rows"), "neuron_intervention.intervention_rows")
    lines = [
        "# Candidate Neuron Intervention",
        "",
        "This report tests whether neuron sets from a coalition map are causally required by ablating their MLP hidden activations.",
        "",
        "## Inputs",
        "",
        f"- Coalition map: `{report_payload['coalition_map_path']}`",
        f"- Checkpoint dir: `{report_payload['checkpoint_dir']}`",
        f"- Checkpoint step: {report_payload['checkpoint_step']}",
        f"- Selected score targets: {', '.join(report_payload['target_ids'])}",
        f"- Top K per set: {report_payload['selection']['top_k_per_set']}",
        "",
        "## Baseline",
        "",
        "| loss | token accuracy | answer accuracy | heldout accuracy | structural OOD accuracy |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            [
                _markdown_number(baseline["loss"]),
                _markdown_number(baseline["token_accuracy"]),
                _markdown_number(baseline["answer_accuracy"]),
                _markdown_number(baseline["heldout_answer_accuracy"]),
                _markdown_number(baseline["structural_ood_answer_accuracy"]),
            ]
        )
        + " |",
        "",
        "## Neuron-Set Ablations",
        "",
        "| neuron set | neurons | answer drop | heldout drop | structural OOD drop | loss increase | mean candidate score drop | all candidate scores drop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for raw_row in intervention_rows:
        row = _require_dict(raw_row, "neuron_intervention.intervention_rows[]")
        drops = _require_dict(row["behavior_effect"]["drops"], f"{row['neuron_set_id']}.drops")
        score_summary = _require_dict(row["candidate_score_drop_summary"], f"{row['neuron_set_id']}.candidate_score_drop_summary")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["neuron_set_id"]),
                    str(row["neuron_count"]),
                    _markdown_number(drops["answer_accuracy_drop"]),
                    _markdown_number(drops["heldout_answer_accuracy_drop"]),
                    _markdown_number(drops["structural_ood_answer_accuracy_drop"]),
                    _markdown_number(drops["loss_increase"]),
                    _markdown_number(score_summary["candidate_score_drop_mean"]),
                    str(score_summary["all_candidate_scores_drop"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Target Score Drops", ""])
    lines.append("| neuron set | target | score drop | baseline score | ablated score |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for raw_row in intervention_rows:
        row = _require_dict(raw_row, "neuron_intervention.intervention_rows[]")
        for raw_score in _require_list(row.get("target_score_rows"), f"{row['neuron_set_id']}.target_score_rows"):
            score = _require_dict(raw_score, f"{row['neuron_set_id']}.target_score_rows[]")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["neuron_set_id"]),
                        str(score["target_id"]),
                        _markdown_number(score["score_drop"]),
                        _markdown_number(score["baseline_score"]),
                        _markdown_number(score["intervened_score"]),
                    ]
                )
                + " |"
            )
    single_rows = _require_list(report_payload.get("single_neuron_rows"), "neuron_intervention.single_neuron_rows")
    if single_rows:
        lines.extend(["", "## Single-Neuron Ablations", ""])
        lines.append("| neuron | answer drop | mean candidate score drop | all candidate scores drop |")
        lines.append("| --- | ---: | ---: | --- |")
        for raw_row in single_rows:
            row = _require_dict(raw_row, "neuron_intervention.single_neuron_rows[]")
            drops = _require_dict(row["behavior_effect"]["drops"], f"{row['neuron_label']}.drops")
            score_summary = _require_dict(row["candidate_score_drop_summary"], f"{row['neuron_label']}.score_summary")
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["neuron_label"]),
                        _markdown_number(drops["answer_accuracy_drop"]),
                        _markdown_number(score_summary["candidate_score_drop_mean"]),
                        str(score_summary["all_candidate_scores_drop"]),
                    ]
                )
                + " |"
            )
    omitted = _require_list(report_payload.get("omitted_empty_set_ids"), "neuron_intervention.omitted_empty_set_ids")
    if omitted:
        lines.extend(["", "## Empty Sets Not Run", ""])
        for set_id in omitted:
            lines.append(f"- {set_id}")
    lines.extend(["", "## Unsupported Claims", ""])
    for claim in _require_list(report_payload.get("unsupported_claims"), "neuron_intervention.unsupported_claims"):
        lines.append(f"- {claim}")
    lines.extend(["", "## Plots", ""])
    for name, plot_path in _require_dict(report_payload.get("plots"), "neuron_intervention.plots").items():
        lines.append(f"- {name}: `{plot_path}`")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_candidate_neuron_intervention(
    *,
    config_path: Path,
    probe_set_path: Path,
    coalition_map_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    checkpoint_step: int,
    device_name: str = "cpu",
    top_k_per_set: int = 8,
    single_neuron_top_k: int = 0,
    score_individual_features: bool = False,
) -> tuple[Path, Path, dict[str, Path]]:
    if checkpoint_step < 0:
        raise ValueError("checkpoint_step must be non-negative.")
    if top_k_per_set <= 0:
        raise ValueError("top_k_per_set must be positive.")
    if single_neuron_top_k < 0:
        raise ValueError("single_neuron_top_k must be non-negative.")

    spec = TrainSpec.from_path(config_path)
    device = require_device(device_name)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    batches = _load_probe_batches(spec=spec, probe_set_path=probe_set_path, vocab=vocab, device=device)
    coalition_payload = _load_coalition_map_payload(coalition_map_path)
    target_specs = _candidate_intervention_targets(
        coalition_payload=coalition_payload,
        score_individual_features=score_individual_features,
    )
    target_ids = [
        _sanitize_candidate_id(_require_non_empty_str(target.get("target_id"), "intervention.target.target_id"))
        for target in target_specs
    ]
    candidate_target_ids = _coalition_candidate_target_ids(coalition_payload)
    intervention_set_payload = _build_candidate_intervention_sets(
        coalition_payload=coalition_payload,
        top_k_per_set=top_k_per_set,
    )
    intervention_sets = [
        _require_dict(row, "intervention_sets[]")
        for row in _require_list(intervention_set_payload.get("intervention_sets"), "intervention_sets")
    ]

    checkpoint_paths_by_step = _resolve_checkpoint_paths_by_step(checkpoint_dir)
    if checkpoint_step not in checkpoint_paths_by_step:
        raise FileNotFoundError(f"Checkpoint for step {checkpoint_step} not found in {checkpoint_dir}.")
    checkpoint_path = checkpoint_paths_by_step[checkpoint_step]
    checkpoint = load_checkpoint(checkpoint_path, device)
    if int(checkpoint["step"]) != checkpoint_step:
        raise ValueError(f"Checkpoint payload step mismatch for {checkpoint_path}: {checkpoint['step']} != {checkpoint_step}.")
    model = build_model(spec.model, len(vocab.tokens), device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    basis_cache: dict[str, dict[str, Any]] = {}
    baseline = _evaluate_intervention_probe_metrics(
        model=model,
        batches=batches,
        pad_token_id=vocab.pad_token_id,
    )
    baseline_scores = _feature_scores_for_intervention_targets(
        model=model,
        batches=batches,
        target_specs=target_specs,
        basis_cache=basis_cache,
        device=device,
    )

    intervention_rows: list[dict[str, Any]] = []
    for intervention_set in intervention_sets:
        neuron_mask = _neuron_mask_for_records(
            spec=spec,
            neurons=_require_list(intervention_set.get("neurons"), f"{intervention_set['neuron_set_id']}.neurons"),
            device=device,
        )
        intervened = _evaluate_intervention_probe_metrics(
            model=model,
            batches=batches,
            pad_token_id=vocab.pad_token_id,
            neuron_mask=neuron_mask,
        )
        intervened_scores = _feature_scores_for_intervention_targets(
            model=model,
            batches=batches,
            target_specs=target_specs,
            basis_cache=basis_cache,
            device=device,
            neuron_mask=neuron_mask,
        )
        target_score_rows, candidate_score_drop_summary = _target_score_rows(
            baseline_scores=baseline_scores,
            intervened_scores=intervened_scores,
            candidate_target_ids=candidate_target_ids,
        )
        intervention_rows.append(
            {
                "neuron_set_id": intervention_set["neuron_set_id"],
                "set_kind": intervention_set["set_kind"],
                "candidate_id": intervention_set.get("candidate_id"),
                "source": intervention_set["source"],
                "selection_metric": intervention_set["selection_metric"],
                "neuron_count": intervention_set["neuron_count"],
                "neurons": intervention_set["neurons"],
                "category_score_sums": intervention_set["category_score_sums"],
                "candidate_score_sums": intervention_set["candidate_score_sums"],
                "intervened": intervened,
                "behavior_effect": _intervention_metric_deltas(baseline=baseline, intervened=intervened),
                "target_score_rows": target_score_rows,
                "candidate_score_drop_summary": candidate_score_drop_summary,
            }
        )

    single_neuron_rows: list[dict[str, Any]] = []
    if single_neuron_top_k > 0:
        shared_positive_sets = [row for row in intervention_sets if str(row["neuron_set_id"]) == "shared_positive"]
        if not shared_positive_sets:
            raise ValueError("single_neuron_top_k was requested, but the shared_positive intervention set is absent.")
        for raw_neuron in _require_list(shared_positive_sets[0].get("neurons"), "shared_positive.neurons")[:single_neuron_top_k]:
            neuron = _require_dict(raw_neuron, "shared_positive.neurons[]")
            neuron_mask = _neuron_mask_for_records(spec=spec, neurons=[neuron], device=device)
            intervened = _evaluate_intervention_probe_metrics(
                model=model,
                batches=batches,
                pad_token_id=vocab.pad_token_id,
                neuron_mask=neuron_mask,
            )
            intervened_scores = _feature_scores_for_intervention_targets(
                model=model,
                batches=batches,
                target_specs=target_specs,
                basis_cache=basis_cache,
                device=device,
                neuron_mask=neuron_mask,
            )
            target_score_rows, candidate_score_drop_summary = _target_score_rows(
                baseline_scores=baseline_scores,
                intervened_scores=intervened_scores,
                candidate_target_ids=candidate_target_ids,
            )
            single_neuron_rows.append(
                {
                    "neuron_label": neuron["label"],
                    "layer": neuron["layer"],
                    "neuron": neuron["neuron"],
                    "intervened": intervened,
                    "behavior_effect": _intervention_metric_deltas(baseline=baseline, intervened=intervened),
                    "target_score_rows": target_score_rows,
                    "candidate_score_drop_summary": candidate_score_drop_summary,
                }
            )

    plot_paths = _render_neuron_intervention_plots(
        output_dir=output_dir,
        intervention_sets=intervention_sets,
        intervention_rows=intervention_rows,
        target_ids=target_ids,
        single_neuron_rows=single_neuron_rows,
    )
    unsupported_claims = [
        "causal_sufficiency_of_shared_neurons",
        "source_to_target_neuron_activation_patch",
        "cross_seed_intervention_stability",
        "per_minibatch_intervention_trace",
        "complete_dense_circuit_decomposition",
    ]
    report_path = output_dir / "candidate_neuron_intervention_report.json"
    markdown_path = output_dir / "candidate_neuron_intervention_report.md"
    report_payload = {
        "schema_version": NEURON_INTERVENTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "coalition_map_path": str(coalition_map_path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "checkpoint_recorded_step": int(checkpoint["step"]),
        "device": device_name,
        "candidate_target_ids": candidate_target_ids,
        "target_ids": target_ids,
        "target_specs": target_specs,
        "selection": {
            "top_k_per_set": top_k_per_set,
            "single_neuron_top_k": single_neuron_top_k,
            "score_individual_features": score_individual_features,
        },
        "baseline": baseline,
        "baseline_target_scores": baseline_scores,
        "intervention_sets": intervention_sets,
        "omitted_empty_set_ids": intervention_set_payload["omitted_empty_set_ids"],
        "intervention_rows": intervention_rows,
        "single_neuron_rows": single_neuron_rows,
        "unsupported_claims": unsupported_claims,
        "plots": {name: str(path) for name, path in plot_paths.items()},
    }
    write_json(report_path, report_payload)
    _write_neuron_intervention_markdown(output_path=markdown_path, report_payload=report_payload)
    return report_path, markdown_path, plot_paths
