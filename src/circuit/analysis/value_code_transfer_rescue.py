from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.contextual_svd_alignment import CONTEXTUAL_GROUP_BY_OPTIONS
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    GEOMETRY_POSITION_ROLES,
    _checkpoint_step_from_path,
    _intervention_positions_for_query,
)
from circuit.analysis.output_route_closure import _mean, _safe_r_squared
from circuit.analysis.value_code_subspace_report import (
    _make_probe_loader,
    _resolve_checkpoint_paths,
    _token_label,
    _valid_residual_stages,
    _value_logit_metrics,
)
from circuit.analysis.value_code_transfer_map_report import (
    VALUE_CODE_TRANSFER_CONTROLS,
    _basis_from_examples,
    _collect_examples,
    _fit_affine_map,
    _predict_affine,
    _project,
    _random_basis,
    _reconstruct,
    _validate_basis_rank,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


VALUE_CODE_TRANSFER_RESCUE_SCHEMA_VERSION = 1


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / "value_code_transfer_rescue_report.json",
        output_dir / "value_code_transfer_rescue_report.md",
        output_dir / "value_code_transfer_rescue_rows.jsonl",
        output_dir / "value_code_transfer_rescue_summary_rows.jsonl",
        output_dir / "value_code_transfer_rescue_subspaces.jsonl",
        output_dir / "value_code_transfer_rescue_pairs.jsonl",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing value-code-transfer-rescue outputs without --overwrite: "
            f"{[str(path) for path in existing]}"
        )


def _example_key(*, sample_id: str, query_index: int) -> tuple[str, int]:
    return str(sample_id), int(query_index)


def _scalar_payload(
    *,
    logits: torch.Tensor,
    answer_token_id: int,
    value_token_ids: torch.Tensor,
    vocab: Vocabulary,
    label: str,
    fixed_competitor_token_ids: dict[str, int] | None = None,
) -> dict[str, float]:
    value_metrics = _value_logit_metrics(
        logits=logits,
        answer_token_id=answer_token_id,
        value_token_ids=value_token_ids,
        vocab=vocab,
        label=label,
    )
    target = torch.tensor([int(answer_token_id)], dtype=torch.long, device=logits.device)
    negative_loss = -F.cross_entropy(logits.float().unsqueeze(0), target, reduction="mean")
    payload = {
        "correct_value_logit": float(value_metrics["correct_value_logit"]),
        "value_margin": float(value_metrics["value_margin"]),
        "negative_answer_loss": float(negative_loss.detach().cpu().item()),
        "value_accuracy": float(value_metrics["value_accuracy"]),
    }
    if fixed_competitor_token_ids is not None:
        for scalar_name, token_id in sorted(fixed_competitor_token_ids.items()):
            matches = (value_token_ids == int(token_id)).nonzero(as_tuple=False)
            if int(matches.size(0)) != 1:
                raise RuntimeError(
                    f"{label} fixed competitor {_token_label(vocab, int(token_id))} not found exactly once in value ids."
                )
            competitor_index = int(matches[0].item())
            value_logits = logits.index_select(dim=0, index=value_token_ids.to(logits.device))
            payload[scalar_name] = float(
                value_metrics["correct_value_logit"] - float(value_logits[competitor_index].item())
            )
    return payload


def _prediction_payloads_for_rank(
    *,
    checkpoint_path: Path,
    examples: list[dict[str, Any]],
    fit_examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    rank: int,
    controls: list[str],
    group_by: str,
    source_stage: str,
    source_position_role: str,
    target_stage: str,
    target_position_role: str,
    context_stage: str | None,
    context_position_role: str | None,
    context_rank: int | None,
    ridge_lambda: float,
    random_seed: int,
    vocab: Vocabulary,
) -> tuple[dict[str, dict[int, torch.Tensor]], torch.Tensor, list[dict[str, Any]]]:
    step = _checkpoint_step_from_path(checkpoint_path)
    source_subspace, source_summary = _basis_from_examples(
        examples=fit_examples,
        vector_field="_source_vector",
        token_field="source_group_token_id",
        role_label=f"{checkpoint_path.name}:rescue:source:{source_stage}:{source_position_role}:{group_by}:rank{rank}",
        context_role=source_position_role,
        group_by=group_by,
        vocab=vocab,
        rank=rank,
    )
    target_subspace, target_summary = _basis_from_examples(
        examples=fit_examples,
        vector_field="_target_vector",
        token_field="target_group_token_id",
        role_label=f"{checkpoint_path.name}:rescue:target:{target_stage}:{target_position_role}:{group_by}:rank{rank}",
        context_role=target_position_role,
        group_by=group_by,
        vocab=vocab,
        rank=rank,
    )
    source_basis = source_subspace["identity_basis"].float()
    target_basis = target_subspace["identity_basis"].float()
    subspace_rows = [
        {
            "schema_version": VALUE_CODE_TRANSFER_RESCUE_SCHEMA_VERSION,
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_step": step,
            "basis_rank": rank,
            "basis_role": "source_value_identity",
            **source_summary,
        },
        {
            "schema_version": VALUE_CODE_TRANSFER_RESCUE_SCHEMA_VERSION,
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_step": step,
            "basis_rank": rank,
            "basis_role": "target_value_identity",
            **target_summary,
        },
    ]

    source_coords = {
        int(example["example_index"]): _project(
            example["_source_vector"],
            source_basis,
            label=f"{checkpoint_path.name}/rank{rank}/{example['example_index']}.source",
        )
        for example in examples
    }
    context_coords: dict[int, torch.Tensor] | None = None
    if context_stage is not None or context_position_role is not None or context_rank is not None:
        if context_stage is None or context_position_role is None or context_rank is None:
            raise ValueError(
                "Contextual transfer requires --context-stage, --context-position-role, and --context-rank together."
            )
        missing_context = [int(example["example_index"]) for example in examples if "_context_vector" not in example]
        if missing_context:
            raise RuntimeError(f"Missing context vectors for examples: {missing_context[:5]}.")
        context_subspace, context_summary = _basis_from_examples(
            examples=fit_examples,
            vector_field="_context_vector",
            token_field="target_group_token_id",
            role_label=(
                f"{checkpoint_path.name}:rescue:context:{context_stage}:"
                f"{context_position_role}:{group_by}:rank{context_rank}"
            ),
            context_role=context_position_role,
            group_by=group_by,
            vocab=vocab,
            rank=int(context_rank),
        )
        context_basis = context_subspace["identity_basis"].float()
        subspace_rows.append(
            {
                "schema_version": VALUE_CODE_TRANSFER_RESCUE_SCHEMA_VERSION,
                "checkpoint": str(checkpoint_path),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_step": step,
                "basis_rank": int(context_rank),
                "basis_role": "prediction_context_identity",
                "context_stage": context_stage,
                "context_position_role": context_position_role,
                **context_summary,
            }
        )
        context_coords = {
            int(example["example_index"]): _project(
                example["_context_vector"],
                context_basis,
                label=f"{checkpoint_path.name}/rank{rank}/{example['example_index']}.context",
            )
            for example in examples
        }
    target_coords = {
        int(example["example_index"]): _project(
            example["_target_vector"],
            target_basis,
            label=f"{checkpoint_path.name}/rank{rank}/{example['example_index']}.target",
        )
        for example in examples
    }
    fit_source_matrix = torch.stack([source_coords[int(example["example_index"])] for example in fit_examples], dim=0)
    fit_target_matrix = torch.stack([target_coords[int(example["example_index"])] for example in fit_examples], dim=0)
    transfer = _fit_affine_map(
        source_coords=fit_source_matrix,
        target_coords=fit_target_matrix,
        ridge_lambda=ridge_lambda,
        label=f"{checkpoint_path.name}/rank{rank}/true_transfer_rescue",
    )

    prediction_coords: dict[str, dict[int, torch.Tensor]] = {
        "oracle_actual_projected": {},
        "true_transfer": {},
    }
    for example in eval_examples:
        example_index = int(example["example_index"])
        prediction_coords["oracle_actual_projected"][example_index] = target_coords[example_index]
        prediction_coords["true_transfer"][example_index] = _predict_affine(
            coords=source_coords[example_index],
            transfer=transfer,
            label=f"{checkpoint_path.name}/rank{rank}/{example_index}.true_transfer",
        )
    if context_coords is not None:
        fit_context_matrix = torch.stack(
            [context_coords[int(example["example_index"])] for example in fit_examples],
            dim=0,
        )
        context_transfer = _fit_affine_map(
            source_coords=fit_context_matrix,
            target_coords=fit_target_matrix,
            ridge_lambda=ridge_lambda,
            label=f"{checkpoint_path.name}/rank{rank}/context_only_transfer_rescue",
        )
        fit_source_context_matrix = torch.stack(
            [
                torch.cat(
                    [
                        source_coords[int(example["example_index"])],
                        context_coords[int(example["example_index"])],
                    ],
                    dim=0,
                )
                for example in fit_examples
            ],
            dim=0,
        )
        source_context_transfer = _fit_affine_map(
            source_coords=fit_source_context_matrix,
            target_coords=fit_target_matrix,
            ridge_lambda=ridge_lambda,
            label=f"{checkpoint_path.name}/rank{rank}/source_plus_context_transfer_rescue",
        )
        prediction_coords["context_only"] = {}
        prediction_coords["source_plus_context"] = {}
        for example in eval_examples:
            example_index = int(example["example_index"])
            prediction_coords["context_only"][example_index] = _predict_affine(
                coords=context_coords[example_index],
                transfer=context_transfer,
                label=f"{checkpoint_path.name}/rank{rank}/{example_index}.context_only",
            )
            prediction_coords["source_plus_context"][example_index] = _predict_affine(
                coords=torch.cat([source_coords[example_index], context_coords[example_index]], dim=0),
                transfer=source_context_transfer,
                label=f"{checkpoint_path.name}/rank{rank}/{example_index}.source_plus_context",
            )

    if "shuffled_answer_value" in controls:
        if len(eval_examples) < 2:
            raise RuntimeError("shuffled_answer_value control requires at least two eval examples.")
        prediction_coords["shuffled_answer_value"] = {}
        shifted = eval_examples[1:] + eval_examples[:1]
        for example, shifted_example in zip(eval_examples, shifted, strict=True):
            example_index = int(example["example_index"])
            shifted_index = int(shifted_example["example_index"])
            prediction_coords["shuffled_answer_value"][example_index] = _predict_affine(
                coords=source_coords[shifted_index],
                transfer=transfer,
                label=f"{checkpoint_path.name}/rank{rank}/{example_index}.shuffled_answer_value",
            )
            if context_coords is not None:
                prediction_coords.setdefault("source_plus_context_shuffled_answer_value", {})[
                    example_index
                ] = _predict_affine(
                    coords=torch.cat([source_coords[shifted_index], context_coords[example_index]], dim=0),
                    transfer=source_context_transfer,
                    label=f"{checkpoint_path.name}/rank{rank}/{example_index}.source_plus_context_shuffled_answer_value",
                )
    if "wrong_support_value" in controls:
        wrong_source_coords = {
            int(example["example_index"]): _project(
                example["_wrong_source_vector"],
                source_basis,
                label=f"{checkpoint_path.name}/rank{rank}/{example['example_index']}.wrong_source",
            )
            for example in examples
        }
        prediction_coords["wrong_support_value"] = {}
        for example in eval_examples:
            example_index = int(example["example_index"])
            prediction_coords["wrong_support_value"][example_index] = _predict_affine(
                coords=wrong_source_coords[example_index],
                transfer=transfer,
                label=f"{checkpoint_path.name}/rank{rank}/{example_index}.wrong_support_value",
            )
            if context_coords is not None:
                prediction_coords.setdefault("source_plus_context_wrong_support_value", {})[
                    example_index
                ] = _predict_affine(
                    coords=torch.cat([wrong_source_coords[example_index], context_coords[example_index]], dim=0),
                    transfer=source_context_transfer,
                    label=f"{checkpoint_path.name}/rank{rank}/{example_index}.source_plus_context_wrong_support_value",
                )
    if "key_identity" in controls:
        key_subspace, key_summary = _basis_from_examples(
            examples=fit_examples,
            vector_field="_key_source_vector",
            token_field="key_group_token_id",
            role_label=f"{checkpoint_path.name}:rescue:key_control:{source_stage}:support_key:support_key:rank{rank}",
            context_role="support_key",
            group_by="support_key",
            vocab=vocab,
            rank=rank,
        )
        key_basis = key_subspace["identity_basis"].float()
        subspace_rows.append(
            {
                "schema_version": VALUE_CODE_TRANSFER_RESCUE_SCHEMA_VERSION,
                "checkpoint": str(checkpoint_path),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_step": step,
                "basis_rank": rank,
                "basis_role": "source_key_identity_control",
                **key_summary,
            }
        )
        key_coords = {
            int(example["example_index"]): _project(
                example["_key_source_vector"],
                key_basis,
                label=f"{checkpoint_path.name}/rank{rank}/{example['example_index']}.key_source",
            )
            for example in examples
        }
        key_fit_source = torch.stack([key_coords[int(example["example_index"])] for example in fit_examples], dim=0)
        key_transfer = _fit_affine_map(
            source_coords=key_fit_source,
            target_coords=fit_target_matrix,
            ridge_lambda=ridge_lambda,
            label=f"{checkpoint_path.name}/rank{rank}/key_identity_transfer_rescue",
        )
        prediction_coords["key_identity"] = {}
        for example in eval_examples:
            example_index = int(example["example_index"])
            prediction_coords["key_identity"][example_index] = _predict_affine(
                coords=key_coords[example_index],
                transfer=key_transfer,
                label=f"{checkpoint_path.name}/rank{rank}/{example_index}.key_identity",
            )
    if "random_subspace" in controls:
        random_basis = _random_basis(
            ambient_dim=int(source_basis.size(0)),
            rank=rank,
            seed=int(random_seed) + step * 1009 + rank * 9176,
            label=f"{checkpoint_path.name}/rank{rank}/random_source",
        )
        subspace_rows.append(
            {
                "schema_version": VALUE_CODE_TRANSFER_RESCUE_SCHEMA_VERSION,
                "checkpoint": str(checkpoint_path),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_step": step,
                "basis_rank": rank,
                "basis_role": "random_source_control",
                "role": f"{checkpoint_path.name}:rescue:random_source:rank{rank}",
                "context_role": source_position_role,
                "group_by": "random_subspace",
                "num_vectors": None,
                "num_unique_tokens": None,
                "tokens": None,
                "ambient_dim": int(random_basis.size(0)),
                "pca_rank": rank,
                "identity_centered_rank": None,
                "identity_singular_values": None,
                "all_vector_centered_rank": None,
                "all_vector_singular_values": None,
            }
        )
        random_coords = {
            int(example["example_index"]): _project(
                example["_source_vector"],
                random_basis,
                label=f"{checkpoint_path.name}/rank{rank}/{example['example_index']}.random_source",
            )
            for example in examples
        }
        random_fit_source = torch.stack(
            [random_coords[int(example["example_index"])] for example in fit_examples],
            dim=0,
        )
        random_transfer = _fit_affine_map(
            source_coords=random_fit_source,
            target_coords=fit_target_matrix,
            ridge_lambda=ridge_lambda,
            label=f"{checkpoint_path.name}/rank{rank}/random_subspace_transfer_rescue",
        )
        prediction_coords["random_subspace"] = {}
        for example in eval_examples:
            example_index = int(example["example_index"])
            prediction_coords["random_subspace"][example_index] = _predict_affine(
                coords=random_coords[example_index],
                transfer=random_transfer,
                label=f"{checkpoint_path.name}/rank{rank}/{example_index}.random_subspace",
            )
            if context_coords is not None:
                prediction_coords.setdefault("source_plus_context_random_subspace", {})[
                    example_index
                ] = _predict_affine(
                    coords=torch.cat([random_coords[example_index], context_coords[example_index]], dim=0),
                    transfer=source_context_transfer,
                    label=f"{checkpoint_path.name}/rank{rank}/{example_index}.source_plus_context_random_subspace",
                )

    predicted_vectors = {
        eval_kind: {
            example_index: _reconstruct(
                coords,
                target_basis,
                label=f"{checkpoint_path.name}/rank{rank}/{eval_kind}/{example_index}.predicted_vector",
            )
            for example_index, coords in coords_by_index.items()
        }
        for eval_kind, coords_by_index in prediction_coords.items()
    }
    return predicted_vectors, target_basis, subspace_rows


def _patch_target_value_code(
    *,
    clean_stage: torch.Tensor,
    batch: dict[str, Any],
    metadata: dict[str, torch.Tensor],
    examples_by_key: dict[tuple[str, int], dict[str, Any]],
    target_basis: torch.Tensor,
    predicted_vectors_by_index: dict[int, torch.Tensor] | None,
    target_position_role: str,
) -> torch.Tensor:
    patched = clean_stage.detach().clone()
    basis = target_basis.to(patched.device).float()
    for flat_index in range(int(metadata["rows"].numel())):
        batch_row = int(metadata["rows"][flat_index].item())
        query_index = int(metadata["query_indices"][flat_index].item())
        record = batch["records"][batch_row]
        key = _example_key(sample_id=str(record["sample_id"]), query_index=query_index)
        if key not in examples_by_key:
            continue
        example = examples_by_key[key]
        target_batch_row, target_positions = _intervention_positions_for_query(
            batch=batch,
            metadata=metadata,
            flat_index=flat_index,
            position_role=target_position_role,
        )
        if len(target_positions) != 1:
            raise RuntimeError(
                "value-code-transfer-rescue currently requires a single target patch position; "
                f"got positions={target_positions} for {record['sample_id']} query={query_index}."
            )
        position = int(target_positions[0])
        clean_vector = clean_stage[target_batch_row, position, :].detach().float()
        actual_coords = clean_vector.matmul(basis)
        actual_projected = basis.matmul(actual_coords).to(patched.device)
        if predicted_vectors_by_index is None:
            replacement_projected = torch.zeros_like(actual_projected)
        else:
            example_index = int(example["example_index"])
            if example_index not in predicted_vectors_by_index:
                raise KeyError(f"Missing predicted transfer vector for example_index={example_index}.")
            replacement_projected = predicted_vectors_by_index[example_index].to(patched.device).float()
        patched[target_batch_row, position, :] = clean_vector - actual_projected + replacement_projected
    return patched


def _metrics_by_example_key(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    metadata: dict[str, torch.Tensor],
    batch: dict[str, Any],
    examples_by_key: dict[tuple[str, int], dict[str, Any]],
    value_token_ids: torch.Tensor,
    vocab: Vocabulary,
    label: str,
    fixed_competitors_by_key: dict[tuple[str, int], dict[str, int]] | None = None,
) -> dict[tuple[str, int], dict[str, float]]:
    values: dict[tuple[str, int], dict[str, float]] = {}
    for flat_index in range(int(metadata["rows"].numel())):
        batch_row = int(metadata["rows"][flat_index].item())
        query_index = int(metadata["query_indices"][flat_index].item())
        record = batch["records"][batch_row]
        key = _example_key(sample_id=str(record["sample_id"]), query_index=query_index)
        if key not in examples_by_key:
            continue
        target_id = int(targets[flat_index].detach().cpu().item())
        expected_target = int(examples_by_key[key]["answer_value_token_id"])
        if target_id != expected_target:
            raise RuntimeError(
                f"{label} answer target mismatch for {key}: expected={expected_target} got={target_id}."
            )
        values[key] = _scalar_payload(
            logits=logits[flat_index],
            answer_token_id=target_id,
            value_token_ids=value_token_ids,
            vocab=vocab,
            label=f"{label}/{key[0]}/{key[1]}",
            fixed_competitor_token_ids=None
            if fixed_competitors_by_key is None
            else fixed_competitors_by_key.get(key),
        )
    return values


def _logits_by_example_key(
    *,
    logits: torch.Tensor,
    metadata: dict[str, torch.Tensor],
    batch: dict[str, Any],
    examples_by_key: dict[tuple[str, int], dict[str, Any]],
    label: str,
) -> dict[tuple[str, int], torch.Tensor]:
    values: dict[tuple[str, int], torch.Tensor] = {}
    for flat_index in range(int(metadata["rows"].numel())):
        batch_row = int(metadata["rows"][flat_index].item())
        query_index = int(metadata["query_indices"][flat_index].item())
        record = batch["records"][batch_row]
        key = _example_key(sample_id=str(record["sample_id"]), query_index=query_index)
        if key not in examples_by_key:
            continue
        if key in values:
            raise RuntimeError(f"{label} saw duplicate answer row for {key}.")
        values[key] = logits[flat_index]
    return values


def _rescue_rows_for_rank(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    loader: torch.utils.data.DataLoader[Any],
    eval_examples: list[dict[str, Any]],
    target_stage: str,
    target_position_role: str,
    rank: int,
    predicted_vectors_by_kind: dict[str, dict[int, torch.Tensor]],
    target_basis: torch.Tensor,
    value_token_ids: torch.Tensor,
    vocab: Vocabulary,
) -> list[dict[str, Any]]:
    examples_by_key = {
        _example_key(sample_id=str(example["sample_id"]), query_index=int(example["query_index"])): example
        for example in eval_examples
    }
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, next(model.parameters()).device)
            clean_outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_residual_streams=True,
            )
            if clean_outputs.residual_streams is None:
                raise RuntimeError("value-code-transfer-rescue requires residual streams.")
            if target_stage not in clean_outputs.residual_streams:
                raise KeyError(
                    f"Missing target stage {target_stage!r}. Available stages: {sorted(clean_outputs.residual_streams)}"
                )
            clean_logits, clean_targets, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            clean_values = _metrics_by_example_key(
                logits=clean_logits,
                targets=clean_targets,
                metadata=clean_metadata,
                batch=batch,
                examples_by_key=examples_by_key,
                value_token_ids=value_token_ids,
                vocab=vocab,
                label="value-code-transfer-rescue/clean",
            )
            if not clean_values:
                continue
            clean_stage = clean_outputs.residual_streams[target_stage].detach()
            ablated_patch = _patch_target_value_code(
                clean_stage=clean_stage,
                batch=batch,
                metadata=clean_metadata,
                examples_by_key=examples_by_key,
                target_basis=target_basis,
                predicted_vectors_by_index=None,
                target_position_role=target_position_role,
            )
            ablated_outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                residual_patch={target_stage: ablated_patch},
            )
            ablated_logits, ablated_targets, ablated_metadata = extract_answer_logits(ablated_outputs.logits, batch)
            ablated_values = _metrics_by_example_key(
                logits=ablated_logits,
                targets=ablated_targets,
                metadata=ablated_metadata,
                batch=batch,
                examples_by_key=examples_by_key,
                value_token_ids=value_token_ids,
                vocab=vocab,
                label="value-code-transfer-rescue/target_subspace_removed",
            )
            clean_logits_by_key = _logits_by_example_key(
                logits=clean_logits,
                metadata=clean_metadata,
                batch=batch,
                examples_by_key=examples_by_key,
                label="value-code-transfer-rescue/clean-logits",
            )
            ablated_logits_by_key = _logits_by_example_key(
                logits=ablated_logits,
                metadata=ablated_metadata,
                batch=batch,
                examples_by_key=examples_by_key,
                label="value-code-transfer-rescue/target-subspace-removed-logits",
            )
            fixed_competitors_by_key: dict[tuple[str, int], dict[str, int]] = {}
            for key in clean_values:
                if key not in ablated_values:
                    raise KeyError(f"Missing ablated values for fixed competitor key {key}.")
                if key not in clean_logits_by_key:
                    raise KeyError(f"Missing clean logits for fixed competitor key {key}.")
                if key not in ablated_logits_by_key:
                    raise KeyError(f"Missing ablated logits for fixed competitor key {key}.")
                clean_competitor = int(
                    _value_logit_metrics(
                        logits=clean_logits_by_key[key],
                        answer_token_id=int(examples_by_key[key]["answer_value_token_id"]),
                        value_token_ids=value_token_ids,
                        vocab=vocab,
                        label=f"value-code-transfer-rescue/fixed-clean/{key[0]}/{key[1]}",
                    )["best_wrong_value_token_id"]
                )
                removed_competitor = int(
                    _value_logit_metrics(
                        logits=ablated_logits_by_key[key],
                        answer_token_id=int(examples_by_key[key]["answer_value_token_id"]),
                        value_token_ids=value_token_ids,
                        vocab=vocab,
                        label=f"value-code-transfer-rescue/fixed-removed/{key[0]}/{key[1]}",
                    )["best_wrong_value_token_id"]
                )
                fixed_competitors_by_key[key] = {
                    "fixed_clean_competitor_margin": clean_competitor,
                    "fixed_removed_competitor_margin": removed_competitor,
                }
            clean_values = _metrics_by_example_key(
                logits=clean_logits,
                targets=clean_targets,
                metadata=clean_metadata,
                batch=batch,
                examples_by_key=examples_by_key,
                value_token_ids=value_token_ids,
                vocab=vocab,
                label="value-code-transfer-rescue/clean-fixed",
                fixed_competitors_by_key=fixed_competitors_by_key,
            )
            ablated_values = _metrics_by_example_key(
                logits=ablated_logits,
                targets=ablated_targets,
                metadata=ablated_metadata,
                batch=batch,
                examples_by_key=examples_by_key,
                value_token_ids=value_token_ids,
                vocab=vocab,
                label="value-code-transfer-rescue/target_subspace_removed-fixed",
                fixed_competitors_by_key=fixed_competitors_by_key,
            )
            patched_values_by_kind: dict[str, dict[tuple[str, int], dict[str, float]]] = {}
            for eval_kind, predicted_vectors_by_index in predicted_vectors_by_kind.items():
                patch = _patch_target_value_code(
                    clean_stage=clean_stage,
                    batch=batch,
                    metadata=clean_metadata,
                    examples_by_key=examples_by_key,
                    target_basis=target_basis,
                    predicted_vectors_by_index=predicted_vectors_by_index,
                    target_position_role=target_position_role,
                )
                patched_outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    residual_patch={target_stage: patch},
                )
                patched_logits, patched_targets, patched_metadata = extract_answer_logits(patched_outputs.logits, batch)
                patched_values_by_kind[eval_kind] = _metrics_by_example_key(
                    logits=patched_logits,
                    targets=patched_targets,
                    metadata=patched_metadata,
                    batch=batch,
                    examples_by_key=examples_by_key,
                    value_token_ids=value_token_ids,
                    vocab=vocab,
                    label=f"value-code-transfer-rescue/{eval_kind}",
                    fixed_competitors_by_key=fixed_competitors_by_key,
                )
            for key, clean_payload in clean_values.items():
                if key not in ablated_values:
                    raise KeyError(f"Missing ablated values for {key}.")
                example = examples_by_key[key]
                ablated_payload = ablated_values[key]
                for eval_kind, patched_values in patched_values_by_kind.items():
                    if key not in patched_values:
                        raise KeyError(f"Missing patched values for {key} eval_kind={eval_kind}.")
                    patched_payload = patched_values[key]
                    for scalar_name in [
                        "correct_value_logit",
                        "value_margin",
                        "fixed_clean_competitor_margin",
                        "fixed_removed_competitor_margin",
                        "negative_answer_loss",
                        "value_accuracy",
                    ]:
                        clean_scalar = float(clean_payload[scalar_name])
                        ablated_scalar = float(ablated_payload[scalar_name])
                        patched_scalar = float(patched_payload[scalar_name])
                        total_drop = clean_scalar - ablated_scalar
                        rescue = patched_scalar - ablated_scalar
                        rows.append(
                            {
                                "schema_version": VALUE_CODE_TRANSFER_RESCUE_SCHEMA_VERSION,
                                "checkpoint": str(checkpoint_path),
                                "checkpoint_name": checkpoint_path.name,
                                "checkpoint_step": _checkpoint_step_from_path(checkpoint_path),
                                "basis_rank": rank,
                                "eval_kind": eval_kind,
                                "sample_id": key[0],
                                "split": str(example["split"]),
                                "query_index": key[1],
                                "source_stage": str(example["source_stage"]),
                                "target_stage": target_stage,
                                "source_position_role": str(example["source_position_role"]),
                                "target_position_role": target_position_role,
                                "group_by": str(example["group_by"]),
                                "answer_value_token_id": int(example["answer_value_token_id"]),
                                "answer_value_token": str(example["answer_value_token"]),
                                "scalar_name": scalar_name,
                                "clean_scalar": clean_scalar,
                                "target_subspace_removed_scalar": ablated_scalar,
                                "patched_scalar": patched_scalar,
                                "total_drop": total_drop,
                                "rescue": rescue,
                                "unrecovered": clean_scalar - patched_scalar,
                                "rescue_fraction": None if abs(total_drop) <= 1.0e-12 else rescue / total_drop,
                                "improved_by_patch": abs(clean_scalar - patched_scalar) < abs(total_drop),
                            }
                        )
    if not rows:
        raise RuntimeError(f"No value-code-transfer-rescue rows produced for {checkpoint_path} rank={rank}.")
    return rows


def _summarize_rescue_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["checkpoint_step"]),
                int(row["basis_rank"]),
                str(row["eval_kind"]),
                str(row["scalar_name"]),
            )
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (step, rank, eval_kind, scalar_name), group_rows in sorted(grouped.items()):
        total_drops = [float(row["total_drop"]) for row in group_rows]
        rescues = [float(row["rescue"]) for row in group_rows]
        mean_total_drop = _mean(total_drops, label=f"{step}/{rank}/{eval_kind}/{scalar_name}/drop")
        mean_rescue = _mean(rescues, label=f"{step}/{rank}/{eval_kind}/{scalar_name}/rescue")
        fraction_rows = [
            float(row["rescue_fraction"])
            for row in group_rows
            if row["rescue_fraction"] is not None
        ]
        summary_rows.append(
            {
                "checkpoint_step": step,
                "basis_rank": rank,
                "eval_kind": eval_kind,
                "scalar_name": scalar_name,
                "num_rows": len(group_rows),
                "mean_clean_scalar": _mean(
                    [float(row["clean_scalar"]) for row in group_rows],
                    label=f"{step}/{rank}/{eval_kind}/{scalar_name}/clean",
                ),
                "mean_target_subspace_removed_scalar": _mean(
                    [float(row["target_subspace_removed_scalar"]) for row in group_rows],
                    label=f"{step}/{rank}/{eval_kind}/{scalar_name}/removed",
                ),
                "mean_patched_scalar": _mean(
                    [float(row["patched_scalar"]) for row in group_rows],
                    label=f"{step}/{rank}/{eval_kind}/{scalar_name}/patched",
                ),
                "mean_total_drop": mean_total_drop,
                "mean_rescue": mean_rescue,
                "mean_unrecovered": _mean(
                    [float(row["unrecovered"]) for row in group_rows],
                    label=f"{step}/{rank}/{eval_kind}/{scalar_name}/unrecovered",
                ),
                "mean_rescue_fraction_from_means": None if abs(mean_total_drop) <= 1.0e-12 else mean_rescue / mean_total_drop,
                "mean_rescue_fraction_per_row": None
                if not fraction_rows
                else _mean(fraction_rows, label=f"{step}/{rank}/{eval_kind}/{scalar_name}/fraction"),
                "improved_fraction": _mean(
                    [1.0 if bool(row["improved_by_patch"]) else 0.0 for row in group_rows],
                    label=f"{step}/{rank}/{eval_kind}/{scalar_name}/improved",
                ),
                "rescue_vs_drop_r_squared": _safe_r_squared(
                    y_values=total_drops,
                    predicted_values=rescues,
                ),
            }
        )
    return summary_rows


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    rows = sorted(
        report["summary_rows"],
        key=lambda row: (
            int(row["checkpoint_step"]),
            int(row["basis_rank"]),
            str(row["scalar_name"]),
            str(row["eval_kind"]),
        ),
    )
    lines = [
        "# Value-Code Transfer Rescue",
        "",
        "This report tests whether a fitted source-to-target value-code transfer can causally replace the target value-code component.",
        "",
        "## Calculation",
        "",
        "For each eval query, the tool removes the target-stage value-code projection at the target position.",
        "It then patches back either the actual projected component, the fitted transfer prediction, or a control prediction.",
        "",
        "```text",
        "target_removed = clean_target - project_target_value_code(clean_target)",
        "patched = target_removed + predicted_target_value_code(source)",
        "rescue = scalar(patched) - scalar(target_removed)",
        "rescue_fraction = rescue / (scalar(clean) - scalar(target_removed))",
        "```",
        "",
        "The oracle row patches back the actual projected value-code component. The true-transfer row patches the fitted support-to-prediction prediction. Control rows patch shuffled, wrong-value, key-code, or random-subspace predictions.",
        "The report also includes fixed-clean and fixed-removed competitor margins so moving best-wrong branches cannot hide a successful transfer.",
        "",
        "## Scope",
        "",
        f"- checkpoints: `{len(report['checkpoints'])}`",
        f"- records: `{report['num_probe_records']}`",
        f"- source: `{report['source_stage']} / {report['source_position_role']}`",
        f"- target: `{report['target_stage']} / {report['target_position_role']}`",
        f"- context: `{report['context_stage']} / {report['context_position_role']} / rank {report['context_rank']}`",
        f"- group-by: `{report['group_by']}`",
        f"- basis ranks: `{', '.join(str(rank) for rank in report['basis_ranks'])}`",
        f"- controls: `{', '.join(report['controls']) if report['controls'] else 'none'}`",
        "",
        "## Summary",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| step | rank | scalar | patch | rows | clean | removed | patched | drop | rescue | rescue frac | improved | R2 |",
                "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows[: int(report["markdown_top_k_rows"])]:
            rescue_fraction = (
                "n/a"
                if row["mean_rescue_fraction_from_means"] is None
                else f"{float(row['mean_rescue_fraction_from_means']):.4f}"
            )
            r2 = "n/a" if row["rescue_vs_drop_r_squared"] is None else f"{float(row['rescue_vs_drop_r_squared']):.4f}"
            lines.append(
                f"| {row['checkpoint_step']} | {row['basis_rank']} | `{row['scalar_name']}` | `{row['eval_kind']}` | "
                f"{row['num_rows']} | {float(row['mean_clean_scalar']):.6g} | "
                f"{float(row['mean_target_subspace_removed_scalar']):.6g} | "
                f"{float(row['mean_patched_scalar']):.6g} | {float(row['mean_total_drop']):.6g} | "
                f"{float(row['mean_rescue']):.6g} | {rescue_fraction} | "
                f"{float(row['improved_fraction']):.4f} | {r2} |"
            )
    else:
        lines.append("No summary rows were produced.")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "If true-transfer rescue is close to oracle and clearly above controls, the fitted transfer has causal sufficiency for the removed value-code component. If true-transfer is weak while oracle is strong, the target code is causal but the fitted transfer map is not the full write operator.",
            "",
            "## Outputs",
            "",
            f"- rescue rows: `{report['rows_path']}`",
            f"- summary rows: `{report['summary_rows_path']}`",
            f"- subspace rows: `{report['subspace_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_value_code_transfer_rescue(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    checkpoint_paths: list[Path] | None,
    device_name: str,
    source_stage: str,
    target_stage: str,
    source_position_role: str,
    target_position_role: str,
    context_stage: str | None,
    context_position_role: str | None,
    context_rank: int | None,
    group_by: str,
    split_filter: list[str] | None,
    max_records: int | None,
    batch_size: int | None,
    basis_ranks: list[int],
    controls: list[str],
    fit_fraction: float,
    ridge_lambda: float,
    random_seed: int,
    markdown_top_k_rows: int,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    if source_position_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported source_position_role {source_position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if target_position_role not in GEOMETRY_POSITION_ROLES:
        raise ValueError(f"Unsupported target_position_role {target_position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")
    context_requested = context_stage is not None or context_position_role is not None or context_rank is not None
    if context_requested:
        if context_stage is None or context_position_role is None or context_rank is None:
            raise ValueError("--context-stage, --context-position-role, and --context-rank must be provided together.")
        if context_position_role not in GEOMETRY_POSITION_ROLES:
            raise ValueError(
                f"Unsupported context_position_role {context_position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}."
            )
        _validate_basis_rank(int(context_rank))
    if group_by not in CONTEXTUAL_GROUP_BY_OPTIONS:
        raise ValueError(f"Unsupported group_by {group_by!r}; expected one of {CONTEXTUAL_GROUP_BY_OPTIONS}.")
    if not basis_ranks:
        raise ValueError("At least one --basis-rank is required.")
    for rank in basis_ranks:
        _validate_basis_rank(int(rank))
    unsupported_controls = sorted(set(controls) - set(VALUE_CODE_TRANSFER_CONTROLS))
    if unsupported_controls:
        raise ValueError(f"Unsupported controls {unsupported_controls}; expected one of {VALUE_CODE_TRANSFER_CONTROLS}.")
    if markdown_top_k_rows <= 0:
        raise ValueError(f"markdown_top_k_rows must be positive, got {markdown_top_k_rows}.")
    _prepare_output_dir(output_dir, overwrite=overwrite)

    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    value_token_ids_cpu = torch.tensor(vocab.value_token_ids, dtype=torch.long)
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if split_filter is not None:
        split_set = set(split_filter)
        probe_records = [record for record in probe_records if str(record["split"]) in split_set]
        if not probe_records:
            raise RuntimeError(f"Split filter {sorted(split_set)} matched no probe records in {probe_set_path}.")
    if max_records is not None:
        if max_records <= 0:
            raise ValueError(f"max_records must be positive when provided, got {max_records}.")
        probe_records = probe_records[:max_records]
    if not probe_records:
        raise RuntimeError(f"No probe records selected from {probe_set_path}.")
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    device = require_device(device_name)
    model = build_model(spec.model, vocab_size=len(vocab.tokens), device=device)
    valid_stages = _valid_residual_stages(len(model.blocks))
    requested_stages = [source_stage, target_stage]
    if context_stage is not None:
        requested_stages.append(context_stage)
    unsupported_stages = sorted(set(requested_stages) - set(valid_stages))
    if unsupported_stages:
        raise ValueError(f"Unsupported stages {unsupported_stages}; expected one of {valid_stages}.")
    resolved_batch_size = int(spec.evaluation.batch_size if batch_size is None else batch_size)
    loader = _make_probe_loader(
        probe_records=probe_records,
        batch_size=resolved_batch_size,
        pad_token_id=vocab.pad_token_id,
    )
    print(
        "[value-code-transfer-rescue] "
        f"checkpoints={len(checkpoints)} records={len(probe_records)} source={source_stage}/{source_position_role} "
        f"target={target_stage}/{target_position_role} group_by={group_by} ranks={basis_ranks} controls={controls} "
        f"context={context_stage}/{context_position_role}/rank{context_rank} "
        f"device={device_name}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        print(
            f"[value-code-transfer-rescue] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        payload_step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if payload_step != path_step:
            raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={payload_step} path={path_step}.")
        model.eval()
        examples = _collect_examples(
            model=model,
            checkpoint_path=checkpoint_path,
            loader=loader,
            source_stage=source_stage,
            target_stage=target_stage,
            source_position_role=source_position_role,
            target_position_role=target_position_role,
            context_stage=context_stage,
            context_position_role=context_position_role,
            group_by=group_by,
            fit_fraction=fit_fraction,
            vocab=vocab,
            value_token_ids=value_token_ids_cpu.to(device),
            device=device,
        )
        fit_examples = [example for example in examples if example["fit_split"] == "fit"]
        eval_examples = [example for example in examples if example["fit_split"] == "eval"]
        pair_rows.extend(
            [
                {key: value for key, value in example.items() if not key.startswith("_")}
                for example in examples
            ]
        )
        for rank in basis_ranks:
            predicted_vectors_by_kind, target_basis, checkpoint_subspace_rows = _prediction_payloads_for_rank(
                checkpoint_path=checkpoint_path,
                examples=examples,
                fit_examples=fit_examples,
                eval_examples=eval_examples,
                rank=int(rank),
                controls=controls,
                group_by=group_by,
                source_stage=source_stage,
                source_position_role=source_position_role,
                target_stage=target_stage,
                target_position_role=target_position_role,
                context_stage=context_stage,
                context_position_role=context_position_role,
                context_rank=None if context_rank is None else int(context_rank),
                ridge_lambda=ridge_lambda,
                random_seed=random_seed,
                vocab=vocab,
            )
            subspace_rows.extend(checkpoint_subspace_rows)
            rows.extend(
                _rescue_rows_for_rank(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    loader=loader,
                    eval_examples=eval_examples,
                    target_stage=target_stage,
                    target_position_role=target_position_role,
                    rank=int(rank),
                    predicted_vectors_by_kind=predicted_vectors_by_kind,
                    target_basis=target_basis,
                    value_token_ids=value_token_ids_cpu.to(device),
                    vocab=vocab,
                )
            )
        print(
            f"[value-code-transfer-rescue] finished step={path_step}",
            flush=True,
        )
    summary_rows = _summarize_rescue_rows(rows)
    rows_path = output_dir / "value_code_transfer_rescue_rows.jsonl"
    summary_rows_path = output_dir / "value_code_transfer_rescue_summary_rows.jsonl"
    subspace_rows_path = output_dir / "value_code_transfer_rescue_subspaces.jsonl"
    pair_rows_path = output_dir / "value_code_transfer_rescue_pairs.jsonl"
    report_path = output_dir / "value_code_transfer_rescue_report.json"
    markdown_path = output_dir / "value_code_transfer_rescue_report.md"
    write_jsonl(rows_path, rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(subspace_rows_path, subspace_rows)
    write_jsonl(pair_rows_path, pair_rows)
    report = {
        "schema_version": VALUE_CODE_TRANSFER_RESCUE_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "probe_metadata": probe_metadata,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoints": [str(path) for path in checkpoints],
        "output_dir": str(output_dir),
        "device": device_name,
        "num_probe_records": len(probe_records),
        "batch_size": resolved_batch_size,
        "source_stage": source_stage,
        "target_stage": target_stage,
        "source_position_role": source_position_role,
        "target_position_role": target_position_role,
        "context_stage": context_stage,
        "context_position_role": context_position_role,
        "context_rank": context_rank,
        "group_by": group_by,
        "split_filter": split_filter,
        "max_records": max_records,
        "basis_ranks": [int(rank) for rank in basis_ranks],
        "controls": controls,
        "fit_fraction": fit_fraction,
        "ridge_lambda": ridge_lambda,
        "random_seed": random_seed,
        "markdown_top_k_rows": markdown_top_k_rows,
        "rows_path": str(rows_path),
        "summary_rows_path": str(summary_rows_path),
        "subspace_rows_path": str(subspace_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(
        f"[value-code-transfer-rescue] complete report={report_path} rows={rows_path}",
        flush=True,
    )
    return report_path, markdown_path, rows_path, summary_rows_path, subspace_rows_path, pair_rows_path
