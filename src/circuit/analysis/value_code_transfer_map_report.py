from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.contextual_svd_alignment import CONTEXTUAL_GROUP_BY_OPTIONS, _role_subspace
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    GEOMETRY_POSITION_ROLES,
    _checkpoint_step_from_path,
    _intervention_positions_for_query,
    _positions_for_query,
)
from circuit.analysis.output_route_closure import _mean
from circuit.analysis.value_code_subspace_report import (
    _group_token_id,
    _make_probe_loader,
    _resolve_checkpoint_paths,
    _stage_lens_logits,
    _token_label,
    _valid_residual_stages,
    _value_logit_metrics,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


VALUE_CODE_TRANSFER_MAP_SCHEMA_VERSION = 1
VALUE_CODE_TRANSFER_CONTROLS = [
    "shuffled_answer_value",
    "wrong_support_value",
    "key_identity",
    "random_subspace",
]


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / "value_code_transfer_map_report.json",
        output_dir / "value_code_transfer_map_report.md",
        output_dir / "value_code_transfer_map_rows.jsonl",
        output_dir / "value_code_transfer_map_summary_rows.jsonl",
        output_dir / "value_code_transfer_map_subspaces.jsonl",
        output_dir / "value_code_transfer_map_pairs.jsonl",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing value-code-transfer-map outputs without --overwrite: "
            f"{[str(path) for path in existing]}"
        )


def _validate_basis_rank(rank: int) -> None:
    if rank <= 0:
        raise ValueError(f"basis rank must be positive, got {rank}.")


def _deterministic_fit_split(*, example: dict[str, Any], fit_fraction: float) -> str:
    if not (0.0 < fit_fraction < 1.0):
        raise ValueError(f"fit_fraction must be in (0, 1), got {fit_fraction}.")
    key = f"{example['sample_id']}:{example['query_index']}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big") / float(2**64 - 1)
    return "fit" if bucket < fit_fraction else "eval"


def _mean_selected(stage_tensor: torch.Tensor, *, row: int, positions: list[int]) -> torch.Tensor:
    if not positions:
        raise ValueError("positions must not be empty.")
    return torch.stack([stage_tensor[row, int(position), :].float() for position in positions], dim=0).mean(dim=0)


def _causal_value_distractor_positions(
    *,
    record: dict[str, Any],
    query_index: int,
    prediction_position: int,
) -> list[int]:
    query_geometry = _positions_for_query(record, query_index, prediction_position)
    positions = sorted(
        {
            int(position)
            for position in query_geometry["value_distractors"]
            if int(position) <= int(prediction_position)
        }
    )
    if not positions:
        raise RuntimeError(
            f"No causal-prefix value distractor positions for {record['sample_id']} query {query_index}."
        )
    return positions


def _project(vector: torch.Tensor, basis: torch.Tensor, *, label: str) -> torch.Tensor:
    if vector.ndim != 1:
        raise ValueError(f"{label} vector must be rank-1, got shape {tuple(vector.shape)}.")
    if basis.ndim != 2:
        raise ValueError(f"{label} basis must be rank-2, got shape {tuple(basis.shape)}.")
    if int(vector.numel()) != int(basis.size(0)):
        raise ValueError(f"{label} vector dim {vector.numel()} does not match basis dim {basis.size(0)}.")
    return vector.float().matmul(basis.float())


def _reconstruct(coords: torch.Tensor, basis: torch.Tensor, *, label: str) -> torch.Tensor:
    if coords.ndim != 1:
        raise ValueError(f"{label} coords must be rank-1, got shape {tuple(coords.shape)}.")
    if basis.ndim != 2:
        raise ValueError(f"{label} basis must be rank-2, got shape {tuple(basis.shape)}.")
    if int(coords.numel()) != int(basis.size(1)):
        raise ValueError(f"{label} coords dim {coords.numel()} does not match basis rank {basis.size(1)}.")
    return basis.float().matmul(coords.float())


def _fit_affine_map(
    *,
    source_coords: torch.Tensor,
    target_coords: torch.Tensor,
    ridge_lambda: float,
    label: str,
) -> dict[str, torch.Tensor]:
    if ridge_lambda < 0.0:
        raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda}.")
    if source_coords.ndim != 2 or target_coords.ndim != 2:
        raise ValueError(
            f"{label} source/target coords must be rank-2, got {tuple(source_coords.shape)} and {tuple(target_coords.shape)}."
        )
    if source_coords.size(0) != target_coords.size(0):
        raise ValueError(f"{label} source/target row mismatch: {source_coords.size(0)} vs {target_coords.size(0)}.")
    if source_coords.size(0) <= max(source_coords.size(1), target_coords.size(1)):
        raise RuntimeError(
            f"{label} needs more fit rows than coordinate dimensions; got rows={source_coords.size(0)}, "
            f"source_dim={source_coords.size(1)}, target_dim={target_coords.size(1)}."
        )
    source_mean = source_coords.float().mean(dim=0)
    target_mean = target_coords.float().mean(dim=0)
    source_centered = source_coords.float() - source_mean
    target_centered = target_coords.float() - target_mean
    identity = torch.eye(int(source_centered.size(1)), dtype=source_centered.dtype)
    normal = source_centered.T.matmul(source_centered) + float(ridge_lambda) * identity
    rhs = source_centered.T.matmul(target_centered)
    weights = torch.linalg.solve(normal, rhs)
    return {"weights": weights, "source_mean": source_mean, "target_mean": target_mean}


def _predict_affine(*, coords: torch.Tensor, transfer: dict[str, torch.Tensor], label: str) -> torch.Tensor:
    if coords.ndim != 1:
        raise ValueError(f"{label} coords must be rank-1, got shape {tuple(coords.shape)}.")
    weights = transfer["weights"]
    source_mean = transfer["source_mean"]
    target_mean = transfer["target_mean"]
    if int(coords.numel()) != int(source_mean.numel()):
        raise ValueError(f"{label} coords dim {coords.numel()} does not match source mean dim {source_mean.numel()}.")
    return (coords.float() - source_mean).matmul(weights) + target_mean


def _squared_error(left: torch.Tensor, right: torch.Tensor, *, label: str) -> float:
    if left.shape != right.shape:
        raise ValueError(f"{label} shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}.")
    return float(((left.float() - right.float()) ** 2).sum().item())


def _cosine(left: torch.Tensor, right: torch.Tensor, *, label: str) -> float:
    if left.shape != right.shape:
        raise ValueError(f"{label} shape mismatch: {tuple(left.shape)} vs {tuple(right.shape)}.")
    left_norm = left.float().norm()
    right_norm = right.float().norm()
    if float(left_norm.item()) <= 0.0 or float(right_norm.item()) <= 0.0:
        raise RuntimeError(f"Cannot compute cosine with zero vector for {label}.")
    return float(left.float().dot(right.float()).div(left_norm * right_norm).item())


def _random_basis(*, ambient_dim: int, rank: int, seed: int, label: str) -> torch.Tensor:
    if rank <= 0:
        raise ValueError(f"{label} rank must be positive, got {rank}.")
    if rank > ambient_dim:
        raise ValueError(f"{label} rank {rank} exceeds ambient_dim {ambient_dim}.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn((ambient_dim, rank), generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q[:, :rank].contiguous()


def _token_centroids(
    *,
    examples: list[dict[str, Any]],
    coords_by_index: dict[int, torch.Tensor],
    token_field: str,
    label: str,
) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = defaultdict(list)
    for example in examples:
        index = int(example["example_index"])
        token_id = int(example[token_field])
        grouped[token_id].append(coords_by_index[index].float())
    if len(grouped) < 2:
        raise RuntimeError(f"{label} requires at least two token centroids, got {len(grouped)}.")
    return {token_id: torch.stack(vectors, dim=0).mean(dim=0) for token_id, vectors in grouped.items()}


def _centroid_prediction(
    *,
    predicted_coords: torch.Tensor,
    true_token_id: int,
    centroids: dict[int, torch.Tensor],
    vocab: Vocabulary,
    label: str,
) -> dict[str, Any]:
    if true_token_id not in centroids:
        return {
            "centroid_scored": False,
            "centroid_prediction_token_id": None,
            "centroid_prediction_token": None,
            "centroid_prediction_correct": None,
            "centroid_true_cosine": None,
            "centroid_best_wrong_cosine": None,
            "centroid_cosine_margin": None,
        }
    scores = [
        (token_id, _cosine(predicted_coords, centroid, label=f"{label}/{token_id}"))
        for token_id, centroid in centroids.items()
    ]
    if len(scores) < 2:
        raise RuntimeError(f"{label} needs at least two centroid scores.")
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    predicted_token_id = int(scores[0][0])
    true_score = next(score for token_id, score in scores if int(token_id) == int(true_token_id))
    wrong_scores = [score for token_id, score in scores if int(token_id) != int(true_token_id)]
    if not wrong_scores:
        raise RuntimeError(f"{label} has no wrong centroid scores.")
    best_wrong_score = max(wrong_scores)
    return {
        "centroid_scored": True,
        "centroid_prediction_token_id": predicted_token_id,
        "centroid_prediction_token": _token_label(vocab, predicted_token_id),
        "centroid_prediction_correct": float(predicted_token_id == int(true_token_id)),
        "centroid_true_cosine": float(true_score),
        "centroid_best_wrong_cosine": float(best_wrong_score),
        "centroid_cosine_margin": float(true_score - best_wrong_score),
    }


def _collect_examples(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    loader: DataLoader[Any],
    source_stage: str,
    target_stage: str,
    source_position_role: str,
    target_position_role: str,
    group_by: str,
    fit_fraction: float,
    vocab: Vocabulary,
    value_token_ids: torch.Tensor,
    device: torch.device,
    context_stage: str | None = None,
    context_position_role: str | None = None,
) -> list[dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()
    payload_step = int(checkpoint["step"])
    path_step = _checkpoint_step_from_path(checkpoint_path)
    if payload_step != path_step:
        raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={payload_step} path={path_step}.")
    examples: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            batch = move_batch_to_device(raw_batch, device)
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], return_residual_streams=True)
            if outputs.residual_streams is None:
                raise RuntimeError("value-code-transfer-map-report requires residual streams.")
            requested_stages = [source_stage, target_stage]
            if context_stage is not None:
                requested_stages.append(context_stage)
            missing_stages = [stage for stage in requested_stages if stage not in outputs.residual_streams]
            if missing_stages:
                raise KeyError(
                    f"Missing residual stage(s) {missing_stages}. Available stages: {sorted(outputs.residual_streams)}"
                )
            answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
            source_tensor = outputs.residual_streams[source_stage]
            target_tensor = outputs.residual_streams[target_stage]
            context_tensor = None if context_stage is None else outputs.residual_streams[context_stage]
            for flat_index in range(int(metadata["rows"].numel())):
                query_batch_row = int(metadata["rows"][flat_index].item())
                query_index = int(metadata["query_indices"][flat_index].item())
                record = batch["records"][query_batch_row]
                source_batch_row, source_positions = _intervention_positions_for_query(
                    batch=batch,
                    metadata=metadata,
                    flat_index=flat_index,
                    position_role=source_position_role,
                )
                target_batch_row, target_positions = _intervention_positions_for_query(
                    batch=batch,
                    metadata=metadata,
                    flat_index=flat_index,
                    position_role=target_position_role,
                )
                context_batch_row = None
                context_positions = None
                if context_position_role is not None:
                    context_batch_row, context_positions = _intervention_positions_for_query(
                        batch=batch,
                        metadata=metadata,
                        flat_index=flat_index,
                        position_role=context_position_role,
                    )
                key_batch_row, key_positions = _intervention_positions_for_query(
                    batch=batch,
                    metadata=metadata,
                    flat_index=flat_index,
                    position_role="support_key",
                )
                prediction_position = int(metadata["prediction_positions"][flat_index].item())
                wrong_positions = _causal_value_distractor_positions(
                    record=record,
                    query_index=query_index,
                    prediction_position=prediction_position,
                )
                source_vector = _mean_selected(source_tensor, row=source_batch_row, positions=source_positions).detach().cpu()
                target_vector = _mean_selected(target_tensor, row=target_batch_row, positions=target_positions).detach().cpu()
                context_vector = None
                if context_tensor is not None:
                    if context_batch_row is None or context_positions is None:
                        raise RuntimeError("context tensor was requested but context positions were not resolved.")
                    context_vector = _mean_selected(
                        context_tensor,
                        row=context_batch_row,
                        positions=context_positions,
                    ).detach().cpu()
                wrong_source_vector = _mean_selected(
                    source_tensor,
                    row=query_batch_row,
                    positions=wrong_positions,
                ).detach().cpu()
                key_source_vector = _mean_selected(
                    source_tensor,
                    row=key_batch_row,
                    positions=key_positions,
                ).detach().cpu()
                answer_value_token_id = int(answer_targets[flat_index].item())
                source_group_token_id = _group_token_id(
                    group_by=group_by,
                    batch=batch,
                    metadata=metadata,
                    answer_targets=answer_targets,
                    flat_index=flat_index,
                    context_batch_row=source_batch_row,
                    context_position=int(source_positions[0]),
                )
                target_group_token_id = _group_token_id(
                    group_by=group_by,
                    batch=batch,
                    metadata=metadata,
                    answer_targets=answer_targets,
                    flat_index=flat_index,
                    context_batch_row=target_batch_row,
                    context_position=int(target_positions[0]),
                )
                key_group_token_id = _group_token_id(
                    group_by="support_key",
                    batch=batch,
                    metadata=metadata,
                    answer_targets=answer_targets,
                    flat_index=flat_index,
                    context_batch_row=key_batch_row,
                    context_position=int(key_positions[0]),
                )
                final_metrics = _value_logit_metrics(
                    logits=answer_logits[flat_index],
                    answer_token_id=answer_value_token_id,
                    value_token_ids=value_token_ids,
                    vocab=vocab,
                    label=f"{checkpoint_path.name}/{record['sample_id']}/{query_index}/final",
                )
                row = {
                    "schema_version": VALUE_CODE_TRANSFER_MAP_SCHEMA_VERSION,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_step": path_step,
                    "batch_index": batch_index,
                    "sample_id": str(record["sample_id"]),
                    "split": str(record["split"]),
                    "query_index": query_index,
                    "source_stage": source_stage,
                    "target_stage": target_stage,
                    "source_position_role": source_position_role,
                    "target_position_role": target_position_role,
                    "context_stage": context_stage,
                    "context_position_role": context_position_role,
                    "source_positions": [int(position) for position in source_positions],
                    "target_positions": [int(position) for position in target_positions],
                    "context_positions": None
                    if context_positions is None
                    else [int(position) for position in context_positions],
                    "group_by": group_by,
                    "source_group_token_id": int(source_group_token_id),
                    "source_group_token": _token_label(vocab, int(source_group_token_id)),
                    "target_group_token_id": int(target_group_token_id),
                    "target_group_token": _token_label(vocab, int(target_group_token_id)),
                    "key_group_token_id": int(key_group_token_id),
                    "key_group_token": _token_label(vocab, int(key_group_token_id)),
                    "answer_value_token_id": answer_value_token_id,
                    "answer_value_token": _token_label(vocab, answer_value_token_id),
                    "final_value_accuracy": final_metrics["value_accuracy"],
                    "final_value_margin": final_metrics["value_margin"],
                    "_source_vector": source_vector.float(),
                    "_target_vector": target_vector.float(),
                    "_wrong_source_vector": wrong_source_vector.float(),
                    "_key_source_vector": key_source_vector.float(),
                }
                if context_vector is not None:
                    row["_context_vector"] = context_vector.float()
                row["fit_split"] = _deterministic_fit_split(example=row, fit_fraction=fit_fraction)
                row["example_index"] = len(examples)
                examples.append(row)
    if not examples:
        raise RuntimeError(f"No examples collected for {checkpoint_path}.")
    fit_count = sum(1 for example in examples if example["fit_split"] == "fit")
    eval_count = sum(1 for example in examples if example["fit_split"] == "eval")
    if fit_count <= 0 or eval_count <= 0:
        raise RuntimeError(
            f"Fit/eval split produced fit={fit_count} eval={eval_count}; adjust --fit-fraction or probe records."
        )
    return examples


def _basis_from_examples(
    *,
    examples: list[dict[str, Any]],
    vector_field: str,
    token_field: str,
    role_label: str,
    context_role: str,
    group_by: str,
    vocab: Vocabulary,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vectors_by_token: dict[int, list[torch.Tensor]] = defaultdict(list)
    for example in examples:
        vectors_by_token[int(example[token_field])].append(example[vector_field].float())
    return _role_subspace(
        role_label=role_label,
        context_role=context_role,
        group_by=group_by,
        vectors_by_token=vectors_by_token,
        vocab=vocab,
        pca_rank=rank,
    )


def _evaluate_transfer(
    *,
    checkpoint_path: Path,
    examples: list[dict[str, Any]],
    eval_examples: list[dict[str, Any]],
    eval_kind: str,
    rank: int,
    source_basis: torch.Tensor,
    target_basis: torch.Tensor,
    transfer: dict[str, torch.Tensor],
    target_centroids: dict[int, torch.Tensor],
    eval_target_mean: torch.Tensor,
    source_coords_by_index: dict[int, torch.Tensor],
    target_coords_by_index: dict[int, torch.Tensor],
    model: torch.nn.Module,
    target_stage: str,
    value_token_ids: torch.Tensor,
    vocab: Vocabulary,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if eval_kind == "shuffled_answer_value" and len(eval_examples) < 2:
        raise RuntimeError("shuffled_answer_value control requires at least two eval examples.")
    shifted_source_coords: dict[int, torch.Tensor] = {}
    if eval_kind == "shuffled_answer_value":
        shifted = eval_examples[1:] + eval_examples[:1]
        for example, shifted_example in zip(eval_examples, shifted, strict=True):
            shifted_source_coords[int(example["example_index"])] = source_coords_by_index[int(shifted_example["example_index"])]
    for example in eval_examples:
        example_index = int(example["example_index"])
        if eval_kind == "shuffled_answer_value":
            source_coords = shifted_source_coords[example_index]
        else:
            source_coords = source_coords_by_index[example_index]
        target_coords = target_coords_by_index[example_index]
        predicted_coords = _predict_affine(
            coords=source_coords,
            transfer=transfer,
            label=f"{checkpoint_path.name}/{eval_kind}/{example['sample_id']}/{example['query_index']}",
        )
        actual_projected_vector = _reconstruct(
            target_coords,
            target_basis,
            label=f"{checkpoint_path.name}/{eval_kind}/{example_index}.actual_projected",
        )
        predicted_projected_vector = _reconstruct(
            predicted_coords,
            target_basis,
            label=f"{checkpoint_path.name}/{eval_kind}/{example_index}.predicted_projected",
        )
        model_device = next(model.parameters()).device
        predicted_logits = _stage_lens_logits(
            model=model,
            stage=target_stage,
            vector=predicted_projected_vector.to(model_device),
        )
        predicted_value_metrics = _value_logit_metrics(
            logits=predicted_logits,
            answer_token_id=int(example["answer_value_token_id"]),
            value_token_ids=value_token_ids,
            vocab=vocab,
            label=f"{checkpoint_path.name}/{eval_kind}/{example['sample_id']}/{example['query_index']}/predicted",
        )
        centroid = _centroid_prediction(
            predicted_coords=predicted_coords,
            true_token_id=int(example["target_group_token_id"]),
            centroids=target_centroids,
            vocab=vocab,
            label=f"{checkpoint_path.name}/{eval_kind}/{example['sample_id']}/{example['query_index']}.centroid",
        )
        coord_sse = _squared_error(
            predicted_coords,
            target_coords,
            label=f"{checkpoint_path.name}/{eval_kind}/{example_index}.coord_sse",
        )
        coord_sst = _squared_error(
            target_coords,
            eval_target_mean,
            label=f"{checkpoint_path.name}/{eval_kind}/{example_index}.coord_sst",
        )
        rows.append(
            {
                "schema_version": VALUE_CODE_TRANSFER_MAP_SCHEMA_VERSION,
                "checkpoint": str(checkpoint_path),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_step": _checkpoint_step_from_path(checkpoint_path),
                "basis_rank": rank,
                "eval_kind": eval_kind,
                "fit_split": example["fit_split"],
                "sample_id": example["sample_id"],
                "split": example["split"],
                "query_index": example["query_index"],
                "source_stage": example["source_stage"],
                "target_stage": example["target_stage"],
                "source_position_role": example["source_position_role"],
                "target_position_role": example["target_position_role"],
                "group_by": example["group_by"],
                "source_group_token_id": example["source_group_token_id"],
                "source_group_token": example["source_group_token"],
                "target_group_token_id": example["target_group_token_id"],
                "target_group_token": example["target_group_token"],
                "answer_value_token_id": example["answer_value_token_id"],
                "answer_value_token": example["answer_value_token"],
                "source_coord_norm": float(source_coords.float().norm().item()),
                "target_coord_norm": float(target_coords.float().norm().item()),
                "predicted_coord_norm": float(predicted_coords.float().norm().item()),
                "coord_squared_error": coord_sse,
                "coord_target_centered_squared_norm": coord_sst,
                "coord_cosine": _cosine(
                    predicted_coords,
                    target_coords,
                    label=f"{checkpoint_path.name}/{eval_kind}/{example_index}.coord_cosine",
                ),
                "projected_vector_cosine": _cosine(
                    predicted_projected_vector,
                    actual_projected_vector,
                    label=f"{checkpoint_path.name}/{eval_kind}/{example_index}.projected_vector_cosine",
                ),
                "predicted_stage_lens_value_accuracy": predicted_value_metrics["value_accuracy"],
                "predicted_stage_lens_value_margin": predicted_value_metrics["value_margin"],
                "predicted_stage_lens_correct_value_logit": predicted_value_metrics["correct_value_logit"],
                "predicted_stage_lens_best_value_token_id": predicted_value_metrics["best_value_token_id"],
                "predicted_stage_lens_best_value_token": predicted_value_metrics["best_value_token"],
                **centroid,
            }
        )
    return rows


def _summarize_transfer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["checkpoint_step"]),
                int(row["basis_rank"]),
                str(row["eval_kind"]),
                str(row["source_stage"]),
                str(row["target_stage"]),
                str(row["source_position_role"]),
                str(row["target_position_role"]),
            )
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        checkpoint_step, basis_rank, eval_kind, source_stage, target_stage, source_position_role, target_position_role = key
        scored = [row for row in group_rows if bool(row["centroid_scored"])]
        sse = sum(float(row["coord_squared_error"]) for row in group_rows)
        sst = sum(float(row["coord_target_centered_squared_norm"]) for row in group_rows)
        coord_r2 = None if sst <= 0.0 else 1.0 - sse / sst
        summary_rows.append(
            {
                "checkpoint_step": checkpoint_step,
                "basis_rank": basis_rank,
                "eval_kind": eval_kind,
                "source_stage": source_stage,
                "target_stage": target_stage,
                "source_position_role": source_position_role,
                "target_position_role": target_position_role,
                "num_rows": len(group_rows),
                "coord_r_squared": coord_r2,
                "mean_coord_cosine": _mean(
                    [float(row["coord_cosine"]) for row in group_rows],
                    label=f"{key}.coord_cosine",
                ),
                "mean_projected_vector_cosine": _mean(
                    [float(row["projected_vector_cosine"]) for row in group_rows],
                    label=f"{key}.projected_vector_cosine",
                ),
                "predicted_stage_lens_value_accuracy": _mean(
                    [float(row["predicted_stage_lens_value_accuracy"]) for row in group_rows],
                    label=f"{key}.stage_lens_accuracy",
                ),
                "mean_predicted_stage_lens_value_margin": _mean(
                    [float(row["predicted_stage_lens_value_margin"]) for row in group_rows],
                    label=f"{key}.stage_lens_margin",
                ),
                "centroid_scored_rows": len(scored),
                "centroid_accuracy": None
                if not scored
                else _mean(
                    [float(row["centroid_prediction_correct"]) for row in scored],
                    label=f"{key}.centroid_accuracy",
                ),
                "mean_centroid_cosine_margin": None
                if not scored
                else _mean(
                    [float(row["centroid_cosine_margin"]) for row in scored],
                    label=f"{key}.centroid_margin",
                ),
            }
        )
    return summary_rows


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    summary_rows = sorted(
        report["summary_rows"],
        key=lambda row: (
            int(row["checkpoint_step"]),
            int(row["basis_rank"]),
            str(row["eval_kind"]),
        ),
    )
    lines = [
        "# Value-Code Transfer Map Report",
        "",
        "This report asks whether value-code coordinates at a source position predict value-code coordinates at the prediction position.",
        "",
        "## Scope",
        "",
        f"- checkpoints: `{len(report['checkpoints'])}`",
        f"- records: `{report['num_probe_records']}`",
        f"- source: `{report['source_stage']} / {report['source_position_role']}`",
        f"- target: `{report['target_stage']} / {report['target_position_role']}`",
        f"- group-by: `{report['group_by']}`",
        f"- basis ranks: `{', '.join(str(rank) for rank in report['basis_ranks'])}`",
        f"- controls: `{', '.join(report['controls']) if report['controls'] else 'none'}`",
        f"- fit fraction: `{report['fit_fraction']}`",
        "",
        "## Calculation",
        "",
        "For each checkpoint and rank, the tool builds source and target identity bases on the deterministic fit split.",
        "It fits a ridge-stabilized affine map from source coordinates to target coordinates, then evaluates that map on held-out rows.",
        "",
        "Controls use the same held-out rows. `shuffled_answer_value` permutes source coordinates, `wrong_support_value` uses value-distractor residuals, `key_identity` fits a support-key-code map, and `random_subspace` fits a random-source-subspace map.",
        "",
        "## Transfer Summary",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| step | rank | eval | rows | coord R2 | coord cos | centroid acc | stage-lens acc | stage-lens margin |",
                "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary_rows[: int(report["markdown_top_k_rows"])]:
            coord_r2 = "n/a" if row["coord_r_squared"] is None else f"{float(row['coord_r_squared']):.4f}"
            centroid_acc = "n/a" if row["centroid_accuracy"] is None else f"{float(row['centroid_accuracy']):.4f}"
            lines.append(
                f"| {row['checkpoint_step']} | {row['basis_rank']} | `{row['eval_kind']}` | {row['num_rows']} | "
                f"{coord_r2} | {float(row['mean_coord_cosine']):.4f} | {centroid_acc} | "
                f"{float(row['predicted_stage_lens_value_accuracy']):.4f} | "
                f"{float(row['mean_predicted_stage_lens_value_margin']):.6g} |"
            )
    else:
        lines.append("No summary rows were produced.")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "A strong true-transfer row means source value-code coordinates predict prediction-position value-code coordinates better than controls. It does not by itself prove causal sufficiency; use a transfer-map patch/rescue experiment for that.",
            "",
            "## Outputs",
            "",
            f"- transfer rows: `{report['rows_path']}`",
            f"- summary rows: `{report['summary_rows_path']}`",
            f"- subspace rows: `{report['subspace_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _collect_checkpoint_transfer_rows(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    loader: DataLoader[Any],
    source_stage: str,
    target_stage: str,
    source_position_role: str,
    target_position_role: str,
    group_by: str,
    basis_ranks: list[int],
    controls: list[str],
    fit_fraction: float,
    ridge_lambda: float,
    random_seed: int,
    vocab: Vocabulary,
    value_token_ids: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    examples = _collect_examples(
        model=model,
        checkpoint_path=checkpoint_path,
        loader=loader,
        source_stage=source_stage,
        target_stage=target_stage,
        source_position_role=source_position_role,
        target_position_role=target_position_role,
        group_by=group_by,
        fit_fraction=fit_fraction,
        vocab=vocab,
        value_token_ids=value_token_ids,
        device=device,
    )
    fit_examples = [example for example in examples if example["fit_split"] == "fit"]
    eval_examples = [example for example in examples if example["fit_split"] == "eval"]
    rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for example in examples:
        pair_rows.append(
            {
                key: value
                for key, value in example.items()
                if not key.startswith("_")
            }
        )
    step = _checkpoint_step_from_path(checkpoint_path)
    for rank in basis_ranks:
        source_subspace, source_summary = _basis_from_examples(
            examples=fit_examples,
            vector_field="_source_vector",
            token_field="source_group_token_id",
            role_label=f"{checkpoint_path.name}:source:{source_stage}:{source_position_role}:{group_by}:rank{rank}",
            context_role=source_position_role,
            group_by=group_by,
            vocab=vocab,
            rank=rank,
        )
        target_subspace, target_summary = _basis_from_examples(
            examples=fit_examples,
            vector_field="_target_vector",
            token_field="target_group_token_id",
            role_label=f"{checkpoint_path.name}:target:{target_stage}:{target_position_role}:{group_by}:rank{rank}",
            context_role=target_position_role,
            group_by=group_by,
            vocab=vocab,
            rank=rank,
        )
        source_basis = source_subspace["identity_basis"].float()
        target_basis = target_subspace["identity_basis"].float()
        subspace_rows.extend(
            [
                {
                    "schema_version": VALUE_CODE_TRANSFER_MAP_SCHEMA_VERSION,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_step": step,
                    "basis_rank": rank,
                    "basis_role": "source_value_identity",
                    **source_summary,
                },
                {
                    "schema_version": VALUE_CODE_TRANSFER_MAP_SCHEMA_VERSION,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_step": step,
                    "basis_rank": rank,
                    "basis_role": "target_value_identity",
                    **target_summary,
                },
            ]
        )
        source_coords = {
            int(example["example_index"]): _project(
                example["_source_vector"],
                source_basis,
                label=f"{checkpoint_path.name}/{rank}/{example['example_index']}.source",
            )
            for example in examples
        }
        target_coords = {
            int(example["example_index"]): _project(
                example["_target_vector"],
                target_basis,
                label=f"{checkpoint_path.name}/{rank}/{example['example_index']}.target",
            )
            for example in examples
        }
        fit_source_matrix = torch.stack([source_coords[int(example["example_index"])] for example in fit_examples], dim=0)
        fit_target_matrix = torch.stack([target_coords[int(example["example_index"])] for example in fit_examples], dim=0)
        transfer = _fit_affine_map(
            source_coords=fit_source_matrix,
            target_coords=fit_target_matrix,
            ridge_lambda=ridge_lambda,
            label=f"{checkpoint_path.name}/rank{rank}/true_transfer",
        )
        target_centroids = _token_centroids(
            examples=fit_examples,
            coords_by_index=target_coords,
            token_field="target_group_token_id",
            label=f"{checkpoint_path.name}/rank{rank}/target_centroids",
        )
        eval_target_mean = torch.stack([target_coords[int(example["example_index"])] for example in eval_examples], dim=0).mean(dim=0)
        rows.extend(
            _evaluate_transfer(
                checkpoint_path=checkpoint_path,
                examples=examples,
                eval_examples=eval_examples,
                eval_kind="true_transfer",
                rank=rank,
                source_basis=source_basis,
                target_basis=target_basis,
                transfer=transfer,
                target_centroids=target_centroids,
                eval_target_mean=eval_target_mean,
                source_coords_by_index=source_coords,
                target_coords_by_index=target_coords,
                model=model,
                target_stage=target_stage,
                value_token_ids=value_token_ids,
                vocab=vocab,
            )
        )
        if "shuffled_answer_value" in controls:
            rows.extend(
                _evaluate_transfer(
                    checkpoint_path=checkpoint_path,
                    examples=examples,
                    eval_examples=eval_examples,
                    eval_kind="shuffled_answer_value",
                    rank=rank,
                    source_basis=source_basis,
                    target_basis=target_basis,
                    transfer=transfer,
                    target_centroids=target_centroids,
                    eval_target_mean=eval_target_mean,
                    source_coords_by_index=source_coords,
                    target_coords_by_index=target_coords,
                    model=model,
                    target_stage=target_stage,
                    value_token_ids=value_token_ids,
                    vocab=vocab,
                )
            )
        if "wrong_support_value" in controls:
            wrong_source_coords = {
                int(example["example_index"]): _project(
                    example["_wrong_source_vector"],
                    source_basis,
                    label=f"{checkpoint_path.name}/{rank}/{example['example_index']}.wrong_source",
                )
                for example in examples
            }
            rows.extend(
                _evaluate_transfer(
                    checkpoint_path=checkpoint_path,
                    examples=examples,
                    eval_examples=eval_examples,
                    eval_kind="wrong_support_value",
                    rank=rank,
                    source_basis=source_basis,
                    target_basis=target_basis,
                    transfer=transfer,
                    target_centroids=target_centroids,
                    eval_target_mean=eval_target_mean,
                    source_coords_by_index=wrong_source_coords,
                    target_coords_by_index=target_coords,
                    model=model,
                    target_stage=target_stage,
                    value_token_ids=value_token_ids,
                    vocab=vocab,
                )
            )
        if "key_identity" in controls:
            key_subspace, key_summary = _basis_from_examples(
                examples=fit_examples,
                vector_field="_key_source_vector",
                token_field="key_group_token_id",
                role_label=f"{checkpoint_path.name}:key_control:{source_stage}:support_key:support_key:rank{rank}",
                context_role="support_key",
                group_by="support_key",
                vocab=vocab,
                rank=rank,
            )
            key_basis = key_subspace["identity_basis"].float()
            subspace_rows.append(
                {
                    "schema_version": VALUE_CODE_TRANSFER_MAP_SCHEMA_VERSION,
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
                    label=f"{checkpoint_path.name}/{rank}/{example['example_index']}.key_source",
                )
                for example in examples
            }
            key_fit_source = torch.stack([key_coords[int(example["example_index"])] for example in fit_examples], dim=0)
            key_transfer = _fit_affine_map(
                source_coords=key_fit_source,
                target_coords=fit_target_matrix,
                ridge_lambda=ridge_lambda,
                label=f"{checkpoint_path.name}/rank{rank}/key_identity",
            )
            rows.extend(
                _evaluate_transfer(
                    checkpoint_path=checkpoint_path,
                    examples=examples,
                    eval_examples=eval_examples,
                    eval_kind="key_identity",
                    rank=rank,
                    source_basis=key_basis,
                    target_basis=target_basis,
                    transfer=key_transfer,
                    target_centroids=target_centroids,
                    eval_target_mean=eval_target_mean,
                    source_coords_by_index=key_coords,
                    target_coords_by_index=target_coords,
                    model=model,
                    target_stage=target_stage,
                    value_token_ids=value_token_ids,
                    vocab=vocab,
                )
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
                    "schema_version": VALUE_CODE_TRANSFER_MAP_SCHEMA_VERSION,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_step": step,
                    "basis_rank": rank,
                    "basis_role": "random_source_control",
                    "role": f"{checkpoint_path.name}:random_source:rank{rank}",
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
                    label=f"{checkpoint_path.name}/{rank}/{example['example_index']}.random_source",
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
                label=f"{checkpoint_path.name}/rank{rank}/random_subspace",
            )
            rows.extend(
                _evaluate_transfer(
                    checkpoint_path=checkpoint_path,
                    examples=examples,
                    eval_examples=eval_examples,
                    eval_kind="random_subspace",
                    rank=rank,
                    source_basis=random_basis,
                    target_basis=target_basis,
                    transfer=random_transfer,
                    target_centroids=target_centroids,
                    eval_target_mean=eval_target_mean,
                    source_coords_by_index=random_coords,
                    target_coords_by_index=target_coords,
                    model=model,
                    target_stage=target_stage,
                    value_token_ids=value_token_ids,
                    vocab=vocab,
                )
            )
    return rows, subspace_rows, pair_rows


def run_value_code_transfer_map_report(
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
    value_token_ids = torch.tensor(vocab.value_token_ids, dtype=torch.long)
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
    unsupported_stages = sorted(set([source_stage, target_stage]) - set(valid_stages))
    if unsupported_stages:
        raise ValueError(f"Unsupported stages {unsupported_stages}; expected one of {valid_stages}.")
    resolved_batch_size = int(spec.evaluation.batch_size if batch_size is None else batch_size)
    loader = _make_probe_loader(
        probe_records=probe_records,
        batch_size=resolved_batch_size,
        pad_token_id=vocab.pad_token_id,
    )
    print(
        "[value-code-transfer-map-report] "
        f"checkpoints={len(checkpoints)} records={len(probe_records)} source={source_stage}/{source_position_role} "
        f"target={target_stage}/{target_position_role} group_by={group_by} ranks={basis_ranks} controls={controls} "
        f"device={device_name}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        print(
            f"[value-code-transfer-map-report] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        checkpoint_rows, checkpoint_subspace_rows, checkpoint_pair_rows = _collect_checkpoint_transfer_rows(
            model=model,
            checkpoint_path=checkpoint_path,
            loader=loader,
            source_stage=source_stage,
            target_stage=target_stage,
            source_position_role=source_position_role,
            target_position_role=target_position_role,
            group_by=group_by,
            basis_ranks=[int(rank) for rank in basis_ranks],
            controls=controls,
            fit_fraction=fit_fraction,
            ridge_lambda=ridge_lambda,
            random_seed=random_seed,
            vocab=vocab,
            value_token_ids=value_token_ids.to(device),
            device=device,
        )
        rows.extend(checkpoint_rows)
        subspace_rows.extend(checkpoint_subspace_rows)
        pair_rows.extend(checkpoint_pair_rows)
        true_rows = [row for row in checkpoint_rows if row["eval_kind"] == "true_transfer"]
        if true_rows:
            mean_cosine = _mean(
                [float(row["coord_cosine"]) for row in true_rows],
                label=f"{checkpoint_path.name}.true_transfer_coord_cosine",
            )
            print(
                f"[value-code-transfer-map-report] finished step={_checkpoint_step_from_path(checkpoint_path)} "
                f"true_transfer_coord_cosine={mean_cosine:.6g}",
                flush=True,
            )
        else:
            print(
                f"[value-code-transfer-map-report] finished step={_checkpoint_step_from_path(checkpoint_path)}",
                flush=True,
            )
    summary_rows = _summarize_transfer_rows(rows)
    rows_path = output_dir / "value_code_transfer_map_rows.jsonl"
    summary_rows_path = output_dir / "value_code_transfer_map_summary_rows.jsonl"
    subspace_rows_path = output_dir / "value_code_transfer_map_subspaces.jsonl"
    pair_rows_path = output_dir / "value_code_transfer_map_pairs.jsonl"
    report_path = output_dir / "value_code_transfer_map_report.json"
    markdown_path = output_dir / "value_code_transfer_map_report.md"
    write_jsonl(rows_path, rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(subspace_rows_path, subspace_rows)
    write_jsonl(pair_rows_path, pair_rows)
    report = {
        "schema_version": VALUE_CODE_TRANSFER_MAP_SCHEMA_VERSION,
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
        f"[value-code-transfer-map-report] complete report={report_path} rows={rows_path}",
        flush=True,
    )
    return report_path, markdown_path, rows_path, summary_rows_path, subspace_rows_path, pair_rows_path
