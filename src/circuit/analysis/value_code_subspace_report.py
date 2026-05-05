from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.contextual_svd_alignment import (
    CONTEXTUAL_GROUP_BY_OPTIONS,
    _abs_cosine,
    _role_subspace,
    _subspace_overlap,
)
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    GEOMETRY_POSITION_ROLES,
    _checkpoint_step_from_path,
    _intervention_positions_for_query,
)
from circuit.analysis.output_route_closure import _mean, _safe_r_squared
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


VALUE_CODE_SUBSPACE_SCHEMA_VERSION = 1


def _resolve_checkpoint_paths(*, checkpoint_dir: Path, checkpoint_paths: list[Path] | None) -> list[Path]:
    if checkpoint_paths is None:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        resolved = sorted(checkpoint_dir.glob("step_*.pt"), key=_checkpoint_step_from_path)
    else:
        resolved = [Path(path) for path in checkpoint_paths]
    if not resolved:
        raise FileNotFoundError(f"No checkpoints provided or found in {checkpoint_dir}.")
    missing = [path for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint path(s): {[str(path) for path in missing]}")
    return sorted(resolved, key=_checkpoint_step_from_path)


def _valid_residual_stages(num_layers: int) -> list[str]:
    stages = ["embedding"]
    for layer_index in range(num_layers):
        stages.extend([f"layer_{layer_index}_post_attn", f"layer_{layer_index}_post_mlp"])
    stages.append("final_norm")
    return stages


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / "value_code_subspace_report.json",
        output_dir / "value_code_subspace_report.md",
        output_dir / "value_code_subspace_rows.jsonl",
        output_dir / "value_code_subspace_summary_rows.jsonl",
        output_dir / "value_code_subspace_subspaces.jsonl",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing value-code-subspace outputs without --overwrite: "
            f"{[str(path) for path in existing]}"
        )


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


def _token_label(vocab: Vocabulary, token_id: int) -> str:
    if token_id < 0 or token_id >= len(vocab.tokens):
        raise ValueError(f"Token id {token_id} outside vocabulary size {len(vocab.tokens)}.")
    return str(vocab.tokens[token_id])


def _group_token_id(
    *,
    group_by: str,
    batch: dict[str, Any],
    metadata: dict[str, torch.Tensor],
    answer_targets: torch.Tensor,
    flat_index: int,
    context_batch_row: int,
    context_position: int,
) -> int:
    query_batch_row = int(metadata["rows"][flat_index].item())
    if group_by == "position_token":
        return int(batch["input_ids"][context_batch_row, context_position].item())
    if group_by == "query_key":
        return int(batch["input_ids"][query_batch_row, int(metadata["query_key_positions"][flat_index].item())].item())
    if group_by == "support_value":
        return int(
            batch["input_ids"][query_batch_row, int(metadata["support_value_positions"][flat_index].item())].item()
        )
    if group_by == "answer_value":
        return int(answer_targets[flat_index].item())
    if group_by == "support_key":
        support_batch_row, support_positions = _intervention_positions_for_query(
            batch=batch,
            metadata=metadata,
            flat_index=flat_index,
            position_role="support_key",
        )
        if support_batch_row != query_batch_row or len(support_positions) != 1:
            record = batch["records"][query_batch_row]
            query_index = int(metadata["query_indices"][flat_index].item())
            raise RuntimeError(
                f"Expected exactly one support_key position for {record['sample_id']} query {query_index}, "
                f"got row={support_batch_row} positions={support_positions}."
            )
        return int(batch["input_ids"][support_batch_row, support_positions[0]].item())
    raise ValueError(f"Unhandled group_by mode: {group_by}")


def _stage_lens_logits(*, model: torch.nn.Module, stage: str, vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1:
        raise ValueError(f"Expected a rank-1 residual vector, got shape {tuple(vector.shape)}.")
    hidden = vector if stage == "final_norm" else model.final_norm(vector)
    return model.lm_head(hidden)


def _value_logit_metrics(
    *,
    logits: torch.Tensor,
    answer_token_id: int,
    value_token_ids: torch.Tensor,
    vocab: Vocabulary,
    label: str,
) -> dict[str, Any]:
    if logits.ndim != 1:
        raise ValueError(f"{label} logits must be rank-1, got shape {tuple(logits.shape)}.")
    matches = (value_token_ids == int(answer_token_id)).nonzero(as_tuple=False)
    if int(matches.size(0)) != 1:
        raise RuntimeError(f"{label} answer token {_token_label(vocab, answer_token_id)} not found exactly once in value ids.")
    answer_value_index = int(matches[0].item())
    value_logits = logits.index_select(dim=0, index=value_token_ids.to(logits.device))
    correct_logit = float(value_logits[answer_value_index].item())
    masked = value_logits.clone()
    masked[answer_value_index] = torch.finfo(masked.dtype).min
    best_wrong_value_logit, best_wrong_value_index_tensor = masked.max(dim=0)
    best_value_logit, best_value_index_tensor = value_logits.max(dim=0)
    best_wrong_value_index = int(best_wrong_value_index_tensor.item())
    best_value_index = int(best_value_index_tensor.item())
    best_value_token_id = int(value_token_ids[best_value_index].item())
    best_wrong_value_token_id = int(value_token_ids[best_wrong_value_index].item())
    return {
        "correct_value_logit": correct_logit,
        "best_wrong_value_logit": float(best_wrong_value_logit.item()),
        "value_margin": correct_logit - float(best_wrong_value_logit.item()),
        "best_value_logit": float(best_value_logit.item()),
        "best_value_token_id": best_value_token_id,
        "best_value_token": _token_label(vocab, best_value_token_id),
        "best_wrong_value_token_id": best_wrong_value_token_id,
        "best_wrong_value_token": _token_label(vocab, best_wrong_value_token_id),
        "value_accuracy": float(best_value_token_id == int(answer_token_id)),
    }


def _normalize(vector: torch.Tensor, *, label: str) -> torch.Tensor:
    vector = vector.float()
    norm = vector.norm()
    if float(norm.item()) <= 0.0:
        raise RuntimeError(f"Cannot normalize zero vector: {label}")
    return vector / norm


def _leave_one_out_centroid_predictions(
    *,
    payloads: list[dict[str, Any]],
    vocab: Vocabulary,
    label: str,
) -> dict[int, dict[str, Any]]:
    if not payloads:
        raise RuntimeError(f"{label} requires at least one payload.")
    by_token: dict[int, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for payload_index, payload in enumerate(payloads):
        token_id = int(payload["group_token_id"])
        vector = payload["_vector"].float()
        by_token[token_id].append((payload_index, vector))
    if len(by_token) < 2:
        raise RuntimeError(f"{label} requires at least two group tokens, got {len(by_token)}.")
    token_ids = sorted(by_token)
    predictions: dict[int, dict[str, Any]] = {}
    for payload_index, payload in enumerate(payloads):
        true_token_id = int(payload["group_token_id"])
        vector_unit = _normalize(payload["_vector"], label=f"{label}/{payload_index}.vector")
        scores: list[tuple[int, float]] = []
        for token_id in token_ids:
            members = by_token[token_id]
            if token_id == true_token_id:
                if len(members) < 2:
                    continue
                centroid = torch.stack(
                    [member_vector for member_index, member_vector in members if member_index != payload_index],
                    dim=0,
                ).mean(dim=0)
            else:
                centroid = torch.stack([member_vector for _, member_vector in members], dim=0).mean(dim=0)
            centroid_unit = _normalize(centroid, label=f"{label}/{payload_index}/{token_id}.centroid")
            scores.append((token_id, float(vector_unit.dot(centroid_unit).item())))
        if true_token_id not in {token_id for token_id, _ in scores}:
            predictions[payload_index] = {
                "centroid_scored": False,
                "centroid_prediction_token_id": None,
                "centroid_prediction_token": None,
                "centroid_prediction_correct": None,
                "centroid_true_cosine": None,
                "centroid_best_wrong_cosine": None,
                "centroid_cosine_margin": None,
            }
            continue
        sorted_scores = sorted(scores, key=lambda item: item[1], reverse=True)
        predicted_token_id = int(sorted_scores[0][0])
        true_score = next(score for token_id, score in sorted_scores if token_id == true_token_id)
        wrong_scores = [score for token_id, score in sorted_scores if token_id != true_token_id]
        if not wrong_scores:
            raise RuntimeError(f"{label}/{payload_index} has no wrong-token centroid scores.")
        best_wrong_score = float(max(wrong_scores))
        predictions[payload_index] = {
            "centroid_scored": True,
            "centroid_prediction_token_id": predicted_token_id,
            "centroid_prediction_token": _token_label(vocab, predicted_token_id),
            "centroid_prediction_correct": float(predicted_token_id == true_token_id),
            "centroid_true_cosine": float(true_score),
            "centroid_best_wrong_cosine": best_wrong_score,
            "centroid_cosine_margin": float(true_score - best_wrong_score),
        }
    return predictions


def _summarize_rows(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["checkpoint_step"]), str(row["stage"]), str(row["position_role"]), str(row["group_by"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (checkpoint_step, stage, position_role, group_by), group_rows in sorted(grouped.items()):
        vector_norms = [float(row["vector_norm"]) for row in group_rows]
        identity = [float(row["identity_overlap"]) for row in group_rows]
        all_vector = [float(row["all_vector_overlap"]) for row in group_rows]
        mean_cosines = [float(row["mean_abs_cosine"]) for row in group_rows]
        stage_margins = [float(row["stage_lens_value_margin"]) for row in group_rows]
        final_margins = [float(row["final_value_margin"]) for row in group_rows]
        stage_accuracies = [float(row["stage_lens_value_accuracy"]) for row in group_rows]
        final_accuracies = [float(row["final_value_accuracy"]) for row in group_rows]
        scored_centroid_rows = [row for row in group_rows if bool(row["centroid_scored"])]
        token_counts: dict[int, int] = defaultdict(int)
        for row in group_rows:
            token_counts[int(row["group_token_id"])] += 1
        if not token_counts:
            raise RuntimeError(f"No token counts for {checkpoint_step}/{stage}/{position_role}/{group_by}.")
        summary = {
            "checkpoint_step": checkpoint_step,
            "stage": stage,
            "position_role": position_role,
            "group_by": group_by,
            "num_rows": len(group_rows),
            "num_unique_groups": len(token_counts),
            "min_group_count": min(token_counts.values()),
            "max_group_count": max(token_counts.values()),
            "mean_vector_norm": _mean(vector_norms, label=f"{stage}/{position_role}/{group_by}.vector_norm"),
            "mean_identity_overlap": _mean(identity, label=f"{stage}/{position_role}/{group_by}.identity"),
            "mean_all_vector_overlap": _mean(all_vector, label=f"{stage}/{position_role}/{group_by}.all_vector"),
            "mean_mean_abs_cosine": _mean(mean_cosines, label=f"{stage}/{position_role}/{group_by}.mean_cosine"),
            "stage_lens_value_accuracy": _mean(
                stage_accuracies,
                label=f"{stage}/{position_role}/{group_by}.stage_lens_value_accuracy",
            ),
            "final_value_accuracy": _mean(
                final_accuracies,
                label=f"{stage}/{position_role}/{group_by}.final_value_accuracy",
            ),
            "mean_stage_lens_value_margin": _mean(
                stage_margins,
                label=f"{stage}/{position_role}/{group_by}.stage_lens_value_margin",
            ),
            "mean_final_value_margin": _mean(final_margins, label=f"{stage}/{position_role}/{group_by}.final_value_margin"),
            "stage_lens_margin_r_squared_vs_final_margin": _safe_r_squared(final_margins, stage_margins),
            "identity_overlap_r_squared_vs_stage_lens_margin": _safe_r_squared(stage_margins, identity),
        }
        if scored_centroid_rows:
            centroid_correct = [float(row["centroid_prediction_correct"]) for row in scored_centroid_rows]
            centroid_margins = [float(row["centroid_cosine_margin"]) for row in scored_centroid_rows]
            summary.update(
                {
                    "centroid_scored_rows": len(scored_centroid_rows),
                    "leave_one_out_centroid_accuracy": _mean(
                        centroid_correct,
                        label=f"{stage}/{position_role}/{group_by}.centroid_accuracy",
                    ),
                    "mean_centroid_cosine_margin": _mean(
                        centroid_margins,
                        label=f"{stage}/{position_role}/{group_by}.centroid_margin",
                    ),
                }
            )
        else:
            summary.update(
                {
                    "centroid_scored_rows": 0,
                    "leave_one_out_centroid_accuracy": None,
                    "mean_centroid_cosine_margin": None,
                }
            )
        summary_rows.append(summary)
    return summary_rows


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    summary_rows = sorted(
        report["summary_rows"],
        key=lambda row: (
            -1.0 if row["leave_one_out_centroid_accuracy"] is None else -float(row["leave_one_out_centroid_accuracy"]),
            -float(row["mean_identity_overlap"]),
        ),
    )[: int(report["markdown_top_k_rows"])]
    lines = [
        "# Value-Code Subspace Report",
        "",
        "This report asks where the residual stream carries a linearly readable value identity.",
        "",
        "## Scope",
        "",
        f"- checkpoints: `{len(report['checkpoints'])}`",
        f"- records: `{report['num_probe_records']}`",
        f"- stages: `{', '.join(report['stages'])}`",
        f"- position roles: `{', '.join(report['position_roles'])}`",
        f"- group-by labels: `{', '.join(report['group_by'])}`",
        f"- PCA rank: `{report['pca_rank']}`",
        "",
        "## Calculation",
        "",
        "For each checkpoint, stage, and position role, the tool collects clean residual vectors.",
        "It groups those vectors by a token identity such as `answer_value` or `support_value`, builds a strict PCA subspace over token means, and asks whether individual residual vectors lie in that identity subspace.",
        "",
        "The nearest-centroid score is leave-one-out inside each token group. Rows with singleton groups are not scored.",
        "",
        "## Top Value-Code Candidates",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| step | stage | position | group | rows | centroid acc | identity overlap | stage lens acc | stage lens margin |",
                "|---:|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary_rows:
            centroid = (
                "n/a"
                if row["leave_one_out_centroid_accuracy"] is None
                else f"{float(row['leave_one_out_centroid_accuracy']):.4f}"
            )
            lines.append(
                f"| {row['checkpoint_step']} | `{row['stage']}` | `{row['position_role']}` | `{row['group_by']}` | "
                f"{row['num_rows']} | {centroid} | {float(row['mean_identity_overlap']):.4f} | "
                f"{float(row['stage_lens_value_accuracy']):.4f} | {float(row['mean_stage_lens_value_margin']):.6g} |"
            )
    else:
        lines.append("No summary rows were produced.")
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "A strong value-code row means the residual vectors separate value identities at that stage and position. It does not by itself prove this code is causally sufficient for the answer logit; use residual rescue or projected patching for that.",
            "",
            "## Outputs",
            "",
            f"- value-code rows: `{report['rows_path']}`",
            f"- summary rows: `{report['summary_rows_path']}`",
            f"- subspace rows: `{report['subspace_rows_path']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _collect_checkpoint_rows(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    loader: DataLoader[Any],
    stages: list[str],
    position_roles: list[str],
    group_by_values: list[str],
    pca_rank: int,
    vocab: Vocabulary,
    value_token_ids: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    load_model_state(model, checkpoint["model_state"])
    model.eval()
    payload_step = int(checkpoint["step"])
    path_step = _checkpoint_step_from_path(checkpoint_path)
    if payload_step != path_step:
        raise RuntimeError(f"Checkpoint step mismatch for {checkpoint_path}: payload={payload_step} path={path_step}.")
    payloads_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {
        (stage, position_role, group_by): []
        for stage in stages
        for position_role in position_roles
        for group_by in group_by_values
    }
    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            batch = move_batch_to_device(raw_batch, device)
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], return_residual_streams=True)
            if outputs.residual_streams is None:
                raise RuntimeError("Value-code subspace report requires residual streams.")
            missing_stages = [stage for stage in stages if stage not in outputs.residual_streams]
            if missing_stages:
                raise KeyError(
                    f"Missing residual stage(s) {missing_stages}. Available stages: {sorted(outputs.residual_streams)}"
                )
            answer_logits, answer_targets, metadata = extract_answer_logits(outputs.logits, batch)
            for flat_index in range(int(metadata["rows"].numel())):
                query_batch_row = int(metadata["rows"][flat_index].item())
                query_index = int(metadata["query_indices"][flat_index].item())
                record = batch["records"][query_batch_row]
                answer_value_token_id = int(answer_targets[flat_index].item())
                query_key_token_id = int(
                    batch["input_ids"][
                        query_batch_row,
                        int(metadata["query_key_positions"][flat_index].item()),
                    ].item()
                )
                support_value_token_id = int(
                    batch["input_ids"][
                        query_batch_row,
                        int(metadata["support_value_positions"][flat_index].item()),
                    ].item()
                )
                final_metrics = _value_logit_metrics(
                    logits=answer_logits[flat_index],
                    answer_token_id=answer_value_token_id,
                    value_token_ids=value_token_ids,
                    vocab=vocab,
                    label=f"{checkpoint_path.name}/{record['sample_id']}/{query_index}/final",
                )
                for stage in stages:
                    stage_tensor = outputs.residual_streams[stage]
                    for position_role in position_roles:
                        context_batch_row, positions = _intervention_positions_for_query(
                            batch=batch,
                            metadata=metadata,
                            flat_index=flat_index,
                            position_role=position_role,
                        )
                        selected = torch.stack(
                            [stage_tensor[context_batch_row, int(position), :] for position in positions],
                            dim=0,
                        ).mean(dim=0)
                        stage_logits = _stage_lens_logits(model=model, stage=stage, vector=selected)
                        stage_metrics = _value_logit_metrics(
                            logits=stage_logits,
                            answer_token_id=answer_value_token_id,
                            value_token_ids=value_token_ids,
                            vocab=vocab,
                            label=f"{checkpoint_path.name}/{record['sample_id']}/{query_index}/{stage}/{position_role}",
                        )
                        vector = selected.detach().float().cpu().clone()
                        for group_by in group_by_values:
                            group_token_id = _group_token_id(
                                group_by=group_by,
                                batch=batch,
                                metadata=metadata,
                                answer_targets=answer_targets,
                                flat_index=flat_index,
                                context_batch_row=context_batch_row,
                                context_position=int(positions[0]),
                            )
                            payloads_by_key[(stage, position_role, group_by)].append(
                                {
                                    "schema_version": VALUE_CODE_SUBSPACE_SCHEMA_VERSION,
                                    "checkpoint": str(checkpoint_path),
                                    "checkpoint_name": checkpoint_path.name,
                                    "checkpoint_step": path_step,
                                    "batch_index": batch_index,
                                    "sample_id": str(record["sample_id"]),
                                    "split": str(record["split"]),
                                    "query_index": query_index,
                                    "stage": stage,
                                    "position_role": position_role,
                                    "selected_positions": [int(position) for position in positions],
                                    "num_positions": len(positions),
                                    "group_by": group_by,
                                    "group_token_id": group_token_id,
                                    "group_token": _token_label(vocab, group_token_id),
                                    "query_key_token_id": query_key_token_id,
                                    "query_key_token": _token_label(vocab, query_key_token_id),
                                    "support_value_token_id": support_value_token_id,
                                    "support_value_token": _token_label(vocab, support_value_token_id),
                                    "answer_value_token_id": answer_value_token_id,
                                    "answer_value_token": _token_label(vocab, answer_value_token_id),
                                    "vector_norm": float(vector.norm().item()),
                                    "final_correct_value_logit": final_metrics["correct_value_logit"],
                                    "final_best_wrong_value_logit": final_metrics["best_wrong_value_logit"],
                                    "final_value_margin": final_metrics["value_margin"],
                                    "final_best_value_token_id": final_metrics["best_value_token_id"],
                                    "final_best_value_token": final_metrics["best_value_token"],
                                    "final_value_accuracy": final_metrics["value_accuracy"],
                                    "stage_lens_correct_value_logit": stage_metrics["correct_value_logit"],
                                    "stage_lens_best_wrong_value_logit": stage_metrics["best_wrong_value_logit"],
                                    "stage_lens_value_margin": stage_metrics["value_margin"],
                                    "stage_lens_best_value_token_id": stage_metrics["best_value_token_id"],
                                    "stage_lens_best_value_token": stage_metrics["best_value_token"],
                                    "stage_lens_value_accuracy": stage_metrics["value_accuracy"],
                                    "_vector": vector,
                                }
                            )
    rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    for (stage, position_role, group_by), payloads in sorted(payloads_by_key.items()):
        if not payloads:
            raise RuntimeError(f"No vectors collected for {checkpoint_path.name}/{stage}/{position_role}/{group_by}.")
        vectors_by_token: dict[int, list[torch.Tensor]] = defaultdict(list)
        for payload in payloads:
            vectors_by_token[int(payload["group_token_id"])].append(payload["_vector"])
        subspace, summary = _role_subspace(
            role_label=f"{stage}:{position_role}:{group_by}",
            context_role=position_role,
            group_by=group_by,
            vectors_by_token=vectors_by_token,
            vocab=vocab,
            pca_rank=pca_rank,
        )
        subspace_rows.append(
            {
                "schema_version": VALUE_CODE_SUBSPACE_SCHEMA_VERSION,
                "checkpoint": str(checkpoint_path),
                "checkpoint_name": checkpoint_path.name,
                "checkpoint_step": path_step,
                "stage": stage,
                "position_role": position_role,
                "group_by": group_by,
                **summary,
            }
        )
        centroid_predictions = _leave_one_out_centroid_predictions(
            payloads=payloads,
            vocab=vocab,
            label=f"{checkpoint_path.name}/{stage}/{position_role}/{group_by}",
        )
        for payload_index, payload in enumerate(payloads):
            vector = payload.pop("_vector")
            row = dict(payload)
            row["mean_abs_cosine"] = _abs_cosine(
                vector,
                subspace["mean_direction"],
                label=f"{checkpoint_path.name}/{row['sample_id']}/{row['query_index']}/{stage}/{position_role}/{group_by}.mean",
            )
            row["identity_overlap"] = _subspace_overlap(
                vector,
                subspace["identity_basis"],
                label=f"{checkpoint_path.name}/{row['sample_id']}/{row['query_index']}/{stage}/{position_role}/{group_by}.identity",
            )
            row["all_vector_overlap"] = _subspace_overlap(
                vector,
                subspace["all_vector_basis"],
                label=f"{checkpoint_path.name}/{row['sample_id']}/{row['query_index']}/{stage}/{position_role}/{group_by}.all_vector",
            )
            row.update(centroid_predictions[payload_index])
            rows.append(row)
    return rows, subspace_rows


def run_value_code_subspace_report(
    *,
    config_path: Path,
    probe_set_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    checkpoint_paths: list[Path] | None,
    device_name: str,
    stages: list[str],
    position_roles: list[str],
    group_by_values: list[str],
    split_filter: list[str] | None,
    max_records: int | None,
    batch_size: int | None,
    pca_rank: int,
    markdown_top_k_rows: int,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path]:
    if not stages:
        raise ValueError("At least one --stage is required.")
    unsupported_roles = sorted(set(position_roles) - set(GEOMETRY_POSITION_ROLES))
    if unsupported_roles:
        raise ValueError(f"Unsupported position roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    unsupported_group_by = sorted(set(group_by_values) - set(CONTEXTUAL_GROUP_BY_OPTIONS))
    if unsupported_group_by:
        raise ValueError(f"Unsupported group_by values {unsupported_group_by}; expected one of {CONTEXTUAL_GROUP_BY_OPTIONS}.")
    if pca_rank <= 0:
        raise ValueError(f"pca_rank must be positive, got {pca_rank}.")
    if max_records is not None and max_records <= 0:
        raise ValueError(f"max_records must be positive when provided, got {max_records}.")
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
        probe_records = probe_records[:max_records]
    if not probe_records:
        raise RuntimeError(f"No probe records selected from {probe_set_path}.")
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    device = require_device(device_name)
    model = build_model(spec.model, vocab_size=len(vocab.tokens), device=device)
    valid_stages = _valid_residual_stages(len(model.blocks))
    unsupported_stages = sorted(set(stages) - set(valid_stages))
    if unsupported_stages:
        raise ValueError(f"Unsupported stages {unsupported_stages}; expected one of {valid_stages}.")
    resolved_batch_size = int(spec.evaluation.batch_size if batch_size is None else batch_size)
    loader = _make_probe_loader(
        probe_records=probe_records,
        batch_size=resolved_batch_size,
        pad_token_id=vocab.pad_token_id,
    )

    rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    print(
        "[value-code-subspace-report] "
        f"checkpoints={len(checkpoints)} records={len(probe_records)} stages={stages} "
        f"position_roles={position_roles} group_by={group_by_values} pca_rank={pca_rank} device={device_name}",
        flush=True,
    )
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        print(
            f"[value-code-subspace-report] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}",
            flush=True,
        )
        checkpoint_rows, checkpoint_subspace_rows = _collect_checkpoint_rows(
            model=model,
            checkpoint_path=checkpoint_path,
            loader=loader,
            stages=stages,
            position_roles=position_roles,
            group_by_values=group_by_values,
            pca_rank=pca_rank,
            vocab=vocab,
            value_token_ids=value_token_ids.to(device),
            device=device,
        )
        rows.extend(checkpoint_rows)
        subspace_rows.extend(checkpoint_subspace_rows)
        final_norm_rows = [
            row
            for row in checkpoint_rows
            if row["stage"] == "final_norm" and row["position_role"] == "prediction" and row["group_by"] == group_by_values[0]
        ]
        if final_norm_rows:
            mean_margin = _mean(
                [float(row["stage_lens_value_margin"]) for row in final_norm_rows],
                label=f"{checkpoint_path.name}.final_norm_prediction_margin",
            )
            print(
                f"[value-code-subspace-report] finished step={_checkpoint_step_from_path(checkpoint_path)} "
                f"final_norm_prediction_margin={mean_margin:.6g}",
                flush=True,
            )
        else:
            print(
                f"[value-code-subspace-report] finished step={_checkpoint_step_from_path(checkpoint_path)}",
                flush=True,
            )

    summary_rows = _summarize_rows(rows=rows)
    rows_path = output_dir / "value_code_subspace_rows.jsonl"
    summary_rows_path = output_dir / "value_code_subspace_summary_rows.jsonl"
    subspace_rows_path = output_dir / "value_code_subspace_subspaces.jsonl"
    report_path = output_dir / "value_code_subspace_report.json"
    markdown_path = output_dir / "value_code_subspace_report.md"
    write_jsonl(rows_path, rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(subspace_rows_path, subspace_rows)
    report = {
        "schema_version": VALUE_CODE_SUBSPACE_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "probe_metadata": probe_metadata,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoints": [str(path) for path in checkpoints],
        "output_dir": str(output_dir),
        "device": device_name,
        "num_probe_records": len(probe_records),
        "batch_size": resolved_batch_size,
        "stages": stages,
        "position_roles": position_roles,
        "group_by": group_by_values,
        "split_filter": split_filter,
        "max_records": max_records,
        "pca_rank": pca_rank,
        "markdown_top_k_rows": markdown_top_k_rows,
        "rows_path": str(rows_path),
        "summary_rows_path": str(summary_rows_path),
        "subspace_rows_path": str(subspace_rows_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(
        f"[value-code-subspace-report] complete report={report_path} rows={rows_path}",
        flush=True,
    )
    return report_path, markdown_path, rows_path, summary_rows_path, subspace_rows_path
