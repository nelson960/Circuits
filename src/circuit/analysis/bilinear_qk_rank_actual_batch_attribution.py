from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.actual_batch_route_attribution import (
    _load_trace_batch_rows,
    _load_trace_step_rows,
    _records_by_sample_id,
    _records_for_trace_batch,
)
from circuit.analysis.bilinear_qk_rank_data_attribution import (
    LOSS_SCOPES,
    _compute_loss_gradient_for_records_by_scope,
    _data_row,
    _rank_route_row,
    _summarize,
)
from circuit.analysis.bilinear_qk_rank_update_attribution import (
    LAYER_NORM_MODES,
    _assert_finite_gradients,
    _rank_match_payload_for_pairs,
    _rank_qk_source_basis,
)
from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.geometric_mechanisms import (
    ATTENTION_SCORE_RECORD_SIDES,
    GEOMETRY_POSITION_ROLES,
    _build_causal_patch_pairs,
    _checkpoint_step_from_path,
    _gradient_dot_summary,
    _holdout_pair_set,
    _model_parameter_snapshot,
    _pair_metadata,
    _parameter_delta,
    _resolve_checkpoint_paths,
    _route_objective_pairs,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import read_symbolic_kv_stream_metadata
from circuit.io import append_jsonl, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, require_device
from circuit.vocab import Vocabulary


BILINEAR_QK_RANK_ACTUAL_BATCH_ATTRIBUTION_SCHEMA_VERSION = 1


def _write_markdown(
    *,
    path: Path,
    report: dict[str, Any],
) -> None:
    summary = report["summary"]
    lines = [
        "# Bilinear QK Rank Actual-Batch Attribution",
        "",
        "This report tests whether the real optimizer-trace batches and real parameter updates support the fixed-basis low-rank QK matcher.",
        "",
        "```text",
        "C_rank(theta) = mean score_rank(prediction, support_value) - mean score_rank(prediction, value_distractors)",
        "actual_update_delta = grad C_rank(theta_source) . (theta_target - theta_source)",
        "actual_batch_support = < -grad loss_actual_batch(theta_source), grad C_rank(theta_source) >",
        "```",
        "",
        "## Replay Status",
        "",
        f"- optimizer trace: `{report['optimizer_trace_dir']}`",
        f"- trace status: `{report['optimizer_trace_status']}`",
        f"- trace blocker: {report['optimizer_trace_blocker']}",
        "",
        "## Run",
        "",
        f"- head: `L{report['head_layer']}H{report['head']}`",
        f"- ranks: `{report['ranks']}`",
        f"- context stage: `{report['context_stage']}`",
        f"- layernorm mode: `{report['layernorm_mode']}`",
        f"- route pair type: `{report['route_pair_type']}`",
        f"- loss scope: `{report['loss_scope']}`",
        f"- intervals: `{summary['intervals']}`",
        "",
        "## Route Summary",
        "",
        "| rank | actual delta | update-predicted delta | sign match | mean relative error |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["route_summaries"]:
        lines.append(
            "| {rank} | {actual:.6g} | {predicted:.6g} | {sign}/{total} | {error:.6g} |".format(
                rank=int(row["rank"]),
                actual=float(row["sum_actual_rank_match_delta"]),
                predicted=float(row["sum_actual_update_predicted_rank_match_delta"]),
                sign=int(row["actual_update_sign_match_count"]),
                total=int(row["actual_update_sign_match_total"]),
                error=float(row["mean_actual_update_rank_match_relative_error"]),
            )
        )
    lines.extend(
        [
            "",
            "## Actual Batch Support",
            "",
            "| rank | group | batch route support | SGD-equivalent delta | loss reduction under actual update | mean route cosine |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["cumulative_data_rows"]:
        lines.append(
            "| {rank} | `{group}` | {support:.6g} | {delta:.6g} | {update:.6g} | {cosine:.6g} |".format(
                rank=int(row["rank"]),
                group=row["data_group_id"],
                support=float(row["sum_negative_loss_dot_rank_gradient"]),
                delta=float(row["sum_sgd_equivalent_rank_match_delta_linearized"]),
                update=float(row["sum_loss_reduction_under_actual_update_linearized"]),
                cosine=float(row["mean_loss_negative_rank_gradient_cosine"]),
            )
        )
    lines.extend(
        [
            "",
            "## Raw Outputs",
            "",
            f"- route rows: `{report['route_rows_path']}`",
            f"- actual-batch rows: `{report['actual_batch_rows_path']}`",
            f"- route pair rows: `{report['route_pair_rows_path']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _trace_metadata(optimizer_trace_dir: Path) -> tuple[str, str]:
    config_path = optimizer_trace_dir / "optimizer_update_trace_config.json"
    if not config_path.exists():
        return "unknown_missing_trace_config", "Missing optimizer_update_trace_config.json."
    from circuit.io import read_json

    payload = read_json(config_path)
    return (
        str(payload.get("historical_replay_status", "unknown")),
        str(payload.get("historical_replay_blocker", "")),
    )


def run_bilinear_qk_rank_actual_batch_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    optimizer_trace_dir: Path,
    output_dir: Path,
    head_layer: int,
    head: int,
    ranks: list[int],
    context_stage: str,
    layernorm_mode: str,
    score_query_role: str,
    support_key_role: str,
    distractor_key_role: str,
    route_pair_types: list[str],
    route_pair_type: str,
    device_name: str = "cpu",
    checkpoint_paths: list[Path] | None = None,
    record_side: str = "clean",
    route_split_filter: list[str] | None = None,
    route_split: str = "__all__",
    train_split: str = "train",
    max_route_pairs_per_type: int = 64,
    min_route_pairs_per_type: int = 1,
    loss_scope: str = "full_lm",
    loss_match_tolerance: float = 1.0e-4,
    top_k_data_groups: int = 24,
    min_error_denominator: float = 1.0e-9,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    if not ranks:
        raise ValueError("At least one --rank is required.")
    resolved_ranks = sorted(set(ranks))
    if any(rank <= 0 for rank in resolved_ranks):
        raise ValueError(f"Ranks must be positive integers, got {resolved_ranks}.")
    if record_side not in ATTENTION_SCORE_RECORD_SIDES:
        raise ValueError(f"Unsupported record_side {record_side!r}; expected one of {ATTENTION_SCORE_RECORD_SIDES}.")
    if layernorm_mode not in LAYER_NORM_MODES:
        raise ValueError(f"Unsupported layernorm_mode {layernorm_mode!r}; expected one of {LAYER_NORM_MODES}.")
    unsupported_roles = [
        role
        for role in [score_query_role, support_key_role, distractor_key_role]
        if role not in GEOMETRY_POSITION_ROLES
    ]
    if unsupported_roles:
        raise ValueError(f"Unsupported position roles {unsupported_roles}; expected one of {GEOMETRY_POSITION_ROLES}.")
    if loss_scope not in LOSS_SCOPES:
        raise ValueError(f"Unsupported loss_scope {loss_scope!r}; expected one of {LOSS_SCOPES}.")
    if loss_match_tolerance < 0.0:
        raise ValueError("loss_match_tolerance must be non-negative.")
    if top_k_data_groups <= 0:
        raise ValueError("top_k_data_groups must be positive.")
    if min_error_denominator <= 0.0:
        raise ValueError("min_error_denominator must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = TrainSpec.from_path(config_path)
    if float(spec.model.dropout) != 0.0:
        raise RuntimeError(
            "Actual-batch attribution requires dropout=0.0 because optimizer trace checkpoints do not save per-step dropout RNG state."
        )
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    device = require_device(device_name)
    model = build_model(spec.model, len(vocab.tokens), device)
    if head_layer < 0 or head_layer >= len(model.blocks):
        raise ValueError(f"head_layer {head_layer} outside model range 0..{len(model.blocks) - 1}.")
    if head < 0 or head >= model.blocks[head_layer].attn.n_heads:
        raise ValueError(f"head {head} outside model range 0..{model.blocks[head_layer].attn.n_heads - 1}.")

    checkpoint_dir = optimizer_trace_dir / "checkpoints"
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("bilinear-qk-rank-actual-batch-attribution requires at least two trace checkpoints.")
    trace_batch_rows = _load_trace_batch_rows(optimizer_trace_dir / "optimizer_update_trace_batches.jsonl")
    trace_step_rows = _load_trace_step_rows(optimizer_trace_dir / "optimizer_update_trace_steps.jsonl")
    records_by_id = _records_by_sample_id(benchmark_dir=spec.benchmark_dir, split_name=train_split)
    optimizer_trace_status, optimizer_trace_blocker = _trace_metadata(optimizer_trace_dir)

    route_pair_types = sorted(set(route_pair_types), key=route_pair_types.index)
    route_pairs_all, route_pair_construction = _build_causal_patch_pairs(
        probe_records=probe_records,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=route_pair_types,
        max_pairs_per_type=max_route_pairs_per_type,
        min_pairs_per_type=min_route_pairs_per_type,
        split_filter=route_split_filter,
    )
    route_pairs = _route_objective_pairs(
        pairs=route_pairs_all,
        route_split=route_split,
        route_pair_type=route_pair_type,
    )

    report_path = output_dir / "bilinear_qk_rank_actual_batch_attribution_report.json"
    markdown_path = output_dir / "bilinear_qk_rank_actual_batch_attribution_report.md"
    route_rows_path = output_dir / "bilinear_qk_rank_actual_batch_attribution_route_rows.jsonl"
    actual_batch_rows_path = output_dir / "bilinear_qk_rank_actual_batch_attribution_rows.jsonl"
    route_pair_rows_path = output_dir / "bilinear_qk_rank_actual_batch_attribution_route_pairs.jsonl"
    progress_path = output_dir / "bilinear_qk_rank_actual_batch_attribution_progress.json"
    write_jsonl(route_pair_rows_path, [_pair_metadata(pair) for pair in route_pairs])

    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[bilinear-qk-rank-actual-batch-attribution] "
        f"intervals={len(intervals)} checkpoints={len(checkpoints)} route_pairs={len(route_pairs)} "
        f"ranks={resolved_ranks} device={device_name} head=L{head_layer}H{head} "
        f"trace={optimizer_trace_dir} loss_scope={loss_scope}",
        flush=True,
    )

    all_route_rows: list[dict[str, Any]] = []
    all_actual_batch_rows: list[dict[str, Any]] = []
    for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
        source_step = _checkpoint_step_from_path(source_checkpoint_path)
        target_step = _checkpoint_step_from_path(target_checkpoint_path)
        if target_step - source_step != 1:
            raise RuntimeError(
                "Actual-batch bilinear QK attribution requires one-step checkpoint intervals. "
                f"Got {source_step}->{target_step}. Re-run optimizer-update-trace with --checkpoint-every 1."
            )
        if target_step not in trace_batch_rows:
            raise KeyError(f"No optimizer trace batch row found for target step {target_step}.")
        if target_step not in trace_step_rows:
            raise KeyError(f"No optimizer trace step row found for target step {target_step}.")
        batch_row = trace_batch_rows[target_step]
        step_row = trace_step_rows[target_step]
        learning_rate = float(step_row["learning_rate"])
        actual_batch_records = _records_for_trace_batch(batch_row=batch_row, records_by_id=records_by_id)
        print(
            "[bilinear-qk-rank-actual-batch-attribution] starting "
            f"{interval_index}/{len(intervals)} {source_checkpoint_path.name}->{target_checkpoint_path.name}",
            flush=True,
        )

        source_checkpoint = load_checkpoint(source_checkpoint_path, device)
        target_checkpoint = load_checkpoint(target_checkpoint_path, device)
        if int(source_checkpoint["step"]) != source_step:
            raise RuntimeError(f"Source checkpoint step mismatch: payload={source_checkpoint['step']} path={source_step}")
        if int(target_checkpoint["step"]) != target_step:
            raise RuntimeError(f"Target checkpoint step mismatch: payload={target_checkpoint['step']} path={target_step}")
        load_model_state(model, source_checkpoint["model_state"])
        source_parameters = _model_parameter_snapshot(model)
        loss_payload = _compute_loss_gradient_for_records_by_scope(
            model=model,
            records=actual_batch_records,
            batch_size=spec.batch_size,
            pad_token_id=vocab.pad_token_id,
            device=device,
            loss_scope=loss_scope,
        )
        loss_delta = None
        if loss_scope == "full_lm":
            loss_delta = float(loss_payload["loss"]) - float(step_row["loss"])
            if abs(loss_delta) > loss_match_tolerance:
                raise RuntimeError(
                    f"Actual-batch loss mismatch at step {target_step}: recomputed={loss_payload['loss']:.8g} "
                    f"trace={float(step_row['loss']):.8g} delta={loss_delta:.8g} tolerance={loss_match_tolerance:.8g}"
                )

        source_payloads: dict[int, dict[str, Any]] = {}
        route_gradients_by_rank: dict[int, dict[str, torch.Tensor]] = {}
        basis_by_rank: dict[int, tuple[torch.Tensor, torch.Tensor, dict[str, float]]] = {}
        for rank in resolved_ranks:
            left_basis, right_basis, singular_summary = _rank_qk_source_basis(
                model=model,
                head_layer=head_layer,
                head=head,
                rank=rank,
            )
            basis_by_rank[rank] = (left_basis, right_basis, singular_summary)
            source_payload = _rank_match_payload_for_pairs(
                model=model,
                pairs=route_pairs,
                head_layer=head_layer,
                head=head,
                rank=rank,
                left_basis=left_basis,
                right_basis=right_basis,
                singular_summary=singular_summary,
                context_stage=context_stage,
                layernorm_mode=layernorm_mode,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
                record_side=record_side,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
                track_grad=True,
            )
            gradients = source_payload["gradients"]
            if not isinstance(gradients, dict):
                raise TypeError("Rank QK source payload gradients must be a dictionary.")
            _assert_finite_gradients(gradients, label=f"rank-{rank} actual-batch route objective")
            source_payloads[rank] = source_payload
            route_gradients_by_rank[rank] = gradients

        load_model_state(model, target_checkpoint["model_state"])
        target_parameters = _model_parameter_snapshot(model)
        delta_parameters = _parameter_delta(
            source_parameters=source_parameters,
            target_parameters=target_parameters,
            label=f"bilinear QK rank actual batch attribution {source_step}->{target_step}",
        )

        for rank in resolved_ranks:
            left_basis, right_basis, singular_summary = basis_by_rank[rank]
            target_payload = _rank_match_payload_for_pairs(
                model=model,
                pairs=route_pairs,
                head_layer=head_layer,
                head=head,
                rank=rank,
                left_basis=left_basis,
                right_basis=right_basis,
                singular_summary=singular_summary,
                context_stage=context_stage,
                layernorm_mode=layernorm_mode,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
                record_side=record_side,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
                track_grad=False,
            )
            route_gradients = route_gradients_by_rank[rank]
            actual_update_dot_summary = _gradient_dot_summary(
                left_gradients=delta_parameters,
                right_gradients=route_gradients,
                label=f"rank actual batch update {source_step}->{target_step} rank={rank}",
            )
            route_row = _rank_route_row(
                source_step=source_step,
                target_step=target_step,
                source_checkpoint=source_checkpoint_path,
                target_checkpoint=target_checkpoint_path,
                learning_rate=learning_rate,
                route_split=route_split,
                route_pair_type=route_pair_type,
                record_side=record_side,
                rank=rank,
                context_stage=context_stage,
                layernorm_mode=layernorm_mode,
                head_layer=head_layer,
                head=head,
                score_query_role=score_query_role,
                support_key_role=support_key_role,
                distractor_key_role=distractor_key_role,
                source_payload=source_payloads[rank],
                target_payload=target_payload,
                actual_update_dot_summary=actual_update_dot_summary,
                min_error_denominator=min_error_denominator,
            )
            append_jsonl(route_rows_path, route_row)
            all_route_rows.append(route_row)

            loss_gradients = loss_payload["gradients"]
            if not isinstance(loss_gradients, dict):
                raise TypeError("Loss payload gradients must be a dictionary.")
            loss_rank_dot_summary = _gradient_dot_summary(
                left_gradients=loss_gradients,
                right_gradients=route_gradients,
                label=f"rank actual batch loss-route {source_step}->{target_step} rank={rank}",
            )
            loss_update_dot_summary = _gradient_dot_summary(
                left_gradients=loss_gradients,
                right_gradients=delta_parameters,
                label=f"rank actual batch loss-actual-update {source_step}->{target_step} rank={rank}",
            )
            actual_batch_row = _data_row(
                route_row=route_row,
                data_group_id="actual_batch",
                data_group_values={"batch_source": "optimizer_trace"},
                loss_side="actual_batch",
                loss_scope=loss_scope,
                loss_payload=loss_payload,
                loss_rank_dot_summary=loss_rank_dot_summary,
                loss_update_dot_summary=loss_update_dot_summary,
            )
            actual_batch_row.update(
                {
                    "optimizer_trace_dir": str(optimizer_trace_dir),
                    "optimizer_trace_batch_step": target_step,
                    "optimizer_trace_loss": float(step_row["loss"]),
                    "optimizer_trace_token_accuracy": float(step_row["token_accuracy"]),
                    "optimizer_trace_parameter_delta_l2": float(step_row["parameter_delta_l2"]),
                    "optimizer_trace_pre_clip_grad_norm": float(step_row["pre_clip_grad_norm"]),
                    "loss_delta_vs_optimizer_trace": loss_delta,
                    "actual_batch_sample_count": len(actual_batch_records),
                    "actual_batch_query_event_count": int(batch_row["query_event_count"]),
                    "actual_batch_unique_sample_count": len(set(str(sample_id) for sample_id in batch_row["sample_ids"])),
                }
            )
            append_jsonl(actual_batch_rows_path, actual_batch_row)
            all_actual_batch_rows.append(actual_batch_row)

        primary = next(
            row
            for row in all_actual_batch_rows
            if int(row["source_step"]) == source_step
            and int(row["target_step"]) == target_step
            and int(row["rank"]) == resolved_ranks[-1]
        )
        print(
            "[bilinear-qk-rank-actual-batch-attribution] finished "
            f"{source_step}->{target_step} rank={primary['rank']} "
            f"actual_delta={float(primary['actual_rank_match_delta']):.6g} "
            f"update_predicted={float(primary['actual_update_predicted_rank_match_delta']):.6g} "
            f"batch_support={float(primary['negative_loss_dot_rank_gradient']):.6g} "
            f"batch_update_alignment={float(primary['negative_loss_dot_actual_update']):.6g}",
            flush=True,
        )
        write_json(
            progress_path,
            {
                "status": "running",
                "completed_intervals": interval_index,
                "total_intervals": len(intervals),
                "last_source_step": source_step,
                "last_target_step": target_step,
                "route_rows_path": str(route_rows_path),
                "actual_batch_rows_path": str(actual_batch_rows_path),
            },
        )

    summary = _summarize(
        route_rows=all_route_rows,
        data_rows=all_actual_batch_rows,
        top_k_data_groups=top_k_data_groups,
    )
    report = {
        "schema_version": BILINEAR_QK_RANK_ACTUAL_BATCH_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "optimizer_trace_dir": str(optimizer_trace_dir),
        "optimizer_trace_status": optimizer_trace_status,
        "optimizer_trace_blocker": optimizer_trace_blocker,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "output_dir": str(output_dir),
        "device": device_name,
        "train_split": train_split,
        "head_layer": head_layer,
        "head": head,
        "ranks": resolved_ranks,
        "context_stage": context_stage,
        "layernorm_mode": layernorm_mode,
        "score_query_role": score_query_role,
        "support_key_role": support_key_role,
        "distractor_key_role": distractor_key_role,
        "record_side": record_side,
        "route_pair_types": route_pair_types,
        "route_pair_type": route_pair_type,
        "route_split": route_split,
        "route_split_filter": route_split_filter,
        "max_route_pairs_per_type": max_route_pairs_per_type,
        "min_route_pairs_per_type": min_route_pairs_per_type,
        "loss_scope": loss_scope,
        "loss_match_tolerance": loss_match_tolerance,
        "top_k_data_groups": top_k_data_groups,
        "min_error_denominator": min_error_denominator,
        "calculation": {
            "rank_match_score": "mean score_rank(prediction, support_value) - mean score_rank(prediction, value_distractors)",
            "rank_basis": "source checkpoint fixed SVD basis for each interval and rank",
            "actual_update_predicted_delta": "grad rank_match_score(theta_source) . (theta_target - theta_source)",
            "actual_batch_rank_support": "< -grad loss_actual_batch(theta_source), grad rank_match_score(theta_source) >",
            "sgd_equivalent_delta": "learning_rate * actual_batch_rank_support",
            "actual_batch_update_alignment": "< -grad loss_actual_batch(theta_source), theta_target - theta_source >",
        },
        "route_pair_construction": route_pair_construction,
        "route_num_pairs": len(route_pairs),
        "route_rows_path": str(route_rows_path),
        "actual_batch_rows_path": str(actual_batch_rows_path),
        "route_pair_rows_path": str(route_pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    write_json(
        progress_path,
        {
            "status": "complete",
            "completed_intervals": len(intervals),
            "total_intervals": len(intervals),
            "report_path": str(report_path),
            "markdown_path": str(markdown_path),
            "route_rows_path": str(route_rows_path),
            "actual_batch_rows_path": str(actual_batch_rows_path),
        },
    )
    print(
        f"[bilinear-qk-rank-actual-batch-attribution] complete report={report_path} rows={actual_batch_rows_path}",
        flush=True,
    )
    return report_path, markdown_path, route_rows_path, actual_batch_rows_path, route_pair_rows_path
