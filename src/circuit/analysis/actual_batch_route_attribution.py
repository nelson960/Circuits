from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.checkpoint_sweep import load_probe_set
from circuit.analysis.geometric_mechanisms import (
    GEOMETRY_POSITION_ROLES,
    _build_route_competition_pairs,
    _checkpoint_step_from_path,
    _compute_loss_gradient_for_records,
    _compute_route_score_for_pairs,
    _compute_route_score_gradient_for_pairs,
    _data_update_group_row,
    _data_update_route_metric_row,
    _gradient_dot_summary,
    _holdout_pair_set,
    _model_parameter_snapshot,
    _pair_metadata,
    _parameter_delta,
    _parse_route_competition_specs,
    _resolve_causal_patch_basis,
    _resolve_checkpoint_paths,
    _route_competition_route_metadata,
    _route_objective_pairs,
    _validate_geometry_stage,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import SymbolicKVDataset, read_symbolic_kv_stream_metadata
from circuit.io import iter_jsonl, write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, require_device
from circuit.vocab import Vocabulary


ACTUAL_BATCH_ROUTE_ATTRIBUTION_SCHEMA_VERSION = 1


def _load_trace_batch_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Optimizer trace batch rows not found: {path}")
    rows: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        step = int(row["step"])
        if step in rows:
            raise RuntimeError(f"Duplicate optimizer trace batch row for step {step}.")
        sample_ids = row.get("sample_ids")
        if not isinstance(sample_ids, list) or not sample_ids:
            raise RuntimeError(f"Optimizer trace batch row for step {step} has no sample_ids.")
        rows[step] = row
    if not rows:
        raise RuntimeError(f"Optimizer trace batch rows file is empty: {path}")
    return rows


def _load_trace_step_rows(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Optimizer trace step rows not found: {path}")
    rows: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        step = int(row["step"])
        if step in rows:
            raise RuntimeError(f"Duplicate optimizer trace step row for step {step}.")
        rows[step] = row
    if not rows:
        raise RuntimeError(f"Optimizer trace step rows file is empty: {path}")
    return rows


def _records_by_sample_id(*, benchmark_dir: Path, split_name: str) -> dict[str, dict[str, Any]]:
    dataset = SymbolicKVDataset(benchmark_dir, split_name)
    records: dict[str, dict[str, Any]] = {}
    for record in dataset.records:
        sample_id = str(record["sample_id"])
        if sample_id in records:
            raise RuntimeError(f"Duplicate sample_id {sample_id!r} in split {split_name!r}.")
        records[sample_id] = record
    if not records:
        raise RuntimeError(f"No records found in split {split_name!r}.")
    return records


def _records_for_trace_batch(
    *,
    batch_row: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    sample_ids = [str(sample_id) for sample_id in batch_row["sample_ids"]]
    missing = [sample_id for sample_id in sample_ids if sample_id not in records_by_id]
    if missing:
        raise KeyError(f"Optimizer trace batch references sample_ids not found in dataset split: {missing[:10]}")
    return [records_by_id[sample_id] for sample_id in sample_ids]


def _summarize_actual_batch_route_attribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize actual-batch route attribution without rows.")
    routes = sorted({str(row["route_label"]) for row in rows})
    route_summaries: list[dict[str, Any]] = []
    for route_label in routes:
        route_rows = sorted(
            [row for row in rows if str(row["route_label"]) == route_label],
            key=lambda row: int(row["source_step"]),
        )
        route_summaries.append(
            {
                "route_label": route_label,
                "stage": route_rows[0]["stage"],
                "subspace_name": route_rows[0]["subspace_name"],
                "position_role": route_rows[0]["position_role"],
                "rank": route_rows[0]["rank"],
                "head_label": route_rows[0]["head_label"],
                "num_intervals": len(route_rows),
                "source_step_first": int(route_rows[0]["source_step"]),
                "target_step_last": int(route_rows[-1]["target_step"]),
                "source_route_score_first": float(route_rows[0]["source_route_score"]),
                "target_route_score_last": float(route_rows[-1]["target_route_score"]),
                "actual_route_delta_sum": sum(float(row["actual_route_delta"]) for row in route_rows),
                "actual_update_predicted_route_delta_sum": sum(
                    float(row["actual_update_predicted_route_delta"]) for row in route_rows
                ),
                "actual_update_route_residual_sum": sum(float(row["actual_update_route_residual"]) for row in route_rows),
                "actual_update_route_sign_match_fraction": sum(
                    1 for row in route_rows if bool(row["actual_update_route_sign_match"])
                )
                / float(len(route_rows)),
                "actual_batch_route_support_sum": sum(
                    float(row["negative_loss_dot_route_gradient"]) for row in route_rows
                ),
                "actual_batch_local_sgd_route_delta_sum": sum(
                    float(row["local_sgd_route_delta_linearized"]) for row in route_rows
                ),
                "actual_batch_update_loss_reduction_sum": sum(
                    float(row["loss_reduction_under_actual_update_linearized"]) for row in route_rows
                ),
                "actual_batch_loss_mean": sum(float(row["loss"]) for row in route_rows) / float(len(route_rows)),
                "optimizer_trace_loss_mean": sum(float(row["optimizer_trace_loss"]) for row in route_rows)
                / float(len(route_rows)),
                "max_abs_loss_mismatch": max(abs(float(row["loss_delta_vs_optimizer_trace"])) for row in route_rows),
            }
        )
    return {
        "num_routes": len(routes),
        "num_rows": len(rows),
        "num_intervals": len({(int(row["source_step"]), int(row["target_step"])) for row in rows}),
        "route_summaries": sorted(
            route_summaries,
            key=lambda row: float(row["actual_route_delta_sum"]),
            reverse=True,
        ),
        "ranked_by_actual_batch_route_support": sorted(
            route_summaries,
            key=lambda row: float(row["actual_batch_route_support_sum"]),
            reverse=True,
        ),
        "ranked_by_actual_route_delta": sorted(
            route_summaries,
            key=lambda row: float(row["actual_route_delta_sum"]),
            reverse=True,
        ),
    }


def _write_actual_batch_route_attribution_markdown(
    *,
    path: Path,
    report: dict[str, Any],
) -> None:
    summary = report["summary"]
    lines = [
        "# Actual Batch Route Attribution",
        "",
        "## Calculation",
        "",
        "This report uses the optimizer trace's recorded batch for each step.",
        "",
        "```text",
        "actual_route_delta_t = route(theta_{t+1}; source_basis_t) - route(theta_t; source_basis_t)",
        "actual_update_predicted_route_delta_t = grad route(theta_t) . (theta_{t+1} - theta_t)",
        "actual_batch_route_support_t = < -grad loss_batch_t(theta_t), grad route(theta_t) >",
        "actual_batch_update_alignment_t = < -grad loss_batch_t(theta_t), theta_{t+1} - theta_t >",
        "```",
        "",
        "The first two terms test whether the actual optimizer update explains route movement. "
        "The last two terms test whether the recorded batch gradient supports that route and aligns with the actual update.",
        "",
        "## Inputs",
        "",
        f"- optimizer trace directory: `{report['optimizer_trace_dir']}`",
        f"- checkpoint directory: `{report['checkpoint_dir']}`",
        f"- batch rows: `{report['optimizer_trace_batch_rows_path']}`",
        f"- step rows: `{report['optimizer_trace_step_rows_path']}`",
        f"- probe set: `{report['probe_set_path']}`",
        "",
        "## Route Summary",
        "",
        "| route | actual route delta | predicted by actual update | actual-batch route support | local SGD route delta | batch/update loss reduction | sign match |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["route_summaries"]:
        lines.append(
            "| {route} | {actual:.6g} | {pred:.6g} | {support:.6g} | {sgd:.6g} | {loss:.6g} | {sign:.3f} |".format(
                route=row["route_label"],
                actual=float(row["actual_route_delta_sum"]),
                pred=float(row["actual_update_predicted_route_delta_sum"]),
                support=float(row["actual_batch_route_support_sum"]),
                sgd=float(row["actual_batch_local_sgd_route_delta_sum"]),
                loss=float(row["actual_batch_update_loss_reduction_sum"]),
                sign=float(row["actual_update_route_sign_match_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "## Ranked By Actual-Batch Route Support",
            "",
            "| rank | route | actual-batch route support | actual route delta | local SGD route delta |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for index, row in enumerate(summary["ranked_by_actual_batch_route_support"], start=1):
        lines.append(
            "| {rank} | {route} | {support:.6g} | {actual:.6g} | {sgd:.6g} |".format(
                rank=index,
                route=row["route_label"],
                support=float(row["actual_batch_route_support_sum"]),
                actual=float(row["actual_route_delta_sum"]),
                sgd=float(row["actual_batch_local_sgd_route_delta_sum"]),
            )
        )
    lines.extend(
        [
            "",
            "## Integrity Checks",
            "",
            f"- max absolute loss mismatch against optimizer trace: `{float(report['summary_max_abs_loss_mismatch']):.6g}`",
            f"- loss match tolerance: `{float(report['loss_match_tolerance']):.6g}`",
            "",
            "## Raw Outputs",
            "",
            f"- rows: `{report['rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_actual_batch_route_attribution(
    *,
    config_path: Path,
    probe_set_path: Path,
    optimizer_trace_dir: Path,
    output_dir: Path,
    raw_route_specs: list[str],
    route_pair_type: str,
    pair_types: list[str],
    device_name: str = "mps",
    checkpoint_paths: list[Path] | None = None,
    split_filter: list[str] | None = None,
    train_split: str = "train",
    max_pairs_per_type: int = 64,
    min_pairs_per_type: int = 1,
    loss_match_tolerance: float = 1.0e-4,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path]:
    if loss_match_tolerance < 0.0:
        raise ValueError("loss_match_tolerance must be non-negative.")
    routes = _parse_route_competition_specs(raw_route_specs)
    spec = TrainSpec.from_path(config_path)
    if float(spec.model.dropout) != 0.0:
        raise RuntimeError(
            "Actual-batch attribution requires dropout=0.0 because optimizer trace checkpoints do not save "
            "per-step dropout RNG state."
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory already exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = optimizer_trace_dir / "checkpoints"
    batch_rows_path = optimizer_trace_dir / "optimizer_update_trace_batches.jsonl"
    step_rows_path = optimizer_trace_dir / "optimizer_update_trace_steps.jsonl"
    trace_batch_rows = _load_trace_batch_rows(batch_rows_path)
    trace_step_rows = _load_trace_step_rows(step_rows_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    holdout_pairs = _holdout_pair_set(metadata)
    records_by_id = _records_by_sample_id(benchmark_dir=spec.benchmark_dir, split_name=train_split)
    device = require_device(device_name)
    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    if len(checkpoints) < 2:
        raise ValueError("actual-batch-route-attribution requires at least two trace checkpoints.")
    probe_records, probe_metadata = load_probe_set(probe_set_path)
    if str(probe_metadata["benchmark_dir"]) != str(spec.benchmark_dir):
        raise ValueError(
            f"Probe set benchmark mismatch: probe={probe_metadata['benchmark_dir']} config={spec.benchmark_dir}"
        )
    pair_types = sorted(set(pair_types), key=pair_types.index)
    pairs, pair_construction = _build_route_competition_pairs(
        probe_set_path=probe_set_path,
        spec=spec,
        vocab=vocab,
        holdout_pairs=holdout_pairs,
        pair_types=pair_types,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        split_filter=split_filter,
    )
    route_pairs = _route_objective_pairs(
        pairs=pairs,
        route_split="__all__",
        route_pair_type=route_pair_type,
    )
    model = build_model(spec.model, len(vocab.tokens), device)
    for route in routes:
        _validate_geometry_stage(model=model, stage_name=route.stage_name)
        if route.position_role not in GEOMETRY_POSITION_ROLES:
            raise ValueError(f"Unsupported position role {route.position_role!r}; expected one of {GEOMETRY_POSITION_ROLES}.")

    rows_path = output_dir / "actual_batch_route_attribution_rows.jsonl"
    pair_rows_path = output_dir / "actual_batch_route_attribution_pairs.jsonl"
    report_path = output_dir / "actual_batch_route_attribution_report.json"
    markdown_path = output_dir / "actual_batch_route_attribution_report.md"
    write_jsonl(pair_rows_path, [_pair_metadata(pair) for pair in pairs])
    intervals = list(zip(checkpoints[:-1], checkpoints[1:], strict=True))
    print(
        "[actual-batch-route-attribution] "
        f"routes={len(routes)} intervals={len(intervals)} route_pairs={len(route_pairs)} "
        f"trace={optimizer_trace_dir} device={device_name}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    final_subspace_summaries: dict[str, dict[str, Any]] = {}
    for route in routes:
        for interval_index, (source_checkpoint_path, target_checkpoint_path) in enumerate(intervals, start=1):
            source_step = _checkpoint_step_from_path(source_checkpoint_path)
            target_step = _checkpoint_step_from_path(target_checkpoint_path)
            if target_step not in trace_batch_rows:
                raise KeyError(f"No optimizer trace batch row found for target step {target_step}.")
            if target_step not in trace_step_rows:
                raise KeyError(f"No optimizer trace step row found for target step {target_step}.")
            batch_row = trace_batch_rows[target_step]
            step_row = trace_step_rows[target_step]
            learning_rate = float(step_row["learning_rate"])
            actual_batch_records = _records_for_trace_batch(
                batch_row=batch_row,
                records_by_id=records_by_id,
            )
            print(
                "[actual-batch-route-attribution] starting "
                f"route={route.label} interval={interval_index}/{len(intervals)} "
                f"{source_checkpoint_path.name}->{target_checkpoint_path.name}",
                flush=True,
            )

            source_checkpoint = load_checkpoint(source_checkpoint_path, device)
            load_model_state(model, source_checkpoint["model_state"])
            if int(source_checkpoint["step"]) != source_step:
                raise RuntimeError(
                    f"Source checkpoint step mismatch for {source_checkpoint_path}: "
                    f"payload={source_checkpoint['step']} path={source_step}"
                )
            source_parameters = _model_parameter_snapshot(model)
            basis, subspace_summary = _resolve_causal_patch_basis(
                model=model,
                vocab=vocab,
                subspace_name=route.subspace_name,
                rank=route.rank,
                head_layer=route.head_layer,
                head=route.head,
                device=device,
            )
            final_subspace_summaries[route.label] = subspace_summary
            source_route_payload = _compute_route_score_gradient_for_pairs(
                model=model,
                pairs=route_pairs,
                vocab=vocab,
                basis=basis,
                stage_name=route.stage_name,
                position_role=route.position_role,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
            )
            route_gradients = source_route_payload["gradients"]
            if not isinstance(route_gradients, dict):
                raise TypeError("Route gradient payload must contain a gradients dictionary.")
            loss_payload = _compute_loss_gradient_for_records(
                model=model,
                records=actual_batch_records,
                batch_size=spec.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
            )
            loss_delta = float(loss_payload["loss"]) - float(step_row["loss"])
            if abs(loss_delta) > loss_match_tolerance:
                raise RuntimeError(
                    f"Actual-batch loss mismatch at step {target_step}: recomputed={loss_payload['loss']:.8g} "
                    f"trace={float(step_row['loss']):.8g} delta={loss_delta:.8g} tolerance={loss_match_tolerance:.8g}"
                )

            target_checkpoint = load_checkpoint(target_checkpoint_path, device)
            load_model_state(model, target_checkpoint["model_state"])
            if int(target_checkpoint["step"]) != target_step:
                raise RuntimeError(
                    f"Target checkpoint step mismatch for {target_checkpoint_path}: "
                    f"payload={target_checkpoint['step']} path={target_step}"
                )
            target_parameters = _model_parameter_snapshot(model)
            delta_parameters = _parameter_delta(
                source_parameters=source_parameters,
                target_parameters=target_parameters,
                label=f"{source_step}->{target_step}",
            )
            target_route_payload = _compute_route_score_for_pairs(
                model=model,
                pairs=route_pairs,
                vocab=vocab,
                basis=basis,
                stage_name=route.stage_name,
                position_role=route.position_role,
                batch_size=spec.evaluation.batch_size,
                pad_token_id=vocab.pad_token_id,
                device=device,
            )
            update_route_dot_summary = _gradient_dot_summary(
                left_gradients=delta_parameters,
                right_gradients=route_gradients,
                label=f"actual batch route update {route.label} {source_step}->{target_step}",
            )
            route_metric_row = _data_update_route_metric_row(
                source_step=source_step,
                target_step=target_step,
                source_checkpoint=source_checkpoint_path,
                target_checkpoint=target_checkpoint_path,
                learning_rate=learning_rate,
                route_split="__all__",
                route_pair_type=route_pair_type,
                stage_name=route.stage_name,
                subspace_name=route.subspace_name,
                subspace_summary=subspace_summary,
                rank=route.rank,
                position_role=route.position_role,
                source_payload=source_route_payload,
                target_payload=target_route_payload,
                dot_summary=update_route_dot_summary,
                min_error_denominator=1.0e-9,
            )
            loss_gradients = loss_payload["gradients"]
            if not isinstance(loss_gradients, dict):
                raise TypeError("Loss gradient payload must contain a gradients dictionary.")
            loss_route_dot_summary = _gradient_dot_summary(
                left_gradients=loss_gradients,
                right_gradients=route_gradients,
                label=f"actual batch loss-route {route.label} {source_step}->{target_step}",
            )
            loss_update_dot_summary = _gradient_dot_summary(
                left_gradients=loss_gradients,
                right_gradients=delta_parameters,
                label=f"actual batch loss-update {route.label} {source_step}->{target_step}",
            )
            row = _data_update_group_row(
                route_metric_row=route_metric_row,
                data_group_id="actual_batch",
                data_group_values={"optimizer_trace_step": str(target_step)},
                loss_side="actual_batch",
                loss_payload=loss_payload,
                loss_route_dot_summary=loss_route_dot_summary,
                loss_update_dot_summary=loss_update_dot_summary,
            )
            row.update(_route_competition_route_metadata(route))
            row.update(
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
            rows.append(row)
            print(
                "[actual-batch-route-attribution] finished "
                f"route={route.label} {source_step}->{target_step} "
                f"actual_delta={float(row['actual_route_delta']):.6g} "
                f"predicted_delta={float(row['actual_update_predicted_route_delta']):.6g} "
                f"batch_support={float(row['negative_loss_dot_route_gradient']):.6g} "
                f"batch_update_alignment={float(row['negative_loss_dot_actual_update']):.6g}",
                flush=True,
            )

    write_jsonl(rows_path, rows)
    summary = _summarize_actual_batch_route_attribution(rows)
    max_abs_loss_mismatch = max(abs(float(row["loss_delta_vs_optimizer_trace"])) for row in rows)
    report = {
        "schema_version": ACTUAL_BATCH_ROUTE_ATTRIBUTION_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "optimizer_trace_dir": str(optimizer_trace_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "optimizer_trace_batch_rows_path": str(batch_rows_path),
        "optimizer_trace_step_rows_path": str(step_rows_path),
        "output_dir": str(output_dir),
        "device": device_name,
        "train_split": train_split,
        "routes": [_route_competition_route_metadata(route) for route in routes],
        "subspaces": final_subspace_summaries,
        "route_pair_type": route_pair_type,
        "pair_types": pair_types,
        "split_filter": split_filter,
        "max_pairs_per_type": max_pairs_per_type,
        "min_pairs_per_type": min_pairs_per_type,
        "loss_match_tolerance": loss_match_tolerance,
        "summary_max_abs_loss_mismatch": max_abs_loss_mismatch,
        "basis_mode": "source_checkpoint_per_interval",
        "pair_construction": pair_construction,
        "calculation": {
            "actual_route_delta": "route(theta_target; source_basis) - route(theta_source; source_basis)",
            "actual_update_predicted_route_delta": "grad route(theta_source; source_basis) . Delta theta_actual",
            "actual_batch_route_support": "< -grad loss_actual_batch(theta_source), grad route(theta_source; source_basis) >",
            "actual_batch_update_alignment": "< -grad loss_actual_batch(theta_source), Delta theta_actual >",
        },
        "rows_path": str(rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary": summary,
    }
    write_json(report_path, report)
    _write_actual_batch_route_attribution_markdown(path=markdown_path, report=report)
    print(
        f"[actual-batch-route-attribution] complete report={report_path} rows={rows_path}",
        flush=True,
    )
    return report_path, markdown_path, rows_path, pair_rows_path
