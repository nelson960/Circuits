from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
from typing import Any

import torch

from circuit.analysis.contextual_svd_alignment import (
    CONTEXTUAL_GROUP_BY_OPTIONS,
    _abs_cosine,
    _role_subspace,
    _subspace_overlap,
)
from circuit.analysis.formation import extract_answer_logits
from circuit.analysis.geometric_mechanisms import (
    GEOMETRY_POSITION_ROLES,
    _attention_key_positions,
    _checkpoint_step_from_path,
    _validate_single_query_batch,
)
from circuit.analysis.output_component_causal_validation import _component_mask_kwargs
from circuit.analysis.output_route_closure import (
    OUTPUT_ROUTE_MARGIN_SIDES,
    OUTPUT_ROUTE_SCALARS,
    _checkpoint_paths_by_step,
    _component_labels,
    _filter_component_labels,
    _load_scalar_pair_rows,
    _mean,
    _resolve_unique_values,
    _safe_r_squared,
    _selected_pairs_by_id,
)
from circuit.analysis.residual_state_rescue import (
    RESIDUAL_STATE_RESCUE_ENDPOINT_ROLES,
    _resolve_patch_stages,
    _validate_maskable_components,
)
from circuit.analysis.shared_feature_dynamics import _import_matplotlib
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import collate_symbolic_kv, read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, move_batch_to_device, require_device
from circuit.vocab import Vocabulary


RESIDUAL_DELTA_VECTOR_SCHEMA_VERSION = 1


def _filter_scalar_pair_rows_for_delta(
    *,
    rows: list[dict[str, Any]],
    margin_sides: list[str],
    pair_types: list[str],
    scalar_names: list[str],
) -> list[dict[str, Any]]:
    pair_type_filter = set(pair_types)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if str(row["margin_side"]) not in margin_sides:
            continue
        if str(row["pair_type"]) not in pair_type_filter:
            continue
        scalars = row.get("scalars")
        if not isinstance(scalars, dict):
            raise RuntimeError(f"Scalar pair row is missing scalars: {row.get('interval_pair_id')}")
        missing = [name for name in scalar_names if name not in scalars]
        if missing:
            raise KeyError(f"Scalar pair row {row.get('interval_pair_id')} is missing scalar names: {missing}")
        filtered.append(row)
    if not filtered:
        raise RuntimeError("No scalar pair rows survived residual-delta-vector filters.")
    return filtered


def _endpoint_pair_ids(
    *,
    scalar_pair_rows: list[dict[str, Any]],
    endpoint_roles: list[str],
) -> dict[tuple[int, str, str], set[str]]:
    pair_ids_by_endpoint: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    for row in scalar_pair_rows:
        margin_side = str(row["margin_side"])
        pair_id = str(row["pair_id"])
        for endpoint_kind in endpoint_roles:
            step = int(row[f"{endpoint_kind}_step"])
            pair_ids_by_endpoint[(step, margin_side, endpoint_kind)].add(pair_id)
    if not pair_ids_by_endpoint:
        raise RuntimeError("No endpoint pair ids could be built from scalar pair rows.")
    return pair_ids_by_endpoint


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
    prediction_position = int(metadata["prediction_positions"][flat_index].item())
    if group_by == "position_token":
        return int(batch["input_ids"][context_batch_row, context_position].item())
    if group_by == "query_key":
        return int(batch["input_ids"][query_batch_row, int(metadata["query_key_positions"][flat_index].item())].item())
    if group_by == "support_value":
        return int(batch["input_ids"][query_batch_row, int(metadata["support_value_positions"][flat_index].item())].item())
    if group_by == "answer_value":
        return int(answer_targets[flat_index].item())
    if group_by == "support_key":
        support_batch_row, support_positions = _attention_key_positions(
            batch=batch,
            metadata=metadata,
            flat_index=flat_index,
            position_role="support_key",
            max_position=prediction_position,
        )
        if support_batch_row != query_batch_row or len(support_positions) != 1:
            record = batch["records"][query_batch_row]
            query_index = int(metadata["query_indices"][flat_index].item())
            raise RuntimeError(
                f"Expected exactly one causal support_key position for {record['sample_id']} query {query_index}, "
                f"got row={support_batch_row} positions={support_positions}."
            )
        return int(batch["input_ids"][support_batch_row, support_positions[0]].item())
    raise ValueError(f"Unhandled group_by mode: {group_by}")


def _token_label(vocab: Vocabulary, token_id: int) -> str:
    if token_id < 0 or token_id >= len(vocab.tokens):
        raise ValueError(f"Token id {token_id} outside vocabulary size {len(vocab.tokens)}.")
    return str(vocab.tokens[token_id])


def _summarize_delta_rows(*, delta_rows: list[dict[str, Any]], group_by_values: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        for group_by in group_by_values:
            key = (
                str(row["source_component"]),
                int(row["step"]),
                str(row["endpoint_kind"]),
                str(row["margin_side"]),
                str(row["stage"]),
                str(row["position_role"]),
                group_by,
            )
            grouped[key].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (
        source_component,
        step,
        endpoint_kind,
        margin_side,
        stage,
        position_role,
        group_by,
    ), rows in sorted(grouped.items()):
        delta_norms = [float(row["delta_norm"]) for row in rows]
        identity = [float(row[f"{group_by}_identity_overlap"]) for row in rows]
        all_vector = [float(row[f"{group_by}_all_vector_overlap"]) for row in rows]
        mean_cosines = [float(row[f"{group_by}_mean_abs_cosine"]) for row in rows]
        summary_rows.append(
            {
                "source_component": source_component,
                "step": step,
                "endpoint_kind": endpoint_kind,
                "margin_side": margin_side,
                "stage": stage,
                "position_role": position_role,
                "group_by": group_by,
                "num_rows": len(rows),
                "mean_delta_norm": _mean(delta_norms, label=f"{stage}/{position_role}/{group_by} delta_norm"),
                "mean_identity_overlap": _mean(identity, label=f"{stage}/{position_role}/{group_by} identity"),
                "mean_all_vector_overlap": _mean(all_vector, label=f"{stage}/{position_role}/{group_by} all_vector"),
                "mean_mean_abs_cosine": _mean(mean_cosines, label=f"{stage}/{position_role}/{group_by} mean_cosine"),
                "identity_overlap_r_squared_vs_delta_norm": _safe_r_squared(delta_norms, identity),
            }
        )
    return summary_rows


def _plot_delta_summary(
    *,
    summary_rows: list[dict[str, Any]],
    output_path: Path,
    top_k_rows: int,
) -> Path | None:
    if not summary_rows:
        return None
    _, plt = _import_matplotlib()

    rows = sorted(summary_rows, key=lambda row: abs(float(row["mean_identity_overlap"])), reverse=True)[:top_k_rows]
    if not rows:
        return None
    labels = [
        f"{row['source_component']} {row['endpoint_kind']} {row['step']} {row['stage']} "
        f"{row['position_role']}->{row['group_by']}"
        for row in rows
    ]
    values = [float(row["mean_identity_overlap"]) for row in rows]
    height = max(4.0, 0.35 * len(rows))
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(range(len(rows)), values, color="#2f7d67")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean overlap with contextual identity subspace")
    ax.set_title("Residual Delta Vector Semantic Alignment")
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _write_markdown(
    *,
    path: Path,
    report: dict[str, Any],
    plot_paths: dict[str, Path],
) -> None:
    summary_rows = sorted(
        report["summary_rows"],
        key=lambda row: abs(float(row["mean_identity_overlap"])),
        reverse=True,
    )[: int(report["markdown_top_k_rows"])]
    lines = [
        "# Residual Delta Vector Report",
        "",
        "This report asks what semantic directions are present in the residual vector changed by a source component.",
        "",
        "## Scope",
        "",
        f"- sources: `{', '.join(report['source_components'])}`",
        f"- stages: `{', '.join(report['stages'])}`",
        f"- position roles: `{', '.join(report['position_roles'])}`",
        f"- group-by labels: `{', '.join(report['group_by'])}`",
        f"- pair types: `{', '.join(report['pair_types'])}`",
        f"- endpoint roles: `{', '.join(report['endpoint_roles'])}`",
        f"- PCA rank: `{report['pca_rank']}`",
        "",
        "## Calculation",
        "",
        "For each selected endpoint, the tool runs a clean forward pass and a source-component-ablated forward pass.",
        "At a selected residual stage and position role it computes:",
        "",
        "`delta = clean_residual - source_ablated_residual`",
        "",
        "The delta vector is then compared with contextual residual subspaces built from the clean run.",
        "For example, `group_by=query_key` asks whether the delta points along directions that separate queried keys.",
        "",
        "## Top Identity-Subspace Alignments",
        "",
    ]
    if summary_rows:
        lines.extend(
            [
                "| source | step | endpoint | stage | position | group | rows | delta norm | identity overlap | all-vector overlap |",
                "|---|---:|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in summary_rows:
            lines.append(
                f"| `{row['source_component']}` | {row['step']} | `{row['endpoint_kind']}` | "
                f"`{row['stage']}` | `{row['position_role']}` | `{row['group_by']}` | "
                f"{row['num_rows']} | {row['mean_delta_norm']:.6g} | "
                f"{row['mean_identity_overlap']:.6g} | {row['mean_all_vector_overlap']:.6g} |"
            )
    else:
        lines.append("No summary rows were produced.")
    lines.extend(["", "## Outputs", ""])
    lines.append(f"- delta rows: `{report['delta_rows_path']}`")
    lines.append(f"- summary rows: `{report['summary_rows_path']}`")
    lines.append(f"- subspace rows: `{report['subspace_rows_path']}`")
    lines.append(f"- pair rows: `{report['pair_rows_path']}`")
    for label, plot_path in sorted(plot_paths.items()):
        lines.append(f"- {label}: `{plot_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _compute_delta_rows_for_source(
    *,
    model: torch.nn.Module,
    checkpoint_paths_by_step: dict[int, Path],
    pairs_by_id: dict[str, dict[str, Any]],
    endpoint_pair_ids: dict[tuple[int, str, str], set[str]],
    source_component: str,
    stages: list[str],
    position_roles: list[str],
    group_by_values: list[str],
    pca_rank: int,
    batch_size: int,
    pad_token_id: int,
    vocab: Vocabulary,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    num_layers = len(model.blocks)
    num_heads = model.spec.n_heads
    source_mask_kwargs = _component_mask_kwargs(
        component=source_component,
        num_layers=num_layers,
        num_heads=num_heads,
        device=device,
    )
    delta_payloads: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    pair_ids = sorted(pairs_by_id)
    for step, margin_side, endpoint_kind in sorted(endpoint_pair_ids):
        if step not in checkpoint_paths_by_step:
            raise KeyError(f"No checkpoint path for step {step}.")
        checkpoint_path = checkpoint_paths_by_step[step]
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        model.eval()
        payload_step = int(checkpoint["step"])
        path_step = _checkpoint_step_from_path(checkpoint_path)
        if payload_step != step or path_step != step:
            raise RuntimeError(
                f"Checkpoint step mismatch for {checkpoint_path}: requested={step} payload={payload_step} path={path_step}"
            )
        side_key = f"{margin_side}_record"
        endpoint_required_pair_ids = endpoint_pair_ids[(step, margin_side, endpoint_kind)]
        vectors_by_key: dict[tuple[str, str, str], dict[int, list[torch.Tensor]]] = {
            (stage, position_role, group_by): defaultdict(list)
            for stage in stages
            for position_role in position_roles
            for group_by in group_by_values
        }
        endpoint_payloads: list[dict[str, Any]] = []
        for start_index in range(0, len(pair_ids), batch_size):
            batch_pair_ids = pair_ids[start_index : start_index + batch_size]
            active_pair_ids = [pair_id for pair_id in batch_pair_ids if pair_id in endpoint_required_pair_ids]
            if not active_pair_ids:
                continue
            records = [pairs_by_id[pair_id][side_key] for pair_id in batch_pair_ids]
            batch = move_batch_to_device(collate_symbolic_kv(records, pad_token_id), device)
            with torch.no_grad():
                clean_outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_residual_streams=True,
                )
                ablated_outputs = model(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_residual_streams=True,
                    **source_mask_kwargs,
                )
            if clean_outputs.residual_streams is None or ablated_outputs.residual_streams is None:
                raise RuntimeError("Residual-delta vector report requires residual streams.")
            missing_clean = [stage for stage in stages if stage not in clean_outputs.residual_streams]
            missing_ablated = [stage for stage in stages if stage not in ablated_outputs.residual_streams]
            if missing_clean or missing_ablated:
                raise KeyError(
                    f"Missing residual stages clean={missing_clean} ablated={missing_ablated}. "
                    f"Available clean stages: {sorted(clean_outputs.residual_streams)}"
                )
            _, answer_targets, clean_metadata = extract_answer_logits(clean_outputs.logits, batch)
            _, _, ablated_metadata = extract_answer_logits(ablated_outputs.logits, batch)
            _validate_single_query_batch(batch=batch, metadata=clean_metadata, label="residual-delta clean")
            _validate_single_query_batch(batch=batch, metadata=ablated_metadata, label="residual-delta ablated")
            for flat_index in range(int(clean_metadata["rows"].numel())):
                query_batch_row = int(clean_metadata["rows"][flat_index].item())
                pair_id = str(batch_pair_ids[query_batch_row])
                if pair_id not in endpoint_required_pair_ids:
                    continue
                query_index = int(clean_metadata["query_indices"][flat_index].item())
                record = batch["records"][query_batch_row]
                query_key_token_id = int(
                    batch["input_ids"][
                        query_batch_row,
                        int(clean_metadata["query_key_positions"][flat_index].item()),
                    ].item()
                )
                support_value_token_id = int(
                    batch["input_ids"][
                        query_batch_row,
                        int(clean_metadata["support_value_positions"][flat_index].item()),
                    ].item()
                )
                answer_value_token_id = int(answer_targets[flat_index].item())
                prediction_position = int(clean_metadata["prediction_positions"][flat_index].item())
                for stage in stages:
                    clean_stage = clean_outputs.residual_streams[stage].detach().float().cpu()
                    ablated_stage = ablated_outputs.residual_streams[stage].detach().float().cpu()
                    if clean_stage.shape != ablated_stage.shape:
                        raise RuntimeError(
                            f"Clean/ablated stage shape mismatch at {stage}: "
                            f"clean={tuple(clean_stage.shape)} ablated={tuple(ablated_stage.shape)}"
                        )
                    for position_role in position_roles:
                        context_batch_row, positions = _attention_key_positions(
                            batch=batch,
                            metadata=clean_metadata,
                            flat_index=flat_index,
                            position_role=position_role,
                            max_position=prediction_position,
                        )
                        clean_vectors = torch.stack(
                            [clean_stage[context_batch_row, int(position), :].clone() for position in positions],
                            dim=0,
                        )
                        ablated_vectors = torch.stack(
                            [ablated_stage[context_batch_row, int(position), :].clone() for position in positions],
                            dim=0,
                        )
                        clean_mean = clean_vectors.mean(dim=0)
                        ablated_mean = ablated_vectors.mean(dim=0)
                        delta = clean_mean - ablated_mean
                        for position_index, position in enumerate(positions):
                            clean_vector = clean_vectors[position_index].clone()
                            for group_by in group_by_values:
                                group_token_id = _group_token_id(
                                    group_by=group_by,
                                    batch=batch,
                                    metadata=clean_metadata,
                                    answer_targets=answer_targets,
                                    flat_index=flat_index,
                                    context_batch_row=context_batch_row,
                                    context_position=int(position),
                                )
                                vectors_by_key[(stage, position_role, group_by)][group_token_id].append(clean_vector)
                        endpoint_payloads.append(
                            {
                                "source_component": source_component,
                                "step": step,
                                "checkpoint": str(checkpoint_path),
                                "endpoint_kind": endpoint_kind,
                                "margin_side": margin_side,
                                "pair_id": pair_id,
                                "pair_type": str(pairs_by_id[pair_id]["pair_type"]),
                                "sample_id": str(record["sample_id"]),
                                "split": str(record["split"]),
                                "query_index": query_index,
                                "stage": stage,
                                "position_role": position_role,
                                "selected_positions": [int(position) for position in positions],
                                "num_positions": len(positions),
                                "query_key_token_id": query_key_token_id,
                                "query_key_token": _token_label(vocab, query_key_token_id),
                                "support_value_token_id": support_value_token_id,
                                "support_value_token": _token_label(vocab, support_value_token_id),
                                "answer_value_token_id": answer_value_token_id,
                                "answer_value_token": _token_label(vocab, answer_value_token_id),
                                "clean_mean_norm": float(clean_mean.norm().item()),
                                "ablated_mean_norm": float(ablated_mean.norm().item()),
                                "delta_norm": float(delta.norm().item()),
                                "_delta_vector": delta,
                            }
                        )
        if not endpoint_payloads:
            raise RuntimeError(
                f"No residual delta payloads built for source={source_component} endpoint="
                f"{step}/{margin_side}/{endpoint_kind}."
            )
        subspaces: dict[tuple[str, str, str], dict[str, torch.Tensor]] = {}
        for key, vectors_by_token in sorted(vectors_by_key.items()):
            stage, position_role, group_by = key
            subspace, summary = _role_subspace(
                role_label=f"{stage}:{position_role}:{group_by}",
                context_role=position_role,
                group_by=group_by,
                vectors_by_token=vectors_by_token,
                vocab=vocab,
                pca_rank=pca_rank,
            )
            subspaces[key] = subspace
            subspace_rows.append(
                {
                    "source_component": source_component,
                    "step": step,
                    "checkpoint": str(checkpoint_path),
                    "endpoint_kind": endpoint_kind,
                    "margin_side": margin_side,
                    "stage": stage,
                    "position_role": position_role,
                    "group_by": group_by,
                    **summary,
                }
            )
        for payload in endpoint_payloads:
            delta = payload.pop("_delta_vector")
            row = dict(payload)
            for group_by in group_by_values:
                subspace_key = (str(row["stage"]), str(row["position_role"]), group_by)
                if subspace_key not in subspaces:
                    raise KeyError(f"Missing contextual subspace for {subspace_key}.")
                subspace = subspaces[subspace_key]
                row[f"{group_by}_mean_abs_cosine"] = _abs_cosine(
                    delta,
                    subspace["mean_direction"],
                    label=f"{source_component}/{step}/{row['pair_id']}/{row['stage']}/{row['position_role']}/{group_by}.mean",
                )
                row[f"{group_by}_identity_overlap"] = _subspace_overlap(
                    delta,
                    subspace["identity_basis"],
                    label=f"{source_component}/{step}/{row['pair_id']}/{row['stage']}/{row['position_role']}/{group_by}.identity",
                )
                row[f"{group_by}_all_vector_overlap"] = _subspace_overlap(
                    delta,
                    subspace["all_vector_basis"],
                    label=f"{source_component}/{step}/{row['pair_id']}/{row['stage']}/{row['position_role']}/{group_by}.all_vector",
                )
            delta_payloads.append(row)
        expected = len(endpoint_required_pair_ids) * len(stages) * len(position_roles)
        actual = len(endpoint_payloads)
        if actual != expected:
            raise RuntimeError(
                f"Residual delta payload count mismatch for {source_component} endpoint={step}/{margin_side}/{endpoint_kind}: "
                f"expected={expected} got={actual}"
            )
    return delta_payloads, subspace_rows


def run_residual_delta_vector_report(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    device_name: str,
    pair_types: list[str],
    source_components: list[str],
    stages: list[str],
    position_roles: list[str],
    group_by_values: list[str],
    scalar_names: list[str] | None,
    margin_sides: list[str] | None,
    endpoint_roles: list[str] | None,
    split_filter: list[str] | None,
    max_pairs_per_type: int,
    min_pairs_per_type: int,
    pca_rank: int,
    markdown_top_k_rows: int,
    plot_top_k_rows: int,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path, Path, dict[str, Path]]:
    if pca_rank <= 0:
        raise ValueError(f"pca_rank must be positive, got {pca_rank}.")
    resolved_margin_sides = _resolve_unique_values(
        values=margin_sides,
        default_values=["clean"],
        allowed_values=OUTPUT_ROUTE_MARGIN_SIDES,
        label="margin side",
    )
    resolved_scalars = _resolve_unique_values(
        values=scalar_names,
        default_values=["correct_value_logit"],
        allowed_values=OUTPUT_ROUTE_SCALARS,
        label="scalar",
    )
    resolved_endpoint_roles = _resolve_unique_values(
        values=endpoint_roles,
        default_values=["source", "target"],
        allowed_values=RESIDUAL_STATE_RESCUE_ENDPOINT_ROLES,
        label="endpoint role",
    )
    resolved_position_roles = _resolve_unique_values(
        values=position_roles,
        default_values=[],
        allowed_values=GEOMETRY_POSITION_ROLES,
        label="position role",
    )
    resolved_group_by = _resolve_unique_values(
        values=group_by_values,
        default_values=[],
        allowed_values=CONTEXTUAL_GROUP_BY_OPTIONS,
        label="group by",
    )
    pair_types = sorted(set(pair_types), key=pair_types.index)

    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    device = require_device(device_name)
    model = build_model(spec.model, len(vocab.tokens), device)
    available_components = _component_labels(num_layers=len(model.blocks), num_heads=model.spec.n_heads)
    resolved_sources = _filter_component_labels(
        requested_components=source_components,
        available_components=available_components,
    )
    _validate_maskable_components(
        components=resolved_sources,
        num_layers=len(model.blocks),
        num_heads=model.spec.n_heads,
        device=device,
    )
    resolved_stages = _resolve_patch_stages(patch_stages=stages, num_layers=len(model.blocks))

    scalar_pair_rows = _filter_scalar_pair_rows_for_delta(
        rows=_load_scalar_pair_rows(scalar_pair_rows_path),
        margin_sides=resolved_margin_sides,
        pair_types=pair_types,
        scalar_names=resolved_scalars,
    )
    checkpoint_paths_by_step = _checkpoint_paths_by_step(scalar_pair_rows)
    endpoint_pair_ids = _endpoint_pair_ids(
        scalar_pair_rows=scalar_pair_rows,
        endpoint_roles=resolved_endpoint_roles,
    )
    required_pair_ids = {str(row["pair_id"]) for row in scalar_pair_rows}
    pairs_by_id, pair_construction = _selected_pairs_by_id(
        config_path=config_path,
        probe_set_path=probe_set_path,
        pair_types=pair_types,
        split_filter=split_filter,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        required_pair_ids=required_pair_ids,
    )

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[residual-delta-vector-report] "
        f"pairs={len(pairs_by_id)} scalar_rows={len(scalar_pair_rows)} endpoints={len(endpoint_pair_ids)} "
        f"sources={resolved_sources} stages={resolved_stages} position_roles={resolved_position_roles} "
        f"group_by={resolved_group_by} pca_rank={pca_rank} device={device_name}",
        flush=True,
    )
    delta_rows: list[dict[str, Any]] = []
    subspace_rows: list[dict[str, Any]] = []
    for source_component in resolved_sources:
        print(f"[residual-delta-vector-report] source={source_component} collecting clean-minus-ablated deltas", flush=True)
        source_delta_rows, source_subspace_rows = _compute_delta_rows_for_source(
            model=model,
            checkpoint_paths_by_step=checkpoint_paths_by_step,
            pairs_by_id=pairs_by_id,
            endpoint_pair_ids=endpoint_pair_ids,
            source_component=source_component,
            stages=resolved_stages,
            position_roles=resolved_position_roles,
            group_by_values=resolved_group_by,
            pca_rank=pca_rank,
            batch_size=spec.evaluation.batch_size,
            pad_token_id=vocab.pad_token_id,
            vocab=vocab,
            device=device,
        )
        delta_rows.extend(source_delta_rows)
        subspace_rows.extend(source_subspace_rows)
    summary_rows = _summarize_delta_rows(delta_rows=delta_rows, group_by_values=resolved_group_by)

    delta_rows_path = output_dir / "residual_delta_vector_rows.jsonl"
    summary_rows_path = output_dir / "residual_delta_vector_summary_rows.jsonl"
    subspace_rows_path = output_dir / "residual_delta_vector_subspaces.jsonl"
    pair_rows_path = output_dir / "residual_delta_vector_pairs.jsonl"
    report_path = output_dir / "residual_delta_vector_report.json"
    markdown_path = output_dir / "residual_delta_vector_report.md"
    write_jsonl(delta_rows_path, delta_rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(subspace_rows_path, subspace_rows)
    write_jsonl(
        pair_rows_path,
        [{key: value for key, value in pair.items() if key not in {"clean_record", "corrupted_record"}} for pair in pairs_by_id.values()],
    )
    plot_paths: dict[str, Path] = {}
    summary_plot = _plot_delta_summary(
        summary_rows=summary_rows,
        output_path=output_dir / "residual_delta_vector_summary.svg",
        top_k_rows=plot_top_k_rows,
    )
    if summary_plot is not None:
        plot_paths["summary"] = summary_plot

    report = {
        "schema_version": RESIDUAL_DELTA_VECTOR_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "device": device_name,
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names_required_in_rows": resolved_scalars,
        "endpoint_roles": resolved_endpoint_roles,
        "source_components": resolved_sources,
        "stages": resolved_stages,
        "position_roles": resolved_position_roles,
        "group_by": resolved_group_by,
        "pca_rank": pca_rank,
        "markdown_top_k_rows": markdown_top_k_rows,
        "plot_top_k_rows": plot_top_k_rows,
        "checkpoint_paths_by_step": {str(step): str(path) for step, path in checkpoint_paths_by_step.items()},
        "pair_construction": pair_construction,
        "calculation": {
            "delta": "clean_residual - source_component_ablated_residual",
            "position_delta": "mean delta over all causal positions selected by position_role",
            "mean_abs_cosine": "absolute cosine between delta and clean contextual mean direction",
            "identity_overlap": "norm of delta projected into the PCA subspace of per-token clean contextual means",
            "all_vector_overlap": "norm of delta projected into the PCA subspace of all clean contextual vectors",
        },
        "delta_rows_path": str(delta_rows_path),
        "summary_rows_path": str(summary_rows_path),
        "subspace_rows_path": str(subspace_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report, plot_paths=plot_paths)
    print(f"[residual-delta-vector-report] complete report={report_path} rows={delta_rows_path}", flush=True)
    return report_path, markdown_path, delta_rows_path, summary_rows_path, subspace_rows_path, pair_rows_path, plot_paths
