from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil
from typing import Any

from circuit.analysis.contextual_svd_alignment import CONTEXTUAL_GROUP_BY_OPTIONS, _subspace_overlap
from circuit.analysis.geometric_mechanisms import GEOMETRY_POSITION_ROLES
from circuit.analysis.mlp_input_functional_subspace_report import (
    _compute_functional_rows,
    _grouped_summary,
    _mlp_input_stage,
    _pca_basis,
)
from circuit.analysis.output_route_closure import (
    OUTPUT_ROUTE_MARGIN_SIDES,
    OUTPUT_ROUTE_SCALARS,
    _checkpoint_paths_by_step,
    _component_labels,
    _filter_component_labels,
    _load_scalar_pair_rows,
    _mean,
    _resolve_unique_values,
    _selected_pairs_by_id,
)
from circuit.analysis.residual_delta_vector_report import _endpoint_pair_ids, _filter_scalar_pair_rows_for_delta
from circuit.analysis.residual_state_rescue import RESIDUAL_STATE_RESCUE_ENDPOINT_ROLES, _validate_maskable_components
from circuit.analysis.component_output_rescue import _component_patch_stage
from circuit.analysis.mlp_local_write_map_report import _validate_mlp_components
from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, require_device
from circuit.vocab import Vocabulary


MLP_FUNCTIONAL_SUBSPACE_TRAJECTORY_SCHEMA_VERSION = 1
FUNCTIONAL_BASIS_KINDS = [
    "input_delta",
    "input_gradient",
    "post_mlp_gradient",
    "mlp_output_delta",
]
_VECTOR_FIELD_BY_KIND = {
    "input_delta": "_input_delta_mean",
    "input_gradient": "_input_gradient_mean",
    "post_mlp_gradient": "_post_mlp_gradient_mean",
    "mlp_output_delta": "_mlp_output_delta_mean",
}


def _trajectory_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(row["source_component"]),
        str(row["mlp_component"]),
        str(row["margin_side"]),
        str(row["scalar_name"]),
        str(row["position_role"]),
        str(row["group_by"]),
        str(row["input_stage"]),
    )


def _build_reference_bases(
    *,
    functional_rows: list[dict[str, Any]],
    reference_steps: list[int],
    basis_kinds: list[str],
    subspace_rank: int,
    ) -> tuple[dict[tuple[tuple[str, str, str, str, str, str, str], str], Any], list[dict[str, Any]]]:
    reference_step_set = {int(step) for step in reference_steps}
    grouped_vectors: dict[tuple[tuple[str, str, str, str, str, str, str], str], list[Any]] = defaultdict(list)
    for row in functional_rows:
        if int(row["step"]) not in reference_step_set:
            continue
        key = _trajectory_key(row)
        for basis_kind in basis_kinds:
            field = _VECTOR_FIELD_BY_KIND[basis_kind]
            if field not in row:
                raise KeyError(f"Functional row is missing vector field {field!r}.")
            grouped_vectors[(key, basis_kind)].append(row[field])

    expected_keys = {_trajectory_key(row) for row in functional_rows}
    bases: dict[tuple[tuple[str, str, str, str, str, str, str], str], Any] = {}
    basis_rows: list[dict[str, Any]] = []
    for key in sorted(expected_keys):
        for basis_kind in basis_kinds:
            basis_key = (key, basis_kind)
            vectors = grouped_vectors.get(basis_key)
            if not vectors:
                raise RuntimeError(
                    f"No reference vectors for key={key} basis_kind={basis_kind} "
                    f"reference_steps={sorted(reference_step_set)}."
                )
            basis, summary = _pca_basis(vectors, rank=subspace_rank, label=f"{key}/{basis_kind}/reference")
            bases[basis_key] = basis
            (
                source_component,
                mlp_component,
                margin_side,
                scalar_name,
                position_role,
                group_by,
                input_stage,
            ) = key
            basis_rows.append(
                {
                    "source_component": source_component,
                    "mlp_component": mlp_component,
                    "margin_side": margin_side,
                    "scalar_name": scalar_name,
                    "position_role": position_role,
                    "group_by": group_by,
                    "input_stage": input_stage,
                    "basis_kind": basis_kind,
                    "reference_steps": sorted(reference_step_set),
                    **summary,
                }
            )
    return bases, basis_rows


def _score_trajectory_rows(
    *,
    functional_rows: list[dict[str, Any]],
    bases: dict[tuple[tuple[str, str, str, str, str, str, str], str], Any],
    basis_kinds: list[str],
) -> list[dict[str, Any]]:
    trajectory_rows: list[dict[str, Any]] = []
    for row in functional_rows:
        clean_row = {key: value for key, value in row.items() if not key.startswith("_")}
        key = _trajectory_key(row)
        for basis_kind in basis_kinds:
            basis_key = (key, basis_kind)
            if basis_key not in bases:
                raise KeyError(f"Missing reference basis for {basis_key}.")
            basis = bases[basis_key]
            scored_row = dict(clean_row)
            scored_row["reference_basis_kind"] = basis_kind
            for vector_kind, field in _VECTOR_FIELD_BY_KIND.items():
                if field not in row:
                    raise KeyError(f"Functional row is missing vector field {field!r}.")
                scored_row[f"{vector_kind}_overlap_with_reference_{basis_kind}_basis"] = _subspace_overlap(
                    row[field],
                    basis,
                    label=f"{row['pair_id']}/{row['step']}/{vector_kind}/reference_{basis_kind}",
                )
            trajectory_rows.append(scored_row)
    return trajectory_rows


def _summarize_trajectory_rows(*, trajectory_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory_rows:
        grouped[
            (
                str(row["source_component"]),
                str(row["mlp_component"]),
                int(row["step"]),
                str(row["endpoint_kind"]),
                str(row["margin_side"]),
                str(row["scalar_name"]),
                str(row["position_role"]),
                str(row["group_by"]),
                str(row["reference_basis_kind"]),
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        (
            source_component,
            mlp_component,
            step,
            endpoint_kind,
            margin_side,
            scalar_name,
            position_role,
            group_by,
            reference_basis_kind,
        ) = key
        input_dot_values = [float(row["input_stage_gradient_dot_input_delta"]) for row in rows]
        skip_dot_values = [float(row["post_mlp_gradient_dot_skip_delta"]) for row in rows]
        mlp_dot_values = [float(row["post_mlp_gradient_dot_mlp_output_delta"]) for row in rows]
        total_dot_values = [float(row["post_mlp_gradient_dot_total_delta"]) for row in rows]
        summary = {
            "source_component": source_component,
            "mlp_component": mlp_component,
            "step": step,
            "endpoint_kind": endpoint_kind,
            "margin_side": margin_side,
            "scalar_name": scalar_name,
            "input_stage": str(rows[0]["input_stage"]),
            "output_stage": str(rows[0]["output_stage"]),
            "position_role": position_role,
            "group_by": group_by,
            "reference_basis_kind": reference_basis_kind,
            "num_rows": len(rows),
            "sum_input_stage_gradient_dot_input_delta": sum(input_dot_values),
            "sum_post_mlp_gradient_dot_skip_delta": sum(skip_dot_values),
            "sum_post_mlp_gradient_dot_mlp_output_delta": sum(mlp_dot_values),
            "sum_post_mlp_gradient_dot_total_delta": sum(total_dot_values),
        }
        for vector_kind in FUNCTIONAL_BASIS_KINDS:
            column = f"{vector_kind}_overlap_with_reference_{reference_basis_kind}_basis"
            values = [float(row[column]) for row in rows]
            summary[f"mean_{column}"] = _mean(values, label=f"{key}/{column}")
        summary_rows.append(summary)
    return summary_rows


def _write_markdown(*, path: Path, report: dict[str, Any]) -> None:
    rows = sorted(
        report["summary_rows"],
        key=lambda row: (
            str(row["reference_basis_kind"]) != "input_gradient",
            -abs(float(row["sum_post_mlp_gradient_dot_total_delta"])),
        ),
    )[: int(report["markdown_top_k_rows"])]
    lines = [
        "# MLP Functional Subspace Trajectory",
        "",
        "This report fixes a reference functional subspace and measures how strongly each checkpoint lands in it.",
        "",
        "The reference basis is built from the requested reference step(s). Each checkpoint is scored against the same basis.",
        "",
        "## Top Rows",
        "",
    ]
    if rows:
        lines.extend(
            [
                "| mlp | step | endpoint | scalar | position | group | basis | total dot | input_delta overlap | input_gradient overlap | mlp_output overlap |",
                "|---|---:|---|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            basis_kind = str(row["reference_basis_kind"])
            lines.append(
                f"| `{row['mlp_component']}` | {row['step']} | `{row['endpoint_kind']}` | "
                f"`{row['scalar_name']}` | `{row['position_role']}` | `{row['group_by']}` | "
                f"`{basis_kind}` | {row['sum_post_mlp_gradient_dot_total_delta']:.6g} | "
                f"{row[f'mean_input_delta_overlap_with_reference_{basis_kind}_basis']:.6g} | "
                f"{row[f'mean_input_gradient_overlap_with_reference_{basis_kind}_basis']:.6g} | "
                f"{row[f'mean_mlp_output_delta_overlap_with_reference_{basis_kind}_basis']:.6g} |"
            )
    else:
        lines.append("No summary rows were produced.")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- trajectory rows: `{report['trajectory_rows_path']}`",
            f"- summary rows: `{report['summary_rows_path']}`",
            f"- reference basis rows: `{report['basis_rows_path']}`",
            f"- functional summary rows: `{report['functional_summary_rows_path']}`",
            f"- pair rows: `{report['pair_rows_path']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_mlp_functional_subspace_trajectory_report(
    *,
    config_path: Path,
    probe_set_path: Path,
    scalar_pair_rows_path: Path,
    output_dir: Path,
    device_name: str,
    pair_types: list[str],
    source_component: str,
    mlp_component: str,
    position_roles: list[str],
    group_by_values: list[str],
    scalar_names: list[str] | None,
    margin_sides: list[str] | None,
    endpoint_roles: list[str] | None,
    endpoint_steps: list[int] | None,
    reference_steps: list[int],
    reference_basis_kinds: list[str],
    split_filter: list[str] | None,
    max_pairs_per_type: int,
    min_pairs_per_type: int,
    batch_size: int | None,
    subspace_rank: int,
    markdown_top_k_rows: int,
    overwrite: bool,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    if not reference_steps:
        raise ValueError("At least one reference step is required.")
    if subspace_rank <= 0:
        raise ValueError("subspace_rank must be positive.")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    resolved_reference_basis_kinds = _resolve_unique_values(
        values=reference_basis_kinds,
        default_values=[],
        allowed_values=FUNCTIONAL_BASIS_KINDS,
        label="reference basis kind",
    )
    resolved_margin_sides = _resolve_unique_values(
        values=margin_sides,
        default_values=["clean"],
        allowed_values=OUTPUT_ROUTE_MARGIN_SIDES,
        label="margin side",
    )
    resolved_scalars = _resolve_unique_values(
        values=scalar_names,
        default_values=["fixed_source_competitor_margin", "fixed_target_competitor_margin"],
        allowed_values=OUTPUT_ROUTE_SCALARS,
        label="scalar",
    )
    resolved_endpoint_roles = _resolve_unique_values(
        values=endpoint_roles,
        default_values=["source"],
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
        requested_components=[source_component],
        available_components=available_components,
    )
    if len(resolved_sources) != 1:
        raise RuntimeError(f"Expected exactly one source component, got {resolved_sources}.")
    resolved_mlps = _filter_component_labels(
        requested_components=[mlp_component],
        available_components=available_components,
    )
    if len(resolved_mlps) != 1:
        raise RuntimeError(f"Expected exactly one MLP component, got {resolved_mlps}.")
    resolved_source = resolved_sources[0]
    resolved_mlp = resolved_mlps[0]
    _validate_mlp_components(components=[resolved_mlp])
    _validate_maskable_components(
        components=[resolved_source],
        num_layers=len(model.blocks),
        num_heads=model.spec.n_heads,
        device=device,
    )

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
    if endpoint_steps is not None:
        endpoint_step_set = set(int(step) for step in endpoint_steps)
        endpoint_pair_ids = {
            key: value
            for key, value in endpoint_pair_ids.items()
            if int(key[0]) in endpoint_step_set
        }
        if not endpoint_pair_ids:
            raise RuntimeError(f"Endpoint step filter matched no endpoints: {sorted(endpoint_step_set)}")
    endpoint_step_set = {int(key[0]) for key in endpoint_pair_ids}
    missing_reference_steps = sorted(set(int(step) for step in reference_steps) - endpoint_step_set)
    if missing_reference_steps:
        raise RuntimeError(
            f"Reference step(s) are not present in selected endpoints: {missing_reference_steps}. "
            f"Selected endpoint steps: {sorted(endpoint_step_set)[:20]}..."
        )

    scalar_pair_rows_by_endpoint: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    allowed_endpoint_keys = set(endpoint_pair_ids)
    for row in scalar_pair_rows:
        margin_side = str(row["margin_side"])
        for endpoint_kind in resolved_endpoint_roles:
            key = (int(row[f"{endpoint_kind}_step"]), margin_side, endpoint_kind)
            if key in allowed_endpoint_keys:
                scalar_pair_rows_by_endpoint[key].append(row)
    missing_scalar_endpoints = sorted(allowed_endpoint_keys - set(scalar_pair_rows_by_endpoint))
    if missing_scalar_endpoints:
        raise RuntimeError(f"Missing scalar rows for endpoint(s): {missing_scalar_endpoints}")

    required_pair_ids = {pair_id for pair_ids in endpoint_pair_ids.values() for pair_id in pair_ids}
    pairs_by_id, pair_construction = _selected_pairs_by_id(
        config_path=config_path,
        probe_set_path=probe_set_path,
        pair_types=pair_types,
        split_filter=split_filter,
        max_pairs_per_type=max_pairs_per_type,
        min_pairs_per_type=min_pairs_per_type,
        required_pair_ids=required_pair_ids,
    )
    analysis_batch_size = int(spec.evaluation.batch_size) if batch_size is None else int(batch_size)

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "[mlp-functional-subspace-trajectory-report] "
        f"pairs={len(pairs_by_id)} endpoints={len(endpoint_pair_ids)} source={resolved_source} mlp={resolved_mlp} "
        f"reference_steps={sorted(set(int(step) for step in reference_steps))} "
        f"basis_kinds={resolved_reference_basis_kinds} position_roles={resolved_position_roles} "
        f"group_by={resolved_group_by} scalars={resolved_scalars} subspace_rank={subspace_rank} device={device_name}",
        flush=True,
    )
    functional_rows = _compute_functional_rows(
        model=model,
        checkpoint_paths_by_step=checkpoint_paths_by_step,
        pairs_by_id=pairs_by_id,
        scalar_pair_rows_by_endpoint=scalar_pair_rows_by_endpoint,
        endpoint_pair_ids=endpoint_pair_ids,
        source_component=resolved_source,
        mlp_component=resolved_mlp,
        position_roles=resolved_position_roles,
        group_by_values=resolved_group_by,
        scalar_names=resolved_scalars,
        batch_size=analysis_batch_size,
        pad_token_id=vocab.pad_token_id,
        vocab=vocab,
        device=device,
    )
    bases, basis_rows = _build_reference_bases(
        functional_rows=functional_rows,
        reference_steps=reference_steps,
        basis_kinds=resolved_reference_basis_kinds,
        subspace_rank=subspace_rank,
    )
    trajectory_rows = _score_trajectory_rows(
        functional_rows=functional_rows,
        bases=bases,
        basis_kinds=resolved_reference_basis_kinds,
    )
    summary_rows = _summarize_trajectory_rows(trajectory_rows=trajectory_rows)
    functional_summary_rows, _ = _grouped_summary(rows=functional_rows, subspace_rank=subspace_rank)

    trajectory_rows_path = output_dir / "mlp_functional_subspace_trajectory_rows.jsonl"
    summary_rows_path = output_dir / "mlp_functional_subspace_trajectory_summary_rows.jsonl"
    basis_rows_path = output_dir / "mlp_functional_subspace_trajectory_reference_bases.jsonl"
    functional_summary_rows_path = output_dir / "mlp_functional_subspace_trajectory_functional_summary_rows.jsonl"
    pair_rows_path = output_dir / "mlp_functional_subspace_trajectory_pairs.jsonl"
    report_path = output_dir / "mlp_functional_subspace_trajectory_report.json"
    markdown_path = output_dir / "mlp_functional_subspace_trajectory_report.md"

    write_jsonl(trajectory_rows_path, trajectory_rows)
    write_jsonl(summary_rows_path, summary_rows)
    write_jsonl(basis_rows_path, basis_rows)
    write_jsonl(functional_summary_rows_path, functional_summary_rows)
    write_jsonl(
        pair_rows_path,
        [{key: value for key, value in pair.items() if key not in {"clean_record", "corrupted_record"}} for pair in pairs_by_id.values()],
    )

    report = {
        "schema_version": MLP_FUNCTIONAL_SUBSPACE_TRAJECTORY_SCHEMA_VERSION,
        "config_path": str(config_path),
        "probe_set_path": str(probe_set_path),
        "scalar_pair_rows_path": str(scalar_pair_rows_path),
        "device": device_name,
        "pair_types": pair_types,
        "margin_sides": resolved_margin_sides,
        "scalar_names": resolved_scalars,
        "endpoint_roles": resolved_endpoint_roles,
        "endpoint_steps": None if endpoint_steps is None else sorted(set(int(step) for step in endpoint_steps)),
        "reference_steps": sorted(set(int(step) for step in reference_steps)),
        "reference_basis_kinds": resolved_reference_basis_kinds,
        "source_component": resolved_source,
        "mlp_component": resolved_mlp,
        "input_stage": _mlp_input_stage(resolved_mlp),
        "output_stage": _component_patch_stage(resolved_mlp),
        "position_roles": resolved_position_roles,
        "group_by": resolved_group_by,
        "subspace_rank": subspace_rank,
        "batch_size": analysis_batch_size,
        "markdown_top_k_rows": markdown_top_k_rows,
        "checkpoint_paths_by_step": {str(step): str(path) for step, path in checkpoint_paths_by_step.items()},
        "pair_construction": pair_construction,
        "calculation": {
            "reference_basis": "PCA basis built from selected vector kind at reference step(s).",
            "trajectory_overlap": "Norm of each checkpoint vector projected into the fixed reference basis.",
            "functional_vectors": list(_VECTOR_FIELD_BY_KIND),
        },
        "trajectory_rows_path": str(trajectory_rows_path),
        "summary_rows_path": str(summary_rows_path),
        "basis_rows_path": str(basis_rows_path),
        "functional_summary_rows_path": str(functional_summary_rows_path),
        "pair_rows_path": str(pair_rows_path),
        "summary_rows": summary_rows,
    }
    write_json(report_path, report)
    _write_markdown(path=markdown_path, report=report)
    print(
        f"[mlp-functional-subspace-trajectory-report] complete report={report_path} rows={trajectory_rows_path}",
        flush=True,
    )
    return (
        report_path,
        markdown_path,
        trajectory_rows_path,
        summary_rows_path,
        basis_rows_path,
        functional_summary_rows_path,
        pair_rows_path,
    )
