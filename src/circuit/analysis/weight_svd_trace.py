from __future__ import annotations

import csv
from pathlib import Path
import re
from typing import Any

import torch

from circuit.config import TrainSpec
from circuit.data.symbolic_kv_stream import read_symbolic_kv_stream_metadata
from circuit.io import write_json, write_jsonl
from circuit.runtime import build_model, load_checkpoint, load_model_state, require_device
from circuit.vocab import Vocabulary


_STEP_CHECKPOINT_RE = re.compile(r"^step_(?P<step>\d+)\.pt$")


def _resolve_checkpoint_paths(*, checkpoint_dir: Path, checkpoint_paths: list[Path] | None) -> list[Path]:
    if checkpoint_paths is not None:
        resolved = [Path(path) for path in checkpoint_paths]
    else:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        resolved = sorted(checkpoint_dir.glob("step_*.pt"))
    if not resolved:
        raise FileNotFoundError(f"No checkpoints provided or found in {checkpoint_dir}")
    missing = [path for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint path(s): {[str(path) for path in missing]}")
    return resolved


def _checkpoint_step_from_payload_and_path(*, checkpoint: dict[str, Any], checkpoint_path: Path) -> int:
    if "step" not in checkpoint:
        raise KeyError(f"Checkpoint payload is missing required key 'step': {checkpoint_path}")
    payload_step = int(checkpoint["step"])
    match = _STEP_CHECKPOINT_RE.match(checkpoint_path.name)
    if match is not None:
        path_step = int(match.group("step"))
        if payload_step != path_step:
            raise RuntimeError(
                f"Checkpoint step mismatch for {checkpoint_path}: payload step={payload_step}, path step={path_step}"
            )
    return payload_step


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        output_dir / "weight_svd_singular_values.jsonl",
        output_dir / "weight_svd_singular_values.csv",
        output_dir / "weight_svd_top_singular_vectors.jsonl",
        output_dir / "weight_svd_trace_report.json",
        output_dir / "weight_svd_trace_report.md",
    ]
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing weight SVD trace outputs without --overwrite: "
            f"{[str(path) for path in existing]}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _append_svd_rows(
    *,
    singular_value_rows: list[dict[str, Any]],
    vector_rows: list[dict[str, Any]],
    checkpoint_path: Path,
    checkpoint_step: int,
    layer: int,
    head: int | None,
    component_type: str,
    matrix_name: str,
    matrix: torch.Tensor,
    left_vector_label: str,
    right_vector_label: str,
    max_singular_values: int | None,
    top_vector_ranks: int,
) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"Expected rank-2 matrix for {matrix_name}, got shape {tuple(matrix.shape)}")
    matrix_cpu = matrix.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(matrix_cpu).all():
        raise RuntimeError(f"Non-finite value encountered in {matrix_name} at checkpoint {checkpoint_path}")
    u, singular_values, vh = torch.linalg.svd(matrix_cpu, full_matrices=False)
    available = int(singular_values.numel())
    if available <= 0:
        raise RuntimeError(f"SVD produced no singular values for {matrix_name} at checkpoint {checkpoint_path}")
    if max_singular_values is None:
        keep = available
    else:
        if max_singular_values <= 0:
            raise ValueError("--max-singular-values must be positive when provided.")
        keep = min(available, max_singular_values)
    if top_vector_ranks <= 0:
        raise ValueError("--top-vector-ranks must be positive.")
    vector_keep = min(available, top_vector_ranks)

    singular_value_sum = float(singular_values.sum().item())
    singular_value_sq_sum = float(singular_values.square().sum().item())
    if singular_value_sq_sum <= 0.0:
        raise RuntimeError(f"Cannot compute effective rank for zero singular spectrum: {matrix_name} at {checkpoint_path}")
    effective_rank = (singular_value_sum * singular_value_sum) / singular_value_sq_sum
    spectral_mass_top3 = float((singular_values[: min(3, available)].sum() / singular_values.sum()).item())

    base_row = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "step": int(checkpoint_step),
        "layer": int(layer),
        "head": None if head is None else int(head),
        "component_type": component_type,
        "matrix_name": matrix_name,
        "matrix_rows": int(matrix_cpu.shape[0]),
        "matrix_cols": int(matrix_cpu.shape[1]),
        "singular_value_count": available,
        "singular_value_sum": singular_value_sum,
        "singular_value_sq_sum": singular_value_sq_sum,
        "effective_rank": effective_rank,
        "spectral_mass_top3": spectral_mass_top3,
    }
    for rank_index, value in enumerate(singular_values[:keep].tolist(), start=1):
        singular_value_rows.append(
            {
                **base_row,
                "singular_value_rank": rank_index,
                "singular_value": float(value),
            }
        )

    for rank_index in range(1, vector_keep + 1):
        vector_rows.append(
            {
                **base_row,
                "singular_value_rank": rank_index,
                "singular_value": float(singular_values[rank_index - 1].item()),
                "vector_side": "left",
                "vector_label": left_vector_label,
                "vector_dim": int(u.shape[0]),
                "vector": [float(value) for value in u[:, rank_index - 1].tolist()],
            }
        )
        vector_rows.append(
            {
                **base_row,
                "singular_value_rank": rank_index,
                "singular_value": float(singular_values[rank_index - 1].item()),
                "vector_side": "right",
                "vector_label": right_vector_label,
                "vector_dim": int(vh.shape[1]),
                "vector": [float(value) for value in vh[rank_index - 1, :].tolist()],
            }
        )


def _extract_checkpoint_svd_rows(
    *,
    model: torch.nn.Module,
    checkpoint_path: Path,
    checkpoint_step: int,
    max_singular_values: int | None,
    top_vector_ranks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    singular_value_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    for layer_index, block in enumerate(model.blocks):
        n_heads = int(block.attn.n_heads)
        head_dim = int(block.attn.head_dim)
        q_weight = block.attn.q_proj.weight.detach()
        k_weight = block.attn.k_proj.weight.detach()
        v_weight = block.attn.v_proj.weight.detach()
        out_weight = block.attn.out_proj.weight.detach()
        for head_index in range(n_heads):
            head_slice = slice(head_index * head_dim, (head_index + 1) * head_dim)
            q_rows = q_weight[head_slice, :]
            k_rows = k_weight[head_slice, :]
            v_rows = v_weight[head_slice, :]
            out_head = out_weight[:, head_slice]
            qk_matrix = q_rows.T.matmul(k_rows)
            ov_matrix = v_rows.T.matmul(out_head.T)
            _append_svd_rows(
                singular_value_rows=singular_value_rows,
                vector_rows=vector_rows,
                checkpoint_path=checkpoint_path,
                checkpoint_step=checkpoint_step,
                layer=layer_index,
                head=head_index,
                component_type="attention_head",
                matrix_name="W_QK",
                matrix=qk_matrix,
                left_vector_label="query_residual_direction",
                right_vector_label="key_residual_direction",
                max_singular_values=max_singular_values,
                top_vector_ranks=top_vector_ranks,
            )
            _append_svd_rows(
                singular_value_rows=singular_value_rows,
                vector_rows=vector_rows,
                checkpoint_path=checkpoint_path,
                checkpoint_step=checkpoint_step,
                layer=layer_index,
                head=head_index,
                component_type="attention_head",
                matrix_name="W_OV",
                matrix=ov_matrix,
                left_vector_label="input_residual_direction",
                right_vector_label="output_residual_direction",
                max_singular_values=max_singular_values,
                top_vector_ranks=top_vector_ranks,
            )

        _append_svd_rows(
            singular_value_rows=singular_value_rows,
            vector_rows=vector_rows,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
            layer=layer_index,
            head=None,
            component_type="mlp",
            matrix_name="W_in",
            matrix=block.ff.fc_in.weight.detach(),
            left_vector_label="mlp_hidden_direction",
            right_vector_label="input_residual_direction",
            max_singular_values=max_singular_values,
            top_vector_ranks=top_vector_ranks,
        )
        _append_svd_rows(
            singular_value_rows=singular_value_rows,
            vector_rows=vector_rows,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
            layer=layer_index,
            head=None,
            component_type="mlp",
            matrix_name="W_out",
            matrix=block.ff.fc_out.weight.detach(),
            left_vector_label="output_residual_direction",
            right_vector_label="mlp_hidden_direction",
            max_singular_values=max_singular_values,
            top_vector_ranks=top_vector_ranks,
        )
    return singular_value_rows, vector_rows


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Weight SVD Trace",
        "",
        "This report extracts clean SVD trajectories from checkpoint weights only. It does not use activations, probes, or plots.",
        "",
        "## Outputs",
        "",
        f"- singular values JSONL: `{report['singular_values_jsonl_path']}`",
        f"- singular values CSV: `{report['singular_values_csv_path']}`",
        f"- top singular vectors JSONL: `{report['top_singular_vectors_jsonl_path']}`",
        "",
        "## Scope",
        "",
        f"- checkpoints: `{report['num_checkpoints']}`",
        f"- singular value rows: `{report['num_singular_value_rows']}`",
        f"- top singular vector rows: `{report['num_top_singular_vector_rows']}`",
        f"- max singular values per matrix: `{report['max_singular_values']}`",
        f"- top singular vector ranks per matrix side: `{report['top_vector_ranks']}`",
        "",
        "## Matrix Conventions",
        "",
        "- `W_QK` uses the repo's functional attention-score orientation: `q_rows.T @ k_rows`.",
        "- `W_OV` uses the repo's functional value-write orientation: `v_rows.T @ out_head.T`.",
        "- Top singular vectors are written for both `left` and `right` sides so later runs can track rotations.",
        "",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def run_weight_svd_trace(
    *,
    config_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    device_name: str = "cpu",
    checkpoint_paths: list[Path] | None = None,
    max_singular_values: int | None = None,
    top_vector_ranks: int = 1,
    overwrite: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    _prepare_output_dir(output_dir, overwrite=overwrite)
    device = require_device(device_name)
    spec = TrainSpec.from_path(config_path)
    metadata = read_symbolic_kv_stream_metadata(spec.benchmark_dir)
    vocab = Vocabulary.from_metadata(metadata["vocabulary"])
    model = build_model(spec.model, len(vocab.tokens), device)
    model.eval()

    checkpoints = _resolve_checkpoint_paths(checkpoint_dir=checkpoint_dir, checkpoint_paths=checkpoint_paths)
    singular_values_jsonl_path = output_dir / "weight_svd_singular_values.jsonl"
    singular_values_csv_path = output_dir / "weight_svd_singular_values.csv"
    top_vectors_jsonl_path = output_dir / "weight_svd_top_singular_vectors.jsonl"
    report_path = output_dir / "weight_svd_trace_report.json"
    markdown_path = output_dir / "weight_svd_trace_report.md"

    all_singular_value_rows: list[dict[str, Any]] = []
    all_vector_rows: list[dict[str, Any]] = []
    print(
        "[weight-svd-trace] "
        f"checkpoints={len(checkpoints)} device={device_name} "
        f"max_singular_values={max_singular_values} top_vector_ranks={top_vector_ranks}",
        flush=True,
    )
    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        print(f"[weight-svd-trace] starting {checkpoint_index}/{len(checkpoints)} {checkpoint_path.name}", flush=True)
        checkpoint = load_checkpoint(checkpoint_path, device)
        load_model_state(model, checkpoint["model_state"])
        checkpoint_step = _checkpoint_step_from_payload_and_path(checkpoint=checkpoint, checkpoint_path=checkpoint_path)
        singular_value_rows, vector_rows = _extract_checkpoint_svd_rows(
            model=model,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
            max_singular_values=max_singular_values,
            top_vector_ranks=top_vector_ranks,
        )
        all_singular_value_rows.extend(singular_value_rows)
        all_vector_rows.extend(vector_rows)
        print(
            "[weight-svd-trace] finished "
            f"step={checkpoint_step} singular_rows={len(singular_value_rows)} vector_rows={len(vector_rows)}",
            flush=True,
        )

    write_jsonl(singular_values_jsonl_path, all_singular_value_rows)
    _write_csv(
        singular_values_csv_path,
        all_singular_value_rows,
        fieldnames=[
            "checkpoint",
            "checkpoint_name",
            "step",
            "layer",
            "head",
            "component_type",
            "matrix_name",
            "matrix_rows",
            "matrix_cols",
            "singular_value_rank",
            "singular_value",
            "singular_value_count",
            "singular_value_sum",
            "singular_value_sq_sum",
            "effective_rank",
            "spectral_mass_top3",
        ],
    )
    write_jsonl(top_vectors_jsonl_path, all_vector_rows)

    matrix_keys = {
        (
            int(row["step"]),
            int(row["layer"]),
            None if row["head"] is None else int(row["head"]),
            str(row["component_type"]),
            str(row["matrix_name"]),
        )
        for row in all_singular_value_rows
    }
    report = {
        "config_path": str(config_path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "output_dir": str(output_dir),
        "device": device_name,
        "num_checkpoints": len(checkpoints),
        "num_matrices": len(matrix_keys),
        "num_singular_value_rows": len(all_singular_value_rows),
        "num_top_singular_vector_rows": len(all_vector_rows),
        "max_singular_values": max_singular_values,
        "top_vector_ranks": top_vector_ranks,
        "singular_values_jsonl_path": str(singular_values_jsonl_path),
        "singular_values_csv_path": str(singular_values_csv_path),
        "top_singular_vectors_jsonl_path": str(top_vectors_jsonl_path),
        "matrix_conventions": {
            "W_QK": "q_rows.T @ k_rows, matching the repo's attention-score functional orientation",
            "W_OV": "v_rows.T @ out_head.T, matching the repo's value-write functional orientation",
            "svd": "matrix = U diag(S) Vh; both left and right rank-1 vectors are saved",
        },
    }
    write_json(report_path, report)
    _write_markdown_report(markdown_path, report)
    print(f"[weight-svd-trace] complete report={report_path} rows={singular_values_jsonl_path}", flush=True)
    return report_path, markdown_path, singular_values_jsonl_path, singular_values_csv_path, top_vectors_jsonl_path
