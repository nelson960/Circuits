from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from circut.io import iter_jsonl, read_json, write_json


def _require_non_empty_rows(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError(f"No rows found in checkpoint-sweep metrics file: {path}")
    return rows


def _unique_top_items(
    items: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    metric_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in items:
        key = tuple(item[field] for field in key_fields)
        current_best = best_by_key.get(key)
        if current_best is None or float(item[metric_name]) > float(current_best[metric_name]):
            best_by_key[key] = item
    return sorted(best_by_key.values(), key=lambda item: float(item[metric_name]), reverse=True)[:top_k]


def _window_rows(
    rows: list[dict[str, Any]],
    *,
    start_step: int,
    end_step: int,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if start_step <= int(row["step"]) <= end_step]
    if not selected:
        raise ValueError(f"No sweep rows found for window {start_step}-{end_step}.")
    return selected


def _aggregate_slice(
    rows: list[dict[str, Any]],
    field_name: str,
    *,
    key_fields: tuple[str, ...],
    metric_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        for item in row[field_name]:
            flattened.append(
                {
                    **item,
                    "step": int(row["step"]),
                }
            )
    return _unique_top_items(flattened, key_fields=key_fields, metric_name=metric_name, top_k=top_k)


def _stage_summary(rows: list[dict[str, Any]], field_name: str) -> list[dict[str, Any]]:
    stage_scores: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        for stage_name, score in row[field_name].items():
            stage_scores[str(stage_name)].append((int(row["step"]), float(score)))
    summary: list[dict[str, Any]] = []
    for stage_name, values in sorted(stage_scores.items()):
        best_step, best_score = max(values, key=lambda item: item[1])
        summary.append(
            {
                "stage": stage_name,
                "max_value": best_score,
                "max_step": best_step,
                "mean_value": sum(score for _, score in values) / len(values),
            }
        )
    return sorted(summary, key=lambda item: float(item["max_value"]), reverse=True)


def _drift_summary(rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    stage_scores: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        for stage_name, score in row["prediction_state_drift_by_stage"].items():
            stage_scores[str(stage_name)].append((int(row["step"]), float(score)))
    summary: list[dict[str, Any]] = []
    for stage_name, values in sorted(stage_scores.items()):
        max_step, max_score = max(values, key=lambda item: item[1])
        summary.append(
            {
                "stage": stage_name,
                "max_drift": max_score,
                "max_step": max_step,
                "mean_drift": sum(score for _, score in values) / len(values),
            }
        )
    return sorted(summary, key=lambda item: float(item["max_drift"]), reverse=True)[:top_k]


def _layer_status(
    *,
    name: str,
    status: str,
    evidence: list[str],
    next_additions: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "evidence": evidence,
        "next_additions": next_additions,
    }


def build_layered_plan_status() -> list[dict[str, Any]]:
    return [
        _layer_status(
            name="probe_input",
            status="implemented",
            evidence=["fixed probe-set generation", "benchmark/probe consistency checks"],
            next_additions=["anchor prompt families", "explicit easy/hard tags", "fixed counterfactual patch pairs"],
        ),
        _layer_status(
            name="behavior",
            status="implemented",
            evidence=["answer_accuracy", "heldout_answer_accuracy", "Q/R/W", "token-role accuracies"],
            next_additions=["slice accuracy in sweep rows", "answer entropy", "calibration"],
        ),
        _layer_status(
            name="residual_stream",
            status="implemented",
            evidence=["answer/query/support probes by stage", "answer margins by stage", "checkpoint drift"],
            next_additions=["CKA", "drift-to-reference", "stale-value probe", "overwrite probe"],
        ),
        _layer_status(
            name="heads",
            status="implemented",
            evidence=["head localization", "head ablation"],
            next_additions=["full per-head tables", "turnover", "attention entropy", "QK/OV decomposition"],
        ),
        _layer_status(
            name="mlp_blocks",
            status="implemented",
            evidence=["MLP block ablation", "write norms", "answer-margin delta"],
            next_additions=["direct logit effect", "block interaction scores", "write-direction drift"],
        ),
        _layer_status(
            name="neurons",
            status="partial",
            evidence=["candidate-layer neuron write screening", "top-neuron ablation on candidate layers"],
            next_additions=["neuron birth time", "neuron drift", "direct logit effect", "co-activation groups", "neuron patching"],
        ),
        _layer_status(
            name="features",
            status="partial",
            evidence=["targeted SAE-based residual-stage feature analysis"],
            next_additions=["birth-window feature comparison", "MLP-hidden feature decomposition", "feature-level causal patching"],
        ),
        _layer_status(
            name="causal_validation",
            status="missing",
            evidence=[],
            next_additions=["activation patching", "path patching", "residual-state patching", "faithfulness tests"],
        ),
        _layer_status(
            name="geometry_dynamics",
            status="partial",
            evidence=["checkpoint deltas", "birth-window detection", "residual drift"],
            next_additions=["formal birth detector", "update alignment", "weight geometry", "state-space fits"],
        ),
        _layer_status(
            name="cross_run",
            status="missing",
            evidence=[],
            next_additions=["cross-seed alignment", "factor-sweep aggregation", "circuit family comparison"],
        ),
    ]


def analyze_birth_windows(
    *,
    sweep_metrics_path: Path,
    sweep_summary_path: Path,
    output_path: Path,
    top_k: int = 6,
) -> Path:
    rows = _require_non_empty_rows(list(iter_jsonl(sweep_metrics_path)), sweep_metrics_path)
    summary = read_json(sweep_summary_path)
    probe_metadata = read_json(Path(str(summary["probe_set_path"])).with_suffix(".metadata.json"))
    windows = list(summary["birth_windows"]["windows"])
    if not windows:
        raise ValueError(f"No birth windows found in sweep summary: {sweep_summary_path}")

    window_reports: list[dict[str, Any]] = []
    for window in windows:
        start_step = int(window["start_step"])
        center_step = int(window["center_step"])
        end_step = int(window["end_step"])
        selected_rows = _window_rows(rows, start_step=start_step, end_step=end_step)
        first_row = min(selected_rows, key=lambda row: int(row["step"]))
        last_row = max(selected_rows, key=lambda row: int(row["step"]))
        best_answer_row = max(selected_rows, key=lambda row: float(row["answer_accuracy"]))
        best_heldout_row = max(selected_rows, key=lambda row: float(row["heldout_answer_accuracy"]))
        max_delta_answer_row = max(selected_rows, key=lambda row: float(row["delta_answer_accuracy"]))
        max_delta_heldout_row = max(selected_rows, key=lambda row: float(row["delta_heldout_answer_accuracy"]))

        window_reports.append(
            {
                "window": {
                    "start_step": start_step,
                    "center_step": center_step,
                    "end_step": end_step,
                },
                "selected_checkpoints": {
                    "first_step": int(first_row["step"]),
                    "best_answer_step": int(best_answer_row["step"]),
                    "best_heldout_step": int(best_heldout_row["step"]),
                    "last_step": int(last_row["step"]),
                },
                "behavior": {
                    "first": {
                        "answer_accuracy": float(first_row["answer_accuracy"]),
                        "heldout_answer_accuracy": float(first_row["heldout_answer_accuracy"]),
                        "q": float(first_row["q"]),
                        "r": float(first_row["r"]),
                        "w": float(first_row["w"]),
                    },
                    "best_answer": {
                        "step": int(best_answer_row["step"]),
                        "answer_accuracy": float(best_answer_row["answer_accuracy"]),
                        "heldout_answer_accuracy": float(best_answer_row["heldout_answer_accuracy"]),
                    },
                    "best_heldout": {
                        "step": int(best_heldout_row["step"]),
                        "answer_accuracy": float(best_heldout_row["answer_accuracy"]),
                        "heldout_answer_accuracy": float(best_heldout_row["heldout_answer_accuracy"]),
                    },
                    "largest_answer_jump": {
                        "step": int(max_delta_answer_row["step"]),
                        "delta_answer_accuracy": float(max_delta_answer_row["delta_answer_accuracy"]),
                    },
                    "largest_heldout_jump": {
                        "step": int(max_delta_heldout_row["step"]),
                        "delta_heldout_answer_accuracy": float(max_delta_heldout_row["delta_heldout_answer_accuracy"]),
                    },
                    "last": {
                        "answer_accuracy": float(last_row["answer_accuracy"]),
                        "heldout_answer_accuracy": float(last_row["heldout_answer_accuracy"]),
                        "q": float(last_row["q"]),
                        "r": float(last_row["r"]),
                        "w": float(last_row["w"]),
                    },
                },
                "residual_stream": {
                    "top_answer_probe_stages": _stage_summary(selected_rows, "answer_probe_accuracy_by_stage")[:top_k],
                    "top_query_probe_stages": _stage_summary(selected_rows, "query_probe_accuracy_by_stage")[:top_k],
                    "top_support_probe_stages": _stage_summary(selected_rows, "support_probe_accuracy_by_stage")[:top_k],
                    "top_answer_margin_stages": _stage_summary(selected_rows, "answer_margin_by_stage")[:top_k],
                    "top_drift_stages": _drift_summary(selected_rows, top_k=top_k),
                },
                "heads": {
                    "top_by_ablation": _aggregate_slice(
                        selected_rows,
                        "top_heads_by_ablation",
                        key_fields=("layer", "head"),
                        metric_name="accuracy_drop",
                        top_k=top_k,
                    ),
                    "top_by_localization": _aggregate_slice(
                        selected_rows,
                        "top_heads_by_localization",
                        key_fields=("layer", "head"),
                        metric_name="support_attention_mean",
                        top_k=top_k,
                    ),
                },
                "mlp_blocks": {
                    "top_by_ablation": _aggregate_slice(
                        selected_rows,
                        "top_mlps_by_ablation",
                        key_fields=("layer",),
                        metric_name="accuracy_drop",
                        top_k=top_k,
                    ),
                    "top_by_write": _aggregate_slice(
                        selected_rows,
                        "top_mlps_by_write",
                        key_fields=("layer",),
                        metric_name="answer_margin_delta_mean",
                        top_k=top_k,
                    ),
                },
                "neurons": {
                    "candidate_layers": sorted(
                        {
                            int(layer_index)
                            for row in selected_rows
                            for layer_index in row["candidate_mlp_layers_for_neurons"]
                        }
                    ),
                    "top_by_write": _aggregate_slice(
                        selected_rows,
                        "top_neurons_by_write",
                        key_fields=("layer", "neuron"),
                        metric_name="write_strength",
                        top_k=top_k,
                    ),
                    "top_by_ablation": _aggregate_slice(
                        selected_rows,
                        "top_neurons_by_ablation",
                        key_fields=("layer", "neuron"),
                        metric_name="accuracy_drop",
                        top_k=top_k,
                    ),
                },
                "unimplemented_layers": ["causal_validation", "cross_run"],
                "partially_implemented_layers": ["features", "neurons", "geometry_dynamics"],
            }
        )

    payload = {
        "sweep_metrics_path": str(sweep_metrics_path),
        "sweep_summary_path": str(sweep_summary_path),
        "probe_set_path": str(summary["probe_set_path"]),
        "probe_set_metadata": probe_metadata,
        "layered_plan_status": build_layered_plan_status(),
        "birth_windows": window_reports,
    }
    write_json(output_path, payload)
    return output_path
