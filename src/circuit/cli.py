from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from circuit.analysis.formation import (
    collect_analysis_batches,
    compute_head_ablation_importance,
    compute_head_localization,
    summarize_formation_trace,
)
from circuit.analysis.analysis_report import build_analysis_report
from circuit.analysis.birth_windows import analyze_birth_windows
from circuit.analysis.birth_window_compare import compare_birth_window_checkpoints
from circuit.analysis.checkpoint_sweep import generate_probe_set, run_checkpoint_sweep
from circuit.analysis.candidate_dynamics import (
    build_candidate_birth_model,
    build_candidate_coalition_map,
    build_candidate_mechanism_report,
    build_candidate_neuron_intervention,
    build_candidate_circuit_registry,
    run_candidate_sweep,
    run_circuit_gradient_link,
)
from circuit.analysis.feature_analysis import analyze_checkpoint_features
from circuit.analysis.geometric_mechanisms import (
    build_dataset_geometry_report,
    run_attention_downstream_update_attribution,
    run_attention_retrieval_chain_report,
    run_attention_retrieval_separation_update_attribution,
    run_attention_score_delta_decomposition,
    run_attention_score_update_attribution,
    run_checkpoint_update_attribution,
    run_candidate_route_gradient_selection,
    run_causal_variable_patch,
    run_data_update_attribution,
    run_attention_geometry_trace,
    run_geometry_subspace_intervention,
    run_path_logit_decomposition,
    run_prompt_neuron_trace,
    run_route_competition_report,
    run_route_gradient_decomposition,
)
from circuit.analysis.actual_batch_route_attribution import run_actual_batch_route_attribution
from circuit.analysis.optimizer_update_trace import run_optimizer_update_trace
from circuit.analysis.shared_feature_dynamics import (
    family_update_link,
    feature_birth_analyze,
    feature_compare,
    feature_family_birth_analyze,
    feature_family_cluster,
    feature_family_compare,
    feature_family_patch,
    feature_family_lineage,
    feature_family_rank,
    feature_family_subpatch,
    feature_family_trace,
    feature_lineage,
    feature_patch,
    feature_trajectory_sweep,
    shared_feature_fit,
    subset_competition,
    subset_birth_analyze,
    subset_trajectory,
)
from circuit.config import TrainSpec
from circuit.data.symbolic_kv import generate_symbolic_kv_benchmark
from circuit.data.symbolic_kv_stream import generate_symbolic_kv_stream_benchmark
from circuit.eval import evaluate_split
from circuit.io import append_jsonl, write_json
from circuit.reference import select_reference_configuration
from circuit.train import load_model_from_checkpoint, make_data_loader, train_from_config


def _list_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints


def _evaluate_command(config_path: Path, checkpoint_path: Path, split_name: str | None) -> dict[str, Any]:
    context = load_model_from_checkpoint(config_path=config_path, checkpoint_path=checkpoint_path)
    spec = context["spec"]
    vocab = context["vocab"]
    device = context["device"]
    model = context["model"]

    split_names = [split_name] if split_name is not None else [
        "validation_iid",
        "test_iid",
        "heldout_pairs",
        "structural_ood",
        "counterfactual",
    ]
    results: dict[str, Any] = {}
    for current_split in split_names:
        loader = make_data_loader(
            benchmark_dir=spec.benchmark_dir,
            split_name=current_split,
            batch_size=spec.evaluation.batch_size,
            shuffle=False,
            num_workers=spec.num_workers,
            pad_token_id=vocab.pad_token_id,
        )
        results[current_split] = evaluate_split(
            model=model,
            data_loader=loader,
            device=device,
            pad_token_id=vocab.pad_token_id,
            value_token_ids=vocab.value_token_ids,
            max_batches=spec.evaluation.max_eval_batches,
            include_analysis=(current_split == "validation_iid"),
        )
    return results


def _analyze_checkpoint_command(config_path: Path, checkpoint_path: Path, output_path: Path | None) -> Path:
    context = load_model_from_checkpoint(config_path=config_path, checkpoint_path=checkpoint_path)
    spec = context["spec"]
    vocab = context["vocab"]
    device = context["device"]
    model = context["model"]
    loader = make_data_loader(
        benchmark_dir=spec.benchmark_dir,
        split_name="validation_iid",
        batch_size=spec.evaluation.batch_size,
        shuffle=False,
        num_workers=spec.num_workers,
        pad_token_id=vocab.pad_token_id,
    )
    batches = collect_analysis_batches(
        loader,
        device=device,
        max_batches=spec.evaluation.max_analysis_batches,
    )
    payload = {
        "checkpoint": str(checkpoint_path),
        "head_localization": compute_head_localization(model=model, batches=batches),
        "head_ablation": compute_head_ablation_importance(model=model, batches=batches),
    }
    resolved_output = output_path or checkpoint_path.with_name(f"{checkpoint_path.stem}_analysis.json")
    write_json(resolved_output, payload)
    return resolved_output


def _formation_trace_command(config_path: Path, checkpoint_dir: Path | None, output_path: Path | None) -> tuple[Path, Path]:
    spec = TrainSpec.from_path(config_path)
    effective_checkpoint_dir = checkpoint_dir or (spec.output_dir / "checkpoints")
    checkpoints = _list_checkpoints(effective_checkpoint_dir)
    trace_path = output_path or (spec.output_dir / "formation_trace.jsonl")
    summary_path = trace_path.with_name("formation_summary.json")
    if trace_path.exists():
        trace_path.unlink()

    rows: list[dict[str, Any]] = []
    for checkpoint_path in checkpoints:
        context = load_model_from_checkpoint(config_path=config_path, checkpoint_path=checkpoint_path)
        vocab = context["vocab"]
        device = context["device"]
        model = context["model"]
        validation_loader = make_data_loader(
            benchmark_dir=spec.benchmark_dir,
            split_name="validation_iid",
            batch_size=spec.evaluation.batch_size,
            shuffle=False,
            num_workers=spec.num_workers,
            pad_token_id=vocab.pad_token_id,
        )
        counterfactual_loader = make_data_loader(
            benchmark_dir=spec.benchmark_dir,
            split_name="counterfactual",
            batch_size=spec.evaluation.batch_size,
            shuffle=False,
            num_workers=spec.num_workers,
            pad_token_id=vocab.pad_token_id,
        )
        validation_metrics = evaluate_split(
            model=model,
            data_loader=validation_loader,
            device=device,
            pad_token_id=vocab.pad_token_id,
            value_token_ids=vocab.value_token_ids,
            max_batches=spec.evaluation.max_analysis_batches,
            include_analysis=True,
        )
        counterfactual_metrics = evaluate_split(
            model=model,
            data_loader=counterfactual_loader,
            device=device,
            pad_token_id=vocab.pad_token_id,
            value_token_ids=vocab.value_token_ids,
            max_batches=spec.evaluation.max_analysis_batches,
            include_analysis=False,
        )
        row = {
            "step": int(context["checkpoint"]["step"]),
            "answer_accuracy": validation_metrics["answer_accuracy"],
            "loss": validation_metrics["loss"],
            "q": validation_metrics["q"],
            "r": validation_metrics["r"],
            "w": validation_metrics["w"],
            "counterfactual_accuracy": counterfactual_metrics["answer_accuracy"],
        }
        append_jsonl(trace_path, row)
        rows.append(row)

    summary = summarize_formation_trace(rows=rows, thresholds=spec.evaluation.birth_thresholds)
    write_json(summary_path, summary)
    return trace_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(prog="circuit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate-benchmark")
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--overwrite", action="store_true")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.add_argument("--resume-checkpoint", type=Path, default=None)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--split", type=str, default=None)

    analyze_parser = subparsers.add_parser("analyze-checkpoint")
    analyze_parser.add_argument("--config", type=Path, required=True)
    analyze_parser.add_argument("--checkpoint", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, default=None)

    formation_parser = subparsers.add_parser("formation-trace")
    formation_parser.add_argument("--config", type=Path, required=True)
    formation_parser.add_argument("--checkpoint-dir", type=Path, default=None)
    formation_parser.add_argument("--output", type=Path, default=None)

    probe_parser = subparsers.add_parser("generate-probe-set")
    probe_parser.add_argument("--benchmark-dir", type=Path, required=True)
    probe_parser.add_argument("--output", type=Path, required=True)
    probe_parser.add_argument("--examples-per-split", type=int, default=24)
    probe_parser.add_argument("--seed", type=int, default=17)
    probe_parser.add_argument("--split", type=str, action="append", default=None)
    probe_parser.add_argument("--overwrite", action="store_true")

    sweep_parser = subparsers.add_parser("checkpoint-sweep")
    sweep_parser.add_argument("--config", type=Path, required=True)
    sweep_parser.add_argument("--probe-set", type=Path, required=True)
    sweep_parser.add_argument("--output", type=Path, required=True)
    sweep_parser.add_argument("--checkpoint-dir", type=Path, default=None)
    sweep_parser.add_argument("--device", type=str, default="cpu")
    sweep_parser.add_argument("--create-probe-set", action="store_true")
    sweep_parser.add_argument("--probe-examples-per-split", type=int, default=24)
    sweep_parser.add_argument("--probe-seed", type=int, default=17)
    sweep_parser.add_argument("--overwrite-probe-set", action="store_true")

    birth_window_parser = subparsers.add_parser("birth-window-analyze")
    birth_window_parser.add_argument("--sweep-metrics", type=Path, required=True)
    birth_window_parser.add_argument("--sweep-summary", type=Path, required=True)
    birth_window_parser.add_argument("--output", type=Path, required=True)
    birth_window_parser.add_argument("--top-k", type=int, default=6)

    compare_parser = subparsers.add_parser("birth-window-compare")
    compare_parser.add_argument("--config", type=Path, required=True)
    compare_parser.add_argument("--probe-set", type=Path, required=True)
    compare_parser.add_argument("--sweep-metrics", type=Path, required=True)
    compare_parser.add_argument("--target-step", type=int, required=True)
    compare_parser.add_argument("--source-step", type=int, action="append", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--checkpoint-dir", type=Path, default=None)
    compare_parser.add_argument("--device", type=str, default="cpu")
    compare_parser.add_argument("--top-k-components", type=int, default=6)
    compare_parser.add_argument("--max-neurons-per-layer", type=int, default=4)
    compare_parser.add_argument("--stage", type=str, action="append", default=None)

    feature_parser = subparsers.add_parser("feature-analyze")
    feature_parser.add_argument("--config", type=Path, required=True)
    feature_parser.add_argument("--checkpoint", type=Path, required=True)
    feature_parser.add_argument("--probe-set", type=Path, required=True)
    feature_parser.add_argument("--stage", type=str, required=True)
    feature_parser.add_argument("--output", type=Path, required=True)
    feature_parser.add_argument("--source-checkpoint", type=Path, default=None)
    feature_parser.add_argument("--device", type=str, default="cpu")
    feature_parser.add_argument("--num-features", type=int, default=64)
    feature_parser.add_argument("--train-steps", type=int, default=400)
    feature_parser.add_argument("--learning-rate", type=float, default=1e-3)
    feature_parser.add_argument("--l1-coefficient", type=float, default=1e-3)
    feature_parser.add_argument("--sae-batch-size", type=int, default=256)
    feature_parser.add_argument("--top-k-features", type=int, default=12)
    feature_parser.add_argument("--top-k-examples", type=int, default=6)

    shared_fit_parser = subparsers.add_parser("shared-feature-fit")
    shared_fit_parser.add_argument("--config", type=Path, required=True)
    shared_fit_parser.add_argument("--probe-set", type=Path, required=True)
    shared_fit_parser.add_argument("--stage", type=str, required=True)
    shared_fit_parser.add_argument("--output-dir", type=Path, required=True)
    shared_fit_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    shared_fit_parser.add_argument("--checkpoint-dir", type=Path, default=None)
    shared_fit_parser.add_argument("--device", type=str, default="cpu")
    shared_fit_parser.add_argument("--num-features", type=int, default=64)
    shared_fit_parser.add_argument("--train-steps", type=int, default=400)
    shared_fit_parser.add_argument("--learning-rate", type=float, default=1e-3)
    shared_fit_parser.add_argument("--l1-coefficient", type=float, default=1e-3)
    shared_fit_parser.add_argument("--batch-size", type=int, default=256)

    trajectory_parser = subparsers.add_parser("feature-trajectory-sweep")
    trajectory_parser.add_argument("--config", type=Path, required=True)
    trajectory_parser.add_argument("--probe-set", type=Path, required=True)
    trajectory_parser.add_argument("--basis", type=Path, required=True)
    trajectory_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    trajectory_parser.add_argument("--output-dir", type=Path, required=True)
    trajectory_parser.add_argument("--device", type=str, default="cpu")

    birth_parser = subparsers.add_parser("feature-birth-analyze")
    birth_parser.add_argument("--trajectories", type=Path, required=True)
    birth_parser.add_argument("--output-dir", type=Path, required=True)
    birth_parser.add_argument("--threshold-mean-activation", type=float, default=0.5)
    birth_parser.add_argument("--threshold-active-fraction", type=float, default=0.25)
    birth_parser.add_argument("--threshold-correctness-gap", type=float, default=0.05)
    birth_parser.add_argument("--threshold-heldout-gap", type=float, default=0.05)
    birth_parser.add_argument("--delta-threshold", type=float, default=0.02)
    birth_parser.add_argument("--window", type=int, default=2)

    subset_trajectory_parser = subparsers.add_parser("subset-trajectory")
    subset_trajectory_parser.add_argument("--trajectories", type=Path, required=True)
    subset_trajectory_parser.add_argument("--output", type=Path, required=True)
    subset_trajectory_parser.add_argument("--feature", type=int, action="append", default=None)
    subset_trajectory_parser.add_argument("--feature-family-rank", type=Path, default=None)
    subset_trajectory_parser.add_argument("--ranking-name", type=str, default=None)
    subset_trajectory_parser.add_argument("--subset-size", type=int, default=None)

    subset_birth_parser = subparsers.add_parser("subset-birth")
    subset_birth_parser.add_argument("--subset-trajectory", type=Path, required=True)
    subset_birth_parser.add_argument("--output", type=Path, required=True)
    subset_birth_parser.add_argument("--threshold-mean-activation", type=float, default=0.5)
    subset_birth_parser.add_argument("--threshold-active-fraction", type=float, default=0.25)
    subset_birth_parser.add_argument("--threshold-correctness-gap", type=float, default=0.05)
    subset_birth_parser.add_argument("--threshold-heldout-gap", type=float, default=0.05)
    subset_birth_parser.add_argument("--delta-threshold", type=float, default=0.02)
    subset_birth_parser.add_argument("--window", type=int, default=2)

    subset_competition_parser = subparsers.add_parser("subset-competition")
    subset_competition_parser.add_argument("--config", type=Path, required=True)
    subset_competition_parser.add_argument("--probe-set", type=Path, required=True)
    subset_competition_parser.add_argument("--basis", type=Path, required=True)
    subset_competition_parser.add_argument("--source-checkpoint", type=Path, required=True)
    subset_competition_parser.add_argument("--target-checkpoint", type=Path, required=True)
    subset_competition_parser.add_argument("--stage", type=str, required=True)
    subset_competition_parser.add_argument("--output", type=Path, required=True)
    subset_competition_parser.add_argument("--device", type=str, default="cpu")
    subset_competition_parser.add_argument("--patch-mode", type=str, default="replace")
    subset_competition_parser.add_argument("--subset-a-feature", type=int, action="append", default=None)
    subset_competition_parser.add_argument("--subset-a-feature-family-rank", type=Path, default=None)
    subset_competition_parser.add_argument("--subset-a-ranking-name", type=str, default=None)
    subset_competition_parser.add_argument("--subset-a-subset-size", type=int, default=None)
    subset_competition_parser.add_argument("--subset-b-feature", type=int, action="append", default=None)
    subset_competition_parser.add_argument("--subset-b-feature-family-rank", type=Path, default=None)
    subset_competition_parser.add_argument("--subset-b-ranking-name", type=str, default=None)
    subset_competition_parser.add_argument("--subset-b-subset-size", type=int, default=None)

    family_update_link_parser = subparsers.add_parser("family-update-link")
    family_update_link_parser.add_argument("--feature-family-trace", type=Path, required=True)
    family_update_link_parser.add_argument("--subset-trajectory", type=Path, required=True)
    family_update_link_parser.add_argument("--sweep-metrics", type=Path, required=True)
    family_update_link_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    family_update_link_parser.add_argument("--output", type=Path, required=True)

    candidate_registry_parser = subparsers.add_parser("candidate-circuit-registry")
    candidate_registry_parser.add_argument("--feature-family-trace", type=Path, action="append", required=True)
    candidate_registry_parser.add_argument("--subset-trajectory", type=Path, action="append", required=True)
    candidate_registry_parser.add_argument("--candidate-id", type=str, action="append", default=None)
    candidate_registry_parser.add_argument("--basis", type=Path, action="append", default=None)
    candidate_registry_parser.add_argument("--subset-birth", type=Path, action="append", default=None)
    candidate_registry_parser.add_argument("--family-update-link", type=Path, action="append", default=None)
    candidate_registry_parser.add_argument("--output", type=Path, required=True)

    gradient_link_parser = subparsers.add_parser("circuit-gradient-link")
    gradient_link_parser.add_argument("--config", type=Path, required=True)
    gradient_link_parser.add_argument("--probe-set", type=Path, required=True)
    gradient_link_parser.add_argument("--registry", type=Path, required=True)
    gradient_link_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    gradient_link_parser.add_argument("--output", type=Path, required=True)
    gradient_link_parser.add_argument("--device", type=str, default="cpu")
    gradient_link_parser.add_argument("--sweep-metrics", type=Path, default=None)
    gradient_link_parser.add_argument("--start-step", type=int, default=None)
    gradient_link_parser.add_argument("--end-step", type=int, default=None)

    candidate_sweep_parser = subparsers.add_parser("candidate-sweep")
    candidate_sweep_parser.add_argument("--config", type=Path, required=True)
    candidate_sweep_parser.add_argument("--probe-set", type=Path, required=True)
    candidate_sweep_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    candidate_sweep_parser.add_argument("--output-dir", type=Path, required=True)
    candidate_sweep_parser.add_argument("--stage", type=str, action="append", required=True)
    candidate_sweep_parser.add_argument("--families", type=Path, action="append", required=True)
    candidate_sweep_parser.add_argument("--feature-compare", type=Path, action="append", required=True)
    candidate_sweep_parser.add_argument("--trajectories", type=Path, action="append", required=True)
    candidate_sweep_parser.add_argument("--basis", type=Path, action="append", required=True)
    candidate_sweep_parser.add_argument("--sweep-metrics", type=Path, default=None)
    candidate_sweep_parser.add_argument("--device", type=str, default="cpu")
    candidate_sweep_parser.add_argument("--ranking-name", type=str, default="by_useful_delta")
    candidate_sweep_parser.add_argument("--subset-size", type=int, default=3)
    candidate_sweep_parser.add_argument("--min-family-size", type=int, default=2)
    candidate_sweep_parser.add_argument("--top-k-families", type=int, default=None)
    candidate_sweep_parser.add_argument("--start-step", type=int, default=None)
    candidate_sweep_parser.add_argument("--end-step", type=int, default=None)

    candidate_mechanism_parser = subparsers.add_parser("candidate-mechanism-report")
    candidate_mechanism_parser.add_argument("--registry", type=Path, required=True)
    candidate_mechanism_parser.add_argument("--gradient-link", type=Path, required=True)
    candidate_mechanism_parser.add_argument("--output-dir", type=Path, required=True)
    candidate_mechanism_parser.add_argument("--candidate-id", type=str, action="append", default=None)
    candidate_mechanism_parser.add_argument("--top-k", type=int, default=4)
    candidate_mechanism_parser.add_argument("--ranking-metric", type=str, default="sum_useful_delta")
    candidate_mechanism_parser.add_argument("--phase-epsilon", type=float, default=0.0)
    candidate_mechanism_parser.add_argument("--top-interval-k", type=int, default=5)

    candidate_birth_parser = subparsers.add_parser("candidate-birth-model")
    candidate_birth_parser.add_argument("--registry", type=Path, required=True)
    candidate_birth_parser.add_argument("--gradient-link", type=Path, required=True)
    candidate_birth_parser.add_argument("--output-dir", type=Path, required=True)
    candidate_birth_parser.add_argument("--candidate-id", type=str, action="append", default=None)
    candidate_birth_parser.add_argument("--birth-metric", type=str, choices=["birth_step", "useful_birth_step"], default="useful_birth_step")
    candidate_birth_parser.add_argument("--prediction-cutoff-step", type=int, default=None)
    candidate_birth_parser.add_argument("--lookback-intervals", type=int, default=None)
    candidate_birth_parser.add_argument("--birth-score-threshold", type=float, default=0.0)

    candidate_coalition_parser = subparsers.add_parser("candidate-coalition-map")
    candidate_coalition_parser.add_argument("--config", type=Path, required=True)
    candidate_coalition_parser.add_argument("--probe-set", type=Path, required=True)
    candidate_coalition_parser.add_argument("--registry", type=Path, required=True)
    candidate_coalition_parser.add_argument("--gradient-link", type=Path, required=True)
    candidate_coalition_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    candidate_coalition_parser.add_argument("--output-dir", type=Path, required=True)
    candidate_coalition_parser.add_argument("--candidate-id", type=str, action="append", default=None)
    candidate_coalition_parser.add_argument("--device", type=str, default="cpu")
    candidate_coalition_parser.add_argument("--start-step", type=int, default=None)
    candidate_coalition_parser.add_argument("--end-step", type=int, default=None)
    candidate_coalition_parser.add_argument("--neuron-layer", type=int, action="append", default=None)
    candidate_coalition_parser.add_argument("--candidate-only", action="store_true")
    candidate_coalition_parser.add_argument("--top-k-neurons", type=int, default=24)
    candidate_coalition_parser.add_argument("--trajectory-top-k", type=int, default=8)
    candidate_coalition_parser.add_argument("--sign-epsilon", type=float, default=0.0)

    candidate_neuron_intervention_parser = subparsers.add_parser("candidate-neuron-intervention")
    candidate_neuron_intervention_parser.add_argument("--config", type=Path, required=True)
    candidate_neuron_intervention_parser.add_argument("--probe-set", type=Path, required=True)
    candidate_neuron_intervention_parser.add_argument("--coalition-map", type=Path, required=True)
    candidate_neuron_intervention_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    candidate_neuron_intervention_parser.add_argument("--output-dir", type=Path, required=True)
    candidate_neuron_intervention_parser.add_argument("--checkpoint-step", type=int, required=True)
    candidate_neuron_intervention_parser.add_argument("--device", type=str, default="cpu")
    candidate_neuron_intervention_parser.add_argument("--top-k-per-set", type=int, default=8)
    candidate_neuron_intervention_parser.add_argument("--single-neuron-top-k", type=int, default=0)
    candidate_neuron_intervention_parser.add_argument("--score-individual-features", action="store_true")

    feature_compare_parser = subparsers.add_parser("feature-compare")
    feature_compare_parser.add_argument("--trajectories", type=Path, required=True)
    feature_compare_parser.add_argument("--source-step", type=int, required=True)
    feature_compare_parser.add_argument("--target-step", type=int, required=True)
    feature_compare_parser.add_argument("--output", type=Path, required=True)
    feature_compare_parser.add_argument("--top-k", type=int, default=12)

    feature_family_parser = subparsers.add_parser("feature-family-cluster")
    feature_family_parser.add_argument("--trajectories", type=Path, required=True)
    feature_family_parser.add_argument("--output-dir", type=Path, required=True)
    feature_family_parser.add_argument("--metric", type=str, action="append", default=None)
    feature_family_parser.add_argument("--similarity-threshold", type=float, default=0.85)
    feature_family_parser.add_argument("--feature-births", type=Path, default=None)
    feature_family_parser.add_argument("--top-k-families", type=int, default=8)

    feature_family_birth_parser = subparsers.add_parser("feature-family-birth")
    feature_family_birth_parser.add_argument("--family-trajectories", type=Path, required=True)
    feature_family_birth_parser.add_argument("--families", type=Path, required=True)
    feature_family_birth_parser.add_argument("--output-dir", type=Path, required=True)
    feature_family_birth_parser.add_argument("--threshold-mean-activation", type=float, default=0.5)
    feature_family_birth_parser.add_argument("--threshold-active-fraction", type=float, default=0.25)
    feature_family_birth_parser.add_argument("--threshold-correctness-gap", type=float, default=0.05)
    feature_family_birth_parser.add_argument("--threshold-heldout-gap", type=float, default=0.05)
    feature_family_birth_parser.add_argument("--delta-threshold", type=float, default=0.02)
    feature_family_birth_parser.add_argument("--window", type=int, default=2)

    feature_patch_parser = subparsers.add_parser("feature-patch")
    feature_patch_parser.add_argument("--config", type=Path, required=True)
    feature_patch_parser.add_argument("--probe-set", type=Path, required=True)
    feature_patch_parser.add_argument("--basis", type=Path, required=True)
    feature_patch_parser.add_argument("--source-checkpoint", type=Path, required=True)
    feature_patch_parser.add_argument("--target-checkpoint", type=Path, required=True)
    feature_patch_parser.add_argument("--stage", type=str, required=True)
    feature_patch_parser.add_argument("--feature", type=int, action="append", required=True)
    feature_patch_parser.add_argument("--output", type=Path, required=True)
    feature_patch_parser.add_argument("--device", type=str, default="cpu")
    feature_patch_parser.add_argument("--patch-mode", type=str, default="replace")

    feature_family_compare_parser = subparsers.add_parser("feature-family-compare")
    feature_family_compare_parser.add_argument("--family-trajectories", type=Path, required=True)
    feature_family_compare_parser.add_argument("--families", type=Path, required=True)
    feature_family_compare_parser.add_argument("--source-step", type=int, required=True)
    feature_family_compare_parser.add_argument("--target-step", type=int, required=True)
    feature_family_compare_parser.add_argument("--output", type=Path, required=True)
    feature_family_compare_parser.add_argument("--top-k", type=int, default=12)

    feature_family_rank_parser = subparsers.add_parser("feature-family-rank")
    feature_family_rank_parser.add_argument("--families", type=Path, required=True)
    feature_family_rank_parser.add_argument("--feature-compare", type=Path, required=True)
    feature_family_rank_parser.add_argument("--family", type=int, required=True)
    feature_family_rank_parser.add_argument("--output", type=Path, required=True)

    feature_family_subpatch_parser = subparsers.add_parser("feature-family-subpatch")
    feature_family_subpatch_parser.add_argument("--config", type=Path, required=True)
    feature_family_subpatch_parser.add_argument("--probe-set", type=Path, required=True)
    feature_family_subpatch_parser.add_argument("--basis", type=Path, required=True)
    feature_family_subpatch_parser.add_argument("--feature-family-rank", type=Path, required=True)
    feature_family_subpatch_parser.add_argument("--source-checkpoint", type=Path, required=True)
    feature_family_subpatch_parser.add_argument("--target-checkpoint", type=Path, required=True)
    feature_family_subpatch_parser.add_argument("--stage", type=str, required=True)
    feature_family_subpatch_parser.add_argument("--ranking-name", type=str, required=True)
    feature_family_subpatch_parser.add_argument("--subset-size", type=int, action="append", required=True)
    feature_family_subpatch_parser.add_argument("--output", type=Path, required=True)
    feature_family_subpatch_parser.add_argument("--device", type=str, default="cpu")
    feature_family_subpatch_parser.add_argument("--patch-mode", type=str, default="replace")

    feature_family_lineage_parser = subparsers.add_parser("feature-family-lineage")
    feature_family_lineage_parser.add_argument("--config", type=Path, required=True)
    feature_family_lineage_parser.add_argument("--probe-set", type=Path, required=True)
    feature_family_lineage_parser.add_argument("--basis", type=Path, required=True)
    feature_family_lineage_parser.add_argument("--feature-family-rank", type=Path, required=True)
    feature_family_lineage_parser.add_argument("--checkpoint", type=Path, required=True)
    feature_family_lineage_parser.add_argument("--ranking-name", type=str, required=True)
    feature_family_lineage_parser.add_argument("--subset-size", type=int, required=True)
    feature_family_lineage_parser.add_argument("--output", type=Path, required=True)
    feature_family_lineage_parser.add_argument("--device", type=str, default="cpu")
    feature_family_lineage_parser.add_argument("--sweep-metrics", type=Path, required=True)

    feature_family_trace_parser = subparsers.add_parser("feature-family-trace")
    feature_family_trace_parser.add_argument("--feature-family-births", type=Path, required=True)
    feature_family_trace_parser.add_argument("--feature-family-rank", type=Path, required=True)
    feature_family_trace_parser.add_argument("--feature-family-subpatch", type=Path, required=True)
    feature_family_trace_parser.add_argument("--feature-family-lineage", type=Path, required=True)
    feature_family_trace_parser.add_argument("--output", type=Path, required=True)

    feature_family_patch_parser = subparsers.add_parser("feature-family-patch")
    feature_family_patch_parser.add_argument("--config", type=Path, required=True)
    feature_family_patch_parser.add_argument("--probe-set", type=Path, required=True)
    feature_family_patch_parser.add_argument("--basis", type=Path, required=True)
    feature_family_patch_parser.add_argument("--families", type=Path, required=True)
    feature_family_patch_parser.add_argument("--source-checkpoint", type=Path, required=True)
    feature_family_patch_parser.add_argument("--target-checkpoint", type=Path, required=True)
    feature_family_patch_parser.add_argument("--stage", type=str, required=True)
    feature_family_patch_parser.add_argument("--family", type=int, action="append", required=True)
    feature_family_patch_parser.add_argument("--output", type=Path, required=True)
    feature_family_patch_parser.add_argument("--device", type=str, default="cpu")
    feature_family_patch_parser.add_argument("--patch-mode", type=str, default="replace")

    feature_lineage_parser = subparsers.add_parser("feature-lineage")
    feature_lineage_parser.add_argument("--config", type=Path, required=True)
    feature_lineage_parser.add_argument("--probe-set", type=Path, required=True)
    feature_lineage_parser.add_argument("--basis", type=Path, required=True)
    feature_lineage_parser.add_argument("--checkpoint", type=Path, required=True)
    feature_lineage_parser.add_argument("--feature", type=int, action="append", required=True)
    feature_lineage_parser.add_argument("--output", type=Path, required=True)
    feature_lineage_parser.add_argument("--device", type=str, default="cpu")
    feature_lineage_parser.add_argument("--sweep-metrics", type=Path, default=None)

    report_parser = subparsers.add_parser("analysis-report")
    report_parser.add_argument("--analysis-dir", type=Path, required=True)
    report_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser.add_argument("--overwrite", action="store_true")

    reference_parser = subparsers.add_parser("select-reference")
    reference_parser.add_argument("--run-dir", type=Path, action="append", required=True)
    reference_parser.add_argument("--device", type=str, default="cpu")
    reference_parser.add_argument("--min-validation-answer-accuracy", type=float, default=0.9)
    reference_parser.add_argument("--output", type=Path, default=None)

    dataset_geometry_parser = subparsers.add_parser("dataset-geometry-report")
    dataset_geometry_parser.add_argument("--benchmark-dir", type=Path, required=True)
    dataset_geometry_parser.add_argument("--output-dir", type=Path, required=True)
    dataset_geometry_parser.add_argument("--top-k-pairs", type=int, default=20)

    attention_geometry_parser = subparsers.add_parser("attention-geometry-trace")
    attention_geometry_parser.add_argument("--config", type=Path, required=True)
    attention_geometry_parser.add_argument("--probe-set", type=Path, required=True)
    attention_geometry_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    attention_geometry_parser.add_argument("--output-dir", type=Path, required=True)
    attention_geometry_parser.add_argument("--device", type=str, default="mps")
    attention_geometry_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    attention_geometry_parser.add_argument("--top-k-tokens", type=int, default=8)
    attention_geometry_parser.add_argument("--top-k-plot-heads", type=int, default=6)

    path_logit_parser = subparsers.add_parser("path-logit-decomposition")
    path_logit_parser.add_argument("--config", type=Path, required=True)
    path_logit_parser.add_argument("--probe-set", type=Path, required=True)
    path_logit_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    path_logit_parser.add_argument("--output-dir", type=Path, required=True)
    path_logit_parser.add_argument("--device", type=str, default="mps")
    path_logit_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    path_logit_parser.add_argument("--ablation-top-k", type=int, default=3)
    path_logit_parser.add_argument("--ablation-step", type=int, action="append", default=None)
    path_logit_parser.add_argument("--top-k-plot-components", type=int, default=8)

    prompt_neuron_parser = subparsers.add_parser("prompt-neuron-trace")
    prompt_neuron_parser.add_argument("--config", type=Path, required=True)
    prompt_neuron_parser.add_argument("--probe-set", type=Path, required=True)
    prompt_neuron_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    prompt_neuron_parser.add_argument("--output-dir", type=Path, required=True)
    prompt_neuron_parser.add_argument("--device", type=str, default="mps")
    prompt_neuron_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    prompt_neuron_parser.add_argument("--mlp-layer", type=int, action="append", default=None)
    prompt_neuron_parser.add_argument("--activation-threshold", type=float, default=0.0)
    prompt_neuron_parser.add_argument("--top-k-per-query", type=int, default=8)
    prompt_neuron_parser.add_argument("--ablation-top-k-per-layer", type=int, default=4)
    prompt_neuron_parser.add_argument("--ablation-step", type=int, action="append", default=None)
    prompt_neuron_parser.add_argument("--ablation-neuron", type=str, action="append", default=None)
    prompt_neuron_parser.add_argument("--top-k-plot-neurons", type=int, default=12)

    geometry_subspace_parser = subparsers.add_parser("geometry-subspace-intervention")
    geometry_subspace_parser.add_argument("--config", type=Path, required=True)
    geometry_subspace_parser.add_argument("--probe-set", type=Path, required=True)
    geometry_subspace_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    geometry_subspace_parser.add_argument("--output-dir", type=Path, required=True)
    geometry_subspace_parser.add_argument("--device", type=str, default="mps")
    geometry_subspace_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    geometry_subspace_parser.add_argument("--stage", type=str, required=True)
    geometry_subspace_parser.add_argument("--subspace", type=str, required=True)
    geometry_subspace_parser.add_argument("--rank", type=int, required=True)
    geometry_subspace_parser.add_argument("--operation", type=str, required=True)
    geometry_subspace_parser.add_argument("--position-role", type=str, required=True)
    geometry_subspace_parser.add_argument("--query-mode", type=str, required=True)
    geometry_subspace_parser.add_argument("--head-layer", type=int, default=None)
    geometry_subspace_parser.add_argument("--head", type=int, default=None)
    geometry_subspace_parser.add_argument("--progress-every-queries", type=int, default=100)

    causal_patch_parser = subparsers.add_parser("causal-variable-patch")
    causal_patch_parser.add_argument("--config", type=Path, required=True)
    causal_patch_parser.add_argument("--probe-set", type=Path, required=True)
    causal_patch_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    causal_patch_parser.add_argument("--output-dir", type=Path, required=True)
    causal_patch_parser.add_argument("--device", type=str, default="mps")
    causal_patch_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    causal_patch_parser.add_argument("--stage", type=str, required=True)
    causal_patch_parser.add_argument("--subspace", type=str, required=True)
    causal_patch_parser.add_argument("--rank", type=int, default=None)
    causal_patch_parser.add_argument("--position-role", type=str, required=True)
    causal_patch_parser.add_argument("--pair-type", type=str, action="append", required=True)
    causal_patch_parser.add_argument("--head-layer", type=int, default=None)
    causal_patch_parser.add_argument("--head", type=int, default=None)
    causal_patch_parser.add_argument("--max-pairs-per-type", type=int, default=128)
    causal_patch_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    causal_patch_parser.add_argument("--split", type=str, action="append", default=None)
    causal_patch_parser.add_argument("--min-recovery-denominator", type=float, default=1.0e-6)
    causal_patch_parser.add_argument("--progress-every-pairs", type=int, default=64)

    route_gradient_parser = subparsers.add_parser("candidate-route-gradient-selection")
    route_gradient_parser.add_argument("--config", type=Path, required=True)
    route_gradient_parser.add_argument("--probe-set", type=Path, required=True)
    route_gradient_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    route_gradient_parser.add_argument("--output-dir", type=Path, required=True)
    route_gradient_parser.add_argument("--device", type=str, default="mps")
    route_gradient_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    route_gradient_parser.add_argument("--stage", type=str, required=True)
    route_gradient_parser.add_argument("--subspace", type=str, required=True)
    route_gradient_parser.add_argument("--rank", type=int, default=None)
    route_gradient_parser.add_argument("--position-role", type=str, required=True)
    route_gradient_parser.add_argument("--pair-type", type=str, action="append", required=True)
    route_gradient_parser.add_argument("--head-layer", type=int, default=None)
    route_gradient_parser.add_argument("--head", type=int, default=None)
    route_gradient_parser.add_argument("--max-pairs-per-type", type=int, default=64)
    route_gradient_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    route_gradient_parser.add_argument("--split", type=str, action="append", default=None)
    route_gradient_parser.add_argument("--loss-side", type=str, default="both")

    route_decomposition_parser = subparsers.add_parser("route-gradient-decomposition")
    route_decomposition_parser.add_argument("--config", type=Path, required=True)
    route_decomposition_parser.add_argument("--probe-set", type=Path, required=True)
    route_decomposition_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    route_decomposition_parser.add_argument("--output-dir", type=Path, required=True)
    route_decomposition_parser.add_argument("--device", type=str, default="mps")
    route_decomposition_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    route_decomposition_parser.add_argument("--stage", type=str, required=True)
    route_decomposition_parser.add_argument("--subspace", type=str, required=True)
    route_decomposition_parser.add_argument("--rank", type=int, default=None)
    route_decomposition_parser.add_argument("--position-role", type=str, required=True)
    route_decomposition_parser.add_argument("--pair-type", type=str, action="append", required=True)
    route_decomposition_parser.add_argument("--head-layer", type=int, default=None)
    route_decomposition_parser.add_argument("--head", type=int, default=None)
    route_decomposition_parser.add_argument("--max-pairs-per-type", type=int, default=64)
    route_decomposition_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    route_decomposition_parser.add_argument("--split", type=str, action="append", default=None)
    route_decomposition_parser.add_argument("--loss-side", type=str, default="both")
    route_decomposition_parser.add_argument("--decompose", type=str, action="append", default=None)
    route_decomposition_parser.add_argument("--top-k-groups", type=int, default=24)

    checkpoint_update_parser = subparsers.add_parser("checkpoint-update-attribution")
    checkpoint_update_parser.add_argument("--config", type=Path, required=True)
    checkpoint_update_parser.add_argument("--probe-set", type=Path, required=True)
    checkpoint_update_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    checkpoint_update_parser.add_argument("--output-dir", type=Path, required=True)
    checkpoint_update_parser.add_argument("--device", type=str, default="mps")
    checkpoint_update_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    checkpoint_update_parser.add_argument("--stage", type=str, required=True)
    checkpoint_update_parser.add_argument("--subspace", type=str, required=True)
    checkpoint_update_parser.add_argument("--rank", type=int, default=None)
    checkpoint_update_parser.add_argument("--position-role", type=str, required=True)
    checkpoint_update_parser.add_argument("--pair-type", type=str, action="append", required=True)
    checkpoint_update_parser.add_argument("--head-layer", type=int, default=None)
    checkpoint_update_parser.add_argument("--head", type=int, default=None)
    checkpoint_update_parser.add_argument("--max-pairs-per-type", type=int, default=64)
    checkpoint_update_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    checkpoint_update_parser.add_argument("--split", type=str, action="append", default=None)
    checkpoint_update_parser.add_argument("--decompose", type=str, action="append", default=None)
    checkpoint_update_parser.add_argument("--top-k-groups", type=int, default=24)
    checkpoint_update_parser.add_argument("--min-error-denominator", type=float, default=1.0e-9)

    data_update_parser = subparsers.add_parser("data-update-attribution")
    data_update_parser.add_argument("--config", type=Path, required=True)
    data_update_parser.add_argument("--probe-set", type=Path, required=True)
    data_update_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    data_update_parser.add_argument("--output-dir", type=Path, required=True)
    data_update_parser.add_argument("--device", type=str, default="mps")
    data_update_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    data_update_parser.add_argument("--stage", type=str, required=True)
    data_update_parser.add_argument("--subspace", type=str, required=True)
    data_update_parser.add_argument("--rank", type=int, default=None)
    data_update_parser.add_argument("--position-role", type=str, required=True)
    data_update_parser.add_argument("--pair-type", type=str, action="append", required=True)
    data_update_parser.add_argument("--route-pair-type", type=str, required=True)
    data_update_parser.add_argument("--route-split", type=str, default="__all__")
    data_update_parser.add_argument("--data-group-field", type=str, action="append", required=True)
    data_update_parser.add_argument("--head-layer", type=int, default=None)
    data_update_parser.add_argument("--head", type=int, default=None)
    data_update_parser.add_argument("--max-pairs-per-type", type=int, default=64)
    data_update_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    data_update_parser.add_argument("--split", type=str, action="append", default=None)
    data_update_parser.add_argument("--loss-side", type=str, default="both")
    data_update_parser.add_argument("--top-k-data-groups", type=int, default=24)
    data_update_parser.add_argument("--min-error-denominator", type=float, default=1.0e-9)

    route_competition_parser = subparsers.add_parser("route-competition-report")
    route_competition_parser.add_argument("--config", type=Path, required=True)
    route_competition_parser.add_argument("--probe-set", type=Path, required=True)
    route_competition_parser.add_argument("--train-probe-set", type=Path, required=True)
    route_competition_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    route_competition_parser.add_argument("--output-dir", type=Path, required=True)
    route_competition_parser.add_argument("--device", type=str, default="mps")
    route_competition_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    route_competition_parser.add_argument("--route", type=str, action="append", required=True)
    route_competition_parser.add_argument("--route-pair-type", type=str, required=True)
    route_competition_parser.add_argument("--pair-type", type=str, action="append", required=True)
    route_competition_parser.add_argument("--train-pair-type", type=str, action="append", required=True)
    route_competition_parser.add_argument("--data-group-field", type=str, action="append", required=True)
    route_competition_parser.add_argument("--eval-split", type=str, action="append", default=None)
    route_competition_parser.add_argument("--train-split", type=str, action="append", default=None)
    route_competition_parser.add_argument("--eval-loss-side", type=str, default="both")
    route_competition_parser.add_argument("--train-loss-side", type=str, default="clean")
    route_competition_parser.add_argument("--max-pairs-per-type", type=int, default=64)
    route_competition_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    route_competition_parser.add_argument("--min-error-denominator", type=float, default=1.0e-9)

    attention_score_delta_parser = subparsers.add_parser("attention-score-delta-decomposition")
    attention_score_delta_parser.add_argument("--config", type=Path, required=True)
    attention_score_delta_parser.add_argument("--probe-set", type=Path, required=True)
    attention_score_delta_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    attention_score_delta_parser.add_argument("--output-dir", type=Path, required=True)
    attention_score_delta_parser.add_argument("--device", type=str, default="mps")
    attention_score_delta_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    attention_score_delta_parser.add_argument("--head-layer", type=int, required=True)
    attention_score_delta_parser.add_argument("--head", type=int, required=True)
    attention_score_delta_parser.add_argument("--score-query-role", type=str, required=True)
    attention_score_delta_parser.add_argument("--score-key-role", type=str, action="append", required=True)
    attention_score_delta_parser.add_argument("--record-side", type=str, action="append", default=None)
    attention_score_delta_parser.add_argument("--pair-type", type=str, action="append", required=True)
    attention_score_delta_parser.add_argument("--max-pairs-per-type", type=int, default=64)
    attention_score_delta_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    attention_score_delta_parser.add_argument("--split", type=str, action="append", default=None)
    attention_score_delta_parser.add_argument("--reconstruction-tolerance", type=float, default=1.0e-3)
    attention_score_delta_parser.add_argument("--top-k-components", type=int, default=16)

    attention_score_update_parser = subparsers.add_parser("attention-score-update-attribution")
    attention_score_update_parser.add_argument("--config", type=Path, required=True)
    attention_score_update_parser.add_argument("--probe-set", type=Path, required=True)
    attention_score_update_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    attention_score_update_parser.add_argument("--output-dir", type=Path, required=True)
    attention_score_update_parser.add_argument("--device", type=str, default="mps")
    attention_score_update_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    attention_score_update_parser.add_argument("--head-layer", type=int, required=True)
    attention_score_update_parser.add_argument("--head", type=int, required=True)
    attention_score_update_parser.add_argument("--score-query-role", type=str, required=True)
    attention_score_update_parser.add_argument("--score-key-role", type=str, action="append", required=True)
    attention_score_update_parser.add_argument("--record-side", type=str, action="append", default=None)
    attention_score_update_parser.add_argument("--score-component", type=str, action="append", default=None)
    attention_score_update_parser.add_argument("--pair-type", type=str, action="append", required=True)
    attention_score_update_parser.add_argument("--max-pairs-per-type", type=int, default=64)
    attention_score_update_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    attention_score_update_parser.add_argument("--split", type=str, action="append", default=None)
    attention_score_update_parser.add_argument("--decompose", type=str, action="append", default=None)
    attention_score_update_parser.add_argument("--reconstruction-tolerance", type=float, default=1.0e-3)
    attention_score_update_parser.add_argument("--top-k-groups", type=int, default=24)
    attention_score_update_parser.add_argument("--min-error-denominator", type=float, default=1.0e-9)

    attention_retrieval_update_parser = subparsers.add_parser("attention-retrieval-separation-update-attribution")
    attention_retrieval_update_parser.add_argument("--config", type=Path, required=True)
    attention_retrieval_update_parser.add_argument("--probe-set", type=Path, required=True)
    attention_retrieval_update_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    attention_retrieval_update_parser.add_argument("--output-dir", type=Path, required=True)
    attention_retrieval_update_parser.add_argument("--device", type=str, default="mps")
    attention_retrieval_update_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    attention_retrieval_update_parser.add_argument("--head-layer", type=int, required=True)
    attention_retrieval_update_parser.add_argument("--head", type=int, required=True)
    attention_retrieval_update_parser.add_argument("--score-query-role", type=str, required=True)
    attention_retrieval_update_parser.add_argument("--support-key-role", type=str, required=True)
    attention_retrieval_update_parser.add_argument("--distractor-key-role", type=str, required=True)
    attention_retrieval_update_parser.add_argument("--record-side", type=str, action="append", default=None)
    attention_retrieval_update_parser.add_argument("--score-component", type=str, action="append", default=None)
    attention_retrieval_update_parser.add_argument("--pair-type", type=str, action="append", required=True)
    attention_retrieval_update_parser.add_argument("--max-pairs-per-type", type=int, default=64)
    attention_retrieval_update_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    attention_retrieval_update_parser.add_argument("--split", type=str, action="append", default=None)
    attention_retrieval_update_parser.add_argument("--decompose", type=str, action="append", default=None)
    attention_retrieval_update_parser.add_argument("--reconstruction-tolerance", type=float, default=1.0e-3)
    attention_retrieval_update_parser.add_argument("--top-k-groups", type=int, default=24)
    attention_retrieval_update_parser.add_argument("--min-error-denominator", type=float, default=1.0e-9)

    attention_retrieval_chain_parser = subparsers.add_parser("attention-retrieval-chain-report")
    attention_retrieval_chain_parser.add_argument("--config", type=Path, required=True)
    attention_retrieval_chain_parser.add_argument("--probe-set", type=Path, required=True)
    attention_retrieval_chain_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    attention_retrieval_chain_parser.add_argument("--output-dir", type=Path, required=True)
    attention_retrieval_chain_parser.add_argument("--device", type=str, default="mps")
    attention_retrieval_chain_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    attention_retrieval_chain_parser.add_argument("--head-layer", type=int, required=True)
    attention_retrieval_chain_parser.add_argument("--head", type=int, required=True)
    attention_retrieval_chain_parser.add_argument("--score-query-role", type=str, required=True)
    attention_retrieval_chain_parser.add_argument("--support-key-role", type=str, required=True)
    attention_retrieval_chain_parser.add_argument("--distractor-key-role", type=str, required=True)
    attention_retrieval_chain_parser.add_argument("--record-side", type=str, action="append", default=None)
    attention_retrieval_chain_parser.add_argument("--pair-type", type=str, action="append", required=True)
    attention_retrieval_chain_parser.add_argument("--max-pairs-per-type", type=int, default=64)
    attention_retrieval_chain_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    attention_retrieval_chain_parser.add_argument("--split", type=str, action="append", default=None)

    attention_downstream_update_parser = subparsers.add_parser("attention-downstream-update-attribution")
    attention_downstream_update_parser.add_argument("--config", type=Path, required=True)
    attention_downstream_update_parser.add_argument("--probe-set", type=Path, required=True)
    attention_downstream_update_parser.add_argument("--checkpoint-dir", type=Path, required=True)
    attention_downstream_update_parser.add_argument("--output-dir", type=Path, required=True)
    attention_downstream_update_parser.add_argument("--device", type=str, default="mps")
    attention_downstream_update_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    attention_downstream_update_parser.add_argument("--head-layer", type=int, required=True)
    attention_downstream_update_parser.add_argument("--head", type=int, required=True)
    attention_downstream_update_parser.add_argument("--score-query-role", type=str, required=True)
    attention_downstream_update_parser.add_argument("--support-key-role", type=str, required=True)
    attention_downstream_update_parser.add_argument("--distractor-key-role", type=str, required=True)
    attention_downstream_update_parser.add_argument("--record-side", type=str, action="append", default=None)
    attention_downstream_update_parser.add_argument("--scalar", type=str, action="append", default=None)
    attention_downstream_update_parser.add_argument("--pair-type", type=str, action="append", required=True)
    attention_downstream_update_parser.add_argument("--max-pairs-per-type", type=int, default=64)
    attention_downstream_update_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    attention_downstream_update_parser.add_argument("--split", type=str, action="append", default=None)
    attention_downstream_update_parser.add_argument("--decompose", type=str, action="append", default=None)
    attention_downstream_update_parser.add_argument("--top-k-groups", type=int, default=24)
    attention_downstream_update_parser.add_argument("--min-error-denominator", type=float, default=1.0e-9)

    optimizer_trace_parser = subparsers.add_parser("optimizer-update-trace")
    optimizer_trace_parser.add_argument("--config", type=Path, required=True)
    optimizer_trace_parser.add_argument("--resume-checkpoint", type=Path, required=True)
    optimizer_trace_parser.add_argument("--output-dir", type=Path, required=True)
    optimizer_trace_parser.add_argument("--device", type=str, default=None)
    optimizer_trace_end = optimizer_trace_parser.add_mutually_exclusive_group(required=True)
    optimizer_trace_end.add_argument("--end-step", type=int)
    optimizer_trace_end.add_argument("--num-steps", type=int)
    optimizer_trace_parser.add_argument("--train-split", type=str, default="train")
    optimizer_trace_parser.add_argument("--checkpoint-every", type=int, required=True)
    optimizer_trace_parser.add_argument("--progress-every", type=int, default=10)
    optimizer_trace_parser.add_argument("--top-k-parameters", type=int, default=24)
    optimizer_trace_parser.add_argument("--overwrite", action="store_true")
    optimizer_trace_parser.add_argument("--require-historical-replay", action="store_true")

    actual_batch_route_parser = subparsers.add_parser("actual-batch-route-attribution")
    actual_batch_route_parser.add_argument("--config", type=Path, required=True)
    actual_batch_route_parser.add_argument("--probe-set", type=Path, required=True)
    actual_batch_route_parser.add_argument("--optimizer-trace-dir", type=Path, required=True)
    actual_batch_route_parser.add_argument("--output-dir", type=Path, required=True)
    actual_batch_route_parser.add_argument("--device", type=str, default="mps")
    actual_batch_route_parser.add_argument("--checkpoint", type=Path, action="append", default=None)
    actual_batch_route_parser.add_argument("--route", type=str, action="append", required=True)
    actual_batch_route_parser.add_argument("--route-pair-type", type=str, required=True)
    actual_batch_route_parser.add_argument("--pair-type", type=str, action="append", required=True)
    actual_batch_route_parser.add_argument("--split", type=str, action="append", default=None)
    actual_batch_route_parser.add_argument("--train-split", type=str, default="train")
    actual_batch_route_parser.add_argument("--max-pairs-per-type", type=int, default=64)
    actual_batch_route_parser.add_argument("--min-pairs-per-type", type=int, default=1)
    actual_batch_route_parser.add_argument("--loss-match-tolerance", type=float, default=1.0e-4)
    actual_batch_route_parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command == "generate-benchmark":
        from circuit.io import read_json

        raw_config = read_json(args.config)
        benchmark_type = raw_config.get("benchmark_type")
        if benchmark_type is None:
            raise KeyError(f"Missing benchmark_type in benchmark config: {args.config}")
        if benchmark_type == "symbolic_kv_stream":
            output_dir = generate_symbolic_kv_stream_benchmark(args.config, overwrite=args.overwrite)
        elif benchmark_type == "legacy_symbolic_kv":
            output_dir = generate_symbolic_kv_benchmark(args.config, overwrite=args.overwrite)
        else:
            raise ValueError(f"Unsupported benchmark_type: {benchmark_type}")
        print(output_dir)
        return
    if args.command == "train":
        output_dir = train_from_config(
            args.config,
            overwrite=args.overwrite,
            resume_checkpoint=args.resume_checkpoint,
        )
        print(output_dir)
        return
    if args.command == "evaluate":
        results = _evaluate_command(args.config, args.checkpoint, args.split)
        print(results)
        return
    if args.command == "analyze-checkpoint":
        output_path = _analyze_checkpoint_command(args.config, args.checkpoint, args.output)
        print(output_path)
        return
    if args.command == "formation-trace":
        trace_path, summary_path = _formation_trace_command(args.config, args.checkpoint_dir, args.output)
        print({"trace": str(trace_path), "summary": str(summary_path)})
        return
    if args.command == "generate-probe-set":
        probe_path, metadata_path = generate_probe_set(
            benchmark_dir=args.benchmark_dir,
            output_path=args.output,
            examples_per_split=args.examples_per_split,
            seed=args.seed,
            overwrite=args.overwrite,
            split_names=args.split,
        )
        print({"probe_set": str(probe_path), "metadata": str(metadata_path)})
        return
    if args.command == "checkpoint-sweep":
        metrics_path, summary_path = run_checkpoint_sweep(
            config_path=args.config,
            probe_set_path=args.probe_set,
            output_path=args.output,
            checkpoint_dir=args.checkpoint_dir,
            device_name=args.device,
            create_probe_set_if_missing=args.create_probe_set,
            probe_examples_per_split=args.probe_examples_per_split,
            probe_seed=args.probe_seed,
            overwrite_probe_set=args.overwrite_probe_set,
        )
        print({"metrics": str(metrics_path), "summary": str(summary_path)})
        return
    if args.command == "birth-window-analyze":
        output_path = analyze_birth_windows(
            sweep_metrics_path=args.sweep_metrics,
            sweep_summary_path=args.sweep_summary,
            output_path=args.output,
            top_k=args.top_k,
        )
        print(output_path)
        return
    if args.command == "birth-window-compare":
        output_path = compare_birth_window_checkpoints(
            config_path=args.config,
            probe_set_path=args.probe_set,
            sweep_metrics_path=args.sweep_metrics,
            target_step=args.target_step,
            source_steps=args.source_step,
            output_path=args.output,
            checkpoint_dir=args.checkpoint_dir,
            device_name=args.device,
            top_k_components=args.top_k_components,
            max_neurons_per_layer=args.max_neurons_per_layer,
            stage_names=args.stage,
        )
        print(output_path)
        return
    if args.command == "feature-analyze":
        output_path, sae_state_path = analyze_checkpoint_features(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            probe_set_path=args.probe_set,
            stage_name=args.stage,
            output_path=args.output,
            source_checkpoint_path=args.source_checkpoint,
            device_name=args.device,
            num_features=args.num_features,
            train_steps=args.train_steps,
            learning_rate=args.learning_rate,
            l1_coefficient=args.l1_coefficient,
            sae_batch_size=args.sae_batch_size,
            top_k_features=args.top_k_features,
            top_k_examples=args.top_k_examples,
        )
        print({"report": str(output_path), "sae_weights": str(sae_state_path)})
        return
    if args.command == "shared-feature-fit":
        basis_path, manifest_path, feature_summary_path = shared_feature_fit(
            config_path=args.config,
            probe_set_path=args.probe_set,
            stage_name=args.stage,
            output_dir=args.output_dir,
            checkpoint_paths=args.checkpoint,
            checkpoint_dir=args.checkpoint_dir,
            device_name=args.device,
            num_features=args.num_features,
            train_steps=args.train_steps,
            learning_rate=args.learning_rate,
            l1_coefficient=args.l1_coefficient,
            batch_size=args.batch_size,
        )
        print(
            {
                "basis": str(basis_path),
                "manifest": str(manifest_path),
                "feature_summary": str(feature_summary_path),
            }
        )
        return
    if args.command == "feature-trajectory-sweep":
        trajectories_path, summary_path, split_profiles_path, plot_paths = feature_trajectory_sweep(
            config_path=args.config,
            probe_set_path=args.probe_set,
            basis_path=args.basis,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
        )
        print(
            {
                "trajectories": str(trajectories_path),
                "summary": str(summary_path),
                "split_profiles": str(split_profiles_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "feature-birth-analyze":
        births_path, summary_path, plot_path = feature_birth_analyze(
            trajectories_path=args.trajectories,
            output_dir=args.output_dir,
            thresholds={
                "mean_activation": args.threshold_mean_activation,
                "active_fraction": args.threshold_active_fraction,
                "correctness_gap": args.threshold_correctness_gap,
                "heldout_gap": args.threshold_heldout_gap,
            },
            delta_threshold=args.delta_threshold,
            window=args.window,
        )
        print({"births": str(births_path), "summary": str(summary_path), "plot": str(plot_path)})
        return
    if args.command == "subset-trajectory":
        trajectory_path, plot_path = subset_trajectory(
            trajectories_path=args.trajectories,
            output_path=args.output,
            feature_ids=args.feature,
            feature_family_rank_path=args.feature_family_rank,
            ranking_name=args.ranking_name,
            subset_size=args.subset_size,
        )
        print({"subset_trajectory": str(trajectory_path), "plot": str(plot_path)})
        return
    if args.command == "subset-birth":
        birth_path, plot_path = subset_birth_analyze(
            subset_trajectory_path=args.subset_trajectory,
            output_path=args.output,
            thresholds={
                "mean_activation": args.threshold_mean_activation,
                "active_fraction": args.threshold_active_fraction,
                "correctness_gap": args.threshold_correctness_gap,
                "heldout_gap": args.threshold_heldout_gap,
            },
            delta_threshold=args.delta_threshold,
            window=args.window,
        )
        print({"subset_birth": str(birth_path), "plot": str(plot_path)})
        return
    if args.command == "subset-competition":
        competition_path, plot_path = subset_competition(
            config_path=args.config,
            probe_set_path=args.probe_set,
            basis_path=args.basis,
            source_checkpoint_path=args.source_checkpoint,
            target_checkpoint_path=args.target_checkpoint,
            stage_name=args.stage,
            output_path=args.output,
            subset_a_feature_ids=args.subset_a_feature,
            subset_a_feature_family_rank_path=args.subset_a_feature_family_rank,
            subset_a_ranking_name=args.subset_a_ranking_name,
            subset_a_subset_size=args.subset_a_subset_size,
            subset_b_feature_ids=args.subset_b_feature,
            subset_b_feature_family_rank_path=args.subset_b_feature_family_rank,
            subset_b_ranking_name=args.subset_b_ranking_name,
            subset_b_subset_size=args.subset_b_subset_size,
            device_name=args.device,
            patch_mode=args.patch_mode,
        )
        print({"subset_competition": str(competition_path), "plot": str(plot_path)})
        return
    if args.command == "family-update-link":
        update_link_path, plot_paths = family_update_link(
            feature_family_trace_path=args.feature_family_trace,
            subset_trajectory_path=args.subset_trajectory,
            sweep_metrics_path=args.sweep_metrics,
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output,
        )
        print({"update_link": str(update_link_path), "plots": {key: str(value) for key, value in plot_paths.items()}})
        return
    if args.command == "candidate-circuit-registry":
        output_path = build_candidate_circuit_registry(
            feature_family_trace_paths=args.feature_family_trace,
            subset_trajectory_paths=args.subset_trajectory,
            candidate_ids=args.candidate_id,
            basis_paths=args.basis,
            subset_birth_paths=args.subset_birth,
            family_update_link_paths=args.family_update_link,
            output_path=args.output,
        )
        print(output_path)
        return
    if args.command == "circuit-gradient-link":
        output_path = run_circuit_gradient_link(
            config_path=args.config,
            probe_set_path=args.probe_set,
            registry_path=args.registry,
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output,
            device_name=args.device,
            sweep_metrics_path=args.sweep_metrics,
            start_step=args.start_step,
            end_step=args.end_step,
        )
        print(output_path)
        return
    if args.command == "candidate-sweep":
        summary_path, plot_paths = run_candidate_sweep(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            stage_names=args.stage,
            families_paths=args.families,
            feature_compare_paths=args.feature_compare,
            trajectories_paths=args.trajectories,
            basis_paths=args.basis,
            sweep_metrics_path=args.sweep_metrics,
            device_name=args.device,
            ranking_name=args.ranking_name,
            subset_size=args.subset_size,
            min_family_size=args.min_family_size,
            top_k_families=args.top_k_families,
            start_step=args.start_step,
            end_step=args.end_step,
        )
        print({"summary": str(summary_path), "plots": {key: str(value) for key, value in plot_paths.items()}})
        return
    if args.command == "candidate-mechanism-report":
        report_path, markdown_path, plot_paths = build_candidate_mechanism_report(
            registry_path=args.registry,
            gradient_link_path=args.gradient_link,
            output_dir=args.output_dir,
            candidate_ids=args.candidate_id,
            top_k=args.top_k,
            ranking_metric=args.ranking_metric,
            phase_epsilon=args.phase_epsilon,
            top_interval_k=args.top_interval_k,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "candidate-birth-model":
        report_path, markdown_path, plot_paths = build_candidate_birth_model(
            registry_path=args.registry,
            gradient_link_path=args.gradient_link,
            output_dir=args.output_dir,
            candidate_ids=args.candidate_id,
            birth_metric=args.birth_metric,
            prediction_cutoff_step=args.prediction_cutoff_step,
            lookback_intervals=args.lookback_intervals,
            birth_score_threshold=args.birth_score_threshold,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "candidate-coalition-map":
        report_path, markdown_path, plot_paths = build_candidate_coalition_map(
            config_path=args.config,
            probe_set_path=args.probe_set,
            registry_path=args.registry,
            gradient_link_path=args.gradient_link,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            candidate_ids=args.candidate_id,
            device_name=args.device,
            start_step=args.start_step,
            end_step=args.end_step,
            neuron_layers=args.neuron_layer,
            include_individual_features=not args.candidate_only,
            top_k_neurons=args.top_k_neurons,
            trajectory_top_k=args.trajectory_top_k,
            sign_epsilon=args.sign_epsilon,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "candidate-neuron-intervention":
        report_path, markdown_path, plot_paths = build_candidate_neuron_intervention(
            config_path=args.config,
            probe_set_path=args.probe_set,
            coalition_map_path=args.coalition_map,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            checkpoint_step=args.checkpoint_step,
            device_name=args.device,
            top_k_per_set=args.top_k_per_set,
            single_neuron_top_k=args.single_neuron_top_k,
            score_individual_features=args.score_individual_features,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "feature-compare":
        compare_path, plot_path = feature_compare(
            trajectories_path=args.trajectories,
            source_step=args.source_step,
            target_step=args.target_step,
            output_path=args.output,
            top_k=args.top_k,
        )
        print({"compare": str(compare_path), "plot": str(plot_path)})
        return
    if args.command == "feature-family-cluster":
        families_path, family_trajectories_path, graph_path, plot_paths = feature_family_cluster(
            trajectories_path=args.trajectories,
            output_dir=args.output_dir,
            metrics=args.metric,
            similarity_threshold=args.similarity_threshold,
            feature_births_path=args.feature_births,
            top_k_families=args.top_k_families,
        )
        print(
            {
                "families": str(families_path),
                "family_trajectories": str(family_trajectories_path),
                "graph": str(graph_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "feature-family-birth":
        births_path, summary_path, plot_path = feature_family_birth_analyze(
            family_trajectories_path=args.family_trajectories,
            families_path=args.families,
            output_dir=args.output_dir,
            thresholds={
                "mean_activation": args.threshold_mean_activation,
                "active_fraction": args.threshold_active_fraction,
                "correctness_gap": args.threshold_correctness_gap,
                "heldout_gap": args.threshold_heldout_gap,
            },
            delta_threshold=args.delta_threshold,
            window=args.window,
        )
        print({"births": str(births_path), "summary": str(summary_path), "plot": str(plot_path)})
        return
    if args.command == "feature-family-compare":
        compare_path, plot_path = feature_family_compare(
            family_trajectories_path=args.family_trajectories,
            families_path=args.families,
            source_step=args.source_step,
            target_step=args.target_step,
            output_path=args.output,
            top_k=args.top_k,
        )
        print({"compare": str(compare_path), "plot": str(plot_path)})
        return
    if args.command == "feature-family-rank":
        rank_path, plot_paths = feature_family_rank(
            families_path=args.families,
            feature_compare_path=args.feature_compare,
            family_id=args.family,
            output_path=args.output,
        )
        print({"rank": str(rank_path), "plots": {key: str(value) for key, value in plot_paths.items()}})
        return
    if args.command == "feature-family-subpatch":
        subpatch_path, plot_path = feature_family_subpatch(
            config_path=args.config,
            probe_set_path=args.probe_set,
            basis_path=args.basis,
            feature_family_rank_path=args.feature_family_rank,
            source_checkpoint_path=args.source_checkpoint,
            target_checkpoint_path=args.target_checkpoint,
            stage_name=args.stage,
            ranking_name=args.ranking_name,
            subset_sizes=args.subset_size,
            output_path=args.output,
            device_name=args.device,
            patch_mode=args.patch_mode,
        )
        print({"subpatch": str(subpatch_path), "plot": str(plot_path)})
        return
    if args.command == "feature-family-lineage":
        lineage_path, graph_path, plot_paths = feature_family_lineage(
            config_path=args.config,
            probe_set_path=args.probe_set,
            basis_path=args.basis,
            feature_family_rank_path=args.feature_family_rank,
            checkpoint_path=args.checkpoint,
            ranking_name=args.ranking_name,
            subset_size=args.subset_size,
            output_path=args.output,
            device_name=args.device,
            sweep_metrics_path=args.sweep_metrics,
        )
        print(
            {
                "lineage": str(lineage_path),
                "graph": str(graph_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "feature-family-trace":
        trace_path, plot_paths = feature_family_trace(
            feature_family_births_path=args.feature_family_births,
            feature_family_rank_path=args.feature_family_rank,
            feature_family_subpatch_path=args.feature_family_subpatch,
            feature_family_lineage_path=args.feature_family_lineage,
            output_path=args.output,
        )
        print({"trace": str(trace_path), "plots": {key: str(value) for key, value in plot_paths.items()}})
        return
    if args.command == "feature-patch":
        output_path = feature_patch(
            config_path=args.config,
            probe_set_path=args.probe_set,
            basis_path=args.basis,
            source_checkpoint_path=args.source_checkpoint,
            target_checkpoint_path=args.target_checkpoint,
            stage_name=args.stage,
            feature_ids=args.feature,
            output_path=args.output,
            device_name=args.device,
            patch_mode=args.patch_mode,
        )
        print(output_path)
        return
    if args.command == "feature-family-patch":
        output_path = feature_family_patch(
            config_path=args.config,
            probe_set_path=args.probe_set,
            basis_path=args.basis,
            families_path=args.families,
            family_ids=args.family,
            source_checkpoint_path=args.source_checkpoint,
            target_checkpoint_path=args.target_checkpoint,
            stage_name=args.stage,
            output_path=args.output,
            device_name=args.device,
            patch_mode=args.patch_mode,
        )
        print(output_path)
        return
    if args.command == "feature-lineage":
        output_path, graph_path = feature_lineage(
            config_path=args.config,
            probe_set_path=args.probe_set,
            basis_path=args.basis,
            checkpoint_path=args.checkpoint,
            feature_ids=args.feature,
            output_path=args.output,
            device_name=args.device,
            sweep_metrics_path=args.sweep_metrics,
        )
        print({"lineage": str(output_path), "graph": str(graph_path)})
        return
    if args.command == "analysis-report":
        report_path, manifest_path, timeline_plot_path = build_analysis_report(
            analysis_dir=args.analysis_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(
            {
                "report": str(report_path),
                "manifest": str(manifest_path),
                "timeline_plot": str(timeline_plot_path),
            }
        )
        return
    if args.command == "select-reference":
        result = select_reference_configuration(
            args.run_dir,
            device_name=args.device,
            min_validation_answer_accuracy=args.min_validation_answer_accuracy,
        )
        if args.output is not None:
            write_json(args.output, result)
            print(args.output)
        else:
            print(result)
        return
    if args.command == "dataset-geometry-report":
        report_path, markdown_path, plot_paths = build_dataset_geometry_report(
            benchmark_dir=args.benchmark_dir,
            output_dir=args.output_dir,
            top_k_pairs=args.top_k_pairs,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "attention-geometry-trace":
        report_path, markdown_path, rows_path, plot_paths = run_attention_geometry_trace(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            top_k_tokens=args.top_k_tokens,
            top_k_plot_heads=args.top_k_plot_heads,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "rows": str(rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "path-logit-decomposition":
        report_path, markdown_path, plot_paths = run_path_logit_decomposition(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            ablation_top_k=args.ablation_top_k,
            ablation_steps=args.ablation_step,
            top_k_plot_components=args.top_k_plot_components,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "prompt-neuron-trace":
        report_path, markdown_path, plot_paths = run_prompt_neuron_trace(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            mlp_layers=args.mlp_layer,
            activation_threshold=args.activation_threshold,
            top_k_per_query=args.top_k_per_query,
            ablation_top_k_per_layer=args.ablation_top_k_per_layer,
            ablation_steps=args.ablation_step,
            ablation_neurons=args.ablation_neuron,
            top_k_plot_neurons=args.top_k_plot_neurons,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "geometry-subspace-intervention":
        report_path, markdown_path, aggregate_rows_path, query_rows_path, plot_paths = run_geometry_subspace_intervention(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            stage_name=args.stage,
            subspace_name=args.subspace,
            rank=args.rank,
            operation=args.operation,
            position_role=args.position_role,
            query_mode=args.query_mode,
            head_layer=args.head_layer,
            head=args.head,
            progress_every_queries=args.progress_every_queries,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "aggregate_rows": str(aggregate_rows_path),
                "query_rows": str(query_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "causal-variable-patch":
        report_path, markdown_path, aggregate_rows_path, query_rows_path, pair_rows_path, plot_paths = run_causal_variable_patch(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            stage_name=args.stage,
            subspace_name=args.subspace,
            rank=args.rank,
            position_role=args.position_role,
            pair_types=args.pair_type,
            head_layer=args.head_layer,
            head=args.head,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            split_filter=args.split,
            min_recovery_denominator=args.min_recovery_denominator,
            progress_every_pairs=args.progress_every_pairs,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "aggregate_rows": str(aggregate_rows_path),
                "query_rows": str(query_rows_path),
                "pair_rows": str(pair_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "candidate-route-gradient-selection":
        report_path, markdown_path, metric_rows_path, pairwise_rows_path, pair_rows_path, plot_paths = run_candidate_route_gradient_selection(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            stage_name=args.stage,
            subspace_name=args.subspace,
            rank=args.rank,
            position_role=args.position_role,
            pair_types=args.pair_type,
            head_layer=args.head_layer,
            head=args.head,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            split_filter=args.split,
            loss_side=args.loss_side,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "metric_rows": str(metric_rows_path),
                "pairwise_rows": str(pairwise_rows_path),
                "pair_rows": str(pair_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "route-gradient-decomposition":
        report_path, markdown_path, metric_rows_path, decomposition_rows_path, group_rows_path, pair_rows_path, plot_paths = run_route_gradient_decomposition(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            stage_name=args.stage,
            subspace_name=args.subspace,
            rank=args.rank,
            position_role=args.position_role,
            pair_types=args.pair_type,
            head_layer=args.head_layer,
            head=args.head,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            split_filter=args.split,
            loss_side=args.loss_side,
            decomposition_modes=args.decompose,
            top_k_groups=args.top_k_groups,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "metric_rows": str(metric_rows_path),
                "decomposition_rows": str(decomposition_rows_path),
                "group_rows": str(group_rows_path),
                "pair_rows": str(pair_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "checkpoint-update-attribution":
        report_path, markdown_path, metric_rows_path, decomposition_rows_path, group_rows_path, pair_rows_path, plot_paths = run_checkpoint_update_attribution(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            stage_name=args.stage,
            subspace_name=args.subspace,
            rank=args.rank,
            position_role=args.position_role,
            pair_types=args.pair_type,
            head_layer=args.head_layer,
            head=args.head,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            split_filter=args.split,
            decomposition_modes=args.decompose,
            top_k_groups=args.top_k_groups,
            min_error_denominator=args.min_error_denominator,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "metric_rows": str(metric_rows_path),
                "decomposition_rows": str(decomposition_rows_path),
                "group_rows": str(group_rows_path),
                "pair_rows": str(pair_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "data-update-attribution":
        report_path, markdown_path, route_rows_path, data_rows_path, pair_rows_path, plot_paths = run_data_update_attribution(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            stage_name=args.stage,
            subspace_name=args.subspace,
            rank=args.rank,
            position_role=args.position_role,
            pair_types=args.pair_type,
            route_pair_type=args.route_pair_type,
            route_split=args.route_split,
            data_group_fields=args.data_group_field,
            head_layer=args.head_layer,
            head=args.head,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            split_filter=args.split,
            loss_side=args.loss_side,
            top_k_data_groups=args.top_k_data_groups,
            min_error_denominator=args.min_error_denominator,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "route_rows": str(route_rows_path),
                "data_rows": str(data_rows_path),
                "pair_rows": str(pair_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "route-competition-report":
        report_path, markdown_path, route_rows_path, data_rows_path, pair_rows_path, plot_paths = run_route_competition_report(
            config_path=args.config,
            probe_set_path=args.probe_set,
            train_probe_set_path=args.train_probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            raw_route_specs=args.route,
            route_pair_type=args.route_pair_type,
            eval_pair_types=args.pair_type,
            train_pair_types=args.train_pair_type,
            data_group_fields=args.data_group_field,
            eval_split_filter=args.eval_split,
            train_split_filter=args.train_split,
            eval_loss_side=args.eval_loss_side,
            train_loss_side=args.train_loss_side,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            min_error_denominator=args.min_error_denominator,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "route_rows": str(route_rows_path),
                "data_rows": str(data_rows_path),
                "pair_rows": str(pair_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "attention-score-delta-decomposition":
        report_path, markdown_path, metric_rows_path, score_rows_path, component_rows_path, pair_rows_path, plot_paths = run_attention_score_delta_decomposition(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            head_layer=args.head_layer,
            head=args.head,
            score_query_role=args.score_query_role,
            score_key_roles=args.score_key_role,
            record_sides=args.record_side,
            pair_types=args.pair_type,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            split_filter=args.split,
            reconstruction_tolerance=args.reconstruction_tolerance,
            top_k_components=args.top_k_components,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "metric_rows": str(metric_rows_path),
                "score_rows": str(score_rows_path),
                "component_rows": str(component_rows_path),
                "pair_rows": str(pair_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "attention-score-update-attribution":
        report_path, markdown_path, metric_rows_path, decomposition_rows_path, group_rows_path, score_rows_path, pair_rows_path, plot_paths = run_attention_score_update_attribution(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            head_layer=args.head_layer,
            head=args.head,
            score_query_role=args.score_query_role,
            score_key_roles=args.score_key_role,
            record_sides=args.record_side,
            score_components=args.score_component,
            pair_types=args.pair_type,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            split_filter=args.split,
            decomposition_modes=args.decompose,
            reconstruction_tolerance=args.reconstruction_tolerance,
            top_k_groups=args.top_k_groups,
            min_error_denominator=args.min_error_denominator,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "metric_rows": str(metric_rows_path),
                "decomposition_rows": str(decomposition_rows_path),
                "group_rows": str(group_rows_path),
                "score_rows": str(score_rows_path),
                "pair_rows": str(pair_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "attention-retrieval-separation-update-attribution":
        report_path, markdown_path, metric_rows_path, decomposition_rows_path, group_rows_path, score_rows_path, pair_rows_path, plot_paths = run_attention_retrieval_separation_update_attribution(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            head_layer=args.head_layer,
            head=args.head,
            score_query_role=args.score_query_role,
            support_key_role=args.support_key_role,
            distractor_key_role=args.distractor_key_role,
            record_sides=args.record_side,
            score_components=args.score_component,
            pair_types=args.pair_type,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            split_filter=args.split,
            decomposition_modes=args.decompose,
            reconstruction_tolerance=args.reconstruction_tolerance,
            top_k_groups=args.top_k_groups,
            min_error_denominator=args.min_error_denominator,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "metric_rows": str(metric_rows_path),
                "decomposition_rows": str(decomposition_rows_path),
                "group_rows": str(group_rows_path),
                "score_rows": str(score_rows_path),
                "pair_rows": str(pair_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "attention-retrieval-chain-report":
        report_path, markdown_path, checkpoint_rows_path, delta_rows_path, pair_metric_rows_path, pair_rows_path, plot_paths = run_attention_retrieval_chain_report(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            head_layer=args.head_layer,
            head=args.head,
            score_query_role=args.score_query_role,
            support_key_role=args.support_key_role,
            distractor_key_role=args.distractor_key_role,
            record_sides=args.record_side,
            pair_types=args.pair_type,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            split_filter=args.split,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "checkpoint_rows": str(checkpoint_rows_path),
                "delta_rows": str(delta_rows_path),
                "pair_metric_rows": str(pair_metric_rows_path),
                "pair_rows": str(pair_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "attention-downstream-update-attribution":
        report_path, markdown_path, metric_rows_path, decomposition_rows_path, group_rows_path, scalar_rows_path, pair_rows_path, plot_paths = run_attention_downstream_update_attribution(
            config_path=args.config,
            probe_set_path=args.probe_set,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            head_layer=args.head_layer,
            head=args.head,
            score_query_role=args.score_query_role,
            support_key_role=args.support_key_role,
            distractor_key_role=args.distractor_key_role,
            record_sides=args.record_side,
            scalar_names=args.scalar,
            pair_types=args.pair_type,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            split_filter=args.split,
            decomposition_modes=args.decompose,
            top_k_groups=args.top_k_groups,
            min_error_denominator=args.min_error_denominator,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "metric_rows": str(metric_rows_path),
                "decomposition_rows": str(decomposition_rows_path),
                "group_rows": str(group_rows_path),
                "scalar_rows": str(scalar_rows_path),
                "pair_rows": str(pair_rows_path),
                "plots": {key: str(value) for key, value in plot_paths.items()},
            }
        )
        return
    if args.command == "optimizer-update-trace":
        if args.end_step is None and args.num_steps is None:
            raise ValueError("Expected --end-step or --num-steps.")
        if args.end_step is not None and args.num_steps is not None:
            raise ValueError("Expected only one of --end-step or --num-steps.")
        if args.end_step is None:
            from circuit.runtime import load_checkpoint, require_device

            from circuit.config import TrainSpec

            spec = TrainSpec.from_path(args.config)
            device = require_device(args.device if args.device is not None else spec.device)
            checkpoint = load_checkpoint(args.resume_checkpoint, device)
            end_step = int(checkpoint["step"]) + int(args.num_steps)
        else:
            end_step = int(args.end_step)
        report_path, markdown_path, step_rows_path, batch_rows_path, parameter_update_rows_path, checkpoint_dir = run_optimizer_update_trace(
            config_path=args.config,
            resume_checkpoint=args.resume_checkpoint,
            output_dir=args.output_dir,
            end_step=end_step,
            device_name=args.device,
            train_split=args.train_split,
            checkpoint_every_steps=args.checkpoint_every,
            progress_every_steps=args.progress_every,
            top_k_parameters=args.top_k_parameters,
            overwrite=args.overwrite,
            require_historical_replay=args.require_historical_replay,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "step_rows": str(step_rows_path),
                "batch_rows": str(batch_rows_path),
                "parameter_update_rows": str(parameter_update_rows_path),
                "checkpoint_dir": str(checkpoint_dir),
            }
        )
        return
    if args.command == "actual-batch-route-attribution":
        report_path, markdown_path, rows_path, pair_rows_path = run_actual_batch_route_attribution(
            config_path=args.config,
            probe_set_path=args.probe_set,
            optimizer_trace_dir=args.optimizer_trace_dir,
            output_dir=args.output_dir,
            device_name=args.device,
            checkpoint_paths=args.checkpoint,
            raw_route_specs=args.route,
            route_pair_type=args.route_pair_type,
            pair_types=args.pair_type,
            split_filter=args.split,
            train_split=args.train_split,
            max_pairs_per_type=args.max_pairs_per_type,
            min_pairs_per_type=args.min_pairs_per_type,
            loss_match_tolerance=args.loss_match_tolerance,
            overwrite=args.overwrite,
        )
        print(
            {
                "report": str(report_path),
                "markdown": str(markdown_path),
                "rows": str(rows_path),
                "pair_rows": str(pair_rows_path),
            }
        )
        return
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
