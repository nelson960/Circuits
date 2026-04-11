from __future__ import annotations

from pathlib import Path
import shutil

import torch
from torch.utils.data import DataLoader

from circuit.analysis.analysis_report import build_analysis_report
from circuit.analysis.formation import compute_head_ablation_importance, compute_head_localization, compute_qrw_batch
from circuit.analysis.birth_windows import analyze_birth_windows
from circuit.analysis.birth_window_compare import compare_birth_window_checkpoints
from circuit.analysis.checkpoint_sweep import generate_probe_set, run_checkpoint_sweep
from circuit.analysis.feature_analysis import analyze_checkpoint_features
from circuit.analysis.shared_feature_dynamics import (
    family_update_link,
    feature_birth_analyze,
    feature_compare,
    feature_family_birth_analyze,
    feature_family_cluster,
    feature_family_compare,
    feature_family_lineage,
    feature_family_patch,
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
from circuit.config import LearningRateScheduleSpec, OptimizationSpec, TrainSpec
from circuit.data.symbolic_kv_stream import (
    SymbolicKVDataset,
    collate_symbolic_kv,
    generate_symbolic_kv_stream_benchmark,
)
from circuit.eval import evaluate_split
from circuit.io import iter_jsonl, read_json, write_json
from circuit.model.decoder import DecoderOnlyTransformer
from circuit.reference import rank_reference_candidates
from circuit.runtime import load_model_state
from circuit.train import _compute_learning_rate, load_model_from_checkpoint, train_from_config
from circuit.vocab import Vocabulary

from .helpers import write_small_train_config


def test_analysis_and_training_pipeline(tmp_path: Path, benchmark_config_path: Path) -> None:
    benchmark_dir = generate_symbolic_kv_stream_benchmark(benchmark_config_path)
    dataset = SymbolicKVDataset(benchmark_dir, "validation_iid")
    vocab = Vocabulary.from_metadata(dataset.metadata["vocabulary"])
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: collate_symbolic_kv(batch, vocab.pad_token_id),
    )
    batch = next(iter(loader))

    model = DecoderOnlyTransformer(
        spec=type("Spec", (), {
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "max_seq_len": 96,
        })(),
        vocab_size=len(vocab.tokens),
    )
    fast_outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
    outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], return_attentions=True)
    mlp_outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"], return_mlp_states=True)
    qrw = compute_qrw_batch(
        logits=outputs.logits,
        attentions=outputs.attentions or [],
        batch=batch,
        value_token_ids=batch["input_ids"].new_tensor(vocab.value_token_ids),
    )
    localization = compute_head_localization(model=model, batches=[batch])
    ablation = compute_head_ablation_importance(model=model, batches=[batch])

    assert set(qrw) == {"q", "r", "w", "answer_accuracy"}
    assert fast_outputs.logits.shape == outputs.logits.shape
    assert len(localization) == 4
    assert len(ablation) == 4
    assert mlp_outputs.mlp_states is not None
    assert set(mlp_outputs.mlp_states) == {
        "layer_0_hidden",
        "layer_0_hidden_pre",
        "layer_1_hidden",
        "layer_1_hidden_pre",
    }

    train_config = write_small_train_config(tmp_path / "train_config.json", benchmark_dir)
    run_dir = train_from_config(train_config)
    checkpoint_path = run_dir / "checkpoints" / "step_000004.pt"
    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    context = load_model_from_checkpoint(config_path=train_config, checkpoint_path=checkpoint_path)

    assert checkpoint_path.exists()
    assert best_checkpoint_path.exists()
    assert context["checkpoint"]["step"] == 4
    probe_set_path, probe_metadata_path = generate_probe_set(
        benchmark_dir=benchmark_dir,
        output_path=tmp_path / "probe_set.jsonl",
        examples_per_split=2,
        seed=13,
    )
    metrics_path, summary_path = run_checkpoint_sweep(
        config_path=train_config,
        probe_set_path=probe_set_path,
        output_path=tmp_path / "checkpoint_metrics.jsonl",
    )
    sweep_rows = list(iter_jsonl(metrics_path))
    sweep_summary = read_json(summary_path)

    assert probe_set_path.exists()
    assert probe_metadata_path.exists()
    assert len(sweep_rows) == 2
    assert "answer_probe_accuracy_by_stage" in sweep_rows[0]
    assert "top_mlps_by_ablation" in sweep_rows[0]
    assert "top_neurons_by_write" in sweep_rows[0]
    assert "top_neurons_by_ablation" in sweep_rows[0]
    assert sweep_summary["num_checkpoints"] == 2

    auto_metrics_path, auto_summary_path = run_checkpoint_sweep(
        config_path=train_config,
        probe_set_path=tmp_path / "auto_probe_set.jsonl",
        output_path=tmp_path / "auto_checkpoint_metrics.jsonl",
        create_probe_set_if_missing=True,
        probe_examples_per_split=2,
        probe_seed=19,
    )
    auto_rows = list(iter_jsonl(auto_metrics_path))
    auto_summary = read_json(auto_summary_path)

    assert (tmp_path / "auto_probe_set.jsonl").exists()
    assert (tmp_path / "auto_probe_set.metadata.json").exists()
    assert len(auto_rows) == 2
    assert auto_summary["num_checkpoints"] == 2

    birth_windows_path = analyze_birth_windows(
        sweep_metrics_path=metrics_path,
        sweep_summary_path=summary_path,
        output_path=tmp_path / "birth_windows.json",
        top_k=3,
    )
    birth_windows = read_json(birth_windows_path)

    assert birth_windows_path.exists()
    assert "layered_plan_status" in birth_windows
    assert "birth_windows" in birth_windows
    assert birth_windows["birth_windows"]
    assert "residual_stream" in birth_windows["birth_windows"][0]
    assert "heads" in birth_windows["birth_windows"][0]
    assert "mlp_blocks" in birth_windows["birth_windows"][0]
    assert "neurons" in birth_windows["birth_windows"][0]

    compare_output_path = compare_birth_window_checkpoints(
        config_path=train_config,
        probe_set_path=probe_set_path,
        sweep_metrics_path=metrics_path,
        target_step=2,
        source_steps=[4],
        output_path=tmp_path / "birth_window_compare.json",
        device_name="cpu",
        top_k_components=3,
        max_neurons_per_layer=2,
    )
    compare_payload = read_json(compare_output_path)

    assert compare_output_path.exists()
    assert compare_payload["target_step"] == 2
    assert compare_payload["source_comparisons"][0]["source_step"] == 4
    assert compare_payload["source_comparisons"][0]["residual_patch_results"]
    assert compare_payload["source_comparisons"][0]["head_ablation_comparisons"]
    assert compare_payload["source_comparisons"][0]["mlp_ablation_comparisons"]
    assert compare_payload["source_comparisons"][0]["neuron_group_ablation_comparisons"]

    feature_output_path, feature_sae_path = analyze_checkpoint_features(
        config_path=train_config,
        checkpoint_path=checkpoint_path,
        source_checkpoint_path=best_checkpoint_path,
        probe_set_path=probe_set_path,
        stage_name="layer_1_post_mlp",
        output_path=tmp_path / "feature_analysis.json",
        device_name="cpu",
        num_features=8,
        train_steps=20,
        learning_rate=1e-3,
        l1_coefficient=1e-3,
        sae_batch_size=16,
        top_k_features=4,
        top_k_examples=2,
    )
    feature_payload = read_json(feature_output_path)

    assert feature_output_path.exists()
    assert feature_sae_path.exists()
    assert feature_payload["stage_name"] == "layer_1_post_mlp"
    assert feature_payload["checkpoint_step"] == 4
    assert feature_payload["source_checkpoint_step"] is not None
    assert "by_mean_activation" in feature_payload["top_features"]
    assert "by_abs_mean_activation_delta_vs_source" in feature_payload["top_features"]
    assert len(feature_payload["feature_rows"]) == 8

    shared_basis_dir = tmp_path / "shared_features"
    basis_path, basis_manifest_path, basis_feature_summary_path = shared_feature_fit(
        config_path=train_config,
        probe_set_path=probe_set_path,
        stage_name="layer_1_post_mlp",
        output_dir=shared_basis_dir,
        checkpoint_paths=[run_dir / "checkpoints" / "step_000002.pt", checkpoint_path],
        device_name="cpu",
        num_features=8,
        train_steps=20,
        learning_rate=1e-3,
        l1_coefficient=1e-3,
        batch_size=16,
    )
    basis_manifest = read_json(basis_manifest_path)

    assert basis_path.exists()
    assert basis_manifest_path.exists()
    assert basis_feature_summary_path.exists()
    assert basis_manifest["stage_name"] == "layer_1_post_mlp"
    assert basis_manifest["num_features"] == 8

    trajectories_path, trajectory_summary_path, split_profiles_path, plot_paths = feature_trajectory_sweep(
        config_path=train_config,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        checkpoint_dir=run_dir / "checkpoints",
        output_dir=shared_basis_dir / "trajectories",
        device_name="cpu",
    )
    trajectory_rows = list(iter_jsonl(trajectories_path))
    trajectory_summary = read_json(trajectory_summary_path)

    assert trajectories_path.exists()
    assert trajectory_summary_path.exists()
    assert split_profiles_path.exists()
    assert plot_paths["trajectory_plot"].exists()
    assert plot_paths["heatmap_plot"].exists()
    assert trajectory_rows
    assert trajectory_summary["num_checkpoints"] == 2

    births_path, birth_summary_path, birth_plot_path = feature_birth_analyze(
        trajectories_path=trajectories_path,
        output_dir=shared_basis_dir / "births",
        thresholds={
            "mean_activation": 0.0,
            "active_fraction": 0.0,
            "correctness_gap": -1.0,
            "heldout_gap": -1.0,
        },
        delta_threshold=0.0,
        window=1,
    )
    births_payload = read_json(births_path)

    assert births_path.exists()
    assert birth_summary_path.exists()
    assert birth_plot_path.exists()
    assert births_payload["features"]

    compare_path, compare_plot_path = feature_compare(
        trajectories_path=trajectories_path,
        source_step=2,
        target_step=4,
        output_path=shared_basis_dir / "feature_compare.json",
        top_k=4,
    )
    compare_payload = read_json(compare_path)

    assert compare_path.exists()
    assert compare_plot_path.exists()
    assert compare_payload["source_step"] == 2
    assert compare_payload["target_step"] == 4
    assert compare_payload["diff_rows"]

    family_path, family_trajectories_path, family_graph_path, family_plot_paths = feature_family_cluster(
        trajectories_path=trajectories_path,
        output_dir=shared_basis_dir / "families",
        metrics=["mean_activation", "active_fraction", "correctness_gap", "heldout_gap", "structural_ood_gap"],
        similarity_threshold=0.0,
        feature_births_path=births_path,
        top_k_families=4,
    )
    family_payload = read_json(family_path)
    family_trajectory_rows = list(iter_jsonl(family_trajectories_path))
    family_graph_payload = read_json(family_graph_path)

    assert family_path.exists()
    assert family_trajectories_path.exists()
    assert family_graph_path.exists()
    assert family_plot_paths["similarity_heatmap"].exists()
    assert family_plot_paths["family_trajectory_plot"].exists()
    assert family_payload["num_features"] == 8
    assert family_payload["families"]
    assert sum(int(row["size"]) for row in family_payload["families"]) == 8
    assert family_trajectory_rows
    assert family_graph_payload["nodes"]
    assert family_graph_payload["edges"]
    multi_feature_family_ids = [
        int(row["family_id"])
        for row in family_payload["families"]
        if int(row["size"]) > 1
    ]
    assert multi_feature_family_ids

    family_births_path, family_birth_summary_path, family_birth_plot_path = feature_family_birth_analyze(
        family_trajectories_path=family_trajectories_path,
        families_path=family_path,
        output_dir=shared_basis_dir / "family_births",
        thresholds={
            "mean_activation": 0.0,
            "active_fraction": 0.0,
            "correctness_gap": -1.0,
            "heldout_gap": -1.0,
        },
        delta_threshold=0.0,
        window=1,
    )
    family_births_payload = read_json(family_births_path)

    assert family_births_path.exists()
    assert family_birth_summary_path.exists()
    assert family_birth_plot_path.exists()
    assert family_births_payload["families"]

    family_compare_path, family_compare_plot_path = feature_family_compare(
        family_trajectories_path=family_trajectories_path,
        families_path=family_path,
        source_step=2,
        target_step=4,
        output_path=shared_basis_dir / "feature_family_compare.json",
        top_k=4,
    )
    family_compare_payload = read_json(family_compare_path)

    assert family_compare_path.exists()
    assert family_compare_plot_path.exists()
    assert family_compare_payload["source_step"] == 2
    assert family_compare_payload["target_step"] == 4
    assert family_compare_payload["diff_rows"]

    family_rank_path, family_rank_plot_paths = feature_family_rank(
        families_path=family_path,
        feature_compare_path=compare_path,
        family_id=multi_feature_family_ids[0],
        output_path=shared_basis_dir / "feature_family_rank.json",
    )
    family_rank_payload = read_json(family_rank_path)

    assert family_rank_path.exists()
    assert family_rank_plot_paths["useful_delta_bar"].exists()
    assert family_rank_plot_paths["tradeoff_scatter"].exists()
    assert family_rank_payload["family_id"] == multi_feature_family_ids[0]
    assert family_rank_payload["member_rows"]
    assert family_rank_payload["rankings"]["by_useful_delta"]
    assert family_rank_payload["suggested_subsets"]["by_useful_delta"]

    patch_path = feature_patch(
        config_path=train_config,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        source_checkpoint_path=run_dir / "checkpoints" / "step_000002.pt",
        target_checkpoint_path=checkpoint_path,
        stage_name="layer_1_post_mlp",
        feature_ids=[0, 1],
        output_path=shared_basis_dir / "feature_patch.json",
        device_name="cpu",
        patch_mode="replace",
    )
    patch_payload = read_json(patch_path)

    assert patch_path.exists()
    assert "deltas" in patch_payload
    assert "reconstruction" in patch_payload

    family_patch_path = feature_family_patch(
        config_path=train_config,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        families_path=family_path,
        family_ids=[multi_feature_family_ids[0]],
        source_checkpoint_path=run_dir / "checkpoints" / "step_000002.pt",
        target_checkpoint_path=checkpoint_path,
        stage_name="layer_1_post_mlp",
        output_path=shared_basis_dir / "feature_family_patch.json",
        device_name="cpu",
        patch_mode="replace",
    )
    family_patch_payload = read_json(family_patch_path)

    assert family_patch_path.exists()
    assert family_patch_payload["family_ids"] == [multi_feature_family_ids[0]]
    assert family_patch_payload["resolved_feature_ids"]
    assert family_patch_payload["selected_families"]
    assert "deltas" in family_patch_payload

    family_subpatch_path, family_subpatch_plot_path = feature_family_subpatch(
        config_path=train_config,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        feature_family_rank_path=family_rank_path,
        source_checkpoint_path=run_dir / "checkpoints" / "step_000002.pt",
        target_checkpoint_path=checkpoint_path,
        stage_name="layer_1_post_mlp",
        ranking_name="by_useful_delta",
        subset_sizes=[1, 2],
        output_path=shared_basis_dir / "feature_family_subpatch.json",
        device_name="cpu",
        patch_mode="replace",
    )
    family_subpatch_payload = read_json(family_subpatch_path)

    assert family_subpatch_path.exists()
    assert family_subpatch_plot_path.exists()
    assert family_subpatch_payload["family_id"] == multi_feature_family_ids[0]
    assert family_subpatch_payload["ranking_name"] == "by_useful_delta"
    assert len(family_subpatch_payload["subset_results"]) == 2
    assert family_subpatch_payload["best_subset_by_heldout"]

    family_lineage_path, family_lineage_graph_path, family_lineage_plot_paths = feature_family_lineage(
        config_path=train_config,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        feature_family_rank_path=family_rank_path,
        checkpoint_path=checkpoint_path,
        ranking_name="by_useful_delta",
        subset_size=2,
        output_path=shared_basis_dir / "feature_family_lineage.json",
        device_name="cpu",
        sweep_metrics_path=metrics_path,
    )
    family_lineage_payload = read_json(family_lineage_path)
    family_lineage_graph_payload = read_json(family_lineage_graph_path)

    assert family_lineage_path.exists()
    assert family_lineage_graph_path.exists()
    assert family_lineage_plot_paths["head_bar"].exists()
    assert family_lineage_plot_paths["mlp_bar"].exists()
    assert family_lineage_plot_paths["neuron_group_bar"].exists()
    assert family_lineage_payload["family_id"] == multi_feature_family_ids[0]
    assert family_lineage_payload["selected_feature_ids"]
    assert family_lineage_payload["aggregated_head_effects"]
    assert family_lineage_payload["aggregated_mlp_effects"]
    assert family_lineage_payload["aggregated_neuron_group_effects"]
    assert family_lineage_graph_payload["nodes"]
    assert family_lineage_graph_payload["edges"]

    family_trace_path, family_trace_plot_paths = feature_family_trace(
        feature_family_births_path=family_births_path,
        feature_family_rank_path=family_rank_path,
        feature_family_subpatch_path=family_subpatch_path,
        feature_family_lineage_path=family_lineage_path,
        output_path=shared_basis_dir / "feature_family_trace.json",
    )
    family_trace_payload = read_json(family_trace_path)

    assert family_trace_path.exists()
    assert family_trace_plot_paths["trace_plot"].exists()
    assert family_trace_payload["family_id"] == multi_feature_family_ids[0]
    assert family_trace_payload["trace_subset"]["feature_ids"]
    assert "patch_result" in family_trace_payload["trace_subset"]
    assert "lineage" in family_trace_payload["trace_subset"]
    assert "family_birth" in family_trace_payload
    assert "trace_summary" in family_trace_payload

    subset_trajectory_path, subset_trajectory_plot_path = subset_trajectory(
        trajectories_path=trajectories_path,
        output_path=shared_basis_dir / "subset_trajectory.json",
        feature_family_rank_path=family_rank_path,
        ranking_name="by_useful_delta",
        subset_size=2,
    )
    subset_trajectory_payload = read_json(subset_trajectory_path)

    assert subset_trajectory_path.exists()
    assert subset_trajectory_plot_path.exists()
    assert subset_trajectory_payload["feature_ids"]
    assert len(subset_trajectory_payload["rows"]) == 2

    subset_birth_path, subset_birth_plot_path = subset_birth_analyze(
        subset_trajectory_path=subset_trajectory_path,
        output_path=shared_basis_dir / "subset_birth.json",
        thresholds={
            "mean_activation": 0.0,
            "active_fraction": 0.0,
            "correctness_gap": -1.0,
            "heldout_gap": -1.0,
        },
        delta_threshold=0.0,
        window=1,
    )
    subset_birth_payload = read_json(subset_birth_path)

    assert subset_birth_path.exists()
    assert subset_birth_plot_path.exists()
    assert subset_birth_payload["feature_ids"] == subset_trajectory_payload["feature_ids"]
    assert "births" in subset_birth_payload

    competition_path, competition_plot_path = subset_competition(
        config_path=train_config,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        source_checkpoint_path=run_dir / "checkpoints" / "step_000002.pt",
        target_checkpoint_path=checkpoint_path,
        stage_name="layer_1_post_mlp",
        output_path=shared_basis_dir / "subset_competition.json",
        subset_a_feature_family_rank_path=family_rank_path,
        subset_a_ranking_name="by_useful_delta",
        subset_a_subset_size=1,
        subset_b_feature_ids=[0, 1],
        device_name="cpu",
        patch_mode="replace",
    )
    competition_payload = read_json(competition_path)

    assert competition_path.exists()
    assert competition_plot_path.exists()
    assert competition_payload["subset_a"]["feature_ids"]
    assert competition_payload["subset_b"]["feature_ids"] == [0, 1]
    assert competition_payload["union_subset"]["feature_ids"]
    assert "interaction" in competition_payload

    family_update_link_path, family_update_link_plot_paths = family_update_link(
        feature_family_trace_path=family_trace_path,
        subset_trajectory_path=subset_trajectory_path,
        sweep_metrics_path=metrics_path,
        checkpoint_dir=run_dir / "checkpoints",
        output_path=shared_basis_dir / "family_update_link.json",
    )
    family_update_link_payload = read_json(family_update_link_path)

    assert family_update_link_path.exists()
    assert family_update_link_plot_paths["interval_plot"].exists()
    assert family_update_link_plot_paths["useful_correlation_plot"].exists()
    assert family_update_link_payload["family_id"] == multi_feature_family_ids[0]
    assert family_update_link_payload["selected_feature_ids"] == subset_trajectory_payload["feature_ids"]
    assert len(family_update_link_payload["interval_rows"]) == 1
    assert "correlation_summary" in family_update_link_payload
    assert "top_intervals" in family_update_link_payload

    lineage_path, lineage_graph_path = feature_lineage(
        config_path=train_config,
        probe_set_path=probe_set_path,
        basis_path=basis_path,
        checkpoint_path=checkpoint_path,
        feature_ids=[0, 1],
        output_path=shared_basis_dir / "feature_lineage.json",
        device_name="cpu",
        sweep_metrics_path=metrics_path,
    )
    lineage_payload = read_json(lineage_path)
    lineage_graph_payload = read_json(lineage_graph_path)

    assert lineage_path.exists()
    assert lineage_graph_path.exists()
    assert lineage_payload["head_effects"]
    assert lineage_payload["mlp_effects"]
    assert "nodes" in lineage_graph_payload
    assert "edges" in lineage_graph_payload

    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    shutil.copy2(metrics_path, analysis_dir / "checkpoint_metrics.jsonl")
    shutil.copy2(summary_path, analysis_dir / "checkpoint_metrics_summary.json")
    shutil.copy2(birth_windows_path, analysis_dir / "birth_window_analysis.json")
    stage_dir = analysis_dir / "shared_features" / "layer_1_post_mlp"
    shutil.copytree(shared_basis_dir, stage_dir)

    report_path, report_manifest_path, timeline_plot_path = build_analysis_report(
        analysis_dir=analysis_dir,
        output_dir=analysis_dir / "report",
    )
    report_html = report_path.read_text(encoding="utf-8")
    report_manifest = read_json(report_manifest_path)

    assert report_path.exists()
    assert report_manifest_path.exists()
    assert timeline_plot_path.exists()
    assert "Stage: layer_1_post_mlp" in report_html
    assert "Checkpoint timeline" in report_html
    assert report_manifest["num_checkpoints"] == 2
    assert report_manifest["stages"][0]["stage_name"] == "layer_1_post_mlp"


def test_resume_training_from_checkpoint(tmp_path: Path, benchmark_config_path: Path) -> None:
    benchmark_dir = generate_symbolic_kv_stream_benchmark(benchmark_config_path)
    initial_config = write_small_train_config(tmp_path / "train_initial.json", benchmark_dir, num_steps=2)
    run_dir = train_from_config(initial_config)
    resume_checkpoint = run_dir / "checkpoints" / "step_000002.pt"
    resumed_config = write_small_train_config(tmp_path / "train_resumed.json", benchmark_dir, num_steps=4)

    resumed_run_dir = train_from_config(resumed_config, resume_checkpoint=resume_checkpoint)
    final_checkpoint = resumed_run_dir / "checkpoints" / "step_000004.pt"
    resumed_context = load_model_from_checkpoint(config_path=resumed_config, checkpoint_path=final_checkpoint)

    assert final_checkpoint.exists()
    assert resumed_context["checkpoint"]["step"] == 4


def test_best_checkpoint_only_mode(tmp_path: Path, benchmark_config_path: Path) -> None:
    benchmark_dir = generate_symbolic_kv_stream_benchmark(benchmark_config_path)
    train_config = write_small_train_config(
        tmp_path / "train_best_only.json",
        benchmark_dir,
        save_step_checkpoints=False,
        save_best_checkpoint=True,
    )

    run_dir = train_from_config(train_config)
    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    numbered_checkpoints = sorted((run_dir / "checkpoints").glob("step_*.pt"))

    assert best_checkpoint_path.exists()
    assert numbered_checkpoints == []


def test_train_config_supports_full_eval(tmp_path: Path, benchmark_config_path: Path) -> None:
    benchmark_dir = generate_symbolic_kv_stream_benchmark(benchmark_config_path)
    train_config = write_small_train_config(
        tmp_path / "train_full_eval.json",
        benchmark_dir,
        max_eval_batches=None,
    )
    payload = read_json(train_config)
    write_json(train_config, payload)

    spec = TrainSpec.from_path(train_config)

    assert spec.evaluation.max_eval_batches is None


def test_cosine_decay_learning_rate_schedule() -> None:
    optimization = OptimizationSpec(
        learning_rate=4e-4,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.95,
        grad_clip_norm=1.0,
        warmup_steps=200,
        schedule=LearningRateScheduleSpec(
            kind="cosine_decay",
            decay_start_step=8000,
            decay_end_step=16000,
            min_learning_rate=4e-5,
        ),
    )

    assert torch.isclose(torch.tensor(_compute_learning_rate(optimization, 100)), torch.tensor(2e-4))
    assert torch.isclose(torch.tensor(_compute_learning_rate(optimization, 8000)), torch.tensor(4e-4))
    midpoint_lr = _compute_learning_rate(optimization, 12000)
    assert 4e-5 < midpoint_lr < 4e-4
    assert torch.isclose(torch.tensor(_compute_learning_rate(optimization, 16000)), torch.tensor(4e-5))


def test_evaluate_split_reports_token_role_metrics(tmp_path: Path, benchmark_config_path: Path) -> None:
    benchmark_dir = generate_symbolic_kv_stream_benchmark(benchmark_config_path)
    dataset = SymbolicKVDataset(benchmark_dir, "validation_iid")
    vocab = Vocabulary.from_metadata(dataset.metadata["vocabulary"])
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda batch: collate_symbolic_kv(batch, vocab.pad_token_id),
    )
    model = DecoderOnlyTransformer(
        spec=type("Spec", (), {
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 2,
            "d_ff": 64,
            "dropout": 0.0,
            "max_seq_len": 96,
        })(),
        vocab_size=len(vocab.tokens),
    )

    metrics = evaluate_split(
        model=model,
        data_loader=loader,
        device=torch.device("cpu"),
        pad_token_id=vocab.pad_token_id,
        value_token_ids=vocab.value_token_ids,
        max_batches=1,
        include_analysis=False,
    )

    assert "token_role_accuracy" in metrics
    assert "token_role_fraction" in metrics
    assert "write_value_accuracy" in metrics
    assert set(metrics["token_role_accuracy"]) == {
        "eos",
        "key_read",
        "key_write",
        "op_read",
        "op_write",
        "value_answer",
        "value_write",
    }
    assert abs(sum(metrics["token_role_fraction"].values()) - 1.0) < 1e-6


def test_rank_reference_candidates_prefers_heldout_then_validation() -> None:
    ranked = rank_reference_candidates(
        [
            {
                "run_name": "candidate_a",
                "model_parameter_count": 100,
                "metrics": {
                    "validation_iid": {"answer_accuracy": 0.96},
                    "test_iid": {"answer_accuracy": 0.95},
                    "heldout_pairs": {"answer_accuracy": 0.85},
                    "structural_ood": {"answer_accuracy": 0.50},
                    "counterfactual": {"answer_accuracy": 0.95},
                },
            },
            {
                "run_name": "candidate_b",
                "model_parameter_count": 100,
                "metrics": {
                    "validation_iid": {"answer_accuracy": 0.95},
                    "test_iid": {"answer_accuracy": 0.95},
                    "heldout_pairs": {"answer_accuracy": 0.87},
                    "structural_ood": {"answer_accuracy": 0.49},
                    "counterfactual": {"answer_accuracy": 0.95},
                },
            },
        ],
        min_validation_answer_accuracy=0.9,
    )

    assert [candidate["run_name"] for candidate in ranked] == ["candidate_b", "candidate_a"]


def test_legacy_feedforward_checkpoint_keys_migrate() -> None:
    spec = type("Spec", (), {
        "d_model": 32,
        "n_layers": 2,
        "n_heads": 2,
        "d_ff": 64,
        "dropout": 0.0,
        "max_seq_len": 96,
    })()
    source_model = DecoderOnlyTransformer(spec=spec, vocab_size=32)
    target_model = DecoderOnlyTransformer(spec=spec, vocab_size=32)

    legacy_state: dict[str, torch.Tensor] = {}
    for key, value in source_model.state_dict().items():
        legacy_key = key
        legacy_key = legacy_key.replace(".ff.fc_in.", ".ff.net.0.")
        legacy_key = legacy_key.replace(".ff.fc_out.", ".ff.net.2.")
        legacy_state[legacy_key] = value.clone()

    load_model_state(target_model, legacy_state)

    for key, value in source_model.state_dict().items():
        assert torch.equal(value, target_model.state_dict()[key])
