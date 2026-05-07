---
layout: default
title: "Analysis CLI Guide"
description: Practical guide for the circuit-formation analysis CLI: QK/OV analysis, residual tracing, optimizer attribution, and cross-seed comparison.
---

# Analysis CLI Guide

This file is a practical operator guide for the analysis tools used in the symbolic KV circuit work and for extending the toolkit carefully.

It is not a paper and it is not a replacement for `src/circuit/cli.py`. The goal is simpler:

- tell you which command to run for which research question
- give one working command shape per tool
- tell you which outputs matter
- document the failure modes that actually happened in this repo

The tools are intentionally strict. They do not hide mismatches. If inputs disagree, they should fail.

The current command examples target the repo's symbolic KV model. To use the toolkit on other open-weight models, add explicit adapters for model loading, tokenizer/task construction, module names, activation hook points, checkpoint format, and optimizer-state traces. QK/OV analysis and residual tracing can generalize through those adapters; optimizer-update attribution requires actual optimizer states or a replayable training trace.

## Base Paths

Most commands in this guide use the reference run:

```bash
export CIRCUIT_PYTHON="${CIRCUIT_PYTHON:-python}"
export CIRCUIT_DEVICE="${CIRCUIT_DEVICE:-cpu}"
export CIRCUIT="PYTHONPATH=src $CIRCUIT_PYTHON -m circuit.cli"

RUN=artifacts/runs/symbolic_kv_reference_formation
CONFIG=$RUN/run_config.json
PROBE=$RUN/analysis/probe_set.jsonl
TRAIN_PROBE=$RUN/analysis/probe_set_train.jsonl
CKPT_DIR=$RUN/checkpoints
ANALYSIS=$RUN/analysis
```

For cross-seed work:

```bash
CROSS_ROOT=artifacts/runs/symbolic_kv_cross_seed_adam
```

## Research Workflow

Use the tools in this order.

| Question | Main tool | What it answers |
| --- | --- | --- |
| Which trained mechanism matters? | `attention-geometry-trace`, `path-logit-decomposition`, `route-competition-report` | Finds the candidate route and downstream path |
| Does a QK route form in weight space? | `weight-svd-trace`, `weight-svd-patterns` | Shows singular growth, effective-rank drop, vector stabilization |
| Does the route become semantic? | `contextual-key-separability`, `contextual-svd-alignment`, `bilinear-qk-match-separation` | Tests whether the route aligns with contextual residual structure and support-vs-distractor separation |
| Did checkpoint updates grow the route? | `checkpoint-update-attribution`, `bilinear-qk-rank-update-attribution`, `attention-retrieval-separation-update-attribution` | First-order route-growth attribution across checkpoints |
| Did the actual training batch grow the route? | `optimizer-update-trace`, `bilinear-qk-rank-actual-batch-attribution`, `actual-batch-route-attribution` | Uses exact traced batches and parameter updates |
| Why did the QK route grow? | `bilinear-qk-rank-adam-state-attribution` | Decomposes actual update into raw SGD-equivalent, clipped SGD-equivalent, Adam current, momentum, weight decay for a rank-limited QK matcher |
| Why did the OV/write scalar grow? | `attention-downstream-adam-state-attribution` | Decomposes actual update into AdamW pieces and splits pressure over the traced head's `W_Q`, `W_K`, `W_V`, and `W_O` slices |
| What value code does the readout use? | `value-code-subspace-report`, `geometry-subspace-intervention` | Tracks prediction-position value identity and tests whether removing/keeping that subspace changes behavior |
| Does support value-code predict prediction value-code? | `value-code-transfer-map-report` | Fits and controls a support-to-prediction value-code transfer map |
| Can that transfer causally replace the prediction value-code? | `value-code-transfer-rescue` | Removes the target value-code component and patches back the fitted transfer or controls |
| Does the same role repeat across seeds? | `scripts/cross_seed_adam_pipeline.py` | Winner / runner-up / bottom-control comparison across seeds |

## Minimal Decision Tree

If you only remember one section, use this one.

### I want to know where the route forms

Run:

- `weight-svd-trace`
- then `weight-svd-patterns`
- then `bilinear-qk-match-separation`

### I want to know whether the route became task-meaningful

Run:

- `contextual-key-separability`
- `contextual-svd-alignment`
- `bilinear-qk-match-separation`

### I want to know whether optimizer updates selected that geometry

Run:

- `optimizer-update-trace`
- `bilinear-qk-rank-actual-batch-attribution`
- `bilinear-qk-rank-adam-state-attribution`

### I want to know whether optimizer updates built the write side

Run:

- `ov-write-progress-report`
- `optimizer-update-trace`
- `attention-downstream-update-attribution`
- `attention-downstream-adam-state-attribution`

### I want to know what the write side writes into readout

Run:

- `value-code-subspace-report`
- `value-code-transfer-map-report`
- `value-code-transfer-rescue`
- `geometry-subspace-intervention` with `--subspace embedding_value_identity`
- a rank-matched `embedding_key_identity` control

### I want to know whether this is seed-specific

Run:

- `scripts/cross_seed_adam_pipeline.py`

## Command Reference

The examples below are the canonical shapes used in this repo. Replace only the parts that actually need changing.

### 1. Trained-model geometry

#### `attention-geometry-trace`

Use this first when you want a checkpoint timeline for attention/readout geometry.

```bash
$CIRCUIT attention-geometry-trace \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_005000.pt \
  --checkpoint $CKPT_DIR/step_005250.pt \
  --checkpoint $CKPT_DIR/step_007500.pt \
  --checkpoint $CKPT_DIR/step_007750.pt \
  --checkpoint $CKPT_DIR/step_008000.pt \
  --checkpoint $CKPT_DIR/step_008250.pt \
  --output-dir $ANALYSIS/attention_geometry/l2h1_value_write_timeline \
  --device "$CIRCUIT_DEVICE" \
  --top-k-tokens 8 \
  --top-k-plot-heads 12
```

Important outputs:

- `report`
- `markdown`
- `rows`
- `plots.checkpoint_summary`
- `plots.role_attention`

Use it for:

- answer margin trajectory
- answer accuracy trajectory
- role-level attention and value-alignment summaries

#### `path-logit-decomposition`

Use this when you want direct-logit attribution and ablation-vs-DLA comparisons.

```bash
$CIRCUIT path-logit-decomposition \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_005000.pt \
  --checkpoint $CKPT_DIR/step_005250.pt \
  --checkpoint $CKPT_DIR/step_007500.pt \
  --checkpoint $CKPT_DIR/step_007750.pt \
  --checkpoint $CKPT_DIR/step_008000.pt \
  --checkpoint $CKPT_DIR/step_008250.pt \
  --output-dir $ANALYSIS/path_logit_decomposition/l2h1_value_write_timeline \
  --device "$CIRCUIT_DEVICE" \
  --ablation-top-k 8 \
  --ablation-step 5250 \
  --ablation-step 8000 \
  --top-k-plot-components 16
```

Important outputs:

- `report`
- `markdown`
- `plots.component_trajectory`
- `plots.stage_readout`
- `plots.ablation_vs_dla`

### 2. Route competition

#### `route-competition-report`

Use this to compare candidate routes in a common evaluation frame.

```bash
$CIRCUIT route-competition-report \
  --config $CONFIG \
  --probe-set $PROBE \
  --train-probe-set $TRAIN_PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_005000.pt \
  --checkpoint $CKPT_DIR/step_005250.pt \
  --output-dir $ANALYSIS/route_competition/query_key_routes_5000_5250 \
  --device "$CIRCUIT_DEVICE" \
  --route 'label=L2H1_qk_query,stage=layer_1_post_mlp,subspace=head_qk_query,rank=4,head_layer=2,head=1,position_role=query_key' \
  --route 'label=L1H2_qk_query,stage=layer_0_post_mlp,subspace=head_qk_query,rank=4,head_layer=1,head=2,position_role=query_key' \
  --route 'label=L0H0_qk_query,stage=embedding,subspace=head_qk_query,rank=4,head_layer=0,head=0,position_role=query_key' \
  --route-pair-type query_key \
  --pair-type query_key \
  --pair-type distractor \
  --train-pair-type query_key \
  --data-group-field pair_type \
  --eval-split validation_iid \
  --train-split train \
  --eval-loss-side both \
  --train-loss-side clean \
  --max-pairs-per-type 64 \
  --min-pairs-per-type 16
```

Important outputs:

- `route_rows`
- `data_rows`
- `pair_rows`
- `plots.train_support`
- `plots.eval_actual_delta`

Use it for:

- candidate ranking
- winner / runner-up / bottom controls
- cross-seed candidate selection

### 3. Weight-space formation

#### `weight-svd-trace`

Use this when you want raw SVD trajectories for `W_Q`, `W_K`, `W_V`, `W_O`, `W_QK`, `W_OV`, `W_in`, `W_out`.

```bash
$CIRCUIT weight-svd-trace \
  --config $CONFIG \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_000250.pt \
  --checkpoint $CKPT_DIR/step_000500.pt \
  --checkpoint $CKPT_DIR/step_000750.pt \
  --checkpoint $CKPT_DIR/step_001000.pt \
  --checkpoint $CKPT_DIR/step_001500.pt \
  --checkpoint $CKPT_DIR/step_002000.pt \
  --checkpoint $CKPT_DIR/step_002500.pt \
  --checkpoint $CKPT_DIR/step_003000.pt \
  --checkpoint $CKPT_DIR/step_003500.pt \
  --checkpoint $CKPT_DIR/step_004000.pt \
  --checkpoint $CKPT_DIR/step_004500.pt \
  --checkpoint $CKPT_DIR/step_005000.pt \
  --checkpoint $CKPT_DIR/step_005250.pt \
  --checkpoint $CKPT_DIR/step_005500.pt \
  --output-dir $ANALYSIS/weight_svd_trace/phase1_000250_5500_top16 \
  --device cpu \
  --max-singular-values 16 \
  --top-vector-ranks 16 \
  --overwrite
```

Important outputs:

- `singular_values_jsonl`
- `singular_values_csv`
- `top_singular_vectors_jsonl`

What to inspect:

- top singular value growth
- effective rank
- spectral mass concentration
- singular-vector rotation/stabilization

#### `weight-svd-patterns`

Use this after `weight-svd-trace` to summarize births, stabilization windows, and coordination windows.

```bash
$CIRCUIT weight-svd-patterns \
  --singular-values $ANALYSIS/weight_svd_trace/phase1_000250_5500_top16/weight_svd_singular_values.jsonl \
  --top-singular-vectors $ANALYSIS/weight_svd_trace/phase1_000250_5500_top16/weight_svd_top_singular_vectors.jsonl \
  --output-dir $ANALYSIS/weight_svd_patterns/phase1_000250_5500_top16 \
  --max-vector-rank 16 \
  --markdown-top-k 24 \
  --overwrite
```

Important outputs:

- `matrix_summary_rows`
- `vector_alignment_rows`
- `interval_event_rows`
- `coordination_window_rows`

### 4. Contextual semanticity

#### `contextual-key-separability`

Use this to test whether contextual residual states are separating the relevant key groups.

```bash
$CIRCUIT contextual-key-separability \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_000250.pt \
  --checkpoint $CKPT_DIR/step_000500.pt \
  --checkpoint $CKPT_DIR/step_000750.pt \
  --checkpoint $CKPT_DIR/step_001000.pt \
  --checkpoint $CKPT_DIR/step_001500.pt \
  --checkpoint $CKPT_DIR/step_002000.pt \
  --checkpoint $CKPT_DIR/step_002500.pt \
  --checkpoint $CKPT_DIR/step_003000.pt \
  --checkpoint $CKPT_DIR/step_003500.pt \
  --checkpoint $CKPT_DIR/step_004000.pt \
  --checkpoint $CKPT_DIR/step_004500.pt \
  --checkpoint $CKPT_DIR/step_005000.pt \
  --checkpoint $CKPT_DIR/step_005500.pt \
  --output-dir $ANALYSIS/contextual_key_separability/l2h1_prediction_query_key_stage_sweep_000250_005500 \
  --device "$CIRCUIT_DEVICE" \
  --head-layer 2 \
  --head 1 \
  --context-stage layer_1_post_mlp \
  --context-role prediction \
  --group-by query_key \
  --projection-rank 8 \
  --split validation_iid \
  --window-start 750 \
  --window-end 3500 \
  --include-full-residual \
  --overwrite
```

Important outputs:

- `metric_rows`
- `metric_csv`
- `group_rows`
- `plots.trajectory`

Use it for:

- pairwise key separability
- stage comparison
- windowed semanticity checks

#### `contextual-svd-alignment`

Use this when you want to compare singular directions against contextual residual subspaces.

```bash
$CIRCUIT contextual-svd-alignment \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_000250.pt \
  --checkpoint $CKPT_DIR/step_000500.pt \
  --checkpoint $CKPT_DIR/step_000750.pt \
  --checkpoint $CKPT_DIR/step_001000.pt \
  --checkpoint $CKPT_DIR/step_001500.pt \
  --checkpoint $CKPT_DIR/step_002000.pt \
  --checkpoint $CKPT_DIR/step_002500.pt \
  --checkpoint $CKPT_DIR/step_003000.pt \
  --checkpoint $CKPT_DIR/step_003500.pt \
  --checkpoint $CKPT_DIR/step_004000.pt \
  --checkpoint $CKPT_DIR/step_004500.pt \
  --checkpoint $CKPT_DIR/step_005000.pt \
  --checkpoint $CKPT_DIR/step_005500.pt \
  --output-dir $ANALYSIS/contextual_svd_alignment/l2h1_prediction_grouped_by_query_key_layer1_post_mlp_000250_005500 \
  --device "$CIRCUIT_DEVICE" \
  --head-layer 2 \
  --head 1 \
  --context-stage layer_1_post_mlp \
  --role prediction \
  --role-spec support_value:position_role=support_value \
  --plot-left-role prediction \
  --plot-right-role support_value \
  --top-ranks 4 \
  --pca-rank 4 \
  --batch-size 16 \
  --split validation_iid \
  --overwrite
```

Important outputs:

- `alignment_rows`
- `rank_aggregate_rows`
- `subspace_rows`
- `role_vector_rows`

### 5. QK route measurements

#### `bilinear-qk-match-separation`

Use this to define and track support-vs-distractor QK route quality directly.

```bash
$CIRCUIT bilinear-qk-match-separation \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_000250.pt \
  --checkpoint $CKPT_DIR/step_000500.pt \
  --checkpoint $CKPT_DIR/step_000750.pt \
  --checkpoint $CKPT_DIR/step_001000.pt \
  --checkpoint $CKPT_DIR/step_001500.pt \
  --checkpoint $CKPT_DIR/step_002000.pt \
  --checkpoint $CKPT_DIR/step_002500.pt \
  --checkpoint $CKPT_DIR/step_003000.pt \
  --checkpoint $CKPT_DIR/step_003500.pt \
  --checkpoint $CKPT_DIR/step_004000.pt \
  --checkpoint $CKPT_DIR/step_004500.pt \
  --checkpoint $CKPT_DIR/step_005000.pt \
  --checkpoint $CKPT_DIR/step_005500.pt \
  --output-dir $ANALYSIS/bilinear_qk_match_separation/l2h1_support_value_vs_distractors_000250_005500_stage_sweep \
  --device "$CIRCUIT_DEVICE" \
  --head-layer 2 \
  --head 1 \
  --context-stage layer_1_post_mlp \
  --score-query-role prediction \
  --support-role support_value \
  --distractor-role value_distractors \
  --layernorm-mode head_ln1 \
  --rank 8 \
  --group-by query_key \
  --split validation_iid \
  --window-start 750 \
  --window-end 3500 \
  --overwrite
```

Important outputs:

- `metric_rows`
- `metric_csv`
- `event_rows`
- `group_rows`

Key fields:

- `qk_match_separation_mean`
- `support_beats_all_rate`
- `answer_margin_mean`
- `qk_singular_value_top`

### 6. Checkpoint-to-checkpoint first-order attribution

#### `checkpoint-update-attribution`

Use this for generic route/subspace update attribution between checkpoints.

```bash
$CIRCUIT checkpoint-update-attribution \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_005000.pt \
  --checkpoint $CKPT_DIR/step_005250.pt \
  --output-dir $ANALYSIS/checkpoint_update_attribution/l2h1_qk_query_rank4_5000_5250_top40 \
  --device "$CIRCUIT_DEVICE" \
  --stage layer_1_post_mlp \
  --subspace head_qk_query \
  --rank 4 \
  --head-layer 2 \
  --head 1 \
  --position-role query_key \
  --pair-type query_key \
  --pair-type distractor \
  --max-pairs-per-type 64 \
  --min-pairs-per-type 16 \
  --decompose module_blocks \
  --decompose attention_heads \
  --decompose attention_projections \
  --decompose mlp_neurons \
  --top-k-groups 40
```

Important outputs:

- `metric_rows`
- `decomposition_rows`
- `group_rows`
- `pair_rows`

#### `bilinear-qk-rank-update-attribution`

Use this when the object of interest is a bilinear QK rank, not a generic residual subspace.

```bash
$CIRCUIT bilinear-qk-rank-update-attribution \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_000750.pt \
  --checkpoint $CKPT_DIR/step_001000.pt \
  --checkpoint $CKPT_DIR/step_001250.pt \
  --checkpoint $CKPT_DIR/step_001500.pt \
  --checkpoint $CKPT_DIR/step_002000.pt \
  --checkpoint $CKPT_DIR/step_002500.pt \
  --checkpoint $CKPT_DIR/step_003000.pt \
  --checkpoint $CKPT_DIR/step_003500.pt \
  --output-dir $ANALYSIS/bilinear_qk_rank_update_attribution/l2h1_rank4_rank8_support_value_minus_distractors_000750_003500_formation \
  --device "$CIRCUIT_DEVICE" \
  --head-layer 2 \
  --head 1 \
  --rank 4 \
  --rank 8 \
  --context-stage layer_1_post_mlp \
  --layernorm-mode head_ln1 \
  --score-query-role prediction \
  --support-key-role support_value \
  --distractor-key-role value_distractors \
  --record-side clean \
  --pair-type support_value \
  --pair-type distractor \
  --split validation_iid \
  --decompose module_blocks \
  --decompose attention_heads \
  --decompose attention_projections \
  --decompose mlp_neurons \
  --top-k-groups 40
```

Important outputs:

- `metric_rows`
- `decomposition_rows`
- `group_rows`
- `score_rows`
- `pair_rows`

### 7. Stepwise route behavior

#### `attention-retrieval-separation-update-attribution`

Use this for stepwise support-vs-distractor attention separation.

```bash
$CIRCUIT attention-retrieval-separation-update-attribution \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_000750.pt \
  --checkpoint $CKPT_DIR/step_001000.pt \
  --checkpoint $CKPT_DIR/step_001250.pt \
  --checkpoint $CKPT_DIR/step_001500.pt \
  --checkpoint $CKPT_DIR/step_002000.pt \
  --checkpoint $CKPT_DIR/step_002500.pt \
  --checkpoint $CKPT_DIR/step_003000.pt \
  --checkpoint $CKPT_DIR/step_003500.pt \
  --output-dir $ANALYSIS/attention_retrieval_separation_update_attribution/l2h1_support_value_minus_distractors_000750_003500_formation \
  --device "$CIRCUIT_DEVICE" \
  --head-layer 2 \
  --head 1 \
  --score-query-role prediction \
  --support-key-role support_value \
  --distractor-key-role value_distractors \
  --record-side clean \
  --score-component score \
  --pair-type support_value \
  --pair-type distractor \
  --split validation_iid \
  --decompose module_blocks \
  --decompose attention_heads \
  --decompose attention_projections \
  --decompose mlp_neurons \
  --top-k-groups 40
```

Important outputs:

- `metric_rows`
- `decomposition_rows`
- `group_rows`
- `score_rows`
- `pair_rows`

#### `attention-retrieval-chain-report`

Use this to get the checkpoint-level chain summary for one head.

```bash
$CIRCUIT attention-retrieval-chain-report \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_005500.pt \
  --checkpoint $CKPT_DIR/step_005750.pt \
  --checkpoint $CKPT_DIR/step_006000.pt \
  --checkpoint $CKPT_DIR/step_006250.pt \
  --checkpoint $CKPT_DIR/step_006500.pt \
  --checkpoint $CKPT_DIR/step_006750.pt \
  --checkpoint $CKPT_DIR/step_007000.pt \
  --checkpoint $CKPT_DIR/step_007250.pt \
  --checkpoint $CKPT_DIR/step_007500.pt \
  --output-dir $ANALYSIS/attention_retrieval_chain/l2h1_support_value_minus_distractors_5500_7500_neighbor_intervals \
  --device "$CIRCUIT_DEVICE" \
  --head-layer 2 \
  --head 1 \
  --score-query-role prediction \
  --support-key-role support_value \
  --distractor-key-role value_distractors \
  --record-side clean \
  --pair-type support_value \
  --pair-type distractor
```

Important outputs:

- `checkpoint_rows`
- `delta_rows`
- `pair_metric_rows`
- `plots.trajectory`

### 8. Exact traced training updates

#### `optimizer-update-trace`

Use this before any actual-batch or Adam-state attribution. This tool is the source of truth for traced batches, checkpoints, and parameter updates.

From initialization:

```bash
$CIRCUIT optimizer-update-trace \
  --config $CONFIG \
  --from-initialization \
  --output-dir $ANALYSIS/optimizer_update_trace/from_init_seed7_0000_6000_stepwise \
  --device "$CIRCUIT_DEVICE" \
  --end-step 6000 \
  --train-split train \
  --checkpoint-every 1 \
  --checkpoint-start-step 0 \
  --progress-every 100 \
  --top-k-parameters 40 \
  --overwrite
```

Resume from a checkpoint:

```bash
$CIRCUIT optimizer-update-trace \
  --config $CONFIG \
  --resume-checkpoint $CKPT_DIR/step_005500.pt \
  --output-dir $ANALYSIS/optimizer_update_trace/l2h1_qk_rank_0550_0750_stepwise \
  --device "$CIRCUIT_DEVICE" \
  --num-steps 2000 \
  --train-split train \
  --checkpoint-every 1 \
  --checkpoint-start-step 5500 \
  --progress-every 100 \
  --top-k-parameters 40 \
  --overwrite
```

Important outputs:

- `step_rows`
- `batch_rows`
- `parameter_update_rows`
- `checkpoints/`

Do not treat this as optional if you need exact update attribution.

#### `bilinear-qk-rank-actual-batch-attribution`

Use this to project actual traced batch updates onto a QK-rank route.

```bash
$CIRCUIT bilinear-qk-rank-actual-batch-attribution \
  --config $CONFIG \
  --probe-set $PROBE \
  --optimizer-trace-dir $ANALYSIS/optimizer_update_trace/l2h1_qk_rank_0750_1000_stepwise \
  --output-dir $ANALYSIS/bilinear_qk_rank_actual_batch_attribution/l2h1_rank8_support_value_0750_1000_stepwise \
  --device "$CIRCUIT_DEVICE" \
  --head-layer 2 \
  --head 1 \
  --rank 8 \
  --context-stage layer_1_post_mlp \
  --layernorm-mode head_ln1 \
  --score-query-role prediction \
  --support-key-role support_value \
  --distractor-key-role value_distractors \
  --record-side clean \
  --route-pair-type support_value \
  --route-pair-source-type support_value \
  --max-route-pairs-per-type 64 \
  --min-route-pairs-per-type 16 \
  --loss-scope full_lm \
  --overwrite
```

Important outputs:

- `route_rows`
- `actual_batch_rows`
- `route_pair_rows`

#### `bilinear-qk-rank-adam-state-attribution`

Use this for the optimizer-level “why” question.

```bash
$CIRCUIT bilinear-qk-rank-adam-state-attribution \
  --config $CONFIG \
  --probe-set $PROBE \
  --optimizer-trace-dir $ANALYSIS/optimizer_update_trace/from_init_seed7_0000_6000_stepwise \
  --output-dir $ANALYSIS/bilinear_qk_rank_adam_state_attribution/from_init_l2h1_rank8_support_value_0000_6000_stepwise \
  --device "$CIRCUIT_DEVICE" \
  --head-layer 2 \
  --head 1 \
  --rank 8 \
  --context-stage layer_1_post_mlp \
  --layernorm-mode head_ln1 \
  --score-query-role prediction \
  --support-key-role support_value \
  --distractor-key-role value_distractors \
  --record-side clean \
  --route-pair-type support_value \
  --route-pair-source-type support_value \
  --max-route-pairs-per-type 64 \
  --min-route-pairs-per-type 16 \
  --loss-scope full_lm \
  --overwrite
```

Important outputs:

- `metric_rows`
- `component_rows`
- `route_pair_rows`

This is the command that answers:

```text
How much came from the raw SGD-equivalent update?
How much came from Adam current gradient?
How much came from momentum?
How much came from weight decay?
```

#### `attention-downstream-adam-state-attribution`

Use this for the OV/write-side optimizer question. It uses the same exact optimizer trace as the QK Adam tool, but the target scalar is a downstream write quantity such as `qk_ov_product`, `support_mass_ov_value_margin`, `attended_support_ov_value_margin`, `head_value_margin_dla`, or `head_margin_dla_fixed_readout`.

The output has two levels:

- global AdamW decomposition for the scalar
- parameter-group decomposition over the traced head's `q_proj`, `k_proj`, `v_proj`, `out_proj`, and combined `qkvo` slices
- optional extra groups such as `module:L0.mlp`, `module:L1.attention`, or another head's projection via repeated `--parameter-group`

```bash
$CIRCUIT attention-downstream-adam-state-attribution \
  --config $CONFIG \
  --probe-set $PROBE \
  --optimizer-trace-dir $ANALYSIS/optimizer_update_trace/from_init_seed7_0000_6000_stepwise \
  --output-dir $ANALYSIS/attention_downstream_adam_state_attribution/l1h2_support_value_write_5500_5501_smoke \
  --device "$CIRCUIT_DEVICE" \
  --checkpoint $ANALYSIS/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/checkpoints/step_005500.pt \
  --checkpoint $ANALYSIS/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/checkpoints/step_005501.pt \
  --head-layer 1 \
  --head 2 \
  --score-query-role prediction \
  --support-key-role support_value \
  --distractor-key-role value_distractors \
  --record-side clean \
  --scalar qk_ov_product \
  --scalar support_mass_ov_value_margin \
  --scalar attended_support_ov_value_margin \
  --scalar head_value_margin_dla \
  --scalar head_margin_dla_fixed_readout \
  --objective-pair-type support_value \
  --route-pair-source-type support_value \
  --max-route-pairs-per-type 64 \
  --min-route-pairs-per-type 16 \
  --loss-scope full_lm \
  --overwrite
```

Important outputs:

- `metric_rows`
- `component_rows`
- `group_rows`
- `route_pair_rows`

This command answers:

```text
Did the actual AdamW update increase the write scalar?
Was the raw SGD-equivalent update tiny or large for that write scalar?
Did Adam current gradient or historical momentum carry the update?
Did the useful pressure land in W_V, W_O, QK slices, or outside the traced head?
```

The two audit-selected scalar forms are:

```text
support_mass_ov_value_margin = total support attention mass * OV value margin
qk_ov_product = QK support-minus-distractor score separation * OV value margin
```

#### `ov-write-progress-report`

Use this before OV/write optimizer attribution. It audits candidate heads and write scalars across checkpoints, with readout-level attention-frozen and shuffled-value controls.

This is the scalar-selection step. Do not skip it and jump straight to AdamW decomposition unless you already know which write scalar is meaningful.

```bash
$CIRCUIT ov-write-progress-report \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_000750.pt \
  --checkpoint $CKPT_DIR/step_001000.pt \
  --checkpoint $CKPT_DIR/step_001250.pt \
  --checkpoint $CKPT_DIR/step_001500.pt \
  --checkpoint $CKPT_DIR/step_001750.pt \
  --checkpoint $CKPT_DIR/step_002000.pt \
  --checkpoint $CKPT_DIR/step_002250.pt \
  --checkpoint $CKPT_DIR/step_002500.pt \
  --checkpoint $CKPT_DIR/step_002750.pt \
  --checkpoint $CKPT_DIR/step_003000.pt \
  --checkpoint $CKPT_DIR/step_003250.pt \
  --checkpoint $CKPT_DIR/step_003500.pt \
  --output-dir $ANALYSIS/ov_write_progress/l0_l1_l2_attention_0750_3500_formation \
  --device "$CIRCUIT_DEVICE" \
  --head L0H0 \
  --head L0H1 \
  --head L0H2 \
  --head L0H3 \
  --head L1H2 \
  --head L2H1 \
  --score-query-role prediction \
  --support-key-role support_value \
  --distractor-key-role value_distractors \
  --record-side clean \
  --pair-type support_value \
  --max-pairs-per-type 64 \
  --min-pairs-per-type 16 \
  --top-k-correlations 32 \
  --overwrite
```

Important outputs:

- `checkpoint_rows`
- `delta_rows`
- `correlation_rows`
- `pair_rows`

The four conditions are:

```text
real_attention_real_values
correct_support_attention_real_values
real_attention_shuffled_values
correct_support_attention_shuffled_values
```

This command answers:

```text
Which OV/write scalar has a clean birth curve?
Does forcing correct-support attention make the write useful?
Does shuffling the support value destroy the write signal?
Which write scalar best tracks fixed-competitor margin, correct-value logit, or negative loss?
```

### 9. Value-code readout

#### `value-code-subspace-report`

Use this after the write-side audit when the question is no longer "does the residual change matter?" but "what code does the mature readout use?"

This command tracks whether prediction-position residual states become separable by answer value, support value, or both.

```bash
TRACE_CKPTS=$ANALYSIS/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/checkpoints

$CIRCUIT value-code-subspace-report \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $TRACE_CKPTS \
  --checkpoint $TRACE_CKPTS/step_001500.pt \
  --checkpoint $TRACE_CKPTS/step_001750.pt \
  --checkpoint $TRACE_CKPTS/step_002000.pt \
  --checkpoint $TRACE_CKPTS/step_002500.pt \
  --checkpoint $TRACE_CKPTS/step_003000.pt \
  --checkpoint $TRACE_CKPTS/step_003500.pt \
  --output-dir $ANALYSIS/value_code_subspace/prediction_answer_value_1500_3500_cli \
  --device "$CIRCUIT_DEVICE" \
  --stage layer_0_post_mlp \
  --stage layer_1_post_mlp \
  --stage layer_2_post_mlp \
  --stage final_norm \
  --position-role prediction \
  --position-role support_value \
  --group-by answer_value \
  --group-by support_value \
  --split validation_iid \
  --max-records 256 \
  --pca-rank 4 \
  --overwrite
```

Important outputs:

- `value_code_rows`
- `summary_rows`
- `subspace_rows`

This command answers:

```text
When does the prediction residual start reading out the answer value?
Is the readable code grouped by answer value or just by support position?
Is the value-code object low-rank or broad?
```

#### `value-code-transfer-map-report`

Use this when the remaining question is the support-to-prediction bridge:

```text
support-value residual state -> prediction-position value-code state
```

The tool builds value-identity bases on a deterministic fit split, fits a ridge-stabilized affine map from source coordinates to target coordinates, and evaluates the map on heldout probe rows. Controls use the same heldout rows.

```bash
TRACE_CKPTS=$ANALYSIS/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/checkpoints

$CIRCUIT value-code-transfer-map-report \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $TRACE_CKPTS \
  --checkpoint $TRACE_CKPTS/step_001500.pt \
  --checkpoint $TRACE_CKPTS/step_001750.pt \
  --checkpoint $TRACE_CKPTS/step_002000.pt \
  --checkpoint $TRACE_CKPTS/step_002500.pt \
  --checkpoint $TRACE_CKPTS/step_003000.pt \
  --checkpoint $TRACE_CKPTS/step_003500.pt \
  --output-dir $ANALYSIS/value_code_transfer_map/support_to_prediction_1500_3500_cli \
  --device "$CIRCUIT_DEVICE" \
  --source-stage layer_1_post_mlp \
  --target-stage layer_2_post_mlp \
  --source-position-role support_value \
  --target-position-role prediction \
  --group-by answer_value \
  --split validation_iid \
  --max-records 256 \
  --basis-rank 4 \
  --basis-rank 8 \
  --basis-rank 16 \
  --control shuffled_answer_value \
  --control wrong_support_value \
  --control random_subspace \
  --fit-fraction 0.75 \
  --overwrite
```

Important outputs:

- `transfer_rows`
- `summary_rows`
- `subspace_rows`
- `pair_rows`

This command answers:

```text
Can source value-code coordinates predict prediction value-code coordinates?
Does the true transfer beat shuffled-source, wrong-support, and random-subspace controls?
Does the transferred code itself point toward the correct value under a stage lens?
```

The optional `key_identity` control fits a support-key-code map. It is rank-limited by the key-token identity rank, so do not combine it with high value-code ranks unless you expect the command to fail loudly. For the current 8-key task, run it separately with a small rank:

```bash
$CIRCUIT value-code-transfer-map-report \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $TRACE_CKPTS \
  --checkpoint $TRACE_CKPTS/step_002500.pt \
  --output-dir $ANALYSIS/value_code_transfer_map/support_to_prediction_key_control_rank4_cli \
  --device "$CIRCUIT_DEVICE" \
  --source-stage layer_1_post_mlp \
  --target-stage layer_2_post_mlp \
  --source-position-role support_value \
  --target-position-role prediction \
  --group-by answer_value \
  --split validation_iid \
  --max-records 256 \
  --basis-rank 4 \
  --control key_identity \
  --overwrite
```

#### `value-code-transfer-rescue`

Use this after `value-code-transfer-map-report` when you need causal sufficiency rather than coordinate prediction.

The tool removes the target value-code projection at the prediction position, then patches back either the actual projected value-code component, the fitted support-to-prediction transfer, or a control transfer.

```text
target_removed = clean_target - project_target_value_code(clean_target)
patched = target_removed + predicted_target_value_code(source)
rescue = scalar(patched) - scalar(target_removed)
```

The oracle row checks whether the removed target value-code component itself is causal. The true-transfer row checks whether the fitted transfer can replace it.
The output also includes `fixed_clean_competitor_margin` and `fixed_removed_competitor_margin`, which hold the wrong-token branch fixed so moving best-wrong switches cannot hide a successful transfer.

The optional context arguments test the next write-side hypothesis: the support value-code alone may not be enough, because the prediction-position residual state can choose how the support code is interpreted. Passing `--context-stage`, `--context-position-role`, and `--context-rank` adds `context_only`, `source_plus_context`, and rank-matched contextual control rows. Use this when you need to distinguish a static support-to-prediction transfer from a contextual write operator.

```bash
TRACE_CKPTS=$ANALYSIS/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/checkpoints

$CIRCUIT value-code-transfer-rescue \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $TRACE_CKPTS \
  --checkpoint $TRACE_CKPTS/step_001750.pt \
  --checkpoint $TRACE_CKPTS/step_002000.pt \
  --checkpoint $TRACE_CKPTS/step_002500.pt \
  --checkpoint $TRACE_CKPTS/step_003000.pt \
  --checkpoint $TRACE_CKPTS/step_003500.pt \
  --output-dir $ANALYSIS/value_code_transfer_rescue/support_to_prediction_rank16_1750_3500_cli \
  --device "$CIRCUIT_DEVICE" \
  --source-stage layer_1_post_mlp \
  --target-stage layer_2_post_mlp \
  --source-position-role support_value \
  --target-position-role prediction \
  --group-by answer_value \
  --split validation_iid \
  --max-records 256 \
  --basis-rank 16 \
  --control shuffled_answer_value \
  --control wrong_support_value \
  --control random_subspace \
  --fit-fraction 0.75 \
  --overwrite
```

Run the rank-limited key control separately:

```bash
$CIRCUIT value-code-transfer-rescue \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $TRACE_CKPTS \
  --checkpoint $TRACE_CKPTS/step_002500.pt \
  --output-dir $ANALYSIS/value_code_transfer_rescue/support_to_prediction_key_control_rank4_cli \
  --device "$CIRCUIT_DEVICE" \
  --source-stage layer_1_post_mlp \
  --target-stage layer_2_post_mlp \
  --source-position-role support_value \
  --target-position-role prediction \
  --group-by answer_value \
  --split validation_iid \
  --max-records 256 \
  --basis-rank 4 \
  --control key_identity \
  --fit-fraction 0.75 \
  --overwrite
```

Run the contextual transfer version when the source-only transfer rescues answer evidence but not the moving/fixed margin cleanly:

```bash
$CIRCUIT value-code-transfer-rescue \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $TRACE_CKPTS \
  --checkpoint $TRACE_CKPTS/step_001750.pt \
  --checkpoint $TRACE_CKPTS/step_002000.pt \
  --checkpoint $TRACE_CKPTS/step_002500.pt \
  --checkpoint $TRACE_CKPTS/step_003000.pt \
  --checkpoint $TRACE_CKPTS/step_003500.pt \
  --output-dir $ANALYSIS/value_code_transfer_rescue/support_to_prediction_context_rank16_1750_3500_cli \
  --device "$CIRCUIT_DEVICE" \
  --source-stage layer_1_post_mlp \
  --target-stage layer_2_post_mlp \
  --source-position-role support_value \
  --target-position-role prediction \
  --context-stage layer_1_post_mlp \
  --context-position-role prediction \
  --context-rank 16 \
  --group-by answer_value \
  --split validation_iid \
  --max-records 256 \
  --basis-rank 16 \
  --control shuffled_answer_value \
  --control wrong_support_value \
  --control random_subspace \
  --fit-fraction 0.75 \
  --overwrite
```

#### `geometry-subspace-intervention`

Use this to test whether a geometry subspace is causal, not merely readable.

For the value-code claim, remove value identity from `layer_2_post_mlp / prediction` and compare it against a rank-matched key-identity control.

```bash
TRACE_CKPTS=$ANALYSIS/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/checkpoints

$CIRCUIT geometry-subspace-intervention \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $TRACE_CKPTS \
  --checkpoint $TRACE_CKPTS/step_001500.pt \
  --checkpoint $TRACE_CKPTS/step_001750.pt \
  --checkpoint $TRACE_CKPTS/step_002000.pt \
  --checkpoint $TRACE_CKPTS/step_002500.pt \
  --checkpoint $TRACE_CKPTS/step_003000.pt \
  --checkpoint $TRACE_CKPTS/step_003500.pt \
  --output-dir $ANALYSIS/value_code_causal_intervention/embedding_value_identity_prediction_layer2_remove_rank16_1500_3500 \
  --device "$CIRCUIT_DEVICE" \
  --stage layer_2_post_mlp \
  --subspace embedding_value_identity \
  --rank 16 \
  --operation remove \
  --position-role prediction \
  --query-mode single_query
```

Important outputs:

- `aggregate_rows`
- `query_rows`
- `plots.margin_drop`
- `plots.accuracy_drop`

Use it for:

- value-identity removal: `--subspace embedding_value_identity --operation remove`
- rank-matched key control: `--subspace embedding_key_identity --rank 7 --operation remove`
- sufficiency checks: `--operation keep` at higher value-identity rank

### 10. Route-family closure

#### `route-family-closure-report`

Use this after `route-to-margin-closure` when the expensive run already measured all route deltas, but you need to compare families such as QK-only, OV-only, and QK+OV on the same observations.

This command does not recompute activations. It refits the existing closure rows with different route subsets.

```bash
$CIRCUIT route-family-closure-report \
  --route-closure-rows $ANALYSIS/route_to_margin_closure/qk_ov_output_routes_5500_5550_stepwise/route_to_margin_closure_rows.jsonl \
  --output-dir $ANALYSIS/route_family_closure/qk_vs_ov_vs_joint_5500_5550_stepwise \
  --family label=qk,route=L2H1_qk_query,route=L1H2_qk_query,route=L0H0_qk_query,route=embedding_key_identity,route=full_layer1_query_key,route=full_layer0_query_key \
  --family label=ov_input,route=L2H1_ov_input_support_value,route=L1H2_ov_input_support_value,route=L0H0_ov_input_support_value,route=embedding_value_identity,route=full_layer1_support_value,route=full_layer0_support_value \
  --family label=ov_output,route=L1H2_ov_output_prediction,route=L2H1_ov_output_prediction,route=full_layer1_post_attn_prediction,route=full_layer2_post_attn_prediction \
  --family label=qk_plus_ov,route=L2H1_qk_query,route=L1H2_qk_query,route=L0H0_qk_query,route=embedding_key_identity,route=full_layer1_query_key,route=full_layer0_query_key,route=L2H1_ov_input_support_value,route=L1H2_ov_input_support_value,route=L0H0_ov_input_support_value,route=embedding_value_identity,route=full_layer1_support_value,route=full_layer0_support_value,route=L1H2_ov_output_prediction,route=L2H1_ov_output_prediction,route=full_layer1_post_attn_prediction,route=full_layer2_post_attn_prediction \
  --target-scalar answer_margin \
  --record-side clean \
  --overwrite
```

Important outputs:

- `family_summary_rows`
- `interval_rows`
- `coefficient_rows`
- `plots.r_squared`
- `plots.abs_residual`

Use it for:

- whether OV routes add explanatory power beyond QK routes
- whether full QK+OV route-family closure improves answer-margin closure
- deciding whether the OV side is head-local or residual-family-local

### 11. Output-side validation

Use these after route-level closure, not before.

#### `output-route-closure`

Fits output-component routes against scalar rows.

Key outputs:

- `closure_rows`
- `endpoint_component_rows`
- `coefficient_rows`

#### `output-component-causal-validation`

Tests whether DLA-like component effects match causal interventions.

Key outputs:

- `validation_rows`
- `summary_rows`
- `plots.causal_vs_dla`

#### `output-mediated-causal-decomposition`

Tests whether one component’s effect is mediated through downstream components.

Key outputs:

- `source_rows`
- `downstream_rows`
- `source_summary_rows`
- `downstream_summary_rows`

#### `residual-state-rescue`

Tests whether patching residual state at later stages rescues a removed source component.

Key outputs:

- `rescue_rows`
- `summary_rows`
- `plots.rescue_fraction`

#### `component-output-rescue`

Tests whether patching one clean downstream component write rescues a removed source component.

This is stricter than `residual-state-rescue`: it does not patch the whole residual stream. For MLPs it replaces the MLP residual write. For attention heads it replaces the single-head residual contribution, computed by subtracting all-heads-off attention output from only-this-head attention output so the shared output bias cancels.

Pass `--patch-component L0MLP` for single-component rescue. Pass `--patch-group L0MLP,L2MLP` for ordered multi-component rescue; the tool patches components in model order and recomputes intermediate residual states between patch stages.

Key outputs:

- `rescue_rows`
- `summary_rows`
- `pair_rows`

### 12. Cross-seed pipeline

#### `scripts/cross_seed_adam_pipeline.py`

This is the supported driver for:

1. preparing seed configs
2. traced scan checkpoints
3. head scan with `bilinear-qk-match-separation`
4. winner selection
5. exact Adam-state attribution for winner / controls

Example:

```bash
PYTHONPATH=src $CIRCUIT_PYTHON scripts/cross_seed_adam_pipeline.py \
  --base-config $CONFIG \
  --probe-set $PROBE \
  --run-root $CROSS_ROOT \
  --seed 11 \
  --seed 13 \
  --seed 17 \
  --seed 23 \
  --seed 29 \
  --python "$CIRCUIT_PYTHON" \
  --device "$CIRCUIT_DEVICE" \
  --end-step 6000 \
  --layers 3 \
  --heads 4 \
  --rank 8 \
  --window-start 750 \
  --window-end 3500 \
  --adam-start-step 750 \
  --adam-end-step 2500 \
  --split validation_iid \
  --stage configs \
  --stage trace-scan \
  --stage scan \
  --stage select \
  --stage trace-adam \
  --stage adam \
  --adam-candidate winner \
  --adam-candidate runner_up \
  --adam-candidate bottom \
  --overwrite
```

Important outputs:

- `cross_seed_manifest.json`
- `cross_seed_winners.json`
- `cross_seed_winners.csv`
- per-seed `analysis/cross_seed_head_selection.json`
- per-seed scan reports
- per-seed Adam-state attribution reports

Use stages separately when disk is tight.

## Output Contract

Most analysis commands print a JSON object. Treat that JSON as the contract.

Typical fields:

- `report`: machine-readable JSON summary
- `markdown`: human-readable report
- `*_rows` or `*_csv`: row-level artifacts for scripting
- `plots`: SVG figure paths

Do not guess file names when the command already printed them.

## Common Failure Modes

These all happened in real use.

### `Output directory already exists and is non-empty`

Cause:

- rerunning into the same directory without `--overwrite`

Fix:

- add `--overwrite`, or
- choose a new output directory

### `Checkpoint directory not found: .../checkpoints`

Cause:

- using an `optimizer-update-trace` directory that was never produced
- running the cross-seed `adam` stage without the matching `trace-adam` stage

Fix:

- confirm `<trace_dir>/checkpoints` exists
- run `optimizer-update-trace` first

### `Probe-set file not found`

Cause:

- wrong probe-set path
- forgetting that train and validation probe sets are separate files

Fix:

- use `$PROBE` for validation-style analyses
- use `$TRAIN_PROBE` when the command really needs train examples

### `Failed to construct the requested minimum causal patch pairs`

Cause:

- split / pair-type / probe-set combination produced zero valid pairs

Fix:

- check `--pair-type`
- check `--split`
- check whether the probe set actually contains those examples
- lower `--min-pairs-per-type` only if that matches the experiment

### `Intervention positions must be in the causal prefix`

Cause:

- invalid query/key role combination for the chosen attention-score experiment

Fix:

- make sure the key role is available before the query position
- do not use future positions for causal interventions

### `Scalar recomputation mismatch`

Cause:

- scalar rows and output-route closure were built from inconsistent pair sets or tolerances

Fix:

- regenerate the scalar rows and closure from the same pair universe
- keep the same margin-side / pair-type / split filters

### `Data group values changed across intervals`

Cause:

- a grouped attribution summary assumed stable group IDs across intervals, but the actual grouping changed

Fix:

- avoid aggregating that run as one summary
- rerun on a smaller interval window or with a stable grouping field

### `Optimizer param-group lr mismatch` or `Recomputed gradient norm mismatch`

Cause:

- the optimizer trace and the attribution command do not correspond to the same exact run / trace / replay assumptions

Fix:

- use the exact trace generated for that run
- do not mix traces from different seeds or different replay modes
- check scheduler / LR state consistency

### `No space left on device`

Cause:

- stepwise optimizer traces are large

Fix:

- run fewer stages at once
- use `--scan-checkpoint-every 250` for scans
- clean old trace directories
- use `--cleanup-adam-trace` in the cross-seed driver if you only need the final Adam reports

## Recommended Paper Reproduction Sequence

For the current paper result:

1. `attention-geometry-trace`
2. `path-logit-decomposition`
3. `weight-svd-trace`
4. `weight-svd-patterns`
5. `contextual-key-separability`
6. `bilinear-qk-match-separation`
7. `bilinear-qk-rank-update-attribution`
8. `optimizer-update-trace`
9. `bilinear-qk-rank-actual-batch-attribution`
10. `bilinear-qk-rank-adam-state-attribution`
11. `scripts/cross_seed_adam_pipeline.py`

## What This Guide Does Not Cover

This guide does not try to document every exploratory command in `src/circuit/cli.py`.

It focuses on the formation-analysis stack that was actually used in the current paper:

- trained-model route discovery
- weight-space birth
- contextual semanticity
- exact update attribution
- Adam-state decomposition
- cross-seed role validation

For older feature-family and candidate-registry commands, read:

- [src/circuit/cli.py](/Users/nelson/py/circuit/src/circuit/cli.py)
- [shared_feature_dynamics_plan.md](/Users/nelson/py/circuit/docs/shared_feature_dynamics_plan.md)
