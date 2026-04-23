---
layout: default
title: "Analysis CLI Guide"
description: Practical guide for the formation-analysis CLI in the symbolic KV circuit repo.
---

# Analysis CLI Guide

This file is a practical operator guide for the analysis tools used in the symbolic KV circuit work.

It is not a paper and it is not a replacement for `src/circuit/cli.py`. The goal is simpler:

- tell you which command to run for which research question
- give one working command shape per tool
- tell you which outputs matter
- document the failure modes that actually happened in this repo

The tools are intentionally strict. They do not hide mismatches. If inputs disagree, they should fail.

## Base Paths

Most commands in this guide use the reference run:

```bash
RUN=artifacts/runs/symbolic_kv_reference_formation
CONFIG=$RUN/run_config.json
PROBE=$RUN/analysis/probe_set.jsonl
TRAIN_PROBE=$RUN/analysis/probe_set_train.jsonl
CKPT_DIR=$RUN/checkpoints
ANALYSIS=$RUN/analysis
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli ...
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
| Why did the route grow? | `bilinear-qk-rank-adam-state-attribution` | Decomposes actual update into raw SGD, clipped SGD, Adam current, momentum, weight decay |
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

### I want to know whether this is seed-specific

Run:

- `scripts/cross_seed_adam_pipeline.py`

## Command Reference

The examples below are the canonical shapes used in this repo. Replace only the parts that actually need changing.

### 1. Trained-model geometry

#### `attention-geometry-trace`

Use this first when you want a checkpoint timeline for attention/readout geometry.

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli attention-geometry-trace \
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
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli path-logit-decomposition \
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
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli route-competition-report \
  --config $CONFIG \
  --probe-set $PROBE \
  --train-probe-set $TRAIN_PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_005000.pt \
  --checkpoint $CKPT_DIR/step_005250.pt \
  --output-dir $ANALYSIS/route_competition/query_key_routes_5000_5250 \
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli weight-svd-trace \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli weight-svd-patterns \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli contextual-key-separability \
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
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli contextual-svd-alignment \
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
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli bilinear-qk-match-separation \
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
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli checkpoint-update-attribution \
  --config $CONFIG \
  --probe-set $PROBE \
  --checkpoint-dir $CKPT_DIR \
  --checkpoint $CKPT_DIR/step_005000.pt \
  --checkpoint $CKPT_DIR/step_005250.pt \
  --output-dir $ANALYSIS/checkpoint_update_attribution/l2h1_qk_query_rank4_5000_5250_top40 \
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli bilinear-qk-rank-update-attribution \
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
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli attention-retrieval-separation-update-attribution \
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
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli attention-retrieval-chain-report \
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
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli optimizer-update-trace \
  --config $CONFIG \
  --from-initialization \
  --output-dir $ANALYSIS/optimizer_update_trace/from_init_seed7_0000_6000_stepwise \
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli optimizer-update-trace \
  --config $CONFIG \
  --resume-checkpoint $CKPT_DIR/step_005500.pt \
  --output-dir $ANALYSIS/optimizer_update_trace/l2h1_qk_rank_0550_0750_stepwise \
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli bilinear-qk-rank-actual-batch-attribution \
  --config $CONFIG \
  --probe-set $PROBE \
  --optimizer-trace-dir $ANALYSIS/optimizer_update_trace/l2h1_qk_rank_0750_1000_stepwise \
  --output-dir $ANALYSIS/bilinear_qk_rank_actual_batch_attribution/l2h1_rank8_support_value_0750_1000_stepwise \
  --device mps \
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
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli bilinear-qk-rank-adam-state-attribution \
  --config $CONFIG \
  --probe-set $PROBE \
  --optimizer-trace-dir $ANALYSIS/optimizer_update_trace/from_init_seed7_0000_6000_stepwise \
  --output-dir $ANALYSIS/bilinear_qk_rank_adam_state_attribution/from_init_l2h1_rank8_support_value_0000_6000_stepwise \
  --device mps \
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
How much came from raw SGD?
How much came from Adam current gradient?
How much came from momentum?
How much came from weight decay?
```

### 9. Output-side validation

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

### 10. Cross-seed pipeline

#### `scripts/cross_seed_adam_pipeline.py`

This is the supported driver for:

1. preparing seed configs
2. traced scan checkpoints
3. head scan with `bilinear-qk-match-separation`
4. winner selection
5. exact Adam-state attribution for winner / controls

Example:

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python scripts/cross_seed_adam_pipeline.py \
  --base-config $CONFIG \
  --probe-set $PROBE \
  --run-root $CROSS_ROOT \
  --seed 11 \
  --seed 13 \
  --seed 17 \
  --seed 23 \
  --seed 29 \
  --python /opt/miniconda3/envs/ml/bin/python \
  --device mps \
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
