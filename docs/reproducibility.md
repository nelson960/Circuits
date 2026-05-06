---
layout: default
title: "Reproducibility"
description: Environment, data, model, training, and command entry points for reproducing the symbolic KV circuit formation results.
---

# Reproducibility

This page is the reproducibility contract for the paper. It gives the environment, task, model, training run, and command entry points. The detailed tool manual is [Analysis CLI Guide](analysis_cli_guide.md).

The repository does not assume that analysis runs are uploaded. Reproduction means regenerating the benchmark, training or replaying the relevant runs, and producing the expected artifact directories locally.

## Environment

The reference artifacts in the current draft were produced on:

| item | value |
| --- | --- |
| OS | macOS 26.4.1 |
| hardware | MacBook Pro, Apple M2 Pro, 12 CPU cores, 16 GB memory |
| Python | 3.12.2 |
| main device | `mps` |
| torch observed | 2.9.1 |
| numpy observed | 2.4.2 |
| matplotlib observed | 3.10.1 |
| tqdm observed | 4.67.1 |
| pytest observed | 8.3.3 |

Create the environment:

```bash
conda env create -f environment.yml
conda activate ml
pip install -e ".[dev]"
```

The package floors are in `environment.yml` and `pyproject.toml`.

## Data

The benchmark is symbolic latest-write key-value lookup.

Generate it with:

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli generate-benchmark \
  --config configs/benchmark/symbolic_kv_base.json \
  --overwrite
```

Important config values:

| field | value |
| --- | --- |
| benchmark type | `symbolic_kv_stream` |
| seed | 7 |
| keys | 8 |
| values | 128 |
| heldout answer-pair fraction | 0.1 |
| train samples | 8000 |
| validation/test/heldout samples | 1024 each |
| active train keys per sample | 2 to 3 |
| overwrites per train sample | 8 |
| train queries per sample | 6 to 7 |

Build the paper probe set:

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli generate-probe-set \
  --benchmark-dir data/generated/symbolic_kv_stream_learnability \
  --output artifacts/runs/symbolic_kv_reference_formation/analysis/probe_set.jsonl \
  --examples-per-split 96 \
  --split validation_iid \
  --overwrite
```

Expected output:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/probe_set.jsonl
```

## Runs Used In The Paper

The paper uses two closely related seed-7 runs. They share the same task, model size, optimizer recipe, and `16000`-step training budget, but they serve different purposes.

| run | config | purpose |
| --- | --- | --- |
| `symbolic_kv_heldout_generalization` | `configs/train/symbolic_kv_generalization.json` | sparse-checkpoint run used to select a strong heldout-generalizing model |
| `symbolic_kv_reference_formation` | `configs/train/symbolic_kv_formation.json` | dense-checkpoint run used for formation traces, SVD, causal patching, and exact optimizer accounting |

Most exact formation claims in the paper use `symbolic_kv_reference_formation`, especially the `0 -> 6000` optimizer-trace horizon. Do not mix those numbers with the best-checkpoint selection metrics from `symbolic_kv_heldout_generalization`.

## Model And Training

The dense formation training config is:

```text
configs/train/symbolic_kv_formation.json
```

Important values:

| field | value |
| --- | --- |
| seed | 7 |
| layers | 3 |
| heads | 4 |
| `d_model` | 128 |
| `d_ff` | 512 |
| max sequence length | 96 |
| dropout | 0.0 |
| batch size | 128 |
| training steps | 16000 |
| optimizer | AdamW |
| learning rate | 0.0004 |
| beta1 / beta2 | 0.9 / 0.95 |
| weight decay | 0.01 |
| gradient clip | 1.0 |
| warmup | 200 steps |
| schedule | constant |
| checkpoint frequency | 250 steps |

Train the dense formation run:

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli train \
  --config configs/train/symbolic_kv_formation.json \
  --overwrite
```

Expected output:

```text
artifacts/runs/symbolic_kv_reference_formation/
```

To regenerate the sparse heldout-generalization selection run instead, use:

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli train \
  --config configs/train/symbolic_kv_generalization.json \
  --overwrite
```

Evaluate the best checkpoint:

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli evaluate \
  --config artifacts/runs/symbolic_kv_reference_formation/run_config.json \
  --checkpoint artifacts/runs/symbolic_kv_reference_formation/checkpoints/best.pt \
  --split heldout_pairs
```

## Main Analysis Entry Points

These are the command entry points for the main paper claims. Use [Analysis CLI Guide](analysis_cli_guide.md) for the full command catalog and failure modes.

### Exact Optimizer Trace

This trace is the source of truth for exact batch and optimizer-state attribution.

```bash
RUN=artifacts/runs/symbolic_kv_reference_formation
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli optimizer-update-trace \
  --config $RUN/run_config.json \
  --from-initialization \
  --output-dir $RUN/analysis/optimizer_update_trace/from_init_seed7_0000_6000_stepwise \
  --device mps \
  --end-step 6000 \
  --train-split train \
  --checkpoint-every 1 \
  --checkpoint-start-step 0 \
  --progress-every 100 \
  --top-k-parameters 40 \
  --require-historical-replay \
  --overwrite
```

Expected output:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/
```

### QK AdamW Attribution

This command reproduces the optimizer decomposition for the reference QK route.

```bash
RUN=artifacts/runs/symbolic_kv_reference_formation
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli bilinear-qk-rank-adam-state-attribution \
  --config $RUN/run_config.json \
  --probe-set $RUN/analysis/probe_set.jsonl \
  --optimizer-trace-dir $RUN/analysis/optimizer_update_trace/from_init_seed7_0000_6000_stepwise \
  --output-dir $RUN/analysis/bilinear_qk_rank_adam_state_attribution/from_init_l2h1_rank8_support_value_0000_6000_stepwise \
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

Expected output:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/bilinear_qk_rank_adam_state_attribution/from_init_l2h1_rank8_support_value_0000_6000_stepwise/
```

### Optimizer Ablation

This reproduces the matched seed-7 optimizer ablation. The paper uses it only as
a bounded control: same architecture, data, seed, and `6000`-step budget.

Pilot configs:

```text
configs/train/optimizer_ablation/pilot_seed0007/
```

SGD LR-sweep configs:

```text
configs/train/optimizer_ablation/sgd_lr_sweep_seed0007/
```

Train the pilot variants:

```bash
for CONFIG in configs/train/optimizer_ablation/pilot_seed0007/*.json; do
  echo "training $CONFIG"
  PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli train \
    --config "$CONFIG" \
    --overwrite
done
```

Train the SGD LR sweep:

```bash
for CONFIG in configs/train/optimizer_ablation/sgd_lr_sweep_seed0007/*.json; do
  echo "training $CONFIG"
  PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli train \
    --config "$CONFIG" \
    --overwrite
done
```

Evaluate final checkpoints:

```bash
for CONFIG in configs/train/optimizer_ablation/pilot_seed0007/*.json configs/train/optimizer_ablation/sgd_lr_sweep_seed0007/*.json; do
  RUN="$(jq -r '.output_dir' "$CONFIG")"
  echo "evaluating $CONFIG"
  PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli evaluate \
    --config "$CONFIG" \
    --checkpoint "$RUN/checkpoints/step_006000.pt"
done
```

Run the QK/OV progress audit for each optimizer run:

```bash
PROBE="artifacts/runs/symbolic_kv_reference_formation/analysis/probe_set.jsonl"

for CONFIG in configs/train/optimizer_ablation/pilot_seed0007/*.json configs/train/optimizer_ablation/sgd_lr_sweep_seed0007/*.json; do
  RUN="$(jq -r '.output_dir' "$CONFIG")"
  CKPT_DIR="$RUN/checkpoints"

  if [[ "$CONFIG" == *sgd_lr_sweep_seed0007* ]]; then
    OUT="$RUN/analysis/ov_write_progress/all_heads_0750_6000_sgd_lr_sweep"
  else
    OUT="$RUN/analysis/ov_write_progress/all_heads_0750_6000_optimizer_ablation"
  fi

  echo "ov/qk progress $CONFIG"

  PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli ov-write-progress-report \
    --config "$CONFIG" \
    --probe-set "$PROBE" \
    --checkpoint-dir "$CKPT_DIR" \
    --checkpoint "$CKPT_DIR/step_000750.pt" \
    --checkpoint "$CKPT_DIR/step_001000.pt" \
    --checkpoint "$CKPT_DIR/step_001250.pt" \
    --checkpoint "$CKPT_DIR/step_001500.pt" \
    --checkpoint "$CKPT_DIR/step_001750.pt" \
    --checkpoint "$CKPT_DIR/step_002000.pt" \
    --checkpoint "$CKPT_DIR/step_002250.pt" \
    --checkpoint "$CKPT_DIR/step_002500.pt" \
    --checkpoint "$CKPT_DIR/step_002750.pt" \
    --checkpoint "$CKPT_DIR/step_003000.pt" \
    --checkpoint "$CKPT_DIR/step_003500.pt" \
    --checkpoint "$CKPT_DIR/step_004000.pt" \
    --checkpoint "$CKPT_DIR/step_004500.pt" \
    --checkpoint "$CKPT_DIR/step_005000.pt" \
    --checkpoint "$CKPT_DIR/step_005500.pt" \
    --checkpoint "$CKPT_DIR/step_006000.pt" \
    --output-dir "$OUT" \
    --device mps \
    --head L0H0 --head L0H1 --head L0H2 --head L0H3 \
    --head L1H0 --head L1H1 --head L1H2 --head L1H3 \
    --head L2H0 --head L2H1 --head L2H2 --head L2H3 \
    --score-query-role prediction \
    --support-key-role support_value \
    --distractor-key-role value_distractors \
    --record-side clean \
    --pair-type support_value \
    --max-pairs-per-type 64 \
    --min-pairs-per-type 16 \
    --split validation_iid \
    --top-k-correlations 32 \
    --overwrite
done
```

Expected check:

```bash
find artifacts/runs/symbolic_kv_optimizer_ablation -path '*ov_write_progress_report.json' -print | sort | wc -l
```

Expected count:

```text
15
```

### Weight-Level QK Birth

This produces the weight SVD trace used for the low-rank birth story.

```bash
RUN=artifacts/runs/symbolic_kv_reference_formation
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli weight-svd-trace \
  --config $RUN/run_config.json \
  --checkpoint-dir $RUN/checkpoints \
  --checkpoint $RUN/checkpoints/step_000250.pt \
  --checkpoint $RUN/checkpoints/step_000500.pt \
  --checkpoint $RUN/checkpoints/step_000750.pt \
  --checkpoint $RUN/checkpoints/step_001000.pt \
  --checkpoint $RUN/checkpoints/step_001500.pt \
  --checkpoint $RUN/checkpoints/step_002000.pt \
  --checkpoint $RUN/checkpoints/step_002500.pt \
  --checkpoint $RUN/checkpoints/step_003000.pt \
  --checkpoint $RUN/checkpoints/step_003500.pt \
  --checkpoint $RUN/checkpoints/step_004000.pt \
  --checkpoint $RUN/checkpoints/step_004500.pt \
  --checkpoint $RUN/checkpoints/step_005000.pt \
  --checkpoint $RUN/checkpoints/step_005250.pt \
  --checkpoint $RUN/checkpoints/step_005500.pt \
  --output-dir $RUN/analysis/weight_svd_trace/phase1_000250_5500_top16 \
  --device cpu \
  --max-singular-values 16 \
  --top-vector-ranks 16 \
  --overwrite
```

Expected output:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/weight_svd_trace/phase1_000250_5500_top16/
```

### Write Functional Subspace

This audits whether a source component creates a residual perturbation that downstream readout directions use.

```bash
RUN=artifacts/runs/symbolic_kv_reference_formation
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli mlp-input-functional-subspace-report \
  --config $RUN/run_config.json \
  --probe-set $RUN/analysis/probe_set.jsonl \
  --scalar-pair-rows $RUN/analysis/answer_scalar_residual_diagnosis/functional_subspace_trajectory_0750_3500_stride250/answer_scalar_residual_diagnosis_pair_rows.jsonl \
  --output-dir $RUN/analysis/mlp_input_functional_subspace/l0h0_to_l0mlp_support_prediction_1500_2500 \
  --device mps \
  --pair-type support_value \
  --source-component L0H0 \
  --component L0MLP \
  --position-role prediction \
  --position-role support_value \
  --group-by answer_value \
  --group-by support_value \
  --scalar fixed_source_competitor_margin \
  --scalar fixed_target_competitor_margin \
  --endpoint-role source \
  --endpoint-role target \
  --subspace-rank 4 \
  --overwrite
```

Expected output:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_input_functional_subspace/l0h0_to_l0mlp_support_prediction_1500_2500/
```

### Value-Code Readout

This reproduces the result that the mature prediction-position residual carries a broad value-token identity code. The trajectory command measures when the value code becomes readable.

```bash
RUN=artifacts/runs/symbolic_kv_reference_formation
TRACE_CKPTS=$RUN/analysis/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/checkpoints

PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli value-code-subspace-report \
  --config $RUN/run_config.json \
  --probe-set $RUN/analysis/probe_set.jsonl \
  --checkpoint-dir $TRACE_CKPTS \
  --checkpoint $TRACE_CKPTS/step_000750.pt \
  --checkpoint $TRACE_CKPTS/step_001000.pt \
  --checkpoint $TRACE_CKPTS/step_001250.pt \
  --checkpoint $TRACE_CKPTS/step_001500.pt \
  --checkpoint $TRACE_CKPTS/step_001750.pt \
  --checkpoint $TRACE_CKPTS/step_002000.pt \
  --checkpoint $TRACE_CKPTS/step_002250.pt \
  --checkpoint $TRACE_CKPTS/step_002500.pt \
  --checkpoint $TRACE_CKPTS/step_002750.pt \
  --checkpoint $TRACE_CKPTS/step_003000.pt \
  --checkpoint $TRACE_CKPTS/step_003250.pt \
  --checkpoint $TRACE_CKPTS/step_003500.pt \
  --output-dir $RUN/analysis/value_code_subspace/prediction_answer_value_0750_3500 \
  --device mps \
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

Expected output:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_subspace/prediction_answer_value_0750_3500/value_code_subspace_report.json
```

The causal intervention removes value identity from `layer_2_post_mlp / prediction`.

```bash
RUN=artifacts/runs/symbolic_kv_reference_formation
TRACE_CKPTS=$RUN/analysis/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/checkpoints

PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli geometry-subspace-intervention \
  --config $RUN/run_config.json \
  --probe-set $RUN/analysis/probe_set.jsonl \
  --checkpoint-dir $TRACE_CKPTS \
  --checkpoint $TRACE_CKPTS/step_001500.pt \
  --checkpoint $TRACE_CKPTS/step_001750.pt \
  --checkpoint $TRACE_CKPTS/step_002000.pt \
  --checkpoint $TRACE_CKPTS/step_002500.pt \
  --checkpoint $TRACE_CKPTS/step_003000.pt \
  --checkpoint $TRACE_CKPTS/step_003500.pt \
  --output-dir $RUN/analysis/value_code_causal_intervention/embedding_value_identity_prediction_layer2_remove_rank16_1500_3500 \
  --device mps \
  --stage layer_2_post_mlp \
  --subspace embedding_value_identity \
  --rank 16 \
  --operation remove \
  --position-role prediction \
  --query-mode single_query
```

Expected output:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_causal_intervention/embedding_value_identity_prediction_layer2_remove_rank16_1500_3500/geometry_subspace_intervention_report.json
```

The key control and high-rank sufficiency check use the same command shape. Keep the same config, probe set, stage, position role, and query mode, but change the listed arguments:

| purpose | changed arguments |
| --- | --- |
| rank-matched key control | `--output-dir $RUN/analysis/value_code_causal_intervention/embedding_key_identity_prediction_layer2_remove_rank7_1500_3500 --subspace embedding_key_identity --rank 7 --operation remove` |
| rank-matched value control | `--output-dir $RUN/analysis/value_code_causal_intervention/embedding_value_identity_prediction_layer2_remove_rank7_1500_3500 --subspace embedding_value_identity --rank 7 --operation remove` |
| high-rank value keep | replace the checkpoint list with `step_002000.pt`, `step_002500.pt`, `step_003000.pt`, `step_003500.pt`; use `--output-dir $RUN/analysis/value_code_causal_intervention/embedding_value_identity_prediction_layer2_keep_rank127_2000_3500 --subspace embedding_value_identity --rank 127 --operation keep` |

The contextual transfer rescue tests whether support value-code plus prediction-position context can replace the removed prediction value-code component.

```bash
RUN=artifacts/runs/symbolic_kv_reference_formation
TRACE_CKPTS=$RUN/analysis/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/checkpoints

PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli value-code-transfer-rescue \
  --config $RUN/run_config.json \
  --probe-set $RUN/analysis/probe_set.jsonl \
  --checkpoint-dir $TRACE_CKPTS \
  --checkpoint $TRACE_CKPTS/step_001750.pt \
  --checkpoint $TRACE_CKPTS/step_002000.pt \
  --checkpoint $TRACE_CKPTS/step_002500.pt \
  --checkpoint $TRACE_CKPTS/step_003000.pt \
  --checkpoint $TRACE_CKPTS/step_003500.pt \
  --output-dir $RUN/analysis/value_code_transfer_rescue/support_to_prediction_context_rank16_1750_3500 \
  --device mps \
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
  --batch-size 32 \
  --basis-rank 16 \
  --control shuffled_answer_value \
  --control wrong_support_value \
  --control random_subspace \
  --fit-fraction 0.75 \
  --overwrite
```

Expected output:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_transfer_rescue/support_to_prediction_context_rank16_1750_3500/value_code_transfer_rescue_report.json
```

### Write AdamW Attribution

The cross-seed write AdamW result uses one selected winner path per seed. After the relevant cross-seed traces are produced, the expected reports are:

```bash
find artifacts/runs/symbolic_kv_cross_seed_adam -path '*mlp_functional_write_adam_state_attribution*report.json' -print | sort
```

Expected winner directories:

```text
seed_0011/.../winner_L1H3_to_L1MLP_prediction_ref2500_postgrad_total_1500_2500/
seed_0013/.../winner_L1H3_to_L1MLP_prediction_ref2500_postgrad_total_1500_2500/
seed_0017/.../winner_L1H1_to_L1MLP_prediction_ref2500_postgrad_total_1500_2500/
seed_0023/.../winner_L2H1_to_L2MLP_prediction_ref2500_postgrad_total_1500_2500/
seed_0029/.../winner_L1H1_to_L1MLP_prediction_ref2500_postgrad_total_1500_2500/
```

### Scalar Closure

This refits scalar closure using route deltas and answer-scalar rows.

```bash
RUN=artifacts/runs/symbolic_kv_reference_formation
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli route-to-scalar-closure \
  --route-closure-rows $RUN/analysis/route_to_margin_closure/qk_ov_output_routes_1500_2500_formation/route_to_margin_closure_rows.jsonl \
  --scalar-pair-rows $RUN/analysis/answer_scalar_residual_diagnosis/qk_ov_output_routes_1500_2500_formation/answer_scalar_residual_diagnosis_pair_rows.jsonl \
  --output-dir $RUN/analysis/route_to_scalar_closure/qk_ov_output_routes_1500_2500_formation \
  --scalar moving_answer_margin \
  --scalar fixed_source_competitor_margin \
  --scalar fixed_target_competitor_margin \
  --scalar correct_value_logit \
  --scalar negative_answer_loss \
  --switch-bucket all \
  --overwrite
```

Expected output:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/route_to_scalar_closure/qk_ov_output_routes_1500_2500_formation/route_to_scalar_closure_report.json
```

## Runtime Notes

The cheap commands are reports that refit existing rows. The expensive commands are exact optimizer traces and attribution runs that need stepwise checkpoints. For long runs, keep one terminal per seed and verify that each expected report exists before launching the next dependent command.

The tools are intentionally strict. If an input file, checkpoint, scalar row, route row, or optimizer trace is missing, the correct behavior is to fail loudly.
