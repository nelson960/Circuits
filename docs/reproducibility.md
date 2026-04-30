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

## Model And Training

The reference training config is:

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

Train:

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli train \
  --config configs/train/symbolic_kv_formation.json \
  --overwrite
```

Expected output:

```text
artifacts/runs/symbolic_kv_reference_formation/
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
