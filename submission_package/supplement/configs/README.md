# Config Metadata

This directory contains the minimal configuration metadata for the submitted case study.

## Dataset / Benchmark

- `benchmark_symbolic_kv_base.json`: generator config for the symbolic latest-write key-value benchmark.
- `dataset_metadata.json`: generated dataset metadata, split sizes, holdout-pair metadata, overlap checks, leakage checks, and heuristic baselines.

## Training

- `train_reference_formation.json`: dense-checkpoint AdamW run used for formation tracing, SVD, causal patching, and optimizer-update attribution.
- `train_heldout_generalization.json`: sparse-checkpoint run used to select performant reference behavior.

The paper treats the dense formation run as the traceable reference trajectory, not as an independent cross-seed replication.

## Optimizer Ablation

The `optimizer_ablation/` directory contains the seed-7 AdamW and SGD-family configs used for the bounded optimizer comparison.

These configs are included so reviewers can inspect the optimizer recipe directly. The paper does not claim that all possible SGD recipes fail.
