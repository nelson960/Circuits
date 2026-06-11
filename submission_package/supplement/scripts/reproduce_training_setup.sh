#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

python -m circuit.cli generate-benchmark \
  --config submission_package/supplement/configs/benchmark_symbolic_kv_base.json

python -m circuit.cli train \
  --config submission_package/supplement/configs/train_reference_formation.json

python -m circuit.cli train \
  --config submission_package/supplement/configs/train_heldout_generalization.json
