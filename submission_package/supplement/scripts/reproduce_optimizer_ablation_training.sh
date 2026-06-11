#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

for CONFIG in submission_package/supplement/configs/optimizer_ablation/*.json; do
  python -m circuit.cli train --config "$CONFIG"
done
