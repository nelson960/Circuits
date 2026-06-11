#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

RUN="artifacts/runs/symbolic_kv_reference_formation"
CONFIG="$RUN/run_config.json"
PROBE="$RUN/analysis/probe_set.jsonl"
CKPTS="$RUN/analysis/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/checkpoints"

python -m circuit.cli value-code-transfer-rescue \
  --config "$CONFIG" \
  --probe-set "$PROBE" \
  --checkpoint-dir "$CKPTS" \
  --checkpoint "$CKPTS/step_001750.pt" \
  --checkpoint "$CKPTS/step_002500.pt" \
  --checkpoint "$CKPTS/step_003500.pt" \
  --output-dir "$RUN/analysis/value_code_transfer_rescue/reviewer_reproduce_support_to_prediction_context_rank16" \
  --device cpu \
  --source-stage layer_1_post_mlp \
  --target-stage layer_2_post_mlp \
  --context-stage layer_1_post_mlp \
  --source-position-role support_value \
  --target-position-role prediction \
  --context-position-role prediction \
  --group-by answer_value \
  --split validation_iid \
  --max-records 256 \
  --basis-rank 16 \
  --fit-fraction 0.75 \
  --overwrite
