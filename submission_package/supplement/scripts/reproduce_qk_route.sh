#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

RUN="artifacts/runs/symbolic_kv_reference_formation"
CONFIG="$RUN/run_config.json"
PROBE="$RUN/analysis/probe_set.jsonl"
CKPTS="$RUN/analysis/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/checkpoints"

python -m circuit.cli bilinear-qk-match-separation \
  --config "$CONFIG" \
  --probe-set "$PROBE" \
  --checkpoint-dir "$CKPTS" \
  --output-dir "$RUN/analysis/bilinear_qk_match_separation/reviewer_reproduce_l2h1_support_value_vs_distractors" \
  --device cpu \
  --layer 2 \
  --head 1 \
  --position-role prediction \
  --positive-role support_value \
  --negative-role distractor_value \
  --split validation_iid \
  --overwrite
