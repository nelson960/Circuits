---
layout: default
title: Checkpoint Analysis Plan
description: Checkpoint-level measurement design for symbolic key-value circuit formation.
---

# Checkpoint Analysis Plan

## Current Status

This document began as the checkpoint-level measurement plan. The project has now moved beyond the original coarse checkpoint sweep into route-level and actual-update analysis.

Current state:

```text
reference run:
  artifacts/runs/symbolic_kv_reference_formation

fixed probe sets:
  analysis/probe_set.jsonl
  analysis/probe_set_train.jsonl

current proof direction:
  dataset relation
    -> attention/residual route geometry
    -> actual optimizer update
    -> actual recorded batch support
    -> answer-margin effect
```

The old checkpoint plan is still useful, but it is no longer the whole research process. The next phase is not another broad checkpoint dashboard. It is closing the proof gaps found by the completed experiments.

## Purpose

This document defines what to analyze at every saved checkpoint in the reference formation run.

The goal is not to track every changing weight. That is not tractable and it is not the right object.
The goal is to track the emergence of the **effective mechanism**:

- what behavior appears
- what information becomes available in the residual stream
- which heads and MLP blocks start to matter
- when the mechanism stabilizes
- how that trajectory changes across seeds or factor sweeps

## Main Principle

Use a **hierarchical analysis stack**:

1. behavior
2. residual-stream state
3. heads and MLP blocks as writers/readers
4. residual features or subspaces
5. individual neurons only inside already-localized components

The unit of analysis is not "all weights". The unit of analysis is the reduced effective mechanism inside a dense model.

## Current Reference Regime

Current formation run:

- config: `configs/train/symbolic_kv_formation.json`
- output: `artifacts/runs/symbolic_kv_reference_formation`
- checkpoint cadence: every `250` steps
- main public report: `docs/index.md`
- internal notes: `results.md`
- research ledger: `artifacts/runs/symbolic_kv_reference_formation/analysis/research_ledger/research_ledger.md`

Important selected windows:

- early feature/coalition formation: around `1750 -> 2500`
- mid route formation: around `4500 -> 8250`
- traced optimizer continuation: `5500 -> 5550`
- late reference behavior: through `16000`

Important completed actual-update artifacts:

- optimizer trace: `analysis/optimizer_update_trace/l2h1_support_value_5500_5550_stepwise/`
- actual-batch route attribution: `analysis/actual_batch_route_attribution/support_value_routes_5500_5550_stepwise/`
- stepwise retrieval-separation attribution: `analysis/attention_retrieval_separation_update_attribution/`
- support-value route competition: `analysis/route_competition/support_value_routes_5500_5550_stepwise/`

## Fixed Analysis Inputs

Every checkpoint analysis should use the same fixed inputs:

### 1. Fixed Probe Set

Create a small, immutable probe set for repeated checkpoint analysis.

It should include:

- easy IID examples
- hard IID examples
- heldout-pair examples
- structural OOD examples
- counterfactual examples
- explicit slice coverage for:
  - `active_keys`
  - `writes_since_support`
  - `tokens_since_support`
  - `slot_after_write`

The probe set should be small enough for every-checkpoint analysis and stable enough that metric changes are comparable across checkpoints.

### 2. Full Evaluation Splits

Keep full-split evaluation separate from probe-set analysis.

Use full splits for:

- reference selection
- final checkpoint comparison
- factor-sweep summaries

Do not depend on full-split evaluation for all mechanistic measurements at dense checkpoint cadence.

## What To Analyze At Every Checkpoint

These are the required analyses for **every saved checkpoint**.

### A. Behavioral Metrics

These answer: "what behavior is present now?"

- `answer_accuracy` on the fixed probe set
- slice accuracy on the fixed probe set
- answer logit margin
- answer entropy / confidence
- delta from previous checkpoint

Required outputs:

- overall answer accuracy
- per-slice answer accuracy
- answer-margin summary
- birth-threshold flags

### B. Residual-Stream Probes

These answer: "what information is linearly available in the residual stream now?"

Measure at the key positions needed for the task:

- query-key position
- answer-prediction position
- support-value position

Measure at every layer boundary:

- embedding output
- after each attention block
- after each MLP block
- final residual state before logits

Probe targets:

- queried key identity
- support-value identity
- correct answer value
- stale overwritten value
- overwrite-relevant binary label: "current value vs stale value"

Required outputs:

- per-layer probe accuracy
- per-layer answer logit lens accuracy
- per-layer answer margin
- residual-state drift to previous checkpoint
- residual-state drift to final reference checkpoint

Interpretation:

- `Q` should correspond to query-relevant information becoming available
- `R` should correspond to relevant support information appearing at downstream positions
- `W` should correspond to answer-relevant writeout into the residual stream

### C. Head-Level Metrics

These answer: "which attention heads are starting to do useful work?"

For every head:

- query localization
- support localization
- ablation accuracy drop on the probe set
- change in importance from previous checkpoint

Required outputs:

- per-head localization table
- per-head ablation table
- top-k heads by accuracy drop
- head-rank correlation vs previous checkpoint

### D. MLP-Block Metrics

These answer: "which MLP blocks start to matter, even if individual neurons are too unstable to track directly?"

For every MLP block:

- block ablation accuracy drop
- block residual write norm at task-relevant positions
- block contribution to answer logit direction
- change from previous checkpoint

Required outputs:

- per-layer MLP ablation table
- per-layer write norm summaries
- top-k MLP blocks by causal importance

### E. Dynamics Metrics

These answer: "is the mechanism stabilizing or still reorganizing?"

Track:

- metric delta from previous checkpoint
- representation drift
- head turnover
- MLP turnover
- birth-threshold crossing times

Required outputs:

- checkpoint-to-checkpoint deltas
- birth-step estimates
- stabilization flags

## What Not To Run At Every Checkpoint

The following are too expensive or too detailed to run on every saved checkpoint:

- full activation patching over all layers and positions
- path patching over all component pairs
- neuron-level screening over the full model
- exhaustive feature decomposition

These must run only on **selected birth windows**.

## Birth-Window Escalation

After the all-checkpoint sweep, identify narrow windows where something important changes.

Birth-window triggers:

- large jump in answer accuracy
- large jump in heldout accuracy
- large jump in `Q`, `R`, or `W`
- abrupt increase in head or MLP ablation importance
- abrupt drop in residual drift
- rank-order stabilization of top heads

For those windows only, run:

- activation patching
- path patching
- necessity / sufficiency tests
- residual-state patching across checkpoints
- feature-level or neuron-level inspection inside localized components

## Artifact Layout

Each formation run should produce:

- `metrics.jsonl`
  - training and eval summaries
- `checkpoints/step_XXXXXX.pt`
  - model states
- `analysis/checkpoint_metrics.jsonl`
  - one row per saved checkpoint
- `analysis/probe_set.jsonl`
  - immutable probe examples
- `analysis/birth_windows.json`
  - selected windows for expensive analysis
- `analysis/component_candidates/`
  - head and MLP candidate summaries
- `analysis/interventions/`
  - outputs of activation and path patching

## Minimal Schema For `analysis/checkpoint_metrics.jsonl`

Each row should contain:

- `step`
- behavioral metrics:
  - `answer_accuracy`
  - `heldout_answer_accuracy`
  - `slice_accuracy`
  - `answer_margin`
- coarse dynamical metrics:
  - `q`
  - `r`
  - `w`
- residual metrics:
  - `probe_query_key_by_layer`
  - `probe_support_value_by_layer`
  - `probe_answer_value_by_layer`
  - `residual_drift_by_layer`
- head metrics:
  - `top_heads_by_ablation`
  - `top_heads_by_localization`
  - `head_rank_correlation_prev`
- mlp metrics:
  - `top_mlps_by_ablation`
  - `mlp_write_norm_by_layer`
- transition metrics:
  - `delta_answer_accuracy`
  - `delta_q`
  - `delta_r`
  - `delta_w`

## Already Implemented

Currently implemented:

- behavioral evaluation
- slice accuracy
- `Q/R/W`
- head localization
- head ablation
- best-checkpoint selection
- reference selection across runs
- fixed probe-set generation and storage
- dataset-geometry reporting
- attention-geometry trace
- path-logit decomposition / DLA
- geometry subspace interventions
- controlled causal variable patching
- feature-family ranking and lineage tools
- candidate mechanism report
- candidate birth model
- coalition map
- prompt-conditioned neuron trace
- route-gradient selection and decomposition
- checkpoint-update attribution
- attention-score delta decomposition
- attention-score/update attribution
- attention retrieval-chain report
- optimizer-update trace
- actual-batch route attribution
- research proof ledger

## Next Analysis Work To Build

The original missing coarse tools are mostly no longer the blocker. The current missing work is proof closure:

1. actual-batch query-key route attribution
2. route-to-answer-margin closure in the same traced window
3. second-order residual accounting for first-order attribution errors
4. actual-batch route competition across a wider candidate set
5. superposition-aware decomposition that avoids treating neurons as clean atoms
6. cross-seed repeat of the role-level geometry and update results
7. future exact training trace with batch stream recorded from the beginning

The important new constraint from the completed actual-batch support-value run is:

```text
recorded batches support L2H1,
but broad residual routes receive more support,
and batch-support ranking does not equal realized route-growth ranking.
```

So the next analysis must explain:

```text
batch support -> realized route growth
```

not merely measure positive support for one route.

## Practical Research Workflow

### Phase A: Single-Seed Formation Trace

- train `configs/train/symbolic_kv_formation.json`
- run every-checkpoint sweep
- identify birth windows
- run interventions on selected windows

Status: mostly completed for seed 7.

### Phase B: Route And Update Attribution

- define candidate routes from dataset geometry, QK/OV geometry, DLA, and causal patching
- compare route growth over checkpoints
- attribute actual checkpoint deltas with fixed source bases
- trace one-step optimizer continuations
- connect actual recorded batches to route support

Status: partially completed. The support-value actual-batch route attribution is complete; query-key actual-batch attribution and route-to-answer closure remain.

### Phase C: Seed Replication

- rerun the same formation config for several seeds
- compare:
  - birth times
  - top heads
  - top MLP blocks
  - residual probe trajectories
  - stabilization timing

Status: still missing for final claims.

### Phase D: Factor Screens

Vary one factor at a time:

- depth
- width
- heads
- learning rate
- weight decay
- initialization scale
- curriculum
- dataset difficulty

For each factor, compare:

- final behavior
- birth times
- head and MLP candidate sets
- residual trajectories
- stability across seeds

## Decision Rule For Starting Deep Circuit Work

The project has already started deep circuit work for the reference run. For future runs, start expensive circuit-analysis work only when:

- the reference regime is fixed
- the probe set is fixed
- the every-checkpoint sweep is running
- birth windows are identified

Do not start with neuron-level inspection across all checkpoints.
Start with the all-checkpoint coarse sweep, then escalate only on selected windows.

For the current run, the decision rule has changed:

```text
Do not add more broad observation tools until the proof gaps are closed.
Prioritize actual-batch attribution, route-to-answer closure, residual-error accounting,
and cross-seed repeat.
```
