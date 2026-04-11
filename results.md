# Results

Update policy: manual only. Do not update this file unless explicitly requested.

## Project Status

This repo now has:

- a clean stream-based symbolic KV next-token benchmark
- a provisional reference training regime
- a dense-checkpoint formation run
- a fixed probe set
- a layered analysis stack covering:
  - behavior
  - `Q/R/W`
  - residual-stream probes
  - head localization and ablation
  - MLP block ablation and write metrics
  - candidate-neuron screening
  - checkpoint-to-checkpoint comparisons
  - first-pass feature analysis

This is enough to begin real circuit-formation research. It is not yet enough to claim a final explanation of why SGD selects this mechanism.

The current state is:

- IID behavior is strong enough to analyze
- heldout-pair generalization is real and substantial
- structural OOD is still weak
- the current mechanistic story is partial but already nontrivial

## Scientific Goal

The goal is not merely to produce a benchmark model.

The goal is to study:

- how useful circuits form during training
- which factors affect which circuit is selected
- why gradient descent reinforces one mechanistic family over another
- whether formation can be described with reduced state variables rather than only final behavior

The intended output is both:

- experimental evidence
- a mathematically structured description of formation dynamics

## Relationship To Earlier Work

This project should be understood as a continuation of earlier work on motif emergence rather than a disconnected benchmark project.

Related prior work by the author:

- [Mechanistic Transparency](https://nelson960.github.io/Mechanistic-Transparency/)

The current repo is aimed at the next step after that line of work:

- move from motif emergence in controlled settings to circuit formation in small decoder-only next-token models
- track formation through training rather than only describing mature motifs
- connect emergent structure to optimization and data factors

## Why The Old Path Was Rejected

The earlier microlanguage-style direction was rejected as the main benchmark because it drifted away from GPT-like training.

Main problems:

- answer masking
- benchmark/task mismatch
- special heads and task-specific fixes
- encoder-style reformulations
- too much benchmark-specific machinery

That made it unsuitable for studying circuit formation under plain decoder-only next-token training.

## Benchmark Development Path

### First Attempt

The first fresh-repo benchmark used a rigid terminal-answer format of the form:

```text
SET ...
SET ...
QRY ...
ANS ...
```

That version was rejected as the main benchmark because:

- it had one obvious answer site
- it rewarded format learning more than mechanism learning
- it left too little room for meaningful circuit competition
- it put too little task-relevant supervision into the LM objective

Observed outcome:

- token accuracy looked moderate
- answer accuracy was too low
- the setup was not useful for the research program

### Current Main Benchmark

The benchmark was replaced with a stream-based symbolic KV task:

```text
<bos> W K00 V12 W K03 V04 R K00 V12 W K00 V07 R K03 V04 R K00 V07 <eos>
```

This fixed the main issues:

- multiple answer-bearing read events per sequence
- plain next-token prediction over the full stream
- no answer mask
- no classifier head
- explicit latent program
- clean control over:
  - active keys
  - overwrite count
  - query count
  - query lag

Main dataset config:

- benchmark config: `configs/benchmark/symbolic_kv_base.json`
- benchmark name: `symbolic_kv_stream_learnability`
- output: `data/generated/symbolic_kv_stream_learnability`

Current benchmark settings:

- `num_keys = 8`
- `num_values = 128`
- `holdout_answer_pair_fraction = 0.1`
- train/IID splits:
  - `active_keys = 2..3`
  - `overwrite_count = 8`
  - `num_queries = 6..7`
  - `query_lag = 1..2`
- structural OOD:
  - `active_keys = 4..5`
  - `overwrite_count = 10..12`
  - `num_queries = 8..10`
  - `query_lag = 2..3`

## Benchmark Diagnostics

The current dataset passes the intended sanity checks.

From `data/generated/symbolic_kv_stream_learnability/metadata.json`:

- exact-sequence overlap across splits: `0`
- latent-program overlap across splits: `0`
- heldout leakage outside heldout split: `0`
- simple heuristics are weak:
  - `first_value_for_key = 0.0`
  - `last_value_before_query = 0.0`
  - strongest `most_frequent_value_before_query` is about `0.146`

This does not prove the benchmark is perfect, but it rules out several trivial shortcut explanations.

## Model Development Path

The model remained intentionally small for interpretability.

Current reference architecture:

- config basis: `configs/train/symbolic_kv_generalization.json`
- formation variant: `configs/train/symbolic_kv_formation.json`

Model:

- `d_model = 128`
- `n_layers = 3`
- `n_heads = 4`
- `d_ff = 512`
- dropout `0.0`
- max sequence length `96`

Parameter count:

- `626,048`

This size was chosen as a compromise:

- large enough to solve the learnable regime
- small enough to support dense checkpoint analysis

## Optimization Development Path

### Learnability-Fast Run

The first useful regime was a learnability-oriented run:

- run: `artifacts/runs/symbolic_kv_learnability_fast`
- best checkpoint step: `5000`

This gave strong IID performance but weak heldout generalization.

Main lesson:

- strong IID alone is not enough to define the reference regime

### Heldout-Generalization Run

A new run was introduced that:

- evaluated `validation_iid` and `heldout_pairs` during training
- saved the best checkpoint by `heldout_pairs.answer_accuracy`
- used full heldout evaluation instead of partial-batch estimates

Run:

- `artifacts/runs/symbolic_kv_heldout_generalization`

Best checkpoint:

- step `13000`

### Decay Variant

A cosine-decay variant was tested:

- `artifacts/runs/symbolic_kv_heldout_generalization_decay`

Result:

- worse heldout performance than the constant-LR heldout run

Main lesson:

- the tested decay schedule did not improve the reference regime

## Reference Configuration

The repo now includes an explicit selector that ranks completed runs by:

1. `heldout_pairs.answer_accuracy`
2. `validation_iid.answer_accuracy`
3. `structural_ood.answer_accuracy`
4. `test_iid.answer_accuracy`
5. `counterfactual.answer_accuracy`

Selection artifact:

- `artifacts/reference_selection/reference_selection.json`

Current provisional reference:

- run: `artifacts/runs/symbolic_kv_heldout_generalization`
- config: `configs/train/symbolic_kv_generalization.json`
- best checkpoint: `step 13000`
- selection metric: `heldout_pairs.answer_accuracy`

### Selected Checkpoint Metrics

At the selected checkpoint:

- `validation_iid.answer_accuracy = 0.9579`
- `test_iid.answer_accuracy = 0.9578`
- `heldout_pairs.answer_accuracy = 0.8730`
- `structural_ood.answer_accuracy = 0.5082`
- `counterfactual.answer_accuracy = 0.9599`

Interpretation:

- IID is solved well enough to support mechanistic work
- heldout-pair generalization is real and strong
- structural OOD is still weak
- this regime is good enough to begin the formation-analysis phase, but not yet scientifically final

## Why Global Token Accuracy Is Not The Main Metric

The global `token_accuracy` is not a useful primary objective for this benchmark.

Reason:

- it is computed over every next token in the stream
- many write values are intentionally stochastic under the prefix
- the model is not expected to predict random write values on validation

Important observation from the current benchmark:

- only about `12.75%` of all next-token targets are the actual query-answer value tokens

At the selected reference checkpoint:

- `value_answer` accuracy is high
- `value_write` accuracy is near zero
- overall `token_accuracy` therefore stays around `0.65`

This is expected and not a sign that the task-relevant mechanism failed.

The main metrics for the research are:

- `answer_accuracy`
- slice accuracy
- heldout and structural OOD behavior
- mechanistic localization and causal metrics

## What Tweaks Were Necessary To Reach The Current Regime

### Dataset Tweaks

- replaced terminal-answer format with stream-based `W/R` events
- increased answer-bearing events per sequence
- constrained IID regime to `2..3` active keys
- kept overwrite pressure meaningful but manageable
- defined heldout-pair split explicitly
- added structural OOD with more keys, more queries, and larger lag
- enforced overlap and leakage checks

### Training Tweaks

- moved from generic checkpointing to best-checkpoint selection
- made the best-checkpoint split explicit in config
- introduced a separate heldout-focused training config
- separated learnability runs from formation runs
- created a dense-checkpoint formation config

### Evaluation Tweaks

- added answer-focused evaluation
- added token-role metrics:
  - `read_key_accuracy`
  - `write_key_accuracy`
  - `write_value_accuracy`
- added reference-run selection across completed runs

### Analysis Tweaks

- added residual-stream return path in the model
- added MLP masking for block ablation
- added neuron masking for candidate-neuron analysis
- added fixed probe-set generation
- added per-checkpoint sweep over dense checkpoints
- added birth-window summary and checkpoint-to-checkpoint comparison tooling
- added first-pass feature analysis with sparse autoencoders on selected stages

## Initial Research Tracking Plan

The research should start with hierarchical tracking, not neuron-by-neuron brute force.

Track at every saved checkpoint:

### 1. Behavior

- `answer_accuracy`
- heldout probe performance
- slice accuracy
- answer margin
- confidence / entropy

### 2. Coarse Mechanistic State

- `Q`
- `R`
- `W`

These are currently operational proxies, not final theory.

### 3. Residual-Stream State

At task-relevant positions and stages:

- query-key information
- support-value information
- answer-value information
- stage-level logit-lens style readout quality
- residual drift from previous checkpoint

### 4. Component State

- head localization
- head ablation importance
- MLP ablation importance
- MLP write magnitude / answer-margin effect

### 5. Localized Fine-Grained State

- candidate-neuron write screening
- top-neuron ablation within selected MLP layers
- selected residual-stage feature analysis

### 6. Dynamics

- change from previous checkpoint
- birth-threshold crossing
- stabilization vs turnover

## Current Formation Stack

The formation stack now has:

- config: `configs/train/symbolic_kv_formation.json`
- output: `artifacts/runs/symbolic_kv_reference_formation`
- checkpoint spacing: every `250` steps
- fixed probe set:
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/probe_set.jsonl`
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/probe_set.metadata.json`
- checkpoint sweep outputs:
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/checkpoint_metrics.jsonl`
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/checkpoint_metrics_summary.json`
- birth-window report:
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/birth_window_analysis.json`
- checkpoint comparison reports:
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/compare_1500_vs_1750_2000.json`
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/compare_4250_vs_4500_4750.json`
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/compare_7500_vs_12000_14000_16000.json`
- feature reports:
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/features_14000_vs_7500_layer_2_post_mlp.json`
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/features_14000_vs_7500_final_norm.json`

This is the first real instrumentation layer for the formation study.

## What The Checkpoint Sweep Does

The checkpoint sweep analyzes the actual checkpoint `.pt` files, not just training logs.

For each saved checkpoint, it:

- loads the model state from `step_*.pt`
- evaluates the checkpoint on a fixed probe set
- records:
  - answer behavior
  - heldout probe behavior
  - `Q/R/W`
  - residual-stage probes
  - answer margins by stage
  - head localization
  - head ablation
  - MLP block ablation
  - MLP write metrics
  - candidate neurons
  - top-neuron ablations
  - checkpoint-to-checkpoint drift

The output `checkpoint_metrics.jsonl` is machine-oriented. The output `checkpoint_metrics_summary.json` identifies candidate birth windows.

## What The Birth Windows Show

The first dense sweep found three main windows:

- early birth window: `1500-2000`
- mid consolidation window: `4250-4750`
- late reorganization window: `7500-8000`

From `checkpoint_metrics_summary.json`:

- top answer gain step: `1750`
- top heldout gain step: `4500`
- top `Q` gain step: `7750`

The sweep-level interpretation was:

- early emergence is sharp rather than fully gradual
- heldout improvement is delayed relative to the first behavioral jump
- later changes are still happening after the first usable circuit appears

## What The Birth-Window Analysis Added

The birth-window report was the first structured interpretation layer over the sweep.

It showed:

- early window:
  - `answer_accuracy` rises from about `0.024` to about `0.371`
  - `heldout` rises from `0.0` to about `0.118`
  - `L0H0` becomes strongly localized and strongly causal
  - `layer 0 MLP` becomes almost fully necessary
- mid window:
  - `answer_accuracy` rises from about `0.662` to about `0.709`
  - `heldout` rises from about `0.327` to about `0.477`
  - `L1H2` and `L2H1` become stronger routing candidates
  - `layer 2 MLP` becomes a stronger write candidate
- late window:
  - the strongest drift concentrates in upper stages
  - `layer_2_post_mlp` and `final_norm` become the main late-change candidates

This was the first point where the repo moved beyond “final accuracy plus head ablation” into actual formation-stage hypotheses.

## What The Checkpoint-To-Checkpoint Comparisons Added

The comparison tool moved from screening to explicit causal hypotheses.

### Early Window: `1500` vs `1750` and `2000`

Main result:

- patching `final_norm` from `1750` into `1500` gives about `+0.291` answer accuracy
- patching `layer_2_post_mlp` from `2000` into `1500` gives about `+0.341`
- patching `layer_2_post_attn` also helps, but less

Interpretation:

- the early checkpoint is not just missing token access
- the decisive missing piece is a usable top-layer answer state

At the same time:

- `L0H0` ablation importance jumps from about `0.019` at `1500` to about `0.309-0.350` at `1750-2000`
- `layer 0 MLP` jumps from a small effect to nearly full necessity

Interpretation:

- lower layers form the scaffold
- upper-layer writeout makes that scaffold behaviorally useful

### Mid Window: `4250` vs `4500` and `4750`

Main result:

- patching `layer_2_post_mlp` from `4500` into `4250` gives about `+0.124` heldout
- patching `layer_2_post_mlp` from `4750` into `4250` gives about `+0.157` heldout
- `final_norm` and `layer_2_post_attn` are also strong, but slightly weaker than `layer_2_post_mlp`

Interpretation:

- mid-phase gain is mainly an upper-layer routing-to-writeout improvement
- the model is not discovering an entirely new lower-layer scaffold at this stage

### Late Window: `7500` vs `12000`, `14000`, and `16000`

Main result:

- patching `final_norm` or `layer_2_post_mlp` from `12000/14000` into `7500` improves heldout by about `+0.0458`
- this same patch slightly hurts probe-set answer accuracy at `7500`
- patching from `16000` helps heldout less than `12000/14000`

Interpretation:

- late training is refining upper-layer writeout/readout for heldout performance
- that refinement is not a clean global improvement
- by `16000`, the model appears mildly overspecialized rather than improved across the board

## Current Mechanistic Interpretation

The current evidence is consistent with a staged circuit-formation story:

1. lower-layer scaffold appears
2. routing heads consolidate
3. upper-layer MLP writes become usable
4. late training refines top-layer writeout for heldout generalization
5. continued training overspecializes that writeout slightly

The current best working interpretation is:

- `layer 0` is the bootstrap backbone
- `L0H0` is part of the first retrieval scaffold
- mature routing is mostly a `layer 1/2` head story
- `layer 2 MLP` becomes increasingly important as the late answer writer
- `final_norm` carries the late readout-calibration effect

This is not yet a full proof of the circuit. It is a working mechanistic theory grounded in the current tools.

## What The Neuron Layer Has Shown So Far

The neuron-level screening already added one important negative result:

- the mechanism does not look like a sparse one-neuron bottleneck

Observed pattern:

- top single-neuron ablations are real but small compared with head or MLP-block ablations
- neuron importance shifts upward over training, especially toward `layer 2`
- neuron-level signals are useful mainly after localization, not as the starting point

Interpretation:

- the circuit appears dense at the parameter and neuron level
- the effective mechanism is more visible at the residual/component/feature level

## What The Feature Layer Has Added

The first feature tool fits a sparse autoencoder on selected residual stages and compares checkpoints in the same learned basis.

The first concrete feature comparisons were:

- `step 14000` vs `step 7500` at `layer_2_post_mlp`
- `step 14000` vs `step 7500` at `final_norm`

Main findings:

- late change is not uniform drift across the whole upper layer
- a subset of upper-stage features gets substantially stronger from `7500` to `14000`
- those features often have:
  - positive `correctness_gap`
  - positive `heldout_gap`
  - positive heldout change vs source checkpoint
- but they also usually have negative `structural_ood_gap`

Interpretation:

- late training is strengthening upper-layer answer-writing / readout features
- those features help on heldout-pair generalization
- they do not solve the structural OOD regime
- this supports the view that late training improves familiar-regime writeout rather than discovering a broader abstraction

Important caveat:

- the current SAE basis is still too dense for strong semantic claims
- high `active_fraction` means many features are on most of the time
- therefore the current feature layer is useful for screening, not yet for a final feature-level circuit claim

## What The Current Tools Still Do Not Explain

We still do not yet have a satisfying answer to:

- why SGD reinforced this circuit family rather than another
- what the stable feature basis of the circuit is across training
- which feature groups are causally necessary and sufficient
- how training updates shift the model into or between circuit families

This is the current conceptual gap.

Checkpoint-by-checkpoint analysis can show:

- when behavior appears
- where the useful state appears
- which components become important

But it does not by itself explain:

- why gradient descent selected those states and components

## Updated Methodological Direction

The current direction after the first feature tool is:

- do not aim to track every neuron directly
- do not rely only on heads/MLPs either
- build a stable multi-scale analysis stack

The next useful object is not “all neurons at all checkpoints.”

The next useful object is:

- a shared feature basis across checkpoints
- feature trajectories through training
- causal tests on those features
- lineage from heads and MLP blocks into those features

So the planned next direction is:

### 1. Shared Feature Basis

Fit one feature dictionary over activations pooled across many checkpoints at the same stage, rather than fitting a fresh local basis for one checkpoint pair.

Goal:

- stable feature IDs across training
- actual feature trajectories `a_k(t)`

### 2. Feature Trajectory Sweep

For each important feature, track over checkpoints:

- mean activation
- active fraction
- correctness gap
- heldout gap
- structural OOD gap
- answer-direction alignment
- birth time

### 3. Feature Causal Analysis

Add:

- feature ablation
- feature patching between checkpoints
- feature-group interventions

This is the feature-level analogue of the current residual/head/MLP patching.

### 4. Feature Lineage

For important features, ask:

- which heads most increase them
- which MLP blocks write them
- which neurons are the strongest contributors within those blocks
- which later readout directions consume them

### 5. Dynamics Layer

Connect the formation story to SGD by tracking:

- feature emergence
- feature reinforcement
- alignment with useful output directions
- gradients on relevant residual states or feature activations
- update-to-update stabilization or competition

This is the right path toward a reduced dynamical description.

## What We Are Not Doing

We are not trying to:

- track every weight directly
- explain the full dense circuit at once
- inspect all neurons at all checkpoints as the primary analysis object

The reason is that almost all weights can influence training, but far fewer are part of the effective online mechanism.

The practical object of study is:

- effective residual subspaces
- heads and MLP blocks as writers/readers
- feature groups
- low-dimensional formation trajectories

## Planned Next Research Steps

### Phase 1: Stabilize The Feature Layer

- build shared feature dictionaries across checkpoints
- rerun the layered analysis using stable feature IDs
- identify feature birth times

### Phase 2: Feature-Level Causal Work

- feature ablation
- feature patching
- feature-group causal tests

### Phase 3: Cross-Run Comparison

Repeat the same formation regime over several seeds and compare:

- birth times
- top heads
- top MLP blocks
- feature trajectories
- stabilization order

### Phase 4: Factor Screens

Vary one factor at a time:

- architecture
- optimizer
- initialization
- curriculum
- task difficulty

Then ask how those changes affect:

- final behavior
- birth time
- selected mechanism
- stability across seeds

### Phase 5: Reduced Mathematical Phase

The mathematical target is not a literal equation over every neuron.

The intended target is a reduced state over useful variables such as:

- scaffold strength
- routing quality
- write quality
- feature-family strengths
- drift / stability

Then fit or test dynamical descriptions of how those variables evolve over training.

## Notes For The Paper

### Core Narrative

The paper should frame the benchmark as a controlled system for studying circuit formation under standard autoregressive training, not as a benchmark for its own sake.

### Important Claims That Are Supported

- a clean decoder-only next-token synthetic benchmark can be made learnable
- strong IID performance is not enough; heldout-aware selection changes which checkpoint should be treated as best
- global token accuracy is misleading in this task because many write values are intentionally stochastic
- circuit-formation analysis needs checkpoint-level instrumentation beyond final accuracy
- the learned mechanism forms in stages rather than appearing all at once
- late heldout gains are concentrated in upper-layer writeout/readout changes rather than a completely new routing solution

### Important Claims That Are Not Yet Supported

- a finalized explanation of the learned circuit
- a stable feature-level decomposition across training
- a low-dimensional closed-form dynamical theory
- strong structural OOD generalization
- stable cross-seed mechanistic equivalence
- a direct explanation of why SGD selected this family rather than another

### Paper-Relevant Observations To Preserve

- the first rigid symbolic-KV attempt was too templated and not suitable as the main benchmark
- switching to a stream-based benchmark increased answer-bearing supervision inside the LM objective
- heldout-based checkpoint selection changed the best model choice materially
- a tested cosine-decay schedule underperformed the constant-LR heldout run
- dense checkpoint sweeps exposed distinct early, mid, and late formation windows
- checkpoint-to-checkpoint patching showed that upper-layer writeout becomes decisive after a lower-layer scaffold is already present
- early neuron screening suggests a dense mechanism rather than a single-neuron bottleneck
- first-pass feature analysis suggests late training strengthens upper-layer features that help heldout but not structural OOD

### Current Provisional Thesis

The current evidence is consistent with the view that:

- useful behavior emerges in a small decoder-only model on a controlled synthetic task
- the mechanism forms in stages rather than appearing as a single event
- the effective circuit is dense at the parameter level but more legible at the residual/component/feature level
- lower layers bootstrap the scaffold
- upper-layer writeout and readout become the decisive late mechanism
- the right path to the SGD question is stable feature trajectories plus causal validation, not raw neuron-by-neuron tracking

## Canonical Current State Report

This section freezes the current reference story after the shared-feature and feature-family analysis work.

It should be treated as the current top-level summary. Earlier sections that describe shared features as only a planned next direction are now partly stale: the shared-feature stack has been implemented and run for the main formation artifacts, but it is not yet a final mechanistic proof.

### Current Research Object

The project is studying circuit formation in a small decoder-only transformer trained with plain autoregressive next-token prediction.

The current benchmark is the stream-based symbolic KV retrieval task:

```text
<bos> W K00 V12 W K03 V04 R K00 V12 W K00 V07 R K03 V04 R K00 V07 <eos>
```

This benchmark remains the right object because it has:

- repeated answer-bearing read events inside the LM objective
- no answer mask
- no classifier head
- no task-specific architecture
- explicit control over keys, values, overwrites, query count, and query lag
- heldout-pair and structural-OOD splits for separating interpolation from broader abstraction

The benchmark config remains:

- `configs/benchmark/symbolic_kv_base.json`
- generated data: `data/generated/symbolic_kv_stream_learnability`

The reference model remains intentionally small:

- `d_model = 128`
- `n_layers = 3`
- `n_heads = 4`
- `d_ff = 512`
- parameters: `626,048`

### Current Selected Checkpoint

The current selected reference checkpoint is still the heldout-selected generalization run:

- run: `artifacts/runs/symbolic_kv_heldout_generalization`
- checkpoint: `artifacts/runs/symbolic_kv_heldout_generalization/checkpoints/best.pt`
- step: `13000`
- selection split: `heldout_pairs`
- selection metric: `answer_accuracy`
- selection value: `0.873018247083458`

Full selected-checkpoint answer accuracies:

- `validation_iid.answer_accuracy = 0.9578527137637138`
- `test_iid.answer_accuracy = 0.9578204743320324`
- `heldout_pairs.answer_accuracy = 0.873018247083458`
- `structural_ood.answer_accuracy = 0.5081577525661805`
- `counterfactual.answer_accuracy = 0.9599219453617532`

Interpretation:

- IID behavior is solved well enough for mechanistic work.
- Heldout-pair generalization is strong enough to be scientifically meaningful.
- Structural OOD remains weak and should not be described as solved.
- The selected checkpoint is a good mechanistic reference, not a final robust-reasoning model.

### Current Formation Run

The current formation run remains:

- config: `configs/train/symbolic_kv_formation.json`
- run: `artifacts/runs/symbolic_kv_reference_formation`
- checkpoint directory: `artifacts/runs/symbolic_kv_reference_formation/checkpoints`
- checkpoint cadence: every `250` steps
- analyzed checkpoints: `64`
- probe set: `artifacts/runs/symbolic_kv_reference_formation/analysis/probe_set.jsonl`

The current birth-window summary identifies:

- early birth window: `1500-2000`, centered at `1750`
- mid consolidation window: `4250-4750`, centered at `4500`
- late reorganization window: `7500-8000`, centered at `7750`

Top sweep triggers:

- top answer gain step: `1750`
- top heldout gain step: `4500`
- top `Q` gain step: `7750`

Current interpretation of these windows:

- `1500-2000`: first usable behavior appears; lower-layer scaffold and upper-layer answer state become behaviorally meaningful.
- `4250-4750`: heldout-pair performance improves; routing/writeout becomes more mature.
- `7500-8000`: upper-stage representations reorganize; late changes concentrate around `layer_2_post_mlp` and `final_norm`.

### Current Shared-Feature Layer

The shared-feature layer is no longer only a plan.

Shared feature bases now exist for:

- `layer_2_post_mlp`
- `final_norm`

Both use:

- `64` features
- input dimension `128`
- fit checkpoints: `7500`, `14000`, `16000`
- probe set: `artifacts/runs/symbolic_kv_reference_formation/analysis/probe_set.jsonl`

Current shared-feature fit metrics:

- `layer_2_post_mlp`
  - explained variance: `0.7457791864871979`
  - active fraction: `0.5410973429679871`
  - reconstruction loss: `0.2546698749065399`
- `final_norm`
  - explained variance: `0.7311904430389404`
  - active fraction: `0.5383508801460266`
  - reconstruction loss: `0.26917073130607605`

Important caveat:

- These bases are useful enough for trajectory and family screening.
- They are still too dense for strong semantic feature claims.
- A feature ID should be treated as an analysis coordinate, not automatically as a natural mechanistic unit.

### Current Feature-Family Layer

The most developed family-level analysis is at `layer_2_post_mlp`.

Artifacts include:

- shared basis: `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/shared_feature_basis.json`
- feature trajectories: `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/trajectories/feature_trajectories.jsonl`
- feature births: `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/births/feature_births.json`
- feature families: `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/families/feature_families.json`
- family births: `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/families/births/feature_family_births.json`
- family traces:
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/families/feature_family_trace_0_top3_14000.json`
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/families/feature_family_trace_1_top3_14000.json`
- family update-link reports:
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/families/feature_family_update_link_0_top3_14000.json`
  - `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/families/feature_family_update_link_1_top3_14000.json`

The `layer_2_post_mlp` family clustering currently found:

- `64` features
- `43` families
- `8` multi-feature families

The two most developed family traces are:

#### Family 0

- family ID: `0`
- representative feature: `55`
- members: `6, 8, 13, 21, 29, 35, 42, 49, 55`
- family birth step: `750`
- family useful birth step: `1000`
- selected top-3 subset: `55, 42, 8`
- selected subset patch, `14000 -> 7500`:
  - answer delta: `0.0`
  - heldout delta: `0.006535947712418277`
  - structural OOD delta: `-0.004608294930875667`
- top linked head: `layer 0 head 1`
- top linked MLP: `layer 0 MLP`
- top linked neuron group: `layer 2`, neurons `180, 121, 427, 39`

Interpretation:

- Family 0 is a plausible useful coalition candidate.
- It has early feature-level birth signals and positive heldout-linked movement.
- Its patch effect is real but small.
- It does not improve structural OOD.

#### Family 1

- family ID: `1`
- representative feature: `44`
- members: `7, 10, 28, 39, 43, 44, 62`
- family birth step: `750`
- family useful birth step: `null`
- selected top-3 subset: `7, 10, 44`
- selected subset patch, `14000 -> 7500`:
  - answer delta: `0.004431314623338234`
  - heldout delta: `0.006535947712418277`
  - structural OOD delta: `0.009216589861751112`
- top linked head: `layer 0 head 1`
- top linked MLP: `layer 0 MLP`
- top linked neuron group: `layer 2`, neurons `180, 121, 427, 39`

Interpretation:

- Family 1 is a comparison coalition, not yet a clearly useful family.
- Its selected subset patch has slightly better broad metric deltas than Family 0, but the family-level trajectory does not meet the current useful-birth rule.
- This is a useful warning that patch effects, family-level trajectories, and semantic interpretation can diverge.

### Current Coalition / Subset Layer

The analysis now has an explicit subset layer, which is effectively the current "coalition" layer.

Current subset artifacts:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/subsets/subset_trajectory_family0_top3.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/subsets/subset_birth_family0_top3.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/subsets/subset_trajectory_family1_top3.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/subsets/subset_birth_family1_top3.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/subsets/subset_trajectory_cross_55_7_42.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/subsets/subset_birth_cross_55_7_42.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp/subsets/subset_competition_family0_top3_vs_family1_top3.json`

Current subset-birth findings:

- Family 0 top-3 subset `55, 42, 8`
  - subset birth step: `750`
  - useful birth step: `1000`
  - active-fraction birth: `4750-5000`
- Family 1 top-3 subset `7, 10, 44`
  - subset birth step: `750`
  - useful birth step: `null`
  - active-fraction birth: `4750-5000`
- Cross-family subset `7, 42, 55`
  - subset birth step: `750`
  - useful birth step: `1000`
  - active-fraction birth: `4750-5000`

Interpretation:

- Several feature coalitions show early birth-like behavior at the feature-statistic level.
- Their active-fraction maturation aligns more with the mid consolidation window around `4750-5000`.
- This is consistent with a two-part picture: early useful directions exist before the coalition becomes active/stable in its mature regime.

### Current Update-Link Layer

The newest family-level artifacts link subset changes across checkpoint intervals to update magnitudes in the associated head, MLP, and neuron group.

For Family 0 top-3:

- selected features: `8, 42, 55`
- top head: `layer 0 head 1`
- top MLP: `layer 0 MLP`
- top neuron group: `layer 2`, neurons `180, 121, 427, 39`
- analyzed intervals: `63`

Top update-link correlations for Family 0 include:

- sweep answer delta vs `delta_r`: `0.7250690435592169`
- subset correctness-gap delta vs `delta_r`: `-0.6018879141749028`
- sweep heldout-answer delta vs `delta_w`: `0.5633165757389528`
- sweep answer delta vs top-head attention update share: `-0.5290955525260733`
- subset useful delta vs `delta_r`: `-0.5119173264233698`

For Family 1 top-3:

- selected features: `7, 10, 44`
- top head: `layer 0 head 1`
- top MLP: `layer 0 MLP`
- top neuron group: `layer 2`, neurons `180, 121, 427, 39`
- analyzed intervals: `63`

Top update-link correlations for Family 1 include:

- sweep answer delta vs `delta_r`: `0.7250690435592169`
- sweep heldout-answer delta vs `delta_w`: `0.5633165757389528`
- subset correctness-gap delta vs global relative update norm: `-0.5454852764946101`
- subset correctness-gap delta vs `delta_r`: `-0.5362913133922584`
- sweep answer delta vs top-head attention update share: `-0.5290955525260733`

Interpretation:

- The update-link layer is now the closest artifact to the SGD question.
- It does not yet prove why SGD selected a mechanism.
- It does provide a concrete bridge between checkpoint-to-checkpoint parameter updates, feature-coalition trajectories, and known component candidates.

### Current Mechanistic Hypothesis

The current best working hypothesis is:

1. A lower-layer scaffold appears early.
2. `layer 0` components, especially `L0H0` in the earlier birth-window analysis and `L0H1` in the current feature-family lineage/update-link layer, are strongly implicated in early feature and routing structure.
3. Upper-layer answer state becomes usable around the early birth window, especially by `layer_2_post_mlp` and `final_norm`.
4. Heldout improvement is delayed relative to first IID behavior and is concentrated in upper-stage writeout/readout refinements.
5. Feature families at `layer_2_post_mlp` expose candidate coalitions whose trajectories and small patch effects are consistent with late heldout tuning.
6. The learned mechanism is distributed: single-neuron ablations remain small relative to head, MLP-block, residual-stage, and feature-coalition effects.

This hypothesis is supported enough to guide the next experiments, but it is not a final explanation.

### Current Unsupported Claims

The current repo still does not support claiming:

- a finalized circuit decomposition
- a natural semantic interpretation of individual shared features
- that Family 0 or Family 1 is a complete circuit
- that the identified feature coalitions are necessary and sufficient
- strong structural OOD generalization
- cross-seed stability of the same heads, MLP blocks, features, or families
- that SGD has been explained rather than correlated with feature-family/update trajectories
- that all relevant neurons have been tracked across all checkpoints
- that the feature-family layer is independent of SAE hyperparameters

### Next Canonical Research Steps

The next useful work is no longer simply "build shared features"; that is partly done.

The next stages are:

1. Broaden coalition analysis beyond Family 0 and Family 1.
2. Run the same family/subset birth, competition, trace, lineage, and update-link stack for the strongest `final_norm` families.
3. Add stronger necessity/sufficiency tests for selected feature coalitions.
4. Check sensitivity to shared-feature fit hyperparameters.
5. Repeat the formation run across seeds and compare:
   - birth windows
   - top heads
   - top MLP blocks
   - selected feature families
   - coalition useful-birth timing
   - update-link correlations
6. Only after cross-seed replication, start treating feature-family strengths as candidate state variables for a reduced dynamical model.

## Current Why-Gap And Birth-Model Direction

This section records the conclusion after the traced `family7` and `family4` candidate-mechanism run.

The current tooling is now good at formation phenomenology:

- when a feature family or subset appears
- where its strongest traced component ancestry sits
- how candidate feature scores move across checkpoints
- how much traced parameter groups align with loss gradients and feature-score gradients
- whether a candidate has heldout support or only probe-local movement

That is not yet the final "why" answer.

The unanswered question is stronger:

```text
Given multiple possible circuits or feature families, why does SGD select one family over another?
```

The current answer is still partly observational. It says which family moved, which components moved with it, and which intervals were useful. It does not yet prove that the selected family was predictable before it appeared.

### Current Traced Family7 / Family4 Result

The current traced mechanism report is:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/mechanism_report/candidate_mechanism_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/mechanism_report/candidate_mechanism_report.md`

Selected candidates:

- `layer2_family7_top2`
  - family: `7`
  - features: `27, 54`
  - useful delta: `0.40821056067943573`
  - heldout delta: `0.19631874561309814`
  - traced feature-score drive: `0.10995836968906202`
  - status: `sgd_supported_generalizing_candidate`
- `layer2_family4_top2`
  - family: `4`
  - features: `1, 59`
  - useful delta: `0.23405300080776215`
  - heldout delta: `0.021932989358901978`
  - traced feature-score drive: `0.14723929510814068`
  - status: `sgd_supported_generalizing_candidate`

Interpretation:

- Family7 remains the stronger circuit-family candidate because it has meaningful heldout gain.
- Family4 is real but weaker: it has feature-score movement without comparable heldout generalization.
- Family7 and Family4 are not clean competitors. Their score-drive correlation is high, so they look more like sibling readouts inside a shared dense coalition.

### Current Component Interpretation

Both traced candidates share the same top component groups:

- `layer0_head3`
- `layer0_mlp`
- `layer2` neuron group `180, 121, 427, 39`

Component-level interpretation:

- `layer0_head3` contributes strongly to loss reduction but has negative feature-score drive for both family7 and family4.
- `layer0_mlp` has positive feature-score drive, especially for family7.
- the `layer2` neuron group has strong positive feature-score drive for family4 and weaker positive drive for family7.

The current best component story is:

1. `layer0_head3` helps the task-loss route but is not the direct birth source of these feature families.
2. `layer0_mlp` is the strongest traced candidate for family7 feature formation.
3. the `layer2` neuron group is more like an amplification or readout shard, especially for family4.
4. family7 looks like the more generalizing branch of the shared module.
5. family4 looks like a nearby sibling branch that is amplified but does not generalize as well.

### Current Causal Patch Caveat

The `7500 -> 14000` subpatch result is negative for family7 and family4 subsets.

This does not reject family7. It shows that `7500 -> 14000` is a poor causal validation window for birth, because family7 formed earlier and is being compressed or rebalanced later.

The next causal patch windows should target positive formation intervals:

- family7: `1750 -> 2500`
- family7: `2750 -> 3750`
- family7: `4250 -> 4500`
- family4: `2000 -> 2500`
- family4: `3500 -> 4500`
- family4: `5500 -> 6000`

### Why The Existing Story Is Not Enough

The current mechanism report answers:

```text
What moved, where did it move, and was the movement useful?
```

The missing model must answer:

```text
Before the candidate is useful, can we predict that this candidate should form?
```

The target mathematical object is:

```text
Delta S_c(t) ~= grad_theta S_c(theta_t) . Delta theta_t
```

and, under SGD-like updates:

```text
Delta S_c(t) ~= -eta_t <grad_theta S_c(theta_t), grad_theta L(theta_t)>
```

where:

- `S_c` is a score for candidate circuit or feature family `c`
- `L` is training loss
- `Delta theta_t` is the checkpoint update
- `<grad S_c, grad L>` is the alignment between the direction that would form the candidate and the loss-reducing gradient

A candidate should be considered explained only if pre-birth factors predict its later birth better than competing candidates.

### Candidate Birth Model Target

The next tool is `candidate-birth-model`.

It should consume:

- candidate registry
- circuit-gradient-link output
- subset birth labels
- subset trajectories through the registry

It should report:

- actual birth or useful-birth step
- strict pre-birth prediction window by default
- candidate birth score
- predicted birth rank
- actual birth rank
- factor decomposition
- whether a requested cutoff leaks post-birth information
- unsupported claims

The initial factor model should be deliberately transparent:

- `feature_score_drive`: cumulative projected update in the candidate feature-score direction
- `gradient_alignment`: mean cosine between update and feature-score gradient
- `loss_utility`: cumulative loss reduction in the candidate parameter scope
- `component_accessibility`: candidate update and gradient share relative to global update
- `activation_support`: candidate activation level at the prediction cutoff
- `amplification`: positive pre-birth activation and active-fraction movement
- `interference_cost`: negative feature-score and useful-movement pressure

The first version is a ranking model, not a final theory. It should be judged by whether it can predict that family7 is the better candidate before family7 becomes useful.

### Updated Scientific Standard

From this point, a candidate explanation is not strong enough if it only says:

- this family formed
- this component contributed
- this interval had positive score drive

The stronger standard is:

```text
Using only pre-birth evidence, the model predicts which candidate will form and why.
```

If the birth model cannot predict family7 over family4, family3, and family5 before their useful birth windows, then the current story remains post-hoc.

If it can, the project starts moving from circuit observation toward an explanation of SGD circuit selection.

### Initial Candidate-Birth-Model Smoke Test

After adding the first `candidate-birth-model` implementation, it was run on the traced `family7` and `family4` artifacts.

Output:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/birth_model/candidate_birth_model_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/birth_model/candidate_birth_model_report.md`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/birth_model/candidate_birth_model_scoreboard.svg`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/birth_model/candidate_birth_model_factors.svg`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/birth_model/candidate_birth_model_birth_order.svg`

Settings:

- birth metric: `useful_birth_step`
- prediction mode: `shared_strict_prebirth`
- effective cutoff: `2000`
- post-birth leakage: `false` for both candidates

Result:

- `layer2_family4_top2`
  - birth-model score: `4`
  - predicted rank: `1`
  - actual useful birth: `2500`
  - actual rank: `2`
- `layer2_family7_top2`
  - birth-model score: `0`
  - predicted rank: `2`
  - actual useful birth: `2250`
  - actual rank: `1`

Interpretation:

- The first transparent factor model does not yet explain why family7 becomes the better candidate.
- This is a useful negative result, not a tooling failure.
- Pre-birth score drive and activation support favor family4 at the shared cutoff, but family7 still becomes useful earlier and generalizes better.
- Therefore the missing factor is likely not just raw feature-score drive.

Current implication:

The next birth-model iteration needs additional factors that distinguish early generalizing utility from raw feature amplification, especially:

- per-feature rather than family-sum birth factors
- heldout-specific gradient alignment
- feature-to-answer readout utility before birth
- interference with already-forming families
- separate treatment of `f54`, `f27`, `f1`, and `f59`

This strengthens the current conclusion: the repo can now test a proposed "why" story, and the first simple story fails. The research should now improve the explanatory model rather than only adding more descriptive traces.
