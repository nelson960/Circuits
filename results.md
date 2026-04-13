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

### Candidate Coalition Map

After the first birth-model failure, the next tool added was `candidate-coalition-map`.

Purpose:

- test whether selected candidate families are separate circuits or sibling readouts of one dense MLP-neuron coalition
- compute per-neuron projected feature-score update drive
- compare candidate score gradients on neuron-specific MLP parameter slices
- produce shared-vs-specific neuron categories and plots

The neuron-specific parameter slice is:

```text
fc_in row + fc_in bias + fc_out column
```

For each candidate `c` and neuron `n`, the tool computes:

```text
Delta score_c,n ~= grad_theta_n score_c . Delta theta_n
```

Implemented command:

- `candidate-coalition-map`

Current outputs:

- `candidate_coalition_map_report.json`
- `candidate_coalition_map_report.md`
- `candidate_coalition_neuron_heatmap.svg`
- `candidate_coalition_shared_specific.svg`
- `candidate_coalition_gradient_conflict_matrix.svg`
- `candidate_coalition_neuron_trajectories.svg`

Initial bounded smoke test:

- output directory: `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/coalition_map_early`
- candidates: `layer2_family7_top2`, `layer2_family4_top2`
- window: `1750 -> 2500`
- neuron layers: `0`, `2`
- individual features included: `f1`, `f27`, `f54`, `f59`

Important environment note:

- running with `--device mps` failed because this execution environment reported MPS unavailable
- the bounded smoke test was then run with `--device cpu`
- the tool did not silently fall back

Initial result:

- `layer2_family7_top2` vs `layer2_family4_top2` mean score-gradient cosine on selected MLP-neuron parameters: about `0.738406`
- `layer2_family7_top2` vs `f54`: about `0.975703`
- `layer2_family4_top2` vs `f1`: about `0.969959`

Shared-vs-specific category summary for the bounded early window:

- shared positive neurons: `484`
- shared positive score drive: `0.50029`
- shared negative neurons: `316`
- shared negative score magnitude: `0.304594`
- conflict neurons: `224`
- conflict positive score: `0.0351674`
- conflict negative score magnitude: `0.0303447`

Interpretation:

- This is early evidence for the dense-coalition hypothesis.
- Family7 and family4 are probably not independent clean circuits.
- They look like sibling feature-family readouts supported by many of the same MLP neurons.

Unsupported:

- causal necessity of the shared-positive neurons
- whether family7-specific neurons explain the heldout advantage
- whether family4-specific neurons explain raw amplification without heldout
- cross-seed stability of the same coalition

Next tests:

- run the coalition map over additional windows: `2750 -> 3750`, `3500 -> 4500`, `4250 -> 4500`, `5500 -> 6000`
- run targeted shared/specific/conflict neuron ablation with `candidate-neuron-intervention`
- defer targeted shared/specific/conflict neuron patching until attention/path geometry clarifies what should be patched
- feed coalition-level factors into the next birth model

## Candidate Neuron Intervention Tool

Built after the early coalition map result to move from update-geometry evidence to causal necessity evidence.

Implemented command:

- `candidate-neuron-intervention`

Inputs:

- config
- probe set
- coalition-map report JSON
- checkpoint directory
- explicit checkpoint step

Why the explicit checkpoint step matters:

- the tool does not infer or silently choose a checkpoint
- the user must decide whether to test the early window endpoint, a later consolidated checkpoint, or another formation stage

What it does:

- loads the selected checkpoint
- builds neuron sets from `candidate-coalition-map`
- zeros selected MLP hidden neurons with the model's `neuron_mask`
- recomputes probe loss, token accuracy, answer accuracy, heldout answer accuracy, and structural-OOD answer accuracy
- recomputes candidate feature-family scores under each ablation
- reports score drops as `baseline feature score - ablated feature score`

Neuron sets generated from the coalition map:

- `shared_positive`
- `conflict`
- `shared_negative`
- `top_overlap`
- `candidate_specific:<candidate_id>`

Outputs:

- `candidate_neuron_intervention_report.json`
- `candidate_neuron_intervention_report.md`
- `candidate_neuron_intervention_behavior.svg`
- `candidate_neuron_intervention_feature_scores.svg`
- `candidate_neuron_intervention_set_sizes.svg`
- optional `candidate_neuron_intervention_single_neurons.svg`

Interpretation rule:

- if `shared_positive` ablation drops both family7 and family4 feature-family scores, that supports causal shared-neuron necessity
- if a `candidate_specific:<candidate_id>` ablation mainly drops one family, that supports family-specific specialization inside the dense coalition
- if `conflict` ablation helps one target and hurts another, that is evidence for internal competition rather than shared support

Still unsupported:

- causal sufficiency of the shared-positive neurons
- source-to-target neuron activation patching
- cross-seed stability
- per-minibatch intervention trace

## Candidate Neuron Intervention Result And Mathematical Pivot

After building `candidate-neuron-intervention`, it was run on the early family7/family4 coalition map.

Artifact:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/neuron_intervention_early_step2500/candidate_neuron_intervention_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/neuron_intervention_early_step2500/candidate_neuron_intervention_report.md`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/neuron_intervention_early_step2500/candidate_neuron_intervention_behavior.svg`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/neuron_intervention_early_step2500/candidate_neuron_intervention_feature_scores.svg`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/neuron_intervention_early_step2500/candidate_neuron_intervention_set_sizes.svg`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/neuron_intervention_early_step2500/candidate_neuron_intervention_single_neurons.svg`

Settings:

- coalition map: `coalition_map_early`
- checkpoint step: `2500`
- device: `mps`
- top K per set: `8`
- individual feature scores included: `f1`, `f27`, `f54`, `f59`

Baseline at checkpoint `2500`:

| metric | value |
| --- | ---: |
| loss | `2.45865` |
| token accuracy | `0.489369` |
| answer accuracy | `0.364845` |
| heldout accuracy | `0.104575` |
| structural OOD accuracy | `0.142857` |

Feature-family score proof:

| ablated set | family4 score drop | family7 score drop | mean candidate score drop | all candidate scores drop |
| --- | ---: | ---: | ---: | --- |
| `shared_positive` | `0.01124` (`5.84%`) | `0.00586` (`3.19%`) | `0.00855` | true |
| `top_overlap` | `0.00790` (`4.10%`) | `0.00379` (`2.07%`) | `0.00584` | true |
| `shared_negative` | `0.03691` (`19.19%`) | `0.03818` (`20.79%`) | `0.03754` | true |
| `conflict` | `-0.00311` (`-1.62%`) | `0.00046` (`0.25%`) | `-0.00133` | false |

Supported by this result:

- `shared_positive` neurons causally support both family7 and family4 feature-family scores.
- `top_overlap` neurons also causally support both family scores.
- `conflict` neurons behave like actual internal competition, not shared support.
- `shared_negative` neurons are the strongest current score carriers even though the update direction over the early interval pushed against the candidate scores.

Important correction:

The signs from the coalition map are training-update signs, not static causal utility labels.

`shared_negative` means:

```text
During the selected checkpoint interval,
the SGD update through those neuron parameter slices pushed against the candidate score.
```

It does not mean:

```text
Those neurons do not currently carry the feature-family signal.
```

In fact, ablating the top `shared_negative` set at checkpoint `2500` produced the largest candidate feature-score drop:

```text
family4 drop ~= 19.19%
family7 drop ~= 20.79%
```

This is strong evidence that the current circuit is dense, mixed, and dynamically rebalanced. A static neuron list is not enough.

Behavior-level result:

| ablated set | answer drop | heldout drop | structural OOD drop | loss increase |
| --- | ---: | ---: | ---: | ---: |
| `shared_positive` | `-0.00443` | `0.01307` | `-0.00461` | `-0.00062` |
| `top_overlap` | `-0.00591` | `0` | `-0.00922` | `0.00064` |
| `shared_negative` | `-0.00739` | `0.00654` | `-0.00461` | `-0.03991` |
| `candidate_specific:layer2_family4_top2` | `0.00148` | `0.01961` | `-0.01382` | `-0.00721` |

Interpretation:

- The intervention proves causal feature-score support.
- It does not yet prove clean task-behavior necessity.
- The task behavior is compensated across dense overlapping routes.
- Neuron-level intervention is necessary, but it is not sufficient to explain circuit selection.

Single-neuron proof inside the shared-positive set:

| neuron | mean candidate score drop | interpretation |
| --- | ---: | --- |
| `L0N326` | `0.012998` | strong shared support |
| `L0N376` | `0.012859` | strong shared support |
| `L0N488` | `0.006657` | moderate shared support |
| `L0N411` | `0.006526` | moderate shared support |
| `L0N302` | `-0.019708` | ablation increases family scores |
| `L0N36` | `-0.008512` | ablation increases family scores |

This is direct evidence against a simple sparse-neuron story.

Current conclusion:

The family7/family4 result is now stronger than observation:

```text
dense shared coalition -> causal family-score support -> behavior still compensated
```

That means the research has reached a dead end for the current style of tool if the goal is the why question. More neuron lists will not explain why SGD chooses one internal algorithm over another.

The next object must be mathematical geometry:

```text
dataset relation d(x, y)
  -> attention retrieval geometry
  -> MLP feature geometry
  -> path-level logit contribution
  -> SGD gradient alignment
  -> selected circuit
```

Next planned tools:

1. `dataset-geometry-report`
2. `attention-geometry-trace`
3. `path-logit-decomposition`
4. `example-gradient-geometry`
5. `mechanism-hypothesis-tester`

The new mathematical target:

```text
m_t(x, y) =
  logit_t(y | x) - logsumexp_{z != y} logit_t(z | x)
```

and:

```text
m_t(x, y) ~= sum_P C_P(theta_t, x, y)
```

where `C_P` is a path-level contribution.

The circuit-selection hypothesis should be tested as:

```text
Circuit P wins over circuit Q when:

E_D[<grad_theta C_P(theta_t, x, y), -grad_theta L(theta_t, x, y)>]
>
E_D[<grad_theta C_Q(theta_t, x, y), -grad_theta L(theta_t, x, y)>]
```

subject to architecture, initialization, superposition, interference, and causal faithfulness constraints.

Updated research stance:

- The dense interconnected-family hypothesis is supported.
- The shared-neuron causal-score hypothesis is supported.
- The clean behavior-necessity hypothesis is unsupported.
- The neuron-only explanation path is insufficient for the main question.
- The next phase must analyze dataset geometry, attention scores, QK/OV structure, path margins, and gradient geometry.

## Internal Casual Notes: Geometry Results, Superposition, And Better Research Plan

Status: internal notes only. Do not copy this section to the public docs page until the claims are cleaned up and cross-checked.

Date: 2026-04-13.

### What The Project Is Really Studying Now

The research question is no longer just:

```text
Which heads, MLPs, features, or neurons matter?
```

The actual question is:

```text
Given the data relation d(x, y), why does SGD build one internal algorithm/circuit
rather than another, and how does that algorithm become represented in the model?
```

For this benchmark:

```text
d(x, y) = 1 if y is the value from the latest previous W K V event
          whose key K matches the current R K query.
```

So the abstract algorithm is:

```text
read query key
find latest matching write for that key
extract its value
write that value toward the output logits
```

The current evidence says the model does learn a real version of this, but not as a clean isolated circuit. It learns a dense, mixed retrieval infrastructure.

### Current Hierarchy Of Findings

#### 1. Dataset / Task Level

Artifacts:

- `data/generated/symbolic_kv_stream_learnability/metadata.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/dataset_geometry/dataset_geometry_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/dataset_geometry/dataset_geometry_report.md`

Supported:

- The stream benchmark is the right task object for this repo.
- It uses plain autoregressive next-token prediction.
- There is no answer mask and no classifier head.
- Read answers appear throughout the sequence, not only at one terminal answer site.
- The task relation is explicit enough to define a mathematical target.

The benchmark checks already rule out several easy shortcut explanations:

- exact sequence overlap across splits: `0`
- latent program overlap across splits: `0`
- heldout leakage outside heldout split: `0`
- trivial heuristics are weak:
  - `first_value_for_key = 0.0`
  - `last_value_before_query = 0.0`
  - strongest `most_frequent_value_before_query ~= 0.146`

Interpretation:

The dataset is suitable for circuit-formation research. It does not guarantee a unique circuit, but it gives a clear target relation `d(x, y)` to measure against.

#### 2. Behavior / Training Level

Artifacts:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/checkpoint_metrics.jsonl`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/checkpoint_metrics_summary.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/birth_window_analysis.json`

Main formation windows:

| window | role |
| --- | --- |
| `1500-2000` | first usable behavior / early birth |
| `4250-4750` | heldout consolidation |
| `7500-8000` | late upper-layer reorganization |

Sweep-level triggers:

- top answer gain step: `1750`
- top heldout gain step: `4500`
- top `Q` gain step: `7750`

Recent prompt-neuron trace baseline at selected checkpoints:

| step | mean margin | accuracy |
| ---: | ---: | ---: |
| `1750` | `-1.434105` | `0.326440` |
| `2500` | `-1.031744` | `0.364845` |
| `4500` | `5.123530` | `0.685377` |
| `16000` | `8.388601` | `0.776957` |

Split behavior at `16000`:

| split | margin | accuracy |
| --- | ---: | ---: |
| `validation_iid` | `15.557823` | `0.941176` |
| `heldout_pairs` | `10.634156` | `0.888889` |
| `structural_ood` | `-2.853920` | `0.470046` |
| `counterfactual` | `14.876693` | `0.935065` |

Interpretation:

The model has real IID and heldout-pair retrieval ability. Structural OOD remains weak. This means the learned mechanism generalizes across heldout pairs but has not become a fully robust symbolic algorithm.

#### 3. Residual / Stage Level

Artifacts:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/path_logit_decomposition/path_logit_decomposition_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/path_logit_decomposition/path_logit_stage_rows.jsonl`

Final stage readout at `16000`:

| stage | readout margin | readout accuracy |
| --- | ---: | ---: |
| `embedding` | `-26.757886` | `0.000000` |
| `layer_0_post_attn` | `-22.793784` | `0.000000` |
| `layer_0_post_mlp` | `-18.739200` | `0.011817` |
| `layer_1_post_attn` | `-12.012117` | `0.084195` |
| `layer_1_post_mlp` | `-12.394695` | `0.094535` |
| `layer_2_post_attn` | `-2.345797` | `0.449040` |
| `layer_2_post_mlp` | `8.388601` | `0.776957` |
| `final_norm` | `8.388601` | `0.776957` |

Interpretation:

The answer is not linearly available early. The representation becomes behaviorally usable only after layer 2 attention and especially after layer 2 MLP. This supports a staged hierarchy:

```text
lower layers: scaffold / key-value representation
middle heads: retrieval preparation
upper attention: value routing
upper MLP/final readout: answer write/readout
```

#### 4. Component Level

Artifacts:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/path_logit_decomposition/path_logit_decomposition_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/attention_geometry/attention_geometry_trace_report.json`

Strong final causal ablations at `16000`:

| component | ablated accuracy | accuracy drop | margin drop | DLA mean |
| --- | ---: | ---: | ---: | ---: |
| `L0MLP` | `0.044313` | `0.732644` | `25.374873` | `-2.037040` |
| `L2H1` | `0.274742` | `0.502216` | `15.398116` | `3.761759` |
| `L1H2` | `0.573117` | `0.203840` | `8.282414` | `2.527027` |
| `L1MLP` | `0.508124` | `0.268833` | `7.168099` | `-2.320465` |
| `L0H0` | `0.729690` | `0.047267` | `3.539790` | `1.196879` |

Important interpretation:

- `L2H1` has strong positive direct logit attribution and strong causal ablation effect.
- `L1H2` also has positive DLA and causal effect.
- `L0MLP` and `L1MLP` are causally essential despite negative mean DLA.

This means MLPs are not simply direct answer writers. They are likely shaping the representation that later attention/readout uses. DLA alone is not enough to explain them.

#### 5. Attention Geometry Level

Artifacts:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/attention_geometry/attention_geometry_trace_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/attention_geometry/attention_geometry_trace_rows.jsonl`

At final step `16000`, L2H1 is the clearest mature retrieval head:

| metric | L2H1 value |
| --- | ---: |
| support-value attention mean | `0.794394` |
| support-value QK margin mean | `0.657993` |
| attended OV value margin mean | `1.610091` |
| attention entropy mean | `0.364977` |
| OV output value-subspace alignment | `0.993541` |

First positive joint geometry:

| head | first step | interpretation |
| --- | ---: | --- |
| `L0H0` | `1750` | early bootstrap retrieval/scaffold |
| `L2H1` | `5250` | mature upper retrieval/write head |
| `L1H2` | `5500` | mature retrieval/preparation head |

Interpretation:

The attention mechanism is not just "some head attends there." L2H1 has the full QK/OV signature:

```text
QK: can separate support value from distractors
attention: places mass on the support value
OV: writes useful value information toward the answer direction
```

#### 6. Geometry Intervention Level

Artifacts:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/geometry_interventions/key_query_remove_final/geometry_subspace_intervention_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/geometry_interventions/l2h1_qk_key_remove_final/geometry_subspace_intervention_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/geometry_interventions/l2h1_ov_output_remove_final/geometry_subspace_intervention_report.json`

These are causal subspace interventions, not observational metrics.

The operation was:

```text
remove: z' = z - (z B) B^T
```

where `B` is a selected rank-4 geometric basis.

Final results at `16000`:

| intervention | baseline acc | intervened acc | acc drop | baseline margin | intervened margin | margin drop | positive drop frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| remove embedding key identity at query key | `0.776957` | `0.576071` | `0.200886` | `8.388601` | `0.019758` | `8.368843` | `0.695716` |
| remove `L2H1` QK key-side subspace at support value | `0.776957` | `0.549483` | `0.227474` | `8.388601` | `-0.096333` | `8.484934` | `0.846381` |
| remove `L2H1` OV output subspace at prediction | `0.776957` | `0.695716` | `0.081241` | `8.388601` | `4.494854` | `3.893747` | `0.776957` |

Heldout-specific geometry result:

```text
L2H1 QK key-side removal on heldout_pairs:
baseline accuracy     0.888889
intervened accuracy   0.620915
accuracy drop         0.267974
margin drop           8.198912
positive drop fraction 0.843137
```

Interpretation:

The L2H1 QK key-side subspace is causally important for generalizing retrieval. This is one of the strongest pieces of evidence so far.

Important caveat:

This subspace is necessary-ish, but not proven sufficient. Removing it damages behavior. We have not yet proven that keeping only it preserves behavior.

#### 7. Shared Feature / Family Level

Artifacts:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/shared_features/layer_2_post_mlp`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/mechanism_report/candidate_mechanism_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/birth_model/candidate_birth_model_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/coalition_map_early/candidate_coalition_map_report.json`

Shared feature basis quality:

| stage | features | explained variance | active fraction | reconstruction loss |
| --- | ---: | ---: | ---: | ---: |
| `layer_2_post_mlp` | `64` | `0.745779` | `0.541097` | `0.254670` |
| `final_norm` | `64` | `0.731190` | `0.538351` | `0.269171` |

Important caveat:

These feature IDs are analysis coordinates, not proven natural mechanistic atoms. The basis is too dense for clean semantic claims.

Family7 vs family4:

| candidate | family | feature IDs | useful birth | sum useful delta | sum heldout gap delta | status |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `layer2_family7_top2` | `7` | `27,54` | `2250` | `0.408211` | `0.196319` | `sgd_supported_generalizing_candidate` |
| `layer2_family4_top2` | `4` | `1,59` | `2500` | `0.234053` | `0.021933` | `sgd_supported_generalizing_candidate` |

Pairwise relation:

- score correlation: `0.766310`
- useful correlation: `0.606233`
- score sign conflict fraction: `0.238095`
- simultaneous useful gain fraction: `0.285714`
- family7 useful win fraction: `0.555556`

Birth model failure:

The birth model predicted family4 first, but actual useful birth was family7 first.

| candidate | predicted rank | actual rank | actual birth step | birth score |
| --- | ---: | ---: | ---: | ---: |
| family4 top2 | `1` | `2` | `2500` | `4.0` |
| family7 top2 | `2` | `1` | `2250` | `0.0` |

Why this matters:

The model used raw activation support, amplification, feature-score drive, and aggregate gradient alignment. That favored family4. But family7 had the better generalizing/heldout signal. The missing factor is likely heldout/path-specific gradient alignment, not raw family amplification.

#### 8. Coalition / Neuron Level

Artifacts:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/coalition_map_early/candidate_coalition_map_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/neuron_intervention_early_step2500/candidate_neuron_intervention_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/prompt_neuron_trace_probe/prompt_neuron_trace_report.json`

Coalition result:

Family7 and family4 are not separate sparse neuron circuits. They share many early layer-0 neurons.

Top shared-positive early neurons:

```text
L0N376, L0N302, L0N124, L0N96, L0N36, L0N488, L0N411, L0N326
```

Shared-negative early neurons:

```text
L0N261, L0N504, L0N332, L0N301, L0N458, L0N131, L0N70, L0N416
```

Conflict neurons include:

```text
L2N477, L2N310, L2N340, L2N281, L2N17, L2N185, L0N28, L2N41
```

Early neuron intervention at step `2500`:

- baseline answer accuracy: `0.364845`
- baseline heldout accuracy: `0.104575`
- shared-positive neuron ablations changed family scores, but behavior was mostly compensated
- some "positive" neurons increased candidate scores when ablated

Prompt-neuron trace at `16000`:

Top absolute-DLA neurons are mostly layer 2:

```text
L2N180, L2N121, L2N477, L2N372, L2N39, L2N164, L2N156, L2N96, ...
```

Final neuron ablation examples:

| neuron | DLA mean | abs DLA mean | margin drop | accuracy drop |
| --- | ---: | ---: | ---: | ---: |
| `L2N477` | `0.140675` | `0.903545` | `0.353949` | `0.002954` |
| `L1N366` | `-0.057806` | `0.698906` | `0.192114` | `-0.004431` |
| `L2N180` | `-0.058170` | `1.063912` | `0.192028` | `0.004431` |
| `L2N121` | `-0.106536` | `1.057197` | `0.180531` | `0.002954` |
| `L1N401` | `-0.073953` | `0.656371` | `0.172617` | `-0.005908` |

Interpretation:

Neuron-level effects are real but small compared with component and subspace effects. The sign mismatch between DLA and ablation is not noise; it is evidence that neurons are mixed carriers.

Top-neuron overlap result:

All-prompt overlap is low, but overlap rises for same key/value conditions.

Examples at final step:

- layer 2 DLA top-neuron overlap:
  - all pairs: `0.02949`
  - same answer value: `0.07106`
  - same key-value pair: `0.10183`
- layer 1 activation top-neuron overlap:
  - all pairs: `0.08216`
  - same query key: `0.12414`
  - same key-value pair: `0.17396`

Interpretation:

The model does not use a single universal top-neuron set. The active neuron coalition is prompt-conditioned.

### What The Superposition Problem Means Here

Simple version:

```text
The model uses the same neurons and directions to carry several partially overlapping features.
```

So a neuron or subspace can:

- help one prompt
- hurt another prompt
- support family7 and family4 at the same time
- have negative average DLA but positive causal importance
- look important in one basis but not in another

This is exactly what the current results show.

Superposition is visible at three levels:

#### 1. Feature Basis Superposition

The shared-feature basis is dense:

- explained variance is decent but not near-complete
- active fraction is high, about `0.54`
- feature IDs are not clean semantic atoms

So family7/family4 are useful coordinates but not final mechanistic units.

#### 2. Neuron Superposition

Single neurons have mixed signs:

- some top DLA neurons have negative average DLA but positive ablation drop
- some ablations improve accuracy on some splits
- single-neuron ablations are much weaker than component/subspace ablations

So neurons are not reliable primitive units for the final explanation.

#### 3. Geometry-Level Superposition

Even model-intrinsic QK/OV subspaces are mixed.

Per-query signs for final geometry interventions:

| intervention | positive drops | negative drops | interpretation |
| --- | ---: | ---: | --- |
| key-query identity removal | `471/677` | `206/677` | key identity usually helps, but not uniformly |
| L2H1 QK key-side removal | `573/677` | `104/677` | strongest causal geometry, still mixed |
| L2H1 OV output removal | `526/677` | `151/677` | value-writing direction is also mixed |

Correlations between intervention effects over prompts are weak:

| pair | Pearson correlation | same-sign fraction |
| --- | ---: | ---: |
| key-query vs L2H1 QK | `0.2235` | `0.6721` |
| key-query vs L2H1 OV | `0.1214` | `0.6647` |
| L2H1 QK vs L2H1 OV | `0.1370` | `0.7149` |

Interpretation:

The circuit is not one clean line of computation. The retrieval infrastructure is real and causal, but it is multiplexed with other prompt-conditioned signals.

### Current Best Mechanistic Story

The current best hierarchy is:

```text
dataset relation d(x, y)
  -> key/value identity structure in embeddings
  -> L0MLP builds or stabilizes residual coordinates
  -> L0H0 participates in early bootstrap retrieval
  -> L1H2 becomes a mid-layer retrieval/preparation head
  -> L2H1 becomes the clearest mature support-value retriever/writer
  -> L2MLP/final_norm make the answer readable
  -> neurons implement this through dense prompt-conditioned coalitions
```

The strongest current claim:

```text
L2H1 QK key-side geometry is causally important for final retrieval,
including heldout-pair retrieval.
```

The strongest current limitation:

```text
This geometry is not clean or sufficient by itself.
It is embedded inside a dense MLP/residual infrastructure.
```

### Why Observation And Intervention Are Still Not Enough

Observation answers:

```text
what changed?
```

Causal ablation answers:

```text
what breaks if we remove this?
```

Geometry intervention answers:

```text
does this vector subspace carry necessary information?
```

But the real research question asks:

```text
why did SGD create this representation instead of another one?
```

To answer that, we need a training-dynamics explanation:

```text
which candidate path receives reinforcing gradient pressure,
which path generalizes across examples,
which path has lower interference,
and which path becomes self-stabilizing during training?
```

### Better Research Plan From Here

Do not start with more neurons. Do not start with more open-ended reports.

The better plan is:

#### Stage A: Define The Abstract Algorithm Precisely

Write the symbolic causal variables:

```text
K_query(x)       = key in the current read
K_support(x)     = key in the latest matching write
V_support(x)     = value in the latest matching write
D_key(x)         = distractor keys
D_value(x)       = distractor values
y(x)             = correct answer value
```

The model explanation must implement:

```text
K_query == K_support  ->  select support position  ->  write V_support  ->  output y
```

This gives ground truth variables independent of any head, neuron, or SAE feature.

#### Stage B: Prove The Final Algorithm Before Explaining Birth

For the final model, the goal is a causal abstraction:

```text
abstract variable  ->  model subspace/path  ->  output behavior
```

The proof standard should be:

- remove the variable/subspace and behavior breaks
- keep only the variable/subspace and enough behavior remains
- patch the variable from another example and the output changes predictably
- the result holds on heldout pairs, not only IID examples
- the result survives prompt-level analysis, not only aggregate averages

This is stricter than current geometry intervention. Current geometry results show necessity. They do not yet show sufficiency or clean causal abstraction.

#### Stage C: Decompose The Mechanism Into Route And Content

Use the transformer-circuits split:

```text
QK = routing geometry
OV = content/write geometry
MLP = nonlinear residual infrastructure
```

For this task:

```text
QK should explain where the model looks.
OV should explain what value information gets written.
MLPs should explain how residual coordinates are made usable.
```

Current evidence points to:

```text
L2H1 QK: strongest final routing geometry
L2H1 OV: meaningful but weaker value-writing geometry
L0MLP/L1MLP: essential support infrastructure
L2MLP/final_norm: final readout and calibration
```

#### Stage D: Treat Superposition As A Measured Object

Do not try to "avoid" superposition. Measure it.

Define interference for a subspace or path:

```text
I(P) = fraction or magnitude of prompts where removing P improves the margin
```

For current interventions:

```text
I(key_query_identity) = 206 / 677 = 0.304
I(L2H1_QK_key)        = 104 / 677 = 0.154
I(L2H1_OV_output)     = 151 / 677 = 0.223
```

This tells us:

```text
L2H1 QK is the cleanest current geometric object,
but it is still not monosemantic.
```

Future explanations should include both:

```text
useful signal strength
interference cost
```

#### Stage E: Move From Components To Path Variables

Define path contribution:

```text
m_t(x, y) = logit_t(y | x) - max_{z != y} logit_t(z | x)
```

Then decompose:

```text
m_t(x, y) ~= sum_P C_P(theta_t, x, y) + residual_error
```

A path `P` might be:

```text
embedding key direction -> L1H2 -> L2H1 QK/OV -> L2MLP -> unembed
```

The key is that `C_P` must be a causal/path-level object, not a feature-family score.

#### Stage F: Explain SGD Selection With Gradient Alignment

The mathematical target remains:

```text
Delta C_P ~= -eta * <grad_theta L, grad_theta C_P>
```

Circuit `P` wins over `Q` when:

```text
E_D[< -grad_theta L, grad_theta C_P >] - I(P)
>
E_D[< -grad_theta L, grad_theta C_Q >] - I(Q)
```

where:

- `C_P` is the path contribution to the correct margin
- `I(P)` is interference/superposition cost
- `D` must be split into train, heldout, and structural OOD groups

This is the route from mechanistic analysis to a mathematical explanation of circuit formation.

#### Stage G: Trace Birth Only After The Final Mechanism Is Proven

Once the final mechanism is proven, trace it backward:

```text
when does key identity become usable?
when does QK routing become positive?
when does OV write become value-aligned?
when does L2MLP turn the path into positive margin?
when does heldout alignment separate from IID amplification?
```

This prevents the earlier mistake:

```text
feature families first -> post-hoc birth model -> wrong separating factor
```

The new order should be:

```text
final causal algorithm
  -> path variables
  -> training trajectories
  -> gradient alignment
  -> cross-seed / factor tests
```

### Simple Plan To Tackle Superposition

The simple version:

1. Stop asking whether one neuron or one feature is "the circuit."
2. Ask what information must be carried: query key, support match, support value, answer direction.
3. Find the model subspaces that carry each information type.
4. Test each subspace with remove, keep, and patch interventions.
5. Measure how often each subspace helps vs hurts across prompts.
6. Split mixed subspaces by prompt condition: key, value, key-value pair, split, success/failure.
7. Only then map the subspace back down to neurons and weights.

In short:

```text
information variable -> causal subspace -> path contribution -> neuron implementation
```

not:

```text
neuron list -> guessed circuit
```

### What Full Reverse Engineering Would Mean Here

A real reverse-engineering result would need all of these:

#### Behavioral Equivalence

The proposed algorithm predicts the model's output on normal examples and counterfactual examples.

#### Causal Necessity

Removing the proposed route destroys the relevant behavior.

#### Causal Sufficiency

Keeping or patching the proposed route restores a large fraction of behavior.

#### Variable Alignment

The route encodes the right abstract variables:

```text
query key
support key/value
answer value
distractor separation
```

#### Training Dynamics

The same route can be tracked from birth to maturity over checkpoints.

#### SGD Explanation

The route's growth is explained by gradient alignment on data examples:

```text
train examples that support the true relation reinforce the path
shortcut examples reinforce competing paths
heldout-aligned paths survive better
interference controls which mixed direction wins
```

#### Cross-Seed Stability

The same abstract mechanism appears across seeds, even if exact head or neuron IDs change.

### Updated Claims

Supported:

- The benchmark is suitable for studying circuit formation.
- Circuit formation is staged.
- The final model uses a dense multi-component mechanism.
- `L0MLP`, `L1H2`, and `L2H1` are central components.
- `L2H1` has the clearest mature QK/OV retrieval geometry.
- Removing L2H1 QK key-side geometry causally damages final and heldout behavior.
- Neurons and feature families are mixed, not clean natural units.
- Superposition/interference exists at feature, neuron, and geometric subspace levels.

Partially supported:

- L0MLP probably builds/stabilizes residual coordinates used by later attention.
- L1H2 probably prepares retrieval for L2H1.
- L2MLP/final_norm probably convert routed value information into final answer margin.
- family7 looks more generalizing than family4, but feature-family basis limitations remain.

Unsupported:

- a complete circuit decomposition
- a clean monosemantic feature basis
- a sufficient causal abstraction of the algorithm
- a mathematical proof of why SGD selected this circuit
- cross-seed equivalence
- per-minibatch update-level explanation

### Current Research North Star

The north star is:

```text
Explain how SGD transforms the dataset relation d(x, y)
into a causal path through residual geometry,
and why that path wins over alternatives under gradient pressure and superposition.
```

The current best target equation:

```text
P wins over Q if:

E_D[< -grad_theta L(theta_t; x, y), grad_theta C_P(theta_t; x, y) >]
  - interference(P)
>
E_D[< -grad_theta L(theta_t; x, y), grad_theta C_Q(theta_t; x, y) >]
  - interference(Q)
```

where:

```text
C_P(theta_t, x, y)
```

is the causal contribution of path `P` to the correct answer margin.

This is not solved yet. But the current artifacts now point to the right object:

```text
not a neuron,
not a feature family,
not just a head,
but a causally validated path through QK/OV/residual/MLP geometry.
```

### Research References To Keep In Mind

- Transformer Circuits: decompose attention into QK routing and OV writing, not just attention maps. Reference: `https://transformer-circuits.pub/2021/framework/index.html`
- Induction Head formation: circuits can appear during training and align with measurable progress. Reference: `https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html`
- Toy Models of Superposition: features can be represented in overlapping directions when capacity is limited. Reference: `https://transformer-circuits.pub/2022/toy_model/index.html`
- Towards Monosemanticity: dictionary learning can help separate features, but learned features still need causal validation. Reference: `https://transformer-circuits.pub/2023/monosemantic-features/index.html`
- Progress Measures for Grokking: a good final story needs an algorithm, progress measures, and causal validation. Reference: `https://arxiv.org/abs/2301.05217`
- ACDC: circuit discovery requires choosing dataset, metric, patching unit, and causal graph together. Reference: `https://openreview.net/forum?id=89ia77nZ8u`
- Causal abstraction: an explanation should be an abstract algorithm faithful under interventions, not just a list of active parts. Reference: `https://jmlr.org/papers/v26/23-0058.html`
