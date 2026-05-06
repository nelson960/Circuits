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

## Heldout Route-Comparison Result

Date: 2026-04-14

This note records the first clean route-comparison pass on the controlled heldout query-key variable. This should be treated as a candidate-route finding, not a final proof of SGD selection.

Artifacts:

- `artifacts/runs/symbolic_kv_reference_formation/analysis/route_gradient_selection/heldout_route_comparison/full_residual_query_key/candidate_route_gradient_selection_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/route_gradient_selection/heldout_route_comparison/l2h1_qk_query_query_key/candidate_route_gradient_selection_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/route_gradient_selection/heldout_route_comparison/l2h1_qk_key_query_key/candidate_route_gradient_selection_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/route_gradient_selection/heldout_route_comparison/l2h1_ov_input_query_key/candidate_route_gradient_selection_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/route_gradient_selection/heldout_route_comparison/l2h1_ov_output_query_key/candidate_route_gradient_selection_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/route_gradient_selection/heldout_route_comparison/l0h1_qk_query_query_key/candidate_route_gradient_selection_report.json`
- `artifacts/runs/symbolic_kv_reference_formation/analysis/route_gradient_selection/heldout_route_comparison/l0h3_qk_query_query_key/candidate_route_gradient_selection_report.json`

Run sanity:

```text
split_filter = heldout_pairs
checkpoints = 64 / 64 complete
pair types = query_key, distractor
constructed pairs = 64 query_key + 64 distractor
skip reasons = none
zero route-gradient parameters = 0
```

There is one provenance caveat. A clean rerun of `full_residual_query_key` also exists under:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/route_gradient_selection/user_heldout_route_comparison/full_residual_query_key/
```

The other route reports are complete and use the correct split, but they live in the older `heldout_route_comparison` directory. If we want strict user-run-only provenance, rerun those six into `user_heldout_route_comparison` before using them in public-facing text.

### What Was Tested

The controlled variable was:

```text
query key
```

The pair construction used two pair types:

```text
query_key:
  the queried key changes, so the correct answer should change

distractor:
  an unqueried/distractor value changes, so the correct answer should not change
```

This matters because a route can look important just because it encodes general prompt structure. The distractor control asks whether the route specifically carries the query-key variable instead of merely carrying "this prompt has a read event" or "this is a similar sequence".

The tested candidate routes were:

```text
full_residual at layer_1_post_mlp
L2H1 QK query-side, rank 4
L2H1 QK key-side, rank 4
L2H1 OV input-side, rank 4
L2H1 OV output-side, rank 4
L0H1 QK query-side, rank 4
L0H3 QK query-side, rank 4
```

### Calculation Definitions

For each route `P`, the route-transfer score is:

```text
transfer_P =
  patched_margin_P - corrupted_margin
```

The full residual gives the reference transfer:

```text
transfer_full =
  clean_margin - corrupted_margin
```

The recovery fraction is:

```text
recovery_P =
  transfer_P / transfer_full
```

So if a route has recovery near `1`, patching only that route almost fully moves the model from the corrupted answer behavior back to the clean answer behavior. If recovery is near `0`, the route does not carry much of the transferable variable by itself.

The route-gradient support is:

```text
support_P(t) =
  < -grad_theta L(theta_t), grad_theta C_P(theta_t) >
```

where `C_P` is the candidate route score. The first-order SGD-predicted route-score change is:

```text
linearized_delta_P(t) =
  learning_rate_t * support_P(t)
```

Positive support means the current loss gradient would increase that route score under a first-order SGD approximation. Negative support means the current loss gradient would suppress that route score.

This is still not final proof. Final proof requires comparing the first-order prediction to actual checkpoint-to-checkpoint parameter changes:

```text
actual_delta_P(t) =
  C_P(theta_{t+1}) - C_P(theta_t)

predicted_delta_P(t) =
  grad_theta C_P(theta_t)^T (theta_{t+1} - theta_t)

remainder_P(t) =
  actual_delta_P(t) - predicted_delta_P(t)
```

The route-gradient-selection result only gives a candidate selection signal. The next proof tool must use the actual checkpoint delta `theta_{t+1} - theta_t`.

### Final Checkpoint Route Table

All numbers below are from step `16000`, split `heldout_pairs`, pair type `query_key` unless stated otherwise.

| route | query transfer | recovery | distractor transfer | SGD support | linearized delta | read |
|---|---:|---:|---:|---:|---:|---|
| full residual | `40.583275` | `100.00%` | `0.864908` | `-142.891520` | `-0.057157` | full residual contains essentially all transferable query-key information |
| `L2H1 QK query` | `10.542538` | `25.98%` | `-0.317474` | `25.931873` | `0.010373` | strongest tested rank-4 specific carrier |
| `L0H3 QK query` | `1.699248` | `4.19%` | `-0.021947` | `-151.932807` | `-0.060773` | weak direct carrier at this stage |
| `L2H1 OV output` | `1.575836` | `3.88%` | `-0.141499` | `-16.931023` | `-0.006772` | weak for the query-key variable |
| `L0H1 QK query` | `1.416969` | `3.49%` | `0.269435` | `-82.336787` | `-0.032935` | weak direct carrier at this stage |
| `L2H1 QK key` | `1.387651` | `3.42%` | `-0.025758` | `131.842206` | `0.052737` | strongly gradient-supported, but weak current transfer |
| `L2H1 OV input` | `0.545343` | `1.34%` | `-0.083717` | `-22.876149` | `-0.009150` | weakest tested route |

The main concrete calculation is:

```text
full residual:
  clean margin     = 35.709450
  corrupted margin = -4.873827
  patched margin   = 35.709450

  transfer_full = 35.709450 - (-4.873827)
                = 40.583277

L2H1 QK query:
  patched margin = 5.668711

  transfer_L2H1_QK_query = 5.668711 - (-4.873827)
                          = 10.542538

  recovery = 10.542538 / 40.583277
           = 0.2598
           = 25.98%
```

The distractor control for the same route is:

```text
L2H1 QK query distractor:
  corrupted margin = 1.396942
  patched margin   = 1.079468

  distractor transfer = 1.079468 - 1.396942
                      = -0.317474
```

That is good for specificity: the route transfers a large amount on query-key-change pairs and does not transfer much on distractor pairs.

The SGD support calculation for the same route is:

```text
support = < -grad L, grad C_P >
        = 25.931873

learning rate = 0.0004

linearized delta = 0.0004 * 25.931873
                 = 0.010373
```

So at step `16000`, the heldout query-key loss gradient would still increase this route score under the local first-order approximation.

### Simple Reading

The full residual at `layer_1_post_mlp` contains the query-key variable very strongly. Patching the whole residual recovers almost exactly the full clean-vs-corrupt margin difference.

The best tested small route is `L2H1 QK query-side`. It recovers about one quarter of the full residual query-key transfer by itself:

```text
10.542538 / 40.583277 = 25.98%
```

This is much larger than the other tested rank-4 routes. It also has a good distractor control:

```text
query transfer      = 10.542538
distractor transfer = -0.317474
```

So this route is not merely encoding arbitrary prompt structure. It is much more aligned with the query-key variable than with distractor changes.

The result also shows the mechanism is still dense. The best rank-4 route only explains about `26%` of the full residual transfer. That means most of the heldout query-key information is distributed outside this one small subspace.

### Important Twist

`L2H1 QK key-side` has the strongest positive SGD support:

```text
support = 131.842206
linearized delta = 0.052737
```

but it has weak current transfer:

```text
query transfer = 1.387651
recovery = 3.42%
```

This separates two different concepts:

```text
current causal content:
  does the route already carry the query-key variable?

gradient pressure:
  would the current heldout loss push this route upward?
```

`L2H1 QK query-side` currently carries much more transferable query-key information. `L2H1 QK key-side` receives stronger gradient pressure, but currently transfers little. We cannot collapse these into one claim.

### Timeline Reading

The broad residual route forms early and then appears saturated or suppressed later:

| route | step | query transfer | SGD support | linearized delta |
|---|---:|---:|---:|---:|
| full residual | `4500` | `-2.168145` | `266.560682` | `0.106624` |
| full residual | `8000` | `26.891529` | `39.575928` | `0.015830` |
| full residual | `12000` | `35.679001` | `-201.509556` | `-0.080604` |
| full residual | `16000` | `40.583275` | `-142.891520` | `-0.057157` |

This says the residual-level variable receives strong positive pressure during formation, becomes large by `8000`, and later receives negative local gradient pressure. That does not mean it disappears. It means the current loss gradient no longer wants to increase this already-large broad route.

`L2H1 QK query-side` keeps becoming more visible:

| route | step | query transfer | SGD support | linearized delta |
|---|---:|---:|---:|---:|
| `L2H1 QK query` | `4500` | `0.808970` | `2.142976` | `0.000857` |
| `L2H1 QK query` | `8000` | `2.388395` | `10.357577` | `0.004143` |
| `L2H1 QK query` | `12000` | `4.186921` | `-37.730755` | `-0.015092` |
| `L2H1 QK query` | `16000` | `10.542538` | `25.931873` | `0.010373` |

This says the route is already present by `4500`, grows by `8000`, and is much stronger at `16000`. The sign of local support is not monotonic, which is another reason not to overclaim from a single checkpoint.

### What This Supports

Supported as a candidate finding:

- The heldout query-key variable is causally present in the `layer_1_post_mlp` residual stream.
- Among the tested rank-4 head-derived routes, `L2H1 QK query-side` is the strongest current carrier of transferable heldout query-key information.
- `L2H1 QK query-side` has a good distractor control: large query-key transfer and near-zero distractor transfer.
- The mechanism is distributed: the best tested rank-4 route recovers only about `26%` of the full residual transfer.
- Current causal content and gradient pressure can diverge: `L2H1 QK key-side` has strong positive support but weak current transfer.

Not supported yet:

- SGD selected `L2H1 QK query-side` over all alternatives.
- The route comparison is a mathematical proof of circuit birth.
- The tested rank-4 routes cover the full mechanism.
- Head routes alone explain the dense MLP/residual infrastructure.
- The conclusion is seed-stable.

### How We Got Here

The earlier feature-family and coalition tools showed that the model's internal story is dense:

```text
family7 / family4 are mixed analysis coordinates
MLP neurons are shared across families
positive update alignment and causal carrying can disagree
feature-family birth did not give a clean why-answer
```

That forced the pivot from "which feature family was born?" to "which abstract variable is causally carried by which route?"

The current route-comparison run is the first controlled version of that pivot. Instead of asking only whether a component matters, it asks:

```text
If the queried key changes, can this route transfer the corresponding answer behavior?
If only a distractor changes, does this route stay mostly silent?
Does the current loss gradient push this route up or down?
```

This is better than raw component observation, but it is still not enough for the final why-question.

### What Is Still Missing

The next missing proof step is actual checkpoint update attribution.

For each interval:

```text
theta_t -> theta_{t+1}
```

we need:

```text
actual_delta_P =
  C_P(theta_{t+1}) - C_P(theta_t)

linearized_checkpoint_delta_P =
  grad_theta C_P(theta_t)^T (theta_{t+1} - theta_t)

remainder_P =
  actual_delta_P - linearized_checkpoint_delta_P
```

Then decompose:

```text
linearized_checkpoint_delta_P
```

by:

```text
embedding
L0 attention Q/K/V/O
L0 MLP neurons
L1 attention Q/K/V/O
L1 MLP neurons
L2 attention Q/K/V/O
L2 MLP neurons
layernorms
unembedding
```

That is the mathematical bridge from observation to proof. It will tell us whether the actual training update increased the route, which parameter groups caused the increase, and whether the first-order calculation explains the real checkpoint-to-checkpoint change.

The final route-selection claim should have this form:

```text
Route P is selected over route Q during interval [t0, t1] if:

sum_t grad C_P(theta_t)^T (theta_{t+1} - theta_t)
>
sum_t grad C_Q(theta_t)^T (theta_{t+1} - theta_t)

and this predicted advantage matches:

sum_t [C_P(theta_{t+1}) - C_P(theta_t)]
>
sum_t [C_Q(theta_{t+1}) - C_Q(theta_t)]

with small enough residual error.
```

Only after this update-level calculation should we claim that SGD selected one route over another.

### Immediate Next Research Step

Build or run the next tool only after the route-comparison artifacts are accepted as the candidate-selection layer:

```text
checkpoint-update-attribution
```

Inputs:

```text
candidate routes:
  full_residual
  L2H1 QK query-side
  L2H1 QK key-side
  strongest upstream controls

checkpoint intervals:
  all adjacent checkpoints, especially 4500 -> 8000 and 8000 -> 16000
```

Outputs:

```text
actual route-score change
linearized checkpoint-delta prediction
prediction error / remainder
parameter-group contribution table
attention Q/K/V/O contribution table
MLP neuron contribution table
route competition table
```

This is the next step toward the real proof:

```text
how the data relation d(x, y) becomes residual/attention geometry,
and how actual SGD updates move the model toward one route more than another.
```

## Current Experimental Position In Simple Words

Date: 2026-04-14

We are no longer only asking:

```text
which heads, neurons, or feature families changed?
```

We are now asking:

```text
what variable is being carried, where is it carried, and did training actually
move the model in a way that builds that variable?
```

The current strongest concrete result is:

```text
The heldout query-key variable is present in the layer_1_post_mlp residual stream.
The full residual patch recovers almost all of it.
The best tested small route is L2H1 QK query-side, which recovers about 26%
of the full residual query-key transfer.
```

Simple meaning:

```text
The model does have a transferable internal variable for "which key is being queried".
That variable is not stored in one clean head or one clean neuron.
Part of it is visible in L2H1 QK query-side geometry, but most of it is distributed
through the residual stream and upstream components.
```

So the current position is:

```text
we have found a serious candidate route,
but we have not yet proven how SGD built it.
```

### What The Earlier Neuron/Family Work Means Now

The earlier feature-family and coalition results should not be read as:

```text
family7 is the circuit
family4 is the losing circuit
these neurons intentionally prepare the later circuit
```

The better interpretation is:

```text
feature families were useful analysis coordinates,
but they were not clean mechanistic atoms.
```

The neuron coalition result told us that the model is dense:

```text
many neurons participate in more than one family
positive update alignment and causal contribution can disagree
some neurons look helpful in one projection and harmful in another
```

Simple explanation:

```text
The model is not building one neat labeled part at a time.
It is changing many weights together.
Some early neurons become useful because their activation pattern reduces loss.
Later, when an attention route becomes useful, gradient flows backward through that route.
Then some earlier neurons receive pressure to shape the residual stream in ways that
make the later route work better.
```

Important wording:

```text
Early neurons do not intentionally prepare a later circuit.
They get reinforced when their activation pattern helps reduce loss.
Later, once an attention route becomes useful, backprop sends credit through that route,
and some earlier neurons start receiving gradients that shape the residual geometry
needed by the later route.
```

This is the simple version of the "foundation then support" idea. It is plausible from the current results, especially because upstream `L0/L1` components repeatedly show up as important. But it is not fully proven yet.

### What The Heldout Route Result Adds

The route-comparison result improved the evidence because it used controlled pairs:

```text
query_key pairs:
  the queried key changes and the answer should change

distractor pairs:
  an irrelevant value changes and the answer should stay the same
```

This matters because a route can be generally important without carrying the right variable. The current `L2H1 QK query-side` route passes a stronger test:

```text
query-key transfer is large
distractor transfer is near zero
```

Simple meaning:

```text
this route is not just reacting to random prompt changes;
it is specifically sensitive to the queried-key variable.
```

But the result also shows the limit:

```text
full residual transfer = 40.58
L2H1 QK query transfer = 10.54
```

So:

```text
L2H1 QK query-side explains a meaningful piece,
not the whole mechanism.
```

### How Far We Are

Current level of evidence:

```text
behavioral evidence: strong
component evidence: strong
feature-family evidence: useful but not canonical
neuron coalition evidence: strong evidence of density/superposition
causal variable patch evidence: partial but important
route-comparison evidence: useful candidate-route selection
mathematical SGD proof: not done yet
cross-seed generality: not done yet
```

In plain terms:

```text
We know the model learned the task.
We know the solution is staged.
We know the final mechanism is dense.
We know L2H1 QK query-side carries a real part of the heldout query-key variable.
We know upstream MLP/attention components are probably shaping the residual space.
We do not yet know, with proof, why SGD built this route instead of another.
```

So the project is past raw observation, but not yet at proof.

The current best summary is:

```text
We have identified where to look for the proof.
We have not yet completed the proof.
```

### What The Next Proof Must Show

The next proof should be simple in purpose even if technical in implementation:

```text
Did the actual checkpoint update make this route stronger?
Which weights caused that increase?
Which data examples created the gradient pressure?
Did this route grow more than competing routes?
Did that route growth explain the behavior improvement?
```

The key object is:

```text
Delta theta =
  theta_{t+1} - theta_t
```

This is the actual parameter change between checkpoints. The next analysis should not only use the idealized gradient direction. It should use the real update that happened during training.

The central calculation is:

```text
route growth =
  C_P(theta_{t+1}) - C_P(theta_t)

predicted route growth =
  grad C_P(theta_t) . Delta theta
```

If these match, then we can say:

```text
the actual training update mathematically explains this route's growth
```

If this works across the formation window and beats competing routes, then we can start saying:

```text
SGD built this route because the actual updates repeatedly increased it more
than the alternatives, under pressure from the task data relation.
```

That is the next real step from "candidate mechanism" toward "mathematical explanation".

## Full Picture After Actual-Update And Data-Update Attribution

Date: 2026-04-17

This is the current best story, written as a research status note, not as a final claim.

The project started by tracking behavior, heads, MLPs, shared features, feature families, neuron coalitions, and candidate birth models. That phase was useful because it showed the mechanism is dense. It also showed that the original feature-family basis is not a clean mechanistic unit by itself. Family and neuron reports helped locate where activity changes, but they did not answer why SGD builds one route rather than another.

The current pivot is:

```text
data relation -> actual parameter update -> residual/attention geometry -> route growth -> logit margin
```

The task relation is symbolic key-value lookup:

```text
given a stream of writes W K V and a read R K,
predict the value from the latest previous write for that key.
```

So the model must internalize something like:

```text
query key -> matching support event -> support value -> answer logit
```

### Current Strongest Mechanism Candidate

The strongest candidate route is still:

```text
stage: layer_1_post_mlp
subspace: L2H1 head_qk_query, rank 4
position role: query_key
```

Earlier causal-variable patching showed this route carries a real part of the query-key variable, but not all of it. The full residual carries much more. So `L2H1` is not the whole circuit. It is a useful visible route through a larger dense infrastructure.

The best simple interpretation is:

```text
L0/L1 components shape the residual stream.
L2H1 uses that shaped residual stream to route from the query-side representation
toward the relevant value-bearing token.
L2H1 then writes strongly into the answer direction.
L1H2 is another major direct writer/retriever.
L0MLP is causally essential but not a direct answer writer.
```

### Actual Checkpoint Update Evidence

The cleanest actual-update window is:

```text
step_005000 -> step_005250
```

For validation `query_key` causal pairs, the actual checkpoint movement explains route growth well:

```text
source route score: 2.894687
target route score: 3.736807
actual delta:       +0.842120
predicted delta:    +0.946138
relative error:      0.123520
sign match:          true
```

For validation `distractor` controls:

```text
source route score: 0.028467
target route score: 0.020617
actual delta:       -0.007850
predicted delta:    -0.006540
relative error:      0.166864
sign match:          true
```

Simple meaning:

```text
The real parameter update from 5000 to 5250 grew the query-key route.
The same update did not grow the distractor-control route.
```

That is one of the strongest pieces of evidence so far.

But the group decomposition says the update was not mainly inside `L2H1` itself. Top positive contributors to the query-key route growth were:

```text
L0 MLP              +0.310171
L1 attention        +0.243085
L0 attention        +0.215271
L1H3 qkvo           +0.200743
L0 out_proj         +0.159591
L1 MLP              +0.118495
L0H3 qkvo           +0.102137
L0H2 qkvo           +0.083088
L2H1 qkvo           +0.049206
```

Simple meaning:

```text
SGD did not just tune the final head.
It mostly changed upstream layers so the residual stream became easier for L2H1 to use.
```

### Attention Geometry: Key-Side Hypothesis Failed

We tested whether `L2H1` grows because it attends more strongly from the query key to the matching written key.

That did not hold.

For `5000 -> 5250`, support-key scores became worse:

```text
clean support_key score delta:     -0.488583
corrupted support_key score delta: -0.534208
```

Attention to support-key positions was also tiny:

```text
support_key attention: about 0.001 to 0.003
```

So the narrow hypothesis:

```text
L2H1 solves the task by query-key -> written-key routing
```

is not supported.

### Attention Geometry: Value-Side Route Looks Much Better

The value-side score decomposition is stronger.

For `5000 -> 5250`, `L2H1` support-value routing improved:

```text
clean support_value score delta:     +0.391587
corrupted support_value score delta: +0.273075
```

Value distractor scores decreased in the same early window:

```text
clean value_distractor score delta:     -0.119647
corrupted value_distractor score delta: -0.148914
```

Support-value attention also increased:

```text
clean support_value attention delta:     +0.018370
corrupted support_value attention delta: +0.022667
```

Simple meaning:

```text
The route growth is not mainly query -> key.
It is closer to query representation -> associated value-bearing token.
```

The attention geometry trace for `L2H1` supports this:

```text
step    support_value_qk_margin    support_value_attention    attended_ov_value_margin
5000   -0.057844                   0.717795                   1.335116
5250    0.009791                   0.740199                   1.576235
7500    0.382215                   0.787017                   2.449040
7750    0.424220                   0.791495                   2.436379
8000    0.419836                   0.788349                   2.474642
8250    0.571587                   0.787570                   2.490426
```

By `8250`, `L2H1` is the top head by:

```text
support_value_attention
support_value_qk_margin
attended_ov_value_margin
low attention entropy
```

So the current route-level mechanism is:

```text
L2H1 becomes a value-facing retrieval/write head.
```

Not:

```text
L2H1 simply matches query keys to written keys.
```

### Direct Logit Attribution And Ablation

Direct logit attribution says which components directly write in the correct answer direction.

For final positive direct components:

```text
L2H1   DLA mean +4.674101
L1H2   DLA mean +2.978323
L2MLP  DLA mean +1.541451
L0H0   DLA mean +0.856416
```

At `5250`, ablation confirms `L2H1` is load-bearing:

```text
L2H1 DLA mean:           +4.210372
L2H1 margin drop:        8.256493
L2H1 accuracy drop:      0.267356
baseline margin:         6.944736
ablated margin:         -1.311757
```

At `8000`, `L2H1` remains strongly causal:

```text
L2H1 DLA mean:           +4.695038
L2H1 margin drop:        11.473013
L2H1 accuracy drop:      0.360414
baseline margin:         8.728830
ablated margin:         -2.744183
```

`L1H2` is also major:

```text
5250 L1H2 DLA mean:      +3.918298
5250 L1H2 margin drop:   8.173424

8000 L1H2 DLA mean:      +3.031231
8000 L1H2 margin drop:   8.769184
```

`L0MLP` is the clearest dense-infrastructure result:

```text
5250 L0MLP DLA mean:     -6.964605
5250 L0MLP margin drop:  12.394523
5250 L0MLP accuracy drop 0.740030

8000 L0MLP DLA mean:     -3.247367
8000 L0MLP margin drop:  19.306082
8000 L0MLP accuracy drop 0.790251
```

Simple meaning:

```text
L0MLP is extremely important, but not because it directly writes the answer.
It is shaping or maintaining an internal state that later heads need.
```

This is why neuron-level and feature-family analysis felt dense and confusing:

```text
some components are necessary without being direct answer writers.
some direct writers are late.
some early components prepare geometry rather than output logits.
```

### Data-Update Attribution

The newest tool asks:

```text
Do data-group loss gradients point in the same direction as:
1. the actual checkpoint update Delta theta?
2. the route gradient for the candidate route?
```

This is not a replay of the exact historical optimizer batches. It is a source-checkpoint diagnostic.

#### Validation Data

Validation pair-type result:

```text
pair_type=query_key:
  actual update loss reduction: +0.053188
  route support:                -3.332346

pair_type=distractor:
  actual update loss reduction: +0.037117
  route support:                -7.157533
```

Simple meaning:

```text
The actual update weakly helps validation loss,
but validation loss gradients do not explain the route growth.
They point against the route.
```

This is acceptable because validation is not the training source, but it means:

```text
validation data pressure is not the reason this route grew.
```

#### Train Data

Train clean query-key grouping gives the important positive result.

All queried-key groups have positive route support and positive actual-update alignment:

```text
query key   records   actual update loss reduction   route support
K07         8         +0.087784                      +4.247879
K02         19        +0.059131                      +3.160501
K00         18        +0.044706                      +2.260123
K04         24        +0.037901                      +1.863644
K03         10        +0.028440                      +1.005445
K05         20        +0.028078                      +0.785265
K06         14        +0.044235                      +0.512089
K01         15        +0.037800                      +0.164039
```

Aggregate train clean query-key result:

```text
actual update loss reduction: +0.043717
route support:                +1.691921
local SGD route delta:        +0.000677
```

Simple meaning:

```text
At the source checkpoint, train clean loss gradients do support growing this route.
The support is uneven across keys, strongest for K07, K02, K00, and K04.
```

This is the first direct evidence connecting train data pressure to the candidate route.

But there is an important caveat:

```text
actual route delta:                       +0.160702
actual-update predicted route delta:      +0.393987
relative error:                            1.45167
sign match:                                true
```

So the direction is right, but the magnitude is not reliable in this train query-key run.

This means:

```text
The train data result supports the direction of the SGD story.
It does not close the exact quantitative proof.
```

Also, the train pair-type run should not be used as a pair-type comparison because `--loss-side clean` caused `query_key` and `distractor` groups to reuse the same clean source records. It produced duplicated values:

```text
pair_type=query_key route support:  +5.126363
pair_type=distractor route support: +5.126363
```

This is not meaningful as a query-key versus distractor comparison.

### Current Best Answer To The Why Question

The current best answer is:

```text
SGD did not build an isolated L2H1 circuit from scratch.
It moved many upstream parameters, especially L0/L1 attention and L0MLP,
in a direction that made the layer_1_post_mlp residual stream better aligned
with the route L2H1 can use.
```

Then:

```text
L2H1 routes from the query-side representation toward the associated value-bearing token.
Its OV/readout path writes strongly into the answer direction.
L1H2 also writes strongly.
L0MLP remains necessary because it supports the internal geometry,
even though its direct logit attribution is negative.
```

The data-update result adds:

```text
Train clean query-key gradients support this route.
Validation gradients do not.
```

So the current working explanation is:

```text
The route grows because train loss pressure pushes the model toward a residual geometry
that makes value-token routing useful. The final visible writer is L2H1, but much of the
construction happens upstream.
```

### What Is Supported

Supported by current artifacts:

```text
1. The model has a real query-key variable in the residual stream.
2. L2H1 QK query-side carries a meaningful part of that variable.
3. The real 5000 -> 5250 update grows the L2H1 query-key route on validation pairs.
4. The same update does not grow the distractor-control route.
5. L2H1 does not mainly route query -> support key.
6. L2H1 more strongly routes query-side representations toward support value tokens.
7. L2H1 is a major direct answer writer by DLA.
8. Ablating L2H1 strongly damages margin and accuracy.
9. L1H2 is also a major direct contributor.
10. L0MLP is causally essential but not a direct positive answer writer.
11. Train clean query-key gradients support growing the candidate route.
12. The route support is uneven across query keys.
```

### What Is Not Yet Proven

Not yet proven:

```text
1. Exact historical SGD causality from original minibatches.
2. Exact quantitative first-order prediction on train query-key groups.
3. That L2H1 is the winning route over all competing routes.
4. That the same route appears across seeds.
5. That the whole mechanism has been fully reverse engineered at neuron level.
6. That feature family7/family4 are natural circuit units.
7. That validation gradient pressure explains route growth.
```

### Current Research Position

In simple words:

```text
We are no longer merely observing that components matter.
We have a partially linked chain:

train data gradient pressure
  -> actual checkpoint update
  -> upstream residual geometry changes
  -> L2H1 value-facing route growth
  -> direct answer-logit writing
  -> causal ablation drop
```

But the chain is not closed enough to call it a mathematical proof.

The current strongest conclusion is:

```text
SGD appears to build a dense upstream infrastructure that makes a late value-routing
attention writer useful. L2H1 is one visible writer in that infrastructure, not the
whole circuit.
```

The next proof should compare multiple candidate routes under the same actual-update and data-update framework:

```text
for candidate routes P, Q, R:
  measure actual route delta
  measure grad(route) . Delta theta
  measure train data route support
  measure value-side attention score delta
  measure DLA and ablation drop

Then ask:
  which route is repeatedly selected by actual updates,
  and which data groups explain that selection?
```

That is the next step toward explaining why SGD forms this route rather than another.

## Optimizer-Trace And Stepwise Route-Competition Update

This section appends the newer findings after the earlier `5000 -> 5250` data-update notes.

The newer phase moved from sparse checkpoint attribution to a dense traced optimizer window:

```text
source checkpoint: step_005500.pt
traced updates:    5501 -> 5550
trace length:      50 real optimizer steps
checkpointing:     every step
device:            mps
```

Main artifacts:

```text
optimizer trace:
  artifacts/runs/symbolic_kv_reference_formation/analysis/optimizer_update_trace/l2h1_support_value_5500_5550_stepwise/

L2H1 retrieval-separation attribution:
  artifacts/runs/symbolic_kv_reference_formation/analysis/attention_retrieval_separation_update_attribution/l2h1_support_value_5500_5550_stepwise/

L1H2 retrieval-separation attribution:
  artifacts/runs/symbolic_kv_reference_formation/analysis/attention_retrieval_separation_update_attribution/l1h2_support_value_5500_5550_stepwise/

L0H0 retrieval-separation attribution:
  artifacts/runs/symbolic_kv_reference_formation/analysis/attention_retrieval_separation_update_attribution/l0h0_support_value_5500_5550_stepwise/

support-value route competition:
  artifacts/runs/symbolic_kv_reference_formation/analysis/route_competition/support_value_routes_5500_5550_stepwise/
```

### Optimizer Trace Integrity

The optimizer trace completed cleanly:

```text
steps recorded:              50
batch rows recorded:         50
dense checkpoints saved:     51
total query events observed: 41688
learning rate:               0.0004
mean loss:                   1.166233
mean token accuracy:         0.701193
mean parameter update L2:    0.056640
mean grad norm:              0.641036
grad clipping active:        0 / 50 steps
update dot -grad loss > 0:   50 / 50 steps
```

Simple meaning:

```text
The traced updates are normal optimizer steps.
They are not being dominated by clipping.
Each recorded update locally points in a loss-reducing direction.
```

Important caveat:

```text
This is an instrumented continuation from step 5500.
It is not an exact replay of the original historical minibatches,
because the old checkpoints did not save DataLoader sampler state or iterator offset.
```

So this window can prove:

```text
for this recorded continuation:
  actual batch -> actual optimizer update -> route movement
```

It cannot prove:

```text
the exact original training minibatches at steps 5501 -> 5550
```

### Stepwise QK Retrieval-Separation Result

The retrieval-separation scalar is:

```text
retrieval_separation =
  score(prediction, correct support value)
  - score(prediction, value distractors)
```

This is a QK attention-score geometry measurement, not an OV/write measurement.

Across the 50 real optimizer steps:

```text
head    actual score growth   predicted growth   sign match   median relative error
L2H1    +0.086687             +0.160357          50 / 50      0.037589
L1H2    +0.128266             +0.141779          50 / 50      0.015119
L0H0    +0.045533             +0.046169          49 / 50      0.008970
```

The score-level start and end values were:

```text
head    source score at 5500   final traced score at 5550
L2H1    7.639491               7.726178
L1H2    5.881868               6.010134
L0H0    3.647361               3.692894
```

Simple meaning:

```text
L2H1 is already the strongest absolute support-value retriever in this window.
L1H2 grows faster over these 50 traced steps.
L0H0 grows weakly.
```

This corrects a possible overclaim.

We should not say:

```text
SGD selected L2H1 QK over L1H2 QK during this window.
```

The raw result says:

```text
L2H1 is ahead in absolute QK retrieval separation,
but L1H2 sharpens more during this short continuation.
```

The first-order approximation itself is strong:

```text
grad_theta retrieval_separation(theta_t) . Delta theta_t
```

tracks the sign of actual route movement almost perfectly at one-step resolution.

This is a major improvement over sparse `250` or `500` step attribution windows.

### Query-Side Versus Key-Side Update

For the QK retrieval-separation decomposition:

```text
L2H1:
  q_side actual growth: +0.155511
  q_side predicted:     +0.157958
  q_side sign match:    50 / 50

  k_side actual growth: -0.076688
  k_side predicted:     +0.002399
  k_side sign match:    48 / 50

L1H2:
  q_side actual growth: +0.126756
  q_side predicted:     +0.137788
  q_side sign match:    50 / 50

  k_side actual growth: +0.001139
  k_side predicted:     +0.003992
  k_side sign match:    50 / 50

L0H0:
  q_side actual growth: +0.036452
  q_side predicted:     +0.036909
  q_side sign match:    50 / 50

  k_side actual growth: +0.009048
  k_side predicted:     +0.009260
  k_side sign match:    50 / 50
```

Simple meaning:

```text
For L2H1, the useful QK improvement in this window is mostly query-side.
The key-side term moves against the total improvement.
For L1H2 and L0H0, the query-side term also dominates, but their key-side terms do not conflict as strongly.
```

This supports the earlier intuition that a lot of circuit formation is upstream residual geometry:

```text
the model is shaping the representation at the prediction/query position
more than it is cleanly changing only the support-value key vectors.
```

### Support-Value Route Competition

The route-competition report measured a different object from QK retrieval separation.

It measured support-value route transfer in subspaces like:

```text
head_ov_input at support_value
embedding_value_identity at support_value
full_residual at support_value
```

This is not the same as:

```text
QK attention-score separation
```

So the two results must not be collapsed into one claim.

Cumulative route growth across the 50 traced one-step intervals:

#### Evaluation Domain

```text
route                         actual route growth   predicted growth   sign match
full_layer1_support_value      +0.913913            +2.512411          47 / 50
L2H1_ov_input_support_value    +0.776056            +1.362492          49 / 50
full_layer0_support_value      +0.423815            +3.313448          49 / 50
L0H0_ov_input_support_value    +0.047569            +0.828024          48 / 50
embedding_value_identity       +0.040278            +0.805444          47 / 50
L1H2_ov_input_support_value    +0.008181            +0.735241          50 / 50
```

#### Train-Probe Domain

```text
route                         actual route growth   predicted growth   sign match
full_layer1_support_value      +1.079115            +2.546266          44 / 50
full_layer0_support_value      +0.861900            +3.477037          48 / 50
L2H1_ov_input_support_value    +0.752303            +1.228640          50 / 50
L1H2_ov_input_support_value    +0.309984            +0.970642          47 / 50
embedding_value_identity       +0.228226            +0.916018          43 / 50
L0H0_ov_input_support_value    +0.141234            +0.689243          48 / 50
```

Simple meaning:

```text
For support-value route transfer, L2H1 grows much more than L1H2 and L0H0.
But full residual routes still grow more than individual-head routes.
```

This supports:

```text
L2H1 is a strong visible route, but the mechanism is still dense.
```

It does not support:

```text
L2H1 alone is the whole circuit.
```

### Data-Support Result From Route Competition

The route-competition data rows still use probe-set train/eval examples, not the actual traced optimizer batches.

So this is still a diagnostic:

```text
probe-set loss gradient -> route gradient
```

not the final actual-batch proof:

```text
recorded batch gradient -> actual update -> route growth
```

With that caveat, cumulative route support was:

#### Train-Probe Support

```text
route                         route support sum   local SGD route delta sum
full_layer0_support_value      +164.571831        +0.065829
L0H0_ov_input_support_value     +26.368105        +0.010547
embedding_value_identity        -23.962938        -0.009585
L1H2_ov_input_support_value     -32.577114        -0.013031
full_layer1_support_value       -41.870223        -0.016748
L2H1_ov_input_support_value    -274.764803        -0.109906
```

#### Eval-Probe Support

```text
route                         route support sum   local SGD route delta sum
L2H1_ov_input_support_value    +359.037039        +0.143615
full_layer1_support_value      +130.117306        +0.052047
L0H0_ov_input_support_value     -39.605441        -0.015842
L1H2_ov_input_support_value     -54.686251        -0.021875
embedding_value_identity       -267.812734        -0.107125
full_layer0_support_value      -764.501600        -0.305801
```

This is surprising and important:

```text
In this traced continuation, probe-train gradients do not explain L2H1 support-value route growth.
Eval-probe gradients support L2H1, but train-probe gradients oppose it.
```

This does not mean the actual recorded training batches opposed L2H1.

It means:

```text
the old train-probe diagnostic is not enough.
```

The next necessary measurement is exactly:

```text
actual recorded batch at step t
  -> batch loss gradient at theta_t
  -> dot with route gradient
  -> dot with actual Delta theta_t
```

### Actual-Batch Attribution Result

The missing actual-batch attribution run is now complete.

```text
actual-batch-route-attribution
```

It is designed to compute:

```text
actual_route_delta_t =
  route(theta_{t+1}; source_basis_t) - route(theta_t; source_basis_t)

actual_update_predicted_route_delta_t =
  grad route(theta_t) . (theta_{t+1} - theta_t)

actual_batch_route_support_t =
  < -grad loss_batch_t(theta_t), grad route(theta_t) >

actual_batch_update_alignment_t =
  < -grad loss_batch_t(theta_t), theta_{t+1} - theta_t >
```

The command checks that the recomputed batch loss matches the optimizer-trace loss before trusting the row.

Current status:

```text
tool implemented: yes
focused tests:    passed
result available: yes
intervals:        50
routes:           6
rows:             300
pairs:            128
max loss mismatch: 0
```

Completed report:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/actual_batch_route_attribution/support_value_routes_5500_5550_stepwise/actual_batch_route_attribution_report.md
```

Route summary:

```text
route                         actual growth   predicted by update   batch route support   local SGD delta   sign match
full_layer1_support_value      +0.913913       +2.51241              +76.4356              +0.0305742        0.940
L2H1_ov_input_support_value    +0.776056       +1.36249              +26.8469              +0.0107387        0.980
full_layer0_support_value      +0.423815       +3.31345              +140.974              +0.0563898        0.980
L0H0_ov_input_support_value    +0.0475691      +0.828024             +39.4372              +0.0157749        0.960
embedding_value_identity       +0.0402784      +0.805444             +38.6213              +0.0154485        0.940
L1H2_ov_input_support_value    +0.00818082     +0.735241             +36.3225              +0.014529         1.000
```

Ranked by actual route growth:

```text
1. full_layer1_support_value
2. L2H1_ov_input_support_value
3. full_layer0_support_value
4. L0H0_ov_input_support_value
5. embedding_value_identity
6. L1H2_ov_input_support_value
```

Ranked by actual-batch route support:

```text
1. full_layer0_support_value
2. full_layer1_support_value
3. L0H0_ov_input_support_value
4. embedding_value_identity
5. L1H2_ov_input_support_value
6. L2H1_ov_input_support_value
```

Simple meaning:

```text
The actual recorded batches do support the L2H1 support-value route.
L2H1 is also the second-largest realized isolated/broad route growth in this candidate set.
But actual-batch support does not rank L2H1 first.
The broad full-residual routes receive more batch support.
```

So this result closes one missing link:

```text
recorded batch gradient -> support-value route support
```

But it does not close the whole route-selection proof:

```text
batch support ranking != realized route-growth ranking
```

That mismatch is now a key constraint on the next explanation, not a missing run.

### Updated Position After The 5500 -> 5550 Trace

The evidence is now stronger in one specific way:

```text
At one-step resolution, actual parameter updates predict local route movement much better
than sparse checkpoint intervals did.
```

But it also became clearer that the mechanism is not a simple single-head story:

```text
QK retrieval separation:
  L2H1 is strongest in absolute score,
  but L1H2 grows faster in this short traced window.

Support-value route transfer:
  L2H1 grows much more than L1H2/L0H0,
  but full residual routes grow more than individual-head routes.

Probe-set data support:
  train-probe support did not explain L2H1 growth in this window.
  The completed actual-batch attribution shows the recorded batches do support L2H1,
  but broad residual routes receive more support.
```

Current best simple explanation:

```text
The model is not forming one clean isolated circuit.
It is shaping a dense residual infrastructure.
L2H1 is one strong late value-route/readout path inside that infrastructure.
L1H2 continues sharpening retrieval geometry.
Full residual pathways carry more growth than isolated head subspaces.
```

Current proof chain status:

```text
done:
  actual optimizer update -> local QK retrieval-separation movement
  actual optimizer update -> support-value route transfer movement
  actual recorded batch gradient -> support-value route support
  route competition between L2H1, L1H2, L0H0, embeddings, and full residual routes

not done:
  actual route growth -> final answer-margin growth in the same traced window
  explain why batch support ranking differs from realized route-growth ranking
  actual-batch query-key route attribution
  cross-seed repeat
  longer traced windows beyond 50 steps
```

The honest claim is now:

```text
We have closed the update-to-route part at one-step resolution.
We have also measured actual recorded batch support for support-value routes.
We have not yet explained why the route with the largest batch support is not the route with the largest realized growth.
```

## 2026-04-20 Update: Output-Route Closure And Moving-Margin Branch Diagnosis

After the last update, we moved from route-transfer scores into output-space closure.

The reason was simple:

```text
The previous route-to-margin closure was not enough.
Patch-transfer route scores did not fully explain final answer-margin movement.
So we tested whether component writes explain the final output scalar directly.
```

This is a different measurement from the earlier route reports.

Earlier route reports asked:

```text
If we patch this route/subspace, does the answer transfer?
```

The new output-route closure asks:

```text
For each component write, how much does its movement explain the final scalar movement?
```

The calculation is:

```text
g_s(theta, x) =
  d scalar_s / d final_pre_layernorm_residual

DLA_{component,s}(theta, x) =
  component_write(theta, x) dot g_s(theta, x)

Delta scalar_s
  ~= sum over components beta_component * Delta DLA_{component,s}
```

This uses the final output readout gradient through final layernorm.

Completed artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/output_route_closure/query_key_support_value_5500_5550_stepwise/output_route_closure_report.md
```

### Output-Route Closure Result

The output-route closure used:

```text
window:       5500 -> 5550 stepwise
pairs:        128 causal pairs
observations: 6400 pair-interval rows
components:   embedding + all attention heads + all MLP blocks
margin side:  clean
pair types:   query_key, support_value
```

All-component closure by scalar:

```text
scalar                          R^2       mean abs residual
negative_answer_loss             0.8422    0.00922
correct_value_logit              0.7828    0.19660
target_best_wrong_logit          0.6413    0.15557
source_best_wrong_logit          0.6379    0.15586
fixed_target_competitor_margin   0.6127    0.19603
fixed_source_competitor_margin   0.6117    0.19629
moving_answer_margin             0.0992    0.27991
```

Simple interpretation:

```text
The output-space component movement explains the training-like scalar well.
It also explains correct-answer logit movement well.
It moderately explains fixed wrong-token margins.
It fails badly on raw moving answer margin.
```

This means the problem is not simply:

```text
we cannot explain output movement
```

The more precise result is:

```text
we can explain differentiable/fixed output quantities much better than the moving max-margin quantity.
```

### Important Development Check

The output-route tool initially failed a scalar recomputation guard:

```text
Scalar recomputation mismatch for moving_answer_margin
```

This exposed a real implementation bug:

```text
endpoint residual/component vectors were computed at each checkpoint,
but endpoint readout gradients were accidentally recomputed using the last-loaded checkpoint.
```

That was fixed by reloading and validating the correct checkpoint for every endpoint-gradient group.

The final reported run completed after this fix.

### What The Output-Route Result Says About Components

For `correct_value_logit`, the largest fitted output-route contributions were dense, not isolated:

```text
component   mean contribution
L1H3        +0.02605
L0MLP       -0.01491
L1MLP       -0.01355
L2H0        -0.00915
L2H2        -0.00864
embedding   +0.00847
L0H1        +0.00835
L1H2        +0.00750
```

For `negative_answer_loss`, the biggest all-pair contributions were also distributed:

```text
component   qualitative role in fitted closure
embedding   large contribution
L0MLP       large contribution
L0H1        large contribution
L2MLP       nontrivial contribution
L2H1        nontrivial contribution
L1H3        nontrivial contribution
L1H2        nontrivial contribution
L2H3        nontrivial contribution
```

This supports the dense-infrastructure view:

```text
The local output movement is not carried by one clean head or one clean neuron.
Many components move in partially opposing directions.
The final scalar is produced by their combined readout effect.
```

### Why Moving Answer Margin Looked Bad

The bad scalar was:

```text
moving_answer_margin =
  logit(correct) - logit(best_wrong_at_that_checkpoint)
```

This scalar is unstable because the identity of `best_wrong` can change between checkpoints.

So the metric can silently switch from:

```text
logit(correct) - logit(V040)
```

to:

```text
logit(correct) - logit(V056)
```

That is not just measuring model improvement.
It is also measuring which wrong token is currently second-best.

We therefore built:

```text
answer-margin-branch-decomposition
```

Completed artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/answer_margin_branch_decomposition/query_key_support_value_5500_5550_stepwise/answer_margin_branch_decomposition_report.md
```

The exact algebra is:

```text
Delta moving_margin
  = Delta fixed_source_margin
    + [target_logit(source_wrong) - target_logit(target_wrong)]

Delta moving_margin
  = Delta fixed_target_margin
    + [source_logit(source_wrong) - source_logit(target_wrong)]
```

The reconstruction error was exactly zero within the recorded rows:

```text
source_reconstruction_abs_error_max = 0
target_reconstruction_abs_error_max = 0
```

So the branch decomposition is not a heuristic.
It is an exact accounting identity over the saved scalar rows.

### Branch-Decomposition Result

Competitor switches were rare:

```text
competitor switches: 298 / 6400
switch fraction:     4.65625%
```

Across all examples:

```text
moving margin abs mean:            0.29275
target branch correction abs mean: 0.00939
source branch correction abs mean: 0.00921
target branch energy / moving:     0.02945
source branch energy / moving:     0.02361
```

So across all rows, branch switching is small on average.

But on the switch rows only:

```text
moving margin abs mean:            0.36415
target branch correction abs mean: 0.20161
source branch correction abs mean: 0.19771
target branch energy / moving:     0.43061
source branch energy / moving:     0.34529
```

This means:

```text
Wrong-token switching is rare,
but when it happens it explains a large fraction of the moving-margin instability.
```

### Branch-Aware Closure Result

Before branch correction:

```text
direct moving-margin closure R^2 = 0.0992
```

After using fixed margin plus exact branch correction:

```text
all rows:
  direct moving R^2:             0.0992
  fixed-source + branch R^2:     0.6064
  fixed-target + branch R^2:     0.6080

competitor-switch rows:
  direct moving R^2:             0.1611
  fixed-source + branch R^2:     0.7132
  fixed-target + branch R^2:     0.7380

same-competitor rows:
  direct moving R^2:             0.6063
  fixed-source + branch R^2:     0.6063
  fixed-target + branch R^2:     0.6063
```

Simple interpretation:

```text
The terrible moving-margin result was mostly a branch/metric problem.
When the best wrong token is held fixed, or when the branch correction is added exactly,
moving-margin closure rises from about 10% to about 61%.
```

For the switch-only rows, the improvement is stronger:

```text
16% explained -> 71-74% explained
```

So the old failure does not mean:

```text
the route/output explanation is useless
```

It means:

```text
moving answer margin is a bad local proof scalar unless competitor identity is handled explicitly.
```

### Updated Proof Status

Current supported chain:

```text
actual optimizer update
  -> local route movement

actual recorded batch gradient
  -> support for support-value routes

component output movement
  -> training-like scalar movement

fixed wrong-token branch accounting
  -> much better moving-margin closure
```

The strongest local scalar is now:

```text
negative_answer_loss
```

because:

```text
1. it is closest to the training objective,
2. output-route closure is strong for it,
3. it avoids the hard max branch-switch problem.
```

The main behavioral scalar can still be reported as answer margin, but the proof should use:

```text
negative_answer_loss and fixed-competitor margins
```

with moving margin treated as a downstream summary that requires branch correction.

### Current Simple Research Story

The model is not forming one clean isolated circuit.

The current evidence says:

```text
SGD updates a dense residual infrastructure.
Within that infrastructure, some routes become useful for retrieval and value writing.
Those route/component changes explain much of the final output movement.
The final answer score is produced by many components pushing and cancelling together.
```

The important shift is:

```text
We are no longer only saying "component X matters."
We can now say:

component movement, measured in output-readout coordinates,
explains a large fraction of the actual output scalar movement.
```

### What Is Still Missing

The proof is still not complete.

Remaining gaps:

```text
1. Causal validation of output-DLA components
   The output-route closure is mathematical/readout evidence.
   We still need to ablate or patch the top components and verify behavior changes.

2. The missing 39% of branch-aware moving-margin variance
   Best branch-aware moving-margin R^2 is about 0.61, not 1.0.
   The rest may come from nonlinear final layernorm effects, component interactions,
   residual coupling, or unmeasured subcomponent structure.

3. Actual-batch output-DLA attribution
   We have actual-batch route attribution.
   We have output-DLA closure.
   We have not yet directly connected actual recorded batch updates to output-DLA scalar movement.

4. Query-key side actual-batch proof
   The support-value side is better traced.
   Query-key routing still needs the same level of actual-batch/output closure.

5. Cross-seed replication
   Current results are one seed.
   We cannot claim stable role-level SGD selection until repeated across seeds.
```

### Next Plan

Do not rerun the same route-to-margin or output-route closure tools.

The next stage should be:

```text
output-component-causal-validation
```

Purpose:

```text
Take the components identified by output-route closure,
remove or patch them,
and check whether the scalar changes match the DLA prediction.
```

Suggested component set:

```text
correct_value_logit top components:
  L1H3, L0MLP, L1MLP, L2H0, L2H2, embedding, L0H1, L1H2

negative_answer_loss top components:
  embedding, L0MLP, L0H1, L2MLP, L2H1, L1H3, L1H2, L2H3
```

The tool should measure:

```text
baseline scalar
component-ablated scalar
causal drop
DLA predicted contribution
causal drop vs DLA prediction
same-competitor vs competitor-switch split
query_key vs support_value split
```

Primary scalars:

```text
negative_answer_loss
correct_value_logit
fixed_source_competitor_margin
fixed_target_competitor_margin
branch-aware moving margin
```

Expected outcome:

```text
If high-DLA components also produce high causal drops,
then output-route closure becomes causal evidence.

If DLA and causal drops disagree,
then the remaining explanation must account for nonlinear component interactions.
```

After that:

```text
1. actual-batch output-DLA attribution
2. query-key side actual-batch/output closure
3. cross-seed role-level replication
4. paper update with this proof chain
```

## 2026-04-20 Update: Causal Validation, Mediation, And Residual Rescue

This section records the new results after the output-route closure note above.

Important correction:

```text
These results do not prove why SGD formed the circuit.
They improve the trained-model causal accounting.
```

The recent run sequence was:

```text
1. output-component-causal-validation
2. output-mediated-causal-decomposition
3. output-mediated-causal-decomposition with all later components
4. residual-state-rescue
```

The artifacts are:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/output_component_causal_validation/query_key_support_value_5500_5550_stepwise/

artifacts/runs/symbolic_kv_reference_formation/analysis/output_mediated_causal_decomposition/query_key_support_value_5500_5550_stepwise/

artifacts/runs/symbolic_kv_reference_formation/analysis/output_mediated_causal_decomposition/l0mlp_all_later_components_5500_5550_stepwise/

artifacts/runs/symbolic_kv_reference_formation/analysis/output_mediated_causal_decomposition/l1h3_all_later_components_5500_5550_stepwise/

artifacts/runs/symbolic_kv_reference_formation/analysis/output_mediated_causal_decomposition/l1mlp_all_later_components_5500_5550_stepwise/

artifacts/runs/symbolic_kv_reference_formation/analysis/residual_state_rescue/query_key_support_value_5500_5550_stepwise/
```

### What Output-Component Causal Validation Tested

The output-route closure result was a readout accounting result:

```text
DLA(component, scalar)
  = component residual write dot scalar readout gradient
```

That says how much a component points toward an output scalar. It does not by itself prove that the component is causally load-bearing.

The causal-validation run compared:

```text
DLA contribution:
  component_write dot scalar_gradient

causal effect:
  scalar(normal model) - scalar(model with component removed)

gap:
  causal effect - DLA contribution
```

The command tested `512000` endpoint/component/scalar rows:

```text
128 pairs
6400 scalar rows
51 endpoint checkpoints
10 components
4 scalars
source and target endpoints
```

### Output-Component Causal Validation Result

Late components have much better agreement between direct DLA and causal effect.

For target endpoints:

| scalar | component | mean causal effect | mean DLA | sign match | correlation | R^2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_source_competitor_margin` | `L2MLP` | `3.364054` | `1.850016` | `0.880` | `0.985` | `0.897` |
| `fixed_target_competitor_margin` | `L2MLP` | `3.358048` | `1.844964` | `0.881` | `0.985` | `0.898` |
| `fixed_source_competitor_margin` | `L2H1` | `7.345897` | `5.444075` | `0.921` | `0.881` | `0.626` |
| `fixed_target_competitor_margin` | `L2H1` | `7.347727` | `5.444509` | `0.921` | `0.881` | `0.626` |
| `fixed_source_competitor_margin` | `L1H2` | `6.553232` | `3.990044` | `0.922` | `0.725` | `0.293` |
| `fixed_target_competitor_margin` | `L1H2` | `6.549956` | `3.987632` | `0.922` | `0.727` | `0.295` |

For the correct-value logit:

| component | mean causal effect | mean DLA | sign match | correlation | R^2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `L2H1` | `12.593419` | `7.554457` | `0.973` | `0.857` | `-0.221` |
| `L1H2` | `10.673396` | `5.437859` | `0.974` | `0.641` | `-0.703` |
| `L2MLP` | `6.122651` | `2.515269` | `0.780` | `0.939` | `0.282` |
| `L2H3` | `0.648262` | `0.274151` | `0.804` | `0.747` | `0.443` |
| `L2H0` | `0.004342` | `-0.259079` | `0.856` | `0.810` | `0.591` |

Interpretation:

```text
L2H1, L2MLP, and L1H2 are closer to direct output/readout pieces.
Their output-DLA is not perfect, but it tracks causal effect much better
than early components do.
```

Early components are causally huge, but their direct DLA is a bad explanation of their effect.

For target endpoint `correct_value_logit`:

| component | mean causal effect | mean DLA | sign match | correlation | R^2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `L0MLP` | `27.738898` | `-7.652493` | `0.162` | `-0.055` | `-27.293` |
| `L1H3` | `21.515125` | `-2.711512` | `0.318` | `0.072` | `-9.370` |
| `L1MLP` | `15.712419` | `-0.387060` | `0.495` | `0.490` | `-2.993` |

Simple interpretation:

```text
L0MLP, L1H3, and L1MLP are important.
But their importance is not "they directly write the answer logit."
Removing them changes the rest of the computation.
```

This is the clearest current split:

```text
late components:
  closer to direct writers / readout routes

early components:
  broad upstream infrastructure
```

### Negative Answer Loss Caveat

Earlier output-route closure made `negative_answer_loss` look like the strongest scalar because it was closest to the training objective.

The causal-validation and mediation runs show a limitation:

```text
negative_answer_loss is useful for update/objective accounting,
but it becomes unstable under off-manifold component ablations.
```

For example, target endpoint `negative_answer_loss`:

```text
L0MLP causal effect = 16.927812
L0MLP DLA           = 0.054080
```

The loss/log-prob scalar is therefore not the best scalar for component-mediation claims.

For causal mediation and residual rescue, the cleaner scalars are:

```text
correct_value_logit
fixed_source_competitor_margin
fixed_target_competitor_margin
```

### What Narrow Output Mediation Tested

The narrow mediation run asked:

```text
If an early source component is removed,
does that damage the DLA of the later direct writers?
```

The decomposition was:

```text
total_effect(A)
  = scalar(theta) - scalar(theta with A ablated)

direct_effect(A)
  = DLA_A(theta)

mediated_effect(A -> B)
  = DLA_B(theta) - DLA_B(theta with A ablated)

residual
  = total_effect(A) - direct_effect(A) - sum_B mediated_effect(A -> B)
```

The source components were:

```text
L0MLP
L1H3
L1MLP
```

The downstream components were:

```text
L1H2
L2H1
L2MLP
L2H0
L2H2
L2H3
```

### Narrow Output Mediation Result

For target endpoint `correct_value_logit`:

| source | total causal effect | direct DLA | downstream mediated sum | direct + mediated | abs residual | explained fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `L0MLP` | `27.738898` | `-7.652493` | `16.124124` | `8.471631` | `19.535640` | `0.305` |
| `L1H3` | `21.515125` | `-2.711512` | `12.577657` | `9.866145` | `11.695200` | `0.459` |
| `L1MLP` | `15.712419` | `-0.387060` | `11.584630` | `11.197570` | `5.344625` | `0.713` |

For target endpoint fixed margins:

| source | fixed-source explained fraction | fixed-target explained fraction |
| --- | ---: | ---: |
| `L0MLP` | `0.542` | `0.541` |
| `L1H3` | `0.667` | `0.667` |
| `L1MLP` | `1.078` | `1.079` |

Interpretation:

```text
L1MLP is mostly explainable through downstream direct writers.
L1H3 is partly explainable.
L0MLP is only partly explainable.
```

The largest mediated paths were:

| path | scalar | mediated effect |
| --- | --- | ---: |
| `L0MLP -> L2H1` | `correct_value_logit` | `7.4515` |
| `L0MLP -> L1H2` | `correct_value_logit` | `5.1969` |
| `L0MLP -> L2MLP` | `correct_value_logit` | `3.1648` |
| `L1H3 -> L2H1` | `correct_value_logit` | `6.5816` |
| `L1H3 -> L2MLP` | `correct_value_logit` | `3.3300` |
| `L1MLP -> L2MLP` | `correct_value_logit` | `6.1051` |
| `L1MLP -> L2H1` | `correct_value_logit` | `2.8337` |

This supports a real, but incomplete, upstream-to-downstream story:

```text
early components help maintain later output-writing routes,
but that does not explain the full early-component causal effect.
```

### All-Later Output Mediation Result

We then expanded the downstream set to all later heads and MLPs.

This was meant to test:

```text
Maybe the residual is large only because we left out important later components.
```

That hypothesis failed.

Adding more downstream components did not improve closure. It made closure worse for several source/scalar pairs.

For target endpoint `correct_value_logit`:

| source | narrow explained fraction | all-later explained fraction |
| --- | ---: | ---: |
| `L0MLP` | `0.305` | `0.147` |
| `L1H3` | `0.459` | `0.452` |
| `L1MLP` | `0.713` | `0.629` |

For target endpoint fixed margins:

| source | narrow fixed-source fraction | all-later fixed-source fraction | narrow fixed-target fraction | all-later fixed-target fraction |
| --- | ---: | ---: | ---: | ---: |
| `L0MLP` | `0.542` | `0.370` | `0.541` | `0.369` |
| `L1H3` | `0.667` | `0.526` | `0.667` | `0.525` |
| `L1MLP` | `1.078` | `0.830` | `1.079` | `0.831` |

This is important.

It says the missing effect is not simply:

```text
we forgot to include a later head
```

When all later components are included, positive and negative mediated terms cancel.

Example for `L0MLP -> correct_value_logit`:

```text
positive mediated terms:
  L0MLP -> L2H1   +7.451
  L0MLP -> L1H2   +5.197
  L0MLP -> L2MLP  +3.165

negative mediated terms:
  L0MLP -> L1H3   -2.744
  L0MLP -> L1H0   -0.837
  L0MLP -> L1MLP  -0.458
  L0MLP -> L1H1   -0.364
```

Simple interpretation:

```text
The downstream network is sign-conflicted.
The learned computation is not a clean additive chain.
It is a dense residual system where components push and cancel together.
```

### Residual-State Rescue Result

The residual-state rescue run asked:

```text
If a source component is removed,
can we restore behavior by patching back the clean residual stream at a later stage?
```

The calculation was:

```text
damage = scalar(clean) - scalar(source ablated)

rescue = scalar(source ablated + clean residual patch at stage S)
         - scalar(source ablated)

rescue_fraction = rescue / damage
```

The result is clean, but mostly confirms the intervention boundary.

For target endpoint `correct_value_logit`:

| source | patch stage | damage | rescue | rescue fraction | read |
| --- | --- | ---: | ---: | ---: | --- |
| `L0MLP` | `layer_0_post_mlp` | `27.7389` | `27.7389` | `1.000` | rescued immediately after `L0MLP` |
| `L0MLP` | `layer_1_post_attn` | `27.7389` | `27.7389` | `1.000` | rescued |
| `L1H3` | `layer_0_post_mlp` | `21.5151` | `0.0000` | `0.000` | too early |
| `L1H3` | `layer_1_post_attn` | `21.5151` | `21.5151` | `1.000` | rescued immediately after `L1H3` writes |
| `L1MLP` | `layer_1_post_attn` | `15.7124` | `0.0000` | `0.000` | too early |
| `L1MLP` | `layer_1_post_mlp` | `15.7124` | `15.7124` | `1.000` | rescued immediately after `L1MLP` writes |

The same pattern holds for fixed-source and fixed-target margins.

Interpretation:

```text
L0MLP damage enters the residual stream at layer_0_post_mlp.
L1H3 damage enters at layer_1_post_attn.
L1MLP damage enters at layer_1_post_mlp.
```

But this is not a deep explanation by itself.

It is partly tautological:

```text
If we ablate a component and then patch the full clean residual stream after that component,
we restore that component's effect.
```

What it gives us is a clean boundary:

```text
the missing effect is in the residual state after the source component,
not necessarily in a named downstream head/MLP decomposition.
```

### Current Honest Interpretation

The recent results should change the project story.

Old over-strong story:

```text
SGD builds a dense upstream infrastructure that supports L2H1.
```

More careful story:

```text
The trained model uses a dense residual-stream mechanism.

Late components such as L2H1, L2MLP, and L1H2 behave more like direct output/readout routes.

Early components such as L0MLP, L1H3, and L1MLP are causally essential,
but their effects are not explained by direct DLA or additive mediation through named later components.

Full residual patching restores the effect once we patch after the ablated component,
which localizes the damage but does not identify the abstract variable.
```

This is a trained-model causal-accounting result.

It is not yet an SGD-formation proof.

### Why This Is Hard Even In A Small Model

This task is symbolically simple:

```text
find latest write for queried key
return value
```

But the model does not implement it as symbolic code.

It implements it with:

```text
residual stream vectors
QK dot products
OV writes
MLP nonlinearities
layernorm
unembedding directions
```

The residual stream is a shared workspace:

```text
attention reads it
attention writes it
MLPs read it
MLPs write it
later layers read a mixture of everything
```

So even a small transformer can learn:

```text
overlapping features
shared residual directions
sign-conflicted component contributions
partial routes
shortcuts mixed with real retrieval
```

This is why the analysis keeps becoming dense:

```text
the model did not build a clean software module called lookup_table.
It built a distributed vector process that behaves like lookup.
```

Neuron-level analysis is hard because neurons are not clean units:

```text
one neuron can support multiple features
one feature can be spread over many neurons
the useful variable can be a direction or subspace, not a single neuron
```

Component-level analysis is hard because heads and MLPs are not independent modules:

```text
removing one component changes the inputs seen by later components
patching a full residual state can rescue behavior while hiding which variable inside that state mattered
DLA is local linear readout accounting, not a complete causal model
```

SGD formation is harder still because SGD does not choose circuits explicitly.

It only does:

```text
reduce current batch loss
```

Every update changes many parameters. A route can become useful because:

```text
the current data gradient supports it
the current residual geometry makes it easy to amplify
downstream components can already read it
other routes interfere less
optimizer state pushes parameters in that direction
```

So the correct question is not:

```text
Which component is the circuit?
```

It is:

```text
Which scalar internal variable C actually grows under the recorded update,
and does that growth explain behavior better than competing variables?
```

### Where We Are Stuck

The project is currently strong on:

```text
trained-model causal accounting
direct writer identification
load-bearing upstream component identification
controlled intervention design
short-window update attribution
```

The project is still weak on:

```text
identifying one abstract internal variable C
proving actual historical SGD caused C to grow
showing C beats competing variables under the same update
showing C growth closes behavior
replicating the role across seeds
```

The main problem is not a lack of more measurements.

The main problem is:

```text
we have not fixed one final proof variable.
```

### Stop Condition For Broad Tooling

The recent runs show that broad component accounting has hit diminishing returns.

We should stop creating broad reports of the form:

```text
measure many components
rank many effects
find another residual
```

Those are useful for exploration, but they are not moving the project toward the why question anymore.

The next formation proof should choose one variable:

```text
C(theta, x) = L2H1 support-value retrieval separation
```

or:

```text
C(theta, x) = fixed-competitor correct-value readout contribution from a validated route set
```

Then measure only:

```text
C(theta_t)
C(theta_{t+1})
Delta C_actual
grad_theta C(theta_t) dot Delta theta_actual
Delta behavior
competing route Delta C
```

This is the finite proof unit.

### Updated Simple Summary

The simplest honest summary now is:

```text
We trained a tiny transformer on a simple retrieval task.

The trained model does not contain a clean lookup circuit.
It contains a dense residual-stream implementation.

Late parts like L2H1, L2MLP, and L1H2 look like direct readout/retrieval pieces.

Earlier parts like L0MLP, L1H3, and L1MLP are necessary,
but they do not directly write the answer.
They change the residual state that the rest of the model uses.

Trying to explain those early parts by summing later component effects does not close.
The system is sign-conflicted and nonlinear.

Full residual patching rescues the model after the damaged component,
but that only tells us where the damage enters the residual stream,
not what abstract variable SGD learned.

Therefore we have a good trained-model causal map,
but not yet a proof of circuit formation by SGD.
```

The next research question should be:

```text
Pick one internal variable C.
Did actual SGD updates create and amplify C,
and does C growth explain behavior better than alternatives?
```

## Weight-Space SVD Pivot: Looking For Formation In The Raw Parameters

After the route, residual, and causal-patching work, the project hit a clear limit:

```text
we could show which components and routes matter,
but not yet how the route became a stable object under SGD.
```

The next direction is therefore weight-space dynamics.

Instead of only asking:

```text
which activation route is useful after training?
```

we now also ask:

```text
which raw weight directions rotate, stabilize, grow, or shrink during training?
```

This matters because the model is not made of clean symbolic variables. It is made of tensors.

If SGD forms a retrieval circuit, that formation should leave traces in the actual matrices:

```text
W_Q
W_K
W_V
W_O
MLP W_in
MLP W_out
```

The goal of this pivot is not to replace route-level causal analysis.

The goal is to connect:

```text
loss gradient
parameter update
weight-space geometry change
route growth
answer-margin improvement
```

### New Tooling

We added a raw weight SVD extraction tool:

```text
circuit.cli weight-svd-trace
```

It extracts, for every checkpoint:

```text
attention head W_QK = W_Q^T W_K
attention head W_OV = W_V^T W_O^T
MLP W_in
MLP W_out
```

For each matrix it records:

```text
full singular value spectrum
top singular vectors
effective rank = (sum singular values)^2 / sum(singular values^2)
top-3 spectral mass = sum(top 3 singular values) / sum(all singular values)
```

We also added a second-stage pattern reader:

```text
circuit.cli weight-svd-patterns
```

This does not rerun model checkpoints.

It reads the SVD rows and reports:

```text
which matrices grew
which matrices became more concentrated
which singular vectors rotated toward their final direction
which intervals show coordinated movement across matrices
```

Primary artifacts:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/weight_svd_trace/phase1_000250_5500_top16/
artifacts/runs/symbolic_kv_reference_formation/analysis/weight_svd_patterns/phase1_000250_5500_top16/
```

The earliest saved checkpoint available in this run is `step_000250.pt`, not step zero.

So this is:

```text
250 -> 5500
```

not a true from-initialization trace.

### Main SVD Result

The strongest weight-space formation signal is `L2H1 W_QK`.

From `step_000250 -> step_005500`:

```text
L2H1 W_QK top singular value: 0.6667 -> 3.5004
delta:                         +2.8336
relative growth:                +425%
effective rank:                 27.62 -> 17.21
top-3 spectral mass:             0.164 -> 0.323
```

This is not just general weight growth.

It means:

```text
the matrix becomes more concentrated into a smaller number of directions.
```

That is exactly the kind of signature we expected if a route is forming in weight space.

The top selective-concentration scores were:

```text
L2H1 W_QK   0.4496
L1H2 W_QK   0.3163
L2H0 W_QK   0.2434
L2H2 W_QK   0.1276
L2H3 W_QK   0.1148
```

So the strongest formation signature is in QK routing geometry.

It is not primarily in OV.

### Direction Birth In L2H1 W_QK

The final `L2H1 W_QK` direction was not cleanly present at the earliest saved checkpoint.

For the rank-1 query-side singular vector:

```text
step 250:  final-direction cosine 0.188
step 750:  final-direction cosine 0.588
step 2250: final-direction cosine 0.845
step 3500: final-direction cosine 0.952
step 4250: final-direction cosine 0.987
step 5500: final-direction cosine 1.000
```

For the rank-1 key-side singular vector:

```text
step 250:  final-direction cosine 0.079
step 750:  final-direction cosine 0.405
step 2250: final-direction cosine 0.848
step 3750: final-direction cosine 0.953
step 4250: final-direction cosine 0.983
step 5500: final-direction cosine 1.000
```

Simple interpretation:

```text
L2H1 W_QK starts weak and mixed.
It rotates toward its final route during early training.
Then it stabilizes and grows.
```

This changes the story.

Earlier route tools made `5500 -> 7500` look like the main emergence window.

The SVD trace suggests:

```text
by step 5500, much of the L2H1 QK geometry is already built.
```

The likely birth window for this route is earlier:

```text
roughly 1750 -> 3000
```

### Coordinated Movement Windows

The strongest broad movement happens very early:

```text
250 -> 500
```

That interval has:

```text
28 / 30 matrices with positive top-singular-value growth
29 / 30 matrices with positive top-3 mass growth
```

This is probably broad early training movement, not yet a clean retrieval circuit.

The more relevant route-formation window is:

```text
1750 -> 3000
```

Important intervals:

```text
1750 -> 2000
2000 -> 2250
2250 -> 2500
2500 -> 2750
2750 -> 3000
```

`L2H1 W_QK` growth inside that window:

```text
2000 -> 2250: +0.314
2250 -> 2500: +0.533
2500 -> 2750: +0.388
2750 -> 3000: +0.176
```

This looks like:

```text
the route direction is being carved out and amplified before the later route-level measurements.
```

### Comparison With Other Components

`L1H2 W_QK` also grows strongly:

```text
0.8332 -> 3.4977
delta +2.6645
relative growth +319.8%
effective rank 26.33 -> 16.74
top-3 spectral mass 0.187 -> 0.306
```

But `L1H2` is more aligned with its final direction at the earliest saved checkpoint than `L2H1`.

That suggests `L1H2` may be a more available or earlier route, while `L2H1` looks more like a direction that is built/rotated into place.

`L2H0 W_QK` also shows strong concentration:

```text
0.6767 -> 2.3487
effective rank 27.37 -> 17.91
top-3 spectral mass 0.171 -> 0.317
```

This may be a competing or supporting route, but its task role is not yet proven by this SVD run alone.

`L0MLP W_out` behaves differently.

Its top singular value does not simply grow:

```text
2.2767 -> 2.1484
```

But its top output direction rotates strongly toward the final direction:

```text
step 250:  final-direction cosine 0.224
step 2000: final-direction cosine 0.592
step 2250: final-direction cosine 0.824
step 2500: final-direction cosine 0.893
step 3500: final-direction cosine 0.968
```

This supports the earlier causal story:

```text
L0MLP may be shaping the residual stream rather than directly growing as an answer writer.
```

### Important Negative Result: OV Is Not Clean

The cleanest signal is QK, not OV.

For `L2H1 W_OV`, earlier SVD checks showed only small top singular value growth and more distributed structure:

```text
top singular value changes only weakly
effective rank increases
top-3 mass decreases
```

That means the value/write side does not look like one clean low-rank object.

This matches the superposition problem:

```text
routing can become relatively clean in QK,
while value writing and output readout remain distributed across OV, MLPs, and residual directions.
```

So the project should not expect the entire algorithm to appear as one neat singular vector.

### What This Adds To The Research Story

Before this SVD pivot, the strongest story was:

```text
L2H1 is a useful route.
L0MLP and other early components are load-bearing.
The residual stream is dense and shared.
```

But that still sounded like trained-model observation.

The SVD trace adds a more formation-specific statement:

```text
L2H1 W_QK becomes a more concentrated, stable routing matrix during training.
Its final top direction is weak early, rotates into place, and then amplifies.
```

This is closer to the SGD-formation question because it is about the actual parameters, not only activations.

The current best simple story is:

```text
Early training changes broad residual and MLP geometry.
Around 1750 -> 3000, L2H1 QK rotates toward a stable retrieval-routing direction.
That QK direction becomes more low-rank/concentrated.
By 3500 -> 4250, the direction is mostly locked in.
After that, later training mostly amplifies/refines an already-built route.
The value/output side stays more distributed and superposed.
```

### What This Still Does Not Prove

The SVD results prove a weight-space formation pattern.

They do not yet prove the semantic content of that pattern.

Specifically, we still need to show:

```text
the growing L2H1 W_QK directions align with query-key / support-key task geometry
the growing directions predict route score growth
the growing directions predict answer-margin improvement
competing QK directions do not explain the same behavior as well
the same role-level pattern appears across seeds
```

So the next proof target becomes:

```text
SVD direction -> task geometry -> route score -> answer margin
```

In simple words:

```text
We found weight directions that form.
Now we must prove those directions are the lookup directions.
```

### Updated Research Direction

The candidate proof variable should shift from a broad activation route to a weight-space route geometry variable.

Candidate:

```text
C(theta) = alignment / strength of the top L2H1 W_QK singular subspace
           with the dataset key-retrieval geometry
```

The next proof unit should measure:

```text
C(theta_t)
C(theta_{t+1})
actual Delta C
gradient/update prediction for Delta C
alignment of C with query-key and support-key geometry
correlation of C with route score
correlation of C with answer margin
comparison against L1H2 W_QK, L2H0 W_QK, and other QK routes
```

This does not abandon the route-level work.

It makes the route-level work more grounded:

```text
routes tell us what computation is used.
SVD tells us how the weights are becoming able to implement that route.
```

---

## Update: From Weight Pattern To Optimizer-Level Formation Explanation

The later experiments filled in the missing link after the SVD result.

The previous section showed that `L2H1 W_QK` becomes more concentrated and stable during training. That was a weight-level formation pattern, but by itself it was not enough. A singular direction can grow without being the task mechanism.

The next question was:

```text
Is the growing QK direction actually the retrieval direction?
And if yes, why did the optimizer build that direction?
```

We now have a much stronger answer for this particular model run, seed 7.

### Current Single-Run Claim

For this seed, the model does not learn lookup as a clean neuron table.

It learns a dense role-level retrieval route:

```text
prediction position
  -> L2H1 QK route
  -> support value position beats value distractors
  -> value/write path helps the answer margin
```

The stable object is not one neuron and not one feature family.

The stable object is a geometric route:

```text
L2H1 W_QK builds a low-rank matcher that scores support values above distractors.
```

The reason this matters is that it connects four levels:

```text
behavior level:
  answer margin improves

route level:
  L2H1 increasingly separates support values from distractors

weight level:
  L2H1 W_QK develops a low-rank rank-8 matching direction

optimizer level:
  the real AdamW update explains the growth of that direction
```

This is the first point where the research stops being only observational.

We can now say, for this run:

```text
the optimizer update physically built the measured retrieval geometry.
```

### The Main Route Variable

The key scalar became:

```text
C_rank(theta)
  = mean score_rank(prediction, support_value)
    - mean score_rank(prediction, value_distractors)
```

For the strongest result:

```text
head: L2H1
matrix: W_QK
rank: 8
context stage: layer_1_post_mlp
query role: prediction
support role: support_value
distractor role: value_distractors
```

In simple words:

```text
Does the rank-8 part of L2H1 QK make the prediction position match the real support value more than distractor values?
```

That is a precise route-geometry scalar.

### Formation Window Evidence

The bilinear QK match separation run showed that `L2H1` is not merely growing a random singular direction.

For `750 -> 3500`, `layer_1_post_mlp`, `rank_8`:

```text
support-value separation delta:  +4.19295
correlation with singular value: 0.9934
correlation with answer margin:  0.6664
```

This means:

```text
as the low-rank W_QK structure grows,
the support-value matching behavior grows with it.
```

That is the semantic bridge missing from the raw SVD story.

The earlier SVD evidence said:

```text
L2H1 W_QK becomes concentrated.
```

The bilinear QK evidence says:

```text
the concentrated direction is useful for support-value retrieval.
```

### Update Attribution Evidence

The rank-update attribution checked whether actual checkpoint changes move this route scalar.

For the formation window:

```text
path:
artifacts/runs/symbolic_kv_reference_formation/analysis/bilinear_qk_rank_update_attribution/l2h1_rank4_rank8_support_value_minus_distractors_000750_003500_formation/
```

For rank 8:

```text
actual route delta:      +2.03547
predicted route delta:   +2.21138
sign match:              11 / 11 intervals
```

This says:

```text
the actual parameter movement points in the direction that grows the retrieval matcher.
```

That still did not answer why, because "actual update" is not the same thing as raw SGD gradient.

So we decomposed the update.

### Important Negative Result: Raw Batch Gradient Is Not Enough

The actual-batch attribution on `750 -> 1000` showed:

```text
actual route growth:             +0.0275398
actual-update predicted growth:  +0.0263239
actual-batch route support:      +0.0654552
SGD-equivalent contribution:     +0.00002618
```

That SGD-equivalent contribution is only about:

```text
0.095% of actual route growth
```

This was a critical result.

It means the naive explanation is wrong:

```text
wrong story:
  the current batch gradient directly pushes the route hard enough to build it
```

The real story must involve optimizer dynamics.

### Exact From-Initialization Optimizer Trace

To remove the historical replay concern, we ran a new exact instrumented trace from initialization:

```text
optimizer trace:
artifacts/runs/symbolic_kv_reference_formation/analysis/optimizer_update_trace/from_init_seed7_0000_6000_stepwise

Adam-state attribution:
artifacts/runs/symbolic_kv_reference_formation/analysis/bilinear_qk_rank_adam_state_attribution/from_init_l2h1_rank8_support_value_0000_6000_stepwise/
```

This trace records the batch stream and optimizer updates from step 0.

The report status is:

```text
instrumented_from_initialization_exact_for_this_trace
```

So this is not a guessed replay from old checkpoints. It is exact for this traced run.

### Main Optimizer Result

Across `0 -> 6000`:

```text
actual route growth:             +4.11462
actual-update prediction:        +5.21768
reconstructed AdamW prediction:  +5.21734
reconstruction sign match:       6000 / 6000
```

The AdamW decomposition:

```text
raw SGD:                    +0.03136
clipped SGD:                +0.02404
Adam current gradient:      +2.37417
Adam historical momentum:   +3.04547
Adam preconditioned total:  +5.41964
weight decay:               -0.20230
```

As a fraction of actual route growth:

```text
raw SGD:                    0.76%
clipped SGD:                0.58%
Adam current gradient:      57.7%
Adam historical momentum:   74.0%
weight decay:              -4.9%
```

This is the strongest "why" result so far.

For this model:

```text
the support-value retrieval route is not built by a large immediate raw gradient.
It is built by Adam's preconditioned update, especially accumulated optimizer momentum.
```

### Phase Structure Of Formation

The full `0 -> 6000` trace is not one uniform story.

It has phases.

#### 0 -> 750: early weak setup

```text
actual growth:        +0.07007
predicted growth:     +0.06651
raw SGD:              -0.00142
current gradient:     -0.00101
historical momentum:  +0.06746
```

Even this early, the useful movement is mostly momentum.

#### 750 -> 2500: clean momentum-driven route formation

```text
actual growth:        +1.66529
predicted growth:     +1.59974
raw SGD:              -0.00302
current gradient:     +0.00495
historical momentum:  +1.60536
weight decay:         -0.01058
```

This is the cleanest formation window.

The simple interpretation:

```text
many small gradients are accumulated by Adam,
and the accumulated optimizer state repeatedly pushes the QK matcher into place.
```

The current batch gradient alone is basically not the explanation here.

#### 2500 -> 3500: current gradient and momentum both push

```text
actual growth:        +1.67303
predicted growth:     +2.25247
raw SGD:              +0.01987
current gradient:     +1.16475
historical momentum:  +1.13675
weight decay:         -0.04911
```

Now the current preconditioned gradient becomes large.

This looks like the route has become useful enough that fresh gradients also reinforce it.

#### 3500 -> 6000: optimizer still pushes, realized growth saturates

```text
actual growth:        +0.70622
predicted growth:     +1.29897
raw SGD:              +0.01593
current gradient:     +1.20549
historical momentum:  +0.23590
weight decay:         -0.14267
```

Here the first-order update still points toward route growth, but actual route growth is smaller.

This likely means the route is becoming constrained by nonlinear geometry, basis drift, interaction with other routes, or saturation.

So the best current phase story is:

```text
0 -> 750:
  weak early setup

750 -> 2500:
  Adam momentum builds the QK support-value matcher

2500 -> 3500:
  fresh gradients and momentum jointly amplify it

3500 -> 6000:
  the optimizer still pushes the route, but realized growth is partly limited by nonlinear/dense interactions
```

### Dense-Circuit Interpretation

This is still not a clean neuron story.

The evidence points to dense circuit formation:

```text
QK routing becomes relatively low-rank and measurable.
OV/value writing remains distributed.
MLPs and early residual components are load-bearing but not clean answer writers.
Feature families are useful diagnostics but not stable proof objects.
```

In simple terms:

```text
the model is not writing "K03 maps to V14" into one neuron.
It is shaping a shared residual workspace so that a later attention head can route from prediction positions to support-value positions.
```

This also explains why the research has been hard.

We kept looking for one object:

```text
one neuron
one feature family
one head
one clean OV vector
```

But the model uses overlapping objects:

```text
many neurons
shared residual directions
low-rank QK routing
distributed value/write effects
optimizer state accumulated across many steps
```

The cleanest object is not a neuron.

The cleanest object is:

```text
a role-level route geometry plus the optimizer update that grows it.
```

### What We Can Honestly Claim Now

For this one seed and traced run, we can say:

```text
1. A support-value retrieval route forms.
2. The route is visible in L2H1 W_QK as a low-rank matcher.
3. The matcher separates support values from distractors.
4. Actual parameter updates grow the matcher.
5. AdamW update decomposition reconstructs that growth.
6. Raw SGD is far too small to explain it.
7. Adam momentum dominates early formation.
8. Later, current preconditioned gradients and momentum jointly amplify the route.
```

This gives a proof-style causal accounting for one model:

```text
loss gradients
  -> Adam optimizer state
  -> parameter update
  -> W_QK low-rank route growth
  -> support-value retrieval separation
  -> answer-margin improvement
```

The word "proof" should still be used carefully.

This is not a theorem about all transformers.

It is a detailed mechanistic accounting for one trained model and one exact from-initialization trace.

### What We Still Do Not Know

The remaining open questions are:

```text
Does the same role appear across random seeds?
Does it always appear in L2H1, or does another head implement the same role?
Does Adam momentum always dominate early formation?
Would SGD build the same route more slowly, differently, or not at all?
How much of the later answer margin is closed by this route versus distributed OV/MLP effects?
Can this method scale beyond a small symbolic model?
```

The main missing validation is cross-seed replication.

Right now the strongest claim is:

```text
this is how this seed builds the route.
```

The stronger paper claim would be:

```text
this task/architecture reliably induces a support-value retrieval role,
even if the exact head identity changes.
```

### Why This Matters Beyond This Toy Model

The practical value is not that this tiny model performs symbolic lookup.

The practical value is the method:

```text
watch a circuit form in weight space
connect the weight change to route behavior
decompose the optimizer update that caused it
check whether the same role repeats across seeds
```

Most interpretability work can say:

```text
this component matters in the trained model
```

This work is moving toward:

```text
this optimizer update wrote this route into these weight directions during training
```

That is a different kind of understanding.

If this method generalizes, it could help with:

```text
detecting shortcut circuits during training
tracking when factual-recall routes form
studying refusal or safety circuits under fine-tuning
checking whether a model learned the intended mechanism or a brittle alternative
preserving useful circuits across training changes
```

### The Three Big Walls

There are three major limitations between this result and broad neural network control.

#### 1. Scale

This model is small enough that we can scan heads, SVD matrices, and run one-step optimizer attribution.

Large models have many more layers, heads, parameters, and simultaneous behaviors.

The method would need much stronger candidate selection before it can scale.

#### 2. Superposition

Even here, neurons are polysemantic and the OV/write side is distributed.

In larger models, more of the mechanism may be spread across overlapping directions.

That means we should not expect clean single-neuron or single-vector explanations.

#### 3. Task Overlap

This symbolic KV task has one clean algorithm.

Language models learn many algorithms at once.

A weight direction that helps one behavior may also participate in unrelated behaviors.

So route-level and optimizer-level explanations may need to be conditional on data slice, context, and task role.

### Cross-Seed Validation Plan

The next stage is not to create more tools for this same seed.

The next stage is to test whether this role-level story repeats.

#### Step 1: Keep the dataset fixed and vary training seed

Use the same benchmark/data first.

Change the model/training seed:

```text
seed 11
seed 13
seed 17
seed 23
```

The goal is to ask:

```text
Does the same support-value retrieval role form under different initialization/training randomness?
```

#### Step 2: Do not require the same head identity

The replication target is not:

```text
L2H1 always wins.
```

The replication target is:

```text
some head develops a support-value-over-distractor QK route
using contextual residual states.
```

So for each seed we should scan all heads:

```text
L0H0 ... L0H3
L1H0 ... L1H3
L2H0 ... L2H3
```

#### Step 3: Use a cheap-to-expensive funnel

Do not run exact optimizer decomposition on every head first.

Use this order:

```text
cheap:
  bilinear QK match separation across all heads
  weight SVD pattern scan

medium:
  rank-update attribution on candidate heads
  contextual separability/alignment checks

expensive:
  exact optimizer trace and Adam-state attribution on the winning role/head
```

#### Step 4: Compare role-level metrics

For each seed, record:

```text
seed
best support-value retrieval head
formation window
rank-8 support-value separation growth
top W_QK singular value growth
effective rank drop
answer-margin correlation
actual route delta
actual-update predicted delta
raw SGD contribution
Adam current-gradient contribution
Adam momentum contribution
weight decay contribution
```

#### Step 5: Interpret outcomes

Possible outcomes:

```text
same role, same head:
  architecture strongly biases the role into L2H1

same role, different heads:
  the role is stable but head identity is seed-specific

no consistent role:
  the current result is mostly seed-specific and should be presented as a deep case study

same role but different optimizer split:
  the task induces the same computation, but optimizer dynamics vary by seed
```

### Current Bottom Line

The best current conclusion is:

```text
In one small transformer trained on symbolic KV lookup,
we can trace a dense retrieval circuit from behavior,
to route geometry,
to low-rank QK weight formation,
to exact AdamW update components.
```

And the best current explanation for why the route forms is:

```text
the loss supplies many small, noisy gradient signals;
Adam accumulates and preconditions those signals;
the accumulated optimizer state repeatedly pushes L2H1 W_QK
toward a support-value retrieval matcher;
once the route becomes useful, fresh gradients also reinforce it.
```

This is not yet a universal theory.

But it is now a concrete single-run mechanism.

The next question is whether the same role-level mechanism appears again when we train more seeds.

---

## Cross-Seed Validation Results: Role Pattern Replicates, Head Identity Does Not

We then ran the cross-seed validation plan.

The validation goal was not:

```text
Does L2H1 win every time?
```

That would be the wrong replication target.

The real target was:

```text
Does a support-value retrieval role form across seeds?
Does it have the same geometric and optimizer-level signature?
```

We varied training/model seeds while keeping the same task/data setup:

```text
seed 11
seed 13
seed 17
seed 23
seed 29
```

The output root is:

```text
artifacts/runs/symbolic_kv_cross_seed_adam/
```

Important artifacts:

```text
head scan reports:
artifacts/runs/symbolic_kv_cross_seed_adam/seed_*/analysis/bilinear_qk_match_separation/

winner/runner-up/bottom selections:
artifacts/runs/symbolic_kv_cross_seed_adam/seed_*/analysis/cross_seed_head_selection.json

Adam attribution reports:
artifacts/runs/symbolic_kv_cross_seed_adam/seed_*/analysis/bilinear_qk_rank_adam_state_attribution/
```

One bookkeeping note:

```text
cross_seed_winners.csv was overwritten by a later seed-29-only command.
The per-seed cross_seed_head_selection.json files are the reliable source of truth.
```

### Cross-Seed Procedure

For each seed:

```text
1. train from initialization with recorded optimizer trace
2. save cheap scan checkpoints every 250 steps
3. scan all 12 heads:
   L0H0 ... L2H3
4. score each head by rank-8 QK support-value-over-distractor growth
   during 750 -> 3500
5. select:
   winner
   runner-up
   bottom-control head
6. run exact Adam-state attribution for 750 -> 2500
   on winner, runner-up, and bottom control
```

The scalar was the same one used in the single-seed analysis:

```text
C_rank(theta)
  = mean score_rank(prediction, support_value)
    - mean score_rank(prediction, value_distractors)
```

with:

```text
rank: 8
context stage: layer_1_post_mlp
query role: prediction
support role: support_value
distractor role: value_distractors
analysis window for Adam attribution: 750 -> 2500
```

### Result 1: Same Role, Different Head Address

The winning head changed across seeds:

| seed | winning head | scan score | sep vs singular value | sep vs answer margin | support-win delta |
|---:|---|---:|---:|---:|---:|
| 11 | `L2H0` | 2.815 | 0.882 | 0.668 | 0.157 |
| 13 | `L2H2` | 2.727 | 0.956 | 0.670 | 0.523 |
| 17 | `L2H3` | 1.463 | 0.484 | 0.561 | 0.183 |
| 23 | `L2H1` | 6.361 | 0.868 | 0.918 | 0.843 |
| 29 | `L1H2` | 2.428 | 0.502 | 0.891 | 0.248 |

This means:

```text
4 / 5 seeds put the strongest support-value retrieval role in layer 2.
1 / 5 seeds put it in layer 1.
The original seed-7 head identity, L2H1, is not stable across seeds.
```

That is not a failure.

It is the expected result if the computation is stable but the address is random.

The better claim is:

```text
the circuit is stable as a role pattern,
not stable as a named head.
```

### Result 2: Winner Beats Runner-Up And Bottom Controls

We then compared winner, runner-up, and bottom-control heads using exact Adam-state attribution on the same window:

```text
750 -> 2500
```

Actual route growth:

| seed | winner | winner actual | runner-up | runner-up actual | bottom | bottom actual |
|---:|---|---:|---|---:|---|---:|
| 11 | `L2H0` | 1.448 | `L2H2` | 0.509 | `L0H0` | -0.190 |
| 13 | `L2H2` | 1.451 | `L1H2` | 0.719 | `L1H1` | -0.230 |
| 17 | `L2H3` | 3.178 | `L2H0` | 0.680 | `L0H2` | -0.254 |
| 23 | `L2H1` | 1.500 | `L2H3` | 1.437 | `L1H2` | -0.114 |
| 29 | `L1H2` | 1.439 | `L2H0` | 0.712 | `L1H0` | -2.577 |

This gives a useful control result:

```text
winner actual growth is positive in all 5 seeds
runner-up actual growth is also positive in all 5 seeds
bottom-control actual growth is negative in all 5 seeds
```

So the metric is not simply saying:

```text
all heads grow
```

It distinguishes support-value retrieval heads from weak/wrong heads.

The runner-up result is also important.

Since runner-up heads often grow too, the model is not always building one isolated route.

It often builds a route family:

```text
one dominant support-value retrieval head
plus one or more partially participating heads
```

Seed 23 is the clearest example:

```text
winner L2H1 actual growth:    1.500
runner-up L2H3 actual growth: 1.437
```

That seed looks less like a single winner and more like a shared layer-2 retrieval family.

### Result 3: Raw SGD Is Tiny Across Winners

For winner heads, raw SGD as a fraction of actual route growth:

| seed | winner | raw SGD / actual | clipped SGD / actual |
|---:|---|---:|---:|
| 11 | `L2H0` | -0.13% | -0.65% |
| 13 | `L2H2` | 0.83% | 0.55% |
| 17 | `L2H3` | 1.26% | 1.31% |
| 23 | `L2H1` | 1.15% | 1.04% |
| 29 | `L1H2` | 0.60% | 0.75% |

Mean raw SGD contribution:

```text
0.74% of actual route growth
```

Mean clipped SGD contribution:

```text
0.60% of actual route growth
```

This strongly replicates the seed-7 finding:

```text
the immediate raw per-batch gradient is far too small to explain route formation.
```

So a naive SGD story is not enough.

The optimizer state matters.

### Result 4: Adam State Carries The Formation Update

For winner heads:

| seed | winner | current-gradient / actual | momentum / actual | sign match |
|---:|---|---:|---:|---:|
| 11 | `L2H0` | 39.0% | 79.7% | 99.7% |
| 13 | `L2H2` | 63.8% | 62.5% | 99.7% |
| 17 | `L2H3` | 101.3% | 47.8% | 98.3% |
| 23 | `L2H1` | 91.0% | 55.4% | 99.4% |
| 29 | `L1H2` | 52.8% | 66.9% | 99.3% |

Across winners:

```text
Adam current-gradient contribution: 39% -> 101%
Adam historical momentum:           48% -> 80%
actual/predicted sign match:        98.3% -> 99.7%
```

This supports the refined optimizer story:

```text
Adam's preconditioned update state carries the useful route-growth direction.
```

Momentum is always large, but not always the largest component.

The balance varies by seed:

```text
seed 11: momentum dominates
seed 13: current gradient and momentum are balanced
seed 17: current gradient dominates
seed 23: current gradient dominates
seed 29: momentum is slightly larger
```

So the cross-seed result is not:

```text
momentum always dominates exactly.
```

The correct result is:

```text
raw SGD is consistently tiny;
Adam preconditioned state consistently carries route growth;
the split between current gradient and historical momentum is seed-dependent.
```

### Result 5: Bottom Controls Move The Opposite Way

Bottom-control heads:

| seed | bottom head | scan score | actual route delta | predicted delta |
|---:|---|---:|---:|---:|
| 11 | `L0H0` | -0.225 | -0.190 | -0.317 |
| 13 | `L1H1` | -0.260 | -0.230 | -0.372 |
| 17 | `L0H2` | -0.521 | -0.254 | -0.448 |
| 23 | `L1H2` | -0.279 | -0.114 | -0.221 |
| 29 | `L1H0` | -2.653 | -2.577 | -3.033 |

This is a strong negative control:

```text
bottom scan scores are negative
bottom actual route deltas are negative
bottom predicted deltas are negative
```

That means the method is not just measuring global training progress.

It is sensitive to the direction of the role.

In simple terms:

```text
the heads selected as support-value retrieval routes grow in that direction;
the heads selected as bottom controls move away from that direction.
```

### Cross-Seed Interpretation

The original seed-7 story was:

```text
L2H1 builds a rank-8 QK support-value matcher.
Raw SGD is too small.
Adam momentum/current-gradient drive the route.
```

The cross-seed story is stronger and more general:

```text
a support-value QK matcher appears across seeds;
the exact head changes;
the route's geometric growth is positive for winners;
bottom controls move in the opposite direction;
raw SGD is consistently tiny;
Adam preconditioned update state consistently explains route formation.
```

This supports the central theory:

```text
SGD/Adam does not reliably select one named component.
It reliably builds a role-level retrieval pattern somewhere in the network.
```

The role is:

```text
prediction-position QK matching to support-value positions
over value-distractor positions
```

The repeated formation signature is:

```text
1. support-value-over-distractor QK separation rises
2. the rise correlates with W_QK singular geometry
3. actual optimizer updates grow the route scalar
4. raw SGD is too small
5. Adam preconditioned current/momentum components carry the growth
6. weak-control heads move in the wrong direction
```

### Updated Claim After Cross-Seed Validation

The strongest defensible claim is now:

```text
In this symbolic KV setting, training repeatedly forms a dense support-value retrieval role.
The role is not tied to one head identity.
Across seeds, different heads instantiate the role, usually in late layers.
The role has a repeated weight/route signature:
rank-8 QK support-value matching grows during formation.
The growth is explained by actual AdamW updates,
while raw per-batch SGD is far too small to account for it.
```

This is stronger than the earlier single-run result.

It moves the work from:

```text
this is how seed 7 built L2H1
```

to:

```text
this task tends to induce the same retrieval role,
but random initialization chooses the address.
```

### What This Still Does Not Prove

The cross-seed result still has boundaries.

It does not prove:

```text
the same role appears for every possible seed
SGD without Adam would build the same route
the value/write side is equally clean
the method scales directly to large language models
```

It also suggests the circuit is not always a single clean route.

Runner-up heads often show positive growth:

```text
winner mean actual growth:    1.803
runner-up mean actual growth: 0.811
bottom mean actual growth:   -0.673
```

So the better mechanistic object is probably:

```text
a role family with one dominant route,
not a single isolated head.
```

That matches the broader dense-circuit picture:

```text
the computation is stable,
but its physical implementation is distributed and seed-dependent.
```

### Revised Bottom Line

The best current conclusion is now:

```text
We can trace the formation of a dense retrieval role across seeds.

The role repeatedly appears as a low-rank QK support-value matcher.

The exact head changes with random seed.

Winner heads grow in the support-value retrieval direction.

Bottom-control heads move in the opposite direction.

The immediate raw gradient is too small to explain this formation.

Adam's preconditioned optimizer state carries the useful update direction.
```

In simple words:

```text
the circuit is stable as a pattern,
but unstable as an address.
```

## 2026-04-27 Update: OV Write-Side Is A Downstream Chain, Not One Head Writing The Answer

The QK side now has the cleanest story:

```text
L2H1 W_QK forms a low-rank support-value matcher.
AdamW's preconditioned state explains most of that route growth.
The role repeats across seeds even when the winning head changes.
```

The write side is harder.

The question is different from QK:

```text
QK asks: where does the head look?
OV asks: what does the attended value become after it is written into the residual stream?
```

For QK, a support-vs-distractor score is a direct routing scalar.

For OV, the written vector enters the shared residual stream, passes through later normalization, attention, and MLP blocks, and only later becomes answer evidence.

So the right question became:

```text
which write-side signal grows,
where is it written,
and which downstream components turn it into output evidence?
```

### OV Scalar Audit

The new `ov-write-progress-report` scanned write-side scalars across heads during the formation window.

Artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/ov_write_progress/all_heads_0750_3500_formation/
```

Main result:

```text
L0H0 is the cleanest early OV/write-side candidate.
```

Important numbers:

```text
L0H0 forced-support real OV map final value:       +4.82437
L0H0 forced-support real OV map delta:             +10.12
L0H0 forced-support shuffled-value final value:    -15.5484
real-vs-shuffled final gap:                        about +20.37
```

The strongest scalar was not "raw head output points directly at the answer" in isolation.

The useful scalar was closer to:

```text
attention route × OV write usefulness
```

The `L0H0 qk_ov_product` tracked output-space progress very strongly:

```text
correlation with delta negative_answer_loss: +0.9703
```

This says the early write signal is not random.

It becomes useful when the head reads the right source and writes value-relevant information into the residual stream.

### AdamW Decomposition For The L0H0 Write Route

The next tool applied the same optimizer-state decomposition used for QK, but to the `L0H0 qk_ov_product` write scalar.

Artifacts:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/attention_downstream_adam_state_attribution/l0h0_ov_write_qk_product_0750_1000_stepwise/
artifacts/runs/symbolic_kv_reference_formation/analysis/attention_downstream_adam_state_attribution/l0h0_ov_write_qk_product_1000_1250_stepwise/
artifacts/runs/symbolic_kv_reference_formation/analysis/attention_downstream_adam_state_attribution/l0h0_ov_write_qk_product_1250_1500_stepwise/
artifacts/runs/symbolic_kv_reference_formation/analysis/attention_downstream_adam_state_attribution/l0h0_ov_write_qk_product_1500_1750_stepwise/
artifacts/runs/symbolic_kv_reference_formation/analysis/attention_downstream_adam_state_attribution/l0h0_ov_write_qk_product_1750_2000_stepwise/
artifacts/runs/symbolic_kv_reference_formation/analysis/attention_downstream_adam_state_attribution/l0h0_ov_write_qk_product_2000_2250_stepwise/
artifacts/runs/symbolic_kv_reference_formation/analysis/attention_downstream_adam_state_attribution/l0h0_ov_write_qk_product_2250_2500_stepwise/
```

The phase structure was:

```text
750  -> 1000: -0.405
1000 -> 1250: -1.294
1250 -> 1500: -2.958
1500 -> 1750: +18.433
1750 -> 2000: +8.135
2000 -> 2250: +1.013
2250 -> 2500: +0.328
```

So the write route is suppressed early, then has a sharp birth burst between `1500 -> 2000`, then consolidates.

Across `750 -> 2500`:

```text
actual qk_ov_product growth:             +23.251
AdamW reconstruction:                    +23.852
raw SGD contribution:                    +0.059
current preconditioned-gradient part:    +2.624
historical momentum part:                +21.821
weight decay part:                       -0.592
```

As a fraction of actual growth:

```text
raw SGD:             about 0.25%
current gradient:    about 11.3%
momentum:            about 93.9%
weight decay:        about -2.5%
```

This is a stronger AdamW result than the QK case in one respect:

```text
the L0H0 write-side formation burst is overwhelmingly carried by AdamW momentum.
```

The parameter-slice split also matters.

For the `1500 -> 1750` burst, the signs were:

```text
Q slice: negative
K slice: negative
V slice: strongly positive
O slice: strongly positive
```

So this is not just another QK routing effect.

The write-side birth is mostly a `W_V/W_O` effect.

### L0H0 Is Causal, But Not Directly Sufficient

We then asked whether the `L0H0` write is actually used downstream.

The first path test ablated `L0H0` and measured the target endpoint.

The ablation damage was large:

```text
correct_value_logit drop:              about +5.079
negative_answer_loss drop:             about +2.326
fixed-source competitor margin drop:   about +2.941
fixed-target competitor margin drop:   about +1.482
```

A full residual-state patch rescued almost all of this, which proves the information is present in the residual stream.

But full residual patching is too broad.

It says:

```text
the L0H0 effect is somewhere in the residual stream.
```

It does not say:

```text
which downstream component reads it.
```

### Single-Component Output Rescue

The first narrower rescue patched one downstream component write at a time after ablating `L0H0`.

Artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/component_output_rescue/l0h0_downstream_component_writes_1500_2000/
```

Target endpoint, best single-component rescue:

| scalar | best single component | rescue fraction |
|---|---|---:|
| `negative_answer_loss` | `L0MLP` | 22.1% |
| `correct_value_logit` | `L0MLP` | 12.6% |
| `fixed_source_competitor_margin` | `L0MLP` | 11.6% |
| `fixed_target_competitor_margin` | `L0MLP` | 8.7% |

Secondary signal:

```text
L2MLP helps correct_value_logit by about 7.9%.
L2MLP helps fixed-source competitor margin by about 9.2%.
L2H1 is tiny in this test, usually around 0.4% -> 1.0%.
```

This rejected a simple story:

```text
L0H0 writes value information directly to L2H1.
```

The data says:

```text
L0MLP is the first strong downstream reader/transformer of the L0H0 write.
```

### Grouped Downstream Rescue

The next rescue patched ordered downstream component groups.

Artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/component_output_rescue/l0h0_downstream_component_groups_1500_2000/
```

This was the important result.

Grouped rescues were much stronger than single-component rescues.

Target endpoint:

| scalar | best single rescue | best grouped rescue |
|---|---:|---:|
| `negative_answer_loss` | 22.1% via `L0MLP` | 35.0% via `L0MLP+L2H1+L2MLP` |
| `correct_value_logit` | 12.6% via `L0MLP` | 40.1% via `L0MLP+L1H3+L1MLP+L2MLP` |
| `fixed_source_competitor_margin` | 11.6% via `L0MLP` | 42.5% via `L0MLP+L1H3+L1MLP+L2MLP` |
| `fixed_target_competitor_margin` | 8.7% via `L0MLP` | 15.9% via `L0MLP+L2H1+L2MLP` |

The best grouped rescues also had much cleaner rescue-vs-damage correlations.

Examples:

```text
correct_value_logit target:
  best group = L0MLP+L1H3+L1MLP+L2MLP
  rescue fraction = 40.1%
  correlation = 0.934
  R^2 = 0.343

fixed_source_competitor_margin target:
  best group = L0MLP+L1H3+L1MLP+L2MLP
  rescue fraction = 42.5%
  correlation = 0.835
  R^2 = 0.428

negative_answer_loss target:
  best group = L0MLP+L2H1+L2MLP
  rescue fraction = 35.0%
  correlation = 0.695
```

So the current write-side circuit is not:

```text
L0H0 -> answer
```

and not:

```text
L0H0 -> L2H1 -> answer
```

The better current picture is:

```text
L0H0 writes an early value-bearing residual ingredient.
L0MLP reads/transforms that ingredient.
Middle-layer components refine it.
L2MLP converts much of it into output-space value evidence.
L2H1 helps some loss-relevant paths, but is not the main reader of this early write signal.
```

In simple terms:

```text
QK gives us a clean pointer.
OV gives us a shared residual ingredient.
The answer is produced by a downstream chain that reuses that ingredient.
```

This is exactly where superposition hits hardest.

The value is not written as a clean standalone answer vector.

It is written into a shared residual space where later MLPs and heads can read, rotate, suppress, or amplify parts of it.

### Updated QK Plus OV Closure Context

The existing `route-family-closure-report` result on the `5500 -> 5550` stepwise window already showed that adding OV/output families improves answer-margin closure:

Artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/route_family_closure/qk_vs_ov_vs_joint_5500_5550_stepwise/
```

Family closure results:

| family | R squared | mean absolute residual |
|---|---:|---:|
| `qk` | 0.370 | 0.2417 |
| `ov_input` | 0.188 | 0.2619 |
| `ov_output` | 0.421 | 0.2301 |
| `qk_plus_ov` | 0.456 | 0.2254 |

This matters because QK alone is not the whole behavior story.

Adding OV/output-side routes improves closure, but does not fully close the answer-margin gap.

The grouped rescue result explains why:

```text
the write side is not a single route label;
it is a distributed downstream chain.
```

### Updated Research Position

The current end-to-end picture is now:

```text
1. AdamW builds a low-rank QK support-value matcher.
2. The same training run also builds an early L0H0 value-write signal.
3. The L0H0 write-side birth is mostly V/O and momentum-driven.
4. That write does not directly become the answer.
5. L0MLP and later MLP-heavy chains transform it into output-space value evidence.
6. QK+OV route-family closure is better than QK alone, but still incomplete.
```

The important correction to the paper story is:

```text
the QK side is a clean route-formation result;
the OV side is a dense residual-chain result.
```

That is not a failure.

It is the actual mechanism becoming visible.

### What This Still Does Not Prove

This does not yet prove full behavioral sufficiency.

The best grouped rescues recover about `35% -> 42%` of important target-endpoint damage, not 100%.

Remaining explanations include:

```text
unpatched residual interactions
attention-pattern shifts
normalization effects
additional components not included in the tested groups
nonlinear composition between patched writes
branch sensitivity in the answer-margin scalar
```

It also does not yet prove that a plain SGD optimizer would or would not build the same write route.

The supported claim is narrower:

```text
in this AdamW run, the observed write-side birth is carried mostly by AdamW momentum,
and the causal output path is distributed across downstream residual transformations.
```

### Focused Joint-Path Closure Run

After the grouped rescue result, we ran a focused `route-family-closure-report` over the existing `5500 -> 5550` route-to-margin rows.

Artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/route_family_closure/qk_pointer_l0h0_write_chain_5500_5550_stepwise/
```

Families:

```text
qk_pointer:
  L2H1_qk_query
  L1H2_qk_query
  L0H0_qk_query
  embedding_key_identity
  full_layer1_query_key
  full_layer0_query_key

early_write_chain_proxy:
  L0H0_ov_input_support_value
  embedding_value_identity
  full_layer0_support_value
  full_layer1_support_value

late_output_proxy:
  L1H2_ov_output_prediction
  L2H1_ov_output_prediction
  full_layer1_post_attn_prediction
  full_layer2_post_attn_prediction

qk_plus_early_write:
  qk_pointer + early_write_chain_proxy

qk_plus_write_plus_output:
  qk_pointer + early_write_chain_proxy + late_output_proxy
```

Result:

| family | routes | R squared |
|---|---:|---:|
| `early_write_chain_proxy` | 4 | 0.181 |
| `qk_pointer` | 6 | 0.370 |
| `late_output_proxy` | 4 | 0.421 |
| `qk_plus_early_write` | 10 | 0.383 |
| `qk_plus_write_plus_output` | 14 | 0.451 |

This says:

```text
QK pointer routes explain a real part of answer-margin movement.
The early write proxy alone is weaker on this late 5500 -> 5550 window.
Late output routes explain more than QK alone.
Combining QK, early-write, and late-output routes is best.
But the combined measured family still does not fully close answer margin.
```

The important interpretation is:

```text
the output-side path is not just L0H0 write geometry;
by the late stepwise window, the cleaner explanatory variables are late output-space routes.
```

This matches the rescue result:

```text
L0H0 creates an early value ingredient,
but downstream components, especially MLP-heavy output transformations,
turn it into answer evidence.
```

### Next Experiment After This

The remaining gap is now more specific.

We need to bridge the early write-side birth window to the late output-side closure window.

The next useful run should measure the same focused families over the formation window, not only `5500 -> 5550`.

Target question:

```text
does the L0H0 write-chain proxy explain more during 1500 -> 2500,
when the OV/write route is actually born?
```

If yes, the story becomes:

```text
early window: L0H0/L0MLP write-chain forms
late window: L2H1 QK and late output routes dominate behavior closure
```

If no, the write-side object is probably even more residual-state based than the current route labels capture.

### Formation-Window Joint Closure Result

We then ran the same focused closure over the actual OV/write-side formation window:

```text
1500 -> 1750 -> 2000 -> 2250 -> 2500
```

Artifacts:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/route_to_margin_closure/qk_ov_output_routes_1500_2500_formation/
artifacts/runs/symbolic_kv_reference_formation/analysis/route_family_closure/qk_pointer_l0h0_write_chain_1500_2500_formation/
```

Overall route-to-margin closure:

```text
observations:                 512
mean actual answer-margin Δ:  +0.878392
mean predicted Δ:             +0.630379
mean residual:                +0.248013
mean absolute residual:       1.44093
R^2:                          0.248964
design rank:                  13 / 14
```

Family comparison:

| family | routes | R squared | mean abs residual |
|---|---:|---:|---:|
| `early_write_chain_proxy` | 4 | 0.123 | 1.5395 |
| `late_output_proxy` | 4 | 0.191 | 1.4700 |
| `qk_pointer` | 6 | 0.212 | 1.4389 |
| `qk_plus_early_write` | 10 | 0.235 | 1.4385 |
| `qk_plus_write_plus_output` | 14 | 0.249 | 1.4409 |

This was the important check.

The early write-chain proxy did **not** become the dominant answer-margin closure variable during `1500 -> 2500`.

That means:

```text
L0H0 write-side birth is real,
but route-score growth in this early write proxy does not by itself explain most answer-margin movement.
```

The interval-level closure also shows the fit is uneven:

| interval | actual mean Δ | predicted mean Δ | R squared |
|---|---:|---:|---:|
| `1500 -> 1750` | 0.891 | 0.784 | 0.379 |
| `1750 -> 2000` | 1.926 | 1.150 | 0.194 |
| `2000 -> 2250` | 0.458 | 0.229 | 0.133 |
| `2250 -> 2500` | 0.239 | 0.359 | -0.104 |

So this window is not a clean linear route-to-margin story.

The strongest positive route contributions in the full family were mostly broad residual terms:

```text
full_layer0_query_key:       +1.811
full_layer0_support_value:   +0.295
embedding_value_identity:    +0.027
embedding_key_identity:      +0.023
L0H0_ov_input_support_value: +0.012
L2H1_qk_query:               +0.0066
```

The important interpretation:

```text
during early formation, answer margin is moving through broad residual-state changes,
not only through the named low-rank head routes.
```

This does not invalidate the OV/write birth result.

It says the birth result lives one level below immediate behavior:

```text
optimizer state builds a useful write-side ingredient,
but the behavioral margin depends on wider residual geometry and downstream conversion.
```

### Updated Next Step

The next closure should not use moving answer margin as the only scalar.

For the early formation window, we need route-to-scalar closure against cleaner output scalars:

```text
negative_answer_loss
correct_value_logit
fixed_source_competitor_margin
fixed_target_competitor_margin
```

Reason:

```text
answer margin is branch-sensitive and mixes correct-logit growth with best-wrong-token movement.
early formation is exactly where those branches and residual states are still unstable.
```

The question becomes:

```text
do the QK/write/output routes close a cleaner scalar than raw moving answer margin?
```

If yes, the remaining gap is mostly scalar choice.

If no, the gap is genuinely residual/nonlinear and the next proof object should be residual-state closure rather than route-score closure.

### Route-To-Scalar Closure Result

We tested whether the weak formation-window answer-margin closure was just a bad scalar choice.

Artifacts:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/answer_scalar_residual_diagnosis/qk_ov_output_routes_1500_2500_formation/
artifacts/runs/symbolic_kv_reference_formation/analysis/route_to_scalar_closure/qk_ov_output_routes_1500_2500_formation/
```

First, scalar residual diagnosis showed:

| scalar | first-order mean abs error | first-order sign match |
|---|---:|---:|
| `negative_answer_loss` | 0.450 | 1.000 |
| `fixed_target_competitor_margin` | 0.527 | 0.250 |
| `moving_answer_margin` | 0.615 | 1.000 |
| `fixed_source_competitor_margin` | 1.112 | 1.000 |
| `correct_value_logit` | 1.734 | 1.000 |

This says:

```text
negative_answer_loss is the cleanest scalar for local first-order update prediction.
moving answer margin has branch-switch problems but still gets direction right.
correct_value_logit changes a lot, but first-order magnitude misses more.
```

Then route-to-scalar closure asked whether the measured QK/OV/output route-score deltas explain each scalar's actual movement.

Result:

| scalar | R squared | mean actual Δ | mean predicted Δ | mean abs residual |
|---|---:|---:|---:|---:|
| `correct_value_logit` | 0.373 | 2.315 | 1.429 | 1.717 |
| `fixed_target_competitor_margin` | 0.253 | 0.213 | 0.283 | 1.484 |
| `moving_answer_margin` | 0.249 | 0.878 | 0.630 | 1.441 |
| `fixed_source_competitor_margin` | 0.217 | 2.085 | 1.286 | 1.923 |
| `negative_answer_loss` | 0.098 | 0.733 | 0.341 | 0.882 |

This is a useful negative result.

Changing the scalar helps somewhat:

```text
correct_value_logit closes better than moving answer margin.
```

But it does not solve closure:

```text
best R^2 is only 0.373.
negative_answer_loss is locally predictable from full parameter updates,
but not well explained by the selected route-score family.
```

That distinction matters.

There are two different questions:

```text
Can the parameter update predict the scalar?
Can the selected route scores explain the scalar?
```

For `negative_answer_loss`, the first answer is relatively good and the second answer is poor.

So the gap is not just scalar choice.

The selected QK/OV route labels are missing an important part of the residual transformation.

Top route-to-scalar contributions also point to broad residual terms, not isolated head routes:

```text
correct_value_logit:
  full_layer0_query_key         +2.970
  full_layer1_query_key         -1.208
  full_layer1_post_attn_pred    -1.208
  full_layer0_support_value     +0.595
  L0H0_ov_input_support_value   +0.210

negative_answer_loss:
  full_layer2_post_attn_pred    +0.572
  full_layer1_query_key         -0.327
  full_layer1_post_attn_pred    -0.327
  full_layer0_query_key         +0.235
  embedding_value_identity      +0.132
```

The interpretation is now sharper:

```text
early write-side formation is real and optimizer-explained,
but answer behavior is mediated by broad residual-state transformations.
Named low-rank QK/OV route labels are not enough to close behavior during early formation.
```

### Current Next Step

The next proof object should be residual-state closure, not another isolated head route.

We need to test whether residual states at a few stages explain scalar movement better than route scores:

```text
embedding
layer_0_post_mlp
layer_1_post_mlp
layer_2_post_attn
layer_2_post_mlp
```

The target question:

```text
does patching or regressing broad residual-state deltas close correct_value_logit / negative_answer_loss movement
better than QK/OV route-score deltas?
```

If yes, the paper should say:

```text
the optimizer builds identifiable route geometry,
but behavioral closure at formation time lives in residual-state dynamics.
```

If no, then the remaining gap is likely nonlinear optimizer/model interaction rather than missing route labels.

## Write-Side State Conversion: MLP Local Weight Maps

We then tested the write-side interpretation directly at the local weight-map level.

Artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_local_write_map/l0h0_mlp_write_maps_1500_2500_formation/
```

The question was:

```text
when L0H0 changes the prediction-slot residual state,
which later weight maps turn that perturbation into answer/value geometry?
```

For each selected MLP, the tool defines the local map:

```text
F_i(z) = MLP_i(LN_2(z))
delta_in = z_clean - z_L0H0_ablated
actual_delta_out = F_i(z_clean) - F_i(z_L0H0_ablated)
jvp_ablated = J_F_i(z_L0H0_ablated) @ delta_in
```

Then it asks whether `actual_delta_out` and the local Jacobian-vector prediction point into contextual `answer_value`, `support_value`, and `query_key` subspaces.

The strongest result is that `L1MLP` and `L2MLP` are the main local write-side converters.

Source-endpoint answer/support-value summary:

| component | step | input delta norm | actual output delta norm | answer overlap | JVP answer overlap | JVP cosine to actual | JVP relative error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `L1MLP` | `1500` | `7.024` | `4.205` | `0.799` | `0.794` | `0.9996` | `0.109` |
| `L1MLP` | `1750` | `8.657` | `8.107` | `0.957` | `0.954` | `0.9992` | `0.113` |
| `L1MLP` | `2000` | `7.231` | `12.690` | `0.973` | `0.979` | `0.9969` | `0.302` |
| `L1MLP` | `2250` | `6.904` | `13.863` | `0.966` | `0.982` | `0.9944` | `0.486` |
| `L2MLP` | `1500` | `14.924` | `2.565` | `0.642` | `0.635` | `0.9999` | `0.119` |
| `L2MLP` | `1750` | `18.876` | `7.883` | `0.902` | `0.898` | `0.9996` | `0.271` |
| `L2MLP` | `2000` | `21.063` | `18.881` | `0.954` | `0.950` | `0.9980` | `0.385` |
| `L2MLP` | `2250` | `22.034` | `22.203` | `0.955` | `0.950` | `0.9957` | `0.436` |

`L0MLP` is weaker and earlier:

| component | step | actual output delta norm | answer overlap | JVP cosine to actual |
|---|---:|---:|---:|---:|
| `L0MLP` | `1500` | `3.748` | `0.564` | `0.990` |
| `L0MLP` | `1750` | `4.522` | `0.640` | `0.966` |
| `L0MLP` | `2000` | `3.137` | `0.517` | `0.954` |
| `L0MLP` | `2250` | `2.657` | `0.451` | `0.960` |

This gives a much clearer write-side story:

```text
L0H0 does not directly write a clean answer vector.
L0H0 perturbs the current prediction-slot residual state.
L0MLP weakly shapes that perturbation.
L1MLP sharply converts it into answer/support-value geometry.
L2MLP strongly amplifies and finalizes the answer/value-coded direction.
```

The local Jacobian-vector products are important.

They show this is not just an activation correlation:

```text
the local weight map of L1MLP/L2MLP sends the L0H0-caused residual perturbation
in almost the same direction as the actual nonlinear forward-pass output change.
```

The direction is very well explained:

```text
L1MLP JVP cosine to actual: about 0.994-0.999
L2MLP JVP cosine to actual: about 0.995-0.999
```

The magnitude is not fully closed:

```text
later relative errors are about 0.30-0.49.
```

So the supported claim is:

```text
the write side is a dense residual-state conversion implemented by local MLP weight maps,
not a single clean OV head writing the final answer directly.
```

The next optimizer question is:

```text
does AdamW also build the L1MLP/L2MLP local write conversion,
or does optimizer-state dominance mainly apply to the QK routing side?
```

## Write-Side Converter Chain: Full Component Rescue And Gradient Subspace Tests

After the local MLP write-map result, we tested whether the write side is a compact low-rank direction or a broader downstream converter chain.

The source is still `L0H0`. The question is:

```text
when L0H0 is ablated, which downstream writes can put back the missing behavior?
```

The strongest full-chain artifact is:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/component_output_rescue/l0h0_full_chain_over_time_1500_2500/
```

The tested patch groups were:

```text
L0MLP+L1H3+L1MLP+L2MLP
L0MLP+L1MLP+L2MLP
L1H3+L1MLP+L2MLP
L1MLP+L2MLP
L0MLP+L2MLP
L2MLP
```

The target-endpoint aggregate result is:

| scalar | best group | damage | rescue | rescue fraction | improved fraction | corr |
|---|---|---:|---:|---:|---:|---:|
| `correct_value_logit` | `L0MLP+L1H3+L1MLP+L2MLP` | `6.687` | `2.795` | `0.418` | `0.926` | `0.927` |
| `fixed_source_competitor_margin` | `L0MLP+L1H3+L1MLP+L2MLP` | `3.224` | `1.231` | `0.382` | `0.807` | `0.773` |
| `fixed_target_competitor_margin` | `L0MLP+L1H3+L1MLP+L2MLP` | `2.347` | `0.496` | `0.211` | `0.758` | `0.576` |
| `negative_answer_loss` | `L0MLP+L1H3+L1MLP+L2MLP` | `2.583` | `0.859` | `0.333` | `0.830` | `0.288` |

The main comparison is that groups containing `L0MLP` work much better than groups without it.

For `correct_value_logit`, target endpoint:

| patch group | rescue fraction |
|---|---:|
| `L0MLP+L1H3+L1MLP+L2MLP` | `0.418` |
| `L0MLP+L1MLP+L2MLP` | `0.401` |
| `L0MLP+L2MLP` | `0.265` |
| `L1MLP+L2MLP` | `0.187` |
| `L1H3+L1MLP+L2MLP` | `0.184` |
| `L2MLP` | `0.098` |

For margin and loss, the pattern is sharper. Removing `L0MLP` often makes the patch weak or harmful:

| scalar | `L0MLP+L1H3+L1MLP+L2MLP` | `L1MLP+L2MLP` | `L2MLP` |
|---|---:|---:|---:|
| `fixed_target_competitor_margin` | `0.211` | `-0.070` | `-0.062` |
| `negative_answer_loss` | `0.333` | `-0.088` | `-0.052` |

The time-sweep shows that this is already present by step `1750`, not a late artifact:

| step | scalar | full-chain rescue fraction |
|---:|---|---:|
| `1750` | `correct_value_logit` | `0.395` |
| `2000` | `correct_value_logit` | `0.404` |
| `2250` | `correct_value_logit` | `0.411` |
| `2500` | `correct_value_logit` | `0.444` |
| `1750` | `negative_answer_loss` | `0.393` |
| `2250` | `negative_answer_loss` | `0.334` |
| `2500` | `negative_answer_loss` | `0.341` |

This changes the write-side story:

```text
L0H0 does not hand the answer directly to the unembedding.
It creates a residual perturbation at the prediction slot.
L0MLP is the first important converter of that perturbation.
L1H3/L1MLP add smaller intermediate processing.
L2MLP is part of the downstream output path, but later functional-subspace tests show its own MLP output is not the main positive converter.
```

So the write side is not a clean standalone `W_OV` writer. It is a downstream converter chain.

### Low-Rank Subspace Tests

We then asked whether a small subspace of the converter writes is enough.

PCA artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/causal_write_subspace_rescue/l0h0_joint_mlp_answer_value_support_prediction_1500_2500/
```

Gradient-selected subspace artifacts:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/causal_write_gradient_subspace_rescue/l0h0_joint_mlp_support_prediction_1500_2500/
artifacts/runs/symbolic_kv_reference_formation/analysis/causal_write_gradient_subspace_rescue/l0h0_converter_chain_support_prediction_1500_2500/
```

The PCA result is important but negative in the right way. Rank-8 PCA captures almost all component-delta norm, but it does not cleanly rescue behavior.

For `L1MLP+L2MLP`, target endpoint, rank 8:

| basis | position group | `correct_value_logit` rescue | `fixed_target_competitor_margin` rescue | `negative_answer_loss` rescue |
|---|---|---:|---:|---:|
| `all_delta_pca` | `support_value+prediction` | `0.146` | negative / weak | negative / weak |
| `identity_delta_pca` | `support_value+prediction` | `0.149` | negative / weak | negative / weak |

That means the biggest write-delta directions are not the same thing as the causally useful answer directions.

The scalar-gradient-selected basis is smaller but cleaner.

For `L1MLP+L2MLP`, target endpoint, rank 8:

| scalar | rescue fraction |
|---|---:|
| `correct_value_logit` | `0.063` |
| `fixed_source_competitor_margin` | `0.043` |
| `fixed_target_competitor_margin` | `0.009` |
| `negative_answer_loss` | `0.018` |

Adding the upstream converter pieces improves it:

For `L0MLP+L1H3+L1MLP+L2MLP`, target endpoint, rank 8:

| scalar | rescue fraction | rescue | projection fraction |
|---|---:|---:|---:|
| `correct_value_logit` | `0.136` | `0.909` | `0.123` |
| `fixed_source_competitor_margin` | `0.102` | `0.329` | `0.118` |
| `fixed_target_competitor_margin` | `0.015` | `0.036` | about `0.12` |
| `negative_answer_loss` | `0.068` | `0.175` | `0.128` |

The diagnostic detail matters: the gradient-basis run produced `1024` basis rows, with `896` full-rank cases and `128` zero-gradient cases. The zero-gradient cases were the expected `L2MLP` / `support_value` combinations: after `L2MLP`, no later cross-position operation can move support-position information to the prediction position.

### Current Write-Side Conclusion

The supported write-side claim is:

```text
the value/write half of the circuit is a broad residual converter chain,
not a single low-rank OV map.
```

Full component writes recover about `33%` to `42%` of the L0H0 ablation damage on the best output scalars. Low-rank or gradient-selected projections recover much less, even when they point in the right direction. That means superposition is still active: the useful write effect is spread through a wide state transformation, and the compact subspaces only capture a small behavior-aligned slice.

This does not close the whole answer-margin story yet. It gives a stronger write-side mechanism:

```text
QK side: a low-rank route matcher forms.
write side: L0H0 perturbs prediction state, then MLP converter writes transform it into answer/value geometry.
```

The next optimizer test is to ask whether AdamW builds this converter chain the same way it built the QK route.

### Functional Subspace Split: The Write Signal Lives At The Prediction Slot

We then made the write-side question more precise.

Instead of asking only whether a downstream component can rescue behavior, we asked:

```text
what vector does L0H0 add to the residual stream?
which scalar-gradient/read directions see that vector?
does the downstream MLP transform the vector, or mostly pass it through?
```

The tool was:

```text
mlp-input-functional-subspace-report
```

Artifacts:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_input_functional_subspace/l0h0_to_l0mlp_support_prediction_1500_2500/
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_input_functional_subspace/l0h0_to_l1mlp_support_prediction_1500_2500/
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_input_functional_subspace/l0h0_to_l2mlp_prediction_1500_2500/
```

For an MLP block, the report computes:

```text
delta_in = z_clean[input_stage] - z_L0H0_ablated[input_stage]

mlp_output_delta =
  MLP(z_clean[input_stage]) - MLP(z_L0H0_ablated[input_stage])

skip_dot =
  grad_scalar(output_stage) dot delta_in

mlp_dot =
  grad_scalar(output_stage) dot mlp_output_delta

total_dot =
  grad_scalar(output_stage) dot (delta_in + mlp_output_delta)
```

So this separates the carried residual signal from the MLP-transformed signal.

The prediction-position aggregate is:

| component | total scalar-relevant effect | residual skip/direct part | MLP-transformed part | mean `delta_in` overlap with read subspace | mean MLP-output overlap with read subspace |
|---|---:|---:|---:|---:|---:|
| `L0MLP` | `1648.453` | `1219.088` | `429.365` | `0.259` | `0.162` |
| `L1MLP` | `1329.142` | `1399.580` | `-70.438` | `0.146` | `0.042` |
| `L2MLP` | `1187.722` | `1319.112` | `-131.390` | `0.064` | `0.039` |

This changes the write-side interpretation again.

`L0MLP` is the only tested MLP where the local MLP-transformed part is clearly positive and large. Its transformed part contributes about `429` scalar-dot units, while the residual skip/direct part contributes about `1219`.

For `L1MLP` and `L2MLP`, the residual signal remains strongly useful, but the MLP-transformed part is negative on aggregate. These later MLPs are therefore not the main local positive converters of the L0H0-caused signal. They mostly carry a useful prediction-position residual direction forward, while their own MLP outputs slightly oppose it in this scalar split.

The position split is decisive:

| position | aggregate scalar-relevant effect |
|---|---:|
| `prediction` | `4165.317` |
| `support_value` | `-17.307` |

The `support_value` result is tiny compared with the prediction-position effect. The failed first `L2MLP` run is also informative: `L2MLP` at the `support_value` position had zero gradient to the answer scalar, because after that point there is no later cross-position operation that can move support-position information into the prediction answer.

The supported write-side mechanism is now:

```text
L0H0 creates a prediction-position residual perturbation.
L0MLP partly converts and amplifies it.
After L0MLP, most of the useful signal is carried in the residual stream itself.
L1MLP and L2MLP are part of the downstream model state, but their local MLP outputs are not the main positive writer for this L0H0-caused signal.
```

This means the write-side object is not a clean `W_OV` value writer and not a simple MLP chain where each MLP converts more. The better object is:

```text
the prediction-position functional residual subspace caused by L0H0/L0MLP
```

That subspace is what should be tracked in the next phase.

## 2026-04-29 Update: OV Write-Side Functional Subspace And AdamW Split

This update closes the next layer of the OV/write-side question for the reference seed.

The earlier write-side result said:

```text
L0H0 has a real early value/write signal.
That signal is not a clean standalone OV vector.
The useful effect travels through the residual stream and downstream components.
```

The new result makes that more precise.

The write-side object is:

```text
the L0H0-caused prediction-position residual perturbation,
as read by the mature answer/value readout directions around L0MLP.
```

In plain language:

```text
L0H0 does not simply write "the answer token" into the residual stream.
It changes the residual vector at the prediction position.
That change already points partly in a useful mature direction early in training.
Around 1500 -> 1750, AdamW makes that direction strongly usable by the answer readout.
L0MLP then adds a smaller nonlinear conversion on top.
```

This is the current OV-side mechanism.

### Why Static OV Was The Wrong Object

For QK, the matrix story was clean:

```text
W_QK = W_Q W_K^T
```

and the useful scalar was a route score:

```text
C_QK(theta)
  = E[score(prediction, support_value)
      - mean score(prediction, distractors)]
```

That works because QK is a routing map:

```text
query vector dot key vector -> attention score
```

OV is different.

OV writes into the residual stream:

```text
head_output = attention @ V
residual_write = head_output W_O
```

But that residual write is not judged immediately.

It is later processed by:

```text
residual addition
layer norm
later attention
later MLPs
final norm
unembedding
```

So the right OV question is not:

```text
does W_OV point directly at the answer embedding?
```

The right question is:

```text
does the residual vector written by L0H0 land in directions
that the later network actually reads as answer evidence?
```

This is why the stronger proof object became a contextual residual-write scalar, not raw `W_OV` SVD.

### Local Functional Decomposition

For an MLP block, define:

```text
F_l(z) = MLP_l(LN_2(z))
```

For a clean run and an `L0H0`-ablated run, define the L0H0-caused input change at a selected residual stage:

```text
delta_in
  = z_clean[input_stage] - z_L0H0_ablated[input_stage]
```

The MLP output change caused by that perturbation is:

```text
mlp_output_delta
  = F_l(z_clean[input_stage])
    - F_l(z_L0H0_ablated[input_stage])
```

The post-MLP residual change is therefore:

```text
post_mlp_total_delta
  = delta_in + mlp_output_delta
```

For an output scalar `s`, such as fixed-source or fixed-target answer margin, define the readout gradient:

```text
g_s = grad_z s
```

Then the scalar-relevant write effect is:

```text
C_total = E[g_s . post_mlp_total_delta]
        = E[g_s . delta_in] + E[g_s . mlp_output_delta]
        = C_skip + C_mlp
```

This split asks:

```text
is the useful write signal already present in the residual stream,
or does the MLP create most of it?
```

The answer is:

```text
most of it is already in the L0H0-caused residual perturbation;
L0MLP adds a smaller positive conversion.
```

### Prediction Slot, Not Support Slot

The functional-subspace report compared where the L0H0-caused signal matters.

Artifacts:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_input_functional_subspace/l0h0_to_l0mlp_support_prediction_1500_2500/
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_input_functional_subspace/l0h0_to_l1mlp_support_prediction_1500_2500/
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_input_functional_subspace/l0h0_to_l2mlp_prediction_1500_2500/
```

Aggregate scalar-relevant effect:

| position | aggregate effect |
|---|---:|
| `prediction` | `4165.317` |
| `support_value` | `-17.307` |

So the useful write-side signal is overwhelmingly at the prediction/read position.

This matters because it rules out a misleading picture:

```text
L0H0 writes a value at the support position,
then some later component simply moves it.
```

The better picture is:

```text
L0H0 changes the current prediction-position state.
That state already contains value-relevant information.
Downstream readout directions increasingly use it.
```

### Which MLP Actually Helps?

The local split showed:

| component | total scalar-relevant effect | residual skip/direct part | MLP-transformed part | `delta_in` read overlap | MLP-output read overlap |
|---|---:|---:|---:|---:|---:|
| `L0MLP` | `1648.453` | `1219.088` | `429.365` | `0.259` | `0.162` |
| `L1MLP` | `1329.142` | `1399.580` | `-70.438` | `0.146` | `0.042` |
| `L2MLP` | `1187.722` | `1319.112` | `-131.390` | `0.064` | `0.039` |

The key point:

```text
L0MLP is the only tested MLP whose own output adds a clearly positive local conversion.
L1MLP and L2MLP mostly preserve/carry the useful residual direction,
while their local MLP outputs are negative in this scalar split.
```

This refines the earlier "MLP converter chain" story.

The downstream model is still necessary, but the strongest local positive conversion for the L0H0-caused prediction-slot signal is at `L0MLP`.

### Fixed Mature Functional Subspace Trajectory

Next we asked whether the useful write direction is born as a new direction or whether it exists earlier but becomes useful later.

Artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_functional_subspace_trajectory/l0h0_to_l0mlp_prediction_ref2500_0750_3500/
```

We fixed reference subspaces at step `2500`, then measured every checkpoint against the same basis.

For example, define a mature readout basis at step `2500`:

```text
B_ref = PCA_top4({g_s(step 2500, x_i)})
```

Then for each checkpoint:

```text
overlap(delta_in(t), B_ref)
overlap(mlp_output_delta(t), B_ref)
E[g_s(t) . post_mlp_total_delta(t)]
```

The result was surprising.

The mature-looking write directions are already present early:

| step | `delta_in` overlap with step-2500 `input_delta` basis | MLP-output overlap with step-2500 MLP-output basis |
|---:|---:|---:|
| `750` | `0.661` | `0.721` |
| `1000` | `0.655` | `0.735` |
| `1250` | `0.606` | `0.728` |
| `1500` | `0.608` | `0.750` |
| `1750` | `0.583` | `0.722` |
| `2500` | `0.596` | `0.724` |

So the write side does not look like:

```text
random direction -> new mature direction
```

Instead it looks like:

```text
mature-ish direction exists early,
but it is weakly coupled to the answer readout.
```

The scalar-relevant effect is tiny until the birth window:

| step | total functional write effect | residual skip part | L0MLP transformed part |
|---:|---:|---:|---:|
| `750` | `2.864` | `-1.257` | `4.121` |
| `1000` | `2.607` | `-0.924` | `3.531` |
| `1250` | `2.225` | `-0.155` | `2.380` |
| `1500` | `2.647` | `2.472` | `0.174` |
| `1750` | `50.185` | `32.584` | `17.601` |
| `2000` | `64.801` | `48.013` | `16.788` |
| `2250` | `66.713` | `51.059` | `15.654` |
| `2500` | `91.624` | `70.177` | `21.446` |
| `2750` | `104.699` | `79.161` | `25.537` |
| `3500` | `92.627` | `79.046` | `13.582` |

This gives the current best formation story for the write side:

```text
the direction exists early,
but answer-readout coupling turns on sharply around 1500 -> 1750.
```

That is different from QK.

For QK, the low-rank route direction itself visibly forms.

For the write side, the direction is partly present early; the functional coupling is what crystallizes.

### Fixed-Readout AdamW Attribution

The next experiment asked why the functional write coupling grows.

We fixed the mature step-2500 readout vector and measured:

```text
C_write(t)
  = E[ r_ref(x_i) . post_mlp_total_delta(t, x_i) ]
```

where:

```text
r_ref(x_i) = post-MLP scalar gradient at step 2500
post_mlp_total_delta = delta_in + mlp_output_delta
```

Then for each one-step optimizer interval:

```text
Delta C_write
  ~= grad_theta C_write(theta_t) . Delta theta_actual
```

and the actual AdamW update was decomposed into:

```text
raw SGD
clipped SGD
Adam current-gradient component
Adam historical momentum component
Adam preconditioned total
weight decay
reconstructed AdamW update
```

Artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_functional_write_adam_state_attribution/l0h0_l0mlp_prediction_ref2500_postgrad_total_1500_2500/
```

Across `1500 -> 2500`, averaged across the four fixed-margin endpoint/readout variants:

| quantity | value |
|---|---:|
| actual scalar growth | `1.015` |
| first-order actual-update prediction | `1.034` |
| raw SGD fraction | `0.124%` |
| Adam current-gradient fraction | `11.50%` |
| Adam momentum fraction | `91.35%` |
| weight decay fraction | `-2.85%` |
| actual-update sign match | `99.7%` |
| reconstructed AdamW sign match | `100%` |
| mean reconstruction relative error | `0.00010` |

This is the central optimizer result for the write side:

```text
raw SGD barely moves the fixed functional-write scalar.
AdamW momentum carries almost all of the useful write-coupling update.
```

The timing is also sharp.

Summed over the four scalar variants:

| window | actual change | predicted change | raw SGD | Adam current | Adam momentum | weight decay |
|---|---:|---:|---:|---:|---:|---:|
| `1500 -> 1750` | `+5.978` | `+5.992` | `+0.00584` | `+0.563` | `+5.449` | `-0.0197` |
| `1750 -> 2000` | `-0.375` | `-0.363` | `+0.00005` | `-0.006` | `-0.325` | `-0.0324` |
| `2000 -> 2250` | `-0.738` | `-0.716` | `-0.00043` | `-0.0459` | `-0.638` | `-0.0322` |
| `2250 -> 2500` | `-0.804` | `-0.777` | `-0.00032` | `-0.0362` | `-0.709` | `-0.0315` |

So the write-side fixed scalar is mostly born in:

```text
1500 -> 1750
```

Afterward it partially relaxes or redistributes.

This matches the trajectory result:

```text
the direction was present before;
the useful readout coupling turns on in the birth window.
```

### Residual Write Versus L0MLP Conversion

Finally, we split:

```text
post_mlp_total_delta = delta_in + mlp_output_delta
```

into two separate AdamW attribution runs.

Artifacts:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_functional_write_adam_state_attribution/l0h0_l0mlp_prediction_ref2500_postgrad_input_delta_1500_2500/
artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_functional_write_adam_state_attribution/l0h0_l0mlp_prediction_ref2500_postgrad_mlp_output_delta_1500_2500/
```

The split result:

| part | actual growth | share of total | predicted growth | raw SGD fraction | Adam current fraction | Adam momentum fraction | weight decay fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `input_delta` | `0.789` | `~78%` | `0.802` | `0.157%` | `11.22%` | `91.23%` | `-2.45%` |
| `mlp_output_delta` | `0.226` | `~22%` | `0.232` | `-0.010%` | `13.12%` | `92.37%` | `-5.51%` |

So most of the write-side birth is not the L0MLP nonlinear output.

Most of it is:

```text
the L0H0-caused residual perturbation itself,
arriving at the L0MLP input / prediction slot.
```

L0MLP adds a smaller positive nonlinear correction.

The timing split is:

| window | `input_delta` actual | `mlp_output_delta` actual |
|---|---:|---:|
| `1500 -> 1750` | `+3.858` | `+2.120` |
| `1750 -> 2000` | `+0.204` | `-0.579` |
| `2000 -> 2250` | `-0.410` | `-0.328` |
| `2250 -> 2500` | `-0.494` | `-0.310` |

Again, both pieces mainly turn on in `1500 -> 1750`.

The parameter-group split is also clean.

For `input_delta`, `L0MLP` contributes exactly zero because this scalar is measured before L0MLP acts:

| parameter group | predicted growth |
|---|---:|
| `L0H0.qkvo` | `3.143` |
| `L0 attention block` | `3.143` |
| `L0MLP` | `0.000` |

For `mlp_output_delta`, `L0MLP` is the main converter:

| parameter group | predicted growth |
|---|---:|
| `L0MLP` | `0.882` |
| `L0H0.qkvo` | `0.442` |
| `L0 attention block` | `0.048` |

These parameter groups overlap, so they should not be added as independent effects.

But the interpretation is clear:

```text
L0H0 carries the main residual write.
L0MLP carries the nonlinear conversion.
AdamW momentum builds both.
```

### Current Full OV Picture

The current OV/write-side mechanism is:

```text
1. L0H0 develops an early value-bearing write route.

2. The useful write is not mainly a clean W_OV singular vector.
   It is a contextual residual perturbation at the prediction position.

3. The mature-looking residual/write directions are partly present early.
   What forms sharply is their coupling to answer/value readout directions.

4. The coupling birth happens mainly in 1500 -> 1750.

5. Most of the useful effect is the residual write itself:
   delta_in explains about 78% of the fixed-readout write growth.

6. L0MLP adds a smaller positive nonlinear conversion:
   mlp_output_delta explains about 22%.

7. Raw SGD contributes almost nothing to either part.

8. In this reference-seed fixed-readout split, AdamW momentum carries about 91% -> 92% of both pieces.
```

In simple terms:

```text
QK answers "where should I read?"
OV/write answers "what useful residual state gets created after reading?"

For this model, the answer to the second question is:
L0H0 creates a useful prediction-slot residual state,
L0MLP slightly transforms it,
and AdamW momentum is the optimizer-state mechanism that makes it functionally useful.
```

This is the write-side complement to the QK result.

QK formation was:

```text
AdamW builds a low-rank support-value matcher in W_QK.
```

Write-side formation is:

```text
AdamW makes an already-present contextual residual direction become readout-useful,
mostly through L0H0's residual write and secondarily through L0MLP conversion.
```

### What This Solves And What It Does Not

This solves a specific missing link:

```text
the OV/write side is not just "messy."
It has a measurable functional subspace,
a clear prediction-position location,
a residual-vs-MLP split,
and an exact AdamW decomposition.
```

It also explains why the earlier low-rank OV route labels did not close behavior:

```text
the useful write-side object is a contextual residual-state coupling,
not one static OV matrix direction.
```

But this still does not prove full behavioral sufficiency.

Open gaps:

```text
the full answer-margin movement is not fully closed by the selected route families;
downstream residual interactions still matter;
normalization and later component interactions may explain part of the remaining gap;
plain SGD-vs-AdamW optimizer ablation is still not done;
cross-seed validation of this exact functional-write split was not part of this 2026-04-29 reference-seed update.
```

The supported claim is now:

```text
In the reference seed, both halves of the lookup circuit have optimizer-level explanations.

QK:
  AdamW builds a low-rank support-value route matcher.

OV/write:
  AdamW momentum builds functional coupling between L0H0's prediction-slot residual write
  and mature answer/value readout directions, with L0MLP adding a smaller nonlinear conversion.
```

## 2026-04-30 Update: Computation Ledger, Scalar Closure, And Cross-Seed Write Validation

This update turns the recent runs into the paper-facing accounting layer.

The central reason for the ledger is that three claims are different:

```text
causal claim:
  ablating or patching the object changes behavior

dynamic claim:
  actual optimizer updates built the object during training

computational claim:
  the object implements a specific part of the lookup algorithm
```

The current ledger is:

| object | math target | artifact family | status |
|---|---|---|---|
| behavior | `m_t = logit(correct) - max wrong logit` | best checkpoint, answer-scalar diagnostics | learned lookup supported |
| QK route | `C_QK = E[score(prediction, support) - mean score(prediction, distractors)]` | QK route geometry, contextual alignment, route attribution | strong computational story |
| QK weight birth | `W_QK = W_Q W_K^T = U Sigma V^T` | `weight_svd_trace`, rank-8 QK reports | low-rank matcher birth supported |
| QK optimizer cause | `Delta C_QK ~= grad C_QK . Delta theta_actual` | exact from-init AdamW attribution, cross-seed winner/bottom controls | raw SGD tiny; AdamW state carries route growth |
| write functional subspace | `C_write = E[g_ref . delta_write]` | `mlp_input_functional_subspace`, functional trajectory | supported; contextual residual subspace, not clean `W_OV` |
| write optimizer cause | `Delta C_write ~= grad C_write . Delta theta_actual` | cross-seed `mlp_functional_write_adam_state_attribution` | raw SGD tiny; AdamW-preconditioned update carries write growth |
| scalar closure | `Delta s ~= beta^T Delta routes` | `route_to_scalar_closure`, `route_family_closure`, `output_route_closure` | partial route closure; stronger output-space closure |

The ledger changes the paper's wording.

The old loose statement was:

```text
the OV/write side is still not optimizer explained
```

That is now too weak.

The supported statement is:

```text
the write side is not a clean static W_OV matrix story;
it is a prediction-position functional residual-subspace story.

That write subspace is validated across seeds,
and exact AdamW attribution shows raw SGD is tiny while
AdamW-preconditioned updates carry the useful write growth.
```

### Cross-Seed Functional Write Validation

The cross-seed write audit ran `28 / 28` functional-subspace reports.

The selected winner write sources were:

| seed | source head | downstream MLP |
|---:|---|---|
| `0011` | `L1H3` | `L1MLP` |
| `0013` | `L1H3` | `L1MLP` |
| `0017` | `L1H1` | `L1MLP` |
| `0023` | `L2H1` | `L2MLP` |
| `0029` | `L1H1` | `L1MLP` |

Final-step functional write effect, grouped by role:

| scalar | winner mean | runner mean | bottom mean |
|---|---:|---:|---:|
| `fixed_source_competitor_margin` | `510.43` | `388.22` | `177.01` |
| `negative_answer_loss` | `415.64` | `195.40` | `9.63` |

The winner/runner/bottom ordering is the important validation.

It says:

```text
the write role repeats across seeds;
the component address changes;
bottom controls do not carry the same functional write effect.
```

The split inside the winning write effect is also stable:

| scalar | residual-skip fraction | local MLP-output fraction |
|---|---:|---:|
| `fixed_source_competitor_margin` | `0.902` | `0.098` |
| `negative_answer_loss` | `0.908` | `0.092` |

So the write-side object is mostly:

```text
the source-head-caused residual perturbation at the prediction slot
```

not:

```text
a local MLP-created answer vector
```

The MLP is still part of the readout boundary, but most of the measured functional signal is already in the residual write.

### Cross-Seed Write AdamW Attribution

The write-side AdamW attribution ran on the five selected winner source-to-MLP paths over `1500 -> 2500`.

Aggregate over the selected answer-value write scalars:

| aggregate | actual growth | first-order predicted growth | raw SGD / predicted | Adam current / predicted | Adam momentum / predicted | weight decay / predicted |
|---|---:|---:|---:|---:|---:|---:|
| all write scalars | `23.843` | `41.636` | `1.22%` | `86.00%` | `15.39%` | `-1.40%` |
| `fixed_source_competitor_margin` | `14.802` | `24.840` | `1.14%` | `81.68%` | `19.54%` | `-1.22%` |
| `negative_answer_loss` | `9.041` | `16.795` | `1.34%` | `92.40%` | `9.26%` | `-1.66%` |

Per-seed split:

| seed | path | actual | predicted | raw SGD / pred | current / pred | momentum / pred | decay / pred |
|---:|---|---:|---:|---:|---:|---:|---:|
| `0011` | `L1H3 -> L1MLP` | `3.180` | `9.011` | `2.17%` | `125.36%` | `-23.32%` | `-2.04%` |
| `0013` | `L1H3 -> L1MLP` | `7.112` | `9.426` | `0.91%` | `53.99%` | `47.21%` | `-1.20%` |
| `0017` | `L1H1 -> L1MLP` | `5.929` | `14.695` | `1.34%` | `115.98%` | `-14.02%` | `-1.96%` |
| `0023` | `L2H1 -> L2MLP` | `2.054` | `3.185` | `1.02%` | `73.91%` | `26.73%` | `-0.64%` |
| `0029` | `L1H1 -> L1MLP` | `5.570` | `5.319` | `-0.07%` | `0.47%` | `99.07%` | `0.46%` |

This is not the same as the reference-seed-only statement that momentum carries almost everything.

The better cross-seed statement is:

```text
raw SGD is consistently tiny;
AdamW-preconditioned updates carry the useful write growth;
the split between current-gradient and historical momentum is seed-dependent.
```

That distinction matters. The QK route result is much more cleanly momentum-heavy across the traced winner runs. The write-side result is broader: AdamW state/preconditioning matters, but the current-vs-momentum decomposition changes with seed and write address.

### Scalar Closure During The Formation Window

We then asked whether measured QK/write/output route-score deltas close the answer scalar.

The relevant formation window is:

```text
1500 -> 2500
```

because this is where the write functional coupling turns on.

The 14-route family includes QK pointer terms, early write terms, and late output proxies.

Route-to-scalar closure:

| scalar | observations | actual mean delta | predicted mean delta | mean abs residual | R squared |
|---|---:|---:|---:|---:|---:|
| `correct_value_logit` | `512` | `2.315` | `1.429` | `1.717` | `0.373` |
| `fixed_source_competitor_margin` | `512` | `2.085` | `1.286` | `1.923` | `0.217` |
| `fixed_target_competitor_margin` | `512` | `0.213` | `0.283` | `1.484` | `0.253` |
| `moving_answer_margin` | `512` | `0.878` | `0.630` | `1.441` | `0.249` |
| `negative_answer_loss` | `512` | `0.733` | `0.341` | `0.882` | `0.098` |

Family-level answer-margin closure:

| family | routes | observations | mean predicted delta | R squared |
|---|---:|---:|---:|---:|
| `qk_pointer` | `6` | `512` | `0.531` | `0.212` |
| `qk_plus_early_write` | `10` | `512` | `0.635` | `0.235` |
| `qk_plus_write_plus_output` | `14` | `512` | `0.630` | `0.249` |

This is partial closure, not full closure.

It supports:

```text
the measured route/write coordinates are behaviorally meaningful
```

but it does not support:

```text
these 14 route scores fully explain the answer margin
```

Output-route closure is stronger in the same window:

| scalar | observations | actual mean delta | predicted mean delta | mean abs residual | R squared |
|---|---:|---:|---:|---:|---:|
| `correct_value_logit` | `512` | `2.315` | `2.043` | `0.962` | `0.837` |
| `fixed_source_competitor_margin` | `512` | `2.085` | `1.505` | `1.377` | `0.576` |
| `fixed_target_competitor_margin` | `512` | `0.213` | `0.346` | `1.186` | `0.508` |
| `moving_answer_margin` | `512` | `0.878` | `0.359` | `1.377` | `0.340` |
| `negative_answer_loss` | `512` | `0.733` | `0.091` | `0.866` | `0.022` |

This says the answer scalar is easier to explain once we move into output/readout space.

The scalar hierarchy is now:

```text
cleanest local proof scalars:
  correct_value_logit
  fixed-source competitor margin
  fixed-target competitor margin

branch-sensitive scalar:
  moving answer margin

harder nonlinear scalar:
  negative answer loss in the 1500 -> 2500 output-route closure
```

The current closure claim is therefore:

```text
QK/write route scores partially close answer-scalar movement.
Output-space readout deltas close much more of the local correct-logit/fixed-margin movement.
Raw moving answer margin remains branch-sensitive.
Full answer-margin closure by a small causal route set remains open.
```

## 2026-05-05 Update: Branch/Fixed-Scalar Closure Consolidation

This run consolidated the moving-margin warning on the main `1500 -> 2500`
formation window. The goal was not to find a new circuit component. The goal was
to check whether the usual moving answer margin is a reliable scalar for
formation closure.

Artifacts:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/answer_scalar_residual_diagnosis/branch_fixed_scalar_closure_1500_2500/
artifacts/runs/symbolic_kv_reference_formation/analysis/route_to_scalar_closure/branch_fixed_scalar_closure_1500_2500/
artifacts/runs/symbolic_kv_reference_formation/analysis/output_route_closure/branch_fixed_scalar_closure_1500_2500/
artifacts/runs/symbolic_kv_reference_formation/analysis/answer_margin_branch_decomposition/branch_fixed_scalar_closure_1500_2500/
```

### What Changed Relative To The Older Branch Audit

The older branch-aware result used a larger row set and showed that branch-aware
correction can strongly improve switch-row closure. This new run uses the main
`1500 -> 2500` checkpoint intervals and the same `support_value` / `query_key`
pairs used in the output-route closure story:

```text
observations: 512
checkpoint steps: 1500, 1750, 2000, 2250, 2500
pair types: support_value, query_key
```

So the numbers are not interchangeable with the older `6400`-row audit. They are
a matched audit for the paper's main formation window.

### Competitor Branch Switching Is A Real Measurement Problem

In this window, the best wrong-token competitor changes often:

| bucket | observations | competitor switches | switch fraction |
|---|---:|---:|---:|
| `all` | `512` | `312` | `0.609` |
| `competitor_switch` | `312` | `312` | `1.000` |
| `same_competitor` | `200` | `0` | `0.000` |

This is much stronger than a small nuisance effect. In this formation window,
most moving-margin rows change their wrong-token branch.

The branch correction also carries substantial energy:

| bucket | target-branch correction energy / moving-margin energy | source-branch correction energy / moving-margin energy |
|---|---:|---:|
| `all` | `0.550` | `0.243` |
| `competitor_switch` | `0.716` | `0.317` |
| `same_competitor` | `0.000` | `0.000` |

For switch rows, about 72% of moving-margin energy is tied to the target-branch
correction term. That means a moving-margin line integral can fail because the
quantity being explained changed branches, not because the internal route is
absent.

### Output-Space Closure Confirms The Scalar Hierarchy

The output-route closure result gives the cleanest scalar comparison:

| scalar | observations | R squared | mean abs residual |
|---|---:|---:|---:|
| `correct_value_logit` | `512` | `0.868` | `0.998` |
| `fixed_source_competitor_margin` | `512` | `0.639` | `1.420` |
| `fixed_target_competitor_margin` | `512` | `0.558` | `1.301` |
| `moving_answer_margin` | `512` | `0.407` | `1.392` |
| `negative_answer_loss` | `512` | `0.183` | `0.909` |

On competitor-switch rows specifically:

| scalar | observations | R squared | mean abs residual |
|---|---:|---:|---:|
| `correct_value_logit` | `312` | `0.894` | `1.026` |
| `fixed_source_competitor_margin` | `312` | `0.671` | `1.461` |
| `fixed_target_competitor_margin` | `312` | `0.608` | `1.406` |
| `moving_answer_margin` | `312` | `0.416` | `1.607` |
| `negative_answer_loss` | `312` | `0.221` | `1.058` |

The scalar hierarchy is now very clear:

```text
cleanest:
  correct_value_logit

good fixed-branch proof targets:
  fixed_source_competitor_margin
  fixed_target_competitor_margin

branch-sensitive:
  moving_answer_margin

harder nonlinear objective:
  negative_answer_loss
```

### Branch-Aware Moving-Margin Closure

The branch-aware comparison says:

| bucket | direct moving R^2 | source-fixed + branch R^2 | target-fixed + branch R^2 |
|---|---:|---:|---:|
| `all` | `0.407` | `0.418` | `0.489` |
| `competitor_switch` | `0.416` | `0.506` | `0.517` |
| `same_competitor` | `0.596` | `0.596` | `0.596` |

When the competitor does not switch, all three measurements collapse to the same
quantity. When the competitor switches, fixed-branch plus exact branch correction
is cleaner than direct moving-margin closure.

### Paper-Level Interpretation

This supports a methodological claim, not just a limitation:

```text
Moving answer margin is often the first scalar people reach for, but it is not
always a stable scalar during training.

If the best wrong-token branch changes, the scalar being explained changes.
Fixed-branch and output-space scalars are better proof targets for formation
audits.
```

This closes gap-filler experiment A:

```text
A. consolidate branch-aware closure on the existing reference run
```

It does not close the optimizer-necessity gap. The remaining necessary gap-filler
is still the optimizer ablation:

```text
AdamW baseline
AdamW beta1 = 0
AdamW altered beta2
SGD + momentum
plain SGD
```

### Current Paper-Level Claim After This Update

The paper can now say:

```text
In symbolic KV lookup, training repeatedly forms a support-value retrieval role.

QK side:
  the role becomes a low-rank W_QK matcher;
  exact AdamW decomposition explains its growth;
  raw SGD is far too small.

Write side:
  the role is not a clean W_OV singular-vector story;
  it is a contextual prediction-position residual-write subspace;
  the write role validates across seeds under different component addresses;
  exact AdamW attribution shows AdamW-preconditioned updates carry the write growth;
  raw SGD is again tiny.

Closure:
  route/write scalars are meaningful but partial;
  output-space scalars are stronger;
  full answer-margin sufficiency is still open.
```

## 2026-05-05 Update: Optimizer Ablation Pilot And SGD LR Sweep

This run addresses the largest optimizer-level gap in the paper draft:

```text
Does AdamW merely explain the AdamW-trained trajectory,
or is AdamW-style adaptive optimization actually important for forming the lookup role?
```

The result is not a theorem that SGD can never learn symbolic KV lookup. It is a
matched-budget optimizer ablation:

```text
same model
same seed
same dataset
same 6000-step budget
same checkpoint/evaluation schedule
AdamW variants versus SGD variants
```

### Artifacts

Pilot optimizer-ablation runs:

```text
artifacts/runs/symbolic_kv_optimizer_ablation/adamw_baseline/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/adamw_beta1_0/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/adamw_beta2_0999/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_momentum_09/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_plain/seed_0007/
```

SGD learning-rate sweep:

```text
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_momentum_09_lr_0p00003/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_momentum_09_lr_0p00010/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_momentum_09_lr_0p00030/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_momentum_09_lr_0p00100/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_momentum_09_lr_0p00300/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_plain_lr_0p00001/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_plain_lr_0p00003/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_plain_lr_0p00010/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_plain_lr_0p00030/seed_0007/
artifacts/runs/symbolic_kv_optimizer_ablation/sgd_plain_lr_0p00100/seed_0007/
```

QK/OV progress reports:

```text
artifacts/runs/symbolic_kv_optimizer_ablation/*/seed_0007/analysis/ov_write_progress/all_heads_0750_6000_optimizer_ablation/
artifacts/runs/symbolic_kv_optimizer_ablation/*/seed_0007/analysis/ov_write_progress/all_heads_0750_6000_sgd_lr_sweep/
```

Each completed QK/OV progress report has `3072` checkpoint rows for the LR
sweep runs.

### Pilot Result: AdamW Variants Learn, Matched SGD Does Not

Final step `6000` behavior:

| optimizer variant | validation answer accuracy | heldout answer accuracy | top QK head | top QK separation | support attention mass |
|---|---:|---:|---|---:|---:|
| `adamw_baseline` | `0.976` | `0.702` | `L2H1` | `8.029` | `0.893` |
| `adamw_beta1_0` | `0.984` | `0.546` | `L1H2` | `9.259` | `0.941` |
| `adamw_beta2_0999` | `0.985` | `0.608` | `L1H2` | `7.330` | `0.920` |
| `sgd_momentum_09` | `0.002` | `0.001` | `L1H3` | `0.057` | `0.038` |
| `sgd_plain` | `0.000` | `0.000` | `L1H3` | `0.109` | `0.038` |

The important surprise is `adamw_beta1_0`.

Removing AdamW's first-moment momentum does not prevent the lookup role from
forming. The role still forms strongly, but the winning head moves from `L2H1`
to `L1H2`.

So the result is not:

```text
beta1 momentum is strictly necessary.
```

The result is:

```text
AdamW-style adaptive/preconditioned optimization forms the lookup role here;
matched SGD and SGD+momentum do not.
```

### SGD Learning-Rate Sweep Result

The LR sweep tested whether SGD failed only because the baseline learning rate
was poorly chosen.

Final and best observed behavior:

| SGD variant | learning rate | best validation answer accuracy | best heldout answer accuracy | final validation loss |
|---|---:|---:|---:|---:|
| `sgd_momentum_09` | `0.00003` | `0.0000` | `0.0000` | `39.876` |
| `sgd_momentum_09` | `0.00010` | `0.0000` | `0.0000` | `5.369` |
| `sgd_momentum_09` | `0.00030` | `0.0016` | `0.0000` | `2.763` |
| `sgd_momentum_09` | `0.00100` | `0.0060` | `0.0040` | `2.485` |
| `sgd_momentum_09` | `0.00300` | `0.0085` | `0.0051` | `2.325` |
| `sgd_plain` | `0.00001` | `0.0000` | `0.0000` | `88.512` |
| `sgd_plain` | `0.00003` | `0.0000` | `0.0000` | `85.751` |
| `sgd_plain` | `0.00010` | `0.0000` | `0.0000` | `75.417` |
| `sgd_plain` | `0.00030` | `0.0000` | `0.0000` | `39.831` |
| `sgd_plain` | `0.00100` | `0.0000` | `0.0000` | `5.356` |

The best SGD run is `sgd_momentum_09` at learning rate `0.003`, but its answer
accuracy is still below `1%`.

### SGD Learns Some Surface Structure But Not Lookup

The highest-learning-rate SGD+momentum run is not completely inert:

| metric | `sgd_momentum_09`, lr `0.003` |
|---|---:|
| validation token accuracy | `0.340` |
| validation read-key accuracy | `0.349` |
| validation write-key accuracy | `0.282` |
| validation answer accuracy | `0.0085` |

So SGD begins to model some syntax/key-position regularities, but it does not
learn the value lookup algorithm.

This distinction matters. The failure is not merely:

```text
SGD produced random outputs.
```

It is more specific:

```text
SGD can move into shallow task structure under this budget,
but it does not crystallize the support-value retrieval role.
```

### QK Route Does Not Form Under SGD

The QK/OV progress reports show that the support-value route itself fails to
grow under every SGD sweep run.

The best QK separation observed across the SGD LR sweep is only about:

```text
max QK separation:          0.118
support attention mass:     0.039
probe answer accuracy:      0.000
```

This maximum occurs at step `750`, not after a clean training birth window. That
looks like initialization-level noise, not learned route formation.

By contrast, AdamW variants at step `6000` show:

```text
AdamW baseline:
  QK separation:          8.029
  support attention mass: 0.893

AdamW beta1 = 0:
  QK separation:          9.259
  support attention mass: 0.941

AdamW beta2 = 0.999:
  QK separation:          7.330
  support attention mass: 0.920
```

### Paper-Level Interpretation

This closes the main optimizer-ablation gap in a bounded way:

```text
In the matched 6000-step seed-7 ablation, AdamW variants learn the symbolic KV
lookup role and form a strong support-value QK route.

Across a reasonable SGD learning-rate sweep, SGD and SGD+momentum do not learn
the answer behavior and do not form the support-value route.
```

The strongest honest claim is:

```text
AdamW-style adaptive/preconditioned optimization is important for forming this
lookup role under the studied training budget and recipe.
```

The result does not prove:

```text
SGD can never learn the same task with more steps, other schedules, larger
learning-rate sweeps, different initialization scales, or tuned regularization.
```

But it does rule out the simplest reviewer objection:

```text
Maybe the AdamW story is irrelevant because plain SGD under the same recipe
would form the same circuit.
```

Under the tested matched recipe and LR sweep, it does not.

### Updated Gap Status

Closed or substantially reduced:

```text
A. branch/fixed-scalar closure on the reference run
D. optimizer ablation: AdamW variants versus SGD variants under matched budget
```

Still optional / future:

```text
B. negative-control route attribution
C. route remove/restore causal sufficiency
E. harder-task or larger-scale generalization
```

For the paper, this should replace the older limitation row:

```text
plain SGD-vs-AdamW ablation: not run
```

with:

```text
matched-budget SGD-vs-AdamW ablation: run for seed 7;
AdamW variants learn and form the role;
SGD LR sweep does not.
```

## 2026-05-05 Update: Prediction-Position Value Code Is Causal And Broad

This update addresses the next concrete gap in the OV/write-side story.

The earlier write-side evidence showed that the useful write is not a clean
standalone `W_OV` singular-vector object. It is a contextual residual-state
conversion: L0H0 perturbs the prediction state, and downstream components turn
that perturbation into answer evidence.

The missing question was:

```text
what is the residual object that the downstream readout actually uses?
```

The new answer is:

```text
the prediction-position residual contains a value-token identity code;
that code becomes behaviorally useful in the same 1500 -> 1750 formation band;
and removing it causally damages answer behavior.
```

This still does not make the write side a clean low-rank OV story. The value
code is broad and high-dimensional.

### Artifacts

Value-code subspace trajectory:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_subspace/prediction_answer_value_0750_3500/
```

Value-code causal interventions:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_causal_intervention/embedding_value_identity_prediction_layer2_remove_rank16_1500_3500/
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_causal_intervention/embedding_key_identity_prediction_layer2_remove_rank7_1500_3500/
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_causal_intervention/embedding_value_identity_prediction_layer2_remove_rank7_1500_3500/
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_causal_intervention/embedding_value_identity_prediction_layer2_keep_rank16_1500_3500/
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_causal_intervention/embedding_value_identity_prediction_layer2_keep_rank32_2000_3500/
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_causal_intervention/embedding_value_identity_prediction_layer2_keep_rank64_2000_3500/
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_causal_intervention/embedding_value_identity_prediction_layer2_keep_rank96_2000_3500/
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_causal_intervention/embedding_value_identity_prediction_layer2_keep_rank127_2000_3500/
```

### Value Code Turns On At The Prediction Position

The value-code report tracks clean residual vectors at `prediction` and
`support_value` positions, grouped by `answer_value` and `support_value`.

At the prediction position, the final readout margin is negative before the
write-side formation window and becomes strongly positive immediately after it:

| step | final-norm prediction value accuracy | final-norm prediction value margin |
|---:|---:|---:|
| `750` | `0.0196` | `-0.844` |
| `1000` | `0.0131` | `-0.797` |
| `1250` | `0.0131` | `-0.724` |
| `1500` | `0.0719` | `-0.752` |
| `1750` | `0.6078` | `0.666` |
| `2000` | `0.6797` | `2.730` |
| `2500` | `0.6471` | `3.031` |
| `3000` | `0.7190` | `3.898` |
| `3500` | `0.7647` | `5.264` |

This is the important transition:

```text
before 1500: prediction state does not read out the answer value
after 1750:  prediction state carries answer-value evidence
```

That matches the previously observed write-side formation window.

The support-value position already contains value identity early. That by
itself is not the learned lookup algorithm. The learned part is moving usable
value evidence into the prediction position.

### Removing Value Identity Hurts The Mature Circuit

At `layer_2_post_mlp / prediction`, removing the rank-16
`embedding_value_identity` subspace produces a large causal effect on
`validation_iid` after the value code turns on:

| step | baseline margin | intervened margin | margin drop | baseline acc | intervened acc | acc drop |
|---:|---:|---:|---:|---:|---:|---:|
| `1500` | `-0.752` | `-0.921` | `0.169` | `0.0719` | `0.0261` | `0.0458` |
| `1750` | `0.666` | `-1.157` | `1.822` | `0.6078` | `0.2092` | `0.3987` |
| `2000` | `2.730` | `-1.130` | `3.859` | `0.6797` | `0.2941` | `0.3856` |
| `2500` | `3.031` | `-0.634` | `3.666` | `0.6471` | `0.3595` | `0.2876` |
| `3000` | `3.898` | `-0.263` | `4.161` | `0.7190` | `0.4183` | `0.3007` |
| `3500` | `5.264` | `0.297` | `4.967` | `0.7647` | `0.4771` | `0.2876` |

So the prediction-position value-code subspace is not merely readable. It is
causally used by the answer computation.

### Rank-Matched Key Control Is Weaker

The `embedding_key_identity` subspace has centered rank `7`, so the fair
rank-matched control is value rank `7` versus key rank `7`.

On `validation_iid`:

| step | value rank-7 margin drop | key rank-7 margin drop | value rank-7 acc drop | key rank-7 acc drop |
|---:|---:|---:|---:|---:|
| `1750` | `0.861` | `0.788` | `0.183` | `0.157` |
| `2000` | `1.739` | `0.871` | `0.137` | `0.0719` |
| `2500` | `1.705` | `0.635` | `0.0980` | `0.0131` |
| `3000` | `1.808` | `0.718` | `0.111` | `0.0523` |
| `3500` | `2.294` | `0.593` | `0.0850` | `0.0000` |

The key subspace can perturb margins, especially early, but the mature answer
behavior depends much more specifically on value identity.

### Low-Rank Value Identity Is Not Sufficient

Keeping only rank-16 value identity at `layer_2_post_mlp / prediction` is not
enough. On `validation_iid` at step `3500`:

```text
baseline:      margin  5.264, accuracy 0.765
keep rank 16: margin -4.248, accuracy 0.451
```

This rules out a too-clean story:

```text
answer behavior = one small value-code subspace
```

The rank sweep shows the value code is broad:

| kept value rank | step-3500 validation margin | step-3500 validation accuracy |
|---:|---:|---:|
| `16` | `-4.248` | `0.451` |
| `32` | `5.894` | `0.654` |
| `64` | `9.548` | `0.732` |
| `96` | `8.039` | `0.719` |
| `127` | `5.707` | `0.758` |

At rank `127`, keeping only the value-identity subspace almost preserves
`validation_iid` behavior:

```text
baseline:      margin 5.264, accuracy 0.7647
keep rank127: margin 5.707, accuracy 0.7582
```

The same near-sufficiency appears at step `2000`:

```text
baseline:      margin 2.730, accuracy 0.6797
keep rank127: margin 2.825, accuracy 0.6667
```

This means the write/readout side is not low-rank in the same way QK is.

The QK side forms a compact pointer. The write/readout side forms a broad
value-token identity code at the prediction position.

### Split Boundary

The near-sufficiency result is strongest on `validation_iid` and
`counterfactual`.

At step `3500`, rank-127 keep gives:

| split | baseline acc | keep-rank127 acc | baseline margin | keep-rank127 margin |
|---|---:|---:|---:|---:|
| `validation_iid` | `0.7647` | `0.7582` | `5.264` | `5.707` |
| `counterfactual` | `0.7857` | `0.7792` | `5.587` | `6.125` |
| `heldout_pairs` | `0.2288` | `0.2288` | `-6.297` | `-9.511` |
| `structural_ood` | `0.1659` | `0.1475` | `-3.532` | `-7.598` |

So the current claim should not be generalized to all splits.

The mature IID/counterfactual circuit uses a broad value-code subspace, but the
same intervention does not close heldout-pair or structural-OOD behavior.

### Updated Full-Circuit Interpretation

The current best circuit-level picture is:

```text
1. contextual support states exist at earlier value positions
2. QK routing selects the support-value position
3. L0H0 and downstream write components perturb the prediction residual
4. the prediction residual becomes aligned with broad value-token identity geometry
5. the tied embedding/unembedding readout turns that value identity into the correct value logit
```

This is now stronger than the earlier statement:

```text
L0H0 writes useful information into the prediction residual
```

The more precise version is:

```text
the write side creates a prediction-position value-code state.
That state is broad in the embedding-value identity geometry.
Removing it hurts behavior; keeping nearly all of it almost preserves IID behavior.
```

### What This Closes

Closed or substantially reduced:

```text
exact nature of the downstream readout object:
  not just "useful residual information";
  it is value-token identity geometry at the prediction position.

causal status of that object:
  value identity removal damages answer behavior;
  rank-matched key identity is weaker.

low-rank OV hypothesis:
  rejected for the write/readout side;
  the value-code object is broad/high-dimensional.
```

### What Remains Open

Still not closed:

```text
closed-form operator from support-value residual state to prediction value-code state
```

The existing evidence identifies the component chain and the causal residual
object, but it does not give a simple algebraic theorem of the form:

```text
component outputs implement exactly this matrix map from V_i at support to V_i at prediction
```

Also still open:

```text
neuron-level decomposition of the write/readout side
```

The current evidence points away from a clean neuron-level story and toward a
broad residual subspace. That should be stated as a limitation/future-work
boundary, not hidden.

### Paper-Level Status After This Update

The paper can now make the following stronger, bounded claim:

```text
In the reference run, the full lookup circuit decomposes into a compact QK
pointer and a broad value-code write/readout state.

The QK side identifies where to read. The write side converts that read into a
prediction-position residual state whose value-token identity geometry is
causally used by the final answer logit.
```

This is not yet a Neel-Nanda-style closed-form algorithm in a named analytic
basis like Fourier modes. But it is a substantially deeper circuit explanation
than component ablation:

```text
role scalar
weight-space route birth
optimizer-state attribution
cross-seed role/address validation
component/residual/write-chain evidence
causal value-code identification
branch-aware scalar boundary
optimizer ablation
```

That is the current proof boundary.

## Value-Code Transfer Map And Causal Rescue

The value-code intervention above tells me what residual object the readout
uses. The next question is harder:

```text
does the support-value state predict the prediction-position value-code state?
```

I tested this with two tools:

```text
value-code-transfer-map-report
value-code-transfer-rescue
```

The first is descriptive. It fits a ridge-stabilized affine map:

```text
support_value coordinates at layer_1_post_mlp
  -> prediction coordinates at layer_2_post_mlp
```

inside value-identity bases built on a deterministic fit split.

The second is causal. It removes the target value-code projection at
`layer_2_post_mlp / prediction`, then patches back either:

```text
1. the actual projected value-code component        (oracle)
2. the fitted support -> prediction transfer        (true transfer)
3. shuffled-answer / wrong-value / random controls  (controls)
```

Artifacts:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_transfer_map/support_to_prediction_1500_3500/
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_transfer_map/support_to_prediction_key_control_rank4/
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_transfer_rescue/support_to_prediction_rank16_1500_3500/
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_transfer_rescue/support_to_prediction_key_control_rank4/
```

### Transfer Map: Present, But Not A Full Operator

The support-to-prediction map is real but partial.

At `1750`, rank `16`:

| eval kind | coordinate R2 |
|---|---:|
| `true_transfer` | `0.3978` |
| `wrong_support_value` | `0.2522` |
| `random_subspace` | `0.1624` |
| `shuffled_answer_value` | `-0.2536` |

At `3500`, rank `16`:

| eval kind | coordinate R2 | centroid acc | stage-lens acc | stage-lens margin |
|---|---:|---:|---:|---:|
| `true_transfer` | `0.2236` | `0.1923` | `0.0476` | `-6.855` |
| `random_subspace` | `0.1348` | `0.1538` | `0.0238` | `-9.095` |
| `wrong_support_value` | `0.0189` | `0.0000` | `0.0238` | `-9.220` |
| `shuffled_answer_value` | `-0.6931` | `0.0769` | `0.0000` | `-9.417` |

The true transfer beats controls, especially in coordinate prediction and
late-stage readout margin. But the stage-lens accuracy remains low. So this is
not a clean theorem of the form:

```text
linear map from support value-code alone = final answer state
```

It is a measurable linear component of the write/readout bridge.

The key-identity control is weaker. At `2500`, rank `4`:

| eval kind | coordinate R2 | stage-lens margin |
|---|---:|---:|
| `true_transfer` | `0.1696` | `-3.520` |
| `key_identity` | `0.0138` | `-3.622` |

This matters because it rules out a simpler explanation:

```text
the transfer is just carrying the key identity
```

### Transfer Rescue: The Target Code Is Causal, And The Transfer Partly Replaces It

The oracle patch is the sanity check. If removing the target value-code
projection hurts behavior and patching the same projection back rescues it,
then the removed value-code component is causally used.

That check passes.

At rank `16`, `layer_2_post_mlp / prediction`, the oracle rescue fraction is
`1.0` by construction for the removed value-code projection, and the behavioral
scalars recover accordingly.

The true transfer is strongest on `negative_answer_loss`, which is the most
stable differentiable answer scalar in this rescue setting.

| step | oracle rescue | true transfer | shuffled | wrong value | random |
|---:|---:|---:|---:|---:|---:|
| `1750` | `1.000` | `0.918` | `0.865` | `0.864` | `0.931` |
| `2000` | `1.000` | `0.884` | `0.635` | `0.864` | `0.881` |
| `2500` | `1.000` | `0.835` | `0.453` | `0.671` | `0.754` |
| `3000` | `1.000` | `0.877` | `0.574` | `0.584` | `0.769` |
| `3500` | `1.000` | `0.949` | `0.576` | `0.484` | `0.760` |

On value accuracy, the mature checkpoint also shows a useful gap:

| step | oracle rescue | true transfer | shuffled | wrong value | random |
|---:|---:|---:|---:|---:|---:|
| `3500` | `1.000` | `0.750` | `0.000` | `0.500` | `0.500` |

So the transfer is not merely readable. It can causally replace a substantial
part of the target value-code component for the loss/readout behavior.

### Moving Margin Still Does Not Fully Close

The moving value margin remains messy.

At `2500`, rank `16`:

| eval kind | value-margin rescue fraction |
|---|---:|
| `oracle_actual_projected` | `1.000` |
| `true_transfer` | `0.421` |
| `random_subspace` | `0.186` |
| `wrong_support_value` | `-0.007` |
| `shuffled_answer_value` | `-0.314` |

At `3500`, rank `16`:

| eval kind | value-margin rescue fraction |
|---|---:|
| `oracle_actual_projected` | `1.000` |
| `true_transfer` | `-0.080` |
| `wrong_support_value` | `-0.727` |
| `random_subspace` | `-2.132` |
| `shuffled_answer_value` | `-3.543` |

This should not be hidden. It says the transfer map explains a useful
answer-evidence component, but not the whole moving-margin object.

Given the earlier branch-switching result, this is exactly where fixed-branch
transfer rescue should be used next. The moving best-wrong token can change
under patching, so margin can make a useful transfer look worse than it is.

### Key-Control Rescue

The rank-4 key-control rescue confirms the same boundary.

At `2500`, rank `4`:

| scalar | oracle | true transfer | key identity |
|---|---:|---:|---:|
| `negative_answer_loss` | `1.000` | `0.846` | `0.746` |
| `value_margin` | `1.000` | `-0.480` | `-1.705` |

The key control can recover some loss because the residual system is highly
coupled, but true value transfer is better. On moving margin, neither is a
clean rescue, and key identity is worse.

### Updated Boundary

This closes more of the OV/write side:

```text
support value-code has a measurable linear relationship to prediction value-code
prediction value-code is causally used
the fitted transfer can partially replace the removed prediction value-code
the effect is strongest on loss/readout usability
moving margin remains branch-sensitive and only partially rescued
```

The current best statement is therefore:

```text
The write side contains a causal support -> prediction value-code transfer
component, but it is not a complete low-rank linear OV theorem.
```

### Fixed-Branch Transfer Rescue

I reran the transfer rescue with fixed-branch scalar scoring added to the same
tool:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_transfer_rescue/support_to_prediction_rank16_fixed_branch_1750_3500/
```

This adds two fixed-competitor margins:

```text
fixed_clean_competitor_margin
  hold the clean model's best wrong value fixed

fixed_removed_competitor_margin
  hold the target-subspace-removed model's best wrong value fixed
```

This asks whether the bad moving-margin result was mostly caused by the
best-wrong branch moving under the patch.

The answer is mixed, and useful.

For `true_transfer`, rescue fractions are:

| step | moving margin | fixed clean branch | fixed removed branch | negative loss | value accuracy |
|---:|---:|---:|---:|---:|---:|
| `1750` | `0.860` | `0.904` | `0.807` | `0.918` | `0.546` |
| `2000` | `0.632` | `1.049` | `0.582` | `0.884` | `0.286` |
| `2500` | `0.421` | `0.853` | `0.474` | `0.835` | `-0.750` |
| `3000` | `0.017` | `0.824` | `0.252` | `0.877` | `0.167` |
| `3500` | `-0.080` | `0.650` | `0.320` | `0.949` | `0.750` |

This is exactly the split I needed.

The moving value margin makes the late transfer look almost absent:

```text
3500 moving margin rescue fraction: -0.080
```

But the clean-branch fixed margin still shows a real transfer:

```text
3500 fixed-clean competitor rescue fraction: 0.650
```

and loss rescue remains very strong:

```text
3500 negative-answer-loss rescue fraction: 0.949
```

So the remaining margin failure is partly a branch/competitor problem, not an
absence of value-code transfer.

However, fixed-removed branch rescue is weaker:

```text
3500 fixed-removed competitor rescue fraction: 0.320
```

That means branch switching is not the whole story. The fitted linear transfer
captures a useful answer-evidence direction, but it still does not reproduce
the exact margin geometry induced by removing and restoring the target
value-code projection.

The controls sharpen this. At step `3500`:

| scalar | true transfer | shuffled | wrong value | random |
|---|---:|---:|---:|---:|
| `fixed_clean_competitor_margin` | `0.650` | `1.437` | `0.626` | `1.001` |
| `fixed_removed_competitor_margin` | `0.320` | `-0.321` | `0.146` | `0.017` |
| `negative_answer_loss` | `0.949` | `0.576` | `0.484` | `0.760` |
| `value_accuracy` | `0.750` | `0.000` | `0.500` | `0.500` |

The fixed-clean scalar is not selective enough by itself: random and shuffled
can look strong because that branch is anchored to the clean model's wrong
token and the target-subspace removal changes the residual geometry in a
non-local way.

The more reliable readout is:

```text
negative_answer_loss + fixed-removed competitor margin + value accuracy
```

Together they say:

```text
the transfer carries real answer evidence
it beats branch-destroying controls on loss and accuracy
it partially rescues a fixed removed-branch margin
it still does not close the full margin geometry
```

### Updated Transfer Boundary

The fixed-branch rescue changes the boundary from:

```text
maybe the transfer only helps loss but not margin
```

to:

```text
the transfer helps stable answer evidence and some fixed-branch margins,
but the exact support -> prediction write operator is still more than one
rank-16 linear map in value-code coordinates.
```

The next targeted gap is no longer generic fixed-branch rescue. That is done.
The next gap is the nonlinear/contextual part of the write operator:

```text
what residual context or component-local nonlinear transformation makes the
linear transfer incomplete?
```

That would require a contextual transfer model, not another pure linear
support-to-prediction map.

### Contextual Transfer Rescue

I then tested exactly that contextual hypothesis:

```text
does prediction-position context explain the missing write operator?
```

Artifact:

```text
artifacts/runs/symbolic_kv_reference_formation/analysis/value_code_transfer_rescue/support_to_prediction_context_rank16_1750_3500/
```

The contextual rescue extends the transfer model from:

```text
support_value_code -> prediction_value_code
```

to:

```text
[support_value_code, prediction_context_code] -> prediction_value_code
```

where the context is `layer_1_post_mlp / prediction`, rank `16`.

The result is important, but not in the naive direction. Adding prediction
context strongly improves the rescue, but `context_only` is already very
strong. That means the missing operator is not just a better support-to-
prediction linear map. By this point in training, the prediction-position
residual context already contains most of the information needed to restore the
removed value-code component.

For `source_plus_context`, rescue fractions are:

| step | moving margin | fixed removed branch | negative loss | value accuracy |
|---:|---:|---:|---:|---:|
| `1750` | `0.980` | `0.968` | `0.993` | `1.091` |
| `2000` | `0.944` | `0.945` | `1.004` | `1.286` |
| `2500` | `0.836` | `0.858` | `0.995` | `1.250` |
| `3000` | `0.410` | `0.872` | `0.974` | `0.833` |
| `3500` | `-0.205` | `0.754` | `1.005` | `0.875` |

Compared to source-only transfer, the contextual model is a large improvement:

| step | scalar | source-only | source + context |
|---:|---|---:|---:|
| `2500` | `fixed_removed_competitor_margin` | `0.473` | `0.858` |
| `2500` | `negative_answer_loss` | `0.835` | `0.995` |
| `2500` | `value_margin` | `0.421` | `0.836` |
| `3000` | `fixed_removed_competitor_margin` | `0.252` | `0.872` |
| `3000` | `negative_answer_loss` | `0.877` | `0.974` |
| `3000` | `value_margin` | `0.017` | `0.410` |
| `3500` | `fixed_removed_competitor_margin` | `0.320` | `0.754` |
| `3500` | `negative_answer_loss` | `0.949` | `1.005` |
| `3500` | `value_accuracy` | `0.750` | `0.875` |

But the context-only rows are also near-oracle on the stable scalars:

| step | scalar | context-only rescue |
|---:|---|---:|
| `2500` | `fixed_removed_competitor_margin` | `0.780` |
| `2500` | `negative_answer_loss` | `0.968` |
| `3000` | `fixed_removed_competitor_margin` | `0.718` |
| `3000` | `negative_answer_loss` | `0.940` |
| `3500` | `fixed_removed_competitor_margin` | `0.640` |
| `3500` | `negative_answer_loss` | `0.959` |

This closes the interpretation boundary more tightly:

```text
the target prediction value-code is causal
source value-code transfer is real but partial
prediction-position context carries most of the missing rescue signal
source + context nearly closes loss and fixed-removed branch rescue
moving margin still remains unstable late in training
```

The strongest OV/write statement is now:

```text
The write side is a contextual prediction-position value-code restoration
mechanism. It is not a standalone low-rank W_OV operator and not a pure
support-value transfer. The prediction state already carries a readout-ready
context, and the support value-code helps shape or select that state.
```

This is probably the right stopping point for the OV closure experiments before
paper polishing. Going deeper would require decomposing the prediction context
itself into component-local nonlinear contributions, which is a new subproject
rather than a paper-gap filler.
