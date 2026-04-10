# Shared Feature Dynamics Plan

## Goal

Build a CLI-first analysis stack that tracks stable learned features across checkpoints and uses those features as the main dynamical objects for studying circuit formation.

The stack should support:

- one shared feature basis per stage
- feature trajectories across checkpoints
- feature birth and stabilization analysis
- feature-level source/target comparison
- feature-level causal patching
- lineage from features to heads, MLP blocks, and neuron groups
- graph and plot exports for immediate inspection

This is the next layer beyond:

- checkpoint sweep
- birth-window summaries
- residual patching
- head / MLP / neuron screening

## Scope

Initial scope should be one stage at a time.

Stage order:

1. `layer_2_post_mlp`
2. `final_norm`
3. other residual stages later
4. MLP hidden states only after the residual-stage stack is stable

Do not start with:

- one shared basis for the entire model
- global neuron-level trajectory tracking
- cross-run alignment across seeds

## Why This Layer Exists

The current local feature analysis is useful for checkpoint-pair inspection, but it does not provide stable feature identities across training.

Without stable feature IDs, we cannot cleanly answer:

- when a feature is born
- whether a feature is reinforced by SGD
- whether two checkpoints share the same mechanism
- whether heldout gains correspond to specific feature families

The shared-feature stack should convert the current analysis from:

- per-checkpoint local summaries

to:

- stable feature trajectories over time

## Core Entities

### 1. Shared Feature Basis

One basis per stage.

This is the canonical object that defines feature IDs.

Fields:

- `basis_id`
- `stage_name`
- `checkpoint_steps_used_for_fit`
- `probe_set_path`
- `num_features`
- `input_dim`
- `normalization`
- `fit_hyperparameters`
- `fit_metrics`

### 2. Feature

A feature is one latent dimension in the shared basis.

Fields:

- `feature_id`
- `basis_id`
- `decoder_norm`
- `encoder_norm`
- `global_mean_activation`
- `global_active_fraction`
- `global_answer_direction_alignment`

### 3. Feature Trajectory Row

One feature at one checkpoint.

Fields:

- `basis_id`
- `stage_name`
- `checkpoint_path`
- `step`
- `feature_id`
- `mean_activation`
- `active_fraction`
- `correctness_gap`
- `heldout_gap`
- `structural_ood_gap`
- `margin_correlation`
- `answer_direction_alignment`
- `split_mean_activation`

### 4. Feature Birth Event

Fields:

- `feature_id`
- `birth_metric`
- `birth_step`
- `stabilization_step`
- `peak_step`
- `peak_value`
- `pre_birth_mean`
- `post_birth_mean`
- `birth_window`

### 5. Feature Diff Row

Fields:

- `feature_id`
- `source_step`
- `target_step`
- `mean_activation_delta`
- `active_fraction_delta`
- `correctness_gap_delta`
- `heldout_gap_delta`
- `structural_ood_gap_delta`
- `margin_correlation_delta`

### 6. Feature Patch Result

Fields:

- `feature_id` or `feature_group_id`
- `source_step`
- `target_step`
- `patch_mode`
- `answer_accuracy_delta`
- `heldout_answer_accuracy_delta`
- `structural_ood_answer_accuracy_delta`
- `margin_delta`
- `reconstruction_error`

### 7. Feature Lineage Edge

Fields:

- `feature_id`
- `source_type`
- `source_id`
- `target_type`
- `target_id`
- `score`
- `score_type`

Examples:

- head -> feature
- MLP block -> feature
- neuron group -> feature
- feature -> readout

## Commands

## `shared-feature-fit`

### Purpose

Fit one shared feature basis on pooled activations from multiple checkpoints for one stage.

### Inputs

- `--config`
- `--probe-set`
- `--stage`
- `--checkpoint`
  - repeatable
- `--checkpoint-dir`
  - optional alternative to explicit checkpoints
- `--fit-step-stride`
  - optional, used only with `--checkpoint-dir`
- `--output-dir`
- `--device`
- `--num-features`
- `--train-steps`
- `--learning-rate`
- `--l1-coefficient`
- `--batch-size`

### Behavior

- load all selected checkpoints
- collect activations from the same stage on the same probe set
- normalize activations using one shared normalization scheme
- fit one SAE
- save the basis and a fit summary

### Outputs

- `shared_feature_basis.pt`
- `shared_feature_basis.json`
- `shared_feature_basis_features.json`

### Validation

Fail if:

- stage does not exist
- checkpoints do not exist
- probe set does not match benchmark
- fit quality is too poor

No hidden fallback to a different stage or different checkpoint set.

## `feature-trajectory-sweep`

### Purpose

Encode every checkpoint in one shared basis and write stable feature trajectories.

### Inputs

- `--config`
- `--probe-set`
- `--basis`
- `--checkpoint-dir`
- `--output-dir`
- `--device`

### Behavior

- load shared basis
- verify basis stage matches requested activations
- encode every checkpoint in that same basis
- write per-feature per-checkpoint rows
- write checkpoint-level summaries

### Outputs

- `feature_trajectories.jsonl`
- `feature_checkpoint_summary.json`
- `feature_split_profiles.json`

### Required Metrics

For each checkpoint-feature pair:

- mean activation
- active fraction
- correctness gap
- heldout gap
- structural OOD gap
- answer-direction alignment
- margin correlation
- split means

## `feature-birth-analyze`

### Purpose

Detect birth, stabilization, and drift of features over training.

### Inputs

- `--trajectories`
- `--output`
- `--metric`
  - repeatable
- `--threshold`
- `--delta-threshold`
- `--window`

### Metrics Supported Initially

- `mean_activation`
- `active_fraction`
- `correctness_gap`
- `heldout_gap`

### Behavior

Birth should be defined formally, not visually.

Recommended rule:

- crossing threshold
- sustained over a forward window
- sufficiently above previous-window mean

### Outputs

- `feature_births.json`
- `feature_birth_summary.json`
- `feature_birth_plot_data.json`

## `feature-compare`

### Purpose

Compare source and target checkpoints in the same basis.

### Inputs

- `--trajectories`
- `--source-step`
- `--target-step`
- `--output`
- `--top-k`

### Behavior

Rank features by:

- mean activation change
- heldout gap change
- correctness gap change
- structural OOD change
- absolute change

### Outputs

- `feature_compare_<source>_vs_<target>.json`
- `feature_compare_<source>_vs_<target>_plot_data.json`

## `feature-patch`

### Purpose

Run causal interventions in feature space.

### Inputs

- `--config`
- `--probe-set`
- `--basis`
- `--source-checkpoint`
- `--target-checkpoint`
- `--stage`
- `--feature`
  - repeatable
- `--feature-group`
  - optional file
- `--patch-mode`
  - `replace`, `ablate`, `additive_delta`
- `--output`
- `--device`

### Behavior

For selected features:

- encode target activations
- optionally encode source activations
- intervene in feature space
- decode back to residual space
- run model forward with stage patch

### Outputs

- `feature_patch_<source>_vs_<target>.json`
- `feature_patch_<source>_vs_<target>_examples.json`

### Required Reporting

- answer effect
- heldout effect
- structural OOD effect
- reconstruction error
- note that feature-space intervention is approximate

## `feature-lineage`

### Purpose

Map important features to concrete components.

### Inputs

- `--config`
- `--probe-set`
- `--basis`
- `--checkpoint`
- `--feature`
  - repeatable
- `--sweep-metrics`
  - optional for candidate filtering
- `--output`
- `--device`

### Behavior

Initial lineage methods:

- head ablation effect on feature activation
- MLP block ablation effect on feature activation
- neuron-group ablation effect on feature activation in selected candidate layers
- feature decoder alignment to answer direction

### Outputs

- `feature_lineage_<step>.json`
- `feature_lineage_graph_<step>.json`

## Graph and Plot Outputs

The CLI should also export graph- and plot-ready files from the start.

### Plot Files

- `feature_trajectory_topk.svg`
- `feature_heatmap.svg`
- `feature_birth_raster.svg`
- `feature_compare_bar.svg`
- `feature_split_profile.svg`

These should be generated from compact plot-data JSON files so we can later reuse them in a UI.

### Graph JSON

Graph outputs should contain:

- nodes
- edges
- labels
- checkpoint metadata
- score types

Node types:

- `stage`
- `feature`
- `head`
- `mlp_block`
- `neuron_group`
- `readout`

Edge types:

- `writes`
- `supports`
- `ablates`
- `patch_effect`
- `aligns_with`

## File Layout

Recommended output layout under one run:

```text
artifacts/runs/<run_name>/analysis/shared_features/<stage_name>/
  shared_feature_basis.pt
  shared_feature_basis.json
  shared_feature_basis_features.json
  feature_trajectories.jsonl
  feature_checkpoint_summary.json
  feature_split_profiles.json
  feature_births.json
  feature_birth_summary.json
  feature_compare_<source>_vs_<target>.json
  feature_patch_<source>_vs_<target>.json
  feature_lineage_<step>.json
  graphs/
  plots/
```

## Implementation Strategy

Build in dependency order, but as one milestone.

### 1. Shared Basis Backend

Implement:

- shared activation collection
- shared normalization
- shared SAE fit
- shared basis save/load

### 2. Trajectory Sweep

Implement:

- per-checkpoint encoding
- trajectory row writing
- checkpoint summaries

### 3. Birth Analysis

Implement:

- formal birth detection
- persistence windows
- ranked feature births

### 4. Diff

Implement:

- source-target feature comparison
- family-like ranking by metric deltas

### 5. Patch

Implement:

- feature ablation
- feature replacement from source to target
- reconstruction-error reporting

### 6. Lineage

Implement:

- head -> feature effect
- MLP block -> feature effect
- neuron-group -> feature effect

### 7. Plot / Graph Export

Implement:

- JSON plot data
- SVG plots
- graph JSON

## Integration With Existing Stack

Reuse existing infrastructure wherever possible.

Should reuse:

- current probe-set format
- checkpoint loading
- residual-stage extraction
- head / MLP / neuron masking
- existing sweep outputs for candidate filtering

Should not duplicate:

- benchmark loading
- checkpoint enumeration
- patching logic that already exists in residual compare tools

## Initial Stage Targets

Use this exact order:

1. `layer_2_post_mlp`
2. `final_norm`

Reason:

- both already emerged as the strongest late-stage change locations in the current analysis
- both are central to the current writeout/readout hypothesis

## Technical Risks

### 1. Basis Too Dense

If active fraction stays very high, the feature basis is not sparse enough to support clean claims.

Need explicit fit-quality reporting.

### 2. Feature IDs Still Unstable

If checkpoint pooling is too narrow or normalization is bad, features may still fail to represent stable families.

### 3. Approximate Patching

Feature-space patching requires decoding back to residual space.

This is approximate and must always report reconstruction error.

### 4. Lineage Noise

Feature lineage will be noisy if done before filtering to causally meaningful features.

Need thresholding and ranking.

## Success Criteria

This milestone is successful if we can say, for one stage:

- these are the stable features across training
- these are their activation trajectories
- these are their birth times
- these are the features that grew from source to target
- these are the features whose causal patching changes behavior
- these are the heads / MLP blocks / neuron groups most associated with those features

That would be the first genuinely feature-dynamical layer for the repo and a much stronger basis for answering the SGD-selection question.
