---
layout: default
title: "Artifact Map"
description: Claim-to-artifact ledger for the symbolic KV circuit formation paper.
---

# Artifact Map

This page maps paper claims to local artifact directories. The repository does not need to upload every run. A reproduced run should create these paths locally.

## Reading The Ledger

Each claim has three fields:

| field | meaning |
| --- | --- |
| claim | what the paper says |
| artifact | where the supporting output should live |
| proof type | behavior, causal intervention, weight geometry, optimizer attribution, or closure |

If an artifact is missing, do not silently substitute another result. Regenerate the artifact or mark the claim as not reproduced.

## Main Claims

| claim | artifact | proof type |
| --- | --- | --- |
| The model learns symbolic latest-write lookup. | `artifacts/runs/symbolic_kv_reference_formation/` | behavior |
| Heldout-pair behavior is meaningful. | `artifacts/runs/symbolic_kv_reference_formation/analysis/dataset_geometry/` | dataset audit |
| Component-level analysis finds dense early and late roles. | `artifacts/runs/symbolic_kv_reference_formation/analysis/output_component_causal_validation/` | causal and DLA validation |
| Residual-state patching rescues early-component damage. | `artifacts/runs/symbolic_kv_reference_formation/analysis/residual_state_rescue/` | causal patching |
| Feature families reveal superposition rather than clean atoms. | `artifacts/runs/symbolic_kv_reference_formation/analysis/traced_candidates/layer2_family7_family4/` | feature-family analysis |
| QK route separation grows in the reference seed. | `artifacts/runs/symbolic_kv_reference_formation/analysis/bilinear_qk_match_separation/` | route geometry |
| `L2H1 W_QK` forms a low-rank route matcher. | `artifacts/runs/symbolic_kv_reference_formation/analysis/weight_svd_trace/phase1_000250_5500_top16/` | weight geometry |
| Contextual residual directions explain QK better than static embeddings. | `artifacts/runs/symbolic_kv_reference_formation/analysis/contextual_svd_alignment/` | contextual alignment |
| Actual AdamW updates explain QK route growth. | `artifacts/runs/symbolic_kv_reference_formation/analysis/bilinear_qk_rank_adam_state_attribution/from_init_l2h1_rank8_support_value_0000_6000_stepwise/` | optimizer attribution |
| Cross-seed QK role repeats with different head addresses. | `artifacts/runs/symbolic_kv_cross_seed_adam/` | cross-seed optimizer attribution |
| OV/write is better represented as a contextual residual subspace than a static `W_OV` map. | `artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_input_functional_subspace/` | functional subspace |
| Write coupling turns on around the formation window. | `artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_functional_subspace_trajectory/` | trajectory |
| Reference-seed write coupling is AdamW-explained. | `artifacts/runs/symbolic_kv_reference_formation/analysis/mlp_functional_write_adam_state_attribution/` | optimizer attribution |
| Cross-seed write roles repeat under different addresses. | `artifacts/runs/symbolic_kv_cross_seed_adam/seed_*/analysis/mlp_input_functional_subspace/` | cross-seed functional subspace |
| Cross-seed write growth is AdamW-preconditioned-update driven. | `artifacts/runs/symbolic_kv_cross_seed_adam/seed_*/analysis/mlp_functional_write_adam_state_attribution/` | cross-seed optimizer attribution |
| Route/write scalar closure is partial. | `artifacts/runs/symbolic_kv_reference_formation/analysis/route_to_scalar_closure/` | scalar closure |
| Output-space closure is stronger than route-score closure. | `artifacts/runs/symbolic_kv_reference_formation/analysis/output_route_closure/` | output closure |
| Moving answer margin needs branch-aware accounting. | `artifacts/runs/symbolic_kv_reference_formation/analysis/answer_margin_branch_decomposition/` | scalar branch decomposition |
| Line-integral diagnostics explain part of the write-side closure gap. | `artifacts/runs/symbolic_kv_reference_formation/analysis/component_output_rescue_line_integral/` | nonlinear path diagnostic |

## Figure Sources

| figure | source meaning |
| --- | --- |
| `task_rule_latest_write_lookup.svg` | task rule diagram |
| `dataset_geometry_split_axes.svg` | dataset split axes |
| `dataset_geometry_answer_pair_matrix.svg` | answer-pair split geometry |
| `weight_qk_birth_timeline.svg` | QK singular growth and rank compression |
| `contextual_semantic_alignment.svg` | contextual residual alignment |
| `qk_optimizer_phase_structure.svg` | phase structure of QK optimizer attribution |
| `reference_write_optimizer_split.svg` | reference-seed fixed write optimizer split |
| `write_side_mechanism.svg` | contextual write/readout proof object |
| `write_functional_birth.svg` | reference-seed write coupling trajectory |
| `cross_seed_qk_write_role_map.svg` | QK and write role-address dissociation |
| `closure_boundary.svg` | route closure, output closure, and line-integral boundary |
| `proof_status_ladder_updated.svg` | proven, supported, and open claims |

## Supported Versus Open

| statement | status |
| --- | --- |
| QK route formation has a strong computation-level and optimizer-level explanation. | supported |
| The QK role repeats across five additional seeds with changing head addresses. | supported |
| The write side has a cross-seed functional-subspace signal. | supported |
| AdamW-preconditioned updates carry cross-seed write growth. | supported |
| The raw-gradient, SGD-equivalent update is large enough to explain the measured QK/write growth. | not supported |
| OV/write is a clean low-rank `W_OV` story like QK. | not supported |
| A small route family fully closes all answer-margin improvement. | open |
| Plain SGD could not eventually learn the circuit. | not tested |
| The same story holds under width/depth/task scaling. | open |

## Reproduction Checks

Use these checks after running a reproduction batch:

```bash
test -f artifacts/runs/symbolic_kv_reference_formation/run_config.json
test -f artifacts/runs/symbolic_kv_reference_formation/analysis/probe_set.jsonl
test -f artifacts/runs/symbolic_kv_reference_formation/analysis/route_to_scalar_closure/qk_ov_output_routes_1500_2500_formation/route_to_scalar_closure_report.json
test -f artifacts/runs/symbolic_kv_reference_formation/analysis/output_route_closure/qk_ov_output_routes_1500_2500_formation/output_route_closure_report.json
find artifacts/runs/symbolic_kv_cross_seed_adam -path '*mlp_functional_write_adam_state_attribution*report.json' -print | sort
```

Expected cross-seed write AdamW winner report count:

```text
5
```
