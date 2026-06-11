# Results Ledger

This ledger maps each main paper claim to compact artifacts. It is not a raw research log.

## Claim 1: QK Route Formation

Paper section: QK Route Birth

Reported evidence:

- measured `C_QK` route growth,
- measured `W_QK` singular-value growth,
- actual route growth `+4.11462`,
- AdamW first-order reconstruction `+5.21734`,
- raw-gradient/SGD-equivalent fraction about `0.76%`.

Compact artifacts:

- `tables/qk_route_birth.csv`
- `tables/qk_causal_patching.csv`
- `figures/figure2_qk_birth_timeline.svg`

Source artifact families:

- `analysis/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/`
- `analysis/bilinear_qk_match_separation/`
- `analysis/svd_task_alignment/`
- causal route-transfer artifacts for the `L2H1` QK query route.

Boundary:

The QK route is important and causal, but the rank-4 route recovers only part of full residual transfer. The circuit is distributed.

## Claim 2: Optimizer-Update Attribution

Paper section: Optimizer Accounting

Reported evidence:

- actual `Delta C_QK`: `+4.11`,
- AdamW reconstruction: `+5.22`,
- raw-gradient/SGD-equivalent: `+0.031`,
- Adam current: `+2.37`,
- Adam momentum: `+3.05`,
- weight decay: `-0.20`.

Compact artifacts:

- `tables/optimizer_update_attribution.csv`
- `tables/optimizer_ablation.csv`
- `figures/figure3_qk_optimizer_phase_structure.svg`

Source artifact families:

- `analysis/optimizer_update_trace/from_init_seed7_0000_6000_stepwise/`
- `configs/optimizer_ablation/`

Boundary:

The attribution is first-order in the scalar. The measured parameter update is actual AdamW, but the scalar change is approximated by a local gradient dot product. The SGD comparison is bounded to the tested seed-7 recipe and budget.

## Claim 3: Role/Address Dissociation

Paper section: Role/Address Dissociation Across Seeds

Reported evidence:

- same predefined `C_QK` scalar on the same fixed probe set selects different QK winner heads across seeds,
- winner heads grow positively,
- bottom controls move negatively or weakly,
- write/readout path addresses also vary.

Compact artifacts:

- `tables/cross_seed_role_address.csv`
- `figures/figure4_cross_seed_role_address.svg`

Source artifact families:

- cross-seed Adam analysis runs under `symbolic_kv_cross_seed_adam`.

Boundary:

This supports role recurrence in one task family over six total seeds. It is not a general theorem that all transformer circuits have unstable addresses.

## Claim 4: Write/Readout Value-Code Geometry

Paper section: The Write Side Is Contextual, Not Static OV

Reported evidence:

- rank-16 value-identity removal damages margin and accuracy,
- rank-127 keep nearly preserves IID behavior,
- source-plus-context transfer improves stable write/readout scalars,
- context-only rescue is already strong.

Compact artifacts:

- `tables/write_value_code.csv`
- `figures/figure5_write_side_mechanism.svg`

Source artifact families:

- `analysis/value_code_subspace/`
- `analysis/value_code_causal_intervention/`
- `analysis/value_code_transfer_map/`
- `analysis/value_code_transfer_rescue/`

Boundary:

The rank-127 result is not a compact-code result. It is evidence against a small low-rank value-vector account and for a broadly distributed prediction-position value-readable state.

## Claim 5: Moving-Margin Closure Boundary

Paper section: Methodological Trap / Limitations

Reported evidence:

- route/write scalar closure improves fixed logit movement but remains partial,
- fixed output-space closure is stronger,
- moving answer margin remains harder because best-wrong branches can switch.

Compact artifacts:

- `tables/closure_diagnostics.csv`
- `figures/figure6_closure_boundary.svg`

Source artifact families:

- `analysis/route_to_scalar_closure/`
- `analysis/output_route_closure/`
- `analysis/answer_margin_branch_decomposition/`

Boundary:

This is a diagnostic limitation and a methodological warning, not full answer-margin closure.
