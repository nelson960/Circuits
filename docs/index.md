---
layout: default
title: "From Loss To Lookup: Tracing Circuit Formation In A Small Transformer"
description: A narrative paper on how AdamW training forms dense retrieval machinery in a small symbolic key-value transformer.
---

# From Loss To Lookup: Tracing Circuit Formation In A Small Transformer

Nelson Alex

Living draft: 2026-05-08

## How To Read This Draft

This is a living paper. The main text gives the mechanistic story; the [reproducibility page](reproducibility.html) and [CLI guide](analysis_cli_guide.html) give the command-level audit trail. I am especially looking for feedback on citation boundaries, optimizer-attribution claims, write-side closure, and whether the role/address framing is stated carefully enough.

## Abstract

I study how a transformer circuit forms during training, rather than only analyzing the trained circuit after it exists. The setting is a controlled symbolic latest-write key-value lookup task. A read token must return the most recent value previously written for the queried key. Although the task is synthetic, it requires three separable internal operations: identifying the queried key, retrieving the latest matching support value rather than a distractor or stale value, and writing value identity into the prediction-position residual stream.

The stable object I find is not a named head or neuron. It is a role: a task-level computational function measured by a scalar. In the reference seed, the QK side of the role forms as a low-rank support-value matcher. First-order attribution using the exact AdamW update tracks the route growth, while the instantaneous raw-gradient / SGD-equivalent direction explains little of the measured movement. Across additional seeds, the support-value retrieval role repeats, but the winning head address changes. The write/readout side is different from QK: it is better described as a contextual, high-rank prediction-position value-code operation than as a static `W_OV` theorem.

The contribution is not a new primitive method, but a controlled formation audit: a single retrieval role is followed from behavioral acquisition into route geometry, causal subspaces, optimizer-update attribution, cross-seed address movement, and write/readout structure. The account is strongest for QK, supported but not closed-form for write/readout, and deliberately bounded on optimizer generality, moving-margin closure, and scaling.

## Introduction

A trained circuit is the adult form. This paper studies the developmental process.

Mechanistic interpretability often begins after training: a model has a behavior, and I inspect its weights and activations to find the circuit implementing that behavior. But the training trajectory can have a different natural unit of explanation from the finished model. During formation, many candidate routes and write paths can share the same substrate. Their component names can be unstable, even when the task-level function they implement is stable.

I use **role** to mean a task-level computational function measured by a scalar, independent of which component implements it. The retrieval role is measured by:

```text
C_QK = E[ score(prediction, true support value)
         - mean score(prediction, value distractors) ].
```

The write role is measured by:

```text
C_write = E[ g_ref(x) . delta_write(x) ],
```

where `delta_write` is the residual change caused by a source component and `g_ref` is a mature answer-relevant readout direction.

The role/address split is the first central phenomenon. Across additional seeds, the support-value retrieval role repeats, but the winning head moves:

```text
seed 0011: L2H0
seed 0013: L2H2
seed 0017: L2H3
seed 0023: L2H1
seed 0029: L1H2
```

The role is stable. The component address is not.

The second central phenomenon is optimizer specificity. In the traced reference run, the actual AdamW update explains route growth much better than the instantaneous raw-gradient / SGD-equivalent direction. This does not mean gradients are irrelevant. AdamW is built from gradients. The point is narrower: the local raw-gradient direction alone is not the update object that explains the measured role growth in these runs.

The third central phenomenon is that the write side is not QK again. QK becomes a compact pointer. The write/readout side becomes a broad, contextual value-code state at the prediction position. Removing value identity from that state damages behavior; keeping a broad value-code subspace nearly preserves IID behavior; source-plus-prediction-context transfer rescues stable write/readout scalars much better than source-only transfer.

This paper therefore follows one small transformer's lookup circuit from the outside inward:

```text
behavior
-> activations
-> residual states
-> route geometry
-> weight movement
-> optimizer state
-> cross-seed replication
```

The result is a developmental story, but the claim is bounded. Prior work already studies circuit emergence, progress measures, QK/OV decomposition, causal intervention, and circuit stability across training. This paper contributes a smaller but more instrumented case study: tracing one controlled retrieval role from behavior into route geometry, causal interventions, and optimizer-update growth.

<figure class="paper-figure">
  <img src="assets/figures/updated_loss_to_lookup_chain.svg" alt="Loss to lookup chain">
  <figcaption><strong>Figure 1. The measured chain.</strong> The audit follows one role from loss pressure, to optimizer state, to weight geometry, to route separation, to output behavior.</figcaption>
</figure>

## Related Work

This work sits between mechanistic interpretability and training dynamics.

The transformer architecture comes from [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762). The QK/OV language and the decomposition of attention heads into route and write maps follows the Transformer Circuits framework of [Elhage et al. 2021](https://transformer-circuits.pub/2021/framework/index.html). I use that decomposition, but I do not claim it as novel.

Circuit formation has already been studied. [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) connect induction heads to a training-time phase change. [Singh et al. 2024](https://openreview.net/forum?id=O8rrXl71D5) study what must go right for induction heads to form in a controlled setting, including interacting subcircuits and activation interventions through training. My setting is different: latest-write symbolic lookup rather than induction-copy, and the emphasis is on role-scalar growth, optimizer-update attribution, and a write/readout side that does not reduce to a clean static OV map.

Progress-measure work is the closest precedent for using interpretable scalars to track formation. [Power et al. 2022](https://arxiv.org/abs/2201.02177) introduced grokking as delayed generalization on small algorithmic datasets. [Nanda et al. 2023](https://openreview.net/forum?id=9XFSbDPmdW) reverse-engineer modular addition and use mechanistic progress measures to divide training into phases. I follow the progress-measure tradition, but my target is a retrieval/write role rather than a clean Fourier modular-addition algorithm, and I link role growth to the actual AdamW optimizer trajectory.

Superposition motivates the use of roles and subspaces rather than neurons. [Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html) show how features can share representational capacity. [Bricken et al. 2023](https://transformer-circuits.pub/2023/monosemantic-features/) push analysis toward feature decompositions rather than individual neurons. This paper reaches a similar practical conclusion for a different reason: in this model, component and feature-family analyses found real structure, but the stable explanatory object was a role scalar.

The role/address claim also has important precedent. [Tigges et al. 2024](https://openreview.net/forum?id=3Ds5vNudIE) track circuits across training and scale in decoder-only LLMs, finding that algorithms and functional components can remain consistent even when implementing attention-head identities change. This paper should not be read as the first claim that algorithms can outlive head names. The narrower contribution is a from-initialization formation audit of such a role in a controlled benchmark, tied to route geometry, causal tests, and optimizer-update attribution.

Causal intervention methods also have substantial prior art. Causal tracing and model editing work such as [Meng et al. 2022](https://arxiv.org/abs/2202.05262) localizes facts in trained models. Path patching [Goldowsky-Dill et al. 2023](https://arxiv.org/abs/2304.05969), automated circuit discovery [Conmy et al. 2023](https://openreview.net/forum?id=89ia77nZ8u), and EAP/EAP-IG [Hanna et al. 2024](https://huggingface.co/papers/2403.17806) give methods and evaluation language for circuit discovery and faithfulness. I use related causal logic, but the target is not only a fixed-checkpoint circuit. The target is formation over training.

Mechanistic data attribution also studies where interpretable mechanisms come from. [Chen et al. 2026](https://huggingface.co/papers/2601.21996) trace interpretable LLM units to influential training samples with influence functions and data interventions. My work is complementary: rather than attributing a trained unit to corpus examples, I trace the growth of a role scalar through actual checkpoint and optimizer updates in a fully controlled task.

The optimizer accounting is specific to AdamW. Adam is from [Kingma and Ba 2014](https://arxiv.org/abs/1412.6980), and decoupled weight decay from [Loshchilov and Hutter 2017](https://arxiv.org/abs/1711.05101). The optimizer result here should be read carefully: it is not a theorem that SGD cannot learn the task. It is evidence that, under the tested seed-7 recipe and finite budget, AdamW-family runs formed the role while tested SGD and SGD+momentum runs did not.

## Method / Setup

### Task

The task rule is latest-write lookup. The sequence contains writes and reads:

```text
W K03 V14   W K01 V09   R K03   W K03 V02   R K03
```

The last read should return `V02`, not `V14`, because `V02` is the latest previous write for key `K03`.

<figure class="paper-figure">
  <img src="assets/figures/task_rule_latest_write_lookup.svg" alt="Latest-write lookup task rule">
  <figcaption><strong>Figure 2. Latest-write lookup.</strong> The model must return the most recent value written for the queried key, not just any value associated with the key.</figcaption>
</figure>

The task is small on purpose. It is not meant to be language modeling. It is meant to be a controlled growth medium where I can ask:

```text
when lookup behavior appears, what changed inside the model?
```

The benchmark includes shortcut checks. Exact sequence overlap across splits is `0`, latent-program overlap is `0`, and heldout leakage outside the heldout split is `0`. Trivial heuristics are weak: `first_value_for_key` and `last_value_before_query` score `0`, and the strongest tested `most_frequent_value_before_query` heuristic is only about `0.146`.

<figure class="paper-figure">
  <img src="assets/figures/dataset_geometry_split_axes.svg" alt="Dataset split axes">
  <figcaption><strong>Figure 3. Split geometry.</strong> The benchmark separates ordinary validation from heldout answer-pair and structural tests.</figcaption>
</figure>

<figure class="paper-figure">
  <img src="assets/figures/dataset_geometry_answer_pair_matrix.svg" alt="Answer pair matrix">
  <figcaption><strong>Figure 4. Answer-pair matrix.</strong> Heldout-pair evaluation checks whether the model can answer key-value combinations excluded from training.</figcaption>
</figure>

### Model And Runs

The shared model recipe is:

| field | value |
| --- | --- |
| reference seed | `7` |
| layers / heads | `3 / 4` |
| `d_model` / `d_ff` | `128 / 512` |
| parameters | `626,048` |
| batch size / steps | `128 / 16,000` |
| optimizer | AdamW |
| learning rate | `0.0004` |
| betas / weight decay | `0.9, 0.95 / 0.01` |
| gradient clip / warmup | `1.0 / 200 steps` |

There are two closely related seed-7 runs. `symbolic_kv_heldout_generalization` is the sparse-checkpoint selection run used to choose a strong heldout-generalizing model. `symbolic_kv_reference_formation` uses the same model and optimizer recipe with dense checkpoints; most exact optimizer/SVD formation analysis uses its `0 -> 6000` horizon. The dense run is not treated as an independent cross-seed replication.

The selected heldout-generalization run reaches heldout-pair answer accuracy around `0.8730`. Structural OOD remains much weaker, around `0.5082`. That is enough behavior to study formation, but not enough to claim every generalization problem is solved.

### Measured Objects

The QK route scalar is:

```text
C_QK(theta)
  = E_x[ score_theta(prediction, support_value)
       - mean_d score_theta(prediction, value_distractor_d) ].
```

This asks whether the prediction position scores the true support value above distractor values.

The write scalar is:

```text
C_write(theta)
  = E_x[ g_ref(x) . delta_write_theta(x) ].
```

This asks whether the residual perturbation caused by a source component points in a mature answer-relevant direction.

For first-order attribution, I use:

```text
Delta C
  ~= grad_theta C(theta_t) . Delta theta_actual.
```

For AdamW decomposition, `Delta theta_actual` is split into the raw SGD-equivalent term, clipped SGD term, Adam current-gradient component, Adam historical momentum component, weight decay, and reconstructed AdamW update.

Each major claim is linked to an artifact family: QK route birth, optimizer attribution, cross-seed scans, write-side functional subspaces, value-code interventions, transfer rescue, and closure diagnostics. The reproducibility page and CLI guide give the command-level audit trail.

## Results

The model does not form a finished circuit all at once. The formation story has overlapping growth stages: dense candidate competition, early scaffold, QK pointer crystallization, optimizer pressure, write/readout coupling, broad value-code readout, later heldout consolidation, and role migration across seeds.

<figure class="paper-figure">
  <img src="assets/figures/growth_phase_timeline.svg" alt="Circuit growth phase timeline">
  <figcaption><strong>Figure 5. Circuit growth timeline.</strong> The stages overlap in training time. I use them as a developmental spine, not a strict chronological partition.</figcaption>
</figure>

### 1. The Circuit Did Not Start As A Circuit

The first lesson was negative but important: the circuit was real, but the unit of explanation was not a neuron, feature family, or stable head name.

Component-level maps found load-bearing pieces, but they did not identify a clean atom of computation. Late components often had clean direct readout toward the answer. Early components were different: they were causally important, but their direct logit attribution could be weak or even point the wrong way. That is already a warning sign: an early component can matter by shaping the workspace that later components use, even when it does not itself write the answer logit.

The feature-family pass made the problem sharper. I hoped feature families would give cleaner units than neurons. Instead, they exposed superposition. Candidate stories shared neurons, opposed each other through other neurons, and contained sign-conflicted units. The same physical substrate could support several partial stories at once.

One failure was especially useful. A transparent feature-family birth model used activation support, amplification, feature-score drive, and aggregate gradient alignment. It predicted that one family should form first. The model instead formed a different, more generalizing family first. The lesson was not that the feature machinery was useless. The lesson was that a gradient-flavored feature score was not enough to explain the circuit birth.

That failure forced the pivot. I stopped asking which neuron, head, or feature family was "the circuit" and started asking which task-level role was being written. A role can be tracked even when the address moves, and it can be measured even when the implementation is dense.

The practical conclusion was:

```text
the circuit was real,
but the unit of explanation was not a neuron.
```

### 2. The Pointer Crystallizes

QK is the cleanest part of the account. In the reference seed, `L2H1 W_QK` becomes a low-rank support-value matcher:

```text
W_QK = W_Q W_K^T.
```

The route appears as measured `C_QK` growth, singular-value concentration, and support-value-over-distractor separation.

<figure class="paper-figure">
  <img src="assets/figures/weight_qk_birth_timeline.svg" alt="QK weight birth timeline">
  <figcaption><strong>Figure 6. QK birth.</strong> The reference route becomes visible as low-rank `W_QK` growth and support-value-over-distractor separation.</figcaption>
</figure>

The relevant first-order measurement is:

```text
Delta C_QK
  ~= grad_theta C_QK(theta_t) . Delta theta_actual.
```

In the from-initialization trace:

```text
actual route growth:              +4.11462
AdamW reconstruction:             +5.21734
raw SGD-equivalent contribution:  +0.03136
raw SGD-equivalent / actual:       0.76%
```

So the immediate raw-gradient direction is far too small to explain the route birth in the traced run. The actual AdamW update is the right explanatory object for this scalar.

The update also does not land symmetrically inside QK. In a traced `5500 -> 5550` diagnostic window, the leading route sharpens mostly through the query side:

```text
L2H1 query-side actual growth: +0.155511
L2H1 key-side actual growth:   -0.076688
```

For `L1H2` and `L0H0`, query-side growth also dominates, although their key-side terms do not conflict as strongly. This is a mechanistic detail: AdamW is not merely enlarging a QK matrix. It is mostly shaping the prediction-side geometry that asks the right lookup question.

### 3. Adaptive Preconditioning Shapes The Pointer

The optimizer result should not be simplified to "momentum did it." The traced AdamW trajectory is momentum-heavy during the clean birth window, but the matched optimizer ablation shows that AdamW with `beta1 = 0` still learns the task and forms an even stronger measured route under this recipe. The more precise claim is about adaptive, preconditioned update geometry: AdamW changes which directions are reachable at this budget.

For the main from-initialization trace, the phase structure is:

```text
0 -> 750:
  weak early setup

750 -> 2500:
  clean route birth; actual +1.665
  raw SGD-equivalent -0.003
  Adam momentum +1.605

2500 -> 3500:
  current gradient and momentum both push

3500 -> 6000:
  optimizer still pushes, but realized route growth saturates
```

<figure class="paper-figure">
  <img src="assets/figures/qk_optimizer_phase_structure.svg" alt="QK optimizer phase structure">
  <figcaption><strong>Figure 7. QK formation has windows.</strong> The cleanest birth window is `750 -> 2500`: the route grows while the raw SGD-equivalent term is slightly negative and Adam momentum carries the useful direction.</figcaption>
</figure>

The matched seed-7 optimizer ablation is a bounded control. Under the tested seed-7 recipe and finite `6000`-step budget:

| variant | validation answer accuracy | QK separation |
| --- | ---: | ---: |
| AdamW baseline | `0.976` | `8.03` |
| AdamW `beta1 = 0` | `0.984` | `9.26` |
| best SGD+momentum sweep | `0.0085` | `~0.118` |

The best SGD+momentum run was not random noise. It learned shallow structure: token accuracy around `0.340` and read-key accuracy around `0.349`, but answer accuracy only `0.0085`. It learned where the task has structure, but not the value-retrieval role.

<figure class="paper-figure">
  <img src="assets/figures/optimizer_ablation_summary.svg" alt="Optimizer ablation summary">
  <figcaption><strong>Figure 8. Bounded optimizer ablation.</strong> AdamW-family runs solve and form the measured route under the tested recipe. Same-budget SGD variants do not. This is not a universal SGD impossibility theorem.</figcaption>
</figure>

### 4. The Body Plan Repeats, The Address Moves

For cross-seed scans, heads are ranked by the predefined support-value route scalar on the fixed probe set. Winners are top positive movers. Controls are bottom-ranked or weak/negative movers under the same scalar.

Across five additional seeds, the winning head changed:

| seed | QK winner | scan score | support-win delta |
| ---: | --- | ---: | ---: |
| `0011` | `L2H0` | `2.815` | `0.157` |
| `0013` | `L2H2` | `2.727` | `0.523` |
| `0017` | `L2H3` | `1.463` | `0.183` |
| `0023` | `L2H1` | `6.361` | `0.843` |
| `0029` | `L1H2` | `2.428` | `0.248` |

Winner-vs-control attribution over `750 -> 2500` shows that this is not just "all heads grow":

| seed | winner | winner actual | bottom | bottom actual |
| ---: | --- | ---: | --- | ---: |
| `0011` | `L2H0` | `1.448` | `L0H0` | `-0.190` |
| `0013` | `L2H2` | `1.451` | `L1H1` | `-0.230` |
| `0017` | `L2H3` | `3.178` | `L0H2` | `-0.254` |
| `0023` | `L2H1` | `1.500` | `L1H2` | `-0.114` |
| `0029` | `L1H2` | `1.439` | `L1H0` | `-2.577` |

Winner heads grow positively in all five seeds. Bottom controls move negatively in all five seeds. Raw SGD-equivalent contribution is tiny across winners, with mean raw-gradient fraction around `0.74%` of actual route growth.

<figure class="paper-figure">
  <img src="assets/figures/cross_seed_qk_write_role_map.svg" alt="Cross-seed QK and write role map">
  <figcaption><strong>Figure 9. Role/address dissociation.</strong> The support-value retrieval role repeats across seeds, but the winning head address changes. The write/readout role also repeats with moving component paths.</figcaption>
</figure>

This result should be read in light of Tigges et al. It is not the first evidence that algorithms can persist while heads change. The new part here is the controlled from-initialization formation audit with optimizer attribution and fixed role scalars.

### 5. The Write Side Differentiates Into A Scaffold

The write side is not QK again. For QK, the clean object is a low-rank route matrix:

```text
W_QK = W_Q W_K^T.
```

For write/readout, the better object is a contextual residual perturbation at the prediction position. OV writes into the residual stream, but that write is later processed by residual addition, layer norm, later attention, MLPs, final normalization, and unembedding. So the right question is not:

```text
does W_OV point directly at the answer embedding?
```

The right question is:

```text
does the residual vector written by the component land in directions
that the later network reads as answer evidence?
```

For an MLP block, define:

```text
delta_in
  = z_clean[input_stage] - z_source_ablated[input_stage]

mlp_output_delta
  = MLP(z_clean[input_stage]) - MLP(z_source_ablated[input_stage])

post_mlp_total_delta
  = delta_in + mlp_output_delta
```

For a scalar `s` with residual gradient `g_s = grad_z s`, the scalar-relevant write effect is:

```text
C_total = E[g_s . post_mlp_total_delta]
        = E[g_s . delta_in] + E[g_s . mlp_output_delta]
        = C_skip + C_mlp.
```

The position split is decisive:

| position | aggregate scalar-relevant effect |
| --- | ---: |
| `prediction` | `4165.317` |
| `support_value` | `-17.307` |

The useful write-side signal is overwhelmingly at the prediction/read position. This rules out the simple picture where the model writes the answer at the support slot and then carries it forward. The better picture is:

```text
L0H0 changes the current prediction-position state.
That state already contains value-relevant information.
Downstream readout directions increasingly use it.
```

The local split shows that `L0MLP` is the only tested MLP with a large positive local conversion:

| component | total effect | residual/direct part | MLP-transformed part |
| --- | ---: | ---: | ---: |
| `L0MLP` | `1648.453` | `1219.088` | `429.365` |
| `L1MLP` | `1329.142` | `1399.580` | `-70.438` |
| `L2MLP` | `1187.722` | `1319.112` | `-131.390` |

The write side is therefore mostly residual carrying, with a smaller local positive conversion at `L0MLP`.

<figure class="paper-figure">
  <img src="assets/figures/write_functional_birth.svg" alt="Write functional birth">
  <figcaption><strong>Figure 10. Write coupling birth.</strong> The mature-looking write direction is partly present early. What turns on sharply around `1500 -> 1750` is coupling to answer/value readout directions.</figcaption>
</figure>

The mature-looking write directions are present surprisingly early. Against a step-2500 reference basis, `delta_in` overlap is already about `0.661` at step `750`; MLP-output overlap is about `0.721`. But the scalar-relevant effect is tiny until the birth window:

| step | total functional write effect | residual/direct part | L0MLP transformed part |
| ---: | ---: | ---: | ---: |
| `750` | `2.864` | `-1.257` | `4.121` |
| `1500` | `2.647` | `2.472` | `0.174` |
| `1750` | `50.185` | `32.584` | `17.601` |
| `2500` | `91.624` | `70.177` | `21.446` |
| `3500` | `92.627` | `79.046` | `13.582` |

So the write side does not look like:

```text
random direction -> new mature direction
```

It looks like:

```text
mature-ish direction exists early,
but answer-readout coupling turns on around 1500 -> 1750.
```

The write-side optimizer attribution confirms this. For the reference-seed fixed-readout write scalar over `1500 -> 2500`:

| quantity | value |
| --- | ---: |
| actual scalar growth | `1.015` |
| first-order actual-update prediction | `1.034` |
| raw SGD fraction | `0.124%` |
| Adam current-gradient fraction | `11.50%` |
| Adam momentum fraction | `91.35%` |
| weight decay fraction | `-2.85%` |
| actual-update sign match | `99.7%` |

Splitting the write effect:

| part | actual growth | share of total | raw SGD fraction | Adam momentum fraction |
| --- | ---: | ---: | ---: | ---: |
| `input_delta` | `0.789` | `~78%` | `0.157%` | `91.23%` |
| `mlp_output_delta` | `0.226` | `~22%` | `-0.010%` | `92.37%` |

In the reference seed, AdamW momentum builds both pieces of the fixed functional-write scalar. Across selected cross-seed write paths, the current-vs-momentum split varies more, but raw SGD remains tiny and AdamW-preconditioned updates carry the useful write growth.

<figure class="paper-figure">
  <img src="assets/figures/write_side_mechanism.svg" alt="Write side mechanism">
  <figcaption><strong>Figure 11. Write/readout mechanism.</strong> The measured write is a contextual residual coupling, not a static `W_OV` answer-vector claim.</figcaption>
</figure>

### 6. The Readout Becomes A Broad Value Code

The prediction-position residual state becomes value-readable during formation.

At `layer_2_post_mlp / prediction`, removing the rank-16 `embedding_value_identity` subspace damages validation behavior after the value code turns on:

| step | baseline margin | intervened margin | margin drop | baseline acc | intervened acc | acc drop |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1750` | `0.666` | `-1.157` | `1.822` | `0.6078` | `0.2092` | `0.3987` |
| `2500` | `3.031` | `-0.634` | `3.666` | `0.6471` | `0.3595` | `0.2876` |
| `3500` | `5.264` | `0.297` | `4.967` | `0.7647` | `0.4771` | `0.2876` |

A rank-matched key-identity control is weaker. At step `3500`, rank-7 value removal drops margin by `2.294`, while rank-7 key removal drops margin by only `0.593` and does not drop validation accuracy.

Low-rank value identity is not sufficient. At step `3500`:

```text
baseline:      margin  5.264, accuracy 0.765
keep rank 16: margin -4.248, accuracy 0.451
keep rank127: margin  5.707, accuracy 0.758
```

The rank-127 result should not be read as a compact-code result. It is evidence against a low-rank value-vector story and in favor of a broadly distributed value-readable state.

The write/readout side is therefore broad value-token identity geometry, not one compact vector.

The broad-code claim has split boundaries. Rank-127 keep is strong on `validation_iid` and `counterfactual`, but does not rescue heldout-pair or structural-OOD margins. The current claim is not "this explains every split." The claim is:

```text
for the mature IID/counterfactual answer behavior,
the prediction-position residual contains a broad value-identity code
that the readout causally uses.
```

### 7. The Prediction Slot Supplies The Scaffold

The next question was whether the support value-code is copied into the prediction value-code.

The source-only transfer model is:

```text
z_hat_prediction_value = A z_support_value + b.
```

This helps, but only partly. At step `3500`, source-only transfer rescues `0.949` of negative-answer-loss damage and `0.750` of value-accuracy damage, but only `0.320` of the fixed removed-branch margin and `-0.080` of the moving margin.

The contextual model is:

```text
z_hat_prediction_value
  = A z_support_value + B z_prediction_context + b.
```

At step `3500`, source-plus-context rescue reaches:

| scalar | source-only | context-only | source + context |
| --- | ---: | ---: | ---: |
| fixed removed branch | `0.320` | `0.640` | `0.754` |
| negative answer loss | `0.949` | `0.959` | `1.005` |
| value accuracy | `0.750` | not isolated | `0.875` |

<figure class="paper-figure">
  <img src="assets/figures/contextual_semantic_alignment.svg" alt="Contextual semantic alignment">
  <figcaption><strong>Figure 12. Contextual transfer.</strong> Support value-code transfer is real, but prediction-position context already carries much of the recoverable value-code scaffold.</figcaption>
</figure>

This is the main write-side twist. The support value does not write the whole answer code from scratch. The prediction slot already carries much of the readout-ready value-code scaffold, and support retrieval helps shape that scaffold into the answer-specific state.

### 8. The Write Role Also Moves Addresses

The cross-seed write audit ran `28 / 28` functional-subspace reports. The selected winner write paths were:

| seed | source head | downstream MLP |
| ---: | --- | --- |
| `0011` | `L1H3` | `L1MLP` |
| `0013` | `L1H3` | `L1MLP` |
| `0017` | `L1H1` | `L1MLP` |
| `0023` | `L2H1` | `L2MLP` |
| `0029` | `L1H1` | `L1MLP` |

Final-step functional write effect, grouped by role:

| scalar | winner mean | runner mean | bottom mean |
| --- | ---: | ---: | ---: |
| `fixed_source_competitor_margin` | `510.43` | `388.22` | `177.01` |
| `negative_answer_loss` | `415.64` | `195.40` | `9.63` |

Most of the winner effect is residual-skip effect:

| scalar | residual-skip fraction | local MLP-output fraction |
| --- | ---: | ---: |
| `fixed_source_competitor_margin` | `0.902` | `0.098` |
| `negative_answer_loss` | `0.908` | `0.092` |

So the write role also repeats across seeds, and the address also moves. This matters because the role/address split is not only a QK routing phenomenon.

### 9. Moving Margins Can Hide Formation

The usual answer margin is:

```text
logit(correct) - max_wrong logit(wrong).
```

This is not always a stable scalar during training. The correct token is fixed by the task, but the best wrong token can change across checkpoints or along an interpolation path. When that branch changes, the scalar being explained changes too.

In the matched `1500 -> 2500` formation-window audit:

| bucket | observations | competitor switches | switch fraction |
| --- | ---: | ---: | ---: |
| all | `512` | `312` | `0.609` |
| competitor switch | `312` | `312` | `1.000` |
| same competitor | `200` | `0` | `0.000` |

On switch rows, the target-branch correction carries about `71.6%` of moving-margin energy. Output-space closure therefore looks much cleaner on fixed or branch-aware scalars:

| scalar | R squared |
| --- | ---: |
| `correct_value_logit` | `0.868` |
| `fixed_source_competitor_margin` | `0.639` |
| `fixed_target_competitor_margin` | `0.558` |
| `moving_answer_margin` | `0.407` |
| `negative_answer_loss` | `0.183` |

When the competitor switches, fixed-branch plus exact branch correction improves the comparison:

| bucket | direct moving `R^2` | source-fixed + branch `R^2` | target-fixed + branch `R^2` |
| --- | ---: | ---: | ---: |
| all | `0.407` | `0.418` | `0.489` |
| competitor switch | `0.416` | `0.506` | `0.517` |
| same competitor | `0.596` | `0.596` | `0.596` |

<figure class="paper-figure">
  <img src="assets/figures/closure_boundary.svg" alt="Closure boundary">
  <figcaption><strong>Figure 13. Closure boundary.</strong> Fixed-output and fixed-branch scalars are cleaner formation targets than moving answer margin.</figcaption>
</figure>

This is a methodological result, not just a limitation. Moving answer margin is often the first scalar people reach for, but it can be a bad proof target for formation audits when the wrong-token branch changes.

### 10. Current Computation Ledger

The ledger separates three kinds of claims:

```text
causal claim:
  ablating or patching the object changes behavior

dynamic claim:
  actual optimizer updates built the object during training

computational claim:
  the object implements a specific part of the lookup algorithm
```

The current paper-level ledger is:

| object | math target | artifact family | status |
| --- | --- | --- | --- |
| behavior | `m_t = logit(correct) - max wrong` | evaluation / scalar diagnostics | learned lookup supported |
| QK route | `C_QK` | QK geometry, contextual alignment, route attribution | strong |
| QK weight birth | `W_QK = W_Q W_K^T` | `weight_svd_trace`, rank-8 QK reports | strong |
| QK optimizer attribution | `Delta C_QK ~= grad C_QK . Delta theta_actual` | from-init AdamW attribution, cross-seed controls | strong in traced runs |
| write functional subspace | `C_write = E[g_ref . delta_write]` | `mlp_input_functional_subspace`, trajectory | supported |
| write optimizer attribution | `Delta C_write ~= grad C_write . Delta theta_actual` | write AdamW attribution | supported |
| value-code readout | prediction value-identity projection | value-code intervention / keep / transfer rescue | causal and supported |
| static OV theorem | low-rank `W_OV` copy story | controls and functional split | not supported |
| moving-margin closure | full moving answer margin | branch decomposition / closure | partial |

The strongest safe statement is:

```text
QK is close to fully characterized in this controlled setting.
The write/readout side is causal and contextual, but not closed-form.
```

## Limitations

This is a controlled mechanistic case study, not a universal theory of transformer circuit formation.

The main limitations are:

| limitation | current boundary |
| --- | --- |
| synthetic task | useful for auditability, not language-model realism |
| small model | 3 layers, 4 heads, 626k parameters |
| optimizer ablation | one seed, one architecture, one budget, finite LR sweep |
| SGD claim | tested SGD variants did not form the role; no impossibility theorem |
| write side | contextual value-code account, not closed-form scaffold derivation |
| full margin closure | fixed/output scalars close better than moving margin |
| role/address result | six seeds in this setup, not a universal field-level law |
| scaling | not tested on larger open-weight models |

The write side is the main open mechanistic gap. The current account says:

```text
support value-code + prediction context
  -> prediction-position value-code
  -> answer readout
```

but it does not derive, in closed form, how the prediction context becomes a value-readout scaffold. A Fourier-style closed algorithm would require that missing operator.

The optimizer result also remains bounded. The current evidence supports:

```text
under the tested seed-7 recipe and finite budget,
AdamW-family runs learned and formed the role,
while tested SGD variants did not.
```

It does not support:

```text
plain SGD can never learn this circuit.
```

Longer SGD runs, broader schedules, different initialization scales, cross-seed optimizer ablations, and larger-task variants remain future tests.

## Conclusion

This study does not show that all transformer circuits form this way. It shows that, in one controlled symbolic retrieval setting, the stable unit of formation can be a role rather than a component address.

The QK side becomes visible as a low-rank support-value matcher. In the traced reference run, first-order attribution using the exact AdamW update tracks its growth, while the instantaneous raw-gradient / SGD-equivalent direction explains little of the movement. Across seeds, the retrieval role repeats while the winning head changes.

The write side is real but different. It is a contextual, high-rank prediction-position value-code operation rather than a static `W_OV` theorem. The mature prediction residual contains broad value-token identity geometry that the answer readout causally uses. Source-plus-prediction-context transfer nearly restores stable write/readout scalars, but context-only rescue is already strong, leaving the construction of the prediction scaffold as the main open algorithmic question.

The contribution is therefore a formation audit rather than a full closed-form reverse engineering result: one lookup role is followed from behavior, to route geometry, to causal subspaces, to optimizer-update attribution, to cross-seed address movement, and to write/readout structure. The paper is strongest where the role scalar is fixed and differentiable, and most open where the scalar switches branches or the write operator is broad and contextual.

## References

- [Ameisen et al. 2025, Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
- [Bricken et al. 2023, Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/)
- [Chen et al. 2026, Mechanistic Data Attribution: Tracing the Training Origins of Interpretable LLM Units](https://huggingface.co/papers/2601.21996)
- [Conmy et al. 2023, Towards Automated Circuit Discovery for Mechanistic Interpretability](https://openreview.net/forum?id=89ia77nZ8u)
- [Elhage et al. 2021, A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [Elhage et al. 2022, Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)
- [Geva et al. 2023, Dissecting Recall of Factual Associations in Auto-Regressive Language Models](https://aclanthology.org/2023.emnlp-main.751/)
- [Goldowsky-Dill et al. 2023, Localizing Model Behavior with Path Patching](https://arxiv.org/abs/2304.05969)
- [Hanna et al. 2024, Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms](https://huggingface.co/papers/2403.17806)
- [Kingma and Ba 2014, Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Loshchilov and Hutter 2017, Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Meng et al. 2022, Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- [Nanda et al. 2023, Progress Measures for Grokking via Mechanistic Interpretability](https://openreview.net/forum?id=9XFSbDPmdW)
- [Olsson et al. 2022, In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- [Power et al. 2022, Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)
- [Singh et al. 2024, What Needs To Go Right For An Induction Head?](https://openreview.net/forum?id=O8rrXl71D5)
- [Tigges et al. 2024, LLM Circuit Analyses Are Consistent Across Training and Scale](https://openreview.net/forum?id=3Ds5vNudIE)
- [Vaswani et al. 2017, Attention Is All You Need](https://arxiv.org/abs/1706.03762)
