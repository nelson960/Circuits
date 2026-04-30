---
layout: default
title: "From Loss To Lookup: Tracing Circuit Formation In A Small Transformer"
description: A narrative paper on how AdamW training forms dense retrieval machinery in a small symbolic key-value transformer.
---

# From Loss To Lookup: Tracing Circuit Formation In A Small Transformer

Nelson Alex

Living draft: 2026-04-30

## Abstract

I study circuit formation in a 3-layer decoder-only transformer trained on a symbolic key-value lookup task. Because the task has a known algorithmic structure, I can define role-level progress measures for support-value retrieval and write/readout coupling. I find that the trained mechanism is dense and not localized to a stable head or neuron identity. Instead, a support-value retrieval role repeatedly forms across random seeds, while its implementing head changes. In the reference seed, the QK side appears as a low-rank `W_QK` matcher whose route growth is predicted by exact AdamW update accounting. The instantaneous raw-gradient, SGD-equivalent update explains only a small fraction of this growth, while AdamW-preconditioned current and momentum terms carry the movement. The write side does not reduce to a clean static `W_OV` matrix; it appears as a contextual residual perturbation at the prediction position used by downstream readout directions. These results support role-level, optimizer-state-aware circuit formation in a controlled model, while leaving full answer-margin closure, optimizer ablations, and scaling open.

## Contributions

I make six claims that can be checked against the artifact map.

1. A controlled symbolic key-value benchmark for studying circuit formation under autoregressive training.
2. A role-level route scalar for support-value retrieval.
3. Evidence that QK route formation appears as low-rank `W_QK` crystallization.
4. Exact AdamW update attribution showing that the instantaneous raw-gradient, SGD-equivalent update is tiny relative to AdamW-preconditioned route growth.
5. Cross-seed evidence that the retrieval role repeats while the implementing head changes.
6. A write-side analysis showing contextual residual coupling rather than a clean static `W_OV` theorem.

The question was simple:

```text
How does training find a circuit?
```

The usual answer is "the gradient found it." That is not precise enough. A model is not updated by a slogan. It is updated by a particular optimizer, on particular batches, through particular weights, over a particular trajectory.

I ran into this early. A transparent feature-family birth model used activation support, amplification, feature-score drive, and aggregate gradient alignment. It predicted `family4` should form first. The model actually formed the more generalizing `family7` first: `family7` became useful at step `2250`, while `family4` followed at step `2500`. `family7` also had the larger useful delta (`0.408` versus `0.234`) and heldout-gap delta (`0.196` versus `0.022`). That failure was useful. It told me a gradient-flavored feature score was not enough. I needed to track the role being written, not just the most tempting feature family.

I follow one small transformer's lookup circuit from the outside inward: behavior, activations, residual states, route geometry, weight movement, optimizer state, and then cross-seed replication. The result is not a clean neuron story. It is a role story.

The strongest claim is this:

```text
The task repeatedly induces a support-value retrieval role.
The role is stable.
The named head address is not.
In the traced runs, AdamW-preconditioned updates carry the useful growth,
while the instantaneous raw-gradient, SGD-equivalent direction is tiny.
```

That is the core finding. The write side is real too, but it is not QK again. QK becomes a clean low-rank route matcher. The write side appears as a contextual residual subspace at the prediction position that downstream readout directions can use.

<figure class="paper-figure">
  <img src="assets/figures/updated_loss_to_lookup_chain.svg" alt="Loss to lookup chain">
  <figcaption><strong>Figure 1. The measured chain.</strong> I follow one role from loss pressure, to optimizer state, to weight geometry, to route separation, to output behavior.</figcaption>
</figure>

## The Short Version

I trained a 3-layer decoder-only transformer on symbolic key-value lookup. The model sees writes and reads:

```text
W K03 V14   W K01 V09   R K03   W K03 V02   R K03
```

The correct answer is the latest previous value for the queried key. In the example above, the last read of `K03` should return `V02`, not `V14`, because `V02` is the latest write for that key.

The task is small on purpose. It is not meant to be language modeling. It is meant to be a controlled world where I can ask a sharper question:

```text
when lookup behavior appears, what exactly changed inside the model?
```

The answer has two halves.

The QK half asks where the model reads. In the reference seed, `L2H1 W_QK` becomes a low-rank support-value matcher. Its route score grows during formation, its singular structure crystallizes, and exact AdamW attribution explains the growth. The raw-gradient, SGD-equivalent update accounts for only about `0.76%` of the route growth in the traced from-initialization run.

The write half asks what useful state gets written after reading. This does not reduce to a clean `W_OV` matrix theorem. The better object is a residual perturbation at the prediction position:

```text
C_write(theta) = E_x [ g_ref(x) . delta_write_theta(x) ]
```

In words: remove a source, look at the missing residual change, and ask whether that missing change points in the answer-relevant direction used by the mature model. For the reference seed fixed-readout write scalar over `1500 -> 2500`, the raw-gradient, SGD-equivalent update is about `0.13%` of actual growth. Adam momentum carries about `93%` of the same scalar aggregate in that run.

Across five additional seeds, both sides repeat at the role level but move at the address level. QK winners include `L2H0`, `L2H2`, `L2H3`, `L2H1`, and `L1H2`. Write winners include paths such as `L1H3 -> L1MLP`, `L1H1 -> L1MLP`, and `L2H1 -> L2MLP`.

The ghost moves rooms. The computation repeats, but the named component changes.

## Related Work

This work sits between mechanistic interpretability and training dynamics.

The transformer architecture comes from [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762). The QK/OV language and the habit of decomposing attention heads into route and write maps follows the transformer-circuits line of work, especially [Elhage et al. 2021](https://transformer-circuits.pub/2021/framework/index.html). The closest circuit-formation precedent is the induction-head work of [Olsson et al. 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html), which connects a learned attention circuit to a training-time phase change. I ask a narrower but more optimizer-specific question: which update components actually moved the role scalar during formation?

The superposition framing follows [Elhage et al. 2022](https://transformer-circuits.pub/2022/toy_model/index.html). That is why I do not force a neuron-level theorem when the evidence says the useful object is a subspace. The causal-intervention style is related to causal tracing and model-editing work such as [Meng et al. 2022](https://arxiv.org/abs/2202.05262), but the target here is not just where a finished model stores a fact. The target is how training wrote a role into the model.

The optimizer accounting is specific to AdamW. Adam was introduced by [Kingma and Ba 2014](https://arxiv.org/abs/1412.6980), and decoupled weight decay for AdamW by [Loshchilov and Hutter 2017](https://arxiv.org/abs/1711.05101). Small algorithmic tasks have also been used to study delayed generalization and training dynamics, most famously in grokking work by [Power et al. 2022](https://arxiv.org/abs/2201.02177). I use a small symbolic task for a different purpose: to make a role-level circuit formation story auditable end to end.

## The Tiny World

The task rule is deliberately simple: a read asks for the latest previous write for the same key.

<figure class="paper-figure">
  <img src="assets/figures/task_rule_latest_write_lookup.svg" alt="Latest-write lookup task rule">
  <figcaption><strong>Figure 2. Latest-write lookup.</strong> The model must return the most recent value written for the queried key, not just any value associated with the key.</figcaption>
</figure>

The split is part of the experiment. IID validation asks whether the model learned the training distribution. Heldout-pair validation asks whether it learned the relation rather than memorizing value pairs.

<figure class="paper-figure">
  <img src="assets/figures/dataset_geometry_split_axes.svg" alt="Dataset split axes">
  <figcaption><strong>Figure 3. Split geometry.</strong> The benchmark separates ordinary validation from heldout answer-pair and structural tests.</figcaption>
</figure>

<figure class="paper-figure">
  <img src="assets/figures/dataset_geometry_answer_pair_matrix.svg" alt="Answer pair matrix">
  <figcaption><strong>Figure 4. Answer-pair matrix.</strong> Heldout-pair evaluation checks whether the model can answer key-value combinations excluded from training.</figcaption>
</figure>

The reference run uses a small transformer:

| field | value |
| --- | --- |
| reference seed | 7 |
| layers / heads | 3 / 4 |
| `d_model` / `d_ff` | 128 / 512 |
| parameters | 626,048 |
| batch size / steps | 128 / 16,000 |
| optimizer | AdamW |
| learning rate | 0.0004 |
| betas / weight decay | 0.9, 0.95 / 0.01 |
| gradient clip / warmup | 1.0 / 200 steps |

The selected heldout-generalization run reaches heldout-pair answer accuracy around `0.8730`. Structural OOD remains much weaker, around `0.5082`. That is enough behavior to study formation, but not enough to pretend every generalization question is solved.

## How The Search Changed

I did not start with QK. I started with the obvious maps.

First I asked which components mattered. That found real load-bearing pieces, but it did not explain the computation. Late components often had clean direct readout toward the answer. Early components were different: they were causally important, but their direct logit attribution could be weak or even point the wrong way.

That told me early components were not dead. They were shaping the residual workspace that later components read.

Then I looked for feature families and neurons. This also found structure, but not atoms. The feature-family phase exposed superposition instead of escaping it. In one coalition map, hundreds of neurons were shared by candidate stories, hundreds opposed both, and hundreds were sign-conflicted. A neuron intervention was not isolating one clean variable.

That was the first important lesson:

```text
the circuit was real,
but the unit of explanation was not a neuron.
```

So the search moved from components to roles. A role is a computation-level object: "prefer the true support value over distractors" or "write a prediction-position residual change that the answer readout can use." A role can be implemented by different heads in different seeds.

This changed the project from a component hunt into a formation audit.

## The QK Role

QK is the clean half because QK is routing.

For one attention head, the effective route map is:

```text
W_QK = W_Q W_K^T
```

The attention score between a prediction-position query and a candidate source position is:

```text
score(prediction, source) = q_prediction^T W_QK k_source
```

The route scalar asks whether the head prefers the true support-value position over value distractors:

```text
C_QK(theta)
  = E[ score(prediction, support_value)
       - mean score(prediction, value_distractors) ]
```

This is not a universal importance metric. It is a task-specific progress scalar. It asks one concrete question:

```text
is this route learning to look at the right previous value?
```

In the reference seed, `L2H1` becomes the clearest route carrier. During the main formation window, rank-8 support-value separation increases by `+4.19295`. The route score tracks QK singular-value growth with correlation `0.9934`.

<figure class="paper-figure">
  <img src="assets/figures/weight_qk_birth_timeline.svg" alt="QK birth timeline">
  <figcaption><strong>Figure 5. QK crystallization.</strong> The support-value route becomes concentrated in `L2H1 W_QK`: singular mass grows while effective rank compresses.</figcaption>
</figure>

The singular vectors are not best understood as raw token-embedding directions. The useful route aligns with contextual residual state: what earlier layers have made the key and value positions mean inside the network.

<figure class="paper-figure">
  <img src="assets/figures/contextual_semantic_alignment.svg" alt="Contextual semantic alignment">
  <figcaption><strong>Figure 6. Context matters.</strong> The route aligns with contextual residual directions more cleanly than with a naive static embedding story.</figcaption>
</figure>

This gives the first complete computational object:

```text
prediction-position query
  scores
true support-value state
  above
value distractor states
```

And it gives a weight-level object:

```text
W_QK = U Sigma V^T
```

The top singular structure of `W_QK` is not just decorative. It grows when the support-value route grows.

## The Optimizer Accounting

The raw gradient did not explain the route birth in the traced run. AdamW's actual update did.

For every adjacent checkpoint, I measured:

```text
Delta C_QK ~= grad_theta C_QK(theta_t) . Delta theta_actual
```

Then I decomposed the actual update into pieces:

```text
raw SGD-equivalent update
clipped SGD-equivalent update
Adam current-gradient component
Adam historical momentum component
weight decay
reconstructed AdamW update
```

For the reference seed route over `0 -> 6000`:

```text
actual route growth:              +4.11462
reconstructed AdamW prediction:   +5.21734
raw SGD-equivalent / actual:       0.76%
Adam current / actual:            57.7%
Adam momentum / actual:           74.0%
weight decay / actual:            -4.9%
```

The percentages can sum above 100% because components reinforce and oppose each other. The important fact is that the raw SGD-equivalent update is tiny.

The trace is not one uniform story. It has phases:

```text
0 -> 750:
  weak setup; actual +0.070, momentum +0.067

750 -> 2500:
  clean route birth; actual +1.665, raw SGD-equivalent -0.003, momentum +1.605

2500 -> 3500:
  fresh gradients join; actual +1.673, current +1.165, momentum +1.137

3500 -> 6000:
  optimizer still pushes; actual +0.706, current +1.205, momentum +0.236
```

The critical window is `750 -> 2500`. The route grows, but the raw SGD-equivalent term is slightly negative and the current Adam gradient is almost zero. Momentum carries the useful movement.

<figure class="paper-figure">
  <img src="assets/figures/qk_optimizer_phase_structure.svg" alt="QK optimizer phase structure">
  <figcaption><strong>Figure 7. QK formation has phases.</strong> The cleanest birth window is `750 -> 2500`: the route grows while the raw SGD-equivalent term is slightly negative and Adam momentum carries the useful direction.</figcaption>
</figure>

This does not mean gradients are irrelevant. AdamW is built from gradients. It means the object that wrote the route was not the instantaneous raw gradient alone. The object was the optimizer trajectory: accumulated state, adaptive scaling, and preconditioning.

The gradient was not literally lying. It was answering a local-slope question in a dense competition phase. Several candidate routes are being trained at once. Their instantaneous gradient contributions can cancel in the shared parameters, so the raw gradient can look near-zero or even point against the role that will win. Momentum integrates those noisy local samples across steps. The consistent signal survives the cancellation.

This is not a claim that plain SGD could never learn this role. I did not run that ablation. The measured claim is narrower: in the AdamW-trained runs studied here, the realized route growth is carried by AdamW-preconditioned update components, while the instantaneous raw-gradient, SGD-equivalent direction is far too small to account for the observed movement.

## The Ghost Moves Rooms

A circuit story that only works for one head in one seed is weak. So I repeated the route search across five additional seeds.

The support-value retrieval role repeated. The address changed.

```text
seed 0011: QK winner L2H0
seed 0013: QK winner L2H2
seed 0017: QK winner L2H3
seed 0023: QK winner L2H1
seed 0029: QK winner L1H2
```

Winner heads grew positively in all five seeds. Bottom-control heads moved negatively in all five seeds. The raw SGD-equivalent term remained small across winners, with mean raw-gradient fraction around `0.74%`.

This is one of my main interpretability lessons:

```text
component address is unstable;
role pattern is stable.
```

If the question is "does `L2H1` always do it?", the answer is no. If the question is "does a support-value retrieval role form?", the answer is yes.

## Why OV Was Harder

QK asks where to read. OV/write asks what useful state gets created after reading.

Those are not the same kind of object.

QK produces a score:

```text
query dot key -> attention preference
```

OV writes a vector into the residual stream:

```text
head_output = attention @ V
residual_write = head_output W_O
```

That vector is not judged immediately. It is added to the residual stream, normalized, passed through later attention and MLP blocks, normalized again, and finally unembedded into logits. So the naive question is wrong:

```text
does W_OV point directly at the answer embedding?
```

The useful question is:

```text
does the source write a residual change that the mature model can read as answer-useful?
```

That is why the write scalar is defined through a readout direction:

```text
C_write(theta) = E_x [ g_ref(x) . delta_write_theta(x) ]
```

Here `delta_write_theta(x)` is the residual change caused by a source at a specific position, and `g_ref(x)` is the mature answer-scalar gradient direction at the readout boundary.

<figure class="paper-figure">
  <img src="assets/figures/write_side_mechanism.svg" alt="Write-side residual coupling">
  <figcaption><strong>Figure 8. The write proof object.</strong> The measured write is a residual coupling, not a static `W_OV` answer-vector claim.</figcaption>
</figure>

The value content matters. In the reference seed, forcing `L0H0` to read the correct support-value vector gave a positive OV map score. Keeping the same forced support attention but shuffling the value vector made the score strongly negative. The head had to read the right value-bearing content, not merely attend to a plausible place.

The write lives mostly at the prediction position. Position rescue showed a large prediction-slot effect and almost no useful support-slot effect for the same write scalar. So the story is not "write an answer at the support slot, then carry it forward." The better story is:

```text
the source changes the current prediction residual state;
downstream readout geometry knows how to use that change.
```

## The Write Coupling Turns On

The write direction was not born from nothing. Mature-looking directions are partly present early. What changes sharply is their coupling to answer-readout directions.

This is the key difference from QK. At step `750`, the L0H0-caused input residual delta already overlaps the step-2500 input-delta basis by about `0.661`; the MLP-output delta overlaps the step-2500 MLP-output basis by about `0.721`. So the direction is not random early and mature later. The direction is already partly there. What is missing early is functional coupling: the scalar-relevant write effect is only about `2.864` at step `750`, then jumps to about `50.185` by step `1750` and `91.624` by step `2500`.

For the reference seed, the clearest write/readout boundary is `L0H0 -> L0MLP` at the prediction position. The local decomposition is:

```text
F_l(z) = MLP_l(LN_2(z))

delta_in
  = z_clean[input_stage] - z_L0H0_ablated[input_stage]

mlp_output_delta
  = F_l(z_clean) - F_l(z_ablated)

post_mlp_total_delta
  = delta_in + mlp_output_delta
```

Then the scalar-relevant write effect is:

```text
C_total = E[ g_ref . post_mlp_total_delta ]
        = E[ g_ref . delta_in ]
        + E[ g_ref . mlp_output_delta ]
```

In words: split the useful write into the residual skip part and the nonlinear MLP correction.

<figure class="paper-figure">
  <img src="assets/figures/write_functional_birth.svg" alt="Functional write birth">
  <figcaption><strong>Figure 9. Functional write birth.</strong> The write-readout coupling jumps around `1500 -> 1750`, mostly through the residual skip part, with `L0MLP` adding a smaller positive correction.</figcaption>
</figure>

Over `1500 -> 2500`, the reference seed fixed-readout write scalar grows by about `+4.06` across the four endpoint/scalar rows. The split is:

```text
input/residual part:       about 78% of the growth
L0MLP output part:         about 22% of the growth
raw SGD-equivalent on total scalar: about 0.13% of actual growth
Adam momentum on total:    about 93% of actual growth
```

This is not the same mechanism as QK. QK visibly forms a low-rank route map. The write side looks more like an already-available residual direction becoming functionally coupled to the mature readout.

<figure class="paper-figure">
  <img src="assets/figures/reference_write_optimizer_split.svg" alt="Reference write optimizer split">
  <figcaption><strong>Figure 10. Reference write optimizer split.</strong> In the reference seed and this fixed-readout scalar, the write coupling is momentum-heavy. This is not the same measurement as the later cross-seed aggregate.</figcaption>
</figure>

## Cross-Seed Write Validation

The write role also repeats across seeds, and the address also moves.

The selected write/readout paths are not always the same:

```text
seed 0011: L1H3 -> L1MLP
seed 0013: L1H3 -> L1MLP
seed 0017: L1H1 -> L1MLP
seed 0023: L2H1 -> L2MLP
seed 0029: L1H1 -> L1MLP
```

Winner write paths have much larger final functional write effects than bottom controls. For fixed-source competitor margin, winners average about `510`, runners about `388`, and bottoms about `177`. For negative answer loss, winners average about `416`, runners about `195`, and bottoms about `10`.

Most of the winner effect is the residual write itself. The winning paths are about `90%` residual-skip effect and about `10%` local MLP-output correction on the two cross-seed write scalars.

Exact AdamW attribution across the five selected winner write paths shows the same broad optimizer lesson:

```text
raw SGD-equivalent / predicted: about 1.2%
Adam current / predicted:  about 87%
Adam momentum / predicted: about 14%
weight decay / predicted:  about -1.4%
```

These numbers are different from the reference-seed fixed scalar in Figure 10 because they measure a different object: selected winner write paths across five seeds, aggregated over their write scalars. The current-vs-momentum split varies by seed and address. That matters. The cross-seed write result should be stated as "AdamW-preconditioned updates carry the useful write growth", not "momentum always dominates every write-side run."

<figure class="paper-figure">
  <img src="assets/figures/cross_seed_qk_write_role_map.svg" alt="Cross-seed QK and write role map">
  <figcaption><strong>Figure 11. The ghost moves rooms on both sides.</strong> The retrieval role and the write/readout role repeat, while their component addresses vary with seed.</figcaption>
</figure>

## The Computation Ledger

This is the current end-to-end computation story.

The pointer half:

```text
q_prediction^T W_QK k_support_value
  >
q_prediction^T W_QK k_value_distractor
```

In words: at the prediction position, a QK route scores the true support-value position above distractor value positions.

The write/readout half:

```text
delta_write(x)
  = residual_with_source(x) - residual_without_source(x)

C_write
  = E_x [ g_ref(x) . delta_write(x) ]
```

In words: the source creates a prediction-position residual change that points in a direction the mature answer readout uses.

The local boundary:

```text
post_mlp_total_delta
  = residual_skip_delta + local_mlp_output_delta
```

In words: most of the measured fixed-readout write signal is already present in the residual skip, and `L0MLP` adds a smaller positive correction in the reference seed.

The proof ledger is:

| claim | measured object | status |
| --- | --- | --- |
| QK route | `C_QK = E[score(pred, support) - mean score(pred, distractors)]` | strong |
| QK weight birth | `W_QK = W_Q W_K^T = U Sigma V^T` | strong |
| QK optimizer cause | `Delta C_QK ~= grad C_QK . Delta theta_actual` | strong for AdamW-trained runs |
| cross-seed QK role | winner / runner / bottom scans across five seeds | strong |
| write functional subspace | `C_write = E[g_ref . delta_write]` | supported |
| write optimizer cause | `Delta C_write ~= grad C_write . Delta theta_actual` | supported for AdamW-trained runs |
| static `W_OV` theorem | raw `W_OV` low-rank answer-vector story | not supported |
| full answer-margin closure | small route/write family explains all behavior | partial |
| plain-SGD sufficiency | SGD-only training eventually forms same role or fails | not tested |

The [artifact map](artifact_map.html) links each row of this ledger to the run family that supports it.

The ledger is important because it keeps three claims separate:

```text
causal claim:
  ablating or patching this object changes behavior

dynamic claim:
  optimizer updates built this object during training

computational claim:
  this object implements this operation
```

I have all three strongly for the QK side. I have causal, dynamic, and functional-subspace evidence for the write side, but not a clean static `W_OV` theorem.

## What Would Weaken This Claim

The role-level interpretation has concrete failure modes.

It would be weakened if bottom-control heads showed the same route growth as selected winners, if cross-seed winners failed to separate from runners and bottom controls, or if the support-value route scalar did not survive heldout and distractor controls. It would also be weakened if the AdamW reconstruction failed to track actual route movement.

The write-side interpretation would be weakened if shuffled-value controls preserved the write scalar, if support-position rescue matched prediction-position rescue, or if the selected write/readout subspaces failed cross-seed validation. Those are the reasons I treat QK as strong, the write side as supported but less clean, and full answer-margin closure as still open.

## Closure: What I Explain And What I Do Not

A route can be real without fully explaining the answer margin.

During `1500 -> 2500`, a 14-route family containing QK pointer terms, early write terms, and output proxies gives partial route-to-scalar closure. It explains more of clean fixed or differentiable scalars than of negative answer loss, but it does not close the whole behavior.

Output-space closure is stronger in the same window. For correct-value logit, route/write scalar closure is about `R^2 = 0.37`, while output-space closure reaches about `0.84`. Fixed-source competitor margin improves from about `0.22` to `0.58`; fixed-target competitor margin improves from about `0.25` to `0.51`. Moving answer margin improves only from about `0.25` to `0.34`, because that scalar has an extra branch problem.

<figure class="paper-figure">
  <img src="assets/figures/closure_boundary.svg" alt="Closure boundary">
  <figcaption><strong>Figure 12. Closure boundary.</strong> Route/write scalars are meaningful coordinates, but output-space closure is stronger, and nonlinear path curvature explains part of the remaining gap.</figcaption>
</figure>

There are two reasons moving answer margin is hard.

First, it is branch-sensitive. The "best wrong token" can change. This matters methodologically because moving answer margin is the scalar most people reach for first, but it can be the worst scalar for proving a mechanism. In the late audited window, direct moving-margin closure had only `R^2 = 0.099`. Holding the wrong-token branch fixed and adding the exact branch correction raised closure to `0.606` for fixed-source and `0.608` for fixed-target. Competitor switches were rare, only `298 / 6400` rows (`4.66%`), but on those switch rows the target-branch correction carried about `43%` of moving-margin energy. On switch rows, branch-aware closure rose from `0.161` direct to `0.713 -> 0.738`.

Second, the write side is nonlinear. A first-order endpoint gradient can be badly wrong. The line-integral diagnostic shows that integrating along the path can follow the actual endpoint change much better than a single endpoint linearization, especially for negative answer loss.

So the honest closure statement is:

```text
route/write scalars are meaningful;
output-space closure is stronger;
full answer-margin sufficiency by a small route set remains open.
```

## What This Means

The important interpretability object is not always a named head or neuron. In this model, the stable object is a role.

This matters because seed-level replication fails if I use the wrong address. `L2H1` is not always the circuit. But a support-value retrieval role appears across seeds. A contextual write/readout role appears across seeds too.

Formation tracking adds something that post-hoc interpretability does not usually give. A trained-model circuit analysis can say:

```text
this component matters now.
```

The formation audit can ask:

```text
did training actually write this role into the weights,
and which optimizer-state terms carried that movement?
```

That is the difference between a static circuit map and a developmental account.

<figure class="paper-figure">
  <img src="assets/figures/proof_status_ladder_updated.svg" alt="Proof status ladder">
  <figcaption><strong>Figure 13. Proof status.</strong> I have strong QK formation evidence, supported write functional-subspace evidence, and explicit open gaps.</figcaption>
</figure>

## Limitations

This is a controlled mechanistic case study, not a universal theory of transformers.

The main limitations are:

| limitation | current status |
| --- | --- |
| full answer-margin closure | partial; output-space closure is stronger than route closure |
| plain SGD vs AdamW training ablation | not run |
| scaling across width, depth, and task families | not done here |
| neuron-level decomposition | blocked by superposition; subspace-level methods are more honest |
| static `W_OV` low-rank theorem | not supported by current evidence |

The write-side current-vs-momentum variation is not a weakness to hide. It is a finding: the reference-seed fixed write scalar is momentum-heavy, while the cross-seed selected-winner aggregate is mostly Adam current-gradient with a smaller momentum contribution. The stable statement is that AdamW-preconditioned updates carry write growth and the raw SGD-equivalent term is tiny.

The strongest next experiments are therefore clear:

```text
1. run SGD-vs-AdamW optimizer ablations;
2. scale width/depth and test whether role-address dissociation persists;
3. apply the role-level method to another task family;
4. improve closure with residual-state and line-integral proof objects.
```

## Conclusion

This study does not show that all transformer circuits form this way. It shows that, in a controlled symbolic retrieval setting, the stable unit of formation can be a role rather than a component address. The QK side of that role becomes visible as a low-rank support-value matcher, and exact optimizer accounting shows that AdamW-preconditioned updates, not the instantaneous raw-gradient direction alone, carry its growth. The write side repeats across seeds as a contextual residual coupling rather than a clean `W_OV` matrix. The remaining challenge is to test whether this role-level, optimizer-state-aware account survives optimizer ablations, scaling, and less synthetic tasks.

## Audit Trail

The paper page is the narrative. The other pages are the audit surface.

Use the reproducibility page for exact commands, environment notes, and expected outputs. Use the CLI guide for the tools added during this project. Use the artifact map to connect each claim to its run directory.

The most important artifact families are:

```text
QK birth:
  weight_svd_trace
  bilinear_qk_match_separation
  bilinear_qk_rank_adam_state_attribution

cross-seed QK:
  symbolic_kv_cross_seed_adam/*/cross_seed_head_selection.json
  symbolic_kv_cross_seed_adam/*/bilinear_qk_rank_adam_state_attribution

write-side functional subspace:
  mlp_input_functional_subspace
  mlp_functional_subspace_trajectory
  mlp_functional_write_adam_state_attribution

closure:
  route_to_scalar_closure
  output_route_closure
  answer_margin_branch_decomposition
  component_output_rescue_line_integral
```

The research claim is intentionally bounded:

```text
In this small symbolic transformer,
AdamW-trained circuit formation is best explained at the role and subspace level.

QK forms a low-rank support-value route.
The write side forms a contextual residual coupling.
Both roles replicate across seeds while their component addresses move.
```

## References

- Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 2017.
- Kingma, Diederik P., and Jimmy Ba. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). 2014.
- Loshchilov, Ilya, and Frank Hutter. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). 2017.
- Elhage, Nelson, Neel Nanda, Catherine Olsson, et al. [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). Transformer Circuits Thread, 2021.
- Olsson, Catherine, Nelson Elhage, Neel Nanda, et al. [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html). Transformer Circuits Thread, 2022.
- Elhage, Nelson, Tristan Hume, Catherine Olsson, et al. [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html). Transformer Circuits Thread, 2022.
- Meng, Kevin, David Bau, Alex Andonian, and Yonatan Belinkov. [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262). 2022.
- Power, Alethea, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177). 2022.
