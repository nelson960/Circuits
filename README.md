# circuit formation

Research repo for mechanistic interpretability experiments on how a small decoder-only transformer learns a symbolic latest-write key-value lookup task.

The project is not only about whether the model solves the task. The motivating question is:

```text
How does SGD find a circuit at all?
```

The current paper sharpens that question into a version we can actually measure:

```text
How does gradient-based training, realized here as AdamW rather than raw SGD,
turn a dense, shared, polysemantic substrate into a functional retrieval circuit?
```

## Current result

The current paper result is centered on a symbolic KV benchmark and a reference 3-layer, 4-head, width-128 transformer.

The strongest supported claim in the repo right now is:

```text
training builds a support-value retrieval role;
in the reference seed that role is visible as low-rank L2H1 W_QK formation;
exact AdamW update decomposition explains the route growth much better than raw SGD;
across 5 seeds the role repeats, but the winning head address changes.
```

So the original SGD question is still the right historical motivation, but the current measured answer is narrower:

```text
raw SGD by itself is too small to explain the realized route growth in this run;
the optimizer state is part of the mechanism-selection story.
```

This is a detailed mechanistic account for one task family. It is not a theorem about all transformers.

## Public paper and research docs

- [From Loss To Lookup: Tracing Circuit Formation In A Small Transformer](https://nelson960.github.io/Circuits/)
- [Internal research log](results.md)
- [Analysis CLI guide](docs/analysis_cli_guide.md)
- [Checkpoint analysis plan](docs/checkpoint_analysis_plan.md)
- [Shared feature dynamics plan](docs/shared_feature_dynamics_plan.md)
- [Plain-language notes](notes.md)

Related earlier project:

- [Mechanistic Transparency](https://nelson960.github.io/Mechanistic-Transparency/)

## Repo layout

- [`src/circuit`](src/circuit): training, evaluation, benchmark generation, and analysis code
- [`scripts`](scripts): helper pipelines, including cross-seed execution
- [`configs`](configs): benchmark and training configs
- [`docs`](docs): public paper page and supporting docs
- [`artifacts`](artifacts): generated runs, checkpoints, reports, and plots
- [`tests`](tests): test suite

## Environment

The repo targets Python 3.12.

Conda environment:

```bash
conda env create -f environment.yml
conda activate ml
```

Editable install:

```bash
pip install -e .
```

Dev extras:

```bash
pip install -e ".[dev]"
```

## Minimal workflow

### 1. Generate a benchmark

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli generate-benchmark \
  --config configs/benchmark/symbolic_kv_base.json
```

Use the printed output directory as `BENCHMARK_DIR` in the next step.

### 2. Train a model

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli train \
  --config configs/train/symbolic_kv_formation.json
```

### 3. Evaluate a checkpoint

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli evaluate \
  --config artifacts/runs/symbolic_kv_reference_formation/run_config.json \
  --checkpoint artifacts/runs/symbolic_kv_reference_formation/checkpoints/step_016000.pt
```

### 4. Build a probe set

```bash
PYTHONPATH=src /opt/miniconda3/envs/ml/bin/python -m circuit.cli generate-probe-set \
  --benchmark-dir "$BENCHMARK_DIR" \
  --output probe_set.jsonl \
  --examples-per-split 96 \
  --overwrite
```

### 5. Run formation analysis

Do not start by guessing commands from memory. Use:

- [docs/analysis_cli_guide.md](docs/analysis_cli_guide.md)

That guide documents the analysis stack actually used in the current paper:

- trained-model geometry
- route competition
- weight-space SVD traces
- contextual semanticity checks
- checkpoint-to-checkpoint route attribution
- exact optimizer traces
- actual-batch and Adam-state attribution
- cross-seed validation

## Main analysis entry points

These are the commands that matter most for the current research result:

- `attention-geometry-trace`
- `path-logit-decomposition`
- `route-competition-report`
- `weight-svd-trace`
- `weight-svd-patterns`
- `contextual-key-separability`
- `contextual-svd-alignment`
- `bilinear-qk-match-separation`
- `bilinear-qk-rank-update-attribution`
- `optimizer-update-trace`
- `bilinear-qk-rank-actual-batch-attribution`
- `bilinear-qk-rank-adam-state-attribution`
- `scripts/cross_seed_adam_pipeline.py`

## What to read first

If you are trying to understand the repo:

1. read the [public paper page](https://nelson960.github.io/Circuits/)
2. read [docs/analysis_cli_guide.md](docs/analysis_cli_guide.md)
3. inspect [results.md](results.md) for the run history and current evidence

If you are trying to extend the research:

1. start from [docs/checkpoint_analysis_plan.md](docs/checkpoint_analysis_plan.md)
2. use the CLI guide to stay consistent with existing experiments
3. add new analyses only when they close a real proof gap

## Current limitations

The repo has strong support for:

- route-level causal analysis
- weight-space formation analysis
- exact optimizer-state attribution for traced windows
- cross-seed role-level validation

It does not yet establish:

- full answer-margin closure from a small route family
- that plain SGD without AdamW would form the same route
- that the same method scales directly to large language models

## Development rule

This repo follows a strict rule in analysis code:

```text
no hardcoding, no fallbacks, no hiding errors
```

If an analysis input is inconsistent, the command should fail and say why.
