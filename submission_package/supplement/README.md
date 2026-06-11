# Anonymous Supplement

This supplement is the compact reviewer-facing artifact package for the paper:

**From Loss to Lookup: Tracing Circuit Formation in a Small Transformer**

It contains the metadata needed to audit the submitted claims without including large raw checkpoint sweeps or private research notes.

## Contents

- `configs/`: benchmark, training, dataset, probe, and optimizer-ablation configuration metadata.
- `tables/`: compact CSV tables for the main numbers reported in the paper.
- `figures/`: vector figure assets corresponding to the submission story.
- `scripts/`: reproduction commands and command templates using `python -m circuit.cli`.
- `environment/`: environment metadata.
- `results_ledger.md`: claim-by-claim audit map.
- `artifact_manifest.json`: machine-readable manifest of compact artifacts.

## What Is Not Included

This package does not include:

- raw checkpoint sweeps,
- complete exploratory logs,
- private research notes,
- stale `results.md` entries,
- local absolute paths,
- author-identifying repository or website links.

## How To Use

1. Recreate the environment from `environment/environment.yml`.
2. Generate the benchmark with `scripts/reproduce_training_setup.sh`.
3. Train or inspect the reference configs under `configs/`.
4. Use `results_ledger.md` to map each paper claim to a compact table and reproduction command.

Large experiments can be rerun from the configs and commands, but the compact tables are the fastest way to audit the submitted numerical claims.
