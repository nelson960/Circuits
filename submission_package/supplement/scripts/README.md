# Reproduction Commands

The scripts use:

```bash
export PYTHONPATH=src
python -m circuit.cli ...
```

They intentionally avoid local machine paths such as `/opt/miniconda3/...`.

## Scripts

- `reproduce_training_setup.sh`: regenerate benchmark and train the two seed-7 reference runs.
- `reproduce_optimizer_ablation_training.sh`: rerun the bounded seed-7 optimizer-ablation configs.
- `reproduce_qk_route.sh`: rerun a representative QK route-separation analysis.
- `reproduce_write_analysis.sh`: rerun a representative contextual value-code transfer-rescue analysis.

The compact tables in `../tables/` are the reviewer-facing audit artifacts for the submitted numbers. Full raw checkpoint sweeps are not included in the compact supplement.
