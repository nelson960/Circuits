# Hardware Note

The experiments were run on local accelerator hardware. The original development environment used an Apple Metal/MPS device for many runs, but the code paths are standard PyTorch.

Exact hardware affects runtime and memory limits. It should not change the compact derived artifacts when the same configs, seeds, checkpoints, and probe sets are used.

Large raw checkpoint sweeps are intentionally excluded from this compact supplement.
