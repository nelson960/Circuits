# circuit formation

Mechanistic interpretability research repo for studying circuit formation in small decoder-only next-token models.

The current benchmark is a stream-based symbolic KV retrieval task trained with plain autoregressive next-token prediction. The scientific question is not only whether the model solves the task, but how the circuit forms during training, which factors change that formation, and why gradient descent reinforces one mechanistic solution over another.

Public paper page:

- [From Retrieval Motifs To SGD Circuit Selection](https://nelson960.github.io/Circuits/)

The project is closely related to prior work on motif emergence:

- [Mechanistic Transparency](https://nelson960.github.io/Mechanistic-Transparency/)


For current internal research progress, benchmark history, training results, checkpoint-analysis findings, and the evolving formation-analysis plan, see [results.md](results.md).

For a plain-language overview of the main terms and analysis objects, see [notes.md](notes.md).

For the layered checkpoint-analysis program, see [docs/checkpoint_analysis_plan.md](docs/checkpoint_analysis_plan.md).
