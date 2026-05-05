from __future__ import annotations

import torch

from circuit.analysis.value_code_subspace_report import (
    _leave_one_out_centroid_predictions,
    _summarize_rows,
    _value_logit_metrics,
)
from circuit.vocab import Vocabulary


def test_value_logit_metrics_uses_value_tokens_only() -> None:
    vocab = Vocabulary(
        tokens=["<pad>", "K0", "V0", "V1"],
        token_to_id={"<pad>": 0, "K0": 1, "V0": 2, "V1": 3},
        key_tokens=["K0"],
        value_tokens=["V0", "V1"],
    )
    logits = torch.tensor([100.0, 99.0, 2.5, 1.0])

    metrics = _value_logit_metrics(
        logits=logits,
        answer_token_id=2,
        value_token_ids=torch.tensor(vocab.value_token_ids),
        vocab=vocab,
        label="unit",
    )

    assert metrics["best_value_token"] == "V0"
    assert metrics["best_wrong_value_token"] == "V1"
    assert metrics["value_margin"] == 1.5
    assert metrics["value_accuracy"] == 1.0


def test_leave_one_out_centroid_predictions_separate_two_codes() -> None:
    vocab = Vocabulary(
        tokens=["<pad>", "V0", "V1"],
        token_to_id={"<pad>": 0, "V0": 1, "V1": 2},
        key_tokens=[],
        value_tokens=["V0", "V1"],
    )
    payloads = [
        {"group_token_id": 1, "_vector": torch.tensor([1.0, 0.0])},
        {"group_token_id": 1, "_vector": torch.tensor([1.0, 0.1])},
        {"group_token_id": 2, "_vector": torch.tensor([-1.0, 0.0])},
        {"group_token_id": 2, "_vector": torch.tensor([-1.0, 0.1])},
    ]

    predictions = _leave_one_out_centroid_predictions(payloads=payloads, vocab=vocab, label="unit")

    assert set(predictions) == {0, 1, 2, 3}
    assert all(prediction["centroid_scored"] for prediction in predictions.values())
    assert all(prediction["centroid_prediction_correct"] == 1.0 for prediction in predictions.values())


def test_summarize_rows_reports_value_code_metrics() -> None:
    rows = [
        {
            "checkpoint_step": 10,
            "stage": "final_norm",
            "position_role": "prediction",
            "group_by": "answer_value",
            "group_token_id": 1,
            "vector_norm": 2.0,
            "identity_overlap": 0.8,
            "all_vector_overlap": 0.9,
            "mean_abs_cosine": 0.4,
            "stage_lens_value_margin": 1.0,
            "final_value_margin": 1.0,
            "stage_lens_value_accuracy": 1.0,
            "final_value_accuracy": 1.0,
            "centroid_scored": True,
            "centroid_prediction_correct": 1.0,
            "centroid_cosine_margin": 0.5,
        },
        {
            "checkpoint_step": 10,
            "stage": "final_norm",
            "position_role": "prediction",
            "group_by": "answer_value",
            "group_token_id": 2,
            "vector_norm": 4.0,
            "identity_overlap": 0.6,
            "all_vector_overlap": 0.7,
            "mean_abs_cosine": 0.2,
            "stage_lens_value_margin": -1.0,
            "final_value_margin": -1.0,
            "stage_lens_value_accuracy": 0.0,
            "final_value_accuracy": 0.0,
            "centroid_scored": True,
            "centroid_prediction_correct": 0.0,
            "centroid_cosine_margin": -0.25,
        },
    ]

    summary_rows = _summarize_rows(rows=rows)

    assert len(summary_rows) == 1
    summary = summary_rows[0]
    assert summary["num_rows"] == 2
    assert summary["num_unique_groups"] == 2
    assert summary["mean_vector_norm"] == 3.0
    assert summary["mean_identity_overlap"] == 0.7
    assert summary["leave_one_out_centroid_accuracy"] == 0.5
