from pathlib import Path

import pytest

from circuit.analysis.answer_margin_delta_decomposition import (
    _answer_margin_delta_metric_row,
    _resolve_margin_sides,
)


def test_resolve_margin_sides_defaults_and_deduplicates() -> None:
    assert _resolve_margin_sides(None) == ["clean"]
    assert _resolve_margin_sides(["clean", "corrupted", "clean"]) == ["clean", "corrupted"]


def test_resolve_margin_sides_rejects_unsupported_side() -> None:
    with pytest.raises(ValueError, match="Unsupported margin side"):
        _resolve_margin_sides(["both"])


def test_answer_margin_delta_metric_row_uses_direct_margin_delta() -> None:
    row = _answer_margin_delta_metric_row(
        source_step=10,
        target_step=11,
        source_checkpoint=Path("step_000010.pt"),
        target_checkpoint=Path("step_000011.pt"),
        learning_rate=0.01,
        split="__all__",
        pair_type="__all__",
        margin_side="clean",
        source_actual={
            "num_pairs": 2,
            "num_entries": 2,
            "answer_margin": 1.0,
            "answer_loss": 0.7,
            "answer_accuracy": 0.5,
        },
        target_actual={
            "num_pairs": 2,
            "num_entries": 2,
            "answer_margin": 1.25,
            "answer_loss": 0.6,
            "answer_accuracy": 1.0,
        },
        source_payload={
            "num_entries": 2,
            "answer_margin": 1.0,
            "answer_loss": 0.7,
            "answer_accuracy": 0.5,
            "num_batches": 1,
            "zero_gradient_parameter_names": [],
        },
        dot_summary={
            "dot": 0.2,
            "left_l2_norm": 2.0,
            "right_l2_norm": 4.0,
            "cosine": 0.025,
            "num_parameters": 8,
        },
        min_error_denominator=1.0e-9,
    )

    assert row["actual_delta"] == pytest.approx(0.25)
    assert row["predicted_delta"] == pytest.approx(0.2)
    assert row["residual"] == pytest.approx(0.05)
    assert row["relative_error"] == pytest.approx(0.2)
    assert row["sign_match"] is True
