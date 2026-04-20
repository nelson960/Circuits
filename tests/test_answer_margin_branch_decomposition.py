import pytest

from circuit.analysis.answer_margin_branch_decomposition import (
    build_branch_aware_closure_rows,
    build_branch_decomposition_rows,
    summarize_branch_decomposition,
)


def _scalar_row() -> dict:
    return {
        "source_step": 10,
        "target_step": 11,
        "step_gap": 1,
        "pair_id": "pair_0",
        "interval_pair_id": "pair_0::clean",
        "split": "validation_iid",
        "pair_type": "support_value",
        "margin_side": "clean",
        "competitor_switched": True,
        "answer_target_id": 100,
        "source_best_wrong_token_id": 101,
        "target_best_wrong_token_id": 102,
        "scalars": {
            "moving_answer_margin": {"source": 4.0, "target": 5.0, "delta": 1.0},
            "fixed_source_competitor_margin": {"source": 4.0, "target": 7.0, "delta": 3.0},
            "fixed_target_competitor_margin": {"source": 6.0, "target": 5.0, "delta": -1.0},
            "source_best_wrong_logit": {"source": 6.0, "target": 5.0, "delta": -1.0},
            "target_best_wrong_logit": {"source": 4.0, "target": 7.0, "delta": 3.0},
        },
    }


def test_branch_decomposition_reconstructs_moving_margin() -> None:
    rows = build_branch_decomposition_rows(
        scalar_pair_rows=[_scalar_row()],
        margin_side="clean",
        pair_types=["support_value"],
        switch_buckets=["all", "competitor_switch", "same_competitor"],
        reconstruction_tolerance=1.0e-9,
    )

    assert len(rows) == 2
    row = rows[0]
    assert row["target_branch_correction"] == pytest.approx(-2.0)
    assert row["source_branch_correction"] == pytest.approx(2.0)
    assert row["reconstructed_from_source_fixed"] == pytest.approx(1.0)
    assert row["reconstructed_from_target_fixed"] == pytest.approx(1.0)
    assert row["source_reconstruction_error"] == pytest.approx(0.0)
    assert row["target_reconstruction_error"] == pytest.approx(0.0)


def test_branch_summary_reports_switch_bucket_energy() -> None:
    rows = build_branch_decomposition_rows(
        scalar_pair_rows=[_scalar_row()],
        margin_side="clean",
        pair_types=None,
        switch_buckets=["competitor_switch"],
        reconstruction_tolerance=1.0e-9,
    )
    summary = summarize_branch_decomposition(rows)

    aggregate = next(row for row in summary if row["pair_type"] == "__all__")
    assert aggregate["competitor_switch_fraction"] == pytest.approx(1.0)
    assert aggregate["target_branch_energy_fraction_of_moving"] == pytest.approx(4.0)
    assert aggregate["source_branch_energy_fraction_of_moving"] == pytest.approx(4.0)


def test_branch_aware_closure_adds_exact_branch_correction() -> None:
    branch_rows = build_branch_decomposition_rows(
        scalar_pair_rows=[_scalar_row()],
        margin_side="clean",
        pair_types=None,
        switch_buckets=["all"],
        reconstruction_tolerance=1.0e-9,
    )
    output_rows = {
        (10, 11, "pair_0::clean", "clean", "all", "moving_answer_margin"): {
            "predicted_scalar_delta": 0.0,
        },
        (10, 11, "pair_0::clean", "clean", "all", "fixed_source_competitor_margin"): {
            "predicted_scalar_delta": 2.5,
        },
        (10, 11, "pair_0::clean", "clean", "all", "fixed_target_competitor_margin"): {
            "predicted_scalar_delta": -0.5,
        },
    }

    rows = build_branch_aware_closure_rows(branch_rows=branch_rows, output_closure_rows=output_rows)

    assert len(rows) == 1
    assert rows[0]["source_fixed_branch_predicted_delta"] == pytest.approx(0.5)
    assert rows[0]["target_fixed_branch_predicted_delta"] == pytest.approx(1.5)
    assert rows[0]["source_fixed_branch_residual"] == pytest.approx(0.5)
    assert rows[0]["target_fixed_branch_residual"] == pytest.approx(-0.5)
