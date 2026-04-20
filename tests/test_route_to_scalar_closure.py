import pytest

from circuit.analysis.route_to_scalar_closure import (
    _build_scalar_observations,
    _switch_bucket_matches,
)


def test_switch_bucket_matches() -> None:
    assert _switch_bucket_matches({"competitor_switched": False}, "all") is True
    assert _switch_bucket_matches({"competitor_switched": True}, "all") is True
    assert _switch_bucket_matches({"competitor_switched": False}, "same_competitor") is True
    assert _switch_bucket_matches({"competitor_switched": True}, "same_competitor") is False
    assert _switch_bucket_matches({"competitor_switched": False}, "competitor_switch") is False
    assert _switch_bucket_matches({"competitor_switched": True}, "competitor_switch") is True


def test_build_scalar_observations_joins_route_deltas() -> None:
    scalar_pair_rows = [
        {
            "source_step": 1,
            "target_step": 2,
            "step_gap": 1,
            "pair_id": "pair_0",
            "interval_pair_id": "pair_0::clean",
            "split": "validation_iid",
            "pair_type": "query_key",
            "margin_side": "clean",
            "competitor_switched": False,
            "scalars": {
                "moving_answer_margin": {
                    "source": 10.0,
                    "target": 11.25,
                    "delta": 1.25,
                }
            },
        }
    ]

    rows = _build_scalar_observations(
        scalar_pair_rows=scalar_pair_rows,
        route_closure_rows={(1, 2, "pair_0"): {"route_a": 3.0, "route_b": 2.5}},
        route_labels=["route_a", "route_b"],
        scalar_names=["moving_answer_margin"],
        switch_buckets=["all", "same_competitor", "competitor_switch"],
        margin_side="clean",
        pair_types=["query_key"],
    )

    assert len(rows) == 2
    assert {row["switch_bucket"] for row in rows} == {"all", "same_competitor"}
    assert rows[0]["actual_scalar_delta"] == pytest.approx(1.25)
    assert rows[0]["route_score_deltas"]["route_a"] == pytest.approx(3.0)
    assert rows[0]["route_score_deltas"]["route_b"] == pytest.approx(2.5)
