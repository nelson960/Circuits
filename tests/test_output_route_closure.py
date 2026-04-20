import pytest

from circuit.analysis.output_route_closure import (
    _build_endpoint_requests,
    _build_output_route_observations,
    _filter_component_labels,
)


def test_filter_component_labels_rejects_unknown_component() -> None:
    assert _filter_component_labels(
        requested_components=["embedding", "L0H1"],
        available_components=["embedding", "L0H0", "L0H1", "L0MLP"],
    ) == ["embedding", "L0H1"]
    with pytest.raises(ValueError, match="Unsupported component"):
        _filter_component_labels(
            requested_components=["L9H9"],
            available_components=["embedding", "L0H0"],
        )


def test_build_endpoint_requests_creates_source_and_target_requests() -> None:
    rows = [
        {
            "source_step": 10,
            "target_step": 11,
            "source_checkpoint": "step_000010.pt",
            "target_checkpoint": "step_000011.pt",
            "pair_id": "pair_0",
            "margin_side": "clean",
            "answer_target_id": 100,
            "source_best_wrong_token_id": 101,
            "target_best_wrong_token_id": 102,
        }
    ]

    requests = _build_endpoint_requests(scalar_pair_rows=rows, scalar_names=["moving_answer_margin"])

    assert len(requests) == 2
    assert {request["endpoint_kind"] for request in requests} == {"source", "target"}
    assert requests[0]["request_id"][:5] == (10, 11, "pair_0", "clean", "moving_answer_margin")


def test_build_output_route_observations_computes_component_deltas() -> None:
    rows = [
        {
            "source_step": 1,
            "target_step": 2,
            "step_gap": 1,
            "pair_id": "pair_0",
            "interval_pair_id": "pair_0::clean",
            "split": "validation_iid",
            "pair_type": "support_value",
            "margin_side": "clean",
            "competitor_switched": False,
            "scalars": {
                "correct_value_logit": {
                    "source": 3.0,
                    "target": 5.0,
                    "delta": 2.0,
                }
            },
        }
    ]
    request_values = {
        (1, 2, "pair_0", "clean", "correct_value_logit", "source"): {
            "scalar_value_recomputed": 3.0,
            "component_values": {"L0H0": 1.0, "L0MLP": 4.0},
        },
        (1, 2, "pair_0", "clean", "correct_value_logit", "target"): {
            "scalar_value_recomputed": 5.0,
            "component_values": {"L0H0": 1.5, "L0MLP": 3.5},
        },
    }

    observations = _build_output_route_observations(
        scalar_pair_rows=rows,
        component_values_by_request=request_values,
        component_labels=["L0H0", "L0MLP"],
        scalar_names=["correct_value_logit"],
        switch_buckets=["all", "same_competitor", "competitor_switch"],
        scalar_value_tolerance=1.0e-6,
    )

    assert len(observations) == 2
    assert {row["switch_bucket"] for row in observations} == {"all", "same_competitor"}
    assert observations[0]["component_deltas"]["L0H0"] == pytest.approx(0.5)
    assert observations[0]["component_deltas"]["L0MLP"] == pytest.approx(-0.5)
