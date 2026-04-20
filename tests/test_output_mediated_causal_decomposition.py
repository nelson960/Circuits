import pytest

from circuit.analysis.output_mediated_causal_decomposition import (
    _summarize_downstream_rows,
    _summarize_source_rows,
    _validate_baseline_scalar_values,
)


def test_validate_baseline_scalar_values_rejects_mismatch() -> None:
    rows = [
        {
            "source_step": 1,
            "target_step": 2,
            "pair_id": "pair_0",
            "margin_side": "clean",
            "scalars": {
                "correct_value_logit": {
                    "source": 3.0,
                    "target": 4.0,
                }
            },
        }
    ]
    baseline_values = {
        (1, 2, "pair_0", "clean", "correct_value_logit", "source"): {
            "scalar_value_recomputed": 3.0,
        },
        (1, 2, "pair_0", "clean", "correct_value_logit", "target"): {
            "scalar_value_recomputed": 4.2,
        },
    }

    with pytest.raises(RuntimeError, match="Baseline scalar recomputation mismatch"):
        _validate_baseline_scalar_values(
            scalar_pair_rows=rows,
            baseline_values=baseline_values,
            scalar_names=["correct_value_logit"],
            endpoint_roles=["source", "target"],
            tolerance=0.01,
        )


def test_summarize_source_rows_reports_mediation_terms() -> None:
    rows = [
        {
            "scalar_name": "correct_value_logit",
            "endpoint_kind": "target",
            "source_component": "L0MLP",
            "total_causal_effect": 10.0,
            "direct_source_dla": 1.0,
            "mediated_downstream_sum": 7.0,
            "direct_plus_mediated": 8.0,
            "mediation_residual": 2.0,
        },
        {
            "scalar_name": "correct_value_logit",
            "endpoint_kind": "target",
            "source_component": "L0MLP",
            "total_causal_effect": 6.0,
            "direct_source_dla": 2.0,
            "mediated_downstream_sum": 3.0,
            "direct_plus_mediated": 5.0,
            "mediation_residual": 1.0,
        },
    ]

    summary = _summarize_source_rows(rows)

    assert len(summary) == 1
    assert summary[0]["mean_total_causal_effect"] == pytest.approx(8.0)
    assert summary[0]["mean_direct_source_dla"] == pytest.approx(1.5)
    assert summary[0]["mean_mediated_downstream_sum"] == pytest.approx(5.0)
    assert summary[0]["mean_direct_plus_mediated"] == pytest.approx(6.5)
    assert summary[0]["mean_explained_fraction"] == pytest.approx(6.5 / 8.0)


def test_summarize_downstream_rows_groups_source_to_downstream_effects() -> None:
    rows = [
        {
            "scalar_name": "fixed_source_competitor_margin",
            "endpoint_kind": "target",
            "source_component": "L0MLP",
            "downstream_component": "L2H1",
            "baseline_downstream_dla": 5.0,
            "ablated_downstream_dla": 2.0,
            "mediated_effect": 3.0,
        },
        {
            "scalar_name": "fixed_source_competitor_margin",
            "endpoint_kind": "target",
            "source_component": "L0MLP",
            "downstream_component": "L2H1",
            "baseline_downstream_dla": 7.0,
            "ablated_downstream_dla": 3.0,
            "mediated_effect": 4.0,
        },
    ]

    summary = _summarize_downstream_rows(rows)

    assert len(summary) == 1
    assert summary[0]["mean_baseline_downstream_dla"] == pytest.approx(6.0)
    assert summary[0]["mean_ablated_downstream_dla"] == pytest.approx(2.5)
    assert summary[0]["mean_mediated_effect"] == pytest.approx(3.5)
