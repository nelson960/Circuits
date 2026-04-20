from __future__ import annotations

import pytest

from circuit.analysis.route_to_margin_closure import fit_route_to_margin_closure


def test_fit_route_to_margin_closure_recovers_linear_coefficients() -> None:
    fit = fit_route_to_margin_closure(
        route_delta_columns={
            "route_a": [1.0, 2.0, 3.0, 4.0],
            "route_b": [0.0, 1.0, 0.0, 1.0],
        },
        margin_deltas=[2.0, 7.0, 6.0, 11.0],
        fit_intercept=False,
    )

    assert fit["rank_deficient"] is False
    assert fit["r_squared"] == pytest.approx(1.0)
    assert fit["coefficients"]["route_a"] == pytest.approx(2.0)
    assert fit["coefficients"]["route_b"] == pytest.approx(3.0)
    assert fit["mean_abs_residual"] == pytest.approx(0.0)


def test_fit_route_to_margin_closure_rejects_underdetermined_design() -> None:
    with pytest.raises(ValueError, match="Underdetermined closure fit"):
        fit_route_to_margin_closure(
            route_delta_columns={
                "route_a": [1.0],
                "route_b": [2.0],
            },
            margin_deltas=[3.0],
            fit_intercept=False,
        )
