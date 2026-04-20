import pytest

from circuit.analysis.residual_state_rescue import (
    _resolve_patch_stages,
    _safe_correlation,
    summarize_rescue_rows,
)


def test_resolve_patch_stages_rejects_unknown_stage() -> None:
    assert _resolve_patch_stages(
        patch_stages=["layer_0_post_mlp", "layer_1_post_attn"],
        num_layers=2,
    ) == ["layer_0_post_mlp", "layer_1_post_attn"]
    with pytest.raises(ValueError, match="Unsupported patch stage"):
        _resolve_patch_stages(patch_stages=["layer_9_post_mlp"], num_layers=2)


def test_safe_correlation_handles_constant_values() -> None:
    assert _safe_correlation([1.0, 1.0], [2.0, 3.0]) is None
    assert _safe_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_summarize_rescue_rows_reports_recovery_fraction() -> None:
    rows = [
        {
            "scalar_name": "correct_value_logit",
            "endpoint_kind": "target",
            "source_component": "L0MLP",
            "patch_stage": "layer_1_post_mlp",
            "clean_scalar": 10.0,
            "source_ablated_scalar": 4.0,
            "patched_scalar": 8.0,
            "total_drop": 6.0,
            "rescue": 4.0,
            "unrecovered": 2.0,
            "improved_by_patch": True,
        },
        {
            "scalar_name": "correct_value_logit",
            "endpoint_kind": "target",
            "source_component": "L0MLP",
            "patch_stage": "layer_1_post_mlp",
            "clean_scalar": 12.0,
            "source_ablated_scalar": 8.0,
            "patched_scalar": 10.0,
            "total_drop": 4.0,
            "rescue": 2.0,
            "unrecovered": 2.0,
            "improved_by_patch": True,
        },
    ]

    summary = summarize_rescue_rows(rows=rows, denominator_threshold=1.0e-6)

    assert len(summary) == 1
    assert summary[0]["mean_total_drop"] == pytest.approx(5.0)
    assert summary[0]["mean_rescue"] == pytest.approx(3.0)
    assert summary[0]["mean_rescue_fraction_from_means"] == pytest.approx(0.6)
    assert summary[0]["mean_recovery_fraction_per_row"] == pytest.approx(((4.0 / 6.0) + (2.0 / 4.0)) / 2.0)
    assert summary[0]["improved_fraction"] == pytest.approx(1.0)
