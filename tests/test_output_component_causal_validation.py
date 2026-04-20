import pytest
import torch

from circuit.analysis.output_component_causal_validation import (
    _component_mask_kwargs,
    _scalar_from_logits,
    _summarize_validation_rows,
)


def test_component_mask_kwargs_rejects_embedding() -> None:
    device = torch.device("cpu")
    with pytest.raises(ValueError, match="Embedding DLA cannot be causally validated"):
        _component_mask_kwargs(component="embedding", num_layers=3, num_heads=4, device=device)


def test_component_mask_kwargs_builds_head_and_mlp_masks() -> None:
    device = torch.device("cpu")
    head_kwargs = _component_mask_kwargs(component="L1H2", num_layers=3, num_heads=4, device=device)
    assert "head_mask" in head_kwargs
    assert float(head_kwargs["head_mask"][1][2].item()) == 0.0
    assert float(head_kwargs["head_mask"][1][1].item()) == 1.0

    mlp_kwargs = _component_mask_kwargs(component="L2MLP", num_layers=3, num_heads=4, device=device)
    assert mlp_kwargs["mlp_mask"][2] == 0.0
    assert mlp_kwargs["mlp_mask"][0] == 1.0


def test_scalar_from_logits_matches_endpoint_branch_definitions() -> None:
    logits = torch.tensor([0.0, 10.0, 7.0, 5.0])
    assert _scalar_from_logits(
        logits=logits,
        scalar_name="moving_answer_margin",
        answer_target_id=1,
        source_best_wrong_token_id=2,
        target_best_wrong_token_id=3,
        endpoint_kind="source",
    ) == pytest.approx(3.0)
    assert _scalar_from_logits(
        logits=logits,
        scalar_name="moving_answer_margin",
        answer_target_id=1,
        source_best_wrong_token_id=2,
        target_best_wrong_token_id=3,
        endpoint_kind="target",
    ) == pytest.approx(5.0)
    log_prob = _scalar_from_logits(
        logits=logits,
        scalar_name="negative_answer_loss",
        answer_target_id=1,
        source_best_wrong_token_id=2,
        target_best_wrong_token_id=3,
        endpoint_kind="target",
    )
    assert log_prob == pytest.approx(float(torch.log_softmax(logits, dim=-1)[1].item()))


def test_summarize_validation_rows_compares_causal_effect_to_dla() -> None:
    rows = [
        {
            "scalar_name": "correct_value_logit",
            "endpoint_kind": "target",
            "component": "L0H0",
            "baseline_scalar": 5.0,
            "ablated_scalar": 3.0,
            "causal_effect": 2.0,
            "dla_contribution": 1.5,
            "causal_minus_dla": 0.5,
            "sign_match": True,
        },
        {
            "scalar_name": "correct_value_logit",
            "endpoint_kind": "target",
            "component": "L0H0",
            "baseline_scalar": 4.0,
            "ablated_scalar": 3.5,
            "causal_effect": 0.5,
            "dla_contribution": 0.25,
            "causal_minus_dla": 0.25,
            "sign_match": True,
        },
    ]

    summary = _summarize_validation_rows(rows)

    assert len(summary) == 1
    assert summary[0]["mean_causal_effect"] == pytest.approx(1.25)
    assert summary[0]["mean_dla_contribution"] == pytest.approx(0.875)
    assert summary[0]["sign_match_fraction"] == pytest.approx(1.0)
