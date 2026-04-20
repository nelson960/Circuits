import pytest
import torch

from circuit.analysis.answer_scalar_residual_diagnosis import (
    _pair_in_switch_bucket,
    _resolve_unique_values,
    _scalar_value_for_pair,
)


def test_resolve_unique_values_deduplicates_and_rejects_unknown() -> None:
    assert _resolve_unique_values(
        values=["clean", "corrupted", "clean"],
        default_values=["clean"],
        allowed_values=["clean", "corrupted"],
        label="margin side",
    ) == ["clean", "corrupted"]
    with pytest.raises(ValueError, match="Unsupported margin side"):
        _resolve_unique_values(
            values=["both"],
            default_values=["clean"],
            allowed_values=["clean", "corrupted"],
            label="margin side",
        )


def test_pair_in_switch_bucket() -> None:
    same = {"competitor_switched": False}
    switched = {"competitor_switched": True}
    assert _pair_in_switch_bucket(row=same, switch_bucket="all") is True
    assert _pair_in_switch_bucket(row=switched, switch_bucket="all") is True
    assert _pair_in_switch_bucket(row=same, switch_bucket="same_competitor") is True
    assert _pair_in_switch_bucket(row=switched, switch_bucket="same_competitor") is False
    assert _pair_in_switch_bucket(row=same, switch_bucket="competitor_switch") is False
    assert _pair_in_switch_bucket(row=switched, switch_bucket="competitor_switch") is True


def test_scalar_value_uses_source_and_target_competitors() -> None:
    payload = {
        "correct_logit_by_pair_id": {"pair::clean": 10.0},
        "correct_log_prob_by_pair_id": {"pair::clean": -0.25},
        "best_wrong_token_id_by_pair_id": {"pair::clean": 101},
        "value_index_by_token_id": {101: 0, 202: 1},
        "value_logits_by_pair_id": {"pair::clean": torch.tensor([7.0, 6.0])},
    }

    assert _scalar_value_for_pair(
        scalar_name="moving_answer_margin",
        payload=payload,
        interval_pair_id="pair::clean",
        source_best_wrong_token_id=101,
        target_best_wrong_token_id=202,
    ) == pytest.approx(3.0)
    assert _scalar_value_for_pair(
        scalar_name="fixed_target_competitor_margin",
        payload=payload,
        interval_pair_id="pair::clean",
        source_best_wrong_token_id=101,
        target_best_wrong_token_id=202,
    ) == pytest.approx(4.0)
    assert _scalar_value_for_pair(
        scalar_name="source_best_wrong_logit",
        payload=payload,
        interval_pair_id="pair::clean",
        source_best_wrong_token_id=101,
        target_best_wrong_token_id=202,
    ) == pytest.approx(7.0)
    assert _scalar_value_for_pair(
        scalar_name="negative_answer_loss",
        payload=payload,
        interval_pair_id="pair::clean",
        source_best_wrong_token_id=101,
        target_best_wrong_token_id=202,
    ) == pytest.approx(-0.25)
