import torch
import pytest
from src.cfr.regret_matching import regret_matching_plus


def test_all_negative_returns_uniform():
    advantages = torch.tensor([-1.0, -2.0, -3.0, -0.5, -1.5, -0.1])
    legal_mask = torch.ones(6, dtype=torch.bool)
    probs = regret_matching_plus(advantages, legal_mask)
    expected = torch.full((6,), 1.0 / 6)
    assert torch.allclose(probs, expected, atol=1e-5)


def test_positive_advantages_proportional():
    advantages = torch.tensor([2.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    legal_mask = torch.ones(6, dtype=torch.bool)
    probs = regret_matching_plus(advantages, legal_mask)
    assert abs(probs[0].item() - 1.0 / 3) < 1e-5
    assert abs(probs[1].item() - 2.0 / 3) < 1e-5
    assert probs[2:].sum().item() < 1e-5


def test_mixed_only_positives_get_probability():
    advantages = torch.tensor([3.0, -1.0, 1.0, -2.0, 0.0, 0.0])
    legal_mask = torch.ones(6, dtype=torch.bool)
    probs = regret_matching_plus(advantages, legal_mask)
    assert probs[1].item() == 0.0
    assert probs[3].item() == 0.0
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_illegal_actions_get_zero_probability():
    advantages = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5, 0.1])
    legal_mask = torch.tensor([True, True, False, False, False, False])
    probs = regret_matching_plus(advantages, legal_mask)
    assert probs[2].item() == 0.0
    assert probs[3].item() == 0.0
    assert probs[4].item() == 0.0
    assert probs[5].item() == 0.0
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_output_sums_to_one():
    advantages = torch.randn(6)
    legal_mask = torch.ones(6, dtype=torch.bool)
    probs = regret_matching_plus(advantages, legal_mask)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_all_illegal_mask_returns_uniform_fallback():
    advantages = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    legal_mask = torch.zeros(6, dtype=torch.bool)  # all illegal
    probs = regret_matching_plus(advantages, legal_mask)
    # Should not raise, should not return NaN
    assert not torch.any(torch.isnan(probs))
    assert abs(probs.sum().item() - 1.0) < 1e-5
