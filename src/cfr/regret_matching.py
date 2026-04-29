import torch


def regret_matching_plus(advantages: torch.Tensor,
                         legal_mask: torch.Tensor) -> torch.Tensor:
    """
    Derive a mixed strategy from advantage estimates using regret matching+.

    advantages: (n_actions,) — raw values from AdvantageNet
    legal_mask: (n_actions,) bool — True for legal actions
    Returns:    (n_actions,) probability distribution summing to 1.0
    """
    positive = torch.clamp(advantages, min=0.0) * legal_mask.float()
    total = positive.sum()
    if total < 1e-6:
        n_legal = legal_mask.float().sum()
        if n_legal < 1e-6:
            # No legal actions — return uniform over all actions as fallback
            n_all = float(advantages.shape[0])
            return torch.full_like(advantages, 1.0 / n_all)
        return legal_mask.float() / n_legal
    return positive / total
