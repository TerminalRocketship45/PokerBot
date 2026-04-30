import numpy as np
import torch
from src.models.advantage_net import AdvantageNet
from src.models.policy_net import PolicyNet

N_ACTIONS = 6


class NFSPAgent:
    """
    NFSP agent holding one Q-network (best response) and one π-network (average strategy).

    At each decision:
      - With prob eta  → 'br'  mode: act ε-greedy from Q-net
      - With prob 1-eta → 'avg' mode: sample from π-net softmax
    """

    def __init__(
        self,
        q_net: AdvantageNet,
        pi_net: PolicyNet,
        eta: float = 0.1,
        epsilon: float = 0.30,
    ):
        self.q_net   = q_net
        self.pi_net  = pi_net
        self.eta     = eta
        self.epsilon = epsilon

    def act(self, state_vec: np.ndarray, legal_actions: list) -> tuple:
        """Returns (action: int, mode: str) where mode is 'br' or 'avg'."""
        valid = [a for a in legal_actions if a < N_ACTIONS]
        if not valid:
            valid = list(legal_actions)

        if np.random.random() < self.eta:
            return self._br_act(state_vec, valid), 'br'
        return self._avg_act(state_vec, valid), 'avg'

    def _br_act(self, state_vec: np.ndarray, valid: list) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.choice(valid))
        t = torch.FloatTensor(state_vec).unsqueeze(0)
        self.q_net.eval()
        with torch.no_grad():
            q_vals = self.q_net(t).squeeze(0)
        best = max(valid, key=lambda a: q_vals[a].item())
        return int(best)

    def _avg_act(self, state_vec: np.ndarray, valid: list) -> int:
        t = torch.FloatTensor(state_vec).unsqueeze(0)
        self.pi_net.eval()
        with torch.no_grad():
            probs = self.pi_net(t).squeeze(0)
        legal_p = {a: probs[a].item() for a in valid}
        total = sum(legal_p.values())
        if total < 1e-9:
            return int(np.random.choice(valid))
        actions = list(legal_p.keys())
        weights = [legal_p[a] / total for a in actions]
        return int(np.random.choice(actions, p=weights))
