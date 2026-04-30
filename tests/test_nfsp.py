# tests/test_nfsp.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.models.policy_net import PolicyNet

def test_policy_net_output_shape():
    net = PolicyNet(input_dim=60, n_actions=6, hidden_dim=64)
    x = torch.randn(4, 60)
    out = net(x)
    assert out.shape == (4, 6), f"Expected (4,6), got {out.shape}"

def test_policy_net_sums_to_one():
    net = PolicyNet(input_dim=60, n_actions=6, hidden_dim=64)
    x = torch.randn(8, 60)
    out = net(x)
    sums = out.sum(dim=1)
    assert torch.allclose(sums, torch.ones(8), atol=1e-5), "Rows must sum to 1"

def test_policy_net_all_positive():
    net = PolicyNet(input_dim=60, n_actions=6, hidden_dim=64)
    x = torch.randn(8, 60)
    out = net(x)
    assert (out >= 0).all(), "All probabilities must be >= 0"

from src.nfsp.replay_buffer import ReplayBuffer
import numpy as np

def test_replay_buffer_add_and_sample():
    buf = ReplayBuffer(capacity=100)
    for i in range(50):
        buf.add(np.zeros(60, dtype=np.float32), action=1, G=0.5)
    assert len(buf) == 50
    batch = buf.sample(16)
    assert len(batch) == 16
    s, a, g = batch[0]
    assert s.shape == (60,)
    assert isinstance(a, (int, np.integer))
    assert isinstance(g, float)

def test_replay_buffer_overwrites_when_full():
    buf = ReplayBuffer(capacity=10)
    for i in range(20):
        buf.add(np.zeros(60, dtype=np.float32), action=0, G=float(i))
    assert len(buf) == 10  # capped at capacity
    # After 20 adds to capacity-10 buffer, all stored G values must be >= 10.0.
    # (items 0–9 must have been overwritten by items 10–19)
    batch = buf.sample(10)
    g_values = [g for _, _, g in batch]
    assert all(g >= 10.0 for g in g_values), (
        f"Buffer should contain only items 10-19 (G >= 10.0), got: {sorted(g_values)}"
    )

def test_replay_buffer_sample_raises_when_empty():
    buf = ReplayBuffer(capacity=100)
    try:
        buf.sample(1)
        assert False, "Should have raised"
    except ValueError:
        pass

from src.nfsp.agent import NFSPAgent
from src.models.advantage_net import AdvantageNet

def test_nfsp_agent_act_returns_legal_action():
    q_net = AdvantageNet(input_dim=60, n_actions=6, hidden_dim=64)
    pi_net = PolicyNet(input_dim=60, n_actions=6, hidden_dim=64)
    agent = NFSPAgent(q_net, pi_net, eta=0.5, epsilon=0.0)
    state_vec = np.random.randn(60).astype(np.float32)
    legal = [0, 1, 5]  # FOLD, CHECK_CALL, ALL_IN
    for _ in range(20):
        action, mode = agent.act(state_vec, legal)
        assert action in legal, f"action {action} not in legal {legal}"
        assert mode in ('br', 'avg')

def test_nfsp_agent_epsilon_1_acts_randomly():
    q_net = AdvantageNet(input_dim=60, n_actions=6, hidden_dim=64)
    pi_net = PolicyNet(input_dim=60, n_actions=6, hidden_dim=64)
    agent = NFSPAgent(q_net, pi_net, eta=1.0, epsilon=1.0)  # always br, always random
    state_vec = np.zeros(60, dtype=np.float32)
    legal = [1, 3]
    seen = set()
    for _ in range(100):
        action, mode = agent.act(state_vec, legal)
        seen.add(action)
        assert mode == 'br'
    assert seen == {1, 3}, "With epsilon=1.0, both actions must be seen"
