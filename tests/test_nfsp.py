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

def test_replay_buffer_sample_raises_when_empty():
    buf = ReplayBuffer(capacity=100)
    try:
        buf.sample(1)
        assert False, "Should have raised"
    except ValueError:
        pass
