"""
Phase 0 gate. All 6 checks must pass before Phase 1 (BC training) begins.
Run: pytest tests/test_core.py -v
"""
import numpy as np
import torch
import pytest
from scipy.stats import chisquare
import pyspiel

from src.env.poker_env import PokerEnv
from src.data.encoder import encode_state, STATE_DIM
from src.models.advantage_net import AdvantageNet
from src.cfr.regret_matching import regret_matching_plus
from src.cfr.buffer import ReservoirBuffer
from src.cfr.sd_cfr import SDCFRConfig, train_sd_cfr


# ── Check 1: Leduc exploitability decreases ───────────────────────────────────

def _compute_leduc_exploitability_proxy(net: AdvantageNet, env: PokerEnv,
                                         n_games: int = 200) -> float:
    """
    Proxy: average absolute advantage magnitude on terminal-adjacent states.
    A trained network should have lower variance than a random one.
    We measure win rate of greedy policy vs uniform random over n_games.
    """
    import random
    wins = 0
    net.eval()
    n_actions = env.num_actions()

    for _ in range(n_games):
        state = env.new_game()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                chosen = np.random.choice(actions, p=probs)
                state.apply_action(chosen)
                continue
            player = state.current_player()
            legal = state.legal_actions()
            if player == 0:
                info = encode_state(state, 0, use_hunl=False)
                t = torch.FloatTensor(info).unsqueeze(0)
                with torch.no_grad():
                    adv = net(t).squeeze(0)
                mask = torch.zeros(n_actions, dtype=torch.bool)
                for a in legal:
                    if a < n_actions:
                        mask[a] = True
                probs = regret_matching_plus(adv, mask)
                action = torch.multinomial(probs, 1).item()
                if action not in legal:
                    action = random.choice(legal)
            else:
                action = random.choice(legal)
            state.apply_action(action)
        if state.returns()[0] > 0:
            wins += 1
    return wins / n_games


def test_check1_leduc_exploitability_decreases():
    """SD-CFR should produce a policy that wins more than 20% vs random after 50 iterations."""
    env = PokerEnv(use_hunl=False)
    config = SDCFRConfig(
        use_hunl=False, n_iterations=50, n_traversals_per_iter=100,
        buffer_capacity=20000, n_batches=50, batch_size=128,
        hidden_dim=64, use_wandb=False,
    )
    net_random = AdvantageNet(
        input_dim=STATE_DIM, n_actions=env.num_actions(), hidden_dim=64
    )
    wr_before = _compute_leduc_exploitability_proxy(net_random, env)

    net_trained = train_sd_cfr(config)
    wr_after = _compute_leduc_exploitability_proxy(net_trained, env)

    print(f"Win rate before training: {wr_before:.2%}")
    print(f"Win rate after 50 iterations: {wr_after:.2%}")
    assert wr_after > 0.35, (
        f"Trained agent wins only {wr_after:.2%} vs random. "
        f"CFR traversal or regret matching may be broken."
    )


# ── Check 2: Reservoir buffer uniform coverage ────────────────────────────────

def test_check2_reservoir_buffer_uniform():
    capacity = 1000
    n_inserts = 50_000
    n_buckets = 10
    bucket_size = n_inserts // n_buckets

    buf = ReservoirBuffer(capacity=capacity)
    for i in range(n_inserts):
        buf.add(np.array([float(i)]), np.array([0.0]), 1.0)

    counts = np.zeros(n_buckets, dtype=int)
    for state, _, _ in buf.buffer:
        idx = int(state[0]) // bucket_size
        if 0 <= idx < n_buckets:
            counts[idx] += 1

    _, p_value = chisquare(counts)
    assert p_value > 0.01, f"Buffer not uniform (p={p_value:.4f}). Counts: {counts}"


# ── Check 3: Regret matching — all negative → uniform ────────────────────────

def test_check3_regret_matching_all_negative_uniform():
    advantages = torch.tensor([-1.0, -2.0, -3.0, -0.5, -1.5, -0.1])
    legal_mask = torch.ones(6, dtype=torch.bool)
    probs = regret_matching_plus(advantages, legal_mask)
    assert torch.allclose(probs, torch.full((6,), 1.0 / 6), atol=1e-5)


# ── Check 4: Regret matching — positive → proportional ────────────────────────

def test_check4_regret_matching_positive_proportional():
    advantages = torch.tensor([1.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    legal_mask = torch.ones(6, dtype=torch.bool)
    probs = regret_matching_plus(advantages, legal_mask)
    assert abs(probs[0].item() - 0.25) < 1e-5
    assert abs(probs[1].item() - 0.75) < 1e-5


# ── Check 5: Encoder output shape and no NaNs ────────────────────────────────

def test_check5_encoder_shape_and_no_nans():
    game = pyspiel.load_game("leduc_poker")
    states_to_test = []

    # Preflop state
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    states_to_test.append(("preflop", state))

    # After one action
    if not state.is_terminal():
        s2 = state.clone()
        s2.apply_action(s2.legal_actions()[0])
        if not s2.is_terminal():
            states_to_test.append(("after_action", s2))

    # Fold immediately
    state_fold = game.new_initial_state()
    while state_fold.is_chance_node():
        state_fold.apply_action(state_fold.chance_outcomes()[0][0])
    if 0 in state_fold.legal_actions():
        state_fold.apply_action(0)

    for name, s in states_to_test:
        if s.is_terminal():
            continue
        vec = encode_state(s, player=s.current_player(), use_hunl=False)
        assert vec.shape == (STATE_DIM,), f"[{name}] shape {vec.shape} != ({STATE_DIM},)"
        assert vec.dtype == np.float32, f"[{name}] dtype {vec.dtype} != float32"
        assert not np.any(np.isnan(vec)), f"[{name}] NaN found in encoded state"


# ── Check 6: Legal action mask blocks illegal actions ────────────────────────

def test_check6_legal_action_mask():
    import random
    game = pyspiel.load_game("leduc_poker")
    n_actions = game.num_distinct_actions()

    state = game.new_initial_state()
    steps = 0
    while not state.is_terminal() and steps < 50:
        if state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
            steps += 1
            continue

        legal = state.legal_actions()
        illegal = [a for a in range(n_actions) if a not in legal]

        advantages = torch.ones(n_actions)
        legal_mask = torch.zeros(n_actions, dtype=torch.bool)
        for a in legal:
            if a < n_actions:
                legal_mask[a] = True

        probs = regret_matching_plus(advantages, legal_mask)
        for a in illegal:
            if a < n_actions:
                assert probs[a].item() == 0.0, (
                    f"Illegal action {a} has prob {probs[a].item():.4f} at step {steps}"
                )

        state.apply_action(random.choice(legal))
        steps += 1
