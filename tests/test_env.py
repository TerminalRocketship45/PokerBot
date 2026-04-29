import pytest
from src.env.poker_env import PokerEnv


def test_leduc_creates_game():
    env = PokerEnv(use_hunl=False)
    state = env.new_game()
    assert state is not None


def test_leduc_plays_full_random_game():
    import random
    env = PokerEnv(use_hunl=False)
    state = env.new_game()
    steps = 0
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            state.apply_action(outcomes[0][0])
        else:
            action = random.choice(state.legal_actions())
            state.apply_action(action)
        steps += 1
        assert steps < 200, "Game did not terminate"
    returns = state.returns()
    assert len(returns) == 2
    assert abs(sum(returns)) < 1e-6, "Returns should sum to zero (zero-sum game)"


def test_leduc_num_actions():
    env = PokerEnv(use_hunl=False)
    assert env.num_actions() >= 2


def test_leduc_info_state_size():
    env = PokerEnv(use_hunl=False)
    size = env.info_state_size()
    assert size > 0
    print(f"Leduc info state size: {size}")
