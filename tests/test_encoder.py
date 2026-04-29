import numpy as np
import pytest
import pyspiel
from src.data.encoder import encode_state, STATE_DIM

_HUNL_AVAILABLE = "universal_poker" in pyspiel.registered_names()


def _make_leduc_state(n_actions: int = 0):
    game = pyspiel.load_game("leduc_poker")
    state = game.new_initial_state()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    for _ in range(n_actions):
        if state.is_terminal():
            break
        if state.is_chance_node():
            state.apply_action(state.chance_outcomes()[0][0])
        else:
            state.apply_action(state.legal_actions()[0])
    return state


def test_encoder_shape_preflop():
    state = _make_leduc_state(0)
    vec = encode_state(state, player=0, use_hunl=False)
    assert vec.shape == (STATE_DIM,), f"Expected ({STATE_DIM},), got {vec.shape}"


def test_encoder_dtype():
    state = _make_leduc_state(0)
    vec = encode_state(state, player=0, use_hunl=False)
    assert vec.dtype == np.float32


def test_encoder_no_nans():
    for n in [0, 1, 2]:
        state = _make_leduc_state(n)
        if not state.is_terminal():
            vec = encode_state(state, player=0, use_hunl=False)
            assert not np.any(np.isnan(vec)), f"NaN found after {n} actions"


def test_encoder_deterministic():
    state = _make_leduc_state(0)
    v1 = encode_state(state, player=0, use_hunl=False)
    v2 = encode_state(state, player=0, use_hunl=False)
    np.testing.assert_array_equal(v1, v2)


def test_encoder_hunl_shape_and_no_nans():
    """HUNL encoder returns float32[60] with no NaNs at preflop."""
    if not _HUNL_AVAILABLE:
        pytest.skip("universal_poker not available in this OpenSpiel build")
    from src.env.poker_env import PokerEnv
    env = PokerEnv(use_hunl=True)
    state = env.new_game()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    if state.is_terminal():
        pytest.skip("Game ended immediately")
    player = state.current_player()
    vec = encode_state(state, player=player, use_hunl=True)
    assert vec.shape == (STATE_DIM,)
    assert vec.dtype == np.float32
    assert not np.any(np.isnan(vec))


def test_encoder_hunl_position_bits():
    """HUNL encoder sets exactly one position bit (dealer XOR BB)."""
    if not _HUNL_AVAILABLE:
        pytest.skip("universal_poker not available in this OpenSpiel build")
    from src.env.poker_env import PokerEnv
    env = PokerEnv(use_hunl=True)
    state = env.new_game()
    while state.is_chance_node():
        state.apply_action(state.chance_outcomes()[0][0])
    if state.is_terminal():
        pytest.skip("Game ended immediately")
    v0 = encode_state(state, player=0, use_hunl=True)
    v1 = encode_state(state, player=1, use_hunl=True)
    # player 0 is BTN/SB: is_BTN=1, is_BB=0
    assert v0[22] == 1.0 and v0[23] == 0.0
    # player 1 is BB: is_BTN=0, is_BB=1
    assert v1[22] == 0.0 and v1[23] == 1.0
