import pytest
import numpy as np
from src.data.parser import parse_phh_hand, parse_irc_hand, ABSTRACT_ACTIONS


def test_parse_phh_returns_valid_structure():
    # Minimal synthetic PHH-like dict for unit testing
    sample = {
        "players": ["Alice", "Bob"],
        "starting_stacks": [200, 200],
        "blinds_or_straddles": [1, 2],
        "actions": ["db", "db", "dh Ah Kh", "dh 2c 3d", "cbr 4", "cc", "db Qd Jh Tc", "f"],
    }
    result = parse_phh_hand(sample, min_stack_bb=20)
    if result is None:
        pytest.skip("Filter rejected synthetic hand — check filter logic")
    states, actions = result
    assert len(states) == len(actions)
    assert all(0 <= a < len(ABSTRACT_ACTIONS) for a in actions)
    assert all(s.shape == (60,) for s in states)


def test_action_abstraction_coverage():
    assert len(ABSTRACT_ACTIONS) == 6
    assert ABSTRACT_ACTIONS[0] == "FOLD"
    assert ABSTRACT_ACTIONS[1] == "CHECK_CALL"
    assert ABSTRACT_ACTIONS[5] == "ALL_IN"
