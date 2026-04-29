import pyspiel
from src.env.state_utils import N_ABSTRACT_ACTIONS
from src.env.hunl_game import HUNLGame


class PokerEnv:
    """Thin wrapper around OpenSpiel (Leduc) or pure-Python HUNL game."""

    def __init__(self, use_hunl: bool = False):
        if use_hunl:
            self.game = HUNLGame()
            self._validate_firstplayer()
        else:
            self.game = pyspiel.load_game("leduc_poker")
        self.use_hunl = use_hunl

    def new_game(self):
        return self.game.new_initial_state()

    def num_actions(self) -> int:
        return N_ABSTRACT_ACTIONS if self.use_hunl else self.game.num_distinct_actions()

    def info_state_size(self) -> int:
        return self.game.information_state_tensor_size()

    def _validate_firstplayer(self):
        state = self.game.new_initial_state()
        while state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            import numpy as np
            chosen = np.random.choice(actions, p=probs)
            state.apply_action(chosen)
        assert state.current_player() == 0, (
            f"Expected player 0 to act first pre-flop, "
            f"got player {state.current_player()}."
        )
