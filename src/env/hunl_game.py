"""
Pure-Python Heads-Up No-Limit Texas Hold'em.
Implements the OpenSpiel State interface used by traversal.py and encoder.py.

Player 0 = SB/BTN (acts first preflop, last postflop)
Player 1 = BB    (acts last preflop, first postflop)
"""
from __future__ import annotations
import random
from typing import List, Tuple

FOLD       = 0
CHECK_CALL = 1
RAISE_HALF = 2
RAISE_ONE  = 3
RAISE_TWO  = 4
ALL_IN     = 5

_RANKS = '23456789TJQKA'
_SUITS = 'cdhs'
N_ABSTRACT_ACTIONS = 6

# Number of board cards after each street completes
_BOARD_TARGET = {1: 3, 2: 4, 3: 5}


def _cstr(card: int) -> str:
    return _RANKS[card // 4] + _SUITS[card % 4]


class HUNLGame:
    """Drop-in replacement for pyspiel game object (HUNL only)."""

    def new_initial_state(self) -> HUNLState:
        return HUNLState()

    def num_distinct_actions(self) -> int:
        return N_ABSTRACT_ACTIONS

    def information_state_tensor_size(self) -> int:
        return 60  # STATE_DIM


class HUNLState:
    """
    HUNL game state with OpenSpiel-compatible interface.

    Chips model:
      _pot  = chips collected from previous streets
      _bets = chips committed this round (blinds count as bets pre-flop)
      _stacks = remaining chips (already deducted for posted blinds)
    """

    STARTING_STACK = 200
    SB = 1
    BB = 2

    def __init__(self):
        self._hole: List[List[int]] = [[], []]
        self._board: List[int] = []

        # Blinds deducted up front; bets track current-round commitments
        self._stacks: List[int] = [
            self.STARTING_STACK - self.SB,
            self.STARTING_STACK - self.BB,
        ]
        self._pot: int = 0
        self._bets: List[int] = [self.SB, self.BB]

        self._street: int = 0      # 0=pre-flop … 3=river
        self._phase: str = 'deal_hole'   # 'deal_hole' | 'deal_community' | 'betting'

        # Who acts next; who must close the round; last voluntary aggressor
        self._acting: int = 0      # pre-flop: BTN/SB acts first
        self._round_closer: int = 1  # pre-flop: BB closes
        self._last_agg: int = -1   # -1=no voluntary bet; -2=BB option active

        self._terminal: bool = False
        self._folded_by: int = -1

    # ── Public OpenSpiel interface ─────────────────────────────────────────────

    def is_terminal(self) -> bool:
        return self._terminal

    def is_chance_node(self) -> bool:
        return (not self._terminal) and (self._phase != 'betting')

    def current_player(self) -> int:
        if self._terminal or self.is_chance_node():
            return -1
        return self._acting

    def chance_outcomes(self) -> List[Tuple[int, float]]:
        used = set(self._hole[0] + self._hole[1] + self._board)
        avail = [c for c in range(52) if c not in used]
        p = 1.0 / len(avail)
        return [(c, p) for c in avail]

    def legal_actions(self) -> List[int]:
        if self._terminal or self.is_chance_node():
            return []
        cur = self._acting
        opp = 1 - cur
        to_call = max(0, self._bets[opp] - self._bets[cur])
        stack = self._stacks[cur]

        acts: List[int] = []
        if to_call > 0:
            acts.append(FOLD)
        acts.append(CHECK_CALL)

        # Raises: only when we have chips beyond the call and opp isn't all-in
        if stack > to_call and self._stacks[opp] > 0:
            rp = self._pot + self._bets[0] + self._bets[1] + to_call  # pot-after-call
            min_r = max(to_call, self.BB)

            def can(size: float) -> bool:
                return stack >= to_call + max(int(size), min_r)

            if can(rp * 0.5): acts.append(RAISE_HALF)
            if can(rp):       acts.append(RAISE_ONE)
            if can(rp * 2):   acts.append(RAISE_TWO)
            acts.append(ALL_IN)

        return acts

    def returns(self) -> List[float]:
        if not self._terminal:
            return [0.0, 0.0]
        return [
            float(self._stacks[0] - self.STARTING_STACK),
            float(self._stacks[1] - self.STARTING_STACK),
        ]

    def child(self, action: int) -> HUNLState:
        s = HUNLState.__new__(HUNLState)
        s._hole = [list(self._hole[0]), list(self._hole[1])]
        s._board = list(self._board)
        s._stacks = list(self._stacks)
        s._bets = list(self._bets)
        s._pot = self._pot
        s._street = self._street
        s._phase = self._phase
        s._acting = self._acting
        s._round_closer = self._round_closer
        s._last_agg = self._last_agg
        s._terminal = self._terminal
        s._folded_by = self._folded_by
        s.apply_action(action)
        return s

    def apply_action(self, action: int) -> None:
        if self._phase == 'deal_hole':
            self._deal_hole(action)
        elif self._phase == 'deal_community':
            self._deal_community(action)
        else:
            self._player_act(action)

    def information_state_string(self, player: int) -> str:
        private = ' '.join(_cstr(c) for c in self._hole[player])
        out = f'[Private: {private}]'
        if self._board:
            out += '[Community: ' + ' '.join(_cstr(c) for c in self._board) + ']'
        out += f'[Money: {self._stacks[0]} {self._stacks[1]}]'
        out += f'[Pot: {self._pot + self._bets[0] + self._bets[1]}]'
        return out

    # ── Chance-node actions ────────────────────────────────────────────────────

    def _deal_hole(self, card: int) -> None:
        if len(self._hole[0]) < 2:
            self._hole[0].append(card)
        else:
            self._hole[1].append(card)
        if len(self._hole[0]) + len(self._hole[1]) == 4:
            self._phase = 'betting'

    def _deal_community(self, card: int) -> None:
        self._board.append(card)
        if len(self._board) == _BOARD_TARGET[self._street]:
            self._phase = 'betting'
            self._bets = [0, 0]
            self._last_agg = -1
            self._acting = 1        # OOP (BB) acts first post-flop
            self._round_closer = 0  # IP (BTN) closes post-flop action

    # ── Player-decision actions ────────────────────────────────────────────────

    def _player_act(self, action: int) -> None:
        cur = self._acting
        opp = 1 - cur
        to_call = max(0, self._bets[opp] - self._bets[cur])
        stack = self._stacks[cur]

        if action == FOLD:
            self._pot += self._bets[0] + self._bets[1]
            self._stacks[opp] += self._pot
            self._pot = 0
            self._bets = [0, 0]
            self._folded_by = cur
            self._terminal = True
            return

        if action == CHECK_CALL:
            call_amt = min(to_call, stack)
            self._bets[cur] += call_amt
            self._stacks[cur] -= call_amt

            if to_call == 0:
                # Check: round ends when the round-closer checks with no open bet
                if cur == self._round_closer and self._last_agg in (-1, -2):
                    self._end_round()
                else:
                    self._acting = opp
            else:
                # Call
                if self._last_agg == -1:
                    # Calling blind pre-flop: give BB the option to raise
                    self._last_agg = -2
                    self._round_closer = 1
                    self._acting = 1
                else:
                    # Called a voluntary raise: close
                    if self._stacks[cur] == 0 or self._stacks[opp] == 0:
                        self._allin_runout()
                    else:
                        self._end_round()
            return

        # Raise (RAISE_HALF / RAISE_ONE / RAISE_TWO / ALL_IN)
        rp = self._pot + self._bets[0] + self._bets[1] + to_call
        min_r = max(to_call, self.BB)

        if action == RAISE_HALF:
            size = max(int(rp * 0.5), min_r)
        elif action == RAISE_ONE:
            size = max(int(rp), min_r)
        elif action == RAISE_TWO:
            size = max(int(rp * 2), min_r)
        else:  # ALL_IN
            size = stack

        add = int(min(to_call + size, stack))
        self._bets[cur] += add
        self._stacks[cur] -= add
        self._last_agg = cur
        self._round_closer = opp
        self._acting = opp

    # ── Round / showdown helpers ───────────────────────────────────────────────

    def _end_round(self) -> None:
        self._pot += self._bets[0] + self._bets[1]
        self._bets = [0, 0]
        if self._street == 3:
            self._terminal = True
            self._showdown()
        elif self._stacks[0] == 0 or self._stacks[1] == 0:
            self._allin_runout()
        else:
            self._street += 1
            self._phase = 'deal_community'
            self._last_agg = -1

    def _allin_runout(self) -> None:
        """One player is all-in after a call: deal remaining board, then showdown."""
        self._pot += self._bets[0] + self._bets[1]
        self._bets = [0, 0]
        used = set(self._hole[0] + self._hole[1] + self._board)
        rem = [c for c in range(52) if c not in used]
        random.shuffle(rem)
        self._board.extend(rem[: 5 - len(self._board)])
        self._terminal = True
        self._showdown()

    def _showdown(self) -> None:
        if len(self._board) < 5:
            used = set(self._hole[0] + self._hole[1] + self._board)
            rem = [c for c in range(52) if c not in used]
            random.shuffle(rem)
            self._board.extend(rem[: 5 - len(self._board)])

        try:
            from phevaluator import evaluate_cards

            def rank(p: int) -> int:
                cs = [_cstr(c) for c in (self._hole[p] + self._board)[:7]]
                return evaluate_cards(*cs)

            r0, r1 = rank(0), rank(1)
            if r0 < r1:       # lower = better in phevaluator
                self._stacks[0] += self._pot
            elif r1 < r0:
                self._stacks[1] += self._pot
            else:
                h = self._pot // 2
                self._stacks[0] += h
                self._stacks[1] += self._pot - h
        except Exception:
            h = self._pot // 2
            self._stacks[0] += h
            self._stacks[1] += self._pot - h
        self._pot = 0
