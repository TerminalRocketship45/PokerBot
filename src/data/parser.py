from typing import Optional, List, Tuple
import numpy as np
from src.data.encoder import encode_state, STATE_DIM

ABSTRACT_ACTIONS = ["FOLD", "CHECK_CALL", "RAISE_HALF", "RAISE_ONE", "RAISE_TWO", "ALL_IN"]
N_ACTIONS = len(ABSTRACT_ACTIONS)

FOLD = 0
CHECK_CALL = 1
RAISE_HALF = 2
RAISE_ONE = 3
RAISE_TWO = 4
ALL_IN = 5


def _bet_to_abstract(bet_size: float, pot: float, stack: float) -> int:
    if bet_size <= 0:
        return CHECK_CALL
    if stack <= 0 or bet_size >= stack * 0.95:
        return ALL_IN
    fraction = bet_size / pot if pot > 1e-6 else 1.0
    if fraction <= 0.65:
        return RAISE_HALF
    elif fraction <= 1.5:
        return RAISE_ONE
    else:
        return RAISE_TWO


def parse_phh_hand(
    hand,
    min_stack_bb: float = 20.0,
) -> Optional[Tuple[List[np.ndarray], List[int]]]:
    """
    Parse a PHH hand object (or dict) into (states, abstract_actions) pairs.
    Returns None if hand is filtered out.
    """
    try:
        if hasattr(hand, 'starting_stacks'):
            stacks = list(hand.starting_stacks)
            blinds = list(hand.blinds_or_straddles)
            actions_raw = list(hand.actions)
        else:
            stacks = hand["starting_stacks"]
            blinds = hand["blinds_or_straddles"]
            actions_raw = hand["actions"]

        if len(stacks) != 2:
            return None
        bb = max(blinds) if blinds else 2
        if min(stacks) < min_stack_bb * bb:
            return None

        states, abstract_actions = [], []
        pot = sum(blinds) if blinds else 3.0
        player_stacks = list(stacks)
        hole_cards = [[], []]
        community_cards = []
        street = 0

        for action_str in actions_raw:
            parts = action_str.strip().split()
            if not parts:
                continue
            code = parts[0]

            if code == "dh":
                player_idx = len([c for c in hole_cards if c])
                if len(parts) > 2:
                    hole_cards[min(player_idx, 1)] = parts[1:]
            elif code == "db":
                community_cards.extend(parts[1:])
                street = {0: 1, 3: 2, 4: 3}.get(len(community_cards) - len(parts[1:]), street)
            elif code in ("f", "cc", "cbr", "b"):
                current_player = len(states) % 2
                obs = {
                    "hole_cards": hole_cards[current_player],
                    "community_cards": community_cards,
                    "street": street,
                    "hero_stack": player_stacks[current_player],
                    "villain_stack": player_stacks[1 - current_player],
                    "pot": pot,
                    "to_call": 0.0,
                    "is_dealer": current_player == 0,
                    "starting_stack": stacks[current_player],
                }
                state_vec = _encode_obs_dict(obs)

                if code == "f":
                    abstract_actions.append(FOLD)
                elif code == "cc":
                    abstract_actions.append(CHECK_CALL)
                elif code in ("cbr", "b"):
                    bet = float(parts[1]) if len(parts) > 1 else pot
                    abstract_actions.append(_bet_to_abstract(bet, pot, player_stacks[current_player]))
                    pot += bet
                else:
                    abstract_actions.append(CHECK_CALL)

                states.append(state_vec)

        if len(states) == 0:
            return None
        return states, abstract_actions

    except Exception:
        return None


def _encode_obs_dict(obs: dict) -> np.ndarray:
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    starting = float(obs.get("starting_stack", 200.0))

    rank_map = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,'9':7,
                'T':8,'J':9,'Q':10,'K':11,'A':12}
    suit_map = {'c':0,'d':1,'h':2,'s':3}

    def parse_card(c):
        if len(c) >= 2 and c[0] in rank_map and c[1] in suit_map:
            return rank_map[c[0]] / 12.0, suit_map[c[1]] / 3.0
        return -1.0, -1.0

    for i, card in enumerate(obs.get("hole_cards", [])[:2]):
        r, s = parse_card(card)
        vec[i*2], vec[i*2+1] = r, s

    for i, card in enumerate(obs.get("community_cards", [])[:5]):
        r, s = parse_card(card)
        vec[4 + i*2], vec[4 + i*2+1] = r, s

    street = obs.get("street", 0)
    vec[14 + min(street, 3)] = 1.0

    vec[18] = obs.get("hero_stack", starting) / starting
    vec[19] = obs.get("villain_stack", starting) / starting
    vec[20] = obs.get("pot", 0.0) / starting
    vec[21] = obs.get("to_call", 0.0) / starting
    vec[22] = 1.0 if obs.get("is_dealer", False) else 0.0
    vec[23] = 0.0 if obs.get("is_dealer", False) else 1.0
    return vec


def parse_irc_hand(line: str) -> Optional[Tuple[List[np.ndarray], List[int]]]:
    """
    Minimal IRC parser stub. IRC format varies; implement after inspecting raw files.
    Returns None for non-NLHE hands.
    """
    if "nolimit" not in line.lower():
        return None
    return None
