import numpy as np

STATE_DIM = 60

RANK_MAP = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,'9':7,
            'T':8,'J':9,'Q':10,'K':11,'A':12}
SUIT_MAP = {'c':0,'d':1,'h':2,'s':3}


def encode_state(state, player: int, use_hunl: bool = False,
                 starting_stack: float = 200.0) -> np.ndarray:
    """
    Encode OpenSpiel state to float32[STATE_DIM=60].
    For Leduc (use_hunl=False): pads OpenSpiel's native info tensor to 60 floats.
    For HUNL (use_hunl=True): uses custom 60-dim layout.
    """
    if not use_hunl:
        return _encode_leduc(state, player)
    return _encode_hunl(state, player, starting_stack)


def _encode_leduc(state, player: int) -> np.ndarray:
    tensor = np.array(state.information_state_tensor(player), dtype=np.float32)
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    n = min(len(tensor), STATE_DIM)
    vec[:n] = tensor[:n]
    return vec


def _encode_hunl(state, player: int, starting_stack: float) -> np.ndarray:
    import re
    vec = np.full(STATE_DIM, 0.0, dtype=np.float32)
    info_str = state.information_state_string(player)

    # --- Hole cards [0:4] ---
    private_m = re.search(r'\[Private: ([^\]]+)\]', info_str)
    hole_cards = _parse_cards(private_m.group(1)) if private_m else []
    for i, (rank, suit) in enumerate(hole_cards[:2]):
        vec[i * 2] = rank / 12.0
        vec[i * 2 + 1] = suit / 3.0

    # --- Community cards [4:14] ---
    comm_m = re.search(r'\[Community: ([^\]]+)\]', info_str)
    comm_cards = _parse_cards(comm_m.group(1)) if comm_m else []
    for i, (rank, suit) in enumerate(comm_cards[:5]):
        vec[4 + i * 2] = rank / 12.0
        vec[4 + i * 2 + 1] = suit / 3.0

    # --- Street one-hot [14:18] ---
    n_comm = len(comm_cards)
    street = {0: 0, 3: 1, 4: 2, 5: 3}.get(n_comm, 0)
    vec[14 + street] = 1.0

    # --- Stacks [18:20] ---
    money_m = re.search(r'\[Money: ([\d.]+) ([\d.]+)\]', info_str)
    hero_stack, villain_stack = starting_stack, starting_stack
    if money_m:
        p0_stack = float(money_m.group(1))
        p1_stack = float(money_m.group(2))
        hero_stack = p0_stack if player == 0 else p1_stack
        villain_stack = p1_stack if player == 0 else p0_stack
    vec[18] = hero_stack / starting_stack
    vec[19] = villain_stack / starting_stack

    # --- Pot + to_call [20:22] ---
    pot_m = re.search(r'\[Pot: ([\d.]+)\]', info_str)
    pot = float(pot_m.group(1)) if pot_m else 0.0
    hero_contributed = starting_stack - hero_stack
    villain_contributed = starting_stack - villain_stack
    to_call = max(0.0, villain_contributed - hero_contributed)
    vec[20] = pot / starting_stack
    vec[21] = to_call / starting_stack

    # --- Position [22:24] ---
    vec[22] = 1.0 if player == 0 else 0.0  # is_BTN/SB
    vec[23] = 1.0 if player == 1 else 0.0  # is_BB

    # --- Action history [24:48]: tracked externally, left 0 here ---
    # traverse() fills these via obs dict; direct encode_state calls get zeros.

    # --- Hand strength [48:52] ---
    if hole_cards:
        strength = _hand_strength_bucket(hole_cards, comm_cards)
        vec[48] = strength  # preflop/overall bucket
        vec[49] = strength if comm_cards else 0.5  # postflop equity
        vec[50] = min(strength * 1.2, 1.0) if strength > 0.7 else strength * 0.8  # nut advantage proxy
        vec[51] = _board_texture(comm_cards)  # board texture

    # --- SPR + game context [52:56] ---
    eff_stack = min(hero_stack, villain_stack)
    spr = eff_stack / pot if pot > 1e-6 else 99.0
    pot_odds = to_call / (to_call + pot) if (to_call + pot) > 1e-6 else 0.0
    vec[52] = min(spr / 20.0, 1.0)
    vec[53] = pot_odds
    vec[54] = eff_stack / starting_stack
    vec[55] = pot / (2.0 * starting_stack)

    return vec


def _parse_cards(card_str: str):
    cards = []
    for c in card_str.strip().split():
        if len(c) == 2 and c[0] in RANK_MAP and c[1] in SUIT_MAP:
            cards.append((RANK_MAP[c[0]], SUIT_MAP[c[1]]))
    return cards


def _hand_strength_bucket(hole_cards, comm_cards) -> float:
    try:
        from phevaluator import evaluate_cards
        if len(comm_cards) < 3:
            return _preflop_bucket(hole_cards)
        ranks = '23456789TJQKA'
        def to_str(r, s): return ranks[r] + 'cdhs'[s]
        all_cards = [to_str(r, s) for r, s in hole_cards + comm_cards]
        score = evaluate_cards(*all_cards[:7])
        return 1.0 - (score / 7462.0)
    except Exception:
        return 0.5


def _preflop_bucket(hole_cards) -> float:
    if len(hole_cards) < 2:
        return 0.5
    r0, r1 = hole_cards[0][0], hole_cards[1][0]
    suited = hole_cards[0][1] == hole_cards[1][1]
    strength = (r0 + r1) / 24.0 + (0.1 if suited else 0.0) + (0.15 if r0 == r1 else 0.0)
    return min(strength, 1.0)


def _board_texture(comm_cards) -> float:
    """Board texture score: higher = wetter (more draws possible). 0 if no board."""
    if len(comm_cards) < 3:
        return 0.0
    suits = [s for _, s in comm_cards]
    ranks = sorted([r for r, _ in comm_cards])
    suit_score = 1.0 if len(set(suits)) == 1 else (0.5 if len(set(suits)) == 2 else 0.0)
    gaps = sum(1 for i in range(len(ranks) - 1) if ranks[i+1] - ranks[i] <= 2)
    conn_score = gaps / max(len(ranks) - 1, 1)
    return (suit_score + conn_score) / 2.0
