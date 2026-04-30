"""
Narrated hand demo — shows exactly what the AI does and why.

Runs 3 complete hands with full transparency:
  - Both players' hole cards shown face-up
  - AI decision probabilities printed at every action point
  - Opponent plays a simple "sensible" heuristic so hands are interesting
  - Final explanation of what the evaluation opponent is and what the numbers mean

Usage (from project root):
  conda run -n rl_env python scripts/demo_hand.py
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, r'C:\Users\rohan\AppData\Local\Temp\ospiel_manual_build2\python')

import numpy as np
import torch
from src.env.poker_env import PokerEnv
from src.models.advantage_net import AdvantageNet
from src.data.encoder import encode_state, STATE_DIM
from src.cfr.regret_matching import regret_matching_plus
from src.env.state_utils import ABSTRACT_ACTIONS, N_ABSTRACT_ACTIONS
from src.env.hunl_game import _RANKS, _SUITS, _cstr, FOLD, CHECK_CALL, ALL_IN

CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "checkpoints", "hunl_final.pt")


def card_name(card_int: int) -> str:
    rank = _RANKS[card_int // 4]
    suit = {'c': '♣', 'd': '♦', 'h': '♥', 's': '♠'}[_SUITS[card_int % 4]]
    return rank + suit


def _preflop_score(hole: list) -> float:
    """Preflop hand strength 0-1 based on rank pair + suitedness."""
    r0, r1 = hole[0] // 4, hole[1] // 4  # 0-12
    suited = (hole[0] % 4) == (hole[1] % 4)
    hi, lo = max(r0, r1), min(r0, r1)
    if hi == lo:  # pair
        return 0.5 + hi / 26.0   # pairs range 0.5-1.0
    score = (hi + lo) / 24.0 + (0.06 if suited else 0) - abs(hi - lo) * 0.02
    return max(0.0, min(0.99, score))


def hand_strength_pct(hole: list, board: list) -> float:
    """Rough hand strength percentile using only rank arithmetic."""
    if len(board) < 3:
        return _preflop_score(hole)

    # Post-flop: count made hands (pair / two-pair / trips / straight draw etc.)
    all_ranks = [c // 4 for c in hole + board]
    all_suits = [c % 4 for c in hole + board]
    from collections import Counter
    rank_counts = Counter(all_ranks)
    suit_counts = Counter(all_suits)

    best_count = max(rank_counts.values())
    pairs = sum(1 for v in rank_counts.values() if v >= 2)
    flush_draw = max(suit_counts.values()) >= 4
    # Rough ordering: trips/quads → two-pair → pair → high-card
    if best_count >= 3:
        return 0.85
    if pairs >= 2:
        return 0.70
    if best_count == 2:
        return 0.50
    if flush_draw:
        return 0.45
    # High-card: percentile from best hole-card rank
    hi_rank = max(hole[0] // 4, hole[1] // 4)
    return 0.15 + hi_rank / 30.0


def heuristic_action(state, player: int) -> int:
    """
    Sensible heuristic opponent:
      - top 40% hand → bet/raise (or call if facing raise)
      - middle 40% → check/call
      - bottom 20% → fold if facing any bet, check otherwise
    """
    legal = state.legal_actions()
    valid = [a for a in legal if a < N_ABSTRACT_ACTIONS]
    if not valid:
        return random.choice(legal)

    strength = hand_strength_pct(state._hole[player], state._board)
    facing_bet = state._bets[1 - player] > state._bets[player]

    if strength >= 0.60:
        # Strong hand: raise or call
        for a in [3, 2, ALL_IN, CHECK_CALL]:   # RAISE_ONE, RAISE_HALF, ALL_IN, CALL
            if a in valid:
                return a
    elif strength >= 0.25 or not facing_bet:
        # Medium or no bet to call: check/call
        if CHECK_CALL in valid:
            return CHECK_CALL
        return random.choice(valid)
    else:
        # Weak hand facing bet: fold
        if FOLD in valid:
            return FOLD
        return CHECK_CALL   # can't fold if no bet to face


def get_ai_probs(net, state, ai_player: int):
    legal = state.legal_actions()
    valid = [a for a in legal if a < N_ABSTRACT_ACTIONS]
    info = encode_state(state, ai_player, use_hunl=True)
    t = torch.FloatTensor(info).unsqueeze(0)
    net.eval()
    with torch.no_grad():
        adv = net(t).squeeze(0)
    mask = torch.zeros(N_ABSTRACT_ACTIONS, dtype=torch.bool)
    for a in valid:
        mask[a] = True
    return regret_matching_plus(adv, mask), valid


def play_narrated_hand(net, env, hand_num: int, ai_player: int = 0):
    AI = ai_player
    HUMAN = 1 - ai_player

    state = env.new_game()

    # Deal cards
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        state.apply_action(np.random.choice(actions, p=probs))

    print(f"\n{'═'*60}")
    print(f"  HAND #{hand_num}  —  AI is Player {AI} (SB/BTN)  |  Opponent is Player {HUMAN} (BB)")
    print(f"{'═'*60}")

    ai_cards   = '  '.join(card_name(c) for c in state._hole[AI])
    opp_cards  = '  '.join(card_name(c) for c in state._hole[HUMAN])
    print(f"  AI hole cards   : {ai_cards}")
    print(f"  Opp hole cards  : {opp_cards}")
    print(f"  Starting stacks : {state._stacks[AI]} / {state._stacks[HUMAN]} chips  (blinds 1/2)")

    streets = ['Pre-flop', 'Flop', 'Turn', 'River']
    last_street = -1
    step = 0

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            state.apply_action(np.random.choice(actions, p=probs))
            if state._street != last_street:
                last_street = state._street
                board_str = '  '.join(card_name(c) for c in state._board) if state._board else '(none)'
                print(f"\n  ── {streets[state._street]}  Board: {board_str} ──")
                print(f"     Pot={state._pot}  |  AI stack={state._stacks[AI]}  Opp stack={state._stacks[HUMAN]}")
            continue

        step += 1
        player = state.current_player()
        street = streets[state._street]
        pot = state._pot + state._bets[0] + state._bets[1]
        to_call = state._bets[1 - player] - state._bets[player]

        if player == AI:
            probs, valid = get_ai_probs(net, state, AI)
            prob_strs = ', '.join(
                f"{ABSTRACT_ACTIONS[a]}={probs[a]:.0%}" for a in range(N_ABSTRACT_ACTIONS) if probs[a] > 0.01
            )
            action = torch.multinomial(probs, 1).item()
            if action not in state.legal_actions():
                action = random.choice(valid)
            name = ABSTRACT_ACTIONS[action]

            print(f"\n  Step {step}. AI's turn  (pot={pot}, to_call={to_call})")
            print(f"     AI probs : {prob_strs}")
            print(f"     AI chose : {name}", end="")
            if action == CHECK_CALL and to_call > 0:
                print(f"  (calls {to_call} chips)")
            elif action == ALL_IN:
                print(f"  (shoves remaining {state._stacks[AI]} chips)")
            else:
                print()

        else:
            action = heuristic_action(state, HUMAN)
            strength = hand_strength_pct(state._hole[HUMAN], state._board)
            name = ABSTRACT_ACTIONS[action] if action < N_ABSTRACT_ACTIONS else str(action)
            print(f"\n  Step {step}. Opponent's turn  (pot={pot}, to_call={to_call}, hand strength={strength:.0%})")
            print(f"     Opp chose: {name}")

        state.apply_action(action)

        # Deal community cards after betting rounds close
        while state.is_chance_node() and not state.is_terminal():
            outcomes = state.chance_outcomes()
            actions_c, probs_c = zip(*outcomes)
            state.apply_action(np.random.choice(actions_c, p=probs_c))
            if state._street != last_street:
                last_street = state._street
                board_str = '  '.join(card_name(c) for c in state._board)
                print(f"\n  ── {streets[state._street]}  Board: {board_str} ──")
                print(f"     Pot={state._pot}  |  AI stack={state._stacks[AI]}  Opp stack={state._stacks[HUMAN]}")

    r = state.returns()
    ai_result   = r[AI]
    opp_result  = r[HUMAN]
    board_str = '  '.join(card_name(c) for c in state._board) if state._board else '(none)'
    print(f"\n  ── Showdown ──")
    print(f"     Final board : {board_str}")
    print(f"     AI  : {ai_cards}  →  {'+' if ai_result >= 0 else ''}{ai_result:.0f} chips")
    print(f"     Opp : {opp_cards}  →  {'+' if opp_result >= 0 else ''}{opp_result:.0f} chips")
    winner = "AI" if ai_result > 0 else ("Opponent" if opp_result > 0 else "Split pot")
    print(f"     Winner: {winner}")
    return ai_result


def main():
    print("Loading AI from", CHECKPOINT)
    env = PokerEnv(use_hunl=True)
    net = AdvantageNet(input_dim=STATE_DIM, n_actions=env.num_actions(), hidden_dim=256)
    net.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    net.eval()
    print("Loaded.\n")

    print("""
WHAT IS THIS AI PLAYING AGAINST?
─────────────────────────────────
In this demo the AI plays against a HEURISTIC opponent:
  • Top 60% hands  → bet/raise aggressively
  • Mid 25–60%     → check/call
  • Bottom 25%     → fold if facing a bet

In the training evaluation the AI played against RANDOM opponents
(uniform random over legal actions). That's why win-rate vs random
is so high (+269 bb/100) — random opponents call off chips with garbage.

The AI's strategy after only 100 SD-CFR iterations is "jam-or-fold":
  • Preflop: 72% shove, 28% call — almost never fold
  • This strategy crushes random opponents (they fold good equity, call with trash)
  • A real thinking opponent can easily exploit it (just fold to the shove and wait
    for a strong hand to snap-call)

To get a more realistic strategy you'd need 10,000+ iterations (hours of training).
The implementation is correct — it's a compute/scale issue, not a code bug.
""")

    results = []
    for i in range(3):
        result = play_narrated_hand(net, env, i + 1, ai_player=0)
        results.append(result)

    print(f"\n{'═'*60}")
    net_chips = sum(results)
    print(f"  3-hand summary: AI net chips = {net_chips:+.0f}  ({', '.join(f'{r:+.0f}' for r in results)})")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
