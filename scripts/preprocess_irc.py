"""
IRC Poker Database preprocessor for HUNL Behavioral Cloning.

Processes the holdem .tgz archives in data/IRCdata/, extracts 2-player hands,
and outputs a parquet file with (state_vector, action) pairs ready for BC training.

IRC format reference:
  hdb: timestamp game# hand# num_players preflop_summary flop_summary turn_summary river_summary board_cards...
  pdb: player timestamp game# seat preflop_actions flop_actions turn_actions river_actions bankroll profit pot [hole_card1 hole_card2]

Action characters:
  B = blind post (forced — skip as training label)
  b = bet        → map to RAISE_ONE (3)
  c = call       → map to CHECK_CALL (1)
  k = check      → map to CHECK_CALL (1)
  r = raise      → map to RAISE_TWO (4)  (limit raise = 2x bet, close to 2x pot raise)
  f = fold       → map to FOLD (0)
  Q = quit/fold  → map to FOLD (0)
  A = all-in     → map to ALL_IN (5)

Usage:
  python scripts/preprocess_irc.py
"""
import os
import sys
import glob
import tarfile
import tempfile
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, r'C:\Users\rohan\AppData\Local\Temp\ospiel_manual_build2\python')

IRC_DIR   = "data/IRCdata"
OUT_PATH  = "data/processed/irc_hunl.parquet"
STATE_DIM = 60

# Map IRC action characters to our 6-action space
# FOLD=0  CHECK_CALL=1  RAISE_HALF=2  RAISE_ONE=3  RAISE_TWO=4  ALL_IN=5
IRC_ACTION_MAP = {
    'f': 0, 'Q': 0,          # fold
    'c': 1, 'k': 1,          # check/call
    'b': 3,                   # bet → raise 1x pot
    'r': 4,                   # raise (limit) → raise 2x
    'A': 5,                   # all-in
}
SKIP_CHARS = set('B- ')      # blind posts and no-action markers

RANK_MAP = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,'9':7,'T':8,'J':9,'Q':10,'K':11,'A':12}
SUIT_MAP = {'c':0,'d':1,'h':2,'s':3}

SB_CHIPS = 10   # IRC limit game small blind (estimated from pot sizes)
BB_CHIPS = 20   # big blind


def parse_card(s):
    """'Ah' → (12, 2)  or None"""
    if len(s) == 2 and s[0].upper() in RANK_MAP and s[1].lower() in SUIT_MAP:
        return (RANK_MAP[s[0].upper()], SUIT_MAP[s[1].lower()])
    return None


def parse_cards(token_list):
    cards = []
    for t in token_list:
        c = parse_card(t)
        if c:
            cards.append(c)
    return cards


def build_state_vec(hole_cards, board_cards, hero_stack, villain_stack, pot, is_sb):
    """
    Build a 60-dim state vector matching the encoder's HUNL layout.
    All stacks/pot normalised by 1000 (our new default starting_stack).
    """
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    starting_stack = 1000.0

    # [0:4] hole cards
    for i, (rank, suit) in enumerate(hole_cards[:2]):
        vec[i * 2]     = rank / 12.0
        vec[i * 2 + 1] = suit / 3.0

    # [4:14] community cards
    for i, (rank, suit) in enumerate(board_cards[:5]):
        vec[4 + i * 2]     = rank / 12.0
        vec[4 + i * 2 + 1] = suit / 3.0

    # [14:18] street one-hot
    n_comm = len(board_cards)
    street = {0: 0, 3: 1, 4: 2, 5: 3}.get(n_comm, 0)
    vec[14 + street] = 1.0

    # [18:20] stacks
    vec[18] = hero_stack    / starting_stack
    vec[19] = villain_stack / starting_stack

    # [20:22] pot + to_call
    vec[20] = pot / starting_stack
    hero_committed    = starting_stack - hero_stack
    villain_committed = starting_stack - villain_stack
    to_call = max(0.0, villain_committed - hero_committed)
    vec[21] = to_call / starting_stack

    # [22:24] position
    vec[22] = 1.0 if is_sb else 0.0
    vec[23] = 0.0 if is_sb else 1.0

    # [48:52] hand strength bucket (preflop only for now)
    if hole_cards:
        r0, r1 = hole_cards[0][0], hole_cards[1][0]
        suited  = hole_cards[0][1] == hole_cards[1][1]
        pair_bonus = 0.15 if r0 == r1 else 0.0
        suit_bonus = 0.05 if suited else 0.0
        strength = (r0 + r1) / 24.0 + pair_bonus + suit_bonus
        strength = min(strength, 1.0)
        vec[48] = strength
        vec[49] = strength
        vec[50] = min(strength * 1.2, 1.0) if strength > 0.7 else strength * 0.8
        vec[51] = 0.0  # no board texture preflop

    # [52:56] SPR + context
    eff_stack = min(hero_stack, villain_stack)
    spr = eff_stack / pot if pot > 1e-6 else 99.0
    pot_odds = to_call / (to_call + pot) if (to_call + pot) > 1e-6 else 0.0
    vec[52] = min(spr / 20.0, 1.0)
    vec[53] = pot_odds
    vec[54] = eff_stack / starting_stack
    vec[55] = pot / (2.0 * starting_stack)

    return vec


def parse_action_sequence(action_str):
    """
    Parse IRC action string into list of (action_char, is_decision) tuples.
    Skips blind posts; returns voluntary decision characters only.
    """
    decisions = []
    for ch in action_str.strip():
        if ch in SKIP_CHARS:
            continue
        if ch == 'B':  # blind — forced, skip
            continue
        if ch in IRC_ACTION_MAP:
            decisions.append(ch)
    return decisions


def parse_hdb(hdb_path):
    """
    Returns dict: timestamp (int) → {n_players, board_cards, pot_preflop, pot_flop, pot_turn, pot_river}
    Only 2-player hands.
    """
    hands = {}
    try:
        with open(hdb_path, 'r', errors='ignore') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 9:
                    continue
                try:
                    ts       = int(parts[0])
                    n_pl     = int(parts[3])
                    if n_pl != 2:
                        continue

                    def pot(s):
                        # "2/20" → 20
                        if '/' in s:
                            return int(s.split('/')[1])
                        return 0

                    pot_pre   = pot(parts[4])
                    pot_flop  = pot(parts[5])
                    pot_turn  = pot(parts[6])
                    pot_river = pot(parts[7])

                    board_cards = parse_cards(parts[8:])
                    hands[ts] = {
                        'n_players':  n_pl,
                        'board':      board_cards,
                        'pot_pre':    pot_pre,
                        'pot_flop':   pot_flop,
                        'pot_turn':   pot_turn,
                        'pot_river':  pot_river,
                    }
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    return hands


def parse_pdb_file(pdb_path):
    """
    Returns list of player records:
    {timestamp, seat, preflop, flop, turn, river, bankroll, hole_cards}
    """
    records = []
    try:
        with open(pdb_path, 'r', errors='ignore') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 10:
                    continue
                try:
                    ts      = int(parts[1])
                    seat    = int(parts[3])
                    preflop = parts[4]
                    flop    = parts[5]
                    turn    = parts[6]
                    river   = parts[7]
                    bankroll = int(parts[8])

                    # Hole cards at end of line (only present at showdown)
                    hole_cards = parse_cards(parts[10:])

                    records.append({
                        'timestamp':  ts,
                        'seat':       seat,
                        'preflop':    preflop,
                        'flop':       flop,
                        'turn':       turn,
                        'river':      river,
                        'bankroll':   bankroll,
                        'hole_cards': hole_cards,
                    })
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    return records


def extract_training_pairs(hdb_hands, pdb_records):
    """
    For each 2-player hand where both players have known hole cards,
    extract (state_vector, action) pairs for each decision point.
    """
    # Index pdb by timestamp
    by_ts = {}
    for rec in pdb_records:
        ts = rec['timestamp']
        if ts not in by_ts:
            by_ts[ts] = []
        by_ts[ts].append(rec)

    pairs = []

    for ts, hand in hdb_hands.items():
        players = by_ts.get(ts, [])
        if len(players) != 2:
            continue

        # Sort by seat: seat 1 = SB, seat 2 = BB
        players.sort(key=lambda p: p['seat'])
        sb, bb = players[0], players[1]

        # Only process if at least one player has known hole cards
        # (limit holdem: cards only shown at showdown)
        for hero in (sb, bb):
            if not hero['hole_cards'] or len(hero['hole_cards']) < 2:
                continue

            villain = bb if hero is sb else sb
            is_sb   = (hero is sb)

            # --- Pre-flop decisions ---
            # After blinds are posted: SB is first to act, pot = SB+BB = 30
            preflop_actions = parse_action_sequence(hero['preflop'])

            # Estimate stacks: bankroll already deducted blinds?
            # IRC bankroll = stack BEFORE the hand. Deduct blind posted.
            hero_blind = SB_CHIPS if is_sb else BB_CHIPS
            vill_blind = BB_CHIPS if is_sb else SB_CHIPS

            hero_stack = max(hero['bankroll'] - hero_blind, 0)
            vill_stack = max(villain['bankroll'] - vill_blind, 0)
            pot        = SB_CHIPS + BB_CHIPS   # 30 at start of preflop action

            # Normalize high stacks to our range (cap at 1000)
            hero_stack = min(hero_stack, 1000)
            vill_stack = min(vill_stack, 1000)

            board = []  # preflop: no community cards

            # The action sequence for preflop after blinds
            # In HU: SB acts first, may call/raise/fold
            for act_ch in preflop_actions:
                if act_ch not in IRC_ACTION_MAP:
                    continue
                action_id = IRC_ACTION_MAP[act_ch]

                vec = build_state_vec(
                    hero['hole_cards'][:2],
                    board,
                    hero_stack, vill_stack, pot, is_sb
                )
                pairs.append((vec, action_id))

                # Update pot/stacks after action (approximate)
                if act_ch in ('c',):          # call
                    to_call = max(0, vill_blind - hero_blind)
                    hero_stack -= to_call
                    pot += to_call
                elif act_ch in ('b', 'r'):    # bet/raise: add approximate bet
                    bet_size = BB_CHIPS  # limit = 1 BB preflop
                    hero_stack -= bet_size
                    pot += bet_size
                elif act_ch in ('f', 'Q'):    # fold — stop
                    break

            # --- Post-flop (flop, turn, river) ---
            board_flop  = hand['board'][:3]
            board_turn  = hand['board'][:4]
            board_river = hand['board'][:5]

            for street_name, actions_str, board_cards, pot_size in [
                ('flop',  hero['flop'],  board_flop,  hand['pot_flop']),
                ('turn',  hero['turn'],  board_turn,  hand['pot_turn']),
                ('river', hero['river'], board_river, hand['pot_river']),
            ]:
                if not board_cards or pot_size == 0:
                    break

                street_actions = parse_action_sequence(actions_str)
                if not street_actions:
                    continue

                # Post-flop: approximate hero/villain stacks from pot progression
                # Use starting bankroll minus estimated contributions
                for act_ch in street_actions:
                    if act_ch not in IRC_ACTION_MAP:
                        continue
                    action_id = IRC_ACTION_MAP[act_ch]

                    # Rough stack estimate: bankroll - half pot (50/50 split)
                    h_stk = max(hero['bankroll'] - pot_size // 2, 50)
                    v_stk = max(villain['bankroll'] - pot_size // 2, 50)
                    h_stk = min(h_stk, 1000)
                    v_stk = min(v_stk, 1000)

                    vec = build_state_vec(
                        hero['hole_cards'][:2],
                        board_cards,
                        h_stk, v_stk, float(pot_size), is_sb
                    )
                    pairs.append((vec, action_id))

                    if act_ch in ('f', 'Q'):
                        break

    return pairs


def process_archive(tgz_path, tmpdir):
    """Extract one tgz, parse it, return (state, action) pairs."""
    try:
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(tmpdir)
    except Exception:
        return []

    # Find hdb and pdb directory
    hdb_path = None
    pdb_dir  = None
    for root, dirs, files in os.walk(tmpdir):
        if 'hdb' in files:
            hdb_path = os.path.join(root, 'hdb')
        if 'pdb' in dirs:
            pdb_dir = os.path.join(root, 'pdb')

    if not hdb_path or not pdb_dir:
        return []

    hdb_hands = parse_hdb(hdb_path)
    if not hdb_hands:
        return []

    # Load all pdb files
    pdb_records = []
    for pdb_file in glob.glob(os.path.join(pdb_dir, 'pdb.*')):
        pdb_records.extend(parse_pdb_file(pdb_file))

    return extract_training_pairs(hdb_hands, pdb_records)


def main():
    os.makedirs("data/processed", exist_ok=True)

    tgz_files = sorted(
        glob.glob(os.path.join(IRC_DIR, "holdem.*.tgz")) +
        glob.glob(os.path.join(IRC_DIR, "holdempot.*.tgz"))
    )
    print(f"Found {len(tgz_files)} holdem/holdempot archive files in {IRC_DIR}")

    if not tgz_files:
        print("No files found. Check that IRC data is in data/IRCdata/")
        return

    all_states  = []
    all_actions = []
    n_processed = 0
    n_pairs     = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for tgz in tqdm(tgz_files, desc="Processing IRC archives"):
            pairs = process_archive(tgz, tmpdir)
            for vec, act in pairs:
                all_states.append(vec)
                all_actions.append(act)
            n_pairs     += len(pairs)
            n_processed += 1

            # Clean tmp dir between archives to avoid leftover files
            for item in os.listdir(tmpdir):
                item_path = os.path.join(tmpdir, item)
                try:
                    import shutil
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception:
                    pass

            if n_processed % 20 == 0:
                print(f"  Processed {n_processed}/{len(tgz_files)} archives | "
                      f"pairs so far: {n_pairs:,}")

    if not all_states:
        print("No training pairs extracted. Check IRC data format.")
        return

    print(f"\nTotal (state, action) pairs: {n_pairs:,}")

    # Build dataframe
    states_arr = np.stack(all_states)
    df = pd.DataFrame({
        'state':  list(states_arr),
        'action': all_actions,
    })

    # Show action distribution
    from collections import Counter
    action_names = ['FOLD', 'CHECK_CALL', 'RAISE_HALF', 'RAISE_ONE', 'RAISE_TWO', 'ALL_IN']
    dist = Counter(all_actions)
    print("\nAction distribution:")
    for i, name in enumerate(action_names):
        cnt = dist.get(i, 0)
        print(f"  {name:15s}: {cnt:8,}  ({100*cnt/n_pairs:.1f}%)")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
