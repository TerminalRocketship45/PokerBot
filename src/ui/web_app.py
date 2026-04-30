"""
Web-based poker UI. Serves a poker table at http://localhost:5000
Usage: conda run -n rl_env python src/ui/web_app.py --checkpoint checkpoints/hunl_final.pt
"""
import sys, os, argparse, random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, r'C:\Users\rohan\AppData\Local\Temp\ospiel_manual_build2\python')

from flask import Flask, jsonify, request
from src.env.poker_env import PokerEnv
from src.env.hunl_game import _RANKS, _SUITS, _cstr
from src.models.advantage_net import AdvantageNet
from src.data.encoder import encode_state, STATE_DIM
from src.cfr.regret_matching import regret_matching_plus
from src.env.state_utils import ABSTRACT_ACTIONS, N_ABSTRACT_ACTIONS
from src.env.hunl_game import FOLD as ACT_FOLD, CHECK_CALL as ACT_CALL, ALL_IN as ACT_ALLIN

app = Flask(__name__)

# ── Globals ────────────────────────────────────────────────────────────────────
_env: PokerEnv = None
_net: AdvantageNet = None
_policy_net = None   # PolicyNet loaded from NFSP checkpoint; None for legacy CFR
_state = None
_ai_player = 0
_human_player = 1
_hand_num = 0
_scores = {0: 0, 1: 0}   # cumulative chip delta from starting stack
_last_ai_action = ""
_last_ai_action_detail = ""   # e.g. "ALL_IN (196 chips)"
_last_ai_probs: list = []     # [{name, pct}] for current state
_showdown = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _card_dict(card_int: int) -> dict:
    rank = _RANKS[card_int // 4]
    suit = _SUITS[card_int % 4]
    suit_sym = {'c': '♣', 'd': '♦', 'h': '♥', 's': '♠'}[suit]
    color = 'red' if suit in ('d', 'h') else 'black'
    return {'rank': rank, 'suit': suit_sym, 'color': color, 'str': rank + suit_sym}


def _deal_chance():
    """Advance through all pending chance nodes (dealing cards)."""
    global _state
    while not _state.is_terminal() and _state.is_chance_node():
        outcomes = _state.chance_outcomes()
        actions, probs = zip(*outcomes)
        _state.apply_action(np.random.choice(actions, p=probs))


def _action_label(action_id: int, state, player: int) -> str:
    """Human-readable label including chip amounts."""
    if action_id == ACT_FOLD:
        return '✗ Fold'
    if action_id == ACT_CALL:
        to_call = state._bets[1 - player] - state._bets[player]
        return '✔ Check' if to_call == 0 else f'✔ Call {to_call}'
    if action_id == ACT_ALLIN:
        return f'⚡ All-In ({state._stacks[player]})'
    # raises
    pot = state._pot + state._bets[0] + state._bets[1]
    labels = {2: f'↑ Raise ½ Pot (~{int(pot*0.5)})', 3: f'↑ Raise 1× Pot (~{pot})',
              4: f'↑ Raise 2× Pot (~{int(pot*2)})'}
    return labels.get(action_id, ABSTRACT_ACTIONS[action_id])


def _ai_act():
    """AI takes its turn — supports both legacy CFR (AdvantageNet) and NFSP (PolicyNet) checkpoints."""
    global _state, _last_ai_action, _last_ai_action_detail, _last_ai_probs
    if _state.is_terminal() or _state.is_chance_node():
        return
    if _state.current_player() != _ai_player:
        return
    legal = _state.legal_actions()
    valid = [a for a in legal if a < N_ABSTRACT_ACTIONS]
    info = encode_state(_state, _ai_player, use_hunl=True)
    t = torch.FloatTensor(info).unsqueeze(0)

    if _policy_net is not None:
        _policy_net.eval()
        with torch.no_grad():
            probs_full = _policy_net(t).squeeze(0)
        legal_p = {a: probs_full[a].item() for a in valid}
        total = sum(legal_p.values())
        if total < 1e-9:
            action = random.choice(valid)
            probs = torch.zeros(N_ABSTRACT_ACTIONS)
            probs[action] = 1.0
        else:
            norm = {a: p / total for a, p in legal_p.items()}
            actions_list = list(norm.keys())
            w = [norm[a] for a in actions_list]
            action = int(np.random.choice(actions_list, p=w))
            probs = probs_full
    else:
        _net.eval()
        with torch.no_grad():
            adv = _net(t).squeeze(0)
        mask = torch.zeros(N_ABSTRACT_ACTIONS, dtype=torch.bool)
        for a in valid:
            mask[a] = True
        probs = regret_matching_plus(adv, mask)
        action = torch.multinomial(probs, 1).item()
        if action not in legal:
            action = random.choice(valid)

    _last_ai_probs = [
        {'name': ABSTRACT_ACTIONS[a], 'pct': round(float(probs[a]) * 100)}
        for a in range(N_ABSTRACT_ACTIONS) if probs[a] > 0.005
    ]
    _last_ai_action = ABSTRACT_ACTIONS[action] if action < N_ABSTRACT_ACTIONS else str(action)
    _last_ai_action_detail = _action_label(action, _state, _ai_player)
    _state.apply_action(action)


def _advance():
    """Deal any pending cards, then let AI act if it's AI's turn. Repeat."""
    _deal_chance()
    while not _state.is_terminal() and _state.current_player() == _ai_player:
        _ai_act()
        _deal_chance()


def _build_state_json(reveal_ai: bool = False) -> dict:
    s = _state
    human_hole = [_card_dict(c) for c in s._hole[_human_player]]
    ai_hole_visible = [_card_dict(c) for c in s._hole[_ai_player]] if reveal_ai else \
                      [{'rank': '?', 'suit': '?', 'color': 'black', 'str': '??'}] * len(s._hole[_ai_player])
    board = [_card_dict(c) for c in s._board]

    terminal = s.is_terminal()
    legal = [] if terminal or s.is_chance_node() else s.legal_actions()
    valid = [a for a in legal if a < N_ABSTRACT_ACTIONS]

    result_msg = ""
    if terminal:
        r = s.returns()
        delta = r[_human_player]
        result_msg = f"You {'WIN' if delta > 0 else 'LOSE'} {abs(delta):.0f} chips"

    actions_with_labels = [
        {'id': a, 'name': ABSTRACT_ACTIONS[a], 'label': _action_label(a, s, _human_player)}
        for a in valid
    ]

    return {
        'hand_num': _hand_num,
        'human_score': _scores[_human_player],
        'ai_score': _scores[_ai_player],
        'human_hole': human_hole,
        'ai_hole': ai_hole_visible,
        'board': board,
        'pot': s._pot + s._bets[0] + s._bets[1],
        'human_stack': s._stacks[_human_player],
        'ai_stack': s._stacks[_ai_player],
        'human_bet': s._bets[_human_player],
        'ai_bet': s._bets[_ai_player],
        'to_call': max(0, s._bets[_ai_player] - s._bets[_human_player]),
        'street': ['Preflop', 'Flop', 'Turn', 'River'][s._street],
        'legal_actions': actions_with_labels,
        'your_turn': not terminal and s.current_player() == _human_player,
        'terminal': terminal,
        'result_msg': result_msg,
        'last_ai_action': _last_ai_action,
        'last_ai_action_detail': _last_ai_action_detail,
        'ai_probs': _last_ai_probs,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return HTML_TEMPLATE


@app.route('/api/new_game', methods=['POST'])
def new_game():
    global _state, _hand_num, _last_ai_action, _last_ai_action_detail, _last_ai_probs, _showdown
    _hand_num += 1
    _last_ai_action = ""
    _last_ai_action_detail = ""
    _last_ai_probs = []
    _showdown = False
    _state = _env.new_game()
    _advance()
    return jsonify(_build_state_json())


@app.route('/api/state', methods=['GET'])
def get_state():
    if _state is None:
        return jsonify({'error': 'no game in progress'}), 400
    reveal = _state.is_terminal()
    return jsonify(_build_state_json(reveal_ai=reveal))


@app.route('/api/action', methods=['POST'])
def take_action():
    global _state, _showdown
    if _state is None or _state.is_terminal():
        return jsonify({'error': 'no active game'}), 400
    data = request.get_json()
    action = int(data.get('action', -1))
    legal = _state.legal_actions()
    if action not in legal:
        return jsonify({'error': f'illegal action {action}'}), 400

    _state.apply_action(action)
    _advance()

    terminal = _state.is_terminal()
    if terminal:
        _showdown = True
        r = _state.returns()
        _scores[0] += r[0]
        _scores[1] += r[1]

    return jsonify(_build_state_json(reveal_ai=terminal))


# ── HTML ───────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HUNL Poker Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1a472a;
    font-family: 'Segoe UI', sans-serif;
    color: #fff;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
  }
  h1 { font-size: 1.4em; letter-spacing: 2px; margin-bottom: 12px; color: #ffd700; }

  /* Table */
  #table {
    background: radial-gradient(ellipse at center, #2d6a3f 60%, #1a3d24 100%);
    border: 8px solid #8B4513;
    border-radius: 120px;
    width: 700px;
    min-height: 380px;
    position: relative;
    padding: 30px 60px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 0 40px rgba(0,0,0,0.7), inset 0 0 60px rgba(0,0,0,0.2);
  }

  /* Cards */
  .hand { display: flex; gap: 8px; justify-content: center; margin: 6px 0; }
  .card {
    width: 56px; height: 84px;
    border-radius: 8px;
    border: 2px solid #ccc;
    background: #fff;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: bold;
    box-shadow: 2px 3px 6px rgba(0,0,0,0.4);
    user-select: none;
    position: relative;
  }
  .card .rank { font-size: 20px; line-height: 1; }
  .card .suit { font-size: 16px; line-height: 1; }
  .card.red  { color: #c0392b; }
  .card.black { color: #111; }
  .card.face-down {
    background: repeating-linear-gradient(
      45deg, #003399, #003399 10px, #0044cc 10px, #0044cc 20px
    );
    border-color: #002277;
  }
  .card.empty {
    background: rgba(255,255,255,0.06);
    border: 2px dashed rgba(255,255,255,0.2);
  }

  /* Player zones */
  .player-zone {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
  }
  .player-info { text-align: center; min-width: 120px; }
  .player-info .name { font-size: 0.85em; font-weight: bold; color: #ffd700; margin-bottom: 4px; }
  .player-info .stack { font-size: 1.1em; }
  .player-info .bet { font-size: 0.8em; color: #aef; margin-top: 2px; }

  /* Community + pot */
  #community-area {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    margin: 10px 0;
  }
  #pot-label {
    font-size: 1em;
    color: #ffd700;
    background: rgba(0,0,0,0.35);
    padding: 4px 16px;
    border-radius: 20px;
  }
  #street-label { font-size: 0.8em; color: #9df; margin-top: 2px; }

  /* AI thinking panel */
  #ai-thinking {
    background: rgba(0,0,0,0.45);
    border: 1px solid rgba(255,215,0,0.3);
    border-radius: 10px;
    padding: 10px 18px;
    width: 700px;
    margin-top: 10px;
    font-size: 0.82em;
    color: #ddd;
  }
  #ai-thinking .title { color: #ffd700; font-weight: bold; margin-bottom: 6px; font-size: 0.9em; }
  .prob-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
  .prob-label { width: 120px; color: #aaa; }
  .prob-bar-wrap { flex: 1; background: rgba(255,255,255,0.1); border-radius: 4px; height: 12px; }
  .prob-bar { height: 12px; border-radius: 4px; background: #ffd700; }
  .prob-pct { width: 36px; text-align: right; color: #fff; font-weight: bold; }

  /* Situation hint */
  #situation {
    font-size: 0.82em;
    color: #9df;
    text-align: center;
    margin-top: 4px;
    min-height: 18px;
  }

  /* Action log */
  #ai-action {
    font-size: 0.85em;
    color: #ff9;
    background: rgba(0,0,0,0.3);
    padding: 3px 10px;
    border-radius: 8px;
    min-height: 22px;
    text-align: center;
    margin-top: 4px;
  }

  /* Buttons */
  #controls {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    margin-top: 16px;
    width: 700px;
  }
  #action-buttons {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: center;
  }
  .action-btn {
    padding: 10px 20px;
    border: none;
    border-radius: 8px;
    font-size: 0.95em;
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.1s, box-shadow 0.1s;
    box-shadow: 0 3px 6px rgba(0,0,0,0.3);
  }
  .action-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 10px rgba(0,0,0,0.4); }
  .action-btn:active { transform: translateY(0); }
  .action-btn.fold    { background: #c0392b; color: #fff; }
  .action-btn.check   { background: #27ae60; color: #fff; }
  .action-btn.raise   { background: #e67e22; color: #fff; }
  .action-btn.allin   { background: #8e44ad; color: #fff; }
  #new-hand-btn {
    padding: 10px 30px;
    background: #2980b9;
    color: #fff;
    border: none;
    border-radius: 8px;
    font-size: 1em;
    font-weight: bold;
    cursor: pointer;
    box-shadow: 0 3px 6px rgba(0,0,0,0.3);
  }
  #new-hand-btn:hover { background: #3498db; }

  /* Result overlay */
  #result-banner {
    display: none;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0,0,0,0.88);
    border: 3px solid #ffd700;
    border-radius: 16px;
    padding: 30px 50px;
    text-align: center;
    z-index: 100;
    font-size: 1.5em;
    color: #ffd700;
    box-shadow: 0 0 50px rgba(0,0,0,0.9);
  }

  /* Score bar */
  #scoreboard {
    display: flex;
    gap: 40px;
    margin-bottom: 10px;
    font-size: 0.9em;
    background: rgba(0,0,0,0.3);
    padding: 6px 24px;
    border-radius: 20px;
  }
  .score-item { text-align: center; }
  .score-item .label { color: #aaa; font-size: 0.8em; }
  .score-item .value { font-weight: bold; font-size: 1.1em; }
  .score-item .value.positive { color: #2ecc71; }
  .score-item .value.negative { color: #e74c3c; }
</style>
</head>
<body>

<h1>♠ HUNL Poker Agent ♥</h1>

<div id="scoreboard">
  <div class="score-item">
    <div class="label">Hand</div>
    <div class="value" id="hand-num">0</div>
  </div>
  <div class="score-item">
    <div class="label">Your Score</div>
    <div class="value" id="human-score">0</div>
  </div>
  <div class="score-item">
    <div class="label">AI Score</div>
    <div class="value" id="ai-score">0</div>
  </div>
</div>

<div id="table">
  <!-- AI zone (top) -->
  <div class="player-zone">
    <div class="player-info">
      <div class="name">🤖 AI (BTN/SB)</div>
      <div class="stack" id="ai-stack">200</div>
      <div class="bet" id="ai-bet"></div>
    </div>
    <div class="hand" id="ai-hand">
      <div class="card face-down"></div>
      <div class="card face-down"></div>
    </div>
  </div>

  <!-- Community + pot -->
  <div id="community-area">
    <div id="ai-action">Waiting...</div>
    <div id="pot-label">Pot: 0</div>
    <div class="hand" id="community-cards">
      <div class="card empty"></div>
      <div class="card empty"></div>
      <div class="card empty"></div>
      <div class="card empty"></div>
      <div class="card empty"></div>
    </div>
    <div id="street-label">Preflop</div>
  </div>

  <!-- Human zone (bottom) -->
  <div class="player-zone">
    <div class="player-info">
      <div class="name">👤 You (BB)</div>
      <div class="stack" id="human-stack">200</div>
      <div class="bet" id="human-bet"></div>
    </div>
    <div class="hand" id="human-hand">
      <div class="card empty"></div>
      <div class="card empty"></div>
    </div>
  </div>
</div>

<div id="controls">
  <div id="situation"></div>
  <div id="action-buttons"></div>
  <button id="new-hand-btn" onclick="newHand()">Deal New Hand ▶</button>
</div>

<div id="ai-thinking">
  <div class="title">🤖 AI Last Decision</div>
  <div id="ai-probs-content">— waiting for AI to act —</div>
</div>

<div id="result-banner"></div>

<script>
const ACTION_CLASSES = {
  'FOLD': 'fold',
  'CHECK_CALL': 'check',
  'RAISE_HALF': 'raise',
  'RAISE_ONE': 'raise',
  'RAISE_TWO': 'raise',
  'ALL_IN': 'allin',
};

function cardHTML(card) {
  if (!card || card.rank === '?') {
    return '<div class="card face-down"></div>';
  }
  return `<div class="card ${card.color}">
    <span class="rank">${card.rank}</span>
    <span class="suit">${card.suit}</span>
  </div>`;
}

function renderState(data) {
  document.getElementById('hand-num').textContent = data.hand_num;

  const hs = data.human_score;
  const as = data.ai_score;
  const hsEl = document.getElementById('human-score');
  const asEl = document.getElementById('ai-score');
  hsEl.textContent = (hs >= 0 ? '+' : '') + hs;
  hsEl.className = 'value ' + (hs >= 0 ? 'positive' : 'negative');
  asEl.textContent = (as >= 0 ? '+' : '') + as;
  asEl.className = 'value ' + (as >= 0 ? 'positive' : 'negative');

  // AI hand
  const aiHand = document.getElementById('ai-hand');
  aiHand.innerHTML = data.ai_hole.map(cardHTML).join('');

  // Human hand
  const humanHand = document.getElementById('human-hand');
  humanHand.innerHTML = data.human_hole.map(cardHTML).join('');

  // Board
  const board = document.getElementById('community-cards');
  let boardHTML = data.board.map(cardHTML).join('');
  for (let i = data.board.length; i < 5; i++) {
    boardHTML += '<div class="card empty"></div>';
  }
  board.innerHTML = boardHTML;

  // Pot / street
  document.getElementById('pot-label').textContent = `Pot: ${data.pot}`;
  document.getElementById('street-label').textContent = data.street;

  // Stacks + bets
  document.getElementById('ai-stack').textContent = `Stack: ${data.ai_stack}`;
  document.getElementById('human-stack').textContent = `Stack: ${data.human_stack}`;
  document.getElementById('ai-bet').textContent = data.ai_bet > 0 ? `Bet: ${data.ai_bet}` : '';
  document.getElementById('human-bet').textContent = data.human_bet > 0 ? `Bet: ${data.human_bet}` : '';

  // AI last action
  const aiActionEl = document.getElementById('ai-action');
  if (data.last_ai_action_detail) {
    aiActionEl.textContent = `AI: ${data.last_ai_action_detail}`;
  } else {
    aiActionEl.textContent = '';
  }

  // AI probabilities panel
  const probsEl = document.getElementById('ai-probs-content');
  if (data.ai_probs && data.ai_probs.length > 0) {
    probsEl.innerHTML = data.ai_probs.map(p =>
      `<div class="prob-row">
        <span class="prob-label">${p.name}</span>
        <div class="prob-bar-wrap"><div class="prob-bar" style="width:${p.pct}%"></div></div>
        <span class="prob-pct">${p.pct}%</span>
      </div>`
    ).join('');
  } else if (data.last_ai_action) {
    probsEl.textContent = '— no data —';
  }

  // Situation hint
  const sitEl = document.getElementById('situation');
  if (data.your_turn && data.to_call > 0) {
    sitEl.textContent = `⚠ AI bet ${data.to_call} chips — your move`;
  } else if (data.your_turn) {
    sitEl.textContent = 'Your turn — no bet to call';
  } else if (data.terminal) {
    sitEl.textContent = '';
  } else {
    sitEl.textContent = 'AI is thinking...';
  }

  // Action buttons — use descriptive labels from server
  const btnsEl = document.getElementById('action-buttons');
  btnsEl.innerHTML = '';
  if (data.your_turn && data.legal_actions.length > 0) {
    data.legal_actions.forEach(a => {
      const btn = document.createElement('button');
      btn.className = 'action-btn ' + (ACTION_CLASSES[a.name] || 'raise');
      btn.textContent = a.label || a.name;
      btn.onclick = () => takeAction(a.id);
      btnsEl.appendChild(btn);
    });
  }

  // Terminal
  if (data.terminal && data.result_msg) {
    showResult(data.result_msg);
  }
}

function showResult(msg) {
  const banner = document.getElementById('result-banner');
  banner.textContent = msg;
  banner.style.display = 'block';
  setTimeout(() => { banner.style.display = 'none'; }, 2500);
}

async function newHand() {
  document.getElementById('result-banner').style.display = 'none';
  document.getElementById('action-buttons').innerHTML = '';
  document.getElementById('ai-action').textContent = 'Dealing...';
  const r = await fetch('/api/new_game', { method: 'POST' });
  const data = await r.json();
  renderState(data);
}

async function takeAction(actionId) {
  document.getElementById('action-buttons').innerHTML = '';
  const r = await fetch('/api/action', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: actionId }),
  });
  const data = await r.json();
  renderState(data);
}

// Auto-start first hand
window.onload = () => newHand();
</script>
</body>
</html>
"""


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global _env, _net, _policy_net

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/hunl_final.pt')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    from src.models.policy_net import PolicyNet

    _env = PokerEnv(use_hunl=True)
    ckpt_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        args.checkpoint,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if isinstance(ckpt, dict) and 'pi_net' in ckpt:
        _policy_net = PolicyNet(input_dim=STATE_DIM, n_actions=N_ABSTRACT_ACTIONS, hidden_dim=256)
        _policy_net.load_state_dict(ckpt['pi_net'])
        _net = None
        print(f"Loaded NFSP pi-net from {ckpt_path} (ep {ckpt.get('episode', '?')})")
    else:
        _net = AdvantageNet(input_dim=STATE_DIM, n_actions=N_ABSTRACT_ACTIONS, hidden_dim=256)
        _net.load_state_dict(ckpt)
        print(f"Loaded CFR AdvantageNet from {ckpt_path}")

    print(f"\nOpen your browser at: http://localhost:{args.port}\n")
    app.run(host='0.0.0.0', port=args.port, debug=False)


if __name__ == '__main__':
    main()
