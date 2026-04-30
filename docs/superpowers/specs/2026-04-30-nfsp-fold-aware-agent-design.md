# NFSP Fold-Aware Agent Design

**Goal:** Replace SD-CFR self-play with Neural Fictitious Self-Play (NFSP) to produce an agent that folds bad hands, avoids calling all-ins with weak holdings, and preserves chips — trained on top of the existing BC warm-start.

**Architecture:** Reuse existing code (game engine, encoder, AdvantageNet, BC checkpoint). Add `src/nfsp/` with episode-based training loop. Two networks per agent: Q-net (DQN best-response) + π-net (average strategy). Web app switches to π-net for human play.

**Tech Stack:** PyTorch, existing HUNLGame/HUNLState, phevaluator, existing AdvantageNet + PolicyNet architectures, ReservoirBuffer (reused from CFR phase).

---

## Why NFSP fixes the fold problem

SD-CFR with 1M traversals is ~100x too few for HUNL — the network never accumulates enough regret signal to differentiate hand strengths. NFSP fixes this because the Q-network's reward is direct: calling all-in with 7-2o → reward ≈ −0.9. After a few thousand complete hands, Q(FOLD) > Q(BAD_CALL) and the agent starts folding. No tree traversal required.

The 0% fold rate in BC training data (IRC data never shows hole cards for folding players) is bypassed entirely — NFSP's DQN component learns fold behavior from actual game outcomes, not imitation.

---

## Files

### New files
```
src/nfsp/
  __init__.py
  replay_buffer.py      Circular buffer: (state, action, return) for DQN
  agent.py              NFSPAgent: Q-net + π-net, act(), store_transition()
  train_nfsp.py         Episode-based training loop
configs/nfsp.yaml       Hyperparameters
scripts/train_nfsp.py   Entry point — loads BC checkpoint, runs NFSP
```

### Modified files
```
src/ui/web_app.py       Load π-net from nfsp checkpoint for human play
src/models/policy_net.py  Verify softmax output (may already be correct)
```

### Unchanged (reused)
```
src/env/hunl_game.py, poker_env.py   Game engine
src/data/encoder.py                  State encoding (60-dim)
src/models/advantage_net.py          Q-network architecture
src/cfr/buffer.py                    ReservoirBuffer for π-net supervised data
checkpoints/bc_final.pt              Warm-start weights for both networks
```

---

## Algorithm

### Per episode (one complete HUNL hand)

```
state = env.new_game()
episode_transitions = []

while not state.is_terminal():
    skip chance nodes (deal cards automatically)
    player = state.current_player()
    action, mode = agent.act(state, player)   # mode = 'br' or 'avg'
    episode_transitions.append((state_vec, player, action, mode))
    state = state.child(action)

for player in [0, 1]:
    G = state.returns()[player]               # normalized chip delta
    for (s, p, a, mode) in episode_transitions if p == player:
        if mode == 'br':
            replay_buffer.add(s, a, G)        # DQN update
            reservoir_buffer.add(s, a)        # supervised π-net update
```

### Action selection (NFSPAgent.act)

```
with prob η (=0.1): mode='br',  act greedy from Q-net (ε-greedy exploration)
with prob 1-η:      mode='avg', sample from π-net softmax
```

### Network updates (every `update_every` steps)

```
Q-net:  loss = MSE(Q(s)[a], G)                  sampled from replay_buffer
π-net:  loss = CrossEntropy(π(s), a)             sampled from reservoir_buffer
```

---

## Reward shaping

```python
base_reward = (final_stack - starting_stack) / starting_stack

# Fold bonus: agent folded AND hand equity < 0.30 facing a large bet
if agent_action == FOLD and hand_equity(hole_cards, board_cards) < 0.30:
    fold_bonus = 0.05
else:
    fold_bonus = 0.0

reward = base_reward + fold_bonus
```

`hand_equity` uses phevaluator for postflop and `_preflop_bucket` from encoder.py for preflop. Applied only to the player who folded, not the winner.

---

## Hyperparameters (configs/nfsp.yaml)

```yaml
n_episodes: 200000
eta: 0.1                    # anticipatory parameter
epsilon_start: 0.30         # Q-net exploration
epsilon_end: 0.01
epsilon_decay_episodes: 100000
replay_buffer_size: 100000
reservoir_buffer_size: 200000
update_every: 128           # steps between network updates
batch_size: 512
lr_q: 0.0001
lr_pi: 0.0001
hidden_dim: 256
checkpoint_freq: 10000      # episodes
checkpoint_dir: checkpoints
fold_bonus: 0.05
fold_equity_threshold: 0.30
```

---

## Checkpoint format

```python
# Save
torch.save({
    'q_net': q_net.state_dict(),
    'pi_net': pi_net.state_dict(),
    'episode': episode,
}, 'checkpoints/nfsp_final.pt')

# Web app loads pi_net (Nash-approximate balanced strategy)
# Q-net available as aggressive/exploitative mode
```

---

## Web app change

`web_app.py` currently loads `AdvantageNet` and uses regret-matching to derive policy. After this change:
- Detect checkpoint type: if key `pi_net` exists → NFSP checkpoint, load `PolicyNet`
- If key missing → legacy CFR checkpoint, use existing `AdvantageNet` + regret-matching
- Backward-compatible: old checkpoints still work

---

## Success criteria

After 200K episodes:
1. Agent folds at least 15% of hands (up from ~0%)
2. Agent folds ≥ 70% of the time facing an all-in when holding bottom-20% hands
3. Win rate vs random baseline ≥ +200 bb/100 (down from +269 is acceptable — Nash balance trades exploitability against random for unexploitability against skilled)
4. No more "always all-in" pattern — action distribution shows all 6 actions used

---

## Training time estimate

200K episodes × ~20 decisions/hand × 2 players = ~8M forward passes.
On CPU (no GPU): ~3–4 hours.
With BC warm-start: fold behavior should emerge within first 20K episodes (~15 min).
