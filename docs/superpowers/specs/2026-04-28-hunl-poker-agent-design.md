# HUNL Poker Agent — Design Specification
**Date:** 2026-04-28  
**Status:** Approved  
**Algorithm:** Behavioral Cloning → Single Deep CFR (SD-CFR)  
**Target hardware:** CPU (configurable runtime: ~6h quick / ~40h full)

---

## 0. Project Goal

Build a Heads-Up No-Limit Texas Hold'em (HUNL) poker AI using a two-phase approach:

1. **Phase 1 — Behavioral Cloning (BC):** Pre-train a policy network on real human hand histories using supervised learning. Gives the agent a warm-start above random play.
2. **Phase 2 — Single Deep CFR (SD-CFR):** Evolve the agent via self-play using SD-CFR. The BC weights initialize the advantage network, cutting iterations needed to reach competent play.

**Final deliverable:** A trained agent benchmarkable via approximate exploitability and playable interactively in a terminal UI. Codebase is modular, fully logged with wandb, and portfolio-ready.

---

## 1. Codebase Structure

```
poker_agent/
├── data/
│   ├── raw/                        # Downloaded hand history files (gitignored)
│   ├── processed/                  # Cleaned .parquet files (gitignored)
│   └── download_instructions.md   # Manual download steps
├── src/
│   ├── data/
│   │   ├── parser.py               # IRC + PHH hand history parsers
│   │   ├── encoder.py              # Information state encoder (STATE_DIM=60)
│   │   └── dataset.py              # PyTorch Dataset/DataLoader for BC
│   ├── models/
│   │   ├── advantage_net.py        # Shared network (BC classifier + CFR advantage estimator)
│   │   └── utils.py                # Weight init helpers
│   ├── cfr/
│   │   ├── traversal.py            # External sampling MCCFR tree traversal
│   │   ├── buffer.py               # Reservoir buffer (NOT FIFO)
│   │   ├── sd_cfr.py               # SD-CFR main training loop
│   │   └── regret_matching.py      # Regret matching+ policy derivation
│   ├── env/
│   │   ├── poker_env.py            # Wrapper around OpenSpiel universal_poker
│   │   └── state_utils.py          # Action abstraction, legal action masking
│   ├── eval/
│   │   ├── exploitability.py       # Local best response approximation
│   │   ├── h2h.py                  # Head-to-head tournament (duplicate matching)
│   │   └── metrics.py              # Win rate, bb/100, exploitability tracking
│   ├── bc/
│   │   ├── train_bc.py             # Phase 1 BC trainer
│   │   └── validate_bc.py          # Action prediction accuracy on held-out hands
│   └── ui/
│       └── play.py                 # Terminal UI — post-training only (Step 15)
├── configs/
│   ├── quick.yaml                  # CPU-friendly: ~6 hours
│   └── full.yaml                   # Full spec: ~40 hours
├── scripts/
│   ├── preprocess_irc.py           # Parse IRC → processed/irc_hunl.parquet
│   ├── preprocess_phh.py           # Parse PHH → processed/phh_hunl.parquet
│   ├── train_phase1_bc.py          # Launch BC training
│   ├── train_phase2_sdcfr.py       # Launch SD-CFR: python scripts/train_phase2_sdcfr.py --config quick
│   └── evaluate.py                 # Run evaluation suite
├── tests/
│   └── test_core.py                # Phase 0 gate — must pass before Phase 1 starts
├── checkpoints/                    # Saved model weights (gitignored)
├── logs/                           # Training logs (gitignored)
├── docs/
│   └── superpowers/specs/          # Design documents
├── environment.yml
├── requirements.txt
└── README.md
```

---

## 2. Build Order (Implementation Phases)

Build in this exact order. Each phase has a hard gate before the next begins.

### Phase 0 — Proof of Concept (Leduc Poker, no BC, random init)
Build the core engineering scaffold targeting Leduc Poker (tiny 2-round game, verifiable in seconds on CPU).

Steps:
1. `env/poker_env.py` — OpenSpiel wrapper. Auto-validates `firstPlayer` on construction.
2. `data/encoder.py` — STATE_DIM=60 encoder, shared between BC and SD-CFR.
3. `models/advantage_net.py` — LayerNorm MLP, orthogonal init gain=0.01.
4. `cfr/regret_matching.py` — Regret matching+.
5. `cfr/buffer.py` — Reservoir buffer.
6. `cfr/traversal.py` — External sampling MCCFR traversal.
7. `cfr/sd_cfr.py` — SD-CFR loop (random init, no BC, Leduc only).
8. `tests/test_core.py` — Run automatically. Must pass all 6 checks.

**Gate:** `tests/test_core.py` must pass before Phase 1 begins.

### Phase 1 — Behavioral Cloning (HUNL, PHH + IRC)
Steps:
9. `data/parser.py` — PHH parser (primary), IRC parser (supplementary).
10. `scripts/preprocess_phh.py` + `scripts/preprocess_irc.py` — run once.
11. `data/dataset.py` — PyTorch Dataset for BC.
12. `bc/train_bc.py` + `bc/validate_bc.py` — BC training and validation.

**Gate:** BC action prediction accuracy >40% on held-out 10% split.

### Phase 2 — SD-CFR (HUNL, BC-initialized)
Steps:
13. Load BC checkpoint into advantage net. Run SD-CFR on HUNL.
14. `eval/exploitability.py`, `eval/h2h.py`, `eval/metrics.py` — evaluation suite.

**Gate:** Exploitability decreases monotonically over first 50 iterations.

### Phase 3 — Terminal UI (post-checkpoint only)
Step:
15. `ui/play.py` — interactive play against trained checkpoint. Built last.

---

## 3. `tests/test_core.py` — Phase 0 Gate

Six checks, all must pass before Phase 1 is allowed to start:

1. **Leduc exploitability decreases** over 50 SD-CFR iterations (core correctness)
2. **Reservoir buffer has uniform coverage** — chi-square test after 3M inserts into capacity=1M buffer
3. **Regret matching returns uniform distribution** when all advantages are negative
4. **Regret matching returns proportional distribution** when advantages are positive
5. **State encoder output is exactly STATE_DIM=60 floats with no NaNs** — tested on 10 diverse game states including edge cases (river all-in, preflop fold)
6. **Legal action mask correctly blocks illegal actions** at every game state in a full random game

---

## 4. Game Engine

**Primary:** OpenSpiel `universal_poker` via `pip install open_spiel` on native Windows Python 3.11.  
**Fallback:** After 3 failed install attempts, port `PokerRL/game/` to Python 3.11.

HUNL config:
```python
game = pyspiel.load_game("universal_poker", {
    "betting": "nolimit",
    "numPlayers": 2,
    "numRounds": 4,
    "blind": "1 2",
    "firstPlayer": "2 1 1 1",
    "numSuits": 4,
    "numRanks": 13,
    "numHoleCards": 2,
    "numBoardCards": "0 3 1 1",
    "stack": "200 200",
    "bettingAbstraction": "fullgame",
})
```

`poker_env.py` validates `firstPlayer` on construction via a manual hand trace — asserts player 0 (BTN/SB) acts first preflop. Raises immediately if misconfigured.

**Action abstraction** (6 actions, resolved inside the wrapper):
```
0: FOLD
1: CHECK_CALL
2: RAISE_0.5x_POT
3: RAISE_1x_POT
4: RAISE_2x_POT
5: ALL_IN
```
Bet-size-to-bucket mapping lives in `env/state_utils.py`. The rest of the codebase only sees action indices 0–5.

---

## 5. Datasets

### PHH Dataset (primary)
- Source: https://zenodo.org/records/13997158
- 21,605,687 uncorrupted NLHE hands
- Format: `.phh` files, parsed via `pip install phh`
- Filter: 2-player only, effective stack ≥ 20 BB, complete action sequences, winner-player filter

### IRC Poker Database (supplementary)
- Source: http://poker.cs.ualberta.ca/IRC/IRCdata.tgz (~500MB)
- Filter: `nolimit` tag only — aggressively drop limit and mixed game hands
- **If fewer than 100K clean NLHE hands survive the filter, skip IRC entirely and train on PHH only.** Log exact counts either way.

### Action bucket mapping
Real hand bet sizes are mapped to the nearest abstract bucket. Hands where the actual size falls outside all buckets are dropped. Log drop rate — target <5%.

### Train/val split
90/10, stratified by street (preflop/flop/turn/river) to prevent preflop overrepresentation in validation.

---

## 6. Information State Encoding

`src/data/encoder.py` — output: `float32[60]`, STATE_DIM=60. Identical between BC and SD-CFR.

```
[0:4]    Hole cards: 2 cards × (rank/12, suit/3) normalized
[4:14]   Community cards: 5 cards × (rank/12, suit/3), padded -1 for unseen
[14:18]  Street one-hot: [preflop, flop, turn, river]
[18:20]  Stack sizes: [hero_stack, villain_stack] / starting_stack
[20:22]  Pot info: [pot, amount_to_call] / starting_stack
[22:24]  Position: [is_dealer/SB, is_BB]
[24:30]  Hero action history this street: last 3 actions × 2 floats [action_idx/5, bet_fraction_of_pot]
[30:36]  Villain action history this street: last 3 actions × 2 floats, padded -1
[36:42]  Preflop hero actions: last 3 actions × 2 floats, padded -1
[42:48]  Preflop villain actions: last 3 actions × 2 floats, padded -1
[48:52]  Hand strength: [preflop_bucket, postflop_equity, nut_advantage, texture]
[52:56]  Game context: [spr, pot_odds, eff_stack_bb, pot_size_bb]
[56:60]  Reserved (zeros)
```

**Card encoding:** rank×suit (2 floats/card), NOT one-hot (52 floats/card). Avoids sparsity, improves generalization.  
**Hand strength bucket:** `phevaluator` equity vs random range, 8 bins normalized to [0,1]. Precomputed at dataset load time for BC, computed live during CFR traversal.

---

## 7. Network Architecture

`src/models/advantage_net.py` — shared between BC and SD-CFR phases.

```
Input(60) → Linear(256) → LayerNorm → ReLU → Dropout(0.1)
          → Linear(256) → LayerNorm → ReLU → Dropout(0.1)
          → Linear(128) → LayerNorm → ReLU
          → Linear(6)   [raw output — NO softmax]
```

**LayerNorm (not BatchNorm):** BatchNorm breaks at batch size 1. CFR traversal calls the network one sample at a time. Non-negotiable.  
**Orthogonal init, gain=0.01:** Preserves BC warm-start behavior when SD-CFR begins. Large init destroys the BC signal.  
**No output activation:** Raw logits for BC cross-entropy; raw advantage values for regret matching+. Softmax would break regret matching.

---

## 8. Phase 1: Behavioral Cloning

**Objective:** Train the advantage network as a multiclass classifier on `(info_state, human_action)` pairs. Cross-entropy loss. This warm-starts SD-CFR with human-level poker intuition.

**Hyperparameters:**

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Batch size | 2048 |
| Epochs | 20–50 |
| Optimizer | AdamW, weight_decay=1e-5 |
| Scheduler | CosineAnnealingLR |
| Dropout | 0.1 |
| Grad clip | 1.0 |

**Validation target:** >40% action prediction accuracy on held-out 10%. Random baseline = 16.7% (6 actions). >60% likely indicates overfitting.

---

## 9. Phase 2: Single Deep CFR

### Algorithm summary
SD-CFR maintains one advantage network. External sampling MCCFR traversal generates `(info_state, advantages, reach_prob)` tuples stored in a reservoir buffer. After each iteration, the network is trained on the buffer with weighted MSE loss. Policy is derived at inference time via regret matching+.

### Launch configs

| Parameter | `--config quick` | `--config full` |
|---|---|---|
| `n_iterations` | 100 | 500 |
| `n_traversals_per_iter` | 500 | 1500 |
| `buffer_capacity` | 500,000 | 2,000,000 |
| `n_batches_adv_training` | 200 | 1000 |
| `batch_size` | 256 | 512 |
| Est. runtime (CPU) | ~6 hours | ~40 hours |

Configs live in `configs/quick.yaml` and `configs/full.yaml`.

### Reservoir buffer
Reservoir sampling — NOT a FIFO queue. Maintains a uniform sample of all traversal data seen. Critical for SD-CFR correctness.

### Regret matching+
Policy at inference: clip advantages to ≥0, normalize. If all advantages ≤0, return uniform over legal actions.

### Logging (wandb)
```python
wandb.log({
    "iteration": i,
    "exploitability_mbbh": exploitability,
    "adv_loss": adv_loss,
    "buffer_size": len(buffer),
    "h2h_vs_random_bb100": h2h_score,
})
```

---

## 10. Evaluation

**Approximate exploitability:** Local best response (LBR) — greedy best response without lookahead. Underestimates true exploitability but tractable on CPU. Report in mbb/h. Run every 25 iterations.

**Head-to-head tournament:** 10,000 hands, position swapped every hand, duplicate matching (same deck in both positions). Returns mean bb/100 with 95% CI.

**Benchmark ladder:**
1. Random agent — should beat by >10 BB/100. If not, something is broken.
2. Call station (always call) — should beat by >5 BB/100 after BC.
3. BC-only agent — compare to show SD-CFR improvement.
4. BC + SD-CFR final agent — main showcase result.

---

## 11. README Structure

For ML-literate readers unfamiliar with poker AI:

1. What this is — HUNL as imperfect-information game, Nash equilibrium goal
2. The novelty — BC warm-start + SD-CFR self-play, why neither alone is as good
3. Architecture diagram (Mermaid flowchart)
4. Reproducing results — exact steps: clone, conda env, download data, preprocess, test gate, BC train, SD-CFR train, evaluate, play
5. Results — exploitability curve, H2H win rates (filled in after training)
6. Project structure — file tree

---

## 12. GitHub Push Cadence

Always push to `main` branch. Push after:
- Phase 0 complete + `tests/test_core.py` passes
- Data preprocessing scripts complete
- BC training complete (include loss/accuracy curves)
- SD-CFR 50-iteration checkpoint
- Evaluation suite complete
- Terminal UI complete (final push)

---

## 13. Known Pitfalls

| Pitfall | Consequence | Fix |
|---|---|---|
| FIFO buffer instead of reservoir | Overfits recent traversals, diverges | ReservoirBuffer only |
| BatchNorm in network | Crashes at batch size 1 during traversal | LayerNorm everywhere |
| No legal action masking | Env throws exception on illegal action | Always mask before regret matching |
| Wrong `firstPlayer` in OpenSpiel | Positional advantage reversed | Auto-validated on env construction |
| IRC data not filtered aggressively | Limit hold'em poisons action distribution | Log counts, skip if <100K NLHE |
| Softmax on advantage output | Breaks regret matching | Raw output, no final activation |
| Stack/pot not normalized | Gradient explodes | Always divide by starting_stack |
| Large weight init | Destroys BC warm-start | Orthogonal init gain=0.01 |
| Too many training batches/iter | Overfits buffer, catastrophic forgetting | Cap at 1000 (full) / 200 (quick) |

---

## 14. Dependencies

```yaml
name: poker_agent
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.11
  - pytorch=2.2
  - numpy=1.26
  - pandas=2.2
  - pyarrow
  - tqdm
  - wandb
  - matplotlib
  - seaborn
  - pip:
    - open_spiel        # primary game engine (fallback: port PokerRL/game/)
    - treys
    - phevaluator
    - phh               # PHH dataset parser
```

---

## 15. Key References

| Priority | Resource | URL |
|---|---|---|
| 1 | Deep CFR paper | https://arxiv.org/abs/1811.00164 |
| 2 | Single Deep CFR paper | https://arxiv.org/abs/1901.07621 |
| 3 | EricSteinberger/Deep-CFR | https://github.com/EricSteinberger/Deep-CFR |
| 4 | EricSteinberger/PokerRL | https://github.com/EricSteinberger/PokerRL |
| 5 | EricSteinberger/DREAM | https://github.com/EricSteinberger/DREAM |
| 6 | OpenSpiel | https://github.com/google-deepmind/open_spiel |
| 7 | PHH Dataset | https://zenodo.org/records/13997158 |
| 8 | IRC parsing tutorial | https://allenfrostline.com/blog/texas-holdem-series-2/ |
| 9 | Coherent Soft IL | https://arxiv.org/abs/2305.16498 |
| 10 | phevaluator | https://github.com/HenryRLee/PokerHandEvaluator |
