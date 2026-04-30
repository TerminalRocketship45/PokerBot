# HUNL Poker Agent

Heads-Up No-Limit Texas Hold'em AI trained via **Behavioral Cloning → Single Deep CFR**.

## How it works

Texas Hold'em is an imperfect-information game — players can't see each other's cards. The goal is to reach **Nash equilibrium**: a strategy where neither player can gain by changing their play, regardless of what the opponent does.

**Phase 1 — Behavioral Cloning:** Supervised learning on 1M+ real human hand histories. Trains the network to predict human actions. Gives the agent a warm start above random play.

**Phase 2 — Single Deep CFR:** The agent plays millions of hands against itself. At each decision point it computes *counterfactual regret* — how much better it would have done by taking a different action. Over time, regrets average to zero = Nash equilibrium. BC weights initialize this phase so fewer iterations are needed.

```
  IRC Dataset          Behavioral Cloning       SD-CFR Self-Play       Trained Agent
  1M+ NLHE hands  -->  Cross-entropy loss   -->  Regret minimization -->  Nash-approx
                        (imitates humans)         (beats humans)
```

**Training pipeline:**

```
  [IRC hand histories]
          |
          v
  [preprocess_irc.py]  -->  data/processed/irc_hunl.parquet  (state, action pairs)
          |
          v
  [train_phase1_bc.py]  -->  checkpoints/bc_final.pt  (warm-start weights)
          |
          v
  [run_training.py --config medium --bc_checkpoint bc_final.pt]
          |
          v
  [checkpoints/hunl_final.pt]  -->  play via web UI or evaluate
```

**Action space (6 abstract actions):**

```
  0 FOLD         1 CHECK/CALL         2 RAISE 0.5x pot
  3 RAISE 1x pot    4 RAISE 2x pot    5 ALL-IN
```

**Game rules:**
- 2 players, SB = 10, BB = 20
- Starting stacks: random 100–1000 chips per hand (teaches stack-depth awareness)
- Returns normalized by starting stack so all games contribute equally to training

## Results

| Agent | vs Random (bb/100) | Notes |
|---|---|---|
| Random baseline | 0 | uniform random actions |
| SD-CFR quick (iter 10) | +397 ± 107 | early aggressive policy |
| SD-CFR quick (iter 100) | +269 ± 36 | converged Nash-approximate |

Win rate vs random *decreases* as training progresses — expected. Early iterations learn exploitative all-in policies. Later iterations balance their strategy to be unexploitable, which wins less against random but can't be beaten by a skilled player.

## Setup

```bash
conda env create -f environment.yml
conda activate rl_env
pip install phevaluator
```

## Train

```bash
# Preprocess IRC data
python scripts/preprocess_irc.py

# Phase 1: Behavioral Cloning (~15 min CPU)
python scripts/train_phase1_bc.py

# Phase 2: SD-CFR from BC weights (~hours, 500 iterations)
python scripts/run_training.py --config medium --bc_checkpoint checkpoints/bc_final.pt

# Or run both phases in sequence:
python scripts/train_full_pipeline.py --config medium
```

## Play

```bash
# Start web UI at http://localhost:5000
run_server.bat
```

Or run a narrated demo hand:
```bash
python scripts/demo_hand.py
```

## Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/hunl_final.pt
```

## Project Structure

```
src/
  env/       HUNL game engine (OpenSpiel-compatible), action abstraction
  data/      State encoder, IRC parser, BC dataset loader
  models/    AdvantageNet (shared by BC and CFR phases)
  cfr/       MCCFR traversal, reservoir buffer, SD-CFR training loop
  bc/        Behavioral cloning trainer and validator
  eval/      Exploitability, head-to-head, metrics
  ui/        Flask web UI for playing against the agent
configs/     quick.yaml, medium.yaml training presets
scripts/     All entry points (preprocess, train, evaluate, demo)
checkpoints/ Saved weights (gitignored)
data/        Raw IRC data + processed parquet (gitignored)
```

## References

1. [Deep CFR](https://arxiv.org/abs/1811.00164) — Brown et al. 2018
2. [Single Deep CFR](https://arxiv.org/abs/1901.07621) — Steinberger 2019
3. [OpenSpiel](https://github.com/google-deepmind/open_spiel) — game environment
4. [IRC Poker Database](http://poker.cs.ualberta.ca/IRC/) — hand history dataset
