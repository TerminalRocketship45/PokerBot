"""
Usage: python scripts/evaluate.py --checkpoint checkpoints/iter_0050.pt
"""
import argparse
import torch
from src.env.poker_env import PokerEnv
from src.models.advantage_net import AdvantageNet
from src.data.encoder import STATE_DIM
from src.eval.exploitability import approximate_exploitability
from src.eval.h2h import run_tournament


def make_random_agent(n_actions: int) -> AdvantageNet:
    net = AdvantageNet(input_dim=STATE_DIM, n_actions=n_actions)
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    env = PokerEnv(use_hunl=True)
    n_actions = env.num_actions()

    agent = AdvantageNet(input_dim=STATE_DIM, n_actions=n_actions, hidden_dim=256)
    agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    agent.eval()
    print(f"Loaded: {args.checkpoint}")

    print("\n[1] Approximate exploitability (LBR, 500 hands)...")
    expl = approximate_exploitability(agent, env, n_hands=500)
    print(f"  Exploitability: {expl:.1f} mbb/hand")

    print("\n[2] H2H vs random agent (2000 hands, duplicate)...")
    random_agent = make_random_agent(n_actions)
    result = run_tournament(agent, random_agent, env, n_hands=2000)
    print(f"  Win rate vs random: {result['mean_bb100']:.1f} ± {result['ci95']:.1f} bb/100")
    if result["mean_bb100"] < 10.0:
        print("  WARNING: < 10 bb/100 vs random. Check training.")


if __name__ == "__main__":
    main()
