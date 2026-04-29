"""
Interactive terminal: play against a trained agent.
Usage: python src/ui/play.py --checkpoint checkpoints/iter_0050.pt
"""
import argparse
import random
import torch
from src.env.poker_env import PokerEnv
from src.models.advantage_net import AdvantageNet
from src.data.encoder import encode_state, STATE_DIM
from src.cfr.regret_matching import regret_matching_plus
from src.env.state_utils import ABSTRACT_ACTIONS, N_ABSTRACT_ACTIONS

HUMAN = 1
AI = 0


def display_state(state, human_player: int):
    info = state.information_state_string(human_player)
    print("\n" + "─" * 50)
    print(info)
    legal = state.legal_actions()
    valid = [a for a in legal if a < N_ABSTRACT_ACTIONS]
    print("\nYour actions:")
    for a in valid:
        print(f"  [{a}] {ABSTRACT_ACTIONS[a]}")


def get_human_action(state) -> int:
    legal = state.legal_actions()
    valid = [a for a in legal if a < N_ABSTRACT_ACTIONS]
    while True:
        try:
            choice = int(input("Enter action number: ").strip())
            if choice in valid:
                return choice
            print(f"Invalid. Choose from: {valid}")
        except (ValueError, EOFError):
            return random.choice(valid)


def get_ai_action(net: AdvantageNet, state, ai_player: int) -> int:
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
    probs = regret_matching_plus(adv, mask)
    action = torch.multinomial(probs, 1).item()
    if action not in legal:
        action = random.choice(valid)
    action_name = ABSTRACT_ACTIONS[action] if action < N_ABSTRACT_ACTIONS else str(action)
    print(f"\nAI plays: {action_name} (probs: {[f'{p:.2f}' for p in probs.tolist()]})")
    return action


def play_interactive(checkpoint_path: str):
    env = PokerEnv(use_hunl=True)
    net = AdvantageNet(input_dim=STATE_DIM, n_actions=env.num_actions(), hidden_dim=256)
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    print(f"Loaded agent from {checkpoint_path}")

    ai_wins = human_wins = hands = 0
    try:
        while True:
            hands += 1
            print(f"\n{'='*50}")
            print(f"Hand #{hands}  |  AI wins: {ai_wins}  |  Your wins: {human_wins}")
            state = env.new_game()
            while not state.is_terminal():
                if state.is_chance_node():
                    state.apply_action(state.chance_outcomes()[0][0])
                    continue
                player = state.current_player()
                if player == HUMAN:
                    display_state(state, HUMAN)
                    action = get_human_action(state)
                else:
                    action = get_ai_action(net, state, AI)
                state.apply_action(action)
            returns = state.returns()
            print(f"\nResult: You {'WIN' if returns[HUMAN] > 0 else 'LOSE'} "
                  f"{abs(returns[HUMAN]):.0f} chips")
            if returns[AI] > 0:
                ai_wins += 1
            else:
                human_wins += 1
            input("\nPress Enter for next hand (Ctrl+C to quit)...")
    except KeyboardInterrupt:
        print(f"\nGame over. Played {hands} hands. AI: {ai_wins} | You: {human_wins}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    play_interactive(args.checkpoint)
