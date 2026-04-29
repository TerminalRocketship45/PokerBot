import torch
import numpy as np
import random
from src.env.poker_env import PokerEnv
from src.models.advantage_net import AdvantageNet
from src.cfr.regret_matching import regret_matching_plus
from src.data.encoder import encode_state


def _agent_action(net: AdvantageNet, state, player: int,
                  n_actions: int, use_hunl: bool) -> int:
    legal = state.legal_actions()
    valid = [a for a in legal if a < n_actions]
    info = encode_state(state, player, use_hunl=use_hunl)
    t = torch.FloatTensor(info).unsqueeze(0)
    with torch.no_grad():
        adv = net(t).squeeze(0)
    mask = torch.zeros(n_actions, dtype=torch.bool)
    for a in valid:
        mask[a] = True
    probs = regret_matching_plus(adv, mask)
    action = torch.multinomial(probs, 1).item()
    return action if action in legal else random.choice(legal)


def run_tournament(
    agent_a: AdvantageNet,
    agent_b: AdvantageNet,
    env: PokerEnv,
    n_hands: int = 10_000,
) -> dict:
    """
    Duplicate matching: each deal played twice with positions swapped.
    Returns mean bb/100 for agent_a, std, 95% CI.
    """
    n_actions = env.num_actions()
    use_hunl = env.use_hunl
    agent_a.eval()
    agent_b.eval()

    results = []
    for _ in range(n_hands // 2):
        for swap in [False, True]:
            state = env.new_game()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    actions_list, probs_list = zip(*outcomes)
                    chosen = np.random.choice(actions_list, p=probs_list)
                    state.apply_action(chosen)
                    continue
                p = state.current_player()
                net = (agent_b if swap else agent_a) if p == 0 else \
                      (agent_a if swap else agent_b)
                state.apply_action(_agent_action(net, state, p, n_actions, use_hunl))
            r = state.returns()
            results.append(r[1] if swap else r[0])

    arr = np.array(results)
    mean = arr.mean() * 100
    std = arr.std() * 100
    ci = 1.96 * std / np.sqrt(len(arr))
    return {"mean_bb100": mean, "std_bb100": std, "ci95": ci, "n_hands": len(arr)}
