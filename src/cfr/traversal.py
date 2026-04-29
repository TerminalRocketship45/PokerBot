import numpy as np
import torch
from src.cfr.buffer import ReservoirBuffer
from src.cfr.regret_matching import regret_matching_plus
from src.data.encoder import encode_state
from src.models.advantage_net import AdvantageNet


def traverse(
    state,
    traversing_player: int,
    adv_net: AdvantageNet,
    buffer: ReservoirBuffer,
    reach_prob: float,
    use_hunl: bool = False,
    starting_stack: float = 200.0,
) -> float:
    """
    External sampling MCCFR traversal.
    Returns the expected value for traversing_player from this state.
    """
    if state.is_terminal():
        return state.returns()[traversing_player]

    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        probs = [p for _, p in outcomes]
        actions = [a for a, _ in outcomes]
        chosen_action = np.random.choice(actions, p=probs)
        return traverse(
            state.child(chosen_action), traversing_player, adv_net,
            buffer, reach_prob, use_hunl, starting_stack,
        )

    current_player = state.current_player()
    legal_actions = state.legal_actions()
    n_actions = adv_net.net[-1].out_features

    info_state = encode_state(state, current_player, use_hunl, starting_stack)
    state_tensor = torch.FloatTensor(info_state).unsqueeze(0)

    adv_net.eval()
    with torch.no_grad():
        advantages_full = adv_net(state_tensor).squeeze(0)

    legal_mask = torch.zeros(n_actions, dtype=torch.bool)
    for a in legal_actions:
        if a < n_actions:
            legal_mask[a] = True

    action_probs = regret_matching_plus(advantages_full, legal_mask)

    if current_player == traversing_player:
        action_values = {}
        for a in legal_actions:
            if a >= n_actions:
                continue
            action_values[a] = traverse(
                state.child(a), traversing_player, adv_net,
                buffer, reach_prob, use_hunl, starting_stack,
            )

        node_value = sum(
            action_probs[a].item() * action_values[a]
            for a in action_values
        )

        advantages_target = np.zeros(n_actions, dtype=np.float32)
        for a in action_values:
            advantages_target[a] = action_values[a] - node_value

        buffer.add(info_state, advantages_target, reach_prob)
        return node_value
    else:
        legal_probs = [action_probs[a].item() for a in legal_actions if a < n_actions]
        valid_actions = [a for a in legal_actions if a < n_actions]
        total = sum(legal_probs)
        if total < 1e-9:
            chosen_action = np.random.choice(valid_actions)
            chosen_prob = 1.0 / len(valid_actions)
        else:
            normalized = [p / total for p in legal_probs]
            chosen_action = np.random.choice(valid_actions, p=normalized)
            chosen_prob = normalized[valid_actions.index(chosen_action)]

        return traverse(
            state.child(chosen_action), traversing_player, adv_net,
            buffer, reach_prob * chosen_prob, use_hunl, starting_stack,
        )
