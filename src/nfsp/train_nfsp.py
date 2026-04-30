# src/nfsp/train_nfsp.py
"""
NFSP episode-based training loop.
Plays complete HUNL hands (no tree traversal), trains Q-net (DQN) and
pi-net (supervised average strategy) from episode experience.
"""
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from src.env.poker_env import PokerEnv
from src.data.encoder import encode_state, _hand_strength_bucket
from src.models.advantage_net import AdvantageNet
from src.models.policy_net import PolicyNet
from src.nfsp.agent import NFSPAgent
from src.nfsp.replay_buffer import ReplayBuffer
from src.cfr.buffer import ReservoirBuffer

FOLD = 0


@dataclass
class NFSPConfig:
    n_episodes: int = 200_000
    eta: float = 0.1
    epsilon_start: float = 0.30
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 100_000
    replay_buffer_size: int = 100_000
    reservoir_buffer_size: int = 200_000
    update_every: int = 128
    batch_size: int = 512
    lr_q: float = 1e-4
    lr_pi: float = 1e-4
    hidden_dim: int = 256
    checkpoint_freq: int = 10_000
    checkpoint_dir: str = "checkpoints"
    fold_bonus: float = 0.05
    fold_equity_threshold: float = 0.30


def _fold_equity(hole_ints: list, board_ints: list) -> float:
    """Hand strength in [0,1] using existing encoder helpers."""
    hole  = [(c // 4, c % 4) for c in hole_ints]
    board = [(c // 4, c % 4) for c in board_ints]
    return _hand_strength_bucket(hole, board)


def _run_episode(env: PokerEnv, agent: NFSPAgent):
    """
    Play one complete HUNL hand.
    Returns (terminal_state, transitions) where transitions[player] is a list of
    (state_vec, action, mode, board_len_at_decision) for that player.
    """
    state = env.new_game()
    transitions = [[], []]

    while not state.is_terminal():
        while not state.is_terminal() and state.is_chance_node():
            outcomes = state.chance_outcomes()
            acts, probs = zip(*outcomes)
            state = state.child(int(np.random.choice(acts, p=probs)))

        if state.is_terminal():
            break

        player = state.current_player()
        state_vec = encode_state(state, player, use_hunl=True)
        legal = state.legal_actions()
        board_len = len(state._board)

        action, mode = agent.act(state_vec, legal)
        transitions[player].append((state_vec, action, mode, board_len))
        state = state.child(action)

    return state, transitions


def _compute_rewards(terminal_state, transitions: list, config: NFSPConfig):
    """
    Returns per-player reward (base + fold bonus).
    """
    base = terminal_state.returns()
    rewards = list(base)

    for player in [0, 1]:
        if not transitions[player]:
            continue
        last_action = transitions[player][-1][1]
        last_board_len = transitions[player][-1][3]
        if last_action == FOLD:
            hole_ints  = terminal_state._hole[player]
            board_ints = terminal_state._board[:last_board_len]
            eq = _fold_equity(hole_ints, board_ints)
            if eq < config.fold_equity_threshold:
                rewards[player] += config.fold_bonus

    return rewards


def _update_q_net(q_net: AdvantageNet, replay_buf: ReplayBuffer,
                  optimizer: torch.optim.Optimizer, batch_size: int) -> float:
    if len(replay_buf) < batch_size:
        return 0.0
    batch = replay_buf.sample(batch_size)
    states  = torch.FloatTensor(np.array([s for s, _, _ in batch]))
    actions = torch.LongTensor([a for _, a, _ in batch])
    returns = torch.FloatTensor([g for _, _, g in batch])

    q_net.train()
    optimizer.zero_grad()
    q_vals  = q_net(states)
    q_taken = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = nn.MSELoss()(q_taken, returns)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def _update_pi_net(pi_net: PolicyNet, reservoir_buf: ReservoirBuffer,
                   optimizer: torch.optim.Optimizer, batch_size: int) -> float:
    if len(reservoir_buf) < batch_size:
        return 0.0
    batch = reservoir_buf.sample(batch_size)
    states  = torch.FloatTensor(np.array([s for s, _, _ in batch]))
    actions = torch.LongTensor([int(a) for _, a, _ in batch])

    pi_net.train()
    optimizer.zero_grad()
    log_probs = torch.log(pi_net(states) + 1e-8)
    loss = nn.NLLLoss()(log_probs, actions)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(pi_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def train_nfsp(
    config: NFSPConfig,
    bc_checkpoint: Optional[str] = None,
) -> tuple:
    """
    Train NFSP agent. Returns (q_net, pi_net).
    """
    import os, time
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    env    = PokerEnv(use_hunl=True)
    q_net  = AdvantageNet(input_dim=60, n_actions=6, hidden_dim=config.hidden_dim)
    pi_net = PolicyNet(input_dim=60, n_actions=6, hidden_dim=config.hidden_dim)

    if bc_checkpoint and os.path.exists(bc_checkpoint):
        sd = torch.load(bc_checkpoint, map_location="cpu")
        q_net.load_state_dict(sd)
        pi_net.load_state_dict(sd)
        print(f"Loaded BC weights from {bc_checkpoint}")
    else:
        print("Starting from random init (no BC checkpoint found)")

    replay_buf    = ReplayBuffer(capacity=config.replay_buffer_size)
    reservoir_buf = ReservoirBuffer(capacity=config.reservoir_buffer_size)

    opt_q  = torch.optim.Adam(q_net.parameters(),  lr=config.lr_q)
    opt_pi = torch.optim.Adam(pi_net.parameters(), lr=config.lr_pi)

    agent      = NFSPAgent(q_net, pi_net, eta=config.eta, epsilon=config.epsilon_start)
    total_steps = 0
    t0 = time.time()

    for ep in range(config.n_episodes):
        frac = min(ep / max(config.epsilon_decay_episodes, 1), 1.0)
        agent.epsilon = config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)

        terminal, transitions = _run_episode(env, agent)
        rewards = _compute_rewards(terminal, transitions, config)

        for player in [0, 1]:
            G = rewards[player]
            for state_vec, action, mode, _ in transitions[player]:
                total_steps += 1
                if mode == 'br':
                    replay_buf.add(state_vec, action, G)
                    reservoir_buf.add(state_vec, np.array(action, dtype=np.float32), 1.0)

        q_loss = pi_loss = 0.0
        if total_steps % config.update_every == 0:
            q_loss  = _update_q_net( q_net,  replay_buf,    opt_q,  config.batch_size)
            pi_loss = _update_pi_net(pi_net, reservoir_buf, opt_pi, config.batch_size)

        if (ep + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"Ep {ep+1:6d}/{config.n_episodes} | "
                  f"epsilon={agent.epsilon:.3f} | "
                  f"replay={len(replay_buf):,} reservoir={len(reservoir_buf):,} | "
                  f"q_loss={q_loss:.4f} pi_loss={pi_loss:.4f} | "
                  f"{elapsed/60:.1f}min")

        if (ep + 1) % config.checkpoint_freq == 0:
            path = f"{config.checkpoint_dir}/nfsp_ep{ep+1:06d}.pt"
            torch.save({'q_net': q_net.state_dict(),
                        'pi_net': pi_net.state_dict(),
                        'episode': ep + 1}, path)
            print(f"Saved: {path}")

    final = f"{config.checkpoint_dir}/nfsp_final.pt"
    torch.save({'q_net': q_net.state_dict(),
                'pi_net': pi_net.state_dict(),
                'episode': config.n_episodes}, final)
    print(f"Training complete -> {final}")
    return q_net, pi_net
