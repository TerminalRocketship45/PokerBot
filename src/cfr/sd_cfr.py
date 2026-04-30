import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

from src.env.poker_env import PokerEnv
from src.models.advantage_net import AdvantageNet
from src.cfr.buffer import ReservoirBuffer
from src.cfr.traversal import traverse


@dataclass
class SDCFRConfig:
    use_hunl: bool = False
    n_iterations: int = 50
    n_traversals_per_iter: int = 100
    buffer_capacity: int = 50_000
    n_batches: int = 50
    batch_size: int = 256
    hidden_dim: int = 128
    lr: float = 1e-4
    eval_freq: int = 10
    checkpoint_freq: int = 10
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
    wandb_project: str = "hunl-deep-cfr"


def train_advantage_net_on_buffer(
    net: AdvantageNet,
    buffer: ReservoirBuffer,
    n_batches: int,
    batch_size: int,
    lr: float,
) -> float:
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    net.train()
    total_loss = 0.0
    for _ in range(n_batches):
        if len(buffer) < batch_size:
            break
        batch = buffer.sample(batch_size)
        states, advantages, weights = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        advantages_t = torch.FloatTensor(np.array(advantages))
        weights_t = torch.FloatTensor(np.array(weights))

        pred = net(states_t)
        per_sample_loss = ((pred - advantages_t) ** 2).mean(dim=1)
        loss = (weights_t * per_sample_loss).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(n_batches, 1)


def train_sd_cfr(
    config: SDCFRConfig,
    checkpoint_path: Optional[str] = None,
) -> AdvantageNet:
    env = PokerEnv(use_hunl=config.use_hunl)
    from src.data.encoder import STATE_DIM
    state_dim = STATE_DIM  # always 60 — encode_state() always pads to STATE_DIM
    n_actions = env.num_actions()

    net = AdvantageNet(
        input_dim=state_dim,
        n_actions=n_actions,
        hidden_dim=config.hidden_dim,
    )

    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded checkpoint: {checkpoint_path}")

    buffer = ReservoirBuffer(capacity=config.buffer_capacity)

    if config.use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, config=config.__dict__)

    import os
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for iteration in range(config.n_iterations):
        for _ in range(config.n_traversals_per_iter):
            for player in [0, 1]:
                state = env.new_game()
                traverse(
                    state, player, net, buffer, reach_prob=1.0,
                    use_hunl=config.use_hunl,
                )

        adv_loss = train_advantage_net_on_buffer(
            net, buffer, config.n_batches, config.batch_size, config.lr
        )

        print(f"Iter {iteration:4d} | loss={adv_loss:.4f} | buffer={len(buffer)}")

        if config.use_wandb:
            import wandb
            wandb.log({"iteration": iteration, "adv_loss": adv_loss,
                       "buffer_size": len(buffer)})

        if (iteration + 1) % config.checkpoint_freq == 0:
            path = f"{config.checkpoint_dir}/iter_{iteration+1:04d}.pt"
            torch.save(net.state_dict(), path)
            print(f"Saved checkpoint: {path}")


    return net
