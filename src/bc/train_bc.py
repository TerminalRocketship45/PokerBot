import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import List

from src.models.advantage_net import AdvantageNet
from src.data.dataset import BCDataset
from src.data.encoder import STATE_DIM


@dataclass
class BCConfig:
    parquet_paths: List[str]
    lr: float = 1e-4
    batch_size: int = 2048
    epochs: int = 30
    weight_decay: float = 1e-5
    n_actions: int = 6
    hidden_dim: int = 256
    checkpoint_path: str = "checkpoints/bc_final.pt"
    use_wandb: bool = False
    wandb_project: str = "hunl-deep-cfr"


def train_bc(config: BCConfig) -> AdvantageNet:
    dataset = BCDataset(config.parquet_paths)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    net = AdvantageNet(input_dim=STATE_DIM, n_actions=config.n_actions, hidden_dim=config.hidden_dim)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.CrossEntropyLoss()

    if config.use_wandb:
        import wandb
        wandb.init(project=config.wandb_project, config=config.__dict__)

    for epoch in range(config.epochs):
        net.train()
        total_loss = correct = total = 0

        for states, actions in train_loader:
            logits = net(states)
            loss = criterion(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == actions).sum().item()
            total += len(actions)

        scheduler.step()
        train_acc = correct / total

        # Validation
        dataset.use_val()
        val_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        net.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for states, actions in val_loader:
                logits = net(states)
                val_correct += (logits.argmax(dim=1) == actions).sum().item()
                val_total += len(actions)
        dataset.use_train()
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(f"Epoch {epoch+1:3d}/{config.epochs} | loss={total_loss/len(train_loader):.4f} "
              f"| train_acc={train_acc:.3f} | val_acc={val_acc:.3f}")

        if config.use_wandb:
            import wandb
            wandb.log({"epoch": epoch+1, "train_acc": train_acc, "val_acc": val_acc})

    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(net.state_dict(), config.checkpoint_path)
    print(f"BC checkpoint saved: {config.checkpoint_path}")
    return net
