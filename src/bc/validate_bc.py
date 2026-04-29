import torch
from torch.utils.data import DataLoader
from src.models.advantage_net import AdvantageNet
from src.data.dataset import BCDataset
from src.data.encoder import STATE_DIM


def evaluate_bc_accuracy(checkpoint_path: str, parquet_paths: list) -> float:
    dataset = BCDataset(parquet_paths)
    dataset.use_val()
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)

    net = AdvantageNet(input_dim=STATE_DIM, n_actions=6, hidden_dim=256)
    net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    net.eval()

    correct = total = 0
    with torch.no_grad():
        for states, actions in loader:
            logits = net(states)
            correct += (logits.argmax(dim=1) == actions).sum().item()
            total += len(actions)

    acc = correct / total if total > 0 else 0.0
    print(f"BC validation accuracy: {acc:.3f} ({correct}/{total})")
    print(f"Random baseline: {1/6:.3f}")
    if acc < 0.40:
        print("WARNING: accuracy below 40% target. Consider more epochs or data.")
    return acc
