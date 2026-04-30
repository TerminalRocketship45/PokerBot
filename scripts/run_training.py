"""
Full SD-CFR training with file logging.
Usage: conda run -n rl_env python scripts/run_training.py [--config quick|full]
"""
import sys
import os
import time
import argparse
import yaml
import traceback

sys.path.insert(0, r'C:\Users\rohan\AppData\Local\Temp\ospiel_manual_build2\python')
# Ensure project root is on path (when run from any directory)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "training.log")


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="quick",
                        help="Config name (quick/medium/full) or path to yaml")
    parser.add_argument("--bc_checkpoint", default=None,
                        help="Checkpoint to warm-start from (BC or prior SD-CFR run)")
    args = parser.parse_args()

    log(f"=== SD-CFR Training ({args.config}) ===")

    from src.cfr.sd_cfr import SDCFRConfig, train_sd_cfr, train_advantage_net_on_buffer
    from src.env.poker_env import PokerEnv
    from src.models.advantage_net import AdvantageNet
    from src.cfr.buffer import ReservoirBuffer
    from src.cfr.traversal import traverse
    from src.data.encoder import STATE_DIM
    import torch
    import numpy as np

    config_path = args.config if os.path.exists(args.config) else f"configs/{args.config}.yaml"
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["use_wandb"] = False  # ensure disabled
    config = SDCFRConfig(**cfg_dict)

    log(f"Config: {config}")

    env = PokerEnv(use_hunl=True)
    net = AdvantageNet(input_dim=STATE_DIM, n_actions=env.num_actions(),
                       hidden_dim=config.hidden_dim)

    if args.bc_checkpoint and os.path.exists(args.bc_checkpoint):
        net.load_state_dict(torch.load(args.bc_checkpoint, map_location="cpu"))
        log(f"Loaded BC checkpoint: {args.bc_checkpoint}")
    else:
        log("Starting from random init")

    buffer = ReservoirBuffer(capacity=config.buffer_capacity)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    t0 = time.time()

    for iteration in range(config.n_iterations):
        iter_t = time.time()
        traversal_count = 0

        for _ in range(config.n_traversals_per_iter):
            for player in [0, 1]:
                try:
                    state = env.new_game()
                    traverse(state, player, net, buffer,
                             reach_prob=1.0, use_hunl=True)
                    traversal_count += 1
                except Exception as e:
                    log(f"  ERROR in traversal: {e}")
                    log(traceback.format_exc())

        adv_loss = train_advantage_net_on_buffer(
            net, buffer, config.n_batches, config.batch_size, config.lr
        )

        elapsed = time.time() - t0
        iter_time = time.time() - iter_t
        log(f"Iter {iteration:4d} | loss={adv_loss:.4f} | "
            f"buffer={len(buffer):,} | traversals={traversal_count} | "
            f"iter_time={iter_time:.1f}s | total={elapsed/60:.1f}min")

        if (iteration + 1) % config.checkpoint_freq == 0:
            path = f"{config.checkpoint_dir}/hunl_iter_{iteration+1:04d}.pt"
            torch.save(net.state_dict(), path)
            log(f"Saved checkpoint: {path}")

    run_tag = os.path.splitext(os.path.basename(config_path))[0]
    final_path = f"{config.checkpoint_dir}/hunl_{run_tag}_final.pt"
    torch.save(net.state_dict(), final_path)
    # Also write as hunl_final.pt so other scripts always have a stable name
    torch.save(net.state_dict(), f"{config.checkpoint_dir}/hunl_final.pt")
    log(f"Training complete. Saved: {final_path} + hunl_final.pt")


if __name__ == "__main__":
    main()
