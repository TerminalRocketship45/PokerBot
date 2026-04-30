"""
Full two-phase training pipeline:
  Phase 1: Behavioral Cloning on IRC data → checkpoints/bc_final.pt
  Phase 2: SD-CFR self-play starting from BC weights → checkpoints/hunl_final.pt

Usage:
  python scripts/train_full_pipeline.py [--config medium] [--skip_bc]
"""
import sys, os, argparse, time, traceback

sys.path.insert(0, r'C:\Users\rohan\AppData\Local\Temp\ospiel_manual_build2\python')
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
os.chdir(_root)

LOG_FILE = os.path.join(_root, "logs", "pipeline.log")


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_bc():
    from src.bc.train_bc import BCConfig, train_bc
    from src.bc.validate_bc import evaluate_bc_accuracy

    parquet_paths = [p for p in [
        "data/processed/irc_hunl.parquet",
        "data/processed/phh_hunl.parquet",
    ] if os.path.exists(p)]

    if not parquet_paths:
        log("ERROR: No processed data found. Run preprocess_irc.py first.")
        return False

    log(f"Phase 1 — Behavioral Cloning on: {parquet_paths}")
    config = BCConfig(
        parquet_paths=parquet_paths,
        epochs=30,
        batch_size=2048,
        lr=1e-4,
        use_wandb=False,
        checkpoint_path="checkpoints/bc_final.pt",
    )
    train_bc(config)
    evaluate_bc_accuracy("checkpoints/bc_final.pt", parquet_paths)
    log("Phase 1 complete → checkpoints/bc_final.pt")
    return True


def run_sdcfr(config_name: str, bc_checkpoint: str):
    import yaml, torch, numpy as np
    from src.cfr.sd_cfr import SDCFRConfig, train_advantage_net_on_buffer
    from src.env.poker_env import PokerEnv
    from src.models.advantage_net import AdvantageNet
    from src.cfr.buffer import ReservoirBuffer
    from src.cfr.traversal import traverse
    from src.data.encoder import STATE_DIM

    config_path = config_name if os.path.exists(config_name) else f"configs/{config_name}.yaml"
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict["use_wandb"] = False
    config = SDCFRConfig(**cfg_dict)

    log(f"Phase 2 — SD-CFR ({config_name}): {config.n_iterations} iters × "
        f"{config.n_traversals_per_iter} traversals")

    env = PokerEnv(use_hunl=True)
    net = AdvantageNet(input_dim=STATE_DIM, n_actions=env.num_actions(),
                       hidden_dim=config.hidden_dim)

    if bc_checkpoint and os.path.exists(bc_checkpoint):
        net.load_state_dict(torch.load(bc_checkpoint, map_location="cpu"))
        log(f"Loaded BC weights: {bc_checkpoint}")
    else:
        log("WARNING: No BC checkpoint found, starting from random init")

    buffer = ReservoirBuffer(capacity=config.buffer_capacity)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    t0 = time.time()

    for iteration in range(config.n_iterations):
        iter_t = time.time()
        n_traversals = 0

        for _ in range(config.n_traversals_per_iter):
            for player in [0, 1]:
                try:
                    state = env.new_game()
                    traverse(state, player, net, buffer, reach_prob=1.0, use_hunl=True)
                    n_traversals += 1
                except Exception as e:
                    log(f"  Traversal error: {e}\n{traceback.format_exc()}")

        adv_loss = train_advantage_net_on_buffer(
            net, buffer, config.n_batches, config.batch_size, config.lr
        )

        elapsed = time.time() - t0
        log(f"Iter {iteration+1:4d}/{config.n_iterations} | loss={adv_loss:.4f} | "
            f"buffer={len(buffer):,} | traversals={n_traversals} | "
            f"iter_time={time.time()-iter_t:.1f}s | total={elapsed/60:.1f}min")

        if (iteration + 1) % config.checkpoint_freq == 0:
            path = f"{config.checkpoint_dir}/hunl_iter_{iteration+1:04d}.pt"
            torch.save(net.state_dict(), path)
            log(f"Saved: {path}")

    run_tag = os.path.splitext(os.path.basename(config_path))[0]
    final_path = f"{config.checkpoint_dir}/hunl_{run_tag}_final.pt"
    torch.save(net.state_dict(), final_path)
    torch.save(net.state_dict(), f"{config.checkpoint_dir}/hunl_final.pt")
    log(f"Phase 2 complete → {final_path} + hunl_final.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="medium")
    parser.add_argument("--skip_bc", action="store_true",
                        help="Skip BC phase (use existing bc_final.pt)")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    if not args.skip_bc:
        ok = run_bc()
        if not ok:
            return
    else:
        log("Skipping BC phase (--skip_bc)")

    run_sdcfr(args.config, bc_checkpoint="checkpoints/bc_final.pt")


if __name__ == "__main__":
    main()
