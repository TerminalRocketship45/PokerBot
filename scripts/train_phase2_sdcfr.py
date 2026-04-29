"""
Phase 2: SD-CFR on HUNL
Usage:
  python scripts/train_phase2_sdcfr.py --config quick
  python scripts/train_phase2_sdcfr.py --config full
  python scripts/train_phase2_sdcfr.py --config quick --bc_checkpoint checkpoints/bc_final.pt
"""
import argparse
import yaml
import os
from src.cfr.sd_cfr import SDCFRConfig, train_sd_cfr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["quick", "full"], default="quick")
    parser.add_argument("--bc_checkpoint", type=str, default=None,
                        help="Path to BC checkpoint to warm-start from")
    args = parser.parse_args()

    config_path = f"configs/{args.config}.yaml"
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    config = SDCFRConfig(**cfg_dict)

    checkpoint = args.bc_checkpoint
    if checkpoint and not os.path.exists(checkpoint):
        print(f"WARNING: BC checkpoint {checkpoint} not found. Starting from random init.")
        checkpoint = None

    print(f"Starting SD-CFR ({args.config} config)")
    print(f"  Iterations: {config.n_iterations}")
    print(f"  Traversals/iter: {config.n_traversals_per_iter}")
    print(f"  Buffer capacity: {config.buffer_capacity:,}")
    print(f"  BC checkpoint: {checkpoint or 'None (random init)'}")

    train_sd_cfr(config, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()
