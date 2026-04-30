# scripts/train_nfsp.py
"""
Phase 3: NFSP training. Loads BC checkpoint and trains fold-aware agent.
Usage: python scripts/train_nfsp.py [--config nfsp] [--bc_checkpoint checkpoints/bc_final.pt]
"""
import sys, os, argparse, yaml

sys.path.insert(0, r'C:\Users\rohan\AppData\Local\Temp\ospiel_manual_build2\python')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="nfsp")
    parser.add_argument("--bc_checkpoint", default="checkpoints/bc_final.pt")
    args = parser.parse_args()

    config_path = args.config if os.path.exists(args.config) else f"configs/{args.config}.yaml"
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    from src.nfsp.train_nfsp import NFSPConfig, train_nfsp
    config = NFSPConfig(**cfg_dict)
    print(f"NFSP training: {config.n_episodes} episodes, eta={config.eta}, "
          f"epsilon {config.epsilon_start}->{config.epsilon_end}")
    train_nfsp(config, bc_checkpoint=args.bc_checkpoint)


if __name__ == "__main__":
    main()
