"""
Phase 1: Behavioral Cloning
Usage: python scripts/train_phase1_bc.py
"""
import os, sys
sys.path.insert(0, r'C:\Users\rohan\AppData\Local\Temp\ospiel_manual_build2\python')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.bc.train_bc import BCConfig, train_bc
from src.bc.validate_bc import evaluate_bc_accuracy

PARQUET_PATHS = [
    "data/processed/phh_hunl.parquet",
    "data/processed/irc_hunl.parquet",  # skipped automatically if missing
]


def main():
    existing = [p for p in PARQUET_PATHS if os.path.exists(p)]
    if not existing:
        print("No preprocessed data found. Run preprocess scripts first.")
        return

    print(f"Training BC on: {existing}")
    config = BCConfig(
        parquet_paths=existing,
        epochs=30,
        batch_size=2048,
        lr=1e-4,
        use_wandb=False,
        checkpoint_path="checkpoints/bc_final.pt",
    )
    train_bc(config)
    evaluate_bc_accuracy("checkpoints/bc_final.pt", existing)


if __name__ == "__main__":
    main()
