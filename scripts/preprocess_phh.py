"""Run once: python scripts/preprocess_phh.py"""
import os
import glob
import pandas as pd
from tqdm import tqdm
from src.data.parser import parse_phh_hand

RAW_DIR = "data/raw/phh"
OUT_PATH = "data/processed/phh_hunl.parquet"
MIN_STACK_BB = 20.0


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.phh"), recursive=True))
    print(f"Found {len(files)} PHH files")
    if not files:
        print(f"No .phh files in {RAW_DIR}. Download dataset first.")
        return

    try:
        from phh import PHH
    except ImportError:
        print("Install: pip install phh")
        return

    records = []
    n_total = n_filtered = n_buckets_dropped = 0

    for fpath in tqdm(files, desc="Parsing PHH"):
        try:
            hand = PHH.load(fpath)
            n_total += 1
            result = parse_phh_hand(hand, min_stack_bb=MIN_STACK_BB)
            if result is None:
                n_filtered += 1
                continue
            states, actions = result
            for s, a in zip(states, actions):
                records.append({"state": s.tolist(), "action": int(a)})
        except Exception:
            n_filtered += 1

    print(f"\nTotal hands: {n_total}")
    print(f"Filtered out: {n_filtered} ({n_filtered/max(n_total,1):.1%})")
    print(f"Training records: {len(records)}")

    if not records:
        print("No records produced. Check parser.")
        return

    df = pd.DataFrame(records)
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
