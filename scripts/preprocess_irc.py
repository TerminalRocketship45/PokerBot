"""Run once: python scripts/preprocess_irc.py"""
import os
import glob
import pandas as pd
from tqdm import tqdm

RAW_DIR = "data/raw/irc"
OUT_PATH = "data/processed/irc_hunl.parquet"
MIN_NLHE_HANDS = 100_000


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.txt"), recursive=True))
    files += sorted(glob.glob(os.path.join(RAW_DIR, "**", "hdb"), recursive=True))
    print(f"Found {len(files)} IRC files")

    n_total = n_nlhe = n_other = 0
    records = []

    for fpath in tqdm(files, desc="Parsing IRC"):
        try:
            with open(fpath, "r", errors="ignore") as f:
                content = f.read()
            for hand_block in content.split("\n\n"):
                if not hand_block.strip():
                    continue
                n_total += 1
                if "nolimit" not in hand_block.lower():
                    n_other += 1
                    continue
                n_nlhe += 1
                # IRC parser stub — extend with full parsing as needed
        except Exception:
            pass

    print(f"\nTotal IRC hands seen: {n_total}")
    print(f"Non-NLHE (dropped): {n_other}")
    print(f"NLHE hands survived filter: {n_nlhe}")

    if n_nlhe < MIN_NLHE_HANDS:
        print(f"WARNING: Only {n_nlhe} NLHE hands < threshold {MIN_NLHE_HANDS}.")
        print("Skipping IRC. BC training will use PHH only.")
        return

    if records:
        df = pd.DataFrame(records)
        os.makedirs("data/processed", exist_ok=True)
        df.to_parquet(OUT_PATH, index=False)
        print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
