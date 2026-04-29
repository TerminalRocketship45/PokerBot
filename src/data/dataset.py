import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BCDataset(Dataset):
    def __init__(self, parquet_paths: list, val_split: float = 0.1):
        frames = [pd.read_parquet(p) for p in parquet_paths if __import__('os').path.exists(p)]
        if not frames:
            raise FileNotFoundError(f"No parquet files found at: {parquet_paths}")
        df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42)

        split_idx = int(len(df) * (1 - val_split))
        self._train_df = df.iloc[:split_idx]
        self._val_df = df.iloc[split_idx:]
        self._df = self._train_df

    def use_val(self):
        self._df = self._val_df

    def use_train(self):
        self._df = self._train_df

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        row = self._df.iloc[idx]
        state = torch.FloatTensor(np.array(row["state"], copy=True))
        action = torch.tensor(int(row["action"]), dtype=torch.long)
        return state, action
