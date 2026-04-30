import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BCDataset(Dataset):
    """
    Loads (state, action) pairs from .npz files (preferred) or .parquet fallback.
    Stores everything as contiguous numpy arrays to minimize RAM usage.
    """

    def __init__(self, paths: list, val_split: float = 0.1):
        all_states  = []
        all_actions = []

        for p in paths:
            if not os.path.exists(p):
                continue
            npz = p.replace('.parquet', '.npz')
            if os.path.exists(npz):
                d = np.load(npz)
                all_states.append(d['states'])
                all_actions.append(d['actions'])
            elif p.endswith('.parquet'):
                import pandas as pd
                df = pd.read_parquet(p)
                all_states.append(np.stack(df['state'].values).astype(np.float32))
                all_actions.append(df['action'].values.astype(np.int64))

        if not all_states:
            raise FileNotFoundError(f"No data found at: {paths}")

        states  = np.concatenate(all_states,  axis=0)
        actions = np.concatenate(all_actions, axis=0)

        # Shuffle
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(states))
        states  = states[idx]
        actions = actions[idx]

        split = int(len(states) * (1 - val_split))
        self._train_s = states[:split]
        self._train_a = actions[:split]
        self._val_s   = states[split:]
        self._val_a   = actions[split:]
        self._mode    = 'train'

    def use_val(self):
        self._mode = 'val'

    def use_train(self):
        self._mode = 'train'

    def __len__(self):
        return len(self._train_s) if self._mode == 'train' else len(self._val_s)

    def __getitem__(self, idx):
        if self._mode == 'train':
            return torch.from_numpy(self._train_s[idx]), torch.tensor(int(self._train_a[idx]))
        return torch.from_numpy(self._val_s[idx]), torch.tensor(int(self._val_a[idx]))
