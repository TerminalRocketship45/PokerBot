# src/nfsp/replay_buffer.py
import numpy as np
from typing import List, Tuple


class ReplayBuffer:
    """
    Fixed-size circular buffer for DQN experience tuples.
    Stores (state, action, G) where G is the Monte Carlo return for the episode.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._states  = np.zeros((capacity, 60), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._returns = np.zeros(capacity, dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def add(self, state: np.ndarray, action: int, G: float) -> None:
        self._states[self._ptr]  = state
        self._actions[self._ptr] = action
        self._returns[self._ptr] = G
        self._ptr  = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> List[Tuple[np.ndarray, int, float]]:
        if self._size == 0:
            raise ValueError("ReplayBuffer is empty")
        idx = np.random.choice(self._size, size=min(batch_size, self._size), replace=False)
        return [
            (self._states[i].copy(), int(self._actions[i]), float(self._returns[i]))
            for i in idx
        ]

    def __len__(self) -> int:
        return self._size
