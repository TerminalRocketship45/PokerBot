import numpy as np
from typing import List, Tuple


class ReservoirBuffer:
    """Reservoir-sampled buffer for SD-CFR advantage data. Never use a FIFO queue instead."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.n_seen = 0

    def add(self, state: np.ndarray, advantages: np.ndarray, weight: float):
        self.n_seen += 1
        entry = (state.copy(), advantages.copy(), weight)
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            idx = np.random.randint(0, self.n_seen)
            if idx < self.capacity:
                self.buffer[idx] = entry

    def sample(self, batch_size: int) -> List[Tuple]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)
