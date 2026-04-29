from dataclasses import dataclass, field
from typing import List
import json


@dataclass
class MetricsTracker:
    exploitability: List[float] = field(default_factory=list)
    adv_loss: List[float] = field(default_factory=list)
    h2h_vs_random: List[float] = field(default_factory=list)
    iterations: List[int] = field(default_factory=list)

    def log(self, iteration: int, exploitability: float = None,
            loss: float = None, h2h: float = None):
        self.iterations.append(iteration)
        if exploitability is not None:
            self.exploitability.append(exploitability)
        if loss is not None:
            self.adv_loss.append(loss)
        if h2h is not None:
            self.h2h_vs_random.append(h2h)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
        print(f"Metrics saved to {path}")
