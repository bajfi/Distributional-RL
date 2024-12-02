"""Base agent class for Distributional RL agents."""

from typing import Dict

import numpy as np
import torch


class BaseAgent:
    """Base class for distributional RL agents."""

    def __init__(self):
        pass

    def choose_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        raise NotImplementedError

    def train(self, batch: Dict[str, torch.Tensor]) -> float:
        raise NotImplementedError

    def update_target(self) -> None:
        raise NotImplementedError
