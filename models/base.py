"""Base model class for Distributional RL agents."""

import numpy as np
import torch
import torch.nn as nn


class BaseDistributionalNet(nn.Module):
    """Base class for distributional RL networks."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_action(self, state: np.ndarray) -> int:
        raise NotImplementedError
