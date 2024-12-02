"""DQN model implementation."""

import numpy as np
import torch
import torch.nn as nn

from torch_drl.configs.default import DQN_CONFIG
from torch_drl.models.base import BaseDistributionalNet


class DQN(BaseDistributionalNet):
    """Standard DQN network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = DQN_CONFIG["hidden_dim"],
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> int:
        state = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
        q_values = self.forward(state)
        return q_values.argmax().item()
