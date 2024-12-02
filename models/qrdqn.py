"""Quantile Regression DQN (QR-DQN) model implementation."""

import numpy as np
import torch
import torch.nn as nn

from torch_drl.configs.default import QRDQN_CONFIG
from torch_drl.models.base import BaseDistributionalNet


class QRDQNNet(BaseDistributionalNet):
    """Quantile Regression DQN network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = QRDQN_CONFIG["n_quantiles"],
        hidden_dim: int = QRDQN_CONFIG["hidden_dim"],
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * n_quantiles),
        )

        # Initialize quantile fractions (τ)
        taus = (
            torch.arange(0, n_quantiles + 1, device="cpu", dtype=torch.float32)
            / n_quantiles
        )
        self.register_buffer("taus", (taus[1:] + taus[:-1]) / 2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        quantiles = self.net(state).view(batch_size, self.action_dim, self.n_quantiles)
        return quantiles

    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> int:
        state = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
        quantiles = self.forward(state)
        q_values = quantiles.mean(dim=-1)
        return q_values.argmax().item()
