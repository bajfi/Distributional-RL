"""Categorical DQN (C51) model implementation."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from torch_drl.configs.default import C51_CONFIG
from torch_drl.models.base import BaseDistributionalNet


class C51Net(BaseDistributionalNet):
    """Categorical DQN (C51) network."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_atoms: int = C51_CONFIG["n_atoms"],
        v_min: float = C51_CONFIG["v_min"],
        v_max: float = C51_CONFIG["v_max"],
        hidden_dim: int = C51_CONFIG["hidden_dim"],
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.register_buffer("supports", torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * n_atoms),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        logits = self.net(state).view(batch_size, self.action_dim, self.n_atoms)
        probs = F.softmax(logits, dim=-1)
        return probs

    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> int:
        state = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
        probs = self.forward(state)
        q_values = (probs * self.supports).mean(dim=-1)
        return q_values.argmax().item()
