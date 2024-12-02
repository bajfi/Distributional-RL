"""Implicit Quantile Network (IQN) model implementation."""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_drl.configs.default import IQN_CONFIG
from torch_drl.models.base import BaseDistributionalNet


class IQNNet(BaseDistributionalNet):
    """Implicit Quantile Network implementation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_quantiles: int = IQN_CONFIG["n_quantiles"],
        n_cos_embeddings: int = IQN_CONFIG["n_cos_embeddings"],
        hidden_dim: int = IQN_CONFIG["hidden_dim"],
    ):
        """Initialize IQN Network.

        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            n_quantiles (int): Number of quantile samples
            n_cos_embeddings (int): Number of cosine embeddings
            hidden_dim (int): Hidden layer dimension
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
        self.n_cos_embeddings = n_cos_embeddings

        # State encoder
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Cosine embedding network
        self.cos_embedding = nn.Linear(n_cos_embeddings, hidden_dim)

        # Final network
        self.final_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def calc_cos_embedding(self, tau: torch.Tensor) -> torch.Tensor:
        """Calculate cosine embedding for given quantiles.

        Args:
            tau (torch.Tensor): Sampled quantiles

        Returns:
            torch.Tensor: Cosine embeddings
        """
        batch_size = tau.shape[0]
        n_tau = tau.shape[1]

        # Calculate cos(pi * i * tau) for i=1,...,n_cos_embeddings
        i_pi = torch.arange(1, self.n_cos_embeddings + 1, device=tau.device) * np.pi
        cos_tau = torch.cos(tau.unsqueeze(-1) * i_pi.view(1, 1, -1))

        return cos_tau.view(batch_size * n_tau, self.n_cos_embeddings)

    def forward(
        self, state: torch.Tensor, tau: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of IQN.

        Args:
            state (torch.Tensor): State tensor
            tau (torch.Tensor, optional): Quantile values. If None, will sample.

        Returns:
            tuple: (quantile values, sampled taus)
        """
        batch_size = state.shape[0]

        # Sample tau if not provided
        if tau is None:
            tau = torch.rand(batch_size, self.n_quantiles, device=state.device)
        n_tau = tau.shape[1]

        # Get state features
        state_features = self.state_net(state)  # [batch_size, hidden_dim]
        state_features = state_features.repeat_interleave(n_tau, dim=0)

        # Get quantile embedding
        cos_embedding = self.calc_cos_embedding(
            tau
        )  # [batch_size * n_tau, n_cos_embeddings]
        tau_embedding = F.relu(
            self.cos_embedding(cos_embedding)
        )  # [batch_size * n_tau, hidden_dim]

        # Combine state and quantile embeddings
        combined = state_features * tau_embedding

        # Get quantile values for each action
        quantiles = self.final_net(combined)  # [batch_size * n_tau, action_dim]
        quantiles = quantiles.view(batch_size, n_tau, self.action_dim)

        return quantiles, tau

    @torch.no_grad()
    def get_action(self, state: np.ndarray) -> int:
        """Get action for given state.

        Args:
            state (np.ndarray): State array

        Returns:
            int: Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
        quantiles, _ = self.forward(state)
        expected_values = quantiles.mean(dim=1)
        return expected_values.argmax().item()
