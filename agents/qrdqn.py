"""Quantile Regression DQN (QR-DQN) agent implementation."""

from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from torch_drl.agents.base import BaseAgent
from torch_drl.configs.default import DEVICE, GAMMA, LEARNING_RATE, QRDQN_CONFIG
from torch_drl.models.qrdqn import QRDQNNet


class QRDQNAgent(BaseAgent):
    """Quantile Regression DQN Agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        device: str = DEVICE,
        n_quantiles: int = QRDQN_CONFIG["n_quantiles"],
        hidden_dim: int = QRDQN_CONFIG["hidden_dim"],
    ):
        super().__init__()
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)
        self.n_quantiles = n_quantiles

        # Create networks
        self.online_net = QRDQNNet(state_dim, action_dim, n_quantiles, hidden_dim).to(
            self.device
        )
        self.target_net = QRDQNNet(state_dim, action_dim, n_quantiles, hidden_dim).to(
            self.device
        )
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

    def choose_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        return self.online_net.get_action(state)

    def _huber_loss(self, td_errors: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        """Calculate Huber loss for QR-DQN."""
        return torch.where(
            td_errors.abs() <= kappa,
            0.5 * td_errors.pow(2),
            kappa * (td_errors.abs() - 0.5 * kappa),
        )

    def train(self, batch: Dict[str, torch.Tensor]) -> float:
        states = batch["states"]
        next_states = batch["next_states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # Get current quantiles
        current_quantiles = self.online_net(states)
        # Convert one-hot actions to indices
        action_indices = actions.max(1)[1] if actions.dim() > 1 else actions
        current_quantiles = current_quantiles[
            torch.arange(len(states)), action_indices.long()
        ]

        # Get target quantiles
        with torch.no_grad():
            next_quantiles = self.target_net(next_states)
            next_actions = next_quantiles.mean(2).argmax(1)
            next_quantiles = next_quantiles[torch.arange(len(states)), next_actions]

            # Calculate target quantiles
            target_quantiles = (
                rewards.unsqueeze(1)
                + (1 - dones.unsqueeze(1)) * self.gamma * next_quantiles
            )

        # Calculate quantile regression loss
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        huber_loss = self._huber_loss(td_errors)

        # Calculate quantile weights with proper broadcasting
        tau = self.online_net.taus.view(1, -1, 1)  # [1, N, 1]
        below_threshold = (td_errors < 0).float()  # [B, N, N]
        quantile_loss = torch.abs(tau - below_threshold) * huber_loss / self.n_quantiles
        loss = quantile_loss.sum(dim=(1, 2)).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())
