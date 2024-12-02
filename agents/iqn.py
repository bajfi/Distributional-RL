"""Implicit Quantile Network (IQN) agent implementation."""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_drl.agents.base import BaseAgent
from torch_drl.configs.default import DEVICE, GAMMA, IQN_CONFIG, LEARNING_RATE
from torch_drl.models.iqn import IQNNet


class IQNAgent(BaseAgent):
    """Implicit Quantile Networks (IQN) agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        device: torch.device = DEVICE,
        hidden_dim: int = IQN_CONFIG["hidden_dim"],
        n_quantiles: int = IQN_CONFIG["n_quantiles"],
        n_cos_embeddings: int = IQN_CONFIG["n_cos_embeddings"],
    ):
        super().__init__()
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)

        # Create networks
        self.online_net = IQNNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_quantiles=n_quantiles,
            n_cos_embeddings=n_cos_embeddings,
        ).to(self.device)

        self.target_net = IQNNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_quantiles=n_quantiles,
            n_cos_embeddings=n_cos_embeddings,
        ).to(self.device)

        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

    def choose_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        return self.online_net.get_action(state)

    def train(self, batch: Dict[str, torch.Tensor]) -> float:
        states = batch["states"]
        next_states = batch["next_states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # Calculate current quantile values
        current_quantiles, tau = self.online_net(states)
        actions_one_hot = actions.long()
        current_quantiles = torch.sum(
            current_quantiles * actions_one_hot.unsqueeze(1), dim=-1
        )

        # Calculate target quantile values
        with torch.no_grad():
            # Get greedy actions from online network
            next_quantiles, _ = self.online_net(next_states)
            next_actions = next_quantiles.mean(dim=1).argmax(dim=-1)
            next_actions_one_hot = F.one_hot(next_actions, self.action_dim).float()

            # Get target quantile values
            target_quantiles, _ = self.target_net(next_states)
            target_quantiles = torch.sum(
                target_quantiles * next_actions_one_hot.unsqueeze(1), dim=-1
            )

            # Calculate target values
            target_quantiles = (
                rewards.unsqueeze(1)
                + (1 - dones.unsqueeze(1)) * self.gamma * target_quantiles
            )

        # Calculate quantile huber loss
        diff = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        huber_loss = torch.where(diff.abs() <= 1.0, 0.5 * diff.pow(2), diff.abs() - 0.5)

        quantile_loss = torch.abs(tau.unsqueeze(2) - (diff < 0).float()) * huber_loss
        loss = quantile_loss.mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())
