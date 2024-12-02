"""DQN agent implementation."""

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch_drl.agents.base import BaseAgent
from torch_drl.configs.default import DEVICE, DQN_CONFIG, GAMMA, LEARNING_RATE
from torch_drl.models.dqn import DQN


class DQNAgent(BaseAgent):
    """Standard DQN Agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        device: torch.device = DEVICE,
        hidden_dim: int = DQN_CONFIG["hidden_dim"],
    ):
        super().__init__()
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)

        # Create networks
        self.online_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
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

        # Get current Q values
        current_q = self.online_net(states)
        current_q = torch.sum(current_q * actions, dim=-1)

        # Calculate target Q values with Double DQN
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states)
            next_q = next_q[torch.arange(next_q.size(0)), next_actions]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Calculate loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())
