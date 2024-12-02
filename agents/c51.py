"""Categorical DQN (C51) agent implementation."""

from typing import Dict

import numpy as np
import torch
import torch.optim as optim

from torch_drl.agents.base import BaseAgent
from torch_drl.configs.default import C51_CONFIG, DEVICE, GAMMA, LEARNING_RATE
from torch_drl.models.c51 import C51Net


class C51Agent(BaseAgent):
    """Categorical DQN (C51) Agent."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = LEARNING_RATE,
        gamma: float = GAMMA,
        device: torch.device = DEVICE,
        n_atoms: int = C51_CONFIG["n_atoms"],
        v_min: float = C51_CONFIG["v_min"],
        v_max: float = C51_CONFIG["v_max"],
        hidden_dim: int = C51_CONFIG["hidden_dim"],
    ):
        super().__init__()
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = torch.device(device)
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Create networks
        self.online_net = C51Net(
            state_dim, action_dim, n_atoms, v_min, v_max, hidden_dim
        ).to(self.device)
        self.target_net = C51Net(
            state_dim, action_dim, n_atoms, v_min, v_max, hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)

    def choose_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        return self.online_net.get_action(state)

    def _project_distribution(
        self, next_distr: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Project the distribution for C51."""
        batch_size = len(rewards)

        # Project using the Bellman update
        proj_distr = torch.zeros((batch_size, self.n_atoms), device=self.device)
        delta_z = self.delta_z
        v_min, v_max = self.v_min, self.v_max

        # Compute projection
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        support = self.online_net.supports

        target_z = rewards + (1 - dones) * self.gamma * support
        target_z = target_z.clamp(v_min, v_max)

        b = (target_z - v_min) / delta_z
        low = b.floor().long()
        high = b.ceil().long()

        # Handle corner cases
        low[(high > 0) * (low == high)] -= 1
        high[(low < (self.n_atoms - 1)) * (low == high)] += 1

        # Distribute probability
        offset = (
            torch.linspace(
                0, ((batch_size - 1) * self.n_atoms), batch_size, device=self.device
            )
            .long()
            .unsqueeze(1)
            .expand(batch_size, self.n_atoms)
        )

        proj_distr.view(-1).index_add_(
            0, (low + offset).view(-1), (next_distr * (high.float() - b)).view(-1)
        )
        proj_distr.view(-1).index_add_(
            0, (high + offset).view(-1), (next_distr * (b - low.float())).view(-1)
        )

        return proj_distr

    def train(self, batch: Dict[str, torch.Tensor]) -> float:
        states = batch["states"]
        next_states = batch["next_states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]

        # Get current distribution
        current_distr = self.online_net(states)
        action_indices = actions.argmax(dim=1)
        current_distr = current_distr[torch.arange(len(states)), action_indices]

        # Get target distribution
        with torch.no_grad():
            next_distr = self.target_net(next_states)
            next_actions = (next_distr * self.online_net.supports).sum(2).argmax(1)
            next_distr = next_distr[torch.arange(len(states)), next_actions]

            # Project next distribution
            target_distr = self._project_distribution(next_distr, rewards, dones)

        # Calculate cross-entropy loss
        loss = -(target_distr * torch.log(current_distr + 1e-8)).sum(1).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())
