"""Replay buffer implementation for Distributional RL agents."""

from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch

from torch_drl.configs.default import DEVICE, MEMORY_SIZE


class ReplayBuffer:
    """Experience replay buffer for RL agents."""

    def __init__(self, maxlen: int = MEMORY_SIZE):
        """Initialize replay buffer.

        Args:
            maxlen (int): Maximum size of the buffer
        """
        self.buffer = deque(maxlen=maxlen)

    def append(
        self, experience: Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]
    ) -> None:
        """Add experience to buffer.

        Args:
            experience (tuple): (state, next_state, action, reward, done)
        """
        self.buffer.append(experience)

    def sample(
        self, batch_size: int, device: torch.device = DEVICE
    ) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences.

        Args:
            batch_size (int): Size of batch to sample
            device (torch.device): Device to put tensors on

        Returns:
            dict: Dictionary containing batched experiences
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, next_states, actions, rewards, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )

        # Convert to torch tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        # One-hot encode actions
        actions_one_hot = torch.zeros(
            batch_size, actions.max().item() + 1, device=device
        )
        actions_one_hot.scatter_(1, actions.unsqueeze(1), 1)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)

        return {
            "states": states,
            "next_states": next_states,
            "actions": actions_one_hot,
            "rewards": rewards,
            "dones": dones,
        }

    def __len__(self) -> int:
        return len(self.buffer)
