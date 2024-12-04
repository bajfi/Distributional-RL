import pytest
import torch
from torch_drl.agents.c51 import C51Agent
from torch_drl.configs.default import C51_CONFIG


@pytest.fixture
def c51_agent():
    state_dim = 4
    action_dim = 2
    device = torch.device("cpu")  # Use CPU for testing
    return C51Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        n_atoms=C51_CONFIG["n_atoms"],
        v_min=C51_CONFIG["v_min"],
        v_max=C51_CONFIG["v_max"],
    )


def test_project_distribution_shape(c51_agent):
    batch_size = 32
    next_distr = (
        torch.ones((batch_size, c51_agent.n_atoms)) / c51_agent.n_atoms
    )  # Uniform distribution
    rewards = torch.zeros(batch_size)
    dones = torch.zeros(batch_size)

    projected = c51_agent._project_distribution(next_distr, rewards, dones)

    assert projected.shape == (batch_size, c51_agent.n_atoms)
    assert torch.allclose(projected.sum(dim=1), torch.ones(batch_size))


def test_project_distribution_terminal_states(c51_agent):
    batch_size = 32
    next_distr = torch.ones((batch_size, c51_agent.n_atoms)) / c51_agent.n_atoms
    rewards = torch.ones(batch_size)  # All rewards are 1
    dones = torch.ones(batch_size)  # All states are terminal

    projected = c51_agent._project_distribution(next_distr, rewards, dones)

    # For terminal states with reward 1, the distribution should be concentrated
    # around the immediate reward (1) without future discounting
    expected_atom_index = (
        torch.tensor((1.0 - c51_agent.v_min) / c51_agent.delta_z).floor().long()
    )
    max_prob_indices = torch.argmax(projected, dim=1)
    assert torch.all(max_prob_indices == expected_atom_index)


def test_project_distribution_zero_reward_continuing(c51_agent):
    batch_size = 32
    next_distr = torch.zeros((batch_size, c51_agent.n_atoms))
    # Put all probability mass on the middle atom
    middle_atom = c51_agent.n_atoms // 2
    next_distr[:, middle_atom] = 1.0

    rewards = torch.zeros(batch_size)
    dones = torch.zeros(batch_size)

    projected = c51_agent._project_distribution(next_distr, rewards, dones)

    # Check that probabilities sum to 1
    assert torch.allclose(projected.sum(dim=1), torch.ones(batch_size))
    # Check that the distribution is not degenerate
    assert not torch.allclose(projected, torch.zeros_like(projected))


def test_project_distribution_value_bounds(c51_agent):
    batch_size = 32
    next_distr = torch.ones((batch_size, c51_agent.n_atoms)) / c51_agent.n_atoms
    large_reward = torch.ones(batch_size) * c51_agent.v_max * 2  # Very large reward
    dones = torch.zeros(batch_size)

    projected = c51_agent._project_distribution(next_distr, large_reward, dones)

    # Check that all values are properly bounded
    support = torch.linspace(c51_agent.v_min, c51_agent.v_max, c51_agent.n_atoms)
    expected_max = c51_agent.v_max
    actual_max = (projected * support).sum(dim=1).max()
    assert actual_max <= expected_max
