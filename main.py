"""Main training script for Distributional RL agents."""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Type

import gymnasium as gym
import numpy as np
import torch
import typer
from rich.console import Console
from rich.table import Table

from torch_drl.agents.base import BaseAgent
from torch_drl.agents.c51 import C51Agent
from torch_drl.agents.dqn import DQNAgent
from torch_drl.agents.iqn import IQNAgent
from torch_drl.agents.qrdqn import QRDQNAgent
from torch_drl.configs.default import (
    AGENT_TYPE,
    BATCH_SIZE,
    DEVICE,
    ENV_NAME,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    EVAL_FREQ,
    GAMMA,
    LEARNING_RATE,
    LOG_DIR,
    MAX_EPISODES,
    MEMORY_SIZE,
    MIN_MEMORY_SIZE,
    N_EVAL_EPISODES,
    REWARD_SCALING,
    TARGET_UPDATE_FREQ,
)
from torch_drl.utils.logger import TrainingLogger
from torch_drl.utils.memory import ReplayBuffer
from torch_drl.utils.visualization import plot_training_rewards

# Initialize Typer app
app = typer.Typer(help="Train and evaluate Distributional RL agents.")
console = Console()


class AgentType(str, Enum):
    """Available agent types."""

    DQN = "dqn"
    C51 = "c51"
    QRDQN = "qrdqn"
    IQN = "iqn"


def get_agent_class(agent_type: AgentType) -> Type[BaseAgent]:
    """Get agent class based on type.

    Args:
        agent_type (AgentType): Type of agent to use

    Returns:
        Type[BaseAgent]: Agent class
    """
    agents: Dict[AgentType, Type[BaseAgent]] = {
        AgentType.DQN: DQNAgent,
        AgentType.C51: C51Agent,
        AgentType.QRDQN: QRDQNAgent,
        AgentType.IQN: IQNAgent,
    }
    return agents[agent_type]


def print_training_summary(
    episode_rewards: List[float],
    eval_rewards: List[float],
    best_episode_reward: float,
    best_eval_reward: float,
    total_steps: int,
    training_time: float,
    log_dir: Path,
) -> None:
    """Print a summary table of training metrics.

    Args:
        episode_rewards (List[float]): List of episode rewards
        eval_rewards (List[float]): List of evaluation rewards
        best_episode_reward (float): Best episode reward achieved
        best_eval_reward (float): Best evaluation reward achieved
        total_steps (int): Total steps taken
        training_time (float): Total training time in seconds
        log_dir (Path): Directory where models are saved
    """
    # Create summary table
    table = Table(title="Training Summary", box=None)

    # Add columns
    table.add_column("Metric", style="cyan", justify="right")
    table.add_column("Value", style="green")

    # Calculate metrics
    last_100_avg = (
        np.mean(episode_rewards[-100:])
        if len(episode_rewards) >= 100
        else np.mean(episode_rewards)
    )
    last_100_std = (
        np.std(episode_rewards[-100:])
        if len(episode_rewards) >= 100
        else np.std(episode_rewards)
    )
    overall_avg = np.mean(episode_rewards)
    overall_std = np.std(episode_rewards)
    eval_avg = np.mean(eval_rewards) if eval_rewards else 0
    eval_std = np.std(eval_rewards) if eval_rewards else 0

    # Add rows
    table.add_row("Total Episodes", str(len(episode_rewards)))
    table.add_row("Total Steps", str(total_steps))
    table.add_row("Training Time", f"{training_time:.2f} seconds")
    table.add_row("Best Episode Reward", f"{best_episode_reward:.2f}")
    table.add_row("Best Evaluation Reward", f"{best_eval_reward:.2f}")
    table.add_row(
        "Last 100 Episodes Avg ± Std", f"{last_100_avg:.2f} ± {last_100_std:.2f}"
    )
    table.add_row("Overall Average ± Std", f"{overall_avg:.2f} ± {overall_std:.2f}")
    table.add_row("Evaluation Average ± Std", f"{eval_avg:.2f} ± {eval_std:.2f}")
    table.add_row("Best Models Saved At", str(log_dir))

    # Print table
    console.print("\n")
    console.print(table)
    console.print("\n")


def train(
    env_name: str = ENV_NAME,
    agent_type: AgentType = AgentType.DQN,
    episodes: int = MAX_EPISODES,
    learning_rate: float = LEARNING_RATE,
    gamma: float = GAMMA,
    epsilon_start: float = EPSILON_START,
    epsilon_end: float = EPSILON_END,
    epsilon_decay: int = EPSILON_DECAY,
    memory_size: int = MEMORY_SIZE,
    batch_size: int = BATCH_SIZE,
    target_update_freq: int = TARGET_UPDATE_FREQ,
    eval_freq: int = EVAL_FREQ,
    n_eval_episodes: int = N_EVAL_EPISODES,
    reward_scaling: float = REWARD_SCALING,
    min_memory_size: int = MIN_MEMORY_SIZE,
    device: str = DEVICE,
    log_dir: str = LOG_DIR,
) -> None:
    """Train an agent."""
    import time

    start_time = time.time()

    device = torch.device(device)
    logger = TrainingLogger(agent_type.value, env_name, log_dir)

    # Create environment
    env = gym.make(env_name)

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Initialize agent
    agent_class = get_agent_class(agent_type)
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        device=device,
    )

    # Initialize replay buffer
    memory = ReplayBuffer(maxlen=memory_size)

    # Log training configuration
    config = {
        "Environment": env_name,
        "Agent Type": agent_type.value,
        "Episodes": episodes,
        "Learning Rate": learning_rate,
        "Gamma": gamma,
        "Epsilon Start": epsilon_start,
        "Epsilon End": epsilon_end,
        "Epsilon Decay": epsilon_decay,
        "Memory Size": memory_size,
        "Batch Size": batch_size,
        "Device": device,
    }
    logger.log_training_start(config)

    # Training loop
    total_steps = 0
    best_eval_reward = float("-inf")
    episode_rewards = []
    eval_rewards = []
    eval_episodes = []
    best_episode_reward = float("-inf")

    try:
        with logger.create_progress_bar():
            logger.start_training(episodes)

            for episode in range(episodes):
                state, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
                done = False

                while not done:
                    # Calculate epsilon for exploration
                    epsilon = max(
                        epsilon_end,
                        epsilon_start
                        - (epsilon_start - epsilon_end) * total_steps / epsilon_decay,
                    )

                    # Select action
                    action = agent.choose_action(state, epsilon)

                    # Take step in environment
                    next_state, reward, done, truncated, _ = env.step(action)
                    done = done or truncated
                    episode_reward += reward
                    episode_steps += 1
                    total_steps += 1

                    # Store experience in replay buffer
                    memory.append(
                        (
                            state,
                            next_state,
                            int(action),
                            reward * reward_scaling,
                            done,
                        )
                    )

                    # Update state
                    state = next_state

                    # Update best episode reward
                    best_episode_reward = max(best_episode_reward, episode_reward)

                    # Train agent
                    if len(memory) >= min_memory_size:
                        batch = memory.sample(batch_size)
                        loss = agent.train(batch)
                        logger.log_train(
                            episode=episode,
                            total_steps=total_steps,
                            loss=loss,
                            epsilon=epsilon,
                            reward=episode_reward,
                            best_reward=best_episode_reward,
                        )

                        # Update target network
                        if total_steps % target_update_freq == 0:
                            agent.update_target()

                # Track episode rewards
                episode_rewards.append(episode_reward)

                # Evaluate agent
                if episode % eval_freq == 0:
                    eval_reward_list = []
                    for _ in range(n_eval_episodes):
                        state, _ = env.reset()
                        eval_reward = 0
                        done = False
                        while not done:
                            action = agent.choose_action(state, epsilon=0.0)
                            state, reward, done, truncated, _ = env.step(action)
                            eval_reward += reward
                            done = done or truncated
                        eval_reward_list.append(eval_reward)

                    mean_eval_reward = np.mean(eval_reward_list)
                    eval_rewards.append(mean_eval_reward)
                    eval_episodes.append(episode)

                    if mean_eval_reward > best_eval_reward:
                        best_eval_reward = mean_eval_reward
                        torch.save(
                            agent.online_net.state_dict(),
                            logger.log_dir / "best_model.pth",
                        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
    finally:
        # Calculate training time
        training_time = time.time() - start_time

        # Plot final training progress
        try:
            plot_training_rewards(
                episode_rewards=episode_rewards,
                eval_rewards=eval_rewards,
                eval_episodes=eval_episodes,
                best_reward=best_episode_reward,
                save_path=logger.log_dir,
            )
        except Exception as e:
            console.print(
                f"[yellow]Warning: Failed to plot final training rewards: "
                f"{str(e)}[/yellow]"
            )

        # Print training summary
        print_training_summary(
            episode_rewards=episode_rewards,
            eval_rewards=eval_rewards,
            best_episode_reward=best_episode_reward,
            best_eval_reward=best_eval_reward,
            total_steps=total_steps,
            training_time=training_time,
            log_dir=logger.log_dir,
        )

        # Close environment
        env.close()


@app.command()
def main(
    env: str = typer.Option(ENV_NAME, "--env", "-e", help="Gymnasium environment name"),
    agent: AgentType = typer.Option(
        AGENT_TYPE, "--agent", "-a", help="Type of agent to use"
    ),
    episodes: int = typer.Option(
        MAX_EPISODES, "--episodes", "-n", help="Number of episodes to train"
    ),
    learning_rate: float = typer.Option(LEARNING_RATE, "--lr", help="Learning rate"),
    gamma: float = typer.Option(GAMMA, "--gamma", "-g", help="Discount factor"),
    epsilon_start: float = typer.Option(
        EPSILON_START, "--eps-start", help="Starting epsilon for exploration"
    ),
    epsilon_end: float = typer.Option(
        EPSILON_END, "--eps-end", help="Final epsilon for exploration"
    ),
    epsilon_decay: int = typer.Option(
        EPSILON_DECAY, "--eps-decay", help="Number of steps for epsilon decay"
    ),
    memory_size: int = typer.Option(
        MEMORY_SIZE, "--memory-size", help="Size of replay buffer"
    ),
    batch_size: int = typer.Option(
        BATCH_SIZE, "--batch-size", "-b", help="Batch size for training"
    ),
    target_update_freq: int = typer.Option(
        TARGET_UPDATE_FREQ, "--target-update", help="Frequency of target network update"
    ),
    eval_freq: int = typer.Option(
        EVAL_FREQ, "--eval-freq", help="Frequency of evaluation"
    ),
    n_eval_episodes: int = typer.Option(
        N_EVAL_EPISODES, "--eval-episodes", help="Number of episodes for evaluation"
    ),
    reward_scaling: float = typer.Option(
        REWARD_SCALING, "--reward-scale", help="Scaling factor for rewards"
    ),
    min_memory_size: int = typer.Option(
        MIN_MEMORY_SIZE, "--min-memory", help="Minimum size of memory before training"
    ),
    device: str = typer.Option(
        DEVICE, "--device", "-d", help="Device to use for training"
    ),
    log_dir: str = typer.Option(
        LOG_DIR, "--log-dir", "-l", help="Directory for saving logs"
    ),
) -> None:
    """Train a Distributional RL agent on a Gymnasium environment."""
    train(
        env_name=env,
        agent_type=agent,
        episodes=episodes,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_size=memory_size,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        reward_scaling=reward_scaling,
        min_memory_size=min_memory_size,
        device=device,
        log_dir=log_dir,
    )


if __name__ == "__main__":
    app()
