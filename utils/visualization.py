"""Visualization utilities for training progress."""

import warnings
from pathlib import Path
from typing import List

# Set matplotlib backend to Agg (non-interactive) before importing pyplot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


def plot_training_rewards(
    episode_rewards: List[float],
    eval_rewards: List[float],
    eval_episodes: List[int],
    best_reward: float,
    save_path: Path,
    window_size: int = 10,
) -> None:
    """Plot training and evaluation rewards.

    Args:
        episode_rewards (List[float]): List of episode rewards
        eval_rewards (List[float]): List of evaluation rewards
        eval_episodes (List[int]): List of evaluation episode numbers
        best_reward (float): Best reward achieved
        save_path (Path): Path to save the plot
        window_size (int): Window size for moving average
    """
    try:
        # Create figure outside of style context
        fig, ax = plt.subplots(figsize=(12, 6))

        # Set style after creating figure
        plt.style.use("seaborn-v0_8-paper")

        # Plot training rewards
        episodes = range(1, len(episode_rewards) + 1)
        ax.plot(
            episodes,
            episode_rewards,
            alpha=0.6,
            color="#1f77b4",
            label="Training Rewards",
        )

        # Plot moving average
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(
                episode_rewards, np.ones(window_size) / window_size, mode="valid"
            )
            ax.plot(
                np.arange(window_size, len(episode_rewards) + 1),
                moving_avg,
                color="#2c3e50",
                linewidth=2,
                label=f"Training Moving Average (n={window_size})",
            )

        # Plot evaluation rewards
        if eval_rewards and eval_episodes:
            ax.scatter(
                eval_episodes,
                eval_rewards,
                color="#e74c3c",
                marker="o",
                s=50,
                label="Evaluation Rewards",
            )

        # Plot best reward line
        ax.axhline(
            y=best_reward,
            color="#27ae60",
            linestyle="--",
            label=f"Best Reward: {best_reward:.2f}",
        )

        # Customize plot
        ax.set_xlabel("Episode", fontsize=12, fontweight="bold")
        ax.set_ylabel("Reward", fontsize=12, fontweight="bold")
        ax.set_title("Training Progress", fontsize=14, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Set background color
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#ffffff")

        # Add some padding to the layout
        plt.tight_layout()

        # Ensure the directory exists
        save_path.mkdir(parents=True, exist_ok=True)

        # Save with high quality
        fig.savefig(
            save_path / "training_rewards.png",
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
            edgecolor="none",
        )
    except Exception as e:
        warnings.warn(f"Error plotting training rewards: {str(e)}")
    finally:
        # Clean up
        plt.close("all")
