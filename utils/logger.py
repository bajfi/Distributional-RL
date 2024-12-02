"""Logger implementation for Distributional RL agents."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from torch_drl.configs.default import LOG_DIR

# Set up rich console and plotting style
console = Console()
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 8)


class TrainingLogger:
    """Logger for training progress and visualization."""

    def __init__(
        self,
        agent_type: str,
        env_name: str,
        log_dir: Optional[str] = LOG_DIR,
    ):
        """Initialize logger.

        Args:
            agent_type (str): Type of agent being trained
            env_name (str): Name of the environment
            log_dir (str, optional): Directory to save logs
        """
        self.agent_type = agent_type
        self.env_name = env_name
        self.log_dir = Path(log_dir) / agent_type if log_dir else None
        self.train_losses = []
        self.episode_rewards = []
        self.eval_rewards = []
        self.episodes = []
        self.total_steps = []
        self.progress = None
        self.task_id = None
        self.current_episode = 0

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def create_progress_bar(self) -> Progress:
        """Create a progress bar for training."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("[bold blue]Reward: {task.fields[reward]:.2f}[/bold blue]"),
            TextColumn("[bold red]ε: {task.fields[epsilon]:.3f}[/bold red]"),
            TextColumn("[bold yellow]Avg100: {task.fields[avg100]:.1f}[/bold yellow]"),
            TextColumn("[bold cyan]Best: {task.fields[best]:.1f}[/bold cyan]"),
            TextColumn("[bold magenta]Loss: {task.fields[loss]:.4f}[/bold magenta]"),
            refresh_per_second=4,
            transient=True,
        )
        return self.progress

    def start_training(self, total_episodes: int) -> None:
        """Start the training progress bar.

        Args:
            total_episodes (int): Total number of episodes for training
        """
        if self.progress is None:
            self.create_progress_bar()

        self.task_id = self.progress.add_task(
            "[cyan]Training...",
            total=total_episodes,
            reward=0.0,
            epsilon=1.0,
            avg100=0.0,
            best=float("-inf"),
            loss=0.0,
            completed=0,
        )

    def log_training_start(self, config: Dict) -> None:
        """Log training configuration.

        Args:
            config (dict): Training configuration
        """
        config_table = Table(title="Training Configuration", box=box.ROUNDED)
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="yellow")

        for param, value in config.items():
            config_table.add_row(param, str(value))

        console.print(config_table)
        console.print("\n[bold green]Starting training...[/bold green]\n")

    def log_train(
        self,
        episode: int,
        total_steps: int,
        loss: float,
        epsilon: float = None,
        reward: float = None,
        best_reward: float = None,
    ) -> None:
        """Log training information.

        Args:
            episode (int): Current episode number
            total_steps (int): Total steps taken
            loss (float): Training loss
            epsilon (float, optional): Current epsilon value
            reward (float, optional): Current episode reward
            best_reward (float, optional): Best reward achieved so far
        """
        self.train_losses.append(loss)
        self.episodes.append(episode)
        self.total_steps.append(total_steps)

        if reward is not None:
            self.episode_rewards.append(reward)

        if self.progress and self.task_id is not None:
            update_fields = {"loss": loss}

            if epsilon is not None:
                update_fields["epsilon"] = epsilon

            if reward is not None:
                update_fields["reward"] = reward
                # Update average of last 100 episodes
                if len(self.episode_rewards) >= 100:
                    avg100 = np.mean(self.episode_rewards[-100:])
                    update_fields["avg100"] = avg100
                elif len(self.episode_rewards) > 0:
                    avg100 = np.mean(self.episode_rewards)
                    update_fields["avg100"] = avg100

            # Update best reward if provided
            if best_reward is not None:
                update_fields["best"] = best_reward

            # Update progress
            self.current_episode = episode
            self.progress.update(self.task_id, completed=episode, **update_fields)

    def log_episode_end(
        self, episode: int, solved: bool = False, interrupted: bool = False
    ) -> None:
        """Log episode end status.

        Args:
            episode (int): Episode number
            solved (bool): Whether environment was solved
            interrupted (bool): Whether training was interrupted
        """
        if solved:
            console.print(
                f"\n[bold green]Environment solved in {episode} episodes![/bold green]"
            )
        elif interrupted:
            console.print("\n[yellow]Training interrupted by user[/yellow]")

    def log_training_end(
        self, avg_reward: float, best_reward: float, epsilon: float
    ) -> None:
        """Log training end statistics.

        Args:
            avg_reward (float): Average reward over last 100 episodes
            best_reward (float): Best reward achieved
            epsilon (float): Final epsilon value
        """
        console.print("\n[bold green]Training completed![/bold green]")
        console.print(f"Final average reward (last 100): {avg_reward:.1f}")
        console.print(f"Best reward: {best_reward:.1f}")
        console.print(f"Final epsilon: {epsilon:.3f}")

    def plot_results(
        self, rewards: List[float], losses: List[float], window_size: int = 10
    ) -> None:
        """Plot training results.

        Args:
            rewards (List[float]): List of episode rewards
            losses (List[float]): List of training losses
            window_size (int): Size of moving average window
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        episodes = np.arange(len(rewards))

        # Calculate moving averages
        smoothed_rewards = np.convolve(
            rewards, np.ones(window_size) / window_size, mode="valid"
        )
        smoothed_losses = np.convolve(
            losses, np.ones(window_size) / window_size, mode="valid"
        )
        valid_episodes = episodes[window_size - 1 :]

        # Plot rewards
        ax1.scatter(
            episodes, rewards, alpha=0.4, c="lightblue", label="Raw Rewards", s=30
        )
        ax1.plot(
            valid_episodes,
            smoothed_rewards,
            "r-",
            linewidth=2,
            label=f"Moving Average (n={window_size})",
        )
        ax1.set_title("Training Rewards", fontsize=14, pad=10)
        ax1.set_xlabel("Episode", fontsize=12)
        ax1.set_ylabel("Total Reward", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        # Add trend line
        z = np.polyfit(episodes, rewards, 1)
        p = np.poly1d(z)
        ax1.plot(episodes, p(episodes), "--", color="green", alpha=0.8, label="Trend")

        # Plot losses
        ax2.scatter(
            episodes, losses, alpha=0.4, c="lightblue", label="Raw Losses", s=30
        )
        ax2.plot(
            valid_episodes,
            smoothed_losses,
            "r-",
            linewidth=2,
            label=f"Moving Average (n={window_size})",
        )
        ax2.set_title("Training Losses", fontsize=14, pad=10)
        ax2.set_xlabel("Episode", fontsize=12)
        ax2.set_ylabel("Loss", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()

        if self.log_dir:
            plt.savefig(
                self.log_dir / "training_results.png", dpi=300, bbox_inches="tight"
            )
        plt.close()

        # Print statistics
        stats_table = Table(title="Training Statistics", box=box.ROUNDED)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")

        stats_table.add_row(
            "Final Average Reward", f"{np.mean(rewards[-window_size:]):.2f}"
        )
        stats_table.add_row("Best Reward", f"{max(rewards):.2f}")
        stats_table.add_row("Final Loss", f"{losses[-1]:.4f}")
        stats_table.add_row("Average Loss", f"{np.mean(losses):.4f}")
        stats_table.add_row("Training Episodes", str(len(rewards)))

        console.print(stats_table)
