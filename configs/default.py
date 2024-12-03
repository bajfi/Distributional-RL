"""Default configuration for Distributional RL agents."""

import torch

# Training hyperparameters
BATCH_SIZE = 128  # Batch size for training
LEARNING_RATE = 3e-4  # Learning rate
MEMORY_SIZE = 100_000  # Size of replay buffer
MIN_MEMORY_SIZE = 1000  # Minimum size of memory before training
GAMMA = 0.99  # Discount factor
TARGET_UPDATE_FREQ = 100  # Frequency of target network update

# Exploration parameters
EPSILON_START = 1.0  # Starting epsilon for exploration
EPSILON_END = 0.01  # Final epsilon for exploration
EPSILON_DECAY = 5000  # Number of steps for epsilon decay

# Model parameters
REWARD_SCALING = 1.0  # Scaling factor for rewards
GRADIENT_CLIP = 1.0  # Gradient clipping
HIDDEN_SIZE = 128  # Hidden size of the network

# Agent-specific parameters
DQN_CONFIG = {
    "hidden_dim": HIDDEN_SIZE,
}

C51_CONFIG = {
    "hidden_dim": HIDDEN_SIZE,
    "n_atoms": 51,  # Number of atoms
    "v_min": -10.0,  # Minimum value of the support
    "v_max": 10.0,  # Maximum value of the support
}

QRDQN_CONFIG = {
    "hidden_dim": HIDDEN_SIZE,
    "n_quantiles": 200,  # Number of quantiles
}

IQN_CONFIG = {
    "hidden_dim": HIDDEN_SIZE,
    "n_quantiles": 64,  # Number of quantiles
    "n_cos_embeddings": 128,  # Number of cosine embeddings
}

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment parameters
ENV_NAME = "CartPole-v1"  # Name of the environment
MAX_EPISODES = 500  # Maximum number of episodes
EVAL_FREQ = 10  # Frequency of evaluation
N_EVAL_EPISODES = 2  # Number of episodes for evaluation

# Logging parameters
LOG_DIR = "runs"  # Directory for saving logs

# Agent type
AGENT_TYPE = "dqn"
