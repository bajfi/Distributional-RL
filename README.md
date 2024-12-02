# Distributional Reinforcement Learning Implementation

This repository contains implementations of various distributional reinforcement learning algorithms in PyTorch, including DQN, C51, QR-DQN, and IQN.

## Features

- Clean, modular implementation of distributional RL algorithms
- Support for multiple environments through Gymnasium
- Visualization tools for training progress
- Configurable hyperparameters
- Efficient replay buffer implementation

## Implemented Algorithms

- DQN (Deep Q-Network)
- C51 (Categorical DQN)
- QR-DQN (Quantile Regression DQN)
- IQN (Implicit Quantile Network)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/distributional-rl.git
cd distributional-rl

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage

To train an agent:

```bash
python -m torch_drl.main --env-name CartPole-v1 --agent-type qrdqn
```

Available agent types:

- dqn
- c51
- qrdqn
- iqn

## Project Structure

```
.
├── torch_drl/
│   ├── agents/
│   │   ├── base.py
│   │   ├── dqn.py
│   │   ├── c51.py
│   │   ├── qrdqn.py
│   │   └── iqn.py
│   ├── models/
│   │   ├── base.py
│   │   ├── dqn.py
│   │   ├── c51.py
│   │   ├── qrdqn.py
│   │   └── iqn.py
│   ├── utils/
│   │   ├── logger.py
│   │   ├── memory.py
│   │   └── visualization.py
│   ├── configs/
│   │   └── default.py
│   └── main.py
├── tests/
├── requirements.txt
├── setup.py
└── README.md
```

## Configuration

You can modify the hyperparameters in `torch_drl/configs/default.py`. Key configurations include:

- Learning rate
- Discount factor (gamma)
- Epsilon parameters for exploration
- Network architecture
- Training parameters

## Results

The implemented algorithms have been tested on various Gymnasium environments. Here are some example results:

- CartPole-v1: All algorithms achieve optimal performance
- Acrobot-v1: Distributional methods show improved sample efficiency
- LunarLander-v2: QR-DQN and IQN demonstrate better performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The implementations are based on the following papers:
  - [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
  - [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
  - [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)
