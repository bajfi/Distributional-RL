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
git https://github.com/bajfi/Distributional-RL.git
cd torch_drl

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Install library
pip install -e .
```

## Usage

To train an agent:

```bash
python main -e CartPole-v1 -a dqn -n 200
```

For more options, see `python -m torch_drl.main --help`.

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The implementations are based on the following papers:
  - [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
  - [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)
  - [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923)
