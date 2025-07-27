# Snake Game DQN Agent

A Deep Q-Network (DQN) implementation to train an AI agent to play the classic Snake game.

## Features

- **DQN Architecture**: 3-layer neural network with ReLU activations and dropout
- **State Representation**: 14-dimensional vector including food distance, direction, snake state, and danger detection
- **Reward Function**: Balanced rewards for food proximity (+1), food consumption (+10), death penalty (-10), and survival bonus (+0.1)
- **Experience Replay**: 10,000 experience buffer for stable training
- **Target Network**: Updated every 100 episodes for training stability
- **Automatic Saving**: Models saved every 1000 episodes

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Test the Implementation
```bash
python test_dqn.py
```

### Start Training
```bash
python dqn_agent.py
```

The training will run indefinitely and:
- Print progress every 100 episodes
- Save models every 1000 episodes in the `agents/` folder
- Can be stopped with Ctrl+C

### Play the Original Game
```bash
python snake_game.py
```

## Architecture

### State Representation (14 entries)
1. Distance to food
2. Unit vector toward food (dx, dy)
3. Normalized snake length
4. Snake direction (one-hot encoded, 4 entries)
5. Normalized head coordinates (x, y)
6. Danger detection (4 directions)

### DQN Network
- Input: 14 neurons
- Hidden 1: 64 neurons + ReLU + Dropout(0.2)
- Hidden 2: 32 neurons + ReLU + Dropout(0.2)
- Output: 4 neurons (one for each action)

### Hyperparameters
- Learning rate: 0.001
- Discount factor: 0.99
- Epsilon: 1.0 → 0.01 (decay: 0.9995)
- Batch size: 32
- Experience replay: 10,000
- Max steps per episode: 500

## Files

- `snake_game.py`: Modified Snake game with DQN support
- `dqn_agent.py`: DQN implementation and training loop
- `test_dqn.py`: Test script to verify implementation
- `requirements.txt`: Python dependencies
- `agents/`: Directory where trained models are saved

## Training Progress

The training will display:
- Episode number
- Average reward (last 100 episodes)
- Average score (last 100 episodes)
- Average episode length (last 100 episodes)
- Current epsilon value

Models are automatically saved as `snake_dqn_episode_X.pth` where X is the episode number.
