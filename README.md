# Snake Game DQN Agent

A Deep Q-Network (DQN) implementation to train an AI agent to play the classic Snake game.

## Features

- **DQN Architecture**: 3-layer neural network with ReLU activations and dropout
- **Enhanced State Representation**: 30-dimensional vector with comprehensive game state information
- **Reward Function**: Balanced rewards for food proximity (+1), food consumption (+10), death penalty (-10), and survival bonus (+0.1)
- **Experience Replay**: 10,000 experience buffer for stable training
- **Target Network**: Updated every 100 episodes for training stability
- **Automatic Saving**: Models saved every 1000 episodes
- **Extended Episodes**: Up to 10,000 moves per episode for better learning

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

### Run Trained Model
```bash
python run_trained_model.py
```

### Play the Original Game
```bash
python snake_game.py
```

## Architecture

### Enhanced State Representation (30 entries)
1. **Basic Information (10 entries)**
   - Distance to food
   - Unit vector toward food (dx, dy)
   - Normalized snake length
   - Snake direction (one-hot encoded, 4 entries)
   - Normalized head coordinates (x, y)

2. **Enhanced Danger Detection (8 entries)**
   - Immediate danger (4 directions)
   - 2-step ahead danger (4 directions)

3. **Body Awareness (4 entries)**
   - Body proximity in each direction (how close nearest body segment is)

4. **Food Direction (4 entries)**
   - Food direction relative to current heading (one-hot encoded)

5. **Space Analysis (4 entries)**
   - Available free space in each direction

### DQN Network
- Input: 30 neurons
- Hidden 1: 64 neurons + ReLU + Dropout(0.2)
- Hidden 2: 32 neurons + ReLU + Dropout(0.2)
- Output: 4 neurons (one for each action)

### Hyperparameters
- Learning rate: 0.001
- Discount factor: 0.99
- Epsilon: 1.0 → 0.01 (decay: 0.9995)
- Batch size: 32
- Experience replay: 10,000
- Max steps per episode: 10,000

## Key Improvements

### Enhanced State Representation
- **2-step danger detection**: Agent can see dangers 2 moves ahead
- **Body proximity awareness**: Knows how close its body segments are in each direction
- **Space analysis**: Understands available free space in each direction
- **Food direction encoding**: Clear indication of where food is relative to current heading

### Extended Training
- **10,000 move episodes**: Much longer episodes for better learning
- **Comprehensive state information**: Agent has much better awareness of its environment

## Files

- `snake_game.py`: Enhanced Snake game with comprehensive state representation
- `dqn_agent.py`: DQN implementation and training loop
- `run_trained_model.py`: Interactive model runner with visualization
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

## Expected Performance

With the enhanced state representation, the agent should be able to:
- Achieve much higher scores (potentially 50+ food items)
- Better navigate complex scenarios with long snakes
- Make more informed decisions about movement
- Avoid getting trapped by its own body
