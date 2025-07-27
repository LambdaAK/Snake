# Snake DQN

A Deep Q-Network (DQN) AI for the classic Snake game using PyTorch.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Play manually:
```bash
python snake_game.py
```

Train the AI:
```bash
python dqn_agent.py
```

Test the AI:
```bash
python run_trained_model.py
```

## Implementation

This project implements a Deep Q-Network (DQN) agent to play the classic Snake game, with several enhancements for improved learning and performance:

- **Deep Q-Network Architecture:**  
  The agent uses a neural network with four fully connected layers. The input layer receives a 29-dimensional state vector, followed by two hidden layers (each with 128 neurons and ReLU activations), a third hidden layer with 64 neurons, and an output layer with 4 neurons (corresponding to the four possible movement directions: up, down, left, right). This architecture allows the agent to learn complex strategies and spatial patterns in the game.

- **State Representation (29 Dimensions):**  
  The state vector encodes comprehensive information about the game environment, including:
  - The snake's current direction (one-hot encoded)
  - Relative position of the food to the snake's head
  - Danger indicators for all four directions (e.g., if moving in a direction would result in collision with the wall or the snake's own body)
  - Distances to the walls and to the snake's own body in each direction
  - Information about available space and potential dead ends
  This rich state representation enables the agent to make informed decisions and anticipate future risks.

- **Reward Function with Spatial Awareness:**  
  The reward function is carefully designed to encourage the agent to survive longer, seek food efficiently, and avoid self-trapping. It includes:
  - Positive rewards for moving closer to the food and for eating food
  - Small positive reward for each step survived (to encourage longer games)
  - Penalties for moving away from the food, colliding with walls or the snake's own body, and entering dangerous or cramped spaces
  - Additional spatial awareness: the agent is penalized for reducing its available escape routes or getting too close to its own body, and is rewarded for maintaining open space around the head
  This balanced reward structure helps the agent learn not only to chase food but also to avoid common pitfalls like self-collision and dead ends.

- **Experience Replay and Target Network:**  
  The agent uses an experience replay buffer to store past experiences and samples random batches for training, which stabilizes learning. A separate target network is updated periodically to further improve training stability.

- **Training and Testing Scripts:**  
  The repository includes scripts for training the agent (`dqn_agent.py`), running a trained model (`run_trained_model.py`), and testing core components (`test_dqn.py`).

For more details on the architecture and algorithms, see the code and comments in the respective Python files.
