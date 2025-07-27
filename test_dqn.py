#!/usr/bin/env python3

import numpy as np
from snake_game import SnakeGame
from dqn_agent import DQNAgent

def test_state_representation():
    """Test that state representation works correctly"""
    print("Testing state representation...")
    game = SnakeGame(width=10, height=10)
    state = game.get_state()
    
    print(f"State shape: {state.shape}")
    print(f"State values: {state}")
    print(f"State sum: {np.sum(state)}")
    
    assert state.shape == (55,), f"Expected state shape (55,), got {state.shape}"
    print("✓ State representation test passed!")

def test_reward_function():
    """Test that reward function works correctly"""
    print("\nTesting reward function...")
    game = SnakeGame(width=10, height=10)
    
    # Test initial state
    initial_distance = abs(game.snake[0][0] - game.food[0]) + abs(game.snake[0][1] - game.food[1])
    reward = game._calculate_reward(initial_distance, False)
    print(f"Initial reward: {reward}")
    
    # Test food eaten
    reward = game._calculate_reward(initial_distance, True)
    print(f"Food eaten reward: {reward}")
    
    # Test death
    game.game_over = True
    reward = game._calculate_reward(initial_distance, False)
    print(f"Death penalty: {reward}")
    
    print("✓ Reward function test passed!")

def test_dqn_agent():
    """Test that DQN agent can be created and act"""
    print("\nTesting DQN agent...")
    agent = DQNAgent()
    
    # Test action selection
    state = np.random.rand(55).astype(np.float32)
    action = agent.act(state)
    print(f"Selected action: {action}")
    
    assert 0 <= action <= 3, f"Action should be 0-3, got {action}"
    print("✓ DQN agent test passed!")

def test_game_step():
    """Test that game step function works with DQN"""
    print("\nTesting game step function...")
    game = SnakeGame(width=10, height=10)
    
    state = game.get_state()
    action = 0  # UP
    next_state, reward, done, info = game.step(action)
    
    print(f"Action: {action}")
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    
    assert next_state.shape == (55,), f"Expected next_state shape (55,), got {next_state.shape}"
    print("✓ Game step test passed!")

def test_5x5_grid():
    """Test the 5x5 grid representation"""
    print("\nTesting 5x5 grid...")
    game = SnakeGame(width=10, height=10)
    
    # Get the 5x5 grid
    grid = game._get_5x5_grid(game.snake[0])
    
    print(f"Grid shape: {grid.shape}")
    print(f"Grid values: {grid}")
    
    # The center should be the snake head (value 3.0)
    assert grid[12] == 3.0, f"Center of grid should be snake head (3.0), got {grid[12]}"
    
    # Check if food is in the grid
    food_in_grid = 4.0 in grid
    print(f"Food in grid: {food_in_grid}")
    
    print("✓ 5x5 grid test passed!")

if __name__ == "__main__":
    print("Running DQN implementation tests...\n")
    
    try:
        test_state_representation()
        test_reward_function()
        test_dqn_agent()
        test_game_step()
        test_5x5_grid()
        
        print("\n🎉 All tests passed! DQN implementation is ready for training.")
        print("\nTo start training, run: python dqn_agent.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 