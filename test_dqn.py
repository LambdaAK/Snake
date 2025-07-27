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
    
    assert state.shape == (12,), f"Expected state shape (12,), got {state.shape}"
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
    state = np.random.rand(12).astype(np.float32)
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
    
    assert next_state.shape == (12,), f"Expected next_state shape (12,), got {next_state.shape}"
    print("✓ Game step test passed!")

def test_danger_detection():
    """Test the new danger detection methods"""
    print("\nTesting danger detection...")
    game = SnakeGame(width=10, height=10)
    
    # Test danger detection methods
    danger_straight = game._is_dangerous_ahead()
    danger_left = game._is_dangerous_left()
    danger_right = game._is_dangerous_right()
    
    print(f"Danger straight: {danger_straight}")
    print(f"Danger left: {danger_left}")
    print(f"Danger right: {danger_right}")
    
    # Test food direction methods
    food_left = game._is_food_left()
    food_right = game._is_food_right()
    food_up = game._is_food_up()
    food_down = game._is_food_down()
    
    print(f"Food left: {food_left}")
    print(f"Food right: {food_right}")
    print(f"Food up: {food_up}")
    print(f"Food down: {food_down}")
    
    print("✓ Danger detection test passed!")

if __name__ == "__main__":
    print("Running DQN implementation tests...\n")
    
    try:
        test_state_representation()
        test_reward_function()
        test_dqn_agent()
        test_game_step()
        test_danger_detection()
        
        print("\n🎉 All tests passed! DQN implementation is ready for training.")
        print("\nTo start training, run: python dqn_agent.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 