#!/usr/bin/env python3

import numpy as np
from snake_game import SnakeGame
from dqn_agent import DQNAgent

# Terminal colors for pretty output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    UNDERLINE = '\033[4m'

def print_test_header():
    """Print a pretty header for the test session"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("=" * 50)
    print("🧪 DQN IMPLEMENTATION TESTS 🧪")
    print("=" * 50)
    print(f"{Colors.RESET}")

def print_success(message: str):
    """Print a success message in green"""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")

def print_info(message: str):
    """Print an info message in blue"""
    print(f"{Colors.BLUE}ℹ {message}{Colors.RESET}")

def test_state_representation():
    """Test that state representation works correctly"""
    print(f"{Colors.YELLOW}Testing state representation...{Colors.RESET}")
    game = SnakeGame(width=10, height=10)
    state = game.get_state()
    
    print(f"State shape: {Colors.CYAN}{state.shape}{Colors.RESET}")
    print(f"State values: {Colors.MAGENTA}{state}{Colors.RESET}")
    print(f"State sum: {Colors.CYAN}{np.sum(state)}{Colors.RESET}")
    
    assert state.shape == (29,), f"Expected state shape (29,), got {state.shape}"
    print_success("State representation test passed!")

def test_reward_function():
    """Test that reward function works correctly"""
    print(f"\n{Colors.YELLOW}Testing reward function...{Colors.RESET}")
    game = SnakeGame(width=10, height=10)
    
    # Test initial state
    initial_distance = abs(game.snake[0][0] - game.food[0]) + abs(game.snake[0][1] - game.food[1])
    reward = game._calculate_reward(initial_distance, False)
    print(f"Initial reward: {Colors.CYAN}{reward}{Colors.RESET}")
    
    # Test food eaten
    reward = game._calculate_reward(initial_distance, True)
    print(f"Food eaten reward: {Colors.GREEN}{reward}{Colors.RESET}")
    
    # Test death
    game.game_over = True
    reward = game._calculate_reward(initial_distance, False)
    print(f"Death penalty: {Colors.RED}{reward}{Colors.RESET}")
    
    print_success("Reward function test passed!")

def test_dqn_agent():
    """Test that DQN agent can be created and act"""
    print(f"\n{Colors.YELLOW}Testing DQN agent...{Colors.RESET}")
    agent = DQNAgent()
    
    # Test action selection
    state = np.random.rand(29).astype(np.float32)
    action = agent.act(state)
    print(f"Selected action: {Colors.CYAN}{action}{Colors.RESET}")
    
    assert 0 <= action <= 3, f"Action should be 0-3, got {action}"
    print_success("DQN agent test passed!")

def test_game_step():
    """Test that game step function works with DQN"""
    print(f"\n{Colors.YELLOW}Testing game step function...{Colors.RESET}")
    game = SnakeGame(width=10, height=10)
    
    state = game.get_state()
    action = 0  # UP
    next_state, reward, done, info = game.step(action)
    
    print(f"Action: {Colors.CYAN}{action}{Colors.RESET}")
    print(f"Reward: {Colors.MAGENTA}{reward}{Colors.RESET}")
    print(f"Done: {Colors.CYAN}{done}{Colors.RESET}")
    print(f"Info: {Colors.BLUE}{info}{Colors.RESET}")
    
    assert next_state.shape == (29,), f"Expected next_state shape (29,), got {next_state.shape}"
    print_success("Game step test passed!")

def test_spatial_awareness():
    """Test the new spatial awareness features"""
    print(f"\n{Colors.YELLOW}Testing spatial awareness features...{Colors.RESET}")
    game = SnakeGame(width=10, height=10)
    
    # Test wall distances
    wall_dist_up = game._get_wall_distance('up')
    wall_dist_down = game._get_wall_distance('down')
    wall_dist_left = game._get_wall_distance('left')
    wall_dist_right = game._get_wall_distance('right')
    
    print(f"Wall distances - up: {Colors.CYAN}{wall_dist_up:.2f}{Colors.RESET}, "
          f"down: {Colors.CYAN}{wall_dist_down:.2f}{Colors.RESET}, "
          f"left: {Colors.CYAN}{wall_dist_left:.2f}{Colors.RESET}, "
          f"right: {Colors.CYAN}{wall_dist_right:.2f}{Colors.RESET}")
    
    # Test body distances
    body_dist_up = game._get_body_distance('up')
    body_dist_down = game._get_body_distance('down')
    body_dist_left = game._get_body_distance('left')
    body_dist_right = game._get_body_distance('right')
    
    print(f"Body distances - up: {Colors.MAGENTA}{body_dist_up:.2f}{Colors.RESET}, "
          f"down: {Colors.MAGENTA}{body_dist_down:.2f}{Colors.RESET}, "
          f"left: {Colors.MAGENTA}{body_dist_left:.2f}{Colors.RESET}, "
          f"right: {Colors.MAGENTA}{body_dist_right:.2f}{Colors.RESET}")
    
    # Test safe moves count
    safe_moves = game._count_safe_moves()
    print(f"Safe moves count: {Colors.GREEN}{safe_moves:.2f}{Colors.RESET}")
    
    # Test dead end detection
    dead_end_up = game._is_dead_end_in_direction('up')
    dead_end_down = game._is_dead_end_in_direction('down')
    dead_end_left = game._is_dead_end_in_direction('left')
    dead_end_right = game._is_dead_end_in_direction('right')
    
    print(f"Dead ends - up: {Colors.RED if dead_end_up else Colors.GREEN}{dead_end_up}{Colors.RESET}, "
          f"down: {Colors.RED if dead_end_down else Colors.GREEN}{dead_end_down}{Colors.RESET}, "
          f"left: {Colors.RED if dead_end_left else Colors.GREEN}{dead_end_left}{Colors.RESET}, "
          f"right: {Colors.RED if dead_end_right else Colors.GREEN}{dead_end_right}{Colors.RESET}")
    
    print_success("Spatial awareness test passed!")

if __name__ == "__main__":
    print_test_header()
    
    try:
        test_state_representation()
        test_reward_function()
        test_dqn_agent()
        test_game_step()
        test_spatial_awareness()
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed! DQN implementation is ready for training.{Colors.RESET}")
        print(f"\n{Colors.CYAN}To start training, run: {Colors.BOLD}python dqn_agent.py{Colors.RESET}")
        
    except Exception as e:
        print(f"\n{Colors.RED}❌ Test failed: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc() 