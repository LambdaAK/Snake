#!/usr/bin/env python3

import os
import sys
import random
import termios
import tty
import numpy as np
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any

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

class Direction(Enum):
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

class SnakeGame:
    def __init__(self, width: int = 20, height: int = 15):
        self.width = width
        self.height = height
        self.snake: List[Tuple[int, int]] = [(height // 2, width // 2)]
        self.direction = Direction.RIGHT
        self.food: Optional[Tuple[int, int]] = None
        self.score = 0
        self.game_over = False
        self.steps_since_last_food = 0
        self._spawn_food()
    
    def _spawn_food(self):
        while True:
            food_pos = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if food_pos not in self.snake:
                self.food = food_pos
                break
    
    def reset(self):
        """Reset the game for a new episode"""
        self.snake = [(self.height // 2, self.width // 2)]
        self.direction = Direction.RIGHT
        self.score = 0
        self.game_over = False
        self.steps_since_last_food = 0
        self._spawn_food()
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Get the current state representation for DQN"""
        head = self.snake[0]
        food = self.food
        
        # Calculate distance to food
        distance_to_food = abs(head[0] - food[0]) + abs(head[1] - food[1])
        
        # Calculate unit vector toward food
        dx = food[1] - head[1]
        dy = food[0] - head[0]
        distance = max(1, distance_to_food)  # Avoid division by zero
        unit_vector_dx = dx / distance
        unit_vector_dy = dy / distance
        
        # Normalized snake length
        normalized_snake_length = len(self.snake) / 10.0
        
        # Direction one-hot encoding
        direction_up = 1.0 if self.direction == Direction.UP else 0.0
        direction_down = 1.0 if self.direction == Direction.DOWN else 0.0
        direction_left = 1.0 if self.direction == Direction.LEFT else 0.0
        direction_right = 1.0 if self.direction == Direction.RIGHT else 0.0
        
        # Normalized head coordinates
        normalized_head_x = head[1] / self.width
        normalized_head_y = head[0] / self.height
        
        # Danger detection
        danger_up = self._is_dangerous((head[0] - 1, head[1]))
        danger_down = self._is_dangerous((head[0] + 1, head[1]))
        danger_left = self._is_dangerous((head[0], head[1] - 1))
        danger_right = self._is_dangerous((head[0], head[1] + 1))
        
        state = np.array([
            distance_to_food,
            unit_vector_dx,
            unit_vector_dy,
            normalized_snake_length,
            direction_up,
            direction_down,
            direction_left,
            direction_right,
            normalized_head_x,
            normalized_head_y,
            danger_up,
            danger_down,
            danger_left,
            danger_right
        ], dtype=np.float32)
        
        return state
    
    def _is_dangerous(self, pos: Tuple[int, int]) -> float:
        """Check if a position is dangerous (wall or body)"""
        row, col = pos
        # Check wall collision
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return 1.0
        # Check body collision
        if pos in self.snake:
            return 1.0
        return 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take a step in the game given an action"""
        # Store old state for reward calculation
        old_distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        food_eaten = False
        
        # Convert action to direction
        if action == 0 and self.direction != Direction.DOWN:  # UP
            self.direction = Direction.UP
        elif action == 1 and self.direction != Direction.UP:  # DOWN
            self.direction = Direction.DOWN
        elif action == 2 and self.direction != Direction.RIGHT:  # LEFT
            self.direction = Direction.LEFT
        elif action == 3 and self.direction != Direction.LEFT:  # RIGHT
            self.direction = Direction.RIGHT
        
        # Move snake
        head_row, head_col = self.snake[0]
        dir_row, dir_col = self.direction.value
        new_head = (head_row + dir_row, head_col + dir_col)
        
        # Check collisions
        if (new_head[0] < 0 or new_head[0] >= self.height or 
            new_head[1] < 0 or new_head[1] >= self.width):
            self.game_over = True
        elif new_head in self.snake:
            self.game_over = True
        else:
            self.snake.insert(0, new_head)
            
            # Check food collision
            if new_head == self.food:
                self.score += 1
                self.steps_since_last_food = 0
                food_eaten = True
                self._spawn_food()
            else:
                self.snake.pop()
                self.steps_since_last_food += 1
        
        # Calculate reward
        reward = self._calculate_reward(old_distance, food_eaten)
        
        # Get new state
        new_state = self.get_state()
        
        # Check if episode should end (max steps or game over)
        done = self.game_over or self.steps_since_last_food >= 500
        
        info = {
            'score': self.score,
            'snake_length': len(self.snake),
            'steps_since_food': self.steps_since_last_food
        }
        
        return new_state, reward, done, info
    
    def _calculate_reward(self, old_distance: int, food_eaten: bool) -> float:
        """Calculate reward based on the reward function specification"""
        reward = 0.0
        
        # Food proximity reward
        new_distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        if new_distance < old_distance:
            reward += 1.0
        
        # Food consumption reward
        if food_eaten:
            reward += 10.0
        
        # Death penalty
        if self.game_over:
            reward -= 10.0
        
        # Survival bonus
        reward += 0.1
        
        return reward
    
    def _clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def _render(self):
        self._clear_screen()
        
        print("=" * (self.width + 2))
        for row in range(self.height):
            print("|", end="")
            for col in range(self.width):
                pos = (row, col)
                if pos == self.snake[0]:
                    print("O", end="")  # Snake head
                elif pos in self.snake[1:]:
                    print("o", end="")  # Snake body
                elif pos == self.food:
                    print("*", end="")  # Food
                else:
                    print(" ", end="")
            print("|")
        print("=" * (self.width + 2))
        print(f"Score: {self.score}")
        print("Controls: W=Up, A=Left, S=Down, D=Right, Q=Quit")
        
        if self.game_over:
            print("\nGAME OVER! Press any key to exit...")
    
    def _get_key(self) -> str:
        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1).lower()
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return key
        except (termios.error, OSError):
            # Fallback for non-interactive terminals
            return input("Enter move (w/a/s/d/q): ").lower().strip()[:1] or 'q'
    
    def _move_snake(self):
        if self.game_over:
            return
        
        head_row, head_col = self.snake[0]
        dir_row, dir_col = self.direction.value
        new_head = (head_row + dir_row, head_col + dir_col)
        
        # Check wall collision
        if (new_head[0] < 0 or new_head[0] >= self.height or 
            new_head[1] < 0 or new_head[1] >= self.width):
            self.game_over = True
            return
        
        # Check self collision
        if new_head in self.snake:
            self.game_over = True
            return
        
        self.snake.insert(0, new_head)
        
        # Check food collision
        if new_head == self.food:
            self.score += 1
            self._spawn_food()
        else:
            self.snake.pop()
    
    def _handle_input(self, key: str) -> bool:
        if key == 'q':
            return False
        elif key == 'w' and self.direction != Direction.DOWN:
            self.direction = Direction.UP
        elif key == 'a' and self.direction != Direction.RIGHT:
            self.direction = Direction.LEFT
        elif key == 's' and self.direction != Direction.UP:
            self.direction = Direction.DOWN
        elif key == 'd' and self.direction != Direction.LEFT:
            self.direction = Direction.RIGHT
        
        return True
    
    def play(self):
        print("Welcome to Snake Game!")
        print("Use WASD to control the snake, Q to quit")
        print("Press any key to start...")
        self._get_key()
        
        while True:
            self._render()
            
            if self.game_over:
                self._get_key()
                break
            
            key = self._get_key()
            if not self._handle_input(key):
                break
            
            self._move_snake()

def main():
    game = SnakeGame()
    try:
        game.play()
    except KeyboardInterrupt:
        print("\nGame interrupted. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()