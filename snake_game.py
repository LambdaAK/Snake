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
        
        # Enhanced danger detection - check multiple steps ahead
        danger_up = self._is_dangerous((head[0] - 1, head[1]))
        danger_down = self._is_dangerous((head[0] + 1, head[1]))
        danger_left = self._is_dangerous((head[0], head[1] - 1))
        danger_right = self._is_dangerous((head[0], head[1] + 1))
        
        # Check 2 steps ahead for each direction
        danger_up_2 = self._is_dangerous_2_steps((head[0] - 1, head[1]), (head[0] - 2, head[1]))
        danger_down_2 = self._is_dangerous_2_steps((head[0] + 1, head[1]), (head[0] + 2, head[1]))
        danger_left_2 = self._is_dangerous_2_steps((head[0], head[1] - 1), (head[0], head[1] - 2))
        danger_right_2 = self._is_dangerous_2_steps((head[0], head[1] + 1), (head[0], head[1] + 2))
        
        # Body proximity - how close is the nearest body segment in each direction
        body_proximity_up = self._get_body_proximity(head, 'up')
        body_proximity_down = self._get_body_proximity(head, 'down')
        body_proximity_left = self._get_body_proximity(head, 'left')
        body_proximity_right = self._get_body_proximity(head, 'right')
        
        # Food direction relative to current heading
        food_direction_up = 1.0 if food[0] < head[0] else 0.0
        food_direction_down = 1.0 if food[0] > head[0] else 0.0
        food_direction_left = 1.0 if food[1] < head[1] else 0.0
        food_direction_right = 1.0 if food[1] > head[1] else 0.0
        
        # Available space in each direction (how many free cells)
        space_up = self._get_available_space(head, 'up')
        space_down = self._get_available_space(head, 'down')
        space_left = self._get_available_space(head, 'left')
        space_right = self._get_available_space(head, 'right')
        
        # 5x5 grid centered on snake head
        grid_5x5 = self._get_5x5_grid(head)
        
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
            danger_right,
            danger_up_2,
            danger_down_2,
            danger_left_2,
            danger_right_2,
            body_proximity_up,
            body_proximity_down,
            body_proximity_left,
            body_proximity_right,
            food_direction_up,
            food_direction_down,
            food_direction_left,
            food_direction_right,
            space_up,
            space_down,
            space_left,
            space_right
        ], dtype=np.float32)
        
        # Append the 5x5 grid to the state
        state = np.concatenate([state, grid_5x5])
        
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
    
    def _is_dangerous_2_steps(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Check if moving 2 steps in a direction is dangerous"""
        # Check if first step is dangerous
        if self._is_dangerous(pos1) > 0:
            return 1.0
        
        # Check if second step is dangerous
        if self._is_dangerous(pos2) > 0:
            return 1.0
        
        return 0.0
    
    def _get_body_proximity(self, head: Tuple[int, int], direction: str) -> float:
        """Get how close the nearest body segment is in a given direction"""
        head_row, head_col = head
        
        if direction == 'up':
            for row in range(head_row - 1, -1, -1):
                if (row, head_col) in self.snake[1:]:  # Exclude head
                    return (head_row - row) / 10.0  # Normalize
        elif direction == 'down':
            for row in range(head_row + 1, self.height):
                if (row, head_col) in self.snake[1:]:
                    return (row - head_row) / 10.0
        elif direction == 'left':
            for col in range(head_col - 1, -1, -1):
                if (head_row, col) in self.snake[1:]:
                    return (head_col - col) / 10.0
        elif direction == 'right':
            for col in range(head_col + 1, self.width):
                if (head_row, col) in self.snake[1:]:
                    return (col - head_col) / 10.0
        
        return 0.0  # No body segment found in this direction
    
    def _get_available_space(self, head: Tuple[int, int], direction: str) -> float:
        """Get the number of free cells available in a given direction"""
        head_row, head_col = head
        free_cells = 0
        
        if direction == 'up':
            for row in range(head_row - 1, -1, -1):
                if (row, head_col) not in self.snake and row >= 0:
                    free_cells += 1
                else:
                    break
        elif direction == 'down':
            for row in range(head_row + 1, self.height):
                if (row, head_col) not in self.snake and row < self.height:
                    free_cells += 1
                else:
                    break
        elif direction == 'left':
            for col in range(head_col - 1, -1, -1):
                if (head_row, col) not in self.snake and col >= 0:
                    free_cells += 1
                else:
                    break
        elif direction == 'right':
            for col in range(head_col + 1, self.width):
                if (head_row, col) not in self.snake and col < self.width:
                    free_cells += 1
                else:
                    break
        
        return free_cells / 10.0  # Normalize
    
    def _get_5x5_grid(self, head: Tuple[int, int]) -> np.ndarray:
        """Get a 5x5 grid centered on the snake head"""
        head_row, head_col = head
        grid = np.zeros(25, dtype=np.float32)  # 5x5 = 25 elements
        
        # Define what each value represents:
        # 0.0 = empty space
        # 1.0 = wall
        # 2.0 = snake body
        # 3.0 = snake head
        # 4.0 = food
        
        idx = 0
        for row_offset in range(-2, 3):  # -2, -1, 0, 1, 2
            for col_offset in range(-2, 3):  # -2, -1, 0, 1, 2
                grid_row = head_row + row_offset
                grid_col = head_col + col_offset
                
                # Check if position is within board bounds
                if (grid_row < 0 or grid_row >= self.height or 
                    grid_col < 0 or grid_col >= self.width):
                    grid[idx] = 1.0  # Wall
                else:
                    pos = (grid_row, grid_col)
                    if pos == head:
                        grid[idx] = 3.0  # Snake head
                    elif pos == self.food:
                        grid[idx] = 4.0  # Food
                    elif pos in self.snake[1:]:  # Exclude head
                        grid[idx] = 2.0  # Snake body
                    else:
                        grid[idx] = 0.0  # Empty space
                
                idx += 1
        
        return grid
    
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
        """Calculate reward based on the enhanced reward function"""
        reward = 0.0
        
        # Food proximity reward (more nuanced)
        new_distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        distance_change = old_distance - new_distance
        
        if distance_change > 0:
            reward += 2.0  # Moving closer to food
        elif distance_change < 0:
            reward -= 1.0  # Moving away from food (penalty)
        else:
            reward += 0.1  # No change in distance (small positive)
        
        # Food consumption reward (main objective)
        if food_eaten:
            reward += 15.0  # Increased from 10 to 15
        
        # Death penalty
        if self.game_over:
            reward -= 15.0  # Increased from 10 to 15
        
        # Survival bonus
        reward += 0.1
        
        # Efficiency bonus - reward for eating food quickly
        if food_eaten and self.steps_since_last_food < 20:
            reward += 5.0  # Bonus for eating food quickly
        elif food_eaten and self.steps_since_last_food > 50:
            reward -= 2.0  # Penalty for taking too long
        
        # Strategic rewards
        head = self.snake[0]
        
        # Reward for maintaining open space
        available_space = self._get_total_available_space(head)
        if available_space > 10:
            reward += 0.5  # Bonus for having lots of space
        elif available_space < 5:
            reward -= 0.5  # Penalty for being cramped
        
        # Reward for avoiding dead ends
        if self._is_in_dead_end(head):
            reward -= 1.0  # Penalty for being in a dead end
        
        # Progress milestone rewards
        if food_eaten:
            if len(self.snake) >= 10:
                reward += 2.0  # Bonus for reaching 10 food items
            if len(self.snake) >= 20:
                reward += 3.0  # Bonus for reaching 20 food items
            if len(self.snake) >= 30:
                reward += 5.0  # Bonus for reaching 30 food items
        
        return reward
    
    def _get_total_available_space(self, head: Tuple[int, int]) -> int:
        """Get total available space around the head"""
        total_space = 0
        for direction in ['up', 'down', 'left', 'right']:
            total_space += int(self._get_available_space(head, direction) * 10)
        return total_space
    
    def _is_in_dead_end(self, head: Tuple[int, int]) -> bool:
        """Check if the snake is in a dead end (only one safe direction)"""
        safe_directions = 0
        
        # Check each direction
        for direction in ['up', 'down', 'left', 'right']:
            if self._get_available_space(head, direction) > 0:
                safe_directions += 1
        
        return safe_directions <= 1  # Dead end if only 0 or 1 safe directions
    
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