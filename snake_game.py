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
        """Get the current state representation for DQN - simplified but complete"""
        head = self.snake[0]
        food = self.food
        
        # Current direction (one-hot encoded)
        moving_up = 1.0 if self.direction == Direction.UP else 0.0
        moving_down = 1.0 if self.direction == Direction.DOWN else 0.0
        moving_left = 1.0 if self.direction == Direction.LEFT else 0.0
        moving_right = 1.0 if self.direction == Direction.RIGHT else 0.0
        
        # Danger detection based on current heading
        danger_straight = self._is_dangerous_ahead()
        danger_left = self._is_dangerous_left()
        danger_right = self._is_dangerous_right()
        
        # Food direction relative to current heading
        food_left = self._is_food_left()
        food_right = self._is_food_right()
        food_up = self._is_food_up()
        food_down = self._is_food_down()
        
        # Additional context
        snake_length = len(self.snake) / 20.0  # Normalized length
        
        state = np.array([
            danger_straight,
            danger_left,
            danger_right,
            moving_left,
            moving_right,
            moving_up,
            moving_down,
            food_left,
            food_right,
            food_up,
            food_down,
            snake_length
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
        """Calculate reward based on the improved reward function"""
        reward = 0.0
        
        # Small negative reward every step to encourage faster food collection
        reward -= 0.1
        
        # Food proximity reward (small positive for moving closer)
        new_distance = abs(self.snake[0][0] - self.food[0]) + abs(self.snake[0][1] - self.food[1])
        distance_change = old_distance - new_distance
        
        if distance_change > 0:
            reward += 0.5  # Small reward for moving closer to food
        elif distance_change < 0:
            reward -= 0.2  # Small penalty for moving away from food
        
        # Food consumption reward (main objective)
        if food_eaten:
            reward += 10.0  # Large reward for eating food
        
        # Death penalty
        if self.game_over:
            reward -= 10.0  # Large penalty for dying
        
        return reward
    
    def _is_near_body(self, head: Tuple[int, int]) -> bool:
        """Check if head is adjacent to any body segment"""
        for body_segment in self.snake[1:]:  # Exclude head
            if abs(head[0] - body_segment[0]) + abs(head[1] - body_segment[1]) == 1:
                return True
        return False
    
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

    def _is_dangerous_ahead(self) -> float:
        """Check if moving straight ahead is dangerous"""
        head = self.snake[0]
        if self.direction == Direction.UP:
            next_pos = (head[0] - 1, head[1])
        elif self.direction == Direction.DOWN:
            next_pos = (head[0] + 1, head[1])
        elif self.direction == Direction.LEFT:
            next_pos = (head[0], head[1] - 1)
        else:  # RIGHT
            next_pos = (head[0], head[1] + 1)
        
        return self._is_dangerous(next_pos)
    
    def _is_dangerous_left(self) -> float:
        """Check if turning left is dangerous"""
        head = self.snake[0]
        if self.direction == Direction.UP:
            next_pos = (head[0], head[1] - 1)
        elif self.direction == Direction.DOWN:
            next_pos = (head[0], head[1] + 1)
        elif self.direction == Direction.LEFT:
            next_pos = (head[0] + 1, head[1])
        else:  # RIGHT
            next_pos = (head[0] - 1, head[1])
        
        return self._is_dangerous(next_pos)
    
    def _is_dangerous_right(self) -> float:
        """Check if turning right is dangerous"""
        head = self.snake[0]
        if self.direction == Direction.UP:
            next_pos = (head[0], head[1] + 1)
        elif self.direction == Direction.DOWN:
            next_pos = (head[0], head[1] - 1)
        elif self.direction == Direction.LEFT:
            next_pos = (head[0] - 1, head[1])
        else:  # RIGHT
            next_pos = (head[0] + 1, head[1])
        
        return self._is_dangerous(next_pos)
    
    def _is_food_left(self) -> float:
        """Check if food is to the left of current heading"""
        head = self.snake[0]
        food = self.food
        
        if self.direction == Direction.UP:
            return 1.0 if food[1] < head[1] else 0.0
        elif self.direction == Direction.DOWN:
            return 1.0 if food[1] > head[1] else 0.0
        elif self.direction == Direction.LEFT:
            return 1.0 if food[0] > head[0] else 0.0
        else:  # RIGHT
            return 1.0 if food[0] < head[0] else 0.0
    
    def _is_food_right(self) -> float:
        """Check if food is to the right of current heading"""
        head = self.snake[0]
        food = self.food
        
        if self.direction == Direction.UP:
            return 1.0 if food[1] > head[1] else 0.0
        elif self.direction == Direction.DOWN:
            return 1.0 if food[1] < head[1] else 0.0
        elif self.direction == Direction.LEFT:
            return 1.0 if food[0] < head[0] else 0.0
        else:  # RIGHT
            return 1.0 if food[0] > head[0] else 0.0
    
    def _is_food_up(self) -> float:
        """Check if food is ahead of current heading"""
        head = self.snake[0]
        food = self.food
        
        if self.direction == Direction.UP:
            return 1.0 if food[0] < head[0] else 0.0
        elif self.direction == Direction.DOWN:
            return 1.0 if food[0] > head[0] else 0.0
        elif self.direction == Direction.LEFT:
            return 1.0 if food[1] < head[1] else 0.0
        else:  # RIGHT
            return 1.0 if food[1] > head[1] else 0.0
    
    def _is_food_down(self) -> float:
        """Check if food is behind current heading"""
        head = self.snake[0]
        food = self.food
        
        if self.direction == Direction.UP:
            return 1.0 if food[0] > head[0] else 0.0
        elif self.direction == Direction.DOWN:
            return 1.0 if food[0] < head[0] else 0.0
        elif self.direction == Direction.LEFT:
            return 1.0 if food[1] > head[1] else 0.0
        else:  # RIGHT
            return 1.0 if food[1] < head[1] else 0.0

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