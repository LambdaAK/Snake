#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque
from typing import List, Tuple, Optional
from snake_game import SnakeGame

class DQN(nn.Module):
    def __init__(self, input_size: int = 14, hidden_size: int = 64, output_size: int = 4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class ExperienceReplay:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size: int = 14, action_size: int = 4, hidden_size: int = 64):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        
        # Hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.batch_size = 32
        self.target_update_freq = 100
        
        # Networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = ExperienceReplay(10000)
        
        # Training stats
        self.episode_count = 0
        self.target_update_count = 0
        
        # Create agents directory
        os.makedirs("agents", exist_ok=True)
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays first to avoid the warning
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, episode: int):
        """Save the model"""
        # Ensure agents directory exists
        os.makedirs("agents", exist_ok=True)
        
        torch.save({
            'episode': episode,
            'model_state_dict': self.q_network.state_dict(),
            'target_model_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count
        }, f"agents/snake_dqn_episode_{episode}.pth")
    
    def load_model(self, filepath: str):
        """Load a saved model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_count = checkpoint['episode_count']

def train_dqn():
    """Main training function"""
    # Initialize environment and agent
    env = SnakeGame(width=20, height=15)
    agent = DQNAgent()
    
    # Training variables
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    
    print("Starting DQN training for Snake Game...")
    print(f"Device: {agent.device}")
    print("Press Ctrl+C to stop training")
    
    try:
        episode = 0
        while True:  # Train indefinitely
            episode += 1
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 500:  # Max steps per episode
                # Choose action
                action = agent.act(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Update state and stats
                state = next_state
                total_reward += reward
                steps += 1
                
                # Train the network
                agent.replay()
                
                if done:
                    break
            
            # Update target network every 100 episodes
            agent.target_update_count += 1
            if agent.target_update_count >= 100:
                agent.update_target_network()
                agent.target_update_count = 0
            
            # Store episode stats
            episode_rewards.append(total_reward)
            episode_scores.append(info['score'])
            episode_lengths.append(steps)
            
            # Print progress every 100 episodes
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_score = np.mean(episode_scores[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                print(f"Episode {episode:6d} | Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Score: {avg_score:4.1f} | Avg Length: {avg_length:4.1f} | "
                      f"Epsilon: {agent.epsilon:.3f}")
            
            # Save model every 1000 episodes
            if episode % 1000 == 0:
                agent.save_model(episode)
                print(f"Model saved at episode {episode}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save final model
        agent.save_model(episode)
        print(f"Final model saved at episode {episode}")
        
        # Print final stats
        if episode_rewards:
            print(f"\nTraining Summary:")
            print(f"Total Episodes: {episode}")
            print(f"Final Average Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"Final Average Score (last 100): {np.mean(episode_scores[-100:]):.2f}")
            print(f"Best Score: {max(episode_scores)}")

if __name__ == "__main__":
    train_dqn() 