#!/usr/bin/env python3

import torch
import numpy as np
import time
import os
import glob
from snake_game import SnakeGame
from dqn_agent import DQNAgent

def load_latest_model(agent):
    """Load the most recent saved model"""
    model_files = glob.glob("agents/snake_dqn_episode_*.pth")
    if not model_files:
        print("No saved models found in agents/ directory")
        return False
    
    # Sort by episode number and get the latest
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_model = model_files[-1]
    episode = int(latest_model.split('_')[-1].split('.')[0])
    
    print(f"Loading model from episode {episode}: {latest_model}")
    agent.load_model(latest_model)
    return True

def run_model_with_visualization(model_path=None, num_episodes=5, delay=0.1):
    """Run a trained model and visualize the gameplay"""
    # Initialize environment and agent
    env = SnakeGame(width=20, height=15)
    agent = DQNAgent()
    
    # Load model
    if model_path:
        print(f"Loading model from: {model_path}")
        agent.load_model(model_path)
    else:
        if not load_latest_model(agent):
            return
    
    print(f"Model loaded successfully!")
    print(f"Epsilon: {agent.epsilon:.3f}")
    print(f"Running {num_episodes} episodes...")
    print("Press Ctrl+C to stop")
    
    total_score = 0
    total_reward = 0
    
    try:
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1} ===")
            state = env.reset()
            episode_reward = 0
            steps = 0
            
            while steps < 500:  # Max steps per episode
                # Clear screen and render
                env._clear_screen()
                env._render()
                print(f"Episode {episode + 1} | Step {steps} | Score: {env.score}")
                print(f"Total Reward: {episode_reward:.1f}")
                
                # Choose action (use epsilon=0 for pure exploitation)
                if np.random.random() < 0.01:  # Small amount of exploration
                    action = np.random.randint(0, 4)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    q_values = agent.q_network(state_tensor)
                    action = q_values.argmax().item()
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Update state and stats
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Small delay to make it watchable
                time.sleep(delay)
                
                if done:
                    break
            
            total_score += env.score
            total_reward += episode_reward
            
            print(f"\nEpisode {episode + 1} finished!")
            print(f"Final Score: {env.score}")
            print(f"Final Reward: {episode_reward:.1f}")
            print(f"Steps taken: {steps}")
            
            # Wait a moment before next episode
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Episodes played: {episode + 1}")
    print(f"Average score: {total_score / (episode + 1):.1f}")
    print(f"Average reward: {total_reward / (episode + 1):.1f}")

def run_model_fast(model_path=None, num_episodes=100):
    """Run a trained model quickly without visualization for performance testing"""
    # Initialize environment and agent
    env = SnakeGame(width=20, height=15)
    agent = DQNAgent()
    
    # Load model
    if model_path:
        print(f"Loading model from: {model_path}")
        agent.load_model(model_path)
    else:
        if not load_latest_model(agent):
            return
    
    print(f"Running {num_episodes} episodes quickly...")
    
    scores = []
    rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            # Choose action (pure exploitation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.q_network(state_tensor)
            action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        scores.append(env.score)
        rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            avg_reward = np.mean(rewards[-10:])
            print(f"Episodes {episode-9}-{episode+1}: Avg Score: {avg_score:.1f}, Avg Reward: {avg_reward:.1f}")
    
    # Final summary
    print(f"\n=== Final Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {np.mean(scores):.1f}")
    print(f"Best Score: {max(scores)}")
    print(f"Average Reward: {np.mean(rewards):.1f}")
    print(f"Score Std Dev: {np.std(scores):.1f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a trained DQN model')
    parser.add_argument('--model', type=str, help='Path to specific model file')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--fast', action='store_true', help='Run quickly without visualization')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between steps (for visualization)')
    
    args = parser.parse_args()
    
    if args.fast:
        run_model_fast(args.model, args.episodes)
    else:
        run_model_with_visualization(args.model, args.episodes, args.delay) 