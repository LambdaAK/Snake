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
            
            # Remove the 500-step limit - let it run until game over
            while True:
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
    steps_list = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        # Remove the 500-step limit - let it run until game over
        while True:
            # Choose action (pure exploitation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            q_values = agent.q_network(state_tensor)
            action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(env.score)
        rewards.append(episode_reward)
        steps_list.append(steps)
        
        if (episode + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            avg_reward = np.mean(rewards[-10:])
            avg_steps = np.mean(steps_list[-10:])
            print(f"Episodes {episode-9}-{episode+1}: Avg Score: {avg_score:.1f}, Avg Reward: {avg_reward:.1f}, Avg Steps: {avg_steps:.1f}")
    
    # Final summary
    print(f"\n=== Final Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {np.mean(scores):.1f}")
    print(f"Best Score: {max(scores)}")
    print(f"Average Reward: {np.mean(rewards):.1f}")
    print(f"Average Steps: {np.mean(steps_list):.1f}")
    print(f"Score Std Dev: {np.std(scores):.1f}")

def get_user_input():
    """Get user preferences through interactive input"""
    print("=== DQN Snake Game Model Runner ===")
    print()
    
    # Check for available models
    model_files = glob.glob("agents/snake_dqn_episode_*.pth")
    if not model_files:
        print("❌ No saved models found in agents/ directory!")
        print("Please train a model first using: python dqn_agent.py")
        return None
    
    # Show available models
    print("Available models:")
    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for i, model_file in enumerate(model_files[-5:]):  # Show last 5 models
        episode = int(model_file.split('_')[-1].split('.')[0])
        print(f"  {i+1}. Episode {episode}: {model_file}")
    
    print()
    
    # Get model choice
    while True:
        try:
            choice = input("Choose model (1-5, or 'latest' for most recent): ").strip()
            if choice.lower() == 'latest':
                model_path = model_files[-1]
                break
            else:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(model_files[-5:]):
                    model_path = model_files[-5:][choice_idx]
                    break
                else:
                    print("Invalid choice. Please enter 1-5 or 'latest'.")
        except ValueError:
            print("Invalid input. Please enter a number or 'latest'.")
    
    print()
    
    # Get run mode
    print("Run modes:")
    print("  1. Visual mode (watch the AI play)")
    print("  2. Fast mode (performance testing)")
    
    while True:
        try:
            mode = input("Choose mode (1 or 2): ").strip()
            if mode in ['1', '2']:
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter 1 or 2.")
    
    print()
    
    # Get number of episodes
    while True:
        try:
            if mode == '1':
                episodes = input("Number of episodes to watch (default: 3): ").strip()
                if episodes == "":
                    episodes = 3
                else:
                    episodes = int(episodes)
            else:
                episodes = input("Number of episodes for testing (default: 100): ").strip()
                if episodes == "":
                    episodes = 100
                else:
                    episodes = int(episodes)
            
            if episodes > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Get delay for visual mode
    delay = 0.1
    if mode == '1':
        print()
        delay_input = input("Delay between steps in seconds (default: 0.1): ").strip()
        if delay_input != "":
            try:
                delay = float(delay_input)
            except ValueError:
                print("Invalid delay, using default 0.1 seconds.")
    
    return {
        'model_path': model_path,
        'mode': mode,
        'episodes': episodes,
        'delay': delay
    }

if __name__ == "__main__":
    # Get user preferences
    config = get_user_input()
    
    if config is None:
        exit(1)
    
    print(f"\nStarting with:")
    print(f"  Model: {config['model_path']}")
    print(f"  Mode: {'Visual' if config['mode'] == '1' else 'Fast'}")
    print(f"  Episodes: {config['episodes']}")
    if config['mode'] == '1':
        print(f"  Delay: {config['delay']} seconds")
    
    print("\nPress Enter to start...")
    input()
    
    # Run the model
    if config['mode'] == '1':
        run_model_with_visualization(config['model_path'], config['episodes'], config['delay'])
    else:
        run_model_fast(config['model_path'], config['episodes']) 