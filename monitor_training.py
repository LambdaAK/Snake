#!/usr/bin/env python3

import os
import glob
import time

def monitor_training():
    """Monitor the DQN training progress"""
    print("Monitoring DQN training progress...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            # Check for saved models
            model_files = glob.glob("agents/snake_dqn_episode_*.pth")
            model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            if model_files:
                latest_model = model_files[-1]
                episode = int(latest_model.split('_')[-1].split('.')[0])
                print(f"Latest saved model: Episode {episode}")
                print(f"Total models saved: {len(model_files)}")
            else:
                print("No models saved yet (training in early stages)")
            
            print("-" * 50)
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

if __name__ == "__main__":
    monitor_training() 