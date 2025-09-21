#!/usr/bin/env python3
"""Short test of M1 CNN training script."""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and modify config
from train.cnn_m1_train import create_m1_cnn_config, train_m1_cnn_agent

# Temporarily patch the config for short test
original_config = create_m1_cnn_config

def short_config():
    config = original_config()
    config["num_episodes"] = 50  # Just 50 episodes for testing
    config["eval_frequency"] = 10  # Evaluate every 10 episodes
    config["save_frequency"] = 20  # Save every 20 episodes
    return config

# Monkey patch for testing
import train.cnn_m1_train
train.cnn_m1_train.create_m1_cnn_config = short_config

if __name__ == "__main__":
    print("🧪 Running short M1 CNN training test (50 episodes)...")
    try:
        train_m1_cnn_agent()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()