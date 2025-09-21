#!/usr/bin/env python3
"""Quick test of M1 CNN training loop."""

import os
import sys
import numpy as np
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.cnn_dueling_dqn_agent import CNNDuelingDQNAgent
from agents.random_agent import RandomAgent
from train.cnn_m1_train import play_m1_training_game

def test_m1_training():
    print("🔧 Testing M1 CNN Training Loop...")
    
    # Create M1-optimized agent
    agent = CNNDuelingDQNAgent(
        player_id=1,
        architecture="m1_optimized",
        batch_size=32,  # Smaller for testing
        buffer_size=1000,
        min_buffer_size=100,
        seed=42
    )
    
    # Create opponent
    opponent = RandomAgent(player_id=2, seed=43)
    
    print(f"✅ Agent created: {sum(p.numel() for p in agent.online_net.parameters()):,} parameters")
    print(f"✅ Device: {agent.device}")
    
    # Test a few training episodes
    print("\n🚀 Running test episodes...")
    
    for episode in range(5):  # Just 5 episodes for testing
        episode_num = episode + 1
        
        # Reset episode
        agent.reset_episode()
        
        # Play training game
        winner, game_length, spatial_stats = play_m1_training_game(agent, opponent)
        
        # Calculate reward
        if winner == agent.player_id:
            episode_reward = 25.0
        elif winner is not None:
            episode_reward = -20.0
        else:
            episode_reward = 12.0
        
        spatial_score = spatial_stats.get('spatial_score', 0.0)
        
        print(f"Episode {episode_num}: Winner={winner}, Length={game_length}, "
              f"Reward={episode_reward:.1f}, Spatial={spatial_score:.2f}, "
              f"ε={agent.epsilon:.3f}, Buffer={len(agent.memory)}")
    
    print("\n✅ M1 CNN training loop working correctly!")
    print("🚀 Ready for full training!")

if __name__ == "__main__":
    test_m1_training()