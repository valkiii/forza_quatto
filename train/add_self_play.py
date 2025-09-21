#!/usr/bin/env python3
"""Add true self-play capability to M1 CNN training."""

import os
import sys
import copy
import torch

# Add current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.cnn_dueling_dqn_agent import CNNDuelingDQNAgent


class SelfPlayAgent:
    """Wrapper for self-play agent that copies current agent state."""
    
    def __init__(self, main_agent: CNNDuelingDQNAgent, player_id: int = 2):
        """Initialize self-play agent."""
        self.main_agent = main_agent
        self.player_id = player_id
        self.name = f"SelfPlay-CNN (Player {player_id})"
        
        # Create a copy of the main agent for self-play
        self.self_play_net = None
        self.update_frequency = 1000  # Update every 1000 episodes
        self.last_update_step = 0
        
    def update_self_play_agent(self):
        """Update self-play agent with current main agent weights."""
        if self.self_play_net is None:
            # Create network copy
            self.self_play_net = CNNDuelingDQNAgent(
                player_id=self.player_id,
                input_channels=self.main_agent.input_channels,
                action_size=self.main_agent.action_size,
                architecture=self.main_agent.architecture,
                seed=42
            )
        
        # Copy current weights (but not optimizer state)
        self.self_play_net.online_net.load_state_dict(
            self.main_agent.online_net.state_dict()
        )
        self.self_play_net.target_net.load_state_dict(
            self.main_agent.target_net.state_dict()
        )
        
        # Set to evaluation mode (no exploration)
        self.self_play_net.epsilon = 0.05  # Slight exploration for diversity
        
        self.last_update_step = self.main_agent.train_step_count
        
    def choose_action(self, board_state, legal_moves):
        """Choose action using current version of main agent."""
        # Update self-play agent periodically
        if (self.self_play_net is None or 
            self.main_agent.train_step_count - self.last_update_step >= self.update_frequency):
            self.update_self_play_agent()
        
        return self.self_play_net.choose_action(board_state, legal_moves)
    
    def reset_episode(self):
        """Reset episode (no-op for self-play agent)."""
        pass
    
    def observe(self, *args, **kwargs):
        """Observe (no-op for self-play agent)."""
        pass


def create_enhanced_opponent_selector(agent, random_opponent, heuristic_opponent, config):
    """Create opponent selector with true self-play capability."""
    
    # Create self-play agent
    self_play_agent = SelfPlayAgent(agent, player_id=2)
    
    # Training phases
    WARMUP_PHASE_END = config["warmup_phase_end"]
    RANDOM_PHASE_END = config["random_phase_end"] 
    HEURISTIC_PHASE_END = config["heuristic_phase_end"]
    SELF_PLAY_START = config["self_play_start"]
    
    def get_current_opponent_enhanced(episode_absolute):
        """Enhanced opponent selection with true self-play."""
        if episode_absolute <= WARMUP_PHASE_END:
            return random_opponent, "Random-Warmup"
        elif episode_absolute <= RANDOM_PHASE_END:
            return random_opponent, "Random"
        elif episode_absolute <= SELF_PLAY_START:
            return heuristic_opponent, "Heuristic"
        else:
            # Mixed phase with TRUE self-play
            import numpy as np
            rand_val = np.random.random()
            
            if rand_val < config['heuristic_preservation_rate']:  # 30%
                return heuristic_opponent, "Heuristic"
            elif rand_val < config['heuristic_preservation_rate'] + config['random_diversity_rate']:  # 10%
                return random_opponent, "Random"
            else:  # 60% - TRUE SELF-PLAY
                return self_play_agent, "Self-Play"
    
    return get_current_opponent_enhanced


if __name__ == "__main__":
    print("🔧 Self-Play Module Ready!")
    print("✅ Import this module in your training script to enable true self-play")
    print("✅ The agent will play against periodically-updated copies of itself")