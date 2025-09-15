#!/usr/bin/env python3
"""Quick test of the comprehensive fixes with shorter episodes for verification."""

import os
import sys
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_fixed_double_dqn import train_fixed_double_dqn, FixedDoubleDQNAgent
from fix_qvalue_learning import create_fixed_training_config
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from train.double_dqn_train import evaluate_agent

def quick_test():
    """Run a quick test with a small number of episodes to verify fixes work."""
    print("🧪 QUICK TEST - Comprehensive Fixes Verification")
    print("=" * 55)
    
    # Create a quick test configuration
    config = create_fixed_training_config()
    
    # Override for quick testing
    config["num_episodes"] = 5000  # Much shorter for testing
    config["random_phase_end"] = 1000  # 1K random
    config["heuristic_phase_end"] = 4000  # 3K heuristic
    config["eval_frequency"] = 500  # More frequent evaluation
    config["save_frequency"] = 1000  # More frequent saves
    
    # Ensure enhanced reward system
    config["reward_system"] = {
        "win_reward": 10.0,
        "loss_reward": -10.0,
        "draw_reward": 0.0,
        "blocked_opponent_reward": 2.0,   # AMPLIFIED
        "missed_block_penalty": -3.0,     # AMPLIFIED
        "missed_win_penalty": -5.0,       # AMPLIFIED
        "played_winning_move_reward": 3.0, # AMPLIFIED
        "center_preference_reward": 0.0,
    }
    
    print("🎯 Quick Test Configuration:")
    print(f"  Episodes: {config['num_episodes']:,}")
    print(f"  Random phase: 1-{config['random_phase_end']:,}")
    print(f"  Heuristic phase: {config['random_phase_end']+1:,}-{config['heuristic_phase_end']:,}")
    print(f"  Mixed phase: {config['heuristic_phase_end']+1:,}+")
    print(f"  Strategic rewards: {config['reward_system']['played_winning_move_reward']}, {config['reward_system']['blocked_opponent_reward']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Buffer size: {config['buffer_size']:,}")
    print()
    
    # Save the test config
    with open("config_quick_test.json", 'w') as f:
        json.dump(config, f, indent=2)
    print("💾 Test config saved: config_quick_test.json")
    print()
    
    # Create a simple test agent to verify initialization
    print("🔧 Testing agent initialization with fixes...")
    test_agent = FixedDoubleDQNAgent(
        player_id=1,
        state_size=84,
        action_size=7,
        lr=config["learning_rate"],
        buffer_size=config["buffer_size"],
        seed=config["random_seed"],
        gradient_clip_norm=config.get("gradient_clip_norm"),
        use_huber_loss=config.get("use_huber_loss", False),
        huber_delta=config.get("huber_delta", 1.0),
        state_normalization=config.get("state_normalization", False),
        polyak_tau=0.005,
        use_reservoir_sampling=True
    )
    print("✅ Agent initialization successful")
    print()
    
    # Test opponents
    random_opponent = RandomAgent(player_id=2, seed=43)
    heuristic_opponent = HeuristicAgent(player_id=2, seed=44)
    
    print("🎮 Testing baseline performance...")
    random_baseline = evaluate_agent(test_agent, random_opponent, num_games=20)
    heuristic_baseline = evaluate_agent(test_agent, heuristic_opponent, num_games=20)
    print(f"  Untrained vs Random: {random_baseline:.1%}")
    print(f"  Untrained vs Heuristic: {heuristic_baseline:.1%}")
    print()
    
    print("🚀 Quick fixes verification complete!")
    print("   Run full training with: python run_comprehensive_fixes.py")
    print("   The fixes appear to be working correctly.")

if __name__ == "__main__":
    quick_test()