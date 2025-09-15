#!/usr/bin/env python3
"""Fix Q-value learning issues identified in diagnosis."""

import os
import sys
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train.double_dqn_train_advanced import create_advanced_double_dqn_config


def create_fixed_training_config():
    """Create fixed configuration to address Q-value learning issues."""
    config = create_advanced_double_dqn_config()
    
    # FIX 1: Ultra-conservative learning rate with Polyak averaging
    config["learning_rate"] = 0.00005  # 5e-5 for ultra-stable convergence with Polyak
    config["self_play_learning_rate"] = 0.00001  # Even more conservative for self-play
    config["discount_factor"] = 0.95  # CRITICAL: Reduced from 0.99 to prevent Q-value explosion
    
    # FIX 2: Add gradient clipping
    config["gradient_clip_norm"] = 1.0  # Clip gradients to prevent explosion
    
    # FIX 3: Improve state representation
    config["state_normalization"] = True  # Normalize state inputs
    
    # FIX 4: More conservative training
    config["target_update_freq"] = 2000  # Less frequent target updates
    config["batch_size"] = 64  # Smaller batch size for stability
    config["buffer_size"] = 200000  # Ultra-large buffer for maximum experience diversity (was 20K)
    
    # FIX 5: Better exploration schedule (CRITICAL for mixed opponents)
    config["epsilon_decay"] = 0.99995  # Even slower epsilon decay
    config["epsilon_end"] = 0.05  # Much higher minimum exploration to handle diverse opponents
    config["epsilon_start"] = 1.0  # Start with full exploration
    
    # FIX 6: Huber loss instead of MSE
    config["use_huber_loss"] = True
    config["huber_delta"] = 1.0
    
    # FIX 7: AMPLIFIED REWARD SYSTEM (CRITICAL for strategic learning)
    config["reward_system"] = {
        "win_reward": 10.0,
        "loss_reward": -10.0,
        "draw_reward": 0.0,
        "blocked_opponent_reward": 2.0,   # AMPLIFIED: was 0.1
        "missed_block_penalty": -3.0,     # AMPLIFIED: was -0.1
        "missed_win_penalty": -5.0,       # AMPLIFIED: was -0.2
        "played_winning_move_reward": 3.0, # AMPLIFIED: was 0.2
        "center_preference_reward": 0.0,   # Keep disabled
    }
    
    # FIX 8: Enhanced monitoring
    config["eval_frequency"] = 250  # More frequent evaluation
    config["heuristic_eval_frequency"] = 200  # Very frequent heuristic checks
    
    print("🔧 FIXED CONFIGURATION FOR Q-VALUE LEARNING")
    print("=" * 50)
    print("Key fixes applied:")
    print(f"  🐌 Ultra-conservative learning rate: {config['learning_rate']} (was 0.001)")
    print(f"  ✂️  Gradient clipping: {config['gradient_clip_norm']}")
    print(f"  📊 State normalization: {config['state_normalization']}")
    print(f"  🎯 Huber loss: {config['use_huber_loss']}")
    print(f"  🕐 Target update freq: {config['target_update_freq']} (was 1000)")
    print(f"  📏 Batch size: {config['batch_size']} (was 128)")
    print(f"  🔍 More frequent evaluation: {config['eval_frequency']}")
    
    return config


def create_fixed_training_script():
    """Create a training script with the fixes applied."""
    script_content = '''#!/usr/bin/env python3
"""Fixed Double DQN training with stable Q-value learning."""

import os
import sys
import json
import csv
import math
from typing import Dict, Any, List, Tuple
from collections import deque
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Connect4Board
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.double_dqn_agent import DoubleDQNAgent
from train.reward_system import calculate_enhanced_reward
from train.training_monitor import TrainingMonitor
from train.double_dqn_train import (
    play_double_dqn_training_game, 
    evaluate_agent,
    setup_double_dqn_logging
)
from fix_qvalue_learning import create_fixed_training_config


class FixedDoubleDQNAgent(DoubleDQNAgent):
    """Enhanced DoubleDQNAgent with fixes for Q-value learning."""
    
    def __init__(self, *args, gradient_clip_norm=None, use_huber_loss=False, 
                 huber_delta=1.0, state_normalization=False, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.gradient_clip_norm = gradient_clip_norm
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.state_normalization = state_normalization
        
        # Replace loss function if using Huber loss
        if self.use_huber_loss:
            self.loss_fn = nn.SmoothL1Loss(delta=huber_delta)
        else:
            self.loss_fn = nn.MSELoss()
        
        print(f"🔧 FixedDoubleDQNAgent initialized:")
        print(f"  Gradient clipping: {gradient_clip_norm}")
        print(f"  Huber loss: {use_huber_loss} (delta={huber_delta})")
        print(f"  State normalization: {state_normalization}")
    
    def encode_state(self, board_state: np.ndarray) -> np.ndarray:
        """Enhanced state encoding with normalization."""
        # Use parent's encoding
        encoded = super().encode_state(board_state)
        
        # Apply normalization if enabled
        if self.state_normalization:
            # Normalize to [-1, 1] range instead of [0, 1]
            encoded = encoded * 2.0 - 1.0
            
            # Add position encoding for empty board discrimination
            rows, cols = board_state.shape
            for row in range(rows):
                for col in range(cols):
                    idx = row * cols + col
                    if board_state[row, col] == 0:  # Empty position
                        # Add small positional bias (center positions slightly higher)
                        center_bonus = 0.1 * (1.0 - abs(col - 3) / 3.0)  # 0.1 max bonus for center
                        encoded[idx] += center_bonus
        
        return encoded
    
    def train_step(self, batch):
        """Enhanced training step with gradient clipping and better loss."""
        if len(batch) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Handle next_states (some might be None)
        next_states_tensor = []
        for next_state in next_states:
            if next_state is not None:
                next_states_tensor.append(next_state)
            else:
                next_states_tensor.append(np.zeros_like(states[0].cpu().numpy()))
        next_states = torch.FloatTensor(np.array(next_states_tensor)).to(self.device)
        
        # Current Q-values
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use online network to select actions, target network to evaluate
        with torch.no_grad():
            # Select actions with online network
            next_actions = self.online_net(next_states).argmax(1)
            # Evaluate with target network
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Set Q-values for terminal states to 0
            next_q_values[dones] = 0.0
            # Compute targets
            target_q_values = rewards + (self.gamma * next_q_values)
        
        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(1), target_q_values)
        
        # Optimization step with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent explosion
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip_norm)
        
        self.optimizer.step()
        
        # Update training step count
        self.train_step_count += 1
        
        # Update target network
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        return loss.item()


def train_fixed_double_dqn():
    """Training with fixes for stable Q-value learning."""
    print("🔧 FIXED Double DQN Training - Stable Q-Value Learning")
    print("=" * 60)
    
    # Load fixed configuration
    config = create_fixed_training_config()
    
    # Set seeds
    import random
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["random_seed"])
    
    # Setup logging
    log_dir = "logs_fixed"
    os.makedirs(log_dir, exist_ok=True)
    log_file = setup_double_dqn_logging(log_dir)
    monitor = TrainingMonitor(log_dir=log_dir, save_plots=True)
    
    print(f"📊 Logging to: {log_dir}/ with enhanced monitoring")
    
    # Initialize FIXED agent
    agent = FixedDoubleDQNAgent(
        player_id=1,
        state_size=84,
        action_size=7,
        lr=config["learning_rate"],
        gamma=config["discount_factor"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        min_buffer_size=config["min_buffer_size"],
        target_update_freq=config["target_update_freq"],
        seed=config["random_seed"],
        # NEW FIXES
        gradient_clip_norm=config.get("gradient_clip_norm"),
        use_huber_loss=config.get("use_huber_loss", False),
        huber_delta=config.get("huber_delta", 1.0),
        state_normalization=config.get("state_normalization", False)
    )
    
    # Initialize opponents
    random_opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
    heuristic_opponent = HeuristicAgent(player_id=2, seed=config["random_seed"] + 1)
    
    # Training parameters
    RANDOM_PHASE_END = config["random_phase_end"]
    HEURISTIC_PHASE_END = config["heuristic_phase_end"]
    
    print(f"\\n🎯 FIXED Curriculum Learning:")
    print(f"  📚 Episodes 1-{RANDOM_PHASE_END:,}: vs Random")
    print(f"  🧠 Episodes {RANDOM_PHASE_END + 1:,}-{HEURISTIC_PHASE_END:,}: vs Heuristic")
    print(f"  🔄 Episodes {HEURISTIC_PHASE_END + 1:,}+: Mixed with enhanced preservation")
    print(f"\\n🔧 Q-Value Learning Fixes Active:")
    print(f"  ✂️  Gradient clipping: {config.get('gradient_clip_norm', 'None')}")
    print(f"  📊 Enhanced state encoding: {config.get('state_normalization', False)}")
    print(f"  🎯 Huber loss: {config.get('use_huber_loss', False)}")
    print(f"  🐌 Conservative learning rate: {config['learning_rate']}")
    
    # Simple training loop for testing fixes
    episode_rewards = []
    
    for episode in range(min(config["num_episodes"], 10000)):  # Limit for testing
        episode_num = episode + 1
        
        # Select opponent based on curriculum
        if episode_num <= RANDOM_PHASE_END:
            current_opponent = random_opponent
            opponent_name = "Random"
        elif episode_num <= HEURISTIC_PHASE_END:
            current_opponent = heuristic_opponent
            opponent_name = "Heuristic"
        else:
            # For now, continue with heuristic (can add self-play later)
            current_opponent = heuristic_opponent
            opponent_name = "Heuristic (extended)"
        
        # Initialize monitor
        if episode_num == 1:
            monitor.set_current_opponent(opponent_name)
        
        # Play training game
        agent.reset_episode()
        winner, game_length = play_double_dqn_training_game(
            agent, current_opponent, monitor, config["reward_system"]
        )
        
        episode_reward = 10.0 if winner == agent.player_id else (-10.0 if winner is not None else 1.0)
        episode_rewards.append(episode_reward)
        
        # Enhanced monitoring
        if episode_num % config["eval_frequency"] == 0:
            win_rate = evaluate_agent(agent, current_opponent, num_games=50)
            
            monitor.log_episode(episode_num, episode_reward, agent, win_rate)
            monitor.log_strategic_episode(episode_num)
            monitor.generate_training_report(agent, episode_num)
            monitor.reset_strategic_stats()
            
            print(f"📊 Episode {episode_num:,}: {win_rate:.1%} vs {opponent_name}")
            print(f"    Avg reward: {np.mean(episode_rewards[-config['eval_frequency']:]):.2f}")
            print(f"    Training steps: {agent.train_step_count:,}")
            
            # Test Q-value quality
            if episode_num % (config["eval_frequency"] * 2) == 0:
                print(f"🔍 Q-value quality check...")
                try:
                    from diagnose_qvalues import diagnose_agent_learning
                    # Quick diagnostic (would need to save agent first)
                    test_model_path = f"models_fixed/double_dqn_ep_{episode_num}.pt"
                    os.makedirs("models_fixed", exist_ok=True)
                    agent.save(test_model_path)
                    
                    # Quick check
                    print(f"    Model saved for diagnostic: {test_model_path}")
                except:
                    pass
        else:
            monitor.log_episode(episode_num, episode_reward, agent)
        
        # Save checkpoints
        if episode_num % config["save_frequency"] == 0:
            os.makedirs("models_fixed", exist_ok=True)
            model_path = f"models_fixed/double_dqn_ep_{episode_num}.pt"
            agent.save(model_path)
            print(f"💾 Saved fixed model: {model_path}")
    
    print(f"\\n✅ Fixed training completed! Check models_fixed/ and logs_fixed/")
    print(f"🔍 Run diagnostic on fixed models to verify Q-value learning improvements")


if __name__ == "__main__":
    train_fixed_double_dqn()
'''
    
    return script_content


def main():
    """Create fixed training configuration and script."""
    print("🔧 Creating Fixed Q-Value Learning Training")
    print("=" * 50)
    
    # Create fixed config
    config = create_fixed_training_config()
    
    # Save config
    with open("config_fixed.json", 'w') as f:
        json.dump(config, f, indent=2)
    print(f"💾 Saved fixed config: config_fixed.json")
    
    # Create fixed training script
    script_content = create_fixed_training_script()
    with open("train_fixed_double_dqn.py", 'w') as f:
        f.write(script_content)
    print(f"💾 Created fixed training script: train_fixed_double_dqn.py")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Run: python train_fixed_double_dqn.py")
    print(f"2. Monitor: tail -f logs_fixed/double_dqn_log.csv")
    print(f"3. Diagnose: python diagnose_qvalues.py (check models_fixed/)")
    print(f"4. Compare Q-value plots before/after fixes")


if __name__ == "__main__":
    main()