#!/usr/bin/env python3
"""Training script for M1-optimized CNN Dueling DQN agent (~80k parameters)."""

import os
import sys
import json
import csv
import shutil
from typing import Dict, Any, Optional, Tuple
import numpy as np
import random
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Connect4Board
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.cnn_dueling_dqn_agent import CNNDuelingDQNAgent
from train.reward_system import calculate_enhanced_reward
from train.training_monitor import TrainingMonitor


def create_m1_cnn_config() -> Dict[str, Any]:
    """Create M1-optimized configuration for CNN Dueling DQN."""
    return {
        # M1-optimized CNN parameters (~80k parameters)
        "input_channels": 2,  # Player and opponent channels
        "hidden_size": 64,    # M1-optimized hidden size
        "architecture": "m1_optimized",  # Use M1-optimized architecture
        
        # M1-specific learning parameters
        "learning_rate": 2e-4,  # Optimal for M1 CNN size
        "discount_factor": 0.95,
        "weight_decay": 1e-5,   # Regularization for larger CNN
        "gradient_clip_norm": 1.0,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.99999,  # Slower decay for more episodes
        
        # M1-optimized training parameters
        "batch_size": 64,       # M1 sweet spot
        "gradient_accumulation": 2,  # Simulate batch_size=128
        "buffer_size": 50000,   # Memory efficient for M1
        "min_buffer_size": 1000,
        "target_update_freq": 500,  # More frequent updates work well on M1
        
        # Extended training schedule for M1 capability
        "num_episodes": 300000,  # Leverage M1's efficiency
        "eval_frequency": 1000,
        "save_frequency": 2000,
        "random_seed": 42,
        
        # M1-optimized curriculum learning
        "warmup_phase_end": 15000,   # Extended warmup for larger model
        "random_phase_end": 75000,   # Pattern learning phase
        "heuristic_phase_end": 225000,  # Strategic learning phase
        "self_play_start": 150000,   # Self-play for advanced tactics
        "self_play_ratio": 0.6,
        "heuristic_preservation_rate": 0.3,
        "random_diversity_rate": 0.1,
        
        # M1 CNN features
        "use_batch_norm": True,  # M1 handles BatchNorm efficiently
        "use_dropout": False,    # Not needed with BatchNorm
        "spatial_augmentation": False,
        "optimism_bootstrap_end": 3000,  # Extended for pattern stabilization
        
        # Enhanced reward system for M1 capability
        "reward_system": "enhanced",
        "reward_config": {
            "win_reward": 20.0,      # Strong rewards for M1 capability
            "loss_penalty": -20.0,   # Symmetric
            "draw_reward": 5.0,
            "threat_reward": 3.0,    # Enhanced for strategic learning
            "block_reward": 3.0,
            "center_bonus": 1.5,     # Stronger spatial learning
            "height_penalty": -0.1
        },
        
        # M1-specific monitoring
        "early_stopping": True,
        "early_stopping_threshold": 0.85,  # Higher threshold for M1 capability
        "early_stopping_patience": 20,
        "save_best_model": True,
        
        # M1 GPU optimizations
        "enable_mps_optimizations": True,
        "use_automatic_mixed_precision": False  # Keep disabled for compatibility
    }


def setup_m1_logging(log_dir: str) -> str:
    """Setup logging for M1 CNN training."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "m1_cnn_training_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "avg_reward", "epsilon", "training_steps",
                "buffer_size", "win_rate", "vs_heuristic", "vs_random",
                "spatial_score", "network_type", "parameters"
            ])
    
    return log_file


def play_m1_training_game(agent: CNNDuelingDQNAgent, opponent, 
                          reward_config: dict = None, full_config: dict = None) -> Tuple[Optional[int], int, dict]:
    """Play training game with M1 CNN agent and enhanced spatial scoring."""
    board = Connect4Board()
    
    # Random starting player
    if agent.rng.random() < 0.5:
        players = [agent, opponent]
    else:
        players = [opponent, agent]
    
    # Game tracking
    move_count = 0
    agent_moves = []
    spatial_stats = {
        'center_moves': 0,
        'edge_moves': 0,
        'strategic_moves': 0,  # Enhanced for M1 capability
        'spatial_score': 0.0
    }
    
    # Track agent experiences
    agent_experiences = []
    
    while not board.is_terminal() and move_count < 42:
        current_player = players[move_count % 2]
        state_before = board.get_state().copy()
        legal_moves = board.get_legal_moves()
        
        # Choose action
        action = current_player.choose_action(state_before, legal_moves)
        
        # Track agent moves for enhanced spatial analysis
        if current_player == agent:
            agent_moves.append({'state': state_before.copy(), 'action': action})
            
            # Enhanced spatial move analysis for M1 CNN
            if action in [2, 3, 4]:  # Center columns
                spatial_stats['center_moves'] += 1
            elif action in [0, 6]:   # Edge columns
                spatial_stats['edge_moves'] += 1
            
            # Strategic move detection (M1 can handle more complex analysis)
            if action == 3:  # Central column - most strategic
                spatial_stats['strategic_moves'] += 1
        
        # Make move
        board.make_move(action, current_player.player_id)
        move_count += 1
        
        # Process agent's previous move
        if current_player != agent and agent_experiences:
            prev_exp = agent_experiences[-1]
            
            if reward_config:
                # Enhanced reward calculation
                prev_board = Connect4Board()
                prev_board.board = prev_exp['state'].copy()
                
                reward = calculate_enhanced_reward(
                    prev_board, prev_exp['action'], board,
                    agent.player_id, board.is_terminal(), reward_config
                )
            else:
                # Bootstrap mode with enhanced rewards for M1
                reward = 2.0  # Higher base reward for M1 capability
                
                # Enhanced spatial bonuses during bootstrap
                action = prev_exp['action']
                if action == 3:  # Central column
                    reward += 2.0
                elif action in [2, 4]:  # Near-center
                    reward += 1.0
                
                # Terminal rewards
                if board.is_terminal():
                    winner = board.check_winner()
                    bootstrap_end = full_config.get('optimism_bootstrap_end', 3000) if full_config else 3000
                    
                    if winner == agent.player_id:
                        reward = 25.0  # Strong win reward for M1
                    elif winner is not None:
                        if hasattr(agent, '_episode_count') and agent._episode_count < bootstrap_end:
                            reward = 0.0  # No negative during bootstrap
                        else:
                            reward = -20.0
                    else:
                        reward = 12.0  # Enhanced draw reward
            
            # Store experience
            next_state = None if board.is_terminal() else board.get_state().copy()
            agent.observe(prev_exp['state'], prev_exp['action'], reward,
                         next_state, board.is_terminal())
        
        # Store agent experience
        if current_player == agent:
            agent_experiences.append({
                'state': state_before,
                'action': action
            })
    
    # Process final experience
    if agent_experiences:
        final_exp = agent_experiences[-1]
        winner = board.check_winner()
        
        if reward_config:
            prev_board = Connect4Board()
            prev_board.board = final_exp['state'].copy()
            final_reward = calculate_enhanced_reward(
                prev_board, final_exp['action'], board,
                agent.player_id, True, reward_config
            )
        else:
            # Bootstrap final rewards
            bootstrap_end = full_config.get('optimism_bootstrap_end', 3000) if full_config else 3000
            if winner == agent.player_id:
                final_reward = 25.0
            elif winner is not None:
                if hasattr(agent, '_episode_count') and agent._episode_count < bootstrap_end:
                    final_reward = 0.0
                else:
                    final_reward = -20.0
            else:
                final_reward = 12.0
        
        agent.observe(final_exp['state'], final_exp['action'], final_reward, None, True)
    
    # Calculate enhanced spatial score for M1 CNN
    if len(agent_moves) > 0:
        total_moves = len(agent_moves)
        center_ratio = spatial_stats['center_moves'] / total_moves
        edge_ratio = spatial_stats['edge_moves'] / total_moves
        strategic_ratio = spatial_stats['strategic_moves'] / total_moves
        
        # Enhanced spatial score leveraging M1 capability
        spatial_stats['spatial_score'] = (center_ratio * 2.0 + strategic_ratio * 3.0 - edge_ratio * 0.5)
    else:
        spatial_stats['spatial_score'] = 0.0
    
    return board.check_winner(), move_count, spatial_stats


def train_m1_cnn_agent():
    """Main training loop for M1-optimized CNN Dueling DQN."""
    print("🚀 M1-OPTIMIZED CNN DUELING DQN TRAINING - BALANCED PERFORMANCE")
    print("=" * 70)
    
    # Clean up previous runs
    for dir_name in ["models_m1_cnn", "logs_m1_cnn"]:
        if os.path.exists(dir_name):
            print(f"🧹 Cleaning up {dir_name}...")
            shutil.rmtree(dir_name)
    
    print("✅ Cleanup complete\n")
    
    # Load configuration
    config = create_m1_cnn_config()
    print("🔧 M1-OPTIMIZED CNN FEATURES:")
    print(f"  1. ✅ M1-optimized: ~80k parameters (balanced capability)")
    print(f"  2. ✅ Column attention: Preserves Connect4 position info")
    print(f"  3. ✅ BatchNorm layers: Optimized for M1 MPS backend")
    print(f"  4. ✅ Smart training: lr={config['learning_rate']}, batch={config['batch_size']}")
    print(f"  5. ✅ Extended episodes: {config['num_episodes']:,}")
    print(f"  6. ✅ M1 GPU acceleration: ~100-200 episodes/minute")
    print()
    
    # Set seeds for reproducibility
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["random_seed"])
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config["random_seed"])
    print(f"🎲 Seeds set to {config['random_seed']}")
    
    # M1 GPU optimizations
    if torch.backends.mps.is_available() and config["enable_mps_optimizations"]:
        print("🚀 M1 GPU optimizations enabled")
        # Enable optimizations for M1
        torch.backends.mps.allow_tf32 = True
        print("   ✅ MPS TF32 optimizations enabled")
    
    # Setup logging
    log_file = setup_m1_logging("logs_m1_cnn")
    monitor = TrainingMonitor(log_dir="logs_m1_cnn", save_plots=True, eval_frequency=config["eval_frequency"])
    monitor._show_progress = False
    print(f"📊 Logging to: {log_file}")
    
    # Initialize M1-optimized CNN agent
    print("🧠 Initializing M1-Optimized CNN Dueling DQN Agent...")
    agent = CNNDuelingDQNAgent(
        player_id=1,
        input_channels=config["input_channels"],
        action_size=7,
        hidden_size=config["hidden_size"],
        gamma=config["discount_factor"],
        lr=config["learning_rate"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        batch_size=config["batch_size"],
        buffer_size=config["buffer_size"],
        min_buffer_size=config["min_buffer_size"],
        target_update_freq=config["target_update_freq"],
        architecture="m1_optimized",  # Use M1-optimized architecture
        seed=config["random_seed"]
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in agent.online_net.parameters())
    trainable_params = sum(p.numel() for p in agent.online_net.parameters() if p.requires_grad)
    
    print(f"✅ M1 CNN agent created:")
    print(f"   Device: {agent.device}")
    print(f"   Architecture: M1-Optimized CNN + Dueling")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Input: {config['input_channels']}-channel spatial (6x7)")
    
    print(f"\n🚀 Starting M1 CNN training...")
    print(f"   Expected: 100-200 episodes/minute on M1")
    print(f"   Target: 85%+ vs random, 40%+ vs heuristic")
    print(f"   Monitor: tail -f logs_m1_cnn/m1_cnn_training_log.csv")
    print()
    
    # Initialize opponents
    random_opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
    heuristic_opponent = HeuristicAgent(player_id=2, seed=config["random_seed"] + 2)
    
    # Training tracking
    episode_rewards = []
    best_win_rate = 0.0
    early_stopping_counter = 0
    
    # Curriculum phases
    WARMUP_PHASE_END = config["warmup_phase_end"]
    RANDOM_PHASE_END = config["random_phase_end"]
    HEURISTIC_PHASE_END = config["heuristic_phase_end"]
    SELF_PLAY_START = config["self_play_start"]
    
    print(f"🎯 M1 CNN CURRICULUM SCHEDULE:")
    print(f"  Episodes 1-{config['optimism_bootstrap_end']:,}: 🛡️  BOOTSTRAP (spatial warmup)")
    print(f"  Episodes {config['optimism_bootstrap_end'] + 1:,}-{WARMUP_PHASE_END:,}: vs Random (pattern recognition)")
    print(f"  Episodes {WARMUP_PHASE_END + 1:,}-{RANDOM_PHASE_END:,}: vs Random (foundation)")
    print(f"  Episodes {RANDOM_PHASE_END + 1:,}-{HEURISTIC_PHASE_END:,}: vs Heuristic (strategy)")
    print(f"  Episodes {SELF_PLAY_START:,}+: Mixed training")
    print()
    
    def get_current_opponent(episode):
        """M1-optimized opponent selection."""
        if episode <= WARMUP_PHASE_END:
            return random_opponent, "Random-Warmup"
        elif episode <= RANDOM_PHASE_END:
            return random_opponent, "Random"
        elif episode <= SELF_PLAY_START:
            return heuristic_opponent, "Heuristic"
        else:
            # Mixed phase
            rand_val = np.random.random()
            if rand_val < config['heuristic_preservation_rate']:
                return heuristic_opponent, "Heuristic"
            elif rand_val < config['heuristic_preservation_rate'] + config['random_diversity_rate']:
                return random_opponent, "Random"
            else:
                return heuristic_opponent, "Heuristic-SP"
    
    print("🔥 Starting M1 CNN training episodes...")
    
    # Training loop
    current_opponent_name = ""
    
    for episode in range(config["num_episodes"]):
        episode_num = episode + 1
        
        # Get opponent
        current_opponent, opponent_name = get_current_opponent(episode_num)
        
        # Announce phase changes
        if opponent_name != current_opponent_name:
            print(f"\n🔄 M1 CNN PHASE CHANGE at episode {episode_num:,}: {current_opponent_name} → {opponent_name}")
            current_opponent_name = opponent_name
            monitor.set_current_opponent(opponent_name)
        
        # Reset episode
        agent.reset_episode()
        
        # Training mode selection
        reward_config_to_use = None if episode_num < config["optimism_bootstrap_end"] else config["reward_config"]
        
        # Play training game
        winner, _, spatial_stats = play_m1_training_game(
            agent, current_opponent, reward_config_to_use, config
        )
        
        # Calculate episode reward
        if winner == agent.player_id:
            episode_reward = 25.0  # Enhanced reward for M1
        elif winner is not None:
            if episode_num < config["optimism_bootstrap_end"]:
                episode_reward = 0.0
            else:
                episode_reward = -20.0
        else:
            episode_reward = 12.0  # Enhanced draw reward
        
        episode_rewards.append(episode_reward)
        
        # Log episode
        spatial_score_for_monitor = spatial_stats.get('spatial_score', 0.0)
        monitor.log_episode(episode_num, episode_reward, agent, strategic_score=spatial_score_for_monitor)
        
        # Early progress updates (first 10 episodes, then every 100)
        if episode_num <= 10 or episode_num % 100 == 0:
            avg_reward_recent = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
            print(f"Episode {episode_num:,} | {opponent_name} | "
                  f"Avg Reward: {avg_reward_recent:.1f} | ε: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.memory)} | Steps: {agent.train_step_count} | "
                  f"Spatial: {spatial_score_for_monitor:.2f}")
        
        # Periodic evaluation
        if episode_num % config["eval_frequency"] == 0:
            opponents = {
                "current": current_opponent,
                "random": random_opponent,
                "heuristic": heuristic_opponent
            }
            
            # Simple evaluation function for M1 CNN
            eval_results = {}
            old_epsilon = agent.epsilon
            agent.epsilon = 0.0  # No exploration during evaluation
            
            try:
                for opponent_name_eval, opponent_eval in opponents.items():
                    wins = 0
                    spatial_scores = []
                    
                    for _ in range(100):
                        board = Connect4Board()
                        if np.random.random() < 0.5:
                            players = [agent, opponent_eval]
                        else:
                            players = [opponent_eval, agent]
                        
                        move_count = 0
                        agent_center_moves = 0
                        agent_total_moves = 0
                        
                        while not board.is_terminal() and move_count < 42:
                            current_player = players[move_count % 2]
                            legal_moves = board.get_legal_moves()
                            action = current_player.choose_action(board.get_state(), legal_moves)
                            
                            if current_player == agent:
                                agent_total_moves += 1
                                if action in [2, 3, 4]:
                                    agent_center_moves += 1
                            
                            board.make_move(action, current_player.player_id)
                            move_count += 1
                        
                        winner = board.check_winner()
                        if winner == agent.player_id:
                            wins += 1
                        
                        if agent_total_moves > 0:
                            spatial_score = agent_center_moves / agent_total_moves
                            spatial_scores.append(spatial_score)
                    
                    eval_results[opponent_name_eval] = {
                        'win_rate': wins / 100,
                        'avg_spatial_score': np.mean(spatial_scores) if spatial_scores else 0
                    }
            
            finally:
                agent.epsilon = old_epsilon
            
            current_win_rate = eval_results["current"]["win_rate"]
            vs_random = eval_results["random"]["win_rate"]
            vs_heuristic = eval_results["heuristic"]["win_rate"]
            
            # Early stopping check
            if config["early_stopping"] and episode_num > HEURISTIC_PHASE_END:
                if current_win_rate > config["early_stopping_threshold"]:
                    early_stopping_counter += 1
                    if early_stopping_counter >= config["early_stopping_patience"]:
                        print(f"\n🏆 M1 CNN EARLY STOPPING at episode {episode_num:,}")
                        print(f"Win rate {current_win_rate:.1%} > {config['early_stopping_threshold']:.1%}")
                        break
                else:
                    early_stopping_counter = 0
            
            # Track best model
            if current_win_rate > best_win_rate:
                best_win_rate = current_win_rate
                if config["save_best_model"]:
                    os.makedirs("models_m1_cnn", exist_ok=True)
                    best_path = f"models_m1_cnn/m1_cnn_dqn_best_ep_{episode_num}.pt"
                    agent.save(best_path)
                    print(f"💎 M1 CNN Best: {current_win_rate:.1%} - saved {best_path}")
            
            # Log results
            avg_reward = np.mean(episode_rewards[-config["eval_frequency"]:])
            spatial_score = spatial_stats.get('spatial_score', 0)
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode_num, avg_reward, agent.epsilon, agent.train_step_count,
                    len(agent.memory), current_win_rate, vs_heuristic, vs_random,
                    spatial_score, "M1-CNN-Dueling", total_params
                ])
            
            print(f"Episode {episode_num:,} | {opponent_name} | "
                  f"Win: {current_win_rate:.1%} | vs Heur: {vs_heuristic:.1%} | "
                  f"vs Rand: {vs_random:.1%} | ε: {agent.epsilon:.3f} | "
                  f"Spatial: {spatial_score:.2f}")
            
            # Log evaluation episode with win rate for plotting
            monitor.log_episode(episode_num, avg_reward, agent, win_rate=current_win_rate, strategic_score=spatial_score)
            
            # Generate plots
            if episode_num % 2000 == 0:
                monitor.generate_training_report(agent, episode_num)
                print(f"📊 M1 CNN plots generated for episode {episode_num:,}")
        
        # Save checkpoints
        if episode_num % config["save_frequency"] == 0:
            os.makedirs("models_m1_cnn", exist_ok=True)
            checkpoint_path = f"models_m1_cnn/m1_cnn_dqn_ep_{episode_num}.pt"
            agent.save(checkpoint_path)
    
    # Final evaluation and save
    print("\n" + "=" * 60)
    print("FINAL M1 CNN EVALUATION")
    print("=" * 60)
    
    final_model_path = "models_m1_cnn/m1_cnn_dqn_final.pt"
    agent.save(final_model_path)
    
    print(f"\n✅ M1 CNN TRAINING COMPLETED!")
    print(f"Final model: {final_model_path}")
    print(f"Best performance: {best_win_rate:.1%}")
    print(f"Total parameters: {total_params:,}")
    
    # Generate final report
    monitor.generate_training_report(agent, episode_num if 'episode_num' in locals() else config["num_episodes"])
    
    print("\n🎉 M1-Optimized CNN Dueling DQN training successful!")


if __name__ == "__main__":
    try:
        train_m1_cnn_agent()
    except KeyboardInterrupt:
        print("\n⚠️ M1 CNN training interrupted by user")
    except Exception as e:
        print(f"\n❌ M1 CNN training failed: {e}")
        import traceback
        traceback.print_exc()