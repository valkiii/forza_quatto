#!/usr/bin/env python3
"""Resume M1 CNN training from existing checkpoint."""

import os
import sys
import json
import csv
import glob
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
from train.cnn_m1_train import create_m1_cnn_config, play_m1_training_game


class SelfPlayAgent:
    """True self-play agent that copies main agent for competitive training."""
    
    def __init__(self, main_agent: CNNDuelingDQNAgent, player_id: int = 2):
        """Initialize self-play agent."""
        self.main_agent = main_agent
        self.player_id = player_id
        self.name = f"SelfPlay-CNN (Player {player_id})"
        
        # Create a copy of the main agent for self-play
        self.self_play_net = None
        self.update_frequency = 500  # Update every 500 episodes for variety
        self.last_update_episode = 0
        
    def update_self_play_agent(self, current_episode):
        """Update self-play agent with current main agent weights."""
        if self.self_play_net is None:
            # Create network copy (use default hidden_size since M1 architecture doesn't store it)
            self.self_play_net = CNNDuelingDQNAgent(
                player_id=self.player_id,
                input_channels=self.main_agent.input_channels,
                action_size=self.main_agent.action_size,
                hidden_size=48,  # M1-optimized default
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
        
        # Set to evaluation mode with slight exploration for diversity
        self.self_play_net.epsilon = 0.05  # Small exploration for gameplay variety
        
        self.last_update_episode = current_episode
        
    def choose_action(self, board_state, legal_moves, current_episode=None):
        """Choose action using current version of main agent."""
        # Update self-play agent periodically
        if (self.self_play_net is None or 
            (current_episode is not None and current_episode - self.last_update_episode >= self.update_frequency)):
            self.update_self_play_agent(current_episode or 0)
        
        return self.self_play_net.choose_action(board_state, legal_moves)
    
    def reset_episode(self):
        """Reset episode (no-op for self-play agent)."""
        pass
    
    def observe(self, *args, **kwargs):
        """Observe (no-op for self-play agent)."""
        pass


def find_latest_model(model_dir: str = "models_m1_cnn") -> Optional[str]:
    """Find the latest trained M1 CNN model."""
    if not os.path.exists(model_dir):
        return None
    
    # Look for final model first, then best models, then episode models
    candidates = [
        os.path.join(model_dir, "m1_cnn_dqn_final.pt"),
        *glob.glob(os.path.join(model_dir, "m1_cnn_dqn_best_ep_*.pt")),
        *glob.glob(os.path.join(model_dir, "m1_cnn_dqn_ep_*.pt"))
    ]
    
    existing_models = [f for f in candidates if os.path.exists(f)]
    if not existing_models:
        return None
    
    # Return the most recent model
    return max(existing_models, key=os.path.getmtime)


def extract_episode_from_filename(model_path: str) -> int:
    """Extract episode number from model filename."""
    filename = os.path.basename(model_path)
    
    if "final" in filename:
        # Assume final model was trained to full episodes
        return 300000  # Based on original config
    
    # Extract episode number from filename like "m1_cnn_dqn_best_ep_250000.pt"
    try:
        if "_ep_" in filename:
            episode_str = filename.split("_ep_")[-1].split(".")[0]
            return int(episode_str)
    except (ValueError, IndexError):
        pass
    
    return 0


def resume_m1_cnn_training(target_episodes: int = 500000, resume_model_path: str = None):
    """Resume M1 CNN training from checkpoint."""
    
    print("🔄 M1 CNN TRAINING RESUME - CONTINUING TO 500K")
    print("=" * 60)
    
    # Find model to resume from
    if resume_model_path is None:
        resume_model_path = find_latest_model()
    
    if not resume_model_path or not os.path.exists(resume_model_path):
        print("❌ No M1 CNN model found to resume from!")
        print("   Available models should be in: models_m1_cnn/")
        return
    
    # Extract starting episode
    start_episode = extract_episode_from_filename(resume_model_path)
    remaining_episodes = target_episodes - start_episode
    
    print(f"📂 Resuming from: {os.path.basename(resume_model_path)}")
    print(f"🎯 Starting episode: {start_episode:,}")
    print(f"🎯 Target episodes: {target_episodes:,}")
    print(f"🎯 Remaining episodes: {remaining_episodes:,}")
    
    if remaining_episodes <= 0:
        print("✅ Model already trained to target episodes!")
        return
    
    # Load configuration and modify for resuming
    config = create_m1_cnn_config()
    config["num_episodes"] = remaining_episodes  # Only train remaining episodes
    config["random_seed"] = 42  # Keep same seed for consistency
    
    print(f"\n🔧 RESUME CONFIGURATION:")
    print(f"  Episodes to train: {remaining_episodes:,}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Architecture: M1-optimized CNN")
    
    # Set seeds for reproducibility
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["random_seed"])
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(config["random_seed"])
    
    # M1 GPU optimizations
    if torch.backends.mps.is_available():
        print("🚀 M1 GPU optimizations enabled")
        torch.backends.mps.allow_tf32 = True
    
    # Setup logging (append to existing logs)
    log_dir = "logs_m1_cnn"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "m1_cnn_training_log.csv")
    
    # Initialize monitoring
    monitor = TrainingMonitor(log_dir=log_dir, save_plots=True, eval_frequency=config["eval_frequency"])
    monitor._show_progress = False
    
    print(f"📊 Logging to: {log_file}")
    
    # Load the trained model
    print(f"\n🧠 Loading M1 CNN agent from checkpoint...")
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
        architecture="m1_optimized",
        seed=config["random_seed"]
    )
    
    # Load the checkpoint
    agent.load(resume_model_path, keep_player_id=True)
    
    total_params = sum(p.numel() for p in agent.online_net.parameters())
    print(f"✅ M1 CNN agent loaded:")
    print(f"   Device: {agent.device}")
    print(f"   Parameters: {total_params:,}")
    print(f"   Current epsilon: {agent.epsilon:.6f}")
    print(f"   Training steps: {agent.train_step_count:,}")
    print(f"   Buffer size: {len(agent.memory):,}")
    
    # Initialize opponents
    random_opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
    heuristic_opponent = HeuristicAgent(player_id=2, seed=config["random_seed"] + 2)
    
    # Initialize TRUE self-play agent
    self_play_opponent = SelfPlayAgent(agent, player_id=2)
    print(f"🤖 Self-play agent initialized (updates every 500 episodes)")
    
    # Training tracking
    episode_rewards = []
    best_win_rate = 0.0
    early_stopping_counter = 0
    
    # Curriculum phases (adjusted for resume)
    WARMUP_PHASE_END = config["warmup_phase_end"]
    RANDOM_PHASE_END = config["random_phase_end"] 
    HEURISTIC_PHASE_END = config["heuristic_phase_end"]
    SELF_PLAY_START = config["self_play_start"]
    
    def get_current_opponent(episode_absolute):
        """Get opponent based on absolute episode number with TRUE self-play."""
        if episode_absolute <= WARMUP_PHASE_END:
            return random_opponent, "Random-Warmup"
        elif episode_absolute <= RANDOM_PHASE_END:
            return random_opponent, "Random"
        elif episode_absolute <= SELF_PLAY_START:
            return heuristic_opponent, "Heuristic"
        else:
            # Mixed phase with TRUE self-play
            rand_val = np.random.random()
            if rand_val < config['heuristic_preservation_rate']:  # 30%
                return heuristic_opponent, "Heuristic"
            elif rand_val < config['heuristic_preservation_rate'] + config['random_diversity_rate']:  # 10%
                return random_opponent, "Random"
            else:  # 60% - TRUE SELF-PLAY
                return self_play_opponent, "Self-Play"
    
    print(f"\n🚀 Resuming M1 CNN training...")
    print(f"   Episodes {start_episode + 1:,} → {target_episodes:,}")
    print(f"   Expected time: ~{remaining_episodes // 150 // 60:.1f} hours")
    print(f"   Monitor: tail -f {log_file}")
    print()
    
    # Training loop (resume from where we left off)
    current_opponent_name = ""
    
    for episode in range(remaining_episodes):
        episode_num = episode + 1
        absolute_episode = start_episode + episode_num  # Absolute episode number
        
        # Get opponent based on absolute episode
        current_opponent, opponent_name = get_current_opponent(absolute_episode)
        
        # Update self-play agent with current episode info (if it's self-play)
        if opponent_name == "Self-Play":
            # Safely update self-play agent
            try:
                current_opponent.update_self_play_agent(absolute_episode)
            except Exception as e:
                print(f"⚠️ Self-play update warning: {e}")
                # Fallback to heuristic if self-play fails
                current_opponent = heuristic_opponent
                opponent_name = "Heuristic-Fallback"
        
        # Announce phase changes (but reduce frequency for self-play switching)
        if opponent_name != current_opponent_name:
            if current_opponent_name == "" or opponent_name not in ["Self-Play"] or current_opponent_name not in ["Self-Play"]:
                print(f"\n🔄 M1 CNN PHASE at episode {absolute_episode:,}: {current_opponent_name} → {opponent_name}")
            current_opponent_name = opponent_name
            monitor.set_current_opponent(opponent_name)
        
        # Reset episode
        agent.reset_episode()
        
        # Training mode selection (use absolute episode for consistency)
        reward_config_to_use = None if absolute_episode < config["optimism_bootstrap_end"] else config["reward_config"]
        
        # Play training game
        winner, _, spatial_stats = play_m1_training_game(
            agent, current_opponent, reward_config_to_use, config
        )
        
        # Calculate episode reward
        if winner == agent.player_id:
            episode_reward = 25.0
        elif winner is not None:
            episode_reward = -20.0 if absolute_episode >= config["optimism_bootstrap_end"] else 0.0
        else:
            episode_reward = 12.0
        
        episode_rewards.append(episode_reward)
        
        # Log episode
        spatial_score_for_monitor = spatial_stats.get('spatial_score', 0.0)
        monitor.log_episode(absolute_episode, episode_reward, agent, strategic_score=spatial_score_for_monitor)
        
        # Progress updates
        if episode_num <= 10 or episode_num % 100 == 0:
            avg_reward_recent = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
            print(f"Episode {absolute_episode:,} | {opponent_name} | "
                  f"Avg Reward: {avg_reward_recent:.1f} | ε: {agent.epsilon:.6f} | "
                  f"Buffer: {len(agent.memory)} | Steps: {agent.train_step_count:,} | "
                  f"Spatial: {spatial_score_for_monitor:.2f}")
        
        # Periodic evaluation
        if episode_num % config["eval_frequency"] == 0:
            # Quick evaluation
            old_epsilon = agent.epsilon
            agent.epsilon = 0.0
            
            try:
                wins_vs_random = 0
                wins_vs_heuristic = 0
                
                # Test against random
                for _ in range(50):
                    test_board = Connect4Board()
                    if np.random.random() < 0.5:
                        players = [agent, random_opponent]
                    else:
                        players = [random_opponent, agent]
                    
                    move_count = 0
                    while not test_board.is_terminal() and move_count < 42:
                        current_player = players[move_count % 2]
                        action = current_player.choose_action(test_board.get_state(), test_board.get_legal_moves())
                        test_board.make_move(action, current_player.player_id)
                        move_count += 1
                    
                    if test_board.check_winner() == agent.player_id:
                        wins_vs_random += 1
                
                # Test against heuristic
                for _ in range(50):
                    test_board = Connect4Board()
                    if np.random.random() < 0.5:
                        players = [agent, heuristic_opponent]
                    else:
                        players = [heuristic_opponent, agent]
                    
                    move_count = 0
                    while not test_board.is_terminal() and move_count < 42:
                        current_player = players[move_count % 2]
                        action = current_player.choose_action(test_board.get_state(), test_board.get_legal_moves())
                        test_board.make_move(action, current_player.player_id)
                        move_count += 1
                    
                    if test_board.check_winner() == agent.player_id:
                        wins_vs_heuristic += 1
                
                # Test against self-play if past self-play start
                wins_vs_self = 0
                if absolute_episode > SELF_PLAY_START:
                    for _ in range(30):  # Fewer games since self-play is computationally expensive
                        test_board = Connect4Board()
                        self_play_opponent.update_self_play_agent(absolute_episode)  # Ensure updated
                        
                        if np.random.random() < 0.5:
                            players = [agent, self_play_opponent]
                        else:
                            players = [self_play_opponent, agent]
                        
                        move_count = 0
                        while not test_board.is_terminal() and move_count < 42:
                            current_player = players[move_count % 2]
                            if hasattr(current_player, 'choose_action'):
                                if current_player == self_play_opponent:
                                    action = current_player.choose_action(test_board.get_state(), test_board.get_legal_moves(), absolute_episode)
                                else:
                                    action = current_player.choose_action(test_board.get_state(), test_board.get_legal_moves())
                            test_board.make_move(action, current_player.player_id)
                            move_count += 1
                        
                        if test_board.check_winner() == agent.player_id:
                            wins_vs_self += 1
                
                current_win_rate = wins_vs_random / 50
                vs_heuristic = wins_vs_heuristic / 50
                vs_self_play = wins_vs_self / 30 if absolute_episode > SELF_PLAY_START else 0.0
                
            finally:
                agent.epsilon = old_epsilon
            
            # Track best model
            if current_win_rate > best_win_rate:
                best_win_rate = current_win_rate
                os.makedirs("models_m1_cnn", exist_ok=True)
                best_path = f"models_m1_cnn/m1_cnn_dqn_best_ep_{absolute_episode}.pt"
                agent.save(best_path)
                print(f"💎 M1 CNN Best: {current_win_rate:.1%} - saved {os.path.basename(best_path)}")
            
            # Log results
            avg_reward = np.mean(episode_rewards[-config["eval_frequency"]:])
            spatial_score = spatial_stats.get('spatial_score', 0)
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    absolute_episode, avg_reward, agent.epsilon, agent.train_step_count,
                    len(agent.memory), current_win_rate, vs_heuristic, current_win_rate,
                    spatial_score, "M1-CNN-Resume", total_params
                ])
            
            if absolute_episode > SELF_PLAY_START:
                print(f"Episode {absolute_episode:,} | {opponent_name} | "
                      f"Win: {current_win_rate:.1%} | vs Heur: {vs_heuristic:.1%} | vs Self: {vs_self_play:.1%} | "
                      f"ε: {agent.epsilon:.6f} | Spatial: {spatial_score:.2f}")
            else:
                print(f"Episode {absolute_episode:,} | {opponent_name} | "
                      f"Win: {current_win_rate:.1%} | vs Heur: {vs_heuristic:.1%} | "
                      f"ε: {agent.epsilon:.6f} | Spatial: {spatial_score:.2f}")
            
            # Log for plotting
            monitor.log_episode(absolute_episode, avg_reward, agent, win_rate=current_win_rate, strategic_score=spatial_score)
            
            # Generate plots every 2000 episodes (same as original training)
            if absolute_episode % 2000 == 0:
                monitor.generate_training_report(agent, absolute_episode)
                print(f"📊 M1 CNN plots generated for episode {absolute_episode:,}")
        
        # Save checkpoints
        if episode_num % config["save_frequency"] == 0:
            os.makedirs("models_m1_cnn", exist_ok=True)
            checkpoint_path = f"models_m1_cnn/m1_cnn_dqn_ep_{absolute_episode}.pt"
            agent.save(checkpoint_path)
    
    # Final save
    print("\n" + "=" * 60)
    print("RESUMED M1 CNN TRAINING COMPLETED")
    print("=" * 60)
    
    final_model_path = "models_m1_cnn/m1_cnn_dqn_final_500k.pt"
    agent.save(final_model_path)
    
    print(f"\n✅ M1 CNN RESUME TRAINING COMPLETED!")
    print(f"Final model: {final_model_path}")
    print(f"Best performance: {best_win_rate:.1%}")
    print(f"Total episodes trained: {target_episodes:,}")
    print(f"Final epsilon: {agent.epsilon:.6f}")
    
    # Generate final report
    monitor.generate_training_report(agent, target_episodes)
    
    print("\n🎉 M1 CNN extended training to 500k episodes successful!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Resume M1 CNN training")
    parser.add_argument("--target", type=int, default=500000, help="Target total episodes (default: 500000)")
    parser.add_argument("--model", type=str, default=None, help="Specific model path to resume from")
    
    args = parser.parse_args()
    
    try:
        resume_m1_cnn_training(target_episodes=args.target, resume_model_path=args.model)
    except KeyboardInterrupt:
        print("\n⚠️ Resume training interrupted by user")
    except Exception as e:
        print(f"\n❌ Resume training failed: {e}")
        import traceback
        traceback.print_exc()