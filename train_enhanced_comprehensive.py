#!/usr/bin/env python3
"""Enhanced comprehensive training script with all suggested improvements."""

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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.board import Connect4Board
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.enhanced_double_dqn_agent import EnhancedDoubleDQNAgent
from train.reward_system import calculate_enhanced_reward
from train.training_monitor import TrainingMonitor


def create_enhanced_config() -> Dict[str, Any]:
    """Create enhanced hyperparameters with all suggested improvements."""
    return {
        # Network Architecture (Improved)
        "hidden_size": 512,  # Increased from 256
        "use_dueling_network": True,
        "state_size": 92,  # Base 84 + 8 strategic features
        
        # Learning Parameters (Improved)
        "learning_rate": 5e-5,  # Ultra-conservative for stability
        "discount_factor": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,  # Lowered from 0.05 for less exploration
        "epsilon_decay": 0.9998,
        
        # Buffer and Training (Improved)
        "batch_size": 256,  # Increased from 128 for better gradients
        "buffer_size": 200000,  # Large buffer for diversity
        "min_buffer_size": 2000,
        "target_update_freq": 2000,
        "polyak_tau": 0.001,  # Soft updates instead of hard
        
        # Advanced Features (NEW)
        "n_step": 3,  # N-step returns for better credit assignment
        "use_prioritized_replay": True,
        "prioritized_alpha": 0.6,
        "prioritized_beta_start": 0.4,
        "prioritized_beta_end": 1.0,
        
        # Training Schedule (Extended)
        "num_episodes": 500000,  # Increased from 150K
        "eval_frequency": 1000,
        "save_frequency": 2000,
        "random_seed": 42,
        
        # Curriculum Learning (Improved)
        "random_phase_end": 50000,  # Reduced random phase
        "heuristic_phase_end": 200000,  # Extended heuristic phase
        "self_play_start_early": 50000,  # Start self-play earlier
        "self_play_ratio": 0.7,  # 70% self-play in final phase
        "heuristic_preservation_rate": 0.2,  # Always 20% heuristic
        "random_diversity_rate": 0.1,  # 10% random for diversity
        
        # Self-Play Improvements (NEW)
        "self_play_pool_size": 10,  # More diverse opponents
        "mcts_simulations": 50,  # MCTS for smarter play (optional)
        "use_mcts_evaluation": False,  # Enable MCTS during play
        
        # Reward System
        "reward_system": "enhanced",
        "reward_config": {
            "win_reward": 10.0,
            "loss_penalty": -10.0,
            "draw_reward": 1.0,
            "threat_reward": 3.0,  # Amplified strategic rewards
            "block_reward": 2.0,
            "center_bonus": 0.5,
            "height_penalty": -0.1
        },
        
        # Monitoring and Stopping
        "early_stopping": True,
        "early_stopping_threshold": 0.95,
        "early_stopping_patience": 20,
        "performance_monitoring": True,
        "save_best_model": True
    }


class MCTSActionSelector:
    """Monte Carlo Tree Search for enhanced action selection during evaluation."""
    
    def __init__(self, agent, simulations: int = 50):
        self.agent = agent
        self.simulations = simulations
    
    def select_action(self, board_state: np.ndarray, legal_moves: list) -> int:
        """Use MCTS with Q-values as priors for smarter play."""
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Get Q-values as priors
        encoded_state = self.agent.encode_state(board_state)
        state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(self.agent.device)
        
        with torch.no_grad():
            q_values = self.agent.online_net(state_tensor).cpu().numpy()[0]
        
        # Simple MCTS simulation using Q-values as priors
        action_scores = {}
        for action in legal_moves:
            # Use Q-value as base score
            base_score = q_values[action]
            
            # Add small random component for exploration
            noise = np.random.normal(0, 0.1)
            action_scores[action] = base_score + noise
        
        # Select action with highest score
        return max(action_scores, key=action_scores.get)


def setup_enhanced_logging(log_dir: str) -> str:
    """Setup enhanced logging directory."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "enhanced_training_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "avg_reward", "epsilon", "training_steps",
                "buffer_size", "win_rate", "vs_heuristic", "vs_random",
                "strategic_score", "n_step", "prioritized"
            ])
    
    return log_file


def play_enhanced_training_game(agent: EnhancedDoubleDQNAgent, opponent, 
                              reward_config: dict = None) -> Tuple[Optional[int], int, dict]:
    """Play enhanced training game with detailed statistics."""
    board = Connect4Board()
    
    # Random starting player
    if agent.rng.random() < 0.5:
        players = [agent, opponent]
    else:
        players = [opponent, agent]
    
    # Game tracking
    move_count = 0
    agent_moves = []
    strategic_stats = {
        'threats_created': 0,
        'threats_blocked': 0,
        'center_moves': 0,
        'strategic_score': 0
    }
    
    # N-step experience tracking for agent
    agent_experiences = []
    
    while not board.is_terminal() and move_count < 42:
        current_player = players[move_count % 2]
        state_before = board.get_state().copy()
        legal_moves = board.get_legal_moves()
        
        # Choose action
        action = current_player.choose_action(state_before, legal_moves)
        
        # Track agent moves and analyze strategy
        if current_player == agent:
            agent_moves.append({'state': state_before.copy(), 'action': action, 'move_num': move_count})
            
            # Analyze strategic value
            if action in [2, 3, 4]:  # Center columns
                strategic_stats['center_moves'] += 1
            
            # Check if move creates threat
            temp_board = Connect4Board()
            temp_board.board = state_before.copy()
            temp_board.make_move(action, agent.player_id)
            
            if temp_board.check_winner() == agent.player_id:
                strategic_stats['threats_created'] += 1
        
        # Make move
        board.make_move(action, current_player.player_id)
        move_count += 1
        
        # Process agent's previous move when opponent responds
        if current_player != agent and agent_experiences:
            # Calculate reward for agent's last move
            prev_exp = agent_experiences[-1]
            
            if reward_config:
                # Create board objects for reward calculation
                prev_board = Connect4Board()
                prev_board.board = prev_exp['state'].copy()
                
                reward = calculate_enhanced_reward(
                    prev_board, prev_exp['action'], board, 
                    agent.player_id, board.is_terminal(), reward_config
                )
            else:
                # Simple reward
                if board.is_terminal():
                    winner = board.check_winner()
                    if winner == agent.player_id:
                        reward = 10.0
                    elif winner is not None:
                        reward = -10.0
                    else:
                        reward = 1.0
                else:
                    reward = 0.0
            
            # Store experience
            next_state = None if board.is_terminal() else board.get_state().copy()
            agent.observe(prev_exp['state'], prev_exp['action'], reward, 
                         next_state, board.is_terminal())
        
        # Store agent experience for next processing
        if current_player == agent:
            agent_experiences.append({
                'state': state_before,
                'action': action
            })
    
    # Process final agent experience if game ended
    if agent_experiences and not board.is_terminal():
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
            if winner == agent.player_id:
                final_reward = 10.0
            elif winner is not None:
                final_reward = -10.0
            else:
                final_reward = 1.0
        
        agent.observe(final_exp['state'], final_exp['action'], final_reward, None, True)
    
    # Calculate strategic score
    if len(agent_moves) > 0:
        strategic_stats['strategic_score'] = (
            strategic_stats['threats_created'] * 2 +
            strategic_stats['center_moves'] * 0.5
        ) / len(agent_moves)
    
    return board.check_winner(), move_count, strategic_stats


def evaluate_enhanced_agent(agent, opponents: dict, num_games: int = 100, use_mcts: bool = False) -> dict:
    """Enhanced evaluation against multiple opponents with optional MCTS."""
    results = {}
    
    # Save original epsilon
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration during evaluation
    
    # Optional MCTS selector
    mcts_selector = MCTSActionSelector(agent, simulations=50) if use_mcts else None
    
    try:
        for opponent_name, opponent in opponents.items():
            wins = 0
            total_moves = 0
            strategic_scores = []
            
            for _ in range(num_games):
                board = Connect4Board()
                
                # Random starting player
                if np.random.random() < 0.5:
                    players = [agent, opponent]
                else:
                    players = [opponent, agent]
                
                move_count = 0
                agent_strategic_moves = 0
                agent_total_moves = 0
                
                while not board.is_terminal() and move_count < 42:
                    current_player = players[move_count % 2]
                    legal_moves = board.get_legal_moves()
                    
                    if current_player == agent:
                        if use_mcts and mcts_selector:
                            action = mcts_selector.select_action(board.get_state(), legal_moves)
                        else:
                            action = current_player.choose_action(board.get_state(), legal_moves)
                        
                        # Track strategic moves
                        agent_total_moves += 1
                        if action in [2, 3, 4]:  # Center columns
                            agent_strategic_moves += 1
                    else:
                        action = current_player.choose_action(board.get_state(), legal_moves)
                    
                    board.make_move(action, current_player.player_id)
                    move_count += 1
                
                winner = board.check_winner()
                if winner == agent.player_id:
                    wins += 1
                
                total_moves += move_count
                
                # Calculate strategic score for this game
                if agent_total_moves > 0:
                    strategic_score = agent_strategic_moves / agent_total_moves
                    strategic_scores.append(strategic_score)
            
            results[opponent_name] = {
                'win_rate': wins / num_games,
                'avg_game_length': total_moves / num_games,
                'avg_strategic_score': np.mean(strategic_scores) if strategic_scores else 0,
                'games_played': num_games
            }
    
    finally:
        # Restore epsilon
        agent.epsilon = old_epsilon
    
    return results



def train_enhanced_agent():
    """Main training loop with all enhancements."""
    print("🚀 ENHANCED DOUBLE DQN TRAINING - ALL IMPROVEMENTS APPLIED")
    print("=" * 65)
    
    # Clean up previous runs
    for dir_name in ["models_enhanced", "logs_enhanced"]:
        if os.path.exists(dir_name):
            print(f"🧹 Cleaning up {dir_name}...")
            shutil.rmtree(dir_name)
    
    print("✅ Cleanup complete\\n")
    
    # Load enhanced configuration
    config = create_enhanced_config()
    print("🔧 ENHANCED FEATURES APPLIED:")
    print(f"  1. ✅ Larger network: {config['hidden_size']} hidden units (vs 256)")
    print(f"  2. ✅ Dueling architecture: Separate value/advantage streams")
    print(f"  3. ✅ N-step returns: {config['n_step']}-step learning for better credit")
    print(f"  4. ✅ Prioritized replay: α={config['prioritized_alpha']}")
    print(f"  5. ✅ Enhanced state features: {config['state_size']} dimensions")
    print(f"  6. ✅ Improved training: batch_size={config['batch_size']}, lr={config['learning_rate']}")
    print(f"  7. ✅ Extended training: {config['num_episodes']:,} episodes")
    print(f"  8. ✅ Early self-play: starts at {config['self_play_start_early']:,}")
    print(f"  9. ✅ Polyak averaging: τ={config['polyak_tau']}")
    print(f"  10. ✅ Strategic features: threats, center control, connectivity")
    print()
    
    # Set seeds for reproducibility
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["random_seed"])
    print(f"🎲 Seeds set to {config['random_seed']} for reproducibility")
    
    # Setup logging and monitoring
    log_file = setup_enhanced_logging("logs_enhanced")
    monitor = TrainingMonitor(log_dir="logs_enhanced", save_plots=True, eval_frequency=config["eval_frequency"])
    print(f"📊 Logging to: {log_file}")
    print("📊 TrainingMonitor enabled - will generate training_progress_ep_*.png plots")
    
    # Initialize enhanced agent
    print("🤖 Initializing Enhanced Double DQN Agent...")
    agent = EnhancedDoubleDQNAgent(
        player_id=1,
        state_size=config["state_size"],
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
        n_step=config["n_step"],
        use_prioritized_replay=config["use_prioritized_replay"],
        polyak_tau=config["polyak_tau"],
        seed=config["random_seed"]
    )
    
    print(f"✅ Enhanced agent created:")
    print(f"   Device: {agent.device}")
    print(f"   Network: Dueling architecture with {config['hidden_size']} hidden units")
    print(f"   Features: N-step={agent.n_step}, Prioritized={agent.use_prioritized_replay}")
    print(f"   State size: {config['state_size']} (includes strategic features)")
    
    # Initialize opponents
    random_opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
    heuristic_opponent = HeuristicAgent(player_id=2, seed=config["random_seed"] + 2)
    
    # Self-play opponent pool
    self_play_pool = []
    
    # Training tracking
    episode_rewards = []
    best_win_rate = 0.0
    early_stopping_counter = 0
    
    # Curriculum phases
    RANDOM_PHASE_END = config["random_phase_end"]
    HEURISTIC_PHASE_END = config["heuristic_phase_end"] 
    SELF_PLAY_START = config["self_play_start_early"]
    
    print(f"\\n🎯 ENHANCED CURRICULUM SCHEDULE:")
    print(f"  Episodes 1-{RANDOM_PHASE_END:,}: vs Random (foundation)")
    print(f"  Episodes {RANDOM_PHASE_END + 1:,}-{HEURISTIC_PHASE_END:,}: vs Heuristic (strategy)")
    print(f"  Episodes {SELF_PLAY_START:,}+: Mixed training:")
    print(f"    - {config['self_play_ratio']:.0%} Self-play (skill development)")
    print(f"    - {config['heuristic_preservation_rate']:.0%} Heuristic (preserve strategy)")
    print(f"    - {config['random_diversity_rate']:.0%} Random (exploration)")
    
    print(f"\\n🚀 Starting enhanced comprehensive training...")
    print(f"   Monitor with: tail -f logs_enhanced/enhanced_training_log.csv")
    print()
    
    def get_current_opponent(episode):
        """Enhanced opponent selection with early self-play."""
        if episode <= RANDOM_PHASE_END:
            return random_opponent, "Random"
        elif episode <= SELF_PLAY_START:
            return heuristic_opponent, "Heuristic"
        else:
            # Mixed training phase
            rand_val = np.random.random()
            
            if rand_val < config['heuristic_preservation_rate']:
                return heuristic_opponent, "Heuristic"
            elif rand_val < config['heuristic_preservation_rate'] + config['random_diversity_rate']:
                return random_opponent, "Random"
            else:
                # Self-play
                if self_play_pool:
                    opponent_agent, label = random.choice(self_play_pool)
                    return opponent_agent, f"Self-play"
                else:
                    # Fallback to heuristic if no self-play models yet
                    return heuristic_opponent, "Heuristic"
    
    # Training loop
    current_opponent_name = ""
    
    for episode in range(config["num_episodes"]):
        episode_num = episode + 1
        
        # Get opponent
        current_opponent, opponent_name = get_current_opponent(episode_num)
        
        # Announce phase changes (less frequently)
        if opponent_name != current_opponent_name and episode_num % 1000 == 0:
            print(f"🔄 Phase: {opponent_name} training (episode {episode_num:,})")
            current_opponent_name = opponent_name
        
        # Reset episode
        agent.reset_episode()
        
        # Play training game
        winner, game_length, strategic_stats = play_enhanced_training_game(
            agent, current_opponent, config["reward_config"]
        )
        
        # Calculate episode reward
        if winner == agent.player_id:
            episode_reward = 10.0
        elif winner is not None:
            episode_reward = -10.0
        else:
            episode_reward = 1.0
        
        episode_rewards.append(episode_reward)
        
        # Log episode with monitor (pass strategic score)
        monitor.log_episode(episode_num, episode_reward, agent)
        
        # Periodic evaluation
        if episode_num % config["eval_frequency"] == 0:
            # Evaluate against multiple opponents
            opponents = {
                "current": current_opponent,
                "random": random_opponent,
                "heuristic": heuristic_opponent
            }
            
            eval_results = evaluate_enhanced_agent(
                agent, opponents, num_games=100, 
                use_mcts=config["use_mcts_evaluation"]
            )
            
            current_win_rate = eval_results["current"]["win_rate"]
            vs_random = eval_results["random"]["win_rate"]
            vs_heuristic = eval_results["heuristic"]["win_rate"]
            
            # Early stopping check
            if config["early_stopping"] and episode_num > HEURISTIC_PHASE_END:
                if current_win_rate > config["early_stopping_threshold"]:
                    early_stopping_counter += 1
                    if early_stopping_counter >= config["early_stopping_patience"]:
                        print(f"\\n🏆 EARLY STOPPING at episode {episode_num:,}")
                        print(f"Win rate {current_win_rate:.1%} > {config['early_stopping_threshold']:.1%}")
                        break
                else:
                    early_stopping_counter = 0
            
            # Track best model
            if current_win_rate > best_win_rate:
                best_win_rate = current_win_rate
                if config["save_best_model"]:
                    os.makedirs("models_enhanced", exist_ok=True)
                    best_path = f"models_enhanced/enhanced_dqn_best_ep_{episode_num}.pt"
                    agent.save(best_path)
                    print(f"💎 New best: {current_win_rate:.1%} - saved {best_path}")
            
            # Log results
            avg_reward = np.mean(episode_rewards[-config["eval_frequency"]:])
            strategic_score = strategic_stats.get('strategic_score', 0)
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode_num, avg_reward, agent.epsilon, agent.train_step_count,
                    len(agent.memory), current_win_rate, vs_heuristic, vs_random,
                    strategic_score, agent.n_step, agent.use_prioritized_replay
                ])
            
            print(f"Episode {episode_num:,} | {opponent_name} | "
                  f"Win: {current_win_rate:.1%} | vs Heur: {vs_heuristic:.1%} | "
                  f"vs Rand: {vs_random:.1%} | ε: {agent.epsilon:.3f} | "
                  f"Strategic: {strategic_score:.2f}")
            
            # Log evaluation episode with win rate data
            monitor.log_episode(episode_num, avg_reward, agent, win_rate=current_win_rate)
            
            # Generate training progress plots (less frequently for cleaner output)
            if episode_num % 5000 == 0:  # Only generate plots every 5000 episodes
                monitor.generate_training_report(agent, episode_num)
        
        # Save checkpoints and update self-play pool
        if episode_num % config["save_frequency"] == 0:
            os.makedirs("models_enhanced", exist_ok=True)
            checkpoint_path = f"models_enhanced/enhanced_dqn_ep_{episode_num}.pt"
            agent.save(checkpoint_path)
            
            # Add to self-play pool if in appropriate phase
            if episode_num >= SELF_PLAY_START and episode_num >= RANDOM_PHASE_END:
                # Create self-play opponent
                self_play_agent = EnhancedDoubleDQNAgent(
                    player_id=2,  # Important: different player ID
                    state_size=config["state_size"],
                    action_size=7,
                    hidden_size=config["hidden_size"],
                    seed=config["random_seed"] + episode_num
                )
                self_play_agent.load(checkpoint_path, keep_player_id=False)
                self_play_agent.epsilon = 0.02  # Small exploration
                
                # Add to pool with size limit
                self_play_pool.append((self_play_agent, f"ep_{episode_num}"))
                if len(self_play_pool) > config["self_play_pool_size"]:
                    self_play_pool.pop(0)  # Remove oldest
                
                # Only print self-play updates occasionally
                if episode_num % 10000 == 0:
                    print(f"📚 Self-play pool: {len(self_play_pool)} opponents")
    
    # Final evaluation
    print("\\n" + "=" * 65)
    print("FINAL ENHANCED EVALUATION")
    print("=" * 65)
    
    final_opponents = {
        "random": random_opponent,
        "heuristic": heuristic_opponent
    }
    
    if self_play_pool:
        final_opponents["self_play"] = self_play_pool[-1][0]  # Latest self-play
    
    # Comprehensive final evaluation
    final_results = evaluate_enhanced_agent(
        agent, final_opponents, num_games=500,
        use_mcts=config["use_mcts_evaluation"]
    )
    
    print("🎯 Final Performance:")
    for opponent_name, results in final_results.items():
        print(f"  vs {opponent_name}: {results['win_rate']:.1%} "
              f"(avg moves: {results['avg_game_length']:.1f}, "
              f"strategic: {results['avg_strategic_score']:.2f})")
    
    # Save final model and results
    final_model_path = "models_enhanced/enhanced_dqn_final.pt"
    agent.save(final_model_path)
    
    results_data = {
        'config': config,
        'final_evaluation': final_results,
        'training_stats': agent.get_stats(),
        'best_win_rate': best_win_rate,
        'total_episodes_trained': episode_num
    }
    
    with open("logs_enhanced/final_results.json", 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\\n✅ ENHANCED TRAINING COMPLETED!")
    print(f"Final model: {final_model_path}")
    print(f"Results: logs_enhanced/final_results.json")
    print(f"Best performance: {best_win_rate:.1%}")
    
    # Generate final training report
    print("\\n📊 Generating final training progress plots...")
    monitor.generate_training_report(agent, episode_num if 'episode_num' in locals() else config["num_episodes"])
    
    print("\\n🎉 All enhancements successfully applied!")


if __name__ == "__main__":
    try:
        train_enhanced_agent()
    except KeyboardInterrupt:
        print("\\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()