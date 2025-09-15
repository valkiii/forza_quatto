"""Improved Double DQN training with fixed hyperparameters and debugging."""

import os
import sys
import json
import csv
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Connect4Board
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.double_dqn_agent import DoubleDQNAgent
from reward_system import calculate_enhanced_reward
from training_monitor import TrainingMonitor


def create_improved_double_dqn_config() -> Dict[str, Any]:
    """Create improved hyperparameters for Double DQN training."""
    return {
        "learning_rate": 0.0005,        # Slightly lower LR for stability
        "discount_factor": 0.95,        # Lower gamma for faster learning
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,             # Higher minimum exploration
        "epsilon_decay": 0.9990,        # Faster decay to start learning sooner
        "buffer_size": 10000,           # Smaller buffer for faster cycling
        "batch_size": 64,               # Smaller batch size
        "min_buffer_size": 200,         # Much lower threshold to start learning
        "target_update_freq": 500,      # More frequent target updates
        "num_episodes": 10000,
        "eval_frequency": 500,          # More frequent evaluation
        "save_frequency": 2000,
        "random_seed": 42,
        "opponent_type": "random",      # Start with random for basic learning
        "debug_mode": True              # Enable debug output
    }


def play_improved_training_game(agent: DoubleDQNAgent, opponent, monitor: TrainingMonitor = None, debug: bool = False) -> Tuple[Optional[int], int]:
    """Play training game with improved experience collection."""
    board = Connect4Board()
    
    # Randomly decide who goes first
    agents = [agent, opponent]
    if agent.rng.random() < 0.5:
        agents = [opponent, agent]
    
    move_count = 0
    agent_experiences = []  # Store all agent experiences
    
    if debug and move_count == 0:
        print(f"  Game start: Agent is player {agent.player_id}, goes {'first' if agents[0] == agent else 'second'}")
    
    while not board.is_terminal() and move_count < 42:
        current_agent = agents[move_count % 2]
        prev_board = board.copy()
        board_state = board.get_state()
        legal_moves = board.get_legal_moves()
        
        action = current_agent.choose_action(board_state, legal_moves)
        
        # Store experience for agent (regardless of whose turn it is)
        if current_agent == agent:
            agent_experiences.append({
                'prev_board': prev_board,
                'state': board_state.copy(),
                'action': action,
                'move_number': move_count
            })
        
        board.make_move(action, current_agent.player_id)
        move_count += 1
        
        # Monitor strategic play
        if monitor and current_agent == agent:
            monitor.analyze_strategic_play(prev_board, action, board, agent.player_id)
        
        if debug and current_agent == agent:
            print(f"    Agent move {len(agent_experiences)}: column {action}, legal moves were {legal_moves}")
    
    # Process all agent experiences with rewards
    game_winner = board.check_winner()
    experiences_processed = 0
    
    for i, exp in enumerate(agent_experiences):
        # Determine if this was the final move
        is_final = (i == len(agent_experiences) - 1)
        
        # Calculate enhanced reward
        reward = calculate_enhanced_reward(
            exp['prev_board'],
            exp['action'],
            board if is_final else board.copy(),  # Use final board state
            agent.player_id,
            is_final  # Only final move gets game outcome reward
        )
        
        # Determine next state
        if is_final:
            next_state = None
            done = True
        else:
            # For non-final moves, next state is current board state
            next_state = board.get_state()
            done = False
        
        # Store experience
        agent.observe(exp['state'], exp['action'], reward, next_state, done)
        experiences_processed += 1
        
        if debug:
            print(f"    Experience {i+1}: reward={reward:.3f}, done={done}")
    
    if debug:
        print(f"  Game end: Winner={game_winner}, Agent experiences: {experiences_processed}")
    
    return game_winner, move_count


def train_improved_double_dqn_agent():
    """Improved training loop with better hyperparameters and debugging."""
    print("🚀 IMPROVED Double DQN Agent Training")
    print("=" * 40)
    
    # Load improved configuration
    config = create_improved_double_dqn_config()
    print("🔧 Improved Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Setup monitoring
    monitor = TrainingMonitor(log_dir="logs_improved", save_plots=True)
    print("📊 Enhanced monitoring enabled")
    
    # Initialize agent
    agent = DoubleDQNAgent(
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
        seed=config["random_seed"]
    )
    
    # Start with random opponent
    opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
    
    print(f"🤖 Training: {agent} vs {opponent}")
    print(f"💻 Device: {agent.device}")
    print(f"🧠 Min buffer size: {config['min_buffer_size']} (reduced for faster learning)")
    print()
    
    # Training loop with debug output
    episode_rewards = []
    debug_episodes = [1, 2, 3, 5, 10]  # Episodes to debug
    
    for episode in range(1, config["num_episodes"] + 1):
        agent.reset_episode()
        
        # Debug mode for first few episodes
        debug_mode = config.get("debug_mode", False) and episode in debug_episodes
        
        if debug_mode:
            print(f"🔍 DEBUG Episode {episode}:")
            stats_before = agent.get_stats()
            print(f"  Before: Buffer={stats_before['buffer_size']}, Steps={stats_before['training_steps']}, Epsilon={stats_before['epsilon']:.4f}")
        
        # Play training game
        winner, game_length = play_improved_training_game(agent, opponent, monitor, debug_mode)
        
        # Calculate episode reward
        if winner == agent.player_id:
            episode_reward = 10.0
        elif winner is not None:
            episode_reward = -10.0
        else:
            episode_reward = 1.0
        episode_rewards.append(episode_reward)
        
        if debug_mode:
            stats_after = agent.get_stats()
            print(f"  After: Buffer={stats_after['buffer_size']}, Steps={stats_after['training_steps']}, Epsilon={stats_after['epsilon']:.4f}")
            print(f"  Winner: {winner}, Reward: {episode_reward}")
            print()
        
        # Log every episode to monitor
        monitor.log_episode(episode, episode_reward, agent)
        
        # Periodic evaluation and reporting
        if episode % config["eval_frequency"] == 0:
            print(f"\n⏱️  Episode {episode} Checkpoint")
            
            # Evaluate against opponents
            random_win_rate = evaluate_agent_quickly(agent, RandomAgent(player_id=2, seed=123), 50)
            heuristic_win_rate = evaluate_agent_quickly(agent, HeuristicAgent(player_id=2, seed=124), 50)
            
            # Log with detailed evaluation
            monitor.log_episode(episode, episode_reward, agent, random_win_rate)
            monitor.log_strategic_episode(episode)
            
            # Progress report
            stats = agent.get_stats()
            recent_avg_reward = np.mean(episode_rewards[-config["eval_frequency"]:])
            
            print(f"📊 Performance Summary:")
            print(f"  vs Random: {random_win_rate:.1%}")
            print(f"  vs Heuristic: {heuristic_win_rate:.1%}")  
            print(f"  Recent Avg Reward: {recent_avg_reward:.2f}")
            print(f"  Training Steps: {stats['training_steps']:,}")
            print(f"  Buffer Size: {stats['buffer_size']:,}")
            print(f"  Exploration: {stats['epsilon']:.4f}")
            
            # Check if learning has started
            if stats['training_steps'] == 0:
                print("⚠️  WARNING: No training steps yet! Check min_buffer_size and experience collection.")
            else:
                print("✅ Neural network training active")
            
            monitor.generate_training_report(agent, episode)
            monitor.reset_strategic_stats()
        
        # Save checkpoints
        if episode % config["save_frequency"] == 0:
            os.makedirs("models_improved", exist_ok=True)
            model_path = f"models_improved/double_dqn_ep_{episode}.pt"
            agent.save(model_path)
            print(f"💾 Saved checkpoint: {model_path}")
    
    # Final evaluation
    print(f"\n🏁 FINAL EVALUATION")
    print("=" * 30)
    
    final_random_rate = evaluate_agent_quickly(agent, RandomAgent(player_id=2), 200)
    final_heuristic_rate = evaluate_agent_quickly(agent, HeuristicAgent(player_id=2), 200)
    
    print(f"Final Performance:")
    print(f"  vs Random: {final_random_rate:.1%}")
    print(f"  vs Heuristic: {final_heuristic_rate:.1%}")
    
    # Save final model
    final_model_path = "models_improved/double_dqn_final.pt"
    agent.save(final_model_path)
    print(f"💾 Final model saved: {final_model_path}")
    
    final_stats = agent.get_stats()
    print(f"\n📈 Final Statistics:")
    print(f"  Total training steps: {final_stats['training_steps']:,}")
    print(f"  Final buffer size: {final_stats['buffer_size']:,}")
    print(f"  Final exploration: {final_stats['epsilon']:.4f}")
    
    if final_random_rate > 0.6:
        print("🎉 SUCCESS: Agent learned to beat random opponents!")
    else:
        print("⚠️  CONCERN: Agent performance is still low")
    
    if final_heuristic_rate > 0.1:
        print("🏆 EXCELLENT: Agent shows strategic understanding!")
    else:
        print("💭 INFO: Agent needs more strategic learning")


def evaluate_agent_quickly(agent: DoubleDQNAgent, opponent, num_games: int) -> float:
    """Quick agent evaluation without exploration."""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    wins = 0
    try:
        for _ in range(num_games):
            board = Connect4Board()
            players = [agent, opponent] if agent.player_id == 1 else [opponent, agent]
            current_idx = 0
            
            while not board.is_terminal():
                current_player = players[current_idx]
                legal_moves = board.get_legal_moves()
                action = current_player.choose_action(board.get_state(), legal_moves)
                board.make_move(action, current_player.player_id)
                current_idx = 1 - current_idx
            
            if board.check_winner() == agent.player_id:
                wins += 1
    finally:
        agent.epsilon = old_epsilon
    
    return wins / num_games


if __name__ == "__main__":
    train_improved_double_dqn_agent()