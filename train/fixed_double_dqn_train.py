"""Fixed Double DQN training using proven single-channel encoding and simpler approach."""

import os
import sys
import json
import csv
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Connect4Board
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.dqn_agent import DQNAgent  # Use the proven DQN agent as base


def create_fixed_config() -> Dict[str, Any]:
    """Use proven hyperparameters from working DQN agent."""
    return {
        "learning_rate": 0.001,
        "discount_factor": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.995,
        "buffer_size": 50000,
        "batch_size": 64,
        "min_buffer_size": 1000,
        "target_update_freq": 1000,
        "num_episodes": 15000,    # More episodes for thorough learning
        "eval_frequency": 1000,
        "save_frequency": 3000,
        "random_seed": 42,
        "opponent_type": "random"
    }


def simple_reward(winner: Optional[int], agent_id: int, move_was_winning: bool = False, move_was_blocking: bool = False) -> float:
    """Simplified reward system focusing on game outcomes with small strategic bonuses."""
    reward = 0.0
    
    # Main game outcome rewards
    if winner == agent_id:
        reward = 100.0  # Big reward for winning
    elif winner is not None:
        reward = -100.0  # Big penalty for losing
    else:
        reward = 0.0  # Neutral for draw
    
    # Small immediate strategic bonuses (much smaller than game outcome)
    if move_was_winning:
        reward += 1.0  # Small bonus for winning move
    if move_was_blocking:
        reward += 0.5  # Small bonus for blocking
    
    return reward


def play_simple_training_game(agent: DQNAgent, opponent) -> Tuple[Optional[int], int, float]:
    """Play a training game with simplified experience collection."""
    board = Connect4Board()
    
    # Randomly decide who goes first  
    if np.random.random() < 0.5:
        first_player = agent
        second_player = opponent
    else:
        first_player = opponent
        second_player = agent
    
    move_count = 0
    total_reward = 0.0
    
    while not board.is_terminal() and move_count < 42:
        current_player = first_player if move_count % 2 == 0 else second_player
        
        # Get current state and legal moves
        state = board.get_state()
        legal_moves = board.get_legal_moves()
        
        # Store state for learning (if it's the agent's turn)
        if current_player == agent:
            prev_state = state.copy()
        
        # Make move
        action = current_player.choose_action(state, legal_moves)
        
        # Check strategic value before making move (for agent only)
        move_was_winning = False
        move_was_blocking = False
        if current_player == agent:
            # Check if this move wins the game
            temp_board = board.copy()
            temp_board.make_move(action, agent.player_id)
            if temp_board.check_winner() == agent.player_id:
                move_was_winning = True
            
            # Check if this move blocks opponent win
            opponent_id = 3 - agent.player_id
            for test_col in legal_moves:
                test_board = board.copy()
                test_board.make_move(test_col, opponent_id)
                if test_board.check_winner() == opponent_id and test_col == action:
                    move_was_blocking = True
                    break
        
        # Execute the move
        board.make_move(action, current_player.player_id)
        move_count += 1
        
        # If game ended, give final rewards
        if board.is_terminal() and current_player == agent:
            winner = board.check_winner()
            reward = simple_reward(winner, agent.player_id, move_was_winning, move_was_blocking)
            
            # Store this experience
            next_state = board.get_state()
            agent.observe(prev_state, action, reward, next_state, True)
            total_reward += reward
            
        # If not terminal but agent played, give small strategic rewards
        elif current_player == agent:
            reward = simple_reward(None, agent.player_id, move_was_winning, move_was_blocking)
            next_state = board.get_state()
            agent.observe(prev_state, action, reward, next_state, False)
            total_reward += reward
    
    return board.check_winner(), move_count, total_reward


def evaluate_simple_agent(agent: DQNAgent, opponent, num_games: int = 100) -> float:
    """Simple agent evaluation."""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration
    
    wins = 0
    try:
        for _ in range(num_games):
            board = Connect4Board()
            
            # Randomly assign positions
            if np.random.random() < 0.5:
                players = [agent, opponent]
                agent_is_first = True
            else:
                players = [opponent, agent]  
                agent_is_first = False
                
            current_idx = 0
            
            while not board.is_terminal():
                current_player = players[current_idx]
                legal_moves = board.get_legal_moves()
                action = current_player.choose_action(board.get_state(), legal_moves)
                board.make_move(action, current_player.player_id)
                current_idx = 1 - current_idx
            
            winner = board.check_winner()
            if winner == agent.player_id:
                wins += 1
                
    finally:
        agent.epsilon = old_epsilon
        
    return wins / num_games


def train_fixed_double_dqn():
    """Train using proven DQN approach with simplified Double DQN concepts."""
    print("🔧 FIXED Double DQN Training (Using Proven DQN Base)")
    print("=" * 55)
    
    config = create_fixed_config()
    print("Configuration (proven settings):")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Use the proven DQN agent architecture
    agent = DQNAgent(
        player_id=1,
        learning_rate=config["learning_rate"],
        discount_factor=config["discount_factor"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        target_update_freq=config["target_update_freq"],
        seed=config["random_seed"]
    )
    
    opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
    
    print(f"Training: {agent} vs {opponent}")
    print(f"Device: {agent.device}")
    print(f"Using proven single-channel state encoding (42 inputs)")
    print()
    
    # Training loop
    episode_rewards = []
    
    for episode in range(1, config["num_episodes"] + 1):
        agent.reset_episode()
        
        # Play game with simplified reward system
        winner, game_length, total_reward = play_simple_training_game(agent, opponent)
        episode_rewards.append(total_reward)
        
        # Progress tracking
        if episode <= 10 or episode % 100 == 0:
            stats = agent.get_stats()
            if episode <= 10:
                print(f"Episode {episode:2d}: Winner={winner}, Reward={total_reward:6.1f}, "
                      f"Buffer={stats['buffer_size']:4d}, Steps={stats['training_steps']:5d}, ε={agent.epsilon:.4f}")
        
        # Periodic evaluation
        if episode % config["eval_frequency"] == 0:
            print(f"\n📊 Episode {episode} Evaluation:")
            
            # Test against different opponents
            vs_random = evaluate_simple_agent(agent, RandomAgent(player_id=2, seed=123), 100)
            vs_heuristic = evaluate_simple_agent(agent, HeuristicAgent(player_id=2, seed=124), 100)
            
            # Calculate recent performance
            recent_rewards = episode_rewards[-config["eval_frequency"]:]
            avg_reward = np.mean(recent_rewards)
            
            stats = agent.get_stats()
            print(f"  vs Random: {vs_random:.1%}")
            print(f"  vs Heuristic: {vs_heuristic:.1%}")
            print(f"  Avg Reward (recent): {avg_reward:.2f}")
            print(f"  Training Steps: {stats['training_steps']:,}")
            print(f"  Buffer: {stats['buffer_size']:,}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            
            # Performance check
            if vs_random > 0.7:
                print("  ✅ GOOD: Strong performance vs random!")
            elif vs_random > 0.6:
                print("  ⚠️  OK: Moderate performance vs random")
            else:
                print("  🚨 POOR: Weak performance vs random")
            
            if vs_heuristic > 0.2:
                print("  🏆 EXCELLENT: Learning strategic play!")
            elif vs_heuristic > 0.1:
                print("  📈 PROGRESS: Some strategic understanding")
            else:
                print("  💭 LEARNING: Still developing strategy")
            
        # Save checkpoints
        if episode % config["save_frequency"] == 0:
            os.makedirs("models_fixed", exist_ok=True)
            checkpoint_path = f"models_fixed/dqn_ep_{episode}.pt"
            agent.save(checkpoint_path)
            print(f"💾 Saved: {checkpoint_path}")
    
    # Final comprehensive evaluation
    print(f"\n🏁 FINAL EVALUATION")
    print("=" * 30)
    
    final_vs_random = evaluate_simple_agent(agent, RandomAgent(player_id=2), 500)
    final_vs_heuristic = evaluate_simple_agent(agent, HeuristicAgent(player_id=2), 500)
    
    print(f"Final Performance (500 games each):")
    print(f"  vs Random: {final_vs_random:.1%}")
    print(f"  vs Heuristic: {final_vs_heuristic:.1%}")
    
    # Overall assessment  
    if final_vs_random > 0.75 and final_vs_heuristic > 0.15:
        print("🎉 SUCCESS: Agent achieved good strategic performance!")
    elif final_vs_random > 0.65:
        print("✅ DECENT: Agent learned basic Connect 4 strategy")
    else:
        print("⚠️  SUBOPTIMAL: Agent performance below expectations")
    
    # Save final model
    final_path = "models_fixed/dqn_final.pt"
    agent.save(final_path)
    print(f"💾 Final model: {final_path}")
    
    final_stats = agent.get_stats()
    print(f"\nFinal Statistics:")
    print(f"  Training steps: {final_stats['training_steps']:,}")
    print(f"  Episodes trained: {config['num_episodes']:,}")
    print(f"  Final exploration: {agent.epsilon:.4f}")


if __name__ == "__main__":
    train_fixed_double_dqn()