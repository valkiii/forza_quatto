"""Training script for Deep Q-Network (DQN) agent."""

import os
import json
import csv
from typing import Dict, Any, Optional, Tuple
import numpy as np

from game.board import Connect4Board
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.dqn_agent import DQNAgent


def create_dqn_config() -> Dict[str, Any]:
    """Create default hyperparameters for DQN training.
    
    Returns:
        Dictionary with training configuration
    """
    return {
        "learning_rate": 0.001,
        "discount_factor": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995,
        "buffer_size": 50000,
        "batch_size": 64,
        "target_update_freq": 1000,
        "num_episodes": 10000,
        "eval_frequency": 1000,
        "save_frequency": 2000,
        "warmup_episodes": 1000,  # Episodes before training starts
        "random_seed": 42,
        "opponent_type": "random"  # "random" or "heuristic"
    }


def setup_dqn_logging(log_dir: str) -> str:
    """Setup logging directory and return path to log file.
    
    Args:
        log_dir: Directory to store logs
        
    Returns:
        Path to the CSV log file
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "dqn_log.csv")
    
    # Create CSV header if file doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "avg_reward", "epsilon", "training_steps", 
                "buffer_size", "win_rate"
            ])
            
    return log_file


def calculate_reward(winner: Optional[int], agent_id: int, game_length: int) -> float:
    """Calculate reward for the DQN agent.
    
    Args:
        winner: Winner ID (1, 2) or None for draw
        agent_id: DQN agent's player ID
        game_length: Number of moves in the game
        
    Returns:
        Reward value for the agent
    """
    if winner == agent_id:
        return 10.0  # Win reward
    elif winner is not None:
        return -10.0  # Loss penalty
    else:
        return 1.0  # Draw reward


def play_dqn_training_game(dqn_agent: DQNAgent, opponent: 'BaseAgent') -> Tuple[Optional[int], int]:
    """Play a single training game between DQN agent and opponent.
    
    Args:
        dqn_agent: DQN agent
        opponent: Opponent agent
        
    Returns:
        Tuple of (winner_id, game_length)
    """
    board = Connect4Board()
    
    # Randomly choose who goes first
    agents = [dqn_agent, opponent]
    if dqn_agent.rng.random() < 0.5:
        agents = [opponent, dqn_agent]
        
    move_count = 0
    episode_experiences = []  # Store experiences for the DQN agent
    
    while not board.is_terminal() and move_count < 42:
        current_agent = agents[move_count % 2]
        board_state = board.get_state()
        legal_moves = board.get_legal_moves()
        
        # Agent chooses action
        action = current_agent.choose_action(board_state, legal_moves)
        
        # Store state for DQN agent learning
        if current_agent == dqn_agent:
            episode_experiences.append({
                'state': board_state.copy(),
                'action': action
            })
            
        # Execute move
        board.make_move(action)
        move_count += 1
        
        # Provide learning experience to DQN agent
        if current_agent == dqn_agent and len(episode_experiences) > 1:
            # Get previous experience
            prev_exp = episode_experiences[-2]
            
            # Calculate intermediate reward (0 for ongoing)
            reward = 0.0
            done = board.is_terminal()
            
            if done:
                winner = board.check_winner()
                reward = calculate_reward(winner, dqn_agent.player_id, move_count)
                
            # Let agent learn from this transition
            next_state = board.get_state() if not done else None
            dqn_agent.observe(
                prev_exp['state'], prev_exp['action'], reward, next_state, done
            )
            
    # Handle final experience for DQN agent
    if episode_experiences and agents[-1] == dqn_agent:
        final_exp = episode_experiences[-1]
        winner = board.check_winner()
        final_reward = calculate_reward(winner, dqn_agent.player_id, move_count)
        
        dqn_agent.observe(final_exp['state'], final_exp['action'], final_reward, None, True)
        
    return board.check_winner(), move_count


def evaluate_agent(agent, opponent, num_games: int = 100) -> float:
    """Evaluate agent performance against an opponent.
    
    Args:
        agent: Agent to evaluate
        opponent: Opponent agent
        num_games: Number of evaluation games
        
    Returns:
        Win rate as a float between 0 and 1
    """
    wins = 0
    old_epsilon = getattr(agent, 'epsilon', None)
    
    # Disable exploration during evaluation
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0.0
    
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
                
            winner = board.check_winner()
            if winner == agent.player_id:
                wins += 1
    finally:
        # Restore exploration
        if old_epsilon is not None:
            agent.epsilon = old_epsilon
            
    return wins / num_games


def train_dqn_agent():
    """Main training loop for DQN agent."""
    print("Deep Q-Network (DQN) Agent Training")
    print("=" * 45)
    
    # Load configuration
    config = create_dqn_config()
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Setup logging
    log_file = setup_dqn_logging("logs")
    print(f"Logging to: {log_file}")
    
    # Initialize agents
    dqn_agent = DQNAgent(
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
    
    # Choose opponent
    if config["opponent_type"] == "heuristic":
        opponent = HeuristicAgent(player_id=2, seed=config["random_seed"] + 1)
    else:
        opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
        
    print(f"Training {dqn_agent} vs {opponent}")
    print(f"Device: {dqn_agent.device}")
    print()
    
    # Training loop
    episode_rewards = []
    
    for episode in range(config["num_episodes"]):
        # Reset episode state
        dqn_agent.reset_episode()
        
        # Play training game
        winner, game_length = play_dqn_training_game(dqn_agent, opponent)
        
        # Calculate episode reward
        episode_reward = calculate_reward(winner, dqn_agent.player_id, game_length)
        episode_rewards.append(episode_reward)
        
        # Periodic evaluation and logging
        if (episode + 1) % config["eval_frequency"] == 0:
            # Evaluate current policy
            win_rate = evaluate_agent(dqn_agent, opponent, num_games=100)
            
            # Log metrics
            avg_reward = np.mean(episode_rewards[-config["eval_frequency"]:])
            stats = dqn_agent.get_stats()
            
            print(f"Episode {episode + 1:6d} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Win Rate: {win_rate:5.1%} | "
                  f"Epsilon: {dqn_agent.epsilon:.3f} | "
                  f"Buffer: {stats['buffer_size']:6d} | "
                  f"Steps: {stats['training_steps']:8d}")
                  
            # Log to CSV
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode + 1, avg_reward, dqn_agent.epsilon, 
                    stats['training_steps'], stats['buffer_size'], win_rate
                ])
        
        # Save model checkpoint
        if (episode + 1) % config["save_frequency"] == 0:
            os.makedirs("models", exist_ok=True)
            model_path = f"models/dqn_ep_{episode + 1}.pt"
            dqn_agent.save(model_path)
            print(f"Saved checkpoint: {model_path}")
    
    # Final evaluation
    print("\n" + "=" * 55)
    print("FINAL EVALUATION")
    
    final_win_rate = evaluate_agent(dqn_agent, opponent, num_games=1000)
    print(f"Final win rate vs {opponent.name}: {final_win_rate:.1%}")
    
    # Save final model
    final_model_path = "models/dqn_final.pt"
    dqn_agent.save(final_model_path)
    print(f"Saved final model: {final_model_path}")
    
    # Print statistics
    stats = dqn_agent.get_stats()
    print(f"Training steps: {stats['training_steps']:,}")
    print(f"Final buffer size: {stats['buffer_size']:,}")
    print(f"Final epsilon: {stats['epsilon']:.4f}")
    print("DQN training completed!")


if __name__ == "__main__":
    train_dqn_agent()