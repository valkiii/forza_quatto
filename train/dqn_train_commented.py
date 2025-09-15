"""Training script for Deep Q-Network (DQN) agent with detailed comments.

This file contains detailed line-by-line comments explaining every aspect
of the DQN training process for educational purposes.
"""

# Import standard Python libraries
import os                               # For file and directory operations
import json                            # For pretty-printing configuration
import csv                             # For logging training metrics to CSV
from typing import Dict, Any, Optional, Tuple  # Type hints for better code documentation
import numpy as np                     # For numerical operations

# Import our Connect 4 game and agents
from game.board import Connect4Board           # The Connect 4 game environment
from agents.random_agent import RandomAgent   # Random agent for training opponent
from agents.heuristic_agent import HeuristicAgent  # Strategic agent for training opponent
from agents.dqn_agent import DQNAgent         # The DQN agent we're training


def create_dqn_config() -> Dict[str, Any]:
    """Create default hyperparameters for DQN training."""
    
    # Return dictionary with all training hyperparameters
    return {
        # Neural network hyperparameters
        "learning_rate": 0.001,          # How fast the neural network learns (Adam optimizer)
        "discount_factor": 0.99,         # How much we value future rewards (gamma in Bellman equation)
        
        # Exploration hyperparameters (epsilon-greedy)
        "epsilon_start": 1.0,            # Start with 100% random exploration
        "epsilon_end": 0.05,             # End with 5% random exploration
        "epsilon_decay": 0.995,          # Multiply epsilon by this after each episode
        
        # Experience replay hyperparameters
        "buffer_size": 50000,            # Maximum number of experiences to store
        "batch_size": 64,                # Number of experiences per training step
        "target_update_freq": 1000,      # Update target network every N training steps
        
        # Training schedule hyperparameters
        "num_episodes": 10000,           # Total number of games to play during training
        "eval_frequency": 1000,          # Evaluate agent every N episodes
        "save_frequency": 2000,          # Save model checkpoint every N episodes
        "warmup_episodes": 1000,         # Episodes before training starts (unused in current code)
        
        # Reproducibility and opponent
        "random_seed": 42,               # Random seed for reproducible results
        "opponent_type": "random"        # Type of opponent: "random" or "heuristic"
    }


def setup_dqn_logging(log_dir: str) -> str:
    """Setup logging directory and return path to log file."""
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)     # Create directory, don't error if exists
    
    # Define path to CSV log file
    log_file = os.path.join(log_dir, "dqn_log.csv")  # Full path to log file
    
    # Create CSV file with header row if it doesn't exist
    if not os.path.exists(log_file):        # Only create header if file is new
        with open(log_file, 'w', newline='') as f:  # Open file for writing
            writer = csv.writer(f)          # Create CSV writer object
            # Write header row with column names
            writer.writerow([
                "episode",                  # Episode number
                "avg_reward",              # Average reward over recent episodes
                "epsilon",                 # Current exploration rate
                "training_steps",          # Total number of training steps performed
                "buffer_size",             # Current size of experience replay buffer
                "win_rate"                 # Win rate against opponent in evaluation
            ])
            
    return log_file                         # Return path to log file


def calculate_reward(winner: Optional[int], agent_id: int, game_length: int) -> float:
    """Calculate reward for the DQN agent based on game outcome."""
    
    # Determine reward based on who won the game
    if winner == agent_id:                  # Agent won the game
        return 10.0                         # Large positive reward for winning
    elif winner is not None:                # Opponent won the game
        return -10.0                        # Large negative reward for losing
    else:                                   # Game was a draw
        return 1.0                          # Small positive reward for draw


def play_dqn_training_game(dqn_agent: DQNAgent, opponent: 'BaseAgent') -> Tuple[Optional[int], int]:
    """Play a single training game between DQN agent and opponent."""
    
    # Create new Connect 4 game board
    board = Connect4Board()                 # Initialize empty 6x7 board
    
    # Randomly decide who goes first (important for balanced training)
    agents = [dqn_agent, opponent]          # Default order: DQN first
    if dqn_agent.rng.random() < 0.5:       # 50% chance to swap order
        agents = [opponent, dqn_agent]      # Opponent goes first
        
    # Initialize game tracking variables
    move_count = 0                          # Count moves made in game
    episode_experiences = []                # Store state-action pairs for DQN learning
    
    # Main game loop - continue until game ends or maximum moves reached
    while not board.is_terminal() and move_count < 42:  # 42 = maximum possible moves
        # Determine whose turn it is
        current_agent = agents[move_count % 2]          # Alternate between agents
        
        # Get current game state and legal moves
        board_state = board.get_state()                 # Get current board as numpy array
        legal_moves = board.get_legal_moves()           # Get list of legal columns
        
        # Current agent chooses an action
        action = current_agent.choose_action(board_state, legal_moves)  # Agent selects column
        
        # Store state-action pair if current player is DQN agent
        if current_agent == dqn_agent:                  # Only store DQN agent's moves
            episode_experiences.append({               # Store for later learning
                'state': board_state.copy(),           # Copy current state (before move)
                'action': action                       # Action chosen by DQN agent
            })
            
        # Execute the chosen move on the board
        board.make_move(action)                        # Drop piece in chosen column
        move_count += 1                                # Increment move counter
        
        # Provide learning experience to DQN agent if it just moved
        if current_agent == dqn_agent and len(episode_experiences) > 1:
            # Get the previous state-action pair for learning
            prev_exp = episode_experiences[-2]         # Get previous experience
            
            # Calculate reward for the previous move
            reward = 0.0                               # Default: no reward for ongoing game
            done = board.is_terminal()                 # Check if game ended
            
            if done:                                   # Game ended after this move
                winner = board.check_winner()          # Find out who won
                # Calculate final reward based on game outcome
                reward = calculate_reward(winner, dqn_agent.player_id, move_count)
                
            # Let DQN agent learn from this state transition
            next_state = board.get_state() if not done else None  # Next state (None if game ended)
            dqn_agent.observe(
                prev_exp['state'],                     # Previous state
                prev_exp['action'],                    # Action taken from previous state
                reward,                                # Reward received
                next_state,                            # Resulting state
                done                                   # Whether game ended
            )
            
    # Handle the final experience for DQN agent (last move of the game)
    if episode_experiences and agents[-1] == dqn_agent:  # If DQN agent made last move
        final_exp = episode_experiences[-1]            # Get final state-action pair
        winner = board.check_winner()                  # Determine game winner
        # Calculate final reward
        final_reward = calculate_reward(winner, dqn_agent.player_id, move_count)
        
        # Give final learning experience to DQN agent
        dqn_agent.observe(final_exp['state'], final_exp['action'], final_reward, None, True)
        
    # Return game result and length
    return board.check_winner(), move_count            # Return (winner_id, num_moves)


def evaluate_agent(agent, opponent, num_games: int = 100) -> float:
    """Evaluate agent performance against an opponent."""
    
    # Initialize win counter
    wins = 0                                           # Count agent wins
    
    # Store agent's current exploration rate
    old_epsilon = getattr(agent, 'epsilon', None)      # Save epsilon if agent has it
    
    # Disable exploration during evaluation (pure exploitation)
    if hasattr(agent, 'epsilon'):                      # If agent uses epsilon-greedy
        agent.epsilon = 0.0                            # Set to 0 for best play
    
    try:
        # Play evaluation games
        for _ in range(num_games):                     # Play specified number of games
            # Create new game for each evaluation
            board = Connect4Board()                    # Fresh board
            
            # Set up players in correct order
            players = [agent, opponent] if agent.player_id == 1 else [opponent, agent]
            current_idx = 0                            # Index of current player
            
            # Play one evaluation game
            while not board.is_terminal():             # Until game ends
                current_player = players[current_idx]  # Get current player
                legal_moves = board.get_legal_moves()  # Get legal moves
                # Player chooses action
                action = current_player.choose_action(board.get_state(), legal_moves)
                # Execute move
                board.make_move(action, current_player.player_id)
                current_idx = 1 - current_idx          # Switch to other player
                
            # Check if agent won this game
            winner = board.check_winner()              # Get winner
            if winner == agent.player_id:              # If agent won
                wins += 1                              # Increment win counter
    finally:
        # Restore agent's original exploration rate
        if old_epsilon is not None:                    # If agent had epsilon
            agent.epsilon = old_epsilon                # Restore original value
            
    # Return win rate as fraction
    return wins / num_games                            # Return win percentage


def train_dqn_agent():
    """Main training loop for DQN agent."""
    
    # Print training header
    print("Deep Q-Network (DQN) Agent Training")      # Print title
    print("=" * 45)                                   # Print separator line
    
    # Load configuration and print it
    config = create_dqn_config()                       # Get training configuration
    print(f"Configuration: {json.dumps(config, indent=2)}")  # Pretty-print config
    
    # Setup logging system
    log_file = setup_dqn_logging("logs")               # Create log directory and file
    print(f"Logging to: {log_file}")                   # Print log file path
    
    # Initialize DQN agent with configuration parameters
    dqn_agent = DQNAgent(
        player_id=1,                                   # Agent plays as player 1
        learning_rate=config["learning_rate"],         # Neural network learning rate
        discount_factor=config["discount_factor"],     # Future reward discount
        epsilon_start=config["epsilon_start"],         # Initial exploration rate
        epsilon_end=config["epsilon_end"],             # Final exploration rate
        epsilon_decay=config["epsilon_decay"],         # Exploration decay rate
        buffer_size=config["buffer_size"],             # Experience replay buffer size
        batch_size=config["batch_size"],               # Training batch size
        target_update_freq=config["target_update_freq"], # Target network update frequency
        seed=config["random_seed"]                     # Random seed for reproducibility
    )
    
    # Choose opponent based on configuration
    if config["opponent_type"] == "heuristic":         # If using strategic opponent
        opponent = HeuristicAgent(player_id=2, seed=config["random_seed"] + 1)
    else:                                              # Otherwise use random opponent
        opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
        
    # Print training setup information
    print(f"Training {dqn_agent} vs {opponent}")       # Print agent matchup
    print(f"Device: {dqn_agent.device}")               # Print computation device (CPU/GPU)
    print()                                            # Print blank line
    
    # Initialize training tracking
    episode_rewards = []                               # Store rewards from each episode
    
    # Main training loop
    for episode in range(config["num_episodes"]):      # For each training episode
        
        # Reset agent's episode state (clear last state/action, decay epsilon)
        dqn_agent.reset_episode()
        
        # Play one training game between DQN agent and opponent
        winner, game_length = play_dqn_training_game(dqn_agent, opponent)
        
        # Calculate and store episode reward
        episode_reward = calculate_reward(winner, dqn_agent.player_id, game_length)
        episode_rewards.append(episode_reward)         # Add to reward history
        
        # Periodic evaluation and logging
        if (episode + 1) % config["eval_frequency"] == 0:  # Every 1000 episodes
            
            # Evaluate current agent performance
            win_rate = evaluate_agent(dqn_agent, opponent, num_games=100)  # Play 100 evaluation games
            
            # Calculate metrics for logging
            avg_reward = np.mean(episode_rewards[-config["eval_frequency"]:])  # Average recent rewards
            stats = dqn_agent.get_stats()              # Get agent statistics
            
            # Print training progress
            print(f"Episode {episode + 1:6d} | "       # Episode number (right-aligned)
                  f"Avg Reward: {avg_reward:6.2f} | "  # Average reward (2 decimal places)
                  f"Win Rate: {win_rate:5.1%} | "      # Win rate as percentage
                  f"Epsilon: {dqn_agent.epsilon:.3f} | "  # Current exploration rate
                  f"Buffer: {stats['buffer_size']:6d} | "  # Replay buffer size
                  f"Steps: {stats['training_steps']:8d}")  # Training steps performed
                  
            # Log metrics to CSV file
            with open(log_file, 'a', newline='') as f:  # Append to log file
                writer = csv.writer(f)                  # Create CSV writer
                writer.writerow([                       # Write data row
                    episode + 1,                        # Episode number
                    avg_reward,                         # Average reward
                    dqn_agent.epsilon,                  # Current epsilon
                    stats['training_steps'],            # Training steps
                    stats['buffer_size'],               # Buffer size
                    win_rate                            # Win rate
                ])
        
        # Save model checkpoint periodically
        if (episode + 1) % config["save_frequency"] == 0:  # Every 2000 episodes
            os.makedirs("models", exist_ok=True)        # Create models directory if needed
            model_path = f"models/dqn_ep_{episode + 1}.pt"  # Create checkpoint filename
            dqn_agent.save(model_path)                  # Save agent to file
            print(f"Saved checkpoint: {model_path}")    # Print confirmation
    
    # Final evaluation after training completes
    print("\n" + "=" * 55)                            # Print separator
    print("FINAL EVALUATION")                          # Print final evaluation header
    
    # Run comprehensive final evaluation
    final_win_rate = evaluate_agent(dqn_agent, opponent, num_games=1000)  # 1000 games for accuracy
    print(f"Final win rate vs {opponent.name}: {final_win_rate:.1%}")      # Print final win rate
    
    # Save final trained model
    final_model_path = "models/dqn_final.pt"          # Final model filename
    dqn_agent.save(final_model_path)                   # Save final model
    print(f"Saved final model: {final_model_path}")    # Print confirmation
    
    # Print final statistics
    stats = dqn_agent.get_stats()                      # Get final agent statistics
    print(f"Training steps: {stats['training_steps']:,}")     # Total training steps
    print(f"Final buffer size: {stats['buffer_size']:,}")     # Final buffer size
    print(f"Final epsilon: {stats['epsilon']:.4f}")           # Final exploration rate
    print("DQN training completed!")                           # Print completion message


# Script entry point
if __name__ == "__main__":                             # If script is run directly
    train_dqn_agent()                                  # Start training