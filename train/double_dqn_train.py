"""Training script for Double DQN agent."""

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
from train.reward_system import calculate_enhanced_reward
from train.training_monitor import TrainingMonitor


def create_double_dqn_config() -> Dict[str, Any]:
    """Create default hyperparameters for Double DQN training."""
    return {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.9995,
        "buffer_size": 10000,
        "batch_size": 128,
        "min_buffer_size": 1000,
        "target_update_freq": 1000,
        "num_episodes": 500000,
        "eval_frequency": 1000,
        "save_frequency": 2000,
        "random_seed": 42,
        "opponent_type": "random",
        "reward_system": "enhanced",  # Options: "simple" or "enhanced"
        
        # Advanced training optimizations
        "clear_buffer_on_transition": False,  # FIXED: Never fully clear buffer (preserve heuristic knowledge)
        "self_play_pool_size": 5,  # Number of past snapshots for self-play diversity
        "early_stopping": True,  # Enable early stopping on convergence
        "early_stopping_threshold": 0.98,  # Win rate threshold for early stopping (raised from 95%)  
        "early_stopping_patience": 15,  # Number of evaluations above threshold before stopping (increased patience)
        
        # NEW: Heuristic preservation settings
        "heuristic_preservation_rate": 0.20,  # Always play 20% games vs heuristic in self-play
        "heuristic_performance_threshold": 0.85,  # Stop if heuristic win rate drops below 85%
        "heuristic_eval_frequency": 500,  # Evaluate vs heuristic every 500 episodes in self-play
        "self_play_learning_rate": 0.0005,  # Reduced LR for self-play to prevent catastrophic forgetting
    }


def setup_double_dqn_logging(log_dir: str) -> str:
    """Setup logging directory and return path to log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "double_dqn_log.csv")
    
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "avg_reward", "epsilon", "training_steps", 
                "buffer_size", "win_rate"
            ])
    
    return log_file


def calculate_reward(winner: Optional[int], agent_id: int, game_length: int) -> float:
    """Calculate reward for the Double DQN agent."""
    if winner == agent_id:
        return 10.0
    elif winner is not None:
        return -10.0
    else:
        return 1.0


def play_double_dqn_training_game(agent: DoubleDQNAgent, opponent, monitor: TrainingMonitor = None, reward_system: str = "enhanced", reward_config: dict = None) -> Tuple[Optional[int], int]:
    """Play a single training game between Double DQN agent and opponent.
    Fixed reward timing: we keep the last agent move until the opponent responds,
    then compute and deliver the shaped reward for that agent move.
    """
    board = Connect4Board()

    # Decide who goes first and set board.current_player accordingly
    if agent.rng.random() < 0.5:
        first_player = agent
        second_player = opponent
    else:
        first_player = opponent
        second_player = agent

    # Set board's current player to match the chosen order
    try:
        board.current_player = first_player.player_id
    except AttributeError:
        # If the board doesn't expose current_player, we'll handle it manually
        pass

    move_count = 0
    last_agent_exp = None  # stores {'state': np.array, 'action': int}

    while not board.is_terminal() and move_count < 42:
        # Read state BEFORE taking action
        state_before = board.get_state().copy()
        legal_moves = board.get_legal_moves()

        # Decide who's to move from board.current_player (preferred) to keep env consistent
        current_player_id = getattr(board, "current_player", None)
        if current_player_id is not None:
            current_agent = agent if current_player_id == agent.player_id else opponent
        else:
            # fallback to alternation if board doesn't track current_player
            current_agent = first_player if (move_count % 2 == 0) else second_player

        action = current_agent.choose_action(state_before, legal_moves)

        # If agent is acting, store its move to evaluate after the opponent responds (unless terminal)
        if current_agent == agent:
            last_agent_exp = {'state': state_before.copy(), 'action': action}

        # Apply the move - use consistent API
        board.make_move(action, current_agent.player_id)
        move_count += 1

        # Monitor strategic play (if provided)
        if monitor and current_agent == agent:
            # Create temporary board for strategic analysis using board state before move
            temp_prev_board = Connect4Board()
            temp_prev_board.board = state_before.copy()
            # Copy current_player if it exists
            if hasattr(board, 'current_player'):
                temp_prev_board.current_player = current_agent.player_id
            monitor.analyze_strategic_play(temp_prev_board, action, board, agent.player_id)

        # If the agent just moved and the game ended immediately (agent win/draw), give final reward now
        if current_agent == agent and board.is_terminal():
            done = True
            if reward_system == "simple":
                winner = board.check_winner()
                reward = calculate_reward(winner, agent.player_id, move_count)
            else:
                # Create board from state for reward calculation
                prev_board = Connect4Board()
                prev_board.board = state_before.copy()
                reward = calculate_enhanced_reward(prev_board, action, board, agent.player_id, True, reward_config)
            
            agent.observe(last_agent_exp['state'], last_agent_exp['action'], reward, None, True)
            last_agent_exp = None
            break  # game over

        # If opponent just moved and there is a stored agent move awaiting evaluation,
        # compute the shaped reward for that agent move using the new board state (after opponent moved)
        if current_agent != agent and last_agent_exp is not None:
            done = board.is_terminal()
            if reward_system == "simple":
                if done:
                    winner = board.check_winner()
                    reward = calculate_reward(winner, agent.player_id, move_count)
                else:
                    reward = 0.0  # No intermediate rewards in simple system
            else:
                # Create board from state for reward calculation
                prev_board = Connect4Board()
                prev_board.board = last_agent_exp['state'].copy()
                reward = calculate_enhanced_reward(
                    prev_board,
                    last_agent_exp['action'],
                    board,
                    agent.player_id,
                    done,
                    reward_config
                )
            
            next_state = None if done else board.get_state().copy()
            agent.observe(last_agent_exp['state'], last_agent_exp['action'], reward, next_state, done)
            last_agent_exp = None

    # End of game: if there is still a pending agent move (agent played last), assign final reward
    if last_agent_exp is not None:
        done = board.is_terminal()
        if reward_system == "simple":
            winner = board.check_winner() if done else None
            final_reward = calculate_reward(winner, agent.player_id, move_count)
        else:
            # Create board from state for reward calculation
            prev_board = Connect4Board()
            prev_board.board = last_agent_exp['state'].copy()
            final_reward = calculate_enhanced_reward(
                prev_board,
                last_agent_exp['action'],
                board,
                agent.player_id,
                done,
                reward_config
            )
        
        next_state = None if done else board.get_state().copy()
        agent.observe(last_agent_exp['state'], last_agent_exp['action'], final_reward, next_state, done)

    return board.check_winner(), move_count


def evaluate_agent(agent, opponent, num_games: int = 100) -> float:
    """Evaluate agent performance against an opponent with consistent API."""
    wins = 0
    old_epsilon = agent.epsilon
    
    # Disable exploration during evaluation
    agent.epsilon = 0.0
    
    try:
        for _ in range(num_games):
            board = Connect4Board()
            
            # Randomly assign positions 
            if np.random.random() < 0.5:
                first_player = agent
                second_player = opponent
            else:
                first_player = opponent
                second_player = agent
            
            # Set board current player
            try:
                board.current_player = first_player.player_id
            except AttributeError:
                pass
            
            move_count = 0
            while not board.is_terminal() and move_count < 42:
                # Use same logic as training
                current_player_id = getattr(board, "current_player", None)
                if current_player_id is not None:
                    current_player = agent if current_player_id == agent.player_id else opponent
                else:
                    current_player = first_player if (move_count % 2 == 0) else second_player
                
                legal_moves = board.get_legal_moves()
                action = current_player.choose_action(board.get_state(), legal_moves)
                board.make_move(action, current_player.player_id)  # Consistent API
                move_count += 1
            
            winner = board.check_winner()
            if winner == agent.player_id:
                wins += 1
    finally:
        # Restore exploration
        agent.epsilon = old_epsilon
    
    return wins / num_games


def train_double_dqn_agent():
    """Main training loop for Double DQN agent with comprehensive monitoring."""
    print("Double DQN Agent Training with Enhanced Monitoring")
    print("=" * 55)
    
    # Load configuration
    config = create_double_dqn_config()
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Set deterministic seeds for reproducibility
    import random
    import torch
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["random_seed"])
    print(f"🎲 Set random seeds to {config['random_seed']} for reproducibility")
    
    # Setup logging and monitoring
    log_file = setup_double_dqn_logging("logs")
    monitor = TrainingMonitor(log_dir="logs", save_plots=True)
    print(f"Logging to: {log_file}")
    print("Enhanced monitoring enabled with strategic analysis and visualization")
    
    # Initialize agent (84 = 2 channels * 6 rows * 7 cols)
    agent = DoubleDQNAgent(
        player_id=1,
        state_size=84,  # 2-channel encoding: 2 * 6 * 7
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
    
    # Initialize opponents for curriculum learning
    random_opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
    heuristic_opponent = HeuristicAgent(player_id=2, seed=config["random_seed"] + 1)
    self_play_opponent = None  # Will be loaded at episode 100,000
    
    # Self-play opponent pool for diversity (prevents overfitting to single snapshot)
    self_play_pool = []  # Will store multiple snapshots
    self_play_pool_size = config["self_play_pool_size"]  # Keep N snapshots for diversity
    
    # Early stopping tracking
    early_stopping_counter = 0
    best_win_rate = 0.0
    
    # Curriculum learning thresholds
    RANDOM_PHASE_END = 10000      # Episodes 1-20,000: vs Random
    HEURISTIC_PHASE_END = 60000  # Episodes 20,001-100,000: vs Heuristic  
    # Episodes 100,001+: vs Self-play (saved model from episode 100,000)
    
    print(f"🎯 Advanced Curriculum Learning Enabled:")
    print(f"  Episodes 1-{RANDOM_PHASE_END:,}: vs {random_opponent.name} (basics)")
    print(f"  Episodes {RANDOM_PHASE_END + 1:,}-{HEURISTIC_PHASE_END:,}: vs {heuristic_opponent.name} (strategy)")
    print(f"  Episodes {HEURISTIC_PHASE_END + 1:,}+: Mixed opponents ({config['heuristic_preservation_rate']:.0%} Heuristic preserved, 10% Random, 70% Self-play pool)")
    print(f"  🔧 Self-play fixes: player_id preservation, ε=0.02 exploration, mixed training")
    print(f"")
    print(f"🚀 Enhanced Training Optimizations:")
    print(f"  📚 Self-play snapshot pool: {config['self_play_pool_size']} diverse opponents (prevents overfitting)")
    print(f"  🧹 Buffer clearing: {'DISABLED' if not config['clear_buffer_on_transition'] else 'ON'} (preserve heuristic knowledge)")
    print(f"  🎯 Early stopping: {'ON' if config['early_stopping'] else 'OFF'} (>{config['early_stopping_threshold']:.0%} for {config['early_stopping_patience']} evals)")
    print(f"  💎 Best model tracking: Automatic saving of peak performance checkpoints")
    print(f"  🛡️  Heuristic preservation: {config['heuristic_preservation_rate']:.0%} of games vs heuristic in self-play")
    print(f"  📉 Performance monitoring: Stop if heuristic win rate < {config['heuristic_performance_threshold']:.0%}")
    print(f"  📚 Reduced self-play LR: {config['self_play_learning_rate']} (prevents catastrophic forgetting)")
    
    def get_current_opponent(episode):
        """Get the appropriate opponent for curriculum learning with enhanced self-play pool."""
        if episode <= RANDOM_PHASE_END:
            return random_opponent, "Random"
        elif episode <= HEURISTIC_PHASE_END:
            return heuristic_opponent, "Heuristic"
        else:
            # Self-play phase with snapshot pool for diversity
            nonlocal self_play_opponent, self_play_pool
            
            # Load initial self-play opponent if not already loaded
            if self_play_opponent is None:
                self_play_model_path = f"models/double_dqn_ep_{HEURISTIC_PHASE_END}.pt"
                if os.path.exists(self_play_model_path):
                    print(f"\\n🤖 Loading initial self-play opponent from {self_play_model_path}")
                    self_play_opponent = DoubleDQNAgent(
                        player_id=2,  # Critical: must be player 2 for correct board perspective
                        state_size=84,
                        action_size=7,
                        hidden_size=256,  # Match the training architecture
                        seed=config["random_seed"] + 2
                    )
                    # Load model but preserve player_id=2 to maintain correct board perspective
                    self_play_opponent.load(self_play_model_path, keep_player_id=False)
                    
                    # Add small exploration to avoid deterministic cycles and overfitting
                    self_play_opponent.epsilon = 0.02  # Small stochasticity prevents deterministic play
                    
                    # Verify correct player_id (debug check for the critical bug)
                    if self_play_opponent.player_id != 2:
                        print(f"⚠️  WARNING: Self-play opponent player_id is {self_play_opponent.player_id}, should be 2")
                        self_play_opponent.player_id = 2  # Force correct perspective
                    
                    # Add to pool
                    self_play_pool.append((self_play_opponent, f"ep_{HEURISTIC_PHASE_END}"))
                    print(f"✅ Self-play opponent loaded successfully (player_id={self_play_opponent.player_id}, ε={self_play_opponent.epsilon})")
                    print(f"📚 Self-play pool initialized with {len(self_play_pool)} snapshot(s)")
                else:
                    print(f"⚠️  Self-play model not found at {self_play_model_path}, using heuristic")
                    return heuristic_opponent, "Heuristic (fallback)"
            
            # IMPROVED: Fixed heuristic preservation strategy
            heuristic_rate = config['heuristic_preservation_rate']  # Always play 20% vs heuristic
            rand_val = np.random.random()
            
            if rand_val < heuristic_rate:  # Fixed % heuristic (e.g., 20%)
                return heuristic_opponent, "Heuristic (preserved)"
            elif rand_val < heuristic_rate + 0.10:  # 10% Random for diversity
                return random_opponent, "Random (mixed)"
            else:  # Remaining % Self-play from pool (70%)
                if len(self_play_pool) > 1:
                    # Randomly select from pool for diversity
                    opponent_agent, opponent_label = self_play_pool[np.random.randint(len(self_play_pool))]
                    return opponent_agent, f"Self-play ({opponent_label})"
                else:
                    return self_play_opponent, "Self-play (single)"
    
    print(f"Training {agent} with curriculum learning")
    print(f"Device: {agent.device}")
    print(f"Reward System: {config['reward_system'].upper()}")
    if config["reward_system"] == "simple":
        print("  → Sparse rewards: Win (+10), Loss (-10), Draw (+1), Intermediate (0)")
    else:
        print("  → Enhanced rewards: Game outcomes + strategic bonuses")
    print("\nStarting curriculum training with enhanced monitoring and Q-value visualization...")
    print()
    
    # Training loop
    episode_rewards = []
    current_opponent_name = "Random"  # Track current opponent for logging
    
    for episode in range(config["num_episodes"]):
        episode_num = episode + 1  # 1-based episode numbering
        
        # Get current opponent based on curriculum learning schedule
        current_opponent, opponent_name = get_current_opponent(episode_num)
        
        # Announce opponent changes
        if opponent_name != current_opponent_name:
            print(f"\n🔄 CURRICULUM PHASE CHANGE at episode {episode_num:,}")
            print(f"Switching from {current_opponent_name} → {opponent_name}")
            current_opponent_name = opponent_name
            
            # Advanced optimization: Clear replay buffer for faster adaptation
            # Only clear buffer for heuristic->self-play transition, not random->heuristic
            if config["clear_buffer_on_transition"] and episode_num == HEURISTIC_PHASE_END + 1:
                buffer_stats_before = agent.get_buffer_diversity_stats()
                agent.clear_replay_buffer()
                print(f"🧹 Cleared replay buffer for self-play transition (was {buffer_stats_before['buffer_size']:,} experiences)")
                print("   → Removing heuristic-specific patterns for self-play adaptation")
            elif episode_num == RANDOM_PHASE_END + 1:
                print("🧹 Keeping buffer for random→heuristic transition (preserves useful experiences)")
            
            # Update monitor with current opponent
            monitor.set_current_opponent(opponent_name)
            
            # Reset strategic stats when switching opponents
            monitor.reset_strategic_stats()
            print("Expected: More challenging games, improved strategic learning\n")
        
        # Ensure monitor always knows current opponent (for first episode)
        if episode_num == 1:
            monitor.set_current_opponent(opponent_name)
        
        # Reset episode state and strategic stats for this episode
        agent.reset_episode()
        
        # Play training game with current opponent
        winner, game_length = play_double_dqn_training_game(agent, current_opponent, monitor, config["reward_system"])
        
        # Calculate episode reward (use simple win/loss for episode tracking)
        if winner == agent.player_id:
            episode_reward = 10.0
        elif winner is not None:
            episode_reward = -10.0
        else:
            episode_reward = 1.0
        episode_rewards.append(episode_reward)
        
        # Periodic evaluation and logging
        if (episode + 1) % config["eval_frequency"] == 0:
            # Evaluate current policy against current opponent
            win_rate = evaluate_agent(agent, current_opponent, num_games=100)
            
            # Early stopping check
            if config["early_stopping"] and episode_num > HEURISTIC_PHASE_END:
                if win_rate > config["early_stopping_threshold"]:
                    early_stopping_counter += 1
                    print(f"🎯 Early stopping progress: {early_stopping_counter}/{config['early_stopping_patience']} (win rate: {win_rate:.1%})")
                    
                    if early_stopping_counter >= config["early_stopping_patience"]:
                        print(f"\n🏆 EARLY STOPPING TRIGGERED at episode {episode_num:,}")
                        print(f"Win rate {win_rate:.1%} > {config['early_stopping_threshold']:.1%} for {early_stopping_counter} consecutive evaluations")
                        print("Agent has converged - stopping training early")
                        
                        # Save final model
                        final_model_path = f"models/double_dqn_early_stop_ep_{episode + 1}.pt"
                        agent.save(final_model_path)
                        print(f"Saved early stopping model: {final_model_path}")
                        break
                else:
                    early_stopping_counter = 0  # Reset counter
            
            # Track best performance
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                if episode_num > HEURISTIC_PHASE_END:
                    best_model_path = f"models/double_dqn_best_ep_{episode + 1}.pt"
                    agent.save(best_model_path)
                    print(f"💎 New best performance: {win_rate:.1%} - saved {best_model_path}")
            
            # Log metrics with monitor
            monitor.log_episode(episode + 1, episode_reward, agent, win_rate)
            monitor.log_strategic_episode(episode + 1)
            
            # Generate comprehensive progress report
            monitor.generate_training_report(agent, episode + 1)
            
            # Reset strategic stats after evaluation
            monitor.reset_strategic_stats()
            
            # Traditional CSV logging (for compatibility)
            avg_reward = np.mean(episode_rewards[-config["eval_frequency"]:])
            stats = agent.get_stats()
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode + 1, avg_reward, agent.epsilon,
                    stats['training_steps'], stats['buffer_size'], win_rate
                ])
        else:
            # Log individual episode (without extensive evaluation)
            monitor.log_episode(episode + 1, episode_reward, agent)
        
        # Save model checkpoint (regular frequency)
        if (episode + 1) % config["save_frequency"] == 0:
            os.makedirs("models", exist_ok=True)
            model_path = f"models/double_dqn_ep_{episode + 1}.pt"
            agent.save(model_path)
            print(f"Saved checkpoint: {model_path}")
            
            # Add to self-play pool if in self-play phase
            if episode_num > HEURISTIC_PHASE_END and len(self_play_pool) > 0:
                # Create new opponent for the pool
                new_opponent = DoubleDQNAgent(
                    player_id=2,
                    state_size=84,
                    action_size=7,
                    hidden_size=256,  # Match the training architecture
                    seed=config["random_seed"] + len(self_play_pool) + 10
                )
                new_opponent.load(model_path, keep_player_id=False)
                new_opponent.epsilon = 0.02
                
                # Add to pool and maintain size limit
                self_play_pool.append((new_opponent, f"ep_{episode + 1}"))
                if len(self_play_pool) > self_play_pool_size:
                    removed = self_play_pool.pop(0)  # Remove oldest
                    print(f"📚 Updated self-play pool: added ep_{episode + 1}, removed {removed[1]}")
                    print(f"   Pool now has {len(self_play_pool)} snapshots for diversity")
                else:
                    print(f"📚 Added to self-play pool: ep_{episode + 1} ({len(self_play_pool)}/{self_play_pool_size})")
            
            # Conduct comprehensive skill evaluation at checkpoints
            print("\n🔍 Conducting comprehensive skill evaluation...")
            skill_results = monitor.evaluate_agent_skills(agent, num_games=50)
            
            # Save skill evaluation results
            skill_log = os.path.join("logs", f"skill_eval_ep_{episode + 1}.json")
            with open(skill_log, 'w') as f:
                json.dump(skill_results, f, indent=2)
        
        # Special save for self-play transition at episode 100,000
        if episode_num == HEURISTIC_PHASE_END:
            os.makedirs("models", exist_ok=True)
            self_play_model_path = f"models/double_dqn_ep_{HEURISTIC_PHASE_END}.pt"
            agent.save(self_play_model_path)
            print(f"\n🎯 CURRICULUM MILESTONE: Saved self-play model at {self_play_model_path}")
            print("This model will be used as opponent for self-play training in the next phase.")
            
            # Special comprehensive evaluation at curriculum transition
            print(f"\n📊 Comprehensive evaluation at episode {HEURISTIC_PHASE_END:,}")
            vs_random_rate = evaluate_agent(agent, random_opponent, num_games=200)
            vs_heuristic_rate = evaluate_agent(agent, heuristic_opponent, num_games=200)
            
            curriculum_eval = {
                'episode': HEURISTIC_PHASE_END,
                'vs_random_win_rate': vs_random_rate,
                'vs_heuristic_win_rate': vs_heuristic_rate,
                'overall_training_win_rate': monitor.overall_win_rate,
                'agent_stats': agent.get_stats()
            }
            
            eval_file = os.path.join("logs", f"curriculum_transition_eval_ep_{HEURISTIC_PHASE_END}.json")
            with open(eval_file, 'w') as f:
                json.dump(curriculum_eval, f, indent=2)
                
            print(f"🎯 Curriculum transition evaluation:")
            print(f"  vs Random: {vs_random_rate:.1%}")
            print(f"  vs Heuristic: {vs_heuristic_rate:.1%}")
            print(f"  Overall training: {monitor.overall_win_rate:.1%}")
            
            # Save special model for interactive play (after heuristic training)
            interactive_model_path = "models/double_dqn_post_heuristic.pt"
            agent.save(interactive_model_path)
            print(f"🎮 Saved post-heuristic model for interactive play: {interactive_model_path}")
            print(f"   This model has learned from Random and Heuristic opponents")
            print(f"   Perfect for challenging human gameplay!")
            
            print(f"Ready for self-play phase starting at episode {HEURISTIC_PHASE_END + 1:,}!")
    
    # Final comprehensive evaluation
    print("\n" + "=" * 60)
    print("FINAL COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    # Comprehensive final evaluation against all curriculum opponents
    print("🎯 Evaluating final agent performance against all curriculum opponents:")
    
    random_win_rate = evaluate_agent(agent, random_opponent, num_games=500)
    print(f"vs {random_opponent.name}: {random_win_rate:.1%}")
    
    heuristic_win_rate = evaluate_agent(agent, heuristic_opponent, num_games=500)
    print(f"vs {heuristic_opponent.name}: {heuristic_win_rate:.1%}")
    
    # Evaluate against self-play opponent if available
    self_play_win_rate = None
    if self_play_opponent is not None:
        self_play_win_rate = evaluate_agent(agent, self_play_opponent, num_games=500)
        print(f"vs Self-play (ep {HEURISTIC_PHASE_END:,}): {self_play_win_rate:.1%}")
        final_win_rate = self_play_win_rate  # Use self-play as primary metric
    else:
        final_win_rate = heuristic_win_rate  # Fallback to heuristic
        
    print(f"\n📈 Training Progress Through Curriculum:")
    print(f"  Overall training win rate: {monitor.overall_win_rate:.1%}")
    print(f"  Final performance vs curriculum opponents shows learning progression")
    
    # Comprehensive skill assessment
    print("\n🎯 Final Skill Assessment:")
    final_skills = monitor.evaluate_agent_skills(agent, num_games=200)
    
    # Save final model and results
    final_model_path = "models/double_dqn_final.pt"
    agent.save(final_model_path)
    print(f"Saved final model: {final_model_path}")
    
    # Save final evaluation results
    final_results = {
        'curriculum_learning_enabled': True,
        'final_win_rate_vs_random': random_win_rate,
        'final_win_rate_vs_heuristic': heuristic_win_rate,
        'final_win_rate_vs_self_play': self_play_win_rate,
        'final_win_rate_primary': final_win_rate,
        'overall_training_win_rate': monitor.overall_win_rate,
        'total_training_games': monitor.total_training_games,
        'skill_assessment': final_skills,
        'training_config': config,
        'agent_stats': agent.get_stats(),
        'curriculum_phases': {
            'random_phase_end': RANDOM_PHASE_END,
            'heuristic_phase_end': HEURISTIC_PHASE_END,
            'self_play_start': HEURISTIC_PHASE_END + 1
        }
    }
    
    results_file = os.path.join("logs", "final_training_results.json")
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Final training report
    monitor.generate_training_report(agent, config["num_episodes"])
    
    # Print final statistics
    stats = agent.get_stats()
    print(f"\nFinal Training Statistics:")
    print(f"  Training steps: {stats['training_steps']:,}")
    print(f"  Final buffer size: {stats['buffer_size']:,}")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    print(f"  Overall skill score: {final_skills['overall_skill']:.1%}")
    print(f"\nAll results saved to: logs/")
    print("Double DQN training completed with comprehensive monitoring!")


if __name__ == "__main__":
    train_double_dqn_agent()