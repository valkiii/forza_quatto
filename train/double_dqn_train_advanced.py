#!/usr/bin/env python3
"""Advanced Double DQN training with aggressive heuristic preservation and proper monitoring."""

import os
import sys
import json
import csv
import math
from typing import Dict, Any, List, Tuple
from collections import deque
import numpy as np

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


def create_advanced_double_dqn_config() -> Dict[str, Any]:
    """Create advanced configuration with aggressive heuristic preservation."""
    return {
        "learning_rate": 0.001,
        "discount_factor": 0.95,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.9995,
        "buffer_size": 20000,  # Even larger buffer for diversity
        "batch_size": 128,
        "min_buffer_size": 1000,
        "target_update_freq": 1000,
        "num_episodes": 150000,  # Focused training
        "eval_frequency": 500,   # More frequent evaluation
        "save_frequency": 2000,
        "random_seed": 42,
        "reward_system": "enhanced",
        
        # AGGRESSIVE HEURISTIC PRESERVATION (your suggestions)
        "heuristic_preservation_rate_min": 0.30,  # NEVER below 30%
        "heuristic_preservation_rate_max": 0.50,  # Can go up to 50% if needed
        "heuristic_performance_threshold": 0.90,  # Higher threshold (90% vs heuristic)
        "heuristic_eval_frequency": 300,  # Check every 300 episodes
        
        # DRAMATIC LEARNING RATE REDUCTION (10x, not 2x)
        "self_play_learning_rate": 0.0001,  # 10x reduction for fine-tuning
        
        # STRATIFIED REPLAY BUFFER (maintain experience distribution)
        "buffer_heuristic_ratio": 0.30,  # 30% heuristic experiences
        "buffer_random_ratio": 0.10,     # 10% random experiences  
        "buffer_selfplay_ratio": 0.60,   # 60% self-play experiences
        "never_clear_buffer": True,      # NEVER clear the buffer
        
        # GRADUAL SELF-PLAY INTRODUCTION
        "self_play_start_ratio": 0.30,   # Start with only 30% self-play
        "self_play_end_ratio": 0.60,     # End with 60% self-play
        "self_play_ramp_episodes": 20000, # Gradual increase over 20K episodes
        
        # OPPONENT POPULATION DIVERSITY
        "opponent_pool_size": 7,         # Keep 7 historical snapshots
        "pool_update_frequency": 3000,   # Add new snapshot every 3K episodes
        
        # REGULARIZATION
        "weight_decay": 0.01,            # L2 regularization
        "dropout_rate": 0.1,             # Dropout in training
        
        # CURRICULUM TIMING (extended phases for stronger foundation)
        "random_phase_end": 20000,       # Extended random phase for better exploration
        "heuristic_phase_end": 80000,    # Extended heuristic learning for stronger foundation
        # Mixed phase: 80K+ with optimized preservation
        
        # EARLY STOPPING
        "early_stopping": True,
        "early_stopping_threshold": 0.88,  # Stop if heuristic performance drops
        "early_stopping_patience": 5,      # Less patience - act quickly
    }


def get_gradual_self_play_ratio(episode: int, config: Dict[str, Any], heuristic_phase_end: int) -> float:
    """Calculate gradually increasing self-play ratio."""
    if episode <= heuristic_phase_end:
        return 0.0  # No self-play during heuristic phase
    
    episodes_since_selfplay = episode - heuristic_phase_end
    ramp_episodes = config["self_play_ramp_episodes"]
    
    if episodes_since_selfplay >= ramp_episodes:
        return config["self_play_end_ratio"]  # Full self-play ratio
    
    # Linear interpolation
    progress = episodes_since_selfplay / ramp_episodes
    start_ratio = config["self_play_start_ratio"]
    end_ratio = config["self_play_end_ratio"]
    
    return start_ratio + (end_ratio - start_ratio) * progress


def train_advanced_double_dqn():
    """Advanced training with aggressive heuristic preservation and proper monitoring."""
    print("🚀 ADVANCED Double DQN with Aggressive Heuristic Preservation")
    print("=" * 65)
    
    config = create_advanced_double_dqn_config()
    print(f"📋 Advanced Configuration Loaded")
    
    # Set seeds
    import random
    import torch
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["random_seed"])
    
    # Setup advanced logging with proper monitor
    log_dir = "logs_advanced"
    os.makedirs(log_dir, exist_ok=True)
    log_file = setup_double_dqn_logging(log_dir)
    monitor = TrainingMonitor(log_dir=log_dir, save_plots=True)
    
    print(f"📊 Advanced logging: {log_dir}/ with plot generation enabled")
    
    # Initialize agent with regularization
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
    
    # Add L2 regularization to optimizer
    for param_group in agent.optimizer.param_groups:
        param_group['weight_decay'] = config["weight_decay"]
    
    print(f"🛡️ L2 regularization enabled: {config['weight_decay']}")
    
    # Initialize opponents
    random_opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
    heuristic_opponent = HeuristicAgent(player_id=2, seed=config["random_seed"] + 1)
    
    # Historical opponent pool for diversity
    opponent_pool = []
    
    # Curriculum thresholds
    RANDOM_PHASE_END = config["random_phase_end"]
    HEURISTIC_PHASE_END = config["heuristic_phase_end"]
    
    print(f"\n🎯 ADVANCED Curriculum with Aggressive Preservation:")
    print(f"  📚 Episodes 1-{RANDOM_PHASE_END:,}: vs Random")
    print(f"  🧠 Episodes {RANDOM_PHASE_END + 1:,}-{HEURISTIC_PHASE_END:,}: vs Heuristic")
    print(f"  🔄 Episodes {HEURISTIC_PHASE_END + 1:,}+: Mixed training")
    print(f"\n🛡️ Aggressive Heuristic Preservation:")
    print(f"  🎯 NEVER below {config['heuristic_preservation_rate_min']:.0%} heuristic games")
    print(f"  📊 Performance monitoring every {config['heuristic_eval_frequency']} episodes")
    print(f"  🚨 Emergency stop if < {config['heuristic_performance_threshold']:.0%} vs heuristic")
    print(f"  🧠 Enhanced monitoring with PLOTS in {log_dir}/")
    print(f"  🐌 10x LR reduction for self-play: {config['self_play_learning_rate']}")
    print(f"  📈 Gradual self-play: {config['self_play_start_ratio']:.0%} → {config['self_play_end_ratio']:.0%}")
    
    def get_advanced_opponent(episode):
        """Advanced opponent selection with gradual self-play and guaranteed preservation."""
        nonlocal opponent_pool
        
        if episode <= RANDOM_PHASE_END:
            return random_opponent, "Random"
        elif episode <= HEURISTIC_PHASE_END:
            return heuristic_opponent, "Heuristic"
        else:
            # Mixed phase with gradual self-play introduction
            
            # Initialize opponent pool if needed
            if not opponent_pool and episode == HEURISTIC_PHASE_END + 1:
                model_path = f"models_advanced/double_dqn_ep_{HEURISTIC_PHASE_END}.pt"
                if os.path.exists(model_path):
                    print(f"\n🤖 Initializing opponent pool from {model_path}")
                    initial_opponent = DoubleDQNAgent(
                        player_id=2, state_size=84, action_size=7,
                        hidden_size=256, seed=config["random_seed"] + 100
                    )
                    initial_opponent.load(model_path, keep_player_id=False)
                    initial_opponent.epsilon = 0.02
                    opponent_pool.append((initial_opponent, f"ep_{HEURISTIC_PHASE_END}"))
                    print(f"✅ Opponent pool initialized with {len(opponent_pool)} agent(s)")
            
            # Calculate current self-play ratio (gradual increase)
            current_selfplay_ratio = get_gradual_self_play_ratio(episode, config, HEURISTIC_PHASE_END)
            heuristic_ratio = max(config["heuristic_preservation_rate_min"], 
                                1.0 - current_selfplay_ratio - 0.05)  # At least 30%, leave 5% for random
            random_ratio = 0.05  # Small random component for diversity
            
            # Select opponent based on ratios
            rand_val = np.random.random()
            
            if rand_val < heuristic_ratio:
                return heuristic_opponent, f"Heuristic (preserved {heuristic_ratio:.0%})"
            elif rand_val < heuristic_ratio + random_ratio:
                return random_opponent, "Random (diversity)"
            else:
                if opponent_pool:
                    opponent_agent, label = opponent_pool[np.random.randint(len(opponent_pool))]
                    return opponent_agent, f"Self-play ({label}, ratio={current_selfplay_ratio:.0%})"
                else:
                    # Fallback to heuristic if no self-play opponents available
                    return heuristic_opponent, "Heuristic (fallback)"
    
    # Training state
    episode_rewards = []
    current_phase = "Random"
    heuristic_performance_history = []
    learning_rate_reduced = False
    best_heuristic_performance = 0.0
    
    print(f"\n🚀 Starting advanced training with enhanced monitoring...")
    print()
    
    for episode in range(config["num_episodes"]):
        episode_num = episode + 1
        
        # Get current opponent
        current_opponent, opponent_name = get_advanced_opponent(episode_num)
        
        # Handle phase transitions
        phase_name = opponent_name.split()[0]
        if phase_name != current_phase:
            print(f"\n🔄 PHASE TRANSITION at episode {episode_num:,}")
            print(f"Switching from {current_phase} → {phase_name} phase")
            
            # DRAMATIC learning rate reduction (10x) for self-play
            if episode_num == HEURISTIC_PHASE_END + 1 and not learning_rate_reduced:
                old_lr = agent.optimizer.param_groups[0]['lr']
                new_lr = config['self_play_learning_rate']
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = new_lr
                learning_rate_reduced = True
                print(f"🐌 DRAMATIC LR reduction (10x): {old_lr:.6f} → {new_lr:.6f}")
                print("🛡️ Entering fine-tuning mode to preserve heuristic knowledge")
            
            current_phase = phase_name
            monitor.set_current_opponent(opponent_name)
            monitor.reset_strategic_stats()
        
        # Initialize monitor
        if episode_num == 1:
            monitor.set_current_opponent(opponent_name)
        
        # Ensure monitor always knows current opponent (for proper tracking)
        if opponent_name != getattr(monitor, 'current_opponent', ''):
            monitor.set_current_opponent(opponent_name)
        
        # Reset episode state and strategic stats for this episode
        agent.reset_episode()
        
        # Play training game with strategic monitoring
        winner, game_length = play_double_dqn_training_game(
            agent, current_opponent, monitor, config["reward_system"]
        )
        
        # Calculate episode reward
        episode_reward = 10.0 if winner == agent.player_id else (-10.0 if winner is not None else 1.0)
        episode_rewards.append(episode_reward)
        
        # AGGRESSIVE heuristic performance monitoring
        if (episode_num > HEURISTIC_PHASE_END and 
            episode_num % config["heuristic_eval_frequency"] == 0):
            
            print(f"\n🛡️ CRITICAL heuristic preservation check at episode {episode_num:,}...")
            heuristic_win_rate = evaluate_agent(agent, heuristic_opponent, num_games=150)  # More games for accuracy
            heuristic_performance_history.append((episode_num, heuristic_win_rate))
            
            # Track best performance
            if heuristic_win_rate > best_heuristic_performance:
                best_heuristic_performance = heuristic_win_rate
                # Save best model
                os.makedirs("models_advanced", exist_ok=True)
                best_model_path = f"models_advanced/double_dqn_best_heuristic_ep_{episode_num}.pt"
                agent.save(best_model_path)
                print(f"💎 NEW BEST heuristic performance: {heuristic_win_rate:.1%} - saved {os.path.basename(best_model_path)}")
            
            print(f"📊 Current vs best: {heuristic_win_rate:.1%} vs {best_heuristic_performance:.1%}")
            
            # AGGRESSIVE early stopping
            if heuristic_win_rate < config["heuristic_performance_threshold"]:
                print(f"\n🚨 CRITICAL HEURISTIC DEGRADATION DETECTED!")
                print(f"Performance: {heuristic_win_rate:.1%} < threshold {config['heuristic_performance_threshold']:.1%}")
                print(f"🛑 EMERGENCY STOP - preventing catastrophic forgetting")
                
                # Emergency save
                emergency_save_path = f"models_advanced/double_dqn_emergency_stop_ep_{episode_num}.pt"
                agent.save(emergency_save_path)
                print(f"💾 Emergency save: {emergency_save_path}")
                
                # Save detailed failure analysis
                failure_analysis = {
                    'episode': episode_num,
                    'heuristic_performance': heuristic_win_rate,
                    'threshold': config['heuristic_performance_threshold'],
                    'best_performance_achieved': best_heuristic_performance,
                    'performance_history': heuristic_performance_history,
                    'current_learning_rate': agent.optimizer.param_groups[0]['lr'],
                    'config': config
                }
                
                with open(f"{log_dir}/catastrophic_forgetting_analysis.json", 'w') as f:
                    json.dump(failure_analysis, f, indent=2)
                
                print(f"📋 Failure analysis saved for debugging")
                break
            else:
                performance_status = "🟢 EXCELLENT" if heuristic_win_rate > 0.95 else "✅ GOOD" if heuristic_win_rate > 0.88 else "⚠️ ACCEPTABLE"
                print(f"{performance_status} - Heuristic knowledge strongly preserved")
        
        # Regular evaluation with enhanced monitoring
        if episode_num % config["eval_frequency"] == 0:
            win_rate = evaluate_agent(agent, current_opponent, num_games=100)
            
            # Enhanced logging with proper monitor integration
            monitor.log_episode(episode_num, episode_reward, agent, win_rate)
            monitor.log_strategic_episode(episode_num)  # Log strategic stats
            monitor.generate_training_report(agent, episode_num)  # Generate plots!
            monitor.reset_strategic_stats()  # Reset after logging
            
            # CSV logging for compatibility
            avg_reward = np.mean(episode_rewards[-config["eval_frequency"]:])
            stats = agent.get_stats()
            
            with open(log_file.replace('logs/', f'{log_dir}/'), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode_num, avg_reward, agent.epsilon,
                    stats['training_steps'], stats['buffer_size'], win_rate
                ])
            
            print(f"📊 Episode {episode_num:,}: {win_rate:.1%} vs {opponent_name.split()[0]}")
            
            # Show gradual self-play progress
            if episode_num > HEURISTIC_PHASE_END:
                current_selfplay_ratio = get_gradual_self_play_ratio(episode_num, config, HEURISTIC_PHASE_END)
                print(f"    Self-play ratio: {current_selfplay_ratio:.0%}, Pool size: {len(opponent_pool)}")
        else:
            # Log individual episodes (without extensive evaluation)
            monitor.log_episode(episode_num, episode_reward, agent)
        
        # Save checkpoints and update opponent pool
        if episode_num % config["save_frequency"] == 0:
            os.makedirs("models_advanced", exist_ok=True)
            model_path = f"models_advanced/double_dqn_ep_{episode_num}.pt"
            agent.save(model_path)
            
            # Special saves with comprehensive evaluation
            if episode_num == HEURISTIC_PHASE_END:
                special_path = "models_advanced/double_dqn_post_heuristic_advanced.pt"
                agent.save(special_path)
                print(f"🎮 Saved advanced post-heuristic model: {special_path}")
                
                # Comprehensive evaluation at curriculum transition
                print(f"\n📊 Comprehensive evaluation at episode {HEURISTIC_PHASE_END:,}")
                vs_random_rate = evaluate_agent(agent, random_opponent, num_games=200)
                vs_heuristic_rate = evaluate_agent(agent, heuristic_opponent, num_games=200)
                
                curriculum_eval = {
                    'episode': HEURISTIC_PHASE_END,
                    'vs_random_win_rate': vs_random_rate,
                    'vs_heuristic_win_rate': vs_heuristic_rate,
                    'overall_training_win_rate': getattr(monitor, 'overall_win_rate', 0.0),
                    'agent_stats': agent.get_stats()
                }
                
                eval_file = os.path.join(log_dir, f"curriculum_transition_eval_ep_{HEURISTIC_PHASE_END}.json")
                with open(eval_file, 'w') as f:
                    json.dump(curriculum_eval, f, indent=2)
                
                print(f"🎯 Curriculum transition evaluation:")
                print(f"  vs Random: {vs_random_rate:.1%}")
                print(f"  vs Heuristic: {vs_heuristic_rate:.1%}")
                print(f"  Ready for advanced mixed training!")
            
            # Conduct comprehensive skill evaluation at regular checkpoints
            print(f"\n🔍 Conducting comprehensive skill evaluation...")
            try:
                skill_results = monitor.evaluate_agent_skills(agent, num_games=100)
                
                # Save skill evaluation results
                skill_log = os.path.join(log_dir, f"skill_eval_ep_{episode_num}.json")
                with open(skill_log, 'w') as f:
                    json.dump(skill_results, f, indent=2)
                    
                print(f"✅ Skill evaluation completed: {skill_results.get('overall_skill', 0.0):.1%} overall skill")
            except Exception as e:
                print(f"⚠️ Skill evaluation failed: {e}")
            
            # Update opponent pool
            if (episode_num > HEURISTIC_PHASE_END and 
                episode_num % config["pool_update_frequency"] == 0 and
                len(opponent_pool) < config["opponent_pool_size"]):
                
                new_opponent = DoubleDQNAgent(
                    player_id=2, state_size=84, action_size=7,
                    hidden_size=256, seed=config["random_seed"] + len(opponent_pool) + 200
                )
                new_opponent.load(model_path, keep_player_id=False)
                new_opponent.epsilon = 0.02
                opponent_pool.append((new_opponent, f"ep_{episode_num}"))
                print(f"🏆 Added to opponent pool: {len(opponent_pool)}/{config['opponent_pool_size']} diverse agents")
    
    # Final evaluation and summary
    print(f"\n🏁 Advanced training completed at episode {episode_num:,}")
    
    # Comprehensive final testing
    print(f"\n📊 FINAL COMPREHENSIVE EVALUATION:")
    final_vs_random = evaluate_agent(agent, random_opponent, num_games=300)
    final_vs_heuristic = evaluate_agent(agent, heuristic_opponent, num_games=300)
    
    print(f"📈 Final Performance:")
    print(f"  vs Random:    {final_vs_random:.1%}")
    print(f"  vs Heuristic: {final_vs_heuristic:.1%}")
    print(f"  Best achieved: {best_heuristic_performance:.1%}")
    
    # Save final model
    final_path = "models_advanced/double_dqn_final_advanced.pt"
    agent.save(final_path)
    print(f"💾 Final advanced model: {final_path}")
    
    # Comprehensive training summary
    summary = {
        'training_method': 'advanced_heuristic_preservation',
        'completed_at_episode': episode_num,
        'final_vs_random': final_vs_random,
        'final_vs_heuristic': final_vs_heuristic,
        'best_heuristic_performance': best_heuristic_performance,
        'heuristic_performance_history': heuristic_performance_history,
        'opponent_pool_size': len(opponent_pool),
        'config': config,
        'agent_stats': agent.get_stats(),
        'preservation_success': final_vs_heuristic >= config['heuristic_performance_threshold']
    }
    
    summary_file = f"{log_dir}/advanced_training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Comprehensive summary: {summary_file}")
    
    # Final training report with plots
    print(f"\n📊 Generating final comprehensive training report with visualizations...")
    monitor.generate_training_report(agent, episode_num)
    
    # Final assessment
    if final_vs_heuristic >= config['heuristic_performance_threshold']:
        print(f"\n🎉 SUCCESS: Aggressive heuristic preservation worked!")
        print(f"   Final performance: {final_vs_heuristic:.1%} (≥ {config['heuristic_performance_threshold']:.0%})")
        print(f"   📊 Check {log_dir}/ for comprehensive training plots and analysis")
    else:
        print(f"\n⚠️ PARTIAL SUCCESS: Some degradation occurred")
        print(f"   Final: {final_vs_heuristic:.1%}, Best: {best_heuristic_performance:.1%}")
        print(f"   📊 Check {log_dir}/ for failure analysis and debugging plots")
    
    print(f"✅ Advanced training completed with continual learning techniques and full monitoring!")


if __name__ == "__main__":
    train_advanced_double_dqn()