#!/usr/bin/env python3
"""
Resume M1 CNN training with ENHANCED features:
1. League-based self-play (progressive training against past versions)
2. Ensemble opponents (strongest strategy from tournament)
3. Stochastic evaluation (breaks determinism)
"""

import os
import sys
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
from train.league_manager import LeagueManager, LeagueAgent
from ensemble_agent import EnsembleAgent


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
        return 300000

    # Extract episode number from filename
    try:
        if "_ep_" in filename:
            episode_str = filename.split("_ep_")[-1].split(".")[0]
            return int(episode_str)
    except (ValueError, IndexError):
        pass

    return 0


def create_ensemble_opponent(league: LeagueManager, player_id: int = 2) -> Optional[EnsembleAgent]:
    """
    Create ensemble opponent from top league models.

    Uses Q-value averaging strategy (proven best in tournament).
    """
    if len(league.league_models) < 3:
        return None

    # Select top 3-4 models from league
    num_models = min(4, len(league.league_models))
    top_models = league.league_models[-num_models:]  # Most recent/strongest

    # Create model configs for ensemble
    model_configs = []
    for i, model_info in enumerate(top_models):
        # Weight: favor more recent models
        weight = 0.4 if i == len(top_models) - 1 else 0.6 / (len(top_models) - 1)
        model_configs.append({
            'path': model_info['path'],
            'weight': weight,
            'name': f"League-Tier{model_info['tier']}"
        })

    try:
        ensemble = EnsembleAgent(
            model_configs,
            ensemble_method="q_value_averaging",  # Best performer in tournament
            player_id=player_id,
            name="League-Ensemble",
            show_contributions=False
        )
        return ensemble
    except Exception as e:
        print(f"⚠️ Failed to create ensemble opponent: {e}")
        return None


def resume_m1_cnn_with_league(target_episodes: int = 1000000, resume_model_path: str = None,
                               use_league: bool = True, use_ensemble: bool = True):
    """
    Resume M1 CNN training with enhanced features:
    - League-based progressive self-play
    - Ensemble opponents (strongest from tournament)
    - Curriculum learning with diverse opponents
    """

    print("🚀 M1 CNN TRAINING RESUME - ENHANCED WITH LEAGUE & ENSEMBLE")
    print("=" * 70)

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

    # Load configuration
    config = create_m1_cnn_config()
    config["num_episodes"] = remaining_episodes
    config["random_seed"] = 42

    print(f"\n🔧 ENHANCED TRAINING FEATURES:")
    if use_league:
        print(f"  1. ✅ League System: Progressive self-play vs past versions")
    if use_ensemble:
        print(f"  2. ✅ Ensemble Opponents: Tournament-proven strategy")
    print(f"  3. ✅ Stochastic Evaluation: Breaks determinism")
    print(f"  4. ✅ Diverse Curriculum: Random + Heuristic + League + Ensemble")

    # Set seeds
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

    # Setup logging
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

    # Initialize base opponents
    random_opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
    heuristic_opponent = HeuristicAgent(player_id=2, seed=config["random_seed"] + 2)

    # Initialize League Manager
    league = None
    if use_league:
        print(f"\n🏆 Initializing League Manager...")
        league = LeagueManager(
            model_dir="models_m1_cnn",
            league_size=5,
            win_threshold=0.60,
            min_games_to_promote=50,
            selection_strategy="progressive"
        )
        league.print_status()

    # Initialize Ensemble Opponent
    ensemble_opponent = None
    if use_ensemble and league and len(league.league_models) >= 3:
        print(f"\n🤖 Creating Ensemble Opponent from League...")
        ensemble_opponent = create_ensemble_opponent(league, player_id=2)
        if ensemble_opponent:
            print(f"✅ Ensemble opponent created: {ensemble_opponent.name}")

    # Training tracking
    episode_rewards = []
    best_win_rate = 0.0
    league_game_counter = 0

    # Adaptive heuristic preservation (prevent catastrophic forgetting)
    heuristic_boost_mode = False
    heuristic_boost_episodes_remaining = 0
    last_heuristic_performance = 1.0  # Track performance

    # Performance thresholds for safety
    HEURISTIC_CRITICAL_THRESHOLD = 0.30  # Trigger boost if below 30%
    HEURISTIC_WARNING_THRESHOLD = 0.40   # Warning if below 40%
    HEURISTIC_BOOST_DURATION = 5000      # Boost for 5k episodes

    # Curriculum phases
    WARMUP_PHASE_END = config["warmup_phase_end"]
    RANDOM_PHASE_END = config["random_phase_end"]
    HEURISTIC_PHASE_END = config["heuristic_phase_end"]
    LEAGUE_START = config["self_play_start"]

    def get_current_opponent(episode_absolute):
        """
        Enhanced opponent selection with league and ensemble.

        Distribution after LEAGUE_START (REBALANCED to prevent catastrophic forgetting):
        - Normal mode:
          - 30%: League opponent (progressive self-play)
          - 15%: Ensemble opponent (tournament strategy)
          - 40%: Heuristic (INCREASED - preserve fundamentals)
          - 15%: Random (maintain diversity)

        - Boost mode (triggered if vs Heuristic < 30%):
          - 10%: League opponent
          - 5%: Ensemble opponent
          - 70%: Heuristic (EMERGENCY - relearn fundamentals)
          - 15%: Random

        Note: Heuristic ratio increased from 25% to 40% to prevent overfitting
        to RL opponents and maintain fundamental Connect4 skills.
        """
        nonlocal league_game_counter, heuristic_boost_mode, heuristic_boost_episodes_remaining

        if episode_absolute <= WARMUP_PHASE_END:
            return random_opponent, "Random-Warmup", None
        elif episode_absolute <= RANDOM_PHASE_END:
            return random_opponent, "Random", None
        elif episode_absolute <= HEURISTIC_PHASE_END:
            return heuristic_opponent, "Heuristic", None
        else:
            # Check if in boost mode
            if heuristic_boost_mode and heuristic_boost_episodes_remaining > 0:
                heuristic_boost_episodes_remaining -= 1
                if heuristic_boost_episodes_remaining == 0:
                    heuristic_boost_mode = False
                    print(f"\n✅ Heuristic boost mode ended at episode {episode_absolute}")

            # Enhanced mixed phase with league and ensemble (REBALANCED)
            rand_val = np.random.random()

            # Adjust probabilities based on boost mode
            if heuristic_boost_mode:
                # BOOST MODE: 10% League, 5% Ensemble, 70% Heuristic, 15% Random
                league_threshold = 0.10
                ensemble_threshold = 0.15
                heuristic_threshold = 0.85
            else:
                # NORMAL MODE: 30% League, 15% Ensemble, 40% Heuristic, 15% Random
                league_threshold = 0.30
                ensemble_threshold = 0.45
                heuristic_threshold = 0.85

            if rand_val < league_threshold and league:
                # League opponent
                league_model = league.select_opponent(episode_absolute)
                if league_model:
                    league_game_counter += 1
                    league_agent = LeagueAgent(
                        league_model['path'],
                        CNNDuelingDQNAgent,
                        player_id=2,
                        epsilon=0.05
                    )
                    return league_agent, f"League-T{league_model['tier']}", league_model['tier']
                # Fallback to heuristic
                return heuristic_opponent, "Heuristic", None

            elif rand_val < ensemble_threshold and ensemble_opponent:
                # Ensemble opponent
                return ensemble_opponent, "Ensemble", None

            elif rand_val < heuristic_threshold:
                # Heuristic (preserve fundamentals!)
                return heuristic_opponent, "Heuristic", None

            else:
                # Random (diversity)
                return random_opponent, "Random", None

    print(f"\n🚀 Resuming M1 CNN training...")
    print(f"   Episodes {start_episode + 1:,} → {target_episodes:,}")
    print(f"   Expected time: ~{remaining_episodes // 150 // 60:.1f} hours")
    print(f"   Monitor: tail -f {log_file}")
    print()

    # Training loop
    current_opponent_name = ""

    for episode in range(remaining_episodes):
        episode_num = episode + 1
        absolute_episode = start_episode + episode_num

        # Get opponent
        current_opponent, opponent_name, league_tier = get_current_opponent(absolute_episode)

        # Update monitor (no phase change announcements - too noisy with mixed opponents)
        if opponent_name != current_opponent_name:
            current_opponent_name = opponent_name
            monitor.set_current_opponent(opponent_name)

        # Reset episode
        agent.reset_episode()

        # Training mode selection
        reward_config_to_use = None if absolute_episode < config["optimism_bootstrap_end"] else config["reward_config"]

        # Play training game
        winner, _, spatial_stats = play_m1_training_game(
            agent, current_opponent, reward_config_to_use, config
        )

        # Record league game result
        if league and league_tier is not None:
            agent_won = (winner == agent.player_id)
            league.record_game_result(league_tier, agent_won)

            # Check for tier promotion or backtracking
            if league_game_counter % 50 == 0:  # Check every 50 league games
                # First check if we should backtrack (stuck for too long)
                if league.backtrack_tier():
                    # Recreate ensemble with new league state
                    if use_ensemble:
                        ensemble_opponent = create_ensemble_opponent(league, player_id=2)
                # Otherwise check for promotion
                elif league.promote_tier():
                    # Recreate ensemble with new league state
                    if use_ensemble:
                        ensemble_opponent = create_ensemble_opponent(league, player_id=2)

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
            status_str = f"Episode {absolute_episode:,} | {opponent_name} | " \
                        f"Avg Reward: {avg_reward_recent:.1f} | ε: {agent.epsilon:.6f} | " \
                        f"Buffer: {len(agent.memory)} | Steps: {agent.train_step_count:,} | " \
                        f"Spatial: {spatial_score_for_monitor:.2f}"

            if league:
                league_status = league.get_league_status()
                status_str += f" | League Tier: {league_status['current_tier']}"

            print(status_str)

        # Periodic evaluation
        if episode_num % config["eval_frequency"] == 0:
            # Enhanced evaluation with stochastic mode
            old_epsilon = agent.epsilon
            agent.epsilon = 0.0

            try:
                wins_vs_random = 0
                wins_vs_heuristic = 0
                wins_vs_league = 0

                # Test against random (50 games)
                for _ in range(50):
                    test_board = Connect4Board()
                    if np.random.random() < 0.5:
                        players = [agent, random_opponent]
                    else:
                        players = [random_opponent, agent]

                    move_count = 0
                    while not test_board.is_terminal() and move_count < 42:
                        current_player = players[move_count % 2]

                        # Use stochastic action for agent (breaks determinism)
                        if current_player == agent:
                            action = agent.choose_action_stochastic(
                                test_board.get_state(),
                                test_board.get_legal_moves(),
                                temperature=0.1
                            )
                        else:
                            action = current_player.choose_action(test_board.get_state(), test_board.get_legal_moves())

                        test_board.make_move(action, current_player.player_id)
                        move_count += 1

                    if test_board.check_winner() == agent.player_id:
                        wins_vs_random += 1

                # Test against heuristic (50 games)
                for _ in range(50):
                    test_board = Connect4Board()
                    if np.random.random() < 0.5:
                        players = [agent, heuristic_opponent]
                    else:
                        players = [heuristic_opponent, agent]

                    move_count = 0
                    while not test_board.is_terminal() and move_count < 42:
                        current_player = players[move_count % 2]

                        if current_player == agent:
                            action = agent.choose_action_stochastic(
                                test_board.get_state(),
                                test_board.get_legal_moves(),
                                temperature=0.1
                            )
                        else:
                            action = current_player.choose_action(test_board.get_state(), test_board.get_legal_moves())

                        test_board.make_move(action, current_player.player_id)
                        move_count += 1

                    if test_board.check_winner() == agent.player_id:
                        wins_vs_heuristic += 1

                # Test against league champion (if available)
                if league and league.league_models:
                    champion = league.league_models[-1]
                    league_champ_agent = LeagueAgent(
                        champion['path'],
                        CNNDuelingDQNAgent,
                        player_id=2,
                        epsilon=0.0
                    )

                    for _ in range(30):
                        test_board = Connect4Board()
                        if np.random.random() < 0.5:
                            players = [agent, league_champ_agent]
                        else:
                            players = [league_champ_agent, agent]

                        move_count = 0
                        while not test_board.is_terminal() and move_count < 42:
                            current_player = players[move_count % 2]

                            if current_player == agent:
                                action = agent.choose_action_stochastic(
                                    test_board.get_state(),
                                    test_board.get_legal_moves(),
                                    temperature=0.1
                                )
                            else:
                                action = current_player.choose_action(test_board.get_state(), test_board.get_legal_moves())

                            test_board.make_move(action, current_player.player_id)
                            move_count += 1

                        if test_board.check_winner() == agent.player_id:
                            wins_vs_league += 1

                current_win_rate = wins_vs_random / 50
                vs_heuristic = wins_vs_heuristic / 50
                vs_league_champ = wins_vs_league / 30 if league else 0.0

            finally:
                agent.epsilon = old_epsilon

            # Monitor heuristic performance and trigger boost mode if needed
            if vs_heuristic < HEURISTIC_CRITICAL_THRESHOLD:
                if not heuristic_boost_mode:
                    heuristic_boost_mode = True
                    heuristic_boost_episodes_remaining = HEURISTIC_BOOST_DURATION
                    print(f"\n🚨 HEURISTIC BOOST MODE ACTIVATED at episode {absolute_episode}")
                    print(f"   Performance dropped to {vs_heuristic:.1%} (threshold: {HEURISTIC_CRITICAL_THRESHOLD:.1%})")
                    print(f"   Boosting heuristic training to 70% for next {HEURISTIC_BOOST_DURATION:,} episodes")
            elif vs_heuristic < HEURISTIC_WARNING_THRESHOLD:
                if not heuristic_boost_mode:
                    print(f"\n⚠️  Warning: Heuristic performance at {vs_heuristic:.1%} (threshold: {HEURISTIC_WARNING_THRESHOLD:.1%})")

            # Update boost mode status
            if heuristic_boost_mode:
                heuristic_boost_episodes_remaining -= config["eval_frequency"]
                if heuristic_boost_episodes_remaining <= 0:
                    heuristic_boost_mode = False
                    print(f"\n✅ HEURISTIC BOOST MODE DEACTIVATED at episode {absolute_episode}")
                    print(f"   Current performance: {vs_heuristic:.1%}")
                    print(f"   Returning to normal opponent distribution")

            # Store last heuristic performance
            last_heuristic_performance = vs_heuristic

            # Track best model using composite score (prioritize heuristic performance)
            # Composite score: 20% Random + 60% Heuristic + 20% League
            # This prevents saturation at 100% vs Random and rewards strategic improvement
            composite_score = (0.2 * current_win_rate) + (0.6 * vs_heuristic) + (0.2 * vs_league_champ)

            if composite_score > best_win_rate:
                best_win_rate = composite_score
                os.makedirs("models_m1_cnn", exist_ok=True)
                best_path = f"models_m1_cnn/m1_cnn_dqn_best_ep_{absolute_episode}.pt"
                agent.save(best_path)
                print(f"💎 M1 CNN Best: Composite {composite_score:.1%} " +
                      f"(R:{current_win_rate:.0%} H:{vs_heuristic:.0%} L:{vs_league_champ:.0%}) " +
                      f"- saved {os.path.basename(best_path)}")

                # Add to league if eligible
                if league:
                    league.add_checkpoint_to_league(best_path, absolute_episode)

            # Log results
            avg_reward = np.mean(episode_rewards[-config["eval_frequency"]:])
            spatial_score = spatial_stats.get('spatial_score', 0)

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    absolute_episode, avg_reward, agent.epsilon, agent.train_step_count,
                    len(agent.memory), current_win_rate, vs_heuristic, vs_league_champ,
                    spatial_score, "M1-CNN-League", total_params
                ])

            status_str = f"Episode {absolute_episode:,} | {opponent_name} | " \
                        f"Win: {current_win_rate:.1%} | vs Heur: {vs_heuristic:.1%}"

            if league:
                status_str += f" | vs League: {vs_league_champ:.1%}"
                league.print_status()

            status_str += f" | ε: {agent.epsilon:.6f} | Spatial: {spatial_score:.2f}"
            print(status_str)

            # Log for plotting (with all win rates)
            monitor.log_episode(absolute_episode, avg_reward, agent,
                              win_rate=current_win_rate,
                              win_rate_vs_heuristic=vs_heuristic,
                              win_rate_vs_league=vs_league_champ,
                              strategic_score=spatial_score)

            # Generate plots
            if absolute_episode % 2000 == 0:
                monitor.generate_training_report(agent, absolute_episode)

                # Also generate combined win rates plot
                try:
                    import sys
                    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from plot_win_rates import plot_all_win_rates
                    plot_output = os.path.join(log_dir, f"win_rates_ep_{absolute_episode}.png")
                    plot_all_win_rates(log_file, plot_output,
                                      title=f"All Win Rates - Episode {absolute_episode:,}")
                except Exception as e:
                    print(f"Warning: Could not generate win rates plot: {e}")

                print(f"📊 M1 CNN plots generated for episode {absolute_episode:,}")

        # Save checkpoints
        if episode_num % config["save_frequency"] == 0:
            os.makedirs("models_m1_cnn", exist_ok=True)
            checkpoint_path = f"models_m1_cnn/m1_cnn_dqn_ep_{absolute_episode}.pt"
            agent.save(checkpoint_path)

    # Final save
    print("\n" + "=" * 60)
    print("ENHANCED M1 CNN TRAINING COMPLETED")
    print("=" * 60)

    final_model_path = f"models_m1_cnn/m1_cnn_dqn_final_{target_episodes // 1000}k.pt"
    agent.save(final_model_path)

    print(f"\n✅ M1 CNN ENHANCED TRAINING COMPLETED!")
    print(f"Final model: {final_model_path}")
    print(f"Best performance: {best_win_rate:.1%}")
    print(f"Total episodes trained: {target_episodes:,}")
    print(f"Final epsilon: {agent.epsilon:.6f}")

    if league:
        print(f"\n🏆 Final League Status:")
        league.print_status()

    # Generate final report
    monitor.generate_training_report(agent, target_episodes)

    print("\n🎉 Enhanced M1 CNN training with League & Ensemble successful!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resume M1 CNN training with enhanced features")
    parser.add_argument("--target", type=int, default=1000000, help="Target total episodes (default: 1,000,000)")
    parser.add_argument("--model", type=str, default=None, help="Specific model path to resume from")
    parser.add_argument("--no-league", action="store_true", help="Disable league system")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable ensemble opponents")

    args = parser.parse_args()

    try:
        resume_m1_cnn_with_league(
            target_episodes=args.target,
            resume_model_path=args.model,
            use_league=not args.no_league,
            use_ensemble=not args.no_ensemble
        )
    except KeyboardInterrupt:
        print("\n⚠️ Enhanced training interrupted by user")
    except Exception as e:
        print(f"\n❌ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
