"""Comprehensive training monitoring system for Double DQN agent."""

import os
import json
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Connect4Board
from agents.double_dqn_agent import DoubleDQNAgent
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


class TrainingMonitor:
    """Monitor and evaluate Double DQN training progress."""
    
    def __init__(self, log_dir: str = "logs", save_plots: bool = True, eval_frequency: int = 250):
        """Initialize training monitor.
        
        Args:
            log_dir: Directory to save logs and plots
            save_plots: Whether to save visualization plots
            eval_frequency: How often evaluation occurs (for plotting scale)
        """
        self.log_dir = log_dir
        self.save_plots = save_plots
        self.eval_frequency = eval_frequency
        os.makedirs(log_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.win_rates = []  # vs Random (legacy, for compatibility)
        self.win_rates_vs_random = []
        self.win_rates_vs_heuristic = []
        self.win_rates_vs_league = []
        self.loss_values = []
        self.q_value_stats = []
        self.strategic_metrics = []
        self.exploration_rates = []
        self.training_steps = []
        
        # Performance tracking
        self.recent_wins = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=100)
        self.start_time = time.time()
        
        # Overall win rate tracking
        self.total_training_games = 0
        self.total_training_wins = 0
        self.overall_win_rate = 0.0
        
        # Opponent tracking
        self.current_opponent_name = "Unknown"
        self._show_curriculum_info = True  # Enable curriculum phase display
        
        # Strategic evaluation counters
        self.strategic_stats = {
            'winning_moves_played': 0,
            'winning_moves_missed': 0,
            'blocks_made': 0,
            'blocks_missed': 0,
            'center_plays': 0,
            'total_moves': 0
        }
        
        # Initialize CSV logging
        self._setup_csv_logging()
        
        # Fixed test states for Q-value analysis
        self.test_states = self._create_fixed_test_states()
        self._setup_q_value_logging()
    
    def _setup_csv_logging(self):
        """Setup CSV files for detailed logging."""
        # Main training log
        self.main_log_file = os.path.join(self.log_dir, "training_detailed.csv")
        if not os.path.exists(self.main_log_file):
            with open(self.main_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "reward", "eval_win_rate", "overall_win_rate", "epsilon", "buffer_size",
                    "training_steps", "avg_q_value", "q_value_std", 
                    "strategic_score", "time_elapsed"
                ])
        
        # Strategic analysis log
        self.strategic_log_file = os.path.join(self.log_dir, "strategic_analysis.csv")
        if not os.path.exists(self.strategic_log_file):
            with open(self.strategic_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "winning_moves_played", "winning_moves_missed",
                    "blocks_made", "blocks_missed", "center_plays", "total_moves",
                    "strategic_accuracy"
                ])
    
    def _create_fixed_test_states(self):
        """Create 4 strategically interesting test states for Q-value analysis."""
        from game.board import Connect4Board
        
        test_states = []
        
        # State 1: Empty board
        empty_board = Connect4Board()
        test_states.append({
            'name': 'Empty_Board',
            'description': 'Empty board: Tests opening strategy preferences',
            'board': empty_board.get_state().copy(),
            'visual': self._board_to_string(empty_board.get_state())
        })
        
        # State 2: Mid-game complex position
        # . . . . . . .
        # . . . . . . .
        # . . . . . . .
        # . . x o . . .
        # . x o o . . .
        # x o o x . . .
        mid_game = Connect4Board()
        # Bottom row: x o o x . . .
        mid_game.board[5, 0] = 1  # x (agent)
        mid_game.board[5, 1] = 2  # o (opponent)
        mid_game.board[5, 2] = 2  # o
        mid_game.board[5, 3] = 1  # x
        # Second row: . x o o . . .  
        mid_game.board[4, 1] = 1  # x
        mid_game.board[4, 2] = 2  # o
        mid_game.board[4, 3] = 2  # o
        # Third row: . . x o . . .
        mid_game.board[3, 2] = 1  # x
        mid_game.board[3, 3] = 2  # o
        
        test_states.append({
            'name': 'Mid_Game_Complex',
            'description': 'Mid-game complex: Mixed tactical opportunities',
            'board': mid_game.get_state().copy(),
            'visual': self._board_to_string(mid_game.get_state())
        })
        
        # State 3: Late-game threats
        # . . . . . . .
        # . . . . . . .
        # . . . . . . o
        # . . . . . o x
        # . . o . . x o
        # o x o o o x x
        late_game = Connect4Board()
        # Bottom row: o x o o o x x
        late_game.board[5, 0] = 2  # o
        late_game.board[5, 1] = 1  # x  
        late_game.board[5, 2] = 2  # o
        late_game.board[5, 3] = 2  # o
        late_game.board[5, 4] = 2  # o
        late_game.board[5, 5] = 1  # x
        late_game.board[5, 6] = 1  # x
        # Second row: . . o . . x o
        late_game.board[4, 2] = 2  # o
        late_game.board[4, 5] = 1  # x
        late_game.board[4, 6] = 2  # o
        # Third row: . . . . . o x  
        late_game.board[3, 5] = 2  # o
        late_game.board[3, 6] = 1  # x
        # Fourth row: . . . . . . o
        late_game.board[2, 6] = 2  # o
        
        test_states.append({
            'name': 'Late_Game_Threats',
            'description': 'Late-game threats: Multiple threat management',
            'board': late_game.get_state().copy(),
            'visual': self._board_to_string(late_game.get_state())
        })
        
        # State 4: Critical blocking situation  
        # . . . . . . .
        # . . . . . . .
        # . . . . x . .
        # . . . x o . .
        # . . . x o . .
        # o x o x o x .
        critical_block = Connect4Board()
        # Bottom row: o x o x o x .
        critical_block.board[5, 0] = 2  # o
        critical_block.board[5, 1] = 1  # x
        critical_block.board[5, 2] = 2  # o  
        critical_block.board[5, 3] = 1  # x
        critical_block.board[5, 4] = 2  # o
        critical_block.board[5, 5] = 1  # x
        # Second row: . . . x o . .
        critical_block.board[4, 3] = 1  # x
        critical_block.board[4, 4] = 2  # o
        # Third row: . . . x o . .  
        critical_block.board[3, 3] = 1  # x
        critical_block.board[3, 4] = 2  # o
        # Fourth row: . . . . x . .
        critical_block.board[2, 4] = 1  # x
        
        test_states.append({
            'name': 'Critical_Block',
            'description': 'Critical block needed: Agent must prevent opponent vertical win',
            'board': critical_block.get_state().copy(),
            'visual': self._board_to_string(critical_block.get_state())
        })
        
        return test_states
    
    def set_current_opponent(self, opponent_name: str):
        """Update the current opponent name for progress tracking."""
        self.current_opponent_name = opponent_name
    
    def _board_to_string(self, board_state):
        """Convert board state to string representation."""
        lines = []
        lines.append("  0 1 2 3 4 5 6")
        lines.append("  - - - - - - -")
        for i in range(6):
            row = "  "
            for j in range(7):
                if board_state[i, j] == 0:
                    row += ". "
                elif board_state[i, j] == 1:
                    row += "X "
                else:
                    row += "O "
            lines.append(row)
        return "\n".join(lines)
    
    def _setup_q_value_logging(self):
        """Setup CSV file for Q-value logging."""
        self.q_value_log_file = os.path.join(self.log_dir, "q_values_analysis.csv")
        
        if not os.path.exists(self.q_value_log_file):
            with open(self.q_value_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header with state descriptions
                header = ["episode"]
                for state in self.test_states:
                    # Add state description
                    header.append(f"{state['name']}_description")
                    # Add Q-values for each action
                    for action in range(7):
                        header.append(f"{state['name']}_q_raw_col_{action}")
                        header.append(f"{state['name']}_q_norm_col_{action}")
                
                writer.writerow(header)
                
                # Write state descriptions in second row
                desc_row = ["STATE_DESCRIPTIONS"]
                for state in self.test_states:
                    desc_row.append(state['description'])
                    # Add empty cells for Q-value columns
                    desc_row.extend([''] * 14)  # 7 actions * 2 (raw + norm)
                writer.writerow(desc_row)
                
                # Write visual representations
                for row_idx in range(8):  # Max 8 lines for board visual
                    visual_row = [f"VISUAL_ROW_{row_idx}"]
                    for state in self.test_states:
                        visual_lines = state['visual'].split('\n')
                        if row_idx < len(visual_lines):
                            visual_row.append(visual_lines[row_idx])
                        else:
                            visual_row.append('')
                        visual_row.extend([''] * 14)  # Empty Q-value columns
                    writer.writerow(visual_row)
    
    def log_episode(self, episode: int, reward: float, agent: DoubleDQNAgent,
                   win_rate: float = None, q_values: np.ndarray = None,
                   collect_q_values: bool = True, strategic_score: float = None,
                   win_rate_vs_heuristic: float = None, win_rate_vs_league: float = None):
        """Log metrics for a single episode.

        Args:
            episode: Episode number
            reward: Episode reward
            agent: The training agent
            win_rate: Current win rate vs random (if available)
            q_values: Q-values from recent decisions
            collect_q_values: Whether to collect Q-values for visualization
            strategic_score: Strategic score
            win_rate_vs_heuristic: Win rate vs heuristic opponent
            win_rate_vs_league: Win rate vs league champion
        """
        # Store basic metrics
        self.episode_rewards.append(reward)
        self.recent_rewards.append(reward)

        # Track overall win rate from all training games
        self.total_training_games += 1
        if reward > 0:  # Win is indicated by positive reward
            self.total_training_wins += 1
        self.overall_win_rate = self.total_training_wins / self.total_training_games

        # Track recent wins for display
        self.recent_wins.append(1 if reward > 0 else 0)

        # Track all win rates separately
        if win_rate is not None:
            self.win_rates.append(win_rate)  # Legacy
            self.win_rates_vs_random.append(win_rate)

        if win_rate_vs_heuristic is not None:
            self.win_rates_vs_heuristic.append(win_rate_vs_heuristic)

        if win_rate_vs_league is not None:
            self.win_rates_vs_league.append(win_rate_vs_league)
        
        # Agent statistics
        stats = agent.get_stats()
        self.exploration_rates.append(stats['epsilon'])
        self.training_steps.append(stats['training_steps'])
        
        # Q-value analysis for fixed test states (every episode during evaluation)
        if collect_q_values and (episode % 100 == 0 or win_rate is not None):
            q_values_data = self._collect_fixed_state_q_values(agent)
            self._log_q_values_to_csv(episode, q_values_data)
            
            # Store for visualization
            if q_values_data:
                all_q_values = np.concatenate([data['q_values'] for data in q_values_data])
                q_mean = np.mean(all_q_values)
                q_std = np.std(all_q_values)
                self.q_value_stats.append({
                    'mean': q_mean, 
                    'std': q_std, 
                    'episode': episode,
                    'states_data': q_values_data
                })
            else:
                q_mean = q_std = 0.0
        else:
            q_mean = q_std = 0.0
        
        # Strategic score - use passed value if available, otherwise calculate
        if strategic_score is not None:
            final_strategic_score = strategic_score
        else:
            final_strategic_score = self._calculate_strategic_score()
        self.strategic_metrics.append(final_strategic_score)
        
        # Time tracking
        time_elapsed = time.time() - self.start_time
        
        # Log to CSV
        with open(self.main_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, reward, win_rate or 0.0, self.overall_win_rate, stats['epsilon'],
                stats['buffer_size'], stats['training_steps'],
                q_mean, q_std, final_strategic_score, time_elapsed
            ])
        
        # Print progress (only if not disabled)
        if episode % 100 == 0 and getattr(self, '_show_progress', True):
            self._print_progress(episode, stats, win_rate, final_strategic_score)
    
    def analyze_strategic_play(self, prev_board: Connect4Board, action: int,
                             new_board: Connect4Board, agent_id: int):
        """Analyze strategic quality of a move.
        
        Args:
            prev_board: Board before move
            action: Action taken
            new_board: Board after move  
            agent_id: Agent's player ID
        """
        self.strategic_stats['total_moves'] += 1
        
        # Check if winning move was available and played
        winning_moves = prev_board.find_winning_moves(agent_id)
        if winning_moves:
            if action in winning_moves:
                self.strategic_stats['winning_moves_played'] += 1
            else:
                self.strategic_stats['winning_moves_missed'] += 1
        
        # Check if blocking move was needed and made
        blocking_moves = prev_board.find_blocking_moves(agent_id)
        if blocking_moves:
            if action in blocking_moves:
                self.strategic_stats['blocks_made'] += 1
            else:
                self.strategic_stats['blocks_missed'] += 1
        
        # Track center play preference
        if action in [2, 3, 4]:
            self.strategic_stats['center_plays'] += 1
    
    def _calculate_strategic_score(self) -> float:
        """Calculate overall strategic performance score."""
        stats = self.strategic_stats
        
        if stats['total_moves'] == 0:
            return 0.0
        
        # Calculate accuracy metrics
        total_critical = (stats['winning_moves_played'] + stats['winning_moves_missed'] +
                         stats['blocks_made'] + stats['blocks_missed'])
        
        if total_critical == 0:
            return 0.5  # Neutral score
        
        correct_critical = stats['winning_moves_played'] + stats['blocks_made']
        strategic_accuracy = correct_critical / total_critical
        
        # Bonus for center preference (should be moderate, around 30-50%)
        center_rate = stats['center_plays'] / stats['total_moves']
        center_bonus = 0.1 if 0.3 <= center_rate <= 0.5 else 0.0
        
        return strategic_accuracy + center_bonus
    
    def _collect_fixed_state_q_values(self, agent):
        """Collect Q-values from agent for the 4 fixed test states."""
        import torch
        
        q_values_data = []
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # No exploration for Q-value collection
        
        try:
            for state_info in self.test_states:
                board_state = state_info['board']
                encoded_state = agent.encode_state(board_state)
                state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(agent.device)
                
                with torch.no_grad():
                    q_vals = agent.online_net(state_tensor).cpu().numpy()[0]
                
                # Normalize Q-values (min-max normalization)
                q_min, q_max = q_vals.min(), q_vals.max()
                if q_max > q_min:
                    q_vals_normalized = (q_vals - q_min) / (q_max - q_min)
                else:
                    q_vals_normalized = np.zeros_like(q_vals)
                
                q_values_data.append({
                    'name': state_info['name'],
                    'description': state_info['description'],
                    'q_values': q_vals,
                    'q_values_normalized': q_vals_normalized,
                    'board_state': board_state
                })
        
        finally:
            agent.epsilon = old_epsilon
        
        return q_values_data
    
    def _log_q_values_to_csv(self, episode, q_values_data):
        """Log Q-values to CSV file."""
        if not q_values_data:
            return
            
        with open(self.q_value_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            row = [episode]
            for data in q_values_data:
                row.append(data['description'])
                # Add raw and normalized Q-values for each action
                for action in range(7):
                    row.append(f"{data['q_values'][action]:.6f}")
                    row.append(f"{data['q_values_normalized'][action]:.6f}")
            
            writer.writerow(row)
    
    def _print_progress(self, episode: int, agent_stats: dict, win_rate: float, 
                       strategic_score: float):
        """Print training progress summary."""
        recent_win_rate = np.mean(list(self.recent_wins)) if self.recent_wins else 0.0
        recent_avg_reward = np.mean(list(self.recent_rewards))
        
        print(f"\n=== Episode {episode:,} Progress ===")
        print(f"Current Opponent: {self.current_opponent_name}")
        print(f"Win Rate (recent/overall): {recent_win_rate:.1%} / {self.overall_win_rate:.1%}")
        print(f"Evaluation Win Rate: {win_rate:.1%}" if win_rate is not None else "Evaluation Win Rate: N/A")
        print(f"Avg Reward (recent): {recent_avg_reward:.2f}")
        print(f"Strategic Score: {strategic_score:.3f}")
        print(f"Epsilon: {agent_stats['epsilon']:.4f}")
        print(f"Buffer Size: {agent_stats['buffer_size']:,}")
        print(f"Training Steps: {agent_stats['training_steps']:,}")
        print(f"Total Training Games: {self.total_training_games:,}")
        
        # Add curriculum phase info for context (use correct thresholds for fixed training)
        if hasattr(self, '_show_curriculum_info'):
            if episode <= 8000:
                phase_info = f"📚 Curriculum Phase 1/3: Learning basics vs {self.current_opponent_name} (until ep 8,000)"
            elif episode <= 35000:
                phase_info = f"📚 Curriculum Phase 2/3: Strategic learning vs {self.current_opponent_name} (until ep 35,000)"
            else:
                phase_info = f"📚 Curriculum Phase 3/3: Advanced tactics vs {self.current_opponent_name}"
            print(f"{phase_info}")
        
        # Strategic breakdown
        stats = self.strategic_stats
        if stats['total_moves'] > 0:
            print(f"\nStrategic Analysis:")
            print(f"  Winning moves: {stats['winning_moves_played']}/{stats['winning_moves_played']+stats['winning_moves_missed']}")
            print(f"  Blocks made: {stats['blocks_made']}/{stats['blocks_made']+stats['blocks_missed']}")
            print(f"  Center preference: {stats['center_plays']/stats['total_moves']:.1%}")
    
    def log_strategic_episode(self, episode: int):
        """Log strategic metrics for an episode."""
        stats = self.strategic_stats
        strategic_accuracy = self._calculate_strategic_score()
        
        with open(self.strategic_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, stats['winning_moves_played'], stats['winning_moves_missed'],
                stats['blocks_made'], stats['blocks_missed'], stats['center_plays'],
                stats['total_moves'], strategic_accuracy
            ])
    
    def evaluate_agent_skills(self, agent: DoubleDQNAgent, num_games: int = 50) -> Dict:
        """Comprehensive skill evaluation against different opponents.
        
        Args:
            agent: Agent to evaluate
            num_games: Number of games per opponent type
            
        Returns:
            Dictionary with detailed skill metrics
        """
        print(f"\n🔍 Evaluating agent skills over {num_games} games per opponent...")
        
        results = {}
        
        # Test against random agent
        random_opponent = RandomAgent(player_id=2, seed=123)
        random_wins, random_metrics = self._evaluate_vs_opponent(
            agent, random_opponent, num_games, "Random"
        )
        results['vs_random'] = {
            'win_rate': random_wins,
            'metrics': random_metrics
        }
        
        # Test against heuristic agent
        heuristic_opponent = HeuristicAgent(player_id=2, seed=123)
        heuristic_wins, heuristic_metrics = self._evaluate_vs_opponent(
            agent, heuristic_opponent, num_games, "Heuristic"
        )
        results['vs_heuristic'] = {
            'win_rate': heuristic_wins,
            'metrics': heuristic_metrics
        }
        
        # Overall skill assessment
        overall_score = (random_wins * 0.3 + heuristic_wins * 0.7)  # Weight strategic play higher
        results['overall_skill'] = overall_score
        
        print(f"\n📊 Skill Evaluation Results:")
        print(f"  vs Random: {random_wins:.1%}")
        print(f"  vs Heuristic: {heuristic_wins:.1%}")
        print(f"  Overall Skill Score: {overall_score:.1%}")
        
        return results
    
    def _evaluate_vs_opponent(self, agent: DoubleDQNAgent, opponent, num_games: int, 
                            opponent_name: str) -> Tuple[float, Dict]:
        """Evaluate agent against specific opponent."""
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # No exploration during evaluation
        
        wins = 0
        game_lengths = []
        strategic_moves = 0
        total_moves = 0
        
        try:
            for _ in range(num_games):
                board = Connect4Board()
                players = [agent, opponent] if agent.player_id == 1 else [opponent, agent]
                current_idx = 0
                moves_this_game = 0
                
                while not board.is_terminal() and moves_this_game < 42:
                    current_player = players[current_idx]
                    prev_board = board.copy()
                    
                    legal_moves = board.get_legal_moves()
                    action = current_player.choose_action(board.get_state(), legal_moves)
                    board.make_move(action, current_player.player_id)
                    
                    # Track strategic moves for agent
                    if current_player == agent:
                        total_moves += 1
                        if self._is_strategic_move(prev_board, action, agent.player_id):
                            strategic_moves += 1
                    
                    current_idx = 1 - current_idx
                    moves_this_game += 1
                
                winner = board.check_winner()
                if winner == agent.player_id:
                    wins += 1
                game_lengths.append(moves_this_game)
        
        finally:
            agent.epsilon = old_epsilon
        
        win_rate = wins / num_games
        metrics = {
            'avg_game_length': np.mean(game_lengths),
            'strategic_move_rate': strategic_moves / max(total_moves, 1)
        }
        
        return win_rate, metrics
    
    def _is_strategic_move(self, board: Connect4Board, action: int, player_id: int) -> bool:
        """Check if a move is strategically sound."""
        # Winning move
        if action in board.find_winning_moves(player_id):
            return True
        
        # Blocking move
        if action in board.find_blocking_moves(player_id):
            return True
        
        # Center preference (columns 2, 3, 4)
        if action in [2, 3, 4]:
            return True
        
        return False
    
    def generate_training_report(self, agent: DoubleDQNAgent, episode: int):
        """Generate comprehensive training progress report."""
        # Only print report if progress printing is enabled
        if getattr(self, '_show_progress', True):
            print(f"\n📈 TRAINING REPORT - Episode {episode}")
            print("=" * 50)
            
            if len(self.episode_rewards) < 10:
                print("Not enough data for comprehensive report.")
                return
            
            # Recent performance (last 100 episodes)
            recent_episodes = min(100, len(self.episode_rewards))
            recent_rewards = self.episode_rewards[-recent_episodes:]
            recent_win_rate = sum(1 for r in recent_rewards if r > 0) / recent_episodes
            
            print(f"Recent Performance ({recent_episodes} episodes):")
            print(f"  Win Rate: {recent_win_rate:.1%}")
            print(f"  Avg Reward: {np.mean(recent_rewards):.2f}")
            print(f"  Reward Std: {np.std(recent_rewards):.2f}")
            
            # Learning progress
            stats = agent.get_stats()
            print(f"\nLearning Progress:")
            print(f"  Training Steps: {stats['training_steps']:,}")
            print(f"  Experience Buffer: {stats['buffer_size']:,}")
            print(f"  Exploration Rate: {stats['epsilon']:.4f}")
            
            # Strategic analysis
            strategic_score = self._calculate_strategic_score()
            print(f"\nStrategic Analysis:")
            print(f"  Strategic Score: {strategic_score:.3f}")
            self._print_strategic_breakdown()
            
            # Time analysis
            time_elapsed = time.time() - self.start_time
            episodes_per_hour = episode / (time_elapsed / 3600)
            print(f"\nTraining Efficiency:")
            print(f"  Time Elapsed: {time_elapsed/60:.1f} minutes")
            print(f"  Episodes/Hour: {episodes_per_hour:.1f}")
        
        # Always save visualization if enabled (regardless of progress printing setting)
        if self.save_plots and episode % 1000 == 0:
            self._generate_plots(episode, agent)
    
    def _print_strategic_breakdown(self):
        """Print detailed strategic performance breakdown."""
        stats = self.strategic_stats
        if stats['total_moves'] == 0:
            return
        
        print(f"  Total Moves Analyzed: {stats['total_moves']:,}")
        
        # Winning moves
        total_win_opportunities = stats['winning_moves_played'] + stats['winning_moves_missed']
        if total_win_opportunities > 0:
            win_accuracy = stats['winning_moves_played'] / total_win_opportunities
            print(f"  Winning Move Accuracy: {win_accuracy:.1%} ({stats['winning_moves_played']}/{total_win_opportunities})")
        
        # Blocking moves  
        total_block_opportunities = stats['blocks_made'] + stats['blocks_missed']
        if total_block_opportunities > 0:
            block_accuracy = stats['blocks_made'] / total_block_opportunities
            print(f"  Blocking Accuracy: {block_accuracy:.1%} ({stats['blocks_made']}/{total_block_opportunities})")
        
        # Center preference
        center_rate = stats['center_plays'] / stats['total_moves']
        print(f"  Center Play Rate: {center_rate:.1%}")
    
    def _generate_plots(self, episode: int, agent: DoubleDQNAgent = None):
        """Generate training visualization plots with strategic Q-value heatmaps."""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)
            
            fig.suptitle(f'Double DQN Training Progress - Episode {episode}', fontsize=18)
            
            # Plot 1: Reward progression
            ax1 = fig.add_subplot(gs[0, 0])
            if self.episode_rewards and len(self.episode_rewards) > 0:
                window = min(100, max(1, len(self.episode_rewards) // 10))
                if len(self.episode_rewards) >= window and window > 0:
                    smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                    ax1.plot(smoothed, label='Smoothed Rewards')
                ax1.plot(self.episode_rewards, alpha=0.3, label='Raw Rewards')
                ax1.set_title('Episode Rewards')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Reward')
                ax1.legend()
                ax1.grid(True)
            else:
                ax1.text(0.5, 0.5, 'No reward data yet', 
                        transform=ax1.transAxes, ha='center', va='center')
                ax1.set_title('Episode Rewards')
            
            # Plot 2: All Win Rates (vs Random, Heuristic, League)
            ax2 = fig.add_subplot(gs[0, 1])
            if (self.win_rates_vs_random or self.win_rates_vs_heuristic or self.win_rates_vs_league):
                try:
                    eval_frequency = getattr(self, 'eval_frequency', 250)

                    # Plot vs Random (blue)
                    if self.win_rates_vs_random and len(self.win_rates_vs_random) > 0:
                        episodes = np.arange(1, len(self.win_rates_vs_random) + 1) * eval_frequency
                        ax2.plot(episodes, self.win_rates_vs_random, 'b-',
                                label='vs Random', linewidth=2, marker='o', markersize=3, alpha=0.7)

                    # Plot vs Heuristic (red)
                    if self.win_rates_vs_heuristic and len(self.win_rates_vs_heuristic) > 0:
                        episodes = np.arange(1, len(self.win_rates_vs_heuristic) + 1) * eval_frequency
                        ax2.plot(episodes, self.win_rates_vs_heuristic, 'r-',
                                label='vs Heuristic', linewidth=2, marker='s', markersize=3, alpha=0.7)

                    # Plot vs League (green)
                    if self.win_rates_vs_league and len(self.win_rates_vs_league) > 0:
                        episodes = np.arange(1, len(self.win_rates_vs_league) + 1) * eval_frequency
                        ax2.plot(episodes, self.win_rates_vs_league, 'g-',
                                label='vs League', linewidth=2, marker='^', markersize=3, alpha=0.7)

                    # Add reference lines
                    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.4, label='50% baseline')
                    ax2.axhline(y=0.3, color='orange', linestyle='--', linewidth=1, alpha=0.4, label='30% critical')

                    ax2.set_xlabel('Episode')
                    ax2.set_ylabel('Win Rate')
                    ax2.set_title('Win Rates: All Opponents')
                    ax2.set_ylim(-0.05, 1.05)
                    ax2.legend(loc='best', fontsize=8)
                    ax2.grid(True, alpha=0.3)

                except Exception as e:
                    print(f"Warning: Could not plot win rates: {e}")
                    ax2.text(0.5, 0.5, f'Plot Error: {str(e)}', transform=ax2.transAxes, ha='center')
            else:
                ax2.text(0.5, 0.5, 'No evaluation data yet',
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title('Win Rates: All Opponents')
            
            # Plot 3: Strategic metrics
            ax3 = fig.add_subplot(gs[0, 2:])
            if self.strategic_metrics and len(self.strategic_metrics) > 0:
                ax3.plot(self.strategic_metrics, 'g-')
                ax3.set_title('Strategic Score Progress')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Strategic Score')
                ax3.grid(True)
            else:
                ax3.text(0.5, 0.5, 'No strategic data yet', 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Strategic Score Progress')
            
            # Plot 4-7: Strategic Q-value heatmaps (if agent is provided)
            if agent is not None:
                self._plot_q_value_heatmaps(fig, gs, episode, agent)
            
            # Plot: Q-value distribution over time (bottom row)
            ax_qval_dist = fig.add_subplot(gs[2, :])
            if self.q_value_stats:
                self._plot_q_value_distribution(ax_qval_dist)
            
            # Use bbox_inches='tight' instead of tight_layout to avoid warnings
            plot_path = os.path.join(self.log_dir, f'training_progress_ep_{episode}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
            plt.close()
            
            print(f"📊 Training plots with fixed-state Q-value heatmaps saved to {plot_path}")
            
        except Exception as e:
            import traceback
            print(f"Warning: Could not generate plots: {e}")
            print("Full traceback:")
            traceback.print_exc()
    
    def _plot_legacy_removed(self):
        """Legacy Q-value heatmap plotting removed - using new board-style visualization."""
        pass
    
    def _plot_q_value_heatmaps(self, fig, gs, episode, agent):
        """Plot Q-value evolution heatmaps over time for each test state."""
        import torch
        
        # Collect current Q-values for this episode
        old_epsilon = agent.epsilon
        agent.epsilon = 0.0  # No exploration
        
        # Store Q-values for each state at this episode  
        current_episode_qvalues = []
        
        try:
            for state_info in self.test_states:
                board_state = state_info['board']
                encoded_state = agent.encode_state(board_state)
                state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(agent.device)
                
                with torch.no_grad():
                    q_values = agent.online_net(state_tensor).cpu().numpy()[0]
                
                current_episode_qvalues.append(q_values)
        
        finally:
            agent.epsilon = old_epsilon
        
        # Add current episode data to the stored Q-value history
        if not hasattr(self, 'qvalue_history'):
            self.qvalue_history = {state_info['name']: [] for state_info in self.test_states}
            self.qvalue_episodes = []
        
        self.qvalue_episodes.append(episode)
        for i, state_info in enumerate(self.test_states):
            self.qvalue_history[state_info['name']].append(current_episode_qvalues[i])
        
        # Plot heatmaps for each state showing evolution over episodes
        for idx, state_info in enumerate(self.test_states):
            ax = fig.add_subplot(gs[1, idx])
            state_name = state_info['name']
            
            if len(self.qvalue_history[state_name]) < 2:
                # Not enough data yet
                ax.text(0.5, 0.5, f'{state_name.replace("_", " ")}\\nNot enough data', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{state_name.replace("_", " ")}')
                continue
            
            # Create heatmap data: episodes × actions (7 columns)
            qval_matrix = np.array(self.qvalue_history[state_name])  # Shape: (episodes, 7)

            # Normalize each row to sum to 1 (convert to probability distribution)
            # Use softmax for better numerical stability
            qval_matrix_exp = np.exp(qval_matrix - np.max(qval_matrix, axis=1, keepdims=True))
            qval_matrix_normalized = qval_matrix_exp / np.sum(qval_matrix_exp, axis=1, keepdims=True)

            # Create heatmap with normalized values
            im = ax.imshow(qval_matrix_normalized, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
            
            # Set labels
            ax.set_title(f'{state_name.replace("_", " ")}\n(Softmax Normalized)', fontsize=10, fontweight='bold')
            ax.set_xlabel('Action (Column)', fontsize=8)
            ax.set_ylabel('Training Episode', fontsize=8)
            
            # Set x-axis labels (actions/columns)
            ax.set_xticks(range(7))
            ax.set_xticklabels([f'Col {i}' for i in range(7)], fontsize=8)
            
            # Set y-axis labels (episodes) - show subset
            if len(self.qvalue_episodes) > 1:
                # Show episode numbers, but limit to reasonable number of ticks
                y_ticks = np.arange(0, len(self.qvalue_episodes), max(1, len(self.qvalue_episodes)//5))
                ax.set_yticks(y_ticks)
                ax.set_yticklabels([f'{self.qvalue_episodes[i]}' for i in y_ticks], fontsize=8)
            
            # Add colorbar
            try:
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('Action Probability', fontsize=8)
            except Exception as e:
                print(f"Warning: Could not add colorbar for {state_name}: {e}")
            
            # Check for learning issues in latest episode
            latest_qvalues = current_episode_qvalues[idx]
            if np.allclose(latest_qvalues, latest_qvalues[0], atol=1e-6):
                ax.text(0.02, 0.98, '⚠️ IDENTICAL Q-VALUES!', 
                       transform=ax.transAxes, va='top', ha='left',
                       fontsize=8, color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            else:
                best_action = latest_qvalues.argmax()
                ax.text(0.02, 0.98, f'Latest Best: Col {best_action}', 
                       transform=ax.transAxes, va='top', ha='left',
                       fontsize=8, color='green', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    def _plot_q_value_distribution(self, ax):
        """Plot Q-value distribution over time."""
        if not self.q_value_stats:
            ax.text(0.5, 0.5, 'No Q-value data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Q-Value Statistics Over Training')
            return
            
        episodes = []
        q_means = []
        q_stds = []
        
        for q_stat in self.q_value_stats:
            episodes.append(q_stat['episode'])  # Use actual episode number
            q_means.append(q_stat['mean'])
            q_stds.append(q_stat['std'])
        
        if episodes:
            q_means = np.array(q_means)
            q_stds = np.array(q_stds)
            
            # Plot mean with error bars
            ax.plot(episodes, q_means, 'b-', label='Mean Q-Value', linewidth=2)
            ax.fill_between(episodes, q_means - q_stds, q_means + q_stds, 
                           alpha=0.3, label='±1 Std Dev')
            
            ax.set_title('Q-Value Statistics Over Training')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Q-Value')
            ax.legend()
            ax.grid(True)
    
    def set_current_opponent(self, opponent_name: str):
        """Set the current opponent name for tracking."""
        self.current_opponent_name = opponent_name
    
    def reset_strategic_stats(self):
        """Reset strategic statistics counters."""
        self.strategic_stats = {
            'winning_moves_played': 0,
            'winning_moves_missed': 0,
            'blocks_made': 0,
            'blocks_missed': 0,
            'center_plays': 0,
            'total_moves': 0
        }


if __name__ == "__main__":
    # Test the monitoring system
    print("Testing Training Monitor...")
    
    monitor = TrainingMonitor(log_dir="test_logs")
    agent = DoubleDQNAgent(player_id=1, seed=42)
    
    # Simulate some training episodes
    for ep in range(1, 11):
        # Simulate episode reward
        reward = np.random.choice([10, -10, 1], p=[0.4, 0.5, 0.1])
        
        # Simulate some strategic moves
        board = Connect4Board()
        board.board[5, 3] = 2  # Setup threat
        monitor.analyze_strategic_play(board.copy(), 3, board, 1)  # Block
        
        # Log episode
        monitor.log_episode(ep, reward, agent, win_rate=0.4)
        
        if ep % 5 == 0:
            monitor.log_strategic_episode(ep)
            monitor.generate_training_report(agent, ep)
    
    print("✓ Training monitor test completed!")