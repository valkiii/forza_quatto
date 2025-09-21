#!/usr/bin/env python3
"""Comprehensive simulation script for testing trained RL agent vs opponents."""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.board import Connect4Board
from agents.double_dqn_agent import DoubleDQNAgent
from agents.enhanced_double_dqn_agent import EnhancedDoubleDQNAgent
from agents.cnn_dueling_dqn_agent import CNNDuelingDQNAgent
from train_fixed_double_dqn import FixedDoubleDQNAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent


class GameSimulator:
    """Comprehensive game simulator for Connect 4 agents."""
    
    def __init__(self, rl_model_path: str):
        """Initialize the simulator with trained RL agent (auto-detects type)."""
        self.rl_agent = None
        self.rl_model_path = rl_model_path
        self.agent_type = None
        
        # Load RL agent with auto-detection
        if os.path.exists(rl_model_path):
            try:
                self.rl_agent = self._load_agent_auto_detect(rl_model_path)
                if self.rl_agent:
                    self.rl_agent.epsilon = 0.0  # No exploration during evaluation
                    print(f"✅ {self.agent_type} RL Agent loaded from {os.path.basename(rl_model_path)}")
                else:
                    print(f"❌ Failed to load RL agent from {rl_model_path}")
            except Exception as e:
                print(f"❌ Failed to load RL agent: {e}")
                self.rl_agent = None
        else:
            print(f"❌ Model file not found: {rl_model_path}")
    
    def _load_agent_auto_detect(self, model_path: str):
        """Auto-detect and load the appropriate agent type."""
        # Determine agent type from file path
        model_name = os.path.basename(model_path).lower()
        
        if "m1_cnn" in model_path.lower() or "cnn" in model_name:
            # M1 CNN or CNN model
            if "m1" in model_path.lower():
                architecture = "m1_optimized"
                hidden_size = 48
                self.agent_type = "M1-Optimized CNN"
            else:
                architecture = "ultra_light"
                hidden_size = 16
                self.agent_type = "Ultra-Light CNN"
                
            agent = CNNDuelingDQNAgent(
                player_id=1,  # Will be adjusted per game
                input_channels=2,
                action_size=7,
                hidden_size=hidden_size,
                architecture=architecture,
                seed=42
            )
            agent.load(model_path, keep_player_id=False)
            return agent
            
        elif "enhanced" in model_path.lower():
            # Enhanced Double DQN model
            self.agent_type = "Enhanced Double DQN"
            agent = EnhancedDoubleDQNAgent(
                player_id=1,
                state_size=92,  # Enhanced state with strategic features
                action_size=7,
                hidden_size=512,
                seed=42
            )
            agent.load(model_path, keep_player_id=False)
            return agent
            
        else:
            # Legacy Fixed Double DQN model
            self.agent_type = "Fixed Double DQN"
            agent = FixedDoubleDQNAgent(
                player_id=1,
                state_size=84,  # 2 channels * 6 rows * 7 cols
                action_size=7,
                seed=42,
                # CRITICAL: Match training configuration exactly
                gradient_clip_norm=1.0,
                use_huber_loss=True,
                huber_delta=1.0,
                state_normalization=True
            )
            agent.load(model_path, keep_player_id=False)
            return agent
    
    def _create_self_play_opponent(self):
        """Create a copy of the main agent for self-play."""
        if "M1-Optimized CNN" in self.agent_type:
            opponent = CNNDuelingDQNAgent(
                player_id=2,
                input_channels=2,
                action_size=7,
                hidden_size=48,
                architecture="m1_optimized",
                seed=456
            )
        elif "Ultra-Light CNN" in self.agent_type:
            opponent = CNNDuelingDQNAgent(
                player_id=2,
                input_channels=2,
                action_size=7,
                hidden_size=16,
                architecture="ultra_light",
                seed=456
            )
        elif "Enhanced Double DQN" in self.agent_type:
            opponent = EnhancedDoubleDQNAgent(
                player_id=2,
                state_size=92,
                action_size=7,
                hidden_size=512,
                seed=456
            )
        else:
            # Fixed Double DQN
            opponent = FixedDoubleDQNAgent(
                player_id=2,
                state_size=84,
                action_size=7,
                seed=456,
                gradient_clip_norm=1.0,
                use_huber_loss=True,
                huber_delta=1.0,
                state_normalization=True
            )
        
        opponent.load(self.rl_model_path, keep_player_id=False)
        opponent.epsilon = 0.0
        return opponent
    
    def simulate_game(self, agent1, agent2) -> Tuple[Optional[int], int, List[int]]:
        """
        Simulate a single game between two agents.
        
        Returns:
            winner: player_id of winner (1 or 2) or None for draw
            game_length: number of moves played
            move_history: list of column choices
        """
        board = Connect4Board()
        move_history = []
        move_count = 0
        
        while not board.is_terminal() and move_count < 42:
            # Determine current agent
            current_agent = agent1 if board.current_player == 1 else agent2
            
            # Get legal moves and choose action
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
                
            action = current_agent.choose_action(board.get_state(), legal_moves)
            move_history.append(action)
            
            # Make move
            board.make_move(action, current_agent.player_id)
            move_count += 1
        
        winner = board.check_winner()
        return winner, move_count, move_history
    
    def run_simulation(self, opponent_type: str, num_games: int = 1000, 
                      verbose: bool = True) -> Dict[str, Any]:
        """
        Run comprehensive simulation between RL agent and specified opponent.
        
        Args:
            opponent_type: 'random', 'heuristic', or 'self'
            num_games: number of games to simulate
            verbose: whether to show progress
            
        Returns:
            comprehensive statistics dictionary
        """
        if self.rl_agent is None:
            raise ValueError("RL agent not loaded properly")
        
        # Create opponent
        if opponent_type == 'random':
            opponent = RandomAgent(player_id=2, seed=123)
        elif opponent_type == 'heuristic':
            opponent = HeuristicAgent(player_id=2, seed=123)
        elif opponent_type == 'self':
            # Create a copy of the RL agent as opponent with same config
            opponent = self._create_self_play_opponent()
            if opponent is None:
                raise ValueError("Failed to create self-play opponent")
        else:
            raise ValueError(f"Invalid opponent type: {opponent_type}")
        
        print(f"\n🎮 Running {num_games:,} game simulation:")
        print(f"   RL Agent vs {opponent.name}")
        print(f"   Model: {os.path.basename(self.rl_model_path)}")
        print("=" * 50)
        
        # Statistics tracking
        results = {
            'rl_wins': 0,
            'opponent_wins': 0,
            'draws': 0,
            'rl_wins_as_first': 0,
            'rl_wins_as_second': 0,
            'opponent_wins_as_first': 0,
            'opponent_wins_as_second': 0,
            'draws_rl_first': 0,
            'draws_opponent_first': 0,
            'game_lengths': [],
            'rl_first_game_lengths': [],
            'opponent_first_game_lengths': [],
            'total_moves': 0
        }
        
        start_time = time.time()
        
        for game_num in range(num_games):
            # Alternate who goes first
            if game_num % 2 == 0:
                # RL agent goes first
                self.rl_agent.player_id = 1
                opponent.player_id = 2
                agent1, agent2 = self.rl_agent, opponent
                rl_goes_first = True
            else:
                # Opponent goes first
                self.rl_agent.player_id = 2
                opponent.player_id = 1
                agent1, agent2 = opponent, self.rl_agent
                rl_goes_first = False
            
            # Simulate game
            winner, game_length, move_history = self.simulate_game(agent1, agent2)
            
            # Update statistics
            results['game_lengths'].append(game_length)
            results['total_moves'] += game_length
            
            if winner == self.rl_agent.player_id:
                results['rl_wins'] += 1
                if rl_goes_first:
                    results['rl_wins_as_first'] += 1
                    results['rl_first_game_lengths'].append(game_length)
                else:
                    results['rl_wins_as_second'] += 1
                    results['opponent_first_game_lengths'].append(game_length)
            elif winner == opponent.player_id:
                results['opponent_wins'] += 1
                if rl_goes_first:
                    results['opponent_wins_as_second'] += 1
                    results['rl_first_game_lengths'].append(game_length)
                else:
                    results['opponent_wins_as_first'] += 1
                    results['opponent_first_game_lengths'].append(game_length)
            else:
                results['draws'] += 1
                if rl_goes_first:
                    results['draws_rl_first'] += 1
                    results['rl_first_game_lengths'].append(game_length)
                else:
                    results['draws_opponent_first'] += 1
                    results['opponent_first_game_lengths'].append(game_length)
            
            # Progress update
            if verbose and (game_num + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (game_num + 1) / elapsed
                remaining = (num_games - game_num - 1) / rate
                print(f"Progress: {game_num + 1:4d}/{num_games} games "
                      f"({(game_num + 1)/num_games*100:.1f}%) - "
                      f"Rate: {rate:.1f} games/sec - "
                      f"ETA: {remaining:.0f}s")
        
        simulation_time = time.time() - start_time
        
        # Calculate comprehensive statistics
        stats = self._calculate_statistics(results, num_games, opponent_type, simulation_time)
        
        return stats
    
    def _calculate_statistics(self, results: Dict, num_games: int, 
                            opponent_type: str, simulation_time: float) -> Dict[str, Any]:
        """Calculate comprehensive statistics from simulation results."""
        
        # Basic win rates
        rl_win_rate = results['rl_wins'] / num_games
        opponent_win_rate = results['opponent_wins'] / num_games
        draw_rate = results['draws'] / num_games
        
        # Position-based statistics
        games_as_first = num_games // 2 + (num_games % 2)  # RL goes first in odd-numbered total games
        games_as_second = num_games // 2
        
        rl_win_rate_first = results['rl_wins_as_first'] / games_as_first if games_as_first > 0 else 0
        rl_win_rate_second = results['rl_wins_as_second'] / games_as_second if games_as_second > 0 else 0
        
        opponent_win_rate_first = results['opponent_wins_as_first'] / games_as_second if games_as_second > 0 else 0
        opponent_win_rate_second = results['opponent_wins_as_second'] / games_as_first if games_as_first > 0 else 0
        
        # Game length statistics
        avg_game_length = np.mean(results['game_lengths'])
        median_game_length = np.median(results['game_lengths'])
        min_game_length = np.min(results['game_lengths'])
        max_game_length = np.max(results['game_lengths'])
        
        # Game length by starting position
        avg_length_rl_first = np.mean(results['rl_first_game_lengths']) if results['rl_first_game_lengths'] else 0
        avg_length_opponent_first = np.mean(results['opponent_first_game_lengths']) if results['opponent_first_game_lengths'] else 0
        
        # First move advantage
        first_move_advantage = (results['rl_wins_as_first'] + results['opponent_wins_as_first']) / num_games
        
        return {
            'simulation_info': {
                'opponent_type': opponent_type,
                'num_games': num_games,
                'simulation_time_seconds': round(simulation_time, 2),
                'games_per_second': round(num_games / simulation_time, 2),
                'model_path': self.rl_model_path
            },
            'overall_results': {
                'rl_wins': results['rl_wins'],
                'opponent_wins': results['opponent_wins'],
                'draws': results['draws'],
                'rl_win_rate': round(rl_win_rate, 4),
                'opponent_win_rate': round(opponent_win_rate, 4),
                'draw_rate': round(draw_rate, 4)
            },
            'position_analysis': {
                'games_as_first_player': games_as_first,
                'games_as_second_player': games_as_second,
                'rl_win_rate_when_first': round(rl_win_rate_first, 4),
                'rl_win_rate_when_second': round(rl_win_rate_second, 4),
                'opponent_win_rate_when_first': round(opponent_win_rate_first, 4),
                'opponent_win_rate_when_second': round(opponent_win_rate_second, 4),
                'first_move_advantage': round(first_move_advantage, 4)
            },
            'game_length_analysis': {
                'average_game_length': round(avg_game_length, 2),
                'median_game_length': median_game_length,
                'min_game_length': min_game_length,
                'max_game_length': max_game_length,
                'avg_length_rl_first': round(avg_length_rl_first, 2),
                'avg_length_opponent_first': round(avg_length_opponent_first, 2),
                'total_moves_played': results['total_moves']
            },
            'detailed_breakdown': {
                'rl_wins_as_first': results['rl_wins_as_first'],
                'rl_wins_as_second': results['rl_wins_as_second'],
                'opponent_wins_as_first': results['opponent_wins_as_first'],
                'opponent_wins_as_second': results['opponent_wins_as_second'],
                'draws_when_rl_first': results['draws_rl_first'],
                'draws_when_opponent_first': results['draws_opponent_first']
            }
        }
    
    def print_results(self, stats: Dict[str, Any]) -> None:
        """Print comprehensive simulation results in a formatted way."""
        info = stats['simulation_info']
        overall = stats['overall_results']
        position = stats['position_analysis']
        length = stats['game_length_analysis']
        
        print(f"\n📊 SIMULATION RESULTS")
        print("=" * 60)
        print(f"🤖 RL Agent vs {info['opponent_type'].title()} Agent")
        print(f"📁 Model: {os.path.basename(info['model_path'])}")
        print(f"🎮 Games played: {info['num_games']:,}")
        print(f"⏱️  Simulation time: {info['simulation_time_seconds']}s ({info['games_per_second']} games/sec)")
        
        print(f"\n🏆 OVERALL RESULTS")
        print("-" * 30)
        print(f"RL Agent wins:    {overall['rl_wins']:4d} ({overall['rl_win_rate']:6.1%})")
        print(f"Opponent wins:    {overall['opponent_wins']:4d} ({overall['opponent_win_rate']:6.1%})")
        print(f"Draws:            {overall['draws']:4d} ({overall['draw_rate']:6.1%})")
        
        # Performance assessment
        if overall['rl_win_rate'] > 0.7:
            performance = "🟢 EXCELLENT"
        elif overall['rl_win_rate'] > 0.6:
            performance = "🟡 GOOD"
        elif overall['rl_win_rate'] > 0.5:
            performance = "🟠 FAIR"
        else:
            performance = "🔴 POOR"
        print(f"Performance:      {performance}")
        
        print(f"\n🔄 POSITION ANALYSIS")
        print("-" * 30)
        print(f"When RL goes first:  {position['rl_win_rate_when_first']:6.1%} win rate")
        print(f"When RL goes second: {position['rl_win_rate_when_second']:6.1%} win rate")
        print(f"When opponent first: {position['opponent_win_rate_when_first']:6.1%} win rate")
        print(f"When opponent second: {position['opponent_win_rate_when_second']:6.1%} win rate")
        print(f"First move advantage: {position['first_move_advantage']:6.1%} (higher = first player wins more)")
        
        # Position advantage analysis
        rl_position_advantage = position['rl_win_rate_when_first'] - position['rl_win_rate_when_second']
        if abs(rl_position_advantage) > 0.1:
            pos_analysis = "🎯 Significant positional bias"
        elif abs(rl_position_advantage) > 0.05:
            pos_analysis = "⚖️  Slight positional preference"
        else:
            pos_analysis = "✅ Position-independent play"
        print(f"Position bias:       {pos_analysis} ({rl_position_advantage:+.1%})")
        
        print(f"\n⏱️  GAME LENGTH ANALYSIS")
        print("-" * 30)
        print(f"Average game length: {length['average_game_length']:.1f} moves")
        print(f"Median game length:  {length['median_game_length']} moves")
        print(f"Range:               {length['min_game_length']}-{length['max_game_length']} moves")
        print(f"When RL first:       {length['avg_length_rl_first']:.1f} moves avg")
        print(f"When opponent first: {length['avg_length_opponent_first']:.1f} moves avg")
        print(f"Total moves played:  {length['total_moves_played']:,}")
        
        # Game length interpretation
        if length['average_game_length'] < 15:
            game_style = "⚡ Fast, decisive games"
        elif length['average_game_length'] < 25:
            game_style = "⚖️  Balanced tactical games"
        else:
            game_style = "🏰 Long, strategic battles"
        print(f"Game style:          {game_style}")


def find_best_model() -> Optional[str]:
    """Find the best available trained model with priority order."""
    import glob
    
    # Priority order: M1 CNN → Ultra-light CNN → Enhanced → Fixed Legacy
    model_candidates = [
        # M1-Optimized CNN models (highest priority)
        "models_m1_cnn/m1_cnn_dqn_ep_518000.pt",
        # "models_m1_cnn/m1_cnn_dqn_final.pt",
        # "models_m1_cnn/m1_cnn_dqn_best_ep_*.pt",
        # "models_m1_cnn/m1_cnn_dqn_ep_*.pt",
        
        # # Ultra-light CNN models
        # "models_cnn/cnn_dqn_final.pt",
        # "models_cnn/cnn_dqn_best_ep_*.pt",
        # "models_cnn/cnn_dqn_ep_*.pt",
        
        # # Enhanced Double DQN models
        # "models_enhanced/enhanced_double_dqn_final.pt",
        # "models_enhanced/enhanced_double_dqn_best_ep_*.pt",
        # "models_enhanced/enhanced_double_dqn_ep_*.pt",
        
        # # Fixed Double DQN models (legacy)
        # "models_fixed/double_dqn_ep_150000.pt",
        # "models_fixed/double_dqn_final.pt",
        # "models_fixed/double_dqn_best_ep_*.pt",
        # "models_fixed/double_dqn_ep_*.pt",
    ]
    
    for pattern in model_candidates:
        if '*' in pattern:
            # Handle glob patterns for best models
            matches = glob.glob(pattern)
            if matches:
                # Return the highest episode number
                matches.sort(key=lambda x: int(x.split('_ep_')[1].split('.pt')[0]) if '_ep_' in x else 0)
                return matches[-1]
        else:
            if os.path.exists(pattern):
                return pattern
    
    return None


def main():
    """Main function to run agent simulation."""
    parser = argparse.ArgumentParser(description='Simulate RL Agent vs Opponents')
    parser.add_argument('--opponent', choices=['random', 'heuristic', 'self'], 
                       default='heuristic', help='Opponent type')
    parser.add_argument('--games', type=int, default=1000, 
                       help='Number of games to simulate')
    parser.add_argument('--model', type=str, help='Path to RL model file')
    parser.add_argument('--save-results', action='store_true', 
                       help='Save results to JSON file')
    parser.add_argument('--all-opponents', action='store_true',
                       help='Test against all opponent types')
    
    args = parser.parse_args()
    
    # Find model
    if args.model:
        model_path = args.model
    else:
        model_path = find_best_model()
        if not model_path:
            print("❌ No trained model found. Available models should be in 'models/' directory.")
            print("   Train a model first using: python train/double_dqn_train.py")
            return
    
    print(f"🎯 Connect 4 Agent Simulation")
    print(f"🤖 Using model: {os.path.basename(model_path)}")
    
    # Initialize simulator
    simulator = GameSimulator(model_path)
    
    if args.all_opponents:
        # Test against all opponent types
        opponents = ['random', 'heuristic', 'self']
        all_results = {}
        
        for opponent in opponents:
            print(f"\n{'='*60}")
            print(f"Testing against {opponent.upper()} opponent")
            print(f"{'='*60}")
            
            try:
                stats = simulator.run_simulation(opponent, args.games, verbose=True)
                simulator.print_results(stats)
                all_results[opponent] = stats
                
                if args.save_results:
                    filename = f"simulation_results_{opponent}_{args.games}games.json"
                    with open(filename, 'w') as f:
                        json.dump(stats, f, indent=2)
                    print(f"\n💾 Results saved to: {filename}")
                    
            except Exception as e:
                print(f"❌ Error testing against {opponent}: {e}")
        
        # Summary comparison
        print(f"\n{'='*60}")
        print("📊 COMPARATIVE SUMMARY")
        print(f"{'='*60}")
        print(f"{'Opponent':<15} {'Win Rate':<10} {'Avg Length':<12} {'Performance':<15}")
        print("-" * 60)
        
        for opponent in opponents:
            if opponent in all_results:
                stats = all_results[opponent]
                win_rate = stats['overall_results']['rl_win_rate']
                avg_length = stats['game_length_analysis']['average_game_length']
                
                if win_rate > 0.7:
                    perf = "🟢 Excellent"
                elif win_rate > 0.6:
                    perf = "🟡 Good"
                elif win_rate > 0.5:
                    perf = "🟠 Fair"
                else:
                    perf = "🔴 Poor"
                
                print(f"{opponent.title():<15} {win_rate:<10.1%} {avg_length:<12.1f} {perf:<15}")
        
    else:
        # Single opponent test
        try:
            stats = simulator.run_simulation(args.opponent, args.games, verbose=True)
            simulator.print_results(stats)
            
            if args.save_results:
                filename = f"simulation_results_{args.opponent}_{args.games}games.json"
                with open(filename, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"\n💾 Results saved to: {filename}")
                
        except Exception as e:
            print(f"❌ Simulation error: {e}")


if __name__ == "__main__":
    main()