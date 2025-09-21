#!/usr/bin/env python3
"""RL vs RL simulation script for comparing two trained models."""

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


class RLvsRLSimulator:
    """Simulator for comparing two RL models against each other."""
    
    def __init__(self, model1_path: str, model2_path: str):
        """Initialize simulator with two RL models."""
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.agent1 = None
        self.agent2 = None
        self.agent1_type = None
        self.agent2_type = None
        
        # Load both agents
        self.agent1 = self._load_agent_auto_detect(model1_path, player_id=1)
        self.agent2 = self._load_agent_auto_detect(model2_path, player_id=2)
        
        if self.agent1 is None or self.agent2 is None:
            raise ValueError("Failed to load one or both RL models")
        
        # Set to evaluation mode (no exploration)
        self.agent1.epsilon = 0.0
        self.agent2.epsilon = 0.0
        
        print(f"✅ Agent 1 ({self.agent1_type}): {os.path.basename(model1_path)}")
        print(f"✅ Agent 2 ({self.agent2_type}): {os.path.basename(model2_path)}")
    
    def _load_agent_auto_detect(self, model_path: str, player_id: int):
        """Auto-detect and load the appropriate agent type."""
        model_name = os.path.basename(model_path).lower()
        
        if "m1_cnn" in model_path.lower() or "cnn" in model_name:
            # M1 CNN or CNN model
            if "m1" in model_path.lower():
                architecture = "m1_optimized"
                hidden_size = 48
                agent_type = "M1-Optimized CNN"
            else:
                architecture = "ultra_light"
                hidden_size = 16
                agent_type = "Ultra-Light CNN"
            
            if player_id == 1:
                self.agent1_type = agent_type
            else:
                self.agent2_type = agent_type
                
            agent = CNNDuelingDQNAgent(
                player_id=player_id,
                input_channels=2,
                action_size=7,
                hidden_size=hidden_size,
                architecture=architecture,
                seed=42
            )
            agent.load(model_path, keep_player_id=False)
            agent.player_id = player_id  # Override player ID
            return agent
            
        elif "enhanced" in model_path.lower():
            # Enhanced Double DQN model
            agent_type = "Enhanced Double DQN"
            if player_id == 1:
                self.agent1_type = agent_type
            else:
                self.agent2_type = agent_type
                
            agent = EnhancedDoubleDQNAgent(
                player_id=player_id,
                state_size=92,
                action_size=7,
                hidden_size=512,
                seed=42
            )
            agent.load(model_path, keep_player_id=False)
            agent.player_id = player_id
            return agent
            
        else:
            # Fixed Double DQN model
            agent_type = "Fixed Double DQN"
            if player_id == 1:
                self.agent1_type = agent_type
            else:
                self.agent2_type = agent_type
                
            agent = FixedDoubleDQNAgent(
                player_id=player_id,
                state_size=84,
                action_size=7,
                seed=42,
                gradient_clip_norm=1.0,
                use_huber_loss=True,
                huber_delta=1.0,
                state_normalization=True
            )
            agent.load(model_path, keep_player_id=False)
            agent.player_id = player_id
            return agent
    
    def simulate_game(self) -> Tuple[Optional[int], int, List[int]]:
        """Simulate a single game between the two agents."""
        board = Connect4Board()
        move_history = []
        move_count = 0
        
        while not board.is_terminal() and move_count < 42:
            # Determine current agent based on their actual player_id assignments
            if board.current_player == self.agent1.player_id:
                current_agent = self.agent1
            elif board.current_player == self.agent2.player_id:
                current_agent = self.agent2
            else:
                raise ValueError(f"Unknown player {board.current_player}")
            
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
    
    def run_tournament(self, num_games: int = 1000, verbose: bool = True) -> Dict[str, Any]:
        """Run a tournament between the two RL agents."""
        print(f"\n🎮 RL vs RL TOURNAMENT")
        print("=" * 60)
        print(f"🤖 Agent 1 ({self.agent1_type}) vs Agent 2 ({self.agent2_type})")
        print(f"📁 Model 1: {os.path.basename(self.model1_path)}")
        print(f"📁 Model 2: {os.path.basename(self.model2_path)}")
        print(f"🎯 Games: {num_games:,}")
        print("=" * 60)
        
        # Statistics tracking
        results = {
            'agent1_wins': 0,
            'agent2_wins': 0,
            'draws': 0,
            'agent1_wins_as_first': 0,
            'agent1_wins_as_second': 0,
            'agent2_wins_as_first': 0,
            'agent2_wins_as_second': 0,
            'draws_agent1_first': 0,
            'draws_agent2_first': 0,
            'game_lengths': [],
            'agent1_first_game_lengths': [],
            'agent2_first_game_lengths': []
        }
        
        start_time = time.time()
        
        for game_num in range(num_games):
            # Alternate who goes first
            if game_num % 2 == 0:
                # Agent 1 goes first (board position 1)
                self.agent1.player_id = 1
                self.agent2.player_id = 2
                agent1_goes_first = True
            else:
                # Agent 2 goes first (board position 1)
                self.agent1.player_id = 2
                self.agent2.player_id = 1
                agent1_goes_first = False
            
            # Simulate game
            winner, game_length, move_history = self.simulate_game()
            
            # Update statistics
            results['game_lengths'].append(game_length)
            
            # Determine which agent actually won based on board position and who was where
            if winner == 1:
                # Board position 1 won
                if agent1_goes_first:
                    # Agent 1 was in position 1, so Agent 1 won
                    results['agent1_wins'] += 1
                    results['agent1_wins_as_first'] += 1
                    results['agent1_first_game_lengths'].append(game_length)
                else:
                    # Agent 2 was in position 1, so Agent 2 won
                    results['agent2_wins'] += 1
                    results['agent2_wins_as_first'] += 1
                    results['agent2_first_game_lengths'].append(game_length)
            elif winner == 2:
                # Board position 2 won
                if agent1_goes_first:
                    # Agent 1 was in position 1, Agent 2 was in position 2, so Agent 2 won
                    results['agent2_wins'] += 1
                    results['agent2_wins_as_second'] += 1
                    results['agent1_first_game_lengths'].append(game_length)
                else:
                    # Agent 2 was in position 1, Agent 1 was in position 2, so Agent 1 won
                    results['agent1_wins'] += 1
                    results['agent1_wins_as_second'] += 1
                    results['agent2_first_game_lengths'].append(game_length)
            else:
                # Draw
                results['draws'] += 1
                if agent1_goes_first:
                    results['draws_agent1_first'] += 1
                    results['agent1_first_game_lengths'].append(game_length)
                else:
                    results['draws_agent2_first'] += 1
                    results['agent2_first_game_lengths'].append(game_length)
            
            # Progress update
            if verbose and (game_num + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (game_num + 1) / elapsed
                remaining = (num_games - game_num - 1) / rate
                agent1_current_wins = results['agent1_wins']
                print(f"Progress: {game_num + 1:4d}/{num_games} games "
                      f"({(game_num + 1)/num_games*100:.1f}%) - "
                      f"Agent1: {agent1_current_wins}/{game_num + 1} ({agent1_current_wins/(game_num + 1)*100:.1f}%) - "
                      f"Rate: {rate:.1f} games/sec - "
                      f"ETA: {remaining:.0f}s")
        
        simulation_time = time.time() - start_time
        
        # Calculate comprehensive statistics
        stats = self._calculate_tournament_statistics(results, num_games, simulation_time)
        
        return stats
    
    def _calculate_tournament_statistics(self, results: Dict, num_games: int, 
                                       simulation_time: float) -> Dict[str, Any]:
        """Calculate comprehensive tournament statistics."""
        
        # Basic win rates
        agent1_win_rate = results['agent1_wins'] / num_games
        agent2_win_rate = results['agent2_wins'] / num_games
        draw_rate = results['draws'] / num_games
        
        # Position-based statistics
        games_as_first = num_games // 2 + (num_games % 2)  # Agent 1 goes first in even-indexed games (0, 2, 4...)
        games_as_second = num_games // 2
        
        agent1_win_rate_first = results['agent1_wins_as_first'] / games_as_first if games_as_first > 0 else 0
        agent1_win_rate_second = results['agent1_wins_as_second'] / games_as_second if games_as_second > 0 else 0
        
        agent2_win_rate_first = results['agent2_wins_as_first'] / games_as_second if games_as_second > 0 else 0
        agent2_win_rate_second = results['agent2_wins_as_second'] / games_as_first if games_as_first > 0 else 0
        
        # Game length statistics
        avg_game_length = np.mean(results['game_lengths'])
        median_game_length = np.median(results['game_lengths'])
        min_game_length = np.min(results['game_lengths'])
        max_game_length = np.max(results['game_lengths'])
        
        # Game length by starting position
        avg_length_agent1_first = np.mean(results['agent1_first_game_lengths']) if results['agent1_first_game_lengths'] else 0
        avg_length_agent2_first = np.mean(results['agent2_first_game_lengths']) if results['agent2_first_game_lengths'] else 0
        
        # First move advantage
        first_move_advantage = (results['agent1_wins_as_first'] + results['agent2_wins_as_first']) / num_games
        
        return {
            'tournament_info': {
                'agent1_type': self.agent1_type,
                'agent2_type': self.agent2_type,
                'agent1_model': self.model1_path,
                'agent2_model': self.model2_path,
                'num_games': num_games,
                'simulation_time_seconds': round(simulation_time, 2),
                'games_per_second': round(num_games / simulation_time, 2)
            },
            'overall_results': {
                'agent1_wins': results['agent1_wins'],
                'agent2_wins': results['agent2_wins'],
                'draws': results['draws'],
                'agent1_win_rate': round(agent1_win_rate, 4),
                'agent2_win_rate': round(agent2_win_rate, 4),
                'draw_rate': round(draw_rate, 4)
            },
            'position_analysis': {
                'games_as_first_player': games_as_first,
                'games_as_second_player': games_as_second,
                'agent1_win_rate_when_first': round(agent1_win_rate_first, 4),
                'agent1_win_rate_when_second': round(agent1_win_rate_second, 4),
                'agent2_win_rate_when_first': round(agent2_win_rate_first, 4),
                'agent2_win_rate_when_second': round(agent2_win_rate_second, 4),
                'first_move_advantage': round(first_move_advantage, 4)
            },
            'game_length_analysis': {
                'average_game_length': round(avg_game_length, 2),
                'median_game_length': median_game_length,
                'min_game_length': min_game_length,
                'max_game_length': max_game_length,
                'avg_length_agent1_first': round(avg_length_agent1_first, 2),
                'avg_length_agent2_first': round(avg_length_agent2_first, 2),
                'total_moves_played': sum(results['game_lengths'])
            },
            'detailed_breakdown': {
                'agent1_wins_as_first': results['agent1_wins_as_first'],
                'agent1_wins_as_second': results['agent1_wins_as_second'],
                'agent2_wins_as_first': results['agent2_wins_as_first'],
                'agent2_wins_as_second': results['agent2_wins_as_second'],
                'draws_when_agent1_first': results['draws_agent1_first'],
                'draws_when_agent2_first': results['draws_agent2_first']
            }
        }
    
    def print_tournament_results(self, stats: Dict[str, Any]) -> None:
        """Print comprehensive tournament results."""
        info = stats['tournament_info']
        overall = stats['overall_results']
        position = stats['position_analysis']
        length = stats['game_length_analysis']
        
        print(f"\n🏆 TOURNAMENT RESULTS")
        print("=" * 70)
        print(f"🤖 {info['agent1_type']} vs {info['agent2_type']}")
        print(f"📁 Model 1: {os.path.basename(info['agent1_model'])}")
        print(f"📁 Model 2: {os.path.basename(info['agent2_model'])}")
        print(f"🎮 Games played: {info['num_games']:,}")
        print(f"⏱️  Tournament time: {info['simulation_time_seconds']}s ({info['games_per_second']} games/sec)")
        
        print(f"\n🏆 OVERALL RESULTS")
        print("-" * 40)
        print(f"Agent 1 wins:     {overall['agent1_wins']:4d} ({overall['agent1_win_rate']:6.1%})")
        print(f"Agent 2 wins:     {overall['agent2_wins']:4d} ({overall['agent2_win_rate']:6.1%})")
        print(f"Draws:            {overall['draws']:4d} ({overall['draw_rate']:6.1%})")
        
        # Determine winner
        if overall['agent1_win_rate'] > overall['agent2_win_rate']:
            margin = overall['agent1_win_rate'] - overall['agent2_win_rate']
            if margin > 0.1:
                result = f"🥇 Agent 1 DOMINATES (+{margin:.1%})"
            elif margin > 0.05:
                result = f"🥇 Agent 1 wins clearly (+{margin:.1%})"
            else:
                result = f"🥇 Agent 1 wins narrowly (+{margin:.1%})"
        elif overall['agent2_win_rate'] > overall['agent1_win_rate']:
            margin = overall['agent2_win_rate'] - overall['agent1_win_rate']
            if margin > 0.1:
                result = f"🥇 Agent 2 DOMINATES (+{margin:.1%})"
            elif margin > 0.05:
                result = f"🥇 Agent 2 wins clearly (+{margin:.1%})"
            else:
                result = f"🥇 Agent 2 wins narrowly (+{margin:.1%})"
        else:
            result = "🤝 Perfect TIE!"
        
        print(f"Result:           {result}")
        
        print(f"\n🔄 POSITION ANALYSIS")
        print("-" * 40)
        print(f"Agent 1 when first:  {position['agent1_win_rate_when_first']:6.1%}")
        print(f"Agent 1 when second: {position['agent1_win_rate_when_second']:6.1%}")
        print(f"Agent 2 when first:  {position['agent2_win_rate_when_first']:6.1%}")
        print(f"Agent 2 when second: {position['agent2_win_rate_when_second']:6.1%}")
        print(f"First move advantage: {position['first_move_advantage']:6.1%}")
        
        print(f"\n⏱️  GAME LENGTH ANALYSIS")
        print("-" * 40)
        print(f"Average game length: {length['average_game_length']:.1f} moves")
        print(f"Median game length:  {length['median_game_length']} moves")
        print(f"Range:               {length['min_game_length']}-{length['max_game_length']} moves")
        print(f"When Agent 1 first:  {length['avg_length_agent1_first']:.1f} moves avg")
        print(f"When Agent 2 first:  {length['avg_length_agent2_first']:.1f} moves avg")
        
        # Game style assessment
        if length['average_game_length'] < 15:
            game_style = "⚡ Fast, decisive games"
        elif length['average_game_length'] < 25:
            game_style = "⚖️  Balanced tactical games"
        else:
            game_style = "🏰 Long, strategic battles"
        print(f"Game style:          {game_style}")


def main():
    """Main function for RL vs RL tournament."""
    parser = argparse.ArgumentParser(description='RL vs RL Tournament Simulation')
    parser.add_argument('model1', help='Path to first RL model')
    parser.add_argument('model2', help='Path to second RL model')
    parser.add_argument('--games', type=int, default=1000, help='Number of games to play')
    parser.add_argument('--save-results', action='store_true', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Validate model files
    if not os.path.exists(args.model1):
        print(f"❌ Model 1 not found: {args.model1}")
        return
    
    if not os.path.exists(args.model2):
        print(f"❌ Model 2 not found: {args.model2}")
        return
    
    try:
        # Initialize tournament
        simulator = RLvsRLSimulator(args.model1, args.model2)
        
        # Run tournament
        stats = simulator.run_tournament(args.games, verbose=True)
        
        # Print results
        simulator.print_tournament_results(stats)
        
        # Save results if requested
        if args.save_results:
            model1_name = os.path.splitext(os.path.basename(args.model1))[0]
            model2_name = os.path.splitext(os.path.basename(args.model2))[0]
            filename = f"rl_tournament_{model1_name}_vs_{model2_name}_{args.games}games.json"
            
            with open(filename, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\n💾 Results saved to: {filename}")
            
    except Exception as e:
        print(f"❌ Tournament failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()