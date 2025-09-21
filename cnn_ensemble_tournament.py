#!/usr/bin/env python3
"""
CNN and Ensemble Tournament System with Parallel Processing.

This tournament includes:
- All M1-CNN models (10k, 20k, 50k, then every 50k)
- Random and Heuristic baselines  
- Various ensemble configurations with different weights and methods
- Parallel processing using joblib for speed
"""

import os
import sys
import json
import glob
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joblib import Parallel, delayed
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.board import Connect4Board
from agents.cnn_dueling_dqn_agent import CNNDuelingDQNAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent
from ensemble_agent import EnsembleAgent


class CNNEnsembleTournament:
    """Tournament system for CNN models and ensemble agents with parallel processing."""
    
    def __init__(self, games_per_matchup: int = 100, n_jobs: int = -1):
        """
        Initialize tournament system.
        
        Args:
            games_per_matchup: Number of games per agent matchup
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.games_per_matchup = games_per_matchup
        self.n_jobs = n_jobs
        self.participants = []
        self.agent_cache = {}  # Cache loaded agents
        self.results_matrix = None
        self.win_rate_matrix = None
        
        print(f"🏆 CNN + Ensemble Tournament System")
        print(f"🎮 Games per matchup: {games_per_matchup:,}")
        print(f"⚡ Parallel jobs: {n_jobs if n_jobs > 0 else 'All cores'}")
        
        # Discover and prepare all participants
        self._discover_cnn_models()
        self._create_ensemble_agents()
        self._add_baseline_agents()
        
        print(f"✅ Tournament ready with {len(self.participants)} participants")
    
    def _discover_cnn_models(self):
        """Discover all CNN models following the specified criteria."""
        print(f"\n🔍 Discovering CNN models...")
        
        cnn_models = []
        m1_cnn_dir = "models_m1_cnn"
        
        if not os.path.exists(m1_cnn_dir):
            print(f"❌ {m1_cnn_dir} directory not found!")
            return
        
        # Specific early models
        early_episodes = [10000, 20000, 50000]
        for ep in early_episodes:
            candidates = [
                f"{m1_cnn_dir}/m1_cnn_dqn_best_ep_{ep}.pt",
                f"{m1_cnn_dir}/m1_cnn_dqn_ep_{ep}.pt"
            ]
            for path in candidates:
                if os.path.exists(path):
                    cnn_models.append((f"M1-CNN-{ep//1000}k", "m1_cnn", path))
                    break
        
        # Every 50k episodes starting from 100k
        for ep in range(100000, 700000, 50000):
            candidates = [
                f"{m1_cnn_dir}/m1_cnn_dqn_best_ep_{ep}.pt",
                f"{m1_cnn_dir}/m1_cnn_dqn_ep_{ep}.pt"
            ]
            for path in candidates:
                if os.path.exists(path):
                    cnn_models.append((f"M1-CNN-{ep//1000}k", "m1_cnn", path))
                    break
        
        # Add discovered models to participants
        for name, model_type, path in cnn_models:
            self.participants.append({
                'name': name,
                'type': 'cnn_model',
                'config': {'path': path, 'architecture': 'm1_optimized'}
            })
            print(f"  ✅ {name}: {os.path.basename(path)}")
        
        print(f"📊 Found {len(cnn_models)} CNN models")
    
    def _create_ensemble_agents(self):
        """Create various ensemble agent configurations."""
        print(f"\n🤖 Creating ensemble agents...")
        
        # Get available CNN models for ensembles
        cnn_participants = [p for p in self.participants if p['type'] == 'cnn_model']
        if len(cnn_participants) < 4:
            print("⚠️ Not enough CNN models for ensemble creation")
            return
        
        # Sort by episode number for top performers
        def extract_episode(name):
            try:
                return int(name.split('-')[-1].replace('k', '')) * 1000
            except:
                return 0
        
        sorted_cnn = sorted(cnn_participants, key=lambda x: extract_episode(x['name']), reverse=True)
        
        # 1. Top 4 Performers with Q-value averaging
        top_4 = sorted_cnn[:4]
        top_4_configs = []
        weights = [0.4, 0.3, 0.2, 0.1]  # Decreasing weights
        for i, participant in enumerate(top_4):
            top_4_configs.append({
                'path': participant['config']['path'],
                'weight': weights[i],
                'name': participant['name']
            })
        
        self.participants.append({
            'name': 'Ensemble-Top4-QAvg',
            'type': 'ensemble',
            'config': {
                'models': top_4_configs,
                'method': 'q_value_averaging',
                'name': 'Top4-QAvg'
            }
        })
        
        # 2. Top 4 Performers with Weighted Voting
        self.participants.append({
            'name': 'Ensemble-Top4-Vote',
            'type': 'ensemble',
            'config': {
                'models': top_4_configs,
                'method': 'weighted_voting',
                'name': 'Top4-Vote'
            }
        })
        
        # 3. Top model + 3 random others with equal weights
        if len(sorted_cnn) >= 4:
            random_others = random.sample(sorted_cnn[1:], min(3, len(sorted_cnn)-1))
            random_configs = [{'path': sorted_cnn[0]['config']['path'], 'weight': 1.0, 'name': sorted_cnn[0]['name']}]
            for participant in random_others:
                random_configs.append({
                    'path': participant['config']['path'],
                    'weight': 1.0,
                    'name': participant['name']
                })
            
            self.participants.append({
                'name': 'Ensemble-TopRand-Equal',
                'type': 'ensemble',
                'config': {
                    'models': random_configs,
                    'method': 'confidence_weighted',
                    'name': 'TopRand-Equal'
                }
            })
        
        # 4. Top 5 with exponential decay weights
        if len(sorted_cnn) >= 5:
            top_5 = sorted_cnn[:5]
            exp_weights = [0.5, 0.25, 0.125, 0.0625, 0.0625]  # Exponential decay
            exp_configs = []
            for i, participant in enumerate(top_5):
                exp_configs.append({
                    'path': participant['config']['path'],
                    'weight': exp_weights[i],
                    'name': participant['name']
                })
            
            self.participants.append({
                'name': 'Ensemble-Top5-ExpDecay',
                'type': 'ensemble',
                'config': {
                    'models': exp_configs,
                    'method': 'q_value_averaging',
                    'name': 'Top5-ExpDecay'
                }
            })
        
        # 5. Random selection of 4 models with random weights
        if len(sorted_cnn) >= 4:
            random_selection = random.sample(sorted_cnn, 4)
            random_weights = np.random.dirichlet([1, 1, 1, 1])  # Random weights that sum to 1
            random_configs = []
            for i, participant in enumerate(random_selection):
                random_configs.append({
                    'path': participant['config']['path'],
                    'weight': float(random_weights[i]),
                    'name': participant['name']
                })
            
            self.participants.append({
                'name': 'Ensemble-Random4-RandWeights',
                'type': 'ensemble',
                'config': {
                    'models': random_configs,
                    'method': 'weighted_voting',
                    'name': 'Random4-RandWeights'
                }
            })
        
        ensemble_count = len([p for p in self.participants if p['type'] == 'ensemble'])
        print(f"🎯 Created {ensemble_count} ensemble configurations")
        for p in self.participants:
            if p['type'] == 'ensemble':
                print(f"  • {p['name']}: {p['config']['method']} with {len(p['config']['models'])} models")
    
    def _add_baseline_agents(self):
        """Add baseline agents."""
        print(f"\n🎲 Adding baseline agents...")
        
        self.participants.extend([
            {
                'name': 'Random',
                'type': 'baseline',
                'config': {'agent_type': 'random'}
            },
            {
                'name': 'Heuristic',
                'type': 'baseline', 
                'config': {'agent_type': 'heuristic'}
            }
        ])
        
        print(f"  ✅ Random agent")
        print(f"  ✅ Heuristic agent")
    
    def _create_agent(self, participant_config: Dict) -> Any:
        """Create an agent based on participant configuration."""
        config_key = json.dumps(participant_config, sort_keys=True)
        
        # Check cache first
        if config_key in self.agent_cache:
            agent = self.agent_cache[config_key]
            # Reset player ID for reuse
            agent.player_id = 1
            return agent
        
        # Create new agent
        if participant_config['type'] == 'cnn_model':
            agent = CNNDuelingDQNAgent(
                player_id=1,
                input_channels=2,
                action_size=7,
                hidden_size=48,
                architecture="m1_optimized",
                seed=42
            )
            agent.load(participant_config['config']['path'], keep_player_id=False)
            agent.epsilon = 0.0
            
        elif participant_config['type'] == 'ensemble':
            config = participant_config['config']
            agent = EnsembleAgent(
                model_configs=config['models'],
                ensemble_method=config['method'],
                player_id=1,
                name=config['name'],
                show_contributions=False  # Disable for tournament speed
            )
            
        elif participant_config['type'] == 'baseline':
            if participant_config['config']['agent_type'] == 'random':
                agent = RandomAgent(player_id=1, seed=42)
            else:  # heuristic
                agent = HeuristicAgent(player_id=1, seed=42)
        
        else:
            raise ValueError(f"Unknown participant type: {participant_config['type']}")
        
        # Cache the agent
        self.agent_cache[config_key] = agent
        return agent
    
    def _simulate_single_matchup(self, matchup_data: Tuple) -> Tuple[str, str, int, int, int]:
        """Simulate a single matchup between two agents."""
        agent1_config, agent2_config, games_per_matchup = matchup_data
        
        # Create agents for this process
        agent1 = self._create_agent(agent1_config)
        agent2 = self._create_agent(agent2_config)
        
        wins_1 = 0
        wins_2 = 0
        draws = 0
        
        for game_num in range(games_per_matchup):
            # Alternate who goes first
            if game_num % 2 == 0:
                agent1.player_id = 1
                agent2.player_id = 2
            else:
                agent1.player_id = 2
                agent2.player_id = 1
            
            # Simulate game
            board = Connect4Board()
            move_count = 0
            
            while not board.is_terminal() and move_count < 42:
                if board.current_player == agent1.player_id:
                    current_agent = agent1
                elif board.current_player == agent2.player_id:
                    current_agent = agent2
                else:
                    break
                
                legal_moves = board.get_legal_moves()
                if not legal_moves:
                    break
                
                try:
                    action = current_agent.choose_action(board.get_state(), legal_moves)
                    board.make_move(action, current_agent.player_id)
                except Exception as e:
                    # In case of error, random move
                    action = random.choice(legal_moves)
                    board.make_move(action, current_agent.player_id)
                
                move_count += 1
            
            winner = board.check_winner()
            
            # Determine which agent won
            if winner == agent1.player_id:
                wins_1 += 1
            elif winner == agent2.player_id:
                wins_2 += 1
            else:
                draws += 1
        
        return (agent1_config['name'], agent2_config['name'], wins_1, wins_2, draws)
    
    def run_tournament(self):
        """Run the complete tournament with parallel processing."""
        print(f"\n🏆 STARTING CNN + ENSEMBLE TOURNAMENT")
        print(f"{'='*80}")
        print(f"🎮 Games per matchup: {self.games_per_matchup:,}")
        print(f"🤖 Participants: {len(self.participants)}")
        print(f"⚡ Parallel jobs: {self.n_jobs if self.n_jobs > 0 else 'All cores'}")
        
        total_matchups = len(self.participants) * (len(self.participants) - 1)
        total_games = total_matchups * self.games_per_matchup
        print(f"🎯 Total matchups: {total_matchups:,}")
        print(f"🎯 Total games: {total_games:,}")
        print(f"{'='*80}")
        
        # Prepare all matchups for parallel processing
        matchup_tasks = []
        for i, agent1_config in enumerate(self.participants):
            for j, agent2_config in enumerate(self.participants):
                if i != j:  # Don't play against self
                    matchup_tasks.append((agent1_config, agent2_config, self.games_per_matchup))
        
        print(f"🚀 Starting parallel tournament execution...")
        start_time = time.time()
        
        # Execute all matchups in parallel
        results = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(self._simulate_single_matchup)(task) for task in matchup_tasks
        )
        
        execution_time = time.time() - start_time
        print(f"\n✅ Tournament completed in {execution_time/60:.1f} minutes")
        print(f"⚡ Average: {len(results)/execution_time:.1f} matchups/second")
        
        # Process results
        self._process_results(results)
        
        # Calculate rankings
        self._calculate_rankings()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Save results
        self._save_results()
    
    def _process_results(self, results: List[Tuple]):
        """Process parallel results into matrices."""
        print(f"\n📊 Processing tournament results...")
        
        n_participants = len(self.participants)
        participant_names = [p['name'] for p in self.participants]
        
        # Initialize result matrices
        self.results_matrix = np.zeros((n_participants, n_participants, 3))  # wins, losses, draws
        self.win_rate_matrix = np.zeros((n_participants, n_participants))
        
        # Create name to index mapping
        name_to_idx = {name: i for i, name in enumerate(participant_names)}
        
        # Process each result
        for agent1_name, agent2_name, wins_1, wins_2, draws in results:
            i = name_to_idx[agent1_name]
            j = name_to_idx[agent2_name]
            
            # Store results
            self.results_matrix[i][j] = [wins_1, wins_2, draws]
            win_rate = wins_1 / self.games_per_matchup
            self.win_rate_matrix[i][j] = win_rate
        
        # Set diagonal to 0.5 (agent vs itself)
        np.fill_diagonal(self.win_rate_matrix, 0.5)
        
        print(f"✅ Results processed: {n_participants}x{n_participants} matrix")
    
    def _calculate_rankings(self):
        """Calculate comprehensive rankings."""
        print(f"\n🏅 Calculating rankings...")
        
        participant_names = [p['name'] for p in self.participants]
        n_participants = len(participant_names)
        
        # Calculate metrics
        overall_win_rates = []
        head_to_head_scores = []
        strength_of_schedule = []
        participant_types = []
        
        for i in range(n_participants):
            # Overall win rate (excluding self-matchup)
            other_indices = [j for j in range(n_participants) if j != i]
            win_rate = np.mean([self.win_rate_matrix[i][j] for j in other_indices])
            overall_win_rates.append(win_rate)
            
            # Head-to-head score
            h2h_score = sum(1 if self.win_rate_matrix[i][j] > 0.5 else 0 for j in other_indices)
            head_to_head_scores.append(h2h_score)
            
            # Strength of schedule
            beaten_opponents = [j for j in other_indices if self.win_rate_matrix[i][j] > 0.5]
            if beaten_opponents:
                sos = np.mean([overall_win_rates[j] if j < len(overall_win_rates) else 0.5 for j in beaten_opponents])
            else:
                sos = 0.0
            strength_of_schedule.append(sos)
            
            # Participant type
            participant_types.append(self.participants[i]['type'])
        
        # Create ranking DataFrame
        self.rankings = pd.DataFrame({
            'Agent': participant_names,
            'Type': participant_types,
            'Overall_Win_Rate': overall_win_rates,
            'Head_to_Head_Wins': head_to_head_scores,
            'Strength_of_Schedule': strength_of_schedule,
            'Games_Played': [self.games_per_matchup * (n_participants - 1)] * n_participants
        })
        
        # Calculate composite score
        self.rankings['Composite_Score'] = (
            0.6 * self.rankings['Overall_Win_Rate'] +
            0.3 * self.rankings['Head_to_Head_Wins'] / (n_participants - 1) +
            0.1 * self.rankings['Strength_of_Schedule']
        )
        
        # Sort by composite score
        self.rankings = self.rankings.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
        self.rankings['Rank'] = range(1, n_participants + 1)
        
        # Print top 15
        print(f"\n🏆 TOP 15 RANKINGS")
        print("=" * 100)
        print(f"{'Rank':<4} {'Agent':<25} {'Type':<10} {'Win Rate':<10} {'H2H':<5} {'SoS':<8} {'Score':<8}")
        print("-" * 100)
        
        for idx, row in self.rankings.head(15).iterrows():
            print(f"{row['Rank']:<4} {row['Agent']:<25} {row['Type']:<10} {row['Overall_Win_Rate']:<10.1%} "
                  f"{row['Head_to_Head_Wins']:<5.0f} {row['Strength_of_Schedule']:<8.3f} "
                  f"{row['Composite_Score']:<8.3f}")
    
    def _generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print(f"\n📊 Generating visualizations...")
        
        # Create output directory
        results_dir = "cnn_ensemble_tournament_results"
        os.makedirs(results_dir, exist_ok=True)
        
        participant_names = [p['name'] for p in self.participants]
        
        # 1. Win Rate Clustermap
        plt.figure(figsize=(16, 14))
        win_rate_df = pd.DataFrame(
            self.win_rate_matrix,
            index=participant_names,
            columns=participant_names
        )
        
        g = sns.clustermap(
            win_rate_df,
            annot=False,  # Too many participants for annotations
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0.5,
            square=True,
            figsize=(16, 14),
            cbar_kws={'label': 'Win Rate'},
            dendrogram_ratio=0.1
        )
        g.fig.suptitle('CNN + Ensemble Tournament: Win Rate Matrix (Clustered)', fontsize=16, y=0.98)
        plt.savefig(f'{results_dir}/win_rate_clustermap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Rankings by Type
        plt.figure(figsize=(14, 10))
        
        # Separate by type
        type_colors = {
            'cnn_model': 'blue',
            'ensemble': 'red', 
            'baseline': 'green'
        }
        
        for agent_type in ['ensemble', 'cnn_model', 'baseline']:
            type_data = self.rankings[self.rankings['Type'] == agent_type]
            if len(type_data) > 0:
                plt.scatter(type_data['Rank'], type_data['Overall_Win_Rate'], 
                           c=type_colors[agent_type], label=agent_type.replace('_', ' ').title(), 
                           s=100, alpha=0.7)
        
        plt.xlabel('Rank')
        plt.ylabel('Overall Win Rate')
        plt.title('CNN + Ensemble Tournament: Performance by Agent Type')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/performance_by_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Top 15 Rankings Bar Plot
        plt.figure(figsize=(16, 8))
        top_15 = self.rankings.head(15)
        
        # Color by type
        colors = [type_colors[t] for t in top_15['Type']]
        
        bars = plt.bar(range(len(top_15)), top_15['Overall_Win_Rate'], color=colors, alpha=0.7)
        
        plt.xlabel('Rank')
        plt.ylabel('Overall Win Rate')
        plt.title('Top 15 Agents: CNN + Ensemble Tournament')
        plt.xticks(range(len(top_15)), [f"{i+1}. {name[:20]}" for i, name in enumerate(top_15['Agent'])], 
                   rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, win_rate in zip(bars, top_15['Overall_Win_Rate']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{win_rate:.1%}', ha='center', va='bottom', fontsize=9)
        
        # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7, label=label) 
                          for label, color in type_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/top_15_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. CNN Training Progression
        cnn_models = self.rankings[self.rankings['Type'] == 'cnn_model'].copy()
        if len(cnn_models) > 1:
            def extract_episodes(name):
                try:
                    return int(name.split('-')[-1].replace('k', '')) * 1000
                except:
                    return 0
            
            cnn_models['Episodes'] = cnn_models['Agent'].apply(extract_episodes)
            cnn_models = cnn_models.sort_values('Episodes')
            
            plt.figure(figsize=(12, 6))
            plt.plot(cnn_models['Episodes']/1000, cnn_models['Overall_Win_Rate'], 
                    marker='o', linewidth=2, markersize=8, color='blue')
            plt.xlabel('Training Episodes (thousands)')
            plt.ylabel('Overall Win Rate')
            plt.title('M1-CNN Training Progression')
            plt.grid(True, alpha=0.3)
            
            # Annotate points
            for _, row in cnn_models.iterrows():
                plt.annotate(f"{row['Overall_Win_Rate']:.1%}", 
                           (row['Episodes']/1000, row['Overall_Win_Rate']),
                           textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
            plt.savefig(f'{results_dir}/cnn_training_progression.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Visualizations saved to {results_dir}/")
    
    def _save_results(self):
        """Save comprehensive results."""
        print(f"\n💾 Saving results...")
        
        results_dir = "cnn_ensemble_tournament_results"
        participant_names = [p['name'] for p in self.participants]
        
        # Save rankings
        self.rankings.to_csv(f'{results_dir}/rankings.csv', index=False)
        
        # Save win rate matrix
        win_rate_df = pd.DataFrame(
            self.win_rate_matrix,
            index=participant_names,
            columns=participant_names
        )
        win_rate_df.to_csv(f'{results_dir}/win_rate_matrix.csv')
        
        # Save participant configurations
        with open(f'{results_dir}/participants.json', 'w') as f:
            json.dump(self.participants, f, indent=2)
        
        # Save detailed results
        tournament_config = {
            'games_per_matchup': self.games_per_matchup,
            'n_jobs': self.n_jobs,
            'total_participants': len(self.participants),
            'participant_names': participant_names,
            'participant_types': [p['type'] for p in self.participants]
        }
        
        results_summary = {
            'tournament_config': tournament_config,
            'rankings': self.rankings.to_dict('records'),
            'win_rate_matrix': self.win_rate_matrix.tolist()
        }
        
        with open(f'{results_dir}/tournament_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"✅ Results saved to {results_dir}/")
        
        # Print summary
        print(f"\n📈 TOURNAMENT SUMMARY")
        print("=" * 60)
        print(f"🥇 Champion: {self.rankings.iloc[0]['Agent']} ({self.rankings.iloc[0]['Overall_Win_Rate']:.1%})")
        print(f"🥈 Runner-up: {self.rankings.iloc[1]['Agent']} ({self.rankings.iloc[1]['Overall_Win_Rate']:.1%})")
        print(f"🥉 Third place: {self.rankings.iloc[2]['Agent']} ({self.rankings.iloc[2]['Overall_Win_Rate']:.1%})")
        
        # Best by type
        for agent_type in ['ensemble', 'cnn_model']:
            best_of_type = self.rankings[self.rankings['Type'] == agent_type].iloc[0]
            print(f"🤖 Best {agent_type.replace('_', ' ')}: {best_of_type['Agent']} "
                  f"(#{best_of_type['Rank']}, {best_of_type['Overall_Win_Rate']:.1%})")
        
        baseline_rank = self.rankings[self.rankings['Agent'] == 'Heuristic']['Rank'].iloc[0]
        print(f"🎯 Heuristic baseline rank: #{baseline_rank}")


def main():
    """Run the CNN + Ensemble tournament."""
    import argparse
    
    parser = argparse.ArgumentParser(description='CNN + Ensemble Tournament')
    parser.add_argument('--games', type=int, default=100, help='Games per matchup (default: 100)')
    parser.add_argument('--jobs', type=int, default=-1, help='Parallel jobs (-1 for all cores)')
    
    args = parser.parse_args()
    
    print("🏆 CNN + ENSEMBLE TOURNAMENT SYSTEM")
    print("=" * 80)
    
    try:
        tournament = CNNEnsembleTournament(
            games_per_matchup=args.games,
            n_jobs=args.jobs
        )
        tournament.run_tournament()
        
        print(f"\n🎉 Tournament completed successfully!")
        print(f"📊 Check 'cnn_ensemble_tournament_results/' for detailed results and visualizations")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Tournament interrupted by user")
    except Exception as e:
        print(f"\n❌ Tournament failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()