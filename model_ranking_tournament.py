#!/usr/bin/env python3
"""Comprehensive model ranking tournament with visualization."""

import os
import sys
import json
import glob
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.board import Connect4Board
from agents.double_dqn_agent import DoubleDQNAgent
from agents.enhanced_double_dqn_agent import EnhancedDoubleDQNAgent
from agents.cnn_dueling_dqn_agent import CNNDuelingDQNAgent
from train_fixed_double_dqn import FixedDoubleDQNAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent


class ModelRankingTournament:
    """Comprehensive tournament system for ranking all available models."""
    
    def __init__(self, games_per_matchup: int = 100):
        """Initialize tournament system."""
        self.games_per_matchup = games_per_matchup
        self.models = []
        self.agents = {}
        self.results_matrix = None
        self.win_rate_matrix = None
        
        # Discover and load all models
        self._discover_models()
        self._load_agents()
    
    def _discover_models(self):
        """Discover all available models following the specified criteria."""
        print("🔍 Discovering models...")
        
        # Add baseline agents
        self.models.extend([
            ("Random", "baseline", "random", None),
            ("Heuristic", "baseline", "heuristic", None)
        ])
        
        # M1 CNN models - every 50k + specific early ones
        m1_cnn_dir = "models_m1_cnn"
        if os.path.exists(m1_cnn_dir):
            # Specific early models
            early_episodes = [1000, 5000, 10000, 20000]
            for ep in early_episodes:
                candidates = [
                    f"{m1_cnn_dir}/m1_cnn_dqn_best_ep_{ep}.pt",
                    f"{m1_cnn_dir}/m1_cnn_dqn_ep_{ep}.pt"
                ]
                for path in candidates:
                    if os.path.exists(path):
                        self.models.append((f"M1-CNN-{ep//1000}k", "m1_cnn", "model", path))
                        break
            
            # Every 50k episodes
            for ep in range(50000, 600000, 50000):
                candidates = [
                    f"{m1_cnn_dir}/m1_cnn_dqn_best_ep_{ep}.pt",
                    f"{m1_cnn_dir}/m1_cnn_dqn_ep_{ep}.pt"
                ]
                for path in candidates:
                    if os.path.exists(path):
                        self.models.append((f"M1-CNN-{ep//1000}k", "m1_cnn", "model", path))
                        break
            
            # Final model if available
            final_candidates = [
                f"{m1_cnn_dir}/m1_cnn_dqn_final_500k.pt",
                f"{m1_cnn_dir}/m1_cnn_dqn_final.pt"
            ]
            for path in final_candidates:
                if os.path.exists(path):
                    self.models.append(("M1-CNN-Final", "m1_cnn", "model", path))
                    break
        
        # Ultra-light CNN models
        cnn_dir = "models_cnn"
        if os.path.exists(cnn_dir):
            cnn_files = glob.glob(f"{cnn_dir}/cnn_dqn_*.pt")
            if cnn_files:
                # Take the best/final model
                best_cnn = max(cnn_files, key=os.path.getmtime)
                self.models.append(("Ultra-CNN", "ultra_cnn", "model", best_cnn))
        
        # Enhanced DQN models
        enhanced_dir = "models_enhanced"
        if os.path.exists(enhanced_dir):
            enhanced_files = glob.glob(f"{enhanced_dir}/enhanced_*.pt")
            if enhanced_files:
                best_enhanced = max(enhanced_files, key=os.path.getmtime)
                self.models.append(("Enhanced-DQN", "enhanced", "model", best_enhanced))
        
        # Fixed DQN models
        fixed_dir = "models_fixed"
        if os.path.exists(fixed_dir):
            fixed_files = glob.glob(f"{fixed_dir}/double_dqn_*.pt")
            if fixed_files:
                best_fixed = max(fixed_files, key=os.path.getmtime)
                self.models.append(("Fixed-DQN", "fixed", "model", best_fixed))
        
        print(f"✅ Found {len(self.models)} models/agents to evaluate:")
        for name, model_type, agent_type, path in self.models:
            if path:
                print(f"   {name}: {os.path.basename(path)}")
            else:
                print(f"   {name}: {agent_type}")
    
    def _load_agents(self):
        """Load all discovered agents."""
        print("\n🤖 Loading agents...")
        
        for name, model_type, agent_type, path in self.models:
            try:
                if agent_type == "random":
                    agent = RandomAgent(player_id=1, seed=42)
                elif agent_type == "heuristic":
                    agent = HeuristicAgent(player_id=1, seed=42)
                elif model_type == "m1_cnn":
                    agent = CNNDuelingDQNAgent(
                        player_id=1,
                        input_channels=2,
                        action_size=7,
                        hidden_size=48,
                        architecture="m1_optimized",
                        seed=42
                    )
                    agent.load(path, keep_player_id=False)
                    agent.epsilon = 0.0
                elif model_type == "ultra_cnn":
                    agent = CNNDuelingDQNAgent(
                        player_id=1,
                        input_channels=2,
                        action_size=7,
                        hidden_size=16,
                        architecture="ultra_light",
                        seed=42
                    )
                    agent.load(path, keep_player_id=False)
                    agent.epsilon = 0.0
                elif model_type == "enhanced":
                    agent = EnhancedDoubleDQNAgent(
                        player_id=1,
                        state_size=92,
                        action_size=7,
                        hidden_size=512,
                        seed=42
                    )
                    agent.load(path, keep_player_id=False)
                    agent.epsilon = 0.0
                elif model_type == "fixed":
                    agent = FixedDoubleDQNAgent(
                        player_id=1,
                        state_size=84,
                        action_size=7,
                        seed=42,
                        gradient_clip_norm=1.0,
                        use_huber_loss=True,
                        huber_delta=1.0,
                        state_normalization=True
                    )
                    agent.load(path, keep_player_id=False)
                    agent.epsilon = 0.0
                else:
                    print(f"❌ Unknown agent type: {agent_type}")
                    continue
                
                self.agents[name] = agent
                print(f"✅ {name} loaded")
                
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
        
        print(f"\n🎯 Successfully loaded {len(self.agents)} agents for tournament")
    
    def _simulate_matchup(self, agent1_name: str, agent2_name: str) -> Tuple[int, int, int]:
        """Simulate games between two agents."""
        agent1 = self.agents[agent1_name]
        agent2 = self.agents[agent2_name]
        
        wins_1 = 0
        wins_2 = 0
        draws = 0
        
        for game_num in range(self.games_per_matchup):
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
                
                action = current_agent.choose_action(board.get_state(), legal_moves)
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
        
        return wins_1, wins_2, draws
    
    def run_tournament(self):
        """Run the complete round-robin tournament."""
        print(f"\n🏆 STARTING COMPREHENSIVE TOURNAMENT")
        print(f"{'='*60}")
        print(f"🎮 Games per matchup: {self.games_per_matchup:,}")
        print(f"🤖 Agents: {len(self.agents)}")
        print(f"🎯 Total games: {len(self.agents) * (len(self.agents) - 1) * self.games_per_matchup:,}")
        print(f"{'='*60}")
        
        agent_names = list(self.agents.keys())
        n_agents = len(agent_names)
        
        # Initialize results matrices
        self.results_matrix = np.zeros((n_agents, n_agents, 3))  # wins, losses, draws
        self.win_rate_matrix = np.zeros((n_agents, n_agents))
        
        total_matchups = n_agents * (n_agents - 1)
        completed_matchups = 0
        start_time = time.time()
        
        for i, agent1_name in enumerate(agent_names):
            for j, agent2_name in enumerate(agent_names):
                if i == j:
                    # Agent vs itself - 50% win rate
                    self.win_rate_matrix[i][j] = 0.5
                    continue
                
                print(f"\n🥊 Matchup {completed_matchups + 1}/{total_matchups}: {agent1_name} vs {agent2_name}")
                
                wins_1, wins_2, draws = self._simulate_matchup(agent1_name, agent2_name)
                
                # Store results
                self.results_matrix[i][j] = [wins_1, wins_2, draws]
                win_rate = wins_1 / self.games_per_matchup
                self.win_rate_matrix[i][j] = win_rate
                
                print(f"   Result: {agent1_name} {win_rate:.1%} - {agent2_name} {1-win_rate:.1%} (Draws: {draws})")
                
                completed_matchups += 1
                
                # Progress update
                elapsed = time.time() - start_time
                rate = completed_matchups / elapsed
                remaining = (total_matchups - completed_matchups) / rate
                print(f"   Progress: {completed_matchups}/{total_matchups} ({completed_matchups/total_matchups*100:.1f}%) - "
                      f"ETA: {remaining/60:.1f}min")
        
        print(f"\n✅ Tournament completed in {(time.time() - start_time)/60:.1f} minutes")
        
        # Calculate rankings
        self._calculate_rankings(agent_names)
        
        # Generate visualizations
        self._generate_visualizations(agent_names)
        
        # Save results
        self._save_results(agent_names)
    
    def _calculate_rankings(self, agent_names: List[str]):
        """Calculate comprehensive rankings based on tournament results."""
        n_agents = len(agent_names)
        
        # Calculate various metrics
        overall_win_rates = []
        head_to_head_scores = []
        strength_of_schedule = []
        
        for i in range(n_agents):
            # Overall win rate (excluding self-matchup)
            other_indices = [j for j in range(n_agents) if j != i]
            win_rate = np.mean([self.win_rate_matrix[i][j] for j in other_indices])
            overall_win_rates.append(win_rate)
            
            # Head-to-head score (wins against others)
            h2h_score = sum(1 if self.win_rate_matrix[i][j] > 0.5 else 0 for j in other_indices)
            head_to_head_scores.append(h2h_score)
            
            # Strength of schedule (average quality of opponents beaten)
            beaten_opponents = [j for j in other_indices if self.win_rate_matrix[i][j] > 0.5]
            if beaten_opponents:
                # Ensure indices are valid
                valid_beaten = [j for j in beaten_opponents if j < len(overall_win_rates)]
                if valid_beaten:
                    sos = np.mean([overall_win_rates[j] for j in valid_beaten])
                else:
                    sos = 0.0
            else:
                sos = 0.0
            strength_of_schedule.append(sos)
        
        # Create ranking DataFrame
        self.rankings = pd.DataFrame({
            'Agent': agent_names,
            'Overall_Win_Rate': overall_win_rates,
            'Head_to_Head_Wins': head_to_head_scores,
            'Strength_of_Schedule': strength_of_schedule,
            'Games_Played': [self.games_per_matchup * (n_agents - 1)] * n_agents
        })
        
        # Calculate composite score (weighted combination)
        self.rankings['Composite_Score'] = (
            0.6 * self.rankings['Overall_Win_Rate'] +
            0.3 * self.rankings['Head_to_Head_Wins'] / (n_agents - 1) +
            0.1 * self.rankings['Strength_of_Schedule']
        )
        
        # Sort by composite score
        self.rankings = self.rankings.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
        self.rankings['Rank'] = range(1, n_agents + 1)
        
        # Print top 10
        print(f"\n🏆 TOP 10 RANKINGS")
        print("=" * 80)
        print(f"{'Rank':<4} {'Agent':<15} {'Win Rate':<10} {'H2H Wins':<8} {'SoS':<8} {'Score':<8}")
        print("-" * 80)
        
        for idx, row in self.rankings.head(10).iterrows():
            print(f"{row['Rank']:<4} {row['Agent']:<15} {row['Overall_Win_Rate']:<10.1%} "
                  f"{row['Head_to_Head_Wins']:<8.0f} {row['Strength_of_Schedule']:<8.3f} "
                  f"{row['Composite_Score']:<8.3f}")
    
    def _generate_visualizations(self, agent_names: List[str]):
        """Generate comprehensive visualizations."""
        print(f"\n📊 Generating visualizations...")
        
        # Create output directory
        os.makedirs("tournament_results", exist_ok=True)
        
        # 1. Win Rate Heatmap/Clustermap
        plt.figure(figsize=(14, 12))
        
        # Create DataFrame for seaborn
        win_rate_df = pd.DataFrame(
            self.win_rate_matrix,
            index=agent_names,
            columns=agent_names
        )
        
        # Generate clustermap
        g = sns.clustermap(
            win_rate_df,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0.5,
            square=True,
            figsize=(14, 12),
            cbar_kws={'label': 'Win Rate'},
            dendrogram_ratio=0.15
        )
        g.fig.suptitle('Agent Win Rate Matrix (Clustered)', fontsize=16, y=0.98)
        plt.savefig('tournament_results/win_rate_clustermap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Regular heatmap for reference
        plt.figure(figsize=(12, 10))
        mask = np.eye(len(agent_names), dtype=bool)  # Mask diagonal
        sns.heatmap(
            win_rate_df,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0.5,
            square=True,
            mask=mask,
            cbar_kws={'label': 'Win Rate'}
        )
        plt.title('Agent Win Rate Matrix', fontsize=16)
        plt.xlabel('Opponent')
        plt.ylabel('Agent')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('tournament_results/win_rate_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Rankings bar plot
        plt.figure(figsize=(12, 8))
        top_10 = self.rankings.head(10)
        
        bars = plt.bar(range(len(top_10)), top_10['Overall_Win_Rate'], 
                      color=plt.cm.viridis(np.linspace(0, 1, len(top_10))))
        
        plt.xlabel('Rank')
        plt.ylabel('Overall Win Rate')
        plt.title('Top 10 Agents by Overall Win Rate')
        plt.xticks(range(len(top_10)), [f"{i+1}. {name}" for i, name in enumerate(top_10['Agent'])], 
                   rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, win_rate in zip(bars, top_10['Overall_Win_Rate']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{win_rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('tournament_results/top_10_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Training progression (for M1-CNN models)
        m1_models = self.rankings[self.rankings['Agent'].str.contains('M1-CNN')].copy()
        if len(m1_models) > 1:
            # Extract episode numbers for sorting
            def extract_episodes(name):
                if 'Final' in name:
                    return 500000
                elif 'k' in name:
                    return int(name.split('-')[-1].replace('k', '')) * 1000
                return 0
            
            m1_models['Episodes'] = m1_models['Agent'].apply(extract_episodes)
            m1_models = m1_models.sort_values('Episodes')
            
            plt.figure(figsize=(12, 6))
            plt.plot(m1_models['Episodes']/1000, m1_models['Overall_Win_Rate'], 
                    marker='o', linewidth=2, markersize=8)
            plt.xlabel('Training Episodes (thousands)')
            plt.ylabel('Overall Win Rate')
            plt.title('M1-CNN Training Progression')
            plt.grid(True, alpha=0.3)
            
            # Annotate points
            for _, row in m1_models.iterrows():
                plt.annotate(f"{row['Overall_Win_Rate']:.1%}", 
                           (row['Episodes']/1000, row['Overall_Win_Rate']),
                           textcoords="offset points", xytext=(0,10), ha='center')
            
            plt.tight_layout()
            plt.savefig('tournament_results/m1_cnn_progression.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Visualizations saved to tournament_results/")
    
    def _save_results(self, agent_names: List[str]):
        """Save comprehensive results to files."""
        print(f"\n💾 Saving results...")
        
        # Save rankings
        self.rankings.to_csv('tournament_results/rankings.csv', index=False)
        
        # Save win rate matrix
        win_rate_df = pd.DataFrame(
            self.win_rate_matrix,
            index=agent_names,
            columns=agent_names
        )
        win_rate_df.to_csv('tournament_results/win_rate_matrix.csv')
        
        # Save detailed results as JSON
        results_json = {
            'tournament_config': {
                'games_per_matchup': self.games_per_matchup,
                'total_agents': len(agent_names),
                'agent_names': agent_names
            },
            'rankings': self.rankings.to_dict('records'),
            'win_rate_matrix': self.win_rate_matrix.tolist()
        }
        
        with open('tournament_results/tournament_results.json', 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print("✅ Results saved to tournament_results/")
        
        # Print summary statistics
        print(f"\n📈 TOURNAMENT SUMMARY")
        print("=" * 50)
        print(f"🥇 Champion: {self.rankings.iloc[0]['Agent']} ({self.rankings.iloc[0]['Overall_Win_Rate']:.1%})")
        print(f"🥈 Runner-up: {self.rankings.iloc[1]['Agent']} ({self.rankings.iloc[1]['Overall_Win_Rate']:.1%})")
        print(f"🥉 Third place: {self.rankings.iloc[2]['Agent']} ({self.rankings.iloc[2]['Overall_Win_Rate']:.1%})")
        
        baseline_rank = self.rankings[self.rankings['Agent'] == 'Heuristic']['Rank'].iloc[0]
        print(f"🎯 Heuristic baseline rank: #{baseline_rank}")
        
        best_m1 = self.rankings[self.rankings['Agent'].str.contains('M1-CNN')].iloc[0]
        print(f"🤖 Best M1-CNN: {best_m1['Agent']} (#{best_m1['Rank']}, {best_m1['Overall_Win_Rate']:.1%})")


def main():
    """Run the comprehensive model ranking tournament."""
    print("🏆 CONNECT 4 MODEL RANKING TOURNAMENT")
    print("=" * 60)
    
    try:
        tournament = ModelRankingTournament(games_per_matchup=100)
        tournament.run_tournament()
        
        print(f"\n🎉 Tournament completed successfully!")
        print(f"📊 Check 'tournament_results/' directory for detailed results and visualizations")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Tournament interrupted by user")
    except Exception as e:
        print(f"\n❌ Tournament failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()