#!/usr/bin/env python3
"""Plot enhanced training results from CSV logs."""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_training_progress(log_file: str = "logs_enhanced/enhanced_training_log.csv", 
                          output_dir: str = "logs_enhanced"):
    """Create comprehensive training plots from enhanced training logs."""
    
    if not os.path.exists(log_file):
        print(f"❌ Log file not found: {log_file}")
        print("Run training first with: python train_enhanced_comprehensive.py")
        return
    
    print(f"📊 Plotting enhanced training results from: {log_file}")
    
    # Read training data
    df = pd.read_csv(log_file)
    print(f"✅ Loaded {len(df)} training records")
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced Double DQN Training Progress', fontsize=16, fontweight='bold')
    
    # 1. Win Rates Over Time
    ax1 = axes[0, 0]
    ax1.plot(df['episode'], df['win_rate'], label='Current Opponent', color='blue', alpha=0.7)
    ax1.plot(df['episode'], df['vs_heuristic'], label='vs Heuristic', color='red', alpha=0.7)
    ax1.plot(df['episode'], df['vs_random'], label='vs Random', color='green', alpha=0.7)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Win Rate')
    ax1.set_title('Win Rates vs Different Opponents')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Average Reward
    ax2 = axes[0, 1]
    ax2.plot(df['episode'], df['avg_reward'], color='purple', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Episode Average Reward')
    ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon Decay
    ax3 = axes[0, 2]
    ax3.plot(df['episode'], df['epsilon'], color='orange', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Epsilon')
    ax3.set_title('Exploration Rate (Epsilon) Decay')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Training Steps
    ax4 = axes[1, 0]
    ax4.plot(df['episode'], df['training_steps'], color='brown', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Training Steps')
    ax4.set_title('Cumulative Training Steps')
    ax4.grid(True, alpha=0.3)
    
    # 5. Buffer Usage
    ax5 = axes[1, 1]
    ax5.plot(df['episode'], df['buffer_size'], color='teal', linewidth=2)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Buffer Size')
    ax5.set_title('Experience Replay Buffer Size')
    ax5.grid(True, alpha=0.3)
    
    # 6. Strategic Score
    ax6 = axes[1, 2]
    ax6.plot(df['episode'], df['strategic_score'], color='magenta', linewidth=2)
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Strategic Score')
    ax6.set_title('Strategic Play Quality')
    ax6.grid(True, alpha=0.3)
    
    # Add curriculum phase markers
    for ax in axes.flat:
        ax.axvline(x=50000, color='red', linestyle='--', alpha=0.5, label='Self-play Start')
        ax.axvline(x=200000, color='orange', linestyle='--', alpha=0.5, label='Heuristic End')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "enhanced_training_progress.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training progress plot saved: {plot_path}")
    
    # Show summary statistics
    print("\\n📈 TRAINING SUMMARY:")
    print(f"  Episodes completed: {df['episode'].max():,}")
    print(f"  Final win rate vs current: {df['win_rate'].iloc[-1]:.1%}")
    print(f"  Final win rate vs heuristic: {df['vs_heuristic'].iloc[-1]:.1%}")
    print(f"  Final win rate vs random: {df['vs_random'].iloc[-1]:.1%}")
    print(f"  Final epsilon: {df['epsilon'].iloc[-1]:.4f}")
    print(f"  Total training steps: {df['training_steps'].iloc[-1]:,}")
    print(f"  Final buffer size: {df['buffer_size'].iloc[-1]:,}")
    print(f"  Final strategic score: {df['strategic_score'].iloc[-1]:.2f}")
    
    # Create learning curves plot
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Smooth the win rates for better visualization
    window = max(1, len(df) // 50)  # Adaptive smoothing
    
    def smooth_curve(data, window_size):
        return pd.Series(data).rolling(window=window_size, min_periods=1).mean()
    
    smooth_current = smooth_curve(df['win_rate'], window)
    smooth_heuristic = smooth_curve(df['vs_heuristic'], window)
    smooth_random = smooth_curve(df['vs_random'], window)
    
    ax.plot(df['episode'], smooth_current, label='vs Current Opponent', linewidth=3, color='blue')
    ax.plot(df['episode'], smooth_heuristic, label='vs Heuristic Agent', linewidth=3, color='red')
    ax.plot(df['episode'], smooth_random, label='vs Random Agent', linewidth=3, color='green')
    
    # Add curriculum phases
    ax.axvspan(0, 50000, alpha=0.2, color='yellow', label='Random Phase')
    ax.axvspan(50000, 200000, alpha=0.2, color='orange', label='Heuristic Phase')
    ax.axvspan(200000, df['episode'].max(), alpha=0.2, color='purple', label='Self-play Phase')
    
    ax.set_xlabel('Training Episode', fontsize=12)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_title('Enhanced Double DQN Learning Curves\\n(Smoothed)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Save learning curves
    curves_path = os.path.join(output_dir, "enhanced_learning_curves.png")
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    print(f"✅ Learning curves plot saved: {curves_path}")
    
    # Create enhancement comparison plot if we have the data
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Enhanced Features Analysis', fontsize=16, fontweight='bold')
    
    # N-step learning indicator
    ax_nstep = axes3[0, 0]
    n_step_values = df['n_step'].unique()
    ax_nstep.bar(range(len(n_step_values)), [len(df[df['n_step'] == n]) for n in n_step_values])
    ax_nstep.set_title('N-Step Learning Usage')
    ax_nstep.set_xlabel('N-Step Value')
    ax_nstep.set_ylabel('Episodes')
    
    # Prioritized replay indicator  
    ax_prioritized = axes3[0, 1]
    prioritized_counts = df['prioritized'].value_counts()
    ax_prioritized.pie(prioritized_counts.values, labels=prioritized_counts.index, autopct='%1.1f%%')
    ax_prioritized.set_title('Prioritized Replay Usage')
    
    # Strategic score progression
    ax_strategic = axes3[1, 0]
    ax_strategic.scatter(df['episode'], df['strategic_score'], alpha=0.6, s=10)
    ax_strategic.plot(df['episode'], smooth_curve(df['strategic_score'], window), 
                     color='red', linewidth=2, label='Smoothed Trend')
    ax_strategic.set_xlabel('Episode')
    ax_strategic.set_ylabel('Strategic Score')
    ax_strategic.set_title('Strategic Play Development')
    ax_strategic.legend()
    ax_strategic.grid(True, alpha=0.3)
    
    # Win rate improvement over phases
    ax_phases = axes3[1, 1]
    phase_data = []
    phases = ['Random\\n(0-50K)', 'Heuristic\\n(50K-200K)', 'Self-play\\n(200K+)']
    
    # Calculate average win rates for each phase
    random_phase = df[df['episode'] <= 50000]
    heuristic_phase = df[(df['episode'] > 50000) & (df['episode'] <= 200000)]
    selfplay_phase = df[df['episode'] > 200000]
    
    phase_winrates = []
    if len(random_phase) > 0:
        phase_winrates.append(random_phase['win_rate'].mean())
    if len(heuristic_phase) > 0:
        phase_winrates.append(heuristic_phase['win_rate'].mean())
    if len(selfplay_phase) > 0:
        phase_winrates.append(selfplay_phase['win_rate'].mean())
    
    colors = ['yellow', 'orange', 'purple']
    ax_phases.bar(phases[:len(phase_winrates)], phase_winrates, color=colors[:len(phase_winrates)])
    ax_phases.set_ylabel('Average Win Rate')
    ax_phases.set_title('Performance by Curriculum Phase')
    ax_phases.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save enhancements plot
    enhance_path = os.path.join(output_dir, "enhanced_features_analysis.png")
    plt.savefig(enhance_path, dpi=300, bbox_inches='tight')
    print(f"✅ Enhancement analysis plot saved: {enhance_path}")
    
    plt.show()
    print("\\n🎉 All plots generated successfully!")

def main():
    """Main function to generate plots."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot enhanced training results')
    parser.add_argument('--log-file', default='logs_enhanced/enhanced_training_log.csv',
                       help='Path to training log CSV file')
    parser.add_argument('--output-dir', default='logs_enhanced',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    plot_training_progress(args.log_file, args.output_dir)

if __name__ == "__main__":
    main()