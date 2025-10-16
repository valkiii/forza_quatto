#!/usr/bin/env python3
"""Plot all win rates (vs Random, vs Heuristic, vs League) on the same graph."""

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys


def plot_all_win_rates(csv_path, output_path=None, title="Training Win Rates Over Time"):
    """
    Plot all win rates on the same graph.

    Args:
        csv_path: Path to the training log CSV
        output_path: Path to save the plot (if None, will display)
        title: Title for the plot
    """
    # Read CSV
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Check required columns
    required_cols = ['episode', 'win_rate', 'vs_heuristic']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV")
            print(f"Available columns: {df.columns.tolist()}")
            return

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot win rates
    ax.plot(df['episode'], df['win_rate'], 'b-', linewidth=2,
            label='vs Random', marker='o', markersize=3, alpha=0.7)
    ax.plot(df['episode'], df['vs_heuristic'], 'r-', linewidth=2,
            label='vs Heuristic', marker='s', markersize=3, alpha=0.7)

    # Plot vs_league if available (column might be vs_random or vs_league)
    if 'vs_league' in df.columns:
        ax.plot(df['episode'], df['vs_league'], 'g-', linewidth=2,
                label='vs League Champion', marker='^', markersize=3, alpha=0.7)
    elif 'vs_random' in df.columns and 'vs_random' != 'win_rate':
        # If there's a vs_random column that's different from win_rate, plot it
        ax.plot(df['episode'], df['vs_random'], 'g-', linewidth=2,
                label='vs League (old format)', marker='^', markersize=3, alpha=0.7)

    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% (baseline)')
    ax.axhline(y=0.6, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='60% (good)')
    ax.axhline(y=0.3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='30% (critical)')

    # Formatting
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_ylim(-0.05, 1.05)

    # Add statistics text box
    latest_episode = df['episode'].iloc[-1]
    latest_vs_random = df['win_rate'].iloc[-1]
    latest_vs_heuristic = df['vs_heuristic'].iloc[-1]

    stats_text = f"Latest Performance (Episode {latest_episode:,}):\n"
    stats_text += f"  vs Random: {latest_vs_random:.1%}\n"
    stats_text += f"  vs Heuristic: {latest_vs_heuristic:.1%}"

    if 'vs_league' in df.columns:
        latest_vs_league = df['vs_league'].iloc[-1]
        stats_text += f"\n  vs League: {latest_vs_league:.1%}"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main function to run from command line."""
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "logs_m1_cnn/m1_cnn_training_log.csv"

    # Determine output path
    log_dir = os.path.dirname(csv_path)
    if not log_dir:
        log_dir = "logs_m1_cnn"
    output_path = os.path.join(log_dir, "all_win_rates_plot.png")

    print(f"📊 Plotting win rates from: {csv_path}")
    plot_all_win_rates(csv_path, output_path)


if __name__ == "__main__":
    main()
