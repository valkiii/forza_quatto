#!/usr/bin/env python3
"""Test the plotting fixes."""

import sys
import os
sys.path.append('.')

def test_plotting_fix():
    """Test that plotting scales are fixed."""
    print("🧪 TESTING PLOTTING FIXES")
    print("=" * 35)
    
    try:
        from train.training_monitor import TrainingMonitor
        from train_fixed_double_dqn import FixedDoubleDQNAgent
        import numpy as np
        
        # Create test monitor with eval_frequency
        monitor = TrainingMonitor(log_dir="test_plots", save_plots=True, eval_frequency=250)
        
        # Create test agent
        agent = FixedDoubleDQNAgent(player_id=1, state_size=84, action_size=7, seed=42)
        
        # Simulate some training data with realistic scales
        print("📊 Simulating training data...")
        
        # Simulate 10 evaluation points (every 250 episodes)
        for i in range(10):
            episode = (i + 1) * 250  # Episodes: 250, 500, 750, ..., 2500
            
            # Add win rate data (evaluated every 250 episodes)
            monitor.win_rates.append(0.5 + 0.3 * np.sin(i * 0.5))  # Simulated win rate progression
            
            # Add exploration data (recorded every episode, so add 250 points)
            if i == 0:
                start_eps = 1.0
            else:
                start_eps = monitor.exploration_rates[-1]
                
            for ep in range(250):
                current_episode = i * 250 + ep + 1
                # Realistic epsilon decay: 0.99995 per episode
                epsilon = max(0.05, start_eps * (0.99995 ** ep)) 
                monitor.exploration_rates.append(epsilon)
        
        print(f"✅ Generated {len(monitor.win_rates)} win rate points")
        print(f"✅ Generated {len(monitor.exploration_rates)} epsilon points")
        print(f"   Win rates span episodes: 250, 500, ..., {len(monitor.win_rates) * 250}")
        print(f"   Epsilon spans episodes: 1, 2, ..., {len(monitor.exploration_rates)}")
        
        # Test plotting
        print("🎨 Testing plot generation...")
        monitor._generate_plots(2500, agent)
        
        print("✅ Plot generation successful!")
        print("📁 Check test_plots/ directory for generated plot")
        
        # Check if plot file was created
        plot_file = "test_plots/training_progress_ep_2500.png"
        if os.path.exists(plot_file):
            print(f"✅ Plot file created: {plot_file}")
            print("🔍 The Win Rate vs Exploration plot should now have:")
            print("   - Win rates plotted at episodes 250, 500, 750, ..., 2500")
            print("   - Epsilon plotted at episodes 1, 2, 3, ..., 2500") 
            print("   - Both scales should align properly")
        else:
            print("❌ Plot file not created")
            
        # Cleanup test directory
        import shutil
        if os.path.exists("test_plots"):
            shutil.rmtree("test_plots")
            print("🧹 Cleaned up test files")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_plotting_fix()