#!/usr/bin/env python3
"""Monitor comprehensive training progress."""

import os
import json
import time
from datetime import datetime

def monitor_training():
    """Monitor the comprehensive training progress."""
    log_dir = "logs_fixed"
    
    print("🔍 COMPREHENSIVE TRAINING MONITOR")
    print("=" * 40)
    print(f"Monitoring: {log_dir}/")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if training is running
    models_dir = "models_fixed"
    if not os.path.exists(models_dir):
        print("❌ Training not started yet - no models_fixed/ directory")
        return
    
    # List recent models
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    if model_files:
        model_files.sort()
        print(f"📁 Models saved: {len(model_files)}")
        print(f"   Latest: {model_files[-1]}")
        
        # Extract episode numbers
        episodes = []
        for f in model_files:
            if 'ep_' in f:
                try:
                    ep_num = int(f.split('ep_')[1].split('.pt')[0])
                    episodes.append(ep_num)
                except:
                    pass
        
        if episodes:
            episodes.sort()
            print(f"   Episode range: {min(episodes):,} - {max(episodes):,}")
            
            # Estimate curriculum phase
            latest_ep = max(episodes)
            if latest_ep <= 20000:
                phase = "Random Phase (Foundation)"
                progress = f"{latest_ep/20000*100:.1f}%"
            elif latest_ep <= 80000:
                phase = "Heuristic Phase (Strategic Learning)"
                progress = f"{(latest_ep-20000)/60000*100:.1f}%"
            else:
                phase = "Mixed Phase (Self-play & Preservation)"
                progress = f"{min((latest_ep-80000)/50000*100, 100):.1f}%"
            
            print(f"   📚 Current phase: {phase}")
            print(f"   📊 Phase progress: {progress}")
        print()
    else:
        print("📁 No model files found yet")
        print()
    
    # Check log files
    log_files = []
    if os.path.exists(log_dir):
        for f in os.listdir(log_dir):
            if f.endswith('.json') and ('analysis' in f or 'summary' in f):
                log_files.append(f)
        
        if log_files:
            print(f"📊 Analysis files: {len(log_files)}")
            for f in sorted(log_files):
                print(f"   - {f}")
            print()
    
    # Check for any failure analysis
    failure_file = os.path.join(log_dir, "catastrophic_forgetting_analysis.json")
    if os.path.exists(failure_file):
        print("🚨 FAILURE DETECTED!")
        try:
            with open(failure_file, 'r') as f:
                analysis = json.load(f)
            print(f"   Episode: {analysis.get('episode', 'Unknown')}")
            print(f"   Performance: {analysis.get('heuristic_performance', 0):.1%}")
            print(f"   Threshold: {analysis.get('threshold', 0):.1%}")
            print("   Training stopped to prevent catastrophic forgetting")
        except:
            print("   Failed to read analysis file")
        print()
    
    # Training status
    csv_log = os.path.join(log_dir, "double_dqn_log.csv")
    if os.path.exists(csv_log):
        try:
            with open(csv_log, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 1:
                last_line = lines[-1].strip()
                if last_line:
                    parts = last_line.split(',')
                    if len(parts) >= 6:
                        episode = int(parts[0])
                        avg_reward = float(parts[1])
                        epsilon = float(parts[2])
                        training_steps = int(parts[3])
                        buffer_size = int(parts[4])
                        win_rate = float(parts[5])
                        
                        print(f"📈 LATEST TRAINING STATS:")
                        print(f"   Episode: {episode:,}")
                        print(f"   Win rate: {win_rate:.1%}")
                        print(f"   Avg reward: {avg_reward:.2f}")
                        print(f"   Epsilon: {epsilon:.4f}")
                        print(f"   Buffer: {buffer_size:,} experiences")
                        print(f"   Training steps: {training_steps:,}")
                        
                        # Training efficiency
                        if episode > 0:
                            steps_per_episode = training_steps / episode
                            print(f"   Efficiency: {steps_per_episode:.1f} training steps/episode")
                        print()
        except Exception as e:
            print(f"❌ Error reading CSV log: {e}")
            print()
    else:
        print("📊 No CSV log file found yet")
        print()
    
    print("💡 MONITORING COMMANDS:")
    print("   tail -f logs_fixed/double_dqn_log.csv  # Live training log")
    print("   ls -la models_fixed/                   # Model checkpoints")
    print("   python monitor_training.py             # Run this script again")
    print("   python simulate_agents.py              # Test latest model")

if __name__ == "__main__":
    monitor_training()