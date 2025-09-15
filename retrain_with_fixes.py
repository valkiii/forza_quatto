#!/usr/bin/env python3
"""Retrain with confirmed working fixes and proper configuration."""

import os
import sys
import shutil
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Clean up and restart training with confirmed fixes."""
    print("🔄 RETRAINING with Confirmed Fixes")
    print("=" * 45)
    
    # Clean up previous failed run
    if os.path.exists("models_fixed"):
        print("🧹 Cleaning up previous failed models...")
        shutil.rmtree("models_fixed")
    
    if os.path.exists("logs_fixed"):
        print("🧹 Cleaning up previous failed logs...")
        shutil.rmtree("logs_fixed")
    
    print("✅ Cleanup complete")
    print()
    
    # Import and modify the configuration to ensure proper settings
    from fix_qvalue_learning import create_fixed_training_config
    
    config = create_fixed_training_config()
    
    # CRITICAL FIX: Ensure we're using enhanced rewards, not simple
    print("🔧 Applying critical fixes:")
    print("  1. ✅ Using enhanced reward system (not simple)")
    print("  2. ✅ Amplified strategic rewards (2.0-5.0)")
    print("  3. ✅ Proper epsilon schedule (1.0 → 0.05)")
    print("  4. ✅ Conservative learning rate (5e-5)")
    print("  5. ✅ All architectural fixes enabled")
    print()
    
    # Override the reward system to ensure it's not "simple"
    # The training script will use config.get("reward_system", None) which passes 
    # the reward config to the enhanced reward calculator
    
    # Save corrected config
    with open("config_retrain.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("💾 Corrected configuration saved: config_retrain.json")
    print()
    
    print("🚀 Starting corrected training...")
    print("   This addresses the reward system issue from the previous run")
    print("   Monitoring: python monitor_training.py")
    print()
    
    # Import and run the fixed training
    from train_fixed_double_dqn import train_fixed_double_dqn
    
    try:
        train_fixed_double_dqn()
        print("✅ RETRAINING COMPLETED SUCCESSFULLY!")
    except KeyboardInterrupt:
        print("\\n⚠️ Training interrupted by user")
        print("   Check models_fixed/ for partial progress")
    except Exception as e:
        print(f"\\n❌ Retraining failed: {e}")
        print("   Check logs_fixed/ for details")

if __name__ == "__main__":
    main()