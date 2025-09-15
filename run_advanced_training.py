#!/usr/bin/env python3
"""Launcher for advanced training with aggressive heuristic preservation."""

import os
import sys

def main():
    print("🚀 Advanced Connect 4 Training Launcher")
    print("=" * 50)
    
    print("Select training method:")
    print("1. 🚀 LATEST: Fixed Training (MOST RECOMMENDED)")
    print("   - ✅ Fixes Q-value learning issues (stable gradients)")
    print("   - ✅ 30%+ heuristic preservation (never below!)")
    print("   - ✅ Enhanced state encoding & gradient clipping")
    print("   - ✅ Huber loss, better hyperparameters")
    print("")
    print("2. 🟢 Advanced Training (Heuristic preservation focus)")
    print("   - ✅ All 7 expert suggestions implemented")
    print("   - ✅ 10x learning rate reduction for self-play")
    print("   - ✅ Gradual self-play introduction")
    print("   - ✅ Historical opponent pool & early stopping")
    print("")
    print("3. 🔴 Original Training (Has catastrophic forgetting)")
    print("   - ❌ No heuristic preservation")
    print("   - ❌ Buffer clearing at transitions")
    print("   - ❌ Will likely fail vs heuristic after self-play")
    print("")
    print("4. 📊 Compare Training Methods")
    print("   - Run simulations on existing models")
    print("   - Compare original vs improved approaches")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\n🚀 Starting LATEST FIXED training with Q-value learning fixes...")
                print("Features:")
                print("✅ FIXES Q-value learning issues (gradient clipping, stable training)")
                print("✅ Enhanced state encoding (better positional awareness)")
                print("✅ Huber loss & optimized hyperparameters")
                print("✅ 30%+ heuristic preservation (never drops below)")
                print("✅ All catastrophic forgetting prevention techniques")
                print("✅ Comprehensive monitoring with improved plots")
                print("")
                print("This is the MOST ADVANCED version that addresses:")
                print("- Exploding gradients (215+ norm → <10 norm)")
                print("- Poor Q-value discrimination")
                print("- Strategic decision making")
                print("- Catastrophic forgetting (100% → 40% drop)")
                print("")
                input("Press Enter to start fixed training...")
                os.system("python train_fixed_double_dqn.py")
                break
                
            elif choice == '2':
                print("\n🟢 Starting ADVANCED training with aggressive heuristic preservation...")
                print("Features:")
                print("✅ All 7 expert suggestions implemented")
                print("✅ 10x learning rate reduction (fine-tuning mode)")
                print("✅ Gradual self-play introduction (30% → 60%)")
                print("✅ Historical opponent pool (7 diverse agents)")
                print("✅ Emergency stop at first sign of degradation")
                print("✅ L2 regularization and advanced monitoring")
                print("")
                print("This method focuses on heuristic preservation but may still")
                print("have Q-value learning quality issues.")
                input("Press Enter to start advanced training...")
                os.system("python train/double_dqn_train_advanced.py")
                break
                
            elif choice == '3':
                print("\n🔴 WARNING: Original training has catastrophic forgetting!")
                confirm = input("Are you sure you want to use the problematic method? (y/N): ")
                if confirm.lower() == 'y':
                    os.system("python train/double_dqn_train.py")
                break
                
            elif choice == '4':
                print("\n📊 Running training method comparison...")
                os.system("python compare_training_methods.py")
                break
                
            else:
                print("❌ Invalid choice, please enter 1-4")
                
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return
    
    print("\n✅ Training completed!")
    print("\nNext steps:")
    print("1. Check logs_advanced/ for detailed training analytics")
    print("2. Run: python simulate_agents.py --all-opponents --save-results")
    print("3. Test human gameplay: python play_connect4.py")


if __name__ == "__main__":
    main()