#!/usr/bin/env python3
"""Simple launcher for Connect 4 agent simulations."""

import os
import sys

def main():
    print("🎯 Connect 4 Agent Simulation Launcher")
    print("=" * 45)
    
    # Check if we have a trained model
    model_candidates = [
        "models/double_dqn_best_ep_*.pt",
        "models/double_dqn_post_heuristic.pt", 
        "models/double_dqn_ep_60000.pt",
        "models/double_dqn_ep_52000.pt",
        "models/double_dqn_final.pt"
    ]
    
    found_model = False
    for pattern in model_candidates:
        if '*' in pattern:
            import glob
            matches = glob.glob(pattern)
            if matches:
                found_model = True
                break
        else:
            if os.path.exists(pattern):
                found_model = True
                break
    
    if not found_model:
        print("❌ No trained model found!")
        print("   Please train a model first using:")
        print("   python train/double_dqn_train.py")
        return
    
    print("Select simulation type:")
    print("1. Quick test vs Heuristic (100 games)")
    print("2. Standard test vs Heuristic (1000 games)")
    print("3. Test vs Random opponent (1000 games)")
    print("4. Test vs Self (RL vs RL) (1000 games)")
    print("5. Comprehensive test vs ALL opponents (1000 games each)")
    print("6. Custom simulation")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                cmd = "python simulate_agents.py --opponent heuristic --games 100"
                break
            elif choice == '2':
                cmd = "python simulate_agents.py --opponent heuristic --games 1000 --save-results"
                break
            elif choice == '3':
                cmd = "python simulate_agents.py --opponent random --games 1000 --save-results"
                break
            elif choice == '4':
                cmd = "python simulate_agents.py --opponent self --games 1000 --save-results"
                break
            elif choice == '5':
                cmd = "python simulate_agents.py --all-opponents --games 1000 --save-results"
                break
            elif choice == '6':
                print("\nCustom simulation options:")
                opponent = input("Opponent type (random/heuristic/self): ").strip()
                games = input("Number of games (default 1000): ").strip() or "1000"
                save = input("Save results? (y/n): ").strip().lower() == 'y'
                
                if opponent not in ['random', 'heuristic', 'self']:
                    print("❌ Invalid opponent type")
                    continue
                    
                cmd = f"python simulate_agents.py --opponent {opponent} --games {games}"
                if save:
                    cmd += " --save-results"
                break
            else:
                print("❌ Invalid choice, please enter 1-6")
        except KeyboardInterrupt:
            print("\n👋 Cancelled")
            return
    
    print(f"\n🚀 Running: {cmd}")
    print("-" * 50)
    os.system(cmd)

if __name__ == "__main__":
    main()