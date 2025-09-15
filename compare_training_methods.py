#!/usr/bin/env python3
"""Compare original vs improved training methods."""

import os
import sys
import json
from typing import Dict, List
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulate_agents import GameSimulator


def find_models() -> Dict[str, List[str]]:
    """Find available models from both training methods."""
    models = {"original": [], "improved": []}
    
    # Original models
    if os.path.exists("models"):
        for file in os.listdir("models"):
            if file.endswith(".pt"):
                models["original"].append(f"models/{file}")
    
    # Improved models
    if os.path.exists("models_improved"):
        for file in os.listdir("models_improved"):
            if file.endswith(".pt"):
                models["improved"].append(f"models_improved/{file}")
    
    return models


def compare_models():
    """Compare performance of original vs improved training methods."""
    print("🔍 Comparing Original vs Improved Training Methods")
    print("=" * 60)
    
    models = find_models()
    
    # Key models to compare
    comparisons = [
        {
            "name": "Post-Heuristic Models",
            "original": "models/double_dqn_post_heuristic.pt",
            "improved": "models_improved/double_dqn_post_heuristic_improved.pt"
        },
        {
            "name": "Final Models", 
            "original": "models/double_dqn_final.pt",
            "improved": "models_improved/double_dqn_final_improved.pt"
        }
    ]
    
    results = {}
    
    for comparison in comparisons:
        print(f"\n📊 Testing {comparison['name']}")
        print("-" * 40)
        
        comparison_results = {}
        
        for method in ["original", "improved"]:
            model_path = comparison[method]
            
            if not os.path.exists(model_path):
                print(f"❌ {method.title()} model not found: {model_path}")
                continue
            
            print(f"\n🤖 {method.title()} Model: {os.path.basename(model_path)}")
            
            try:
                simulator = GameSimulator(model_path)
                
                # Test against heuristic (key metric)
                heuristic_stats = simulator.run_simulation("heuristic", 100, verbose=False)
                heuristic_win_rate = heuristic_stats['overall_results']['rl_win_rate']
                
                # Test against random (baseline)
                random_stats = simulator.run_simulation("random", 100, verbose=False)
                random_win_rate = random_stats['overall_results']['rl_win_rate']
                
                comparison_results[method] = {
                    'vs_heuristic': heuristic_win_rate,
                    'vs_random': random_win_rate,
                    'avg_game_length_vs_heuristic': heuristic_stats['game_length_analysis']['average_game_length'],
                    'model_path': model_path
                }
                
                print(f"  vs Heuristic: {heuristic_win_rate:.1%}")
                print(f"  vs Random:    {random_win_rate:.1%}")
                print(f"  Avg game length: {comparison_results[method]['avg_game_length_vs_heuristic']:.1f} moves")
                
            except Exception as e:
                print(f"❌ Error testing {method} model: {e}")
        
        results[comparison['name']] = comparison_results
    
    # Summary comparison
    print(f"\n🏆 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    
    for comp_name, comp_results in results.items():
        if len(comp_results) < 2:
            continue
            
        print(f"\n📊 {comp_name}")
        print("-" * 30)
        
        original = comp_results.get('original', {})
        improved = comp_results.get('improved', {})
        
        if not original or not improved:
            print("❌ Cannot compare - missing model(s)")
            continue
        
        # Heuristic performance comparison
        orig_heuristic = original['vs_heuristic']
        impr_heuristic = improved['vs_heuristic']
        heuristic_diff = impr_heuristic - orig_heuristic
        
        print(f"vs Heuristic:")
        print(f"  Original:  {orig_heuristic:.1%}")
        print(f"  Improved:  {impr_heuristic:.1%}")
        print(f"  Difference: {heuristic_diff:+.1%}", end="")
        
        if heuristic_diff > 0.1:
            print(" 🟢 SIGNIFICANT IMPROVEMENT")
        elif heuristic_diff > 0.05:
            print(" 🟡 MODERATE IMPROVEMENT") 
        elif heuristic_diff > -0.05:
            print(" ⚪ SIMILAR PERFORMANCE")
        else:
            print(" 🔴 PERFORMANCE DEGRADATION")
        
        # Random performance comparison
        orig_random = original['vs_random']
        impr_random = improved['vs_random']
        
        print(f"vs Random:")
        print(f"  Original:  {orig_random:.1%}")
        print(f"  Improved:  {impr_random:.1%}")
        
        # Game length comparison (shorter = more decisive)
        orig_length = original['avg_game_length_vs_heuristic']
        impr_length = improved['avg_game_length_vs_heuristic']
        length_diff = orig_length - impr_length
        
        print(f"Game Length vs Heuristic:")
        print(f"  Original:  {orig_length:.1f} moves")
        print(f"  Improved:  {impr_length:.1f} moves")
        if length_diff > 2:
            print(f"  → Improved is more decisive ({length_diff:+.1f} moves)")
        elif length_diff < -2:
            print(f"  → Original is more decisive ({length_diff:+.1f} moves)")
        else:
            print(f"  → Similar decisiveness")
    
    # Methodology comparison
    print(f"\n🔬 METHODOLOGY DIFFERENCES")
    print("=" * 40)
    print(f"Original Training:")
    print(f"  ❌ Clears replay buffer at self-play transition")
    print(f"  ❌ No heuristic preservation in self-play")
    print(f"  ❌ Fixed learning rate throughout")
    print(f"  ❌ No heuristic performance monitoring")
    print(f"")
    print(f"Improved Training:")
    print(f"  ✅ Smart buffer transition (preserves 30% of experiences)")
    print(f"  ✅ Guaranteed 20% heuristic games in self-play")
    print(f"  ✅ Reduced learning rate for self-play (0.0005)")
    print(f"  ✅ Automatic degradation detection and early stopping")
    print(f"  ✅ Heuristic performance monitoring every 500 episodes")
    
    print(f"\n💡 RECOMMENDATION:")
    if any(comp_results.get('improved', {}).get('vs_heuristic', 0) > 
           comp_results.get('original', {}).get('vs_heuristic', 0) + 0.05 
           for comp_results in results.values()):
        print(f"✅ Use improved training method - shows better heuristic knowledge retention")
    else:
        print(f"⚠️ Results inconclusive - both methods may have similar performance")
    
    return results


def main():
    """Main function."""
    results = compare_models()
    
    # Save comparison results
    with open("training_method_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Comparison results saved to: training_method_comparison.json")


if __name__ == "__main__":
    main()