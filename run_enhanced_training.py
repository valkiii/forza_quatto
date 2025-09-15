#!/usr/bin/env python3
"""Quick start script for enhanced comprehensive training."""

import os
import sys

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run enhanced training with all improvements."""
    print("🚀 ENHANCED CONNECT 4 RL TRAINING")
    print("=" * 40)
    print("This script implements ALL suggested improvements:")
    print("✅ Larger network (512 hidden units)")
    print("✅ Dueling architecture") 
    print("✅ N-step returns (3-step learning)")
    print("✅ Prioritized experience replay")
    print("✅ Strategic state features")
    print("✅ Extended training (500K episodes)")
    print("✅ Early self-play (starts at 50K)")
    print("✅ Polyak averaging")
    print("✅ Enhanced curriculum")
    print("✅ MCTS evaluation capability")
    print()
    
    print("🎯 Expected Training Results:")
    print("• Much cleverer strategic play")
    print("• Better long-term planning")
    print("• Improved threat detection")
    print("• Enhanced center control")
    print("• Stronger self-play performance")
    print()
    
    print("📊 Monitoring:")
    print("• Logs: logs_enhanced/enhanced_training_log.csv")
    print("• Models: models_enhanced/")
    print("• Final results: logs_enhanced/final_results.json")
    print()
    
    response = input("Start enhanced training? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    print("🚀 Starting enhanced training...")
    print("This will take significantly longer but produce much better results!")
    print()
    
    try:
        from train_enhanced_comprehensive import train_enhanced_agent
        train_enhanced_agent()
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        print("Partial progress saved in models_enhanced/")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()