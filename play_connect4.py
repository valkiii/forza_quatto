#!/usr/bin/env python3
"""Simple launcher for Connect 4 game."""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🎮 Connect 4: Human vs AI")
    print("=" * 30)
    print("Click any column to drop your RED chip")
    print("AI will respond with YELLOW chips")
    print("Get 4 in a row to win!")
    print()
    
    try:
        from interactive.game_gui import Connect4GUI
        
        # Look for the best available model (prioritize M1 CNN, then other CNNs, then enhanced)
        model_candidates = [
            "models_m1_cnn/m1_cnn_dqn_final.pt",
            "models_m1_cnn/m1_cnn_dqn_best_ep_*.pt",
            "models_m1_cnn/m1_cnn_dqn_ep_*.pt",
            "models_cnn/cnn_dqn_final.pt",
            "models_cnn/cnn_dqn_best_ep_*.pt",
            "models_cnn/cnn_dqn_ep_*.pt",
            "models_enhanced/enhanced_dqn_final.pt",
            "models_enhanced/enhanced_dqn_best_ep_*.pt",
            "models_enhanced/enhanced_dqn_ep_*.pt"
        ]
        
        model_path = None
        import glob
        
        for candidate in model_candidates:
            if "*" in candidate:
                # Handle glob patterns for models with episode numbers
                matching_files = glob.glob(candidate)
                if matching_files:
                    # Sort by modification time, get the newest
                    model_path = max(matching_files, key=os.path.getmtime)
                    break
            elif os.path.exists(candidate):
                model_path = candidate
                break
        
        if model_path:
            print(f"🤖 Loading AI: {os.path.basename(model_path)}")
        else:
            print("⚠️  No trained AI found - Human vs Human mode")
        
        # Launch the game
        gui = Connect4GUI(model_path)
        gui.run()
        
    except KeyboardInterrupt:
        print("\n👋 Thanks for playing!")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Try: pip install tkinter")
    except Exception as e:
        print(f"❌ Error starting game: {e}")
        print("Make sure you're in the correct directory")

if __name__ == "__main__":
    main()