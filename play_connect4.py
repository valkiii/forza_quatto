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
        
        # Look for the best available model (prioritize trained models)
        model_candidates = [
            "models_fixed/double_dqn_ep_150000.pt"
        ]
        
        model_path = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
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