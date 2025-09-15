#!/usr/bin/env python3
"""Test script for the interactive Connect 4 game."""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interactive.game_gui import Connect4GUI

def test_game():
    """Test the interactive game functionality."""
    print("🎮 Testing Interactive Connect 4...")
    
    # Test with the final trained model
    model_path = "../models/double_dqn_final.pt"
    full_model_path = os.path.join(os.path.dirname(__file__), model_path)
    
    if not os.path.exists(full_model_path):
        print(f"❌ Model not found at {full_model_path}")
        print("Available models:")
        models_dir = os.path.join(os.path.dirname(__file__), "../models/")
        if os.path.exists(models_dir):
            models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
            for model in sorted(models)[-5:]:  # Show last 5 models
                print(f"  - {model}")
        return
    
    try:
        print(f"✅ Loading model: {os.path.basename(full_model_path)}")
        gui = Connect4GUI(full_model_path)
        print("✅ GUI initialized successfully")
        print("✅ Ready to launch interactive game")
        
        print("\n🎯 Game Info:")
        print(f"  Human player: {gui.human_player} (Red pieces)")
        print(f"  AI player: {gui.ai_player} (Yellow pieces)")
        print(f"  AI model loaded: {'Yes' if gui.ai_agent else 'No'}")
        if gui.ai_agent:
            print(f"  AI exploration: {gui.ai_agent.epsilon}")
            print(f"  AI player_id: {gui.ai_agent.player_id}")
        
        print("\n🚀 Starting interactive game...")
        gui.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_game()