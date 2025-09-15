#!/usr/bin/env python3
"""Simple launcher script for the interactive Connect 4 game."""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interactive.game_gui import main

if __name__ == "__main__":
    print("🎮 Starting Connect 4 Interactive Game...")
    print("Click on any column to drop your piece!")
    print("Red pieces are yours, yellow pieces are the AI.")
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Thanks for playing!")
    except Exception as e:
        print(f"❌ Error starting game: {e}")
        print("Make sure you have tkinter installed: pip install tkinter")
        sys.exit(1)