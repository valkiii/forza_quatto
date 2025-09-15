# 🎮 Interactive Connect 4 - Human vs AI

A clean, clickable Connect 4 GUI for playing against the trained Double DQN agent.

## ✨ Features

- **Clickable Board**: Click any column to drop your piece
- **Visual Feedback**: Red pieces for human, yellow for AI
- **AI Integration**: Load and play against trained Double DQN models
- **Game Management**: New game, model loading, win detection
- **Clean UI**: Modern interface with status updates

## 🚀 Quick Start

### Option 1: Direct Launch
```bash
cd interactive
python play.py
```

### Option 2: From Root Directory
```bash
PYTHONPATH=/Users/vc/Research/forza_quattro python interactive/play.py
```

### Option 3: Import in Python
```python
from interactive.game_gui import Connect4GUI

# Start GUI (will auto-find trained models)
gui = Connect4GUI()
gui.run()

# Or specify a model path
gui = Connect4GUI("../models/double_dqn_final.pt")
gui.run()
```

## 🎯 How to Play

1. **Start Game**: Run the launcher - the GUI will open
2. **Your Turn**: Click any column to drop your red piece
3. **AI Turn**: The AI will automatically make its move with yellow pieces
4. **Win Condition**: Get 4 pieces in a row (horizontal, vertical, or diagonal)
5. **New Game**: Click "New Game" to restart
6. **Load Model**: Use "Load AI Model" to try different trained agents

## 🤖 AI Model Support

The game automatically searches for trained models in these locations:
- `../models/double_dqn_final.pt` (final trained model)
- `../models/double_dqn_ep_100000.pt` (curriculum milestone)  
- `../models/double_dqn_ep_200000.pt` (extended training)

If no model is found, you can:
1. Click "Load AI Model" to browse for a `.pt` file
2. Play in Human vs Human mode

## 🏗️ Architecture

**Reuses Existing Components:**
- `game/board.py` - Connect4 game logic and state management
- `agents/double_dqn_agent.py` - Trained Double DQN agent
- Clean separation between game logic, AI, and UI

**Files:**
- `game_gui.py` - Main GUI implementation with tkinter
- `play.py` - Simple launcher script
- `__init__.py` - Package initialization

## 🎨 UI Design

- **Colors**: Red (human), Yellow (AI), Gray (empty), Blue (board)
- **Feedback**: Status messages, turn indicators, hover effects
- **Controls**: Click to play, buttons for game management
- **Responsive**: Board updates in real-time, visual game state

## 🔧 Technical Details

- **Framework**: tkinter (built into Python)
- **Board Size**: Standard 6x7 Connect 4 grid  
- **AI Response**: 1-second delay for visual effect
- **Error Handling**: Graceful fallbacks for missing models
- **Cross-Platform**: Works on Windows, macOS, Linux

The interactive system provides a complete human vs AI gameplay experience while reusing all existing game logic and trained models.