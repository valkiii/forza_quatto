# Connect 4 Reinforcement Learning Project

A comprehensive Connect 4 implementation with advanced reinforcement learning agents, specifically designed to explore and solve catastrophic forgetting in deep RL. Features state-of-the-art techniques including Dueling DQN, anti-catastrophic forgetting measures, and strategic reward systems.

## 🚀 Key Features

### Advanced RL Techniques
- **Dueling DQN Architecture**: Separates state value from action advantages for better learning
- **Anti-Catastrophic Forgetting**: Progressive curriculum learning with experience replay diversity
- **Strategic Reward System**: Amplified intermediate rewards (2.0-5.0x) for tactical learning
- **Q-value Explosion Prevention**: Comprehensive stabilization with gradient clipping and clamping

### Training Stability
- **Smooth Curriculum Transitions**: 4000-episode overlap periods prevent performance drops
- **Polyak Averaging**: Conservative target network updates (τ=0.001)
- **Reservoir Sampling**: Maintains experience diversity across training phases
- **Learning Rate Warmup**: Gradual increase prevents early training instability

## Project Structure

```
forza_quattro/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── CLAUDE.md                         # Development guidelines
├── game/
│   └── board.py                      # Core Connect 4 game logic
├── agents/
│   ├── random_agent.py               # Random baseline agent
│   ├── heuristic_agent.py            # Strategic rule-based agent
│   └── double_dqn_agent.py           # Advanced Deep RL agent
├── train/
│   ├── reward_system.py              # Enhanced strategic rewards
│   ├── training_monitor.py           # Training visualization and logging
│   ├── double_dqn_train.py           # Standard training pipeline
│   └── double_dqn_train_advanced.py  # Advanced training configurations
├── train_fixed_double_dqn.py         # Main training script with all fixes
├── fix_qvalue_learning.py            # Configuration for Q-value stability
├── interactive_game.py               # Human vs AI gameplay
├── evaluate_agents.py                # Agent performance evaluation
├── docs/                             # Development documentation
├── models_fixed/                     # Trained model checkpoints
└── logs_fixed/                       # Training logs and plots
```

## Installation

1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Quick Start

### Train Advanced Double DQN Agent
```bash
python train_fixed_double_dqn.py
```

### Play Against AI
```bash
python interactive_game.py
```

### Evaluate Agent Performance
```bash
python evaluate_agents.py
```

### Monitor Training Progress
```bash
# View real-time logs
tail -f logs_fixed/double_dqn_log.csv

# Check training plots
ls logs_fixed/*.png
```

## 🧠 Technical Details

### Enhanced Reward System
- **Terminal Rewards**: Win (+10.0), Loss (-10.0), Draw (0.0)
- **Strategic Rewards** (Amplified 10-25x for learning visibility):
  - Blocked Opponent: +2.0 (was 0.1)
  - Played Winning Move: +3.0 (was 0.2)
  - Missed Block: -3.0 (was -0.1)
  - Missed Win: -5.0 (was -0.2)

### Dueling DQN Architecture
```
Network (256 hidden units):
├── Shared Features: 84 → 256 → 256 (dropout 0.1)
├── Value Stream: 256 → 128 → 1
├── Advantage Stream: 256 → 128 → 7
└── Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
```

### Training Configuration
- **Learning Rate**: 5e-5 (ultra-conservative)
- **Discount Factor**: 0.95 (prevents Q-value explosion)
- **Batch Size**: 64
- **Buffer Size**: 200,000
- **Target Update**: Every 2000 steps with Polyak averaging (τ=0.001)

## 📊 Anti-Catastrophic Forgetting Features

1. **Q-value Explosion Prevention**:
   - Reduced discount factor (0.95 vs 0.99)
   - Q-value clamping [-100, 100]
   - Gradient clipping (norm=1.0)
   - Proper network initialization

2. **Curriculum Learning**:
   - Smooth phase transitions with 4000-episode overlaps
   - Progressive difficulty: Random → Heuristic → Mixed
   - Reservoir sampling for experience diversity

3. **Strategic Learning Retention**:
   - Amplified intermediate rewards for visibility
   - Immediate tactical feedback
   - Anti-forgetting experience replay

## 🔬 Research Applications

This codebase is designed for studying:
- **Catastrophic Forgetting in Deep RL**: Comprehensive analysis and mitigation strategies
- **Curriculum Learning**: Progressive difficulty with smooth transitions
- **Strategic Game Learning**: Connect4-specific tactical decision making
- **Multi-agent Training Dynamics**: Agent behavior across different opponents

## 📈 Expected Results

With the implemented fixes, training demonstrates:
- ✅ Stable Q-values (range -20 to +20, no explosion)
- ✅ Smooth curriculum transitions (no sudden performance drops)
- ✅ Strategic learning retention (>50% win rate vs heuristic)
- ✅ Convergence without catastrophic oscillations

## Development Milestones

- ✅ **Milestone 1**: Core game engine + strategic agents
- ✅ **Milestone 2**: Tabular Q-learning implementation
- ✅ **Milestone 3**: Deep Q-Network with PyTorch
- ✅ **Milestone 4**: Double DQN with experience replay
- ✅ **Milestone 5**: Dueling DQN architecture
- ✅ **Milestone 6**: Anti-catastrophic forgetting system
- ✅ **Milestone 7**: Advanced training stability features

See `docs/` folder for detailed development documentation.

## 🤝 Contributing

This project follows modular design principles:
- Extend existing components rather than creating new files
- Reference existing code patterns and architectures
- Maintain compatibility with current training pipelines
- Follow the guidelines in `CLAUDE.md`

## Dependencies

Core requirements:
- `torch`: PyTorch for neural networks and training
- `numpy`: Efficient array operations for game state
- `matplotlib`: Training visualization and plotting

See `requirements.txt` for complete dependency list.

## 🏆 Project Achievements

- **Strategic AI**: Human-competitive Connect4 gameplay
- **Stable Training**: Solved Q-value explosion and catastrophic forgetting
- **Research Platform**: Ready for RL research and experimentation
- **Modular Architecture**: Extensible and maintainable codebase

---

*This implementation demonstrates advanced RL techniques applied to a classic game, serving as both a research platform and educational resource for understanding deep reinforcement learning challenges and solutions.*