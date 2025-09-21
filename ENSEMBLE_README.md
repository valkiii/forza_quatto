# 🤖 Connect 4 Ensemble Agent System

A powerful ensemble system that combines multiple trained RL models for improved Connect 4 performance. The ensemble leverages the strengths of different models while mitigating individual weaknesses.

## 📁 Files Overview

- **`ensemble_agent.py`** - Core ensemble agent implementation
- **`play_ensemble.py`** - Interactive game interface
- **`examples/`** - Example configuration files
  - `ensemble_config_top.json` - Top 4 performers from tournament
  - `ensemble_config_diverse.json` - Diverse architectures

## 🚀 Quick Start

### Play Against Preset Ensembles

```bash
# Best tournament performers (recommended)
python play_ensemble.py --preset top_performers

# Diverse model architectures
python play_ensemble.py --preset diverse_architectures

# Training evolution stages
python play_ensemble.py --preset evolution_stages
```

### Show Model Contributions

```bash
# See detailed decision breakdown (enabled by default for presets)
python play_ensemble.py --preset top_performers --show-contributions

# Hide contributions for cleaner display
python play_ensemble.py --preset top_performers --hide-contributions
```

## 🔬 Ensemble Methods

### 1. **Weighted Voting** - Democratic Decision Making

**How it works:**
- Each model votes for its preferred action
- Votes are weighted by model importance
- Action with most weighted votes wins

**Example Decision:**
```
🧠 Ensemble Decision Breakdown:
   Method: weighted_voting
   Chosen Action: Column 3

📊 Vote Distribution:
   Column 3: 0.800 votes
   Column 1: 0.200 votes

🤖 Individual Model Votes:
   • M1-CNN-550k: Column 3 (weight: 0.400)
   • M1-CNN-500k: Column 3 (weight: 0.300)
   • M1-CNN-450k: Column 1 (weight: 0.200)
   • M1-CNN-400k: Column 3 (weight: 0.100)
```

**Best for:** Diverse model types, clear democratic consensus

---

### 2. **Q-Value Averaging** - Collaborative Intelligence

**How it works:**
- Each model provides quality scores (Q-values) for all actions
- Q-values are weighted and averaged across models
- Action with highest averaged score wins

**Example Decision:**
```
🧠 Ensemble Decision Breakdown:
   Method: q_value_averaging
   Chosen Action: Column 3

📊 Averaged Q-Values:
   Column 0: 0.245
   Column 1: 0.512
   Column 3: 0.834
   Column 4: 0.445

🤖 Individual Model Preferences:
   • M1-CNN-550k: Column 3 (Q: 0.923, weight: 0.400)
   • M1-CNN-500k: Column 3 (Q: 0.856, weight: 0.300)
   • M1-CNN-450k: Column 1 (Q: 0.678, weight: 0.200)
   • M1-CNN-400k: Column 3 (Q: 0.745, weight: 0.100)
```

**Best for:** Similar model architectures, leveraging full model knowledge

---

### 3. **Confidence Weighted** - Expertise-Based

**How it works:**
- Calculates model confidence from Q-value spread
- Confident models get more influence
- Combines base weights with confidence scores

**Example Decision:**
```
🧠 Ensemble Decision Breakdown:
   Method: confidence_weighted
   Chosen Action: Column 3

🤖 Model Contributions with Confidence:
   • M1-CNN-550k: Column 3 (base: 0.300, conf: 0.285, effective: 0.386)
   • M1-CNN-300k: Column 3 (base: 0.250, conf: 0.201, effective: 0.300)
   • M1-CNN-150k: Column 1 (base: 0.200, conf: 0.156, effective: 0.231)
   • M1-CNN-50k: Column 3 (base: 0.150, conf: 0.098, effective: 0.165)
```

**Best for:** Mixed confidence levels, rewarding model certainty

## 📊 Method Comparison

| Method | Strengths | Weaknesses | Best Use Case |
|--------|-----------|------------|---------------|
| **weighted_voting** | Simple, interpretable, fast | Ignores Q-value magnitudes | Diverse model types |
| **q_value_averaging** | Uses full model knowledge | Can be dominated by extreme values | Similar architectures |
| **confidence_weighted** | Rewards model certainty | Complex, may amplify overconfident models | Mixed confidence levels |

## 🎯 Preset Configurations

### **Top Performers** (Recommended)
- **Models:** Best 4 from tournament (M1-CNN 550k, 500k, 450k, 400k)
- **Method:** Q-value averaging
- **Weights:** 0.4, 0.3, 0.2, 0.1
- **Strength:** Highest win rate, consistent performance

### **Diverse Architectures**
- **Models:** M1-CNN + Enhanced-DQN + Ultra-CNN + Heuristic
- **Method:** Weighted voting
- **Weights:** 0.4, 0.3, 0.2, 0.1
- **Strength:** Different perspectives, robust strategies

### **Evolution Stages**
- **Models:** M1-CNN from different training stages (550k → 10k)
- **Method:** Confidence weighted
- **Weights:** Decreasing with training stage
- **Strength:** Shows learning progression, rewards maturity

## ⚙️ Custom Configurations

### Creating JSON Configuration Files

Create a JSON file with your custom ensemble configuration:

```json
{
  "name": "My-Custom-Ensemble",
  "method": "q_value_averaging",
  "models": [
    {
      "path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt",
      "weight": 0.5,
      "name": "Best-Model"
    },
    {
      "path": "models_m1_cnn/m1_cnn_dqn_ep_400000.pt",
      "weight": 0.3,
      "name": "Stable-Model"
    },
    {
      "path": "heuristic",
      "weight": 0.2,
      "name": "Strategic-Baseline"
    }
  ]
}
```

### Configuration Fields

- **`name`** - Ensemble display name
- **`method`** - Ensemble method: `weighted_voting`, `q_value_averaging`, or `confidence_weighted`
- **`models`** - Array of model configurations:
  - **`path`** - Path to model file, or `"heuristic"`/`"random"` for baselines
  - **`weight`** - Model influence (will be normalized)
  - **`name`** - Display name for the model

### Using Custom Configurations

```bash
# Load from JSON file
python play_ensemble.py --config my_ensemble.json

# Interactive creation
python play_ensemble.py --interactive

# Save configuration
python play_ensemble.py --preset top_performers --save-config my_config.json
```

### Supported Model Types

The system auto-detects model types based on file paths:

- **M1-CNN models:** `models_m1_cnn/` directory, M1-optimized architecture
- **Ultra-CNN models:** `models_cnn/` directory, lightweight architecture  
- **Enhanced-DQN models:** `models_enhanced/` directory, strategic features
- **Fixed-DQN models:** `models_fixed/` directory, legacy models
- **Baseline agents:** `"heuristic"` or `"random"` in path field

## 🎮 Interactive Play Features

### Game Controls
- **Column selection:** Enter 0-6 to place piece
- **Quit game:** Enter 'q' during any move
- **Multiple games:** Play consecutive games with alternating first player

### Statistics Tracking
- Win/loss/draw counts
- Win percentages
- Performance across multiple games

### Visual Board Display
```
=============================
  0   1   2   3   4   5   6
+---+---+---+---+---+---+---+
|   |   |   |   |   |   |   |
+---+---+---+---+---+---+---+
|   |   |   | ● |   |   |   |
+---+---+---+---+---+---+---+
|   |   | ○ | ● |   |   |   |
+---+---+---+---+---+---+---+
```

## 🔧 Programmatic Usage

### Basic Ensemble Creation

```python
from ensemble_agent import EnsembleAgent

model_configs = [
    {'path': 'models_m1_cnn/m1_cnn_dqn_ep_550000.pt', 'weight': 0.6, 'name': 'Best'},
    {'path': 'models_m1_cnn/m1_cnn_dqn_ep_500000.pt', 'weight': 0.4, 'name': 'Stable'}
]

ensemble = EnsembleAgent(
    model_configs,
    ensemble_method="q_value_averaging",
    player_id=2,
    name="My-Ensemble",
    show_contributions=True
)

# Use in game
action = ensemble.choose_action(board_state, legal_moves)
breakdown = ensemble.get_last_decision_breakdown()
print(breakdown)
```

### Preset Ensembles

```python
from ensemble_agent import create_preset_ensemble

# Create preset ensembles
top_ensemble = create_preset_ensemble("top_performers", player_id=2)
diverse_ensemble = create_preset_ensemble("diverse_architectures", player_id=2)
evolution_ensemble = create_preset_ensemble("evolution_stages", player_id=2)
```

## 📈 Performance Tips

### Ensemble Selection Guidelines

1. **For maximum strength:** Use `top_performers` with top tournament models
2. **For robustness:** Use `diverse_architectures` with different model types
3. **For analysis:** Use `evolution_stages` to see learning progression
4. **For speed:** Use fewer models or `weighted_voting` method

### Model Weight Guidelines

- **Higher weights:** More influential models (better performance)
- **Balanced weights:** Democratic decision making
- **Exponential decay:** Emphasize best models heavily
- **Equal weights:** Pure averaging approach

### Method Selection Guidelines

- **Q-value averaging:** Best for similar architectures (CNN variants)
- **Weighted voting:** Best for diverse architectures (CNN + DQN + Heuristic)
- **Confidence weighted:** Best for models with varying training levels

## 🛠️ Troubleshooting

### Common Issues

**Model Loading Errors:**
```bash
❌ Failed to load M1-CNN-550k: No such file or directory
```
- Check model file paths in configuration
- Ensure models exist in specified directories

**Memory Issues:**
```bash
❌ CUDA out of memory
```
- Reduce number of models in ensemble
- Use CPU-only models by setting device appropriately

**Performance Issues:**
- Use `weighted_voting` for faster decisions
- Reduce ensemble size for speed
- Disable `show_contributions` for cleaner output

### Getting Help

```bash
# Show available models
python ensemble_agent.py

# Show ensemble info
python play_ensemble.py --preset top_performers --info

# Test with verbose output
python play_ensemble.py --preset top_performers --show-contributions
```

## 🎉 Examples

### Quick Tournament Test
```bash
# Play 3 games against top performers
python play_ensemble.py --preset top_performers
```

### Custom Lightweight Ensemble
```json
{
  "name": "Fast-Ensemble",
  "method": "weighted_voting", 
  "models": [
    {"path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt", "weight": 0.7, "name": "Latest"},
    {"path": "heuristic", "weight": 0.3, "name": "Strategic"}
  ]
}
```

### Analysis Ensemble
```json
{
  "name": "Learning-Analysis",
  "method": "confidence_weighted",
  "models": [
    {"path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt", "weight": 0.3, "name": "Expert"},
    {"path": "models_m1_cnn/m1_cnn_dqn_ep_300000.pt", "weight": 0.25, "name": "Advanced"},
    {"path": "models_m1_cnn/m1_cnn_dqn_ep_100000.pt", "weight": 0.2, "name": "Intermediate"},
    {"path": "models_m1_cnn/m1_cnn_dqn_ep_50000.pt", "weight": 0.15, "name": "Novice"},
    {"path": "models_m1_cnn/m1_cnn_dqn_ep_10000.pt", "weight": 0.1, "name": "Beginner"}
  ]
}
```

The ensemble system provides a powerful way to combine your trained models for maximum performance and insights into model behavior!