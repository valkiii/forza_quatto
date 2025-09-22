# 🏆 Custom Tournament Participants Guide

This guide explains how to create custom participant configurations for the CNN + Ensemble Tournament system.

## 📋 Overview

You can now provide your own list of participants by creating a JSON file and passing it to the tournament script. This allows you to:

- **Select specific models** instead of using all discovered models
- **Create custom ensemble configurations** with your preferred weights and methods
- **Mix different model types** (CNN, Enhanced-DQN, baselines) as needed
- **Control tournament scope** for focused comparisons

## 🎯 Usage

### Basic Usage
```bash
# Use auto-discovered participants (default)
python cnn_ensemble_tournament.py --games 100

# Use custom participants from JSON file
python cnn_ensemble_tournament.py --games 100 --participants my_participants.json

# Use existing participants.json from previous tournament
python cnn_ensemble_tournament.py --games 100 --participants cnn_ensemble_tournament_results/participants.json
```

### Advanced Usage
```bash
# Custom participants with full parallelization
python cnn_ensemble_tournament.py --games 1000 --jobs -1 --participants examples/custom_tournament_participants.json
```

## 📄 JSON Format

### Participant Structure
Each participant must have:
- **`name`**: Unique identifier for the participant
- **`type`**: Type of participant (`"cnn_model"`, `"ensemble"`, or `"baseline"`)  
- **`config`**: Configuration specific to the participant type

### CNN Model Participant
```json
{
  "name": "M1-CNN-550k",
  "type": "cnn_model",
  "config": {
    "path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt",
    "architecture": "m1_optimized"
  }
}
```

### Ensemble Participant
```json
{
  "name": "My-Custom-Ensemble",
  "type": "ensemble",
  "config": {
    "models": [
      {
        "path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt",
        "weight": 0.5,
        "name": "Latest-Model"
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
    ],
    "method": "q_value_averaging",
    "name": "MyEnsemble"
  }
}
```

**Ensemble Methods:**
- `"weighted_voting"` - Democratic vote on actions
- `"q_value_averaging"` - Average Q-values then select best action
- `"confidence_weighted"` - Weight by model confidence

### Baseline Participant
```json
{
  "name": "Heuristic",
  "type": "baseline", 
  "config": {
    "agent_type": "heuristic"
  }
}
```

**Baseline Types:**
- `"random"` - Random agent
- `"heuristic"` - Strategic rule-based agent

## 📁 Complete Example

**`my_tournament.json`**:
```json
[
  {
    "name": "Best-M1-CNN",
    "type": "cnn_model",
    "config": {
      "path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt",
      "architecture": "m1_optimized"
    }
  },
  {
    "name": "Early-M1-CNN",
    "type": "cnn_model",
    "config": {
      "path": "models_m1_cnn/m1_cnn_dqn_ep_100000.pt", 
      "architecture": "m1_optimized"
    }
  },
  {
    "name": "Top3-Ensemble",
    "type": "ensemble",
    "config": {
      "models": [
        {
          "path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt",
          "weight": 0.5,
          "name": "Latest"
        },
        {
          "path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt",
          "weight": 0.3,
          "name": "Second"
        },
        {
          "path": "models_m1_cnn/m1_cnn_dqn_ep_500000.pt",
          "weight": 0.2,
          "name": "Third"
        }
      ],
      "method": "q_value_averaging",
      "name": "Top3"
    }
  },
  {
    "name": "Hybrid-Ensemble",
    "type": "ensemble",
    "config": {
      "models": [
        {
          "path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt",
          "weight": 0.7,
          "name": "AI-Model"
        },
        {
          "path": "heuristic",
          "weight": 0.3,
          "name": "Human-Strategy"
        }
      ],
      "method": "weighted_voting",
      "name": "Hybrid"
    }
  },
  {
    "name": "Heuristic",
    "type": "baseline",
    "config": {
      "agent_type": "heuristic"
    }
  },
  {
    "name": "Random",
    "type": "baseline",
    "config": {
      "agent_type": "random"
    }
  }
]
```

## 🔧 Model Path Guidelines

### CNN Models
- **M1-CNN models**: `models_m1_cnn/m1_cnn_dqn_ep_*.pt`
- **Ultra-CNN models**: `models_cnn/cnn_dqn_*.pt`
- **Enhanced-DQN models**: `models_enhanced/enhanced_*.pt` 
- **Fixed-DQN models**: `models_fixed/double_dqn_*.pt`

### Baseline Agents
- **Heuristic**: Use `"heuristic"` as path
- **Random**: Use `"random"` as path

## ⚙️ Ensemble Configuration Tips

### Weight Strategies
- **Performance-based**: Higher weights for better models
- **Equal weights**: Democratic approach (all weights = 1.0)
- **Exponential decay**: [0.5, 0.25, 0.125, 0.0625] for ranked models
- **Random weights**: Use `np.random.dirichlet([1,1,1,1])` for exploration

### Method Selection
- **Q-value averaging**: Best for similar architectures (all CNN models)
- **Weighted voting**: Best for diverse architectures (CNN + Enhanced + Heuristic) 
- **Confidence weighted**: Best for models with different training levels

## 📊 Tournament Results

The tournament will generate the same comprehensive results regardless of participant source:

### Generated Files
- **`rankings.csv`** - Complete rankings with metrics
- **`win_rate_matrix.csv`** - Win rates between all participants
- **`win_rate_heatmap.png`** - Regular heatmap visualization
- **`win_rate_clustermap.png`** - Clustered heatmap
- **`participants.json`** - Your exact participant configuration
- **`tournament_results.json`** - Complete tournament data

### Visualizations
1. **Win Rate Heatmap** - Regular grid showing all matchups
2. **Win Rate Clustermap** - Clustered by similarity
3. **Performance by Type** - Scatter plot colored by participant type
4. **Top 15 Rankings** - Bar chart of best performers
5. **CNN Training Progression** - Learning curve (if CNN models included)

## 🚀 Example Workflows

### Compare Training Stages
```json
[
  {"name": "M1-CNN-100k", "type": "cnn_model", "config": {"path": "models_m1_cnn/m1_cnn_dqn_ep_100000.pt", "architecture": "m1_optimized"}},
  {"name": "M1-CNN-300k", "type": "cnn_model", "config": {"path": "models_m1_cnn/m1_cnn_dqn_ep_300000.pt", "architecture": "m1_optimized"}},
  {"name": "M1-CNN-500k", "type": "cnn_model", "config": {"path": "models_m1_cnn/m1_cnn_dqn_ep_500000.pt", "architecture": "m1_optimized"}},
  {"name": "Heuristic", "type": "baseline", "config": {"agent_type": "heuristic"}}
]
```

### Ensemble Method Comparison
```json
[
  {"name": "Vote-Ensemble", "type": "ensemble", "config": {"models": [...], "method": "weighted_voting", "name": "Vote"}},
  {"name": "QAvg-Ensemble", "type": "ensemble", "config": {"models": [...], "method": "q_value_averaging", "name": "QAvg"}},
  {"name": "Conf-Ensemble", "type": "ensemble", "config": {"models": [...], "method": "confidence_weighted", "name": "Conf"}},
  {"name": "Best-Single", "type": "cnn_model", "config": {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "architecture": "m1_optimized"}}
]
```

### Weight Strategy Comparison  
```json
[
  {"name": "Equal-Weight", "type": "ensemble", "config": {"models": [{"path": "...", "weight": 1.0}, ...], "method": "q_value_averaging", "name": "Equal"}},
  {"name": "Performance-Weight", "type": "ensemble", "config": {"models": [{"path": "...", "weight": 0.5}, {"path": "...", "weight": 0.3}, ...], "method": "q_value_averaging", "name": "Performance"}},
  {"name": "Exponential-Weight", "type": "ensemble", "config": {"models": [{"path": "...", "weight": 0.5}, {"path": "...", "weight": 0.25}, ...], "method": "q_value_averaging", "name": "Exponential"}}
]
```

## ✅ Validation

The system will validate your participants file and fall back to auto-discovery if there are issues:

- **Missing files**: Models that don't exist will be skipped
- **Invalid JSON**: Syntax errors will trigger fallback
- **Missing fields**: Required fields will cause errors
- **Invalid paths**: Non-existent model paths will be caught during loading

## 🎯 Best Practices

1. **Start small**: Test with 4-6 participants before scaling up
2. **Include baselines**: Always include Heuristic for reference
3. **Document weights**: Use meaningful names to remember your strategy
4. **Save configurations**: Keep successful participant files for reuse
5. **Validate models**: Ensure all model files exist before running tournament

## 📈 Performance Notes

- **Parallel processing**: Use `--jobs -1` for maximum speed
- **Model caching**: Agents are cached within each process
- **Memory usage**: Scales with number of ensemble participants
- **Execution time**: ~1-2 minutes per 1000 games with 8 participants on modern hardware

Ready to create your custom tournament? Start with the example file in `examples/custom_tournament_participants.json`!