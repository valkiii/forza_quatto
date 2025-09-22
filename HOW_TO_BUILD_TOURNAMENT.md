# 🏆 How to Build Custom Connect 4 Tournaments

A comprehensive guide to creating and running custom tournaments with CNN models and ensemble agents.

## 🎯 Quick Start

### Run Default Tournament
```bash
# Auto-discover all models and create diverse ensembles
python cnn_ensemble_tournament.py --games 100 --jobs -1
```

### Run Custom Tournament
```bash
# Use your own participant configuration
python cnn_ensemble_tournament.py --games 1000 --jobs -1 --participants my_config.json
```

### Use Existing Configuration
```bash
# Reuse participants from previous tournament
python cnn_ensemble_tournament.py --games 1000 --participants cnn_ensemble_tournament_results/participants.json
```

## 📋 Tournament Features

### What You Get
- **CNN Models**: M1-optimized models from 10k to 600k+ episodes
- **Ensemble Agents**: Multiple ensemble configurations with different methods
- **Baseline Agents**: Random and Heuristic for reference
- **Parallel Processing**: Fast execution using all CPU cores
- **Comprehensive Analysis**: Rankings, win rates, visualizations

### Generated Outputs
- **`rankings.csv`** - Complete rankings with all metrics
- **`win_rate_matrix.csv`** - Win rates between all participants  
- **`win_rate_heatmap.png`** - Regular grid heatmap
- **`win_rate_clustermap.png`** - Hierarchically clustered heatmap
- **`performance_by_type.png`** - Performance by agent type
- **`top_15_rankings.png`** - Bar chart of best performers
- **`cnn_training_progression.png`** - Learning progression
- **`participants.json`** - Your exact participant configuration
- **`tournament_results.json`** - Complete tournament data

## 🔧 Custom Participants Configuration

### Method 1: Edit Existing Configuration

```bash
# 1. Run initial tournament to generate participants.json
python cnn_ensemble_tournament.py --games 100

# 2. Edit the generated file
nano cnn_ensemble_tournament_results/participants.json

# 3. Reuse your customized configuration
python cnn_ensemble_tournament.py --games 1000 --participants cnn_ensemble_tournament_results/participants.json
```

### Method 2: Create From Scratch

```bash
# 1. Use example as template
cp examples/custom_tournament_participants.json my_tournament.json

# 2. Edit with your desired participants
nano my_tournament.json

# 3. Run tournament
python cnn_ensemble_tournament.py --games 1000 --participants my_tournament.json
```

### Method 3: Use Ready-Made Example

```bash
# Example with 8 participants: 4 CNN models + 2 ensembles + 2 baselines
python cnn_ensemble_tournament.py --games 1000 --participants examples/custom_tournament_participants.json
```

## 📄 JSON Configuration Format

### Complete Example Configuration
```json
[
  {
    "name": "M1-CNN-600k",
    "type": "cnn_model",
    "config": {
      "path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt",
      "architecture": "m1_optimized"
    }
  },
  {
    "name": "M1-CNN-550k", 
    "type": "cnn_model",
    "config": {
      "path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt",
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

### Participant Types

#### CNN Model Participant
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

**Available Model Paths:**
- **M1-CNN**: `models_m1_cnn/m1_cnn_dqn_ep_*.pt`
- **Ultra-CNN**: `models_cnn/cnn_dqn_*.pt`
- **Enhanced-DQN**: `models_enhanced/enhanced_*.pt`
- **Fixed-DQN**: `models_fixed/double_dqn_*.pt`

#### Ensemble Participant
```json
{
  "name": "My-Custom-Ensemble",
  "type": "ensemble",
  "config": {
    "models": [
      {
        "path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt",
        "weight": 0.4,
        "name": "Best-Model"
      },
      {
        "path": "models_m1_cnn/m1_cnn_dqn_ep_500000.pt",
        "weight": 0.3,
        "name": "Stable-Model"
      },
      {
        "path": "heuristic",
        "weight": 0.3,
        "name": "Strategic-Baseline"
      }
    ],
    "method": "q_value_averaging",
    "name": "MyEnsemble"
  }
}
```

**Ensemble Methods:**
- **`weighted_voting`** - Democratic vote on actions
- **`q_value_averaging`** - Average Q-values then select best action  
- **`confidence_weighted`** - Weight by model confidence

**Weight Strategies:**
- **Performance-based**: `[0.5, 0.3, 0.2]` - Higher weights for better models
- **Equal weights**: `[1.0, 1.0, 1.0]` - Democratic approach
- **Exponential decay**: `[0.5, 0.25, 0.125]` - Emphasize top models

#### Baseline Participant
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
- **`"heuristic"`** - Strategic rule-based agent
- **`"random"`** - Random agent

## 🎯 Tournament Scenarios

### 1. Compare Training Stages
```json
[
  {"name": "M1-CNN-100k", "type": "cnn_model", "config": {"path": "models_m1_cnn/m1_cnn_dqn_ep_100000.pt", "architecture": "m1_optimized"}},
  {"name": "M1-CNN-300k", "type": "cnn_model", "config": {"path": "models_m1_cnn/m1_cnn_dqn_ep_300000.pt", "architecture": "m1_optimized"}},
  {"name": "M1-CNN-500k", "type": "cnn_model", "config": {"path": "models_m1_cnn/m1_cnn_dqn_ep_500000.pt", "architecture": "m1_optimized"}},
  {"name": "M1-CNN-600k", "type": "cnn_model", "config": {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "architecture": "m1_optimized"}},
  {"name": "Heuristic", "type": "baseline", "config": {"agent_type": "heuristic"}}
]
```

### 2. Ensemble Method Comparison
```json
[
  {
    "name": "Vote-Ensemble",
    "type": "ensemble",
    "config": {
      "models": [
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "weight": 0.4, "name": "Latest"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt", "weight": 0.3, "name": "Second"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_500000.pt", "weight": 0.3, "name": "Third"}
      ],
      "method": "weighted_voting",
      "name": "Vote"
    }
  },
  {
    "name": "QAvg-Ensemble", 
    "type": "ensemble",
    "config": {
      "models": [
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "weight": 0.4, "name": "Latest"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt", "weight": 0.3, "name": "Second"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_500000.pt", "weight": 0.3, "name": "Third"}
      ],
      "method": "q_value_averaging",
      "name": "QAvg"
    }
  },
  {
    "name": "Conf-Ensemble",
    "type": "ensemble", 
    "config": {
      "models": [
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "weight": 0.4, "name": "Latest"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt", "weight": 0.3, "name": "Second"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_500000.pt", "weight": 0.3, "name": "Third"}
      ],
      "method": "confidence_weighted",
      "name": "Conf"
    }
  },
  {"name": "Best-Single", "type": "cnn_model", "config": {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "architecture": "m1_optimized"}}
]
```

### 3. Weight Strategy Comparison
```json
[
  {
    "name": "Equal-Weights",
    "type": "ensemble",
    "config": {
      "models": [
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "weight": 1.0, "name": "Latest"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt", "weight": 1.0, "name": "Second"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_500000.pt", "weight": 1.0, "name": "Third"}
      ],
      "method": "q_value_averaging",
      "name": "Equal"
    }
  },
  {
    "name": "Performance-Weights",
    "type": "ensemble",
    "config": {
      "models": [
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "weight": 0.5, "name": "Latest"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt", "weight": 0.3, "name": "Second"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_500000.pt", "weight": 0.2, "name": "Third"}
      ],
      "method": "q_value_averaging",
      "name": "Performance"
    }
  },
  {
    "name": "Exponential-Weights",
    "type": "ensemble",
    "config": {
      "models": [
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "weight": 0.5, "name": "Latest"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt", "weight": 0.25, "name": "Second"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_500000.pt", "weight": 0.125, "name": "Third"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_450000.pt", "weight": 0.125, "name": "Fourth"}
      ],
      "method": "q_value_averaging",
      "name": "Exponential"
    }
  }
]
```

### 4. Architecture Comparison
```json
[
  {"name": "M1-CNN-Best", "type": "cnn_model", "config": {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "architecture": "m1_optimized"}},
  {"name": "Ultra-CNN", "type": "cnn_model", "config": {"path": "models_cnn/cnn_dqn_final.pt", "architecture": "ultra_light"}},
  {"name": "Enhanced-DQN", "type": "cnn_model", "config": {"path": "models_enhanced/enhanced_double_dqn_final.pt", "architecture": "enhanced"}},
  {"name": "Fixed-DQN", "type": "cnn_model", "config": {"path": "models_fixed/double_dqn_final.pt", "architecture": "fixed"}},
  {"name": "Heuristic", "type": "baseline", "config": {"agent_type": "heuristic"}},
  {"name": "Random", "type": "baseline", "config": {"agent_type": "random"}}
]
```

### 5. Human vs AI Hybrid
```json
[
  {
    "name": "Pure-AI",
    "type": "ensemble",
    "config": {
      "models": [
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "weight": 1.0, "name": "AI-600k"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt", "weight": 1.0, "name": "AI-550k"},
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_500000.pt", "weight": 1.0, "name": "AI-500k"}
      ],
      "method": "q_value_averaging",
      "name": "PureAI"
    }
  },
  {
    "name": "AI-Heavy-Hybrid",
    "type": "ensemble",
    "config": {
      "models": [
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "weight": 0.8, "name": "AI-Latest"},
        {"path": "heuristic", "weight": 0.2, "name": "Human-Strategy"}
      ],
      "method": "weighted_voting",
      "name": "AIHeavy"
    }
  },
  {
    "name": "Balanced-Hybrid",
    "type": "ensemble",
    "config": {
      "models": [
        {"path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt", "weight": 0.5, "name": "AI-Latest"},
        {"path": "heuristic", "weight": 0.5, "name": "Human-Strategy"}
      ],
      "method": "weighted_voting",
      "name": "Balanced"
    }
  },
  {"name": "Pure-Human", "type": "baseline", "config": {"agent_type": "heuristic"}},
  {"name": "Pure-Random", "type": "baseline", "config": {"agent_type": "random"}}
]
```

## ⚙️ Command Line Options

```bash
python cnn_ensemble_tournament.py [OPTIONS]
```

### Required Options
None - all have defaults

### Optional Options
- `--games N` - Games per matchup (default: 100)
- `--jobs N` - Parallel jobs, -1 for all cores (default: -1)  
- `--participants FILE` - JSON file with custom participants (default: auto-discover)

### Examples
```bash
# Quick test with 10 games per matchup
python cnn_ensemble_tournament.py --games 10

# Full tournament with 1000 games per matchup 
python cnn_ensemble_tournament.py --games 1000

# Custom participants with 500 games per matchup
python cnn_ensemble_tournament.py --games 500 --participants my_config.json

# Single-threaded execution (for debugging)
python cnn_ensemble_tournament.py --games 100 --jobs 1
```

## 📊 Understanding Results

### Rankings CSV Columns
- **Agent** - Participant name
- **Type** - Agent type (cnn_model, ensemble, baseline)  
- **Overall_Win_Rate** - Win percentage across all opponents
- **Head_to_Head_Wins** - Number of opponents beaten (>50% win rate)
- **Strength_of_Schedule** - Average quality of beaten opponents
- **Games_Played** - Total games played (games_per_matchup × opponents)
- **Composite_Score** - Weighted combination: 60% win rate + 30% H2H + 10% SoS
- **Rank** - Final ranking (1 = best)

### Win Rate Matrix
- **Rows** - Agent playing as Player 1
- **Columns** - Agent playing as Player 2  
- **Values** - Win rate from row agent's perspective
- **Diagonal** - Always 0.5 (agent vs itself)

### Performance Interpretation
- **>70%** - Excellent performance
- **60-70%** - Good performance
- **50-60%** - Fair performance  
- **<50%** - Poor performance

## 🚀 Performance Tips

### Optimal Settings
```bash
# Balanced speed vs accuracy
python cnn_ensemble_tournament.py --games 100 --jobs -1

# High accuracy (takes longer)
python cnn_ensemble_tournament.py --games 1000 --jobs -1

# Quick testing
python cnn_ensemble_tournament.py --games 10 --jobs 2
```

### Execution Time Estimates
- **10 games**: ~30 seconds with 8 participants
- **100 games**: ~3-5 minutes with 8 participants  
- **1000 games**: ~30-45 minutes with 8 participants
- **1000 games**: ~2-3 hours with 21 participants (full auto-discovery)

### Hardware Recommendations
- **CPU**: More cores = faster execution (scales linearly)
- **RAM**: ~2GB per parallel job for ensemble agents
- **Storage**: ~100MB for results and visualizations

## ✅ Validation and Troubleshooting

### Common Issues

#### Model Files Not Found
```bash
❌ Failed to load M1-CNN-600k: No such file or directory
```
**Solution**: Check model file paths in your JSON configuration

#### JSON Syntax Errors
```bash
❌ Failed to load participants from file: Invalid JSON
```
**Solution**: Validate JSON syntax using online JSON validator

#### Missing Required Fields
```bash
❌ Unknown participant type: cnn_modle
```
**Solution**: Check spelling of `"type"` field (cnn_model, ensemble, baseline)

### Validation Steps
1. **Test small first**: Start with 4-6 participants and 10 games
2. **Verify paths**: Ensure all model files exist
3. **Check JSON**: Validate syntax before running
4. **Monitor logs**: Watch for loading errors during initialization

### Fallback Behavior
If custom participants file fails to load, the system automatically falls back to auto-discovery mode and creates default participants.

## 🎯 Best Practices

### Configuration Design
1. **Start simple**: Begin with 4-6 participants
2. **Include baselines**: Always include Heuristic for reference
3. **Document strategy**: Use meaningful names for weights/methods
4. **Save configurations**: Keep successful setups for reuse

### Tournament Design  
1. **Sufficient games**: Use 100+ games per matchup for stable results
2. **Diverse participants**: Mix individual models and ensembles
3. **Reference points**: Include known strong and weak agents
4. **Progressive testing**: Test configurations before long runs

### Analysis Workflow
1. **Quick validation**: Run with 10 games to verify setup
2. **Full tournament**: Run with 100-1000 games for final results
3. **Compare configurations**: Use same game count for fair comparison
4. **Document findings**: Save insights for future tournaments

## 📈 Advanced Techniques

### Dynamic Weight Generation
```python
import numpy as np

# Generate random weights
weights = np.random.dirichlet([1, 1, 1, 1])  # 4 models
for i, w in enumerate(weights):
    print(f"Model {i+1}: {w:.3f}")
```

### Model Selection Strategies
```python
# Performance-based weights (exponential)
episode_counts = [600000, 550000, 500000, 450000]
base_weights = [2**i for i in range(len(episode_counts))]
normalized = base_weights / sum(base_weights)
```

### Systematic Comparison
1. Create multiple configurations with systematic variations
2. Run identical tournaments on each configuration  
3. Compare Composite_Score distributions
4. Identify optimal ensemble strategies

This guide provides everything you need to design, configure, and run sophisticated Connect 4 tournaments with complete control over participants and analysis! 🏆