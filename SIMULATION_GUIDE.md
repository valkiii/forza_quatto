# Connect 4 Agent Simulation Guide

This guide explains how to use the comprehensive simulation system to test your trained RL agent against different opponents.

## Quick Start

### Option 1: Interactive Launcher (Recommended)
```bash
python run_simulation.py
```
This provides a menu-driven interface for common simulation scenarios.

### Option 2: Direct Command Line
```bash
python simulate_agents.py --opponent heuristic --games 1000 --save-results
```

## Available Opponents

### 1. Random Agent (`--opponent random`)
- Makes completely random legal moves
- Good baseline to test basic competency
- Expected RL win rate: 80-95%

### 2. Heuristic Agent (`--opponent heuristic`)  
- Uses strategic Connect 4 heuristics
- Tests tactical understanding
- Expected RL win rate: 60-80%

### 3. Self-Play (`--opponent self`)
- RL agent vs identical copy of itself
- Tests consistency and style
- Expected win rate: ~50% (should be balanced)

## Command Line Options

```bash
python simulate_agents.py [OPTIONS]

Options:
  --opponent {random,heuristic,self}  Opponent type (default: heuristic)
  --games GAMES                       Number of games (default: 1000)
  --model MODEL                       Path to specific model file
  --save-results                      Save detailed results to JSON
  --all-opponents                     Test against all opponent types
```

## Example Commands

### Quick Tests
```bash
# Quick 100-game test vs heuristic
python simulate_agents.py --opponent heuristic --games 100

# Test vs random opponent
python simulate_agents.py --opponent random --games 1000
```

### Comprehensive Analysis
```bash
# Test against all opponents with full statistics
python simulate_agents.py --all-opponents --games 1000 --save-results

# Use specific model
python simulate_agents.py --model models/double_dqn_best_ep_150000.pt --opponent heuristic --games 2000
```

## Understanding the Results

### Overall Performance Levels
- **🟢 EXCELLENT (70%+ win rate)**: Strong strategic play
- **🟡 GOOD (60-70% win rate)**: Solid tactical understanding  
- **🟠 FAIR (50-60% win rate)**: Basic competency, room for improvement
- **🔴 POOR (<50% win rate)**: Needs more training

### Position Analysis
The simulation alternates who goes first and tracks performance by position:

- **Position-independent play**: Win rates similar regardless of starting position
- **First-move advantage**: How much the starting player benefits
- **Position bias**: Whether your agent prefers going first or second

### Game Length Analysis
- **Fast games (<15 moves)**: Quick decisive victories/defeats
- **Balanced games (15-25 moves)**: Tactical back-and-forth
- **Long games (25+ moves)**: Strategic endgame battles

## Interpreting Results

### vs Random Agent
- **Expected**: 80-95% win rate
- **If lower**: Agent may have learning issues or poor action selection
- **Position bias**: Should be minimal against random play

### vs Heuristic Agent  
- **Expected**: 60-80% win rate depending on training progress
- **Key metric**: This tests real strategic understanding
- **Position analysis**: May show preference for first/second based on training

### vs Self (Self-Play)
- **Expected**: ~50% win rate (balanced)
- **Consistency check**: Both copies should play similarly
- **Style analysis**: Shows your agent's typical game patterns

## Sample Output Interpretation

```
🏆 OVERALL RESULTS
RL Agent wins:    742 ( 74.2%)  <- Strong performance vs heuristic
Opponent wins:    258 ( 25.8%)
Performance:      🟢 EXCELLENT

🔄 POSITION ANALYSIS  
When RL goes first:   78.0% win rate  <- Slight first-move preference
When RL goes second:  70.4% win rate
Position bias:       🎯 Significant positional bias (+7.6%)

⏱️ GAME LENGTH ANALYSIS
Average game length: 19.3 moves       <- Balanced tactical games
Game style:          ⚖️ Balanced tactical games
```

## Saved Results

When using `--save-results`, detailed JSON files are created containing:

- Complete win/loss statistics
- Position-based breakdowns  
- Game length distributions
- Performance metadata
- Model information

Example files:
- `simulation_results_heuristic_1000games.json`
- `simulation_results_random_1000games.json`

## Troubleshooting

### "No trained model found"
Ensure you have trained models in the `models/` directory:
```bash
ls models/
# Should show files like: double_dqn_final.pt, double_dqn_ep_100000.pt, etc.
```

### Low Performance vs Random
This indicates training issues. Check:
- Model architecture matches training (256 hidden units)
- Proper state encoding (84-dimensional)
- Legal action masking is working

### Inconsistent Self-Play Results  
May indicate:
- Model loading issues
- Player ID confusion
- Stochastic behavior differences

## Performance Benchmarks

Based on our curriculum training:

| Opponent | Training Phase | Expected Win Rate |
|----------|----------------|-------------------|
| Random   | After 10K episodes | 85%+ |
| Heuristic| After 60K episodes | 70%+ |
| Self-play| After 100K+ episodes | ~50% |

## Tips for Analysis

1. **Run multiple simulations**: Results can vary, especially with smaller game counts
2. **Compare across training checkpoints**: Track learning progress over time
3. **Analyze position bias**: Understand your agent's strategic preferences
4. **Game length trends**: Shorter games may indicate more decisive play
5. **Save comprehensive results**: JSON files contain detailed breakdowns for analysis

## Integration with Training

Use simulations to:
- **Monitor training progress**: Regular evaluation during training
- **Compare model versions**: Find the best checkpoint
- **Validate curriculum learning**: Ensure agents improve against harder opponents
- **Debug training issues**: Identify problems early

## Next Steps

After running simulations:
1. Analyze the detailed JSON results
2. Compare performance across different model checkpoints  
3. Use insights to improve training parameters
4. Test human gameplay with the best-performing model

The simulation system provides comprehensive insights into your RL agent's strategic capabilities and helps validate the effectiveness of your training approach.