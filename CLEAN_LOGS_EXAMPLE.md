# Clean Logging Format

## Changes Made

**Problem:** With mixed opponent distribution, the training was logging phase changes every single episode (noisy):
```
🔄 M1 CNN PHASE at episode 1,234,300: Heuristic → League-T4
🔄 M1 CNN PHASE at episode 1,234,301: League-T4 → Ensemble
🔄 M1 CNN PHASE at episode 1,234,302: Ensemble → Random
... (hundreds of these per evaluation period)
```

**Solution:** Removed phase change announcements since with mixed opponents (30% league, 15% ensemble, 40% heuristic, 15% random), the opponent changes every 1-3 episodes on average.

## New Log Format

### Every 100 Episodes (Training Progress)
```
Episode 1,221,100 | League-T2 | Avg Reward: 22.8 | ε: 0.010000 | Buffer: 50000 | Steps: 11,564,718 | Spatial: 2.25 | League Tier: 4
Episode 1,221,200 | League-T1 | Avg Reward: 22.3 | ε: 0.010000 | Buffer: 50000 | Steps: 11,565,449 | Spatial: 1.71 | League Tier: 4
Episode 1,221,300 | League-T4 | Avg Reward: 20.1 | ε: 0.010000 | Buffer: 50000 | Steps: 11,566,180 | Spatial: 2.17 | League Tier: 4
```

**Explanation:**
- Shows whichever opponent was used for that specific episode
- Displays running average reward (last 100 episodes)
- Shows current league tier (highest tier unlocked, not necessarily current opponent)

### Every 1000 Episodes (Evaluation)
```
Episode 1,221,000 | League-T0 | Win: 94.0% | vs Heur: 60.0% | vs League: 40.0% | ε: 0.010000 | Spatial: 1.30

🏆 LEAGUE STATUS:
   Current Tier: 4/4
   Training vs: m1_cnn_dqn_best_ep_1053000.pt
   Win Rate: 71.5% (167830 games)
   Need 0 more games or higher win rate for promotion

   League Models:
       Tier 0: m1_cnn_dqn_ep_244000.pt
       Tier 1: m1_cnn_dqn_ep_478000.pt
       Tier 2: m1_cnn_dqn_ep_742000.pt
       Tier 3: m1_cnn_dqn_final_500k.pt
     → Tier 4: m1_cnn_dqn_best_ep_1053000.pt
```

**Explanation:**
- **Win**: vs Random (should be 90%+)
- **vs Heur**: vs Heuristic (target 55-65%, critical if < 30%)
- **vs League**: vs League Champion (Tier 4) - gradual improvement expected
- League status shows overall performance against current tier

### Boost Mode Activation (if triggered)
```
⚠️  Warning: Heuristic performance at 38.0% (threshold: 40.0%)

🚨 HEURISTIC BOOST MODE ACTIVATED at episode 1,235,000
   Performance dropped to 28.0% (threshold: 30.0%)
   Boosting heuristic training to 70% for next 5,000 episodes
```

### Boost Mode Deactivation
```
✅ HEURISTIC BOOST MODE DEACTIVATED at episode 1,240,000
   Current performance: 52.0%
   Returning to normal opponent distribution
```

### Every 2000 Episodes (Plotting)
```
📊 M1 CNN plots generated for episode 1,222,000
```

## Log Interpretation

### Normal Training (Healthy)
```
Episode 1,011,000 | Heuristic | Win: 94.0% | vs Heur: 60.0% | vs League: 40.0%
Episode 1,012,000 | Random | Win: 95.0% | vs Heur: 58.0% | vs League: 42.0%
Episode 1,013,000 | League-T3 | Win: 93.0% | vs Heur: 61.0% | vs League: 45.0%
```

**Good signs:**
- vs Heur stays 55-65%
- vs League gradually increases
- Win (vs Random) stays 90%+
- No boost activations

### Warning Signs
```
Episode 1,217,000 | League-T4 | Win: 88.0% | vs Heur: 38.0% | vs League: 70.0%

⚠️  Warning: Heuristic performance at 38.0% (threshold: 40.0%)
```

**Concerning:**
- vs Heur dropping below 40%
- vs League improving rapidly while vs Heur drops
- Indicates overfitting to league opponents

### Critical - Catastrophic Forgetting
```
Episode 1,221,000 | League-T4 | Win: 88.0% | vs Heur: 8.0% | vs League: 50.0%

🚨 HEURISTIC BOOST MODE ACTIVATED at episode 1,221,000
   Performance dropped to 8.0% (threshold: 30.0%)
   Boosting heuristic training to 70% for next 5,000 episodes
```

**Critical issue:**
- vs Heur collapsed to single digits
- System auto-activates boost mode
- Should recover within 5000 episodes

## Comparison: Old vs New

### Old Logging (Before Fixes)
- Only showed one opponent type for long stretches
- Announced phase changes when switching curricula
- Single win rate metric (vs Random)
- No automatic safety mechanisms

### New Logging (After Fixes)
- Shows current opponent every 100 episodes (may vary)
- No noisy phase announcements (mixed distribution)
- Three win rates tracked: vs Random, vs Heuristic, vs League
- Auto-detects and responds to performance drops
- Cleaner, more informative output

## Key Metrics to Watch

1. **vs Heuristic (Most Important)**
   - Target: 55-65%
   - Warning: < 40%
   - Critical: < 30%

2. **vs Random**
   - Should stay 90%+
   - If dropping, fundamental issues

3. **vs League**
   - Should gradually increase from 40% → 70%
   - Shows learning of advanced strategies

4. **Buffer Size**
   - Should stay at 50,000 (max capacity)
   - If drops, indicates replay buffer issues

5. **Spatial Score**
   - Tactical awareness metric
   - Higher is better (strategic moves)
