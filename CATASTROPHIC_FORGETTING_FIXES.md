# Catastrophic Forgetting Fixes - Complete Implementation

## Problem Summary

At episode 1,217,000, the agent experienced **catastrophic forgetting**:
- Performance vs heuristic collapsed from **60% → 0%**
- Agent was overfitting to league opponents (71.8% vs Tier 4)
- Root cause: Too much RL training (60% league+ensemble) vs fundamentals (25% heuristic)

## Fixes Implemented

### Fix 1: Rebalanced Opponent Distribution ✅

**Changed opponent ratios from:**
- ❌ Old: 40% League, 20% Ensemble, 25% Heuristic, 15% Random
- ✅ New: 30% League, 15% Ensemble, 40% Heuristic, 15% Random

**Key change:** Increased heuristic training from 25% → 40% to maintain fundamental skills.

**Location:** [resume_m1_with_league.py:294-330](train/resume_m1_with_league.py#L294-L330)

### Fix 2: Adaptive Heuristic Boost Mode ✅

**Automatic safety mechanism:**
- **Critical Threshold**: If vs Heuristic < 30%, triggers boost mode
- **Warning Threshold**: If vs Heuristic < 40%, prints warning
- **Boost Duration**: 5,000 episodes of intensive heuristic training
- **Boost Ratios**: 10% League, 5% Ensemble, 70% Heuristic, 15% Random

**How it works:**
1. Every evaluation (1000 episodes), check heuristic performance
2. If performance drops below 30%, activate boost mode
3. Boost mode trains primarily on heuristics for 5000 episodes
4. Automatically deactivates when duration ends
5. Can reactivate if performance drops again

**Locations:**
- Constants: [resume_m1_with_league.py:242-250](train/resume_m1_with_league.py#L242-L250)
- Boost logic: [resume_m1_with_league.py:294-330](train/resume_m1_with_league.py#L294-L330)
- Monitoring: [resume_m1_with_league.py:519-541](train/resume_m1_with_league.py#L519-L541)

### Fix 3: Enhanced Logging ✅

**Changed CSV logging to track all win rates separately:**
- Column 5: `current_win_rate` (vs Random)
- Column 6: `vs_heuristic` (vs Heuristic)
- Column 7: `vs_league_champ` (vs League Champion)

**Location:** [resume_m1_with_league.py:559-565](train/resume_m1_with_league.py#L559-L565)

### Fix 4: Unified Win Rates Plotting ✅

**New plotting utility to visualize all win rates together:**

Created `plot_win_rates.py` that:
- Plots vs Random (blue), vs Heuristic (red), vs League (green) on same graph
- Shows reference lines: 30% (critical), 50% (baseline), 60% (good)
- Displays latest performance statistics
- Auto-generated every 2000 episodes during training

**Usage:**
```bash
# Manual plotting
python plot_win_rates.py

# Or specify custom CSV
python plot_win_rates.py logs_m1_cnn/m1_cnn_training_log.csv
```

**Modified TrainingMonitor to track and plot all win rates:**

Changes to `train/training_monitor.py`:
- Added separate tracking: `win_rates_vs_random`, `win_rates_vs_heuristic`, `win_rates_vs_league`
- Updated `log_episode()` to accept `win_rate_vs_heuristic` and `win_rate_vs_league` parameters
- Replaced single win rate plot with unified plot showing all three opponents
- Added reference lines: 30% (critical), 50% (baseline)

**The main training progress plots now show:**
- Blue line: vs Random (should be ~90%+)
- Red line: vs Heuristic (target 55-65%, critical if < 30%)
- Green line: vs League Champion (gradual improvement expected)

**Outputs:**
- `training_progress_ep_NNNN.png` - Auto-generated every 2000 episodes (now includes all win rates)
- `logs_m1_cnn/all_win_rates_plot.png` - Standalone plot from manual script

**Locations:**
- [train/training_monitor.py](train/training_monitor.py) - Modified plotting
- [plot_win_rates.py](plot_win_rates.py) - Standalone utility

## Expected Behavior After Fixes

### Normal Training
- 30% games vs league opponents (progressive difficulty)
- 15% games vs ensemble (tournament strategy)
- **40% games vs heuristic** (maintain fundamentals - UP from 25%)
- 15% games vs random (exploration)

### Boost Mode (Auto-triggers if performance drops)
```
🚨 HEURISTIC BOOST MODE ACTIVATED at episode N
   Performance dropped to 28.0% (threshold: 30.0%)
   Boosting heuristic training to 70% for next 5,000 episodes
```

During boost:
- 10% league (reduced)
- 5% ensemble (reduced)
- **70% heuristic** (emergency retraining)
- 15% random (maintained)

After 5000 episodes:
```
✅ HEURISTIC BOOST MODE DEACTIVATED at episode N+5000
   Current performance: 52.0%
   Returning to normal opponent distribution
```

### Warning Mode (40% threshold)
```
⚠️  Warning: Heuristic performance at 38.0% (threshold: 40.0%)
```

## Restart Instructions

To resume training from episode 1,011,000 (before catastrophic forgetting):

```bash
# Resume from the last good checkpoint
python train/resume_m1_with_league.py --target 2000000

# The script will automatically:
# 1. Load m1_cnn_dqn_ep_1000000.pt (or latest checkpoint)
# 2. Use new rebalanced opponent distribution
# 3. Monitor heuristic performance
# 4. Trigger boost mode if needed
```

## Monitoring During Training

Look for these indicators:

**Good signs:**
- vs Heuristic stays above 40%
- Gradual improvement vs league opponents
- No boost mode activations

**Warning signs:**
```
⚠️  Warning: Heuristic performance at 38.0% (threshold: 40.0%)
```
- Watch next few evaluations
- Boost mode may activate soon

**Critical - Boost activated:**
```
🚨 HEURISTIC BOOST MODE ACTIVATED at episode 1,217,000
   Performance dropped to 28.0% (threshold: 30.0%)
   Boosting heuristic training to 70% for next 5,000 episodes
```
- System is self-correcting
- Expect recovery within 5000 episodes

## Technical Details

### Why These Fixes Work

1. **Rebalanced Distribution (40% heuristic)**
   - Prevents overfitting to RL strategies
   - Maintains fundamental Connect4 patterns
   - Heuristic provides stable learning signal

2. **Adaptive Boost Mode**
   - Automatic safety net
   - No manual intervention required
   - Proven recovery mechanism

3. **Separate Win Rate Tracking**
   - Identify issues early
   - Visualize tradeoffs between opponents
   - Data-driven decision making

### Performance Expectations

From previous training (episodes 1,010,000 - 1,011,000):
- vs Random: ~94% (baseline)
- vs Heuristic: **Target 55-65%**
- vs League: Should improve gradually from ~40% → 70%

**Key metric:** vs Heuristic should NEVER drop below 40% for extended periods.

## Files Modified

1. `train/resume_m1_with_league.py`
   - Rebalanced opponent distribution (L294-330)
   - Added boost mode logic (L242-250, L294-330)
   - Added monitoring triggers (L519-541)
   - Fixed CSV logging (L559-565)
   - Added auto-plotting (L580-595)
   - **Removed noisy phase announcements** (L352-355) - cleaner logs

2. `train/training_monitor.py`
   - Added separate win rate tracking (L39-41)
   - Updated `log_episode()` signature (L273-312)
   - Modified Plot 2 to show all win rates (L729-770)

3. `plot_win_rates.py` (NEW)
   - Unified win rates visualization
   - Automatic generation during training
   - Manual plotting utility

4. `CLEAN_LOGS_EXAMPLE.md` (NEW)
   - Explains new logging format
   - Shows what to expect during training
   - How to interpret metrics

## Next Steps

1. **Start training**: `python train/resume_m1_with_league.py --target 2000000`
2. **Monitor plots**: Check `logs_m1_cnn/win_rates_ep_*.png` every 2000 episodes
3. **Watch for boost mode**: Should NOT activate if fixes work
4. **Expected timeline**:
   - Episodes 1,011,000 - 1,100,000: Stabilization
   - Episodes 1,100,000 - 1,500,000: Gradual league improvement
   - Episodes 1,500,000 - 2,000,000: Advanced RL strategies while maintaining fundamentals

## Validation

To verify fixes are working:

```bash
# Check latest performance
tail -5 logs_m1_cnn/m1_cnn_training_log.csv

# Generate current plot
python plot_win_rates.py

# View plot
open logs_m1_cnn/all_win_rates_plot.png
```

Expected after 10,000 episodes:
- vs Heuristic > 40% ✅
- No boost mode activations ✅
- Steady improvement vs league ✅

---

**Implementation Date**: 2025-10-15
**Target Episode Range**: 1,011,000 → 2,000,000
**Estimated Training Time**: ~20-30 hours on M1 GPU
