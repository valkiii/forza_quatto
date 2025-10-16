# Composite Score Fix for Best Model Selection

## Problem Identified

**Issue:** Agent training has stagnated at episode ~1,277,000 with no new "best" models saved, despite significant improvements in strategic performance.

### Root Cause
The "best" model selection was based ONLY on **vs Random** win rate, which:
1. **Already saturated** near 100% (random variance between 96-100%)
2. **Doesn't measure strategic improvement** (heuristic and league performance)
3. **Blocks league progression** (new models not added to league)

### Evidence
```
Episode 1,277,000 (last "best"):
- vs Random: 100% (50/50 wins)
- vs Heuristic: 80%
- vs League: 77%

Episode 1,612,000 (current, 335k episodes later):
- vs Random: 96% (48/50 wins) ← 2% "worse" due to random variance
- vs Heuristic: 94% ← 14% BETTER!
- vs League: 67% (different opponent tier)
```

**Result:** Despite massive improvement vs Heuristic (80% → 94%), no new "best" model was saved because vs Random dropped from 100% → 96% (just 2 games difference, within random variance).

## Solution

Changed best model selection from single metric to **composite score**:

```python
# Old (saturated metric)
if current_win_rate > best_win_rate:  # Only vs Random

# New (composite scoring)
composite_score = (0.2 * current_win_rate) +    # 20% Random
                  (0.6 * vs_heuristic) +        # 60% Heuristic (primary)
                  (0.2 * vs_league_champ)       # 20% League
if composite_score > best_win_rate:
```

### Rationale for Weights

1. **60% Heuristic** - Primary measure of strategic play
   - Most important opponent for real gameplay
   - Shows fundamental Connect4 understanding
   - Target range: 55-65% (not saturated)

2. **20% Random** - Baseline competence
   - Should stay 90%+ naturally
   - Detects catastrophic failures
   - Already near-perfect, so low weight

3. **20% League** - Advanced strategy
   - Measures RL vs RL performance
   - Varies as league expands
   - Secondary importance

## Impact

### Immediate Effect
Using the composite score, episode 1,612,000 would score:
```
Old best (1,277,000): 0.2×1.0 + 0.6×0.8 + 0.2×0.77 = 83.4%
Current  (1,612,000): 0.2×0.96 + 0.6×0.94 + 0.2×0.67 = 89.0%
```

The current agent is **5.6% better** using composite scoring!

### Expected Behavior Going Forward

**New "best" models will be saved when:**
- Heuristic performance improves (60% weight)
- League performance improves significantly (20% weight)
- Overall strategic capability increases

**Prevents:**
- Saturation at 100% vs Random blocking progress
- Ignoring strategic improvements
- League stagnation

### Example Scenarios

**Scenario 1: Strong strategic improvement**
```
Old: R:98% H:85% L:70% → Composite: 85.6%
New: R:96% H:92% L:75% → Composite: 89.4% ✅ SAVED (better despite lower vs Random)
```

**Scenario 2: Maintains strategic edge**
```
Old: R:96% H:92% L:75% → Composite: 89.4%
New: R:100% H:90% L:75% → Composite: 89.0% ❌ NOT SAVED (worse overall)
```

**Scenario 3: Catastrophic forgetting**
```
Old: R:96% H:92% L:75% → Composite: 89.4%
New: R:94% H:30% L:80% → Composite: 52.8% ❌ NOT SAVED (heuristic collapsed)
```

## Implementation Details

**File Modified:** `train/resume_m1_with_league.py` (Lines 541-557)

**Before:**
```python
if current_win_rate > best_win_rate:
    best_win_rate = current_win_rate
    print(f"💎 M1 CNN Best: {current_win_rate:.1%} - saved")
```

**After:**
```python
composite_score = (0.2 * current_win_rate) + (0.6 * vs_heuristic) + (0.2 * vs_league_champ)
if composite_score > best_win_rate:
    best_win_rate = composite_score
    print(f"💎 M1 CNN Best: Composite {composite_score:.1%} " +
          f"(R:{current_win_rate:.0%} H:{vs_heuristic:.0%} L:{vs_league_champ:.0%}) " +
          f"- saved {os.path.basename(best_path)}")
```

## Validation

### Next Evaluation (Episode ~1,613,000)
Expected behavior:
- If vs Heuristic ≥ 92% and composite > 89%, new "best" will be saved
- League will receive new checkpoint
- Progress will resume

### Long-term Monitoring
Look for:
- ✅ Regular "best" model saves (every 10-50k episodes during improvement)
- ✅ League expansion with new strong models
- ✅ Gradual composite score improvement
- ⚠️ If composite stops improving for >100k episodes → investigate

## Comparison: Old vs New Metric

| Metric | Advantage | Disadvantage |
|--------|-----------|--------------|
| **Old: vs Random only** | Simple, clear | Saturates at 100%, ignores strategy |
| **New: Composite (20/60/20)** | Rewards strategic improvement, doesn't saturate | Slightly more complex |

## Historical Analysis

If we recalculate recent episodes with composite score:

```
Episode     vs_Random  vs_Heuristic  vs_League  Old_Best  Composite  Would_Save?
1,277,000   100%       80%           77%        YES       83.4%      YES (baseline)
1,563,000   100%       98%           87%        YES*      97.4%      YES! (14% better!)
1,566,000   90%        90%           90%        NO        90.0%      YES! (6.6% better!)
1,568,000   98%        96%           80%        NO        94.4%      NO (worse than 1,563)
1,612,000   96%        94%           67%        NO        89.0%      NO (worse than 1,563)

* These weren't actually saved because vs_Random alone wasn't new best
```

This shows the composite metric would have captured peaks at 1,563,000 and 1,566,000!

## Related Issues Fixed

1. **League Stagnation** - New best models → league expansion
2. **Progress Tracking** - Composite score better reflects true capability
3. **Strategic Focus** - Rewards what matters (heuristic > random)

## Future Considerations

If training continues beyond 2M episodes and league performance becomes dominant:
- Could adjust weights to 10% Random / 50% Heuristic / 40% League
- Monitor which metric drives improvement
- Ensure no single metric saturates

---

**Implementation Date:** 2025-10-16
**First Effect:** Next evaluation (~episode 1,613,000)
**Expected Impact:** Resume league progression and model improvement recognition
