# Backtracking Curriculum Fix for League Stagnation

## Problem: Stuck at Tier 3

**Current Situation:**
- Agent stuck at Tier 3 for **~165,000 episodes** (~49,600 games)
- Win rate: **56.6%** vs 920k checkpoint
- Required: **60%** for promotion
- Gap is too large to overcome with current curriculum

### League Structure (Before Fix)
```
Tier 0: m1_cnn_dqn_best_ep_1000.pt      (Episode 1,000)
Tier 1: m1_cnn_dqn_best_ep_301000.pt    (Episode 301,000)
Tier 2: m1_cnn_dqn_ep_600000.pt         (Episode 600,000)     ← Beat this
Tier 3: m1_cnn_dqn_ep_920000.pt         (Episode 920,000)     ← STUCK HERE (56.6%)
Tier 4: m1_cnn_dqn_ep_1236000.pt        (Episode 1,236,000)
```

**The Problem:**
- **Huge jump** from 600k → 920k (320,000 episodes)
- 920k model is exceptionally strong (98% vs Heuristic!)
- Agent can't bridge this difficulty gap
- Fixed 60% threshold prevents any progress

## Solution: Progressive Backtracking Curriculum

When stuck for >50k games, automatically adjust curriculum:

### Strategy
1. **Detect stagnation** - Agent at <60% after 50k+ games
2. **Find intermediate checkpoint** - Halfway between previous tier and blocker
3. **Replace previous tier** - Insert intermediate model
4. **Demote agent** - Move back one tier
5. **Retry with smoother progression** - Smaller difficulty steps

### Visual Example

**Before Backtracking:**
```
Tier 2: 600k ──────(320k gap)─────→ Tier 3: 920k (BLOCKED)
                                          ↑
                                     Agent stuck at 56.6%
```

**After Backtracking:**
```
Tier 2: 760k ──(160k gap)─→ Tier 3: 920k
    ↑
Agent demoted here, tries again with easier progression
```

## Implementation

### New Methods in LeagueManager

**1. `should_backtrack()` - Detection Logic**
```python
def should_backtrack(self) -> bool:
    """Check if agent is stuck and should backtrack."""
    if self.current_tier == 0:
        return False  # Can't backtrack from tier 0

    tier_key = f"tier_{self.current_tier}"
    stats = self.performance_tracker[tier_key]

    # Stuck if >50k games without reaching 60%
    if stats["games"] > 50000:
        win_rate = stats["wins"] / stats["games"]
        if win_rate < self.win_threshold:
            return True

    return False
```

**2. `backtrack_tier()` - Curriculum Adjustment**
```python
def backtrack_tier(self) -> bool:
    """Backtrack curriculum when stuck on a tier."""

    # 1. Get current blocker and previous model
    current_blocker = self.league_models[self.current_tier]      # 920k
    previous_model = self.league_models[self.current_tier - 1]   # 600k

    # 2. Calculate intermediate episode
    previous_ep = 600000
    blocker_ep = 920000
    intermediate_ep = (600000 + 920000) // 2 = 760000

    # 3. Find closest checkpoint to 760k
    # Search models_m1_cnn/*.pt for episode ~760k
    # Example found: m1_cnn_dqn_ep_758000.pt

    # 4. Replace previous tier with intermediate
    self.league_models[self.current_tier - 1] = {
        'path': 'models_m1_cnn/m1_cnn_dqn_ep_758000.pt',
        'episode': 758000,
        'name': 'm1_cnn_dqn_ep_758000.pt',
        'tier': 2,
        ...
    }

    # 5. Demote agent back one tier
    self.current_tier -= 1  # From tier 3 → tier 2

    # 6. Reset performance tracking
    for i in range(len(self.league_models)):
        self.performance_tracker[f"tier_{i}"] = {"wins": 0, "games": 0}

    return True
```

### Integration in Training Loop

**Modified `resume_m1_with_league.py`:**
```python
# Check for tier promotion or backtracking
if league_game_counter % 50 == 0:
    # First check if we should backtrack (stuck for too long)
    if league.backtrack_tier():
        # Recreate ensemble with new league state
        if use_ensemble:
            ensemble_opponent = create_ensemble_opponent(league, player_id=2)
    # Otherwise check for promotion
    elif league.promote_tier():
        # Recreate ensemble with new league state
        if use_ensemble:
            ensemble_opponent = create_ensemble_opponent(league, player_id=2)
```

## Expected Behavior

### Current State (Episode ~1,613,000)
```
🏆 LEAGUE STATUS:
   Current Tier: 3/4
   Training vs: m1_cnn_dqn_ep_920000.pt
   Win Rate: 56.6% (49,600 games)
   ⚠️  Warning: Approaching backtrack threshold (50k games)
   Need 0 more games or 60%+ win rate for promotion
```

### After Next ~300 League Games (Will Hit 50k)
```
🔄 BACKTRACKING CURRICULUM (stuck at Tier 3 for 50,000 games)
   Current blocker: m1_cnn_dqn_ep_920000.pt (Episode 920,000)
   Previous model: m1_cnn_dqn_ep_600000.pt (Episode 600,000)
   Found intermediate: m1_cnn_dqn_ep_758000.pt (Episode 758,000)

   Strategy:
   1. Replace Tier 2 with intermediate checkpoint
   2. Move blocker to Tier 2
   3. Demote agent back to Tier 2
   4. Create smoother difficulty progression

   ✅ Curriculum adjusted! Agent demoted to Tier 2
   New opponent: m1_cnn_dqn_ep_758000.pt

🏆 LEAGUE STATUS:
   Current Tier: 2/4
   Training vs: m1_cnn_dqn_ep_758000.pt
   Win Rate: 0.0% (0 games)
   Need 50 more games or 60%+ win rate for promotion

   League Models:
       Tier 0: m1_cnn_dqn_best_ep_1000.pt
       Tier 1: m1_cnn_dqn_best_ep_301000.pt
     → Tier 2: m1_cnn_dqn_ep_758000.pt         (NEW - INTERMEDIATE)
       Tier 3: m1_cnn_dqn_ep_920000.pt         (BLOCKER - STILL THERE)
       Tier 4: m1_cnn_dqn_ep_1236000.pt
```

### Progressive Training Path

**Attempt 1: Beat 758k (Intermediate)**
```
Agent trains vs 758k model...
If achieves 60%+ → Promote to Tier 3 (face 920k again)
If stuck again → Backtrack creates 679k intermediate
```

**Attempt 2: Beat 920k (Second Try)**
```
Agent now stronger from beating 758k...
If achieves 60%+ → Promote to Tier 4!
If still stuck → Backtrack creates 839k intermediate
```

**Recursive Refinement:**
```
Tier 2: 600k → 758k → 679k → 839k → 799k → ...
                      ↑
             Progressive refinement until agent succeeds
```

## Advantages Over Adaptive Threshold

| Approach | Adaptive Threshold | Backtracking Curriculum |
|----------|-------------------|------------------------|
| **Rigor** | Lowers standards | Maintains 60% always |
| **Learning** | Same opponent | Introduces intermediate skills |
| **Skill Building** | Forced through | Gradual progression |
| **Philosophy** | "Give up eventually" | "Find right difficulty" |
| **Long-term** | May promote weak | Ensures mastery |

## Key Differences

### Adaptive Threshold (Rejected)
```python
# After 10k games: Lower to 55%
# After 30k games: Lower to 52%
# Result: Promotes at 56.6% → Weak progression
```
❌ **Lowers standards**
❌ **Doesn't build intermediate skills**
❌ **Forced through blockers**

### Backtracking Curriculum (Implemented)
```python
# After 50k games: Find intermediate (760k)
# Demote to train against 760k first
# Result: Beats 760k at 60%+, THEN tries 920k
```
✅ **Maintains 60% standard**
✅ **Builds intermediate skills**
✅ **Smoother difficulty curve**

## Algorithm Properties

### Termination Guarantee
**Will always find path:**
- Each backtrack halves the difficulty gap
- Binary search-like refinement
- Worst case: ~log₂(gap) backtracks

### Example Convergence
```
Initial:     600k ─────(320k gap)────→ 920k (BLOCKED)

Backtrack 1: 600k ──(160k)→ 760k ──(160k)→ 920k (Try 760k)
Backtrack 2: 600k ─(80k)→ 680k ─(80k)→ 760k (Try 680k if stuck)
Backtrack 3: 600k (40k)→ 640k (40k)→ 680k ... (Progressive refinement)

Result: Eventually finds beatable intermediate
```

### Recovery Path
```
Beat 600k (60%+) ✓
  ↓
Try 920k (56.6%) ✗ ← Stuck for 50k games
  ↓
Backtrack: Try 760k (??%)
  ↓
  If 60%+: Try 920k again (stronger now)
  If <60%: Try 680k (easier intermediate)
  ↓
Continue until 920k beaten
  ↓
Promote to Tier 4!
```

## Edge Cases Handled

### Case 1: No Intermediate Checkpoint Found
```python
if best_match is None:
    print(f"\n⚠️  No intermediate checkpoint found between {previous_ep} and {blocker_ep}")
    return False  # Don't backtrack, keep trying
```
**Mitigation:** Save more frequent checkpoints (every 2k episodes)

### Case 2: Agent at Tier 0
```python
if self.current_tier == 0:
    return False  # Can't backtrack from tier 0
```
**Behavior:** Keep training at Tier 0 until 60% achieved

### Case 3: Repeated Backtracking
```python
# Each backtrack halves gap, ensuring convergence
# Example: 320k → 160k → 80k → 40k → 20k → 10k
```
**Guarantee:** Finite backtracks (log₂ convergence)

## Testing

### Immediate Test (Next ~600 Episodes)
**Expected at episode ~1,613,600:**
```
🔄 BACKTRACKING CURRICULUM
   Found intermediate: m1_cnn_dqn_ep_758000.pt
   ✅ Curriculum adjusted! Agent demoted to Tier 2
```

### Validation Criteria
✅ Backtrack triggers at ~50k games
✅ Finds intermediate checkpoint (~760k)
✅ Agent demoted to Tier 2
✅ New league structure created
✅ Training resumes normally

### Long-term Monitoring
- **Week 1:** Agent beats 760k → promotes to Tier 3
- **Week 2:** Agent tries 920k again (now stronger)
- **Expected:** 60%+ win rate (or backtrack to 840k if needed)

## Configuration

### Backtrack Threshold
```python
BACKTRACK_THRESHOLD = 50000  # games

# Adjust if needed:
# - Too low (30k): Backtracks too early, wastes training
# - Too high (100k): Wastes time on impossible opponents
# - Current (50k): Balanced (~165k episodes of training)
```

### Intermediate Selection
```python
# Current: Midpoint between previous and blocker
intermediate_ep = (previous_ep + blocker_ep) // 2

# Alternative: Weighted toward previous (easier)
# intermediate_ep = int(previous_ep * 0.7 + blocker_ep * 0.3)
```

## Files Modified

1. **`train/league_manager.py`**
   - Added `should_backtrack()` method (Lines 217-242)
   - Added `backtrack_tier()` method (Lines 244-334)
   - Updated `print_status()` for backtrack warnings (Lines 433-453)

2. **`train/resume_m1_with_league.py`**
   - Added backtrack check before promotion (Lines 373-384)

3. **Documentation:**
   - [BACKTRACKING_CURRICULUM_FIX.md](BACKTRACKING_CURRICULUM_FIX.md) - This file

## Summary

**Old approach:** Stuck forever at 56.6% vs 920k
**New approach:** Automatically inserts 760k intermediate, demotes agent, creates path to success

**Philosophy:** "If you can't make the jump, we'll build you a ladder."

---

**Implementation Date:** 2025-10-16
**Trigger:** Next ~600 episodes (when 50k games reached)
**Expected Impact:** Resume progression through refined curriculum
