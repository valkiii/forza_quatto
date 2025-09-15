#!/usr/bin/env python3
"""Test epsilon decay calculation."""

def test_epsilon_decay():
    """Test the epsilon decay progression."""
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.99995
    
    print("🧪 EPSILON DECAY TEST")
    print("=" * 30)
    print(f"Start: {epsilon_start}")
    print(f"End: {epsilon_end}")
    print(f"Decay: {epsilon_decay}")
    print()
    
    epsilon = epsilon_start
    test_episodes = [100, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000]
    
    print("Episode -> Expected Epsilon")
    for episode in test_episodes:
        # Calculate epsilon after episode steps
        epsilon_at_episode = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        print(f"{episode:>6,} -> {epsilon_at_episode:.4f}")
    
    print()
    print("📊 ANALYSIS:")
    
    # Find when epsilon reaches certain thresholds
    thresholds = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    
    for threshold in thresholds:
        if threshold >= epsilon_end:
            # Calculate episode when threshold is reached
            import math
            episode_to_reach = math.log(threshold / epsilon_start) / math.log(epsilon_decay)
            print(f"Epsilon {threshold:.2f} reached at episode: {episode_to_reach:,.0f}")
    
    print()
    print("🎯 At episode 62000 (your observation):")
    epsilon_62k = max(epsilon_end, epsilon_start * (epsilon_decay ** 62000))
    print(f"Expected epsilon: {epsilon_62k:.4f}")
    if epsilon_62k > 0.9:
        print("❌ This suggests epsilon decay is too slow!")
    else:
        print("✅ Epsilon decay seems reasonable")

if __name__ == "__main__":
    test_epsilon_decay()