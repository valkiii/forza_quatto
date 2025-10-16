#!/usr/bin/env python3
"""League Manager for progressive self-play training with past model versions."""

import os
import glob
import random
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

class LeagueManager:
    """
    Manages a league of past model checkpoints for progressive self-play training.

    Instead of playing against a single frozen copy, the agent plays against
    multiple past versions to create a curriculum of increasingly difficult opponents.
    """

    def __init__(self,
                 model_dir: str,
                 league_size: int = 5,
                 win_threshold: float = 0.60,
                 min_games_to_promote: int = 50,
                 selection_strategy: str = "progressive"):
        """
        Initialize League Manager.

        Args:
            model_dir: Directory containing saved models
            league_size: Maximum number of models to keep in league
            win_threshold: Win rate required to promote to next league level
            min_games_to_promote: Minimum games needed before considering promotion
            selection_strategy: How to select opponents from league
                - "progressive": Focus on current tier, occasionally harder
                - "balanced": Equal probability across all league members
                - "curriculum": Start easy, gradually increase difficulty
        """
        self.model_dir = model_dir
        self.league_size = league_size
        self.win_threshold = win_threshold
        self.min_games_to_promote = min_games_to_promote
        self.selection_strategy = selection_strategy

        # League state
        self.league_models: List[Dict] = []  # List of model info dicts
        self.current_tier = 0  # Current difficulty tier
        self.performance_tracker = defaultdict(lambda: {"wins": 0, "games": 0})

        # Load initial league from saved models
        self._initialize_league()

    def _initialize_league(self):
        """Initialize league from existing model checkpoints."""
        if not os.path.exists(self.model_dir):
            print(f"⚠️ Model directory not found: {self.model_dir}")
            return

        # Find all checkpoint models
        model_files = sorted(
            glob.glob(os.path.join(self.model_dir, "*.pt")),
            key=lambda x: os.path.getmtime(x)
        )

        if not model_files:
            print(f"⚠️ No models found in {self.model_dir}")
            return

        # Select diverse checkpoints for initial league
        # Strategy: Take models from different training stages
        if len(model_files) <= self.league_size:
            selected_models = model_files
        else:
            # Sample evenly across training progression
            indices = np.linspace(0, len(model_files) - 1, self.league_size, dtype=int)
            selected_models = [model_files[i] for i in indices]

        # Load league models
        for path in selected_models:
            episode_num = self._extract_episode_number(path)
            self.league_models.append({
                'path': path,
                'episode': episode_num,
                'name': os.path.basename(path),
                'tier': len(self.league_models),  # Earlier models = lower tier
                'wins_against': 0,
                'games_against': 0
            })

        print(f"\n🏆 LEAGUE INITIALIZED with {len(self.league_models)} models:")
        for i, model in enumerate(self.league_models):
            print(f"   Tier {i}: {model['name']} (Episode {model['episode']:,})")

    def _extract_episode_number(self, model_path: str) -> int:
        """Extract episode number from model filename."""
        import re
        filename = os.path.basename(model_path)

        # Try to find episode number pattern
        match = re.search(r'ep_(\d+)', filename)
        if match:
            return int(match.group(1))

        # Fallback: use file creation time as proxy
        return 0

    def select_opponent(self, current_episode: int) -> Optional[Dict]:
        """
        Select an opponent from the league based on strategy.

        Args:
            current_episode: Current training episode number

        Returns:
            Model info dict or None if league is empty
        """
        if not self.league_models:
            return None

        if self.selection_strategy == "progressive":
            return self._progressive_selection()
        elif self.selection_strategy == "balanced":
            return self._balanced_selection()
        elif self.selection_strategy == "curriculum":
            return self._curriculum_selection(current_episode)
        else:
            # Default: random selection
            return random.choice(self.league_models)

    def _progressive_selection(self) -> Dict:
        """
        Progressive selection: Focus on current tier, occasionally pick harder.

        Strategy:
        - 60%: Current tier (model agent is trying to beat)
        - 25%: Random from league (for diversity)
        - 15%: Hardest model (aspirational)
        """
        rand_val = random.random()

        if rand_val < 0.60 and self.current_tier < len(self.league_models):
            # Current tier
            return self.league_models[self.current_tier]
        elif rand_val < 0.85:
            # Random from league
            return random.choice(self.league_models)
        else:
            # Hardest model (highest tier)
            return self.league_models[-1]

    def _balanced_selection(self) -> Dict:
        """Balanced selection: Equal probability for all league members."""
        return random.choice(self.league_models)

    def _curriculum_selection(self, current_episode: int) -> Dict:
        """
        Curriculum selection: Gradually increase difficulty based on episode.

        Early episodes: easier models
        Later episodes: harder models
        """
        # Map episode to tier (0 to league_size-1)
        # This creates a smooth curriculum over training
        progress = min(1.0, current_episode / 200000)  # Normalize to [0, 1]
        target_tier = int(progress * (len(self.league_models) - 1))

        # Add some randomness around target tier
        tier_range = max(1, len(self.league_models) // 3)
        lower_bound = max(0, target_tier - tier_range // 2)
        upper_bound = min(len(self.league_models) - 1, target_tier + tier_range // 2)

        tier = random.randint(lower_bound, upper_bound)
        return self.league_models[tier]

    def record_game_result(self, model_tier: int, agent_won: bool):
        """
        Record result of a game against a league opponent.

        Args:
            model_tier: Tier of the opponent model
            agent_won: Whether the training agent won
        """
        if model_tier < 0 or model_tier >= len(self.league_models):
            return

        model = self.league_models[model_tier]
        model['games_against'] += 1
        if agent_won:
            model['wins_against'] += 1

        # Update performance tracker for current tier
        tier_key = f"tier_{self.current_tier}"
        self.performance_tracker[tier_key]["games"] += 1
        if agent_won and model_tier == self.current_tier:
            self.performance_tracker[tier_key]["wins"] += 1

    def should_promote(self) -> bool:
        """
        Check if agent should be promoted to next tier.

        Returns:
            True if agent has beaten current tier consistently
        """
        if self.current_tier >= len(self.league_models) - 1:
            # Already at highest tier
            return False

        tier_key = f"tier_{self.current_tier}"
        stats = self.performance_tracker[tier_key]

        if stats["games"] < self.min_games_to_promote:
            # Not enough games yet
            return False

        win_rate = stats["wins"] / stats["games"]
        return win_rate >= self.win_threshold

    def should_backtrack(self) -> bool:
        """
        Check if agent is stuck and should backtrack to easier opponent.

        If agent can't beat current tier after many games (>50k), we backtrack:
        1. Replace previous tier with an intermediate checkpoint
        2. Move current blocker down to previous tier
        3. Reset and try again with easier progression

        Returns:
            True if agent is stuck and needs curriculum adjustment
        """
        if self.current_tier == 0:
            # Can't backtrack from tier 0
            return False

        tier_key = f"tier_{self.current_tier}"
        stats = self.performance_tracker[tier_key]

        # Stuck if played >50k games without reaching 60% threshold
        if stats["games"] > 50000:
            win_rate = stats["wins"] / stats["games"]
            if win_rate < self.win_threshold:
                return True

        return False

    def backtrack_tier(self) -> bool:
        """
        Backtrack curriculum when stuck on a tier.

        Strategy:
        1. Find an intermediate checkpoint between (current_tier - 1) and current_tier
        2. Replace the (current_tier - 1) model with the intermediate
        3. Move current blocking model down to previous tier
        4. Demote agent back one tier
        5. Reset stats and try smoother progression

        Returns:
            True if backtracking occurred
        """
        if not self.should_backtrack():
            return False

        current_blocker = self.league_models[self.current_tier]
        previous_model = self.league_models[self.current_tier - 1]

        # Find intermediate checkpoint (halfway between previous and blocker)
        previous_ep = previous_model['episode']
        blocker_ep = current_blocker['episode']
        intermediate_ep = (previous_ep + blocker_ep) // 2

        # Try to find a model close to intermediate episode
        import glob
        potential_models = glob.glob(os.path.join(self.model_dir, "*.pt"))

        # Find closest model to intermediate_ep
        best_match = None
        best_distance = float('inf')

        for model_path in potential_models:
            # Extract episode number from filename
            basename = os.path.basename(model_path)
            if 'ep_' in basename:
                try:
                    ep_str = basename.split('ep_')[1].split('.pt')[0]
                    ep_num = int(ep_str)

                    # Must be between previous and blocker
                    if previous_ep < ep_num < blocker_ep:
                        distance = abs(ep_num - intermediate_ep)
                        if distance < best_distance:
                            best_distance = distance
                            best_match = (model_path, ep_num)
                except (ValueError, IndexError):
                    continue

        if best_match is None:
            print(f"\n⚠️  No intermediate checkpoint found between {previous_ep} and {blocker_ep}")
            return False

        intermediate_path, intermediate_ep = best_match

        print(f"\n🔄 BACKTRACKING CURRICULUM (stuck at Tier {self.current_tier} for {self.performance_tracker[f'tier_{self.current_tier}']['games']:,} games)")
        print(f"   Current blocker: {current_blocker['name']} (Episode {blocker_ep:,})")
        print(f"   Previous model: {previous_model['name']} (Episode {previous_ep:,})")
        print(f"   Found intermediate: {os.path.basename(intermediate_path)} (Episode {intermediate_ep:,})")
        print()
        print(f"   Strategy:")
        print(f"   1. Replace Tier {self.current_tier - 1} with intermediate checkpoint")
        print(f"   2. Move blocker to Tier {self.current_tier - 1}")
        print(f"   3. Demote agent back to Tier {self.current_tier - 1}")
        print(f"   4. Create smoother difficulty progression")

        # Replace previous tier with intermediate
        self.league_models[self.current_tier - 1] = {
            'path': intermediate_path,
            'episode': intermediate_ep,
            'name': os.path.basename(intermediate_path),
            'tier': self.current_tier - 1,
            'wins_against': 0,
            'games_against': 0
        }

        # Move blocker down to current tier (it stays where it is, but we'll demote agent)
        # No need to move it, just reset agent's position

        # Demote agent back one tier
        self.current_tier -= 1

        # Reset performance tracking for new setup
        for i in range(len(self.league_models)):
            self.performance_tracker[f"tier_{i}"] = {"wins": 0, "games": 0}

        print(f"\n   ✅ Curriculum adjusted! Agent demoted to Tier {self.current_tier}")
        print(f"   New opponent: {self.league_models[self.current_tier]['name']}")

        return True

    def promote_tier(self) -> bool:
        """
        Promote agent to next tier if eligible.

        Returns:
            True if promotion occurred
        """
        if self.should_promote():
            # Get stats for promotion message
            tier_key = f"tier_{self.current_tier}"
            stats = self.performance_tracker[tier_key]
            win_rate = stats["wins"] / stats["games"] if stats["games"] > 0 else 0

            self.current_tier += 1
            print(f"\n🎉 PROMOTED to Tier {self.current_tier}!")
            print(f"   Beat Tier {self.current_tier - 1} with {win_rate:.1%} win rate ({stats['wins']}/{stats['games']} games)")
            print(f"   Now training against: {self.league_models[self.current_tier]['name']}")

            # Reset performance tracker for new tier
            tier_key = f"tier_{self.current_tier}"
            self.performance_tracker[tier_key] = {"wins": 0, "games": 0}

            return True
        return False

    def add_checkpoint_to_league(self, model_path: str, episode: int) -> bool:
        """
        Add a new checkpoint to the league.

        Args:
            model_path: Path to the model checkpoint
            episode: Episode number when saved

        Returns:
            True if model was added to league
        """
        # Only add if agent has beaten current league champion
        if self.current_tier < len(self.league_models) - 1:
            # Not ready to expand league yet
            return False

        # Check if agent has strong win rate against champion
        tier_key = f"tier_{self.current_tier}"
        stats = self.performance_tracker[tier_key]

        if stats["games"] >= self.min_games_to_promote:
            win_rate = stats["wins"] / stats["games"]
            if win_rate >= self.win_threshold:
                # Add new model to league
                new_model = {
                    'path': model_path,
                    'episode': episode,
                    'name': os.path.basename(model_path),
                    'tier': len(self.league_models),
                    'wins_against': 0,
                    'games_against': 0
                }

                self.league_models.append(new_model)

                # Maintain league size limit
                if len(self.league_models) > self.league_size:
                    # Remove weakest (earliest) model
                    removed = self.league_models.pop(0)
                    print(f"   Removed {removed['name']} from league")

                    # Update tiers
                    for i, model in enumerate(self.league_models):
                        model['tier'] = i

                    # Adjust current tier
                    self.current_tier = max(0, self.current_tier - 1)

                print(f"   ✨ Added {new_model['name']} to league (Tier {new_model['tier']})")
                return True

        return False

    def get_league_status(self) -> Dict:
        """Get current league status for monitoring."""
        tier_key = f"tier_{self.current_tier}"
        current_stats = self.performance_tracker[tier_key]

        win_rate = 0.0
        if current_stats["games"] > 0:
            win_rate = current_stats["wins"] / current_stats["games"]

        return {
            'current_tier': self.current_tier,
            'total_tiers': len(self.league_models),
            'games_vs_current': current_stats["games"],
            'win_rate_vs_current': win_rate,
            'promotion_ready': self.should_promote(),
            'current_opponent': self.league_models[self.current_tier]['name'] if self.league_models else None,
            'league_models': [m['name'] for m in self.league_models]
        }

    def print_status(self):
        """Print league status summary."""
        status = self.get_league_status()

        print(f"\n🏆 LEAGUE STATUS:")
        print(f"   Current Tier: {status['current_tier']}/{status['total_tiers'] - 1}")
        print(f"   Training vs: {status['current_opponent']}")
        print(f"   Win Rate: {status['win_rate_vs_current']:.1%} ({status['games_vs_current']} games)")

        if status['promotion_ready']:
            print(f"   ⭐ READY FOR PROMOTION! (≥{self.win_threshold:.0%} win rate)")
        else:
            games_needed = max(0, self.min_games_to_promote - status['games_vs_current'])

            # Check if approaching backtrack threshold
            if status['games_vs_current'] > 40000 and status['win_rate_vs_current'] < self.win_threshold:
                print(f"   ⚠️  Warning: Approaching backtrack threshold (50k games)")
            elif status['games_vs_current'] > 50000:
                print(f"   🔄 Will backtrack curriculum next evaluation (>50k games, <60% win rate)")

            print(f"   Need {games_needed} more games or {self.win_threshold:.0%}+ win rate for promotion")

        print(f"\n   League Models:")
        for i, name in enumerate(status['league_models']):
            marker = "→" if i == status['current_tier'] else " "
            print(f"     {marker} Tier {i}: {name}")


class LeagueAgent:
    """Wrapper agent that loads and uses a model from the league."""

    def __init__(self, model_path: str, agent_class, player_id: int = 2,
                 architecture: str = "m1_optimized", epsilon: float = 0.05):
        """
        Initialize league agent from saved model.

        Args:
            model_path: Path to saved model
            agent_class: Class to instantiate (e.g., CNNDuelingDQNAgent)
            player_id: Player ID for this agent
            architecture: Model architecture type
            epsilon: Exploration rate (small value for variety)
        """
        self.model_path = model_path
        self.player_id = player_id
        self.name = f"League-{os.path.basename(model_path)}"

        # Create agent and load weights
        self.agent = agent_class(
            player_id=player_id,
            input_channels=2,
            action_size=7,
            hidden_size=64,
            architecture=architecture,
            seed=42
        )

        self.agent.load(model_path, keep_player_id=True)
        self.agent.epsilon = epsilon  # Small exploration for variety

    def choose_action(self, board_state, legal_moves):
        """Choose action using loaded model."""
        return self.agent.choose_action(board_state, legal_moves)

    def reset_episode(self):
        """Reset episode."""
        if hasattr(self.agent, 'reset_episode'):
            self.agent.reset_episode()

    def observe(self, *args, **kwargs):
        """Observe (no-op for league agent - doesn't learn)."""
        pass


if __name__ == "__main__":
    # Demo: Initialize league manager
    print("🏆 LEAGUE MANAGER DEMO")
    print("=" * 60)

    league = LeagueManager(
        model_dir="models_m1_cnn",
        league_size=5,
        win_threshold=0.60,
        selection_strategy="progressive"
    )

    league.print_status()

    print("\n📊 OPPONENT SELECTION DEMO (10 selections):")
    for i in range(10):
        opponent = league.select_opponent(current_episode=150000 + i * 1000)
        if opponent:
            print(f"   Selection {i+1}: Tier {opponent['tier']} - {opponent['name']}")

    print("\n✅ League Manager ready for training integration")
