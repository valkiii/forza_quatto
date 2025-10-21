#!/usr/bin/env python3
"""Curriculum manager for smooth opponent transitions in training."""

import random
from typing import Optional, Tuple, Dict, Any
from enum import Enum


class CurriculumPhase(Enum):
    """Training curriculum phases."""
    RANDOM = "random"
    RANDOM_TO_HEURISTIC = "random_to_heuristic"
    HEURISTIC = "heuristic"
    HEURISTIC_TO_LEAGUE = "heuristic_to_league"
    LEAGUE = "league"


class CurriculumManager:
    """Manages smooth curriculum transitions to prevent catastrophic forgetting."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize curriculum manager.

        Args:
            config: Dictionary with curriculum configuration:
                - random_phase_end: Episode when random phase ends
                - heuristic_phase_end: Episode when heuristic phase ends
                - transition_episodes: Number of episodes for smooth transition
                - league_start: Episode when league play starts
        """
        self.random_phase_end = config.get("random_phase_end", 50000)
        self.heuristic_phase_end = config.get("heuristic_phase_end", 150000)
        self.transition_episodes = config.get("transition_episodes", 5000)
        self.league_start = config.get("league_start", 200000)

        # Track transition state
        self.current_phase = CurriculumPhase.RANDOM
        self.in_transition = False
        self.transition_start_episode = 0

    def get_phase(self, episode: int) -> CurriculumPhase:
        """Get current curriculum phase based on episode number.

        Args:
            episode: Current episode number

        Returns:
            Current curriculum phase
        """
        if episode < self.random_phase_end:
            return CurriculumPhase.RANDOM
        elif episode < self.random_phase_end + self.transition_episodes:
            return CurriculumPhase.RANDOM_TO_HEURISTIC
        elif episode < self.heuristic_phase_end:
            return CurriculumPhase.HEURISTIC
        elif episode < self.heuristic_phase_end + self.transition_episodes:
            return CurriculumPhase.HEURISTIC_TO_LEAGUE
        else:
            return CurriculumPhase.LEAGUE

    def get_opponent_probabilities(self, episode: int) -> Dict[str, float]:
        """Get opponent selection probabilities for current phase.

        During transitions, gradually shifts from old to new opponent type.

        Args:
            episode: Current episode number

        Returns:
            Dictionary with probabilities: {"random": p1, "heuristic": p2, "league": p3}
        """
        phase = self.get_phase(episode)

        if phase == CurriculumPhase.RANDOM:
            return {"random": 1.0, "heuristic": 0.0, "league": 0.0}

        elif phase == CurriculumPhase.RANDOM_TO_HEURISTIC:
            # Smooth transition: gradually increase heuristic, decrease random
            progress = (episode - self.random_phase_end) / self.transition_episodes
            random_prob = 1.0 - progress * 0.8  # Keep 20% random for diversity
            heuristic_prob = progress * 0.8
            return {"random": random_prob, "heuristic": heuristic_prob, "league": 0.0}

        elif phase == CurriculumPhase.HEURISTIC:
            # Mostly heuristic with some random for diversity
            return {"random": 0.2, "heuristic": 0.8, "league": 0.0}

        elif phase == CurriculumPhase.HEURISTIC_TO_LEAGUE:
            # Smooth transition: gradually introduce league opponents
            progress = (episode - self.heuristic_phase_end) / self.transition_episodes
            random_prob = 0.2
            heuristic_prob = 0.8 - progress * 0.4  # Reduce heuristic from 80% to 40%
            league_prob = progress * 0.4
            return {"random": random_prob, "heuristic": heuristic_prob, "league": league_prob}

        else:  # LEAGUE phase
            return {"random": 0.2, "heuristic": 0.4, "league": 0.4}

    def should_reset_target_network(self, episode: int) -> bool:
        """Check if target network should be reset at curriculum boundary.

        Resetting helps prevent stale Q-values from contaminating new phase.

        Args:
            episode: Current episode number

        Returns:
            True if target network should be reset
        """
        # Reset at the start of each transition phase
        return episode in [
            self.random_phase_end,
            self.heuristic_phase_end
        ]

    def get_epsilon_boost(self, episode: int, base_epsilon: float) -> float:
        """Get epsilon boost at curriculum transitions for re-exploration.

        Args:
            episode: Current episode number
            base_epsilon: Current epsilon value

        Returns:
            Boosted epsilon value
        """
        phase = self.get_phase(episode)

        # Boost exploration during transitions
        if phase in [CurriculumPhase.RANDOM_TO_HEURISTIC, CurriculumPhase.HEURISTIC_TO_LEAGUE]:
            progress = self._get_transition_progress(episode, phase)
            # Peak boost at start of transition, gradually decrease
            boost = 0.3 * (1.0 - progress)
            return min(1.0, base_epsilon + boost)

        return base_epsilon

    def _get_transition_progress(self, episode: int, phase: CurriculumPhase) -> float:
        """Get progress through current transition phase (0.0 to 1.0).

        Args:
            episode: Current episode number
            phase: Current curriculum phase

        Returns:
            Progress through transition (0.0 = start, 1.0 = end)
        """
        if phase == CurriculumPhase.RANDOM_TO_HEURISTIC:
            return (episode - self.random_phase_end) / self.transition_episodes
        elif phase == CurriculumPhase.HEURISTIC_TO_LEAGUE:
            return (episode - self.heuristic_phase_end) / self.transition_episodes
        return 0.0

    def get_phase_description(self, episode: int) -> str:
        """Get human-readable description of current phase.

        Args:
            episode: Current episode number

        Returns:
            Phase description string
        """
        phase = self.get_phase(episode)
        probs = self.get_opponent_probabilities(episode)

        if phase == CurriculumPhase.RANDOM:
            return "Random opponents (foundation learning)"
        elif phase == CurriculumPhase.RANDOM_TO_HEURISTIC:
            return f"Transitioning to Heuristic ({probs['heuristic']*100:.0f}% heuristic)"
        elif phase == CurriculumPhase.HEURISTIC:
            return "Heuristic opponents (strategic learning)"
        elif phase == CurriculumPhase.HEURISTIC_TO_LEAGUE:
            return f"Transitioning to League ({probs['league']*100:.0f}% league)"
        else:
            return "League play (advanced tactics)"


def select_opponent_type(curriculum: CurriculumManager, episode: int, rng: random.Random = None) -> str:
    """Select opponent type based on curriculum probabilities.

    Args:
        curriculum: CurriculumManager instance
        episode: Current episode number
        rng: Random number generator (optional)

    Returns:
        Opponent type: "random", "heuristic", or "league"
    """
    if rng is None:
        rng = random.Random()

    probs = curriculum.get_opponent_probabilities(episode)

    # Sample from probabilities
    r = rng.random()
    if r < probs["random"]:
        return "random"
    elif r < probs["random"] + probs["heuristic"]:
        return "heuristic"
    else:
        return "league"


if __name__ == "__main__":
    # Test curriculum manager
    print("Testing Curriculum Manager")
    print("=" * 60)

    config = {
        "random_phase_end": 50000,
        "heuristic_phase_end": 150000,
        "transition_episodes": 5000,
        "league_start": 200000
    }

    curriculum = CurriculumManager(config)

    # Test key episode points
    test_episodes = [0, 25000, 49999, 50000, 52500, 55000, 100000, 149999, 150000, 152500, 200000, 250000]

    print("\nCurriculum Schedule:")
    print("-" * 60)
    for episode in test_episodes:
        phase = curriculum.get_phase(episode)
        probs = curriculum.get_opponent_probabilities(episode)
        desc = curriculum.get_phase_description(episode)
        reset = curriculum.should_reset_target_network(episode)

        print(f"\nEpisode {episode:,}:")
        print(f"  Phase: {phase.value}")
        print(f"  Description: {desc}")
        print(f"  Probabilities: Random={probs['random']:.2f}, Heuristic={probs['heuristic']:.2f}, League={probs['league']:.2f}")
        print(f"  Reset target network: {reset}")

    print("\n" + "=" * 60)
    print("✓ Curriculum manager tested successfully!")
