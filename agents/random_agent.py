"""Random agent that chooses moves randomly from legal moves."""

import random
from typing import List
import numpy as np

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that selects moves randomly from available legal moves.
    
    This serves as a baseline agent for comparison and as an opponent
    during early training phases. It's completely stateless and requires
    no learning.
    """
    
    def __init__(self, player_id: int, seed: int = None):
        """Initialize the random agent.
        
        Args:
            player_id: Player ID (1 or 2) this agent represents
            seed: Random seed for reproducibility (optional)
        """
        super().__init__(player_id, "Random")
        self.rng = random.Random(seed)
        
    def choose_action(self, board_state: np.ndarray, legal_moves: List[int]) -> int:
        """Choose a random move from legal moves.
        
        Args:
            board_state: Current board state (unused by random agent)
            legal_moves: List of legal column indices
            
        Returns:
            Randomly selected column index from legal moves
        """
        if not legal_moves:
            raise ValueError("No legal moves available")
            
        return self.rng.choice(legal_moves)
        
    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility.
        
        Args:
            seed: Random seed value
        """
        self.rng = random.Random(seed)