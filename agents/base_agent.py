"""Base agent interface for Connect 4 agents."""

from abc import ABC, abstractmethod
from typing import Optional, Any
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all Connect 4 agents.
    
    This interface ensures all agents can be used interchangeably in the game
    and training systems. Agents can be stateless (like random) or stateful
    (like RL agents that learn).
    """
    
    def __init__(self, player_id: int, name: str):
        """Initialize the agent.
        
        Args:
            player_id: Player ID (1 or 2) this agent represents
            name: Human-readable name for the agent
        """
        self.player_id = player_id
        self.name = name
        
    @abstractmethod
    def choose_action(self, board_state: np.ndarray, legal_moves: list) -> int:
        """Choose an action given the current board state.
        
        Args:
            board_state: Current board state as numpy array
            legal_moves: List of legal column indices
            
        Returns:
            Column index (0-indexed) to play
        """
        pass
        
    def observe(self, board_state: np.ndarray, action: int, reward: float, 
                next_state: Optional[np.ndarray], done: bool) -> None:
        """Observe a transition for learning (optional for stateless agents).
        
        Args:
            board_state: Previous board state
            action: Action that was taken  
            reward: Reward received
            next_state: Resulting board state (None if terminal)
            done: Whether the episode ended
        """
        pass  # Default implementation does nothing
        
    def reset_episode(self) -> None:
        """Reset any episode-specific state (optional)."""
        pass
        
    def save(self, filepath: str) -> None:
        """Save agent parameters/model (optional for stateless agents)."""
        pass
        
    def load(self, filepath: str) -> None:
        """Load agent parameters/model (optional for stateless agents)."""
        pass
        
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name} (Player {self.player_id})"