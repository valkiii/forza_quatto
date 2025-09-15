"""Tabular Q-learning agent implemented from scratch."""

import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

from .base_agent import BaseAgent


class QLearningAgent(BaseAgent):
    """Tabular Q-learning agent with epsilon-greedy exploration.
    
    This implementation uses a relative state encoding where the agent's pieces
    are always represented as 1 and opponent's as 2, reducing the state space
    and improving learning efficiency.
    """
    
    def __init__(self, player_id: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay: float = 0.995,
                 seed: int = None):
        """Initialize Q-learning agent.
        
        Args:
            player_id: Player ID (1 or 2) this agent represents
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon after each episode
            seed: Random seed for reproducibility
        """
        super().__init__(player_id, "Q-Learning")
        
        # Q-learning hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q-table: maps (state, action) -> Q-value
        # Using defaultdict for automatic initialization to 0
        self.q_table: Dict[Tuple[str, int], float] = defaultdict(float)
        
        # Experience tracking for learning
        self.last_state = None
        self.last_action = None
        
        # Random number generator
        self.rng = random.Random(seed)
        
    def encode_state(self, board_state: np.ndarray) -> str:
        """Encode board state as string from agent's perspective.
        
        The relative encoding represents:
        - 0: Empty cell
        - 1: Agent's piece (regardless of actual player_id)
        - 2: Opponent's piece
        
        This reduces state space by factor of 2 and helps generalization.
        
        Args:
            board_state: Raw board state with actual player IDs
            
        Returns:
            String representation of relative board state
        """
        # Create relative board from agent's perspective
        relative_board = np.zeros_like(board_state)
        
        # Agent's pieces become 1, opponent's become 2
        relative_board[board_state == self.player_id] = 1
        relative_board[board_state == (3 - self.player_id)] = 2
        
        # Convert to string for hashing
        return ''.join(relative_board.flatten().astype(str))
        
    def choose_action(self, board_state: np.ndarray, legal_moves: List[int]) -> int:
        """Choose action using epsilon-greedy strategy.
        
        Args:
            board_state: Current board state
            legal_moves: List of legal column indices
            
        Returns:
            Selected column index
        """
        if not legal_moves:
            raise ValueError("No legal moves available")
            
        state_key = self.encode_state(board_state)
        
        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            # Explore: random action
            action = self.rng.choice(legal_moves)
        else:
            # Exploit: choose action with highest Q-value
            action = self._get_best_action(state_key, legal_moves)
            
        # Store for learning
        self.last_state = state_key
        self.last_action = action
        
        return action
        
    def _get_best_action(self, state_key: str, legal_moves: List[int]) -> int:
        """Get action with highest Q-value for the given state.
        
        Args:
            state_key: Encoded state string
            legal_moves: Available actions
            
        Returns:
            Action with highest Q-value (random tie-breaking)
        """
        # Get Q-values for all legal actions
        q_values = [(self.q_table[(state_key, action)], action) for action in legal_moves]
        
        # Find maximum Q-value
        max_q = max(q_values)[0]
        
        # Get all actions with maximum Q-value (for tie-breaking)
        best_actions = [action for q_val, action in q_values if q_val == max_q]
        
        return self.rng.choice(best_actions)
        
    def observe(self, board_state: np.ndarray, action: int, reward: float,
                next_state: Optional[np.ndarray], done: bool) -> None:
        """Update Q-values based on observed transition.
        
        Implements the Q-learning update rule:
        Q(s,a) += α[r + γ*max_a'(Q(s',a')) - Q(s,a)]
        
        Args:
            board_state: Previous board state (before action)
            action: Action that was taken
            reward: Immediate reward received
            next_state: Resulting board state (None if terminal)
            done: Whether episode ended
        """
        if self.last_state is None or self.last_action is None:
            return  # No previous experience to learn from
            
        # Current Q-value
        current_q = self.q_table[(self.last_state, self.last_action)]
        
        if done or next_state is None:
            # Terminal state: no future value
            target_q = reward
        else:
            # Non-terminal: include discounted future value
            next_state_key = self.encode_state(next_state)
            legal_next_moves = self._get_legal_moves_from_state(next_state)
            
            if legal_next_moves:
                # Max Q-value for next state
                max_next_q = max(self.q_table[(next_state_key, a)] for a in legal_next_moves)
            else:
                max_next_q = 0.0  # No legal moves (shouldn't happen in Connect 4)
                
            target_q = reward + self.discount_factor * max_next_q
            
        # Q-learning update
        self.q_table[(self.last_state, self.last_action)] += \
            self.learning_rate * (target_q - current_q)
            
    def _get_legal_moves_from_state(self, board_state: np.ndarray) -> List[int]:
        """Extract legal moves from board state.
        
        Args:
            board_state: Board state array
            
        Returns:
            List of legal column indices
        """
        return [col for col in range(board_state.shape[1]) if board_state[0, col] == 0]
        
    def reset_episode(self) -> None:
        """Reset episode-specific state and decay epsilon."""
        self.last_state = None
        self.last_action = None
        
        # Decay epsilon for less exploration over time
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def get_q_value(self, board_state: np.ndarray, action: int) -> float:
        """Get Q-value for a specific state-action pair.
        
        Args:
            board_state: Board state
            action: Action (column index)
            
        Returns:
            Q-value for the state-action pair
        """
        state_key = self.encode_state(board_state)
        return self.q_table[(state_key, action)]
        
    def get_policy(self, board_state: np.ndarray, legal_moves: List[int]) -> Dict[int, float]:
        """Get action probabilities under current policy.
        
        Args:
            board_state: Current board state
            legal_moves: Available actions
            
        Returns:
            Dictionary mapping actions to their selection probabilities
        """
        state_key = self.encode_state(board_state)
        
        if not legal_moves:
            return {}
            
        # Get Q-values for legal actions
        q_values = {action: self.q_table[(state_key, action)] for action in legal_moves}
        best_actions = [a for a, q in q_values.items() if q == max(q_values.values())]
        
        # Epsilon-greedy probabilities
        num_actions = len(legal_moves)
        num_best = len(best_actions)
        
        policy = {}
        for action in legal_moves:
            if action in best_actions:
                # Best actions get (1-epsilon)/num_best + epsilon/num_actions
                policy[action] = (1 - self.epsilon) / num_best + self.epsilon / num_actions
            else:
                # Other actions get epsilon/num_actions
                policy[action] = self.epsilon / num_actions
                
        return policy
        
    def save(self, filepath: str) -> None:
        """Save Q-table and agent parameters.
        
        Args:
            filepath: Path to save the agent
        """
        data = {
            'q_table': dict(self.q_table),  # Convert defaultdict to regular dict
            'player_id': self.player_id,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, filepath: str) -> None:
        """Load Q-table and agent parameters.
        
        Args:
            filepath: Path to load the agent from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.q_table = defaultdict(float, data['q_table'])
        self.player_id = data['player_id']
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.epsilon = data['epsilon']
        self.epsilon_end = data['epsilon_end']
        self.epsilon_decay = data['epsilon_decay']
        
    def get_stats(self) -> Dict[str, any]:
        """Get training statistics.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            'q_table_size': len(self.q_table),
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor
        }