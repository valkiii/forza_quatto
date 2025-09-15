"""Heuristic agent that uses simple rules to play Connect 4."""

import random
from typing import List, Tuple, Optional
import numpy as np

from .base_agent import BaseAgent


class HeuristicAgent(BaseAgent):
    """Rule-based agent that checks for wins and blocks threats.
    
    Strategy priority:
    1. Win immediately if possible (4 in a row)
    2. Block opponent from winning (block their 3 in a row)
    3. Create threats (make own 3 in a row)
    4. Play center columns (better position)
    5. Random fallback
    """
    
    def __init__(self, player_id: int, seed: int = None):
        """Initialize the heuristic agent.
        
        Args:
            player_id: Player ID (1 or 2) this agent represents
            seed: Random seed for tie-breaking (optional)
        """
        super().__init__(player_id, "Heuristic")
        self.opponent_id = 3 - player_id  # 1→2, 2→1
        self.rng = random.Random(seed)
        
    def choose_action(self, board_state: np.ndarray, legal_moves: List[int]) -> int:
        """Choose action using heuristic strategy.
        
        Args:
            board_state: Current board state
            legal_moves: List of legal column indices
            
        Returns:
            Column index to play
        """
        if not legal_moves:
            raise ValueError("No legal moves available")
            
        # Priority 1: Win immediately (our 4-in-a-row)
        winning_moves = self._find_moves_by_connection_count(board_state, self.player_id, legal_moves, 4)
        if winning_moves:
            return self.rng.choice(winning_moves)
            
        # Priority 2: Block opponent wins (their 4-in-a-row)
        blocking_moves = self._find_moves_by_connection_count(board_state, self.opponent_id, legal_moves, 4)
        if blocking_moves:
            return self.rng.choice(blocking_moves)
            
        # Priority 3: Create threats (our 3-in-a-row)
        threat_moves = self._find_moves_by_connection_count(board_state, self.player_id, legal_moves, 3)
        if threat_moves:
            return self.rng.choice(threat_moves)
            
        # Priority 4: Prefer center columns
        center_moves = self._prefer_center_moves(legal_moves)
        if center_moves:
            return self.rng.choice(center_moves)
            
        # Fallback: Random move
        return self.rng.choice(legal_moves)
        
    def _find_moves_by_connection_count(self, board: np.ndarray, player: int,
                                       legal_moves: List[int], target_count: int) -> List[int]:
        """Find moves that create a specific number of connections for a player.
        
        This unified function handles both winning moves (4+ connections) and threat moves 
        (3+ connections) by varying the target_count parameter. The perspective is defined
        by which player we're checking for.
        
        Args:
            board: Current board state
            player: Player ID to check connections for
            legal_moves: Available moves to check
            target_count: Minimum number of connections needed (3 for threats, 4 for wins)
            
        Returns:
            List of columns that create >= target_count connections for the player
        """
        good_moves = []
        
        for col in legal_moves:
            # Simulate the move
            test_board = board.copy()
            row = self._get_drop_row(test_board, col)
            if row is not None:
                test_board[row, col] = player
                
                # Check if this creates the target number of connections
                connection_count = self._count_connections_at_position(test_board, row, col, player)
                if connection_count >= target_count:
                    good_moves.append(col)
                    
        return good_moves
        
    def _prefer_center_moves(self, legal_moves: List[int]) -> List[int]:
        """Prefer moves closer to the center of the board.
        
        Args:
            legal_moves: Available moves
            
        Returns:
            Legal moves sorted by preference for center columns
        """
        center_col = 3  # Center of standard 7-column board
        
        # Sort by distance from center, keep only closest ones
        center_distance = [(abs(col - center_col), col) for col in legal_moves]
        center_distance.sort()
        
        # Return columns with minimum distance from center
        min_distance = center_distance[0][0]
        return [col for dist, col in center_distance if dist == min_distance]
        
    def _get_drop_row(self, board: np.ndarray, col: int) -> Optional[int]:
        """Find the row where a piece would land in the given column.
        
        Args:
            board: Current board state
            col: Column index
            
        Returns:
            Row index where piece lands, or None if column is full
        """
        rows = board.shape[0]
        for row in range(rows - 1, -1, -1):
            if board[row, col] == 0:
                return row
        return None
        
        
    def _count_connections_at_position(self, board: np.ndarray, row: int, 
                                     col: int, player: int) -> int:
        """Count the longest line through the given position for the player.
        
        This is the core algorithm that enables both winning move detection and threat
        detection. By checking all 4 directions (horizontal, vertical, both diagonals)
        and counting connected pieces in each direction, we can determine if placing
        a piece here creates 2, 3, 4+ in a row.
        
        The unified approach means:
        - target_count=4 → winning moves
        - target_count=3 → threat creation  
        - Works for any player (self or opponent)
        
        Args:
            board: Board state
            row: Row position
            col: Column position
            player: Player ID
            
        Returns:
            Length of longest line of connected pieces through this position
        """
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical  
            (1, 1),   # Diagonal /
            (1, -1)   # Diagonal \
        ]
        
        max_length = 1  # Count the piece itself
        
        for dr, dc in directions:
            length = 1  # Start with the piece itself
            
            # Count in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and 
                   board[r, c] == player):
                length += 1
                r, c = r + dr, c + dc
                
            # Count in negative direction  
            r, c = row - dr, col - dc
            while (0 <= r < board.shape[0] and 0 <= c < board.shape[1] and
                   board[r, c] == player):
                length += 1
                r, c = r - dr, c - dc
                
            max_length = max(max_length, length)
            
        return max_length