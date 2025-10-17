"""Connect 4 board implementation with game logic and state management."""

from typing import Optional, List, Tuple
import numpy as np


class Connect4Board:
    """A Connect 4 game board with all game logic and state management.
    
    The board is represented as a 2D numpy array where:
    - 0 represents empty cell
    - 1 represents player 1 (typically human or first agent)  
    - 2 represents player 2 (typically second agent)
    """
    
    def __init__(self, rows: int = 6, cols: int = 7):
        """Initialize a new Connect 4 board.
        
        Args:
            rows: Number of rows (default 6 for standard Connect 4)
            cols: Number of columns (default 7 for standard Connect 4)
        """
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = 1  # Player 1 starts
        
    def reset(self) -> np.ndarray:
        """Reset the board to initial empty state.
        
        Returns:
            The reset board state as a numpy array
        """
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.board.copy()
        
    def get_legal_moves(self) -> List[int]:
        """Get list of legal column indices where pieces can be dropped.
        
        Returns:
            List of column indices (0-indexed) where moves are legal
        """
        return [col for col in range(self.cols) if self.board[0, col] == 0]
        
    def is_legal_move(self, col: int) -> bool:
        """Check if dropping a piece in the given column is legal.
        
        Args:
            col: Column index (0-indexed)
            
        Returns:
            True if the move is legal, False otherwise
        """
        if col < 0 or col >= self.cols:
            return False
        return self.board[0, col] == 0
        
    def make_move(self, col: int, player: Optional[int] = None) -> bool:
        """Drop a piece in the specified column.
        
        Args:
            col: Column index (0-indexed) to drop piece
            player: Player number (1 or 2). If None, uses current_player
            
        Returns:
            True if move was successful, False if illegal move
        """
        if not self.is_legal_move(col):
            return False
            
        if player is None:
            player = self.current_player
            
        # Find the lowest empty row in the column
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = player
                break
                
        # Switch to next player only if using current_player
        if player == self.current_player:
            self.current_player = 3 - self.current_player  # Switch between 1 and 2
            
        return True
        
    def check_winner(self) -> Optional[int]:
        """Check if there's a winner on the board.
        
        Returns:
            Player number (1 or 2) if there's a winner, None otherwise
        """
        # Check horizontal wins
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if (self.board[row, col] != 0 and 
                    self.board[row, col] == self.board[row, col+1] == 
                    self.board[row, col+2] == self.board[row, col+3]):
                    return self.board[row, col]
                    
        # Check vertical wins
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if (self.board[row, col] != 0 and
                    self.board[row, col] == self.board[row+1, col] == 
                    self.board[row+2, col] == self.board[row+3, col]):
                    return self.board[row, col]
                    
        # Check diagonal wins (top-left to bottom-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if (self.board[row, col] != 0 and
                    self.board[row, col] == self.board[row+1, col+1] == 
                    self.board[row+2, col+2] == self.board[row+3, col+3]):
                    return self.board[row, col]
                    
        # Check diagonal wins (top-right to bottom-left)
        for row in range(self.rows - 3):
            for col in range(3, self.cols):
                if (self.board[row, col] != 0 and
                    self.board[row, col] == self.board[row+1, col-1] == 
                    self.board[row+2, col-2] == self.board[row+3, col-3]):
                    return self.board[row, col]
                    
        return None
        
    def is_draw(self) -> bool:
        """Check if the game is a draw (board full, no winner).
        
        Returns:
            True if the game is a draw, False otherwise
        """
        return len(self.get_legal_moves()) == 0 and self.check_winner() is None
        
    def is_terminal(self) -> bool:
        """Check if the game is in a terminal state (win or draw).
        
        Returns:
            True if game is over, False otherwise
        """
        return self.check_winner() is not None or self.is_draw()
        
    def get_state(self) -> np.ndarray:
        """Get the current board state.
        
        Returns:
            Copy of the current board state as numpy array
        """
        return self.board.copy()
        
    def render(self) -> str:
        """Create a string representation of the board for display.
        
        Returns:
            Multi-line string showing the current board state
        """
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        
        # Column numbers header
        lines.append(' '.join(str(i) for i in range(self.cols)))
        lines.append(' '.join('-' for _ in range(self.cols)))
        
        # Board rows
        for row in range(self.rows):
            line = ' '.join(symbols[self.board[row, col]] for col in range(self.cols))
            lines.append(line)
            
        return '\n'.join(lines)
        
    def copy(self) -> 'Connect4Board':
        """Create a deep copy of the current board state.
        
        Returns:
            New Connect4Board instance with identical state
        """
        new_board = Connect4Board(self.rows, self.cols)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        return new_board
    
    def find_winning_moves(self, player: int) -> List[int]:
        """Find all columns where player can win immediately.
        
        Args:
            player: Player ID (1 or 2)
            
        Returns:
            List of column indices where player can win
        """
        winning_moves = []
        for col in self.get_legal_moves():
            # Simulate the move
            temp_board = self.copy()
            temp_board.make_move(col, player)
            if temp_board.check_winner() == player:
                winning_moves.append(col)
        return winning_moves
    
    def find_blocking_moves(self, player: int) -> List[int]:
        """Find all columns where player must move to block opponent's win.
        
        Args:
            player: Player ID (1 or 2)
            
        Returns:
            List of column indices where player should block opponent
        """
        opponent = 3 - player
        return self.find_winning_moves(opponent)
    
    def has_winning_move(self, player: int) -> bool:
        """Check if player has any winning moves available.
        
        Args:
            player: Player ID (1 or 2)
            
        Returns:
            True if player can win in one move
        """
        return len(self.find_winning_moves(player)) > 0
    
    def has_blocking_move(self, player: int) -> bool:
        """Check if player needs to block opponent's winning move.
        
        Args:
            player: Player ID (1 or 2)
            
        Returns:
            True if opponent can win and must be blocked
        """
        return len(self.find_blocking_moves(player)) > 0