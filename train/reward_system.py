"""Enhanced reward system for Connect 4 reinforcement learning agents."""

from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Connect4Board


class StrategicRewardCalculator:
    """Calculate rewards based on strategic game moves."""
    
    def __init__(self, reward_config=None):
        """Initialize reward calculator with configurable reward values.

        Args:
            reward_config: Dictionary with reward values, if None uses defaults
        """
        if reward_config is None:
            # Base rewards (normalized scale)
            self.win_reward = 1.0
            self.loss_reward = -1.0
            self.draw_reward = 0.1

            # Strategic rewards (AMPLIFIED to be visible vs terminal rewards)
            self.blocked_opponent_reward = 2.0   # AMPLIFIED: was 0.1, now visible vs 10.0 win
            self.missed_block_penalty = -3.0     # AMPLIFIED: strong penalty for missing blocks
            self.missed_win_penalty = -5.0       # AMPLIFIED: very strong penalty for missing wins
            self.played_winning_move_reward = 3.0 # AMPLIFIED: strong reward for winning moves
            self.center_preference_reward = 0.0  # Disabled - commented out below

            # NEW: Tactical 3-in-a-row rewards
            self.create_threat_reward = 1.5      # Reward for creating 3-in-a-row threat
            self.block_threat_reward = 1.5       # Reward for blocking opponent's 3-in-a-row threat
        else:
            # Use provided configuration (CRITICAL for training with amplified rewards)
            self.win_reward = reward_config.get("win_reward", 10.0)
            self.loss_reward = reward_config.get("loss_reward", -10.0)
            self.draw_reward = reward_config.get("draw_reward", 0.0)
            self.blocked_opponent_reward = reward_config.get("blocked_opponent_reward", 2.0)
            self.missed_block_penalty = reward_config.get("missed_block_penalty", -3.0)
            self.missed_win_penalty = reward_config.get("missed_win_penalty", -5.0)
            self.played_winning_move_reward = reward_config.get("played_winning_move_reward", 3.0)
            self.center_preference_reward = reward_config.get("center_preference_reward", 0.0)

            # NEW: Tactical 3-in-a-row rewards
            self.create_threat_reward = reward_config.get("create_threat_reward", 1.5)
            self.block_threat_reward = reward_config.get("block_threat_reward", 1.5)
        
    def calculate_reward(self, prev_board: Connect4Board, action: int, 
                        new_board: Connect4Board, agent_id: int, 
                        game_over: bool) -> float:
        """Calculate reward for agent's action.
        
        Args:
            prev_board: Board state before action
            action: Column where agent played
            new_board: Board state after action
            agent_id: Agent's player ID (1 or 2)
            game_over: Whether the game ended
            
        Returns:
            Total reward for the action
        """
        total_reward = 0.0
        
        # Base game outcome rewards (only at game end)
        if game_over:
            winner = new_board.check_winner()
            if winner == agent_id:
                total_reward += self.win_reward
            elif winner is not None:
                total_reward += self.loss_reward
            else:
                total_reward += self.draw_reward
        
        # Strategic move rewards (given immediately)
        total_reward += self._evaluate_strategic_move(
            prev_board, action, new_board, agent_id
        )
        
        return total_reward
    
    def _evaluate_strategic_move(self, prev_board: Connect4Board, action: int,
                               new_board: Connect4Board, agent_id: int) -> float:
        """Evaluate strategic value of the move.

        Args:
            prev_board: Board state before action
            action: Column where agent played
            new_board: Board state after action
            agent_id: Agent's player ID

        Returns:
            Strategic reward component
        """
        strategic_reward = 0.0

        # Check if agent played a winning move
        agent_winning_moves = prev_board.find_winning_moves(agent_id)
        if action in agent_winning_moves:
            strategic_reward += self.played_winning_move_reward

        # Check if agent missed a winning move
        if agent_winning_moves and action not in agent_winning_moves:
            strategic_reward += self.missed_win_penalty

        # Check if agent blocked opponent's winning move
        opponent_winning_moves = prev_board.find_winning_moves(3 - agent_id)
        if opponent_winning_moves:
            if action in opponent_winning_moves:
                strategic_reward += self.blocked_opponent_reward
            else:
                strategic_reward += self.missed_block_penalty

        # NEW: Tactical 3-in-a-row evaluation
        tactical_info = prev_board.evaluate_move_tactical(action, agent_id)

        # Reward creating threats (3-in-a-row patterns)
        if tactical_info['creates_threat'] > 0:
            strategic_reward += self.create_threat_reward * tactical_info['creates_threat']

        # Reward blocking opponent threats
        if tactical_info['blocks_threat'] > 0:
            strategic_reward += self.block_threat_reward * tactical_info['blocks_threat']

        # Center preference disabled (set to 0.0)
        # if action in [2, 3, 4]:
        #     strategic_reward += self.center_preference_reward

        return strategic_reward


def calculate_enhanced_reward(prev_board: Connect4Board, action: int,
                             new_board: Connect4Board, agent_id: int,
                             game_over: bool, reward_config=None) -> float:
    """Calculate enhanced reward using strategic evaluation.
    
    This is the main function to use in training scripts.
    
    Args:
        prev_board: Board state before action
        action: Column where agent played  
        new_board: Board state after action
        agent_id: Agent's player ID (1 or 2)
        game_over: Whether the game ended
        reward_config: Optional reward configuration dictionary
        
    Returns:
        Enhanced reward value
    """
    calculator = StrategicRewardCalculator(reward_config)
    return calculator.calculate_reward(prev_board, action, new_board, agent_id, game_over)


if __name__ == "__main__":
    # Test the reward system
    print("Testing Enhanced Reward System")
    print("=" * 32)
    
    # Create test scenario
    board = Connect4Board()
    
    # Set up a situation where player 2 is about to win
    board.board[5, 3] = 2  # Player 2
    board.board[4, 3] = 2  # Player 2
    board.board[3, 3] = 2  # Player 2
    # Player 2 can win by playing column 3
    
    print("Test board:")
    print(board.render())
    print()
    
    prev_board = board.copy()
    
    # Test 1: Agent blocks opponent's win
    print("Test 1: Agent (Player 1) blocks opponent's winning move")
    new_board = board.copy()
    new_board.make_move(3, 1)  # Agent blocks
    reward = calculate_enhanced_reward(prev_board, 3, new_board, 1, False)
    print(f"Reward for blocking: {reward:.3f}")
    print()
    
    # Test 2: Agent misses block
    print("Test 2: Agent (Player 1) misses blocking opportunity")
    new_board2 = board.copy()
    new_board2.make_move(0, 1)  # Agent plays elsewhere
    reward2 = calculate_enhanced_reward(prev_board, 0, new_board2, 1, False)
    print(f"Reward for missing block: {reward2:.3f}")
    print()
    
    # Test 3: Agent has winning move
    board3 = Connect4Board()
    board3.board[5, 2] = 1  # Agent
    board3.board[4, 2] = 1  # Agent  
    board3.board[3, 2] = 1  # Agent
    # Agent can win by playing column 2
    
    prev_board3 = board3.copy()
    
    print("Test 3: Agent (Player 1) plays winning move")
    new_board3 = board3.copy()
    new_board3.make_move(2, 1)  # Winning move
    reward3 = calculate_enhanced_reward(prev_board3, 2, new_board3, 1, True)
    print(f"Reward for winning move: {reward3:.3f}")
    print()
    
    print("Test 4: Agent (Player 1) misses winning move")
    new_board4 = board3.copy()  
    new_board4.make_move(0, 1)  # Non-winning move
    reward4 = calculate_enhanced_reward(prev_board3, 0, new_board4, 1, False)
    print(f"Reward for missing win: {reward4:.3f}")
    
    print("\n✓ Enhanced reward system tested successfully!")