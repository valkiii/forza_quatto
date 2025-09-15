"""Visual demonstration of board representation in DQN agent."""

import sys
sys.path.append('/Users/vc/Research/forza_quattro')

import numpy as np
import torch
from game.board import Connect4Board
from agents.dqn_agent import DQNAgent


def visualize_board_transformation():
    """Show step-by-step board representation transformation."""
    print("🎯 BOARD REPRESENTATION DEMO")
    print("=" * 50)
    
    # Create a game situation
    board = Connect4Board()
    
    # Set up an interesting board position
    moves = [
        (3, 1), (3, 2), (1, 1), (4, 2),  # Some initial moves
        (3, 1), (2, 2), (0, 1), (5, 2)   # More moves
    ]
    
    print("Setting up game position...")
    for col, player in moves:
        board.make_move(col, player)
        
    print("\n1️⃣ VISUAL BOARD STATE")
    print("-" * 25)
    print(board.render())
    
    print("\n2️⃣ RAW BOARD ARRAY (from game engine)")
    print("-" * 40)
    raw_board = board.get_state()
    print("Raw board shape:", raw_board.shape)
    print("Raw board array:")
    for i, row in enumerate(raw_board):
        print(f"Row {i}: {row}")
    
    print("\nValue meanings:")
    print("  0 = Empty cell")
    print("  1 = Player 1 (X)")
    print("  2 = Player 2 (O)")
    
    # Create DQN agents for both players
    dqn_player1 = DQNAgent(player_id=1, seed=42)
    dqn_player2 = DQNAgent(player_id=2, seed=42)
    
    print("\n3️⃣ RELATIVE ENCODING (Agent's Perspective)")
    print("-" * 45)
    
    # Show encoding from Player 1's perspective
    encoded_p1 = dqn_player1.encode_state(raw_board)
    relative_2d_p1 = encoded_p1.reshape(6, 7)
    
    print("From Player 1's perspective (DQN as Player 1):")
    print("Relative board shape:", relative_2d_p1.shape)
    for i, row in enumerate(relative_2d_p1):
        print(f"Row {i}: {row}")
    
    print("\nEncoding meaning (Player 1's view):")
    print("  0.0 = Empty cell")
    print("  1.0 = My pieces (Player 1)")
    print("  2.0 = Opponent pieces (Player 2)")
    
    # Show encoding from Player 2's perspective
    encoded_p2 = dqn_player2.encode_state(raw_board)
    relative_2d_p2 = encoded_p2.reshape(6, 7)
    
    print("\nFrom Player 2's perspective (DQN as Player 2):")
    for i, row in enumerate(relative_2d_p2):
        print(f"Row {i}: {row}")
        
    print("\nEncoding meaning (Player 2's view):")
    print("  0.0 = Empty cell") 
    print("  1.0 = My pieces (Player 2)")
    print("  2.0 = Opponent pieces (Player 1)")
    
    print("\n4️⃣ FLATTENED VECTOR (Neural Network Input)")
    print("-" * 48)
    print("Player 1's flattened input (42 elements):")
    print("Shape:", encoded_p1.shape)
    print("First 14 elements (rows 0-1):", encoded_p1[:14])
    print("Next 14 elements (rows 2-3): ", encoded_p1[14:28])
    print("Last 14 elements (rows 4-5): ", encoded_p1[28:42])
    
    print("\nIndex mapping example:")
    print("  encoded_p1[0] = board[0,0] =", encoded_p1[0])
    print("  encoded_p1[7] = board[1,0] =", encoded_p1[7])
    print("  encoded_p1[35] = board[5,0] =", encoded_p1[35])
    
    print("\n5️⃣ NEURAL NETWORK FORWARD PASS")
    print("-" * 37)
    
    # Get Q-values from both agents
    legal_moves = board.get_legal_moves()
    print("Legal moves:", legal_moves)
    
    # Player 1's Q-values
    state_tensor_p1 = torch.FloatTensor(encoded_p1).unsqueeze(0).to(dqn_player1.device)
    with torch.no_grad():
        q_values_p1 = dqn_player1.q_network(state_tensor_p1)
        
    print("\nPlayer 1's Q-values (before masking):")
    q_vals_p1 = q_values_p1.cpu().numpy()[0]
    for i, q_val in enumerate(q_vals_p1):
        print(f"  Column {i}: {q_val:.4f}")
    
    # Apply legal move masking for Player 1
    masked_q_p1 = q_vals_p1.copy()
    for col in range(7):
        if col not in legal_moves:
            masked_q_p1[col] = float('-inf')
            
    print("\nPlayer 1's Q-values (after legal move masking):")
    for i, q_val in enumerate(masked_q_p1):
        if q_val == float('-inf'):
            print(f"  Column {i}: -∞ (illegal)")
        else:
            print(f"  Column {i}: {q_val:.4f}")
            
    best_action_p1 = np.argmax(masked_q_p1)
    print(f"\nPlayer 1 would choose: Column {best_action_p1}")
    
    print("\n6️⃣ PERSPECTIVE COMPARISON")
    print("-" * 28)
    
    print("Key insight: Same board, different perspectives!")
    print("\nDifferences in encoding:")
    diff_positions = np.where(relative_2d_p1 != relative_2d_p2)
    if len(diff_positions[0]) > 0:
        for i in range(len(diff_positions[0])):
            row, col = diff_positions[0][i], diff_positions[1][i]
            print(f"  Position ({row},{col}): P1 sees {relative_2d_p1[row,col]}, P2 sees {relative_2d_p2[row,col]}")
    else:
        print("  No differences (empty board or symmetric position)")
        
    print("\n7️⃣ BATCH PROCESSING EXAMPLE")
    print("-" * 32)
    
    # Create batch of states
    batch_states = []
    batch_boards = []
    
    # Generate a few different board states
    for i in range(3):
        temp_board = Connect4Board()
        # Make some random moves
        for _ in range(i * 2 + 1):
            legal = temp_board.get_legal_moves()
            if legal:
                col = np.random.choice(legal)
                temp_board.make_move(col)
        batch_boards.append(temp_board)
        batch_states.append(dqn_player1.encode_state(temp_board.get_state()))
    
    # Process batch
    batch_tensor = torch.FloatTensor(batch_states).to(dqn_player1.device)
    print("Batch tensor shape:", batch_tensor.shape)
    
    with torch.no_grad():
        batch_q_values = dqn_player1.q_network(batch_tensor)
        
    print("Batch Q-values shape:", batch_q_values.shape)
    print("Q-values for each board in batch:")
    for i, q_vals in enumerate(batch_q_values.cpu().numpy()):
        print(f"  Board {i}: {q_vals}")
        
    print("\n🎯 SUMMARY")
    print("-" * 10)
    print("✓ Raw board: 6×7 integer array (0, 1, 2)")
    print("✓ Relative encoding: Agent perspective (0.0, 1.0, 2.0)")
    print("✓ Flattened: 42-element vector for neural network")
    print("✓ Q-values: 7 outputs (one per column)")
    print("✓ Legal masking: Prevents illegal moves")
    print("✓ Batch processing: Multiple boards simultaneously")


if __name__ == "__main__":
    visualize_board_transformation()