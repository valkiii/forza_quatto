"""Evaluate agent performance through multiple games."""

from game.board import Connect4Board
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent


def play_game(player1, player2, verbose=False):
    """Play a single game between two agents.
    
    Args:
        player1: First player agent
        player2: Second player agent
        verbose: Whether to print game moves
        
    Returns:
        Winner ID (1, 2) or None for draw
    """
    board = Connect4Board()
    move_count = 0
    max_moves = 42
    
    if verbose:
        print(f"{player1} vs {player2}")
        print(board.render())
        print()
    
    while not board.is_terminal() and move_count < max_moves:
        current_player = player1 if board.current_player == 1 else player2
        legal_moves = board.get_legal_moves()
        action = current_player.choose_action(board.get_state(), legal_moves)
        
        board.make_move(action)
        move_count += 1
        
        if verbose:
            print(f"Move {move_count}: {current_player.name} plays column {action}")
            print(board.render())
            print()
    
    winner = board.check_winner()
    if verbose:
        if winner:
            print(f"Winner: Player {winner} ({player1.name if winner == 1 else player2.name})")
        else:
            print("Draw!")
        print()
    
    return winner


def evaluate_matchup(agent1, agent2, num_games=100):
    """Evaluate two agents against each other.
    
    Args:
        agent1: First agent
        agent2: Second agent  
        num_games: Number of games to play
        
    Returns:
        Dict with win statistics
    """
    wins_1 = 0
    wins_2 = 0
    draws = 0
    
    for game in range(num_games):
        # Alternate who goes first
        if game % 2 == 0:
            winner = play_game(agent1, agent2)
            if winner == 1:
                wins_1 += 1
            elif winner == 2:
                wins_2 += 1
            else:
                draws += 1
        else:
            winner = play_game(agent2, agent1)
            if winner == 1:
                wins_2 += 1
            elif winner == 2:
                wins_1 += 1
            else:
                draws += 1
    
    return {
        'agent1_wins': wins_1,
        'agent2_wins': wins_2,
        'draws': draws,
        'agent1_winrate': wins_1 / num_games,
        'agent2_winrate': wins_2 / num_games,
        'draw_rate': draws / num_games
    }


def main():
    """Run agent evaluation."""
    print("Agent Performance Evaluation")
    print("=" * 40)
    
    # Create agents
    random1 = RandomAgent(player_id=1, seed=42)
    random2 = RandomAgent(player_id=2, seed=123)
    heuristic = HeuristicAgent(player_id=1, seed=42)
    
    print("1. Random vs Random (baseline)")
    results = evaluate_matchup(random1, random2, num_games=100)
    print(f"Random1 wins: {results['agent1_wins']}/100 ({results['agent1_winrate']:.1%})")
    print(f"Random2 wins: {results['agent2_wins']}/100 ({results['agent2_winrate']:.1%})")
    print(f"Draws: {results['draws']}/100 ({results['draw_rate']:.1%})")
    print()
    
    print("2. Heuristic vs Random")
    results = evaluate_matchup(heuristic, random2, num_games=100)
    print(f"Heuristic wins: {results['agent1_wins']}/100 ({results['agent1_winrate']:.1%})")
    print(f"Random wins: {results['agent2_wins']}/100 ({results['agent2_winrate']:.1%})")
    print(f"Draws: {results['draws']}/100 ({results['draw_rate']:.1%})")
    print()
    
    print("Sample game: Heuristic vs Random")
    print("-" * 30)
    heuristic_demo = HeuristicAgent(player_id=1, seed=999)
    random_demo = RandomAgent(player_id=2, seed=999)
    play_game(heuristic_demo, random_demo, verbose=True)


if __name__ == "__main__":
    main()