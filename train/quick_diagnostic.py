"""Quick diagnostic to compare agent performance."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Connect4Board
from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.dqn_agent import DQNAgent
import numpy as np

def test_agent_performance(agent, opponent, num_games=100):
    """Test agent performance."""
    old_epsilon = getattr(agent, 'epsilon', None)
    if old_epsilon is not None:
        agent.epsilon = 0.0  # No exploration
    
    wins = 0
    game_lengths = []
    
    for _ in range(num_games):
        board = Connect4Board()
        
        # Random starting player
        if np.random.random() < 0.5:
            players = [agent, opponent]
        else:
            players = [opponent, agent]
        
        move_count = 0
        while not board.is_terminal() and move_count < 42:
            current_player = players[move_count % 2]
            legal_moves = board.get_legal_moves()
            action = current_player.choose_action(board.get_state(), legal_moves)
            board.make_move(action, current_player.player_id)
            move_count += 1
        
        winner = board.check_winner()
        if winner == agent.player_id:
            wins += 1
        game_lengths.append(move_count)
    
    if old_epsilon is not None:
        agent.epsilon = old_epsilon
    
    return wins / num_games, np.mean(game_lengths)

def main():
    print("🔍 Agent Performance Diagnostic")
    print("=" * 35)
    
    # Test random agent baseline
    random_agent = RandomAgent(player_id=1, seed=42)
    random_vs_random, _ = test_agent_performance(random_agent, RandomAgent(player_id=2, seed=43), 500)
    print(f"Random vs Random: {random_vs_random:.1%} (should be ~50%)")
    
    random_vs_heuristic, _ = test_agent_performance(random_agent, HeuristicAgent(player_id=2, seed=43), 500)  
    print(f"Random vs Heuristic: {random_vs_heuristic:.1%} (should be <30%)")
    
    # Test fresh DQN agent (untrained)
    print(f"\nTesting untrained DQN agent:")
    untrained_dqn = DQNAgent(player_id=1, epsilon_start=0.0, seed=42)  # No exploration for testing
    
    untrained_vs_random, _ = test_agent_performance(untrained_dqn, RandomAgent(player_id=2, seed=43), 100)
    print(f"Untrained DQN vs Random: {untrained_vs_random:.1%}")
    
    untrained_vs_heuristic, _ = test_agent_performance(untrained_dqn, HeuristicAgent(player_id=2, seed=43), 100)
    print(f"Untrained DQN vs Heuristic: {untrained_vs_heuristic:.1%}")
    
    # Test with some basic training
    print(f"\nTesting DQN with 100 training games:")
    training_dqn = DQNAgent(player_id=1, seed=42)
    
    # Simple training loop
    for episode in range(100):
        board = Connect4Board()
        opponent = RandomAgent(player_id=2, seed=episode + 100)
        
        # Simple game loop
        move_count = 0
        experiences = []
        
        if np.random.random() < 0.5:
            players = [training_dqn, opponent]
        else:
            players = [opponent, training_dqn]
        
        while not board.is_terminal() and move_count < 42:
            current_player = players[move_count % 2]
            
            if current_player == training_dqn:
                state = board.get_state()
                experiences.append(state.copy())
            
            legal_moves = board.get_legal_moves()
            action = current_player.choose_action(board.get_state(), legal_moves)
            
            if current_player == training_dqn:
                experiences.append(action)
            
            board.make_move(action, current_player.player_id)
            move_count += 1
        
        # Give final reward to DQN
        if len(experiences) >= 2:
            winner = board.check_winner()
            if winner == training_dqn.player_id:
                reward = 10.0
            elif winner is not None:
                reward = -10.0
            else:
                reward = 1.0
            
            # Store one experience (simplified)
            for i in range(0, len(experiences) - 1, 2):
                if i + 1 < len(experiences):
                    state = experiences[i]
                    action = experiences[i + 1]
                    next_state = board.get_state()
                    training_dqn.observe(state, action, reward, next_state, True)
        
        training_dqn.reset_episode()
    
    # Test trained agent
    trained_vs_random, _ = test_agent_performance(training_dqn, RandomAgent(player_id=2, seed=200), 100)
    print(f"DQN after 100 episodes vs Random: {trained_vs_random:.1%}")
    
    trained_vs_heuristic, _ = test_agent_performance(training_dqn, HeuristicAgent(player_id=2, seed=201), 100)
    print(f"DQN after 100 episodes vs Heuristic: {trained_vs_heuristic:.1%}")
    
    # Check training stats
    stats = training_dqn.get_stats()
    print(f"\nTraining Stats after 100 episodes:")
    print(f"  Training steps: {stats.get('training_steps', 'N/A')}")
    print(f"  Buffer size: {stats.get('buffer_size', 'N/A')}")
    print(f"  Epsilon: {training_dqn.epsilon:.4f}")
    
    print(f"\n📊 Analysis:")
    if trained_vs_random > untrained_vs_random + 0.1:
        print("✅ DQN is learning from experience!")
    else:
        print("⚠️  DQN shows minimal learning improvement")
        
    if trained_vs_random > 0.6:
        print("✅ Good performance vs random opponents")
    else:
        print("⚠️  Poor performance vs random opponents")

if __name__ == "__main__":
    main()