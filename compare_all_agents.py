"""Compare all agents: Random, Heuristic, Q-Learning, and DQN."""

import sys
sys.path.append('/Users/vc/Research/forza_quattro')

from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from train.q_learning_train import evaluate_agent


def load_trained_agents():
    """Load the trained RL agents."""
    # Load Q-learning agent
    q_agent = QLearningAgent(player_id=1, seed=42)
    try:
        q_agent.load("models/q_learning_final.pkl")
        q_agent.epsilon = 0.0  # No exploration for evaluation
        print(f"✅ Loaded Q-learning agent (Q-table: {len(q_agent.q_table):,} entries)")
    except FileNotFoundError:
        print("❌ Q-learning model not found. Train first with: python train/quick_q_train.py")
        q_agent = None
    
    # Load DQN agent
    dqn_agent = DQNAgent(player_id=1, seed=42)
    try:
        dqn_agent.load("models/dqn_final.pt")
        dqn_agent.epsilon = 0.0  # No exploration for evaluation
        stats = dqn_agent.get_stats()
        print(f"✅ Loaded DQN agent (Training steps: {stats['training_steps']:,})")
    except FileNotFoundError:
        print("❌ DQN model not found. Train first with: python train/quick_dqn_train.py")
        dqn_agent = None
    
    return q_agent, dqn_agent


def main():
    """Compare all agents in a comprehensive evaluation."""
    print("🎮 Connect 4 Agent Comparison")
    print("=" * 50)
    
    # Load trained agents
    q_agent, dqn_agent = load_trained_agents()
    
    # Initialize baseline agents
    random_agent = RandomAgent(player_id=2, seed=123)
    heuristic_agent = HeuristicAgent(player_id=2, seed=123)
    
    agents = [
        ("Random", RandomAgent(player_id=1, seed=42)),
        ("Heuristic", HeuristicAgent(player_id=1, seed=42)),
    ]
    
    if q_agent:
        agents.append(("Q-Learning", q_agent))
    if dqn_agent:
        agents.append(("DQN", dqn_agent))
    
    opponents = [
        ("Random", random_agent),
        ("Heuristic", heuristic_agent)
    ]
    
    print()
    print("🏆 WIN RATES (500 games each matchup)")
    print("-" * 50)
    print(f"{'Agent':<12} {'vs Random':<10} {'vs Heuristic':<12}")
    print("-" * 50)
    
    results = {}
    
    for agent_name, agent in agents:
        results[agent_name] = {}
        
        for opp_name, opponent in opponents:
            win_rate = evaluate_agent(agent, opponent, num_games=500)
            results[agent_name][opp_name] = win_rate
            
        print(f"{agent_name:<12} {results[agent_name]['Random']:<9.1%} "
              f"{results[agent_name]['Heuristic']:<11.1%}")
    
    print("-" * 50)
    print()
    
    # Analysis
    print("📊 ANALYSIS")
    print("-" * 20)
    
    # Improvement over random baseline
    random_vs_random = results["Random"]["Random"]
    print(f"Random vs Random baseline: {random_vs_random:.1%}")
    print()
    
    for agent_name in ["Heuristic", "Q-Learning", "DQN"]:
        if agent_name in results:
            improvement = results[agent_name]["Random"] - random_vs_random
            print(f"{agent_name} improvement over random: {improvement:+.1%} points")
    
    print()
    
    # Strategic play assessment
    print("🎯 STRATEGIC PLAY ASSESSMENT")
    print("-" * 30)
    
    for agent_name in ["Q-Learning", "DQN"]:
        if agent_name in results:
            vs_heuristic = results[agent_name]["Heuristic"]
            if vs_heuristic > 0.30:
                print(f"✅ {agent_name}: Can compete strategically ({vs_heuristic:.1%})")
            elif vs_heuristic > 0.10:
                print(f"⚠️  {agent_name}: Shows some strategy ({vs_heuristic:.1%})")
            else:
                print(f"❌ {agent_name}: Needs more strategic training ({vs_heuristic:.1%})")
    
    print()
    
    # Method comparison
    if "Q-Learning" in results and "DQN" in results:
        print("🤖 TABULAR vs NEURAL COMPARISON")
        print("-" * 32)
        q_vs_random = results["Q-Learning"]["Random"]
        dqn_vs_random = results["DQN"]["Random"]
        q_vs_heuristic = results["Q-Learning"]["Heuristic"]
        dqn_vs_heuristic = results["DQN"]["Heuristic"]
        
        print(f"vs Random   - Q-Learning: {q_vs_random:.1%}, DQN: {dqn_vs_random:.1%}")
        print(f"vs Heuristic - Q-Learning: {q_vs_heuristic:.1%}, DQN: {dqn_vs_heuristic:.1%}")
        
        if dqn_vs_random > q_vs_random:
            diff = dqn_vs_random - q_vs_random
            print(f"🏅 DQN outperforms Q-Learning by {diff:+.1%} points vs Random")
        else:
            diff = q_vs_random - dqn_vs_random
            print(f"🏅 Q-Learning outperforms DQN by {diff:+.1%} points vs Random")


if __name__ == "__main__":
    main()