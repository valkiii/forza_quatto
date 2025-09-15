#!/usr/bin/env python3
"""Diagnostic script to check Q-value learning and discrimination."""

import os
import sys
import numpy as np
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.double_dqn_agent import DoubleDQNAgent
from game.board import Connect4Board


def create_diagnostic_states():
    """Create test states with clear strategic differences."""
    states = []
    
    # Empty board
    board = Connect4Board()
    states.append(("Empty Board", board.get_state().copy()))
    
    # About to win horizontally
    board = Connect4Board()
    board.make_move(0, 1)  # X
    board.make_move(1, 1)  # X  
    board.make_move(2, 1)  # X
    # Column 3 should have very high Q-value for winning
    states.append(("About to Win H", board.get_state().copy()))
    
    # Need to block opponent win
    board = Connect4Board()
    board.make_move(0, 2)  # O
    board.make_move(1, 2)  # O
    board.make_move(2, 2)  # O
    # Column 3 should have high Q-value for blocking
    states.append(("Must Block", board.get_state().copy()))
    
    # Random position
    board = Connect4Board()
    board.make_move(3, 1)  # Center
    board.make_move(4, 2)  # Opponent response
    states.append(("Mid Game", board.get_state().copy()))
    
    return states


def diagnose_agent_learning(model_path: str):
    """Diagnose Q-value learning for a trained agent."""
    print(f"🔍 Diagnosing Q-value learning for: {os.path.basename(model_path)}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load agent
    try:
        agent = DoubleDQNAgent(
            player_id=1,
            state_size=84,
            action_size=7,
            hidden_size=256,  # Match training architecture
            seed=42
        )
        agent.load(model_path, keep_player_id=False)
        agent.epsilon = 0.0  # No exploration
        print(f"✅ Agent loaded successfully")
        print(f"   Device: {agent.device}")
        print(f"   Network architecture: {agent.online_net}")
    except Exception as e:
        print(f"❌ Failed to load agent: {e}")
        return
    
    # Test states
    test_states = create_diagnostic_states()
    
    print(f"\n📊 Q-VALUE ANALYSIS")
    print("-" * 40)
    
    for name, board_state in test_states:
        print(f"\n🎯 {name}:")
        
        # Show board
        print("Board state:")
        for row in range(6):
            line = ""
            for col in range(7):
                if board_state[row, col] == 0:
                    line += ". "
                elif board_state[row, col] == 1:
                    line += "X "
                else:
                    line += "O "
            print(f"  {line}")
        
        # Get Q-values
        try:
            encoded_state = agent.encode_state(board_state)
            print(f"  State encoding shape: {encoded_state.shape}")
            print(f"  State encoding range: [{encoded_state.min():.3f}, {encoded_state.max():.3f}]")
            print(f"  State encoding unique values: {len(np.unique(encoded_state))}")
            
            state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(agent.device)
            
            with torch.no_grad():
                q_values = agent.online_net(state_tensor).cpu().numpy()[0]
            
            print(f"  Raw Q-values: {q_values}")
            print(f"  Q-value range: [{q_values.min():.6f}, {q_values.max():.6f}]")
            print(f"  Q-value std: {q_values.std():.6f}")
            print(f"  Best action: Column {q_values.argmax()}")
            
            # Check if all Q-values are identical
            if np.allclose(q_values, q_values[0], atol=1e-6):
                print(f"  ⚠️  ALL Q-VALUES ARE IDENTICAL! ({q_values[0]:.6f})")
            else:
                print(f"  ✅ Q-values show variation")
                
        except Exception as e:
            print(f"  ❌ Error getting Q-values: {e}")
    
    # Check network weights
    print(f"\n🧠 NETWORK ANALYSIS")
    print("-" * 40)
    
    total_params = sum(p.numel() for p in agent.online_net.parameters())
    trainable_params = sum(p.numel() for p in agent.online_net.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check if weights are reasonable
    weight_stats = []
    for name, param in agent.online_net.named_parameters():
        if param.requires_grad:
            weights = param.data.cpu().numpy()
            weight_stats.append({
                'name': name,
                'shape': weights.shape,
                'mean': weights.mean(),
                'std': weights.std(),
                'min': weights.min(),
                'max': weights.max()
            })
    
    print(f"\nWeight statistics:")
    for stats in weight_stats:
        print(f"  {stats['name']}: shape={stats['shape']}")
        print(f"    mean={stats['mean']:.6f}, std={stats['std']:.6f}")
        print(f"    range=[{stats['min']:.6f}, {stats['max']:.6f}]")
        
        # Check for potential issues
        if abs(stats['mean']) < 1e-6 and stats['std'] < 1e-6:
            print(f"    ⚠️  Weights appear uninitialized or dead!")
        elif stats['std'] < 1e-4:
            print(f"    ⚠️  Very small weight variation - possible dead neurons")
        else:
            print(f"    ✅ Weights look reasonable")
    
    # Test gradient flow
    print(f"\n🔄 GRADIENT FLOW TEST")
    print("-" * 40)
    
    try:
        agent.online_net.train()
        
        # Create dummy training example
        state = torch.FloatTensor(agent.encode_state(test_states[1][1])).unsqueeze(0).to(agent.device)
        target = torch.FloatTensor([1.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0]).to(agent.device)  # High target for column 3
        
        # Forward pass
        q_values = agent.online_net(state)
        loss = torch.nn.MSELoss()(q_values, target.unsqueeze(0))
        
        # Backward pass
        agent.optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        total_norm = 0
        param_count = 0
        for param in agent.online_net.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        print(f"Loss: {loss.item():.6f}")
        print(f"Gradient norm: {total_norm:.6f}")
        print(f"Parameters with gradients: {param_count}")
        
        if total_norm < 1e-7:
            print(f"⚠️  Vanishing gradients detected!")
        elif total_norm > 100:
            print(f"⚠️  Exploding gradients detected!")
        else:
            print(f"✅ Gradients look reasonable")
            
        agent.online_net.eval()
        
    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
    
    # Training statistics
    if hasattr(agent, 'train_step_count'):
        print(f"\n📈 TRAINING STATISTICS")
        print("-" * 40)
        print(f"Training steps: {agent.train_step_count:,}")
        print(f"Current epsilon: {agent.epsilon:.6f}")
        print(f"Buffer size: {len(agent.memory):,}")
        
        if agent.train_step_count == 0:
            print(f"⚠️  Agent has never been trained!")
        elif agent.train_step_count < 1000:
            print(f"⚠️  Very little training - may not have learned yet")


def main():
    """Main diagnostic function."""
    print("🔍 Connect 4 Q-Value Learning Diagnostic")
    print("=" * 50)
    
    # Test available models
    model_candidates = [
        "models/double_dqn_post_heuristic.pt",
        "models_advanced/double_dqn_post_heuristic_advanced.pt",
        "models/double_dqn_final.pt",
        "models_advanced/double_dqn_final_advanced.pt",
        "models/double_dqn_ep_60000.pt",
        "models/double_dqn_ep_100000.pt"
    ]
    
    found_models = [path for path in model_candidates if os.path.exists(path)]
    
    if not found_models:
        print("❌ No trained models found!")
        print("Available directories:")
        for dir_name in ["models", "models_advanced", "models_improved"]:
            if os.path.exists(dir_name):
                files = [f for f in os.listdir(dir_name) if f.endswith('.pt')]
                print(f"  {dir_name}/: {len(files)} .pt files")
                if files:
                    print(f"    Examples: {', '.join(files[:3])}")
        return
    
    print(f"Found {len(found_models)} trained models")
    
    # Test each model
    for model_path in found_models:
        try:
            diagnose_agent_learning(model_path)
            print(f"\n" + "="*60)
        except Exception as e:
            print(f"❌ Failed to diagnose {model_path}: {e}")
            print(f"\n" + "="*60)


if __name__ == "__main__":
    main()