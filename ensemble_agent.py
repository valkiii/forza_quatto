#!/usr/bin/env python3
"""Ensemble agent that combines multiple RL models for improved performance."""

import os
import sys
import json
import glob
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.board import Connect4Board
from agents.double_dqn_agent import DoubleDQNAgent
from agents.enhanced_double_dqn_agent import EnhancedDoubleDQNAgent
from agents.cnn_dueling_dqn_agent import CNNDuelingDQNAgent
from train_fixed_double_dqn import FixedDoubleDQNAgent
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent


class EnsembleAgent:
    """Ensemble agent that combines multiple models using different strategies."""
    
    def __init__(self, 
                 model_configs: List[Dict], 
                 ensemble_method: str = "weighted_voting",
                 player_id: int = 1,
                 name: str = "Ensemble",
                 show_contributions: bool = False):
        """
        Initialize ensemble agent.
        
        Args:
            model_configs: List of model configuration dictionaries with keys:
                - 'path': Path to model file (or 'heuristic'/'random' for baselines)
                - 'weight': Weight for this model in ensemble (default: 1.0)
                - 'name': Optional name for the model
            ensemble_method: Method to combine predictions
                - 'weighted_voting': Weighted majority vote on actions
                - 'q_value_averaging': Average Q-values then select best action
                - 'confidence_weighted': Weight by model confidence
            player_id: Player ID for the ensemble agent
            name: Name for the ensemble agent
            show_contributions: Whether to show individual model contributions
        """
        self.player_id = player_id
        self.name = name
        self.ensemble_method = ensemble_method
        self.show_contributions = show_contributions
        self.models = []
        self.weights = []
        self.model_names = []
        self.last_decision_info = None  # Store details of last decision
        
        print(f"🤖 Initializing {name} with {len(model_configs)} models...")
        
        # Load all models
        for i, config in enumerate(model_configs):
            model_path = config.get('path', '')
            weight = config.get('weight', 1.0)
            model_name = config.get('name', f"Model-{i+1}")
            
            try:
                model = self._load_model(model_path, player_id)
                if model:
                    self.models.append(model)
                    self.weights.append(weight)
                    self.model_names.append(model_name)
                    print(f"  ✅ {model_name}: {os.path.basename(model_path) if '/' in model_path else model_path} (weight: {weight})")
                else:
                    print(f"  ❌ Failed to load {model_name}: {model_path}")
            except Exception as e:
                print(f"  ❌ Error loading {model_name}: {e}")
        
        if not self.models:
            raise ValueError("No models successfully loaded for ensemble")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        print(f"✅ Ensemble ready with {len(self.models)} models using {ensemble_method}")
    
    def _load_model(self, model_path: str, player_id: int):
        """Load a single model based on path and auto-detect type."""
        if model_path.lower() == 'heuristic':
            return HeuristicAgent(player_id=player_id, seed=42)
        elif model_path.lower() == 'random':
            return RandomAgent(player_id=player_id, seed=42)
        
        if not os.path.exists(model_path):
            return None
        
        # Auto-detect model type based on path
        model_name = os.path.basename(model_path).lower()
        
        if "m1_cnn" in model_path.lower() or ("cnn" in model_name and "m1" in model_path.lower()):
            # M1-Optimized CNN
            agent = CNNDuelingDQNAgent(
                player_id=player_id,
                input_channels=2,
                action_size=7,
                hidden_size=48,
                architecture="m1_optimized",
                seed=42
            )
            agent.load(model_path, keep_player_id=False)
            agent.epsilon = 0.0  # No exploration in ensemble
            return agent
            
        elif "cnn" in model_name:
            # Ultra-light CNN
            agent = CNNDuelingDQNAgent(
                player_id=player_id,
                input_channels=2,
                action_size=7,
                hidden_size=16,
                architecture="ultra_light",
                seed=42
            )
            agent.load(model_path, keep_player_id=False)
            agent.epsilon = 0.0
            return agent
            
        elif "enhanced" in model_path.lower():
            # Enhanced Double DQN
            agent = EnhancedDoubleDQNAgent(
                player_id=player_id,
                state_size=92,
                action_size=7,
                hidden_size=512,
                seed=42
            )
            agent.load(model_path, keep_player_id=False)
            agent.epsilon = 0.0
            return agent
            
        else:
            # Fixed Double DQN (legacy)
            agent = FixedDoubleDQNAgent(
                player_id=player_id,
                state_size=84,
                action_size=7,
                seed=42,
                gradient_clip_norm=1.0,
                use_huber_loss=True,
                huber_delta=1.0,
                state_normalization=True
            )
            agent.load(model_path, keep_player_id=False)
            agent.epsilon = 0.0
            return agent
    
    def _get_model_q_values(self, model, state, legal_moves):
        """Get Q-values from a model, handling different model types."""
        if hasattr(model, 'get_q_values'):
            # Direct Q-value access
            return model.get_q_values(state)
        elif hasattr(model, 'online_net'):
            # Neural network models
            if hasattr(model, 'encode_state'):
                # CNN models
                encoded_state = model.encode_state(state)
                # Convert to tensor if it's numpy array
                if isinstance(encoded_state, np.ndarray):
                    encoded_state = torch.FloatTensor(encoded_state).to(model.device)
                # Add batch dimension if needed
                if len(encoded_state.shape) == 1:
                    encoded_state = encoded_state.unsqueeze(0)
                elif len(encoded_state.shape) == 3:  # CNN input: [channels, height, width]
                    encoded_state = encoded_state.unsqueeze(0)  # Add batch dimension
                
                with torch.no_grad():
                    q_values = model.online_net(encoded_state).cpu().numpy().flatten()
            else:
                # Standard DQN models
                state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(model.device)
                with torch.no_grad():
                    q_values = model.online_net(state_tensor).cpu().numpy().flatten()
            return q_values
        else:
            # Heuristic/Random agents - simulate Q-values
            q_values = np.full(7, -1000.0)  # Very low default
            for move in legal_moves:
                if hasattr(model, 'evaluate_move'):
                    # Heuristic agent with move evaluation
                    q_values[move] = model.evaluate_move(state, move)
                else:
                    # Random agent
                    q_values[move] = np.random.random()
            return q_values
    
    def choose_action(self, state, legal_moves) -> int:
        """Choose action using ensemble method."""
        if self.ensemble_method == "weighted_voting":
            return self._weighted_voting(state, legal_moves)
        elif self.ensemble_method == "q_value_averaging":
            return self._q_value_averaging(state, legal_moves)
        elif self.ensemble_method == "confidence_weighted":
            return self._confidence_weighted(state, legal_moves)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _weighted_voting(self, state, legal_moves) -> int:
        """Weighted majority vote on actions."""
        action_votes = {}
        model_contributions = []
        
        for model, weight, name in zip(self.models, self.weights, self.model_names):
            try:
                action = model.choose_action(state, legal_moves)
                action_votes[action] = action_votes.get(action, 0) + weight
                model_contributions.append({
                    'name': name,
                    'action': action,
                    'weight': weight,
                    'contribution': weight
                })
            except Exception as e:
                print(f"⚠️ Warning: {name} failed to choose action: {e}")
                model_contributions.append({
                    'name': name,
                    'action': None,
                    'weight': weight,
                    'contribution': 0,
                    'error': str(e)
                })
                continue
        
        if not action_votes:
            chosen_action = np.random.choice(legal_moves)
        else:
            chosen_action = max(action_votes, key=action_votes.get)
        
        # Store decision info
        self.last_decision_info = {
            'method': 'weighted_voting',
            'chosen_action': chosen_action,
            'action_votes': action_votes.copy(),
            'model_contributions': model_contributions,
            'legal_moves': legal_moves.copy()
        }
        
        return chosen_action
    
    def _q_value_averaging(self, state, legal_moves) -> int:
        """Average Q-values across models."""
        avg_q_values = np.zeros(7)
        model_contributions = []
        valid_models = 0
        
        for model, weight, name in zip(self.models, self.weights, self.model_names):
            try:
                q_values = self._get_model_q_values(model, state, legal_moves)
                avg_q_values += weight * q_values
                valid_models += 1
                
                # Find this model's preferred action
                legal_q_values = q_values.copy()
                for col in range(7):
                    if col not in legal_moves:
                        legal_q_values[col] = -float('inf')
                preferred_action = np.argmax(legal_q_values)
                
                model_contributions.append({
                    'name': name,
                    'action': preferred_action,
                    'weight': weight,
                    'q_values': q_values.copy(),
                    'max_q_value': q_values[preferred_action],
                    'contribution': weight * q_values[preferred_action]
                })
            except Exception as e:
                print(f"⚠️ Warning: {name} failed to provide Q-values: {e}")
                model_contributions.append({
                    'name': name,
                    'action': None,
                    'weight': weight,
                    'contribution': 0,
                    'error': str(e)
                })
                continue
        
        if valid_models == 0:
            chosen_action = np.random.choice(legal_moves)
        else:
            # Mask illegal moves
            for col in range(7):
                if col not in legal_moves:
                    avg_q_values[col] = -float('inf')
            chosen_action = np.argmax(avg_q_values)
        
        # Store decision info
        self.last_decision_info = {
            'method': 'q_value_averaging',
            'chosen_action': chosen_action,
            'averaged_q_values': avg_q_values.copy(),
            'model_contributions': model_contributions,
            'legal_moves': legal_moves.copy(),
            'valid_models': valid_models
        }
        
        return chosen_action
    
    def _confidence_weighted(self, state, legal_moves) -> int:
        """Weight by model confidence (Q-value spread)."""
        weighted_q_values = np.zeros(7)
        model_contributions = []
        total_confidence = 0
        
        for model, base_weight, name in zip(self.models, self.weights, self.model_names):
            try:
                q_values = self._get_model_q_values(model, state, legal_moves)
                
                # Calculate confidence as Q-value spread
                legal_q_values = [q_values[move] for move in legal_moves]
                confidence = np.std(legal_q_values) if len(legal_q_values) > 1 else 1.0
                
                # Combine base weight with confidence
                effective_weight = base_weight * (1 + confidence)
                weighted_q_values += effective_weight * q_values
                total_confidence += effective_weight
                
                # Find preferred action
                legal_q_values_array = q_values.copy()
                for col in range(7):
                    if col not in legal_moves:
                        legal_q_values_array[col] = -float('inf')
                preferred_action = np.argmax(legal_q_values_array)
                
                model_contributions.append({
                    'name': name,
                    'action': preferred_action,
                    'base_weight': base_weight,
                    'confidence': confidence,
                    'effective_weight': effective_weight,
                    'q_values': q_values.copy(),
                    'contribution': effective_weight * q_values[preferred_action]
                })
                
            except Exception as e:
                print(f"⚠️ Warning: {name} failed in confidence weighting: {e}")
                model_contributions.append({
                    'name': name,
                    'action': None,
                    'base_weight': base_weight,
                    'contribution': 0,
                    'error': str(e)
                })
                continue
        
        if total_confidence == 0:
            chosen_action = np.random.choice(legal_moves)
        else:
            weighted_q_values /= total_confidence
            
            # Mask illegal moves
            for col in range(7):
                if col not in legal_moves:
                    weighted_q_values[col] = -float('inf')
            
            chosen_action = np.argmax(weighted_q_values)
        
        # Store decision info
        self.last_decision_info = {
            'method': 'confidence_weighted',
            'chosen_action': chosen_action,
            'weighted_q_values': weighted_q_values.copy(),
            'model_contributions': model_contributions,
            'legal_moves': legal_moves.copy(),
            'total_confidence': total_confidence
        }
        
        return chosen_action
    
    def get_model_info(self) -> Dict:
        """Get information about ensemble composition."""
        return {
            'ensemble_method': self.ensemble_method,
            'num_models': len(self.models),
            'models': [
                {
                    'name': name,
                    'weight': weight,
                    'type': type(model).__name__
                }
                for model, weight, name in zip(self.models, self.weights, self.model_names)
            ]
        }
    
    def reset_episode(self):
        """Reset episode for all models that need it."""
        for model in self.models:
            if hasattr(model, 'reset_episode'):
                model.reset_episode()
    
    def observe(self, *args, **kwargs):
        """Observe method (no-op for ensemble)."""
        pass
    
    def get_last_decision_breakdown(self) -> str:
        """Get a formatted breakdown of the last decision made by the ensemble."""
        if not self.last_decision_info:
            return "No decision information available"
        
        info = self.last_decision_info
        breakdown = []
        
        breakdown.append(f"🧠 {self.name} Decision Breakdown:")
        breakdown.append(f"   Method: {info['method']}")
        breakdown.append(f"   Chosen Action: Column {info['chosen_action']}")
        breakdown.append(f"   Legal Moves: {info['legal_moves']}")
        
        if info['method'] == 'weighted_voting':
            breakdown.append(f"\n📊 Vote Distribution:")
            for action, votes in sorted(info['action_votes'].items()):
                breakdown.append(f"   Column {action}: {votes:.3f} votes")
            
            breakdown.append(f"\n🤖 Individual Model Votes:")
            for contrib in info['model_contributions']:
                if 'error' in contrib:
                    breakdown.append(f"   ❌ {contrib['name']}: Failed ({contrib['error']})")
                else:
                    breakdown.append(f"   • {contrib['name']}: Column {contrib['action']} (weight: {contrib['weight']:.3f})")
        
        elif info['method'] == 'q_value_averaging':
            breakdown.append(f"\n📊 Averaged Q-Values:")
            for col in info['legal_moves']:
                breakdown.append(f"   Column {col}: {info['averaged_q_values'][col]:.3f}")
            
            breakdown.append(f"\n🤖 Individual Model Preferences:")
            for contrib in info['model_contributions']:
                if 'error' in contrib:
                    breakdown.append(f"   ❌ {contrib['name']}: Failed ({contrib['error']})")
                else:
                    breakdown.append(f"   • {contrib['name']}: Column {contrib['action']} "
                                   f"(Q: {contrib['max_q_value']:.3f}, weight: {contrib['weight']:.3f})")
        
        elif info['method'] == 'confidence_weighted':
            breakdown.append(f"\n📊 Confidence-Weighted Q-Values:")
            for col in info['legal_moves']:
                breakdown.append(f"   Column {col}: {info['weighted_q_values'][col]:.3f}")
            
            breakdown.append(f"\n🤖 Model Contributions with Confidence:")
            for contrib in info['model_contributions']:
                if 'error' in contrib:
                    breakdown.append(f"   ❌ {contrib['name']}: Failed ({contrib['error']})")
                else:
                    breakdown.append(f"   • {contrib['name']}: Column {contrib['action']} "
                                   f"(base: {contrib['base_weight']:.3f}, conf: {contrib['confidence']:.3f}, "
                                   f"effective: {contrib['effective_weight']:.3f})")
        
        return '\n'.join(breakdown)


def create_preset_ensemble(preset: str, player_id: int = 1) -> EnsembleAgent:
    """Create preset ensemble configurations."""
    
    if preset == "top_performers":
        """Top 4 performing models from tournament"""
        model_configs = [
            {'path': 'models_m1_cnn/m1_cnn_dqn_ep_550000.pt', 'weight': 0.4, 'name': 'M1-CNN-550k'},
            {'path': 'models_m1_cnn/m1_cnn_dqn_ep_500000.pt', 'weight': 0.3, 'name': 'M1-CNN-500k'},
            {'path': 'models_m1_cnn/m1_cnn_dqn_ep_450000.pt', 'weight': 0.2, 'name': 'M1-CNN-450k'},
            {'path': 'models_m1_cnn/m1_cnn_dqn_ep_400000.pt', 'weight': 0.1, 'name': 'M1-CNN-400k'},
        ]
        return EnsembleAgent(model_configs, "q_value_averaging", player_id, "Top-Performers", show_contributions=True)
    
    elif preset == "diverse_architectures":
        """Different architectures for diverse perspectives"""
        model_configs = [
            {'path': 'models_m1_cnn/m1_cnn_dqn_ep_550000.pt', 'weight': 0.4, 'name': 'M1-CNN-550k'},
            {'path': 'models_enhanced/enhanced_double_dqn_final.pt', 'weight': 0.3, 'name': 'Enhanced-DQN'},
            {'path': 'models_cnn/cnn_dqn_final.pt', 'weight': 0.2, 'name': 'Ultra-CNN'},
            {'path': 'heuristic', 'weight': 0.1, 'name': 'Heuristic'},
        ]
        return EnsembleAgent(model_configs, "weighted_voting", player_id, "Diverse-Ensemble", show_contributions=True)
    
    elif preset == "evolution_stages":
        """Models from different training stages"""
        model_configs = [
            {'path': 'models_m1_cnn/m1_cnn_dqn_ep_550000.pt', 'weight': 0.3, 'name': 'M1-CNN-550k'},
            {'path': 'models_m1_cnn/m1_cnn_dqn_ep_300000.pt', 'weight': 0.25, 'name': 'M1-CNN-300k'},
            {'path': 'models_m1_cnn/m1_cnn_dqn_ep_150000.pt', 'weight': 0.2, 'name': 'M1-CNN-150k'},
            {'path': 'models_m1_cnn/m1_cnn_dqn_ep_50000.pt', 'weight': 0.15, 'name': 'M1-CNN-50k'},
            {'path': 'models_m1_cnn/m1_cnn_dqn_ep_10000.pt', 'weight': 0.1, 'name': 'M1-CNN-10k'},
        ]
        return EnsembleAgent(model_configs, "confidence_weighted", player_id, "Evolution-Stages", show_contributions=True)
    
    else:
        raise ValueError(f"Unknown preset: {preset}. Available: top_performers, diverse_architectures, evolution_stages")


def find_available_models() -> Dict[str, List[str]]:
    """Find all available trained models."""
    models = {
        'M1_CNN': [],
        'Ultra_CNN': [],
        'Enhanced_DQN': [],
        'Fixed_DQN': [],
        'Baselines': ['heuristic', 'random']
    }
    
    # M1 CNN models
    m1_dir = "models_m1_cnn"
    if os.path.exists(m1_dir):
        m1_files = glob.glob(f"{m1_dir}/*.pt")
        models['M1_CNN'] = sorted(m1_files, key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Ultra-light CNN models
    cnn_dir = "models_cnn"
    if os.path.exists(cnn_dir):
        cnn_files = glob.glob(f"{cnn_dir}/*.pt")
        models['Ultra_CNN'] = sorted(cnn_files, key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Enhanced DQN models
    enhanced_dir = "models_enhanced"
    if os.path.exists(enhanced_dir):
        enhanced_files = glob.glob(f"{enhanced_dir}/*.pt")
        models['Enhanced_DQN'] = sorted(enhanced_files, key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Fixed DQN models
    fixed_dir = "models_fixed"
    if os.path.exists(fixed_dir):
        fixed_files = glob.glob(f"{fixed_dir}/*.pt")
        models['Fixed_DQN'] = sorted(fixed_files, key=lambda x: os.path.getmtime(x), reverse=True)
    
    return models


if __name__ == "__main__":
    print("🤖 CONNECT 4 ENSEMBLE AGENT")
    print("=" * 50)
    
    # Display available models
    available_models = find_available_models()
    
    print("\n📁 Available Models:")
    for category, model_list in available_models.items():
        print(f"\n{category}:")
        if model_list:
            for model in model_list[:5]:  # Show first 5
                print(f"  • {os.path.basename(model)}")
            if len(model_list) > 5:
                print(f"  ... and {len(model_list) - 5} more")
        else:
            print("  (none found)")
    
    print(f"\n🎯 Preset Ensembles Available:")
    print("  • top_performers     - Best 4 models from tournament")
    print("  • diverse_architectures - Different model types")
    print("  • evolution_stages   - Models from different training stages")
    
    print(f"\n🔬 Ensemble Methods:")
    print("  • weighted_voting    - Weighted majority vote")
    print("  • q_value_averaging  - Average Q-values")
    print("  • confidence_weighted - Weight by model confidence")
    
    print(f"\n🎮 Usage Examples:")
    print("  python play_ensemble.py --preset top_performers")
    print("  python play_ensemble.py --custom models.json")
    print("  python ensemble_agent.py  # Interactive setup")