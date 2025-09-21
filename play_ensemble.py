#!/usr/bin/env python3
"""Interactive Connect 4 game against ensemble agent."""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game.board import Connect4Board
from ensemble_agent import EnsembleAgent, create_preset_ensemble, find_available_models


def display_board(board: Connect4Board):
    """Display the board in a nice format."""
    print("\n" + "=" * 29)
    print("  0   1   2   3   4   5   6")
    print("+" + "---+" * 7)
    
    for row in range(6):
        print("|", end="")
        for col in range(7):
            piece = board.board[row][col]
            if piece == 0:
                print("   |", end="")
            elif piece == 1:
                print(" ● |", end="")  # Human player
            else:
                print(" ○ |", end="")  # Ensemble agent
        print()
        print("+" + "---+" * 7)
    print("  0   1   2   3   4   5   6")
    print("=" * 29)


def get_human_move(board: Connect4Board) -> int:
    """Get a valid move from the human player."""
    legal_moves = board.get_legal_moves()
    
    while True:
        try:
            print(f"\nLegal moves: {legal_moves}")
            move = input("Your move (column 0-6, or 'q' to quit): ").strip()
            
            if move.lower() == 'q':
                return -1
            
            move = int(move)
            if move in legal_moves:
                return move
            else:
                print(f"❌ Invalid move! Column {move} is full or out of range.")
        except ValueError:
            print("❌ Please enter a valid number (0-6) or 'q' to quit.")


def create_custom_ensemble_interactive() -> EnsembleAgent:
    """Interactive ensemble creation."""
    print("\n🎯 Custom Ensemble Creation")
    print("=" * 40)
    
    available_models = find_available_models()
    
    # Show available models
    all_models = []
    print("\n📁 Available Models:")
    model_index = 0
    
    for category, model_list in available_models.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"  {model_index}: {os.path.basename(model)}")
            all_models.append(model)
            model_index += 1
    
    # Get ensemble method
    print(f"\n🔬 Choose ensemble method:")
    print("  0: weighted_voting")
    print("  1: q_value_averaging") 
    print("  2: confidence_weighted")
    
    while True:
        try:
            method_choice = int(input("Method (0-2): "))
            methods = ["weighted_voting", "q_value_averaging", "confidence_weighted"]
            ensemble_method = methods[method_choice]
            break
        except (ValueError, IndexError):
            print("❌ Please enter 0, 1, or 2")
    
    # Get models and weights
    model_configs = []
    print(f"\n🤖 Add models to ensemble (enter -1 when done):")
    
    while True:
        try:
            model_idx = int(input(f"Model index (-1 to finish): "))
            if model_idx == -1:
                break
            if 0 <= model_idx < len(all_models):
                weight = float(input(f"Weight for {os.path.basename(all_models[model_idx])} (default 1.0): ") or "1.0")
                model_configs.append({
                    'path': all_models[model_idx],
                    'weight': weight,
                    'name': os.path.basename(all_models[model_idx])
                })
                print(f"✅ Added {os.path.basename(all_models[model_idx])} with weight {weight}")
            else:
                print(f"❌ Index {model_idx} out of range")
        except ValueError:
            print("❌ Please enter valid numbers")
    
    if not model_configs:
        print("❌ No models selected, using default ensemble")
        return create_preset_ensemble("top_performers", player_id=2)
    
    return EnsembleAgent(model_configs, ensemble_method, player_id=2, name="Custom-Ensemble")


def load_ensemble_from_config(config_file: str) -> EnsembleAgent:
    """Load ensemble from JSON configuration file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    model_configs = config.get('models', [])
    ensemble_method = config.get('method', 'weighted_voting')
    name = config.get('name', 'Config-Ensemble')
    
    return EnsembleAgent(model_configs, ensemble_method, player_id=2, name=name)


def save_ensemble_config(ensemble: EnsembleAgent, filename: str):
    """Save ensemble configuration to JSON file."""
    config = {
        'name': ensemble.name,
        'method': ensemble.ensemble_method,
        'models': [
            {
                'path': 'heuristic' if model.__class__.__name__ == 'HeuristicAgent' 
                       else 'random' if model.__class__.__name__ == 'RandomAgent'
                       else f"model_{i}",  # Would need to store original paths
                'weight': weight,
                'name': name
            }
            for i, (model, weight, name) in enumerate(zip(ensemble.models, ensemble.weights, ensemble.model_names))
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"💾 Ensemble configuration saved to {filename}")


def play_game(ensemble: EnsembleAgent, human_starts: bool = True) -> Optional[int]:
    """Play a single game between human and ensemble."""
    board = Connect4Board()
    
    # Set player IDs
    human_player = 1 if human_starts else 2
    ensemble.player_id = 2 if human_starts else 1
    
    print(f"\n🎮 Starting new game!")
    print(f"You are {'●' if human_player == 1 else '○'} (Player {human_player})")
    print(f"Ensemble is {'○' if ensemble.player_id == 2 else '●'} (Player {ensemble.player_id})")
    
    move_count = 0
    
    while not board.is_terminal() and move_count < 42:
        display_board(board)
        
        if board.current_player == human_player:
            # Human turn
            print(f"\n🧑 Your turn!")
            move = get_human_move(board)
            if move == -1:  # Quit
                return None
            
            board.make_move(move, human_player)
            print(f"You played column {move}")
            
        else:
            # Ensemble turn
            print(f"\n🤖 {ensemble.name} thinking...")
            legal_moves = board.get_legal_moves()
            move = ensemble.choose_action(board.get_state(), legal_moves)
            
            # Show detailed decision breakdown
            if hasattr(ensemble, 'show_contributions') and ensemble.show_contributions:
                breakdown = ensemble.get_last_decision_breakdown()
                print(f"\n{breakdown}")
            
            board.make_move(move, ensemble.player_id)
            print(f"\n➡️  {ensemble.name} played column {move}")
        
        move_count += 1
    
    display_board(board)
    winner = board.check_winner()
    
    if winner == human_player:
        print(f"\n🎉 Congratulations! You won!")
        return human_player
    elif winner == ensemble.player_id:
        print(f"\n🤖 {ensemble.name} wins! Better luck next time!")
        return ensemble.player_id
    else:
        print(f"\n🤝 It's a draw!")
        return 0


def main():
    """Main game loop."""
    parser = argparse.ArgumentParser(description='Play Connect 4 against Ensemble Agent')
    parser.add_argument('--preset', choices=['top_performers', 'diverse_architectures', 'evolution_stages'],
                       help='Use preset ensemble configuration')
    parser.add_argument('--config', type=str, help='JSON file with custom ensemble configuration')
    parser.add_argument('--interactive', action='store_true', help='Create ensemble interactively')
    parser.add_argument('--save-config', type=str, help='Save ensemble config to file')
    parser.add_argument('--info', action='store_true', help='Show ensemble info and exit')
    parser.add_argument('--show-contributions', action='store_true', help='Show detailed model contributions during play')
    parser.add_argument('--hide-contributions', action='store_true', help='Hide model contributions (default for custom ensembles)')
    
    args = parser.parse_args()
    
    print("🎮 CONNECT 4 vs ENSEMBLE AGENT")
    print("=" * 50)
    
    # Create ensemble based on arguments
    ensemble = None
    
    if args.config:
        try:
            ensemble = load_ensemble_from_config(args.config)
            print(f"✅ Loaded ensemble from {args.config}")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return
    
    elif args.preset:
        try:
            ensemble = create_preset_ensemble(args.preset, player_id=2)
            print(f"✅ Created {args.preset} ensemble")
        except Exception as e:
            print(f"❌ Failed to create preset: {e}")
            return
    
    elif args.interactive:
        ensemble = create_custom_ensemble_interactive()
    
    else:
        # Default to top performers
        try:
            ensemble = create_preset_ensemble("top_performers", player_id=2)
            print(f"✅ Using default 'top_performers' ensemble")
        except Exception as e:
            print(f"❌ Failed to create default ensemble: {e}")
            return
    
    # Override contribution display based on command line args
    if args.show_contributions:
        ensemble.show_contributions = True
    elif args.hide_contributions:
        ensemble.show_contributions = False
    
    # Show ensemble info
    if args.info or not ensemble:
        if ensemble:
            info = ensemble.get_model_info()
            print(f"\n🤖 Ensemble Information:")
            print(f"Name: {ensemble.name}")
            print(f"Method: {info['ensemble_method']}")
            print(f"Models: {info['num_models']}")
            for model_info in info['models']:
                print(f"  • {model_info['name']} ({model_info['type']}) - weight: {model_info['weight']:.3f}")
        
        if args.info:
            return
    
    # Save configuration if requested
    if args.save_config and ensemble:
        save_ensemble_config(ensemble, args.save_config)
    
    if not ensemble:
        print("❌ No ensemble created")
        return
    
    # Game statistics
    games_played = 0
    human_wins = 0
    ensemble_wins = 0
    draws = 0
    
    print(f"\n🎯 Ready to play against {ensemble.name}!")
    print("Enter 'q' during any move to quit")
    
    try:
        while True:
            games_played += 1
            human_starts = games_played % 2 == 1  # Alternate who starts
            
            print(f"\n" + "="*50)
            print(f"GAME {games_played}")
            print(f"="*50)
            
            result = play_game(ensemble, human_starts)
            
            if result is None:  # User quit
                break
            elif result == 1 or result == 2:
                if (result == 1 and human_starts) or (result == 2 and not human_starts):
                    human_wins += 1
                else:
                    ensemble_wins += 1
            else:
                draws += 1
            
            # Show statistics
            print(f"\n📊 STATISTICS (after {games_played} games):")
            print(f"You: {human_wins} wins ({human_wins/games_played*100:.1f}%)")
            print(f"{ensemble.name}: {ensemble_wins} wins ({ensemble_wins/games_played*100:.1f}%)")
            print(f"Draws: {draws} ({draws/games_played*100:.1f}%)")
            
            # Ask to continue
            if games_played >= 1:
                continue_game = input(f"\nPlay another game? (y/n): ").lower()
                if continue_game != 'y':
                    break
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️ Game interrupted by user")
    
    print(f"\n🏁 Final Statistics:")
    print(f"Games played: {games_played}")
    print(f"Your wins: {human_wins} ({human_wins/games_played*100:.1f}%)")
    print(f"{ensemble.name} wins: {ensemble_wins} ({ensemble_wins/games_played*100:.1f}%)")
    print(f"Draws: {draws} ({draws/games_played*100:.1f}%)")
    
    print(f"\nThanks for playing! 🎮")


if __name__ == "__main__":
    main()