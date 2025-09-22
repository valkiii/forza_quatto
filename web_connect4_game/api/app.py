#!/usr/bin/env python3
"""
Flask API server for Connect 4 Ensemble AI
Loads actual .pt models and serves them via REST API
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.extend([parent_dir, grandparent_dir])

try:
    from ensemble_agent import EnsembleAgent
    from game.board import Connect4Board
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure you're running this from the correct directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web frontend

# Global variables
ensemble_agent = None
game_board = None
model_info = None


def load_ensemble_from_config(config_path: str = None) -> EnsembleAgent:
    """Load ensemble from JSON configuration or use default Top5Models-q."""
    
    # Default configuration matching examples/ensemble_config_v.json
    default_config = {
        "name": "Custom-Ensemble-Top5Models-q",
        "method": "q_value_averaging",
        "models": [
            {
                "path": "models_m1_cnn/m1_cnn_dqn_ep_750000.pt",
                "weight": 0.3,
                "name": "M1-CNN-750k"
            },
            {
                "path": "models_m1_cnn/m1_cnn_dqn_ep_700000.pt",
                "weight": 0.2,
                "name": "M1-CNN-700k"
            },
            {
                "path": "models_m1_cnn/m1_cnn_dqn_ep_650000.pt",
                "weight": 0.2,
                "name": "M1-CNN-650k"
            },
            {
                "path": "models_m1_cnn/m1_cnn_dqn_ep_600000.pt",
                "weight": 0.15,
                "name": "M1-CNN-600k"
            },
            {
                "path": "models_m1_cnn/m1_cnn_dqn_ep_550000.pt",
                "weight": 0.15,
                "name": "M1-CNN-550k"
            }
        ]
    }
    
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading ensemble config from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        logger.info("Using default Top5Models-q configuration")
        config = default_config
    
    # Convert to full paths
    base_dir = grandparent_dir
    for model_config in config["models"]:
        if not model_config["path"].startswith("/") and not model_config["path"] in ["heuristic", "random"]:
            model_config["path"] = os.path.join(base_dir, model_config["path"])
    
    # Create ensemble agent
    return EnsembleAgent(
        model_configs=config["models"],
        ensemble_method=config["method"],
        player_id=2,  # AI is player 2
        name=config["name"],
        show_contributions=True
    )


def initialize_api():
    """Initialize the API with ensemble agent."""
    global ensemble_agent, game_board, model_info
    
    try:
        logger.info("🚀 Initializing Connect 4 Ensemble API...")
        
        # Try to load from examples/ensemble_config_v.json first
        config_path = os.path.join(grandparent_dir, "examples", "ensemble_config_v.json")
        ensemble_agent = load_ensemble_from_config(config_path)
        
        # Initialize game board
        game_board = Connect4Board()
        
        # Store model information
        model_info = ensemble_agent.get_model_info()
        
        logger.info("✅ API initialized successfully!")
        logger.info(f"   Ensemble: {ensemble_agent.name}")
        logger.info(f"   Method: {ensemble_agent.ensemble_method}")
        logger.info(f"   Models: {len(ensemble_agent.models)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {e}")
        return False


@app.route('/')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Connect 4 Ensemble AI API",
        "ensemble_loaded": ensemble_agent is not None,
        "model_info": model_info if model_info else "Not loaded"
    })


@app.route('/api/ensemble/info', methods=['GET'])
def get_ensemble_info():
    """Get information about the loaded ensemble."""
    if not ensemble_agent:
        return jsonify({"error": "Ensemble not loaded"}), 500
    
    return jsonify({
        "ensemble_info": model_info,
        "last_decision": ensemble_agent.get_last_decision_breakdown() if hasattr(ensemble_agent, 'last_decision_info') and ensemble_agent.last_decision_info else None
    })


@app.route('/api/game/move', methods=['POST'])
def get_ai_move():
    """Get AI move for current board state."""
    if not ensemble_agent:
        return jsonify({"error": "Ensemble not loaded"}), 500
    
    try:
        data = request.get_json()
        
        # Validate input
        if 'board' not in data:
            return jsonify({"error": "Missing 'board' in request"}), 400
        
        board_state = data['board']
        
        # Validate board format (6x7 array)
        if not isinstance(board_state, list) or len(board_state) != 6:
            return jsonify({"error": "Board must be a 6x7 array"}), 400
        
        for row in board_state:
            if not isinstance(row, list) or len(row) != 7:
                return jsonify({"error": "Each row must have 7 columns"}), 400
            for cell in row:
                if cell not in [0, 1, 2]:
                    return jsonify({"error": "Board cells must be 0, 1, or 2"}), 400
        
        # Convert to numpy array for the AI
        board_array = np.array(board_state, dtype=int)
        
        # Get legal moves
        legal_moves = []
        for col in range(7):
            if board_array[0][col] == 0:  # Top row is empty
                legal_moves.append(col)
        
        if not legal_moves:
            return jsonify({"error": "No legal moves available"}), 400
        
        # Get AI move
        ai_move = ensemble_agent.choose_action(board_array, legal_moves)
        
        # Get decision breakdown
        decision_breakdown = ensemble_agent.get_last_decision_breakdown()
        
        # Get move evaluation details
        move_details = None
        if hasattr(ensemble_agent, 'last_decision_info') and ensemble_agent.last_decision_info:
            info = ensemble_agent.last_decision_info
            move_details = {
                "method": info['method'],
                "legal_moves": info['legal_moves'],
                "model_contributions": info.get('model_contributions', [])
            }
            
            if info['method'] == 'q_value_averaging':
                move_details["q_values"] = {
                    str(col): float(info['averaged_q_values'][col]) 
                    for col in info['legal_moves']
                }
        
        return jsonify({
            "move": int(ai_move),
            "legal_moves": legal_moves,
            "decision_breakdown": decision_breakdown,
            "move_details": move_details,
            "ensemble_method": ensemble_agent.ensemble_method,
            "model_count": len(ensemble_agent.models)
        })
        
    except Exception as e:
        logger.error(f"Error in get_ai_move: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/game/evaluate', methods=['POST'])
def evaluate_position():
    """Evaluate a board position and return Q-values for all legal moves."""
    if not ensemble_agent:
        return jsonify({"error": "Ensemble not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if 'board' not in data:
            return jsonify({"error": "Missing 'board' in request"}), 400
        
        board_state = np.array(data['board'], dtype=int)
        
        # Get legal moves
        legal_moves = []
        for col in range(7):
            if board_state[0][col] == 0:
                legal_moves.append(col)
        
        if not legal_moves:
            return jsonify({"error": "No legal moves available"}), 400
        
        # Force q_value_averaging to get detailed Q-values
        original_method = ensemble_agent.ensemble_method
        ensemble_agent.ensemble_method = "q_value_averaging"
        
        # Get AI evaluation
        ai_move = ensemble_agent.choose_action(board_state, legal_moves)
        
        # Restore original method
        ensemble_agent.ensemble_method = original_method
        
        # Extract detailed Q-values
        evaluation = {}
        if hasattr(ensemble_agent, 'last_decision_info') and ensemble_agent.last_decision_info:
            info = ensemble_agent.last_decision_info
            
            evaluation = {
                "best_move": int(ai_move),
                "legal_moves": legal_moves,
                "q_values": {
                    str(col): float(info['averaged_q_values'][col]) 
                    for col in legal_moves
                },
                "model_evaluations": []
            }
            
            # Individual model evaluations
            for contrib in info.get('model_contributions', []):
                if 'error' not in contrib and 'q_values' in contrib:
                    model_eval = {
                        "name": contrib['name'],
                        "preferred_move": contrib['action'],
                        "weight": contrib['weight'],
                        "q_values": {
                            str(col): float(contrib['q_values'][col]) 
                            for col in legal_moves
                        }
                    }
                    evaluation["model_evaluations"].append(model_eval)
        
        return jsonify(evaluation)
        
    except Exception as e:
        logger.error(f"Error in evaluate_position: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/game/hint', methods=['POST'])
def get_move_hint():
    """Get a hint for the human player (what would AI do in their position)."""
    if not ensemble_agent:
        return jsonify({"error": "Ensemble not loaded"}), 500
    
    try:
        data = request.get_json()
        
        if 'board' not in data:
            return jsonify({"error": "Missing 'board' in request"}), 400
        
        board_state = np.array(data['board'], dtype=int)
        
        # Flip perspective: AI analyzes as if it were player 1
        # Create a temporary agent for player 1
        hint_agent = EnsembleAgent(
            model_configs=[
                {'path': model.models[i], 'weight': model.weights[i], 'name': model.model_names[i]}
                for i, model in enumerate(ensemble_agent.models)
            ],
            ensemble_method=ensemble_agent.ensemble_method,
            player_id=1,  # Human player
            name="Hint-Agent"
        )
        
        # Get legal moves
        legal_moves = []
        for col in range(7):
            if board_state[0][col] == 0:
                legal_moves.append(col)
        
        if not legal_moves:
            return jsonify({"error": "No legal moves available"}), 400
        
        # Get hint move
        hint_move = hint_agent.choose_action(board_state, legal_moves)
        
        return jsonify({
            "hint_move": int(hint_move),
            "legal_moves": legal_moves,
            "explanation": f"The AI suggests column {hint_move + 1} as the best move for you.",
            "confidence": "high" if len(legal_moves) <= 3 else "medium"
        })
        
    except Exception as e:
        logger.error(f"Error in get_move_hint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/api/models/reload', methods=['POST'])
def reload_models():
    """Reload the ensemble models (useful for development)."""
    global ensemble_agent, model_info
    
    try:
        data = request.get_json() or {}
        config_path = data.get('config_path')
        
        logger.info("🔄 Reloading ensemble models...")
        ensemble_agent = load_ensemble_from_config(config_path)
        model_info = ensemble_agent.get_model_info()
        
        return jsonify({
            "status": "success",
            "message": "Models reloaded successfully",
            "model_info": model_info
        })
        
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        return jsonify({"error": f"Failed to reload models: {str(e)}"}), 500


if __name__ == '__main__':
    # Initialize the API
    if not initialize_api():
        logger.error("Failed to initialize API. Exiting.")
        sys.exit(1)
    
    # Start the server
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🚀 Starting Connect 4 Ensemble API on port {port}")
    logger.info(f"🌐 Access the API at: http://localhost:{port}")
    logger.info(f"📊 Health check: http://localhost:{port}/")
    logger.info(f"🤖 Ensemble info: http://localhost:{port}/api/ensemble/info")
    
    app.run(host='0.0.0.0', port=port, debug=debug)