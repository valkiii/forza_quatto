#!/usr/bin/env python3
"""Fixed Double DQN training with stable Q-value learning."""

import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn

class DuelingNetwork(nn.Module):
    """Dueling DQN architecture for better value/advantage separation."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        
        # Shared feature extraction layers
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),  # Regularization
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Value stream: V(s) - estimates state value
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Advantage stream: A(s,a) - estimates action advantage
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size)
        )
    
    def forward(self, x):
        # Extract shared features
        features = self.feature(x)
        
        # Compute value and advantage
        value = self.value(features)  # (B, 1)
        advantage = self.advantage(features)  # (B, A)
        
        # Combine using dueling architecture formula:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # This ensures V(s) represents state value, A(s,a) represents relative advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.random_agent import RandomAgent
from agents.heuristic_agent import HeuristicAgent
from agents.double_dqn_agent import DoubleDQNAgent
from train.training_monitor import TrainingMonitor
from train.double_dqn_train import (
    play_double_dqn_training_game, 
    evaluate_agent,
    setup_double_dqn_logging
)
from fix_qvalue_learning import create_fixed_training_config


class FixedDoubleDQNAgent(DoubleDQNAgent):
    """Enhanced DoubleDQNAgent with fixes for Q-value learning."""
    
    def __init__(self, *args, gradient_clip_norm=None, use_huber_loss=False, 
                 huber_delta=1.0, state_normalization=False, polyak_tau=0.001, 
                 use_reservoir_sampling=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.gradient_clip_norm = gradient_clip_norm
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        self.state_normalization = state_normalization
        self.polyak_tau = polyak_tau  # Polyak averaging coefficient
        self.use_reservoir_sampling = use_reservoir_sampling
        
        # Reservoir sampling for experience diversity
        if self.use_reservoir_sampling:
            self.reservoir_heuristic_experiences = []  # Keep diverse heuristic experiences
            self.reservoir_random_experiences = []     # Keep diverse random experiences
            self.heuristic_reservoir_size = min(5000, self.buffer_size // 4)  # 25% for heuristic
            self.random_reservoir_size = min(2000, self.buffer_size // 10)    # 10% for random
            self.experience_count = 0
            self.current_opponent_type = "unknown"  # Track current opponent type
        
        # Use Dueling DQN architecture instead of simple sequential network
        self.online_net = DuelingNetwork(self.state_size, self.action_size, hidden_size=256).to(self.device)
        self.target_net = DuelingNetwork(self.state_size, self.action_size, hidden_size=256).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Update optimizer for new network
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)
        
        # Replace loss function if using Huber loss
        if self.use_huber_loss:
            self.loss_fn = nn.SmoothL1Loss(beta=huber_delta)  # PyTorch uses 'beta', not 'delta'
        else:
            self.loss_fn = nn.MSELoss()
        
        print(f"🔧 FixedDoubleDQNAgent initialized:")
        print(f"  Gradient clipping: {gradient_clip_norm}")
        print(f"  Huber loss: {use_huber_loss} (delta={huber_delta})")
        print(f"  State normalization: {state_normalization}")
        print(f"  Polyak averaging: τ={polyak_tau}")
        if self.use_reservoir_sampling:
            print(f"  Reservoir sampling: {self.heuristic_reservoir_size + self.random_reservoir_size} reserved slots")
    
    def _legal_mask_from_encoded(self, encoded_states: torch.Tensor) -> torch.BoolTensor:
        """Extract legal action mask from encoded states."""
        # encoded_states: (B, state_size), state_size = 2 * rows * cols
        B = encoded_states.shape[0]
        rows, cols = 6, 7
        states_2d = encoded_states.view(B, 2, rows, cols)
        top_row_occupied = (states_2d[:, :, 0, :].sum(dim=1) > 0)  # (B, cols)
        legal_mask = ~top_row_occupied  # True where legal
        return legal_mask.to(self.device)

    def encode_state(self, board_state: np.ndarray) -> np.ndarray:
        """Enhanced state encoding with correct 2-channel normalization."""
        # get parent encoding (flattened 2-channel)
        encoded = super().encode_state(board_state)
        if not self.state_normalization:
            return encoded
        
        rows, cols = board_state.shape
        two_channel = encoded.reshape(2, rows, cols)
        
        # normalize channels to [-1,1]
        two_channel = two_channel * 2.0 - 1.0
        
        # center bonus per column
        col_idx = np.arange(cols)
        center_bonus_col = 0.1 * (1.0 - np.abs(col_idx - (cols-1)/2) / ((cols-1)/2))
        center_bonus = center_bonus_col[np.newaxis, :]  # shape (1, cols)
        empties = (board_state == 0)[np.newaxis, :, :]  # (1, rows, cols)
        two_channel = two_channel + (empties * center_bonus).astype(np.float32)
        
        return two_channel.reshape(-1)
    
    
    def observe(self, state, action, reward, next_state, done):
        """Enhanced observe with reservoir sampling for experience diversity."""
        # Call parent observe method first to handle normal storage and encoding
        super().observe(state, action, reward, next_state, done)
        
        # Reservoir sampling for diversity (use the encoded experience from parent)
        if self.use_reservoir_sampling and len(self.memory) > 0:
            # Get the last experience that was just added
            latest_experience = self.memory[-1]
            self.experience_count += 1
            
            # Store in appropriate reservoir based on current opponent type
            opponent_type = getattr(self, 'current_opponent_type', 'unknown')
            
            if opponent_type == "heuristic":
                if len(self.reservoir_heuristic_experiences) < self.heuristic_reservoir_size:
                    self.reservoir_heuristic_experiences.append(latest_experience)
                elif len(self.reservoir_heuristic_experiences) > 0 and self.rng.random() < self.heuristic_reservoir_size / self.experience_count:
                    replace_idx = self.rng.randint(0, len(self.reservoir_heuristic_experiences) - 1)
                    self.reservoir_heuristic_experiences[replace_idx] = latest_experience
            
            elif opponent_type == "random":
                if len(self.reservoir_random_experiences) < self.random_reservoir_size:
                    self.reservoir_random_experiences.append(latest_experience)
                elif len(self.reservoir_random_experiences) > 0 and self.rng.random() < self.random_reservoir_size / self.experience_count:
                    replace_idx = self.rng.randint(0, len(self.reservoir_random_experiences) - 1)
                    self.reservoir_random_experiences[replace_idx] = latest_experience
    
    def _train_step(self):
        """Enhanced training with reservoir-sampled batch diversity."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Create diverse batch with reservoir sampling
        if self.use_reservoir_sampling and (self.reservoir_heuristic_experiences or self.reservoir_random_experiences):
            # Calculate how many samples to take from each reservoir
            heuristic_samples = min(len(self.reservoir_heuristic_experiences), self.batch_size // 4)
            random_samples = min(len(self.reservoir_random_experiences), self.batch_size // 8)
            regular_samples = self.batch_size - heuristic_samples - random_samples
            
            batch = []
            
            # Add reservoir samples (ensure we have enough regular memory first)
            if regular_samples > 0 and len(self.memory) >= regular_samples:
                batch.extend(self.rng.sample(self.memory, regular_samples))
            
            if heuristic_samples > 0:
                batch.extend(self.rng.sample(self.reservoir_heuristic_experiences, heuristic_samples))
            if random_samples > 0:
                batch.extend(self.rng.sample(self.reservoir_random_experiences, random_samples))
            
            # If we don't have enough samples, fill with regular memory
            if len(batch) < self.batch_size and len(self.memory) >= self.batch_size - len(batch):
                additional_needed = self.batch_size - len(batch)
                batch.extend(self.rng.sample(self.memory, additional_needed))
            
            # Fallback: if still not enough, use only regular memory
            if len(batch) < self.batch_size:
                batch = self.rng.sample(self.memory, self.batch_size)
        else:
            # Standard sampling
            batch = self.rng.sample(self.memory, self.batch_size)
        
        # Continue with existing training logic...
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Create proper tensor arrays
        state_zero = np.zeros((self.state_size,), dtype=np.float32)
        next_states_array = np.stack([ns if ns is not None else state_zero for ns in next_states])
        states_array = np.stack(states).astype(np.float32)
        
        # Convert to tensors with proper device placement
        states_t = torch.from_numpy(states_array).float().to(self.device)
        next_states_t = torch.from_numpy(next_states_array).float().to(self.device)
        actions_t = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards_t = torch.from_numpy(np.array(rewards, dtype=np.float32)).to(self.device)
        dones_t = torch.from_numpy(np.array(dones, dtype=np.uint8)).to(self.device)
        
        # Current Q-values with explosion prevention
        q_values_raw = self.online_net(states_t)
        # CRITICAL: Clamp current Q-values to prevent explosion
        q_values_clamped = torch.clamp(q_values_raw, min=-100, max=100)
        current_q = q_values_clamped.gather(1, actions_t.unsqueeze(1)).view(-1)
        
        # Legal-action mask for next_states
        legal_mask = self._legal_mask_from_encoded(next_states_t)  # (B, A)
        large_neg = -1e9
        
        # Double DQN with legal action masking and Q-value explosion prevention
        with torch.no_grad():
            q_next_online = self.online_net(next_states_t)  # (B,A)
            # CRITICAL: Clamp Q-values to prevent explosion
            q_next_online = torch.clamp(q_next_online, min=-100, max=100)
            q_next_online_masked = q_next_online.masked_fill(~legal_mask, large_neg)
            next_actions = q_next_online_masked.argmax(dim=1, keepdim=True)  # (B,1)
            
            q_next_target = self.target_net(next_states_t)
            # CRITICAL: Clamp target Q-values to prevent explosion
            q_next_target = torch.clamp(q_next_target, min=-100, max=100)
            q_next_target_masked = q_next_target.masked_fill(~legal_mask, large_neg)
            next_q_vals = q_next_target_masked.gather(1, next_actions).squeeze(1)  # (B,)
            
            # Zero-out terminal next_q
            next_q_vals = next_q_vals * (1.0 - dones_t.float())
            # CRITICAL: Clamp next Q-values to prevent compound explosion
            next_q_vals = torch.clamp(next_q_vals, min=-100, max=100)
            
            targets = rewards_t + self.gamma * next_q_vals
            # CRITICAL: Final target clamping for numerical stability
            targets = torch.clamp(targets, min=-100, max=100)
        
        targets = targets.detach()
        
        # Compute loss with explicit dtype matching and explosion prevention
        current_q = current_q.to(targets.dtype)
        loss = self.loss_fn(current_q, targets)
        # CRITICAL: Clip loss magnitude to prevent gradient explosion
        loss = torch.clamp(loss, max=10.0)
        
        # Optimization step with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip_norm)
        
        self.optimizer.step()
        
        # Update training step count
        self.train_step_count += 1
        
        # CRITICAL FIX: Polyak averaging EVERY step (not conditional)
        # This is essential for stable learning and preventing catastrophic forgetting
        tau = self.polyak_tau
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )
        
        return loss.item()


def monitor_q_value_distribution(agent, num_samples: int = 100) -> dict:
    """Monitor Q-value distribution to detect divergence or collapse."""
    import numpy as np
    from game.board import Connect4Board
    
    agent_copy = agent.__class__(
        player_id=agent.player_id,
        state_size=agent.state_size,
        action_size=agent.action_size,
        seed=42
    )
    agent_copy.online_net.load_state_dict(agent.online_net.state_dict())
    agent_copy.epsilon = 0.0  # No exploration for monitoring
    
    all_q_values = []
    
    # Sample random board states
    for _ in range(num_samples):
        board = Connect4Board()
        
        # Create partially filled random board
        num_moves = np.random.randint(0, 15)  # Random game progress
        for move in range(num_moves):
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            col = np.random.choice(legal_moves)
            player = (move % 2) + 1
            board.make_move(col, player)
        
        # Get Q-values for this state
        encoded_state = agent_copy.encode_state(board.get_state())
        state_tensor = torch.from_numpy(encoded_state).float().unsqueeze(0).to(agent_copy.device)
        
        with torch.no_grad():
            q_values = agent_copy.online_net(state_tensor).cpu().numpy()[0]
            all_q_values.extend(q_values)
    
    all_q_values = np.array(all_q_values)
    
    # Calculate distribution statistics
    q_stats = {
        'mean': float(np.mean(all_q_values)),
        'std': float(np.std(all_q_values)),
        'min': float(np.min(all_q_values)),
        'max': float(np.max(all_q_values)),
        'median': float(np.median(all_q_values)),
        'q25': float(np.percentile(all_q_values, 25)),
        'q75': float(np.percentile(all_q_values, 75)),
        'num_samples': len(all_q_values),
        'range': float(np.max(all_q_values) - np.min(all_q_values))
    }
    
    # Detect potential issues
    warnings = []
    if q_stats['std'] > 50:
        warnings.append("High Q-value variance - possible divergence")
    if q_stats['range'] > 100:
        warnings.append("Very wide Q-value range - check for instability")
    if abs(q_stats['mean']) > 20:
        warnings.append("Q-values very far from zero - possible bias")
    if q_stats['std'] < 0.1:
        warnings.append("Very low Q-value variance - possible collapse")
    
    q_stats['warnings'] = warnings
    
    return q_stats


def evaluate_ensemble_performance(checkpoint_models: list, opponent, num_games: int = 100) -> dict:
    """Evaluate ensemble of models for more robust performance measurement."""
    if not checkpoint_models:
        return {"ensemble_win_rate": 0.0, "individual_results": []}
    
    individual_results = []
    
    for model_info in checkpoint_models:
        model_path, episode_num, win_rate = model_info
        
        # Load and evaluate this checkpoint
        temp_agent = FixedDoubleDQNAgent(player_id=1, state_size=84, action_size=7, seed=42)
        try:
            temp_agent.load(model_path)
            temp_agent.epsilon = 0.0  # No exploration during evaluation
            
            # Evaluate this checkpoint
            individual_win_rate = evaluate_agent(temp_agent, opponent, num_games=num_games)
            individual_results.append({
                'model_path': model_path,
                'episode': episode_num,
                'stored_win_rate': win_rate,
                'current_win_rate': individual_win_rate
            })
            
        except Exception as e:
            print(f"⚠️ Failed to evaluate {model_path}: {e}")
            individual_results.append({
                'model_path': model_path,
                'episode': episode_num,
                'stored_win_rate': win_rate,
                'current_win_rate': 0.0,
                'error': str(e)
            })
    
    # Calculate ensemble statistics
    if individual_results:
        current_win_rates = [r['current_win_rate'] for r in individual_results if 'error' not in r]
        ensemble_win_rate = np.mean(current_win_rates) if current_win_rates else 0.0
        ensemble_std = np.std(current_win_rates) if len(current_win_rates) > 1 else 0.0
    else:
        ensemble_win_rate = 0.0
        ensemble_std = 0.0
    
    return {
        'ensemble_win_rate': ensemble_win_rate,
        'ensemble_std': ensemble_std,
        'individual_results': individual_results,
        'num_models': len([r for r in individual_results if 'error' not in r])
    }


def train_fixed_double_dqn():
    """Training with fixes for stable Q-value learning."""
    print("🔧 FIXED Double DQN Training - Stable Q-Value Learning")
    print("=" * 60)
    
    # Load fixed configuration
    config = create_fixed_training_config()
    
    # Set seeds for reproducibility
    import random
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config["random_seed"])
        torch.cuda.manual_seed_all(config["random_seed"])  # Multi-GPU support
    
    # Set deterministic flags for full reproducibility (optional performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup logging
    log_dir = "logs_fixed"
    os.makedirs(log_dir, exist_ok=True)
    setup_double_dqn_logging(log_dir)  # Keep log file creation but don't store unused variable
    monitor = TrainingMonitor(log_dir=log_dir, save_plots=True, eval_frequency=config["eval_frequency"])
    
    print(f"📊 Logging to: {log_dir}/ with enhanced monitoring")
    
    # Initialize FIXED agent
    agent = FixedDoubleDQNAgent(
        player_id=1,
        state_size=84,
        action_size=7,
        lr=config["learning_rate"],
        gamma=config["discount_factor"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        epsilon_decay=config["epsilon_decay"],
        buffer_size=config["buffer_size"],
        batch_size=config["batch_size"],
        min_buffer_size=config["min_buffer_size"],
        target_update_freq=config["target_update_freq"],
        seed=config["random_seed"],
        # NEW FIXES
        gradient_clip_norm=config.get("gradient_clip_norm"),
        use_huber_loss=config.get("use_huber_loss", False),
        huber_delta=config.get("huber_delta", 1.0),
        state_normalization=config.get("state_normalization", False),
        polyak_tau=0.001,  # Polyak averaging coefficient for soft target updates
        use_reservoir_sampling=True  # Enable reservoir sampling for experience diversity
    )
    
    # Initialize opponents
    random_opponent = RandomAgent(player_id=2, seed=config["random_seed"] + 1)
    heuristic_opponent = HeuristicAgent(player_id=2, seed=config["random_seed"] + 1)
    
    # Training parameters
    RANDOM_PHASE_END = config["random_phase_end"]
    HEURISTIC_PHASE_END = config["heuristic_phase_end"]
    
    # Heuristic preservation tracking
    best_heuristic_performance = 0.0
    heuristic_performance_history = []
    
    # Ensemble evaluation tracking (multiple checkpoints for robust evaluation)
    checkpoint_models = []
    ensemble_size = 5  # Keep 5 best checkpoints for ensemble evaluation
    
    # Self-play opponent pool for curriculum learning
    self_play_opponents = []
    max_self_play_opponents = 3  # Keep 3 past model versions for self-play
    
    print(f"\n🎯 SMOOTH Curriculum Learning (Anti-Catastrophic Forgetting):")
    print(f"  📚 Episodes 1-{RANDOM_PHASE_END-2000:,}: Pure Random phase")
    print(f"  🔄 Episodes {RANDOM_PHASE_END-2000+1:,}-{RANDOM_PHASE_END+2000:,}: Smooth Random->Heuristic transition")
    print(f"  🧠 Episodes {RANDOM_PHASE_END+2000+1:,}-{HEURISTIC_PHASE_END-2000:,}: Pure Heuristic phase")
    print(f"  🔄 Episodes {HEURISTIC_PHASE_END-2000+1:,}-{HEURISTIC_PHASE_END+2000:,}: Smooth Heuristic->Mixed transition")
    print(f"  🎆 Episodes {HEURISTIC_PHASE_END+2000+1:,}+: Progressive mixed with self-play")
    print(f"\n🔧 Q-Value Learning Fixes Active:")
    print(f"  ✂️  Gradient clipping: {config.get('gradient_clip_norm', 'None')}")
    print(f"  📊 Enhanced state encoding: {config.get('state_normalization', False)}")
    print(f"  🎯 Huber loss: {config.get('use_huber_loss', False)}")
    print(f"  🐌 Conservative learning rate: {config['learning_rate']}")
    print(f"  🔄 Polyak averaging: τ=0.005 (soft target updates)")
    print(f"  📊 Ensemble evaluation: {ensemble_size} best checkpoints")
    print(f"  📈 Q-value distribution monitoring: histogram analysis")
    print(f"  📦 Ultra-large buffer: {config['buffer_size']:,} experiences")
    print(f"  🏯 Reservoir sampling: Preserve diverse heuristic & random experiences")
    print(f"  🎨 Dueling DQN architecture: Value/Advantage separation")
    print(f"  🚨 Q-value explosion prevention: Clamping [-100, 100]")
    print(f"  🔥 Learning rate warmup: 1e-6 -> 5e-5 over 5K episodes")
    print(f"  💯 Reduced discount factor: 0.95 (was 0.99) to prevent compound growth")
    
    # Simple training loop for testing fixes
    episode_rewards = []
    
    for episode in range(config["num_episodes"]):  # Full training (no artificial limit)
        episode_num = episode + 1
        
        # Select opponent with SMOOTHER curriculum transitions to prevent catastrophic forgetting
        import random as py_random
        
        if episode_num <= RANDOM_PHASE_END - 2000:  # Pure random phase
            current_opponent = random_opponent
            opponent_name = "Random"
        elif episode_num <= RANDOM_PHASE_END + 2000:  # SMOOTH TRANSITION PERIOD 1: Random -> Heuristic
            # Gradually increase heuristic probability over 4000 episodes
            transition_progress = (episode_num - (RANDOM_PHASE_END - 2000)) / 4000.0
            heuristic_prob = min(1.0, max(0.0, transition_progress))
            if py_random.random() < heuristic_prob:
                current_opponent = heuristic_opponent
                opponent_name = f"Transition R->H ({heuristic_prob:.1%} heuristic)"
            else:
                current_opponent = random_opponent
                opponent_name = f"Transition R->H ({1-heuristic_prob:.1%} random)"
        elif episode_num <= HEURISTIC_PHASE_END - 2000:  # Pure heuristic phase
            current_opponent = heuristic_opponent
            opponent_name = "Heuristic"
        elif episode_num <= HEURISTIC_PHASE_END + 2000:  # SMOOTH TRANSITION PERIOD 2: Heuristic -> Mixed
            # Gradually introduce mixed opponents over 4000 episodes
            transition_progress = (episode_num - (HEURISTIC_PHASE_END - 2000)) / 4000.0
            mixed_prob = min(1.0, max(0.0, transition_progress))
            
            if py_random.random() < mixed_prob:
                # Start introducing mixed opponents gradually
                if py_random.random() < 0.3:  # 30% random in mixed
                    current_opponent = random_opponent
                    opponent_name = f"Transition H->M (random, {mixed_prob:.1%} mixed)"
                else:  # 70% heuristic in mixed
                    current_opponent = heuristic_opponent  
                    opponent_name = f"Transition H->M (heuristic, {mixed_prob:.1%} mixed)"
            else:
                current_opponent = heuristic_opponent
                opponent_name = f"Transition H->M ({1-mixed_prob:.1%} pure heuristic)"
        else:
            # PROGRESSIVE MIXED PHASE WITH SELF-PLAY: After smooth transitions
            
            # Progressive schedule based on episode number
            if episode_num <= 90000:
                # Early mixed: Heavy heuristic preservation
                heuristic_prob = 0.50  # Maintain strong heuristic connection
                random_prob = 0.30
                self_play_prob = 0.20  # Introduce self-play gradually
            elif episode_num <= 110000:
                # Mid mixed: Balanced exploration with more self-play
                heuristic_prob = 0.35
                random_prob = 0.25
                self_play_prob = 0.40
            elif episode_num <= 130000:
                # Late mixed: More self-play but keep heuristic
                heuristic_prob = 0.25
                random_prob = 0.20
                self_play_prob = 0.55
            else:
                # Final: Self-play dominant but never abandon heuristic completely
                heuristic_prob = 0.20
                random_prob = 0.15
                self_play_prob = 0.65
            
            # Sample opponent based on progressive probabilities
            rand_val = py_random.random()
            if rand_val < heuristic_prob:
                current_opponent = heuristic_opponent
                opponent_name = f"Heuristic (progressive {heuristic_prob:.1%})"
            elif rand_val < heuristic_prob + random_prob:
                current_opponent = random_opponent
                opponent_name = f"Random (progressive {random_prob:.1%})"
            else:
                # Self-play: Use past model checkpoint
                if self_play_opponents:
                    self_play_opponent, model_episode = py_random.choice(self_play_opponents)
                    current_opponent = self_play_opponent
                    opponent_name = f"Self-play (ep_{model_episode}, prob={self_play_prob:.1%})"
                else:
                    # Fallback to heuristic if no self-play models available
                    current_opponent = heuristic_opponent
                    opponent_name = f"Heuristic (self-play fallback)"
        
        # Update monitor with current opponent (important for curriculum learning)
        monitor.set_current_opponent(opponent_name)
        
        # CRITICAL: Dynamic learning rate with warmup to prevent instability
        if episode_num < 5000:  # Warmup phase
            base_lr = config["learning_rate"]
            warmup_lr = 1e-6 + (base_lr - 1e-6) * (episode_num / 5000.0)
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        elif episode_num % 5000 == 0:  # Decay every 5000 episodes after warmup - LESS FREQUENT
            decay_factor = 0.9  # Less aggressive decay
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= decay_factor
                print(f"📉 Learning rate decayed to: {param_group['lr']:.6f}")
        
        # Play training game
        agent.reset_episode()
        
        # Set opponent type for reservoir sampling
        if "Heuristic" in opponent_name:
            agent.current_opponent_type = "heuristic"
        elif "Random" in opponent_name:
            agent.current_opponent_type = "random"
        else:
            agent.current_opponent_type = "self_play"
        
        winner, _ = play_double_dqn_training_game(
            agent, current_opponent, monitor, "enhanced", config.get("reward_system")
        )
        
        # Calculate episode reward WITHOUT clipping to preserve strategic signals
        if winner == agent.player_id:
            episode_reward = 10.0  # Win
        elif winner is not None:
            episode_reward = -10.0  # Loss
        else:
            episode_reward = 1.0  # Draw
        
        # CRITICAL: No reward clipping! Strategic rewards (2-5 magnitude) need to be visible
        episode_rewards.append(episode_reward)
        
        # Enhanced monitoring
        if episode_num % config["eval_frequency"] == 0:
            win_rate = evaluate_agent(agent, current_opponent, num_games=50)
            
            monitor.log_episode(episode_num, episode_reward, agent, win_rate)
            monitor.log_strategic_episode(episode_num)
            monitor.generate_training_report(agent, episode_num)
            monitor.reset_strategic_stats()
            
            # CSV logging for compatibility and monitoring
            csv_file = os.path.join(log_dir, "double_dqn_log.csv")
            avg_reward = np.mean(episode_rewards[-config['eval_frequency']:]) if len(episode_rewards) >= config['eval_frequency'] else np.mean(episode_rewards)
            
            # Write CSV data
            file_exists = os.path.exists(csv_file)
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["episode", "avg_reward", "epsilon", "training_steps", "buffer_size", "win_rate"])
                writer.writerow([episode_num, avg_reward, agent.epsilon, agent.train_step_count, len(agent.memory), win_rate])
            
            print(f"📊 Episode {episode_num:,}: {win_rate:.1%} vs {opponent_name}")
            print(f"    Avg reward: {np.mean(episode_rewards[-config['eval_frequency']:]):.2f}")
            print(f"    Training steps: {agent.train_step_count:,}")
            
            # Debug: Check if rewards are amplified (every 1000 episodes)
            if episode_num % 1000 == 0 and len(agent.memory) > 0:
                recent_rewards = [exp[2] for exp in list(agent.memory)[-20:]]
                strategic_rewards = [r for r in recent_rewards if abs(r) not in [10.0, 1.0]]  # Non-terminal rewards
                if strategic_rewards:
                    print(f"    🔍 Strategic rewards detected: {strategic_rewards[:5]} (amplified!)")
                else:
                    print(f"    ⚠️ No strategic rewards in recent {len(recent_rewards)} experiences")
                
                # CRITICAL: Monitor Q-value magnitudes to detect explosion
                try:
                    with torch.no_grad():
                        # Sample a few states to check Q-value magnitudes
                        sample_states = [exp[0] for exp in list(agent.memory)[-10:]]
                        if sample_states:
                            # Convert list of numpy arrays to single numpy array first (efficiency fix)
                            states_array = np.array(sample_states)
                            states_tensor = torch.from_numpy(states_array).float().to(agent.device)
                            q_values = agent.online_net(states_tensor)
                            q_max = torch.max(torch.abs(q_values)).item()
                            q_mean = torch.mean(torch.abs(q_values)).item()
                            
                            print(f"    📊 Q-value magnitudes: max={q_max:.2f}, mean={q_mean:.2f}")
                            if q_max > 50:
                                print(f"    ⚠️ WARNING: Q-values getting large (max={q_max:.1f})")
                            if q_max > 100:
                                print(f"    🚨 CRITICAL: Q-value explosion detected! (max={q_max:.1f})")
                except Exception as e:
                    print(f"    ❌ Q-value monitoring failed: {e}")
            
            # CRITICAL: Heuristic preservation monitoring (only after heuristic phase)
            if episode_num > HEURISTIC_PHASE_END:
                print(f"\n🛡️ Heuristic preservation check at episode {episode_num:,}...")
                heuristic_win_rate = evaluate_agent(agent, heuristic_opponent, num_games=100)
                heuristic_performance_history.append((episode_num, heuristic_win_rate))
                
                # Track best performance and maintain ensemble
                if heuristic_win_rate > best_heuristic_performance:
                    best_heuristic_performance = heuristic_win_rate
                    # Save best model
                    os.makedirs("models_fixed", exist_ok=True)
                    best_model_path = f"models_fixed/double_dqn_best_heuristic_ep_{episode_num}.pt"
                    agent.save(best_model_path)
                    print(f"💎 NEW BEST heuristic performance: {heuristic_win_rate:.1%} - saved {os.path.basename(best_model_path)}")
                
                # Save checkpoint for ensemble tracking
                os.makedirs("models_fixed", exist_ok=True)
                checkpoint_path = f"models_fixed/double_dqn_ep_{episode_num}.pt"
                agent.save(checkpoint_path)
                
                # Maintain ensemble of top checkpoints
                checkpoint_models.append((
                    checkpoint_path,
                    episode_num,
                    heuristic_win_rate
                ))
                
                # Keep only top N checkpoints
                if len(checkpoint_models) > ensemble_size:
                    # Sort by win rate (descending) and keep top N
                    checkpoint_models.sort(key=lambda x: x[2], reverse=True)
                    checkpoint_models = checkpoint_models[:ensemble_size]
                
                # Ensemble evaluation every few evaluations
                if len(checkpoint_models) >= 3 and episode_num % (config["eval_frequency"] * 4) == 0:
                    print(f"\n🎯 ENSEMBLE EVALUATION at episode {episode_num:,}...")
                    ensemble_results = evaluate_ensemble_performance(
                        checkpoint_models, heuristic_opponent, num_games=50
                    )
                    
                    print(f"📊 Ensemble performance ({ensemble_results['num_models']} models):")
                    print(f"    Mean win rate: {ensemble_results['ensemble_win_rate']:.1%}")
                    print(f"    Std deviation: {ensemble_results['ensemble_std']:.3f}")
                    print(f"    Individual models:")
                    for result in ensemble_results['individual_results'][:3]:  # Show top 3
                        if 'error' not in result:
                            print(f"      Episode {result['episode']:,}: {result['current_win_rate']:.1%}")
                
                print(f"📊 Current vs best: {heuristic_win_rate:.1%} vs {best_heuristic_performance:.1%}")
                
                # FIXED: Early stopping only after heuristic phase with 30% threshold
                HEURISTIC_DEGRADATION_THRESHOLD = 0.30  # 30% threshold (not 90%!)
                if heuristic_win_rate < HEURISTIC_DEGRADATION_THRESHOLD:
                    print(f"\n🚨 CRITICAL HEURISTIC DEGRADATION DETECTED!")
                    print(f"Performance: {heuristic_win_rate:.1%} < threshold {HEURISTIC_DEGRADATION_THRESHOLD:.1%}")
                    print(f"🛑 EMERGENCY STOP - preventing catastrophic forgetting")
                    
                    # Emergency save
                    emergency_save_path = f"models_fixed/double_dqn_emergency_stop_ep_{episode_num}.pt"
                    agent.save(emergency_save_path)
                    print(f"💾 Emergency save: {emergency_save_path}")
                    
                    # Save detailed failure analysis
                    failure_analysis = {
                        'episode': episode_num,
                        'heuristic_performance': heuristic_win_rate,
                        'threshold': HEURISTIC_DEGRADATION_THRESHOLD,
                        'best_performance_achieved': best_heuristic_performance,
                        'performance_history': heuristic_performance_history,
                        'current_learning_rate': agent.optimizer.param_groups[0]['lr'],
                        'config': config,
                        'note': 'Early stopping triggered only after heuristic phase (episode 35K+) with 30% threshold'
                    }
                    
                    with open(f"{log_dir}/catastrophic_forgetting_analysis.json", 'w') as f:
                        json.dump(failure_analysis, f, indent=2)
                    
                    print(f"📋 Failure analysis saved for debugging")
                    break  # Stop training
                else:
                    performance_status = "🟢 EXCELLENT" if heuristic_win_rate > 0.80 else "✅ GOOD" if heuristic_win_rate > 0.50 else "⚠️ ACCEPTABLE"
                    print(f"{performance_status} - Heuristic knowledge preserved")
            
            # Q-value distribution monitoring
            if episode_num % (config["eval_frequency"] * 2) == 0:
                print(f"🔍 Q-value distribution analysis...")
                try:
                    q_stats = monitor_q_value_distribution(agent, num_samples=50)
                    print(f"    Q-value stats: μ={q_stats['mean']:.2f}, σ={q_stats['std']:.2f}")
                    print(f"    Range: [{q_stats['min']:.2f}, {q_stats['max']:.2f}]")
                    print(f"    Median: {q_stats['median']:.2f}, IQR: [{q_stats['q25']:.2f}, {q_stats['q75']:.2f}]")
                    
                    if q_stats['warnings']:
                        print(f"    ⚠️ Q-value warnings:")
                        for warning in q_stats['warnings']:
                            print(f"      - {warning}")
                    else:
                        print(f"    ✅ Q-values look healthy")
                    
                    # Save model for potential diagnostic and self-play
                    test_model_path = f"models_fixed/double_dqn_ep_{episode_num}.pt"
                    os.makedirs("models_fixed", exist_ok=True)
                    agent.save(test_model_path)
                    print(f"    💾 Model saved: {os.path.basename(test_model_path)}")
                    
                    # Add to self-play opponent pool periodically (every 20K episodes in mixed phase)
                    if (episode_num > HEURISTIC_PHASE_END and 
                        episode_num % 20000 == 0 and 
                        len(self_play_opponents) < max_self_play_opponents):
                        
                        print(f"\n🤖 Adding self-play opponent from episode {episode_num:,}...")
                        try:
                            # Create new agent instance for self-play
                            self_play_agent = FixedDoubleDQNAgent(
                                player_id=2,  # Opponent player ID
                                state_size=84,
                                action_size=7,
                                seed=config["random_seed"] + len(self_play_opponents) + 100
                            )
                            self_play_agent.load(test_model_path)
                            self_play_agent.epsilon = 0.02  # Very low exploration for consistent play
                            
                            # Add to opponent pool
                            self_play_opponents.append((self_play_agent, episode_num))
                            print(f"✅ Self-play opponent added: {len(self_play_opponents)}/{max_self_play_opponents} models")
                            
                            # Remove oldest if at capacity
                            if len(self_play_opponents) > max_self_play_opponents:
                                _, removed_episode = self_play_opponents.pop(0)
                                print(f"🗑️ Removed oldest self-play model from episode {removed_episode:,}")
                                
                        except Exception as e:
                            print(f"    ❌ Failed to create self-play opponent: {e}")
                except Exception as e:
                    print(f"    ❌ Q-value analysis failed: {e}")
                    pass
        else:
            monitor.log_episode(episode_num, episode_reward, agent)
        
        # Save checkpoints
        if episode_num % config["save_frequency"] == 0:
            os.makedirs("models_fixed", exist_ok=True)
            model_path = f"models_fixed/double_dqn_ep_{episode_num}.pt"
            agent.save(model_path)
            print(f"💾 Saved fixed model: {model_path}")
    
    print(f"\n✅ Fixed training completed! Check models_fixed/ and logs_fixed/")
    print(f"🔍 Run diagnostic on fixed models to verify Q-value learning improvements")


if __name__ == "__main__":
    train_fixed_double_dqn()
