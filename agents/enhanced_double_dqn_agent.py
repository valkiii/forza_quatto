"""Enhanced Double DQN agent with larger network, dueling architecture, and strategic improvements."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Optional, Tuple
import math

try:
    from .base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent

# Named tuple for storing experiences with priority support
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DuelingNetwork(nn.Module):
    """Dueling network architecture that separates value and advantage estimation."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 512):
        super().__init__()
        
        # Store action_size for initialization
        self.action_size = action_size
        
        # Deeper feature extraction with 3 layers
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Larger value head
        self.value = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Larger advantage head
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
        
        # Initialize weights for stability
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with stronger optimistic bias to overcome pessimistic attractor."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Keep Xavier for weights (stable) but scale down final weights slightly
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # If this is the final output of the value head (out_features == 1)
                    # or the advantage head (out_features == action_size), set a positive bias.
                    if m.out_features == 1 or m.out_features == self.action_size:
                        # Strong optimistic initialization for outputs
                        nn.init.constant_(m.bias, 10.0)
                    else:
                        # Small positive bias in hidden layers so activations start slightly optimistic
                        nn.init.constant_(m.bias, 0.0)


    def forward(self, x):
        """Forward pass through dueling network."""
        features = self.feature(x)
        
        value = self.value(features)
        advantage = self.advantage(features)
        
        # Combine value and advantage: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with importance sampling."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, beta_end: float = 1.0):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        self._max_priority = 1.0
    
    def add(self, experience: Experience, td_error: float = None):
        """Add experience with priority based on TD error."""
        if td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        else:
            priority = self._max_priority
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self._max_priority = max(self._max_priority, priority)
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights."""
        if self.size == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors."""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)
    
    def __len__(self):
        return self.size


class EnhancedDoubleDQNAgent(BaseAgent):
    """Enhanced Double DQN agent with all proposed improvements."""
    
    def __init__(self, player_id: int, state_size: int = 84, action_size: int = 7,
                 hidden_size: int = 512, gamma: float = 0.95, lr: float = 5e-5,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: float = 0.9998,
                 batch_size: int = 256, buffer_size: int = 200000, min_buffer_size: int = 1000,
                 target_update_freq: int = 2000, n_step: int = 3, use_prioritized_replay: bool = True,
                 polyak_tau: float = 0.001, seed: int = None):
        """Initialize Enhanced Double DQN agent."""
        super().__init__(player_id, "Enhanced-Double-DQN")
        
        # Agent parameters
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.n_step = n_step
        self.use_prioritized_replay = use_prioritized_replay
        self.polyak_tau = polyak_tau
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks with dueling architecture
        self.online_net = DuelingNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net = DuelingNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        
        # Experience replay buffer
        if use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size)
        else:
            self.memory = deque(maxlen=buffer_size)
        
        # N-step learning buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Training tracking
        self.train_step_count = 0
        self._episode_count = 0  # Track episodes for optimism bootstrap
        self.beta_scheduler = lambda step: min(1.0, 0.4 + 0.6 * step / 100000)
        
        # Random number generator
        self.rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def encode_state_with_features(self, board_state: np.ndarray) -> np.ndarray:
        """Enhanced state representation with strategic features."""
        # Basic two-channel encoding
        base_encoding = self._encode_base_state(board_state)
        
        # Strategic features
        features = []
        
        # Threat detection
        features.append(self._count_threats(board_state, self.player_id))
        features.append(self._count_threats(board_state, 3 - self.player_id))
        
        # Center control
        features.append(self._count_center_pieces(board_state, self.player_id))
        features.append(self._count_center_pieces(board_state, 3 - self.player_id))
        
        # Connectivity
        features.append(self._max_connected_pieces(board_state, self.player_id))
        features.append(self._max_connected_pieces(board_state, 3 - self.player_id))
        
        # Height advantage
        features.append(self._calculate_height_advantage(board_state, self.player_id))
        
        # Mobility (number of legal moves)
        features.append(len(self._get_legal_moves_from_state(board_state)))
        
        return np.concatenate([base_encoding, np.array(features, dtype=np.float32)])
    
    def _encode_base_state(self, board_state: np.ndarray) -> np.ndarray:
        """Base two-channel encoding with center bias."""
        if not isinstance(board_state, np.ndarray):
            board_state = np.array(board_state)
        
        # Create 2-channel representation
        relative_board = np.zeros((2, *board_state.shape), dtype=np.float32)
        
        # Channel 0: Agent's pieces
        relative_board[0][board_state == self.player_id] = 1.0
        
        # Channel 1: Opponent's pieces
        relative_board[1][board_state == (3 - self.player_id)] = 1.0
        
        # Add center bias to empty positions
        rows, cols = board_state.shape
        for row in range(rows):
            for col in range(cols):
                if board_state[row, col] == 0:  # Empty position
                    center_bonus = 0.1 * (1.0 - abs(col - 3) / 3.0)
                    relative_board[0, row, col] += center_bonus * 0.5
                    relative_board[1, row, col] += center_bonus * 0.5
        
        return relative_board.flatten()
    
    def _count_threats(self, board_state: np.ndarray, player_id: int) -> float:
        """Count immediate winning threats for a player."""
        threats = 0
        rows, cols = board_state.shape
        
        # Check each column for winning moves
        for col in range(cols):
            # Find the lowest empty row
            for row in range(rows - 1, -1, -1):
                if board_state[row, col] == 0:
                    # Temporarily place piece
                    temp_board = board_state.copy()
                    temp_board[row, col] = player_id
                    
                    # Check if this creates a win
                    if self._check_win_at_position(temp_board, row, col, player_id):
                        threats += 1
                    break
        
        return float(threats)
    
    def _count_center_pieces(self, board_state: np.ndarray, player_id: int) -> float:
        """Count pieces in center columns (3, 4)."""
        center_cols = [2, 3, 4]  # 0-indexed columns 2, 3, 4
        count = 0
        for col in center_cols:
            count += np.sum(board_state[:, col] == player_id)
        return float(count)
    
    def _max_connected_pieces(self, board_state: np.ndarray, player_id: int) -> float:
        """Find maximum number of connected pieces."""
        max_connected = 0
        rows, cols = board_state.shape
        
        # Check all directions: horizontal, vertical, diagonal
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for row in range(rows):
            for col in range(cols):
                if board_state[row, col] == player_id:
                    for dr, dc in directions:
                        connected = 1
                        # Check positive direction
                        r, c = row + dr, col + dc
                        while 0 <= r < rows and 0 <= c < cols and board_state[r, c] == player_id:
                            connected += 1
                            r, c = r + dr, c + dc
                        
                        # Check negative direction
                        r, c = row - dr, col - dc
                        while 0 <= r < rows and 0 <= c < cols and board_state[r, c] == player_id:
                            connected += 1
                            r, c = r - dr, c - dc
                        
                        max_connected = max(max_connected, connected)
        
        return float(max_connected)
    
    def _calculate_height_advantage(self, board_state: np.ndarray, player_id: int) -> float:
        """Calculate height advantage (higher pieces are often better)."""
        player_height = opponent_height = 0
        rows, cols = board_state.shape
        
        for col in range(cols):
            for row in range(rows):
                if board_state[row, col] == player_id:
                    player_height += (rows - row)
                elif board_state[row, col] == (3 - player_id):
                    opponent_height += (rows - row)
        
        return float(player_height - opponent_height)
    
    def _get_legal_moves_from_state(self, board_state: np.ndarray) -> List[int]:
        """Get legal moves from board state."""
        legal_moves = []
        rows, cols = board_state.shape
        for col in range(cols):
            if board_state[0, col] == 0:  # Top row is empty
                legal_moves.append(col)
        return legal_moves
    
    def _check_win_at_position(self, board_state: np.ndarray, row: int, col: int, player_id: int) -> bool:
        """Check if placing a piece at (row, col) creates a win."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        rows, cols = board_state.shape
        
        for dr, dc in directions:
            count = 1
            
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < rows and 0 <= c < cols and board_state[r, c] == player_id:
                count += 1
                r, c = r + dr, c + dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < rows and 0 <= c < cols and board_state[r, c] == player_id:
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 4:
                return True
        
        return False
    
    def encode_state(self, board_state: np.ndarray) -> np.ndarray:
        """Main state encoding method - uses enhanced features."""
        return self.encode_state_with_features(board_state)
    
    def choose_action(self, board_state: np.ndarray, legal_moves: List[int]) -> int:
        """Choose action using epsilon-greedy with enhanced features."""
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Encode state with strategic features
        encoded_state = self.encode_state(board_state)
        
        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_moves)
        
        # Get Q-values from online network
        state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            q_values = torch.clamp(q_values, min=-100, max=100)  # Prevent explosion during action selection
            q_values = q_values.cpu().numpy()[0]
        
        # Mask illegal moves
        masked_q_values = q_values.copy()
        for col in range(self.action_size):
            if col not in legal_moves:
                masked_q_values[col] = float('-inf')
        
        return int(np.argmax(masked_q_values))
    
    def compute_n_step_return(self, experiences: List, gamma: float) -> Tuple[np.ndarray, float, np.ndarray, bool]:
        """Compute n-step return with proper bootstrapping."""
        if not experiences:
            return None
        
        # Calculate n-step return
        n_step_return = 0.0
        gamma_n = 1.0
        
        for exp in experiences:
            n_step_return += gamma_n * exp.reward
            gamma_n *= gamma
        
        # Add bootstrap value if not terminal
        first_state = experiences[0].state
        last_exp = experiences[-1]
        
        if not last_exp.done and last_exp.next_state is not None:
            # Bootstrap with target network value
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(last_exp.next_state).unsqueeze(0).to(self.device)
                next_q_values = self.target_net(next_state_tensor)
                next_q_values = torch.clamp(next_q_values, min=-100, max=100)
                max_next_q = next_q_values.max().item()
                max_next_q = max(-100.0, min(100.0, max_next_q))  # Double safety
                n_step_return += gamma_n * max_next_q
        
        # Clamp final n-step return
        n_step_return = max(-100.0, min(100.0, n_step_return))
        
        return first_state, n_step_return, last_exp.next_state, last_exp.done
    
    def observe(self, board_state: np.ndarray, action: int, reward: float,
                next_state: Optional[np.ndarray], done: bool) -> None:
        """Store experience with n-step learning and prioritized replay."""
        # Encode states
        encoded_state = self.encode_state(board_state)
        encoded_next_state = self.encode_state(next_state) if next_state is not None else None
        
        # Add to n-step buffer
        experience = Experience(encoded_state, action, reward, encoded_next_state, done)
        self.n_step_buffer.append(experience)
        
        # If we have n experiences or episode is done, process n-step return
        if len(self.n_step_buffer) >= self.n_step:
            # Compute n-step return for the first experience
            state, n_step_return, final_next_state, final_done = self.compute_n_step_return(
                list(self.n_step_buffer), self.gamma
            )
            
            # Store in main buffer
            n_step_experience = Experience(state, self.n_step_buffer[0].action, n_step_return,
                                         final_next_state, final_done)
            
            if self.use_prioritized_replay:
                # Calculate TD error for priority
                td_error = self._calculate_td_error(n_step_experience)
                self.memory.add(n_step_experience, td_error)
            else:
                self.memory.append(n_step_experience)
            
            # Remove the first experience from n-step buffer
            self.n_step_buffer.popleft()
        
        # If episode is done, process all remaining experiences in buffer
        if done:
            while len(self.n_step_buffer) > 0:
                state, n_step_return, final_next_state, final_done = self.compute_n_step_return(
                    list(self.n_step_buffer), self.gamma
                )
                
                n_step_experience = Experience(state, self.n_step_buffer[0].action, n_step_return,
                                             final_next_state, final_done)
                
                if self.use_prioritized_replay:
                    td_error = self._calculate_td_error(n_step_experience)
                    self.memory.add(n_step_experience, td_error)
                else:
                    self.memory.append(n_step_experience)
                
                self.n_step_buffer.popleft()
        
        # Train if we have enough experiences
        min_size = self.min_buffer_size if not self.use_prioritized_replay else self.min_buffer_size
        if len(self.memory) >= min_size:
            self._train_step()
    
    def _calculate_td_error(self, experience: Experience) -> float:
        """Calculate TD error for prioritized replay."""
        state = torch.FloatTensor(experience.state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([experience.action]).to(self.device)
        reward = experience.reward
        done = experience.done
        
        if experience.next_state is not None:
            next_state = torch.FloatTensor(experience.next_state).unsqueeze(0).to(self.device)
        else:
            next_state = torch.zeros_like(state).to(self.device)
        
        with torch.no_grad():
            current_q = self.online_net(state).gather(1, action.unsqueeze(1)).item()
            
            if not done and experience.next_state is not None:
                # Double DQN target
                next_q_online = self.online_net(next_state)
                next_action = next_q_online.argmax(1, keepdim=True)
                next_q_target = self.target_net(next_state).gather(1, next_action).item()
                target = reward + (self.gamma ** self.n_step) * next_q_target
            else:
                target = reward
        
        return abs(current_q - target)
    
    def _train_step(self) -> None:
        """Enhanced training step with prioritized replay and polyak averaging."""
        if self.use_prioritized_replay:
            # Sample from prioritized buffer
            beta = self.beta_scheduler(self.train_step_count)
            experiences, indices, weights = self.memory.sample(self.batch_size, beta)
            
            if not experiences:
                return
            
            states, actions, rewards, next_states, dones = zip(*experiences)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            # Regular sampling
            batch = self.rng.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            weights = torch.ones(self.batch_size).to(self.device)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Build next_states tensor
        next_states_array = np.array([
            ns if ns is not None else np.zeros(len(states[0]), dtype=np.float32)
            for ns in next_states
        ])
        next_states_t = torch.FloatTensor(next_states_array).to(self.device)
        
        # Current Q values - CLAMP IMMEDIATELY
        q_values_all = self.online_net(states)
        q_values_all = torch.clamp(q_values_all, min=-100, max=100)
        current_q = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target calculation
        with torch.no_grad():
            # Online network selects actions - CLAMP IMMEDIATELY
            next_q_online = self.online_net(next_states_t)
            next_q_online = torch.clamp(next_q_online, min=-100, max=100)
            next_actions = next_q_online.argmax(1, keepdim=True)
            
            # Target network evaluates actions - CLAMP IMMEDIATELY
            next_q_target = self.target_net(next_states_t)
            next_q_target = torch.clamp(next_q_target, min=-100, max=100)
            next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
            
            gamma_n = self.gamma ** self.n_step
            targets = rewards + (gamma_n * next_q_values * (1.0 - dones))
            targets = torch.clamp(targets, min=-100, max=100)
        
            # ✅ Optimism protection during bootstrap phase only
            if hasattr(self, "_episode_count") and self._episode_count < 2000:
                targets = torch.clamp(targets, min=0.0)
        
        # Compute loss with importance sampling weights
        td_errors = current_q - targets
        loss = (weights * F.smooth_l1_loss(current_q, targets, reduction='none')).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay and hasattr(self.memory, 'update_priorities'):
            td_errors_np = td_errors.detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors_np)
        
        self.train_step_count += 1
        
        # Polyak averaging for target network (every step)
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
            target_param.data.copy_(
                self.polyak_tau * online_param.data + (1.0 - self.polyak_tau) * target_param.data
            )
    
    def reset_episode(self) -> None:
        """Reset episode state and decay epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str) -> None:
        """Save the enhanced agent."""
        checkpoint = {
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'player_id': self.player_id,
            'epsilon': self.epsilon,
            'train_step_count': self.train_step_count,
            'n_step': self.n_step,
            'use_prioritized_replay': self.use_prioritized_replay
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str, keep_player_id: bool = True) -> None:
        """Load the enhanced agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if keep_player_id:
            self.player_id = checkpoint.get('player_id', self.player_id)
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.train_step_count = checkpoint.get('train_step_count', self.train_step_count)
    
    def get_stats(self) -> dict:
        """Get enhanced training statistics."""
        stats = {
            'training_steps': self.train_step_count,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'device': str(self.device),
            'n_step': self.n_step,
            'use_prioritized_replay': self.use_prioritized_replay
        }
        
        if hasattr(self.memory, 'size'):
            stats['buffer_usage'] = len(self.memory) / self.memory.capacity
        
        return stats


if __name__ == "__main__":
    # Quick test of enhanced agent
    print("Testing Enhanced Double DQN Agent...")
    
    # Test with larger state size due to strategic features
    enhanced_agent = EnhancedDoubleDQNAgent(player_id=1, state_size=92, seed=42)
    print(f"✓ Enhanced agent created: {enhanced_agent}")
    print(f"✓ Device: {enhanced_agent.device}")
    print(f"✓ Network architecture: Dueling with 512 hidden units")
    print(f"✓ Features: N-step={enhanced_agent.n_step}, Prioritized={enhanced_agent.use_prioritized_replay}")
    
    # Test state encoding with strategic features
    import numpy as np
    test_board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0]
    ])
    
    encoded = enhanced_agent.encode_state(test_board)
    print(f"✓ Enhanced state encoding shape: {encoded.shape} (includes strategic features)")
    
    # Test action selection
    legal_moves = [0, 1, 2, 3, 4, 5, 6]
    action = enhanced_agent.choose_action(test_board, legal_moves)
    print(f"✓ Action selected: {action}")
    
    # Test enhanced statistics
    stats = enhanced_agent.get_stats()
    print(f"✓ Enhanced stats: {stats}")
    
    print("Enhanced Double DQN Agent test completed successfully!")