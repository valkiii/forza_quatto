"""CNN Dueling DQN agent with convolutional feature extraction for spatial learning."""

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

# Named tuple for storing experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class CNNDuelingNetwork(nn.Module):
    """CNN-based Dueling network that processes board as spatial 2D data."""
    
    def __init__(self, input_channels: int = 2, action_size: int = 7, hidden_size: int = 256, architecture: str = "ultra_light"):
        super().__init__()
        
        # CRITICAL: Store action_size FIRST before any initialization
        self.action_size = action_size
        self.architecture = architecture
        
        if architecture == "m1_optimized":
            self._build_m1_optimized(input_channels, action_size)
        else:
            self._build_ultra_light(input_channels, action_size)
        
        # Initialize weights (now self.action_size is available)
        self._initialize_weights()
    
    def _build_ultra_light(self, input_channels: int, action_size: int):
        """Build ultra-lightweight architecture (~5.8k parameters)."""
        # CNN feature extraction for Connect4-specific patterns
        # Input: 2 channels (player, opponent) x 6 rows x 7 cols
        
        # Ultra-lightweight pattern detection (~10k parameters total)
        self.basic_conv = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),  # 8 x 6 x 7
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 16 x 6 x 7
            nn.ReLU()
        )
        
        # Connect4-specific pattern detectors (minimal but focused)
        # Vertical 4-in-a-row detection
        self.vertical_conv = nn.Conv2d(16, 4, kernel_size=(4, 1), padding=0)  # 4 x 3 x 7
        
        # Horizontal 4-in-a-row detection  
        self.horizontal_conv = nn.Conv2d(16, 4, kernel_size=(1, 4), padding=0)  # 4 x 6 x 4
        
        # Diagonal pattern detection (minimal channels)
        self.diagonal_conv = nn.Conv2d(16, 8, kernel_size=4, padding=0)  # 8 x 3 x 4
        
        # Simple global pooling to reduce dimensions dramatically
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Reduce to 1x1 per channel
        
        # Calculate total channels after pattern detection: 16 + 4 + 4 + 8 = 32
        # After global pooling: 32 * 1 * 1 = 32 features
        self.conv_output_size = 32
        
        # Ultra-lightweight shared feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(self.conv_output_size, 32),  # 32 -> 32 features
            nn.ReLU(),
            nn.Linear(32, 16),  # Compress to 16 features  
            nn.ReLU()
        )
        
        # Ultra-lightweight dueling heads
        self.value_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),  
            nn.Linear(8, action_size)
        )
        
    def _build_m1_optimized(self, input_channels: int, action_size: int):
        """Build M1-optimized architecture (~80k parameters) - balances performance with capability."""
        # M1 GPU handles 32-64 channels efficiently for this size target
        self.conv_block = nn.Sequential(
            # Layer 1: Basic feature detection (optimal for M1)
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),  # 32 x 6 x 7
            nn.ReLU(),
            nn.BatchNorm2d(32),  # M1's MPS backend handles BatchNorm efficiently
            
            # Layer 2: Pattern combination
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 x 6 x 7
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        # Spatial reduction (but not global!) - Keep columns, reduce rows
        # This preserves critical Connect4 positional information
        self.spatial_reduce = nn.Conv2d(64, 32, kernel_size=(3, 1), stride=(2, 1))  # Output: 32 x 2 x 7
        
        # Flatten: 32 * 2 * 7 = 448 features (preserves column structure)
        self.conv_output_size = 448
        
        # Fully connected layers optimized for M1 (corrected input size)
        self.feature_fc = nn.Sequential(
            nn.Linear(448, 96),   # Corrected: 448 input features
            nn.ReLU(),
            nn.Dropout(0.1),  # Light dropout for generalization
            nn.Linear(96, 48),    # Reduced from 64 to 48
            nn.ReLU()
        )
        
        # Dueling heads (adjusted for new feature size)
        self.value_head = nn.Linear(48, 1)
        self.advantage_head = nn.Linear(48, action_size)
    
    def _initialize_weights(self):
        """Initialize weights with optimistic bias for output layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # Optimistic initialization for output layers
                    if m.out_features == 1 or m.out_features == self.action_size:
                        nn.init.constant_(m.bias, 5.0)  # Moderate optimism
                    else:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through Connect4 CNN dueling network."""
        # x shape: (batch_size, 2, 6, 7)
        
        if self.architecture == "m1_optimized":
            return self._forward_m1_optimized(x)
        else:
            return self._forward_ultra_light(x)
    
    def _forward_ultra_light(self, x):
        """Forward pass through ultra-lightweight architecture."""
        # Basic pattern extraction
        basic_features = self.basic_conv(x)  # (batch_size, 16, 6, 7)
        
        # Connect4-specific pattern detection
        vertical_features = self.vertical_conv(basic_features)  # (batch_size, 4, 3, 7)
        horizontal_features = self.horizontal_conv(basic_features)  # (batch_size, 4, 6, 4)  
        diagonal_features = self.diagonal_conv(basic_features)  # (batch_size, 8, 3, 4)
        
        # Global pooling to get fixed-size features (much more efficient)
        basic_pooled = self.global_pool(basic_features).squeeze(-1).squeeze(-1)  # (batch_size, 16)
        vertical_pooled = self.global_pool(vertical_features).squeeze(-1).squeeze(-1)  # (batch_size, 4)
        horizontal_pooled = self.global_pool(horizontal_features).squeeze(-1).squeeze(-1)  # (batch_size, 4)
        diagonal_pooled = self.global_pool(diagonal_features).squeeze(-1).squeeze(-1)  # (batch_size, 8)
        
        # Concatenate all pooled features
        conv_flat = torch.cat([
            basic_pooled,      # 16 features
            vertical_pooled,   # 4 features
            horizontal_pooled, # 4 features  
            diagonal_pooled    # 8 features
        ], dim=1)  # Total: 32 features
        
        # Shared feature processing
        features = self.feature_fc(conv_flat)  # (batch_size, 16)
        
        # Dueling heads
        value = self.value_head(features)  # (batch_size, 1)
        advantage = self.advantage_head(features)  # (batch_size, action_size)
        
        # Combine using dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
    
    def _forward_m1_optimized(self, x):
        """Forward pass through M1-optimized architecture."""
        # Convolutional feature extraction
        conv_features = self.conv_block(x)  # (batch_size, 64, 6, 7)
        
        # Spatial reduction - preserves critical Connect4 positional information
        reduced_features = self.spatial_reduce(conv_features)  # (batch_size, 32, 2, 7)
        
        # Flatten while preserving column structure
        conv_flat = reduced_features.view(-1, self.conv_output_size)  # (batch_size, 448)
        
        # Shared feature processing
        features = self.feature_fc(conv_flat)  # (batch_size, 48)
        
        # Dueling heads
        value = self.value_head(features)  # (batch_size, 1)
        advantage = self.advantage_head(features)  # (batch_size, action_size)
        
        # Combine using dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class CNNDuelingDQNAgent(BaseAgent):
    """CNN-based Dueling Double DQN agent for spatial pattern learning."""
    
    def __init__(self, player_id: int, input_channels: int = 2, action_size: int = 7,
                 hidden_size: int = 256, gamma: float = 0.95, lr: float = 1e-4,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: float = 0.99999001,
                 batch_size: int = 128, buffer_size: int = 100000, min_buffer_size: int = 1000,
                 target_update_freq: int = 1000, architecture: str = "ultra_light", seed: int = None):
        """Initialize CNN Dueling DQN agent."""
        super().__init__(player_id, "CNN-Dueling-DQN")
        
        # Agent parameters
        self.input_channels = input_channels
        self.action_size = action_size
        self.architecture = architecture
        
        # Hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device - prioritize M1 GPU (MPS) for CNN acceleration
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Networks with architecture selection
        self.online_net = CNNDuelingNetwork(input_channels, action_size, hidden_size, architecture).to(self.device)
        self.target_net = CNNDuelingNetwork(input_channels, action_size, hidden_size, architecture).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with weight decay for CNN
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Training tracking
        self.train_step_count = 0
        self._episode_count = 0
        
        # Random number generator
        self.rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def encode_state_cnn(self, board_state: np.ndarray) -> np.ndarray:
        """Encode board state for CNN processing (2-channel spatial format)."""
        if not isinstance(board_state, np.ndarray):
            board_state = np.array(board_state)
        
        # Create 2-channel representation for CNN
        # Channel 0: Current player's pieces
        # Channel 1: Opponent's pieces
        channels = np.zeros((2, *board_state.shape), dtype=np.float32)
        
        # Fill channels
        channels[0][board_state == self.player_id] = 1.0
        channels[1][board_state == (3 - self.player_id)] = 1.0
        
        return channels
    
    def encode_state(self, board_state: np.ndarray) -> np.ndarray:
        """Main state encoding method for compatibility with TrainingMonitor."""
        return self.encode_state_cnn(board_state)
    
    def choose_action(self, board_state: np.ndarray, legal_moves: List[int]) -> int:
        """Choose action using epsilon-greedy with CNN feature extraction."""
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_moves)
        
        # Encode state for CNN
        encoded_state = self.encode_state_cnn(board_state)
        state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(self.device)
        
        # Get Q-values from online network
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
            q_values = torch.clamp(q_values, min=-100, max=100)
            q_values = q_values.cpu().numpy()[0]
        
        # Mask illegal moves
        masked_q_values = q_values.copy()
        for col in range(self.action_size):
            if col not in legal_moves:
                masked_q_values[col] = float('-inf')
        
        return int(np.argmax(masked_q_values))
    
    def observe(self, board_state: np.ndarray, action: int, reward: float,
                next_state: Optional[np.ndarray], done: bool) -> None:
        """Store experience and train."""
        # Encode states for CNN
        encoded_state = self.encode_state_cnn(board_state)
        encoded_next_state = self.encode_state_cnn(next_state) if next_state is not None else None
        
        # Store experience
        experience = Experience(encoded_state, action, reward, encoded_next_state, done)
        self.memory.append(experience)
        
        # Train if enough experiences
        if len(self.memory) >= self.min_buffer_size:
            self._train_step()
    
    def _train_step(self) -> None:
        """Training step with CNN features."""
        # Sample batch
        batch = self.rng.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        # Handle next states (some may be None)
        next_states_array = np.array([
            ns if ns is not None else np.zeros((2, 6, 7), dtype=np.float32)
            for ns in next_states
        ])
        next_states_t = torch.FloatTensor(next_states_array).to(self.device)
        
        # Current Q values
        q_values_all = self.online_net(states)
        q_values_all = torch.clamp(q_values_all, min=-100, max=100)
        current_q = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target calculation
        with torch.no_grad():
            # Online network selects actions
            next_q_online = self.online_net(next_states_t)
            next_q_online = torch.clamp(next_q_online, min=-100, max=100)
            next_actions = next_q_online.argmax(1, keepdim=True)
            
            # Target network evaluates actions
            next_q_target = self.target_net(next_states_t)
            next_q_target = torch.clamp(next_q_target, min=-100, max=100)
            next_q_values = next_q_target.gather(1, next_actions).squeeze(1)
            
            targets = rewards + (self.gamma * next_q_values * (1.0 - dones))
            targets = torch.clamp(targets, min=-100, max=100)
            
            # Optimism protection during early training
            if hasattr(self, "_episode_count") and self._episode_count < 1000:
                targets = torch.clamp(targets, min=0.0)
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q, targets)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.train_step_count += 1
        
        # Update target network
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
    
    def reset_episode(self) -> None:
        """Reset episode state and decay epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._episode_count += 1
    
    def save(self, filepath: str) -> None:
        """Save the CNN agent."""
        checkpoint = {
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'player_id': self.player_id,
            'epsilon': self.epsilon,
            'train_step_count': self.train_step_count,
            '_episode_count': self._episode_count
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str, keep_player_id: bool = True) -> None:
        """Load the CNN agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if keep_player_id:
            self.player_id = checkpoint.get('player_id', self.player_id)
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.train_step_count = checkpoint.get('train_step_count', self.train_step_count)
        self._episode_count = checkpoint.get('_episode_count', self._episode_count)
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'training_steps': self.train_step_count,
            'episode_count': self._episode_count,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'device': str(self.device),
            'network_type': 'CNN-Dueling'
        }


if __name__ == "__main__":
    # Quick test of CNN agent
    print("Testing CNN Dueling DQN Agent...")
    
    cnn_agent = CNNDuelingDQNAgent(player_id=1, seed=42)
    print(f"✓ CNN agent created: {cnn_agent}")
    print(f"✓ Device: {cnn_agent.device}")
    print(f"✓ Network type: CNN with Dueling architecture")
    
    # Test state encoding
    import numpy as np
    test_board = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0]
    ])
    
    encoded = cnn_agent.encode_state_cnn(test_board)
    print(f"✓ CNN state encoding shape: {encoded.shape} (2-channel spatial)")
    
    # Test action selection
    legal_moves = [0, 1, 2, 3, 4, 5, 6]
    action = cnn_agent.choose_action(test_board, legal_moves)
    print(f"✓ Action selected: {action}")
    
    # Test stats
    stats = cnn_agent.get_stats()
    print(f"✓ Stats: {stats}")
    
    print("CNN Dueling DQN Agent test completed successfully!")