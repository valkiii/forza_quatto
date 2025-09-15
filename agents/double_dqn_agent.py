"""Double DQN agent for Connect 4 - Fixed version."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Optional
try:
    from .base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent


class DoubleDQNAgent(BaseAgent):
    """Double DQN agent that fixes overestimation bias in standard DQN.
    
    Uses two networks: online network for action selection, target network for Q-value estimation.
    This reduces overoptimistic value estimates that plague standard DQN.
    """
    
    def __init__(self, player_id: int, state_size: int = 84, action_size: int = 7,
                 hidden_size: int = 256, gamma: float = 0.95, lr: float = 5e-5,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: float = 0.9998,
                 batch_size: int = 64, buffer_size: int = 200000, min_buffer_size: int = 1000,
                 target_update_freq: int = 2000, seed: int = None):
        """Initialize Double DQN agent."""
        super().__init__(player_id, "Double-DQN")
        
        # Agent parameters
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size  # Store buffer_size for subclasses
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        
        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.online_net = self._build_network(hidden_size).to(self.device)
        self.target_net = self._build_network(hidden_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)
        
        # Experience replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Training tracking
        self.train_step_count = 0
        
        # Random number generator
        self.rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def _build_network(self, hidden_size: int) -> nn.Module:
        """Build the neural network architecture."""
        return nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size)
        )
    
    def encode_state(self, board_state: np.ndarray) -> np.ndarray:
        """Enhanced two-channel encoding with center bias for better strategic play."""
        # Ensure input is numpy array
        if not isinstance(board_state, np.ndarray):
            board_state = np.array(board_state)
        
        # Create 2-channel representation
        relative_board = np.zeros((2, *board_state.shape), dtype=np.float32)
        
        # Channel 0: Agent's pieces
        relative_board[0][board_state == self.player_id] = 1.0
        
        # Channel 1: Opponent's pieces  
        relative_board[1][board_state == (3 - self.player_id)] = 1.0
        
        # Add center bias to empty positions (Connect 4 strategy)
        rows, cols = board_state.shape
        for row in range(rows):
            for col in range(cols):
                if board_state[row, col] == 0:  # Empty position
                    # Center columns are more valuable
                    center_bonus = 0.1 * (1.0 - abs(col - 3) / 3.0)
                    # Add to both channels for consistency
                    relative_board[0, row, col] += center_bonus * 0.5
                    relative_board[1, row, col] += center_bonus * 0.5
        
        # Flatten for neural network input
        return relative_board.flatten()
    
    def choose_action(self, board_state: np.ndarray, legal_moves: List[int]) -> int:
        """Choose action using epsilon-greedy with Double DQN."""
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Encode state
        encoded_state = self.encode_state(board_state)
        
        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_moves)
        
        # Get Q-values from online network
        state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor).cpu().numpy()[0]
        
        # Mask illegal moves
        masked_q_values = q_values.copy()
        for col in range(self.action_size):
            if col not in legal_moves:
                masked_q_values[col] = float('-inf')
        
        return int(np.argmax(masked_q_values))
    
    def observe(self, board_state: np.ndarray, action: int, reward: float,
                next_state: Optional[np.ndarray], done: bool) -> None:
        """Store experience and train the network."""
        # Encode states
        encoded_state = self.encode_state(board_state)
        encoded_next_state = self.encode_state(next_state) if next_state is not None else None
        
        # Store experience
        self.memory.append((encoded_state, action, reward, encoded_next_state, done))
        
        # Train if we have enough experiences
        if len(self.memory) >= self.min_buffer_size:
            self._train_step()
    
    def _legal_mask_from_encoded(self, encoded_states: torch.Tensor) -> torch.Tensor:
        """Compute legal-action mask for a batch of encoded states.
        encoded_states: (B, state_size) where state_size = 2 * 6 * 7 = 84
        Returns: boolean tensor (B, action_size) where True indicates legal.
        """
        B = encoded_states.shape[0]
        # Reshape to (B, 2, 6, 7) then check if top row is occupied
        states_2d = encoded_states.view(B, 2, 6, 7)
        # Check if top row (row 0) has any pieces (either channel)
        top_row_occupied = (states_2d[:, :, 0, :].sum(dim=1) > 0)  # (B, 7)
        # Legal moves are where top row is NOT occupied
        legal_mask = ~top_row_occupied  # (B, 7), True where legal
        return legal_mask

    def _train_step(self) -> None:
        """Perform one training step using Double DQN with legal masking and Huber loss."""
        batch = self.rng.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)            # (B, state_size)
        actions = torch.LongTensor(np.array(actions)).to(self.device)           # (B,)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)          # (B,)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)              # (B,)

        # Build next_states tensor (zero vector for terminal nexts)
        next_states_array = np.array([
            ns if ns is not None else np.zeros(self.state_size, dtype=np.float32)
            for ns in next_states
        ])
        next_states_t = torch.FloatTensor(next_states_array).to(self.device)    # (B, state_size)

        # Current Q for actions taken
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Compute legal-action mask for next_states
        legal_mask = self._legal_mask_from_encoded(next_states_t)  # (B, action_size), bool
        large_neg = -1e9

        with torch.no_grad():
            # Online network selects best next action (masked)
            q_next_online = self.online_net(next_states_t)                          # (B, A)
            q_next_online_masked = q_next_online.masked_fill(~legal_mask, large_neg) 
            next_actions = q_next_online_masked.argmax(dim=1, keepdim=True)         # (B,1)

            # Target network evaluates selected actions (masked)
            q_next_target = self.target_net(next_states_t)                          # (B, A)
            q_next_target_masked = q_next_target.masked_fill(~legal_mask, large_neg)
            next_q_values = q_next_target_masked.gather(1, next_actions).squeeze(1) # (B,)

            # If done -> don't bootstrap (next_q_values will be multiplied by (1-done))
            targets = rewards + (self.gamma * next_q_values * (1.0 - dones))

        # Huber loss / Smooth L1
        loss = F.smooth_l1_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
    
    def reset_episode(self) -> None:
        """Reset episode state and decay epsilon."""
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str) -> None:
        """Save the agent."""
        checkpoint = {
            'online_net_state_dict': self.online_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'player_id': self.player_id,
            'epsilon': self.epsilon,
            'train_step_count': self.train_step_count
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str, keep_player_id: bool = True) -> None:
        """Load the agent.
        
        Args:
            filepath: Path to the checkpoint file
            keep_player_id: If False, preserve current player_id instead of loading from checkpoint
                           (useful for self-play opponents to maintain correct viewpoint)
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Preserve current player_id if requested (critical for self-play opponents)
        if keep_player_id:
            self.player_id = checkpoint.get('player_id', self.player_id)
        # else: keep current self.player_id unchanged
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.train_step_count = checkpoint.get('train_step_count', self.train_step_count)
    
    def clear_replay_buffer(self) -> None:
        """Clear the replay buffer for curriculum transitions."""
        self.memory.clear()
        
    def get_buffer_diversity_stats(self) -> dict:
        """Get statistics about buffer composition for monitoring."""
        return {
            'buffer_size': len(self.memory),
            'buffer_capacity': self.memory.maxlen,
            'buffer_usage': len(self.memory) / self.memory.maxlen if self.memory.maxlen > 0 else 0
        }
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'training_steps': self.train_step_count,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'device': str(self.device)
        }


if __name__ == "__main__":
    # Quick test when run directly
    print("Testing Double DQN Agent...")
    
    agent = DoubleDQNAgent(player_id=1, seed=42)
    print(f"✓ Agent created: {agent}")
    print(f"✓ Device: {agent.device}")
    
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
    
    encoded = agent.encode_state(test_board)
    print(f"✓ State encoding shape: {encoded.shape}")
    
    # Test action selection
    legal_moves = [0, 1, 2, 3, 4, 5, 6]
    action = agent.choose_action(test_board, legal_moves)
    print(f"✓ Action selected: {action}")
    
    # Test statistics
    stats = agent.get_stats()
    print(f"✓ Stats: {stats}")
    
    print("Double DQN Agent test completed successfully!")