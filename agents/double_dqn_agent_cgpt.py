"""Double DQN agent for Connect 4 - Fixed version."""

import torch
import torch.nn as nn
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
                 hidden_size: int = 128, gamma: float = 0.99, lr: float = 1e-3,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.1, epsilon_decay: float = 0.995,
                 batch_size: int = 64, buffer_size: int = 50000, min_buffer_size: int = 1000,
                 target_update_freq: int = 1000, seed: int = None):
        """Initialize Double DQN agent."""
        super().__init__(player_id, "Double-DQN")
        
        # Agent parameters
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
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
            nn.Linear(hidden_size, self.action_size)
        )
    
    def encode_state(self, board_state: np.ndarray) -> np.ndarray:
        """Two-channel encoding: separate channels for agent and opponent pieces."""
        # Create 2-channel representation
        relative_board = np.zeros((2, *board_state.shape), dtype=np.float32)
        
        # Channel 0: Agent's pieces
        relative_board[0][board_state == self.player_id] = 1.0
        
        # Channel 1: Opponent's pieces  
        relative_board[1][board_state == (3 - self.player_id)] = 1.0
        
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
    
    def _train_step(self) -> None:
        """Perform one training step using Double DQN."""
        # Sample batch from replay buffer
        batch = self.rng.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor([
            next_state if next_state is not None else np.zeros(self.state_size)
            for next_state in next_states
        ]).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values from online network
        current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN target calculation
        with torch.no_grad():
            # Use online network to select actions
            next_q_online = self.online_net(next_states)
            next_actions = torch.argmax(next_q_online, dim=1)
            
            # Use target network to evaluate selected actions
            next_q_target = self.target_net(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute targets
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update training step count
        self.train_step_count += 1
        
        # Update target network periodically
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
    
    def load(self, filepath: str) -> None:
        """Load the agent."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.online_net.load_state_dict(checkpoint['online_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.player_id = checkpoint['player_id']
        self.epsilon = checkpoint['epsilon']
        self.train_step_count = checkpoint['train_step_count']
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'training_steps': self.train_step_count,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'device': str(self.device)
        }