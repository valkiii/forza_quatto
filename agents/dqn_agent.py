"""Deep Q-Network (DQN) agent implemented from scratch with PyTorch."""

import random
import pickle
from collections import deque, namedtuple
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base_agent import BaseAgent


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNetwork(nn.Module):
    """Deep Q-Network for Connect 4.
    
    Architecture:
    - Input: 42 values (6x7 board flattened) with values 0, 1, 2
    - Hidden layers: 3 fully connected layers with ReLU activation
    - Output: 7 Q-values (one for each column)
    """
    
    def __init__(self, input_size: int = 42, hidden_size: int = 128, output_size: int = 7):
        """Initialize the DQN.
        
        Args:
            input_size: Size of input state (42 for Connect 4 board)
            hidden_size: Size of hidden layers
            output_size: Number of actions (7 columns)
        """
        super(DQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        # Xavier initialization for better training stability
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ExperienceReplayBuffer:
    """Experience replay buffer for DQN training.
    
    Stores experiences and provides random sampling for training.
    This breaks temporal correlations and improves sample efficiency.
    """
    
    def __init__(self, capacity: int = 100000, seed: int = None):
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            seed: Random seed for sampling
        """
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: Optional[np.ndarray], done: bool) -> None:
        """Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state (None if terminal)
            done: Whether episode ended
        """
        experience = Experience(
            state=state.copy() if state is not None else None,
            action=action,
            reward=reward,
            next_state=next_state.copy() if next_state is not None else None,
            done=done
        )
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of randomly sampled experiences
        """
        return self.rng.sample(self.buffer, batch_size)
        
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with experience replay and target network.
    
    Key features:
    - Neural network for Q-value approximation
    - Experience replay buffer for sample efficiency
    - Target network for training stability
    - Epsilon-greedy exploration
    - Same relative state encoding as tabular Q-learning for fair comparison
    """
    
    def __init__(self, player_id: int, learning_rate: float = 0.001,
                 discount_factor: float = 0.99, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay: float = 0.995,
                 buffer_size: int = 100000, batch_size: int = 64,
                 target_update_freq: int = 1000, seed: int = None):
        """Initialize DQN agent.
        
        Args:
            player_id: Player ID (1 or 2)
            learning_rate: Learning rate for neural network
            discount_factor: Discount factor (gamma) for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_size: Experience replay buffer size
            batch_size: Mini-batch size for training
            target_update_freq: Frequency to update target network
            seed: Random seed for reproducibility
        """
        super().__init__(player_id, "DQN")
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNetwork().to(self.device)
        self.target_network = DQNetwork().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Experience replay
        self.replay_buffer = ExperienceReplayBuffer(buffer_size, seed)
        
        # Training tracking
        self.training_step = 0
        self.last_state = None
        self.last_action = None
        
        # Random generator
        self.rng = random.Random(seed)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
    def encode_state(self, board_state: np.ndarray) -> np.ndarray:
        """Encode board state for neural network input.
        
        Uses the same relative encoding as Q-learning agent for fair comparison.
        
        Args:
            board_state: Raw board state with actual player IDs
            
        Returns:
            Numpy array with relative encoding (agent=1, opponent=2, empty=0)
        """
        # Create relative board from agent's perspective
        relative_board = np.zeros_like(board_state, dtype=np.float32)
        
        # Agent's pieces become 1, opponent's become 2
        relative_board[board_state == self.player_id] = 1.0
        relative_board[board_state == (3 - self.player_id)] = 2.0
        
        # Flatten for neural network input
        return relative_board.flatten()
        
    def choose_action(self, board_state: np.ndarray, legal_moves: List[int]) -> int:
        """Choose action using epsilon-greedy strategy with neural network.
        
        Args:
            board_state: Current board state
            legal_moves: List of legal column indices
            
        Returns:
            Selected column index
        """
        if not legal_moves:
            raise ValueError("No legal moves available")
            
        # Store for learning
        self.last_state = self.encode_state(board_state)
        
        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            # Explore: random action
            action = self.rng.choice(legal_moves)
        else:
            # Exploit: use neural network to select best action
            state_tensor = torch.FloatTensor(self.last_state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                
            # Mask illegal actions by setting their Q-values to negative infinity
            q_values_numpy = q_values.cpu().numpy()[0]
            for col in range(7):
                if col not in legal_moves:
                    q_values_numpy[col] = float('-inf')
                    
            action = int(np.argmax(q_values_numpy))
            
        self.last_action = action
        return action
        
    def observe(self, board_state: np.ndarray, action: int, reward: float,
                next_state: Optional[np.ndarray], done: bool) -> None:
        """Store experience and train the network.
        
        Args:
            board_state: Previous board state
            action: Action that was taken
            reward: Reward received
            next_state: Resulting board state (None if terminal)
            done: Whether episode ended
        """
        if self.last_state is None:
            return  # No previous experience to learn from
            
        # Encode next state
        encoded_next_state = self.encode_state(next_state) if next_state is not None else None
        
        # Store experience in replay buffer
        self.replay_buffer.add(
            self.last_state, self.last_action, reward, encoded_next_state, done
        )
        
        # Train the network if we have enough experiences
        if len(self.replay_buffer) >= self.batch_size:
            self._train_network()
            
    def _train_network(self) -> None:
        """Train the neural network using experience replay."""
        # Sample batch of experiences
        experiences = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([
            exp.next_state if exp.next_state is not None else np.zeros(42)
            for exp in experiences
        ]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network (Double DQN: use main network for action selection)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
            
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update training step counter
        self.training_step += 1
        
        # Update target network periodically
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
    def reset_episode(self) -> None:
        """Reset episode-specific state and decay epsilon."""
        self.last_state = None
        self.last_action = None
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def get_q_value(self, board_state: np.ndarray, action: int) -> float:
        """Get Q-value for a specific state-action pair.
        
        Args:
            board_state: Board state
            action: Action (column index)
            
        Returns:
            Q-value for the state-action pair
        """
        encoded_state = self.encode_state(board_state)
        state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return float(q_values[0, action].cpu().numpy())
            
    def save(self, filepath: str) -> None:
        """Save the agent's neural network and parameters.
        
        Args:
            filepath: Path to save the agent
        """
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'player_id': self.player_id,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon': self.epsilon,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'training_step': self.training_step
        }
        torch.save(checkpoint, filepath)
        
    def load(self, filepath: str) -> None:
        """Load the agent's neural network and parameters.
        
        Args:
            filepath: Path to load the agent from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.player_id = checkpoint['player_id']
        self.learning_rate = checkpoint['learning_rate']
        self.discount_factor = checkpoint['discount_factor']
        self.epsilon = checkpoint['epsilon']
        self.epsilon_end = checkpoint['epsilon_end']
        self.epsilon_decay = checkpoint['epsilon_decay']
        self.training_step = checkpoint['training_step']
        
    def get_stats(self) -> dict:
        """Get training statistics.
        
        Returns:
            Dictionary with agent statistics
        """
        return {
            'training_steps': self.training_step,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'device': str(self.device)
        }