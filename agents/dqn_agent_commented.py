"""Deep Q-Network (DQN) agent implemented from scratch with PyTorch.

This file contains detailed line-by-line comments explaining every aspect
of the DQN implementation for educational purposes.
"""

# Import standard Python libraries
import random                           # For random number generation (epsilon-greedy)
import pickle                           # For saving/loading Python objects (unused here)
from collections import deque, namedtuple  # deque for efficient buffer, namedtuple for Experience
from typing import List, Optional, Tuple   # Type hints for better code documentation

# Import numerical and ML libraries
import numpy as np                      # For numerical operations on game states
import torch                           # PyTorch main module
import torch.nn as nn                  # Neural network layers and functions
import torch.optim as optim            # Optimization algorithms (Adam optimizer)
import torch.nn.functional as F        # Functional interface (ReLU, MSE loss)

# Import our custom base class
from .base_agent import BaseAgent      # Common interface for all agents

# Create a named tuple to store experience transitions
# This makes code more readable than using raw tuples
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNetwork(nn.Module):
    """Deep Q-Network for Connect 4.
    
    This neural network takes a flattened Connect 4 board (42 numbers) as input
    and outputs Q-values for each of the 7 possible actions (columns).
    """
    
    def __init__(self, input_size: int = 42, hidden_size: int = 128, output_size: int = 7):
        # Call parent class constructor (required for all PyTorch modules)
        super(DQNetwork, self).__init__()
        
        # Define the neural network layers
        # fc = "fully connected" layer (also called Linear layer)
        self.fc1 = nn.Linear(input_size, hidden_size)     # Input layer: 42 → 128 neurons
        self.fc2 = nn.Linear(hidden_size, hidden_size)    # Hidden layer 1: 128 → 128 neurons
        self.fc3 = nn.Linear(hidden_size, hidden_size)    # Hidden layer 2: 128 → 128 neurons
        self.fc4 = nn.Linear(hidden_size, output_size)    # Output layer: 128 → 7 neurons
        
        # Initialize weights using Xavier uniform distribution
        # This helps with gradient flow and training stability
        nn.init.xavier_uniform_(self.fc1.weight)          # Initialize first layer weights
        nn.init.xavier_uniform_(self.fc2.weight)          # Initialize second layer weights
        nn.init.xavier_uniform_(self.fc3.weight)          # Initialize third layer weights
        nn.init.xavier_uniform_(self.fc4.weight)          # Initialize output layer weights
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        This defines how data flows through the network when we call network(input).
        """
        # Pass input through first layer and apply ReLU activation
        x = F.relu(self.fc1(x))                           # Apply: ReLU(input * weights1 + bias1)
        
        # Pass through second layer with ReLU activation
        x = F.relu(self.fc2(x))                           # Apply: ReLU(x * weights2 + bias2)
        
        # Pass through third layer with ReLU activation
        x = F.relu(self.fc3(x))                           # Apply: ReLU(x * weights3 + bias3)
        
        # Output layer - no activation function (raw Q-values)
        x = self.fc4(x)                                   # Apply: x * weights4 + bias4
        
        return x                                          # Return Q-values for each action


class ExperienceReplayBuffer:
    """Experience replay buffer for DQN training.
    
    Stores past experiences and allows random sampling for training.
    This breaks the temporal correlation between consecutive game states.
    """
    
    def __init__(self, capacity: int = 100000, seed: int = None):
        # Create a deque (double-ended queue) with fixed maximum size
        # When capacity is reached, old experiences are automatically removed
        self.buffer = deque(maxlen=capacity)              # Store experiences with auto-removal
        
        # Create random number generator for sampling
        self.rng = random.Random(seed)                    # Seeded RNG for reproducible sampling
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: Optional[np.ndarray], done: bool) -> None:
        """Add an experience to the buffer."""
        
        # Create an Experience namedtuple with all transition information
        experience = Experience(
            # Copy state arrays to avoid reference issues
            state=state.copy() if state is not None else None,       # Current state (before action)
            action=action,                                           # Action taken (0-6 for columns)
            reward=reward,                                           # Reward received (usually 0, ±10, +1)
            next_state=next_state.copy() if next_state is not None else None,  # State after action
            done=done                                                # True if episode ended
        )
        
        # Add experience to buffer (oldest will be removed if buffer is full)
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences randomly from the buffer."""
        # Use random sampling to break temporal correlations
        return self.rng.sample(self.buffer, batch_size)   # Return random batch of experiences
        
    def __len__(self) -> int:
        """Get current number of experiences in buffer."""
        return len(self.buffer)                           # Return current buffer size


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with experience replay and target network."""
    
    def __init__(self, player_id: int, learning_rate: float = 0.001,
                 discount_factor: float = 0.99, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay: float = 0.995,
                 buffer_size: int = 100000, batch_size: int = 64,
                 target_update_freq: int = 1000, seed: int = None):
        
        # Call parent constructor to set player_id and name
        super().__init__(player_id, "DQN")                # Set agent name to "DQN"
        
        # Store hyperparameters as instance variables
        self.learning_rate = learning_rate                # How fast the network learns (0.001)
        self.discount_factor = discount_factor            # How much we value future rewards (0.99)
        self.epsilon = epsilon_start                      # Current exploration rate (starts at 1.0)
        self.epsilon_end = epsilon_end                    # Minimum exploration rate (0.05)
        self.epsilon_decay = epsilon_decay                # How fast exploration decreases (0.995)
        self.batch_size = batch_size                      # Number of experiences per training step (64)
        self.target_update_freq = target_update_freq      # How often to update target network (1000)
        
        # Setup neural networks and device
        # Check if CUDA (GPU) is available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create main Q-network and move it to device (CPU/GPU)
        self.q_network = DQNetwork().to(self.device)      # Main network for action selection
        
        # Create target Q-network and move it to device
        self.target_network = DQNetwork().to(self.device) # Target network for stable training
        
        # Create Adam optimizer for training the main network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize target network with same weights as main network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Create experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(buffer_size, seed)
        
        # Training tracking variables
        self.training_step = 0                            # Count of training steps performed
        self.last_state = None                            # Previous state for learning
        self.last_action = None                           # Previous action for learning
        
        # Setup random number generator for action selection
        self.rng = random.Random(seed)                    # Seeded RNG for reproducible exploration
        
        # Set random seeds for reproducibility if seed provided
        if seed is not None:
            torch.manual_seed(seed)                       # Set PyTorch random seed
            np.random.seed(seed)                          # Set NumPy random seed
            
    def encode_state(self, board_state: np.ndarray) -> np.ndarray:
        """Encode board state for neural network input.
        
        Converts the raw board state to a relative encoding from agent's perspective.
        """
        # Create new array with same shape as input, but using float32 for neural network
        relative_board = np.zeros_like(board_state, dtype=np.float32)
        
        # Convert to agent's perspective: agent's pieces = 1.0, opponent's pieces = 2.0
        relative_board[board_state == self.player_id] = 1.0           # Agent's pieces become 1.0
        relative_board[board_state == (3 - self.player_id)] = 2.0     # Opponent's pieces become 2.0
        # Empty cells remain 0.0
        
        # Flatten from 6x7 matrix to 42-element vector for neural network input
        return relative_board.flatten()                   # Convert to 1D array for network
        
    def choose_action(self, board_state: np.ndarray, legal_moves: List[int]) -> int:
        """Choose action using epsilon-greedy strategy with neural network."""
        
        # Safety check - should never happen in Connect 4
        if not legal_moves:
            raise ValueError("No legal moves available")
            
        # Encode current state for later use in learning
        self.last_state = self.encode_state(board_state)  # Store encoded state for learning
        
        # Epsilon-greedy exploration vs exploitation
        if self.rng.random() < self.epsilon:
            # EXPLORE: Choose random action from legal moves
            action = self.rng.choice(legal_moves)          # Random exploration
        else:
            # EXPLOIT: Use neural network to choose best action
            
            # Convert state to PyTorch tensor and add batch dimension
            state_tensor = torch.FloatTensor(self.last_state).unsqueeze(0).to(self.device)
            
            # Get Q-values from neural network (disable gradient computation for speed)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)    # Forward pass through network
                
            # Convert Q-values back to NumPy for processing
            q_values_numpy = q_values.cpu().numpy()[0]    # Remove batch dimension and move to CPU
            
            # Mask illegal actions by setting their Q-values to negative infinity
            for col in range(7):                          # For each possible column
                if col not in legal_moves:                # If this column is not legal
                    q_values_numpy[col] = float('-inf')   # Set Q-value to -infinity
                    
            # Choose action with highest Q-value (argmax)
            action = int(np.argmax(q_values_numpy))       # Select best legal action
            
        # Store action for learning
        self.last_action = action                         # Remember action for experience replay
        return action                                     # Return chosen action
        
    def observe(self, board_state: np.ndarray, action: int, reward: float,
                next_state: Optional[np.ndarray], done: bool) -> None:
        """Store experience and train the network."""
        
        # Check if we have previous state to learn from
        if self.last_state is None:
            return  # No previous experience to learn from (first move of episode)
            
        # Encode next state for storage (None if terminal state)
        encoded_next_state = self.encode_state(next_state) if next_state is not None else None
        
        # Store experience in replay buffer for later training
        self.replay_buffer.add(
            self.last_state,                              # Previous state (encoded)
            self.last_action,                             # Action taken from previous state
            reward,                                       # Reward received
            encoded_next_state,                           # Resulting state (encoded)
            done                                          # Whether episode ended
        )
        
        # Train network if we have enough experiences in buffer
        if len(self.replay_buffer) >= self.batch_size:   # Only train when buffer has enough data
            self._train_network()                         # Perform one training step
            
    def _train_network(self) -> None:
        """Train the neural network using experience replay."""
        
        # Sample random batch of experiences from replay buffer
        experiences = self.replay_buffer.sample(self.batch_size)  # Get random batch
        
        # Convert experiences to PyTorch tensors for training
        # Extract states from experiences and convert to tensor
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        
        # Extract actions and convert to long tensor (required for indexing)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        
        # Extract rewards and convert to tensor
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        
        # Extract next states, using zeros for terminal states
        next_states = torch.FloatTensor([
            exp.next_state if exp.next_state is not None else np.zeros(42)  # Use zeros if terminal
            for exp in experiences
        ]).to(self.device)
        
        # Extract done flags and convert to boolean tensor
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)
        
        # Compute current Q-values for the actions that were taken
        # q_network(states) gives Q-values for all actions
        # .gather(1, actions.unsqueeze(1)) selects Q-values for actions that were actually taken
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values using target network (for stability)
        with torch.no_grad():  # Don't compute gradients for target network
            # Get maximum Q-value for next states from target network
            next_q_values = self.target_network(next_states).max(1)[0]  # Max Q-value for each state
            
            # Compute target using Bellman equation: r + γ * max(Q(s', a'))
            # Use ~dones to zero out future rewards for terminal states
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
            
        # Compute loss between current Q-values and target Q-values
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)  # Mean squared error loss
        
        # Perform gradient descent step
        self.optimizer.zero_grad()                        # Clear previous gradients
        loss.backward()                                   # Compute gradients via backpropagation
        
        # Clip gradients to prevent exploding gradient problem
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        # Update network weights
        self.optimizer.step()                             # Apply gradients to update weights
        
        # Increment training step counter
        self.training_step += 1                           # Track number of training steps
        
        # Update target network periodically for stability
        if self.training_step % self.target_update_freq == 0:  # Every 1000 steps
            # Copy main network weights to target network
            self.target_network.load_state_dict(self.q_network.state_dict())
            
    def reset_episode(self) -> None:
        """Reset episode-specific state and decay epsilon."""
        # Clear episode-specific state
        self.last_state = None                            # Reset last state
        self.last_action = None                           # Reset last action
        
        # Decay epsilon (reduce exploration over time)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def get_q_value(self, board_state: np.ndarray, action: int) -> float:
        """Get Q-value for a specific state-action pair."""
        # Encode state for neural network
        encoded_state = self.encode_state(board_state)    # Convert to relative encoding
        
        # Convert to tensor and add batch dimension
        state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(self.device)
        
        # Get Q-values from network
        with torch.no_grad():  # No gradients needed for evaluation
            q_values = self.q_network(state_tensor)       # Forward pass
            # Extract Q-value for specific action and convert to Python float
            return float(q_values[0, action].cpu().numpy())
            
    def save(self, filepath: str) -> None:
        """Save the agent's neural network and parameters."""
        # Create checkpoint dictionary with all necessary information
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),        # Main network weights
            'target_network_state_dict': self.target_network.state_dict(),  # Target network weights
            'optimizer_state_dict': self.optimizer.state_dict(),        # Optimizer state
            'player_id': self.player_id,                                # Agent's player ID
            'learning_rate': self.learning_rate,                        # Learning rate
            'discount_factor': self.discount_factor,                    # Discount factor
            'epsilon': self.epsilon,                                    # Current epsilon
            'epsilon_end': self.epsilon_end,                            # Minimum epsilon
            'epsilon_decay': self.epsilon_decay,                        # Epsilon decay rate
            'training_step': self.training_step                         # Training progress
        }
        
        # Save checkpoint to file
        torch.save(checkpoint, filepath)                  # Use PyTorch's save function
        
    def load(self, filepath: str) -> None:
        """Load the agent's neural network and parameters."""
        # Load checkpoint from file
        checkpoint = torch.load(filepath, map_location=self.device)  # Load to correct device
        
        # Restore network weights
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore hyperparameters and training state
        self.player_id = checkpoint['player_id']                     # Restore player ID
        self.learning_rate = checkpoint['learning_rate']             # Restore learning rate
        self.discount_factor = checkpoint['discount_factor']         # Restore discount factor
        self.epsilon = checkpoint['epsilon']                         # Restore current epsilon
        self.epsilon_end = checkpoint['epsilon_end']                 # Restore minimum epsilon
        self.epsilon_decay = checkpoint['epsilon_decay']             # Restore epsilon decay
        self.training_step = checkpoint['training_step']             # Restore training progress
        
    def get_stats(self) -> dict:
        """Get training statistics."""
        # Return dictionary with current training statistics
        return {
            'training_steps': self.training_step,                    # Number of training steps
            'epsilon': self.epsilon,                                 # Current exploration rate
            'buffer_size': len(self.replay_buffer),                  # Replay buffer size
            'learning_rate': self.learning_rate,                     # Learning rate
            'discount_factor': self.discount_factor,                 # Discount factor
            'device': str(self.device)                               # Device (CPU/GPU)
        }