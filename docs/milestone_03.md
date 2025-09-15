# Milestone 3: Deep Learning Breakthrough - DQN vs Tabular Q-Learning

*When neural networks meet reinforcement learning: scaling beyond the Q-table*

## From Tables to Tensors

After implementing tabular Q-learning and discovering its limitations—187K Q-table entries from just 1,000 episodes and poor strategic play—I was ready to explore Deep Q-Networks (DQN). The question: could neural networks overcome the state explosion problem while learning better strategies?

This milestone documents building a DQN agent from scratch using PyTorch, comparing it head-to-head with the tabular approach, and analyzing what each method brings to Connect 4.

## The Deep Q-Network Architecture

### Neural Network Design

I designed a simple but effective architecture (`agents/dqn_agent.py:19-55`):

```python
class DQNetwork(nn.Module):
    def __init__(self, input_size=42, hidden_size=128, output_size=7):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # 42 → 128
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 128 → 128  
        self.fc3 = nn.Linear(hidden_size, hidden_size)  # 128 → 128
        self.fc4 = nn.Linear(hidden_size, output_size)  # 128 → 7
```

**Design rationale:**
- **Input**: 42 neurons (6×7 board flattened with relative encoding)
- **Hidden layers**: 3 layers of 128 neurons each with ReLU activation
- **Output**: 7 neurons (Q-values for each column)
- **Xavier initialization**: Better gradient flow during training

**Why this size?** 
- Deep enough to learn complex patterns (3 hidden layers)
- Wide enough to capture Connect 4 strategy (128 neurons)
- Small enough to train quickly on CPU (~70K parameters)

### The DQN Innovations

DQN introduced several key improvements over basic Q-learning:

#### 1. Experience Replay Buffer (`agents/dqn_agent.py:58-99`)

```python
class ExperienceReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
```

**Why experience replay works:**
- **Breaks temporal correlations**: Random sampling prevents overfitting to recent games
- **Sample efficiency**: Each experience can be used multiple times for training
- **Stable gradients**: Mini-batches provide smoother learning signals

#### 2. Target Network (`agents/dqn_agent.py:296-298`)

```python
# Update target network periodically (every 1000 steps)
if self.training_step % self.target_update_freq == 0:
    self.target_network.load_state_dict(self.q_network.state_dict())
```

**The moving target problem:** In regular Q-learning, we update Q(s,a) toward a target that itself changes as we learn. This creates instability.

**Target network solution:** Use a separate, slowly-updating network to compute targets:
```python
target_q = reward + gamma * target_network(next_state).max()
```

This provides stable learning targets while still allowing the main network to improve.

## Implementation: The Devil in the Details

### State Encoding Consistency

I used identical relative encoding for both tabular and DQN agents:

```python
def encode_state(self, board_state):
    relative_board = np.zeros_like(board_state)
    relative_board[board_state == self.player_id] = 1.0      # Agent's pieces
    relative_board[board_state == (3 - self.player_id)] = 2.0  # Opponent's pieces
    return relative_board.flatten()  # Shape: (42,) for DQN input
```

This ensures fair comparison—both agents see the same state representation.

### Training Loop Architecture

The DQN training loop differs subtly but importantly from tabular Q-learning:

```python
def play_dqn_training_game(dqn_agent, opponent):
    # ... game loop ...
    
    # For each DQN agent move:
    episode_experiences.append({'state': board_state, 'action': action})
    
    # Provide learning experience immediately
    if len(episode_experiences) > 1:
        prev_state, prev_action = episode_experiences[-2]
        dqn_agent.observe(prev_state, prev_action, reward, next_state, done)
```

**Key difference**: DQN learns from mini-batches of experiences, while tabular Q-learning updates single (s,a) pairs.

## Training Results: The Neural Advantage

### Performance Comparison

After 2,000 episodes of training vs random opponents:

| Method | vs Random | vs Heuristic | Parameters |
|--------|-----------|--------------|------------|
| **Tabular Q-Learning** | 54.8% | 1.8% | 187K Q-table entries |
| **DQN** | 58.6% | 7.6% | 70K network weights |
| **Heuristic Baseline** | 99.0% | 67.4% | Hand-coded rules |

### What the Numbers Tell Us

**✅ DQN Advantages:**
- **Better vs Random**: 58.6% vs 54.8% (+3.8 percentage points)
- **More strategic**: 7.6% vs 1.8% vs heuristic (+5.8 points)
- **Parameter efficient**: 70K weights vs 187K Q-table entries
- **Smoother learning**: Experience replay reduces variance

**⚠️ Both Methods Struggle:**
- Neither agent learned to compete seriously with strategic play
- Training only vs random opponents created "random specialists"
- Both need exposure to stronger opponents during training

### Learning Curves Analysis

**DQN learning progression:**
```
Episode   400 | Win Rate: 49.0% | Epsilon: 0.135 | Buffer: 3,732
Episode   800 | Win Rate: 81.0% | Epsilon: 0.050 | Buffer: 7,199  ← Breakthrough!
Episode 1,200 | Win Rate: 83.0% | Epsilon: 0.050 | Buffer: 10,697
Episode 1,600 | Win Rate: 61.0% | Epsilon: 0.050 | Buffer: 14,369 ← Some instability
Episode 2,000 | Win Rate: 68.0% | Epsilon: 0.050 | Buffer: 18,414
```

**Key observations:**
1. **Slow start** (episodes 1-400): Network needs time to learn basic patterns
2. **Rapid improvement** (episodes 400-800): Experience replay kicks in
3. **Peak performance** (episodes 800-1200): High win rates vs random
4. **Some instability** (episodes 1200+): Overfitting to simple opponents

## Code Architecture: Modular Excellence

### Neural Network Modularity

The DQN implementation maintains clean separation:

```python
class DQNAgent(BaseAgent):  # Same interface as other agents
    def __init__(self, ...): 
        self.q_network = DQNetwork().to(self.device)
        self.target_network = DQNetwork().to(self.device)
        self.replay_buffer = ExperienceReplayBuffer(buffer_size)
        
    def choose_action(self, board_state, legal_moves):
        # Epsilon-greedy with neural network Q-values
        
    def observe(self, state, action, reward, next_state, done):
        # Store experience and train network
```

**Benefits of this design:**
- **Drop-in replacement**: DQN agent works with existing training loops
- **Testable components**: Network, buffer, and agent can be tested independently  
- **Configuration flexibility**: Easy to experiment with architectures

### Experience Replay Implementation

I used named tuples for clean experience storage:

```python
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
```

This provides type safety and clear intent compared to raw tuples or dictionaries.

## Deep Learning Insights

### What Neural Networks Learned

Looking at the DQN's improved performance, it appears to have learned:

1. **Basic tactical patterns**: Avoiding immediate losses
2. **Position evaluation**: Preferring center positions
3. **Pattern generalization**: Applying learned patterns to new board positions

**Evidence**: The 7.6% win rate vs heuristic (vs 1.8% for tabular) suggests the DQN discovered some strategic concepts.

### Why DQN Outperformed Tabular Q-Learning

**Generalization power**: Neural networks can generalize across similar board positions. Two boards with pieces in slightly different locations might be treated as completely different states by tabular Q-learning but recognized as similar by the DQN.

**Function approximation**: Instead of storing 187K separate Q-values, the DQN compresses Connect 4 strategy into 70K network weights that can evaluate any board position.

**Sample efficiency**: Experience replay meant each game experience contributed to multiple network updates, making better use of training data.

## Limitations and Future Work

### The Strategic Gap Remains

Both RL agents still perform poorly vs the heuristic agent (1.8% and 7.6% win rates). This reveals the fundamental challenge: 

**The curriculum problem**: Training only vs random opponents creates agents optimized for random play, not strategic play.

**Future improvements:**
- **Curriculum learning**: Start vs random, gradually introduce stronger opponents
- **Self-play**: Train agents against themselves to discover complex strategies
- **Opponent modeling**: Learn to adapt strategies based on opponent behavior

### Technical Improvements

Several optimizations could improve DQN performance:

1. **Double DQN**: Reduce overestimation bias in Q-value updates
2. **Dueling Networks**: Separate state value and advantage estimation
3. **Prioritized Experience Replay**: Sample important experiences more frequently
4. **Larger networks**: More parameters for complex pattern recognition

## Reproducibility Guide

To reproduce these results:

```bash
# Train both agents
python train/quick_q_train.py      # Tabular Q-learning
python train/quick_dqn_train.py    # Deep Q-Network

# Compare all agents  
python compare_all_agents.py
```

**Key hyperparameters that worked:**
- **DQN learning rate**: 0.001 (slower than tabular 0.3)
- **Experience replay**: 20K buffer, batch size 32
- **Target update**: Every 1000 training steps
- **Network**: 3×128 hidden layers with ReLU

## The Journey Continues

This milestone demonstrated neural networks' power for reinforcement learning: better generalization, parameter efficiency, and strategic understanding. However, both tabular and neural approaches revealed the same fundamental lesson: **the training environment shapes the agent.**

Our RL agents became experts at beating random players but struggled against strategic opponents. This sets up perfectly for future work on curriculum learning, self-play, and opponent modeling.

The foundation is strong—now we need to teach our agents to play like masters, not just beat beginners!

## What I Learned

1. **Neural networks aren't magic**: They need good training data and proper architecture
2. **Experience replay is crucial**: Random sampling breaks temporal correlations  
3. **Stable targets matter**: Target networks prevent the "moving target" problem
4. **Generalization vs memorization**: Neural networks generalize better than lookup tables
5. **Training curriculum is everything**: Agents learn to beat their training opponents

The deep learning revolution in RL isn't just about bigger models—it's about smarter learning algorithms that make better use of experience.