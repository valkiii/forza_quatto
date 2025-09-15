# Milestone 1: Building the Foundation - Connect 4 Game Engine

*Learning reinforcement learning by building Connect 4 from scratch*

## The Journey Begins

When I set out to learn reinforcement learning, I knew I needed a project that was simple enough to understand completely, yet rich enough to explore different AI approaches. Connect 4 emerged as the perfect candidate—it's a game we all know, with clear rules and a finite state space, but complex enough that finding optimal strategies isn't trivial.

In this first milestone, I focused on building the foundation: a robust game engine that could serve as the testbed for various AI agents. The key insight here is that good RL experiments require solid, bug-free environments. If your game logic is flawed, your agents will learn the wrong things!

## Design Philosophy: Modularity First

I structured the codebase with clear separation of concerns:

```
forza_quattro/
├── game/board.py          # Pure game logic, no AI
├── agents/base_agent.py   # Agent interface
├── agents/random_agent.py # Baseline agent
├── train/                 # Training scripts
└── tests/                 # Unit tests
```

This modular approach means I can swap different agents easily, test components independently, and extend the system without breaking existing code.

## The Heart: Connect4Board Class

The `Connect4Board` class (`game/board.py:13-190`) handles all game mechanics:

```python
class Connect4Board:
    def __init__(self, rows: int = 6, cols: int = 7):
        self.board = np.zeros((rows, cols), dtype=int)
        self.current_player = 1
```

I represented the board as a NumPy array where 0=empty, 1=player 1, 2=player 2. This choice makes it easy to:
- Copy states efficiently for tree search algorithms (later milestones)
- Encode states as neural network inputs
- Visualize and debug the game state

### Key Methods That Enable RL

The board provides everything an RL agent needs:

1. **State observation**: `get_state()` returns the current board
2. **Action space**: `get_legal_moves()` lists valid columns
3. **Environment interaction**: `make_move(col)` executes actions
4. **Terminal detection**: `check_winner()` and `is_draw()` identify episode ends

The win detection logic (`game/board.py:78-116`) checks all four directions efficiently. I initially wrote nested loops, but the current implementation is readable and fast enough for our needs.

## Agent Architecture: Planning for the Future

The `BaseAgent` interface (`agents/base_agent.py`) defines the contract all agents must follow:

```python
@abstractmethod
def choose_action(self, board_state: np.ndarray, legal_moves: list) -> int:
    """Choose column to play given current state."""
    pass
```

This simple interface hides the complexity—whether it's a random agent, Q-learning, or deep neural network, they all just need to choose a column. The `observe()` method allows learning agents to update from experience while keeping stateless agents simple.

## First Agent: The Humble Random Player

The `RandomAgent` (`agents/random_agent.py`) might seem trivial, but it serves crucial roles:
- **Baseline performance**: How well must our RL agents perform to be useful?
- **Training opponent**: Provides varied gameplay for learning agents
- **Debugging aid**: Helps verify the game engine works correctly

## Second Agent: Strategic Heuristics

Before diving into machine learning, I implemented a `HeuristicAgent` (`agents/heuristic_agent.py`) that uses simple rules. This agent demonstrates how far you can get with basic strategy and serves as a stronger benchmark for our RL agents.

### The Strategy Hierarchy

The heuristic agent follows a clear priority system:

1. **Win immediately** (4-in-a-row completion)
2. **Block opponent wins** (prevent their 4-in-a-row)
3. **Create threats** (make 3-in-a-row)
4. **Prefer center columns** (better positioning)
5. **Random fallback**

### Unified Connection Detection

Initially, I wrote separate functions for finding winning moves vs threat moves. But then I realized both were doing the same thing—counting connections! This led to a much cleaner design:

```python
def _find_moves_by_connection_count(self, board, player, legal_moves, target_count):
    """Unified function that finds moves creating N-in-a-row for any player."""
    # Simulate each move and count resulting connections
    good_moves = []
    for col in legal_moves:
        # ... simulate move ...
        if connection_count >= target_count:
            good_moves.append(col)
    return good_moves
```

**The beauty of this approach:**
- **DRY principle**: One algorithm, multiple uses
- **Perspective flexibility**: Same function works for "our wins" vs "opponent threats"
- **Extensible**: Could easily add 2-in-a-row detection for even more strategic play

**Usage examples:**
```python
# Find our winning moves (4+ connections)
wins = self._find_moves_by_connection_count(board, self.player_id, moves, 4)

# Find opponent winning moves to block
blocks = self._find_moves_by_connection_count(board, self.opponent_id, moves, 4)

# Find our threat creation opportunities (3+ connections)  
threats = self._find_moves_by_connection_count(board, self.player_id, moves, 3)
```

### Performance Results

The heuristic agent dramatically outperforms random play:
- **Heuristic vs Random**: 97% win rate
- **Random vs Random**: ~50% (baseline)

This 47 percentage point improvement shows how much strategic thinking matters in Connect 4!

## Example Gameplay

Let me show you the system in action:

```python
from game.board import Connect4Board
from agents.random_agent import RandomAgent

board = Connect4Board()
player1 = RandomAgent(player_id=1, seed=42)
player2 = RandomAgent(player_id=2, seed=123)

# Game loop
while not board.is_terminal():
    current_player = player1 if board.current_player == 1 else player2
    legal_moves = board.get_legal_moves() 
    action = current_player.choose_action(board.get_state(), legal_moves)
    board.make_move(action)
    
    print(board.render())
    print()

winner = board.check_winner()
print(f"Winner: Player {winner}" if winner else "Draw!")
```

Output:
```
0 1 2 3 4 5 6
- - - - - - -
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . . X . .
. . O . X . .

... (game continues)

Winner: Player 1
```

## What I Learned

Building this foundation taught me several important lessons:

1. **Start simple**: I resisted the urge to add fancy features like graphical UI or complex heuristics. The simple text representation is perfect for debugging and logging.

2. **Test everything**: Game logic bugs are particularly nasty because they create subtle biases that ruin RL training. I spent time writing thorough tests (coming next!).

3. **Interfaces matter**: The clean agent interface makes it trivial to swap different AI approaches. This design choice will pay dividends as we add Q-learning and neural networks.

4. **NumPy is your friend**: Using NumPy arrays from the start makes everything else easier—state copying, batch processing, and later neural network integration.

## Next Steps: The Real Learning Begins

With our solid foundation in place, Milestone 2 will tackle the exciting part: implementing Q-learning from scratch! We'll need to:

- Design a state encoding scheme (full board vs. features?)
- Implement the Q-learning update rule
- Balance exploration vs. exploitation  
- Train against random opponents and evaluate performance

The training skeleton in `train/q_learning_train.py` already outlines the structure. The real challenge will be making Q-learning work well on Connect 4's large state space.

Stay tuned—the AI is about to get smart!