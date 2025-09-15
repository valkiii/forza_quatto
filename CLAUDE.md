# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


📋 STEP 1: READ REQUIREMENTS
Claude, read the rules in u/CLAUDE.md, then use sequential thinking and proceed to the next step.
STOP. Before reading further, confirm you understand:
1. This is a code reuse and consolidation project
2. Creating new files requires exhaustive justification  
3. Every suggestion must reference existing code
4. Violations of these rules make your response invalid

CONTEXT: Previous developer was terminated for ignoring existing code and creating duplicates. You must prove you can work within existing architecture.

MANDATORY PROCESS:
1. Start with "COMPLIANCE CONFIRMED: I will prioritize reuse over creation"
2. Analyze existing code BEFORE suggesting anything new
3. Reference specific files from the provided analysis
4. Include validation checkpoints throughout your response
5. End with compliance confirmation

RULES (violating ANY invalidates your response):
❌ No new files without exhaustive reuse analysis
❌ No rewrites when refactoring is possible
❌ No generic advice - provide specific implementations
❌ No ignoring existing codebase architecture
✅ Extend existing services and components
✅ Consolidate duplicate code
✅ Reference specific file paths
✅ Provide migration strategies
✅ Create modular code

# Task
You are an expert Python developer and teacher. Help me build a small Connect 4 project to learn reinforcement learning. Produce modular, well-documented code that runs in the terminal, unit tests, training scripts for RL agents, evaluation scripts, and step-by-step blog-ready documentation for each development milestone. Emphasize clarity, reproducibility, and didactic explanations that I can paste into a blog post.

# High-level goals
1. Build a terminal-playable Connect 4 game with:
   - Clear game logic module, no UI framework required (text-based).
   - Pluggable agents (random / heuristic / RL).
2. Start with random agents, then implement RL agents **from scratch first**:
   - Tabular Q-Learning (implement all logic manually: table, updates, epsilon-greedy).
   - Deep Q-Network (DQN) built from scratch using basic PyTorch (manual replay buffer, target network).
   - Only after the scratch implementations, optionally show how to use existing libraries (e.g., stable-baselines3) to speed up development and compare results.
3. Provide training scripts, evaluation metrics, and a CLI to play against trained agents.
4. Create a blog post (or series) documenting design choices, theory, experiments, plots, and lessons learned for each milestone.
5. Keep files small and modular; each file < ~250 lines where reasonable.

# Deliverables (for each milestone)
- Code:
  - `game/board.py` - board state, legal moves, win/draw detection, simple text rendering.
  - `agents/base_agent.py` - agent interface (choose_action, observe / learn).
  - `agents/random_agent.py`, `agents/heuristic_agent.py`.
  - `agents/q_learning_agent.py` - tabular Q-learning (scratch).
  - `agents/dqn_agent.py` - DQN (scratch with PyTorch).
  - `agents/dqn_agent_lib.py` - (optional) refactor using an RL library like stable-baselines3 for speed/comparison.
  - `train/*.py` - training pipelines for each RL approach with hyperparameters exposed.
  - `eval/evaluate.py` - run tournaments, compute win rates, learning curves.
  - `cli/play.py` - CLI to play against any trained agent.
  - `tests/test_board.py`, `tests/test_agents.py` - unit tests for core behavior.
- Documentation:
  - `README.md` - project overview, install & run instructions.
  - `docs/milestone_01.md`, `docs/milestone_02.md`... - blog-ready writeups per milestone:
    * motivation and learning goals
    * design choices and code walk-through
    * training procedure, hyperparameters, and results
    * plots and interpretation
    * "what I learned" short reflection
- Small example dataset or saved model files (if relevant) and scripts to reproduce the experiments.

# Constraints & style
- RL must first be implemented from scratch before showing any library shortcuts.
- Use Python 3.10+ style; prefer PyTorch for DQN. Keep dependencies minimal: `numpy`, `pytest`, `matplotlib`, `torch` (optional for DQN), `stable-baselines3` (optional for later milestone).
- Keep code modular and small functions/classes. Aim for single responsibility.
- Include type hints and docstrings for all public functions/classes.
- Logging: simple CSV/JSON logger of episode rewards and evaluation metrics. Save checkpoints.
- Random seeds: ensure reproducibility (document seed usage).
- Tests: write unit tests covering board logic, move legality, win detection, and basic agent API.
- CLI: provide simple commands and examples (e.g., `python cli/play.py --agent dqn --model models/dqn_best.pt`).
- Notebook/plots: provide script to generate training curves that can be included in blog posts (PNG outputs).

# Milestone plan (concrete, iterative)
1. Milestone 1 — Core game + random agents
   - Implement board and legal moves, text rendering.
   - Implement random and a simple heuristic agent.
   - Add unit tests for game logic.
   - Doc: `docs/milestone_01.md` — design, example gameplay transcript.
2. Milestone 2 — Tabular Q-learning agent (from scratch)
   - Design a compact state encoding or feature extractor (explain tradeoffs).
   - Implement tabular Q-learning training and evaluation scripts.
   - Run experiments vs random/heuristic agents and report metrics (win %).
   - Doc: `docs/milestone_02.md` — algorithm explanation, code walk-through, results.
3. Milestone 3 — DQN agent (scratch first, then library comparison)
   - Implement minimal DQN (PyTorch): replay buffer, epsilon-greedy, small NN, target network.
   - Training script with checkpoints and evaluation.
   - Compare DQN vs tabular Q and heuristic.
   - Optionally: re-implement DQN using a library (stable-baselines3) to show speed vs control tradeoff.
   - Doc: `docs/milestone_03.md` — architecture, hyperparameters, scratch vs library, training curves.
4. Milestone 4 — Playable CLI & blog polishing
   - Finalize CLI to play against any saved agent.
   - Proofread and transform docs into blog-style articles (storytelling voice, with code snippets and plots).
   - Add a `how-to-reproduce` section.

# What to produce in this reply
1. A compact file/folder skeleton (tree) for the entire project.
2. A complete `board.py` implementation (clean, tested) and `random_agent.py`.
3. A minimal `train/q_learning_train.py` sketch (train loop + logging).
4. `docs/milestone_01.md` blog-ready writeup draft (storytelling style, ~400–700 words) that explains the motivation and the code above and includes a gameplay example.
5. A short checklist of next steps (Milestone 2) with a recommended Q-learning hyperparameter set to try.

# Tone & formatting
- Use clear headings in Markdown where appropriate.
- When writing blog text, use a storytelling style (explain what you tried, what surprised you, what broke, what you learned).
- Write code in readable, idiomatic Python. Include inline comments for teachable moments.

# Additional helpful tips for me
- Always start with from-scratch RL implementations before introducing libraries.
- If a design choice has tradeoffs (e.g., state encoding size vs generalization), briefly explain pros/cons and recommend a default.
- When providing training scripts, prioritize clarity over performance — make it easy to read and modify.
- Include at least one simple command example for running training, evaluation, and CLI play.

# Short form (if you prefer minimal prompt)
You are an expert Python teacher. Build a small, modular Connect 4 project (terminal UI) that introduces RL: random baseline → tabular Q-learning (scratch) → DQN (scratch with PyTorch, then optional stable-baselines3). Deliver code modules, tests, training & eval scripts, and blog-ready docs for each milestone. Keep files small, documented, reproducible. Start by returning a project tree, `board.py`, `agents/random_agent.py`, a q-learning training skeleton, `docs/milestone_01.md`, and next-steps checklist.


FINAL REMINDER: If you suggest creating new files, explain why existing files cannot be extended. If you recommend rewrites, justify why refactoring won't work.
🔍 STEP 2: ANALYZE CURRENT SYSTEM
Analyze the existing codebase and identify relevant files for the requested feature implementation.
Then proceed to Step 3.
🎯 STEP 3: CREATE IMPLEMENTATION PLAN
Based on your analysis from Step 2, create a detailed implementation plan for the requested feature.
Then proceed to Step 4.
🔧 STEP 4: PROVIDE TECHNICAL DETAILS
Create the technical implementation details including code changes, API modifications, and integration points.
Then proceed to Step 5.
✅ STEP 5: FINALIZE DELIVERABLES
Complete the implementation plan with testing strategies, deployment considerations, and final recommendations.
🎯 INSTRUCTIONS
Follow each step sequentially. Complete one step before moving to the next. Use the findings from each previous step to inform the next step.


