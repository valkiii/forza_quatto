import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# =============================
# Double DQN Agent for Connect4
# =============================
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, player_id, hidden_size=128, gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.9995,
                 batch_size=64, buffer_size=10000, min_buffer_size=500, target_update_freq=1000):
        self.state_size = state_size
        self.action_size = action_size
        self.player_id = player_id

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

        # Replay buffer
        self.memory = deque(maxlen=buffer_size)

        # Networks
        self.online_net = self.build_network(hidden_size)
        self.target_net = self.build_network(hidden_size)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)

        # Internal step counter for target network sync
        self.train_step_count = 0

    def build_network(self, hidden_size):
        return nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size)
        )

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, legal_moves):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.online_net(state_tensor).detach().numpy()[0]

        # Mask illegal moves by setting them to very negative values
        mask = np.full(self.action_size, -np.inf)
        mask[legal_moves] = 0
        masked_q_values = q_values + mask

        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_moves)
        else:
            return int(np.argmax(masked_q_values))

    def train_step(self):
        if len(self.memory) < self.min_buffer_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Online network prediction for current state
        q_values = self.online_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target calculation
        with torch.no_grad():
            next_q_online = self.online_net(next_states)
            next_actions = torch.argmax(next_q_online, dim=1)
            next_q_target = self.target_net(next_states)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Loss and optimization
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_model(self, path="double_dqn_connect4.pt"):
        torch.save(self.online_net.state_dict(), path)

    def load_model(self, path="double_dqn_connect4.pt"):
        self.online_net.load_state_dict(torch.load(path))
        self.online_net.eval()

# =============================
# Example Training Loop
# =============================
def train_agent(env, num_episodes=10000):
    agent = DoubleDQNAgent(state_size=env.state_size,
                           action_size=env.action_size,
                           player_id=1)

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            legal_moves = env.get_legal_moves()
            action = agent.choose_action(state, legal_moves)
            next_state, reward, done, info = env.step(action)

            # Reward shaping example
            reward += custom_reward(env)

            agent.remember(state, action, reward, next_state, done)
            agent.train_step()

            state = next_state

        if episode % 500 == 0:
            agent.save_model(f"double_dqn_checkpoint_{episode}.pt")

    agent.save_model()
    return agent

def custom_reward(env):
    # Implement your +0.1/-0.1/-1 heuristic logic here based on env state
    return 0.0
