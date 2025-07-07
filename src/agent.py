import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import logging


logger = logging.getLogger(__name__)


class DQN(nn.Module):
    """Deep Q-Network for the BBS environment."""
    
    def __init__(self, observation_shape, action_size, hidden_size=256):
        super(DQN, self).__init__()
        
        # Input is (max_observations, observation_window)
        self.observation_shape = observation_shape
        self.input_size = observation_shape[0] * observation_shape[1]
        
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Flatten the observation
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class BBSAgent:
    """Base agent class for BBS environment."""
    
    def __init__(self, action_size, observation_shape):
        self.action_size = action_size
        self.observation_shape = observation_shape
        
    def act(self, state):
        """Choose an action given the current state."""
        raise NotImplementedError
    
    def update(self, state, action, reward, next_state, done):
        """Update the agent with experience."""
        pass
    
    def save(self, filepath):
        """Save the agent's parameters."""
        pass
    
    def load(self, filepath):
        """Load the agent's parameters."""
        pass


class RandomAgent(BBSAgent):
    """Random agent that chooses actions randomly."""
    
    def __init__(self, action_size, observation_shape):
        super().__init__(action_size, observation_shape)
        logger.info("Initialized RandomAgent")
    
    def act(self, state):
        return random.randint(0, self.action_size - 1)


class DQNAgent(BBSAgent):
    """Deep Q-Network agent."""
    
    def __init__(self, action_size, observation_shape, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=32, update_frequency=4):
        super().__init__(action_size, observation_shape)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.step_count = 0
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQN(observation_shape, action_size).to(self.device)
        self.target_network = DQN(observation_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ReplayBuffer(buffer_size)
        
        # Copy weights to target network
        self.update_target_network()
        
        logger.info(f"Initialized DQNAgent with device: {self.device}")
    
    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.random() > self.epsilon:
            # Exploit: choose best action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        else:
            # Explore: choose random action
            action = random.randint(0, self.action_size - 1)
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Update the agent with experience."""
        # Store experience in replay buffer
        self.memory.push(state, action, reward, next_state, done)
        
        self.step_count += 1
        
        # Update network if enough experiences are collected
        if len(self.memory) >= self.batch_size and self.step_count % self.update_frequency == 0:
            self._train()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _train(self):
        """Train the DQN on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        if self.step_count % (self.update_frequency * 100) == 0:
            self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """Save the agent's parameters."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }, filepath)
        logger.info(f"Saved agent to {filepath}")
    
    def load(self, filepath):
        """Load the agent's parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        logger.info(f"Loaded agent from {filepath}")


def create_agent(agent_type, action_size, observation_shape, **kwargs):
    """Factory function to create agents."""
    if agent_type.lower() == 'random':
        return RandomAgent(action_size, observation_shape)
    elif agent_type.lower() == 'dqn':
        return DQNAgent(action_size, observation_shape, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
