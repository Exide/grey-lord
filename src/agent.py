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


class TemporalDQN(nn.Module):
    """
    Enhanced DQN with temporal reasoning capabilities.
    Designed for agent-driven temporal learning without external attribution.
    """
    
    def __init__(self, observation_shape, action_size, embed_size=128, lstm_hidden=256, num_heads=4, vocab_size=501):
        super(TemporalDQN, self).__init__()
        
        # observation_shape: (max_observations, observation_window)
        self.max_observations, self.observation_window = observation_shape
        self.action_size = action_size
        self.embed_size = embed_size
        self.lstm_hidden = lstm_hidden
        self.vocab_size = vocab_size
        
        # Token embedding layer - convert token IDs to dense vectors  
        # Use PAD token (500) as padding_idx for GreyLordTokenizer
        self.token_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=500)
        
        # Sequence encoding layers
        self.sequence_encoder = nn.Linear(self.observation_window * embed_size, embed_size)
        
        # LSTM for temporal memory across game steps
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=lstm_hidden,
            batch_first=True,
            dropout=0.2
        )
        
        # Multi-head attention for focusing on relevant temporal patterns
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Action value head
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden // 2, action_size)
        )
        
        # Auxiliary prediction heads for state inference learning
        self.reward_predictor = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden // 2, 1)
        )
        
        self.next_token_predictor = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden // 2, vocab_size)  # Predict next token
        )
        
        # Amygdala head - survival expert for immediate threats
        self.amygdala_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden // 2, 3)  # [threat_level, health_urgency, escape_availability]
        )
        
        # Future implementation heads (commented out for now):
        # 
        # # Cerebrum head - strategic expert for long-term planning
        # self.cerebrum_head = nn.Sequential(
        #     nn.Linear(lstm_hidden, lstm_hidden // 2),
        #     nn.ReLU(),
        #     nn.Linear(lstm_hidden // 2, 5)  # [goal_progress, resource_status, strategic_priority, etc.]
        # )
        # 
        # # Cartographer head - navigation expert for spatial understanding
        # self.cartographer_head = nn.Sequential(
        #     nn.Linear(lstm_hidden, lstm_hidden // 2),
        #     nn.ReLU(),
        #     nn.Linear(lstm_hidden // 2, 8)  # [north, south, east, west, up, down, current_room_type, landmark_proximity]
        # )
        # 
        # # Quest head - mission expert for quest management
        # self.quest_head = nn.Sequential(
        #     nn.Linear(lstm_hidden, lstm_hidden // 2),
        #     nn.ReLU(),
        #     nn.Linear(lstm_hidden // 2, 4)  # [quest_progress, completion_probability, reward_value, priority_level]
        # )
        # 
        # # Quartermaster head - equipment expert for gear optimization
        # self.quartermaster_head = nn.Sequential(
        #     nn.Linear(lstm_hidden, lstm_hidden // 2),
        #     nn.ReLU(),
        #     nn.Linear(lstm_hidden // 2, 6)  # [weapon_quality, armor_quality, inventory_full, needs_repair, upgrade_available, encumbrance]
        # )
        
    def forward(self, x, return_auxiliaries=False):
        """
        Forward pass with temporal reasoning.
        
        Args:
            x: Shape (batch_size, max_observations, observation_window)
            return_auxiliaries: Whether to return auxiliary predictions
        """
        batch_size = x.size(0)
        
        # Embed tokens: (batch, max_obs, obs_window) -> (batch, max_obs, obs_window, embed_size)
        embedded = self.token_embedding(x.long())
        
        # Encode each observation: (batch, max_obs, obs_window * embed_size)
        obs_encoded = embedded.view(batch_size, self.max_observations, -1)
        obs_encoded = self.sequence_encoder(obs_encoded)  # (batch, max_obs, embed_size)
        
        # LSTM for temporal reasoning: (batch, max_obs, lstm_hidden)
        lstm_out, (hidden, cell) = self.lstm(obs_encoded)
        
        # Multi-head attention for temporal correlation
        # Query: most recent observation, Key/Value: all observations
        query = lstm_out[:, -1:, :]  # Last timestep as query
        attended_out, attention_weights = self.temporal_attention(
            query, lstm_out, lstm_out
        )
        
        # Use attended representation for final predictions
        final_repr = attended_out.squeeze(1)  # (batch, lstm_hidden)
        
        # Action values
        q_values = self.value_head(final_repr)
        
        if return_auxiliaries:
            # Auxiliary predictions for self-supervised learning
            reward_pred = self.reward_predictor(final_repr)
            next_token_pred = self.next_token_predictor(final_repr)
            amygdala_pred = self.amygdala_head(final_repr)
            
            return q_values, {
                'reward_prediction': reward_pred,
                'next_token_prediction': next_token_pred,
                'amygdala_prediction': amygdala_pred,
                'attention_weights': attention_weights
            }
        
        return q_values
    
    def update_vocab_size(self, vocab_size):
        """Update embedding layer with actual tokenizer vocab size."""
        self.token_embedding = nn.Embedding(vocab_size, self.embed_size, padding_idx=0)


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


class TemporalDQNAgent(BBSAgent):
    """
    Enhanced DQN Agent with temporal reasoning capabilities.
    Uses LSTM + attention for agent-driven temporal learning.
    """
    
    def __init__(self, action_size, observation_shape, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=32, update_frequency=4, 
                 aux_loss_weight=0.1, vocab_size=50000):
        super().__init__(action_size, observation_shape)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.aux_loss_weight = aux_loss_weight
        self.step_count = 0
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = TemporalDQN(observation_shape, action_size, vocab_size=vocab_size).to(self.device)
        self.target_network = TemporalDQN(observation_shape, action_size, vocab_size=vocab_size).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Enhanced experience replay with auxiliary data
        self.memory = TemporalReplayBuffer(buffer_size)
        
        # Copy weights to target network
        self.update_target_network()
        
        logger.info(f"Initialized TemporalDQNAgent with device: {self.device}")
    
    def act(self, state):
        """Choose action using epsilon-greedy policy with temporal reasoning."""
        if random.random() > self.epsilon:
            # Exploit: choose best action using temporal network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
        else:
            # Explore: choose random action
            action = random.randint(0, self.action_size - 1)
        
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Update the agent with enhanced temporal learning."""
        # Store experience in enhanced replay buffer
        self.memory.push(state, action, reward, next_state, done)
        
        self.step_count += 1
        
        # Update network if enough experiences are collected
        if len(self.memory) >= self.batch_size and self.step_count % self.update_frequency == 0:
            self._train()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _train(self):
        """Enhanced training with auxiliary losses for temporal reasoning."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch_data = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch_data
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Forward pass with auxiliary predictions
        q_values, aux_outputs = self.q_network(states, return_auxiliaries=True)
        current_q_values = q_values.gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Main Q-learning loss
        q_loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Auxiliary losses for temporal reasoning
        aux_loss = 0.0
        
        # Reward prediction loss - helps learn state-reward correlation
        if 'reward_prediction' in aux_outputs:
            reward_pred_loss = F.mse_loss(
                aux_outputs['reward_prediction'].squeeze(), 
                rewards
            )
            aux_loss += reward_pred_loss
        
        # Next token prediction loss - helps learn game dynamics
        if 'next_token_prediction' in aux_outputs:
            # Predict first token of next observation
            next_first_tokens = next_states[:, 0, 0].long()  # First token of next state
            token_pred_loss = F.cross_entropy(
                aux_outputs['next_token_prediction'], 
                next_first_tokens
            )
            aux_loss += token_pred_loss * 0.1  # Lower weight for token prediction
        
        # Amygdala loss - survival expert learning
        if 'amygdala_prediction' in aux_outputs:
            # Create survival targets based on game state
            amygdala_targets = self._create_amygdala_targets(states, rewards)
            amygdala_loss = F.mse_loss(
                aux_outputs['amygdala_prediction'],
                amygdala_targets
            )
            aux_loss += amygdala_loss * 0.5  # Moderate weight for survival learning
        
        # Total loss
        total_loss = q_loss + (self.aux_loss_weight * aux_loss)
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        if self.step_count % (self.update_frequency * 100) == 0:
            self.update_target_network()
        
        # Log training info
        if self.step_count % 1000 == 0:
            logger.debug(f"Step {self.step_count}: Q-loss={q_loss:.4f}, Aux-loss={aux_loss:.4f}")
    
    def _create_amygdala_targets(self, states, rewards):
        """
        Create survival-related targets for amygdala head training.
        
        Args:
            states: Batch of game states (batch_size, max_observations, observation_window)
            rewards: Batch of rewards (batch_size,)
            
        Returns:
            targets: (batch_size, 3) tensor with [threat_level, health_urgency, escape_availability]
        """
        batch_size = states.size(0)
        targets = torch.zeros(batch_size, 3, device=states.device)
        
        # Simple heuristic-based targets (would be improved with more sophisticated analysis)
        for i in range(batch_size):
            reward = rewards[i].item()
            
            # Threat level (0-1): High when taking damage or in combat
            if reward < -5:  # Significant damage
                targets[i, 0] = 1.0  # High threat
            elif reward < 0:  # Minor damage
                targets[i, 0] = 0.7  # Moderate threat
            else:
                targets[i, 0] = 0.1  # Low threat
            
            # Health urgency (0-1): High when health is low or taking damage
            if reward < -10:  # Major damage
                targets[i, 1] = 1.0  # Critical health urgency
            elif reward < -3:  # Moderate damage
                targets[i, 1] = 0.6  # Moderate health urgency
            else:
                targets[i, 1] = 0.2  # Low health urgency
            
            # Escape availability (0-1): Lower when in combat, higher when safe
            if reward < -2:  # In combat/danger
                targets[i, 2] = 0.3  # Limited escape options
            else:
                targets[i, 2] = 0.8  # Good escape options
        
        return targets
    
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
        logger.info(f"Saved TemporalDQNAgent to {filepath}")
    
    def load(self, filepath):
        """Load the agent's parameters."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        logger.info(f"Loaded TemporalDQNAgent from {filepath}")


class TemporalReplayBuffer:
    """Enhanced replay buffer for temporal learning."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store experience with additional temporal context."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample batch of experiences."""
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


def create_agent(agent_type, action_size, observation_shape, **kwargs):
    """Factory function to create agents."""
    if agent_type.lower() == 'random':
        return RandomAgent(action_size, observation_shape)
    elif agent_type.lower() == 'dqn':
        return DQNAgent(action_size, observation_shape, **kwargs)
    elif agent_type.lower() == 'temporal_dqn':
        return TemporalDQNAgent(action_size, observation_shape, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
