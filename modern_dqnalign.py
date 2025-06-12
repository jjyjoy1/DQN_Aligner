#!/usr/bin/env python3
"""
Modern DQNalign: Deep Reinforcement Learning for DNA Sequence Alignment

A modernized implementation of DQNalign with improvements:
- PyTorch instead of TensorFlow 1.x
- Rainbow DQN with modern RL techniques
- Better software engineering practices
- Type hints and comprehensive documentation
- Configurable hyperparameters
- Efficient training and evaluation

Author: Modernized implementation based on Song & Cho (2021)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import json
import logging
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Dict, Any
from collections import deque, namedtuple
from pathlib import Path
import math
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Experience replay transition
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class DQNConfig:
    """Configuration for DQN alignment agent"""
    # Environment parameters
    window_size: int = 50
    sequence_max_len: int = 1000
    
    # Scoring parameters
    match_score: int = 1
    mismatch_score: int = -1
    gap_score: int = -2
    
    # Network parameters
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 100000
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    
    # Rainbow DQN parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000
    target_update_freq: int = 1000
    double_dqn: bool = True
    dueling_dqn: bool = True
    noisy_networks: bool = True
    
    # Training parameters
    num_episodes: int = 10000
    max_steps_per_episode: int = 500
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class NoisyLinear(nn.Module):
    """Noisy network layer for exploration"""
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Register noise buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class RainbowDQN(nn.Module):
    """Rainbow DQN with Dueling architecture and Noisy networks"""
    def __init__(self, config: DQNConfig):
        super().__init__()
        self.config = config
        self.num_actions = 3  # Forward/match, delete, insert
        
        # Input encoding for sequence pairs
        self.sequence_encoder = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=3, padding=1),  # 4 nucleotides: A,T,G,C
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(config.window_size)
        )
        
        # Feature extraction
        input_size = 128 * config.window_size * 2  # Two sequences
        
        if config.noisy_networks:
            self.feature_layer = NoisyLinear(input_size, config.hidden_size)
            self.hidden_layers = nn.ModuleList([
                NoisyLinear(config.hidden_size, config.hidden_size) 
                for _ in range(config.num_layers - 1)
            ])
        else:
            self.feature_layer = nn.Linear(input_size, config.hidden_size)
            self.hidden_layers = nn.ModuleList([
                nn.Linear(config.hidden_size, config.hidden_size) 
                for _ in range(config.num_layers - 1)
            ])
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Dueling DQN heads
        if config.dueling_dqn:
            if config.noisy_networks:
                self.value_head = NoisyLinear(config.hidden_size, 1)
                self.advantage_head = NoisyLinear(config.hidden_size, self.num_actions)
            else:
                self.value_head = nn.Linear(config.hidden_size, 1)
                self.advantage_head = nn.Linear(config.hidden_size, self.num_actions)
        else:
            if config.noisy_networks:
                self.q_head = NoisyLinear(config.hidden_size, self.num_actions)
            else:
                self.q_head = nn.Linear(config.hidden_size, self.num_actions)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        batch_size = state.shape[0]
        
        # Encode both sequences
        seq1, seq2 = state[:, :, :self.config.window_size], state[:, :, self.config.window_size:]
        
        enc1 = self.sequence_encoder(seq1)
        enc2 = self.sequence_encoder(seq2)
        
        # Concatenate encoded sequences
        features = torch.cat([enc1.flatten(1), enc2.flatten(1)], dim=1)
        
        # Feature extraction
        x = F.relu(self.feature_layer(features))
        x = self.dropout(x)
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        # Dueling or standard Q-values
        if self.config.dueling_dqn:
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            q_values = self.q_head(x)
        
        return q_values
    
    def reset_noise(self):
        """Reset noise for noisy networks"""
        if self.config.noisy_networks:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

class ReplayBuffer:
    """Experience replay buffer with prioritization support"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition: Transition):
        """Add a transition to the buffer"""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class SequenceAlignmentEnvironment:
    """Environment for sequence alignment task"""
    def __init__(self, config: DQNConfig):
        self.config = config
        self.nucleotides = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        self.reset()
    
    def reset(self, seq1: Optional[str] = None, seq2: Optional[str] = None) -> np.ndarray:
        """Reset environment with new sequences"""
        if seq1 is None or seq2 is None:
            # Generate random sequences for training
            seq1 = self._generate_random_sequence(np.random.randint(50, 200))
            seq2 = self._mutate_sequence(seq1, mutation_rate=0.1)
        
        self.seq1 = seq1.upper()
        self.seq2 = seq2.upper()
        self.pos1 = 0
        self.pos2 = 0
        self.alignment_score = 0
        self.done = False
        
        return self._get_state()
    
    def _generate_random_sequence(self, length: int) -> str:
        """Generate a random DNA sequence"""
        return ''.join(random.choices('ATGC', k=length))
    
    def _mutate_sequence(self, sequence: str, mutation_rate: float = 0.1) -> str:
        """Apply mutations to create a similar sequence"""
        mutated = list(sequence)
        for i in range(len(mutated)):
            if random.random() < mutation_rate:
                if random.random() < 0.7:  # Substitution
                    mutated[i] = random.choice('ATGC')
                elif random.random() < 0.85:  # Deletion
                    mutated[i] = ''
                else:  # Insertion
                    mutated[i] += random.choice('ATGC')
        return ''.join(mutated)
    
    def _encode_sequence(self, sequence: str, start: int, length: int) -> np.ndarray:
        """Encode sequence segment as one-hot"""
        encoded = np.zeros((4, length))
        for i, char in enumerate(sequence[start:start+length]):
            if char in self.nucleotides and i < length:
                encoded[self.nucleotides[char], i] = 1
        return encoded
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Extract windows around current positions
        window = self.config.window_size
        
        # Get sequence windows
        seq1_window = self.seq1[max(0, self.pos1-window//2):self.pos1+window//2]
        seq2_window = self.seq2[max(0, self.pos2-window//2):self.pos2+window//2]
        
        # Pad if necessary
        seq1_window = seq1_window.ljust(window, 'N')[:window]
        seq2_window = seq2_window.ljust(window, 'N')[:window]
        
        # Encode sequences
        enc1 = self._encode_sequence(seq1_window, 0, window)
        enc2 = self._encode_sequence(seq2_window, 0, window)
        
        # Concatenate along sequence dimension
        state = np.concatenate([enc1, enc2], axis=1)
        
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take an action in the environment"""
        reward = 0
        
        if action == 0:  # Match/Forward
            if (self.pos1 < len(self.seq1) and self.pos2 < len(self.seq2) and 
                self.seq1[self.pos1] == self.seq2[self.pos2]):
                reward = self.config.match_score
            else:
                reward = self.config.mismatch_score
            self.pos1 += 1
            self.pos2 += 1
            
        elif action == 1:  # Delete from seq1
            reward = self.config.gap_score
            self.pos1 += 1
            
        elif action == 2:  # Insert to seq1 (delete from seq2)
            reward = self.config.gap_score
            self.pos2 += 1
        
        self.alignment_score += reward
        
        # Check if done
        self.done = (self.pos1 >= len(self.seq1) or self.pos2 >= len(self.seq2))
        
        next_state = self._get_state()
        info = {
            'pos1': self.pos1,
            'pos2': self.pos2,
            'alignment_score': self.alignment_score
        }
        
        return next_state, reward, self.done, info

class DQNAgent:
    """Rainbow DQN Agent for sequence alignment"""
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.q_network = RainbowDQN(config).to(self.device)
        self.target_network = RainbowDQN(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training state
        self.steps_done = 0
        self.episode_rewards = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy or noisy networks"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if training and not self.config.noisy_networks:
            # Epsilon-greedy exploration
            epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                     math.exp(-1. * self.steps_done / self.config.epsilon_decay)
            
            if random.random() < epsilon:
                return random.randrange(3)
        
        # Get Q-values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.max(1)[1].item()
        
        return action
    
    def store_transition(self, transition: Transition):
        """Store transition in replay buffer"""
        self.replay_buffer.push(transition)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        # Sample batch
        transitions = self.replay_buffer.sample(self.config.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([s for s in batch.next_state if s is not None])).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q values
        next_q_values = torch.zeros(self.config.batch_size).to(self.device)
        non_final_mask = ~done_batch
        
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_state_batch).max(1)[1]
                next_q_values[non_final_mask] = self.target_network(next_state_batch).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                next_q_values[non_final_mask] = self.target_network(next_state_batch).max(1)[0]
        
        # Target Q values
        target_q_values = reward_batch + (self.config.gamma * next_q_values)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Reset noise
        if self.config.noisy_networks:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        self.steps_done += 1
        
        # Update target network
        if self.steps_done % self.config.target_update_freq == 0:
            self.soft_update_target_network()
    
    def soft_update_target_network(self):
        """Soft update target network"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data)
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'steps_done': self.steps_done
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']

def train_agent(config: DQNConfig, save_path: str = "models/dqnalign_model.pth"):
    """Train the DQN agent"""
    # Create environment and agent
    env = SequenceAlignmentEnvironment(config)
    agent = DQNAgent(config)
    
    # Create save directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    for episode in tqdm(range(config.num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        
        for step in range(config.max_steps_per_episode):
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Store transition
            agent.store_transition(Transition(state, action, reward, next_state if not done else None, done))
            
            # Train agent
            agent.train_step()
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Log progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.config.epsilon_end + (agent.config.epsilon_start - agent.config.epsilon_end) * math.exp(-1. * agent.steps_done / agent.config.epsilon_decay):.3f}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_model(save_path)
                logger.info(f"New best model saved with reward: {best_reward:.2f}")
    
    return agent, episode_rewards

def evaluate_agent(agent: DQNAgent, test_sequences: List[Tuple[str, str]], config: DQNConfig) -> Dict[str, float]:
    """Evaluate agent on test sequences"""
    env = SequenceAlignmentEnvironment(config)
    
    total_rewards = []
    alignment_scores = []
    
    for seq1, seq2 in test_sequences:
        state = env.reset(seq1, seq2)
        episode_reward = 0
        
        for _ in range(config.max_steps_per_episode):
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        alignment_scores.append(info['alignment_score'])
    
    results = {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_alignment_score': np.mean(alignment_scores),
        'std_alignment_score': np.std(alignment_scores)
    }
    
    return results

def main():
    """Main training and evaluation pipeline"""
    # Configuration
    config = DQNConfig(
        window_size=50,
        hidden_size=256,
        learning_rate=1e-4,
        num_episodes=5000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    logger.info(f"Starting training with config: {asdict(config)}")
    logger.info(f"Using device: {config.device}")
    
    # Train agent
    agent, rewards = train_agent(config)
    
    # Generate test sequences
    test_sequences = [
        ("ATGCGATCGATCGATCG", "ATGCGATCGATCGATCG"),  # Perfect match
        ("ATGCGATCGATCGATCG", "ATGCGATCGATCGATCC"),  # Single mismatch
        ("ATGCGATCGATCGATCG", "ATGCGATCGATCGAT"),    # Deletion
        ("ATGCGATCGATCGATCG", "ATGCGATCGATCGATCGA"), # Insertion
    ]
    
    # Evaluate
    results = evaluate_agent(agent, test_sequences, config)
    logger.info(f"Evaluation results: {results}")
    
    # Save training results
    with open("training_results.json", "w") as f:
        json.dump({
            "config": asdict(config),
            "episode_rewards": rewards,
            "evaluation_results": results
        }, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()


