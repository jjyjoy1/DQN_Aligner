#!/usr/bin/env python3
"""
DQNalign Reference-Based Sequence Aligner with BAM Output

A modernized implementation that:
1. Aligns multiple sequences to reference sequences
2. Outputs standard BAM files
3. Supports batch processing of multiple sequence files
4. Compatible with standard bioinformatics workflows

Author: Enhanced DQNalign for reference-based alignment
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import json
import logging
import pysam
import os
import sys
from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional, Dict, Any, Iterator
from collections import deque, namedtuple
from pathlib import Path
import math
from tqdm import tqdm
import argparse
from Bio import SeqIO
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experience replay transition
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

# Alignment result for BAM output
AlignmentResult = namedtuple('AlignmentResult', [
    'query_name', 'query_sequence', 'reference_name', 'reference_start', 'reference_end',
    'query_start', 'query_end', 'alignment_score', 'cigar_string', 'is_mapped'
])

@dataclass
class DQNAlignerConfig:
    """Configuration for reference-based DQN aligner"""
    # Reference and input parameters
    reference_file: str = "reference.fasta"
    input_files: List[str] = None
    output_dir: str = "output"
    
    # Alignment parameters
    window_size: int = 64
    overlap_size: int = 16
    min_alignment_score: int = 20
    max_reference_chunk: int = 1000
    
    # Scoring parameters
    match_score: int = 2
    mismatch_score: int = -1
    gap_open_score: int = -3
    gap_extend_score: int = -1
    
    # Network parameters
    hidden_size: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    
    # Training parameters
    learning_rate: float = 5e-5
    batch_size: int = 128
    buffer_size: int = 500000
    gamma: float = 0.99
    tau: float = 0.001
    
    # Rainbow DQN parameters
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: int = 100000
    target_update_freq: int = 5000
    double_dqn: bool = True
    dueling_dqn: bool = True
    noisy_networks: bool = True
    
    # Training parameters
    num_episodes: int = 50000
    max_steps_per_episode: int = 200
    
    # Processing parameters
    num_processes: int = multiprocessing.cpu_count()
    chunk_size: int = 1000
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class NoisyLinear(nn.Module):
    """Noisy network layer for exploration"""
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
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

class ReferenceDQN(nn.Module):
    """DQN for reference-based sequence alignment"""
    def __init__(self, config: DQNAlignerConfig):
        super().__init__()
        self.config = config
        self.num_actions = 4  # Match, mismatch, insert, delete
        
        # Sequence encoders
        self.query_encoder = nn.Sequential(
            nn.Conv1d(5, 128, kernel_size=5, padding=2),  # 5 channels: A,T,G,C,N
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(config.window_size)
        )
        
        self.reference_encoder = nn.Sequential(
            nn.Conv1d(5, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(config.window_size)
        )
        
        # Position encoders
        self.position_encoder = nn.Sequential(
            nn.Linear(4, 64),  # query_pos, ref_pos, query_len, ref_len
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Feature fusion
        input_size = 256 * config.window_size * 2 + 128  # Two sequences + position
        
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
        """Forward pass"""
        batch_size = state.shape[0]
        
        # Split state into components
        query_seq = state[:, :5, :self.config.window_size]
        ref_seq = state[:, :5, self.config.window_size:2*self.config.window_size]
        positions = state[:, 5, :4]  # Position information
        
        # Encode sequences
        query_enc = self.query_encoder(query_seq)
        ref_enc = self.reference_encoder(ref_seq)
        
        # Encode positions
        pos_enc = self.position_encoder(positions)
        
        # Concatenate features
        features = torch.cat([
            query_enc.flatten(1),
            ref_enc.flatten(1),
            pos_enc
        ], dim=1)
        
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

class ReferenceAlignmentEnvironment:
    """Environment for reference-based sequence alignment"""
    def __init__(self, config: DQNAlignerConfig):
        self.config = config
        self.nucleotides = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        self.action_names = ['match', 'mismatch', 'insert', 'delete']
        self.reset()
    
    def reset(self, query_seq: str = None, ref_seq: str = None, ref_start: int = 0) -> np.ndarray:
        """Reset environment with new sequences"""
        if query_seq is None:
            query_seq = self._generate_random_sequence(np.random.randint(50, 150))
        if ref_seq is None:
            ref_seq = self._generate_random_sequence(np.random.randint(100, 500))
        
        self.query_seq = query_seq.upper()
        self.ref_seq = ref_seq.upper()
        self.ref_start = ref_start
        
        self.query_pos = 0
        self.ref_pos = 0
        self.alignment_score = 0
        self.cigar_ops = []
        self.done = False
        
        return self._get_state()
    
    def _generate_random_sequence(self, length: int) -> str:
        """Generate a random DNA sequence"""
        return ''.join(random.choices('ATGC', k=length))
    
    def _encode_sequence(self, sequence: str, start: int, length: int) -> np.ndarray:
        """Encode sequence segment as one-hot"""
        encoded = np.zeros((5, length))  # 5 channels for A,T,G,C,N
        end = min(start + length, len(sequence))
        
        for i, char in enumerate(sequence[start:end]):
            if char in self.nucleotides and i < length:
                encoded[self.nucleotides[char], i] = 1
        
        return encoded
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        window = self.config.window_size
        
        # Get sequence windows
        query_window = self.query_seq[self.query_pos:self.query_pos + window]
        ref_window = self.ref_seq[self.ref_pos:self.ref_pos + window]
        
        # Pad if necessary
        query_window = query_window.ljust(window, 'N')[:window]
        ref_window = ref_window.ljust(window, 'N')[:window]
        
        # Encode sequences
        query_enc = self._encode_sequence(query_window, 0, window)
        ref_enc = self._encode_sequence(ref_window, 0, window)
        
        # Position information
        positions = np.array([
            self.query_pos / len(self.query_seq),
            self.ref_pos / len(self.ref_seq),
            len(self.query_seq) / 1000.0,  # Normalized lengths
            len(self.ref_seq) / 1000.0
        ], dtype=np.float32)
        
        # Create state tensor
        state = np.zeros((6, max(window * 2, 4)), dtype=np.float32)
        state[:5, :window] = query_enc
        state[:5, window:2*window] = ref_enc
        state[5, :4] = positions
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take an action in the environment"""
        reward = 0
        cigar_op = None
        
        if action == 0:  # Match
            if (self.query_pos < len(self.query_seq) and self.ref_pos < len(self.ref_seq)):
                if self.query_seq[self.query_pos] == self.ref_seq[self.ref_pos]:
                    reward = self.config.match_score
                    cigar_op = 'M'
                else:
                    reward = self.config.mismatch_score
                    cigar_op = 'M'
                self.query_pos += 1
                self.ref_pos += 1
        
        elif action == 1:  # Mismatch (explicit)
            if (self.query_pos < len(self.query_seq) and self.ref_pos < len(self.ref_seq)):
                reward = self.config.mismatch_score
                cigar_op = 'M'
                self.query_pos += 1
                self.ref_pos += 1
        
        elif action == 2:  # Insert to query (deletion from reference)
            if self.query_pos < len(self.query_seq):
                reward = self.config.gap_open_score
                cigar_op = 'I'
                self.query_pos += 1
        
        elif action == 3:  # Delete from query (insertion to reference)
            if self.ref_pos < len(self.ref_seq):
                reward = self.config.gap_open_score
                cigar_op = 'D'
                self.ref_pos += 1
        
        if cigar_op:
            self.cigar_ops.append(cigar_op)
        
        self.alignment_score += reward
        
        # Check if done
        self.done = (self.query_pos >= len(self.query_seq) or 
                    self.ref_pos >= len(self.ref_seq) or
                    len(self.cigar_ops) >= self.config.max_steps_per_episode)
        
        next_state = self._get_state()
        info = {
            'query_pos': self.query_pos,
            'ref_pos': self.ref_pos,
            'alignment_score': self.alignment_score,
            'cigar_ops': self.cigar_ops.copy(),
            'action_name': self.action_names[action]
        }
        
        return next_state, reward, self.done, info

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition: Transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNReferenceAligner:
    """DQN-based reference sequence aligner"""
    def __init__(self, config: DQNAlignerConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.q_network = ReferenceDQN(config).to(self.device)
        self.target_network = ReferenceDQN(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training state
        self.steps_done = 0
        self.episode_rewards = []
        
        # Load reference sequences
        self.reference_seqs = self._load_reference_sequences()
    
    def _load_reference_sequences(self) -> Dict[str, str]:
        """Load reference sequences from FASTA file"""
        references = {}
        try:
            for record in SeqIO.parse(self.config.reference_file, "fasta"):
                references[record.id] = str(record.seq).upper()
            logger.info(f"Loaded {len(references)} reference sequences")
        except FileNotFoundError:
            logger.warning(f"Reference file {self.config.reference_file} not found. Using dummy reference.")
            references['dummy_ref'] = 'ATGC' * 250  # 1000bp dummy reference
        return references
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy or noisy networks"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if training and not self.config.noisy_networks:
            epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * \
                     math.exp(-1. * self.steps_done / self.config.epsilon_decay)
            
            if random.random() < epsilon:
                return random.randrange(4)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.max(1)[1].item()
        
        return action
    
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
            self._soft_update_target_network()
    
    def _soft_update_target_network(self):
        """Soft update target network"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data)
    
    def align_sequence(self, query_seq: str, reference_name: str = None) -> AlignmentResult:
        """Align a single sequence to reference"""
        if reference_name is None:
            reference_name = list(self.reference_seqs.keys())[0]
        
        ref_seq = self.reference_seqs[reference_name]
        
        # Try alignment at different reference positions
        best_alignment = None
        best_score = float('-inf')
        
        # Sample multiple starting positions
        num_trials = min(10, len(ref_seq) // 100)
        for trial in range(num_trials):
            ref_start = random.randint(0, max(0, len(ref_seq) - len(query_seq) - 100))
            
            env = ReferenceAlignmentEnvironment(self.config)
            state = env.reset(query_seq, ref_seq, ref_start)
            
            for step in range(self.config.max_steps_per_episode):
                action = self.select_action(state, training=False)
                state, reward, done, info = env.step(action)
                
                if done:
                    break
            
            if info['alignment_score'] > best_score:
                best_score = info['alignment_score']
                best_alignment = AlignmentResult(
                    query_name=f"query_{random.randint(1000, 9999)}",
                    query_sequence=query_seq,
                    reference_name=reference_name,
                    reference_start=ref_start + env.ref_pos - len(info['cigar_ops']),
                    reference_end=ref_start + env.ref_pos,
                    query_start=0,
                    query_end=env.query_pos,
                    alignment_score=info['alignment_score'],
                    cigar_string=self._compress_cigar(info['cigar_ops']),
                    is_mapped=info['alignment_score'] >= self.config.min_alignment_score
                )
        
        return best_alignment
    
    def _compress_cigar(self, cigar_ops: List[str]) -> str:
        """Compress CIGAR operations"""
        if not cigar_ops:
            return "*"
        
        compressed = []
        current_op = cigar_ops[0]
        current_count = 1
        
        for op in cigar_ops[1:]:
            if op == current_op:
                current_count += 1
            else:
                compressed.append(f"{current_count}{current_op}")
                current_op = op
                current_count = 1
        
        compressed.append(f"{current_count}{current_op}")
        return ''.join(compressed)
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'steps_done': self.steps_done,
            'reference_seqs': self.reference_seqs
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        if 'reference_seqs' in checkpoint:
            self.reference_seqs = checkpoint['reference_seqs']

def create_bam_file(alignments: List[AlignmentResult], reference_seqs: Dict[str, str], output_path: str):
    """Create BAM file from alignment results"""
    # Create SAM header
    header = {'HD': {'VN': '1.0'}, 'SQ': []}
    
    for ref_name, ref_seq in reference_seqs.items():
        header['SQ'].append({'LN': len(ref_seq), 'SN': ref_name})
    
    # Write BAM file
    with pysam.AlignmentFile(output_path, "wb", header=header) as bam_file:
        for alignment in alignments:
            if not alignment.is_mapped:
                continue
            
            # Create alignment record
            read = pysam.AlignedSegment()
            read.query_name = alignment.query_name
            read.query_sequence = alignment.query_sequence
            read.flag = 0 if alignment.is_mapped else 4  # 4 = unmapped
            
            if alignment.is_mapped:
                read.reference_id = bam_file.get_tid(alignment.reference_name)
                read.reference_start = alignment.reference_start
                read.mapping_quality = min(60, max(0, int(alignment.alignment_score * 2)))
                read.cigarstring = alignment.cigar_string
            
            # Set quality scores (dummy values)
            read.query_qualities = [30] * len(alignment.query_sequence)
            
            bam_file.write(read)
    
    # Sort and index BAM file
    pysam.sort("-o", output_path.replace('.bam', '_sorted.bam'), output_path)
    pysam.index(output_path.replace('.bam', '_sorted.bam'))
    
    logger.info(f"Created BAM file: {output_path}")

def process_sequence_file(file_path: str, aligner: DQNReferenceAligner, output_dir: str) -> str:
    """Process a single sequence file"""
    output_path = Path(output_dir) / f"{Path(file_path).stem}_aligned.bam"
    alignments = []
    
    # Determine file format
    file_format = "fastq" if file_path.lower().endswith(('.fastq', '.fq')) else "fasta"
    
    logger.info(f"Processing {file_path} ({file_format} format)")
    
    # Process sequences
    sequence_count = 0
    for record in SeqIO.parse(file_path, file_format):
        query_seq = str(record.seq).upper()
        
        # Skip very short sequences
        if len(query_seq) < 20:
            continue
        
        # Align sequence
        alignment = aligner.align_sequence(query_seq)
        alignment = alignment._replace(query_name=record.id)
        alignments.append(alignment)
        
        sequence_count += 1
        
        if sequence_count % 100 == 0:
            logger.info(f"Processed {sequence_count} sequences from {Path(file_path).name}")
    
    # Create BAM file
    create_bam_file(alignments, aligner.reference_seqs, str(output_path))
    
    logger.info(f"Completed processing {file_path}: {sequence_count} sequences -> {output_path}")
    return str(output_path)

def batch_process_files(config: DQNAlignerConfig, model_path: str = None) -> List[str]:
    """Batch process multiple sequence files"""
    # Initialize aligner
    aligner = DQNReferenceAligner(config)
    
    # Load or train model
    if model_path and Path(model_path).exists():
        aligner.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.info("Training new model...")
        train_reference_aligner(config, save_path=model_path or "models/reference_aligner.pth")
        aligner.load_model(model_path or "models/reference_aligner.pth")
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process files
    output_files = []
    
    if config.num_processes > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
            futures = {
                executor.submit(process_sequence_file, file_path, aligner, config.output_dir): file_path
                for file_path in config.input_files
            }
            
            for future in as_completed(futures):
                try:
                    output_path = future.result()
                    output_files.append(output_path)
                except Exception as e:
                    logger.error(f"Failed to process {futures[future]}: {e}")
    else:
        # Sequential processing
        for file_path in config.input_files:
            try:
                output_path = process_sequence_file(file_path, aligner, config.output_dir)
                output_files.append(output_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
    
    return output_files

def train_reference_aligner(config: DQNAlignerConfig, save_path: str = "models/reference_aligner.pth"):
    """Train the reference-based aligner"""
    # Create aligner and environment
    aligner = DQNReferenceAligner(config)
    env = ReferenceAlignmentEnvironment(config)
    
    # Create save directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    logger.info(f"Starting training for {config.num_episodes} episodes")
    
    for episode in tqdm(range(config.num_episodes), desc="Training"):
        # Generate random query and select random reference
        ref_name = random.choice(list(aligner.reference_seqs.keys()))
        ref_seq = aligner.reference_seqs[ref_name]
        
        # Create similar query sequence
        ref_start = random.randint(0, max(0, len(ref_seq) - 200))
        query_length = random.randint(50, 150)
        query_seq = ref_seq[ref_start:ref_start + query_length]
        
        # Add some mutations
        query_seq = list(query_seq)
        for i in range(len(query_seq)):
            if random.random() < 0.05:  # 5% mutation rate
                query_seq[i] = random.choice('ATGC')
        query_seq = ''.join(query_seq)
        
        state = env.reset(query_seq, ref_seq, ref_start)
        episode_reward = 0
        
        for step in range(config.max_steps_per_episode):
            action = aligner.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Store transition
            aligner.replay_buffer.push(Transition(
                state, action, reward, next_state if not done else None, done
            ))
            
            # Train agent
            aligner.train_step()
            
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Log progress
        if episode % 1000 == 0:
            avg_reward = np.mean(episode_rewards[-1000:])
            logger.info(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                aligner.save_model(save_path)
                logger.info(f"New best model saved with reward: {best_reward:.2f}")
    
    return aligner, episode_rewards

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="DQNalign: Reference-based sequence aligner with BAM output")
    
    parser.add_argument("--reference", "-r", required=True, help="Reference FASTA file")
    parser.add_argument("--input", "-i", nargs="+", required=True, help="Input sequence files (FASTA/FASTQ)")
    parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    parser.add_argument("--model", "-m", help="Pre-trained model file")
    parser.add_argument("--train", action="store_true", help="Train new model")
    parser.add_argument("--num-processes", "-p", type=int, default=multiprocessing.cpu_count(), help="Number of processes")
    parser.add_argument("--window-size", type=int, default=64, help="Alignment window size")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Configure device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Create configuration
    config = DQNAlignerConfig(
        reference_file=args.reference,
        input_files=args.input,
        output_dir=args.output_dir,
        num_processes=args.num_processes,
        window_size=args.window_size,
        device=device
    )
    
    logger.info(f"DQNalign starting with {len(config.input_files)} input files")
    logger.info(f"Reference: {config.reference_file}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Using device: {config.device}")
    
    # Process files
    if args.train:
        logger.info("Training mode: will train new model before processing")
    
    model_path = args.model or "models/reference_aligner.pth"
    output_files = batch_process_files(config, model_path)
    
    logger.info("Processing completed!")
    logger.info(f"Output BAM files:")
    for output_file in output_files:
        logger.info(f"  • {output_file}")

if __name__ == "__main__":
    main()

