#!/usr/bin/env python3
"""
Quick usage example for Modern DQNalign
"""

from modern_dqnalign import DQNConfig, DQNAgent, SequenceAlignmentEnvironment, train_agent
import torch

# 1. Basic training example
def quick_train():
    """Quick training example with default parameters"""
    config = DQNConfig(
        window_size=30,
        num_episodes=1000,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Training DQN agent for sequence alignment...")
    agent, rewards = train_agent(config, save_path="models/quick_model.pth")
    
    print(f"Training completed! Final reward: {rewards[-1]:.2f}")
    return agent

# 2. Custom sequence alignment
def align_custom_sequences():
    """Align custom DNA sequences"""
    # Load trained model
    config = DQNConfig()
    agent = DQNAgent(config)
    
    try:
        agent.load_model("models/quick_model.pth")
        print("Loaded pre-trained model")
    except:
        print("No pre-trained model found, training new one...")
        agent, _ = train_agent(config)
    
    # Create environment
    env = SequenceAlignmentEnvironment(config)
    
    # Test sequences
    seq1 = "ATGCGATCGATCGATCGTAGCTAG"
    seq2 = "ATGCGATCGATCGATCGTAGCTAG"  # Same sequence
    
    print(f"Aligning sequences:")
    print(f"Seq1: {seq1}")
    print(f"Seq2: {seq2}")
    
    # Perform alignment
    state = env.reset(seq1, seq2)
    alignment_path = []
    
    for step in range(100):  # Max steps
        action = agent.select_action(state, training=False)
        state, reward, done, info = env.step(action)
        
        action_names = ["Match/Forward", "Delete", "Insert"]
        alignment_path.append(action_names[action])
        
        if done:
            break
    
    print(f"Alignment completed in {step+1} steps")
    print(f"Final score: {info['alignment_score']}")
    print(f"Alignment path: {' -> '.join(alignment_path[:10])}...")

# 3. Batch evaluation on multiple sequences
def evaluate_on_dataset():
    """Evaluate on a set of sequence pairs"""
    from Bio.Seq import Seq
    import random
    
    # Load or train model
    config = DQNConfig()
    agent = DQNAgent(config)
    
    try:
        agent.load_model("models/quick_model.pth")
    except:
        print("Training new model...")
        agent, _ = train_agent(config)
    
    # Generate test dataset
    def generate_test_pairs(n_pairs=10):
        pairs = []
        for _ in range(n_pairs):
            # Generate original sequence
            seq1 = ''.join(random.choices('ATGC', k=random.randint(20, 50)))
            
            # Create mutated version
            seq2 = list(seq1)
            for i in range(len(seq2)):
                if random.random() < 0.1:  # 10% mutation rate
                    seq2[i] = random.choice('ATGC')
            
            pairs.append((seq1, ''.join(seq2)))
        return pairs
    
    test_pairs = generate_test_pairs(20)
    
    # Evaluate
    env = SequenceAlignmentEnvironment(config)
    results = []
    
    for i, (seq1, seq2) in enumerate(test_pairs):
        state = env.reset(seq1, seq2)
        score = 0
        
        for _ in range(100):
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            score += reward
            
            if done:
                break
        
        results.append({
            'pair_id': i,
            'seq1_len': len(seq1),
            'seq2_len': len(seq2),
            'alignment_score': score,
            'normalized_score': score / max(len(seq1), len(seq2))
        })
    
    # Print results
    avg_score = sum(r['alignment_score'] for r in results) / len(results)
    avg_normalized = sum(r['normalized_score'] for r in results) / len(results)
    
    print(f"Evaluated {len(test_pairs)} sequence pairs")
    print(f"Average alignment score: {avg_score:.2f}")
    print(f"Average normalized score: {avg_normalized:.2f}")
    
    return results

# 4. Advanced configuration example
def advanced_training():
    """Example with advanced Rainbow DQN configuration"""
    config = DQNConfig(
        # Environment
        window_size=64,
        sequence_max_len=500,
        
        # Scoring
        match_score=2,
        mismatch_score=-1,
        gap_score=-1,
        
        # Network architecture
        hidden_size=512,
        num_layers=4,
        dropout=0.1,
        
        # Training
        learning_rate=5e-5,
        batch_size=128,
        buffer_size=200000,
        gamma=0.995,
        tau=0.001,
        
        # Rainbow DQN features
        double_dqn=True,
        dueling_dqn=True,
        noisy_networks=True,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=50000,
        
        # Training schedule
        num_episodes=10000,
        max_steps_per_episode=300,
        target_update_freq=2000,
        
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("Training with advanced Rainbow DQN configuration...")
    agent, rewards = train_agent(config, save_path="models/advanced_model.pth")
    
    # Plot training progress
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(1, 2, 2)
        # Moving average
        window = 100
        moving_avg = [sum(rewards[max(0, i-window):i+1]) / min(i+1, window) 
                     for i in range(len(rewards))]
        plt.plot(moving_avg)
        plt.title('Moving Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
        
    except ImportError:
        print("matplotlib not available for plotting")
    
    return agent

if __name__ == "__main__":
    print("Modern DQNalign Usage Examples")
    print("=" * 40)
    
    # Run examples
    print("\n1. Quick Training:")
    agent = quick_train()
    
    print("\n2. Custom Sequence Alignment:")
    align_custom_sequences()
    
    print("\n3. Batch Evaluation:")
    results = evaluate_on_dataset()
    
    print("\n4. Advanced Training (commented out due to time):")
    print("   # Uncomment the line below for advanced training")
    # advanced_agent = advanced_training()
    
    print("\nAll examples completed!")



