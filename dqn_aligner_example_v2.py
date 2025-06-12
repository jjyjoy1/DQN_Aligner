#!/usr/bin/env python3
"""
Complete Input/Output Example for Modern DQNalign
Shows exactly what goes in and what comes out
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
from modern_dqnalign import DQNConfig, DQNAgent, SequenceAlignmentEnvironment, train_agent

def create_input_files():
    """Create example input files"""
    
    # 1. Create FASTA input file
    fasta_content = """
>sequence_1
ATGCGATCGATCGATCGTAGCTAGCTAG
>sequence_2  
ATGCGATCGATCGATCGTAGCTAGCTAG
>sequence_3
ATGCGATCGATCGATCGTAGCT
>sequence_4
ATGCGATCGATCGATCGTAGCTAGCTAGAA
>sequence_5
ATGCGATCGATCGATCGTAGCTAGCTAGCC
"""
    
    with open("input_sequences.fasta", "w") as f:
        f.write(fasta_content.strip())
    
    # 2. Create sequence pairs file
    pairs_data = [
        {"id": "pair_1", "seq1": "ATGCGATCGATCGATCG", "seq2": "ATGCGATCGATCGATCG"},
        {"id": "pair_2", "seq1": "ATGCGATCGATCGATCG", "seq2": "ATGCGATCGATCGATCC"},
        {"id": "pair_3", "seq1": "ATGCGATCGATCGATCG", "seq2": "ATGCGATCGATCGAT"},
        {"id": "pair_4", "seq1": "ATGCGATCGATCGATCG", "seq2": "ATGCGATCGATCGATCGA"},
    ]
    
    with open("sequence_pairs.json", "w") as f:
        json.dump(pairs_data, f, indent=2)
    
    # 3. Create configuration file
    config_data = {
        "window_size": 50,
        "sequence_max_len": 1000,
        "match_score": 2,
        "mismatch_score": -1,
        "gap_score": -2,
        "hidden_size": 256,
        "learning_rate": 0.0001,
        "batch_size": 64,
        "num_episodes": 1000,
        "device": "cuda"
    }
    
    with open("config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    print("✅ Created input files:")
    print("   📄 input_sequences.fasta")
    print("   📄 sequence_pairs.json")
    print("   📄 config.json")

def load_sequences_from_fasta(filepath: str) -> List[str]:
    """Load sequences from FASTA file"""
    try:
        from Bio import SeqIO
        sequences = []
        for record in SeqIO.parse(filepath, "fasta"):
            sequences.append(str(record.seq))
        return sequences
    except ImportError:
        # Fallback without BioPython
        sequences = []
        with open(filepath, 'r') as f:
            current_seq = ""
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                sequences.append(current_seq)
        return sequences

def load_config_from_file(filepath: str) -> DQNConfig:
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    return DQNConfig(**config_dict)

def load_sequence_pairs(filepath: str) -> List[Dict]:
    """Load sequence pairs from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_alignment_results(results: List[Dict], filepath: str):
    """Save alignment results to file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def save_training_log(episode_rewards: List[float], filepath: str):
    """Save training progress log"""
    with open(filepath, 'w') as f:
        f.write("Episode,Reward,Moving_Average\n")
        window = 100
        for i, reward in enumerate(episode_rewards):
            moving_avg = sum(episode_rewards[max(0, i-window):i+1]) / min(i+1, window)
            f.write(f"{i},{reward},{moving_avg:.3f}\n")

def generate_alignment_report(results: List[Dict], filepath: str):
    """Generate detailed alignment report"""
    with open(filepath, 'w') as f:
        f.write("DQNalign Alignment Report\n")
        f.write("=" * 50 + "\n\n")
        
        total_pairs = len(results)
        avg_score = sum(r['alignment_score'] for r in results) / total_pairs
        
        f.write(f"Total sequence pairs processed: {total_pairs}\n")
        f.write(f"Average alignment score: {avg_score:.2f}\n")
        f.write(f"Score range: {min(r['alignment_score'] for r in results):.2f} to {max(r['alignment_score'] for r in results):.2f}\n\n")
        
        f.write("Individual Results:\n")
        f.write("-" * 30 + "\n")
        
        for i, result in enumerate(results):
            f.write(f"Pair {i+1}:\n")
            f.write(f"  Sequence 1 length: {result['seq1_len']}\n")
            f.write(f"  Sequence 2 length: {result['seq2_len']}\n")
            f.write(f"  Alignment score: {result['alignment_score']}\n")
            f.write(f"  Normalized score: {result['normalized_score']:.3f}\n")
            if 'alignment_path' in result:
                f.write(f"  Alignment path: {' -> '.join(result['alignment_path'][:5])}...\n")
            f.write("\n")

def main_pipeline():
    """Complete input/output pipeline"""
    
    print("🚀 DQNalign Input/Output Pipeline")
    print("=" * 50)
    
    # Step 1: Create input files
    print("\n📥 Step 1: Creating input files...")
    create_input_files()
    
    # Step 2: Load inputs
    print("\n📖 Step 2: Loading inputs...")
    
    # Load configuration
    config = load_config_from_file("config.json")
    print(f"   ✅ Loaded config: {config.window_size} window size")
    
    # Load sequence pairs
    sequence_pairs = load_sequence_pairs("sequence_pairs.json")
    print(f"   ✅ Loaded {len(sequence_pairs)} sequence pairs")
    
    # Load FASTA sequences (optional)
    try:
        fasta_sequences = load_sequences_from_fasta("input_sequences.fasta")
        print(f"   ✅ Loaded {len(fasta_sequences)} sequences from FASTA")
    except:
        print("   ⚠️  Could not load FASTA (BioPython not available)")
        fasta_sequences = []
    
    # Step 3: Training
    print("\n🎯 Step 3: Training agent...")
    
    # Create output directories
    Path("models").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Train the agent
    agent, episode_rewards = train_agent(config, save_path="models/dqnalign_trained.pth")
    print(f"   ✅ Training completed: {len(episode_rewards)} episodes")
    
    # Step 4: Evaluation
    print("\n📊 Step 4: Evaluating on sequence pairs...")
    
    env = SequenceAlignmentEnvironment(config)
    results = []
    
    for pair_data in sequence_pairs:
        seq1 = pair_data["seq1"]
        seq2 = pair_data["seq2"]
        pair_id = pair_data["id"]
        
        # Reset environment with sequences
        state = env.reset(seq1, seq2)
        episode_reward = 0
        alignment_path = []
        
        # Perform alignment
        for step in range(config.max_steps_per_episode):
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            action_names = ["Match", "Delete", "Insert"]
            alignment_path.append(action_names[action])
            
            if done:
                break
        
        # Store results
        result = {
            'pair_id': pair_id,
            'seq1': seq1,
            'seq2': seq2,
            'seq1_len': len(seq1),
            'seq2_len': len(seq2),
            'alignment_score': info['alignment_score'],
            'episode_reward': episode_reward,
            'normalized_score': info['alignment_score'] / max(len(seq1), len(seq2)),
            'final_pos1': info['pos1'],
            'final_pos2': info['pos2'],
            'alignment_path': alignment_path,
            'num_steps': step + 1
        }
        results.append(result)
        
        print(f"   ✅ {pair_id}: Score {info['alignment_score']}")
    
    # Step 5: Save outputs
    print("\n💾 Step 5: Saving outputs...")
    
    # Save alignment results
    save_alignment_results(results, "output/alignment_results.json")
    print("   ✅ Saved alignment_results.json")
    
    # Save training log
    save_training_log(episode_rewards, "logs/training_log.csv")
    print("   ✅ Saved training_log.csv")
    
    # Save detailed report
    generate_alignment_report(results, "output/alignment_report.txt")
    print("   ✅ Saved alignment_report.txt")
    
    # Save final configuration
    with open("output/final_config.json", "w") as f:
        json.dump({
            "config": config.__dict__,
            "training_episodes": len(episode_rewards),
            "final_reward": episode_rewards[-1] if episode_rewards else 0,
            "evaluation_pairs": len(results)
        }, f, indent=2)
    print("   ✅ Saved final_config.json")
    
    # Step 6: Summary
    print("\n📋 Step 6: Summary of outputs...")
    print("\n📁 Generated Files:")
    print("   📂 models/")
    print("      └── dqnalign_trained.pth     # Trained model weights")
    print("   📂 output/")
    print("      ├── alignment_results.json   # Detailed alignment results")
    print("      ├── alignment_report.txt     # Human-readable report")
    print("      └── final_config.json        # Configuration used")
    print("   📂 logs/")
    print("      └── training_log.csv         # Episode-by-episode training data")
    
    # Print summary statistics
    avg_score = sum(r['alignment_score'] for r in results) / len(results)
    avg_normalized = sum(r['normalized_score'] for r in results) / len(results)
    
    print(f"\n📈 Results Summary:")
    print(f"   • Processed {len(results)} sequence pairs")
    print(f"   • Average alignment score: {avg_score:.2f}")
    print(f"   • Average normalized score: {avg_normalized:.3f}")
    print(f"   • Training episodes completed: {len(episode_rewards)}")
    print(f"   • Final training reward: {episode_rewards[-1]:.2f}")
    
    return results

def demonstrate_file_formats():
    """Show examples of all input/output file formats"""
    
    print("\n📄 File Format Examples:")
    print("=" * 50)
    
    print("\n1. INPUT: FASTA file (input_sequences.fasta)")
    print("""
>sequence_1
ATGCGATCGATCGATCGTAGCTAGCTAG
>sequence_2
ATGCGATCGATCGATCGTAGCTAGCTAG
""")
    
    print("\n2. INPUT: Sequence pairs JSON (sequence_pairs.json)")
    print("""
[
  {
    "id": "pair_1",
    "seq1": "ATGCGATCGATCGATCG",
    "seq2": "ATGCGATCGATCGATCG"
  }
]
""")
    
    print("\n3. INPUT: Configuration JSON (config.json)")
    print("""
{
  "window_size": 50,
  "match_score": 2,
  "mismatch_score": -1,
  "learning_rate": 0.0001
}
""")
    
    print("\n4. OUTPUT: Alignment results JSON (alignment_results.json)")
    print("""
[
  {
    "pair_id": "pair_1",
    "seq1": "ATGCGATCGATCGATCG",
    "seq2": "ATGCGATCGATCGATCG",
    "alignment_score": 42,
    "normalized_score": 0.875,
    "alignment_path": ["Match", "Match", "Delete", "Insert"],
    "num_steps": 15
  }
]
""")
    
    print("\n5. OUTPUT: Training log CSV (training_log.csv)")
    print("""
Episode,Reward,Moving_Average
0,10.2,10.2
1,15.8,13.0
2,22.1,16.0
""")
    
    print("\n6. OUTPUT: Model file (dqnalign_trained.pth)")
    print("   Binary PyTorch checkpoint containing:")
    print("   • Neural network weights")
    print("   • Optimizer state")
    print("   • Training configuration")
    print("   • Training progress")

if __name__ == "__main__":
    print("🧬 Modern DQNalign Input/Output Demo")
    
    # Show file format examples
    demonstrate_file_formats()
    
    # Run the complete pipeline
    try:
        results = main_pipeline()
        print("\n🎉 Pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("This is expected in demo mode without running the actual training.")
    
    print("\n📖 To use in practice:")
    print("1. Prepare your DNA sequences in FASTA format")
    print("2. Create configuration JSON with your parameters")
    print("3. Run: python input_output_example.py")
    print("4. Check output/ directory for results")
