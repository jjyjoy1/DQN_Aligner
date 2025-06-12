# DQNalign: Reference-Based Sequence Aligner
This is my first program using reinforcement learning algorithm, the program combine deep reinforcement learning (specifically Deep Q-Networks) with the x-drop algorithm could create a more efficient local DNA sequence alignment method that:

Performs alignment with linear computational complexity
Reduces dependence on human-designed heuristics
Maintains similar accuracy to conventional methods while being computationally more efficient

##Algorithm: DQN x-drop
The algorithm consists of two main components:
1. DQNalign Foundation
DQNalign is an algorithm that learns and performs sequence alignment through deep reinforcement learning. The learned Deep Q-network (DQN) observes only parts of predetermined length (window size) of sequences and continuously selects the optimal alignment direction to proceed. Local Alignment of DNA Sequence Based on Deep Reinforcement Learning - PMC
Key Elements:

State: Sub-sequence pairs
Action: Alignment direction (forward, deletion, insertion)
Reward: Alignment scoring system (match, mismatch, gap penalties)

2. DQN x-drop Integration
The proposed DQN x-drop algorithm performs local alignment by repeatedly observing the subsequences and selecting the next alignment direction until the x-drop algorithm terminates the DQNalign algorithm. Local Alignment of DNA Sequence Based on Deep Reinforcement Learning - PMC
Technical Features:

Uses a faster DDDQN (Dueling Double Deep Q-Network) structure with separable convolutional layers
Complexity reduced about 1/9 to 1/26 times compared to the DDDQN structure Local Alignment of DNA Sequence Based on Deep Reinforcement Learning - PMC
Employs Model-Agnostic Meta-Learning (MAML) for better generalization to real sequences
Window-based sliding approach that makes computational complexity independent of sequence length

#Key Improvements Made:
##1. Modern Technology Stack

PyTorch 2.0+ instead of TensorFlow 1.x (much more flexible and current)
Python 3.8+ with full type hints for better code quality
GPU acceleration with automatic CUDA detection
Modern dependencies and package management

##2. Advanced Deep RL Algorithms

Rainbow DQN implementation with multiple improvements:

Double DQN: Reduces overestimation bias
Dueling DQN: Better value function estimation
Noisy Networks: Parameter space exploration without ε-greedy
Soft target updates: More stable training

##3. Better Software Engineering

Dataclass configuration system for easy parameter management
Modular design with clear separation of concerns
Comprehensive documentation and type hints
Robust error handling and logging
Model checkpointing and resumable training

##4. Enhanced Performance

Efficient state representation with convolutional encoding
Vectorized operations for faster computation
Memory-optimized replay buffer
Gradient clipping for training stability

##5. Extensibility & Research Features

Configurable reward systems for different alignment scenarios
Batch evaluation capabilities for systematic testing
Multiple exploration strategies (ε-greedy + noisy networks)
Easy integration with existing bioinformatics tools

What This Enables:

Faster Training: Modern optimizations and GPU acceleration
Better Performance: Advanced RL algorithms with proven improvements
Easier Research: Modular design for experimenting with new ideas
Production Ready: Robust code suitable for real applications
Maintainable: Clean, documented code that's easy to understand and modify

##Potential Extensions:

Transformer-based sequence encoding for better representation learning
Multi-agent RL for multiple sequence alignment
Protein sequence support beyond just DNA/RNA
Distributed training for large-scale experiments
Integration with existing tools (BLAST, ClustalW benchmarking)
## Installation

```bash
pip install -r requirements_reference.txt

# For BAM file processing, you may also need samtools
# On Ubuntu/Debian:
sudo apt-get install samtools

# On macOS:
brew install samtools

# On Windows (via conda):
conda install -c bioconda samtools
```

## Input Files

### 1. Reference FASTA file
```fasta
# reference.fasta
>chromosome_1
ATGCGATCGATCGATCGTAGCTAGCTAGATGCGATCGATCGATCGTAGCTAGCTAG...
>chromosome_2  
ATGCGATCGATCGATCGTAGCTAGCTAGATGCGATCGATCGATCGTAGCTAGCTAG...
```

### 2. Query sequence files (FASTA or FASTQ)
```fasta
# reads.fasta
>read_001
ATGCGATCGATCGATCGTAGCTAG
>read_002
ATGCGATCGATCGATCGTAGCTAGCTAG
>read_003
ATGCGATCGATCGATCGTAGCT
```

```fastq
# reads.fastq
@read_001
ATGCGATCGATCGATCGTAGCTAG
+
IIIIIIIIIIIIIIIIIIIIIII
@read_002
ATGCGATCGATCGATCGTAGCTAGCTAG
+
IIIIIIIIIIIIIIIIIIIIIIIIIII
```

## Output Files

### 1. BAM files (one per input file)
```
output/
├── reads_aligned.bam           # Aligned reads in BAM format
├── reads_aligned_sorted.bam    # Sorted BAM file
├── reads_aligned_sorted.bam.bai # BAM index file
└── sample2_aligned.bam         # Additional input files
```

### 2. Training logs and models
```
models/
└── reference_aligner.pth       # Trained model weights

logs/
├── training.log               # Training progress
└── alignment_stats.json       # Alignment statistics
```

## Usage Examples

### 1. Basic Usage (Command Line)

```bash
# Basic alignment with single input file
python dqnalign_reference_bam.py \
    --reference reference.fasta \
    --input reads.fasta \
    --output-dir output/

# Multiple input files
python dqnalign_reference_bam.py \
    --reference reference.fasta \
    --input reads1.fasta reads2.fastq sample3.fasta \
    --output-dir alignments/ \
    --num-processes 4

# Use pre-trained model
python dqnalign_reference_bam.py \
    --reference reference.fasta \
    --input reads.fasta \
    --model pretrained_model.pth \
    --output-dir output/

# Train new model first
python dqnalign_reference_bam.py \
    --reference reference.fasta \
    --input reads.fasta \
    --train \
    --output-dir output/
```

### 2. Python API Usage

```python
#!/usr/bin/env python3
"""
Python API usage examples for reference-based DQNalign
"""

from dqnalign_reference_bam import (
    DQNAlignerConfig, DQNReferenceAligner, 
    batch_process_files, create_bam_file
)
import pysam
from pathlib import Path

def example_1_basic_alignment():
    """Basic alignment example"""
    print("=== Example 1: Basic Alignment ===")
    
    # Configure aligner
    config = DQNAlignerConfig(
        reference_file="reference.fasta",
        input_files=["reads.fasta"],
        output_dir="output",
        window_size=64,
        num_processes=2
    )
    
    # Process files
    output_files = batch_process_files(config)
    
    print(f"Generated BAM files: {output_files}")
    return output_files

def example_2_custom_scoring():
    """Custom scoring parameters"""
    print("=== Example 2: Custom Scoring ===")
    
    config = DQNAlignerConfig(
        reference_file="reference.fasta",
        input_files=["reads.fastq"],
        output_dir="custom_output",
        
        # Custom scoring
        match_score=3,           # Higher reward for matches
        mismatch_score=-2,       # Penalty for mismatches
        gap_open_score=-5,       # Gap opening penalty
        gap_extend_score=-1,     # Gap extension penalty
        min_alignment_score=25,  # Minimum score to report
        
        # Network parameters
        window_size=128,         # Larger window for better context
        hidden_size=1024,        # Larger network
        
        # Training parameters
        num_episodes=20000,      # More training episodes
        learning_rate=1e-5       # Lower learning rate
    )
    
    output_files = batch_process_files(config)
    print(f"Custom alignment completed: {output_files}")

def example_3_analyze_bam_output():
    """Analyze BAM output files"""
    print("=== Example 3: Analyze BAM Output ===")
    
    bam_file = "output/reads_aligned_sorted.bam"
    
    if not Path(bam_file).exists():
        print(f"BAM file {bam_file} not found. Run alignment first.")
        return
    
    # Read BAM file
    with pysam.AlignmentFile(bam_file, "rb") as bam:
        total_reads = 0
        mapped_reads = 0
        total_score = 0
        
        for read in bam:
            total_reads += 1
            
            if not read.is_unmapped:
                mapped_reads += 1
                total_score += read.mapping_quality
        
        mapping_rate = mapped_reads / total_reads if total_reads > 0 else 0
        avg_score = total_score / mapped_reads if mapped_reads > 0 else 0
        
        print(f"BAM Analysis Results:")
        print(f"  Total reads: {total_reads}")
        print(f"  Mapped reads: {mapped_reads}")
        print(f"  Mapping rate: {mapping_rate:.2%}")
        print(f"  Average mapping quality: {avg_score:.1f}")

def example_4_batch_processing():
    """Batch process multiple files"""
    print("=== Example 4: Batch Processing ===")
    
    # Create sample input files
    sample_files = []
    for i in range(3):
        filename = f"sample_{i+1}.fasta"
        sample_files.append(filename)
        
        # Create dummy sequences
        with open(filename, "w") as f:
            for j in range(10):
                f.write(f">read_{i+1}_{j+1}\n")
                f.write("ATGCGATCGATCGATCGTAGCTAGCTAG\n")
    
    # Configure for batch processing
    config = DQNAlignerConfig(
        reference_file="reference.fasta",
        input_files=sample_files,
        output_dir="batch_output",
        num_processes=4,  # Parallel processing
        chunk_size=500    # Process in chunks
    )
    
    # Process all files
    output_files = batch_process_files(config)
    
    print(f"Batch processing completed:")
    for input_file, output_file in zip(sample_files, output_files):
        print(f"  {input_file} -> {output_file}")
    
    # Cleanup
    for filename in sample_files:
        Path(filename).unlink(missing_ok=True)

def example_5_advanced_configuration():
    """Advanced configuration example"""
    print("=== Example 5: Advanced Configuration ===")
    
    config = DQNAlignerConfig(
        # Input/Output
        reference_file="large_genome.fasta",
        input_files=["long_reads.fastq", "short_reads.fasta"],
        output_dir="advanced_output",
        
        # Alignment parameters
        window_size=256,           # Large window for long reads
        overlap_size=64,           # Overlap between windows
        min_alignment_score=50,    # Strict alignment threshold
        max_reference_chunk=5000,  # Handle large references
        
        # Advanced scoring
        match_score=5,
        mismatch_score=-2,
        gap_open_score=-10,
        gap_extend_score=-2,
        
        # Rainbow DQN parameters
        hidden_size=2048,
        num_layers=6,
        dropout=0.2,
        
        # Training parameters
        learning_rate=1e-5,
        batch_size=256,
        buffer_size=1000000,
        gamma=0.995,
        
        # Rainbow features
        double_dqn=True,
        dueling_dqn=True,
        noisy_networks=True,
        epsilon_start=0.95,
        epsilon_end=0.02,
        epsilon_decay=200000,
        
        # Processing
        num_processes=8,
        chunk_size=2000,
        
        device="cuda"  # GPU acceleration
    )
    
    print("Advanced configuration created")
    print(f"Using {config.num_processes} processes")
    print(f"Window size: {config.window_size}")
    print(f"Network size: {config.hidden_size} hidden units")

def example_6_model_management():
    """Model training and management"""
    print("=== Example 6: Model Management ===")
    
    from dqnalign_reference_bam import train_reference_aligner
    
    # Training configuration
    config = DQNAlignerConfig(
        reference_file="reference.fasta",
        num_episodes=5000,
        learning_rate=1e-4,
        window_size=64
    )
    
    # Train new model
    print("Training new model...")
    aligner, rewards = train_reference_aligner(
        config, 
        save_path="models/custom_aligner.pth"
    )
    
    print(f"Training completed. Final reward: {rewards[-1]:.2f}")
    
    # Load and use trained model
    new_aligner = DQNReferenceAligner(config)
    new_aligner.load_model("models/custom_aligner.pth")
    
    print("Model loaded successfully")
    
    # Test alignment
    test_query = "ATGCGATCGATCGATCGTAGCTAGCTAG"
    result = new_aligner.align_sequence(test_query)
    
    print(f"Test alignment:")
    print(f"  Score: {result.alignment_score}")
    print(f"  CIGAR: {result.cigar_string}")
    print(f"  Mapped: {result.is_mapped}")

def create_sample_data():
    """Create sample reference and read files for testing"""
    print("Creating sample data files...")
    
    # Create reference file
    reference_seq = (
        "ATGCGATCGATCGATCGTAGCTAGCTAGATGCGATCGATCGATCGTAGCTAGCTAG" * 20
    )  # 1080 bp reference
    
    with open("reference.fasta", "w") as f:
        f.write(">chr1\n")
        f.write(reference_seq + "\n")
    
    # Create read files
    with open("reads.fasta", "w") as f:
        for i in range(50):
            start = i * 10
            read_seq = reference_seq[start:start+30]  # 30bp reads
            f.write(f">read_{i+1}\n")
            f.write(read_seq + "\n")
    
    with open("reads.fastq", "w") as f:
        for i in range(30):
            start = i * 15
            read_seq = reference_seq[start:start+25]  # 25bp reads
            f.write(f"@read_{i+1}\n")
            f.write(read_seq + "\n")
            f.write("+\n")
            f.write("I" * len(read_seq) + "\n")
    
    print("Sample data created:")
    print("  • reference.fasta (1080 bp)")
    print("  • reads.fasta (50 reads)")
    print("  • reads.fastq (30 reads)")

def main():
    """Run all examples"""
    print("🧬 DQNalign Reference-Based Aligner Examples")
    print("=" * 60)
    
    # Create sample data
    create_sample_data()
    
    # Run examples
    examples = [
        example_1_basic_alignment,
        example_2_custom_scoring,
        example_3_analyze_bam_output,
        example_4_batch_processing,
        example_5_advanced_configuration,
        example_6_model_management
    ]
    
    for i, example_func in enumerate(examples, 1):
        print(f"\n{'='*60}")
        try:
            example_func()
        except Exception as e:
            print(f"Example {i} failed: {e}")
        print("="*60)
    
    print("\n🎉 All examples completed!")
    print("\nGenerated files:")
    for file in Path(".").glob("**/*.bam"):
        print(f"  📄 {file}")
    for file in Path(".").glob("**/*.pth"):
        print(f"  🤖 {file}")

if __name__ == "__main__":
    main()
```

## Comparison with Traditional Aligners

| Feature | BWA | Bowtie2 | STAR | DQNalign |
|---------|-----|---------|------|----------|
| Algorithm | Burrows-Wheeler | Burrows-Wheeler | Suffix Array | Deep RL |
| Speed | Fast | Fast | Very Fast | Moderate |
| Accuracy | High | High | Very High | Learning-based |
| Memory | Moderate | Low | High | Moderate |
| Adaptability | Fixed | Fixed | Fixed | **Adaptive** |
| Training | None | None | None | **Self-improving** |
| Custom Scoring | Limited | Limited | Limited | **Fully customizable** |

## Workflow Integration

### 1. Standard Bioinformatics Pipeline
```bash
# 1. Index reference (traditional tools)
bwa index reference.fasta

# 2. Align with DQNalign
python dqnalign_reference_bam.py \
    --reference reference.fasta \
    --input reads.fastq \
    --output-dir dqn_output/

# 3. Compare with BWA
bwa mem reference.fasta reads.fastq > bwa_output.sam
samtools view -bS bwa_output.sam > bwa_output.bam

# 4. Analyze differences
samtools flagstat dqn_output/reads_aligned_sorted.bam
samtools flagstat bwa_output.bam
```

### 2. Integration with Galaxy/Nextflow
```nextflow
// Nextflow pipeline example
process DQN_ALIGN {
    input:
    path reference
    path reads
    
    output:
    path "*.bam"
    
    script:
    """
    python dqnalign_reference_bam.py \
        --reference ${reference} \
        --input ${reads} \
        --output-dir .
    """
}
```

## Performance Considerations

### 1. Memory Usage
- **Reference sequences**: Loaded into memory
- **Model size**: ~50-200MB depending on configuration
- **Batch processing**: Configurable chunk sizes

### 2. Speed Optimization
```python
# Fast configuration for large datasets
config = DQNAlignerConfig(
    window_size=32,        # Smaller windows
    num_processes=16,      # More parallel processes
    chunk_size=5000,       # Larger chunks
    hidden_size=256,       # Smaller network
    device="cuda"          # GPU acceleration
)
```

### 3. Accuracy Tuning
```python
# High-accuracy configuration
config = DQNAlignerConfig(
    window_size=128,       # Larger context
    num_episodes=100000,   # More training
    hidden_size=1024,      # Larger network
    match_score=5,         # Higher match reward
    min_alignment_score=40 # Stricter threshold
)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```python
   config.device = "cpu"
   config.batch_size = 32
   ```

2. **Slow alignment**
   ```python
   config.num_processes = multiprocessing.cpu_count()
   config.window_size = 32
   ```

3. **Poor alignment quality**
   ```python
   config.num_episodes = 50000  # More training
   config.window_size = 128     # Better context
   ```

4. **BAM file issues**
   ```bash
   # Check BAM file integrity
   samtools quickcheck output/*.bam
   
   # Repair if needed
   samtools sort -o fixed.bam broken.bam
   samtools index fixed.bam
   ```

This reference-based implementation transforms DQNalign into a practical tool that can integrate with standard bioinformatics workflows while maintaining the adaptive learning advantages of deep reinforcement learning.
