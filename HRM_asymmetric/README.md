# Asymmetric HRM Model - System 1/System 2 Architecture

This module implements a Kahneman-inspired hierarchical reasoning model that combines fast, intuitive processing (System 1) with deliberate, analytical processing (System 2).

## Architecture Overview

### System 1 (Low-Level) - Fast, Intuitive Processing
- **Purpose**: Rapid pattern recognition and associative thinking
- **Characteristics**: 
  - Fewer transformer layers (speed over depth)
  - More processing cycles (high throughput)
  - Simplified attention mechanisms
  - No persistent working memory
  - Direct input-output mapping

### System 2 (High-Level) - Deliberate, Analytical Processing
- **Purpose**: Complex reasoning with working memory
- **Characteristics**:
  - More transformer layers (depth over speed)
  - Fewer processing cycles (but more sophisticated)
  - Advanced attention mechanisms
  - Working memory buffer
  - Attention to previous reasoning steps
  - State persistence across cycles

## Key Features

1. **Asymmetric Processing**: Different architectures and complexities for each system
2. **Working Memory**: System 2 maintains reasoning history across steps
3. **Adaptive Computation**: Q-learning based halting mechanism
4. **Parameter Efficiency**: Reuses blocks across steps and cycles
5. **Hierarchical Reasoning**: Specialized processing strategies per level

## Usage

### Basic Usage

```python
from HRM_asymmetric import create_asymmetric_hrm_model

# Create model
model = create_asymmetric_hrm_model(
    vocab_size=1000,
    hidden_size=512,
    num_heads=8,
    intermediate_size=2048,
    max_seq_len=128,
    num_puzzle_ids=100,
    L_blocks=2,        # Fewer blocks for speed
    H_blocks=4,        # More blocks for sophistication
    H_memory_size=64,
    T_cycles=2,        # Number of cycles per segment
    M_segments=16,     # Maximum number of segments
)

# Forward pass
input_ids = torch.randint(0, 1000, (batch_size, seq_len))
puzzle_ids = torch.randint(0, 100, (batch_size,))
outputs = model(input_ids, puzzle_ids)

# Access outputs
logits = outputs['logits']                    # [batch, seq, vocab]
q_halt = outputs['q_halt_logits']             # [batch]
q_continue = outputs['q_continue_logits']     # [batch]
final_L_state = outputs['final_L_state']     # [batch, seq, hidden]
final_H_state = outputs['final_H_state']     # [batch, seq, hidden]
segments_taken = outputs['segments_taken']   # scalar
```

### Advanced Usage

```python
# Custom model configuration
model = create_asymmetric_hrm_model(
    vocab_size=2000,
    hidden_size=768,
    num_heads=12,
    intermediate_size=3072,
    max_seq_len=256,
    num_puzzle_ids=200,
    L_blocks=3,             # Low-level: 3 blocks
    H_blocks=8,             # High-level: 8 blocks
    H_memory_size=128,      # Larger working memory
    T_cycles=2,             # Number of cycles per segment
    M_segments=32,          # More computation segments
)

# Reset reasoning history
model.reset_reasoning_history()

# Get model information
info = model.get_model_info()
print(f"Total parameters: {info['total_parameters']:,}")
print(f"L_module parameters: {info['L_parameters']:,}")
print(f"H_module parameters: {info['H_parameters']:,}")
```

## Architecture Details

### System 1 Module
- **FastAttention**: Simplified attention for speed
- **FastMLP**: Smaller intermediate size for efficiency
- **System1Block**: Lightweight transformer block
- **System1Module**: Orchestrates fast processing

### System 2 Module
- **WorkingMemory**: Persistent memory buffer
- **SophisticatedAttention**: Advanced attention with reasoning history
- **SwiGLU**: Gated linear unit activation
- **System2Block**: Complex transformer block with memory
- **System2Module**: Orchestrates deliberate processing

### Asymmetric HRM Model
- **Embeddings**: Token and puzzle embeddings
- **System Integration**: Coordinates System 1 and System 2
- **Adaptive Computation**: Q-learning based halting
- **Output Generation**: Language model predictions

## Testing

Run the comprehensive test suite:

```bash
cd HRM_asymmetric
python test_asymmetric_hrm.py
```

Run the example demonstration:

```bash
python example.py
```

## Key Differences from Original HRM

1. **Asymmetric Architecture**: Different complexities for each system
2. **Working Memory**: System 2 maintains reasoning history
3. **Specialized Processing**: Optimized for different reasoning types
4. **Memory Mechanisms**: Attention to previous reasoning steps
5. **Parameter Efficiency**: Better utilization of model capacity

## Performance Characteristics

- **System 1**: Fast, high-throughput processing
- **System 2**: Sophisticated, memory-enhanced reasoning
- **Adaptive**: Halts when reasoning is complete
- **Efficient**: Reuses parameters across steps and cycles
- **Scalable**: Configurable complexity and memory size

## Inspiration

This architecture is inspired by:
- **Kahneman's Dual-Process Theory**: System 1 vs System 2 thinking
- **Hierarchical Reasoning**: Different processing levels
- **Working Memory**: Cognitive science principles
- **Adaptive Computation**: Dynamic processing time
- **Parameter Efficiency**: Reuse and specialization

## Future Improvements

1. **Dynamic Memory Management**: More sophisticated memory updates
2. **Attention Mechanisms**: Cross-system attention patterns
3. **Learning Strategies**: Different learning rates per system
4. **Specialized Architectures**: Task-specific optimizations
5. **Memory Compression**: Efficient long-term memory storage
