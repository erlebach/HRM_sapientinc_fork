# HRM Didactic Implementation

This is a simplified, educational implementation of the Hierarchical Reasoning Model (HRM) that focuses on clarity and understanding rather than efficiency. All casting, Pydantic, and optimization frills have been removed in favor of straightforward PyTorch code.

## Overview

The HRM model implements hierarchical reasoning through two interdependent recurrent modules:

- **High-level module (H)**: Handles slow, abstract planning
- **Low-level module (L)**: Handles rapid, detailed computations  
- **Q-module**: Decides when to halt computation using reinforcement learning

## Key Features

- **Simplified Architecture**: Clean PyTorch implementation without complex casting or optimization
- **Educational Focus**: Code is designed for understanding rather than maximum efficiency
- **Complete Pipeline**: Includes model, dataset, training, and evaluation
- **Adaptive Computation Time**: Model learns when to stop reasoning
- **Puzzle Dataset**: Built-in dataset with various reasoning tasks

## Files

- `hrm_model.py`: Core HRM model implementation
- `puzzle_dataset.py`: Simple puzzle dataset for training
- `train.py`: Training script with loss computation and optimization
- `evaluate.py`: Evaluation and inference script
- `example.py`: Simple example showing how to use the model
- `README.md`: This file

## Quick Start

### 1. Install Dependencies

```bash
pip install torch
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Create a simple puzzle dataset
- Train the HRM model for 5 epochs
- Save checkpoints to `./hrm_checkpoints/`
- Print training progress and metrics

### 3. Evaluate the Model

```bash
python evaluate.py --checkpoint ./hrm_checkpoints/best_model_epoch_5.pt --mode eval
```

### 4. Interactive Demo

```bash
python evaluate.py --checkpoint ./hrm_checkpoints/best_model_epoch_5.pt --mode demo
```

## Model Architecture

### Core Components

1. **Token Embeddings**: Convert input tokens to vectors
2. **Puzzle Embeddings**: Task-specific conditioning
3. **High-level Module**: Abstract reasoning with H cycles
4. **Low-level Module**: Detailed computation with L cycles
5. **Q-head**: Halting decision (halt vs continue)
6. **Language Model Head**: Output predictions

### Forward Pass

```python
# Single reasoning step
for l_step in range(L_cycles):
    l_state = L_module(l_state, h_state + input_embeddings)

for h_step in range(H_cycles):
    h_state = H_module(h_state, l_state)

# Q-values for halting
q_logits = Q_head(h_state[:, 0])
```

### Loss Function

The model uses a combined loss:

- **Language Modeling Loss**: Cross-entropy on target tokens
- **Q-learning Loss**: MSE on Q-values for halting decisions
- **Total Loss**: `lm_loss + 0.1 * q_loss`

## Dataset

The puzzle dataset includes:

- **Arithmetic**: Basic math problems
- **Patterns**: Sequence completion
- **Logic**: Simple logical reasoning
- **Sequences**: Number/pattern sequences
- **Word Puzzles**: Riddles and word problems

Each puzzle has:
- Input text describing the problem
- Output text with step-by-step solution
- Puzzle type and ID for conditioning

## Training

### Hyperparameters

- **Learning Rate**: 1e-4 (LM), 1e-3 (Q-learning)
- **Batch Size**: 16 (adjustable)
- **Hidden Size**: 256
- **Attention Heads**: 8
- **H/L Layers**: 2 each
- **H/L Cycles**: 2 each
- **Max Steps**: 8

### Training Process

1. **Forward Pass**: Model processes input through hierarchical reasoning
2. **Loss Computation**: Combined LM and Q-learning losses
3. **Backward Pass**: Gradients computed and clipped
4. **Optimization**: Separate optimizers for LM and Q-learning
5. **Validation**: Periodic evaluation on validation set

## Usage Examples

### Basic Model Creation

```python
from hrm_model import create_hrm_model

model = create_hrm_model(
    vocab_size=1000,
    hidden_size=256,
    num_heads=8,
    max_seq_len=64
)
```

### Training

```python
from train import train_model

trainer, test_metrics = train_model(
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-4
)
```

### Inference

```python
from evaluate import HRMEvaluator

evaluator = HRMEvaluator(model)
response = evaluator.generate_response("What is 15 + 27?")
print(response['response_text'])
```

## Key Differences from Original

This didactic implementation simplifies the original HRM in several ways:

1. **No Type Casting**: Uses standard float32 throughout
2. **No Pydantic**: Uses simple function arguments
3. **Simplified Dataset**: Basic puzzle generation instead of complex data loading
4. **Educational Focus**: Code prioritizes clarity over efficiency
5. **Self-contained**: All dependencies are minimal and standard

## Understanding the Model

### Hierarchical Reasoning

The model's key insight is that complex reasoning can be decomposed into:
- **High-level planning**: Abstract strategy and goal setting
- **Low-level execution**: Detailed step-by-step computation

### Adaptive Computation Time

The Q-module learns when the model has done enough reasoning:
- **Continue**: Model needs more computation
- **Halt**: Model is confident in its answer

### Input Injection

Unlike standard transformers, the low-level module receives both:
- Previous low-level state
- High-level state + input embeddings

This creates a hierarchical information flow.

## Limitations

This implementation is designed for education and has several limitations:

1. **Simplified Tokenization**: Basic character-based tokenization
2. **Simple Q-learning**: Basic reward structure
3. **Limited Dataset**: Small, synthetic puzzle dataset
4. **No Pre-training**: Starts from scratch
5. **Memory Inefficient**: Not optimized for large-scale training

## Extending the Model

To extend this implementation:

1. **Better Tokenization**: Use proper tokenizers (BPE, WordPiece)
2. **Larger Dataset**: Add more complex reasoning tasks
3. **Advanced Q-learning**: Implement proper reward functions
4. **Pre-training**: Add language model pre-training
5. **Efficiency**: Add gradient checkpointing, mixed precision

## References

This implementation is based on the paper:
"Hierarchical Reasoning Model for Complex Question Answering" (Sapient Inc.)

The original implementation can be found in the parent directory.
