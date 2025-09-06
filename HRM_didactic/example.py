"""
Simple Example Script for HRM Model

This script demonstrates how to use the HRM model for basic inference
and training without going through the full training pipeline.
"""

import torch
import torch.nn.functional as F
from hrm_model import create_hrm_model
from puzzle_dataset import PuzzleDataset
from train import HRMTrainer


def basic_model_example():
    """Demonstrate basic model creation and forward pass."""
    print("=== Basic Model Example ===")

    # Create a small model for demonstration
    model = create_hrm_model(
        vocab_size=1000,
        hidden_size=128,  # Small for demo
        num_heads=4,
        intermediate_size=512,
        max_seq_len=32,
        num_puzzle_ids=10,
        h_layers=1,
        l_layers=1,
        h_cycles=1,
        l_cycles=1,
        halt_max_steps=4,
    )

    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Create dummy inputs
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    puzzle_ids = torch.randint(0, 10, (batch_size,))

    print(f"Input shape: {input_ids.shape}")
    print(f"Puzzle IDs: {puzzle_ids.tolist()}")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, puzzle_ids)

    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Q-values shape: {outputs['q_halt_logits'].shape}")
    print(f"Steps taken: {outputs['steps_taken']}")
    print(f"Q-halt mean: {outputs['q_halt_logits'].mean().item():.3f}")
    print(f"Q-continue mean: {outputs['q_continue_logits'].mean().item():.3f}")
    print()


def dataset_example():
    """Demonstrate the puzzle dataset."""
    print("=== Dataset Example ===")

    # Create a small dataset
    dataset = PuzzleDataset(num_samples=5, max_seq_len=32)

    print(f"Dataset size: {len(dataset)}")

    # Show a few examples
    for i in range(3):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Input shape: {sample['input_ids'].shape}")
        print(f"  Target shape: {sample['target_ids'].shape}")
        print(f"  Puzzle ID: {sample['puzzle_ids'].item()}")
        print(f"  Puzzle type: {sample['puzzle_type'].item()}")

        # Show some tokens (simplified)
        input_tokens = sample["input_ids"][:10].tolist()  # First 10 tokens
        target_tokens = sample["target_ids"][:10].tolist()
        print(f"  Input tokens: {input_tokens}")
        print(f"  Target tokens: {target_tokens}")
    print()


def training_example():
    """Demonstrate basic training setup."""
    print("=== Training Example ===")

    # Create model and dataset
    model = create_hrm_model(
        vocab_size=1000,
        hidden_size=64,  # Very small for demo
        num_heads=4,
        intermediate_size=256,
        max_seq_len=16,
        num_puzzle_ids=10,
        h_layers=1,
        l_layers=1,
        h_cycles=1,
        l_cycles=1,
        halt_max_steps=3,
    )

    dataset = PuzzleDataset(num_samples=20, max_seq_len=16)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # Create trainer
    trainer = HRMTrainer(model, learning_rate=1e-3, device="cpu")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches: {len(dataloader)}")

    # Train for a few steps
    print("\nTraining for 3 steps...")
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break

        metrics = trainer.train_step(batch)
        print(
            f"Step {i+1}: Loss = {metrics['total_loss']:.4f}, "
            f"Acc = {metrics['accuracy']:.3f}, "
            f"Steps = {metrics['avg_steps']:.1f}"
        )

    print()


def inference_example():
    """Demonstrate inference on a simple puzzle."""
    print("=== Inference Example ===")

    # Create model
    model = create_hrm_model(
        vocab_size=1000,
        hidden_size=128,
        num_heads=4,
        intermediate_size=512,
        max_seq_len=32,
        num_puzzle_ids=10,
        h_layers=2,
        l_layers=2,
        h_cycles=2,
        l_cycles=2,
        halt_max_steps=6,
    )

    # Create a simple puzzle
    puzzle_text = "What is 15 + 27?"
    puzzle_id = 0

    # Simple tokenization (in practice, use proper tokenizer)
    input_tokens = (
        [1]
        + [ord(c) % 1000 + 10 for c in puzzle_text if c.isalnum() or c in " .,!?"]
        + [2]
    )
    input_ids = torch.tensor([input_tokens], dtype=torch.long)
    puzzle_ids = torch.tensor([puzzle_id], dtype=torch.long)

    print(f"Puzzle: {puzzle_text}")
    print(f"Input tokens: {input_tokens}")

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, puzzle_ids)

    # Get predictions
    logits = outputs["logits"]
    predictions = torch.argmax(logits, dim=-1)

    print(f"Output logits shape: {logits.shape}")
    print(f"Predictions: {predictions[0].tolist()}")
    print(f"Steps taken: {outputs['steps_taken']}")
    print(f"Q-halt: {outputs['q_halt_logits'][0].item():.3f}")
    print(f"Q-continue: {outputs['q_continue_logits'][0].item():.3f}")
    print(f"Halt probability: {torch.sigmoid(outputs['q_halt_logits'][0]).item():.3f}")
    print()


def model_analysis():
    """Analyze the model architecture."""
    print("=== Model Analysis ===")

    model = create_hrm_model(
        vocab_size=1000,
        hidden_size=256,
        num_heads=8,
        intermediate_size=1024,
        max_seq_len=64,
        num_puzzle_ids=100,
        h_layers=4,
        l_layers=4,
        h_cycles=2,
        l_cycles=2,
        halt_max_steps=16,
    )

    # Count parameters by component
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {total_params:,}")
    print("\nParameter breakdown:")

    for name, param in model.named_parameters():
        print(f"  {name}: {param.numel():,} ({param.numel()/total_params*100:.1f}%)")

    # Analyze model structure
    print(f"\nModel structure:")
    print(f"  Hidden size: {model.hidden_size}")
    print(f"  H cycles: {model.h_cycles}")
    print(f"  L cycles: {model.l_cycles}")
    print(f"  H layers: {len(model.h_level.layers)}")
    print(f"  L layers: {len(model.l_level.layers)}")
    print(f"  Max steps: {model.halt_max_steps}")

    # Memory usage estimate
    param_memory = total_params * 4 / (1024**2)  # 4 bytes per float32
    print(f"\nEstimated memory usage:")
    print(f"  Parameters: {param_memory:.1f} MB")
    print(f"  Activations (batch=32): ~{param_memory * 2:.1f} MB")
    print()


if __name__ == "__main__":
    print("HRM Didactic Implementation - Examples")
    print("=" * 50)

    # Run all examples
    basic_model_example()
    dataset_example()
    training_example()
    inference_example()
    model_analysis()

    print("All examples completed!")
    print("\nTo run the full training pipeline:")
    print("  python train.py")
    print("\nTo evaluate a trained model:")
    print(
        "  python evaluate.py --checkpoint ./hrm_checkpoints/best_model_epoch_5.pt --mode eval"
    )
