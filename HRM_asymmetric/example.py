"""
Example usage of the Asymmetric HRM Model

This script demonstrates how to use the System 1/System 2 architecture
for hierarchical reasoning tasks.
"""

import torch
import torch.nn.functional as F
from asymmetric_hrm import AsymmetricHRMModel, create_asymmetric_hrm_model
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor


@beartype
def create_sample_puzzle_data(
    batch_size: int = 4,
    seq_len: int = 20,
    vocab_size: int = 1000,
    num_puzzle_ids: int = 10,
) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch"]]:
    """Create sample puzzle data for testing."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    puzzle_ids = torch.randint(0, num_puzzle_ids, (batch_size,))
    return input_ids, puzzle_ids


@beartype
def demonstrate_system_differences() -> None:
    """Demonstrate the differences between System 1 and System 2 processing."""
    print("=" * 60)
    print("DEMONSTRATING SYSTEM 1 vs SYSTEM 2 DIFFERENCES")
    print("=" * 60)

    # Create a small model for demonstration
    model = create_asymmetric_hrm_model(
        vocab_size=100,
        hidden_size=128,
        num_heads=4,
        intermediate_size=256,
        max_seq_len=32,
        num_puzzle_ids=5,
        s1_layers=1,  # Minimal System 1
        s1_cycles=2,
        s2_layers=2,  # More sophisticated System 2
        s2_cycles=1,
        s2_memory_size=16,
        halt_max_steps=4,
    )

    # Create sample data
    input_ids, puzzle_ids = create_sample_puzzle_data(
        batch_size=2, seq_len=10, vocab_size=100, num_puzzle_ids=5
    )

    print(f"Input shape: {input_ids.shape}")
    print(f"Puzzle IDs: {puzzle_ids}")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, puzzle_ids)

    print(f"\nOutput logits shape: {outputs['logits'].shape}")
    print(f"Steps taken: {outputs['steps_taken']}")

    # Show model architecture differences
    model_info = model.get_model_info()
    print(f"\nModel Architecture:")
    print(f"  System 1 layers: {model_info['system1_layers']}")
    print(f"  System 1 cycles: {model_info['system1_cycles']}")
    print(f"  System 1 parameters: {model_info['system1_parameters']:,}")
    print(f"  System 2 layers: {model_info['system2_layers']}")
    print(f"  System 2 cycles: {model_info['system2_cycles']}")
    print(f"  System 2 parameters: {model_info['system2_parameters']:,}")
    print(f"  System 2 memory size: {model_info['system2_memory_size']}")

    # Show Q-values (halting decisions)
    halt_logits = outputs["q_halt_logits"]
    continue_logits = outputs["q_continue_logits"]
    print(f"\nHalting Decisions:")
    print(f"  Halt logits: {halt_logits}")
    print(f"  Continue logits: {continue_logits}")
    print(f"  Should halt: {halt_logits > continue_logits}")


@beartype
def demonstrate_reasoning_history() -> None:
    """Demonstrate how System 2 maintains reasoning history."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING REASONING HISTORY IN SYSTEM 2")
    print("=" * 60)

    model = create_asymmetric_hrm_model(
        vocab_size=50,
        hidden_size=64,
        num_heads=2,
        intermediate_size=128,
        max_seq_len=16,
        num_puzzle_ids=3,
        s1_layers=1,
        s1_cycles=1,
        s2_layers=2,
        s2_cycles=1,
        s2_memory_size=8,
        halt_max_steps=3,
    )

    # Create sample data
    input_ids, puzzle_ids = create_sample_puzzle_data(
        batch_size=1, seq_len=8, vocab_size=50, num_puzzle_ids=3
    )

    print("Processing multiple steps to show reasoning history...")

    # Process multiple steps
    for step in range(3):
        print(f"\n--- Step {step + 1} ---")

        with torch.no_grad():
            outputs = model(input_ids, puzzle_ids)

        print(f"Steps taken: {outputs['steps_taken']}")
        print(
            f"System 2 has reasoning history: {model.system2.reasoning_history is not None}"
        )

        if model.system2.reasoning_history is not None:
            print(f"Reasoning history shape: {model.system2.reasoning_history.shape}")

        # Reset for next step (in practice, you'd continue with the same states)
        model.reset_reasoning_history()


@beartype
def demonstrate_adaptive_computation() -> None:
    """Demonstrate adaptive computation time based on Q-values."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING ADAPTIVE COMPUTATION TIME")
    print("=" * 60)

    model = create_asymmetric_hrm_model(
        vocab_size=200,
        hidden_size=96,
        num_heads=3,
        intermediate_size=192,
        max_seq_len=24,
        num_puzzle_ids=8,
        s1_layers=1,
        s1_cycles=2,
        s2_layers=3,
        s2_cycles=1,
        s2_memory_size=12,
        halt_max_steps=8,
    )

    # Create sample data
    input_ids, puzzle_ids = create_sample_puzzle_data(
        batch_size=3, seq_len=12, vocab_size=200, num_puzzle_ids=8
    )

    print("Testing adaptive computation with different inputs...")

    # Test in training mode (uses Q-values for halting)
    model.train()
    with torch.no_grad():
        train_outputs = model(input_ids, puzzle_ids)

    print(f"Training mode - Steps taken: {train_outputs['steps_taken']}")
    print(
        f"Q-values (halt, continue): {torch.stack([train_outputs['q_halt_logits'], train_outputs['q_continue_logits']], dim=1)}"
    )

    # Test in eval mode (uses max steps)
    model.eval()
    with torch.no_grad():
        eval_outputs = model(input_ids, puzzle_ids)

    print(f"Eval mode - Steps taken: {eval_outputs['steps_taken']}")


@beartype
def demonstrate_parameter_efficiency() -> None:
    """Demonstrate parameter efficiency compared to standard transformer."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING PARAMETER EFFICIENCY")
    print("=" * 60)

    # Create Asymmetric HRM
    asymmetric_model = create_asymmetric_hrm_model(
        vocab_size=1000,
        hidden_size=512,
        num_heads=8,
        intermediate_size=2048,
        max_seq_len=128,
        num_puzzle_ids=100,
        s1_layers=2,
        s1_cycles=4,
        s2_layers=4,
        s2_cycles=2,
        s2_memory_size=64,
        halt_max_steps=16,
    )

    # Create equivalent standard transformer for comparison
    from torch.nn import TransformerEncoder, TransformerEncoderLayer

    standard_transformer = TransformerEncoder(
        TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, batch_first=True
        ),
        num_layers=6,  # Total layers = s1_layers + s2_layers
    )

    # Count parameters
    asymmetric_params = sum(p.numel() for p in asymmetric_model.parameters())
    standard_params = sum(p.numel() for p in standard_transformer.parameters())

    print(f"Asymmetric HRM parameters: {asymmetric_params:,}")
    print(f"Standard Transformer parameters: {standard_params:,}")
    print(f"Parameter ratio: {asymmetric_params / standard_params:.2f}x")

    # Show the advantage of hierarchical processing
    max_computation = 16 * 4 * 2 + 16 * 2 * 4  # steps * cycles * layers
    print(f"\nMaximum computation (steps × cycles × layers):")
    print(f"  System 1: {16 * 4 * 2} = {16 * 4 * 2:,} block executions")
    print(f"  System 2: {16 * 2 * 4} = {16 * 2 * 4:,} block executions")
    print(f"  Total: {max_computation:,} block executions")
    print(f"  Equivalent to {max_computation / 6:.1f}x deeper standard transformer")


def main() -> None:
    """Main demonstration function."""
    print("ASYMMETRIC HRM MODEL DEMONSTRATION")
    print("System 1/System 2 Architecture for Hierarchical Reasoning")
    print("Inspired by Kahneman's dual-process theory")

    # Run demonstrations
    demonstrate_system_differences()
    demonstrate_reasoning_history()
    demonstrate_adaptive_computation()
    demonstrate_parameter_efficiency()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey advantages of the Asymmetric HRM:")
    print("1. System 1: Fast, intuitive processing for pattern recognition")
    print("2. System 2: Deliberate, analytical processing with working memory")
    print("3. Adaptive computation: Halts when reasoning is complete")
    print("4. Parameter efficiency: Reuses blocks across steps and cycles")
    print("5. Hierarchical reasoning: Different processing strategies per level")


if __name__ == "__main__":
    main()
