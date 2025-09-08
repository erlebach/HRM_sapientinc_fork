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
    print(f"\n==> Create sample puzzle data...")
    print(f"input_ids shape: {input_ids.shape}")
    print(f"puzzle_ids shape: {puzzle_ids.shape}")
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
        L_hidden_size=128,
        H_hidden_size=128,
        num_heads=4,
        intermediate_size=256,
        max_seq_len=32,
        num_puzzle_ids=5,
        L_blocks=1,  # Minimal System 1
        H_blocks=2,  # More sophisticated System 2
        H_memory_size=16,
        T_cycles=2,
        M_segments=4,
    )

    # Create sample data
    input_ids, puzzle_ids = create_sample_puzzle_data(
        batch_size=3, seq_len=10, vocab_size=100, num_puzzle_ids=5
    )

    print("\n==> Return from create_sample_puzzle_data ...")

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, puzzle_ids)

    print(f"\nOutput logits shape: {outputs['logits'].shape}")
    print(f"Segments taken: {outputs['segments_taken']}")

    # Show model architecture differences
    model_info = model.get_model_info()
    print(f"\nModel Architecture:")
    print(f"  L_blocks: {model_info['L_blocks']}")
    print(f"  H_blocks: {model_info['H_blocks']}")
    print(f"  L_parameters: {model_info['L_parameters']:,}")
    print(f"  H_parameters: {model_info['H_parameters']:,}")
    print(f"  T_cycles: {model_info['T_cycles']}")
    print(f"  M_segments: {model_info['M_segments']}")
    print(f"  H_memory_size: {model_info['H_memory_size']}")

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
        L_hidden_size=64,
        H_hidden_size=64,
        num_heads=2,
        intermediate_size=128,
        max_seq_len=16,
        num_puzzle_ids=3,
        L_blocks=1,
        H_blocks=2,
        H_memory_size=8,
        T_cycles=1,
        M_segments=3,
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

        print(f"Segments taken: {outputs['segments_taken']}")
        print(
            f"System 2 has reasoning history: {model.H_module.reasoning_history is not None}"
        )

        if model.H_module.reasoning_history is not None:
            print(f"Reasoning history shape: {model.H_module.reasoning_history.shape}")

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
        L_hidden_size=96,
        H_hidden_size=96,
        num_heads=3,
        intermediate_size=192,
        max_seq_len=24,
        num_puzzle_ids=8,
        L_blocks=1,
        H_blocks=3,
        H_memory_size=12,
        T_cycles=2,
        M_segments=8,
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

    print(f"Training mode - Segments taken: {train_outputs['segments_taken']}")
    print(
        f"Q-values (halt, continue): {torch.stack([train_outputs['q_halt_logits'], train_outputs['q_continue_logits']], dim=1)}"
    )

    # Test in eval mode (uses max segments)
    model.eval()
    with torch.no_grad():
        eval_outputs = model(input_ids, puzzle_ids)

    print(f"Eval mode - Segments taken: {eval_outputs['segments_taken']}")


@beartype
def demonstrate_parameter_efficiency() -> None:
    """Demonstrate parameter efficiency compared to standard transformer."""
    print("\n" + "=" * 60)
    print("DEMONSTRATING PARAMETER EFFICIENCY")
    print("=" * 60)

    # Create Asymmetric HRM
    asymmetric_model = create_asymmetric_hrm_model(
        vocab_size=1000,
        L_hidden_size=512,
        H_hidden_size=512,
        num_heads=8,
        intermediate_size=2048,
        max_seq_len=128,
        num_puzzle_ids=100,
        L_blocks=2,
        H_blocks=4,
        H_memory_size=64,
        T_cycles=2,
        M_segments=16,
    )

    # Create equivalent standard transformer for comparison
    from torch.nn import TransformerEncoder, TransformerEncoderLayer

    standard_transformer = TransformerEncoder(
        TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, batch_first=True
        ),
        num_layers=6,  # Total layers = L_blocks + H_blocks
    )

    # Count parameters
    asymmetric_params = sum(p.numel() for p in asymmetric_model.parameters())
    standard_params = sum(p.numel() for p in standard_transformer.parameters())

    print(f"Asymmetric HRM parameters: {asymmetric_params:,}")
    print(f"Standard Transformer parameters: {standard_params:,}")
    print(f"Parameter ratio: {asymmetric_params / standard_params:.2f}x")

    # Show the advantage of hierarchical processing
    max_computation = 16 * 2 * 2 + 16 * 2 * 4  # segments * T_cycles * blocks
    print(f"\nMaximum computation (segments × T_cycles × blocks):")
    print(f"  L_module: {16 * 2 * 2} = {16 * 2 * 2:,} block executions")
    print(f"  H_module: {16 * 2 * 4} = {16 * 2 * 4:,} block executions")
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
