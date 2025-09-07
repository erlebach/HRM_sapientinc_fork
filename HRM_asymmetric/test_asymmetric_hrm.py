"""
Tests for the Asymmetric HRM Model

This module contains comprehensive tests for the System 1/System 2 architecture,
including structure tests, functionality tests, and integration tests.
"""

import torch
import torch.nn as nn
from asymmetric_hrm import AsymmetricHRMModel, create_asymmetric_hrm_model
from beartype import beartype
from jaxtyping import Float, Int
from system1_module import System1Module
from system2_module import System2Module
from torch import Tensor


def test_system1_module_structure() -> None:
    """Test System1Module structure and basic functionality."""
    print("Testing System1Module structure...")

    # Create System1Module
    model = System1Module(
        hidden_size=128,
        num_heads=4,
        intermediate_size=256,
        num_layers=2,
        max_seq_len=32,
    )

    # Test basic properties
    assert isinstance(model, nn.Module)
    assert hasattr(model, "layers")
    assert len(model.layers) == 1  # Should be num_layers // 2
    assert model.hidden_size == 128
    assert model.num_layers == 2

    # Test forward pass
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, 128)
    input_injection = torch.randn(batch_size, seq_len, 128)

    output = model(hidden_states, input_injection)
    assert output.shape == hidden_states.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    print("✓ System1Module structure test passed")


def test_system2_module_structure() -> None:
    """Test System2Module structure and basic functionality."""
    print("Testing System2Module structure...")

    # Create System2Module
    model = System2Module(
        hidden_size=128,
        num_heads=4,
        intermediate_size=256,
        num_layers=3,
        max_seq_len=32,
        memory_size=16,
    )

    # Test basic properties
    assert isinstance(model, nn.Module)
    assert hasattr(model, "layers")
    assert len(model.layers) == 3  # Should be max(2, num_layers)
    assert model.hidden_size == 128
    assert model.num_layers == 3
    assert model.memory_size == 16

    # Test forward pass
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, 128)
    input_injection = torch.randn(batch_size, seq_len, 128)

    output = model(hidden_states, input_injection)
    assert output.shape == hidden_states.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

    # Test reasoning history
    model.eval()  # Ensure we start in eval mode
    model.reset_reasoning_history()  # Clear any existing history
    assert model.reasoning_history is None

    model.train()
    output = model(hidden_states, input_injection)
    assert model.reasoning_history is not None
    assert model.reasoning_history.shape[0] == batch_size

    # Test history reset
    model.reset_reasoning_history()
    assert model.reasoning_history is None

    print("✓ System2Module structure test passed")


def test_asymmetric_hrm_structure() -> None:
    """Test AsymmetricHRMModel structure and basic functionality."""
    print("Testing AsymmetricHRMModel structure...")

    # Create AsymmetricHRMModel
    model = create_asymmetric_hrm_model(
        vocab_size=100,
        hidden_size=128,
        num_heads=4,
        intermediate_size=256,
        max_seq_len=32,
        num_puzzle_ids=10,
        L_blocks=2,
        H_blocks=3,
        H_memory_size=16,
        T_cycles=2,
        M_segments=4,
    )

    # Test basic properties
    assert isinstance(model, nn.Module)
    assert hasattr(model, "L_module")
    assert hasattr(model, "H_module")
    assert hasattr(model, "token_embedding")
    assert hasattr(model, "puzzle_embedding")
    assert hasattr(model, "lm_head")
    assert hasattr(model, "q_head")

    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    puzzle_ids = torch.randint(0, 10, (batch_size,))

    outputs = model(input_ids, puzzle_ids)

    # Test output structure
    assert "logits" in outputs
    assert "q_halt_logits" in outputs
    assert "q_continue_logits" in outputs
    assert "final_L_state" in outputs
    assert "final_H_state" in outputs
    assert "segments_taken" in outputs

    # Test output shapes
    assert outputs["logits"].shape == (batch_size, seq_len, 100)
    assert outputs["q_halt_logits"].shape == (batch_size,)
    assert outputs["q_continue_logits"].shape == (batch_size,)
    assert outputs["final_L_state"].shape == (batch_size, seq_len, 128)
    assert outputs["final_H_state"].shape == (batch_size, seq_len, 128)
    assert outputs["segments_taken"].shape == ()

    # Test no NaN or Inf values
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            assert not torch.isnan(value).any(), f"NaN found in {key}"
            assert not torch.isinf(value).any(), f"Inf found in {key}"

    print("✓ AsymmetricHRMModel structure test passed")


def test_parameter_efficiency() -> None:
    """Test that the model is parameter efficient."""
    print("Testing parameter efficiency...")

    # Create Asymmetric HRM
    asymmetric_model = create_asymmetric_hrm_model(
        vocab_size=1000,
        hidden_size=512,
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

    # Create equivalent standard transformer
    from torch.nn import TransformerEncoder, TransformerEncoderLayer

    standard_transformer = TransformerEncoder(
        TransformerEncoderLayer(
            d_model=512, nhead=8, dim_feedforward=2048, batch_first=True
        ),
        num_layers=6,  # L_blocks + H_blocks
    )

    # Count parameters
    asymmetric_params = sum(p.numel() for p in asymmetric_model.parameters())
    standard_params = sum(p.numel() for p in standard_transformer.parameters())

    # Asymmetric model should have similar or fewer parameters
    assert (
        asymmetric_params <= standard_params * 1.5
    ), f"Too many parameters: {asymmetric_params} vs {standard_params}"

    print(
        f"✓ Parameter efficiency test passed: {asymmetric_params:,} vs {standard_params:,}"
    )


def test_adaptive_computation() -> None:
    """Test adaptive computation time behavior."""
    print("Testing adaptive computation...")

    model = create_asymmetric_hrm_model(
        vocab_size=100,
        hidden_size=64,
        num_heads=2,
        intermediate_size=128,
        max_seq_len=16,
        num_puzzle_ids=5,
        L_blocks=1,
        H_blocks=2,
        H_memory_size=8,
        T_cycles=1,
        M_segments=4,
    )

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    puzzle_ids = torch.randint(0, 5, (batch_size,))

    # Test training mode (uses Q-values)
    model.train()
    train_outputs = model(input_ids, puzzle_ids)
    assert 1 <= train_outputs["segments_taken"] <= 4

    # Test eval mode (uses max segments)
    model.eval()
    eval_outputs = model(input_ids, puzzle_ids)
    assert eval_outputs["segments_taken"] == 4

    print("✓ Adaptive computation test passed")


def test_reasoning_history() -> None:
    """Test reasoning history functionality in System 2."""
    print("Testing reasoning history...")

    model = create_asymmetric_hrm_model(
        vocab_size=50,
        hidden_size=64,
        num_heads=2,
        intermediate_size=128,
        max_seq_len=16,
        num_puzzle_ids=3,
        L_blocks=1,
        H_blocks=2,
        H_memory_size=8,
        T_cycles=1,
        M_segments=2,
    )

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, 50, (batch_size, seq_len))
    puzzle_ids = torch.randint(0, 3, (batch_size,))

    # Initially no history
    assert model.H_module.reasoning_history is None

    # Process in training mode to build history
    model.train()
    outputs = model(input_ids, puzzle_ids)

    # Should have reasoning history now
    assert model.H_module.reasoning_history is not None
    assert model.H_module.reasoning_history.shape[0] == batch_size

    # Reset history
    model.reset_reasoning_history()
    assert model.H_module.reasoning_history is None

    print("✓ Reasoning history test passed")


def test_model_info() -> None:
    """Test model information functionality."""
    print("Testing model info...")

    model = create_asymmetric_hrm_model(
        vocab_size=200,
        hidden_size=96,
        num_heads=3,
        intermediate_size=192,
        max_seq_len=24,
        num_puzzle_ids=8,
        L_blocks=2,
        H_blocks=4,
        H_memory_size=12,
        T_cycles=2,
        M_segments=6,
    )

    info = model.get_model_info()

    # Test required keys
    required_keys = [
        "total_parameters",
        "L_parameters",
        "H_parameters",
        "L_blocks",
        "H_blocks",
        "T_cycles",
        "M_segments",
        "H_memory_size",
        "hidden_size",
    ]

    for key in required_keys:
        assert key in info, f"Missing key: {key}"

    # Test values make sense
    assert info["L_blocks"] == 1  # num_layers // 2
    assert info["H_blocks"] == 4  # max(2, num_layers)
    assert info["T_cycles"] == 2
    assert info["M_segments"] == 6
    assert info["H_memory_size"] == 12
    assert info["hidden_size"] == 96

    # Test parameter counts
    assert info["total_parameters"] > 0
    assert info["L_parameters"] > 0
    assert info["H_parameters"] > 0

    # Test that total is at least the sum of system parameters
    # (there are also embedding and output head parameters)
    assert info["total_parameters"] >= info["L_parameters"] + info["H_parameters"]

    print("✓ Model info test passed")


def test_gradient_flow() -> None:
    """Test that gradients flow properly through the model."""
    print("Testing gradient flow...")

    model = create_asymmetric_hrm_model(
        vocab_size=100,
        hidden_size=64,
        num_heads=2,
        intermediate_size=128,
        max_seq_len=16,
        num_puzzle_ids=5,
        L_blocks=1,
        H_blocks=2,
        H_memory_size=8,
        T_cycles=1,
        M_segments=2,
    )

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    puzzle_ids = torch.randint(0, 5, (batch_size,))

    # Forward pass in training mode to build reasoning history
    model.train()
    outputs = model(input_ids, puzzle_ids)

    # Compute loss
    target = torch.randint(0, 100, (batch_size, seq_len))
    loss = torch.nn.functional.cross_entropy(
        outputs["logits"].view(-1, 100), target.view(-1)
    )

    # Backward pass
    loss.backward()

    # Check that gradients exist for parameters that were used
    # Some parameters might not have gradients if they weren't used in the forward pass
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    # Check that at least some gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    assert grad_count > 0, "No gradients found"
    assert (
        grad_count >= total_params * 0.5
    ), f"Too few gradients: {grad_count}/{total_params}"

    print("✓ Gradient flow test passed")


def run_all_tests() -> None:
    """Run all tests."""
    print("RUNNING ASYMMETRIC HRM TESTS")
    print("=" * 50)

    try:
        test_system1_module_structure()
        test_system2_module_structure()
        test_asymmetric_hrm_structure()
        test_parameter_efficiency()
        test_adaptive_computation()
        test_reasoning_history()
        test_model_info()
        test_gradient_flow()

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
