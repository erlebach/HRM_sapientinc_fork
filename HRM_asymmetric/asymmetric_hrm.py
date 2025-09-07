"""Asymmetric HRM Model - System 1/System 2 Architecture.

This module implements a Kahneman-inspired hierarchical reasoning model that combines fast, intuitive processing (System 1) with deliberate, analytical processing (System 2).

Key differences from the original HRM:
- System 1 (L): Fast, LLM-like processing with minimal memory
- System 2 (H): Sophisticated reasoning with working memory and attention to history
- Asymmetric processing blocks and complexity
- Specialized architectures for different reasoning types
- Different hidden dimensions: L_hidden_size > H_hidden_size (fine-to-coarse hierarchy)

Standard HRM Notation:
- M: Number of segments (reasoning steps)
- T: Number of cycles per segment (same for L and H)
- L_blocks: Number of transformer blocks in Low-level module
- H_blocks: Number of transformer blocks in High-level module
- L_hidden_size: Hidden dimension for System 1 (fine-grained)
- H_hidden_size: Hidden dimension for System 2 (coarse-grained)
"""

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, Int

# from .system1_module import System1Module
from system1_module import System1Module

# from .system2_module import System2Module
from system2_module import System2Module
from torch import Tensor, nn


@beartype
class AsymmetricHRMModel(nn.Module):
    """Asymmetric Hierarchical Reasoning Model with System 1/System 2 architecture.

    This model implements Kahneman's dual-process theory:
    - System 1 (L): Fast, intuitive processing like an LLM
    - System 2 (H): Deliberate, analytical processing with memory

    The architecture allows for different processing strategies at each level,
    with System 1 handling rapid pattern recognition and System 2 performing
    complex reasoning with working memory.

    Key architectural innovation:
    - L_hidden_size > H_hidden_size: Fine-to-coarse hierarchical processing
    - Similar to CNN progression from fine-grained to abstract representations

    Standard HRM Algorithm:
    - M segments (reasoning steps)
    - T cycles per segment (same for L and H)
    - L_blocks/H_blocks transformer blocks per module
    - L_hidden_size/H_hidden_size different dimensions per module
    """

    def __init__(
        self,
        vocab_size: int,
        L_hidden_size: int,  # System 1 hidden dimension (fine-grained)
        H_hidden_size: int,  # System 2 hidden dimension (coarse-grained)
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int,
        num_puzzle_ids: int,
        # Low-level (L) parameters - System 1
        L_blocks: int,
        # High-level (H) parameters - System 2
        H_blocks: int,
        H_memory_size: int = 64,
        # HRM algorithm parameters
        T_cycles: int = 2,  # Number of cycles per segment (same for L and H)
        M_segments: int = 16,  # Maximum number of segments
    ):
        super().__init__()

        self.L_hidden_size = L_hidden_size
        self.H_hidden_size = H_hidden_size
        self.T_cycles = T_cycles
        self.M_segments = M_segments

        # Embeddings - use L_hidden_size as base (fine-grained input)
        self.token_embedding = nn.Embedding(vocab_size, L_hidden_size)
        self.puzzle_embedding = nn.Embedding(num_puzzle_ids, L_hidden_size)
        self.embed_scale = math.sqrt(L_hidden_size)

        # HRM reasoning modules with different hidden sizes
        self.L_module = System1Module(
            hidden_size=L_hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_layers=L_blocks,
            max_seq_len=max_seq_len,
        )

        self.H_module = System2Module(
            hidden_size=H_hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_layers=H_blocks,
            max_seq_len=max_seq_len,
            memory_size=H_memory_size,
        )

        # Projection layers for dimension matching (only if dimensions differ)
        if L_hidden_size != H_hidden_size:
            self.L_to_H_proj = nn.Linear(L_hidden_size, H_hidden_size, bias=False)
            self.H_to_L_proj = nn.Linear(H_hidden_size, L_hidden_size, bias=False)
        else:
            self.L_to_H_proj = None
            self.H_to_L_proj = None

        # Output heads - use H_hidden_size for final predictions (abstract reasoning)
        self.lm_head = nn.Linear(H_hidden_size, vocab_size, bias=False)
        self.q_head = nn.Linear(H_hidden_size, 2, bias=True)  # halt vs continue

        # Initial states for L and H modules
        self.L_init: Tensor = nn.parameter.Parameter(torch.randn(L_hidden_size))
        self.H_init: Tensor = nn.parameter.Parameter(torch.randn(H_hidden_size))

        # Initialize Q-head to prefer continuing initially
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)  # Bias toward continuing

    def get_embeddings(
        self,
        input_ids: Int[Tensor, "batch seq"],
        puzzle_ids: Int[Tensor, "batch"],
    ) -> Float[Tensor, "batch seq L_hidden"]:
        """Get input embeddings with puzzle-specific conditioning.

        Args:
            input_ids: Tensor of token indices with shape [batch, seq].
            puzzle_ids: Tensor of puzzle indices with shape [batch].

        Returns:
            Embeddings tensor of shape [batch, seq, L_hidden], combining token and
            puzzle-specific embeddings, scaled by the embedding scale factor.

        """
        # Token embeddings (shape: [batch, seq, L_hidden])
        token_emb = self.token_embedding(input_ids)

        # Puzzle embeddings (broadcast to sequence length)
        puzzle_emb = self.puzzle_embedding(puzzle_ids).unsqueeze(
            1
        )  # [batch, 1, L_hidden]
        puzzle_emb = puzzle_emb.expand(
            -1, input_ids.size(1), -1
        )  # [batch, seq, L_hidden]

        # Combine embeddings
        # They have both been rescaled to (batch, seq, L_hidden)
        embeddings = token_emb + puzzle_emb
        return self.embed_scale * embeddings

    def forward_single_segment(
        self,
        L_state: Float[Tensor, "batch seq L_hidden"],
        H_state: Float[Tensor, "batch seq H_hidden"],
        input_embeddings: Float[Tensor, "batch seq L_hidden"],
    ) -> tuple[
        Float[Tensor, "batch seq L_hidden"],
        Float[Tensor, "batch seq H_hidden"],
        Float[Tensor, "batch 2"],
    ]:
        """Execute a single HRM segment with T cycles.

        Args:
            L_state: Low-level state [batch, seq, L_hidden]
            H_state: High-level state [batch, seq, H_hidden]
            input_embeddings: Input embeddings [batch, seq, L_hidden]

        Returns:
            new_L_state: Updated Low-level state
            new_H_state: Updated High-level state
            q_logits: Q-values for halt/continue decision [batch, 2]
        """
        # Project H state to L dimensions once per segment (H_state is constant during L processing)
        if self.H_to_L_proj is not None:
            H_to_L = self.H_to_L_proj(H_state)  # [batch, seq, L_hidden]
        else:
            H_to_L = H_state  # Same dimensions, no projection needed

        # T cycles per segment (standard HRM algorithm)
        for _ in range(self.T_cycles):
            # L processing (L_blocks iterations within each T cycle)
            # L receives input + H state for context
            L_input = H_to_L + input_embeddings
            L_state = self.L_module(L_state, L_input)

        # H processing (once per segment after all T cycles)
        # H receives L state for context (with projection if needed)
        if self.L_to_H_proj is not None:
            L_to_H = self.L_to_H_proj(L_state)  # [batch, seq, H_hidden]
            H_input = L_to_H + self.L_to_H_proj(input_embeddings)
        else:
            L_to_H = L_state  # Same dimensions, no projection needed
            H_input = L_to_H + input_embeddings
        H_state = self.H_module(H_state, H_input)

        # Q-values for halting decision (using first token of H state)
        q_logits = self.q_head(H_state[:, 0])  # [batch, 2]

        return L_state, H_state, q_logits

    def forward(
        self,
        input_ids: Int[Tensor, "batch seq"],
        puzzle_ids: Int[Tensor, "batch"],
        L_state: Float[Tensor, "batch seq L_hidden"] | None = None,
        H_state: Float[Tensor, "batch seq H_hidden"] | None = None,
        max_segments: Optional[int] = None,
    ) -> dict[str, Tensor]:
        """Execute forward pass through the Asymmetric HRM model.

        Args:
            input_ids: Input token IDs [batch, seq]
            puzzle_ids: Puzzle identifiers [batch]
            L_state: Initial Low-level state (optional)
            H_state: Initial High-level state (optional)
            max_segments: Maximum number of segments (optional)

        Returns:
            Dictionary containing:
            - logits: Language model predictions [batch, seq, vocab]
            - q_halt_logits: Q-values for halting [batch]
            - q_continue_logits: Q-values for continuing [batch]
            - final_L_state: Final Low-level state
            - final_H_state: Final High-level state
            - segments_taken: Number of segments taken
        """
        batch_size, seq_len = input_ids.shape
        max_segments = max_segments or self.M_segments

        # Initialize states if not provided
        if L_state is None:
            L_state: Float[Tensor, "batch seq L_hidden"] = (
                self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            )
        if H_state is None:
            H_state: Float[Tensor, "batch seq H_hidden"] = (
                self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            )

        # Get input embeddings
        input_embeddings = self.get_embeddings(input_ids, puzzle_ids)

        # M segments (adaptive computation)
        segments_taken = 0
        for _ in range(max_segments):
            # Forward segment through HRM algorithm
            L_state, H_state, q_logits = self.forward_single_segment(
                L_state, H_state, input_embeddings
            )
            segments_taken += 1

            # Check if we should halt (during training, use Q-values; during eval, use max segments)
            if self.training:
                # In training mode, check Q-values for early stopping
                halt_logits, continue_logits = q_logits[:, 0], q_logits[:, 1]
                should_halt = halt_logits > continue_logits
                if should_halt.all():
                    break
            # In eval mode, continue for max_segments (don't break early)

        # Generate final predictions using H state (more sophisticated)
        logits = self.lm_head(H_state)

        return {
            "logits": logits,
            "q_halt_logits": q_logits[:, 0],
            "q_continue_logits": q_logits[:, 1],
            "final_L_state": L_state,
            "final_H_state": H_state,
            "segments_taken": torch.tensor(segments_taken),
        }

    def reset_reasoning_history(self) -> None:
        """Reset the reasoning history in H module."""
        self.H_module.reset_reasoning_history()

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model architecture."""
        L_params = sum(p.numel() for p in self.L_module.parameters())
        H_params = sum(p.numel() for p in self.H_module.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "total_parameters": total_params,
            "L_parameters": L_params,
            "H_parameters": H_params,
            "L_blocks": len(self.L_module.layers),
            "H_blocks": len(self.H_module.layers),
            "T_cycles": self.T_cycles,
            "M_segments": self.M_segments,
            "H_memory_size": self.H_module.memory_size,
            "L_hidden_size": self.L_hidden_size,
            "H_hidden_size": self.H_hidden_size,
            "uses_projection_layers": self.L_to_H_proj is not None,
        }


@beartype
def create_asymmetric_hrm_model(
    vocab_size: int = 1000,
    L_hidden_size: int = 768,  # System 1: Fine-grained processing
    H_hidden_size: int = 512,  # System 2: Coarse-grained processing
    num_heads: int = 8,
    intermediate_size: int = 2048,
    max_seq_len: int = 128,
    num_puzzle_ids: int = 100,
    # Low-level (L) parameters - System 1
    L_blocks: int = 2,  # Fewer blocks for speed
    # High-level (H) parameters - System 2
    H_blocks: int = 4,  # More blocks for sophistication
    H_memory_size: int = 64,
    # HRM algorithm parameters
    T_cycles: int = 2,  # Number of cycles per segment (same for L and H)
    M_segments: int = 16,  # Maximum number of segments
) -> AsymmetricHRMModel:
    """Create an Asymmetric HRM model with System 1/System 2 architecture.

    Args:
        vocab_size: Vocabulary size
        L_hidden_size: System 1 hidden dimension (fine-grained, larger)
        H_hidden_size: System 2 hidden dimension (coarse-grained, smaller)
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate size
        max_seq_len: Maximum sequence length
        num_puzzle_ids: Number of puzzle types
        L_blocks: Number of transformer blocks in Low-level module
        H_blocks: Number of transformer blocks in High-level module
        H_memory_size: Working memory size in H module
        T_cycles: Number of cycles per segment (same for L and H)
        M_segments: Maximum number of segments

    Returns:
        Configured AsymmetricHRMModel
    """
    return AsymmetricHRMModel(
        vocab_size=vocab_size,
        L_hidden_size=L_hidden_size,
        H_hidden_size=H_hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
        num_puzzle_ids=num_puzzle_ids,
        L_blocks=L_blocks,
        H_blocks=H_blocks,
        H_memory_size=H_memory_size,
        T_cycles=T_cycles,
        M_segments=M_segments,
    )


@beartype
def create_symmetric_hrm_model(
    vocab_size: int = 1000,
    hidden_size: int = 512,  # Same dimension for both systems
    num_heads: int = 8,
    intermediate_size: int = 2048,
    max_seq_len: int = 128,
    num_puzzle_ids: int = 100,
    # Low-level (L) parameters - System 1
    L_blocks: int = 2,  # Fewer blocks for speed
    # High-level (H) parameters - System 2
    H_blocks: int = 4,  # More blocks for sophistication
    H_memory_size: int = 64,
    # HRM algorithm parameters
    T_cycles: int = 2,  # Number of cycles per segment (same for L and H)
    M_segments: int = 16,  # Maximum number of segments
) -> AsymmetricHRMModel:
    """Create an Asymmetric HRM model with equal hidden dimensions (no projection layers).

    This is a convenience function for when you want both systems to use the same
    hidden dimension, eliminating the need for projection layers and reducing parameters.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension for both System 1 and System 2
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate size
        max_seq_len: Maximum sequence length
        num_puzzle_ids: Number of puzzle types
        L_blocks: Number of transformer blocks in Low-level module
        H_blocks: Number of transformer blocks in High-level module
        H_memory_size: Working memory size in H module
        T_cycles: Number of cycles per segment (same for L and H)
        M_segments: Maximum number of segments

    Returns:
        Configured AsymmetricHRMModel with equal hidden dimensions
    """
    return AsymmetricHRMModel(
        vocab_size=vocab_size,
        L_hidden_size=hidden_size,
        H_hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
        num_puzzle_ids=num_puzzle_ids,
        L_blocks=L_blocks,
        H_blocks=H_blocks,
        H_memory_size=H_memory_size,
        T_cycles=T_cycles,
        M_segments=M_segments,
    )


if __name__ == "__main__":
    # Test the Asymmetric HRM model with different hidden dimensions
    print("=== Testing Asymmetric HRM with Different Hidden Dimensions ===")
    model = create_asymmetric_hrm_model()

    # Create dummy inputs
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    puzzle_ids = torch.randint(0, 100, (batch_size,))

    # Forward pass
    outputs = model(input_ids, puzzle_ids)

    print("Asymmetric HRM Model created successfully!")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Q-values shape: {outputs['q_halt_logits'].shape}")
    print(f"Segments taken: {outputs['segments_taken']}")

    # Print model information
    model_info = model.get_model_info()
    print("\nModel Architecture:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("=== Testing Symmetric HRM with Equal Hidden Dimensions ===")

    # Test the symmetric version (no projection layers)
    symmetric_model = create_symmetric_hrm_model()
    symmetric_outputs = symmetric_model(input_ids, puzzle_ids)

    print("Symmetric HRM Model created successfully!")
    print(f"Uses projection layers: {symmetric_model.L_to_H_proj is not None}")

    # Print symmetric model information
    symmetric_info = symmetric_model.get_model_info()
    print("\nSymmetric Model Architecture:")
    for key, value in symmetric_info.items():
        print(f"  {key}: {value}")

    print(
        f"\nParameter reduction: {model_info['total_parameters'] - symmetric_info['total_parameters']:,} parameters saved by using equal dimensions"
    )
