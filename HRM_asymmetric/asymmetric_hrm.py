"""
Asymmetric HRM Model - System 1/System 2 Architecture

This module combines System 1 (fast, intuitive) and System 2 (deliberate, analytical)
reasoning modules in a hierarchical architecture inspired by Kahneman's dual-process theory.

Key differences from the original HRM:
- System 1: Fast, LLM-like processing with minimal memory
- System 2: Sophisticated reasoning with working memory and attention to history
- Asymmetric processing cycles and complexity
- Specialized architectures for different reasoning types
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, Int

# from .system1_module import System1Module
from system1_module import System1Module

# from .system2_module import System2Module
from system2_module import System2Module
from torch import Tensor


@beartype
class AsymmetricHRMModel(nn.Module):
    """Asymmetric Hierarchical Reasoning Model with System 1/System 2 architecture.

    This model implements Kahneman's dual-process theory:
    - System 1 (Low-level): Fast, intuitive processing like an LLM
    - System 2 (High-level): Deliberate, analytical processing with memory

    The architecture allows for different processing strategies at each level,
    with System 1 handling rapid pattern recognition and System 2 performing
    complex reasoning with working memory.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int,
        num_puzzle_ids: int,
        # System 1 parameters (fast processing)
        s1_layers: int,
        s1_cycles: int,
        # System 2 parameters (deliberate processing)
        s2_layers: int,
        s2_cycles: int,
        s2_memory_size: int = 64,
        # Overall model parameters
        halt_max_steps: int = 5,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.s1_cycles = s1_cycles
        self.s2_cycles = s2_cycles
        self.halt_max_steps = halt_max_steps

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.puzzle_embedding = nn.Embedding(num_puzzle_ids, hidden_size)
        self.embed_scale = math.sqrt(hidden_size)

        # Asymmetric reasoning modules
        self.system1 = System1Module(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_layers=s1_layers,
            max_seq_len=max_seq_len,
        )

        self.system2 = System2Module(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_layers=s2_layers,
            max_seq_len=max_seq_len,
            memory_size=s2_memory_size,
        )

        # Output heads
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.q_head = nn.Linear(hidden_size, 2, bias=True)  # halt vs continue

        # Initial states for System 1 and System 2
        self.s1_init: Tensor = nn.parameter.Parameter(torch.randn(hidden_size))
        self.s2_init: Tensor = nn.parameter.Parameter(torch.randn(hidden_size))

        # Initialize Q-head to prefer continuing initially
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)  # Bias toward continuing

    def get_embeddings(
        self,
        input_ids: Int[Tensor, "batch seq"],
        puzzle_ids: Int[Tensor, "batch"],
    ) -> Float[Tensor, "batch seq hidden"]:
        """Get input embeddings with puzzle-specific conditioning."""
        # Token embeddings
        token_emb = self.token_embedding(input_ids)

        # Puzzle embeddings (broadcast to sequence length)
        puzzle_emb = self.puzzle_embedding(puzzle_ids).unsqueeze(
            1
        )  # [batch, 1, hidden]
        puzzle_emb = puzzle_emb.expand(
            -1, input_ids.size(1), -1
        )  # [batch, seq, hidden]

        # Combine embeddings
        embeddings = token_emb + puzzle_emb
        return self.embed_scale * embeddings

    def forward_single_step(
        self,
        s1_state: Float[Tensor, "batch seq hidden"],
        s2_state: Float[Tensor, "batch seq hidden"],
        input_embeddings: Float[Tensor, "batch seq hidden"],
    ) -> tuple[
        Float[Tensor, "batch seq hidden"],
        Float[Tensor, "batch seq hidden"],
        Float[Tensor, "batch 2"],
    ]:
        """Execute a single forward step through the asymmetric reasoning process.

        Args:
            s1_state: System 1 state [batch, seq, hidden]
            s2_state: System 2 state [batch, seq, hidden]
            input_embeddings: Input embeddings [batch, seq, hidden]

        Returns:
            new_s1_state: Updated System 1 state
            new_s2_state: Updated System 2 state
            q_logits: Q-values for halt/continue decision [batch, 2]
        """
        # System 1 processing cycles (fast, intuitive)
        for _ in range(self.s1_cycles):
            # System 1 receives input + System 2 state for context
            s1_input = s2_state + input_embeddings
            s1_state = self.system1(s1_state, s1_input)

        # System 2 processing cycles (deliberate, analytical)
        for _ in range(self.s2_cycles):
            # System 2 receives System 1 state + input for context
            s2_input = s1_state + input_embeddings
            s2_state = self.system2(s2_state, s2_input)

        # Q-values for halting decision (using first token of System 2 state)
        q_logits = self.q_head(s2_state[:, 0])  # [batch, 2]

        return s1_state, s2_state, q_logits

    def forward(
        self,
        input_ids: Int[Tensor, "batch seq"],
        puzzle_ids: Int[Tensor, "batch"],
        s1_state: Optional[Float[Tensor, "batch seq hidden"]] = None,
        s2_state: Optional[Float[Tensor, "batch seq hidden"]] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Execute forward pass through the Asymmetric HRM model.

        Args:
            input_ids: Input token IDs [batch, seq]
            puzzle_ids: Puzzle identifiers [batch]
            s1_state: Initial System 1 state (optional)
            s2_state: Initial System 2 state (optional)
            max_steps: Maximum computation steps (optional)

        Returns:
            Dictionary containing:
            - logits: Language model predictions [batch, seq, vocab]
            - q_halt_logits: Q-values for halting [batch]
            - q_continue_logits: Q-values for continuing [batch]
            - final_s1_state: Final System 1 state
            - final_s2_state: Final System 2 state
            - steps_taken: Number of computation steps taken
        """
        batch_size, seq_len = input_ids.shape
        max_steps = max_steps or self.halt_max_steps

        # Initialize states if not provided
        if s1_state is None:
            s1_state: Float[Tensor, "batch seq hidden"] = (
                self.s1_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            )
        if s2_state is None:
            s2_state: Float[Tensor, "batch seq hidden"] = (
                self.s2_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            )

        # Get input embeddings
        input_embeddings = self.get_embeddings(input_ids, puzzle_ids)

        # Adaptive computation time loop
        steps_taken = 0
        for _ in range(max_steps):
            # Forward step through asymmetric reasoning
            s1_state, s2_state, q_logits = self.forward_single_step(
                s1_state, s2_state, input_embeddings
            )
            steps_taken += 1

            # Check if we should halt (during training, use Q-values; during eval, use max steps)
            if self.training:
                # In training mode, check Q-values for early stopping
                halt_logits, continue_logits = q_logits[:, 0], q_logits[:, 1]
                should_halt = halt_logits > continue_logits
                if should_halt.all():
                    break
            # In eval mode, continue for max_steps (don't break early)

        # Generate final predictions using System 2 state (more sophisticated)
        logits = self.lm_head(s2_state)

        return {
            "logits": logits,
            "q_halt_logits": q_logits[:, 0],
            "q_continue_logits": q_logits[:, 1],
            "final_s1_state": s1_state,
            "final_s2_state": s2_state,
            "steps_taken": torch.tensor(steps_taken),
        }

    def reset_reasoning_history(self) -> None:
        """Reset the reasoning history in System 2."""
        self.system2.reset_reasoning_history()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        s1_params = sum(p.numel() for p in self.system1.parameters())
        s2_params = sum(p.numel() for p in self.system2.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            "total_parameters": total_params,
            "system1_parameters": s1_params,
            "system2_parameters": s2_params,
            "system1_layers": len(self.system1.layers),
            "system2_layers": len(self.system2.layers),
            "system1_cycles": self.s1_cycles,
            "system2_cycles": self.s2_cycles,
            "system2_memory_size": self.system2.memory_size,
            "hidden_size": self.hidden_size,
        }


@beartype
def create_asymmetric_hrm_model(
    vocab_size: int = 1000,
    hidden_size: int = 512,
    num_heads: int = 8,
    intermediate_size: int = 2048,
    max_seq_len: int = 128,
    num_puzzle_ids: int = 100,
    # System 1 parameters (fast processing)
    s1_layers: int = 2,  # Fewer layers for speed
    s1_cycles: int = 4,  # More cycles for fast processing
    # System 2 parameters (deliberate processing)
    s2_layers: int = 6,  # More layers for sophistication
    s2_cycles: int = 2,  # Fewer cycles but more complex
    s2_memory_size: int = 64,
    # Overall model parameters
    halt_max_steps: int = 16,
) -> AsymmetricHRMModel:
    """Create an Asymmetric HRM model with System 1/System 2 architecture.

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate size
        max_seq_len: Maximum sequence length
        num_puzzle_ids: Number of puzzle types
        s1_layers: Number of layers in System 1 (fewer for speed)
        s1_cycles: Number of processing cycles in System 1 (more for throughput)
        s2_layers: Number of layers in System 2 (more for sophistication)
        s2_cycles: Number of processing cycles in System 2 (fewer but complex)
        s2_memory_size: Working memory size in System 2
        halt_max_steps: Maximum computation steps

    Returns:
        Configured AsymmetricHRMModel
    """
    return AsymmetricHRMModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
        num_puzzle_ids=num_puzzle_ids,
        s1_layers=s1_layers,
        s1_cycles=s1_cycles,
        s2_layers=s2_layers,
        s2_cycles=s2_cycles,
        s2_memory_size=s2_memory_size,
        halt_max_steps=halt_max_steps,
    )


if __name__ == "__main__":
    # Test the Asymmetric HRM model
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
    print(f"Steps taken: {outputs['steps_taken']}")

    # Print model information
    model_info = model.get_model_info()
    print("\nModel Architecture:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
