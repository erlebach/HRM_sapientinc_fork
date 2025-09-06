"""
Simplified Hierarchical Reasoning Model (HRM) - Didactic Version

This is a simplified, educational implementation of the HRM model that focuses on clarity
and understanding rather than efficiency. All casting, Pydantic, and optimization frills
have been removed in favor of straightforward PyTorch code.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, Int
from torch import Tensor


@beartype
def rms_norm(
    x: Float[Tensor, "batch seq hidden"], eps: float = 1e-5
) -> Float[Tensor, "batch seq hidden"]:
    """Root Mean Square normalization without learnable parameters."""
    variance = x.square().mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


@beartype
def apply_rotary_pos_emb(
    q: Float[Tensor, "batch heads seq head_dim"],
    k: Float[Tensor, "batch heads seq head_dim"],
    cos: Float[Tensor, "seq head_dim"],
    sin: Float[Tensor, "seq head_dim"],
) -> tuple[
    Float[Tensor, "batch heads seq head_dim"], Float[Tensor, "batch heads seq head_dim"]
]:
    """Apply rotary position embeddings to query and key tensors."""

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@beartype
class RotaryEmbedding(nn.Module):
    """Rotary position embeddings for transformer attention."""

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(
        self, seq_len: int
    ) -> tuple[Float[Tensor, "seq head_dim"], Float[Tensor, "seq head_dim"]]:
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


@beartype
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with rotary position embeddings."""

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(
        self, x: Float[Tensor, "batch seq hidden"]
    ) -> Float[Tensor, "batch seq hidden"]:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        cos, sin = self.rotary_emb(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        return self.out_proj(attn_output)


@beartype
class SwiGLU(nn.Module):
    """SwiGLU activation function with gated linear unit."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(
        self, x: Float[Tensor, "batch seq hidden"]
    ) -> Float[Tensor, "batch seq hidden"]:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


@beartype
class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and MLP."""

    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int, max_seq_len: int
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, max_seq_len)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(
        self, x: Float[Tensor, "batch seq hidden"]
    ) -> Float[Tensor, "batch seq hidden"]:
        # Self-attention with residual connection and RMS norm
        x = x + self.attention(x)
        x = rms_norm(x)

        # MLP with residual connection and RMS norm
        x = x + self.mlp(x)
        x = rms_norm(x)

        return x


@beartype
class ReasoningModule(nn.Module):
    """High-level or Low-level reasoning module with multiple transformer blocks."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, intermediate_size, max_seq_len)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: Float[Tensor, "batch seq hidden"],
        input_injection: Float[Tensor, "batch seq hidden"],
    ) -> Float[Tensor, "batch seq hidden"]:
        # Add input injection (this is the key difference from standard transformers)
        hidden_states = hidden_states + input_injection

        # Apply all transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


@beartype
class HRMModel(nn.Module):
    """
    Hierarchical Reasoning Model with High-level and Low-level reasoning modules.

    The model operates through hierarchical recurrent processing where:
    - High-level module (H) handles slow, abstract planning
    - Low-level module (L) handles rapid, detailed computations
    - Q-module decides when to halt computation
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int,
        num_puzzle_ids: int,
        h_layers: int,
        l_layers: int,
        h_cycles: int,
        l_cycles: int,
        halt_max_steps: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.halt_max_steps = halt_max_steps

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.puzzle_embedding = nn.Embedding(num_puzzle_ids, hidden_size)
        self.embed_scale = math.sqrt(hidden_size)

        # Reasoning modules
        self.h_level = ReasoningModule(
            hidden_size, num_heads, intermediate_size, h_layers, max_seq_len
        )
        self.l_level = ReasoningModule(
            hidden_size, num_heads, intermediate_size, l_layers, max_seq_len
        )

        # Output heads
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.q_head = nn.Linear(hidden_size, 2, bias=True)  # halt vs continue

        # Initial states for H and L modules
        self.h_init = nn.Parameter(torch.randn(hidden_size))
        self.l_init = nn.Parameter(torch.randn(hidden_size))

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
        h_state: Float[Tensor, "batch seq hidden"],
        l_state: Float[Tensor, "batch seq hidden"],
        input_embeddings: Float[Tensor, "batch seq hidden"],
    ) -> tuple[
        Float[Tensor, "batch seq hidden"],
        Float[Tensor, "batch seq hidden"],
        Float[Tensor, "batch 2"],
    ]:
        """
        Single forward step through the hierarchical reasoning process.

        Args:
            h_state: High-level reasoning state [batch, seq, hidden]
            l_state: Low-level reasoning state [batch, seq, hidden]
            input_embeddings: Input embeddings [batch, seq, hidden]

        Returns:
            new_h_state: Updated high-level state
            new_l_state: Updated low-level state
            q_logits: Q-values for halt/continue decision [batch, 2]
        """
        # Low-level processing cycles
        for _ in range(self.l_cycles):
            l_state = self.l_level(l_state, h_state + input_embeddings)

        # High-level processing cycles
        for _ in range(self.h_cycles):
            h_state = self.h_level(h_state, l_state)

        # Q-values for halting decision (using first token of high-level state)
        q_logits = self.q_head(h_state[:, 0])  # [batch, 2]

        return h_state, l_state, q_logits

    def forward(
        self,
        input_ids: Int[Tensor, "batch seq"],
        puzzle_ids: Int[Tensor, "batch"],
        h_state: Optional[Float[Tensor, "batch seq hidden"]] = None,
        l_state: Optional[Float[Tensor, "batch seq hidden"]] = None,
        max_steps: Optional[int] = None,
    ) -> dict[str, Tensor]:
        """Execute forward pass through the HRM model.

        - includes adaptive computation time.

        Args:
            input_ids: Input token IDs [batch, seq]
            puzzle_ids: Puzzle identifiers [batch]
            h_state: Initial high-level state (optional)
            l_state: Initial low-level state (optional)
            max_steps: Maximum computation steps (optional)

        Returns:
            Dictionary containing:
            - logits: Language model predictions [batch, seq, vocab]
            - q_halt_logits: Q-values for halting [batch]
            - q_continue_logits: Q-values for continuing [batch]
            - final_h_state: Final high-level state
            - final_l_state: Final low-level state
            - steps_taken: Number of computation steps taken
        """
        batch_size, seq_len = input_ids.shape
        max_steps = max_steps or self.halt_max_steps

        # Initialize states if not provided
        if h_state is None:
            h_state = (
                self.h_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            )
        if l_state is None:
            l_state = (
                self.l_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            )

        # Get input embeddings
        input_embeddings = self.get_embeddings(input_ids, puzzle_ids)

        # Adaptive computation time loop
        steps_taken = 0
        for step in range(max_steps):
            # Forward step through hierarchical reasoning
            h_state, l_state, q_logits = self.forward_single_step(
                h_state, l_state, input_embeddings
            )
            steps_taken += 1

            # Check if we should halt (during training, use Q-values; during eval, use max steps)
            if not self.training:
                break

            # Halt if Q-values suggest stopping
            halt_logits, continue_logits = q_logits[:, 0], q_logits[:, 1]
            should_halt = halt_logits > continue_logits
            if should_halt.all():
                break

        # Generate final predictions
        logits = self.lm_head(h_state)

        return {
            "logits": logits,
            "q_halt_logits": q_logits[:, 0],
            "q_continue_logits": q_logits[:, 1],
            "final_h_state": h_state,
            "final_l_state": l_state,
            "steps_taken": torch.tensor(steps_taken),
        }


@beartype
def create_hrm_model(
    vocab_size: int = 1000,
    hidden_size: int = 512,
    num_heads: int = 8,
    intermediate_size: int = 2048,
    max_seq_len: int = 128,
    num_puzzle_ids: int = 100,
    h_layers: int = 4,
    l_layers: int = 4,
    h_cycles: int = 2,
    l_cycles: int = 2,
    halt_max_steps: int = 16,
) -> HRMModel:
    """Create an HRM model with specified configuration."""
    return HRMModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_seq_len=max_seq_len,
        num_puzzle_ids=num_puzzle_ids,
        h_layers=h_layers,
        l_layers=l_layers,
        h_cycles=h_cycles,
        l_cycles=l_cycles,
        halt_max_steps=halt_max_steps,
    )


if __name__ == "__main__":
    # Test the model
    model = create_hrm_model()

    # Create dummy inputs
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    puzzle_ids = torch.randint(0, 100, (batch_size,))

    # Forward pass
    outputs = model(input_ids, puzzle_ids)

    print("Model created successfully!")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Q-values shape: {outputs['q_halt_logits'].shape}")
    print(f"Steps taken: {outputs['steps_taken']}")
