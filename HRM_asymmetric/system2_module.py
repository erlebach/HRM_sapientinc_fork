"""
System 2 Module - Deliberate, Analytical Processing with Memory

This module implements Kahneman's System 2 thinking:
- Slow, deliberate, analytical processing
- Complex reasoning with working memory
- Attention to previous reasoning steps
- State persistence across reasoning cycles
- Sophisticated attention mechanisms
"""

import math
from typing import Optional, Tuple

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

    def rotate_half(x: Float[Tensor, "... head_dim"]) -> Float[Tensor, "... head_dim"]:
        """Rotate the last dimension of the tensor by splitting in half and swapping."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
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
class WorkingMemory(nn.Module):
    """Working memory buffer for System 2 reasoning.

    Maintains a persistent memory of previous reasoning steps
    that can be attended to during current processing.
    """

    def __init__(self, hidden_size: int, memory_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        # Memory buffer (learnable parameters)
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_size))

        # Memory update mechanisms
        self.memory_update = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.memory_gate = nn.Linear(hidden_size, memory_size, bias=False)

    def forward(
        self,
        current_state: Float[Tensor, "batch seq hidden"],
        update_memory: bool = True,
    ) -> Float[Tensor, "batch seq hidden"]:
        """Update working memory and return memory-enhanced state.

        Args:
            current_state: Current reasoning state [batch, seq, hidden]
            update_memory: Whether to update the memory buffer

        Returns:
            Memory-enhanced state [batch, seq, hidden]
        """
        batch_size, seq_len, _ = current_state.shape

        # Compute attention to memory
        memory_attn = torch.matmul(
            current_state, self.memory.T
        )  # [batch, seq, memory_size]
        memory_weights = F.softmax(memory_attn, dim=-1)

        # Retrieve relevant memory
        memory_retrieved = torch.matmul(
            memory_weights, self.memory
        )  # [batch, seq, hidden]

        # Combine current state with retrieved memory
        memory_enhanced = current_state + memory_retrieved

        if update_memory:
            # Update memory buffer based on current state
            # Use first token as representative for memory update
            representative = current_state[:, 0]  # [batch, hidden]

            # Compute memory update gate
            update_gate = torch.sigmoid(
                self.memory_gate(representative)
            )  # [batch, memory_size]

            # Update memory (simplified - could be more sophisticated)
            memory_update = self.memory_update(
                torch.cat(
                    [
                        representative,
                        self.memory.mean(0).unsqueeze(0).expand(batch_size, -1),
                    ],
                    dim=-1,
                )
            )  # [batch, hidden]

            # Apply update to memory (this is a simplified version)
            # In practice, you might want more sophisticated memory management
            with torch.no_grad():
                # This is a placeholder - real memory update would be more complex
                pass

        return memory_enhanced


@beartype
class SophisticatedAttention(nn.Module):
    """Advanced attention mechanism for System 2 reasoning."""

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Separate projections for more control
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Additional attention mechanisms
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

        # Attention to previous reasoning steps
        self.reasoning_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )

    def forward(
        self,
        x: Float[Tensor, "batch seq hidden"],
        reasoning_history: Optional[Float[Tensor, "batch prev_seq hidden"]] = None,
    ) -> Float[Tensor, "batch seq hidden"]:
        """Advanced attention with reasoning history.

        Args:
            x: Current input [batch, seq, hidden]
            reasoning_history: Previous reasoning steps [batch, prev_seq, hidden]

        Returns:
            Attended output [batch, seq, hidden]
        """
        batch_size, seq_len, _ = x.shape

        # Standard self-attention
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

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

        # Reshape
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        # Additional attention to reasoning history if available
        if reasoning_history is not None:
            history_attn, _ = self.reasoning_attention(
                attn_output, reasoning_history, reasoning_history
            )
            attn_output = attn_output + history_attn

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
class System2Block(nn.Module):
    """Sophisticated transformer block for System 2 processing."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int,
        memory_size: int = 64,
    ):
        super().__init__()
        self.attention = SophisticatedAttention(hidden_size, num_heads, max_seq_len)
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        self.working_memory = WorkingMemory(hidden_size, memory_size)

    def forward(
        self,
        x: Float[Tensor, "batch seq hidden"],
        reasoning_history: Optional[Float[Tensor, "batch prev_seq hidden"]] = None,
    ) -> Float[Tensor, "batch seq hidden"]:
        """Forward pass with working memory and reasoning history.

        Args:
            x: Input state [batch, seq, hidden]
            reasoning_history: Previous reasoning steps [batch, prev_seq, hidden]

        Returns:
            Updated state [batch, seq, hidden]
        """
        # Working memory enhancement
        x = self.working_memory(x, update_memory=True)

        # Self-attention with reasoning history
        x = x + self.attention(x, reasoning_history)
        x = rms_norm(x)

        # MLP processing
        x = x + self.mlp(x)
        x = rms_norm(x)

        return x


@beartype
class System2Module(nn.Module):
    """System 2 reasoning module - deliberate, analytical processing with memory.

    This module implements Kahneman's System 2 thinking:
    - Complex reasoning with working memory
    - Attention to previous reasoning steps
    - State persistence across cycles
    - Sophisticated attention mechanisms
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_layers: int,
        max_seq_len: int,
        memory_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.memory_size = memory_size

        # More layers for System 2 (depth over speed)
        effective_layers = max(2, num_layers)

        self.layers = nn.ModuleList(
            [
                System2Block(
                    hidden_size, num_heads, intermediate_size, max_seq_len, memory_size
                )
                for _ in range(effective_layers)
            ]
        )

        # Reasoning history buffer
        self.reasoning_history: Optional[Float[Tensor, "batch prev_seq hidden"]] = None
        self.max_history_length = 10

    def forward(
        self,
        hidden_states: Float[Tensor, "batch seq hidden"],
        input_injection: Float[Tensor, "batch seq hidden"],
        reasoning_history: Optional[Float[Tensor, "batch prev_seq hidden"]] = None,
    ) -> Float[Tensor, "batch seq hidden"]:
        """Deliberate forward pass with memory and reasoning history.

        Args:
            hidden_states: Current state [batch, seq, hidden]
            input_injection: Input to inject [batch, seq, hidden]
            reasoning_history: Previous reasoning steps [batch, prev_seq, hidden]

        Returns:
            Updated hidden states [batch, seq, hidden]
        """
        # More sophisticated input injection
        # Could include gating or other mechanisms
        hidden_states = hidden_states + input_injection

        # Apply all layers with reasoning history
        for layer in self.layers:
            hidden_states = layer(hidden_states, reasoning_history)

        # Update reasoning history
        if self.training:
            self._update_reasoning_history(hidden_states)

        return hidden_states

    def _update_reasoning_history(
        self, current_state: Float[Tensor, "batch seq hidden"]
    ) -> None:
        """Update the reasoning history buffer."""
        if self.reasoning_history is None:
            self.reasoning_history = current_state
        else:
            # Concatenate with previous history
            combined = torch.cat([self.reasoning_history, current_state], dim=1)

            # Keep only the most recent history
            if combined.size(1) > self.max_history_length:
                self.reasoning_history = combined[:, -self.max_history_length :]
            else:
                self.reasoning_history = combined

    def reset_reasoning_history(self) -> None:
        """Reset the reasoning history buffer."""
        self.reasoning_history = None


if __name__ == "__main__":
    # Test System2Module
    batch_size, seq_len, hidden_size = 2, 10, 512
    num_heads, intermediate_size, max_seq_len = 8, 2048, 128
    num_layers = 4
    memory_size = 64

    model = System2Module(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        memory_size=memory_size,
    )

    # Create dummy inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    input_injection = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    output = model(hidden_states, input_injection)

    print("System2Module created successfully!")
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Effective layers: {len(model.layers)}")
    print(f"Memory size: {model.memory_size}")
    print(f"Has reasoning history: {model.reasoning_history is not None}")
