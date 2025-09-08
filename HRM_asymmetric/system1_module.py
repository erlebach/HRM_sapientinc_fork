"""
System 1 Module - Fast, Intuitive Processing

This module implements Kahneman's System 1 thinking:
- Fast, automatic, intuitive processing
- Pattern recognition and associative thinking
- Minimal reasoning, direct input-output mapping
- High throughput with simple operations
- No persistent working memory
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
class FastAttention(nn.Module):
    """Optimized attention for System 1 - faster but less sophisticated."""

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Simplified QKV projection (single matrix for speed)
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

        # Simplified attention (no complex masking or special handling)
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
class FastMLP(nn.Module):
    """Simplified MLP for System 1 - faster computation."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # Smaller intermediate size for speed
        self.intermediate_size = min(intermediate_size, hidden_size * 2)
        self.gate_proj = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)

    def forward(
        self, x: Float[Tensor, "batch seq hidden"]
    ) -> Float[Tensor, "batch seq hidden"]:
        # Simplified activation (just ReLU for speed)
        gate = self.gate_proj(x)
        return self.down_proj(F.relu(gate))


@beartype
class System1Block(nn.Module):
    """Fast transformer block optimized for System 1 processing."""

    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int, max_seq_len: int
    ):
        super().__init__()
        self.attention = FastAttention(hidden_size, num_heads, max_seq_len)
        self.mlp = FastMLP(hidden_size, intermediate_size)

    def forward(
        self, x: Float[Tensor, "batch seq hidden"]
    ) -> Float[Tensor, "batch seq hidden"]:
        # Simplified residual connections and normalization
        x = x + self.attention(x)
        x = rms_norm(x)

        x = x + self.mlp(x)
        x = rms_norm(x)

        return x


@beartype
class System1Module(nn.Module):
    """System 1 reasoning module - fast, intuitive processing.

    This module implements Kahneman's System 1 thinking:
    - Fast pattern recognition
    - Associative processing
    - Minimal reasoning
    - High throughput
    - No persistent memory
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Fewer layers for System 1 (speed over depth)
        # effective_layers = max(1, num_layers // 2)
        effective_layers = num_layers

        self.layers = nn.ModuleList(
            [
                System1Block(hidden_size, num_heads, intermediate_size, max_seq_len)
                for _ in range(effective_layers)
            ]
        )

    def forward(
        self,
        hidden_states: Float[Tensor, "batch seq hidden"],
        input_injection: Float[Tensor, "batch seq hidden"],
    ) -> Float[Tensor, "batch seq hidden"]:
        """Fast forward pass with minimal processing.

        Args:
            hidden_states: Current state [batch, seq, hidden]
            input_injection: Input to inject [batch, seq, hidden]

        input_injection is an additional input tensor (same shape as
        hidden_states) that is directly added to the transformer's
        hidden states. This allows external information or new input
        features to be incorporated at each forward pass, enabling
        fast, shallow integration of new data—consistent with
        System 1's rapid, intuitive processing.

        Returns:
            Updated hidden states [batch, seq, hidden]
        """
        # Simple input injection (just addition)
        hidden_states = hidden_states + input_injection

        # Apply all layers with minimal overhead
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


if __name__ == "__main__":
    # Test System1Module
    batch_size, seq_len, hidden_size = 2, 10, 512
    num_heads, intermediate_size, max_seq_len = 8, 1024, 128
    num_layers = 4

    model = System1Module(
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )

    # Create dummy inputs
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    input_injection = torch.randn(batch_size, seq_len, hidden_size)

    # Forward pass
    output = model(hidden_states, input_injection)

    print("System1Module created successfully!")
    print(f"Input shape: {hidden_states.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Effective layers: {len(model.layers)}")
