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
import time
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
class RMSNorm(nn.Module):
    """Root Mean Square normalization with learnable scale parameter.

    This is the normalization used in modern LLMs like LLaMA, PaLM, etc.
    It's more efficient than LayerNorm as it doesn't compute mean.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self, x: Float[Tensor, "batch seq hidden"]
    ) -> Float[Tensor, "batch seq hidden"]:
        """Apply RMS normalization.

        Args:
            x: Input tensor [batch, seq, hidden]

        Returns:
            Normalized tensor [batch, seq, hidden]
        """
        variance = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


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
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if max_seq_len > 100000:  # Reasonable upper bound
            raise ValueError(f"max_seq_len {max_seq_len} is unreasonably large (>100k)")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        self.max_seq_len = max_seq_len  # Store for reference

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

    def __init__(
        self,
        hidden_size: int,
        memory_size: int,
        update_strategy: str = "simple",
        retrieval_strategy: str = "simple",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.update_strategy = update_strategy
        self.retrieval_strategy = retrieval_strategy

        # Timing statistics with double indexing
        self.timing_stats = {
            "retrieval_times": {},  # [retrieval_type] => [list of times]
            "update_times": {},  # [update_type] => [list of times]
            "total_times": {},  # [update_type][retrieval_type] => total_time
            "avg_times": {},  # [update_type][retrieval_type] => avg_time
            "counts": {},  # [update_type][retrieval_type] => count
        }

        # Memory buffer (learnable parameters)
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_size))

        # Memory update mechanisms
        self.memory_update = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.memory_gate = nn.Linear(hidden_size, memory_size, bias=False)

        # Additional parameters for sophisticated updates
        if update_strategy in ["attention", "sophisticated"]:
            self.memory_attention = nn.MultiheadAttention(
                hidden_size, num_heads=4, batch_first=True
            )
            self.memory_norm = RMSNorm(hidden_size)

        if update_strategy == "sophisticated":
            self.memory_decay = nn.Parameter(torch.ones(memory_size) * 0.95)
            self.memory_importance = nn.Linear(hidden_size, 1, bias=False)

        # Additional parameters for sophisticated retrieval
        if retrieval_strategy in ["attention", "sophisticated"]:
            self.retrieval_attention = nn.MultiheadAttention(
                hidden_size, num_heads=4, batch_first=True
            )
            self.retrieval_norm = RMSNorm(hidden_size)

        if retrieval_strategy == "sophisticated":
            self.retrieval_decay = nn.Parameter(torch.ones(memory_size) * 0.95)
            self.retrieval_importance = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        current_state: Float[Tensor, "batch seq hidden"],
        update_memory: bool = True,
    ) -> Float[Tensor, "batch seq hidden"]:
        """Update working memory and return memory-enhanced state."""
        batch_size, seq_len, _ = current_state.shape

        # Time memory retrieval
        start_retrieval = time.perf_counter()
        if self.retrieval_strategy == "simple":
            memory_retrieved = self._retrieve_memory_simple(current_state)
        elif self.retrieval_strategy == "attention":
            memory_retrieved = self._retrieve_memory_attention(current_state)
        elif self.retrieval_strategy == "sophisticated":
            memory_retrieved = self._retrieve_memory_sophisticated(current_state)
        else:
            raise ValueError(f"Unknown retrieval strategy: {self.retrieval_strategy}")
        end_retrieval = time.perf_counter()

        # Combine current state with retrieved memory
        memory_enhanced = current_state + memory_retrieved

        # Time memory update
        start_update = time.perf_counter()
        if update_memory:
            if self.update_strategy == "simple":
                self._update_memory_simple(current_state)
            elif self.update_strategy == "attention":
                self._update_memory_attention(current_state)
            elif self.update_strategy == "sophisticated":
                self._update_memory_sophisticated(current_state)
        end_update = time.perf_counter()

        # Store timing data
        retrieval_time = (end_retrieval - start_retrieval) * 1000  # Convert to ms
        update_time = (end_update - start_update) * 1000  # Convert to ms

        # Store in timing stats
        if self.update_strategy not in self.timing_stats["retrieval_times"]:
            self.timing_stats["retrieval_times"][self.update_strategy] = {}
        if (
            self.retrieval_strategy
            not in self.timing_stats["retrieval_times"][self.update_strategy]
        ):
            self.timing_stats["retrieval_times"][self.update_strategy][
                self.retrieval_strategy
            ] = []

        if self.update_strategy not in self.timing_stats["update_times"]:
            self.timing_stats["update_times"][self.update_strategy] = {}
        if (
            self.retrieval_strategy
            not in self.timing_stats["update_times"][self.update_strategy]
        ):
            self.timing_stats["update_times"][self.update_strategy][
                self.retrieval_strategy
            ] = []

        self.timing_stats["retrieval_times"][self.update_strategy][
            self.retrieval_strategy
        ].append(retrieval_time)
        self.timing_stats["update_times"][self.update_strategy][
            self.retrieval_strategy
        ].append(update_time)

        return memory_enhanced

    def _update_memory_simple(
        self, current_state: Float[Tensor, "batch seq hidden"]
    ) -> None:
        """Update (simple) memory using first token as representative.

        Updates memory slots using gated combination of representative
        and current memory state.
        """
        start_time = time.perf_counter()

        batch_size = current_state.shape[0]

        # Use first token as representative
        representative = current_state[:, 0]  # [batch, hidden]

        # Compute update gate for each memory slot
        update_gate = torch.sigmoid(
            self.memory_gate(representative)
        )  # [batch, memory_size]

        # Prepare memory update
        memory_mean = self.memory.mean(0).unsqueeze(0).expand(batch_size, -1)
        memory_update = self.memory_update(
            torch.cat([representative, memory_mean], dim=-1)
        )  # [batch, hidden]

        # Apply gated update to memory (vectorized)
        with torch.no_grad():
            # Vectorized update: [memory_size, hidden] = [memory_size, hidden] * [batch, memory_size, 1] + [batch, hidden] * [batch, memory_size, 1]
            # We need to average across batch dimension for each memory slot
            memory_weights = update_gate.mean(dim=0, keepdim=True).T  # [memory_size, 1]
            memory_update_avg = memory_update.mean(dim=0, keepdim=True)  # [1, hidden]

            self.memory.data = (
                1 - memory_weights
            ) * self.memory.data + memory_weights * memory_update_avg

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Update timing statistics
        self._update_timing_stats("update", elapsed_time)

    def _update_memory_attention(
        self, current_state: Float[Tensor, "batch seq hidden"]
    ) -> None:
        """Update attention-based memory.

        Uses attention mechanism to determine which memory slots to update
        and how much to update them.
        """
        start_time = time.perf_counter()

        batch_size, seq_len, _ = current_state.shape

        # Reshape memory for attention
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute attention between current state and memory
        attn_output, attn_weights = self.memory_attention(
            current_state, memory_expanded, memory_expanded
        )  # [batch, seq, hidden], [batch, seq, memory_size]

        # Use attention-weighted average as update signal
        update_signal = attn_output.mean(dim=1)  # [batch, hidden]

        # Compute update gates based on attention weights
        attention_importance = attn_weights.mean(dim=1)  # [batch, memory_size]
        update_gates = torch.sigmoid(attention_importance)

        # Apply updates (vectorized)
        with torch.no_grad():
            # Vectorized update: average across batch dimension
            memory_weights = update_gates.mean(
                dim=0, keepdim=True
            ).T  # [memory_size, 1]
            update_signal_avg = update_signal.mean(dim=0, keepdim=True)  # [1, hidden]

            self.memory.data = (
                1 - memory_weights
            ) * self.memory.data + memory_weights * update_signal_avg

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Update timing statistics
        self._update_timing_stats("update", elapsed_time)

    def _update_memory_sophisticated(
        self, current_state: Float[Tensor, "batch seq hidden"]
    ) -> None:
        """Update (sophisticated) memory with decay and importance weighting.

        Features:
        - Memory decay over time
        - Importance-based slot selection
        - Attention-based update mechanism
        - Adaptive learning rates per slot
        """
        start_time = time.perf_counter()

        batch_size, seq_len, _ = current_state.shape

        # Apply memory decay first
        with torch.no_grad():
            self.memory.data *= self.memory_decay.unsqueeze(1)

        # Compute importance scores for current state
        importance_scores = torch.sigmoid(
            self.memory_importance(current_state)
        )  # [batch, seq, 1]

        # Weighted representative (importance-weighted average)
        weights = F.softmax(importance_scores.squeeze(-1), dim=1)  # [batch, seq]
        representative = torch.sum(
            current_state * weights.unsqueeze(-1), dim=1
        )  # [batch, hidden]

        # Attention-based memory interaction
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        representative_expanded = representative.unsqueeze(1)  # [batch, 1, hidden]

        attn_output, attn_weights = self.memory_attention(
            representative_expanded, memory_expanded, memory_expanded
        )  # [batch, 1, hidden], [batch, 1, memory_size]

        # Compute update gates with memory-specific learning rates
        base_gates = torch.sigmoid(
            self.memory_gate(representative)
        )  # [batch, memory_size]

        attention_modulation = attn_weights.squeeze(1)  # [batch, memory_size]
        update_gates = base_gates * attention_modulation

        # Prepare memory update with residual connection
        memory_mean = self.memory.mean(0).unsqueeze(0).expand(batch_size, -1)
        memory_update = self.memory_update(
            torch.cat([representative, memory_mean], dim=-1)
        )  # [batch, hidden]

        # Apply sophisticated updates (vectorized)
        with torch.no_grad():
            # Vectorized update: average across batch dimension
            memory_weights = update_gates.mean(dim=0)  # [memory_size]
            memory_update_avg = memory_update.mean(dim=0)  # [hidden]

            # Adaptive learning rate based on memory decay
            adaptive_lr = memory_weights * (1 - self.memory_decay)  # [memory_size]

            # Vectorized update: [memory_size, hidden] = (1 - adaptive_lr) * memory + adaptive_lr * update
            adaptive_lr_expanded = adaptive_lr.unsqueeze(1)  # [memory_size, 1]
            memory_update_expanded = memory_update_avg.unsqueeze(0)  # [1, hidden]

            self.memory.data = (
                1 - adaptive_lr_expanded
            ) * self.memory.data + adaptive_lr_expanded * memory_update_expanded

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Update timing statistics
        self._update_timing_stats("update", elapsed_time)

    def _retrieve_memory_simple(
        self, current_state: Float[Tensor, "batch seq hidden"]
    ) -> Float[Tensor, "batch seq hidden"]:
        """Simple attention-based memory retrieval.

        Args:
            current_state: Current reasoning state [batch, seq, hidden]

        Returns:
            Retrieved memory [batch, seq, hidden]
        """
        start_time = time.perf_counter()

        # Compute attention to memory
        memory_attn = torch.matmul(
            current_state, self.memory.T
        )  # [batch, seq, memory_size]
        memory_weights = F.softmax(memory_attn, dim=-1)

        # Retrieve relevant memory
        memory_retrieved = torch.matmul(
            memory_weights, self.memory
        )  # [batch, seq, hidden]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Update timing statistics
        self._update_timing_stats("retrieval", elapsed_time)

        return memory_retrieved

    def _retrieve_memory_attention(
        self, current_state: Float[Tensor, "batch seq hidden"]
    ) -> Float[Tensor, "batch seq hidden"]:
        """Multi-head attention-based memory retrieval.

        Args:
            current_state: Current reasoning state [batch, seq, hidden]

        Returns:
            Retrieved memory [batch, seq, hidden]
        """
        start_time = time.perf_counter()

        batch_size = current_state.shape[0]

        # Expand memory for batch processing
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)

        # Multi-head attention between current state and memory
        memory_retrieved, _ = self.retrieval_attention(
            current_state, memory_expanded, memory_expanded
        )  # [batch, seq, hidden]

        # Apply RMS normalization
        memory_retrieved = self.retrieval_norm(memory_retrieved)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Update timing statistics
        self._update_timing_stats("retrieval", elapsed_time)

        return memory_retrieved

    def _retrieve_memory_sophisticated(
        self, current_state: Float[Tensor, "batch seq hidden"]
    ) -> Float[Tensor, "batch seq hidden"]:
        """Sophisticated memory retrieval with decay and importance weighting.

        Args:
            current_state: Current reasoning state [batch, seq, hidden]

        Returns:
            Retrieved memory [batch, seq, hidden]
        """
        start_time = time.perf_counter()

        batch_size = current_state.shape[0]

        # Apply memory decay for retrieval
        decayed_memory = self.memory * self.retrieval_decay.unsqueeze(1)

        # Expand decayed memory for batch processing
        memory_expanded = decayed_memory.unsqueeze(0).expand(batch_size, -1, -1)

        # Multi-head attention between current state and decayed memory
        memory_retrieved, _ = self.retrieval_attention(
            current_state, memory_expanded, memory_expanded
        )  # [batch, seq, hidden]

        # Apply RMS normalization
        memory_retrieved = self.retrieval_norm(memory_retrieved)

        # Apply importance weighting
        importance_weights = torch.sigmoid(
            self.retrieval_importance(current_state)
        )  # [batch, seq, 1]
        memory_retrieved = memory_retrieved * importance_weights

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Update timing statistics
        self._update_timing_stats("retrieval", elapsed_time)

        return memory_retrieved

    def _update_timing_stats(self, operation_type: str, elapsed_time: float) -> None:
        """Update timing statistics with structured storage.

        Args:
            operation_type: Either 'retrieval' or 'update'
            elapsed_time: Time elapsed in seconds
        """
        update_type = self.update_strategy
        retrieval_type = self.retrieval_strategy

        # Initialize nested dictionaries if needed
        if operation_type not in self.timing_stats:
            self.timing_stats[operation_type] = {}
        if update_type not in self.timing_stats["total_times"]:
            self.timing_stats["total_times"][update_type] = {}
        if update_type not in self.timing_stats["avg_times"]:
            self.timing_stats["avg_times"][update_type] = {}
        if update_type not in self.timing_stats["counts"]:
            self.timing_stats["counts"][update_type] = {}

        if retrieval_type not in self.timing_stats["total_times"][update_type]:
            self.timing_stats["total_times"][update_type][retrieval_type] = 0.0
        if retrieval_type not in self.timing_stats["avg_times"][update_type]:
            self.timing_stats["avg_times"][update_type][retrieval_type] = 0.0
        if retrieval_type not in self.timing_stats["counts"][update_type]:
            self.timing_stats["counts"][update_type][retrieval_type] = 0

        # Store timing data
        if update_type not in self.timing_stats[operation_type]:
            self.timing_stats[operation_type][update_type] = []
        self.timing_stats[operation_type][update_type].append(elapsed_time)

        # Update combined statistics
        self.timing_stats["total_times"][update_type][retrieval_type] += elapsed_time
        self.timing_stats["counts"][update_type][retrieval_type] += 1

    def get_timing_stats(self) -> dict:
        """Get timing statistics for memory operations.

        Returns:
            Dictionary containing timing statistics
        """
        stats = self.timing_stats.copy()

        # Calculate averages for each combination
        for update_type in stats["total_times"]:
            for retrieval_type in stats["total_times"][update_type]:
                if stats["counts"][update_type][retrieval_type] > 0:
                    stats["avg_times"][update_type][retrieval_type] = (
                        stats["total_times"][update_type][retrieval_type]
                        / stats["counts"][update_type][retrieval_type]
                    )
                else:
                    stats["avg_times"][update_type][retrieval_type] = 0.0

        return stats

    def print_timing_table(self) -> None:
        """Print a 3x3 table of timing results in milliseconds."""
        strategies = ["simple", "attention", "sophisticated"]

        print("\n" + "=" * 80)
        print("MEMORY OPERATIONS TIMING TABLE (milliseconds)")
        print("=" * 80)
        print(
            f"{'Update\\Retrieval':<15} {'Simple':<12} {'Attention':<12} {'Sophisticated':<12}"
        )
        print("-" * 80)

        for update_type in strategies:
            row = f"{update_type.capitalize():<15}"
            for retrieval_type in strategies:
                if (
                    update_type in self.timing_stats["avg_times"]
                    and retrieval_type in self.timing_stats["avg_times"][update_type]
                ):
                    avg_time_ms = (
                        self.timing_stats["avg_times"][update_type][retrieval_type]
                        * 1000
                    )
                    row += f"{avg_time_ms:<12.1f}"
                else:
                    row += f"{'N/A':<12}"
            print(row)

        print("-" * 80)
        print("Note: Times are average per operation in milliseconds")
        print("=" * 80)

    def reset_timing_stats(self) -> None:
        """Reset timing statistics."""
        self.timing_stats = {
            "retrieval_times": {},  # [retrieval_type] => [list of times]
            "update_times": {},  # [update_type] => [list of times]
            "total_times": {},  # [update_type][retrieval_type] => total_time
            "avg_times": {},  # [update_type][retrieval_type] => avg_time
            "counts": {},  # [update_type][retrieval_type] => count
        }


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
        memory_update_strategy: str = "simple",
        memory_retrieval_strategy: str = "simple",
    ):
        super().__init__()
        self.attention = SophisticatedAttention(hidden_size, num_heads, max_seq_len)
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        self.working_memory = WorkingMemory(
            hidden_size, memory_size, memory_update_strategy, memory_retrieval_strategy
        )

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
        # Experiment with memory before attention, after attention, and
        # in parallel with attention or in combination.
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
        memory_update_strategy: str = "simple",
        memory_retrieval_strategy: str = "simple",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.memory_size = memory_size
        self.memory_update_strategy = memory_update_strategy
        self.memory_retrieval_strategy = memory_retrieval_strategy

        # More layers for System 2 (depth over speed)
        # effective_layers = max(2, num_layers)
        # The user is reponsible for setting the correct number of layers
        effective_layers = num_layers

        self.layers = nn.ModuleList(
            [
                System2Block(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    max_seq_len,
                    memory_size,
                    memory_update_strategy,
                    memory_retrieval_strategy,
                )
                for _ in range(effective_layers)
            ]
        )

        # Reasoning history buffer
        self.reasoning_history: Float[Tensor, "batch prev_seq hidden"] | None = None
        self.max_history_length = 10

    def forward(
        self,
        hidden_states: Float[Tensor, "batch seq hidden"],
        input_injection: Float[Tensor, "batch seq hidden"],
        reasoning_history: Float[Tensor, "batch prev_seq hidden"] | None = None,
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
    # Test System2Module with different memory update and retrieval strategies
    batch_size, seq_len, hidden_size = 1, 8000, 256
    num_heads, intermediate_size, max_seq_len = (
        8,
        hidden_size * 4,
        8000,
    )  # Changed from 128 to 256
    num_layers = 1
    memory_size = 64

    # Test all combinations of memory update and retrieval strategies
    update_strategies = ["simple", "attention", "sophisticated"]
    retrieval_strategies = ["simple", "attention", "sophisticated"]

    # Global timing collector
    global_timing_stats = {
        "total_times": {},  # [update_type][retrieval_type] => total_time
        "avg_times": {},  # [update_type][retrieval_type] => avg_time
        "counts": {},  # [update_type][retrieval_type] => count
    }

    for update_strategy in update_strategies:
        for retrieval_strategy in retrieval_strategies:
            print(
                f"\n=== Testing {update_strategy.upper()} update + {retrieval_strategy.upper()} retrieval ==="
            )

            model = System2Module(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                memory_size=memory_size,
                memory_update_strategy=update_strategy,
                memory_retrieval_strategy=retrieval_strategy,
            )

            # Create dummy inputs
            hidden_states = torch.randn(batch_size, seq_len, hidden_size)
            input_injection = torch.randn(batch_size, seq_len, hidden_size)

            # Store initial memory state
            initial_memory = model.layers[0].working_memory.memory.clone()

            # Forward pass - run multiple times for accurate timing
            num_iterations = 5
            warmup_iterations = 2

            # Warmup runs (discard timing)
            for _ in range(warmup_iterations):
                _ = model(hidden_states, input_injection)

            # Timed runs
            for _ in range(num_iterations):
                output = model(hidden_states, input_injection)

            # Collect timing from all layers and normalize
            total_retrieval_time = 0
            total_update_time = 0
            layer_count = 0

            for layer in model.layers:
                if hasattr(layer, "working_memory"):
                    layer_count += 1
                    retrieval_times = layer.working_memory.timing_stats[
                        "retrieval_times"
                    ][update_strategy][retrieval_strategy]
                    update_times = layer.working_memory.timing_stats["update_times"][
                        update_strategy
                    ][retrieval_strategy]

                    total_retrieval_time += sum(
                        retrieval_times[-num_iterations:]
                    )  # Last num_iterations
                    total_update_time += sum(
                        update_times[-num_iterations:]
                    )  # Last num_iterations

            # Calculate normalized timings
            avg_retrieval_time = total_retrieval_time / (num_iterations * layer_count)
            avg_update_time = total_update_time / (num_iterations * layer_count)
            total_memory_time = avg_retrieval_time + avg_update_time

            # Normalize by batch_size, seq_len, hidden_size
            normalized_time = total_memory_time / (batch_size * seq_len * hidden_size)
            per_token_time = total_memory_time / (batch_size * seq_len)

            # Check if memory was updated
            final_memory = model.layers[0].working_memory.memory
            memory_changed = not torch.allclose(initial_memory, final_memory)

            print(f"Update strategy: {update_strategy}")
            print(f"Retrieval strategy: {retrieval_strategy}")
            print(f"Input shape: {hidden_states.shape}")
            print(f"Output shape: {output.shape}")
            print(
                f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}"
            )
            print(f"Effective layers: {len(model.layers)}")
            print(f"Memory size: {model.memory_size}")
            print(f"Memory updated: {memory_changed}")
            print(
                f"Memory change magnitude: {torch.norm(final_memory - initial_memory).item():.6f}"
            )

            # Test memory retrieval mechanism directly
            working_memory = model.layers[0].working_memory
            test_state = torch.randn(1, 5, hidden_size)

            print(f"Testing {retrieval_strategy} retrieval mechanism:")
            if retrieval_strategy == "simple":
                retrieved = working_memory._retrieve_memory_simple(test_state)
            elif retrieval_strategy == "attention":
                retrieved = working_memory._retrieve_memory_attention(test_state)
            else:  # sophisticated
                retrieved = working_memory._retrieve_memory_sophisticated(test_state)

            print(f"  Retrieved shape: {retrieved.shape}")
            print(f"  Retrieved magnitude: {torch.norm(retrieved).item():.6f}")
            print(f"  Retrieval successful: {retrieved.shape == test_state.shape}")

            # Collect timing statistics for global table
            timing_stats = working_memory.get_timing_stats()

            # Initialize global stats for this combination
            if update_strategy not in global_timing_stats["total_times"]:
                global_timing_stats["total_times"][update_strategy] = {}
                global_timing_stats["avg_times"][update_strategy] = {}
                global_timing_stats["counts"][update_strategy] = {}

            if (
                retrieval_strategy
                not in global_timing_stats["total_times"][update_strategy]
            ):
                global_timing_stats["total_times"][update_strategy][
                    retrieval_strategy
                ] = 0.0
                global_timing_stats["avg_times"][update_strategy][
                    retrieval_strategy
                ] = 0.0
                global_timing_stats["counts"][update_strategy][retrieval_strategy] = 0

            # Accumulate timing data
            if (
                update_strategy in timing_stats["total_times"]
                and retrieval_strategy in timing_stats["total_times"][update_strategy]
            ):
                global_timing_stats["total_times"][update_strategy][
                    retrieval_strategy
                ] += timing_stats["total_times"][update_strategy][retrieval_strategy]
                global_timing_stats["counts"][update_strategy][retrieval_strategy] += (
                    timing_stats["counts"][update_strategy][retrieval_strategy]
                )

                # Calculate average
                if (
                    global_timing_stats["counts"][update_strategy][retrieval_strategy]
                    > 0
                ):
                    global_timing_stats["avg_times"][update_strategy][
                        retrieval_strategy
                    ] = (
                        global_timing_stats["total_times"][update_strategy][
                            retrieval_strategy
                        ]
                        / global_timing_stats["counts"][update_strategy][
                            retrieval_strategy
                        ]
                    )

            # Display individual timing statistics
            print(f"\n  === TIMING STATISTICS ===")
            if (
                update_strategy in timing_stats["total_times"]
                and retrieval_strategy in timing_stats["total_times"][update_strategy]
            ):
                total_time = timing_stats["total_times"][update_strategy][
                    retrieval_strategy
                ]
                count = timing_stats["counts"][update_strategy][retrieval_strategy]
                avg_time = timing_stats["avg_times"][update_strategy][
                    retrieval_strategy
                ]

                print(f"  Total Operations: {count}")
                print(f"  Total time: {total_time:.6f}s")
                print(f"  Average time: {avg_time:.6f}s ({avg_time*1000:.1f}ms)")
            else:
                print("  No timing data available")

    # Move this OUTSIDE the loop, after all combinations are processed
    # Store normalized timings for each combination
    for update_strategy in update_strategies:
        for retrieval_strategy in retrieval_strategies:
            if (
                update_strategy in global_timing_stats["total_times"]
                and retrieval_strategy
                in global_timing_stats["total_times"][update_strategy]
            ):
                # Calculate normalized values for this combination
                total_time = global_timing_stats["total_times"][update_strategy][
                    retrieval_strategy
                ]
                count = global_timing_stats["counts"][update_strategy][
                    retrieval_strategy
                ]
                avg_time = total_time / count if count > 0 else 0

                # Calculate normalized values
                normalized_time = avg_time / (batch_size * seq_len * hidden_size)
                per_token_time = avg_time / (batch_size * seq_len)

                # Store normalized data
                if "normalized_times" not in global_timing_stats:
                    global_timing_stats["normalized_times"] = {}
                if update_strategy not in global_timing_stats["normalized_times"]:
                    global_timing_stats["normalized_times"][update_strategy] = {}
                if (
                    retrieval_strategy
                    not in global_timing_stats["normalized_times"][update_strategy]
                ):
                    global_timing_stats["normalized_times"][update_strategy][
                        retrieval_strategy
                    ] = []

                global_timing_stats["normalized_times"][update_strategy][
                    retrieval_strategy
                ].append(
                    {
                        "raw_ms": avg_time,
                        "normalized_per_element": normalized_time,
                        "per_token_ms": per_token_time,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "hidden_size": hidden_size,
                    }
                )

    # Print final 3x3 timing table
    print("\n" + "=" * 140)
    print("FINAL MEMORY OPERATIONS TIMING TABLE")
    print("=" * 140)
    print(
        f"{'Update\\Retrieval':<15} | {'Raw (ms)':<12} {'ns/element':<12} {'ns/token':<12} | {'Raw (ms)':<12} {'ns/element':<12} {'ns/token':<12} | {'Raw (ms)':<12} {'ns/element':<12} {'ns/token':<12}"
    )
    print("-" * 140)
    print(f"{'':<15} | {'Simple':<36} | {'Attention':<36} | {'Sophisticated':<36}")
    print("-" * 140)

    for update_type in update_strategies:
        row = f"{update_type.capitalize():<15} |"
        for retrieval_type in retrieval_strategies:
            if (
                update_type in global_timing_stats["avg_times"]
                and retrieval_type in global_timing_stats["avg_times"][update_type]
            ):
                # Raw timing in milliseconds
                avg_time_ms = (
                    global_timing_stats["avg_times"][update_type][retrieval_type] * 1000
                )

                # Get normalized data if available
                if (
                    "normalized_times" in global_timing_stats
                    and update_type in global_timing_stats["normalized_times"]
                    and retrieval_type
                    in global_timing_stats["normalized_times"][update_type]
                    and global_timing_stats["normalized_times"][update_type][
                        retrieval_type
                    ]
                ):
                    # Get the latest normalized data
                    latest_data = global_timing_stats["normalized_times"][update_type][
                        retrieval_type
                    ][-1]
                    normalized_per_element = (
                        latest_data["normalized_per_element"] * 1e9
                    )  # Convert to nanoseconds
                    per_token_ns = (
                        latest_data["per_token_ms"] * 1e6
                    )  # Convert to nanoseconds

                    row += f" {avg_time_ms:<12.1f} {normalized_per_element:<12.1f} {per_token_ns:<12.1f} |"
                else:
                    row += f" {avg_time_ms:<12.1f} {'N/A':<12} {'N/A':<12} |"
            else:
                row += f" {'N/A':<12} {'N/A':<12} {'N/A':<12} |"
        print(row)

    print("-" * 140)
    print("TIMING METRICS EXPLANATION:")
    print("-" * 140)
    print("Raw (ms):        Total memory operation time in milliseconds")
    print(
        "ns/element:      Nanoseconds per data element (batch_size × seq_len × hidden_size)"
    )
    print("                 - Measures computational efficiency per element processed")
    print("                 - Lower values indicate more efficient memory operations")
    print(
        "                 - Should be roughly constant across different problem sizes if operations scale linearly"
    )
    print("")
    print("ns/token:        Nanoseconds per token (batch_size × seq_len)")
    print("                 - Measures computational cost per token in the sequence")
    print("                 - Shows how memory operations scale with sequence length")
    print(
        "                 - Linear scaling: ns/token remains constant as seq_len increases"
    )
    print(
        "                 - Quadratic scaling: ns/token increases linearly with seq_len"
    )
    print("")
    print(
        f"Problem size: batch={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}"
    )
    print(f"Total elements: {batch_size * seq_len * hidden_size:,}")
    print(f"Total tokens: {batch_size * seq_len:,}")
    print("=" * 140)
