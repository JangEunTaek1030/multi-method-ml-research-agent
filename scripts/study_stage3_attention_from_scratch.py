"""Educational Stage 3A script: Transformer attention from scratch.

Run with:
    python scripts/study_stage3_attention_from_scratch.py

This script intentionally uses tiny toy tensors so the attention mechanics are
clear and easy to inspect.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def explain_and_print_shape(name: str, tensor: torch.Tensor, extra: str = "") -> None:
    """Print a beginner-friendly shape explanation for a tensor."""
    if extra:
        print(f"{name}: shape={tuple(tensor.shape)} | {extra}")
    else:
        print(f"{name}: shape={tuple(tensor.shape)}")


class MultiHeadSelfAttention(nn.Module):
    """A minimal multi-head self-attention module built from scratch for learning.

    Steps:
    1) Project input embeddings into Query/Key/Value.
    2) Split embedding dimension into multiple heads.
    3) Compute scaled dot-product attention per head.
    4) Concatenate heads back together.
    5) Apply final linear projection.
    """

    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Linear layers that learn how to create Query, Key, and Value vectors.
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        # Final layer after concatenating all heads.
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split (B, S, E) into (B, H, S, D)."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate (B, H, S, D) back into (B, S, E)."""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_len, self.embedding_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("\n=== Multi-Head Self-Attention (from scratch) ===")
        explain_and_print_shape("Input embeddings", x, "(batch_size, seq_len, embedding_dim)")

        # 1) Linear projections: create Q, K, V from the same input x.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        explain_and_print_shape("Q projected", q)
        explain_and_print_shape("K projected", k)
        explain_and_print_shape("V projected", v)

        # 2) Split each tensor into attention heads.
        q_heads = self._split_heads(q)
        k_heads = self._split_heads(k)
        v_heads = self._split_heads(v)
        explain_and_print_shape("Q split heads", q_heads, "(batch_size, num_heads, seq_len, head_dim)")
        explain_and_print_shape("K split heads", k_heads, "(batch_size, num_heads, seq_len, head_dim)")
        explain_and_print_shape("V split heads", v_heads, "(batch_size, num_heads, seq_len, head_dim)")

        # 3) Scaled dot-product attention per head.
        scores = q_heads @ k_heads.transpose(-2, -1) / math.sqrt(self.head_dim)
        explain_and_print_shape("Attention scores", scores, "(batch_size, num_heads, seq_len, seq_len)")

        weights = F.softmax(scores, dim=-1)
        explain_and_print_shape("Attention weights", weights, "softmax over key positions")

        context_heads = weights @ v_heads
        explain_and_print_shape("Context per head", context_heads, "(batch_size, num_heads, seq_len, head_dim)")

        # 4) Concatenate heads.
        context_concat = self._concat_heads(context_heads)
        explain_and_print_shape("Concatenated heads", context_concat, "(batch_size, seq_len, embedding_dim)")

        # 5) Final output projection.
        output = self.out_proj(context_concat)
        explain_and_print_shape("Final output", output, "(batch_size, seq_len, embedding_dim)")
        return output


def single_head_attention_walkthrough(input_embeddings: torch.Tensor) -> None:
    """Walk through raw attention math once before the module abstraction."""
    print("=== Single-head scaled dot-product attention walkthrough ===")
    explain_and_print_shape("Input embeddings", input_embeddings, "(batch_size, seq_len, embedding_dim)")

    batch_size, seq_len, embedding_dim = input_embeddings.shape
    d_k = embedding_dim

    # We use linear layers only for demonstration.
    q_layer = nn.Linear(embedding_dim, embedding_dim)
    k_layer = nn.Linear(embedding_dim, embedding_dim)
    v_layer = nn.Linear(embedding_dim, embedding_dim)

    q = q_layer(input_embeddings)
    k = k_layer(input_embeddings)
    v = v_layer(input_embeddings)
    explain_and_print_shape("Q", q)
    explain_and_print_shape("K", k)
    explain_and_print_shape("V", v)

    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    context = weights @ v

    explain_and_print_shape("Attention scores", scores, "Q @ K^T / sqrt(d_k)")
    explain_and_print_shape("Attention weights", weights, "softmax(scores, dim=-1)")
    explain_and_print_shape("Context", context, "weights @ V")

    assert context.shape == (batch_size, seq_len, embedding_dim)
    print("Single-head output shape check passed.")


def main() -> None:
    print("Stage 3A Study Script: Attention From Scratch")
    print("This is an educational toy example (not training on real dataset).\n")

    # Fix randomness so runs are reproducible for learning.
    torch.manual_seed(42)

    # Tiny toy setup.
    batch_size = 2
    seq_len = 4
    embedding_dim = 8
    num_heads = 2

    # Toy input embeddings, representing 2 short sequences of 4 tokens each.
    input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)

    single_head_attention_walkthrough(input_embeddings)

    mha = MultiHeadSelfAttention(embedding_dim=embedding_dim, num_heads=num_heads)
    output = mha(input_embeddings)

    print("\n=== Final checks ===")
    expected_shape = (batch_size, seq_len, embedding_dim)
    print(f"Expected final output shape: {expected_shape}")
    print(f"Actual final output shape:   {tuple(output.shape)}")
    assert tuple(output.shape) == expected_shape
    print("Final output shape check passed.")


if __name__ == "__main__":
    main()
