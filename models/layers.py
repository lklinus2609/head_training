"""Shared building blocks: AdaptiveLayerNorm, sinusoidal PE, custom transformer layers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Tensor of shape [B, T, d_model].
        """
        return x + self.pe[:, : x.size(1)]


class AdaptiveLayerNorm(nn.Module):
    """Adaptive Layer Normalization conditioned on an external signal.

    Modulates the scale and shift of layer norm based on a conditioning vector.
    Used for emotion conditioning in the generator.
    """

    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, d_model].
            cond: Conditioning vector [B, cond_dim].
        """
        normalized = self.norm(x)
        gamma_beta = self.proj(cond)  # [B, 2 * d_model]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # Each [B, d_model]
        gamma = gamma.unsqueeze(1)  # [B, 1, d_model]
        beta = beta.unsqueeze(1)  # [B, 1, d_model]
        return gamma * normalized + beta


class AdaLNTransformerDecoderLayer(nn.Module):
    """Custom transformer decoder layer with Adaptive Layer Norm for conditioning.

    Replaces standard LayerNorm with AdaLN, where the conditioning signal (emotion)
    modulates normalization before self-attention, cross-attention, and feedforward.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.adaln1 = AdaptiveLayerNorm(d_model, cond_dim)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.adaln2 = AdaptiveLayerNorm(d_model, cond_dim)

        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.adaln3 = AdaptiveLayerNorm(d_model, cond_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        cond: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence [B, T_tgt, d_model].
            memory: Encoder output (audio) [B, T_mem, d_model].
            cond: Conditioning vector [B, cond_dim].
            tgt_mask: Causal attention mask [T_tgt, T_tgt].
            memory_key_padding_mask: Padding mask for memory.
        """
        # Self-attention with AdaLN
        x = self.adaln1(tgt, cond)
        sa_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(sa_out)

        # Cross-attention with AdaLN
        x = self.adaln2(tgt, cond)
        ca_out, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(ca_out)

        # Feedforward with AdaLN
        x = self.adaln3(tgt, cond)
        tgt = tgt + self.ff(x)

        return tgt


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Generate a causal (upper-triangular) attention mask.

    Returns a mask where True means "do not attend".
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask
