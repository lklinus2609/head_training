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


class AdaLNTransformerDecoderLayerInference(nn.Module):
    """Inference-only mirror of AdaLNTransformerDecoderLayer with KV-cache support.

    Self-attention is reimplemented as separate Q/K/V linears + scaled_dot_product_attention
    so past K/V can be concatenated across autoregressive steps. Cross-attention keeps
    nn.MultiheadAttention (the audio memory is fixed per window; no cache needed).

    The training class AdaLNTransformerDecoderLayer is NOT modified — this is an
    inference-only construct populated via load_from_training_layer.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, cond_dim: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Self-attention as unpacked Q/K/V linears (so K/V can be cached)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.adaln1 = AdaptiveLayerNorm(d_model, cond_dim)

        # Cross-attention stays as MHA (memory is fixed per window)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        self.adaln2 = AdaptiveLayerNorm(d_model, cond_dim)

        # Feedforward (no dropout at inference)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.adaln3 = AdaptiveLayerNorm(d_model, cond_dim)

    @torch.no_grad()
    def load_from_training_layer(self, train_layer: "AdaLNTransformerDecoderLayer"):
        """Copy weights from a training-time layer. Splits MHA's packed Q/K/V projection."""
        in_w = train_layer.self_attn.in_proj_weight  # [3*d_model, d_model]
        in_b = train_layer.self_attn.in_proj_bias    # [3*d_model]
        d = self.d_model
        self.q_proj.weight.copy_(in_w[0 * d:1 * d])
        self.q_proj.bias.copy_(in_b[0 * d:1 * d])
        self.k_proj.weight.copy_(in_w[1 * d:2 * d])
        self.k_proj.bias.copy_(in_b[1 * d:2 * d])
        self.v_proj.weight.copy_(in_w[2 * d:3 * d])
        self.v_proj.bias.copy_(in_b[2 * d:3 * d])
        self.out_proj.weight.copy_(train_layer.self_attn.out_proj.weight)
        self.out_proj.bias.copy_(train_layer.self_attn.out_proj.bias)

        self.adaln1.load_state_dict(train_layer.adaln1.state_dict())
        self.adaln2.load_state_dict(train_layer.adaln2.state_dict())
        self.adaln3.load_state_dict(train_layer.adaln3.state_dict())
        self.cross_attn.load_state_dict(train_layer.cross_attn.state_dict())

        # Feedforward: training uses Sequential(Linear, GELU, Dropout, Linear, Dropout);
        # inference uses Sequential(Linear, GELU, Linear). Index 0 and 3 carry the weights.
        self.ff[0].load_state_dict(train_layer.ff[0].state_dict())
        self.ff[2].load_state_dict(train_layer.ff[3].state_dict())

    def _self_attention(
        self,
        x: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Self-attention with optional KV cache.

        If past_kv is None (prefill): x is the full initial sequence; applies causal mask.
        If past_kv is given (step): x is a single new token; attends to cached K/V + own K/V.
            No explicit mask — causality holds because cache does not include future tokens.

        Returns (output_with_same_shape_as_x, (K_all, V_all)).
        """
        B, T_new, _ = x.shape
        H, Dh = self.n_heads, self.d_head

        # [B, T_new, d_model] -> [B, H, T_new, Dh]
        q = self.q_proj(x).view(B, T_new, H, Dh).transpose(1, 2)
        k_new = self.k_proj(x).view(B, T_new, H, Dh).transpose(1, 2)
        v_new = self.v_proj(x).view(B, T_new, H, Dh).transpose(1, 2)

        if past_kv is None:
            k_all, v_all = k_new, v_new
            is_causal = T_new > 1  # single-token prefill needs no mask
        else:
            past_k, past_v = past_kv  # [B, H, T_past, Dh]
            k_all = torch.cat([past_k, k_new], dim=2)
            v_all = torch.cat([past_v, v_new], dim=2)
            # Causal by construction: cache has no future tokens, Q is only new positions
            is_causal = False

        attn = F.scaled_dot_product_attention(q, k_all, v_all, is_causal=is_causal)
        # [B, H, T_new, Dh] -> [B, T_new, d_model]
        attn = attn.transpose(1, 2).contiguous().view(B, T_new, self.d_model)
        out = self.out_proj(attn)
        return out, (k_all, v_all)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        cond: torch.Tensor,
        past_self_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            tgt: Target tokens. For prefill, the full initial sequence [B, T0, d_model].
                For step, a single new token [B, 1, d_model].
            memory: Encoder output (audio) [B, T_mem, d_model].
            cond: Conditioning vector [B, cond_dim].
            past_self_kv: Cached self-attention (K, V) from prior calls, or None for prefill.

        Returns:
            (output [same shape as tgt], updated_self_kv).
        """
        # Self-attention with AdaLN (residual on original tgt, matching training layer)
        x = self.adaln1(tgt, cond)
        sa_out, new_self_kv = self._self_attention(x, past_self_kv)
        tgt = tgt + sa_out

        # Cross-attention with AdaLN
        x = self.adaln2(tgt, cond)
        ca_out, _ = self.cross_attn(x, memory, memory)
        tgt = tgt + ca_out

        # Feedforward with AdaLN
        x = self.adaln3(tgt, cond)
        tgt = tgt + self.ff(x)

        return tgt, new_self_kv
