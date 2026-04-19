"""Flow-matching generator for FLAME expression windows.

Parallel track to the deterministic autoregressive `models.generator.Generator`.
Trained with a rectified-flow objective: given a noised target window
`x_t = (1 - t) * x_0 + t * x_1` and conditioning `(audio, emotion, prev)`,
predict the velocity field `u_theta ≈ x_1 - x_0` via MSE.

At inference: draw `x_0 ~ N(0, I)`, Euler-integrate `x_t` from 0 → 1 over
`nfe` steps to recover a predicted window `x_1`. The window is non-AR
internally; sliding between windows feeds the last `prev_frames` generated
frames as conditioning for the next window.

Reuses `AudioEncoder` and `AdaLNTransformerDecoderLayer` unchanged — the
only architectural changes are (a) sinusoidal time conditioning added to
the AdaLN `cond` vector, and (b) no causal mask on self-attention.

The existing `models/generator.py` is NOT modified — this lives alongside
it so both approaches can be trained/compared on the same data and metrics
without interfering with each other or with the inference KV-cache work in
`models/generator_inference.py`.
"""

import math

import torch
import torch.nn as nn

from models.audio_encoder import AudioEncoder
from models.layers import AdaLNTransformerDecoderLayer, SinusoidalPositionalEncoding


def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal embedding of a scalar time `t` ∈ [0, 1].

    Args:
        t: [B] tensor of flow-matching times.
        dim: Output embedding dimension (should be even).

    Returns:
        [B, dim] time embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim_even]
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


class GeneratorFM(nn.Module):
    """Flow-matching velocity predictor for FLAME expression windows.

    Input per forward call:
        x_t: noised target window [B, window_size, expr_dim]
        t:   flow-matching time in [0, 1], shape [B]
        audio: audio features [B, T_audio, audio_dim]
        emotion: emotion label indices [B]
        prev_expression: [B, prev_frames, expr_dim]

    Output:
        velocity u_theta with shape [B, window_size, expr_dim].
    """

    def __init__(
        self,
        audio_dim: int,
        expr_dim: int = 100,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_emotions: int = 8,
        emotion_embed_dim: int = 64,
        prev_frames: int = 2,
        time_embed_dim: int = 64,
        audio_conv_channels: list[int] | None = None,
        audio_conv_kernel_sizes: list[int] | None = None,
    ):
        super().__init__()

        self.expr_dim = expr_dim
        self.d_model = d_model
        self.prev_frames = prev_frames
        self.time_embed_dim = time_embed_dim

        cond_dim = emotion_embed_dim + time_embed_dim

        self.audio_encoder = AudioEncoder(
            input_dim=audio_dim,
            d_model=d_model,
            conv_channels=audio_conv_channels,
            kernel_sizes=audio_conv_kernel_sizes,
        )

        self.emotion_embed = nn.Embedding(n_emotions, emotion_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.prev_proj = nn.Linear(expr_dim * prev_frames, d_model)
        self.x_proj = nn.Linear(expr_dim, d_model)

        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        self.decoder_layers = nn.ModuleList([
            AdaLNTransformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                cond_dim=cond_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.output_head = nn.Linear(d_model, expr_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        audio: torch.Tensor,
        emotion: torch.Tensor,
        prev_expression: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the velocity field at (x_t, t).

        Shapes:
            x_t:  [B, W, expr_dim]
            t:    [B]
            audio: [B, T_audio, audio_dim]
            emotion: [B]
            prev_expression: [B, P, expr_dim]

        Returns:
            u: [B, W, expr_dim]
        """
        B, W, _ = x_t.shape

        audio_enc = self.audio_encoder(audio)  # [B, T_audio, d_model]

        emotion_emb = self.emotion_embed(emotion)  # [B, emotion_embed_dim]
        t_sin = _sinusoidal_time_embedding(t, self.time_embed_dim)
        t_emb = self.time_proj(t_sin)  # [B, time_embed_dim]
        cond = torch.cat([emotion_emb, t_emb], dim=-1)  # [B, cond_dim]

        prev_flat = prev_expression.reshape(B, -1)
        prev_token = self.prev_proj(prev_flat).unsqueeze(1)  # [B, 1, d_model]

        x_tokens = self.x_proj(x_t)  # [B, W, d_model]

        decoder_input = torch.cat([prev_token, x_tokens], dim=1)  # [B, 1+W, d_model]
        decoder_input = self.pos_enc(decoder_input)

        # No causal mask: FM is non-AR within the window. Prev-token attends
        # to x-tokens and vice versa; the target distribution is joint over
        # the whole window.
        x = decoder_input
        for layer in self.decoder_layers:
            x = layer(x, audio_enc, cond, tgt_mask=None)

        u = self.output_head(x)  # [B, 1+W, expr_dim]
        return u[:, 1:, :]  # drop the prev token's slot

    @torch.no_grad()
    def sample_window(
        self,
        audio: torch.Tensor,
        emotion: torch.Tensor,
        prev_expression: torch.Tensor,
        window_size: int,
        nfe: int = 4,
    ) -> torch.Tensor:
        """Draw a single FM window via Euler integration from t=0 → t=1.

        Args:
            audio: [B, T_audio, audio_dim] aligned to the window being sampled.
            emotion: [B]
            prev_expression: [B, P, expr_dim]
            window_size: Target window length (frames).
            nfe: Number of function evaluations (Euler steps). Higher is more
                accurate; rectified flow often works at 1–4.

        Returns:
            x_1 predicted window: [B, window_size, expr_dim].
        """
        B = audio.shape[0]
        device = audio.device
        x = torch.randn(B, window_size, self.expr_dim, device=device)
        dt = 1.0 / max(nfe, 1)
        for step in range(nfe):
            t_val = step * dt
            t = torch.full((B,), t_val, device=device)
            u = self.forward(x, t, audio, emotion, prev_expression)
            x = x + dt * u
        return x
