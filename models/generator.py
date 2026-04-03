"""Autoregressive transformer generator: audio + emotion + previous frames → FLAME expression."""

import torch
import torch.nn as nn

from models.audio_encoder import AudioEncoder
from models.layers import (
    AdaLNTransformerDecoderLayer,
    SinusoidalPositionalEncoding,
    generate_causal_mask,
)


class Generator(nn.Module):
    """Deterministic autoregressive transformer for FLAME expression generation.

    Takes audio features, emotion label, and previous expression frames as input.
    Produces FLAME expression parameter sequences synchronized with speech.
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
        audio_conv_channels: list[int] | None = None,
        audio_conv_kernel_sizes: list[int] | None = None,
    ):
        super().__init__()

        self.expr_dim = expr_dim
        self.d_model = d_model
        self.prev_frames = prev_frames

        # Audio encoder
        self.audio_encoder = AudioEncoder(
            input_dim=audio_dim,
            d_model=d_model,
            conv_channels=audio_conv_channels,
            kernel_sizes=audio_conv_kernel_sizes,
        )

        # Emotion embedding
        self.emotion_embed = nn.Embedding(n_emotions, emotion_embed_dim)

        # Project previous expression frames to d_model
        self.prev_proj = nn.Linear(expr_dim * prev_frames, d_model)

        # Project single expression frame to d_model (for sequence building)
        self.expr_proj = nn.Linear(expr_dim, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # Transformer decoder layers with AdaLN
        self.decoder_layers = nn.ModuleList([
            AdaLNTransformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                cond_dim=emotion_embed_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Output head
        self.output_head = nn.Linear(d_model, expr_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        audio: torch.Tensor,
        emotion: torch.Tensor,
        prev_expression: torch.Tensor,
        target_expression: torch.Tensor | None = None,
        max_len: int | None = None,
    ) -> torch.Tensor:
        """Forward pass with teacher forcing.

        Args:
            audio: Audio features [B, T_audio, C_audio] where T_audio = C + seq_len + F.
            emotion: Emotion label indices [B].
            prev_expression: Previous P expression frames [B, P, expr_dim].
            target_expression: Ground truth expression for teacher forcing [B, T, expr_dim].
                If None, runs autoregressive inference.
            max_len: Number of frames to generate in autoregressive mode.

        Returns:
            Predicted expression parameters [B, T, expr_dim].
        """
        B = audio.size(0)
        device = audio.device

        # Encode audio
        audio_enc = self.audio_encoder(audio)  # [B, T_audio, d_model]

        # Emotion conditioning
        emotion_emb = self.emotion_embed(emotion)  # [B, emotion_embed_dim]

        if target_expression is not None:
            # Teacher forcing: use ground truth as input
            return self._forward_teacher_forcing(
                audio_enc, emotion_emb, prev_expression, target_expression, device
            )
        else:
            # Autoregressive inference
            return self._forward_autoregressive(
                audio_enc, emotion_emb, prev_expression, device, max_len=max_len
            )

    def _forward_teacher_forcing(
        self,
        audio_enc: torch.Tensor,
        emotion_emb: torch.Tensor,
        prev_expression: torch.Tensor,
        target_expression: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Teacher forcing: predict all frames in parallel using GT as input."""
        B, T, _ = target_expression.shape

        # Build input sequence: [prev_frames..., target_frame_0, ..., target_frame_{T-2}]
        # We shift the target right by 1 so each position predicts the next frame
        prev_flat = prev_expression.reshape(B, -1)  # [B, P * expr_dim]
        prev_token = self.prev_proj(prev_flat).unsqueeze(1)  # [B, 1, d_model]

        # Project target expressions (shifted right - exclude last frame)
        target_tokens = self.expr_proj(target_expression[:, :-1])  # [B, T-1, d_model]

        # Concatenate: [prev_context, target_0, ..., target_{T-2}]
        decoder_input = torch.cat([prev_token, target_tokens], dim=1)  # [B, T, d_model]

        # Add positional encoding
        decoder_input = self.pos_enc(decoder_input)

        # Causal mask
        causal_mask = generate_causal_mask(T, device)

        # Run through decoder layers
        x = decoder_input
        for layer in self.decoder_layers:
            x = layer(x, audio_enc, emotion_emb, tgt_mask=causal_mask)

        # Output projection
        output = self.output_head(x)  # [B, T, expr_dim]
        return output

    def _forward_autoregressive(
        self,
        audio_enc: torch.Tensor,
        emotion_emb: torch.Tensor,
        prev_expression: torch.Tensor,
        device: torch.device,
        max_len: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive inference: predict one frame at a time."""
        B = audio_enc.size(0)

        # Start with prev_expression context
        prev_flat = prev_expression.reshape(B, -1)
        prev_token = self.prev_proj(prev_flat).unsqueeze(1)  # [B, 1, d_model]

        # Apply positional encoding to the initial token once
        prev_token_pe = prev_token + self.pos_enc.pe[:, 0:1]

        T = max_len if max_len is not None else audio_enc.size(1)

        generated = []
        # Store tokens with PE already applied
        tokens_with_pe = [prev_token_pe]

        for t in range(T):
            seq = torch.cat(tokens_with_pe, dim=1)  # [B, t+1, d_model]

            causal_mask = generate_causal_mask(seq.size(1), device)

            x = seq
            for layer in self.decoder_layers:
                x = layer(x, audio_enc, emotion_emb, tgt_mask=causal_mask)

            # Take last position output
            pred = self.output_head(x[:, -1:])  # [B, 1, expr_dim]

            # Clamp to prevent drift (only at inference; avoids flat gradients during training)
            if not self.training:
                pred = pred.clamp(-8.0, 8.0)
            generated.append(pred)

            # Project predicted frame and add positional encoding for position t+1
            next_token = self.expr_proj(pred)  # [B, 1, d_model]
            next_token_pe = next_token + self.pos_enc.pe[:, t + 1 : t + 2]
            tokens_with_pe.append(next_token_pe)

        return torch.cat(generated, dim=1)  # [B, T, expr_dim]
