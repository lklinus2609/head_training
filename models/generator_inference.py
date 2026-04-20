"""Inference-only mirror of Generator with KV-cache autoregressive loop.

Built to match the output of models.generator.Generator at FP32 (modulo floating-point
rounding) while avoiding the O(T^2) per-window work of recomputing attention over the
growing token sequence on every AR step. Training code does not import this module.
"""

import torch
import torch.nn as nn

from models.audio_encoder import AudioEncoder
from models.generator import Generator
from models.layers import (
    AdaLNTransformerDecoderLayerInference,
    SinusoidalPositionalEncoding,
)


class GeneratorInference(nn.Module):
    """Inference-time generator with KV-cached autoregressive decoding.

    Architecture mirrors Generator exactly. Construction path is via
    from_training_generator(gen) — weights are copied from a loaded training
    model, with nn.MultiheadAttention's packed Q/K/V split out per-layer.
    """

    def __init__(
        self,
        audio_dim: int,
        expr_dim: int = 100,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
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

        self.audio_encoder = AudioEncoder(
            input_dim=audio_dim,
            d_model=d_model,
            conv_channels=audio_conv_channels,
            kernel_sizes=audio_conv_kernel_sizes,
        )
        self.emotion_embed = nn.Embedding(n_emotions, emotion_embed_dim)
        self.prev_proj = nn.Linear(expr_dim * prev_frames, d_model)
        self.expr_proj = nn.Linear(expr_dim, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.decoder_layers = nn.ModuleList([
            AdaLNTransformerDecoderLayerInference(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                cond_dim=emotion_embed_dim,
            )
            for _ in range(n_layers)
        ])
        self.output_head = nn.Linear(d_model, expr_dim)

    @classmethod
    @torch.no_grad()
    def from_training_generator(cls, gen: Generator) -> "GeneratorInference":
        """Build a matching inference model and copy weights from a trained Generator."""
        first_layer = gen.decoder_layers[0]
        d_model = first_layer.self_attn.embed_dim
        n_heads = first_layer.self_attn.num_heads
        d_ff = first_layer.ff[0].out_features
        cond_dim = first_layer.adaln1.proj.in_features

        audio_enc = gen.audio_encoder
        first_conv = next(m for m in audio_enc.conv_layers if isinstance(m, nn.Conv1d))
        input_dim = first_conv.in_channels
        conv_channels = [m.out_channels for m in audio_enc.conv_layers if isinstance(m, nn.Conv1d)]
        kernel_sizes = [m.kernel_size[0] for m in audio_enc.conv_layers if isinstance(m, nn.Conv1d)]

        inf = cls(
            audio_dim=input_dim,
            expr_dim=gen.expr_dim,
            d_model=d_model,
            n_layers=len(gen.decoder_layers),
            n_heads=n_heads,
            d_ff=d_ff,
            n_emotions=gen.emotion_embed.num_embeddings,
            emotion_embed_dim=cond_dim,
            prev_frames=gen.prev_frames,
            audio_conv_channels=conv_channels,
            audio_conv_kernel_sizes=kernel_sizes,
        )

        inf.audio_encoder.load_state_dict(gen.audio_encoder.state_dict())
        inf.emotion_embed.load_state_dict(gen.emotion_embed.state_dict())
        inf.prev_proj.load_state_dict(gen.prev_proj.state_dict())
        inf.expr_proj.load_state_dict(gen.expr_proj.state_dict())
        inf.output_head.load_state_dict(gen.output_head.state_dict())
        # pos_enc is a buffer set by __init__, identical across both classes

        for inf_layer, train_layer in zip(inf.decoder_layers, gen.decoder_layers):
            inf_layer.load_from_training_layer(train_layer)

        return inf.eval()

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor,
        emotion: torch.Tensor,
        prev_expression: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """KV-cached autoregressive inference.

        Args:
            audio: [B, T_audio, C_audio].
            emotion: [B].
            prev_expression: [B, P, expr_dim].
            max_len: Number of frames to emit.

        Returns:
            Predicted expression parameters [B, max_len, expr_dim].
        """
        B = audio.size(0)
        L = len(self.decoder_layers)

        audio_enc = self.audio_encoder(audio)               # [B, T_audio, d_model]
        emotion_emb = self.emotion_embed(emotion)            # [B, cond_dim]

        prev_flat = prev_expression.reshape(B, -1)
        prev_token = self.prev_proj(prev_flat).unsqueeze(1)  # [B, 1, d_model]
        token_pe = prev_token + self.pos_enc.pe[:, 0:1]      # position 0

        # Per-layer KV cache
        past_kvs = [None] * L

        generated = []
        for t in range(max_len):
            x = token_pe
            for i, layer in enumerate(self.decoder_layers):
                x, past_kvs[i] = layer(x, audio_enc, emotion_emb, past_self_kv=past_kvs[i])

            pred = self.output_head(x)                       # [B, 1, expr_dim]
            pred = pred.clamp(-8.0, 8.0)                     # match Generator eval-mode clamp
            generated.append(pred)

            # Next input: project prediction, add PE for next position
            next_token = self.expr_proj(pred)
            token_pe = next_token + self.pos_enc.pe[:, t + 1 : t + 2]

        return torch.cat(generated, dim=1)                   # [B, max_len, expr_dim]
