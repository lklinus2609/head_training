"""Audio-conditioned temporal discriminator for adversarial motion prior."""

import torch
import torch.nn as nn

from models.layers import SinusoidalPositionalEncoding


class TemporalDiscriminator(nn.Module):
    """Discriminates real vs generated facial motion windows.

    Takes K-frame windows containing expression parameters, temporal derivatives,
    and audio features. Audio conditioning makes the discriminator judge whether
    the motion matches the given speech.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        fc_hidden: int = 128,
    ):
        """
        Args:
            input_dim: Total dimension of input features per frame.
                For concatenated [expression(100) + velocity(100) + acceleration(100) + audio(C_audio)].
            d_model: Transformer hidden dimension.
            n_layers: Number of transformer encoder layers.
            n_heads: Number of attention heads.
            d_ff: Feedforward dimension.
            dropout: Dropout rate.
            fc_hidden: Hidden dimension in classification head.
        """
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input windows [B, K, input_dim] where K is the temporal window size.
                Contains concatenated expression, velocity, acceleration, and audio features.

        Returns:
            Discriminator score [B, 1] in range [0, 1].
        """
        # Project input
        h = self.input_proj(x)  # [B, K, d_model]

        # Add positional encoding
        h = self.pos_enc(h)

        # Transformer encoding
        h = self.transformer(h)  # [B, K, d_model]

        # Global average pooling over temporal dimension
        h = h.mean(dim=1)  # [B, d_model]

        # Classify
        score = self.classifier(h)  # [B, 1]
        return score
