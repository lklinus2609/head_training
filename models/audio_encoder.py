"""1D convolutional audio encoder for processing mel or Wav2Vec features."""

import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """Encodes audio features (mel spectrogram or Wav2Vec) into d_model-dimensional representations.

    Architecture: stack of 1D conv layers followed by a linear projection.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        conv_channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
    ):
        """
        Args:
            input_dim: Dimension of input audio features (80 for mel, 768 for wav2vec).
            d_model: Output embedding dimension.
            conv_channels: List of output channels for each conv layer.
            kernel_sizes: List of kernel sizes for each conv layer.
        """
        super().__init__()

        if conv_channels is None:
            conv_channels = [128, 256]
        if kernel_sizes is None:
            kernel_sizes = [5, 3]

        layers = []
        in_ch = input_dim
        for out_ch, k in zip(conv_channels, kernel_sizes):
            padding = k // 2
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=padding),
                nn.ReLU(),
                nn.LayerNorm(out_ch),
            ])
            in_ch = out_ch

        self.conv_layers = nn.ModuleList(layers)
        self.output_proj = nn.Linear(in_ch, d_model) if in_ch != d_model else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Audio features [B, T, C_audio].

        Returns:
            Encoded audio [B, T, d_model].
        """
        # Conv1d expects [B, C, T]
        out = x.transpose(1, 2)

        for layer in self.conv_layers:
            if isinstance(layer, nn.LayerNorm):
                # LayerNorm expects [..., C], transpose back temporarily
                out = out.transpose(1, 2)
                out = layer(out)
                out = out.transpose(1, 2)
            else:
                out = layer(out)

        # Back to [B, T, C]
        out = out.transpose(1, 2)
        out = self.output_proj(out)
        return out
