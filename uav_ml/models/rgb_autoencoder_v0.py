"""Compact convolutional RGB Autoencoder baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RgbAutoencoderConfig:
    """Serializable fixed-resolution model configuration."""

    input_channels: int = 3
    image_height: int = 72
    image_width: int = 128
    latent_dimension: int = 64

    def to_dict(self) -> dict:
        return asdict(self)


class RgbAutoencoderV0(nn.Module):
    """Encode 128x72 RGB images to a vector and reconstruct them."""

    def __init__(self, config: RgbAutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or RgbAutoencoderConfig()
        if (self.config.image_height, self.config.image_width) != (72, 128):
            raise ValueError("RgbAutoencoderV0 currently requires 128x72 input")
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 8, self.config.latent_dimension),
        )
        self.decoder_head = nn.Sequential(
            nn.Linear(self.config.latent_dimension, 128 * 5 * 8),
            nn.ReLU(inplace=True),
        )
        self.decoder_cnn = nn.Sequential(
            nn.Unflatten(1, (128, 5, 8)),
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=(0, 1),
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Return one latent vector per normalized RGB image."""
        self._validate_input(image)
        return self.encoder_head(self.encoder_cnn(image))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct normalized RGB images from latent vectors."""
        if latent.ndim != 2 or latent.shape[1] != self.config.latent_dimension:
            raise ValueError(
                f"latent must have shape [B, {self.config.latent_dimension}]"
            )
        return self.decoder_cnn(self.decoder_head(latent))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reconstruction and latent vector."""
        latent = self.encode(image)
        return self.decode(latent), latent

    def _validate_input(self, image: torch.Tensor) -> None:
        expected = (
            self.config.input_channels,
            self.config.image_height,
            self.config.image_width,
        )
        if image.ndim != 4 or tuple(image.shape[1:]) != expected:
            raise ValueError(f"image must have shape [B, {expected[0]}, {expected[1]}, {expected[2]}]")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
