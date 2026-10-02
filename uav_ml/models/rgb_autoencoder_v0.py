"""Compact convolutional RGB Autoencoder baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping

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


@dataclass(frozen=True)
class MultiImageAutoencoderConfig:
    """Configuration for a fixed set of aligned image streams."""

    image_sources: tuple[str, ...]
    image_height: int = 72
    image_width: int = 128
    latent_dimension: int = 64

    def __post_init__(self) -> None:
        sources = tuple(str(source) for source in self.image_sources)
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("image_sources must be nonempty and unique")
        object.__setattr__(self, "image_sources", sources)
        if self.latent_dimension < 2:
            raise ValueError("latent_dimension must be at least 2")
        if (self.image_height, self.image_width) != (72, 128):
            raise ValueError("MultiImageAutoencoderV0 currently requires 128x72 input")

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "image_sources": list(self.image_sources),
        }

    @classmethod
    def from_dict(cls, values: Mapping) -> "MultiImageAutoencoderConfig":
        return cls(
            image_sources=tuple(values["image_sources"]),
            image_height=int(values.get("image_height", 72)),
            image_width=int(values.get("image_width", 128)),
            latent_dimension=int(values.get("latent_dimension", 64)),
        )


class MultiImageAutoencoderV0(nn.Module):
    """Fuse aligned image encodings into one latent and reconstruct each view."""

    def __init__(
        self, config: MultiImageAutoencoderConfig | None = None
    ) -> None:
        super().__init__()
        if config is None:
            raise ValueError("MultiImageAutoencoderV0 requires a configuration")
        self.config = config
        branch_config = RgbAutoencoderConfig(
            image_height=config.image_height,
            image_width=config.image_width,
            latent_dimension=config.latent_dimension,
        )
        self.branches = nn.ModuleDict({
            source: RgbAutoencoderV0(branch_config)
            for source in config.image_sources
        })
        self.fusion = nn.Linear(
            len(config.image_sources) * config.latent_dimension,
            config.latent_dimension,
        )
        # Begin with the first selected view as the latent anchor. Auxiliary
        # streams can contribute during training without changing the contract.
        with torch.no_grad():
            self.fusion.weight.zero_()
            self.fusion.bias.zero_()
            self.fusion.weight[:, :config.latent_dimension].copy_(
                torch.eye(config.latent_dimension)
            )

    def encode(self, images: Mapping[str, torch.Tensor]) -> torch.Tensor:
        expected = set(self.config.image_sources)
        if set(images) != expected:
            raise ValueError(
                f"images must contain exactly {self.config.image_sources}"
            )
        latents = []
        batch_size = None
        for source in self.config.image_sources:
            image = images[source]
            if batch_size is None:
                batch_size = int(image.shape[0]) if image.ndim else None
            elif image.ndim == 0 or image.shape[0] != batch_size:
                raise ValueError("all image sources must have the same batch size")
            latents.append(self.branches[source].encode(image))
        return self.fusion(torch.cat(latents, dim=1))

    def decode(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent.ndim != 2 or latent.shape[1] != self.config.latent_dimension:
            raise ValueError(
                f"latent must have shape [B, {self.config.latent_dimension}]"
            )
        return {
            source: self.branches[source].decode(latent)
            for source in self.config.image_sources
        }

    def forward(
        self, images: Mapping[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        latent = self.encode(images)
        return self.decode(latent), latent

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
