"""Checkpoint-backed RGB encoder for future policy observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from uav_ml.datasets.rgb_episode_dataset import preprocess_rgb_image
from uav_ml.models import RgbAutoencoderConfig, RgbAutoencoderV0
from uav_ml.train_bc import resolve_device


RESNET_PREPROCESSING = (
    "PIL RGB -> bilinear 128x72 -> CHW float32 [0,1] -> "
    "ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]"
)


class ResNet18Encoder(torch.nn.Module):
    """Frozen ImageNet features; keep the entire 128x72 navigation image."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        # AE users do not need torchvision installed.
        from torchvision.models import ResNet18_Weights, resnet18

        self.config = RgbAutoencoderConfig(latent_dimension=512)
        self.backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone.fc = torch.nn.Identity()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 72, 128):
            raise ValueError("ResNet18 input must have shape [B,3,72,128]")
        return self.backbone((image - self.mean) / self.std)


def load_frozen_encoder(
    path: Path, device: torch.device,
) -> tuple[RgbAutoencoderV0 | ResNet18Encoder, dict]:
    """Load recorded weights without downloading anything at inference time."""
    payload = torch.load(path, map_location=device, weights_only=False)
    architecture = payload.get("model_class")
    if architecture == "RgbAutoencoderV0":
        model = RgbAutoencoderV0(RgbAutoencoderConfig(**payload["model_config"]))
    elif architecture == "ResNet18Encoder":
        model = ResNet18Encoder()
        if payload["model_config"] != model.config.to_dict():
            raise ValueError("ResNet18 encoder configuration mismatch")
        if payload.get("preprocessing") != RESNET_PREPROCESSING:
            raise ValueError("ResNet18 encoder preprocessing mismatch")
    else:
        raise ValueError(f"unsupported encoder model class: {architecture!r}")
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    model.requires_grad_(False)
    return model, payload


class RgbEncoderInference:
    """Preprocess one RGB frame and return the pretrained latent vector."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.model, payload = load_frozen_encoder(Path(checkpoint_path), self.device)
        self.metadata = payload["metadata"]

    def preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """Convert HWC/CHW uint8 or [0,1] RGB into [1,3,72,128] float32."""
        array = np.asarray(rgb)
        if array.ndim != 3:
            raise ValueError("RGB input must be a three-dimensional array")
        if array.shape[-1] == 3:
            array = np.moveaxis(array, -1, 0)
        elif array.shape[0] != 3:
            raise ValueError("RGB input must have HWC or CHW channel layout")
        if np.issubdtype(array.dtype, np.integer):
            if array.min() < 0 or array.max() > 255:
                raise ValueError("integer RGB values must be in [0,255]")
            array = array.astype(np.uint8)
        elif not np.isfinite(array).all() or array.min() < 0.0 or array.max() > 1.0:
            raise ValueError("floating RGB values must be finite and in [0,1]")
        else:
            array = np.rint(array * 255.0).astype(np.uint8)
        image = Image.fromarray(np.moveaxis(array, 0, -1), mode="RGB")
        tensor = preprocess_rgb_image(
            image,
            image_width=self.model.config.image_width,
            image_height=self.model.config.image_height,
        )
        return tensor.unsqueeze(0).to(self.device)

    def encode(self, rgb: np.ndarray) -> np.ndarray:
        """Return one unbounded float32 latent vector."""
        image = self.preprocess(rgb)
        with torch.inference_mode():
            latent = self.model.encode(image)
        return latent.squeeze(0).cpu().numpy()
