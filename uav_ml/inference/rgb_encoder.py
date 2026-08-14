"""Checkpoint-backed RGB encoder for future policy observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from uav_ml.datasets.rgb_episode_dataset import preprocess_rgb_image
from uav_ml.models import RgbAutoencoderConfig, RgbAutoencoderV0
from uav_ml.train_bc import resolve_device


class RgbEncoderInference:
    """Preprocess one RGB frame and return the pretrained latent vector."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.device = resolve_device(device)
        payload = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        if payload.get("model_class") != "RgbAutoencoderV0":
            raise ValueError("checkpoint model class is not RgbAutoencoderV0")
        self.model = RgbAutoencoderV0(
            RgbAutoencoderConfig(**payload["model_config"])
        ).to(self.device)
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
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
