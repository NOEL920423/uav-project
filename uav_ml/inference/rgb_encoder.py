"""Checkpoint-backed RGB encoder for future policy observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from uav_ml.datasets.expert_image_dataset import preprocess_rgb_image
from uav_ml.models import (
    MultiImageAutoencoderConfig,
    MultiImageAutoencoderV0,
    RgbAutoencoderConfig,
    RgbAutoencoderV0,
)
from uav_ml.train_bc import resolve_device


RESNET_PREPROCESSING = (
    "PIL RGB -> bilinear 128x72 -> CHW float32 [0,1] -> "
    "ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]"
)
RESNET_DEPTH_PREPROCESSING = (
    "PNG uint16 millimetres; invalid 0 -> 0; clip [50,30000] mm; "
    "linear [0,1] -> repeat 3 channels -> bilinear 128x72 -> "
    "ImageNet mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]"
)


def resnet_preprocessing_for_sources(
    image_sources: tuple[str, ...] | list[str],
) -> str | dict[str, str]:
    """Return the recorded ResNet preprocessing contract for each source."""
    sources = tuple(image_sources)
    values = {
        source: RESNET_DEPTH_PREPROCESSING if source == "fpv_depth"
        else RESNET_PREPROCESSING
        for source in sources
    }
    return values if len(sources) > 1 else values[sources[0]]


class ResNet18Encoder(torch.nn.Module):
    """Frozen ImageNet features for one or more aligned image streams."""

    def __init__(
        self,
        pretrained: bool = False,
        image_sources: tuple[str, ...] | list[str] = ("fpv_rgb",),
    ) -> None:
        super().__init__()
        # AE users do not need torchvision installed.
        from torchvision.models import ResNet18_Weights, resnet18

        sources = tuple(image_sources)
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("image_sources must be nonempty and unique")
        self.image_sources = sources
        if len(sources) == 1:
            self.config = RgbAutoencoderConfig(latent_dimension=512)
            self.backbone = self._build_backbone(resnet18, ResNet18_Weights, pretrained)
        else:
            self.config = MultiImageAutoencoderConfig(
                image_sources=sources,
                latent_dimension=512,
            )
            self.backbones = torch.nn.ModuleDict({
                source: self._build_backbone(
                    resnet18, ResNet18_Weights, pretrained
                )
                for source in sources
            })
            self.fusion = torch.nn.Linear(len(sources) * 512, 512)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _build_backbone(resnet18, weights_enum, pretrained: bool) -> torch.nn.Module:
        backbone = resnet18(
            weights=weights_enum.IMAGENET1K_V1 if pretrained else None
        )
        backbone.fc = torch.nn.Identity()
        return backbone

    @property
    def preprocessing(self) -> str | dict[str, str]:
        return resnet_preprocessing_for_sources(self.image_sources)

    @property
    def checkpoint_config(self) -> dict:
        return {**self.config.to_dict(), "image_sources": list(self.image_sources)}

    def encode(
        self, image: torch.Tensor | dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if len(self.image_sources) > 1:
            if not isinstance(image, dict):
                raise ValueError("multi-image ResNet input must be a source mapping")
            if set(image) != set(self.image_sources):
                raise ValueError(
                    f"images must contain exactly {self.image_sources}"
                )
            features = [
                self._encode_single(image[source], self.backbones[source])
                for source in self.image_sources
            ]
            return self.fusion(torch.cat(features, dim=1))
        if not isinstance(image, torch.Tensor):
            raise ValueError("single-image ResNet input must be a tensor")
        return self._encode_single(image, self.backbone)

    def _encode_single(
        self, image: torch.Tensor, backbone: torch.nn.Module
    ) -> torch.Tensor:
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, 72, 128):
            raise ValueError("ResNet18 input must have shape [B,3,72,128]")
        return backbone((image - self.mean) / self.std)


def load_frozen_encoder(
    path: Path, device: torch.device,
) -> tuple[RgbAutoencoderV0 | MultiImageAutoencoderV0 | ResNet18Encoder, dict]:
    """Load recorded weights without downloading anything at inference time."""
    payload = torch.load(path, map_location=device, weights_only=False)
    architecture = payload.get("model_class")
    if architecture == "RgbAutoencoderV0":
        model = RgbAutoencoderV0(RgbAutoencoderConfig(**payload["model_config"]))
    elif architecture == "MultiImageAutoencoderV0":
        model = MultiImageAutoencoderV0(
            MultiImageAutoencoderConfig.from_dict(payload["model_config"])
        )
    elif architecture == "ResNet18Encoder":
        model_config = payload["model_config"]
        model = ResNet18Encoder(
            image_sources=tuple(model_config.get("image_sources", ("fpv_rgb",)))
        )
        expected_config = (
            model.checkpoint_config
            if "image_sources" in model_config else model.config.to_dict()
        )
        if payload["model_config"] != expected_config:
            raise ValueError("ResNet18 encoder configuration mismatch")
        if payload.get("preprocessing") != model.preprocessing:
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
