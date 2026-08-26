"""TOP RGB behavior-cloning contracts for the live PX4 flight graph."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from uav_ml.datasets.rgb_episode_dataset import preprocess_rgb_image
from uav_ml.inference.bc_flight_contract import (
    ACTION_LIMITS,
    IMPLEMENTED_IMAGE_SOURCES,
    body_action_to_ned,
    build_state8,
    canonical_image_source,
    freshness_error,
    validate_live_image,
    yaw_from_quaternion,
)
from uav_ml.models import (
    LatentBcPolicy,
    LatentBcPolicyConfig,
    RgbAutoencoderConfig,
    RgbAutoencoderV0,
)


BC_CHECKPOINT_FORMAT = "bc_baseline_v1.0"
BC_MODEL_CLASS = "LatentBcPolicy"
BC_OBSERVATION_CONTRACT = "latent64_plus_body_state8_v1.0"
BC_ACTION_CONTRACT = "normalized_body_forward_right_yaw_v1.0"
BC_IMAGE_PREPROCESSING = (
    "PIL RGB -> bilinear 128x72 -> CHW float32 [0,1]"
)
BC_ACTION_LIMITS = {
    "v_forward_mps": ACTION_LIMITS[0],
    "v_right_mps": ACTION_LIMITS[1],
    "yaw_rate_radps": ACTION_LIMITS[2],
}
DEFAULT_LATEST_POINTER = Path(
    "artifacts/experiments/bc/bc_expert_cylinder_v1/top/latest.json"
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(
    repository_root: Path,
    checkpoint: Path | None = None,
) -> Path:
    """Resolve an explicit checkpoint or the current cylinder TOP pointer."""
    if checkpoint is not None:
        path = checkpoint.expanduser().resolve()
    else:
        pointer = repository_root.resolve() / DEFAULT_LATEST_POINTER
        if not pointer.is_file():
            raise FileNotFoundError(
                f"BC checkpoint pointer is missing: {pointer}"
            )
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        path = Path(payload["best_checkpoint"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BC checkpoint is missing: {path}")
    return path


def _numpy_compatibility_aliases() -> None:
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def load_checkpoint_payload(
    checkpoint_path: Path,
    requested_image_source: str,
    device: torch.device,
) -> dict:
    """Load and fail closed on every training/runtime contract mismatch."""
    requested = canonical_image_source(requested_image_source)
    _numpy_compatibility_aliases()
    payload = torch.load(
        checkpoint_path.resolve(), map_location=device, weights_only=False
    )
    checks = {
        "format_version": BC_CHECKPOINT_FORMAT,
        "model_class": BC_MODEL_CLASS,
        "observation_contract": BC_OBSERVATION_CONTRACT,
        "action_contract": BC_ACTION_CONTRACT,
        "latent_dimension": 64,
        "encoder_architecture": "RgbAutoencoderV0",
        "encoder_frozen": True,
        "image_preprocessing": BC_IMAGE_PREPROCESSING,
        "physical_action_limits": BC_ACTION_LIMITS,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"checkpoint {key} mismatch: expected {expected!r}, "
                f"got {payload.get(key)!r}"
            )
    checkpoint_source = canonical_image_source(
        payload.get("image_source", "")
    )
    if checkpoint_source != requested:
        raise ValueError(
            "checkpoint image source mismatch: "
            f"requested {requested!r}, checkpoint requires "
            f"{checkpoint_source!r}"
        )
    if requested not in IMPLEMENTED_IMAGE_SOURCES:
        raise ValueError(
            f"image source {requested!r} is not implemented for live BC flight"
        )
    return payload


@dataclass(frozen=True, slots=True)
class PolicyIdentity:
    """Machine-readable identity for the loaded policy and encoder."""

    checkpoint_path: str
    checkpoint_sha256: str
    encoder_path: str
    encoder_sha256: str
    image_source: str


class TopRgbBcPolicy:
    """Run the matching frozen encoder and normalized BC actor."""

    def __init__(
        self,
        checkpoint_path: Path,
        requested_image_source: str,
        device: torch.device,
        encoder_override: Path | None = None,
    ) -> None:
        checkpoint_path = checkpoint_path.resolve()
        payload = load_checkpoint_payload(
            checkpoint_path, requested_image_source, device
        )
        encoder_path = (
            encoder_override.expanduser().resolve()
            if encoder_override is not None
            else Path(payload["autoencoder_checkpoint"]).resolve()
        )
        if not encoder_path.is_file():
            raise FileNotFoundError(
                f"encoder checkpoint is missing: {encoder_path}"
            )
        encoder_sha256 = sha256_file(encoder_path)
        if encoder_sha256 != payload.get("autoencoder_checkpoint_sha256"):
            raise ValueError(
                "encoder SHA-256 differs from the BC training checkpoint"
            )
        encoder_payload = torch.load(
            encoder_path, map_location=device, weights_only=False
        )
        if encoder_payload.get("model_class") != "RgbAutoencoderV0":
            raise ValueError("encoder checkpoint is not RgbAutoencoderV0")
        encoder_config = RgbAutoencoderConfig(
            **encoder_payload["model_config"]
        )
        if encoder_config.latent_dimension != 64:
            raise ValueError("encoder latent dimension must be 64")
        self.encoder = RgbAutoencoderV0(encoder_config).to(device)
        self.encoder.load_state_dict(encoder_payload["model_state"])
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.policy = LatentBcPolicy(
            LatentBcPolicyConfig(**payload["model_config"])
        ).to(device)
        self.policy.load_state_dict(payload["model_state"])
        self.policy.eval()
        self.mean = torch.as_tensor(
            payload["observation_mean"], dtype=torch.float32, device=device
        )
        self.std = torch.as_tensor(
            payload["observation_std"], dtype=torch.float32, device=device
        )
        if self.mean.shape != (72,) or self.std.shape != (72,):
            raise ValueError("checkpoint normalization must contain 72 values")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(
            self.std
        ).all() or torch.any(self.std <= 0):
            raise ValueError("checkpoint normalization is invalid")
        self.device = device
        self.identity = PolicyIdentity(
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=sha256_file(checkpoint_path),
            encoder_path=str(encoder_path),
            encoder_sha256=encoder_sha256,
            image_source=canonical_image_source(requested_image_source),
        )

    @torch.inference_mode()
    def act(self, jpeg_bytes: bytes, state8: np.ndarray) -> np.ndarray:
        """Infer one normalized action from one TOP JPEG and exact state8."""
        state = np.asarray(state8, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("state8 must be a finite 8-vector")
        validate_live_image(jpeg_bytes, self.identity.image_source)
        try:
            with Image.open(BytesIO(jpeg_bytes)) as source:
                image = source.convert("RGB")
                tensor = preprocess_rgb_image(
                    image,
                    image_width=self.encoder.config.image_width,
                    image_height=self.encoder.config.image_height,
                )
        except Exception as error:
            raise ValueError(f"TOP RGB JPEG decode failed: {error}") from error
        image_tensor = tensor.unsqueeze(0).to(self.device)
        latent = self.encoder.encode(image_tensor)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        combined = torch.cat((latent, state_tensor), dim=1)
        normalized = (combined - self.mean) / self.std
        return self.policy(normalized)[0].cpu().numpy().astype(np.float32)
