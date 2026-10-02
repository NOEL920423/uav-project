"""TOP RGB behavior-cloning contracts for the live PX4 flight graph."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from uav_ml.datasets.expert_image_dataset import (
    IMAGE_PREPROCESSING,
    preprocess_depth_image,
)
from uav_ml.datasets.rgb_episode_dataset import preprocess_rgb_image
from uav_ml.inference.rgb_encoder import RESNET_PREPROCESSING, load_frozen_encoder
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
    MultiImageAutoencoderV0,
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
BC_EXPERIMENTS_ROOT = Path("artifacts/experiments/bc")


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
    """Resolve an explicit checkpoint or the current cylinder TOP pointer.

    Relative checkpoints may be given either from the current directory or
    from ``artifacts/experiments/bc`` for concise experiment selection.
    """
    if checkpoint is not None:
        candidate = checkpoint.expanduser()
        if candidate.is_absolute():
            path = candidate.resolve()
        else:
            from_working_directory = candidate.resolve()
            from_bc_experiments = (
                repository_root.resolve() / BC_EXPERIMENTS_ROOT / candidate
            ).resolve()
            path = (
                from_working_directory
                if from_working_directory.is_file()
                else from_bc_experiments
            )
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
    requested_image_source: str | tuple[str, ...] | list[str],
    device: torch.device,
) -> dict:
    """Load and fail closed on every training/runtime contract mismatch."""
    requested_single_source = isinstance(requested_image_source, str)
    requested_sources = tuple(
        canonical_image_source(source)
        for source in (
            (requested_image_source,)
            if isinstance(requested_image_source, str)
            else requested_image_source
        )
    )
    if not requested_sources or len(set(requested_sources)) != len(requested_sources):
        raise ValueError("requested image sources must be nonempty and unique")
    _numpy_compatibility_aliases()
    payload = torch.load(
        checkpoint_path.resolve(), map_location=device, weights_only=False
    )
    architecture = payload.get("encoder_architecture")
    dimensions = {
        "RgbAutoencoderV0": 64,
        "MultiImageAutoencoderV0": 64,
        "ResNet18Encoder": 512,
    }
    if architecture not in dimensions:
        raise ValueError(f"checkpoint encoder_architecture mismatch: {architecture!r}")
    dimension = dimensions[architecture]
    if architecture == "ResNet18Encoder" and payload.get("encoder_preprocessing") != RESNET_PREPROCESSING:
        raise ValueError("checkpoint encoder_preprocessing mismatch")
    checkpoint_sources = tuple(
        canonical_image_source(source)
        for source in payload.get(
            "image_sources", [payload.get("image_source", "")]
        )
    )
    if requested_single_source:
        if not checkpoint_sources or requested_sources[0] != checkpoint_sources[0]:
            raise ValueError(
                "requested primary image source does not match checkpoint: "
                f"requested {requested_sources!r}, checkpoint starts with "
                f"{checkpoint_sources!r}"
            )
        effective_sources = checkpoint_sources
    else:
        effective_sources = requested_sources
    expected_preprocessing = (
        {
            ("top" if source == "top_rgb" else source): IMAGE_PREPROCESSING[
                "top" if source == "top_rgb" else source
            ]
            for source in effective_sources
        }
        if len(effective_sources) > 1 else
        IMAGE_PREPROCESSING[
            "top" if effective_sources[0] == "top_rgb" else effective_sources[0]
        ]
    )
    checks = {
        "format_version": BC_CHECKPOINT_FORMAT,
        "model_class": BC_MODEL_CLASS,
        "observation_contract": f"latent{dimension}_plus_body_state8_v1.0",
        "action_contract": BC_ACTION_CONTRACT,
        "latent_dimension": dimension,
        "encoder_architecture": architecture,
        "encoder_frozen": True,
        "image_preprocessing": expected_preprocessing,
        "physical_action_limits": BC_ACTION_LIMITS,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"checkpoint {key} mismatch: expected {expected!r}, "
                f"got {payload.get(key)!r}"
            )
    if checkpoint_sources != effective_sources:
        raise ValueError(
            "checkpoint image sources mismatch: "
            f"requested {effective_sources!r}, checkpoint requires "
            f"{checkpoint_sources!r}"
        )
    unsupported = [
        source for source in effective_sources
        if source not in IMPLEMENTED_IMAGE_SOURCES
    ]
    if unsupported:
        raise ValueError(
            f"image sources are not implemented for live BC flight: {unsupported}"
        )
    payload["_runtime_image_sources"] = list(effective_sources)
    return payload


@dataclass(frozen=True, slots=True)
class PolicyIdentity:
    """Machine-readable identity for the loaded policy and encoder."""

    checkpoint_path: str
    checkpoint_sha256: str
    encoder_path: str
    encoder_sha256: str
    image_source: str
    image_sources: tuple[str, ...]


class BcFlightPolicy:
    """Run the matching frozen encoder and normalized BC actor."""

    def __init__(
        self,
        checkpoint_path: Path,
        requested_image_source: str | tuple[str, ...] | list[str],
        device: torch.device,
        encoder_override: Path | None = None,
    ) -> None:
        checkpoint_path = checkpoint_path.resolve()
        requested_sources = tuple(
            canonical_image_source(source)
            for source in (
                (requested_image_source,)
                if isinstance(requested_image_source, str)
                else requested_image_source
            )
        )
        payload = load_checkpoint_payload(
            checkpoint_path, requested_image_source, device
        )
        requested_sources = tuple(payload["_runtime_image_sources"])
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
        self.encoder, encoder_payload = load_frozen_encoder(encoder_path, device)
        if (encoder_payload["model_class"] != payload["encoder_architecture"]
                or self.encoder.config.latent_dimension != payload["latent_dimension"]):
            raise ValueError("encoder architecture or latent dimension mismatch")
        if isinstance(self.encoder, MultiImageAutoencoderV0):
            if tuple(self.encoder.config.image_sources) != tuple(
                "top" if source == "top_rgb" else source
                for source in requested_sources
            ):
                raise ValueError("multi-image encoder source order mismatch")
        elif len(requested_sources) != 1:
            raise ValueError("single-image encoder cannot accept multiple sources")
        observation_dimension = self.encoder.config.latent_dimension + 8
        if payload["model_config"]["observation_dimension"] != observation_dimension:
            raise ValueError("policy observation dimension does not match encoder")
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
        if self.mean.shape != (observation_dimension,) or self.std.shape != (observation_dimension,):
            raise ValueError(f"checkpoint normalization must contain {observation_dimension} values")
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
            image_source=requested_sources[0],
            image_sources=requested_sources,
        )

    @torch.inference_mode()
    def act(
        self, image_bytes: bytes | dict[str, bytes], state8: np.ndarray
    ) -> np.ndarray:
        """Infer one action from the checkpoint's selected image sources."""
        action, _ = self._act(image_bytes, state8, profile=False)
        return action

    @torch.inference_mode()
    def act_profiled(
        self, image_bytes: bytes | dict[str, bytes], state8: np.ndarray
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Infer an action and measure its local processing stages."""
        return self._act(image_bytes, state8, profile=True)

    def _act(
        self, image_bytes: bytes | dict[str, bytes], state8: np.ndarray, *, profile: bool
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Run policy stages with optional synchronized accelerator timing."""
        timings: dict[str, float] = {}

        def start_stage() -> int:
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return time.perf_counter_ns()

        def finish_stage(name: str, started_ns: int) -> None:
            if profile and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if profile:
                timings[name] = (time.perf_counter_ns() - started_ns) / 1e6

        state = np.asarray(state8, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("state8 must be a finite 8-vector")
        if isinstance(image_bytes, bytes):
            images = {self.identity.image_source: image_bytes}
        elif isinstance(image_bytes, dict):
            images = {
                canonical_image_source(source): data
                for source, data in image_bytes.items()
            }
        else:
            raise TypeError("image_bytes must be bytes or a source-to-bytes mapping")
        if set(images) != set(self.identity.image_sources):
            raise ValueError(
                f"image inputs must contain exactly {self.identity.image_sources}"
            )
        started_ns = start_stage()
        tensors = {}
        for image_source, encoded in images.items():
            validate_live_image(encoded, image_source)
            try:
                with Image.open(BytesIO(encoded)) as source:
                    if image_source == "fpv_depth":
                        tensor = preprocess_depth_image(
                            source,
                            self.encoder.config.image_width,
                            self.encoder.config.image_height,
                        )
                    else:
                        tensor = preprocess_rgb_image(
                            source.convert("RGB"),
                            image_width=self.encoder.config.image_width,
                            image_height=self.encoder.config.image_height,
                        )
            except Exception as error:
                raise ValueError(
                    f"{image_source} image decode failed: {error}"
                ) from error
            tensors[image_source] = tensor
        finish_stage("image_decode_preprocess_ms", started_ns)

        started_ns = start_stage()
        if isinstance(self.encoder, MultiImageAutoencoderV0):
            image_tensor = {
                "top" if source == "top_rgb" else source:
                tensor.unsqueeze(0).to(self.device)
                for source, tensor in tensors.items()
            }
        else:
            image_tensor = tensors[self.identity.image_source].unsqueeze(0).to(
                self.device
            )
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        finish_stage("host_to_device_ms", started_ns)

        started_ns = start_stage()
        latent = self.encoder.encode(image_tensor)
        finish_stage("encoder_ms", started_ns)

        started_ns = start_stage()
        combined = torch.cat((latent, state_tensor), dim=1)
        normalized = (combined - self.mean) / self.std
        action_tensor = self.policy(normalized)[0]
        finish_stage("policy_ms", started_ns)

        started_ns = start_stage()
        action = action_tensor.cpu().numpy().astype(np.float32)
        finish_stage("output_transfer_ms", started_ns)
        return action, timings


# Kept as a compatibility alias for existing imports and checkpoints.
TopRgbBcPolicy = BcFlightPolicy
