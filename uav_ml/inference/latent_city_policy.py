"""Load and run the frozen RGB encoder plus latent city actor."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

from uav_ml.models import (
    LatentBcPolicy,
    LatentBcPolicyConfig,
    RgbAutoencoderConfig,
    RgbAutoencoderV0,
)


class LatentCityPolicyInput:
    """Convert live RGB/state observations to the normalized 72-D contract."""

    def __init__(self, checkpoint: dict, device: torch.device) -> None:
        self.device = device
        autoencoder_path = Path(checkpoint["autoencoder_checkpoint"])
        autoencoder_checkpoint = torch.load(
            autoencoder_path, map_location=device, weights_only=False
        )
        config = RgbAutoencoderConfig(**autoencoder_checkpoint["model_config"])
        self.encoder = RgbAutoencoderV0(config).to(device)
        self.encoder.load_state_dict(autoencoder_checkpoint["model_state"])
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.mean = torch.as_tensor(
            checkpoint["observation_mean"], dtype=torch.float32, device=device
        )
        self.std = torch.as_tensor(
            checkpoint["observation_std"], dtype=torch.float32, device=device
        )

    @torch.inference_mode()
    def encode(self, observation: dict[str, np.ndarray]) -> torch.Tensor:
        rgb = torch.from_numpy(observation["rgb"].copy()).to(self.device)
        image = rgb.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        latent = self.encoder.encode(image)
        state = torch.as_tensor(
            observation["state"], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        combined = torch.cat((latent, state), dim=1)
        return (combined - self.mean) / self.std


def load_bc_actor(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[LatentBcPolicy, LatentCityPolicyInput, dict]:
    """Load the exact BC actor and its frozen live-observation transform."""
    # Training currently uses NumPy 2 while Isaac Sim 5.1 embeds NumPy 1.x.
    # NumPy arrays pickled by torch therefore reference this newer module name.
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor = LatentBcPolicy(LatentBcPolicyConfig(**checkpoint["model_config"])).to(device)
    actor.load_state_dict(checkpoint["model_state"])
    actor.eval()
    return actor, LatentCityPolicyInput(checkpoint, device), checkpoint
