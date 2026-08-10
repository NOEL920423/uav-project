"""Validated simulator-only BC inference wrapper."""

import numpy as np
import torch

from uav_ml.contracts import clip_action, validate_observation
from uav_ml.training.checkpoint import load_checkpoint
from uav_ml.training.normalization import TorchNormalizer


class BcPolicyInference:
    """Load, preprocess, infer, denormalize, and conservatively clip."""

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model, self.stats, self.checkpoint = load_checkpoint(
            checkpoint_path, self.device
        )
        self.normalizer = TorchNormalizer(self.stats, self.device)

    @torch.inference_mode()
    def predict(
        self,
        depth: np.ndarray,
        velocity: np.ndarray,
        goal_direction: np.ndarray,
    ) -> np.ndarray:
        """Return one finite clipped [vn, ve, vd, yaw_rate] action."""
        depth, velocity, goal_direction = validate_observation(
            depth, velocity, goal_direction
        )
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).to(self.device)
        velocity_tensor = torch.from_numpy(velocity).unsqueeze(0).to(self.device)
        goal_tensor = torch.from_numpy(goal_direction).unsqueeze(0).to(self.device)
        normalized = self.normalizer.observation(
            depth_tensor, velocity_tensor, goal_tensor
        )
        predicted = self.model(*normalized)
        action = self.normalizer.denormalize_action(predicted)[0]
        return clip_action(action.detach().cpu().numpy())

