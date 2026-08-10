"""Training-split-only normalization statistics."""

from dataclasses import asdict, dataclass

import numpy as np
import torch

from uav_ml.datasets.dataset import BcEpisodeDataset


@dataclass(frozen=True)
class NormalizationStats:
    """Scalar depth and per-component state/action normalization."""

    depth_mean: float
    depth_std: float
    velocity_mean: tuple[float, float, float]
    velocity_std: tuple[float, float, float]
    goal_mean: tuple[float, float, float]
    goal_std: tuple[float, float, float]
    action_mean: tuple[float, float, float, float]
    action_std: tuple[float, float, float, float]
    source_split: str = "train"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "NormalizationStats":
        converted = dict(values)
        for name in (
            "velocity_mean",
            "velocity_std",
            "goal_mean",
            "goal_std",
            "action_mean",
            "action_std",
        ):
            converted[name] = tuple(converted[name])
        return cls(**converted)


def _safe_std(values: np.ndarray, axis=None) -> np.ndarray:
    return np.maximum(np.std(values, axis=axis), 1e-6)


def compute_normalization(dataset: BcEpisodeDataset) -> NormalizationStats:
    """Compute statistics only from a dataset explicitly opened as train."""
    if dataset.split != "train":
        raise ValueError("normalization statistics must come from train split")
    depths = np.concatenate([episode["depth"] for episode in dataset.episodes])
    velocities = np.concatenate(
        [episode["velocity"] for episode in dataset.episodes]
    )
    goals = np.concatenate(
        [episode["goal_direction"] for episode in dataset.episodes]
    )
    actions = np.concatenate(
        [episode["expert_action"] for episode in dataset.episodes]
    )
    return NormalizationStats(
        depth_mean=float(depths.mean()),
        depth_std=float(_safe_std(depths)),
        velocity_mean=tuple(map(float, velocities.mean(axis=0))),
        velocity_std=tuple(map(float, _safe_std(velocities, axis=0))),
        goal_mean=tuple(map(float, goals.mean(axis=0))),
        goal_std=tuple(map(float, _safe_std(goals, axis=0))),
        action_mean=tuple(map(float, actions.mean(axis=0))),
        action_std=tuple(map(float, _safe_std(actions, axis=0))),
    )


class TorchNormalizer:
    """Device-aware tensor normalization used by training and inference."""

    def __init__(self, stats: NormalizationStats, device: torch.device) -> None:
        self.stats = stats
        self.depth_mean = torch.tensor(stats.depth_mean, device=device)
        self.depth_std = torch.tensor(stats.depth_std, device=device)
        self.velocity_mean = torch.tensor(stats.velocity_mean, device=device)
        self.velocity_std = torch.tensor(stats.velocity_std, device=device)
        self.goal_mean = torch.tensor(stats.goal_mean, device=device)
        self.goal_std = torch.tensor(stats.goal_std, device=device)
        self.action_mean = torch.tensor(stats.action_mean, device=device)
        self.action_std = torch.tensor(stats.action_std, device=device)

    def observation(
        self, depth: torch.Tensor, velocity: torch.Tensor, goal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            (depth - self.depth_mean) / self.depth_std,
            (velocity - self.velocity_mean) / self.velocity_std,
            (goal - self.goal_mean) / self.goal_std,
        )

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_std + self.action_mean

