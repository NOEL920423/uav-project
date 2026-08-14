"""BC actor shared with the future PPO policy mean network."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LatentBcPolicyConfig:
    observation_dimension: int = 72
    hidden_dimension: int = 128
    action_dimension: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


class LatentBcPolicy(nn.Module):
    """Map normalized latent+state observations to normalized body actions."""

    def __init__(self, config: LatentBcPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or LatentBcPolicyConfig()
        self.backbone = nn.Sequential(
            nn.Linear(self.config.observation_dimension, self.config.hidden_dimension),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dimension, self.config.hidden_dimension),
            nn.Tanh(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(self.config.hidden_dimension, self.config.action_dimension),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.config.observation_dimension:
            raise ValueError(
                f"observation must have shape [B,{self.config.observation_dimension}]"
            )
        return self.action_head(self.backbone(observation))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
