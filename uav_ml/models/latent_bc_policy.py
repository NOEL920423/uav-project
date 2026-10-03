"""Latent BC actor and BC-compatible actor-critic used by clipped PPO."""

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


class LatentActorCritic(nn.Module):
    """Keep the BC actor unchanged and add a separate value network."""

    def __init__(self, config: LatentBcPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or LatentBcPolicyConfig()
        self.actor = LatentBcPolicy(self.config)
        self.critic = nn.Sequential(
            nn.Linear(self.config.observation_dimension, self.config.hidden_dimension),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dimension, self.config.hidden_dimension),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dimension, 1),
        )
        # Low initial exploration is intentional: PPO starts from a useful BC
        # controller and should not immediately replace it with random flight.
        self.log_std = nn.Parameter(torch.full((self.config.action_dimension,), -2.5))

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(observation), self.log_std.exp())

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)
