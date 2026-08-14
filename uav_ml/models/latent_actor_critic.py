"""BC-compatible actor-critic used by clipped PPO."""

from __future__ import annotations

import torch
from torch import nn

from uav_ml.models.latent_bc_policy import LatentBcPolicy, LatentBcPolicyConfig


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
