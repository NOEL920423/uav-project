"""Small depth-CNN and state-MLP behavior-cloning policy."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BcPolicyConfig:
    """Serializable BC v0 model configuration."""

    depth_channels: int = 1
    state_dimension: int = 6
    action_dimension: int = 4
    depth_feature_dimension: int = 32
    state_feature_dimension: int = 32
    hidden_dimension: int = 64

    def to_dict(self) -> dict:
        return asdict(self)


class BcPolicyV0(nn.Module):
    """Predict normalized NED velocity and yaw-rate from depth and state."""

    def __init__(self, config: BcPolicyConfig | None = None) -> None:
        super().__init__()
        self.config = config or BcPolicyConfig()
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(self.config.depth_channels, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(24 * 4 * 4, self.config.depth_feature_dimension),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.config.state_dimension, self.config.state_feature_dimension),
            nn.ReLU(),
            nn.Linear(
                self.config.state_feature_dimension,
                self.config.state_feature_dimension,
            ),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(
                self.config.depth_feature_dimension
                + self.config.state_feature_dimension,
                self.config.hidden_dimension,
            ),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dimension, self.config.action_dimension),
        )

    def forward(
        self,
        depth: torch.Tensor,
        velocity: torch.Tensor,
        goal_direction: torch.Tensor,
    ) -> torch.Tensor:
        """Return a batch of four normalized action components."""
        if depth.ndim != 4:
            raise ValueError("depth batch must have shape [B, C, H, W]")
        if velocity.ndim != 2 or velocity.shape[-1] != 3:
            raise ValueError("velocity batch must have shape [B, 3]")
        if goal_direction.ndim != 2 or goal_direction.shape[-1] != 3:
            raise ValueError("goal_direction batch must have shape [B, 3]")
        depth_features = self.depth_encoder(depth)
        state = torch.cat((velocity, goal_direction), dim=-1)
        state_features = self.state_encoder(state)
        return self.action_head(torch.cat((depth_features, state_features), dim=-1))

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters())

