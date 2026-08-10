"""Versioned observation and action contracts for BC v0."""

from dataclasses import asdict, dataclass

import numpy as np


DATASET_VERSION = "bc_v0.1"
OBSERVATION_CONTRACT_VERSION = "uav_bc_depth_state_v0.1"
ACTION_CONTRACT_VERSION = "uav_velocity_yaw_rate_ned_v0.1"
ACTION_NAMES = ("v_north", "v_east", "v_down", "yaw_rate")


@dataclass(frozen=True)
class ContractConfig:
    """Machine-readable dimensions, units, frames, and safety bounds."""

    depth_height: int = 64
    depth_width: int = 64
    depth_min_m: float = 0.2
    depth_max_m: float = 20.0
    velocity_limit_mps: float = 5.0
    action_horizontal_limit_mps: float = 2.0
    action_vertical_limit_mps: float = 1.0
    action_total_limit_mps: float = 2.0
    yaw_rate_limit_radps: float = 1.5
    observation_frame: str = "velocity:px4_ned;goal_direction:body_frd"
    action_frame: str = "px4_ned"

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return asdict(self)


DEFAULT_CONTRACT = ContractConfig()


def validate_observation(
    depth: np.ndarray,
    velocity: np.ndarray,
    goal_direction: np.ndarray,
    contract: ContractConfig = DEFAULT_CONTRACT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and clip one observation without silently filling bad data."""
    depth_array = np.asarray(depth, dtype=np.float32)
    velocity_array = np.asarray(velocity, dtype=np.float32)
    goal_array = np.asarray(goal_direction, dtype=np.float32)
    expected_depth = (1, contract.depth_height, contract.depth_width)
    if depth_array.shape != expected_depth:
        raise ValueError(
            f"depth shape {depth_array.shape} does not match {expected_depth}"
        )
    if velocity_array.shape != (3,):
        raise ValueError("velocity shape must be (3,)")
    if goal_array.shape != (3,):
        raise ValueError("goal_direction shape must be (3,)")
    if not all(
        np.isfinite(array).all()
        for array in (depth_array, velocity_array, goal_array)
    ):
        raise ValueError("observation contains NaN or Inf")
    if np.any(depth_array <= 0.0):
        raise ValueError("depth contains a missing or non-positive measurement")
    depth_array = np.clip(
        depth_array, contract.depth_min_m, contract.depth_max_m
    )
    velocity_array = np.clip(
        velocity_array,
        -contract.velocity_limit_mps,
        contract.velocity_limit_mps,
    )
    norm = float(np.linalg.norm(goal_array))
    if norm < 1e-6:
        raise ValueError("goal_direction is undefined at zero goal distance")
    goal_array = goal_array / norm
    return depth_array, velocity_array, goal_array


def clip_action(
    action: np.ndarray,
    contract: ContractConfig = DEFAULT_CONTRACT,
) -> np.ndarray:
    """Apply the documented horizontal, vertical, total, and yaw bounds."""
    result = np.asarray(action, dtype=np.float32).copy()
    if result.shape != (4,):
        raise ValueError("action shape must be (4,)")
    if not np.isfinite(result).all():
        raise ValueError("action contains NaN or Inf")
    horizontal = float(np.linalg.norm(result[:2]))
    if horizontal > contract.action_horizontal_limit_mps:
        result[:2] *= contract.action_horizontal_limit_mps / horizontal
    result[2] = np.clip(
        result[2],
        -contract.action_vertical_limit_mps,
        contract.action_vertical_limit_mps,
    )
    total = float(np.linalg.norm(result[:3]))
    if total > contract.action_total_limit_mps:
        result[:3] *= contract.action_total_limit_mps / total
    result[3] = np.clip(
        result[3],
        -contract.yaw_rate_limit_radps,
        contract.yaw_rate_limit_radps,
    )
    return result

