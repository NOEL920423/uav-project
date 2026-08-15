"""Pure Phase 10A BC expert dataset V1 contract and geometry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


DATASET_VERSION = "bc_expert_v1.0"
OBSERVATION_VERSION = "rgb64_body_state8_v1.0"
ACTION_VERSION = "body_velocity_yaw_normalized_v1.0"
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 180
IMAGE_FORMAT = "jpeg"
JPEG_QUALITY = 85
SAMPLE_RATE_HZ = 5.0
SYNCHRONIZATION_TOLERANCE_S = 0.100
GOAL_DISTANCE_NORMALIZER_M = 10.0
ACTION_LIMITS = (1.0, 0.8, 1.0)
CSV_FIELDS = (
    "episode_id", "sample_id", "image_timestamp_s", "image_path",
    "state_timestamp_s", "expert_action_timestamp_s",
    "previous_action_timestamp_s", "state_image_error_s",
    "expert_action_image_error_s", "position_north_m", "position_east_m",
    "position_down_m", "yaw_ned_rad", "body_velocity_forward_mps",
    "body_velocity_right_mps", "goal_direction_forward",
    "goal_direction_right", "raw_goal_distance_m",
    "normalized_goal_distance", "expert_v_forward_mps",
    "expert_v_right_mps", "expert_yaw_rate_radps", "expert_action_forward",
    "expert_action_right", "expert_action_yaw_rate",
    "previous_v_forward_mps", "previous_v_right_mps",
    "previous_yaw_rate_radps", "previous_action_forward",
    "previous_action_right", "previous_action_yaw_rate", "mission_phase",
    "success", "failure",
)


@dataclass(frozen=True, slots=True)
class TimedValue:
    """One timestamped value retained by the synchronization buffer."""

    timestamp_s: float
    value: object


def timestamp_seconds(stamp) -> float:
    """Convert a ROS builtin time message to finite seconds."""
    value = float(stamp.sec) + float(stamp.nanosec) / 1e9
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("timestamp must be finite and positive")
    return value


def nearest(values: list[TimedValue], target_s: float) -> TimedValue | None:
    """Return the nearest timestamped value without hiding join error."""
    if not values:
        return None
    return min(values, key=lambda item: abs(item.timestamp_s - target_s))


def previous(
    values: list[TimedValue], selected: TimedValue
) -> TimedValue | None:
    """Return the control sample immediately before ``selected``."""
    candidates = [
        item for item in values
        if item.timestamp_s < selected.timestamp_s
    ]
    return max(candidates, key=lambda item: item.timestamp_s, default=None)


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract NED yaw from a finite ROS-order quaternion."""
    values = tuple(float(value) for value in (x, y, z, w))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion must be finite")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-8:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = (value / norm for value in values)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def ned_to_body(
    north: float, east: float, yaw_rad: float
) -> tuple[float, float]:
    """Rotate one horizontal NED vector into body forward/right."""
    north = float(north)
    east = float(east)
    yaw_rad = float(yaw_rad)
    if not all(math.isfinite(value) for value in (north, east, yaw_rad)):
        raise ValueError("NED vector and yaw must be finite")
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return (
        cosine * north + sine * east,
        -sine * north + cosine * east,
    )


def goal_features(
    position_north_m: float,
    position_east_m: float,
    goal_north_m: float,
    goal_east_m: float,
    yaw_rad: float,
) -> tuple[float, float, float, float]:
    """Return body unit direction, raw distance, and V1 normalized distance."""
    delta_north = float(goal_north_m) - float(position_north_m)
    delta_east = float(goal_east_m) - float(position_east_m)
    distance = math.hypot(delta_north, delta_east)
    if not math.isfinite(distance):
        raise ValueError("goal delta must be finite")
    if distance <= 1e-6:
        direction = (0.0, 0.0)
    else:
        direction = ned_to_body(
            delta_north / distance,
            delta_east / distance,
            yaw_rad,
        )
    return (
        direction[0],
        direction[1],
        distance,
        min(max(distance / GOAL_DISTANCE_NORMALIZER_M, 0.0), 1.0),
    )


def normalize_action(
    forward_mps: float, right_mps: float, yaw_rate_radps: float
) -> tuple[float, float, float]:
    """Map physical body action to the unchanged city BC target contract."""
    physical = (float(forward_mps), float(right_mps), float(yaw_rate_radps))
    if not all(math.isfinite(value) for value in physical):
        raise ValueError("expert action must be finite")
    return tuple(
        min(max(value / limit, -1.0), 1.0)
        for value, limit in zip(physical, ACTION_LIMITS)
    )


def episode_outcome_success(
    terminal_state: str,
    failure_reason: str,
    goal_reached: bool,
    landing_commanded: bool,
    landed_after_landing_command: bool,
) -> bool:
    """Evaluate terminal success from accumulated mission evidence."""
    return bool(
        terminal_state == "COMPLETE"
        and not failure_reason
        and goal_reached
        and landing_commanded
        and landed_after_landing_command
    )


def contract_manifest() -> dict:
    """Return the machine-readable immutable part of the V1 contract."""
    return {
        "dataset_version": DATASET_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "master_image": {
            "camera": "fpv",
            "color_space": "RGB",
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "format": IMAGE_FORMAT,
            "jpeg_quality": JPEG_QUALITY,
            "sampling_rate_hz": SAMPLE_RATE_HZ,
        },
        "encoder_preprocessing": {
            "conversion": "PIL RGB",
            "resize": "bilinear to 128x72",
            "crop": "none",
            "tensor_layout": "CHW",
            "pixel_scaling": "uint8 [0,255] divided by 255 to float32 [0,1]",
            "latent_dimension": 64,
        },
        "state": {
            "body_velocity": "[forward,right] m/s body FR",
            "goal_direction": "horizontal unit vector [forward,right] body FR",
            "goal_distance": "min(max(raw_distance_m / 10.0, 0.0), 1.0)",
            "previous_action": (
                "immediately preceding applied ASTAR command, normalized"
            ),
            "state_dimension": 8,
            "observation_dimension": 72,
        },
        "expert_action": {
            "physical": "[v_forward_mps,v_right_mps,yaw_rate_radps]",
            "target": "physical / [1.0,0.8,1.0], clipped to [-1,1]",
            "limits": list(ACTION_LIMITS),
            "dimension": 3,
        },
        "synchronization": {
            "anchor": "image header timestamp",
            "rule": "nearest state/action/mux/flight sample by ROS timestamp",
            "maximum_absolute_error_s": SYNCHRONIZATION_TOLERANCE_S,
            "over_tolerance": "reject and count; never append to samples.csv",
            "static_goal": (
                "latest durable mission goal; frame conversion recorded"
            ),
        },
    }
