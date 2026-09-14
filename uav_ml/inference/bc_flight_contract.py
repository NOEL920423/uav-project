"""Dependency-light observation and action contracts for live BC flight."""

from __future__ import annotations

import math
from io import BytesIO

import numpy as np
from PIL import Image

from uav_ml.navigation_imports import add_data_recorder_source_path


add_data_recorder_source_path()
from uav_data_recorder.expert_dataset_contract import (  # noqa: E402
    ACTION_LIMITS,
    goal_features,
    ned_to_body,
    yaw_from_quaternion,
)


IMAGE_SOURCE_ALIASES = {
    "top": "top_rgb",
    "top_rgb": "top_rgb",
    "fpv_rgb": "fpv_rgb",
    "fpv_depth": "fpv_depth",
}
IMPLEMENTED_IMAGE_SOURCES = frozenset({"top_rgb", "fpv_rgb", "fpv_depth"})
LIVE_IMAGE_SIZES = {
    "top_rgb": (640, 360),
    "fpv_rgb": (320, 180),
    "fpv_depth": (320, 180),
}


def canonical_image_source(value: str) -> str:
    """Map legacy source labels to the public runtime source labels."""
    source = str(value).strip().lower()
    if source not in IMAGE_SOURCE_ALIASES:
        supported = ", ".join(sorted(IMAGE_SOURCE_ALIASES))
        raise ValueError(
            f"unsupported image source {value!r}; expected one of {supported}"
        )
    return IMAGE_SOURCE_ALIASES[source]


def validate_live_image(image_bytes: bytes, image_source: str) -> None:
    """Require the formal source-specific live image format and resolution."""
    source = canonical_image_source(image_source)
    expected = LIVE_IMAGE_SIZES.get(source)
    if expected is None:
        raise ValueError(f"no live image contract is registered for {source!r}")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if source in {"top_rgb", "fpv_rgb"}:
                if image.format != "JPEG":
                    raise ValueError("live RGB image must use JPEG compression")
                if image.mode != "RGB":
                    raise ValueError("live RGB image must contain three RGB channels")
            elif image.format != "PNG":
                raise ValueError("live FPV depth image must use PNG compression")
            elif image.mode not in {"I", "I;16", "I;16B", "I;16L"}:
                raise ValueError("live FPV depth image must be uint16")
            elif np.asarray(image).ndim != 2:
                raise ValueError("live FPV depth image must be single-channel uint16")
            if image.size != expected:
                raise ValueError(
                    f"{source} image must be {expected[0]}x{expected[1]}, "
                    f"got {image.size[0]}x{image.size[1]}"
                )
            image.load()
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"live {source} image decode failed: {error}") from error


def build_state8(
    velocity_north_mps: float,
    velocity_east_mps: float,
    position_north_m: float,
    position_east_m: float,
    goal_north_m: float,
    goal_east_m: float,
    yaw_ned_rad: float,
    previous_normalized_action: tuple[float, float, float],
) -> np.ndarray:
    """Build the exact live body-state vector used during training."""
    previous = tuple(float(value) for value in previous_normalized_action)
    values = (
        velocity_north_mps,
        velocity_east_mps,
        position_north_m,
        position_east_m,
        goal_north_m,
        goal_east_m,
        yaw_ned_rad,
        *previous,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("state8 inputs must be finite")
    if len(previous) != 3 or any(abs(value) > 1.000001 for value in previous):
        raise ValueError("previous BC action must be normalized to [-1, 1]")
    body_forward, body_right = ned_to_body(
        velocity_north_mps, velocity_east_mps, yaw_ned_rad
    )
    goal_forward, goal_right, _, normalized_distance = goal_features(
        position_north_m,
        position_east_m,
        goal_north_m,
        goal_east_m,
        yaw_ned_rad,
    )
    return np.asarray((
        body_forward,
        body_right,
        goal_forward,
        goal_right,
        normalized_distance,
        *previous,
    ), dtype=np.float32)


def body_action_to_ned(
    normalized_action: np.ndarray | tuple[float, float, float],
    yaw_ned_rad: float,
    down_velocity_mps: float = 0.0,
) -> tuple[float, float, float, float]:
    """Convert normalized body forward/right/yaw into physical PX4 NED."""
    action = np.asarray(normalized_action, dtype=np.float64)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError("BC action must be a finite normalized 3-vector")
    if not math.isfinite(float(yaw_ned_rad)):
        raise ValueError("NED yaw must be finite")
    clipped = np.clip(action, -1.0, 1.0)
    forward = float(clipped[0]) * ACTION_LIMITS[0]
    right = float(clipped[1]) * ACTION_LIMITS[1]
    cosine = math.cos(float(yaw_ned_rad))
    sine = math.sin(float(yaw_ned_rad))
    north = cosine * forward - sine * right
    east = sine * forward + cosine * right
    return north, east, float(down_velocity_mps), (
        float(clipped[2]) * ACTION_LIMITS[2]
    )


def freshness_error(
    now_s: float,
    image_receipt_s: float | None,
    odometry_receipt_s: float | None,
    goal_available: bool,
    image_timeout_s: float,
    odometry_timeout_s: float,
    image_source: str = "top_rgb",
) -> str | None:
    """Return a stable reason when live observations are incomplete or stale."""
    source = canonical_image_source(image_source)
    if image_receipt_s is None:
        return f"waiting_for_{source}"
    if odometry_receipt_s is None:
        return "waiting_for_odometry"
    if not goal_available:
        return "waiting_for_goal"
    if now_s - image_receipt_s > image_timeout_s:
        return f"stale_{source}"
    if now_s - odometry_receipt_s > odometry_timeout_s:
        return "stale_odometry"
    return None
