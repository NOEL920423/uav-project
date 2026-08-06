"""Typed ROS-independent trajectory configuration and result models."""

import math
from dataclasses import dataclass

from uav_navigation.models import Point3D


@dataclass(frozen=True, slots=True)
class TrajectoryConfig:
    """Conservative offline dynamic limits with SI units."""

    maximum_speed_mps: float = 2.0
    maximum_longitudinal_acceleration_mps2: float = 1.5
    maximum_longitudinal_deceleration_mps2: float = 1.5
    maximum_lateral_acceleration_mps2: float = 1.5
    maximum_jerk_mps3: float = 3.0
    maximum_yaw_rate_radps: float = 1.5
    maximum_yaw_acceleration_radps2: float = 2.0
    start_speed_mps: float = 0.0
    end_speed_mps: float = 0.0
    minimum_segment_time_s: float = 0.001
    minimum_speed_mps: float = 0.001
    curvature_epsilon: float = 1e-9
    maximum_time_scaling_iterations: int = 8
    maximum_total_time_scale: float = 100.0
    trajectory_minimum_points: int = 2
    require_zero_start_speed: bool = True
    require_zero_end_speed: bool = True

    def __post_init__(self) -> None:
        """Reject non-finite, contradictory, or unsafe configuration."""
        positive = (
            "maximum_speed_mps",
            "maximum_longitudinal_acceleration_mps2",
            "maximum_longitudinal_deceleration_mps2",
            "maximum_lateral_acceleration_mps2",
            "maximum_jerk_mps3",
            "maximum_yaw_rate_radps",
            "maximum_yaw_acceleration_radps2",
            "minimum_segment_time_s",
            "minimum_speed_mps",
            "curvature_epsilon",
            "maximum_total_time_scale",
        )
        nonnegative = ("start_speed_mps", "end_speed_mps")
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            if value > self.maximum_speed_mps:
                raise ValueError(f"{name} must not exceed maximum speed")
            object.__setattr__(self, name, value)
        for name in (
            "maximum_time_scaling_iterations",
            "trajectory_minimum_points",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.maximum_time_scaling_iterations < 1:
            raise ValueError(
                "maximum_time_scaling_iterations must be positive"
            )
        if self.trajectory_minimum_points < 2:
            raise ValueError("trajectory_minimum_points must be at least two")
        if self.minimum_speed_mps > self.maximum_speed_mps:
            raise ValueError("minimum_speed_mps must not exceed maximum speed")
        if self.maximum_total_time_scale < 1.0:
            raise ValueError("maximum_total_time_scale must be at least one")
        for name in (
            "require_zero_start_speed",
            "require_zero_end_speed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.require_zero_start_speed and self.start_speed_mps != 0.0:
            raise ValueError("required zero start speed conflicts with value")
        if self.require_zero_end_speed and self.end_speed_mps != 0.0:
            raise ValueError("required zero end speed conflicts with value")


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """One point of a time-parameterized NED trajectory."""

    time_from_start_s: float
    position: Point3D
    velocity: Point3D
    acceleration: Point3D
    jerk: Point3D
    yaw_ned: float
    yaw_rate_radps: float
    yaw_acceleration_radps2: float
    arc_length_m: float
    curvature_inverse_m: float


@dataclass(frozen=True, slots=True)
class TrajectoryDiagnostic:
    """Independent validation failure with measured and allowed values."""

    constraint: str
    point_index: int
    measured_value: float
    limit_value: float
    message: str


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    """Quantitative measurements derived from a timed trajectory."""

    point_count: int = 0
    path_length_m: float = 0.0
    total_duration_s: float = 0.0
    maximum_speed_mps: float = 0.0
    maximum_longitudinal_acceleration_mps2: float = 0.0
    maximum_lateral_acceleration_mps2: float = 0.0
    maximum_jerk_mps3: float = 0.0
    maximum_yaw_rate_radps: float = 0.0
    maximum_yaw_acceleration_radps2: float = 0.0
    start_speed_mps: float = 0.0
    end_speed_mps: float = 0.0


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    """Structured parameterization and independent validation result."""

    success: bool = False
    valid: bool = False
    status_message: str = "not attempted"
    rejection_reason: str = ""
    trajectory_points: tuple[TrajectoryPoint, ...] = ()
    source_path_point_count: int = 0
    output_trajectory_point_count: int = 0
    path_length_m: float = 0.0
    total_duration_s: float = 0.0
    time_scale: float = 1.0
    maximum_speed_mps: float = 0.0
    maximum_longitudinal_acceleration_mps2: float = 0.0
    maximum_lateral_acceleration_mps2: float = 0.0
    maximum_jerk_mps3: float = 0.0
    maximum_yaw_rate_radps: float = 0.0
    maximum_yaw_acceleration_radps2: float = 0.0
    start_speed_mps: float = 0.0
    end_speed_mps: float = 0.0
    validation_diagnostics: tuple[TrajectoryDiagnostic, ...] = ()
