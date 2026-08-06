"""Typed ROS-independent models for offline trajectory tracking."""

import math
from dataclasses import dataclass
from enum import Enum

from uav_navigation.models import Point3D
from uav_navigation.trajectory_models import TrajectoryPoint

VALID_TRACKING_FRAME = "px4_ned"


class TrackingState(str, Enum):
    """Explicit Phase 5 follower states."""

    WAITING_TRAJECTORY = "WAITING_TRAJECTORY"
    WAITING_VALIDITY = "WAITING_VALIDITY"
    WAITING_ODOMETRY = "WAITING_ODOMETRY"
    PRESTART_HOLD = "PRESTART_HOLD"
    TRACKING = "TRACKING"
    GOAL_SETTLING = "GOAL_SETTLING"
    GOAL_HOLD = "GOAL_HOLD"
    HOLD_STALE_TRAJECTORY = "HOLD_STALE_TRAJECTORY"
    HOLD_STALE_ODOMETRY = "HOLD_STALE_ODOMETRY"
    HOLD_INVALID_FRAME = "HOLD_INVALID_FRAME"
    HOLD_TIME_JUMP = "HOLD_TIME_JUMP"
    HOLD_TRACKING_ERROR = "HOLD_TRACKING_ERROR"
    HOLD_INVALID_COMMAND = "HOLD_INVALID_COMMAND"
    TERMINAL_NOT_REACHED = "TERMINAL_NOT_REACHED"


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Conservative offline gains, bounds, timeouts, and goal gates."""

    position_kp: float = 1.0
    velocity_kd: float = 0.2
    yaw_kp: float = 1.5
    maximum_command_speed_mps: float = 2.0
    maximum_horizontal_command_speed_mps: float = 2.0
    maximum_vertical_command_speed_mps: float = 1.0
    maximum_command_acceleration_mps2: float = 1.5
    maximum_yaw_rate_command_radps: float = 1.5
    maximum_yaw_acceleration_command_radps2: float = 2.0
    odometry_timeout_s: float = 0.25
    trajectory_validity_timeout_s: float = 0.50
    trajectory_start_delay_s: float = 0.10
    control_period_s: float = 0.02
    goal_position_tolerance_m: float = 0.15
    goal_velocity_tolerance_mps: float = 0.15
    goal_yaw_tolerance_rad: float = 0.20
    goal_settle_time_s: float = 0.50
    maximum_tracking_error_m: float = 2.0
    maximum_terminal_wait_s: float = 2.0
    reject_wrong_frame: bool = True
    require_validity_topic: bool = True
    hold_command_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        """Reject non-finite, negative, or contradictory configuration."""
        positive = (
            "maximum_command_speed_mps",
            "maximum_horizontal_command_speed_mps",
            "maximum_vertical_command_speed_mps",
            "maximum_command_acceleration_mps2",
            "maximum_yaw_rate_command_radps",
            "maximum_yaw_acceleration_command_radps2",
            "odometry_timeout_s",
            "trajectory_validity_timeout_s",
            "control_period_s",
            "goal_position_tolerance_m",
            "goal_velocity_tolerance_mps",
            "goal_yaw_tolerance_rad",
            "goal_settle_time_s",
            "maximum_tracking_error_m",
            "maximum_terminal_wait_s",
        )
        nonnegative = (
            "position_kp",
            "velocity_kd",
            "yaw_kp",
            "trajectory_start_delay_s",
            "hold_command_epsilon",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        for name in ("reject_wrong_frame", "require_validity_topic"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if (
            self.maximum_horizontal_command_speed_mps
            > self.maximum_command_speed_mps
        ):
            raise ValueError("horizontal speed limit exceeds total limit")
        if (
            self.maximum_vertical_command_speed_mps
            > self.maximum_command_speed_mps
        ):
            raise ValueError("vertical speed limit exceeds total limit")


@dataclass(frozen=True, slots=True)
class VehicleState:
    """One measured NED state and its node-clock receipt timestamp."""

    timestamp_s: float
    frame_id: str
    position: Point3D
    velocity: Point3D
    yaw_ned: float
    yaw_rate_radps: float


@dataclass(frozen=True, slots=True)
class ReferenceSample:
    """Sampled trajectory reference with location flags."""

    point: TrajectoryPoint
    reference_index: int
    prestart: bool = False
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class VelocityCommand:
    """ROS-independent candidate velocity command."""

    timestamp_s: float
    frame_id: str
    linear: Point3D
    yaw_rate_radps: float
    hold_active: bool = False
    hold_reason: str = ""


@dataclass(frozen=True, slots=True)
class TrackingErrors:
    """Independent scalar errors used by status and metrics."""

    position_error_m: float
    horizontal_position_error_m: float
    vertical_position_error_m: float
    along_track_error_m: float
    cross_track_error_m: float
    velocity_error_mps: float
    yaw_error_rad: float


@dataclass(frozen=True, slots=True)
class SaturationFlags:
    """Ordered command-bound decisions."""

    horizontal_speed: bool = False
    vertical_speed: bool = False
    total_speed: bool = False
    acceleration: bool = False
    yaw_rate: bool = False
    yaw_acceleration: bool = False

    @property
    def count(self) -> int:
        """Return the number of active saturation categories."""
        return sum((
            self.horizontal_speed,
            self.vertical_speed,
            self.total_speed,
            self.acceleration,
            self.yaw_rate,
            self.yaw_acceleration,
        ))


@dataclass(frozen=True, slots=True)
class TrackingDiagnostic:
    """Independent command-validation diagnostic."""

    constraint: str
    measured_value: float
    limit_value: float
    timestamp_s: float
    cycle_index: int
    message: str


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """One complete controller/state-machine cycle result."""

    state: TrackingState
    trajectory_valid: bool
    odometry_valid: bool
    command_valid: bool
    trajectory_time_s: float
    reference: ReferenceSample | None
    errors: TrackingErrors | None
    unsaturated_command: VelocityCommand
    selected_command: VelocityCommand
    saturations: SaturationFlags = SaturationFlags()
    diagnostics: tuple[TrackingDiagnostic, ...] = ()
    status_message: str = ""


@dataclass(frozen=True, slots=True)
class OfflineTrackingMetrics:
    """Aggregate deterministic closed-loop tracking measurements."""

    cycle_count: int = 0
    position_rmse_m: float = 0.0
    horizontal_position_rmse_m: float = 0.0
    vertical_position_rmse_m: float = 0.0
    maximum_position_error_m: float = 0.0
    terminal_position_error_m: float = 0.0
    velocity_rmse_mps: float = 0.0
    yaw_rmse_rad: float = 0.0
    maximum_yaw_error_rad: float = 0.0
    maximum_command_speed_mps: float = 0.0
    maximum_command_acceleration_mps2: float = 0.0
    maximum_yaw_rate_radps: float = 0.0
    maximum_yaw_acceleration_radps2: float = 0.0
    saturation_count: int = 0
    hold_cycle_count: int = 0
    stale_detection_latency_s: float = 0.0
    terminal_settling_time_s: float = 0.0
    completion_status: str = "INCOMPLETE"
