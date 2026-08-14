"""ROS-independent models for the Phase 7 PX4 output boundary."""

import math
from dataclasses import dataclass
from enum import Enum

from uav_px4_control.control_source_models import CONTROL_SOURCES


PX4_NED_FRAME = "px4_ned"
UINT64_MAX = (1 << 64) - 1


class Px4OutputGateState(str, Enum):
    """Explicit non-flight output-gate states."""

    OUTPUT_DISABLED = "OUTPUT_DISABLED"
    WAITING_SELECTED_COMMAND = "WAITING_SELECTED_COMMAND"
    WAITING_MUX_HEALTH = "WAITING_MUX_HEALTH"
    WAITING_TELEMETRY = "WAITING_TELEMETRY"
    WAITING_VEHICLE_STATE = "WAITING_VEHICLE_STATE"
    READY_DISABLED = "READY_DISABLED"
    ENABLE_PENDING = "ENABLE_PENDING"
    SAFE_TO_FORWARD = "SAFE_TO_FORWARD"
    DISABLED_STALE_COMMAND = "DISABLED_STALE_COMMAND"
    DISABLED_STALE_TELEMETRY = "DISABLED_STALE_TELEMETRY"
    DISABLED_INVALID_COMMAND = "DISABLED_INVALID_COMMAND"
    DISABLED_MUX_HOLD = "DISABLED_MUX_HOLD"
    DISABLED_FAILSAFE = "DISABLED_FAILSAFE"
    DISABLED_TIME_JUMP = "DISABLED_TIME_JUMP"
    DISABLED_STATE_CHANGE = "DISABLED_STATE_CHANGE"
    LATCHED_FAULT = "LATCHED_FAULT"


@dataclass(frozen=True, slots=True)
class Px4MappingConfig:
    """Conservative mapping and gate limits no looser than Phase 6."""

    maximum_north_velocity_mps: float = 2.0
    maximum_east_velocity_mps: float = 2.0
    maximum_down_velocity_mps: float = 1.0
    maximum_horizontal_velocity_mps: float = 2.0
    maximum_total_velocity_mps: float = 2.0
    maximum_yaw_rate_radps: float = 1.5
    selected_command_timeout_s: float = 0.25
    telemetry_timeout_s: float = 0.50
    require_mux_valid: bool = True
    require_known_source: bool = True
    require_px4_ned_frame: bool = True
    latch_faults: bool = True
    require_explicit_enable: bool = True
    lock_vehicle_state_signature: bool = True

    def __post_init__(self) -> None:
        """Reject expanded, non-finite, or contradictory limits."""
        positive = (
            "maximum_north_velocity_mps",
            "maximum_east_velocity_mps",
            "maximum_down_velocity_mps",
            "maximum_horizontal_velocity_mps",
            "maximum_total_velocity_mps",
            "maximum_yaw_rate_radps",
            "selected_command_timeout_s",
            "telemetry_timeout_s",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        boolean_fields = (
            "require_mux_valid",
            "require_known_source",
            "require_px4_ned_frame",
            "latch_faults",
            "require_explicit_enable",
            "lock_vehicle_state_signature",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.maximum_horizontal_velocity_mps > 2.0:
            raise ValueError("horizontal limit may not exceed Phase 6")
        if self.maximum_total_velocity_mps > 2.0:
            raise ValueError("total limit may not exceed Phase 6")
        if self.maximum_down_velocity_mps > 1.0:
            raise ValueError("vertical limit may not exceed Phase 6")
        if self.maximum_yaw_rate_radps > 1.5:
            raise ValueError("yaw-rate limit may not exceed Phase 6")


@dataclass(frozen=True, slots=True)
class Px4VelocitySetpointCandidate:
    """Diagnostic mirror of the local velocity-only TrajectorySetpoint."""

    timestamp_us: int
    position_ned_m: tuple[float, float, float]
    velocity_ned_mps: tuple[float, float, float]
    acceleration_ned_mps2: tuple[float, float, float]
    jerk_ned_mps3: tuple[float, float, float]
    yaw_ned_rad: float
    yaw_rate_ned_radps: float
    source: str
    frame_id: str
    selected_receipt_time_s: float
    use_position: bool = False
    use_velocity: bool = True
    use_acceleration: bool = False
    use_yaw: bool = False
    use_yaw_rate: bool = True
    valid: bool = True
    rejection_reason: str = ""


@dataclass(frozen=True, slots=True)
class CandidateValidation:
    """Independent PX4-candidate validation result."""

    valid: bool
    reason: str
    horizontal_speed_mps: float
    total_speed_mps: float


@dataclass(frozen=True, slots=True)
class Px4TelemetryState:
    """Synthetic evidence derivable from local PX4 status messages."""

    receipt_time_s: float
    timestamp_us: int
    connected: bool = True
    arming_state: int = 1
    nav_state: int = 0
    offboard_control_signal_lost: bool = False
    offboard_active: bool = False
    failsafe: bool = False
    pre_flight_checks_pass: bool = True
    local_position_valid: bool = True
    local_velocity_valid: bool = True
    odometry_valid: bool = True
    pose_frame_ned: bool = True
    velocity_frame_ned: bool = True


@dataclass(frozen=True, slots=True)
class MuxHealthEvidence:
    """Phase 6 status evidence consumed by the independent gate."""

    received: bool
    selected_command_valid: bool
    hold_active: bool
    active_source: str
    receipt_time_s: float


@dataclass(frozen=True, slots=True)
class Px4OutputGateResult:
    """One deterministic output-boundary decision without forwarding."""

    enabled: bool
    safe_to_forward: bool
    state: Px4OutputGateState
    hold_reason: str
    selected_command_valid: bool
    mux_valid: bool
    telemetry_valid: bool
    failsafe: bool
    active_source: str
    selected_command_age_s: float
    telemetry_age_s: float
    transition_count: int
    fault_latched: bool


@dataclass(frozen=True, slots=True)
class GateEnableResponse:
    """Result of an explicit output-enable or disable request."""

    accepted: bool
    enabled: bool
    message: str


def source_is_known(source: str) -> bool:
    """Return whether a source is in the canonical Phase 6 vocabulary."""
    return source in CONTROL_SOURCES
