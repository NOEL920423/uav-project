"""Pure adapter from Phase 7 candidates to local PX4 v1.14 messages."""

import math
from dataclasses import dataclass

from uav_px4_control.px4_stream_models import StreamCandidate


@dataclass(frozen=True, slots=True)
class TrajectorySetpointFields:
    """Exact local TrajectorySetpoint values without a px4_msgs import."""

    timestamp: int
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    jerk: tuple[float, float, float]
    yaw: float
    yawspeed: float


@dataclass(frozen=True, slots=True)
class OffboardControlModeFields:
    """Exact local velocity-only OffboardControlMode values."""

    timestamp: int
    position: bool = False
    velocity: bool = True
    acceleration: bool = False
    attitude: bool = False
    body_rate: bool = False
    actuator: bool = False


def validate_stream_candidate(candidate: StreamCandidate) -> tuple[bool, str]:
    """Independently recheck the Phase 7 message at the live boundary."""
    if not candidate.valid:
        return False, "Phase 7 candidate is invalid"
    if candidate.frame_id != "px4_ned":
        return False, "candidate frame must be px4_ned"
    if (
        not isinstance(candidate.timestamp_us, int)
        or candidate.timestamp_us <= 0
    ):
        return False, "candidate timestamp must be a positive integer"
    values = (*candidate.velocity_ned_mps, candidate.yaw_rate_ned_radps)
    if not all(math.isfinite(value) for value in values):
        return False, "candidate commanded fields must be finite"
    north, east, down = candidate.velocity_ned_mps
    if abs(north) > 2.0 or abs(east) > 2.0 or abs(down) > 1.0:
        return False, "candidate component velocity exceeds Phase 7 limits"
    if math.hypot(north, east) > 2.0:
        return False, "candidate horizontal velocity exceeds Phase 7 limit"
    if math.sqrt(north * north + east * east + down * down) > 2.0:
        return False, "candidate total velocity exceeds Phase 7 limit"
    if abs(candidate.yaw_rate_ned_radps) > 1.5:
        return False, "candidate yaw rate exceeds Phase 7 limit"
    return True, "candidate mapping valid"


def trajectory_setpoint_fields(
    candidate: StreamCandidate,
    outgoing_timestamp_us: int,
) -> TrajectorySetpointFields:
    """Identity-map NED velocity/yaw rate and disable every unused field."""
    valid, reason = validate_stream_candidate(candidate)
    if not valid:
        raise ValueError(reason)
    if (
        not isinstance(outgoing_timestamp_us, int)
        or outgoing_timestamp_us <= 0
    ):
        raise ValueError("outgoing timestamp must be a positive integer")
    unused = (math.nan, math.nan, math.nan)
    return TrajectorySetpointFields(
        timestamp=outgoing_timestamp_us,
        position=unused,
        velocity=candidate.velocity_ned_mps,
        acceleration=unused,
        jerk=unused,
        yaw=math.nan,
        yawspeed=candidate.yaw_rate_ned_radps,
    )


def offboard_control_mode_fields(
    outgoing_timestamp_us: int,
) -> OffboardControlModeFields:
    """Select only velocity intent without requesting OFFBOARD mode."""
    if (
        not isinstance(outgoing_timestamp_us, int)
        or outgoing_timestamp_us <= 0
    ):
        raise ValueError("outgoing timestamp must be a positive integer")
    return OffboardControlModeFields(timestamp=outgoing_timestamp_us)
