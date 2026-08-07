"""Independent validator for diagnostic PX4 setpoint candidates."""

import math

from uav_px4_control.px4_boundary_models import (
    CandidateValidation,
    PX4_NED_FRAME,
    Px4MappingConfig,
    Px4VelocitySetpointCandidate,
    UINT64_MAX,
    source_is_known,
)


def _all_nan(values: tuple[float, float, float]) -> bool:
    return all(math.isnan(value) for value in values)


def validate_px4_candidate(
    candidate: Px4VelocitySetpointCandidate,
    config: Px4MappingConfig,
    current_time_s: float,
    mux_valid: bool,
    mux_active_source: str,
    previous_timestamp_us: int | None = None,
) -> CandidateValidation:
    """Validate mapping, freshness, limits, NaN fields, and mux consistency."""
    velocity = candidate.velocity_ned_mps
    horizontal = math.hypot(velocity[0], velocity[1])
    total = math.sqrt(sum(value * value for value in velocity))

    def reject(reason: str) -> CandidateValidation:
        return CandidateValidation(False, reason, horizontal, total)

    if not candidate.valid:
        return reject(candidate.rejection_reason or "mapping rejected")
    if candidate.frame_id != PX4_NED_FRAME and config.require_px4_ned_frame:
        return reject("candidate frame must be px4_ned")
    if config.require_known_source and not source_is_known(candidate.source):
        return reject("candidate source is unknown")
    if not isinstance(candidate.timestamp_us, int):
        return reject("candidate timestamp must be integer microseconds")
    if candidate.timestamp_us < 0 or candidate.timestamp_us > UINT64_MAX:
        return reject("candidate timestamp is outside uint64")
    if (
        previous_timestamp_us is not None
        and candidate.timestamp_us <= previous_timestamp_us
    ):
        return reject("candidate timestamp is not strictly monotonic")
    age = float(current_time_s) - candidate.selected_receipt_time_s
    if not math.isfinite(age) or age < 0.0:
        return reject("selected command receipt time is invalid")
    if age > config.selected_command_timeout_s:
        return reject("selected command is stale")
    if config.require_mux_valid and not mux_valid:
        return reject("mux reports invalid selected command")
    if candidate.source != mux_active_source:
        return reject("candidate source disagrees with mux active source")
    if not all(math.isfinite(value) for value in velocity):
        return reject("candidate velocity is non-finite")
    if not math.isfinite(candidate.yaw_rate_ned_radps):
        return reject("candidate yaw rate is non-finite")
    if not _all_nan(candidate.position_ned_m):
        return reject("position must be unused NaN")
    if not _all_nan(candidate.acceleration_ned_mps2):
        return reject("acceleration must be unused NaN")
    if not _all_nan(candidate.jerk_ned_mps3):
        return reject("jerk must be unused NaN")
    if not math.isnan(candidate.yaw_ned_rad):
        return reject("absolute yaw must be unused NaN")
    if not candidate.use_velocity or candidate.use_position:
        return reject("candidate is not velocity-only")
    if candidate.use_acceleration or candidate.use_yaw:
        return reject("unused control modes are enabled")
    if not candidate.use_yaw_rate:
        return reject("yaw-rate mapping is disabled")
    if abs(velocity[0]) > config.maximum_north_velocity_mps:
        return reject("north velocity exceeds boundary limit")
    if abs(velocity[1]) > config.maximum_east_velocity_mps:
        return reject("east velocity exceeds boundary limit")
    if abs(velocity[2]) > config.maximum_down_velocity_mps:
        return reject("down velocity exceeds boundary limit")
    if horizontal > config.maximum_horizontal_velocity_mps:
        return reject("horizontal velocity exceeds boundary limit")
    if total > config.maximum_total_velocity_mps:
        return reject("total velocity exceeds boundary limit")
    if abs(candidate.yaw_rate_ned_radps) > config.maximum_yaw_rate_radps:
        return reject("yaw rate exceeds boundary limit")
    return CandidateValidation(True, "candidate accepted", horizontal, total)
