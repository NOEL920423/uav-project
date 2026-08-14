"""ROS-independent models for fail-closed PX4 SITL setpoint streaming."""

import math
from dataclasses import dataclass
from enum import Enum


class Px4StreamState(str, Enum):
    """Explicit Phase 8 states which never imply flight activation."""

    STREAM_DISABLED = "STREAM_DISABLED"
    WAITING_SITL = "WAITING_SITL"
    WAITING_DDS = "WAITING_DDS"
    WAITING_GATE = "WAITING_GATE"
    WAITING_CANDIDATE = "WAITING_CANDIDATE"
    WAITING_TELEMETRY = "WAITING_TELEMETRY"
    PRESTREAM_READY = "PRESTREAM_READY"
    PRESTREAMING = "PRESTREAMING"
    STREAMING = "STREAMING"
    STOPPED_GATE_FALSE = "STOPPED_GATE_FALSE"
    STOPPED_STALE_CANDIDATE = "STOPPED_STALE_CANDIDATE"
    STOPPED_STALE_GATE = "STOPPED_STALE_GATE"
    STOPPED_STALE_TELEMETRY = "STOPPED_STALE_TELEMETRY"
    STOPPED_DDS_LOSS = "STOPPED_DDS_LOSS"
    STOPPED_FAILSAFE = "STOPPED_FAILSAFE"
    STOPPED_ARMED = "STOPPED_ARMED"
    STOPPED_OFFBOARD_ACTIVE = "STOPPED_OFFBOARD_ACTIVE"
    STOPPED_TIME_JUMP = "STOPPED_TIME_JUMP"
    STOPPED_PUBLISH_GAP = "STOPPED_PUBLISH_GAP"
    STOPPED_INVALID_MAPPING = "STOPPED_INVALID_MAPPING"
    LATCHED_STREAM_FAULT = "LATCHED_STREAM_FAULT"


@dataclass(frozen=True, slots=True)
class Px4StreamConfig:
    """Typed Phase 8 stream timing and safety policy."""

    stream_rate_hz: float = 20.0
    minimum_prestream_duration_s: float = 2.0
    minimum_prestream_messages: int = 40
    maximum_publish_gap_s: float = 0.20
    candidate_timeout_s: float = 0.25
    gate_status_timeout_s: float = 0.50
    telemetry_timeout_s: float = 0.50
    minimum_candidate_updates: int = 3
    require_safe_to_forward: bool = True
    require_gate_status_agreement: bool = True
    require_disarmed: bool = True
    require_offboard_inactive: bool = True
    require_no_failsafe: bool = True
    simulation_mode: bool = True
    allow_sitl_streaming_only: bool = True
    latch_stream_faults: bool = True

    def __post_init__(self) -> None:
        """Reject unsafe, non-finite, or internally inconsistent values."""
        positive = (
            "stream_rate_hz",
            "minimum_prestream_duration_s",
            "maximum_publish_gap_s",
            "candidate_timeout_s",
            "gate_status_timeout_s",
            "telemetry_timeout_s",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if not isinstance(self.minimum_prestream_messages, int):
            raise ValueError("minimum_prestream_messages must be an integer")
        if self.minimum_prestream_messages < 1:
            raise ValueError("minimum_prestream_messages must be positive")
        if not isinstance(self.minimum_candidate_updates, int):
            raise ValueError("minimum_candidate_updates must be an integer")
        if self.minimum_candidate_updates < 2:
            raise ValueError("minimum_candidate_updates must be at least two")
        boolean_fields = (
            "require_safe_to_forward",
            "require_gate_status_agreement",
            "require_disarmed",
            "require_offboard_inactive",
            "require_no_failsafe",
            "simulation_mode",
            "allow_sitl_streaming_only",
            "latch_stream_faults",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.maximum_publish_gap_s <= 1.0 / self.stream_rate_hz:
            raise ValueError("maximum_publish_gap_s must exceed one period")
        if not self.simulation_mode or not self.allow_sitl_streaming_only:
            raise ValueError("Phase 8 permits SITL-only simulation streaming")


@dataclass(frozen=True, slots=True)
class StreamCandidate:
    """Validated Phase 7 candidate evidence received by the streamer."""

    receipt_time_s: float
    timestamp_us: int
    velocity_ned_mps: tuple[float, float, float]
    yaw_rate_ned_radps: float
    frame_id: str = "px4_ned"
    valid: bool = True


@dataclass(frozen=True, slots=True)
class StreamGateEvidence:
    """Phase 7 boolean and detailed status agreement evidence."""

    bool_receipt_time_s: float
    bool_safe_to_forward: bool
    status_receipt_time_s: float
    status_safe_to_forward: bool
    status_state: str


@dataclass(frozen=True, slots=True)
class StreamTelemetry:
    """Aggregated read-only evidence from the four required PX4 topics."""

    oldest_receipt_time_s: float
    newest_timestamp_us: int
    vehicle_armed: bool
    offboard_active: bool
    failsafe: bool
    odometry_valid: bool


@dataclass(frozen=True, slots=True)
class StreamReadiness:
    """External graph, gate, candidate, and telemetry evidence for one tick."""

    sitl_guard_valid: bool
    dds_ready: bool
    gate: StreamGateEvidence | None
    candidate: StreamCandidate | None
    telemetry: StreamTelemetry | None


@dataclass(frozen=True, slots=True)
class Px4StreamResult:
    """State-machine output used by both ROS and offline harnesses."""

    state: Px4StreamState
    stream_enable_requested: bool
    streaming: bool
    should_publish: bool
    sitl_guard_valid: bool
    dds_ready: bool
    gate_valid: bool
    candidate_valid: bool
    telemetry_fresh: bool
    vehicle_armed: bool
    offboard_active: bool
    failsafe: bool
    candidate_age_s: float
    gate_status_age_s: float
    telemetry_age_s: float
    trajectory_setpoint_count: int
    offboard_mode_count: int
    dropped_cycle_count: int
    transition_count: int
    stop_reason: str
