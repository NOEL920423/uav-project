"""ROS-independent models for deterministic control-source arbitration."""

import math
from dataclasses import dataclass
from enum import Enum


VALID_COMMAND_FRAME = "px4_ned"
HOLD = "HOLD"
ASTAR_EXPERT = "ASTAR_EXPERT"
HUMAN_JOYSTICK = "HUMAN_JOYSTICK"
NAVRL_POLICY = "NAVRL_POLICY"
MOVEMENT_SOURCES = (ASTAR_EXPERT, HUMAN_JOYSTICK, NAVRL_POLICY)
CONTROL_SOURCES = (HOLD,) + MOVEMENT_SOURCES
SOURCE_TOPICS = {
    ASTAR_EXPERT: "/uav/control/astar_command",
    HUMAN_JOYSTICK: "/uav/control/joystick_command",
    NAVRL_POLICY: "/uav/control/navrl_command",
    HOLD: "/uav/control/hold_command",
}


class ControlMuxState(str, Enum):
    """Explicit Phase 6 mux states."""

    HOLD_STARTUP = "HOLD_STARTUP"
    HOLD_REQUESTED = "HOLD_REQUESTED"
    HOLD_WAITING_SOURCE = "HOLD_WAITING_SOURCE"
    HOLD_SWITCH_BARRIER = "HOLD_SWITCH_BARRIER"
    ACTIVE_ASTAR_EXPERT = "ACTIVE_ASTAR_EXPERT"
    ACTIVE_HUMAN_JOYSTICK = "ACTIVE_HUMAN_JOYSTICK"
    ACTIVE_NAVRL_POLICY = "ACTIVE_NAVRL_POLICY"
    HOLD_STALE_SOURCE = "HOLD_STALE_SOURCE"
    HOLD_INVALID_SOURCE = "HOLD_INVALID_SOURCE"
    HOLD_INVALID_COMMAND = "HOLD_INVALID_COMMAND"
    HOLD_WRONG_FRAME = "HOLD_WRONG_FRAME"
    HOLD_TIME_JUMP = "HOLD_TIME_JUMP"
    HOLD_LATCHED_FAULT = "HOLD_LATCHED_FAULT"


ACTIVE_STATES = {
    ASTAR_EXPERT: ControlMuxState.ACTIVE_ASTAR_EXPERT,
    HUMAN_JOYSTICK: ControlMuxState.ACTIVE_HUMAN_JOYSTICK,
    NAVRL_POLICY: ControlMuxState.ACTIVE_NAVRL_POLICY,
}


@dataclass(frozen=True, slots=True)
class Vector3:
    """Minimal three-dimensional vector without ROS dependencies."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class ControlCommand:
    """Candidate or selected NED velocity and yaw-rate command."""

    source: str
    timestamp_s: float
    frame_id: str
    linear: Vector3
    angular_x: float = 0.0
    angular_y: float = 0.0
    yaw_rate_radps: float = 0.0
    hold_active: bool = False
    hold_reason: str = ""


@dataclass(frozen=True, slots=True)
class ControlMuxConfig:
    """Locked Phase 6 source, timeout, handoff, and command limits."""

    default_source: str = HOLD
    publish_rate_hz: float = 50.0
    astar_timeout_s: float = 0.25
    joystick_timeout_s: float = 0.20
    navrl_timeout_s: float = 0.20
    hold_timeout_s: float = 0.50
    switch_hold_duration_s: float = 0.10
    minimum_source_dwell_time_s: float = 0.20
    maximum_selected_speed_mps: float = 2.0
    maximum_selected_horizontal_speed_mps: float = 2.0
    maximum_selected_vertical_speed_mps: float = 1.0
    maximum_selected_acceleration_mps2: float = 1.5
    maximum_selected_yaw_rate_radps: float = 1.5
    maximum_selected_yaw_acceleration_radps2: float = 2.0
    reject_wrong_frame: bool = True
    require_monotonic_candidate_stamps: bool = True
    require_fresh_command_before_switch: bool = True
    latch_hold_after_fault: bool = True
    hold_command_epsilon: float = 1e-9

    def __post_init__(self) -> None:
        """Reject non-finite, negative, or contradictory configuration."""
        if self.default_source != HOLD:
            raise ValueError("Phase 6 default_source must be HOLD")
        positive = (
            "publish_rate_hz",
            "astar_timeout_s",
            "joystick_timeout_s",
            "navrl_timeout_s",
            "hold_timeout_s",
            "maximum_selected_speed_mps",
            "maximum_selected_horizontal_speed_mps",
            "maximum_selected_vertical_speed_mps",
            "maximum_selected_acceleration_mps2",
            "maximum_selected_yaw_rate_radps",
            "maximum_selected_yaw_acceleration_radps2",
        )
        nonnegative = (
            "switch_hold_duration_s",
            "minimum_source_dwell_time_s",
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
        boolean_fields = (
            "reject_wrong_frame",
            "require_monotonic_candidate_stamps",
            "require_fresh_command_before_switch",
            "latch_hold_after_fault",
        )
        for name in boolean_fields:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if (
            self.maximum_selected_horizontal_speed_mps
            > self.maximum_selected_speed_mps
        ):
            raise ValueError("horizontal speed limit exceeds total limit")
        if (
            self.maximum_selected_vertical_speed_mps
            > self.maximum_selected_speed_mps
        ):
            raise ValueError("vertical speed limit exceeds total limit")

    def timeout_for(self, source: str) -> float:
        """Return the exact configured receipt timeout for one source."""
        timeouts = {
            ASTAR_EXPERT: self.astar_timeout_s,
            HUMAN_JOYSTICK: self.joystick_timeout_s,
            NAVRL_POLICY: self.navrl_timeout_s,
            HOLD: self.hold_timeout_s,
        }
        if source not in timeouts:
            raise ValueError(f"unknown control source: {source}")
        return timeouts[source]


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """Latest candidate plus independent receipt and validation evidence."""

    source: str
    command: ControlCommand | None = None
    receipt_time_s: float | None = None
    message_stamp_s: float | None = None
    finite: bool = False
    valid: bool = False
    reason: str = "never received"
    update_count: int = 0


@dataclass(frozen=True, slots=True)
class SourceHealth:
    """Health classification at one node-clock instant."""

    source: str
    received: bool
    finite: bool
    valid: bool
    fresh: bool
    age_s: float
    reason: str
    update_count: int

    @property
    def healthy(self) -> bool:
        """Return true only for a received, valid, fresh candidate."""
        return self.received and self.finite and self.valid and self.fresh


@dataclass(frozen=True, slots=True)
class SelectedCommandDiagnostic:
    """One independent selected-command validation failure."""

    constraint: str
    source: str
    measured_value: float
    limit_value: float
    cycle_index: int
    timestamp_s: float
    reason: str


@dataclass(frozen=True, slots=True)
class SelectionResponse:
    """Pure equivalent of the source-selection service response."""

    accepted: bool
    requested_source: str
    active_source: str
    status_message: str


@dataclass(frozen=True, slots=True)
class MuxSaturationFlags:
    """Ordered selected-command limiting decisions."""

    horizontal_speed: bool = False
    vertical_speed: bool = False
    total_speed: bool = False
    acceleration: bool = False
    yaw_rate: bool = False
    yaw_acceleration: bool = False

    @property
    def count(self) -> int:
        """Return the number of active limit categories."""
        return sum((
            self.horizontal_speed,
            self.vertical_speed,
            self.total_speed,
            self.acceleration,
            self.yaw_rate,
            self.yaw_acceleration,
        ))


@dataclass(frozen=True, slots=True)
class ControlMuxResult:
    """One complete deterministic mux publication cycle."""

    state: ControlMuxState
    requested_source: str
    active_source: str
    selected_command: ControlCommand
    selected_command_valid: bool
    hold_active: bool
    hold_reason: str
    switch_in_progress: bool
    switch_remaining_time_s: float
    selected_source_age_s: float
    transition_count: int
    healthy_sources: tuple[str, ...]
    stale_sources: tuple[str, ...]
    saturations: MuxSaturationFlags = MuxSaturationFlags()
    diagnostics: tuple[SelectedCommandDiagnostic, ...] = ()
    fault_latched: bool = False
    status_message: str = ""


def command_speed(command: ControlCommand) -> float:
    """Return total linear speed."""
    vector = command.linear
    return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)


def zero_hold(timestamp_s: float, reason: str) -> ControlCommand:
    """Construct the always-available internal exact-zero HOLD."""
    if not reason:
        raise ValueError("internal HOLD requires a specific reason")
    return ControlCommand(
        source=HOLD,
        timestamp_s=float(timestamp_s),
        frame_id=VALID_COMMAND_FRAME,
        linear=Vector3(0.0, 0.0, 0.0),
        hold_active=True,
        hold_reason=reason,
    )
