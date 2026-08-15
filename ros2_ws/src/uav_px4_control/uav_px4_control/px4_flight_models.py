"""ROS-independent contracts for one guarded PX4 SITL flight milestone."""

import math
from dataclasses import dataclass
from enum import Enum


class Px4FlightState(str, Enum):
    """Observable mission phases with explicit terminal states."""

    DISABLED = "DISABLED"
    WAITING_PIPELINE = "WAITING_PIPELINE"
    SELECTING_ASTAR = "SELECTING_ASTAR"
    ENABLING_OUTPUT_GATE = "ENABLING_OUTPUT_GATE"
    ENABLING_STREAM = "ENABLING_STREAM"
    PRESTREAMING = "PRESTREAMING"
    REQUESTING_OFFBOARD = "REQUESTING_OFFBOARD"
    REQUESTING_ARM = "REQUESTING_ARM"
    STARTING_TAKEOFF = "STARTING_TAKEOFF"
    TAKEOFF = "TAKEOFF"
    REPLANNING = "REPLANNING"
    STARTING_TRACKING = "STARTING_TRACKING"
    TRACKING = "TRACKING"
    GOAL_HOLD = "GOAL_HOLD"
    LANDING = "LANDING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class Px4FlightConfig:
    """Timing and physical acceptance limits for the SITL-only mission."""

    minimum_stream_rate_hz: float = 19.0
    takeoff_altitude_m: float = 2.0
    takeoff_altitude_tolerance_m: float = 0.20
    goal_tolerance_m: float = 0.35
    pipeline_timeout_s: float = 20.0
    prestream_timeout_s: float = 15.0
    offboard_timeout_s: float = 8.0
    arm_timeout_s: float = 8.0
    takeoff_timeout_s: float = 20.0
    replan_timeout_s: float = 10.0
    tracking_timeout_s: float = 45.0
    landing_timeout_s: float = 30.0
    command_retry_s: float = 0.50

    def __post_init__(self) -> None:
        """Reject non-finite or non-positive flight limits."""
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.takeoff_altitude_tolerance_m >= self.takeoff_altitude_m:
            raise ValueError("takeoff tolerance must be below flight altitude")


@dataclass(frozen=True, slots=True)
class FlightEvidence:
    """One coherent evidence snapshot consumed by the pure supervisor."""

    pipeline_ready: bool = False
    mission_trajectory_ready: bool = False
    follower_command_valid: bool = False
    astar_selected: bool = False
    output_gate_ready: bool = False
    output_gate_safe: bool = False
    stream_stable: bool = False
    stream_rate_hz: float = 0.0
    offboard_active: bool = False
    vehicle_armed: bool = False
    altitude_m: float = 0.0
    tracking_active: bool = False
    goal_reached: bool = False
    goal_distance_m: float = math.inf
    landed: bool = True
    telemetry_fresh: bool = False
    source_valid: bool = False
    failsafe: bool = False
    fatal_command_ack: str = ""
    environment_valid: bool = True


@dataclass(frozen=True, slots=True)
class FlightDecision:
    """State-machine result and idempotent actions for the ROS adapter."""

    state: Px4FlightState
    actions: tuple[str, ...]
    failure_reason: str
    transition_count: int


def altitude_above_ground(
    ground_down_m: float, current_down_m: float
) -> float:
    """Measure positive-up altitude from the enable-time NED ground datum."""
    ground = float(ground_down_m)
    current = float(current_down_m)
    if not math.isfinite(ground) or not math.isfinite(current):
        raise ValueError("NED ground and current down must be finite")
    return ground - current


def vehicle_command_was_accepted(last_ack: str, command: int) -> bool:
    """Return true only for the exact PX4 command's accepted ACK."""
    return str(last_ack) == f"{int(command)}:ACCEPTED"
