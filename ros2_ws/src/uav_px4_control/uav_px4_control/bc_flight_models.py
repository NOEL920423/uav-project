"""ROS-independent lifecycle contracts for one live BC flight."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class BcFlightState(str, Enum):
    """Observable lifecycle states from startup through landing."""

    WAITING_INPUTS = "WAITING_INPUTS"
    SELECTING_LIFECYCLE = "SELECTING_LIFECYCLE"
    ENABLING_OUTPUT = "ENABLING_OUTPUT"
    ENABLING_STREAM = "ENABLING_STREAM"
    REQUESTING_OFFBOARD = "REQUESTING_OFFBOARD"
    REQUESTING_ARM = "REQUESTING_ARM"
    TAKING_OFF = "TAKING_OFF"
    ENABLING_BC = "ENABLING_BC"
    SELECTING_BC = "SELECTING_BC"
    NAVIGATING = "NAVIGATING"
    HOLDING = "HOLDING"
    LANDING = "LANDING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class BcFlightConfig:
    """Timing and fixed-height takeoff settings for one evaluation."""

    flight_altitude_m: float = 1.5
    altitude_tolerance_m: float = 0.20
    takeoff_up_speed_mps: float = 0.55
    readiness_timeout_s: float = 45.0
    service_timeout_s: float = 15.0
    takeoff_timeout_s: float = 25.0
    landing_timeout_s: float = 75.0

    def __post_init__(self) -> None:
        """Reject non-finite and contradictory flight settings."""
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.altitude_tolerance_m >= self.flight_altitude_m:
            raise ValueError(
                "altitude tolerance must be below flight altitude"
            )


@dataclass(frozen=True, slots=True)
class BcFlightEvidence:
    """Actual ROS and PX4 evidence consumed by the lifecycle controller."""

    runtime_ready: bool = False
    observations_ready: bool = False
    lifecycle_selected: bool = False
    bc_enabled: bool = False
    bc_ready: bool = False
    bc_selected: bool = False
    source_valid: bool = False
    output_ready: bool = False
    output_safe: bool = False
    stream_stable: bool = False
    offboard_active: bool = False
    vehicle_armed: bool = False
    landed: bool = True
    telemetry_fresh: bool = False
    failsafe: bool = False
    altitude_m: float = 0.0
    terminal_reason: str = ""


@dataclass(frozen=True, slots=True)
class BcFlightDecision:
    """One state-machine result plus idempotent adapter actions."""

    state: BcFlightState
    actions: tuple[str, ...]
    terminal_reason: str
    failure_reason: str


class BcFlightController:
    """Sequence takeoff, BC handoff, safety HOLD, and landing."""

    def __init__(self, config: BcFlightConfig | None = None) -> None:
        """Create a controller waiting for fresh runtime evidence."""
        self.config = config or BcFlightConfig()
        self.state = BcFlightState.WAITING_INPUTS
        self.terminal_reason = ""
        self.failure_reason = ""
        self._state_started_s = 0.0
        self._last_step_s: float | None = None

    def _set_state(self, state: BcFlightState, now_s: float) -> None:
        self.state = state
        self._state_started_s = now_s

    def _decision(self, *actions: str) -> BcFlightDecision:
        return BcFlightDecision(
            self.state,
            tuple(actions),
            self.terminal_reason,
            self.failure_reason,
        )

    def _abort(self, now_s: float, reason: str) -> BcFlightDecision:
        if not self.failure_reason:
            self.failure_reason = reason
        self.terminal_reason = self.terminal_reason or "runtime_failure"
        self._set_state(BcFlightState.HOLDING, now_s)
        return self._decision("SELECT_HOLD")

    def _timed_out(
        self, now_s: float, limit_s: float, reason: str, *actions: str
    ) -> BcFlightDecision:
        if now_s - self._state_started_s > limit_s:
            return self._abort(now_s, reason)
        return self._decision(*actions)

    def step(
        self, now_s: float, evidence: BcFlightEvidence
    ) -> BcFlightDecision:
        """Advance only when the matching live evidence is present."""
        now = float(now_s)
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("flight clock must be finite and nonnegative")
        if self._last_step_s is None:
            self._state_started_s = now
        elif now < self._last_step_s:
            return self._abort(now, "flight clock moved backward")
        self._last_step_s = now
        if self.state in {BcFlightState.COMPLETE, BcFlightState.FAILED}:
            return self._decision()
        if evidence.failsafe and self.state not in {
            BcFlightState.HOLDING,
            BcFlightState.LANDING,
        }:
            return self._abort(now, "PX4 failsafe became active")
        if self.state == BcFlightState.WAITING_INPUTS:
            if (
                evidence.runtime_ready
                and evidence.observations_ready
                and evidence.telemetry_fresh
            ):
                self._set_state(BcFlightState.SELECTING_LIFECYCLE, now)
                return self._decision("SELECT_LIFECYCLE")
            return self._timed_out(
                now,
                self.config.readiness_timeout_s,
                "runtime inputs did not become ready",
            )
        if self.state == BcFlightState.SELECTING_LIFECYCLE:
            if evidence.lifecycle_selected and evidence.source_valid:
                self._set_state(BcFlightState.ENABLING_OUTPUT, now)
                return self._decision("ENABLE_OUTPUT")
            return self._timed_out(
                now,
                self.config.service_timeout_s,
                "lifecycle command selection timed out",
                "SELECT_LIFECYCLE",
            )
        if self.state == BcFlightState.ENABLING_OUTPUT:
            if evidence.output_safe:
                self._set_state(BcFlightState.ENABLING_STREAM, now)
                return self._decision("ENABLE_STREAM")
            actions = ("ENABLE_OUTPUT",) if evidence.output_ready else ()
            return self._timed_out(
                now,
                self.config.service_timeout_s,
                "PX4 output gate did not become safe",
                *actions,
            )
        if self.state == BcFlightState.ENABLING_STREAM:
            if evidence.stream_stable:
                self._set_state(BcFlightState.REQUESTING_OFFBOARD, now)
                return self._decision("SEND_OFFBOARD")
            return self._timed_out(
                now,
                self.config.service_timeout_s,
                "PX4 setpoint stream did not stabilize",
                "ENABLE_STREAM",
            )
        if self.state == BcFlightState.REQUESTING_OFFBOARD:
            if evidence.offboard_active:
                self._set_state(BcFlightState.REQUESTING_ARM, now)
                return self._decision("SEND_ARM")
            return self._timed_out(
                now,
                self.config.service_timeout_s,
                "PX4 did not enter OFFBOARD",
                "SEND_OFFBOARD",
            )
        if self.state == BcFlightState.REQUESTING_ARM:
            if evidence.vehicle_armed:
                self._set_state(BcFlightState.TAKING_OFF, now)
                return self._decision()
            return self._timed_out(
                now,
                self.config.service_timeout_s,
                "PX4 did not arm",
                "SEND_ARM",
            )
        if self.state == BcFlightState.TAKING_OFF:
            if not evidence.telemetry_fresh or not evidence.source_valid:
                return self._abort(
                    now, "takeoff control evidence became stale"
                )
            minimum = (
                self.config.flight_altitude_m
                - self.config.altitude_tolerance_m
            )
            if evidence.altitude_m >= minimum:
                self._set_state(BcFlightState.ENABLING_BC, now)
                return self._decision("ENABLE_BC")
            return self._timed_out(
                now,
                self.config.takeoff_timeout_s,
                "takeoff altitude was not reached",
            )
        if self.state == BcFlightState.ENABLING_BC:
            if evidence.bc_enabled and evidence.bc_ready:
                self._set_state(BcFlightState.SELECTING_BC, now)
                return self._decision("SELECT_BC")
            return self._timed_out(
                now,
                self.config.service_timeout_s,
                "BC policy did not become ready",
                "ENABLE_BC",
            )
        if self.state == BcFlightState.SELECTING_BC:
            if evidence.bc_selected and evidence.source_valid:
                self._set_state(BcFlightState.NAVIGATING, now)
                return self._decision()
            return self._timed_out(
                now,
                self.config.service_timeout_s,
                "BC control handoff timed out",
                "SELECT_BC",
            )
        if self.state == BcFlightState.NAVIGATING:
            if evidence.terminal_reason:
                self.terminal_reason = evidence.terminal_reason
                self._set_state(BcFlightState.HOLDING, now)
                return self._decision("SELECT_HOLD", "DISABLE_BC")
            if (
                not evidence.runtime_ready
                or not evidence.telemetry_fresh
                or not evidence.bc_ready
                or not evidence.bc_selected
                or not evidence.source_valid
                or not evidence.output_safe
                or not evidence.stream_stable
            ):
                return self._abort(
                    now, "BC flight evidence became stale or invalid"
                )
            return self._decision()
        if self.state == BcFlightState.HOLDING:
            if not evidence.bc_selected:
                self._set_state(BcFlightState.LANDING, now)
                return self._decision(
                    "DISABLE_BC",
                    "DISABLE_STREAM",
                    "DISABLE_OUTPUT",
                    "SEND_LAND",
                )
            return self._decision("SELECT_HOLD")
        if self.state == BcFlightState.LANDING:
            if evidence.landed and not evidence.vehicle_armed:
                final_state = (
                    BcFlightState.FAILED
                    if self.failure_reason
                    else BcFlightState.COMPLETE
                )
                self._set_state(final_state, now)
                return self._decision()
            if now - self._state_started_s > self.config.landing_timeout_s:
                self.failure_reason = (
                    self.failure_reason or "landing timed out"
                )
                self._set_state(BcFlightState.FAILED, now)
                return self._decision()
            return self._decision("SEND_LAND")
        raise RuntimeError(f"unhandled BC flight state: {self.state.value}")
