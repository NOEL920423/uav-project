"""ROS-independent lifecycle contracts for one live BC flight."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from enum import Enum


class BcFlightState(str, Enum):
    """Observable lifecycle states from startup through landing."""

    WAITING_INPUTS = "WAITING_INPUTS"
    SELECTING_LIFECYCLE = "SELECTING_LIFECYCLE"
    ENABLING_OUTPUT = "ENABLING_OUTPUT"
    ENABLING_STREAM = "ENABLING_STREAM"
    RECOVERING_STREAM = "RECOVERING_STREAM"
    RECOVERING_OUTPUT = "RECOVERING_OUTPUT"
    RECOVERING_READINESS = "RECOVERING_READINESS"
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


STARTUP_STATES = frozenset({
    BcFlightState.WAITING_INPUTS, BcFlightState.SELECTING_LIFECYCLE,
    BcFlightState.ENABLING_OUTPUT, BcFlightState.ENABLING_STREAM,
    BcFlightState.REQUESTING_OFFBOARD, BcFlightState.REQUESTING_ARM,
})
RECOVERY_STATES = frozenset({
    BcFlightState.RECOVERING_STREAM, BcFlightState.RECOVERING_OUTPUT,
    BcFlightState.RECOVERING_READINESS,
})


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
class StartupStale:
    """Preserve the first observed stale measurement before latch updates."""

    source: str
    age_s: float
    threshold_s: float
    reason: str


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
    landed: bool | None = None
    telemetry_fresh: bool = False
    failsafe: bool = False
    altitude_m: float = 0.0
    terminal_reason: str = ""
    startup_stale: StartupStale | None = None
    recovery_vehicle_state_fresh: bool = False
    stream_reset_complete: bool = False
    output_reset_complete: bool = False


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
        self.startup_deadline_s: float | None = None
        self.first_stale: StartupStale | None = None
        self.recovery_history: list[dict] = []
        self._flight_authority_seen = False
        self._preserve_startup_latches = False

    @property
    def recovery_diagnostics(self) -> dict:
        """Return diagnostics retained across retries and terminal failure."""
        return {
            "first_stale": None if self.first_stale is None else asdict(self.first_stale),
            "recovery_history": self.recovery_history,
            "startup_deadline_s": self.startup_deadline_s,
        }

    @property
    def state_started_s(self) -> float:
        """Expose the current phase boundary for asynchronous reset evidence."""
        return self._state_started_s

    def observe_flight_authority(self, armed: bool, offboard: bool) -> None:
        """Close recovery even if a PX4 transition occurs between timer ticks."""
        self._flight_authority_seen |= armed or offboard

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
        if (
            self.first_stale is not None or self._flight_authority_seen
        ) and self.state in STARTUP_STATES | RECOVERY_STATES:
            self._preserve_startup_latches = True
        if not self.failure_reason:
            if self.first_stale is not None:
                reason += "; startup_recovery=" + json.dumps(self.recovery_diagnostics)
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

    def _require_lifecycle_source(
        self, now: float, evidence: BcFlightEvidence
    ) -> BcFlightDecision | None:
        """Return to lifecycle selection if pre-arm control authority is lost."""
        if evidence.vehicle_armed:
            return None
        if evidence.lifecycle_selected and evidence.source_valid:
            return None
        if self._flight_authority_seen or evidence.offboard_active or not evidence.landed:
            return self._abort(now, "lifecycle authority lost after startup recovery boundary")
        self._set_state(BcFlightState.SELECTING_LIFECYCLE, now)
        return self._decision(
            "SELECT_HOLD", "SELECT_LIFECYCLE"
        )

    def step(
        self, now_s: float, evidence: BcFlightEvidence
    ) -> BcFlightDecision:
        """Advance only when the matching live evidence is present."""
        now = float(now_s)
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("flight clock must be finite and nonnegative")
        if self._last_step_s is None:
            self._state_started_s = now
            # One budget for readiness plus the five pre-arm service stages.
            # State changes and recovery attempts never extend this deadline.
            self.startup_deadline_s = (
                now + self.config.readiness_timeout_s
                + 5 * self.config.service_timeout_s
            )
        elif now < self._last_step_s:
            return self._abort(now, "flight clock moved backward")
        self._last_step_s = now
        self.observe_flight_authority(evidence.vehicle_armed, evidence.offboard_active)
        if self.state in {BcFlightState.COMPLETE, BcFlightState.FAILED}:
            return self._decision()
        if evidence.failsafe and self.state not in {
            BcFlightState.HOLDING,
            BcFlightState.LANDING,
        }:
            return self._abort(now, "PX4 failsafe became active")
        if self.state in STARTUP_STATES | RECOVERY_STATES:
            if evidence.startup_stale is not None and self.first_stale is None:
                self.first_stale = evidence.startup_stale
            if now > self.startup_deadline_s:
                reason = "overall startup deadline exceeded"
                if not self._flight_authority_seen and (
                    evidence.landed is None or not evidence.recovery_vehicle_state_fresh
                ):
                    reason += ": fresh disarmed and landed confirmation unavailable"
                return self._abort(now, reason)
        if (
            self.state in STARTUP_STATES | RECOVERY_STATES
            and not self._flight_authority_seen
        ):
            if evidence.landed is False:
                return self._abort(now, "startup prohibited: vehicle confirmed airborne")
            if evidence.landed is None or not evidence.recovery_vehicle_state_fresh:
                # Unknown ground state never authorizes enable, mode, or reset.
                # Keep the original deadline and retry budget while awaiting it.
                if self.state == BcFlightState.WAITING_INPUTS:
                    return self._timed_out(
                        now, self.config.readiness_timeout_s,
                        "fresh disarmed and landed confirmation did not arrive",
                    )
                return self._decision("SELECT_HOLD")
        if self.state in RECOVERY_STATES:
            if self._flight_authority_seen or evidence.landed is False:
                return self._abort(now, "startup stale recovery no longer permitted")
            # Unknown or stale vehicle state is not permission to reset.
            if not evidence.recovery_vehicle_state_fresh:
                return self._decision()
            if self.state == BcFlightState.RECOVERING_STREAM:
                if not evidence.stream_reset_complete:
                    return self._decision("DISABLE_STREAM")
                self._set_state(BcFlightState.RECOVERING_OUTPUT, now)
                return self._decision("DISABLE_OUTPUT")
            if self.state == BcFlightState.RECOVERING_OUTPUT:
                if not evidence.output_reset_complete:
                    return self._decision("DISABLE_OUTPUT")
                self._set_state(BcFlightState.RECOVERING_READINESS, now)
                return self._decision("SELECT_LIFECYCLE")
            if (
                evidence.runtime_ready and evidence.observations_ready
                and evidence.telemetry_fresh and evidence.lifecycle_selected
                and evidence.source_valid and evidence.output_ready
                and evidence.startup_stale is None
            ):
                self.recovery_history[-1]["ready_time_s"] = now
                self._set_state(BcFlightState.ENABLING_OUTPUT, now)
                return self._decision("ENABLE_OUTPUT")
            return self._decision("SELECT_LIFECYCLE")
        if self.state in STARTUP_STATES and evidence.startup_stale is not None:
            if self._flight_authority_seen:
                return self._abort(now, "startup stale recovery prohibited after flight authority")
            if len(self.recovery_history) >= 2:
                return self._abort(now, "startup stale recovery limit exceeded (2)")
            self.recovery_history.append({
                "attempt": len(self.recovery_history) + 1,
                "time_s": now, "stale": asdict(evidence.startup_stale),
            })
            self._set_state(BcFlightState.RECOVERING_STREAM, now)
            actions = ("DISABLE_STREAM",) if evidence.recovery_vehicle_state_fresh else ()
            return self._decision(*actions)
        if self.state == BcFlightState.WAITING_INPUTS:
            if self._flight_authority_seen:
                return self._abort(now, "flight authority observed before startup readiness")
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
                # The gate receives telemetry independently of this supervisor.
                # Enabling before its readiness check would latch a startup fault.
                actions = ("ENABLE_OUTPUT",) if evidence.output_ready else ()
                return self._decision(*actions)
            return self._timed_out(
                now,
                self.config.service_timeout_s,
                "lifecycle command selection timed out",
                "SELECT_LIFECYCLE",
            )
        if self.state == BcFlightState.ENABLING_OUTPUT:
            guard = self._require_lifecycle_source(now, evidence)
            if guard is not None:
                return guard
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
            guard = self._require_lifecycle_source(now, evidence)
            if guard is not None:
                return guard
            if evidence.stream_stable:
                # A mode request may still be in flight while status is inactive.
                # Never reset after committing to OFFBOARD, even before its ACK.
                self._flight_authority_seen = True
                self._set_state(BcFlightState.REQUESTING_OFFBOARD, now)
                return self._decision("SEND_OFFBOARD")
            return self._timed_out(
                now,
                self.config.service_timeout_s,
                "PX4 setpoint stream did not stabilize",
                "ENABLE_STREAM",
            )
        if self.state == BcFlightState.REQUESTING_OFFBOARD:
            guard = self._require_lifecycle_source(now, evidence)
            if guard is not None:
                return guard
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
            guard = self._require_lifecycle_source(now, evidence)
            if guard is not None:
                return guard
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
            guard = self._require_lifecycle_source(now, evidence)
            if guard is not None:
                return guard
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
            if (
                evidence.bc_selected
                and evidence.source_valid
                and evidence.output_safe
                and evidence.stream_stable
            ):
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
                if self._preserve_startup_latches:
                    # Cleanup must not bypass the recovery boundary or limit.
                    return self._decision("DISABLE_BC", "SEND_LAND")
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
