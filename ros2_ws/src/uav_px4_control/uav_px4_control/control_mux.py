"""Pure deterministic Phase 6 control-source multiplexer."""

import math

from uav_px4_control.control_source_models import (
    ACTIVE_STATES,
    ASTAR_EXPERT,
    CONTROL_SOURCES,
    CandidateRecord,
    ControlCommand,
    ControlMuxConfig,
    ControlMuxResult,
    ControlMuxState,
    HOLD,
    MOVEMENT_SOURCES,
    MuxSaturationFlags,
    SelectionResponse,
    Vector3,
    zero_hold,
)
from uav_px4_control.control_source_registry import ControlSourceRegistry
from uav_px4_control.selected_command_validator import (
    validate_selected_command,
)


def _norm(vector: Vector3) -> float:
    return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)


def _scale(vector: Vector3, limit: float) -> tuple[Vector3, bool]:
    magnitude = _norm(vector)
    if magnitude <= limit or magnitude == 0.0:
        return vector, False
    ratio = limit / magnitude
    return Vector3(vector.x * ratio, vector.y * ratio, vector.z * ratio), True


class ControlSourceMux:
    """Fail-closed source selection, handoff, limiting, and validation."""

    def __init__(self, config: ControlMuxConfig | None = None) -> None:
        """Initialize exact startup HOLD and empty source histories."""
        self.config = config or ControlMuxConfig()
        self.registry = ControlSourceRegistry(self.config)
        self.requested_source = self.config.default_source
        self.active_source = HOLD
        self.state = ControlMuxState.HOLD_STARTUP
        self.hold_reason = "mux startup has no selected movement source"
        self.pending_source: str | None = None
        self.switch_until_s: float | None = None
        self.transition_count = 0
        self.last_transition_s: float | None = None
        self.last_now_s: float | None = None
        self.previous_selected: ControlCommand | None = None
        self.cycle_index = 0
        self.fault_latched = False
        self.latched_reason = ""
        self._fault_state_reported = False

    def accept_candidate(
        self,
        source: str,
        command: ControlCommand,
        receipt_time_s: float,
    ) -> CandidateRecord:
        """Pass one untrusted source message to the independent registry."""
        return self.registry.update(source, command, receipt_time_s)

    def _set_active(self, source: str, now_s: float) -> None:
        if source != self.active_source:
            self.active_source = source
            self.transition_count += 1
            self.last_transition_s = now_s

    def _rate_anchor(self, now_s: float, reason: str) -> None:
        period = 1.0 / self.config.publish_rate_hz
        self.previous_selected = zero_hold(now_s - period, reason)

    def _enter_hold(
        self,
        now_s: float,
        state: ControlMuxState,
        reason: str,
        latch: bool,
    ) -> None:
        self._set_active(HOLD, now_s)
        self.pending_source = None
        self.switch_until_s = None
        self.state = state
        self.hold_reason = reason
        if latch and self.config.latch_hold_after_fault:
            self.fault_latched = True
            self.latched_reason = reason
            self._fault_state_reported = False

    def _handle_backward_time(self, now_s: float) -> None:
        self.registry.clear()
        self.previous_selected = None
        self.last_transition_s = None
        self.requested_source = HOLD
        self._enter_hold(
            now_s,
            ControlMuxState.HOLD_TIME_JUMP,
            "ROS time moved backward; fresh data and explicit request "
            "required",
            True,
        )
        self.last_now_s = now_s

    def request_source(
        self, source: str, current_time_s: float
    ) -> SelectionResponse:
        """Apply deterministic service selection and fail-closed rejection."""
        now = float(current_time_s)
        if not math.isfinite(now):
            raise ValueError("selection request time must be finite")
        if self.last_now_s is not None and now < self.last_now_s:
            self._handle_backward_time(now)
            return SelectionResponse(
                False, source, HOLD,
                "request rejected after backward time jump",
            )
        self.last_now_s = now
        self.requested_source = source
        if source == HOLD:
            self._enter_hold(
                now, ControlMuxState.HOLD_REQUESTED,
                "explicit HOLD requested", False,
            )
            self.fault_latched = False
            self.latched_reason = ""
            self._rate_anchor(now, self.hold_reason)
            return SelectionResponse(
                True, source, self.active_source, "HOLD accepted immediately"
            )
        if source not in MOVEMENT_SOURCES:
            self._enter_hold(
                now, ControlMuxState.HOLD_INVALID_SOURCE,
                f"unknown control source requested: {source}", True,
            )
            return SelectionResponse(
                False, source, self.active_source,
                "unknown source rejected; mux entered fail-closed HOLD",
            )
        if source == self.active_source and self.pending_source is None:
            return SelectionResponse(
                True, source, self.active_source,
                "duplicate active-source request accepted idempotently",
            )
        health = self.registry.health(source, now)
        if (
            self.config.require_fresh_command_before_switch
            and not health.healthy
        ):
            if not health.received:
                state = ControlMuxState.HOLD_WAITING_SOURCE
            elif "frame" in health.reason:
                state = ControlMuxState.HOLD_WRONG_FRAME
            else:
                state = ControlMuxState.HOLD_INVALID_COMMAND
            if self.active_source == HOLD:
                self.state = state
                self.hold_reason = (
                    f"source request rejected: {source}: {health.reason}"
                )
            return SelectionResponse(
                False, source, self.active_source,
                f"source is not healthy: {health.reason}",
            )
        if (
            self.last_transition_s is not None
            and now - self.last_transition_s
            < self.config.minimum_source_dwell_time_s
        ):
            return SelectionResponse(
                False, source, self.active_source,
                "minimum source dwell time has not elapsed",
            )
        self.fault_latched = False
        self.latched_reason = ""
        if self.active_source in MOVEMENT_SOURCES:
            self.pending_source = source
            self.switch_until_s = now + self.config.switch_hold_duration_s
            self._set_active(HOLD, now)
            self.state = ControlMuxState.HOLD_SWITCH_BARRIER
            self.hold_reason = (
                f"switch barrier before activating {source}"
            )
            self._rate_anchor(now, self.hold_reason)
            return SelectionResponse(
                True, source, self.active_source,
                "movement-source request accepted; switch HOLD in progress",
            )
        self.pending_source = None
        self.switch_until_s = None
        self._set_active(source, now)
        self.state = ACTIVE_STATES[source]
        self.hold_reason = ""
        self._rate_anchor(now, "safe source activation anchor")
        return SelectionResponse(
            True, source, self.active_source,
            "fresh movement source activated",
        )

    def _limit_candidate(
        self, command: ControlCommand, now_s: float
    ) -> tuple[ControlCommand, MuxSaturationFlags]:
        linear = command.linear
        horizontal_magnitude = math.hypot(linear.x, linear.y)
        horizontal = False
        if (
            horizontal_magnitude
            > self.config.maximum_selected_horizontal_speed_mps
        ):
            ratio = (
                self.config.maximum_selected_horizontal_speed_mps
                / horizontal_magnitude
            )
            linear = Vector3(linear.x * ratio, linear.y * ratio, linear.z)
            horizontal = True
        vertical = (
            abs(linear.z)
            > self.config.maximum_selected_vertical_speed_mps
        )
        if vertical:
            linear = Vector3(
                linear.x,
                linear.y,
                math.copysign(
                    self.config.maximum_selected_vertical_speed_mps,
                    linear.z,
                ),
            )
        linear, total = _scale(
            linear, self.config.maximum_selected_speed_mps
        )
        acceleration = False
        yaw_acceleration = False
        yaw_rate = command.yaw_rate_radps
        yaw_limited = (
            abs(yaw_rate) > self.config.maximum_selected_yaw_rate_radps
        )
        if yaw_limited:
            yaw_rate = math.copysign(
                self.config.maximum_selected_yaw_rate_radps, yaw_rate
            )
        if self.previous_selected is not None:
            elapsed = now_s - self.previous_selected.timestamp_s
            if elapsed > 0.0:
                difference = Vector3(
                    linear.x - self.previous_selected.linear.x,
                    linear.y - self.previous_selected.linear.y,
                    linear.z - self.previous_selected.linear.z,
                )
                difference, acceleration = _scale(
                    difference,
                    self.config.maximum_selected_acceleration_mps2 * elapsed,
                )
                if acceleration:
                    linear = Vector3(
                        self.previous_selected.linear.x + difference.x,
                        self.previous_selected.linear.y + difference.y,
                        self.previous_selected.linear.z + difference.z,
                    )
                maximum_yaw_change = (
                    self.config.maximum_selected_yaw_acceleration_radps2
                    * elapsed
                )
                yaw_change = (
                    yaw_rate - self.previous_selected.yaw_rate_radps
                )
                yaw_acceleration = abs(yaw_change) > maximum_yaw_change
                if yaw_acceleration:
                    yaw_rate = (
                        self.previous_selected.yaw_rate_radps
                        + math.copysign(maximum_yaw_change, yaw_change)
                    )
        return ControlCommand(
            source=self.active_source,
            timestamp_s=now_s,
            frame_id=command.frame_id,
            linear=linear,
            angular_x=command.angular_x,
            angular_y=command.angular_y,
            yaw_rate_radps=yaw_rate,
        ), MuxSaturationFlags(
            horizontal_speed=horizontal,
            vertical_speed=vertical,
            total_speed=total,
            acceleration=acceleration,
            yaw_rate=yaw_limited,
            yaw_acceleration=yaw_acceleration,
        )

    def _hold_result(
        self,
        now_s: float,
        attempted_valid: bool = True,
        diagnostics=(),
    ) -> ControlMuxResult:
        reason = self.hold_reason or "fail-closed internal HOLD"
        command = zero_hold(now_s, reason)
        fallback_diagnostics = validate_selected_command(
            command,
            self.config,
            self.state,
            HOLD,
            self.cycle_index,
            None,
        )
        if fallback_diagnostics:
            raise RuntimeError(
                "internal HOLD failed validation: "
                f"{fallback_diagnostics[0].constraint}"
            )
        self.previous_selected = command
        healthy = self.registry.healthy_sources(now_s)
        stale = self.registry.stale_sources(now_s)
        remaining = 0.0
        if self.switch_until_s is not None:
            remaining = max(0.0, self.switch_until_s - now_s)
        state = self.state
        if self.fault_latched and state not in {
            ControlMuxState.HOLD_TIME_JUMP,
            ControlMuxState.HOLD_STALE_SOURCE,
            ControlMuxState.HOLD_INVALID_SOURCE,
            ControlMuxState.HOLD_INVALID_COMMAND,
            ControlMuxState.HOLD_WRONG_FRAME,
        }:
            state = ControlMuxState.HOLD_LATCHED_FAULT
        result = ControlMuxResult(
            state=state,
            requested_source=self.requested_source,
            active_source=HOLD,
            selected_command=command,
            selected_command_valid=attempted_valid,
            hold_active=True,
            hold_reason=reason,
            switch_in_progress=self.pending_source is not None,
            switch_remaining_time_s=remaining,
            selected_source_age_s=(
                self.registry.health(self.pending_source, now_s).age_s
                if self.pending_source is not None else math.inf
            ),
            transition_count=self.transition_count,
            healthy_sources=healthy,
            stale_sources=stale,
            diagnostics=tuple(diagnostics),
            fault_latched=self.fault_latched,
            status_message=f"{state.value}: {reason}",
        )
        if self.fault_latched:
            self._fault_state_reported = True
        return result

    @staticmethod
    def _fault_state(reason: str, fresh: bool) -> ControlMuxState:
        if not fresh or "stale" in reason:
            return ControlMuxState.HOLD_STALE_SOURCE
        if "frame" in reason:
            return ControlMuxState.HOLD_WRONG_FRAME
        return ControlMuxState.HOLD_INVALID_COMMAND

    def step(self, current_time_s: float) -> ControlMuxResult:
        """Run one freshness, arbitration, limiting, and validation cycle."""
        now = float(current_time_s)
        if not math.isfinite(now):
            raise ValueError("mux cycle time must be finite")
        self.cycle_index += 1
        if self.last_now_s is not None and now < self.last_now_s:
            self._handle_backward_time(now)
            return self._hold_result(now)
        if (
            self.previous_selected is not None
            and now <= self.previous_selected.timestamp_s
        ):
            self.previous_selected = None
            self._enter_hold(
                now, ControlMuxState.HOLD_TIME_JUMP,
                "ROS time did not advance; output history reset", True,
            )
            self.last_now_s = now
            return self._hold_result(now)
        self.last_now_s = now
        if self.fault_latched:
            self._set_active(HOLD, now)
            if self._fault_state_reported:
                self.state = ControlMuxState.HOLD_LATCHED_FAULT
            self.hold_reason = self.latched_reason or self.hold_reason
            return self._hold_result(now)
        if self.pending_source is not None:
            target = self.pending_source
            health = self.registry.health(target, now)
            if not health.healthy:
                state = self._fault_state(health.reason, health.fresh)
                self._enter_hold(
                    now, state,
                    f"handoff target {target} invalid: {health.reason}", True,
                )
                return self._hold_result(now)
            if self.switch_until_s is not None and now < self.switch_until_s:
                self.state = ControlMuxState.HOLD_SWITCH_BARRIER
                return self._hold_result(now)
            self.pending_source = None
            self.switch_until_s = None
            self._set_active(target, now)
            self.state = ACTIVE_STATES[target]
            self.hold_reason = ""
        if self.active_source == HOLD:
            return self._hold_result(now)
        health = self.registry.health(self.active_source, now)
        if not health.healthy:
            state = self._fault_state(health.reason, health.fresh)
            source = self.active_source
            self._enter_hold(
                now, state,
                f"active source {source} invalid: {health.reason}", True,
            )
            return self._hold_result(now)
        record = self.registry.record(self.active_source)
        if record.command is None:
            self._enter_hold(
                now, ControlMuxState.HOLD_INVALID_COMMAND,
                "healthy source has no candidate command", True,
            )
            return self._hold_result(now)
        command, saturations = self._limit_candidate(record.command, now)
        diagnostics = validate_selected_command(
            command,
            self.config,
            self.state,
            self.active_source,
            self.cycle_index,
            self.previous_selected,
        )
        if diagnostics:
            source = self.active_source
            self._enter_hold(
                now, ControlMuxState.HOLD_INVALID_COMMAND,
                "selected command validation failed for "
                f"{source}: {diagnostics[0].constraint}", True,
            )
            return self._hold_result(now, False, diagnostics)
        self.previous_selected = command
        return ControlMuxResult(
            state=self.state,
            requested_source=self.requested_source,
            active_source=self.active_source,
            selected_command=command,
            selected_command_valid=True,
            hold_active=False,
            hold_reason="",
            switch_in_progress=False,
            switch_remaining_time_s=0.0,
            selected_source_age_s=health.age_s,
            transition_count=self.transition_count,
            healthy_sources=self.registry.healthy_sources(now),
            stale_sources=self.registry.stale_sources(now),
            saturations=saturations,
            fault_latched=False,
            status_message=(
                f"{self.state.value}: selected {self.active_source}; "
                f"saturations={saturations.count}"
            ),
        )


def fixed_candidate(
    source: str = ASTAR_EXPERT,
    stamp_s: float = 1.0,
    north_mps: float = 0.5,
    east_mps: float = 0.0,
    down_mps: float = 0.0,
    yaw_rate_radps: float = 0.0,
) -> ControlCommand:
    """Construct a concise valid command for pure fixtures and examples."""
    if source not in CONTROL_SOURCES:
        raise ValueError(f"unknown control source: {source}")
    return ControlCommand(
        source=source,
        timestamp_s=stamp_s,
        frame_id="px4_ned",
        linear=Vector3(north_mps, east_mps, down_mps),
        yaw_rate_radps=yaw_rate_radps,
    )
