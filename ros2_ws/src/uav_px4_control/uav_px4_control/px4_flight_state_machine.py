"""Pure guarded OFFBOARD/arm/takeoff/track/land mission sequencing."""

import math

from uav_px4_control.px4_flight_models import (
    FlightDecision,
    FlightEvidence,
    Px4FlightConfig,
    Px4FlightState,
)


_AIRBORNE_STATES = frozenset({
    Px4FlightState.STARTING_TAKEOFF,
    Px4FlightState.TAKEOFF,
    Px4FlightState.REPLANNING,
    Px4FlightState.STARTING_TRACKING,
    Px4FlightState.TRACKING,
})


class Px4FlightStateMachine:
    """Sequence one explicitly enabled SITL mission and fail toward landing."""

    def __init__(self, config: Px4FlightConfig | None = None) -> None:
        """Create a disabled mission with no previous clock sample."""
        self.config = config or Px4FlightConfig()
        self.state = Px4FlightState.DISABLED
        self.mission_enable_requested = False
        self.failure_reason = ""
        self.transition_count = 0
        self._state_started_s = 0.0
        self._last_step_s: float | None = None

    def request_enable(self, enable: bool, now_s: float) -> tuple[bool, str]:
        """Start once from disabled or convert an active request to landing."""
        now = float(now_s)
        if not math.isfinite(now) or now < 0.0:
            return False, "request time must be finite and nonnegative"
        if not isinstance(enable, bool):
            return False, "enable must be bool"
        if enable:
            if self.state not in {
                Px4FlightState.DISABLED,
                Px4FlightState.COMPLETE,
                Px4FlightState.FAILED,
            }:
                return (
                    False,
                    f"mission is already active in {self.state.value}",
                )
            self.mission_enable_requested = True
            self.failure_reason = ""
            self._last_step_s = now
            self._set_state(Px4FlightState.WAITING_PIPELINE, now)
            return True, "SITL flight mission explicitly enabled"
        self.mission_enable_requested = False
        if self.state in _AIRBORNE_STATES or self.state in {
            Px4FlightState.REQUESTING_ARM,
            Px4FlightState.LANDING,
        }:
            self.failure_reason = "explicit abort requested"
            self._set_state(Px4FlightState.LANDING, now)
            return True, "abort accepted; controlled landing requested"
        self._set_state(Px4FlightState.DISABLED, now)
        return True, "mission disabled"

    def step(self, now_s: float, evidence: FlightEvidence) -> FlightDecision:
        """Advance using actual PX4 and ROS evidence, never topic presence."""
        now = float(now_s)
        if not math.isfinite(now) or now < 0.0:
            return self._fail(now, evidence, "mission clock is invalid")
        if self._last_step_s is not None and now < self._last_step_s:
            return self._fail(now, evidence, "mission clock moved backward")
        self._last_step_s = now
        if self.state in {
            Px4FlightState.DISABLED,
            Px4FlightState.COMPLETE,
            Px4FlightState.FAILED,
        }:
            return self._decision()

        if evidence.fatal_command_ack:
            return self._fail(
                now,
                evidence,
                f"PX4 command rejected: {evidence.fatal_command_ack}",
            )
        if evidence.failsafe:
            return self._fail(now, evidence, "PX4 failsafe became active")
        if (
            not evidence.environment_valid
            and self.state != Px4FlightState.LANDING
        ):
            return self._fail(
                now,
                evidence,
                "external simulator environment became stale or invalid",
            )
        if self.state in _AIRBORNE_STATES and evidence.vehicle_armed:
            if not evidence.telemetry_fresh:
                return self._fail(now, evidence, "PX4 telemetry became stale")
            if not evidence.source_valid or not evidence.astar_selected:
                return self._fail(
                    now, evidence, "ASTAR_EXPERT source became invalid"
                )
            if not evidence.output_gate_safe or not evidence.stream_stable:
                return self._fail(
                    now, evidence, "setpoint safety chain stopped"
                )

        elapsed = now - self._state_started_s
        if self.state == Px4FlightState.WAITING_PIPELINE:
            if evidence.pipeline_ready and evidence.follower_command_valid:
                self._set_state(Px4FlightState.SELECTING_ASTAR, now)
                return self._decision("SELECT_ASTAR")
            return self._timeout(
                elapsed,
                self.config.pipeline_timeout_s,
                now,
                evidence,
                "A*/B-spline/trajectory pipeline readiness timed out",
            )
        if self.state == Px4FlightState.SELECTING_ASTAR:
            if evidence.astar_selected and evidence.source_valid:
                self._set_state(Px4FlightState.ENABLING_OUTPUT_GATE, now)
                return self._decision()
            return self._timeout(
                elapsed,
                self.config.pipeline_timeout_s,
                now,
                evidence,
                "ASTAR_EXPERT selection timed out",
                "SELECT_ASTAR",
            )
        if self.state == Px4FlightState.ENABLING_OUTPUT_GATE:
            if evidence.output_gate_safe:
                self._set_state(Px4FlightState.ENABLING_STREAM, now)
                return self._decision("ENABLE_STREAM")
            actions = (
                ("ENABLE_OUTPUT_GATE",)
                if evidence.output_gate_ready
                else ()
            )
            return self._timeout(
                elapsed,
                self.config.prestream_timeout_s,
                now,
                evidence,
                "PX4 output gate did not become safe",
                *actions,
            )
        if self.state in {
            Px4FlightState.ENABLING_STREAM,
            Px4FlightState.PRESTREAMING,
        }:
            if evidence.stream_stable and (
                evidence.stream_rate_hz >= self.config.minimum_stream_rate_hz
            ):
                self._set_state(Px4FlightState.REQUESTING_OFFBOARD, now)
                return self._decision("SEND_OFFBOARD")
            if self.state == Px4FlightState.ENABLING_STREAM:
                self._set_state(Px4FlightState.PRESTREAMING, now)
            return self._timeout(
                now - self._state_started_s,
                self.config.prestream_timeout_s,
                now,
                evidence,
                "20 Hz setpoint prestream did not stabilize",
                "ENABLE_STREAM",
            )
        if self.state == Px4FlightState.REQUESTING_OFFBOARD:
            if evidence.offboard_active:
                self._set_state(Px4FlightState.REQUESTING_ARM, now)
                return self._decision("SEND_ARM")
            return self._timeout(
                elapsed,
                self.config.offboard_timeout_s,
                now,
                evidence,
                "PX4 did not enter OFFBOARD",
                "SEND_OFFBOARD",
            )
        if self.state == Px4FlightState.REQUESTING_ARM:
            if evidence.vehicle_armed:
                self._set_state(Px4FlightState.STARTING_TAKEOFF, now)
                return self._decision("START_TRACKING")
            return self._timeout(
                elapsed,
                self.config.arm_timeout_s,
                now,
                evidence,
                "PX4 did not arm",
                "SEND_ARM",
            )
        if self.state == Px4FlightState.STARTING_TAKEOFF:
            if evidence.tracking_active:
                self._set_state(Px4FlightState.TAKEOFF, now)
                return self._decision()
            return self._timeout(
                elapsed,
                self.config.pipeline_timeout_s,
                now,
                evidence,
                "trajectory follower did not start takeoff",
                "START_TRACKING",
            )
        if self.state == Px4FlightState.TAKEOFF:
            minimum_altitude = (
                self.config.takeoff_altitude_m
                - self.config.takeoff_altitude_tolerance_m
            )
            if evidence.altitude_m >= minimum_altitude:
                self._set_state(Px4FlightState.REPLANNING, now)
                return self._decision("STOP_TRACKING", "PUBLISH_MISSION_SCENE")
            return self._timeout(
                elapsed,
                self.config.takeoff_timeout_s,
                now,
                evidence,
                "takeoff altitude was not reached",
            )
        if self.state == Px4FlightState.REPLANNING:
            if evidence.mission_trajectory_ready:
                self._set_state(Px4FlightState.STARTING_TRACKING, now)
                return self._decision("START_TRACKING")
            return self._timeout(
                elapsed,
                self.config.replan_timeout_s,
                now,
                evidence,
                "mission A*/B-spline replanning timed out",
                "PUBLISH_MISSION_SCENE",
            )
        if self.state == Px4FlightState.STARTING_TRACKING:
            if evidence.tracking_active:
                self._set_state(Px4FlightState.TRACKING, now)
                return self._decision()
            return self._timeout(
                elapsed,
                self.config.pipeline_timeout_s,
                now,
                evidence,
                "trajectory follower did not start mission tracking",
                "START_TRACKING",
            )
        if self.state == Px4FlightState.TRACKING:
            if evidence.goal_reached and (
                evidence.goal_distance_m <= self.config.goal_tolerance_m
            ):
                self._set_state(Px4FlightState.GOAL_HOLD, now)
                return self._decision(
                    "STOP_TRACKING",
                    "DISABLE_STREAM",
                    "DISABLE_OUTPUT_GATE",
                    "SEND_LAND",
                )
            return self._timeout(
                elapsed,
                self.config.tracking_timeout_s,
                now,
                evidence,
                "goal was not reached within the mission timeout",
            )
        if self.state == Px4FlightState.GOAL_HOLD:
            self._set_state(Px4FlightState.LANDING, now)
            return self._decision("SEND_LAND")
        if self.state == Px4FlightState.LANDING:
            if evidence.landed and not evidence.vehicle_armed:
                self.mission_enable_requested = False
                terminal = (
                    Px4FlightState.FAILED
                    if self.failure_reason
                    else Px4FlightState.COMPLETE
                )
                self._set_state(terminal, now)
                return self._decision(
                    "STOP_TRACKING",
                    "DISABLE_STREAM",
                    "DISABLE_OUTPUT_GATE",
                    "SELECT_HOLD",
                )
            if elapsed > self.config.landing_timeout_s:
                self.failure_reason = (
                    self.failure_reason or "landing timed out"
                )
                self.mission_enable_requested = False
                self._set_state(Px4FlightState.FAILED, now)
                return self._decision("SEND_LAND")
            return self._decision("SEND_LAND")
        return self._fail(now, evidence, f"unhandled state {self.state.value}")

    def _timeout(
        self,
        elapsed: float,
        limit: float,
        now: float,
        evidence: FlightEvidence,
        reason: str,
        *actions: str,
    ) -> FlightDecision:
        if elapsed > limit:
            return self._fail(now, evidence, reason)
        return self._decision(*actions)

    def _fail(
        self,
        now: float,
        evidence: FlightEvidence,
        reason: str,
    ) -> FlightDecision:
        self.failure_reason = reason
        if evidence.vehicle_armed or not evidence.landed:
            self._set_state(Px4FlightState.LANDING, now)
            return self._decision(
                "STOP_TRACKING",
                "DISABLE_STREAM",
                "DISABLE_OUTPUT_GATE",
                "SEND_LAND",
            )
        self.mission_enable_requested = False
        self._set_state(Px4FlightState.FAILED, now)
        return self._decision(
            "DISABLE_STREAM", "DISABLE_OUTPUT_GATE", "SELECT_HOLD"
        )

    def _set_state(self, state: Px4FlightState, now: float) -> None:
        if state != self.state:
            self.transition_count += 1
            self._state_started_s = now
        self.state = state

    def _decision(self, *actions: str) -> FlightDecision:
        return FlightDecision(
            state=self.state,
            actions=tuple(actions),
            failure_reason=self.failure_reason,
            transition_count=self.transition_count,
        )
