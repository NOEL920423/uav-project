"""Pure fail-closed gate for the diagnostic PX4 output boundary."""

import math

from uav_px4_control.px4_boundary_models import (
    CandidateValidation,
    GateEnableResponse,
    MuxHealthEvidence,
    Px4MappingConfig,
    Px4OutputGateResult,
    Px4OutputGateState,
    Px4TelemetryState,
    Px4VelocitySetpointCandidate,
    source_is_known,
)


_ACCEPTABLE_ARMING_STATES = frozenset((1, 2))
_NAVIGATION_STATE_MAX = 22
_NAVIGATION_STATE_TERMINATION = 13


class Px4OutputSafetyGate:
    """Require independent, fresh evidence before granting permission."""

    def __init__(self, config: Px4MappingConfig | None = None) -> None:
        """Create a startup-disabled gate with no accepted evidence."""
        self.config = config or Px4MappingConfig()
        self.enabled = False
        self.fault_latched = False
        self.transition_count = 0
        self.state = Px4OutputGateState.OUTPUT_DISABLED
        self.hold_reason = "output has not been explicitly enabled"
        self._last_step_time_s: float | None = None
        self._latest_candidate_timestamp_us: int | None = None
        self._latest_telemetry_timestamp_us: int | None = None
        self._safe_vehicle_signature: tuple[int, int, bool] | None = None
        self._safe_active_source: str | None = None
        self._enable_pending = False

    def request_enable(self, enable: bool) -> GateEnableResponse:
        """Set explicit permission intent; disable also clears the latch."""
        if not isinstance(enable, bool):
            return GateEnableResponse(
                False, self.enabled, "enable must be bool"
            )
        if not enable:
            self.enabled = False
            self.fault_latched = False
            self._enable_pending = False
            self._safe_vehicle_signature = None
            self._safe_active_source = None
            self._latest_candidate_timestamp_us = None
            self._latest_telemetry_timestamp_us = None
            self._set_state(
                Px4OutputGateState.OUTPUT_DISABLED,
                "explicit disable/reset request",
            )
            return GateEnableResponse(True, False, "output disabled and reset")
        if self.fault_latched:
            return GateEnableResponse(
                False,
                False,
                "fault is latched; request disable/reset before enable",
            )
        if self.enabled:
            return GateEnableResponse(True, True, "output already enabled")
        self.enabled = True
        self._enable_pending = True
        return GateEnableResponse(True, True, "enable request accepted")

    def step(
        self,
        current_time_s: float,
        candidate: Px4VelocitySetpointCandidate | None,
        candidate_validation: CandidateValidation | None,
        mux: MuxHealthEvidence | None,
        telemetry: Px4TelemetryState | None,
    ) -> Px4OutputGateResult:
        """Evaluate all evidence without publishing or forwarding a command."""
        now = float(current_time_s)
        candidate_age = self._age(
            now,
            None if candidate is None else candidate.selected_receipt_time_s,
        )
        telemetry_age = self._age(
            now,
            None if telemetry is None else telemetry.receipt_time_s,
        )
        time_jump = (
            not math.isfinite(now)
            or (
                self._last_step_time_s is not None
                and now < self._last_step_time_s
            )
        )
        self._last_step_time_s = now if math.isfinite(now) else None

        state, reason = self._classify(
            time_jump,
            candidate,
            candidate_validation,
            candidate_age,
            mux,
            telemetry,
            telemetry_age,
        )
        healthy = state == Px4OutputGateState.SAFE_TO_FORWARD
        if self.fault_latched:
            state = Px4OutputGateState.LATCHED_FAULT
            reason = "fault latched; explicit disable/reset required"
            healthy = False
        elif not healthy:
            if self.enabled and self.config.latch_faults:
                self.fault_latched = True
                self.enabled = False
                self._enable_pending = False
        elif not self.enabled:
            state = Px4OutputGateState.READY_DISABLED
            reason = "all evidence healthy; explicit enable required"
            healthy = False
        elif self._enable_pending:
            self._enable_pending = False
            state = Px4OutputGateState.ENABLE_PENDING
            reason = "enable accepted; one complete healthy cycle required"
            healthy = False
        else:
            signature = self._vehicle_signature(telemetry)
            if self._safe_vehicle_signature is None:
                self._safe_vehicle_signature = signature
                self._safe_active_source = mux.active_source if mux else None
            elif (
                self.config.lock_vehicle_state_signature
                and signature != self._safe_vehicle_signature
            ):
                state = Px4OutputGateState.DISABLED_STATE_CHANGE
                reason = (
                    "vehicle arming, navigation, or offboard state changed"
                )
                self.fault_latched = self.config.latch_faults
                self.enabled = False
                healthy = False
            elif (
                mux is not None
                and mux.active_source != self._safe_active_source
            ):
                if self.config.allow_controlled_mux_handoff:
                    self._safe_active_source = mux.active_source
                else:
                    state = Px4OutputGateState.DISABLED_STATE_CHANGE
                    reason = (
                        "active control source changed while gate was enabled"
                    )
                    self.fault_latched = self.config.latch_faults
                    self.enabled = False
                    healthy = False

        self._set_state(state, reason)
        return Px4OutputGateResult(
            enabled=self.enabled,
            safe_to_forward=healthy,
            state=self.state,
            hold_reason=self.hold_reason,
            selected_command_valid=(
                candidate_validation.valid
                if candidate_validation is not None
                else False
            ),
            mux_valid=self._mux_valid(mux, now),
            telemetry_valid=self._telemetry_valid(telemetry, telemetry_age),
            failsafe=False if telemetry is None else telemetry.failsafe,
            active_source="" if mux is None else mux.active_source,
            selected_command_age_s=candidate_age,
            telemetry_age_s=telemetry_age,
            transition_count=self.transition_count,
            fault_latched=self.fault_latched,
        )

    def _classify(
        self,
        time_jump: bool,
        candidate: Px4VelocitySetpointCandidate | None,
        validation: CandidateValidation | None,
        candidate_age: float,
        mux: MuxHealthEvidence | None,
        telemetry: Px4TelemetryState | None,
        telemetry_age: float,
    ) -> tuple[Px4OutputGateState, str]:
        if time_jump:
            return (
                Px4OutputGateState.DISABLED_TIME_JUMP,
                "clock moved backward",
            )
        if candidate is None or validation is None:
            return (
                Px4OutputGateState.WAITING_SELECTED_COMMAND,
                "waiting for selected command candidate",
            )
        if candidate_age > self.config.selected_command_timeout_s:
            return (
                Px4OutputGateState.DISABLED_STALE_COMMAND,
                "selected command is stale",
            )
        if not validation.valid:
            state = Px4OutputGateState.DISABLED_INVALID_COMMAND
            if "stale" in validation.reason:
                state = Px4OutputGateState.DISABLED_STALE_COMMAND
            return state, validation.reason
        if not self._candidate_timestamp_valid(candidate.timestamp_us):
            return (
                Px4OutputGateState.DISABLED_INVALID_COMMAND,
                "candidate timestamp moved backward",
            )
        if mux is None or not mux.received:
            return (
                Px4OutputGateState.WAITING_MUX_HEALTH,
                "waiting for mux status",
            )
        if (
            not source_is_known(mux.active_source)
            or candidate.source != mux.active_source
        ):
            return (
                Px4OutputGateState.DISABLED_MUX_HOLD,
                "mux source is unknown or disagrees with candidate",
            )
        if not self._mux_valid(mux, self._last_step_time_s):
            return (
                Px4OutputGateState.DISABLED_MUX_HOLD,
                "mux is stale, invalid, in HOLD, or source-disagrees",
            )
        if telemetry is None:
            return (
                Px4OutputGateState.WAITING_TELEMETRY,
                "waiting for telemetry",
            )
        if telemetry_age > self.config.telemetry_timeout_s:
            return (
                Px4OutputGateState.DISABLED_STALE_TELEMETRY,
                "synthetic telemetry is stale",
            )
        if not self._telemetry_timestamp_valid(telemetry.timestamp_us):
            return (
                Px4OutputGateState.DISABLED_STALE_TELEMETRY,
                "telemetry timestamp moved backward",
            )
        if telemetry.failsafe or telemetry.offboard_control_signal_lost:
            return (
                Px4OutputGateState.DISABLED_FAILSAFE,
                "failsafe evidence active",
            )
        if not self._vehicle_state_acceptable(telemetry):
            return (
                Px4OutputGateState.WAITING_VEHICLE_STATE,
                "vehicle state or local telemetry evidence is unacceptable",
            )
        return (
            Px4OutputGateState.SAFE_TO_FORWARD,
            "all safety evidence healthy",
        )

    def _candidate_timestamp_valid(self, timestamp_us: int) -> bool:
        if (
            self._latest_candidate_timestamp_us is not None
            and timestamp_us < self._latest_candidate_timestamp_us
        ):
            return False
        self._latest_candidate_timestamp_us = timestamp_us
        return True

    def _telemetry_timestamp_valid(self, timestamp_us: int) -> bool:
        if (
            self._latest_telemetry_timestamp_us is not None
            and timestamp_us < self._latest_telemetry_timestamp_us
        ):
            return False
        self._latest_telemetry_timestamp_us = timestamp_us
        return True

    def _mux_valid(
        self, mux: MuxHealthEvidence | None, current_time_s: float | None
    ) -> bool:
        if mux is None or not mux.received or current_time_s is None:
            return False
        age = current_time_s - mux.receipt_time_s
        return (
            math.isfinite(age)
            and 0.0 <= age <= self.config.selected_command_timeout_s
            and mux.selected_command_valid
            and (
                not mux.hold_active
                or self.config.allow_controlled_mux_handoff
            )
            and bool(mux.active_source)
        )

    def _telemetry_valid(
        self,
        telemetry: Px4TelemetryState | None,
        age: float,
    ) -> bool:
        return (
            telemetry is not None
            and 0.0 <= age <= self.config.telemetry_timeout_s
            and self._vehicle_state_acceptable(telemetry)
            and not telemetry.failsafe
            and not telemetry.offboard_control_signal_lost
        )

    @staticmethod
    def _vehicle_state_acceptable(telemetry: Px4TelemetryState) -> bool:
        nav_valid = (
            0 <= telemetry.nav_state <= _NAVIGATION_STATE_MAX
            and telemetry.nav_state != _NAVIGATION_STATE_TERMINATION
        )
        return (
            telemetry.connected
            and telemetry.arming_state in _ACCEPTABLE_ARMING_STATES
            and nav_valid
            and telemetry.pre_flight_checks_pass
            and telemetry.local_position_valid
            and telemetry.local_velocity_valid
            and telemetry.odometry_valid
            and telemetry.pose_frame_ned
            and telemetry.velocity_frame_ned
        )

    @staticmethod
    def _vehicle_signature(
        telemetry: Px4TelemetryState | None,
    ) -> tuple[int, int, bool] | None:
        if telemetry is None:
            return None
        return (
            telemetry.arming_state,
            telemetry.nav_state,
            telemetry.offboard_active,
        )

    @staticmethod
    def _age(current_time_s: float, receipt_time_s: float | None) -> float:
        if receipt_time_s is None:
            return math.inf
        return current_time_s - receipt_time_s

    def _set_state(self, state: Px4OutputGateState, reason: str) -> None:
        if state != self.state:
            self.transition_count += 1
        self.state = state
        self.hold_reason = reason
