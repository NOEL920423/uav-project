"""Fail-closed state machine for the Phase 8 PX4 SITL streamer."""

import math

from uav_px4_control.px4_message_adapter import validate_stream_candidate
from uav_px4_control.px4_stream_models import (
    Px4StreamConfig,
    Px4StreamResult,
    Px4StreamState,
    StreamCandidate,
    StreamReadiness,
)


_STREAM_STATES = {
    Px4StreamState.PRESTREAMING,
    Px4StreamState.STREAMING,
}


class Px4StreamStateMachine:
    """Require independent enable, gate, SITL, DDS, and telemetry evidence."""

    def __init__(self, config: Px4StreamConfig | None = None) -> None:
        """Initialize disabled with no publication authority."""
        self.config = config or Px4StreamConfig()
        self.state = Px4StreamState.STREAM_DISABLED
        self.stream_enable_requested = False
        self.fault_latched = False
        self.stop_reason = "explicit stream enable required"
        self._latched_stop_reason = ""
        self.transition_count = 0
        self.trajectory_setpoint_count = 0
        self.offboard_mode_count = 0
        self.dropped_cycle_count = 0
        self._last_step_time_s: float | None = None
        self._last_candidate_timestamp_us: int | None = None
        self._candidate_receipts: list[tuple[int, float]] = []
        self._candidate_time_regressed = False
        self._prestream_start_time_s: float | None = None
        self._prestream_start_count = 0
        self._last_publish_time_s: float | None = None
        self._first_publish_time_s: float | None = None
        self._session_publish_count = 0
        self._last_outgoing_timestamp_us: int | None = None
        self._maximum_observed_gap_s = 0.0

    @property
    def observed_rate_hz(self) -> float:
        """Return cumulative achieved rate between first and last output."""
        if (
            self._first_publish_time_s is None
            or self._last_publish_time_s is None
            or self._session_publish_count < 2
        ):
            return 0.0
        duration = self._last_publish_time_s - self._first_publish_time_s
        if duration <= 0.0:
            return 0.0
        return (self._session_publish_count - 1) / duration

    @property
    def maximum_observed_gap_s(self) -> float:
        """Return the largest interval seen between output pairs."""
        return self._maximum_observed_gap_s

    def request_enable(self, enable: bool) -> tuple[bool, str]:
        """Apply explicit enable, or disable/reset without PX4 side effects."""
        if not isinstance(enable, bool):
            return False, "enable must be a boolean"
        if not enable:
            self.stream_enable_requested = False
            self.fault_latched = False
            self._latched_stop_reason = ""
            self._reset_readiness()
            self._set_state(
                Px4StreamState.STREAM_DISABLED,
                "stream explicitly disabled; fault latch cleared",
            )
            return True, self.stop_reason
        if self.fault_latched:
            return False, "disable/reset is required before re-enabling"
        self.stream_enable_requested = True
        self.stop_reason = "waiting for complete stream readiness"
        return True, "stream enable accepted; readiness is still required"

    def observe_candidate(self, candidate: StreamCandidate) -> None:
        """Track distinct monotonic Phase 7 candidate heartbeats."""
        previous = self._last_candidate_timestamp_us
        if previous is not None and candidate.timestamp_us < previous:
            self._candidate_time_regressed = True
            return
        if previous is None or candidate.timestamp_us > previous:
            self._candidate_receipts.append(
                (candidate.timestamp_us, candidate.receipt_time_s)
            )
            self._last_candidate_timestamp_us = candidate.timestamp_us

    def step(
        self,
        now_s: float,
        readiness: StreamReadiness,
        outgoing_timestamp_us: int,
    ) -> Px4StreamResult:
        """Evaluate safety evidence and authorize at most one publish pair."""
        if not math.isfinite(now_s) or now_s < 0.0:
            return self._fault(
                Px4StreamState.STOPPED_TIME_JUMP,
                "ROS node time is invalid",
                now_s,
                readiness,
            )
        time_jump = (
            self._last_step_time_s is not None
            and now_s < self._last_step_time_s
        )
        self._last_step_time_s = now_s
        self._trim_candidate_receipts(now_s)
        if self.fault_latched:
            self._set_state(
                Px4StreamState.LATCHED_STREAM_FAULT,
                "stream fault latched after: "
                f"{self._latched_stop_reason}; disable/reset is required",
            )
            return self._result(now_s, readiness, False)
        if not self.stream_enable_requested:
            self._set_state(
                Px4StreamState.STREAM_DISABLED,
                "explicit stream enable required",
            )
            return self._result(now_s, readiness, False)
        if time_jump or self._candidate_time_regressed:
            return self._fault(
                Px4StreamState.STOPPED_TIME_JUMP,
                "ROS or candidate timestamp moved backward",
                now_s,
                readiness,
            )

        failure = self._readiness_failure(now_s, readiness)
        if failure is not None:
            state, reason, latch = failure
            if latch:
                return self._fault(state, reason, now_s, readiness)
            self._set_state(state, reason)
            return self._result(now_s, readiness, False)

        if self._last_publish_time_s is not None:
            gap = now_s - self._last_publish_time_s
            if gap > self.config.maximum_publish_gap_s:
                self._record_dropped_cycles(gap)
                return self._fault(
                    Px4StreamState.STOPPED_PUBLISH_GAP,
                    "outgoing setpoint publication gap exceeded limit",
                    now_s,
                    readiness,
                )
        if (
            not isinstance(outgoing_timestamp_us, int)
            or outgoing_timestamp_us <= 0
            or (
                self._last_outgoing_timestamp_us is not None
                and outgoing_timestamp_us
                <= self._last_outgoing_timestamp_us
            )
        ):
            return self._fault(
                Px4StreamState.STOPPED_TIME_JUMP,
                "outgoing PX4 timestamp is zero, invalid, or non-monotonic",
                now_s,
                readiness,
            )

        if (
            self.state not in _STREAM_STATES
            and self.state != Px4StreamState.PRESTREAM_READY
        ):
            self._set_state(
                Px4StreamState.PRESTREAM_READY,
                "all readiness evidence is stable",
            )
            return self._result(now_s, readiness, False)

        if self.state == Px4StreamState.PRESTREAM_READY:
            self._set_state(
                Px4StreamState.PRESTREAMING,
                "publishing begins on this validated timer tick",
            )

        self._record_publication(now_s, outgoing_timestamp_us)
        if self._prestream_start_time_s is None:
            self._prestream_start_time_s = now_s
            self._prestream_start_count = self.trajectory_setpoint_count
        elapsed = now_s - self._prestream_start_time_s
        count = (
            self.trajectory_setpoint_count
            - self._prestream_start_count
            + 1
        )
        if (
            elapsed >= self.config.minimum_prestream_duration_s
            and count >= self.config.minimum_prestream_messages
        ):
            self._set_state(
                Px4StreamState.STREAMING,
                "verified SITL setpoint stream; no mode request is sent",
            )
        else:
            self._set_state(
                Px4StreamState.PRESTREAMING,
                "publishing SITL-only velocity prestream",
            )
        return self._result(now_s, readiness, True)

    def force_mapping_fault(
        self,
        now_s: float,
        readiness: StreamReadiness,
        reason: str,
    ) -> Px4StreamResult:
        """Latch an adapter failure before either live message is published."""
        return self._fault(
            Px4StreamState.STOPPED_INVALID_MAPPING,
            reason,
            now_s,
            readiness,
        )

    def _readiness_failure(
        self,
        now_s: float,
        readiness: StreamReadiness,
    ) -> tuple[Px4StreamState, str, bool] | None:
        started = self._last_publish_time_s is not None
        if not readiness.sitl_guard_valid:
            state = (
                Px4StreamState.STOPPED_DDS_LOSS
                if started
                else Px4StreamState.WAITING_SITL
            )
            return state, "PX4 SITL identity guard is not satisfied", started
        if not readiness.dds_ready:
            state = (
                Px4StreamState.STOPPED_DDS_LOSS
                if started
                else Px4StreamState.WAITING_DDS
            )
            return state, "required DDS endpoints are not ready", started
        gate = readiness.gate
        if gate is None:
            return (
                Px4StreamState.WAITING_GATE,
                "waiting for Phase 7 gate",
                False,
            )
        gate_age = now_s - gate.status_receipt_time_s
        bool_age = now_s - gate.bool_receipt_time_s
        if (
            gate_age < 0.0
            or bool_age < 0.0
            or gate_age > self.config.gate_status_timeout_s
            or bool_age > self.config.gate_status_timeout_s
        ):
            return (
                Px4StreamState.STOPPED_STALE_GATE,
                "Phase 7 gate evidence is stale",
                True,
            )
        agreement = (
            gate.bool_safe_to_forward == gate.status_safe_to_forward
        )
        gate_safe = (
            gate.bool_safe_to_forward
            and gate.status_safe_to_forward
            and gate.status_state == "SAFE_TO_FORWARD"
        )
        if self.config.require_gate_status_agreement and not agreement:
            return (
                Px4StreamState.STOPPED_GATE_FALSE,
                "Phase 7 gate bool and status disagree",
                True,
            )
        if self.config.require_safe_to_forward and not gate_safe:
            return (
                Px4StreamState.STOPPED_GATE_FALSE,
                "Phase 7 safe_to_forward is false",
                True,
            )
        candidate = readiness.candidate
        if candidate is None:
            return (
                Px4StreamState.WAITING_CANDIDATE,
                "waiting for validated Phase 7 candidate",
                False,
            )
        candidate_age = now_s - candidate.receipt_time_s
        if (
            candidate_age < 0.0
            or candidate_age > self.config.candidate_timeout_s
        ):
            return (
                Px4StreamState.STOPPED_STALE_CANDIDATE,
                "Phase 7 candidate is stale",
                True,
            )
        valid, reason = validate_stream_candidate(candidate)
        if not valid:
            return Px4StreamState.STOPPED_INVALID_MAPPING, reason, True
        if (
            not started
            and len(self._candidate_receipts)
            < self.config.minimum_candidate_updates
        ):
            return (
                Px4StreamState.WAITING_CANDIDATE,
                "waiting for stable monotonic candidate heartbeat window",
                False,
            )
        telemetry = readiness.telemetry
        if telemetry is None:
            return (
                Px4StreamState.WAITING_TELEMETRY,
                "waiting for all required PX4 telemetry",
                False,
            )
        telemetry_age = now_s - telemetry.oldest_receipt_time_s
        if (
            telemetry_age < 0.0
            or telemetry_age > self.config.telemetry_timeout_s
        ):
            return (
                Px4StreamState.STOPPED_STALE_TELEMETRY,
                "required PX4 telemetry is stale",
                True,
            )
        if self.config.require_no_failsafe and telemetry.failsafe:
            return (
                Px4StreamState.STOPPED_FAILSAFE,
                "PX4 failsafe is active",
                True,
            )
        if self.config.require_disarmed and telemetry.vehicle_armed:
            return (
                Px4StreamState.STOPPED_ARMED,
                "PX4 unexpectedly armed",
                True,
            )
        if self.config.require_offboard_inactive and telemetry.offboard_active:
            return (
                Px4StreamState.STOPPED_OFFBOARD_ACTIVE,
                "PX4 unexpectedly reports OFFBOARD active",
                True,
            )
        if not telemetry.odometry_valid:
            return (
                Px4StreamState.STOPPED_INVALID_MAPPING,
                "PX4 odometry is invalid",
                True,
            )
        return None

    def _record_publication(self, now_s: float, timestamp_us: int) -> None:
        if self._session_publish_count == 0:
            self._first_publish_time_s = now_s
        if self._last_publish_time_s is not None:
            gap = now_s - self._last_publish_time_s
            self._maximum_observed_gap_s = max(
                self._maximum_observed_gap_s,
                gap,
            )
            self._record_dropped_cycles(gap)
        self._last_publish_time_s = now_s
        self._last_outgoing_timestamp_us = timestamp_us
        self._session_publish_count += 1
        self.trajectory_setpoint_count += 1
        self.offboard_mode_count += 1

    def _record_dropped_cycles(self, gap_s: float) -> None:
        expected = 1.0 / self.config.stream_rate_hz
        missed = max(0, int(math.floor(gap_s / expected + 1e-9)) - 1)
        self.dropped_cycle_count += missed

    def _trim_candidate_receipts(self, now_s: float) -> None:
        cutoff = now_s - self.config.candidate_timeout_s
        self._candidate_receipts = [
            item for item in self._candidate_receipts if item[1] >= cutoff
        ]

    def _fault(
        self,
        state: Px4StreamState,
        reason: str,
        now_s: float,
        readiness: StreamReadiness,
    ) -> Px4StreamResult:
        self._set_state(state, reason)
        self._latched_stop_reason = reason
        self.stream_enable_requested = False
        self.fault_latched = self.config.latch_stream_faults
        self._prestream_start_time_s = None
        return self._result(now_s, readiness, False)

    def _reset_readiness(self) -> None:
        self._candidate_receipts.clear()
        self._last_candidate_timestamp_us = None
        self._candidate_time_regressed = False
        self._prestream_start_time_s = None
        self._prestream_start_count = 0
        self._first_publish_time_s = None
        self._last_publish_time_s = None
        self._session_publish_count = 0
        self._last_outgoing_timestamp_us = None

    def _set_state(self, state: Px4StreamState, reason: str) -> None:
        if state != self.state:
            self.transition_count += 1
        self.state = state
        self.stop_reason = reason

    def _result(
        self,
        now_s: float,
        readiness: StreamReadiness,
        should_publish: bool,
    ) -> Px4StreamResult:
        gate = readiness.gate
        candidate = readiness.candidate
        telemetry = readiness.telemetry
        gate_age = (
            math.inf if gate is None else now_s - gate.status_receipt_time_s
        )
        candidate_age = (
            math.inf if candidate is None else now_s - candidate.receipt_time_s
        )
        telemetry_age = (
            math.inf
            if telemetry is None
            else now_s - telemetry.oldest_receipt_time_s
        )
        candidate_valid = False
        if candidate is not None:
            candidate_valid = validate_stream_candidate(candidate)[0]
        gate_valid = bool(
            gate is not None
            and gate.bool_safe_to_forward
            and gate.status_safe_to_forward
            and gate.status_state == "SAFE_TO_FORWARD"
            and gate.bool_safe_to_forward == gate.status_safe_to_forward
            and 0.0 <= gate_age <= self.config.gate_status_timeout_s
        )
        telemetry_fresh = bool(
            telemetry is not None
            and 0.0 <= telemetry_age <= self.config.telemetry_timeout_s
        )
        return Px4StreamResult(
            state=self.state,
            stream_enable_requested=self.stream_enable_requested,
            streaming=self.state in _STREAM_STATES and should_publish,
            should_publish=should_publish,
            sitl_guard_valid=readiness.sitl_guard_valid,
            dds_ready=readiness.dds_ready,
            gate_valid=gate_valid,
            candidate_valid=candidate_valid,
            telemetry_fresh=telemetry_fresh,
            vehicle_armed=(
                False if telemetry is None else telemetry.vehicle_armed
            ),
            offboard_active=(
                False if telemetry is None else telemetry.offboard_active
            ),
            failsafe=False if telemetry is None else telemetry.failsafe,
            candidate_age_s=candidate_age,
            gate_status_age_s=gate_age,
            telemetry_age_s=telemetry_age,
            trajectory_setpoint_count=self.trajectory_setpoint_count,
            offboard_mode_count=self.offboard_mode_count,
            dropped_cycle_count=self.dropped_cycle_count,
            transition_count=self.transition_count,
            stop_reason=self.stop_reason,
        )
