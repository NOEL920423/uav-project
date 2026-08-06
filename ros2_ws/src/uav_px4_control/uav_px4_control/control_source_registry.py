"""ROS-independent candidate registry and receipt-time freshness gates."""

import math

from uav_px4_control.control_source_models import (
    CONTROL_SOURCES,
    CandidateRecord,
    ControlCommand,
    ControlMuxConfig,
    HOLD,
    SourceHealth,
    VALID_COMMAND_FRAME,
    command_speed,
)


class ControlSourceRegistry:
    """Retain and classify only the latest candidate from each source."""

    def __init__(self, config: ControlMuxConfig) -> None:
        """Initialize every canonical source as never received."""
        self.config = config
        self._records = {
            source: CandidateRecord(source=source)
            for source in CONTROL_SOURCES
        }

    def clear(self) -> None:
        """Invalidate all receipt and publisher-stamp evidence."""
        self._records = {
            source: CandidateRecord(source=source)
            for source in CONTROL_SOURCES
        }

    def record(self, source: str) -> CandidateRecord:
        """Return one immutable record or reject an unknown identifier."""
        if source not in self._records:
            raise ValueError(f"unknown control source: {source}")
        return self._records[source]

    def update(
        self,
        source: str,
        command: ControlCommand,
        receipt_time_s: float,
    ) -> CandidateRecord:
        """Record an update without trusting its publisher stamp or fields."""
        if source not in self._records:
            raise ValueError(f"unknown control source: {source}")
        receipt = float(receipt_time_s)
        if not math.isfinite(receipt):
            raise ValueError("candidate receipt time must be finite")
        previous = self._records[source]
        finite, valid, reason = self._validate_static(
            source, command, previous
        )
        record = CandidateRecord(
            source=source,
            command=command,
            receipt_time_s=receipt,
            message_stamp_s=command.timestamp_s,
            finite=finite,
            valid=valid,
            reason=reason,
            update_count=previous.update_count + 1,
        )
        self._records[source] = record
        return record

    def _validate_static(
        self,
        source: str,
        command: ControlCommand,
        previous: CandidateRecord,
    ) -> tuple[bool, bool, str]:
        values = (
            command.timestamp_s,
            command.linear.x,
            command.linear.y,
            command.linear.z,
            command.angular_x,
            command.angular_y,
            command.yaw_rate_radps,
        )
        finite = all(math.isfinite(value) for value in values)
        if not finite:
            return False, False, "candidate contains non-finite fields"
        if command.source != source:
            return True, False, "candidate source ownership mismatch"
        if (
            self.config.reject_wrong_frame
            and command.frame_id != VALID_COMMAND_FRAME
        ):
            return True, False, f"candidate frame is {command.frame_id}"
        epsilon = self.config.hold_command_epsilon
        if (
            abs(command.angular_x) > epsilon
            or abs(command.angular_y) > epsilon
        ):
            return True, False, "candidate angular x/y must be zero"
        if (
            self.config.require_monotonic_candidate_stamps
            and previous.message_stamp_s is not None
            and command.timestamp_s <= previous.message_stamp_s
        ):
            return True, False, "candidate publisher stamp is non-monotonic"
        horizontal = math.hypot(command.linear.x, command.linear.y)
        if (
            horizontal
            > self.config.maximum_selected_horizontal_speed_mps + 1e-8
        ):
            return True, False, "candidate horizontal speed exceeds limit"
        if (
            abs(command.linear.z)
            > self.config.maximum_selected_vertical_speed_mps + 1e-8
        ):
            return True, False, "candidate vertical speed exceeds limit"
        if (
            command_speed(command)
            > self.config.maximum_selected_speed_mps + 1e-8
        ):
            return True, False, "candidate total speed exceeds limit"
        if (
            abs(command.yaw_rate_radps)
            > self.config.maximum_selected_yaw_rate_radps + 1e-8
        ):
            return True, False, "candidate yaw rate exceeds limit"
        if source == HOLD:
            magnitude = max(
                command_speed(command), abs(command.yaw_rate_radps)
            )
            if magnitude > epsilon:
                return True, False, "external HOLD candidate is nonzero"
        return True, True, "candidate is statically valid"

    def health(self, source: str, now_s: float) -> SourceHealth:
        """Classify one source using node-clock receipt age."""
        now = float(now_s)
        if not math.isfinite(now):
            raise ValueError("health query time must be finite")
        record = self.record(source)
        if record.receipt_time_s is None:
            return SourceHealth(
                source, False, False, False, False, math.inf,
                "never received", record.update_count,
            )
        age = now - record.receipt_time_s
        if age < 0.0:
            return SourceHealth(
                source, True, record.finite, False, False, age,
                "candidate receipt is in the future", record.update_count,
            )
        fresh = age <= self.config.timeout_for(source)
        reason = record.reason if record.valid else record.reason
        if record.valid and not fresh:
            reason = "candidate receipt is stale"
        return SourceHealth(
            source=source,
            received=True,
            finite=record.finite,
            valid=record.valid,
            fresh=fresh,
            age_s=age,
            reason=reason,
            update_count=record.update_count,
        )

    def healthy_sources(self, now_s: float) -> tuple[str, ...]:
        """Return canonical sources currently healthy in stable order."""
        return tuple(
            source for source in CONTROL_SOURCES
            if self.health(source, now_s).healthy
        )

    def stale_sources(self, now_s: float) -> tuple[str, ...]:
        """Return received sources whose receipt timeout has expired."""
        return tuple(
            source for source in CONTROL_SOURCES
            if self.health(source, now_s).received
            and not self.health(source, now_s).fresh
        )
