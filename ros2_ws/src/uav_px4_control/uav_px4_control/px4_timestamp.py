"""Pure deterministic ROS stamp to PX4 microsecond conversion."""

from dataclasses import dataclass

from uav_px4_control.px4_boundary_models import UINT64_MAX


def ros_stamp_to_microseconds(seconds: int, nanoseconds: int) -> int:
    """Convert a normalized ROS stamp, truncating sub-microseconds."""
    if isinstance(seconds, bool) or not isinstance(seconds, int):
        raise TypeError("seconds must be an integer")
    if isinstance(nanoseconds, bool) or not isinstance(nanoseconds, int):
        raise TypeError("nanoseconds must be an integer")
    if seconds < 0:
        raise ValueError("seconds must be nonnegative")
    if nanoseconds < 0 or nanoseconds >= 1_000_000_000:
        raise ValueError("nanoseconds must be normalized")
    timestamp_us = seconds * 1_000_000 + nanoseconds // 1_000
    if timestamp_us > UINT64_MAX:
        raise OverflowError("PX4 uint64 microsecond timestamp overflow")
    return timestamp_us


@dataclass(slots=True)
class MonotonicTimestampTracker:
    """Reject equal or backward timestamps after the first observation."""

    last_timestamp_us: int | None = None

    def accept(self, timestamp_us: int) -> None:
        """Store one strictly increasing uint64 timestamp."""
        if isinstance(timestamp_us, bool) or not isinstance(timestamp_us, int):
            raise TypeError("timestamp_us must be an integer")
        if timestamp_us < 0 or timestamp_us > UINT64_MAX:
            raise ValueError("timestamp_us is outside uint64")
        if (
            self.last_timestamp_us is not None
            and timestamp_us <= self.last_timestamp_us
        ):
            raise ValueError("timestamp is not strictly monotonic")
        self.last_timestamp_us = timestamp_us

    def reset(self) -> None:
        """Clear history only during an explicit recovery flow."""
        self.last_timestamp_us = None
