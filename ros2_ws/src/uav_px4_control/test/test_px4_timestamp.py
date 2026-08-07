"""Pure PX4 timestamp conversion and monotonicity tests."""

import pytest

from uav_px4_control.px4_boundary_models import UINT64_MAX
from uav_px4_control.px4_timestamp import (
    MonotonicTimestampTracker,
    ros_stamp_to_microseconds,
)


def test_zero_exact_and_sub_microsecond_truncation() -> None:
    """Use deterministic integer truncation, never float rounding."""
    assert ros_stamp_to_microseconds(0, 0) == 0
    assert ros_stamp_to_microseconds(0, 999) == 0
    assert ros_stamp_to_microseconds(0, 1_000) == 1
    assert ros_stamp_to_microseconds(12, 345_678_999) == 12_345_678


def test_large_timestamp_and_overflow() -> None:
    """Accept the uint64 boundary and reject the next microsecond."""
    seconds, micros = divmod(UINT64_MAX, 1_000_000)
    assert ros_stamp_to_microseconds(seconds, micros * 1_000) == UINT64_MAX
    with pytest.raises(OverflowError):
        ros_stamp_to_microseconds(seconds + 1, 0)


@pytest.mark.parametrize(
    "seconds,nanoseconds,error",
    [(-1, 0, ValueError), (0, -1, ValueError), (0, 1_000_000_000, ValueError)],
)
def test_invalid_ros_stamps_rejected(seconds, nanoseconds, error) -> None:
    """Require normalized nonnegative ROS time."""
    with pytest.raises(error):
        ros_stamp_to_microseconds(seconds, nanoseconds)


def test_monotonic_tracker_rejects_equal_and_backward_time() -> None:
    """Detect replay and backward time until explicit reset."""
    tracker = MonotonicTimestampTracker()
    tracker.accept(10)
    tracker.accept(11)
    with pytest.raises(ValueError, match="strictly monotonic"):
        tracker.accept(11)
    with pytest.raises(ValueError, match="strictly monotonic"):
        tracker.accept(9)
    tracker.reset()
    tracker.accept(9)
