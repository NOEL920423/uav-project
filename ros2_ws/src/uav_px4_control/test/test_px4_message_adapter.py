"""Regression tests for the exact local PX4 v1.14 message mapping."""

import math

import pytest

from uav_px4_control.px4_message_adapter import (
    offboard_control_mode_fields,
    trajectory_setpoint_fields,
    validate_stream_candidate,
)
from uav_px4_control.px4_stream_models import StreamCandidate


def candidate(velocity=(0.0, 0.0, 0.0), yaw_rate=0.0):
    """Create a fresh validated diagnostic candidate."""
    return StreamCandidate(
        receipt_time_s=1.0,
        timestamp_us=1_000_000,
        velocity_ned_mps=velocity,
        yaw_rate_ned_radps=yaw_rate,
    )


@pytest.mark.parametrize(
    "velocity",
    [
        (0.10, 0.0, 0.0),
        (0.0, 0.10, 0.0),
        (0.0, 0.0, 0.10),
        (0.0, 0.0, -0.10),
        (0.0, 0.0, 0.0),
    ],
)
def test_ned_velocity_is_identity_mapped(velocity):
    """Preserve north, east, and signed down without a second conversion."""
    fields = trajectory_setpoint_fields(candidate(velocity), 2_000_000)
    assert fields.velocity == velocity


def test_unused_fields_are_nan_and_yaw_rate_maps_to_yawspeed():
    """Disable unused fields and retain independent NED yaw rate."""
    fields = trajectory_setpoint_fields(candidate(yaw_rate=-0.25), 2_000_000)
    assert all(math.isnan(value) for value in fields.position)
    assert all(math.isnan(value) for value in fields.acceleration)
    assert all(math.isnan(value) for value in fields.jerk)
    assert math.isnan(fields.yaw)
    assert fields.yawspeed == -0.25


def test_offboard_control_mode_is_velocity_only():
    """Set exactly one intent flag without requesting an actual mode change."""
    fields = offboard_control_mode_fields(2_000_000)
    assert fields.timestamp == 2_000_000
    assert fields.velocity is True
    assert fields.position is False
    assert fields.acceleration is False
    assert fields.attitude is False
    assert fields.body_rate is False
    assert fields.actuator is False


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"frame_id": "map"}, "frame"),
        ({"timestamp_us": 0}, "timestamp"),
        ({"valid": False}, "invalid"),
        ({"velocity_ned_mps": (math.nan, 0.0, 0.0)}, "finite"),
        ({"velocity_ned_mps": (2.1, 0.0, 0.0)}, "component"),
        ({"yaw_rate_ned_radps": 1.6}, "yaw rate"),
    ],
)
def test_adapter_rejects_invalid_candidate(overrides, reason):
    """Independently reject malformed or expanded Phase 7 candidates."""
    values = {
        "receipt_time_s": 1.0,
        "timestamp_us": 1_000_000,
        "velocity_ned_mps": (0.0, 0.0, 0.0),
        "yaw_rate_ned_radps": 0.0,
        "frame_id": "px4_ned",
        "valid": True,
    }
    values.update(overrides)
    valid, message = validate_stream_candidate(StreamCandidate(**values))
    assert valid is False
    assert reason in message


def test_outgoing_timestamp_must_be_positive():
    """Reject zero before either live message is created."""
    with pytest.raises(ValueError, match="positive"):
        trajectory_setpoint_fields(candidate(), 0)
    with pytest.raises(ValueError, match="positive"):
        offboard_control_mode_fields(0)
