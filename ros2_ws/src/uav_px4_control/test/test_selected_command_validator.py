"""Independent selected-command validator regression tests."""

import math

import pytest

from uav_px4_control.control_source_models import (
    ASTAR_EXPERT,
    ControlCommand,
    ControlMuxConfig,
    ControlMuxState,
    HOLD,
    Vector3,
    zero_hold,
)
from uav_px4_control.selected_command_validator import (
    validate_selected_command,
)


def command(**changes) -> ControlCommand:
    """Build one valid selected movement command with field overrides."""
    values = {
        "source": ASTAR_EXPERT,
        "timestamp_s": 1.0,
        "frame_id": "px4_ned",
        "linear": Vector3(0.5, 0.0, 0.0),
        "yaw_rate_radps": 0.1,
    }
    values.update(changes)
    return ControlCommand(**values)


def diagnostics(value: ControlCommand, previous=None) -> set[str]:
    """Return constraint names for one independent validation call."""
    return {
        item.constraint for item in validate_selected_command(
            value,
            ControlMuxConfig(),
            ControlMuxState.ACTIVE_ASTAR_EXPERT,
            ASTAR_EXPERT,
            7,
            previous,
        )
    }


def test_valid_movement_and_hold_commands_pass() -> None:
    """Accept independently coherent normal and exact-zero HOLD outputs."""
    assert not diagnostics(command())
    hold = zero_hold(1.0, "test HOLD")
    assert not validate_selected_command(
        hold,
        ControlMuxConfig(),
        ControlMuxState.HOLD_REQUESTED,
        HOLD,
        1,
    )


@pytest.mark.parametrize(
    "value,constraint",
    [
        (command(source="ASTAR"), "allowed_source"),
        (command(source=HOLD), "active_source_consistency"),
        (command(frame_id="map"), "selected_frame"),
        (command(linear=Vector3(math.nan, 0.0, 0.0)),
         "finite_selected_command"),
        (command(angular_x=0.01), "selected_angular_xy"),
        (command(linear=Vector3(0.0, 2.01, 0.0)),
         "selected_horizontal_speed"),
        (command(linear=Vector3(0.0, 0.0, 1.01)),
         "selected_vertical_speed"),
        (command(linear=Vector3(1.8, 0.0, 1.0)),
         "selected_total_speed"),
        (command(yaw_rate_radps=1.51), "selected_yaw_rate"),
    ],
)
def test_independent_static_constraints(
    value: ControlCommand, constraint: str
) -> None:
    """Report ownership, frame, finite, component, and magnitude failures."""
    assert constraint in diagnostics(value)


def test_timestamp_acceleration_and_yaw_acceleration_constraints() -> None:
    """Independently reject output history and derivative violations."""
    previous = command(timestamp_s=1.0, linear=Vector3(0.0, 0.0, 0.0),
                       yaw_rate_radps=0.0)
    assert "selected_timestamp_monotonicity" in diagnostics(
        command(timestamp_s=1.0), previous
    )
    names = diagnostics(
        command(timestamp_s=1.1, linear=Vector3(0.5, 0.0, 0.0)), previous
    )
    assert "selected_acceleration" in names
    names = diagnostics(
        command(timestamp_s=1.1, linear=Vector3(0.0, 0.0, 0.0),
                yaw_rate_radps=0.5),
        previous,
    )
    assert "selected_yaw_acceleration" in names


def test_hold_magnitude_reason_and_state_constraints() -> None:
    """Reject nonzero, unexplained, or movement-state HOLD commands."""
    invalid = ControlCommand(
        HOLD, 1.0, "px4_ned", Vector3(0.1, 0.0, 0.0),
        hold_active=True,
    )
    names = {
        item.constraint for item in validate_selected_command(
            invalid,
            ControlMuxConfig(),
            ControlMuxState.ACTIVE_ASTAR_EXPERT,
            ASTAR_EXPERT,
            2,
        )
    }
    assert "selected_hold_magnitude" in names
    assert "selected_hold_reason" in names
    assert "selected_hold_state" in names


def test_diagnostic_carries_source_cycle_timestamp_and_reason() -> None:
    """Preserve the required structured evidence for every rejection."""
    item = validate_selected_command(
        command(frame_id="map"),
        ControlMuxConfig(),
        ControlMuxState.ACTIVE_ASTAR_EXPERT,
        ASTAR_EXPERT,
        19,
    )[0]
    assert item.source == ASTAR_EXPERT
    assert item.cycle_index == 19
    assert item.timestamp_s == pytest.approx(1.0)
    assert item.reason
