"""Velocity-only NED PX4 candidate mapping and validator tests."""

import math
from dataclasses import replace

import pytest

from uav_px4_control.control_source_models import (
    ASTAR_EXPERT,
    ControlCommand,
    Vector3,
)
from uav_px4_control.px4_boundary_models import Px4MappingConfig
from uav_px4_control.px4_candidate_validator import validate_px4_candidate
from uav_px4_control.px4_setpoint_mapper import map_selected_command


def command(
    north=0.0, east=0.0, down=0.0, yaw_rate=0.0,
    frame="px4_ned", source=ASTAR_EXPERT,
) -> ControlCommand:
    """Create one selected-command fixture."""
    return ControlCommand(
        source, 1.0, frame, Vector3(north, east, down),
        yaw_rate_radps=yaw_rate,
    )


def mapped(selected=None):
    """Map one selected fixture with deterministic evidence."""
    return map_selected_command(selected or command(), 1_000_000, 1.0)


def validate(candidate, **kwargs):
    """Validate one candidate with matching healthy mux evidence."""
    return validate_px4_candidate(
        candidate,
        Px4MappingConfig(),
        current_time_s=kwargs.pop("current_time_s", 1.01),
        mux_valid=kwargs.pop("mux_valid", True),
        mux_active_source=kwargs.pop(
            "mux_active_source", candidate.source
        ),
        **kwargs,
    )


@pytest.mark.parametrize(
    "north,east,down",
    [
        (0.0, 0.0, 0.0),
        (0.4, 0.0, 0.0),
        (0.0, 0.4, 0.0),
        (0.0, 0.0, 0.4),
        (0.0, 0.0, -0.4),
        (1.2, 1.6, 0.0),
    ],
)
def test_ned_components_map_identically(north, east, down) -> None:
    """Preserve north/east/down signs without ENU conversion."""
    candidate = mapped(command(north, east, down))
    assert candidate.velocity_ned_mps == (north, east, down)
    assert validate(candidate).valid


def test_yaw_rate_maps_without_inventing_absolute_yaw() -> None:
    """Use local yawspeed and keep absolute yaw explicitly unused."""
    candidate = mapped(command(yaw_rate=-0.7))
    assert candidate.yaw_rate_ned_radps == -0.7
    assert math.isnan(candidate.yaw_ned_rad)
    assert candidate.use_yaw_rate and not candidate.use_yaw
    assert validate(candidate).valid


def test_local_unused_fields_are_all_nan() -> None:
    """Mirror the local TrajectorySetpoint NaN disable convention."""
    candidate = mapped()
    assert all(math.isnan(value) for value in candidate.position_ned_m)
    assert all(math.isnan(value) for value in candidate.acceleration_ned_mps2)
    assert all(math.isnan(value) for value in candidate.jerk_ned_mps3)
    assert candidate.use_velocity and not candidate.use_position


@pytest.mark.parametrize(
    "selected,reason",
    [
        (command(frame="map"), "frame"),
        (command(source="ASTAR"), "unknown"),
        (command(north=math.nan), "non-finite"),
        (command(east=math.inf), "non-finite"),
        (command(north=1.5, east=1.5), "horizontal"),
        (command(down=1.01), "down"),
        (command(yaw_rate=1.51), "yaw rate"),
    ],
)
def test_independent_validator_rejects_invalid_mapping(
    selected, reason
) -> None:
    """Reject frame, source, finite, velocity, and yaw boundary failures."""
    result = validate(mapped(selected))
    assert not result.valid
    assert reason in result.reason


def test_stale_mux_mismatch_and_timestamp_regression_rejected() -> None:
    """Bind candidate to fresh selected evidence and the active mux source."""
    candidate = mapped()
    assert not validate(candidate, current_time_s=1.26).valid
    assert not validate(candidate, mux_valid=False).valid
    assert not validate(
        candidate, mux_active_source="HUMAN_JOYSTICK"
    ).valid
    assert not validate(
        candidate, previous_timestamp_us=1_000_000
    ).valid


def test_validator_rejects_non_nan_unused_field() -> None:
    """Prevent an accidental position or absolute-yaw command."""
    position = replace(mapped(), position_ned_m=(0.0, math.nan, math.nan))
    yaw = replace(mapped(), yaw_ned_rad=0.0)
    assert not validate(position).valid
    assert not validate(yaw).valid


def test_config_cannot_expand_phase6_boundary() -> None:
    """Keep Phase 7 defaults and overrides no looser than selected command."""
    with pytest.raises(ValueError, match="Phase 6"):
        Px4MappingConfig(maximum_total_velocity_mps=2.1)
