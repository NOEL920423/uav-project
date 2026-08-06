"""Pure tests for tracking law, ordered bounds, and command validation."""

import math

import pytest

from uav_navigation.models import Point3D
from uav_navigation.tracking_models import (
    ReferenceSample,
    TrackingConfig,
    TrackingState,
    VehicleState,
    VelocityCommand,
)
from uav_navigation.tracking_validator import validate_tracking_command
from uav_navigation.trajectory_models import TrajectoryPoint
from uav_navigation.trajectory_tracker import (
    compute_tracking_command,
    hold_command,
    tracking_errors,
)


def _reference(position=Point3D(1.0, 2.0, -3.0), yaw=0.0):
    """Build a finite pure reference sample."""
    return ReferenceSample(TrajectoryPoint(
        time_from_start_s=0.5,
        position=position,
        velocity=Point3D(0.5, -0.5, 0.25),
        acceleration=Point3D(0.0, 0.0, 0.0),
        jerk=Point3D(0.0, 0.0, 0.0),
        yaw_ned=yaw,
        yaw_rate_radps=0.1,
        yaw_acceleration_radps2=0.0,
        arc_length_m=1.0,
        curvature_inverse_m=0.0,
    ), 0)


def _state(position=Point3D(0.0, 0.0, -2.0), yaw=0.0):
    """Build finite px4_ned odometry state."""
    return VehicleState(
        timestamp_s=1.0,
        frame_id="px4_ned",
        position=position,
        velocity=Point3D(0.1, 0.2, -0.1),
        yaw_ned=yaw,
        yaw_rate_radps=0.0,
    )


def test_tracking_config_rejects_bad_values_and_contradictory_limits():
    """Typed configuration rejects NaN, booleans, and limit inversion."""
    assert TrackingConfig().position_kp == 1.0
    with pytest.raises(ValueError):
        TrackingConfig(position_kp=math.nan)
    with pytest.raises(ValueError):
        TrackingConfig(control_period_s=0.0)
    with pytest.raises(ValueError):
        TrackingConfig(reject_wrong_frame=1)
    with pytest.raises(ValueError):
        TrackingConfig(
            maximum_command_speed_mps=1.0,
            maximum_horizontal_command_speed_mps=2.0,
        )


def test_feedforward_feedback_preserves_ned_signs():
    """Unsaturated command equals the specified feedforward-feedback law."""
    config = TrackingConfig(
        maximum_command_speed_mps=20.0,
        maximum_horizontal_command_speed_mps=20.0,
        maximum_vertical_command_speed_mps=20.0,
    )
    reference = _reference()
    state = _state()
    unsaturated, selected, flags = compute_tracking_command(
        reference, state, 1.0, config
    )
    assert unsaturated.linear == Point3D(1.58, 1.36, -0.68)
    assert selected == unsaturated
    assert flags.count == 0
    errors = tracking_errors(reference, state)
    assert errors.vertical_position_error_m == 1.0
    assert errors.along_track_error_m == 1.0
    assert errors.cross_track_error_m == 2.0


def test_zero_tracking_error_returns_exact_reference_feedforward():
    """Zero feedback error leaves the complete NED feedforward unchanged."""
    reference = _reference(yaw=0.4)
    point = reference.point
    state = VehicleState(
        timestamp_s=1.0,
        frame_id="px4_ned",
        position=point.position,
        velocity=point.velocity,
        yaw_ned=point.yaw_ned,
        yaw_rate_radps=point.yaw_rate_radps,
    )
    unsaturated, selected, flags = compute_tracking_command(
        reference,
        state,
        1.0,
        TrackingConfig(
            maximum_command_speed_mps=20.0,
            maximum_horizontal_command_speed_mps=20.0,
            maximum_vertical_command_speed_mps=20.0,
        ),
    )
    assert unsaturated.linear == point.velocity
    assert unsaturated.yaw_rate_radps == point.yaw_rate_radps
    assert selected == unsaturated
    assert flags.count == 0


def test_ordered_speed_vertical_total_and_yaw_saturations_are_reported():
    """Each spatial and yaw clamp is explicit in the structured flags."""
    config = TrackingConfig(
        maximum_command_speed_mps=1.0,
        maximum_horizontal_command_speed_mps=0.8,
        maximum_vertical_command_speed_mps=0.4,
        maximum_yaw_rate_command_radps=0.5,
    )
    reference = _reference(Point3D(10.0, 10.0, 10.0), yaw=math.pi)
    _, selected, flags = compute_tracking_command(
        reference, _state(), 1.0, config
    )
    speed = math.sqrt(
        selected.linear.x**2
        + selected.linear.y**2
        + selected.linear.z**2
    )
    assert flags.horizontal_speed
    assert flags.vertical_speed
    assert flags.yaw_rate
    assert speed <= 1.0 + 1e-9
    assert abs(selected.yaw_rate_radps) <= 0.5


def test_acceleration_and_yaw_acceleration_limit_from_previous_valid_command():
    """Rate bounds use elapsed time and prior independently valid command."""
    previous = VelocityCommand(
        1.0, "px4_ned", Point3D(0.0, 0.0, 0.0), 0.0
    )
    _, selected, flags = compute_tracking_command(
        _reference(Point3D(10.0, 0.0, -2.0), yaw=math.pi),
        _state(),
        1.1,
        TrackingConfig(),
        previous,
    )
    assert flags.acceleration
    assert flags.yaw_acceleration
    assert math.sqrt(
        selected.linear.x**2
        + selected.linear.y**2
        + selected.linear.z**2
    ) == pytest.approx(0.15)
    assert abs(selected.yaw_rate_radps) == pytest.approx(0.2)


def test_hold_is_exact_and_validator_is_independent_of_tracker():
    """Validator accepts exact HOLD and rejects a manually corrupted result."""
    config = TrackingConfig()
    hold = hold_command(1.0, "test gate")
    assert hold.linear == Point3D(0.0, 0.0, 0.0)
    assert not validate_tracking_command(
        hold, config, TrackingState.HOLD_STALE_ODOMETRY, 1
    )
    corrupt = VelocityCommand(
        1.0, "px4_ned", Point3D(3.0, 0.0, 0.0), 0.0
    )
    diagnostics = validate_tracking_command(
        corrupt, config, TrackingState.TRACKING, 2
    )
    assert {item.constraint for item in diagnostics} >= {
        "horizontal_speed", "total_speed"
    }
    assert diagnostics[0].cycle_index == 2
