"""Unit and regression tests for pure trajectory parameterization."""

import math

import pytest

from uav_navigation.models import Point3D
from uav_navigation.trajectory_metrics import vector_norm
from uav_navigation.trajectory_models import TrajectoryConfig
from uav_navigation.trajectory_parameterizer import parameterize_trajectory


def path(*coordinates):
    """Build a constant-altitude pure path."""
    return tuple(Point3D(x, y, -2.0) for x, y in coordinates)


def test_configuration_defaults_and_rejections():
    """Defaults are valid and malformed limits fail at construction."""
    assert TrajectoryConfig().maximum_speed_mps == 2.0
    with pytest.raises(ValueError):
        TrajectoryConfig(maximum_speed_mps=-1.0)
    with pytest.raises(ValueError):
        TrajectoryConfig(maximum_jerk_mps3=math.nan)
    with pytest.raises(ValueError):
        TrajectoryConfig(start_speed_mps=3.0)
    with pytest.raises(ValueError):
        TrajectoryConfig(start_speed_mps=0.1)
    with pytest.raises(ValueError):
        TrajectoryConfig(require_zero_start_speed=1)


def test_straight_profile_reaches_limit_and_stops_at_ends():
    """A long straight path reaches the speed cap with zero endpoints."""
    result = parameterize_trajectory(path((0, 0), (4, 0), (8, 0), (12, 0)))
    speeds = [
        vector_norm(point.velocity) for point in result.trajectory_points
    ]
    assert result.valid
    assert max(speeds) == pytest.approx(2.0)
    assert speeds[0] == 0.0
    assert speeds[-1] == 0.0


def test_short_path_is_acceleration_and_deceleration_limited():
    """Forward/backward squared-speed passes constrain short paths."""
    result = parameterize_trajectory(path((0, 0), (0.1, 0), (0.2, 0)))
    middle_speed = vector_norm(result.trajectory_points[1].velocity)
    expected = math.sqrt(2.0 * 1.5 * 0.1)
    assert result.valid
    assert 0.0 < middle_speed <= expected + 1e-9


def test_curvature_reduces_speed_and_respects_lateral_limit():
    """A tight bend has a lower curvature-bound point speed."""
    fixture = path((0, 0), (2, 0), (2, 0.25), (2, 2), (4, 2))
    result = parameterize_trajectory(fixture)
    speeds = [
        vector_norm(point.velocity) for point in result.trajectory_points
    ]
    assert result.valid
    assert min(speeds[1:-1]) < 2.0
    assert result.maximum_lateral_acceleration_mps2 <= 1.5 + 1e-7


def test_time_is_deterministic_strict_and_duplicate_safe():
    """Duplicate cleanup keeps deterministic positive movement time."""
    fixture = path((0, 0), (0, 0), (1, 0), (2, 0))
    first = parameterize_trajectory(fixture)
    second = parameterize_trajectory(fixture)
    times = [point.time_from_start_s for point in first.trajectory_points]
    assert first.valid
    assert len(first.trajectory_points) == 3
    assert times[0] == 0.0
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    assert first.total_duration_s == second.total_duration_s


def test_two_point_zero_endpoint_case_has_positive_duration():
    """Two zero-speed endpoints still receive a conservative motion time."""
    result = parameterize_trajectory(path((0, 0), (1, 0)))
    assert result.valid
    assert result.total_duration_s > 0.0


def test_dynamics_limits_and_time_scaling():
    """A sharp path is globally slowed until jerk and yaw limits pass."""
    config = TrajectoryConfig(
        maximum_jerk_mps3=0.25,
        maximum_yaw_rate_radps=0.3,
        maximum_yaw_acceleration_radps2=0.4,
    )
    result = parameterize_trajectory(
        path((0, 0), (1, 0), (1, 0.2), (1, 1), (2, 1)),
        config=config,
    )
    assert result.valid
    assert result.time_scale > 1.0
    assert result.maximum_speed_mps <= config.maximum_speed_mps + 1e-7
    assert result.maximum_jerk_mps3 <= config.maximum_jerk_mps3 + 1e-7
    assert result.maximum_yaw_rate_radps <= (
        config.maximum_yaw_rate_radps + 1e-7
    )


def test_impossible_scaling_budget_is_structured_rejection():
    """An inadequate scale budget returns finite points but valid false."""
    config = TrajectoryConfig(
        maximum_jerk_mps3=0.001,
        maximum_total_time_scale=1.001,
    )
    result = parameterize_trajectory(
        path((0, 0), (1, 0), (1, 1), (2, 1)), config=config
    )
    assert result.success
    assert not result.valid
    assert result.trajectory_points
    assert result.rejection_reason
    assert result.validation_diagnostics


def test_geometry_and_altitude_are_exactly_preserved():
    """Parameterization does not alter source positions or NED altitude."""
    fixture = (
        Point3D(0.0, 0.0, -1.5),
        Point3D(1.0, 0.2, -1.7),
        Point3D(2.0, 0.5, -1.9),
    )
    result = parameterize_trajectory(fixture)
    assert result.valid
    output_positions = tuple(
        point.position for point in result.trajectory_points
    )
    assert output_positions == fixture
    assert result.trajectory_points[0].position == fixture[0]
    assert result.trajectory_points[-1].position == fixture[-1]


def test_wrong_frame_and_too_few_points_are_rejected():
    """Invalid path contracts never yield a valid result."""
    assert not parameterize_trajectory(
        path((0, 0), (1, 0)), frame_id="map"
    ).valid
    assert not parameterize_trajectory(path((0, 0))).valid
