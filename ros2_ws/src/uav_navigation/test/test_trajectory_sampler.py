"""Pure regression tests for deterministic trajectory reference sampling."""

import math
from dataclasses import replace

import pytest

from uav_navigation.models import Point3D
from uav_navigation.trajectory_models import TrajectoryPoint
from uav_navigation.trajectory_sampler import sample_trajectory


def _point(time_s, value, yaw):
    """Build a point whose fields make interpolation observable."""
    return TrajectoryPoint(
        time_from_start_s=time_s,
        position=Point3D(value, value + 1.0, value + 2.0),
        velocity=Point3D(value + 3.0, value + 4.0, value + 5.0),
        acceleration=Point3D(value + 6.0, value + 7.0, value + 8.0),
        jerk=Point3D(value + 9.0, value + 10.0, value + 11.0),
        yaw_ned=yaw,
        yaw_rate_radps=value + 12.0,
        yaw_acceleration_radps2=value + 13.0,
        arc_length_m=value + 14.0,
        curvature_inverse_m=value + 15.0,
    )


def test_before_exact_and_after_samples_preserve_endpoints():
    """Boundary sampling returns the exact immutable source points."""
    points = (_point(0.0, 0.0, 3.0), _point(2.0, 2.0, 3.4))
    before = sample_trajectory(points, -0.1)
    first = sample_trajectory(points, 0.0)
    final = sample_trajectory(points, 2.0)
    after = sample_trajectory(points, 4.0)
    assert before.point is points[0] and before.prestart
    assert first.point is points[0] and not first.prestart
    assert final.point is points[-1] and final.terminal
    assert after.point is points[-1] and after.terminal


def test_interior_interpolates_every_field_and_unwrapped_yaw():
    """Interior reference uses one ratio for all scalar/vector fields."""
    points = (_point(0.0, 0.0, 3.0), _point(2.0, 2.0, 3.4))
    sample = sample_trajectory(points, 1.0)
    assert sample.reference_index == 0
    assert sample.point.position == Point3D(1.0, 2.0, 3.0)
    assert sample.point.velocity == Point3D(4.0, 5.0, 6.0)
    assert sample.point.acceleration == Point3D(7.0, 8.0, 9.0)
    assert sample.point.jerk == Point3D(10.0, 11.0, 12.0)
    assert sample.point.yaw_ned == pytest.approx(3.2)
    assert sample.point.yaw_rate_radps == pytest.approx(13.0)
    assert sample.point.yaw_acceleration_radps2 == pytest.approx(14.0)
    assert sample.point.arc_length_m == pytest.approx(15.0)
    assert sample.point.curvature_inverse_m == pytest.approx(16.0)


def test_invalid_timestamps_nonfinite_fields_and_time_are_rejected():
    """Sampling never crosses malformed timestamps or non-finite fields."""
    first = _point(0.0, 0.0, 0.0)
    second = _point(1.0, 1.0, 0.1)
    with pytest.raises(ValueError):
        sample_trajectory((first, replace(second, time_from_start_s=0.0)), 0.0)
    with pytest.raises(ValueError):
        sample_trajectory((first, replace(second, yaw_ned=math.nan)), 0.5)
    with pytest.raises(ValueError):
        sample_trajectory((first, second), math.inf)
