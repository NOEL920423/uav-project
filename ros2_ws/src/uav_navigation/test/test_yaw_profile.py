"""NED yaw and angular derivative tests."""

import math

import pytest

from uav_navigation.models import Point3D
from uav_navigation.trajectory_parameterizer import parameterize_trajectory
from uav_navigation.yaw_profile import ned_yaw_profile, unwrap_angles


def test_cardinal_ned_headings():
    """North/east/south/west follow atan2(east, north)."""
    origin = Point3D(0.0, 0.0, 0.0)
    headings = (
        ned_yaw_profile((origin, Point3D(1.0, 0.0, 0.0)))[0],
        ned_yaw_profile((origin, Point3D(0.0, 1.0, 0.0)))[0],
        ned_yaw_profile((origin, Point3D(-1.0, 0.0, 0.0)))[0],
        ned_yaw_profile((origin, Point3D(0.0, -1.0, 0.0)))[0],
    )
    assert headings[0] == pytest.approx(0.0)
    assert headings[1] == pytest.approx(math.pi / 2.0)
    assert abs(headings[2]) == pytest.approx(math.pi)
    assert headings[3] == pytest.approx(-math.pi / 2.0)


def test_unwrap_crosses_pi_continuously():
    """A pi-boundary transition does not create a two-pi jump."""
    angles = unwrap_angles((math.pi - 0.1, -math.pi + 0.1))
    assert angles[1] - angles[0] == pytest.approx(0.2)


def test_yaw_derivatives_are_finite_and_scaled():
    """Angular derivatives remain finite and participate in time scaling."""
    fixture = (
        Point3D(0.0, 0.01, -2.0),
        Point3D(-1.0, 0.001, -2.0),
        Point3D(-2.0, -0.001, -2.0),
        Point3D(-3.0, -0.01, -2.0),
    )
    result = parameterize_trajectory(fixture)
    assert result.valid
    assert all(
        math.isfinite(value)
        for point in result.trajectory_points
        for value in (point.yaw_rate_radps, point.yaw_acceleration_radps2)
    )
    assert all(
        abs(second.yaw_ned - first.yaw_ned) < math.pi
        for first, second in zip(
            result.trajectory_points, result.trajectory_points[1:]
        )
    )
