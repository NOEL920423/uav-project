"""Tests for pure geometry and geometric metrics."""

import math

import pytest

from uav_navigation.geometry import (
    absolute_heading_changes,
    clamp,
    distance_2d,
    point_to_segment_distance_2d,
    polyline_length_2d,
)
from uav_navigation.models import CircularObstacle, Point3D
from uav_navigation.path_metrics import calculate_path_metrics


def point(x: float, y: float, z: float = -2.0) -> Point3D:
    """Build a concise fixture point."""
    return Point3D(x, y, z)


def test_distance_projection_and_clamp() -> None:
    """Check exact elementary geometry including a degenerate segment."""
    assert distance_2d(point(0, 0), point(3, 4)) == 5.0
    assert point_to_segment_distance_2d(
        point(1, 1), point(0, 0), point(2, 0)
    ) == 1.0
    assert point_to_segment_distance_2d(
        point(3, 4), point(0, 0), point(0, 0)
    ) == 5.0
    assert clamp(2.0, 0.0, 1.0) == 1.0
    with pytest.raises(ValueError):
        clamp(math.inf, 0.0, 1.0)


def test_polyline_heading_changes_and_metrics() -> None:
    """Verify metric meanings on a right-angle polyline."""
    path = (point(0, 0), point(1, 0), point(1, 1))
    obstacle = CircularObstacle("far", point(4, 4), 0.5, 4.0)
    assert polyline_length_2d(path) == 2.0
    assert absolute_heading_changes(path) == (math.pi / 2.0,)
    metrics = calculate_path_metrics(path, (obstacle,))
    assert metrics.point_count == 3
    assert metrics.path_length_m == 2.0
    assert metrics.mean_segment_length_m == 1.0
    assert metrics.maximum_segment_length_m == 1.0
    assert metrics.mean_absolute_heading_change_rad == math.pi / 2.0
    assert metrics.heading_change_variance_rad2 == 0.0


def test_metrics_for_empty_obstacle_set() -> None:
    """Use infinity to represent unbounded physical obstacle clearance."""
    metrics = calculate_path_metrics((point(0, 0), point(1, 0)), ())
    assert math.isinf(metrics.minimum_physical_clearance_m)
