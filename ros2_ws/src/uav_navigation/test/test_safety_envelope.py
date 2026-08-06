"""Tests for physical, planning, validation, and overflight contracts."""

import math

import pytest

from uav_navigation.models import CircularObstacle, PlannerConfig, Point3D
from uav_navigation.path_validator import (
    filter_overflyable_obstacles,
    minimum_segment_clearance,
    nearest_obstacle_clearance,
    planning_radius,
    point_clearance,
    segment_clearance,
    validate_path,
    validation_radius,
)


def obstacle(
    name: str = "tower",
    *,
    center: Point3D = Point3D(0.0, 0.0, -1.0),
    radius: float = 0.2,
    height: float = 4.0,
) -> CircularObstacle:
    """Build a deterministic obstacle fixture."""
    return CircularObstacle(name, center, radius, height)


def test_default_safety_radii_are_not_conflated() -> None:
    """Lock the physical planning and stricter validation formulas."""
    config = PlannerConfig()
    item = obstacle()
    assert planning_radius(item, config) == pytest.approx(0.51)
    assert validation_radius(item, config) == pytest.approx(0.58)
    grid_reserve = 0.5 * math.sqrt(2.0) * config.grid_resolution_m
    assert validation_radius(item, config) != pytest.approx(
        planning_radius(item, config) + grid_reserve
    )


@pytest.mark.parametrize(
    ("y", "expected_sign"),
    ((0.70, 1), (0.58, 0), (0.40, -1)),
)
def test_signed_continuous_segment_clearance(
    y: float,
    expected_sign: int,
) -> None:
    """Check outside, tangent, and inside signed clearances."""
    item = obstacle()
    clearance = minimum_segment_clearance(
        Point3D(-1.0, y, -2.0),
        Point3D(1.0, y, -2.0),
        (item,),
        PlannerConfig(),
    )
    if expected_sign == 0:
        assert clearance == pytest.approx(0.0, abs=1e-12)
    else:
        assert math.copysign(1.0, clearance) == expected_sign


def test_point_segment_and_nearest_clearance_helpers() -> None:
    """Expose each required clearance boundary with the same signed formula."""
    item = obstacle()
    config = PlannerConfig()
    tangent = Point3D(0.0, 0.58, -2.0)
    assert point_clearance(tangent, item, config) == pytest.approx(0.0)
    assert segment_clearance(
        Point3D(-1.0, 0.58, -2.0),
        Point3D(1.0, 0.58, -2.0),
        item,
        config,
    ) == pytest.approx(0.0)
    clearance = nearest_obstacle_clearance(tangent, (item,), config)
    assert clearance == pytest.approx(0.0)
    assert math.isinf(nearest_obstacle_clearance(tangent, (), config))


def test_validator_names_colliding_segment_and_obstacle() -> None:
    """Return a structured collision reason for an unsafe shortcut."""
    result = validate_path(
        (Point3D(-1, 0, -2), Point3D(1, 0, -2)),
        (obstacle(),),
        PlannerConfig(maximum_waypoint_spacing_m=3.0),
    )
    assert not result.valid
    assert "segment 0" in result.reason
    assert "tower" in result.reason


def test_overflight_threshold_and_disabled_mode() -> None:
    """Filter clearly short and exact-threshold obstacles only when enabled."""
    short = obstacle(
        "short",
        center=Point3D(0.0, 0.0, -0.25),
        height=0.5,
    )
    threshold = obstacle(
        "threshold",
        center=Point3D(0.0, 0.0, -1.15),
        height=1.0,
    )
    tall = obstacle(
        "tall",
        center=Point3D(0.0, 0.0, -1.151),
        height=1.0,
    )
    config = PlannerConfig()
    assert filter_overflyable_obstacles((short,), config) == ()
    assert filter_overflyable_obstacles((threshold,), config) == ()
    assert filter_overflyable_obstacles((tall,), config) == (tall,)
    disabled = PlannerConfig(enable_overfly_short_obstacles=False)
    assert filter_overflyable_obstacles((short,), disabled) == (short,)


def test_invalid_obstacle_height_is_rejected() -> None:
    """Reject invalid obstacle heights at the typed boundary."""
    with pytest.raises(ValueError):
        obstacle(height=-0.1)
    with pytest.raises(ValueError):
        obstacle(height=math.nan)
