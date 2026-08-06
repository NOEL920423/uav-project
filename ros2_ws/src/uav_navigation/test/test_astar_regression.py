"""Deterministic regression tests for canonical A* and safe fallback."""

import math

from uav_navigation.astar_planner import plan_path
from uav_navigation.models import CircularObstacle, PlannerConfig, Point3D
from uav_navigation.path_simplifier import (
    select_validated_path,
    simplify_with_fallback,
)
from uav_navigation.path_validator import validate_path

ALTITUDE = -2.0


def point(x: float, y: float) -> Point3D:
    """Build a point at the canonical fixed planning altitude."""
    return Point3D(x, y, ALTITUDE)


def tower(
    name: str,
    x: float,
    y: float,
    radius: float = 0.2,
) -> CircularObstacle:
    """Build a retained obstacle whose top exceeds flight altitude."""
    return CircularObstacle(name, Point3D(x, y, -1.5), radius, 3.0)


def assert_safe_result(result, obstacles, config=PlannerConfig()) -> None:
    """Assert success, exact endpoints, and independent final validation."""
    assert result.success, result.status
    assert result.raw_path[0] == result.final_path[0]
    assert result.raw_path[-1] == result.final_path[-1]
    assert result.simplified_path[0] == result.raw_path[0]
    assert result.simplified_path[-1] == result.raw_path[-1]
    validation = validate_path(
        result.final_path,
        obstacles,
        config,
        expected_start=result.raw_path[0],
        expected_goal=result.raw_path[-1],
    )
    assert validation.valid, validation.reason


def test_no_obstacles_produces_direct_safe_final_path() -> None:
    """Simplify an unobstructed A* grid path to exact direct endpoints."""
    result = plan_path(point(-1, -0.25), point(1, 0.25), ())
    assert_safe_result(result, ())
    assert result.final_path[0] == point(-1, -0.25)
    assert result.final_path[-1] == point(1, 0.25)
    assert math.isclose(result.final_metrics.path_length_m, math.hypot(2, 0.5))
    assert result.simplification_method == "rdp"


def test_direct_blocker_is_continuously_avoided() -> None:
    """Route around one validation envelope without asserting fragile cells."""
    blocker = tower("blocker", 0.0, 0.0)
    result = plan_path(point(-2, 0), point(2, 0), (blocker,))
    assert_safe_result(result, (blocker,))
    clearance = result.final_metrics.minimum_physical_clearance_m
    assert clearance >= blocker.radius


def test_wide_gap_passes_and_narrow_gap_is_never_crossed() -> None:
    """Distinguish a valid corridor from overlapping validation envelopes."""
    wide = (tower("upper", 0, 0.75), tower("lower", 0, -0.75))
    wide_result = plan_path(point(-2, 0), point(2, 0), wide)
    assert_safe_result(wide_result, wide)
    assert all(item.y == 0.0 for item in wide_result.final_path)

    narrow = (tower("upper", 0, 0.5), tower("lower", 0, -0.5))
    narrow_result = plan_path(point(-2, 0), point(2, 0), narrow)
    assert_safe_result(narrow_result, narrow)
    assert any(item.y != 0.0 for item in narrow_result.final_path[1:-1])


def test_endpoint_policy_accepts_nearby_safe_and_rejects_forbidden() -> None:
    """Preserve a safe start but reject endpoints inside envelopes."""
    item = tower("origin", 0, 0)
    near = plan_path(point(0.59, 0), point(2, 0), (item,))
    assert_safe_result(near, (item,))
    assert near.final_path[0] == point(0.59, 0)

    bad_start = plan_path(point(0.50, 0), point(2, 0), (item,))
    assert not bad_start.success
    assert bad_start.status.startswith("invalid start")
    bad_goal = plan_path(point(-2, 0), point(-0.50, 0), (item,))
    assert not bad_goal.success
    assert bad_goal.status.startswith("invalid goal")


def test_short_obstacle_is_filtered_but_tall_one_is_avoided() -> None:
    """Exercise the canonical 2.5D top-height rule in full planning."""
    short = CircularObstacle("short", Point3D(0, 0, -0.25), 0.2, 0.5)
    short_result = plan_path(point(-2, 0), point(2, 0), (short,))
    assert short_result.success
    assert all(item.y == 0.0 for item in short_result.final_path)

    tall = tower("tall", 0, 0)
    tall_result = plan_path(point(-2, 0), point(2, 0), (tall,))
    assert_safe_result(tall_result, (tall,))
    assert tall_result.final_path != (point(-2, 0), point(2, 0))


def test_complete_barrier_with_explicit_bounds_has_no_path() -> None:
    """Return a structured failure for a wall spanning bounded free space."""
    wall = tuple(
        tower(f"wall-{index}", 0.0, y)
        for index, y in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0))
    )
    config = PlannerConfig(planning_bounds=(-2.0, 2.0, -1.0, 1.0))
    result = plan_path(point(-1, 0), point(1, 0), wall, config)
    assert not result.success
    assert result.status.startswith("no path")
    assert dict(result.diagnostics)["attempts"] == "2"


def test_unsafe_rdp_uses_greedy_then_raw_selector_can_fallback() -> None:
    """Reject unsafe shortcuts and demonstrate both safe fallback stages."""
    item = tower("center", 0, 0, radius=0.3)
    raw = (
        point(-2, 0),
        point(-1, 1),
        point(0, 1),
        point(1, 1),
        point(2, 0),
    )
    config = PlannerConfig(
        simplification_tolerance_m=10.0,
        maximum_waypoint_spacing_m=2.0,
    )
    simplified = simplify_with_fallback(raw, (item,), config)
    assert simplified.validation.valid
    assert simplified.method == "greedy"
    assert "rdp:" in simplified.fallback_reason

    selected = select_validated_path(
        (
            ("rdp", (raw[0], raw[-1])),
            ("greedy", (raw[0], raw[-1])),
            ("raw", raw),
        ),
        (item,),
        config,
        raw[0],
        raw[-1],
    )
    assert selected.validation.valid
    assert selected.method == "raw"
    assert "rdp:" in selected.fallback_reason
    assert "greedy:" in selected.fallback_reason


def test_identical_inputs_produce_identical_results() -> None:
    """Lock deterministic equality without overspecifying route cells."""
    inputs = (point(-2, -0.2), point(2, 0.2), (tower("block", 0, 0),))
    first = plan_path(*inputs)
    second = plan_path(*inputs)
    assert first == second
