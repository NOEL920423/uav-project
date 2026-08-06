"""Planner-level B-spline selection and validated A* fallback regressions."""

from uav_navigation.astar_planner import plan_path
from uav_navigation.models import (
    BSplineConfig,
    CircularObstacle,
    PlannerConfig,
    Point3D,
)
from uav_navigation.path_validator import validate_path

ALTITUDE = -2.0


def point(x: float, y: float) -> Point3D:
    """Build a point at the fixed planning altitude."""
    return Point3D(x, y, ALTITUDE)


def tower(name: str, x: float, y: float) -> CircularObstacle:
    """Build a retained deterministic obstacle."""
    return CircularObstacle(name, Point3D(x, y, -1.5), 0.2, 3.0)


def assert_final_safe(result, obstacles, config=PlannerConfig()) -> None:
    """Independently validate the selected final result."""
    assert result.success, result.status
    validation = validate_path(
        result.final_path,
        obstacles,
        config,
        expected_start=result.raw_path[0],
        expected_goal=result.raw_path[-1],
    )
    assert validation.valid, validation.reason


def test_disabled_bspline_selects_validated_simplified_path() -> None:
    """Report explicit disabled state without changing overall A* success."""
    result = plan_path(
        point(-1, 0),
        point(1, 0),
        (),
        bspline_config=BSplineConfig(enable_bspline=False),
    )
    assert_final_safe(result, ())
    assert result.final_path == result.simplified_path
    assert result.final_path_source == "ASTAR_SIMPLIFIED"
    assert not result.bspline_enabled
    assert not result.bspline_valid
    assert result.bspline_rejection_reason == "disabled"


def test_safe_candidate_is_selected_with_comparison_metrics() -> None:
    """Select a validated open-space candidate and retain baseline metrics."""
    result = plan_path(point(-2, -0.4), point(2, 0.4), ())
    assert_final_safe(result, ())
    assert result.bspline_enabled
    assert result.bspline_valid
    assert result.bspline_selected
    assert result.final_path_source == "BSPLINE"
    assert result.final_path == result.bspline_candidate
    assert result.bspline_metrics is not None
    assert result.simplified_metrics is not None
    assert result.final_metrics.maximum_curvature_inverse_m >= 0.0


def test_rejected_candidate_falls_back_without_overall_failure() -> None:
    """Reject a conservatively limited curve and keep the validated A* path."""
    obstacle = tower("center", 0, 0)
    result = plan_path(
        point(-2, 0),
        point(2, 0),
        (obstacle,),
        bspline_config=BSplineConfig(bspline_maximum_curvature=1e-6),
    )
    assert_final_safe(result, (obstacle,))
    assert result.bspline_enabled
    assert not result.bspline_valid
    assert not result.bspline_selected
    assert result.bspline_candidate
    assert result.bspline_rejection_reason
    assert result.final_path == result.simplified_path
    assert result.final_path_source == "ASTAR_FALLBACK"


def test_no_valid_astar_baseline_remains_an_overall_failure() -> None:
    """Never use B-spline to bypass failure of the validated A* baseline."""
    wall = tuple(
        tower(f"wall-{index}", 0.0, y)
        for index, y in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0))
    )
    config = PlannerConfig(planning_bounds=(-2.0, 2.0, -1.0, 1.0))
    result = plan_path(point(-1, 0), point(1, 0), wall, config)
    assert not result.success
    assert not result.final_path
    assert result.final_path_source == "NONE"
    assert not result.bspline_selected


def test_repeated_full_pipeline_is_identical() -> None:
    """Lock deterministic candidate, metrics, rejection, and selection."""
    inputs = (point(-2, 0), point(2, 0), (tower("center", 0, 0),))
    first = plan_path(*inputs)
    second = plan_path(*inputs)
    assert first == second
