"""Deterministic regression tests for pure B-spline geometry and validation."""

import math
from types import SimpleNamespace

import pytest

from uav_navigation.bspline_smoother import (
    basis_values,
    de_boor_evaluate,
    generate_bspline_candidate,
    open_uniform_knots,
    uniform_arc_length_resample,
    validate_bspline_candidate,
)
from uav_navigation.geometry import (
    discrete_curvatures_2d,
    distance_2d,
    polyline_self_intersection_2d,
)
from uav_navigation.models import (
    BSplineConfig,
    CircularObstacle,
    PlannerConfig,
    Point3D,
)

ALTITUDE = -2.0


def point(x: float, y: float) -> Point3D:
    """Build a finite point at the canonical planning altitude."""
    return Point3D(x, y, ALTITUDE)


def test_open_uniform_basis_partition_and_endpoints() -> None:
    """Verify partition of unity and both clamped endpoint evaluations."""
    controls = (point(0, 0), point(1, 1), point(2, 1), point(3, 0))
    knots = open_uniform_knots(len(controls), 3)
    for parameter in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
        values = basis_values(parameter, knots, 3, len(controls))
        assert math.isclose(sum(values), 1.0, abs_tol=1e-12)
        assert all(value >= 0.0 for value in values)
    assert de_boor_evaluate(controls, 3, 0.0, knots) == controls[0]
    assert de_boor_evaluate(controls, 3, 1.0, knots) == controls[-1]


def test_de_boor_is_deterministic_and_rejects_invalid_inputs() -> None:
    """Repeat evaluation exactly and reject invalid degree or knot vectors."""
    controls = (point(0, 0), point(1, 1), point(2, 0))
    first = de_boor_evaluate(controls, 2, 0.375)
    second = de_boor_evaluate(controls, 2, 0.375)
    assert first == second
    assert all(math.isfinite(value) for value in (first.x, first.y, first.z))
    with pytest.raises(ValueError, match="degree"):
        de_boor_evaluate(controls, 3, 0.5)
    with pytest.raises(ValueError, match="knot vector"):
        de_boor_evaluate(controls, 2, 0.5, (0.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="nondecreasing"):
        basis_values(0.5, (0, 0, 0, 0.7, 0.6, 1, 1), 2, 4)


def test_short_paths_reduce_degree_and_preserve_exact_endpoints() -> None:
    """Handle two, three, and four controls with exact endpoints."""
    planner = PlannerConfig()
    config = BSplineConfig()
    for controls, degree in (
        ((point(0, 0), point(1, 0)), 1),
        ((point(0, 0), point(0.5, 0.4), point(1, 0)), 2),
        (
            (
                point(0, 0),
                point(0.3, 0.2),
                point(0.7, 0.2),
                point(1, 0),
            ),
            3,
        ),
    ):
        result = generate_bspline_candidate(controls, (), planner, config)
        assert result.valid, result.rejection_reason
        assert result.effective_degree == degree
        assert result.candidate_path[0] == controls[0]
        assert result.candidate_path[-1] == controls[-1]
        assert all(item.z == ALTITUDE for item in result.candidate_path)


def test_duplicate_controls_are_removed_and_single_point_fails() -> None:
    """Remove adjacent duplicates but fail when fewer than two remain."""
    repeated = (
        point(0, 0),
        point(0, 0),
        point(0.5, 0.2),
        point(1, 0),
        point(1, 0),
    )
    result = generate_bspline_candidate(
        repeated,
        (),
        PlannerConfig(),
        BSplineConfig(),
    )
    assert result.valid
    assert result.control_point_count == 3
    failure = generate_bspline_candidate(
        (point(0, 0),),
        (),
        PlannerConfig(),
        BSplineConfig(),
    )
    assert not failure.valid
    assert "fewer than two" in failure.rejection_reason


def test_arc_length_resampling_is_uniform_bounded_and_exact() -> None:
    """Resample by distance with monotonic progress and exact endpoints."""
    source = (point(0, 0), point(0.3, 0), point(1, 0))
    result = uniform_arc_length_resample(source, 0.2, 2, 20, 1e-9)
    lengths = tuple(
        distance_2d(first, second)
        for first, second in zip(result, result[1:])
    )
    assert result[0] == source[0]
    assert result[-1] == source[-1]
    assert max(lengths) <= 0.2 + 1e-9
    assert max(lengths) - min(lengths) <= 1e-9
    assert 2 <= len(result) <= 20
    with pytest.raises(ValueError, match="maximum samples"):
        uniform_arc_length_resample(source, 0.01, 2, 10, 1e-9)


def test_candidate_is_deterministic_spatially_bounded_and_finite() -> None:
    """Lock repeated equality without fixing the full sample sequence."""
    controls = (
        point(-2, 0),
        point(-1, 0.8),
        point(0, 1.0),
        point(1, 0.8),
        point(2, 0),
    )
    first = generate_bspline_candidate(
        controls,
        (),
        PlannerConfig(maximum_waypoint_spacing_m=2.0),
        BSplineConfig(),
    )
    second = generate_bspline_candidate(
        controls,
        (),
        PlannerConfig(maximum_waypoint_spacing_m=2.0),
        BSplineConfig(),
    )
    assert first == second
    assert first.valid
    assert all(
        distance_2d(start, end) <= 0.08 + 1e-9
        for start, end in zip(first.candidate_path, first.candidate_path[1:])
    )


def test_continuous_obstacle_clearance_and_stricter_margin_reject() -> None:
    """Reject both direct collision and a curve below stricter clearance."""
    candidate = tuple(point(-2 + index * 0.25, 0) for index in range(17))
    blocker = CircularObstacle("block", Point3D(0, 0, -1.5), 0.2, 3.0)
    planner = PlannerConfig(maximum_waypoint_spacing_m=0.5)
    config = BSplineConfig(
        bspline_sample_spacing_m=0.5,
        bspline_minimum_samples=2,
    )
    validation, _, _ = validate_bspline_candidate(
        candidate,
        (blocker,),
        planner,
        config,
        candidate[0],
        candidate[-1],
    )
    assert not validation.valid
    assert "intersects obstacle block" in validation.reason

    offset = tuple(point(-2 + index * 0.25, 0.65) for index in range(17))
    stricter = BSplineConfig(
        bspline_sample_spacing_m=0.5,
        bspline_minimum_samples=2,
        bspline_minimum_clearance_m=0.20,
    )
    validation, _, _ = validate_bspline_candidate(
        offset,
        (blocker,),
        planner,
        stricter,
        offset[0],
        offset[-1],
    )
    assert not validation.valid


def test_nonfinite_bounds_and_zero_length_candidates_reject() -> None:
    """Exercise independent finite, bounds, and zero-length diagnostics."""
    planner = PlannerConfig(
        maximum_waypoint_spacing_m=2.0,
        planning_bounds=(-2, 2, -2, 2),
    )
    config = BSplineConfig(
        bspline_sample_spacing_m=2.0,
        bspline_minimum_samples=2,
        bspline_allowed_bounds_margin_m=0.2,
    )
    outside = (point(-1.9, 0), point(1.9, 0))
    validation, _, _ = validate_bspline_candidate(
        outside, (), planner, config, outside[0], outside[-1]
    )
    assert not validation.valid
    assert "outside planning bounds" in validation.reason

    nonfinite = (
        point(0, 0),
        SimpleNamespace(x=math.nan, y=0.0, z=ALTITUDE),
    )
    validation, _, _ = validate_bspline_candidate(
        nonfinite, (), planner, config, nonfinite[0], point(1, 0)
    )
    assert not validation.valid
    assert "non-finite" in validation.reason

    duplicate = (point(0, 0), point(0, 0))
    validation, _, _ = validate_bspline_candidate(
        duplicate, (), planner, config, duplicate[0], duplicate[-1]
    )
    assert not validation.valid
    assert "zero length" in validation.reason


def test_curvature_estimator_and_limit() -> None:
    """Check straight/circular geometry and reject a sharp turn."""
    straight = (point(0, 0), point(1, 0), point(2, 0))
    assert discrete_curvatures_2d(straight) == (0.0,)
    radius = 2.0
    arc = tuple(
        point(radius * math.cos(angle), radius * math.sin(angle))
        for angle in (0.0, math.pi / 4.0, math.pi / 2.0)
    )
    assert math.isclose(discrete_curvatures_2d(arc)[0], 0.5, rel_tol=1e-9)

    corner = (point(0, 0), point(0.1, 0), point(0.1, 0.1))
    config = BSplineConfig(
        bspline_sample_spacing_m=0.2,
        bspline_minimum_samples=2,
        bspline_maximum_curvature=1.0,
    )
    validation, maximum, _ = validate_bspline_candidate(
        corner,
        (),
        PlannerConfig(maximum_waypoint_spacing_m=0.2),
        config,
        corner[0],
        corner[-1],
    )
    assert not validation.valid
    assert "maximum curvature" in validation.reason
    assert maximum > 1.0


def test_self_intersection_detects_nonadjacent_crossing() -> None:
    """Identify crossing segment indices without rejecting adjacent joints."""
    crossing = (point(0, 0), point(1, 1), point(0, 1), point(1, 0))
    assert polyline_self_intersection_2d(crossing) == (0, 2)
    assert polyline_self_intersection_2d(
        (point(0, 0), point(1, 0), point(1, 1))
    ) is None
    config = BSplineConfig(
        bspline_sample_spacing_m=2.0,
        bspline_minimum_samples=2,
        bspline_maximum_curvature=100.0,
    )
    validation, _, intersection = validate_bspline_candidate(
        crossing,
        (),
        PlannerConfig(maximum_waypoint_spacing_m=2.0),
        config,
        crossing[0],
        crossing[-1],
    )
    assert not validation.valid
    assert intersection == (0, 2)
    assert "segments 0 and 2" in validation.reason


@pytest.mark.parametrize(
    "kwargs, message",
    (
        ({"bspline_degree": 0}, "degree"),
        ({"bspline_sample_spacing_m": 0.0}, "spacing"),
        ({"bspline_minimum_samples": 1}, "minimum"),
        (
            {"bspline_minimum_samples": 10, "bspline_maximum_samples": 5},
            "maximum",
        ),
        ({"bspline_minimum_clearance_m": -0.1}, "clearance"),
        ({"bspline_maximum_curvature": 0.0}, "curvature"),
        ({"bspline_preserve_endpoints": False}, "preserve"),
        ({"bspline_control_point_strategy": "unknown"}, "strategy"),
    ),
)
def test_invalid_configuration_is_rejected(kwargs, message) -> None:
    """Reject every unsafe startup-parameter class required by the contract."""
    with pytest.raises(ValueError, match=message):
        BSplineConfig(**kwargs)
