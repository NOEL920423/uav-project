"""Deterministic pure-Python B-spline generation and independent validation."""

import bisect
import math
from collections.abc import Sequence
from dataclasses import replace

from uav_navigation.geometry import (
    cumulative_polyline_lengths_2d,
    discrete_curvatures_2d,
    distance_2d,
    interpolate_segment,
    polyline_length_2d,
    polyline_self_intersection_2d,
)
from uav_navigation.models import (
    BSplineConfig,
    BSplineResult,
    CircularObstacle,
    PlannerConfig,
    Point3D,
    ValidationResult,
)
from uav_navigation.path_metrics import calculate_path_metrics
from uav_navigation.path_validator import validate_path


def open_uniform_knots(
    control_point_count: int,
    degree: int,
) -> tuple[float, ...]:
    """Return a clamped open-uniform knot vector on the unit interval."""
    if not isinstance(control_point_count, int) or control_point_count < 2:
        raise ValueError(
            "control_point_count must be an integer of at least two"
        )
    if not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer")
    if degree >= control_point_count:
        raise ValueError("degree must be below the control point count")
    interior_count = control_point_count - degree - 1
    knots = [0.0] * (degree + 1)
    denominator = interior_count + 1
    knots.extend(
        index / denominator for index in range(1, interior_count + 1)
    )
    knots.extend([1.0] * (degree + 1))
    return tuple(knots)


def _validate_knots(
    knots: Sequence[float],
    control_point_count: int,
    degree: int,
) -> tuple[float, ...]:
    """Normalize and validate a knot vector for basis or De Boor evaluation."""
    normalized = tuple(float(value) for value in knots)
    expected_count = control_point_count + degree + 1
    if len(normalized) != expected_count:
        raise ValueError(
            f"knot vector requires exactly {expected_count} values"
        )
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("knot values must be finite")
    if any(
        first > second for first, second in zip(normalized, normalized[1:])
    ):
        raise ValueError("knot vector must be nondecreasing")
    if normalized[degree] >= normalized[control_point_count]:
        raise ValueError("knot vector has an empty evaluation domain")
    return normalized


def basis_values(
    parameter: float,
    knots: Sequence[float],
    degree: int,
    control_point_count: int,
) -> tuple[float, ...]:
    """Evaluate all Cox-De Boor basis values at one finite parameter."""
    if not isinstance(degree, int) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if degree >= control_point_count:
        raise ValueError("degree must be below the control point count")
    normalized = _validate_knots(knots, control_point_count, degree)
    value = float(parameter)
    lower = normalized[degree]
    upper = normalized[control_point_count]
    if not math.isfinite(value) or value < lower or value > upper:
        raise ValueError("parameter must lie in the finite knot domain")
    if value == upper:
        return (0.0,) * (control_point_count - 1) + (1.0,)

    memo: dict[tuple[int, int], float] = {}

    def evaluate(index: int, order: int) -> float:
        key = index, order
        if key in memo:
            return memo[key]
        if order == 0:
            result = float(normalized[index] <= value < normalized[index + 1])
        else:
            left_denominator = normalized[index + order] - normalized[index]
            right_denominator = (
                normalized[index + order + 1] - normalized[index + 1]
            )
            left = 0.0
            right = 0.0
            if left_denominator > 0.0:
                left = (
                    (value - normalized[index])
                    / left_denominator
                    * evaluate(index, order - 1)
                )
            if right_denominator > 0.0:
                right = (
                    (normalized[index + order + 1] - value)
                    / right_denominator
                    * evaluate(index + 1, order - 1)
                )
            result = left + right
        memo[key] = result
        return result

    return tuple(
        evaluate(index, degree) for index in range(control_point_count)
    )


def de_boor_evaluate(
    control_points: Sequence[Point3D],
    degree: int,
    parameter: float,
    knots: Sequence[float] | None = None,
) -> Point3D:
    """Evaluate one B-spline point with stable iterative De Boor recursion."""
    controls = tuple(control_points)
    if len(controls) < 2:
        raise ValueError("at least two control points are required")
    if not isinstance(degree, int) or degree < 1 or degree >= len(controls):
        raise ValueError(
            "degree must be positive and below control point count"
        )
    normalized = _validate_knots(
        knots or open_uniform_knots(len(controls), degree),
        len(controls),
        degree,
    )
    value = float(parameter)
    lower = normalized[degree]
    upper = normalized[len(controls)]
    if not math.isfinite(value) or value < lower or value > upper:
        raise ValueError("parameter must lie in the finite knot domain")
    if value == upper:
        return controls[-1]
    span = min(
        len(controls) - 1,
        max(degree, bisect.bisect_right(normalized, value) - 1),
    )
    working = [controls[span - degree + index] for index in range(degree + 1)]
    for recursion in range(1, degree + 1):
        for index in range(degree, recursion - 1, -1):
            knot_index = span - degree + index
            denominator = (
                normalized[knot_index + degree - recursion + 1]
                - normalized[knot_index]
            )
            alpha = (
                0.0
                if denominator == 0.0
                else (value - normalized[knot_index]) / denominator
            )
            previous = working[index - 1]
            current = working[index]
            working[index] = Point3D(
                (1.0 - alpha) * previous.x + alpha * current.x,
                (1.0 - alpha) * previous.y + alpha * current.y,
                (1.0 - alpha) * previous.z + alpha * current.z,
            )
    return working[degree]


def remove_adjacent_duplicates(
    path: Sequence[Point3D],
    tolerance: float,
) -> tuple[Point3D, ...]:
    """Remove adjacent controls coincident within a finite tolerance."""
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("duplicate tolerance must be finite and nonnegative")
    if not path:
        return ()
    controls = [path[0]]
    for point in path[1:]:
        if not point.almost_equals(controls[-1], tolerance):
            controls.append(point)
    return tuple(controls)


def uniform_arc_length_resample(
    path: Sequence[Point3D],
    spacing_m: float,
    minimum_samples: int,
    maximum_samples: int,
    tolerance: float,
) -> tuple[Point3D, ...]:
    """Resample a finite polyline approximately uniformly by 2D arc length."""
    if not math.isfinite(spacing_m) or spacing_m <= 0.0:
        raise ValueError("resampling spacing must be finite and positive")
    if minimum_samples < 2 or maximum_samples < minimum_samples:
        raise ValueError("invalid resampling sample limits")
    clean = remove_adjacent_duplicates(path, tolerance)
    if len(clean) < 2:
        raise ValueError("resampling requires two distinct points")
    cumulative = cumulative_polyline_lengths_2d(clean)
    if not all(math.isfinite(value) for value in cumulative):
        raise ValueError("cumulative arc length must remain finite")
    if any(
        first >= second for first, second in zip(cumulative, cumulative[1:])
    ):
        raise ValueError("cumulative arc length must increase monotonically")
    total_length = cumulative[-1]
    if total_length <= tolerance:
        raise ValueError("resampling path has zero total length")
    spacing_count = math.ceil(total_length / spacing_m) + 1
    sample_count = max(minimum_samples, spacing_count)
    if sample_count > maximum_samples:
        raise ValueError(
            "maximum samples cannot satisfy configured spatial spacing"
        )
    targets = (
        total_length * index / (sample_count - 1)
        for index in range(sample_count)
    )
    result = []
    for target in targets:
        if target <= 0.0:
            result.append(clean[0])
            continue
        if target >= total_length:
            result.append(clean[-1])
            continue
        upper_index = bisect.bisect_left(cumulative, target)
        lower_index = upper_index - 1
        segment_length = cumulative[upper_index] - cumulative[lower_index]
        ratio = (target - cumulative[lower_index]) / segment_length
        result.append(
            interpolate_segment(clean[lower_index], clean[upper_index], ratio)
        )
    result[0] = clean[0]
    result[-1] = clean[-1]
    if any(
        distance_2d(first, second) > spacing_m + tolerance
        for first, second in zip(result, result[1:])
    ):
        raise ValueError("resampled path exceeds configured spatial spacing")
    return tuple(result)


def _validation_config(
    planner_config: PlannerConfig,
    bspline_config: BSplineConfig,
) -> PlannerConfig:
    """Return stricter config for independent candidate validation."""
    clearance = max(
        planner_config.minimum_segment_clearance_m,
        bspline_config.bspline_minimum_clearance_m,
    )
    bounds = planner_config.planning_bounds
    if bounds is not None and bspline_config.bspline_allowed_bounds_margin_m:
        margin = bspline_config.bspline_allowed_bounds_margin_m
        xmin, xmax, ymin, ymax = bounds
        bounds = xmin + margin, xmax - margin, ymin + margin, ymax - margin
        if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
            raise ValueError("B-spline bounds margin leaves no allowed region")
    return replace(
        planner_config,
        minimum_segment_clearance_m=clearance,
        maximum_waypoint_spacing_m=bspline_config.bspline_sample_spacing_m,
        planning_bounds=bounds,
    )


def validate_bspline_candidate(
    candidate: Sequence[Point3D],
    obstacles: Sequence[CircularObstacle],
    planner_config: PlannerConfig,
    bspline_config: BSplineConfig,
    expected_start: Point3D,
    expected_goal: Point3D,
) -> tuple[ValidationResult, float, tuple[int, int] | None]:
    """Apply continuous, curvature, and optional crossing validation."""
    if len(candidate) < bspline_config.bspline_minimum_samples:
        result = ValidationResult(
            False,
            "candidate has fewer than bspline_minimum_samples",
            math.inf,
            0.0,
        )
        return result, 0.0, None
    if len(candidate) > bspline_config.bspline_maximum_samples:
        result = ValidationResult(
            False,
            "candidate exceeds bspline_maximum_samples",
            math.inf,
            0.0,
        )
        return result, 0.0, None
    strict_config = _validation_config(planner_config, bspline_config)
    validation = validate_path(
        candidate,
        obstacles,
        strict_config,
        expected_start=expected_start,
        expected_goal=expected_goal,
    )
    if not validation.valid:
        return validation, 0.0, None
    curvatures = discrete_curvatures_2d(
        candidate,
        planner_config.numerical_tolerance,
    )
    maximum_curvature = max(curvatures, default=0.0)
    if maximum_curvature > (
        bspline_config.bspline_maximum_curvature
        + planner_config.numerical_tolerance
    ):
        result = ValidationResult(
            False,
            "candidate exceeds maximum curvature",
            validation.minimum_validation_clearance_m,
            validation.maximum_segment_length_m,
        )
        return result, maximum_curvature, None
    intersection = None
    if bspline_config.bspline_reject_self_intersection:
        intersection = polyline_self_intersection_2d(
            candidate,
            planner_config.numerical_tolerance,
        )
        if intersection is not None:
            result = ValidationResult(
                False,
                "candidate self-intersection between segments "
                f"{intersection[0]} and {intersection[1]}",
                validation.minimum_validation_clearance_m,
                validation.maximum_segment_length_m,
            )
            return result, maximum_curvature, intersection
    return validation, maximum_curvature, intersection


def generate_bspline_candidate(
    baseline_path: Sequence[Point3D],
    obstacles: Sequence[CircularObstacle],
    planner_config: PlannerConfig,
    bspline_config: BSplineConfig,
) -> BSplineResult:
    """Generate, spatially resample, and validate one candidate."""
    if not bspline_config.enable_bspline:
        return BSplineResult(
            status_message="B-spline disabled; validated A* selected",
            rejection_reason="disabled",
        )
    controls: tuple[Point3D, ...] = ()
    candidate: tuple[Point3D, ...] = ()
    effective_degree = 0
    provisional_count = 0
    try:
        controls = remove_adjacent_duplicates(
            baseline_path,
            planner_config.numerical_tolerance,
        )
        if len(controls) < 2:
            raise ValueError("fewer than two distinct B-spline control points")
        effective_degree = min(
            bspline_config.bspline_degree,
            len(controls) - 1,
        )
        control_length = polyline_length_2d(controls)
        desired_provisional = max(
            len(controls) * 16,
            bspline_config.bspline_minimum_samples * 4,
            math.ceil(control_length / bspline_config.bspline_sample_spacing_m)
            * 4
            + 1,
        )
        provisional_limit = max(
            bspline_config.bspline_maximum_samples * 4,
            len(controls) * 16,
        )
        provisional_count = min(desired_provisional, provisional_limit)
        knots = open_uniform_knots(len(controls), effective_degree)
        provisional = tuple(
            de_boor_evaluate(
                controls,
                effective_degree,
                index / (provisional_count - 1),
                knots,
            )
            for index in range(provisional_count)
        )
        candidate = uniform_arc_length_resample(
            provisional,
            bspline_config.bspline_sample_spacing_m,
            bspline_config.bspline_minimum_samples,
            bspline_config.bspline_maximum_samples,
            planner_config.numerical_tolerance,
        )
        candidate = (controls[0],) + candidate[1:-1] + (controls[-1],)
        validation, maximum_curvature, intersection = (
            validate_bspline_candidate(
                candidate,
                obstacles,
                planner_config,
                bspline_config,
                controls[0],
                controls[-1],
            )
        )
        metrics = calculate_path_metrics(candidate, obstacles)
        if not validation.valid:
            return BSplineResult(
                candidate_path=candidate,
                status_message=(
                    "B-spline rejected; validated A* fallback selected"
                ),
                rejection_reason=validation.reason,
                effective_degree=effective_degree,
                control_point_count=len(controls),
                provisional_sample_count=provisional_count,
                final_sample_count=len(candidate),
                minimum_clearance_m=(
                    validation.minimum_validation_clearance_m
                ),
                maximum_curvature_inverse_m=maximum_curvature,
                self_intersection=intersection is not None,
                metrics=metrics,
            )
        return BSplineResult(
            candidate_path=candidate,
            valid=True,
            selected=True,
            status_message="B-spline accepted and selected",
            effective_degree=effective_degree,
            control_point_count=len(controls),
            provisional_sample_count=provisional_count,
            final_sample_count=len(candidate),
            minimum_clearance_m=validation.minimum_validation_clearance_m,
            maximum_curvature_inverse_m=maximum_curvature,
            metrics=metrics,
        )
    except (TypeError, ValueError) as error:
        return BSplineResult(
            candidate_path=candidate,
            status_message="B-spline generation failed; A* fallback required",
            rejection_reason=str(error),
            effective_degree=effective_degree,
            control_point_count=len(controls),
            provisional_sample_count=provisional_count,
            final_sample_count=len(candidate),
        )
