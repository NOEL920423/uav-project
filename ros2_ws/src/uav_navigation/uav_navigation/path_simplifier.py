"""Safe path simplification with deterministic fallback."""

import math
from collections.abc import Callable, Sequence

from uav_navigation.geometry import (
    distance_2d,
    interpolate_segment,
    point_to_segment_distance_2d,
)
from uav_navigation.models import (
    CircularObstacle,
    PlannerConfig,
    Point3D,
    SimplificationResult,
)
from uav_navigation.path_validator import validate_path


def rdp_simplify(
    path: Sequence[Point3D],
    tolerance: float,
) -> tuple[Point3D, ...]:
    """Simplify a path with deterministic Ramer-Douglas-Peucker recursion."""
    if tolerance < 0.0:
        raise ValueError("RDP tolerance must be nonnegative")
    if len(path) <= 2:
        return tuple(path)
    maximum_distance = -1.0
    split_index = 0
    for index in range(1, len(path) - 1):
        distance = point_to_segment_distance_2d(path[index], path[0], path[-1])
        if distance > maximum_distance:
            maximum_distance = distance
            split_index = index
    if maximum_distance > tolerance:
        left = rdp_simplify(path[: split_index + 1], tolerance)
        right = rdp_simplify(path[split_index:], tolerance)
        return left[:-1] + right
    return path[0], path[-1]


def densify_path(
    path: Sequence[Point3D],
    maximum_spacing: float,
) -> tuple[Point3D, ...]:
    """Insert linearly spaced points so every XY segment is bounded."""
    if maximum_spacing <= 0.0:
        raise ValueError("maximum spacing must be positive")
    if not path:
        return ()
    dense = [path[0]]
    for start, end in zip(path, path[1:]):
        length = distance_2d(start, end)
        subdivisions = max(1, math.ceil(length / maximum_spacing))
        for step in range(1, subdivisions + 1):
            dense.append(interpolate_segment(start, end, step / subdivisions))
    return tuple(dense)


def greedy_safe_simplify(
    path: Sequence[Point3D],
    segment_is_safe: Callable[[Point3D, Point3D], bool],
) -> tuple[Point3D, ...]:
    """Take the farthest safe visible point at every deterministic step."""
    if len(path) <= 2:
        return tuple(path)
    simplified = [path[0]]
    current = 0
    while current < len(path) - 1:
        selected = current + 1
        for candidate in range(len(path) - 1, current, -1):
            if segment_is_safe(path[current], path[candidate]):
                selected = candidate
                break
        simplified.append(path[selected])
        current = selected
    return tuple(simplified)


def select_validated_path(
    candidates: Sequence[tuple[str, Sequence[Point3D]]],
    obstacles: Sequence[CircularObstacle],
    config: PlannerConfig,
    expected_start: Point3D,
    expected_goal: Point3D,
) -> SimplificationResult:
    """Return the first safe candidate or a structured failure."""
    failures = []
    last_validation = validate_path((), obstacles, config)
    for method, candidate in candidates:
        dense = densify_path(candidate, config.maximum_waypoint_spacing_m)
        validation = validate_path(
            dense,
            obstacles,
            config,
            expected_start=expected_start,
            expected_goal=expected_goal,
        )
        if validation.valid:
            fallback_reason = "; ".join(failures)
            return SimplificationResult(
                dense,
                method,
                fallback_reason,
                validation,
            )
        failures.append(f"{method}: {validation.reason}")
        last_validation = validation
    return SimplificationResult(
        (),
        "none",
        "; ".join(failures),
        last_validation,
    )


def simplify_with_fallback(
    raw_path: Sequence[Point3D],
    obstacles: Sequence[CircularObstacle],
    config: PlannerConfig,
) -> SimplificationResult:
    """Try RDP, greedy visibility, then the dense raw path."""
    if len(raw_path) < 2:
        validation = validate_path(raw_path, obstacles, config)
        return SimplificationResult((), "none", validation.reason, validation)

    def segment_is_safe(start: Point3D, end: Point3D) -> bool:
        return validate_path(
            (start, end),
            obstacles,
            config,
            expected_start=start,
            expected_goal=end,
            enforce_spacing=False,
        ).valid

    rdp = rdp_simplify(raw_path, config.simplification_tolerance_m)
    greedy = greedy_safe_simplify(raw_path, segment_is_safe)
    return select_validated_path(
        (("rdp", rdp), ("greedy", greedy), ("raw", raw_path)),
        obstacles,
        config,
        raw_path[0],
        raw_path[-1],
    )
