"""Independent continuous safety validation for candidate paths."""

import math
from collections.abc import Sequence

from uav_navigation.geometry import distance_2d, point_to_segment_distance_2d
from uav_navigation.models import (
    CircularObstacle,
    PlannerConfig,
    Point3D,
    ValidationResult,
)


def planning_radius(
    obstacle: CircularObstacle,
    config: PlannerConfig,
) -> float:
    """Return the physical UAV plus static-margin planning radius."""
    return (
        obstacle.radius
        + config.uav_physical_radius_m
        + config.static_safety_margin_m
    )


def validation_radius(
    obstacle: CircularObstacle,
    config: PlannerConfig,
) -> float:
    """Return the stricter continuous segment-validation radius."""
    return (
        planning_radius(obstacle, config)
        + config.minimum_segment_clearance_m
    )


def filter_overflyable_obstacles(
    obstacles: Sequence[CircularObstacle],
    config: PlannerConfig,
) -> tuple[CircularObstacle, ...]:
    """Remove obstacles safely below the fixed flight altitude."""
    if not config.enable_overfly_short_obstacles:
        return tuple(obstacles)
    retained = []
    for obstacle in obstacles:
        isaac_center_z = config.ned_origin_offset_z_m - obstacle.center.z
        isaac_top_z = isaac_center_z + 0.5 * obstacle.height
        protected_top = isaac_top_z + config.overfly_vertical_clearance_m
        if protected_top > config.flight_altitude_m:
            retained.append(obstacle)
    return tuple(retained)


def minimum_segment_clearance(
    start: Point3D,
    end: Point3D,
    obstacles: Sequence[CircularObstacle],
    config: PlannerConfig,
    *,
    use_validation_radius: bool = True,
) -> float:
    """Return minimum signed clearance for one segment."""
    minimum = math.inf
    for obstacle in obstacles:
        radius = (
            validation_radius(obstacle, config)
            if use_validation_radius
            else planning_radius(obstacle, config)
        )
        clearance = point_to_segment_distance_2d(
            obstacle.center,
            start,
            end,
        ) - radius
        minimum = min(minimum, clearance)
    return minimum


def point_clearance(
    point: Point3D,
    obstacle: CircularObstacle,
    config: PlannerConfig,
    *,
    use_validation_radius: bool = True,
) -> float:
    """Return signed clearance from a point to one obstacle envelope."""
    radius = (
        validation_radius(obstacle, config)
        if use_validation_radius
        else planning_radius(obstacle, config)
    )
    return distance_2d(point, obstacle.center) - radius


def segment_clearance(
    start: Point3D,
    end: Point3D,
    obstacle: CircularObstacle,
    config: PlannerConfig,
    *,
    use_validation_radius: bool = True,
) -> float:
    """Return signed clearance from one segment to one obstacle envelope."""
    radius = (
        validation_radius(obstacle, config)
        if use_validation_radius
        else planning_radius(obstacle, config)
    )
    return point_to_segment_distance_2d(
        obstacle.center,
        start,
        end,
    ) - radius


def nearest_obstacle_clearance(
    point: Point3D,
    obstacles: Sequence[CircularObstacle],
    config: PlannerConfig,
    *,
    use_validation_radius: bool = True,
) -> float:
    """Return the nearest signed point clearance, or infinity when empty."""
    return min(
        (
            point_clearance(
                point,
                obstacle,
                config,
                use_validation_radius=use_validation_radius,
            )
            for obstacle in obstacles
        ),
        default=math.inf,
    )


def validate_path(
    path: Sequence[Point3D],
    obstacles: Sequence[CircularObstacle],
    config: PlannerConfig,
    *,
    expected_start: Point3D | None = None,
    expected_goal: Point3D | None = None,
    enforce_spacing: bool = True,
) -> ValidationResult:
    """Validate endpoints, bounds, spacing, and every continuous segment."""
    if len(path) < 2:
        return ValidationResult(
            False,
            "path requires at least two points",
            math.inf,
            0.0,
        )
    tolerance = config.numerical_tolerance
    if expected_start is not None:
        if not path[0].almost_equals(expected_start, tolerance):
            return ValidationResult(
                False,
                "path start does not match requested start",
                math.inf,
                0.0,
            )
    if expected_goal is not None:
        if not path[-1].almost_equals(expected_goal, tolerance):
            return ValidationResult(
                False,
                "path goal does not match requested goal",
                math.inf,
                0.0,
            )
    if config.planning_bounds is not None:
        xmin, xmax, ymin, ymax = config.planning_bounds
        for index, point in enumerate(path):
            if not (
                xmin - tolerance <= point.x <= xmax + tolerance
                and ymin - tolerance <= point.y <= ymax + tolerance
            ):
                reason = f"point {index} lies outside planning bounds"
                return ValidationResult(False, reason, math.inf, 0.0)

    minimum = math.inf
    maximum_length = 0.0
    for index, (start, end) in enumerate(zip(path, path[1:])):
        length = distance_2d(start, end)
        maximum_length = max(maximum_length, length)
        if length <= tolerance:
            reason = f"segment {index} has zero length"
            return ValidationResult(False, reason, minimum, maximum_length)
        spacing_limit = config.maximum_waypoint_spacing_m + tolerance
        if enforce_spacing and length > spacing_limit:
            reason = f"segment {index} exceeds maximum waypoint spacing"
            return ValidationResult(False, reason, minimum, maximum_length)
        for obstacle in obstacles:
            clearance = point_to_segment_distance_2d(
                obstacle.center,
                start,
                end,
            ) - validation_radius(obstacle, config)
            minimum = min(minimum, clearance)
            if clearance < -tolerance:
                reason = f"segment {index} intersects obstacle {obstacle.name}"
                return ValidationResult(False, reason, minimum, maximum_length)
    return ValidationResult(True, "safe", minimum, maximum_length)
