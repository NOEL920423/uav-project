"""Independent dynamic and structural validator for timed trajectories."""

import math
from collections.abc import Sequence

from uav_navigation.models import Point3D
from uav_navigation.trajectory_metrics import (
    measure_trajectory,
    vector_norm,
)
from uav_navigation.trajectory_models import (
    TrajectoryConfig,
    TrajectoryDiagnostic,
    TrajectoryMetrics,
    TrajectoryPoint,
)
from uav_navigation.yaw_profile import finite_difference

VALID_FRAME = "px4_ned"
TOLERANCE = 1e-7


def _diagnostic(
    constraint: str,
    index: int,
    measured: float,
    limit: float,
) -> TrajectoryDiagnostic:
    """Create a consistent limit diagnostic."""
    return TrajectoryDiagnostic(
        constraint=constraint,
        point_index=index,
        measured_value=measured,
        limit_value=limit,
        message=(
            f"{constraint} at point {index}: measured={measured:.9g}, "
            f"limit={limit:.9g}"
        ),
    )


def validate_trajectory(
    source_path: Sequence[Point3D],
    points: Sequence[TrajectoryPoint],
    frame_id: str,
    config: TrajectoryConfig,
) -> tuple[tuple[TrajectoryDiagnostic, ...], TrajectoryMetrics]:
    """Independently validate geometry, timing, fields, and dynamic limits."""
    diagnostics: list[TrajectoryDiagnostic] = []
    if frame_id != VALID_FRAME:
        diagnostics.append(
            _diagnostic("frame", -1, 0.0, 0.0)
        )
    if len(points) < config.trajectory_minimum_points:
        diagnostics.append(
            _diagnostic(
                "minimum_points",
                -1,
                float(len(points)),
                float(config.trajectory_minimum_points),
            )
        )
    if len(points) != len(source_path):
        diagnostics.append(
            _diagnostic(
                "geometry_point_count",
                -1,
                float(len(points)),
                float(len(source_path)),
            )
        )
    for index, (source, point) in enumerate(zip(source_path, points)):
        if point.position != source:
            diagnostics.append(
                _diagnostic("geometry_preservation", index, 1.0, 0.0)
            )
    scalar_names = (
        "time_from_start_s",
        "yaw_ned",
        "yaw_rate_radps",
        "yaw_acceleration_radps2",
        "arc_length_m",
        "curvature_inverse_m",
    )
    for index, point in enumerate(points):
        values = [getattr(point, name) for name in scalar_names]
        values.extend(
            coordinate
            for vector in (
                point.position,
                point.velocity,
                point.acceleration,
                point.jerk,
            )
            for coordinate in (vector.x, vector.y, vector.z)
        )
        if not all(math.isfinite(value) for value in values):
            diagnostics.append(
                _diagnostic("finite_fields", index, math.inf, 0.0)
            )
    if points:
        if abs(points[0].time_from_start_s) > TOLERANCE:
            diagnostics.append(
                _diagnostic(
                    "initial_time", 0, points[0].time_from_start_s, 0.0
                )
            )
        if abs(points[0].arc_length_m) > TOLERANCE:
            diagnostics.append(
                _diagnostic(
                    "initial_arc_length", 0, points[0].arc_length_m, 0.0
                )
            )
    for index, (first, second) in enumerate(zip(points, points[1:]), 1):
        if second.time_from_start_s <= first.time_from_start_s:
            diagnostics.append(
                _diagnostic(
                    "strict_time",
                    index,
                    second.time_from_start_s - first.time_from_start_s,
                    config.minimum_segment_time_s,
                )
            )
        if second.arc_length_m <= first.arc_length_m:
            diagnostics.append(
                _diagnostic(
                    "strict_arc_length",
                    index,
                    second.arc_length_m - first.arc_length_m,
                    0.0,
                )
            )
    if diagnostics and not points:
        return tuple(diagnostics), TrajectoryMetrics()
    try:
        metrics = measure_trajectory(points)
    except ValueError:
        diagnostics.append(_diagnostic("metric_computation", -1, 1.0, 0.0))
        return tuple(diagnostics), TrajectoryMetrics()
    speeds = [vector_norm(point.velocity) for point in points]
    times = [point.time_from_start_s for point in points]
    try:
        longitudinal = finite_difference(speeds, times)
    except ValueError:
        longitudinal = [math.inf] * len(points)
    checks = (
        (
            "maximum_speed_mps",
            speeds,
            config.maximum_speed_mps,
        ),
        (
            "maximum_lateral_acceleration_mps2",
            [
                speed**2 * abs(point.curvature_inverse_m)
                for point, speed in zip(points, speeds)
            ],
            config.maximum_lateral_acceleration_mps2,
        ),
        (
            "maximum_jerk_mps3",
            [vector_norm(point.jerk) for point in points],
            config.maximum_jerk_mps3,
        ),
        (
            "maximum_yaw_rate_radps",
            [abs(point.yaw_rate_radps) for point in points],
            config.maximum_yaw_rate_radps,
        ),
        (
            "maximum_yaw_acceleration_radps2",
            [abs(point.yaw_acceleration_radps2) for point in points],
            config.maximum_yaw_acceleration_radps2,
        ),
    )
    for constraint, values, limit in checks:
        for index, measured in enumerate(values):
            if measured > limit + TOLERANCE:
                diagnostics.append(
                    _diagnostic(constraint, index, measured, limit)
                )
    for index, measured in enumerate(longitudinal):
        limit = (
            config.maximum_longitudinal_acceleration_mps2
            if measured >= 0.0
            else config.maximum_longitudinal_deceleration_mps2
        )
        if abs(measured) > limit + TOLERANCE:
            constraint = (
                "maximum_longitudinal_acceleration_mps2"
                if measured >= 0.0
                else "maximum_longitudinal_deceleration_mps2"
            )
            diagnostics.append(
                _diagnostic(constraint, index, abs(measured), limit)
            )
    endpoint_checks = (
        ("start_speed_mps", 0, speeds[0], config.start_speed_mps),
        ("end_speed_mps", -1, speeds[-1], config.end_speed_mps),
    )
    for constraint, index, measured, limit in endpoint_checks:
        if abs(measured - limit) > TOLERANCE:
            diagnostics.append(_diagnostic(constraint, index, measured, limit))
    if config.require_zero_start_speed and speeds[0] > TOLERANCE:
        diagnostics.append(
            _diagnostic("require_zero_start_speed", 0, speeds[0], 0.0)
        )
    if config.require_zero_end_speed and speeds[-1] > TOLERANCE:
        diagnostics.append(
            _diagnostic("require_zero_end_speed", -1, speeds[-1], 0.0)
        )
    return tuple(diagnostics), metrics
