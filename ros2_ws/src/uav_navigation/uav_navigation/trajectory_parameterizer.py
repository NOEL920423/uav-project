"""Deterministic ROS-independent path time parameterization."""

import math
from collections.abc import Sequence

from uav_navigation.bspline_smoother import remove_adjacent_duplicates
from uav_navigation.geometry import discrete_curvatures_2d
from uav_navigation.models import Point3D
from uav_navigation.trajectory_models import (
    TrajectoryConfig,
    TrajectoryMetrics,
    TrajectoryPoint,
    TrajectoryResult,
)
from uav_navigation.trajectory_validator import (
    VALID_FRAME,
    validate_trajectory,
)
from uav_navigation.yaw_profile import finite_difference, ned_yaw_profile


def _distance_3d(first: Point3D, second: Point3D) -> float:
    return math.sqrt(
        (second.x - first.x) ** 2
        + (second.y - first.y) ** 2
        + (second.z - first.z) ** 2
    )


def _arc_lengths(path: Sequence[Point3D]) -> tuple[float, ...]:
    result = [0.0]
    for first, second in zip(path, path[1:]):
        result.append(result[-1] + _distance_3d(first, second))
    return tuple(result)


def _point_curvatures(path: Sequence[Point3D]) -> tuple[float, ...]:
    if len(path) == 2:
        return (0.0, 0.0)
    interior = discrete_curvatures_2d(path)
    signed = []
    for value, first, middle, last in zip(
        interior, path, path[1:], path[2:]
    ):
        cross = (
            (middle.x - first.x) * (last.y - middle.y)
            - (middle.y - first.y) * (last.x - middle.x)
        )
        signed.append(math.copysign(value, cross) if cross else 0.0)
    return (signed[0], *signed, signed[-1])


def _unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return tuple(value / magnitude for value in vector)


def _tangents(
    path: Sequence[Point3D],
) -> tuple[tuple[float, float, float], ...]:
    result = []
    for index in range(len(path)):
        if index == 0:
            first, last = path[0], path[1]
        elif index == len(path) - 1:
            first, last = path[-2], path[-1]
        else:
            first, last = path[index - 1], path[index + 1]
            if first == last:
                first, last = path[index], path[index + 1]
        result.append(
            _unit((last.x - first.x, last.y - first.y, last.z - first.z))
        )
    return tuple(result)


def _speed_profile(
    arc_lengths: Sequence[float],
    curvatures: Sequence[float],
    config: TrajectoryConfig,
) -> tuple[float, ...]:
    speeds = [
        min(
            config.maximum_speed_mps,
            math.sqrt(
                config.maximum_lateral_acceleration_mps2
                / max(abs(curvature), config.curvature_epsilon)
            ),
        )
        for curvature in curvatures
    ]
    speeds[0] = min(speeds[0], config.start_speed_mps)
    speeds[-1] = min(speeds[-1], config.end_speed_mps)
    for index in range(1, len(speeds)):
        distance = arc_lengths[index] - arc_lengths[index - 1]
        reachable = math.sqrt(
            speeds[index - 1] ** 2
            + 2.0
            * config.maximum_longitudinal_acceleration_mps2
            * distance
        )
        speeds[index] = min(speeds[index], reachable)
    for index in range(len(speeds) - 2, -1, -1):
        distance = arc_lengths[index + 1] - arc_lengths[index]
        reachable = math.sqrt(
            speeds[index + 1] ** 2
            + 2.0
            * config.maximum_longitudinal_deceleration_mps2
            * distance
        )
        speeds[index] = min(speeds[index], reachable)
    return tuple(speeds)


def _base_times(
    arc_lengths: Sequence[float],
    speeds: Sequence[float],
    config: TrajectoryConfig,
) -> tuple[float, ...]:
    times = [0.0]
    fallback_acceleration = min(
        config.maximum_longitudinal_acceleration_mps2,
        config.maximum_longitudinal_deceleration_mps2,
    )
    for index, (first_speed, second_speed) in enumerate(
        zip(speeds, speeds[1:])
    ):
        distance = arc_lengths[index + 1] - arc_lengths[index]
        speed_sum = first_speed + second_speed
        if speed_sum > config.minimum_speed_mps:
            duration = 2.0 * distance / speed_sum
        else:
            duration = 2.0 * math.sqrt(distance / fallback_acceleration)
        times.append(
            times[-1] + max(duration, config.minimum_segment_time_s)
        )
    return tuple(times)


def _vector_derivative(
    vectors: Sequence[Point3D],
    times: Sequence[float],
) -> tuple[Point3D, ...]:
    return tuple(
        Point3D(*components)
        for components in zip(
            finite_difference([item.x for item in vectors], times),
            finite_difference([item.y for item in vectors], times),
            finite_difference([item.z for item in vectors], times),
        )
    )


def _build_points(
    path: Sequence[Point3D],
    arcs: Sequence[float],
    curvatures: Sequence[float],
    base_speeds: Sequence[float],
    base_times: Sequence[float],
    scale: float,
) -> tuple[TrajectoryPoint, ...]:
    times = tuple(value * scale for value in base_times)
    speeds = tuple(value / scale for value in base_speeds)
    tangents = _tangents(path)
    velocities = tuple(
        Point3D(*(speed * component for component in tangent))
        for speed, tangent in zip(speeds, tangents)
    )
    tangential = finite_difference(speeds, times)
    accelerations = []
    for speed, tangent, along, curvature in zip(
        speeds, tangents, tangential, curvatures
    ):
        horizontal = math.hypot(tangent[0], tangent[1])
        if horizontal > 1e-12:
            normal = (-tangent[1] / horizontal, tangent[0] / horizontal, 0.0)
        else:
            normal = (0.0, 0.0, 0.0)
        lateral = speed**2 * curvature
        accelerations.append(
            Point3D(
                *(along * component + lateral * normal_component
                  for component, normal_component in zip(tangent, normal))
            )
        )
    jerks = _vector_derivative(accelerations, times)
    yaws = ned_yaw_profile(path)
    yaw_rates = finite_difference(yaws, times)
    yaw_accelerations = finite_difference(yaw_rates, times)
    return tuple(
        TrajectoryPoint(
            time_from_start_s=time,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            yaw_ned=yaw,
            yaw_rate_radps=yaw_rate,
            yaw_acceleration_radps2=yaw_acceleration,
            arc_length_m=arc,
            curvature_inverse_m=curvature,
        )
        for time, position, velocity, acceleration, jerk, yaw, yaw_rate,
        yaw_acceleration, arc, curvature in zip(
            times,
            path,
            velocities,
            accelerations,
            jerks,
            yaws,
            yaw_rates,
            yaw_accelerations,
            arcs,
            curvatures,
        )
    )


def _result(
    points: Sequence[TrajectoryPoint],
    source_count: int,
    scale: float,
    valid: bool,
    reason: str,
    diagnostics: tuple = (),
    metrics: TrajectoryMetrics | None = None,
) -> TrajectoryResult:
    measured = metrics or TrajectoryMetrics()
    return TrajectoryResult(
        success=bool(points),
        valid=bool(points) and valid,
        status_message="SUCCESS" if valid else "REJECTED",
        rejection_reason="" if valid else reason,
        trajectory_points=tuple(points),
        source_path_point_count=source_count,
        output_trajectory_point_count=len(points),
        path_length_m=measured.path_length_m,
        total_duration_s=measured.total_duration_s,
        time_scale=scale,
        maximum_speed_mps=measured.maximum_speed_mps,
        maximum_longitudinal_acceleration_mps2=(
            measured.maximum_longitudinal_acceleration_mps2
        ),
        maximum_lateral_acceleration_mps2=(
            measured.maximum_lateral_acceleration_mps2
        ),
        maximum_jerk_mps3=measured.maximum_jerk_mps3,
        maximum_yaw_rate_radps=measured.maximum_yaw_rate_radps,
        maximum_yaw_acceleration_radps2=(
            measured.maximum_yaw_acceleration_radps2
        ),
        start_speed_mps=measured.start_speed_mps,
        end_speed_mps=measured.end_speed_mps,
        validation_diagnostics=diagnostics,
    )


def parameterize_trajectory(
    path: Sequence[Point3D],
    frame_id: str = VALID_FRAME,
    config: TrajectoryConfig | None = None,
) -> TrajectoryResult:
    """Parameterize a path and accept it only after independent validation."""
    limits = config or TrajectoryConfig()
    source_count = len(path)
    if frame_id != VALID_FRAME:
        return _result((), source_count, 1.0, False, "wrong frame")
    cleaned = remove_adjacent_duplicates(path, 1e-9)
    if len(cleaned) < limits.trajectory_minimum_points:
        return _result(
            (), source_count, 1.0, False, "insufficient path points"
        )
    try:
        arcs = _arc_lengths(cleaned)
        if any(second <= first for first, second in zip(arcs, arcs[1:])):
            raise ValueError("arc length is not strictly increasing")
        curvatures = _point_curvatures(cleaned)
        speeds = _speed_profile(arcs, curvatures, limits)
        times = _base_times(arcs, speeds, limits)
    except (TypeError, ValueError, OverflowError) as error:
        return _result((), source_count, 1.0, False, str(error))
    scale = 1.0
    last_points: tuple[TrajectoryPoint, ...] = ()
    last_diagnostics = ()
    last_metrics = TrajectoryMetrics()
    for _ in range(limits.maximum_time_scaling_iterations):
        last_points = _build_points(
            cleaned, arcs, curvatures, speeds, times, scale
        )
        last_diagnostics, last_metrics = validate_trajectory(
            cleaned, last_points, frame_id, limits
        )
        if not last_diagnostics:
            return _result(
                last_points,
                source_count,
                scale,
                True,
                "",
                (),
                last_metrics,
            )
        ratios = []
        for diagnostic in last_diagnostics:
            if diagnostic.limit_value <= 0.0:
                continue
            ratio = diagnostic.measured_value / diagnostic.limit_value
            if "jerk" in diagnostic.constraint:
                ratios.append(ratio ** (1.0 / 3.0))
            elif "acceleration" in diagnostic.constraint:
                ratios.append(math.sqrt(ratio))
            elif "speed" in diagnostic.constraint or "rate" in (
                diagnostic.constraint
            ):
                ratios.append(ratio)
        if not ratios:
            break
        next_scale = scale * max(1.001, max(ratios) * 1.001)
        if next_scale > limits.maximum_total_time_scale:
            break
        scale = next_scale
    reason = (
        last_diagnostics[0].message
        if last_diagnostics
        else "time scaling did not converge"
    )
    return _result(
        last_points,
        source_count,
        scale,
        False,
        reason,
        last_diagnostics,
        last_metrics,
    )
