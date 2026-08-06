"""Deterministic binary-search sampler for pure timed trajectories."""

import bisect
import math
from collections.abc import Sequence

from uav_navigation.models import Point3D
from uav_navigation.tracking_models import ReferenceSample
from uav_navigation.trajectory_models import TrajectoryPoint


def _point_values(point: TrajectoryPoint) -> tuple[float, ...]:
    return (
        point.time_from_start_s,
        point.position.x, point.position.y, point.position.z,
        point.velocity.x, point.velocity.y, point.velocity.z,
        point.acceleration.x, point.acceleration.y, point.acceleration.z,
        point.jerk.x, point.jerk.y, point.jerk.z,
        point.yaw_ned,
        point.yaw_rate_radps,
        point.yaw_acceleration_radps2,
        point.arc_length_m,
        point.curvature_inverse_m,
    )


def validate_sampleable_trajectory(
    points: Sequence[TrajectoryPoint],
) -> tuple[TrajectoryPoint, ...]:
    """Return immutable points or reject invalid sampling structure."""
    result = tuple(points)
    if len(result) < 2:
        raise ValueError("trajectory must contain at least two points")
    for index, point in enumerate(result):
        if not all(math.isfinite(value) for value in _point_values(point)):
            raise ValueError(f"trajectory point {index} is non-finite")
    if abs(result[0].time_from_start_s) > 1e-9:
        raise ValueError("trajectory initial time must be zero")
    if any(
        second.time_from_start_s <= first.time_from_start_s
        for first, second in zip(result, result[1:])
    ):
        raise ValueError("trajectory timestamps must be strictly increasing")
    return result


def _lerp(first: float, second: float, ratio: float) -> float:
    return first + ratio * (second - first)


def _lerp_vector(first: Point3D, second: Point3D, ratio: float) -> Point3D:
    return Point3D(
        _lerp(first.x, second.x, ratio),
        _lerp(first.y, second.y, ratio),
        _lerp(first.z, second.z, ratio),
    )


def _interpolate(
    first: TrajectoryPoint,
    second: TrajectoryPoint,
    time_s: float,
) -> TrajectoryPoint:
    duration = second.time_from_start_s - first.time_from_start_s
    if duration <= 0.0:
        raise ValueError("cannot interpolate across invalid timestamps")
    ratio = (time_s - first.time_from_start_s) / duration
    point = TrajectoryPoint(
        time_from_start_s=time_s,
        position=_lerp_vector(first.position, second.position, ratio),
        velocity=_lerp_vector(first.velocity, second.velocity, ratio),
        acceleration=_lerp_vector(
            first.acceleration, second.acceleration, ratio
        ),
        jerk=_lerp_vector(first.jerk, second.jerk, ratio),
        yaw_ned=_lerp(first.yaw_ned, second.yaw_ned, ratio),
        yaw_rate_radps=_lerp(
            first.yaw_rate_radps, second.yaw_rate_radps, ratio
        ),
        yaw_acceleration_radps2=_lerp(
            first.yaw_acceleration_radps2,
            second.yaw_acceleration_radps2,
            ratio,
        ),
        arc_length_m=_lerp(first.arc_length_m, second.arc_length_m, ratio),
        curvature_inverse_m=_lerp(
            first.curvature_inverse_m,
            second.curvature_inverse_m,
            ratio,
        ),
    )
    if not all(math.isfinite(value) for value in _point_values(point)):
        raise ValueError("interpolated reference is non-finite")
    return point


def sample_trajectory(
    points: Sequence[TrajectoryPoint], time_s: float
) -> ReferenceSample:
    """Sample before, within, or after a validated relative-time trajectory."""
    trajectory = validate_sampleable_trajectory(points)
    requested = float(time_s)
    if not math.isfinite(requested):
        raise ValueError("sample time must be finite")
    if requested <= 0.0:
        return ReferenceSample(
            trajectory[0], reference_index=0, prestart=requested < 0.0
        )
    final_time = trajectory[-1].time_from_start_s
    if requested >= final_time:
        return ReferenceSample(
            trajectory[-1],
            reference_index=len(trajectory) - 1,
            terminal=True,
        )
    times = [point.time_from_start_s for point in trajectory]
    upper = bisect.bisect_right(times, requested)
    lower = upper - 1
    if requested == trajectory[lower].time_from_start_s:
        return ReferenceSample(trajectory[lower], reference_index=lower)
    return ReferenceSample(
        _interpolate(trajectory[lower], trajectory[upper], requested),
        reference_index=lower,
    )
