"""Independent pure measurements for timed trajectories."""

import math
from collections.abc import Sequence

from uav_navigation.trajectory_models import (
    TrajectoryMetrics,
    TrajectoryPoint,
)
from uav_navigation.yaw_profile import finite_difference


def vector_norm(vector: object) -> float:
    """Return the Euclidean norm of an object with x, y, and z fields."""
    return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)


def measure_trajectory(
    points: Sequence[TrajectoryPoint],
) -> TrajectoryMetrics:
    """Derive constraint metrics without trusting parameterizer summaries."""
    if not points:
        return TrajectoryMetrics()
    speeds = [vector_norm(point.velocity) for point in points]
    jerks = [vector_norm(point.jerk) for point in points]
    times = [point.time_from_start_s for point in points]
    longitudinal = finite_difference(speeds, times)
    lateral = [
        speed**2 * abs(point.curvature_inverse_m)
        for point, speed in zip(points, speeds)
    ]
    return TrajectoryMetrics(
        point_count=len(points),
        path_length_m=points[-1].arc_length_m,
        total_duration_s=points[-1].time_from_start_s,
        maximum_speed_mps=max(speeds),
        maximum_longitudinal_acceleration_mps2=max(
            (abs(value) for value in longitudinal), default=0.0
        ),
        maximum_lateral_acceleration_mps2=max(lateral, default=0.0),
        maximum_jerk_mps3=max(jerks),
        maximum_yaw_rate_radps=max(
            (abs(point.yaw_rate_radps) for point in points),
            default=0.0,
        ),
        maximum_yaw_acceleration_radps2=max(
            (abs(point.yaw_acceleration_radps2) for point in points),
            default=0.0,
        ),
        start_speed_mps=speeds[0],
        end_speed_mps=speeds[-1],
    )
