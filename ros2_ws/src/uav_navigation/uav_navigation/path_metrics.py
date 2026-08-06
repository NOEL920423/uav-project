"""Geometric path metrics kept separate from dynamics and tracking claims."""

import math
from collections.abc import Sequence

from uav_navigation.geometry import (
    absolute_heading_changes,
    point_to_segment_distance_2d,
    polyline_length_2d,
    segment_lengths_2d,
)
from uav_navigation.models import CircularObstacle, PathMetrics, Point3D


def calculate_path_metrics(
    path: Sequence[Point3D],
    obstacles: Sequence[CircularObstacle],
) -> PathMetrics:
    """Calculate deterministic 2D geometry metrics for a path."""
    lengths = segment_lengths_2d(path)
    changes = absolute_heading_changes(path)
    clearances = [
        point_to_segment_distance_2d(obstacle.center, start, end)
        - obstacle.radius
        for start, end in zip(path, path[1:])
        for obstacle in obstacles
    ]
    mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    mean_change = sum(changes) / len(changes) if changes else 0.0
    variance = (
        sum((change - mean_change) ** 2 for change in changes) / len(changes)
        if changes
        else 0.0
    )
    return PathMetrics(
        point_count=len(path),
        path_length_m=polyline_length_2d(path),
        minimum_physical_clearance_m=min(clearances, default=math.inf),
        mean_segment_length_m=mean_length,
        maximum_segment_length_m=max(lengths, default=0.0),
        mean_absolute_heading_change_rad=mean_change,
        maximum_absolute_heading_change_rad=max(changes, default=0.0),
        heading_change_variance_rad2=variance,
    )
