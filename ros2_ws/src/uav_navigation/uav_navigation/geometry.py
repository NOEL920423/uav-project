"""Deterministic dependency-free planar geometry helpers."""

import math
from collections.abc import Sequence

from uav_navigation.models import Point3D


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a finite value to an ordered finite interval."""
    if not all(math.isfinite(item) for item in (value, lower, upper)):
        raise ValueError("clamp arguments must be finite")
    if lower > upper:
        raise ValueError("clamp lower bound must not exceed upper bound")
    return max(lower, min(upper, value))


def distance_2d(first: Point3D, second: Point3D) -> float:
    """Return Euclidean XY distance."""
    return math.hypot(second.x - first.x, second.y - first.y)


def point_to_segment_distance_2d(
    point: Point3D,
    start: Point3D,
    end: Point3D,
) -> float:
    """Return the shortest XY distance from a point to a closed segment."""
    dx = end.x - start.x
    dy = end.y - start.y
    squared_length = dx * dx + dy * dy
    if squared_length == 0.0:
        return distance_2d(point, start)
    projection = (
        (point.x - start.x) * dx + (point.y - start.y) * dy
    ) / squared_length
    ratio = clamp(projection, 0.0, 1.0)
    closest = Point3D(start.x + ratio * dx, start.y + ratio * dy, start.z)
    return distance_2d(point, closest)


def polyline_length_2d(path: Sequence[Point3D]) -> float:
    """Return cumulative XY length."""
    return sum(
        distance_2d(first, second)
        for first, second in zip(path, path[1:])
    )


def segment_lengths_2d(path: Sequence[Point3D]) -> tuple[float, ...]:
    """Return each XY segment length."""
    return tuple(
        distance_2d(first, second)
        for first, second in zip(path, path[1:])
    )


def interpolate_segment(
    start: Point3D,
    end: Point3D,
    ratio: float,
) -> Point3D:
    """Linearly interpolate a point at a ratio in the closed unit interval."""
    ratio = clamp(float(ratio), 0.0, 1.0)
    return Point3D(
        start.x + ratio * (end.x - start.x),
        start.y + ratio * (end.y - start.y),
        start.z + ratio * (end.z - start.z),
    )


def absolute_heading_changes(path: Sequence[Point3D]) -> tuple[float, ...]:
    """Return wrapped absolute heading changes, ignoring zero-length legs."""
    headings: list[float] = []
    for first, second in zip(path, path[1:]):
        if distance_2d(first, second) > 0.0:
            headings.append(math.atan2(second.y - first.y, second.x - first.x))
    changes = []
    for previous, current in zip(headings, headings[1:]):
        wrapped = (current - previous + math.pi) % (2.0 * math.pi) - math.pi
        changes.append(abs(wrapped))
    return tuple(changes)
