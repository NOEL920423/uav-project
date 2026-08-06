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


def cumulative_polyline_lengths_2d(
    path: Sequence[Point3D],
) -> tuple[float, ...]:
    """Return cumulative XY length at each point, beginning with zero."""
    if not path:
        return ()
    cumulative = [0.0]
    for first, second in zip(path, path[1:]):
        cumulative.append(cumulative[-1] + distance_2d(first, second))
    return tuple(cumulative)


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


def points_are_finite(path: Sequence[Point3D]) -> bool:
    """Confirm finite coordinates at an explicit collection boundary."""
    return all(
        math.isfinite(coordinate)
        for point in path
        for coordinate in (point.x, point.y, point.z)
    )


def discrete_curvatures_2d(
    path: Sequence[Point3D],
    tolerance: float = 1e-12,
) -> tuple[float, ...]:
    """Estimate circumcircle curvature for each consecutive point triple."""
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("curvature tolerance must be finite and positive")
    curvatures = []
    for first, middle, last in zip(path, path[1:], path[2:]):
        first_leg = distance_2d(first, middle)
        second_leg = distance_2d(middle, last)
        chord = distance_2d(first, last)
        if min(first_leg, second_leg) <= tolerance:
            raise ValueError(
                "curvature is undefined for a zero-length segment"
            )
        if chord <= tolerance:
            raise ValueError(
                "curvature is undefined for coincident triple ends"
            )
        twice_area = abs(
            (middle.x - first.x) * (last.y - first.y)
            - (middle.y - first.y) * (last.x - first.x)
        )
        denominator = first_leg * second_leg * chord
        curvature = 2.0 * twice_area / denominator
        if not math.isfinite(curvature):
            raise ValueError(
                "curvature calculation produced a non-finite value"
            )
        curvatures.append(0.0 if curvature <= tolerance else curvature)
    return tuple(curvatures)


def polyline_self_intersection_2d(
    path: Sequence[Point3D],
    tolerance: float = 1e-9,
) -> tuple[int, int] | None:
    """Return the first crossing pair of non-adjacent segments."""
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "intersection tolerance must be finite and nonnegative"
        )

    def orientation(first: Point3D, second: Point3D, third: Point3D) -> float:
        return (
            (second.x - first.x) * (third.y - first.y)
            - (second.y - first.y) * (third.x - first.x)
        )

    def on_segment(first: Point3D, point: Point3D, second: Point3D) -> bool:
        return (
            min(first.x, second.x) - tolerance
            <= point.x
            <= max(first.x, second.x) + tolerance
            and min(first.y, second.y) - tolerance
            <= point.y
            <= max(first.y, second.y) + tolerance
        )

    def intersects(
        first_start: Point3D,
        first_end: Point3D,
        second_start: Point3D,
        second_end: Point3D,
    ) -> bool:
        values = (
            orientation(first_start, first_end, second_start),
            orientation(first_start, first_end, second_end),
            orientation(second_start, second_end, first_start),
            orientation(second_start, second_end, first_end),
        )
        first_crosses = (values[0] > tolerance and values[1] < -tolerance) or (
            values[0] < -tolerance and values[1] > tolerance
        )
        second_crosses = (
            values[2] > tolerance and values[3] < -tolerance
        ) or (values[2] < -tolerance and values[3] > tolerance)
        if first_crosses and second_crosses:
            return True
        collinear_checks = (
            (values[0], first_start, second_start, first_end),
            (values[1], first_start, second_end, first_end),
            (values[2], second_start, first_start, second_end),
            (values[3], second_start, first_end, second_end),
        )
        return any(
            abs(value) <= tolerance and on_segment(start, point, end)
            for value, start, point, end in collinear_checks
        )

    segment_count = max(0, len(path) - 1)
    for first_index in range(segment_count):
        for second_index in range(first_index + 2, segment_count):
            if intersects(
                path[first_index],
                path[first_index + 1],
                path[second_index],
                path[second_index + 1],
            ):
                return first_index, second_index
    return None
