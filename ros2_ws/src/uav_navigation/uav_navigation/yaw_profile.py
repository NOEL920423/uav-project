"""Pure NED yaw generation and finite-difference derivatives."""

import math
from collections.abc import Sequence

from uav_navigation.models import Point3D


def unwrap_angles(angles: Sequence[float]) -> tuple[float, ...]:
    """Return a continuous angle sequence by removing two-pi jumps."""
    if not angles:
        return ()
    result = [float(angles[0])]
    for angle in angles[1:]:
        delta = (float(angle) - result[-1] + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        result.append(result[-1] + delta)
    return tuple(result)


def ned_yaw_profile(path: Sequence[Point3D]) -> tuple[float, ...]:
    """Compute unwrapped yaw as atan2(east, north) from path tangents."""
    if len(path) < 2:
        return ()
    headings = [
        math.atan2(second.y - first.y, second.x - first.x)
        for first, second in zip(path, path[1:])
    ]
    return unwrap_angles((*headings, headings[-1]))


def finite_difference(
    values: Sequence[float],
    times: Sequence[float],
) -> tuple[float, ...]:
    """Differentiate samples using one-sided ends and centred interiors."""
    if len(values) != len(times):
        raise ValueError("values and times must have equal lengths")
    if not values:
        return ()
    if len(values) == 1:
        return (0.0,)
    result = []
    for index in range(len(values)):
        if index == 0:
            first, last = 0, 1
        elif index == len(values) - 1:
            first, last = index - 1, index
        else:
            first, last = index - 1, index + 1
        duration = times[last] - times[first]
        if duration <= 0.0:
            raise ValueError("times must be strictly increasing")
        result.append((values[last] - values[first]) / duration)
    return tuple(result)
