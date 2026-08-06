"""Verified position and planar-vector mappings for the planner milestone."""

import math

from uav_navigation.models import Point3D


ISAAC_WORLD_FRAME = "isaac_world"
PLANNING_FRAME = "px4_ned"
BODY_FRAME = "base_link"
FPV_CAMERA_FRAME = "uav_fpv_camera"
OBSERVER_CAMERA_FRAME = "uav_observer_camera"
QUATERNION_CONVERSION_SUPPORTED = False


class UnsupportedOrientationError(NotImplementedError):
    """Raised when an unverified full orientation conversion is requested."""


def _finite_offsets(
    ned_offset_x: float,
    ned_offset_y: float,
    ned_offset_z: float,
) -> tuple[float, float, float]:
    offsets = (
        float(ned_offset_x),
        float(ned_offset_y),
        float(ned_offset_z),
    )
    if not all(math.isfinite(value) for value in offsets):
        raise ValueError("NED translation offsets must be finite")
    return offsets


def isaac_to_ned_position(
    point: Point3D,
    ned_offset_x: float = 0.0,
    ned_offset_y: float = 0.0,
    ned_offset_z: float = 0.0,
) -> Point3D:
    """Convert an Isaac-world position to PX4 local NED with translation."""
    offset_x, offset_y, offset_z = _finite_offsets(
        ned_offset_x,
        ned_offset_y,
        ned_offset_z,
    )
    return Point3D(
        x=point.y + offset_x,
        y=point.x + offset_y,
        z=-point.z + offset_z,
    )


def ned_to_isaac_position(
    point: Point3D,
    ned_offset_x: float = 0.0,
    ned_offset_y: float = 0.0,
    ned_offset_z: float = 0.0,
) -> Point3D:
    """Invert the translated Isaac-world to PX4-NED position mapping."""
    offset_x, offset_y, offset_z = _finite_offsets(
        ned_offset_x,
        ned_offset_y,
        ned_offset_z,
    )
    return Point3D(
        x=point.y - offset_y,
        y=point.x - offset_x,
        z=-(point.z - offset_z),
    )


def isaac_to_ned_vector(vector: Point3D) -> Point3D:
    """Convert a free vector without applying any translation offset."""
    return Point3D(x=vector.y, y=vector.x, z=-vector.z)


def ned_to_isaac_vector(vector: Point3D) -> Point3D:
    """Invert the free-vector axis mapping without translation."""
    return Point3D(x=vector.y, y=vector.x, z=-vector.z)


def isaac_to_ned_velocity(velocity: Point3D) -> Point3D:
    """Convert an Isaac velocity vector to NED axes."""
    return isaac_to_ned_vector(velocity)


def isaac_to_ned_acceleration(acceleration: Point3D) -> Point3D:
    """Convert an Isaac acceleration vector to NED axes."""
    return isaac_to_ned_vector(acceleration)


def isaac_to_ned_heading(x: float, y: float) -> tuple[float, float]:
    """Convert a nonzero Isaac XY heading to NED north/east components."""
    x_value = float(x)
    y_value = float(y)
    if not math.isfinite(x_value) or not math.isfinite(y_value):
        raise ValueError("heading components must be finite")
    if math.hypot(x_value, y_value) <= 1e-12:
        raise ValueError("heading vector must be nonzero")
    return y_value, x_value


def ned_to_isaac_heading(north: float, east: float) -> tuple[float, float]:
    """Invert the nonzero NED north/east heading mapping."""
    north_value = float(north)
    east_value = float(east)
    if not math.isfinite(north_value) or not math.isfinite(east_value):
        raise ValueError("heading components must be finite")
    if math.hypot(north_value, east_value) <= 1e-12:
        raise ValueError("heading vector must be nonzero")
    return east_value, north_value


def normalize_angle(angle_rad: float) -> float:
    """Normalize a finite angle to the half-open interval [-pi, pi)."""
    angle = float(angle_rad)
    if not math.isfinite(angle):
        raise ValueError("angle must be finite")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def isaac_yaw_to_ned(yaw_isaac_rad: float) -> float:
    """Convert planar Isaac yaw to NED yaw under the documented conventions."""
    return normalize_angle(0.5 * math.pi - float(yaw_isaac_rad))


def ned_yaw_to_isaac(yaw_ned_rad: float) -> float:
    """Invert the documented planar yaw conversion."""
    return normalize_angle(0.5 * math.pi - float(yaw_ned_rad))


def isaac_quaternion_to_ned(
    *_components: float,
) -> tuple[float, float, float, float]:
    """Reject quaternion conversion until the flight-frame contract exists."""
    raise UnsupportedOrientationError(
        "Quaternion conversion is unsupported before flight-frame validation"
    )
