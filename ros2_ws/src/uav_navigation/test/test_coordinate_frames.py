"""Regression tests for the verified Isaac-world to PX4-NED mapping."""

import math
import random

import pytest

from uav_navigation.coordinate_frames import (
    QUATERNION_CONVERSION_SUPPORTED,
    UnsupportedOrientationError,
    isaac_quaternion_to_ned,
    isaac_to_ned_acceleration,
    isaac_to_ned_heading,
    isaac_to_ned_position,
    isaac_to_ned_velocity,
    isaac_yaw_to_ned,
    ned_to_isaac_heading,
    ned_to_isaac_position,
    ned_yaw_to_isaac,
)
from uav_navigation.models import Point3D


def test_origin_and_basis_vectors() -> None:
    """Map the origin and each Isaac basis vector exactly."""
    assert isaac_to_ned_position(Point3D(0, 0, 0)) == Point3D(0, 0, 0)
    assert isaac_to_ned_position(Point3D(1, 0, 0)) == Point3D(0, 1, 0)
    assert isaac_to_ned_position(Point3D(0, 1, 0)) == Point3D(1, 0, 0)
    assert isaac_to_ned_position(Point3D(0, 0, 1)) == Point3D(0, 0, -1)


def test_fixed_seed_round_trip() -> None:
    """Round-trip deterministic random finite positions."""
    generator = random.Random(614420090)
    for _ in range(100):
        source = Point3D(
            generator.uniform(-100.0, 100.0),
            generator.uniform(-100.0, 100.0),
            generator.uniform(-20.0, 20.0),
        )
        converted = isaac_to_ned_position(source)
        assert ned_to_isaac_position(converted).almost_equals(source)


def test_offsets_apply_only_to_positions() -> None:
    """Apply configured translation to positions but never to free vectors."""
    source = Point3D(1.0, 2.0, 3.0)
    converted = isaac_to_ned_position(source, 10.0, 20.0, 30.0)
    assert converted == Point3D(12.0, 21.0, 27.0)
    assert ned_to_isaac_position(converted, 10.0, 20.0, 30.0) == source
    assert isaac_to_ned_velocity(source) == Point3D(2.0, 1.0, -3.0)
    assert isaac_to_ned_acceleration(source) == Point3D(2.0, 1.0, -3.0)


def test_heading_and_yaw_conventions() -> None:
    """Verify heading bases and the documented planar yaw mapping."""
    assert isaac_to_ned_heading(1.0, 0.0) == (0.0, 1.0)
    assert isaac_to_ned_heading(0.0, 1.0) == (1.0, 0.0)
    assert ned_to_isaac_heading(1.0, 0.0) == (0.0, 1.0)
    assert math.isclose(isaac_yaw_to_ned(0.0), math.pi / 2.0)
    assert math.isclose(isaac_yaw_to_ned(math.pi / 2.0), 0.0)
    for yaw in (-math.pi, -1.0, 0.0, 1.0, math.pi - 1e-6):
        round_trip = ned_yaw_to_isaac(isaac_yaw_to_ned(yaw))
        assert math.isclose(round_trip, yaw, abs_tol=1e-9)


def test_invalid_values_are_rejected() -> None:
    """Reject non-finite coordinates, offsets, headings, and yaw."""
    with pytest.raises(ValueError, match="finite"):
        Point3D(float("nan"), 0.0, 0.0)
    with pytest.raises(ValueError, match="offsets"):
        isaac_to_ned_position(Point3D(0, 0, 0), float("inf"), 0.0, 0.0)
    with pytest.raises(ValueError, match="nonzero"):
        isaac_to_ned_heading(0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        isaac_yaw_to_ned(float("nan"))


def test_quaternion_conversion_is_explicitly_unsupported() -> None:
    """Prevent callers from mistaking Phase 2 for a complete pose transform."""
    assert QUATERNION_CONVERSION_SUPPORTED is False
    with pytest.raises(UnsupportedOrientationError, match="unsupported"):
        isaac_quaternion_to_ned(0.0, 0.0, 0.0, 1.0)
