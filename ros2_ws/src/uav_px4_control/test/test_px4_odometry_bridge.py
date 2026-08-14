"""Pure validation checks for the PX4-to-follower odometry boundary."""

import math

from uav_px4_control.px4_odometry_bridge_node import px4_odometry_is_valid


class FakeOdometry:
    """Supply the exact fields consumed by the validation function."""

    POSE_FRAME_NED = 1
    VELOCITY_FRAME_NED = 1

    def __init__(self):
        """Create one valid stationary NED sample."""
        self.pose_frame = 1
        self.velocity_frame = 1
        self.position = [0.0, 0.0, -2.0]
        self.velocity = [0.0, 0.0, 0.0]
        self.q = [1.0, 0.0, 0.0, 0.0]
        self.angular_velocity = [0.0, 0.0, 0.0]


def test_finite_ned_odometry_is_accepted():
    """Finite NED pose, twist, and quaternion cross the bridge."""
    assert px4_odometry_is_valid(FakeOdometry())


def test_wrong_frame_nonfinite_and_zero_quaternion_are_rejected():
    """Each malformed external-state family is fail-closed."""
    message = FakeOdometry()
    message.pose_frame = 2
    assert not px4_odometry_is_valid(message)
    message = FakeOdometry()
    message.velocity[1] = math.nan
    assert not px4_odometry_is_valid(message)
    message = FakeOdometry()
    message.q = [0.0, 0.0, 0.0, 0.0]
    assert not px4_odometry_is_valid(message)
