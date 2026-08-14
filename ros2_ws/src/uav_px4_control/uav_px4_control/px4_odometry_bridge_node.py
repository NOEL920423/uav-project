"""Bridge validated PX4 NED odometry into the existing trajectory follower."""

import importlib
import math

from nav_msgs.msg import Odometry

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from uav_navigation.trajectory_follower_node import ODOMETRY_TOPIC, live_qos

from uav_px4_control.px4_setpoint_streamer_node import (
    VEHICLE_ODOMETRY_TOPIC,
    px4_output_qos,
)


def px4_odometry_is_valid(message) -> bool:
    """Require NED position/velocity and finite pose, twist, and quaternion."""
    values = (
        *message.position,
        *message.velocity,
        *message.q,
        *message.angular_velocity,
    )
    return bool(
        message.pose_frame == message.POSE_FRAME_NED
        and message.velocity_frame == message.VELOCITY_FRAME_NED
        and all(math.isfinite(float(value)) for value in values)
        and sum(float(value) ** 2 for value in message.q) > 1e-8
    )


class Px4OdometryBridgeNode(Node):
    """Publish only finite NED state through the Phase 5 odometry contract."""

    def __init__(self) -> None:
        """Create one validated PX4 input and one follower output."""
        super().__init__("px4_odometry_bridge")
        message_module = importlib.import_module("px4_msgs.msg")
        self._publisher = self.create_publisher(
            Odometry, ODOMETRY_TOPIC, live_qos()
        )
        self.create_subscription(
            message_module.VehicleOdometry,
            VEHICLE_ODOMETRY_TOPIC,
            self._callback,
            px4_output_qos(),
        )
        self.invalid_message_count = 0

    def _callback(self, source) -> None:
        if not px4_odometry_is_valid(source):
            self.invalid_message_count += 1
            self.get_logger().error(
                "refusing invalid or non-NED PX4 odometry; follower will HOLD"
            )
            return
        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "px4_ned"
        message.child_frame_id = "uav_base_frd"
        message.pose.pose.position.x = float(source.position[0])
        message.pose.pose.position.y = float(source.position[1])
        message.pose.pose.position.z = float(source.position[2])
        # PX4 VehicleOdometry uses [w, x, y, z]; ROS uses x/y/z/w.
        message.pose.pose.orientation.w = float(source.q[0])
        message.pose.pose.orientation.x = float(source.q[1])
        message.pose.pose.orientation.y = float(source.q[2])
        message.pose.pose.orientation.z = float(source.q[3])
        message.twist.twist.linear.x = float(source.velocity[0])
        message.twist.twist.linear.y = float(source.velocity[1])
        message.twist.twist.linear.z = float(source.velocity[2])
        message.twist.twist.angular.x = float(source.angular_velocity[0])
        message.twist.twist.angular.y = float(source.angular_velocity[1])
        message.twist.twist.angular.z = float(source.angular_velocity[2])
        self._publisher.publish(message)


def main(args=None) -> int:
    """Run the read-only PX4-to-follower odometry bridge."""
    rclpy.init(args=args)
    node = Px4OdometryBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    main()
