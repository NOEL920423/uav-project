"""ROS 2 adapter for expert-flight trajectory generation."""

import math

from nav_msgs.msg import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import Bool, String

from uav_interfaces.msg import (
    TimedTrajectory,
    TrajectoryPoint as TrajectoryPointMessage,
)

from uav_navigation.planner.models import Point3D
from uav_navigation.trajectory.trajectory_models import TrajectoryConfig
from uav_navigation.trajectory.trajectory_parameterizer import (
    parameterize_trajectory,
)

PATH_TOPIC = "/uav/planner/path"
CANDIDATE_TOPIC = "/uav/trajectory/candidate"
VALID_TOPIC = "/uav/trajectory/valid"
STATUS_TOPIC = "/uav/trajectory/status"


def _durable_qos() -> QoSProfile:
    """Return reliable transient-local QoS for finite offline graphs."""
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class TrajectoryParameterizerNode(Node):
    """Convert only the Phase 3 selected path into a validated candidate."""

    def __init__(self) -> None:
        """Declare limits and establish the three-output topic contract."""
        super().__init__("trajectory_parameterizer")
        defaults = TrajectoryConfig()
        for name in defaults.__dataclass_fields__:
            self.declare_parameter(name, getattr(defaults, name))
        self._config = TrajectoryConfig(
            **{
                name: self.get_parameter(name).value
                for name in defaults.__dataclass_fields__
            }
        )
        qos = _durable_qos()
        self._candidate_publisher = self.create_publisher(
            TimedTrajectory, CANDIDATE_TOPIC, qos
        )
        self._valid_publisher = self.create_publisher(Bool, VALID_TOPIC, qos)
        self._status_publisher = self.create_publisher(
            String, STATUS_TOPIC, qos
        )
        self.create_subscription(Path, PATH_TOPIC, self._receive_path, qos)
        self._last_signature = None
        self.parameterization_count = 0

    @staticmethod
    def _signature(message: Path) -> tuple:
        """Create a stable identity that ignores timestamps and orientation."""
        def canonical(value: float):
            if math.isnan(value):
                return "nan"
            if math.isinf(value):
                return "inf" if value > 0.0 else "-inf"
            return value

        return (
            message.header.frame_id,
            tuple(
                tuple(
                    canonical(value)
                    for value in (
                        pose.pose.position.x,
                        pose.pose.position.y,
                        pose.pose.position.z,
                    )
                )
                for pose in message.poses
            ),
        )

    def _receive_path(self, message: Path) -> None:
        """Parameterize one new path and publish post-validation truth."""
        signature = self._signature(message)
        if signature == self._last_signature:
            self.get_logger().info("Ignored identical planner path")
            return
        self._last_signature = signature
        self.parameterization_count += 1
        try:
            path = tuple(
                Point3D(
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z,
                )
                for pose in message.poses
            )
            result = parameterize_trajectory(
                path, message.header.frame_id, self._config
            )
        except (TypeError, ValueError, OverflowError) as error:
            result = None
            rejection = f"invalid input: {error}"
        if result is not None and result.trajectory_points:
            self._candidate_publisher.publish(
                self._to_message(result, message.header.frame_id)
            )
        valid = bool(result is not None and result.valid)
        self._valid_publisher.publish(Bool(data=valid))
        status = self._status(
            result,
            rejection if result is None else "",
            len(message.poses),
        )
        self._status_publisher.publish(String(data=status))
        if valid:
            self.get_logger().info(status)
        else:
            self.get_logger().warning(status)

    def _to_message(self, result, frame_id: str) -> TimedTrajectory:
        """Convert a finite pure result to the custom ROS interface."""
        message = TimedTrajectory()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = frame_id
        message.source_path_topic = PATH_TOPIC
        message.source_path_frame = frame_id
        message.trajectory_source = "PHASE4_TIME_PARAMETERIZED"
        message.valid = result.valid
        message.status_message = (
            result.status_message
            if result.valid
            else f"REJECTED: {result.rejection_reason}"
        )
        for point in result.trajectory_points:
            item = TrajectoryPointMessage()
            seconds = int(math.floor(point.time_from_start_s))
            item.time_from_start.sec = seconds
            item.time_from_start.nanosec = int(
                round((point.time_from_start_s - seconds) * 1e9)
            )
            if item.time_from_start.nanosec == 1_000_000_000:
                item.time_from_start.sec += 1
                item.time_from_start.nanosec = 0
            for target, source in (
                (item.position, point.position),
                (item.velocity, point.velocity),
                (item.acceleration, point.acceleration),
                (item.jerk, point.jerk),
            ):
                target.x, target.y, target.z = source.x, source.y, source.z
            item.yaw_ned = point.yaw_ned
            item.yaw_rate = point.yaw_rate_radps
            item.yaw_acceleration = point.yaw_acceleration_radps2
            item.arc_length = point.arc_length_m
            item.curvature = point.curvature_inverse_m
            message.points.append(item)
        return message

    @staticmethod
    def _status(result, fallback_reason: str, input_count: int) -> str:
        """Build controlled status fields required by offline automation."""
        if result is None:
            return (
                f"REJECTED|input_points={input_count}|trajectory_points=0|"
                "duration=0|"
                "time_scale=1|max_speed=0|max_acceleration=0|max_jerk=0|"
                f"max_yaw_rate=0|reason={fallback_reason}"
            )
        reason = result.rejection_reason or "none"
        return (
            f"{'SUCCESS' if result.valid else 'REJECTED'}|"
            f"input_points={result.source_path_point_count}|"
            f"trajectory_points={result.output_trajectory_point_count}|"
            f"duration={result.total_duration_s:.6f}|"
            f"time_scale={result.time_scale:.6f}|"
            f"max_speed={result.maximum_speed_mps:.6f}|"
            "max_acceleration="
            f"{result.maximum_longitudinal_acceleration_mps2:.6f}|"
            f"max_jerk={result.maximum_jerk_mps3:.6f}|"
            f"max_yaw_rate={result.maximum_yaw_rate_radps:.6f}|"
            f"reason={reason}"
        )


def _spin(node_type, args=None) -> int:
    rclpy.init(args=args)
    node = node_type()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        code = getattr(node, "exit_code", 0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


def main(args=None) -> int:
    """Run the trajectory parameterizer node."""
    return _spin(TrajectoryParameterizerNode, args)


if __name__ == "__main__":
    main()
