"""ROS 2 adapter for the offline bounded trajectory-follower candidate."""

import math

from geometry_msgs.msg import PoseStamped, TwistStamped

from nav_msgs.msg import Odometry

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from std_msgs.msg import Bool

from std_srvs.srv import SetBool

from uav_interfaces.msg import TimedTrajectory, TrajectoryTrackingStatus

from uav_navigation.models import Point3D
from uav_navigation.tracking_models import (
    TrackingConfig,
    TrackingResult,
    TrackingState,
    VehicleState,
    VelocityCommand,
)
from uav_navigation.tracking_validator import validate_tracking_command
from uav_navigation.trajectory_models import TrajectoryPoint
from uav_navigation.trajectory_tracker import (
    OfflineTrackingController,
    hold_command,
)

TRAJECTORY_TOPIC = "/uav/trajectory/candidate"
VALIDITY_TOPIC = "/uav/trajectory/valid"
ODOMETRY_TOPIC = "/uav/vehicle/odometry"
COMMAND_TOPIC = "/uav/control/astar_command"
REFERENCE_POSE_TOPIC = "/uav/control/astar_reference_pose"
REFERENCE_TWIST_TOPIC = "/uav/control/astar_reference_twist"
TRACKING_STATUS_TOPIC = "/uav/control/astar_tracking_status"
SET_TRACKING_ENABLE_SERVICE = "/uav/control/set_tracking_enable"


def durable_qos() -> QoSProfile:
    """Match the Phase 4 transient-local trajectory contract."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def live_qos() -> QoSProfile:
    """Use reliable volatile history for continuously changing state."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _duration_seconds(duration) -> float:
    return float(duration.sec) + float(duration.nanosec) / 1e9


def _unsafe_point(x: float, y: float, z: float) -> Point3D:
    """Preserve malformed external state for the pure finite gate."""
    try:
        return Point3D(x, y, z)
    except ValueError:
        point = object.__new__(Point3D)
        object.__setattr__(point, "x", float(x))
        object.__setattr__(point, "y", float(y))
        object.__setattr__(point, "z", float(z))
        return point


def _yaw_from_quaternion(orientation) -> float:
    """Extract planar diagnostic yaw from a standard ROS quaternion."""
    sin_yaw = 2.0 * (
        orientation.w * orientation.z
        + orientation.x * orientation.y
    )
    cos_yaw = 1.0 - 2.0 * (
        orientation.y * orientation.y
        + orientation.z * orientation.z
    )
    return math.atan2(sin_yaw, cos_yaw)


class TrajectoryFollowerNode(Node):
    """Publish only independently validated ROS-level candidate commands."""

    def __init__(self) -> None:
        """Declare config, subscriptions, publishers, and control timer."""
        super().__init__("trajectory_follower")
        defaults = TrackingConfig()
        for name in defaults.__dataclass_fields__:
            self.declare_parameter(name, getattr(defaults, name))
        config = TrackingConfig(**{
            name: self.get_parameter(name).value
            for name in defaults.__dataclass_fields__
        })
        self.controller = OfflineTrackingController(config)
        self._command_publisher = self.create_publisher(
            TwistStamped, COMMAND_TOPIC, live_qos()
        )
        self._reference_pose_publisher = self.create_publisher(
            PoseStamped, REFERENCE_POSE_TOPIC, live_qos()
        )
        self._reference_twist_publisher = self.create_publisher(
            TwistStamped, REFERENCE_TWIST_TOPIC, live_qos()
        )
        self._status_publisher = self.create_publisher(
            TrajectoryTrackingStatus, TRACKING_STATUS_TOPIC, live_qos()
        )
        self.create_subscription(
            TimedTrajectory,
            TRAJECTORY_TOPIC,
            self._trajectory_callback,
            durable_qos(),
        )
        self.create_subscription(
            Bool, VALIDITY_TOPIC, self._validity_callback, durable_qos()
        )
        self._tracking_service = self.create_service(
            SetBool,
            SET_TRACKING_ENABLE_SERVICE,
            self._tracking_enable_callback,
        )
        self.create_subscription(
            Odometry, ODOMETRY_TOPIC, self._odometry_callback, live_qos()
        )
        self._timer = self.create_timer(config.control_period_s, self._tick)
        self.last_result: TrackingResult | None = None

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _trajectory_points(message: TimedTrajectory) -> tuple:
        points = []
        for item in message.points:
            points.append(TrajectoryPoint(
                time_from_start_s=_duration_seconds(item.time_from_start),
                position=Point3D(
                    item.position.x, item.position.y, item.position.z
                ),
                velocity=Point3D(
                    item.velocity.x, item.velocity.y, item.velocity.z
                ),
                acceleration=Point3D(
                    item.acceleration.x,
                    item.acceleration.y,
                    item.acceleration.z,
                ),
                jerk=Point3D(item.jerk.x, item.jerk.y, item.jerk.z),
                yaw_ned=item.yaw_ned,
                yaw_rate_radps=item.yaw_rate,
                yaw_acceleration_radps2=item.yaw_acceleration,
                arc_length_m=item.arc_length,
                curvature_inverse_m=item.curvature,
            ))
        return tuple(points)

    def _trajectory_callback(self, message: TimedTrajectory) -> None:
        now = self._now_seconds()
        try:
            accepted = self.controller.accept_trajectory(
                self._trajectory_points(message),
                message.header.frame_id,
                message.valid,
                now,
            )
        except (TypeError, ValueError, OverflowError) as error:
            self.controller.trajectory = None
            self.controller.previous_command = None
            self.get_logger().error(f"Rejected trajectory input: {error}")
            return
        if accepted:
            self.get_logger().info(
                "Accepted changed trajectory with "
                f"{len(message.points)} points"
            )
        else:
            self.get_logger().info("Ignored duplicate identical trajectory")

    def _validity_callback(self, message: Bool) -> None:
        self.controller.accept_validity(message.data, self._now_seconds())

    def _odometry_callback(self, message: Odometry) -> None:
        now = self._now_seconds()
        position = message.pose.pose.position
        velocity = message.twist.twist.linear
        state = VehicleState(
            timestamp_s=now,
            frame_id=message.header.frame_id,
            position=_unsafe_point(position.x, position.y, position.z),
            velocity=_unsafe_point(velocity.x, velocity.y, velocity.z),
            yaw_ned=_yaw_from_quaternion(message.pose.pose.orientation),
            yaw_rate_radps=message.twist.twist.angular.z,
        )
        self.controller.accept_odometry(state, now)

    def _tracking_enable_callback(self, request, response):
        accepted, message = self.controller.request_tracking_enable(
            request.data, self._now_seconds()
        )
        response.success = accepted
        response.message = message
        return response

    def _tick(self) -> None:
        now = self._now_seconds()
        try:
            result = self.controller.step(now)
        except (TypeError, ValueError, OverflowError) as error:
            command = hold_command(now, f"follower cycle failed: {error}")
            diagnostics = validate_tracking_command(
                command,
                self.config,
                TrackingState.HOLD_INVALID_COMMAND,
                self.controller.cycle_index,
            )
            if diagnostics:
                self.get_logger().error(
                    "refusing to publish an invalid fallback HOLD command: "
                    f"{diagnostics[0].constraint}"
                )
                return
            self._command_publisher.publish(self._command_message(command))
            self.get_logger().error(command.hold_reason)
            return
        self.last_result = result
        stamp = self.get_clock().now().to_msg()
        self._command_publisher.publish(
            self._command_message(result.selected_command, stamp)
        )
        if result.reference is not None:
            self._publish_reference(result, stamp)
        self._status_publisher.publish(self._status_message(result, stamp))

    @staticmethod
    def _command_message(command: VelocityCommand, stamp=None) -> TwistStamped:
        message = TwistStamped()
        if stamp is not None:
            message.header.stamp = stamp
        message.header.frame_id = command.frame_id
        message.twist.linear.x = command.linear.x
        message.twist.linear.y = command.linear.y
        message.twist.linear.z = command.linear.z
        message.twist.angular.z = command.yaw_rate_radps
        return message

    def _publish_reference(self, result: TrackingResult, stamp) -> None:
        point = result.reference.point
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "px4_ned"
        pose.pose.position.x = point.position.x
        pose.pose.position.y = point.position.y
        pose.pose.position.z = point.position.z
        pose.pose.orientation.z = math.sin(point.yaw_ned / 2.0)
        pose.pose.orientation.w = math.cos(point.yaw_ned / 2.0)
        twist = TwistStamped()
        twist.header = pose.header
        twist.twist.linear.x = point.velocity.x
        twist.twist.linear.y = point.velocity.y
        twist.twist.linear.z = point.velocity.z
        twist.twist.angular.z = point.yaw_rate_radps
        self._reference_pose_publisher.publish(pose)
        self._reference_twist_publisher.publish(twist)

    def _terminal_error(self) -> float:
        if (
            self.controller.trajectory is None
            or self.controller.odometry is None
        ):
            return 0.0
        final = self.controller.trajectory[-1].position
        measured = self.controller.odometry.position
        return math.sqrt(
            (final.x - measured.x) ** 2
            + (final.y - measured.y) ** 2
            + (final.z - measured.z) ** 2
        )

    def _status_message(self, result: TrackingResult, stamp):
        message = TrajectoryTrackingStatus()
        message.header.stamp = stamp
        message.header.frame_id = "px4_ned"
        message.state = result.state.value
        message.trajectory_valid = result.trajectory_valid
        message.odometry_valid = result.odometry_valid
        message.command_valid = result.command_valid
        command = result.selected_command
        message.hold_active = command.hold_active
        message.hold_reason = command.hold_reason
        message.trajectory_time = result.trajectory_time_s
        message.reference_index = (
            result.reference.reference_index
            if result.reference is not None else -1
        )
        if result.errors is not None:
            errors = result.errors
            message.position_error = errors.position_error_m
            message.horizontal_position_error = (
                errors.horizontal_position_error_m
            )
            message.vertical_position_error = errors.vertical_position_error_m
            message.along_track_error = errors.along_track_error_m
            message.cross_track_error = errors.cross_track_error_m
            message.velocity_error = errors.velocity_error_mps
            message.yaw_error = errors.yaw_error_rad
        message.command_speed = math.sqrt(
            command.linear.x**2
            + command.linear.y**2
            + command.linear.z**2
        )
        message.command_yaw_rate = command.yaw_rate_radps
        message.terminal_position_error = self._terminal_error()
        saturation_names = [
            name for name in result.saturations.__dataclass_fields__
            if getattr(result.saturations, name)
        ]
        message.status_message = (
            f"{result.status_message}|saturations="
            f"{','.join(saturation_names) or 'none'}|"
            f"diagnostics={len(result.diagnostics)}"
        )
        return message

    def destroy_node(self):
        """Publish the safely available shutdown HOLD before destruction."""
        if hasattr(self, "_command_publisher") and rclpy.ok():
            now = self._now_seconds()
            command = hold_command(now, "trajectory follower shutting down")
            self._command_publisher.publish(self._command_message(command))
        return super().destroy_node()


def main(args=None) -> int:
    """Run the offline trajectory-follower candidate node."""
    rclpy.init(args=args)
    node = TrajectoryFollowerNode()
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
