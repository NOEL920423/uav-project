"""Finite ROS nodes for deterministic Phase 5 offline tracking graphs."""

import math
import time

from geometry_msgs.msg import PoseStamped, TwistStamped

from nav_msgs.msg import Odometry

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import Bool

from uav_interfaces.msg import (
    Obstacle,
    ObstacleArray,
    TimedTrajectory,
    TrajectoryPoint as TrajectoryPointMessage,
    TrajectoryTrackingStatus,
)

from uav_navigation.models import Point3D
from uav_navigation.offline_kinematic_plant import (
    KinematicPlantConfig,
    OfflineKinematicPlant,
)
from uav_navigation.tracking_fixtures import tracking_fixture
from uav_navigation.tracking_models import VelocityCommand
from uav_navigation.trajectory_follower_node import (
    COMMAND_TOPIC,
    ODOMETRY_TOPIC,
    REFERENCE_POSE_TOPIC,
    REFERENCE_TWIST_TOPIC,
    TRACKING_STATUS_TOPIC,
    TRAJECTORY_TOPIC,
    VALIDITY_TOPIC,
    durable_qos,
    live_qos,
)
from uav_navigation.trajectory_parameterizer import parameterize_trajectory


def _timed_message(node: Node, fixture) -> TimedTrajectory:
    result = parameterize_trajectory(fixture.path)
    if not result.valid:
        raise RuntimeError(
            f"fixture parameterization failed: {result.rejection_reason}"
        )
    message = TimedTrajectory()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = "px4_ned"
    message.source_path_topic = "/phase5/fixed_fixture"
    message.source_path_frame = "px4_ned"
    message.trajectory_source = fixture.name
    message.valid = fixture.validity_mode != "false"
    message.status_message = "Phase 5 deterministic offline fixture"
    for point in result.trajectory_points:
        item = TrajectoryPointMessage()
        seconds = int(math.floor(point.time_from_start_s))
        nanoseconds = int(round((point.time_from_start_s - seconds) * 1e9))
        if nanoseconds == 1_000_000_000:
            seconds += 1
            nanoseconds = 0
        item.time_from_start.sec = seconds
        item.time_from_start.nanosec = nanoseconds
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


class FixedTrajectoryPublisher(Node):
    """Publish one fixed trajectory and controlled validity heartbeat."""

    def __init__(self) -> None:
        """Create durable publishers and a finite fixture timer."""
        super().__init__("fixed_tracking_trajectory_publisher")
        self.declare_parameter("fixture", "straight-trajectory")
        self.fixture = tracking_fixture(
            str(self.get_parameter("fixture").value)
        )
        self._trajectory_publisher = self.create_publisher(
            TimedTrajectory, TRAJECTORY_TOPIC, durable_qos()
        )
        self._validity_publisher = self.create_publisher(
            Bool, VALIDITY_TOPIC, durable_qos()
        )
        self._message = _timed_message(self, self.fixture)
        self._started = time.monotonic()
        self._published = False
        self._duplicated = False
        self.create_timer(0.1, self._tick)

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._started
        if not self._published:
            self._trajectory_publisher.publish(self._message)
            self._published = True
            self.get_logger().info(
                f"Published fixed tracking fixture: {self.fixture.name}"
            )
        mode = self.fixture.validity_mode
        if mode != "stale" or elapsed < 0.35:
            self._validity_publisher.publish(Bool(data=mode != "false"))
        if mode == "duplicate" and elapsed > 0.30 and not self._duplicated:
            self._trajectory_publisher.publish(self._message)
            self._duplicated = True
            self.get_logger().info("Published duplicate identical trajectory")


class FixedTrackingScenePublisher(Node):
    """Publish one fixed scene for the complete Phase 2-to-5 pipeline."""

    def __init__(self) -> None:
        """Create scene publishers without starting a simulator."""
        super().__init__("fixed_tracking_scene_publisher")
        qos = durable_qos()
        self._obstacles = self.create_publisher(
            ObstacleArray, "/uav/scene/obstacles", qos
        )
        self._start = self.create_publisher(
            PoseStamped, "/uav/scene/start", qos
        )
        self._goal = self.create_publisher(
            PoseStamped, "/uav/scene/goal", qos
        )
        self._published = False
        self.create_timer(0.1, self._tick)

    @staticmethod
    def _pose(east: float, stamp) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "isaac_world"
        pose.header.stamp = stamp
        pose.pose.position.x = east
        pose.pose.orientation.w = 1.0
        return pose

    def _tick(self) -> None:
        if self._published:
            return
        stamp = self.get_clock().now().to_msg()
        obstacles = ObstacleArray()
        obstacles.header.frame_id = "isaac_world"
        obstacles.header.stamp = stamp
        obstacle = Obstacle()
        obstacle.name = "phase5_pipeline_tower"
        obstacle.center.z = 1.5
        obstacle.radius = 0.2
        obstacle.height = 3.0
        obstacles.obstacles.append(obstacle)
        self._obstacles.publish(obstacles)
        self._start.publish(self._pose(-2.0, stamp))
        self._goal.publish(self._pose(2.0, stamp))
        self._published = True
        self.get_logger().info("Published fixed Phase 5 full-pipeline scene")


class OfflineKinematicPlantNode(Node):
    """Adapt candidate Twist messages to the pure deterministic plant."""

    def __init__(self) -> None:
        """Create one fixture-configured plant and odometry publisher."""
        super().__init__("offline_kinematic_plant")
        self.declare_parameter("fixture", "straight-trajectory")
        self.declare_parameter("full_pipeline", False)
        self.declare_parameter("command_topic", COMMAND_TOPIC)
        self.fixture = tracking_fixture(
            str(self.get_parameter("fixture").value)
        )
        initial_position = self.fixture.initial_position
        if bool(self.get_parameter("full_pipeline").value):
            initial_position = Point3D(0.0, -2.0, -2.0)
        config = KinematicPlantConfig(
            initial_position=initial_position,
            disturbance_velocity=self.fixture.disturbance_velocity,
        )
        self.plant = OfflineKinematicPlant(config)
        self._command = VelocityCommand(
            0.0, "px4_ned", Point3D(0.0, 0.0, 0.0), 0.0,
            hold_active=True, hold_reason="plant startup",
        )
        self._odometry_publisher = self.create_publisher(
            Odometry, ODOMETRY_TOPIC, live_qos()
        )
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("command_topic").value),
            self._command_callback,
            live_qos(),
        )
        self._started = time.monotonic()
        self.create_timer(config.integration_timestep_s, self._tick)

    def _command_callback(self, message: TwistStamped) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        self._command = VelocityCommand(
            now,
            message.header.frame_id,
            Point3D(
                message.twist.linear.x,
                message.twist.linear.y,
                message.twist.linear.z,
            ),
            message.twist.angular.z,
        )

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._started
        if self.fixture.odometry_mode == "stale" and elapsed > 0.60:
            return
        if self.fixture.odometry_mode != "frozen":
            self.plant.step(self._command)
        now = self.get_clock().now().nanoseconds / 1e9
        state = self.plant.measurement(now)
        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = (
            "map"
            if self.fixture.odometry_mode == "wrong-frame" else "px4_ned"
        )
        message.child_frame_id = "phase5_kinematic_fixture"
        message.pose.pose.position.x = state.position.x
        message.pose.pose.position.y = state.position.y
        message.pose.pose.position.z = state.position.z
        if self.fixture.odometry_mode == "nonfinite":
            message.pose.pose.position.x = math.nan
        message.pose.pose.orientation.z = math.sin(state.yaw_ned / 2.0)
        message.pose.pose.orientation.w = math.cos(state.yaw_ned / 2.0)
        message.twist.twist.linear.x = state.velocity.x
        message.twist.twist.linear.y = state.velocity.y
        message.twist.twist.linear.z = state.velocity.z
        message.twist.twist.angular.z = state.yaw_rate_radps
        self._odometry_publisher.publish(message)


class TrackingResultMonitor(Node):
    """Independently verify finite graph outputs and terminate the launch."""

    def __init__(self) -> None:
        """Subscribe to every required output and initialize metric sums."""
        super().__init__("tracking_result_monitor")
        self.declare_parameter("fixture", "straight-trajectory")
        self.declare_parameter("full_pipeline", False)
        self.fixture = tracking_fixture(
            str(self.get_parameter("fixture").value)
        )
        self._full_pipeline = bool(
            self.get_parameter("full_pipeline").value
        )
        qos = live_qos()
        self.create_subscription(
            TwistStamped, COMMAND_TOPIC, self._command_callback, qos
        )
        self.create_subscription(
            PoseStamped, REFERENCE_POSE_TOPIC, self._pose_callback, qos
        )
        self.create_subscription(
            TwistStamped, REFERENCE_TWIST_TOPIC, self._twist_callback, qos
        )
        self.create_subscription(
            TrajectoryTrackingStatus,
            TRACKING_STATUS_TOPIC,
            self._status_callback,
            qos,
        )
        self._started = time.monotonic()
        self._finished = False
        self.exit_code = 1
        self._last_command = None
        self._last_status = None
        self._pose_count = 0
        self._twist_count = 0
        self._tracking_seen = False
        self._saturation_seen = False
        self._contract_error = ""
        self._samples = 0
        self._position_square = 0.0
        self._max_position = 0.0
        self._hold_cycles = 0
        self.create_timer(0.1, self._tick)

    def _command_callback(self, message: TwistStamped) -> None:
        self._last_command = message
        if message.header.frame_id != "px4_ned":
            self._contract_error = "command frame is not px4_ned"
        values = (
            message.twist.linear.x,
            message.twist.linear.y,
            message.twist.linear.z,
            message.twist.angular.x,
            message.twist.angular.y,
            message.twist.angular.z,
        )
        if not all(math.isfinite(value) for value in values):
            self._contract_error = "command contains non-finite values"
        speed = math.sqrt(sum(value * value for value in values[:3]))
        if speed > 2.0 + 1e-6 or abs(values[5]) > 1.5 + 1e-6:
            self._contract_error = "command exceeded configured bounds"
        if values[3] != 0.0 or values[4] != 0.0:
            self._contract_error = "command angular x/y are not zero"

    def _pose_callback(self, message: PoseStamped) -> None:
        self._pose_count += 1
        if message.header.frame_id != "px4_ned":
            self._contract_error = "reference pose frame is not px4_ned"

    def _twist_callback(self, message: TwistStamped) -> None:
        self._twist_count += 1
        if message.header.frame_id != "px4_ned":
            self._contract_error = "reference twist frame is not px4_ned"

    def _status_callback(self, message: TrajectoryTrackingStatus) -> None:
        self._last_status = message
        self._tracking_seen |= message.state in {"TRACKING", "GOAL_SETTLING"}
        self._saturation_seen |= (
            "saturations=none" not in message.status_message
        )
        self._samples += 1
        self._position_square += message.position_error**2
        self._max_position = max(self._max_position, message.position_error)
        self._hold_cycles += int(message.hold_active)
        if message.header.frame_id != "px4_ned":
            self._contract_error = "tracking status frame is not px4_ned"

    def _tick(self) -> None:
        if self._finished:
            return
        elapsed = time.monotonic() - self._started
        if self._contract_error:
            self._finish(1, self._contract_error)
            return
        status = self._last_status
        if status is not None:
            expected = self.fixture.expected
            if expected in {"SUCCESS", "SUCCESS_WITH_SATURATION"}:
                if status.state == "GOAL_HOLD":
                    if not self._tracking_seen:
                        self._finish(1, "GOAL_HOLD occurred without tracking")
                        return
                    if expected.endswith("SATURATION") and not (
                        self._saturation_seen
                    ):
                        self._finish(1, "required saturation was not observed")
                        return
                    self._finish(0, "closed-loop goal settled")
                    return
            elif status.state == expected:
                command = self._last_command
                if command is None or not status.hold_active:
                    self._finish(1, "expected HOLD command was not observed")
                    return
                magnitude = sum(abs(value) for value in (
                    command.twist.linear.x,
                    command.twist.linear.y,
                    command.twist.linear.z,
                    command.twist.angular.z,
                ))
                if magnitude > 1e-9 or not status.hold_reason:
                    self._finish(1, "HOLD was nonzero or lacked a reason")
                    return
                self._finish(0, f"expected safety state {expected}")
                return
        if elapsed > 18.0:
            self._finish(1, "tracking graph timed out")

    def _finish(self, code: int, detail: str) -> None:
        if self._finished:
            return
        topics = dict(self.get_topic_names_and_types())
        if any(name.startswith("/fmu/in/") for name in topics):
            code, detail = 1, "forbidden /fmu/in/* topic detected"
        self._finished = True
        self.exit_code = code
        rmse = math.sqrt(self._position_square / max(1, self._samples))
        marker = "full pipeline" if self._full_pipeline else "tracking"
        text = (
            f"{marker} offline integration passed: "
            f"fixture={self.fixture.name}, "
            f"expected={self.fixture.expected}, detail={detail}, "
            f"commands={'yes' if self._last_command else 'no'}, "
            f"references={self._pose_count}/{self._twist_count}, frame=px4_ned"
        )
        metrics = (
            "Offline Closed-Loop Tracking Metrics|"
            f"position_rmse={rmse:.6f}|max_position_error="
            f"{self._max_position:.6f}|hold_cycles={self._hold_cycles}|"
            f"samples={self._samples}|completion={self.fixture.expected}"
        )
        logger = self.get_logger()
        if code == 0:
            logger.info(text)
            logger.info(metrics)
        else:
            logger.error(f"tracking integration failed: {detail}")
        if rclpy.ok():
            rclpy.shutdown()


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


def trajectory_publisher_main(args=None) -> int:
    """Run the fixed direct-trajectory publisher."""
    return _spin(FixedTrajectoryPublisher, args)


def scene_publisher_main(args=None) -> int:
    """Run the fixed scene publisher for full integration."""
    return _spin(FixedTrackingScenePublisher, args)


def plant_main(args=None) -> int:
    """Run the ROS adapter around the deterministic pure plant."""
    return _spin(OfflineKinematicPlantNode, args)


def monitor_main(args=None) -> int:
    """Run the finite independent tracking result monitor."""
    return _spin(TrackingResultMonitor, args)
