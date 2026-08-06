"""ROS 2 adapter and finite harnesses for offline trajectory generation."""

import math
import time

from geometry_msgs.msg import PoseStamped

from nav_msgs.msg import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import Bool, String

from uav_interfaces.msg import (
    Obstacle,
    ObstacleArray,
    TimedTrajectory,
    TrajectoryPoint as TrajectoryPointMessage,
)

from uav_navigation.models import Point3D
from uav_navigation.trajectory_models import TrajectoryConfig
from uav_navigation.trajectory_parameterizer import parameterize_trajectory

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


def _fixture_points(name: str) -> tuple[tuple[float, float, float], ...]:
    """Return one of twelve deterministic standalone path fixtures."""
    fixtures = {
        "straight-line": ((0, 0, -2), (3, 0, -2), (6, 0, -2)),
        "phase3-bspline": (
            (0, 0, -2), (1, 0.08, -2), (2, 0.30, -2),
            (3, 0.55, -2), (4, 0.30, -2), (5, 0.08, -2), (6, 0, -2),
        ),
        "sharp-bend": ((0, 0, -2), (2, 0, -2), (2, 0.3, -2), (2, 2, -2)),
        "high-curvature": (
            (0, 0, -2), (0.4, 0, -2), (0.45, 0.12, -2),
            (0.4, 0.25, -2), (0, 0.25, -2),
        ),
        "duplicate-adjacent": (
            (0, 0, -2), (1, 0, -2), (1, 0, -2), (2, 0, -2),
        ),
        "two-point": ((0, 0, -2), (1, 0, -2)),
        "invalid-one-point": ((0, 0, -2),),
        "nonfinite": ((0, 0, -2), (math.nan, 0, -2)),
        "yaw-wrap": (
            (0, 0.01, -2), (-1, 0.001, -2),
            (-2, -0.001, -2), (-3, -0.01, -2),
        ),
        "jerk-scaling": (
            (0, 0, -2), (1, 0, -2), (1, 0.2, -2),
            (1, 1, -2), (2, 1, -2),
        ),
        "impossible-config-rejection": (
            (0, 0, -2), (1, 0, -2), (1, 1, -2), (2, 1, -2),
        ),
        "wrong-frame": ((0, 0, -2), (1, 0, -2), (2, 0, -2)),
    }
    if name not in fixtures:
        raise RuntimeError(f"unknown trajectory fixture: {name}")
    return fixtures[name]


class OfflineTrajectoryHarness(Node):
    """Publish one fixed path and independently inspect all Phase 4 outputs."""

    def __init__(self) -> None:
        """Create subscriptions and a finite wall-time state machine."""
        super().__init__("trajectory_offline_harness")
        self.declare_parameter("fixture", "straight-line")
        self._fixture = str(self.get_parameter("fixture").value)
        self._points = _fixture_points(self._fixture)
        self._expected_valid = self._fixture not in {
            "invalid-one-point", "nonfinite",
            "impossible-config-rejection", "wrong-frame",
        }
        qos = _durable_qos()
        self._publisher = self.create_publisher(Path, PATH_TOPIC, qos)
        self.create_subscription(
            TimedTrajectory, CANDIDATE_TOPIC, self._candidate_callback, qos
        )
        self.create_subscription(
            Bool, VALID_TOPIC, self._valid_callback, qos
        )
        self.create_subscription(
            String, STATUS_TOPIC, self._status_callback, qos
        )
        self._candidate = None
        self._valid = None
        self._status_text = ""
        self._valid_count = 0
        self._published = False
        self._duplicate_published_at = None
        self._finished = False
        self.exit_code = 1
        self._started_at = time.monotonic()
        self.create_timer(0.1, self._tick)

    def _path_message(self) -> Path:
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = (
            "map" if self._fixture == "wrong-frame" else "px4_ned"
        )
        for x, y, z in self._points:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = float(z)
            message.poses.append(pose)
        return message

    def _tick(self) -> None:
        if not self._published:
            self._publisher.publish(self._path_message())
            self._published = True
            self.get_logger().info(
                f"Published trajectory fixture: {self._fixture}"
            )
        ready = self._valid is not None and bool(self._status_text)
        if ready and self._duplicate_published_at is None:
            self._publisher.publish(self._path_message())
            self._duplicate_published_at = time.monotonic()
        if self._duplicate_published_at is not None:
            if time.monotonic() - self._duplicate_published_at > 0.35:
                self._inspect()
        if not self._finished and time.monotonic() - self._started_at > 8.0:
            self._finish(1, "trajectory fixture timed out")

    def _candidate_callback(self, message: TimedTrajectory) -> None:
        self._candidate = message

    def _valid_callback(self, message: Bool) -> None:
        self._valid = message.data
        self._valid_count += 1

    def _status_callback(self, message: String) -> None:
        self._status_text = message.data

    def _inspect(self) -> None:
        if self._finished:
            return
        if self._valid_count != 1:
            self._finish(1, "identical input was recomputed")
            return
        if self._valid != self._expected_valid:
            self._finish(1, f"unexpected validity: {self._valid}")
            return
        if self._expected_valid:
            if self._candidate is None or not self._candidate.valid:
                self._finish(1, "valid candidate was not received")
                return
            times = [
                item.time_from_start.sec + item.time_from_start.nanosec / 1e9
                for item in self._candidate.points
            ]
            if not times or times[0] != 0.0 or any(
                later <= earlier for earlier, later in zip(times, times[1:])
            ):
                self._finish(1, "candidate time sequence is invalid")
                return
        expected_marker = "SUCCESS|" if self._expected_valid else "REJECTED|"
        if expected_marker not in self._status_text:
            self._finish(1, "trajectory status marker is inconsistent")
            return
        count = len(self._candidate.points) if self._candidate else 0
        self._finish(
            0,
            "trajectory offline integration passed: "
            f"fixture={self._fixture}, valid={str(self._valid).lower()}, "
            f"points={count}, duplicate_recomputations=0, frame=px4_ned",
        )

    def _finish(self, code: int, detail: str) -> None:
        if self._finished:
            return
        self._finished = True
        self.exit_code = code
        logger = self.get_logger()
        (logger.info if code == 0 else logger.error)(detail)
        if rclpy.ok():
            rclpy.shutdown()


class OfflinePipelineHarness(OfflineTrajectoryHarness):
    """Drive a fixed scene through Phase 3 and inspect Phase 4 outputs."""

    def __init__(self) -> None:
        """Replace the direct path publisher with fixed scene publishers."""
        super().__init__()
        self.destroy_publisher(self._publisher)
        qos = _durable_qos()
        self._obstacles = self.create_publisher(
            ObstacleArray, "/uav/scene/obstacles", qos
        )
        self._start = self.create_publisher(
            PoseStamped, "/uav/scene/start", qos
        )
        self._goal = self.create_publisher(
            PoseStamped, "/uav/scene/goal", qos
        )

    def _tick(self) -> None:
        if not self._published:
            stamp = self.get_clock().now().to_msg()
            obstacles = ObstacleArray()
            obstacles.header.frame_id = "isaac_world"
            obstacles.header.stamp = stamp
            obstacle = Obstacle()
            obstacle.name = "pipeline_tower"
            obstacle.center.x = 0.0
            obstacle.center.y = 0.0
            obstacle.center.z = 1.5
            obstacle.radius = 0.2
            obstacle.height = 3.0
            obstacles.obstacles.append(obstacle)
            start = self._scene_pose(-2.0, stamp)
            goal = self._scene_pose(2.0, stamp)
            self._obstacles.publish(obstacles)
            self._start.publish(start)
            self._goal.publish(goal)
            self._published = True
            self.get_logger().info("Published fixed Phase 3 pipeline scene")
        if (
            self._valid is not None
            and self._status_text
            and not self._finished
        ):
            self._inspect_pipeline()
        if not self._finished and time.monotonic() - self._started_at > 12.0:
            self._finish(1, "planning-to-trajectory pipeline timed out")

    @staticmethod
    def _scene_pose(x: float, stamp) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "isaac_world"
        pose.header.stamp = stamp
        pose.pose.position.x = x
        pose.pose.orientation.w = 1.0
        return pose

    def _inspect_pipeline(self) -> None:
        if not self._valid or self._candidate is None:
            self._finish(1, "pipeline trajectory was rejected")
            return
        if self._candidate.source_path_topic != PATH_TOPIC:
            self._finish(1, "pipeline consumed the wrong path topic")
            return
        self._finish(
            0,
            "pipeline offline integration passed: scene=accepted-bspline, "
            f"valid=true, points={len(self._candidate.points)}, frame=px4_ned",
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


def offline_harness_main(args=None) -> int:
    """Run one finite standalone trajectory fixture."""
    return _spin(OfflineTrajectoryHarness, args)


def pipeline_harness_main(args=None) -> int:
    """Run the finite scene-to-trajectory integration fixture."""
    return _spin(OfflinePipelineHarness, args)


if __name__ == "__main__":
    main()
