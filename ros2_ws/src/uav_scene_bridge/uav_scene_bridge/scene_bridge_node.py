"""Validate Isaac runtime state and publish coherent typed scene snapshots."""

import json
import math

from geometry_msgs.msg import PoseStamped

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from std_msgs.msg import String

from uav_interfaces.msg import Obstacle, ObstacleArray

from uav_scene_bridge.runtime_contract import (
    ISAAC_FRAME,
    RuntimeSnapshot,
    parse_runtime_snapshot,
)


ISAAC_POSE_TOPIC = "/isaac_uav/pose"
ISAAC_STATUS_TOPIC = "/uav/isaac/runtime_status"
BRIDGE_STATUS_TOPIC = "/uav/isaac/bridge_status"
SCENE_OBSTACLES_TOPIC = "/uav/isaac/scene/obstacles"
SCENE_START_TOPIC = "/uav/isaac/scene/start"
SCENE_GOAL_TOPIC = "/uav/isaac/scene/goal"


def _durable_qos() -> QoSProfile:
    """Return reliable transient-local depth-one scene QoS."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _pose_is_valid(message: PoseStamped) -> bool:
    """Require the documented frame and a finite position/quaternion."""
    values = (
        message.pose.position.x,
        message.pose.position.y,
        message.pose.position.z,
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
        message.pose.orientation.w,
    )
    return message.header.frame_id == ISAAC_FRAME and all(
        math.isfinite(float(value)) for value in values
    )


class SceneBridgeNode(Node):
    """Fail closed until fresh Isaac pose and runtime heartbeat agree."""

    def __init__(self) -> None:
        """Initialize an explicitly enabled simulator-data boundary."""
        super().__init__("scene_bridge")
        self.declare_parameter("enable_scene_access", False)
        self.declare_parameter("runtime_timeout_s", 0.50)
        self._enabled = bool(
            self.get_parameter("enable_scene_access").value
        )
        self._timeout_s = float(self.get_parameter("runtime_timeout_s").value)
        if self._timeout_s <= 0.0 or not math.isfinite(self._timeout_s):
            raise ValueError("runtime_timeout_s must be finite and positive")
        self._snapshot: RuntimeSnapshot | None = None
        self._pose: PoseStamped | None = None
        self._status_receipt_s: float | None = None
        self._pose_receipt_s: float | None = None
        self._last_sequence = -1
        self._last_published_scene: tuple[str, int] | None = None
        self._last_error = "scene access disabled"

        scene_qos = _durable_qos()
        live_qos = QoSProfile(depth=10)
        self._obstacles_publisher = self.create_publisher(
            ObstacleArray, SCENE_OBSTACLES_TOPIC, scene_qos
        )
        self._start_publisher = self.create_publisher(
            PoseStamped, SCENE_START_TOPIC, scene_qos
        )
        self._goal_publisher = self.create_publisher(
            PoseStamped, SCENE_GOAL_TOPIC, scene_qos
        )
        self._bridge_status_publisher = self.create_publisher(
            String, BRIDGE_STATUS_TOPIC, live_qos
        )
        self.create_subscription(
            PoseStamped, ISAAC_POSE_TOPIC, self._pose_callback, live_qos
        )
        self.create_subscription(
            String, ISAAC_STATUS_TOPIC, self._status_callback, live_qos
        )
        self._timer = self.create_timer(0.10, self._tick)
        if self._enabled:
            self.get_logger().warning(
                "Isaac scene access enabled; waiting for fresh runtime data"
            )
        else:
            self.get_logger().info(
                "Scene bridge remains fail-closed because access is disabled"
            )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _pose_callback(self, message: PoseStamped) -> None:
        if not self._enabled:
            return
        if not _pose_is_valid(message):
            self._pose = None
            self._last_error = "invalid Isaac pose"
            return
        self._pose = message
        self._pose_receipt_s = self._now_seconds()

    def _status_callback(self, message: String) -> None:
        if not self._enabled:
            return
        try:
            snapshot = parse_runtime_snapshot(message.data)
        except ValueError as error:
            self._snapshot = None
            self._last_error = str(error)
            return
        if snapshot.sequence <= self._last_sequence:
            self._snapshot = None
            self._last_error = "runtime heartbeat sequence did not increase"
            return
        self._last_sequence = snapshot.sequence
        self._snapshot = snapshot
        self._status_receipt_s = self._now_seconds()
        self._last_error = "" if snapshot.ready else "Isaac runtime not ready"

    def _ages(self, now: float) -> tuple[float, float]:
        pose_age = math.inf
        status_age = math.inf
        if self._pose_receipt_s is not None:
            pose_age = max(0.0, now - self._pose_receipt_s)
        if self._status_receipt_s is not None:
            status_age = max(0.0, now - self._status_receipt_s)
        return pose_age, status_age

    def _ready(self, now: float) -> bool:
        pose_age, status_age = self._ages(now)
        return bool(
            self._enabled
            and self._snapshot is not None
            and self._snapshot.ready
            and self._pose is not None
            and pose_age <= self._timeout_s
            and status_age <= self._timeout_s
        )

    def _tick(self) -> None:
        now = self._now_seconds()
        ready = self._ready(now)
        if ready:
            self._publish_scene_once()
        pose_age, status_age = self._ages(now)
        status = {
            "ready": ready,
            "enabled": self._enabled,
            "pose_age_s": pose_age if math.isfinite(pose_age) else None,
            "runtime_status_age_s": (
                status_age if math.isfinite(status_age) else None
            ),
            "scene_id": (
                "" if self._snapshot is None else self._snapshot.scene_id
            ),
            "scene_revision": (
                0 if self._snapshot is None else self._snapshot.scene_revision
            ),
            "runtime_sequence": self._last_sequence,
            "error": "" if ready else self._last_error or "runtime stale",
        }
        message = String()
        message.data = json.dumps(
            status, sort_keys=True, separators=(",", ":")
        )
        self._bridge_status_publisher.publish(message)

    def _publish_scene_once(self) -> None:
        assert self._snapshot is not None
        assert self._pose is not None
        scene_key = (
            self._snapshot.scene_id,
            self._snapshot.scene_revision,
        )
        if scene_key == self._last_published_scene:
            return
        stamp = self.get_clock().now().to_msg()
        obstacles = ObstacleArray()
        obstacles.header.stamp = stamp
        obstacles.header.frame_id = ISAAC_FRAME
        for source in self._snapshot.obstacles:
            item = Obstacle()
            item.name = source.name
            item.center.x = source.x
            item.center.y = source.y
            item.center.z = source.z
            item.radius = source.radius
            item.height = source.height
            obstacles.obstacles.append(item)
        start = PoseStamped()
        start.header.stamp = stamp
        start.header.frame_id = ISAAC_FRAME
        start.pose = self._pose.pose
        goal = PoseStamped()
        goal.header.stamp = stamp
        goal.header.frame_id = ISAAC_FRAME
        goal.pose.position.x = self._snapshot.goal[0]
        goal.pose.position.y = self._snapshot.goal[1]
        goal.pose.position.z = self._snapshot.goal[2]
        goal.pose.orientation.w = 1.0
        self._obstacles_publisher.publish(obstacles)
        self._start_publisher.publish(start)
        self._goal_publisher.publish(goal)
        self._last_published_scene = scene_key
        self.get_logger().info(
            f"ISAAC_SCENE_READY id={scene_key[0]} revision={scene_key[1]} "
            f"obstacles={len(obstacles.obstacles)}"
        )


def main(args=None) -> None:
    """Run the fail-closed Isaac scene boundary until ROS shutdown."""
    rclpy.init(args=args)
    node = SceneBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
