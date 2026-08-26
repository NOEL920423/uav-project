"""Observe one live BC rollout, terminate it, and save measured metrics."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from geometry_msgs.msg import PoseStamped

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

from std_msgs.msg import String

from uav_interfaces.msg import ObstacleArray

from uav_ml.inference.bc_flight_contract import yaw_from_quaternion

from uav_px4_control.bc_episode_monitor import (
    TerminationConfig,
    cylinder_clearance_m,
    select_terminal_reason,
)
from uav_px4_control.bc_flight_models import BcFlightState
from uav_px4_control.bc_flight_supervisor_node import (
    EPISODE_TERMINATION_TOPIC,
    SUPERVISOR_STATUS_SCHEMA,
    SUPERVISOR_STATUS_TOPIC,
    TERMINATION_SCHEMA,
)
from uav_px4_control.bc_policy_node import (
    POLICY_STATUS_SCHEMA,
    POLICY_STATUS_TOPIC,
)
from uav_px4_control.control_mux_node import control_qos


SCENE_GOAL_TOPIC = "/uav/scene/goal"
SCENE_OBSTACLES_TOPIC = "/uav/scene/obstacles"
ODOMETRY_TOPIC = "/uav/vehicle/odometry"
RESULT_SCHEMA = "uav_bc_flight_result/v1"
COLLISION_DETECTOR = "geometric_cylinder_v1"

MSG_RESULT_SAVED = "[BC Flight] Result saved: {path}"
MSG_MONITOR_FAILED = "[BC Flight] Monitor failed: {error}"


def scene_qos() -> QoSProfile:
    """Receive the latest coherent scene even after it was first published."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class BcEpisodeMonitorNode(Node):
    """Keep evaluation metrics out of policy inference and source selection."""

    def __init__(self) -> None:
        """Subscribe to measured flight evidence and configure termination."""
        super().__init__("bc_episode_monitor")
        self.declare_parameter("result_path", "/tmp/uav_bc_flight_result.json")
        self.declare_parameter("episode", 1)
        self.declare_parameter("seed", 0)
        self.declare_parameter("image_source", "top_rgb")
        defaults = TerminationConfig()
        for name in defaults.__dataclass_fields__:
            self.declare_parameter(name, getattr(defaults, name))
        self.config = TerminationConfig(**{
            name: self.get_parameter(name).value
            for name in defaults.__dataclass_fields__
        })
        self._result_path = Path(
            str(self.get_parameter("result_path").value)
        ).expanduser().resolve()
        self._episode = int(self.get_parameter("episode").value)
        self._seed = int(self.get_parameter("seed").value)
        self._image_source = str(self.get_parameter("image_source").value)
        self._goal: PoseStamped | None = None
        self._obstacles: ObstacleArray | None = None
        self._odometry: Odometry | None = None
        self._supervisor: dict = {}
        self._policy: dict = {}
        self._episode_started_s: float | None = None
        self._bc_started_s: float | None = None
        self._bc_ended_s: float | None = None
        self._finished_s: float | None = None
        self._terminal_reason = ""
        self._minimum_goal_distance_m = math.inf
        self._minimum_clearance_m = math.inf
        self._path_length_m = 0.0
        self._previous_path_position: tuple[float, float] | None = None
        self._result_written = False
        self._last_error = ""

        qos = control_qos()
        self._termination_publisher = self.create_publisher(
            String, EPISODE_TERMINATION_TOPIC, qos
        )
        self.create_subscription(
            String, SUPERVISOR_STATUS_TOPIC, self._supervisor_callback, qos
        )
        self.create_subscription(
            String, POLICY_STATUS_TOPIC, self._policy_callback, qos
        )
        self.create_subscription(
            Odometry, ODOMETRY_TOPIC, self._odometry_callback, qos
        )
        durable = scene_qos()
        self.create_subscription(
            PoseStamped, SCENE_GOAL_TOPIC, self._goal_callback, durable
        )
        self.create_subscription(
            ObstacleArray,
            SCENE_OBSTACLES_TOPIC,
            self._obstacles_callback,
            durable,
        )
        self._timer = self.create_timer(0.10, self._tick)

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _supervisor_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get("schema") != SUPERVISOR_STATUS_SCHEMA:
                return
            self._supervisor = payload
            if self._episode_started_s is None:
                self._episode_started_s = self._now_seconds()
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    def _policy_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get("schema") == POLICY_STATUS_SCHEMA:
                self._policy = payload
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    def _goal_callback(self, message: PoseStamped) -> None:
        self._goal = message

    def _obstacles_callback(self, message: ObstacleArray) -> None:
        self._obstacles = message

    def _odometry_callback(self, message: Odometry) -> None:
        self._odometry = message
        if self._supervisor.get("state") != BcFlightState.NAVIGATING.value:
            return
        north = float(message.pose.pose.position.x)
        east = float(message.pose.pose.position.y)
        position = (north, east)
        if self._previous_path_position is not None:
            self._path_length_m += math.dist(
                position, self._previous_path_position
            )
        self._previous_path_position = position

    def _goal_distance(self) -> float:
        if self._odometry is None or self._goal is None:
            return math.inf
        north = float(self._odometry.pose.pose.position.x)
        east = float(self._odometry.pose.pose.position.y)
        return math.hypot(
            self._goal.pose.position.y - north,
            self._goal.pose.position.x - east,
        )

    def _clearance(self) -> float:
        if self._odometry is None or self._obstacles is None:
            return math.inf
        north = float(self._odometry.pose.pose.position.x)
        east = float(self._odometry.pose.pose.position.y)
        clearances = [
            cylinder_clearance_m(
                north,
                east,
                item.center.y,
                item.center.x,
                item.radius,
                self.config.uav_radius_m,
            )
            for item in self._obstacles.obstacles
        ]
        return min(clearances, default=math.inf)

    def _publish_termination(self) -> None:
        message = String()
        message.data = json.dumps({
            "schema": TERMINATION_SCHEMA,
            "terminal_reason": self._terminal_reason,
        }, sort_keys=True, separators=(",", ":"))
        self._termination_publisher.publish(message)

    @staticmethod
    def _optional(value: float) -> float | None:
        return float(value) if math.isfinite(value) else None

    def _result(self, now: float) -> dict:
        terminal = self._terminal_reason or str(
            self._supervisor.get("terminal_reason", "")
        )
        failure = str(self._supervisor.get("failure_reason", ""))
        final_goal = self._goal_distance()
        clearance = self._clearance()
        if math.isfinite(final_goal):
            self._minimum_goal_distance_m = min(
                self._minimum_goal_distance_m, final_goal
            )
        if math.isfinite(clearance):
            self._minimum_clearance_m = min(
                self._minimum_clearance_m, clearance
            )
        position = None
        yaw = None
        if self._odometry is not None:
            pose = self._odometry.pose.pose
            position = {
                "north_m": float(pose.position.x),
                "east_m": float(pose.position.y),
                "down_m": float(pose.position.z),
            }
            yaw = yaw_from_quaternion(
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
        episode_duration = (
            None
            if self._episode_started_s is None
            else now - self._episode_started_s
        )
        bc_end = self._bc_ended_s or now
        bc_duration = (
            None
            if self._bc_started_s is None
            else max(0.0, bc_end - self._bc_started_s)
        )
        return {
            "schema": RESULT_SCHEMA,
            "episode": self._episode,
            "seed": self._seed,
            "image_source": self._image_source,
            "checkpoint": self._policy.get("checkpoint_path"),
            "checkpoint_sha256": self._policy.get("checkpoint_sha256"),
            "encoder": self._policy.get("encoder_path"),
            "encoder_sha256": self._policy.get("encoder_sha256"),
            "success": terminal == "success" and not failure,
            "collision": terminal == "collision",
            "timeout": terminal == "timeout",
            "out_of_bounds": terminal == "out_of_bounds",
            "terminal_reason": terminal or "runtime_failure",
            "failure_reason": failure,
            "minimum_goal_distance_m": self._optional(
                self._minimum_goal_distance_m
            ),
            "final_goal_distance_m": self._optional(final_goal),
            "path_length_m": self._path_length_m,
            "episode_duration_s": episode_duration,
            "bc_control_duration_s": bc_duration,
            "steps": self._policy.get("inference_count"),
            "minimum_obstacle_clearance_m": self._optional(
                self._minimum_clearance_m
            ),
            "final_position": position,
            "final_yaw_rad": yaw,
            "collision_detector": COLLISION_DETECTOR,
            "physics_contact_verified": False,
            "completed_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
        }

    def _write_result(self, now: float) -> None:
        payload = self._result(now)
        self._result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._result_path.with_suffix(
            self._result_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._result_path)
        self._result_written = True
        self.get_logger().info(
            MSG_RESULT_SAVED.format(path=self._result_path)
        )

    def _tick(self) -> None:
        now = self._now_seconds()
        state = str(self._supervisor.get("state", ""))
        if state == BcFlightState.NAVIGATING.value:
            if self._bc_started_s is None:
                self._bc_started_s = now
            goal_distance = self._goal_distance()
            clearance = self._clearance()
            if math.isfinite(goal_distance):
                self._minimum_goal_distance_m = min(
                    self._minimum_goal_distance_m, goal_distance
                )
            if math.isfinite(clearance):
                self._minimum_clearance_m = min(
                    self._minimum_clearance_m, clearance
                )
            if (
                not self._terminal_reason
                and self._odometry is not None
                and math.isfinite(goal_distance)
                and math.isfinite(clearance)
            ):
                pose = self._odometry.pose.pose.position
                self._terminal_reason = select_terminal_reason(
                    goal_distance_m=goal_distance,
                    minimum_clearance_m=clearance,
                    bc_duration_s=now - self._bc_started_s,
                    north_m=pose.x,
                    east_m=pose.y,
                    config=self.config,
                ) or ""
                if self._terminal_reason:
                    self._bc_ended_s = now
        if self._terminal_reason:
            self._publish_termination()
        if state in {
            BcFlightState.COMPLETE.value,
            BcFlightState.FAILED.value,
        } and not self._result_written:
            try:
                self._write_result(now)
            except (OSError, TypeError, ValueError) as error:
                self._last_error = str(error)
                self.get_logger().error(
                    MSG_MONITOR_FAILED.format(error=error)
                )
                return
            rclpy.shutdown()


def main(args=None) -> int:
    """Run the finite BC evaluation monitor."""
    rclpy.init(args=args)
    node = BcEpisodeMonitorNode()
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
