"""Prepare one seeded Isaac scene after proving PX4 is safely reset."""

from __future__ import annotations

import importlib
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from std_msgs.msg import String


COMMAND_TOPIC = "/uav/isaac/episode_command"
STATUS_TOPIC = "/uav/isaac/runtime_status"
VEHICLE_STATUS_TOPIC = "/fmu/out/vehicle_status"
LAND_DETECTED_TOPIC = "/fmu/out/vehicle_land_detected"


class EpisodeSceneClient(Node):
    """Send an idempotent prepare request only from landed/disarmed state."""

    def __init__(self) -> None:
        """Initialize safety telemetry, scene status, and command endpoints."""
        super().__init__("episode_scene_client")
        self.declare_parameter("episode_id", "episode_000001")
        self.declare_parameter("random_seed", 0)
        self.declare_parameter("scene_mode", "normal")
        self.declare_parameter("timeout_s", 20.0)
        self.episode_id = str(self.get_parameter("episode_id").value)
        self.random_seed = int(self.get_parameter("random_seed").value)
        self.scene_mode = str(self.get_parameter("scene_mode").value)
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.started = time.monotonic()
        self.exit_code = 1
        self.finished = False
        self.vehicle_status = None
        self.land_detected = None
        self.command_count = 0

        messages = importlib.import_module("px4_msgs.msg")
        px4_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.publisher = self.create_publisher(String, COMMAND_TOPIC, 10)
        self.create_subscription(String, STATUS_TOPIC, self._status, 10)
        self.create_subscription(
            messages.VehicleStatus,
            VEHICLE_STATUS_TOPIC,
            self._vehicle_status,
            px4_qos,
        )
        self.create_subscription(
            messages.VehicleLandDetected,
            LAND_DETECTED_TOPIC,
            self._land_status,
            px4_qos,
        )
        self.create_timer(0.25, self._tick)

    def _vehicle_status(self, message) -> None:
        self.vehicle_status = message

    def _land_status(self, message) -> None:
        self.land_detected = message

    def _safe_reset_state(self) -> bool:
        status = self.vehicle_status
        return bool(
            status is not None
            and self.land_detected is not None
            and self.land_detected.landed
            and status.arming_state != status.ARMING_STATE_ARMED
            and not status.failsafe
        )

    def _status(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if status.get("episode_command_error"):
            self._finish(
                1, "Isaac rejected scene command: "
                + str(status["episode_command_error"])
            )
            return
        if (
            status.get("episode_id") == self.episode_id
            and status.get("random_seed") == self.random_seed
            and status.get("scene_configuration", {}).get("mode")
            == self.scene_mode
            and status.get("timeline_playing") is True
            and status.get("pose_valid") is True
        ):
            self.get_logger().info(
                "EXPERT_SCENE_READY " + json.dumps(
                    status["scene_configuration"], sort_keys=True
                )
            )
            self._finish(0, "scene acknowledged from safe reset state")

    def _tick(self) -> None:
        if self.finished:
            return
        if time.monotonic() - self.started > self.timeout_s:
            self._finish(1, "timed out waiting for safe reset/scene ack")
            return
        if not self._safe_reset_state():
            return
        request = String()
        request.data = json.dumps({
            "command": "prepare_episode",
            "episode_id": self.episode_id,
            "random_seed": self.random_seed,
            "mode": self.scene_mode,
        }, sort_keys=True, separators=(",", ":"))
        self.publisher.publish(request)
        self.command_count += 1

    def _finish(self, code: int, detail: str) -> None:
        if self.finished:
            return
        self.finished = True
        self.exit_code = code
        logger = (
            self.get_logger().info
            if code == 0 else self.get_logger().error
        )
        logger(
            f"EXPERT_SCENE_RESULT success={str(code == 0).lower()} "
            f"commands={self.command_count} detail={detail}"
        )
        rclpy.shutdown()


def main(args=None) -> int:
    """Wait for a safe reset state, prepare one scene, and exit."""
    rclpy.init(args=args)
    node = EpisodeSceneClient()
    try:
        rclpy.spin(node)
    finally:
        code = node.exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
