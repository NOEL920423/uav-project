"""Bounded client for the diagnostic Isaac persistent-runtime protocol."""

from __future__ import annotations

import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from std_msgs.msg import String


COMMAND_TOPIC = "/uav/isaac/runtime_smoke/command"
STATUS_TOPIC = "/uav/isaac/runtime_smoke/status"
SCHEMA = "uav_persistent_runtime_smoke/v1"
EXPECTED_STATES = {
    "observe": "booted",
    "stop_episode": "stopped",
    "reset_episode": "ready_for_px4",
}


class RuntimeSmokeLifecycleClient(Node):
    """Publish one lifecycle request and save its acknowledgement."""

    def __init__(self) -> None:
        """Create the bounded command/status lifecycle client."""
        super().__init__("runtime_smoke_lifecycle_client")
        self.declare_parameter("command", "observe")
        self.declare_parameter("command_id", "observe")
        self.declare_parameter("generation", 0)
        self.declare_parameter("timeout_s", 30.0)
        self.declare_parameter("require_camera_ready", False)
        self.declare_parameter(
            "evidence_path", "/tmp/runtime_smoke_status.json"
        )
        self.command = str(self.get_parameter("command").value)
        self.command_id = str(self.get_parameter("command_id").value)
        self.generation = int(self.get_parameter("generation").value)
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.require_camera_ready = bool(
            self.get_parameter("require_camera_ready").value
        )
        self.evidence_path = Path(
            str(self.get_parameter("evidence_path").value)
        )
        if self.command not in EXPECTED_STATES:
            raise ValueError(
                f"unsupported smoke lifecycle command: {self.command}"
            )
        self.expected_state = EXPECTED_STATES[self.command]
        self.started = time.monotonic()
        self.last_publish = 0.0
        self.finished = False
        self.exit_code = 1
        self.publisher = self.create_publisher(String, COMMAND_TOPIC, 10)
        self.create_subscription(String, STATUS_TOPIC, self._status, 10)
        self.create_timer(0.10, self._tick)

    def _status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if payload.get("schema") != SCHEMA:
            return
        if payload.get("state") == "failed":
            self._finish(1, payload)
            return
        if self.command == "observe":
            command_matches = True
            generation_matches = True
        else:
            command_matches = payload.get("last_command_id") == self.command_id
            generation_matches = payload.get("generation") == self.generation
        if not command_matches or not generation_matches:
            return
        if payload.get("state") != self.expected_state:
            return
        if self.require_camera_ready and not payload.get("camera_ready"):
            return
        self._finish(0, payload)

    def _tick(self) -> None:
        if self.finished:
            return
        now = time.monotonic()
        if now - self.started > self.timeout_s:
            self._finish(1, {
                "schema": SCHEMA,
                "state": "client_timeout",
                "generation": self.generation,
                "last_command_id": self.command_id,
                "failure_reason": (
                    f"timed out waiting for {self.expected_state}"
                ),
            })
            return
        if self.command != "observe" and now - self.last_publish >= 0.25:
            request = String()
            request.data = json.dumps({
                "command": self.command,
                "command_id": self.command_id,
                "generation": self.generation,
            }, sort_keys=True, separators=(",", ":"))
            self.publisher.publish(request)
            self.last_publish = now

    def _finish(self, code: int, payload: dict) -> None:
        if self.finished:
            return
        self.finished = True
        self.exit_code = code
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        log = self.get_logger().info if code == 0 else self.get_logger().error
        log(
            "RUNTIME_SMOKE_LIFECYCLE_RESULT "
            f"success={str(code == 0).lower()} command={self.command} "
            f"state={payload.get('state')} evidence={self.evidence_path}"
        )
        rclpy.shutdown()


def main(args=None) -> int:
    """Run the bounded persistent-runtime lifecycle client."""
    rclpy.init(args=args)
    node = RuntimeSmokeLifecycleClient()
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
