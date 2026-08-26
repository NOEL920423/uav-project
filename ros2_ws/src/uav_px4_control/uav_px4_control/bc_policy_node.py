"""Publish live TOP RGB behavior-cloning commands as an independent source."""

from __future__ import annotations

import base64
import json
import math
import select
import subprocess
from pathlib import Path

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

from sensor_msgs.msg import CompressedImage

from std_msgs.msg import String

from std_srvs.srv import SetBool

from uav_ml.inference.bc_flight_contract import (
    body_action_to_ned,
    build_state8,
    canonical_image_source,
    freshness_error,
    validate_live_image,
    yaw_from_quaternion,
)

from uav_px4_control.control_mux_node import control_qos
from uav_px4_control.control_source_models import (
    BC_POLICY,
    SOURCE_TOPICS,
    VALID_COMMAND_FRAME,
)


TOP_RGB_TOPIC = "/uav/isaac/observer/image/compressed"
IMAGE_TOPICS = {"top_rgb": TOP_RGB_TOPIC}
ODOMETRY_TOPIC = "/uav/vehicle/odometry"
SCENE_GOAL_TOPIC = "/uav/scene/goal"
POLICY_STATUS_TOPIC = "/uav/bc/policy_status"
SET_POLICY_ENABLE_SERVICE = "/uav/bc/set_enabled"
POLICY_STATUS_SCHEMA = "uav_bc_policy_status/v1"

MSG_LOADING_POLICY = "[BC Flight] Loading BC policy..."
MSG_WAITING_FOR_IMAGE = "[BC Flight] Waiting for TOP RGB..."
MSG_POLICY_READY = "[BC Flight] BC policy is ready."
MSG_POLICY_ENABLED = "[BC Flight] BC control enabled."
MSG_POLICY_DISABLED = "[BC Flight] BC control disabled."
MSG_INFERENCE_FAILED = "[BC Flight] Inference failed: {error}"
MSG_WORKER_FAILED = "[BC Flight] Inference worker failed: {error}"
MSG_IMAGE_CONTRACT_FAILED = "[BC Flight] TOP RGB contract failed: {error}"


def scene_qos() -> QoSProfile:
    """Receive the last published scene goal from the durable boundary."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class BcPolicyNode(Node):
    """Own TOP preprocessing, state8 construction, inference, and BC output."""

    def __init__(self) -> None:
        """Start the ML worker and create live observation boundaries."""
        super().__init__("bc_policy")
        self.declare_parameter("repository_root", ".")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("ml_python", "python3")
        self.declare_parameter("image_source", "top_rgb")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("image_freshness_timeout_s", 0.35)
        self.declare_parameter("odometry_freshness_timeout_s", 0.25)
        self.declare_parameter("command_publish_rate_hz", 20.0)
        self.declare_parameter("inference_timeout_s", 0.50)
        self._image_timeout_s = float(
            self.get_parameter("image_freshness_timeout_s").value
        )
        self._odometry_timeout_s = float(
            self.get_parameter("odometry_freshness_timeout_s").value
        )
        publish_rate = float(
            self.get_parameter("command_publish_rate_hz").value
        )
        self._inference_timeout_s = float(
            self.get_parameter("inference_timeout_s").value
        )
        if min(self._image_timeout_s, self._odometry_timeout_s) <= 0.0:
            raise ValueError("observation freshness timeouts must be positive")
        if not math.isfinite(publish_rate) or publish_rate <= 0.0:
            raise ValueError("command publish rate must be positive")
        requested_source = canonical_image_source(
            str(self.get_parameter("image_source").value)
        )
        if requested_source not in IMAGE_TOPICS:
            raise ValueError(
                f"no live image topic is registered for {requested_source!r}"
            )
        repository_root = Path(
            str(self.get_parameter("repository_root").value)
        ).expanduser().resolve()
        checkpoint_value = str(
            self.get_parameter("checkpoint_path").value
        ).strip()
        device_name = str(self.get_parameter("device").value)
        self.get_logger().info(MSG_LOADING_POLICY)
        command = [
            str(self.get_parameter("ml_python").value),
            "-m",
            "uav_ml.inference.bc_flight_worker",
            "--repository-root",
            str(repository_root),
            "--image-source",
            requested_source,
            "--device",
            device_name,
        ]
        if checkpoint_value:
            command.extend(["--checkpoint", checkpoint_value])
        self._worker = subprocess.Popen(
            command,
            cwd=repository_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        handshake = self._read_worker(60.0)
        if not handshake.get("ready"):
            error = handshake.get("error", "worker exited during startup")
            raise RuntimeError(MSG_WORKER_FAILED.format(error=error))
        self._identity = dict(handshake["identity"])
        self._requested_source = requested_source
        self._enabled = False
        self._image: bytes | None = None
        self._image_receipt_s: float | None = None
        self._image_sequence = 0
        self._inferred_image_sequence = -1
        self._odometry: Odometry | None = None
        self._odometry_receipt_s: float | None = None
        self._goal: PoseStamped | None = None
        self._previous_action = (0.0, 0.0, 0.0)
        self._last_command: tuple[float, float, float, float] | None = None
        self._last_error = "disabled"
        self._image_contract_error = ""
        self._inference_count = 0
        self._waiting_logged = False

        qos = control_qos()
        self._command_publisher = self.create_publisher(
            TwistStamped, SOURCE_TOPICS[BC_POLICY], qos
        )
        self._status_publisher = self.create_publisher(
            String, POLICY_STATUS_TOPIC, qos
        )
        self.create_subscription(
            CompressedImage,
            IMAGE_TOPICS[requested_source],
            self._image_callback,
            QoSProfile(depth=2),
        )
        self.create_subscription(
            Odometry, ODOMETRY_TOPIC, self._odometry_callback, qos
        )
        self.create_subscription(
            PoseStamped, SCENE_GOAL_TOPIC, self._goal_callback, scene_qos()
        )
        self._enable_service = self.create_service(
            SetBool, SET_POLICY_ENABLE_SERVICE, self._enable_callback
        )
        self._timer = self.create_timer(1.0 / publish_rate, self._tick)
        self.get_logger().info(MSG_POLICY_READY)

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _image_callback(self, message: CompressedImage) -> None:
        if message.format and "jpeg" not in message.format.lower():
            self._last_error = "top_rgb_format_is_not_jpeg"
            return
        image = bytes(message.data)
        try:
            validate_live_image(image, self._requested_source)
        except ValueError as error:
            self._image = None
            self._image_receipt_s = None
            self._image_contract_error = f"top_rgb_contract_error:{error}"
            if self._last_error != self._image_contract_error:
                self.get_logger().error(
                    MSG_IMAGE_CONTRACT_FAILED.format(error=error)
                )
            self._last_error = self._image_contract_error
            return
        self._image = image
        self._image_contract_error = ""
        self._image_receipt_s = self._now_seconds()
        self._image_sequence += 1

    def _odometry_callback(self, message: Odometry) -> None:
        if message.header.frame_id != VALID_COMMAND_FRAME:
            self._last_error = "odometry_frame_is_not_px4_ned"
            return
        self._odometry = message
        self._odometry_receipt_s = self._now_seconds()

    def _goal_callback(self, message: PoseStamped) -> None:
        if message.header.frame_id != "isaac_world":
            self._last_error = "goal_frame_is_not_isaac_world"
            return
        self._goal = message

    def _enable_callback(self, request, response):
        self._enabled = bool(request.data)
        self._previous_action = (0.0, 0.0, 0.0)
        self._last_command = None
        self._last_error = "" if self._enabled else "disabled"
        response.success = True
        response.message = (
            MSG_POLICY_ENABLED if self._enabled else MSG_POLICY_DISABLED
        )
        self.get_logger().info(response.message)
        return response

    def _observation_error(self, now: float) -> str | None:
        if self._image_contract_error:
            return self._image_contract_error
        return freshness_error(
            now,
            self._image_receipt_s,
            self._odometry_receipt_s,
            self._goal is not None,
            self._image_timeout_s,
            self._odometry_timeout_s,
        )

    def _infer(self) -> None:
        assert self._image is not None
        assert self._odometry is not None
        assert self._goal is not None
        pose = self._odometry.pose.pose
        twist = self._odometry.twist.twist
        yaw = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        state = build_state8(
            twist.linear.x,
            twist.linear.y,
            pose.position.x,
            pose.position.y,
            self._goal.pose.position.y,
            self._goal.pose.position.x,
            yaw,
            self._previous_action,
        )
        request = {
            "jpeg_base64": base64.b64encode(self._image).decode("ascii"),
            "state8": [float(value) for value in state],
        }
        assert self._worker.stdin is not None
        self._worker.stdin.write(
            json.dumps(request, separators=(",", ":")) + "\n"
        )
        self._worker.stdin.flush()
        response = self._read_worker(self._inference_timeout_s)
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        action = response.get("action")
        if not isinstance(action, list) or len(action) != 3:
            raise RuntimeError("inference worker returned an invalid action")
        self._last_command = body_action_to_ned(action, yaw)
        self._previous_action = tuple(float(value) for value in action)
        self._inference_count += 1
        self._inferred_image_sequence = self._image_sequence

    def _publish_command(self) -> None:
        assert self._last_command is not None
        north, east, down, yaw_rate = self._last_command
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = VALID_COMMAND_FRAME
        message.twist.linear.x = north
        message.twist.linear.y = east
        message.twist.linear.z = down
        message.twist.angular.z = yaw_rate
        self._command_publisher.publish(message)

    def _publish_status(self, ready: bool, reason: str) -> None:
        status = {
            "schema": POLICY_STATUS_SCHEMA,
            "enabled": self._enabled,
            "ready": ready,
            "reason": reason,
            "image_source": self._identity["image_source"],
            "checkpoint_path": self._identity["checkpoint_path"],
            "checkpoint_sha256": self._identity["checkpoint_sha256"],
            "encoder_path": self._identity["encoder_path"],
            "encoder_sha256": self._identity["encoder_sha256"],
            "inference_count": self._inference_count,
            "image_sequence": self._image_sequence,
        }
        message = String()
        message.data = json.dumps(
            status, sort_keys=True, separators=(",", ":")
        )
        self._status_publisher.publish(message)

    def _tick(self) -> None:
        now = self._now_seconds()
        reason = self._observation_error(now)
        if not self._enabled:
            self._publish_status(reason is None, "disabled")
            return
        if reason is not None:
            self._last_command = None
            self._last_error = reason
            if reason == "waiting_for_top_rgb" and not self._waiting_logged:
                self.get_logger().info(MSG_WAITING_FOR_IMAGE)
                self._waiting_logged = True
            self._publish_status(False, reason)
            return
        try:
            if self._image_sequence != self._inferred_image_sequence:
                self._infer()
            if self._last_command is not None:
                self._publish_command()
            self._last_error = ""
            self._publish_status(self._last_command is not None, "ready")
        except (ValueError, RuntimeError) as error:
            self._last_command = None
            self._last_error = f"inference_failed:{error}"
            self.get_logger().error(
                MSG_INFERENCE_FAILED.format(error=error)
            )
            self._publish_status(False, self._last_error)

    def _read_worker(self, timeout_s: float) -> dict:
        assert self._worker.stdout is not None
        ready, _, _ = select.select(
            [self._worker.stdout], [], [], float(timeout_s)
        )
        if not ready:
            raise RuntimeError("inference worker response timed out")
        line = self._worker.stdout.readline()
        if not line:
            detail = ""
            if self._worker.stderr is not None:
                detail = self._worker.stderr.read().strip()
            raise RuntimeError(
                detail or "inference worker closed its output"
            )
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError("inference worker response is not an object")
        return payload

    def destroy_node(self):
        """Stop the private ML worker before destroying ROS resources."""
        if hasattr(self, "_worker") and self._worker.poll() is None:
            self._worker.terminate()
            try:
                self._worker.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._worker.kill()
                self._worker.wait(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> int:
    """Run the ROS-facing TOP RGB BC policy adapter."""
    rclpy.init(args=args)
    node = BcPolicyNode()
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
