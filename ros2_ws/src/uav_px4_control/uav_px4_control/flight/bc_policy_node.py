"""Publish live source-matched behavior-cloning commands independently."""

from __future__ import annotations

import base64
import json
import time
import math
import queue
import select
import subprocess
import sys
import threading
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
from uav_ml.tools.bc_flight_video import capture_policy_input_frame

from uav_px4_control.control.control_mux_node import control_qos
from uav_px4_control.control.control_source_models import (
    BC_POLICY,
    SOURCE_TOPICS,
    VALID_COMMAND_FRAME,
)


from uav_px4_control.diagnostics import TimingRecorder, timed_callback


IMAGE_TOPICS = {
    "top_rgb": "/uav/isaac/observer/image/compressed",
    "fpv_rgb": "/uav/isaac/fpv/image/compressed",
    "fpv_depth": "/uav/isaac/fpv/depth/compressed",
}
ODOMETRY_TOPIC = "/uav/vehicle/odometry"
SCENE_GOAL_TOPIC = "/uav/scene/goal"
POLICY_STATUS_TOPIC = "/uav/bc/policy_status"
SET_POLICY_ENABLE_SERVICE = "/uav/bc/set_enabled"
POLICY_STATUS_SCHEMA = "uav_bc_policy_status/v1"

MSG_LOADING_POLICY = "[BC Flight] Loading BC policy..."
MSG_WAITING_FOR_IMAGE = "[BC Flight] Waiting for {source}..."
MSG_POLICY_READY = "[BC Flight] BC policy is ready."
MSG_POLICY_ENABLED = "[BC Flight] BC control enabled."
MSG_POLICY_DISABLED = "[BC Flight] BC control disabled."
MSG_INFERENCE_FAILED = "[BC Flight] Inference failed: {error}"
MSG_WORKER_FAILED = "[BC Flight] Inference worker failed: {error}"
MSG_IMAGE_CONTRACT_FAILED = "[BC Flight] {source} contract failed: {error}"


def scene_qos() -> QoSProfile:
    """Receive the last published scene goal from the durable boundary."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class BcPolicyNode(Node):
    """Own source-matched preprocessing, state8 construction, and BC output."""

    def __init__(self) -> None:
        """Start the ML worker and create live observation boundaries."""
        super().__init__("bc_policy")
        self._timing = TimingRecorder(self)
        self.declare_parameter("repository_root", ".")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("ml_python", "python3")
        self.declare_parameter("image_source", "top_rgb")
        self.declare_parameter("image_sources", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("image_freshness_timeout_s", 0.35)
        self.declare_parameter("image_sync_tolerance_s", 0.10)
        self.declare_parameter("odometry_freshness_timeout_s", 0.25)
        self.declare_parameter("command_publish_rate_hz", 20.0)
        self.declare_parameter("inference_timeout_s", 0.50)
        self.declare_parameter("action_freshness_timeout_s", 0.25)
        self.declare_parameter("video_spool_dir", "")
        self._image_timeout_s = float(
            self.get_parameter("image_freshness_timeout_s").value
        )
        self._image_sync_tolerance_s = float(
            self.get_parameter("image_sync_tolerance_s").value
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
        self._action_freshness_timeout_s = float(
            self.get_parameter("action_freshness_timeout_s").value
        )
        if min(
            self._image_timeout_s,
            self._image_sync_tolerance_s,
            self._odometry_timeout_s,
        ) <= 0.0:
            raise ValueError("observation freshness timeouts must be positive")
        if (
            not math.isfinite(self._action_freshness_timeout_s)
            or self._action_freshness_timeout_s <= 0.0
        ):
            raise ValueError("action freshness timeout must be finite and positive")
        if not math.isfinite(publish_rate) or publish_rate <= 0.0:
            raise ValueError("command publish rate must be positive")
        configured_sources = self.get_parameter("image_sources").value
        if isinstance(configured_sources, str):
            configured_sources = [
                value for value in configured_sources.replace(",", " ").split()
            ]
        if configured_sources:
            requested_sources = tuple(
                canonical_image_source(str(value)) for value in configured_sources
            )
        else:
            requested_sources = (canonical_image_source(
                str(self.get_parameter("image_source").value)
            ),)
        if (
            not requested_sources
            or len(set(requested_sources)) != len(requested_sources)
            or any(source not in IMAGE_TOPICS for source in requested_sources)
        ):
            raise ValueError(f"invalid live image sources: {requested_sources!r}")
        requested_source = requested_sources[0]
        repository_root = Path(
            str(self.get_parameter("repository_root").value)
        ).expanduser().resolve()
        checkpoint_value = str(
            self.get_parameter("checkpoint_path").value
        ).strip()
        device_name = str(self.get_parameter("device").value)
        self.get_logger().info(MSG_LOADING_POLICY)
        source_argument = (
            ["--image-sources", *requested_sources]
            if len(requested_sources) > 1
            else ["--image-source", requested_source]
        )
        command = [
            str(self.get_parameter("ml_python").value),
            "-m",
            "uav_ml.inference.bc_flight_worker",
            "--repository-root",
            str(repository_root),
            *source_argument,
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
        self._worker_stderr_thread = threading.Thread(
            target=self._drain_worker_stderr, daemon=True
        )
        self._worker_stderr_thread.start()
        handshake = self._read_worker(60.0)
        if not handshake.get("ready"):
            error = handshake.get("error", "worker exited during startup")
            raise RuntimeError(MSG_WORKER_FAILED.format(error=error))
        self._identity = dict(handshake["identity"])
        checkpoint_sources = tuple(
            canonical_image_source(source)
            for source in self._identity.get(
                "image_sources", [self._identity.get("image_source", "")]
            )
        )
        if not checkpoint_sources or checkpoint_sources[0] != requested_sources[0]:
            raise ValueError(
                "BC checkpoint primary image source does not match the launch "
                f"request: requested {requested_sources!r}, checkpoint has "
                f"{checkpoint_sources!r}"
            )
        if len(requested_sources) > 1 and requested_sources != checkpoint_sources:
            raise ValueError(
                "BC checkpoint image sources do not match the launch request: "
                f"requested {requested_sources!r}, checkpoint has "
                f"{checkpoint_sources!r}"
            )
        requested_sources = checkpoint_sources
        requested_source = requested_sources[0]
        self._runtime_device = dict(handshake.get("runtime", {}))
        self._requested_source = requested_source
        self._requested_sources = requested_sources
        self._enabled = False
        self._source_images: dict[str, bytes | None] = {
            source: None for source in requested_sources
        }
        self._source_receipt_s: dict[str, float | None] = {
            source: None for source in requested_sources
        }
        self._source_receipt_monotonic_ns: dict[str, int | None] = {
            source: None for source in requested_sources
        }
        self._source_receipt_ros_ns: dict[str, int | None] = {
            source: None for source in requested_sources
        }
        self._source_header_stamp: dict[str, tuple[int, int] | None] = {
            source: None for source in requested_sources
        }
        self._source_sequence: dict[str, int] = {
            source: 0 for source in requested_sources
        }
        self._source_contract_errors: dict[str, str] = {}
        self._image: bytes | None = None
        self._image_receipt_s: float | None = None
        self._image_receipt_monotonic_ns: int | None = None
        self._image_receipt_ros_ns: int | None = None
        self._image_header_stamp: tuple[int, int] | None = None
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
        self._inference_position: dict[str, float] | None = None
        video_spool_value = str(
            self.get_parameter("video_spool_dir").value
        ).strip()
        self._video_spool_dir = (
            Path(video_spool_value).expanduser().resolve()
            if video_spool_value else None
        )
        self._video_capture_queue: queue.Queue | None = None
        self._video_capture_thread: threading.Thread | None = None
        self._video_dropped_count = 0
        if self._video_spool_dir is not None:
            self._video_capture_queue = queue.Queue(maxsize=16)
        self._action_origin = {}
        self._action_publish_count = 0
        self._pending_action_body: tuple[float, float, float] | None = None
        self._last_action_publish_monotonic_ns: int | None = None
        self._last_new_action_publish_monotonic_ns: int | None = None
        self._last_command_stamp_ros_ns: int | None = None
        self._last_accepted_request_id = 0
        self._next_request_id = 0
        self._waiting_logged = False
        self._requested_image_sequence: tuple[int, ...] | None = None
        self._enable_generation = 0
        self._inference_worker_failed = False
        self._inference_condition = threading.Condition()
        self._pending_inference: dict | None = None
        self._completed_inference: dict | None = None
        self._inference_busy = False
        self._inference_stop = False

        qos = control_qos()
        self._command_publisher = self.create_publisher(
            TwistStamped, SOURCE_TOPICS[BC_POLICY], qos
        )
        self._status_publisher = self.create_publisher(
            String, POLICY_STATUS_TOPIC, qos
        )
        for source in requested_sources:
            self.create_subscription(
                CompressedImage,
                IMAGE_TOPICS[source],
                lambda message, selected_source=source: self._image_callback(
                    message, selected_source
                ),
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
        self._inference_guard = self.create_guard_condition(
            self._inference_ready_callback
        )
        self._timer = self.create_timer(1.0 / publish_rate, self._tick)
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            name="bc-inference-io",
            daemon=True,
        )
        self._inference_thread.start()
        if self._video_capture_queue is not None:
            self._video_capture_thread = threading.Thread(
                target=self._video_capture_loop,
                name="bc-policy-video-recorder",
                daemon=True,
            )
            self._video_capture_thread.start()
        self._timing.record(
            "inference_runtime",
            requested_device=device_name,
            **self._runtime_device,
        )
        self.get_logger().info(MSG_POLICY_READY)

    def _drain_worker_stderr(self) -> None:
        """Drain worker diagnostics without mixing them into stdout IPC."""
        stream = self._worker.stderr
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            text = line.rstrip("\n")
            if text:
                print(f"[Inference][stderr] {text}", file=sys.stderr, flush=True)
                self.get_logger().error(text)
        code = self._worker.poll()
        if code is not None and code != 0:
            message = f"[Inference] worker exited with return code {code}"
            print(message, file=sys.stderr, flush=True)
            self.get_logger().error(message)

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    @timed_callback
    def _image_callback(
        self, message: CompressedImage, image_source: str | None = None
    ) -> None:
        source_name = image_source or self._requested_source
        self._timing.receive(IMAGE_TOPICS[source_name], message)
        image_receipt_ros_ns = self.get_clock().now().nanoseconds
        receipt_monotonic_ns = time.monotonic_ns()
        receipt_s = image_receipt_ros_ns / 1e9
        image = bytes(message.data)
        try:
            validate_live_image(image, source_name)
        except ValueError as error:
            self._source_images[source_name] = None
            self._source_receipt_s[source_name] = None
            self._source_contract_errors[source_name] = (
                f"{source_name}_contract_error:{error}"
            )
            if source_name == self._requested_source:
                self._image = None
                self._image_receipt_s = None
                self._image_contract_error = self._source_contract_errors[source_name]
            if self._last_error != self._source_contract_errors[source_name]:
                self.get_logger().error(
                    MSG_IMAGE_CONTRACT_FAILED.format(
                        source=source_name, error=error
                    )
                )
            self._last_error = self._source_contract_errors[source_name]
            return
        self._source_images[source_name] = image
        self._source_contract_errors.pop(source_name, None)
        header_stamp = (
            int(message.header.stamp.sec),
            int(message.header.stamp.nanosec),
        )
        self._source_receipt_s[source_name] = receipt_s
        self._source_receipt_monotonic_ns[source_name] = receipt_monotonic_ns
        self._source_receipt_ros_ns[source_name] = image_receipt_ros_ns
        self._source_header_stamp[source_name] = header_stamp
        self._source_sequence[source_name] += 1
        self._timing.record(
            "image_received",
            image_source=source_name,
            image_sequence=self._source_sequence[source_name],
            enable_generation=self._enable_generation,
            image_header_stamp=header_stamp,
            image_receipt_ros_ns=image_receipt_ros_ns,
            image_receipt_monotonic_ns=receipt_monotonic_ns,
            source_to_receipt_latency_ms=None,
            source_to_receipt_latency_reason="source_clock_not_synchronized",
        )
        if source_name == self._requested_source:
            self._image = image
            self._image_contract_error = ""
            self._image_receipt_s = receipt_s
            self._image_receipt_ros_ns = image_receipt_ros_ns
            self._image_receipt_monotonic_ns = receipt_monotonic_ns
            self._image_header_stamp = header_stamp
            self._image_sequence = self._source_sequence[source_name]
        if self._enabled and self._observation_error(self._now_seconds()) is None:
            self._schedule_inference()

    @timed_callback
    def _odometry_callback(self, message: Odometry) -> None:
        if message.header.frame_id != VALID_COMMAND_FRAME:
            self._last_error = "odometry_frame_is_not_px4_ned"
            return
        self._timing.receive(ODOMETRY_TOPIC, message)
        self._odometry = message
        self._odometry_receipt_s = self._now_seconds()

    @timed_callback
    def _goal_callback(self, message: PoseStamped) -> None:
        if message.header.frame_id != "isaac_world":
            self._last_error = "goal_frame_is_not_isaac_world"
            return
        self._timing.receive(SCENE_GOAL_TOPIC, message)
        self._goal = message

    @timed_callback
    def _enable_callback(self, request, response):
        self._enabled = bool(request.data)
        self._enable_generation += 1
        with self._inference_condition:
            self._pending_inference = None
        self._previous_action = (0.0, 0.0, 0.0)
        self._last_command = None
        self._pending_action_body = None
        self._action_origin = {}
        self._action_publish_count = 0
        self._last_action_publish_monotonic_ns = None
        self._last_new_action_publish_monotonic_ns = None
        self._last_command_stamp_ros_ns = None
        self._last_error = "" if self._enabled else "disabled"
        response.success = True
        response.message = (
            MSG_POLICY_ENABLED if self._enabled else MSG_POLICY_DISABLED
        )
        self.get_logger().info(response.message)
        return response

    def _observation_error(self, now: float) -> str | None:
        for source in self._requested_sources:
            contract_error = self._source_contract_errors.get(source)
            if contract_error:
                return contract_error
            error = freshness_error(
                now,
                self._source_receipt_s[source],
                self._odometry_receipt_s,
                self._goal is not None,
                self._image_timeout_s,
                self._odometry_timeout_s,
                source,
            )
            if error is not None:
                return error
        if len(self._requested_sources) > 1:
            stamps = []
            for source in self._requested_sources:
                header = self._source_header_stamp[source]
                receipt = self._source_receipt_s[source]
                if header is not None and (header[0] or header[1]):
                    stamps.append(header[0] + header[1] / 1e9)
                elif receipt is not None:
                    stamps.append(receipt)
            synchronization_limit = self._image_sync_tolerance_s
            if "top_rgb" in self._requested_sources:
                synchronization_limit = min(synchronization_limit, 0.001)
            if (
                len(stamps) != len(self._requested_sources)
                or max(stamps) - min(stamps) > synchronization_limit
            ):
                return "waiting_for_synchronized_images"
        return None

    def _inference_snapshot(self) -> dict:
        """Freeze the latest observation for one background inference."""
        assert all(self._source_images[source] is not None
                   for source in self._requested_sources)
        assert all(self._source_receipt_monotonic_ns[source] is not None
                   for source in self._requested_sources)
        assert self._odometry is not None
        assert self._goal is not None
        self._timing.consume(image_sequence=self._image_sequence)
        observation_inputs = {
            topic: dict(entry) for topic, entry in self._timing.inputs.items()
        }
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
        return {
            "image": self._source_images[self._requested_source],
            "images": {
                source: self._source_images[source]
                for source in self._requested_sources
            },
            "state8": [float(value) for value in state],
            "previous_action": self._previous_action,
            "image_sequence": self._image_sequence,
            "image_sequences": dict(self._source_sequence),
            "image_receipt_monotonic_ns": int(
                self._source_receipt_monotonic_ns[self._requested_source]
            ),
            "oldest_image_receipt_monotonic_ns": min(
                int(self._source_receipt_monotonic_ns[source])
                for source in self._requested_sources
            ),
            "image_receipts_monotonic_ns": {
                source: int(self._source_receipt_monotonic_ns[source])
                for source in self._requested_sources
            },
            "image_receipt_ros_ns": self._image_receipt_ros_ns,
            "image_header_stamp": self._source_header_stamp[
                self._requested_source
            ],
            "image_header_stamps": {
                source: self._source_header_stamp[source]
                for source in self._requested_sources
            },
            "enable_generation": self._enable_generation,
            "observation_inputs": observation_inputs,
            "yaw": yaw,
            "position": {
                "north_m": float(pose.position.x),
                "east_m": float(pose.position.y),
            },
        }

    def _schedule_inference(self) -> None:
        if self._inference_worker_failed:
            return
        sequence = tuple(
            self._source_sequence[source] for source in self._requested_sources
        )
        if self._requested_image_sequence is not None and any(
            current <= previous
            for current, previous in zip(sequence, self._requested_image_sequence)
        ):
            return
        request = self._inference_snapshot()
        self._next_request_id += 1
        request["request_id"] = self._next_request_id
        request["scheduled_monotonic_ns"] = time.monotonic_ns()
        self._requested_image_sequence = sequence
        with self._inference_condition:
            replaced = self._pending_inference is not None
            self._pending_inference = request
            self._inference_condition.notify()
        self._timing.record(
            "inference_scheduled",
            request_id=request["request_id"],
            enable_generation=request["enable_generation"],
            image_sequence=request["image_sequence"],
            replaced_pending=replaced,
            observation_inputs=request["observation_inputs"],
        )
        if replaced:
            self._timing.record(
                "inference_pending_replaced",
                request_id=request["request_id"],
                enable_generation=request["enable_generation"],
                image_sequence=request["image_sequence"],
            )

    def _inference_loop(self) -> None:
        """Run one IPC request at a time; the waiting slot keeps only newest."""
        while True:
            with self._inference_condition:
                while (
                    (self._pending_inference is None
                     or self._completed_inference is not None)
                    and not self._inference_stop
                ):
                    self._inference_condition.wait()
                if self._inference_stop:
                    return
                request = self._pending_inference
                self._pending_inference = None
                self._inference_busy = True
            assert request is not None
            inference_started_ns = time.monotonic_ns()
            try:
                profile_enabled = self._timing.stream is not None
                encode_started_ns = time.monotonic_ns()
                worker_request = {
                    "images_base64": {
                        source: base64.b64encode(request["images"][source]).decode("ascii")
                        for source in self._requested_sources
                    },
                    "state8": request["state8"],
                    "profile": profile_enabled,
                }
                encoded_request = json.dumps(
                    worker_request, separators=(",", ":")
                ) + "\n"
                request_encode_ms = (
                    time.monotonic_ns() - encode_started_ns
                ) / 1e6
                assert self._worker.stdin is not None
                write_started_ns = time.monotonic_ns()
                self._worker.stdin.write(encoded_request)
                self._worker.stdin.flush()
                request_write_ms = (
                    time.monotonic_ns() - write_started_ns
                ) / 1e6
                wait_started_ns = time.monotonic_ns()
                response = self._read_worker(self._inference_timeout_s)
                worker_wait_ms = (time.monotonic_ns() - wait_started_ns) / 1e6
                if "error" in response:
                    raise RuntimeError(str(response["error"]))
                action = response.get("action")
                if not isinstance(action, list) or len(action) != 3:
                    raise RuntimeError(
                        "inference worker returned an invalid action"
                    )
                completion = {
                    "request": request,
                    "action": [float(value) for value in action],
                    "worker_profile": response.get("profile"),
                    "request_encode_ms": request_encode_ms,
                    "request_write_ms": request_write_ms,
                    "worker_wait_ms": worker_wait_ms,
                    "inference_started_ns": inference_started_ns,
                    "inference_completed_ns": time.monotonic_ns(),
                }
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
                completion = {
                    "request": request,
                    "error": str(error),
                    "inference_started_ns": inference_started_ns,
                    "inference_completed_ns": time.monotonic_ns(),
                }
                if "timed out" in str(error):
                    self._inference_worker_failed = True
                    if self._worker.poll() is None:
                        self._worker.kill()
            with self._inference_condition:
                self._completed_inference = completion
                self._inference_busy = False
            self._inference_guard.trigger()

    @timed_callback
    def _inference_ready_callback(self) -> None:
        """Handle worker completion on the ROS executor without a timer wait."""
        self._update_policy()

    def _consume_inference_completion(self) -> None:
        with self._inference_condition:
            completion = self._completed_inference
            self._completed_inference = None
            self._inference_condition.notify()
        if completion is None:
            return
        request = completion["request"]
        handled_ns = time.monotonic_ns()
        completed_ns = int(completion["inference_completed_ns"])
        self._timing.record(
            "inference_result_handled",
            request_id=request["request_id"],
            enable_generation=request["enable_generation"],
            image_sequence=request["image_sequence"],
            scheduling_delay_ms=(
                int(completion["inference_started_ns"])
                - request["scheduled_monotonic_ns"]
            ) / 1e6,
            inference_ms=(completed_ns - completion["inference_started_ns"]) / 1e6,
            completion_to_handle_ms=(handled_ns - completed_ns) / 1e6,
            observation_age_ms=(
                handled_ns - request["oldest_image_receipt_monotonic_ns"]
            ) / 1e6,
        )
        if (
            not self._enabled
            or request["enable_generation"] != self._enable_generation
        ):
            self._timing.record(
                "inference_discarded", request_id=request["request_id"],
                enable_generation=request["enable_generation"],
                image_sequence=request["image_sequence"],
                reason="policy_disabled_or_generation_changed",
            )
            return
        if request["request_id"] <= self._last_accepted_request_id:
            self._timing.record(
                "inference_discarded", request_id=request["request_id"],
                enable_generation=request["enable_generation"],
                image_sequence=request["image_sequence"],
                reason="older_than_accepted_result",
            )
            return
        error = completion.get("error")
        if error:
            self._inference_worker_failed = True
            self._last_command = None
            self._pending_action_body = None
            with self._inference_condition:
                self._pending_inference = None
            self._last_error = f"inference_failed:{error}"
            self.get_logger().error(MSG_INFERENCE_FAILED.format(error=error))
            self._timing.record(
                "inference_error", request_id=request["request_id"],
                enable_generation=request["enable_generation"],
                image_sequence=request["image_sequence"],
                error=error,
            )
            return
        observation_age_ms = (
            handled_ns - request["oldest_image_receipt_monotonic_ns"]
        ) / 1e6
        if observation_age_ms > self._action_freshness_timeout_s * 1000.0:
            self._timing.record(
                "inference_discarded",
                request_id=request["request_id"],
                enable_generation=request["enable_generation"],
                image_sequence=request["image_sequence"],
                reason="action_exceeded_freshness_timeout",
                observation_age_ms=observation_age_ms,
                freshness_timeout_ms=self._action_freshness_timeout_s * 1000.0,
            )
            return
        action = completion["action"]
        self._last_command = body_action_to_ned(action, request["yaw"])
        self._pending_action_body = tuple(action)
        self._last_accepted_request_id = request["request_id"]
        self._inference_count += 1
        self._inference_position = request["position"]
        self._inferred_image_sequence = request["image_sequence"]
        self._action_origin = {
            "request_id": request["request_id"],
            "enable_generation": request["enable_generation"],
            "action_id": self._inference_count,
            "image_topic": IMAGE_TOPICS[self._requested_source],
            "image_topics": {
                source: IMAGE_TOPICS[source]
                for source in self._requested_sources
            },
            "image_sequence": request["image_sequence"],
            "image_sequences": request["image_sequences"],
            "image_header_stamp": request["image_header_stamp"],
            "image_header_stamps": request["image_header_stamps"],
            "image_receipt_monotonic_ns": request["image_receipt_monotonic_ns"],
            "oldest_image_receipt_monotonic_ns": request[
                "oldest_image_receipt_monotonic_ns"
            ],
            "image_receipts_monotonic_ns": request[
                "image_receipts_monotonic_ns"
            ],
            "image_receipt_ros_ns": request["image_receipt_ros_ns"],
            "observation_inputs": request["observation_inputs"],
            "inference_started_ns": completion["inference_started_ns"],
            "inference_completed_ns": completed_ns,
            "scheduled_monotonic_ns": request["scheduled_monotonic_ns"],
            "runtime_device": self._runtime_device,
        }
        self._action_publish_count = 0
        total_ms = (
            completed_ns - completion["inference_started_ns"]
        ) / 1e6
        self._timing.record(
            "inference_end",
            request_id=request["request_id"],
            enable_generation=request["enable_generation"],
            image_sequence=request["image_sequence"],
            inference_ms=total_ms,
            observation_age_ms=observation_age_ms,
            request_write_ms=completion.get("request_write_ms"),
            worker_wait_ms=completion.get("worker_wait_ms"),
            worker_profile=completion.get("worker_profile"),
            runtime_device=self._runtime_device,
            inference_count=self._inference_count,
            action_origin=self._action_origin,
            state8=request["state8"],
            action_body=action,
            command_ned=list(self._last_command),
        )
        if self._video_spool_dir is not None:
            video_job = (
                request["image"], self._inference_count, completed_ns,
                tuple(action),
            )
            try:
                assert self._video_capture_queue is not None
                self._video_capture_queue.put_nowait(video_job)
            except queue.Full:
                self._video_dropped_count += 1
                self._timing.record(
                    "policy_input_video_frame_dropped",
                    inference_count=self._inference_count,
                    drop_count=self._video_dropped_count,
                    reason="diagnostic_writer_queue_full",
                )

    def _video_capture_loop(self) -> None:
        """Write optional policy-input frames outside ROS control callbacks."""
        assert self._video_capture_queue is not None
        assert self._video_spool_dir is not None
        while True:
            job = self._video_capture_queue.get()
            if job is None:
                return
            image, inference_count, timestamp_ns, action = job
            try:
                capture_policy_input_frame(
                    self._video_spool_dir,
                    image_bytes=image,
                    image_source=self._requested_source,
                    inference_count=inference_count,
                    timestamp_ns=timestamp_ns,
                    action_body=action,
                )
            except (OSError, ValueError) as error:
                message = f"video_capture_failed:{error}"
                print(f"[BC Flight] {message}", file=sys.stderr, flush=True)

    def _publish_command(self) -> bool:
        assert self._last_command is not None
        if self._action_origin.get("enable_generation") != self._enable_generation:
            return False
        origin_ns = self._action_origin.get(
            "oldest_image_receipt_monotonic_ns",
            self._action_origin.get("image_receipt_monotonic_ns"),
        )
        if origin_ns is None:
            return False
        publish_ns = time.monotonic_ns()
        action_age_ms = (publish_ns - int(origin_ns)) / 1e6
        if action_age_ms > self._action_freshness_timeout_s * 1000.0:
            return False
        stamp_ros_ns = self.get_clock().now().nanoseconds
        if (
            self._last_command_stamp_ros_ns is not None
            and stamp_ros_ns <= self._last_command_stamp_ros_ns
        ):
            self._timing.record(
                "action_publish_rejected",
                request_id=self._action_origin["request_id"],
                enable_generation=self._enable_generation,
                reason="command_stamp_not_monotonic",
            )
            return False
        north, east, down, yaw_rate = self._last_command
        message = TwistStamped()
        message.header.stamp.sec = stamp_ros_ns // 1_000_000_000
        message.header.stamp.nanosec = stamp_ros_ns % 1_000_000_000
        message.header.frame_id = VALID_COMMAND_FRAME
        message.twist.linear.x = north
        message.twist.linear.y = east
        message.twist.linear.z = down
        message.twist.angular.z = yaw_rate
        self._timing.publish(
            SOURCE_TOPICS[BC_POLICY], message,
            request_id=self._action_origin["request_id"],
            action_id=self._action_origin["action_id"],
            enable_generation=self._action_origin["enable_generation"],
            image_sequence=self._inferred_image_sequence,
            inference_count=self._inference_count,
            action_origin=self._action_origin,
            action_age_ms=action_age_ms,
            completion_to_publish_ms=(
                publish_ns - self._action_origin["inference_completed_ns"]
            ) / 1e6,
            new_action_interval_ms=(
                None if self._action_publish_count > 0
                or self._last_new_action_publish_monotonic_ns is None
                else (publish_ns - self._last_new_action_publish_monotonic_ns) / 1e6
            ),
            command_publication_gap_ms=(
                None if self._last_action_publish_monotonic_ns is None
                else (publish_ns - self._last_action_publish_monotonic_ns) / 1e6
            ),
            repeated_publication_gap_ms=(
                None if self._action_publish_count == 0
                or self._last_action_publish_monotonic_ns is None
                else (publish_ns - self._last_action_publish_monotonic_ns) / 1e6
            ),
            command_stamp_role="publication_ros_time",
            repeated_action=self._action_publish_count > 0,
        )
        send_age_ms = (time.monotonic_ns() - int(origin_ns)) / 1e6
        if send_age_ms > self._action_freshness_timeout_s * 1000.0:
            self._timing.record(
                "action_publish_rejected",
                request_id=self._action_origin["request_id"],
                enable_generation=self._enable_generation,
                image_sequence=self._inferred_image_sequence,
                reason="action_exceeded_freshness_timeout_before_send",
                action_age_ms=send_age_ms,
            )
            return False
        self._timing.send(self._command_publisher, message)
        if self._action_publish_count == 0:
            self._last_new_action_publish_monotonic_ns = time.monotonic_ns()
        self._action_publish_count += 1
        self._last_action_publish_monotonic_ns = time.monotonic_ns()
        self._last_command_stamp_ros_ns = stamp_ros_ns
        if self._pending_action_body is not None:
            self._previous_action = self._pending_action_body
            self._pending_action_body = None
            self._timing.record(
                "previous_action_updated",
                request_id=self._action_origin["request_id"],
                enable_generation=self._enable_generation,
                image_sequence=self._inferred_image_sequence,
                reason="policy_publisher_send_returned",
            )
        return True

    def _publish_status(self, ready: bool, reason: str) -> None:
        action_age_ms = None
        action_origin = None
        if self._action_origin:
            action_age_ms = (
                time.monotonic_ns()
                - int(self._action_origin.get(
                    "oldest_image_receipt_monotonic_ns",
                    self._action_origin["image_receipt_monotonic_ns"],
                ))
            ) / 1e6
            action_origin = {
                name: self._action_origin[name]
                for name in (
                    "action_id", "request_id", "enable_generation",
                    "image_sequence", "image_header_stamp",
                    "image_receipt_ros_ns", "image_receipt_monotonic_ns",
                    "oldest_image_receipt_monotonic_ns",
                    "inference_started_ns", "inference_completed_ns",
                )
            }
        status = {
            "schema": POLICY_STATUS_SCHEMA,
            "enabled": self._enabled,
            "ready": ready,
            "reason": reason,
            "image_source": self._identity["image_source"],
            "image_sources": self._identity.get(
                "image_sources", [self._identity["image_source"]]
            ),
            "checkpoint_path": self._identity["checkpoint_path"],
            "checkpoint_sha256": self._identity["checkpoint_sha256"],
            "encoder_path": self._identity["encoder_path"],
            "encoder_sha256": self._identity["encoder_sha256"],
            "inference_count": self._inference_count,
            "inference_position": self._inference_position,
            "image_sequence": self._image_sequence,
            "runtime_device": self._runtime_device,
            "inference_in_flight": self._inference_busy,
            "action_origin": action_origin,
            "action_age_ms": action_age_ms,
        }
        message = String()
        message.data = json.dumps(
            status, sort_keys=True, separators=(",", ":")
        )
        self._timing.send(self._status_publisher, message)

    @timed_callback
    def _tick(self) -> None:
        self._timing.tick()
        self._update_policy()

    def _update_policy(self) -> None:
        """Keep the timer watchdog and completion wakeup on one safety path."""
        self._consume_inference_completion()
        now = self._now_seconds()
        reason = self._observation_error(now)
        if not self._enabled:
            self._publish_status(reason is None, "disabled")
            return
        if reason is not None:
            self._last_command = None
            self._pending_action_body = None
            self._last_error = reason
            with self._inference_condition:
                self._pending_inference = None
            if reason.startswith("waiting_for_") and not self._waiting_logged:
                self.get_logger().info(
                    MSG_WAITING_FOR_IMAGE.format(source=reason.removeprefix("waiting_for_"))
                )
                self._waiting_logged = True
            self._publish_status(False, reason)
            return
        try:
            self._schedule_inference()
            action_age_ms = None
            if self._action_origin:
                action_age_ms = (
                    time.monotonic_ns()
                    - int(self._action_origin.get(
                        "oldest_image_receipt_monotonic_ns",
                        self._action_origin["image_receipt_monotonic_ns"],
                    ))
                ) / 1e6
            action_fresh = bool(
                self._last_command is not None
                and not self._inference_worker_failed
                and action_age_ms is not None
                and action_age_ms <= self._action_freshness_timeout_s * 1000.0
            )
            if action_fresh:
                sent = self._publish_command()
                if not sent:
                    action_age_ms = (
                        time.monotonic_ns()
                        - int(self._action_origin.get(
                            "oldest_image_receipt_monotonic_ns",
                            self._action_origin["image_receipt_monotonic_ns"],
                        ))
                    ) / 1e6
                    action_fresh = bool(
                        self._action_publish_count > 0
                        and action_age_ms <= self._action_freshness_timeout_s * 1000.0
                    )
            if not action_fresh and self._last_command is not None and (
                action_age_ms is not None
                and action_age_ms > self._action_freshness_timeout_s * 1000.0
            ):
                expired_sequence = self._inferred_image_sequence
                self._last_command = None
                self._pending_action_body = None
                self._timing.record(
                    "action_expired",
                    request_id=self._action_origin.get("request_id"),
                    enable_generation=self._action_origin.get("enable_generation"),
                    image_sequence=expired_sequence,
                    action_age_ms=action_age_ms,
                    freshness_timeout_ms=(
                        self._action_freshness_timeout_s * 1000.0
                    ),
                )
            if not self._last_error.startswith("inference_failed:"):
                self._last_error = ""
            status_reason = (
                "ready" if action_fresh else "waiting_for_fresh_inference"
            )
            if self._inference_worker_failed:
                status_reason = "inference_worker_failed"
            self._publish_status(action_fresh, status_reason)
        except (AssertionError, ValueError, RuntimeError) as error:
            self._last_command = None
            self._pending_action_body = None
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
            code = self._worker.poll()
            raise RuntimeError(
                f"inference worker closed its output (exit code {code})"
            )
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError("inference worker response is not an object")
        return payload

    def destroy_node(self):
        """Stop the private ML worker before destroying ROS resources."""
        if hasattr(self, "_inference_condition"):
            with self._inference_condition:
                self._inference_stop = True
                self._pending_inference = None
                self._inference_condition.notify_all()
            if hasattr(self, "_inference_thread"):
                self._inference_thread.join(timeout=1.0)
        if self._video_capture_queue is not None:
            try:
                self._video_capture_queue.put_nowait(None)
            except queue.Full:
                while True:
                    try:
                        self._video_capture_queue.get_nowait()
                    except queue.Empty:
                        break
                self._video_capture_queue.put_nowait(None)
            if self._video_capture_thread is not None:
                self._video_capture_thread.join(timeout=1.0)
        if hasattr(self, "_worker") and self._worker.poll() is None:
            self._worker.terminate()
            try:
                self._worker.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._worker.kill()
                self._worker.wait(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> int:
    """Run the ROS-facing source-matched BC policy adapter."""
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
