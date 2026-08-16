"""Capture three in-flight Phase 10C frames for human visual QA only."""

from __future__ import annotations

import json
import math
import time
from io import BytesIO
from pathlib import Path

from PIL import Image

from nav_msgs.msg import Path as PathMessage

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage

from std_msgs.msg import String

from uav_interfaces.msg import Px4FlightStatus


FPV_TOPIC = "/uav/isaac/fpv/image/compressed"
OBSERVER_TOPIC = "/uav/isaac/observer/image/compressed"
DEPTH_TOPIC = "/uav/isaac/fpv/depth/compressed"
STATUS_TOPIC = "/uav/isaac/runtime_status"
PATH_TOPIC = "/uav/planner/path"
CAPTURE_PHASES = ("start", "mid_flight", "near_goal")
MID_FLIGHT_GOAL_DISTANCE_M = 3.6
NEAR_GOAL_DISTANCE_M = 1.0


def _timestamp_s(message: CompressedImage) -> float:
    return float(message.header.stamp.sec) + float(
        message.header.stamp.nanosec
    ) / 1_000_000_000.0


def normalized_depth_preview(payload: bytes) -> tuple[Image.Image, dict]:
    """Convert uint16-mm raw depth into an inverted 8-bit QA preview."""
    with Image.open(BytesIO(payload)) as source:
        depth = np.asarray(source, dtype=np.uint16)
    if depth.shape != (180, 320):
        raise ValueError(f"unexpected depth shape: {depth.shape}")
    valid = depth > 0
    if not valid.any():
        raise ValueError("depth preview has no valid pixels")
    values = depth[valid].astype(np.float32)
    near = float(np.percentile(values, 2.0))
    far = float(np.percentile(values, 98.0))
    if far <= near:
        far = near + 1.0
    clipped = np.clip(depth.astype(np.float32), near, far)
    normalized = np.rint((1.0 - (clipped - near) / (far - near)) * 255.0)
    normalized = np.clip(normalized, 0.0, 255.0).astype(np.uint8)
    normalized[~valid] = 0
    return Image.fromarray(normalized, mode="L"), {
        "unit": "millimeter",
        "encoding": "PNG uint16",
        "valid_range_mm": [50, 30000],
        "invalid_value": 0,
        "valid_pixel_count": int(valid.sum()),
        "minimum_valid_mm": int(values.min()),
        "maximum_valid_mm": int(values.max()),
        "preview_percentiles_mm": [near, far],
        "preview_mapping": "near=255, far=0, invalid=0",
    }


def _path_metrics(message: PathMessage) -> dict | None:
    points = [
        [
            float(pose.pose.position.x),
            float(pose.pose.position.y),
            float(pose.pose.position.z),
        ]
        for pose in message.poses
    ]
    if len(points) < 2 or not all(
        math.isfinite(value) for point in points for value in point
    ):
        return None
    length = sum(
        math.dist(left[:2], right[:2])
        for left, right in zip(points, points[1:])
    )
    direct = math.dist(points[0][:2], points[-1][:2])
    return {
        "frame_id": message.header.frame_id,
        "point_count": len(points),
        "path_length_xy_m": length,
        "direct_distance_xy_m": direct,
        "detour_distance_xy_m": length - direct,
        "detour_ratio": length / direct if direct > 1e-6 else None,
        "points": points,
    }


class VisualQACapture(Node):
    """Save three in-flight QA frames without controlling the flight."""

    def __init__(self) -> None:
        """Create read-only subscriptions and a bounded capture timer."""
        super().__init__("phase10c_visual_qa_capture")
        self.declare_parameter("episode_id", "episode_000001")
        self.declare_parameter("random_seed", 102001)
        self.declare_parameter(
            "output_root",
            "artifacts/visual_qa/phase10c_highrise_rigid_fpv",
        )
        self.declare_parameter("timeout_s", 90.0)
        self.episode_id = str(self.get_parameter("episode_id").value)
        self.random_seed = int(self.get_parameter("random_seed").value)
        self.output_root = Path(
            str(self.get_parameter("output_root").value)
        ).expanduser().resolve()
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.started = time.monotonic()
        self.exit_code = 1
        self.finished = False
        self.scene: dict | None = None
        self.flight_status: dict | None = None
        self.planner_path: dict | None = None
        self.phase_index = -1
        self.pending: dict[str, CompressedImage] = {}
        self.captures: dict[str, dict] = {}
        self.create_subscription(String, STATUS_TOPIC, self._status, 10)
        self.create_subscription(CompressedImage, FPV_TOPIC, self._fpv, 10)
        self.create_subscription(
            CompressedImage, OBSERVER_TOPIC, self._observer, 10
        )
        self.create_subscription(CompressedImage, DEPTH_TOPIC, self._depth, 10)
        self.create_subscription(
            Px4FlightStatus,
            "/uav/px4/flight_status",
            self._flight_status,
            10,
        )
        self.create_subscription(PathMessage, PATH_TOPIC, self._path, 10)
        self.create_timer(0.05, self._tick)

    def _status(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        scene = status.get("scene_configuration")
        if (
            status.get("episode_id") == self.episode_id
            and status.get("random_seed") == self.random_seed
            and isinstance(scene, dict)
        ):
            self.scene = scene

    def _fpv(self, message: CompressedImage) -> None:
        self._accept_stream("fpv_rgb", message)

    def _observer(self, message: CompressedImage) -> None:
        self._accept_stream("observer_rgb", message)

    def _depth(self, message: CompressedImage) -> None:
        self._accept_stream("fpv_depth_raw", message)

    def _accept_stream(self, name: str, message: CompressedImage) -> None:
        if self.phase_index >= 0 and name not in self.pending:
            self.pending[name] = message

    def _path(self, message: PathMessage) -> None:
        metrics = _path_metrics(message)
        if metrics is not None:
            self.planner_path = metrics

    def _flight_status(self, message: Px4FlightStatus) -> None:
        self.flight_status = {
            "state": message.state,
            "tracking_active": bool(message.tracking_active),
            "astar_selected": bool(message.astar_selected),
            "altitude_m": float(message.altitude_m),
            "goal_distance_m": float(message.goal_distance_m),
        }
        if message.state != "TRACKING" or not message.tracking_active:
            return
        goal_distance = float(message.goal_distance_m)
        if self.phase_index < 0:
            self._begin_phase(0)
        elif (
            self.phase_index == 0
            and "start" in self.captures
            and goal_distance <= MID_FLIGHT_GOAL_DISTANCE_M
        ):
            self._begin_phase(1)
        elif (
            self.phase_index == 1
            and "mid_flight" in self.captures
            and goal_distance <= NEAR_GOAL_DISTANCE_M
        ):
            self._begin_phase(2)

    def _begin_phase(self, index: int) -> None:
        self.phase_index = index
        self.pending = {}
        self.get_logger().info(f"capturing phase={CAPTURE_PHASES[index]}")

    def _tick(self) -> None:
        if self.finished:
            return
        if time.monotonic() - self.started > self.timeout_s:
            missing = sorted(set(CAPTURE_PHASES) - set(self.captures))
            self._finish(1, f"timed out; missing phases={missing}")
            return
        if self.phase_index < 0 or len(self.pending) != 3:
            return
        phase = CAPTURE_PHASES[self.phase_index]
        if phase not in self.captures:
            timestamps = [_timestamp_s(item) for item in self.pending.values()]
            self.captures[phase] = {
                "flight_status": dict(self.flight_status or {}),
                "messages": dict(self.pending),
                "capture_span_s": max(timestamps) - min(timestamps),
            }
            self.get_logger().info(f"captured phase={phase}")
        if len(self.captures) != len(CAPTURE_PHASES):
            return
        try:
            output = self._write_capture()
        except (KeyError, OSError, TypeError, ValueError) as error:
            self._finish(1, f"capture validation failed: {error}")
            return
        self._finish(0, f"saved={output}")

    @staticmethod
    def _validate_payloads(messages: dict[str, CompressedImage]) -> None:
        if not bytes(messages["fpv_rgb"].data).startswith(b"\xff\xd8"):
            raise ValueError("FPV stream is not JPEG")
        if not bytes(messages["observer_rgb"].data).startswith(b"\xff\xd8"):
            raise ValueError("Observer stream is not JPEG")
        if not bytes(messages["fpv_depth_raw"].data).startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise ValueError("depth stream is not PNG")

    def _write_capture(self) -> Path:
        assert self.scene is not None
        if len(self.scene.get("obstacles", [])) != 8:
            raise ValueError("normal QA scene must contain eight obstacles")
        if self.planner_path is None:
            raise ValueError("validated planner path was not observed")
        directory = self.output_root / (
            f"{self.episode_id}_seed_{self.random_seed}"
        )
        if directory.exists():
            raise FileExistsError(f"refusing to overwrite {directory}")
        directory.mkdir(parents=True)
        capture_metadata = {}
        for phase in CAPTURE_PHASES:
            capture = self.captures[phase]
            messages = capture["messages"]
            self._validate_payloads(messages)
            fpv_name = f"fpv_rgb_{phase}.jpg"
            observer_name = f"observer_rgb_{phase}.jpg"
            depth_name = f"fpv_depth_raw_{phase}.png"
            preview_name = f"fpv_depth_preview_{phase}.png"
            (directory / fpv_name).write_bytes(bytes(messages["fpv_rgb"].data))
            (directory / observer_name).write_bytes(
                bytes(messages["observer_rgb"].data)
            )
            depth_payload = bytes(messages["fpv_depth_raw"].data)
            (directory / depth_name).write_bytes(depth_payload)
            preview, depth_metadata = normalized_depth_preview(depth_payload)
            preview.save(directory / preview_name, format="PNG")
            capture_metadata[phase] = {
                "flight_status": capture["flight_status"],
                "stream_capture_span_s": capture["capture_span_s"],
                "images": {
                    "fpv_rgb": {
                        "path": fpv_name,
                        "timestamp_s": _timestamp_s(messages["fpv_rgb"]),
                        "resolution": [320, 180],
                        "format": "JPEG",
                    },
                    "observer_rgb": {
                        "path": observer_name,
                        "timestamp_s": _timestamp_s(messages["observer_rgb"]),
                        "resolution": [320, 180],
                        "format": "JPEG",
                    },
                    "fpv_depth_raw": {
                        "path": depth_name,
                        "timestamp_s": _timestamp_s(
                            messages["fpv_depth_raw"]
                        ),
                        **depth_metadata,
                    },
                    "fpv_depth_preview": {
                        "path": preview_name,
                        "format": "PNG 8-bit normalized visual QA only",
                    },
                },
            }
        metadata = {
            "phase": "10C",
            "purpose": "three-phase visual QA only; not training data",
            "episode_id": self.episode_id,
            "random_seed": self.random_seed,
            "scene_configuration": self.scene,
            "obstacle_count": len(self.scene["obstacles"]),
            "planner_path": self.planner_path,
            "camera_contract": {
                "fpv": {
                    "direction": "BODY_AXIS +X",
                    "forward_offset_m": 0.45,
                    "height_m": 0.12,
                    "look_ahead_m": 3.5,
                    "look_down_m": -0.8,
                    "focal_length": 12.0,
                    "horizontal_aperture": 28.0,
                    "position_smoothing": "disabled_rigid_body_mount",
                },
                "observer": {
                    "mode": "TOP",
                    "height_m": 9.0,
                    "look_at_height_m": 0.0,
                    "focal_length": 18.0,
                    "horizontal_aperture": 22.0,
                    "position_smoothing": 0.18,
                },
            },
            "captures": capture_metadata,
        }
        (directory / "scene_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return directory

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
            f"PHASE10C_VISUAL_QA_RESULT success={str(code == 0).lower()} "
            f"detail={detail}"
        )
        rclpy.shutdown()


def main(args=None) -> int:
    """Capture one scene and exit without exposing any control publisher."""
    rclpy.init(args=args)
    node = VisualQACapture()
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
