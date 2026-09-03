"""Record one synchronized Phase 9 ASTAR_EXPERT episode as BC dataset V1."""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections import Counter, deque
from pathlib import Path

from geometry_msgs.msg import PoseStamped, TwistStamped

from nav_msgs.msg import Odometry, Path as PathMessage

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import CompressedImage

from std_msgs.msg import String

from uav_data_recorder.expert_dataset_contract import (
    CSV_FIELDS,
    DATASET_VERSION,
    SAMPLE_RATE_HZ,
    SYNCHRONIZATION_TOLERANCE_S,
    TimedValue,
    contract_manifest,
    episode_outcome_success,
    goal_features,
    latest_at_or_before,
    nearest,
    ned_to_body,
    normalize_action,
    previous,
    recording_window_rejection,
    timestamp_seconds,
    update_recording_window,
    yaw_from_quaternion,
)

from uav_interfaces.msg import ControlMuxStatus, Px4FlightStatus


IMAGE_TOPIC = "/uav/isaac/fpv/image/compressed"
OBSERVER_IMAGE_TOPIC = "/uav/isaac/observer/image/compressed"
DEPTH_TOPIC = "/uav/isaac/fpv/depth/compressed"
RUNTIME_STATUS_TOPIC = "/uav/isaac/runtime_status"
ODOMETRY_TOPIC = "/uav/vehicle/odometry"
EXPERT_COMMAND_TOPIC = "/uav/control/astar_command"
MUX_STATUS_TOPIC = "/uav/control/mux_status"
FLIGHT_STATUS_TOPIC = "/uav/px4/flight_status"
GOAL_TOPIC = "/uav/scene/goal"
EPISODE_ID = "episode_000001"
AUXILIARY_FIELDS = (
    "episode_id", "sample_id", "primary_image_timestamp_s",
    "observer_rgb_available", "observer_rgb_timestamp_s",
    "observer_rgb_error_s", "observer_rgb_path", "observer_rgb_status",
    "fpv_depth_available",
    "fpv_depth_timestamp_s", "fpv_depth_error_s", "fpv_depth_path",
    "fpv_depth_status",
)
OBSERVER_SYNCHRONIZATION_TOLERANCE_S = 0.35
RUNTIME_TO_DATASET_STATUS_FIELDS = {
    "fpv_rgb_enabled": "phase10a_camera_enabled",
    "fpv_rgb_ready": "phase10a_camera_ready",
    "fpv_rgb_error": "phase10a_camera_error",
    "observer_rgb_enabled": "phase10c_observer_rgb_enabled",
    "observer_rgb_ready": "phase10c_observer_rgb_ready",
    "observer_rgb_error": "phase10c_observer_rgb_error",
    "observer_mode": "phase10c_observer_mode",
    "fpv_depth_enabled": "phase10b_fpv_depth_enabled",
    "fpv_depth_ready": "phase10b_fpv_depth_ready",
    "fpv_depth_error": "phase10b_fpv_depth_error",
}


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class ExpertDatasetRecorderNode(Node):
    """Join image/state/expert streams and finalize exactly one episode."""

    def __init__(self) -> None:
        """Create output paths, buffers, subscriptions, and join timer."""
        super().__init__("expert_dataset_recorder")
        self.declare_parameter(
            "dataset_root", "artifacts/datasets/bc_expert_v1"
        )
        self.declare_parameter("episode_id", EPISODE_ID)
        self.declare_parameter("collection_mode", "single")
        self.declare_parameter("random_seed", 0)
        self.declare_parameter("expected_runtime_generation", -1)
        self.declare_parameter("expected_scene_revision", 0)
        self.declare_parameter("minimum_fpv_frame_count", 0)
        self.declare_parameter("minimum_observer_frame_count", 0)
        self.declare_parameter("minimum_depth_frame_count", 0)
        self.declare_parameter(
            "synchronization_tolerance_s", SYNCHRONIZATION_TOLERANCE_S
        )
        self.dataset_root = Path(
            str(self.get_parameter("dataset_root").value)
        ).expanduser().resolve()
        self.episode_id = str(self.get_parameter("episode_id").value)
        self.collection_mode = str(
            self.get_parameter("collection_mode").value
        )
        self.random_seed = int(self.get_parameter("random_seed").value)
        self.expected_runtime_generation = int(
            self.get_parameter("expected_runtime_generation").value
        )
        self.expected_scene_revision = int(
            self.get_parameter("expected_scene_revision").value
        )
        self._minimum_camera_counts = {
            "fpv_rgb": int(
                self.get_parameter("minimum_fpv_frame_count").value
            ),
            "observer_rgb": int(
                self.get_parameter("minimum_observer_frame_count").value
            ),
            "fpv_depth": int(
                self.get_parameter("minimum_depth_frame_count").value
            ),
        }
        self.tolerance_s = float(
            self.get_parameter("synchronization_tolerance_s").value
        )
        if not re.fullmatch(r"episode_[0-9]{6,}", self.episode_id):
            raise ValueError(
                "episode_id must use episode_ followed by at least six digits"
            )
        if self.collection_mode not in {"single", "batch"}:
            raise ValueError("collection_mode must be single or batch")
        if self.collection_mode == "single" and self.episode_id != EPISODE_ID:
            raise ValueError("Phase 10A single mode records episode_000001")
        if not math.isfinite(self.tolerance_s) or self.tolerance_s <= 0.0:
            raise ValueError("synchronization tolerance must be positive")
        if abs(self.tolerance_s - SYNCHRONIZATION_TOLERANCE_S) > 1e-9:
            raise ValueError(
                "Phase 10A synchronization tolerance is fixed at 0.100 s"
            )

        self.episode_dir = self.dataset_root / self.episode_id
        if self.episode_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite existing episode: {self.episode_dir}"
            )
        self.fpv_rgb_dir = self.episode_dir / "fpv_rgb"
        self.fpv_rgb_dir.mkdir(parents=True)
        self.started_utc = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        self._write_initial_metadata()

        self._states: deque[TimedValue] = deque(maxlen=400)
        self._actions: deque[TimedValue] = deque(maxlen=400)
        self._mux: deque[TimedValue] = deque(maxlen=400)
        self._flight: deque[TimedValue] = deque(maxlen=400)
        self._images: deque[TimedValue] = deque(maxlen=20)
        self._observer_images: deque[TimedValue] = deque(maxlen=20)
        self._depth_images: deque[TimedValue] = deque(maxlen=20)
        self._goal: PoseStamped | None = None
        self._rows: list[dict] = []
        self._auxiliary_rows: list[dict] = []
        self._rejections: Counter[str] = Counter()
        self._timeline: list[dict] = []
        self._last_phase = ""
        self._last_status: Px4FlightStatus | None = None
        self._recording_start_timestamp_s: float | None = None
        self._recording_end_timestamp_s: float | None = None
        self._observed_goal_reached = False
        self._observed_landing_commanded = False
        self._observed_landed_after_landing = False
        self._finalized = False
        self._scene_configuration: dict | None = None
        self._sensor_runtime_status: dict = {}
        self._boundary_guard_enabled = bool(
            self.expected_runtime_generation >= 0
            or self.expected_scene_revision > 0
        )
        self._episode_boundary_ready = not self._boundary_guard_enabled
        self._episode_boundary_timestamp_s: float | None = None
        self._boundary_discarded = Counter()
        self._planner_path: dict | None = None
        self._planner_status = ""
        self._fpv_rgb_received = 0
        self._observer_rgb_received = 0
        self._fpv_depth_received = 0
        self._stream_time_bounds: dict[str, list[float | None]] = {
            "fpv_rgb": [None, None],
            "observer_rgb": [None, None],
            "fpv_depth": [None, None],
        }

        live = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
        durable = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            CompressedImage, IMAGE_TOPIC, self._image_callback, live
        )
        self.create_subscription(
            CompressedImage,
            OBSERVER_IMAGE_TOPIC,
            self._observer_image_callback,
            live,
        )
        self.create_subscription(
            CompressedImage, DEPTH_TOPIC, self._depth_image_callback, live
        )
        self.create_subscription(
            String, RUNTIME_STATUS_TOPIC, self._runtime_status_callback, live
        )
        self.create_subscription(
            Odometry, ODOMETRY_TOPIC, self._state_callback, live
        )
        self.create_subscription(
            TwistStamped, EXPERT_COMMAND_TOPIC, self._action_callback, live
        )
        self.create_subscription(
            ControlMuxStatus, MUX_STATUS_TOPIC, self._mux_callback, live
        )
        self.create_subscription(
            Px4FlightStatus, FLIGHT_STATUS_TOPIC, self._flight_callback, live
        )
        self.create_subscription(
            PoseStamped, GOAL_TOPIC, self._goal_callback, durable
        )
        self.create_subscription(
            PathMessage,
            "/uav/planner/path",
            self._planner_path_callback,
            durable,
        )
        self.create_subscription(
            String,
            "/uav/planner/status",
            self._planner_status_callback,
            durable,
        )
        self._timer = self.create_timer(0.02, self._process_ready_images)
        self._progress_timer = self.create_timer(1.0, self._write_progress)
        self.get_logger().info(
            f"EXPERT_RECORDER_READY root={self.dataset_root} "
            f"episode={self.episode_id} mode={self.collection_mode} "
            f"rate={SAMPLE_RATE_HZ:.1f}Hz tolerance={self.tolerance_s:.3f}s"
        )

    def _write_initial_metadata(self) -> None:
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.dataset_root / "dataset_manifest.json"
        if self.collection_mode == "single" or not manifest_path.exists():
            manifest = contract_manifest()
            manifest.update({
                "created_utc": self.started_utc,
                "collection_mode": self.collection_mode,
                "episodes": [],
                "episode_count": 0,
                "sample_count": 0,
                "status": "recording",
            })
            _atomic_json(manifest_path, manifest)
        _atomic_json(self.episode_dir / "episode.json", {
            "dataset_version": DATASET_VERSION,
            "episode_id": self.episode_id,
            "random_seed": self.random_seed,
            "started_utc": self.started_utc,
            "status": "recording",
            "success": False,
            "failure": "episode not finalized",
        })

    @staticmethod
    def _timed(message) -> TimedValue | None:
        try:
            return TimedValue(timestamp_seconds(message.header.stamp), message)
        except ValueError:
            return None

    def _state_callback(self, message: Odometry) -> None:
        item = self._timed(message)
        if item is not None and message.header.frame_id == "px4_ned":
            self._states.append(item)

    def _action_callback(self, message: TwistStamped) -> None:
        item = self._timed(message)
        if item is not None and message.header.frame_id == "px4_ned":
            self._actions.append(item)

    def _mux_callback(self, message: ControlMuxStatus) -> None:
        item = self._timed(message)
        if item is not None:
            self._mux.append(item)

    def _flight_callback(self, message: Px4FlightStatus) -> None:
        item = self._timed(message)
        if item is None:
            return
        self._flight.append(item)
        self._last_status = message
        (
            self._recording_start_timestamp_s,
            self._recording_end_timestamp_s,
        ) = update_recording_window(
            self._recording_start_timestamp_s,
            self._recording_end_timestamp_s,
            message.state,
            item.timestamp_s,
        )
        self._observed_goal_reached |= bool(message.goal_reached)
        self._observed_landing_commanded |= bool(message.landing_commanded)
        self._observed_landed_after_landing |= bool(
            message.landed and message.landing_commanded
        )
        if message.state != self._last_phase:
            self._timeline.append({
                "timestamp_s": item.timestamp_s,
                "phase": message.state,
                "altitude_m": float(message.altitude_m),
                "goal_distance_m": float(message.goal_distance_m),
                "offboard": bool(message.offboard_active),
                "armed": bool(message.vehicle_armed),
                "landed": bool(message.landed),
            })
            self._last_phase = message.state

    def _goal_callback(self, message: PoseStamped) -> None:
        if message.header.frame_id == "isaac_world":
            self._goal = message

    def _image_callback(self, message: CompressedImage) -> None:
        item = self._timed(message)
        if item is None:
            self._rejections["invalid_image_timestamp"] += 1
            return
        if not self._image_after_boundary("fpv_rgb", item.timestamp_s):
            return
        self._fpv_rgb_received += 1
        self._observe_stream_time("fpv_rgb", item.timestamp_s)
        self._images.append(TimedValue(item.timestamp_s, bytes(message.data)))

    def _observer_image_callback(self, message: CompressedImage) -> None:
        item = self._timed(message)
        if item is None or not self._image_after_boundary(
            "observer_rgb", item.timestamp_s
        ):
            return
        self._observer_rgb_received += 1
        self._observe_stream_time("observer_rgb", item.timestamp_s)
        self._observer_images.append(TimedValue(
            item.timestamp_s, (bytes(message.data), message.format)
        ))

    def _depth_image_callback(self, message: CompressedImage) -> None:
        item = self._timed(message)
        if item is None or not self._image_after_boundary(
            "fpv_depth", item.timestamp_s
        ):
            return
        self._fpv_depth_received += 1
        self._observe_stream_time("fpv_depth", item.timestamp_s)
        self._depth_images.append(TimedValue(
            item.timestamp_s, (bytes(message.data), message.format)
        ))

    def _image_after_boundary(self, name: str, timestamp_s: float) -> bool:
        if not self._episode_boundary_ready:
            self._boundary_discarded[name] += 1
            return False
        boundary = self._episode_boundary_timestamp_s
        if boundary is not None and timestamp_s < boundary:
            self._boundary_discarded[name] += 1
            return False
        return True

    def _runtime_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(status, dict):
            return
        self._sensor_runtime_status = {
            dataset_key: status.get(runtime_key, status.get(dataset_key))
            for runtime_key, dataset_key
            in RUNTIME_TO_DATASET_STATUS_FIELDS.items()
        }
        episode_matches = (
            status.get("episode_id") == self.episode_id
            and status.get("random_seed") == self.random_seed
            and isinstance(status.get("scene_configuration"), dict)
        )
        generation_matches = bool(
            self.expected_runtime_generation < 0
            or status.get("runtime_generation")
            == self.expected_runtime_generation
        )
        revision_matches = bool(
            self.expected_scene_revision <= 0
            or status.get("scene_revision") == self.expected_scene_revision
        )
        camera_counts_advanced = bool(
            not self._boundary_guard_enabled
            or all((
                int(status.get(runtime_name, 0))
                > self._minimum_camera_counts[dataset_name]
            ) for runtime_name, dataset_name in (
                ("fpv_rgb_frame_count", "fpv_rgb"),
                ("observer_rgb_frame_count", "observer_rgb"),
                ("fpv_depth_frame_count", "fpv_depth"),
            ))
        )
        if (
            episode_matches
            and generation_matches
            and revision_matches
            and camera_counts_advanced
        ):
            self._scene_configuration = status["scene_configuration"]
            if not self._episode_boundary_ready:
                self._episode_boundary_timestamp_s = (
                    self.get_clock().now().nanoseconds / 1e9
                )
            self._episode_boundary_ready = True

    def _planner_path_callback(self, message: PathMessage) -> None:
        points = [pose.pose.position for pose in message.poses]
        path_length = sum(
            math.hypot(right.x - left.x, right.y - left.y)
            for left, right in zip(points, points[1:])
        )
        self._planner_path = {
            "frame_id": message.header.frame_id,
            "point_count": len(points),
            "path_length_xy_m": path_length,
            "start": (
                None if not points
                else [points[0].x, points[0].y, points[0].z]
            ),
            "goal": (
                None if not points
                else [points[-1].x, points[-1].y, points[-1].z]
            ),
        }

    def _planner_status_callback(self, message: String) -> None:
        self._planner_status = message.data

    def _observe_stream_time(self, name: str, timestamp_s: float) -> None:
        bounds = self._stream_time_bounds[name]
        if bounds[0] is None:
            bounds[0] = timestamp_s
        bounds[1] = timestamp_s

    def _stream_rate(self, name: str, count: int) -> float:
        first, last = self._stream_time_bounds[name]
        if count < 2 or first is None or last is None or last <= first:
            return 0.0
        return (count - 1) / (last - first)

    def _write_progress(self) -> None:
        if self._finalized:
            return
        status = self._last_status
        _atomic_json(self.episode_dir / "progress.json", {
            "episode_id": self.episode_id,
            "state": "INITIALIZING" if status is None else status.state,
            "goal_distance_m": (
                None if status is None else float(status.goal_distance_m)
            ),
            "accepted_samples": len(self._rows),
            "rejected_samples": sum(self._rejections.values()),
            "rejections_by_reason": dict(sorted(self._rejections.items())),
            "fpv_rgb_received": self._fpv_rgb_received,
            "observer_rgb_received": self._observer_rgb_received,
            "fpv_depth_received": self._fpv_depth_received,
            "episode_boundary_ready": self._episode_boundary_ready,
            "runtime_generation": self.expected_runtime_generation,
            "scene_revision": self.expected_scene_revision,
        })

    def _process_ready_images(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        while (
            self._images
            and now - self._images[0].timestamp_s >= self.tolerance_s
        ):
            if (
                self._recording_end_timestamp_s is None
                and not any(
                    item.timestamp_s >= self._images[0].timestamp_s
                    for item in self._flight
                )
            ):
                break
            self._process_image(self._images.popleft())

    def _reject(self, reason: str) -> None:
        self._rejections[reason] += 1

    def _process_image(self, image: TimedValue) -> None:
        window_rejection = recording_window_rejection(
            self._recording_start_timestamp_s,
            self._recording_end_timestamp_s,
            image.timestamp_s,
        )
        if window_rejection is not None:
            self._reject(window_rejection)
            return
        if self._goal is None:
            self._reject("goal_missing")
            return
        state = nearest(list(self._states), image.timestamp_s)
        action = nearest(list(self._actions), image.timestamp_s)
        mux = nearest(list(self._mux), image.timestamp_s)
        flight = latest_at_or_before(list(self._flight), image.timestamp_s)
        joined = {
            "state": state, "action": action, "mux": mux, "flight": flight
        }
        missing = next(
            (name for name, item in joined.items() if item is None), None
        )
        if missing is not None:
            self._reject(f"{missing}_missing")
            return
        for name, item in joined.items():
            assert item is not None
            if abs(item.timestamp_s - image.timestamp_s) > self.tolerance_s:
                self._reject(f"{name}_over_tolerance")
                return
        assert state is not None and action is not None
        assert mux is not None and flight is not None
        mux_message = mux.value
        flight_message = flight.value
        if (
            mux_message.active_source != "ASTAR_EXPERT"
            or not mux_message.selected_command_valid
            or mux_message.hold_active
        ):
            self._reject("mux_not_astar_expert")
            return
        if (
            flight_message.state != "TRACKING"
            or not flight_message.follower_command_valid
            or not flight_message.astar_selected
        ):
            self._reject("not_tracking")
            return
        prior_action = previous(list(self._actions), action)
        if prior_action is None:
            self._reject("previous_action_missing")
            return
        prior_state = nearest(list(self._states), prior_action.timestamp_s)
        if (
            prior_state is None
            or abs(prior_state.timestamp_s - prior_action.timestamp_s)
            > self.tolerance_s
        ):
            self._reject("previous_action_state_over_tolerance")
            return
        try:
            row = self._build_row(
                image, state, action, prior_action, prior_state, flight
            )
        except (ValueError, TypeError, OverflowError) as error:
            self.get_logger().warning(
                f"rejecting invalid synchronized sample: {error}"
            )
            self._reject("invalid_values")
            return
        if not image.value.startswith(b"\xff\xd8"):
            self._reject("image_not_jpeg")
            return
        sample_id = len(self._rows) + 1
        relative = (
            Path(self.episode_id)
            / "fpv_rgb"
            / f"frame_{sample_id:06d}.jpg"
        )
        output = self.dataset_root / relative
        output.write_bytes(image.value)
        row["sample_id"] = sample_id
        row["image_path"] = str(relative)
        self._rows.append(row)
        self._record_auxiliary(image, sample_id)

    def _record_auxiliary(self, image: TimedValue, sample_id: int) -> None:
        row = {
            "episode_id": self.episode_id,
            "sample_id": sample_id,
            "primary_image_timestamp_s": image.timestamp_s,
        }
        specifications = (
            (
                "observer_rgb", "top_rgb", self._observer_images,
                OBSERVER_SYNCHRONIZATION_TOLERANCE_S, b"\xff\xd8", ".jpg",
            ),
            (
                "fpv_depth", "fpv_depth", self._depth_images,
                self.tolerance_s, b"\x89PNG\r\n\x1a\n", ".png",
            ),
        )
        for (
            name, directory_name, buffer, tolerance, signature, suffix
        ) in specifications:
            selected = nearest(list(buffer), image.timestamp_s)
            error = (
                None if selected is None
                else abs(selected.timestamp_s - image.timestamp_s)
            )
            available = bool(
                selected is not None
                and error is not None
                and error <= tolerance
                and selected.value[0].startswith(signature)
            )
            row[f"{name}_available"] = available
            row[f"{name}_timestamp_s"] = (
                "" if selected is None else selected.timestamp_s
            )
            row[f"{name}_error_s"] = "" if error is None else error
            row[f"{name}_path"] = ""
            if available:
                directory = self.episode_dir / directory_name
                directory.mkdir(exist_ok=True)
                relative = Path(self.episode_id) / directory_name / (
                    f"frame_{sample_id:06d}{suffix}"
                )
                (self.dataset_root / relative).write_bytes(selected.value[0])
                row[f"{name}_path"] = str(relative)
                row[f"{name}_status"] = "matched"
            elif selected is None:
                row[f"{name}_status"] = "stream_unavailable"
            elif error is not None and error > tolerance:
                row[f"{name}_status"] = "over_tolerance"
            else:
                row[f"{name}_status"] = "invalid_format"
        self._auxiliary_rows.append(row)

    def _build_row(
        self,
        image: TimedValue,
        state: TimedValue,
        action: TimedValue,
        prior_action: TimedValue,
        prior_state: TimedValue,
        flight: TimedValue,
    ) -> dict:
        odometry = state.value
        orientation = odometry.pose.pose.orientation
        yaw = yaw_from_quaternion(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        velocity = odometry.twist.twist.linear
        velocity_body = ned_to_body(velocity.x, velocity.y, yaw)
        goal = self._goal.pose.position
        goal_body = goal_features(
            odometry.pose.pose.position.x,
            odometry.pose.pose.position.y,
            goal.y,
            goal.x,
            yaw,
        )
        command = action.value.twist
        expert_body = ned_to_body(command.linear.x, command.linear.y, yaw)
        expert_physical = (expert_body[0], expert_body[1], command.angular.z)
        expert_target = normalize_action(*expert_physical)

        previous_odometry = prior_state.value
        previous_orientation = previous_odometry.pose.pose.orientation
        previous_yaw = yaw_from_quaternion(
            previous_orientation.x,
            previous_orientation.y,
            previous_orientation.z,
            previous_orientation.w,
        )
        prior_command = prior_action.value.twist
        previous_body = ned_to_body(
            prior_command.linear.x, prior_command.linear.y, previous_yaw
        )
        previous_physical = (
            previous_body[0], previous_body[1], prior_command.angular.z
        )
        previous_target = normalize_action(*previous_physical)
        values = (
            image.timestamp_s,
            state.timestamp_s,
            action.timestamp_s,
            prior_action.timestamp_s,
            *velocity_body,
            *goal_body,
            *expert_physical,
            *expert_target,
            *previous_physical,
            *previous_target,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("sample contains non-finite values")
        position = odometry.pose.pose.position
        return {
            "episode_id": self.episode_id,
            "sample_id": 0,
            "image_timestamp_s": image.timestamp_s,
            "image_path": "",
            "state_timestamp_s": state.timestamp_s,
            "expert_action_timestamp_s": action.timestamp_s,
            "previous_action_timestamp_s": prior_action.timestamp_s,
            "state_image_error_s": abs(state.timestamp_s - image.timestamp_s),
            "expert_action_image_error_s": abs(
                action.timestamp_s - image.timestamp_s
            ),
            "position_north_m": float(position.x),
            "position_east_m": float(position.y),
            "position_down_m": float(position.z),
            "yaw_ned_rad": yaw,
            "body_velocity_forward_mps": velocity_body[0],
            "body_velocity_right_mps": velocity_body[1],
            "goal_direction_forward": goal_body[0],
            "goal_direction_right": goal_body[1],
            "raw_goal_distance_m": goal_body[2],
            "normalized_goal_distance": goal_body[3],
            "expert_v_forward_mps": expert_physical[0],
            "expert_v_right_mps": expert_physical[1],
            "expert_yaw_rate_radps": expert_physical[2],
            "expert_action_forward": expert_target[0],
            "expert_action_right": expert_target[1],
            "expert_action_yaw_rate": expert_target[2],
            "previous_v_forward_mps": previous_physical[0],
            "previous_v_right_mps": previous_physical[1],
            "previous_yaw_rate_radps": previous_physical[2],
            "previous_action_forward": previous_target[0],
            "previous_action_right": previous_target[1],
            "previous_action_yaw_rate": previous_target[2],
            "mission_phase": flight.value.state,
            "success": False,
            "failure": "episode not finalized",
        }

    def finalize(self) -> None:
        """Flush pending images and publish final episode metadata."""
        if self._finalized:
            return
        self._finalized = True
        while self._images:
            self._process_image(self._images.popleft())
        status = self._last_status
        success = episode_outcome_success(
            "" if status is None else status.state,
            (
                "flight status unavailable"
                if status is None else status.failure_reason
            ),
            self._observed_goal_reached,
            self._observed_landing_commanded,
            self._observed_landed_after_landing,
        )
        if self._boundary_guard_enabled and not self._episode_boundary_ready:
            success = False
            failure = "persistent runtime recorder boundary was not ready"
        else:
            failure = "" if success else (
                "flight status unavailable" if status is None else
                status.failure_reason
                or f"terminal flight state was {status.state}"
            )
        for row in self._rows:
            row["success"] = success
            row["failure"] = failure
        with (self.episode_dir / "samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self._rows)
        with (self.episode_dir / "auxiliary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=AUXILIARY_FIELDS)
            writer.writeheader()
            writer.writerows(self._auxiliary_rows)
        image_times = [float(row["image_timestamp_s"]) for row in self._rows]
        rate = 0.0
        if len(image_times) > 1 and image_times[-1] > image_times[0]:
            rate = (len(image_times) - 1) / (image_times[-1] - image_times[0])
        state_errors = [
            float(row["state_image_error_s"]) for row in self._rows
        ]
        action_errors = [
            float(row["expert_action_image_error_s"]) for row in self._rows
        ]
        path_length = sum(
            math.hypot(
                float(right["position_north_m"])
                - float(left["position_north_m"]),
                float(right["position_east_m"])
                - float(left["position_east_m"]),
            )
            for left, right in zip(self._rows, self._rows[1:])
        )

        def percentile95(values: list[float]) -> float | None:
            if not values:
                return None
            ordered = sorted(values)
            return ordered[math.ceil(0.95 * len(ordered)) - 1]

        episode = {
            "dataset_version": DATASET_VERSION,
            "episode_id": self.episode_id,
            "random_seed": self.random_seed,
            "scene_configuration": self._scene_configuration,
            "started_utc": self.started_utc,
            "completed_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "status": "complete" if success else "failed",
            "success": success,
            "failure": failure,
            "sample_count": len(self._rows),
            "rejected_sample_count": sum(self._rejections.values()),
            "rejections_by_reason": dict(sorted(self._rejections.items())),
            "observed_sampling_rate_hz": rate,
            "path_length_m": path_length,
            "astar_path_information": {
                "validated_path": self._planner_path,
                "planner_status": self._planner_status,
            },
            "final_tracking_goal_distance_m": (
                None if not self._rows
                else float(self._rows[-1]["raw_goal_distance_m"])
            ),
            "synchronization_statistics_s": {
                "state": {
                    "mean": (
                        None if not state_errors
                        else sum(state_errors) / len(state_errors)
                    ),
                    "p95": percentile95(state_errors),
                    "max": max(state_errors, default=None),
                },
                "expert_action": {
                    "mean": (
                        None if not action_errors
                        else sum(action_errors) / len(action_errors)
                    ),
                    "p95": percentile95(action_errors),
                    "max": max(action_errors, default=None),
                },
            },
            "available_sensor_streams": {
                "fpv_rgb": {
                    "required": True,
                    "accepted": len(self._rows),
                    "received": self._fpv_rgb_received,
                    "observed_rate_hz": self._stream_rate(
                        "fpv_rgb", self._fpv_rgb_received
                    ),
                },
                "observer_rgb": {
                    "required": False,
                    "geometry": (
                        "canonical legacy Episode Manager TOP observer"
                    ),
                    "matched": sum(
                        row["observer_rgb_available"]
                        for row in self._auxiliary_rows
                    ),
                    "received": self._observer_rgb_received,
                    "observed_rate_hz": self._stream_rate(
                        "observer_rgb", self._observer_rgb_received
                    ),
                },
                "fpv_depth": {
                    "required": False,
                    "encoding": (
                        "PNG uint16 millimetres, clip [50,30000], invalid 0"
                    ),
                    "matched": sum(
                        row["fpv_depth_available"]
                        for row in self._auxiliary_rows
                    ),
                    "received": self._fpv_depth_received,
                    "observed_rate_hz": self._stream_rate(
                        "fpv_depth", self._fpv_depth_received
                    ),
                },
                "runtime_status": self._sensor_runtime_status,
            },
            "persistent_runtime_boundary": {
                "guard_enabled": self._boundary_guard_enabled,
                "ready": self._episode_boundary_ready,
                "runtime_generation": self.expected_runtime_generation,
                "scene_revision": self.expected_scene_revision,
                "minimum_camera_frame_counts": self._minimum_camera_counts,
                "boundary_timestamp_s": self._episode_boundary_timestamp_s,
                "discarded_pre_boundary_frames": dict(
                    sorted(self._boundary_discarded.items())
                ),
            },
            "maximum_state_image_error_s": max(
                (float(row["state_image_error_s"]) for row in self._rows),
                default=None,
            ),
            "maximum_action_image_error_s": max(
                (
                    float(row["expert_action_image_error_s"])
                    for row in self._rows
                ),
                default=None,
            ),
            "timeline": self._timeline,
            "accumulated_flight_evidence": {
                "goal_reached": self._observed_goal_reached,
                "landing_commanded": self._observed_landing_commanded,
                "landed_after_landing_command": (
                    self._observed_landed_after_landing
                ),
                "terminal_complete": bool(
                    status is not None and status.state == "COMPLETE"
                ),
            },
            "terminal_flight_status": (
                None if status is None else {
                    "state": status.state,
                    "goal_reached": bool(status.goal_reached),
                    "landing_commanded": bool(status.landing_commanded),
                    "landed": bool(status.landed),
                    "offboard_active": bool(status.offboard_active),
                    "vehicle_armed": bool(status.vehicle_armed),
                    "failure_reason": status.failure_reason,
                }
            ),
        }
        _atomic_json(self.episode_dir / "episode.json", episode)
        _atomic_json(self.episode_dir / "progress.json", {
            "episode_id": self.episode_id,
            "state": "DATASET_VALIDATION_PENDING",
            "goal_distance_m": episode["final_tracking_goal_distance_m"],
            "accepted_samples": len(self._rows),
            "rejected_samples": sum(self._rejections.values()),
            "rejections_by_reason": dict(sorted(self._rejections.items())),
            "fpv_rgb_received": self._fpv_rgb_received,
            "observer_rgb_received": self._observer_rgb_received,
            "fpv_depth_received": self._fpv_depth_received,
        })
        episode["episode_disk_usage_bytes"] = sum(
            item.stat().st_size
            for item in self.episode_dir.rglob("*")
            if item.is_file()
        )
        _atomic_json(self.episode_dir / "episode.json", episode)
        manifest_path = self.dataset_root / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        episodes = list(manifest.get("episodes", []))
        if self.episode_id not in episodes:
            episodes.append(self.episode_id)
        manifest.update({
            "episodes": episodes,
            "episode_count": len(episodes),
            "sample_count": int(manifest.get("sample_count", 0))
            + len(self._rows),
            "status": (
                "complete" if self.collection_mode == "single"
                else "collecting"
            ),
        })
        if self.collection_mode == "single":
            manifest["all_success"] = success
        _atomic_json(self.dataset_root / "dataset_manifest.json", manifest)
        self.get_logger().info(
            f"EXPERT_DATASET_FINALIZED episode={self.episode_id} "
            f"success={str(success).lower()} "
            f"samples={len(self._rows)} "
            f"rejected={sum(self._rejections.values())}"
        )

    def destroy_node(self):
        """Finalize the dataset before releasing ROS resources."""
        self.finalize()
        return super().destroy_node()


def main(args=None) -> int:
    """Run the finite episode recorder until the flight launch shuts down."""
    rclpy.init(args=args)
    node = ExpertDatasetRecorderNode()
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
