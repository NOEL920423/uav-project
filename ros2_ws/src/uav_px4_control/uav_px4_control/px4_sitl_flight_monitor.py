"""Finite acceptance monitor for the live PX4 SITL flight milestone."""

import importlib
import json
import math
import time
from collections import deque
from pathlib import Path

from geometry_msgs.msg import PoseStamped

import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from uav_interfaces.msg import Px4FlightStatus, Px4StreamStatus
from uav_interfaces.srv import SetPx4FlightEnable

from uav_navigation.trajectory_follower_node import durable_qos

from uav_px4_control.control_mux_node import control_qos
from uav_px4_control.px4_setpoint_streamer_node import (
    STREAM_STATUS_TOPIC,
    VEHICLE_STATUS_TOPIC,
    px4_output_qos,
)
from uav_px4_control.px4_sitl_flight_supervisor_node import (
    EXTERNAL_SCENE_GOAL_TOPIC,
    FLIGHT_STATUS_TOPIC,
    ISAAC_BRIDGE_STATUS_TOPIC,
    SET_FLIGHT_ENABLE_SERVICE,
    VEHICLE_LAND_DETECTED_TOPIC,
)


ISAAC_POSE_TOPIC = "/isaac_uav/pose"


class Px4SitlFlightMonitor(Node):
    """Require state-backed evidence for every milestone acceptance item."""

    def __init__(self) -> None:
        """Create the explicit-start client and independent evidence inputs."""
        super().__init__("px4_sitl_flight_monitor")
        self.declare_parameter(
            "evidence_path", "/tmp/uav_px4_sitl_flight_evidence.json"
        )
        self.declare_parameter("timeout_s", 120.0)
        self.declare_parameter("start_delay_s", 2.0)
        self.declare_parameter("minimum_stream_rate_hz", 19.0)
        self.declare_parameter("require_isaac_evidence", False)
        self.declare_parameter("minimum_isaac_pose_rate_hz", 15.0)
        self.declare_parameter("isaac_goal_tolerance_m", 0.40)
        self.declare_parameter("isaac_landing_height_tolerance_m", 0.25)
        self.evidence_path = Path(
            str(self.get_parameter("evidence_path").value)
        )
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.start_delay_s = float(
            self.get_parameter("start_delay_s").value
        )
        self.minimum_stream_rate_hz = float(
            self.get_parameter("minimum_stream_rate_hz").value
        )
        self.require_isaac_evidence = bool(
            self.get_parameter("require_isaac_evidence").value
        )
        self.minimum_isaac_pose_rate_hz = float(
            self.get_parameter("minimum_isaac_pose_rate_hz").value
        )
        self.isaac_goal_tolerance_m = float(
            self.get_parameter("isaac_goal_tolerance_m").value
        )
        self.isaac_landing_height_tolerance_m = float(
            self.get_parameter("isaac_landing_height_tolerance_m").value
        )
        message_module = importlib.import_module("px4_msgs.msg")
        self._VehicleStatus = message_module.VehicleStatus
        self._status: Px4FlightStatus | None = None
        self._vehicle_status = None
        self._land_detected = None
        self._started = time.monotonic()
        self._start_requested = False
        self._start_future = None
        self._last_state = ""
        self._exit_code = 1
        self._finished = False
        self._timeline: list[dict] = []
        self._first_evidence: dict[str, dict] = {}
        self._max_altitude_m = 0.0
        self._minimum_goal_distance_m = math.inf
        self._maximum_stream_rate_hz = 0.0
        self._isaac_pose: tuple[float, float, float] | None = None
        self._isaac_initial_pose: tuple[float, float, float] | None = None
        self._isaac_tracking_start: tuple[float, float, float] | None = None
        self._isaac_goal: tuple[float, float, float] | None = None
        self._isaac_pose_times = deque(maxlen=200)
        self._isaac_pose_count = 0
        self._isaac_max_altitude_m = 0.0
        self._isaac_max_tracking_displacement_m = 0.0
        self._isaac_minimum_goal_distance_m = math.inf
        self._isaac_maximum_pose_rate_hz = 0.0
        self._isaac_bridge_status: dict = {}
        self._stream_status: Px4StreamStatus | None = None
        self._stream_faults: list[dict] = []
        self.create_subscription(
            Px4FlightStatus,
            FLIGHT_STATUS_TOPIC,
            self._status_callback,
            control_qos(),
        )
        self.create_subscription(
            self._VehicleStatus,
            VEHICLE_STATUS_TOPIC,
            self._vehicle_status_callback,
            px4_output_qos(),
        )
        self.create_subscription(
            Px4StreamStatus,
            STREAM_STATUS_TOPIC,
            self._stream_status_callback,
            control_qos(),
        )
        self.create_subscription(
            message_module.VehicleLandDetected,
            VEHICLE_LAND_DETECTED_TOPIC,
            self._land_callback,
            px4_output_qos(),
        )
        self.create_subscription(
            PoseStamped,
            ISAAC_POSE_TOPIC,
            self._isaac_pose_callback,
            control_qos(),
        )
        self.create_subscription(
            PoseStamped,
            EXTERNAL_SCENE_GOAL_TOPIC,
            self._isaac_goal_callback,
            durable_qos(),
        )
        self.create_subscription(
            String,
            ISAAC_BRIDGE_STATUS_TOPIC,
            self._isaac_bridge_status_callback,
            control_qos(),
        )
        self._client = self.create_client(
            SetPx4FlightEnable, SET_FLIGHT_ENABLE_SERVICE
        )
        self.create_timer(0.05, self._tick)

    def _elapsed(self) -> float:
        return time.monotonic() - self._started

    def _vehicle_status_callback(self, message) -> None:
        self._vehicle_status = message
        if message.nav_state == message.NAVIGATION_STATE_OFFBOARD:
            self._record_once("px4_offboard_confirmed")
        if message.arming_state == message.ARMING_STATE_ARMED:
            self._record_once("px4_armed_confirmed")

    def _stream_status_callback(self, message: Px4StreamStatus) -> None:
        self._stream_status = message
        if message.state.startswith(("STOPPED_", "LATCHED_")):
            fault = {
                "elapsed_s": self._elapsed(),
                "state": message.state,
                "reason": message.stop_reason,
                "observed_rate_hz": message.observed_rate_hz,
                "maximum_publish_gap_s": message.maximum_publish_gap,
                "candidate_age_s": message.candidate_age,
                "gate_status_age_s": message.gate_status_age,
                "telemetry_age_s": message.telemetry_age,
            }
            if not self._stream_faults or fault["state"] != (
                self._stream_faults[-1]["state"]
            ):
                self._stream_faults.append(fault)

    def _land_callback(self, message) -> None:
        self._land_detected = message
        if (
            message.landed
            and "landing_commanded" in self._first_evidence
        ):
            self._record_once("px4_landed_confirmed")
            self._maybe_record_isaac_landing()

    @staticmethod
    def _finite_pose(
        message: PoseStamped,
    ) -> tuple[float, float, float] | None:
        position = message.pose.position
        values = (float(position.x), float(position.y), float(position.z))
        if message.header.frame_id != "isaac_world":
            return None
        valid = all(math.isfinite(value) for value in values)
        return values if valid else None

    def _isaac_pose_callback(self, message: PoseStamped) -> None:
        pose = self._finite_pose(message)
        if pose is None:
            return
        now = time.monotonic()
        self._isaac_pose = pose
        self._isaac_pose_count += 1
        self._isaac_pose_times.append(now)
        if self._isaac_initial_pose is None:
            self._isaac_initial_pose = pose
        initial = self._isaac_initial_pose
        altitude = pose[2] - initial[2]
        self._isaac_max_altitude_m = max(
            self._isaac_max_altitude_m, altitude
        )
        self._update_isaac_pose_rate()
        if (
            self._start_requested
            and altitude >= 1.25
        ):
            self._record_once("isaac_takeoff_confirmed")
        if self._status is not None and self._status.state in {
            "STARTING_TRACKING",
            "TRACKING",
            "GOAL_HOLD",
        }:
            if self._isaac_tracking_start is None:
                self._isaac_tracking_start = pose
            displacement = math.dist(
                pose[:2], self._isaac_tracking_start[:2]
            )
            self._isaac_max_tracking_displacement_m = max(
                self._isaac_max_tracking_displacement_m,
                displacement,
            )
            if displacement >= 0.50:
                self._record_once("isaac_trajectory_motion_confirmed")
            self._update_isaac_goal_distance()
        self._maybe_record_isaac_landing()

    def _update_isaac_pose_rate(self) -> None:
        if len(self._isaac_pose_times) < 10:
            return
        span = self._isaac_pose_times[-1] - self._isaac_pose_times[0]
        if span <= 0.0:
            return
        rate = (len(self._isaac_pose_times) - 1) / span
        self._isaac_maximum_pose_rate_hz = max(
            self._isaac_maximum_pose_rate_hz, rate
        )
        if rate >= self.minimum_isaac_pose_rate_hz:
            self._record_once("isaac_pose_stream_confirmed")

    def _isaac_goal_callback(self, message: PoseStamped) -> None:
        goal = self._finite_pose(message)
        if goal is not None:
            self._isaac_goal = goal

    def _update_isaac_goal_distance(self) -> None:
        if self._isaac_pose is None or self._isaac_goal is None:
            return
        distance = math.dist(self._isaac_pose, self._isaac_goal)
        self._isaac_minimum_goal_distance_m = min(
            self._isaac_minimum_goal_distance_m, distance
        )
        if distance <= self.isaac_goal_tolerance_m:
            self._record_once("isaac_goal_confirmed")

    def _isaac_bridge_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(status, dict):
            return
        self._isaac_bridge_status = status
        if status.get("ready") is True:
            self._record_once("isaac_runtime_ready")

    def _maybe_record_isaac_landing(self) -> None:
        if (
            self._isaac_pose is None
            or self._isaac_initial_pose is None
            or self._land_detected is None
            or not self._land_detected.landed
            or "isaac_takeoff_confirmed" not in self._first_evidence
        ):
            return
        height_error = abs(
            self._isaac_pose[2] - self._isaac_initial_pose[2]
        )
        if height_error <= self.isaac_landing_height_tolerance_m:
            self._record_once("isaac_landing_confirmed")

    def _status_callback(self, message: Px4FlightStatus) -> None:
        self._status = message
        self._max_altitude_m = max(
            self._max_altitude_m, float(message.altitude_m)
        )
        if math.isfinite(float(message.goal_distance_m)):
            self._minimum_goal_distance_m = min(
                self._minimum_goal_distance_m,
                float(message.goal_distance_m),
            )
        self._maximum_stream_rate_hz = max(
            self._maximum_stream_rate_hz,
            float(message.stream_rate_hz),
        )
        if message.state != self._last_state:
            self._timeline.append({
                "elapsed_s": self._elapsed(),
                "state": message.state,
                "altitude_m": float(message.altitude_m),
                "north_m": float(message.position_north_m),
                "east_m": float(message.position_east_m),
                "down_m": float(message.position_down_m),
                "goal_distance_m": float(message.goal_distance_m),
                "stream_rate_hz": float(message.stream_rate_hz),
                "last_command_ack": message.last_command_ack,
            })
            self.get_logger().info(
                f"ACCEPTANCE_TIMELINE state={message.state} "
                f"t={self._elapsed():.3f}s altitude={message.altitude_m:.3f}"
            )
            self._last_state = message.state
        evidence = {
            "astar_path_valid": message.planner_path_valid,
            "bspline_valid": message.bspline_valid,
            "trajectory_valid": message.trajectory_valid,
            "follower_command_valid": message.follower_command_valid,
            "astar_expert_selected": message.astar_selected,
            "output_gate_safe": message.output_gate_safe,
            "stream_20hz": (
                message.stream_stable
                and message.stream_rate_hz >= self.minimum_stream_rate_hz
            ),
            "offboard_status": message.offboard_active,
            "armed_status": message.vehicle_armed,
            "takeoff_altitude": message.takeoff_altitude_reached,
            "trajectory_tracking": message.tracking_active,
            "goal_reached": message.goal_reached,
            "landing_commanded": message.landing_commanded,
            "landed_status": (
                message.landed and message.landing_commanded
            ),
        }
        for name, value in evidence.items():
            if value:
                self._record_once(name)
        if (
            message.landing_commanded
            and self._land_detected is not None
            and self._land_detected.landed
        ):
            self._record_once("px4_landed_confirmed")

    def _record_once(self, name: str) -> None:
        if name in self._first_evidence:
            return
        status = self._status
        self._first_evidence[name] = {
            "elapsed_s": self._elapsed(),
            "state": "" if status is None else status.state,
        }
        if self._isaac_pose is not None:
            self._first_evidence[name]["isaac_pose"] = list(
                self._isaac_pose
            )

    def _tick(self) -> None:
        if self._finished:
            return
        elapsed = self._elapsed()
        if (
            not self._start_requested
            and elapsed >= self.start_delay_s
            and self._client.service_is_ready()
        ):
            request = SetPx4FlightEnable.Request()
            request.enable = True
            self._start_future = self._client.call_async(request)
            self._start_requested = True
            self.get_logger().warning(
                "Explicitly requesting the authorized PX4 SITL flight"
            )
        if self._start_future is not None and self._start_future.done():
            try:
                response = self._start_future.result()
            except Exception as error:  # noqa: BLE001 - ROS future boundary
                self._finish(1, f"flight enable service failed: {error}")
                return
            self._start_future = None
            if not response.accepted:
                self._finish(
                    1,
                    "flight enable rejected: " + response.status_message,
                )
                return
            self._record_once("explicit_flight_enable")
        if self._status is not None and self._status.state == "COMPLETE":
            missing = self._missing_acceptance()
            if missing:
                self._finish(
                    1,
                    "mission completed with missing evidence: "
                    + ",".join(missing),
                )
            else:
                self._finish(
                    0,
                    "all PX4 SITL flight acceptance evidence observed",
                )
            return
        if self._status is not None and self._status.state == "FAILED":
            self._finish(
                1,
                "flight supervisor failed: " + self._status.failure_reason,
            )
            return
        if elapsed > self.timeout_s:
            self._finish(1, "flight acceptance monitor timed out")

    def _missing_acceptance(self) -> list[str]:
        required = {
            "explicit_flight_enable",
            "astar_path_valid",
            "bspline_valid",
            "trajectory_valid",
            "follower_command_valid",
            "astar_expert_selected",
            "output_gate_safe",
            "stream_20hz",
            "px4_offboard_confirmed",
            "px4_armed_confirmed",
            "takeoff_altitude",
            "trajectory_tracking",
            "goal_reached",
            "landing_commanded",
            "px4_landed_confirmed",
        }
        if self.require_isaac_evidence:
            required.update({
                "isaac_runtime_ready",
                "isaac_pose_stream_confirmed",
                "isaac_takeoff_confirmed",
                "isaac_trajectory_motion_confirmed",
                "isaac_goal_confirmed",
                "isaac_landing_confirmed",
            })
        status = self._vehicle_status
        if status is None or status.arming_state == status.ARMING_STATE_ARMED:
            required.add("final_disarmed")
        else:
            self._record_once("final_disarmed")
        return sorted(required - self._first_evidence.keys())

    def _finish(self, code: int, detail: str) -> None:
        if self._finished:
            return
        self._finished = True
        self._exit_code = code
        payload = {
            "success": code == 0,
            "detail": detail,
            "elapsed_s": self._elapsed(),
            "timeline": self._timeline,
            "first_evidence": self._first_evidence,
            "max_altitude_m": self._max_altitude_m,
            "minimum_goal_distance_m": self._minimum_goal_distance_m,
            "maximum_stream_rate_hz": self._maximum_stream_rate_hz,
            "stream_faults": self._stream_faults,
            "final_stream_status": self._stream_payload(),
            "isaac": self._isaac_payload(),
            "final_flight_status": self._status_payload(),
            "final_px4": self._px4_payload(),
        }
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger = (
            self.get_logger().info
            if code == 0
            else self.get_logger().error
        )
        logger(
            f"PX4_SITL_FLIGHT_RESULT success={str(code == 0).lower()} "
            f"detail={detail} evidence={self.evidence_path}"
        )
        rclpy.shutdown()

    def _status_payload(self) -> dict:
        if self._status is None:
            return {}
        return {
            "state": self._status.state,
            "failure_reason": self._status.failure_reason,
            "altitude_m": self._status.altitude_m,
            "goal_distance_m": self._status.goal_distance_m,
            "stream_rate_hz": self._status.stream_rate_hz,
            "vehicle_command_count": self._status.vehicle_command_count,
            "last_command_ack": self._status.last_command_ack,
        }

    def _px4_payload(self) -> dict:
        if self._vehicle_status is None:
            return {}
        return {
            "arming_state": int(self._vehicle_status.arming_state),
            "nav_state": int(self._vehicle_status.nav_state),
            "failsafe": bool(self._vehicle_status.failsafe),
            "landed": bool(
                self._land_detected is not None
                and self._land_detected.landed
            ),
        }

    def _isaac_payload(self) -> dict:
        return {
            "required": self.require_isaac_evidence,
            "pose_count": self._isaac_pose_count,
            "initial_pose": self._isaac_initial_pose,
            "final_pose": self._isaac_pose,
            "goal": self._isaac_goal,
            "max_altitude_m": self._isaac_max_altitude_m,
            "max_tracking_displacement_m": (
                self._isaac_max_tracking_displacement_m
            ),
            "minimum_goal_distance_m": (
                self._isaac_minimum_goal_distance_m
            ),
            "maximum_pose_rate_hz": self._isaac_maximum_pose_rate_hz,
            "final_bridge_status": self._isaac_bridge_status,
        }

    def _stream_payload(self) -> dict:
        if self._stream_status is None:
            return {}
        return {
            "state": self._stream_status.state,
            "stop_reason": self._stream_status.stop_reason,
            "streaming": self._stream_status.streaming,
            "observed_rate_hz": self._stream_status.observed_rate_hz,
            "maximum_publish_gap_s": (
                self._stream_status.maximum_publish_gap
            ),
            "dropped_cycle_count": self._stream_status.dropped_cycle_count,
        }


def main(args=None) -> int:
    """Run the finite live acceptance monitor."""
    rclpy.init(args=args)
    node = Px4SitlFlightMonitor()
    try:
        rclpy.spin(node)
    finally:
        exit_code = node._exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
