"""Finite acceptance monitor for the live PX4 SITL flight milestone."""

import importlib
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from uav_interfaces.msg import Px4FlightStatus
from uav_interfaces.srv import SetPx4FlightEnable

from uav_px4_control.control_mux_node import control_qos
from uav_px4_control.px4_setpoint_streamer_node import (
    VEHICLE_STATUS_TOPIC,
    px4_output_qos,
)
from uav_px4_control.px4_sitl_flight_supervisor_node import (
    FLIGHT_STATUS_TOPIC,
    SET_FLIGHT_ENABLE_SERVICE,
    VEHICLE_LAND_DETECTED_TOPIC,
)


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
            message_module.VehicleLandDetected,
            VEHICLE_LAND_DETECTED_TOPIC,
            self._land_callback,
            px4_output_qos(),
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

    def _land_callback(self, message) -> None:
        self._land_detected = message
        if message.landed:
            self._record_once("px4_landed_confirmed")

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
            "landed_status": message.landed,
        }
        for name, value in evidence.items():
            if value:
                self._record_once(name)

    def _record_once(self, name: str) -> None:
        if name in self._first_evidence:
            return
        status = self._status
        self._first_evidence[name] = {
            "elapsed_s": self._elapsed(),
            "state": "" if status is None else status.state,
        }

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
