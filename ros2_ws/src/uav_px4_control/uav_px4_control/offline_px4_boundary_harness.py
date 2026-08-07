"""Finite synthetic ROS fixtures for the Phase 7 PX4 boundary."""

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import Bool

from uav_interfaces.msg import (
    ControlMuxStatus,
    Px4OutputGateStatus,
    Px4SetpointCandidate,
    Px4SyntheticTelemetry,
    TrajectoryTrackingStatus,
)
from uav_interfaces.srv import SetControlSource, SetPx4OutputEnable

from uav_px4_control.control_mux_node import (
    MUX_STATUS_TOPIC,
    SELECTED_COMMAND_TOPIC,
    SET_SOURCE_SERVICE,
    control_qos,
)
from uav_px4_control.control_source_models import ASTAR_EXPERT
from uav_px4_control.px4_mapping_gate_node import (
    CANDIDATE_TOPIC,
    GATE_STATUS_TOPIC,
    SAFE_TO_FORWARD_TOPIC,
    SET_OUTPUT_ENABLE_SERVICE,
    SYNTHETIC_TELEMETRY_TOPIC,
)


TRACKING_STATUS_TOPIC = "/uav/control/astar_tracking_status"


class SyntheticPx4TelemetryPublisher(Node):
    """Publish architecture-level synthetic state without starting PX4."""

    def __init__(self) -> None:
        """Create a deterministic healthy or fault-injection fixture."""
        super().__init__("synthetic_px4_telemetry")
        self.declare_parameter("behavior", "healthy")
        self.behavior = str(self.get_parameter("behavior").value)
        self._publisher = self.create_publisher(
            Px4SyntheticTelemetry,
            SYNTHETIC_TELEMETRY_TOPIC,
            control_qos(),
        )
        self._started = time.monotonic()
        self.create_timer(0.04, self._tick)

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._started
        fault = False
        if self.behavior == "gate-fault":
            fault = 2.5 <= elapsed < 3.3
        elif self.behavior == "boundary-fault":
            fault = 6.0 <= elapsed < 6.8
        message = Px4SyntheticTelemetry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "px4_ned"
        message.timestamp_us = self.get_clock().now().nanoseconds // 1_000
        message.connected = True
        message.arming_state = 1
        message.navigation_state = 0
        message.offboard_control_signal_lost = False
        message.offboard_active = False
        message.failsafe = fault
        message.pre_flight_checks_pass = True
        message.local_position_valid = True
        message.local_velocity_valid = True
        message.odometry_valid = True
        message.pose_frame_ned = True
        message.velocity_frame_ned = True
        message.fixture = self.behavior
        self._publisher.publish(message)


class Px4BoundaryResultMonitor(Node):
    """Drive services and independently verify the finite ROS boundary."""

    def __init__(self) -> None:
        """Subscribe to all diagnostics and initialize the scenario."""
        super().__init__("px4_boundary_result_monitor")
        self.declare_parameter("mode", "mapping")
        self.mode = str(self.get_parameter("mode").value)
        qos = control_qos()
        self.create_subscription(
            ControlMuxStatus, MUX_STATUS_TOPIC, self._mux_callback, qos
        )
        self.create_subscription(
            Px4SetpointCandidate,
            CANDIDATE_TOPIC,
            self._candidate_callback,
            qos,
        )
        self.create_subscription(
            Px4OutputGateStatus,
            GATE_STATUS_TOPIC,
            self._status_callback,
            qos,
        )
        self.create_subscription(
            Bool,
            SAFE_TO_FORWARD_TOPIC,
            self._safe_callback,
            qos,
        )
        if self.mode == "boundary":
            self.create_subscription(
                TrajectoryTrackingStatus,
                TRACKING_STATUS_TOPIC,
                self._tracking_callback,
                qos,
            )
        self._source_client = self.create_client(
            SetControlSource, SET_SOURCE_SERVICE
        )
        self._enable_client = self.create_client(
            SetPx4OutputEnable, SET_OUTPUT_ENABLE_SERVICE
        )
        self._started = time.monotonic()
        self._stage = "WAIT_ASTAR"
        self._mux = None
        self._candidate = None
        self._status = None
        self._safe = False
        self._safe_seen = False
        self._fault_seen = False
        self._tracking_seen = False
        self._last_candidate_timestamp_us = None
        self._pending = None
        self._pending_kind = ""
        self._error = ""
        self._finished = False
        self.exit_code = 1
        self.create_timer(0.05, self._tick)

    def _mux_callback(self, message: ControlMuxStatus) -> None:
        self._mux = message

    def _candidate_callback(self, message: Px4SetpointCandidate) -> None:
        self._candidate = message
        values = (
            message.velocity_ned.x,
            message.velocity_ned.y,
            message.velocity_ned.z,
            message.yaw_rate_ned,
        )
        if message.header.frame_id != "px4_ned":
            self._error = "candidate frame is not px4_ned"
        elif not all(math.isfinite(value) for value in values):
            self._error = "candidate contains non-finite used fields"
        elif self._last_candidate_timestamp_us is not None and (
            message.timestamp_us < self._last_candidate_timestamp_us
        ):
            self._error = "candidate diagnostic timestamp moved backward"
        self._last_candidate_timestamp_us = message.timestamp_us

    def _status_callback(self, message: Px4OutputGateStatus) -> None:
        self._status = message
        if message.safe_to_forward:
            self._safe_seen = True
        if self._safe_seen and message.state in {
            "DISABLED_FAILSAFE",
            "LATCHED_FAULT",
        }:
            self._fault_seen = True

    def _safe_callback(self, message: Bool) -> None:
        self._safe = message.data

    def _tracking_callback(self, message: TrajectoryTrackingStatus) -> None:
        self._tracking_seen |= message.state in {
            "PRESTART",
            "TRACKING",
            "GOAL_SETTLING",
            "GOAL_HOLD",
        }

    def _request_source(self) -> None:
        if (
            self._pending is not None
            or not self._source_client.service_is_ready()
        ):
            return
        request = SetControlSource.Request()
        request.source = ASTAR_EXPERT
        self._pending = self._source_client.call_async(request)
        self._pending_kind = "source"

    def _request_enable(self, enable: bool) -> None:
        if (
            self._pending is not None
            or not self._enable_client.service_is_ready()
        ):
            return
        request = SetPx4OutputEnable.Request()
        request.enable = enable
        self._pending = self._enable_client.call_async(request)
        self._pending_kind = "enable" if enable else "disable"

    def _response_ready(self, kind: str) -> bool:
        if (
            self._pending is None
            or self._pending_kind != kind
            or not self._pending.done()
        ):
            return False
        response = self._pending.result()
        self._pending = None
        self._pending_kind = ""
        if response is None or not response.accepted:
            detail = (
                "no response"
                if response is None
                else response.status_message
            )
            self._error = f"{kind} service failed: {detail}"
            return False
        return True

    def _active_candidate_ready(self) -> bool:
        return (
            self._mux is not None
            and self._mux.active_source == ASTAR_EXPERT
            and self._candidate is not None
            and self._candidate.valid
            and self._status is not None
            and self._status.telemetry_valid
            and (self.mode != "boundary" or self._tracking_seen)
        )

    def _mapping_contract_valid(self) -> bool:
        candidate = self._candidate
        if candidate is None:
            return False
        return (
            abs(candidate.velocity_ned.x - 0.40) <= 1e-6
            and abs(candidate.velocity_ned.y) <= 1e-6
            and abs(candidate.velocity_ned.z) <= 1e-6
            and abs(candidate.yaw_rate_ned - 0.10) <= 1e-6
            and candidate.source == ASTAR_EXPERT
            and not self._safe
            and self._status is not None
            and self._status.state == "READY_DISABLED"
        )

    def _tick(self) -> None:
        if self._finished:
            return
        if self._error:
            self._finish(1, self._error)
            return
        if self._stage == "WAIT_ASTAR":
            if (
                self._mux is not None
                and ASTAR_EXPERT in self._mux.healthy_sources
            ):
                self._request_source()
                if self._pending is not None:
                    self._stage = "REQUEST_SOURCE"
        elif (
            self._stage == "REQUEST_SOURCE"
            and self._response_ready("source")
        ):
            self._stage = "WAIT_ACTIVE"
        elif self._stage == "WAIT_ACTIVE" and self._active_candidate_ready():
            if self.mode == "mapping":
                if self._mapping_contract_valid():
                    self._finish(0, "NED mapping and disabled gate validated")
            else:
                self._request_enable(True)
                if self._pending is not None:
                    self._stage = "REQUEST_ENABLE"
        elif (
            self._stage == "REQUEST_ENABLE"
            and self._response_ready("enable")
        ):
            self._stage = "WAIT_SAFE"
        elif self._stage == "WAIT_SAFE" and self._safe:
            self._safe_seen = True
            self._stage = "WAIT_FAULT"
        elif (
            self._stage == "WAIT_FAULT"
            and self._fault_seen
            and not self._safe
        ):
            self._stage = "WAIT_LATCHED_RECOVERY"
        elif self._stage == "WAIT_LATCHED_RECOVERY":
            if (
                self._status is not None
                and self._status.state == "LATCHED_FAULT"
                and self._status.telemetry_valid
                and not self._status.failsafe
            ):
                self._request_enable(False)
                if self._pending is not None:
                    self._stage = "REQUEST_DISABLE"
        elif (
            self._stage == "REQUEST_DISABLE"
            and self._response_ready("disable")
        ):
            self._request_enable(True)
            if self._pending is not None:
                self._stage = "REQUEST_REENABLE"
        elif (
            self._stage == "REQUEST_REENABLE"
            and self._response_ready("enable")
        ):
            self._stage = "WAIT_RECOVERED_SAFE"
        elif self._stage == "WAIT_RECOVERED_SAFE" and self._safe:
            self._finish(
                0,
                "safe, fault, latch, and explicit recovery validated",
            )
        timeout = 14.0 if self.mode == "boundary" else 9.0
        if time.monotonic() - self._started > timeout:
            self._finish(1, f"{self.mode} boundary timed out at {self._stage}")

    def _finish(self, code: int, detail: str) -> None:
        if self._finished:
            return
        topics = dict(self.get_topic_names_and_types())
        if any(name.startswith("/fmu/in/") for name in topics):
            code, detail = 1, "forbidden live PX4 input topic detected"
        owners = self.get_publishers_info_by_topic(SELECTED_COMMAND_TOPIC)
        if len(owners) != 1:
            code, detail = 1, "selected_command does not have one owner"
        self._finished = True
        self.exit_code = code
        markers = {
            "mapping": "px4 mapping offline integration passed:",
            "gate": "px4 gate offline integration passed:",
            "boundary": "px4 boundary offline integration passed:",
        }
        if code == 0:
            summary = (
                f"{markers[self.mode]} mode={self.mode}, detail={detail}, "
                f"safe_seen={self._safe_seen}, "
                f"fault_seen={self._fault_seen}, frame=px4_ned"
            )
            self.get_logger().info(summary)
        else:
            summary = (
                f"px4 boundary integration failed: mode={self.mode}, "
                f"detail={detail}, stage={self._stage}"
            )
            self.get_logger().error(summary)
        if rclpy.ok():
            rclpy.shutdown()


def _spin(node) -> int:
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        code = getattr(node, "exit_code", 0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


def telemetry_main(args=None) -> int:
    """Run the deterministic synthetic telemetry publisher."""
    rclpy.init(args=args)
    return _spin(SyntheticPx4TelemetryPublisher())


def monitor_main(args=None) -> int:
    """Run the independent finite PX4 boundary monitor."""
    rclpy.init(args=args)
    return _spin(Px4BoundaryResultMonitor())
