"""Drive one generic takeoff, BC handoff, HOLD, and PX4 landing."""

from __future__ import annotations

import importlib
import json
import math

from geometry_msgs.msg import TwistStamped

from nav_msgs.msg import Odometry

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import String

from std_srvs.srv import SetBool

from uav_interfaces.msg import (
    ControlMuxStatus,
    Px4OutputGateStatus,
    Px4StreamStatus,
)
from uav_interfaces.srv import (
    SendPx4VehicleCommand,
    SetControlSource,
    SetPx4OutputEnable,
    SetPx4StreamEnable,
)

from uav_px4_control.bc_flight_models import (
    BcFlightConfig,
    BcFlightController,
    BcFlightEvidence,
    BcFlightState,
)
from uav_px4_control.bc_policy_node import (
    POLICY_STATUS_SCHEMA,
    POLICY_STATUS_TOPIC,
    SET_POLICY_ENABLE_SERVICE,
)
from uav_px4_control.control_mux_node import (
    MUX_STATUS_TOPIC,
    SET_SOURCE_SERVICE,
    control_qos,
)
from uav_px4_control.control_source_models import (
    BC_POLICY,
    FLIGHT_LIFECYCLE,
    HOLD,
    SOURCE_TOPICS,
    VALID_COMMAND_FRAME,
)
from uav_px4_control.px4_mapping_gate_node import (
    GATE_STATUS_TOPIC,
    SET_OUTPUT_ENABLE_SERVICE,
)
from uav_px4_control.px4_setpoint_streamer_node import (
    SET_STREAM_ENABLE_SERVICE,
    STREAM_STATUS_TOPIC,
    VEHICLE_STATUS_TOPIC,
    px4_output_qos,
)
from uav_px4_control.px4_vehicle_command_owner_node import (
    SEND_VEHICLE_COMMAND_SERVICE,
)


SUPERVISOR_STATUS_TOPIC = "/uav/bc/flight_status"
ODOMETRY_TOPIC = "/uav/vehicle/odometry"
EPISODE_TERMINATION_TOPIC = "/uav/bc/episode_termination"
VEHICLE_LAND_DETECTED_TOPIC = "/fmu/out/vehicle_land_detected"
ISAAC_BRIDGE_STATUS_TOPIC = "/uav/isaac/bridge_status"
SUPERVISOR_STATUS_SCHEMA = "uav_bc_flight_status/v1"
TERMINATION_SCHEMA = "uav_bc_episode_termination/v1"

MSG_WAITING = "[BC Flight] Waiting for runtime inputs..."
MSG_TAKEOFF_STARTED = "[BC Flight] Taking off."
MSG_TAKEOFF_COMPLETE = "[BC Flight] Takeoff complete."
MSG_HANDOFF_COMPLETE = "[BC Flight] Control handed to BC_POLICY."
MSG_GOAL_REACHED = "[BC Flight] Goal reached."
MSG_COLLISION = "[BC Flight] Collision detected."
MSG_TIMEOUT = "[BC Flight] Episode timed out."
MSG_OUT_OF_BOUNDS = "[BC Flight] UAV left the evaluation bounds."
MSG_LANDING = "[BC Flight] Landing..."
MSG_COMPLETE = "[BC Flight] Episode complete."
MSG_FAILED = "[BC Flight] Flight failed: {reason}"
MSG_ACTION_FAILED = "[BC Flight] {action} failed: {error}"
MSG_ACTION_REJECTED = "[BC Flight] {action} rejected: {message}"


class BcFlightSupervisorNode(Node):
    """Request lifecycle commands while navigation decisions stay in BC."""

    def __init__(self) -> None:
        """Create evidence subscriptions, service clients, and timers."""
        super().__init__("bc_flight_supervisor")
        message_module = importlib.import_module("px4_msgs.msg")
        self._VehicleCommand = message_module.VehicleCommand
        self._VehicleStatus = message_module.VehicleStatus
        self._VehicleLandDetected = message_module.VehicleLandDetected

        defaults = BcFlightConfig()
        for name in defaults.__dataclass_fields__:
            self.declare_parameter(name, getattr(defaults, name))
        self.declare_parameter("command_retry_s", 0.50)
        self.declare_parameter("telemetry_timeout_s", 0.75)
        self.config = BcFlightConfig(**{
            name: self.get_parameter(name).value
            for name in defaults.__dataclass_fields__
        })
        self._retry_s = float(self.get_parameter("command_retry_s").value)
        self._telemetry_timeout_s = float(
            self.get_parameter("telemetry_timeout_s").value
        )
        self._controller = BcFlightController(self.config)
        self._runtime_ready = False
        self._policy_status: dict = {}
        self._termination_reason = ""
        self._mux_status: ControlMuxStatus | None = None
        self._gate_status: Px4OutputGateStatus | None = None
        self._stream_status: Px4StreamStatus | None = None
        self._vehicle_status = None
        self._vehicle_status_receipt_s: float | None = None
        self._land_detected = None
        self._odometry: Odometry | None = None
        self._odometry_receipt_s: float | None = None
        self._ground_down_m: float | None = None
        self._last_action_s: dict[str, float] = {}
        self._pending: dict[str, object] = {}
        self._last_state = self._controller.state

        qos = control_qos()
        self._lifecycle_publisher = self.create_publisher(
            TwistStamped, SOURCE_TOPICS[FLIGHT_LIFECYCLE], qos
        )
        self._status_publisher = self.create_publisher(
            String, SUPERVISOR_STATUS_TOPIC, qos
        )
        self.create_subscription(
            String, ISAAC_BRIDGE_STATUS_TOPIC, self._runtime_callback, qos
        )
        self.create_subscription(
            String, POLICY_STATUS_TOPIC, self._policy_callback, qos
        )
        self.create_subscription(
            String,
            EPISODE_TERMINATION_TOPIC,
            self._termination_callback,
            qos,
        )
        self.create_subscription(
            ControlMuxStatus, MUX_STATUS_TOPIC, self._mux_callback, qos
        )
        self.create_subscription(
            Px4OutputGateStatus, GATE_STATUS_TOPIC, self._gate_callback, qos
        )
        self.create_subscription(
            Px4StreamStatus, STREAM_STATUS_TOPIC, self._stream_callback, qos
        )
        self.create_subscription(
            Odometry, ODOMETRY_TOPIC, self._odometry_callback, qos
        )
        telemetry_qos = px4_output_qos()
        self.create_subscription(
            self._VehicleStatus,
            VEHICLE_STATUS_TOPIC,
            self._vehicle_status_callback,
            telemetry_qos,
        )
        self.create_subscription(
            self._VehicleLandDetected,
            VEHICLE_LAND_DETECTED_TOPIC,
            self._land_callback,
            telemetry_qos,
        )
        self._mux_client = self.create_client(
            SetControlSource, SET_SOURCE_SERVICE
        )
        self._gate_client = self.create_client(
            SetPx4OutputEnable, SET_OUTPUT_ENABLE_SERVICE
        )
        self._stream_client = self.create_client(
            SetPx4StreamEnable, SET_STREAM_ENABLE_SERVICE
        )
        self._policy_client = self.create_client(
            SetBool, SET_POLICY_ENABLE_SERVICE
        )
        self._vehicle_command_client = self.create_client(
            SendPx4VehicleCommand, SEND_VEHICLE_COMMAND_SERVICE
        )
        self._timer = self.create_timer(0.05, self._tick)
        self.get_logger().info(MSG_WAITING)

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _runtime_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            self._runtime_ready = bool(payload.get("ready"))
        except (TypeError, ValueError, json.JSONDecodeError):
            self._runtime_ready = False

    def _policy_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get("schema") != POLICY_STATUS_SCHEMA:
                raise ValueError("unknown policy status schema")
            self._policy_status = payload
        except (TypeError, ValueError, json.JSONDecodeError):
            self._policy_status = {}

    def _termination_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get("schema") != TERMINATION_SCHEMA:
                raise ValueError("unknown termination schema")
            reason = str(payload.get("terminal_reason", ""))
            if reason in {"success", "collision", "timeout", "out_of_bounds"}:
                self._termination_reason = reason
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    def _mux_callback(self, message: ControlMuxStatus) -> None:
        self._mux_status = message

    def _gate_callback(self, message: Px4OutputGateStatus) -> None:
        self._gate_status = message

    def _stream_callback(self, message: Px4StreamStatus) -> None:
        self._stream_status = message

    def _vehicle_status_callback(self, message) -> None:
        self._vehicle_status = message
        self._vehicle_status_receipt_s = self._now_seconds()

    def _land_callback(self, message) -> None:
        self._land_detected = message

    def _odometry_callback(self, message: Odometry) -> None:
        self._odometry = message
        self._odometry_receipt_s = self._now_seconds()
        if self._ground_down_m is None and self._is_landed():
            down = float(message.pose.pose.position.z)
            if math.isfinite(down):
                self._ground_down_m = down

    def _is_armed(self) -> bool:
        status = self._vehicle_status
        return bool(
            status is not None
            and status.arming_state == status.ARMING_STATE_ARMED
        )

    def _is_offboard(self) -> bool:
        status = self._vehicle_status
        return bool(
            status is not None
            and status.nav_state == status.NAVIGATION_STATE_OFFBOARD
        )

    def _is_landed(self) -> bool:
        return bool(
            self._land_detected is not None and self._land_detected.landed
        )

    def _evidence(self, now: float) -> BcFlightEvidence:
        mux = self._mux_status
        lifecycle_selected = bool(
            mux is not None and mux.active_source == FLIGHT_LIFECYCLE
        )
        bc_selected = bool(mux is not None and mux.active_source == BC_POLICY)
        source_valid = bool(
            mux is not None
            and mux.selected_command_valid
            and not mux.hold_active
            and (lifecycle_selected or bc_selected)
        )
        gate = self._gate_status
        output_ready = bool(
            gate is not None
            and not gate.enable_requested
            and gate.state == "READY_DISABLED"
        )
        output_safe = bool(
            gate is not None
            and gate.safe_to_forward
            and gate.state == "SAFE_TO_FORWARD"
        )
        stream = self._stream_status
        stream_stable = bool(
            stream is not None
            and stream.streaming
            and stream.state == "STREAMING"
        )
        telemetry_fresh = bool(
            self._vehicle_status_receipt_s is not None
            and self._odometry_receipt_s is not None
            and now - self._vehicle_status_receipt_s
            <= self._telemetry_timeout_s
            and now - self._odometry_receipt_s <= 0.25
        )
        altitude = 0.0
        if self._odometry is not None and self._ground_down_m is not None:
            altitude = self._ground_down_m - float(
                self._odometry.pose.pose.position.z
            )
        policy = self._policy_status
        return BcFlightEvidence(
            runtime_ready=self._runtime_ready,
            observations_ready=bool(policy.get("ready")),
            lifecycle_selected=lifecycle_selected,
            bc_enabled=bool(policy.get("enabled")),
            bc_ready=bool(policy.get("ready")),
            bc_selected=bc_selected,
            source_valid=source_valid,
            output_ready=output_ready,
            output_safe=output_safe,
            stream_stable=stream_stable,
            offboard_active=self._is_offboard(),
            vehicle_armed=self._is_armed(),
            landed=self._is_landed(),
            telemetry_fresh=telemetry_fresh,
            failsafe=bool(
                self._vehicle_status is not None
                and self._vehicle_status.failsafe
            ),
            altitude_m=altitude,
            terminal_reason=self._termination_reason,
        )

    def _publish_lifecycle_command(self) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = VALID_COMMAND_FRAME
        if self._controller.state == BcFlightState.TAKING_OFF:
            message.twist.linear.z = -self.config.takeoff_up_speed_mps
        self._lifecycle_publisher.publish(message)

    def _request(self, key: str, client, request, now: float) -> None:
        if now - self._last_action_s.get(key, -math.inf) < self._retry_s:
            return
        pending = self._pending.get(key)
        if pending is not None and not pending.done():
            return
        if not client.service_is_ready():
            return
        self._last_action_s[key] = now
        future = client.call_async(request)
        self._pending[key] = future
        future.add_done_callback(
            lambda completed, action=key: self._service_done(action, completed)
        )

    def _service_done(self, action: str, future) -> None:
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001
            self.get_logger().error(MSG_ACTION_FAILED.format(
                action=action, error=error
            ))
            return
        accepted = getattr(response, "accepted", None)
        if accepted is None:
            accepted = getattr(response, "success", False)
        message = getattr(response, "status_message", None)
        if message is None:
            message = getattr(response, "message", "")
        if not accepted:
            self.get_logger().warning(
                MSG_ACTION_REJECTED.format(action=action, message=message)
            )

    def _publish_vehicle_command(
        self, key: str, command: int, now: float, **parameters: float
    ) -> None:
        request = SendPx4VehicleCommand.Request()
        for index in range(1, 8):
            setattr(
                request,
                f"param{index}",
                float(parameters.get(f"param{index}", 0.0)),
            )
        request.command = int(command)
        self._request(key, self._vehicle_command_client, request, now)

    def _execute(self, action: str, now: float) -> None:
        if action in {"SELECT_LIFECYCLE", "SELECT_BC", "SELECT_HOLD"}:
            sources = {
                "SELECT_LIFECYCLE": FLIGHT_LIFECYCLE,
                "SELECT_BC": BC_POLICY,
                "SELECT_HOLD": HOLD,
            }
            request = SetControlSource.Request()
            request.source = sources[action]
            self._request(action, self._mux_client, request, now)
        elif action in {"ENABLE_OUTPUT", "DISABLE_OUTPUT"}:
            request = SetPx4OutputEnable.Request()
            request.enable = action == "ENABLE_OUTPUT"
            self._request(action, self._gate_client, request, now)
        elif action in {"ENABLE_STREAM", "DISABLE_STREAM"}:
            request = SetPx4StreamEnable.Request()
            request.enable = action == "ENABLE_STREAM"
            self._request(action, self._stream_client, request, now)
        elif action in {"ENABLE_BC", "DISABLE_BC"}:
            request = SetBool.Request()
            request.data = action == "ENABLE_BC"
            self._request(action, self._policy_client, request, now)
        elif action == "SEND_OFFBOARD":
            self._publish_vehicle_command(
                action,
                self._VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                now,
                param1=1.0,
                param2=6.0,
            )
        elif action == "SEND_ARM":
            self._publish_vehicle_command(
                action,
                self._VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                now,
                param1=1.0,
            )
        elif action == "SEND_LAND":
            self._publish_vehicle_command(
                action, self._VehicleCommand.VEHICLE_CMD_NAV_LAND, now
            )

    def _log_transition(
        self, previous: BcFlightState, current: BcFlightState
    ) -> None:
        if current == BcFlightState.TAKING_OFF:
            self.get_logger().info(MSG_TAKEOFF_STARTED)
        elif current == BcFlightState.ENABLING_BC:
            self.get_logger().info(MSG_TAKEOFF_COMPLETE)
        elif current == BcFlightState.NAVIGATING:
            self.get_logger().info(MSG_HANDOFF_COMPLETE)
        elif current == BcFlightState.HOLDING:
            messages = {
                "success": MSG_GOAL_REACHED,
                "collision": MSG_COLLISION,
                "timeout": MSG_TIMEOUT,
                "out_of_bounds": MSG_OUT_OF_BOUNDS,
            }
            message = messages.get(self._controller.terminal_reason)
            if message:
                self.get_logger().info(message)
        elif current == BcFlightState.LANDING:
            self.get_logger().info(MSG_LANDING)
        elif current == BcFlightState.COMPLETE:
            self.get_logger().info(MSG_COMPLETE)
        elif current == BcFlightState.FAILED:
            self.get_logger().error(
                MSG_FAILED.format(reason=self._controller.failure_reason)
            )

    def _publish_status(self, evidence: BcFlightEvidence) -> None:
        payload = {
            "schema": SUPERVISOR_STATUS_SCHEMA,
            "state": self._controller.state.value,
            "terminal_reason": self._controller.terminal_reason,
            "failure_reason": self._controller.failure_reason,
            "altitude_m": evidence.altitude_m,
            "runtime_ready": evidence.runtime_ready,
            "bc_ready": evidence.bc_ready,
            "active_source": (
                ""
                if self._mux_status is None
                else self._mux_status.active_source
            ),
            "vehicle_armed": evidence.vehicle_armed,
            "landed": evidence.landed,
        }
        message = String()
        message.data = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
        self._status_publisher.publish(message)

    def _tick(self) -> None:
        now = self._now_seconds()
        self._publish_lifecycle_command()
        evidence = self._evidence(now)
        previous = self._controller.state
        decision = self._controller.step(now, evidence)
        for action in decision.actions:
            self._execute(action, now)
        if decision.state != previous:
            self._log_transition(previous, decision.state)
        self._publish_status(evidence)


def main(args=None) -> int:
    """Run one automatically started BC flight supervisor."""
    rclpy.init(args=args)
    node = BcFlightSupervisorNode()
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
