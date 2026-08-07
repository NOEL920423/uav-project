"""ROS diagnostics adapter for the offline PX4 mapping and output gate."""

import math

from geometry_msgs.msg import TwistStamped

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import Bool, String

from uav_interfaces.msg import (
    ControlMuxStatus,
    Px4OutputGateStatus,
    Px4SetpointCandidate,
    Px4SyntheticTelemetry,
)
from uav_interfaces.srv import SetPx4OutputEnable

from uav_px4_control.control_mux_node import (
    MUX_STATUS_TOPIC,
    SELECTED_COMMAND_TOPIC,
    SOURCE_TOPIC,
    control_qos,
)
from uav_px4_control.control_source_models import (
    ControlCommand,
    Vector3,
)
from uav_px4_control.px4_boundary_models import (
    CandidateValidation,
    MuxHealthEvidence,
    Px4MappingConfig,
    Px4OutputGateResult,
    Px4TelemetryState,
    Px4VelocitySetpointCandidate,
)
from uav_px4_control.px4_candidate_validator import validate_px4_candidate
from uav_px4_control.px4_output_gate import Px4OutputSafetyGate
from uav_px4_control.px4_setpoint_mapper import map_selected_command
from uav_px4_control.px4_timestamp import ros_stamp_to_microseconds


CANDIDATE_TOPIC = "/uav/px4/setpoint_candidate"
GATE_STATUS_TOPIC = "/uav/px4/output_gate_status"
SAFE_TO_FORWARD_TOPIC = "/uav/px4/safe_to_forward"
SYNTHETIC_TELEMETRY_TOPIC = "/uav/test/px4/telemetry_status"
SET_OUTPUT_ENABLE_SERVICE = "/uav/px4/set_output_enable"


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


class Px4MappingGateNode(Node):
    """Map selected commands and publish diagnostics through a pure gate."""

    def __init__(self) -> None:
        """Create subscriptions, diagnostic publishers, service, and timer."""
        super().__init__("px4_mapping_gate")
        defaults = Px4MappingConfig()
        for name in defaults.__dataclass_fields__:
            self.declare_parameter(name, getattr(defaults, name))
        self.declare_parameter("publish_rate_hz", 50.0)
        self.config = Px4MappingConfig(**{
            name: self.get_parameter(name).value
            for name in defaults.__dataclass_fields__
        })
        publish_rate = float(self.get_parameter("publish_rate_hz").value)
        if not math.isfinite(publish_rate) or publish_rate <= 0.0:
            raise ValueError("publish_rate_hz must be finite and positive")
        self.gate = Px4OutputSafetyGate(self.config)
        self._selected_message: TwistStamped | None = None
        self._selected_receipt_time_s: float | None = None
        self._source = ""
        self._mux_status: ControlMuxStatus | None = None
        self._mux_receipt_time_s: float | None = None
        self._telemetry: Px4TelemetryState | None = None
        self._last_candidate_timestamp_us: int | None = None
        self._last_result: Px4OutputGateResult | None = None
        qos = control_qos()
        self.create_subscription(
            TwistStamped,
            SELECTED_COMMAND_TOPIC,
            self._selected_callback,
            qos,
        )
        self.create_subscription(
            String, SOURCE_TOPIC, self._source_callback, qos
        )
        self.create_subscription(
            ControlMuxStatus,
            MUX_STATUS_TOPIC,
            self._mux_callback,
            qos,
        )
        self.create_subscription(
            Px4SyntheticTelemetry,
            SYNTHETIC_TELEMETRY_TOPIC,
            self._telemetry_callback,
            qos,
        )
        self._candidate_publisher = self.create_publisher(
            Px4SetpointCandidate, CANDIDATE_TOPIC, qos
        )
        self._status_publisher = self.create_publisher(
            Px4OutputGateStatus, GATE_STATUS_TOPIC, qos
        )
        self._safe_publisher = self.create_publisher(
            Bool, SAFE_TO_FORWARD_TOPIC, qos
        )
        self._service = self.create_service(
            SetPx4OutputEnable,
            SET_OUTPUT_ENABLE_SERVICE,
            self._enable_callback,
        )
        self._timer = self.create_timer(1.0 / publish_rate, self._tick)
        self.get_logger().warning(
            "Phase 7 diagnostic boundary active; no PX4 publisher exists"
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _selected_callback(self, message: TwistStamped) -> None:
        self._selected_message = message
        self._selected_receipt_time_s = self._now_seconds()

    def _source_callback(self, message: String) -> None:
        self._source = message.data

    def _mux_callback(self, message: ControlMuxStatus) -> None:
        self._mux_status = message
        self._mux_receipt_time_s = self._now_seconds()

    def _telemetry_callback(self, message: Px4SyntheticTelemetry) -> None:
        self._telemetry = Px4TelemetryState(
            receipt_time_s=self._now_seconds(),
            timestamp_us=message.timestamp_us,
            connected=message.connected,
            arming_state=message.arming_state,
            nav_state=message.navigation_state,
            offboard_control_signal_lost=(
                message.offboard_control_signal_lost
            ),
            offboard_active=message.offboard_active,
            failsafe=message.failsafe,
            pre_flight_checks_pass=message.pre_flight_checks_pass,
            local_position_valid=message.local_position_valid,
            local_velocity_valid=message.local_velocity_valid,
            odometry_valid=message.odometry_valid,
            pose_frame_ned=message.pose_frame_ned,
            velocity_frame_ned=message.velocity_frame_ned,
        )

    def _mux_evidence(self) -> MuxHealthEvidence | None:
        message = self._mux_status
        receipt = self._mux_receipt_time_s
        if message is None or receipt is None:
            return None
        return MuxHealthEvidence(
            received=True,
            selected_command_valid=message.selected_command_valid,
            hold_active=message.hold_active,
            active_source=message.active_source,
            receipt_time_s=receipt,
        )

    def _candidate(
        self, now: float, mux: MuxHealthEvidence | None
    ) -> tuple[
        Px4VelocitySetpointCandidate | None,
        CandidateValidation | None,
    ]:
        message = self._selected_message
        receipt = self._selected_receipt_time_s
        if message is None or receipt is None or not self._source:
            return None, None
        try:
            timestamp_us = ros_stamp_to_microseconds(
                message.header.stamp.sec,
                message.header.stamp.nanosec,
            )
            command = ControlCommand(
                source=self._source,
                timestamp_s=_stamp_seconds(message.header.stamp),
                frame_id=message.header.frame_id,
                linear=Vector3(
                    message.twist.linear.x,
                    message.twist.linear.y,
                    message.twist.linear.z,
                ),
                angular_x=message.twist.angular.x,
                angular_y=message.twist.angular.y,
                yaw_rate_radps=message.twist.angular.z,
            )
            candidate = map_selected_command(command, timestamp_us, receipt)
        except (TypeError, ValueError, OverflowError) as error:
            self.get_logger().error(f"candidate mapping failed: {error}")
            return None, None
        previous = None
        if timestamp_us != self._last_candidate_timestamp_us:
            previous = self._last_candidate_timestamp_us
        mux_valid = mux is not None and mux.selected_command_valid
        mux_source = "" if mux is None else mux.active_source
        validation = validate_px4_candidate(
            candidate,
            self.config,
            now,
            mux_valid,
            mux_source,
            previous,
        )
        if (
            validation.valid
            and timestamp_us != self._last_candidate_timestamp_us
        ):
            self._last_candidate_timestamp_us = timestamp_us
        return candidate, validation

    def _enable_callback(self, request, response):
        result = self.gate.request_enable(request.enable)
        response.accepted = result.accepted
        response.enable_requested = self.gate.enabled
        response.safe_to_forward = (
            self._last_result.safe_to_forward
            if self._last_result is not None
            else False
        )
        response.state = self.gate.state.value
        response.status_message = result.message
        return response

    def _tick(self) -> None:
        now = self._now_seconds()
        mux = self._mux_evidence()
        candidate, validation = self._candidate(now, mux)
        result = self.gate.step(
            now, candidate, validation, mux, self._telemetry
        )
        self._last_result = result
        stamp = self.get_clock().now().to_msg()
        if candidate is not None and validation is not None:
            self._candidate_publisher.publish(
                self._candidate_message(candidate, validation, stamp)
            )
        self._status_publisher.publish(
            self._status_message(result, validation, candidate, stamp)
        )
        self._safe_publisher.publish(Bool(data=result.safe_to_forward))

    @staticmethod
    def _candidate_message(candidate, validation, stamp):
        message = Px4SetpointCandidate()
        message.header.stamp = stamp
        message.header.frame_id = candidate.frame_id
        message.source = candidate.source
        message.velocity_ned.x = candidate.velocity_ned_mps[0]
        message.velocity_ned.y = candidate.velocity_ned_mps[1]
        message.velocity_ned.z = candidate.velocity_ned_mps[2]
        message.yaw_rate_ned = candidate.yaw_rate_ned_radps
        message.timestamp_us = candidate.timestamp_us
        message.valid = validation.valid
        message.status_message = validation.reason
        return message

    def _status_message(self, result, validation, candidate, stamp):
        message = Px4OutputGateStatus()
        message.header.stamp = stamp
        message.header.frame_id = "px4_ned"
        message.state = result.state.value
        message.enable_requested = result.enabled
        message.safe_to_forward = result.safe_to_forward
        message.selected_command_valid = result.selected_command_valid
        message.mux_valid = result.mux_valid
        message.telemetry_valid = result.telemetry_valid
        message.failsafe = result.failsafe
        message.active_source = result.active_source
        message.selected_command_age = result.selected_command_age_s
        message.telemetry_age = result.telemetry_age_s
        message.candidate_horizontal_speed = (
            validation.horizontal_speed_mps if validation else math.nan
        )
        message.candidate_total_speed = (
            validation.total_speed_mps if validation else math.nan
        )
        message.candidate_yaw_rate = (
            candidate.yaw_rate_ned_radps if candidate else math.nan
        )
        message.transition_count = result.transition_count
        message.hold_reason = result.hold_reason
        message.status_message = (
            "offline diagnostic only; no PX4 command was published"
        )
        return message


def main(args=None) -> int:
    """Run the offline diagnostic PX4 mapping and safety-gate node."""
    rclpy.init(args=args)
    node = Px4MappingGateNode()
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
