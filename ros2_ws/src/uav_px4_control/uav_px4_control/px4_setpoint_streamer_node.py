"""Sole Phase 8 owner of the two allowed live PX4 SITL input topics."""

import importlib
import math
import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from std_msgs.msg import Bool

from uav_interfaces.msg import (
    Px4OutputGateStatus,
    Px4SetpointCandidate,
    Px4StreamStatus,
)
from uav_interfaces.srv import SetPx4StreamEnable

from uav_px4_control.control_mux_node import control_qos
from uav_px4_control.px4_mapping_gate_node import (
    CANDIDATE_TOPIC,
    GATE_STATUS_TOPIC,
    SAFE_TO_FORWARD_TOPIC,
)
from uav_px4_control.px4_message_adapter import (
    offboard_control_mode_fields,
    trajectory_setpoint_fields,
)
from uav_px4_control.px4_stream_models import (
    Px4StreamConfig,
    StreamCandidate,
    StreamGateEvidence,
    StreamReadiness,
    StreamTelemetry,
)
from uav_px4_control.px4_stream_state_machine import Px4StreamStateMachine


TRAJECTORY_SETPOINT_TOPIC = "/fmu/in/trajectory_setpoint"
OFFBOARD_CONTROL_MODE_TOPIC = "/fmu/in/offboard_control_mode"
VEHICLE_STATUS_TOPIC = "/fmu/out/vehicle_status"
VEHICLE_CONTROL_MODE_TOPIC = "/fmu/out/vehicle_control_mode"
VEHICLE_ODOMETRY_TOPIC = "/fmu/out/vehicle_odometry"
FAILSAFE_FLAGS_TOPIC = "/fmu/out/failsafe_flags"
STREAM_STATUS_TOPIC = "/uav/px4/stream_status"
SET_STREAM_ENABLE_SERVICE = "/uav/px4/set_stream_enable"


def px4_input_qos() -> QoSProfile:
    """Match the local uXRCE input subscriber contract."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def px4_output_qos() -> QoSProfile:
    """Request the local uXRCE telemetry publisher contract."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def pegasus_sitl_identity_matches(
    expected_fragment: str,
    executable: str,
    command: str,
    environment: set[bytes],
) -> bool:
    """Recognize only the audited Pegasus instance-zero PX4 launch shape."""
    if not executable.endswith(expected_fragment):
        return False
    build_marker = "/build/px4_sitl_default/bin/px4"
    if build_marker not in executable:
        return False
    px4_root = executable.split(build_marker, 1)[0]
    rc_script = px4_root + "/ROMFS/px4fmu_common/init.d-posix/rcS"
    required_arguments = f" -s {rc_script} -i 0 -d"
    return bool(
        required_arguments in command
        and b"PX4_SIM_MODEL=gazebo-classic_iris" in environment
    )


def sitl_process_matches(expected_fragment: str) -> bool:
    """Return true only for a local PX4 SITL process with the expected path."""
    if not expected_fragment or "px4_sitl" not in expected_fragment:
        return False
    try:
        process_ids = os.listdir("/proc")
    except OSError:
        return False
    for process_id in process_ids:
        if not process_id.isdigit():
            continue
        try:
            with open(
                f"/proc/{process_id}/cmdline",
                "rb",
            ) as command_file:
                command = command_file.read().replace(b"\0", b" ").decode(
                    errors="replace"
                )
        except (OSError, PermissionError):
            continue
        try:
            executable = os.readlink(f"/proc/{process_id}/exe")
        except OSError:
            executable = ""
        identity_matches = (
            expected_fragment in command
            or expected_fragment in executable
        )
        if not identity_matches:
            continue
        if "etc/init.d-posix/rcS" in command:
            return True

        # PX4's locally audited ``sihsim_*`` CMake targets start the same SITL
        # executable without spelling rcS on argv.  Accept that exact launch
        # shape only when the process itself supplies all three independent
        # pieces of evidence: the expected binary, the build rootfs cwd, and
        # the built-in SIH environment selected by the target definition.
        try:
            working_directory = os.readlink(f"/proc/{process_id}/cwd")
            with open(f"/proc/{process_id}/environ", "rb") as env_file:
                environment = set(env_file.read().split(b"\0"))
        except (OSError, PermissionError):
            continue
        sih_environment = (
            b"PX4_SIMULATOR=sihsim" in environment
            and any(
                item.startswith(b"PX4_SIM_MODEL=sihsim_")
                for item in environment
            )
        )
        if pegasus_sitl_identity_matches(
            expected_fragment,
            executable,
            command,
            environment,
        ):
            return True
        expected_rootfs = executable.rsplit("/bin/px4", 1)[0] + "/rootfs"
        if (
            executable.endswith(expected_fragment)
            and working_directory == expected_rootfs
            and sih_environment
        ):
            return True
    return False


def xrce_agent_detected() -> bool:
    """Require the audited local UDP/8888 XRCE Agent process."""
    try:
        process_ids = os.listdir("/proc")
    except OSError:
        return False
    for process_id in process_ids:
        if not process_id.isdigit():
            continue
        try:
            executable = os.readlink(f"/proc/{process_id}/exe")
            with open(f"/proc/{process_id}/cmdline", "rb") as command_file:
                command = command_file.read().replace(b"\0", b" ").decode(
                    errors="replace"
                )
        except (OSError, PermissionError):
            continue
        if (
            "MicroXRCEAgent" in executable
            and "udp4" in command
            and "8888" in command
        ):
            return True
    return False


class Px4SetpointStreamerNode(Node):
    """Publish only velocity setpoint and mode heartbeat after both gates."""

    def __init__(self) -> None:
        """Create the single live owner; fail if px4_msgs is unavailable."""
        super().__init__("px4_setpoint_streamer")
        message_module = importlib.import_module("px4_msgs.msg")
        self._TrajectorySetpoint = message_module.TrajectorySetpoint
        self._OffboardControlMode = message_module.OffboardControlMode
        vehicle_status_type = message_module.VehicleStatus
        vehicle_control_mode_type = message_module.VehicleControlMode
        vehicle_odometry_type = message_module.VehicleOdometry
        failsafe_flags_type = message_module.FailsafeFlags

        defaults = Px4StreamConfig()
        for name in defaults.__dataclass_fields__:
            self.declare_parameter(name, getattr(defaults, name))
        self.declare_parameter(
            "expected_sitl_process_fragment",
            "/PX4-Autopilot/build/px4_sitl_default/bin/px4",
        )
        self.config = Px4StreamConfig(**{
            name: self.get_parameter(name).value
            for name in defaults.__dataclass_fields__
        })
        self.machine = Px4StreamStateMachine(self.config)
        self._candidate: StreamCandidate | None = None
        self._safe_value: bool | None = None
        self._safe_receipt_time_s: float | None = None
        self._gate_status: Px4OutputGateStatus | None = None
        self._gate_receipt_time_s: float | None = None
        self._vehicle_status = None
        self._vehicle_status_receipt: float | None = None
        self._vehicle_control_mode = None
        self._vehicle_control_mode_receipt: float | None = None
        self._vehicle_odometry = None
        self._vehicle_odometry_receipt: float | None = None
        self._failsafe_flags = None
        self._failsafe_flags_receipt: float | None = None
        self._last_result = None

        boundary_qos = control_qos()
        telemetry_qos = px4_output_qos()
        self.create_subscription(
            Px4SetpointCandidate,
            CANDIDATE_TOPIC,
            self._candidate_callback,
            boundary_qos,
        )
        self.create_subscription(
            Bool,
            SAFE_TO_FORWARD_TOPIC,
            self._safe_callback,
            boundary_qos,
        )
        self.create_subscription(
            Px4OutputGateStatus,
            GATE_STATUS_TOPIC,
            self._gate_callback,
            boundary_qos,
        )
        self.create_subscription(
            vehicle_status_type,
            VEHICLE_STATUS_TOPIC,
            self._vehicle_status_callback,
            telemetry_qos,
        )
        self.create_subscription(
            vehicle_control_mode_type,
            VEHICLE_CONTROL_MODE_TOPIC,
            self._vehicle_control_mode_callback,
            telemetry_qos,
        )
        self.create_subscription(
            vehicle_odometry_type,
            VEHICLE_ODOMETRY_TOPIC,
            self._vehicle_odometry_callback,
            telemetry_qos,
        )
        self.create_subscription(
            failsafe_flags_type,
            FAILSAFE_FLAGS_TOPIC,
            self._failsafe_flags_callback,
            telemetry_qos,
        )

        input_qos = px4_input_qos()
        self._trajectory_publisher = self.create_publisher(
            self._TrajectorySetpoint,
            TRAJECTORY_SETPOINT_TOPIC,
            input_qos,
        )
        self._mode_publisher = self.create_publisher(
            self._OffboardControlMode,
            OFFBOARD_CONTROL_MODE_TOPIC,
            input_qos,
        )
        self._status_publisher = self.create_publisher(
            Px4StreamStatus,
            STREAM_STATUS_TOPIC,
            boundary_qos,
        )
        self._service = self.create_service(
            SetPx4StreamEnable,
            SET_STREAM_ENABLE_SERVICE,
            self._enable_callback,
        )
        self._timer = self.create_timer(
            1.0 / self.config.stream_rate_hz,
            self._tick,
        )
        self.get_logger().warning(
            "Phase 8 SITL-only streamer starts disabled; it sends no mode or "
            "arming command"
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _candidate_callback(self, message: Px4SetpointCandidate) -> None:
        now = self._now_seconds()
        self._candidate = StreamCandidate(
            receipt_time_s=now,
            timestamp_us=int(message.timestamp_us),
            velocity_ned_mps=(
                float(message.velocity_ned.x),
                float(message.velocity_ned.y),
                float(message.velocity_ned.z),
            ),
            yaw_rate_ned_radps=float(message.yaw_rate_ned),
            frame_id=message.header.frame_id,
            valid=bool(message.valid),
        )
        self.machine.observe_candidate(self._candidate)

    def _safe_callback(self, message: Bool) -> None:
        self._safe_value = bool(message.data)
        self._safe_receipt_time_s = self._now_seconds()

    def _gate_callback(self, message: Px4OutputGateStatus) -> None:
        self._gate_status = message
        self._gate_receipt_time_s = self._now_seconds()

    def _vehicle_status_callback(self, message) -> None:
        self._vehicle_status = message
        self._vehicle_status_receipt = self._now_seconds()

    def _vehicle_control_mode_callback(self, message) -> None:
        self._vehicle_control_mode = message
        self._vehicle_control_mode_receipt = self._now_seconds()

    def _vehicle_odometry_callback(self, message) -> None:
        self._vehicle_odometry = message
        self._vehicle_odometry_receipt = self._now_seconds()

    def _failsafe_flags_callback(self, message) -> None:
        self._failsafe_flags = message
        self._failsafe_flags_receipt = self._now_seconds()

    def _gate_evidence(self) -> StreamGateEvidence | None:
        if (
            self._safe_value is None
            or self._safe_receipt_time_s is None
            or self._gate_status is None
            or self._gate_receipt_time_s is None
        ):
            return None
        return StreamGateEvidence(
            bool_receipt_time_s=self._safe_receipt_time_s,
            bool_safe_to_forward=self._safe_value,
            status_receipt_time_s=self._gate_receipt_time_s,
            status_safe_to_forward=self._gate_status.safe_to_forward,
            status_state=self._gate_status.state,
        )

    def _telemetry(self) -> StreamTelemetry | None:
        messages = (
            self._vehicle_status,
            self._vehicle_control_mode,
            self._vehicle_odometry,
            self._failsafe_flags,
        )
        receipts = (
            self._vehicle_status_receipt,
            self._vehicle_control_mode_receipt,
            self._vehicle_odometry_receipt,
            self._failsafe_flags_receipt,
        )
        if any(message is None for message in messages):
            return None
        if any(receipt is None for receipt in receipts):
            return None
        status = self._vehicle_status
        mode = self._vehicle_control_mode
        odometry = self._vehicle_odometry
        flags = self._failsafe_flags
        odometry_values = (*odometry.position, *odometry.velocity)
        odometry_valid = (
            odometry.pose_frame == odometry.POSE_FRAME_NED
            and odometry.velocity_frame == odometry.VELOCITY_FRAME_NED
            and all(math.isfinite(float(value)) for value in odometry_values)
        )
        vehicle_armed = bool(
            status.arming_state == status.ARMING_STATE_ARMED
            or mode.flag_armed
        )
        offboard_active = bool(
            status.nav_state == status.NAVIGATION_STATE_OFFBOARD
            or mode.flag_control_offboard_enabled
        )
        # ``offboard_control_signal_lost`` is expected before the first
        # OffboardControlMode heartbeat and must not deadlock prestream.  It is
        # still consumed and exposed through the Phase 7 telemetry adapter;
        # unexpected OFFBOARD activity is independently fail-closed above.
        failsafe = bool(status.failsafe or flags.fd_critical_failure)
        return StreamTelemetry(
            oldest_receipt_time_s=min(float(value) for value in receipts),
            newest_timestamp_us=max(
                int(message.timestamp) for message in messages
            ),
            vehicle_armed=vehicle_armed,
            offboard_active=offboard_active,
            failsafe=failsafe,
            odometry_valid=odometry_valid,
        )

    def _sitl_guard_valid(self) -> bool:
        if not self.config.simulation_mode:
            return False
        if not self.config.allow_sitl_streaming_only:
            return False
        fragment = str(
            self.get_parameter("expected_sitl_process_fragment").value
        )
        return sitl_process_matches(fragment)

    def _dds_ready(self) -> bool:
        return bool(
            xrce_agent_detected()
            and self._trajectory_publisher.get_subscription_count() >= 1
            and self._mode_publisher.get_subscription_count() >= 1
        )

    def _readiness(self) -> StreamReadiness:
        return StreamReadiness(
            sitl_guard_valid=self._sitl_guard_valid(),
            dds_ready=self._dds_ready(),
            gate=self._gate_evidence(),
            candidate=self._candidate,
            telemetry=self._telemetry(),
        )

    def _enable_callback(self, request, response):
        accepted, status_message = self.machine.request_enable(request.enable)
        response.accepted = accepted
        response.stream_enable_requested = (
            self.machine.stream_enable_requested
        )
        response.streaming = bool(
            self._last_result is not None and self._last_result.streaming
        )
        response.state = self.machine.state.value
        response.status_message = status_message
        return response

    def _tick(self) -> None:
        now_clock = self.get_clock().now()
        now = now_clock.nanoseconds / 1e9
        timestamp_us = now_clock.nanoseconds // 1000
        readiness = self._readiness()
        result = self.machine.step(now, readiness, timestamp_us)
        if result.should_publish:
            try:
                trajectory = trajectory_setpoint_fields(
                    readiness.candidate,
                    timestamp_us,
                )
                mode = offboard_control_mode_fields(timestamp_us)
                self._trajectory_publisher.publish(
                    self._trajectory_message(trajectory)
                )
                self._mode_publisher.publish(self._mode_message(mode))
            except (TypeError, ValueError) as error:
                result = self.machine.force_mapping_fault(
                    now,
                    readiness,
                    f"message adapter rejected output: {error}",
                )
        self._last_result = result
        self._status_publisher.publish(
            self._status_message(result, now_clock.to_msg())
        )

    def _trajectory_message(self, fields):
        message = self._TrajectorySetpoint()
        message.timestamp = fields.timestamp
        message.position = list(fields.position)
        message.velocity = list(fields.velocity)
        message.acceleration = list(fields.acceleration)
        message.jerk = list(fields.jerk)
        message.yaw = fields.yaw
        message.yawspeed = fields.yawspeed
        return message

    def _mode_message(self, fields):
        message = self._OffboardControlMode()
        message.timestamp = fields.timestamp
        message.position = fields.position
        message.velocity = fields.velocity
        message.acceleration = fields.acceleration
        message.attitude = fields.attitude
        message.body_rate = fields.body_rate
        message.actuator = fields.actuator
        return message

    def _status_message(self, result, stamp):
        message = Px4StreamStatus()
        message.header.stamp = stamp
        message.header.frame_id = "px4_ned"
        message.state = result.state.value
        message.stream_enable_requested = result.stream_enable_requested
        message.streaming = result.streaming
        message.sitl_guard_valid = result.sitl_guard_valid
        message.dds_ready = result.dds_ready
        message.gate_valid = result.gate_valid
        message.candidate_valid = result.candidate_valid
        message.telemetry_fresh = result.telemetry_fresh
        message.vehicle_armed = result.vehicle_armed
        message.offboard_active = result.offboard_active
        message.failsafe = result.failsafe
        message.requested_rate_hz = self.config.stream_rate_hz
        message.observed_rate_hz = self.machine.observed_rate_hz
        message.maximum_publish_gap = self.machine.maximum_observed_gap_s
        message.candidate_age = result.candidate_age_s
        message.gate_status_age = result.gate_status_age_s
        message.telemetry_age = result.telemetry_age_s
        message.trajectory_setpoint_count = result.trajectory_setpoint_count
        message.offboard_mode_count = result.offboard_mode_count
        message.dropped_cycle_count = result.dropped_cycle_count
        message.transition_count = result.transition_count
        message.stop_reason = result.stop_reason
        message.status_message = (
            "SITL-only setpoint stream; no mode, arm, or flight command exists"
        )
        return message


def main(args=None) -> int:
    """Run the sole Phase 8 live PX4 setpoint stream owner."""
    rclpy.init(args=args)
    node = Px4SetpointStreamerNode()
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
