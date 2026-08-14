"""Adapt read-only local PX4 telemetry to the existing Phase 7 gate model."""

import importlib
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from uav_interfaces.msg import Px4SyntheticTelemetry

from uav_px4_control.control_mux_node import control_qos
from uav_px4_control.px4_mapping_gate_node import SYNTHETIC_TELEMETRY_TOPIC
from uav_px4_control.px4_setpoint_streamer_node import (
    FAILSAFE_FLAGS_TOPIC,
    VEHICLE_CONTROL_MODE_TOPIC,
    VEHICLE_ODOMETRY_TOPIC,
    VEHICLE_STATUS_TOPIC,
    px4_output_qos,
)


class Px4LiveTelemetryAdapter(Node):
    """Publish only custom safety evidence derived from four live outputs."""

    def __init__(self) -> None:
        """Subscribe read-only and expose the established Phase 7 contract."""
        super().__init__("px4_live_telemetry_adapter")
        message_module = importlib.import_module("px4_msgs.msg")
        self._status = None
        self._mode = None
        self._odometry = None
        self._flags = None
        qos = px4_output_qos()
        self.create_subscription(
            message_module.VehicleStatus,
            VEHICLE_STATUS_TOPIC,
            self._status_callback,
            qos,
        )
        self.create_subscription(
            message_module.VehicleControlMode,
            VEHICLE_CONTROL_MODE_TOPIC,
            self._mode_callback,
            qos,
        )
        self.create_subscription(
            message_module.VehicleOdometry,
            VEHICLE_ODOMETRY_TOPIC,
            self._odometry_callback,
            qos,
        )
        self.create_subscription(
            message_module.FailsafeFlags,
            FAILSAFE_FLAGS_TOPIC,
            self._flags_callback,
            qos,
        )
        self._publisher = self.create_publisher(
            Px4SyntheticTelemetry,
            SYNTHETIC_TELEMETRY_TOPIC,
            control_qos(),
        )
        self.create_timer(0.05, self._tick)

    def _status_callback(self, message) -> None:
        self._status = message

    def _mode_callback(self, message) -> None:
        self._mode = message

    def _odometry_callback(self, message) -> None:
        self._odometry = message

    def _flags_callback(self, message) -> None:
        self._flags = message

    def _tick(self) -> None:
        if any(
            item is None
            for item in (self._status, self._mode, self._odometry, self._flags)
        ):
            return
        status = self._status
        mode = self._mode
        odometry = self._odometry
        flags = self._flags
        values = (*odometry.position, *odometry.velocity)
        odometry_valid = all(math.isfinite(float(value)) for value in values)
        message = Px4SyntheticTelemetry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "px4_ned"
        message.timestamp_us = max(
            int(status.timestamp),
            int(mode.timestamp),
            int(odometry.timestamp),
            int(flags.timestamp),
        )
        message.connected = True
        message.arming_state = status.arming_state
        message.navigation_state = status.nav_state
        offboard_active = bool(
            mode.flag_control_offboard_enabled
            or status.nav_state == status.NAVIGATION_STATE_OFFBOARD
        )
        message.offboard_active = offboard_active
        # Before prestream, PX4 correctly reports that it has no offboard
        # heartbeat.  Feeding that mode-specific flag into the Phase 7 gate
        # while OFFBOARD is inactive would make prestream impossible.  If
        # OFFBOARD ever becomes active unexpectedly, preserve the live flag;
        # ``offboard_active`` independently closes both safety layers too.
        message.offboard_control_signal_lost = bool(
            offboard_active and flags.offboard_control_signal_lost
        )
        message.failsafe = bool(
            status.failsafe or flags.fd_critical_failure
        )
        message.pre_flight_checks_pass = status.pre_flight_checks_pass
        message.local_position_valid = not flags.local_position_invalid
        message.local_velocity_valid = not flags.local_velocity_invalid
        message.odometry_valid = odometry_valid
        message.pose_frame_ned = (
            odometry.pose_frame == odometry.POSE_FRAME_NED
        )
        message.velocity_frame_ned = (
            odometry.velocity_frame == odometry.VELOCITY_FRAME_NED
        )
        message.fixture = "live-px4-read-only"
        self._publisher.publish(message)


def main(args=None) -> int:
    """Run the read-only live PX4 telemetry adapter."""
    rclpy.init(args=args)
    node = Px4LiveTelemetryAdapter()
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
