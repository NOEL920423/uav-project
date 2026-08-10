"""Read-only finite doctor for the local PX4 SITL and uXRCE graph."""

import importlib
import time

import rclpy
from rclpy.node import Node

from uav_px4_control.px4_setpoint_streamer_node import (
    FAILSAFE_FLAGS_TOPIC,
    OFFBOARD_CONTROL_MODE_TOPIC,
    TRAJECTORY_SETPOINT_TOPIC,
    VEHICLE_CONTROL_MODE_TOPIC,
    VEHICLE_ODOMETRY_TOPIC,
    VEHICLE_STATUS_TOPIC,
    px4_output_qos,
    sitl_process_matches,
    xrce_agent_detected,
)


TELEMETRY_TOPICS = (
    VEHICLE_STATUS_TOPIC,
    VEHICLE_CONTROL_MODE_TOPIC,
    VEHICLE_ODOMETRY_TOPIC,
    FAILSAFE_FLAGS_TOPIC,
)


def _qos_text(endpoint) -> str:
    profile = endpoint.qos_profile
    return (
        f"reliability={profile.reliability.name},"
        f"durability={profile.durability.name},"
        f"history={profile.history.name},depth={profile.depth}"
    )


class Px4SitlDoctorNode(Node):
    """Collect telemetry without creating any PX4 input publisher."""

    def __init__(self) -> None:
        """Subscribe to the four required live output topics read-only."""
        super().__init__("px4_sitl_doctor")
        message_module = importlib.import_module("px4_msgs.msg")
        self.status = None
        self.mode = None
        self.odometry = None
        self.flags = None
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

    def _status_callback(self, message) -> None:
        self.status = message

    def _mode_callback(self, message) -> None:
        self.mode = message

    def _odometry_callback(self, message) -> None:
        self.odometry = message

    def _flags_callback(self, message) -> None:
        self.flags = message

    def complete(self) -> bool:
        """Return true after all required telemetry has arrived."""
        return all(
            item is not None
            for item in (self.status, self.mode, self.odometry, self.flags)
        )

    def readiness_snapshot(self) -> bool:
        """Return one conservative process/telemetry/endpoint observation."""
        expected = "/PX4-Autopilot/build/px4_sitl_default/bin/px4"
        if not self.complete():
            return False
        if not sitl_process_matches(expected) or not xrce_agent_detected():
            return False
        if any(
            not self.get_publishers_info_by_topic(topic)
            for topic in TELEMETRY_TOPICS
        ):
            return False
        return all(
            self.get_subscriptions_info_by_topic(topic)
            for topic in (
                TRAJECTORY_SETPOINT_TOPIC,
                OFFBOARD_CONTROL_MODE_TOPIC,
            )
        )

    def validate(self) -> tuple[bool, list[str]]:
        """Validate process, endpoint, QoS, and vehicle safety evidence."""
        details: list[str] = []
        ok = True
        expected = "/PX4-Autopilot/build/px4_sitl_default/bin/px4"
        sitl = sitl_process_matches(expected)
        agent = xrce_agent_detected()
        details.append(f"PX4 SITL detected: {str(sitl).upper()}")
        details.append(f"XRCE Agent detected: {str(agent).upper()}")
        ok = ok and sitl and agent and self.complete()

        for topic in TELEMETRY_TOPICS:
            endpoints = self.get_publishers_info_by_topic(topic)
            details.append(
                f"{topic}: publishers={len(endpoints)} "
                + ";".join(_qos_text(endpoint) for endpoint in endpoints)
            )
            ok = ok and len(endpoints) >= 1
        for topic in (
            TRAJECTORY_SETPOINT_TOPIC,
            OFFBOARD_CONTROL_MODE_TOPIC,
        ):
            endpoints = self.get_subscriptions_info_by_topic(topic)
            details.append(
                f"{topic}: subscribers={len(endpoints)} "
                + ";".join(_qos_text(endpoint) for endpoint in endpoints)
            )
            ok = ok and len(endpoints) >= 1

        command_publishers = self.get_publishers_info_by_topic(
            "/fmu/in/vehicle_command"
        )
        details.append(
            "VehicleCommand publisher absent: "
            + str(len(command_publishers) == 0).upper()
        )
        ok = ok and len(command_publishers) == 0
        if not self.complete():
            details.append("required live telemetry did not arrive")
            return False, details

        armed = bool(
            self.status.arming_state == self.status.ARMING_STATE_ARMED
            or self.mode.flag_armed
        )
        offboard = bool(
            self.status.nav_state == self.status.NAVIGATION_STATE_OFFBOARD
            or self.mode.flag_control_offboard_enabled
        )
        failsafe = bool(
            self.status.failsafe or self.flags.fd_critical_failure
        )
        details.append(f"vehicle DISARMED: {str(not armed).upper()}")
        details.append(f"OFFBOARD INACTIVE: {str(not offboard).upper()}")
        details.append(f"failsafe FALSE: {str(not failsafe).upper()}")
        details.append(
            "prestream offboard signal lost (observed, not failsafe): "
            + str(bool(self.flags.offboard_control_signal_lost)).upper()
        )
        details.append(
            "telemetry timestamps: "
            f"status={self.status.timestamp},mode={self.mode.timestamp},"
            f"odometry={self.odometry.timestamp},flags={self.flags.timestamp}"
        )
        ok = ok and not armed and not offboard and not failsafe
        return ok, details


def main(args=None) -> int:
    """Run a finite read-only SITL doctor and print concise evidence."""
    rclpy.init(args=args)
    node = Px4SitlDoctorNode()
    deadline = time.monotonic() + 10.0
    stable_readiness_count = 0
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.10)
            if node.readiness_snapshot():
                stable_readiness_count += 1
            else:
                stable_readiness_count = 0
            if stable_readiness_count >= 3:
                break
        passed, details = node.validate()
        for detail in details:
            print(detail)
        if passed:
            print("px4 SITL doctor passed: read-only safety contract verified")
            return 0
        print("px4 SITL doctor failed: streaming remains prohibited")
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
