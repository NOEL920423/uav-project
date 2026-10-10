"""Read a Linux USB joystick and publish existing human command candidates."""

import errno
import fcntl
import math
import os
import struct
import time
from dataclasses import dataclass

from geometry_msgs.msg import TwistStamped

from nav_msgs.msg import Odometry

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from uav_data_recorder.expert_dataset_contract import (
    ACTION_LIMITS,
    yaw_from_quaternion,
)

from uav_interfaces.msg import Px4FlightStatus

from uav_px4_control.control.control_mux_node import control_qos
from uav_px4_control.control.control_source_models import (
    HUMAN_JOYSTICK,
    SOURCE_TOPICS,
    VALID_COMMAND_FRAME,
)
from uav_px4_control.flight.px4_flight_models import (
    altitude_hold_down_velocity,
)


ODOMETRY_TOPIC = "/uav/vehicle/odometry"
JS_EVENT = struct.Struct("=IhBB")
JSIOCGAXES = 0x80016A11
JSIOCGBUTTONS = 0x80016A12
JSIOCGNAME = 0x80806A13


@dataclass(frozen=True)
class JoystickMapping:
    """Keep calibration and command conversion in this single node module."""

    axes: tuple = (1, 0, 3)
    signs: tuple = (1.0, 1.0, 1.0)
    centers: tuple = (0.0, 0.0, 0.0)
    scales: tuple = (1.0, 1.0, 1.0)
    deadzone: float = 0.05

    def __post_init__(self) -> None:
        """Reject ambiguous axes and invalid calibration before publishing."""
        if (
            len(self.axes) != 3 or len(set(self.axes)) != 3
            or any(type(axis) is not int or axis < 0 for axis in self.axes)
        ):
            raise ValueError("axis_indices require three distinct indices")
        for name in ("signs", "centers", "scales"):
            values = getattr(self, name)
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError(
                    f"axis_{name} must contain three finite values"
                )
        if any(value not in (-1.0, 1.0) for value in self.signs):
            raise ValueError("axis_signs must contain only -1.0 or 1.0")
        if any(abs(value) >= 1.0 for value in self.centers):
            raise ValueError("axis_centers must be inside (-1, 1)")
        if any(value <= 0.0 for value in self.scales):
            raise ValueError("axis_scales must be positive")
        if not math.isfinite(self.deadzone) or not 0 <= self.deadzone < 1:
            raise ValueError("deadzone must be in [0, 1)")

    def normalized(self, axes: dict) -> tuple:
        """Calibrate three movement axes with a continuous deadzone."""
        result = []
        for index, sign, center, scale in zip(
            self.axes, self.signs, self.centers, self.scales
        ):
            value = float(axes[index])
            if not math.isfinite(value):
                raise ValueError("joystick axis must be finite")
            value = max(-1.0, min(1.0, (value - center) * scale * sign))
            magnitude = max(0.0, abs(value) - self.deadzone)
            result.append(math.copysign(
                magnitude / (1.0 - self.deadzone), value
            ))
        return tuple(result)

    def command(self, axes: dict, yaw: float) -> tuple:
        """Convert calibrated body action into NED velocity and yaw rate."""
        if not math.isfinite(yaw):
            raise ValueError("NED yaw must be finite")
        forward, right, yaw_rate = (
            value * limit
            for value, limit in zip(self.normalized(axes), ACTION_LIMITS)
        )
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return (
            cosine * forward - sine * right,
            sine * forward + cosine * right,
            0.0,
            yaw_rate,
        )


class LinuxJoystick:
    """Drain nonblocking joydev events without another ROS node or thread."""

    def __init__(self, device: str) -> None:
        """Open one device and discover its axes and buttons."""
        self.fd = os.open(
            device, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC
        )
        self.axes = {}
        self.buttons = {}
        try:
            axes, buttons, name = bytearray(1), bytearray(1), bytearray(128)
            fcntl.ioctl(self.fd, JSIOCGAXES, axes)
            fcntl.ioctl(self.fd, JSIOCGBUTTONS, buttons)
            fcntl.ioctl(self.fd, JSIOCGNAME, name)
            self.axis_count, self.button_count = axes[0], buttons[0]
            self.name = name.split(b"\0", 1)[0].decode(errors="replace")
        except OSError:
            self.close()
            raise

    def read_pending(self) -> bool:
        """Retain unchanged sticks; report disconnects and initialization."""
        initialized = False
        for _ in range(256):
            try:
                data = os.read(self.fd, JS_EVENT.size)
            except BlockingIOError:
                return initialized
            if len(data) != JS_EVENT.size:
                raise OSError(errno.EIO, "joystick EOF or incomplete event")
            _, value, event_type, index = JS_EVENT.unpack(data)
            initialized = initialized or bool(event_type & 0x80)
            kind = event_type & 0x7F
            if kind == 2 and index < self.axis_count:
                self.axes[index] = max(-1.0, min(1.0, value / 32767.0))
            elif kind == 1 and index < self.button_count and value in (0, 1):
                self.buttons[index] = bool(value)
            else:
                raise OSError(errno.EIO, "invalid joystick event")
        raise OSError(errno.EOVERFLOW, "joystick event queue did not drain")

    def close(self) -> None:
        """Release the device exactly once."""
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class JoystickControlNode(Node):
    """Publish human candidates after explicit enable and neutral input."""

    def __init__(self) -> None:
        """Start in inspection mode and reuse the NED odometry topic."""
        super().__init__("joystick_control")
        defaults = {
            "device": "/dev/input/js0",
            "enable_control": False,
            "axis_indices": [1, 0, 3],
            "axis_signs": [1.0, 1.0, 1.0],
            "axis_centers": [0.0, 0.0, 0.0],
            "axis_scales": [1.0, 1.0, 1.0],
            "deadzone": 0.05,
            "deadman_axis": 4,
            "deadman_button": -1,
            "deadman_threshold": 0.5,
            "publish_rate_hz": 50.0,
            "odometry_timeout_s": 0.20,
            "altitude_hold_m": 0.0,
            "latch_after_stop": False,
            "report_raw_inputs": True,
            "show_operator_status": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        values = {name: self.get_parameter(name).value for name in defaults}
        self.mapping = JoystickMapping(
            tuple(values["axis_indices"]), tuple(values["axis_signs"]),
            tuple(values["axis_centers"]), tuple(values["axis_scales"]),
            values["deadzone"],
        )
        self.enabled = values["enable_control"]
        self.deadman_axis = values["deadman_axis"]
        self.deadman_button = values["deadman_button"]
        self.deadman_threshold = values["deadman_threshold"]
        rate = values["publish_rate_hz"]
        self.odometry_timeout_s = values["odometry_timeout_s"]
        self.altitude_hold_m = values["altitude_hold_m"]
        self.latch_after_stop = values["latch_after_stop"]
        self.report_raw_inputs = values["report_raw_inputs"]
        self.show_operator_status = values["show_operator_status"]
        self._flight_status = None
        self._flight_status_receipt_s = None
        self._reported_operator_status = None
        self._input_stop_latched = False
        if not math.isfinite(self.altitude_hold_m) or self.altitude_hold_m < 0:
            raise ValueError("altitude_hold_m must be finite and nonnegative")
        self._ground_down_m = None
        self._current_down_m = None
        if not math.isfinite(rate) or not 20.0 <= rate <= 100.0:
            raise ValueError("publish_rate_hz must be between 20 and 100")
        if (
            not math.isfinite(self.odometry_timeout_s)
            or not 0.0 < self.odometry_timeout_s <= 0.25
        ):
            raise ValueError("odometry_timeout_s must be in (0, 0.25]")
        if (
            self.deadman_axis < -1 or self.deadman_button < -1
            or (self.deadman_axis >= 0) == (self.deadman_button >= 0)
        ):
            raise ValueError("configure exactly one deadman axis or button")
        if (
            not math.isfinite(self.deadman_threshold)
            or not -1.0 < self.deadman_threshold < 1.0
        ):
            raise ValueError("deadman_threshold must be inside (-1, 1)")
        if self.deadman_axis in self.mapping.axes:
            raise ValueError("deadman axis must differ from movement axes")
        self.reader = LinuxJoystick(values["device"])
        try:
            if self.enabled and (
                max(self.mapping.axes) >= self.reader.axis_count
                or self.deadman_axis >= self.reader.axis_count
                or self.deadman_button >= self.reader.button_count
            ):
                raise ValueError("configured input index is absent on device")
        except ValueError:
            self.reader.close()
            raise
        self._yaw = None
        self._odometry_receipt_s = None
        self._released = False
        self._active = False
        self._last_tick_s = None
        self._last_report_s = -math.inf
        self._reported_axes = None
        self._reported_buttons = {}
        self._reported_reason = None
        self._reason = "inspection only"
        self._publisher = self.create_publisher(
            TwistStamped, SOURCE_TOPICS[HUMAN_JOYSTICK], control_qos()
        )
        self.create_subscription(
            Odometry, ODOMETRY_TOPIC, self._odometry_callback, control_qos()
        )
        if self.show_operator_status:
            self.create_subscription(
                Px4FlightStatus, "/uav/px4/flight_status",
                self._flight_status_callback, control_qos(),
            )
        self._timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"USB joystick {self.reader.name!r}: "
            f"axes={self.reader.axis_count} "
            f"buttons={self.reader.button_count}; "
            f"enable_control={self.enabled}. "
            "Axis indices are device-specific."
        )
        self.get_logger().warning(
            "Candidate input only; this node does not select the mux, arm, "
            "take off, or publish PX4 inputs. "
            "Restart after changing parameters."
        )

    def _odometry_callback(self, message: Odometry) -> None:
        orientation = message.pose.pose.orientation
        try:
            if message.header.frame_id != VALID_COMMAND_FRAME:
                raise ValueError("odometry frame must be px4_ned")
            self._yaw = yaw_from_quaternion(
                orientation.x, orientation.y, orientation.z, orientation.w
            )
            down = float(message.pose.pose.position.z)
            if not math.isfinite(down):
                raise ValueError("NED down position must be finite")
            self._current_down_m = down
            if self._ground_down_m is None:
                self._ground_down_m = down
            self._odometry_receipt_s = time.monotonic()
        except ValueError as error:
            self._yaw = None
            self._odometry_receipt_s = None
            self._stop(str(error))

    def _deadman(self):
        """Return None until the configured switch has been observed."""
        if self.deadman_axis >= 0:
            value = self.reader.axes.get(self.deadman_axis)
            return None if value is None else value > self.deadman_threshold
        return self.reader.buttons.get(self.deadman_button)

    def _stop(self, reason: str) -> None:
        """Send one zero then stop heartbeats so the mux latches HOLD."""
        if self._active:
            if self.latch_after_stop:
                self._input_stop_latched = True
            self._publish((0.0, 0.0, 0.0, 0.0))
            self.get_logger().warning(
                f"Human input stopped: {reason}; mux timeout selects HOLD. "
                "Resuming requires an explicit healthy mux source request."
            )
        self._active = False
        self._released = False
        self._reason = reason

    def _publish(self, command: tuple) -> None:
        """Stamp a candidate on the same ROS clock used by the mux."""
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = VALID_COMMAND_FRAME
        (
            message.twist.linear.x, message.twist.linear.y,
            message.twist.linear.z, message.twist.angular.z,
        ) = command
        self._publisher.publish(message)

    def _tick(self) -> None:
        now = time.monotonic()
        if self.reader.fd is None:
            if self.show_operator_status and now - self._last_report_s >= 1.0:
                self._last_report_s = now
                self._report_inputs()
            return
        try:
            if self.reader.read_pending():
                self._stop("joystick initialized or resynchronized")
        except OSError as error:
            self._stop("USB device read failed")
            self.reader.close()
            self.get_logger().error(
                f"USB joystick read failed: {error!r}. "
                "Repair the device connection and restart this node."
            )
            return
        if self._last_tick_s is not None and now - self._last_tick_s > 0.20:
            self._stop("joystick publish loop exceeded 0.20 s")
        self._last_tick_s = now
        if self.enabled:
            self._update_control(now)
        if now - self._last_report_s >= 1.0:
            self._last_report_s = now
            self._report_inputs()

    def _report_inputs(self) -> None:
        """Show an initial axis table, then only compact input changes."""
        reason = self._reason if self.enabled else "inspection only"
        if self.show_operator_status:
            self._report_operator_status(reason)
            return
        if not self.report_raw_inputs:
            if reason != self._reported_reason:
                self.get_logger().info(f"Control: {reason}")
            self._reported_reason = reason
            return
        axes = {
            index: round(value, 2)
            for index, value in sorted(self.reader.axes.items())
        }
        buttons = self.reader.buttons
        if self._reported_axes is None:
            if not axes:
                return
            rows = [
                "USB input inspection" if not self.enabled
                else "USB joystick inputs",
                "  Axis   Value",
                "  ----   -----",
                *(f"  A{index:<3}   {value:+.2f}"
                  for index, value in axes.items()),
                "Move one stick or switch at a time; watch its A-number.",
                "Values: -1.00 minimum, 0.00 center, +1.00 maximum.",
            ]
            pressed = [f"B{index}" for index in sorted(buttons)
                       if buttons[index]]
            rows.append("Pressed buttons: " + (" ".join(pressed) or "none"))
            self.get_logger().info("\n" + "\n".join(rows))
        else:
            changed = [
                f"A{index}={value:+.2f}"
                for index, value in axes.items()
                if value != self._reported_axes.get(index)
            ]
            changed.extend(
                f"B{index}={'pressed' if pressed else 'released'}"
                for index, pressed in sorted(buttons.items())
                if pressed != self._reported_buttons.get(index, False)
            )
            if changed:
                self.get_logger().info("Changed: " + "  ".join(changed))
        if reason != self._reported_reason:
            self.get_logger().info(f"Control: {reason}")
        self._reported_axes = axes
        self._reported_buttons = dict(buttons)
        self._reported_reason = reason

    def _flight_status_callback(self, message: Px4FlightStatus) -> None:
        """Observe the supervisor without requesting flight actions."""
        self._flight_status = message
        self._flight_status_receipt_s = time.monotonic()

    def _report_operator_status(self, reason: str) -> None:
        """Show readiness and the next action only when they change."""
        now = time.monotonic()
        pose_ready = bool(
            self._yaw is not None and self._odometry_receipt_s is not None
            and 0.0 <= now - self._odometry_receipt_s
            <= self.odometry_timeout_s
        )
        status = self._flight_status
        supervisor_ready = bool(
            status is not None and self._flight_status_receipt_s is not None
            and 0.0 <= now - self._flight_status_receipt_s <= 1.0
        )
        stage = status.state if supervisor_ready else "WAIT_SUPERVISOR"
        scene_ready = bool(
            supervisor_ready
            and "environment_valid=true" in status.status_message
        )
        switch = self._deadman()
        sa = "ON (+1)" if switch else "OFF (-1)"
        if switch is None:
            sa = "NOT READ"
        next_action = {
            "ready": "Move SA to +1 while keeping both sticks centered.",
            "center sticks": "Center the right stick and left yaw stick.",
            "release deadman and center sticks first": (
                "Move SA to -1 first, and center both sticks."
            ),
            "center sticks before enabling human input": (
                "Move SA to -1, center sticks, then move SA to +1."
            ),
        }.get(reason, reason)
        if self.reader.fd is None:
            next_action = "Reconnect USB/IP, then restart this flight launch."
        elif stage == "LANDING":
            next_action = "Wait for landing/disarm. Keep the launch running."
        elif self._input_stop_latched:
            next_action = "After landing/disarm, restart the flight launch."
        elif not pose_ready:
            next_action = (
                "Wait for PX4 position data; check the Isaac runtime."
            )
        elif not supervisor_ready:
            next_action = "Wait for the flight supervisor to start."
        elif not scene_ready:
            next_action = "Wait for the Isaac scene/runtime to become ready."
        elif stage == "HUMAN_CONTROL":
            next_action = "Fly with the sticks. Move SA to -1 to land."
        elif stage not in {"DISABLED", "COMPLETE", "FAILED"}:
            next_action = "Keep sticks centered until PILOT CONTROL appears."
        elif self._active:
            if (
                status.vehicle_armed or status.offboard_active
                or not status.landed
            ):
                next_action = "Wait for landed/disarmed PX4 state."
            elif status.failure_reason:
                next_action = "Check FAILURE below before restarting."
            else:
                next_action = "READY: send the takeoff request in terminal 3."
        flight = {
            "DISABLED": "NOT STARTED", "TAKEOFF": "TAKING OFF",
            "SELECTING_HUMAN": "HANDING OVER",
            "HUMAN_CONTROL": "PILOT CONTROL", "LANDING": "LANDING",
            "COMPLETE": "LANDED", "FAILED": "STOPPED",
        }.get(stage, "STARTING" if supervisor_ready else "WAITING")
        failure = status.failure_reason if supervisor_ready else ""
        snapshot = (
            self.reader.fd is not None, pose_ready, scene_ready, sa,
            flight, next_action, failure,
        )
        if snapshot == self._reported_operator_status:
            return
        self._reported_operator_status = snapshot
        lines = [
            "", "========== FLIGHT CONTROL ==========",
            f"  USB       : {'CONNECTED' if snapshot[0] else 'DISCONNECTED'}",
            f"  PX4 pose  : {'READY' if pose_ready else 'WAITING'}",
            f"  Isaac     : {'READY' if scene_ready else 'WAITING'}",
            f"  SA switch : {sa}",
            f"  Flight    : {flight}",
            f"  NEXT      : {next_action}",
        ]
        if failure:
            lines.append(f"  FAILURE   : {failure}")
        lines.append("====================================")
        self.get_logger().info("\n".join(lines))

    def _update_control(self, now: float) -> None:
        if self._input_stop_latched:
            self._reason = (
                "flight input stopped; restart the human flight launch"
            )
            return
        if (
            self._yaw is None or self._odometry_receipt_s is None
            or now - self._odometry_receipt_s > self.odometry_timeout_s
        ):
            self._stop("waiting for fresh valid NED odometry")
            return
        try:
            normalized = self.mapping.normalized(self.reader.axes)
            deadman = self._deadman()
            if deadman is None:
                self._stop("waiting for deadman input")
            elif not deadman:
                self._stop("deadman released")
                self._released = all(value == 0.0 for value in normalized)
                self._reason = "ready" if self._released else "center sticks"
            elif not self._released:
                self._reason = "release deadman and center sticks first"
            elif not self._active and any(
                value != 0.0 for value in normalized
            ):
                self._stop("center sticks before enabling human input")
            else:
                command = self.mapping.command(
                    self.reader.axes, self._yaw
                )
                if self.altitude_hold_m > 0.0:
                    down_speed = altitude_hold_down_velocity(
                        self._ground_down_m - self.altitude_hold_m,
                        self._current_down_m,
                    )
                    command = (command[0], command[1], down_speed, command[3])
                self._publish(command)
                self._active = True
                self._reason = "publishing HUMAN_JOYSTICK candidates"
        except (KeyError, ValueError) as error:
            self._stop(f"invalid or uninitialized movement axis: {error}")

    def destroy_node(self):
        """Close input and stop candidates before destroying ROS resources."""
        if rclpy.ok():
            self._stop("node shutdown")
        self.reader.close()
        return super().destroy_node()


def main(args=None) -> None:
    """Run USB input with visible startup exceptions and clean shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = JoystickControlNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
