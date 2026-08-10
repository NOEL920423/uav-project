"""Finite SITL monitor driving only repository-local enable services."""

import importlib
import math
import time

import rclpy
from rclpy.node import Node

from uav_interfaces.msg import (
    ControlMuxStatus,
    Px4OutputGateStatus,
    Px4StreamStatus,
)
from uav_interfaces.srv import (
    SetControlSource,
    SetPx4OutputEnable,
    SetPx4StreamEnable,
)

from uav_px4_control.control_mux_node import (
    MUX_STATUS_TOPIC,
    SET_SOURCE_SERVICE,
    control_qos,
)
from uav_px4_control.control_source_models import ASTAR_EXPERT
from uav_px4_control.px4_mapping_gate_node import (
    GATE_STATUS_TOPIC,
    SET_OUTPUT_ENABLE_SERVICE,
)
from uav_px4_control.px4_setpoint_streamer_node import (
    OFFBOARD_CONTROL_MODE_TOPIC,
    SET_STREAM_ENABLE_SERVICE,
    STREAM_STATUS_TOPIC,
    TRAJECTORY_SETPOINT_TOPIC,
    VEHICLE_CONTROL_MODE_TOPIC,
    VEHICLE_STATUS_TOPIC,
    px4_input_qos,
    px4_output_qos,
)


EXPECTED_FIXTURES = {
    "zero": (0.0, 0.0, 0.0, 0.0),
    "north-0.10": (0.10, 0.0, 0.0, 0.0),
    "east-0.10": (0.0, 0.10, 0.0, 0.0),
    "down-0.10": (0.0, 0.0, 0.10, 0.0),
    "yaw-rate-0.10": (0.0, 0.0, 0.0, 0.10),
}


class Px4SitlStreamMonitor(Node):
    """Verify a disabled-to-prestream-to-disabled SITL-only sequence."""

    def __init__(self) -> None:
        """Create read-only observations and three local service clients."""
        super().__init__("px4_sitl_stream_monitor")
        self.declare_parameter("fixture", "zero")
        self.fixture = str(self.get_parameter("fixture").value)
        if self.fixture not in EXPECTED_FIXTURES:
            raise ValueError(f"unsupported live fixture: {self.fixture}")
        message_module = importlib.import_module("px4_msgs.msg")
        qos = control_qos()
        self.create_subscription(
            ControlMuxStatus,
            MUX_STATUS_TOPIC,
            self._mux_callback,
            qos,
        )
        self.create_subscription(
            Px4OutputGateStatus,
            GATE_STATUS_TOPIC,
            self._gate_callback,
            qos,
        )
        self.create_subscription(
            Px4StreamStatus,
            STREAM_STATUS_TOPIC,
            self._stream_callback,
            qos,
        )
        self.create_subscription(
            message_module.TrajectorySetpoint,
            TRAJECTORY_SETPOINT_TOPIC,
            self._trajectory_callback,
            px4_input_qos(),
        )
        self.create_subscription(
            message_module.OffboardControlMode,
            OFFBOARD_CONTROL_MODE_TOPIC,
            self._mode_callback,
            px4_input_qos(),
        )
        self.create_subscription(
            message_module.VehicleStatus,
            VEHICLE_STATUS_TOPIC,
            self._vehicle_status_callback,
            px4_output_qos(),
        )
        self.create_subscription(
            message_module.VehicleControlMode,
            VEHICLE_CONTROL_MODE_TOPIC,
            self._vehicle_mode_callback,
            px4_output_qos(),
        )
        self._mux_client = self.create_client(
            SetControlSource,
            SET_SOURCE_SERVICE,
        )
        self._gate_client = self.create_client(
            SetPx4OutputEnable,
            SET_OUTPUT_ENABLE_SERVICE,
        )
        self._stream_client = self.create_client(
            SetPx4StreamEnable,
            SET_STREAM_ENABLE_SERVICE,
        )
        self._mux = None
        self._gate = None
        self._gate_ready_heartbeat_count = 0
        self._stream = None
        self._vehicle_status = None
        self._vehicle_mode = None
        self._trajectory_count = 0
        self._mode_count = 0
        self._trajectory_timestamps: list[int] = []
        self._trajectory_receipts: list[float] = []
        self._mapping_error = ""
        self._stage = "WAIT_READY"
        self._stage_started = time.monotonic()
        self._pending = None
        self._stop_baseline: tuple[int, int] | None = None
        self._finished = False
        self.exit_code = 1
        self.create_timer(0.05, self._tick)

    def _mux_callback(self, message) -> None:
        self._mux = message

    def _gate_callback(self, message) -> None:
        self._gate = message
        if (
            message.state == "READY_DISABLED"
            and not message.enable_requested
            and not message.safe_to_forward
            and message.selected_command_valid
            and message.mux_valid
            and message.telemetry_valid
        ):
            self._gate_ready_heartbeat_count += 1
        else:
            self._gate_ready_heartbeat_count = 0

    def _stream_callback(self, message) -> None:
        self._stream = message
        if message.vehicle_armed:
            self._mapping_error = "stream status reported unexpected armed"
        elif message.offboard_active:
            self._mapping_error = "stream status reported unexpected OFFBOARD"
        elif message.failsafe:
            self._mapping_error = "stream status reported failsafe"

    def _vehicle_status_callback(self, message) -> None:
        self._vehicle_status = message

    def _vehicle_mode_callback(self, message) -> None:
        self._vehicle_mode = message

    def _trajectory_callback(self, message) -> None:
        expected = EXPECTED_FIXTURES[self.fixture]
        actual = (*message.velocity, message.yawspeed)
        if any(
            abs(left - right) > 1e-6
            for left, right in zip(actual, expected)
        ):
            self._mapping_error = f"trajectory mapping mismatch: {actual}"
        unused = (*message.position, *message.acceleration, *message.jerk)
        if not all(math.isnan(float(value)) for value in unused):
            self._mapping_error = "an unused vector field was not NaN"
        if not math.isnan(float(message.yaw)):
            self._mapping_error = "absolute yaw was unexpectedly populated"
        timestamp = int(message.timestamp)
        if (
            self._trajectory_timestamps
            and timestamp <= self._trajectory_timestamps[-1]
        ):
            self._mapping_error = "trajectory timestamp was non-monotonic"
        self._trajectory_timestamps.append(timestamp)
        self._trajectory_receipts.append(time.monotonic())
        self._trajectory_count += 1

    def _mode_callback(self, message) -> None:
        if not (
            message.velocity
            and not message.position
            and not message.acceleration
            and not message.attitude
            and not message.body_rate
            and not message.actuator
        ):
            self._mapping_error = "OffboardControlMode was not velocity-only"
        self._mode_count += 1

    def _set_stage(self, stage: str) -> None:
        self._stage = stage
        self._stage_started = time.monotonic()
        self.get_logger().info(f"live stream monitor stage={stage}")

    def _call(self, client, request, next_stage: str) -> bool:
        if self._pending is not None or not client.service_is_ready():
            return False
        self._pending = client.call_async(request)
        self._set_stage(next_stage)
        return True

    def _response(self) -> bool | None:
        if self._pending is None or not self._pending.done():
            return None
        response = self._pending.result()
        self._pending = None
        if response is None or not response.accepted:
            detail = (
                "no response"
                if response is None
                else response.status_message
            )
            self._finish(1, f"service rejected live test step: {detail}")
            return False
        return True

    def _tick(self) -> None:
        if self._finished:
            return
        if self._mapping_error:
            self._finish(1, self._mapping_error)
            return
        if time.monotonic() - self._stage_started > 12.0:
            self._finish(1, f"live stream timed out at {self._stage}")
            return
        if self._stage == "WAIT_READY":
            if (
                self._mux is not None
                and ASTAR_EXPERT in self._mux.healthy_sources
                and self._stream is not None
                and self._stream.state == "STREAM_DISABLED"
            ):
                request = SetControlSource.Request()
                request.source = ASTAR_EXPERT
                self._call(self._mux_client, request, "REQUEST_MUX")
        elif self._stage == "REQUEST_MUX" and self._response():
            self._set_stage("WAIT_MUX")
        elif self._stage == "WAIT_MUX":
            if (
                self._mux is not None
                and self._mux.active_source == ASTAR_EXPERT
                and self._gate_ready_heartbeat_count >= 3
            ):
                request = SetPx4OutputEnable.Request()
                request.enable = True
                self._call(self._gate_client, request, "REQUEST_GATE")
        elif self._stage == "REQUEST_GATE" and self._response():
            self._set_stage("WAIT_GATE")
        elif self._stage == "WAIT_GATE":
            if self._gate is not None and self._gate.safe_to_forward:
                if self._trajectory_count or self._mode_count:
                    self._finish(
                        1,
                        "streamer published before explicit enable",
                    )
                    return
                request = SetPx4StreamEnable.Request()
                request.enable = True
                self._call(self._stream_client, request, "REQUEST_STREAM")
        elif self._stage == "REQUEST_STREAM" and self._response():
            self._set_stage("WAIT_STREAM")
        elif self._stage == "WAIT_STREAM":
            if (
                self._stream is not None
                and self._stream.state == "STREAMING"
                and self._trajectory_count >= 40
                and self._mode_count >= 40
            ):
                request = SetPx4StreamEnable.Request()
                request.enable = False
                self._call(self._stream_client, request, "REQUEST_DISABLE")
        elif self._stage == "REQUEST_DISABLE" and self._response():
            self._set_stage("SETTLE_STOP")
        elif self._stage == "SETTLE_STOP":
            if time.monotonic() - self._stage_started >= 0.10:
                self._stop_baseline = (
                    self._trajectory_count,
                    self._mode_count,
                )
                self._set_stage("VERIFY_STOP")
        elif self._stage == "VERIFY_STOP":
            if time.monotonic() - self._stage_started >= 0.40:
                stopped = self._stop_baseline == (
                    self._trajectory_count,
                    self._mode_count,
                )
                if not stopped:
                    self._finish(1, "publication continued after disable")
                else:
                    self._finish(0, "disabled -> zero prestream -> disabled")

    def _vehicle_safe(self) -> bool:
        if self._vehicle_status is None or self._vehicle_mode is None:
            return False
        return bool(
            self._vehicle_status.arming_state
            != self._vehicle_status.ARMING_STATE_ARMED
            and not self._vehicle_mode.flag_armed
            and self._vehicle_status.nav_state
            != self._vehicle_status.NAVIGATION_STATE_OFFBOARD
            and not self._vehicle_mode.flag_control_offboard_enabled
            and not self._vehicle_status.failsafe
        )

    def _finish(self, code: int, detail: str) -> None:
        if self._finished:
            return
        trajectory_publishers = self.get_publishers_info_by_topic(
            TRAJECTORY_SETPOINT_TOPIC
        )
        mode_publishers = self.get_publishers_info_by_topic(
            OFFBOARD_CONTROL_MODE_TOPIC
        )
        command_publishers = self.get_publishers_info_by_topic(
            "/fmu/in/vehicle_command"
        )
        if len(trajectory_publishers) != 1 or len(mode_publishers) != 1:
            code, detail = 1, "live PX4 input publisher ownership is not one"
        elif command_publishers:
            code, detail = 1, "forbidden VehicleCommand publisher exists"
        elif not self._vehicle_safe():
            code = 1
            detail = "vehicle did not remain disarmed and non-OFFBOARD"
        intervals = [
            right - left
            for left, right in zip(
                self._trajectory_receipts,
                self._trajectory_receipts[1:],
            )
        ]
        minimum_gap = min(intervals, default=0.0)
        maximum_gap = max(intervals, default=0.0)
        nominal_interval = 0.05
        rms_jitter = math.sqrt(
            sum(
                (interval - nominal_interval) ** 2
                for interval in intervals
            )
            / len(intervals)
        ) if intervals else 0.0
        duration = (
            self._trajectory_receipts[-1] - self._trajectory_receipts[0]
            if len(self._trajectory_receipts) >= 2
            else 0.0
        )
        rate = (
            (len(self._trajectory_receipts) - 1) / duration
            if duration > 0.0
            else 0.0
        )
        print(
            "PX4 SITL STREAM EVIDENCE "
            f"fixture={self.fixture} "
            f"trajectory_count={self._trajectory_count} "
            f"mode_count={self._mode_count} rate_hz={rate:.6f} "
            f"minimum_interval_s={minimum_gap:.6f} "
            f"maximum_gap_s={maximum_gap:.6f} "
            f"rms_jitter_s={rms_jitter:.6f} "
            f"first_timestamp={self._first_timestamp()} "
            f"last_timestamp={self._last_timestamp()}"
        )
        self._finished = True
        self.exit_code = code
        if code == 0:
            print(f"px4 SITL stream integration passed: {detail}")
        else:
            print(f"px4 SITL stream integration failed: {detail}")
        rclpy.shutdown()

    def _first_timestamp(self) -> int:
        if not self._trajectory_timestamps:
            return 0
        return self._trajectory_timestamps[0]

    def _last_timestamp(self) -> int:
        if not self._trajectory_timestamps:
            return 0
        return self._trajectory_timestamps[-1]


def main(args=None) -> int:
    """Run the finite live SITL stream monitor."""
    rclpy.init(args=args)
    node = Px4SitlStreamMonitor()
    try:
        rclpy.spin(node)
    finally:
        exit_code = node.exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
