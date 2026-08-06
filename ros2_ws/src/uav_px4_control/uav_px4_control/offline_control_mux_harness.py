"""Finite synthetic ROS nodes for Phase 6 mux integration graphs."""

import math
import time

from geometry_msgs.msg import TwistStamped

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import String

from uav_interfaces.msg import ControlMuxStatus, TrajectoryTrackingStatus
from uav_interfaces.srv import SetControlSource

from uav_px4_control.control_mux_node import (
    MUX_STATUS_TOPIC,
    SELECTED_COMMAND_TOPIC,
    SET_SOURCE_SERVICE,
    SOURCE_TOPIC,
    control_qos,
)
from uav_px4_control.control_source_models import (
    ASTAR_EXPERT,
    HOLD,
    HUMAN_JOYSTICK,
    NAVRL_POLICY,
    SOURCE_TOPICS,
)


TRACKING_STATUS_TOPIC = "/uav/control/astar_tracking_status"
SOURCE_DWELL_SETTLE_S = 0.35


class SyntheticCandidatePublisher(Node):
    """Publish one source only with configurable deterministic faults."""

    def __init__(self, source: str) -> None:
        """Create the one-topic synthetic source fixture."""
        node_name = source.lower() + "_synthetic_publisher"
        super().__init__(node_name)
        self.source = source
        self.declare_parameter("behavior", "normal")
        self.behavior = str(self.get_parameter("behavior").value)
        self._publisher = self.create_publisher(
            TwistStamped, SOURCE_TOPICS[source], control_qos()
        )
        self._started = time.monotonic()
        self._sequence = 0
        self._last_stamp = None
        self.create_timer(0.04, self._tick)
        self.get_logger().info(
            f"synthetic source={self.source}, behavior={self.behavior}"
        )

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._started
        behavior = self.behavior
        if behavior == "invalid-external-hold" and self._sequence > 0:
            return
        if behavior == "delayed" and elapsed < 0.60:
            return
        if behavior == "stale" and elapsed > 0.75:
            return
        if behavior == "shutdown" and elapsed > 0.80:
            return
        if behavior == "safety-recovery" and 0.75 < elapsed < 1.35:
            return
        message = TwistStamped()
        stamp = self.get_clock().now().to_msg()
        if behavior == "nonmonotonic" and self._last_stamp is not None:
            stamp = self._last_stamp
        message.header.stamp = stamp
        self._last_stamp = stamp
        message.header.frame_id = (
            "map" if behavior == "wrong-frame" else "px4_ned"
        )
        values = {
            ASTAR_EXPERT: (0.40, 0.00, 0.00, 0.10),
            HUMAN_JOYSTICK: (0.00, 0.35, 0.00, -0.10),
            NAVRL_POLICY: (-0.30, 0.00, 0.00, 0.05),
            HOLD: (0.00, 0.00, 0.00, 0.00),
        }
        north, east, down, yaw_rate = values[self.source]
        if behavior == "varying":
            north *= math.sin(self._sequence * 0.1)
            east += 0.1 * math.cos(self._sequence * 0.1)
        elif behavior == "nonfinite":
            north = math.nan
        elif behavior == "excessive":
            north = 2.10
        elif behavior == "invalid-external-hold":
            north = 0.10
        message.twist.linear.x = north
        message.twist.linear.y = east
        message.twist.linear.z = down
        message.twist.angular.z = yaw_rate
        self._publisher.publish(message)
        self._sequence += 1


class ControlMuxResultMonitor(Node):
    """Drive source service requests and independently validate outputs."""

    def __init__(self) -> None:
        """Subscribe to mux outputs and initialize the finite scenario."""
        super().__init__("control_mux_result_monitor")
        self.declare_parameter("mode", "normal")
        self.mode = str(self.get_parameter("mode").value)
        qos = control_qos()
        self.create_subscription(
            TwistStamped,
            SELECTED_COMMAND_TOPIC,
            self._command_callback,
            qos,
        )
        self.create_subscription(
            String, SOURCE_TOPIC, self._source_callback, qos
        )
        self.create_subscription(
            ControlMuxStatus,
            MUX_STATUS_TOPIC,
            self._status_callback,
            qos,
        )
        if self.mode == "control-stack":
            self.create_subscription(
                TrajectoryTrackingStatus,
                TRACKING_STATUS_TOPIC,
                self._tracking_callback,
                qos,
            )
        self._client = self.create_client(
            SetControlSource, SET_SOURCE_SERVICE
        )
        self._started = time.monotonic()
        self._stage_started = self._started
        self._stage = "WAIT_STARTUP"
        self._pending = None
        self._pending_source = ""
        self._status = None
        self._source = ""
        self._command = None
        self._tracking_state = ""
        self._startup_hold_seen = False
        self._stale_hold_seen = False
        self._latched_hold_seen = False
        self._barrier_count = 0
        self._movement_count = 0
        self._transition_max = 0
        self._last_state = ""
        self._contract_error = ""
        self._finished = False
        self.exit_code = 1
        self.create_timer(0.05, self._tick)

    @staticmethod
    def _magnitude(message: TwistStamped) -> float:
        return sum(abs(value) for value in (
            message.twist.linear.x,
            message.twist.linear.y,
            message.twist.linear.z,
            message.twist.angular.x,
            message.twist.angular.y,
            message.twist.angular.z,
        ))

    def _command_callback(self, message: TwistStamped) -> None:
        self._command = message
        values = (
            message.twist.linear.x,
            message.twist.linear.y,
            message.twist.linear.z,
            message.twist.angular.x,
            message.twist.angular.y,
            message.twist.angular.z,
        )
        if message.header.frame_id != "px4_ned":
            self._contract_error = "selected command frame is not px4_ned"
        elif not all(math.isfinite(value) for value in values):
            self._contract_error = "selected command contains non-finite data"
        elif values[3] != 0.0 or values[4] != 0.0:
            self._contract_error = "selected angular x/y are not zero"
        speed = math.sqrt(sum(value * value for value in values[:3]))
        if speed > 2.0 + 1e-6 or abs(values[5]) > 1.5 + 1e-6:
            self._contract_error = "selected command exceeded limits"
        if self._magnitude(message) > 1e-9:
            self._movement_count += 1

    def _source_callback(self, message: String) -> None:
        self._source = message.data

    def _status_callback(self, message: ControlMuxStatus) -> None:
        self._status = message
        state = message.status_message.split(":", 1)[0]
        if state != self._last_state:
            self.get_logger().info(
                f"observed mux state={state}, active={message.active_source}"
            )
            self._last_state = state
        self._transition_max = max(
            self._transition_max, message.transition_count
        )
        if message.hold_active:
            if not message.hold_reason:
                self._contract_error = "HOLD status lacks a reason"
            if (
                self._command is not None
                and self._magnitude(self._command) > 1e-9
            ):
                self._contract_error = "HOLD status accompanied nonzero output"
        if message.status_message.startswith("HOLD_STARTUP"):
            self._startup_hold_seen = True
        elif message.status_message.startswith("HOLD_STALE_SOURCE"):
            self._stale_hold_seen = True
        elif message.status_message.startswith("HOLD_LATCHED_FAULT"):
            self._latched_hold_seen = True
        if message.switch_in_progress:
            self._barrier_count += 1

    def _tracking_callback(self, message: TrajectoryTrackingStatus) -> None:
        self._tracking_state = message.state

    def _request(self, source: str) -> None:
        if self._pending is not None or not self._client.service_is_ready():
            return
        request = SetControlSource.Request()
        request.source = source
        self._pending_source = source
        self._pending = self._client.call_async(request)

    def _response_ready(self) -> bool:
        if self._pending is None or not self._pending.done():
            return False
        response = self._pending.result()
        source = self._pending_source
        self._pending = None
        self._pending_source = ""
        if response is None or not response.accepted:
            detail = (
                "no response" if response is None
                else response.status_message
            )
            self._contract_error = (
                f"service rejected required source {source}: {detail}"
            )
            return False
        return True

    def _set_stage(self, stage: str) -> None:
        self._stage = stage
        self._stage_started = time.monotonic()

    def _normal_tick(self) -> None:
        status = self._status
        if status is None:
            return
        if self._stage == "WAIT_STARTUP":
            if (
                self._startup_hold_seen
                and ASTAR_EXPERT in status.healthy_sources
            ):
                self._request(ASTAR_EXPERT)
                self._set_stage("REQUEST_ASTAR")
        elif self._stage == "REQUEST_ASTAR" and self._response_ready():
            self._set_stage("WAIT_ASTAR")
        elif self._stage == "WAIT_ASTAR":
            if status.active_source == ASTAR_EXPERT:
                self._set_stage("DWELL_ASTAR")
        elif self._stage == "DWELL_ASTAR":
            if time.monotonic() - self._stage_started > SOURCE_DWELL_SETTLE_S:
                self._request(HUMAN_JOYSTICK)
                self._set_stage("REQUEST_JOYSTICK")
        elif self._stage == "REQUEST_JOYSTICK" and self._response_ready():
            self._set_stage("WAIT_JOYSTICK")
        elif self._stage == "WAIT_JOYSTICK":
            if status.active_source == HUMAN_JOYSTICK:
                if self._barrier_count == 0:
                    self._contract_error = (
                        "joystick switch lacked HOLD barrier"
                    )
                self._set_stage("DWELL_JOYSTICK")
        elif self._stage == "DWELL_JOYSTICK":
            if time.monotonic() - self._stage_started > SOURCE_DWELL_SETTLE_S:
                self._request(NAVRL_POLICY)
                self._set_stage("REQUEST_NAVRL")
        elif self._stage == "REQUEST_NAVRL" and self._response_ready():
            self._set_stage("WAIT_NAVRL")
        elif self._stage == "WAIT_NAVRL":
            if status.active_source == NAVRL_POLICY:
                self._request(HOLD)
                self._set_stage("REQUEST_HOLD")
        elif self._stage == "REQUEST_HOLD" and self._response_ready():
            self._set_stage("WAIT_HOLD")
        elif self._stage == "WAIT_HOLD":
            if status.active_source == HOLD and status.hold_active:
                self._finish(0, "A* -> joystick -> NavRL -> HOLD complete")

    def _safety_tick(self) -> None:
        status = self._status
        if status is None:
            return
        if self._stage == "WAIT_STARTUP":
            if (
                self._startup_hold_seen
                and ASTAR_EXPERT in status.healthy_sources
            ):
                self._request(ASTAR_EXPERT)
                self._set_stage("REQUEST_ASTAR")
        elif self._stage == "REQUEST_ASTAR" and self._response_ready():
            self._set_stage("WAIT_ASTAR")
        elif self._stage == "WAIT_ASTAR":
            if status.active_source == ASTAR_EXPERT:
                self._set_stage("WAIT_STALE")
        elif self._stage == "WAIT_STALE":
            if self._stale_hold_seen:
                self._set_stage("WAIT_LATCH")
        elif self._stage == "WAIT_LATCH":
            if self._latched_hold_seen:
                self._set_stage("WAIT_FRESH_LATCHED")
        elif self._stage == "WAIT_FRESH_LATCHED":
            if (
                ASTAR_EXPERT in status.healthy_sources
                and time.monotonic() - self._stage_started > 0.25
            ):
                if status.active_source != HOLD:
                    self._contract_error = "fresh data bypassed fault latch"
                    return
                self._request(ASTAR_EXPERT)
                self._set_stage("REQUEST_RECOVERY")
        elif self._stage == "REQUEST_RECOVERY" and self._response_ready():
            self._set_stage("WAIT_RECOVERY")
        elif self._stage == "WAIT_RECOVERY":
            if status.active_source == ASTAR_EXPERT:
                self._finish(0, "stale latch and explicit recovery complete")

    def _control_stack_tick(self) -> None:
        status = self._status
        if status is None:
            return
        if self._stage == "WAIT_STARTUP":
            if ASTAR_EXPERT in status.healthy_sources:
                self._request(ASTAR_EXPERT)
                self._set_stage("REQUEST_ASTAR")
        elif self._stage == "REQUEST_ASTAR" and self._response_ready():
            self._set_stage("WAIT_TRACKING")
        elif self._stage == "WAIT_TRACKING":
            if (
                status.active_source == ASTAR_EXPERT
                and self._tracking_state == "GOAL_HOLD"
                and self._movement_count > 0
            ):
                self._finish(0, "follower -> mux -> plant reached GOAL_HOLD")

    def _tick(self) -> None:
        if self._finished:
            return
        if self._contract_error:
            self._finish(1, self._contract_error)
            return
        if self.mode == "normal":
            self._normal_tick()
        elif self.mode == "safety":
            self._safety_tick()
        elif self.mode == "control-stack":
            self._control_stack_tick()
        else:
            self._finish(1, f"unknown monitor mode: {self.mode}")
            return
        timeout = 20.0 if self.mode == "control-stack" else 8.0
        if time.monotonic() - self._started > timeout:
            self._finish(
                1, f"{self.mode} mux graph timed out at {self._stage}"
            )

    def _finish(self, code: int, detail: str) -> None:
        if self._finished:
            return
        topics = dict(self.get_topic_names_and_types())
        if any(name.startswith("/fmu/in/") for name in topics):
            code, detail = 1, "forbidden /fmu/in/* topic detected"
        publishers = self.get_publishers_info_by_topic(SELECTED_COMMAND_TOPIC)
        if len(publishers) != 1:
            code, detail = 1, "selected_command publisher ownership is not one"
        self._finished = True
        self.exit_code = code
        marker = (
            "control stack offline integration passed:"
            if self.mode == "control-stack"
            else "mux offline integration passed:"
        )
        summary = (
            f"{marker} mode={self.mode}, detail={detail}, "
            f"barrier_cycles={self._barrier_count}, "
            f"movement_cycles={self._movement_count}, "
            f"transitions={self._transition_max}, frame=px4_ned"
        )
        if code == 0:
            self.get_logger().info(summary)
        else:
            self.get_logger().error(f"mux integration failed: {detail}")
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


def _publisher_main(source: str, args=None) -> int:
    rclpy.init(args=args)
    return _spin(SyntheticCandidatePublisher(source))


def astar_publisher_main(args=None) -> int:
    """Run the synthetic A* candidate source."""
    return _publisher_main(ASTAR_EXPERT, args)


def joystick_publisher_main(args=None) -> int:
    """Run the synthetic joystick candidate source without hardware."""
    return _publisher_main(HUMAN_JOYSTICK, args)


def navrl_publisher_main(args=None) -> int:
    """Run the synthetic NavRL source without a model or runtime."""
    return _publisher_main(NAVRL_POLICY, args)


def hold_publisher_main(args=None) -> int:
    """Run the optional external HOLD safety fixture source."""
    return _publisher_main(HOLD, args)


def monitor_main(args=None) -> int:
    """Run the independent finite mux result monitor."""
    rclpy.init(args=args)
    return _spin(ControlMuxResultMonitor())
