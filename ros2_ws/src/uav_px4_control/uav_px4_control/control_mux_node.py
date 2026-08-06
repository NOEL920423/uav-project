"""ROS 2 adapter for the offline Phase 6 control-source multiplexer."""

from geometry_msgs.msg import TwistStamped

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from std_msgs.msg import String

from uav_interfaces.msg import ControlMuxStatus
from uav_interfaces.srv import SetControlSource

from uav_px4_control.control_mux import ControlSourceMux
from uav_px4_control.control_source_models import (
    ControlCommand,
    ControlMuxConfig,
    ControlMuxResult,
    SOURCE_TOPICS,
    Vector3,
    command_speed,
)


SELECTED_COMMAND_TOPIC = "/uav/control/selected_command"
SOURCE_TOPIC = "/uav/control/source"
MUX_STATUS_TOPIC = "/uav/control/mux_status"
SET_SOURCE_SERVICE = "/uav/control/set_source"


def control_qos() -> QoSProfile:
    """Use reliable volatile keep-last QoS for live control contracts."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


class ControlMuxNode(Node):
    """Own the selected ROS command and expose deterministic arbitration."""

    def __init__(self) -> None:
        """Declare parameters, four candidates, service, and output timer."""
        super().__init__("control_mux")
        defaults = ControlMuxConfig()
        for name in defaults.__dataclass_fields__:
            self.declare_parameter(name, getattr(defaults, name))
        self.config = ControlMuxConfig(**{
            name: self.get_parameter(name).value
            for name in defaults.__dataclass_fields__
        })
        self.mux = ControlSourceMux(self.config)
        qos = control_qos()
        self._selected_publisher = self.create_publisher(
            TwistStamped, SELECTED_COMMAND_TOPIC, qos
        )
        self._source_publisher = self.create_publisher(
            String, SOURCE_TOPIC, qos
        )
        self._status_publisher = self.create_publisher(
            ControlMuxStatus, MUX_STATUS_TOPIC, qos
        )
        self._candidate_subscriptions = []
        for source, topic in SOURCE_TOPICS.items():
            callback = self._candidate_callback(source)
            self._candidate_subscriptions.append(self.create_subscription(
                TwistStamped, topic, callback, qos
            ))
        self._service = self.create_service(
            SetControlSource, SET_SOURCE_SERVICE, self._set_source_callback
        )
        self._timer = self.create_timer(
            1.0 / self.config.publish_rate_hz, self._tick
        )
        self.last_result: ControlMuxResult | None = None
        self.get_logger().warning(
            "Phase 6 offline mux active; selected_command is not PX4 output"
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _candidate_callback(self, source: str):
        def callback(message: TwistStamped) -> None:
            command = ControlCommand(
                source=source,
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
            try:
                record = self.mux.accept_candidate(
                    source, command, self._now_seconds()
                )
            except (TypeError, ValueError, OverflowError) as error:
                self.get_logger().error(
                    f"candidate callback failed for {source}: {error}"
                )
                return
            if not record.valid:
                self.get_logger().warning(
                    f"unhealthy {source} candidate: {record.reason}"
                )
        return callback

    def _set_source_callback(self, request, response):
        try:
            result = self.mux.request_source(
                request.source, self._now_seconds()
            )
        except (TypeError, ValueError, OverflowError) as error:
            response.accepted = False
            response.requested_source = request.source
            response.active_source = self.mux.active_source
            response.status_message = f"selection request failed: {error}"
            return response
        response.accepted = result.accepted
        response.requested_source = result.requested_source
        response.active_source = result.active_source
        response.status_message = result.status_message
        return response

    def _tick(self) -> None:
        now = self._now_seconds()
        try:
            result = self.mux.step(now)
        except (TypeError, ValueError, OverflowError, RuntimeError) as error:
            self.get_logger().error(f"mux cycle failed closed: {error}")
            selection = self.mux.request_source("HOLD", now)
            if not selection.accepted:
                self.get_logger().fatal("internal HOLD request was rejected")
                return
            result = self.mux.step(now)
        self.last_result = result
        stamp = self.get_clock().now().to_msg()
        self._selected_publisher.publish(
            self._command_message(result, stamp)
        )
        self._source_publisher.publish(String(data=result.active_source))
        self._status_publisher.publish(self._status_message(result, stamp))

    @staticmethod
    def _command_message(result: ControlMuxResult, stamp) -> TwistStamped:
        command = result.selected_command
        message = TwistStamped()
        message.header.stamp = stamp
        message.header.frame_id = command.frame_id
        message.twist.linear.x = command.linear.x
        message.twist.linear.y = command.linear.y
        message.twist.linear.z = command.linear.z
        message.twist.angular.x = command.angular_x
        message.twist.angular.y = command.angular_y
        message.twist.angular.z = command.yaw_rate_radps
        return message

    @staticmethod
    def _status_message(result: ControlMuxResult, stamp) -> ControlMuxStatus:
        message = ControlMuxStatus()
        message.header.stamp = stamp
        message.header.frame_id = "px4_ned"
        message.requested_source = result.requested_source
        message.active_source = result.active_source
        message.selected_command_valid = result.selected_command_valid
        message.hold_active = result.hold_active
        message.hold_reason = result.hold_reason
        message.switch_in_progress = result.switch_in_progress
        message.switch_remaining_time = result.switch_remaining_time_s
        message.selected_source_age = result.selected_source_age_s
        message.selected_linear_speed = command_speed(
            result.selected_command
        )
        message.selected_yaw_rate = result.selected_command.yaw_rate_radps
        message.transition_count = result.transition_count
        message.healthy_sources = list(result.healthy_sources)
        message.stale_sources = list(result.stale_sources)
        message.status_message = (
            f"{result.status_message}|fault_latched="
            f"{str(result.fault_latched).lower()}|"
            f"diagnostics={len(result.diagnostics)}"
        )
        return message

    def destroy_node(self):
        """Publish one final exact-zero selected HOLD before shutdown."""
        if hasattr(self, "_selected_publisher") and rclpy.ok():
            now = self._now_seconds()
            self.mux.request_source("HOLD", now)
            result = self.mux.step(now + 1.0 / self.config.publish_rate_hz)
            stamp = self.get_clock().now().to_msg()
            self._selected_publisher.publish(
                self._command_message(result, stamp)
            )
        return super().destroy_node()


def main(args=None) -> int:
    """Run the offline selected-command owner."""
    rclpy.init(args=args)
    node = ControlMuxNode()
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
