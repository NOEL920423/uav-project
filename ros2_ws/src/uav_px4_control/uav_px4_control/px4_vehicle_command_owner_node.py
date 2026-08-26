"""Sole validated publisher for PX4 mode, arm, and land commands."""

import importlib
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from uav_interfaces.srv import SendPx4VehicleCommand

from uav_px4_control.px4_setpoint_streamer_node import px4_input_qos


VEHICLE_COMMAND_TOPIC = "/fmu/in/vehicle_command"
SEND_VEHICLE_COMMAND_SERVICE = "/uav/px4/send_vehicle_command"
STATUS_UNSUPPORTED_COMMAND = "unsupported PX4 lifecycle command"
STATUS_COMMAND_PUBLISHED = "PX4 lifecycle command published"


class Px4VehicleCommandOwnerNode(Node):
    """Validate the narrow lifecycle command vocabulary and publish it."""

    def __init__(self) -> None:
        """Create the sole live publisher and one internal service."""
        super().__init__("px4_vehicle_command_owner")
        module = importlib.import_module("px4_msgs.msg")
        self._VehicleCommand = module.VehicleCommand
        self._publisher = self.create_publisher(
            self._VehicleCommand, VEHICLE_COMMAND_TOPIC, px4_input_qos()
        )
        self._service = self.create_service(
            SendPx4VehicleCommand,
            SEND_VEHICLE_COMMAND_SERVICE,
            self._callback,
        )

    def _valid(self, request) -> bool:
        parameters = tuple(
            float(getattr(request, f"param{index}"))
            for index in range(1, 8)
        )
        if not all(math.isfinite(value) for value in parameters):
            return False
        command = int(request.command)
        if command == self._VehicleCommand.VEHICLE_CMD_DO_SET_MODE:
            return parameters[0] == 1.0 and parameters[1] == 6.0
        if command == self._VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM:
            return parameters[0] in {0.0, 1.0}
        return command == self._VehicleCommand.VEHICLE_CMD_NAV_LAND

    def _callback(self, request, response):
        if not self._valid(request):
            response.accepted = False
            response.status_message = STATUS_UNSUPPORTED_COMMAND
            return response
        message = self._VehicleCommand()
        message.timestamp = self.get_clock().now().nanoseconds // 1000
        for index in range(1, 8):
            setattr(
                message,
                f"param{index}",
                getattr(request, f"param{index}"),
            )
        message.command = int(request.command)
        message.target_system = 1
        message.target_component = 1
        message.source_system = 1
        message.source_component = 1
        message.from_external = True
        self._publisher.publish(message)
        response.accepted = True
        response.status_message = STATUS_COMMAND_PUBLISHED
        return response


def main(args=None) -> int:
    """Run the sole PX4 VehicleCommand publisher."""
    rclpy.init(args=args)
    node = Px4VehicleCommandOwnerNode()
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
