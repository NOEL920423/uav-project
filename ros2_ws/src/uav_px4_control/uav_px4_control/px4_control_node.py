"""Harmless Phase 1 PX4 control placeholder."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class Px4ControlNode(Node):
    """Declare the future command boundary without creating publishers."""

    def __init__(self) -> None:
        """Initialize a disabled PX4 output placeholder."""
        super().__init__("px4_control")
        self.declare_parameter("enable_px4_output", False)
        self.get_logger().warning(
            "Phase 1 scaffold active; PX4 output is disabled and no "
            "/fmu/in/* publisher exists."
        )


def main(args=None) -> None:
    """Run the placeholder until ROS requests shutdown."""
    rclpy.init(args=args)
    node = Px4ControlNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
