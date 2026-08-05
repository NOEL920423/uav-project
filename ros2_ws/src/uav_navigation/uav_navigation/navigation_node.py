"""Harmless Phase 1 navigation placeholder."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class NavigationNode(Node):
    """Declare the future planning boundary without running a planner."""

    def __init__(self) -> None:
        """Initialize a disabled navigation placeholder."""
        super().__init__("navigation")
        self.declare_parameter("enable_planning", False)
        self.get_logger().info(
            "Phase 1 scaffold active; planning is disabled."
        )


def main(args=None) -> None:
    """Run the placeholder until ROS requests shutdown."""
    rclpy.init(args=args)
    node = NavigationNode()
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
