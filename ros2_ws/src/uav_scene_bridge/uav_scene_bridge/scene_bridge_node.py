"""Harmless Phase 1 scene bridge placeholder."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class SceneBridgeNode(Node):
    """Declare the future scene bridge boundary without simulator access."""

    def __init__(self) -> None:
        """Initialize a disabled scene bridge placeholder."""
        super().__init__("scene_bridge")
        self.declare_parameter("enable_scene_access", False)
        self.get_logger().info(
            "Phase 1 scaffold active; Isaac scene access is disabled."
        )


def main(args=None) -> None:
    """Run the placeholder until ROS requests shutdown."""
    rclpy.init(args=args)
    node = SceneBridgeNode()
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
