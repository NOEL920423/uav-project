"""Harmless Phase 1 camera bridge placeholder."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class CameraBridgeNode(Node):
    """Declare the future camera boundary without touching render products."""

    def __init__(self) -> None:
        """Initialize a disabled camera bridge placeholder."""
        super().__init__("camera_bridge")
        self.declare_parameter("enable_camera_access", False)
        self.get_logger().info(
            "Phase 1 scaffold active; camera access is disabled."
        )


def main(args=None) -> None:
    """Run the placeholder until ROS requests shutdown."""
    rclpy.init(args=args)
    node = CameraBridgeNode()
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
