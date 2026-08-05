"""Harmless Phase 1 data recorder placeholder."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class DataRecorderNode(Node):
    """Declare the future recorder boundary without opening output files."""

    def __init__(self) -> None:
        """Initialize a disabled data recorder placeholder."""
        super().__init__("data_recorder")
        self.declare_parameter("enable_recording", False)
        self.get_logger().info(
            "Phase 1 scaffold active; recording is disabled."
        )


def main(args=None) -> None:
    """Run the placeholder until ROS requests shutdown."""
    rclpy.init(args=args)
    node = DataRecorderNode()
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
