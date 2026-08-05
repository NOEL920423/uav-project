"""Launch only harmless Phase 1 placeholder nodes."""

from launch import LaunchDescription

from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Return a launch graph with all real capabilities disabled."""
    return LaunchDescription(
        [
            Node(
                package="uav_scene_bridge",
                executable="scene_bridge_node",
                parameters=[{"enable_scene_access": False}],
                output="screen",
            ),
            Node(
                package="uav_camera_bridge",
                executable="camera_bridge_node",
                parameters=[{"enable_camera_access": False}],
                output="screen",
            ),
            Node(
                package="uav_navigation",
                executable="navigation_node",
                parameters=[{"enable_planning": False}],
                output="screen",
            ),
            Node(
                package="uav_px4_control",
                executable="px4_control_node",
                parameters=[{"enable_px4_output": False}],
                output="screen",
            ),
            Node(
                package="uav_data_recorder",
                executable="data_recorder_node",
                parameters=[{"enable_recording": False}],
                output="screen",
            ),
        ]
    )
