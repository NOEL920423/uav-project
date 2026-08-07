"""Launch the finite ROS output-gate fault and recovery fixture."""

from launch import LaunchDescription
from launch.actions import Shutdown
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Connect mapping diagnostics to synthetic telemetry fault injection."""
    share = FindPackageShare("uav_px4_control")
    mux = PathJoinSubstitution([share, "config", "control_mux.yaml"])
    gate = PathJoinSubstitution([share, "config", "px4_mapping_gate.yaml"])
    return LaunchDescription([
        Node(
            package="uav_px4_control",
            executable="control_mux_node",
            parameters=[mux],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="synthetic_astar_candidate",
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_mapping_gate_node",
            parameters=[gate],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="synthetic_px4_telemetry",
            parameters=[{"behavior": "gate-fault"}],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_boundary_result_monitor",
            parameters=[{"mode": "gate"}],
            output="screen",
            on_exit=Shutdown(reason="PX4 gate fixture completed"),
        ),
    ])
