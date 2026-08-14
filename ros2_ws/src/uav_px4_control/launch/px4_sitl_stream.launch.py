"""Launch the finite Phase 8 boundary against already-running PX4 SITL."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Connect the mux and both safety gates without starting PX4 or XRCE."""
    control = FindPackageShare("uav_px4_control")
    mux = PathJoinSubstitution([control, "config", "control_mux.yaml"])
    gate = PathJoinSubstitution(
        [control, "config", "px4_mapping_gate.yaml"]
    )
    streamer = PathJoinSubstitution(
        [control, "config", "px4_setpoint_streamer.yaml"]
    )
    fixture = LaunchConfiguration("fixture")
    run_monitor = LaunchConfiguration("run_monitor")
    return LaunchDescription([
        DeclareLaunchArgument("fixture", default_value="zero"),
        DeclareLaunchArgument("run_monitor", default_value="true"),
        Node(
            package="uav_px4_control",
            executable="synthetic_astar_candidate",
            parameters=[{"behavior": fixture}],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="control_mux_node",
            parameters=[mux],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_live_telemetry_adapter",
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
            executable="px4_setpoint_streamer_node",
            parameters=[streamer],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_sitl_stream_monitor",
            parameters=[{"fixture": fixture}],
            output="screen",
            condition=IfCondition(run_monitor),
            on_exit=Shutdown(reason="Phase 8 SITL stream check completed"),
        ),
    ])
