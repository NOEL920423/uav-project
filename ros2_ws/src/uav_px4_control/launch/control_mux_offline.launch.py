"""Launch a finite synthetic Phase 6 non-flight mux graph."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Connect synthetic candidates, mux service, and finite monitor."""
    share = FindPackageShare("uav_px4_control")
    config = PathJoinSubstitution([share, "config", "control_mux.yaml"])
    mode = LaunchConfiguration("mode")
    use_sim_time = LaunchConfiguration("use_sim_time")
    astar_behavior = LaunchConfiguration("astar_behavior")
    hold_behavior = LaunchConfiguration("hold_behavior")
    common = {"use_sim_time": use_sim_time}
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="normal"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("astar_behavior", default_value="normal"),
        DeclareLaunchArgument("hold_behavior", default_value="normal"),
        Node(
            package="uav_px4_control",
            executable="control_mux_node",
            name="control_mux",
            output="screen",
            parameters=[config, common],
        ),
        Node(
            package="uav_px4_control",
            executable="synthetic_astar_candidate",
            output="screen",
            parameters=[common, {"behavior": astar_behavior}],
        ),
        Node(
            package="uav_px4_control",
            executable="synthetic_joystick_candidate",
            output="screen",
            parameters=[common],
        ),
        Node(
            package="uav_px4_control",
            executable="synthetic_navrl_candidate",
            output="screen",
            parameters=[common],
        ),
        Node(
            package="uav_px4_control",
            executable="synthetic_hold_candidate",
            output="screen",
            parameters=[common, {"behavior": hold_behavior}],
        ),
        Node(
            package="uav_px4_control",
            executable="control_mux_result_monitor",
            output="screen",
            parameters=[common, {"mode": mode}],
            on_exit=Shutdown(reason="control mux fixture completed"),
        ),
    ])
