"""Launch the complete finite Phase 2-to-6 non-flight control graph."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Connect planning, follower, mux-selected command, and offline plant."""
    navigation_share = FindPackageShare("uav_navigation")
    control_share = FindPackageShare("uav_px4_control")
    planner = PathJoinSubstitution(
        [navigation_share, "config", "astar_planner.yaml"]
    )
    trajectory = PathJoinSubstitution(
        [navigation_share, "config", "trajectory_parameterizer.yaml"]
    )
    follower = PathJoinSubstitution(
        [navigation_share, "config", "trajectory_follower.yaml"]
    )
    mux = PathJoinSubstitution(
        [control_share, "config", "control_mux.yaml"]
    )
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_bspline = LaunchConfiguration("enable_bspline")
    common = {"use_sim_time": use_sim_time}
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("enable_bspline", default_value="true"),
        Node(
            package="uav_navigation",
            executable="astar_planner_node",
            name="astar_planner",
            output="screen",
            parameters=[planner, common, {"enable_bspline": enable_bspline}],
        ),
        Node(
            package="uav_navigation",
            executable="trajectory_parameterizer_node",
            name="trajectory_parameterizer",
            output="screen",
            parameters=[trajectory, common],
        ),
        Node(
            package="uav_navigation",
            executable="trajectory_follower_node",
            name="trajectory_follower",
            output="screen",
            parameters=[follower, common, {
                "trajectory_validity_timeout_s": 30.0,
            }],
        ),
        Node(
            package="uav_navigation",
            executable="tracking_scene_publisher",
            output="screen",
            parameters=[common],
        ),
        Node(
            package="uav_px4_control",
            executable="control_mux_node",
            name="control_mux",
            output="screen",
            parameters=[mux, common],
        ),
        Node(
            package="uav_navigation",
            executable="offline_kinematic_plant",
            output="screen",
            parameters=[common, {
                "fixture": "straight-trajectory",
                "full_pipeline": True,
                "command_topic": "/uav/control/selected_command",
            }],
        ),
        Node(
            package="uav_px4_control",
            executable="control_mux_result_monitor",
            output="screen",
            parameters=[common, {"mode": "control-stack"}],
            on_exit=Shutdown(reason="Phase 6 control stack completed"),
        ),
    ])
