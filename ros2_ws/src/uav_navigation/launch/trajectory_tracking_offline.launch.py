"""Launch the direct finite Phase 5 non-flight tracking graph."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Connect fixed trajectory, follower, pure plant adapter, and monitor."""
    share = FindPackageShare("uav_navigation")
    parameters = PathJoinSubstitution(
        [share, "config", "trajectory_follower.yaml"]
    )
    fixture = LaunchConfiguration("fixture")
    use_sim_time = LaunchConfiguration("use_sim_time")
    common = {"fixture": fixture, "use_sim_time": use_sim_time}
    return LaunchDescription([
        DeclareLaunchArgument(
            "fixture", default_value="straight-trajectory"
        ),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        Node(
            package="uav_navigation",
            executable="tracking_trajectory_publisher",
            name="fixed_tracking_trajectory_publisher",
            output="screen",
            parameters=[common],
        ),
        Node(
            package="uav_navigation",
            executable="trajectory_follower_node",
            name="trajectory_follower",
            output="screen",
            parameters=[parameters, {"use_sim_time": use_sim_time}],
        ),
        Node(
            package="uav_navigation",
            executable="offline_kinematic_plant",
            name="offline_kinematic_plant",
            output="screen",
            parameters=[common],
        ),
        Node(
            package="uav_navigation",
            executable="tracking_result_monitor",
            name="tracking_result_monitor",
            output="screen",
            parameters=[common],
            on_exit=Shutdown(reason="tracking fixture completed"),
        ),
    ])
