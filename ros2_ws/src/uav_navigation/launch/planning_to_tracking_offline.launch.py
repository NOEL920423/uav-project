"""Launch the complete finite Phase 2-to-5 non-flight tracking graph."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Connect fixed scene through planning, timing, tracking, and plant."""
    share = FindPackageShare("uav_navigation")
    planner_parameters = PathJoinSubstitution(
        [share, "config", "astar_planner.yaml"]
    )
    trajectory_parameters = PathJoinSubstitution(
        [share, "config", "trajectory_parameterizer.yaml"]
    )
    follower_parameters = PathJoinSubstitution(
        [share, "config", "trajectory_follower.yaml"]
    )
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_bspline = LaunchConfiguration("enable_bspline")
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("enable_bspline", default_value="true"),
        Node(
            package="uav_navigation",
            executable="astar_planner_node",
            name="astar_planner",
            output="screen",
            parameters=[planner_parameters, {
                "use_sim_time": use_sim_time,
                "enable_bspline": enable_bspline,
            }],
        ),
        Node(
            package="uav_navigation",
            executable="trajectory_parameterizer_node",
            name="trajectory_parameterizer",
            output="screen",
            parameters=[trajectory_parameters, {
                "use_sim_time": use_sim_time,
            }],
        ),
        Node(
            package="uav_navigation",
            executable="trajectory_follower_node",
            name="trajectory_follower",
            output="screen",
            parameters=[follower_parameters, {
                "use_sim_time": use_sim_time,
                "trajectory_validity_timeout_s": 30.0,
            }],
        ),
        Node(
            package="uav_navigation",
            executable="tracking_scene_publisher",
            name="fixed_tracking_scene_publisher",
            output="screen",
            parameters=[{"use_sim_time": use_sim_time}],
        ),
        Node(
            package="uav_navigation",
            executable="offline_kinematic_plant",
            name="offline_kinematic_plant",
            output="screen",
            parameters=[{
                "fixture": "straight-trajectory",
                "full_pipeline": True,
                "use_sim_time": use_sim_time,
            }],
        ),
        Node(
            package="uav_navigation",
            executable="tracking_result_monitor",
            name="tracking_result_monitor",
            output="screen",
            parameters=[{
                "fixture": "straight-trajectory",
                "full_pipeline": True,
                "use_sim_time": use_sim_time,
            }],
            on_exit=Shutdown(reason="full tracking pipeline completed"),
        ),
    ])
