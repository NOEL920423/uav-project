"""Launch the finite Phase 3-to-Phase 4 non-flight integration graph."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Connect one fixed scene to planner, B-spline, and parameterizer."""
    share = FindPackageShare("uav_navigation")
    planner_parameters = PathJoinSubstitution(
        [share, "config", "astar_planner.yaml"]
    )
    trajectory_parameters = PathJoinSubstitution(
        [share, "config", "trajectory_parameterizer.yaml"]
    )
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_bspline = LaunchConfiguration("enable_bspline")
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("enable_bspline", default_value="true"),
            Node(
                package="uav_navigation",
                executable="astar_planner_node",
                name="astar_planner",
                output="screen",
                parameters=[
                    planner_parameters,
                    {
                        "use_sim_time": use_sim_time,
                        "enable_bspline": enable_bspline,
                    },
                ],
            ),
            Node(
                package="uav_navigation",
                executable="trajectory_parameterizer_node",
                name="trajectory_parameterizer",
                output="screen",
                parameters=[
                    trajectory_parameters,
                    {"use_sim_time": use_sim_time},
                ],
            ),
            Node(
                package="uav_navigation",
                executable="trajectory_pipeline_harness",
                name="trajectory_pipeline_harness",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
                on_exit=Shutdown(reason="pipeline fixture completed"),
            ),
        ]
    )
