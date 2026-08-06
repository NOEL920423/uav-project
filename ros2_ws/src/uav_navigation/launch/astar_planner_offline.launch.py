"""Launch the non-flight A* planner and optional finite offline fixture."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Create a launch graph containing no simulator or control processes."""
    parameter_file = LaunchConfiguration("planner_parameter_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    with_fixture = LaunchConfiguration("with_offline_fixture")
    default_parameters = PathJoinSubstitution(
        [FindPackageShare("uav_navigation"), "config", "astar_planner.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "planner_parameter_file",
                default_value=default_parameters,
                description="A* planner YAML parameter file",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use a ROS /clock source when true",
            ),
            DeclareLaunchArgument(
                "with_offline_fixture",
                default_value="true",
                description="Run the finite fixed-scene test harness",
            ),
            Node(
                package="uav_navigation",
                executable="astar_planner_node",
                name="astar_planner",
                output="screen",
                parameters=[
                    parameter_file,
                    {"use_sim_time": use_sim_time},
                ],
            ),
            Node(
                package="uav_navigation",
                executable="astar_offline_harness",
                name="astar_offline_harness",
                output="screen",
                condition=IfCondition(with_fixture),
                parameters=[{"use_sim_time": use_sim_time}],
                on_exit=Shutdown(reason="offline fixture completed"),
            ),
        ]
    )
