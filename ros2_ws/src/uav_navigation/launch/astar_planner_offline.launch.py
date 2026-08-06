"""Launch the non-flight A* planner and optional finite offline fixture."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _launch_nodes(context):
    """Create fixture-aware nodes after resolving safe launch arguments."""
    parameter_file = LaunchConfiguration("planner_parameter_file")
    use_sim_time_text = LaunchConfiguration("use_sim_time").perform(context)
    enable_text = LaunchConfiguration("enable_bspline").perform(context)
    fixture = LaunchConfiguration("fixture").perform(context)
    enabled = enable_text.lower() in {"1", "true", "yes", "on"}
    if fixture == "bspline-disabled":
        enabled = False
    overrides = {
        "use_sim_time": use_sim_time_text.lower()
        in {"1", "true", "yes", "on"},
        "enable_bspline": enabled,
    }
    if fixture == "bspline-rejected-clearance":
        overrides["bspline_minimum_clearance_m"] = 0.40
    if fixture == "curvature-limit-rejection":
        overrides["bspline_maximum_curvature"] = 0.01
    return [
        Node(
            package="uav_navigation",
            executable="astar_planner_node",
            name="astar_planner",
            output="screen",
            parameters=[parameter_file, overrides],
        ),
        Node(
            package="uav_navigation",
            executable="astar_offline_harness",
            name="astar_offline_harness",
            output="screen",
            condition=IfCondition(LaunchConfiguration("with_offline_fixture")),
            parameters=[
                {
                    "use_sim_time": overrides["use_sim_time"],
                    "enable_bspline": enabled,
                    "fixture": fixture,
                }
            ],
            on_exit=Shutdown(reason="offline fixture completed"),
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """Create a launch graph containing no simulator or control processes."""
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
                "enable_bspline",
                default_value="true",
                description=(
                    "Generate and independently validate B-spline candidate"
                ),
            ),
            DeclareLaunchArgument(
                "fixture",
                default_value="bspline-safe-single-obstacle",
                description="Named deterministic Phase 3 offline fixture",
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
            OpaqueFunction(function=_launch_nodes),
        ]
    )
