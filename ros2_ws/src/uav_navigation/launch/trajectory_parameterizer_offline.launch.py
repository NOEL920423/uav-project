"""Launch only the Phase 4 node and a finite direct-path harness."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _nodes(context):
    """Resolve fixture-specific safe parameter overrides."""
    fixture = LaunchConfiguration("fixture").perform(context)
    overrides = {
        "use_sim_time": LaunchConfiguration("use_sim_time").perform(
            context
        ).lower() in {"true", "1", "yes", "on"},
    }
    if fixture == "jerk-scaling":
        overrides["maximum_jerk_mps3"] = 0.25
        overrides["maximum_yaw_rate_radps"] = 0.3
    if fixture == "impossible-config-rejection":
        overrides["maximum_jerk_mps3"] = 0.001
        overrides["maximum_total_time_scale"] = 1.001
    return [
        Node(
            package="uav_navigation",
            executable="trajectory_parameterizer_node",
            name="trajectory_parameterizer",
            output="screen",
            parameters=[LaunchConfiguration("parameter_file"), overrides],
        ),
        Node(
            package="uav_navigation",
            executable="trajectory_offline_harness",
            name="trajectory_offline_harness",
            output="screen",
            parameters=[{"fixture": fixture, **overrides}],
            on_exit=Shutdown(reason="trajectory fixture completed"),
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """Create a non-flight graph containing no planner or simulator."""
    default_parameters = PathJoinSubstitution(
        [
            FindPackageShare("uav_navigation"),
            "config",
            "trajectory_parameterizer.yaml",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "parameter_file", default_value=default_parameters
            ),
            DeclareLaunchArgument("fixture", default_value="straight-line"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            OpaqueFunction(function=_nodes),
        ]
    )
