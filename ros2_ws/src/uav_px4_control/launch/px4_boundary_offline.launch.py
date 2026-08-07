"""Launch the complete scene-to-PX4-diagnostic-boundary graph."""

from launch import LaunchDescription
from launch.actions import Shutdown
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Stop scene-to-control integration at safe_to_forward diagnostics."""
    navigation = FindPackageShare("uav_navigation")
    control = FindPackageShare("uav_px4_control")
    planner = PathJoinSubstitution(
        [navigation, "config", "astar_planner.yaml"]
    )
    trajectory = PathJoinSubstitution(
        [navigation, "config", "trajectory_parameterizer.yaml"]
    )
    follower = PathJoinSubstitution(
        [navigation, "config", "trajectory_follower.yaml"]
    )
    mux = PathJoinSubstitution([control, "config", "control_mux.yaml"])
    gate = PathJoinSubstitution(
        [control, "config", "px4_mapping_gate.yaml"]
    )
    return LaunchDescription([
        Node(
            package="uav_navigation",
            executable="astar_planner_node",
            parameters=[planner, {"enable_bspline": True}],
            output="screen",
        ),
        Node(
            package="uav_navigation",
            executable="trajectory_parameterizer_node",
            parameters=[trajectory],
            output="screen",
        ),
        Node(
            package="uav_navigation",
            executable="trajectory_follower_node",
            parameters=[follower, {"trajectory_validity_timeout_s": 30.0}],
            output="screen",
        ),
        Node(
            package="uav_navigation",
            executable="tracking_scene_publisher",
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="control_mux_node",
            parameters=[mux],
            output="screen",
        ),
        Node(
            package="uav_navigation",
            executable="offline_kinematic_plant",
            parameters=[{
                "fixture": "straight-trajectory",
                "full_pipeline": True,
                "command_topic": "/uav/control/selected_command",
            }],
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
            parameters=[{"behavior": "boundary-fault"}],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_boundary_result_monitor",
            parameters=[{"mode": "boundary"}],
            output="screen",
            on_exit=Shutdown(reason="PX4 boundary fixture completed"),
        ),
    ])
