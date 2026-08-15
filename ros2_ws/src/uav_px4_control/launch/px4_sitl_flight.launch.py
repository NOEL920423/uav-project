"""Launch one finite ASTAR_EXPERT PX4 SITL flight acceptance mission."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Reuse the Phase 2-8 graph and add only live flight boundaries."""
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
    streamer = PathJoinSubstitution(
        [control, "config", "px4_setpoint_streamer.yaml"]
    )
    flight = PathJoinSubstitution(
        [control, "config", "px4_sitl_flight.yaml"]
    )
    evidence_path = LaunchConfiguration("evidence_path")
    timeout_s = LaunchConfiguration("timeout_s")
    start_delay_s = LaunchConfiguration("start_delay_s")
    use_external_scene = LaunchConfiguration("use_external_scene")
    require_isaac_evidence = LaunchConfiguration("require_isaac_evidence")
    return LaunchDescription([
        DeclareLaunchArgument(
            "evidence_path",
            default_value="/tmp/uav_px4_sitl_flight_evidence.json",
        ),
        DeclareLaunchArgument("timeout_s", default_value="120.0"),
        DeclareLaunchArgument("start_delay_s", default_value="2.0"),
        DeclareLaunchArgument("use_external_scene", default_value="false"),
        DeclareLaunchArgument(
            "require_isaac_evidence", default_value="false"
        ),
        Node(
            package="uav_scene_bridge",
            executable="scene_bridge_node",
            parameters=[{
                "enable_scene_access": True,
                "runtime_timeout_s": 0.50,
            }],
            condition=IfCondition(use_external_scene),
            output="screen",
        ),
        Node(
            package="uav_navigation",
            executable="astar_planner_node",
            parameters=[planner, flight],
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
            parameters=[follower, flight],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_odometry_bridge_node",
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
            parameters=[gate, flight],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_setpoint_streamer_node",
            parameters=[streamer, flight],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_sitl_flight_supervisor_node",
            parameters=[flight, {
                "use_external_scene": use_external_scene,
            }],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_sitl_flight_monitor",
            parameters=[{
                "evidence_path": evidence_path,
                "timeout_s": timeout_s,
                "start_delay_s": start_delay_s,
                "require_isaac_evidence": require_isaac_evidence,
            }],
            output="screen",
            on_exit=Shutdown(reason="PX4 SITL flight milestone completed"),
        ),
    ])
