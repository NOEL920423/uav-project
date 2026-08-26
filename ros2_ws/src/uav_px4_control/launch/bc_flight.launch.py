"""Launch one formal TOP RGB BC flight without planner or follower nodes."""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    Shutdown,
)
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Connect TOP inference to the guarded PX4/Pegasus control boundary."""
    control = FindPackageShare("uav_px4_control")
    mux = PathJoinSubstitution([control, "config", "control_mux.yaml"])
    gate = PathJoinSubstitution(
        [control, "config", "px4_mapping_gate.yaml"]
    )
    streamer = PathJoinSubstitution(
        [control, "config", "px4_setpoint_streamer.yaml"]
    )
    flight = PathJoinSubstitution([control, "config", "bc_flight.yaml"])
    repository_root = LaunchConfiguration("repository_root")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    ml_python = LaunchConfiguration("ml_python")
    device = LaunchConfiguration("device")
    image_source = LaunchConfiguration("image_source")
    result_path = LaunchConfiguration("result_path")
    episode = LaunchConfiguration("episode")
    seed = LaunchConfiguration("seed")
    return LaunchDescription([
        DeclareLaunchArgument("repository_root"),
        DeclareLaunchArgument("checkpoint_path", default_value=""),
        DeclareLaunchArgument("ml_python"),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("image_source", default_value="top_rgb"),
        DeclareLaunchArgument(
            "result_path", default_value="/tmp/uav_bc_flight_result.json"
        ),
        DeclareLaunchArgument("episode", default_value="1"),
        DeclareLaunchArgument("seed", default_value="0"),
        SetEnvironmentVariable(
            "PYTHONPATH",
            [
                repository_root,
                ":",
                EnvironmentVariable("PYTHONPATH", default_value=""),
            ],
        ),
        Node(
            package="uav_scene_bridge",
            executable="scene_bridge_node",
            parameters=[{
                "enable_scene_access": True,
                "runtime_timeout_s": 0.50,
            }],
            remappings=[
                ("/uav/isaac/scene/obstacles", "/uav/scene/obstacles"),
                ("/uav/isaac/scene/start", "/uav/scene/start"),
                ("/uav/isaac/scene/goal", "/uav/scene/goal"),
            ],
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
            executable="px4_vehicle_command_owner_node",
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="bc_policy_node",
            parameters=[flight, {
                "repository_root": repository_root,
                "checkpoint_path": checkpoint_path,
                "ml_python": ml_python,
                "image_source": image_source,
                "device": device,
            }],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="bc_flight_supervisor_node",
            parameters=[flight],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="bc_episode_monitor_node",
            parameters=[flight, {
                "result_path": result_path,
                "episode": episode,
                "seed": seed,
                "image_source": image_source,
            }],
            output="screen",
            on_exit=Shutdown(reason="BC flight result saved"),
        ),
    ])
