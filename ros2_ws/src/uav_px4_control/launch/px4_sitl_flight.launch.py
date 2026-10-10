"""Launch A* acceptance or an explicitly enabled physical joystick flight."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration, PathJoinSubstitution, PythonExpression,
)

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _validate_human_options(context):
    """Refuse a human run with the existing A*-only dataset recorder."""
    human = LaunchConfiguration("human_control").perform(context).lower()
    record = LaunchConfiguration("record_expert_dataset").perform(context)
    record = record.lower()
    if human in {"true", "1"} and record in {"true", "1"}:
        raise RuntimeError("Human dataset recording is not implemented yet")
    return []


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
    human_control = LaunchConfiguration("human_control")
    joystick_device = LaunchConfiguration("joystick_device")
    verbose_diagnostics = ParameterValue(
        PythonExpression([
            "'", human_control, "'.lower() not in ('true', '1')",
        ]),
        value_type=bool,
    )
    require_isaac_evidence = LaunchConfiguration("require_isaac_evidence")
    record_expert_dataset = LaunchConfiguration("record_expert_dataset")
    dataset_root = LaunchConfiguration("dataset_root")
    episode_id = LaunchConfiguration("episode_id")
    collection_mode = LaunchConfiguration("collection_mode")
    random_seed = LaunchConfiguration("random_seed")
    expected_runtime_generation = LaunchConfiguration(
        "expected_runtime_generation"
    )
    expected_scene_revision = LaunchConfiguration("expected_scene_revision")
    minimum_fpv_frame_count = LaunchConfiguration("minimum_fpv_frame_count")
    minimum_observer_frame_count = LaunchConfiguration(
        "minimum_observer_frame_count"
    )
    minimum_depth_frame_count = LaunchConfiguration(
        "minimum_depth_frame_count"
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            "evidence_path",
            default_value="/tmp/uav_px4_sitl_flight_evidence.json",
        ),
        DeclareLaunchArgument("timeout_s", default_value="120.0"),
        DeclareLaunchArgument("start_delay_s", default_value="2.0"),
        DeclareLaunchArgument("use_external_scene", default_value="false"),
        DeclareLaunchArgument("human_control", default_value="false"),
        DeclareLaunchArgument(
            "joystick_device", default_value="/dev/input/js0"
        ),
        DeclareLaunchArgument(
            "require_isaac_evidence", default_value="false"
        ),
        DeclareLaunchArgument(
            "record_expert_dataset", default_value="false"
        ),
        DeclareLaunchArgument(
            "dataset_root",
            default_value="artifacts/datasets/bc_expert_v1",
        ),
        DeclareLaunchArgument("episode_id", default_value="episode_000001"),
        DeclareLaunchArgument("collection_mode", default_value="single"),
        DeclareLaunchArgument("random_seed", default_value="0"),
        DeclareLaunchArgument(
            "expected_runtime_generation", default_value="-1"
        ),
        DeclareLaunchArgument("expected_scene_revision", default_value="0"),
        DeclareLaunchArgument("minimum_fpv_frame_count", default_value="0"),
        DeclareLaunchArgument(
            "minimum_observer_frame_count", default_value="0"
        ),
        DeclareLaunchArgument("minimum_depth_frame_count", default_value="0"),
        OpaqueFunction(function=_validate_human_options),
        Node(
            package="uav_px4_control",
            executable="joystick_control_node",
            parameters=[{
                "device": joystick_device,
                "enable_control": True,
                "altitude_hold_m": 1.5,
                "latch_after_stop": True,
                "report_raw_inputs": False,
                "show_operator_status": True,
            }],
            condition=IfCondition(human_control),
            output="screen",
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
            condition=UnlessCondition(human_control),
            parameters=[planner, flight],
            output="screen",
        ),
        Node(
            package="uav_navigation",
            executable="trajectory_parameterizer_node",
            condition=UnlessCondition(human_control),
            parameters=[trajectory],
            output="screen",
        ),
        Node(
            package="uav_navigation",
            executable="trajectory_follower_node",
            condition=UnlessCondition(human_control),
            parameters=[follower, flight],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_odometry_bridge_node",
            output="screen",
        ),
        Node(
            package="uav_data_recorder",
            executable="expert_dataset_recorder",
            parameters=[{
                "dataset_root": dataset_root,
                "episode_id": episode_id,
                "collection_mode": collection_mode,
                "random_seed": random_seed,
                "expected_runtime_generation": expected_runtime_generation,
                "expected_scene_revision": expected_scene_revision,
                "minimum_fpv_frame_count": minimum_fpv_frame_count,
                "minimum_observer_frame_count": (
                    minimum_observer_frame_count
                ),
                "minimum_depth_frame_count": minimum_depth_frame_count,
                "synchronization_tolerance_s": 0.100,
            }],
            condition=IfCondition(record_expert_dataset),
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
            parameters=[{
                "log_telemetry_diagnostics": verbose_diagnostics,
            }],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_mapping_gate_node",
            parameters=[gate, flight, {
                "allow_controlled_mux_handoff": human_control,
                "log_startup_diagnostics": verbose_diagnostics,
            }],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_setpoint_streamer_node",
            parameters=[streamer, flight, {
                "log_startup_diagnostics": verbose_diagnostics,
            }],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_vehicle_command_owner_node",
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_sitl_flight_supervisor_node",
            parameters=[flight, {
                "use_external_scene": use_external_scene,
                "human_control": human_control,
            }],
            output="screen",
        ),
        Node(
            package="uav_px4_control",
            executable="px4_sitl_flight_monitor",
            condition=UnlessCondition(human_control),
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
