"""Static single-owner contracts for the explicitly authorized SITL flight."""

import ast
import runpy
from pathlib import Path

import pytest


PACKAGE = Path(__file__).parents[1] / "uav_px4_control"
SUPERVISOR = PACKAGE / "flight" / "px4_sitl_flight_supervisor_node.py"
COMMAND_OWNER = PACKAGE / "flight" / "px4_vehicle_command_owner_node.py"
FLIGHT_CONFIG = PACKAGE.parent / "config" / "px4_sitl_flight.yaml"


def _publisher_topics(source: Path) -> list[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    constants = {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    topics = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_publisher"
            and len(node.args) >= 2
        ):
            continue
        topic = None
        if isinstance(node.args[1], ast.Constant):
            topic = node.args[1].value
        elif isinstance(node.args[1], ast.Name):
            topic = constants.get(node.args[1].id)
        if isinstance(topic, str) and topic.startswith("/fmu/in/"):
            topics.append(topic)
    return topics


def test_vehicle_command_has_exactly_one_tracked_owner():
    """Only the explicitly authorized supervisor may publish commands."""
    owners = []
    for source in sorted(PACKAGE.rglob("*.py")):
        if "/fmu/in/vehicle_command" in _publisher_topics(source):
            owners.append(source.name)
    assert owners == [COMMAND_OWNER.name]


def test_flight_supervisor_does_not_duplicate_setpoint_publishers():
    """The supervisor leaves both Phase 8 setpoint topics to the streamer."""
    assert _publisher_topics(SUPERVISOR) == []


def test_supervisor_accepts_existing_phase4_trajectory_provenance():
    """Flight readiness must match the reused parameterizer contract."""
    parameterizer = (
        PACKAGE.parents[1]
        / "uav_navigation"
        / "uav_navigation"
        / "trajectory"
        / "trajectory_parameterizer_node.py"
    )
    marker = '"PHASE4_TIME_PARAMETERIZED"'
    assert marker in parameterizer.read_text(encoding="utf-8")
    assert marker in SUPERVISOR.read_text(encoding="utf-8")


def test_takeoff_acceptance_matches_follower_terminal_tolerance():
    """The follower cannot settle below the supervisor takeoff boundary."""
    config = FLIGHT_CONFIG.read_text(encoding="utf-8")
    assert "goal_position_tolerance_m: 0.25" in config
    assert "takeoff_altitude_tolerance_m: 0.25" in config


def test_human_launch_uses_existing_nodes_and_rejects_astar_recording():
    """Human mode omits automatic missions and prevents mislabeled data."""
    from launch import LaunchContext
    from launch_ros.actions import Node

    module = runpy.run_path(str(
        PACKAGE.parent / "launch/px4_sitl_flight.launch.py"
    ))
    context = LaunchContext()
    context.launch_configurations.update({
        "human_control": "true", "record_expert_dataset": "false",
        "use_external_scene": "true",
    })
    module["_validate_human_options"](context)
    enabled = {
        action.node_executable
        for action in module["generate_launch_description"]().entities
        if isinstance(action, Node)
        and (action.condition is None or action.condition.evaluate(context))
    }
    assert "joystick_control_node" in enabled
    assert "px4_sitl_flight_supervisor_node" in enabled
    assert not enabled.intersection({
        "astar_planner_node", "trajectory_parameterizer_node",
        "trajectory_follower_node", "px4_sitl_flight_monitor",
        "expert_dataset_recorder",
    })
    context.launch_configurations["record_expert_dataset"] = "true"
    with pytest.raises(RuntimeError, match="not implemented"):
        module["_validate_human_options"](context)


def test_human_supervisor_ros_adapter_requires_live_neutral_landed_input():
    """Check input, human selection and fault landing at the adapter."""
    from geometry_msgs.msg import TwistStamped
    import rclpy
    from uav_interfaces.msg import (
        ControlMuxStatus, Px4OutputGateStatus, Px4StreamStatus,
    )
    from uav_interfaces.srv import SetPx4FlightEnable
    from uav_px4_control.flight.px4_flight_models import Px4FlightState
    from uav_px4_control.flight.px4_sitl_flight_supervisor_node import (
        Px4SitlFlightSupervisorNode,
    )

    rclpy.init(args=["--ros-args", "-p", "human_control:=true"])
    node = None
    try:
        node = Px4SitlFlightSupervisorNode()
        now = [1.0]
        node._now_seconds = lambda: now[0]
        node._sitl_guard_valid = lambda: True
        request = SetPx4FlightEnable.Request(enable=True)
        response = node._flight_enable_callback(
            request, SetPx4FlightEnable.Response()
        )
        assert not response.accepted
        status = node._VehicleStatus()
        status.arming_state = status.ARMING_STATE_STANDBY
        node._vehicle_status_callback(status)
        odometry = node._VehicleOdometry()
        odometry.position = [0.0, 0.0, 0.3]
        node._vehicle_odometry_callback(odometry)
        node._land_callback(node._VehicleLandDetected(landed=True))
        command = TwistStamped()
        command.header.frame_id = "px4_ned"
        command.header.stamp.sec = 1
        command.twist.linear.z = -0.6
        command.twist.linear.x = 0.1
        node._human_callback(command)
        assert not node._flight_enable_callback(
            request, SetPx4FlightEnable.Response()
        ).accepted
        command.twist.linear.x = 0.0
        node._human_callback(command)
        assert node._flight_enable_callback(
            request, SetPx4FlightEnable.Response()
        ).accepted
        assert node._ground_down_m == pytest.approx(0.3)
        assert node._scene_kind == ""  # No synthetic planner scene.
        node._mux_callback(ControlMuxStatus(
            active_source="HUMAN_JOYSTICK", selected_command_valid=True,
        ))
        node._gate_callback(Px4OutputGateStatus(
            state="SAFE_TO_FORWARD", safe_to_forward=True,
        ))
        node._stream_callback(Px4StreamStatus(
            state="STREAMING", streaming=True, observed_rate_hz=20.0,
        ))
        status.arming_state = status.ARMING_STATE_ARMED
        status.nav_state = status.NAVIGATION_STATE_OFFBOARD
        node._vehicle_status_callback(status)
        node._land_callback(node._VehicleLandDetected(landed=False))
        node.machine.state = Px4FlightState.HUMAN_CONTROL
        evidence = node._evidence(now[0])
        assert evidence.source_valid and evidence.human_selected
        assert not evidence.astar_selected
        actions = []
        node._execute_action = lambda action, stamp: actions.append(action)
        node._tick()
        assert node.machine.state == Px4FlightState.HUMAN_CONTROL
        # Simulate USB publisher disappearance while telemetry stays fresh.
        now[0] = 1.21
        node._tick()
        assert node.machine.state == Px4FlightState.LANDING
        assert "SEND_LAND" in actions and "DISABLE_STREAM" in actions
        command.twist.linear.x = float("nan")
        node._human_callback(command)
        assert node._human_command is None
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
