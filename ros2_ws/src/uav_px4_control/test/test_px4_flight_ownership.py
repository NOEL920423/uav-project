"""Ownership and ROS adapter regressions for the guarded SITL flight."""

import ast
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("recording", [False, True])
def test_automated_expert_launch_has_only_formal_nodes(recording):
    """Start the expert graph with its recorder enabled only on request."""
    from launch import LaunchContext
    from launch_ros.actions import Node

    module = runpy.run_path(str(
        PACKAGE.parent / "launch/px4_sitl_flight.launch.py"
    ))
    context = LaunchContext()
    context.launch_configurations.update({
        "record_expert_dataset": str(recording).lower(),
        "use_external_scene": "true",
    })
    enabled = {
        action.node_executable
        for action in module["generate_launch_description"]().entities
        if isinstance(action, Node)
        and (action.condition is None or action.condition.evaluate(context))
    }
    expected = {
        "scene_bridge_node", "px4_odometry_bridge_node", "control_mux_node",
        "px4_live_telemetry_adapter", "px4_mapping_gate_node",
        "px4_setpoint_streamer_node", "px4_vehicle_command_owner_node",
        "px4_sitl_flight_supervisor_node",
        "astar_planner_node", "trajectory_parameterizer_node",
        "trajectory_follower_node", "px4_sitl_flight_monitor",
    }
    if recording:
        expected.add("expert_dataset_recorder")
    assert enabled == expected


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


@pytest.fixture
def telemetry_supervisor():
    """Exercise real ROS messages and supervisor decisions without PX4."""
    import rclpy
    from uav_interfaces.msg import (
        ControlMuxStatus, Px4OutputGateStatus, Px4StreamStatus,
    )
    from uav_px4_control.flight.px4_flight_models import Px4FlightState
    from uav_px4_control.flight.px4_sitl_flight_supervisor_node import (
        Px4SitlFlightSupervisorNode,
    )

    rclpy.init(args=[])
    node = None
    try:
        node = Px4SitlFlightSupervisorNode()
        node._now_seconds = lambda: 1.0
        node._sitl_guard_valid = lambda: True
        status = node._VehicleStatus(timestamp=123_000)
        status.arming_state = status.ARMING_STATE_ARMED
        status.nav_state = status.NAVIGATION_STATE_OFFBOARD
        node._vehicle_status_callback(status)
        node._vehicle_odometry_callback(
            node._VehicleOdometry(timestamp=456_000)
        )
        node._mux_status = ControlMuxStatus(
            active_source="ASTAR_EXPERT", selected_command_valid=True,
        )
        node._gate_status = Px4OutputGateStatus(
            state="SAFE_TO_FORWARD", safe_to_forward=True,
        )
        node._stream_status = Px4StreamStatus(
            state="STREAMING", streaming=True, observed_rate_hz=20.0,
        )
        node._scene_kind = "takeoff"
        node.machine.request_enable(True, 0.0)
        node.machine.state = Px4FlightState.TAKEOFF
        errors, actions, statuses = [], [], []
        node.get_logger = lambda: SimpleNamespace(
            info=lambda message: None, error=errors.append,
        )
        node._execute_action = lambda action, now: actions.append(action)
        node._status_publisher = SimpleNamespace(publish=statuses.append)
        yield node, errors, actions, statuses
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


@pytest.mark.parametrize("status_age,odometry_age,stale", [
    (0.751, 0.25, ["/fmu/out/vehicle_status"]),
    (0.75, 0.251, ["/fmu/out/vehicle_odometry"]),
    (0.751, 0.251, [
        "/fmu/out/vehicle_status", "/fmu/out/vehicle_odometry",
    ]),
])
def test_telemetry_timeout_identifies_topic_and_preserves_landing(
    telemetry_supervisor, status_age, odometry_age, stale
):
    """Report the triggering ages without replacing the original failure."""
    node, errors, actions, statuses = telemetry_supervisor
    node._vehicle_status_receipt_s = 1.0 - status_age
    node._vehicle_odometry_receipt_s = 1.0 - odometry_age
    node._tick()

    assert actions == [
        "STOP_TRACKING", "DISABLE_STREAM", "DISABLE_OUTPUT_GATE", "SEND_LAND",
    ]
    assert statuses[-1].state == "LANDING"
    assert statuses[-1].failure_reason == "PX4 telemetry became stale"
    assert len(errors) == 1
    prefix, raw = errors[0].split(" ", 1)
    assert prefix == "PX4_TELEMETRY_TIMEOUT"
    snapshot = json.loads(raw)
    assert snapshot["failure_reason"] == statuses[-1].failure_reason
    assert snapshot["state_before"] == "TAKEOFF"
    assert snapshot["state_after"] == "LANDING"
    assert snapshot["decision_ros_s"] == 1.0
    assert snapshot["receive_clock"] == "ros_node_clock"
    assert snapshot["stale_topics"] == stale
    for topic, age, threshold, timestamp in (
        ("/fmu/out/vehicle_status", status_age, 0.75, 123_000),
        ("/fmu/out/vehicle_odometry", odometry_age, 0.25, 456_000),
    ):
        details = snapshot["topics"][topic]
        assert details["receive_age_s"] == pytest.approx(age)
        assert details["last_receive_ros_s"] == pytest.approx(1.0 - age)
        assert details["timeout_s"] == threshold
        assert details["px4_timestamp_us"] == timestamp
        assert details["fresh"] == (topic not in stale)


@pytest.mark.parametrize("status_age,odometry_age", [
    (0.0, 0.0), (0.749, 0.249), (0.75, 0.25),
])
def test_telemetry_at_original_limits_remains_fresh_and_quiet(
    telemetry_supervisor, status_age, odometry_age
):
    """Keep the inclusive 0.75/0.25 second limits and normal takeoff."""
    node, errors, actions, statuses = telemetry_supervisor
    node._vehicle_status_receipt_s = 1.0 - status_age
    node._vehicle_odometry_receipt_s = 1.0 - odometry_age
    assert node._evidence(1.0).telemetry_fresh
    node._tick()
    assert errors == []
    assert actions == []
    assert statuses[-1].state == "TAKEOFF"
    assert statuses[-1].failure_reason == ""


def test_missing_odometry_is_reported_without_invented_timestamps(
    telemetry_supervisor,
):
    """Serialize missing measurements as null and retain safe landing."""
    node, errors, actions, statuses = telemetry_supervisor
    node._vehicle_odometry = None
    node._vehicle_odometry_receipt_s = None
    node._tick()
    snapshot = json.loads(errors[0].split(" ", 1)[1])
    assert snapshot["stale_topics"] == ["/fmu/out/vehicle_odometry"]
    assert snapshot["topics"]["/fmu/out/vehicle_odometry"] == {
        "receive_age_s": None, "timeout_s": 0.25, "fresh": False,
        "last_receive_ros_s": None, "px4_timestamp_us": None,
    }
    assert "SEND_LAND" in actions
    assert statuses[-1].failure_reason == "PX4 telemetry became stale"


@pytest.mark.parametrize("failsafe", [False, True])
def test_stale_data_does_not_relabel_readiness_or_failsafe(
    telemetry_supervisor, failsafe
):
    """Log only an actual telemetry failure selected by the state machine."""
    from uav_px4_control.flight.px4_flight_models import Px4FlightState

    node, errors, actions, statuses = telemetry_supervisor
    node._vehicle_status_receipt_s = 0.0
    node._vehicle_status.failsafe = failsafe
    if not failsafe:
        node.machine.state = Px4FlightState.WAITING_PIPELINE
    node._tick()
    assert errors == []
    if failsafe:
        assert statuses[-1].failure_reason == "PX4 failsafe became active"
        assert "SEND_LAND" in actions
    else:
        assert statuses[-1].state == "WAITING_PIPELINE"
        assert statuses[-1].failure_reason == ""


def test_telemetry_timeout_logs_once_and_again_for_a_new_mission(
    telemetry_supervisor,
):
    """Landing retains one snapshot; a separately enabled mission gets its own."""
    from uav_interfaces.srv import SetPx4FlightEnable
    from uav_px4_control.flight.px4_flight_models import Px4FlightState

    node, errors, actions, statuses = telemetry_supervisor
    node._vehicle_status_receipt_s = 0.0
    node._tick()
    original_snapshot = errors[0]
    for _ in range(3):
        node._tick()
    node._vehicle_status.arming_state = (
        node._vehicle_status.ARMING_STATE_STANDBY
    )
    node._land_callback(node._VehicleLandDetected(landed=True))
    node._tick()
    assert statuses[-1].state == "FAILED"
    assert statuses[-1].failure_reason == "PX4 telemetry became stale"
    assert errors == [original_snapshot]

    node._publish_takeoff_scene = lambda: None
    response = node._flight_enable_callback(
        SetPx4FlightEnable.Request(enable=True), SetPx4FlightEnable.Response()
    )
    assert response.accepted
    node.machine.state = Px4FlightState.TAKEOFF
    node._scene_kind = "takeoff"
    node._vehicle_status.arming_state = node._vehicle_status.ARMING_STATE_ARMED
    node._land_callback(node._VehicleLandDetected(landed=False))
    node._tick()
    assert len(errors) == 2
    assert statuses[-1].failure_reason == "PX4 telemetry became stale"
