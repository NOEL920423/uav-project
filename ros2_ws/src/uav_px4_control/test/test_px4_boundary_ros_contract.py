"""Static ROS ownership and forbidden-output regressions for Phase 7."""

import ast
from pathlib import Path
from types import SimpleNamespace


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE.parent
REPOSITORY = SOURCE_ROOT.parents[1]


def test_quiet_telemetry_keeps_safety_data_and_diagnostics(monkeypatch):
    """Only routine console diagnostics change; safety data remain live."""
    import px4_msgs.msg as px4
    import rclpy
    from rclpy.parameter import Parameter
    from uav_px4_control.px4.px4_live_telemetry_adapter import (
        Px4LiveTelemetryAdapter,
    )

    monkeypatch.delenv("UAV_TIMING_DIR", raising=False)
    rclpy.init(args=[
        "--ros-args", "-p", "log_telemetry_diagnostics:=false",
    ])
    node = None
    try:
        node = Px4LiveTelemetryAdapter()
        info, records, published = [], [], []
        node.get_logger = lambda: SimpleNamespace(info=info.append)
        node._timing.record = lambda event, **fields: records.append(
            (event, fields)
        )
        node._publisher = SimpleNamespace(publish=published.append)
        node._status = px4.VehicleStatus()
        node._mode = px4.VehicleControlMode()
        node._odometry = px4.VehicleOdometry()
        node._flags = px4.FailsafeFlags()
        node._odometry.pose_frame = node._odometry.POSE_FRAME_NED
        node._odometry.velocity_frame = node._odometry.VELOCITY_FRAME_NED
        node._record_topic("vehicle_odometry")
        node._tick()
        assert info == []
        assert len(published) == 1 and published[0].odometry_valid
        diagnostic = next(fields for event, fields in records
                          if event == "telemetry_diagnostic")
        assert diagnostic["topics"]["vehicle_odometry"]["received_count"] == 1
        # Default/full output remains available without changing the data.
        node.set_parameters([
            Parameter("log_telemetry_diagnostics", value=True),
        ])
        node._last_diagnostic_log_s = 0.0
        node._tick()
        assert "PX4_TELEMETRY_DIAG" in info[-1]
        assert len(published) == 2 and published[1].odometry_valid
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_quiet_gate_and_streamer_keep_status_and_diagnostic_records(monkeypatch):
    """Hide large startup dictionaries without enabling or dropping outputs."""
    import rclpy
    from rclpy.parameter import Parameter
    from uav_px4_control.px4.px4_mapping_gate_node import Px4MappingGateNode
    from uav_px4_control.px4.px4_setpoint_streamer_node import (
        Px4SetpointStreamerNode,
    )

    monkeypatch.delenv("UAV_TIMING_DIR", raising=False)
    rclpy.init(args=["--ros-args", "-p", "log_startup_diagnostics:=false"])
    nodes = []
    try:
        for cls in (Px4MappingGateNode, Px4SetpointStreamerNode):
            node = cls()
            nodes.append(node)
            info, records, statuses = [], [], []
            node.get_logger = lambda: SimpleNamespace(
                info=info.append, warning=info.append, error=info.append,
            )
            node._timing.record = lambda event, **fields: records.append(
                (event, fields)
            )
            node._status_publisher = SimpleNamespace(publish=statuses.append)
            node._tick()
            assert info == []
            assert len(statuses) == 1
            assert any("diagnostic_json" in fields for _, fields in records)
            first_state = statuses[-1].state
            node.set_parameters([
                Parameter("log_startup_diagnostics", value=True),
            ])
            node._diagnostic_key = None
            node._tick()
            assert "BC_STARTUP_DIAG" in info[-1]
            assert statuses[-1].state == first_state
    finally:
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()


def text(path: Path) -> str:
    """Read one contract file as UTF-8."""
    return path.read_text(encoding="utf-8")


def test_px4_output_gate_status_interface_is_exact() -> None:
    """Lock every architecture-level gate diagnostic field."""
    expected = """std_msgs/Header header
string state
bool enable_requested
bool safe_to_forward
bool selected_command_valid
bool mux_valid
bool telemetry_valid
bool failsafe
string active_source
float64 selected_command_age
float64 telemetry_age
float64 candidate_horizontal_speed
float64 candidate_total_speed
float64 candidate_yaw_rate
uint64 transition_count
string hold_reason
string status_message
"""
    path = SOURCE_ROOT / "uav_interfaces/msg/Px4OutputGateStatus.msg"
    assert text(path) == expected


def test_set_px4_output_enable_service_is_exact() -> None:
    """Lock the simulated-gate service without a PX4 control field."""
    expected = """bool enable
---
bool accepted
bool enable_requested
bool safe_to_forward
string state
string status_message
"""
    path = SOURCE_ROOT / "uav_interfaces/srv/SetPx4OutputEnable.srv"
    assert text(path) == expected


def test_mapping_gate_config_contains_all_locked_parameters() -> None:
    """Keep all pure boundary parameters in one installed YAML."""
    config = text(PACKAGE / "config/px4_mapping_gate.yaml")
    names = {
        line.strip().split(":", 1)[0]
        for line in config.splitlines()
        if line.startswith("    ") and not line.startswith("      ")
    }
    names -= {"use_sim_time", "publish_rate_hz"}
    expected = {
        "maximum_north_velocity_mps",
        "maximum_east_velocity_mps",
        "maximum_down_velocity_mps",
        "maximum_horizontal_velocity_mps",
        "maximum_total_velocity_mps",
        "maximum_yaw_rate_radps",
        "selected_command_timeout_s",
        "telemetry_timeout_s",
        "require_mux_valid",
        "require_known_source",
        "require_px4_ned_frame",
        "latch_faults",
        "require_explicit_enable",
        "lock_vehicle_state_signature",
        "allow_controlled_mux_handoff",
    }
    assert names == expected


def test_only_mux_creates_selected_command_publisher_after_phase7() -> None:
    """Preserve one selected-command owner across the complete workspace."""
    owners = []
    for path in SOURCE_ROOT.glob("*/**/*.py"):
        source = text(path)
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "create_publisher":
                continue
            call = ast.get_source_segment(source, node) or ""
            if "SELECTED_COMMAND_TOPIC" in call:
                owners.append(path.name)
    assert owners == ["control_mux_node.py"]


def test_mapping_node_consumes_mux_and_only_publishes_diagnostics() -> None:
    """Lock mapping input and all three non-live output topic names."""
    node = text(PACKAGE / "uav_px4_control/px4/px4_mapping_gate_node.py")
    assert "SELECTED_COMMAND_TOPIC" in node
    assert 'CANDIDATE_TOPIC = "/uav/px4/setpoint_candidate"' in node
    assert 'GATE_STATUS_TOPIC = "/uav/px4/output_gate_status"' in node
    assert 'SAFE_TO_FORWARD_TOPIC = "/uav/px4/safe_to_forward"' in node
    assert "px4_msgs" not in node


def test_no_phase7_publisher_targets_a_live_px4_input() -> None:
    """Inspect every publisher call and reject live bridge destinations."""
    paths = list(PACKAGE.glob("uav_px4_control/*.py"))
    paths += list(PACKAGE.glob("launch/*.py"))
    for path in paths:
        source = text(path)
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "create_publisher":
                continue
            call = ast.get_source_segment(source, node) or ""
            assert "/fmu/in/" not in call


def test_no_phase7_adapter_constructs_px4_flight_messages() -> None:
    """Keep real setpoint, mode, and vehicle-command messages absent."""
    paths = [
        PACKAGE / "uav_px4_control/px4/px4_mapping_gate_node.py",
        PACKAGE / "uav_px4_control/diagnostics/offline_px4_boundary_harness.py",
    ]
    content = "\n".join(text(path) for path in paths)
    forbidden = (
        "VehicleCommand",
        "OffboardControlMode",
        "TrajectorySetpoint",
        "px4_msgs",
    )
    assert not any(name in content for name in forbidden)


def test_all_phase7_wrappers_and_launches_are_installed() -> None:
    """Expose mapping, gate, and full-boundary finite checks."""
    wrapper = text(REPOSITORY / "uav")
    setup = text(PACKAGE / "setup.py")
    for command in (
        "px4-map-check",
        "px4-gate-check",
        "px4-boundary-check",
    ):
        assert command in wrapper
    for launch in (
        "px4_mapping_offline.launch.py",
        "px4_gate_offline.launch.py",
        "px4_boundary_offline.launch.py",
    ):
        assert launch in setup


def test_boundary_launch_stops_at_diagnostic_permission() -> None:
    """Keep the full graph terminal output at safe_to_forward."""
    launch = text(PACKAGE / "launch/px4_boundary_offline.launch.py")
    assert 'executable="px4_mapping_gate_node"' in launch
    assert 'executable="synthetic_px4_telemetry"' in launch
    assert 'executable="px4_boundary_result_monitor"' in launch
    assert "/fmu/" not in launch
