"""Static ROS ownership and forbidden-output regressions for Phase 7."""

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE.parent
REPOSITORY = SOURCE_ROOT.parents[1]


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
    """Keep all 13 pure boundary parameters in one installed YAML."""
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
    node = text(PACKAGE / "uav_px4_control/px4_mapping_gate_node.py")
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
        PACKAGE / "uav_px4_control/px4_mapping_gate_node.py",
        PACKAGE / "uav_px4_control/offline_px4_boundary_harness.py",
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
