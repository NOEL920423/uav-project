"""Static ROS interface, ownership, config, and launch regressions."""

import ast
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE.parent
REPOSITORY = SOURCE_ROOT.parents[1]


def text(path: Path) -> str:
    """Read one tracked contract file as UTF-8."""
    return path.read_text(encoding="utf-8")


def test_control_mux_status_interface_is_exact() -> None:
    """Lock every status field, type, and ordering."""
    expected = """std_msgs/Header header
string requested_source
string active_source
bool selected_command_valid
bool hold_active
string hold_reason
bool switch_in_progress
float64 switch_remaining_time
float64 selected_source_age
float64 selected_linear_speed
float64 selected_yaw_rate
uint64 transition_count
string[] healthy_sources
string[] stale_sources
string status_message
"""
    path = SOURCE_ROOT / "uav_interfaces/msg/ControlMuxStatus.msg"
    assert text(path) == expected


def test_set_control_source_service_is_exact() -> None:
    """Lock request and response fields without aliases."""
    expected = """string source
---
bool accepted
string requested_source
string active_source
string status_message
"""
    path = SOURCE_ROOT / "uav_interfaces/srv/SetControlSource.srv"
    assert text(path) == expected


def test_mux_config_contains_exact_nineteen_contract_parameters() -> None:
    """Keep one installed YAML with all locked Phase 6 mux parameters."""
    config = text(PACKAGE / "config/control_mux.yaml")
    names = {
        line.strip().split(":", 1)[0]
        for line in config.splitlines()
        if line.startswith("    ") and not line.startswith("      ")
    }
    names.remove("use_sim_time")
    expected = {
        "default_source", "publish_rate_hz", "astar_timeout_s",
        "joystick_timeout_s", "navrl_timeout_s", "hold_timeout_s",
        "switch_hold_duration_s", "minimum_source_dwell_time_s",
        "maximum_selected_speed_mps",
        "maximum_selected_horizontal_speed_mps",
        "maximum_selected_vertical_speed_mps",
        "maximum_selected_acceleration_mps2",
        "maximum_selected_yaw_rate_radps",
        "maximum_selected_yaw_acceleration_radps2",
        "reject_wrong_frame", "require_monotonic_candidate_stamps",
        "require_fresh_command_before_switch", "latch_hold_after_fault",
        "hold_command_epsilon",
    }
    assert names == expected


def test_only_mux_node_creates_selected_command_publisher() -> None:
    """Enforce sole selected-command publisher ownership through Python AST."""
    publishers = []
    for path in SOURCE_ROOT.glob("*/**/*.py"):
        tree = ast.parse(text(path), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Attribute):
                continue
            if function.attr != "create_publisher":
                continue
            source = ast.get_source_segment(text(path), node) or ""
            if "SELECTED_COMMAND_TOPIC" in source:
                publishers.append(path.name)
    assert publishers == ["control_mux_node.py"]


def test_candidate_and_selected_layers_remain_separate() -> None:
    """Keep follower output, mux ownership, and plant remap explicit."""
    follower = text(
        SOURCE_ROOT
        / "uav_navigation/uav_navigation/trajectory_follower_node.py"
    )
    assert 'COMMAND_TOPIC = "/uav/control/astar_command"' in follower
    assert '"/uav/control/selected_command"' not in follower
    plant = text(
        SOURCE_ROOT
        / "uav_navigation/uav_navigation/offline_tracking_harness.py"
    )
    assert 'declare_parameter("command_topic", COMMAND_TOPIC)' in plant
    stack = text(PACKAGE / "launch/control_stack_offline.launch.py")
    assert '"command_topic": "/uav/control/selected_command"' in stack


def test_synthetic_sources_have_no_hardware_or_model_runtime() -> None:
    """Keep joystick and NavRL fixtures synthetic and topic-scoped."""
    harness = text(
        PACKAGE / "uav_px4_control/offline_control_mux_harness.py"
    )
    forbidden = ("pygame", "inputs", "torch", "tensorflow", "onnxruntime")
    assert not any(name in harness for name in forbidden)
    assert "joystick_publisher_main" in harness
    assert "navrl_publisher_main" in harness
    assert "hold_publisher_main" in harness


def test_phase6_graphs_and_modules_have_no_flight_output() -> None:
    """Reject PX4 inputs, OFFBOARD, arming, and simulator runtime paths."""
    paths = [
        PACKAGE / "uav_px4_control/control_mux_node.py",
        PACKAGE / "uav_px4_control/offline_control_mux_harness.py",
        PACKAGE / "launch/control_mux_offline.launch.py",
        PACKAGE / "launch/control_stack_offline.launch.py",
    ]
    forbidden = ("px4_msgs", "OFFBOARD", "arm(", "omni.isaac")
    content = "\n".join(text(path) for path in paths)
    assert not any(value in content for value in forbidden)
    assert "create_publisher" not in "\n".join(
        line for line in content.splitlines() if "/fmu/in/" in line
    )


def test_wrapper_exposes_all_three_finite_phase6_checks() -> None:
    """Keep mux, safety, and complete-stack wrapper commands installed."""
    wrapper = text(REPOSITORY / "uav")
    for command in ("mux-check", "mux-safety-check", "control-stack-check"):
        assert command in wrapper


def test_live_mux_monitor_has_deterministic_dwell_margin() -> None:
    """Keep the harness request later than the configured mux dwell gate."""
    harness = text(
        PACKAGE / "uav_px4_control/offline_control_mux_harness.py"
    )
    assert "SOURCE_DWELL_SETTLE_S = 0.35" in harness
    assert harness.count("SOURCE_DWELL_SETTLE_S") == 3


def test_mux_status_topic_is_canonical() -> None:
    """Lock the typed mux status topic required by the Phase 6 contract."""
    node = text(PACKAGE / "uav_px4_control/control_mux_node.py")
    assert 'MUX_STATUS_TOPIC = "/uav/control/mux_status"' in node
