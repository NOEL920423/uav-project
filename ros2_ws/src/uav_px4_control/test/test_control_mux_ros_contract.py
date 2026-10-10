"""Static ROS interface, ownership, config, and launch regressions."""

import ast
import errno
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from uav_px4_control.control import joystick_control_node as joystick
from uav_px4_control.diagnostics.offline_control_mux_harness import (
    CandidateTrafficEvidence,
)


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


def test_mux_config_contains_exact_contract_parameters() -> None:
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
        "lifecycle_timeout_s", "bc_timeout_s", "joystick_timeout_s",
        "navrl_timeout_s", "hold_timeout_s",
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
        / "uav_navigation/uav_navigation/tracking/trajectory_follower_node.py"
    )
    assert 'COMMAND_TOPIC = "/uav/control/astar_command"' in follower
    assert '"/uav/control/selected_command"' not in follower
    plant = text(
        SOURCE_ROOT
        / "uav_navigation/uav_navigation/diagnostics/offline_tracking_harness.py"
    )
    assert 'declare_parameter("command_topic", COMMAND_TOPIC)' in plant
    stack = text(PACKAGE / "launch/control_stack_offline.launch.py")
    assert '"command_topic": "/uav/control/selected_command"' in stack


def test_synthetic_sources_have_no_hardware_or_model_runtime() -> None:
    """Keep joystick and NavRL fixtures synthetic and topic-scoped."""
    harness = text(
        PACKAGE / "uav_px4_control/diagnostics/offline_control_mux_harness.py"
    )
    forbidden = ("pygame", "inputs", "torch", "tensorflow", "onnxruntime")
    assert not any(name in harness for name in forbidden)
    assert "joystick_publisher_main" in harness
    assert "navrl_publisher_main" in harness
    assert "hold_publisher_main" in harness


def test_phase6_graphs_and_modules_have_no_flight_output() -> None:
    """Reject PX4 inputs, OFFBOARD, arming, and simulator runtime paths."""
    paths = [
        PACKAGE / "uav_px4_control/control/control_mux_node.py",
        PACKAGE / "uav_px4_control/diagnostics/offline_control_mux_harness.py",
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
        PACKAGE / "uav_px4_control/diagnostics/offline_control_mux_harness.py"
    )
    assert "SOURCE_DWELL_SETTLE_S = 0.35" in harness
    assert harness.count("SOURCE_DWELL_SETTLE_S") == 3


def test_live_mux_monitor_requires_sustained_fresh_candidate_traffic() -> None:
    """Select only after several recent, monotonic heartbeat arrivals."""
    evidence = CandidateTrafficEvidence()
    evidence.record(10.00, 1.00)
    evidence.record(10.04, 1.04)
    assert not evidence.ready(1.05)
    evidence.record(10.08, 1.08)
    assert evidence.ready(1.09)
    assert not evidence.ready(1.21)


def test_live_mux_monitor_rejects_replayed_candidate_stamp_readiness() -> None:
    """A repeated stamp cannot satisfy the pre-selection heartbeat gate."""
    evidence = CandidateTrafficEvidence()
    evidence.record(10.00, 1.00)
    evidence.record(10.04, 1.04)
    evidence.record(10.04, 1.08)
    assert not evidence.stamps_monotonic
    assert not evidence.ready(1.09)


def test_mux_status_topic_is_canonical() -> None:
    """Lock the typed mux status topic required by the Phase 6 contract."""
    node = text(PACKAGE / "uav_px4_control/control/control_mux_node.py")
    assert 'MUX_STATUS_TOPIC = "/uav/control/mux_status"' in node


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -math.pi / 2, math.pi])
def test_usb_joystick_mapping_preserves_body_action_contract(yaw) -> None:
    """Match BC limits and rotate body intent into NED at varied headings."""
    mapping = joystick.JoystickMapping()
    axes = {1: 1.0, 0: 0.5, 3: 1.0}
    north, east, down, yaw_rate = mapping.command(axes, yaw)
    forward, right = (
        math.cos(yaw) * north + math.sin(yaw) * east,
        -math.sin(yaw) * north + math.cos(yaw) * east,
    )
    assert forward == pytest.approx(1.0)
    assert right == pytest.approx(0.8 * 0.45 / 0.95)
    assert down == 0.0
    assert yaw_rate == 1.0
    assert mapping.normalized({1: 0.01, 0: -0.05, 3: 0.0}) == (0, 0, 0)


def test_usb_joystick_calibration_and_invalid_configuration() -> None:
    """Support measured offsets and reject bad calibration or missing axes."""
    mapping = joystick.JoystickMapping(
        centers=(0.1, -0.1, 0.0), scales=(2.0, 1.0, 1.0)
    )
    assert mapping.normalized({1: 0.1, 0: -0.1, 3: 0}) == (0, 0, 0)
    assert mapping.normalized({1: 0.6, 0: -0.1, 3: 0}) == (1, 0, 0)
    for kwargs in (
        {"axes": (0, 0, 1)}, {"signs": (0, 1, 1)},
        {"scales": (1, 0, 1)}, {"centers": (math.nan, 0, 0)},
        {"deadzone": 1.0},
    ):
        with pytest.raises(ValueError):
            joystick.JoystickMapping(**kwargs)
    with pytest.raises(KeyError):
        mapping.command({}, 0)
    with pytest.raises(ValueError):
        mapping.command({1: math.nan, 0: 0, 3: 0}, 0)
    with pytest.raises(ValueError):
        mapping.command({1: 0, 0: 0, 3: 0}, math.inf)


def test_usb_joystick_reads_initial_events_and_unchanged_sticks(
    monkeypatch,
) -> None:
    """Drain initialization, retain idle values, and expose unplug errors."""
    reader = joystick.LinuxJoystick.__new__(joystick.LinuxJoystick)
    reader.fd, reader.axis_count, reader.button_count = 123, 5, 1
    reader.axes, reader.buttons = {}, {}
    events = iter([
        joystick.JS_EVENT.pack(0, -32768, 0x82, 1),
        joystick.JS_EVENT.pack(0, 1, 0x81, 0),
    ])

    def read_event(fd, size):
        assert fd == 123 and size == 8
        try:
            return next(events)
        except StopIteration:
            raise BlockingIOError(errno.EAGAIN, "queue empty")

    monkeypatch.setattr(joystick.os, "read", read_event)
    assert reader.read_pending()
    assert reader.axes == {1: -1.0}
    assert reader.buttons == {0: True}
    assert not reader.read_pending()
    assert reader.axes == {1: -1.0}

    def disconnected(fd, size):
        raise OSError(errno.ENODEV, "USB unplugged")

    monkeypatch.setattr(joystick.os, "read", disconnected)
    with pytest.raises(OSError, match="USB unplugged"):
        reader.read_pending()


def test_usb_joystick_closes_device_after_ioctl_failure(monkeypatch) -> None:
    """Do not leak input descriptors on unsupported devices."""
    closed = []
    monkeypatch.setattr(joystick.os, "open", lambda *args: 123)
    monkeypatch.setattr(joystick.os, "close", closed.append)

    def failed_ioctl(*args):
        raise OSError(errno.ENOTTY, "not a joystick")

    monkeypatch.setattr(joystick.fcntl, "ioctl", failed_ioctl)
    with pytest.raises(OSError, match="not a joystick"):
        joystick.LinuxJoystick("/dev/input/js0")
    assert closed == [123]


def usb_control_stub():
    """Exercise control callbacks without a ROS graph or physical device."""
    node = joystick.JoystickControlNode.__new__(joystick.JoystickControlNode)
    node.mapping = joystick.JoystickMapping()
    node.reader = SimpleNamespace(axes={1: 0, 0: 0, 3: 0, 4: 1}, buttons={})
    node.deadman_axis, node.deadman_button, node.deadman_threshold = 4, -1, 0.5
    node._yaw, node._odometry_receipt_s, node.odometry_timeout_s = 0, 1, 0.2
    node._released, node._active = False, False
    node.altitude_hold_m = 0.0
    node.latch_after_stop = node._input_stop_latched = False
    node.report_raw_inputs = True
    node.show_operator_status = False
    node._ground_down_m = node._current_down_m = 0.0
    node.commands = []
    node._publish = node.commands.append
    node.get_logger = lambda: SimpleNamespace(warning=lambda value: None)
    return node


def test_usb_joystick_requires_neutral_release_and_fresh_odometry() -> None:
    """No startup movement; releasing deadman or losing yaw stops commands."""
    node = usb_control_stub()
    node._update_control(1.0)
    assert node.commands == []
    node.reader.axes[4] = -1
    node.reader.axes[1] = 1
    node._update_control(1.01)
    assert not node._released
    node.reader.axes[1] = 0
    node._update_control(1.02)
    assert node._released
    node.reader.axes[4] = 1
    node._update_control(1.03)
    node.reader.axes[1] = 1
    node._update_control(1.04)
    assert node.commands[-1] == (1, 0, 0, 0)
    node.reader.axes[4] = -1
    node._update_control(1.05)
    assert node.commands[-1] == (0, 0, 0, 0)
    assert not node._active
    count = len(node.commands)
    node._update_control(1.06)
    assert len(node.commands) == count
    node.reader.axes.update({1: 0, 4: -1})
    node._update_control(1.07)
    node.reader.axes[4] = 1
    node._update_control(1.08)
    node._update_control(1.21)
    assert node.commands[-1] == (0, 0, 0, 0)
    assert not node._active and not node._released
    node._odometry_receipt_s = 1.22
    node._update_control(1.22)
    assert not node._active


def test_usb_joystick_refuses_nonzero_deadman_activation() -> None:
    """A held nonzero stick cannot start input even after a neutral release."""
    node = usb_control_stub()
    node.reader.axes[4] = -1
    node._update_control(1.0)
    node.reader.axes.update({1: -1, 4: 1})
    node._update_control(1.01)
    assert node.commands == []
    assert not node._released


def test_usb_joystick_button_deadman_and_missing_input() -> None:
    """Support a deadman switch exposed as a USB button."""
    node = usb_control_stub()
    node.deadman_axis, node.deadman_button = -1, 0
    assert node._deadman() is None
    node.reader.buttons[0] = False
    node._update_control(1.0)
    node.reader.buttons[0] = True
    node._update_control(1.01)
    assert node._active
    del node.reader.axes[1]
    node._update_control(1.02)
    assert not node._active
    assert node.commands[-1] == (0, 0, 0, 0)


def test_usb_joystick_candidate_expiry_latches_real_mux_hold() -> None:
    """Stopping USB heartbeats must stop movement through the existing mux."""
    from uav_px4_control.control.control_mux import ControlSourceMux
    from uav_px4_control.control.control_source_models import (
        ControlCommand, HOLD, HUMAN_JOYSTICK, Vector3,
    )

    mux = ControlSourceMux()
    mapping = joystick.JoystickMapping()
    north, east, down, yaw_rate = mapping.command({1: 1, 0: 0, 3: 0}, 0)
    command = ControlCommand(
        HUMAN_JOYSTICK, 1.0, "px4_ned", Vector3(north, east, down),
        yaw_rate_radps=yaw_rate,
    )
    mux.accept_candidate(HUMAN_JOYSTICK, command, 1.0)
    assert mux.request_source(HUMAN_JOYSTICK, 1.0).accepted
    assert mux.step(1.01).active_source == HUMAN_JOYSTICK
    result = mux.step(1.21)
    assert result.active_source == HOLD
    assert result.fault_latched
    assert result.selected_command.linear == Vector3(0, 0, 0)


def test_usb_joystick_owns_only_existing_candidate_topic() -> None:
    """Keep raw input topics, PX4 commands, and source requests out of node."""
    content = text(
        PACKAGE / "uav_px4_control/control/joystick_control_node.py"
    )
    assert "SOURCE_TOPICS[HUMAN_JOYSTICK]" in content
    assert content.count("self.create_publisher(") == 1
    assert "create_client(" not in content
    assert "/fmu/in/" not in content
    assert '"/joy"' not in content


@pytest.mark.parametrize("failure", ["disconnect", "resync", "late_tick"])
def test_usb_joystick_tick_fault_stops_and_requires_release(
    monkeypatch, failure,
) -> None:
    """Handle device loss, kernel resync, and a stalled ROS input loop."""
    node = usb_control_stub()
    node.enabled = True
    node._active, node._released = True, True
    node._last_tick_s = 0.7 if failure == "late_tick" else 0.99
    node._last_report_s = 0.99
    node.reader.fd = 123
    node.errors = []
    node.get_logger = lambda: SimpleNamespace(
        warning=lambda value: None, error=node.errors.append,
        info=lambda value: None,
    )
    monkeypatch.setattr(joystick.time, "monotonic", lambda: 1.0)

    def read_pending():
        if failure == "disconnect":
            raise OSError(errno.ENODEV, "USB unplugged")
        return failure == "resync"

    node.reader.read_pending = read_pending
    node.reader.close = lambda: setattr(node.reader, "fd", None)
    node._tick()
    assert node.commands == [(0, 0, 0, 0)]
    assert not node._active and not node._released
    node._tick()
    assert len(node.commands) == 1
    if failure == "disconnect":
        assert node.reader.fd is None
        assert "USB unplugged" in node.errors[0]


@pytest.mark.parametrize("fault", ["wrong_frame", "invalid_quaternion"])
def test_usb_joystick_invalid_odometry_stops_active_input(fault) -> None:
    """Never substitute a guessed heading for invalid vehicle orientation."""
    node = usb_control_stub()
    node._active, node._released = True, True
    message = joystick.Odometry()
    message.header.frame_id = "map" if fault == "wrong_frame" else "px4_ned"
    message.pose.pose.orientation.w = 1.0 if fault == "wrong_frame" else 0.0
    node._odometry_callback(message)
    assert node._yaw is None and node._odometry_receipt_s is None
    assert node.commands == [(0, 0, 0, 0)]
    assert not node._active and not node._released


@pytest.mark.parametrize("data", [b"", b"short"])
def test_usb_joystick_refuses_partial_or_eof_events(monkeypatch, data) -> None:
    """Treat truncated device input as failure rather than stale movement."""
    reader = joystick.LinuxJoystick.__new__(joystick.LinuxJoystick)
    reader.fd = 123
    monkeypatch.setattr(joystick.os, "read", lambda *args: data)
    with pytest.raises(OSError, match="EOF or incomplete"):
        reader.read_pending()


def test_usb_joystick_altitude_feedback_preserves_ground_datum(monkeypatch):
    """A fixed target compensates vertical drift without using throttle A2."""
    node = usb_control_stub()
    node.altitude_hold_m = 1.5
    node._ground_down_m = None
    node.reader.axes[4] = -1
    monkeypatch.setattr(joystick.time, "monotonic", lambda: 1.0)
    message = joystick.Odometry()
    message.header.frame_id = "px4_ned"
    message.pose.pose.orientation.w = 1.0
    message.pose.pose.position.z = 0.3
    node._odometry_callback(message)
    node._update_control(1.0)
    node.reader.axes[4] = 1
    node._update_control(1.01)
    assert node.commands[-1] == (0, 0, -0.6, 0)
    message.pose.pose.position.z = -1.4
    node._odometry_callback(message)
    node._update_control(1.02)
    assert node._ground_down_m == 0.3
    assert node.commands[-1][2] == pytest.approx(0.2)


def test_usb_human_flight_latches_even_a_brief_deadman_release():
    """Fast switch toggling cannot conceal release from the supervisor."""
    node = usb_control_stub()
    node.latch_after_stop = True
    node._released = node._active = True
    node.reader.axes[4] = -1
    node._update_control(1.01)
    assert node.commands == [(0, 0, 0, 0)]
    node.reader.axes[4] = 1
    node._update_control(1.02)
    node._update_control(1.03)
    assert node.commands == [(0, 0, 0, 0)]
    assert not node._active


def test_usb_quiet_reporting_keeps_state_changes_and_stop_warnings():
    """Quiet flight hides raw axes while preserving operator state/faults."""
    node = usb_control_stub()
    info, warnings = [], []
    node.get_logger = lambda: SimpleNamespace(
        info=info.append, warning=warnings.append,
    )
    node.enabled = True
    node.report_raw_inputs = False
    node._reported_reason = None
    node._reason = "ready"
    node._report_inputs()
    node.reader.axes[1] = 0.5
    node._report_inputs()
    assert info == ["Control: ready"]
    node._reason = "publishing HUMAN_JOYSTICK candidates"
    node._report_inputs()
    assert len(info) == 2 and "HUMAN_JOYSTICK" in info[-1]
    node._active = True
    node._stop("USB device read failed")
    assert "USB device read failed" in warnings[-1]
    # Standalone inspection can still show the full raw axis table.
    node.report_raw_inputs = True
    node._reported_axes = None
    node._report_inputs()
    assert any("USB joystick inputs" in message for message in info)


def test_operator_panel_shows_sa_interlock_takeoff_and_landing(monkeypatch):
    """Guide a real startup order without interpreting USB as pilot control."""
    node = usb_control_stub()
    node.show_operator_status = True
    node.reader.fd = 123
    node._reported_operator_status = None
    node._flight_status_receipt_s = None
    monkeypatch.setattr(joystick.time, "monotonic", lambda: 1.0)
    status = joystick.Px4FlightStatus(
        state="DISABLED", landed=True,
        status_message="environment_valid=true",
    )
    node._flight_status_callback(status)
    info = []
    node.get_logger = lambda: SimpleNamespace(info=info.append)
    node._update_control(1.0)
    node._report_operator_status(node._reason)
    assert "SA switch : ON (+1)" in info[-1]
    assert "Move SA to -1 first" in info[-1]
    assert "Flight    : NOT STARTED" in info[-1]
    # Repeated healthy input must not flood the terminal.
    node._report_operator_status(node._reason)
    assert len(info) == 1
    node.reader.axes[4] = -1
    node._update_control(1.0)
    node._report_operator_status(node._reason)
    assert "Move SA to +1" in info[-1]
    node.reader.axes[4] = 1
    node._update_control(1.0)
    node._report_operator_status(node._reason)
    assert "READY: send the takeoff request" in info[-1]
    status.state = "HUMAN_CONTROL"
    node._report_operator_status(node._reason)
    assert "PILOT CONTROL" in info[-1] and "Fly with the sticks" in info[-1]
    status.state = "LANDING"
    node._report_operator_status(node._reason)
    assert "Wait for landing/disarm" in info[-1]
    node.reader.fd = None
    node._report_operator_status(node._reason)
    assert "DISCONNECTED" in info[-1] and "Reconnect USB/IP" in info[-1]


def test_operator_panel_explains_stale_data_before_suggesting_takeoff(monkeypatch):
    """Stale pose or supervisor status must never produce a READY prompt."""
    node = usb_control_stub()
    node.reader.fd = 123
    node._active = True
    node._reported_operator_status = None
    node._flight_status = joystick.Px4FlightStatus(
        state="DISABLED", landed=True, status_message="environment_valid=true",
    )
    node._flight_status_receipt_s = 1.0
    info = []
    node.get_logger = lambda: SimpleNamespace(info=info.append)
    monkeypatch.setattr(joystick.time, "monotonic", lambda: 1.3)
    node._report_operator_status("publishing HUMAN_JOYSTICK candidates")
    assert "Wait for PX4 position data" in info[-1]
    assert "READY: send" not in info[-1]
    node._odometry_receipt_s = 1.3
    node._flight_status_receipt_s = 0.0
    node._report_operator_status("publishing HUMAN_JOYSTICK candidates")
    assert "Wait for the flight supervisor" in info[-1]
