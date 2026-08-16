"""Static guardrails for the embedded Phase 9 Isaac adapter."""

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
BOOTSTRAP = ROOT / "isaac" / "runtime" / "bootstrap.py"
BRIDGE = ROOT / "isaac" / "runtime" / "runtime_bridge.py"
VISUAL_QA_CAPTURE = (
    ROOT / "ros2_ws" / "src" / "uav_data_recorder" /
    "uav_data_recorder" / "visual_qa_capture.py"
)


def test_embedded_runtime_scripts_parse():
    """Both --exec scripts must remain valid Python source."""
    ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    ast.parse(BRIDGE.read_text(encoding="utf-8"))


def test_runtime_bridge_has_no_storage_or_control_boundary():
    """The embedded adapter cannot write datasets or command PX4."""
    source = BRIDGE.read_text(encoding="utf-8")
    forbidden = (
        "csv.writer",
        "open(",
        "MonocularCamera",
        "/fmu/in/",
        "VehicleCommand",
        "ControlCommand",
    )
    assert not [token for token in forbidden if token in source]


def test_bootstrap_uses_runtime_bridge_without_episode_manager():
    """Phase 9 startup remains isolated from episode lifecycle."""
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'RUNTIME_BRIDGE_SCRIPT = SCRIPT_ROOT / "runtime_bridge.py"' in source
    assert "6.isaac_ros2_episode_manager.py" not in source
    assert "MonocularCamera" not in source


def test_phase10a_camera_is_explicitly_opt_in():
    """Default Phase 9 runtime must not pay the camera rendering cost."""
    source = BRIDGE.read_text(encoding="utf-8")
    assert 'os.environ.get("UAV_PHASE10A_CAMERA", "0") == "1"' in source
    assert "/uav/isaac/fpv/image/compressed" in source
    assert "JPEG_QUALITY = 85" in source


def test_isaac_adapter_uses_standard_ros_messages_only():
    """Isaac embedded Python must not require the custom ROS overlay."""
    source = BRIDGE.read_text(encoding="utf-8")
    assert "geometry_msgs.msg" in source
    assert "std_msgs.msg" in source
    assert "sensor_msgs.msg" in source
    assert "uav_interfaces" not in source


def test_phase10b_scene_commands_do_not_control_or_teleport_vehicle():
    """Scene resets may alter scene USD only after a landed-height check."""
    source = BRIDGE.read_text(encoding="utf-8")
    assert "/uav/isaac/episode_command" in source
    assert "vehicle must be landed before scene reset" in source
    assert "SetWorldPose" not in source
    assert "set_world_pose" not in source
    assert "/fmu/in/" not in source


def test_phase10c_recovered_camera_contract_is_opt_in_and_storage_free():
    """The bridge publishes legacy FPV/Observer sensors but never writes data."""
    source = BRIDGE.read_text(encoding="utf-8")
    assert 'os.environ.get("UAV_PHASE10B_SENSORS", "0") == "1"' in source
    assert "/uav/isaac/observer/image/compressed" in source
    assert "/uav/isaac/fpv/depth/compressed" in source
    assert "unit=millimeter" in source
    assert "FPV_FOCAL_LENGTH = 12.0" in source
    assert "FPV_HORIZONTAL_APERTURE = 28.0" in source
    assert "FPV_LOOK_DOWN_M = -0.8" in source
    assert 'OBSERVER_MODE = "TOP"' in source
    assert "OBSERVER_TOP_HEIGHT_M = 9.0" in source
    assert "self._fpv_camera_position = fpv_eye" in source
    assert "write_bytes" not in source


def test_phase10c_runtime_creates_canonical_highrise_hierarchy_and_lights():
    """USD application owns canonical buildings while planner sees one each."""
    source = BRIDGE.read_text(encoding="utf-8")
    apply_scene = source.split("def _apply_scene", 1)[1].split(
        "def _update_camera_pose", 1
    )[0]
    assert 'SCENE_ROOT = "/World/GeneratedEpisode"' in source
    assert "UsdGeom.Cube.Define" in apply_scene
    assert "UsdGeom.Cylinder.Define" in apply_scene
    assert "/Body" in apply_scene
    assert "/Windows" in apply_scene
    assert "/Roof" in apply_scene
    assert "/Crown" in apply_scene
    assert "/Antenna" in apply_scene
    assert "UsdLux.DomeLight.Define" in apply_scene
    assert "UsdLux.DistantLight.Define" in apply_scene
    assert "collision=True" in apply_scene


def test_phase10c_capture_is_read_only_and_collects_three_flight_phases():
    """Visual QA must observe path/images/status without commanding flight."""
    source = VISUAL_QA_CAPTURE.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'CAPTURE_PHASES = ("start", "mid_flight", "near_goal")' in source
    assert 'PATH_TOPIC = "/uav/planner/path"' in source
    assert '"look_down_m": -0.8' in source
    assert '"mode": "TOP"' in source
    assert '"position_smoothing": "disabled_rigid_body_mount"' in source
    assert "create_publisher" not in source
    assert "/fmu/in/" not in source
