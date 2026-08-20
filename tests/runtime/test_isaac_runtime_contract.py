"""Static guardrails for the embedded Isaac runtime adapter."""

import ast
from pathlib import Path


ROOT = Path(__file__).parents[2]
BOOTSTRAP = ROOT / "isaac" / "runtime" / "bootstrap.py"
BRIDGE = ROOT / "isaac" / "runtime" / "runtime_bridge.py"
VISUAL_QA_CAPTURE = (
    ROOT / "ros2_ws" / "src" / "uav_data_recorder" /
    "uav_data_recorder" / "visual_qa_capture.py"
)
EPISODE_SCENE_CLIENT = (
    ROOT / "ros2_ws" / "src" / "uav_data_recorder" /
    "uav_data_recorder" / "episode_scene_client.py"
)
EXPERT_DATASET_RECORDER = (
    ROOT / "ros2_ws" / "src" / "uav_data_recorder" /
    "uav_data_recorder" / "expert_dataset_recorder_node.py"
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
    """Bootstrap startup remains isolated from episode lifecycle."""
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'RUNTIME_BRIDGE_SCRIPT = SCRIPT_ROOT / "runtime_bridge.py"' in source
    assert "def create_bootstrap_scene" in source
    assert 'BOOTSTRAP_SCENE_ROOT = "/World/BootstrapScene"' in source
    assert "6.isaac_ros2_episode_manager.py" not in source
    assert "MonocularCamera" not in source


def test_fpv_camera_is_explicitly_opt_in_with_legacy_alias():
    """The bootstrap runtime must not pay camera cost unless requested."""
    source = BRIDGE.read_text(encoding="utf-8")
    assert 'os.environ.get("UAV_FPV_CAMERA", "0") == "1"' in source
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


def test_episode_scene_commands_do_not_control_or_teleport_vehicle():
    """Scene resets may alter scene USD only after a landed-height check."""
    source = BRIDGE.read_text(encoding="utf-8")
    assert "/uav/isaac/episode_command" in source
    assert "vehicle must be landed before scene reset" in source
    assert "SetWorldPose" not in source
    assert "set_world_pose" not in source
    assert "/fmu/in/" not in source


def test_expert_sensor_contract_is_opt_in_and_storage_free():
    """The bridge publishes expert sensors but never writes data."""
    source = BRIDGE.read_text(encoding="utf-8")
    assert 'os.environ.get("UAV_EXPERT_SENSORS", "0") == "1"' in source
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
    assert '"fpv_rgb_ready"' in source
    assert '"observer_rgb_ready"' in source
    assert '"fpv_depth_ready"' in source
    assert "write_bytes" not in source


def test_semantic_runtime_status_preserves_dataset_compatibility_aliases():
    """New runtime keys map to the unchanged dataset evidence contract."""
    bridge = BRIDGE.read_text(encoding="utf-8")
    recorder = EXPERT_DATASET_RECORDER.read_text(encoding="utf-8")
    for key in ("fpv_rgb_ready", "observer_rgb_ready", "fpv_depth_ready"):
        assert f'"{key}"' in bridge
    assert '"fpv_rgb_ready": "phase10a_camera_ready"' in recorder
    assert '"observer_rgb_ready": "phase10c_observer_rgb_ready"' in recorder
    assert '"fpv_depth_ready": "phase10b_fpv_depth_ready"' in recorder


def test_episode_scene_client_uses_semantic_runtime_identity():
    """The active client node and result markers describe their function."""
    source = EPISODE_SCENE_CLIENT.read_text(encoding="utf-8")
    assert 'super().__init__("episode_scene_client")' in source
    assert "EXPERT_SCENE_READY" in source
    assert "EXPERT_SCENE_RESULT" in source


def test_runtime_creates_canonical_cylinders_and_lights():
    """USD application owns cylinders while planner sees one obstacle each."""
    source = BRIDGE.read_text(encoding="utf-8")
    apply_scene = source.split("def _apply_scene", 1)[1].split(
        "def _update_camera_pose", 1
    )[0]
    cylinder_helper = source.split(
        "def _create_cylinder_obstacle", 1
    )[1].split("def _create_episode_lighting", 1)[0]
    assert 'SCENE_ROOT = "/World/GeneratedEpisode"' in source
    assert "self._create_cylinder_obstacle" in apply_scene
    assert '"episode:shape"' in apply_scene
    assert '"cylinder"' in apply_scene
    assert "UsdGeom.Cylinder.Define" in cylinder_helper
    assert "CreateRadiusAttr" in cylinder_helper
    assert "CreateHeightAttr" in cylinder_helper
    assert 'source["color"]' in cylinder_helper
    assert "UsdPhysics.CollisionAPI.Apply(prim)" in cylinder_helper
    assert "UsdLux.DomeLight.Define" in apply_scene
    assert "UsdLux.DistantLight.Define" in apply_scene


def test_visual_qa_capture_is_read_only_and_collects_three_flight_phases():
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
