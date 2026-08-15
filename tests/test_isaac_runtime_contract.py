"""Static guardrails for the embedded Phase 9 Isaac adapter."""

import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
BOOTSTRAP = ROOT / "ros2_isaac_scripts" / "7.isaac_uav_bootstrap.py"
BRIDGE = ROOT / "ros2_isaac_scripts" / "8.isaac_runtime_bridge.py"


def test_embedded_runtime_scripts_parse():
    """Both --exec scripts must remain valid Python source."""
    ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    ast.parse(BRIDGE.read_text(encoding="utf-8"))


def test_runtime_bridge_has_no_dataset_or_control_boundary():
    """The embedded adapter cannot record data or command PX4."""
    source = BRIDGE.read_text(encoding="utf-8")
    forbidden = (
        "csv.writer",
        "open(",
        "MonocularCamera",
        "omni.replicator",
        "/fmu/in/",
        "VehicleCommand",
        "ControlCommand",
    )
    assert not [token for token in forbidden if token in source]


def test_bootstrap_uses_runtime_bridge_without_episode_manager_or_camera():
    """Phase 9 startup is isolated from dataset and camera lifecycle."""
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "8.isaac_runtime_bridge.py" in source
    assert "6.isaac_ros2_episode_manager.py" not in source
    assert "MonocularCamera" not in source


def test_isaac_adapter_uses_standard_ros_messages_only():
    """Isaac embedded Python must not require the custom ROS overlay."""
    source = BRIDGE.read_text(encoding="utf-8")
    assert "geometry_msgs.msg" in source
    assert "std_msgs.msg" in source
    assert "uav_interfaces" not in source
