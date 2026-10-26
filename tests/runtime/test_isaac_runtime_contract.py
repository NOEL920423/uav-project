"""Static guardrails for the embedded Isaac runtime adapter."""

import ast
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from PIL import Image
import pytest

from isaac.runtime.formal_expert_sensor_contract import (
    FORMAL_RGB_EXPECTED_RATE_RANGE_HZ,
    FORMAL_RGB_NOMINAL_RATE_HZ,
    FORMAL_RGB_PUBLISH_PERIOD_S,
    LEGACY_OBSERVER_RGB_PUBLISH_PERIOD_S,
    TOP_RGB_HEIGHT,
    TOP_RGB_PUBLISH_PERIOD_S,
    TOP_RGB_WIDTH,
    TOP_VIDEO_JPEG_QUALITY,
    TOP_VIDEO_RGB_HEIGHT,
    TOP_VIDEO_RGB_WIDTH,
)


ROOT = Path(__file__).parents[2]
BOOTSTRAP = ROOT / "isaac" / "runtime" / "bootstrap.py"
BRIDGE = ROOT / "isaac" / "runtime" / "environment.py"
SENSOR_CONTRACT = (
    ROOT / "isaac" / "runtime" / "formal_expert_sensor_contract.py"
)
EPISODE_SCENE_CLIENT = (
    ROOT / "ros2_ws" / "src" / "uav_data_recorder" /
    "uav_data_recorder" / "episode_scene_client.py"
)
EXPERT_DATASET_RECORDER = (
    ROOT / "ros2_ws" / "src" / "uav_data_recorder" /
    "uav_data_recorder" / "expert_dataset_recorder_node.py"
)


def _bridge_method(name, **scope):
    """Run an existing runtime method without loading Isaac or ROS."""
    tree = ast.parse(BRIDGE.read_text(encoding="utf-8"))
    cls = next(node for node in ast.walk(tree)
               if isinstance(node, ast.ClassDef) and node.name == "IsaacRuntimeBridge")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(BRIDGE), "exec"), scope)
    return scope[name]


def test_recording_jpeg_preserves_model_bytes_and_adds_high_resolution_top():
    """The standard camera encoder stays byte-identical to its prior defaults."""
    message = lambda: SimpleNamespace(header=SimpleNamespace())
    encode = _bridge_method("_jpeg_message", CAMERA_WIDTH=320, CAMERA_HEIGHT=180,
                            JPEG_QUALITY=85, BytesIO=BytesIO, CompressedImage=message)
    for size in ((320, 180), (640, 640)):
        pixels = np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(size[1], size[0], 3)
        previous = BytesIO()
        Image.fromarray(pixels).save(previous, format="JPEG", quality=85)
        actual = encode(pixels, "stamp", "optical", size)
        assert actual.data == previous.getvalue()
        assert actual.header.stamp == "stamp"
    size = (TOP_VIDEO_RGB_WIDTH, TOP_VIDEO_RGB_HEIGHT)
    video = encode(np.zeros((size[1], size[0], 3), dtype=np.uint8), "stamp", "optical", size,
                   quality=TOP_VIDEO_JPEG_QUALITY, subsampling=0)
    from PIL.JpegImagePlugin import get_sampling
    with Image.open(BytesIO(video.data)) as image:
        assert image.size == (1024, 1024)
        assert get_sampling(image) == 0


def test_recording_render_product_is_dedicated_and_released_on_stop():
    """Only the new recording product is destroyed by its lifecycle owner."""
    source = BRIDGE.read_text(encoding="utf-8")
    setup = source.split("if self._observer_video_enabled:", 1)[1].split(
        "if self._observer_viewport_requested:", 1
    )[0]
    assert "force_new=True" in setup
    assert "(TOP_VIDEO_RGB_WIDTH, TOP_VIDEO_RGB_HEIGHT)" in setup
    calls = []
    owner = SimpleNamespace(
        _stopped=False, _subscription=None, _rgb_annotator=None, _observer_rgb_annotator=None,
        _observer_video_rgb_annotator=SimpleNamespace(detach=lambda: calls.append("detach")),
        _observer_video_render_product=SimpleNamespace(destroy=lambda: calls.append("destroy")),
        _depth_annotator=None, _owns_rclpy=False,
        _timing=SimpleNamespace(close=lambda: calls.append("timing-close")),
        _node=SimpleNamespace(destroy_node=lambda: calls.append("node-destroy")),
    )
    _bridge_method("stop")(owner)
    assert calls == ["detach", "destroy", "timing-close", "node-destroy"]
    assert owner._observer_video_render_product is None
    assert owner._observer_video_rgb_annotator is None


@pytest.mark.parametrize("enabled, failed", [(False, False), (True, False), (True, True)])
def test_recording_top_publishes_on_model_tick_without_changing_model_streams(enabled, failed):
    """Optional video failures preserve the model frames and original reason."""
    from isaac.runtime.formal_expert_sensor_contract import TOP_VIDEO_TOPIC

    sent, errors = {}, []

    def publisher(name):
        sent[name] = []
        return SimpleNamespace(publish=sent[name].append)

    def encode(data, stamp, frame_id, expected_size=(320, 180), **options):
        return {"data": data, "stamp": stamp, "size": expected_size, **options}

    def video_data():
        if failed:
            raise RuntimeError("video annotator unavailable")
        return b"video-top"

    owner = SimpleNamespace(
        _camera_enabled=True, _last_camera_publish_monotonic=0,
        _camera_frame_count=0, _camera_error="", _expert_sensors_enabled=True,
        _last_observer_publish_monotonic=0, _observer_publish_period_s=0.2,
        _observer_resolution=(640, 640), _observer_frame_count=0,
        _observer_camera_error="", _last_depth_publish_monotonic=1.0,
        _observer_video_enabled=enabled, _observer_video_error="warming",
        _observer_video_frame_count=0, _update_camera_pose=lambda: True,
        _rgb_annotator=SimpleNamespace(get_data=lambda: b"fpv"),
        _observer_rgb_annotator=SimpleNamespace(get_data=lambda: b"model-top"),
        _observer_video_rgb_annotator=SimpleNamespace(get_data=video_data),
        _camera_publisher=publisher("fpv"), _observer_camera_publisher=publisher("model-top"),
        _observer_video_publisher=publisher("video-top"), _jpeg_message=encode,
        _timing=SimpleNamespace(record=lambda *args, **kwargs: None,
                                publish=lambda *args, **kwargs: None),
        _node=SimpleNamespace(get_logger=lambda: SimpleNamespace(error=errors.append)),
    )
    publish = _bridge_method(
        "_publish_camera", time=time, CAMERA_PUBLISH_PERIOD_S=0.2,
        CAMERA_TOPIC="fpv", OBSERVER_CAMERA_TOPIC="model-top", DEPTH_PUBLISH_PERIOD_S=0.2,
        TOP_VIDEO_TOPIC=TOP_VIDEO_TOPIC, TOP_VIDEO_RGB_WIDTH=TOP_VIDEO_RGB_WIDTH,
        TOP_VIDEO_RGB_HEIGHT=TOP_VIDEO_RGB_HEIGHT, TOP_VIDEO_JPEG_QUALITY=TOP_VIDEO_JPEG_QUALITY,
    )
    publish(owner, "same-tick", 1.0)
    assert [sent[key][0]["size"] for key in ("fpv", "model-top")] == [(320, 180), (640, 640)]
    assert all(sent[key][0]["stamp"] == "same-tick" for key in ("fpv", "model-top"))
    assert owner._camera_error == owner._observer_camera_error == ""
    if enabled and not failed:
        assert sent["video-top"] == [{"data": b"video-top", "stamp": "same-tick",
                                      "size": (1024, 1024), "quality": 95, "subsampling": 0}]
    else:
        assert not sent["video-top"]
    if failed:
        assert owner._observer_video_error == "RuntimeError: video annotator unavailable"
        assert errors == ["TOP video camera failed: RuntimeError: video annotator unavailable"]


def test_embedded_runtime_scripts_parse():
    """Both --exec scripts must remain valid Python source."""
    ast.parse(BOOTSTRAP.read_text(encoding="utf-8"))
    ast.parse(BRIDGE.read_text(encoding="utf-8"))
    ast.parse(SENSOR_CONTRACT.read_text(encoding="utf-8"))


def test_formal_sensor_contract_is_dependency_free_and_self_consistent():
    """Ordinary Python tools can share the embedded sensor constants."""
    tree = ast.parse(SENSOR_CONTRACT.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    )
    assert not imports
    assert TOP_RGB_WIDTH > 0
    assert TOP_RGB_HEIGHT > 0
    assert TOP_RGB_PUBLISH_PERIOD_S == FORMAL_RGB_PUBLISH_PERIOD_S
    assert LEGACY_OBSERVER_RGB_PUBLISH_PERIOD_S != TOP_RGB_PUBLISH_PERIOD_S
    assert (
        FORMAL_RGB_EXPECTED_RATE_RANGE_HZ[0]
        < FORMAL_RGB_NOMINAL_RATE_HZ
        < FORMAL_RGB_EXPECTED_RATE_RANGE_HZ[1]
    )


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


def test_bootstrap_uses_environment_without_episode_manager():
    """Bootstrap startup remains isolated from episode lifecycle."""
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'RUNTIME_BRIDGE_SCRIPT = SCRIPT_ROOT / "environment.py"' in source
    assert "def create_bootstrap_scene" in source
    assert 'BOOTSTRAP_SCENE_ROOT = "/World/BootstrapScene"' in source
    assert "6.isaac_ros2_episode_manager.py" not in source
    assert "MonocularCamera" not in source


def test_fpv_camera_is_explicitly_opt_in():
    """The bootstrap runtime must not pay camera cost unless requested."""
    source = BRIDGE.read_text(encoding="utf-8")
    assert 'os.environ.get("UAV_FPV_CAMERA", "0") == "1"' in source
    assert "UAV_" + "PHASE" not in source
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
    assert "UAV_" + "PHASE" not in source
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


def test_formal_observer_is_fixed_orthographic_without_changing_fpv():
    """Formal TOP is global while legacy observer and FPV remain unchanged."""
    source = BRIDGE.read_text(encoding="utf-8")
    setup = source.split("def _setup_camera", 1)[1].split(
        "def _episode_command_callback", 1
    )[0]
    pose_update = source.split("def _update_camera_pose", 1)[1].split(
        "def _smooth_position", 1
    )[0]

    assert "CAMERA_WIDTH = FPV_RGB_WIDTH" in source
    assert "CAMERA_HEIGHT = FPV_RGB_HEIGHT" in source
    assert "(TOP_RGB_WIDTH, TOP_RGB_HEIGHT)" in source
    assert "TOP_RGB_MODE" in source
    assert "TOP_RGB_PUBLISH_PERIOD_S" in source
    assert "LEGACY_OBSERVER_RGB_PUBLISH_PERIOD_S" in source
    assert "self._observer_publish_period_s" in source
    assert "FORMAL_OBSERVER_CAMERA_WIDTH" not in source
    assert "FORMAL_OBSERVER_CAMERA_HEIGHT" not in source
    assert "OBSERVER_CAMERA_PUBLISH_PERIOD_S" not in source
    assert "FORMAL_OBSERVER_EYE = (0.0, 2.5, 15.0)" in source
    assert "FORMAL_OBSERVER_TARGET = (0.0, 2.5, 0.0)" in source
    assert "FORMAL_OBSERVER_UP = (0.0, 1.0, 0.0)" in source
    assert "FORMAL_OBSERVER_COVERAGE_M = TOP_RGB_COVERAGE_M" in source
    assert "UsdGeom.GetStageMetersPerUnit(stage)" in source
    assert "Gf.Camera.APERTURE_UNIT" in source
    assert "UsdGeom.Tokens.orthographic" in setup
    assert "UsdGeom.Tokens.perspective" in setup
    assert "OBSERVER_FOCAL_LENGTH" in setup
    assert "OBSERVER_HORIZONTAL_APERTURE" in setup
    assert "self._observer_resolution" in setup
    assert "if self._formal_expert_sensors_enabled:" in pose_update
    assert "Gf.Vec3d(*FORMAL_OBSERVER_EYE)" in pose_update
    assert "Gf.Vec3d(*FORMAL_OBSERVER_TARGET)" in pose_update
    assert "Gf.Vec3d(*FORMAL_OBSERVER_UP)" in pose_update
    assert "if not self._formal_expert_sensors_enabled:" in pose_update
    assert "self._smooth_position" in pose_update

    fpv_setup = setup.split("if self._expert_sensors_enabled:", 1)[0]
    fpv_pose = pose_update.split(
        "if self._observer_camera_transform is not None:", 1
    )[0]
    assert "FORMAL_OBSERVER" not in fpv_setup
    assert "FORMAL_OBSERVER" not in fpv_pose
    assert "(CAMERA_WIDTH, CAMERA_HEIGHT)" in fpv_setup
    assert "self._fpv_camera_position = fpv_eye" in fpv_pose


def test_semantic_runtime_status_is_recorded_directly():
    """The recorder stores semantic runtime status keys unchanged."""
    bridge = BRIDGE.read_text(encoding="utf-8")
    recorder = EXPERT_DATASET_RECORDER.read_text(encoding="utf-8")
    for key in ("fpv_rgb_ready", "observer_rgb_ready", "fpv_depth_ready"):
        assert f'"{key}"' in bridge
    assert '"fpv_rgb_ready": "fpv_rgb_ready"' in recorder
    assert '"observer_rgb_ready": "observer_rgb_ready"' in recorder
    assert '"fpv_depth_ready": "fpv_depth_ready"' in recorder


def test_formal_dataset_uses_explicit_storage_directories():
    """Storage names change without renaming the observer stream contract."""
    source = EXPERT_DATASET_RECORDER.read_text(encoding="utf-8")
    ast.parse(source)
    assert 'self.fpv_rgb_dir = self.episode_dir / "fpv_rgb"' in source
    assert '"observer_rgb", "top_rgb", self._observer_images' in source
    assert '"observer_rgb_available"' in source
    assert '"observer_rgb_path"' in source
    assert '"observer_rgb": {' in source


def test_recorder_uses_causal_inclusive_exclusive_flight_window():
    """BC writes stop at the flight boundary without stopping evidence."""
    source = EXPERT_DATASET_RECORDER.read_text(encoding="utf-8")
    callback = source.split("def _flight_callback", 1)[1].split(
        "def _goal_callback", 1
    )[0]
    process = source.split("def _process_image", 1)[1].split(
        "def _record_auxiliary", 1
    )[0]
    finalize = source.split("def finalize", 1)[1].split(
        "def destroy_node", 1
    )[0]

    assert "update_recording_window" in callback
    assert "self._recording_start_timestamp_s" in callback
    assert "self._recording_end_timestamp_s" in callback
    assert "self._observed_goal_reached" in callback
    assert "self._observed_landing_commanded" in callback
    assert "self._observed_landed_after_landing" in callback

    gate = process.index("recording_window_rejection")
    primary_write = process.index("output.write_bytes")
    auxiliary_write = process.index("self._record_auxiliary")
    assert gate < primary_write < auxiliary_write
    assert "state = nearest(" in process
    assert "action = nearest(" in process
    assert "mux = nearest(" in process
    assert "flight = latest_at_or_before(" in process
    assert "prior_action = previous(" in process
    assert process.count("self._record_auxiliary(") == 1

    assert 'status.state == "COMPLETE"' in finalize
    assert '"goal_reached": self._observed_goal_reached' in finalize
    assert '"landing_commanded": self._observed_landing_commanded' in finalize
    assert "landed_after_landing_command" in finalize


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
    assert 'self._set_display_color(prim, source["color"])' in cylinder_helper
    assert 'materials["obstacle"]' in apply_scene
    assert "bind_material(prim, material)" in cylinder_helper
    assert "UsdPhysics.CollisionAPI.Apply(prim)" in cylinder_helper
    assert 'self._create_episode_lighting(scene["lighting"])' in apply_scene
    lighting_helper = source.split(
        "def _create_episode_lighting", 1
    )[1].split("def _update_camera_pose", 1)[0]
    assert "UsdLux.DomeLight.Define" in lighting_helper
    assert "UsdLux.DistantLight.Define" not in lighting_helper
