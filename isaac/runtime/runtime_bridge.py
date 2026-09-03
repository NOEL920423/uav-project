#!/usr/bin/env python3
"""Publish deterministic Isaac runtime evidence without recording datasets.

This script runs inside Isaac Sim after Pegasus has created the vehicle and
started the physics timeline.  It intentionally uses only standard ROS 2
messages so Isaac's embedded Python does not need the project's custom message
overlay.
"""

from __future__ import annotations

import builtins
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
import time

from geometry_msgs.msg import PoseStamped

import omni.kit.app
import omni.timeline
import omni.usd

import rclpy

from std_msgs.msg import String

from sensor_msgs.msg import CompressedImage

from pxr import Gf, UsdGeom, UsdLux, UsdPhysics


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from episode_scene import generate_episode_scene
from formal_expert_sensor_contract import (
    FORMAL_RGB_PUBLISH_PERIOD_S,
    FPV_RGB_HEIGHT,
    FPV_RGB_WIDTH,
    LEGACY_OBSERVER_RGB_HEIGHT,
    LEGACY_OBSERVER_RGB_PUBLISH_PERIOD_S,
    LEGACY_OBSERVER_RGB_WIDTH,
    TOP_RGB_HEIGHT,
    TOP_RGB_MODE,
    TOP_RGB_PUBLISH_PERIOD_S,
    TOP_RGB_WIDTH,
)
from scene_visual_materials import (
    FLOOR_COLOR,
    GOAL_MARKER_COLOR,
    OBSTACLE_COLOR,
    START_MARKER_COLOR,
    bind_material,
    create_scene_materials,
)


VEHICLE_BODY_PATH = "/World/quadrotor/body"
POSE_TOPIC = "/isaac_uav/pose"
STATUS_TOPIC = "/uav/isaac/runtime_status"
FRAME_ID = "isaac_world"
SCHEMA = "uav_isaac_runtime/v1"
BOOTSTRAP_SCENE_ID = "bootstrap_fixed_scene_v1"
SCENE_REVISION = 1
PUBLISH_PERIOD_S = 0.05
CAMERA_PUBLISH_PERIOD_S = FORMAL_RGB_PUBLISH_PERIOD_S
CAMERA_TOPIC = "/uav/isaac/fpv/image/compressed"
CAMERA_PATH = "/World/RuntimeSensors/FPVCamera"
OBSERVER_CAMERA_PATH = "/World/RuntimeSensors/ObserverCamera"
OBSERVER_CAMERA_TOPIC = "/uav/isaac/observer/image/compressed"
DEPTH_TOPIC = "/uav/isaac/fpv/depth/compressed"
EPISODE_COMMAND_TOPIC = "/uav/isaac/episode_command"
BOOTSTRAP_SCENE_ROOT = "/World/BootstrapScene"
SCENE_ROOT = "/World/GeneratedEpisode"
CAMERA_WIDTH = FPV_RGB_WIDTH
CAMERA_HEIGHT = FPV_RGB_HEIGHT
MSG_WEBRTC_VIEWPORT = (
    "[IsaacRuntimeBridge] WebRTC viewport uses the fixed TOP camera."
)

JPEG_QUALITY = 85
DEPTH_PUBLISH_PERIOD_S = 0.20
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 30.0
FPV_FORWARD_OFFSET_M = 0.45
FPV_HEIGHT_M = 0.12
FPV_LOOK_AHEAD_M = 3.5
FPV_LOOK_DOWN_M = -0.8
FPV_FOCAL_LENGTH = 12.0
FPV_HORIZONTAL_APERTURE = 28.0
OBSERVER_MODE = "TOP"
OBSERVER_BACK_DISTANCE_M = 3.2
OBSERVER_HEIGHT_M = 5.2
OBSERVER_SIDE_OFFSET_M = 2.2
OBSERVER_LOOK_AHEAD_M = 2.5
OBSERVER_LOOK_AT_HEIGHT_M = -1.2
OBSERVER_TOP_HEIGHT_M = 9.0
OBSERVER_TOP_LOOK_AT_HEIGHT_M = 0.0
OBSERVER_FOCAL_LENGTH = 18.0
OBSERVER_HORIZONTAL_APERTURE = 22.0

# 固定的上視圖影像
FORMAL_OBSERVER_EYE = (0.0, 2.5, 15.0)
FORMAL_OBSERVER_TARGET = (0.0, 2.5, 0.0)
FORMAL_OBSERVER_UP = (0.0, 1.0, 0.0)
FORMAL_OBSERVER_COVERAGE_M = (20.0, 11.25)
CAMERA_CLIPPING_RANGE = (0.05, 10000.0)
CAMERA_SMOOTHING = 0.18
GOAL = (0.5, 3.0, 1.5)
OBSTACLES = ({
    "name": "BootstrapObstacle_001",
    "x": -1.5,
    "y": 1.5,
    "z": 1.25,
    "radius": 0.43,
    "height": 2.5,
},)


def _world_pose(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    matrix = omni.usd.get_world_transform_matrix(prim)
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotation().GetQuat()
    imaginary = rotation.GetImaginary()
    values = (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
        float(rotation.GetReal()),
    )
    return values if all(math.isfinite(value) for value in values) else None


def _orthographic_aperture(stage, coverage_m):
    """Convert a world-space coverage in metres to USD camera aperture units."""
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise RuntimeError("stage meters per unit must be finite and positive")
    return (
        float(coverage_m)
        / meters_per_unit
        / float(Gf.Camera.APERTURE_UNIT)
    )


class IsaacRuntimeBridge:
    """Own one update callback and publish actual stage state at 20 Hz."""

    def __init__(self):
        self._stage = omni.usd.get_context().get_stage()
        if self._stage is None:
            raise RuntimeError("Isaac Sim has no active USD stage")
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)
        self._node = rclpy.create_node("isaac_runtime_bridge")
        self._pose_publisher = self._node.create_publisher(
            PoseStamped, POSE_TOPIC, 10
        )
        self._status_publisher = self._node.create_publisher(
            String, STATUS_TOPIC, 10
        )
        self._camera_enabled = (
            os.environ.get("UAV_FPV_CAMERA", "0") == "1"
            or os.environ.get("UAV_PHASE10A_CAMERA", "0") == "1"
        )
        self._formal_expert_sensors_enabled = (
            os.environ.get("UAV_EXPERT_SENSORS", "0") == "1"
        )
        self._legacy_expert_sensors_enabled = (
            os.environ.get("UAV_PHASE10B_SENSORS", "0") == "1"
        )
        self._expert_sensors_enabled = (
            self._formal_expert_sensors_enabled
            or self._legacy_expert_sensors_enabled
        )
        self._observer_resolution = (
            (TOP_RGB_WIDTH, TOP_RGB_HEIGHT)
            if self._formal_expert_sensors_enabled
            else (LEGACY_OBSERVER_RGB_WIDTH, LEGACY_OBSERVER_RGB_HEIGHT)
        )
        self._observer_publish_period_s = (
            TOP_RGB_PUBLISH_PERIOD_S
            if self._formal_expert_sensors_enabled
            else LEGACY_OBSERVER_RGB_PUBLISH_PERIOD_S
        )
        self._observer_mode = (
            TOP_RGB_MODE
            if self._formal_expert_sensors_enabled
            else OBSERVER_MODE.lower()
        )
        self._camera_enabled = (
            self._camera_enabled or self._expert_sensors_enabled
        )
        self._scene_id = BOOTSTRAP_SCENE_ID
        self._scene_revision = SCENE_REVISION
        self._goal = GOAL
        self._obstacles = list(OBSTACLES)
        self._episode_id = ""
        self._random_seed = None
        self._scene_configuration = None
        self._scene_camera_boundary = None
        self._episode_command_error = ""
        self._camera_publisher = None
        self._observer_camera_publisher = None
        self._depth_publisher = None
        self._camera_transform = None
        self._observer_camera_transform = None
        self._rgb_annotator = None
        self._observer_rgb_annotator = None
        self._depth_annotator = None
        self._render_product = None
        self._observer_render_product = None
        self._last_camera_publish_monotonic = 0.0
        self._last_observer_publish_monotonic = 0.0
        self._last_depth_publish_monotonic = 0.0
        self._camera_frame_count = 0
        self._observer_frame_count = 0
        self._depth_frame_count = 0
        self._camera_error = "disabled"
        self._observer_camera_error = "disabled"
        self._depth_error = "disabled"
        self._fpv_camera_position = None
        self._observer_camera_position = None
        self._observer_viewport_requested = (
            os.environ.get("UAV_OBSERVER_VIEWPORT", "0") == "1"
        )
        self._observer_viewport_selected = False
        if self._camera_enabled:
            self._setup_camera()
        self._episode_command_subscription = self._node.create_subscription(
            String, EPISODE_COMMAND_TOPIC, self._episode_command_callback, 10
        )
        self._sequence = 0
        self._last_publish_monotonic = 0.0
        self._stopped = False
        self._subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update,
                name="IsaacRuntimeBridgeUpdate",
            )
        )
        print(
            "[IsaacRuntimeBridge] Started: pose/status at 20 Hz, "
            f"FPV camera={'enabled' if self._camera_enabled else 'disabled'}, "
            "expert sensors="
            f"{'enabled' if self._expert_sensors_enabled else 'disabled'}"
        )

    def _setup_camera(self):
        import omni.replicator.core as rep

        existing = self._stage.GetPrimAtPath(CAMERA_PATH)
        if existing and existing.IsValid():
            self._stage.RemovePrim(CAMERA_PATH)
        camera = UsdGeom.Camera.Define(self._stage, CAMERA_PATH)
        camera.GetFocalLengthAttr().Set(FPV_FOCAL_LENGTH)
        camera.GetHorizontalApertureAttr().Set(FPV_HORIZONTAL_APERTURE)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(*CAMERA_CLIPPING_RANGE))
        self._camera_transform = UsdGeom.Xformable(
            camera.GetPrim()
        ).AddTransformOp()
        self._render_product = rep.create.render_product(
            CAMERA_PATH, (CAMERA_WIDTH, CAMERA_HEIGHT)
        )
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annotator.attach([self._render_product])
        self._camera_publisher = self._node.create_publisher(
            CompressedImage, CAMERA_TOPIC, 10
        )
        self._camera_error = "warming"
        if self._expert_sensors_enabled:
            existing_observer = self._stage.GetPrimAtPath(
                OBSERVER_CAMERA_PATH
            )
            if existing_observer and existing_observer.IsValid():
                self._stage.RemovePrim(OBSERVER_CAMERA_PATH)
            observer_camera = UsdGeom.Camera.Define(
                self._stage, OBSERVER_CAMERA_PATH
            )
            if self._formal_expert_sensors_enabled:
                observer_camera.GetProjectionAttr().Set(
                    UsdGeom.Tokens.orthographic
                )
                observer_camera.GetHorizontalApertureAttr().Set(
                    _orthographic_aperture(
                        self._stage, FORMAL_OBSERVER_COVERAGE_M[0]
                    )
                )
                observer_camera.GetVerticalApertureAttr().Set(
                    _orthographic_aperture(
                        self._stage, FORMAL_OBSERVER_COVERAGE_M[1]
                    )
                )
            else:
                observer_camera.GetProjectionAttr().Set(
                    UsdGeom.Tokens.perspective
                )
                observer_camera.GetFocalLengthAttr().Set(
                    OBSERVER_FOCAL_LENGTH
                )
                observer_camera.GetHorizontalApertureAttr().Set(
                    OBSERVER_HORIZONTAL_APERTURE
                )
            observer_camera.GetClippingRangeAttr().Set(
                Gf.Vec2f(*CAMERA_CLIPPING_RANGE)
            )
            self._observer_camera_transform = UsdGeom.Xformable(
                observer_camera.GetPrim()
            ).AddTransformOp()
            self._observer_render_product = rep.create.render_product(
                OBSERVER_CAMERA_PATH, self._observer_resolution
            )
            self._observer_rgb_annotator = (
                rep.AnnotatorRegistry.get_annotator("rgb")
            )
            self._observer_rgb_annotator.attach([
                self._observer_render_product
            ])
            self._depth_annotator = rep.AnnotatorRegistry.get_annotator(
                "distance_to_camera"
            )
            self._depth_annotator.attach([self._render_product])
            self._observer_camera_publisher = self._node.create_publisher(
                CompressedImage, OBSERVER_CAMERA_TOPIC, 10
            )
            self._depth_publisher = self._node.create_publisher(
                CompressedImage, DEPTH_TOPIC, 10
            )
            self._observer_camera_error = "warming"
            self._depth_error = "warming"
            if self._observer_viewport_requested:
                self._select_observer_viewport()
        print(
            f"[IsaacRuntimeBridge] FPV render product: "
            f"{CAMERA_WIDTH}x{CAMERA_HEIGHT} JPEG quality {JPEG_QUALITY}"
        )

    def _select_observer_viewport(self):
        """Show the fixed observer camera without changing its render product."""
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is None:
            return False
        viewport.set_active_camera(OBSERVER_CAMERA_PATH)
        if not self._observer_viewport_selected:
            print(MSG_WEBRTC_VIEWPORT)
        self._observer_viewport_selected = True
        return True

    def _episode_command_callback(self, message):
        """Apply one seeded scene only while the vehicle is safely landed."""
        try:
            command = json.loads(message.data)
            if command.get("command") != "prepare_episode":
                raise ValueError("unsupported episode command")
            episode_id = str(command["episode_id"])
            random_seed = int(command["random_seed"])
            mode = str(command.get("mode", "normal"))
            if (
                self._episode_id == episode_id
                and self._random_seed == random_seed
                and isinstance(self._scene_configuration, dict)
                and self._scene_configuration.get("mode") == mode
            ):
                self._episode_command_error = ""
                return
            pose = _world_pose(self._stage, VEHICLE_BODY_PATH)
            if pose is None or pose[2] > 0.25:
                raise RuntimeError("vehicle must be landed before scene reset")
            scene = generate_episode_scene(
                episode_id,
                random_seed,
                pose[0],
                pose[1],
                mode,
            )
            self._apply_scene(scene)
            self._scene_revision += 1
            self._scene_id = (
                f"expert_{scene['episode_id']}_seed_{scene['random_seed']}"
            )
            self._episode_id = scene["episode_id"]
            self._random_seed = scene["random_seed"]
            self._goal = tuple(scene["goal"])
            self._obstacles = list(scene["obstacles"])
            self._scene_configuration = scene
            self._scene_camera_boundary = {
                "fpv_rgb_frame_count": self._camera_frame_count,
                "observer_rgb_frame_count": self._observer_frame_count,
                "fpv_depth_frame_count": self._depth_frame_count,
            }
            self._episode_command_error = ""
            print(
                f"[IsaacRuntimeBridge] Prepared {self._scene_id} "
                f"revision={self._scene_revision} obstacles={len(self._obstacles)}"
            )
        except Exception as error:
            self._episode_command_error = f"{type(error).__name__}: {error}"
            print(f"[IsaacRuntimeBridge][ERROR] {self._episode_command_error}")

    def _apply_scene(self, scene):
        if self._stage.GetPrimAtPath(BOOTSTRAP_SCENE_ROOT).IsValid():
            self._stage.RemovePrim(BOOTSTRAP_SCENE_ROOT)
        if self._stage.GetPrimAtPath(SCENE_ROOT).IsValid():
            self._stage.RemovePrim(SCENE_ROOT)
        root = UsdGeom.Xform.Define(self._stage, SCENE_ROOT)
        root.GetPrim().SetCustomDataByKey("episode:id", scene["episode_id"])
        root.GetPrim().SetCustomDataByKey("episode:seed", scene["random_seed"])
        root.GetPrim().SetCustomDataByKey(
            "episode:generator", scene["generator"]
        )
        materials = create_scene_materials(self._stage, SCENE_ROOT)
        # ------------------------------------------------------------
        # Plain visual floor
        # ------------------------------------------------------------
        # 地板方形墊子的參數(僅外型，沒有碰撞)
        floor = self._create_box(
            f"{SCENE_ROOT}/PlainFloor",
            (100.0, 100.0, 0.01), # 地板尺寸
            (0.0, 0.0, 0.01), # 位置
            FLOOR_COLOR,
            collision=False,
        )
        bind_material(floor, materials["floor"])
        self._create_episode_lighting(scene["lighting"])
        UsdGeom.Xform.Define(
            self._stage,
            f"{SCENE_ROOT}/Obstacles",
        )

        for index, source in enumerate(
            scene["obstacles"],
            start=1,
        ):
            obstacle = self._create_cylinder_obstacle(
                source,
                index,
                materials["obstacle"],
            )

            obstacle.SetCustomDataByKey(
                "episode:shape",
                "cylinder",
            )
            obstacle.SetCustomDataByKey(
                "episode:radius",
                float(source["radius"]),
            )
            obstacle.SetCustomDataByKey(
                "episode:height",
                float(source["height"]),
            )
        start = UsdGeom.Cylinder.Define(
            self._stage, f"{SCENE_ROOT}/Start/StartDisk"
        )
        start.CreateRadiusAttr(0.5)
        start.CreateHeightAttr(0.05)
        start.AddTranslateOp().Set(Gf.Vec3d(
            scene["start"][0], scene["start"][1], 0.025
        ))
        start.CreateDisplayColorAttr([Gf.Vec3f(*START_MARKER_COLOR)])
        bind_material(start.GetPrim(), materials["start_marker"])
        goal = UsdGeom.Cylinder.Define(
            self._stage, f"{SCENE_ROOT}/Target/TargetDisk"
        )
        goal.CreateRadiusAttr(0.5)
        goal.CreateHeightAttr(0.05)
        goal.AddTranslateOp().Set(Gf.Vec3d(
            scene["target_marker"][0], scene["target_marker"][1], 0.025
        ))
        goal.CreateDisplayColorAttr([Gf.Vec3f(*GOAL_MARKER_COLOR)])
        bind_material(goal.GetPrim(), materials["goal_marker"])

    @staticmethod
    def _set_prim_transform(prim, position, rotation_deg=None, scale=None):
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
        if rotation_deg is not None:
            xformable.AddRotateXYZOp().Set(
                Gf.Vec3f(*map(float, rotation_deg))
            )
        if scale is not None:
            xformable.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))

    @staticmethod
    def _set_display_color(prim, color):
        UsdGeom.Gprim(prim).CreateDisplayColorAttr([
            Gf.Vec3f(*map(float, color))
        ])

    def _create_box(self, path, size, position, color, collision=False):
        cube = UsdGeom.Cube.Define(self._stage, path)
        cube.CreateSizeAttr(1.0)
        prim = cube.GetPrim()
        self._set_prim_transform(prim, position, scale=size)
        self._set_display_color(prim, color)
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim)
        return prim

    def _create_cylinder_obstacle(self, source, index, material):
        name = str(source.get("name") or f"Obstacle_{index:03d}")
        path = f"{SCENE_ROOT}/Obstacles/{name}"
        cylinder = UsdGeom.Cylinder.Define(self._stage, path)
        cylinder.CreateRadiusAttr(float(source["radius"]))
        cylinder.CreateHeightAttr(float(source["height"]))
        prim = cylinder.GetPrim()
        self._set_prim_transform(
            prim,
            (source["x"], source["y"], source["z"]),
        )
        self._set_display_color(prim, OBSTACLE_COLOR)
        bind_material(prim, material)
        if bool(source["collision"]):
            UsdPhysics.CollisionAPI.Apply(prim)
        return prim

    def _create_episode_lighting(self, lighting):
        light_root = f"{SCENE_ROOT}/Lights"
        UsdGeom.Xform.Define(self._stage, light_root)
        dome_spec = lighting["dome"]
        dome = UsdLux.DomeLight.Define(self._stage, f"{light_root}/Dome")
        dome.CreateIntensityAttr(float(dome_spec["intensity"]))
        dome.CreateExposureAttr(float(dome_spec["exposure"]))
        dome.CreateColorAttr(Gf.Vec3f(*map(float, dome_spec["color"])))

    def _update_camera_pose(self):
        if self._camera_transform is None:
            return False
        prim = self._stage.GetPrimAtPath(VEHICLE_BODY_PATH)
        if not prim or not prim.IsValid():
            return False
        matrix = omni.usd.get_world_transform_matrix(prim)
        position = matrix.ExtractTranslation()
        forward = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        forward = Gf.Vec3d(forward[0], forward[1], 0.0)
        length = math.hypot(float(forward[0]), float(forward[1]))
        if length <= 1e-6:
            return False
        direction = Gf.Vec3d(*(float(value) / length for value in forward))
        fpv_eye = Gf.Vec3d(
            position[0] + FPV_FORWARD_OFFSET_M * direction[0],
            position[1] + FPV_FORWARD_OFFSET_M * direction[1],
            position[2] + FPV_HEIGHT_M,
        )
        fpv_target = Gf.Vec3d(
            position[0] + FPV_LOOK_AHEAD_M * direction[0],
            position[1] + FPV_LOOK_AHEAD_M * direction[1],
            position[2] + FPV_LOOK_DOWN_M,
        )
        # FPV is a rigid body mount. World-space interpolation makes the eye
        # lag behind the current body pose/yaw while the look target does not,
        # which can put the UAV itself between eye and target during flight.
        self._fpv_camera_position = fpv_eye
        transform = Gf.Matrix4d().SetLookAt(
            self._fpv_camera_position,
            fpv_target,
            Gf.Vec3d(0.0, 0.0, 1.0),
        ).GetInverse()
        self._camera_transform.Set(transform)
        if self._observer_camera_transform is not None:
            if self._formal_expert_sensors_enabled:
                observer_eye = Gf.Vec3d(*FORMAL_OBSERVER_EYE)
                observer_target = Gf.Vec3d(*FORMAL_OBSERVER_TARGET)
                observer_up = Gf.Vec3d(*FORMAL_OBSERVER_UP)
                self._observer_camera_position = observer_eye
            elif OBSERVER_MODE == "TOP":
                observer_eye = Gf.Vec3d(
                    position[0],
                    position[1],
                    position[2] + OBSERVER_TOP_HEIGHT_M,
                )
                observer_target = Gf.Vec3d(
                    position[0],
                    position[1],
                    position[2] + OBSERVER_TOP_LOOK_AT_HEIGHT_M,
                )
                observer_up = Gf.Vec3d(0.0, 1.0, 0.0)
            else:
                right = Gf.Vec3d(direction[1], -direction[0], 0.0)
                observer_eye = Gf.Vec3d(
                    position[0] - direction[0] * OBSERVER_BACK_DISTANCE_M
                    + right[0] * OBSERVER_SIDE_OFFSET_M,
                    position[1] - direction[1] * OBSERVER_BACK_DISTANCE_M
                    + right[1] * OBSERVER_SIDE_OFFSET_M,
                    position[2] + OBSERVER_HEIGHT_M,
                )
                observer_target = Gf.Vec3d(
                    position[0] + direction[0] * OBSERVER_LOOK_AHEAD_M,
                    position[1] + direction[1] * OBSERVER_LOOK_AHEAD_M,
                    position[2] + OBSERVER_LOOK_AT_HEIGHT_M,
                )
                observer_up = Gf.Vec3d(0.0, 0.0, 1.0)
            if not self._formal_expert_sensors_enabled:
                self._observer_camera_position = self._smooth_position(
                    self._observer_camera_position, observer_eye
                )
            observer_transform = Gf.Matrix4d().SetLookAt(
                self._observer_camera_position,
                observer_target,
                observer_up,
            ).GetInverse()
            self._observer_camera_transform.Set(observer_transform)
        return True

    @staticmethod
    def _smooth_position(current, target):
        if current is None:
            return target
        return Gf.Vec3d(*(
            current[index] * (1.0 - CAMERA_SMOOTHING)
            + target[index] * CAMERA_SMOOTHING
            for index in range(3)
        ))

    @staticmethod
    def _jpeg_message(
        data, stamp, frame_id, expected_size=(CAMERA_WIDTH, CAMERA_HEIGHT)
    ):
        import numpy as np
        from PIL import Image

        if isinstance(data, dict):
            data = data.get("data")
        if data is None or getattr(data, "size", 0) == 0:
            raise RuntimeError("RGB annotator has no frame")
        rgb = np.asarray(data)[..., :3]
        width, height = expected_size
        if rgb.shape != (height, width, 3):
            raise RuntimeError(f"unexpected RGB shape {rgb.shape}")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        stream = BytesIO()
        Image.fromarray(rgb, mode="RGB").save(
            stream, format="JPEG", quality=JPEG_QUALITY
        )
        message = CompressedImage()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.format = "jpeg; rgb8"
        message.data = stream.getvalue()
        return message

    def _publish_camera(self, stamp, now_monotonic):
        if (
            not self._camera_enabled
            or now_monotonic - self._last_camera_publish_monotonic
            < CAMERA_PUBLISH_PERIOD_S
        ):
            return
        self._last_camera_publish_monotonic = now_monotonic
        try:
            if not self._update_camera_pose():
                raise RuntimeError("vehicle pose unavailable")
            message = self._jpeg_message(
                self._rgb_annotator.get_data(), stamp, "isaac_fpv_optical"
            )
            self._camera_publisher.publish(message)
            self._camera_frame_count += 1
            self._camera_error = ""
        except Exception as error:
            self._camera_error = f"{type(error).__name__}: {error}"

        if (
            self._expert_sensors_enabled
            and now_monotonic - self._last_observer_publish_monotonic
            >= self._observer_publish_period_s
        ):
            self._last_observer_publish_monotonic = now_monotonic
            try:
                message = self._jpeg_message(
                    self._observer_rgb_annotator.get_data(),
                    stamp,
                    "isaac_observer_optical",
                    self._observer_resolution,
                )
                self._observer_camera_publisher.publish(message)
                self._observer_frame_count += 1
                self._observer_camera_error = ""
            except Exception as error:
                self._observer_camera_error = (
                    f"{type(error).__name__}: {error}"
                )

        if (
            self._expert_sensors_enabled
            and now_monotonic - self._last_depth_publish_monotonic
            >= DEPTH_PUBLISH_PERIOD_S
        ):
            self._last_depth_publish_monotonic = now_monotonic
            try:
                import numpy as np
                from PIL import Image

                depth = self._depth_annotator.get_data()
                if isinstance(depth, dict):
                    depth = depth.get("data")
                depth = np.asarray(depth, dtype=np.float32).squeeze()
                if depth.shape != (CAMERA_HEIGHT, CAMERA_WIDTH):
                    raise RuntimeError(f"unexpected depth shape {depth.shape}")
                valid = np.isfinite(depth) & (depth >= DEPTH_MIN_M)
                depth_mm = np.zeros(depth.shape, dtype=np.uint16)
                depth_mm[valid] = np.rint(
                    np.clip(depth[valid], DEPTH_MIN_M, DEPTH_MAX_M) * 1000.0
                ).astype(np.uint16)
                stream = BytesIO()
                Image.fromarray(depth_mm, mode="I;16").save(stream, format="PNG")
                message = CompressedImage()
                message.header.stamp = stamp
                message.header.frame_id = "isaac_fpv_optical"
                message.format = (
                    "png; 16UC1; unit=millimeter; range=50..30000; invalid=0"
                )
                message.data = stream.getvalue()
                self._depth_publisher.publish(message)
                self._depth_frame_count += 1
                self._depth_error = ""
            except Exception as error:
                self._depth_error = f"{type(error).__name__}: {error}"

    def _on_update(self, _event):
        if self._stopped:
            return
        if (
            self._observer_viewport_requested
            and not self._observer_viewport_selected
        ):
            self._select_observer_viewport()
        rclpy.spin_once(self._node, timeout_sec=0.0)
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_publish_monotonic < PUBLISH_PERIOD_S:
            return
        self._last_publish_monotonic = now_monotonic
        timeline_playing = bool(
            omni.timeline.get_timeline_interface().is_playing()
        )
        prim = self._stage.GetPrimAtPath(VEHICLE_BODY_PATH)
        prim_valid = bool(prim and prim.IsValid())
        pose = _world_pose(self._stage, VEHICLE_BODY_PATH) if prim_valid else None
        pose_valid = pose is not None
        if pose_valid:
            message = PoseStamped()
            stamp = self._node.get_clock().now().to_msg()
            message.header.stamp = stamp
            message.header.frame_id = FRAME_ID
            message.pose.position.x = pose[0]
            message.pose.position.y = pose[1]
            message.pose.position.z = pose[2]
            message.pose.orientation.x = pose[3]
            message.pose.orientation.y = pose[4]
            message.pose.orientation.z = pose[5]
            message.pose.orientation.w = pose[6]
            self._pose_publisher.publish(message)
            self._publish_camera(stamp, now_monotonic)
        self._sequence += 1
        status = String()
        status.data = json.dumps({
            "schema": SCHEMA,
            "sequence": self._sequence,
            "scene_id": self._scene_id,
            "scene_revision": self._scene_revision,
            "timeline_playing": timeline_playing,
            "prim_valid": prim_valid,
            "pose_valid": pose_valid,
            "vehicle_prim_path": VEHICLE_BODY_PATH,
            "goal": list(self._goal),
            "obstacles": list(self._obstacles),
            "episode_id": self._episode_id,
            "random_seed": self._random_seed,
            "scene_configuration": self._scene_configuration,
            "scene_camera_boundary": self._scene_camera_boundary,
            "runtime_generation": int(getattr(
                builtins, "_isaac_uav_runtime_generation", 0
            )),
            "episode_command_error": self._episode_command_error,
            "fpv_rgb_enabled": self._camera_enabled,
            "fpv_rgb_ready": self._camera_frame_count > 0,
            "fpv_rgb_frame_count": self._camera_frame_count,
            "fpv_rgb_error": self._camera_error,
            "observer_rgb_enabled": self._expert_sensors_enabled,
            "observer_rgb_ready": self._observer_frame_count > 0,
            "observer_rgb_frame_count": self._observer_frame_count,
            "observer_rgb_error": self._observer_camera_error,
            "observer_mode": self._observer_mode,
            "fpv_depth_enabled": self._expert_sensors_enabled,
            "fpv_depth_ready": self._depth_frame_count > 0,
            "fpv_depth_frame_count": self._depth_frame_count,
            "fpv_depth_error": self._depth_error,
            # Compatibility aliases for historical CLI and dataset evidence.
            "phase10a_camera_enabled": self._camera_enabled,
            "phase10a_camera_ready": self._camera_frame_count > 0,
            "phase10a_camera_frame_count": self._camera_frame_count,
            "phase10a_camera_error": self._camera_error,
            "phase10c_observer_rgb_enabled": self._expert_sensors_enabled,
            "phase10c_observer_rgb_ready": self._observer_frame_count > 0,
            "phase10c_observer_rgb_frame_count": self._observer_frame_count,
            "phase10c_observer_rgb_error": self._observer_camera_error,
            "phase10c_observer_mode": self._observer_mode,
            "phase10b_fpv_depth_enabled": self._expert_sensors_enabled,
            "phase10b_fpv_depth_ready": self._depth_frame_count > 0,
            "phase10b_fpv_depth_frame_count": self._depth_frame_count,
            "phase10b_fpv_depth_error": self._depth_error,
        }, sort_keys=True, separators=(",", ":"))
        self._status_publisher.publish(status)

    def stop(self):
        """Stop callbacks and ROS resources deterministically."""
        if self._stopped:
            return
        self._stopped = True
        if self._subscription is not None:
            self._subscription.unsubscribe()
            self._subscription = None
        if self._rgb_annotator is not None:
            self._rgb_annotator.detach()
            self._rgb_annotator = None
        if self._observer_rgb_annotator is not None:
            self._observer_rgb_annotator.detach()
            self._observer_rgb_annotator = None
        if self._depth_annotator is not None:
            self._depth_annotator.detach()
            self._depth_annotator = None
        self._node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
        print("[IsaacRuntimeBridge] Stopped.")


def stop_isaac_runtime_bridge():
    """Public Script Editor cleanup hook."""
    bridge = getattr(builtins, "_isaac_runtime_bridge", None)
    if bridge is not None:
        bridge.stop()
        builtins._isaac_runtime_bridge = None


stop_isaac_runtime_bridge()
builtins._isaac_runtime_bridge = IsaacRuntimeBridge()
builtins.stop_isaac_runtime_bridge = stop_isaac_runtime_bridge
