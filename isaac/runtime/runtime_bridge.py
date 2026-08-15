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

from pxr import Gf, UsdGeom, UsdPhysics


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from episode_scene import generate_episode_scene


VEHICLE_BODY_PATH = "/World/quadrotor/body"
POSE_TOPIC = "/isaac_uav/pose"
STATUS_TOPIC = "/uav/isaac/runtime_status"
FRAME_ID = "isaac_world"
SCHEMA = "uav_isaac_runtime/v1"
SCENE_ID = "phase9_fixed_scene_v1"
SCENE_REVISION = 1
PUBLISH_PERIOD_S = 0.05
CAMERA_PUBLISH_PERIOD_S = 0.20
CAMERA_TOPIC = "/uav/isaac/fpv/image/compressed"
CAMERA_PATH = "/World/Phase10A/FPVCamera"
TOP_CAMERA_PATH = "/World/Phase10B/TopCamera"
TOP_CAMERA_TOPIC = "/uav/isaac/top/image/compressed"
DEPTH_TOPIC = "/uav/isaac/fpv/depth/compressed"
EPISODE_COMMAND_TOPIC = "/uav/isaac/episode_command"
SCENE_ROOT = "/World/Phase9Runtime"
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 180
JPEG_QUALITY = 85
TOP_CAMERA_PUBLISH_PERIOD_S = 0.50
DEPTH_PUBLISH_PERIOD_S = 0.20
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 30.0
GOAL = (0.5, 3.0, 1.5)
OBSTACLES = ({
    "name": "Building_Phase9_001",
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
        self._camera_enabled = os.environ.get("UAV_PHASE10A_CAMERA", "0") == "1"
        self._phase10b_enabled = (
            os.environ.get("UAV_PHASE10B_SENSORS", "0") == "1"
        )
        self._camera_enabled = self._camera_enabled or self._phase10b_enabled
        self._scene_id = SCENE_ID
        self._scene_revision = SCENE_REVISION
        self._goal = GOAL
        self._obstacles = list(OBSTACLES)
        self._episode_id = ""
        self._random_seed = None
        self._scene_configuration = None
        self._episode_command_error = ""
        self._camera_publisher = None
        self._top_camera_publisher = None
        self._depth_publisher = None
        self._camera_transform = None
        self._top_camera_transform = None
        self._rgb_annotator = None
        self._top_rgb_annotator = None
        self._depth_annotator = None
        self._render_product = None
        self._top_render_product = None
        self._last_camera_publish_monotonic = 0.0
        self._last_top_publish_monotonic = 0.0
        self._last_depth_publish_monotonic = 0.0
        self._camera_frame_count = 0
        self._top_frame_count = 0
        self._depth_frame_count = 0
        self._camera_error = "disabled"
        self._top_camera_error = "disabled"
        self._depth_error = "disabled"
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
            f"Phase10B auxiliary={'enabled' if self._phase10b_enabled else 'disabled'}"
        )

    def _setup_camera(self):
        import omni.replicator.core as rep

        existing = self._stage.GetPrimAtPath(CAMERA_PATH)
        if existing and existing.IsValid():
            self._stage.RemovePrim(CAMERA_PATH)
        camera = UsdGeom.Camera.Define(self._stage, CAMERA_PATH)
        camera.GetFocalLengthAttr().Set(18.0)
        camera.GetHorizontalApertureAttr().Set(20.955)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 30.0))
        camera.CreateExposureAttr().Set(2.0)
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
        if self._phase10b_enabled:
            top_camera = UsdGeom.Camera.Define(self._stage, TOP_CAMERA_PATH)
            top_camera.GetFocalLengthAttr().Set(18.0)
            top_camera.GetHorizontalApertureAttr().Set(20.955)
            top_camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 50.0))
            top_camera.CreateExposureAttr().Set(2.0)
            self._top_camera_transform = UsdGeom.Xformable(
                top_camera.GetPrim()
            ).AddTransformOp()
            self._top_render_product = rep.create.render_product(
                TOP_CAMERA_PATH, (CAMERA_WIDTH, CAMERA_HEIGHT)
            )
            self._top_rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._top_rgb_annotator.attach([self._top_render_product])
            self._depth_annotator = rep.AnnotatorRegistry.get_annotator(
                "distance_to_camera"
            )
            self._depth_annotator.attach([self._render_product])
            self._top_camera_publisher = self._node.create_publisher(
                CompressedImage, TOP_CAMERA_TOPIC, 10
            )
            self._depth_publisher = self._node.create_publisher(
                CompressedImage, DEPTH_TOPIC, 10
            )
            self._top_camera_error = "warming"
            self._depth_error = "warming"
        print(
            f"[IsaacRuntimeBridge] Phase10A FPV render product: "
            f"{CAMERA_WIDTH}x{CAMERA_HEIGHT} JPEG quality {JPEG_QUALITY}"
        )

    def _episode_command_callback(self, message):
        """Apply one seeded scene only while the vehicle is safely landed."""
        try:
            command = json.loads(message.data)
            if command.get("command") != "prepare_episode":
                raise ValueError("unsupported episode command")
            pose = _world_pose(self._stage, VEHICLE_BODY_PATH)
            if pose is None or pose[2] > 0.25:
                raise RuntimeError("vehicle must be landed before scene reset")
            scene = generate_episode_scene(
                str(command["episode_id"]),
                int(command["random_seed"]),
                pose[0],
                pose[1],
                str(command.get("mode", "normal")),
            )
            self._apply_scene(scene)
            self._scene_revision += 1
            self._scene_id = f"phase10b_{scene['episode_id']}_seed_{scene['random_seed']}"
            self._episode_id = scene["episode_id"]
            self._random_seed = scene["random_seed"]
            self._goal = tuple(scene["goal"])
            self._obstacles = list(scene["obstacles"])
            self._scene_configuration = scene
            self._episode_command_error = ""
            print(
                f"[IsaacRuntimeBridge] Prepared {self._scene_id} "
                f"revision={self._scene_revision} obstacles={len(self._obstacles)}"
            )
        except Exception as error:
            self._episode_command_error = f"{type(error).__name__}: {error}"
            print(f"[IsaacRuntimeBridge][ERROR] {self._episode_command_error}")

    def _apply_scene(self, scene):
        if self._stage.GetPrimAtPath(SCENE_ROOT).IsValid():
            self._stage.RemovePrim(SCENE_ROOT)
        root = UsdGeom.Xform.Define(self._stage, SCENE_ROOT)
        root.GetPrim().SetCustomDataByKey("phase10b:episode_id", scene["episode_id"])
        root.GetPrim().SetCustomDataByKey("phase10b:seed", scene["random_seed"])
        for index, source in enumerate(scene["obstacles"], start=1):
            obstacle = UsdGeom.Cube.Define(
                self._stage, f"{SCENE_ROOT}/Obstacle_{index:02d}"
            )
            obstacle.CreateSizeAttr(1.0)
            obstacle.AddTranslateOp().Set(Gf.Vec3d(
                source["x"], source["y"], source["z"]
            ))
            obstacle.AddScaleOp().Set(Gf.Vec3d(
                source["radius"] * 2.0,
                source["radius"] * 2.0,
                source["height"],
            ))
            color = 0.25 + 0.12 * (index % 4)
            obstacle.CreateDisplayColorAttr([Gf.Vec3f(color, 0.35, 0.70 - color / 2)])
            UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
        goal = UsdGeom.Cylinder.Define(self._stage, f"{SCENE_ROOT}/Goal")
        goal.CreateRadiusAttr(0.25)
        goal.CreateHeightAttr(0.02)
        goal.AddTranslateOp().Set(Gf.Vec3d(
            scene["goal"][0], scene["goal"][1], 0.01
        ))
        goal.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.85, 0.25)])

    def _update_camera_pose(self):
        if self._camera_transform is None:
            return False
        prim = self._stage.GetPrimAtPath(VEHICLE_BODY_PATH)
        if not prim or not prim.IsValid():
            return False
        matrix = omni.usd.get_world_transform_matrix(prim)
        position = matrix.ExtractTranslation()
        forward = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        # Match the already-proven legacy FPV mount: keep the view horizontal
        # and place it beyond the Iris body/propeller envelope.
        forward = Gf.Vec3d(forward[0], forward[1], 0.0)
        length = math.hypot(float(forward[0]), float(forward[1]))
        if length <= 1e-6:
            return False
        direction = Gf.Vec3d(*(float(value) / length for value in forward))
        eye = Gf.Vec3d(
            position[0] + 0.45 * direction[0],
            position[1] + 0.45 * direction[1],
            position[2] + 0.12,
        )
        target = Gf.Vec3d(
            eye[0] + 4.0 * direction[0],
            eye[1] + 4.0 * direction[1],
            eye[2] - 1.20,
        )
        transform = Gf.Matrix4d().SetLookAt(
            eye, target, Gf.Vec3d(0.0, 0.0, 1.0)
        ).GetInverse()
        self._camera_transform.Set(transform)
        if self._top_camera_transform is not None:
            top_eye = Gf.Vec3d(position[0], position[1], position[2] + 9.0)
            top_target = Gf.Vec3d(position[0], position[1], position[2])
            top_transform = Gf.Matrix4d().SetLookAt(
                top_eye, top_target, Gf.Vec3d(0.0, 1.0, 0.0)
            ).GetInverse()
            self._top_camera_transform.Set(top_transform)
        return True

    @staticmethod
    def _jpeg_message(data, stamp, frame_id):
        import numpy as np
        from PIL import Image

        if isinstance(data, dict):
            data = data.get("data")
        if data is None or getattr(data, "size", 0) == 0:
            raise RuntimeError("RGB annotator has no frame")
        rgb = np.asarray(data)[..., :3]
        if rgb.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, 3):
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
            self._phase10b_enabled
            and now_monotonic - self._last_top_publish_monotonic
            >= TOP_CAMERA_PUBLISH_PERIOD_S
        ):
            self._last_top_publish_monotonic = now_monotonic
            try:
                message = self._jpeg_message(
                    self._top_rgb_annotator.get_data(),
                    stamp,
                    "isaac_top_optical",
                )
                self._top_camera_publisher.publish(message)
                self._top_frame_count += 1
                self._top_camera_error = ""
            except Exception as error:
                self._top_camera_error = f"{type(error).__name__}: {error}"

        if (
            self._phase10b_enabled
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
            "episode_command_error": self._episode_command_error,
            "phase10a_camera_enabled": self._camera_enabled,
            "phase10a_camera_ready": self._camera_frame_count > 0,
            "phase10a_camera_frame_count": self._camera_frame_count,
            "phase10a_camera_error": self._camera_error,
            "phase10b_top_rgb_enabled": self._phase10b_enabled,
            "phase10b_top_rgb_ready": self._top_frame_count > 0,
            "phase10b_top_rgb_frame_count": self._top_frame_count,
            "phase10b_top_rgb_error": self._top_camera_error,
            "phase10b_fpv_depth_enabled": self._phase10b_enabled,
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
        if self._top_rgb_annotator is not None:
            self._top_rgb_annotator.detach()
            self._top_rgb_annotator = None
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
