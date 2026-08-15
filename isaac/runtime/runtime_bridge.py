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
import time

from geometry_msgs.msg import PoseStamped

import omni.kit.app
import omni.timeline
import omni.usd

import rclpy

from std_msgs.msg import String

from sensor_msgs.msg import CompressedImage

from pxr import Gf, UsdGeom


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
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 180
JPEG_QUALITY = 85
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
        self._camera_publisher = None
        self._camera_transform = None
        self._rgb_annotator = None
        self._render_product = None
        self._last_camera_publish_monotonic = 0.0
        self._camera_frame_count = 0
        self._camera_error = "disabled"
        if self._camera_enabled:
            self._setup_camera()
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
            f"Phase10A camera={'enabled' if self._camera_enabled else 'disabled'}"
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
        print(
            f"[IsaacRuntimeBridge] Phase10A FPV render product: "
            f"{CAMERA_WIDTH}x{CAMERA_HEIGHT} JPEG quality {JPEG_QUALITY}"
        )

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
            eye[2] - 0.18,
        )
        transform = Gf.Matrix4d().SetLookAt(
            eye, target, Gf.Vec3d(0.0, 0.0, 1.0)
        ).GetInverse()
        self._camera_transform.Set(transform)
        return True

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
            data = self._rgb_annotator.get_data()
            if isinstance(data, dict):
                data = data.get("data")
            if data is None or getattr(data, "size", 0) == 0:
                raise RuntimeError("RGB annotator has no frame")
            import numpy as np
            from PIL import Image

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
            message.header.frame_id = "isaac_fpv_optical"
            message.format = "jpeg; rgb8"
            message.data = stream.getvalue()
            self._camera_publisher.publish(message)
            self._camera_frame_count += 1
            self._camera_error = ""
        except Exception as error:
            self._camera_error = f"{type(error).__name__}: {error}"

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
            "scene_id": SCENE_ID,
            "scene_revision": SCENE_REVISION,
            "timeline_playing": timeline_playing,
            "prim_valid": prim_valid,
            "pose_valid": pose_valid,
            "vehicle_prim_path": VEHICLE_BODY_PATH,
            "goal": list(GOAL),
            "obstacles": list(OBSTACLES),
            "phase10a_camera_enabled": self._camera_enabled,
            "phase10a_camera_ready": self._camera_frame_count > 0,
            "phase10a_camera_frame_count": self._camera_frame_count,
            "phase10a_camera_error": self._camera_error,
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
