#!/usr/bin/env python3
"""Publish deterministic Isaac runtime evidence without recording datasets.

This script runs inside Isaac Sim after Pegasus has created the vehicle and
started the physics timeline.  It intentionally uses only standard ROS 2
messages so Isaac's embedded Python does not need the project's custom message
overlay.
"""

from __future__ import annotations

import builtins
import json
import math
import time

from geometry_msgs.msg import PoseStamped

import omni.kit.app
import omni.timeline
import omni.usd

import rclpy

from std_msgs.msg import String

from pxr import UsdGeom


VEHICLE_BODY_PATH = "/World/quadrotor/body"
POSE_TOPIC = "/isaac_uav/pose"
STATUS_TOPIC = "/uav/isaac/runtime_status"
FRAME_ID = "isaac_world"
SCHEMA = "uav_isaac_runtime/v1"
SCENE_ID = "phase9_fixed_scene_v1"
SCENE_REVISION = 1
PUBLISH_PERIOD_S = 0.05
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
            "no camera, recorder, or dataset lifecycle"
        )

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
            message.header.stamp = self._node.get_clock().now().to_msg()
            message.header.frame_id = FRAME_ID
            message.pose.position.x = pose[0]
            message.pose.position.y = pose[1]
            message.pose.position.z = pose[2]
            message.pose.orientation.x = pose[3]
            message.pose.orientation.y = pose[4]
            message.pose.orientation.z = pose[5]
            message.pose.orientation.w = pose[6]
            self._pose_publisher.publish(message)
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
