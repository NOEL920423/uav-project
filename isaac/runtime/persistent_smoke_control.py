#!/usr/bin/env python3
"""Diagnostic-only lifecycle control for a persistent Isaac/Pegasus runtime."""

from __future__ import annotations

import asyncio
import builtins
import json
import time

import numpy as np

import omni.kit.app
import omni.usd

import rclpy
from std_msgs.msg import String


COMMAND_TOPIC = "/uav/isaac/runtime_smoke/command"
STATUS_TOPIC = "/uav/isaac/runtime_smoke/status"
SCHEMA = "uav_persistent_runtime_smoke/v1"
CANONICAL_POSITION = np.array([0.0, 0.0, 0.1], dtype=np.float32)
CANONICAL_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
ZERO3 = np.zeros(3, dtype=np.float32)
ZERO6 = np.zeros(6, dtype=np.float32)
PUBLISH_PERIOD_S = 0.10


class PersistentSmokeControl:
    """Expose bounded stop/reset operations without recreating stage resources."""

    def __init__(self):
        self._world = getattr(builtins, "_isaac_uav_smoke_world", None)
        self._vehicle = getattr(builtins, "_isaac_uav_smoke_vehicle", None)
        self._backend = getattr(builtins, "_isaac_uav_smoke_backend", None)
        self._runtime_bridge = getattr(builtins, "_isaac_runtime_bridge", None)
        if any(item is None for item in (
            self._world, self._vehicle, self._backend, self._runtime_bridge
        )):
            raise RuntimeError("persistent smoke bootstrap handles are unavailable")

        self._node = rclpy.create_node("isaac_persistent_runtime_smoke")
        self._publisher = self._node.create_publisher(String, STATUS_TOPIC, 10)
        self._subscription = self._node.create_subscription(
            String, COMMAND_TOPIC, self._command_callback, 10
        )
        self._state = "booted"
        self._generation = 0
        self._last_command_id = ""
        self._failure_reason = ""
        self._reset_evidence = {}
        self._active_task = None
        self._last_publish_monotonic = 0.0
        self._stopped = False
        self._resource_identity = self._capture_resource_identity()
        self._update_subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update,
                name="PersistentSmokeControlUpdate",
            )
        )
        print(
            "[PersistentSmoke] Control ready "
            f"world={self._resource_identity['world_object_id']} "
            f"vehicle={self._resource_identity['vehicle_object_id']} "
            f"backend={self._resource_identity['backend_object_id']}"
        )

    def _capture_resource_identity(self):
        stage = omni.usd.get_context().get_stage()
        root_layer = stage.GetRootLayer() if stage is not None else None
        bridge = self._runtime_bridge
        return {
            "world_object_id": id(self._world),
            "stage_object_id": id(stage),
            "stage_root_identifier": (
                str(root_layer.identifier) if root_layer is not None else ""
            ),
            "vehicle_object_id": id(self._vehicle),
            "backend_object_id": id(self._backend),
            "runtime_bridge_object_id": id(bridge),
            "fpv_camera_path": "/World/RuntimeSensors/FPVCamera",
            "observer_camera_path": "/World/RuntimeSensors/ObserverCamera",
            "fpv_render_product_object_id": id(bridge._render_product),
            "observer_render_product_object_id": id(
                bridge._observer_render_product
            ),
            "rgb_annotator_object_id": id(bridge._rgb_annotator),
            "observer_rgb_annotator_object_id": id(
                bridge._observer_rgb_annotator
            ),
            "depth_annotator_object_id": id(bridge._depth_annotator),
        }

    def _camera_ready(self):
        bridge = self._runtime_bridge
        return bool(
            bridge._camera_frame_count > 0
            and bridge._observer_frame_count > 0
            and bridge._depth_frame_count > 0
        )

    def _status_payload(self):
        identity = self._capture_resource_identity()
        return {
            "schema": SCHEMA,
            "state": self._state,
            "generation": self._generation,
            "last_command_id": self._last_command_id,
            "failure_reason": self._failure_reason,
            "timeline_playing": bool(self._world.is_playing()),
            "backend_running": bool(self._backend._is_running),
            "backend_connection_open": self._backend._connection is not None,
            "camera_ready": self._camera_ready(),
            "resource_identity": identity,
            "resource_identity_unchanged": identity == self._resource_identity,
            "reset_evidence": self._reset_evidence,
        }

    def _publish_status(self):
        message = String()
        message.data = json.dumps(
            self._status_payload(), sort_keys=True, separators=(",", ":")
        )
        self._publisher.publish(message)

    def _command_callback(self, message):
        try:
            command = json.loads(message.data)
            command_id = str(command["command_id"])
            operation = str(command["command"])
            generation = int(command.get("generation", self._generation))
            if command_id == self._last_command_id:
                return
            if self._active_task is not None and not self._active_task.done():
                return
            if operation not in {"stop_episode", "reset_episode"}:
                raise ValueError(f"unsupported command: {operation}")
            self._failure_reason = ""
            self._active_task = asyncio.ensure_future(
                self._execute(operation, command_id, generation)
            )
        except Exception as error:
            self._state = "failed"
            self._failure_reason = f"{type(error).__name__}: {error}"
            print(f"[PersistentSmoke][ERROR] {self._failure_reason}")

    async def _execute(self, operation, command_id, generation):
        try:
            if operation == "stop_episode":
                self._state = "stopping"
                await self._world.stop_async()
                self._clear_actuator_state()
                if self._backend._is_running:
                    raise RuntimeError("Pegasus backend remained running after stop")
                if self._backend._connection is not None:
                    raise RuntimeError("MAVLink connection remained open after stop")
                self._state = "stopped"
                self._reset_evidence = {
                    "backend_stopped": True,
                    "actuator_reference_zero": self._actuator_reference_zero(),
                }
            else:
                if not self._world.is_stopped():
                    raise RuntimeError("reset_episode requires a stopped world")
                self._state = "resetting"
                self._vehicle.set_default_state(
                    position=CANONICAL_POSITION,
                    orientation=CANONICAL_ORIENTATION,
                )
                self._clear_actuator_state()
                await self._world.reset_async()
                self._vehicle.set_world_pose(
                    position=CANONICAL_POSITION,
                    orientation=CANONICAL_ORIENTATION,
                )
                self._vehicle.set_world_velocity(ZERO6)
                if self._vehicle.num_dof:
                    joint_zeros = np.zeros(
                        self._vehicle.num_dof, dtype=np.float32
                    )
                    self._vehicle.set_joint_positions(joint_zeros)
                    self._vehicle.set_joint_velocities(joint_zeros)
                self._vehicle.set_linear_velocity(ZERO3)
                self._vehicle.set_angular_velocity(ZERO3)
                self._clear_actuator_state()
                self._reset_evidence = self._verify_reset()
                if not all(self._reset_evidence.values()):
                    raise RuntimeError(
                        "UAV reset verification failed: "
                        + json.dumps(self._reset_evidence, sort_keys=True)
                    )
                if not self._backend._is_running:
                    raise RuntimeError("Pegasus backend did not resume")
                if self._backend._connection is None:
                    raise RuntimeError("MAVLink listener was not recreated")
                builtins._isaac_uav_runtime_generation = generation
                self._state = "ready_for_px4"
            self._generation = generation
            self._last_command_id = command_id
            self._failure_reason = ""
            print(
                f"[PersistentSmoke] command={operation} generation={generation} "
                f"state={self._state}"
            )
        except Exception as error:
            self._state = "failed"
            self._generation = generation
            self._last_command_id = command_id
            self._failure_reason = f"{type(error).__name__}: {error}"
            print(f"[PersistentSmoke][ERROR] {self._failure_reason}")

    def _clear_actuator_state(self):
        rotor_data = getattr(self._backend, "_rotor_data", None)
        if rotor_data is not None:
            rotor_data.zero_input_reference()
        thrusters = getattr(self._vehicle, "_thrusters", None)
        if thrusters is not None:
            thrusters.set_input_reference(
                [0.0 for _ in range(int(thrusters._num_rotors))]
            )

    def _actuator_reference_zero(self):
        reference = np.asarray(self._backend.input_reference(), dtype=float)
        return bool(reference.size and np.allclose(reference, 0.0, atol=1e-6))

    @staticmethod
    def _finite_close(values, expected, tolerance=1e-4):
        values = np.asarray(values, dtype=float)
        expected = np.asarray(expected, dtype=float)
        return bool(
            values.shape == expected.shape
            and np.all(np.isfinite(values))
            and np.allclose(values, expected, atol=tolerance)
        )

    def _verify_reset(self):
        position, orientation = self._vehicle.get_world_pose()
        linear = self._vehicle.get_linear_velocity()
        angular = self._vehicle.get_angular_velocity()
        joints = self._vehicle.get_joints_state()
        joint_positions_zero = True
        joint_velocities_zero = True
        if joints is not None and joints.positions is not None:
            joint_positions_zero = self._finite_close(
                joints.positions, np.zeros_like(joints.positions), tolerance=1e-3
            )
        if joints is not None and joints.velocities is not None:
            joint_velocities_zero = self._finite_close(
                joints.velocities, np.zeros_like(joints.velocities), tolerance=1e-3
            )
        return {
            "pose_canonical": self._finite_close(position, CANONICAL_POSITION),
            "orientation_canonical": self._finite_close(
                orientation, CANONICAL_ORIENTATION
            ),
            "linear_velocity_zero": self._finite_close(linear, ZERO3),
            "angular_velocity_zero": self._finite_close(angular, ZERO3),
            "joint_positions_zero": joint_positions_zero,
            "joint_velocities_zero": joint_velocities_zero,
            "actuator_reference_zero": self._actuator_reference_zero(),
            "backend_listener_ready": bool(
                self._backend._is_running
                and self._backend._connection is not None
            ),
            "resource_identity_unchanged": (
                self._capture_resource_identity() == self._resource_identity
            ),
        }

    def _on_update(self, _event):
        if self._stopped:
            return
        rclpy.spin_once(self._node, timeout_sec=0.0)
        now = time.monotonic()
        if now - self._last_publish_monotonic >= PUBLISH_PERIOD_S:
            self._last_publish_monotonic = now
            self._publish_status()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        if self._active_task is not None and not self._active_task.done():
            self._active_task.cancel()
        if self._update_subscription is not None:
            self._update_subscription.unsubscribe()
            self._update_subscription = None
        self._node.destroy_node()


def stop_persistent_smoke_control():
    control = getattr(builtins, "_isaac_uav_smoke_control", None)
    if control is not None:
        control.stop()
        builtins._isaac_uav_smoke_control = None


stop_persistent_smoke_control()
builtins._isaac_uav_smoke_control = PersistentSmokeControl()
builtins.stop_persistent_smoke_control = stop_persistent_smoke_control
