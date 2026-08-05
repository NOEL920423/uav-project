#!/usr/bin/env python3
"""ROS 2 service manager for Isaac Sim UAV episode preparation.

Run this file once inside Isaac Sim. It keeps an rclpy node alive inside the
Isaac update loop and exposes services that control Isaac-only features:

- cleanup generated USD prims and old controllers
- generate a random obstacle episode
- create FPV and observer cameras
- start and stop dual-camera PNG recording
- start and stop Isaac pose publishing and CSV logging
- plan and publish an A* path on /uav/planned_path
- prepare all of the above in one service call

The external ROS 2 lookahead follower remains a separate node. This manager
never publishes PX4 flight-control setpoints directly.
"""

from __future__ import annotations

import builtins
import runpy
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import omni.kit.app
import omni.usd
from pxr import Sdf

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String
    from std_srvs.srv import Trigger
except Exception as exc:
    raise RuntimeError(
        "ROS 2 Python modules are not available inside Isaac Sim. "
        "Enable the Isaac Sim ROS 2 Bridge and use the built-in Jazzy environment."
    ) from exc


DEFAULT_SCRIPT_ROOT = Path.home() / "uav-project" / "ros2_isaac_scripts"

GENERATED_PRIM_PATHS = [
    "/World/GeneratedEpisode",
    "/World/UAV_Camera_FPV",
    "/World/UAV_Camera_Observer",
    "/World/DebugFlightPath",
    "/World/DebugSafetyEnvelope",
    "/World/FlightPath",
    "/World/ExecutedTrail",
    "/World/WaypointMarkers",
]


class IsaacUavEpisodeManager(Node):
    """Expose Isaac Sim episode operations through ROS 2 Trigger services."""

    def __init__(self) -> None:
        super().__init__("isaac_uav_episode_manager")

        self.declare_parameter("script_root", str(DEFAULT_SCRIPT_ROOT))
        # Keep corrected, active-flight-only datasets separate from older
        # recordings so BC training cannot accidentally ingest legacy frames.
        self.declare_parameter("episode_prefix", "bc_astar")
        self.declare_parameter("fpv_forward_axis", "X")
        self.declare_parameter("fpv_look_down_m", -0.8)
        self.declare_parameter("observer_mode", "TOP")
        # The external orchestrator starts capture only after path/PX4 READY.
        # Keep preparation side-effect free so every episode has one recorder
        # and one pose log with a shared episode ID.
        self.declare_parameter("start_image_recording", False)
        self.declare_parameter("start_pose_logger", False)

        self.script_root = Path(str(self.get_parameter("script_root").value)).expanduser()
        self.current_episode_id = ""
        self.last_status = "idle"
        self.last_error = ""
        self.busy = False

        status_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.status_pub = self.create_publisher(String, "/uav_sim/status", status_qos)
        self.episode_id_pub = self.create_publisher(String, "/uav_sim/episode_id", status_qos)

        self.create_service(Trigger, "/uav_sim/prepare_episode", self.prepare_episode_callback)
        self.create_service(Trigger, "/uav_sim/cleanup", self.cleanup_callback)
        self.create_service(Trigger, "/uav_sim/generate_scene", self.generate_scene_callback)
        self.create_service(Trigger, "/uav_sim/setup_cameras", self.setup_cameras_callback)
        self.create_service(Trigger, "/uav_sim/start_recording", self.start_recording_callback)
        self.create_service(Trigger, "/uav_sim/stop_recording", self.stop_recording_callback)
        self.create_service(Trigger, "/uav_sim/start_pose_logger", self.start_pose_logger_callback)
        self.create_service(Trigger, "/uav_sim/stop_pose_logger", self.stop_pose_logger_callback)
        self.create_service(Trigger, "/uav_sim/plan_path", self.plan_path_callback)
        self.create_service(Trigger, "/uav_sim/stop_all", self.stop_all_callback)
        self.create_service(Trigger, "/uav_sim/get_status", self.get_status_callback)

        self.publish_status("ready")
        self.get_logger().info(f"Isaac UAV episode manager ready. script_root={self.script_root}")
        self.get_logger().info("Primary service: /uav_sim/prepare_episode")

    def publish_status(self, text: str) -> None:
        self.last_status = str(text)
        message = String()
        message.data = self.last_status
        self.status_pub.publish(message)

        episode_message = String()
        episode_message.data = self.current_episode_id
        self.episode_id_pub.publish(episode_message)

    def make_episode_id(self) -> str:
        prefix = str(self.get_parameter("episode_prefix").value).strip() or "episode"
        return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"

    def script_path(self, filename: str) -> Path:
        path = self.script_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Isaac script not found: {path}")
        return path

    def run_script(self, filename: str) -> dict:
        path = self.script_path(filename)
        self.get_logger().info(f"Running Isaac script: {path}")
        return runpy.run_path(str(path), run_name=f"__isaac_service_{path.stem}__")

    def execute(self, operation_name: str, operation: Callable[[], str], response: Trigger.Response):
        if self.busy:
            response.success = False
            response.message = f"Manager is busy: {self.last_status}"
            return response

        self.busy = True
        self.last_error = ""
        self.publish_status(f"running:{operation_name}")

        try:
            result = operation()
            self.publish_status(f"success:{operation_name}")
            response.success = True
            response.message = result
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.publish_status(f"error:{operation_name}:{self.last_error}")
            self.get_logger().error(self.last_error)
            traceback.print_exc()
            response.success = False
            response.message = self.last_error
        finally:
            self.busy = False

        return response

    @staticmethod
    def call_builtin(name: str, *args, **kwargs) -> bool:
        function = getattr(builtins, name, None)
        if not callable(function):
            return False
        function(*args, **kwargs)
        return True

    def stop_active_components(self) -> None:
        stop_functions = [
            "stop_astar_ros2_path_publisher",
            "stop_front_camera_png_recorder",
            "stop_ros2_uav_pose_publisher_logger",
            "stop_dual_uav_camera_controller",
        ]
        for name in stop_functions:
            try:
                self.call_builtin(name)
            except Exception as exc:
                self.get_logger().warning(f"Failed to call {name}: {exc}")

    def remove_generated_prims(self) -> int:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage found.")

        deleted = 0
        for prim_path in GENERATED_PRIM_PATHS:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                stage.RemovePrim(Sdf.Path(prim_path))
                deleted += 1
                self.get_logger().info(f"Removed prim: {prim_path}")
        return deleted

    def cleanup_operation(self) -> str:
        self.stop_active_components()
        deleted = self.remove_generated_prims()
        return f"Cleanup complete. deleted_prim_groups={deleted}"

    def generate_scene_operation(self) -> str:
        namespace = self.run_script("2.scene_episode_generator.py")
        generator = namespace.get("generate_scene_episode")
        if not callable(generator):
            raise RuntimeError("generate_scene_episode() was not found in the scene script.")
        records = generator()
        return f"Random scene generated. record_count={len(records)}"

    def setup_cameras_operation(self) -> str:
        self.run_script("1.dual_uav_camera.py")

        axis = str(self.get_parameter("fpv_forward_axis").value).strip().upper()
        look_down = float(self.get_parameter("fpv_look_down_m").value)
        observer_mode = str(self.get_parameter("observer_mode").value).strip().upper()

        if not self.call_builtin("set_uav_fpv_view", look_down=look_down):
            raise RuntimeError("Camera script did not register set_uav_fpv_view().")
        self.call_builtin("set_uav_fpv_axis", axis)
        self.call_builtin("set_uav_observer_mode", observer_mode)
        self.call_builtin("assign_dual_uav_viewports")

        return (
            "Cameras configured. "
            f"fpv_axis={axis}, fpv_look_down_m={look_down:.2f}, observer_mode={observer_mode}"
        )

    def ensure_recorder_loaded(self) -> None:
        # Always reload the recorder script so Isaac does not reuse a stale class.
        self.run_script("3.dual_camera_png_recorder.py")

    def start_recording_operation(self) -> str:
        if not self.current_episode_id:
            self.current_episode_id = self.make_episode_id()
        self.ensure_recorder_loaded()
        if not self.call_builtin("start_front_camera_png_recorder", self.current_episode_id):
            raise RuntimeError("start_front_camera_png_recorder() is unavailable.")
        return f"Image recording started. episode_id={self.current_episode_id}"

    def stop_recording_operation(self) -> str:
        if not self.call_builtin("stop_front_camera_png_recorder"):
            return "Image recorder was not active."
        return "Image recording stopped."

    def start_pose_logger_operation(self) -> str:
        if not self.current_episode_id:
            self.current_episode_id = self.make_episode_id()
        builtins._uav_pose_episode_id = self.current_episode_id
        if not callable(getattr(builtins, "start_ros2_uav_pose_publisher_logger", None)):
            self.run_script("5.ros2_uav_pose_publisher_logger.py")
        else:
            self.call_builtin(
                "start_ros2_uav_pose_publisher_logger",
                self.current_episode_id,
            )
        return (
            "Isaac pose publisher/logger started on /isaac_uav/pose. "
            f"episode_id={self.current_episode_id}"
        )

    def stop_pose_logger_operation(self) -> str:
        if not self.call_builtin("stop_ros2_uav_pose_publisher_logger"):
            return "Pose publisher/logger was not active."
        return "Isaac pose publisher/logger stopped."

    def plan_path_operation(self) -> str:
        self.run_script("5.astar_ros2_path_publisher.py")
        publisher = getattr(builtins, "_astar_ros2_path_publisher", None)
        if publisher is None:
            raise RuntimeError("A* path publisher did not start. Check the Isaac console for planner errors.")
        waypoint_count = len(getattr(publisher, "waypoints_ned", []))
        return f"A* path planned and published. waypoint_count={waypoint_count}"

    def prepare_episode_operation(self) -> str:
        self.current_episode_id = self.make_episode_id()
        self.stop_active_components()
        self.remove_generated_prims()

        steps = []
        steps.append(self.generate_scene_operation())
        steps.append(self.setup_cameras_operation())

        if bool(self.get_parameter("start_image_recording").value):
            steps.append(self.start_recording_operation())
        if bool(self.get_parameter("start_pose_logger").value):
            steps.append(self.start_pose_logger_operation())

        steps.append(self.plan_path_operation())
        self.publish_status(f"episode_ready:{self.current_episode_id}")
        return f"Episode prepared: {self.current_episode_id}. " + " | ".join(steps)

    def stop_all_operation(self) -> str:
        self.stop_active_components()
        return "All Isaac-side publishers, recorders, and camera controllers were stopped."

    def prepare_episode_callback(self, _request, response):
        return self.execute("prepare_episode", self.prepare_episode_operation, response)

    def cleanup_callback(self, _request, response):
        return self.execute("cleanup", self.cleanup_operation, response)

    def generate_scene_callback(self, _request, response):
        return self.execute("generate_scene", self.generate_scene_operation, response)

    def setup_cameras_callback(self, _request, response):
        return self.execute("setup_cameras", self.setup_cameras_operation, response)

    def start_recording_callback(self, _request, response):
        return self.execute("start_recording", self.start_recording_operation, response)

    def stop_recording_callback(self, _request, response):
        return self.execute("stop_recording", self.stop_recording_operation, response)

    def start_pose_logger_callback(self, _request, response):
        return self.execute("start_pose_logger", self.start_pose_logger_operation, response)

    def stop_pose_logger_callback(self, _request, response):
        return self.execute("stop_pose_logger", self.stop_pose_logger_operation, response)

    def plan_path_callback(self, _request, response):
        return self.execute("plan_path", self.plan_path_operation, response)

    def stop_all_callback(self, _request, response):
        return self.execute("stop_all", self.stop_all_operation, response)

    def get_status_callback(self, _request, response):
        response.success = not bool(self.last_error)
        response.message = (
            f"status={self.last_status}; episode_id={self.current_episode_id}; "
            f"busy={self.busy}; last_error={self.last_error or 'none'}"
        )
        return response

    def shutdown(self) -> None:
        self.stop_active_components()
        self.destroy_node()


class IsaacManagerRuntime:
    """Integrate rclpy spinning with the Isaac Sim update event stream."""

    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = IsaacUavEpisodeManager()
        self.subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self.on_update,
                name="IsaacUavEpisodeManagerUpdate",
            )
        )
        self.running = True

    def on_update(self, _event) -> None:
        if not self.running:
            return
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            print(f"[IsaacUavEpisodeManager] spin_once error: {exc}")

    def stop(self) -> None:
        self.running = False
        if self.subscription is not None:
            try:
                self.subscription.unsubscribe()
            except Exception:
                pass
            self.subscription = None
        try:
            self.node.shutdown()
        except Exception:
            traceback.print_exc()
        print("[IsaacUavEpisodeManager] Stopped.")


def stop_existing_manager() -> None:
    old_runtime = getattr(builtins, "_isaac_uav_episode_manager_runtime", None)
    if old_runtime is None:
        return
    try:
        old_runtime.stop()
    except Exception:
        traceback.print_exc()
    try:
        delattr(builtins, "_isaac_uav_episode_manager_runtime")
    except Exception:
        pass


def start_manager() -> IsaacManagerRuntime:
    stop_existing_manager()
    runtime = IsaacManagerRuntime()
    builtins._isaac_uav_episode_manager_runtime = runtime
    builtins.stop_isaac_uav_episode_manager = runtime.stop
    print("[IsaacUavEpisodeManager] Ready.")
    print("[IsaacUavEpisodeManager] Call from ROS 2:")
    print("  ros2 service call /uav_sim/prepare_episode std_srvs/srv/Trigger {}")
    print("[IsaacUavEpisodeManager] Stop inside Isaac with:")
    print("  builtins.stop_isaac_uav_episode_manager()")
    return runtime


start_manager()
