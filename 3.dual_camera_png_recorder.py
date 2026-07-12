# 3.front_camera_png_recorder.py
#
# Front camera PNG recorder for Isaac Sim / Omniverse.
#
# Purpose:
#   Record the UAV FPV camera view as PNG images and write a frame metadata CSV.
#
# Run inside Isaac Sim:
#   Window -> Script Editor -> paste/run
#
# Recommended workflow:
#   1. Run dual_uav_camera.py first.
#   2. Run this recorder.
#   3. Run 2.px4_astar.py.
#
# Output:
#   ~/uav-project/uav_vision_dataset/episode_xxxxx/
#       images/
#           frame_000001.png
#           frame_000002.png
#           ...
#       camera_frames.csv
#
# Notes:
#   - This script does not control the UAV.
#   - This script does not read obstacle ground-truth.
#   - This script records only what the FPV camera sees.
#   - The camera_frames.csv can later be synchronized with mission_log.csv by time_wall.

import builtins
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import omni
import omni.usd
import omni.kit.app
from pxr import UsdGeom, Gf

try:
    from omni.kit.viewport.utility import get_active_viewport
except Exception:
    get_active_viewport = None

try:
    from omni.kit.viewport.utility import get_viewport_window_instances
except Exception:
    get_viewport_window_instances = None

try:
    from omni.kit.viewport.utility import capture_viewport_to_file
except Exception:
    capture_viewport_to_file = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import omni.replicator.core as rep
except Exception:
    rep = None


# =============================================================================
# User settings
# =============================================================================

RUN_ON_PASTE = False

FPV_CAMERA_PATH = "/World/UAV_Camera_FPV"
TOP_CAMERA_PATH = "/World/UAV_Camera_Observer"

CAMERA_SPECS = [
    {
        "name": "fpv",
        "path": FPV_CAMERA_PATH,
    },
    {
        "name": "top",
        "path": TOP_CAMERA_PATH,
    },
]

UAV_BODY_PATH = "/World/quadrotor/body"

DATASET_ROOT = str(Path.home() / "uav-project" / "uav_vision_dataset")

IMAGE_FOLDER_NAME = "images"
CSV_FILENAME = "camera_frames.csv"

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540

# 0.20 s = 5 FPS. This is safer for Isaac Sim + PX4.
# If stable, you can try 0.10 s = 10 FPS.
CAPTURE_INTERVAL_S = 0.20

# If True, the active viewport will be switched to CAMERA_PATH once at start.
# This is the most practical way to capture a front-camera PNG quickly.
SET_ACTIVE_VIEWPORT_TO_CAMERA = True

# If True, tries to resize active viewport resolution.
# Some Isaac Sim versions may ignore this safely.
TRY_SET_VIEWPORT_RESOLUTION = True

# Recorder behavior.
MAX_FRAME_COUNT = None
PRINT_STATUS = True
PRINT_INTERVAL_S = 2.0

# Stop automatically after this many seconds.
# Set to None to record until builtins.stop_front_camera_png_recorder() is called.
AUTO_STOP_AFTER_SECONDS = None

# If True, keep recording even if UAV body path is missing.
ALLOW_MISSING_UAV_BODY = True


# =============================================================================
# Helper functions
# =============================================================================

def log(message):
    print(f"[FrontCameraPNGRecorder] {message}")


def warn(message):
    print(f"[FrontCameraPNGRecorder][Warning] {message}")


def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage found.")
    return stage


def prim_exists(stage, path):
    prim = stage.GetPrimAtPath(path)
    return bool(prim and prim.IsValid())


def get_world_matrix(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None

    try:
        return omni.usd.get_world_transform_matrix(prim)
    except Exception:
        cache = UsdGeom.XformCache()
        return cache.GetLocalToWorldTransform(prim)


def get_world_position_xyz(stage, prim_path):
    matrix = get_world_matrix(stage, prim_path)
    if matrix is None:
        return None

    t = matrix.ExtractTranslation()
    return float(t[0]), float(t[1]), float(t[2])


def try_get_yaw_from_matrix(matrix):
    if matrix is None:
        return None

    try:
        # Approximate yaw from transformed local X axis.
        local_x = Gf.Vec3d(1.0, 0.0, 0.0)
        world_x = matrix.TransformDir(local_x)
        return float(__import__("math").atan2(world_x[1], world_x[0]))
    except Exception:
        return None


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_episode_dir(episode_id=None):
    if episode_id is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        episode_id = f"dual_camera_episode_{stamp}"
    else:
        episode_id = f"dual_camera_episode_{episode_id}"

    episode_dir = Path(DATASET_ROOT).expanduser() / episode_id
    images_dir = episode_dir / IMAGE_FOLDER_NAME

    ensure_directory(episode_dir)
    ensure_directory(images_dir)

    image_dirs = {}

    for camera_spec in CAMERA_SPECS:
        camera_name = camera_spec["name"]
        camera_dir = images_dir / camera_name
        ensure_directory(camera_dir)
        image_dirs[camera_name] = str(camera_dir)

    return episode_id, str(episode_dir), image_dirs


def set_viewport_camera_and_resolution(viewport):
    if viewport is None:
        return

    if TRY_SET_VIEWPORT_RESOLUTION:
        try:
            viewport.resolution = (int(IMAGE_WIDTH), int(IMAGE_HEIGHT))
            log(f"Active viewport resolution requested: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        except Exception as exc:
            warn(f"Could not set viewport resolution: {exc}")

    if TRY_SET_VIEWPORT_RESOLUTION:
        try:
            viewport.resolution = (int(IMAGE_WIDTH), int(IMAGE_HEIGHT))
            log(f"Active viewport resolution requested: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        except Exception as exc:
            warn(f"Could not set viewport resolution: {exc}")


# =============================================================================
# Recorder class
# =============================================================================

class FrontCameraPNGRecorder:
    def __init__(self):
        self.stage = get_stage()
        self.viewport = None
        self.viewports_by_name = {}
        self.render_products = {}
        self.rgb_annotators = {}  
        self.subscription = None

        self.episode_id = None
        self.episode_dir = None
        self.image_dirs = None
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None

        self.is_running = False
        self.start_wall_time = None
        self.last_capture_wall_time = 0.0
        self.last_print_wall_time = 0.0

        self.frame_index = 0
        self.capture_count = 0



    def start(self):
        for camera_spec in CAMERA_SPECS:
            camera_path = camera_spec["path"]
            camera_name = camera_spec["name"]

            if not prim_exists(self.stage, camera_path):
                raise RuntimeError(
                    f"{camera_name} camera does not exist: {camera_path}. "
                    "Run dual_uav_camera.py first."
                )

        if not prim_exists(self.stage, UAV_BODY_PATH):
            message = f"UAV body does not exist: {UAV_BODY_PATH}"
            if ALLOW_MISSING_UAV_BODY:
                warn(message + ". Recorder will continue without UAV pose columns.")
            else:
                raise RuntimeError(message)

        if get_active_viewport is None:
            raise RuntimeError("get_active_viewport is not available in this Isaac Sim environment.")

        if capture_viewport_to_file is None:
            raise RuntimeError("capture_viewport_to_file is not available in this Isaac Sim environment.")

        self._setup_camera_render_products()

        external_episode_id = getattr(self, "external_episode_id", None)
        self.episode_id, self.episode_dir, self.image_dirs = make_episode_dir(external_episode_id)        
        self.csv_path = os.path.join(self.episode_dir, CSV_FILENAME)

        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "episode_id",
                "frame_index",
                "time_wall",
                "record_time",
                "sim_time",

                "fpv_image_path",
                "top_image_path",
                "fpv_camera_path",
                "top_camera_path",
                
                "uav_body_path",
                "uav_x_isaac",
                "uav_y_isaac",
                "uav_z_isaac",
                "uav_yaw_approx_rad",
                "capture_interval_s",
                "image_width",
                "image_height",
            ],
        )
        self.csv_writer.writeheader()
        self.csv_file.flush()

        self.start_wall_time = time.time()
        self.last_capture_wall_time = 0.0
        self.last_print_wall_time = 0.0
        self.frame_index = 0
        self.capture_count = 0
        self.is_running = True

        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="FrontCameraPNGRecorderUpdate",
        )

        log("Started.")
        log(f"Episode ID: {self.episode_id}")
        log(f"CSV output: {self.csv_path}")
        log("Stop with: builtins.stop_front_camera_png_recorder()")
        log(f"FPV image output: {self.image_dirs.get('fpv', '')}")
        log(f"TOP image output: {self.image_dirs.get('top', '')}")
        log(f"FPV camera: {FPV_CAMERA_PATH}")
        log(f"TOP camera: {TOP_CAMERA_PATH}")

    def stop(self):
        self.is_running = False

        if self.subscription is not None:
            try:
                self.subscription.unsubscribe()
                log("Update subscription stopped.")
            except Exception as exc:
                warn(f"Failed to unsubscribe update callback: {exc}")
            self.subscription = None

        if self.csv_file is not None:
            try:
                self.csv_file.flush()
                self.csv_file.close()
                log(f"CSV closed: {self.csv_path}")
            except Exception as exc:
                warn(f"Failed to close CSV: {exc}")

        self.csv_file = None
        self.csv_writer = None

        log(f"Stopped. Captured frames: {self.capture_count}")

    def _should_stop(self, now):
        if MAX_FRAME_COUNT is not None and self.capture_count >= int(MAX_FRAME_COUNT):
            return True

        if AUTO_STOP_AFTER_SECONDS is not None:
            if now - self.start_wall_time >= float(AUTO_STOP_AFTER_SECONDS):
                return True

        return False

    def _capture_viewport_to_file(self, viewport, image_path):
        try:
            try:
                capture_viewport_to_file(viewport, image_path)
            except TypeError:
                capture_viewport_to_file(image_path, viewport)

            return True

        except Exception as exc:
            warn(f"Capture failed: {image_path}, reason={exc}")
            return False

    def _capture_one_frame(self, event):
        self.frame_index += 1

        image_paths = {}

        for camera_spec in CAMERA_SPECS:
            camera_name = camera_spec["name"]

            viewport = self.viewports_by_name.get(camera_name)
            if viewport is None:
                warn(f"No viewport found for camera name: {camera_name}")
                image_paths[camera_name] = ""
                continue

            image_name = f"frame_{self.frame_index:06d}.png"
            image_path = os.path.join(self.image_dirs[camera_name], image_name)

            success = self._capture_viewport_to_file(viewport, image_path)

            if success:
                image_paths[camera_name] = image_path
            else:
                image_paths[camera_name] = ""

        now_wall = time.time()
        record_time = now_wall - self.start_wall_time
        sim_time = getattr(event, "current_time", None)

        uav_pos = get_world_position_xyz(self.stage, UAV_BODY_PATH)
        uav_matrix = get_world_matrix(self.stage, UAV_BODY_PATH)
        uav_yaw = try_get_yaw_from_matrix(uav_matrix)

        if uav_pos is None:
            uav_x, uav_y, uav_z = "", "", ""
        else:
            uav_x, uav_y, uav_z = uav_pos

        row = {
            "episode_id": self.episode_id,
            "frame_index": self.frame_index,
            "time_wall": now_wall,
            "record_time": record_time,
            "sim_time": "" if sim_time is None else float(sim_time),

            "fpv_image_path": image_paths.get("fpv", ""),
            "top_image_path": image_paths.get("top", ""),
            "fpv_camera_path": FPV_CAMERA_PATH,
            "top_camera_path": TOP_CAMERA_PATH,

            "uav_body_path": UAV_BODY_PATH,
            "uav_x_isaac": uav_x,
            "uav_y_isaac": uav_y,
            "uav_z_isaac": uav_z,
            "uav_yaw_approx_rad": "" if uav_yaw is None else uav_yaw,
            "capture_interval_s": CAPTURE_INTERVAL_S,
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
        }

        self.csv_writer.writerow(row)
        self.capture_count += 1

        if self.capture_count % 10 == 0:
            self.csv_file.flush()

    def _print_status_if_needed(self, now):
        if not PRINT_STATUS:
            return

        if now - self.last_print_wall_time < PRINT_INTERVAL_S:
            return

        self.last_print_wall_time = now
        elapsed = now - self.start_wall_time
        log(
            f"recording... frames={self.capture_count}, "
            f"elapsed={elapsed:.1f}s, output={self.episode_dir}"
        )

    def _on_update(self, event):
        if not self.is_running:
            return

        now = time.time()

        if self._should_stop(now):
            log("Auto stop condition reached.")
            self.stop()
            return

        if now - self.last_capture_wall_time < CAPTURE_INTERVAL_S:
            self._print_status_if_needed(now)
            return

        self.last_capture_wall_time = now
        self._capture_one_frame(event)
        self._print_status_if_needed(now)


# =============================================================================
# Public helper functions
# =============================================================================

def collect_available_viewports():
    viewports = []

    if get_active_viewport is not None:
        try:
            active_viewport = get_active_viewport()
            if active_viewport is not None:
                viewports.append(active_viewport)
        except Exception as exc:
            warn(f"Could not get active viewport: {exc}")

    if get_viewport_window_instances is not None:
        try:
            windows = list(get_viewport_window_instances())
            for window in windows:
                viewport_api = getattr(window, "viewport_api", None)
                if viewport_api is not None:
                    viewports.append(viewport_api)
        except Exception as exc:
            warn(f"Could not list viewport windows: {exc}")

    unique_viewports = []
    seen_ids = set()

    for viewport in viewports:
        viewport_id = id(viewport)
        if viewport_id in seen_ids:
            continue
        seen_ids.add(viewport_id)
        unique_viewports.append(viewport)

    return unique_viewports


def get_viewport_camera_path(viewport):
    try:
        return str(getattr(viewport, "camera_path", ""))
    except Exception:
        return ""


def find_viewports_by_camera_path():
    result = {}

    viewports = collect_available_viewports()

    print("[DualCameraPNGRecorder] Available viewports:")

    for index, viewport in enumerate(viewports):
        camera_path = get_viewport_camera_path(viewport)
        print(f"  viewport[{index}] camera_path={camera_path}")

        for camera_spec in CAMERA_SPECS:
            camera_name = camera_spec["name"]
            expected_path = camera_spec["path"]

            if camera_path == expected_path:
                result[camera_name] = viewport

    missing = []

    for camera_spec in CAMERA_SPECS:
        camera_name = camera_spec["name"]
        if camera_name not in result:
            missing.append(camera_name)

    if missing:
        raise RuntimeError(
            "Could not find viewport(s) for camera(s): "
            + ", ".join(missing)
            + ". Make sure View 1 uses /World/UAV_Camera_FPV "
            + "and View 2 uses /World/UAV_Camera_Observer."
        )

    return result

def stop_existing_front_camera_png_recorder():
    old_recorder = getattr(builtins, "_front_camera_png_recorder", None)

    if old_recorder is None:
        return

    try:
        old_recorder.stop()
        log("Stopped old recorder.")
    except Exception as exc:
        warn(f"Failed to stop old recorder: {exc}")

    try:
        delattr(builtins, "_front_camera_png_recorder")
    except Exception:
        pass


def stop_front_camera_png_recorder():
    recorder = getattr(builtins, "_front_camera_png_recorder", None)

    if recorder is None:
        log("No active recorder.")
        return

    recorder.stop()

    try:
        delattr(builtins, "_front_camera_png_recorder")
    except Exception:
        pass


def print_front_camera_png_recorder_status():
    recorder = getattr(builtins, "_front_camera_png_recorder", None)

    if recorder is None:
        log("No active recorder.")
        return

    log("Status:")
    log(f"  running: {recorder.is_running}")
    log(f"  episode_id: {recorder.episode_id}")
    log(f"  frames: {recorder.capture_count}")
    log(f"  episode_dir: {recorder.episode_dir}")
    log(f"  csv_path: {recorder.csv_path}")


builtins.stop_front_camera_png_recorder = stop_front_camera_png_recorder
builtins.print_front_camera_png_recorder_status = print_front_camera_png_recorder_status


# =============================================================================
# Start recorder
# =============================================================================

def start_front_camera_png_recorder(episode_id=None):
    old_recorder = getattr(builtins, "_front_camera_png_recorder", None)

    if old_recorder is not None:
        try:
            old_recorder.stop()
            print("[FrontCameraPNGRecorder] Old recorder stopped before starting new one.")
        except Exception as exc:
            print(f"[FrontCameraPNGRecorder] Failed to stop old recorder: {exc}")

        try:
            delattr(builtins, "_front_camera_png_recorder")
        except Exception:
            pass

    recorder = FrontCameraPNGRecorder()

    if episode_id is not None:
        recorder.external_episode_id = str(episode_id)

    builtins._front_camera_png_recorder = recorder
    recorder.start()
    print("[FrontCameraPNGRecorder] Recording started by A* runner.")


def stop_front_camera_png_recorder():
    recorder = getattr(builtins, "_front_camera_png_recorder", None)

    if recorder is None:
        print("[FrontCameraPNGRecorder] No active recorder.")
        return

    try:
        recorder.stop()
        print("[FrontCameraPNGRecorder] Recording stopped.")
    except Exception as exc:
        print(f"[FrontCameraPNGRecorder] Failed to stop recorder: {exc}")

    try:
        delattr(builtins, "_front_camera_png_recorder")
    except Exception:
        pass


def print_front_camera_png_recorder_status():
    recorder = getattr(builtins, "_front_camera_png_recorder", None)

    if recorder is None:
        print("[FrontCameraPNGRecorder] No active recorder.")
        return

    print("[FrontCameraPNGRecorder] Status:")
    print(f"  running: {recorder.is_running}")
    print(f"  episode_id: {recorder.episode_id}")
    print(f"  frames: {recorder.capture_count}")
    print(f"  episode_dir: {recorder.episode_dir}")
    print(f"  csv_path: {recorder.csv_path}")


builtins.start_front_camera_png_recorder = start_front_camera_png_recorder
builtins.stop_front_camera_png_recorder = stop_front_camera_png_recorder
builtins.print_front_camera_png_recorder_status = print_front_camera_png_recorder_status

print("[FrontCameraPNGRecorder] Ready.")
print("[FrontCameraPNGRecorder] This script will NOT record immediately.")
print("[FrontCameraPNGRecorder] A* runner should call:")
print("  builtins.start_front_camera_png_recorder(episode_id)")
print("  builtins.stop_front_camera_png_recorder()")

# else:
#     log("RUN_ON_PASTE=False. Create FrontCameraPNGRecorder() manually if needed.")