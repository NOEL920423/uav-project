# 3.dual_camera_png_recorder.py
#
# Dual-camera PNG recorder for Isaac Sim 5.1 / Omniverse.
#
# Key design:
# - Does not capture GUI viewports.
# - Does not depend on the last viewport clicked by the mouse.
# - Creates one off-screen Replicator render product for each camera prim.
# - Records synchronized FPV and TOP images into separate folders.
# - Keeps the legacy builtins.start_front_camera_png_recorder() API so existing
#   ROS2 service code and the A* runner do not need to be changed.
#
# Required camera prims:
#   /World/UAV_Camera_FPV
#   /World/UAV_Camera_Observer
#
# Output:
#   ~/uav-project/uav_vision_dataset/dual_camera_episode_xxxxx/
#       images/
#           fpv/frame_000001.png
#           top/frame_000001.png
#       camera_frames.csv

import builtins
import csv
import math
import os
import time
from datetime import datetime
from pathlib import Path

import omni.kit.app
import omni.timeline
import omni.usd
import omni.replicator.core as rep
import numpy as np
from PIL import Image
from pxr import Gf, UsdGeom


# =============================================================================
# User settings
# =============================================================================

RUN_ON_PASTE = False

FPV_CAMERA_PATH = "/World/UAV_Camera_FPV"
TOP_CAMERA_PATH = "/World/UAV_Camera_Observer"
UAV_BODY_PATH = "/World/quadrotor/body"

CAMERA_SPECS = (
    {"name": "fpv", "path": FPV_CAMERA_PATH},
    {"name": "top", "path": TOP_CAMERA_PATH},
)

DATASET_ROOT = str(Path.home() / "uav-project" / "uav_vision_dataset")
IMAGE_FOLDER_NAME = "images"
CSV_FILENAME = "camera_frames.csv"

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540

# 0.10 simulation seconds = 10 FPS. Scheduling against simulation time is
# important because Isaac often runs slower than real time; wall-clock based
# scheduling used to produce many near-identical frames per simulated second.
CAPTURE_INTERVAL_S = 0.10

MAX_FRAME_COUNT = None
AUTO_STOP_AFTER_SECONDS = None
PRINT_STATUS = True
PRINT_INTERVAL_S = 2.0
ALLOW_MISSING_UAV_BODY = True

# When True, each PNG is saved as RGB instead of RGBA.
SAVE_AS_RGB = True


# =============================================================================
# Logging and USD helpers
# =============================================================================

def log(message):
    print(f"[DualCameraPNGRecorder] {message}")


def warn(message):
    print(f"[DualCameraPNGRecorder][Warning] {message}")


def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage found.")
    return stage


def prim_exists(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    return bool(prim and prim.IsValid())


def get_sim_time(event):
    event_time = getattr(event, "current_time", None)
    if event_time is not None:
        return float(event_time)
    try:
        return float(omni.timeline.get_timeline_interface().get_current_time())
    except Exception:
        return None


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

    translation = matrix.ExtractTranslation()
    return (
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    )


def try_get_yaw_from_matrix(matrix):
    if matrix is None:
        return None

    try:
        local_x = Gf.Vec3d(1.0, 0.0, 0.0)
        world_x = matrix.TransformDir(local_x)
        return float(math.atan2(world_x[1], world_x[0]))
    except Exception:
        return None


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def make_episode_dir(episode_id=None):
    if episode_id is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_episode_id = f"dual_camera_episode_{stamp}"
    else:
        resolved_episode_id = f"dual_camera_episode_{episode_id}"

    episode_dir = Path(DATASET_ROOT).expanduser() / resolved_episode_id
    images_root = episode_dir / IMAGE_FOLDER_NAME

    ensure_directory(episode_dir)
    ensure_directory(images_root)

    image_dirs = {}
    for camera_spec in CAMERA_SPECS:
        camera_dir = images_root / camera_spec["name"]
        ensure_directory(camera_dir)
        image_dirs[camera_spec["name"]] = str(camera_dir)

    return resolved_episode_id, str(episode_dir), image_dirs


# =============================================================================
# Image conversion helpers
# =============================================================================

def extract_annotator_array(raw_data):
    """Return an HxWxC uint8 NumPy array from a Replicator RGB annotator result."""
    if raw_data is None:
        return None

    if isinstance(raw_data, dict):
        raw_data = raw_data.get("data")

    if raw_data is None:
        return None

    try:
        array = np.asarray(raw_data)
    except Exception:
        return None

    if array.size == 0 or array.ndim < 3:
        return None

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.shape[2] < 3:
        return None

    return array


def save_rgb_array_to_png(rgb_array, image_path):
    if rgb_array is None:
        raise RuntimeError("RGB annotator returned no valid image data.")

    channel_count = int(rgb_array.shape[2])

    if channel_count >= 4:
        image = Image.fromarray(rgb_array[:, :, :4], mode="RGBA")
        if SAVE_AS_RGB:
            image = image.convert("RGB")
    else:
        image = Image.fromarray(rgb_array[:, :, :3], mode="RGB")

    image.save(image_path, format="PNG")


# =============================================================================
# Recorder
# =============================================================================

class DualCameraPNGRecorder:
    def __init__(self):
        self.stage = get_stage()
        self.subscription = None

        self.render_products = {}
        self.rgb_annotators = {}

        self.episode_id = None
        self.episode_dir = None
        self.image_dirs = None
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None

        self.is_running = False
        self.start_wall_time = None
        self.last_capture_wall_time = 0.0
        self.last_capture_sim_time = None
        self.last_print_wall_time = 0.0

        self.frame_index = 0
        self.capture_count = 0
        self.skipped_capture_count = 0
        self.external_episode_id = None

    def start(self):
        if self.is_running:
            log("Recorder is already running.")
            return

        self._validate_scene()
        self._setup_output()

        try:
            self._setup_render_products()
        except Exception:
            self._close_csv()
            raise

        self.start_wall_time = time.time()
        self.last_capture_wall_time = 0.0
        self.last_capture_sim_time = None
        self.last_print_wall_time = 0.0
        self.frame_index = 0
        self.capture_count = 0
        self.skipped_capture_count = 0
        self.is_running = True

        self.subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update,
                name="DualCameraPNGRecorderUpdate",
            )
        )

        log("Started with off-screen Replicator render products.")
        log("Viewport selection and mouse focus are no longer used.")
        log(f"Episode ID: {self.episode_id}")
        log(f"FPV output: {self.image_dirs['fpv']}")
        log(f"TOP output: {self.image_dirs['top']}")
        log(f"CSV output: {self.csv_path}")

    def stop(self):
        was_running = self.is_running
        self.is_running = False

        if self.subscription is not None:
            try:
                self.subscription.unsubscribe()
            except Exception as exc:
                warn(f"Failed to unsubscribe update callback: {exc}")
            self.subscription = None

        self._destroy_render_products()
        self._close_csv()

        if was_running:
            log(
                "Stopped. "
                f"captured_pairs={self.capture_count}, "
                f"skipped_attempts={self.skipped_capture_count}"
            )

    def _validate_scene(self):
        for camera_spec in CAMERA_SPECS:
            camera_name = camera_spec["name"]
            camera_path = camera_spec["path"]

            if not prim_exists(self.stage, camera_path):
                raise RuntimeError(
                    f"{camera_name} camera does not exist: {camera_path}. "
                    "Run 1.dual_uav_camera.py first."
                )

        if not prim_exists(self.stage, UAV_BODY_PATH):
            message = f"UAV body does not exist: {UAV_BODY_PATH}"
            if ALLOW_MISSING_UAV_BODY:
                warn(message + ". Pose columns will be empty.")
            else:
                raise RuntimeError(message)

    def _setup_output(self):
        self.episode_id, self.episode_dir, self.image_dirs = make_episode_dir(
            self.external_episode_id
        )
        self.csv_path = os.path.join(self.episode_dir, CSV_FILENAME)

        self.csv_file = open(
            self.csv_path,
            "w",
            newline="",
            encoding="utf-8",
        )

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
                "capture_clock",
                "image_width",
                "image_height",
            ],
        )
        self.csv_writer.writeheader()
        self.csv_file.flush()

    def _setup_render_products(self):
        # Capture must stay active while the Isaac timeline is playing.
        try:
            rep.orchestrator.set_capture_on_play(True)
        except Exception as exc:
            warn(f"Could not set capture_on_play=True: {exc}")

        for camera_spec in CAMERA_SPECS:
            camera_name = camera_spec["name"]
            camera_path = camera_spec["path"]

            try:
                render_product = rep.create.render_product(
                    camera_path,
                    resolution=(int(IMAGE_WIDTH), int(IMAGE_HEIGHT)),
                    force_new=True,
                )
            except TypeError:
                # Compatibility fallback for Replicator versions without force_new.
                render_product = rep.create.render_product(
                    camera_path,
                    (int(IMAGE_WIDTH), int(IMAGE_HEIGHT)),
                )

            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach(render_product)

            self.render_products[camera_name] = render_product
            self.rgb_annotators[camera_name] = annotator

            log(
                f"Render product created: camera={camera_path}, "
                f"resolution={IMAGE_WIDTH}x{IMAGE_HEIGHT}"
            )

        # Initialize the Replicator graph. If preview is unavailable or already active,
        # the normal Isaac update loop will still warm the annotators up.
        try:
            rep.orchestrator.preview()
        except Exception as exc:
            warn(f"Replicator preview was skipped: {exc}")

    def _destroy_render_products(self):
        for camera_name, annotator in list(self.rgb_annotators.items()):
            render_product = self.render_products.get(camera_name)

            try:
                if render_product is not None:
                    annotator.detach(render_product)
                else:
                    annotator.detach()
            except TypeError:
                try:
                    annotator.detach()
                except Exception as exc:
                    warn(f"Could not detach {camera_name} annotator: {exc}")
            except Exception as exc:
                warn(f"Could not detach {camera_name} annotator: {exc}")

        self.rgb_annotators.clear()

        for camera_name, render_product in list(self.render_products.items()):
            try:
                render_product.destroy()
            except Exception as exc:
                warn(f"Could not destroy {camera_name} render product: {exc}")

        self.render_products.clear()

    def _close_csv(self):
        if self.csv_file is not None:
            try:
                self.csv_file.flush()
                self.csv_file.close()
            except Exception as exc:
                warn(f"Failed to close CSV: {exc}")

        self.csv_file = None
        self.csv_writer = None

    def _should_stop(self, now):
        if MAX_FRAME_COUNT is not None:
            if self.capture_count >= int(MAX_FRAME_COUNT):
                return True

        if AUTO_STOP_AFTER_SECONDS is not None and self.start_wall_time is not None:
            if now - self.start_wall_time >= float(AUTO_STOP_AFTER_SECONDS):
                return True

        return False

    def _read_all_camera_images(self):
        image_arrays = {}

        for camera_spec in CAMERA_SPECS:
            camera_name = camera_spec["name"]
            annotator = self.rgb_annotators.get(camera_name)

            if annotator is None:
                return None, f"Missing RGB annotator for {camera_name}."

            try:
                raw_data = annotator.get_data()
                rgb_array = extract_annotator_array(raw_data)
            except Exception as exc:
                return None, f"Failed to read {camera_name} annotator: {exc}"

            if rgb_array is None:
                return None, f"{camera_name} render product is still warming up."

            image_arrays[camera_name] = rgb_array

        return image_arrays, ""

    def _capture_one_frame(self, event):
        image_arrays, error_message = self._read_all_camera_images()

        if image_arrays is None:
            self.skipped_capture_count += 1
            if self.skipped_capture_count <= 5:
                warn(error_message)
            return False

        next_frame_index = self.frame_index + 1
        image_name = f"frame_{next_frame_index:06d}.png"
        image_paths = {
            camera_name: os.path.join(self.image_dirs[camera_name], image_name)
            for camera_name in self.image_dirs
        }

        saved_paths = []

        try:
            for camera_spec in CAMERA_SPECS:
                camera_name = camera_spec["name"]
                image_path = image_paths[camera_name]
                save_rgb_array_to_png(image_arrays[camera_name], image_path)
                saved_paths.append(image_path)
        except Exception as exc:
            for saved_path in saved_paths:
                try:
                    os.remove(saved_path)
                except OSError:
                    pass

            self.skipped_capture_count += 1
            warn(f"Synchronized image pair was not saved: {exc}")
            return False

        now_wall = time.time()
        record_time = now_wall - self.start_wall_time
        sim_time = get_sim_time(event)

        uav_position = get_world_position_xyz(self.stage, UAV_BODY_PATH)
        uav_matrix = get_world_matrix(self.stage, UAV_BODY_PATH)
        uav_yaw = try_get_yaw_from_matrix(uav_matrix)

        if uav_position is None:
            uav_x, uav_y, uav_z = "", "", ""
        else:
            uav_x, uav_y, uav_z = uav_position

        row = {
            "episode_id": self.episode_id,
            "frame_index": next_frame_index,
            "time_wall": now_wall,
            "record_time": record_time,
            "sim_time": "" if sim_time is None else float(sim_time),
            "fpv_image_path": image_paths["fpv"],
            "top_image_path": image_paths["top"],
            "fpv_camera_path": FPV_CAMERA_PATH,
            "top_camera_path": TOP_CAMERA_PATH,
            "uav_body_path": UAV_BODY_PATH,
            "uav_x_isaac": uav_x,
            "uav_y_isaac": uav_y,
            "uav_z_isaac": uav_z,
            "uav_yaw_approx_rad": "" if uav_yaw is None else uav_yaw,
            "capture_interval_s": CAPTURE_INTERVAL_S,
            "capture_clock": "sim_time" if sim_time is not None else "wall_time",
            "image_width": IMAGE_WIDTH,
            "image_height": IMAGE_HEIGHT,
        }

        self.csv_writer.writerow(row)

        self.frame_index = next_frame_index
        self.capture_count += 1

        if self.capture_count % 10 == 0:
            self.csv_file.flush()

        return True

    def _print_status_if_needed(self, now):
        if not PRINT_STATUS:
            return

        if now - self.last_print_wall_time < PRINT_INTERVAL_S:
            return

        self.last_print_wall_time = now
        elapsed = now - self.start_wall_time
        log(
            f"recording... pairs={self.capture_count}, "
            f"skipped={self.skipped_capture_count}, "
            f"elapsed={elapsed:.1f}s, "
            f"output={self.episode_dir}"
        )

    def _on_update(self, event):
        if not self.is_running:
            return

        now = time.time()

        if self._should_stop(now):
            log("Auto-stop condition reached.")
            self.stop()
            return

        sim_time = get_sim_time(event)
        if sim_time is not None:
            # A timeline reset may move time backwards.  Treat the next frame
            # as the start of a new scheduling interval.
            if (
                self.last_capture_sim_time is not None
                and sim_time < self.last_capture_sim_time
            ):
                self.last_capture_sim_time = None
            if (
                self.last_capture_sim_time is not None
                and sim_time - self.last_capture_sim_time < CAPTURE_INTERVAL_S
            ):
                self._print_status_if_needed(now)
                return
            self.last_capture_sim_time = sim_time
        else:
            if now - self.last_capture_wall_time < CAPTURE_INTERVAL_S:
                self._print_status_if_needed(now)
                return
            self.last_capture_wall_time = now

        self._capture_one_frame(event)
        self._print_status_if_needed(now)


# Legacy class alias for code that imports the old class name.
FrontCameraPNGRecorder = DualCameraPNGRecorder


# =============================================================================
# Public builtins API
# =============================================================================

def _get_existing_recorder():
    recorder = getattr(builtins, "_dual_camera_png_recorder", None)
    if recorder is None:
        recorder = getattr(builtins, "_front_camera_png_recorder", None)
    return recorder


def _clear_recorder_references():
    for attribute_name in (
        "_dual_camera_png_recorder",
        "_front_camera_png_recorder",
    ):
        try:
            delattr(builtins, attribute_name)
        except Exception:
            pass


def stop_existing_dual_camera_png_recorder():
    recorder = _get_existing_recorder()
    if recorder is None:
        return

    try:
        recorder.stop()
        log("Previous recorder stopped.")
    except Exception as exc:
        warn(f"Failed to stop previous recorder: {exc}")

    _clear_recorder_references()


def start_dual_camera_png_recorder(episode_id=None):
    stop_existing_dual_camera_png_recorder()

    recorder = DualCameraPNGRecorder()
    if episode_id is not None:
        recorder.external_episode_id = str(episode_id)

    # Keep both names for compatibility with the existing ROS2 service and A* code.
    builtins._dual_camera_png_recorder = recorder
    builtins._front_camera_png_recorder = recorder

    try:
        recorder.start()
    except Exception:
        _clear_recorder_references()
        raise

    log("Recording started.")
    return recorder


def stop_dual_camera_png_recorder():
    recorder = _get_existing_recorder()

    if recorder is None:
        log("No active recorder.")
        return

    try:
        recorder.stop()
    finally:
        _clear_recorder_references()


def print_dual_camera_png_recorder_status():
    recorder = _get_existing_recorder()

    if recorder is None:
        log("No active recorder.")
        return

    log("Status:")
    log(f"  running: {recorder.is_running}")
    log(f"  episode_id: {recorder.episode_id}")
    log(f"  captured_pairs: {recorder.capture_count}")
    log(f"  skipped_attempts: {recorder.skipped_capture_count}")
    log(f"  episode_dir: {recorder.episode_dir}")
    log(f"  csv_path: {recorder.csv_path}")


# Preserve the old API names used by the ROS2 service and A* runner.
def start_front_camera_png_recorder(episode_id=None):
    return start_dual_camera_png_recorder(episode_id=episode_id)


def stop_front_camera_png_recorder():
    return stop_dual_camera_png_recorder()


def print_front_camera_png_recorder_status():
    return print_dual_camera_png_recorder_status()


# Stop any old viewport-based recorder before replacing the builtins functions.
stop_existing_dual_camera_png_recorder()

builtins.start_dual_camera_png_recorder = start_dual_camera_png_recorder
builtins.stop_dual_camera_png_recorder = stop_dual_camera_png_recorder
builtins.print_dual_camera_png_recorder_status = print_dual_camera_png_recorder_status

builtins.start_front_camera_png_recorder = start_front_camera_png_recorder
builtins.stop_front_camera_png_recorder = stop_front_camera_png_recorder
builtins.print_front_camera_png_recorder_status = print_front_camera_png_recorder_status

log("Ready.")
log("This version records FPV and TOP from independent render products.")
log("Mouse focus and active viewport selection are irrelevant.")
log("Start through ROS2 service or builtins.start_dual_camera_png_recorder().")

if RUN_ON_PASTE:
    start_dual_camera_png_recorder()
