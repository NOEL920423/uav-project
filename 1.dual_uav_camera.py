"""
Dual UAV Camera Controller for Isaac Sim / Omniverse

Purpose
-------
Create and update two cameras at the same time:

1. FPV wide-angle nose camera
   - Use this for manual flying.
   - It behaves like a UAV first-person camera.
   - It looks forward and slightly downward so lower target disks are easier to see.
   - It uses a short focal length and larger aperture for a wide-angle feeling.

2. Observer camera
   - Use this for spatial awareness.
   - It can be a high chase camera or a top-down camera.

Run inside Isaac Sim:
    Window -> Script Editor -> paste/run
or execute it from Isaac Sim's Python environment.

After running, open two viewports if possible:
    View 1 -> Cameras -> /World/UAV_Camera_FPV
    View 2 -> Cameras -> /World/UAV_Camera_Observer

This script creates exactly one dual-camera controller at a time.
Re-running this file automatically stops the previous controller first.
"""

import builtins
import math
from dataclasses import dataclass

import omni
import omni.kit.app
from pxr import Usd, UsdGeom, Gf

import time

try:
    from omni.kit.viewport.utility import get_active_viewport
except Exception:
    get_active_viewport = None

try:
    from omni.kit.viewport.utility import get_viewport_window_instances
except Exception:
    get_viewport_window_instances = None

try:
    from omni.kit.viewport.utility import create_viewport_window
except Exception:
    create_viewport_window = None


# ============================================================
# User settings
# ============================================================

# Recommended target for Pegasus quadrotor.
# If the camera does not follow the UAV, confirm which prim actually moves.
TARGET_PRIM_PATH = "/World/quadrotor/body"

# If True, the currently selected prim in the Stage will override TARGET_PRIM_PATH.
USE_SELECTED_PRIM_AS_TARGET = False

# Camera prim paths created by this script.
FPV_CAMERA_PATH = "/World/UAV_Camera_FPV"
OBSERVER_CAMERA_PATH = "/World/UAV_Camera_Observer"

# Main viewport camera.
# Valid values: "FPV", "OBSERVER"
PRIMARY_VIEW_CAMERA = "FPV"

# Try to assign cameras to two Isaac Sim viewport windows automatically.
# If this fails, you can still manually select the cameras from each viewport camera menu.
AUTO_ASSIGN_VIEWPORTS = True
AUTO_CREATE_SECOND_VIEWPORT = True

# FPV camera settings.
# Direction source:
#   "BODY_AXIS" : true first-person style; camera looks along the UAV body axis.
#   "MOTION"    : camera looks along recent horizontal movement direction; more stable for keyboard flying.
FPV_DIRECTION_SOURCE = "BODY_AXIS"
FPV_FORWARD_AXIS = "X"     # Try "X", "-X", "Y", "-Y" if the FPV direction is wrong.
FPV_FORWARD_OFFSET_M = 0.45
FPV_HEIGHT_M = 0.12
FPV_LOOK_AHEAD_M = 3.5
FPV_LOOK_DOWN_M = -2.4
FPV_FOCAL_LENGTH = 12.0
FPV_HORIZONTAL_APERTURE = 28.0

# Observer camera settings.
# Valid modes: "CHASE", "TOP"
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

# Common camera settings.
CLIPPING_RANGE = (0.05, 10000.0)
SMOOTHING = 0.18
MIN_MOVE_DISTANCE_M = 0.015

# If True, the camera prims are removed when stopping the controller.
REMOVE_CAMERAS_ON_STOP = False

# Console status.
PRINT_STATUS = False
PRINT_INTERVAL_SECONDS = 2.0


# ============================================================
# Controller implementation
# ============================================================

@dataclass
class DualCameraConfig:
    target_prim_path: str = TARGET_PRIM_PATH
    fpv_camera_path: str = FPV_CAMERA_PATH
    observer_camera_path: str = OBSERVER_CAMERA_PATH

    primary_view_camera: str = PRIMARY_VIEW_CAMERA
    auto_assign_viewports: bool = AUTO_ASSIGN_VIEWPORTS
    auto_create_second_viewport: bool = AUTO_CREATE_SECOND_VIEWPORT

    fpv_direction_source: str = FPV_DIRECTION_SOURCE
    fpv_forward_axis: str = FPV_FORWARD_AXIS
    fpv_forward_offset_m: float = FPV_FORWARD_OFFSET_M
    fpv_height_m: float = FPV_HEIGHT_M
    fpv_look_ahead_m: float = FPV_LOOK_AHEAD_M
    fpv_look_down_m: float = FPV_LOOK_DOWN_M
    fpv_focal_length: float = FPV_FOCAL_LENGTH
    fpv_horizontal_aperture: float = FPV_HORIZONTAL_APERTURE

    observer_mode: str = OBSERVER_MODE
    observer_back_distance_m: float = OBSERVER_BACK_DISTANCE_M
    observer_height_m: float = OBSERVER_HEIGHT_M
    observer_side_offset_m: float = OBSERVER_SIDE_OFFSET_M
    observer_look_ahead_m: float = OBSERVER_LOOK_AHEAD_M
    observer_look_at_height_m: float = OBSERVER_LOOK_AT_HEIGHT_M
    observer_top_height_m: float = OBSERVER_TOP_HEIGHT_M
    observer_top_look_at_height_m: float = OBSERVER_TOP_LOOK_AT_HEIGHT_M
    observer_focal_length: float = OBSERVER_FOCAL_LENGTH
    observer_horizontal_aperture: float = OBSERVER_HORIZONTAL_APERTURE

    clipping_range: tuple = CLIPPING_RANGE
    smoothing: float = SMOOTHING
    min_move_distance_m: float = MIN_MOVE_DISTANCE_M
    remove_cameras_on_stop: bool = REMOVE_CAMERAS_ON_STOP
    print_status: bool = PRINT_STATUS
    print_interval_seconds: float = PRINT_INTERVAL_SECONDS


class DualUAVCameraController:
    def __init__(self, config):
        self.config = config
        self.stage = omni.usd.get_context().get_stage()

        self.subscription = None
        self.fpv_transform_op = None
        self.observer_transform_op = None

        self.fpv_camera_pos = None
        self.observer_camera_pos = None
        self.last_target_pos = None
        self.forward_dir = Gf.Vec3d(1.0, 0.0, 0.0)
        self.last_print_time = 0.0
        self.last_camera_update_wall = 0.0
        self.camera_update_interval_s = 0.10
        self.is_running = False

    def start(self):
        if self.stage is None:
            raise RuntimeError("No active USD stage found.")

        self._resolve_selected_target()
        self._create_cameras()
        self.assign_viewports()

        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="DualUAVCameraControllerUpdate",
        )

        self.is_running = True
        print("[DualUAVCameraController] Started.")
        print(f"[DualUAVCameraController] Target: {self.config.target_prim_path}")
        print(f"[DualUAVCameraController] FPV camera: {self.config.fpv_camera_path}")
        print(f"[DualUAVCameraController] Observer camera: {self.config.observer_camera_path}")
        print("[DualUAVCameraController] Stop with: builtins.stop_dual_uav_camera_controller()")
        print("[DualUAVCameraController] If View 2 is not assigned automatically, open a second viewport and select /World/UAV_Camera_Observer.")

    def stop(self):
        if self.subscription is not None:
            try:
                self.subscription.unsubscribe()
                print("[DualUAVCameraController] Subscription stopped.")
            except Exception as error:
                print(f"[DualUAVCameraController] Failed to stop subscription: {error}")

        self.subscription = None
        self.is_running = False

        if get_active_viewport is not None:
            viewport = get_active_viewport()
            if viewport is not None:
                try:
                    viewport.camera_path = "/OmniverseKit_Persp"
                    print("[DualUAVCameraController] Active viewport switched to Perspective.")
                except Exception as error:
                    print(f"[DualUAVCameraController] Failed to switch active viewport: {error}")

        if self.config.remove_cameras_on_stop:
            self._remove_camera(self.config.fpv_camera_path)
            self._remove_camera(self.config.observer_camera_path)

    def set_target(self, prim_path):
        prim_path = str(prim_path).strip()

        if not prim_path:
            print("[DualUAVCameraController] Empty prim path ignored.")
            return

        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"[DualUAVCameraController] Target prim does not exist: {prim_path}")
            return

        self.config.target_prim_path = prim_path
        self.fpv_camera_pos = None
        self.observer_camera_pos = None
        self.last_target_pos = None
        print(f"[DualUAVCameraController] Target switched to: {prim_path}")

    def set_primary_view_camera(self, name):
        name = str(name).upper().strip()
        if name not in ["FPV", "OBSERVER"]:
            print("[DualUAVCameraController] Valid primary cameras: FPV, OBSERVER")
            return
        self.config.primary_view_camera = name
        self.assign_viewports()
        print(f"[DualUAVCameraController] Primary view camera set to: {name}")

    def set_fpv_view(
        self,
        forward_offset=None,
        height=None,
        look_ahead=None,
        look_down=None,
        focal_length=None,
        horizontal_aperture=None,
        direction_source=None,
        forward_axis=None,
    ):
        if forward_offset is not None:
            self.config.fpv_forward_offset_m = float(forward_offset)
        if height is not None:
            self.config.fpv_height_m = float(height)
        if look_ahead is not None:
            self.config.fpv_look_ahead_m = float(look_ahead)
        if look_down is not None:
            self.config.fpv_look_down_m = float(look_down)
        if focal_length is not None:
            self.config.fpv_focal_length = float(focal_length)
            self._set_camera_lens(self.config.fpv_camera_path, focal_length=self.config.fpv_focal_length)
        if horizontal_aperture is not None:
            self.config.fpv_horizontal_aperture = float(horizontal_aperture)
            self._set_camera_lens(self.config.fpv_camera_path, horizontal_aperture=self.config.fpv_horizontal_aperture)
        if direction_source is not None:
            direction_source = str(direction_source).upper().strip()
            if direction_source not in ["MOTION", "BODY_AXIS"]:
                print("[DualUAVCameraController] FPV direction_source must be MOTION or BODY_AXIS.")
            else:
                self.config.fpv_direction_source = direction_source
        if forward_axis is not None:
            forward_axis = str(forward_axis).upper().strip()
            if forward_axis not in ["X", "-X", "Y", "-Y", "Z", "-Z"]:
                print("[DualUAVCameraController] forward_axis must be X, -X, Y, -Y, Z, or -Z.")
            else:
                self.config.fpv_forward_axis = forward_axis

        self.fpv_camera_pos = None
        print("[DualUAVCameraController] FPV view updated.")
        self._print_fpv_settings()

    def set_fpv_axis(self, axis):
        self.set_fpv_view(direction_source="BODY_AXIS", forward_axis=axis)

    def set_fpv_motion_direction(self):
        self.config.fpv_direction_source = "MOTION"
        self.fpv_camera_pos = None
        print("[DualUAVCameraController] FPV direction source set to MOTION.")

    def set_fpv_wide_angle(self, level="WIDE"):
        level = str(level).upper().strip()

        if level in ["ULTRA", "ULTRAWIDE", "ULTRA_WIDE"]:
            self.set_fpv_view(
                forward_offset=0.40,
                height=0.10,
                look_ahead=3.0,
                look_down=-3.0,
                focal_length=8.0,
                horizontal_aperture=32.0,
            )
            print("[DualUAVCameraController] FPV lens preset: ULTRA WIDE")
            return

        if level in ["NORMAL", "STANDARD"]:
            self.set_fpv_view(
                forward_offset=0.55,
                height=0.16,
                look_ahead=4.0,
                look_down=-2.2,
                focal_length=20.0,
                horizontal_aperture=22.0,
            )
            print("[DualUAVCameraController] FPV lens preset: NORMAL")
            return

        self.set_fpv_view(
            forward_offset=0.45,
            height=0.12,
            look_ahead=3.5,
            look_down=-2.4,
            focal_length=12.0,
            horizontal_aperture=28.0,
        )
        print("[DualUAVCameraController] FPV lens preset: WIDE")

    def set_observer_view(
        self,
        mode=None,
        distance=None,
        height=None,
        side_offset=None,
        look_ahead=None,
        look_height=None,
        top_height=None,
        focal_length=None,
        horizontal_aperture=None,
    ):
        if mode is not None:
            self.set_observer_mode(mode, reset_position=False)
        if distance is not None:
            self.config.observer_back_distance_m = float(distance)
        if height is not None:
            self.config.observer_height_m = float(height)
        if side_offset is not None:
            self.config.observer_side_offset_m = float(side_offset)
        if look_ahead is not None:
            self.config.observer_look_ahead_m = float(look_ahead)
        if look_height is not None:
            self.config.observer_look_at_height_m = float(look_height)
        if top_height is not None:
            self.config.observer_top_height_m = float(top_height)
        if focal_length is not None:
            self.config.observer_focal_length = float(focal_length)
            self._set_camera_lens(self.config.observer_camera_path, focal_length=self.config.observer_focal_length)
        if horizontal_aperture is not None:
            self.config.observer_horizontal_aperture = float(horizontal_aperture)
            self._set_camera_lens(self.config.observer_camera_path, horizontal_aperture=self.config.observer_horizontal_aperture)

        self.observer_camera_pos = None
        print("[DualUAVCameraController] Observer view updated.")
        self._print_observer_settings()

    def set_observer_mode(self, mode, reset_position=True):
        mode = str(mode).upper().strip()
        if mode not in ["CHASE", "TOP"]:
            print("[DualUAVCameraController] Observer mode must be CHASE or TOP.")
            return
        self.config.observer_mode = mode
        if reset_position:
            self.observer_camera_pos = None
        print(f"[DualUAVCameraController] Observer mode switched to: {mode}")

    def assign_viewports(self):
        if not self.config.auto_assign_viewports:
            return

        primary_camera = self.config.fpv_camera_path
        secondary_camera = self.config.observer_camera_path

        if self.config.primary_view_camera.upper() == "OBSERVER":
            primary_camera = self.config.observer_camera_path
            secondary_camera = self.config.fpv_camera_path

        active_assigned = False
        if get_active_viewport is not None:
            try:
                active_viewport = get_active_viewport()
                if active_viewport is not None:
                    active_viewport.camera_path = primary_camera
                    active_assigned = True
                    print(f"[DualUAVCameraController] Active viewport assigned to: {primary_camera}")
            except Exception as error:
                print(f"[DualUAVCameraController] Failed to assign active viewport: {error}")

        windows = []
        if get_viewport_window_instances is not None:
            try:
                windows = list(get_viewport_window_instances())
            except Exception as error:
                print(f"[DualUAVCameraController] Could not list viewport windows: {error}")

        if self.config.auto_create_second_viewport and create_viewport_window is not None and len(windows) < 2:
            try:
                create_viewport_window("UAV Observer View", width=640, height=360)
                if get_viewport_window_instances is not None:
                    windows = list(get_viewport_window_instances())
                print("[DualUAVCameraController] Created a second viewport window.")
            except Exception as error:
                print(f"[DualUAVCameraController] Could not create second viewport window: {error}")

        assigned_secondary = False
        if len(windows) >= 2:
            for window in windows:
                if assigned_secondary:
                    break

                viewport_api = getattr(window, "viewport_api", None)
                if viewport_api is None:
                    continue

                try:
                    current_camera = str(getattr(viewport_api, "camera_path", ""))
                except Exception:
                    current_camera = ""

                # Avoid immediately overwriting the active viewport if it already uses the primary camera.
                if active_assigned and current_camera == primary_camera:
                    continue

                try:
                    viewport_api.camera_path = secondary_camera
                    assigned_secondary = True
                    print(f"[DualUAVCameraController] Secondary viewport assigned to: {secondary_camera}")
                except Exception as error:
                    print(f"[DualUAVCameraController] Failed to assign secondary viewport: {error}")

        if not assigned_secondary:
            print("[DualUAVCameraController] Second viewport was not assigned automatically.")
            print(f"[DualUAVCameraController] Manually set View 2 camera to: {secondary_camera}")

    def _resolve_selected_target(self):
        if not USE_SELECTED_PRIM_AS_TARGET:
            return

        selection = omni.usd.get_context().get_selection().get_selected_prim_paths()

        if not selection:
            print("[DualUAVCameraController] No selected prim. Using configured target.")
            return

        selected_path = selection[0]
        selected_prim = self.stage.GetPrimAtPath(selected_path)

        if selected_prim.IsValid():
            self.config.target_prim_path = selected_path
            print(f"[DualUAVCameraController] Selected prim used as target: {selected_path}")
        else:
            print("[DualUAVCameraController] Selected prim is invalid. Using configured target.")

    def _create_cameras(self):
        self.fpv_transform_op = self._create_camera(
            self.config.fpv_camera_path,
            self.config.fpv_focal_length,
            self.config.fpv_horizontal_aperture,
        )
        self.observer_transform_op = self._create_camera(
            self.config.observer_camera_path,
            self.config.observer_focal_length,
            self.config.observer_horizontal_aperture,
        )

    def _create_camera(self, camera_path, focal_length, horizontal_aperture):
        existing_camera = self.stage.GetPrimAtPath(camera_path)

        if existing_camera.IsValid():
            self.stage.RemovePrim(camera_path)
            print(f"[DualUAVCameraController] Removed old camera: {camera_path}")

        camera = UsdGeom.Camera.Define(self.stage, camera_path)
        camera.GetFocalLengthAttr().Set(float(focal_length))
        camera.GetHorizontalApertureAttr().Set(float(horizontal_aperture))
        camera.GetClippingRangeAttr().Set(self.config.clipping_range)

        camera_prim = camera.GetPrim()
        camera_xform = UsdGeom.Xformable(camera_prim)
        return camera_xform.AddTransformOp()

    def _remove_camera(self, camera_path):
        camera_prim = self.stage.GetPrimAtPath(camera_path)

        if camera_prim.IsValid():
            self.stage.RemovePrim(camera_path)
            print(f"[DualUAVCameraController] Camera removed: {camera_path}")

    def _set_camera_lens(self, camera_path, focal_length=None, horizontal_aperture=None):
        prim = self.stage.GetPrimAtPath(camera_path)
        if not prim.IsValid():
            print(f"[DualUAVCameraController] Camera does not exist: {camera_path}")
            return

        camera = UsdGeom.Camera(prim)
        if focal_length is not None:
            camera.GetFocalLengthAttr().Set(float(focal_length))
        if horizontal_aperture is not None:
            camera.GetHorizontalApertureAttr().Set(float(horizontal_aperture))

    def _get_world_matrix(self, prim_path):
        prim = self.stage.GetPrimAtPath(prim_path)

        if not prim.IsValid():
            return None

        try:
            return omni.usd.get_world_transform_matrix(prim)
        except Exception:
            cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            return cache.GetLocalToWorldTransform(prim)

    def _get_world_position(self, prim_path):
        matrix = self._get_world_matrix(prim_path)
        if matrix is None:
            return None
        return matrix.ExtractTranslation()

    def _normalize(self, vector, fallback=None):
        length = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])

        if length < 1e-6:
            if fallback is not None:
                return fallback
            return Gf.Vec3d(1.0, 0.0, 0.0)

        return Gf.Vec3d(vector[0] / length, vector[1] / length, vector[2] / length)

    def _normalize_xy(self, vector, fallback=None):
        length = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])

        if length < 1e-6:
            if fallback is not None:
                return fallback
            return Gf.Vec3d(1.0, 0.0, 0.0)

        return Gf.Vec3d(vector[0] / length, vector[1] / length, 0.0)

    def _lerp_vec3(self, current, target, alpha):
        return Gf.Vec3d(
            current[0] * (1.0 - alpha) + target[0] * alpha,
            current[1] * (1.0 - alpha) + target[1] * alpha,
            current[2] * (1.0 - alpha) + target[2] * alpha,
        )

    def _set_camera_look_at(self, camera_path, transform_op, eye, target, up):
        camera_prim = self.stage.GetPrimAtPath(camera_path)

        if not camera_prim.IsValid():
            print(f"[DualUAVCameraController] Camera prim is invalid: {camera_path}")
            self.stop()
            return False

        if transform_op is None:
            print(f"[DualUAVCameraController] Camera transform op is missing: {camera_path}")
            self.stop()
            return False

        diff = Gf.Vec3d(target[0] - eye[0], target[1] - eye[1], target[2] - eye[2])
        if math.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) < 1e-5:
            target = Gf.Vec3d(eye[0] + 1.0, eye[1], eye[2] - 0.2)

        try:
            view_matrix = Gf.Matrix4d().SetLookAt(eye, target, up)
            camera_world_matrix = view_matrix.GetInverse()
            transform_op.Set(camera_world_matrix)
            return True
        except Exception as error:
            print(f"[DualUAVCameraController] Failed to update camera transform for {camera_path}: {error}")
            self.stop()
            return False

    def _get_local_axis_vector(self, axis_name):
        axis_name = str(axis_name).upper().strip()
        if axis_name == "X":
            return Gf.Vec3d(1.0, 0.0, 0.0)
        if axis_name == "-X":
            return Gf.Vec3d(-1.0, 0.0, 0.0)
        if axis_name == "Y":
            return Gf.Vec3d(0.0, 1.0, 0.0)
        if axis_name == "-Y":
            return Gf.Vec3d(0.0, -1.0, 0.0)
        if axis_name == "Z":
            return Gf.Vec3d(0.0, 0.0, 1.0)
        if axis_name == "-Z":
            return Gf.Vec3d(0.0, 0.0, -1.0)
        return Gf.Vec3d(1.0, 0.0, 0.0)

    def _get_body_axis_direction(self, world_matrix):
        local_axis = self._get_local_axis_vector(self.config.fpv_forward_axis)

        try:
            world_axis = world_matrix.TransformDir(local_axis)
        except Exception:
            # Fallback: transform local origin and local axis endpoint.
            try:
                p0 = world_matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
                p1 = world_matrix.Transform(local_axis)
                world_axis = Gf.Vec3d(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
            except Exception:
                return self.forward_dir

        # Flatten to horizontal to avoid violent camera roll/pitch from UAV body tilt.
        return self._normalize_xy(world_axis, self.forward_dir)

    def _update_motion_direction(self, target_pos):
        if self.last_target_pos is None:
            self.last_target_pos = Gf.Vec3d(target_pos[0], target_pos[1], target_pos[2])
            return

        delta = Gf.Vec3d(
            target_pos[0] - self.last_target_pos[0],
            target_pos[1] - self.last_target_pos[1],
            0.0,
        )

        move_distance = math.sqrt(delta[0] * delta[0] + delta[1] * delta[1])

        if move_distance > self.config.min_move_distance_m:
            self.forward_dir = self._normalize_xy(delta, self.forward_dir)

        self.last_target_pos = Gf.Vec3d(target_pos[0], target_pos[1], target_pos[2])

    def _get_forward_direction(self, target_matrix, target_pos):
        self._update_motion_direction(target_pos)

        if self.config.fpv_direction_source.upper() == "BODY_AXIS" and target_matrix is not None:
            return self._get_body_axis_direction(target_matrix)

        return self.forward_dir

    def _get_right_direction(self, forward_dir):
        # Horizontal right vector. Positive/negative side_offset can be used to swap shoulders.
        return self._normalize_xy(Gf.Vec3d(forward_dir[1], -forward_dir[0], 0.0), Gf.Vec3d(0.0, -1.0, 0.0))

    def _compute_fpv_camera(self, target_pos, forward_dir):
        eye = Gf.Vec3d(
            target_pos[0] + forward_dir[0] * self.config.fpv_forward_offset_m,
            target_pos[1] + forward_dir[1] * self.config.fpv_forward_offset_m,
            target_pos[2] + self.config.fpv_height_m,
        )

        target = Gf.Vec3d(
            target_pos[0] + forward_dir[0] * self.config.fpv_look_ahead_m,
            target_pos[1] + forward_dir[1] * self.config.fpv_look_ahead_m,
            target_pos[2] + self.config.fpv_look_down_m,
        )

        up = Gf.Vec3d(0.0, 0.0, 1.0)
        return eye, target, up

    def _compute_observer_chase_camera(self, target_pos, forward_dir):
        right_dir = self._get_right_direction(forward_dir)

        eye = Gf.Vec3d(
            target_pos[0]
            - forward_dir[0] * self.config.observer_back_distance_m
            + right_dir[0] * self.config.observer_side_offset_m,
            target_pos[1]
            - forward_dir[1] * self.config.observer_back_distance_m
            + right_dir[1] * self.config.observer_side_offset_m,
            target_pos[2] + self.config.observer_height_m,
        )

        target = Gf.Vec3d(
            target_pos[0] + forward_dir[0] * self.config.observer_look_ahead_m,
            target_pos[1] + forward_dir[1] * self.config.observer_look_ahead_m,
            target_pos[2] + self.config.observer_look_at_height_m,
        )

        up = Gf.Vec3d(0.0, 0.0, 1.0)
        return eye, target, up

    def _compute_observer_top_camera(self, target_pos):
        eye = Gf.Vec3d(
            target_pos[0],
            target_pos[1],
            target_pos[2] + self.config.observer_top_height_m,
        )

        target = Gf.Vec3d(
            target_pos[0],
            target_pos[1],
            target_pos[2] + self.config.observer_top_look_at_height_m,
        )

        up = Gf.Vec3d(0.0, 1.0, 0.0)
        return eye, target, up

    def _compute_observer_camera(self, target_pos, forward_dir):
        mode = self.config.observer_mode.upper()
        if mode == "TOP":
            return self._compute_observer_top_camera(target_pos)
        return self._compute_observer_chase_camera(target_pos, forward_dir)

    def _print_status_if_needed(self, event, target_pos):
        if not self.config.print_status:
            return

        current_time = getattr(event, "current_time", 0.0)

        if current_time - self.last_print_time < self.config.print_interval_seconds:
            return

        self.last_print_time = current_time
        print(
            "[DualUAVCameraController] "
            f"target_pos=({target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}), "
            f"fpv_source={self.config.fpv_direction_source}, "
            f"observer={self.config.observer_mode}"
        )

    def _print_fpv_settings(self):
        print(
            "[DualUAVCameraController] FPV settings: "
            f"source={self.config.fpv_direction_source}, "
            f"axis={self.config.fpv_forward_axis}, "
            f"forward_offset={self.config.fpv_forward_offset_m}, "
            f"height={self.config.fpv_height_m}, "
            f"look_ahead={self.config.fpv_look_ahead_m}, "
            f"look_down={self.config.fpv_look_down_m}, "
            f"focal_length={self.config.fpv_focal_length}, "
            f"horizontal_aperture={self.config.fpv_horizontal_aperture}"
        )

    def _print_observer_settings(self):
        print(
            "[DualUAVCameraController] Observer settings: "
            f"mode={self.config.observer_mode}, "
            f"distance={self.config.observer_back_distance_m}, "
            f"height={self.config.observer_height_m}, "
            f"side_offset={self.config.observer_side_offset_m}, "
            f"look_ahead={self.config.observer_look_ahead_m}, "
            f"look_height={self.config.observer_look_at_height_m}, "
            f"top_height={self.config.observer_top_height_m}, "
            f"focal_length={self.config.observer_focal_length}, "
            f"horizontal_aperture={self.config.observer_horizontal_aperture}"
        )

    def _on_update(self, event):
        if not self.is_running:
            return


        now = time.time()
        if now - self.last_camera_update_wall < self.camera_update_interval_s:
            return
        self.last_camera_update_wall = now

        target_matrix = self._get_world_matrix(self.config.target_prim_path)
        if target_matrix is None:
            return

        target_pos = target_matrix.ExtractTranslation()
        forward_dir = self._get_forward_direction(target_matrix, target_pos)

        fpv_eye, fpv_target, fpv_up = self._compute_fpv_camera(target_pos, forward_dir)
        observer_eye, observer_target, observer_up = self._compute_observer_camera(target_pos, forward_dir)

        if self.fpv_camera_pos is None:
            self.fpv_camera_pos = fpv_eye
        else:
            self.fpv_camera_pos = self._lerp_vec3(self.fpv_camera_pos, fpv_eye, self.config.smoothing)

        if self.observer_camera_pos is None:
            self.observer_camera_pos = observer_eye
        else:
            self.observer_camera_pos = self._lerp_vec3(self.observer_camera_pos, observer_eye, self.config.smoothing)

        fpv_ok = self._set_camera_look_at(
            self.config.fpv_camera_path,
            self.fpv_transform_op,
            self.fpv_camera_pos,
            fpv_target,
            fpv_up,
        )

        observer_ok = self._set_camera_look_at(
            self.config.observer_camera_path,
            self.observer_transform_op,
            self.observer_camera_pos,
            observer_target,
            observer_up,
        )

        if fpv_ok and observer_ok:
            self._print_status_if_needed(event, target_pos)


# ============================================================
# Public helper functions stored in builtins
# ============================================================

def stop_existing_camera_controllers():
    old_controller_names = [
        "_dual_uav_camera_controller",
        "_uav_camera_controller",
    ]

    for name in old_controller_names:
        old_controller = getattr(builtins, name, None)
        if old_controller is None:
            continue

        try:
            old_controller.stop()
            print(f"[DualUAVCameraController] Stopped old controller: {name}")
        except Exception as error:
            print(f"[DualUAVCameraController] Failed to stop old controller {name}: {error}")

        try:
            delattr(builtins, name)
        except Exception:
            pass

    old_subscription_names = [
        "_uav_follow_camera_subscription",
        "_follow_camera_subscription",
        "_uav_chase_camera_subscription",
        "_uav_camera_subscription",
        "_uav_top_camera_subscription",
        "_dual_uav_camera_subscription",
    ]

    for name in old_subscription_names:
        old_sub = getattr(builtins, name, None)

        if old_sub is None:
            continue

        try:
            old_sub.unsubscribe()
            print(f"[DualUAVCameraController] Stopped old subscription: {name}")
        except Exception as error:
            print(f"[DualUAVCameraController] Failed to stop {name}: {error}")

        try:
            delattr(builtins, name)
        except Exception:
            pass


def stop_dual_uav_camera_controller():
    controller = getattr(builtins, "_dual_uav_camera_controller", None)

    if controller is None:
        print("[DualUAVCameraController] No active controller.")
        return

    controller.stop()

    try:
        delattr(builtins, "_dual_uav_camera_controller")
    except Exception:
        pass

    print("[DualUAVCameraController] Controller stopped.")


def set_dual_uav_camera_target(prim_path):
    controller = getattr(builtins, "_dual_uav_camera_controller", None)

    if controller is None:
        print("[DualUAVCameraController] No active controller. Run the script first.")
        return

    controller.set_target(prim_path)


def set_uav_fpv_view(
    forward_offset=None,
    height=None,
    look_ahead=None,
    look_down=None,
    focal_length=None,
    horizontal_aperture=None,
    direction_source=None,
    forward_axis=None,
):
    controller = getattr(builtins, "_dual_uav_camera_controller", None)

    if controller is None:
        print("[DualUAVCameraController] No active controller. Run the script first.")
        return

    controller.set_fpv_view(
        forward_offset=forward_offset,
        height=height,
        look_ahead=look_ahead,
        look_down=look_down,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        direction_source=direction_source,
        forward_axis=forward_axis,
    )


def set_uav_fpv_axis(axis):
    controller = getattr(builtins, "_dual_uav_camera_controller", None)

    if controller is None:
        print("[DualUAVCameraController] No active controller. Run the script first.")
        return

    controller.set_fpv_axis(axis)


def set_uav_fpv_motion_direction():
    controller = getattr(builtins, "_dual_uav_camera_controller", None)

    if controller is None:
        print("[DualUAVCameraController] No active controller. Run the script first.")
        return

    controller.set_fpv_motion_direction()


def set_uav_fpv_wide_angle(level="WIDE"):
    controller = getattr(builtins, "_dual_uav_camera_controller", None)

    if controller is None:
        print("[DualUAVCameraController] No active controller. Run the script first.")
        return

    controller.set_fpv_wide_angle(level)


def set_uav_observer_view(
    mode=None,
    distance=None,
    height=None,
    side_offset=None,
    look_ahead=None,
    look_height=None,
    top_height=None,
    focal_length=None,
    horizontal_aperture=None,
):
    controller = getattr(builtins, "_dual_uav_camera_controller", None)

    if controller is None:
        print("[DualUAVCameraController] No active controller. Run the script first.")
        return

    controller.set_observer_view(
        mode=mode,
        distance=distance,
        height=height,
        side_offset=side_offset,
        look_ahead=look_ahead,
        look_height=look_height,
        top_height=top_height,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
    )


def set_uav_observer_mode(mode):
    controller = getattr(builtins, "_dual_uav_camera_controller", None)

    if controller is None:
        print("[DualUAVCameraController] No active controller. Run the script first.")
        return

    controller.set_observer_mode(mode)


def set_dual_uav_primary_view(camera_name):
    controller = getattr(builtins, "_dual_uav_camera_controller", None)

    if controller is None:
        print("[DualUAVCameraController] No active controller. Run the script first.")
        return

    controller.set_primary_view_camera(camera_name)


def assign_dual_uav_viewports():
    controller = getattr(builtins, "_dual_uav_camera_controller", None)

    if controller is None:
        print("[DualUAVCameraController] No active controller. Run the script first.")
        return

    controller.assign_viewports()


def print_dual_uav_camera_help():
    print("")
    print("Dual UAV Camera Controller commands:")
    print("  builtins.set_uav_fpv_view(forward_offset=0.45, height=0.12, look_ahead=3.5, look_down=-2.4, focal_length=12.0, horizontal_aperture=28.0)")
    print("  builtins.set_uav_fpv_wide_angle('WIDE')")
    print("  builtins.set_uav_fpv_wide_angle('ULTRA')")
    print("  builtins.set_uav_fpv_wide_angle('NORMAL')")
    print("  builtins.set_uav_fpv_axis('X')       # Try 'X', '-X', 'Y', '-Y' if BODY_AXIS direction is needed")
    print("  builtins.set_uav_fpv_motion_direction()")
    print("  builtins.set_uav_observer_mode('TOP')")
    print("  builtins.set_uav_observer_mode('CHASE')")
    print("  builtins.set_uav_observer_view(distance=3.2, height=5.2, side_offset=2.2, look_ahead=2.5, look_height=-1.2, focal_length=18.0)")
    print("  builtins.set_dual_uav_primary_view('FPV')")
    print("  builtins.set_dual_uav_primary_view('OBSERVER')")
    print("  builtins.assign_dual_uav_viewports()")
    print("  builtins.set_dual_uav_camera_target('/World/quadrotor/body')")
    print("  builtins.stop_dual_uav_camera_controller()")
    print("")
    print("Camera prims:")
    print(f"  FPV      : {FPV_CAMERA_PATH}")
    print(f"  Observer : {OBSERVER_CAMERA_PATH}")
    print("")


builtins.stop_dual_uav_camera_controller = stop_dual_uav_camera_controller
builtins.set_dual_uav_camera_target = set_dual_uav_camera_target
builtins.set_uav_fpv_view = set_uav_fpv_view
builtins.set_uav_fpv_axis = set_uav_fpv_axis
builtins.set_uav_fpv_motion_direction = set_uav_fpv_motion_direction
builtins.set_uav_fpv_wide_angle = set_uav_fpv_wide_angle
builtins.set_uav_observer_view = set_uav_observer_view
builtins.set_uav_observer_mode = set_uav_observer_mode
builtins.set_dual_uav_primary_view = set_dual_uav_primary_view
builtins.assign_dual_uav_viewports = assign_dual_uav_viewports
builtins.print_dual_uav_camera_help = print_dual_uav_camera_help


# ============================================================
# Start controller
# ============================================================

stop_existing_camera_controllers()

config = DualCameraConfig()
controller = DualUAVCameraController(config)
builtins._dual_uav_camera_controller = controller
controller.start()
print_dual_uav_camera_help()
