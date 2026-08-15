# 0.cleanup_uav_isaac_env.py
#
# Safe cleanup script for Noel's Isaac Sim UAV project.
#
# Run inside Isaac Sim:
#   Window -> Script Editor -> paste/run
#
# Purpose:
#   1. Stop old camera / PX4 / keyboard controllers if possible.
#   2. Switch viewport back to Perspective.
#   3. Remove generated episode objects.
#   4. Remove debug flight path / safety envelope.
#   5. Remove UAV camera prims.
#
# This script DOES NOT delete:
#   - /World/quadrotor
#   - /World/groundPlane
#   - /World/physicsScene
#   - Pegasus / PX4 vehicle assets
#
# Recommended workflow:
#   1. Run this cleanup script.
#   2. Run scene generator.
#   3. Run dual camera script.
#   4. Run A* flight script.

import builtins
import sys
import time
import traceback

from pxr import Sdf

try:
    import omni.usd
    import omni.kit.app
except Exception as exc:
    raise RuntimeError(
        "This script must be executed inside Isaac Sim / Omniverse Kit."
    ) from exc


# =============================================================================
# User settings
# =============================================================================

# If the UAV is currently flying, it is safer to land manually first.
# This option tries to stop controllers without forcing a landing command.
REQUEST_LAND_BEFORE_STOP = False

# Main cleanup switches.
STOP_CONTROLLERS = True
SWITCH_VIEWPORTS_TO_PERSPECTIVE = True
DELETE_GENERATED_EPISODE = True
DELETE_CAMERA_PRIMS = True
DELETE_EXTRA_DEBUG_PRIMS = True
REMOVE_BUILTIN_HELPER_FUNCTIONS = True

# Keep the actual UAV and core simulation objects.
PROTECTED_PATH_PREFIXES = [
    "/World/quadrotor",
    "/World/groundPlane",
    "/World/physicsScene",
    "/World/PhysicsScene",
    "/World/defaultGroundPlane",
    "/World/Pegasus",
]

# Generated scene root used by your scene generator and A* scripts.
GENERATED_ROOT_PATHS = [
    "/World/GeneratedEpisode",
]

# Camera prims from your camera scripts.
CAMERA_PRIM_PATHS = [
    "/World/UAV_Camera_FPV",
    "/World/UAV_Camera_Observer",
    "/World/UAV_Camera",
    "/World/UAV_Cameras",
    "/World/FollowCamera",
    "/World/UAV_FollowCamera",
    "/World/UAV_ChaseCamera",
    "/World/UAV_TopCamera",
    "/World/UAV_ObserverCamera",
    "/World/FrontDatasetCamera",
    "/World/GodViewCamera",
]

# Extra debug paths, in case some older scripts created debug objects outside GeneratedEpisode.
EXTRA_DEBUG_PRIM_PATHS = [
    "/World/DebugFlightPath",
    "/World/DebugSafetyEnvelope",
    "/World/FlightPath",
    "/World/ExecutedTrail",
    "/World/WaypointMarkers",
]

# Builtins controller names used by previous camera / control scripts.
BUILTIN_CONTROLLER_NAMES = [
    "_dual_uav_camera_controller",
    "_uav_camera_controller",
    "_front_camera_png_recorder",
    "_px4_webrtc_keyboard_logger_controller",
    "_px4_webrtc_keyboard_controller",
    "_isaac_webrtc_px4_drive_controller",
    "_uav_keyboard_controller",
]

# Builtins subscription names used by older camera scripts.
BUILTIN_SUBSCRIPTION_NAMES = [
    "_uav_follow_camera_subscription",
    "_follow_camera_subscription",
    "_uav_chase_camera_subscription",
    "_uav_camera_subscription",
    "_uav_top_camera_subscription",
    "_dual_uav_camera_subscription",
    "_px4_keyboard_subscription",
    "_px4_update_subscription",
]

# Helper functions installed by dual_uav_camera.py.
BUILTIN_HELPER_FUNCTION_NAMES = [
    "stop_dual_uav_camera_controller",
    "set_dual_uav_camera_target",
    "set_uav_fpv_view",
    "set_uav_fpv_axis",
    "set_uav_fpv_motion_direction",
    "set_uav_fpv_wide_angle",
    "set_uav_observer_view",
    "set_uav_observer_mode",
    "set_dual_uav_primary_view",
    "assign_dual_uav_viewports",
    "print_dual_uav_camera_help",
    "start_front_camera_png_recorder",
    "stop_front_camera_png_recorder",
    "print_front_camera_png_recorder_status",
]

# Global runner names from Script Editor / A* script.
GLOBAL_RUNNER_NAMES = [
    "PX4_EPISODE_RUNNER",
]


# =============================================================================
# Logging helpers
# =============================================================================

def log(message):
    print(f"[UAV Cleanup] {message}")


def warn(message):
    print(f"[UAV Cleanup][Warning] {message}")


def error(message):
    print(f"[UAV Cleanup][Error] {message}")


# =============================================================================
# Basic USD helpers
# =============================================================================

def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage found.")
    return stage


def is_protected_path(path):
    path = str(path)
    for prefix in PROTECTED_PATH_PREFIXES:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def prim_exists(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    return bool(prim and prim.IsValid())


def delete_prim_if_exists(stage, prim_path):
    prim_path = str(prim_path)

    if is_protected_path(prim_path):
        warn(f"Protected path skipped: {prim_path}")
        return False

    if not prim_exists(stage, prim_path):
        return False

    try:
        stage.RemovePrim(Sdf.Path(prim_path))
        log(f"Deleted prim: {prim_path}")
        return True
    except Exception as exc:
        error(f"Failed to delete prim {prim_path}: {exc}")
        return False


# =============================================================================
# Controller cleanup
# =============================================================================

def safe_call_method(obj, method_name, *args, **kwargs):
    if obj is None:
        return False

    method = getattr(obj, method_name, None)
    if method is None:
        return False

    try:
        method(*args, **kwargs)
        log(f"Called {type(obj).__name__}.{method_name}()")
        return True
    except TypeError:
        try:
            method()
            log(f"Called {type(obj).__name__}.{method_name}()")
            return True
        except Exception as exc:
            warn(f"Failed to call {method_name}(): {exc}")
            return False
    except Exception as exc:
        warn(f"Failed to call {method_name}(): {exc}")
        return False


def stop_object_gracefully(obj, name):
    if obj is None:
        return False

    log(f"Stopping object: {name}")

    stopped = False

    # PX4 A* runner style.
    if hasattr(obj, "request_stop"):
        try:
            obj.request_stop(send_land=REQUEST_LAND_BEFORE_STOP)
            log(f"request_stop(send_land={REQUEST_LAND_BEFORE_STOP}) sent to {name}")
            stopped = True
        except TypeError:
            stopped = safe_call_method(obj, "request_stop") or stopped
        except Exception as exc:
            warn(f"request_stop failed for {name}: {exc}")

    # WebRTC keyboard controller style.
    if hasattr(obj, "stop"):
        try:
            obj.stop(land_vehicle=REQUEST_LAND_BEFORE_STOP)
            log(f"stop(land_vehicle={REQUEST_LAND_BEFORE_STOP}) sent to {name}")
            stopped = True
        except TypeError:
            stopped = safe_call_method(obj, "stop") or stopped
        except Exception as exc:
            warn(f"stop failed for {name}: {exc}")

    # Last resort: send zero velocity if available.
    if hasattr(obj, "send_velocity_setpoint"):
        try:
            obj.send_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
            log(f"Zero velocity setpoint sent to {name}")
            stopped = True
        except Exception:
            pass

    return stopped


def stop_builtin_controllers():
    stopped_count = 0

    # Use official helper first if available.
    helper = getattr(builtins, "stop_dual_uav_camera_controller", None)
    if callable(helper):
        try:
            helper()
            stopped_count += 1
            log("Official dual camera stop helper executed.")
        except Exception as exc:
            warn(f"Official dual camera stop helper failed: {exc}")

    for name in BUILTIN_CONTROLLER_NAMES:
        obj = getattr(builtins, name, None)
        if obj is None:
            continue

        if stop_object_gracefully(obj, f"builtins.{name}"):
            stopped_count += 1

        try:
            delattr(builtins, name)
            log(f"Removed builtins.{name}")
        except Exception:
            pass

    for name in BUILTIN_SUBSCRIPTION_NAMES:
        sub = getattr(builtins, name, None)
        if sub is None:
            continue

        try:
            sub.unsubscribe()
            log(f"Unsubscribed builtins.{name}")
            stopped_count += 1
        except Exception as exc:
            warn(f"Failed to unsubscribe builtins.{name}: {exc}")

        try:
            delattr(builtins, name)
            log(f"Removed builtins.{name}")
        except Exception:
            pass

    if REMOVE_BUILTIN_HELPER_FUNCTIONS:
        for name in BUILTIN_HELPER_FUNCTION_NAMES:
            if hasattr(builtins, name):
                try:
                    delattr(builtins, name)
                    log(f"Removed helper builtins.{name}")
                except Exception:
                    pass

    return stopped_count


def stop_global_runners():
    stopped_count = 0

    # Script Editor globals may live in __main__.
    main_module = sys.modules.get("__main__", None)

    possible_namespaces = []

    if main_module is not None:
        possible_namespaces.append(("__main__", main_module.__dict__))

    # Current script namespace.
    possible_namespaces.append(("globals", globals()))

    for namespace_name, namespace in possible_namespaces:
        for runner_name in GLOBAL_RUNNER_NAMES:
            obj = namespace.get(runner_name, None)
            if obj is None:
                continue

            if stop_object_gracefully(obj, f"{namespace_name}.{runner_name}"):
                stopped_count += 1

            try:
                namespace[runner_name] = None
                log(f"Cleared {namespace_name}.{runner_name}")
            except Exception:
                pass

    return stopped_count


def stop_controllers():
    if not STOP_CONTROLLERS:
        log("Controller cleanup disabled.")
        return

    log("Stopping old controllers and subscriptions...")

    count = 0
    count += stop_builtin_controllers()
    count += stop_global_runners()

    log(f"Controller cleanup finished. Stopped/cleared count: {count}")


# =============================================================================
# Viewport cleanup
# =============================================================================

def switch_viewports_to_perspective():
    if not SWITCH_VIEWPORTS_TO_PERSPECTIVE:
        return

    perspective_path = "/OmniverseKit_Persp"

    try:
        from omni.kit.viewport.utility import get_active_viewport
    except Exception:
        get_active_viewport = None

    try:
        from omni.kit.viewport.utility import get_viewport_window_instances
    except Exception:
        get_viewport_window_instances = None

    # Active viewport.
    if get_active_viewport is not None:
        try:
            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = perspective_path
                log("Active viewport switched to Perspective.")
        except Exception as exc:
            warn(f"Failed to switch active viewport: {exc}")

    # All viewport windows if available.
    if get_viewport_window_instances is not None:
        try:
            windows = list(get_viewport_window_instances())
            for index, window in enumerate(windows):
                viewport_api = getattr(window, "viewport_api", None)
                if viewport_api is None:
                    continue
                try:
                    viewport_api.camera_path = perspective_path
                    log(f"Viewport window {index} switched to Perspective.")
                except Exception as exc:
                    warn(f"Failed to switch viewport window {index}: {exc}")
        except Exception as exc:
            warn(f"Could not list viewport windows: {exc}")


# =============================================================================
# Prim cleanup
# =============================================================================

def delete_generated_episode(stage):
    if not DELETE_GENERATED_EPISODE:
        return 0

    deleted_count = 0
    for path in GENERATED_ROOT_PATHS:
        if delete_prim_if_exists(stage, path):
            deleted_count += 1

    return deleted_count


def delete_camera_prims(stage):
    if not DELETE_CAMERA_PRIMS:
        return 0

    deleted_count = 0
    for path in CAMERA_PRIM_PATHS:
        if delete_prim_if_exists(stage, path):
            deleted_count += 1

    return deleted_count


def delete_extra_debug_prims(stage):
    if not DELETE_EXTRA_DEBUG_PRIMS:
        return 0

    deleted_count = 0
    for path in EXTRA_DEBUG_PRIM_PATHS:
        if delete_prim_if_exists(stage, path):
            deleted_count += 1

    return deleted_count


def update_app_ticks(count=3):
    try:
        app = omni.kit.app.get_app()
        for _ in range(int(count)):
            app.update()
        log(f"Isaac app updated for {count} tick(s).")
    except Exception as exc:
        warn(f"Could not force app update ticks: {exc}")


# =============================================================================
# Main cleanup
# =============================================================================

def cleanup_uav_isaac_environment():
    print("")
    print("=" * 72)
    print("UAV Isaac Sim Environment Cleanup")
    print("=" * 72)

    stage = get_stage()

    if REQUEST_LAND_BEFORE_STOP:
        warn("REQUEST_LAND_BEFORE_STOP=True. This may block briefly while landing.")
    else:
        warn("This cleanup does NOT force landing. Run after landing or while safely stopped.")

    stop_controllers()

    # Important: switch viewport away from camera prims before deleting them.
    switch_viewports_to_perspective()

    deleted_count = 0
    deleted_count += delete_generated_episode(stage)
    deleted_count += delete_extra_debug_prims(stage)
    deleted_count += delete_camera_prims(stage)

    update_app_ticks(3)

    print("-" * 72)
    print(f"Deleted prim group count: {deleted_count}")
    print("Protected objects were preserved:")
    for path in PROTECTED_PATH_PREFIXES:
        print(f"  - {path}")
    print("-" * 72)
    print("Recommended next steps:")
    print("  1. Run 1.dual_uav_camera.py")
    print("  2. Run 2.scene_episode_generator.py")
    print("  3. Run 3.front_camera_png_recorder.py")
    print("  4. Run 4.px4_astar.py")
    print("=" * 72)
    print("")


cleanup_uav_isaac_environment()