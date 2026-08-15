from pathlib import Path
import traceback
import time

try:
    import omni.kit.app
except Exception:
    omni = None


# ============================================================
# User settings
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "isaac_direct_pipeline"

SCENE_SCRIPT = SCRIPT_DIR / "2.scene_episode_generator.py"
DUAL_CAMERA_SCRIPT = SCRIPT_DIR / "1.dual_uav_camera.py"
RECORDER_SCRIPT = SCRIPT_DIR / "3.dual_camera_png_recorder.py"
ASTAR_SCRIPT = SCRIPT_DIR / "4.px4_astar.py"

RUN_SCENE_GENERATOR = True
RUN_DUAL_CAMERA = True
INSTALL_FRONT_CAMERA_RECORDER = False

# Safety switch:
# False = only prepare scene, camera, recorder.
# True  = also start A* + PX4 mission.
RUN_ASTAR_MISSION = True

TICKS_AFTER_EACH_SCRIPT = 2


# ============================================================
# Helpers
# ============================================================

def tick_isaac_app(count=1):
    try:
        app = omni.kit.app.get_app()
        for _ in range(int(count)):
            app.update()
    except Exception:
        pass


def run_script(script_path, label, run_as_main):
    script_path = Path(script_path).expanduser()

    if not script_path.exists():
        print(f"[UAVLaunch][ERROR] Missing script: {script_path}")
        return False

    print("")
    print("=" * 80)
    print(f"[UAVLaunch] Running {label}")
    print(f"[UAVLaunch] File: {script_path}")
    print("=" * 80)

    namespace = {
        "__file__": str(script_path),
        "__name__": "__main__" if run_as_main else f"uav_launch_{label}",
    }

    try:
        code = script_path.read_text(encoding="utf-8")
        exec(compile(code, str(script_path), "exec"), namespace)
        tick_isaac_app(TICKS_AFTER_EACH_SCRIPT)
        print(f"[UAVLaunch] Done: {label}")
        return True

    except Exception as exc:
        print(f"[UAVLaunch][ERROR] Failed: {label}")
        print(f"[UAVLaunch][ERROR] Reason: {exc}")
        traceback.print_exc()
        return False


# ============================================================
# Main launch flow
# ============================================================

def main():
    print("")
    print("############################################################")
    print("[UAVLaunch] UAV pipeline launch started.")
    print("############################################################")

    if RUN_SCENE_GENERATOR:
        ok = run_script(
            SCENE_SCRIPT,
            label="scene_episode_generator",
            run_as_main=True,
        )
        if not ok:
            print("[UAVLaunch] Stop: scene generation failed.")
            return

    if RUN_DUAL_CAMERA:
        ok = run_script(
            DUAL_CAMERA_SCRIPT,
            label="dual_uav_camera",
            run_as_main=False,
        )
        if not ok:
            print("[UAVLaunch] Stop: dual camera setup failed.")
            return

    if INSTALL_FRONT_CAMERA_RECORDER:
        ok = run_script(
            RECORDER_SCRIPT,
            label="front_camera_png_recorder",
            run_as_main=False,
        )
        if not ok:
            print("[UAVLaunch] Stop: recorder setup failed.")
            return

    if RUN_ASTAR_MISSION:
        print("")
        print("[UAVLaunch] RUN_ASTAR_MISSION=True")
        print("[UAVLaunch] A* + PX4 mission will start now.")
        print("[UAVLaunch] Make sure Pegasus/PX4 is already running.")
        ok = run_script(
            ASTAR_SCRIPT,
            label="px4_astar_mission",
            run_as_main=False,
        )
        if not ok:
            print("[UAVLaunch] A* mission failed.")
            return
    else:
        print("")
        print("[UAVLaunch] Preparation complete.")
        print("[UAVLaunch] A* mission was NOT started.")
        print("[UAVLaunch] To start flight mission, set RUN_ASTAR_MISSION=True or run 4.px4_astar.py manually.")

    print("")
    print("############################################################")
    print("[UAVLaunch] UAV pipeline launch finished.")
    print("############################################################")


main()
