#!/usr/bin/env python3
"""Bring up the complete Pegasus/PX4/ROS 2 UAV simulation automatically.

This file is intended for Isaac Sim's ``--exec`` option.  It reproduces the
manual Pegasus UI sequence used by this project:

1. load the default environment;
2. spawn an Iris with the PX4 MAVLink backend;
3. start the physics timeline (which auto-launches PX4 SITL);
4. start the narrow bootstrap pose/scene/runtime bridge.

The episode itself is still started by the external ROS 2 orchestrator, so
booting Isaac Sim never arms the vehicle by itself.
"""

from __future__ import annotations

import asyncio
import builtins
import runpy
import traceback
from pathlib import Path

import omni.kit.app
import omni.usd

from pegasus.simulator.logic.backends import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS, WORLD_SETTINGS

from pxr import Gf, UsdGeom, UsdPhysics


SCRIPT_ROOT = Path(__file__).resolve().parent
RUNTIME_BRIDGE_SCRIPT = SCRIPT_ROOT / "runtime_bridge.py"
PX4_ROOT = Path.home() / "PX4-Autopilot"
VEHICLE_PRIM_PATH = "/World/quadrotor"
BOOTSTRAP_SCENE_ROOT = "/World/BootstrapScene"


def create_bootstrap_scene(stage) -> None:
    """Create the simple bootstrap scene with a floor and one obstacle."""
    root = UsdGeom.Xform.Define(stage, BOOTSTRAP_SCENE_ROOT)
    root.GetPrim().SetCustomDataByKey(
        "bootstrap:scene_id",
        "bootstrap_simple_scene_v1",
    )

    # ------------------------------------------------------------
    # Plain visual floor
    # ------------------------------------------------------------
    floor = UsdGeom.Cube.Define(
        stage,
        f"{BOOTSTRAP_SCENE_ROOT}/PlainFloor",
    )
    floor.CreateSizeAttr(1.0)

    # Final physical dimensions:
    # X = 20 m
    # Y = 20 m
    # Z = 0.01 m
    floor.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.01)
    )
    floor.AddScaleOp().Set(
        Gf.Vec3d(20.0, 20.0, 0.01)
    )

    floor.CreateDisplayColorAttr(
        [Gf.Vec3f(0.40, 0.40, 0.40)]
    )

    # No CollisionAPI here.
    # Keep the original Pegasus GroundPlane responsible for physics.

    # ------------------------------------------------------------
    # Obstacle
    # ------------------------------------------------------------
    obstacle = UsdGeom.Cylinder.Define(
        stage,
        f"{BOOTSTRAP_SCENE_ROOT}/Obstacle_001",
    )
    obstacle.CreateRadiusAttr(0.35)
    obstacle.CreateHeightAttr(3.0)
    obstacle.AddTranslateOp().Set(
        Gf.Vec3d(-1.5, 1.5, 1.5)
    )

    obstacle.CreateDisplayColorAttr(
        [Gf.Vec3f(0.30, 0.30, 0.30)]
    )
    UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())

    # ------------------------------------------------------------
    # Goal marker
    # ------------------------------------------------------------
    goal = UsdGeom.Cylinder.Define(
        stage,
        f"{BOOTSTRAP_SCENE_ROOT}/Goal",
    )
    goal.CreateRadiusAttr(0.25)
    goal.CreateHeightAttr(0.02)
    goal.AddTranslateOp().Set(
        Gf.Vec3d(0.5, 3.0, 0.01)
    )
    goal.CreateDisplayColorAttr(
        [Gf.Vec3f(0.20, 0.85, 0.25)]
    )

    print("[UAVBootstrap]  Scene created.")
    print("[UAVBootstrap] Plain floor size: 20 x 20 x 0.01 m.")


async def wait_for_updates(count: int) -> None:
    app = omni.kit.app.get_app()
    for _ in range(max(0, int(count))):
        await app.next_update_async()


async def bootstrap() -> None:
    try:
        print("[UAVBootstrap] Starting automatic Isaac/Pegasus/PX4 setup.")
        await wait_for_updates(10)

        pegasus = PegasusInterface()
        pegasus.set_world_settings(**WORLD_SETTINGS["px4"])
        await pegasus.load_environment_async(
            SIMULATION_ENVIRONMENTS["Default Environment"],
            force_clear=True,
        )
        await wait_for_updates(30)

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim has no active USD stage after environment loading.")

        existing_vehicle = stage.GetPrimAtPath(VEHICLE_PRIM_PATH)
        create_bootstrap_scene(stage)
        if not existing_vehicle or not existing_vehicle.IsValid():
            backend_config = PX4MavlinkBackendConfig({
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": str(PX4_ROOT),
                "px4_vehicle_model": "gazebo-classic_iris",
                "enable_lockstep": True,
            })
            multirotor_config = MultirotorConfig()
            multirotor_config.backends = [PX4MavlinkBackend(config=backend_config)]
            Multirotor(
                VEHICLE_PRIM_PATH,
                ROBOTS["Iris"],
                0,
                [0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
                config=multirotor_config,
            )
            print("[UAVBootstrap] Iris spawned with PX4 auto-launch enabled.")
        else:
            print("[UAVBootstrap] Existing Iris retained.")

        await wait_for_updates(30)
        await pegasus.world.play_async()
        await wait_for_updates(60)
        print("[UAVBootstrap] Physics timeline running; PX4 launch requested.")

        if not RUNTIME_BRIDGE_SCRIPT.is_file():
            raise FileNotFoundError(
                f"Runtime bridge not found: {RUNTIME_BRIDGE_SCRIPT}"
            )
        runpy.run_path(
            str(RUNTIME_BRIDGE_SCRIPT),
            run_name="__isaac_runtime_bridge__",
        )
        print("[UAVBootstrap] Isaac ROS 2 runtime bridge started.")
    except Exception as exc:
        builtins._isaac_uav_bootstrap_error = f"{type(exc).__name__}: {exc}"
        print(f"[UAVBootstrap][ERROR] {builtins._isaac_uav_bootstrap_error}")
        traceback.print_exc()


old_task = getattr(builtins, "_isaac_uav_bootstrap_task", None)
if old_task is not None and not old_task.done():
    old_task.cancel()

builtins._isaac_uav_bootstrap_error = ""
builtins._isaac_uav_bootstrap_task = asyncio.ensure_future(bootstrap())
