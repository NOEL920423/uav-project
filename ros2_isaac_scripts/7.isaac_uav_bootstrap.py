#!/usr/bin/env python3
"""Bring up the complete Pegasus/PX4/ROS 2 UAV simulation automatically.

This file is intended for Isaac Sim's ``--exec`` option.  It reproduces the
manual Pegasus UI sequence used by this project:

1. load the default environment;
2. spawn an Iris with the PX4 MAVLink backend;
3. start the physics timeline (which auto-launches PX4 SITL);
4. start the Isaac ROS 2 episode manager.

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
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS, WORLD_SETTINGS


SCRIPT_ROOT = Path.home() / "uav-project" / "ros2_isaac_scripts"
MANAGER_SCRIPT = SCRIPT_ROOT / "6.isaac_ros2_episode_manager.py"
PX4_ROOT = Path.home() / "PX4-Autopilot"
VEHICLE_PRIM_PATH = "/World/quadrotor"


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
            multirotor_config.graphical_sensors = [
                MonocularCamera("camera", config={"update_rate": 60.0})
            ]
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

        if not MANAGER_SCRIPT.is_file():
            raise FileNotFoundError(f"Episode manager not found: {MANAGER_SCRIPT}")
        runpy.run_path(str(MANAGER_SCRIPT), run_name="__isaac_uav_episode_manager__")
        print("[UAVBootstrap] Isaac ROS 2 episode manager started.")
    except Exception as exc:
        builtins._isaac_uav_bootstrap_error = f"{type(exc).__name__}: {exc}"
        print(f"[UAVBootstrap][ERROR] {builtins._isaac_uav_bootstrap_error}")
        traceback.print_exc()


old_task = getattr(builtins, "_isaac_uav_bootstrap_task", None)
if old_task is not None and not old_task.done():
    old_task.cancel()

builtins._isaac_uav_bootstrap_error = ""
builtins._isaac_uav_bootstrap_task = asyncio.ensure_future(bootstrap())

