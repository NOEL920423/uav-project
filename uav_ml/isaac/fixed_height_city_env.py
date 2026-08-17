"""Isaac-rendered Gym environment backed by fixed-height deterministic dynamics."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

from uav_ml.envs import FixedHeightCityConfig, FixedHeightCityCore


class IsaacFixedHeightCityEnv(gym.Env):
    """One clean FPV render product with synchronous reset/step semantics."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        device: str = "cuda:0",
        config: FixedHeightCityConfig | None = None,
    ) -> None:
        super().__init__()
        self.core = FixedHeightCityCore(config)
        self.device = device
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(
                    0, 255, shape=(72, 128, 3), dtype=np.uint8
                ),
                "state": gym.spaces.Box(
                    -np.inf, np.inf, shape=(8,), dtype=np.float32
                ),
            }
        )
        sim_cfg = sim_utils.SimulationCfg(
            dt=self.core.config.control_dt_s,
            render_interval=1,
            device=device,
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self._setup_static_scene()
        self.camera = self._create_camera()
        self.sim.reset()
        self._warm_camera()

    def _setup_static_scene(self) -> None:
        ground = sim_utils.GroundPlaneCfg(
            color=(0.18, 0.32, 0.42),
            size=(30.0, 30.0),
        )
        ground.func("/World/Ground", ground)
        dome = sim_utils.DomeLightCfg(
            intensity=1800.0,
            color=(0.82, 0.88, 1.0),
        )
        dome.func("/World/Light", dome)

    def _create_camera(self) -> Camera:
        cfg = CameraCfg(
            prim_path="/World/UAV/FPVCamera",
            update_period=0.0,
            # RTX/DLSS in Isaac 5.1 requires an internal dimension near 300.
            # Render at 16:9 320x180, then use the same bilinear downsampling as
            # the Autoencoder input contract.
            height=180,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=20.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 30.0),
            ),
        )
        return Camera(cfg=cfg)

    def _spawn_city(self) -> None:
        stage = self.sim.stage
        old = stage.GetPrimAtPath("/World/GeneratedCity")
        if old and old.IsValid():
            stage.RemovePrim(old.GetPath())
        sim_utils.create_prim("/World/GeneratedCity", "Xform")
        for index, building in enumerate(self.core.buildings):
            cfg = sim_utils.CuboidCfg(
                size=(building.width, building.depth, building.height),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=building.color,
                    roughness=0.72,
                    metallic=0.08,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            )
            cfg.func(
                f"/World/GeneratedCity/Building_{index:03d}",
                cfg,
                translation=(building.x, building.y, 0.5 * building.height),
            )

    def _set_camera_pose(self) -> None:
        x, y = map(float, self.core.position)
        yaw = float(self.core.yaw)
        camera_position = torch.tensor(
            [[x, y, self.core.config.altitude_m]],
            dtype=torch.float32,
            device=self.device,
        )
        look_distance = 4.0
        target = torch.tensor(
            [
                [
                    x + math.sin(yaw) * look_distance,
                    y + math.cos(yaw) * look_distance,
                    self.core.config.altitude_m - 0.18,
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.camera.set_world_poses_from_view(camera_position, target)

    def _warm_camera(self) -> None:
        self._set_camera_pose()
        for _ in range(1):
            self.sim.step(render=True)
            self.camera.update(self.core.config.control_dt_s)

    def _rgb(self) -> np.ndarray:
        rgb = self.camera.data.output.get("rgb")
        if rgb is None or rgb.numel() == 0:
            raise RuntimeError("Isaac FPV camera has no RGB output")
        array = rgb[0, ..., :3].detach().cpu().numpy()
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="RGB").resize(
            (128, 72), resample=Image.Resampling.BILINEAR
        )
        return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))

    def _observation(self, state: dict) -> dict[str, np.ndarray]:
        return {"rgb": self._rgb(), "state": state["state"]}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        selected_seed = 0 if seed is None else int(seed)
        state = self.core.reset(selected_seed)
        self._spawn_city()
        self._warm_camera()
        return self._observation(state), {
            "seed": selected_seed,
            "building_count": len(self.core.buildings),
            "city_generation_attempt": self.core.city_generation_attempt,
            "debug_markers_visible": False,
            "goal_distance_m": float(state["goal_distance_m"]),
            "position_xy": state["position_xy"].copy(),
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        state, reward, terminated, truncated, info = self.core.step(action)
        self._set_camera_pose()
        self.sim.step(render=True)
        self.camera.update(self.core.config.control_dt_s)
        info = dict(info)
        info.update(
            {
                "seed": self.core.seed,
                "step_index": self.core.step_index,
                "sim_time_s": self.core.step_index * self.core.config.control_dt_s,
                "synchronized": True,
                "debug_markers_visible": False,
                "position_xy": state["position_xy"].copy(),
            }
        )
        return self._observation(state), reward, terminated, truncated, info

    def expert_action(self) -> np.ndarray:
        return self.core.expert_action()

    def close(self) -> None:
        # SimulationApp owns Camera/SimulationContext shutdown. Explicit sensor
        # reset after a render-product episode can deadlock Isaac Sim 5.1.
        return None
