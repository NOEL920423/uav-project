"""Deterministic fixed-height city dynamics shared by Isaac and tests."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from uav_ml.navigation_imports import add_navigation_source_path

add_navigation_source_path()

from uav_navigation.astar_planner import plan_path  # noqa: E402
from uav_navigation.models import (  # noqa: E402
    BSplineConfig,
    CircularObstacle,
    PlannerConfig,
    Point3D,
)


@dataclass(frozen=True)
class Building:
    """Axis-aligned high-rise used for rendering and conservative collision."""

    x: float
    y: float
    width: float
    depth: float
    height: float
    color: tuple[float, float, float]

    @property
    def planning_radius(self) -> float:
        return 0.5 * math.hypot(self.width, self.depth)


@dataclass(frozen=True)
class FixedHeightCityConfig:
    """Versioned minimum navigation task configuration."""

    start_xy: tuple[float, float] = (0.0, 0.0)
    goal_xy: tuple[float, float] = (3.0, 5.0)
    altitude_m: float = 2.0
    world_bounds: tuple[float, float, float, float] = (-2.0, 5.0, -2.0, 7.0)
    obstacle_count: int = 8
    building_width_range: tuple[float, float] = (0.46, 0.72)
    building_depth_range: tuple[float, float] = (0.46, 0.72)
    building_height_range: tuple[float, float] = (2.8, 5.2)
    endpoint_clearance_m: float = 1.0
    building_gap_m: float = 0.45
    uav_radius_m: float = 0.18
    control_dt_s: float = 0.1
    velocity_time_constant_s: float = 0.25
    maximum_acceleration_mps2: float = 2.0
    maximum_forward_speed_mps: float = 1.0
    maximum_right_speed_mps: float = 0.8
    maximum_yaw_rate_radps: float = 1.0
    success_distance_m: float = 0.35
    maximum_steps: int = 160
    expert_lookahead_m: float = 0.8
    progress_reward_scale: float = 3.0
    success_reward: float = 20.0
    collision_penalty: float = -20.0
    timeout_penalty: float = -2.0
    time_penalty_per_step: float = -0.01
    action_smoothness_scale: float = -0.02
    maximum_city_generation_attempts: int = 100

    def to_dict(self) -> dict:
        return asdict(self)


class FixedHeightCityCore:
    """Pure reset/step state machine; Isaac supplies only clean RGB rendering."""

    ACTION_DIMENSION = 3
    STATE_DIMENSION = 8

    def __init__(self, config: FixedHeightCityConfig | None = None) -> None:
        self.config = config or FixedHeightCityConfig()
        self.rng = np.random.default_rng(0)
        self.seed = 0
        self.buildings: tuple[Building, ...] = ()
        self.position = np.zeros(2, dtype=np.float32)
        self.velocity_world = np.zeros(2, dtype=np.float32)
        self.yaw = 0.0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.step_index = 0
        self.terminated = False
        self.truncated = False
        self.path_world: tuple[tuple[float, float], ...] = ()
        self.city_generation_attempt = 0

    def reset(self, seed: int) -> dict:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.position = np.asarray(self.config.start_xy, dtype=np.float32)
        self.velocity_world = np.zeros(2, dtype=np.float32)
        direction = np.asarray(self.config.goal_xy) - self.position
        self.yaw = math.atan2(float(direction[0]), float(direction[1]))
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.step_index = 0
        self.terminated = False
        self.truncated = False
        last_planning_error: RuntimeError | None = None
        for attempt in range(1, self.config.maximum_city_generation_attempts + 1):
            self.buildings = self._generate_buildings()
            try:
                self.path_world = self._plan_expert_path()
            except RuntimeError as error:
                last_planning_error = error
                continue
            self.city_generation_attempt = attempt
            break
        else:
            raise RuntimeError(
                f"could not generate a reachable city for seed {self.seed} after "
                f"{self.config.maximum_city_generation_attempts} attempts"
            ) from last_planning_error
        return self.state_observation()

    def _generate_buildings(self) -> tuple[Building, ...]:
        cfg = self.config
        xmin, xmax, ymin, ymax = cfg.world_bounds
        colors = (
            (0.20, 0.30, 0.40),
            (0.30, 0.25, 0.24),
            (0.26, 0.31, 0.34),
            (0.34, 0.34, 0.30),
        )
        buildings: list[Building] = []
        attempts = 0
        while len(buildings) < cfg.obstacle_count and attempts < 10_000:
            attempts += 1
            width = float(self.rng.uniform(*cfg.building_width_range))
            depth = float(self.rng.uniform(*cfg.building_depth_range))
            height = float(self.rng.uniform(*cfg.building_height_range))
            x = float(self.rng.uniform(xmin + width, xmax - width))
            y = float(self.rng.uniform(ymin + depth, ymax - depth))
            radius = 0.5 * math.hypot(width, depth)
            if any(
                math.hypot(x - endpoint[0], y - endpoint[1])
                < cfg.endpoint_clearance_m + radius
                for endpoint in (cfg.start_xy, cfg.goal_xy)
            ):
                continue
            if any(
                math.hypot(x - other.x, y - other.y)
                < radius + other.planning_radius + cfg.building_gap_m
                for other in buildings
            ):
                continue
            color = colors[int(self.rng.integers(0, len(colors)))]
            buildings.append(Building(x, y, width, depth, height, color))
        if len(buildings) != cfg.obstacle_count:
            raise RuntimeError("could not generate requested collision-free city")
        return tuple(buildings)

    def _plan_expert_path(self) -> tuple[tuple[float, float], ...]:
        cfg = self.config
        start = Point3D(self.position[1], self.position[0], -cfg.altitude_m)
        goal = Point3D(cfg.goal_xy[1], cfg.goal_xy[0], -cfg.altitude_m)
        obstacles = tuple(
            CircularObstacle(
                name=f"Building_{index:03d}",
                center=Point3D(building.y, building.x, -0.5 * building.height),
                radius=building.planning_radius,
                height=building.height,
            )
            for index, building in enumerate(self.buildings)
        )
        planner_cfg = PlannerConfig(
            flight_altitude_m=cfg.altitude_m,
            planning_bounds=(
                cfg.world_bounds[2],
                cfg.world_bounds[3],
                cfg.world_bounds[0],
                cfg.world_bounds[1],
            ),
            enable_overfly_short_obstacles=False,
        )
        result = plan_path(
            start,
            goal,
            obstacles,
            planner_cfg,
            BSplineConfig(enable_bspline=False),
        )
        if not result.success:
            raise RuntimeError(f"A* expert failed for seed {self.seed}: {result.status}")
        return tuple((point.y, point.x) for point in result.final_path)

    def state_observation(self) -> dict:
        goal_delta = np.asarray(self.config.goal_xy, dtype=np.float32) - self.position
        distance = float(np.linalg.norm(goal_delta))
        goal_unit_world = goal_delta / max(distance, 1e-6)
        velocity_body = self._world_to_body(self.velocity_world)
        goal_body = self._world_to_body(goal_unit_world)
        normalized_distance = min(distance / 10.0, 1.0)
        state = np.concatenate(
            (
                velocity_body,
                goal_body,
                np.asarray([normalized_distance], dtype=np.float32),
                self.previous_action,
            )
        ).astype(np.float32)
        return {
            "state": state,
            "position_xy": self.position.copy(),
            "yaw_rad": np.float32(self.yaw),
            "goal_distance_m": np.float32(distance),
            "step_index": self.step_index,
            "seed": self.seed,
            "city_generation_attempt": self.city_generation_attempt,
        }

    def expert_action(self) -> np.ndarray:
        """Return normalized body velocity/yaw action from privileged A*."""
        target = self._lookahead_target()
        delta = np.asarray(target, dtype=np.float32) - self.position
        distance = float(np.linalg.norm(delta))
        direction_world = delta / max(distance, 1e-6)
        desired_speed = min(self.config.maximum_forward_speed_mps, max(0.25, distance))
        desired_world = direction_world * desired_speed
        desired_body = self._world_to_body(desired_world)
        desired_yaw = math.atan2(float(direction_world[0]), float(direction_world[1]))
        yaw_error = (desired_yaw - self.yaw + math.pi) % (2.0 * math.pi) - math.pi
        action = np.asarray(
            [
                desired_body[0] / self.config.maximum_forward_speed_mps,
                desired_body[1] / self.config.maximum_right_speed_mps,
                yaw_error / self.config.maximum_yaw_rate_radps,
            ],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0)

    def _lookahead_target(self) -> tuple[float, float]:
        if not self.path_world:
            return self.config.goal_xy
        points = np.asarray(self.path_world, dtype=np.float32)
        nearest = int(np.argmin(np.linalg.norm(points - self.position, axis=1)))
        accumulated = 0.0
        previous = self.position
        for point in points[nearest:]:
            accumulated += float(np.linalg.norm(point - previous))
            if accumulated >= self.config.expert_lookahead_m:
                return float(point[0]), float(point[1])
            previous = point
        return self.config.goal_xy

    def step(self, normalized_action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        if self.terminated or self.truncated:
            raise RuntimeError("reset is required after episode termination")
        action = np.asarray(normalized_action, dtype=np.float32)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("action must be finite with shape [3]")
        action = np.clip(action, -1.0, 1.0)
        cfg = self.config
        old_distance = float(np.linalg.norm(np.asarray(cfg.goal_xy) - self.position))
        desired_body = np.asarray(
            [
                action[0] * cfg.maximum_forward_speed_mps,
                action[1] * cfg.maximum_right_speed_mps,
            ],
            dtype=np.float32,
        )
        desired_world = self._body_to_world(desired_body)
        acceleration = (desired_world - self.velocity_world) / cfg.velocity_time_constant_s
        norm = float(np.linalg.norm(acceleration))
        if norm > cfg.maximum_acceleration_mps2:
            acceleration *= cfg.maximum_acceleration_mps2 / norm
        self.velocity_world += acceleration * cfg.control_dt_s
        self.position += self.velocity_world * cfg.control_dt_s
        self.yaw += float(action[2]) * cfg.maximum_yaw_rate_radps * cfg.control_dt_s
        self.yaw = (self.yaw + math.pi) % (2.0 * math.pi) - math.pi
        self.step_index += 1

        new_distance = float(np.linalg.norm(np.asarray(cfg.goal_xy) - self.position))
        collision = self._collision()
        success = new_distance < cfg.success_distance_m
        out_of_bounds = not self._inside_bounds()
        self.terminated = collision or success or out_of_bounds
        self.truncated = self.step_index >= cfg.maximum_steps and not self.terminated
        reward_terms = {
            "progress": cfg.progress_reward_scale * (old_distance - new_distance),
            "time": cfg.time_penalty_per_step,
            "smoothness": cfg.action_smoothness_scale
            * float(np.square(action - self.previous_action).sum()),
            "success": cfg.success_reward if success else 0.0,
            "collision": cfg.collision_penalty if collision or out_of_bounds else 0.0,
            "timeout": cfg.timeout_penalty if self.truncated else 0.0,
        }
        self.previous_action = action.copy()
        observation = self.state_observation()
        info = {
            "success": success,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "timeout": self.truncated,
            "goal_distance_m": new_distance,
            "reward_terms": reward_terms,
        }
        return observation, float(sum(reward_terms.values())), self.terminated, self.truncated, info

    def _collision(self) -> bool:
        return any(
            math.hypot(self.position[0] - building.x, self.position[1] - building.y)
            <= building.planning_radius + self.config.uav_radius_m
            for building in self.buildings
        )

    def _inside_bounds(self) -> bool:
        xmin, xmax, ymin, ymax = self.config.world_bounds
        return xmin <= self.position[0] <= xmax and ymin <= self.position[1] <= ymax

    def _world_to_body(self, vector: np.ndarray) -> np.ndarray:
        forward = math.sin(self.yaw) * vector[0] + math.cos(self.yaw) * vector[1]
        right = math.cos(self.yaw) * vector[0] - math.sin(self.yaw) * vector[1]
        return np.asarray([forward, right], dtype=np.float32)

    def _body_to_world(self, vector: np.ndarray) -> np.ndarray:
        x = math.sin(self.yaw) * vector[0] + math.cos(self.yaw) * vector[1]
        y = math.cos(self.yaw) * vector[0] - math.sin(self.yaw) * vector[1]
        return np.asarray([x, y], dtype=np.float32)
