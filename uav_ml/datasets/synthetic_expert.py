"""Clearly labeled analytic fixture for validating the BC software path."""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

from uav_ml.contracts import (
    ACTION_CONTRACT_VERSION,
    DATASET_VERSION,
    DEFAULT_CONTRACT,
    OBSERVATION_CONTRACT_VERSION,
)
from uav_ml.datasets.split import split_episode_ids
from uav_ml.navigation_imports import add_navigation_source_path


def _goal_body_direction(
    position: np.ndarray, goal: np.ndarray, yaw_ned: float
) -> np.ndarray:
    delta = goal - position
    delta /= max(float(np.linalg.norm(delta)), 1e-6)
    cosine, sine = math.cos(yaw_ned), math.sin(yaw_ned)
    return np.asarray(
        [
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
            delta[2],
        ],
        dtype=np.float32,
    )


def _analytic_depth(
    position: np.ndarray,
    yaw_ned: float,
    obstacles: np.ndarray,
) -> np.ndarray:
    """Make a deterministic 64x64 pinhole-like depth fixture in metres."""
    height, width = DEFAULT_CONTRACT.depth_height, DEFAULT_CONTRACT.depth_width
    image = np.full((height, width), DEFAULT_CONTRACT.depth_max_m, np.float32)
    cosine, sine = math.cos(yaw_ned), math.sin(yaw_ned)
    for north, east, radius in obstacles:
        delta_n, delta_e = north - position[0], east - position[1]
        forward = cosine * delta_n + sine * delta_e
        right = -sine * delta_n + cosine * delta_e
        if forward <= 0.05:
            continue
        angle = math.atan2(right, forward)
        if abs(angle) > math.radians(45.0):
            continue
        center_x = int(round((angle / math.radians(45.0) + 1.0) * 0.5 * (width - 1)))
        distance = max(DEFAULT_CONTRACT.depth_min_m, math.hypot(forward, right) - radius)
        radius_px = max(1, int(round(width * math.atan2(radius, forward) / math.radians(90.0))))
        center_y = height // 2
        y_radius = min(height // 2, max(3, radius_px * 3))
        left, right_px = max(0, center_x - radius_px), min(width, center_x + radius_px + 1)
        top, bottom = max(0, center_y - y_radius), min(height, center_y + y_radius + 1)
        image[top:bottom, left:right_px] = np.minimum(
            image[top:bottom, left:right_px], distance
        )
    return image[None]


def _episode(seed: int, maximum_steps: int) -> dict[str, np.ndarray]:
    add_navigation_source_path()
    from uav_navigation.astar_planner import plan_path
    from uav_navigation.models import (
        BSplineConfig,
        CircularObstacle,
        PlannerConfig,
        Point3D,
    )
    from uav_navigation.offline_kinematic_plant import (
        KinematicPlantConfig,
        OfflineKinematicPlant,
    )
    from uav_navigation.tracking_models import TrackingConfig
    from uav_navigation.trajectory_parameterizer import parameterize_trajectory
    from uav_navigation.trajectory_sampler import sample_trajectory
    from uav_navigation.trajectory_tracker import compute_tracking_command

    generator = np.random.default_rng(seed)
    start = np.asarray([0.0, 0.0, -2.0], dtype=np.float32)
    goal = np.asarray(
        [6.0 + generator.uniform(-0.4, 0.4), generator.uniform(-1.0, 1.0), -2.0],
        dtype=np.float32,
    )
    obstacle_array = np.asarray(
        [
            [2.0, generator.uniform(-0.35, 0.35), 0.35],
            [4.0, generator.uniform(-1.1, 1.1), 0.32],
        ],
        dtype=np.float32,
    )
    obstacles = tuple(
        CircularObstacle(
            f"fixture_{index}",
            Point3D(float(north), float(east), -2.0),
            float(radius),
            4.0,
        )
        for index, (north, east, radius) in enumerate(obstacle_array)
    )
    planner_config = PlannerConfig(
        grid_resolution_m=0.10,
        planning_bounds=(-1.5, 7.5, -3.5, 3.5),
        maximum_waypoint_spacing_m=0.8,
    )
    result = plan_path(
        Point3D(*map(float, start)),
        Point3D(*map(float, goal)),
        obstacles,
        planner_config,
        BSplineConfig(bspline_sample_spacing_m=0.16, bspline_minimum_samples=12),
    )
    if not result.success:
        raise RuntimeError(f"synthetic expert A* failed: {result.status}")
    trajectory = parameterize_trajectory(result.final_path)
    if not trajectory.valid:
        raise RuntimeError(
            f"synthetic expert trajectory failed: {trajectory.rejection_reason}"
        )
    dt = max(0.08, trajectory.total_duration_s / max(maximum_steps - 1, 1))
    plant = OfflineKinematicPlant(
        KinematicPlantConfig(
            integration_timestep_s=dt,
            initial_position=Point3D(*map(float, start)),
            initial_yaw_ned=float(trajectory.trajectory_points[0].yaw_ned),
        )
    )
    tracking_config = TrackingConfig(control_period_s=dt)
    previous = None
    rows: dict[str, list] = {
        "depth": [],
        "velocity": [],
        "goal_direction": [],
        "expert_action": [],
        "step": [],
        "timestamp_s": [],
        "goal_distance_m": [],
    }
    sample_count = min(
        maximum_steps,
        max(8, math.ceil(trajectory.total_duration_s / dt) + 1),
    )
    for step in range(sample_count):
        timestamp = min(step * dt, trajectory.total_duration_s)
        measurement = plant.measurement(timestamp)
        reference = sample_trajectory(trajectory.trajectory_points, timestamp)
        _, command, _ = compute_tracking_command(
            reference,
            measurement,
            timestamp,
            tracking_config,
            previous,
        )
        position = np.asarray(
            [measurement.position.x, measurement.position.y, measurement.position.z],
            dtype=np.float32,
        )
        velocity = np.asarray(
            [measurement.velocity.x, measurement.velocity.y, measurement.velocity.z],
            dtype=np.float32,
        )
        action = np.asarray(
            [command.linear.x, command.linear.y, command.linear.z, command.yaw_rate_radps],
            dtype=np.float32,
        )
        rows["depth"].append(
            _analytic_depth(position, measurement.yaw_ned, obstacle_array)
        )
        rows["velocity"].append(velocity)
        rows["goal_direction"].append(
            _goal_body_direction(position, goal, measurement.yaw_ned)
        )
        rows["expert_action"].append(action)
        rows["step"].append(step)
        rows["timestamp_s"].append(timestamp)
        rows["goal_distance_m"].append(float(np.linalg.norm(goal - position)))
        previous = command
        plant.step(command)
    arrays = {
        name: np.asarray(values, dtype=(np.int64 if name == "step" else np.float32))
        for name, values in rows.items()
    }
    arrays.update(
        {
            "start_ned": start,
            "goal_ned": goal,
            "obstacles_north_east_radius": obstacle_array,
            "scene_seed": np.asarray(seed, dtype=np.int64),
            "planner_path_source": np.asarray(result.final_path_source),
        }
    )
    return arrays


def generate_synthetic_dataset(
    output: str | Path,
    episodes: int = 8,
    maximum_steps: int = 32,
    seed: int = 17,
    validation_fraction: float = 0.2,
    overwrite: bool = False,
) -> dict:
    """Generate a bounded software fixture; this is not Isaac Sim evidence."""
    if not 2 <= episodes <= 100:
        raise ValueError("episodes must be in [2, 100]")
    if not 2 <= maximum_steps <= 1000:
        raise ValueError("maximum_steps must be in [2, 1000]")
    root = Path(output)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"dataset path already exists: {root}")
        shutil.rmtree(root)
    episode_ids = [f"synthetic_{index:05d}" for index in range(episodes)]
    train_ids, validation_ids = split_episode_ids(
        episode_ids, validation_fraction, seed
    )
    split_by_id = {episode_id: "train" for episode_id in train_ids}
    split_by_id.update({episode_id: "validation" for episode_id in validation_ids})
    for split in ("train", "validation"):
        (root / split).mkdir(parents=True, exist_ok=True)
    sample_count = 0
    for index, episode_id in enumerate(episode_ids):
        arrays = _episode(seed + index, maximum_steps)
        arrays["episode_id"] = np.asarray(episode_id)
        split = split_by_id[episode_id]
        np.savez_compressed(root / split / f"episode_{episode_id}.npz", **arrays)
        sample_count += int(arrays["expert_action"].shape[0])
    metadata = {
        "dataset_version": DATASET_VERSION,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "fixture_kind": "synthetic_analytic_depth_software_validation_only",
        "real_uav_learning_evidence": False,
        "action_frame": DEFAULT_CONTRACT.action_frame,
        "observation_frame": DEFAULT_CONTRACT.observation_frame,
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "episode_limit": episodes,
        "maximum_steps_per_episode": maximum_steps,
        "sample_stride": 1,
        "synchronization_rule": (
            "capture observation from logical step t, compute bounded expert "
            "action from the same state/reference t, save both, then advance"
        ),
        "episodes": episodes,
        "samples": sample_count,
        "estimated_uncompressed_bytes": (
            sample_count * DEFAULT_CONTRACT.depth_height
            * DEFAULT_CONTRACT.depth_width * 4
        ),
        "contract": DEFAULT_CONTRACT.to_dict(),
    }
    with (root / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/bc_v0")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = generate_synthetic_dataset(
        args.output,
        args.episodes,
        args.max_steps,
        args.seed,
        args.validation_fraction,
        args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

