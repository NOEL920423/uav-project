"""Strict dataset contract validation and descriptive statistics."""

from pathlib import Path

import numpy as np

from uav_ml.contracts import ACTION_CONTRACT_VERSION, DEFAULT_CONTRACT
from uav_ml.datasets.dataset import (
    REQUIRED_ARRAYS,
    discover_episodes,
    load_episode,
    load_metadata,
)


def _episode_id(episode: dict[str, np.ndarray], path: Path) -> str:
    if "episode_id" not in episode:
        raise ValueError(f"{path} has no episode_id")
    value = episode["episode_id"]
    if value.shape != ():
        raise ValueError(f"{path} episode_id must be scalar")
    result = str(value.item())
    if not result:
        raise ValueError(f"{path} episode_id is empty")
    return result


def validate_dataset(dataset_path: str | Path) -> dict:
    """Validate every episode and return useful aggregate statistics."""
    root = Path(dataset_path)
    metadata = load_metadata(root)
    if metadata.get("action_contract_version") != ACTION_CONTRACT_VERSION:
        raise ValueError("dataset action contract version is incorrect")
    if metadata.get("action_frame") != DEFAULT_CONTRACT.action_frame:
        raise ValueError("dataset action frame must be px4_ned")

    split_ids: dict[str, set[str]] = {"train": set(), "validation": set()}
    actions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    split_counts: dict[str, int] = {}
    for split in ("train", "validation"):
        paths = discover_episodes(root, split)
        if not paths:
            raise ValueError(f"dataset split {split} has no episodes")
        split_counts[split] = 0
        for path in paths:
            episode = load_episode(path)
            episode_id = _episode_id(episode, path)
            required_metadata = {
                "start_ned": (3,),
                "goal_ned": (3,),
                "obstacles_north_east_radius": None,
                "scene_seed": (),
                "planner_path_source": (),
            }
            for name, expected_shape in required_metadata.items():
                if name not in episode:
                    raise ValueError(f"{path} has no episode metadata {name}")
                if expected_shape is not None and episode[name].shape != expected_shape:
                    raise ValueError(
                        f"{path} metadata {name} shape is {episode[name].shape}"
                    )
            if (
                episode["obstacles_north_east_radius"].ndim != 2
                or episode["obstacles_north_east_radius"].shape[1] != 3
            ):
                raise ValueError(f"{path} obstacle metadata must have shape [N, 3]")
            for name in ("start_ned", "goal_ned", "obstacles_north_east_radius"):
                if not np.isfinite(episode[name]).all():
                    raise ValueError(f"{path} metadata {name} is non-finite")
            if episode_id in split_ids[split]:
                raise ValueError(f"duplicate episode ID: {episode_id}")
            split_ids[split].add(episode_id)
            count = int(episode["expert_action"].shape[0])
            if count <= 0:
                raise ValueError(f"{path} contains no synchronized samples")
            for name in REQUIRED_ARRAYS:
                if int(episode[name].shape[0]) != count:
                    raise ValueError(
                        f"{path} has unsynchronized length for {name}"
                    )
            expected_depth = (
                count,
                1,
                DEFAULT_CONTRACT.depth_height,
                DEFAULT_CONTRACT.depth_width,
            )
            expected_shapes = {
                "depth": expected_depth,
                "velocity": (count, 3),
                "goal_direction": (count, 3),
                "expert_action": (count, 4),
                "step": (count,),
                "timestamp_s": (count,),
                "goal_distance_m": (count,),
            }
            for name, expected in expected_shapes.items():
                if episode[name].shape != expected:
                    raise ValueError(
                        f"{path} {name} shape {episode[name].shape} != {expected}"
                    )
                if not np.isfinite(episode[name]).all():
                    raise ValueError(f"{path} {name} contains NaN or Inf")
            if episode["depth"].dtype != np.float32:
                raise ValueError(f"{path} depth dtype must be float32")
            if np.any(episode["depth"] < DEFAULT_CONTRACT.depth_min_m):
                raise ValueError(f"{path} depth is below contract range")
            if np.any(episode["depth"] > DEFAULT_CONTRACT.depth_max_m):
                raise ValueError(f"{path} depth is above contract range")
            if not np.array_equal(
                episode["step"], np.arange(count, dtype=episode["step"].dtype)
            ):
                raise ValueError(f"{path} step indices are not contiguous")
            if np.any(np.diff(episode["timestamp_s"]) <= 0.0):
                raise ValueError(f"{path} timestamps are not increasing")
            split_counts[split] += count
            actions.append(episode["expert_action"])
            velocities.append(episode["velocity"])
            depths.append(episode["depth"])
            distances.append(episode["goal_distance_m"])

    overlap = split_ids["train"] & split_ids["validation"]
    if overlap:
        raise ValueError(f"train/validation episode overlap: {sorted(overlap)}")
    action = np.concatenate(actions)
    velocity = np.concatenate(velocities)
    depth = np.concatenate(depths)
    distance = np.concatenate(distances)
    near_fraction = float(np.mean(np.min(depth, axis=(1, 2, 3)) < 1.0))
    return {
        "episodes": sum(len(ids) for ids in split_ids.values()),
        "train_episodes": len(split_ids["train"]),
        "validation_episodes": len(split_ids["validation"]),
        "samples": int(action.shape[0]),
        "train_samples": split_counts["train"],
        "validation_samples": split_counts["validation"],
        "action_mean": action.mean(axis=0).tolist(),
        "action_std": action.std(axis=0).tolist(),
        "velocity_min": velocity.min(axis=0).tolist(),
        "velocity_max": velocity.max(axis=0).tolist(),
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "depth_mean": float(depth.mean()),
        "near_obstacle_fraction": near_fraction,
        "goal_distance_min": float(distance.min()),
        "goal_distance_max": float(distance.max()),
        "goal_distance_mean": float(distance.mean()),
    }
