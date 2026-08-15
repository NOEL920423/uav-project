"""Collect same-step clean Isaac RGB/state/A* action demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="datasets/isaac_city_bc_v0")
parser.add_argument("--train-episodes", type=int, default=24)
parser.add_argument("--validation-episodes", type=int, default=6)
parser.add_argument("--test-episodes", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402

from uav_ml.isaac.fixed_height_city_env import IsaacFixedHeightCityEnv  # noqa: E402


DATASET_VERSION = "isaac_city_rgb_bc_v0.1"
OBSERVATION_VERSION = "rgb72x128_state8_v0.1"
ACTION_VERSION = "body_velocity_yaw_normalized_v0.1"


def _collect_episode(env: IsaacFixedHeightCityEnv, seed: int, output: Path) -> dict:
    observation, reset_info = env.reset(seed=seed)
    rgb_frames = []
    states = []
    expert_actions = []
    rewards = []
    terminated_flags = []
    truncated_flags = []
    distances = []
    final_info = {}
    while True:
        # observation_t and action_t are captured before the exact same step.
        action = env.expert_action()
        rgb_frames.append(observation["rgb"].copy())
        states.append(observation["state"].copy())
        expert_actions.append(action.copy())
        observation, reward, terminated, truncated, final_info = env.step(action)
        rewards.append(reward)
        terminated_flags.append(terminated)
        truncated_flags.append(truncated)
        distances.append(final_info["goal_distance_m"])
        if terminated or truncated:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    buildings = np.asarray(
        [
            [item.x, item.y, item.width, item.depth, item.height]
            for item in env.core.buildings
        ],
        dtype=np.float32,
    )
    np.savez_compressed(
        output,
        rgb=np.asarray(rgb_frames, dtype=np.uint8),
        state=np.asarray(states, dtype=np.float32),
        expert_action=np.asarray(expert_actions, dtype=np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=np.bool_),
        truncated=np.asarray(truncated_flags, dtype=np.bool_),
        goal_distance_m=np.asarray(distances, dtype=np.float32),
        buildings=buildings,
        seed=np.asarray(seed, dtype=np.int64),
    )
    return {
        "seed": seed,
        "steps": len(rgb_frames),
        "success": bool(final_info.get("success")),
        "collision": bool(final_info.get("collision")),
        "timeout": bool(final_info.get("timeout")),
        "synchronized": bool(final_info.get("synchronized")),
        "debug_markers_visible": bool(reset_info["debug_markers_visible"]),
        "path": str(output.resolve()),
    }


def main() -> None:
    counts = {
        "train": args.train_episodes,
        "validation": args.validation_episodes,
        "test": args.test_episodes,
    }
    if any(value < 1 for value in counts.values()):
        raise ValueError("every split requires at least one episode")
    seed_bases = {"train": 0, "validation": 10_000, "test": 20_000}
    output_root = Path(args.output)
    env = IsaacFixedHeightCityEnv(device=args.device)
    episodes = []
    for split, count in counts.items():
        for offset in range(count):
            seed = seed_bases[split] + offset
            episode_path = output_root / split / f"episode_seed_{seed:06d}.npz"
            result = _collect_episode(env, seed, episode_path)
            result["split"] = split
            episodes.append(result)
            print(
                f"split={split} seed={seed} steps={result['steps']} "
                f"success={result['success']} synchronized={result['synchronized']}",
                flush=True,
            )
    metadata = {
        "dataset_version": DATASET_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "environment": "IsaacFixedHeightCityEnv",
        "environment_config": env.core.config.to_dict(),
        "camera": {
            "source": "clean_fpV_rgb",
            "shape": [72, 128, 3],
            "dtype": "uint8",
            "debug_markers_visible": False,
        },
        "synchronization_order": [
            "capture_observation_t",
            "compute_astar_expert_action_t",
            "save_pair_t",
            "apply_action_t",
            "advance_isaac",
        ],
        "split_seed_bases": seed_bases,
        "counts": counts,
        "episodes": episodes,
        "all_success": all(item["success"] for item in episodes),
        "all_synchronized": all(item["synchronized"] for item in episodes),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    env.close()
    if not metadata["all_success"] or not metadata["all_synchronized"]:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
