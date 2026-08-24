"""Validate the formal canonical cylinder expert dataset collection."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

import numpy as np

from isaac.runtime.episode_scene import (
    CYLINDER_HEIGHT_MAX,
    CYLINDER_HEIGHT_MIN,
    OBSTACLE_YAW_MAX_DEG,
    OBSTACLE_YAW_MIN_DEG,
    DIRECT_PATH_BLOCKER_COUNT,
    LIGHTING_CONTRACT,
    MIN_OBSTACLE_GAP,
    NUM_OBSTACLES,
    RADIUS_BASIS_DEPTH_MAX,
    RADIUS_BASIS_DEPTH_MIN,
    RADIUS_BASIS_WIDTH_MAX,
    RADIUS_BASIS_WIDTH_MIN,
    START_POS,
    TARGET_POS,
)
from isaac.runtime.formal_expert_sensor_contract import (
    FORMAL_RGB_EXPECTED_RATE_RANGE_HZ,
    LEGACY_OBSERVER_RGB_EXPECTED_RATE_RANGE_HZ,
)
from uav_ml.tools.validate_expert_batch import (
    _uses_formal_top_rgb,
    _validate_auxiliary,
)
from uav_ml.tools.validate_expert_dataset import (
    _directory_size,
    validate_episode,
)


DEFAULT_DATASET = Path("artifacts/datasets/bc_expert_cylinder_v1")
DEFAULT_AUTOENCODER = Path(
    "autoencoder_runs/rgb_ae_v0_baseline_20260811/best.pt"
)
COLLECTION_MANIFEST = "collection_manifest.json"
FINAL_EPISODE_STATES = {"complete", "failed"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _positive_finite(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def validate_cylinder_scene(
    scene: dict, episode_id: str, seed: int
) -> dict:
    """Enforce the frozen scene without regenerating or changing it."""
    if scene.get("episode_id") != episode_id:
        raise ValueError(f"{episode_id}: scene identity mismatch")
    if scene.get("random_seed") != seed:
        raise ValueError(f"{episode_id}: scene seed mismatch")
    if scene.get("generator") != "canonical_cylinder_scene_generator_v1":
        raise ValueError(f"{episode_id}: wrong scene generator")
    if scene.get("mode") != "normal":
        raise ValueError(
            f"{episode_id}: formal collection requires normal mode"
        )
    if scene.get("start") != list(START_POS):
        raise ValueError(f"{episode_id}: canonical start changed")
    if scene.get("target_marker") != list(TARGET_POS):
        raise ValueError(f"{episode_id}: canonical goal changed")
    obstacles = scene.get("obstacles")
    if (
        not isinstance(obstacles, list)
        or len(obstacles) != NUM_OBSTACLES
        or scene.get("obstacle_count") != NUM_OBSTACLES
        or scene.get("normal_obstacle_count") != NUM_OBSTACLES
    ):
        raise ValueError(
            f"{episode_id}: expected exactly {NUM_OBSTACLES} obstacles"
        )
    blockers = 0
    for index, obstacle in enumerate(obstacles, start=1):
        if obstacle.get("shape") != "cylinder":
            raise ValueError(
                f"{episode_id}: obstacle {index} is not a cylinder"
            )
        for name, lower, upper in (
            (
                "radius_basis_width",
                RADIUS_BASIS_WIDTH_MIN,
                RADIUS_BASIS_WIDTH_MAX,
            ),
            (
                "radius_basis_depth",
                RADIUS_BASIS_DEPTH_MIN,
                RADIUS_BASIS_DEPTH_MAX,
            ),
            ("height", CYLINDER_HEIGHT_MIN, CYLINDER_HEIGHT_MAX),
            ("yaw_deg", OBSTACLE_YAW_MIN_DEG, OBSTACLE_YAW_MAX_DEG),
        ):
            try:
                value = float(obstacle[name])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{episode_id}: obstacle {index} has invalid {name}"
                ) from error
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(
                    f"{episode_id}: obstacle {index} {name} is out of range"
                )
        blockers += (
            obstacle.get("placement_mode")
            == "guaranteed_direct_path_blocker"
        )
    if blockers != DIRECT_PATH_BLOCKER_COUNT or scene.get(
        "direct_path_blocker_count"
    ) != DIRECT_PATH_BLOCKER_COUNT:
        raise ValueError(f"{episode_id}: direct-path blocker contract changed")
    for left_index, left in enumerate(obstacles):
        for right in obstacles[left_index + 1:]:
            separation = math.hypot(
                float(left["x"]) - float(right["x"]),
                float(left["y"]) - float(right["y"]),
            )
            required = (
                float(left["radius"])
                + float(right["radius"])
                + MIN_OBSTACLE_GAP
            )
            if separation + 1e-6 < required:
                raise ValueError(
                    f"{episode_id}: minimum obstacle gap violated"
                )
    if scene.get("lighting") != LIGHTING_CONTRACT:
        raise ValueError(f"{episode_id}: canonical lighting changed")
    return {
        "obstacle_count": len(obstacles),
        "direct_path_blocker_count": blockers,
        "minimum_gap_m": MIN_OBSTACLE_GAP,
    }


def validate_episode_metadata(episode: dict, validation: dict) -> dict:
    """Validate formal per-episode metadata not covered by the BC validator."""
    episode_id = validation["episode_id"]
    terminal_reason = str(episode.get("terminal_reason", ""))
    if terminal_reason.startswith("invalid_scene:"):
        scene_result = {
            "valid": False,
            "collection_prevented": True,
            "reason": terminal_reason,
        }
    else:
        scene_result = validate_cylinder_scene(
            episode["scene_configuration"],
            episode_id,
            int(episode["random_seed"]),
        )
    if not episode.get("terminal_reason"):
        raise ValueError(f"{episode_id}: terminal reason missing")
    duration = episode.get("flight_duration_s")
    if (
        duration is None
        or not math.isfinite(float(duration))
        or float(duration) < 0
    ):
        raise ValueError(f"{episode_id}: flight duration missing or invalid")
    path = episode.get("astar_path_information")
    if not isinstance(path, dict) or "validated_path" not in path:
        raise ValueError(f"{episode_id}: A* path information missing")
    validated_path = path.get("validated_path")
    if validation["episode_success"] and (
        not isinstance(validated_path, dict)
        or int(validated_path.get("point_count", 0)) < 2
        or not _positive_finite(validated_path.get("path_length_xy_m"))
    ):
        raise ValueError(
            f"{episode_id}: successful flight lacks a valid A* path"
        )
    streams = episode.get("available_sensor_streams")
    if not isinstance(streams, dict):
        raise ValueError(f"{episode_id}: stream statistics missing")
    stream_counts = {}
    stream_rates = {}
    for name in ("fpv_rgb", "fpv_depth", "observer_rgb"):
        details = streams.get(name)
        if (
            not isinstance(details, dict)
            or int(details.get("received", -1)) < 0
        ):
            raise ValueError(f"{episode_id}: {name} count missing")
        stream_counts[name] = int(details["received"])
        rate = float(details.get("observed_rate_hz", 0.0))
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError(f"{episode_id}: {name} rate is invalid")
        stream_rates[name] = rate
    if validation["episode_success"]:
        formal_top_rgb = _uses_formal_top_rgb(episode)
        if formal_top_rgb and int(
            streams["observer_rgb"].get("matched", -1)
        ) != int(validation["sample_count"]):
            raise ValueError(
                f"{episode_id}: formal TOP RGB match count is incomplete"
            )
        observer_rate_range = (
            FORMAL_RGB_EXPECTED_RATE_RANGE_HZ
            if formal_top_rgb
            else LEGACY_OBSERVER_RGB_EXPECTED_RATE_RANGE_HZ
        )
        for name, lower, upper in (
            ("fpv_rgb", *FORMAL_RGB_EXPECTED_RATE_RANGE_HZ),
            ("fpv_depth", 3.0, 7.0),
            ("observer_rgb", *observer_rate_range),
        ):
            if (
                stream_counts[name] < 2
                or not lower <= stream_rates[name] <= upper
            ):
                raise ValueError(
                    f"{episode_id}: {name} stream rate/count is unreasonable"
                )
    return {
        "scene": scene_result,
        "stream_counts": stream_counts,
        "stream_rates_hz": stream_rates,
    }


def validate_collection_episode(
    dataset_root: Path,
    autoencoder_checkpoint: Path,
    episode_id: str,
    device: str = "auto",
    write_result: bool = True,
) -> dict:
    """Rebuild one episode's 72D observations and validate all streams."""
    dataset_root = dataset_root.resolve()
    validation = validate_episode(
        dataset_root,
        autoencoder_checkpoint,
        episode_id=episode_id,
        device=device,
        write_result=write_result,
        require_success=None,
        require_single_manifest=False,
        update_manifest=False,
    )
    episode = _load(dataset_root / episode_id / "episode.json")
    validation["formal_metadata"] = validate_episode_metadata(
        episode, validation
    )
    validation["auxiliary_availability"] = _validate_auxiliary(
        dataset_root, episode_id
    )
    auxiliary = validation["auxiliary_availability"]
    if validation["episode_success"] and (
        int(auxiliary.get("observer_rgb_available", 0)) < 1
        or int(auxiliary.get("fpv_depth_available", 0)) < 1
    ):
        raise ValueError(
            f"{episode_id}: successful episode lacks auxiliary stream joins"
        )
    return validation


def _statistics(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def validate_collection(
    dataset_root: Path,
    autoencoder_checkpoint: Path,
    expected_episodes: int | None = None,
    device: str = "auto",
    write_result: bool = True,
) -> dict:
    """Validate manifests, scenes, episode data, and aggregates."""
    dataset_root = dataset_root.resolve()
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    collection_manifest_path = dataset_root / COLLECTION_MANIFEST
    dataset_manifest = _load(dataset_manifest_path)
    collection_manifest = _load(collection_manifest_path)
    total = int(collection_manifest.get("target_episodes", -1))
    if expected_episodes is not None and total != expected_episodes:
        raise ValueError(
            f"collection target is {total}, expected {expected_episodes}"
        )
    if total <= 0:
        raise ValueError("collection target must be positive")
    entries = collection_manifest.get("episodes")
    if not isinstance(entries, list) or len(entries) != total:
        raise ValueError("collection manifest episode plan is incomplete")
    expected_ids = [
        f"episode_{index:06d}" for index in range(1, total + 1)
    ]
    ids = [entry.get("episode_id") for entry in entries]
    if ids != expected_ids or dataset_manifest.get("episodes") != expected_ids:
        raise ValueError("dataset episode IDs are incomplete or out of order")
    if any(
        entry.get("status") not in FINAL_EPISODE_STATES
        for entry in entries
    ):
        raise ValueError("collection has unfinished episodes")
    seeds = [int(entry["seed"]) for entry in entries]
    if len(set(seeds)) != total:
        raise ValueError("collection seeds are not unique")

    validations = []
    rejection_counts = Counter()
    auxiliary_counts = Counter()
    state_errors: list[float] = []
    action_errors: list[float] = []
    rates: list[float] = []
    scene_keys = set()
    for entry, episode_id, seed in zip(entries, expected_ids, seeds):
        episode = _load(dataset_root / episode_id / "episode.json")
        if int(episode.get("random_seed", -1)) != seed:
            raise ValueError(f"{episode_id}: collection/episode seed mismatch")
        result = validate_collection_episode(
            dataset_root,
            autoencoder_checkpoint,
            episode_id,
            device=device,
            write_result=write_result,
        )
        if bool(entry.get("success")) != bool(result["episode_success"]):
            raise ValueError(f"{episode_id}: collection outcome mismatch")
        scene = episode["scene_configuration"]
        scene_signature = {
            "obstacles": scene.get("obstacles"),
            "lighting": scene.get("lighting"),
            "start": scene.get("start"),
            "goal": scene.get("goal"),
        }
        if str(episode.get("terminal_reason", "")).startswith(
            "invalid_scene:"
        ):
            scene_signature.update({
                "validation_error": scene.get("validation_error"),
                "seed": seed,
            })
        scene_key = json.dumps(
            scene_signature,
            sort_keys=True,
            separators=(",", ":"),
        )
        if scene_key in scene_keys:
            raise ValueError("collection scenes are not unique")
        scene_keys.add(scene_key)
        rejection_counts.update(episode.get("rejections_by_reason", {}))
        auxiliary_counts.update(result["auxiliary_availability"])
        if result["sample_count"] > 1:
            rates.append(float(result["observed_sampling_rate_hz"]))
        with (dataset_root / episode_id / "samples.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            for row in csv.DictReader(stream):
                state_errors.append(float(row["state_image_error_s"]))
                action_errors.append(float(row["expert_action_image_error_s"]))
        validations.append(result)

    success_count = sum(bool(item["episode_success"]) for item in validations)
    sample_count = sum(int(item["sample_count"]) for item in validations)
    if success_count < 1 or sample_count < 1:
        raise ValueError("collection contains no usable successful samples")
    qa_entries = collection_manifest.get("visual_qa", [])
    expected_qa = total // int(
        collection_manifest.get("visual_qa_interval", 20)
    )
    valid_qa = sum(
        entry.get("status") == "complete"
        and (dataset_root / str(entry.get("path", ""))).is_file()
        for entry in qa_entries
    )
    if valid_qa < expected_qa:
        raise ValueError("scheduled visual QA contact sheets are incomplete")

    result = {
        "valid": True,
        "dataset_version": dataset_manifest.get("dataset_version"),
        "episode_count": total,
        "successful_episodes": success_count,
        "failed_episodes": total - success_count,
        "unique_seeds": len(set(seeds)),
        "unique_scenes": len(scene_keys),
        "accepted_samples_total": sample_count,
        "rejection_breakdown": dict(sorted(rejection_counts.items())),
        "effective_sampling_rate_hz": _statistics(rates),
        "state_image_synchronization_error_s": _statistics(state_errors),
        "action_image_synchronization_error_s": _statistics(action_errors),
        "auxiliary_availability": dict(sorted(auxiliary_counts.items())),
        "visual_qa_contact_sheets": valid_qa,
        "total_dataset_size_bytes": _directory_size(dataset_root),
        "bc_contract_rebuilt": {
            "latent_dimension": 64,
            "observation_dimension": 72,
            "target_dimension": 3,
            "accepted_samples": sample_count,
        },
    }
    if write_result:
        output = dataset_root / "collection_validation.json"
        output.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        dataset_manifest.update({
            "status": "complete",
            "episode_count": total,
            "sample_count": sample_count,
            "successful_episodes": success_count,
            "failed_episodes": total - success_count,
            "validation": {"valid": True, "path": output.name},
        })
        dataset_manifest_path.write_text(
            json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    """Validate one episode or the complete formal collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--autoencoder", default=str(DEFAULT_AUTOENCODER))
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.episode:
        result = validate_collection_episode(
            Path(args.dataset),
            Path(args.autoencoder),
            args.episode,
            device=args.device,
        )
    else:
        result = validate_collection(
            Path(args.dataset),
            Path(args.autoencoder),
            expected_episodes=args.episodes,
            device=args.device,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
