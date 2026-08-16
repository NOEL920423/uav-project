"""Validate a Phase 10B multi-episode BC expert dataset pilot."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from uav_ml.tools.validate_expert_dataset import (
    _directory_size,
    validate_episode,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_auxiliary(dataset_root: Path, episode_id: str) -> dict:
    episode_dir = dataset_root / episode_id
    with (episode_dir / "samples.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        sample_rows = list(csv.DictReader(stream))
    with (episode_dir / "auxiliary.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        reader = csv.DictReader(stream)
        auxiliary_fields = set(reader.fieldnames or ())
        rows = list(reader)
    if len(rows) != len(sample_rows):
        raise ValueError(f"{episode_id}: auxiliary row count mismatch")
    counts = Counter()
    for index, row in enumerate(rows, start=1):
        if row["episode_id"] != episode_id or int(row["sample_id"]) != index:
            raise ValueError(f"{episode_id}: auxiliary identity mismatch")
        observer_name = (
            "observer_rgb"
            if "observer_rgb_available" in auxiliary_fields
            else "top_rgb"
        )
        for name, tolerance, expected_format in (
            (observer_name, 0.35, "JPEG"),
            ("fpv_depth", 0.10, "PNG"),
        ):
            available = row[f"{name}_available"].lower() == "true"
            counts[f"{name}_{'available' if available else 'missing'}"] += 1
            if not available:
                if row[f"{name}_path"]:
                    raise ValueError(
                        f"{episode_id}: unavailable {name} has a path"
                    )
                continue
            error = float(row[f"{name}_error_s"])
            if error > tolerance + 1e-9:
                raise ValueError(f"{episode_id}: {name} join over tolerance")
            path = (dataset_root / row[f"{name}_path"]).resolve()
            if not path.is_file() or not path.is_relative_to(dataset_root):
                raise ValueError(f"{episode_id}: invalid {name} path")
            with Image.open(path) as image:
                if image.format != expected_format or image.size != (320, 180):
                    raise ValueError(f"{episode_id}: invalid {name} image")
                if name == "fpv_depth" and image.mode not in {"I;16", "I"}:
                    raise ValueError(f"{episode_id}: depth is not uint16 PNG")
                if name == "fpv_depth":
                    depth = np.asarray(image)
                    if depth.min() < 0 or depth.max() > 30000:
                        raise ValueError(
                            f"{episode_id}: depth is outside 0..30000 mm"
                        )
    return dict(counts)


def _validate_episode_metadata(episode: dict, validation: dict) -> None:
    episode_id = validation["episode_id"]
    scene = episode["scene_configuration"]
    if len(scene.get("goal", [])) != 3 or not scene.get("obstacles"):
        raise ValueError(f"{episode_id}: goal/obstacle metadata missing")
    rejection_count = int(episode.get("rejected_sample_count", -1))
    rejection_reasons = episode.get("rejections_by_reason")
    if not isinstance(rejection_reasons, dict) or rejection_count != sum(
        int(value) for value in rejection_reasons.values()
    ):
        raise ValueError(f"{episode_id}: rejection metadata mismatch")
    if int(episode.get("sample_count", -1)) != validation["sample_count"]:
        raise ValueError(f"{episode_id}: sample metadata mismatch")
    path_length = float(episode.get("path_length_m", -1.0))
    if not math.isfinite(path_length) or path_length < 0.0:
        raise ValueError(f"{episode_id}: invalid path length")
    final_distance = episode.get("final_tracking_goal_distance_m")
    if validation["sample_count"]:
        if final_distance is None or not math.isfinite(float(final_distance)):
            raise ValueError(f"{episode_id}: final goal distance missing")
    elif final_distance is not None:
        raise ValueError(f"{episode_id}: empty episode has a final goal distance")
    synchronization = episode.get("synchronization_statistics_s")
    if not isinstance(synchronization, dict):
        raise ValueError(f"{episode_id}: synchronization statistics missing")
    streams = episode.get("available_sensor_streams")
    if (
        not isinstance(streams, dict)
        or streams.get("fpv_rgb", {}).get("accepted")
        != validation["sample_count"]
    ):
        raise ValueError(f"{episode_id}: sensor stream metadata mismatch")
    if int(episode.get("episode_disk_usage_bytes", 0)) <= 0:
        raise ValueError(f"{episode_id}: disk usage metadata missing")


def validate_batch(
    dataset_root: Path,
    autoencoder_checkpoint: Path,
    expected_episodes: int = 10,
    device: str = "auto",
    write_result: bool = True,
) -> dict:
    """Validate every pilot episode, scene, sensor join, and aggregate stat."""
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = _load(manifest_path)
    episodes = list(manifest.get("episodes", []))
    expected_ids = [
        f"episode_{index:06d}" for index in range(1, expected_episodes + 1)
    ]
    if episodes != expected_ids:
        raise ValueError(
            f"expected ordered episodes {expected_ids}, received {episodes}"
        )

    validations = []
    metadata = []
    scene_keys = set()
    seeds = set()
    rejections = Counter()
    auxiliary = Counter()
    state_errors = []
    action_errors = []
    sample_counts = []
    rates = []
    for episode_id in episodes:
        episode = _load(dataset_root / episode_id / "episode.json")
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
        scene = episode.get("scene_configuration")
        if not isinstance(scene, dict):
            raise ValueError(f"{episode_id}: scene configuration missing")
        if scene.get("episode_id") != episode_id:
            raise ValueError(f"{episode_id}: scene identity mismatch")
        seed = int(episode.get("random_seed"))
        if scene.get("random_seed") != seed:
            raise ValueError(f"{episode_id}: scene seed mismatch")
        scene_key = json.dumps(scene, sort_keys=True, separators=(",", ":"))
        if scene_key in scene_keys or seed in seeds:
            raise ValueError("pilot scenes and seeds must be unique")
        scene_keys.add(scene_key)
        seeds.add(seed)
        _validate_episode_metadata(episode, validation)
        rejections.update(episode.get("rejections_by_reason", {}))
        auxiliary.update(_validate_auxiliary(dataset_root, episode_id))
        with (dataset_root / episode_id / "samples.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            for row in csv.DictReader(stream):
                state_errors.append(float(row["state_image_error_s"]))
                action_errors.append(float(row["expert_action_image_error_s"]))
        sample_counts.append(validation["sample_count"])
        if validation["sample_count"] > 1:
            rates.append(validation["observed_sampling_rate_hz"])
        validations.append(validation)
        metadata.append(episode)

    success_count = sum(item["episode_success"] for item in validations)
    failure_count = len(validations) - success_count
    if success_count == 0 or failure_count == 0:
        raise ValueError(
            "pilot must prove both successful collection and safe failure handling"
        )

    def statistics(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "p95": None, "max": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "p95": float(np.percentile(array, 95)),
            "max": float(array.max()),
        }

    total_bytes = _directory_size(dataset_root)
    result = {
        "valid": True,
        "dataset_version": manifest.get("dataset_version"),
        "episode_count": len(episodes),
        "successful_episodes": success_count,
        "failed_episodes": failure_count,
        "failure_episode_ids": [
            item["episode_id"] for item in validations
            if not item["episode_success"]
        ],
        "scenes_unique": len(scene_keys),
        "random_seeds_unique": len(seeds),
        "accepted_samples_per_episode": dict(zip(episodes, sample_counts)),
        "accepted_samples_total": sum(sample_counts),
        "rejection_breakdown": dict(sorted(rejections.items())),
        "effective_sampling_rate_hz": statistics(rates),
        "state_image_synchronization_error_s": statistics(state_errors),
        "action_image_synchronization_error_s": statistics(action_errors),
        "auxiliary_availability": dict(sorted(auxiliary.items())),
        "total_dataset_size_bytes": total_bytes,
        "average_mb_per_episode": total_bytes / len(episodes) / 1_000_000,
        "estimated_gb_per_1000_episodes": (
            total_bytes / len(episodes) * 1000 / 1_000_000_000
        ),
        "bc_contract_rebuilt": {
            "latent_dimension": 64,
            "observation_dimension": 72,
            "target_dimension": 3,
            "accepted_samples": sum(sample_counts),
        },
        "all_failure_episodes_safe": all(
            item.get("safe_terminal_evidence", {}).get("landed") is True
            and item.get("safe_terminal_evidence", {}).get("disarmed") is True
            and item.get("safe_terminal_evidence", {}).get("failsafe") is False
            for item in metadata if not item.get("success")
        ),
    }
    if write_result:
        output = dataset_root / "batch_validation.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["total_dataset_size_bytes"] = _directory_size(dataset_root)
        result["average_mb_per_episode"] = (
            result["total_dataset_size_bytes"] / len(episodes) / 1_000_000
        )
        result["estimated_gb_per_1000_episodes"] = (
            result["total_dataset_size_bytes"]
            / len(episodes) * 1000 / 1_000_000_000
        )
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        manifest.update({
            "status": "complete",
            "episode_count": len(episodes),
            "sample_count": sum(sample_counts),
            "successful_episodes": success_count,
            "failed_episodes": failure_count,
            "validation": {
                "valid": True,
                "path": "batch_validation.json",
            },
        })
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="artifacts/datasets/bc_expert_v1_phase10b"
    )
    parser.add_argument(
        "--autoencoder",
        default="autoencoder_runs/rgb_ae_v0_baseline_20260811/best.pt",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.episode:
        result = validate_episode(
            Path(args.dataset),
            Path(args.autoencoder),
            episode_id=args.episode,
            device=args.device,
            require_success=None,
            require_single_manifest=False,
            update_manifest=False,
        )
        result["auxiliary_availability"] = _validate_auxiliary(
            Path(args.dataset).resolve(), args.episode
        )
    else:
        result = validate_batch(
            Path(args.dataset),
            Path(args.autoencoder),
            expected_episodes=args.episodes,
            device=args.device,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
