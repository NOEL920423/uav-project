"""Validate and rebuild the Phase 10A BC expert dataset V1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from uav_ml.inference.rgb_encoder import RgbEncoderInference


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RECORDER_SOURCE = REPOSITORY_ROOT / "ros2_ws" / "src" / "uav_data_recorder"
sys.path.insert(0, str(RECORDER_SOURCE))

from uav_data_recorder.expert_dataset_contract import (  # noqa: E402
    CSV_FIELDS,
    DATASET_VERSION,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SYNCHRONIZATION_TOLERANCE_S,
)


def _float(row: dict[str, str], name: str) -> float:
    value = float(row[name])
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def validate(
    dataset_root: Path,
    autoencoder_checkpoint: Path,
    device: str = "auto",
    write_result: bool = True,
) -> dict:
    """Validate one episode and rebuild every 72D observation/3D target."""
    dataset_root = dataset_root.resolve()
    manifest_path = dataset_root / "dataset_manifest.json"
    episode_dir = dataset_root / "episode_000001"
    episode_path = episode_dir / "episode.json"
    samples_path = episode_dir / "samples.csv"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    with episode_path.open(encoding="utf-8") as stream:
        episode = json.load(stream)
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError("dataset manifest version mismatch")
    if manifest.get("episodes") != ["episode_000001"]:
        raise ValueError("Phase 10A must contain exactly episode_000001")
    if not episode.get("success") or episode.get("status") != "complete":
        raise ValueError(f"episode did not complete successfully: {episode.get('failure')}")
    with samples_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("samples.csv fields do not match the V1 contract")
        rows = list(reader)
    if not rows:
        raise ValueError("episode contains no accepted samples")
    if len(rows) != int(episode.get("sample_count", -1)):
        raise ValueError("episode sample count does not match samples.csv")

    encoder = RgbEncoderInference(autoencoder_checkpoint, device=device)
    image_times: list[float] = []
    state_times: list[float] = []
    action_times: list[float] = []
    state_errors: list[float] = []
    action_errors: list[float] = []
    latent_norms: list[float] = []
    image_luminance_means: list[float] = []
    image_dynamic_ranges: list[int] = []
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index, row in enumerate(rows, start=1):
        if row["episode_id"] != "episode_000001":
            raise ValueError(f"sample {index} episode ID mismatch")
        if int(row["sample_id"]) != index:
            raise ValueError("sample IDs must be contiguous from one")
        if row["success"].lower() != "true" or row["failure"]:
            raise ValueError(f"sample {index} final outcome fields are invalid")
        image_path = dataset_root / row["image_path"]
        if not image_path.is_file() or not image_path.resolve().is_relative_to(dataset_root):
            raise ValueError(f"sample {index} image path is invalid")
        with Image.open(image_path) as image:
            if image.format != "JPEG":
                raise ValueError(f"sample {index} image is not JPEG")
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise ValueError(f"sample {index} image resolution mismatch")
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        luminance_mean = float(rgb.astype(np.float32).mean())
        dynamic_range = int(rgb.max()) - int(rgb.min())
        if luminance_mean < 5.0 or dynamic_range < 32:
            raise ValueError(
                f"sample {index} is blank/dark: mean={luminance_mean:.3f}, "
                f"range={dynamic_range}"
            )
        latent = encoder.encode(rgb).astype(np.float32)
        if latent.shape != (64,) or not np.isfinite(latent).all():
            raise ValueError(f"sample {index} did not produce a finite 64D latent")
        state = np.asarray([
            _float(row, "body_velocity_forward_mps"),
            _float(row, "body_velocity_right_mps"),
            _float(row, "goal_direction_forward"),
            _float(row, "goal_direction_right"),
            _float(row, "normalized_goal_distance"),
            _float(row, "previous_action_forward"),
            _float(row, "previous_action_right"),
            _float(row, "previous_action_yaw_rate"),
        ], dtype=np.float32)
        observation = np.concatenate((latent, state)).astype(np.float32)
        target = np.asarray([
            _float(row, "expert_action_forward"),
            _float(row, "expert_action_right"),
            _float(row, "expert_action_yaw_rate"),
        ], dtype=np.float32)
        if observation.shape != (72,) or not np.isfinite(observation).all():
            raise ValueError(f"sample {index} 72D observation rebuild failed")
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError(f"sample {index} 3D target rebuild failed")
        if np.max(np.abs(target)) > 1.0 + 1e-6:
            raise ValueError(f"sample {index} normalized target exceeds [-1,1]")

        image_time = _float(row, "image_timestamp_s")
        state_time = _float(row, "state_timestamp_s")
        action_time = _float(row, "expert_action_timestamp_s")
        state_error = abs(state_time - image_time)
        action_error = abs(action_time - image_time)
        if max(state_error, action_error) > SYNCHRONIZATION_TOLERANCE_S + 1e-9:
            raise ValueError(f"sample {index} exceeds synchronization tolerance")
        if abs(state_error - _float(row, "state_image_error_s")) > 1e-6:
            raise ValueError(f"sample {index} state error metadata mismatch")
        if abs(action_error - _float(row, "expert_action_image_error_s")) > 1e-6:
            raise ValueError(f"sample {index} action error metadata mismatch")
        image_times.append(image_time)
        state_times.append(state_time)
        action_times.append(action_time)
        state_errors.append(state_error)
        action_errors.append(action_error)
        latent_norms.append(float(np.linalg.norm(latent)))
        image_luminance_means.append(luminance_mean)
        image_dynamic_ranges.append(dynamic_range)
        observations.append(observation)
        targets.append(target)

    for name, values in (
        ("image", image_times), ("state", state_times), ("expert action", action_times)
    ):
        if any(current <= previous for previous, current in zip(values, values[1:])):
            raise ValueError(f"{name} timestamps are not strictly monotonic")
    observed_rate = 0.0
    if len(image_times) > 1:
        observed_rate = (len(image_times) - 1) / (image_times[-1] - image_times[0])
    if len(image_times) > 1 and not 3.0 <= observed_rate <= 7.0:
        raise ValueError(f"observed dataset rate is unreasonable: {observed_rate:.3f} Hz")
    terminal = episode.get("terminal_flight_status") or {}
    accumulated = episode.get("accumulated_flight_evidence") or {}
    required_accumulated = {
        "goal_reached": True,
        "landing_commanded": True,
        "landed_after_landing_command": True,
        "terminal_complete": True,
    }
    if terminal.get("state") != "COMPLETE" or any(
        accumulated.get(name) != value
        for name, value in required_accumulated.items()
    ):
        raise ValueError("terminal flight status lacks goal/landing success evidence")

    observation_array = np.stack(observations)
    target_array = np.stack(targets)
    result = {
        "valid": True,
        "dataset_version": DATASET_VERSION,
        "episode_id": "episode_000001",
        "episode_success": True,
        "sample_count": len(rows),
        "observed_sampling_rate_hz": observed_rate,
        "timestamps_strictly_monotonic": True,
        "maximum_state_image_error_s": max(state_errors),
        "maximum_action_image_error_s": max(action_errors),
        "synchronization_tolerance_s": SYNCHRONIZATION_TOLERANCE_S,
        "images_opened": len(rows),
        "image_resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "image_format": "JPEG",
        "image_luminance_mean_min_max": [
            min(image_luminance_means), max(image_luminance_means)
        ],
        "image_dynamic_range_min_max": [
            min(image_dynamic_ranges), max(image_dynamic_ranges)
        ],
        "latent_dimension": int(observation_array.shape[1] - 8),
        "observation_dimension": int(observation_array.shape[1]),
        "target_dimension": int(target_array.shape[1]),
        "latent_norm_min_max": [min(latent_norms), max(latent_norms)],
        "target_min": target_array.min(axis=0).tolist(),
        "target_max": target_array.max(axis=0).tolist(),
        "dataset_disk_usage_bytes": _directory_size(dataset_root),
        "autoencoder_checkpoint": str(autoencoder_checkpoint.resolve()),
        "autoencoder_checkpoint_sha256": _sha256(autoencoder_checkpoint),
        "preprocessing_rebuild": "RGB JPEG -> bilinear 128x72 -> [0,1] -> frozen 64D encoder",
    }
    if write_result:
        validation_path = episode_dir / "validation.json"
        validation_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["dataset_disk_usage_bytes"] = _directory_size(dataset_root)
        validation_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        manifest["validation"] = {
            "valid": True,
            "path": "episode_000001/validation.json",
            "sample_count": len(rows),
            "dataset_disk_usage_bytes": result["dataset_disk_usage_bytes"],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="artifacts/datasets/bc_expert_v1"
    )
    parser.add_argument(
        "--autoencoder",
        default="autoencoder_runs/rgb_ae_v0_baseline_20260811/best.pt",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = validate(
        Path(args.dataset), Path(args.autoencoder), device=args.device
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
