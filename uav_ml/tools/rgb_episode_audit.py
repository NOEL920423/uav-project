"""Audit legacy dual-camera RGB episodes without modifying source data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


REQUIRED_COLUMNS = {
    "episode_id",
    "frame_index",
    "time_wall",
    "record_time",
    "sim_time",
    "fpv_image_path",
    "top_image_path",
    "uav_x_isaac",
    "uav_y_isaac",
    "uav_z_isaac",
    "capture_clock",
    "image_width",
    "image_height",
}


@dataclass
class EpisodeAudit:
    episode_id: str
    category: str
    environment: str
    eligible_autoencoder: bool
    eligible_expert_bc: bool
    manifest_rows: int
    fpv_files: int
    top_files: int
    pose_rows: int
    duration_sim_s: float | None
    displacement_m: float | None
    fpv_dimensions: str
    top_dimensions: str
    corrupt_images: int
    missing_images: int
    orphan_images: int
    duplicate_consecutive_fpv: int
    duplicate_consecutive_top: int
    size_bytes: int
    issue_count: int
    status: str
    issues: list[str]


def _classify(episode_id: str) -> tuple[str, str]:
    if "forced_astar_bc_" in episode_id or "city_bc_" in episode_id:
        category = "policy_rollout"
    elif "_astar_" in episode_id:
        category = "astar_expert"
    else:
        category = "unknown"

    if "city_" in episode_id:
        environment = "city"
    elif "natural_" in episode_id:
        environment = "natural"
    elif "forced_" in episode_id:
        environment = "forced"
    else:
        environment = "baseline"
    return category, environment


def _pose_path(pose_root: Path, episode_id: str) -> Path:
    suffix = episode_id.removeprefix("dual_camera_episode_")
    return pose_root / f"uav_pose_{suffix}.csv"


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as stream:
        return max(sum(1 for _ in csv.reader(stream)) - 1, 0)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite number: {value}")
    return number


def _verify_image(path: Path) -> tuple[tuple[int, int], str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    with Image.open(path) as image:
        dimensions = image.size
        mode = image.mode
        image.verify()
    return dimensions, mode, digest.hexdigest()


def _monotonic(values: Iterable[float]) -> bool:
    sequence = list(values)
    return all(right > left for left, right in zip(sequence, sequence[1:]))


def audit_episode(episode_dir: Path, pose_root: Path) -> EpisodeAudit:
    episode_id = episode_dir.name
    category, environment = _classify(episode_id)
    manifest_path = episode_dir / "camera_frames.csv"
    fpv_dir = episode_dir / "images" / "fpv"
    top_dir = episode_dir / "images" / "top"
    fpv_files = sorted(fpv_dir.glob("*.png"))
    top_files = sorted(top_dir.glob("*.png"))
    issues: list[str] = []
    corrupt_images = 0
    missing_images = 0
    orphan_images = 0
    duplicate_fpv = 0
    duplicate_top = 0
    dimensions: dict[str, Counter[str]] = {"fpv": Counter(), "top": Counter()}
    digests: dict[str, list[str]] = {"fpv": [], "top": []}
    rows: list[dict[str, str]] = []

    if not manifest_path.is_file():
        issues.append("missing_manifest")
    else:
        try:
            with manifest_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                missing_columns = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
                if missing_columns:
                    issues.append("missing_columns:" + ",".join(missing_columns))
                rows = list(reader)
        except (csv.Error, UnicodeError) as exc:
            issues.append(f"invalid_manifest:{type(exc).__name__}")

    expected: dict[str, set[Path]] = {"fpv": set(), "top": set()}
    frame_indices: list[int] = []
    sim_times: list[float] = []
    positions: list[tuple[float, float, float]] = []
    for row_number, row in enumerate(rows, start=2):
        if row.get("episode_id") != episode_id:
            issues.append(f"episode_id_mismatch:row_{row_number}")
        try:
            frame_indices.append(int(row["frame_index"]))
            sim_times.append(_finite_float(row["sim_time"]))
            positions.append(
                (
                    _finite_float(row["uav_x_isaac"]),
                    _finite_float(row["uav_y_isaac"]),
                    _finite_float(row["uav_z_isaac"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            issues.append(f"invalid_numeric_value:row_{row_number}")

        for camera in ("fpv", "top"):
            raw_path = row.get(f"{camera}_image_path", "")
            name = Path(raw_path).name if raw_path else ""
            local_path = episode_dir / "images" / camera / name
            if name:
                expected[camera].add(local_path)
            if not name or not local_path.is_file():
                missing_images += 1
                issues.append(f"missing_{camera}_image:row_{row_number}")

    expected_indices = list(range(1, len(rows) + 1))
    if frame_indices != expected_indices:
        issues.append("nonsequential_frame_indices")
    if sim_times and not _monotonic(sim_times):
        issues.append("nonmonotonic_sim_time")

    actual = {"fpv": set(fpv_files), "top": set(top_files)}
    orphan_images = sum(len(actual[camera] - expected[camera]) for camera in actual)
    if orphan_images:
        issues.append(f"orphan_images:{orphan_images}")

    for camera, files in (("fpv", fpv_files), ("top", top_files)):
        for path in files:
            try:
                image_size, mode, digest = _verify_image(path)
                dimensions[camera][f"{image_size[0]}x{image_size[1]}:{mode}"] += 1
                digests[camera].append(digest)
            except (OSError, ValueError):
                corrupt_images += 1
                issues.append(f"corrupt_image:{path.relative_to(episode_dir)}")
        duplicates = sum(
            left == right for left, right in zip(digests[camera], digests[camera][1:])
        )
        if camera == "fpv":
            duplicate_fpv = duplicates
        else:
            duplicate_top = duplicates

    if len(rows) != len(fpv_files) or len(rows) != len(top_files):
        issues.append("manifest_image_count_mismatch")
    if len(dimensions["fpv"]) > 1:
        issues.append("mixed_fpv_dimensions")
    if len(dimensions["top"]) > 1:
        issues.append("mixed_top_dimensions")

    pose_path = _pose_path(pose_root, episode_id)
    pose_rows = _count_csv_rows(pose_path)
    if not pose_path.is_file():
        issues.append("missing_pose_log")
    elif pose_rows == 0:
        issues.append("empty_pose_log")

    duration = sim_times[-1] - sim_times[0] if len(sim_times) >= 2 else None
    displacement = None
    if len(positions) >= 2:
        displacement = math.dist(positions[0], positions[-1])

    issues = list(dict.fromkeys(issues))
    structurally_valid = not any(
        issue.startswith(
            (
                "missing_",
                "invalid_",
                "corrupt_",
                "orphan_",
                "manifest_",
                "nonsequential_",
                "nonmonotonic_",
                "mixed_",
                "episode_id_",
                "empty_",
            )
        )
        for issue in issues
    )
    # Policy rollouts remain evaluation evidence.  Excluding them prevents an
    # unsupervised encoder from seeing evaluation observations during training.
    eligible_autoencoder = (
        structurally_valid and len(fpv_files) > 0 and category == "astar_expert"
    )
    eligible_expert_bc = eligible_autoencoder and category == "astar_expert"

    def dimension_summary(camera: str) -> str:
        return ";".join(
            f"{label}={count}" for label, count in sorted(dimensions[camera].items())
        )

    return EpisodeAudit(
        episode_id=episode_id,
        category=category,
        environment=environment,
        eligible_autoencoder=eligible_autoencoder,
        eligible_expert_bc=eligible_expert_bc,
        manifest_rows=len(rows),
        fpv_files=len(fpv_files),
        top_files=len(top_files),
        pose_rows=pose_rows,
        duration_sim_s=duration,
        displacement_m=displacement,
        fpv_dimensions=dimension_summary("fpv"),
        top_dimensions=dimension_summary("top"),
        corrupt_images=corrupt_images,
        missing_images=missing_images,
        orphan_images=orphan_images,
        duplicate_consecutive_fpv=duplicate_fpv,
        duplicate_consecutive_top=duplicate_top,
        size_bytes=_directory_size(episode_dir),
        issue_count=len(issues),
        status="valid" if structurally_valid else "quarantine_candidate",
        issues=issues,
    )


def _episode_split(
    audits: list[EpisodeAudit], seed: int
) -> dict[str, list[str]]:
    """Make an episode-level split with every environment in val and test."""
    rng = random.Random(seed)
    by_environment: dict[str, list[str]] = {}
    for audit in audits:
        if audit.eligible_autoencoder:
            by_environment.setdefault(audit.environment, []).append(audit.episode_id)

    split = {"train": [], "validation": [], "test": []}
    for environment in sorted(by_environment):
        episode_ids = sorted(by_environment[environment])
        if len(episode_ids) < 3:
            raise ValueError(
                f"Environment {environment!r} needs at least three valid expert episodes"
            )
        rng.shuffle(episode_ids)
        split["validation"].append(episode_ids[0])
        split["test"].append(episode_ids[1])
        split["train"].extend(episode_ids[2:])
    for episode_ids in split.values():
        episode_ids.sort()
    return split


def write_reports(
    audits: list[EpisodeAudit], output_dir: Path, split_seed: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(audit) for audit in audits]
    with (output_dir / "inventory.json").open("w", encoding="utf-8") as stream:
        json.dump(records, stream, indent=2, ensure_ascii=False)

    fieldnames = [key for key in records[0] if key != "issues"] + ["issues"]
    with (output_dir / "inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            record["issues"] = ";".join(record["issues"])
            writer.writerow(record)

    selections = {
        "autoencoder_episodes.txt": [a.episode_id for a in audits if a.eligible_autoencoder],
        "expert_bc_episodes.txt": [a.episode_id for a in audits if a.eligible_expert_bc],
        "policy_rollout_episodes.txt": [a.episode_id for a in audits if a.category == "policy_rollout"],
        "quarantine_candidates.txt": [a.episode_id for a in audits if a.status != "valid"],
    }
    for filename, episode_ids in selections.items():
        text = "".join(f"{episode_id}\n" for episode_id in episode_ids)
        (output_dir / filename).write_text(text, encoding="utf-8")

    split = _episode_split(audits, split_seed)
    split_payload = {
        "seed": split_seed,
        "unit": "episode",
        "camera": "fpv",
        "top_camera_usage": "excluded_external_observer_view",
        "policy_rollout_usage": "excluded_from_all_training_splits",
        "splits": split,
    }
    with (output_dir / "autoencoder_split.json").open("w", encoding="utf-8") as stream:
        json.dump(split_payload, stream, indent=2, ensure_ascii=False)
    for split_name, episode_ids in split.items():
        text = "".join(f"{episode_id}\n" for episode_id in episode_ids)
        (output_dir / f"autoencoder_{split_name}_episodes.txt").write_text(
            text, encoding="utf-8"
        )

    summary = {
        "episodes": len(audits),
        "valid_episodes": sum(a.status == "valid" for a in audits),
        "quarantine_candidates": sum(a.status != "valid" for a in audits),
        "astar_expert_episodes": sum(a.category == "astar_expert" for a in audits),
        "policy_rollout_episodes": sum(a.category == "policy_rollout" for a in audits),
        "autoencoder_eligible_episodes": sum(a.eligible_autoencoder for a in audits),
        "expert_bc_eligible_episodes": sum(a.eligible_expert_bc for a in audits),
        "manifest_frames": sum(a.manifest_rows for a in audits),
        "rgb_images": sum(a.fpv_files + a.top_files for a in audits),
        "fpv_images": sum(a.fpv_files for a in audits),
        "top_images": sum(a.top_files for a in audits),
        "corrupt_images": sum(a.corrupt_images for a in audits),
        "missing_images": sum(a.missing_images for a in audits),
        "orphan_images": sum(a.orphan_images for a in audits),
        "size_bytes": sum(a.size_bytes for a in audits),
        "recommended_autoencoder_camera": "fpv",
        "excluded_autoencoder_camera": "top",
        "split_seed": split_seed,
        "train_episodes": len(split["train"]),
        "validation_episodes": len(split["validation"]),
        "test_episodes": len(split["test"]),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("uav_vision_dataset"))
    parser.add_argument("--pose-root", type=Path, default=Path("ros2_uav_pose_logs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("uav_vision_dataset/_audit"),
    )
    parser.add_argument("--split-seed", type=int, default=614420090)
    args = parser.parse_args()

    episode_dirs = sorted(
        path
        for path in args.dataset_root.glob("dual_camera_episode_*")
        if path.is_dir()
    )
    if not episode_dirs:
        raise SystemExit(f"No RGB episodes found under {args.dataset_root}")
    audits = [audit_episode(path, args.pose_root) for path in episode_dirs]
    write_reports(audits, args.output_dir, args.split_seed)

    valid = sum(audit.status == "valid" for audit in audits)
    experts = sum(audit.eligible_expert_bc for audit in audits)
    policy = sum(audit.category == "policy_rollout" for audit in audits)
    images = sum(audit.fpv_files + audit.top_files for audit in audits)
    print(f"episodes={len(audits)} valid={valid} quarantine={len(audits) - valid}")
    print(f"expert_bc={experts} policy_rollout={policy} images={images}")
    print(f"report={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
