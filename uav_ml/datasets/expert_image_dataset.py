"""Image datasets and preprocessing for formal and legacy expert episodes."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from collections.abc import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_SOURCES = ("fpv_rgb", "fpv_depth", "top")
IMAGE_PREPROCESSING = {
    "fpv_rgb": "PIL RGB -> bilinear 128x72 -> CHW float32 [0,1]",
    "top": "PIL RGB -> bilinear 128x72 -> CHW float32 [0,1]",
    "fpv_depth": (
        "PNG uint16 millimetres; invalid 0 -> 0; clip [50,30000] mm; "
        "linear [0,1] -> repeat 3 channels -> bilinear 128x72"
    ),
}


def preprocess_rgb_image(
    image: Image.Image, image_width: int = 128, image_height: int = 72
) -> torch.Tensor:
    """Apply the canonical RGB conversion, bilinear resize, and [0,1] scaling."""
    resized = image.convert("RGB").resize(
        (image_width, image_height),
        resample=Image.Resampling.BILINEAR,
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def normalize_image_sources(image_sources: Sequence[str]) -> tuple[str, ...]:
    """Validate and preserve the requested image-source order."""
    sources = tuple(str(source).strip().lower() for source in image_sources)
    if not sources:
        raise ValueError("at least one image source is required")
    if len(set(sources)) != len(sources):
        raise ValueError("image sources must not contain duplicates")
    unsupported = [source for source in sources if source not in IMAGE_SOURCES]
    if unsupported:
        raise ValueError(f"unsupported image sources: {', '.join(unsupported)}")
    return sources


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV has no samples: {path}")
    return rows


def _finite(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def select_episode_images(
    dataset_root: Path, episode_id: str, image_source: str
) -> list[dict[str, object]]:
    """Resolve source images with an exact sample/auxiliary identity join."""
    if image_source not in IMAGE_SOURCES:
        raise ValueError(f"unsupported image source: {image_source}")
    dataset_root = dataset_root.resolve()
    episode_dir = dataset_root / episode_id
    samples = _read_rows(episode_dir / "samples.csv")
    auxiliary = None
    if image_source != "fpv_rgb":
        auxiliary = _read_rows(episode_dir / "auxiliary.csv")
        if len(auxiliary) != len(samples):
            raise ValueError(f"{episode_id}: auxiliary row count mismatch")

    selected = []
    for index, sample in enumerate(samples):
        sample_id = sample.get("sample_id", "")
        if sample.get("episode_id") != episode_id:
            raise ValueError(f"{episode_id}: samples.csv episode identity mismatch")
        primary_timestamp = _finite(
            sample["image_timestamp_s"], "primary image timestamp"
        )
        source_timestamp = primary_timestamp
        source_error = 0.0
        if image_source == "fpv_rgb":
            relative_path = sample["image_path"]
        else:
            assert auxiliary is not None
            joined = auxiliary[index]
            if (
                joined.get("episode_id") != episode_id
                or joined.get("sample_id") != sample_id
            ):
                raise ValueError(
                    f"{episode_id}: auxiliary identity mismatch at sample {sample_id}"
                )
            auxiliary_primary = _finite(
                joined["primary_image_timestamp_s"], "auxiliary primary timestamp"
            )
            if abs(auxiliary_primary - primary_timestamp) > 1e-6:
                raise ValueError(
                    f"{episode_id}: auxiliary primary timestamp mismatch at "
                    f"sample {sample_id}"
                )
            prefix = "observer_rgb" if image_source == "top" else "fpv_depth"
            tolerance = 0.35 if image_source == "top" else 0.10
            if joined.get(f"{prefix}_available", "").lower() != "true":
                raise ValueError(
                    f"{episode_id}: {image_source} unavailable at sample {sample_id}"
                )
            if joined.get(f"{prefix}_status") != "matched":
                raise ValueError(
                    f"{episode_id}: {image_source} is not matched at sample {sample_id}"
                )
            relative_path = joined.get(f"{prefix}_path", "")
            source_timestamp = _finite(
                joined[f"{prefix}_timestamp_s"], f"{image_source} timestamp"
            )
            source_error = _finite(
                joined[f"{prefix}_error_s"], f"{image_source} timestamp error"
            )
            measured_error = abs(source_timestamp - primary_timestamp)
            if source_error > tolerance + 1e-9 or measured_error > tolerance + 1e-6:
                raise ValueError(
                    f"{episode_id}: {image_source} join exceeds {tolerance:.2f}s "
                    f"at sample {sample_id}"
                )
            if abs(source_error - measured_error) > 1e-5:
                raise ValueError(
                    f"{episode_id}: {image_source} timestamp error metadata mismatch "
                    f"at sample {sample_id}"
                )
        if not relative_path:
            raise ValueError(
                f"{episode_id}: {image_source} path missing at sample {sample_id}"
            )
        path = (dataset_root / relative_path).resolve()
        if not path.is_relative_to(dataset_root) or not path.is_file():
            raise FileNotFoundError(
                f"{episode_id}: invalid {image_source} path at sample "
                f"{sample_id}: {path}"
            )
        with Image.open(path) as image:
            image.verify()
        selected.append({
            "episode_id": episode_id,
            "sample_id": int(sample_id),
            "primary_timestamp_s": primary_timestamp,
            "source_timestamp_s": source_timestamp,
            "source_error_s": source_error,
            "image_path": path,
        })
    return selected


def select_episode_image_sets(
    dataset_root: Path, episode_id: str, image_sources: Sequence[str]
) -> list[dict[str, object]]:
    """Resolve a timestamp-matched image bundle for each primary sample."""
    sources = normalize_image_sources(image_sources)
    selected_by_source = {
        source: select_episode_images(dataset_root, episode_id, source)
        for source in sources
    }
    primary = selected_by_source[sources[0]]
    for source in sources[1:]:
        rows = selected_by_source[source]
        if len(rows) != len(primary):
            raise ValueError(f"{episode_id}: {source} row count mismatch")
        for index, (primary_row, row) in enumerate(zip(primary, rows)):
            if row["sample_id"] != primary_row["sample_id"]:
                raise ValueError(
                    f"{episode_id}: {source} sample identity mismatch at row {index}"
                )
    for source in sources:
        if source == "fpv_rgb":
            continue
        tolerance = 0.001 if source == "top" else 0.10
        for row in selected_by_source[source]:
            if float(row["source_error_s"]) > tolerance + 1e-9:
                raise ValueError(
                    f"{episode_id}: {source} is not synchronized closely enough "
                    f"for multi-image input at sample {row['sample_id']}"
                )
    return [
        {
            **primary_row,
            "images": {
                source: selected_by_source[source][index]["image_path"]
                for source in sources
            },
            "source_timestamps_s": {
                source: selected_by_source[source][index]["source_timestamp_s"]
                for source in sources
            },
            "source_errors_s": {
                source: selected_by_source[source][index]["source_error_s"]
                for source in sources
            },
        }
        for index, primary_row in enumerate(primary)
    ]


def preprocess_expert_image(
    path: Path, image_source: str, image_width: int = 128, image_height: int = 72
) -> torch.Tensor:
    """Apply the recorded source-specific preprocessing contract."""
    if image_source in {"fpv_rgb", "top"}:
        with Image.open(path) as image:
            return preprocess_rgb_image(image, image_width, image_height)
    if image_source != "fpv_depth":
        raise ValueError(f"unsupported image source: {image_source}")
    with Image.open(path) as image:
        return preprocess_depth_image(image, image_width, image_height)


def preprocess_depth_image(
    image: Image.Image, image_width: int = 128, image_height: int = 72
) -> torch.Tensor:
    """Apply the formal uint16-millimetre depth preprocessing to one image."""
    depth = np.asarray(image, dtype=np.uint16)
    if depth.ndim != 2:
        raise ValueError("FPV depth must be single-channel uint16")
    valid = depth != 0
    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = (
        np.clip(depth[valid], 50, 30000).astype(np.float32) - 50.0
    ) / 29950.0
    rgb = np.repeat(normalized[..., None], 3, axis=2)
    # PIL mode F preserves normalized values while applying bilinear resize.
    channels = []
    for channel in range(3):
        resized = Image.fromarray(rgb[..., channel], mode="F").resize(
            (image_width, image_height), resample=Image.Resampling.BILINEAR
        )
        channels.append(np.asarray(resized, dtype=np.float32).copy())
    return torch.from_numpy(np.stack(channels)).contiguous()


class ExpertImageDataset(Dataset):
    """Load one strictly associated image for each expert action sample."""

    def __init__(
        self,
        dataset_root: str | Path,
        split_manifest: dict,
        split: str,
        image_source: str | None = None,
        image_sources: Sequence[str] | None = None,
        image_width: int = 128,
        image_height: int = 72,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.dataset_root = Path(dataset_root).resolve()
        self.split = split
        if image_sources is None:
            if image_source is None:
                raise ValueError("image_source or image_sources must be provided")
            sources = normalize_image_sources((image_source,))
        else:
            if image_source is not None:
                raise ValueError("provide image_source or image_sources, not both")
            sources = normalize_image_sources(image_sources)
        self.image_sources = sources
        self.image_source = sources[0] if len(sources) == 1 else None
        self.image_width = image_width
        self.image_height = image_height
        self.split_metadata = split_manifest
        try:
            self.episode_ids = tuple(split_manifest["splits"][split])
        except (KeyError, TypeError) as error:
            raise ValueError(f"split manifest does not define {split!r}") from error
        if not self.episode_ids:
            raise ValueError(f"split {split!r} is empty")
        self.samples = []
        for episode_id in self.episode_ids:
            if len(sources) == 1:
                self.samples.extend(select_episode_images(
                    self.dataset_root, str(episode_id), sources[0]
                ))
            else:
                self.samples.extend(select_episode_image_sets(
                    self.dataset_root, str(episode_id), sources
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        result = {
            **sample,
            "image_path": str(sample["image_path"]),
        }
        if len(self.image_sources) == 1:
            result["image"] = preprocess_expert_image(
                sample["image_path"],
                self.image_sources[0],
                self.image_width,
                self.image_height,
            )
        else:
            result["image_paths"] = {
                source: str(sample["images"][source])
                for source in self.image_sources
            }
            result["images"] = {
                source: preprocess_expert_image(
                    sample["images"][source],
                    source,
                    self.image_width,
                    self.image_height,
                )
                for source in self.image_sources
            }
        return result


class RgbEpisodeDataset(Dataset):
    """Load FPV frames from one precomputed, episode-disjoint split."""

    def __init__(
        self,
        dataset_root: str | Path,
        split_file: str | Path,
        split: str,
        image_width: int = 128,
        image_height: int = 72,
    ) -> None:
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if image_width < 16 or image_height < 16:
            raise ValueError("image dimensions must be at least 16 pixels")
        self.dataset_root = Path(dataset_root)
        self.split_file = Path(split_file)
        self.split = split
        self.image_width = image_width
        self.image_height = image_height

        with self.split_file.open(encoding="utf-8") as stream:
            self.split_metadata = json.load(stream)
        if self.split_metadata.get("unit") != "episode":
            raise ValueError("RGB split must use complete episodes")
        if self.split_metadata.get("camera") != "fpv":
            raise ValueError("Autoencoder baseline must use the FPV camera")
        try:
            episode_ids = self.split_metadata["splits"][split]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"split file does not define {split!r}") from exc
        if not episode_ids:
            raise ValueError(f"split {split!r} is empty")

        self.episode_ids = tuple(str(value) for value in episode_ids)
        self.samples: list[dict[str, str | int | float | Path]] = []
        for episode_id in self.episode_ids:
            episode_dir = self.dataset_root / episode_id
            manifest_path = episode_dir / "camera_frames.csv"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            with manifest_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    if row["episode_id"] != episode_id:
                        raise ValueError(f"episode ID mismatch in {manifest_path}")
                    image_path = (
                        episode_dir
                        / "images"
                        / "fpv"
                        / Path(row["fpv_image_path"]).name
                    )
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    self.samples.append(
                        {
                            "episode_id": episode_id,
                            "frame_index": int(row["frame_index"]),
                            "sim_time": float(row["sim_time"]),
                            "image_path": image_path,
                        }
                    )
        if not self.samples:
            raise ValueError(f"split {split!r} contains no RGB frames")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        with Image.open(sample["image_path"]) as image:
            tensor = preprocess_rgb_image(
                image,
                image_width=self.image_width,
                image_height=self.image_height,
            )
        return {
            "image": tensor,
            "episode_id": sample["episode_id"],
            "frame_index": sample["frame_index"],
            "sim_time": sample["sim_time"],
            "image_path": str(sample["image_path"]),
        }
