"""Episode-split FPV RGB dataset for Autoencoder pretraining."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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
