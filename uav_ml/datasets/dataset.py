"""Deterministic episode-level NPZ dataset access."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from uav_ml.contracts import DATASET_VERSION


REQUIRED_ARRAYS = (
    "depth",
    "velocity",
    "goal_direction",
    "expert_action",
    "step",
    "timestamp_s",
    "goal_distance_m",
)


def load_metadata(dataset_path: str | Path) -> dict:
    """Load and minimally validate the root dataset metadata."""
    root = Path(dataset_path)
    path = root / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset metadata is missing: {path}")
    with path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("dataset_version") != DATASET_VERSION:
        raise ValueError(
            f"unsupported dataset version: {metadata.get('dataset_version')}"
        )
    return metadata


def discover_episodes(dataset_path: str | Path, split: str) -> list[Path]:
    """Return episode containers in stable lexical order."""
    if split not in ("train", "validation"):
        raise ValueError("split must be train or validation")
    directory = Path(dataset_path) / split
    return sorted(directory.glob("episode_*.npz"))


def load_episode(path: str | Path) -> dict[str, np.ndarray]:
    """Load one NPZ container without pickle or implicit object arrays."""
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_ARRAYS) - set(archive.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {missing}")
        return {name: archive[name].copy() for name in archive.files}


class BcEpisodeDataset(Dataset):
    """Frame dataset whose index order is deterministic across runs."""

    def __init__(self, dataset_path: str | Path, split: str) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.metadata = load_metadata(self.dataset_path)
        self.episode_paths = discover_episodes(self.dataset_path, split)
        if not self.episode_paths:
            raise ValueError(f"no {split} episodes found in {self.dataset_path}")
        self.episodes = [load_episode(path) for path in self.episode_paths]
        self.index: list[tuple[int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            for step_index in range(int(episode["expert_action"].shape[0])):
                self.index.append((episode_index, step_index))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, step_index = self.index[index]
        episode = self.episodes[episode_index]
        return {
            "depth": torch.from_numpy(episode["depth"][step_index]),
            "velocity": torch.from_numpy(episode["velocity"][step_index]),
            "goal_direction": torch.from_numpy(
                episode["goal_direction"][step_index]
            ),
            "action": torch.from_numpy(
                episode["expert_action"][step_index]
            ),
        }

