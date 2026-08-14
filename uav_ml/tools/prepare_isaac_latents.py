"""Encode synchronized Isaac RGB demonstrations and fit train-only normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from uav_ml.models import RgbAutoencoderConfig, RgbAutoencoderV0
from uav_ml.train_bc import resolve_device


def _load_encoder(checkpoint: Path, device: torch.device) -> RgbAutoencoderV0:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = RgbAutoencoderV0(RgbAutoencoderConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval()


def _encode(rgb: np.ndarray, model: RgbAutoencoderV0, batch_size: int) -> np.ndarray:
    outputs = []
    device = next(model.parameters()).device
    for start in range(0, len(rgb), batch_size):
        tensor = torch.from_numpy(rgb[start : start + batch_size]).to(
            device=device, dtype=torch.float32
        )
        tensor = tensor.permute(0, 3, 1, 2) / 255.0
        with torch.inference_mode():
            outputs.append(model.encode(tensor).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def prepare(
    dataset_root: Path,
    output_root: Path,
    autoencoder_checkpoint: Path,
    device_name: str,
    batch_size: int,
) -> dict:
    device = resolve_device(device_name)
    model = _load_encoder(autoencoder_checkpoint, device)
    output_root.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, list[dict]] = {}
    train_observations = []
    for split in ("train", "validation", "test"):
        records = []
        for path in sorted((dataset_root / split).glob("episode_*.npz")):
            with np.load(path) as data:
                rgb = data["rgb"]
                state = data["state"].astype(np.float32)
                action = data["expert_action"].astype(np.float32)
                seed = int(data["seed"])
            latent = _encode(rgb, model, batch_size)
            observation = np.concatenate((latent, state), axis=1).astype(np.float32)
            output_path = output_root / split / path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                latent=latent,
                state=state,
                observation=observation,
                expert_action=action,
                seed=np.asarray(seed, dtype=np.int64),
            )
            if split == "train":
                train_observations.append(observation)
            records.append(
                {"seed": seed, "samples": len(observation), "path": str(output_path.resolve())}
            )
        if not records:
            raise ValueError(f"no episodes found for split {split!r}")
        split_records[split] = records
    train = np.concatenate(train_observations)
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    normalization_path = output_root / "observation_normalization.npz"
    np.savez(
        normalization_path,
        mean=mean,
        std=std,
        latent_mean=mean[:64],
        latent_std=std[:64],
        state_mean=mean[64:],
        state_std=std[64:],
    )
    summary = {
        "format_version": "isaac_city_latent_bc_v0.1",
        "autoencoder_checkpoint": str(autoencoder_checkpoint.resolve()),
        "latent_dimension": 64,
        "state_dimension": 8,
        "observation_dimension": 72,
        "action_dimension": 3,
        "normalization_source": "train_only",
        "normalization_path": str(normalization_path.resolve()),
        "latent_mean_min_max": [float(mean[:64].min()), float(mean[:64].max())],
        "latent_std_min_max": [float(std[:64].min()), float(std[:64].max())],
        "splits": split_records,
        "sample_counts": {
            split: sum(item["samples"] for item in records)
            for split, records in split_records.items()
        },
    }
    (output_root / "metadata.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/isaac_city_bc_v0")
    parser.add_argument("--output", default="datasets/isaac_city_bc_v0_latent")
    parser.add_argument(
        "--autoencoder",
        default="autoencoder_runs/rgb_ae_v0_baseline_20260811/best.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    prepare(
        Path(args.dataset),
        Path(args.output),
        Path(args.autoencoder),
        args.device,
        args.batch_size,
    )


if __name__ == "__main__":
    main()
