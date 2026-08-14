"""Train the latent/state A* behavior-cloning actor for the Isaac city task."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from uav_ml.models import LatentBcPolicy, LatentBcPolicyConfig
from uav_ml.train_bc import resolve_device, set_seeds


ACTION_NAMES = ("forward", "right", "yaw_rate")


def _load_split(root: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    observations = []
    actions = []
    for path in sorted((root / split).glob("episode_*.npz")):
        with np.load(path) as data:
            observations.append(data["observation"].astype(np.float32))
            actions.append(data["expert_action"].astype(np.float32))
    if not observations:
        raise ValueError(f"no latent BC episodes in split {split!r}")
    return (
        torch.from_numpy(np.concatenate(observations)),
        torch.from_numpy(np.concatenate(actions)),
    )


def _metrics(
    model: LatentBcPolicy,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    squared = torch.zeros(3, dtype=torch.float64)
    absolute = torch.zeros(3, dtype=torch.float64)
    samples = 0
    for observation, target in loader:
        observation = observation.to(device)
        target = target.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(observation)
            loss = nn.functional.mse_loss(prediction, target)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        error = prediction.detach() - target
        squared += error.square().sum(dim=0).cpu().double()
        absolute += error.abs().sum(dim=0).cpu().double()
        samples += len(target)
    mse = squared / samples
    mae = absolute / samples
    return {"mse": float(mse.mean()), "mse_components": mse.tolist(), "mae_components": mae.tolist()}


def train(
    dataset_root: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
    seed: int,
) -> dict:
    set_seeds(seed)
    device = resolve_device(device_name)
    with np.load(dataset_root / "observation_normalization.npz") as norm:
        mean = norm["mean"].astype(np.float32)
        std = norm["std"].astype(np.float32)
    split_tensors = {}
    for split in ("train", "validation", "test"):
        observation, action = _load_split(dataset_root, split)
        normalized = (observation - torch.from_numpy(mean)) / torch.from_numpy(std)
        split_tensors[split] = (normalized, action)
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            TensorDataset(*tensors),
            batch_size=batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
        )
        for split, tensors in split_tensors.items()
    }
    model = LatentBcPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_mse = math.inf
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        _metrics(model, loaders["train"], device, optimizer)
        train_metrics = _metrics(model, loaders["train"], device, None)
        validation_metrics = _metrics(model, loaders["validation"], device, None)
        row = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "validation_mse": validation_metrics["mse"],
            **{
                f"train_{name}_mse": value
                for name, value in zip(ACTION_NAMES, train_metrics["mse_components"])
            },
            **{
                f"validation_{name}_mse": value
                for name, value in zip(ACTION_NAMES, validation_metrics["mse_components"])
            },
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_mse={row['train_mse']:.6f} "
            f"validation_mse={row['validation_mse']:.6f}",
            flush=True,
        )
        if validation_metrics["mse"] < best_mse:
            best_mse = validation_metrics["mse"]
            best_epoch = epoch
            torch.save(
                {
                    "format_version": "latent_bc_checkpoint_v0.1",
                    "model_class": "LatentBcPolicy",
                    "model_config": model.config.to_dict(),
                    "model_state": model.state_dict(),
                    "observation_mean": mean,
                    "observation_std": std,
                    "autoencoder_checkpoint": json.loads(
                        (dataset_root / "metadata.json").read_text(encoding="utf-8")
                    )["autoencoder_checkpoint"],
                    "best_epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "loss": "equal_component_normalized_action_mse",
                    "input_contract": "normalized_latent64_plus_state8_v0.1",
                    "output_contract": "normalized_body_forward_right_yaw_v0.1",
                },
                output_dir / "best.pt",
            )
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    best_model = LatentBcPolicy(LatentBcPolicyConfig(**checkpoint["model_config"])).to(device)
    best_model.load_state_dict(checkpoint["model_state"])
    test_metrics = _metrics(best_model, loaders["test"], device, None)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_mse": best_mse,
        "test_metrics": test_metrics,
        "parameter_count": best_model.parameter_count,
        "samples": {split: len(values[0]) for split, values in split_tensors.items()},
        "checkpoint": str((output_dir / "best.pt").resolve()),
        "history": str((output_dir / "history.csv").resolve()),
        "checkpoint_reload_verified": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/isaac_city_bc_v0_latent")
    parser.add_argument("--output", default="training_runs/latent_bc_city_v0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=614420090)
    args = parser.parse_args()
    train(
        Path(args.dataset), Path(args.output), args.epochs, args.batch_size,
        args.learning_rate, args.device, args.seed
    )


if __name__ == "__main__":
    main()
