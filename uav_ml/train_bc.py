"""Standalone ROS-free trainer for BcPolicyV0."""

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from uav_ml.contracts import ACTION_NAMES, DATASET_VERSION
from uav_ml.datasets.dataset import BcEpisodeDataset
from uav_ml.datasets.validation import validate_dataset
from uav_ml.models import BcPolicyV0
from uav_ml.training.checkpoint import load_checkpoint, save_checkpoint
from uav_ml.training.normalization import (
    TorchNormalizer,
    compute_normalization,
)


def set_seeds(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and all visible CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda without silently accepting unavailable CUDA."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _loader(
    dataset: BcEpisodeDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _epoch(
    model: BcPolicyV0,
    loader: DataLoader,
    normalizer: TorchNormalizer,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float | list[float]]:
    training = optimizer is not None
    model.train(training)
    squared_error = torch.zeros(4, dtype=torch.float64)
    absolute_error = torch.zeros(4, dtype=torch.float64)
    sample_count = 0
    for batch in loader:
        depth = batch["depth"].to(device)
        velocity = batch["velocity"].to(device)
        goal = batch["goal_direction"].to(device)
        target = batch["action"].to(device)
        depth, velocity, goal = normalizer.observation(depth, velocity, goal)
        normalized_target = normalizer.normalize_action(target)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            normalized_prediction = model(depth, velocity, goal)
            loss = nn.functional.mse_loss(
                normalized_prediction, normalized_target
            )
            if training:
                loss.backward()
                optimizer.step()
        prediction = normalizer.denormalize_action(normalized_prediction.detach())
        error = prediction - target
        squared_error += error.square().sum(dim=0).cpu().double()
        absolute_error += error.abs().sum(dim=0).cpu().double()
        sample_count += int(target.shape[0])
    if sample_count == 0:
        raise ValueError("empty data loader")
    mse = squared_error / sample_count
    mae = absolute_error / sample_count
    rmse = torch.sqrt(mse)
    return {
        "loss": float(mse.mean()),
        "mse": mse.tolist(),
        "mae": mae.tolist(),
        "rmse": rmse.tolist(),
    }


def train(
    dataset_path: str | Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device_name: str,
    seed: int,
    checkpoint_path: str | Path,
    history_path: str | Path,
    resume: str | Path | None = None,
) -> dict:
    """Run deterministic supervised regression and persist full metadata."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    dataset_stats = validate_dataset(dataset_path)
    set_seeds(seed)
    device = resolve_device(device_name)
    train_dataset = BcEpisodeDataset(dataset_path, "train")
    validation_dataset = BcEpisodeDataset(dataset_path, "validation")
    normalization = compute_normalization(train_dataset)
    if resume:
        model, resumed_normalization, payload = load_checkpoint(resume, device)
        if resumed_normalization != normalization:
            raise ValueError("resume checkpoint normalization does not match dataset")
    else:
        model = BcPolicyV0().to(device)
        payload = {}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if resume and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    normalizer = TorchNormalizer(normalization, device)
    train_loader = _loader(train_dataset, batch_size, True, seed)
    train_eval_loader = _loader(train_dataset, batch_size, False, seed)
    validation_loader = _loader(validation_dataset, batch_size, False, seed)
    initial_train = _epoch(model, train_eval_loader, normalizer, device, None)
    initial_validation = _epoch(
        model, validation_loader, normalizer, device, None
    )
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch(
            model, train_loader, normalizer, device, optimizer
        )
        validation_metrics = _epoch(
            model, validation_loader, normalizer, device, None
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "validation_mse": validation_metrics["mse"],
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
        }
        history.append(row)
        components = " ".join(
            f"{name}_mse={value:.6f}"
            for name, value in zip(ACTION_NAMES, train_metrics["mse"])
        )
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"validation_loss={validation_metrics['loss']:.6f} {components}"
        )
    metadata = {
        "git_sha": _git_sha(),
        "dataset_version": DATASET_VERSION,
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_statistics": dataset_stats,
        "seeds": {
            "python": seed,
            "numpy": seed,
            "torch": seed,
            "dataset_split": train_dataset.metadata["split_seed"],
        },
        "model_config": model.config.to_dict(),
        "parameter_count": model.parameter_count,
        "optimizer": {"name": "Adam", "learning_rate": learning_rate},
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "final_train": history[-1],
    }
    checkpoint = save_checkpoint(
        checkpoint_path, model, normalization, metadata, optimizer
    )
    history_output = Path(history_path)
    history_output.parent.mkdir(parents=True, exist_ok=True)
    with history_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["epoch", "train_loss", "validation_loss"]
        for split in ("train", "validation"):
            for metric in ("mse", "mae", "rmse"):
                header.extend(f"{split}_{name}_{metric}" for name in ACTION_NAMES)
        writer.writerow(header)
        for row in history:
            values = [row["epoch"], row["train_loss"], row["validation_loss"]]
            for split in ("train", "validation"):
                for metric in ("mse", "mae", "rmse"):
                    values.extend(row[f"{split}_{metric}"])
            writer.writerow(values)
    reloaded, _, _ = load_checkpoint(checkpoint, device)
    if reloaded.parameter_count != model.parameter_count:
        raise RuntimeError("checkpoint reload parameter count mismatch")
    result = {
        "initial_train_loss": initial_train["loss"],
        "final_train_loss": history[-1]["train_loss"],
        "initial_validation_loss": initial_validation["loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "checkpoint_path": str(checkpoint.resolve()),
        "history_path": str(history_output.resolve()),
        "parameter_count": model.parameter_count,
        "dataset_episodes": dataset_stats["episodes"],
        "dataset_samples": dataset_stats["samples"],
        "checkpoint_reload": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/bc_v0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--checkpoint", default="checkpoints/bc_v0.pt")
    parser.add_argument("--history", default="training_runs/bc_v0_history.csv")
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(
        args.dataset,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.device,
        args.seed,
        args.checkpoint,
        args.history,
        args.resume,
    )


if __name__ == "__main__":
    main()

