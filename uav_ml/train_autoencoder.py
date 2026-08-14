"""Train and evaluate the FPV RGB Autoencoder baseline without ROS or Isaac."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from uav_ml.datasets.rgb_episode_dataset import RgbEpisodeDataset
from uav_ml.models import RgbAutoencoderConfig, RgbAutoencoderV0
from uav_ml.train_bc import resolve_device, set_seeds


INPUT_CONTRACT = "fpv_rgb_128x72_float_v0.1"
LATENT_CONTRACT_TEMPLATE = "rgb_autoencoder_latent_{dimension}_v0.1"
OUTPUT_CONTRACT = "fpv_rgb_reconstruction_128x72_float_v0.1"
DATA_WARNING = (
    "Legacy FPV contains simulator-only red goal and rendered path/waypoint cues; "
    "this run validates representation learning, not map-free navigation."
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _loader(
    dataset: RgbEpisodeDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        generator=generator,
    )


def _epoch(
    model: RgbAutoencoderV0,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction, _ = model(image)
            loss = nn.functional.mse_loss(reconstruction, image)
            if training:
                loss.backward()
                optimizer.step()
        error = reconstruction.detach() - image
        squared_error += float(error.square().sum().cpu())
        absolute_error += float(error.abs().sum().cpu())
        element_count += error.numel()
    if element_count == 0:
        raise ValueError("empty Autoencoder data loader")
    mse = squared_error / element_count
    mae = absolute_error / element_count
    return {
        "mse": mse,
        "mae": mae,
        "psnr_db": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
    }


def _save_checkpoint(
    path: Path,
    model: RgbAutoencoderV0,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": "rgb_autoencoder_checkpoint_v0.1",
            "model_class": "RgbAutoencoderV0",
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "metadata": metadata,
        },
        path,
    )


def _load_model(path: Path, device: torch.device) -> tuple[RgbAutoencoderV0, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_class") != "RgbAutoencoderV0":
        raise ValueError("checkpoint model class is not RgbAutoencoderV0")
    model = RgbAutoencoderV0(RgbAutoencoderConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload


def _save_history(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _save_loss_curve(history: list[dict], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_mse"] for row in history], label="train MSE")
    plt.plot(
        epochs,
        [row["validation_mse"] for row in history],
        label="validation MSE",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Per-pixel MSE")
    plt.title("RGB Autoencoder baseline")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _save_reconstructions(
    model: RgbAutoencoderV0,
    dataset: RgbEpisodeDataset,
    device: torch.device,
    path: Path,
    count: int = 8,
) -> None:
    indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
    images = torch.stack([dataset[int(index)]["image"] for index in indices]).to(device)
    with torch.inference_mode():
        reconstructions, _ = model(images)
    width = model.config.image_width
    height = model.config.image_height
    label_height = 20
    canvas = Image.new("RGB", (width * len(indices), (height + label_height) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (original, reconstruction) in enumerate(zip(images, reconstructions)):
        x = column * width
        canvas.paste(_to_pil(original), (x, label_height))
        canvas.paste(_to_pil(reconstruction), (x, height + label_height * 2))
        draw.text((x + 4, 3), "original", fill="black")
        draw.text((x + 4, height + label_height + 3), "reconstruction", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def train(
    dataset_root: str | Path,
    split_file: str | Path,
    output_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    latent_dimension: int = 64,
    device_name: str = "auto",
    seed: int = 614420090,
    workers: int = 4,
) -> dict:
    """Train, select on validation MSE, then evaluate the held-out test split."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0.0 or workers < 0:
        raise ValueError("epochs/batch size/LR must be positive and workers nonnegative")
    if latent_dimension < 2:
        raise ValueError("latent dimension must be at least 2")
    set_seeds(seed)
    device = resolve_device(device_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    datasets = {
        split: RgbEpisodeDataset(dataset_root, split_file, split)
        for split in ("train", "validation", "test")
    }
    loaders = {
        "train": _loader(
            datasets["train"], batch_size, True, seed, workers, device.type == "cuda"
        ),
        "train_eval": _loader(
            datasets["train"], batch_size, False, seed, workers, device.type == "cuda"
        ),
        "validation": _loader(
            datasets["validation"], batch_size, False, seed, workers, device.type == "cuda"
        ),
        "test": _loader(
            datasets["test"], batch_size, False, seed, workers, device.type == "cuda"
        ),
    }
    model = RgbAutoencoderV0(
        RgbAutoencoderConfig(latent_dimension=latent_dimension)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metadata = {
        "git_sha": _git_sha(),
        "input_contract": INPUT_CONTRACT,
        "latent_contract": LATENT_CONTRACT_TEMPLATE.format(
            dimension=latent_dimension
        ),
        "output_contract": OUTPUT_CONTRACT,
        "dataset_root": str(Path(dataset_root).resolve()),
        "split_file": str(Path(split_file).resolve()),
        "split_seed": datasets["train"].split_metadata["seed"],
        "split_unit": "episode",
        "camera": "fpv",
        "top_camera": "excluded",
        "policy_rollouts": "excluded",
        "data_warning": DATA_WARNING,
        "samples": {split: len(dataset) for split, dataset in datasets.items()},
        "episodes": {split: len(dataset.episode_ids) for split, dataset in datasets.items()},
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "parameter_count": model.parameter_count,
        "optimizer": {"name": "Adam", "learning_rate": learning_rate},
        "batch_size": batch_size,
    }
    best_path = output / "best.pt"
    last_path = output / "last.pt"
    best_validation_mse = math.inf
    best_epoch = 0
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        _epoch(model, loaders["train"], device, optimizer)
        train_metrics = _epoch(model, loaders["train_eval"], device, None)
        validation_metrics = _epoch(model, loaders["validation"], device, None)
        row = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "validation_mse": validation_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "validation_mae": validation_metrics["mae"],
            "train_psnr_db": train_metrics["psnr_db"],
            "validation_psnr_db": validation_metrics["psnr_db"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_mse={row['train_mse']:.6f} "
            f"validation_mse={row['validation_mse']:.6f} "
            f"validation_psnr_db={row['validation_psnr_db']:.3f}",
            flush=True,
        )
        if validation_metrics["mse"] < best_validation_mse:
            best_validation_mse = validation_metrics["mse"]
            best_epoch = epoch
            _save_checkpoint(best_path, model, optimizer, epoch, row, metadata)

    _save_checkpoint(last_path, model, optimizer, epochs, history[-1], metadata)
    _save_history(history, output / "history.csv")
    _save_loss_curve(history, output / "loss_curve.png")
    best_model, best_payload = _load_model(best_path, device)
    test_metrics = _epoch(best_model, loaders["test"], device, None)
    reloaded_validation = _epoch(best_model, loaders["validation"], device, None)
    if not math.isclose(
        reloaded_validation["mse"], best_validation_mse, rel_tol=1e-6, abs_tol=1e-9
    ):
        raise RuntimeError("best checkpoint reload metric mismatch")
    _save_reconstructions(
        best_model,
        datasets["test"],
        device,
        output / "test_reconstructions.png",
    )
    summary = {
        **metadata,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_train_metrics": {
            key.removeprefix("train_"): value
            for key, value in history[best_epoch - 1].items()
            if key.startswith("train_")
        },
        "best_validation_metrics": {
            "mse": best_validation_mse,
            "mae": best_payload["metrics"]["validation_mae"],
            "psnr_db": best_payload["metrics"]["validation_psnr_db"],
        },
        "test_metrics": test_metrics,
        "generalization_gap_mse": test_metrics["mse"]
        - history[best_epoch - 1]["train_mse"],
        "artifacts": {
            "best_checkpoint": str(best_path.resolve()),
            "last_checkpoint": str(last_path.resolve()),
            "history": str((output / "history.csv").resolve()),
            "loss_curve": str((output / "loss_curve.png").resolve()),
            "test_reconstructions": str(
                (output / "test_reconstructions.png").resolve()
            ),
        },
        "checkpoint_reload_verified": True,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="uav_vision_dataset")
    parser.add_argument(
        "--split-file",
        default="uav_vision_dataset/_audit/autoencoder_split.json",
    )
    parser.add_argument("--output-dir", default=f"autoencoder_runs/rgb_ae_v0_{timestamp}")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dimension", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=614420090)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    train(
        dataset_root=args.dataset_root,
        split_file=args.split_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        latent_dimension=args.latent_dimension,
        device_name=args.device,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
