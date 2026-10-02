"""Train an FPV RGB, FPV depth, or TOP Autoencoder without ROS or Isaac."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # Keep --help usable before optional install.
    SummaryWriter = None  # type: ignore[assignment,misc]

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from uav_ml.datasets.expert_image_dataset import (
    ExpertImageDataset,
    IMAGE_PREPROCESSING,
    IMAGE_SOURCES,
    normalize_image_sources,
)
from uav_ml.datasets.rgb_episode_dataset import RgbEpisodeDataset
from uav_ml.models import (
    MultiImageAutoencoderConfig,
    MultiImageAutoencoderV0,
    RgbAutoencoderConfig,
    RgbAutoencoderV0,
)
from uav_ml.tools.bc_baseline import audit_dataset, create_episode_split
from uav_ml.tools.training_cli import (
    TensorBoardServer,
    add_tensorboard_arguments,
    autoencoder_latest_path,
    experiment_run_directory,
    publish_autoencoder_latest,
    resolve_dataset,
)
from uav_ml.tools.validate_expert_collection import DEFAULT_DATASET
from uav_ml.train_bc import resolve_device, set_seeds


LATENT_CONTRACT_TEMPLATE = "rgb_autoencoder_latent_{dimension}_v0.1"
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
    model: RgbAutoencoderV0 | MultiImageAutoencoderV0,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0
    source_metrics: dict[str, dict[str, float]] = {}
    for batch in loader:
        if isinstance(model, MultiImageAutoencoderV0):
            images = {
                source: batch["images"][source].to(device, non_blocking=True)
                for source in model.config.image_sources
            }
        else:
            images = batch["image"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            reconstruction, _ = model(images)
            if isinstance(model, MultiImageAutoencoderV0):
                losses = {
                    source: nn.functional.mse_loss(
                        reconstruction[source], images[source]
                    )
                    for source in model.config.image_sources
                }
                loss = torch.stack(list(losses.values())).mean()
            else:
                loss = nn.functional.mse_loss(reconstruction, images)
            if training:
                loss.backward()
                optimizer.step()
        if isinstance(model, MultiImageAutoencoderV0):
            for source in model.config.image_sources:
                error = reconstruction[source].detach() - images[source]
                values = source_metrics.setdefault(
                    source,
                    {"squared": 0.0, "absolute": 0.0, "count": 0.0},
                )
                values["squared"] += float(error.square().sum().cpu())
                values["absolute"] += float(error.abs().sum().cpu())
                values["count"] += error.numel()
        else:
            error = reconstruction.detach() - images
            squared_error += float(error.square().sum().cpu())
            absolute_error += float(error.abs().sum().cpu())
            element_count += error.numel()
    if isinstance(model, MultiImageAutoencoderV0):
        if not source_metrics:
            raise ValueError("empty Autoencoder data loader")
        per_source = {
            source: {
                "mse": values["squared"] / values["count"],
                "mae": values["absolute"] / values["count"],
            }
            for source, values in source_metrics.items()
        }
        mse = sum(value["mse"] for value in per_source.values()) / len(per_source)
        mae = sum(value["mae"] for value in per_source.values()) / len(per_source)
        return {
            "mse": mse,
            "mae": mae,
            "psnr_db": 10.0 * math.log10(1.0 / max(mse, 1e-12)),
            "per_source": per_source,
        }
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
    model: RgbAutoencoderV0 | MultiImageAutoencoderV0,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": (
                "multi_image_autoencoder_checkpoint_v0.1"
                if isinstance(model, MultiImageAutoencoderV0)
                else "rgb_autoencoder_checkpoint_v0.1"
            ),
            "model_class": type(model).__name__,
            "model_config": model.config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "metadata": metadata,
        },
        path,
    )


def _load_model(
    path: Path, device: torch.device
) -> tuple[RgbAutoencoderV0 | MultiImageAutoencoderV0, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_class") == "RgbAutoencoderV0":
        model = RgbAutoencoderV0(RgbAutoencoderConfig(**payload["model_config"]))
    elif payload.get("model_class") == "MultiImageAutoencoderV0":
        model = MultiImageAutoencoderV0(
            MultiImageAutoencoderConfig.from_dict(payload["model_config"])
        )
    else:
        raise ValueError("checkpoint model class is not a supported Autoencoder")
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


def _fixed_validation_images(
    dataset: Dataset, count: int = 8
) -> torch.Tensor | dict[str, torch.Tensor]:
    indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
    if "images" in dataset[int(indices[0])]:
        sources = tuple(dataset.image_sources)
        return {
            source: torch.stack([
                dataset[int(index)]["images"][source] for index in indices
            ])
            for source in sources
        }
    return torch.stack([dataset[int(index)]["image"] for index in indices])


def _write_tensorboard_reconstructions(
    writer: SummaryWriter,
    model: RgbAutoencoderV0 | MultiImageAutoencoderV0,
    images: torch.Tensor | dict[str, torch.Tensor],
    device: torch.device,
    image_source: str,
    epoch: int,
) -> None:
    model.eval()
    with torch.inference_mode():
        if isinstance(model, MultiImageAutoencoderV0):
            assert isinstance(images, dict)
            reconstructions, _ = model({
                source: image.to(device) for source, image in images.items()
            })
            for source, image in images.items():
                comparison = torch.cat(
                    (image.cpu(), reconstructions[source].cpu()), dim=3
                )
                writer.add_images(
                    f"ae/{source}/validation_original_vs_reconstructed",
                    comparison,
                    epoch,
                )
            return
        assert isinstance(images, torch.Tensor)
        reconstruction, _ = model(images.to(device))
    comparison = torch.cat((images.cpu(), reconstruction.cpu()), dim=3)
    writer.add_images(
        f"ae/{image_source}/validation_original_vs_reconstructed", comparison, epoch
    )


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
    model: RgbAutoencoderV0 | MultiImageAutoencoderV0,
    dataset: Dataset,
    device: torch.device,
    path: Path,
    count: int = 8,
) -> None:
    indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
    first = dataset[int(indices[0])]
    if isinstance(model, MultiImageAutoencoderV0):
        sources = model.config.image_sources
        images = {
            source: torch.stack([
                dataset[int(index)]["images"][source] for index in indices
            ]).to(device)
            for source in sources
        }
        with torch.inference_mode():
            reconstructions, _ = model(images)
        width = model.config.image_width
        height = model.config.image_height
        label_height = 20
        canvas = Image.new(
            "RGB",
            (width * len(indices), (height * 2 + label_height) * len(sources)),
            "white",
        )
        draw = ImageDraw.Draw(canvas)
        for source_index, source in enumerate(sources):
            top = source_index * (height * 2 + label_height)
            for column, (original, reconstruction) in enumerate(zip(
                images[source], reconstructions[source]
            )):
                x = column * width
                canvas.paste(_to_pil(original), (x, top + label_height))
                canvas.paste(
                    _to_pil(reconstruction), (x, top + height + label_height)
                )
                draw.text((x + 4, top + 3), f"{source} original", fill="black")
                draw.text(
                    (x + 4, top + height + label_height + 3),
                    f"{source} reconstruction",
                    fill="black",
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        return
    images = torch.stack([first["image"]] + [
        dataset[int(index)]["image"] for index in indices[1:]
    ]).to(device)
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
    split_file: str | Path | None,
    output_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    latent_dimension: int = 64,
    device_name: str = "auto",
    seed: int = 614420090,
    workers: int = 4,
    image_source: str = "fpv_rgb",
    image_log_interval: int = 5,
    dataset_name: str | None = None,
    tensorboard_enabled: bool = False,
    tensorboard_port: int = 6006,
    early_stopping_patience: int = 1000,
    min_epochs: int = 0,
    latest_index_path: Path | None = None,
    initialize_from: str | Path | None = None,
    image_sources: Sequence[str] | None = None,
    initialize_sources: Mapping[str, str | Path] | None = None,
) -> dict:
    """Train, select on validation MSE, then evaluate the held-out test split."""
    if SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard is required; run: python3 -m pip install -r "
            "requirements-ml.txt"
        )
    if (
        epochs < 1
        or batch_size < 1
        or learning_rate <= 0.0
        or workers < 0
        or image_log_interval < 1
        or early_stopping_patience < 1
        or min_epochs < 0
    ):
        raise ValueError("epochs/batch size/LR must be positive and workers nonnegative")
    if min_epochs > epochs:
        raise ValueError("min_epochs cannot exceed epochs")
    if latent_dimension < 2:
        raise ValueError("latent dimension must be at least 2")
    set_seeds(seed)
    device = resolve_device(device_name)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_dataset = Path(dataset_root).resolve()
    resolved_dataset_name = dataset_name or resolved_dataset.name

    sources = normalize_image_sources(
        image_sources if image_sources is not None else (image_source,)
    )
    source_label = sources[0] if len(sources) == 1 else "+".join(sources)
    multi_image = len(sources) > 1
    if split_file is not None and multi_image:
        raise ValueError(
            "multi-image training requires the formal episode-aligned dataset; "
            "omit --split-file"
        )
    if split_file is None:
        audit = audit_dataset(
            resolved_dataset,
            image_source=sources[0] if len(sources) == 1 else sources,
        )
        audit["dataset_name"] = resolved_dataset_name
        split_manifest = create_episode_split(audit, seed)
        if multi_image:
            datasets = {
                split: ExpertImageDataset(
                    dataset_root, split_manifest, split, image_sources=sources
                )
                for split in ("train", "validation", "test")
            }
        else:
            datasets = {
                split: ExpertImageDataset(
                    dataset_root, split_manifest, split, sources[0]
                )
                for split in ("train", "validation", "test")
            }
        split_path = None
    else:
        if sources != ("fpv_rgb",):
            raise ValueError(
                "legacy --split-file datasets only support fpv_rgb; omit "
                "--split-file for formal multi-source training"
            )
        datasets = {
            split: RgbEpisodeDataset(dataset_root, split_file, split)
            for split in ("train", "validation", "test")
        }
        split_manifest = datasets["train"].split_metadata
        split_path = str(Path(split_file).resolve())
        audit = {
            "dataset_name": resolved_dataset_name,
            "dataset_root": str(resolved_dataset),
            "legacy_split_file": split_path,
        }
    split_manifest = dict(split_manifest)
    split_manifest["dataset_name"] = resolved_dataset_name
    split_manifest["dataset_path"] = str(resolved_dataset)
    with (output / "dataset_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
    with (output / "split_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(split_manifest, stream, indent=2, sort_keys=True)
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
    if multi_image:
        model: RgbAutoencoderV0 | MultiImageAutoencoderV0 = (
            MultiImageAutoencoderV0(MultiImageAutoencoderConfig(
                image_sources=sources,
                latent_dimension=latent_dimension,
            ))
        ).to(device)
    else:
        model = RgbAutoencoderV0(
            RgbAutoencoderConfig(latent_dimension=latent_dimension)
        ).to(device)
    if initialize_from is not None:
        initial_checkpoint = Path(initialize_from).expanduser().resolve()
        payload = torch.load(
            initial_checkpoint, map_location="cpu", weights_only=False
        )
        initial_metadata = payload.get("metadata", {})
        if isinstance(model, MultiImageAutoencoderV0):
            if payload.get("model_class") != "MultiImageAutoencoderV0":
                raise ValueError(
                    "multi-image training initialization requires a multi-image "
                    "Autoencoder checkpoint"
                )
            initial_config = MultiImageAutoencoderConfig.from_dict(
                payload["model_config"]
            )
            if initial_config != model.config:
                raise ValueError(
                    "initial checkpoint sources or architecture do not match"
                )
            if initial_metadata.get("image_preprocessing") != {
                source: IMAGE_PREPROCESSING[source] for source in sources
            }:
                raise ValueError(
                    "initial checkpoint preprocessing does not match selected sources"
                )
            model.load_state_dict(payload["model_state"])
        else:
            if payload.get("model_class") != "RgbAutoencoderV0":
                raise ValueError("initial checkpoint model class is not RgbAutoencoderV0")
            initial_config = RgbAutoencoderConfig(**payload["model_config"])
            initial_source = initial_metadata.get("image_source")
            initial_preprocessing = initial_metadata.get("image_preprocessing")
            if initial_source != sources[0]:
                raise ValueError(
                    "initial checkpoint image source does not match training source"
                )
            if initial_preprocessing != IMAGE_PREPROCESSING[sources[0]]:
                raise ValueError(
                    "initial checkpoint preprocessing does not match training source"
                )
            if initial_config != model.config:
                raise ValueError(
                    "initial checkpoint architecture does not match requested model"
                )
            model.load_state_dict(payload["model_state"])
        model.to(device)
        print(f"Initialized Autoencoder weights from {initial_checkpoint}", flush=True)
    if initialize_sources:
        if not isinstance(model, MultiImageAutoencoderV0):
            raise ValueError("--initialize-source requires multiple image sources")
        for source, checkpoint_path in initialize_sources.items():
            if source not in sources:
                raise ValueError(f"initializer source {source!r} is not selected")
            source_checkpoint = Path(checkpoint_path).expanduser().resolve()
            payload = torch.load(
                source_checkpoint, map_location="cpu", weights_only=False
            )
            metadata = payload.get("metadata", {})
            if payload.get("model_class") != "RgbAutoencoderV0":
                raise ValueError(
                    f"initializer for {source} must be a single-source RgbAutoencoderV0"
                )
            if (
                metadata.get("image_source") != source
                or metadata.get("image_preprocessing") != IMAGE_PREPROCESSING[source]
            ):
                raise ValueError(f"initializer provenance mismatch for {source}")
            config = RgbAutoencoderConfig(**payload["model_config"])
            expected = RgbAutoencoderConfig(latent_dimension=latent_dimension)
            if config != expected:
                raise ValueError(f"initializer architecture mismatch for {source}")
            model.branches[source].load_state_dict(payload["model_state"])
            print(f"Initialized {source} branch from {source_checkpoint}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    fixed_validation = _fixed_validation_images(datasets["validation"])
    metadata = {
        "git_sha": _git_sha(),
        "input_contract": (
            f"{source_label}_128x72_float_v1.0"
        ),
        "latent_contract": LATENT_CONTRACT_TEMPLATE.format(
            dimension=latent_dimension
        ),
        "output_contract": f"{source_label}_reconstruction_128x72_float_v1.0",
        "dataset_name": resolved_dataset_name,
        "dataset_root": str(resolved_dataset),
        "split_file": split_path,
        "split_manifest": split_manifest,
        "split_seed": split_manifest["seed"],
        "split_unit": "episode",
        "camera": "fpv" if sources == ("fpv_rgb",) else source_label,
        "image_source": source_label,
        "image_sources": list(sources),
        "image_preprocessing": (
            IMAGE_PREPROCESSING[sources[0]] if len(sources) == 1 else
            {source: IMAGE_PREPROCESSING[source] for source in sources}
        ),
        "encoder_architecture": type(model).__name__,
        "latent_dimension": latent_dimension,
        "policy_rollouts": "excluded",
        "data_warning": (
            DATA_WARNING if split_file is not None else
            "Formal expert dataset; auxiliary sources remain action-aligned."
        ),
        "samples": {split: len(dataset) for split, dataset in datasets.items()},
        "episodes": {split: len(dataset.episode_ids) for split, dataset in datasets.items()},
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "parameter_count": model.parameter_count,
        "optimizer": {"name": "Adam", "learning_rate": learning_rate},
        "batch_size": batch_size,
        "tensorboard_enabled": tensorboard_enabled,
        "tensorboard_port": tensorboard_port,
        "early_stopping_patience": early_stopping_patience,
        "min_epochs": min_epochs,
        "initialized_from": (
            str(Path(initialize_from).expanduser().resolve())
            if initialize_from is not None else None
        ),
        "initialized_source_branches": {
            source: str(Path(path).expanduser().resolve())
            for source, path in (initialize_sources or {}).items()
        },
    }
    training_config = {
        "dataset_name": resolved_dataset_name,
        "dataset_path": str(resolved_dataset),
        "image_source": source_label,
        "image_sources": list(sources),
        "maximum_epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "latent_dimension": latent_dimension,
        "seed": seed,
        "tensorboard_enabled": tensorboard_enabled,
        "tensorboard_port": tensorboard_port,
        "early_stopping_patience": early_stopping_patience,
        "min_epochs": min_epochs,
        "initialized_from": (
            str(Path(initialize_from).expanduser().resolve())
            if initialize_from is not None else None
        ),
    }
    with (output / "training_config.json").open("w", encoding="utf-8") as stream:
        json.dump(training_config, stream, indent=2, sort_keys=True)
    print(
        "========== AutoEncoder Training ==========\n\n"
        f"Dataset:\n{resolved_dataset_name}\n\n"
        f"Resolved path:\n{resolved_dataset}\n\n"
        f"Image sources:\n{', '.join(sources)}\n\n"
        f"Usable episodes:\n{sum(len(value) for value in split_manifest['splits'].values())}\n\n"
        "Train / Validation / Test:\n"
        f"{len(datasets['train'].episode_ids)} / "
        f"{len(datasets['validation'].episode_ids)} / "
        f"{len(datasets['test'].episode_ids)}\n\n"
        f"Maximum epochs:\n{epochs}\n\n"
        f"Batch size:\n{batch_size}\n\n"
        f"Learning rate:\n{learning_rate}\n\n"
        f"Early stopping patience:\n{early_stopping_patience}\n\n"
        f"Minimum epochs:\n{min_epochs}\n\n"
        f"Output:\n{output.resolve()}\n\n"
        f"TensorBoard:\n{'enabled' if tensorboard_enabled else 'disabled'}\n\n"
        f"TensorBoard port:\n{tensorboard_port}\n\n"
        f"Reconstruction sources:\n{', '.join(sources)}\n\n"
        "==========================================",
        flush=True,
    )
    best_path = output / "best.pt"
    last_path = output / "last.pt"
    best_validation_mse = math.inf
    best_epoch = 0
    stale_epochs = 0
    early_stopping_triggered = False
    history: list[dict] = []
    writer = SummaryWriter(log_dir=str(output / "tensorboard"))
    try:
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
            if "per_source" in train_metrics:
                for source in sources:
                    row[f"train_{source}_mse"] = (
                        train_metrics["per_source"][source]["mse"]
                    )
                    row[f"validation_{source}_mse"] = (
                        validation_metrics["per_source"][source]["mse"]
                    )
            history.append(row)
            writer.add_scalar(
                "ae/train_reconstruction_loss", row["train_mse"], epoch
            )
            writer.add_scalar(
                "ae/validation_reconstruction_loss",
                row["validation_mse"],
                epoch,
            )
            for source in sources:
                if "per_source" in train_metrics:
                    writer.add_scalar(
                        f"ae/{source}/train_reconstruction_loss",
                        train_metrics["per_source"][source]["mse"],
                        epoch,
                    )
                    writer.add_scalar(
                        f"ae/{source}/validation_reconstruction_loss",
                        validation_metrics["per_source"][source]["mse"],
                        epoch,
                    )
            if validation_metrics["mse"] < best_validation_mse:
                best_validation_mse = validation_metrics["mse"]
                best_epoch = epoch
                stale_epochs = 0
                _save_checkpoint(best_path, model, optimizer, epoch, row, metadata)
            else:
                stale_epochs += 1
            should_stop = (
                epoch >= min_epochs
                and stale_epochs >= early_stopping_patience
            )
            print(
                f"Epoch {epoch}/{epochs} "
                f"train_loss={row['train_mse']:.6f} "
                f"val_loss={row['validation_mse']:.6f} "
                f"best_val={best_validation_mse:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.1e}",
                flush=True,
            )
            if (
                epoch == 1
                or epoch % image_log_interval == 0
                or epoch == epochs
                or should_stop
            ):
                _write_tensorboard_reconstructions(
                    writer,
                    model,
                    fixed_validation,
                    device,
                    source_label,
                    epoch,
                )
            if should_stop:
                early_stopping_triggered = True
                break
    except BaseException:
        writer.flush()
        writer.close()
        raise

    actual_epochs_trained = len(history)
    try:
        _save_checkpoint(
            last_path,
            model,
            optimizer,
            actual_epochs_trained,
            history[-1],
            metadata,
        )
        _save_history(history, output / "history.csv")
        _save_loss_curve(history, output / "reconstruction_loss_curves.png")
        best_model, best_payload = _load_model(best_path, device)
        test_metrics = _epoch(best_model, loaders["test"], device, None)
        reloaded_validation = _epoch(
            best_model, loaders["validation"], device, None
        )
        if not math.isclose(
            reloaded_validation["mse"],
            best_validation_mse,
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise RuntimeError("best checkpoint reload metric mismatch")
        _save_reconstructions(
            best_model,
            datasets["test"],
            device,
            output / "test_reconstructions.png",
        )
    except BaseException:
        writer.flush()
        writer.close()
        raise
    writer.flush()
    writer.close()
    summary = {
        **metadata,
        "run_status": "completed",
        "epochs": epochs,
        "maximum_epochs_requested": epochs,
        "actual_epochs_trained": actual_epochs_trained,
        "best_epoch": best_epoch,
        "early_stopping_triggered": early_stopping_triggered,
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
            "loss_curve": str((output / "reconstruction_loss_curves.png").resolve()),
            "reconstruction_loss_curves": str(
                (output / "reconstruction_loss_curves.png").resolve()
            ),
            "test_reconstructions": str(
                (output / "test_reconstructions.png").resolve()
            ),
            "tensorboard": str((output / "tensorboard").resolve()),
            "summary": str((output / "summary.json").resolve()),
            "latest_index": (
                str(latest_index_path.resolve())
                if latest_index_path is not None else None
            ),
        },
        "checkpoint_reload_verified": True,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    training_config.update({
        "actual_epochs_trained": actual_epochs_trained,
        "best_epoch": best_epoch,
        "early_stopping_triggered": early_stopping_triggered,
    })
    with (output / "training_config.json").open("w", encoding="utf-8") as stream:
        json.dump(training_config, stream, indent=2, sort_keys=True)
    if latest_index_path is not None:
        publish_autoencoder_latest(latest_index_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        help="dataset name under artifacts/datasets, or explicit path",
    )
    dataset_group.add_argument(
        "--dataset-root",
        help="deprecated explicit-path alias for backward compatibility",
    )
    parser.add_argument(
        "--split-file",
        default=None,
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help=(
            "initialize weights from a source-compatible Autoencoder checkpoint; "
            "optimizer and epoch count start fresh"
        ),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--latent-dimension", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=614420090)
    parser.add_argument("--workers", type=int, default=4)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--image-source", choices=IMAGE_SOURCES, default="top"
    )
    source_group.add_argument(
        "--image-sources", choices=IMAGE_SOURCES, nargs="+",
        help="one or more aligned image streams for joint Autoencoder training",
    )
    parser.add_argument(
        "--initialize-source", action="append", default=[], metavar="SOURCE=PATH",
        help="initialize a multi-image branch from a single-source AE checkpoint; repeatable",
    )
    parser.add_argument("--image-log-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="do not apply early stopping before this epoch",
    )
    add_tensorboard_arguments(parser)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    try:
        if args.dataset_root is not None:
            location = resolve_dataset(
                args.dataset_root,
                must_exist=True,
                short_name=False,
                project_root=repository_root,
            )
        else:
            location = resolve_dataset(
                args.dataset or DEFAULT_DATASET,
                must_exist=True,
                project_root=repository_root,
            )
    except (OSError, ValueError) as error:
        print(f"ERROR: Autoencoder training stopped: {error}")
        return 1
    timestamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%dT%H%M")
    sources = normalize_image_sources(
        args.image_sources if args.image_sources else (args.image_source,)
    )
    source_label = sources[0] if len(sources) == 1 else "+".join(sources)
    initialize_sources: dict[str, Path] = {}
    try:
        for entry in args.initialize_source:
            source, separator, checkpoint = entry.partition("=")
            if not separator or not source or not checkpoint:
                raise ValueError(
                    "--initialize-source must use SOURCE=PATH"
                )
            if source in initialize_sources:
                raise ValueError(f"duplicate initializer for {source}")
            initialize_sources[source] = Path(checkpoint)
    except ValueError as error:
        print(f"ERROR: Autoencoder training stopped: {error}")
        return 1
    output = (
        Path(args.output_dir)
        if args.output_dir
        else experiment_run_directory(
            "autoencoder",
            location,
            source_label,
            timestamp,
            project_root=repository_root,
        )
    )
    latest_index = autoencoder_latest_path(
        location, source_label, project_root=repository_root
    )
    try:
        with TensorBoardServer(
            output / "tensorboard",
            enabled=args.tensorboard,
            port=args.tensorboard_port,
        ) as tensorboard_server:
            summary = train(
                dataset_root=location.path,
                split_file=args.split_file,
                output_dir=output,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                latent_dimension=args.latent_dimension,
                device_name=args.device,
                seed=args.seed,
                workers=args.workers,
                image_source=sources[0],
                image_sources=(sources if len(sources) > 1 else None),
                image_log_interval=args.image_log_interval,
                dataset_name=location.name,
                tensorboard_enabled=args.tensorboard,
                tensorboard_port=args.tensorboard_port,
                early_stopping_patience=args.patience,
                min_epochs=args.min_epochs,
                latest_index_path=latest_index,
                initialize_from=args.initialize_from,
                initialize_sources=initialize_sources,
            )
            print(
                "=========================================\n"
                "AutoEncoder training completed successfully\n\n"
                f"Dataset:\n{location.name}\n\n"
                f"Image sources:\n{', '.join(sources)}\n\n"
                f"Maximum epochs requested:\n"
                f"{summary['maximum_epochs_requested']}\n\n"
                f"Actual epochs trained:\n"
                f"{summary['actual_epochs_trained']}\n\n"
                f"Best epoch:\n{summary['best_epoch']}\n\n"
                f"Early stopping:\n"
                f"{'yes' if summary['early_stopping_triggered'] else 'no'}\n\n"
                "Best validation reconstruction loss:\n"
                f"{summary['best_validation_metrics']['mse']:.8f}\n\n"
                "Final test reconstruction loss:\n"
                f"{summary['test_metrics']['mse']:.8f}\n\n"
                f"Best checkpoint:\n"
                f"{summary['artifacts']['best_checkpoint']}\n\n"
                f"TensorBoard:\n"
                f"{tensorboard_server.url if tensorboard_server.active else 'not managed'}\n"
                "=========================================",
                flush=True,
            )
            tensorboard_server.wait_until_interrupted()
    except KeyboardInterrupt:
        print("Autoencoder training interrupted; TensorBoard was cleaned up.")
        return 130
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: Autoencoder training stopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
