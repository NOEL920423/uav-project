"""Train and offline-evaluate the formal frozen-encoder BC baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # Keep --help usable before optional install.
    SummaryWriter = None  # type: ignore[assignment,misc]

from uav_ml.datasets.expert_image_dataset import (
    ExpertImageDataset,
    IMAGE_PREPROCESSING,
    IMAGE_SOURCES,
    normalize_image_sources,
    preprocess_expert_image,
    select_episode_image_sets,
    select_episode_images,
)
from uav_ml.models import (
    LatentBcPolicy,
    LatentBcPolicyConfig,
    MultiImageAutoencoderV0,
    RgbAutoencoderV0,
)
from uav_ml.inference.rgb_encoder import (
    ResNet18Encoder, load_frozen_encoder,
)
from uav_ml.tools.validate_expert_collection import (
    DEFAULT_DATASET,
)
from uav_ml.tools.training_cli import (
    TensorBoardServer,
    add_episode_split_arguments,
    add_tensorboard_arguments,
    experiment_run_directory,
    resolve_dataset,
    select_autoencoder_checkpoint,
    select_multi_autoencoder_checkpoint,
)
from uav_ml.training import resolve_device, set_seeds

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FORMAT_VERSION = "bc_baseline_v1.0"
ACTION_NAMES = ("v_forward", "v_right", "yaw_rate")
PHYSICAL_ACTION_LIMITS = {
    "v_forward_mps": 1.0,
    "v_right_mps": 0.8,
    "yaw_rate_radps": 1.0,
}
STATE_FIELDS = (
    "body_velocity_forward_mps",
    "body_velocity_right_mps",
    "goal_direction_forward",
    "goal_direction_right",
    "normalized_goal_distance",
    "previous_action_forward",
    "previous_action_right",
    "previous_action_yaw_rate",
)
TARGET_FIELDS = (
    "expert_action_forward",
    "expert_action_right",
    "expert_action_yaw_rate",
)
DEFAULT_SEED = 614420090


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stamp() -> str:
    return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%dT%H%M")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _publish_latest_bc(repository_root: Path, summary: dict) -> None:
    """Atomically publish a completed CLI training run as the global default."""
    if summary.get("run_status") != "completed":
        raise ValueError("refusing to publish an incomplete BC run")
    checkpoint = Path(summary["best_checkpoint"]).resolve()
    if not checkpoint.is_file() or not (checkpoint.parent / "summary.json").is_file():
        raise FileNotFoundError(f"completed BC artifacts are missing: {checkpoint}")
    pointer = repository_root / "experiments/bc/latest.json"
    temporary = pointer.with_suffix(".json.tmp")
    _write_json(temporary, {
        "format_version": FORMAT_VERSION,
        "updated_utc": _utc_now(),
        "run_dir": str(checkpoint.parent),
        "best_checkpoint": str(checkpoint),
    })
    temporary.replace(pointer)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@dataclass(frozen=True)
class TrainingConfig:
    """Serializable reproducible baseline hyperparameters."""

    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    early_stopping_patience: int = 1000
    min_epochs: int = 0
    seed: int = DEFAULT_SEED
    episodes: int | None = None
    train_episodes: int | None = None
    validation_episodes: int | None = None
    test_episodes: int | None = None


def _validate_sample_row(dataset_root: Path, row: dict) -> None:
    values = [float(row[name]) for name in STATE_FIELDS + TARGET_FIELDS]
    values.extend((
        float(row["image_timestamp_s"]),
        float(row["state_timestamp_s"]),
        float(row["expert_action_timestamp_s"]),
        float(row["state_image_error_s"]),
        float(row["expert_action_image_error_s"]),
    ))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("sample contains NaN or Inf")
    if any(abs(float(row[name])) > 1.000001 for name in TARGET_FIELDS):
        raise ValueError("normalized expert action is outside [-1,1]")
    if (
        float(row["state_image_error_s"]) > 0.1
        or float(row["expert_action_image_error_s"]) > 0.1
    ):
        raise ValueError("sample synchronization exceeds 0.100 s")
    image_path = dataset_root / row["image_path"]
    if not image_path.is_file():
        raise FileNotFoundError(f"FPV image is missing: {image_path}")
    with Image.open(image_path) as image:
        image.verify()


def audit_dataset(
    dataset_root: Path,
    encoder_checkpoint: Path | None = None,
    image_source: str | tuple[str, ...] | list[str] = "fpv_rgb",
) -> dict:
    """Select only complete, validated, readable successful episodes."""
    dataset_root = dataset_root.resolve()
    image_sources = normalize_image_sources(
        (image_source,) if isinstance(image_source, str) else image_source
    )
    source_label = image_sources[0] if len(image_sources) == 1 else "+".join(image_sources)
    encoder_checkpoint = (
        encoder_checkpoint.resolve() if encoder_checkpoint is not None else None
    )
    if encoder_checkpoint is not None and not encoder_checkpoint.is_file():
        raise FileNotFoundError(
            f"frozen encoder checkpoint is missing: {encoder_checkpoint}"
        )
    collection_path = dataset_root / "collection_manifest.json"
    validation_path = dataset_root / "collection_validation.json"
    if not collection_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("formal collection validation artifacts missing")
    collection = _read_json(collection_path)
    aggregate = _read_json(validation_path)
    if collection.get("status") != "complete" or not aggregate.get("valid"):
        raise ValueError("formal expert dataset validator has not passed")
    entries = collection.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("collection manifest contains no episodes")
    encoder_hash = _sha256(encoder_checkpoint) if encoder_checkpoint else None
    # When the dataset contains the formal one-to-one TOP stream, all image
    # sources use that same cohort. This keeps FPV/TOP/depth comparisons on an
    # identical episode universe while explicitly excluding legacy sparse TOP.
    formal_top_episodes: set[str] = set()
    for entry in entries:
        episode_id = str(entry.get("episode_id", ""))
        episode_path = dataset_root / episode_id / "episode.json"
        if not episode_path.is_file():
            continue
        episode = _read_json(episode_path)
        runtime = (
            episode.get("available_sensor_streams", {}).get(
                "runtime_status", {}
            )
        )
        if runtime.get("observer_mode") == "fixed_global_top":
            formal_top_episodes.add(episode_id)
    comparison_cohort_enabled = bool(formal_top_episodes)
    usable: list[dict] = []
    excluded: list[dict] = []
    reasons: Counter[str] = Counter()
    for entry in entries:
        episode_id = str(entry.get("episode_id", ""))
        if entry.get("status") != "complete" or entry.get("success") is not True:
            detail = str(entry.get("terminal_reason", "")).strip()
            reason = (
                f"episode_not_successful:{detail}"
                if detail else "episode_not_successful"
            )
        else:
            reason = ""
            if comparison_cohort_enabled and episode_id not in formal_top_episodes:
                reason = "outside_formal_image_comparison_cohort"
        if not reason:
            try:
                episode_path = dataset_root / episode_id / "episode.json"
                per_validation_path = (
                    dataset_root / episode_id / "validation.json"
                )
                samples_path = dataset_root / episode_id / "samples.csv"
                episode = _read_json(episode_path)
                validation = _read_json(per_validation_path)
                if (
                    episode.get("status") != "complete"
                    or episode.get("success") is not True
                    or validation.get("valid") is not True
                    or validation.get("episode_success") is not True
                ):
                    raise ValueError("success/validation contract failed")
                with samples_path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                if not rows or len(rows) != int(validation.get("sample_count", -1)):
                    raise ValueError("accepted sample count is inconsistent")
                for row in rows:
                    _validate_sample_row(dataset_root, row)
                # Require every selected stream on every sample; multi-image
                # training never falls back to a different camera or modality.
                if len(image_sources) == 1:
                    selected = select_episode_images(
                        dataset_root, episode_id, image_sources[0]
                    )
                    if len(selected) != len(rows):
                        raise ValueError("selected image count differs from samples")
                else:
                    selected = select_episode_image_sets(
                        dataset_root, episode_id, image_sources
                    )
                    if len(selected) != len(rows):
                        raise ValueError("selected image bundle count differs from samples")
            except (OSError, KeyError, TypeError, ValueError) as error:
                if image_sources != ("fpv_rgb",):
                    raise ValueError(
                        f"{source_label} association failed loudly for "
                        f"{episode_id}: {error}"
                    ) from error
                reason = f"invalid_or_corrupt:{error}"
            else:
                usable.append({
                    "episode_id": episode_id,
                    "seed": int(entry["seed"]),
                    "samples": len(rows),
                })
        if reason:
            reason_key = (
                reason if reason.startswith("episode_not_successful:") else
                reason.split(":", 1)[0]
            )
            reasons[reason_key] += 1
            excluded.append({
                "episode_id": episode_id,
                "seed": entry.get("seed"),
                "reason": reason,
            })
    if len(usable) < 3:
        raise ValueError("at least three usable episodes are required")
    return {
        "format_version": FORMAT_VERSION,
        "created_utc": _utc_now(),
        "dataset_root": str(dataset_root),
        "image_source": source_label,
        "image_sources": list(image_sources),
        "image_preprocessing": (
            IMAGE_PREPROCESSING[image_sources[0]] if len(image_sources) == 1 else
            {source: IMAGE_PREPROCESSING[source] for source in image_sources}
        ),
        "comparison_cohort": (
            "fixed_global_top_complete" if comparison_cohort_enabled else
            "legacy_dataset_eligibility"
        ),
        "collection_manifest": str(collection_path.resolve()),
        "collection_manifest_sha256": _sha256(collection_path),
        "collection_validation": str(validation_path.resolve()),
        "collection_validation_sha256": _sha256(validation_path),
        "encoder_checkpoint": (
            str(encoder_checkpoint) if encoder_checkpoint is not None else None
        ),
        "encoder_sha256": encoder_hash,
        "total_episodes": len(entries),
        "usable_episodes": len(usable),
        "excluded_episodes": len(excluded),
        "exclusion_reasons": dict(sorted(reasons.items())),
        "total_accepted_samples": sum(item["samples"] for item in usable),
        "usable": usable,
        "excluded": excluded,
    }


def create_episode_split(
    audit: dict,
    seed: int,
    *,
    episodes: int | None = None,
    train_episodes: int | None = None,
    validation_episodes: int | None = None,
    test_episodes: int | None = None,
) -> dict:
    """Select complete episodes and split deterministically without frame leakage."""
    requested_counts = (train_episodes, validation_episodes, test_episodes)
    explicit_counts = any(count is not None for count in requested_counts)
    if explicit_counts and any(count is None for count in requested_counts):
        raise ValueError(
            "provide --train-episodes, --validation-episodes and --test-episodes together"
        )
    for name, count in zip(
        ("episodes", "train-episodes", "validation-episodes", "test-episodes"),
        (episodes, *requested_counts),
    ):
        if count is not None and (type(count) is not int or count <= 0):
            raise ValueError(f"--{name} must be a positive integer")
    if explicit_counts:
        selected_count = sum(requested_counts)
        if episodes is not None and episodes != selected_count:
            raise ValueError("--episodes must equal the sum of the three split counts")
    else:
        selected_count = len(audit["usable"]) if episodes is None else episodes
    if selected_count > len(audit["usable"]):
        raise ValueError(
            f"requested {selected_count} episodes but only "
            f"{len(audit['usable'])} usable episodes are available"
        )
    if selected_count < 3:
        raise ValueError("at least three selected episodes are required")

    episodes = [item["episode_id"] for item in audit["usable"]]
    random.Random(seed).shuffle(episodes)
    unused_episodes = sorted(episodes[selected_count:])
    episodes = episodes[:selected_count]
    if explicit_counts:
        train_count, validation_count, test_count = requested_counts
    else:
        validation_count = max(1, round(len(episodes) * 0.10))
        test_count = max(1, round(len(episodes) * 0.10))
        train_count = len(episodes) - validation_count - test_count
    if train_count < 1:
        raise ValueError("dataset is too small for train/validation/test split")
    splits = {
        "train": sorted(episodes[:train_count]),
        "validation": sorted(
            episodes[train_count:train_count + validation_count]
        ),
        "test": sorted(episodes[train_count + validation_count:]),
    }
    members = [episode for values in splits.values() for episode in values]
    if len(members) != len(set(members)) or set(members) != set(episodes):
        raise RuntimeError("episode split leakage or omission detected")
    samples = {item["episode_id"]: item["samples"] for item in audit["usable"]}
    return {
        "format_version": "bc_episode_split_v1.0",
        "seed": seed,
        "strategy": (
            "episode_level_explicit_counts" if explicit_counts else "episode_level_80_10_10"
        ),
        "available_episode_count": len(audit["usable"]),
        "selected_episode_count": selected_count,
        "unused_episode_ids": unused_episodes,
        "splits": splits,
        "episode_counts": {
            name: len(values) for name, values in splits.items()
        },
        "sample_counts": {
            name: sum(samples[item] for item in values)
            for name, values in splits.items()
        },
        "excluded": audit["excluded"],
    }


def _load_frozen_encoder(
    path: Path, device: torch.device
) -> tuple[RgbAutoencoderV0 | MultiImageAutoencoderV0 | ResNet18Encoder, dict]:
    return load_frozen_encoder(path, device)


def _episode_arrays(
    dataset_root: Path,
    episode_id: str,
    encoder: RgbAutoencoderV0 | MultiImageAutoencoderV0 | ResNet18Encoder,
    device: torch.device,
    encode_batch_size: int,
    image_source: str = "fpv_rgb",
    image_sources: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with (dataset_root / episode_id / "samples.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    states = np.asarray(
        [[float(row[name]) for name in STATE_FIELDS] for row in rows],
        dtype=np.float32,
    )
    targets = np.asarray(
        [[float(row[name]) for name in TARGET_FIELDS] for row in rows],
        dtype=np.float32,
    )
    sources = normalize_image_sources(
        image_sources if image_sources is not None else (image_source,)
    )
    selected = (
        select_episode_images(dataset_root, episode_id, sources[0])
        if len(sources) == 1 else
        select_episode_image_sets(dataset_root, episode_id, sources)
    )
    if len(selected) != len(rows):
        raise ValueError(f"{episode_id}: selected image/sample count mismatch")
    latents = []
    for start in range(0, len(rows), encode_batch_size):
        if len(sources) > 1:
            batch = {
                source: torch.stack([
                    preprocess_expert_image(
                        item["images"][source],
                        source,
                        image_width=encoder.config.image_width,
                        image_height=encoder.config.image_height,
                    )
                    for item in selected[start:start + encode_batch_size]
                ]).to(device)
                for source in sources
            }
        else:
            batch = torch.stack([
                preprocess_expert_image(
                    item["image_path"],
                    sources[0],
                    image_width=encoder.config.image_width,
                    image_height=encoder.config.image_height,
                )
                for item in selected[start:start + encode_batch_size]
            ]).to(device)
        with torch.inference_mode():
            latents.append(encoder.encode(batch).cpu().numpy())
    latent = np.concatenate(latents).astype(np.float32)
    observation = np.concatenate((latent, states), axis=1).astype(np.float32)
    if observation.shape[1] != encoder.config.latent_dimension + 8 or targets.shape[1] != 3:
        raise ValueError("rebuilt BC tensors violate the latent+state8/3D contract")
    return observation, targets


def encode_splits(
    dataset_root: Path,
    split_manifest: dict,
    encoder: RgbAutoencoderV0 | MultiImageAutoencoderV0 | ResNet18Encoder,
    device: torch.device,
    encode_batch_size: int,
    image_source: str = "fpv_rgb",
    image_sources: tuple[str, ...] | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Encode selected images once and return tensors by episode split."""
    result = {}
    for split, episode_ids in split_manifest["splits"].items():
        observations = []
        targets = []
        for episode_id in episode_ids:
            observation, target = _episode_arrays(
                dataset_root,
                episode_id,
                encoder,
                device,
                encode_batch_size,
                image_source,
                image_sources,
            )
            observations.append(observation)
            targets.append(target)
        result[split] = (
            torch.from_numpy(np.concatenate(observations)),
            torch.from_numpy(np.concatenate(targets)),
        )
    return result


def action_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute equal-component normalized-action errors."""
    error = prediction.double() - target.double()
    mse = error.square().mean(dim=0)
    mae = error.abs().mean(dim=0)
    rmse = mse.sqrt()
    return {
        "mse": float(mse.mean()),
        "per_action": {
            name: {
                "mse": float(mse[index]),
                "mae": float(mae[index]),
                "rmse": float(rmse[index]),
            }
            for index, name in enumerate(ACTION_NAMES)
        },
    }


def _run_loader(
    model: LatentBcPolicy,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    encoder: ResNet18Encoder | None = None,
    observation_mean: torch.Tensor | None = None,
    observation_std: torch.Tensor | None = None,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    if encoder is not None:
        encoder.train(training)
        if observation_mean is None or observation_std is None:
            raise ValueError("image training requires observation normalization")
    parameters = list(model.parameters())
    if encoder is not None:
        parameters.extend(encoder.parameters())
    predictions = []
    targets = []
    for batch in loader:
        if encoder is None:
            observation, target = (value.to(device) for value in batch)
        else:
            values = batch["sample_values"].to(device)
            state, target = values[:, :8], values[:, 8:]
            images = (
                {source: image.to(device) for source, image in batch["images"].items()}
                if "images" in batch else batch["image"].to(device)
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if encoder is not None:
                observation = (
                    torch.cat((encoder.encode(images), state), dim=1) - observation_mean
                ) / observation_std
            prediction = model(observation)
            loss = nn.functional.mse_loss(prediction, target)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
        predictions.append(prediction.detach().cpu())
        targets.append(target.detach().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return action_metrics(prediction, target), prediction, target


def _checkpoint(
    model: LatentBcPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: dict,
    observation_mean: np.ndarray,
    observation_std: np.ndarray,
    config: TrainingConfig,
    audit: dict,
    split_manifest: dict,
    image_source: str,
    encoder: RgbAutoencoderV0 | MultiImageAutoencoderV0 | ResNet18Encoder,
) -> dict:
    image_sources = list(getattr(encoder.config, "image_sources", (image_source,)))
    image_preprocessing = (
        {source: IMAGE_PREPROCESSING[source] for source in image_sources}
        if len(image_sources) > 1 else IMAGE_PREPROCESSING[image_sources[0]]
    )
    return {
        "format_version": FORMAT_VERSION,
        "model_class": "LatentBcPolicy",
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "training_config": asdict(config),
        # Tensors avoid NumPy 2.x pickle module names that Isaac Sim's bundled
        # NumPy 1.x cannot import. The evaluator still supports earlier runs.
        "observation_mean": torch.from_numpy(observation_mean.copy()),
        "observation_std": torch.from_numpy(observation_std.copy()),
        "dataset_root": audit["dataset_root"],
        "dataset_manifest": audit["collection_manifest"],
        "dataset_manifest_sha256": audit["collection_manifest_sha256"],
        "split_manifest": split_manifest,
        "autoencoder_checkpoint": audit["encoder_checkpoint"],
        "autoencoder_checkpoint_sha256": audit["encoder_sha256"],
        "encoder_architecture": type(encoder).__name__,
        "encoder_frozen": True,
        "image_source": image_source,
        "image_sources": image_sources,
        "image_preprocessing": image_preprocessing,
        "encoder_preprocessing": (
            encoder.preprocessing if isinstance(encoder, ResNet18Encoder)
            else image_preprocessing
        ),
        "latent_dimension": encoder.config.latent_dimension,
        "observation_contract": f"latent{encoder.config.latent_dimension}_plus_body_state8_v1.0",
        "action_contract": "normalized_body_forward_right_yaw_v1.0",
        "physical_action_limits": PHYSICAL_ACTION_LIMITS,
        "loss": "equal_component_normalized_action_mse",
    }


def _history_row(epoch: int, train: dict, validation: dict) -> dict:
    row = {
        "epoch": epoch,
        "train_mse": train["mse"],
        "validation_mse": validation["mse"],
    }
    for split, metrics in (("train", train), ("validation", validation)):
        for action, values in metrics["per_action"].items():
            for metric, value in values.items():
                row[f"{split}_{action}_{metric}"] = value
    return row


def _progress(
    epoch: int,
    epochs: int,
    train_loss: float,
    validation_loss: float,
    best_loss: float,
    learning_rate: float,
    started: float,
    stage: str = "BC",
) -> None:
    elapsed = time.monotonic() - started
    eta = elapsed / epoch * (epochs - epoch) if epoch else None
    filled = int(30 * epoch / epochs)
    bar = "█" * filled + "-" * (30 - filled)
    print(
        f"\n{stage} Baseline Training\n\n"
        f"Epoch {epoch} / {epochs} [{bar}] {100 * epoch // epochs}%\n\n"
        f"Train loss : {train_loss:.6f}\n"
        f"Val loss   : {validation_loss:.6f}\n"
        f"Best val   : {best_loss:.6f}\n"
        f"LR         : {learning_rate:.1e}\n\n"
        f"Elapsed : {_format_duration(elapsed)}\n"
        f"ETA     : {_format_duration(eta)}\n",
        flush=True,
    )


def _write_history(path: Path, history: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def create_training_plots(
    output_dir: Path,
    history: list[dict],
    test_metrics: dict,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> list[str]:
    """Create the three required plots from recorded metrics only."""
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["train_mse"] for row in history], label="Train")
    axis.plot(
        epochs,
        [row["validation_mse"] for row in history],
        label="Validation",
    )
    axis.set(xlabel="Epoch", ylabel="Normalized action MSE")
    axis.set_title("BC baseline loss")
    axis.grid(alpha=0.3)
    axis.legend()
    loss_path = plots / "loss_curves.png"
    figure.tight_layout()
    figure.savefig(loss_path, dpi=160)
    plt.close(figure)

    values = [
        test_metrics["per_action"][name]["rmse"] for name in ACTION_NAMES
    ]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(ACTION_NAMES, values)
    axis.set(ylabel="RMSE", title="Held-out test action RMSE")
    axis.grid(axis="y", alpha=0.3)
    action_path = plots / "per_action_rmse.png"
    figure.tight_layout()
    figure.savefig(action_path, dpi=160)
    plt.close(figure)

    predicted = prediction.numpy()
    expert = target.numpy()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for index, (axis, name) in enumerate(zip(axes, ACTION_NAMES)):
        axis.scatter(expert[:, index], predicted[:, index], s=8, alpha=0.35)
        axis.plot((-1, 1), (-1, 1), "k--", linewidth=1)
        axis.set(
            xlim=(-1.05, 1.05),
            ylim=(-1.05, 1.05),
            xlabel="Expert",
            ylabel="Predicted",
            title=name,
        )
        axis.grid(alpha=0.2)
    comparison_path = plots / "expert_vs_predicted.png"
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=160)
    plt.close(figure)
    return [
        str(loss_path.resolve()),
        str(action_path.resolve()),
        str(comparison_path.resolve()),
    ]


def train_baseline(
    dataset_root: Path,
    encoder_checkpoint: Path | None,
    output_dir: Path,
    config: TrainingConfig,
    device_name: str,
    encode_batch_size: int = 128,
    image_source: str = "fpv_rgb",
    dataset_name: str | None = None,
    tensorboard_enabled: bool = False,
    tensorboard_port: int = 6006,
    encoder_selection: str = "explicit",
    encoder_type: str = "autoencoder",
    image_sources: tuple[str, ...] | None = None,
    initialize_policy: Path | None = None,
) -> dict:
    """Run reproducible BC training and held-out offline evaluation."""
    training_started_s = time.monotonic()
    if SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard is required; run: python3 -m pip install -r "
            "requirements-ml.txt"
        )
    if min(
        config.epochs,
        config.batch_size,
        config.early_stopping_patience,
        encode_batch_size,
    ) <= 0 or config.learning_rate <= 0:
        raise ValueError("training counts and learning rate must be positive")
    if config.min_epochs < 0 or config.min_epochs > config.epochs:
        raise ValueError("min_epochs must be nonnegative and cannot exceed epochs")
    if encoder_type not in ("autoencoder", "resnet18", "resnet18_trainable"):
        raise ValueError("encoder_type must be autoencoder, resnet18, or resnet18_trainable")
    train_encoder = encoder_type == "resnet18_trainable"
    stage = "ResNet18" if train_encoder else "BC"
    scalar_prefix = "resnet" if train_encoder else "bc"
    sources = normalize_image_sources(
        image_sources if image_sources is not None else (image_source,)
    )
    source_label = sources[0] if len(sources) == 1 else "+".join(sources)
    multi_image = len(sources) > 1
    image_preprocessing = (
        {source: IMAGE_PREPROCESSING[source] for source in sources}
        if multi_image else IMAGE_PREPROCESSING[sources[0]]
    )
    if encoder_type != "autoencoder" and encoder_checkpoint is not None:
        raise ValueError("ResNet18 does not accept an --encoder override")
    if encoder_type == "autoencoder" and encoder_checkpoint is None:
        raise ValueError("autoencoder requires an encoder checkpoint")
    audit = audit_dataset(
        dataset_root,
        encoder_checkpoint,
        sources[0] if len(sources) == 1 else sources,
    )
    resolved_dataset_name = dataset_name or Path(audit["dataset_root"]).name
    audit["dataset_name"] = resolved_dataset_name
    split_manifest = create_episode_split(
        audit, config.seed,
        episodes=config.episodes,
        train_episodes=config.train_episodes,
        validation_episodes=config.validation_episodes,
        test_episodes=config.test_episodes,
    )
    split_manifest["dataset_name"] = resolved_dataset_name
    split_manifest["dataset_path"] = audit["dataset_root"]
    if output_dir.exists():
        unexpected = [
            path for path in output_dir.iterdir() if path.name != "tensorboard"
        ]
        if unexpected:
            raise FileExistsError(f"refusing to overwrite experiment: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    if encoder_type in ("resnet18", "resnet18_trainable"):
        set_seeds(config.seed)
        initial_encoder = ResNet18Encoder(
            pretrained=not train_encoder, image_sources=sources,
            latent_dimension=64 if train_encoder else 512,
        )
        encoder_checkpoint = output_dir / "resnet18_encoder.pt"
        torch.save({
            "model_class": "ResNet18Encoder",
            "model_config": initial_encoder.checkpoint_config,
            "model_state": initial_encoder.state_dict(),
            "preprocessing": initial_encoder.preprocessing,
            "metadata": {
                "image_source": source_label,
                "image_sources": list(sources),
                "weights": "random_initialization" if train_encoder else "IMAGENET1K_V1",
                "dataset_root": audit["dataset_root"],
                "image_preprocessing": image_preprocessing,
                "training_objective": (
                    "expert_normalized_action_mse" if train_encoder else "none"
                ),
            },
        }, encoder_checkpoint)
        del initial_encoder
        audit["encoder_checkpoint"] = str(encoder_checkpoint.resolve())
        audit["encoder_sha256"] = _sha256(encoder_checkpoint)
    _write_json(output_dir / "dataset_audit.json", audit)
    _write_json(output_dir / "split_manifest.json", split_manifest)
    training_config = {
        **asdict(config),
        "maximum_epochs_requested": config.epochs,
        "dataset_name": resolved_dataset_name,
        "dataset_path": audit["dataset_root"],
        "image_source": source_label,
        "image_sources": list(sources),
        "image_preprocessing": image_preprocessing,
        "tensorboard_enabled": tensorboard_enabled,
        "tensorboard_port": tensorboard_port,
        "encoder_checkpoint": str(encoder_checkpoint.resolve()),
        "encoder_selection": encoder_selection,
        "encoder_type": encoder_type,
        "encoder_trainable": train_encoder,
        "training_stage": "resnet" if train_encoder else "bc",
        "initialized_policy": (
            str(initialize_policy.expanduser().resolve())
            if initialize_policy is not None else None
        ),
    }
    _write_json(output_dir / "training_config.json", training_config)
    print(
        f"{stage} Dataset Eligibility\n"
        f"Total episodes     : {audit['total_episodes']}\n"
        f"Usable episodes    : {audit['usable_episodes']}\n"
        f"Selected episodes  : {split_manifest['selected_episode_count']}\n"
        f"Excluded episodes  : {audit['excluded_episodes']}\n"
        f"Exclusion reasons  : {audit['exclusion_reasons']}\n"
        f"Accepted samples   : {audit['total_accepted_samples']}\n"
        f"Episode split      : {split_manifest['episode_counts']}",
        flush=True,
    )
    set_seeds(config.seed)
    device = resolve_device(device_name)
    encoder, encoder_payload = _load_frozen_encoder(
        encoder_checkpoint.resolve(), device
    )
    checkpoint_source = encoder_payload.get("metadata", {}).get(
        "image_source",
        encoder_payload.get("metadata", {}).get("camera", "fpv_rgb"),
    )
    if checkpoint_source == "fpv":
        checkpoint_source = "fpv_rgb"
    if hasattr(encoder.config, "image_sources"):
        encoder_sources = tuple(encoder.config.image_sources)
        metadata_sources = tuple(
            encoder_payload.get("metadata", {}).get("image_sources", ())
        )
        if encoder_sources != sources or metadata_sources != sources:
            raise ValueError(
                f"encoder image_sources={encoder_sources!r} do not match "
                f"requested {sources!r}"
            )
    elif checkpoint_source != sources[0]:
        raise ValueError(
            f"encoder image_source={checkpoint_source!r} does not match "
            f"requested {sources[0]!r}"
        )
    if encoder.config.latent_dimension != (512 if encoder_type == "resnet18" else 64):
        raise ValueError("encoder latent dimension does not match selected encoder type")
    print(
        f"========== {stage} Training ==========\n\n"
        f"Dataset:\n{resolved_dataset_name}\n\n"
        f"Resolved path:\n{audit['dataset_root']}\n\n"
        f"Image sources:\n{', '.join(sources)}\n\n"
        f"Usable episodes:\n{audit['usable_episodes']}\n\n"
        "Train / Validation / Test:\n"
        f"{split_manifest['episode_counts']['train']} / "
        f"{split_manifest['episode_counts']['validation']} / "
        f"{split_manifest['episode_counts']['test']}\n\n"
        f"Maximum epochs:\n{config.epochs}\n\n"
        f"Batch size:\n{config.batch_size}\n\n"
        f"Learning rate:\n{config.learning_rate}\n\n"
        f"Early stopping patience:\n"
        f"{config.early_stopping_patience}\n\n"
        f"Minimum epochs:\n{config.min_epochs}\n\n"
        f"TensorBoard:\n{'enabled' if tensorboard_enabled else 'disabled'}\n\n"
        f"TensorBoard port:\n{tensorboard_port}\n\n"
        f"Encoder:\n{encoder_checkpoint.resolve()}\n\n"
        f"Encoder selection:\n{encoder_selection}\n\n"
        f"Encoder image sources:\n{', '.join(sources)}\n\n"
        f"Output:\n{output_dir.resolve()}\n\n"
        "Action:\n[v_forward, v_right, yaw_rate]\n\n"
        "=================================",
        flush=True,
    )
    loader_options = {}
    if train_encoder:
        encoder.requires_grad_(True)
        datasets = {
            split: ExpertImageDataset(
                dataset_root, split_manifest, split, image_sources=sources,
                sample_fields=STATE_FIELDS + TARGET_FIELDS,
            )
            for split in split_manifest["splits"]
        }
        train_states = torch.tensor([
            sample["sample_values"][:8] for sample in datasets["train"].samples
        ], dtype=torch.float64)
        # Learned latents change during training; only normalize the fixed state8.
        mean = np.concatenate((
            np.zeros(64, dtype=np.float32), train_states.mean(dim=0).float().numpy(),
        ))
        std = np.concatenate((
            np.ones(64, dtype=np.float32),
            train_states.std(dim=0, unbiased=False).float().clamp_min(1e-6).numpy(),
        ))
        loader_options = {
            "encoder": encoder,
            "observation_mean": torch.from_numpy(mean).to(device),
            "observation_std": torch.from_numpy(std).to(device),
        }
    else:
        tensors = encode_splits(
            dataset_root.resolve(), split_manifest, encoder, device,
            encode_batch_size, source_label,
            image_sources=sources if multi_image else None,
        )
        if any(parameter.requires_grad for parameter in encoder.parameters()):
            raise RuntimeError("encoder freeze contract was violated")
        train_observation = tensors["train"][0]
        mean = train_observation.double().mean(dim=0).float().numpy()
        std = train_observation.double().std(dim=0, unbiased=False).float().numpy()
        std = np.maximum(std, 1e-6).astype(np.float32)
        datasets = {
            split: TensorDataset(
                (observation - torch.from_numpy(mean)) / torch.from_numpy(std), action,
            )
            for split, (observation, action) in tensors.items()
        }
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
        )
        for split, dataset in datasets.items()
    }
    model = LatentBcPolicy(LatentBcPolicyConfig(
        observation_dimension=encoder.config.latent_dimension + 8
    )).to(device)
    if initialize_policy is not None:
        initial_path = initialize_policy.expanduser().resolve()
        initial = torch.load(initial_path, map_location=device, weights_only=False)
        initial_sources = tuple(initial.get(
            "image_sources", [initial.get("image_source", "")]
        ))
        if (
            not initial_sources
            or initial_sources[0] != sources[0]
            or int(initial.get("latent_dimension", -1))
            != encoder.config.latent_dimension
            or initial.get("model_class") != "LatentBcPolicy"
        ):
            raise ValueError(
                "initial BC policy must use the same primary image source, "
                "latent dimension, and LatentBcPolicy architecture"
            )
        initial_config = LatentBcPolicyConfig(**initial["model_config"])
        if initial_config != model.config:
            raise ValueError("initial BC policy observation architecture mismatch")
        model.load_state_dict(initial["model_state"])
        print(f"Initialized BC actor from {initial_path}", flush=True)
    parameters = list(model.parameters())
    if train_encoder:
        parameters.extend(encoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    early_stopping_triggered = False
    history = []
    started = time.monotonic()
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    try:
        for epoch in range(1, config.epochs + 1):
            _run_loader(
                model, loaders["train"], device, optimizer, **loader_options
            )
            train_metrics, _, _ = _run_loader(
                model, loaders["train"], device, **loader_options
            )
            validation_metrics, _, _ = _run_loader(
                model, loaders["validation"], device, **loader_options
            )
            history.append(_history_row(
                epoch, train_metrics, validation_metrics
            ))
            epoch_audit = audit
            if train_encoder:
                encoder_payload["model_state"] = encoder.state_dict()
                last_encoder = output_dir / "resnet18_encoder_last.pt"
                torch.save(encoder_payload, last_encoder)
                epoch_audit = {
                    **audit, "encoder_checkpoint": str(last_encoder.resolve()),
                    "encoder_sha256": _sha256(last_encoder),
                }
            payload = _checkpoint(
                model,
                optimizer,
                epoch,
                validation_metrics,
                mean,
                std,
                config,
                epoch_audit,
                split_manifest,
                image_source,
                encoder,
            )
            torch.save(payload, output_dir / "last.pt")
            if validation_metrics["mse"] < best_loss:
                best_loss = validation_metrics["mse"]
                best_epoch = epoch
                stale_epochs = 0
                if train_encoder:
                    torch.save(encoder_payload, encoder_checkpoint)
                    audit["encoder_sha256"] = _sha256(encoder_checkpoint)
                    payload = {
                        **payload, "autoencoder_checkpoint": str(encoder_checkpoint.resolve()),
                        "autoencoder_checkpoint_sha256": audit["encoder_sha256"],
                    }
                torch.save(payload, output_dir / "best.pt")
            else:
                stale_epochs += 1
            writer.add_scalar(
                f"{scalar_prefix}/train_action_loss", train_metrics["mse"], epoch
            )
            writer.add_scalar(
                f"{scalar_prefix}/validation_action_loss", validation_metrics["mse"], epoch
            )
            _progress(
                epoch,
                config.epochs,
                train_metrics["mse"],
                validation_metrics["mse"],
                best_loss,
                optimizer.param_groups[0]["lr"],
                started,
                stage=stage,
            )
            if (
                epoch >= config.min_epochs
                and stale_epochs >= config.early_stopping_patience
            ):
                early_stopping_triggered = True
                break
    except BaseException:
        writer.flush()
        writer.close()
        raise
    try:
        _write_history(output_dir / "training_history.csv", history)
        best = torch.load(
            output_dir / "best.pt", map_location=device, weights_only=False
        )
        best_model = LatentBcPolicy(
            LatentBcPolicyConfig(**best["model_config"])
        ).to(device)
        best_model.load_state_dict(best["model_state"])
        best_model.eval()
        if train_encoder:
            encoder, encoder_payload = _load_frozen_encoder(encoder_checkpoint, device)
            loader_options["encoder"] = encoder
            _write_json(output_dir / "dataset_audit.json", audit)
        test_metrics, prediction, target = _run_loader(
            best_model, loaders["test"], device, **loader_options
        )
        tensorboard_action_names = {
            "v_forward": "forward",
            "v_right": "right",
            "yaw_rate": "yaw_rate",
        }
        for name in ACTION_NAMES:
            writer.add_scalar(
                f"{scalar_prefix}/test_{tensorboard_action_names[name]}_rmse",
                test_metrics["per_action"][name]["rmse"],
                best_epoch,
            )
    except BaseException:
        writer.flush()
        writer.close()
        raise
    writer.flush()
    writer.close()
    metrics = {
        "format_version": FORMAT_VERSION,
        "run_status": "completed",
        "maximum_epochs_requested": config.epochs,
        "actual_epochs_trained": len(history),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopping_triggered": early_stopping_triggered,
        "train_loss": history[best_epoch - 1]["train_mse"],
        "best_validation_loss": best_loss,
        "test_mse": test_metrics["mse"],
        "test": test_metrics,
        "sample_counts": split_manifest["sample_counts"],
        "episode_counts": split_manifest["episode_counts"],
        "checkpoint_reload_verified": True,
        "image_source": source_label,
        "image_sources": list(sources),
        "image_preprocessing": image_preprocessing,
        "tensorboard_dir": str((output_dir / "tensorboard").resolve()),
        "dataset_name": resolved_dataset_name,
        "dataset_path": audit["dataset_root"],
        "tensorboard_enabled": tensorboard_enabled,
        "tensorboard_port": tensorboard_port,
        "encoder_selection": encoder_selection,
        "training_stage": "resnet" if train_encoder else "bc",
    }
    plots = create_training_plots(
        output_dir, history, test_metrics, prediction, target
    )
    metrics["plots"] = plots
    metrics["training_duration_s"] = time.monotonic() - training_started_s
    _write_json(output_dir / "metrics.json", metrics)
    metadata = dict(encoder_payload.get("metadata", {}))
    summary = {
        **metrics,
        "output_dir": str(output_dir.resolve()),
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        "last_checkpoint": str((output_dir / "last.pt").resolve()),
        "best_encoder_checkpoint": str(encoder_checkpoint.resolve()),
        "dataset_root": audit["dataset_root"],
        "encoder_architecture": type(encoder).__name__,
        "latent_dimension": encoder.config.latent_dimension,
        "encoder": {
            "architecture": type(encoder).__name__,
            "image_source": source_label,
            "image_sources": list(sources),
            "image_preprocessing": image_preprocessing,
            "latent_dimension": encoder.config.latent_dimension,
            "checkpoint": str(encoder_checkpoint.resolve()),
            "sha256": audit["encoder_sha256"],
            "model_config": encoder.config.to_dict(),
            "metadata": metadata,
            "frozen": True,
            "trained": train_encoder,
        },
        "policy": {
            "architecture": "LatentBcPolicy",
            "model_config": best_model.config.to_dict(),
            "parameter_count": best_model.parameter_count,
            "physical_action_limits": PHYSICAL_ACTION_LIMITS,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    training_config.update({
        "actual_epochs_trained": len(history),
        "best_epoch": best_epoch,
        "early_stopping_triggered": early_stopping_triggered,
        "training_duration_s": metrics["training_duration_s"],
    })
    _write_json(output_dir / "training_config.json", training_config)
    latest = output_dir.parent / "latest.json"
    _write_json(latest, {
        "format_version": FORMAT_VERSION,
        "updated_utc": _utc_now(),
        "run_dir": str(output_dir.resolve()),
        "best_checkpoint": summary["best_checkpoint"],
    })
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _parser(resnet_training: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=(
            "./uav resnet-train"
            if resnet_training
            else "./uav bc-train"
        ),
        description=(
            "Train BC with a frozen encoder, or train a 64D ResNet18 "
            "with expert action supervision and held-out evaluation."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="dataset name under datasets, or explicit path",
    )
    if resnet_training:
        parser.set_defaults(encoder=None)
        parser.add_argument(
            "--encoder-type", choices=("resnet18_trainable",),
            default="resnet18_trainable", help=argparse.SUPPRESS,
        )
    else:
        parser.add_argument("--encoder", help="trained 64D AE/ResNet checkpoint override (autoencoder mode)")
        parser.add_argument(
            "--encoder-type", choices=("autoencoder", "resnet18", "resnet18_trainable"),
            default="autoencoder", help="encoder training mode (default: autoencoder)",
        )
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int, default=100)
    add_episode_split_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=0,
        help="do not apply early stopping before this epoch",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--encode-batch-size", type=int, default=128,
        help=argparse.SUPPRESS if resnet_training else "batch size for frozen feature extraction",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--image-source", choices=IMAGE_SOURCES, default="top"
    )
    source_group.add_argument(
        "--image-sources", choices=IMAGE_SOURCES, nargs="+",
        help="one or more aligned image streams for supervised training",
    )
    parser.add_argument(
        "--initialize-policy", type=Path,
        help="initialize the action head from a checkpoint with the same primary source and latent size",
    )
    add_tensorboard_arguments(parser)
    return parser


def main() -> int:
    args = _parser(
        resnet_training=sys.argv[-2:] == ["--encoder-type", "resnet18_trainable"]
    ).parse_args()
    train_encoder = args.encoder_type == "resnet18_trainable"
    stage = "ResNet18" if train_encoder else "BC"
    error_stage = "ResNet18 training" if train_encoder else "BC baseline training"
    repository_root = Path(__file__).resolve().parents[2]
    try:
        dataset_location = resolve_dataset(
            args.dataset,
            must_exist=True,
            project_root=repository_root,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error_stage} stopped: {error}")
        return 1
    try:
        sources = normalize_image_sources(
            args.image_sources if args.image_sources else (args.image_source,)
        )
        source_label = sources[0] if len(sources) == 1 else "+".join(sources)
        if args.encoder_type != "autoencoder" and args.encoder:
            raise ValueError("ResNet18 does not accept an --encoder override")
        if args.encoder_type == "autoencoder":
            if len(sources) > 1:
                encoder = select_multi_autoencoder_checkpoint(
                    dataset_location,
                    sources,
                    {source: IMAGE_PREPROCESSING[source] for source in sources},
                    explicit=Path(args.encoder) if args.encoder else None,
                    project_root=repository_root,
                )
            else:
                encoder = select_autoencoder_checkpoint(
                    dataset_location,
                    sources[0],
                    IMAGE_PREPROCESSING[sources[0]],
                    explicit=Path(args.encoder) if args.encoder else None,
                    project_root=repository_root,
                )
        else:
            encoder = None
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {error_stage} stopped: {error}")
        return 1
    print(
        f"Encoder type: {args.encoder_type}\n"
        f"Encoder: {encoder.checkpoint if encoder else 'trainable ResNet18 (64D)' if train_encoder else 'ImageNet IMAGENET1K_V1 (frozen)'}\n\n"
        f"Dataset:\n{dataset_location.name}\n\n"
        f"Image sources:\n{', '.join(sources)}\n\n"
        + ("Encoder provenance:\nmatched" if encoder else
         "Encoder training:\nexpert normalized-action MSE" if train_encoder else
         "Encoder weights:\nImageNet pretrained; saved with this BC run"),
        flush=True,
    )
    if args.output:
        output = Path(args.output)
    else:
        output = experiment_run_directory(
            "resnet" if train_encoder else "bc",
            dataset_location,
            source_label,
            _stamp(),
            project_root=repository_root,
        )
        if args.encoder_type == "resnet18":
            output = output.parent / "resnet18" / output.name
    try:
        with TensorBoardServer(
            output / "tensorboard",
            enabled=args.tensorboard,
            port=args.tensorboard_port,
        ) as tensorboard_server:
            summary = train_baseline(
                dataset_location.path,
                encoder.checkpoint if encoder else None,
                output,
                TrainingConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    early_stopping_patience=args.patience,
                    min_epochs=args.min_epochs,
                    seed=args.seed,
                    episodes=args.episodes,
                    train_episodes=args.train_episodes,
                    validation_episodes=args.validation_episodes,
                    test_episodes=args.test_episodes,
                ),
                args.device,
                args.encode_batch_size,
                source_label,
                dataset_name=dataset_location.name,
                tensorboard_enabled=args.tensorboard,
                tensorboard_port=args.tensorboard_port,
                encoder_selection=(
                    ("automatic" if encoder.automatic else "explicit")
                    if encoder else "expert_action_supervision" if train_encoder else "imagenet_pretrained"
                ),
                encoder_type=args.encoder_type,
                image_sources=sources if len(sources) > 1 else None,
                initialize_policy=args.initialize_policy,
            )
            if not train_encoder:
                _publish_latest_bc(repository_root, summary)
            test = summary["test"]["per_action"]
            print(
                "=========================================\n"
                f"{stage} training completed successfully\n\n"
                f"Dataset:\n{dataset_location.name}\n\n"
                f"Image sources:\n{', '.join(sources)}\n\n"
                f"Maximum epochs requested:\n"
                f"{summary['maximum_epochs_requested']}\n\n"
                f"Actual epochs trained:\n"
                f"{summary['actual_epochs_trained']}\n\n"
                f"Best epoch:\n{summary['best_epoch']}\n\n"
                f"Total training duration:\n{summary['training_duration_s']:.2f} s\n\n"
                f"Early stopping:\n"
                f"{'yes' if summary['early_stopping_triggered'] else 'no'}\n\n"
                "Best validation action loss:\n"
                f"{summary['best_validation_loss']:.8f}\n\n"
                "Final test metrics:\n"
                f"forward RMSE = {test['v_forward']['rmse']:.8f}\n"
                f"right RMSE = {test['v_right']['rmse']:.8f}\n"
                f"yaw-rate RMSE = {test['yaw_rate']['rmse']:.8f}\n\n"
                f"Best checkpoint:\n{summary['best_checkpoint']}\n\n"
                f"Encoder checkpoint:\n{summary['best_encoder_checkpoint']}\n\n"
                f"TensorBoard:\n"
                f"{tensorboard_server.url if tensorboard_server.active else 'not managed'}\n"
                "=========================================",
                flush=True,
            )
            tensorboard_server.wait_until_interrupted()
    except KeyboardInterrupt:
        print(f"{stage} training interrupted; TensorBoard was cleaned up.")
        return 130
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {error_stage} stopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
