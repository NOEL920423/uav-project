"""Train and offline-evaluate the formal frozen-encoder BC baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

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
    IMAGE_PREPROCESSING,
    IMAGE_SOURCES,
    preprocess_expert_image,
    select_episode_images,
)
from uav_ml.models import (
    LatentBcPolicy,
    LatentBcPolicyConfig,
    RgbAutoencoderConfig,
    RgbAutoencoderV0,
)
from uav_ml.tools.validate_expert_collection import (
    DEFAULT_DATASET,
)
from uav_ml.tools.training_cli import (
    TensorBoardServer,
    add_tensorboard_arguments,
    experiment_run_directory,
    resolve_dataset,
    select_autoencoder_checkpoint,
)
from uav_ml.train_bc import resolve_device, set_seeds

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
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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
    early_stopping_patience: int = 12
    seed: int = DEFAULT_SEED


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
    image_source: str = "fpv_rgb",
) -> dict:
    """Select only complete, validated, readable successful episodes."""
    dataset_root = dataset_root.resolve()
    if image_source not in IMAGE_SOURCES:
        raise ValueError(f"unsupported image source: {image_source}")
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
        if runtime.get("phase10c_observer_mode") == "fixed_global_top":
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
                # Validate every selected input. Formal TOP cohort membership
                # above makes each image-source run share the same episodes.
                selected = select_episode_images(dataset_root, episode_id, image_source)
                if len(selected) != len(rows):
                    raise ValueError("selected image count differs from samples")
            except (OSError, KeyError, TypeError, ValueError) as error:
                if image_source != "fpv_rgb":
                    raise ValueError(
                        f"{image_source} association failed loudly for "
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
        "image_source": image_source,
        "image_preprocessing": IMAGE_PREPROCESSING[image_source],
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


def create_episode_split(audit: dict, seed: int) -> dict:
    """Create a deterministic 80/10/10 split without frame leakage."""
    episodes = [item["episode_id"] for item in audit["usable"]]
    random.Random(seed).shuffle(episodes)
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
        "strategy": "episode_level_80_10_10",
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
) -> tuple[RgbAutoencoderV0, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("model_class") != "RgbAutoencoderV0":
        raise ValueError("encoder checkpoint is not RgbAutoencoderV0")
    model = RgbAutoencoderV0(
        RgbAutoencoderConfig(**payload["model_config"])
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def _episode_arrays(
    dataset_root: Path,
    episode_id: str,
    encoder: RgbAutoencoderV0,
    device: torch.device,
    encode_batch_size: int,
    image_source: str = "fpv_rgb",
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
    selected = select_episode_images(dataset_root, episode_id, image_source)
    if len(selected) != len(rows):
        raise ValueError(f"{episode_id}: selected image/sample count mismatch")
    latents = []
    for start in range(0, len(rows), encode_batch_size):
        images = [
            preprocess_expert_image(
                item["image_path"],
                image_source,
                image_width=encoder.config.image_width,
                image_height=encoder.config.image_height,
            )
            for item in selected[start:start + encode_batch_size]
        ]
        batch = torch.stack(images).to(device)
        with torch.inference_mode():
            latents.append(encoder.encode(batch).cpu().numpy())
    latent = np.concatenate(latents).astype(np.float32)
    observation = np.concatenate((latent, states), axis=1).astype(np.float32)
    if observation.shape[1] != 72 or targets.shape[1] != 3:
        raise ValueError("rebuilt BC tensors violate the 72D/3D contract")
    return observation, targets


def encode_splits(
    dataset_root: Path,
    split_manifest: dict,
    encoder: RgbAutoencoderV0,
    device: torch.device,
    encode_batch_size: int,
    image_source: str = "fpv_rgb",
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
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    training = optimizer is not None
    model.train(training)
    predictions = []
    targets = []
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
    encoder: RgbAutoencoderV0,
) -> dict:
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
        "encoder_architecture": "RgbAutoencoderV0",
        "encoder_frozen": True,
        "image_source": image_source,
        "image_preprocessing": IMAGE_PREPROCESSING[image_source],
        "encoder_preprocessing": IMAGE_PREPROCESSING[image_source],
        "latent_dimension": encoder.config.latent_dimension,
        "observation_contract": "latent64_plus_body_state8_v1.0",
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
) -> None:
    elapsed = time.monotonic() - started
    eta = elapsed / epoch * (epochs - epoch) if epoch else None
    filled = int(30 * epoch / epochs)
    bar = "█" * filled + "-" * (30 - filled)
    print(
        "\nBC Baseline Training\n\n"
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
    encoder_checkpoint: Path,
    output_dir: Path,
    config: TrainingConfig,
    device_name: str,
    encode_batch_size: int = 128,
    image_source: str = "fpv_rgb",
    dataset_name: str | None = None,
    tensorboard_enabled: bool = False,
    tensorboard_port: int = 6006,
    encoder_selection: str = "explicit",
) -> dict:
    """Run reproducible BC training and held-out offline evaluation."""
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
    audit = audit_dataset(dataset_root, encoder_checkpoint, image_source)
    resolved_dataset_name = dataset_name or Path(audit["dataset_root"]).name
    audit["dataset_name"] = resolved_dataset_name
    split_manifest = create_episode_split(audit, config.seed)
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
    _write_json(output_dir / "dataset_audit.json", audit)
    _write_json(output_dir / "split_manifest.json", split_manifest)
    training_config = {
        **asdict(config),
        "maximum_epochs_requested": config.epochs,
        "dataset_name": resolved_dataset_name,
        "dataset_path": audit["dataset_root"],
        "image_source": image_source,
        "image_preprocessing": IMAGE_PREPROCESSING[image_source],
        "tensorboard_enabled": tensorboard_enabled,
        "tensorboard_port": tensorboard_port,
        "encoder_checkpoint": str(encoder_checkpoint.resolve()),
        "encoder_selection": encoder_selection,
    }
    _write_json(output_dir / "training_config.json", training_config)
    print(
        "BC Dataset Eligibility\n"
        f"Total episodes     : {audit['total_episodes']}\n"
        f"Usable episodes    : {audit['usable_episodes']}\n"
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
    if checkpoint_source != image_source:
        raise ValueError(
            f"encoder image_source={checkpoint_source!r} does not match "
            f"requested {image_source!r}"
        )
    if encoder.config.latent_dimension != 64:
        raise ValueError("formal BC baseline requires a 64D encoder latent")
    print(
        "========== BC Training ==========\n\n"
        f"Dataset:\n{resolved_dataset_name}\n\n"
        f"Resolved path:\n{audit['dataset_root']}\n\n"
        f"Image source:\n{image_source}\n\n"
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
        f"TensorBoard:\n{'enabled' if tensorboard_enabled else 'disabled'}\n\n"
        f"TensorBoard port:\n{tensorboard_port}\n\n"
        f"Encoder:\n{encoder_checkpoint.resolve()}\n\n"
        f"Encoder selection:\n{encoder_selection}\n\n"
        f"Encoder image source:\n{checkpoint_source}\n\n"
        f"Output:\n{output_dir.resolve()}\n\n"
        "Action:\n[v_forward, v_right, yaw_rate]\n\n"
        "=================================",
        flush=True,
    )
    tensors = encode_splits(
        dataset_root.resolve(),
        split_manifest,
        encoder,
        device,
        encode_batch_size,
        image_source,
    )
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("encoder freeze contract was violated")
    train_observation = tensors["train"][0]
    mean = train_observation.double().mean(dim=0).float().numpy()
    std = train_observation.double().std(dim=0, unbiased=False).float().numpy()
    std = np.maximum(std, 1e-6).astype(np.float32)
    normalized = {
        split: (
            (observation - torch.from_numpy(mean)) / torch.from_numpy(std),
            action,
        )
        for split, (observation, action) in tensors.items()
    }
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        split: DataLoader(
            TensorDataset(*values),
            batch_size=config.batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
        )
        for split, values in normalized.items()
    }
    model = LatentBcPolicy().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate
    )
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
                model, loaders["train"], device, optimizer
            )
            train_metrics, _, _ = _run_loader(
                model, loaders["train"], device
            )
            validation_metrics, _, _ = _run_loader(
                model, loaders["validation"], device
            )
            history.append(_history_row(
                epoch, train_metrics, validation_metrics
            ))
            payload = _checkpoint(
                model,
                optimizer,
                epoch,
                validation_metrics,
                mean,
                std,
                config,
                audit,
                split_manifest,
                image_source,
                encoder,
            )
            torch.save(payload, output_dir / "last.pt")
            if validation_metrics["mse"] < best_loss:
                best_loss = validation_metrics["mse"]
                best_epoch = epoch
                stale_epochs = 0
                torch.save(payload, output_dir / "best.pt")
            else:
                stale_epochs += 1
            writer.add_scalar(
                "bc/train_action_loss", train_metrics["mse"], epoch
            )
            writer.add_scalar(
                "bc/validation_action_loss", validation_metrics["mse"], epoch
            )
            _progress(
                epoch,
                config.epochs,
                train_metrics["mse"],
                validation_metrics["mse"],
                best_loss,
                optimizer.param_groups[0]["lr"],
                started,
            )
            if stale_epochs >= config.early_stopping_patience:
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
        test_metrics, prediction, target = _run_loader(
            best_model, loaders["test"], device
        )
        tensorboard_action_names = {
            "v_forward": "forward",
            "v_right": "right",
            "yaw_rate": "yaw_rate",
        }
        for name in ACTION_NAMES:
            writer.add_scalar(
                f"bc/test_{tensorboard_action_names[name]}_rmse",
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
        "image_source": image_source,
        "image_preprocessing": IMAGE_PREPROCESSING[image_source],
        "tensorboard_dir": str((output_dir / "tensorboard").resolve()),
        "dataset_name": resolved_dataset_name,
        "dataset_path": audit["dataset_root"],
        "tensorboard_enabled": tensorboard_enabled,
        "tensorboard_port": tensorboard_port,
        "encoder_selection": encoder_selection,
    }
    plots = create_training_plots(
        output_dir, history, test_metrics, prediction, target
    )
    metrics["plots"] = plots
    _write_json(output_dir / "metrics.json", metrics)
    metadata = dict(encoder_payload.get("metadata", {}))
    summary = {
        **metrics,
        "output_dir": str(output_dir.resolve()),
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        "last_checkpoint": str((output_dir / "last.pt").resolve()),
        "encoder": {
            "architecture": "RgbAutoencoderV0",
            "image_source": image_source,
            "image_preprocessing": IMAGE_PREPROCESSING[image_source],
            "latent_dimension": encoder.config.latent_dimension,
            "checkpoint": str(encoder_checkpoint.resolve()),
            "sha256": audit["encoder_sha256"],
            "model_config": encoder.config.to_dict(),
            "metadata": metadata,
            "frozen": True,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./uav bc-train",
        description=(
            "Train the formal 72D frozen-RGB-encoder BC baseline and run "
            "held-out offline evaluation."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="dataset name under artifacts/datasets, or explicit path",
    )
    parser.add_argument("--encoder")
    parser.add_argument("--output")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument(
        "--image-source", choices=IMAGE_SOURCES, default="top"
    )
    add_tensorboard_arguments(parser)
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository_root = Path(__file__).resolve().parents[2]
    try:
        dataset_location = resolve_dataset(
            args.dataset,
            must_exist=True,
            project_root=repository_root,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: BC baseline training stopped: {error}")
        return 1
    try:
        encoder = select_autoencoder_checkpoint(
            dataset_location,
            args.image_source,
            IMAGE_PREPROCESSING[args.image_source],
            explicit=Path(args.encoder) if args.encoder else None,
            project_root=repository_root,
        )
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: BC baseline training stopped: {error}")
        return 1
    print(
        f"{'Auto-selected' if encoder.automatic else 'Explicit'} encoder:\n"
        f"{encoder.checkpoint}\n\n"
        f"Dataset:\n{dataset_location.name}\n\n"
        f"Image source:\n{args.image_source}\n\n"
        "Encoder provenance:\nmatched",
        flush=True,
    )
    if args.output:
        output = Path(args.output)
    else:
        output = experiment_run_directory(
            "bc",
            dataset_location,
            args.image_source,
            _stamp(),
            project_root=repository_root,
        )
    try:
        with TensorBoardServer(
            output / "tensorboard",
            enabled=args.tensorboard,
            port=args.tensorboard_port,
        ) as tensorboard_server:
            summary = train_baseline(
                dataset_location.path,
                encoder.checkpoint,
                output,
                TrainingConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    early_stopping_patience=args.patience,
                    seed=args.seed,
                ),
                args.device,
                args.encode_batch_size,
                args.image_source,
                dataset_name=dataset_location.name,
                tensorboard_enabled=args.tensorboard,
                tensorboard_port=args.tensorboard_port,
                encoder_selection=(
                    "automatic" if encoder.automatic else "explicit"
                ),
            )
            test = summary["test"]["per_action"]
            print(
                "=========================================\n"
                "BC training completed successfully\n\n"
                f"Dataset:\n{dataset_location.name}\n\n"
                f"Image source:\n{args.image_source}\n\n"
                f"Maximum epochs requested:\n"
                f"{summary['maximum_epochs_requested']}\n\n"
                f"Actual epochs trained:\n"
                f"{summary['actual_epochs_trained']}\n\n"
                f"Best epoch:\n{summary['best_epoch']}\n\n"
                f"Early stopping:\n"
                f"{'yes' if summary['early_stopping_triggered'] else 'no'}\n\n"
                "Best validation action loss:\n"
                f"{summary['best_validation_loss']:.8f}\n\n"
                "Final test metrics:\n"
                f"forward RMSE = {test['v_forward']['rmse']:.8f}\n"
                f"right RMSE = {test['v_right']['rmse']:.8f}\n"
                f"yaw-rate RMSE = {test['yaw_rate']['rmse']:.8f}\n\n"
                f"Best checkpoint:\n{summary['best_checkpoint']}\n\n"
                f"TensorBoard:\n"
                f"{tensorboard_server.url if tensorboard_server.active else 'not managed'}\n"
                "=========================================",
                flush=True,
            )
            tensorboard_server.wait_until_interrupted()
    except KeyboardInterrupt:
        print("BC training interrupted; TensorBoard was cleaned up.")
        return 130
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: BC baseline training stopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
