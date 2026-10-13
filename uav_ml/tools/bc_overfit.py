"""Overfit all recorded valid pairs in one episode, with isolated artifacts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import math
from pathlib import Path
import re
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from uav_ml.datasets.expert_image_dataset import (
    IMAGE_PREPROCESSING,
    select_episode_image_sets,
)
from uav_ml.inference.rgb_encoder import load_frozen_encoder
from uav_ml.models import LatentBcPolicy, LatentBcPolicyConfig
from uav_ml.tools.bc_baseline import (
    ACTION_NAMES,
    DEFAULT_SEED,
    PHYSICAL_ACTION_LIMITS,
    _episode_arrays,
    _read_json,
    _run_loader,
    _sha256,
    _validate_sample_row,
    _write_history,
    _write_json,
    action_metrics,
)
from uav_ml.tools.training_cli import PROJECT_ROOT, resolve_dataset, select_autoencoder_checkpoint
from uav_ml.train_bc import _git_sha, set_seeds

import matplotlib.pyplot as plt  # BC utilities select the noninteractive Agg backend.


FORMAT_VERSION = "bc_single_episode_diagnostic_v1.0"
EVALUATION_SCOPE = "all_recorded_valid_pairs_in_selected_episode"
MILESTONE_EPOCHS = (100, 300, 500, 1000)


@dataclass(frozen=True)
class OverfitConfig:
    """Hyperparameters for the diagnostic; no split, resume, or early stopping."""

    epochs: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.001
    seed: int = DEFAULT_SEED
    encode_batch_size: int = 128
    forward_sign_epsilon: float = 0.01
    expected_samples: int | None = None

    def validate(self) -> None:
        if not 1 <= self.epochs <= 1000:
            raise ValueError("epochs must be between 1 and 1000")
        if min(self.batch_size, self.encode_batch_size) <= 0:
            raise ValueError("batch sizes must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if not math.isfinite(self.forward_sign_epsilon) or not 0 <= self.forward_sign_epsilon < 1:
            raise ValueError("forward sign epsilon must be finite and in [0, 1)")
        if self.expected_samples is not None and self.expected_samples <= 0:
            raise ValueError("expected samples must be positive")


def audit_single_episode(dataset_root: Path, episode_id: str) -> dict:
    """Read only the requested episode and check every retained observation/action pair."""
    if re.fullmatch(r"episode_[0-9]{6}", episode_id) is None:
        raise ValueError("episode ID must have the form episode_000001")
    dataset_root = dataset_root.resolve()
    episode_dir = dataset_root / episode_id
    collection = _read_json(dataset_root / "collection_manifest.json")
    aggregate = _read_json(dataset_root / "collection_validation.json")
    if collection.get("status") != "complete" or aggregate.get("valid") is not True:
        raise ValueError("collection must be complete and validated")
    entries = [entry for entry in collection["episodes"] if entry.get("episode_id") == episode_id]
    if len(entries) != 1:
        raise ValueError(f"requested episode is missing or duplicated: {episode_id}")
    entry = entries[0]
    episode = _read_json(episode_dir / "episode.json")
    validation = _read_json(episode_dir / "validation.json")
    if any(item.get("status") != "complete" or item.get("success") is not True
           for item in (entry, episode)) or entry.get("accepted") is False:
        raise ValueError(f"requested episode is not successful: {episode_id}")
    if episode.get("episode_id") != episode_id or validation.get("episode_id") != episode_id:
        raise ValueError("episode/validation identity mismatch")
    if validation.get("valid") is not True or validation.get("episode_success") is not True:
        raise ValueError("requested episode has not passed validation")
    evidence = episode.get("accumulated_flight_evidence", {})
    if episode.get("terminal_flight_status", {}).get("state") != "COMPLETE" or not all(
        evidence.get(name) is True for name in (
            "goal_reached", "landing_commanded", "landed_after_landing_command", "terminal_complete"
        )
    ):
        raise ValueError("requested episode lacks goal/landing success evidence")
    runtime = episode.get("available_sensor_streams", {}).get("runtime_status", {})
    if runtime.get("observer_mode") != "fixed_global_top":
        raise ValueError("diagnostic requires the formal fixed_global_top stream")

    with (episode_dir / "samples.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(len(rows) != item.get("sample_count") for item in (episode, validation)):
        raise ValueError("retained sample count is empty or inconsistent")
    if "accepted_samples" in entry and entry["accepted_samples"] != len(rows):
        raise ValueError("collection accepted sample count is inconsistent")
    time_fields = ("image_timestamp_s", "state_timestamp_s", "expert_action_timestamp_s")
    for index, row in enumerate(rows, start=1):
        _validate_sample_row(dataset_root, row)
        if row["episode_id"] != episode_id or int(row["sample_id"]) != index:
            raise ValueError("sample identities must be contiguous and belong to the selected episode")
        if row.get("mission_phase") != "TRACKING" or row.get("success", "").lower() != "true" or row.get("failure"):
            raise ValueError("sample must be a successful retained TRACKING pair")
        if not (dataset_root / row["image_path"]).resolve().is_relative_to(dataset_root):
            raise ValueError("primary image path escapes the dataset")
        previous_time = float(row["previous_action_timestamp_s"])
        if not math.isfinite(previous_time) or previous_time >= float(row["expert_action_timestamp_s"]):
            raise ValueError("previous action timestamp must be finite and precede the target action")
        for time_field, error_field in (
            ("state_timestamp_s", "state_image_error_s"),
            ("expert_action_timestamp_s", "expert_action_image_error_s"),
        ):
            actual = abs(float(row[time_field]) - float(row["image_timestamp_s"]))
            if actual > 0.1 + 1e-9 or abs(actual - float(row[error_field])) > 1e-6:
                raise ValueError(f"sample {index}: synchronization metadata mismatch or over tolerance")
    for field in time_fields:
        times = [float(row[field]) for row in rows]
        if any(current <= previous for previous, current in zip(times, times[1:])):
            raise ValueError(f"{field} must be strictly monotonic")

    # The existing bundle selector applies the formal 1 ms TOP alignment check,
    # even with one source; the legacy single-image selector permits 350 ms.
    selected = select_episode_image_sets(dataset_root, episode_id, ("top",))
    image_paths = [item["images"]["top"] for item in selected]
    top_times = [float(item["source_timestamps_s"]["top"]) for item in selected]
    if len(selected) != len(rows) or len(set(image_paths)) != len(rows):
        raise ValueError("TOP images must map one-to-one to all retained samples")
    if any(current <= previous for previous, current in zip(top_times, top_times[1:])):
        raise ValueError("TOP image timestamps must be strictly monotonic")
    files = [dataset_root / name for name in ("collection_manifest.json", "collection_validation.json")]
    files.extend(episode_dir / name for name in ("episode.json", "validation.json", "samples.csv", "auxiliary.csv"))
    files.extend(image_paths)
    return {
        "episode_id": episode_id,
        "dataset_root": str(dataset_root),
        "evaluation_scope": EVALUATION_SCOPE,
        "sample_count": len(rows),
        "sample_ids": [int(row["sample_id"]) for row in rows],
        "image_timestamps_s": [float(row["image_timestamp_s"]) for row in rows],
        "maximum_top_alignment_error_s": max(item["source_errors_s"]["top"] for item in selected),
        "rejected_sample_count": episode.get("rejected_sample_count"),
        "rejections_by_reason": episode.get("rejections_by_reason", {}),
        "files_sha256": {str(path.relative_to(dataset_root)): _sha256(path) for path in files},
    }


def offline_action_metrics(prediction: torch.Tensor, target: torch.Tensor, epsilon: float) -> dict:
    """Compare same-set actions in normalized and physical units."""
    if not math.isfinite(epsilon) or not 0 <= epsilon < 1:
        raise ValueError("forward sign epsilon must be finite and in [0, 1)")
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3 or len(target) == 0:
        raise ValueError("action comparison requires matching nonempty [N,3] tensors")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("action comparison contains NaN or Inf")
    normalized = action_metrics(prediction, target)
    error = prediction.double() - target.double()
    physical = {}
    for index, (name, (physical_name, scale)) in enumerate(zip(ACTION_NAMES, PHYSICAL_ACTION_LIMITS.items())):
        normalized["per_action"][name]["maximum_absolute_error"] = float(error[:, index].abs().max())
        values = normalized["per_action"][name]
        physical[name] = {
            "unit": "rad/s" if physical_name.endswith("radps") else "m/s",
            "mse": values["mse"] * scale**2,
            **{metric: values[metric] * scale for metric in ("mae", "rmse", "maximum_absolute_error")},
        }
    eligible = target[:, 0].abs() > epsilon
    count = int(eligible.sum())
    correct = int((prediction[eligible, 0].sign() == target[eligible, 0].sign()).sum())
    return {
        "training_mse": normalized["mse"],
        "sample_count": len(target),
        "normalized": normalized,
        "physical": physical,
        "forward_sign": {
            "epsilon_normalized": epsilon,
            "eligible_samples": count,
            "excluded_samples": len(target) - count,
            "correct_samples": correct,
            "accuracy": correct / count if count else None,
        },
    }


def _validate_output(output: Path, dataset: Path, encoder: Path, project_root: Path) -> Path:
    output = output.expanduser().resolve()
    diagnostic_root = (project_root / "experiments/diagnostics/bc_single_episode").resolve()
    if output == diagnostic_root or not output.is_relative_to(diagnostic_root):
        raise ValueError(f"output must be a new run directory under {diagnostic_root}")
    protected = (dataset, encoder, project_root / "datasets", project_root / "experiments/bc",
                 project_root / "experiments/autoencoder", project_root / "artifacts/experiments")
    if any(output.is_relative_to(path.resolve()) or path.resolve().is_relative_to(output) for path in protected):
        raise ValueError("output overlaps a dataset or a protected training artifact directory")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    return output


def _save_predictions(path: Path, audit: dict, prediction: torch.Tensor, target: torch.Tensor) -> None:
    rows = []
    for index, sample_id in enumerate(audit["sample_ids"]):
        row = {"episode_id": audit["episode_id"], "sample_id": sample_id,
               "image_timestamp_s": audit["image_timestamps_s"][index]}
        for component, (name, scale) in enumerate(zip(ACTION_NAMES, PHYSICAL_ACTION_LIMITS.values())):
            expert, predicted = float(target[index, component]), float(prediction[index, component])
            for label, value in (("expert", expert), ("predicted", predicted), ("error", predicted - expert)):
                row[f"{name}_{label}_normalized"] = value
                row[f"{name}_{label}_physical"] = value * scale
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _create_plots(output: Path, history: list[dict], audit: dict, prediction: torch.Tensor, target: torch.Tensor) -> list[str]:
    plots = output / "plots"
    plots.mkdir()
    paths = []
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot([row["epoch"] for row in history], [row["train_mse"] for row in history])
    axis.set(xlabel="Epoch", ylabel="Normalized training MSE", title="Single-episode BC diagnostic")
    paths.append(plots / "training_loss.png")
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=160)
    plt.close(figure)
    steps = audit["sample_ids"]
    predicted, expert = prediction.numpy(), target.numpy()
    for component, (name, scale) in enumerate(zip(ACTION_NAMES, PHYSICAL_ACTION_LIMITS.values())):
        figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        for axis, multiplier, unit in zip(axes, (1.0, scale), ("Normalized", "rad/s" if name == "yaw_rate" else "m/s")):
            axis.plot(steps, expert[:, component] * multiplier, label="Expert")
            axis.plot(steps, predicted[:, component] * multiplier, label="BC")
            axis.set(ylabel=unit, title=name)
            axis.grid(alpha=0.3)
            axis.legend()
        axes[-1].set_xlabel("Accepted sample ID (recorded order)")
        paths.append(plots / f"{name}_comparison.png")
        figure.tight_layout()
        figure.savefig(paths[-1], dpi=160)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 4))
    for component, name in enumerate(ACTION_NAMES):
        axis.plot(steps, predicted[:, component] - expert[:, component], label=name)
    axis.set(xlabel="Accepted sample ID (recorded order)", ylabel="Normalized prediction error")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.legend()
    paths.append(plots / "action_error.png")
    figure.tight_layout()
    figure.savefig(paths[-1], dpi=160)
    plt.close(figure)
    return [str(path) for path in paths]


def train_single_episode(
    dataset_root: Path,
    episode_id: str,
    encoder_checkpoint: Path,
    output_dir: Path,
    config: OverfitConfig = OverfitConfig(),
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    """Fit retained pairs with the existing batch trainer; never publish BC pointers."""
    config.validate()
    started = time.monotonic()
    dataset = resolve_dataset(dataset_root, must_exist=True, short_name=False, project_root=project_root)
    checkpoint = encoder_checkpoint.expanduser().resolve()
    output = _validate_output(output_dir, dataset.path, checkpoint, project_root)
    audit = audit_single_episode(dataset.path, episode_id)
    if config.expected_samples is not None and audit["sample_count"] != config.expected_samples:
        raise ValueError(f"expected {config.expected_samples} valid samples, found {audit['sample_count']}")
    selection = select_autoencoder_checkpoint(dataset, "top", IMAGE_PREPROCESSING["top"], explicit=checkpoint)
    device = torch.device("cpu")
    encoder, encoder_payload = load_frozen_encoder(selection.checkpoint, device)
    if encoder_payload["model_class"] not in ("RgbAutoencoderV0", "ResNet18AutoencoderV0"):
        raise ValueError("diagnostic requires a pretrained single-source 64D autoencoder")
    if (encoder.config.latent_dimension, encoder.config.image_width, encoder.config.image_height) != (64, 128, 72):
        raise ValueError("encoder must use the existing 64D, 128x72 image contract")
    encoder_hash = _sha256(checkpoint)
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    history = []
    try:
        # One row = [64 frozen image features, 8 recorded state values].
        observation, action = _episode_arrays(dataset.path, episode_id, encoder, device, config.encode_batch_size, "top")
        if observation.shape != (audit["sample_count"], 72) or action.shape != (audit["sample_count"], 3):
            raise ValueError("encoded tensors must contain every selected 72D/3D pair")
        if not np.isfinite(observation).all() or not np.isfinite(action).all():
            raise ValueError("encoded observations/actions contain NaN or Inf")
        if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
            raise RuntimeError("frozen encoder contract was violated")
        observation = torch.from_numpy(observation)
        action = torch.from_numpy(action)
        # Match formal frozen-AE BC: population std, float64 reduction, 1e-6 floor.
        mean = observation.double().mean(dim=0).float()
        std = observation.double().std(dim=0, unbiased=False).float().clamp_min(1e-6)
        pairs = TensorDataset((observation - mean) / std, action)
        train_loader = DataLoader(pairs, batch_size=config.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(config.seed))
        evaluation_loader = DataLoader(pairs, batch_size=config.batch_size, shuffle=False)
        set_seeds(config.seed)
        model = LatentBcPolicy(LatentBcPolicyConfig()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        metadata = {
            **asdict(config), "format_version": FORMAT_VERSION, "dataset_root": str(dataset.path),
            "dataset_name": dataset.name, "episode_id": episode_id, "sample_count": len(pairs),
            "evaluation_scope": EVALUATION_SCOPE, "device": "cpu", "image_source": "top",
            "image_preprocessing": IMAGE_PREPROCESSING["top"], "latent_dimension": 64,
            "encoder_architecture": encoder_payload["model_class"], "encoder_model_config": encoder_payload["model_config"],
            "autoencoder_checkpoint": str(checkpoint), "autoencoder_checkpoint_sha256": encoder_hash,
            "encoder_frozen": True, "bc_initialization": "random", "optimizer": "Adam",
            "gradient_clipping": 5.0, "early_stopping": False, "data_augmentation": False,
            "checkpoint_selection": "minimum_post_epoch_training_mse", "deterministic_algorithms": True,
            "physical_action_limits": PHYSICAL_ACTION_LIMITS, "git_commit": _git_sha(),
            "software_versions": {"python": sys.version.split()[0], "torch": str(torch.__version__), "numpy": np.__version__},
            "code_sha256": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in (
                Path(__file__), PROJECT_ROOT / "uav_ml/tools/bc_baseline.py",
                PROJECT_ROOT / "uav_ml/models/latent_bc_policy.py",
                PROJECT_ROOT / "uav_ml/datasets/expert_image_dataset.py",
                PROJECT_ROOT / "uav_ml/inference/rgb_encoder.py",
            )},
        }
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "experiment_config.json", metadata)
        _write_json(output / "episode_audit.json", audit)
        _write_json(output / "observation_normalization.json", {"mean": mean.tolist(), "std": std.tolist(), "source": EVALUATION_SCOPE})
        best_mse, best_epoch = math.inf, 0

        # Core training loop: update on batches, then score all pairs at fixed weights.
        for epoch in range(1, config.epochs + 1):
            _run_loader(model, train_loader, device, optimizer)
            _, prediction, target = _run_loader(model, evaluation_loader, device)
            metrics = offline_action_metrics(prediction, target, config.forward_sign_epsilon)
            row = {"epoch": epoch, "train_mse": metrics["training_mse"]}
            for name, values in metrics["normalized"]["per_action"].items():
                row.update({f"{name}_{metric}": value for metric, value in values.items()})
            history.append(row)
            payload = {
                "format_version": FORMAT_VERSION, "model_class": "LatentBcPolicy",
                "model_config": model.config.to_dict(), "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "epoch": epoch, "training_metrics": metrics,
                "experiment_config": metadata, "episode_audit": audit,
                "observation_mean": mean, "observation_std": std,
            }
            torch.save(payload, output / "last.pt")
            if metrics["training_mse"] < best_mse:
                best_mse, best_epoch = metrics["training_mse"], epoch
                torch.save(payload, output / "best.pt")
            print(f"Epoch {epoch}/{config.epochs}: training MSE={metrics['training_mse']:.9g}; best={best_mse:.9g}", flush=True)
            if epoch in MILESTONE_EPOCHS:
                _write_json(output / f"epoch_{epoch:04d}_metrics.json", metrics)
                print(f"Milestone {epoch}: {metrics}", flush=True)

        # Reload the chosen weights AND normalization for same-set inference.
        best = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model_state"])
        inference_pairs = TensorDataset((observation - best["observation_mean"]) / best["observation_std"], action)
        inference_loader = DataLoader(inference_pairs, batch_size=config.batch_size, shuffle=False)
        _, prediction, target = _run_loader(model, inference_loader, device)
        evaluation = offline_action_metrics(prediction, target, config.forward_sign_epsilon)
        if not math.isclose(evaluation["training_mse"], best_mse, rel_tol=1e-7, abs_tol=1e-12):
            raise RuntimeError("reloaded checkpoint predictions differ from the selected training MSE")
        _save_predictions(output / "action_predictions.csv", audit, prediction, target)
        plots = _create_plots(output, history, audit, prediction, target)
        summary = {
            "format_version": FORMAT_VERSION, "run_status": "completed", "evaluation_scope": EVALUATION_SCOPE,
            "episode_id": episode_id, "sample_count": len(pairs), "epochs_completed": len(history),
            "best_epoch": best_epoch, "best_training_mse": best_mse, "last_training_mse": history[-1]["train_mse"],
            "checkpoint_reload_verified": True, "training_duration_s": time.monotonic() - started,
            "best_checkpoint": str(output / "best.pt"), "last_checkpoint": str(output / "last.pt"),
            "output_dir": str(output), "offline_evaluation": evaluation, "plots": plots,
        }
        _write_json(output / "offline_metrics.json", evaluation)
        _write_json(output / "summary.json", summary)
        return summary
    finally:
        try:
            if history:
                _write_history(output / "training_history.csv", history)
        finally:
            torch.use_deterministic_algorithms(deterministic_before, warn_only=warn_only_before)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="./uav bc-overfit", description=__doc__)
    parser.add_argument("--dataset", default="cylinder_v4")
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--image-source", choices=("top",), default="top")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--forward-sign-epsilon", type=float, default=0.01, help="exclude |expert forward| <= epsilon in normalized units")
    parser.add_argument("--expected-samples", type=int, help="fail if the selected valid sample count has changed")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        dataset = resolve_dataset(args.dataset, must_exist=True)
        stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%dT%H%M%S_%f")
        output = args.output or PROJECT_ROOT / "experiments/diagnostics/bc_single_episode" / dataset.name / "top" / args.episode_id / f"run_{stamp}"
        config = OverfitConfig(**{name: getattr(args, name) for name in OverfitConfig.__dataclass_fields__})
        summary = train_single_episode(dataset.path, args.episode_id, args.encoder, output, config)
        print(f"Diagnostic completed: {summary['output_dir']}\nBest training MSE: {summary['best_training_mse']:.9g}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("Single-episode diagnostic interrupted; recorded history/checkpoints are retained.", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
