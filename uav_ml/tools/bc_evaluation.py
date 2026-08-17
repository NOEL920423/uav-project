"""Reusable closed-loop evaluation contracts for the formal BC baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Callable, Protocol

import numpy as np
from PIL import Image
import torch

from uav_ml.datasets.rgb_episode_dataset import preprocess_rgb_image
from uav_ml.models import (
    LatentBcPolicy,
    LatentBcPolicyConfig,
    RgbAutoencoderConfig,
    RgbAutoencoderV0,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


FORMAT_VERSION = "bc_closed_loop_v1.0"
CONTROL_SOURCE = "BC_POLICY"


class ClosedLoopEnvironment(Protocol):
    """Minimal environment surface required by the evaluator."""

    def reset(self, *, seed: int) -> tuple[dict, dict]:
        ...

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        ...

    def close(self) -> None:
        ...


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def resolve_checkpoint(checkpoint: Path | None, repository_root: Path) -> Path:
    """Resolve an explicit checkpoint or the latest completed training run."""
    if checkpoint is not None:
        path = checkpoint.expanduser().resolve()
    else:
        latest_path = repository_root.joinpath(
            "artifacts", "experiments", "bc_baseline", "latest.json"
        )
        if not latest_path.is_file():
            raise FileNotFoundError(
                "no BC baseline checkpoint was selected and latest.json is "
                f"missing: {latest_path}; run ./uav bc-train first"
            )
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        path = Path(latest["best_checkpoint"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BC checkpoint is missing: {path}")
    return path


def unseen_evaluation_seeds(
    dataset_root: Path,
    episodes: int,
    seed_base: int | None = None,
) -> list[int]:
    """Choose deterministic evaluation seeds disjoint from every expert split."""
    if episodes <= 0:
        raise ValueError("--episodes must be a positive integer")
    manifest_path = dataset_root / "collection_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    used = {
        int(item["seed"])
        for item in manifest.get("episodes", [])
        if item.get("seed") is not None
    }
    start = (
        seed_base if seed_base is not None else
        max(900_000, max(used, default=-100_000) + 100_000)
    )
    if start < 0:
        raise ValueError("--seed-base must be nonnegative")
    selected = []
    candidate = start
    while len(selected) < episodes:
        if candidate not in used:
            selected.append(candidate)
        candidate += 1
    if set(selected) & used:
        raise RuntimeError("evaluation seed leakage detected")
    return selected


class BcPolicyRuntime:
    """Exact frozen-encoder and normalized-policy inference pipeline."""

    def __init__(
        self,
        checkpoint_path: Path,
        device: torch.device,
        encoder_override: Path | None = None,
    ) -> None:
        checkpoint_path = checkpoint_path.resolve()
        # Training may use NumPy 2 while Isaac Sim 5.1 embeds NumPy 1.x. Older
        # checkpoints therefore refer to numpy._core during torch unpickling.
        if not hasattr(np, "_core"):
            sys.modules.setdefault("numpy._core", np.core)
            sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
            sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        payload = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if payload.get("format_version") != "bc_baseline_v1.0":
            raise ValueError("checkpoint is not a formal BC baseline checkpoint")
        if payload.get("model_class") != "LatentBcPolicy":
            raise ValueError("checkpoint model class is not LatentBcPolicy")
        if payload.get("physical_action_limits") != {
            "v_forward_mps": 1.0,
            "v_right_mps": 0.8,
            "yaw_rate_radps": 1.0,
        }:
            raise ValueError("checkpoint physical action limits are incompatible")
        encoder_path = (
            encoder_override.resolve() if encoder_override is not None else
            Path(payload["autoencoder_checkpoint"]).resolve()
        )
        if not encoder_path.is_file():
            raise FileNotFoundError(f"encoder checkpoint is missing: {encoder_path}")
        encoder_hash = _sha256(encoder_path)
        if encoder_hash != payload["autoencoder_checkpoint_sha256"]:
            raise ValueError("encoder SHA-256 differs from the training checkpoint")
        encoder_payload = torch.load(
            encoder_path, map_location=device, weights_only=False
        )
        if encoder_payload.get("model_class") != "RgbAutoencoderV0":
            raise ValueError("encoder checkpoint is not RgbAutoencoderV0")
        encoder_config = RgbAutoencoderConfig(**encoder_payload["model_config"])
        self.encoder = RgbAutoencoderV0(encoder_config).to(device)
        self.encoder.load_state_dict(encoder_payload["model_state"])
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.policy = LatentBcPolicy(
            LatentBcPolicyConfig(**payload["model_config"])
        ).to(device)
        self.policy.load_state_dict(payload["model_state"])
        self.policy.eval()
        self.mean = torch.as_tensor(
            payload["observation_mean"], dtype=torch.float32, device=device
        )
        self.std = torch.as_tensor(
            payload["observation_std"], dtype=torch.float32, device=device
        )
        if self.mean.shape != (72,) or self.std.shape != (72,):
            raise ValueError("checkpoint normalization must contain 72 values")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("checkpoint normalization contains NaN or Inf")
        if torch.any(self.std <= 0):
            raise ValueError("checkpoint normalization standard deviation is invalid")
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = _sha256(checkpoint_path)
        self.encoder_path = encoder_path
        self.encoder_sha256 = encoder_hash
        self.training_dataset_root = Path(payload["dataset_root"]).resolve()

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        """Map only FPV RGB plus the public 8D state to a 3D BC action."""
        if set(("rgb", "state")) - set(observation):
            raise ValueError("observation must contain the required rgb/state inputs")
        rgb = np.asarray(observation["rgb"])
        state = np.asarray(observation["state"], dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or state.shape != (8,):
            raise ValueError("observation violates the FPV RGB + 8D state contract")
        if not np.isfinite(state).all():
            raise ValueError("state contains NaN or Inf")
        image = Image.fromarray(rgb[..., :3].astype(np.uint8), mode="RGB")
        image_tensor = preprocess_rgb_image(
            image,
            image_width=self.encoder.config.image_width,
            image_height=self.encoder.config.image_height,
        ).unsqueeze(0).to(self.device)
        latent = self.encoder.encode(image_tensor)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        policy_input = torch.cat((latent, state_tensor), dim=1)
        normalized = (policy_input - self.mean) / self.std
        action = self.policy(normalized)[0]
        return action.cpu().numpy().astype(np.float32)


@dataclass
class EvaluationProgress:
    """Low-frequency closed-loop progress display."""

    total: int
    started: float

    def render(
        self,
        *,
        completed: int,
        seed: int,
        state: str,
        goal_distance: float | None,
        result: str,
        records: list[dict],
    ) -> None:
        elapsed = time.monotonic() - self.started
        eta = elapsed / completed * (self.total - completed) if completed else None
        filled = int(30 * completed / self.total)
        bar = "█" * filled + "-" * (30 - filled)
        collisions = sum(item["collision"] for item in records)
        timeouts = sum(item["timeout"] for item in records)
        successes = sum(item["success"] for item in records)
        distance = "--" if goal_distance is None else f"{goal_distance:.3f} m"
        print(
            "\nBC Closed-Loop Evaluation\n\n"
            f"CONTROL SOURCE = {CONTROL_SOURCE}\n\n"
            f"Episodes: {completed} / {self.total} [{bar}] "
            f"{100 * completed // self.total}%\n\n"
            "Current:\n"
            f"  seed       : {seed}\n"
            f"  state      : {state}\n"
            f"  goal dist  : {distance}\n"
            f"  result     : {result}\n\n"
            "Results:\n"
            f"  success    : {successes}\n"
            f"  collision  : {collisions}\n"
            f"  timeout    : {timeouts}\n\n"
            f"Elapsed : {_duration(elapsed)}\n"
            f"ETA     : {_duration(eta)}\n",
            flush=True,
        )


def _terminal_reason(info: dict) -> str:
    if info.get("success"):
        return "goal_reached"
    if info.get("collision"):
        return "collision"
    if info.get("out_of_bounds"):
        return "out_of_bounds"
    if info.get("timeout"):
        return "timeout"
    return "other_failure"


def summarize_records(records: list[dict]) -> dict:
    """Create aggregate navigation metrics without fabricating missing data."""
    attempted = len(records)
    if not attempted:
        raise ValueError("at least one evaluation record is required")
    successful = sum(item["success"] for item in records)
    collisions = sum(item["collision"] for item in records)
    timeouts = sum(item["timeout"] for item in records)
    other = attempted - successful - collisions - timeouts
    return {
        "attempted_episodes": attempted,
        "successful_episodes": successful,
        "success_rate": successful / attempted,
        "collision_count": collisions,
        "collision_rate": collisions / attempted,
        "timeout_count": timeouts,
        "timeout_rate": timeouts / attempted,
        "other_failure_count": other,
        "mean_minimum_goal_distance_m": float(np.mean([
            item["minimum_goal_distance_m"] for item in records
        ])),
        "mean_final_goal_distance_m": float(np.mean([
            item["final_goal_distance_m"] for item in records
        ])),
        "mean_flight_duration_s": float(np.mean([
            item["flight_duration_s"] for item in records
        ])),
        "mean_path_length_m": float(np.mean([
            item["path_length_m"] for item in records
        ])),
    }


def create_evaluation_plots(output_dir: Path, records: list[dict]) -> list[str]:
    """Create outcome and goal-distance plots from measured episode records."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    count = len(records)
    categories = ("Success", "Collision", "Timeout", "Other Failure")
    values = (
        sum(item["success"] for item in records),
        sum(item["collision"] for item in records),
        sum(item["timeout"] for item in records),
        sum(
            not item["success"] and not item["collision"] and not item["timeout"]
            for item in records
        ),
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(categories, values)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value}\n({100 * value / count:.1f}%)",
            ha="center",
            va="bottom",
        )
    axis.set(ylabel="Episodes", title="BC closed-loop outcomes")
    axis.set_ylim(0, max(values + (1,)) * 1.25)
    axis.grid(axis="y", alpha=0.3)
    outcome_path = plot_dir / "closed_loop_outcomes.png"
    figure.tight_layout()
    figure.savefig(outcome_path, dpi=160)
    plt.close(figure)

    indexes = np.arange(1, count + 1)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        indexes,
        [item["minimum_goal_distance_m"] for item in records],
        "o-",
        label="Minimum",
    )
    axis.plot(
        indexes,
        [item["final_goal_distance_m"] for item in records],
        "s-",
        label="Final",
    )
    axis.set(
        xlabel="Evaluation episode",
        ylabel="Goal distance (m)",
        title="BC closed-loop goal distance",
    )
    axis.grid(alpha=0.3)
    axis.legend()
    goal_path = plot_dir / "goal_distance_by_episode.png"
    figure.tight_layout()
    figure.savefig(goal_path, dpi=160)
    plt.close(figure)
    return [str(outcome_path.resolve()), str(goal_path.resolve())]


def run_closed_loop_evaluation(
    *,
    environment: ClosedLoopEnvironment,
    policy: Callable[[dict], np.ndarray],
    seeds: list[int],
    output_dir: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    encoder_checkpoint: Path,
    encoder_sha256: str,
    dataset_root: Path,
    progress_interval_s: float = 5.0,
) -> dict:
    """Run policy-only episodes and save per-episode and aggregate evidence."""
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("evaluation seeds must be non-empty and unique")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output_dir}")
    output_dir.mkdir(parents=True)
    records = []
    display = EvaluationProgress(len(seeds), time.monotonic())
    try:
        for episode_index, seed in enumerate(seeds, start=1):
            observation, reset_info = environment.reset(seed=seed)
            initial_distance = float(
                reset_info.get(
                    "goal_distance_m",
                    np.asarray(observation["state"])[4] * 10.0,
                )
            )
            minimum_distance = initial_distance
            path_length = 0.0
            previous_position = reset_info.get("position_xy")
            steps = 0
            last_print = time.monotonic()
            display.render(
                completed=episode_index - 1,
                seed=seed,
                state="TRACKING",
                goal_distance=initial_distance,
                result="running",
                records=records,
            )
            while True:
                action = policy(observation)
                observation, _, terminated, truncated, info = environment.step(action)
                steps += 1
                distance = float(info["goal_distance_m"])
                minimum_distance = min(minimum_distance, distance)
                position = info.get("position_xy")
                if position is None and "position_xy" in observation:
                    position = observation["position_xy"]
                if position is not None and previous_position is not None:
                    path_length += float(np.linalg.norm(
                        np.asarray(position) - np.asarray(previous_position)
                    ))
                if position is not None:
                    previous_position = position
                now = time.monotonic()
                if now - last_print >= progress_interval_s and not (
                    terminated or truncated
                ):
                    display.render(
                        completed=episode_index - 1,
                        seed=seed,
                        state="TRACKING",
                        goal_distance=distance,
                        result="running",
                        records=records,
                    )
                    last_print = now
                if terminated or truncated:
                    break
            collision = bool(info.get("collision") or info.get("out_of_bounds"))
            record = {
                "episode": episode_index,
                "seed": seed,
                "success": bool(info.get("success")),
                "collision": collision,
                "timeout": bool(info.get("timeout") or truncated and not terminated),
                "terminal_reason": _terminal_reason(info),
                "minimum_goal_distance_m": minimum_distance,
                "final_goal_distance_m": float(info["goal_distance_m"]),
                "flight_duration_s": float(
                    info.get("sim_time_s", steps * 0.1)
                ),
                "path_length_m": path_length,
                "steps": steps,
                "control_source": CONTROL_SOURCE,
                "safety_abort": False,
                "action_blending": False,
            }
            records.append(record)
            _write_json(output_dir / "episodes" / f"episode_{episode_index:06d}.json", record)
            display.render(
                completed=episode_index,
                seed=seed,
                state="COMPLETE" if record["success"] else "FAILED",
                goal_distance=record["final_goal_distance_m"],
                result=record["terminal_reason"],
                records=records,
            )
    finally:
        environment.close()
    aggregate = summarize_records(records)
    plots = create_evaluation_plots(output_dir, records)
    result = {
        "format_version": FORMAT_VERSION,
        "created_utc": _utc_now(),
        "control_source": CONTROL_SOURCE,
        "policy_owns_control": True,
        "expert_action_calls": 0,
        "action_blending": False,
        "privileged_observation_inputs": False,
        "observation_contract": "fpv_rgb_to_frozen_latent64_plus_state8_v1.0",
        "action_contract": "normalized_body_forward_right_yaw_v1.0",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "encoder_checkpoint": str(encoder_checkpoint.resolve()),
        "encoder_sha256": encoder_sha256,
        "dataset_reference": str(dataset_root.resolve()),
        "evaluation_seeds": seeds,
        "aggregate": aggregate,
        "records": records,
        "plots": plots,
    }
    _write_json(output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./uav bc-eval",
        description=(
            "Run policy-only closed-loop evaluation of the formal BC baseline "
            "on expert-disjoint Isaac city seeds."
        ),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--dataset",
        default="artifacts/datasets/bc_expert_highrise_v1",
    )
    parser.add_argument("--encoder")
    parser.add_argument("--output")
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable_cameras", action="store_true")
    return parser


def main() -> int:
    """Provide dependency-free help; runtime execution uses the UAV wrapper."""
    _help_parser().parse_args()
    print("Run closed-loop evaluation through ./uav bc-eval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
