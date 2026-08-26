"""Shared dataset and TensorBoard lifecycle helpers for training CLIs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_BASE = PROJECT_ROOT / "artifacts" / "datasets"


@dataclass(frozen=True)
class DatasetLocation:
    """A user-facing dataset name and its resolved absolute path."""

    name: str
    path: Path


@dataclass(frozen=True)
class EncoderSelection:
    """One provenance-validated Autoencoder checkpoint selection."""

    checkpoint: Path
    automatic: bool
    summary: dict


def resolve_dataset(
    value: str | Path,
    *,
    must_exist: bool,
    short_name: bool = True,
    project_root: Path = PROJECT_ROOT,
) -> DatasetLocation:
    """Resolve a short dataset name or a backward-compatible explicit path."""
    raw_text = str(value)
    raw = Path(value).expanduser()
    if not str(raw).strip():
        raise ValueError("dataset cannot be empty")
    dataset_base = project_root.resolve() / "artifacts" / "datasets"
    is_short_name = (
        short_name
        and not raw.is_absolute()
        and not raw_text.startswith(".")
        and len(raw.parts) == 1
        and raw.name not in {".", ".."}
    )
    if is_short_name:
        resolved = (dataset_base / raw.name).resolve()
    elif raw.is_absolute():
        resolved = raw.resolve()
    else:
        resolved = (project_root / raw).resolve()
    if must_exist and not resolved.is_dir():
        raise FileNotFoundError(f"training dataset does not exist: {resolved}")
    return DatasetLocation(name=resolved.name, path=resolved)


def print_dataset_location(location: DatasetLocation) -> None:
    print(
        f"Dataset name: {location.name}\n"
        f"Dataset path:\n{location.path}",
        flush=True,
    )


def experiment_run_directory(
    experiment: str,
    dataset: DatasetLocation,
    image_source: str,
    timestamp: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Return the canonical provenance-preserving experiment run path."""
    return (
        project_root.resolve()
        / "artifacts"
        / "experiments"
        / experiment
        / dataset.name
        / image_source
        / f"run_{timestamp}"
    )


def autoencoder_latest_path(
    dataset: DatasetLocation,
    image_source: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    return (
        project_root.resolve()
        / "artifacts"
        / "experiments"
        / "autoencoder"
        / dataset.name
        / image_source
        / "latest.json"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def publish_autoencoder_latest(index_path: Path, summary: dict) -> None:
    """Publish an AE run only after all successful artifacts exist."""
    if summary.get("run_status") != "completed":
        raise ValueError("refusing to publish an incomplete Autoencoder run")
    checkpoint = Path(summary["artifacts"]["best_checkpoint"]).resolve()
    summary_path = Path(summary["artifacts"]["summary"]).resolve()
    if not checkpoint.is_file() or not summary_path.is_file():
        raise FileNotFoundError("completed Autoencoder artifacts are missing")
    payload = {
        "format_version": "autoencoder_latest_v1.0",
        "run_status": "completed",
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_name": summary["dataset_name"],
        "dataset_path": summary["dataset_root"],
        "image_source": summary["image_source"],
        "encoder_architecture": summary["encoder_architecture"],
        "image_preprocessing": summary["image_preprocessing"],
        "latent_dimension": summary["latent_dimension"],
        "run_dir": str(summary_path.parent),
        "summary": str(summary_path),
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": _sha256(checkpoint),
    }
    _atomic_json(index_path, payload)


def validate_autoencoder_checkpoint(
    checkpoint: Path,
    dataset: DatasetLocation,
    image_source: str,
    image_preprocessing: str,
    *,
    automatic: bool,
    index: dict | None = None,
) -> EncoderSelection:
    """Require matching dataset/source/model contracts and readable artifacts."""
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"AutoEncoder checkpoint is missing: {checkpoint}")
    summary_path = checkpoint.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"AutoEncoder summary.json is missing beside checkpoint: {checkpoint}"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    checkpoint_source = metadata.get("image_source", metadata.get("camera"))
    if checkpoint_source == "fpv":
        checkpoint_source = "fpv_rgb"
    checkpoint_preprocessing = metadata.get("image_preprocessing")
    if (
        checkpoint_preprocessing is None
        and checkpoint_source == "fpv_rgb"
        and str(metadata.get("input_contract", "")).startswith("fpv_rgb_")
    ):
        checkpoint_preprocessing = image_preprocessing
    summary_status_ok = (
        summary.get("run_status") == "completed"
        or summary.get("checkpoint_reload_verified") is True
    )
    checks = {
        "completed run": summary_status_ok,
        "dataset path": Path(
            summary.get("dataset_root", metadata.get("dataset_root", ""))
        ).resolve() == dataset.path.resolve(),
        "image source": summary.get("image_source", checkpoint_source)
        == image_source == checkpoint_source,
        "encoder architecture": summary.get(
            "encoder_architecture", payload.get("model_class")
        ) == "RgbAutoencoderV0" == payload.get("model_class"),
        "preprocessing": summary.get(
            "image_preprocessing", checkpoint_preprocessing
        ) == image_preprocessing == checkpoint_preprocessing,
        "latent dimension": int(summary.get(
            "latent_dimension", payload.get("model_config", {}).get(
                "latent_dimension", -1
            )
        )) == 64 == int(payload.get("model_config", {}).get(
            "latent_dimension", -1
        )),
        "model state": isinstance(payload.get("model_state"), dict),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "AutoEncoder checkpoint provenance mismatch: " + ", ".join(failed)
        )
    if index is not None:
        if (
            index.get("run_status") != "completed"
            or Path(index.get("best_checkpoint", "")).resolve() != checkpoint
            or index.get("best_checkpoint_sha256") != _sha256(checkpoint)
            or Path(index.get("summary", "")).resolve() != summary_path.resolve()
        ):
            raise ValueError("AutoEncoder latest index is stale or inconsistent")
    return EncoderSelection(checkpoint, automatic, summary)


def select_autoencoder_checkpoint(
    dataset: DatasetLocation,
    image_source: str,
    image_preprocessing: str,
    *,
    explicit: Path | None,
    project_root: Path = PROJECT_ROOT,
) -> EncoderSelection:
    """Select explicit AE or the single provenance-indexed latest matching run."""
    if explicit is not None:
        return validate_autoencoder_checkpoint(
            explicit,
            dataset,
            image_source,
            image_preprocessing,
            automatic=False,
        )
    index_path = autoencoder_latest_path(
        dataset, image_source, project_root=project_root
    )
    if not index_path.is_file():
        raise FileNotFoundError(
            "No compatible AutoEncoder checkpoint found for:\n"
            f"dataset={dataset.name}\nimage_source={image_source}\n\n"
            "Run AE training first:\n"
            f"./uav ae-train --dataset {dataset.name} --epochs <N>"
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_name": dataset.name,
        "dataset_path": str(dataset.path.resolve()),
        "image_source": image_source,
        "encoder_architecture": "RgbAutoencoderV0",
        "image_preprocessing": image_preprocessing,
        "latent_dimension": 64,
    }
    mismatches = [key for key, value in expected.items() if index.get(key) != value]
    if mismatches:
        raise ValueError(
            "AutoEncoder latest index provenance mismatch: " + ", ".join(mismatches)
        )
    return validate_autoencoder_checkpoint(
        Path(index["best_checkpoint"]),
        dataset,
        image_source,
        image_preprocessing,
        automatic=True,
        index=index,
    )


def add_tensorboard_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="start a managed TensorBoard server (default: enabled)",
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="managed TensorBoard localhost port (default: 6006)",
    )


def _validate_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("TensorBoard port must be between 1 and 65535")


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


class TensorBoardServer:
    """Own and clean up exactly one TensorBoard subprocess."""

    def __init__(
        self,
        log_directory: Path,
        *,
        enabled: bool,
        port: int,
        host: str = "127.0.0.1",
    ) -> None:
        _validate_port(port)
        self.log_directory = log_directory.resolve()
        self.enabled = enabled
        self.port = port
        self.host = host
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> "TensorBoardServer":
        self.log_directory.mkdir(parents=True, exist_ok=True)
        if not self.enabled:
            print("TensorBoard: disabled", flush=True)
            return self
        if not _port_available(self.host, self.port):
            print(
                f"TensorBoard port {self.port} is already in use.\n"
                "Training will continue without starting a new TensorBoard server.",
                flush=True,
            )
            return self
        executable = shutil.which("tensorboard")
        if executable is None:
            print(
                "TensorBoard executable was not found. Training will continue "
                "without starting a server.\n"
                "Install it with: python3 -m pip install -r requirements-ml.txt",
                flush=True,
            )
            return self
        self.process = subprocess.Popen(
            [
                executable,
                "--logdir",
                str(self.log_directory),
                "--host",
                self.host,
                "--port",
                str(self.port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(
            "TensorBoard started\n"
            f"Log directory: {self.log_directory}\n"
            f"Server: http://{self.host}:{self.port}\n"
            f"PID: {self.process.pid}\n"
            "Remote server users must use SSH port forwarding\n"
            f"to open http://localhost:{self.port} on the local computer.",
            flush=True,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    @property
    def active(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def wait_until_interrupted(self) -> None:
        """Keep a successfully started server alive until Ctrl+C or exit."""
        if not self.active:
            return
        print(
            "Training is finished. TensorBoard is still running.\n"
            "Press Ctrl+C when you are finished viewing TensorBoard.",
            flush=True,
        )
        try:
            while self.active:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print(
                "TensorBoard shutdown requested. Training artifacts have "
                "been preserved.",
                flush=True,
            )

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5.0)
        print(
            "TensorBoard stopped. Training artifacts have been preserved.\n"
            f"Event files remain in {self.log_directory}",
            flush=True,
        )
        self.process = None
