"""One-command, resumable canonical cylinder expert dataset collector."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Callable

from PIL import UnidentifiedImageError

from isaac.runtime.episode_scene import generate_episode_scene
from uav_ml.tools.expert_visual_qa import create_contact_sheet
from uav_ml.tools.persistent_runtime import (
    FatalRuntimeError,
    PersistentRuntimeManager,
    RecoverableAttemptError,
    stop_process,
    stop_process_group,
)
from uav_ml.tools.training_cli import (
    DatasetLocation,
    print_dataset_location,
    resolve_dataset,
)
from uav_ml.tools.validate_expert_collection import (
    COLLECTION_MANIFEST,
    DEFAULT_AUTOENCODER,
    DEFAULT_DATASET,
    validate_collection,
    validate_collection_episode,
    validate_cylinder_scene,
)


TOOL_VERSION = "expert_collection_v1.1"
LEGACY_TOOL_VERSION = "expert_collection_v1.0"
DEFAULT_BASE_SEED = 103000
DEFAULT_MAX_ATTEMPT_MULTIPLIER = 1.5
VISUAL_QA_INTERVAL = 20
TERMINAL_STATES = {"complete", "rejected", "failed"}
AUXILIARY_FIELDS = (
    "episode_id", "sample_id", "primary_image_timestamp_s",
    "observer_rgb_available", "observer_rgb_timestamp_s",
    "observer_rgb_error_s", "observer_rgb_path", "observer_rgb_status",
    "fpv_depth_available", "fpv_depth_timestamp_s", "fpv_depth_error_s",
    "fpv_depth_path", "fpv_depth_status",
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        item.stat().st_size for item in path.rglob("*") if item.is_file()
    )


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_size(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1000.0
    return f"{value:.1f} TB"


@dataclass(frozen=True)
class EpisodeOutcome:
    """Final facts returned by one collection attempt."""

    success: bool
    accepted_samples: int
    rejected_samples: int
    terminal_reason: str
    dataset_bytes: int
    failure_category: str | None = None
    validation_result: dict | None = None


def default_max_attempts(requested_episodes: int) -> int:
    """Return the documented total-attempt limit for an accepted target."""
    return max(
        requested_episodes,
        math.ceil(requested_episodes * DEFAULT_MAX_ATTEMPT_MULTIPLIER),
    )


def _failure_category(reason: str) -> str:
    """Map an episode terminal reason to a stable summary category."""
    normalized = reason.lower().replace("-", "_")
    if "collision" in normalized or "tracking" in normalized:
        return "collision_tracking"
    if "image" in normalized or "visual_qa" in normalized:
        return "image_qa"
    if any(token in normalized for token in (
        "blocked", "invalid_scene", "unusable", "no_path", "planner"
    )):
        return "blocked_scene"
    if "validation" in normalized or "dataset" in normalized:
        return "dataset_validation"
    if any(token in normalized for token in (
        "flight", "goal", "timeout", "mission", "takeoff", "landing"
    )):
        return "flight_failure"
    return "other"


def _validation_failure_category(error: Exception) -> str:
    """Separate image QA rejection from other episode dataset rejection."""
    reason = str(error).lower()
    if isinstance(error, UnidentifiedImageError) or any(
        token in reason for token in (
            "image", "jpeg", "blank", "dark", "luminance", "fpv_rgb",
            "fpv_depth", "observer_rgb",
        )
    ):
        return "image_qa"
    return "dataset_validation"


class ProgressDisplay:
    """Low-frequency progress reporter independent of ROS and sensor timing."""

    def __init__(
        self, total: int, started_monotonic: float | None = None
    ) -> None:
        """Initialize counters without starting any background updater."""
        self.total = total
        self.started = started_monotonic or time.monotonic()
        self._last_state = ""
        self._last_block = 0.0

    def transition(self, index: int, state: str) -> None:
        """Print one state transition only when the state changes."""
        if state == self._last_state:
            return
        print(f"[{index}/{self.total}] {state}", flush=True)
        self._last_state = state

    def render(
        self,
        *,
        index: int,
        seed: int,
        state: str,
        success_count: int,
        failure_count: int,
        current_samples: int,
        current_rejected: int,
        total_samples: int,
        dataset_bytes: int,
        force: bool = False,
    ) -> None:
        """Print a throttled batch snapshot from already-collected counters."""
        now = time.monotonic()
        if not force and now - self._last_block < 5.0:
            return
        self._last_block = now
        percent = min(100, int(100 * success_count / self.total))
        filled = min(30, int(30 * success_count / self.total))
        bar = "█" * filled + "-" * (30 - filled)
        elapsed = now - self.started
        eta = None
        if success_count:
            eta = elapsed / success_count * (self.total - success_count)
        print(
            "\nExpert Dataset Collection\n\n"
            f"Accepted: {success_count} / {self.total} [{bar}] {percent}%\n"
            f"Attempts: {index}\n\n"
            "Current:\n"
            f"  seed     : {seed}\n"
            f"  state    : {state}\n"
            f"  samples  : {current_samples}\n"
            f"  rejected : {current_rejected}\n\n"
            "Batch:\n"
            f"  accepted : {success_count}\n"
            f"  rejected : {failure_count}\n\n"
            "Dataset:\n"
            f"  samples  : {total_samples + current_samples}\n"
            f"  size     : {_format_size(dataset_bytes)}\n\n"
            f"Elapsed : {_format_duration(elapsed)}\n"
            f"ETA     : {_format_duration(eta)}\n",
            flush=True,
        )


class CollectionManifestStore:
    """Atomic source of truth for planning, interruption, and resume."""

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root.resolve()
        self.path = self.dataset_root / COLLECTION_MANIFEST
        self.data: dict = {}

    @staticmethod
    def _episode_entry(
        index: int,
        base_seed: int,
        fixed_seed: int | None = None,
    ) -> dict:
        """Build one deterministic append-only attempt entry."""
        return {
            "index": index,
            "attempt_index": index,
            "episode_id": f"episode_{index:06d}",
            "seed": base_seed + index if fixed_seed is None else fixed_seed,
            "status": "pending",
            "accepted": None,
            "success": None,
            "accepted_samples": 0,
            "rejected_samples": 0,
            "failure_category": None,
            "terminal_reason": "",
        }

    @staticmethod
    def _run_entry(
        run_number: int,
        requested_episodes: int,
        max_attempts: int,
        *,
        status: str,
    ) -> dict:
        """Build an audit record for one accepted-target invocation."""
        return {
            "run_number": run_number,
            "requested_accepted_episodes": requested_episodes,
            "requested_episodes": requested_episodes,
            "max_attempts": max_attempts,
            "status": status,
            "created_utc": _utc_now(),
        }

    def create(
        self,
        episodes: int,
        base_seed: int,
        max_attempts: int | None = None,
        fixed_seed: int | None = None,
    ) -> dict:
        """Create a new accepted-target manifest with no attempted seeds."""
        if self.dataset_root.exists():
            raise FileExistsError(
                f"refusing to overwrite existing dataset: {self.dataset_root}"
            )
        self.dataset_root.mkdir(parents=True)
        attempt_limit = max_attempts or default_max_attempts(episodes)
        self.data = {
            "tool_version": TOOL_VERSION,
            "manifest_version": 2,
            "dataset_name": "bc_expert_cylinder_v1",
            "dataset_root": str(self.dataset_root),
            "target_episodes": episodes,
            "requested_accepted_episodes": episodes,
            "max_attempts": attempt_limit,
            "base_seed": base_seed,
            "visual_qa_interval": VISUAL_QA_INTERVAL,
            "status": "prepared",
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "episodes": [],
            "accepted_episode_ids": [],
            "collection_runs": [
                self._run_entry(1, episodes, attempt_limit, status="prepared")
            ],
            "active_run_number": 1,
            "visual_qa": [],
        }
        if fixed_seed is not None:
            self.data["fixed_seed"] = fixed_seed
        self.save()
        return self.data

    def load_for_collection(
        self,
        episodes: int,
        resume: bool,
        max_attempts: int | None = None,
    ) -> dict:
        """Load an accepted target, migrating the former fixed-attempt plan."""
        if not self.path.is_file():
            raise FileNotFoundError(
                f"resume manifest does not exist: {self.path}"
            )
        self.data = _read_json(self.path)
        version = self.data.get("tool_version")
        if version not in {TOOL_VERSION, LEGACY_TOOL_VERSION}:
            raise ValueError("collection manifest tool version mismatch")
        entries = self.data.get("episodes")
        if not isinstance(entries, list):
            raise ValueError("collection manifest attempt history is invalid")
        if self.data.get("manifest_version") != 2:
            for index, entry in enumerate(entries, start=1):
                entry.setdefault("index", index)
                entry["attempt_index"] = int(entry["index"])
                if entry.get("status") == "failed":
                    entry["status"] = "rejected"
                accepted = (
                    entry.get("status") == "complete"
                    and entry.get("success") is True
                )
                entry["accepted"] = accepted
                if not accepted and entry.get("status") in TERMINAL_STATES:
                    entry["failure_category"] = _failure_category(
                        str(entry.get("terminal_reason", ""))
                    )
            self.data.update({
                "tool_version": TOOL_VERSION,
                "manifest_version": 2,
                "requested_accepted_episodes": episodes,
                "target_episodes": episodes,
            })
        expected_ids = [
            f"episode_{index:06d}" for index in range(1, len(entries) + 1)
        ]
        if [entry.get("episode_id") for entry in entries] != expected_ids:
            raise ValueError("collection manifest episode IDs are invalid")
        seeds = [entry.get("seed") for entry in entries]
        if len(set(seeds)) != len(entries):
            raise ValueError("collection manifest contains duplicate seeds")
        accepted = sum(
            entry.get("status") == "complete"
            and entry.get("success") is True
            for entry in entries
        )
        existing_target = int(
            self.data.get("requested_accepted_episodes", accepted)
        )
        if episodes < accepted:
            raise ValueError(
                f"--episodes target {episodes} is below {accepted} existing "
                "accepted episodes"
            )
        if episodes < existing_target and accepted < existing_target:
            raise ValueError(
                f"cannot reduce unfinished accepted target {existing_target}"
            )
        if not resume and self.data.get("status") not in {"complete", "prepared"}:
            raise ValueError(
                "collection has an unfinished run; rerun with --resume"
            )
        attempt_limit = max_attempts or default_max_attempts(episodes)
        if attempt_limit < len(entries):
            raise ValueError(
                f"--max-attempts {attempt_limit} is below existing attempt "
                f"history ({len(entries)})"
            )
        runs = self.data.setdefault("collection_runs", [])
        run_number = max(
            (int(item.get("run_number", 0)) for item in runs), default=0
        ) + 1
        runs.append(self._run_entry(
            run_number, episodes, attempt_limit, status="prepared"
        ))
        self.data["active_run_number"] = run_number
        self.data["target_episodes"] = episodes
        self.data["requested_accepted_episodes"] = episodes
        self.data["max_attempts"] = attempt_limit
        self.data.pop("completed_utc", None)
        self.data.pop("validation", None)
        self.save()
        return self.data

    def next_attempt(self) -> dict:
        """Return a resumable pending attempt or append the next seed."""
        for entry in self.data["episodes"]:
            if entry.get("status") == "pending":
                return entry
        index = len(self.data["episodes"]) + 1
        entry = self._episode_entry(
            index,
            int(self.data["base_seed"]),
            self.data.get("fixed_seed"),
        )
        self.data["episodes"].append(entry)
        self.save()
        return entry

    def save(self) -> None:
        """Persist the current manifest atomically."""
        self.data["updated_utc"] = _utc_now()
        _atomic_json(self.path, self.data)

    def set_collection_state(self, state: str, detail: str = "") -> None:
        """Update the batch-level lifecycle state."""
        self.data["status"] = state
        self.data["status_detail"] = detail
        active_number = self.data.get("active_run_number")
        for run in self.data.get("collection_runs", []):
            if run.get("run_number") != active_number:
                continue
            run["status"] = state
            run["status_detail"] = detail
            if state == "collecting":
                run.setdefault("started_utc", _utc_now())
            elif state == "complete":
                run["completed_utc"] = _utc_now()
            break
        self.save()

    def update_episode(self, index: int, **changes: object) -> dict:
        """Update one planned episode and persist immediately."""
        entry = self.data["episodes"][index - 1]
        entry.update(changes)
        self.save()
        return entry

    def completed_counts(self) -> tuple[int, int, int]:
        """Return accepted, rejected, and accepted-sample totals."""
        accepted = [
            entry for entry in self.data["episodes"]
            if entry.get("status") == "complete"
            and entry.get("success") is True
        ]
        rejected = [
            entry for entry in self.data["episodes"]
            if entry.get("status") in {"rejected", "failed"}
        ]
        samples = sum(
            int(entry.get("accepted_samples", 0)) for entry in accepted
        )
        return len(accepted), len(rejected), samples


class SubprocessBackend:
    """Own one persistent runtime and one fresh PX4 per attempt."""

    produces_dataset = True

    def __init__(self, repository_root: Path, timeout_s: int = 180) -> None:
        self.repository_root = repository_root.resolve()
        self.timeout_s = timeout_s
        self.uav = self.repository_root / "uav"
        self.runtime: PersistentRuntimeManager | None = None
        self._flight: subprocess.Popen | None = None
        self._flight_stream = None
        self._job_started = False
        self._last_attempt_evidence: dict = {}

    def preflight(self) -> None:
        """Verify dependencies and exclusive ownership before runtime."""
        if not self.uav.is_file() or not os.access(self.uav, os.X_OK):
            raise RuntimeError(f"uav command is not executable: {self.uav}")
        probe = PersistentRuntimeManager(
            self.repository_root,
            self.repository_root / "run_logs" / "expert-preflight-unused",
        )
        probe.preflight()

    def start_job(self, runtime_root: Path) -> dict:
        """Start and verify the one job-level Isaac/XRCE runtime."""
        if self._job_started:
            raise RuntimeError("expert persistent runtime already started")
        self.runtime = PersistentRuntimeManager(
            self.repository_root,
            runtime_root / "job_runtime",
            runtime_timeout_s=120.0,
        )
        self.runtime.preflight()
        evidence = self.runtime.start_job()
        self._job_started = True
        return evidence

    @property
    def job_evidence(self) -> dict:
        """Return additive job-level runtime evidence for the manifest."""
        return {} if self.runtime is None else self.runtime.job_evidence

    @property
    def attempt_evidence(self) -> dict:
        """Return additive evidence from the most recent attempt."""
        if self.runtime is not None:
            return self.runtime.attempt_evidence
        return dict(self._last_attempt_evidence)

    def run_episode(
        self,
        *,
        index: int,
        episode_id: str,
        seed: int,
        dataset_root: Path,
        runtime_dir: Path,
        progress: Callable[[dict], None],
    ) -> int:
        """Reset, apply scene, start fresh PX4, and run one ROS graph."""
        if self.runtime is None or not self._job_started:
            raise FatalRuntimeError("persistent runtime job was not started")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.runtime.prepare_attempt(index, runtime_dir)
        scene = self.runtime.prepare_scene(
            index, episode_id, seed, runtime_dir
        )
        self.runtime.start_px4(index, runtime_dir)
        self.runtime.probe_px4(index, runtime_dir)
        flight_log = runtime_dir / "flight.log"
        self._flight_stream = flight_log.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment["UAV_OFFLINE_TIMEOUT_SECONDS"] = str(self.timeout_s)
        environment["UAV_EXPERT_SCENE_PREPARED"] = "1"
        environment["UAV_EXPERT_RUNTIME_GENERATION"] = str(index)
        environment["UAV_EXPERT_SCENE_REVISION"] = str(
            scene["scene_revision"]
        )
        boundary = scene.get("scene_camera_boundary") or {}
        environment["UAV_EXPERT_MIN_FPV_FRAME"] = str(
            boundary.get("fpv_rgb_frame_count", 0)
        )
        environment["UAV_EXPERT_MIN_OBSERVER_FRAME"] = str(
            boundary.get("observer_rgb_frame_count", 0)
        )
        environment["UAV_EXPERT_MIN_DEPTH_FRAME"] = str(
            boundary.get("fpv_depth_frame_count", 0)
        )
        try:
            self._flight = subprocess.Popen(
                [
                    str(self.uav),
                    "expert-run-episode",
                    episode_id,
                    str(seed),
                    str(dataset_root),
                ],
                cwd=self.repository_root,
                env=environment,
                stdout=self._flight_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            self._flight_stream.close()
            self._flight_stream = None
            raise RecoverableAttemptError(
                f"could not start ROS episode graph: {error}"
            ) from error
        self.runtime.record_attempt_evidence(
            flight_process_group_id=self._flight.pid,
            flight_log=str(flight_log.resolve()),
        )
        progress_path = dataset_root / episode_id / "progress.json"
        deadline = time.monotonic() + self.timeout_s + 30.0
        while self._flight.poll() is None:
            self.runtime.assert_job_alive()
            self.runtime.assert_px4_alive()
            if time.monotonic() >= deadline:
                stop_process(self._flight)
                return 124
            if progress_path.is_file():
                try:
                    progress(_read_json(progress_path))
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            time.sleep(1.0)
        if progress_path.is_file():
            try:
                progress(_read_json(progress_path))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        return int(self._flight.returncode or 0)

    def cleanup_episode(self, index: int, runtime_dir: Path) -> None:
        """Stop the ROS graph, backend listener, and current PX4 only."""
        if self._flight is not None:
            flight_group = self._flight.pid
            flight_shutdown = stop_process(self._flight)
            group_shutdown = stop_process_group(flight_group)
            if self.runtime is not None:
                self.runtime.record_attempt_evidence(
                    flight_shutdown=flight_shutdown,
                    flight_group_shutdown=group_shutdown,
                    flight_processes_remaining=group_shutdown.get(
                        "remaining", []
                    ),
                )
            self._flight = None
        if self._flight_stream is not None:
            self._flight_stream.close()
            self._flight_stream = None
        if self.runtime is not None:
            try:
                self.runtime.stop_episode(index, runtime_dir)
            finally:
                self._last_attempt_evidence = self.runtime.attempt_evidence

    def cleanup(self) -> None:
        """Stop episode resources and the one job-level runtime."""
        if self._flight is not None:
            stop_process(self._flight)
            self._flight = None
        if self._flight_stream is not None:
            self._flight_stream.close()
            self._flight_stream = None
        if self.runtime is not None:
            self.runtime.cleanup_job()
        self._job_started = False


class DryRunBackend:
    """Offline fixture that never starts ROS/PX4 or writes formal data."""

    produces_dataset = False

    def preflight(self) -> None:
        """Require no live dependencies for the offline fixture."""
        return None

    def start_job(self, runtime_root: Path) -> dict:
        """Return deterministic job evidence without starting processes."""
        return {
            "dry_run": True,
            "isaac_pid": None,
            "xrce_pid": None,
            "isaac_restart_count": 0,
            "camera_recreated": False,
        }

    @property
    def job_evidence(self) -> dict:
        """Return the intentionally empty dry-run runtime evidence."""
        return {"dry_run": True}

    @property
    def attempt_evidence(self) -> dict:
        """Return deterministic dry-run attempt evidence."""
        return {
            "dry_run": True,
            "reset_success": True,
            "fresh_dds_verified": True,
            "camera_recreated": False,
        }

    def run_episode(
        self,
        *,
        index: int,
        episode_id: str,
        seed: int,
        dataset_root: Path,
        runtime_dir: Path,
        progress: Callable[[dict], None],
    ) -> int:
        """Emit deterministic lifecycle progress without starting a flight."""
        for state, accepted, rejected in (
            ("OFFBOARD", 0, 2),
            ("ARMED", 0, 4),
            ("TRACKING", 21, 7),
            ("GOAL_HOLD", 42, 9),
            ("LANDING", 42, 9),
            ("COMPLETE", 42, 9),
        ):
            progress({
                "state": state,
                "accepted_samples": accepted,
                "rejected_samples": rejected,
            })
        return 0

    def cleanup(self) -> None:
        """Perform the fixture's intentionally empty cleanup."""
        return None

    def cleanup_episode(self, index: int, runtime_dir: Path) -> None:
        """Perform the fixture's intentionally empty attempt cleanup."""
        return None


class ExpertCollector:
    """Coordinate episodes, resume, validation, QA, and cleanup."""

    def __init__(
        self,
        *,
        repository_root: Path,
        dataset_root: Path,
        episodes: int,
        max_attempts: int | None = None,
        base_seed: int = DEFAULT_BASE_SEED,
        fixed_seed: int | None = None,
        resume: bool = False,
        autoencoder: Path = DEFAULT_AUTOENCODER,
        device: str = "auto",
        backend: SubprocessBackend | DryRunBackend | None = None,
        runtime_root: Path | None = None,
    ) -> None:
        if episodes <= 0:
            raise ValueError("--episodes must be a positive integer")
        if max_attempts is not None and max_attempts < episodes:
            raise ValueError(
                "--max-attempts must be at least the requested accepted "
                "episode count"
            )
        if base_seed < 0:
            raise ValueError("--base-seed must be nonnegative")
        if fixed_seed is not None and fixed_seed < 0:
            raise ValueError("--seed must be nonnegative")
        if fixed_seed is not None and episodes != 1:
            raise ValueError("--seed requires --episodes 1")
        if fixed_seed is not None and resume:
            raise ValueError(
                "--seed starts an isolated run and cannot use --resume"
            )
        if fixed_seed is not None and max_attempts not in {None, 1}:
            raise ValueError("--seed requires --max-attempts 1")
        self.repository_root = repository_root.resolve()
        self.dataset_root = dataset_root.resolve()
        self.episodes = episodes
        self.max_attempts = (
            1 if fixed_seed is not None
            else max_attempts or default_max_attempts(episodes)
        )
        self.base_seed = base_seed
        self.fixed_seed = fixed_seed
        self.resume = resume
        self.autoencoder = autoencoder.resolve()
        self.device = device
        self.backend = backend or SubprocessBackend(self.repository_root)
        self.runtime_root = (
            runtime_root.resolve() if runtime_root else
            self.repository_root / "run_logs" / f"expert-collection_{_stamp()}"
        )
        self.store = CollectionManifestStore(self.dataset_root)
        self.display = ProgressDisplay(episodes)
        self._manifest_prepared = False

    def _prepare(self) -> None:
        if self.store.path.is_file() or self.resume:
            data = self.store.load_for_collection(
                self.episodes, self.resume, self.max_attempts
            )
            self.base_seed = int(data["base_seed"])
            self.max_attempts = int(data["max_attempts"])
            self._manifest_prepared = True
            self._recover_incomplete_episodes()
            self._ensure_rejection_records()
        else:
            self.store.create(
                self.episodes,
                self.base_seed,
                self.max_attempts,
                fixed_seed=self.fixed_seed,
            )
            self._manifest_prepared = True
        self.display = ProgressDisplay(
            int(self.store.data["target_episodes"])
        )
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        accepted, _, _ = self.store.completed_counts()
        attempted = sum(
            entry.get("status") != "pending"
            for entry in self.store.data["episodes"]
        )
        if accepted < self.episodes and attempted < self.max_attempts:
            self.backend.preflight()
            runtime_evidence = self.backend.start_job(self.runtime_root)
            self.store.data.setdefault("persistent_runtime_jobs", []).append({
                "run_number": self.store.data.get("active_run_number"),
                **runtime_evidence,
            })
            self.store.save()
        self.store.set_collection_state("collecting")

    def _update_runtime_job_evidence(self) -> None:
        """Persist the latest additive job-level lifecycle evidence."""
        if not self._manifest_prepared or not self.store.data:
            return
        jobs = self.store.data.setdefault("persistent_runtime_jobs", [])
        run_number = self.store.data.get("active_run_number")
        evidence = self.backend.job_evidence
        for job in reversed(jobs):
            if job.get("run_number") == run_number:
                job.update(evidence)
                break
        self.store.data["persistent_runtime"] = evidence
        self.store.save()

    def _recover_incomplete_episodes(self) -> None:
        recovery_root = self.runtime_root / "recovered_incomplete"
        for entry in self.store.data["episodes"]:
            if entry.get("status") in TERMINAL_STATES:
                episode_dir = self.dataset_root / entry["episode_id"]
                if (
                    self.backend.produces_dataset
                    and entry.get("status") == "complete"
                    and not (
                        episode_dir / "episode.json"
                    ).is_file()
                ):
                    raise FileNotFoundError(
                        "completed episode metadata is missing: "
                        f"{episode_dir / 'episode.json'}"
                    )
                continue
            episode_dir = self.dataset_root / entry["episode_id"]
            if episode_dir.exists():
                recovery_root.mkdir(parents=True, exist_ok=True)
                destination = recovery_root / (
                    f"{entry['episode_id']}_{_stamp()}"
                )
                shutil.move(str(episode_dir), destination)
                entry["recovered_incomplete_path"] = str(destination)
            entry.update({
                "status": "pending",
                "accepted": None,
                "success": None,
                "accepted_samples": 0,
                "rejected_samples": 0,
                "failure_category": None,
                "terminal_reason": "",
            })
        self._reconcile_dataset_manifest()
        self.store.save()

    def _ensure_rejection_records(self) -> None:
        """Add evidence indexes when migrating already-rejected v1 attempts."""
        for entry in self.store.data["episodes"]:
            if entry.get("status") not in {"rejected", "failed"}:
                continue
            existing = entry.get("rejection_record")
            if existing and Path(str(existing)).is_file():
                continue
            outcome = EpisodeOutcome(
                success=False,
                accepted_samples=0,
                rejected_samples=int(entry.get("rejected_samples", 0)),
                terminal_reason=str(entry.get("terminal_reason", "")),
                dataset_bytes=int(entry.get("dataset_bytes", 0)),
                failure_category=str(
                    entry.get("failure_category") or "other"
                ),
                validation_result=entry.get("validation_result"),
            )
            runtime_dir = Path(str(
                entry.get("log_path")
                or self.runtime_root / str(entry["episode_id"])
            ))
            path = self._record_rejection(entry, outcome, runtime_dir)
            entry["status"] = "rejected"
            entry["rejection_record"] = str(path.resolve())
        self.store.save()

    def _reconcile_dataset_manifest(self) -> None:
        accepted_entries = [
            entry for entry in self.store.data["episodes"]
            if entry.get("status") == "complete"
            and entry.get("success") is True
        ]
        self.store.data["accepted_episode_ids"] = [
            entry["episode_id"] for entry in accepted_entries
        ]
        self.store.save()
        path = self.dataset_root / "dataset_manifest.json"
        if not path.is_file():
            return
        manifest = _read_json(path)
        manifest.update({
            "episodes": [entry["episode_id"] for entry in accepted_entries],
            "episode_count": len(accepted_entries),
            "sample_count": sum(
                int(entry.get("accepted_samples", 0))
                for entry in accepted_entries
            ),
            "status": "collecting",
        })
        manifest.pop("validation", None)
        _atomic_json(path, manifest)

    def _live_progress(self, index: int, seed: int, payload: dict) -> None:
        state = str(payload.get("state", "RUNNING"))
        accepted = int(payload.get("accepted_samples", 0))
        rejected = int(payload.get("rejected_samples", 0))
        success_count, failure_count, total_samples = (
            self.store.completed_counts()
        )
        completed_bytes = sum(
            int(entry.get("dataset_bytes", 0))
            for entry in self.store.data["episodes"]
            if entry.get("status") in TERMINAL_STATES
        )
        current_dir = self.dataset_root / f"episode_{index:06d}"
        self.display.transition(index, state)
        self.display.render(
            index=index,
            seed=seed,
            state=state,
            success_count=success_count,
            failure_count=failure_count,
            current_samples=accepted,
            current_rejected=rejected,
            total_samples=total_samples,
            dataset_bytes=completed_bytes + _directory_size(current_dir),
        )

    def _finalize_real_episode(
        self, episode_id: str, command_status: int
    ) -> EpisodeOutcome:
        episode_dir = self.dataset_root / episode_id
        evidence_path = episode_dir / "flight_evidence.json"
        if not episode_dir.is_dir() or not evidence_path.is_file():
            raise RuntimeError(
                f"{episode_id}: recorder/evidence missing after status "
                f"{command_status}"
            )
        from uav_ml.tools.finalize_expert_episode import finalize

        try:
            finalized = finalize(
                self.dataset_root,
                episode_id,
                evidence_path,
                expected_success=None,
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise RuntimeError(
                f"{episode_id}: unsafe or corrupt recorder/flight evidence: "
                f"{error}"
            ) from error
        try:
            validation = validate_collection_episode(
                self.dataset_root,
                self.autoencoder,
                episode_id,
                device=self.device,
            )
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{episode_id}: corrupt JSON output during validation: {error}"
            ) from error
        except (FileNotFoundError, PermissionError) as error:
            raise RuntimeError(
                f"{episode_id}: required recorder output unavailable: {error}"
            ) from error
        except (OSError, ValueError) as error:
            if isinstance(error, OSError) and not isinstance(
                error, UnidentifiedImageError
            ):
                raise RuntimeError(
                    f"{episode_id}: recorder output I/O corruption: {error}"
                ) from error
            category = _validation_failure_category(error)
            reason = f"dataset_validation: {error}"
            return EpisodeOutcome(
                success=False,
                accepted_samples=0,
                rejected_samples=int(
                    finalized.get("rejected_sample_count", 0)
                ),
                terminal_reason=reason,
                dataset_bytes=int(finalized.get(
                    "episode_disk_usage_bytes", _directory_size(episode_dir)
                )),
                failure_category=category,
                validation_result={"valid": False, "error": str(error)},
            )
        reason = str(finalized.get("terminal_reason", ""))
        success = bool(finalized["success"])
        return EpisodeOutcome(
            success=success,
            accepted_samples=int(validation["sample_count"]),
            rejected_samples=int(finalized.get("rejected_sample_count", 0)),
            terminal_reason=reason,
            dataset_bytes=int(finalized.get(
                "episode_disk_usage_bytes", _directory_size(episode_dir)
            )),
            failure_category=None if success else _failure_category(reason),
            validation_result=validation,
        )

    @staticmethod
    def _dry_run_outcome() -> EpisodeOutcome:
        return EpisodeOutcome(
            success=True,
            accepted_samples=42,
            rejected_samples=9,
            terminal_reason="dry_run_complete",
            dataset_bytes=0,
            validation_result={"valid": True, "dry_run": True},
        )

    def _record_invalid_scene(
        self,
        episode_id: str,
        seed: int,
        error: Exception,
        scene: dict | None,
    ) -> EpisodeOutcome:
        """Record a safe no-flight failure and allow the next seed to run."""
        reason = f"invalid_scene: {error}"
        if not self.backend.produces_dataset:
            return EpisodeOutcome(
                False, 0, 0, reason, 0, "blocked_scene",
                {"valid": False, "error": str(error)},
            )
        from uav_ml.tools.validate_expert_dataset import (
            CSV_FIELDS,
            DATASET_VERSION,
            contract_manifest,
        )

        episode_dir = self.dataset_root / episode_id
        episode_dir.mkdir(parents=True)
        (episode_dir / "fpv_rgb").mkdir()
        with (episode_dir / "samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            csv.DictWriter(stream, fieldnames=CSV_FIELDS).writeheader()
        with (episode_dir / "auxiliary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            csv.DictWriter(stream, fieldnames=AUXILIARY_FIELDS).writeheader()
        scene_metadata = scene or {
            "episode_id": episode_id,
            "random_seed": seed,
            "generator": "canonical_cylinder_scene_generator_v1",
            "mode": "normal",
            "validation_error": str(error),
        }
        episode = {
            "dataset_version": DATASET_VERSION,
            "episode_id": episode_id,
            "random_seed": seed,
            "scene_configuration": scene_metadata,
            "started_utc": _utc_now(),
            "completed_utc": _utc_now(),
            "status": "failed",
            "success": False,
            "failure": reason,
            "terminal_reason": reason,
            "flight_duration_s": 0.0,
            "sample_count": 0,
            "rejected_sample_count": 0,
            "rejections_by_reason": {},
            "observed_sampling_rate_hz": 0.0,
            "path_length_m": 0.0,
            "astar_path_information": {
                "validated_path": None,
                "planner_status": "scene rejected before runtime start",
            },
            "final_tracking_goal_distance_m": None,
            "goal_distance_m": None,
            "synchronization_statistics_s": {
                "state": {"mean": None, "p95": None, "max": None},
                "expert_action": {"mean": None, "p95": None, "max": None},
            },
            "available_sensor_streams": {
                "fpv_rgb": {
                    "required": True,
                    "accepted": 0,
                    "received": 0,
                    "observed_rate_hz": 0.0,
                },
                "observer_rgb": {
                    "required": False,
                    "matched": 0,
                    "received": 0,
                    "observed_rate_hz": 0.0,
                },
                "fpv_depth": {
                    "required": False,
                    "matched": 0,
                    "received": 0,
                    "observed_rate_hz": 0.0,
                },
            },
            "maximum_state_image_error_s": None,
            "maximum_action_image_error_s": None,
            "safe_terminal_evidence": {
                "landed": True,
                "disarmed": True,
                "failsafe": False,
                "basis": "scene rejected before runtime start",
            },
            "timeline": [],
        }
        episode_path = episode_dir / "episode.json"
        _atomic_json(episode_path, episode)
        episode["episode_disk_usage_bytes"] = _directory_size(episode_dir)
        _atomic_json(episode_path, episode)
        _atomic_json(episode_dir / "progress.json", {
            "episode_id": episode_id,
            "state": "DATASET_VALIDATION_PENDING",
            "accepted_samples": 0,
            "rejected_samples": 0,
        })
        manifest_path = self.dataset_root / "dataset_manifest.json"
        manifest = (
            _read_json(manifest_path)
            if manifest_path.is_file() else contract_manifest()
        )
        episode_ids = list(manifest.get("episodes", []))
        if episode_id not in episode_ids:
            episode_ids.append(episode_id)
        manifest.update({
            "collection_mode": "batch",
            "episodes": episode_ids,
            "episode_count": len(episode_ids),
            "sample_count": int(manifest.get("sample_count", 0)),
            "status": "collecting",
        })
        _atomic_json(manifest_path, manifest)
        validation = validate_collection_episode(
            self.dataset_root,
            self.autoencoder,
            episode_id,
            device=self.device,
        )
        return EpisodeOutcome(
            False,
            int(validation["sample_count"]),
            0,
            reason,
            _directory_size(episode_dir),
            "blocked_scene",
            validation,
        )

    def _visual_qa(
        self, through_accepted: int, source_episode: str
    ) -> str | None:
        if through_accepted % VISUAL_QA_INTERVAL:
            return None
        entry = {
            "through_accepted_episode": through_accepted,
            "through_episode": through_accepted,
            "status": "pending",
            "path": f"visual_qa/contact_sheet_{through_accepted:06d}.jpg",
        }
        self.store.data.setdefault("visual_qa", []).append(entry)
        self.store.save()
        try:
            result = create_contact_sheet(
                self.dataset_root,
                through_accepted,
                source_episode=source_episode,
            )
        except (UnidentifiedImageError, ValueError) as error:
            entry.update({"status": "failed", "error": str(error)})
            self.store.save()
            return str(error)
        except Exception as error:
            entry.update({
                "status": "infrastructure_failure",
                "error": str(error),
            })
            self.store.save()
            raise
        else:
            entry.update({
                "status": "complete",
                "source_episode": result["source_episode"],
                "path": result["contact_sheet"],
            })
        self.store.save()
        return None

    def _record_rejection(
        self, entry: dict, outcome: EpisodeOutcome, runtime_dir: Path
    ) -> Path:
        """Write a stable evidence index without moving episode artifacts."""
        index = int(entry["attempt_index"])
        episode_id = str(entry["episode_id"])
        episode_dir = self.dataset_root / episode_id
        validation_path = episode_dir / "validation.json"
        evidence_path = episode_dir / "flight_evidence.json"
        record = {
            "attempt_index": index,
            "episode_index": index,
            "episode_id": episode_id,
            "seed": int(entry["seed"]),
            "failure_category": outcome.failure_category
            or _failure_category(outcome.terminal_reason),
            "failure_reason": outcome.terminal_reason,
            "episode_json": (
                str((episode_dir / "episode.json").resolve())
                if (episode_dir / "episode.json").is_file() else None
            ),
            "flight_evidence": (
                str(evidence_path.resolve()) if evidence_path.is_file()
                else None
            ),
            "validation_result": outcome.validation_result,
            "validation_path": (
                str(validation_path.resolve())
                if validation_path.is_file() else None
            ),
            "log_path": str(runtime_dir.resolve()),
            "recorded_utc": _utc_now(),
        }
        path = (
            self.dataset_root / "rejected_attempts"
            / f"attempt_{index:06d}.json"
        )
        _atomic_json(path, record)
        return path

    def _summary(
        self, status: str, infrastructure_failures: int = 0
    ) -> dict:
        """Persist and print the collection outcome for people and tooling."""
        accepted, rejected, samples = self.store.completed_counts()
        attempted = sum(
            entry.get("status") != "pending"
            for entry in self.store.data["episodes"]
        )
        reasons = Counter(
            str(entry.get("failure_category") or "other")
            for entry in self.store.data["episodes"]
            if entry.get("status") in {"rejected", "failed"}
        )
        infrastructure_count = max(
            infrastructure_failures,
            len(self.store.data.get("infrastructure_failures", [])),
        )
        summary = {
            "status": status,
            "complete": accepted >= self.episodes,
            "requested_accepted_episodes": self.episodes,
            "max_attempts": self.max_attempts,
            "attempted": attempted,
            "accepted": accepted,
            "rejected": rejected,
            "rejected_reasons": dict(sorted(reasons.items())),
            "infrastructure_failures": infrastructure_count,
            "accepted_samples_total": samples,
            "updated_utc": _utc_now(),
        }
        self.store.data["summary"] = summary
        self.store.save()
        _atomic_json(self.dataset_root / "collection_summary.json", summary)
        print(
            "\nCollection Summary\n\n"
            f"Requested accepted episodes : {self.episodes}\n"
            f"Maximum attempts            : {self.max_attempts}\n"
            f"Attempted                   : {attempted}\n"
            f"Accepted                    : {accepted}\n"
            f"Rejected                    : {rejected}\n\n"
            "Rejected reasons:\n"
            + "".join(
                f"  {name:<26}: {count}\n"
                for name, count in sorted(reasons.items())
            )
            + f"\nInfrastructure failures     : {infrastructure_count}\n"
            f"Status                      : {status}\n",
            flush=True,
        )
        return summary

    def run(self) -> dict:
        """Run or resume collection and always clean child processes."""
        current_index: int | None = None
        try:
            self._prepare()
            while True:
                accepted_count, _, _ = self.store.completed_counts()
                attempted_count = sum(
                    entry.get("status") != "pending"
                    for entry in self.store.data["episodes"]
                )
                if accepted_count >= self.episodes:
                    break
                if attempted_count >= self.max_attempts:
                    self._reconcile_dataset_manifest()
                    validation = {
                        "valid": False,
                        "complete": False,
                        "dataset_incomplete": True,
                        "reason": "maximum attempts reached before target",
                        "requested_accepted_episodes": self.episodes,
                        "attempted": attempted_count,
                        "accepted": accepted_count,
                    }
                    self.store.data["validation"] = validation
                    self.store.set_collection_state(
                        "incomplete_max_attempts",
                        "maximum attempts reached before accepted target",
                    )
                    summary = self._summary("incomplete_max_attempts")
                    return {**validation, "summary": summary}

                entry = self.store.next_attempt()
                current_index = int(entry["index"])
                episode_id = str(entry["episode_id"])
                seed = int(entry["seed"])
                runtime_dir = self.runtime_root / episode_id
                self.store.update_episode(
                    current_index,
                    status="running",
                    accepted=None,
                    attempted_utc=entry.get("attempted_utc") or _utc_now(),
                    log_path=str(runtime_dir.resolve()),
                )
                scene = None
                try:
                    scene = generate_episode_scene(
                        episode_id, seed, 0.0, 0.0
                    )
                    validate_cylinder_scene(scene, episode_id, seed)
                except (RuntimeError, ValueError) as scene_error:
                    outcome = self._record_invalid_scene(
                        episode_id, seed, scene_error, scene
                    )
                    self.store.update_episode(
                        current_index,
                        status="rejected",
                        accepted=False,
                        success=False,
                        accepted_samples=outcome.accepted_samples,
                        rejected_samples=0,
                        dataset_bytes=outcome.dataset_bytes,
                        failure_category=outcome.failure_category,
                        failure_classification=(
                            "recoverable_attempt_failure"
                        ),
                        terminal_reason=outcome.terminal_reason,
                        completed_utc=_utc_now(),
                        command_status=None,
                    )
                    rejection_path = self._record_rejection(
                        self.store.data["episodes"][current_index - 1],
                        outcome,
                        runtime_dir,
                    )
                    self.store.update_episode(
                        current_index,
                        rejection_record=str(rejection_path.resolve()),
                        validation_result=outcome.validation_result,
                    )
                    self._reconcile_dataset_manifest()
                    self.display.transition(current_index, "INVALID SCENE")
                    continue
                assert scene is not None
                scene_summary = {
                    "generator": scene["generator"],
                    "obstacle_count": len(scene["obstacles"]),
                    "direct_path_blocker_count": scene[
                        "direct_path_blocker_count"
                    ],
                }
                self.store.update_episode(
                    current_index,
                    status="running",
                    started_utc=_utc_now(),
                    scene_validation=scene_summary,
                )
                self.display.transition(current_index, "Scene ready")
                command_status: int | None = None
                outcome: EpisodeOutcome | None = None
                fatal_error: BaseException | None = None
                try:
                    command_status = self.backend.run_episode(
                        index=current_index,
                        episode_id=episode_id,
                        seed=seed,
                        dataset_root=self.dataset_root,
                        runtime_dir=runtime_dir,
                        progress=lambda payload, i=current_index, s=seed: (
                            self._live_progress(i, s, payload)
                        ),
                    )
                except RecoverableAttemptError as error:
                    reason = f"recoverable_runtime: {error}"
                    outcome = EpisodeOutcome(
                        success=False,
                        accepted_samples=0,
                        rejected_samples=0,
                        terminal_reason=reason,
                        dataset_bytes=_directory_size(
                            self.dataset_root / episode_id
                        ),
                        failure_category=_failure_category(reason),
                        validation_result={
                            "valid": False,
                            "recoverable": True,
                            "error": str(error),
                        },
                    )
                except (FatalRuntimeError, KeyboardInterrupt) as error:
                    fatal_error = error
                finally:
                    try:
                        self.backend.cleanup_episode(
                            current_index, runtime_dir
                        )
                    except RecoverableAttemptError as error:
                        if fatal_error is None:
                            reason = f"recoverable_cleanup: {error}"
                            outcome = EpisodeOutcome(
                                False, 0, 0, reason,
                                _directory_size(
                                    self.dataset_root / episode_id
                                ),
                                _failure_category(reason),
                                {
                                    "valid": False,
                                    "recoverable": True,
                                    "error": str(error),
                                },
                            )
                    except FatalRuntimeError as error:
                        if fatal_error is None:
                            fatal_error = error
                if fatal_error is not None:
                    raise fatal_error
                if outcome is None:
                    if self.backend.produces_dataset:
                        try:
                            outcome = self._finalize_real_episode(
                                episode_id, int(command_status or 0)
                            )
                        except RuntimeError as error:
                            missing_output = (
                                "recorder/evidence missing after status"
                                in str(error)
                            )
                            if not missing_output:
                                raise
                            reason = f"recoverable_episode_output: {error}"
                            outcome = EpisodeOutcome(
                                False, 0, 0, reason,
                                _directory_size(
                                    self.dataset_root / episode_id
                                ),
                                "dataset_validation",
                                {
                                    "valid": False,
                                    "recoverable": True,
                                    "error": str(error),
                                },
                            )
                    else:
                        outcome = self._dry_run_outcome()
                runtime_evidence = self.backend.attempt_evidence
                failure_classification = (
                    "accepted" if outcome.success
                    else "recoverable_attempt_failure"
                )
                runtime_evidence["failure_classification"] = (
                    failure_classification
                )
                terminal_state = "complete" if outcome.success else "rejected"
                self.store.update_episode(
                    current_index,
                    status=terminal_state,
                    accepted=outcome.success,
                    success=outcome.success,
                    accepted_samples=(
                        outcome.accepted_samples if outcome.success else 0
                    ),
                    rejected_samples=outcome.rejected_samples,
                    dataset_bytes=outcome.dataset_bytes,
                    failure_category=outcome.failure_category,
                    terminal_reason=outcome.terminal_reason,
                    validation_result=outcome.validation_result,
                    completed_utc=_utc_now(),
                    command_status=command_status,
                    runtime_evidence=runtime_evidence,
                    px4_pid=runtime_evidence.get("px4_pid"),
                    px4_start_ticks=runtime_evidence.get(
                        "px4_start_ticks"
                    ),
                    runtime_generation=runtime_evidence.get(
                        "runtime_generation"
                    ),
                    reset_success=runtime_evidence.get("reset_success"),
                    scene_revision=runtime_evidence.get("scene_revision"),
                    fresh_dds_verified=runtime_evidence.get(
                        "fresh_dds_verified"
                    ),
                    camera_identity=runtime_evidence.get("camera_identity"),
                    camera_recreated=runtime_evidence.get(
                        "camera_recreated"
                    ),
                    failure_classification=failure_classification,
                )
                if not outcome.success:
                    rejection_path = self._record_rejection(
                        self.store.data["episodes"][current_index - 1],
                        outcome,
                        runtime_dir,
                    )
                    self.store.update_episode(
                        current_index,
                        rejection_record=str(rejection_path.resolve()),
                    )
                self._reconcile_dataset_manifest()
                success_count, failure_count, total_samples = (
                    self.store.completed_counts()
                )
                qa_error = None
                if self.backend.produces_dataset and outcome.success:
                    qa_error = self._visual_qa(success_count, episode_id)
                if qa_error is not None:
                    runtime_evidence["failure_classification"] = (
                        "recoverable_attempt_failure"
                    )
                    outcome = EpisodeOutcome(
                        success=False,
                        accepted_samples=0,
                        rejected_samples=outcome.rejected_samples,
                        terminal_reason=f"visual_qa: {qa_error}",
                        dataset_bytes=outcome.dataset_bytes,
                        failure_category="image_qa",
                        validation_result=outcome.validation_result,
                    )
                    self.store.update_episode(
                        current_index,
                        status="rejected",
                        accepted=False,
                        success=False,
                        accepted_samples=0,
                        failure_category="image_qa",
                        terminal_reason=outcome.terminal_reason,
                        failure_classification=(
                            "recoverable_attempt_failure"
                        ),
                        runtime_evidence=runtime_evidence,
                    )
                    rejection_path = self._record_rejection(
                        self.store.data["episodes"][current_index - 1],
                        outcome,
                        runtime_dir,
                    )
                    self.store.update_episode(
                        current_index,
                        rejection_record=str(rejection_path.resolve()),
                    )
                    self._reconcile_dataset_manifest()
                    success_count, failure_count, total_samples = (
                        self.store.completed_counts()
                    )
                state = "ACCEPTED" if outcome.success else "REJECTED"
                self.display.transition(current_index, state)
                self.display.render(
                    index=current_index,
                    seed=seed,
                    state=state,
                    success_count=success_count,
                    failure_count=failure_count,
                    current_samples=0,
                    current_rejected=0,
                    total_samples=total_samples,
                    dataset_bytes=_directory_size(self.dataset_root),
                    force=True,
                )
            current_index = None
            if self.backend.produces_dataset:
                self._reconcile_dataset_manifest()
                validation = validate_collection(
                    self.dataset_root,
                    self.autoencoder,
                    expected_episodes=int(
                        self.store.data["target_episodes"]
                    ),
                    device=self.device,
                )
            else:
                success_count, failure_count, samples = (
                    self.store.completed_counts()
                )
                validation = {
                    "valid": True,
                    "dry_run": True,
                    "episode_count": int(
                        self.store.data["target_episodes"]
                    ),
                    "successful_episodes": success_count,
                    "failed_episodes": failure_count,
                    "accepted_samples_total": samples,
                }
            self.store.data["validation"] = validation
            self.store.data["completed_utc"] = _utc_now()
            self.store.set_collection_state("complete")
            summary = self._summary("complete")
            return {**validation, "summary": summary}
        except KeyboardInterrupt:
            if self._manifest_prepared and current_index is not None:
                self.store.update_episode(
                    current_index,
                    status="interrupted",
                    terminal_reason="operator_interrupt",
                )
            if self._manifest_prepared:
                try:
                    self._reconcile_dataset_manifest()
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
                self.store.set_collection_state(
                    "interrupted", "operator interrupt; resume is safe"
                )
                self._summary("interrupted")
            raise
        except Exception as error:
            if self._manifest_prepared and self.store.data:
                if current_index is not None:
                    entry = self.store.data["episodes"][current_index - 1]
                    if entry.get("status") not in TERMINAL_STATES:
                        runtime_evidence = self.backend.attempt_evidence
                        runtime_evidence["failure_classification"] = (
                            "fatal_job_failure"
                        )
                        self.store.update_episode(
                            current_index,
                            status="infrastructure_failure",
                            terminal_reason=str(error),
                            failure_classification="fatal_job_failure",
                            runtime_evidence=runtime_evidence,
                            px4_pid=runtime_evidence.get("px4_pid"),
                            px4_start_ticks=runtime_evidence.get(
                                "px4_start_ticks"
                            ),
                            runtime_generation=runtime_evidence.get(
                                "runtime_generation"
                            ),
                        )
                self.store.data.setdefault(
                    "infrastructure_failures", []
                ).append({
                    "attempt_index": current_index,
                    "reason": str(error),
                    "log_path": (
                        None if current_index is None else str(
                            (
                                self.runtime_root
                                / f"episode_{current_index:06d}"
                            ).resolve()
                        )
                    ),
                    "recorded_utc": _utc_now(),
                })
                self.store.save()
                try:
                    self._reconcile_dataset_manifest()
                except (OSError, ValueError, json.JSONDecodeError):
                    # The original infrastructure failure remains primary.
                    pass
                self.store.set_collection_state(
                    "stopped_infrastructure_failure", str(error)
                )
                self._summary(
                    "stopped_infrastructure_failure",
                    infrastructure_failures=1,
                )
            raise
        finally:
            self.backend.cleanup()
            try:
                self._update_runtime_job_evidence()
            except (OSError, ValueError, json.JSONDecodeError):
                # Runtime shutdown must not mask the primary collection result.
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./uav expert-collect",
        description=(
            "Collect a resumable canonical cylinder ASTAR_EXPERT BC dataset. "
            "The default dataset is artifacts/datasets/bc_expert_cylinder_v1/."
        ),
    )
    parser.add_argument(
        "--episodes",
        required=True,
        type=int,
        help=(
            "dataset-wide target number of accepted successful episodes"
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        help=(
            "total deterministic seed attempts allowed (default: ceil(1.5 "
            "times --episodes))"
        ),
    )
    parser.add_argument(
        "--dataset",
        help=(
            "dataset name under artifacts/datasets, or an explicit relative/"
            "absolute path (default: bc_expert_cylinder_v1)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume existing accepted/rejected seed history; existing accepted "
            "episodes count toward --episodes"
        ),
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help=f"seed base for a new collection (default: {DEFAULT_BASE_SEED})",
    )
    seed_group.add_argument(
        "--seed",
        type=int,
        help=(
            "run one isolated regression episode with this exact seed; "
            "requires --episodes 1"
        ),
    )
    parser.add_argument("--device", default="auto", help=argparse.SUPPRESS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "exercise the offline lifecycle in run_logs; do not start ROS/PX4"
        ),
    )
    return parser


def main() -> int:
    """Parse the formal CLI and return a shell-compatible status."""
    parser = _parser()
    args = parser.parse_args()
    if args.seed is not None and args.episodes != 1:
        parser.error("--seed requires --episodes 1")
    if args.seed is not None and args.resume:
        parser.error("--seed starts an isolated run and cannot use --resume")
    if args.dataset is not None and (args.dry_run or args.seed is not None):
        parser.error(
            "--dataset cannot be combined with isolated --dry-run/--seed modes"
        )
    repository_root = Path(__file__).resolve().parents[2]
    if args.dry_run:
        dry_root = repository_root / "run_logs" / f"expert-dry-run_{_stamp()}"
        dataset_root = dry_root / "mock_dataset"
        backend: SubprocessBackend | DryRunBackend = DryRunBackend()
        runtime_root = dry_root / "runtime"
        dataset_location = DatasetLocation(dataset_root.name, dataset_root.resolve())
    elif args.seed is not None:
        dataset_root = (
            repository_root
            / "artifacts"
            / "regressions"
            / f"expert_seed_{args.seed}_{_stamp()}"
        )
        backend = SubprocessBackend(repository_root)
        runtime_root = None
        dataset_location = DatasetLocation(dataset_root.name, dataset_root.resolve())
    else:
        dataset_location = resolve_dataset(
            args.dataset or DEFAULT_DATASET,
            must_exist=False,
            project_root=repository_root,
        )
        dataset_root = dataset_location.path
        backend = SubprocessBackend(repository_root)
        runtime_root = None
    print_dataset_location(dataset_location)
    collector = ExpertCollector(
        repository_root=repository_root,
        dataset_root=dataset_root,
        episodes=args.episodes,
        max_attempts=args.max_attempts,
        base_seed=args.base_seed,
        fixed_seed=args.seed,
        resume=args.resume,
        autoencoder=repository_root / DEFAULT_AUTOENCODER,
        device=args.device,
        backend=backend,
        runtime_root=runtime_root,
    )
    try:
        result = collector.run()
    except KeyboardInterrupt:
        print("Collection interrupted safely. Re-run with --resume.")
        return 130
    except Exception as error:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: expert collection stopped: {error}")
        return 1
    print(json.dumps(result, indent=2))
    return 0 if result.get("complete", result.get("valid", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
