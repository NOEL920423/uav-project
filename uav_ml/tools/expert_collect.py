"""One-command, resumable canonical cylinder expert dataset collector."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Callable

from isaac.runtime.episode_scene import generate_episode_scene
from uav_ml.tools.expert_visual_qa import create_contact_sheet
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


TOOL_VERSION = "expert_collection_v1.0"
DEFAULT_BASE_SEED = 103000
VISUAL_QA_INTERVAL = 20
TERMINAL_STATES = {"complete", "failed"}
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
        completed = success_count + failure_count
        percent = min(100, int(100 * completed / self.total))
        filled = min(30, int(30 * completed / self.total))
        bar = "█" * filled + "-" * (30 - filled)
        elapsed = now - self.started
        eta = None
        if completed:
            eta = elapsed / completed * (self.total - completed)
        print(
            "\nExpert Dataset Collection\n\n"
            f"Episodes: {index} / {self.total} [{bar}] {percent}%\n\n"
            "Current:\n"
            f"  seed     : {seed}\n"
            f"  state    : {state}\n"
            f"  samples  : {current_samples}\n"
            f"  rejected : {current_rejected}\n\n"
            "Batch:\n"
            f"  success  : {success_count}\n"
            f"  failed   : {failure_count}\n\n"
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
        """Build one deterministic append-only episode-plan entry."""
        return {
            "index": index,
            "episode_id": f"episode_{index:06d}",
            "seed": base_seed + index if fixed_seed is None else fixed_seed,
            "status": "pending",
            "success": None,
            "accepted_samples": 0,
            "rejected_samples": 0,
            "terminal_reason": "",
        }

    @staticmethod
    def _run_entry(
        run_number: int,
        requested_episodes: int,
        first_index: int,
        last_index: int,
        *,
        status: str,
    ) -> dict:
        """Build an audit record for one invocation's requested append."""
        return {
            "run_number": run_number,
            "requested_episodes": requested_episodes,
            "first_episode_index": first_index,
            "last_episode_index": last_index,
            "status": status,
            "created_utc": _utc_now(),
        }

    def create(
        self,
        episodes: int,
        base_seed: int,
        fixed_seed: int | None = None,
    ) -> dict:
        """Create a new immutable episode and seed plan."""
        if self.dataset_root.exists():
            raise FileExistsError(
                f"refusing to overwrite existing dataset: {self.dataset_root}"
            )
        self.dataset_root.mkdir(parents=True)
        self.data = {
            "tool_version": TOOL_VERSION,
            "dataset_name": "bc_expert_cylinder_v1",
            "dataset_root": str(self.dataset_root),
            "target_episodes": episodes,
            "base_seed": base_seed,
            "visual_qa_interval": VISUAL_QA_INTERVAL,
            "status": "prepared",
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "episodes": [
                self._episode_entry(index, base_seed, fixed_seed)
                for index in range(1, episodes + 1)
            ],
            "collection_runs": [
                self._run_entry(
                    1, episodes, 1, episodes, status="prepared"
                )
            ],
            "active_run_number": 1,
            "visual_qa": [],
        }
        if fixed_seed is not None:
            self.data["fixed_seed"] = fixed_seed
        self.save()
        return self.data

    def load_for_collection(self, episodes: int, resume: bool) -> dict:
        """Resume an unfinished run or append a new run to the dataset."""
        if not self.path.is_file():
            raise FileNotFoundError(
                f"resume manifest does not exist: {self.path}"
            )
        self.data = _read_json(self.path)
        if self.data.get("tool_version") != TOOL_VERSION:
            raise ValueError("collection manifest tool version mismatch")
        existing_target = int(self.data.get("target_episodes", -1))
        if existing_target <= 0:
            raise ValueError("collection manifest target is invalid")
        entries = self.data.get("episodes")
        if not isinstance(entries, list) or len(entries) != existing_target:
            raise ValueError("collection manifest episode plan is invalid")
        expected_ids = [
            f"episode_{index:06d}"
            for index in range(1, existing_target + 1)
        ]
        if [entry.get("episode_id") for entry in entries] != expected_ids:
            raise ValueError("collection manifest episode IDs are invalid")
        seeds = [entry.get("seed") for entry in entries]
        if len(set(seeds)) != existing_target:
            raise ValueError("collection manifest contains duplicate seeds")

        unfinished_entries = [
            entry for entry in entries
            if entry.get("status") not in TERMINAL_STATES
        ]
        runs = self.data.get("collection_runs")
        if runs is not None and not isinstance(runs, list):
            raise ValueError("collection manifest run history is invalid")
        active_run = None
        if runs:
            active_number = int(
                self.data.get("active_run_number", len(runs))
            )
            active_runs = [
                item for item in runs
                if int(item.get("run_number", -1)) == active_number
            ]
            if len(active_runs) != 1:
                raise ValueError("collection manifest active run is invalid")
            active_run = active_runs[0]
        unfinished_run = bool(unfinished_entries) or (
            active_run is not None and active_run.get("status") != "complete"
        )
        if not runs and not unfinished_entries:
            validation = self.data.get("validation", {})
            legacy_valid = (
                isinstance(validation, dict)
                and validation.get("valid") is True
            )
            unfinished_run = (
                self.data.get("status") != "complete" and not legacy_valid
            )

        if unfinished_run:
            if not resume:
                raise ValueError(
                    "collection has an unfinished run; rerun with --resume"
                )
            if active_run is not None:
                requested = int(active_run["requested_episodes"])
            else:
                # A v1 manifest predating run history represents one run.
                requested = existing_target
            if episodes != requested:
                raise ValueError(
                    "--episodes must match the unfinished collection run "
                    f"({requested})"
                )
            if not runs:
                self.data["collection_runs"] = [
                    self._run_entry(
                        1,
                        existing_target,
                        1,
                        existing_target,
                        status="interrupted",
                    )
                ]
                self.data["active_run_number"] = 1
                self.save()
            return self.data

        if not runs:
            # Preserve the completed legacy plan as the first historical run.
            legacy = self._run_entry(
                1,
                existing_target,
                1,
                existing_target,
                status="complete",
            )
            legacy["completed_utc"] = self.data.get(
                "completed_utc", self.data.get("updated_utc", _utc_now())
            )
            runs = [legacy]
            self.data["collection_runs"] = runs

        run_number = max(
            int(item.get("run_number", 0)) for item in runs
        ) + 1
        new_target = existing_target + episodes
        base_seed = int(self.data["base_seed"])
        new_entries = [
            self._episode_entry(index, base_seed)
            for index in range(existing_target + 1, new_target + 1)
        ]
        extended_seeds = seeds + [entry["seed"] for entry in new_entries]
        if len(set(extended_seeds)) != new_target:
            raise ValueError("episode append would create duplicate seeds")
        entries.extend(new_entries)
        runs.append(self._run_entry(
            run_number,
            episodes,
            existing_target + 1,
            new_target,
            status="prepared",
        ))
        self.data["active_run_number"] = run_number
        self.data["target_episodes"] = new_target
        self.data.setdefault("target_extensions", []).append({
            "run_number": run_number,
            "additional_episodes": episodes,
            "from_episodes": existing_target,
            "to_episodes": new_target,
            "extended_utc": _utc_now(),
        })
        self.data.pop("completed_utc", None)
        self.data.pop("validation", None)
        self.save()
        return self.data

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
        """Return success, failure, and accepted-sample totals."""
        final = [
            entry for entry in self.data["episodes"]
            if entry.get("status") in TERMINAL_STATES
        ]
        successes = sum(bool(entry.get("success")) for entry in final)
        samples = sum(int(entry.get("accepted_samples", 0)) for entry in final)
        return successes, len(final) - successes, samples


class SubprocessBackend:
    """Own Isaac, XRCE, and the finite ROS/PX4 flight for one episode."""

    produces_dataset = True

    def __init__(self, repository_root: Path, timeout_s: int = 180) -> None:
        self.repository_root = repository_root.resolve()
        self.timeout_s = timeout_s
        self.isaac_release = Path(
            os.environ.get(
                "UAV_ISAAC_SIM_RELEASE",
                str(Path.home() / "isaacsim/_build/linux-x86_64/release"),
            )
        ).resolve()
        self.launcher = self.isaac_release / "isaac-sim.streaming.sh"
        self.bootstrap = self.repository_root / "isaac/runtime/bootstrap.py"
        self.uav = self.repository_root / "uav"
        self._processes: list[subprocess.Popen] = []
        self._streams: list[object] = []
        self._flight: subprocess.Popen | None = None

    def preflight(self) -> None:
        """Verify dependencies and exclusive ownership before runtime."""
        if not self.launcher.is_file() or not os.access(
            self.launcher, os.X_OK
        ):
            raise RuntimeError(
                f"Isaac streaming launcher missing: {self.launcher}"
            )
        if shutil.which("MicroXRCEAgent") is None:
            raise RuntimeError("MicroXRCEAgent is not available")
        if not self.uav.is_file() or not os.access(self.uav, os.X_OK):
            raise RuntimeError(f"uav command is not executable: {self.uav}")
        for command in (
            ["pgrep", "-x", "MicroXRCEAgent"],
            ["pgrep", "-f", f"{self.isaac_release}/kit/kit"],
            [
                "pgrep", "-f",
                "/PX4-Autopilot/build/px4_sitl_default/bin/px4",
            ],
        ):
            if subprocess.run(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ).returncode == 0:
                raise RuntimeError(
                    "expert collection requires no pre-existing "
                    "Isaac/PX4/XRCE process"
                )

    def _open_log(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("w", encoding="utf-8")
        self._streams.append(stream)
        return stream

    def _start_runtime(self, runtime_dir: Path) -> None:
        xrce = subprocess.Popen(
            ["MicroXRCEAgent", "udp4", "-p", "8888"],
            cwd=self.repository_root,
            stdout=self._open_log(runtime_dir / "xrce.log"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        environment = os.environ.copy()
        environment.pop("DISPLAY", None)
        environment.pop("WAYLAND_DISPLAY", None)
        environment["UAV_EXPERT_SENSORS"] = "1"
        isaac = subprocess.Popen(
            [
                str(self.launcher),
                "--livestream", "2",
                "--exec", str(self.bootstrap),
            ],
            cwd=self.isaac_release,
            env=environment,
            stdout=self._open_log(runtime_dir / "isaac.log"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._processes = [isaac, xrce]

    def _run_control(self, arguments: list[str], log_path: Path) -> int:
        with log_path.open("w", encoding="utf-8") as stream:
            return subprocess.run(
                [str(self.uav), *arguments],
                cwd=self.repository_root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_s + 30,
            ).returncode

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
        """Start runtime, run a finite flight, and report progress."""
        self.preflight()
        self._start_runtime(runtime_dir)
        wait_status = self._run_control(
            ["expert-runtime-wait"], runtime_dir / "runtime_wait.log"
        )
        if wait_status != 0:
            raise RuntimeError(
                f"managed Isaac/PX4 runtime failed readiness for {episode_id}"
            )
        flight_log = self._open_log(runtime_dir / "flight.log")
        environment = os.environ.copy()
        environment["UAV_OFFLINE_TIMEOUT_SECONDS"] = str(self.timeout_s)
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
            stdout=flight_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        progress_path = dataset_root / episode_id / "progress.json"
        while self._flight.poll() is None:
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

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        for sent_signal, timeout in (
            (signal.SIGINT, 10.0),
            (signal.SIGTERM, 5.0),
            (signal.SIGKILL, 2.0),
        ):
            try:
                os.killpg(process.pid, sent_signal)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                continue

    def cleanup(self) -> None:
        """Stop all owned process groups and close their log streams."""
        if self._flight is not None:
            self._stop_process(self._flight)
            self._flight = None
        for process in self._processes:
            self._stop_process(process)
        self._processes.clear()
        for stream in self._streams:
            stream.close()
        self._streams.clear()


class DryRunBackend:
    """Offline fixture that never starts ROS/PX4 or writes formal data."""

    produces_dataset = False

    def preflight(self) -> None:
        """Require no live dependencies for the offline fixture."""
        return None

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


class ExpertCollector:
    """Coordinate episodes, resume, validation, QA, and cleanup."""

    def __init__(
        self,
        *,
        repository_root: Path,
        dataset_root: Path,
        episodes: int,
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
        self.repository_root = repository_root.resolve()
        self.dataset_root = dataset_root.resolve()
        self.episodes = episodes
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
                self.episodes, self.resume
            )
            self.base_seed = int(data["base_seed"])
            self._manifest_prepared = True
            self._recover_incomplete_episodes()
        else:
            self.store.create(
                self.episodes,
                self.base_seed,
                fixed_seed=self.fixed_seed,
            )
            self._manifest_prepared = True
        self.display = ProgressDisplay(
            int(self.store.data["target_episodes"])
        )
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        unfinished = any(
            entry.get("status") not in TERMINAL_STATES
            for entry in self.store.data["episodes"]
        )
        if unfinished:
            self.backend.preflight()
        self.store.set_collection_state("collecting")

    def _recover_incomplete_episodes(self) -> None:
        recovery_root = self.runtime_root / "recovered_incomplete"
        for entry in self.store.data["episodes"]:
            if entry.get("status") in TERMINAL_STATES:
                episode_dir = self.dataset_root / entry["episode_id"]
                if self.backend.produces_dataset and not (
                    episode_dir / "episode.json"
                ).is_file():
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
                "success": None,
                "accepted_samples": 0,
                "rejected_samples": 0,
                "terminal_reason": "",
            })
        self._reconcile_dataset_manifest()
        self.store.save()

    def _reconcile_dataset_manifest(self) -> None:
        path = self.dataset_root / "dataset_manifest.json"
        if not path.is_file():
            return
        manifest = _read_json(path)
        final_entries = [
            entry for entry in self.store.data["episodes"]
            if entry.get("status") in TERMINAL_STATES
        ]
        manifest.update({
            "episodes": [entry["episode_id"] for entry in final_entries],
            "episode_count": len(final_entries),
            "sample_count": sum(
                int(entry.get("accepted_samples", 0))
                for entry in final_entries
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

        finalized = finalize(
            self.dataset_root, episode_id, evidence_path, expected_success=None
        )
        validation = validate_collection_episode(
            self.dataset_root,
            self.autoencoder,
            episode_id,
            device=self.device,
        )
        return EpisodeOutcome(
            success=bool(finalized["success"]),
            accepted_samples=int(validation["sample_count"]),
            rejected_samples=int(finalized.get("rejected_sample_count", 0)),
            terminal_reason=str(finalized.get("terminal_reason", "")),
            dataset_bytes=int(finalized.get(
                "episode_disk_usage_bytes", _directory_size(episode_dir)
            )),
        )

    @staticmethod
    def _dry_run_outcome() -> EpisodeOutcome:
        return EpisodeOutcome(
            success=True,
            accepted_samples=42,
            rejected_samples=9,
            terminal_reason="dry_run_complete",
            dataset_bytes=0,
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
            return EpisodeOutcome(False, 0, 0, reason, 0)
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
        )

    def _visual_qa(self, through_episode: int) -> None:
        if through_episode % VISUAL_QA_INTERVAL:
            return
        entry = {
            "through_episode": through_episode,
            "status": "pending",
            "path": f"visual_qa/contact_sheet_{through_episode:06d}.jpg",
        }
        self.store.data.setdefault("visual_qa", []).append(entry)
        self.store.save()
        try:
            result = create_contact_sheet(self.dataset_root, through_episode)
        except Exception as error:  # noqa: BLE001 - noncritical QA boundary
            entry.update({"status": "failed", "error": str(error)})
        else:
            entry.update({
                "status": "complete",
                "source_episode": result["source_episode"],
                "path": result["contact_sheet"],
            })
        self.store.save()

    def run(self) -> dict:
        """Run or resume collection and always clean child processes."""
        current_index: int | None = None
        try:
            self._prepare()
            for entry in self.store.data["episodes"]:
                if entry.get("status") in TERMINAL_STATES:
                    continue
                current_index = int(entry["index"])
                episode_id = str(entry["episode_id"])
                seed = int(entry["seed"])
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
                        status="failed",
                        success=False,
                        accepted_samples=outcome.accepted_samples,
                        rejected_samples=0,
                        dataset_bytes=outcome.dataset_bytes,
                        terminal_reason=outcome.terminal_reason,
                        completed_utc=_utc_now(),
                        command_status=None,
                    )
                    self.display.transition(current_index, "INVALID SCENE")
                    if self.backend.produces_dataset:
                        self._visual_qa(current_index)
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
                runtime_dir = self.runtime_root / episode_id
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
                finally:
                    self.backend.cleanup()
                outcome = (
                    self._finalize_real_episode(episode_id, command_status)
                    if self.backend.produces_dataset
                    else self._dry_run_outcome()
                )
                terminal_state = "complete" if outcome.success else "failed"
                self.store.update_episode(
                    current_index,
                    status=terminal_state,
                    success=outcome.success,
                    accepted_samples=outcome.accepted_samples,
                    rejected_samples=outcome.rejected_samples,
                    dataset_bytes=outcome.dataset_bytes,
                    terminal_reason=outcome.terminal_reason,
                    completed_utc=_utc_now(),
                    command_status=command_status,
                )
                self.display.transition(current_index, "DATASET VALID")
                if self.backend.produces_dataset:
                    self._visual_qa(current_index)
                success_count, failure_count, total_samples = (
                    self.store.completed_counts()
                )
                self.display.render(
                    index=current_index,
                    seed=seed,
                    state="DATASET VALID",
                    success_count=success_count,
                    failure_count=failure_count,
                    current_samples=0,
                    current_rejected=0,
                    total_samples=total_samples,
                    dataset_bytes=_directory_size(self.dataset_root),
                    force=True,
                )
            if self.backend.produces_dataset:
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
            return validation
        except KeyboardInterrupt:
            self.backend.cleanup()
            if self._manifest_prepared and current_index is not None:
                self.store.update_episode(
                    current_index,
                    status="interrupted",
                    terminal_reason="operator_interrupt",
                )
            if self._manifest_prepared:
                self.store.set_collection_state(
                    "interrupted", "operator interrupt; resume is safe"
                )
            raise
        except Exception as error:
            self.backend.cleanup()
            if self._manifest_prepared and self.store.data:
                if current_index is not None:
                    entry = self.store.data["episodes"][current_index - 1]
                    if entry.get("status") not in TERMINAL_STATES:
                        self.store.update_episode(
                            current_index,
                            status="infrastructure_failure",
                            terminal_reason=str(error),
                        )
                self.store.set_collection_state(
                    "stopped_infrastructure_failure", str(error)
                )
            raise


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
            "number of episodes to add in this run; after an interruption, "
            "pass the same value again with --resume"
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
            "resume an unfinished run without overwriting completed episodes; "
            "when no run is unfinished, append a new run"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
