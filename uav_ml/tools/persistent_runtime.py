"""Shared persistent Isaac runtime and per-attempt PX4 process ownership."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time


PX4_MODEL = "gazebo-classic_iris"


class RecoverableAttemptError(RuntimeError):
    """An attempt failed while the persistent Isaac runtime remains usable."""


class FatalRuntimeError(RuntimeError):
    """The job-level Isaac/XRCE runtime can no longer be trusted."""


def atomic_json(path: Path, payload: dict) -> None:
    """Write one JSON artifact atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    """Read one JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def process_start_ticks(pid: int) -> int | None:
    """Return the Linux process generation token from /proc."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return int(data[data.rfind(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def process_rss_kib(pid: int) -> int | None:
    """Return current resident memory for one process, when available."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_exists(command: list[str]) -> bool:
    """Return whether a read-only process lookup finds a match."""
    return subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def live_process_group_members(process_group_id: int) -> list[int]:
    """Return non-zombie members of one owned process group."""
    if process_group_id <= 0:
        return []
    result = subprocess.run(
        ["ps", "-eo", "pid=,pgid=,stat="],
        check=False,
        capture_output=True,
        text=True,
    )
    members = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            pid, pgid = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if pgid == process_group_id and not fields[2].startswith("Z"):
            members.append(pid)
    return members


def stop_process(
    process: subprocess.Popen | None,
    *,
    interrupt_s: float = 10.0,
    terminate_s: float = 5.0,
    kill_s: float = 2.0,
) -> dict:
    """Stop an owned process group with bounded graceful escalation."""
    if process is None:
        return {"method": "not_started", "exit_code": None, "escalated": False}
    if process.poll() is not None:
        return {
            "method": "already_exited",
            "exit_code": process.returncode,
            "escalated": False,
        }
    attempts = (
        (signal.SIGINT, interrupt_s, "SIGINT"),
        (signal.SIGTERM, terminate_s, "SIGTERM"),
        (signal.SIGKILL, kill_s, "SIGKILL"),
    )
    history = []
    for sent_signal, timeout_s, name in attempts:
        try:
            os.killpg(process.pid, sent_signal)
        except ProcessLookupError:
            break
        history.append(name)
        try:
            process.wait(timeout=timeout_s)
            return {
                "method": name,
                "history": history,
                "exit_code": process.returncode,
                "escalated": name != "SIGINT",
            }
        except subprocess.TimeoutExpired:
            continue
    return {
        "method": history[-1] if history else "process_missing",
        "history": history,
        "exit_code": process.poll(),
        "escalated": len(history) > 1,
    }


def stop_process_group(
    process_group_id: int,
    *,
    interrupt_s: float = 3.0,
    terminate_s: float = 2.0,
    kill_s: float = 1.0,
) -> dict:
    """Reap any remaining children in a known, owned process group."""
    if process_group_id <= 0:
        return {"method": "invalid_group", "remaining": []}
    history = []
    for sent_signal, timeout_s, name in (
        (signal.SIGINT, interrupt_s, "SIGINT"),
        (signal.SIGTERM, terminate_s, "SIGTERM"),
        (signal.SIGKILL, kill_s, "SIGKILL"),
    ):
        remaining = live_process_group_members(process_group_id)
        if not remaining:
            return {
                "method": history[-1] if history else "already_empty",
                "history": history,
                "remaining": [],
            }
        try:
            os.killpg(process_group_id, sent_signal)
        except ProcessLookupError:
            return {
                "method": "process_group_missing",
                "history": history,
                "remaining": [],
            }
        history.append(name)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            remaining = live_process_group_members(process_group_id)
            if not remaining:
                return {
                    "method": name,
                    "history": history,
                    "remaining": [],
                }
            time.sleep(0.10)
    return {
        "method": history[-1],
        "history": history,
        "remaining": live_process_group_members(process_group_id),
    }


def clean_ros_environment() -> dict[str, str]:
    """Return the minimal environment used by bounded ROS subprocesses."""
    environment = {
        "HOME": os.environ.get("HOME", str(Path.home())),
        "USER": os.environ.get("USER", ""),
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "ROS_DOMAIN_ID": "0",
    }
    if os.environ.get("TERM"):
        environment["TERM"] = os.environ["TERM"]
    return environment


def ros_command(repository_root: Path, arguments: list[str]) -> list[str]:
    """Wrap one command in clean ROS Jazzy and workspace setup files."""
    return [
        "/usr/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        (
            'set -eo pipefail; source "$1"; source "$2"; '
            'shift 2; exec "$@"'
        ),
        "persistent-runtime-ros",
        "/opt/ros/jazzy/setup.bash",
        str(repository_root / "ros2_ws/install/setup.bash"),
        *arguments,
    ]


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    timeout_s: float,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> int:
    """Run one process group with a hard timeout and captured combined log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            stop_process(process)
            return 124
        except BaseException:
            stop_process(process)
            raise


class PersistentRuntimeManager:
    """Own one Isaac/XRCE job and sequential external PX4 generations."""

    def __init__(
        self,
        repository_root: Path,
        log_root: Path,
        *,
        runtime_timeout_s: float = 120.0,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.log_root = log_root.resolve()
        self.runtime_timeout_s = float(runtime_timeout_s)
        self.isaac_release = Path(os.environ.get(
            "UAV_ISAAC_SIM_RELEASE",
            str(Path.home() / "isaacsim/_build/linux-x86_64/release"),
        )).resolve()
        self.launcher = self.isaac_release / "isaac-sim.streaming.sh"
        self.px4_root = Path.home() / "PX4-Autopilot"
        self.px4_binary = self.px4_root / "build/px4_sitl_default/bin/px4"
        self.px4_data = self.px4_root / "ROMFS/px4fmu_common"
        self.px4_rc = self.px4_data / "init.d-posix/rcS"
        self.xrce: subprocess.Popen | None = None
        self.isaac: subprocess.Popen | None = None
        self.px4: subprocess.Popen | None = None
        self._xrce_stream = None
        self._isaac_stream = None
        self._px4_stream = None
        self._started_monotonic: float | None = None
        self._initial_identity: dict = {}
        self._last_identity: dict = {}
        self._previous_dds_endpoints: dict[str, list[str]] | None = None
        self._process_groups: set[int] = set()
        self._memory_samples: list[dict] = []
        self._scene_revision = 0
        self._consecutive_reset_failures = 0
        self._attempt_evidence: dict = {}
        self._job_evidence: dict = {
            "isaac_pid": None,
            "xrce_pid": None,
            "runtime_start_time": None,
            "isaac_restart_count": 0,
            "isaac_pid_observations": [],
            "camera_recreated": False,
            "px4_ownership": "external",
            "pegasus_px4_autolaunch": False,
            "pegasus_lockstep_enabled": False,
        }

    @property
    def attempt_evidence(self) -> dict:
        """Return a JSON-safe snapshot of the current attempt evidence."""
        return json.loads(json.dumps(self._attempt_evidence))

    def record_attempt_evidence(self, **changes: object) -> None:
        """Add JSON-safe process evidence owned by an outer coordinator."""
        self._attempt_evidence.update(changes)

    @property
    def job_evidence(self) -> dict:
        """Return current job-level lifecycle evidence."""
        evidence = json.loads(json.dumps(self._job_evidence))
        evidence["memory_samples"] = list(self._memory_samples)
        if self._started_monotonic is not None:
            evidence["runtime_elapsed_s"] = (
                time.monotonic() - self._started_monotonic
            )
        return evidence

    def preflight(self) -> None:
        """Verify dependencies and exclusive ownership before job startup."""
        required = (
            self.launcher,
            self.px4_binary,
            self.px4_rc,
            self.repository_root / "isaac/runtime/bootstrap.py",
            self.repository_root / "isaac/runtime/persistent_smoke_control.py",
            self.repository_root / "ros2_ws/install/setup.bash",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "missing runtime files: " + ", ".join(missing)
            )
        if not os.access(self.launcher, os.X_OK):
            raise PermissionError(
                f"Isaac launcher is not executable: {self.launcher}"
            )
        if shutil.which("MicroXRCEAgent") is None:
            raise FileNotFoundError("MicroXRCEAgent is not available")
        existing = []
        if process_exists(["pgrep", "-x", "MicroXRCEAgent"]):
            existing.append("MicroXRCEAgent")
        if process_exists([
            "pgrep", "-f", str(self.isaac_release / "kit/kit")
        ]):
            existing.append("Isaac Sim")
        if process_exists(["pgrep", "-f", str(self.px4_binary)]):
            existing.append("PX4 SITL")
        if existing:
            raise RuntimeError(
                "persistent runtime requires exclusive ownership; already "
                "running: "
                + ", ".join(existing)
            )

    def start_job(self) -> dict:
        """Start XRCE and Isaac once, then wait for camera readiness."""
        if self.isaac is not None or self.xrce is not None:
            raise RuntimeError("persistent runtime job was already started")
        self.log_root.mkdir(parents=True, exist_ok=True)
        self._started_monotonic = time.monotonic()
        self._xrce_stream = (self.log_root / "xrce.log").open(
            "w", encoding="utf-8"
        )
        self.xrce = subprocess.Popen(
            ["MicroXRCEAgent", "udp4", "-p", "8888"],
            cwd=self.repository_root,
            env={**os.environ, "ROS_DOMAIN_ID": "0"},
            stdout=self._xrce_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        isaac_environment = os.environ.copy()
        isaac_environment.update({
            "ROS_DOMAIN_ID": "0",
            "UAV_EXPERT_SENSORS": "1",
            "UAV_OBSERVER_VIEWPORT": "0",
            "UAV_PERSISTENT_RUNTIME": "1",
        })
        isaac_environment.pop("DISPLAY", None)
        isaac_environment.pop("WAYLAND_DISPLAY", None)
        self._isaac_stream = (self.log_root / "isaac.log").open(
            "w", encoding="utf-8"
        )
        self.isaac = subprocess.Popen(
            [
                str(self.launcher),
                "--livestream", "2",
                "--exec", str(
                    self.repository_root / "isaac/runtime/bootstrap.py"
                ),
            ],
            cwd=self.isaac_release,
            env=isaac_environment,
            stdout=self._isaac_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._process_groups.update((self.xrce.pid, self.isaac.pid))
        self._job_evidence.update({
            "isaac_pid": self.isaac.pid,
            "xrce_pid": self.xrce.pid,
            "runtime_start_time": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "isaac_pid_observations": [self.isaac.pid],
        })
        print(f"[PersistentRuntime] xrce_pid={self.xrce.pid}", flush=True)
        print(f"[PersistentRuntime] isaac_pid={self.isaac.pid}", flush=True)
        boot_dir = self.log_root / "bootstrap"
        boot = self.lifecycle(
            "observe", 0, boot_dir, "boot", require_camera=True
        )
        self._initial_identity = boot.get("resource_identity", {})
        self._last_identity = self._initial_identity
        self._job_evidence["resource_identity"] = self._initial_identity
        self.sample_memory("job_ready")
        return self.job_evidence

    def assert_job_alive(self) -> None:
        """Raise a fatal error if either persistent process has exited."""
        if self.isaac is None or self.isaac.poll() is not None:
            code = None if self.isaac is None else self.isaac.returncode
            raise FatalRuntimeError(
                f"Isaac exited unexpectedly with status {code}"
            )
        if self.xrce is None or self.xrce.poll() is not None:
            code = None if self.xrce is None else self.xrce.returncode
            raise FatalRuntimeError(
                f"MicroXRCEAgent exited unexpectedly with status {code}"
            )
        observations = self._job_evidence["isaac_pid_observations"]
        observations.append(self.isaac.pid)

    def assert_px4_alive(self) -> None:
        """Raise an attempt failure if its external PX4 generation exited."""
        if self.px4 is None or self.px4.poll() is not None:
            code = None if self.px4 is None else self.px4.returncode
            raise RecoverableAttemptError(
                f"PX4 exited unexpectedly with status {code}"
            )
        expected_ticks = self._attempt_evidence.get("px4_start_ticks")
        if process_start_ticks(self.px4.pid) != expected_ticks:
            raise RecoverableAttemptError(
                "PX4 process generation changed during the attempt"
            )

    def run_ros(
        self, arguments: list[str], log_path: Path, timeout_s: float
    ) -> int:
        """Run one bounded ROS command in the project overlay."""
        return run_logged(
            ros_command(self.repository_root, arguments),
            log_path,
            timeout_s=timeout_s,
            cwd=self.repository_root,
            environment=clean_ros_environment(),
        )

    def lifecycle(
        self,
        command: str,
        generation: int,
        directory: Path,
        label: str,
        *,
        require_camera: bool = False,
    ) -> dict:
        """Execute one acknowledged stop/reset/observe Isaac command."""
        directory.mkdir(parents=True, exist_ok=True)
        evidence_path = directory / f"{label}_status.json"
        command_id = f"generation-{generation}-{label}-{time.monotonic_ns()}"
        status = self.run_ros([
            "ros2", "run", "uav_px4_control",
            "runtime_smoke_lifecycle_client", "--ros-args",
            "-p", f"command:={command}",
            "-p", f"command_id:={command_id}",
            "-p", f"generation:={generation}",
            "-p", f"timeout_s:={self.runtime_timeout_s}",
            "-p", "require_camera_ready:="
            + ("true" if require_camera else "false"),
            "-p", f"evidence_path:={evidence_path}",
        ], directory / f"{label}.log", self.runtime_timeout_s + 15.0)
        if not evidence_path.is_file():
            self.assert_job_alive()
            raise RecoverableAttemptError(
                f"{label} lifecycle evidence missing (status {status})"
            )
        evidence = read_json(evidence_path)
        if status != 0 or evidence.get("state") in {
            "failed", "client_timeout"
        }:
            self.assert_job_alive()
            raise RecoverableAttemptError(
                f"{label} lifecycle failed: "
                + str(evidence.get("failure_reason") or status)
            )
        identity = evidence.get("resource_identity", {})
        if self._initial_identity and identity != self._initial_identity:
            self._job_evidence["camera_recreated"] = True
            raise FatalRuntimeError(
                "persistent Isaac resource identity changed"
            )
        if identity:
            self._last_identity = identity
        return evidence

    def prepare_attempt(self, generation: int, directory: Path) -> dict:
        """Stop, reset, verify resources, and resume the backend listener."""
        self.assert_job_alive()
        self._attempt_evidence = {
            "runtime_generation": generation,
            "isaac_pid": self.isaac.pid if self.isaac else None,
            "xrce_pid": self.xrce.pid if self.xrce else None,
            "failure_classification": "",
        }
        try:
            self.lifecycle(
                "stop_episode", generation, directory, "pre_reset_stop"
            )
            reset = self.lifecycle(
                "reset_episode", generation, directory, "reset"
            )
        except RecoverableAttemptError as error:
            self._consecutive_reset_failures += 1
            self._attempt_evidence["reset_success"] = False
            self._attempt_evidence["reset_failure"] = str(error)
            if self._consecutive_reset_failures >= 2:
                raise FatalRuntimeError(
                    "two consecutive UAV reset failures: " + str(error)
                ) from error
            raise
        reset_evidence = reset.get("reset_evidence", {})
        reset_success = bool(
            reset_evidence and all(reset_evidence.values())
        )
        self._attempt_evidence.update({
            "reset_success": reset_success,
            "reset_evidence": reset_evidence,
            "camera_identity": reset.get("resource_identity", {}),
            "camera_recreated": not bool(
                reset.get("resource_identity_unchanged", False)
            ),
        })
        if not reset_success:
            self._consecutive_reset_failures += 1
            if self._consecutive_reset_failures >= 2:
                raise FatalRuntimeError(
                    "two consecutive incomplete UAV resets"
                )
            raise RecoverableAttemptError("UAV reset evidence was incomplete")
        self._consecutive_reset_failures = 0
        self.sample_memory(f"generation_{generation}_reset")
        return reset

    def prepare_scene(
        self,
        generation: int,
        episode_id: str,
        seed: int,
        directory: Path,
    ) -> dict:
        """Apply a scene after reset without requiring old PX4 telemetry."""
        evidence_path = directory / "scene_status.json"
        status = self.run_ros([
            "ros2", "run", "uav_data_recorder", "episode_scene_client",
            "--ros-args",
            "-p", f"episode_id:={episode_id}",
            "-p", f"random_seed:={seed}",
            "-p", "scene_mode:=normal",
            "-p", f"timeout_s:={min(30.0, self.runtime_timeout_s)}",
            "-p", "require_px4_safe_state:=false",
            "-p", f"expected_runtime_generation:={generation}",
            "-p", f"minimum_scene_revision:={self._scene_revision}",
            "-p", f"evidence_path:={evidence_path}",
        ], directory / "scene_prepare.log", min(
            45.0, self.runtime_timeout_s + 5.0
        ))
        if status != 0 or not evidence_path.is_file():
            self.assert_job_alive()
            raise RecoverableAttemptError(
                f"scene preparation failed with status {status}"
            )
        evidence = read_json(evidence_path)
        revision = int(evidence.get("scene_revision", 0))
        if revision <= self._scene_revision:
            raise RecoverableAttemptError("scene revision did not advance")
        self._scene_revision = revision
        self._attempt_evidence.update({
            "scene_revision": revision,
            "scene_runtime_sequence": evidence.get("sequence"),
            "camera_frame_boundary": evidence.get(
                "scene_camera_boundary", {}
            ),
            "camera_frame_counts_at_scene_ack": {
                name: evidence.get(name)
                for name in (
                    "fpv_rgb_frame_count",
                    "observer_rgb_frame_count",
                    "fpv_depth_frame_count",
                )
            },
        })
        return evidence

    def start_px4(self, generation: int, directory: Path) -> dict:
        """Start one externally owned PX4 in a fresh temporary rootfs."""
        if self.px4 is not None:
            raise RuntimeError("previous PX4 handle was not reaped")
        rootfs = Path(tempfile.mkdtemp(
            prefix=f"px4_generation_{generation}_", dir=directory
        ))
        self._px4_stream = (directory / "px4.log").open(
            "w", encoding="utf-8"
        )
        environment = os.environ.copy()
        environment.update({
            "PX4_SIM_MODEL": PX4_MODEL,
            "ROS_DOMAIN_ID": "0",
        })
        try:
            self.px4 = subprocess.Popen(
                [
                    str(self.px4_binary),
                    str(self.px4_data) + "/",
                    "-s", str(self.px4_rc),
                    "-i", "0",
                    "-d",
                ],
                cwd=rootfs,
                env=environment,
                stdout=self._px4_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            self._px4_stream.close()
            self._px4_stream = None
            raise RecoverableAttemptError(
                f"could not start external PX4: {error}"
            ) from error
        self._process_groups.add(self.px4.pid)
        ticks = process_start_ticks(self.px4.pid)
        if ticks is None:
            raise RecoverableAttemptError(
                "could not read new PX4 process generation"
            )
        evidence = {
            "px4_pid": self.px4.pid,
            "px4_start_ticks": ticks,
            "px4_process_group_id": self.px4.pid,
            "px4_rootfs": str(rootfs),
            "px4_log": str((directory / "px4.log").resolve()),
        }
        self._attempt_evidence.update(evidence)
        print(
            f"[PersistentRuntime] episode_{generation}_px4_pid={self.px4.pid}",
            flush=True,
        )
        return evidence

    def probe_px4(self, generation: int, directory: Path) -> dict:
        """Require a live expected process and post-subscription DDS data."""
        if self.px4 is None:
            raise RecoverableAttemptError("PX4 was not started")
        pid = self.px4.pid
        ticks = int(self._attempt_evidence["px4_start_ticks"])
        evidence_path = directory / "px4_readiness.json"
        status = self.run_ros([
            "ros2", "run", "uav_px4_control", "px4_generation_probe",
            "--ros-args",
            "-p", f"expected_pid:={pid}",
            "-p", f"expected_start_ticks:={ticks}",
            "-p", f"generation:={generation}",
            "-p", f"timeout_s:={self.runtime_timeout_s}",
            "-p", "minimum_samples:=5",
            "-p", "minimum_span_s:=0.5",
            "-p", f"evidence_path:={evidence_path}",
        ], directory / "px4_readiness.log", self.runtime_timeout_s + 15.0)
        if not evidence_path.is_file():
            self.assert_job_alive()
            raise RecoverableAttemptError(
                f"PX4 readiness evidence missing (status {status})"
            )
        evidence = read_json(evidence_path)
        self._attempt_evidence.update({
            "fresh_dds_verified": False,
            "px4_readiness": evidence,
        })
        if status != 0 or not evidence.get("success"):
            self.assert_job_alive()
            raise RecoverableAttemptError(
                "PX4 fresh readiness failed: "
                + str(evidence.get("failure_reason") or status)
            )
        current_endpoints = {
            "status": list(evidence.get("status_endpoint_gids", [])),
            "odometry": list(evidence.get("odometry_endpoint_gids", [])),
        }
        endpoints_present = all(current_endpoints.values())
        endpoint_generation_changed = bool(
            self._previous_dds_endpoints is None
            or all(
                current_endpoints[name] != self._previous_dds_endpoints[name]
                for name in current_endpoints
            )
        )
        fresh_dds = endpoints_present and endpoint_generation_changed
        evidence["fresh_dds_session_verified"] = fresh_dds
        evidence["previous_dds_endpoints"] = self._previous_dds_endpoints
        self._attempt_evidence["px4_readiness"] = evidence
        if not fresh_dds:
            raise RecoverableAttemptError(
                "DDS endpoint generation did not change"
            )
        self._previous_dds_endpoints = current_endpoints
        doctor = self.run_ros([
            "ros2", "run", "uav_px4_control", "px4_sitl_doctor"
        ], directory / "px4_doctor.log", 30.0)
        evidence["doctor_success"] = doctor == 0
        if doctor != 0:
            raise RecoverableAttemptError(
                "PX4 SITL doctor rejected fresh runtime"
            )
        self._attempt_evidence.update({
            "fresh_dds_verified": True,
            "px4_readiness": evidence,
        })
        return evidence

    def stop_episode(self, generation: int, directory: Path) -> dict:
        """Stop the backend first, then reap the current external PX4."""
        stop_evidence = {}
        try:
            if self.isaac is not None and self.isaac.poll() is None:
                stop_evidence = self.lifecycle(
                    "stop_episode", generation, directory, "post_flight_stop"
                )
        finally:
            px4_group = None if self.px4 is None else self.px4.pid
            shutdown = stop_process(self.px4)
            group_shutdown = (
                {"method": "not_started", "remaining": []}
                if px4_group is None
                else stop_process_group(px4_group)
            )
            if px4_group is not None and not group_shutdown.get("remaining"):
                self._process_groups.discard(px4_group)
            exit_code = None if self.px4 is None else self.px4.poll()
            self._attempt_evidence.update({
                "backend_stop_evidence": stop_evidence,
                "px4_shutdown": shutdown,
                "px4_group_shutdown": group_shutdown,
                "px4_processes_remaining": group_shutdown.get(
                    "remaining", []
                ),
                "px4_exit_code": exit_code,
            })
            self.px4 = None
            if self._px4_stream is not None:
                self._px4_stream.close()
                self._px4_stream = None
            self.sample_memory(f"generation_{generation}_complete")
        return stop_evidence

    def sample_memory(self, label: str) -> None:
        """Record process RSS evidence without external profilers."""
        self._memory_samples.append({
            "label": label,
            "monotonic_s": time.monotonic(),
            "isaac_rss_kib": (
                None if self.isaac is None else process_rss_kib(self.isaac.pid)
            ),
            "xrce_rss_kib": (
                None if self.xrce is None else process_rss_kib(self.xrce.pid)
            ),
        })

    def cleanup_job(self) -> dict:
        """Stop all job-level resources once and report remaining children."""
        if self.px4 is not None:
            px4_group = self.px4.pid
            self._job_evidence["final_px4_cleanup"] = stop_process(self.px4)
            group_cleanup = stop_process_group(px4_group)
            self._job_evidence["final_px4_group_cleanup"] = group_cleanup
            if not group_cleanup.get("remaining"):
                self._process_groups.discard(px4_group)
            self.px4 = None
        self._job_evidence["isaac_shutdown"] = stop_process(
            self.isaac, interrupt_s=15.0, terminate_s=10.0
        )
        self._job_evidence["xrce_shutdown"] = stop_process(self.xrce)
        for name in ("_px4_stream", "_isaac_stream", "_xrce_stream"):
            stream = getattr(self, name)
            if stream is not None:
                stream.close()
                setattr(self, name, None)
        for group in self._process_groups:
            if live_process_group_members(group):
                stop_process_group(group)
        remaining = {
            str(group): live_process_group_members(group)
            for group in self._process_groups
        }
        remaining = {key: value for key, value in remaining.items() if value}
        self._job_evidence["owned_process_groups_remaining"] = remaining
        self._job_evidence["owned_processes_remaining"] = sorted({
            pid for members in remaining.values() for pid in members
        })
        observations = self._job_evidence.get("isaac_pid_observations", [])
        self._job_evidence["isaac_restart_count"] = max(
            0, len(set(observations)) - 1
        )
        self._job_evidence["camera_recreated"] = bool(
            self._job_evidence.get("camera_recreated")
        )
        return self.job_evidence
