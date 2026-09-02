"""Two-generation PX4 smoke test against one persistent Isaac runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time


SCHEMA = "uav_expert_runtime_smoke/v1"
PX4_MODEL = "gazebo-classic_iris"


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


def _process_start_ticks(pid: int) -> int | None:
    try:
        data = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        return int(data[data.rfind(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def _process_exists(command: list[str]) -> bool:
    return subprocess.run(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    ).returncode == 0


def _live_process_group_members(process_group_id: int) -> list[int]:
    """Return non-zombie members of one diagnostic-owned process group."""
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


def _stop_process(
    process: subprocess.Popen | None,
    *,
    interrupt_s: float = 10.0,
    terminate_s: float = 5.0,
    kill_s: float = 2.0,
) -> dict:
    """Stop one owned process group with bounded escalation evidence."""
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


def _clean_ros_environment() -> dict[str, str]:
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


def _ros_command(repository_root: Path, arguments: list[str]) -> list[str]:
    return [
        "/usr/bin/bash",
        "--noprofile",
        "--norc",
        "-c",
        (
            'set -eo pipefail; source "$1"; source "$2"; '
            'shift 2; exec "$@"'
        ),
        "runtime-smoke-ros",
        "/opt/ros/jazzy/setup.bash",
        str(repository_root / "ros2_ws/install/setup.bash"),
        *arguments,
    ]


def _run_logged(
    command: list[str],
    log_path: Path,
    *,
    timeout_s: float,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> int:
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
            _stop_process(process)
            return 124
        except BaseException:
            _stop_process(process)
            raise


class PersistentRuntimeSmoke:
    """Own one Isaac/Agent generation and sequential external PX4 processes."""

    def __init__(
        self,
        repository_root: Path,
        episodes: int,
        artifact_root: Path,
        runtime_timeout_s: float,
        episode_timeout_s: float,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.episodes = episodes
        self.artifact_root = artifact_root.resolve()
        self.runtime_timeout_s = runtime_timeout_s
        self.episode_timeout_s = episode_timeout_s
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
        self.started = time.monotonic()
        self.summary = {
            "schema": SCHEMA,
            "artifact_root": str(self.artifact_root),
            "isaac_pid": None,
            "xrce_pid": None,
            "episode_count": episodes,
            "px4_ownership": "external",
            "pegasus_px4_autolaunch": False,
            "pegasus_lockstep_enabled": False,
            "episodes": [],
            "isaac_pid_observations": [],
            "isaac_restarted": False,
            "camera_recreated": False,
            "success": False,
            "failure_reason": "",
            "total_runtime_s": 0.0,
        }
        self.summary_path = self.artifact_root / "summary.json"

    def _save(self) -> None:
        self.summary["total_runtime_s"] = time.monotonic() - self.started
        _atomic_json(self.summary_path, self.summary)

    def preflight(self) -> None:
        if self.artifact_root.exists():
            raise FileExistsError(
                f"artifact root already exists: {self.artifact_root}"
            )
        required_files = (
            self.launcher,
            self.px4_binary,
            self.px4_rc,
            self.repository_root / "isaac/runtime/bootstrap.py",
            self.repository_root / "isaac/runtime/persistent_smoke_control.py",
            self.repository_root / "ros2_ws/install/setup.bash",
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing runtime files: " + ", ".join(missing))
        if shutil.which("MicroXRCEAgent") is None:
            raise FileNotFoundError("MicroXRCEAgent is not available")
        existing = []
        if _process_exists(["pgrep", "-x", "MicroXRCEAgent"]):
            existing.append("MicroXRCEAgent")
        if _process_exists(["pgrep", "-f", str(self.isaac_release / "kit/kit")]):
            existing.append("Isaac Sim")
        if _process_exists(["pgrep", "-f", str(self.px4_binary)]):
            existing.append("PX4 SITL")
        if existing:
            raise RuntimeError(
                "runtime smoke requires exclusive ownership; already running: "
                + ", ".join(existing)
            )

    def start_runtime(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=False)
        self._xrce_stream = (self.artifact_root / "xrce.log").open(
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
            "UAV_PERSISTENT_RUNTIME_SMOKE": "1",
        })
        isaac_environment.pop("DISPLAY", None)
        isaac_environment.pop("WAYLAND_DISPLAY", None)
        self._isaac_stream = (self.artifact_root / "isaac.log").open(
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
        self.summary["xrce_pid"] = self.xrce.pid
        self.summary["isaac_pid"] = self.isaac.pid
        self.summary["isaac_pid_observations"].append(self.isaac.pid)
        print(f"[RuntimeSmoke] xrce_pid={self.xrce.pid}", flush=True)
        print(f"[RuntimeSmoke] isaac_pid={self.isaac.pid}", flush=True)
        self._save()

    def _assert_runtime_alive(self) -> None:
        if self.isaac is None or self.isaac.poll() is not None:
            code = None if self.isaac is None else self.isaac.returncode
            raise RuntimeError(f"Isaac exited unexpectedly with status {code}")
        if self.xrce is None or self.xrce.poll() is not None:
            code = None if self.xrce is None else self.xrce.returncode
            raise RuntimeError(f"MicroXRCEAgent exited unexpectedly with status {code}")
        self.summary["isaac_pid_observations"].append(self.isaac.pid)

    def _run_ros(self, arguments: list[str], log_path: Path, timeout_s: float) -> int:
        return _run_logged(
            _ros_command(self.repository_root, arguments),
            log_path,
            timeout_s=timeout_s,
            cwd=self.repository_root,
            environment=_clean_ros_environment(),
        )

    def _lifecycle(
        self,
        command: str,
        generation: int,
        episode_dir: Path,
        label: str,
        *,
        require_camera: bool = False,
    ) -> dict:
        evidence_path = episode_dir / f"{label}_status.json"
        command_id = f"generation-{generation}-{label}-{time.monotonic_ns()}"
        status = self._run_ros([
            "ros2", "run", "uav_px4_control",
            "runtime_smoke_lifecycle_client", "--ros-args",
            "-p", f"command:={command}",
            "-p", f"command_id:={command_id}",
            "-p", f"generation:={generation}",
            "-p", f"timeout_s:={self.runtime_timeout_s}",
            "-p", f"require_camera_ready:={'true' if require_camera else 'false'}",
            "-p", f"evidence_path:={evidence_path}",
        ], episode_dir / f"{label}.log", self.runtime_timeout_s + 15.0)
        if not evidence_path.is_file():
            raise RuntimeError(
                f"{label} lifecycle evidence missing (status {status})"
            )
        evidence = _read_json(evidence_path)
        if status != 0 or evidence.get("state") in {"failed", "client_timeout"}:
            raise RuntimeError(
                f"{label} lifecycle failed: "
                + str(evidence.get("failure_reason") or status)
            )
        return evidence

    def wait_for_boot(self) -> dict:
        boot_dir = self.artifact_root / "bootstrap"
        boot_dir.mkdir(parents=True, exist_ok=True)
        return self._lifecycle(
            "observe", 0, boot_dir, "boot", require_camera=True
        )

    def _start_px4(
        self, generation: int, episode_dir: Path
    ) -> tuple[int, int, Path]:
        rootfs = Path(tempfile.mkdtemp(
            prefix=f"px4_generation_{generation}_",
            dir=episode_dir,
        ))
        px4_log = episode_dir / "px4.log"
        self._px4_stream = px4_log.open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update({
            "PX4_SIM_MODEL": PX4_MODEL,
            "ROS_DOMAIN_ID": "0",
        })
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
        start_ticks = _process_start_ticks(self.px4.pid)
        if start_ticks is None:
            raise RuntimeError("could not read new PX4 process generation")
        print(
            f"[RuntimeSmoke] episode_{generation}_px4_pid={self.px4.pid}",
            flush=True,
        )
        return self.px4.pid, start_ticks, rootfs

    def _probe_px4(
        self,
        generation: int,
        pid: int,
        start_ticks: int,
        episode_dir: Path,
    ) -> dict:
        evidence_path = episode_dir / "px4_readiness.json"
        status = self._run_ros([
            "ros2", "run", "uav_px4_control", "px4_generation_probe",
            "--ros-args",
            "-p", f"expected_pid:={pid}",
            "-p", f"expected_start_ticks:={start_ticks}",
            "-p", f"generation:={generation}",
            "-p", f"timeout_s:={self.runtime_timeout_s}",
            "-p", "minimum_samples:=5",
            "-p", "minimum_span_s:=0.5",
            "-p", f"evidence_path:={evidence_path}",
        ], episode_dir / "px4_readiness.log", self.runtime_timeout_s + 15.0)
        if not evidence_path.is_file():
            raise RuntimeError(f"PX4 readiness evidence missing (status {status})")
        evidence = _read_json(evidence_path)
        if status != 0 or not evidence.get("success"):
            raise RuntimeError(
                "PX4 fresh readiness failed: "
                + str(evidence.get("failure_reason") or status)
            )
        doctor_status = self._run_ros([
            "ros2", "run", "uav_px4_control", "px4_sitl_doctor"
        ], episode_dir / "px4_doctor.log", 30.0)
        evidence["doctor_success"] = doctor_status == 0
        if doctor_status != 0:
            raise RuntimeError("PX4 SITL doctor rejected fresh runtime")
        return evidence

    def _run_flight(self, generation: int, episode_dir: Path) -> tuple[int, dict]:
        evidence_path = episode_dir / "flight_evidence.json"
        timeout = max(30.0, self.episode_timeout_s)
        status = self._run_ros([
            "ros2", "launch", "uav_px4_control", "px4_sitl_flight.launch.py",
            f"evidence_path:={evidence_path}",
            f"timeout_s:={max(20.0, timeout - 10.0)}",
            "start_delay_s:=5.0",
            "use_external_scene:=false",
            "require_isaac_evidence:=false",
            "record_expert_dataset:=false",
            f"episode_id:=smoke_episode_{generation:06d}",
            f"random_seed:={generation}",
        ], episode_dir / "flight.log", timeout + 20.0)
        evidence = _read_json(evidence_path) if evidence_path.is_file() else {}
        return status, evidence

    def run_episode(self, generation: int) -> dict:
        episode_dir = self.artifact_root / f"episode_{generation:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "episode": generation,
            "isaac_pid": self.isaac.pid if self.isaac else None,
            "px4_pid": None,
            "px4_start_ticks": None,
            "px4_rootfs": "",
            "px4_log": str(episode_dir / "px4.log"),
            "px4_exit_code": None,
            "px4_shutdown": {},
            "reset_success": False,
            "px4_readiness": {},
            "takeoff_success": False,
            "landing_success": False,
            "disarm_success": False,
            "failure_reason": "",
            "resource_identity": {},
        }
        try:
            self._assert_runtime_alive()
            self._lifecycle(
                "stop_episode", generation, episode_dir, "pre_reset_stop"
            )
            reset = self._lifecycle(
                "reset_episode", generation, episode_dir, "reset"
            )
            reset_evidence = reset.get("reset_evidence", {})
            result["reset_success"] = bool(
                reset_evidence and all(reset_evidence.values())
            )
            result["reset_evidence"] = reset_evidence
            result["resource_identity"] = reset.get("resource_identity", {})
            if not result["reset_success"]:
                raise RuntimeError("Isaac UAV reset evidence was incomplete")

            pid, start_ticks, rootfs = self._start_px4(
                generation, episode_dir
            )
            result["px4_pid"] = pid
            result["px4_start_ticks"] = start_ticks
            result["px4_process_group_id"] = os.getpgid(pid)
            result["px4_rootfs"] = str(rootfs)
            result["px4_readiness"] = self._probe_px4(
                generation, pid, start_ticks, episode_dir
            )
            flight_status, flight = self._run_flight(generation, episode_dir)
            result["flight_exit_code"] = flight_status
            result["flight_evidence"] = flight
            first = flight.get("first_evidence", {})
            result["takeoff_success"] = bool(
                "takeoff_altitude" in first
                and "isaac_takeoff_confirmed" in first
            )
            result["landing_success"] = bool(
                "px4_landed_confirmed" in first
                and "isaac_landing_confirmed" in first
            )
            result["disarm_success"] = "final_disarmed" in first
            if flight_status != 0 or not flight.get("success"):
                raise RuntimeError(
                    "flight failed: "
                    + str(flight.get("detail") or f"exit {flight_status}")
                )
            if not all(result[key] for key in (
                "takeoff_success", "landing_success", "disarm_success"
            )):
                raise RuntimeError("flight completed without required evidence")
        except Exception as error:
            result["failure_reason"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                if self.isaac is not None and self.isaac.poll() is None:
                    result["stop_evidence"] = self._lifecycle(
                        "stop_episode",
                        generation,
                        episode_dir,
                        "post_flight_stop",
                    )
            except Exception as stop_error:
                if not result["failure_reason"]:
                    result["failure_reason"] = (
                        f"{type(stop_error).__name__}: {stop_error}"
                    )
                result["stop_failure"] = str(stop_error)
            result["px4_shutdown"] = _stop_process(self.px4)
            if self.px4 is not None:
                result["px4_exit_code"] = self.px4.poll()
            self.px4 = None
            if self._px4_stream is not None:
                self._px4_stream.close()
                self._px4_stream = None
            result["success"] = not result["failure_reason"]
            self.summary["episodes"].append(result)
            self._save()
        return result

    def run(self) -> dict:
        try:
            self.preflight()
            self.start_runtime()
            boot = self.wait_for_boot()
            self.summary["initial_resource_identity"] = boot.get(
                "resource_identity", {}
            )
            self._save()
            for generation in range(1, self.episodes + 1):
                result = self.run_episode(generation)
                print(
                    f"[RuntimeSmoke] episode={generation} "
                    f"success={str(result['success']).lower()} "
                    f"reason={result['failure_reason'] or 'none'}",
                    flush=True,
                )
            identities = [
                episode.get("resource_identity", {})
                for episode in self.summary["episodes"]
                if episode.get("resource_identity")
            ]
            initial = self.summary.get("initial_resource_identity", {})
            self.summary["camera_recreated"] = any(
                identity != initial for identity in identities
            )
            px4_pids = [
                episode.get("px4_pid") for episode in self.summary["episodes"]
                if episode.get("px4_pid") is not None
            ]
            self.summary["px4_pids_distinct"] = (
                len(px4_pids) == self.episodes
                and len(set(px4_pids)) == len(px4_pids)
            )
            self.summary["isaac_restarted"] = len(set(
                self.summary["isaac_pid_observations"]
            )) > 1
            self.summary["success"] = bool(
                len(self.summary["episodes"]) == self.episodes
                and all(item["success"] for item in self.summary["episodes"])
                and self.summary["px4_pids_distinct"]
                and not self.summary["isaac_restarted"]
                and not self.summary["camera_recreated"]
            )
            if not self.summary["success"]:
                failures = [
                    item["failure_reason"] for item in self.summary["episodes"]
                    if item.get("failure_reason")
                ]
                self.summary["failure_reason"] = "; ".join(failures) or (
                    "runtime identity acceptance failed"
                )
        except Exception as error:
            self.summary["failure_reason"] = f"{type(error).__name__}: {error}"
            self.summary["success"] = False
        finally:
            self.cleanup()
            self._save()
        return self.summary

    def cleanup(self) -> None:
        owned_process_groups = [
            process.pid for process in (self.px4, self.isaac, self.xrce)
            if process is not None
        ]
        if self.px4 is not None:
            self.summary["final_px4_cleanup"] = _stop_process(self.px4)
            self.px4 = None
        self.summary["isaac_shutdown"] = _stop_process(
            self.isaac, interrupt_s=15.0, terminate_s=10.0
        )
        self.summary["xrce_shutdown"] = _stop_process(self.xrce)
        for stream_name in ("_px4_stream", "_isaac_stream", "_xrce_stream"):
            stream = getattr(self, stream_name)
            if stream is not None:
                stream.close()
                setattr(self, stream_name, None)
        self.summary["owned_process_groups_remaining"] = {
            str(process_group): _live_process_group_members(process_group)
            for process_group in owned_process_groups
            if _live_process_group_members(process_group)
        }
        self.summary["owned_processes_remaining"] = sorted({
            pid
            for members in self.summary[
                "owned_process_groups_remaining"
            ].values()
            for pid in members
        })


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restart external PX4 episodes inside one Isaac runtime."
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--runtime-timeout", type=float, default=120.0)
    parser.add_argument("--episode-timeout", type=float, default=190.0)
    parser.add_argument("--artifact-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    repository_root = Path(__file__).resolve().parents[2]
    artifact_root = arguments.artifact_root or (
        repository_root / "artifacts/runtime_smoke" / f"run_{_stamp()}"
    )
    smoke = PersistentRuntimeSmoke(
        repository_root=repository_root,
        episodes=arguments.episodes,
        artifact_root=artifact_root,
        runtime_timeout_s=arguments.runtime_timeout,
        episode_timeout_s=arguments.episode_timeout,
    )
    result = smoke.run()
    print(f"[RuntimeSmoke] summary={smoke.summary_path}", flush=True)
    print(
        f"[RuntimeSmoke] success={str(result['success']).lower()} "
        f"isaac_restarted={str(result['isaac_restarted']).lower()} "
        f"camera_recreated={str(result['camera_recreated']).lower()}",
        flush=True,
    )
    if result.get("failure_reason"):
        print(f"[RuntimeSmoke] failure={result['failure_reason']}", flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
