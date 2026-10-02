"""Managed Isaac Sim, Pegasus, and PX4 evaluation for a TOP RGB BC policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

import torch

from uav_ml.inference.bc_flight import (
    BcFlightPolicy,
    canonical_image_source,
    resolve_checkpoint,
)


MSG_PREFLIGHT = "[BC Flight] Validating checkpoint and runtime..."
MSG_STARTING = "[BC Flight] Starting Isaac Sim, Pegasus, and PX4."
MSG_WEBRTC = (
    "[BC Flight] Open the Isaac Sim WebRTC client; the selected policy "
    "camera is the active viewport."
)
MSG_PREPARING = "[BC Flight] Preparing episode {episode} with seed {seed}."
MSG_RUNNING = "[BC Flight] BC flight started."
MSG_CLEANUP = "[BC Flight] Cleaning up managed processes..."
MSG_FINISHED = "[BC Flight] Evaluation finished: {path}"
MSG_RESULT = "[BC Flight] Episode {episode}: {reason}"
MSG_PLOTS = "[BC Flight] Plots saved under: {path}"
MSG_PLOT_WARNING = "[BC Flight] Plot generation failed: {error}"
MSG_ERROR = "[BC Flight] Error: {error}"
_LAST_ARTIFACT_PATH: Path | None = None
_ERROR_MARKERS = ("ERROR", "FATAL", "TRACEBACK", "EXCEPTION", "UNEXPECTED EXIT")
ENABLE_BC_FLIGHT_DIAGNOSTICS = True


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M")


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    for sent_signal, timeout_s in (
        (signal.SIGINT, 10.0),
        (signal.SIGTERM, 5.0),
        (signal.SIGKILL, 2.0),
    ):
        try:
            os.killpg(process.pid, sent_signal)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            continue


def _process_exists(arguments: list[str]) -> bool:
    return subprocess.run(
        arguments,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def _is_error_line(line: str) -> bool:
    upper = line.upper()
    return any(marker in upper for marker in _ERROR_MARKERS) or "STALE" in upper


def _display_line(line: str) -> str:
    """Condense large startup diagnostics while preserving stale fault context."""
    if "BC_STARTUP_DIAG" not in line or "STALE" not in line.upper():
        return line
    try:
        payload = json.loads(line[line.index("{"):])
    except (ValueError, json.JSONDecodeError):
        return line
    values = {"component": payload.get("component"), "state": payload.get("state")}
    for key, value in payload.items():
        if any(token in key.lower() for token in ("reason", "age", "timeout", "latch")):
            values[key] = value
    return "[BC Flight] STALE_FAULT " + json.dumps(values, sort_keys=True) + "\n"


def _tail(path: Path, count: int = 20) -> str:
    if not path.is_file():
        return ""
    return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-count:])


def _pump(stream, output, *, label: str, verbose: bool) -> None:
    for line in iter(stream.readline, ""):
        output.write(line)
        output.flush()
        if verbose or _is_error_line(line) or label.endswith("stderr"):
            print(_display_line(line), end="", file=sys.stderr if label.endswith("stderr") else sys.stdout, flush=True)
    stream.close()


class ManagedFlightRuntime:
    """Own the XRCE, Isaac/Pegasus/PX4, and ROS launch process groups."""

    def __init__(
        self,
        repository_root: Path,
        isaac_release: Path,
        visible: bool,
        device: str,
        checkpoint: Path,
        image_source: str,
        timeout_s: float,
        verbose: bool = False,
        diagnostics_enabled: bool = True,
    ) -> None:
        self.repository_root = repository_root
        self.isaac_release = isaac_release
        self.visible = visible
        self.device = device
        self.checkpoint = checkpoint
        self.image_source = image_source
        self.timeout_s = timeout_s
        self.verbose = verbose
        self.diagnostics_enabled = diagnostics_enabled
        self._agent: subprocess.Popen | None = None
        self._isaac: subprocess.Popen | None = None
        self._streams = []
        self._pump_threads: list[threading.Thread] = []
        self._episode_start_time_s: float | None = None
        self._runtime_dir: Path | None = None
        self._capture_stop = threading.Event()
        self._capture_thread = None
        self._px4_process = None
        self._ulog_handles = {}

    def preflight(self) -> None:
        launcher = self.isaac_release / (
            "isaac-sim.streaming.sh" if self.visible else "isaac-sim.sh"
        )
        if not launcher.is_file() or not os.access(launcher, os.X_OK):
            raise FileNotFoundError(f"Isaac Sim launcher is missing: {launcher}")
        if shutil.which("MicroXRCEAgent") is None:
            raise FileNotFoundError("MicroXRCEAgent is not available")
        overlay = self.repository_root / "ros2_ws/install/setup.bash"
        if not overlay.is_file():
            raise FileNotFoundError(
                f"ROS workspace overlay is missing: {overlay}; run ./uav build"
            )
        if (
            _process_exists(["pgrep", "-x", "MicroXRCEAgent"])
            or _process_exists(["pgrep", "-f", str(self.isaac_release / "kit/kit")])
            or _process_exists([
                "pgrep", "-f", "/PX4-Autopilot/build/px4_sitl_default/bin/px4"
            ])
        ):
            raise RuntimeError(
                "managed BC flight requires no pre-existing Isaac/PX4/XRCE process"
            )

    def _log(self, path: Path):
        stream = path.open("w", encoding="utf-8")
        self._streams.append(stream)
        return stream

    def start(self, runtime_dir: Path) -> None:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._episode_start_time_s = time.time()
        self._runtime_dir = runtime_dir
        environment = os.environ.copy()
        environment["UAV_BC_FLIGHT_DIAGNOSTICS"] = (
            "1" if self.diagnostics_enabled else "0"
        )
        if self.diagnostics_enabled:
            environment["UAV_TIMING_DIR"] = str(
                (runtime_dir / "timing").resolve()
            )
            # PX4 rcS sources this override before starting its diagnostic logger.
            px4_params = runtime_dir / "px4-rc.params"
            original_params = Path.home() / (
                "PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/px4-rc.params"
            )
            px4_params.write_text(
                f". {shlex.quote(str(original_params))}\nparam set SDLOG_MODE -1\n",
                encoding="utf-8",
            )
            environment["PATH"] = (
                f"{runtime_dir.resolve()}{os.pathsep}"
                f"{environment.get('PATH', '')}"
            )
        else:
            environment.pop("UAV_TIMING_DIR", None)
            environment.pop("UAV_TIMING_SCHED_SECONDS", None)
        environment["UAV_EXPERT_SENSORS"] = "1"
        environment["UAV_OBSERVER_VIEWPORT"] = "1" if self.visible else "0"
        environment["UAV_VIEWPORT_SOURCE"] = self.image_source
        environment.pop("DISPLAY", None)
        environment.pop("WAYLAND_DISPLAY", None)
        self._agent = subprocess.Popen(
            ["MicroXRCEAgent", "udp4", "-p", "8888"],
            cwd=self.repository_root,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
            start_new_session=True,
        )
        self._attach_streams(self._agent, runtime_dir / "xrce.log", "MicroXRCEAgent")
        launcher = self.isaac_release / (
            "isaac-sim.streaming.sh" if self.visible else "isaac-sim.sh"
        )
        command = [str(launcher)]
        if self.visible:
            command.extend(["--livestream", "2"])
        else:
            command.append("--no-window")
        command.extend([
            "--exec", str(self.repository_root / "isaac/runtime/bootstrap.py")
        ])
        self._isaac = subprocess.Popen(
            command,
            cwd=self.isaac_release,
            env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
            start_new_session=True,
        )
        self._attach_streams(self._isaac, runtime_dir / "isaac.log", "Isaac Sim/Pegasus/PX4")
        if self.diagnostics_enabled:
            self._capture_thread = threading.Thread(
                target=self._capture_ulog, daemon=True
            )
            self._capture_thread.start()

    def _owns_px4_process(self, pid: int) -> bool:
        """Restrict logger commands to descendants of this episode's Isaac."""
        if self._isaac is None:
            return False
        visited = set()
        while pid > 1 and pid not in visited:
            if pid == self._isaac.pid:
                return True
            visited.add(pid)
            try:
                status = Path(f"/proc/{pid}/status").read_text()
                pid = int(next(line.split()[1] for line in status.splitlines()
                               if line.startswith("PPid:")))
            except (OSError, ValueError, StopIteration):
                return False
        return False

    def _capture_ulog(self) -> None:
        """Hold open ULogs so temporary-rootfs deletion cannot lose evidence."""
        destination = self._runtime_dir / "px4_ulog"
        destination.mkdir(exist_ok=True)
        logger_started = False
        while not self._capture_stop.wait(0.25):
            if self._px4_process is None:
                for proc in Path("/proc").iterdir():
                    if not proc.name.isdigit():
                        continue
                    try:
                        executable = (proc / "exe").resolve(strict=True)
                        if str(executable).endswith("/px4_sitl_default/bin/px4"):
                            if not self._owns_px4_process(int(proc.name)):
                                continue
                            self._px4_process = (int(proc.name), executable, (proc / "cwd").resolve(strict=True))
                            break
                    except OSError:
                        continue
            if self._px4_process is None:
                continue
            pid, executable, cwd = self._px4_process
            if not logger_started:
                try:
                    # Diagnostic recorder only; no PX4 parameter is changed.
                    with (destination / "logger_command.log").open("a") as log:
                        result = subprocess.run(
                            [str(executable.with_name("px4-logger")), "start", "-e"],
                            cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=2,
                        )
                    logger_started = result.returncode == 0
                    if not logger_started:
                        print(f"[BC Flight] logger start returned {result.returncode}: {destination / 'logger_command.log'}", file=sys.stderr)
                except (OSError, subprocess.TimeoutExpired) as error:
                    print(f"[BC Flight] ULog recorder error: {error}", file=sys.stderr)
            try:
                for source in (cwd / "log").rglob("*.ulg"):
                    if str(source) not in self._ulog_handles:
                        self._ulog_handles[str(source)] = source.open("rb")
            except OSError as error:
                print(f"[BC Flight] ULog discovery error: {error}", file=sys.stderr)
        self._write_ulog_snapshot(destination)

    def _write_ulog_snapshot(self, destination: Path) -> None:
        copied = []
        for source, handle in self._ulog_handles.items():
            try:
                handle.seek(0)
                target = destination / Path(source).name
                with target.open("wb") as output:
                    shutil.copyfileobj(handle, output)
                copied.append({"path": target.name, "size": target.stat().st_size})
            except OSError as error:
                print(f"[BC Flight] ULog copy failed: {error}", file=sys.stderr)
            finally:
                handle.close()
        (destination / "manifest.json").write_text(json.dumps({
            "copied_files": copied, "pid": self._px4_process[0] if self._px4_process else None,
            "failure_reason": "" if copied else "No ULog captured; see logger_command.log",
            "snapshot": "logger stopped before runtime cleanup",
        }, indent=2), encoding="utf-8")
        if not copied:
            print(f"[BC Flight] ULog missing: {destination / 'manifest.json'}", file=sys.stderr)

    def _attach_streams(self, process: subprocess.Popen, log_path: Path, name: str) -> None:
        output = self._log(log_path)
        for stream, label in ((process.stdout, f"{name} stdout"), (process.stderr, f"{name} stderr")):
            if stream is not None:
                thread = threading.Thread(target=_pump, args=(stream, output), kwargs={"label": label, "verbose": self.verbose}, daemon=True)
                self._pump_threads.append(thread)
                thread.start()

    def _uav(self, arguments: list[str], log_path: Path) -> int:
        environment = os.environ.copy()
        environment["UAV_BC_FLIGHT_DIAGNOSTICS"] = (
            "1" if self.diagnostics_enabled else "0"
        )
        if self.diagnostics_enabled:
            environment["UAV_TIMING_DIR"] = str(
                (log_path.parent / "timing").resolve()
            )
        else:
            environment.pop("UAV_TIMING_DIR", None)
            environment.pop("UAV_TIMING_SCHED_SECONDS", None)
        environment["UAV_OFFLINE_TIMEOUT_SECONDS"] = str(int(self.timeout_s))
        process = subprocess.Popen(
                [str(self.repository_root / "uav"), *arguments],
                cwd=self.repository_root,
                env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
            )
        self._attach_streams(process, log_path, "./uav " + arguments[0])
        try:
            code = process.wait(timeout=self.timeout_s + 45.0)
        except subprocess.TimeoutExpired:
            _stop_process(process)
            raise
        if code != 0:
            detail = _tail(log_path)
            print(f"[BC Flight] subprocess failure: stage={arguments[0]} process=./uav {arguments[0]} return code={code}\n{detail}[BC Flight] log: {log_path}", file=sys.stderr, flush=True)
        return code

    def run_episode(
        self,
        episode: int,
        seed: int,
        result_path: Path,
        runtime_dir: Path,
    ) -> dict:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        if self._uav(["expert-runtime-wait"], runtime_dir / "ready.log") != 0:
            raise RuntimeError("Isaac/PX4 runtime did not become ready")
        episode_id = f"episode_{episode:06d}"
        if self._uav(
            ["expert-scene-prepare", episode_id, str(seed)],
            runtime_dir / "scene.log",
        ) != 0:
            raise RuntimeError("seeded Isaac scene preparation failed")
        arguments = [
            "bc-flight-run",
            str(result_path),
            str(episode),
            str(seed),
            self.image_source,
            str(self.checkpoint),
            sys.executable,
            self.device,
        ]
        status = self._uav(arguments, runtime_dir / "flight.log")
        if not result_path.is_file():
            raise RuntimeError(
                f"BC flight result is missing after launch status {status}"
            )
        return json.loads(result_path.read_text(encoding="utf-8"))

    def replay_trace(
        self, trace_path: Path, playback_speed: float, runtime_dir: Path,
    ) -> None:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if trace.get("schema") != "uav_bc_flight_trace/v1":
            raise ValueError("replay trace has an unsupported schema")
        seed = int(trace["seed"])
        episode_id = f"episode_{int(trace.get('episode', 0)):06d}"
        if self._uav(
            ["expert-scene-prepare", episode_id, str(seed)],
            runtime_dir / "scene.log",
        ) != 0:
            raise RuntimeError("trace replay scene preparation failed")
        if self._uav(
            ["isaac-trace-replay", str(trace_path), str(playback_speed)],
            runtime_dir / "replay.log",
        ) != 0:
            raise RuntimeError("Isaac trace replay failed")

    def save_ulog(self, runtime_dir: Path, abnormal: bool) -> None:
        """Snapshot PX4 ULog files before stopping managed processes."""
        if not abnormal:
            return
        destination = runtime_dir / "px4_ulog"
        destination.mkdir(exist_ok=True)
        px4_pid = None
        px4_cwd = None
        failure_reason = ""
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if "/PX4-Autopilot/build/px4_sitl_default/bin/px4" not in command:
                    continue
                px4_pid = int(proc.name)
                px4_cwd = Path(os.readlink(proc / "cwd"))
                break
            except (OSError, ValueError):
                continue
        if px4_pid is None or px4_cwd is None:
            failure_reason = "live PX4 process PID/cwd could not be obtained"
        roots = [] if px4_cwd is None else [px4_cwd / "log"]
        copied = []
        cutoff = (self._episode_start_time_s or time.time()) - 1.0
        candidates = []
        for root in roots:
            if not root.is_dir():
                continue
            for source in root.rglob("*.ulg"):
                if source.is_file():
                    stat = source.stat()
                    selected = stat.st_mtime >= cutoff
                    record = {"source_path": str(source), "mtime": stat.st_mtime,
                              "size": stat.st_size, "selected": selected,
                              "reason": "created_or_updated_after_episode_start" if selected
                              else "older_than_episode_start"}
                    candidates.append(record)
                    if selected:
                        target = destination / source.name
                        try:
                            shutil.copy2(source, target)
                        except OSError as error:
                            record["selected"] = False
                            record["reason"] = f"copy_failed:{error}"
                        else:
                            copied.append(record)
        (destination / "manifest.json").write_text(
            json.dumps({"copied_files": copied, "candidates": candidates,
                        "pid": px4_pid, "cwd": None if px4_cwd is None else str(px4_cwd),
                        "failure_reason": failure_reason,
                        "episode_start_time": self._episode_start_time_s,
                        "cutoff_time": cutoff,
                        "searched_roots": [str(r) for r in roots]}, indent=2),
            encoding="utf-8",
        )

    def cleanup(self) -> None:
        if self._capture_thread is not None:
            if self._px4_process:
                _, executable, cwd = self._px4_process
                try:
                    with (self._runtime_dir / "px4_ulog/logger_command.log").open("a") as log:
                        stopped = subprocess.run(
                            [str(executable.with_name("px4-logger")), "stop"],
                            cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=3,
                        )
                    if stopped.returncode:
                        print(f"[BC Flight] logger stop returned {stopped.returncode}", file=sys.stderr)
                except (OSError, subprocess.TimeoutExpired) as error:
                    print(f"[BC Flight] logger stop failed: {error}", file=sys.stderr)
            self._capture_stop.set()
            self._capture_thread.join(timeout=5)
        _stop_process(self._isaac)
        _stop_process(self._agent)
        self._isaac = None
        self._agent = None
        for thread in self._pump_threads:
            thread.join(timeout=2.0)
        self._pump_threads.clear()
        for stream in self._streams:
            stream.close()
        self._streams.clear()
        if self._runtime_dir is not None and self.diagnostics_enabled:
            try:
                from scripts.diagnostics.summarize_bc_startup import write_timing_report
                write_timing_report(self._runtime_dir)
                print(f"[BC Flight] Timing report: {self._runtime_dir / 'timing_summary.md'}", flush=True)
            except Exception:
                import traceback
                traceback.print_exc()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./uav bc-eval",
        description=(
            "Evaluate a source-matched BC checkpoint in Isaac Sim with Pegasus and PX4."
        ),
    )
    parser.add_argument("--image-source", default="top_rgb")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--replay-trace", type=Path,
        help="Visually replay a recorded trajectory trace; does not run BC.",
    )
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--nas-root",
        type=Path,
        default=(Path(os.environ["UAV_NAS_ROOT"])
                 if os.environ.get("UAV_NAS_ROOT") else None),
        help=(
            "publish this completed evaluation under NAS/evaluations/bc_flight "
            "and delete verified local artifacts"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Show all subprocess stdout.")
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--visible", action="store_true", help="Start the WebRTC runtime."
    )
    display.add_argument(
        "--headless", action="store_true", help="Run without a visible viewport."
    )
    return parser


def _saved_result_records(output_root: Path, results: list[dict]) -> list[dict]:
    """Return valid completed results, including ones written before interruption."""
    records = {}
    for result in results:
        try:
            records[int(result["episode"])] = result
        except (KeyError, TypeError, ValueError):
            continue
    for result_path in sorted(output_root.glob("episode_*/result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("schema") != "uav_bc_flight_result/v1":
                raise ValueError("unsupported result schema")
            records[int(result["episode"])] = result
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            print(
                f"[BC Flight] Ignoring unreadable result artifact {result_path}: "
                f"{error}",
                file=sys.stderr,
                flush=True,
            )
    return [records[episode] for episode in sorted(records)]


def _finalize_episode_video(result_path: Path) -> dict:
    """Render locally captured policy inputs and attach the outcome to result."""
    result = json.loads(result_path.read_text(encoding="utf-8"))
    episode_root = result_path.parent
    output_root = episode_root.parent
    videos_dir = output_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    spool_dir = episode_root / "policy_input_frames"
    video_path = videos_dir / f"episode_{int(result['episode']):06d}.mp4"
    metadata_path = videos_dir / f"episode_{int(result['episode']):06d}.json"
    video = {
        "filename": str(video_path.relative_to(output_root)),
        "metadata_filename": str(metadata_path.relative_to(output_root)),
    }
    try:
        from uav_ml.tools.bc_flight_video import render_policy_input_video

        video.update(render_policy_input_video(
            spool_dir, video_path, metadata_path=metadata_path
        ))
        video["filename"] = str(video_path.relative_to(output_root))
        video["metadata_filename"] = str(metadata_path.relative_to(output_root))
        video["status"] = "complete"
        shutil.rmtree(spool_dir)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        video["status"] = "failed"
        video["failure_reason"] = str(error)
        print(
            f"[BC Flight] Policy input video failed: {error}",
            file=sys.stderr,
            flush=True,
        )
    result["policy_input_video"] = video
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(result_path)
    return result


def _finalize_evaluation(
    output_root: Path,
    results: list[dict],
    *,
    image_source: str,
    identity,
    diagnostics_enabled: bool = True,
) -> None:
    """Persist available evidence even if a later runtime startup fails."""
    records = _saved_result_records(output_root, results)
    plots = {}
    plot_error = ""
    try:
        from uav_ml.tools.bc_flight_plotting import generate_evaluation_plots

        plots = generate_evaluation_plots(output_root, records)
        print(MSG_PLOTS.format(path=output_root), flush=True)
    except Exception as error:  # Plotting must not invalidate flight evidence.
        plot_error = str(error)
        print(MSG_PLOT_WARNING.format(error=error), file=sys.stderr, flush=True)
    summary = {
        "schema": "uav_bc_flight_evaluation/v1",
        "image_source": image_source,
        "image_sources": list(getattr(identity, "image_sources", (image_source,))),
        "checkpoint": identity.checkpoint_path,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "episodes": records,
        "plots": plots,
    }
    if plot_error:
        summary["plot_error"] = plot_error
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if diagnostics_enabled:
        try:
            from scripts.diagnostics.summarize_bc_startup import write_run_reader_summary
            write_run_reader_summary(output_root)
            print(
                f"[BC Flight] Control summary: "
                f"{output_root / 'closed_loop_control_summary.md'}",
                flush=True,
            )
        except Exception:
            import traceback
            traceback.print_exc()
    print(MSG_FINISHED.format(path=summary_path), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be a positive integer")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")
    if args.playback_speed <= 0.0:
        raise ValueError("--playback-speed must be positive")
    repository_root = Path(__file__).resolve().parents[2]
    if args.replay_trace:
        return _replay_trace(args, repository_root)
    image_source = canonical_image_source(args.image_source)
    checkpoint = resolve_checkpoint(repository_root, args.checkpoint)
    if ENABLE_BC_FLIGHT_DIAGNOSTICS and shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg is required for BC policy-input diagnostic video recording"
        )
    print(MSG_PREFLIGHT, flush=True)
    policy = BcFlightPolicy(
        checkpoint, image_source, torch.device(args.device)
    )
    identity = policy.identity
    del policy
    output_root = (
        args.output.expanduser().resolve()
        if args.output
        else repository_root / "artifacts/evaluations/bc_flight" / f"run_{_stamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    global _LAST_ARTIFACT_PATH
    _LAST_ARTIFACT_PATH = output_root
    isaac_release = Path(os.environ.get(
        "UAV_ISAAC_SIM_RELEASE",
        str(Path.home() / "isaacsim/_build/linux-x86_64/release"),
    )).expanduser().resolve()
    results = []
    try:
        for index in range(args.episodes):
            episode = index + 1
            seed = args.seed + index
            episode_root = output_root / f"episode_{episode:06d}"
            result_path = episode_root / "result.json"
            runtime = ManagedFlightRuntime(
                repository_root, isaac_release, bool(args.visible), args.device,
                checkpoint, image_source, args.timeout, args.verbose,
                ENABLE_BC_FLIGHT_DIAGNOSTICS,
            )
            runtime.preflight()
            print(MSG_STARTING, flush=True)
            if args.visible:
                print(MSG_WEBRTC, flush=True)
            try:
                runtime.start(episode_root)
                print(MSG_PREPARING.format(episode=episode, seed=seed), flush=True)
                print(MSG_RUNNING, flush=True)
                result = runtime.run_episode(episode, seed, result_path, episode_root)
                if ENABLE_BC_FLIGHT_DIAGNOSTICS:
                    result = _finalize_episode_video(result_path)
                results.append(result)
                print(MSG_RESULT.format(episode=episode, reason=result.get("terminal_reason", "unknown")), flush=True)
            finally:
                print(MSG_CLEANUP, flush=True)
                runtime.cleanup()
    finally:
        _finalize_evaluation(
            output_root,
            results,
            image_source=image_source,
            identity=identity,
            diagnostics_enabled=ENABLE_BC_FLIGHT_DIAGNOSTICS,
        )
    if args.nas_root is not None:
        from uav_ml.tools.artifact_publish import publish_artifact_tree

        destination = publish_artifact_tree(
            output_root,
            args.nas_root,
            "evaluations/bc_flight",
            delete_local=True,
        )
        print(f"[BC Flight] Published verified artifact: {destination}", flush=True)
    return 0


def _replay_trace(args: argparse.Namespace, repository_root: Path) -> int:
    """Run a visual-only trajectory replay in the managed Isaac runtime."""
    trace_path = args.replay_trace.expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"replay trace is missing: {trace_path}")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if trace.get("schema") != "uav_bc_flight_trace/v1":
        raise ValueError("--replay-trace must be uav_bc_flight_trace/v1")
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ValueError("--replay-trace must contain at least two samples")
    output_root = (
        args.output.expanduser().resolve()
        if args.output
        else repository_root / "artifacts/evaluations/bc_flight" / f"replay_{_stamp()}"
    )
    output_root.mkdir(parents=True, exist_ok=False)
    global _LAST_ARTIFACT_PATH
    _LAST_ARTIFACT_PATH = output_root
    isaac_release = Path(os.environ.get(
        "UAV_ISAAC_SIM_RELEASE",
        str(Path.home() / "isaacsim/_build/linux-x86_64/release"),
    )).expanduser().resolve()
    runtime = ManagedFlightRuntime(
        repository_root, isaac_release, bool(args.visible), args.device,
        trace_path, str(trace.get("image_source", "top_rgb")), args.timeout,
        args.verbose, ENABLE_BC_FLIGHT_DIAGNOSTICS,
    )
    print(MSG_PREFLIGHT, flush=True)
    runtime.preflight()
    print(MSG_STARTING, flush=True)
    if args.visible:
        print(MSG_WEBRTC, flush=True)
    try:
        runtime.start(output_root)
        if runtime._uav(["expert-runtime-wait"], output_root / "ready.log") != 0:
            raise RuntimeError("Isaac/PX4 runtime did not become ready")
        runtime.replay_trace(trace_path, args.playback_speed, output_root)
    finally:
        print(MSG_CLEANUP, flush=True)
        runtime.cleanup()
    digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    result = {
        "schema": "uav_bc_flight_trace_replay/v1",
        "trace_path": str(trace_path), "trace_sha256": digest,
        "episode": trace.get("episode"), "seed": trace.get("seed"),
        "sample_count": len(samples), "duration_s": samples[-1]["time_s"],
        "playback_speed": args.playback_speed, "completed": True,
        "visual_only": True,
    }
    (output_root / "replay_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(MSG_FINISHED.format(path=output_root / "replay_result.json"), flush=True)
    return 0


def cli() -> int:
    """Report expected startup/runtime failures without a Python traceback."""
    try:
        return main()
    except KeyboardInterrupt:
        print(MSG_ERROR.format(error="interrupted by user"), file=sys.stderr)
        return 130
    except Exception as error:
        import traceback
        traceback.print_exc()
        if _LAST_ARTIFACT_PATH:
            print(f"[BC Flight] Artifacts: {_LAST_ARTIFACT_PATH}", file=sys.stderr)
        print(MSG_ERROR.format(error=error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
