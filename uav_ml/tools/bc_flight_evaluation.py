"""Managed Isaac Sim, Pegasus, and PX4 evaluation for a TOP RGB BC policy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch

from uav_ml.inference.bc_flight import (
    TopRgbBcPolicy,
    canonical_image_source,
    resolve_checkpoint,
)


MSG_PREFLIGHT = "[BC Flight] Validating checkpoint and runtime..."
MSG_STARTING = "[BC Flight] Starting Isaac Sim, Pegasus, and PX4."
MSG_WEBRTC = (
    "[BC Flight] Open the Isaac Sim WebRTC client; the fixed TOP camera "
    "is the active viewport."
)
MSG_PREPARING = "[BC Flight] Preparing episode {episode} with seed {seed}."
MSG_RUNNING = "[BC Flight] BC flight started."
MSG_CLEANUP = "[BC Flight] Cleaning up managed processes..."
MSG_FINISHED = "[BC Flight] Evaluation finished: {path}"
MSG_RESULT = "[BC Flight] Episode {episode}: {reason}"
MSG_ERROR = "[BC Flight] Error: {error}"


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


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
    ) -> None:
        self.repository_root = repository_root
        self.isaac_release = isaac_release
        self.visible = visible
        self.device = device
        self.checkpoint = checkpoint
        self.image_source = image_source
        self.timeout_s = timeout_s
        self._agent: subprocess.Popen | None = None
        self._isaac: subprocess.Popen | None = None
        self._streams = []

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
        environment = os.environ.copy()
        environment["UAV_EXPERT_SENSORS"] = "1"
        environment["UAV_OBSERVER_VIEWPORT"] = "1" if self.visible else "0"
        environment.pop("DISPLAY", None)
        environment.pop("WAYLAND_DISPLAY", None)
        self._agent = subprocess.Popen(
            ["MicroXRCEAgent", "udp4", "-p", "8888"],
            cwd=self.repository_root,
            stdout=self._log(runtime_dir / "xrce.log"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
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
            stdout=self._log(runtime_dir / "isaac.log"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _uav(self, arguments: list[str], log_path: Path) -> int:
        environment = os.environ.copy()
        environment["UAV_OFFLINE_TIMEOUT_SECONDS"] = str(int(self.timeout_s))
        with log_path.open("w", encoding="utf-8") as stream:
            return subprocess.run(
                [str(self.repository_root / "uav"), *arguments],
                cwd=self.repository_root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=self.timeout_s + 45.0,
            ).returncode

    def run_episode(
        self,
        episode: int,
        seed: int,
        result_path: Path,
        runtime_dir: Path,
    ) -> dict:
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

    def cleanup(self) -> None:
        _stop_process(self._isaac)
        _stop_process(self._agent)
        self._isaac = None
        self._agent = None
        for stream in self._streams:
            stream.close()
        self._streams.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./uav bc-eval",
        description=(
            "Evaluate a TOP RGB BC checkpoint in Isaac Sim with Pegasus and PX4."
        ),
    )
    parser.add_argument("--image-source", default="top_rgb")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    display = parser.add_mutually_exclusive_group()
    display.add_argument(
        "--visible", action="store_true", help="Start the WebRTC runtime."
    )
    display.add_argument(
        "--headless", action="store_true", help="Run without a visible viewport."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes <= 0:
        raise ValueError("--episodes must be a positive integer")
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive")
    repository_root = Path(__file__).resolve().parents[2]
    image_source = canonical_image_source(args.image_source)
    checkpoint = resolve_checkpoint(repository_root, args.checkpoint)
    print(MSG_PREFLIGHT, flush=True)
    policy = TopRgbBcPolicy(
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
    isaac_release = Path(os.environ.get(
        "UAV_ISAAC_SIM_RELEASE",
        str(Path.home() / "isaacsim/_build/linux-x86_64/release"),
    )).expanduser().resolve()
    results = []
    for index in range(args.episodes):
        episode = index + 1
        seed = args.seed + index
        episode_root = output_root / f"episode_{episode:06d}"
        result_path = episode_root / "result.json"
        runtime = ManagedFlightRuntime(
            repository_root,
            isaac_release,
            bool(args.visible),
            args.device,
            checkpoint,
            image_source,
            args.timeout,
        )
        runtime.preflight()
        print(MSG_STARTING, flush=True)
        if args.visible:
            print(MSG_WEBRTC, flush=True)
        try:
            runtime.start(episode_root)
            print(
                MSG_PREPARING.format(episode=episode, seed=seed), flush=True
            )
            print(MSG_RUNNING, flush=True)
            result = runtime.run_episode(
                episode, seed, result_path, episode_root
            )
            results.append(result)
            print(MSG_RESULT.format(
                episode=episode,
                reason=result.get("terminal_reason", "unknown"),
            ), flush=True)
        finally:
            print(MSG_CLEANUP, flush=True)
            runtime.cleanup()
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps({
        "schema": "uav_bc_flight_evaluation/v1",
        "image_source": image_source,
        "checkpoint": identity.checkpoint_path,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "episodes": results,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(MSG_FINISHED.format(path=summary_path), flush=True)
    return 0


def cli() -> int:
    """Report expected startup/runtime failures without a Python traceback."""
    try:
        return main()
    except KeyboardInterrupt:
        print(MSG_ERROR.format(error="interrupted by user"), file=sys.stderr)
        return 130
    except (
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        print(MSG_ERROR.format(error=error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
