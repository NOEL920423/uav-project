"""Two-generation PX4 smoke test against one persistent Isaac runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from uav_ml.tools.persistent_runtime import (
    PersistentRuntimeManager,
    atomic_json as _atomic_json,
    live_process_group_members as _live_process_group_members,
    process_start_ticks as _process_start_ticks,
    read_json as _read_json,
    stop_process as _stop_process,
)


__all__ = (
    "_atomic_json",
    "_live_process_group_members",
    "_process_start_ticks",
    "_read_json",
    "_stop_process",
)


SCHEMA = "uav_expert_runtime_smoke/v1"


def _stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


class PersistentRuntimeSmoke:
    """Run sequential PX4 flights through the shared persistent manager."""

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
        self.runtime = PersistentRuntimeManager(
            self.repository_root,
            self.artifact_root,
            runtime_timeout_s=runtime_timeout_s,
        )
        self.started = time.monotonic()
        self.summary = {
            "schema": SCHEMA,
            "artifact_root": str(self.artifact_root),
            "isaac_pid": None,
            "xrce_pid": None,
            "episode_count": episodes,
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
        self.runtime.preflight()

    def start_runtime(self) -> None:
        evidence = self.runtime.start_job()
        self.summary.update({
            "isaac_pid": evidence["isaac_pid"],
            "xrce_pid": evidence["xrce_pid"],
            "initial_resource_identity": evidence["resource_identity"],
            "px4_ownership": evidence["px4_ownership"],
            "pegasus_px4_autolaunch": evidence["pegasus_px4_autolaunch"],
            "pegasus_lockstep_enabled": evidence[
                "pegasus_lockstep_enabled"
            ],
        })
        print(f"[RuntimeSmoke] xrce_pid={evidence['xrce_pid']}", flush=True)
        print(f"[RuntimeSmoke] isaac_pid={evidence['isaac_pid']}", flush=True)
        self._save()

    def _run_flight(
        self, generation: int, episode_dir: Path
    ) -> tuple[int, dict]:
        evidence_path = episode_dir / "flight_evidence.json"
        timeout = max(30.0, self.episode_timeout_s)
        status = self.runtime.run_ros([
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
            "reset_success": False,
            "px4_readiness": {},
            "takeoff_success": False,
            "landing_success": False,
            "disarm_success": False,
            "failure_reason": "",
        }
        try:
            reset = self.runtime.prepare_attempt(generation, episode_dir)
            result["reset_success"] = bool(
                reset.get("reset_evidence")
                and all(reset["reset_evidence"].values())
            )
            px4 = self.runtime.start_px4(generation, episode_dir)
            result.update(px4)
            result["px4_readiness"] = self.runtime.probe_px4(
                generation, episode_dir
            )
            flight_status, flight = self._run_flight(
                generation, episode_dir
            )
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
                raise RuntimeError(
                    "flight completed without required evidence"
                )
        except Exception as error:
            result["failure_reason"] = f"{type(error).__name__}: {error}"
        finally:
            try:
                self.runtime.stop_episode(generation, episode_dir)
            except Exception as stop_error:
                if not result["failure_reason"]:
                    result["failure_reason"] = (
                        f"{type(stop_error).__name__}: {stop_error}"
                    )
            result.update(self.runtime.attempt_evidence)
            result["success"] = not result["failure_reason"]
            self.summary["episodes"].append(result)
            self._save()
        return result

    def run(self) -> dict:
        try:
            self.preflight()
            self.start_runtime()
            for generation in range(1, self.episodes + 1):
                result = self.run_episode(generation)
                print(
                    f"[RuntimeSmoke] episode={generation} "
                    f"success={str(result['success']).lower()} "
                    f"reason={result['failure_reason'] or 'none'}",
                    flush=True,
                )
            pids = [item.get("px4_pid") for item in self.summary["episodes"]]
            self.summary["px4_pids_distinct"] = bool(
                len(pids) == self.episodes
                and None not in pids
                and len(set(pids)) == len(pids)
            )
            self.summary["success"] = bool(
                all(item["success"] for item in self.summary["episodes"])
                and self.summary["px4_pids_distinct"]
            )
            if not self.summary["success"]:
                self.summary["failure_reason"] = "; ".join(
                    item["failure_reason"] for item in self.summary["episodes"]
                    if item.get("failure_reason")
                ) or "runtime identity acceptance failed"
        except Exception as error:
            self.summary["failure_reason"] = f"{type(error).__name__}: {error}"
            self.summary["success"] = False
        finally:
            job = self.runtime.cleanup_job()
            self.summary["job_runtime"] = job
            self.summary["isaac_pid_observations"] = job.get(
                "isaac_pid_observations", []
            )
            self.summary["isaac_restarted"] = bool(
                job.get("isaac_restart_count", 0)
            )
            self.summary["camera_recreated"] = bool(
                job.get("camera_recreated", False)
            )
            self.summary["owned_processes_remaining"] = job.get(
                "owned_processes_remaining", []
            )
            self.summary["success"] = bool(
                self.summary["success"]
                and not self.summary["isaac_restarted"]
                and not self.summary["camera_recreated"]
                and not self.summary["owned_processes_remaining"]
            )
            self._save()
        return self.summary


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
