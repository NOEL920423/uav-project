"""Attach flight evidence and enforce safe terminal state for one episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def finalize(
    dataset_root: Path,
    episode_id: str,
    evidence_path: Path,
    expected_success: bool | None,
) -> dict:
    episode_path = dataset_root / episode_id / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    actual_success = bool(evidence.get("success"))
    if expected_success is not None and actual_success != expected_success:
        raise ValueError(
            f"{episode_id}: expected success={expected_success}, "
            f"flight evidence reported {actual_success}"
        )
    px4 = evidence.get("final_px4") or {}
    safe = {
        "landed": px4.get("landed") is True,
        "disarmed": px4.get("arming_state") == 1,
        "failsafe": bool(px4.get("failsafe", True)),
        "arming_state": px4.get("arming_state"),
        "nav_state": px4.get("nav_state"),
    }
    if not safe["landed"] or not safe["disarmed"] or safe["failsafe"]:
        raise ValueError(f"{episode_id}: unsafe terminal PX4 evidence: {safe}")
    if bool(episode.get("success")) != actual_success:
        raise ValueError(
            f"{episode_id}: recorder and flight outcome disagree"
        )
    episode["safe_terminal_evidence"] = safe
    episode["flight_evidence"] = {
        "path": f"{episode_id}/flight_evidence.json",
        "success": actual_success,
        "detail": evidence.get("detail"),
        "elapsed_s": evidence.get("elapsed_s"),
        "maximum_stream_rate_hz": evidence.get("maximum_stream_rate_hz"),
        "stream_faults": evidence.get("stream_faults"),
        "max_altitude_m": evidence.get("max_altitude_m"),
        "minimum_goal_distance_m": evidence.get("minimum_goal_distance_m"),
        "isaac": evidence.get("isaac"),
    }
    episode["flight_duration_s"] = evidence.get("elapsed_s")
    episode["terminal_reason"] = (
        "goal_reached_and_landed"
        if actual_success else evidence.get("detail") or episode.get("failure")
    )
    episode["goal_distance_m"] = episode.get(
        "final_tracking_goal_distance_m"
    )
    episode_path.write_text(
        json.dumps(episode, indent=2) + "\n", encoding="utf-8"
    )
    episode["episode_disk_usage_bytes"] = sum(
        item.stat().st_size
        for item in episode_path.parent.rglob("*")
        if item.is_file()
    )
    episode_path.write_text(
        json.dumps(episode, indent=2) + "\n", encoding="utf-8"
    )
    return episode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument(
        "--expected", choices=("success", "failure", "any"), default="any"
    )
    args = parser.parse_args()
    result = finalize(
        Path(args.dataset),
        args.episode,
        Path(args.evidence),
        None if args.expected == "any" else args.expected == "success",
    )
    print(json.dumps({
        "episode_id": result["episode_id"],
        "success": result["success"],
        "safe_terminal_evidence": result["safe_terminal_evidence"],
    }, indent=2))


if __name__ == "__main__":
    main()
