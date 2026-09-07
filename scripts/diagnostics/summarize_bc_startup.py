"""Summarize startup transitions and preserve missing-evidence limitations."""

import argparse
import json
from collections import Counter
from pathlib import Path


MARKER = "BC_STARTUP_DIAG "


def summarize(root: Path) -> str:
    """Extract ordered events and the first recorded fault latch."""
    lines = ["# BC startup diagnosis", "", f"Run: `{root.name}`", ""]
    outcomes = Counter()
    for episode in sorted(root.glob("episode_*")):
        if not episode.is_dir():
            continue
        result_path = episode / "result.json"
        result = (
            json.loads(result_path.read_text()) if result_path.exists() else {}
        )
        outcome = result.get("terminal_reason", "missing result")
        outcomes[outcome] += 1
        lines.extend([
            f"## {episode.name}", "",
            f"Seed: {result.get('seed', 'unknown')}; outcome: {outcome}; "
            f"steps: {result.get('steps', 'unknown')}", "",
            f"Failure: {result.get('failure_reason') or 'none recorded'}", "",
        ])
        log_path = episode / "flight.log"
        events = []
        malformed = 0
        if log_path.exists():
            log_lines = log_path.read_text(errors="replace").splitlines()
            for number, line in enumerate(log_lines, 1):
                if MARKER not in line:
                    continue
                try:
                    event = json.loads(line.split(MARKER, 1)[1])
                    if not isinstance(event, dict):
                        raise ValueError("diagnostic must be an object")
                    events.append((number, event))
                except (ValueError, TypeError):
                    malformed += 1
        first_fault = None
        for number, event in events:
            status = event.get("result", {})
            state = status.get("state", event.get("state", ""))
            reason = (
                status.get("stop_reason") or status.get("hold_reason")
                or event.get("message") or event.get("failure_reason", "")
            )
            fault = event.get("fault_latched") or status.get("fault_latched")
            if fault and first_fault is None:
                first_fault = (number, event)
            lines.append(
                f"- flight.log:{number}: {event.get('component')} "
                f"{event.get('event')} → {state}: {reason}"
            )
        if first_fault:
            number, event = first_fault
            lines.extend([
                "", f"First recorded explicit fault (flight.log:{number}):",
                "", "```json", json.dumps(event, indent=2), "```", "",
            ])
        else:
            lines.extend([
                "", "No explicit first-fault snapshot recorded; "
                "cause remains unconfirmed.", "",
            ])
        if not events:
            lines.append(
                "No structured diagnostics in this log "
                "(historical or incomplete run).\n"
            )
        if malformed:
            lines.append(f"Malformed diagnostic records: {malformed}\n")
    lines[4:4] = [
        "Outcomes: " + ", ".join(
            f"{key}={value}" for key, value in sorted(outcomes.items())
        ), "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Write a report alongside the existing episode artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    root = args.run_directory.resolve()
    if not root.is_dir():
        parser.error("run_directory must be an existing directory")
    output = root / "startup_diagnosis.md"
    output.write_text(summarize(root), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
