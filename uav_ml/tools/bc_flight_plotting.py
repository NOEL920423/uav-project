"""Plot measured formal BC flight artifacts without touching control logic."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402


TRACE_SCHEMA = "uav_bc_flight_trace/v1"
FIGURE_DPI = 160
EPISODE_FIGURE_SIZE = (8.5, 6.0)
SERIES_FIGURE_SIZE = (9.0, 5.0)
SUMMARY_FIGURE_SIZE = (9.0, 5.0)
LINE_WIDTH = 1.8
MARKER_SIZE = 28
SHOW_OBSTACLE_LABELS = False
SHOW_START_GOAL_TEXT = True
ENABLE_EXTRA_PLOTS = False
STATUS_FIGURE_SIZE = (12.0, 4.8)
STATUS_MARKER_SIZE = 130
STATUS_BAND_ALPHA = 0.085

PLOT_FILENAMES = {
    "trajectory": "trajectory_xy.png",
    "goal_distance": "goal_distance_vs_time.png",
    "action": "bc_action_vs_time.png",
    "clearance": "obstacle_clearance_vs_time.png",
    "outcomes": "outcome_summary.png",
    "status": "flight_status_by_episode.png",
    "final_goal": "final_goal_distance_by_episode.png",
    "minimum_goal": "minimum_goal_distance_by_episode.png",
    "path_length": "path_length_by_episode.png",
    "duration": "episode_duration_by_episode.png",
    "path_goal_scatter": "path_length_vs_final_goal_distance.png",
}

OUTCOME_ORDER = ("success", "collision", "runtime_failure", "out_of_bounds", "timeout")
OUTCOME_LABELS = {
    "runtime_failure": "Control issue",
    "success": "Success",
    "collision": "Collision",
    "timeout": "Timeout",
    "out_of_bounds": "Out of bounds",
}
OUTCOME_COLORS = {
    "runtime_failure": "#8b64bc",
    "success": "#2a9d8f",
    "collision": "#e76f51",
    "timeout": "#e9c46a",
    "out_of_bounds": "#6c757d",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _save(figure, path: Path) -> str:
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return str(path.resolve())


def _time_series(samples: list[dict], key: str) -> tuple[list, list]:
    points = [
        (float(item["time_s"]), float(item[key]))
        for item in samples
        if _finite(item.get("time_s")) and _finite(item.get(key))
    ]
    return [item[0] for item in points], [item[1] for item in points]


def _episode_title(result: dict, subject: str) -> str:
    episode = int(result.get("episode", 0))
    reason = str(result.get("terminal_reason", "unknown"))
    return f"Episode {episode:06d} {subject} — {reason}"


def _trajectory_plot(
    plot_dir: Path, trace: dict, result: dict
) -> str | None:
    samples = trace.get("samples", [])
    points = [
        (float(item["east_m"]), float(item["north_m"]),
         float(item["time_s"]))
        for item in samples
        if all(_finite(item.get(key)) for key in (
            "east_m", "north_m", "time_s"
        ))
    ]
    if not points:
        return None
    east = [item[0] for item in points]
    north = [item[1] for item in points]
    elapsed = [item[2] for item in points]
    figure, axis = plt.subplots(figsize=EPISODE_FIGURE_SIZE)
    for obstacle in trace.get("obstacles", []):
        if not all(_finite(obstacle.get(key)) for key in (
            "east_m", "north_m", "radius_m"
        )):
            continue
        circle = Circle(
            (float(obstacle["east_m"]), float(obstacle["north_m"])),
            float(obstacle["radius_m"]),
            color="#6c757d",
            alpha=0.45,
        )
        axis.add_patch(circle)
        if SHOW_OBSTACLE_LABELS:
            axis.text(
                float(obstacle["east_m"]),
                float(obstacle["north_m"]),
                str(obstacle.get("name", "obstacle")),
                fontsize=7,
                ha="center",
            )
    axis.plot(east, north, color="#264653", linewidth=LINE_WIDTH)
    colored = axis.scatter(
        east,
        north,
        c=elapsed,
        cmap="viridis",
        s=MARKER_SIZE,
        zorder=3,
    )
    figure.colorbar(colored, ax=axis, label="BC control time (s)")
    start = trace.get("start") or {
        "east_m": east[0], "north_m": north[0]
    }
    goal = trace.get("goal")
    if all(_finite(start.get(key)) for key in ("east_m", "north_m")):
        axis.scatter(
            [start["east_m"]], [start["north_m"]],
            marker="o", s=90, color="#2a9d8f", label="Start", zorder=4,
        )
        if SHOW_START_GOAL_TEXT:
            axis.annotate("Start", (start["east_m"], start["north_m"]))
    if goal and all(_finite(goal.get(key)) for key in (
        "east_m", "north_m"
    )):
        axis.scatter(
            [goal["east_m"]], [goal["north_m"]],
            marker="*", s=180, color="#f4a261", label="Goal", zorder=4,
        )
        if SHOW_START_GOAL_TEXT:
            axis.annotate("Goal", (goal["east_m"], goal["north_m"]))
    axis.scatter(
        [east[-1]], [north[-1]], marker="x", s=110,
        linewidths=2.2, color="black", label="Final BC position", zorder=5,
    )
    if result.get("terminal_reason") == "collision":
        axis.scatter(
            [east[-1]], [north[-1]], marker="X", s=150,
            color="#d62828", label="Collision", zorder=6,
        )
    axis.set(
        xlabel="East (m)",
        ylabel="North (m)",
        title=_episode_title(result, "TOP RGB trajectory"),
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    return _save(figure, plot_dir / f"trajectory_xy_{int(result['episode'])}.png")


def _single_series_plot(
    plot_dir: Path,
    samples: list[dict],
    result: dict,
    key: str,
    ylabel: str,
    subject: str,
    filename_key: str,
) -> str | None:
    elapsed, values = _time_series(samples, key)
    if not values:
        return None
    figure, axis = plt.subplots(figsize=SERIES_FIGURE_SIZE)
    axis.plot(elapsed, values, linewidth=LINE_WIDTH)
    axis.set(
        xlabel="BC control time (s)",
        ylabel=ylabel,
        title=_episode_title(result, subject),
    )
    axis.grid(alpha=0.3)
    return _save(figure, plot_dir / PLOT_FILENAMES[filename_key])


def _action_plot(
    plot_dir: Path, samples: list[dict], result: dict
) -> str | None:
    series = {
        "Forward": _time_series(samples, "action_forward"),
        "Right": _time_series(samples, "action_right"),
        "Yaw rate": _time_series(samples, "action_yaw_rate"),
    }
    if not any(values for _, values in series.values()):
        return None
    figure, axis = plt.subplots(figsize=SERIES_FIGURE_SIZE)
    for label, (elapsed, values) in series.items():
        if values:
            axis.plot(
                elapsed, values, linewidth=LINE_WIDTH, label=label
            )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set(
        xlabel="BC control time (s)",
        ylabel="Normalized action",
        title=_episode_title(result, "BC actions"),
        ylim=(-1.05, 1.05),
    )
    axis.grid(alpha=0.3)
    axis.legend()
    return _save(figure, plot_dir / PLOT_FILENAMES["action"])


def _clearance_plot(
    plot_dir: Path, trace: dict, result: dict
) -> str | None:
    samples = trace.get("samples", [])
    elapsed, values = _time_series(samples, "obstacle_clearance_m")
    if not values:
        return None
    threshold = trace.get("collision_clearance_threshold_m")
    figure, axis = plt.subplots(figsize=SERIES_FIGURE_SIZE)
    axis.plot(elapsed, values, linewidth=LINE_WIDTH, label="Clearance")
    if _finite(threshold):
        axis.axhline(
            float(threshold),
            color="#d62828",
            linestyle="--",
            linewidth=1.4,
            label="Collision threshold",
        )
    axis.set(
        xlabel="BC control time (s)",
        ylabel="Obstacle surface clearance (m)",
        title=_episode_title(result, "obstacle clearance"),
    )
    axis.grid(alpha=0.3)
    axis.legend()
    return _save(figure, plot_dir / PLOT_FILENAMES["clearance"])


def create_episode_plots(episode_dir: Path, result: dict) -> list[str]:
    """Create plots only when a real measured trace is available."""
    trace_path = episode_dir / str(
        result.get("trace_file", "trajectory_trace.json")
    )
    if not trace_path.is_file():
        return []
    trace = _read_json(trace_path)
    if trace.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"unsupported BC trace schema: {trace_path}")
    plot_dir = episode_dir.parent / "trajectory_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not ENABLE_EXTRA_PLOTS:
        path = _trajectory_plot(plot_dir, trace, result)
        return [path] if path is not None else []
    samples = trace.get("samples", [])
    trajectory_path = _trajectory_plot(plot_dir, trace, result)
    plot_dir = episode_dir.parent / "extra_plots" / episode_dir.name
    plot_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        trajectory_path,
        _single_series_plot(
            plot_dir,
            samples,
            result,
            "goal_distance_m",
            "Goal distance (m)",
            "goal distance",
            "goal_distance",
        ),
        _action_plot(plot_dir, samples, result),
        _clearance_plot(plot_dir, trace, result),
    ]
    return [path for path in candidates if path is not None]


def _episode_metric_plot(
    plot_dir: Path,
    records: list[dict],
    key: str,
    ylabel: str,
    title: str,
    filename_key: str,
) -> str | None:
    points = [
        (
            int(item["episode"]),
            float(item[key]),
            str(item.get("terminal_reason", "")),
        )
        for item in records if _finite(item.get(key))
    ]
    if not points:
        return None
    indexes = [item[0] for item in points]
    values = [item[1] for item in points]
    colors = [
        OUTCOME_COLORS.get(item[2], "#457b9d") for item in points
    ]
    figure, axis = plt.subplots(figsize=SUMMARY_FIGURE_SIZE)
    axis.bar(indexes, values, color=colors)
    axis.set(xlabel="Episode", ylabel=ylabel, title=title)
    axis.grid(axis="y", alpha=0.3)
    return _save(figure, plot_dir / PLOT_FILENAMES[filename_key])


def _flight_status_plot(plot_dir: Path, records: list[dict]) -> str:
    """Show one categorical outcome per flight, ordered by episode number."""
    ordered = sorted(records, key=lambda item: int(item["episode"]))
    reasons = [str(item.get("terminal_reason", "unknown")) for item in ordered]
    unknown = set(reasons) - set(OUTCOME_ORDER)
    if unknown:
        raise ValueError(f"unsupported flight outcomes: {sorted(unknown)}")
    figure, axis = plt.subplots(figsize=STATUS_FIGURE_SIZE)
    for row, reason in enumerate(OUTCOME_ORDER):
        color = OUTCOME_COLORS[reason]
        axis.axhspan(row - 0.5, row + 0.5, color=color, alpha=STATUS_BAND_ALPHA)
        indexes = [index for index, value in enumerate(reasons, 1) if value == reason]
        axis.scatter(
            indexes, [row] * len(indexes), s=STATUS_MARKER_SIZE,
            color=color, edgecolor="white", linewidth=1.3, zorder=3,
        )
    axis.set_xticks(range(1, len(ordered) + 1))
    axis.set_yticks(range(len(OUTCOME_ORDER)), [OUTCOME_LABELS[r] for r in OUTCOME_ORDER])
    axis.set(
        xlim=(0.5, len(ordered) + 0.5), ylim=(len(OUTCOME_ORDER) - 0.5, -0.5),
        xlabel="Flight order", ylabel="Flight status", title="Flight outcome by episode",
    )
    axis.grid(axis="x", alpha=0.3, linestyle=":")
    return _save(figure, plot_dir / PLOT_FILENAMES["status"])


def create_summary_plots(run_dir: Path, records: list[dict]) -> list[str]:
    """Create aggregate plots from measured per-episode result files."""
    if not records:
        return []
    plot_dir = run_dir / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    counts = [
        sum(
            str(item.get("terminal_reason")) == reason
            for item in records
        )
        for reason in OUTCOME_ORDER
    ]
    figure, axis = plt.subplots(figsize=SUMMARY_FIGURE_SIZE)
    bars = axis.bar(
        [OUTCOME_LABELS[item] for item in OUTCOME_ORDER],
        counts,
        color=[OUTCOME_COLORS[item] for item in OUTCOME_ORDER],
    )
    for bar, value in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value} ({100.0 * value / len(records):.1f}%)",
            ha="center",
            va="bottom",
        )
    axis.set(ylabel="Episodes", title="Formal BC closed-loop outcomes")
    axis.set_ylim(0, max(max(counts), 1) * 1.25)
    axis.grid(axis="y", alpha=0.3)
    paths = [_save(
        figure, plot_dir / PLOT_FILENAMES["outcomes"]
    )]
    paths.append(_flight_status_plot(plot_dir, records))
    if not ENABLE_EXTRA_PLOTS:
        return paths
    metrics = (
        ("final_goal_distance_m", "Final goal distance (m)",
         "Final goal distance by episode", "final_goal"),
        ("minimum_goal_distance_m", "Minimum goal distance (m)",
         "Minimum goal distance by episode", "minimum_goal"),
        ("path_length_m", "Path length (m)",
         "Path length by episode", "path_length"),
        ("episode_duration_s", "Episode duration (s)",
         "Episode duration by episode", "duration"),
    )
    for key, ylabel, title, filename_key in metrics:
        path = _episode_metric_plot(
            plot_dir, records, key, ylabel, title, filename_key
        )
        if path is not None:
            paths.append(path)
    scatter_records = [
        item for item in records
        if _finite(item.get("path_length_m"))
        and _finite(item.get("final_goal_distance_m"))
    ]
    if scatter_records:
        figure, axis = plt.subplots(figsize=SUMMARY_FIGURE_SIZE)
        for reason in OUTCOME_ORDER:
            selected = [
                item for item in scatter_records
                if item.get("terminal_reason") == reason
            ]
            if selected:
                axis.scatter(
                    [item["path_length_m"] for item in selected],
                    [item["final_goal_distance_m"] for item in selected],
                    color=OUTCOME_COLORS[reason],
                    label=OUTCOME_LABELS[reason],
                    s=MARKER_SIZE * 1.5,
                )
        axis.set(
            xlabel="Path length (m)",
            ylabel="Final goal distance (m)",
            title="Path length vs final goal distance",
        )
        axis.grid(alpha=0.3)
        axis.legend()
        paths.append(_save(
            figure, plot_dir / PLOT_FILENAMES["path_goal_scatter"]
        ))
    return paths


def generate_evaluation_plots(
    run_dir: Path, records: list[dict] | None = None
) -> dict:
    """Generate all plots available from one formal evaluation run."""
    root = run_dir.expanduser().resolve()
    if records is None:
        records = [
            _read_json(path)
            for path in sorted(root.glob("episode_*/result.json"))
        ]
    episode_plots = {}
    for result in records:
        episode = int(result["episode"])
        episode_dir = root / f"episode_{episode:06d}"
        paths = create_episode_plots(episode_dir, result)
        if paths:
            episode_plots[f"episode_{episode:06d}"] = paths
    return {
        "episodes": episode_plots,
        "summary": create_summary_plots(root, records),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./uav bc-eval-plot",
        description="Regenerate plots from a formal BC evaluation run.",
    )
    parser.add_argument("run_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plots = generate_evaluation_plots(args.run_dir)
    summary_path = args.run_dir.expanduser().resolve() / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        summary["plots"] = plots
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(plots, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
