"""Plot measured formal BC flight artifacts without touching control logic."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from isaac.runtime.environment import build_environment_parameters  # noqa: E402


TRACE_SCHEMA = "uav_bc_flight_trace/v1"
FIGURE_DPI = 160
EPISODE_FIGURE_SIZE = (8.5, 6.0)
SERIES_FIGURE_SIZE = (9.0, 5.0)
SUMMARY_FIGURE_SIZE = (9.0, 5.0)
LINE_WIDTH = 1.8
MARKER_SIZE = 28
TRAJECTORY_TITLE_FONT_SIZE = 12
TRAJECTORY_LABEL_FONT_SIZE = 11
TRAJECTORY_TICK_FONT_SIZE = 9
TRAJECTORY_LEGEND_FONT_SIZE = 8
SUMMARY_TITLE_FONT_SIZE = 14
SUMMARY_LABEL_FONT_SIZE = 12
SUMMARY_TICK_FONT_SIZE = 11
SUMMARY_COUNT_FONT_SIZE = 11
TRAJECTORIES_PER_GROUP = 4
SHOW_OBSTACLE_LABELS = False
SHOW_START_GOAL_TEXT = True
ENABLE_EXTRA_PLOTS = False
STATUS_FIGURE_SIZE = (12.0, 4.8)
STATUS_MARKER_SIZE = 130
STATUS_BAND_ALPHA = 0.085

PLOT_FILENAMES = {
    "trajectory": "trajectory.png",
    "goal_distance": "goal_distance_vs_time.png",
    "action": "bc_action_vs_time.png",
    "clearance": "obstacle_clearance_vs_time.png",
    "outcomes": "summary.png",
    "status": "flight_status_by_episode.png",
    "final_goal": "final_goal_distance_by_episode.png",
    "minimum_goal": "minimum_goal_distance_by_episode.png",
    "path_length": "path_length_by_episode.png",
    "duration": "episode_duration_by_episode.png",
    "path_goal_scatter": "path_length_vs_final_goal_distance.png",
}

OUTCOME_ORDER = ("success", "timeout", "out_of_bounds", "collision", "runtime_failure")
SUMMARY_OUTCOME_ORDER = ("success", "timeout", "collision")
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
TRAJECTORY_COLORS = {
    "success": ("#238b45", "#006d2c", "#41ab5d", "#74c476"),
    "timeout": ("#2171b5", "#084594", "#4292c6", "#6baed6"),
    "collision": ("#cb181d", "#99000d", "#ef3b2c", "#fb6a4a"),
    "runtime_failure": ("#88419d", "#6e016b", "#8c6bb1", "#9e9ac8"),
    "out_of_bounds": ("#525252", "#252525", "#737373", "#969696"),
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


def _trajectory_points(trace: dict) -> list[tuple[float, float]]:
    samples = trace.get("samples", [])
    points = []
    seen_inference_steps = set()
    for item in samples:
        if not all(_finite(item.get(key)) for key in ("east_m", "north_m")):
            continue
        inference_step = item.get("inference_step")
        if _finite(inference_step):
            if inference_step in seen_inference_steps:
                continue
            seen_inference_steps.add(inference_step)
        points.append((float(item["east_m"]), float(item["north_m"])))
    return points


def _draw_trajectory_scene(
    axis, trace: dict, points: list[tuple[float, float]],
    *, seen: set | None = None,
) -> None:
    """Draw configured walls and measured geometry once for shared scenes."""
    if seen is None:
        seen = set()
    # Traces omit static walls; reuse the canonical scene configuration.
    walls = build_environment_parameters([])["arena"]["walls"]["geometry"]
    for wall in walls:
        east, north = wall["position"][:2]
        width, height = wall["size"][:2]
        wall_key = ("wall", east, north, width, height)
        if wall_key in seen:
            continue
        seen.add(wall_key)
        axis.add_patch(Rectangle(
            (east - width / 2.0, north - height / 2.0), width, height,
            facecolor="#737373", edgecolor="#404040", linewidth=1.0,
            label="Arena walls" if "wall_label" not in seen else None,
            zorder=1,
        ))
        seen.add("wall_label")
    for obstacle in trace.get("obstacles", []):
        if not all(_finite(obstacle.get(key)) for key in (
            "east_m", "north_m", "radius_m"
        )):
            continue
        geometry = tuple(float(obstacle[key]) for key in (
            "east_m", "north_m", "radius_m"
        ))
        obstacle_key = ("obstacle", *geometry)
        if obstacle_key in seen:
            continue
        seen.add(obstacle_key)
        circle = Circle(
            (float(obstacle["east_m"]), float(obstacle["north_m"])),
            float(obstacle["radius_m"]),
            color="#6c757d",
            alpha=0.45,
        )
        axis.add_patch(circle)
        uav_radius = trace.get("uav_radius_m")
        clearance = trace.get("collision_clearance_threshold_m")
        if _finite(uav_radius) and _finite(clearance):
            protected = Circle(
                (float(obstacle["east_m"]), float(obstacle["north_m"])),
                float(obstacle["radius_m"])
                + float(uav_radius) + float(clearance),
                fill=False,
                edgecolor="#d62828",
                linestyle="--",
                linewidth=1.1,
                alpha=0.8,
                label=(
                    "UAV collision envelope"
                    if "collision_envelope_label" not in seen else None
                ),
            )
            axis.add_patch(protected)
            seen.add("collision_envelope_label")
        if SHOW_OBSTACLE_LABELS:
            axis.text(
                float(obstacle["east_m"]),
                float(obstacle["north_m"]),
                str(obstacle.get("name", "obstacle")),
                fontsize=7,
                ha="center",
            )
    start = trace.get("start") or {
        "east_m": points[0][0], "north_m": points[0][1]
    }
    goal = trace.get("goal")
    disk_radius = float(trace.get("start_goal_disk_radius_m", 0.5))
    start_key = ("start", start.get("east_m"), start.get("north_m"), disk_radius)
    if (
        all(_finite(start.get(key)) for key in ("east_m", "north_m"))
        and start_key not in seen
    ):
        seen.add(start_key)
        show_label = "start_label" not in seen
        axis.add_patch(
            Circle(
                (float(start["east_m"]), float(start["north_m"])),
                disk_radius,
                facecolor="#2a9d8f", edgecolor="#16756f", alpha=0.24,
                linewidth=1.8,
                label=f"Start disk (r={disk_radius:g} m)" if show_label else None,
                zorder=2,
            )
        )
        seen.add("start_label")
        if SHOW_START_GOAL_TEXT and show_label:
            axis.annotate(
                "Start", (start["east_m"], start["north_m"]),
                fontsize=TRAJECTORY_TICK_FONT_SIZE,
            )
    goal_key = (
        "goal", (goal or {}).get("east_m"),
        (goal or {}).get("north_m"), disk_radius,
    )
    if (
        goal and all(_finite(goal.get(key)) for key in ("east_m", "north_m"))
        and goal_key not in seen
    ):
        seen.add(goal_key)
        show_label = "goal_label" not in seen
        axis.add_patch(
            Circle(
                (float(goal["east_m"]), float(goal["north_m"])),
                disk_radius,
                facecolor="#f4a261", edgecolor="#d97706", alpha=0.28,
                linewidth=1.8,
                label=f"Goal disk (r={disk_radius:g} m)" if show_label else None,
                zorder=2,
            )
        )
        seen.add("goal_label")
        if SHOW_START_GOAL_TEXT and show_label:
            axis.annotate(
                "Goal", (goal["east_m"], goal["north_m"]),
                fontsize=TRAJECTORY_TICK_FONT_SIZE,
            )


def _trajectory_plot(
    plot_dir: Path, trace: dict, result: dict
) -> str | None:
    points = _trajectory_points(trace)
    if not points:
        return None
    east = [item[0] for item in points]
    north = [item[1] for item in points]
    figure, axis = plt.subplots(figsize=EPISODE_FIGURE_SIZE)
    _draw_trajectory_scene(axis, trace, points)
    axis.plot(east, north, color="#264653", linewidth=LINE_WIDTH)
    axis.scatter(
        east, north, color="#264653", s=MARKER_SIZE, zorder=3,
        label="UAV center (one dot per decision)",
    )
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
        xlabel="x (m)",
        ylabel="y (m)",
        title=_episode_title(result, "TOP RGB trajectory"),
    )
    axis.xaxis.label.set_size(TRAJECTORY_LABEL_FONT_SIZE)
    axis.yaxis.label.set_size(TRAJECTORY_LABEL_FONT_SIZE)
    axis.title.set_size(TRAJECTORY_TITLE_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=TRAJECTORY_TICK_FONT_SIZE)
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=TRAJECTORY_LEGEND_FONT_SIZE)
    return _save(figure, plot_dir / f"trajectory_{int(result['episode'])}.png")


def create_grouped_trajectory_plots(run_dir: Path, records: list[dict]) -> list[str]:
    """Overlay measured flights in fixed groups of four episode numbers."""
    groups = {}
    for result in sorted(records, key=lambda item: int(item["episode"])):
        episode = int(result["episode"])
        trace_path = run_dir / f"episode_{episode:06d}" / str(
            result.get("trace_file", "trajectory_trace.json")
        )
        if not trace_path.is_file():
            continue
        trace = _read_json(trace_path)
        if trace.get("schema") != TRACE_SCHEMA:
            raise ValueError(f"unsupported BC trace schema: {trace_path}")
        points = _trajectory_points(trace)
        if points:
            first = (
                (episode - 1) // TRAJECTORIES_PER_GROUP
            ) * TRAJECTORIES_PER_GROUP + 1
            groups.setdefault(first, []).append((result, trace, points))
    if not groups:
        return []
    plot_dir = run_dir / "trajectory_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for first, flights in groups.items():
        last = first + TRAJECTORIES_PER_GROUP - 1
        figure, axis = plt.subplots(figsize=EPISODE_FIGURE_SIZE)
        seen_scene = set()
        outcome_counts = {}
        for result, trace, points in flights:
            _draw_trajectory_scene(axis, trace, points, seen=seen_scene)
            reason = str(result.get("terminal_reason", "unknown"))
            color_index = outcome_counts.get(reason, 0)
            outcome_counts[reason] = color_index + 1
            colors = TRAJECTORY_COLORS.get(reason, TRAJECTORY_COLORS["out_of_bounds"])
            color = colors[color_index]
            east = [point[0] for point in points]
            north = [point[1] for point in points]
            axis.plot(
                east, north, color=color, linewidth=LINE_WIDTH,
                linestyle="--" if reason == "runtime_failure" else "-",
                label=(f"Episode {int(result['episode']):03d}: "
                       f"{OUTCOME_LABELS.get(reason, reason)}"),
                zorder=3,
            )
            axis.scatter(
                east, north, color=color, s=MARKER_SIZE, zorder=4,
            )
            axis.scatter(
                [east[-1]], [north[-1]], marker="x", s=60,
                linewidths=1.8, color=color, zorder=5,
            )
        axis.set(
            xlabel="x (m)", ylabel="y (m)",
            title=f"Episodes {first:03d}–{last:03d} TOP RGB trajectories",
        )
        axis.xaxis.label.set_size(TRAJECTORY_LABEL_FONT_SIZE)
        axis.yaxis.label.set_size(TRAJECTORY_LABEL_FONT_SIZE)
        axis.title.set_size(TRAJECTORY_TITLE_FONT_SIZE)
        axis.tick_params(axis="both", labelsize=TRAJECTORY_TICK_FONT_SIZE)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", fontsize=TRAJECTORY_LEGEND_FONT_SIZE)
        paths.append(_save(
            figure, plot_dir / f"trajectory_{first:03d}-{last:03d}.png"
        ))
    return paths


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
        for reason in SUMMARY_OUTCOME_ORDER
    ]
    figure, axis = plt.subplots(figsize=SUMMARY_FIGURE_SIZE)
    bars = axis.bar(
        [OUTCOME_LABELS[item] for item in SUMMARY_OUTCOME_ORDER],
        counts,
        color=[OUTCOME_COLORS[item] for item in SUMMARY_OUTCOME_ORDER],
    )
    for bar, value in zip(bars, counts):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value} ({100.0 * value / len(records):.1f}%)",
            ha="center",
            va="bottom",
            fontsize=SUMMARY_COUNT_FONT_SIZE,
        )
    axis.set(ylabel="Episodes", title="Closed-loop outcomes")
    axis.title.set_size(SUMMARY_TITLE_FONT_SIZE)
    axis.yaxis.label.set_size(SUMMARY_LABEL_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=SUMMARY_TICK_FONT_SIZE)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
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
        "trajectory_groups": create_grouped_trajectory_plots(root, records),
        "summary": create_summary_plots(root, records),
    }


def _export_training_model_loss(
    loss_dir: Path, name: str, checkpoint: Path, architecture: str,
) -> dict:
    """Export the exact selected model's training evidence, never a latest run."""
    import torch

    checkpoint = checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("model_class") != architecture:
        raise ValueError(f"training artifact architecture mismatch: {checkpoint}")
    info = {
        "architecture": architecture,
        "checkpoint_path": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "best_train_epoch": None,
        "best_train_loss": None,
        "best_validation_epoch": None,
        "best_validation_loss": None,
        "training_duration_s": None,
        "loss_curve": None,
    }
    if architecture == "ResNet18Encoder":
        info.update({
            "status": "pretrained_frozen",
            "parameters": {
                "weights": payload.get("metadata", {}).get("weights"),
                "frozen": True,
            },
            "reason": "Frozen ImageNet pretrained encoder; no local encoder training loss or duration.",
        })
        return info

    training_dir = checkpoint.parent
    summary_path = training_dir / "summary.json"
    config_path = training_dir / "training_config.json"
    summary = _read_json(summary_path) if summary_path.is_file() else {}
    config = _read_json(config_path) if config_path.is_file() else {}
    values = {
        **payload.get("metadata", {}), **payload.get("training_config", {}),
        **summary, **config,
    }
    parameters = {
        "maximum_epochs": values.get("maximum_epochs_requested", values.get("epochs")),
        "learning_rate": values.get("learning_rate", values.get("optimizer", {}).get("learning_rate")),
        "batch_size": values.get("batch_size"),
        "early_stopping_patience": values.get("early_stopping_patience"),
        "minimum_epochs": values.get("min_epochs"),
        "seed": values.get("seed"),
    }
    history_path = training_dir / ("training_history.csv" if name == "bc" else "history.csv")
    history = []
    if history_path.is_file():
        with history_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                point = {
                    "epoch": int(row["epoch"]),
                    "train_mse": float(row["train_mse"]),
                    "validation_mse": float(row["validation_mse"]),
                }
                if point["epoch"] < 1 or not all(_finite(value) for value in point.values()):
                    raise ValueError(f"invalid training loss history: {history_path}")
                history.append(point)
        history.sort(key=lambda row: row["epoch"])
    info.update({
        "training_summary_path": str(summary_path) if summary_path.is_file() else None,
        "training_config_path": str(config_path) if config_path.is_file() else None,
        "history_path": str(history_path) if history_path.is_file() else None,
        "actual_epochs_trained": values.get("actual_epochs_trained", len(history) or None),
        "early_stopping_triggered": values.get("early_stopping_triggered"),
        "training_duration_s": values.get("training_duration_s"),
        "parameters": parameters,
    })
    if history:
        for split in ("train", "validation"):
            best = min(history, key=lambda row: row[f"{split}_mse"])
            info[f"best_{split}_epoch"] = best["epoch"]
            info[f"best_{split}_loss"] = best[f"{split}_mse"]
    source_curve = training_dir / (
        "plots/loss_curves.png" if name == "bc" else "reconstruction_loss_curves.png"
    )
    target_curve = loss_dir / f"{name}.png"
    if source_curve.is_file():
        shutil.copy2(source_curve, target_curve)
        info["loss_curve"] = f"loss/{name}.png"
    elif history:
        figure, axis = plt.subplots(figsize=SERIES_FIGURE_SIZE)
        for split in ("train", "validation"):
            axis.plot(
                [row["epoch"] for row in history],
                [row[f"{split}_mse"] for row in history], label=split.title(),
            )
        axis.set(
            xlabel="Epoch", ylabel="Normalized action MSE" if name == "bc" else "Reconstruction MSE",
            title="BC training loss" if name == "bc" else "Autoencoder training loss",
        )
        axis.grid(alpha=0.3)
        axis.legend()
        _save(figure, target_curve)
        info["loss_curve"] = f"loss/{name}.png"
    missing = [key for key in (
        "training_duration_s", "best_train_epoch", "best_validation_epoch", "loss_curve",
    ) if info[key] is None]
    missing.extend(f"parameters.{key}" for key, value in parameters.items() if value is None)
    info["status"] = "partial" if missing else "complete"
    if missing:
        info["missing_fields"] = missing
        print(f"[BC Flight] {name} training evidence not recorded: {', '.join(missing)}", flush=True)
    return info


def create_training_loss_artifacts(
    run_dir: Path, checkpoint: Path, encoder_checkpoint: Path, encoder_architecture: str,
) -> dict:
    """Save concise BC and feature-extractor provenance beside evaluation plots."""
    import traceback

    loss_dir = run_dir / "loss"
    loss_dir.mkdir(parents=True, exist_ok=True)
    feature_name = "resnet18" if encoder_architecture == "ResNet18Encoder" else "ae"
    models = {}
    for name, path, architecture in (
        ("bc", checkpoint, "LatentBcPolicy"),
        (feature_name, encoder_checkpoint, encoder_architecture),
    ):
        try:
            models[name] = _export_training_model_loss(loss_dir, name, path, architecture)
        except Exception as error:
            traceback.print_exc()
            models[name] = {
                "architecture": architecture, "checkpoint_path": str(path.expanduser().resolve()),
                "status": "error", "error": str(error),
            }
    summary = {"schema": "uav_bc_training_loss/v1", "models": models}
    (loss_dir / "loss_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    return summary


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
