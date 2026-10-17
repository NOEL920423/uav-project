"""Load shared navigation task settings for ROS and offline environments."""

import math
from pathlib import Path

import yaml


def load_success_distance(config_path: str | Path | None = None) -> float:
    """Read a finite, positive goal distance from the shared task YAML."""
    if config_path is None:
        package_root = Path(__file__).resolve().parents[1]
        if (package_root / "package.xml").is_file():
            config_path = package_root / "config" / "task.yaml"
        else:
            from ament_index_python.packages import (
                get_package_share_directory,
            )

            config_path = (
                Path(get_package_share_directory("uav_navigation"))
                / "config" / "task.yaml"
            )
    config_path = Path(config_path)
    with config_path.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError(
            f"{config_path}: task settings must be a YAML mapping"
        )
    value = settings.get("success_distance_m")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(
            f"{config_path}: success_distance_m "
            "must be a finite positive number"
        )
    return float(value)


SUCCESS_DISTANCE_M = load_success_distance()
