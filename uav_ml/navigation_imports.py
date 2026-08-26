"""Load the repository's ROS-independent navigation package in-place."""

import sys
from pathlib import Path


def add_navigation_source_path() -> Path:
    """Expose pure navigation modules without sourcing a ROS environment."""
    package_root = (
        Path(__file__).resolve().parents[1]
        / "ros2_ws"
        / "src"
        / "uav_navigation"
    )
    if not package_root.is_dir():
        raise FileNotFoundError(f"navigation source is missing: {package_root}")
    path = str(package_root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return package_root


def add_data_recorder_source_path() -> Path:
    """Expose the shared dataset geometry helpers from a source checkout."""
    package_root = (
        Path(__file__).resolve().parents[1]
        / "ros2_ws"
        / "src"
        / "uav_data_recorder"
    )
    if not package_root.is_dir():
        raise FileNotFoundError(
            f"data recorder source is missing: {package_root}"
        )
    path = str(package_root)
    if path not in sys.path:
        sys.path.insert(0, path)
    return package_root
