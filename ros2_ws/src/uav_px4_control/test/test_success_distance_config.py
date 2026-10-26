"""Check shared goal defaults across offline and ROS consumers."""

import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import yaml


ROOT = Path(__file__).resolve().parents[4]
NAVIGATION = ROOT / "ros2_ws/src/uav_navigation"


def _check_consumers(tmp_path, distance, *, ros=False, override=False):
    """Load fresh consumers using an unchanged loader and a changed YAML."""
    package_root = tmp_path / "uav_navigation"
    (package_root / "config").mkdir(parents=True)
    (package_root / "uav_navigation").mkdir()
    loader_path = package_root / "uav_navigation/task_config.py"
    shutil.copyfile(NAVIGATION / "uav_navigation/task_config.py", loader_path)
    shutil.copyfile(NAVIGATION / "package.xml", package_root / "package.xml")
    (package_root / "config/task.yaml").write_text(
        yaml.safe_dump({"success_distance_m": distance}), encoding="utf-8",
    )
    script = textwrap.dedent("""\
        from dataclasses import replace
        from pathlib import Path
        import importlib.util
        import sys

        import uav_navigation

        root, distance, ros, override, loader_path = sys.argv[1:]
        distance = float(distance)
        spec = importlib.util.spec_from_file_location(
            "uav_navigation.task_config", loader_path,
        )
        shared = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = shared
        spec.loader.exec_module(shared)
        assert shared.SUCCESS_DISTANCE_M == distance

        from uav_px4_control.flight.bc_episode_monitor import (
            TerminationConfig, select_terminal_reason,
        )
        from uav_px4_control.flight.px4_flight_models import Px4FlightConfig

        for config in (TerminationConfig(), Px4FlightConfig()):
            assert config.goal_tolerance_m == distance
            explicit = replace(config, goal_tolerance_m=0.2)
            assert explicit.goal_tolerance_m == 0.2
        common = dict(
            minimum_clearance_m=1.0, bc_duration_s=1.0,
            north_m=3.0, east_m=0.0, config=TerminationConfig(),
        )
        reason = select_terminal_reason(goal_distance_m=distance, **common)
        assert reason == "success"
        reason = select_terminal_reason(
            goal_distance_m=distance + 0.01, **common,
        )
        assert reason is None
        common["minimum_clearance_m"] = 0.0
        reason = select_terminal_reason(goal_distance_m=distance, **common)
        assert reason == "collision"

        if ros == "False":
            sys.exit(0)

        import rclpy
        from uav_px4_control.flight.bc_episode_monitor_node import (
            BcEpisodeMonitorNode,
        )
        from uav_px4_control.flight.px4_sitl_flight_supervisor_node import (
            Px4SitlFlightSupervisorNode,
        )
        from uav_px4_control.diagnostics.px4_sitl_flight_monitor import (
            Px4SitlFlightMonitor,
        )

        config_dir = Path(root) / "ros2_ws/src/uav_px4_control/config"
        arguments = [
            "--ros-args", "--params-file", str(config_dir / "bc_flight.yaml"),
            "--params-file", str(config_dir / "px4_sitl_flight.yaml"),
        ]
        expected = distance
        if override == "True":
            expected = 0.2
            arguments += [
                "-p", "goal_tolerance_m:=0.2",
                "-p", "isaac_goal_tolerance_m:=0.2",
            ]
        rclpy.init(args=arguments)
        try:
            for node_type, parameter in (
                (BcEpisodeMonitorNode, "goal_tolerance_m"),
                (Px4SitlFlightSupervisorNode, "goal_tolerance_m"),
                (Px4SitlFlightMonitor, "isaac_goal_tolerance_m"),
            ):
                node = node_type()
                try:
                    assert node.get_parameter(parameter).value == expected
                    if isinstance(node, Px4SitlFlightMonitor):
                        assert node.isaac_goal_tolerance_m == expected
                    else:
                        assert node.config.goal_tolerance_m == expected
                finally:
                    node.destroy_node()
        finally:
            rclpy.shutdown()
    """)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([
        str(ROOT), str(ROOT / "ros2_ws/src/uav_navigation"),
        str(ROOT / "ros2_ws/src/uav_px4_control"),
        environment.get("PYTHONPATH", ""),
    ])
    environment["ROS_DOMAIN_ID"] = "227"
    environment["ROS_LOG_DIR"] = str(tmp_path / "ros_logs")
    environment.pop("UAV_TIMING_DIR", None)
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT), str(distance),
         str(ros), str(override), str(loader_path)],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("distance", [0.4, 0.8])
def test_one_setting_reaches_offline_consumers(tmp_path, distance):
    """Changing only the YAML reaches every pure config on fresh import."""
    _check_consumers(tmp_path, distance)


@pytest.mark.parametrize("distance", [0.4, 0.8])
@pytest.mark.parametrize("override", [False, True])
def test_ros_yaml_inherits_shared_default(tmp_path, distance, override):
    """Real ROS nodes inherit the setting unless explicitly overridden."""
    pytest.importorskip("rclpy")
    _check_consumers(tmp_path, distance, ros=True, override=override)


@pytest.mark.parametrize("value", [
    None, True, False, 0.0, -0.5, float("nan"), float("inf"), "0.5",
])
def test_loader_rejects_invalid_distance(tmp_path, value):
    """Invalid task distances fail instead of using a Python fallback."""
    from uav_navigation.task_config import load_success_distance

    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump({"success_distance_m": value}))
    with pytest.raises(ValueError, match="success_distance_m"):
        load_success_distance(path)


@pytest.mark.parametrize("settings", [None, [], {}])
def test_loader_requires_task_mapping_and_distance(tmp_path, settings):
    """Empty YAML, non-mappings, and missing keys fail explicitly."""
    from uav_navigation.task_config import load_success_distance

    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError):
        load_success_distance(path)


def test_loader_preserves_file_and_yaml_errors(tmp_path):
    """Missing files and malformed YAML retain their original errors."""
    from uav_navigation.task_config import load_success_distance

    path = tmp_path / "task.yaml"
    with pytest.raises(FileNotFoundError):
        load_success_distance(path)
    path.write_text("success_distance_m: [")
    with pytest.raises(yaml.YAMLError):
        load_success_distance(path)


def test_installed_loader_uses_package_share(tmp_path):
    """A loader outside the source tree reads the installed YAML."""
    pytest.importorskip("ament_index_python.packages")
    path = tmp_path / "site-packages/uav_navigation/task_config.py"
    path.parent.mkdir(parents=True)
    shutil.copyfile(NAVIGATION / "uav_navigation/task_config.py", path)
    spec = importlib.util.spec_from_file_location(
        "installed_task_config", path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = yaml.safe_load(
        (NAVIGATION / "config/task.yaml").read_text(encoding="utf-8")
    )
    assert module.SUCCESS_DISTANCE_M == settings["success_distance_m"]
