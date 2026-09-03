"""Targeted guardrails for the isolated persistent-runtime diagnostic."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
from unittest import mock

import pytest

from uav_ml.tools.expert_runtime_smoke import (
    _live_process_group_members,
    _parser,
    _process_start_ticks,
    _stop_process,
)
from uav_ml.tools.persistent_runtime import (
    FatalRuntimeError,
    PersistentRuntimeManager,
    RecoverableAttemptError,
)


ROOT = Path(__file__).parents[2]
BOOTSTRAP = ROOT / "isaac/runtime/bootstrap.py"
CONTROL = ROOT / "isaac/runtime/persistent_smoke_control.py"
RUNTIME_MANAGER = ROOT / "uav_ml/tools/persistent_runtime.py"
COLLECTOR = ROOT / "uav_ml/tools/expert_collect.py"
GENERATION_PROBE = (
    ROOT / "ros2_ws/src/uav_px4_control/uav_px4_control/"
    "px4_generation_probe.py"
)
FLIGHT_SUPERVISOR = (
    ROOT / "ros2_ws/src/uav_px4_control/uav_px4_control/"
    "px4_sitl_flight_supervisor_node.py"
)


def test_smoke_cli_defaults_to_two_isolated_episodes():
    arguments = _parser().parse_args([])
    assert arguments.episodes == 2
    assert arguments.runtime_timeout > 0
    assert arguments.episode_timeout > arguments.runtime_timeout


def test_production_collector_uses_shared_runtime_not_smoke_command():
    source = COLLECTOR.read_text(encoding="utf-8")
    assert "expert_runtime_smoke" not in source
    assert "UAV_PERSISTENT_RUNTIME_SMOKE" not in source
    assert "PersistentRuntimeManager" in source
    assert "self.backend.cleanup_episode(" in source
    assert "finally:\n            self.backend.cleanup()" in source


def test_bootstrap_changes_px4_and_lockstep_only_under_smoke_flag():
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'os.environ.get("UAV_PERSISTENT_RUNTIME", "0") == "1"' in source
    assert (
        'os.environ.get("UAV_PERSISTENT_RUNTIME_SMOKE", "0") == "1"'
        in source
    )
    assert '"px4_autolaunch": not persistent_runtime' in source
    assert '"enable_lockstep": not persistent_runtime' in source
    assert "PERSISTENT_SMOKE_CONTROL_SCRIPT" in source


def test_smoke_control_reuses_resources_and_performs_full_reset():
    source = CONTROL.read_text(encoding="utf-8")
    for token in (
        "await self._world.stop_async()",
        "await self._world.reset_async()",
        "self._vehicle.set_world_pose(",
        "self._vehicle.set_world_velocity(ZERO6)",
        "self._vehicle.set_linear_velocity(ZERO3)",
        "self._vehicle.set_angular_velocity(ZERO3)",
        "self._vehicle.set_joint_velocities(joint_zeros)",
        "rotor_data.zero_input_reference()",
        "resource_identity_unchanged",
    ):
        assert token in source
    for forbidden in ("Multirotor(", "load_environment", "_setup_camera("):
        assert forbidden not in source


def test_orchestrator_owns_one_isaac_one_agent_and_external_px4():
    source = RUNTIME_MANAGER.read_text(encoding="utf-8")
    assert source.count("self.isaac = subprocess.Popen(") == 1
    assert source.count("self.xrce = subprocess.Popen(") == 1
    assert source.count("self.px4 = subprocess.Popen(") == 1
    assert '"PX4_SIM_MODEL": PX4_MODEL' in source
    assert "tempfile.mkdtemp(" in source
    assert '"px4_autolaunch"' not in source
    assert "set -eo pipefail" in source
    assert "set -euo pipefail" not in source


def test_shutdown_escalates_sigint_then_sigterm_before_sigkill():
    process = mock.Mock()
    process.pid = 4242
    process.returncode = -signal.SIGTERM
    process.poll.side_effect = [None]
    process.wait.side_effect = [
        subprocess.TimeoutExpired("px4", 0.01),
        -signal.SIGTERM,
    ]
    with mock.patch("os.killpg") as killpg:
        evidence = _stop_process(
            process, interrupt_s=0.01, terminate_s=0.01, kill_s=0.01
        )
    assert [call.args[1] for call in killpg.call_args_list] == [
        signal.SIGINT,
        signal.SIGTERM,
    ]
    assert evidence["method"] == "SIGTERM"
    assert evidence["escalated"] is True


def test_linux_process_generation_token_is_readable():
    assert isinstance(_process_start_ticks(os.getpid()), int)
    assert _process_start_ticks(999_999_999) is None


def test_live_process_group_evidence_includes_current_process():
    assert os.getpid() in _live_process_group_members(os.getpgrp())


def test_fresh_dds_probe_requires_changing_post_subscription_streams():
    source = GENERATION_PROBE.read_text(encoding="utf-8")
    for token in (
        "expected_start_ticks",
        "timestamps[-1] > timestamps[0]",
        "all(right > left",
        "receive_times[0] >= self.started_monotonic",
        "safe_landed_disarmed",
        "reset_state_matches",
        "status_endpoint_gids",
    ):
        assert token in source


def test_two_consecutive_reset_failures_escalate_to_job_fatal(tmp_path):
    manager = PersistentRuntimeManager(ROOT, tmp_path)
    manager.isaac = mock.Mock(pid=1001)
    manager.xrce = mock.Mock(pid=1002)
    manager.assert_job_alive = mock.Mock()
    manager.lifecycle = mock.Mock(
        side_effect=RecoverableAttemptError("fixture reset failure")
    )

    with pytest.raises(RecoverableAttemptError):
        manager.prepare_attempt(1, tmp_path / "attempt_1")
    with pytest.raises(FatalRuntimeError):
        manager.prepare_attempt(2, tmp_path / "attempt_2")


def test_stream_transition_logging_uses_fixed_severity_call_sites():
    source = FLIGHT_SUPERVISOR.read_text(encoding="utf-8")
    callback = source.split("def _stream_callback", 1)[1].split(
        "def _vehicle_status_callback", 1
    )[0]
    assert "log = (" not in callback
    assert "self.get_logger().warning(transition)" in callback
    assert "self.get_logger().info(transition)" in callback
