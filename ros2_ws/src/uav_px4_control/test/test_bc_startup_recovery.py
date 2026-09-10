"""Bounded startup recovery and asynchronous reset evidence regressions."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from uav_px4_control.flight.bc_flight_models import (
    BcFlightController, BcFlightEvidence, BcFlightState, StartupStale,
)


def ready():
    """Provide positively confirmed ground readiness."""
    return BcFlightEvidence(
        runtime_ready=True, observations_ready=True, telemetry_fresh=True,
        recovery_vehicle_state_fresh=True, lifecycle_selected=True,
        source_valid=True, output_ready=True,
    )


def prestream(controller, now=0.0):
    """Reach prestream through the normal lifecycle."""
    controller.step(now, ready())
    controller.step(now + 0.1, ready())
    controller.step(now + 0.2, replace(ready(), output_safe=True))
    assert controller.state == BcFlightState.ENABLING_STREAM


def stale(source="streamer.candidate_age", age=0.272, threshold=0.25):
    """Represent the original stale measurement, not a later latch age."""
    return replace(ready(), startup_stale=StartupStale(
        source, age, threshold, "original stale reason",
    ))


def recover(controller, now, fault):
    """Reset stream before gate and await readiness before restarting."""
    decision = controller.step(now, fault)
    assert decision.actions == ("DISABLE_STREAM",)
    assert controller.step(now + 0.1, fault).actions == ("DISABLE_STREAM",)
    decision = controller.step(now + 0.2, replace(fault, stream_reset_complete=True))
    assert decision.actions == ("DISABLE_OUTPUT",)
    assert controller.step(now + 0.3, fault).actions == ("DISABLE_OUTPUT",)
    decision = controller.step(now + 0.4, replace(fault, output_reset_complete=True))
    assert decision.state == BcFlightState.RECOVERING_READINESS
    assert controller.step(now + 0.5, replace(ready(), telemetry_fresh=False)).actions == (
        "SELECT_LIFECYCLE",
    )
    assert controller.step(now + 0.6, ready()).actions == ("ENABLE_OUTPUT",)
    assert controller.step(now + 0.7, replace(ready(), output_safe=True)).actions == (
        "ENABLE_STREAM",
    )


@pytest.mark.parametrize("fault", [stale(), stale("streamer.telemetry_age", 0.759, 0.75)])
def test_two_recoveries_then_preserve_original_failure(fault):
    """Replay both run failures and retain diagnostics after retry exhaustion."""
    controller = BcFlightController()
    prestream(controller)
    deadline = controller.startup_deadline_s
    recover(controller, 1.0, fault)
    second = stale("gate.selected_command_age", 0.337, 0.25)
    recover(controller, 3.0, second)
    decision = controller.step(5.0, second)
    assert decision.state == BcFlightState.HOLDING
    assert "limit exceeded (2)" in decision.failure_reason
    assert fault.startup_stale.source in decision.failure_reason
    assert str(fault.startup_stale.age_s) in decision.failure_reason
    assert '"threshold_s"' in decision.failure_reason
    assert '"recovery_history"' in decision.failure_reason
    assert len(controller.recovery_history) == 2
    assert controller.first_stale == fault.startup_stale
    assert controller.startup_deadline_s == deadline
    cleanup = controller.step(5.1, ready())
    assert "DISABLE_STREAM" not in cleanup.actions
    assert "DISABLE_OUTPUT" not in cleanup.actions


@pytest.mark.parametrize("stage", [
    BcFlightState.RECOVERING_STREAM, BcFlightState.RECOVERING_OUTPUT,
    BcFlightState.RECOVERING_READINESS,
])
def test_deadline_survives_reset_and_readiness_wait(stage):
    """No reset or readiness wait grants a new startup budget."""
    controller = BcFlightController()
    prestream(controller)
    controller.step(1.0, stale())
    controller.state = stage
    decision = controller.step(controller.startup_deadline_s + 0.01, stale())
    assert decision.state == BcFlightState.HOLDING
    assert "overall startup deadline" in decision.failure_reason
    assert "original stale reason" in decision.failure_reason


@pytest.mark.parametrize("unsafe", [
    {"vehicle_armed": True}, {"offboard_active": True}, {"landed": False},
])
@pytest.mark.parametrize("during_recovery", [False, True])
def test_no_reset_when_vehicle_leaves_startup_boundary(unsafe, during_recovery):
    """Never reset after armed, OFFBOARD, or airborne evidence."""
    controller = BcFlightController()
    prestream(controller)
    if during_recovery:
        controller.step(1.0, stale())
    decision = controller.step(2.0, replace(stale(), **unsafe))
    assert decision.state == BcFlightState.HOLDING
    assert decision.actions == ("SELECT_HOLD",)
    cleanup = controller.step(2.1, replace(ready(), **unsafe))
    assert "DISABLE_STREAM" not in cleanup.actions
    assert "DISABLE_OUTPUT" not in cleanup.actions


def test_unknown_vehicle_state_waits_without_reset():
    """Missing fresh confirmation is not proof that a reset is safe."""
    controller = BcFlightController()
    prestream(controller)
    unknown = replace(stale(), recovery_vehicle_state_fresh=False)
    assert controller.step(1.0, unknown).actions == ()
    assert controller.step(2.0, unknown).actions == ()
    assert controller.step(3.0, stale()).actions == ("DISABLE_STREAM",)
    assert len(controller.recovery_history) == 1


def test_offboard_history_prevents_reset_even_after_leaving_offboard():
    """A later disarmed status cannot reopen the startup recovery boundary."""
    controller = BcFlightController()
    prestream(controller)
    controller.step(1.0, replace(ready(), stream_stable=True))
    controller.step(1.1, replace(ready(), offboard_active=True))
    assert controller.step(1.2, stale()).state == BcFlightState.HOLDING
    decision = controller.step(1.3, ready())
    assert "DISABLE_STREAM" not in decision.actions
    assert "DISABLE_OUTPUT" not in decision.actions


def test_pending_offboard_request_cannot_race_recovery():
    """Inactive status cannot authorize reset after a mode command was sent."""
    controller = BcFlightController()
    prestream(controller)
    assert controller.step(1.0, replace(ready(), stream_stable=True)).actions == (
        "SEND_OFFBOARD",
    )
    assert controller.step(1.1, stale()).state == BcFlightState.HOLDING
    assert controller.recovery_history == []


@pytest.mark.parametrize("missing", [
    "runtime_ready", "observations_ready", "telemetry_fresh",
    "lifecycle_selected", "source_valid", "output_ready",
])
def test_recovery_requires_complete_readiness(missing):
    """Each required readiness input must be present after resets."""
    controller = BcFlightController()
    prestream(controller)
    controller.step(1.0, stale())
    controller.step(1.1, replace(stale(), stream_reset_complete=True))
    controller.step(1.2, replace(stale(), output_reset_complete=True))
    decision = controller.step(1.3, replace(ready(), **{missing: False}))
    assert decision.state == BcFlightState.RECOVERING_READINESS
    assert "ENABLE_OUTPUT" not in decision.actions


def test_recovered_flight_uses_normal_handoff_and_cleanup():
    """Successful startup recovery must not change navigation or landing."""
    controller = BcFlightController()
    prestream(controller)
    recover(controller, 1.0, stale())
    assert controller.step(2.0, ready()).actions == ("ENABLE_STREAM",)
    evidence = replace(ready(), output_safe=True, stream_stable=True)
    assert controller.step(2.1, evidence).actions == ("SEND_OFFBOARD",)
    evidence = replace(evidence, offboard_active=True)
    assert controller.step(2.2, evidence).actions == ("SEND_ARM",)
    evidence = replace(evidence, vehicle_armed=True, landed=False)
    controller.step(2.3, evidence)
    controller.step(2.4, replace(evidence, altitude_m=1.5))
    evidence = replace(evidence, bc_enabled=True, bc_ready=True)
    controller.step(2.5, evidence)
    evidence = replace(evidence, bc_selected=True, lifecycle_selected=False)
    assert controller.step(2.6, evidence).state == BcFlightState.NAVIGATING
    controller.step(2.7, replace(evidence, terminal_reason="success"))
    cleanup = controller.step(2.8, replace(evidence, bc_selected=False))
    assert "DISABLE_STREAM" in cleanup.actions
    assert "DISABLE_OUTPUT" in cleanup.actions
    assert "SEND_LAND" in cleanup.actions


def supervisor_adapter():
    """Load actual adapter methods without requiring ROS in pure tests."""
    path = (
        Path(__file__).parents[1]
        / "uav_px4_control/flight/bc_flight_supervisor_node.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.bases = []
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    ), cls], type_ignores=[])
    namespace = {
        "BcFlightState": BcFlightState, "StartupStale": StartupStale,
        "math": __import__("math"), "re": __import__("re"),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return object.__new__(namespace[cls.name])


def test_adapter_caches_original_age_before_latched_updates():
    """Repeated latch status must not replace the first stale measurement."""
    node = supervisor_adapter()
    node._controller = BcFlightController()
    node._controller.state = BcFlightState.ENABLING_STREAM
    node._startup_stale = None
    node._stream_telemetry_thresholds_s = {
        "vehicle_status": 1.25, "vehicle_control_mode": 1.25,
        "vehicle_odometry": 0.25, "failsafe_flags": 1.35,
    }
    node._capture_startup_stale(SimpleNamespace(
        state="STOPPED_STALE_TELEMETRY", telemetry_age=1.359,
        stop_reason=(
            "required PX4 telemetry is stale: failsafe_flags "
            "age_s=1.359000000 threshold_s=1.350000000"
        ),
    ), "streamer")
    node._capture_startup_stale(SimpleNamespace(
        state="LATCHED_STREAM_FAULT", telemetry_age=5.0,
    ), "streamer")
    assert node._startup_stale.source == "streamer.failsafe_flags"
    assert node._startup_stale.age_s == 1.359
    assert node._startup_stale.threshold_s == 1.35


def test_reset_requires_new_ack_and_post_ack_disabled_status():
    """Cached disabled status or failed/old responses cannot advance recovery."""
    node = supervisor_adapter()
    node._controller = BcFlightController()
    node._controller._state_started_s = 10.0
    node._reset_ack_s = {}
    node._stream_receipt_s = 10.1
    node._stream_status = SimpleNamespace(
        state="STREAM_DISABLED", stream_enable_requested=False,
    )
    assert not node._reset_confirmed("DISABLE_STREAM", 10.2)
    node._reset_ack_s["DISABLE_STREAM"] = 9.0
    assert not node._reset_confirmed("DISABLE_STREAM", 10.2)
    node._reset_ack_s["DISABLE_STREAM"] = 10.15
    assert not node._reset_confirmed("DISABLE_STREAM", 10.2)
    node._stream_receipt_s = 10.18
    assert node._reset_confirmed("DISABLE_STREAM", 10.2)
    assert not node._reset_confirmed("DISABLE_STREAM", 11.0)


def test_opposite_pending_request_blocks_reset():
    """The reset cannot overtake a pending enable on the same service."""
    node = supervisor_adapter()
    node._pending = {"ENABLE_STREAM": SimpleNamespace(done=lambda: False)}
    # No client is needed: a pending opposite request must return immediately.
    node._request("DISABLE_STREAM", None, None, 10.0)


def test_rejected_reset_does_not_acknowledge_completion():
    """A service rejection cannot be mistaken for a successful reset."""
    node = supervisor_adapter()
    node._reset_ack_s = {}
    node._now_seconds = lambda: 10.0
    node._log_startup_diagnostic = lambda *args, **kwargs: None
    node.get_logger = lambda: SimpleNamespace(warning=lambda message: None)
    node._service_done.__globals__["MSG_ACTION_REJECTED"] = "{action}: {message}"
    node._service_done("DISABLE_STREAM", SimpleNamespace(result=lambda: SimpleNamespace(
        accepted=False, status_message="rejected",
    )))
    assert node._reset_ack_s == {}
    node._service_done("DISABLE_STREAM", SimpleNamespace(result=lambda: SimpleNamespace(
        accepted=True, status_message="reset",
    )))
    assert node._reset_ack_s == {"DISABLE_STREAM": 10.0}


@pytest.mark.parametrize("arming,status_age,land_age,expected", [
    (1, 0.1, 0.1, True), (0, 0.1, 0.1, False), (2, 0.1, 0.1, False),
    (1, 1.26, 0.1, False), (1, 0.1, 1.6, False),
])
def test_adapter_requires_explicit_fresh_disarmed_and_land_evidence(
    arming, status_age, land_age, expected, vehicle_status=None,
):
    """Unknown arming state and expired ground evidence cannot permit reset."""
    node = supervisor_adapter()
    node._evidence.__globals__["BcFlightEvidence"] = BcFlightEvidence
    node._controller = BcFlightController()
    node._mux_status = node._gate_status = node._stream_status = None
    node._vehicle_status = vehicle_status if vehicle_status is not None else SimpleNamespace(
        arming_state=arming, nav_state=4, failsafe=False,
        ARMING_STATE_ARMED=2, ARMING_STATE_STANDBY=1,
        NAVIGATION_STATE_OFFBOARD=14,
    )
    node._vehicle_status_receipt_s = 10.0 - status_age
    node._land_receipt_s = 10.0 - land_age
    node._land_detected = SimpleNamespace(landed=True)
    node._odometry_receipt_s = 10.0
    node._odometry = node._ground_down_m = None
    node._policy_status = {}
    node._runtime_ready = False
    node._termination_reason = ""
    node._startup_stale = None
    node._vehicle_status_timeout_s = 1.25
    node._vehicle_odometry_timeout_s = 0.25
    node._reset_confirmed = lambda action, now: False
    assert node._evidence(10.0).recovery_vehicle_state_fresh is expected


def test_recovery_evidence_with_installed_px4_message():
    """Verify the evidence callback against the installed ROS message contract."""
    messages = pytest.importorskip("px4_msgs.msg")
    status_type = messages.VehicleStatus
    for arming in (
        status_type.ARMING_STATE_INIT, status_type.ARMING_STATE_STANDBY,
        status_type.ARMING_STATE_ARMED, status_type.ARMING_STATE_STANDBY_ERROR,
        status_type.ARMING_STATE_SHUTDOWN, status_type.ARMING_STATE_IN_AIR_RESTORE,
    ):
        status = status_type(arming_state=arming, nav_state=4)
        test_adapter_requires_explicit_fresh_disarmed_and_land_evidence(
            arming, 0.1, 0.1, arming == status_type.ARMING_STATE_STANDBY,
            vehicle_status=status,
        )
