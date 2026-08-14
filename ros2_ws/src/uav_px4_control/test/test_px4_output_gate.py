"""Regression tests for the pure fail-closed PX4 output gate."""

from dataclasses import replace

import pytest

from uav_px4_control.px4_boundary_models import (
    CandidateValidation,
    MuxHealthEvidence,
    Px4MappingConfig,
    Px4OutputGateState,
    Px4TelemetryState,
    Px4VelocitySetpointCandidate,
)
from uav_px4_control.px4_output_gate import Px4OutputSafetyGate
from uav_px4_control.px4_synthetic_telemetry import (
    SYNTHETIC_TELEMETRY_FIXTURES,
    synthetic_telemetry_fixture,
)


def candidate(now: float = 1.0, timestamp_us: int = 1_000_000):
    """Return a healthy velocity-only candidate."""
    nan3 = (float("nan"),) * 3
    return Px4VelocitySetpointCandidate(
        timestamp_us=timestamp_us,
        position_ned_m=nan3,
        velocity_ned_mps=(0.4, -0.2, 0.1),
        acceleration_ned_mps2=nan3,
        jerk_ned_mps3=nan3,
        yaw_ned_rad=float("nan"),
        yaw_rate_ned_radps=0.2,
        source="ASTAR_EXPERT",
        frame_id="px4_ned",
        selected_receipt_time_s=now,
    )


def valid() -> CandidateValidation:
    """Return a successful independent validation result."""
    return CandidateValidation(True, "candidate accepted", 0.45, 0.46)


def mux(now: float = 1.0, **changes) -> MuxHealthEvidence:
    """Return healthy mux evidence with optional field overrides."""
    values = {
        "received": True,
        "selected_command_valid": True,
        "hold_active": False,
        "active_source": "ASTAR_EXPERT",
        "receipt_time_s": now,
    }
    values.update(changes)
    return MuxHealthEvidence(**values)


def telemetry(now: float = 1.0, timestamp_us: int = 1_000_000, **changes):
    """Return healthy synthetic telemetry with optional overrides."""
    values = {"receipt_time_s": now, "timestamp_us": timestamp_us}
    values.update(changes)
    return Px4TelemetryState(**values)


def step(
    gate, now=1.0, cand=None, check=None, health=None, vehicle=None
):
    """Evaluate the gate with healthy defaults."""
    return gate.step(
        now,
        candidate(now) if cand is None else cand,
        valid() if check is None else check,
        mux(now) if health is None else health,
        telemetry(now) if vehicle is None else vehicle,
    )


def enable_healthy(gate: Px4OutputSafetyGate) -> None:
    """Advance a gate through pending to safe forwarding permission."""
    assert gate.request_enable(True).accepted
    assert step(gate).state == Px4OutputGateState.ENABLE_PENDING
    assert step(gate, 1.01).safe_to_forward


def test_startup_is_disabled_and_healthy_requires_explicit_enable() -> None:
    """Healthy evidence alone must remain READY_DISABLED."""
    gate = Px4OutputSafetyGate()
    assert gate.state == Px4OutputGateState.OUTPUT_DISABLED
    result = step(gate)
    assert result.state == Px4OutputGateState.READY_DISABLED
    assert not result.enabled
    assert not result.safe_to_forward


def test_enable_requires_one_complete_healthy_cycle() -> None:
    """Explicit enable still requires a complete pending cycle."""
    gate = Px4OutputSafetyGate()
    response = gate.request_enable(True)
    pending = step(gate)
    forwarding = step(gate, 1.01)
    assert response.accepted
    assert pending.state == Px4OutputGateState.ENABLE_PENDING
    assert not pending.safe_to_forward
    assert forwarding.state == Px4OutputGateState.SAFE_TO_FORWARD
    assert forwarding.safe_to_forward


@pytest.mark.parametrize(
    "expected,kwargs",
    [
        (
            Px4OutputGateState.DISABLED_STALE_COMMAND,
            {"now": 1.30, "cand": candidate(1.0), "health": mux(1.30)},
        ),
        (
            Px4OutputGateState.DISABLED_INVALID_COMMAND,
            {
                "now": 1.02,
                "check": CandidateValidation(False, "bad mapping", 0.0, 0.0),
            },
        ),
        (
            Px4OutputGateState.DISABLED_MUX_HOLD,
            {"now": 1.02, "health": mux(1.02, hold_active=True)},
        ),
        (
            Px4OutputGateState.DISABLED_STALE_TELEMETRY,
            {
                "now": 1.60,
                "cand": candidate(1.60),
                "health": mux(1.60),
                "vehicle": telemetry(1.0),
            },
        ),
        (
            Px4OutputGateState.DISABLED_FAILSAFE,
            {"now": 1.02, "vehicle": telemetry(1.02, failsafe=True)},
        ),
        (
            Px4OutputGateState.WAITING_VEHICLE_STATE,
            {
                "now": 1.02,
                "vehicle": telemetry(1.02, local_velocity_valid=False),
            },
        ),
    ],
)
def test_enabled_faults_fail_closed_and_latch(expected, kwargs) -> None:
    """Every active evidence fault must revoke and latch permission."""
    gate = Px4OutputSafetyGate()
    enable_healthy(gate)
    result = step(gate, **kwargs)
    assert result.state == expected
    assert not result.safe_to_forward
    assert result.fault_latched


def test_recovery_data_cannot_clear_latch_without_disable_reset() -> None:
    """Fresh recovery evidence must not automatically clear a latch."""
    gate = Px4OutputSafetyGate()
    enable_healthy(gate)
    step(gate, 1.02, vehicle=telemetry(1.02, failsafe=True))
    recovered = step(gate, 1.03)
    rejected = gate.request_enable(True)
    assert recovered.state == Px4OutputGateState.LATCHED_FAULT
    assert not recovered.safe_to_forward
    assert not rejected.accepted
    assert "disable/reset" in rejected.message


def test_explicit_disable_reset_then_reenable_recovers() -> None:
    """Explicit reset and re-enable is the sole recovery sequence."""
    gate = Px4OutputSafetyGate()
    enable_healthy(gate)
    step(gate, 1.02, vehicle=telemetry(1.02, failsafe=True))
    assert gate.request_enable(False).accepted
    assert not gate.fault_latched
    assert gate.request_enable(True).accepted
    assert step(gate, 1.03).state == Px4OutputGateState.ENABLE_PENDING
    assert step(gate, 1.04).safe_to_forward


def test_backward_clock_and_candidate_timestamp_fail_closed() -> None:
    """Backward clock and candidate stamps revoke permission."""
    gate = Px4OutputSafetyGate()
    enable_healthy(gate)
    jumped = step(gate, 0.50)
    assert jumped.state == Px4OutputGateState.DISABLED_TIME_JUMP
    gate.request_enable(False)
    gate.request_enable(True)
    step(gate, 2.0, cand=candidate(2.0, 2_000_000))
    backward = step(gate, 2.01, cand=candidate(2.01, 1_900_000))
    assert backward.state == Px4OutputGateState.DISABLED_INVALID_COMMAND


def test_vehicle_state_change_while_forwarding_latches() -> None:
    """A vehicle-state transition while enabled must latch closed."""
    gate = Px4OutputSafetyGate()
    enable_healthy(gate)
    changed = step(gate, 1.02, vehicle=telemetry(1.02, nav_state=14))
    assert changed.state == Px4OutputGateState.DISABLED_STATE_CHANGE
    assert changed.fault_latched
    assert not changed.safe_to_forward


def test_explicit_flight_config_allows_healthy_vehicle_state_transition():
    """The flight-only override keeps every other gate while allowing arm."""
    gate = Px4OutputSafetyGate(Px4MappingConfig(
        lock_vehicle_state_signature=False
    ))
    enable_healthy(gate)
    changed = step(
        gate,
        1.02,
        vehicle=telemetry(
            1.02,
            timestamp_us=1_020_000,
            arming_state=2,
            nav_state=14,
            offboard_active=True,
        ),
    )
    assert changed.state == Px4OutputGateState.SAFE_TO_FORWARD
    assert changed.safe_to_forward
    assert not changed.fault_latched


def test_waiting_inputs_are_explicit_while_disabled() -> None:
    """Missing startup evidence uses an explicit waiting state."""
    gate = Px4OutputSafetyGate()
    waiting = gate.step(0.0, None, None, None, None)
    assert waiting.state == Px4OutputGateState.WAITING_SELECTED_COMMAND
    assert not waiting.safe_to_forward


def test_explicit_disable_request_removes_permission() -> None:
    """A disable request synchronously removes permission and resets."""
    gate = Px4OutputSafetyGate()
    enable_healthy(gate)
    response = gate.request_enable(False)
    assert response.accepted
    assert not response.enabled
    assert gate.state == Px4OutputGateState.OUTPUT_DISABLED


def test_source_handoff_while_enabled_latches_state_change() -> None:
    """A source owner change requires a new explicit enable cycle."""
    gate = Px4OutputSafetyGate()
    enable_healthy(gate)
    moved = replace(
        candidate(1.02, 1_020_000),
        source="HUMAN_JOYSTICK",
    )
    result = step(
        gate,
        1.02,
        cand=moved,
        health=mux(1.02, active_source="HUMAN_JOYSTICK"),
    )
    assert result.state == Px4OutputGateState.DISABLED_STATE_CHANGE
    assert result.fault_latched


def test_hold_and_unknown_active_source_never_forward() -> None:
    """HOLD and unknown mux owners never receive permission."""
    for health in (
        mux(1.0, hold_active=True, active_source="HOLD"),
        mux(1.0, active_source="UNKNOWN"),
    ):
        gate = Px4OutputSafetyGate()
        gate.request_enable(True)
        result = step(gate, health=health)
        assert result.state == Px4OutputGateState.DISABLED_MUX_HOLD
        assert not result.safe_to_forward


def test_multiple_healthy_cycles_remain_safe() -> None:
    """Stable fresh evidence remains safe across repeated cycles."""
    gate = Px4OutputSafetyGate()
    enable_healthy(gate)
    for index in range(2, 12):
        now = 1.0 + index / 100.0
        result = step(
            gate,
            now,
            cand=candidate(now, int(now * 1_000_000)),
            health=mux(now),
            vehicle=telemetry(now, int(now * 1_000_000)),
        )
        assert result.state == Px4OutputGateState.SAFE_TO_FORWARD
        assert result.safe_to_forward


def test_all_required_synthetic_telemetry_fixtures_exist() -> None:
    """Expose every required named synthetic telemetry condition."""
    assert len(SYNTHETIC_TELEMETRY_FIXTURES) == 12
    for name in SYNTHETIC_TELEMETRY_FIXTURES:
        result = synthetic_telemetry_fixture(name, 2.0, 2_000_000)
        assert result is None or isinstance(result, Px4TelemetryState)


@pytest.mark.parametrize("name", ("offboard_inactive", "disarmed"))
def test_mapping_permission_does_not_require_armed_or_offboard(
    name,
) -> None:
    """Permit diagnostic mapping before arming and OFFBOARD activation."""
    gate = Px4OutputSafetyGate()
    gate.request_enable(True)
    vehicle = synthetic_telemetry_fixture(name, 1.0, 1_000_000)
    assert (
        step(gate, vehicle=vehicle).state
        == Px4OutputGateState.ENABLE_PENDING
    )
    vehicle = synthetic_telemetry_fixture(name, 1.01, 1_010_000)
    assert step(gate, 1.01, vehicle=vehicle).safe_to_forward


@pytest.mark.parametrize("name", ("offboard_active", "armed"))
def test_armed_and_offboard_active_synthetic_states_can_be_healthy(
    name,
) -> None:
    """Model armed and OFFBOARD-active telemetry without controlling PX4."""
    gate = Px4OutputSafetyGate()
    gate.request_enable(True)
    vehicle = synthetic_telemetry_fixture(name, 1.0, 1_000_000)
    assert (
        step(gate, vehicle=vehicle).state
        == Px4OutputGateState.ENABLE_PENDING
    )
    vehicle = synthetic_telemetry_fixture(name, 1.01, 1_010_000)
    assert step(gate, 1.01, vehicle=vehicle).safe_to_forward
