"""Pure regression tests for BC handoff and evaluation termination."""

from dataclasses import replace

from uav_px4_control.bc_episode_monitor import (
    TerminationConfig,
    select_terminal_reason,
)
from uav_px4_control.bc_flight_models import (
    BcFlightController,
    BcFlightEvidence,
    BcFlightState,
)
from uav_px4_control.control_mux import ControlSourceMux, fixed_candidate
from uav_px4_control.control_source_models import BC_POLICY, ControlMuxConfig


def test_mux_selects_independent_bc_policy() -> None:
    """Select BC through its own source identity and topic contract."""
    mux = ControlSourceMux(ControlMuxConfig(switch_hold_duration_s=0.0))
    mux.accept_candidate(
        BC_POLICY,
        fixed_candidate(source=BC_POLICY, stamp_s=1.0),
        receipt_time_s=1.0,
    )
    response = mux.request_source(BC_POLICY, 1.0)
    assert response.accepted
    result = mux.step(1.01)
    assert result.active_source == BC_POLICY
    assert result.state.value == "ACTIVE_BC_POLICY"


def test_takeoff_hands_control_to_bc_without_astar_action() -> None:
    """Reach navigation using lifecycle and BC actions only."""
    controller = BcFlightController()
    decisions = []
    evidence = BcFlightEvidence(
        runtime_ready=True,
        observations_ready=True,
        telemetry_fresh=True,
    )
    decisions.append(controller.step(1.0, evidence))
    evidence = BcFlightEvidence(
        runtime_ready=True,
        observations_ready=True,
        telemetry_fresh=True,
        lifecycle_selected=True,
        source_valid=True,
    )
    decisions.append(controller.step(1.1, evidence))
    evidence = replace(evidence, output_ready=True)
    decisions.append(controller.step(1.15, evidence))
    assert decisions[-1].actions == ("ENABLE_OUTPUT",)
    evidence = replace(evidence, output_ready=False, output_safe=True)
    decisions.append(controller.step(1.2, evidence))
    evidence = replace(evidence, stream_stable=True)
    decisions.append(controller.step(1.3, evidence))
    evidence = replace(evidence, offboard_active=True)
    decisions.append(controller.step(1.4, evidence))
    evidence = replace(evidence, vehicle_armed=True, landed=False)
    decisions.append(controller.step(1.5, evidence))
    evidence = replace(evidence, altitude_m=1.5)
    decisions.append(controller.step(1.6, evidence))
    evidence = replace(evidence, bc_enabled=True, bc_ready=True)
    decisions.append(controller.step(1.7, evidence))
    evidence = replace(
        evidence, lifecycle_selected=False, bc_selected=True
    )
    decisions.append(controller.step(1.8, evidence))
    assert controller.state == BcFlightState.NAVIGATING
    actions = {action for item in decisions for action in item.actions}
    assert "SELECT_BC" in actions
    assert "SELECT_ASTAR" not in actions


def test_output_readiness_timeout_does_not_enable_gate() -> None:
    """Missing gate evidence must time out without attempting to enable."""
    controller = BcFlightController()
    evidence = BcFlightEvidence(
        runtime_ready=True,
        observations_ready=True,
        telemetry_fresh=True,
        lifecycle_selected=True,
        source_valid=True,
    )
    controller.step(1.0, evidence)
    assert controller.step(1.1, evidence).actions == ()
    assert controller.step(10.0, evidence).actions == ()
    decision = controller.step(16.2, evidence)
    assert decision.state == BcFlightState.HOLDING
    assert decision.actions == ("SELECT_HOLD",)
    assert decision.failure_reason == "PX4 output gate did not become safe"


def test_source_loss_during_prestream_returns_to_lifecycle_selection() -> None:
    """Never request OFFBOARD or ARM after lifecycle authority is lost."""
    controller = BcFlightController()
    healthy = BcFlightEvidence(
        runtime_ready=True, observations_ready=True, telemetry_fresh=True,
        lifecycle_selected=True, source_valid=True, output_safe=False,
    )
    assert controller.step(1.0, healthy).state == BcFlightState.SELECTING_LIFECYCLE
    assert controller.step(1.1, healthy).state == BcFlightState.ENABLING_OUTPUT
    assert controller.step(1.2, replace(healthy, output_ready=True)).actions == ("ENABLE_OUTPUT",)
    assert controller.step(1.3, replace(healthy, output_safe=True)).state == BcFlightState.ENABLING_STREAM
    lost = replace(healthy, lifecycle_selected=False, source_valid=False, output_safe=True)
    decision = controller.step(1.4, lost)
    assert decision.state == BcFlightState.SELECTING_LIFECYCLE
    assert "SEND_OFFBOARD" not in decision.actions
    assert "SEND_ARM" not in decision.actions
    assert "DISABLE_OUTPUT" not in decision.actions


def test_source_loss_while_requesting_arm_cannot_enter_takeoff() -> None:
    """Source loss after OFFBOARD must not reset safety latches."""
    controller = BcFlightController()
    controller.state = BcFlightState.REQUESTING_ARM
    lost = BcFlightEvidence(
        runtime_ready=True, observations_ready=True, telemetry_fresh=True,
        lifecycle_selected=False, source_valid=False, offboard_active=True,
        vehicle_armed=False,
    )
    decision = controller.step(1.0, lost)
    assert decision.state == BcFlightState.HOLDING
    assert "SEND_ARM" not in decision.actions
    assert "DISABLE_OUTPUT" not in decision.actions


def test_bc_handoff_requires_all_settled_output_evidence() -> None:
    """BC selection alone cannot announce navigation readiness."""
    controller = BcFlightController()
    controller.state = BcFlightState.SELECTING_BC
    evidence = BcFlightEvidence(
        bc_selected=True, source_valid=True, output_safe=True,
        stream_stable=False,
    )
    decision = controller.step(1.0, evidence)
    assert decision.state == BcFlightState.SELECTING_BC
    assert decision.actions == ("SELECT_BC",)
    decision = controller.step(1.1, replace(evidence, stream_stable=True))
    assert decision.state == BcFlightState.NAVIGATING


def test_termination_precedence_is_safety_first() -> None:
    """Prefer collision, bounds, and success over an elapsed timeout."""
    config = TerminationConfig(goal_tolerance_m=0.35, timeout_s=45.0)
    common = {
        "goal_distance_m": 0.1,
        "minimum_clearance_m": -0.01,
        "bc_duration_s": 50.0,
        "north_m": 8.0,
        "east_m": 0.0,
        "config": config,
    }
    assert select_terminal_reason(**common) == "collision"
    common["minimum_clearance_m"] = 1.0
    assert select_terminal_reason(**common) == "out_of_bounds"
    common["north_m"] = 3.0
    assert select_terminal_reason(**common) == "success"
    common["goal_distance_m"] = 2.0
    assert select_terminal_reason(**common) == "timeout"
