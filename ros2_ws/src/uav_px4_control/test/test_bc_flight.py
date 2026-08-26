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
    evidence = replace(evidence, output_ready=True, output_safe=True)
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
