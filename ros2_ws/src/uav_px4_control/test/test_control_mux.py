"""Deterministic arbitration, handoff, latching, and limiting tests."""

import math

import pytest

from uav_px4_control.control_mux import ControlSourceMux, fixed_candidate
from uav_px4_control.control_source_models import (
    ASTAR_EXPERT,
    ControlCommand,
    ControlMuxConfig,
    ControlMuxState,
    HOLD,
    HUMAN_JOYSTICK,
    NAVRL_POLICY,
    Vector3,
)


def magnitude(result) -> float:
    """Return summed output magnitude for exact-zero assertions."""
    command = result.selected_command
    return sum(abs(value) for value in (
        command.linear.x,
        command.linear.y,
        command.linear.z,
        command.angular_x,
        command.angular_y,
        command.yaw_rate_radps,
    ))


def activate_astar() -> ControlSourceMux:
    """Return a mux with one fresh explicitly selected A* source."""
    mux = ControlSourceMux()
    mux.step(0.0)
    mux.accept_candidate(
        ASTAR_EXPERT, fixed_candidate(stamp_s=1.0), 0.01
    )
    assert mux.request_source(ASTAR_EXPERT, 0.01).accepted
    assert mux.step(0.03).active_source == ASTAR_EXPERT
    return mux


def test_startup_and_explicit_hold_are_exact_zero() -> None:
    """Always provide an internal HOLD without an external HOLD publisher."""
    mux = ControlSourceMux()
    startup = mux.step(0.0)
    assert startup.state == ControlMuxState.HOLD_STARTUP
    assert startup.active_source == HOLD
    assert startup.hold_reason
    assert magnitude(startup) == 0.0
    response = mux.request_source(HOLD, 0.02)
    requested = mux.step(0.04)
    assert response.accepted
    assert requested.state == ControlMuxState.HOLD_REQUESTED
    assert magnitude(requested) == 0.0


def test_fresh_astar_selection_and_duplicate_are_idempotent() -> None:
    """Activate only fresh A* and do not count duplicate service requests."""
    mux = activate_astar()
    before = mux.transition_count
    response = mux.request_source(ASTAR_EXPERT, 0.04)
    result = mux.step(0.05)
    assert response.accepted
    assert mux.transition_count == before
    assert result.state == ControlMuxState.ACTIVE_ASTAR_EXPERT
    assert result.selected_command.source == ASTAR_EXPERT


def test_active_stale_latches_and_requires_explicit_recovery() -> None:
    """Fresh messages alone never resume an active-source stale fault."""
    mux = activate_astar()
    stale = mux.step(0.27)
    assert stale.state == ControlMuxState.HOLD_STALE_SOURCE
    assert stale.fault_latched
    assert magnitude(stale) == 0.0
    mux.accept_candidate(
        ASTAR_EXPERT, fixed_candidate(stamp_s=2.0), 0.28
    )
    latched = mux.step(0.30)
    assert latched.active_source == HOLD
    assert latched.state == ControlMuxState.HOLD_LATCHED_FAULT
    assert latched.fault_latched
    response = mux.request_source(ASTAR_EXPERT, 0.48)
    recovered = mux.step(0.50)
    assert response.accepted
    assert recovered.active_source == ASTAR_EXPERT


def test_movement_handoff_has_complete_zero_barrier() -> None:
    """Switch A* to joystick through exact zero and revalidate the target."""
    mux = activate_astar()
    mux.accept_candidate(
        HUMAN_JOYSTICK,
        fixed_candidate(HUMAN_JOYSTICK, stamp_s=1.1, east_mps=0.4),
        0.22,
    )
    response = mux.request_source(HUMAN_JOYSTICK, 0.23)
    barrier_a = mux.step(0.25)
    barrier_b = mux.step(0.31)
    selected = mux.step(0.34)
    assert response.accepted
    assert barrier_a.state == ControlMuxState.HOLD_SWITCH_BARRIER
    assert barrier_b.switch_in_progress
    assert magnitude(barrier_a) == magnitude(barrier_b) == 0.0
    assert selected.active_source == HUMAN_JOYSTICK
    assert selected.selected_command.source == HUMAN_JOYSTICK


def test_target_stale_during_handoff_cancels_switch() -> None:
    """Never activate a target that expires during the switch barrier."""
    config = ControlMuxConfig(
        joystick_timeout_s=0.05,
        switch_hold_duration_s=0.10,
    )
    mux = ControlSourceMux(config)
    mux.accept_candidate(ASTAR_EXPERT, fixed_candidate(stamp_s=1.0), 0.0)
    mux.request_source(ASTAR_EXPERT, 0.0)
    mux.step(0.02)
    mux.accept_candidate(
        HUMAN_JOYSTICK,
        fixed_candidate(HUMAN_JOYSTICK, stamp_s=1.1),
        0.22,
    )
    assert mux.request_source(HUMAN_JOYSTICK, 0.22).accepted
    result = mux.step(0.33)
    assert result.state == ControlMuxState.HOLD_STALE_SOURCE
    assert result.active_source == HOLD
    assert magnitude(result) == 0.0


def test_joystick_to_navrl_handoff() -> None:
    """Apply the same barrier to every movement-source ownership change."""
    mux = ControlSourceMux()
    mux.accept_candidate(
        HUMAN_JOYSTICK, fixed_candidate(HUMAN_JOYSTICK), 0.0
    )
    mux.request_source(HUMAN_JOYSTICK, 0.0)
    mux.step(0.02)
    mux.accept_candidate(
        NAVRL_POLICY, fixed_candidate(NAVRL_POLICY, stamp_s=2.0), 0.21
    )
    assert mux.request_source(NAVRL_POLICY, 0.21).accepted
    assert mux.step(0.25).hold_active
    assert mux.step(0.32).active_source == NAVRL_POLICY


def test_unknown_source_rejected_and_forces_hold() -> None:
    """Reject aliases and fail closed from an active movement source."""
    mux = activate_astar()
    response = mux.request_source("ASTAR", 0.05)
    result = mux.step(0.07)
    assert not response.accepted
    assert result.state == ControlMuxState.HOLD_INVALID_SOURCE
    assert result.active_source == HOLD
    assert magnitude(result) == 0.0


@pytest.mark.parametrize(
    "command,state",
    [
        (
            ControlCommand(
                ASTAR_EXPERT, 2.0, "map", Vector3(0.1, 0.0, 0.0)
            ),
            ControlMuxState.HOLD_WRONG_FRAME,
        ),
        (
            ControlCommand(
                ASTAR_EXPERT, 2.0, "px4_ned",
                Vector3(math.nan, 0.0, 0.0),
            ),
            ControlMuxState.HOLD_INVALID_COMMAND,
        ),
        (
            ControlCommand(
                ASTAR_EXPERT, 2.0, "px4_ned", Vector3(2.1, 0.0, 0.0)
            ),
            ControlMuxState.HOLD_INVALID_COMMAND,
        ),
    ],
)
def test_active_invalid_updates_fail_closed(
    command: ControlCommand, state: ControlMuxState
) -> None:
    """Latch wrong-frame, non-finite, and excessive active candidates."""
    mux = activate_astar()
    mux.accept_candidate(ASTAR_EXPERT, command, 0.04)
    result = mux.step(0.05)
    assert result.state == state
    assert result.fault_latched
    assert magnitude(result) == 0.0


def test_nonmonotonic_active_candidate_fails_closed() -> None:
    """Do not treat replayed publisher stamps as fresh movement evidence."""
    mux = activate_astar()
    mux.accept_candidate(
        ASTAR_EXPERT, fixed_candidate(stamp_s=1.0), 0.04
    )
    result = mux.step(0.05)
    assert result.state == ControlMuxState.HOLD_INVALID_COMMAND
    assert magnitude(result) == 0.0


def test_backward_time_clears_sources_and_requires_explicit_request() -> None:
    """Invalidate histories and enter latched time-jump HOLD."""
    mux = activate_astar()
    result = mux.step(0.01)
    assert result.state == ControlMuxState.HOLD_TIME_JUMP
    assert result.fault_latched
    assert not mux.registry.health(ASTAR_EXPERT, 0.01).received


def test_minimum_dwell_rejects_early_switch_without_mixing() -> None:
    """Keep the active source when another request arrives before dwell."""
    mux = activate_astar()
    mux.accept_candidate(
        HUMAN_JOYSTICK, fixed_candidate(HUMAN_JOYSTICK), 0.04
    )
    response = mux.request_source(HUMAN_JOYSTICK, 0.05)
    result = mux.step(0.06)
    assert not response.accepted
    assert result.active_source == ASTAR_EXPERT
    assert result.selected_command.source == ASTAR_EXPERT


def test_invalid_external_hold_does_not_remove_internal_hold() -> None:
    """The fail-closed path never depends on the external HOLD candidate."""
    mux = ControlSourceMux()
    mux.accept_candidate(
        HOLD, fixed_candidate(HOLD, north_mps=0.1), 0.0
    )
    result = mux.step(0.02)
    assert result.active_source == HOLD
    assert magnitude(result) == 0.0
    assert result.selected_command_valid


def test_simultaneous_and_unselected_stale_sources_do_not_interfere() -> None:
    """Forward only selected ownership and ignore unselected expiration."""
    mux = ControlSourceMux()
    mux.accept_candidate(
        ASTAR_EXPERT, fixed_candidate(north_mps=0.4), 0.0
    )
    mux.accept_candidate(
        HUMAN_JOYSTICK,
        fixed_candidate(HUMAN_JOYSTICK, east_mps=0.7),
        0.0,
    )
    mux.request_source(ASTAR_EXPERT, 0.0)
    first = mux.step(0.02)
    mux.accept_candidate(
        ASTAR_EXPERT, fixed_candidate(stamp_s=2.0, north_mps=0.4), 0.21
    )
    later = mux.step(0.23)
    assert first.selected_command.source == ASTAR_EXPERT
    assert later.active_source == ASTAR_EXPERT
    assert HUMAN_JOYSTICK in later.stale_sources


def test_selected_acceleration_and_yaw_acceleration_are_limited() -> None:
    """Ramp changed candidates within linear and yaw derivative limits."""
    mux = ControlSourceMux()
    mux.accept_candidate(
        ASTAR_EXPERT,
        fixed_candidate(stamp_s=1.0, north_mps=0.0),
        0.0,
    )
    mux.request_source(ASTAR_EXPERT, 0.0)
    first = mux.step(0.02)
    mux.accept_candidate(
        ASTAR_EXPERT,
        fixed_candidate(
            stamp_s=2.0, north_mps=2.0, yaw_rate_radps=1.5
        ),
        0.03,
    )
    second = mux.step(0.04)
    delta_speed = abs(
        second.selected_command.linear.x - first.selected_command.linear.x
    ) / 0.02
    delta_yaw = abs(
        second.selected_command.yaw_rate_radps
        - first.selected_command.yaw_rate_radps
    ) / 0.02
    assert delta_speed <= 1.5 + 1e-8
    assert delta_yaw <= 2.0 + 1e-8
    assert second.saturations.acceleration
    assert second.saturations.yaw_acceleration
