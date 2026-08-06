"""Twenty-four deterministic pure fixtures for Phase 6 arbitration."""

import math
from dataclasses import dataclass

from uav_px4_control.control_mux import ControlSourceMux, fixed_candidate
from uav_px4_control.control_source_models import (
    ASTAR_EXPERT,
    ControlCommand,
    ControlMuxConfig,
    ControlMuxResult,
    HOLD,
    HUMAN_JOYSTICK,
    NAVRL_POLICY,
    SelectionResponse,
    Vector3,
    command_speed,
)


@dataclass(frozen=True, slots=True)
class ControlMuxFixtureObservation:
    """Comparable measurements from one deterministic mux scenario."""

    fixture: str
    requested_source: str
    command_sequence: tuple[str, ...]
    service_result: str
    switch_hold_s: float
    maximum_candidate_age_s: float
    maximum_selected_speed_mps: float
    maximum_selected_yaw_rate_radps: float
    hold_cycles: int
    transition_count: int
    fault_latched: bool
    recovery_required: bool
    expected_terminal: str
    observed_terminal: str


def _observation(
    fixture: str,
    requested: str,
    results: list[ControlMuxResult],
    responses: list[SelectionResponse],
    expected: str,
    observed: str | None = None,
    switch_hold_s: float = 0.0,
    recovery_required: bool = False,
) -> ControlMuxFixtureObservation:
    sequence: list[str] = []
    for result in results:
        value = result.state.value
        if not sequence or sequence[-1] != value:
            sequence.append(value)
    finite_ages = [
        result.selected_source_age_s for result in results
        if math.isfinite(result.selected_source_age_s)
    ]
    service = "none"
    if responses:
        service = "/".join(
            "accepted" if response.accepted else "rejected"
            for response in responses
        )
    return ControlMuxFixtureObservation(
        fixture=fixture,
        requested_source=requested,
        command_sequence=tuple(sequence),
        service_result=service,
        switch_hold_s=switch_hold_s,
        maximum_candidate_age_s=max(finite_ages, default=0.0),
        maximum_selected_speed_mps=max(
            (command_speed(result.selected_command) for result in results),
            default=0.0,
        ),
        maximum_selected_yaw_rate_radps=max(
            (abs(result.selected_command.yaw_rate_radps)
             for result in results),
            default=0.0,
        ),
        hold_cycles=sum(result.hold_active for result in results),
        transition_count=results[-1].transition_count,
        fault_latched=any(result.fault_latched for result in results),
        recovery_required=recovery_required,
        expected_terminal=expected,
        observed_terminal=observed or results[-1].state.value,
    )


def _active_astar(
    config: ControlMuxConfig | None = None,
) -> tuple[ControlSourceMux, list[ControlMuxResult], list[SelectionResponse]]:
    mux = ControlSourceMux(config)
    results = [mux.step(0.0)]
    mux.accept_candidate(ASTAR_EXPERT, fixed_candidate(stamp_s=1.0), 0.01)
    responses = [mux.request_source(ASTAR_EXPERT, 0.01)]
    results.append(mux.step(0.03))
    return mux, results, responses


def run_control_mux_fixtures() -> tuple[ControlMuxFixtureObservation, ...]:
    """Execute and return all locked Phase 6 deterministic fixtures."""
    observations: list[ControlMuxFixtureObservation] = []

    mux = ControlSourceMux()
    results = [mux.step(0.0)]
    observations.append(_observation(
        "startup-hold-no-source", HOLD, results, [],
        "HOLD_STARTUP",
    ))

    mux, results, responses = _active_astar()
    observations.append(_observation(
        "select-fresh-astar", ASTAR_EXPERT, results, responses,
        "ACTIVE_ASTAR_EXPERT",
    ))

    mux, results, responses = _active_astar()
    results.append(mux.step(0.27))
    observations.append(_observation(
        "astar-active-stale", ASTAR_EXPERT, results, responses,
        "HOLD_STALE_SOURCE", recovery_required=True,
    ))

    mux, results, responses = _active_astar()
    results.append(mux.step(0.27))
    mux.accept_candidate(ASTAR_EXPERT, fixed_candidate(stamp_s=2.0), 0.28)
    results.append(mux.step(0.30))
    observations.append(_observation(
        "stale-fault-latch", ASTAR_EXPERT, results, responses,
        "HOLD_LATCHED_FAULT", recovery_required=True,
    ))

    recovery = mux.request_source(ASTAR_EXPERT, 0.48)
    responses.append(recovery)
    results.append(mux.step(0.50))
    observations.append(_observation(
        "explicit-astar-recovery", ASTAR_EXPERT, results, responses,
        "ACTIVE_ASTAR_EXPERT", recovery_required=True,
    ))

    mux, results, responses = _active_astar()
    mux.accept_candidate(
        HUMAN_JOYSTICK,
        fixed_candidate(HUMAN_JOYSTICK, 1.1, east_mps=0.4),
        0.22,
    )
    responses.append(mux.request_source(HUMAN_JOYSTICK, 0.23))
    results.extend((mux.step(0.25), mux.step(0.34)))
    observations.append(_observation(
        "astar-to-joystick", HUMAN_JOYSTICK, results, responses,
        "ACTIVE_HUMAN_JOYSTICK", switch_hold_s=0.10,
    ))

    mux, results, responses = _active_astar()
    mux.accept_candidate(
        HUMAN_JOYSTICK, fixed_candidate(HUMAN_JOYSTICK, 1.1), 0.22
    )
    responses.append(mux.request_source(HUMAN_JOYSTICK, 0.23))
    results.extend((mux.step(0.24), mux.step(0.28), mux.step(0.32)))
    observations.append(_observation(
        "switch-barrier-observed", HUMAN_JOYSTICK, results, responses,
        "HOLD_SWITCH_BARRIER", observed="HOLD_SWITCH_BARRIER",
        switch_hold_s=0.10,
    ))

    config = ControlMuxConfig(joystick_timeout_s=0.05)
    mux, results, responses = _active_astar(config)
    mux.accept_candidate(
        HUMAN_JOYSTICK, fixed_candidate(HUMAN_JOYSTICK, 1.1), 0.22
    )
    responses.append(mux.request_source(HUMAN_JOYSTICK, 0.22))
    results.extend((mux.step(0.24), mux.step(0.33)))
    observations.append(_observation(
        "target-stale-during-handoff", HUMAN_JOYSTICK, results, responses,
        "HOLD_STALE_SOURCE", switch_hold_s=0.10,
        recovery_required=True,
    ))

    mux = ControlSourceMux()
    mux.accept_candidate(
        HUMAN_JOYSTICK, fixed_candidate(HUMAN_JOYSTICK), 0.0
    )
    responses = [mux.request_source(HUMAN_JOYSTICK, 0.0)]
    results = [mux.step(0.02)]
    mux.accept_candidate(
        NAVRL_POLICY, fixed_candidate(NAVRL_POLICY, 2.0), 0.21
    )
    responses.append(mux.request_source(NAVRL_POLICY, 0.21))
    results.extend((mux.step(0.25), mux.step(0.32)))
    observations.append(_observation(
        "joystick-to-navrl", NAVRL_POLICY, results, responses,
        "ACTIVE_NAVRL_POLICY", switch_hold_s=0.10,
    ))

    mux, results, responses = _active_astar()
    responses.append(mux.request_source("ASTAR", 0.05))
    results.append(mux.step(0.07))
    observations.append(_observation(
        "unknown-source-rejected", "ASTAR", results, responses,
        "HOLD_INVALID_SOURCE", recovery_required=True,
    ))

    malformed = (
        (
            "selected-wrong-frame",
            ControlCommand(
                ASTAR_EXPERT, 2.0, "map", Vector3(0.1, 0.0, 0.0)
            ),
            "HOLD_WRONG_FRAME",
        ),
        (
            "selected-nonfinite",
            ControlCommand(
                ASTAR_EXPERT, 2.0, "px4_ned",
                Vector3(math.nan, 0.0, 0.0),
            ),
            "HOLD_INVALID_COMMAND",
        ),
        (
            "selected-excessive-speed",
            ControlCommand(
                ASTAR_EXPERT, 2.0, "px4_ned", Vector3(2.1, 0.0, 0.0)
            ),
            "HOLD_INVALID_COMMAND",
        ),
    )
    for fixture, command, expected in malformed:
        mux, results, responses = _active_astar()
        mux.accept_candidate(ASTAR_EXPERT, command, 0.04)
        results.append(mux.step(0.05))
        observations.append(_observation(
            fixture, ASTAR_EXPERT, results, responses, expected,
            recovery_required=True,
        ))

    mux, results, responses = _active_astar()
    mux.accept_candidate(ASTAR_EXPERT, fixed_candidate(stamp_s=1.0), 0.04)
    results.append(mux.step(0.05))
    observations.append(_observation(
        "nonmonotonic-candidate-stamp", ASTAR_EXPERT, results, responses,
        "HOLD_INVALID_COMMAND", recovery_required=True,
    ))

    mux, results, responses = _active_astar()
    results.append(mux.step(0.01))
    observations.append(_observation(
        "backward-node-time", ASTAR_EXPERT, results, responses,
        "HOLD_TIME_JUMP", recovery_required=True,
    ))

    mux, results, responses = _active_astar()
    mux.accept_candidate(
        HUMAN_JOYSTICK, fixed_candidate(HUMAN_JOYSTICK), 0.04
    )
    responses.append(mux.request_source(HUMAN_JOYSTICK, 0.05))
    results.append(mux.step(0.06))
    observations.append(_observation(
        "minimum-source-dwell", HUMAN_JOYSTICK, results, responses,
        "ACTIVE_ASTAR_EXPERT",
    ))

    mux, results, responses = _active_astar()
    responses.append(mux.request_source(ASTAR_EXPERT, 0.04))
    results.append(mux.step(0.05))
    observations.append(_observation(
        "duplicate-source-request", ASTAR_EXPERT, results, responses,
        "ACTIVE_ASTAR_EXPERT",
    ))

    mux, results, responses = _active_astar()
    responses.append(mux.request_source(HOLD, 0.05))
    results.append(mux.step(0.07))
    observations.append(_observation(
        "explicit-hold", HOLD, results, responses, "HOLD_REQUESTED",
    ))

    mux = ControlSourceMux()
    mux.accept_candidate(HOLD, fixed_candidate(HOLD, north_mps=0.1), 0.0)
    results = [mux.step(0.02)]
    observations.append(_observation(
        "invalid-external-hold", HOLD, results, [], "HOLD_STARTUP",
    ))

    mux = ControlSourceMux()
    mux.accept_candidate(ASTAR_EXPERT, fixed_candidate(north_mps=0.4), 0.0)
    mux.accept_candidate(
        HUMAN_JOYSTICK,
        fixed_candidate(HUMAN_JOYSTICK, east_mps=0.7),
        0.0,
    )
    mux.accept_candidate(
        NAVRL_POLICY, fixed_candidate(NAVRL_POLICY, north_mps=-0.5), 0.0
    )
    responses = [mux.request_source(ASTAR_EXPERT, 0.0)]
    results = [mux.step(0.02)]
    observations.append(_observation(
        "simultaneous-source-exclusivity", ASTAR_EXPERT, results, responses,
        "ACTIVE_ASTAR_EXPERT",
    ))

    mux.accept_candidate(
        ASTAR_EXPERT, fixed_candidate(stamp_s=2.0, north_mps=0.4), 0.21
    )
    results.append(mux.step(0.23))
    observations.append(_observation(
        "unselected-stale-isolation", ASTAR_EXPERT, results, responses,
        "ACTIVE_ASTAR_EXPERT",
    ))

    mux = ControlSourceMux()
    mux.accept_candidate(
        ASTAR_EXPERT, fixed_candidate(north_mps=0.0), 0.0
    )
    responses = [mux.request_source(ASTAR_EXPERT, 0.0)]
    results = [mux.step(0.02)]
    mux.accept_candidate(
        ASTAR_EXPERT, fixed_candidate(stamp_s=2.0, north_mps=2.0), 0.03
    )
    results.append(mux.step(0.04))
    observations.append(_observation(
        "selected-acceleration-limiter", ASTAR_EXPERT, results, responses,
        "ACTIVE_ASTAR_EXPERT",
    ))

    mux = ControlSourceMux()
    mux.accept_candidate(
        ASTAR_EXPERT, fixed_candidate(yaw_rate_radps=0.0), 0.0
    )
    responses = [mux.request_source(ASTAR_EXPERT, 0.0)]
    results = [mux.step(0.02)]
    mux.accept_candidate(
        ASTAR_EXPERT,
        fixed_candidate(stamp_s=2.0, yaw_rate_radps=1.5),
        0.03,
    )
    results.append(mux.step(0.04))
    observations.append(_observation(
        "selected-yaw-acceleration-limiter", ASTAR_EXPERT,
        results, responses, "ACTIVE_ASTAR_EXPERT",
    ))

    mux = ControlSourceMux()
    responses = []
    results = [mux.step(0.0)]
    for index in range(1, 31):
        now = index * 0.02
        velocity = 0.6 if index < 25 else 0.0
        mux.accept_candidate(
            ASTAR_EXPERT,
            fixed_candidate(
                stamp_s=now, north_mps=velocity,
                yaw_rate_radps=0.1 if index < 20 else 0.0,
            ),
            now,
        )
        if index == 1:
            responses.append(mux.request_source(ASTAR_EXPERT, now))
        results.append(mux.step(now + 0.001))
    responses.append(mux.request_source(HOLD, 0.65))
    results.append(mux.step(0.67))
    observations.append(_observation(
        "follower-mux-plant-terminal", ASTAR_EXPERT, results, responses,
        "GOAL_HOLD", observed="GOAL_HOLD",
    ))

    if len(observations) != 24:
        raise RuntimeError(
            f"expected 24 mux fixtures, produced {len(observations)}"
        )
    return tuple(observations)
