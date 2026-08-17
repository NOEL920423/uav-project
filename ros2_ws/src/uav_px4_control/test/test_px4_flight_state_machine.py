"""Pure acceptance and fail-closed tests for the SITL flight supervisor."""

from dataclasses import replace

from uav_px4_control.px4_flight_models import (
    FlightEvidence,
    Px4FlightConfig,
    Px4FlightState,
    altitude_above_ground,
    planner_status_allows_final_path,
    vehicle_command_was_accepted,
)
from uav_px4_control.px4_flight_state_machine import Px4FlightStateMachine


READY = FlightEvidence(
    pipeline_ready=True,
    follower_command_valid=True,
    astar_selected=True,
    output_gate_ready=True,
    output_gate_safe=True,
    stream_stable=True,
    stream_rate_hz=20.0,
    telemetry_fresh=True,
    source_valid=True,
    landed=True,
)


def test_complete_offboard_arm_takeoff_track_land_sequence():
    """Every acceptance transition requires its corresponding evidence."""
    machine = Px4FlightStateMachine()
    assert machine.request_enable(True, 0.0)[0]
    assert machine.step(0.1, READY).state == Px4FlightState.SELECTING_ASTAR
    assert (
        machine.step(0.2, READY).state
        == Px4FlightState.ENABLING_OUTPUT_GATE
    )
    assert machine.step(0.3, READY).state == Px4FlightState.ENABLING_STREAM
    not_streaming = replace(READY, stream_stable=False, stream_rate_hz=0.0)
    assert (
        machine.step(0.4, not_streaming).state
        == Px4FlightState.PRESTREAMING
    )
    assert machine.step(0.5, READY).state == Px4FlightState.REQUESTING_OFFBOARD
    offboard = replace(READY, offboard_active=True)
    assert machine.step(0.6, offboard).state == Px4FlightState.REQUESTING_ARM
    armed = replace(offboard, vehicle_armed=True, landed=False)
    assert machine.step(0.7, armed).state == Px4FlightState.STARTING_TAKEOFF
    tracking = replace(armed, tracking_active=True)
    assert machine.step(0.8, tracking).state == Px4FlightState.TAKEOFF
    altitude = replace(tracking, altitude_m=1.85)
    decision = machine.step(1.0, altitude)
    assert decision.state == Px4FlightState.REPLANNING
    assert decision.actions == ("STOP_TRACKING", "PUBLISH_MISSION_SCENE")
    mission = replace(
        altitude,
        pipeline_ready=False,
        mission_trajectory_ready=True,
        tracking_active=False,
    )
    assert machine.step(1.1, mission).state == Px4FlightState.STARTING_TRACKING
    mission = replace(mission, tracking_active=True)
    assert machine.step(1.2, mission).state == Px4FlightState.TRACKING
    goal = replace(mission, goal_reached=True, goal_distance_m=0.2)
    decision = machine.step(2.0, goal)
    assert decision.state == Px4FlightState.GOAL_HOLD
    assert decision.actions == (
        "STOP_TRACKING",
        "DISABLE_STREAM",
        "DISABLE_OUTPUT_GATE",
        "SEND_LAND",
    )
    assert machine.step(2.1, goal).state == Px4FlightState.LANDING
    landed = replace(
        goal,
        vehicle_armed=False,
        offboard_active=False,
        landed=True,
        tracking_active=False,
    )
    decision = machine.step(3.0, landed)
    assert decision.state == Px4FlightState.COMPLETE
    assert "DISABLE_STREAM" in decision.actions


def test_inflight_stream_loss_requests_land_and_finishes_failed():
    """A lost safety chain never continues tracking or auto-recovers."""
    config = Px4FlightConfig(landing_timeout_s=5.0)
    machine = Px4FlightStateMachine(config)
    machine.request_enable(True, 0.0)
    machine.state = Px4FlightState.TRACKING
    machine._state_started_s = 0.0
    airborne = replace(
        READY,
        vehicle_armed=True,
        offboard_active=True,
        landed=False,
        tracking_active=True,
        stream_stable=False,
    )
    decision = machine.step(1.0, airborne)
    assert decision.state == Px4FlightState.LANDING
    assert decision.actions == (
        "STOP_TRACKING",
        "DISABLE_STREAM",
        "DISABLE_OUTPUT_GATE",
        "SEND_LAND",
    )
    assert "safety chain" in decision.failure_reason
    landed = replace(airborne, vehicle_armed=False, landed=True)
    assert machine.step(2.0, landed).state == Px4FlightState.FAILED


def test_inflight_external_environment_loss_requests_controlled_landing():
    """A stale Isaac runtime cannot leave an armed vehicle tracking."""
    machine = Px4FlightStateMachine()
    machine.request_enable(True, 0.0)
    machine.state = Px4FlightState.TRACKING
    machine._state_started_s = 0.0
    airborne = replace(
        READY,
        vehicle_armed=True,
        offboard_active=True,
        landed=False,
        tracking_active=True,
        environment_valid=False,
    )
    decision = machine.step(1.0, airborne)
    assert decision.state == Px4FlightState.LANDING
    assert decision.actions == (
        "STOP_TRACKING",
        "DISABLE_STREAM",
        "DISABLE_OUTPUT_GATE",
        "SEND_LAND",
    )
    assert "external simulator environment" in decision.failure_reason


def test_prearm_external_environment_loss_fails_without_land_command():
    """A lost Isaac heartbeat before arming closes outputs without NAV_LAND."""
    machine = Px4FlightStateMachine()
    machine.request_enable(True, 0.0)
    decision = machine.step(
        0.1, replace(READY, environment_valid=False)
    )
    assert decision.state == Px4FlightState.FAILED
    assert "SEND_LAND" not in decision.actions
    assert "DISABLE_STREAM" in decision.actions


def test_landing_evaluates_completion_after_environment_loss():
    """The original simulator fault cannot mask landing/disarm evidence."""
    machine = Px4FlightStateMachine()
    machine.request_enable(True, 0.0)
    machine.state = Px4FlightState.LANDING
    machine.failure_reason = "external simulator environment became stale"
    landed = replace(
        READY,
        environment_valid=False,
        vehicle_armed=False,
        landed=True,
    )
    decision = machine.step(1.0, landed)
    assert decision.state == Px4FlightState.FAILED
    assert "DISABLE_STREAM" in decision.actions


def test_prestream_timeout_fails_without_arm_or_land_command():
    """Failure before arm disables boundaries instead of claiming flight."""
    machine = Px4FlightStateMachine(Px4FlightConfig(
        prestream_timeout_s=1.0
    ))
    machine.request_enable(True, 0.0)
    machine.state = Px4FlightState.PRESTREAMING
    machine._state_started_s = 0.0
    decision = machine.step(1.1, replace(READY, stream_stable=False))
    assert decision.state == Px4FlightState.FAILED
    assert "DISABLE_STREAM" in decision.actions
    assert "SEND_LAND" not in decision.actions


def test_output_gate_enable_waits_for_ready_disabled_evidence():
    """The supervisor must not race the existing fail-closed gate latch."""
    machine = Px4FlightStateMachine()
    machine.request_enable(True, 0.0)
    machine.state = Px4FlightState.ENABLING_OUTPUT_GATE
    machine._state_started_s = 0.0
    not_ready = replace(
        READY,
        output_gate_ready=False,
        output_gate_safe=False,
    )
    decision = machine.step(0.1, not_ready)
    assert decision.actions == ()
    ready = replace(not_ready, output_gate_ready=True)
    decision = machine.step(0.2, ready)
    assert decision.actions == ("ENABLE_OUTPUT_GATE",)


def test_flight_altitude_uses_enable_time_local_ground_datum():
    """Reported climb is relative to the enable-time local ground datum."""
    ground_down = 0.73
    assert altitude_above_ground(ground_down, -1.27) == 2.0


def test_safe_astar_fallback_is_usable_without_valid_bspline():
    """A collision-rejected B-spline may use the validated A* final path."""
    status = (
        "SUCCESS|astar_success=true|bspline_valid=false|"
        "bspline_selected=false|final_source=ASTAR_FALLBACK|"
        "final_points=10"
    )
    assert planner_status_allows_final_path(status, bspline_valid=False)


def test_bspline_source_still_requires_independent_validity_evidence():
    """Selecting B-spline never bypasses its separate validity topic."""
    status = (
        "SUCCESS|astar_success=true|bspline_valid=true|"
        "bspline_selected=true|final_source=BSPLINE|final_points=12"
    )
    assert not planner_status_allows_final_path(
        status, bspline_valid=False
    )
    assert planner_status_allows_final_path(status, bspline_valid=True)
    assert not planner_status_allows_final_path(
        "FAILED|reason=no path", bspline_valid=True
    )


def test_takeoff_boundary_matches_follower_terminal_tolerance():
    """A safely settled 1.25 m takeoff advances toward mission planning."""
    config = Px4FlightConfig(
        takeoff_altitude_m=1.5,
        takeoff_altitude_tolerance_m=0.25,
    )
    machine = Px4FlightStateMachine(config)
    machine.request_enable(True, 0.0)
    machine.state = Px4FlightState.TAKEOFF
    machine._state_started_s = 0.0
    below = replace(READY, altitude_m=1.249)
    assert machine.step(0.1, below).state == Px4FlightState.TAKEOFF
    boundary = replace(READY, altitude_m=1.25)
    decision = machine.step(0.2, boundary)
    assert decision.state == Px4FlightState.REPLANNING
    assert decision.actions == ("STOP_TRACKING", "PUBLISH_MISSION_SCENE")


def test_land_retry_stops_only_after_exact_accepted_ack():
    """An unrelated or non-accepted ACK must not suppress a land retry."""
    assert vehicle_command_was_accepted("21:ACCEPTED", 21)
    assert not vehicle_command_was_accepted("21:IN_PROGRESS", 21)
    assert not vehicle_command_was_accepted("400:ACCEPTED", 21)
