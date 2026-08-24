"""Regression tests for the Phase 10A dataset geometry and join contract."""

import math

import pytest

from uav_data_recorder.expert_dataset_contract import (
    SYNCHRONIZATION_TOLERANCE_S,
    TimedValue,
    contract_manifest,
    episode_outcome_success,
    goal_features,
    latest_at_or_before,
    nearest,
    ned_to_body,
    normalize_action,
    previous,
    recording_window_rejection,
    update_recording_window,
)


def test_ned_body_rotation_matches_forward_right_contract():
    """North-facing and east-facing yaw map NED into body FR correctly."""
    assert ned_to_body(1.0, 0.0, 0.0) == pytest.approx((1.0, 0.0))
    assert ned_to_body(0.0, 1.0, math.pi / 2.0) == pytest.approx((1.0, 0.0))
    assert ned_to_body(1.0, 0.0, math.pi / 2.0) == pytest.approx((0.0, -1.0))


def test_goal_distance_and_action_normalization_preserve_bc_contract():
    """V1 uses a 10 m distance scale and unchanged normalized action limits."""
    forward, right, raw, normalized = goal_features(0, 0, 3, 4, 0)
    assert (forward, right) == pytest.approx((0.6, 0.8))
    assert raw == pytest.approx(5.0)
    assert normalized == pytest.approx(0.5)
    assert normalize_action(0.5, -0.4, 0.25) == pytest.approx(
        (0.5, -0.5, 0.25)
    )
    assert normalize_action(5.0, -5.0, 5.0) == (1.0, -1.0, 1.0)


def test_timestamp_join_exposes_error_and_previous_control_sample():
    """Nearest matching and previous-action lookup remain explicit."""
    values = [
        TimedValue(1.00, "a"),
        TimedValue(1.05, "b"),
        TimedValue(1.10, "c"),
    ]
    selected = nearest(values, 1.06)
    assert selected == values[1]
    assert abs(selected.timestamp_s - 1.06) < SYNCHRONIZATION_TOLERANCE_S
    assert previous(values, selected) == values[0]


def test_flight_status_lookup_is_causal_at_tracking_boundaries():
    """Lifecycle lookup never uses a future transition for an earlier image."""
    values = [
        TimedValue(1.00, "STARTING_TRACKING"),
        TimedValue(1.05, "TRACKING"),
        TimedValue(1.10, "GOAL_HOLD"),
    ]
    assert latest_at_or_before(values, 1.049) == values[0]
    assert latest_at_or_before(values, 1.050) == values[1]
    assert latest_at_or_before(values, 1.099) == values[1]
    assert latest_at_or_before(values, 1.100) == values[2]


def test_recording_window_is_tracking_inclusive_and_goal_hold_exclusive():
    """Normal tracking and settling remain while GOAL_HOLD ends recording."""
    start, end = update_recording_window(None, None, "REPLANNING", 9.0)
    assert (start, end) == (None, None)
    start, end = update_recording_window(start, end, "TRACKING", 10.0)
    assert (start, end) == (10.0, None)
    assert recording_window_rejection(start, end, 9.999) == (
        "before_tracking_window"
    )
    assert recording_window_rejection(start, end, 10.0) is None
    assert recording_window_rejection(start, end, 10.5) is None

    start, end = update_recording_window(start, end, "TRACKING", 10.8)
    assert (start, end) == (10.0, None)
    start, end = update_recording_window(start, end, "GOAL_HOLD", 11.0)
    assert recording_window_rejection(start, end, 10.999) is None
    assert recording_window_rejection(start, end, 11.0) == (
        "after_tracking_window"
    )
    assert recording_window_rejection(start, end, 12.0) == (
        "after_tracking_window"
    )
    assert update_recording_window(start, end, "LANDING", 12.0) == (
        10.0, 11.0
    )
    assert update_recording_window(start, end, "COMPLETE", 13.0) == (
        10.0, 11.0
    )


def test_failure_transition_closes_recording_after_tracking():
    """The first non-TRACKING failure phase becomes the immutable end bound."""
    start, end = update_recording_window(None, None, "TRACKING", 20.0)
    assert update_recording_window(start, end, "REPLANNING", 19.9) == (
        20.0, None
    )
    start, end = update_recording_window(start, end, "LANDING", 20.5)
    assert (start, end) == (20.0, 20.5)
    assert recording_window_rejection(start, end, 20.499) is None
    assert recording_window_rejection(start, end, 20.5) == (
        "after_tracking_window"
    )
    assert update_recording_window(start, end, "FAILED", 21.0) == (
        20.0, 20.5
    )


def test_manifest_defines_raw_image_and_rebuild_dimensions():
    """Machine contract retains master images and 72D rebuild shape."""
    manifest = contract_manifest()
    assert manifest["master_image"]["width"] == 320
    assert manifest["master_image"]["height"] == 180
    assert manifest["master_image"]["format"] == "jpeg"
    assert manifest["encoder_preprocessing"]["latent_dimension"] == 64
    assert manifest["state"]["state_dimension"] == 8
    assert manifest["state"]["observation_dimension"] == 72
    assert manifest["expert_action"]["dimension"] == 3


def test_episode_success_uses_accumulated_phase_evidence():
    """A COMPLETE snapshot need not repeat the earlier goal flag."""
    assert episode_outcome_success("COMPLETE", "", True, True, True)
    assert not episode_outcome_success("COMPLETE", "", False, True, True)
    assert not episode_outcome_success("FAILED", "fault", True, True, True)
