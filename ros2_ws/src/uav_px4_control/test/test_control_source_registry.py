"""Regression tests for source identity, records, and freshness."""

import math

import pytest

from uav_px4_control.control_mux import fixed_candidate
from uav_px4_control.control_source_models import (
    ASTAR_EXPERT,
    BC_POLICY,
    CONTROL_SOURCES,
    ControlCommand,
    ControlMuxConfig,
    FLIGHT_LIFECYCLE,
    HOLD,
    HUMAN_JOYSTICK,
    NAVRL_POLICY,
    SOURCE_TOPICS,
    Vector3,
)
from uav_px4_control.control_source_registry import ControlSourceRegistry


def test_exact_source_registry_and_topic_contract() -> None:
    """Expose only the canonical source identifiers and topics."""
    assert CONTROL_SOURCES == (
        HOLD,
        ASTAR_EXPERT,
        FLIGHT_LIFECYCLE,
        BC_POLICY,
        HUMAN_JOYSTICK,
        NAVRL_POLICY,
    )
    assert SOURCE_TOPICS == {
        ASTAR_EXPERT: "/uav/control/astar_command",
        FLIGHT_LIFECYCLE: "/uav/control/lifecycle_command",
        BC_POLICY: "/uav/control/bc_command",
        HUMAN_JOYSTICK: "/uav/control/joystick_command",
        NAVRL_POLICY: "/uav/control/navrl_command",
        HOLD: "/uav/control/hold_command",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("publish_rate_hz", 0.0),
        ("astar_timeout_s", -1.0),
        ("lifecycle_timeout_s", -1.0),
        ("bc_timeout_s", -1.0),
        ("switch_hold_duration_s", -0.1),
        ("maximum_selected_speed_mps", math.nan),
        ("hold_command_epsilon", math.inf),
    ],
)
def test_configuration_rejects_invalid_numeric_values(
    field: str, value: float
) -> None:
    """Reject non-finite, zero-rate, or negative safety configuration."""
    with pytest.raises(ValueError):
        ControlMuxConfig(**{field: value})


def test_configuration_rejects_wrong_types_and_contradictions() -> None:
    """Reject non-HOLD startup, non-booleans, and component over total."""
    with pytest.raises(ValueError):
        ControlMuxConfig(default_source=ASTAR_EXPERT)
    with pytest.raises(ValueError):
        ControlMuxConfig(reject_wrong_frame=1)
    with pytest.raises(ValueError):
        ControlMuxConfig(
            maximum_selected_speed_mps=1.0,
            maximum_selected_horizontal_speed_mps=1.1,
        )


def test_never_received_fresh_boundary_and_stale_age() -> None:
    """Use node receipt age and accept the exact timeout boundary."""
    registry = ControlSourceRegistry(ControlMuxConfig())
    assert not registry.health(ASTAR_EXPERT, 1.0).received
    registry.update(ASTAR_EXPERT, fixed_candidate(stamp_s=7.0), 1.0)
    boundary = registry.health(ASTAR_EXPERT, 1.25)
    assert boundary.healthy
    assert boundary.age_s == pytest.approx(0.25)
    stale = registry.health(ASTAR_EXPERT, 1.250001)
    assert not stale.healthy
    assert not stale.fresh
    assert "stale" in stale.reason


@pytest.mark.parametrize(
    "command,reason",
    [
        (
            ControlCommand(
                ASTAR_EXPERT, 1.0, "map", Vector3(0.1, 0.0, 0.0)
            ),
            "frame",
        ),
        (
            ControlCommand(
                ASTAR_EXPERT, 1.0, "px4_ned",
                Vector3(math.nan, 0.0, 0.0),
            ),
            "non-finite",
        ),
        (
            ControlCommand(
                ASTAR_EXPERT, 1.0, "px4_ned", Vector3(2.1, 0.0, 0.0)
            ),
            "speed",
        ),
        (
            ControlCommand(
                ASTAR_EXPERT, 1.0, "px4_ned", Vector3(0.1, 0.0, 0.0),
                angular_x=0.1,
            ),
            "angular",
        ),
    ],
)
def test_static_candidate_rejections(
    command: ControlCommand, reason: str
) -> None:
    """Classify malformed source data without raising or sanitizing it."""
    registry = ControlSourceRegistry(ControlMuxConfig())
    record = registry.update(ASTAR_EXPERT, command, 1.0)
    assert not record.valid
    assert reason in record.reason
    assert not registry.health(ASTAR_EXPERT, 1.01).healthy


def test_nonmonotonic_stamp_invalidates_latest_source_record() -> None:
    """Equal publisher stamps do not refresh a healthy replay."""
    registry = ControlSourceRegistry(ControlMuxConfig())
    registry.update(ASTAR_EXPERT, fixed_candidate(stamp_s=2.0), 1.0)
    record = registry.update(
        ASTAR_EXPERT, fixed_candidate(stamp_s=2.0), 1.1
    )
    assert not record.valid
    assert "non-monotonic" in record.reason


def test_repeated_payload_with_new_stamp_refreshes_receipt_time() -> None:
    """Treat identical command values as fresh when new heartbeats arrive."""
    registry = ControlSourceRegistry(ControlMuxConfig())
    first = fixed_candidate(stamp_s=2.0, north_mps=0.6)
    second = fixed_candidate(stamp_s=2.04, north_mps=0.6)
    registry.update(ASTAR_EXPERT, first, 1.0)
    record = registry.update(ASTAR_EXPERT, second, 1.04)
    health = registry.health(ASTAR_EXPERT, 1.05)
    assert record.update_count == 2
    assert health.healthy
    assert health.update_count == 2
    assert health.age_s == pytest.approx(0.01)


def test_external_hold_must_be_zero_but_internal_hold_is_independent() -> None:
    """Reject a nonzero external HOLD candidate in its own source record."""
    registry = ControlSourceRegistry(ControlMuxConfig())
    invalid = fixed_candidate(source=HOLD, north_mps=0.1)
    assert not registry.update(HOLD, invalid, 1.0).valid
    assert registry.record(ASTAR_EXPERT).reason == "never received"


def test_registry_clear_invalidates_every_source() -> None:
    """Backward-time recovery can discard all freshness evidence."""
    registry = ControlSourceRegistry(ControlMuxConfig())
    registry.update(ASTAR_EXPERT, fixed_candidate(), 1.0)
    registry.clear()
    assert all(
        not registry.health(source, 1.1).received
        for source in CONTROL_SOURCES
    )


def test_unknown_source_is_rejected() -> None:
    """Do not accept undocumented aliases."""
    registry = ControlSourceRegistry(ControlMuxConfig())
    with pytest.raises(ValueError):
        registry.update("ASTAR", fixed_candidate(), 1.0)
    with pytest.raises(ValueError):
        registry.health("JOYSTICK", 1.0)
