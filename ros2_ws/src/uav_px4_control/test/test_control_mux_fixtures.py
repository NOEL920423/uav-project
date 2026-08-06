"""Regression coverage for all 24 deterministic mux fixtures."""

from uav_px4_control.control_mux_comparison import (
    render_control_mux_comparison,
)
from uav_px4_control.control_mux_fixtures import run_control_mux_fixtures


def test_all_twenty_four_fixtures_match_expected_terminal() -> None:
    """Execute every locked fixture and compare terminal observations."""
    fixtures = run_control_mux_fixtures()
    assert len(fixtures) == 24
    assert len({item.fixture for item in fixtures}) == 24
    assert all(
        item.expected_terminal == item.observed_terminal
        for item in fixtures
    )


def test_fixture_metrics_obey_selected_command_limits() -> None:
    """Keep every fixture at or below the Phase 6 selected limits."""
    for item in run_control_mux_fixtures():
        assert item.maximum_selected_speed_mps <= 2.0 + 1e-8
        assert item.maximum_selected_yaw_rate_radps <= 1.5 + 1e-8
        assert item.transition_count >= 0
        assert item.hold_cycles >= 0


def test_faults_latch_and_normal_isolation_does_not() -> None:
    """Expose fault latches without contaminating healthy cases."""
    fixtures = {
        item.fixture: item for item in run_control_mux_fixtures()
    }
    assert fixtures["astar-active-stale"].fault_latched
    assert fixtures["selected-wrong-frame"].fault_latched
    assert fixtures["selected-nonfinite"].fault_latched
    assert not fixtures["unselected-stale-isolation"].fault_latched


def test_movement_handoffs_observe_zero_barriers() -> None:
    """Record the configured safe interval for all movement handoffs."""
    fixtures = {
        item.fixture: item for item in run_control_mux_fixtures()
    }
    assert fixtures["astar-to-joystick"].switch_hold_s >= 0.10
    assert fixtures["joystick-to-navrl"].switch_hold_s >= 0.10
    assert fixtures["target-stale-during-handoff"].hold_cycles > 0


def test_comparison_renderer_has_fields_and_disclaimers() -> None:
    """Keep report title, columns, rows, and safety boundary reproducible."""
    report = render_control_mux_comparison()
    assert report.startswith(
        "# Offline Control-Source Arbitration Comparison"
    )
    assert "| Fixture | Requested | Sequence | Service |" in report
    assert report.count("\n|") >= 25
    terms = ("PX4", "hardware joystick", "NavRL", "Isaac Sim", "flight")
    for term in terms:
        assert term in report
