"""Regression test for all named Phase 8 offline stream fixtures."""

from uav_px4_control.px4_stream_fixtures import run_stream_offline_fixtures


def test_all_twenty_stream_fixtures_pass():
    """Keep the complete required synthetic matrix deterministic."""
    rows = run_stream_offline_fixtures()
    assert len(rows) == 20
    assert len({name for name, _, _ in rows}) == 20
    assert all(passed for _, passed, _ in rows)
