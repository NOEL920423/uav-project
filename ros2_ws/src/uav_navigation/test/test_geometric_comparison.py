"""Regression contract for the reproducible geometric comparison tool."""

from uav_navigation.geometric_comparison import (
    comparison_scenes,
    render_markdown,
    run_comparisons,
)


def test_comparison_has_fixed_accepted_and_rejected_scenes() -> None:
    """Exercise at least five fixed scenes and both selection outcomes."""
    scenes = comparison_scenes()
    rows = run_comparisons()
    assert len(scenes) == len(rows) >= 5
    assert any(row.accepted for row in rows)
    assert any(not row.accepted for row in rows)
    assert all(
        row.final_source in {"BSPLINE", "ASTAR_FALLBACK"}
        for row in rows
    )


def test_comparison_is_deterministic() -> None:
    """Produce exactly equal structured metrics on repeated runs."""
    assert run_comparisons() == run_comparisons()


def test_report_is_explicitly_geometric_and_complete() -> None:
    """Keep required labels and prohibit an implicit flight claim."""
    report = render_markdown(run_comparisons())
    assert "Geometric Path Comparison" in report
    assert "not a flight-performance" in report
    assert "Final source" in report
    assert "Complete metric set" in report
    assert "Curvature variance" in report
