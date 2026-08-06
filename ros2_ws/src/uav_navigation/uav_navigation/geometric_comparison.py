"""Reproducible non-flight geometric comparison for Phase 3 paths."""

from dataclasses import dataclass

from uav_navigation.astar_planner import plan_path
from uav_navigation.models import (
    BSplineConfig,
    CircularObstacle,
    PathMetrics,
    Point3D,
)

ALTITUDE_NED_M = -2.0


@dataclass(frozen=True, slots=True)
class ComparisonScene:
    """One fixed deterministic comparison scene."""

    name: str
    start: Point3D
    goal: Point3D
    obstacles: tuple[CircularObstacle, ...] = ()
    bspline_config: BSplineConfig = BSplineConfig()


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """A* baseline and B-spline candidate metrics for one scene."""

    scene: str
    accepted: bool
    rejection_reason: str
    final_source: str
    astar: PathMetrics
    bspline: PathMetrics


def _point(x: float, y: float) -> Point3D:
    return Point3D(x, y, ALTITUDE_NED_M)


def _tower(
    name: str,
    x: float,
    y: float,
    radius: float = 0.2,
) -> CircularObstacle:
    return CircularObstacle(name, Point3D(x, y, -1.5), radius, 3.0)


def comparison_scenes() -> tuple[ComparisonScene, ...]:
    """Return six fixed scenes covering accepted and rejected candidates."""
    center = _tower("center", 0.0, 0.0)
    return (
        ComparisonScene(
            "open-straight",
            _point(-2.0, 0.0),
            _point(2.0, 0.0),
        ),
        ComparisonScene(
            "open-diagonal",
            _point(-2.0, -0.6),
            _point(2.0, 0.6),
        ),
        ComparisonScene(
            "single-obstacle",
            _point(-2.0, -0.4),
            _point(2.0, 0.4),
            (center,),
        ),
        ComparisonScene(
            "large-obstacle-detour",
            _point(-3.0, 0.0),
            _point(3.0, 0.0),
            (_tower("large-center", 0.0, 0.0, 1.0),),
        ),
        ComparisonScene(
            "strict-clearance-gate",
            _point(-2.0, -0.4),
            _point(2.0, 0.4),
            (center,),
            BSplineConfig(bspline_minimum_clearance_m=0.40),
        ),
        ComparisonScene(
            "strict-curvature-gate",
            _point(-2.0, -0.4),
            _point(2.0, 0.4),
            (center,),
            BSplineConfig(bspline_maximum_curvature=0.01),
        ),
    )


def run_comparisons() -> tuple[ComparisonRow, ...]:
    """Plan every scene and retain both geometric metric sets."""
    rows: list[ComparisonRow] = []
    for scene in comparison_scenes():
        result = plan_path(
            scene.start,
            scene.goal,
            scene.obstacles,
            bspline_config=scene.bspline_config,
        )
        if not result.success:
            message = f"{scene.name}: planning failed: {result.status}"
            raise RuntimeError(message)
        if (
            result.simplified_metrics is None
            or result.bspline_metrics is None
        ):
            raise RuntimeError(f"{scene.name}: comparison metrics are missing")
        rows.append(
            ComparisonRow(
                scene=scene.name,
                accepted=result.bspline_valid,
                rejection_reason=result.bspline_rejection_reason or "none",
                final_source=result.final_path_source,
                astar=result.simplified_metrics,
                bspline=result.bspline_metrics,
            )
        )
    return tuple(rows)


def _number(value: float) -> str:
    """Format deterministic metrics compactly without hiding infinity."""
    return f"{value:.6f}"


def render_markdown(rows: tuple[ComparisonRow, ...]) -> str:
    """Render the repository's reproducible Markdown comparison tables."""
    lines = [
        "# Phase 3 Geometric Path Comparison",
        "",
        (
            "This is a deterministic geometric comparison only; it is not a "
            "flight-performance or dynamic-feasibility comparison."
        ),
        "",
        (
            "| Scene | Candidate | Rejection / fallback reason | A* length "
            "[m] | B-spline length [m] | A* clearance [m] | B-spline "
            "clearance [m] | A* max heading [rad] | B-spline max heading "
            "[rad] | A* heading variance [rad^2] | B-spline heading "
            "variance [rad^2] | A* max curvature [1/m] | B-spline max "
            "curvature [1/m] | Final source |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---|"
        ),
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row.scene,
                    "accepted" if row.accepted else "rejected",
                    row.rejection_reason,
                    _number(row.astar.path_length_m),
                    _number(row.bspline.path_length_m),
                    _number(row.astar.minimum_physical_clearance_m),
                    _number(row.bspline.minimum_physical_clearance_m),
                    _number(row.astar.maximum_absolute_heading_change_rad),
                    _number(row.bspline.maximum_absolute_heading_change_rad),
                    _number(row.astar.heading_change_variance_rad2),
                    _number(row.bspline.heading_change_variance_rad2),
                    _number(row.astar.maximum_curvature_inverse_m),
                    _number(row.bspline.maximum_curvature_inverse_m),
                    row.final_source,
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Complete metric set",
            "",
            (
                "| Scene/path | Points | Length [m] | Min clearance [m] | "
                "Mean segment [m] | Max segment [m] | Mean heading [rad] | "
                "Max heading [rad] | Heading variance [rad^2] | Mean "
                "curvature [1/m] | Max curvature [1/m] | Curvature "
                "variance [1/m^2] |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    for row in rows:
        for label, metrics in (("A*", row.astar), ("B-spline", row.bspline)):
            values = (
                f"{row.scene}/{label}",
                str(metrics.point_count),
                _number(metrics.path_length_m),
                _number(metrics.minimum_physical_clearance_m),
                _number(metrics.mean_segment_length_m),
                _number(metrics.maximum_segment_length_m),
                _number(metrics.mean_absolute_heading_change_rad),
                _number(metrics.maximum_absolute_heading_change_rad),
                _number(metrics.heading_change_variance_rad2),
                _number(metrics.mean_curvature_inverse_m),
                _number(metrics.maximum_curvature_inverse_m),
                _number(metrics.curvature_variance_inverse_m2),
            )
            lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Print the deterministic report for review or file comparison."""
    print(render_markdown(run_comparisons()), end="")


if __name__ == "__main__":
    main()
