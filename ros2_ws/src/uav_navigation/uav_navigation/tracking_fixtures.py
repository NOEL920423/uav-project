"""Canonical deterministic fixture vocabulary for Phase 5 offline tracking."""

from dataclasses import dataclass

from uav_navigation.models import Point3D


@dataclass(frozen=True, slots=True)
class TrackingFixture:
    """One pure fixture definition and its expected terminal category."""

    name: str
    path: tuple[Point3D, ...]
    expected: str
    initial_position: Point3D = Point3D(0.0, 0.0, -2.0)
    disturbance_velocity: Point3D = Point3D(0.0, 0.0, 0.0)
    validity_mode: str = "heartbeat"
    odometry_mode: str = "normal"


def _path(*coordinates) -> tuple[Point3D, ...]:
    return tuple(
        Point3D(north, east, down) for north, east, down in coordinates
    )


TRACKING_FIXTURES = {
    "straight-trajectory": TrackingFixture(
        "straight-trajectory",
        _path((0, 0, -2), (1, 0, -2), (2, 0, -2)),
        "SUCCESS",
    ),
    "phase3-bspline-accepted": TrackingFixture(
        "phase3-bspline-accepted",
        _path(
            (0, 0, -2), (0.5, 0.05, -2), (1.0, 0.15, -2),
            (1.5, 0.05, -2), (2.0, 0.0, -2),
        ),
        "SUCCESS",
    ),
    "astar-fallback": TrackingFixture(
        "astar-fallback",
        _path((0, 0, -2), (0.8, 0, -2), (0.8, 0.5, -2),
              (1.5, 0.5, -2)),
        "SUCCESS",
    ),
    "sharp-dynamically-valid": TrackingFixture(
        "sharp-dynamically-valid",
        _path((0, 0, -2), (0.5, 0, -2), (0.5, 0.2, -2),
              (0.8, 0.2, -2)),
        "SUCCESS",
    ),
    "start-position-offset": TrackingFixture(
        "start-position-offset",
        _path((0, 0, -2), (1, 0, -2), (2, 0, -2)),
        "SUCCESS",
        initial_position=Point3D(-0.4, 0.2, -2.0),
    ),
    "constant-horizontal-disturbance": TrackingFixture(
        "constant-horizontal-disturbance",
        _path((0, 0, -2), (1, 0, -2), (2, 0, -2)),
        "SUCCESS",
        disturbance_velocity=Point3D(0.0, 0.08, 0.0),
    ),
    "duplicate-trajectory": TrackingFixture(
        "duplicate-trajectory",
        _path((0, 0, -2), (1, 0, -2), (2, 0, -2)),
        "SUCCESS",
        validity_mode="duplicate",
    ),
    "stale-odometry": TrackingFixture(
        "stale-odometry",
        _path((0, 0, -2), (1, 0, -2), (2, 0, -2)),
        "HOLD_STALE_ODOMETRY",
        odometry_mode="stale",
    ),
    "stale-trajectory-validity": TrackingFixture(
        "stale-trajectory-validity",
        _path((0, 0, -2), (1, 0, -2), (2, 0, -2)),
        "HOLD_STALE_TRAJECTORY",
        validity_mode="stale",
    ),
    "invalid-validity-flag": TrackingFixture(
        "invalid-validity-flag",
        _path((0, 0, -2), (1, 0, -2)),
        "WAITING_VALIDITY",
        validity_mode="false",
    ),
    "wrong-odometry-frame": TrackingFixture(
        "wrong-odometry-frame",
        _path((0, 0, -2), (1, 0, -2)),
        "HOLD_INVALID_FRAME",
        odometry_mode="wrong-frame",
    ),
    "nonfinite-odometry": TrackingFixture(
        "nonfinite-odometry",
        _path((0, 0, -2), (1, 0, -2)),
        "HOLD_INVALID_COMMAND",
        odometry_mode="nonfinite",
    ),
    "backward-time-jump": TrackingFixture(
        "backward-time-jump",
        _path((0, 0, -2), (1, 0, -2)),
        "HOLD_TIME_JUMP",
    ),
    "command-speed-saturation": TrackingFixture(
        "command-speed-saturation",
        _path((0, 0, -2), (1, 0, -2), (2, 0, -2)),
        "SUCCESS_WITH_SATURATION",
        initial_position=Point3D(-1.5, 0.0, -2.0),
    ),
    "command-acceleration-saturation": TrackingFixture(
        "command-acceleration-saturation",
        _path((0, 0, -2), (1, 0, -2), (2, 0, -2)),
        "SUCCESS_WITH_SATURATION",
        initial_position=Point3D(-0.8, 0.0, -2.0),
    ),
    "excessive-tracking-error": TrackingFixture(
        "excessive-tracking-error",
        _path((0, 0, -2), (1, 0, -2)),
        "HOLD_TRACKING_ERROR",
        initial_position=Point3D(-3.0, 0.0, -2.0),
    ),
    "successful-goal-settling": TrackingFixture(
        "successful-goal-settling",
        _path((0, 0, -2), (1, 0, -2), (2, 0, -2)),
        "SUCCESS",
    ),
    "terminal-not-reached": TrackingFixture(
        "terminal-not-reached",
        _path((0, 0, -2), (1, 0, -2)),
        "TERMINAL_NOT_REACHED",
        initial_position=Point3D(0.5, 0.0, -2.0),
        odometry_mode="frozen",
    ),
    "yaw-wrap-crossing": TrackingFixture(
        "yaw-wrap-crossing",
        _path((0, 0.01, -2), (-1, 0.001, -2),
              (-2, -0.001, -2), (-3, -0.01, -2)),
        "SUCCESS",
    ),
    "invalid-command-rejection": TrackingFixture(
        "invalid-command-rejection",
        _path((0, 0, -2), (1, 0, -2)),
        "PURE_VALIDATOR_REJECTION",
    ),
}


def tracking_fixture(name: str) -> TrackingFixture:
    """Resolve one named fixture or reject unknown automation input."""
    if name not in TRACKING_FIXTURES:
        raise ValueError(f"unknown tracking fixture: {name}")
    return TRACKING_FIXTURES[name]
