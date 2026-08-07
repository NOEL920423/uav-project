"""Deterministic non-PX4 telemetry fixtures for Phase 7 tests."""

from uav_px4_control.px4_boundary_models import Px4TelemetryState


SYNTHETIC_TELEMETRY_FIXTURES = (
    "no_telemetry",
    "fresh_healthy",
    "stale_telemetry",
    "failsafe_active",
    "invalid_odometry_state",
    "unknown_navigation_state",
    "offboard_inactive",
    "offboard_active",
    "disarmed",
    "armed",
    "telemetry_time_jump",
    "recovery_after_latched_fault",
)


def synthetic_telemetry_fixture(
    name: str,
    current_time_s: float,
    timestamp_us: int,
) -> Px4TelemetryState | None:
    """Construct one named architecture-level telemetry fixture."""
    if name not in SYNTHETIC_TELEMETRY_FIXTURES:
        raise ValueError(f"unknown synthetic telemetry fixture: {name}")
    if name == "no_telemetry":
        return None
    changes: dict[str, object] = {}
    receipt_time_s = current_time_s
    if name == "stale_telemetry":
        receipt_time_s -= 1.0
    elif name == "failsafe_active":
        changes["failsafe"] = True
    elif name == "invalid_odometry_state":
        changes["odometry_valid"] = False
        changes["local_velocity_valid"] = False
    elif name == "unknown_navigation_state":
        changes["nav_state"] = 255
    elif name == "offboard_active":
        changes["offboard_active"] = True
        changes["nav_state"] = 14
    elif name == "armed":
        changes["arming_state"] = 2
    elif name == "telemetry_time_jump":
        timestamp_us -= 1
    return Px4TelemetryState(
        receipt_time_s=receipt_time_s,
        timestamp_us=timestamp_us,
        **changes,
    )
