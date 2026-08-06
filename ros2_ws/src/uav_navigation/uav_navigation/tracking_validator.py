"""Independent validator for offline trajectory-follower commands."""

import math

from uav_navigation.tracking_models import (
    TrackingConfig,
    TrackingDiagnostic,
    TrackingState,
    VALID_TRACKING_FRAME,
    VelocityCommand,
)

TOLERANCE = 1e-8


def _norm(x: float, y: float, z: float = 0.0) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _diagnostic(
    constraint: str,
    measured: float,
    limit: float,
    command: VelocityCommand,
    cycle_index: int,
) -> TrackingDiagnostic:
    return TrackingDiagnostic(
        constraint=constraint,
        measured_value=measured,
        limit_value=limit,
        timestamp_s=command.timestamp_s,
        cycle_index=cycle_index,
        message=(
            f"{constraint}: measured={measured:.9g}, limit={limit:.9g}, "
            f"timestamp={command.timestamp_s:.9g}, cycle={cycle_index}"
        ),
    )


def validate_tracking_command(
    command: VelocityCommand,
    config: TrackingConfig,
    state: TrackingState,
    cycle_index: int,
    previous_command: VelocityCommand | None = None,
    trajectory_fresh: bool = True,
    odometry_fresh: bool = True,
) -> tuple[TrackingDiagnostic, ...]:
    """Validate selected command without trusting the tracker bounds."""
    diagnostics: list[TrackingDiagnostic] = []

    def check(name: str, measured: float, limit: float) -> None:
        if measured > limit + TOLERANCE:
            diagnostics.append(
                _diagnostic(name, measured, limit, command, cycle_index)
            )

    if command.frame_id != VALID_TRACKING_FRAME:
        diagnostics.append(
            _diagnostic("command_frame", 1.0, 0.0, command, cycle_index)
        )
    values = (
        command.timestamp_s,
        command.linear.x,
        command.linear.y,
        command.linear.z,
        command.yaw_rate_radps,
    )
    if not all(math.isfinite(value) for value in values):
        diagnostics.append(
            _diagnostic("finite_command", math.inf, 0.0, command, cycle_index)
        )
        return tuple(diagnostics)
    horizontal = _norm(command.linear.x, command.linear.y)
    speed = _norm(command.linear.x, command.linear.y, command.linear.z)
    check("horizontal_speed", horizontal,
          config.maximum_horizontal_command_speed_mps)
    check("vertical_speed", abs(command.linear.z),
          config.maximum_vertical_command_speed_mps)
    check("total_speed", speed, config.maximum_command_speed_mps)
    check("yaw_rate", abs(command.yaw_rate_radps),
          config.maximum_yaw_rate_command_radps)
    if previous_command is not None:
        elapsed = command.timestamp_s - previous_command.timestamp_s
        if elapsed <= 0.0:
            diagnostics.append(
                _diagnostic(
                    "timestamp_monotonicity",
                    elapsed,
                    0.0,
                    command,
                    cycle_index,
                )
            )
        else:
            acceleration = _norm(
                command.linear.x - previous_command.linear.x,
                command.linear.y - previous_command.linear.y,
                command.linear.z - previous_command.linear.z,
            ) / elapsed
            yaw_acceleration = abs(
                command.yaw_rate_radps - previous_command.yaw_rate_radps
            ) / elapsed
            check(
                "command_acceleration",
                acceleration,
                config.maximum_command_acceleration_mps2,
            )
            check(
                "yaw_acceleration",
                yaw_acceleration,
                config.maximum_yaw_acceleration_command_radps2,
            )
    if not trajectory_fresh:
        diagnostics.append(
            _diagnostic("stale_trajectory", 1.0, 0.0, command, cycle_index)
        )
    if not odometry_fresh:
        diagnostics.append(
            _diagnostic("stale_odometry", 1.0, 0.0, command, cycle_index)
        )
    normal_states = {TrackingState.TRACKING, TrackingState.GOAL_SETTLING}
    if command.hold_active:
        hold_magnitude = max(speed, abs(command.yaw_rate_radps))
        check("hold_magnitude", hold_magnitude, config.hold_command_epsilon)
        if not command.hold_reason:
            diagnostics.append(
                _diagnostic("hold_reason", 1.0, 0.0, command, cycle_index)
            )
        if state in normal_states:
            diagnostics.append(
                _diagnostic(
                    "state_machine_consistency",
                    1.0,
                    0.0,
                    command,
                    cycle_index,
                )
            )
    elif state not in normal_states:
        diagnostics.append(
            _diagnostic(
                "state_machine_consistency", 1.0, 0.0, command, cycle_index
            )
        )
    return tuple(diagnostics)
