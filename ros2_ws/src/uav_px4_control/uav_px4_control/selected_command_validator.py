"""Independent validation of final Phase 6 mux-selected commands."""

import math

from uav_px4_control.control_source_models import (
    ACTIVE_STATES,
    CONTROL_SOURCES,
    ControlCommand,
    ControlMuxConfig,
    ControlMuxState,
    HOLD,
    SelectedCommandDiagnostic,
    VALID_COMMAND_FRAME,
    command_speed,
)


TOLERANCE = 1e-8


def _diagnostic(
    constraint: str,
    command: ControlCommand,
    measured: float,
    limit: float,
    cycle_index: int,
    reason: str,
) -> SelectedCommandDiagnostic:
    return SelectedCommandDiagnostic(
        constraint=constraint,
        source=command.source,
        measured_value=measured,
        limit_value=limit,
        cycle_index=cycle_index,
        timestamp_s=command.timestamp_s,
        reason=reason,
    )


def validate_selected_command(
    command: ControlCommand,
    config: ControlMuxConfig,
    state: ControlMuxState,
    active_source: str,
    cycle_index: int,
    previous_command: ControlCommand | None = None,
) -> tuple[SelectedCommandDiagnostic, ...]:
    """Validate output independently of registry and mux limiting code."""
    diagnostics: list[SelectedCommandDiagnostic] = []

    def check(
        constraint: str, measured: float, limit: float, reason: str
    ) -> None:
        if measured > limit + TOLERANCE:
            diagnostics.append(_diagnostic(
                constraint, command, measured, limit, cycle_index, reason
            ))

    if command.source not in CONTROL_SOURCES:
        diagnostics.append(_diagnostic(
            "allowed_source", command, 1.0, 0.0, cycle_index,
            "selected source identifier is not canonical",
        ))
    if command.source != active_source:
        diagnostics.append(_diagnostic(
            "active_source_consistency", command, 1.0, 0.0,
            cycle_index, "selected command does not belong to active source",
        ))
    if (
        config.reject_wrong_frame
        and command.frame_id != VALID_COMMAND_FRAME
    ):
        diagnostics.append(_diagnostic(
            "selected_frame", command, 1.0, 0.0, cycle_index,
            "selected command frame is not px4_ned",
        ))
    values = (
        command.timestamp_s,
        command.linear.x,
        command.linear.y,
        command.linear.z,
        command.angular_x,
        command.angular_y,
        command.yaw_rate_radps,
    )
    if not all(math.isfinite(value) for value in values):
        diagnostics.append(_diagnostic(
            "finite_selected_command", command, math.inf, 0.0,
            cycle_index, "selected command contains non-finite fields",
        ))
        return tuple(diagnostics)
    angular_xy = max(abs(command.angular_x), abs(command.angular_y))
    check(
        "selected_angular_xy", angular_xy, config.hold_command_epsilon,
        "selected angular x/y must be zero",
    )
    horizontal = math.hypot(command.linear.x, command.linear.y)
    speed = command_speed(command)
    check(
        "selected_horizontal_speed", horizontal,
        config.maximum_selected_horizontal_speed_mps,
        "selected horizontal speed exceeds limit",
    )
    check(
        "selected_vertical_speed", abs(command.linear.z),
        config.maximum_selected_vertical_speed_mps,
        "selected vertical speed exceeds limit",
    )
    check(
        "selected_total_speed", speed,
        config.maximum_selected_speed_mps,
        "selected total speed exceeds limit",
    )
    check(
        "selected_yaw_rate", abs(command.yaw_rate_radps),
        config.maximum_selected_yaw_rate_radps,
        "selected yaw rate exceeds limit",
    )
    if previous_command is not None:
        elapsed = command.timestamp_s - previous_command.timestamp_s
        if elapsed <= 0.0:
            diagnostics.append(_diagnostic(
                "selected_timestamp_monotonicity", command, elapsed, 0.0,
                cycle_index, "selected output timestamp did not advance",
            ))
        elif not command.hold_active:
            acceleration = math.sqrt(
                (command.linear.x - previous_command.linear.x) ** 2
                + (command.linear.y - previous_command.linear.y) ** 2
                + (command.linear.z - previous_command.linear.z) ** 2
            ) / elapsed
            yaw_acceleration = abs(
                command.yaw_rate_radps - previous_command.yaw_rate_radps
            ) / elapsed
            check(
                "selected_acceleration", acceleration,
                config.maximum_selected_acceleration_mps2,
                "selected acceleration exceeds limit",
            )
            check(
                "selected_yaw_acceleration", yaw_acceleration,
                config.maximum_selected_yaw_acceleration_radps2,
                "selected yaw acceleration exceeds limit",
            )
    if command.hold_active:
        check(
            "selected_hold_magnitude",
            max(speed, abs(command.yaw_rate_radps)),
            config.hold_command_epsilon,
            "selected HOLD is not exact zero",
        )
        if not command.hold_reason:
            diagnostics.append(_diagnostic(
                "selected_hold_reason", command, 1.0, 0.0, cycle_index,
                "selected HOLD lacks a reason",
            ))
        if active_source != HOLD or state in ACTIVE_STATES.values():
            diagnostics.append(_diagnostic(
                "selected_hold_state", command, 1.0, 0.0, cycle_index,
                "HOLD command is inconsistent with active movement state",
            ))
    elif active_source == HOLD or state not in ACTIVE_STATES.values():
        diagnostics.append(_diagnostic(
            "selected_movement_state", command, 1.0, 0.0, cycle_index,
            "movement command is inconsistent with HOLD state",
        ))
    return tuple(diagnostics)
