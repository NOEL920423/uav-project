"""Pure identity mapping from selected px4_ned command to PX4 candidate."""

import math

from uav_px4_control.control_source_models import ControlCommand
from uav_px4_control.px4_boundary_models import (
    PX4_NED_FRAME,
    Px4VelocitySetpointCandidate,
)


UNUSED_VECTOR = (math.nan, math.nan, math.nan)


def map_selected_command(
    command: ControlCommand,
    timestamp_us: int,
    selected_receipt_time_s: float,
) -> Px4VelocitySetpointCandidate:
    """Map NED components without coordinate conversion or yaw integration."""
    reason = ""
    valid = True
    if command.frame_id != PX4_NED_FRAME:
        valid = False
        reason = "selected command frame must be px4_ned"
    elif command.angular_x != 0.0 or command.angular_y != 0.0:
        valid = False
        reason = "selected angular x/y must be zero"
    return Px4VelocitySetpointCandidate(
        timestamp_us=timestamp_us,
        position_ned_m=UNUSED_VECTOR,
        velocity_ned_mps=(
            command.linear.x,
            command.linear.y,
            command.linear.z,
        ),
        acceleration_ned_mps2=UNUSED_VECTOR,
        jerk_ned_mps3=UNUSED_VECTOR,
        yaw_ned_rad=math.nan,
        yaw_rate_ned_radps=command.yaw_rate_radps,
        source=command.source,
        frame_id=command.frame_id,
        selected_receipt_time_s=float(selected_receipt_time_s),
        valid=valid,
        rejection_reason=reason,
    )
