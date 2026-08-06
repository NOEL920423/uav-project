"""Pure bounded tracker and deterministic Phase 5 state machine."""

import math
from collections.abc import Sequence

from uav_navigation.models import Point3D
from uav_navigation.tracking_models import (
    ReferenceSample,
    SaturationFlags,
    TrackingConfig,
    TrackingErrors,
    TrackingResult,
    TrackingState,
    VALID_TRACKING_FRAME,
    VehicleState,
    VelocityCommand,
)
from uav_navigation.tracking_validator import validate_tracking_command
from uav_navigation.trajectory_models import TrajectoryPoint
from uav_navigation.trajectory_sampler import (
    sample_trajectory,
    validate_sampleable_trajectory,
)


def vector_norm(vector: Point3D) -> float:
    """Return a three-dimensional Euclidean norm."""
    return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)


def wrap_angle(angle: float) -> float:
    """Wrap one angle to the shortest signed interval [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def state_is_finite(state: VehicleState) -> bool:
    """Check every numerical vehicle-state field."""
    return all(math.isfinite(value) for value in (
        state.timestamp_s,
        state.position.x, state.position.y, state.position.z,
        state.velocity.x, state.velocity.y, state.velocity.z,
        state.yaw_ned, state.yaw_rate_radps,
    ))


def tracking_errors(
    reference: ReferenceSample, state: VehicleState
) -> TrackingErrors:
    """Measure NED position/velocity/yaw and path-relative errors."""
    point = reference.point
    north = point.position.x - state.position.x
    east = point.position.y - state.position.y
    down = point.position.z - state.position.z
    velocity = Point3D(
        point.velocity.x - state.velocity.x,
        point.velocity.y - state.velocity.y,
        point.velocity.z - state.velocity.z,
    )
    along = math.cos(point.yaw_ned) * north + math.sin(point.yaw_ned) * east
    cross = -math.sin(point.yaw_ned) * north + math.cos(point.yaw_ned) * east
    return TrackingErrors(
        position_error_m=math.sqrt(north**2 + east**2 + down**2),
        horizontal_position_error_m=math.hypot(north, east),
        vertical_position_error_m=abs(down),
        along_track_error_m=along,
        cross_track_error_m=cross,
        velocity_error_mps=vector_norm(velocity),
        yaw_error_rad=wrap_angle(point.yaw_ned - state.yaw_ned),
    )


def hold_command(timestamp_s: float, reason: str) -> VelocityCommand:
    """Create the exact Phase 5 zero candidate with a specific reason."""
    if not reason:
        raise ValueError("HOLD requires a specific reason")
    return VelocityCommand(
        timestamp_s=float(timestamp_s),
        frame_id=VALID_TRACKING_FRAME,
        linear=Point3D(0.0, 0.0, 0.0),
        yaw_rate_radps=0.0,
        hold_active=True,
        hold_reason=reason,
    )


def _scale_horizontal(vector: Point3D, limit: float) -> tuple[Point3D, bool]:
    magnitude = math.hypot(vector.x, vector.y)
    if magnitude <= limit:
        return vector, False
    scale = limit / magnitude
    return Point3D(vector.x * scale, vector.y * scale, vector.z), True


def _scale_total(vector: Point3D, limit: float) -> tuple[Point3D, bool]:
    magnitude = vector_norm(vector)
    if magnitude <= limit:
        return vector, False
    scale = limit / magnitude
    return Point3D(vector.x * scale, vector.y * scale, vector.z * scale), True


def compute_tracking_command(
    reference: ReferenceSample,
    state: VehicleState,
    timestamp_s: float,
    config: TrackingConfig,
    previous_command: VelocityCommand | None = None,
) -> tuple[VelocityCommand, VelocityCommand, SaturationFlags]:
    """Apply feedforward/feedback and the required ordered command bounds."""
    errors = tracking_errors(reference, state)
    point = reference.point
    position_error = Point3D(
        point.position.x - state.position.x,
        point.position.y - state.position.y,
        point.position.z - state.position.z,
    )
    velocity_error = Point3D(
        point.velocity.x - state.velocity.x,
        point.velocity.y - state.velocity.y,
        point.velocity.z - state.velocity.z,
    )
    linear = Point3D(
        point.velocity.x
        + config.position_kp * position_error.x
        + config.velocity_kd * velocity_error.x,
        point.velocity.y
        + config.position_kp * position_error.y
        + config.velocity_kd * velocity_error.y,
        point.velocity.z
        + config.position_kp * position_error.z
        + config.velocity_kd * velocity_error.z,
    )
    yaw_rate = (
        point.yaw_rate_radps + config.yaw_kp * errors.yaw_error_rad
    )
    unsaturated = VelocityCommand(
        timestamp_s=float(timestamp_s),
        frame_id=VALID_TRACKING_FRAME,
        linear=linear,
        yaw_rate_radps=yaw_rate,
    )
    if not all(math.isfinite(value) for value in (
        linear.x, linear.y, linear.z, yaw_rate, timestamp_s
    )):
        raise ValueError("unbounded tracking command is non-finite")
    linear, horizontal = _scale_horizontal(
        linear, config.maximum_horizontal_command_speed_mps
    )
    vertical = abs(linear.z) > config.maximum_vertical_command_speed_mps
    if vertical:
        linear = Point3D(
            linear.x,
            linear.y,
            math.copysign(
                config.maximum_vertical_command_speed_mps, linear.z
            ),
        )
    linear, total = _scale_total(linear, config.maximum_command_speed_mps)
    acceleration = False
    yaw_acceleration = False
    elapsed = None
    if previous_command is not None:
        elapsed = timestamp_s - previous_command.timestamp_s
        if elapsed > 0.0:
            difference = Point3D(
                linear.x - previous_command.linear.x,
                linear.y - previous_command.linear.y,
                linear.z - previous_command.linear.z,
            )
            maximum_change = config.maximum_command_acceleration_mps2 * elapsed
            difference, acceleration = _scale_total(difference, maximum_change)
            if acceleration:
                linear = Point3D(
                    previous_command.linear.x + difference.x,
                    previous_command.linear.y + difference.y,
                    previous_command.linear.z + difference.z,
                )
    yaw_rate_limited = abs(yaw_rate) > config.maximum_yaw_rate_command_radps
    if yaw_rate_limited:
        yaw_rate = math.copysign(
            config.maximum_yaw_rate_command_radps, yaw_rate
        )
    if previous_command is not None and elapsed is not None and elapsed > 0.0:
        maximum_yaw_change = (
            config.maximum_yaw_acceleration_command_radps2 * elapsed
        )
        change = yaw_rate - previous_command.yaw_rate_radps
        yaw_acceleration = abs(change) > maximum_yaw_change
        if yaw_acceleration:
            yaw_rate = previous_command.yaw_rate_radps + math.copysign(
                maximum_yaw_change, change
            )
    selected = VelocityCommand(
        timestamp_s=float(timestamp_s),
        frame_id=VALID_TRACKING_FRAME,
        linear=linear,
        yaw_rate_radps=yaw_rate,
    )
    return unsaturated, selected, SaturationFlags(
        horizontal_speed=horizontal,
        vertical_speed=vertical,
        total_speed=total,
        acceleration=acceleration,
        yaw_rate=yaw_rate_limited,
        yaw_acceleration=yaw_acceleration,
    )


class OfflineTrackingController:
    """Deterministic receipt-time state machine around the pure tracker."""

    def __init__(self, config: TrackingConfig | None = None) -> None:
        """Initialize empty input history and deterministic counters."""
        self.config = config or TrackingConfig()
        self.trajectory: tuple[TrajectoryPoint, ...] | None = None
        self.trajectory_frame = ""
        self.trajectory_embedded_valid = False
        self.trajectory_receipt_s: float | None = None
        self.tracking_epoch_s: float | None = None
        self.validity: bool | None = None
        self.validity_receipt_s: float | None = None
        self.odometry: VehicleState | None = None
        self.odometry_receipt_s: float | None = None
        self.previous_command: VelocityCommand | None = None
        self.last_step_s: float | None = None
        self.settling_started_s: float | None = None
        self.cycle_index = 0
        self.accepted_trajectory_count = 0
        self._signature = None

    @staticmethod
    def trajectory_signature(
        points: Sequence[TrajectoryPoint], frame_id: str, valid: bool
    ) -> tuple:
        """Create timestamp-independent content identity for deduplication."""
        return frame_id, bool(valid), tuple(points)

    def accept_trajectory(
        self,
        points: Sequence[TrajectoryPoint],
        frame_id: str,
        embedded_valid: bool,
        receipt_time_s: float,
    ) -> bool:
        """Accept a new valid structure and reset only on changed content."""
        timestamp = float(receipt_time_s)
        if not math.isfinite(timestamp):
            raise ValueError("trajectory receipt time must be finite")
        trajectory = validate_sampleable_trajectory(points)
        signature = self.trajectory_signature(
            trajectory, frame_id, embedded_valid
        )
        if signature == self._signature:
            return False
        self._signature = signature
        self.trajectory = trajectory
        self.trajectory_frame = frame_id
        self.trajectory_embedded_valid = bool(embedded_valid)
        self.trajectory_receipt_s = timestamp
        self.tracking_epoch_s = (
            timestamp + self.config.trajectory_start_delay_s
        )
        self.validity = None
        self.validity_receipt_s = None
        self.previous_command = None
        self.settling_started_s = None
        self.accepted_trajectory_count += 1
        return True

    def accept_validity(self, valid: bool, receipt_time_s: float) -> None:
        """Record the independent Phase 4 validity topic receipt."""
        timestamp = float(receipt_time_s)
        if not math.isfinite(timestamp):
            raise ValueError("validity receipt time must be finite")
        self.validity = bool(valid)
        self.validity_receipt_s = timestamp

    def accept_odometry(
        self, state: VehicleState, receipt_time_s: float | None = None
    ) -> None:
        """Record measured state and node-clock receipt independently."""
        receipt = (
            state.timestamp_s if receipt_time_s is None else receipt_time_s
        )
        self.odometry = state
        self.odometry_receipt_s = float(receipt)

    def _reset_after_backward_jump(self) -> None:
        self.trajectory = None
        self.trajectory_frame = ""
        self.trajectory_embedded_valid = False
        self.trajectory_receipt_s = None
        self.tracking_epoch_s = None
        self.validity = None
        self.validity_receipt_s = None
        self.odometry = None
        self.odometry_receipt_s = None
        self.previous_command = None
        self.settling_started_s = None
        self._signature = None

    def _hold(
        self,
        now_s: float,
        state: TrackingState,
        reason: str,
        trajectory_valid: bool = False,
        odometry_valid: bool = False,
        reference: ReferenceSample | None = None,
        errors: TrackingErrors | None = None,
    ) -> TrackingResult:
        command = hold_command(now_s, reason)
        diagnostics = validate_tracking_command(
            command, self.config, state, self.cycle_index
        )
        self.previous_command = command if not diagnostics else None
        trajectory_time = (
            now_s - self.tracking_epoch_s
            if self.tracking_epoch_s is not None else 0.0
        )
        return TrackingResult(
            state=state,
            trajectory_valid=trajectory_valid,
            odometry_valid=odometry_valid,
            command_valid=not diagnostics,
            trajectory_time_s=trajectory_time,
            reference=reference,
            errors=errors,
            unsaturated_command=command,
            selected_command=command,
            diagnostics=diagnostics,
            status_message=reason,
        )

    def step(self, current_time_s: float) -> TrackingResult:
        """Run one gate, state, tracking, bound, and validation cycle."""
        now = float(current_time_s)
        if not math.isfinite(now):
            raise ValueError("control time must be finite")
        self.cycle_index += 1
        if self.last_step_s is not None and now < self.last_step_s:
            self._reset_after_backward_jump()
            self.last_step_s = now
            return self._hold(
                now, TrackingState.HOLD_TIME_JUMP, "ROS time moved backward"
            )
        if self.last_step_s is not None and now == self.last_step_s:
            return self._hold(
                now, TrackingState.HOLD_TIME_JUMP, "ROS time did not advance"
            )
        self.last_step_s = now
        if self.trajectory is None:
            return self._hold(
                now, TrackingState.WAITING_TRAJECTORY, "no trajectory received"
            )
        if (
            self.config.reject_wrong_frame
            and self.trajectory_frame != VALID_TRACKING_FRAME
        ):
            return self._hold(
                now,
                TrackingState.HOLD_INVALID_FRAME,
                f"trajectory frame is {self.trajectory_frame}",
            )
        if not self.trajectory_embedded_valid:
            return self._hold(
                now,
                TrackingState.WAITING_VALIDITY,
                "trajectory reports valid=false",
            )
        if self.config.require_validity_topic and self.validity is None:
            return self._hold(
                now,
                TrackingState.WAITING_VALIDITY,
                "required validity topic not received",
            )
        if self.validity is False:
            return self._hold(
                now,
                TrackingState.WAITING_VALIDITY,
                "trajectory validity topic is false",
            )
        if (
            self.validity_receipt_s is not None
            and now - self.validity_receipt_s
            > self.config.trajectory_validity_timeout_s
        ):
            return self._hold(
                now,
                TrackingState.HOLD_STALE_TRAJECTORY,
                "trajectory validity is stale",
            )
        if self.odometry is None or self.odometry_receipt_s is None:
            return self._hold(
                now,
                TrackingState.WAITING_ODOMETRY,
                "no odometry received",
                trajectory_valid=True,
            )
        if (
            self.config.reject_wrong_frame
            and self.odometry.frame_id != VALID_TRACKING_FRAME
        ):
            return self._hold(
                now,
                TrackingState.HOLD_INVALID_FRAME,
                f"odometry frame is {self.odometry.frame_id}",
                trajectory_valid=True,
            )
        if not state_is_finite(self.odometry):
            return self._hold(
                now,
                TrackingState.HOLD_INVALID_COMMAND,
                "odometry contains non-finite state",
                trajectory_valid=True,
            )
        if now - self.odometry_receipt_s > self.config.odometry_timeout_s:
            return self._hold(
                now,
                TrackingState.HOLD_STALE_ODOMETRY,
                "odometry is stale",
                trajectory_valid=True,
            )
        if self.tracking_epoch_s is None:
            return self._hold(
                now,
                TrackingState.WAITING_TRAJECTORY,
                "trajectory epoch is unavailable",
            )
        trajectory_time = now - self.tracking_epoch_s
        reference = sample_trajectory(self.trajectory, trajectory_time)
        if reference.prestart:
            return self._hold(
                now,
                TrackingState.PRESTART_HOLD,
                "tracking epoch has not started",
                trajectory_valid=True,
                odometry_valid=True,
                reference=reference,
            )
        errors = tracking_errors(reference, self.odometry)
        if errors.position_error_m > self.config.maximum_tracking_error_m:
            return self._hold(
                now,
                TrackingState.HOLD_TRACKING_ERROR,
                "tracking error exceeds configured maximum",
                trajectory_valid=True,
                odometry_valid=True,
                reference=reference,
                errors=errors,
            )
        state = TrackingState.TRACKING
        if reference.terminal:
            terminal_elapsed = (
                trajectory_time - self.trajectory[-1].time_from_start_s
            )
            within_goal = (
                errors.position_error_m
                <= self.config.goal_position_tolerance_m
                and vector_norm(self.odometry.velocity)
                <= self.config.goal_velocity_tolerance_mps
                and abs(errors.yaw_error_rad)
                <= self.config.goal_yaw_tolerance_rad
            )
            if within_goal:
                if self.settling_started_s is None:
                    self.settling_started_s = now
                if (
                    now - self.settling_started_s
                    >= self.config.goal_settle_time_s
                ):
                    return self._hold(
                        now,
                        TrackingState.GOAL_HOLD,
                        "terminal goal settled",
                        trajectory_valid=True,
                        odometry_valid=True,
                        reference=reference,
                        errors=errors,
                    )
                state = TrackingState.GOAL_SETTLING
            else:
                self.settling_started_s = None
            if terminal_elapsed > self.config.maximum_terminal_wait_s:
                return self._hold(
                    now,
                    TrackingState.TERMINAL_NOT_REACHED,
                    "terminal state was not reached before timeout",
                    trajectory_valid=True,
                    odometry_valid=True,
                    reference=reference,
                    errors=errors,
                )
        else:
            self.settling_started_s = None
        try:
            unsaturated, selected, saturations = compute_tracking_command(
                reference,
                self.odometry,
                now,
                self.config,
                self.previous_command,
            )
        except (ValueError, OverflowError) as error:
            return self._hold(
                now,
                TrackingState.HOLD_INVALID_COMMAND,
                f"command generation failed: {error}",
                trajectory_valid=True,
                odometry_valid=True,
                reference=reference,
                errors=errors,
            )
        diagnostics = validate_tracking_command(
            selected,
            self.config,
            state,
            self.cycle_index,
            self.previous_command,
        )
        if diagnostics:
            hold = hold_command(
                now, f"independent command validation failed: "
                f"{diagnostics[0].constraint}"
            )
            hold_diagnostics = validate_tracking_command(
                hold,
                self.config,
                TrackingState.HOLD_INVALID_COMMAND,
                self.cycle_index,
            )
            if hold_diagnostics:
                raise ValueError(
                    "internally constructed HOLD command failed validation: "
                    f"{hold_diagnostics[0].constraint}"
                )
            self.previous_command = hold
            return TrackingResult(
                state=TrackingState.HOLD_INVALID_COMMAND,
                trajectory_valid=True,
                odometry_valid=True,
                command_valid=False,
                trajectory_time_s=trajectory_time,
                reference=reference,
                errors=errors,
                unsaturated_command=unsaturated,
                selected_command=hold,
                saturations=saturations,
                diagnostics=diagnostics,
                status_message=hold.hold_reason,
            )
        self.previous_command = selected
        return TrackingResult(
            state=state,
            trajectory_valid=True,
            odometry_valid=True,
            command_valid=True,
            trajectory_time_s=trajectory_time,
            reference=reference,
            errors=errors,
            unsaturated_command=unsaturated,
            selected_command=selected,
            saturations=saturations,
            status_message="bounded candidate command validated",
        )
