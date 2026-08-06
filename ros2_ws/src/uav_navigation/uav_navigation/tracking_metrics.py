"""Independent accumulator for Offline Closed-Loop Tracking Metrics."""

import math

from uav_navigation.tracking_models import (
    OfflineTrackingMetrics,
    SaturationFlags,
    TrackingErrors,
    TrackingState,
    VelocityCommand,
)


class TrackingMetricsAccumulator:
    """Accumulate errors and command derivatives without ROS dependencies."""

    def __init__(self) -> None:
        """Initialize independent sums, maxima, and command history."""
        self._cycles = 0
        self._error_cycles = 0
        self._position_square = 0.0
        self._horizontal_square = 0.0
        self._vertical_square = 0.0
        self._velocity_square = 0.0
        self._yaw_square = 0.0
        self._max_position = 0.0
        self._terminal_position = 0.0
        self._max_yaw = 0.0
        self._max_speed = 0.0
        self._max_acceleration = 0.0
        self._max_yaw_rate = 0.0
        self._max_yaw_acceleration = 0.0
        self._saturations = 0
        self._holds = 0
        self._stale_latency = 0.0
        self._settling_time = 0.0
        self._completion = "INCOMPLETE"
        self._previous: VelocityCommand | None = None

    @staticmethod
    def _speed(command: VelocityCommand) -> float:
        vector = command.linear
        return math.sqrt(vector.x**2 + vector.y**2 + vector.z**2)

    def update(
        self,
        errors: TrackingErrors | None,
        command: VelocityCommand,
        saturations: SaturationFlags,
        state: TrackingState,
    ) -> None:
        """Consume one independently selected follower cycle."""
        self._cycles += 1
        if errors is not None:
            self._error_cycles += 1
            self._position_square += errors.position_error_m**2
            self._horizontal_square += errors.horizontal_position_error_m**2
            self._vertical_square += errors.vertical_position_error_m**2
            self._velocity_square += errors.velocity_error_mps**2
            self._yaw_square += errors.yaw_error_rad**2
            self._max_position = max(
                self._max_position, errors.position_error_m
            )
            self._max_yaw = max(self._max_yaw, abs(errors.yaw_error_rad))
            if state in {
                TrackingState.GOAL_SETTLING,
                TrackingState.GOAL_HOLD,
                TrackingState.TERMINAL_NOT_REACHED,
            }:
                self._terminal_position = errors.position_error_m
        speed = self._speed(command)
        self._max_speed = max(self._max_speed, speed)
        self._max_yaw_rate = max(
            self._max_yaw_rate, abs(command.yaw_rate_radps)
        )
        # An exact fail-closed HOLD is a safety override, not a dynamically
        # rate-limited tracking command. Keep counting HOLD cycles, but report
        # derivative maxima only for ordinary controller candidates.
        if self._previous is not None and not command.hold_active:
            elapsed = command.timestamp_s - self._previous.timestamp_s
            if elapsed > 0.0:
                delta = math.sqrt(
                    (command.linear.x - self._previous.linear.x) ** 2
                    + (command.linear.y - self._previous.linear.y) ** 2
                    + (command.linear.z - self._previous.linear.z) ** 2
                )
                self._max_acceleration = max(
                    self._max_acceleration, delta / elapsed
                )
                self._max_yaw_acceleration = max(
                    self._max_yaw_acceleration,
                    abs(
                        command.yaw_rate_radps
                        - self._previous.yaw_rate_radps
                    ) / elapsed,
                )
        self._saturations += saturations.count
        self._holds += int(command.hold_active)
        if state == TrackingState.GOAL_HOLD:
            self._completion = "GOAL_HOLD"
        elif state == TrackingState.TERMINAL_NOT_REACHED:
            self._completion = "TERMINAL_NOT_REACHED"
        self._previous = command

    def record_stale_detection_latency(self, latency_s: float) -> None:
        """Record the measured stale-gate response latency."""
        if not math.isfinite(latency_s) or latency_s < 0.0:
            raise ValueError(
                "stale detection latency must be finite and nonnegative"
            )
        self._stale_latency = float(latency_s)

    def record_terminal_settling_time(self, settling_time_s: float) -> None:
        """Record time continuously inside all terminal tolerances."""
        if not math.isfinite(settling_time_s) or settling_time_s < 0.0:
            raise ValueError("settling time must be finite and nonnegative")
        self._settling_time = float(settling_time_s)

    def result(self) -> OfflineTrackingMetrics:
        """Return immutable aggregate metrics."""
        count = max(1, self._error_cycles)
        return OfflineTrackingMetrics(
            cycle_count=self._cycles,
            position_rmse_m=math.sqrt(self._position_square / count),
            horizontal_position_rmse_m=math.sqrt(
                self._horizontal_square / count
            ),
            vertical_position_rmse_m=math.sqrt(
                self._vertical_square / count
            ),
            maximum_position_error_m=self._max_position,
            terminal_position_error_m=self._terminal_position,
            velocity_rmse_mps=math.sqrt(self._velocity_square / count),
            yaw_rmse_rad=math.sqrt(self._yaw_square / count),
            maximum_yaw_error_rad=self._max_yaw,
            maximum_command_speed_mps=self._max_speed,
            maximum_command_acceleration_mps2=self._max_acceleration,
            maximum_yaw_rate_radps=self._max_yaw_rate,
            maximum_yaw_acceleration_radps2=self._max_yaw_acceleration,
            saturation_count=self._saturations,
            hold_cycle_count=self._holds,
            stale_detection_latency_s=self._stale_latency,
            terminal_settling_time_s=self._settling_time,
            completion_status=self._completion,
        )
