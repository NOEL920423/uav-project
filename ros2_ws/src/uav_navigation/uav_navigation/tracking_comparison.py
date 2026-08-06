"""Deterministic pure comparison runner for Phase 5 tracking fixtures."""

from dataclasses import dataclass

from uav_navigation.offline_kinematic_plant import (
    KinematicPlantConfig,
    OfflineKinematicPlant,
)
from uav_navigation.tracking_fixtures import tracking_fixture
from uav_navigation.tracking_metrics import TrackingMetricsAccumulator
from uav_navigation.tracking_models import (
    OfflineTrackingMetrics,
    TrackingConfig,
    TrackingState,
)
from uav_navigation.trajectory_parameterizer import parameterize_trajectory
from uav_navigation.trajectory_tracker import OfflineTrackingController

COMPARISON_FIXTURES = (
    "straight-trajectory",
    "phase3-bspline-accepted",
    "astar-fallback",
    "sharp-dynamically-valid",
    "start-position-offset",
    "constant-horizontal-disturbance",
    "command-speed-saturation",
    "yaw-wrap-crossing",
)


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """One independently accumulated pure closed-loop comparison result."""

    fixture: str
    trajectory_points: int
    trajectory_duration_s: float
    simulated_duration_s: float
    terminal_state: str
    metrics: OfflineTrackingMetrics


def run_comparison_fixture(name: str) -> ComparisonResult:
    """Run one fixture through controller and fixed-step kinematic plant."""
    fixture = tracking_fixture(name)
    trajectory = parameterize_trajectory(fixture.path)
    if not trajectory.valid:
        raise RuntimeError(trajectory.rejection_reason)
    config = TrackingConfig(
        trajectory_validity_timeout_s=30.0,
        odometry_timeout_s=1.0,
    )
    controller = OfflineTrackingController(config)
    controller.accept_trajectory(
        trajectory.trajectory_points, "px4_ned", True, 0.0
    )
    controller.accept_validity(True, 0.0)
    plant = OfflineKinematicPlant(KinematicPlantConfig(
        integration_timestep_s=config.control_period_s,
        initial_position=fixture.initial_position,
        disturbance_velocity=fixture.disturbance_velocity,
    ))
    accumulator = TrackingMetricsAccumulator()
    now = 0.0
    result = None
    maximum_time = trajectory.total_duration_s + 4.0
    while now <= maximum_time:
        controller.accept_validity(True, now)
        controller.accept_odometry(plant.measurement(now), now)
        result = controller.step(now)
        accumulator.update(
            result.errors,
            result.selected_command,
            result.saturations,
            result.state,
        )
        plant.step(result.selected_command)
        if result.state in {
            TrackingState.GOAL_HOLD,
            TrackingState.TERMINAL_NOT_REACHED,
            TrackingState.HOLD_TRACKING_ERROR,
            TrackingState.HOLD_INVALID_COMMAND,
        }:
            break
        now = round(now + config.control_period_s, 10)
    if result is None:
        raise RuntimeError("comparison produced no controller cycles")
    if result.state == TrackingState.GOAL_HOLD:
        accumulator.record_terminal_settling_time(config.goal_settle_time_s)
    return ComparisonResult(
        fixture=name,
        trajectory_points=len(trajectory.trajectory_points),
        trajectory_duration_s=trajectory.total_duration_s,
        simulated_duration_s=now,
        terminal_state=result.state.value,
        metrics=accumulator.result(),
    )


def main() -> int:
    """Print a Markdown-ready eight-fixture quantitative comparison."""
    print(
        "fixture|points|trajectory_s|simulation_s|position_rmse_m|"
        "max_error_m|terminal_error_m|velocity_rmse_mps|yaw_rmse_rad|"
        "max_cmd_mps|max_cmd_accel_mps2|max_yaw_rate_radps|"
        "max_yaw_accel_radps2|saturations|holds|settling_s|state"
    )
    for name in COMPARISON_FIXTURES:
        result = run_comparison_fixture(name)
        metrics = result.metrics
        print(
            f"{name}|{result.trajectory_points}|"
            f"{result.trajectory_duration_s:.6f}|"
            f"{result.simulated_duration_s:.6f}|"
            f"{metrics.position_rmse_m:.6f}|"
            f"{metrics.maximum_position_error_m:.6f}|"
            f"{metrics.terminal_position_error_m:.6f}|"
            f"{metrics.velocity_rmse_mps:.6f}|"
            f"{metrics.yaw_rmse_rad:.6f}|"
            f"{metrics.maximum_command_speed_mps:.6f}|"
            f"{metrics.maximum_command_acceleration_mps2:.6f}|"
            f"{metrics.maximum_yaw_rate_radps:.6f}|"
            f"{metrics.maximum_yaw_acceleration_radps2:.6f}|"
            f"{metrics.saturation_count}|{metrics.hold_cycle_count}|"
            f"{metrics.terminal_settling_time_s:.6f}|"
            f"{result.terminal_state}"
        )
    return 0


if __name__ == "__main__":
    main()
