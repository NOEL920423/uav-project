"""Regression tests for deterministic plant response and tracking metrics."""

import pytest

from uav_navigation.models import Point3D
from uav_navigation.offline_kinematic_plant import (
    KinematicPlantConfig,
    OfflineKinematicPlant,
)
from uav_navigation.tracking_metrics import TrackingMetricsAccumulator
from uav_navigation.tracking_models import (
    SaturationFlags,
    TrackingErrors,
    TrackingState,
    VelocityCommand,
)


def _command(time_s, north=1.0, yaw_rate=0.5, hold=False):
    """Build one finite candidate for plant and metric tests."""
    return VelocityCommand(
        time_s,
        "px4_ned",
        Point3D(north, 0.0, 0.0),
        yaw_rate,
        hold_active=hold,
        hold_reason="fixture hold" if hold else "",
    )


def test_fixed_step_first_order_response_is_deterministic_and_limited():
    """Identical plants remain bitwise equal and obey acceleration bounds."""
    config = KinematicPlantConfig(
        integration_timestep_s=0.1,
        velocity_time_constant_s=0.2,
        maximum_simulated_acceleration_mps2=1.0,
    )
    first = OfflineKinematicPlant(config)
    second = OfflineKinematicPlant(config)
    first_state = first.step(_command(0.0))
    second_state = second.step(_command(0.0))
    assert first_state == second_state
    assert first_state.velocity.x == pytest.approx(0.1)
    assert first_state.position.x == pytest.approx(0.01)
    assert first.measurement() == second.measurement()


def test_zero_command_keeps_default_plant_spatially_stationary():
    """An exact zero command preserves default position, velocity, and yaw."""
    plant = OfflineKinematicPlant()
    initial = plant.state
    state = plant.step(_command(0.0, north=0.0, yaw_rate=0.0, hold=True))
    assert state.position == initial.position
    assert state.velocity == initial.velocity
    assert state.yaw_ned == initial.yaw_ned
    assert state.yaw_rate_radps == initial.yaw_rate_radps


def test_disturbance_and_optional_noise_are_deterministic_and_default_off():
    """Constant disturbance moves position; analytic noise needs opt-in."""
    disturbed = OfflineKinematicPlant(KinematicPlantConfig(
        integration_timestep_s=0.1,
        disturbance_velocity=Point3D(0.0, 0.2, 0.0),
    ))
    disturbed.step(_command(0.0, north=0.0, yaw_rate=0.0))
    assert disturbed.state.position.y == pytest.approx(0.02)
    assert disturbed.measurement().position == disturbed.state.position
    noisy_config = KinematicPlantConfig(measurement_noise_amplitude_m=0.01)
    noisy_first = OfflineKinematicPlant(noisy_config)
    noisy_second = OfflineKinematicPlant(noisy_config)
    noisy_first.step(_command(0.0))
    noisy_second.step(_command(0.0))
    assert noisy_first.measurement() == noisy_second.measurement()
    assert noisy_first.measurement().position != noisy_first.state.position


def test_metrics_compute_rmse_rates_saturations_holds_and_completion():
    """Accumulator independently measures every required metric category."""
    metrics = TrackingMetricsAccumulator()
    first_error = TrackingErrors(1.0, 0.8, 0.6, 0.5, 0.2, 0.4, 0.3)
    second_error = TrackingErrors(2.0, 1.6, 1.2, 1.0, 0.4, 0.8, 0.6)
    metrics.update(
        first_error,
        _command(0.0, north=0.0, yaw_rate=0.0),
        SaturationFlags(horizontal_speed=True),
        TrackingState.TRACKING,
    )
    metrics.update(
        second_error,
        _command(0.5, north=1.0, yaw_rate=0.5),
        SaturationFlags(acceleration=True, yaw_rate=True),
        TrackingState.TRACKING,
    )
    metrics.update(
        second_error,
        _command(1.0, north=0.0, yaw_rate=0.0, hold=True),
        SaturationFlags(),
        TrackingState.GOAL_HOLD,
    )
    metrics.record_stale_detection_latency(0.02)
    metrics.record_terminal_settling_time(0.5)
    result = metrics.result()
    assert result.position_rmse_m == pytest.approx(3.0 ** 0.5)
    assert result.maximum_position_error_m == 2.0
    assert result.maximum_command_acceleration_mps2 == pytest.approx(2.0)
    assert result.maximum_yaw_acceleration_radps2 == pytest.approx(1.0)
    assert result.saturation_count == 3
    assert result.hold_cycle_count == 1
    assert result.stale_detection_latency_s == 0.02
    assert result.terminal_settling_time_s == 0.5
    assert result.completion_status == "GOAL_HOLD"


def test_zero_and_known_constant_error_metrics_are_exact():
    """Zero and repeated constant errors produce analytic metric values."""
    zero = TrackingErrors(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    offset = TrackingErrors(0.5, 0.4, 0.3, 0.0, 0.0, 0.2, 0.1)
    zero_metrics = TrackingMetricsAccumulator()
    zero_metrics.update(
        zero,
        _command(0.0, north=0.0, yaw_rate=0.0),
        SaturationFlags(),
        TrackingState.TRACKING,
    )
    assert zero_metrics.result().position_rmse_m == 0.0
    offset_metrics = TrackingMetricsAccumulator()
    for time_s in (0.0, 0.5):
        offset_metrics.update(
            offset,
            _command(time_s, north=0.0, yaw_rate=0.0),
            SaturationFlags(),
            TrackingState.TRACKING,
        )
    result = offset_metrics.result()
    assert result.position_rmse_m == pytest.approx(0.5)
    assert result.horizontal_position_rmse_m == pytest.approx(0.4)
    assert result.vertical_position_rmse_m == pytest.approx(0.3)
    assert result.velocity_rmse_mps == pytest.approx(0.2)
    assert result.yaw_rmse_rad == pytest.approx(0.1)
