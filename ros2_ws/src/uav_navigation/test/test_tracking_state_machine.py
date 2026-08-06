"""Pure tests for stale gates, time jumps, settling, and terminal timeout."""

import math

from uav_navigation.models import Point3D
from uav_navigation.tracking_models import (
    TrackingConfig,
    TrackingState,
    VehicleState,
)
from uav_navigation.trajectory_models import TrajectoryPoint
from uav_navigation.trajectory_tracker import OfflineTrackingController


def _point(time_s, north, speed=0.0, yaw=0.0):
    """Build one structurally valid straight reference point."""
    return TrajectoryPoint(
        time_from_start_s=time_s,
        position=Point3D(north, 0.0, -2.0),
        velocity=Point3D(speed, 0.0, 0.0),
        acceleration=Point3D(0.0, 0.0, 0.0),
        jerk=Point3D(0.0, 0.0, 0.0),
        yaw_ned=yaw,
        yaw_rate_radps=0.0,
        yaw_acceleration_radps2=0.0,
        arc_length_m=north,
        curvature_inverse_m=0.0,
    )


TRAJECTORY = (_point(0.0, 0.0), _point(1.0, 1.0))


def _odom(
    time_s,
    position=Point3D(0.0, 0.0, -2.0),
    velocity=Point3D(0.0, 0.0, 0.0),
    frame="px4_ned",
    yaw=0.0,
):
    """Build one measured state at a caller-controlled receipt time."""
    return VehicleState(time_s, frame, position, velocity, yaw, 0.0)


def _nonfinite_point():
    """Construct malformed external data past Point3D's normal guard."""
    point = object.__new__(Point3D)
    object.__setattr__(point, "x", math.nan)
    object.__setattr__(point, "y", 0.0)
    object.__setattr__(point, "z", -2.0)
    return point


def _ready_controller(config=None, receipt=1.0):
    """Build a synchronized controller without running its first cycle."""
    controller = OfflineTrackingController(config)
    controller.accept_trajectory(TRAJECTORY, "px4_ned", True, receipt)
    controller.accept_validity(True, receipt)
    controller.accept_odometry(_odom(receipt), receipt)
    return controller


def test_waiting_prestart_tracking_and_duplicate_epoch_contract():
    """Receipt gates progress and duplicates preserve the tracking epoch."""
    controller = OfflineTrackingController()
    assert controller.step(0.0).state == TrackingState.WAITING_TRAJECTORY
    assert controller.accept_trajectory(TRAJECTORY, "px4_ned", True, 1.0)
    epoch = controller.tracking_epoch_s
    assert not controller.accept_trajectory(
        TRAJECTORY, "px4_ned", True, 1.05
    )
    assert controller.tracking_epoch_s == epoch
    assert controller.step(1.01).state == TrackingState.WAITING_VALIDITY
    controller.accept_validity(True, 1.02)
    assert controller.step(1.03).state == TrackingState.WAITING_ODOMETRY
    controller.accept_odometry(_odom(1.04), 1.04)
    assert controller.step(1.05).state == TrackingState.PRESTART_HOLD
    controller.accept_odometry(_odom(1.11), 1.11)
    assert controller.step(1.11).state == TrackingState.TRACKING
    assert controller.accepted_trajectory_count == 1


def test_false_and_stale_validity_select_specific_hold_states():
    """False or stale validity cannot produce a tracking output."""
    controller = _ready_controller()
    controller.accept_validity(False, 1.01)
    controller.accept_odometry(_odom(1.11), 1.11)
    result = controller.step(1.11)
    assert result.state == TrackingState.WAITING_VALIDITY
    assert result.selected_command.hold_active
    controller = _ready_controller()
    controller.accept_odometry(_odom(1.6), 1.6)
    result = controller.step(1.6)
    assert result.state == TrackingState.HOLD_STALE_TRAJECTORY
    assert "stale" in result.selected_command.hold_reason


def test_stale_wrong_frame_and_nonfinite_odometry_are_held():
    """Odometry freshness, frame, and finite gates have explicit states."""
    config = TrackingConfig(trajectory_validity_timeout_s=10.0)
    controller = _ready_controller(config)
    result = controller.step(1.4)
    assert result.state == TrackingState.HOLD_STALE_ODOMETRY
    controller = _ready_controller(config)
    controller.accept_odometry(_odom(1.1, frame="map"), 1.1)
    assert controller.step(1.11).state == TrackingState.HOLD_INVALID_FRAME
    controller = _ready_controller(config)
    controller.accept_odometry(
        _odom(1.1, position=_nonfinite_point()), 1.1
    )
    result = controller.step(1.11)
    assert result.state == TrackingState.HOLD_INVALID_COMMAND
    assert "non-finite" in result.status_message


def test_excessive_error_and_equal_or_backward_time_select_hold():
    """Tracking envelope and non-monotonic control time fail closed."""
    config = TrackingConfig(trajectory_validity_timeout_s=10.0)
    controller = _ready_controller(config)
    controller.accept_odometry(
        _odom(1.11, position=Point3D(10.0, 0.0, -2.0)), 1.11
    )
    assert controller.step(1.11).state == TrackingState.HOLD_TRACKING_ERROR
    assert controller.step(1.11).state == TrackingState.HOLD_TIME_JUMP
    assert controller.step(0.5).state == TrackingState.HOLD_TIME_JUMP
    assert controller.step(0.6).state == TrackingState.WAITING_TRAJECTORY


def test_terminal_requires_continuous_settling_then_goal_hold():
    """All goal gates must remain true for the complete settle interval."""
    config = TrackingConfig(
        trajectory_start_delay_s=0.0,
        trajectory_validity_timeout_s=10.0,
        odometry_timeout_s=10.0,
        goal_settle_time_s=0.5,
    )
    controller = _ready_controller(config, receipt=0.0)
    final_position = Point3D(1.0, 0.0, -2.0)
    controller.accept_odometry(_odom(1.0, final_position), 1.0)
    assert controller.step(1.0).state == TrackingState.GOAL_SETTLING
    controller.accept_odometry(
        _odom(1.3, Point3D(1.3, 0.0, -2.0)), 1.3
    )
    assert controller.step(1.3).state == TrackingState.TRACKING
    controller.accept_odometry(_odom(1.4, final_position), 1.4)
    assert controller.step(1.4).state == TrackingState.GOAL_SETTLING
    controller.accept_odometry(_odom(1.91, final_position), 1.91)
    result = controller.step(1.91)
    assert result.state == TrackingState.GOAL_HOLD
    assert result.selected_command.hold_active


def test_terminal_not_reached_and_yaw_wrap_shortest_error():
    """Terminal timeout holds while wrap crossing uses shortest yaw error."""
    config = TrackingConfig(
        trajectory_start_delay_s=0.0,
        trajectory_validity_timeout_s=10.0,
        odometry_timeout_s=10.0,
        maximum_terminal_wait_s=1.0,
    )
    controller = _ready_controller(config, receipt=0.0)
    controller.accept_odometry(
        _odom(2.1, Point3D(1.5, 0.0, -2.0)), 2.1
    )
    assert controller.step(2.1).state == TrackingState.TERMINAL_NOT_REACHED
    yaw_trajectory = (
        _point(0.0, 0.0, yaw=3.1),
        _point(1.0, 1.0, yaw=3.2),
    )
    controller = OfflineTrackingController(config)
    controller.accept_trajectory(yaw_trajectory, "px4_ned", True, 0.0)
    controller.accept_validity(True, 0.0)
    controller.accept_odometry(_odom(0.2, yaw=-3.1), 0.2)
    result = controller.step(0.2)
    assert result.state == TrackingState.TRACKING
    assert abs(result.errors.yaw_error_rad) < 0.2
