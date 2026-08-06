"""ROS graph contract tests for the Phase 5 trajectory follower adapter."""

import time

from geometry_msgs.msg import PoseStamped, TwistStamped

from nav_msgs.msg import Odometry

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import Bool

from uav_interfaces.msg import (
    TimedTrajectory,
    TrajectoryPoint,
    TrajectoryTrackingStatus,
)

from uav_navigation.tracking_fixtures import TRACKING_FIXTURES
from uav_navigation.tracking_models import TrackingState
from uav_navigation.trajectory_follower_node import (
    COMMAND_TOPIC,
    REFERENCE_POSE_TOPIC,
    REFERENCE_TWIST_TOPIC,
    TRACKING_STATUS_TOPIC,
    TrajectoryFollowerNode,
    live_qos,
)


def _trajectory():
    """Create a two-point valid relative-time candidate message."""
    message = TimedTrajectory()
    message.header.frame_id = "px4_ned"
    message.valid = True
    for index in range(2):
        point = TrajectoryPoint()
        point.time_from_start.sec = index
        point.position.x = float(index)
        point.position.z = -2.0
        point.arc_length = float(index)
        message.points.append(point)
    return message


def _odometry():
    """Create a finite standard planar odometry message."""
    message = Odometry()
    message.header.frame_id = "px4_ned"
    message.pose.pose.position.z = -2.0
    message.pose.pose.orientation.w = 1.0
    return message


def _spin_until(executor, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
    return predicate()


def test_exact_twenty_fixture_vocabulary_and_expected_categories():
    """Automation exposes every mandated fixture with an expectation."""
    assert len(TRACKING_FIXTURES) == 20
    assert all(item.expected for item in TRACKING_FIXTURES.values())
    assert TRACKING_FIXTURES["stale-odometry"].expected == (
        "HOLD_STALE_ODOMETRY"
    )
    assert TRACKING_FIXTURES["invalid-command-rejection"].expected == (
        "PURE_VALIDATOR_REJECTION"
    )


def test_follower_outputs_references_status_deduplication_and_stale_hold():
    """Exercise the live adapter contract without a planner or simulator."""
    rclpy.init()
    follower = TrajectoryFollowerNode()
    observer = Node("phase5_tracking_contract_observer")
    commands = []
    poses = []
    twists = []
    statuses = []
    qos = live_qos()
    observer.create_subscription(
        TwistStamped, COMMAND_TOPIC, commands.append, qos
    )
    observer.create_subscription(
        PoseStamped, REFERENCE_POSE_TOPIC, poses.append, qos
    )
    observer.create_subscription(
        TwistStamped, REFERENCE_TWIST_TOPIC, twists.append, qos
    )
    observer.create_subscription(
        TrajectoryTrackingStatus, TRACKING_STATUS_TOPIC, statuses.append, qos
    )
    executor = SingleThreadedExecutor()
    executor.add_node(follower)
    executor.add_node(observer)
    try:
        trajectory = _trajectory()
        follower._trajectory_callback(trajectory)
        follower._validity_callback(Bool(data=True))
        follower._odometry_callback(_odometry())
        follower.controller.tracking_epoch_s = follower._now_seconds() - 0.5
        follower._tick()
        assert _spin_until(
            executor,
            lambda: bool(commands and poses and twists and statuses),
        )
        assert follower.last_result.state == TrackingState.TRACKING
        assert commands[-1].header.frame_id == "px4_ned"
        assert commands[-1].twist.angular.x == 0.0
        assert commands[-1].twist.angular.y == 0.0
        assert poses[-1].header.frame_id == "px4_ned"
        assert twists[-1].header.frame_id == "px4_ned"
        assert statuses[-1].header.frame_id == "px4_ned"
        assert statuses[-1].command_valid
        accepted = follower.controller.accepted_trajectory_count
        epoch = follower.controller.tracking_epoch_s
        follower._trajectory_callback(trajectory)
        assert follower.controller.accepted_trajectory_count == accepted
        assert follower.controller.tracking_epoch_s == epoch
        follower.controller.odometry_receipt_s = (
            follower._now_seconds()
            - follower.controller.config.odometry_timeout_s
            - 0.1
        )
        follower._tick()
        assert follower.last_result.state == TrackingState.HOLD_STALE_ODOMETRY
        assert follower.last_result.selected_command.hold_active
        topics = dict(follower.get_topic_names_and_types())
        assert not any(name.startswith("/fmu/in/") for name in topics)
    finally:
        executor.remove_node(observer)
        executor.remove_node(follower)
        observer.destroy_node()
        follower.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
