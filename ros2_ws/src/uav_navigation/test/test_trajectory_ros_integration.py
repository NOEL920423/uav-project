"""ROS graph integration tests for Phase 4 trajectory publications."""

import time

from geometry_msgs.msg import PoseStamped

from nav_msgs.msg import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import Bool, String

from uav_interfaces.msg import TimedTrajectory

from uav_navigation.trajectory_parameterizer_node import (
    CANDIDATE_TOPIC,
    STATUS_TOPIC,
    TrajectoryParameterizerNode,
    VALID_TOPIC,
    _durable_qos,
)


def _path(frame_id="px4_ned"):
    """Create one deterministic ROS path message."""
    message = Path()
    message.header.frame_id = frame_id
    for x, y in ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = -2.0
        message.poses.append(pose)
    return message


def _spin_until(executor, predicate, timeout=2.0):
    """Spin until a deterministic predicate is met or wall time expires."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.05)
    return predicate()


def test_candidate_validity_status_wrong_frame_and_duplicate_suppression():
    """Exercise all Phase 4 ROS outputs without any flight topic."""
    rclpy.init()
    parameterizer = TrajectoryParameterizerNode()
    observer = Node("trajectory_contract_observer")
    candidates = []
    validities = []
    statuses = []
    qos = _durable_qos()
    observer.create_subscription(
        TimedTrajectory, CANDIDATE_TOPIC, candidates.append, qos
    )
    observer.create_subscription(Bool, VALID_TOPIC, validities.append, qos)
    observer.create_subscription(String, STATUS_TOPIC, statuses.append, qos)
    executor = SingleThreadedExecutor()
    executor.add_node(parameterizer)
    executor.add_node(observer)
    try:
        _spin_until(executor, lambda: False, timeout=0.2)
        accepted = _path()
        parameterizer._receive_path(accepted)
        assert _spin_until(
            executor,
            lambda: bool(candidates and validities and statuses),
        )
        assert candidates[-1].valid
        assert validities[-1].data
        assert statuses[-1].data.startswith("SUCCESS|")
        assert parameterizer.parameterization_count == 1
        parameterizer._receive_path(accepted)
        assert parameterizer.parameterization_count == 1
        parameterizer._receive_path(_path("map"))
        assert _spin_until(
            executor,
            lambda: len(validities) >= 2 and len(statuses) >= 2,
        )
        assert not validities[-1].data
        assert statuses[-1].data.startswith("REJECTED|")
        topics = dict(parameterizer.get_topic_names_and_types())
        assert not any(name.startswith("/fmu/in/") for name in topics)
    finally:
        executor.remove_node(observer)
        executor.remove_node(parameterizer)
        observer.destroy_node()
        parameterizer.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()
