"""ROS 2 adapter for expert-flight A* planning."""

from geometry_msgs.msg import PoseStamped

from nav_msgs.msg import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from std_msgs.msg import Bool, String

from uav_interfaces.msg import ObstacleArray

from uav_navigation.planner.astar_planner import plan_path
from uav_navigation.planner.coordinate_frames import (
    ISAAC_WORLD_FRAME,
    PLANNING_FRAME,
    isaac_to_ned_position,
)
from uav_navigation.planner.models import (
    BSplineConfig,
    CircularObstacle,
    PlannerConfig,
    Point3D,
)


def _durable_qos() -> QoSProfile:
    """Return the Phase 2 reliable transient-local depth-one profile."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _stamp_key(message) -> tuple[int, int]:
    """Return an exact ROS stamp key without converting clock domains."""
    return message.header.stamp.sec, message.header.stamp.nanosec


class AstarPlannerNode(Node):
    """Convert coherent scene snapshots and publish validated NED paths."""

    def __init__(self) -> None:
        """Declare fixed parameters, interfaces, and event-driven state."""
        super().__init__("astar_planner")
        self._declare_parameters()
        self._planning_frame = str(self.get_parameter("planning_frame").value)
        if self._planning_frame != PLANNING_FRAME:
            raise RuntimeError("planning_frame must be px4_ned in Phase 2")
        self._offsets = (
            float(self.get_parameter("ned_offset_x").value),
            float(self.get_parameter("ned_offset_y").value),
            float(self.get_parameter("ned_offset_z").value),
        )
        self._config = self._read_config()
        self._bspline_config = self._read_bspline_config()

        qos = _durable_qos()
        self._raw_publisher = self.create_publisher(
            Path,
            "/uav/planner/path_raw",
            qos,
        )
        self._simplified_publisher = self.create_publisher(
            Path,
            "/uav/planner/path_simplified",
            qos,
        )
        self._candidate_publisher = self.create_publisher(
            Path,
            "/uav/planner/path_bspline_candidate",
            qos,
        )
        self._bspline_valid_publisher = self.create_publisher(
            Bool,
            "/uav/planner/bspline_valid",
            qos,
        )
        self._final_publisher = self.create_publisher(
            Path,
            "/uav/planner/path",
            qos,
        )
        self._status_publisher = self.create_publisher(
            String,
            "/uav/planner/status",
            qos,
        )
        self.create_subscription(
            ObstacleArray,
            "/uav/scene/obstacles",
            self._receive_obstacles,
            qos,
        )
        self.create_subscription(
            PoseStamped,
            "/uav/scene/start",
            self._receive_start,
            qos,
        )
        self.create_subscription(
            PoseStamped,
            "/uav/scene/goal",
            self._receive_goal,
            qos,
        )
        self._obstacles: tuple[CircularObstacle, ...] | None = None
        self._start: Point3D | None = None
        self._goal: Point3D | None = None
        self._obstacle_stamp: tuple[int, int] | None = None
        self._start_stamp: tuple[int, int] | None = None
        self._goal_stamp: tuple[int, int] | None = None
        self._last_snapshot = None
        self._publish_status("WAITING|missing=obstacles,start,goal")
        self.get_logger().info(
            "Phase 3 A*/B-spline planner ready; no control or PX4 topics "
            "are created."
        )

    def _declare_parameters(self) -> None:
        """Declare static Phase 2 parameters with canonical defaults."""
        defaults = {
            "planning_frame": PLANNING_FRAME,
            "grid_resolution_m": 0.05,
            "grid_margin_m": 2.0,
            "uav_physical_radius_m": 0.18,
            "static_safety_margin_m": 0.13,
            "minimum_segment_clearance_m": 0.07,
            "endpoint_search_radius_m": 1.0,
            "path_simplification_tolerance_m": 0.05,
            "maximum_waypoint_spacing_m": 1.30,
            "use_direct_path_bias": True,
            "direct_path_bias_weight": 0.07,
            "use_clearance_aware_cost": True,
            "soft_clearance_radius_m": 0.40,
            "clearance_cost_weight": 0.25,
            "flight_altitude_m": 1.5,
            "enable_overfly_short_obstacles": True,
            "overfly_vertical_clearance_m": 0.35,
            "ned_offset_x": 0.0,
            "ned_offset_y": 0.0,
            "ned_offset_z": 0.0,
            "retry_extra_inflation_m": 0.07,
            "maximum_grid_cells": 4_000_000,
            "enable_bspline": True,
            "bspline_degree": 3,
            "bspline_sample_spacing_m": 0.08,
            "bspline_minimum_samples": 16,
            "bspline_maximum_samples": 1000,
            "bspline_maximum_curvature": 8.0,
            "bspline_minimum_clearance_m": 0.07,
            "bspline_preserve_endpoints": True,
            "bspline_allowed_bounds_margin_m": 0.0,
            "bspline_reject_self_intersection": True,
            "bspline_control_point_strategy": "validated_simplified_path",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)

    def _read_config(self) -> PlannerConfig:
        """Construct and validate the immutable pure planner configuration."""

        def value(name: str):
            return self.get_parameter(name).value

        return PlannerConfig(
            grid_resolution_m=value("grid_resolution_m"),
            grid_margin_m=value("grid_margin_m"),
            uav_physical_radius_m=value("uav_physical_radius_m"),
            static_safety_margin_m=value("static_safety_margin_m"),
            minimum_segment_clearance_m=value("minimum_segment_clearance_m"),
            endpoint_search_radius_m=value("endpoint_search_radius_m"),
            simplification_tolerance_m=value(
                "path_simplification_tolerance_m"
            ),
            maximum_waypoint_spacing_m=value("maximum_waypoint_spacing_m"),
            use_direct_path_bias=value("use_direct_path_bias"),
            direct_path_bias_weight=value("direct_path_bias_weight"),
            use_clearance_aware_cost=value("use_clearance_aware_cost"),
            soft_clearance_radius_m=value("soft_clearance_radius_m"),
            clearance_cost_weight=value("clearance_cost_weight"),
            flight_altitude_m=value("flight_altitude_m"),
            enable_overfly_short_obstacles=value(
                "enable_overfly_short_obstacles"
            ),
            overfly_vertical_clearance_m=value(
                "overfly_vertical_clearance_m"
            ),
            ned_origin_offset_z_m=value("ned_offset_z"),
            retry_extra_inflation_m=value("retry_extra_inflation_m"),
            maximum_grid_cells=value("maximum_grid_cells"),
        )

    def _read_bspline_config(self) -> BSplineConfig:
        """Construct and validate immutable Phase 3 smoothing configuration."""

        def value(name: str):
            return self.get_parameter(name).value

        return BSplineConfig(
            enable_bspline=value("enable_bspline"),
            bspline_degree=value("bspline_degree"),
            bspline_sample_spacing_m=value("bspline_sample_spacing_m"),
            bspline_minimum_samples=value("bspline_minimum_samples"),
            bspline_maximum_samples=value("bspline_maximum_samples"),
            bspline_maximum_curvature=value("bspline_maximum_curvature"),
            bspline_minimum_clearance_m=value(
                "bspline_minimum_clearance_m"
            ),
            bspline_preserve_endpoints=value("bspline_preserve_endpoints"),
            bspline_allowed_bounds_margin_m=value(
                "bspline_allowed_bounds_margin_m"
            ),
            bspline_reject_self_intersection=value(
                "bspline_reject_self_intersection"
            ),
            bspline_control_point_strategy=value(
                "bspline_control_point_strategy"
            ),
        )

    def _require_scene_frame(self, message) -> None:
        """Reject missing or mismatched spatial frame identifiers."""
        if message.header.frame_id != ISAAC_WORLD_FRAME:
            raise ValueError(
                f"input frame must be {ISAAC_WORLD_FRAME}, got "
                f"{message.header.frame_id or '<empty>'}"
            )

    def _convert_scene_point(self, point) -> Point3D:
        """Convert one geometry point from Isaac world to translated NED."""
        return isaac_to_ned_position(
            Point3D(point.x, point.y, point.z),
            *self._offsets,
        )

    def _receive_obstacles(self, message: ObstacleArray) -> None:
        """Validate and normalize a complete obstacle snapshot."""
        try:
            self._require_scene_frame(message)
            converted = tuple(
                sorted(
                    (
                        CircularObstacle(
                            item.name,
                            self._convert_scene_point(item.center),
                            item.radius,
                            item.height,
                        )
                        for item in message.obstacles
                    ),
                    key=lambda item: (
                        item.name,
                        item.center.x,
                        item.center.y,
                        item.radius,
                        item.height,
                    ),
                )
            )
        except (TypeError, ValueError) as error:
            self._obstacles = None
            self._publish_failure(f"invalid obstacle input: {error}")
            return
        self._obstacles = converted
        self._obstacle_stamp = _stamp_key(message)
        self._maybe_plan()

    def _receive_start(self, message: PoseStamped) -> None:
        """Validate a start pose while ignoring unsupported attitude."""
        self._receive_endpoint(message, is_start=True)

    def _receive_goal(self, message: PoseStamped) -> None:
        """Validate a goal pose while ignoring unsupported attitude."""
        self._receive_endpoint(message, is_start=False)

    def _receive_endpoint(
        self,
        message: PoseStamped,
        *,
        is_start: bool,
    ) -> None:
        """Convert an endpoint and force the configured planning altitude."""
        label = "start" if is_start else "goal"
        try:
            self._require_scene_frame(message)
            converted = self._convert_scene_point(message.pose.position)
            point = Point3D(
                converted.x,
                converted.y,
                self._config.planning_altitude_ned_m,
            )
        except (TypeError, ValueError) as error:
            if is_start:
                self._start = None
            else:
                self._goal = None
            self._publish_failure(f"invalid {label} input: {error}")
            return
        if is_start:
            self._start = point
            self._start_stamp = _stamp_key(message)
        else:
            self._goal = point
            self._goal_stamp = _stamp_key(message)
        self._maybe_plan()

    def _maybe_plan(self) -> None:
        """Plan once for each new coherent normalized input snapshot."""
        missing = []
        if self._obstacles is None:
            missing.append("obstacles")
        if self._start is None:
            missing.append("start")
        if self._goal is None:
            missing.append("goal")
        if missing:
            self._publish_status(f"WAITING|missing={','.join(missing)}")
            return
        stamps = (self._obstacle_stamp, self._start_stamp, self._goal_stamp)
        if len(set(stamps)) != 1:
            self._publish_failure("scene snapshot timestamps do not match")
            return
        snapshot = self._start, self._goal, self._obstacles
        if snapshot == self._last_snapshot:
            return
        self._last_snapshot = snapshot
        result = plan_path(
            self._start,
            self._goal,
            self._obstacles,
            self._config,
            self._bspline_config,
        )
        if not result.success:
            self._publish_failure(result.status)
            return
        stamp = self.get_clock().now().to_msg()
        self._raw_publisher.publish(self._to_path(result.raw_path, stamp))
        self._simplified_publisher.publish(
            self._to_path(result.simplified_path, stamp)
        )
        self._candidate_publisher.publish(
            self._to_path(result.bspline_candidate, stamp)
        )
        validity = Bool()
        validity.data = result.bspline_valid
        self._bspline_valid_publisher.publish(validity)
        self._final_publisher.publish(self._to_path(result.final_path, stamp))
        metrics = result.final_metrics
        fallback = result.fallback_reason or "none"
        status = (
            f"SUCCESS|method={result.simplification_method}"
            "|astar_success=true"
            f"|bspline_enabled={str(result.bspline_enabled).lower()}"
            f"|bspline_valid={str(result.bspline_valid).lower()}"
            f"|bspline_selected={str(result.bspline_selected).lower()}"
            f"|final_source={result.final_path_source}"
            f"|raw_points={len(result.raw_path)}"
            f"|simplified_points={len(result.simplified_path)}"
            f"|candidate_points={len(result.bspline_candidate)}"
            f"|final_points={len(result.final_path)}"
            f"|length_m={metrics.path_length_m:.6f}"
            "|minimum_physical_clearance_m="
            f"{metrics.minimum_physical_clearance_m:.6f}"
            "|maximum_curvature_inverse_m="
            f"{metrics.maximum_curvature_inverse_m:.6f}"
            "|bspline_rejection="
            f"{result.bspline_rejection_reason or 'none'}"
            f"|fallback={fallback}"
        )
        self._publish_status(status)

    def _to_path(self, points: tuple[Point3D, ...], stamp) -> Path:
        """Build a path with one ROS-clock stamp and identity orientations."""
        message = Path()
        message.header.frame_id = self._planning_frame
        message.header.stamp = stamp
        for point in points:
            pose = PoseStamped()
            pose.header.frame_id = self._planning_frame
            pose.header.stamp = stamp
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.position.z = point.z
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        return message

    def _publish_failure(self, reason: str) -> None:
        """Clear durable paths and publish a failure without unsafe output."""
        self._last_snapshot = None
        stamp = self.get_clock().now().to_msg()
        empty = self._to_path((), stamp)
        self._raw_publisher.publish(empty)
        self._simplified_publisher.publish(self._to_path((), stamp))
        self._candidate_publisher.publish(self._to_path((), stamp))
        validity = Bool()
        validity.data = False
        self._bspline_valid_publisher.publish(validity)
        self._final_publisher.publish(self._to_path((), stamp))
        detail = reason.replace("|", "/")
        self._publish_status(f"FAILURE|reason={detail}")

    def _publish_status(self, text: str) -> None:
        """Publish one controlled-prefix durable status message."""
        message = String()
        message.data = text
        self._status_publisher.publish(message)


def main(args=None) -> None:
    """Run the event-driven planner until ROS requests shutdown."""
    rclpy.init(args=args)
    node = AstarPlannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
