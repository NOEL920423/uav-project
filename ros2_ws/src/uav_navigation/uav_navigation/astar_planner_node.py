"""ROS 2 adapter and fixed-scene offline harness for the pure A* planner."""

import time

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

from uav_interfaces.msg import Obstacle, ObstacleArray

from uav_navigation.astar_planner import plan_path
from uav_navigation.bspline_smoother import (
    generate_bspline_candidate,
    validate_bspline_candidate,
)
from uav_navigation.coordinate_frames import (
    ISAAC_WORLD_FRAME,
    PLANNING_FRAME,
    isaac_to_ned_position,
)
from uav_navigation.models import (
    BSplineConfig,
    CircularObstacle,
    PlannerConfig,
    Point3D,
)
from uav_navigation.path_validator import validate_path


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
            "flight_altitude_m": 2.0,
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


class OfflinePlannerHarness(Node):
    """Publish one fixed scene and fail unless all planner outputs are safe."""

    def __init__(self) -> None:
        """Create the finite non-flight integration harness."""
        super().__init__("astar_offline_harness")
        self.declare_parameter("fixture", "bspline-safe-single-obstacle")
        self.declare_parameter("enable_bspline", True)
        self._fixture = str(self.get_parameter("fixture").value)
        self._bspline_enabled = bool(
            self.get_parameter("enable_bspline").value
        )
        allowed_fixtures = {
            "bspline-safe-open-space",
            "bspline-safe-single-obstacle",
            "bspline-rejected-corner-cut",
            "bspline-rejected-clearance",
            "bspline-disabled",
            "short-two-point-path",
            "three-point-path",
            "duplicate-control-point-path",
            "self-intersection-candidate",
            "curvature-limit-rejection",
        }
        if self._fixture not in allowed_fixtures:
            raise RuntimeError(f"unknown offline fixture: {self._fixture}")
        self._validate_edge_fixture_preflight()
        qos = _durable_qos()
        self._obstacle_publisher = self.create_publisher(
            ObstacleArray,
            "/uav/scene/obstacles",
            qos,
        )
        self._start_publisher = self.create_publisher(
            PoseStamped,
            "/uav/scene/start",
            qos,
        )
        self._goal_publisher = self.create_publisher(
            PoseStamped,
            "/uav/scene/goal",
            qos,
        )
        self.create_subscription(
            Path,
            "/uav/planner/path_raw",
            self._receive_raw,
            qos,
        )
        self.create_subscription(
            Path,
            "/uav/planner/path_simplified",
            self._receive_simplified,
            qos,
        )
        self.create_subscription(
            Path,
            "/uav/planner/path_bspline_candidate",
            self._receive_candidate,
            qos,
        )
        self.create_subscription(
            Bool,
            "/uav/planner/bspline_valid",
            self._receive_bspline_valid,
            qos,
        )
        self.create_subscription(
            Path,
            "/uav/planner/path",
            self._receive_final,
            qos,
        )
        self.create_subscription(
            String,
            "/uav/planner/status",
            self._receive_status,
            qos,
        )
        self._raw: Path | None = None
        self._simplified: Path | None = None
        self._candidate: Path | None = None
        self._bspline_valid: bool | None = None
        self._final: Path | None = None
        self._status = ""
        self._published = False
        self._finished = False
        self.exit_code = 1
        self._started_at = time.monotonic()
        self.create_timer(0.20, self._tick)

    def _tick(self) -> None:
        """Publish once and enforce a wall-time test timeout."""
        if not self._published:
            self._publish_fixture()
            self._published = True
        if not self._finished and time.monotonic() - self._started_at > 10.0:
            self._finish(1, "offline planner timed out without a valid path")

    def _publish_fixture(self) -> None:
        """Publish one coherent Isaac-world scene snapshot."""
        stamp = self.get_clock().now().to_msg()
        obstacle_array = ObstacleArray()
        obstacle_array.header.frame_id = ISAAC_WORLD_FRAME
        obstacle_array.header.stamp = stamp
        obstacle_radius = self._fixture_obstacle_radius()
        if obstacle_radius is not None:
            item = Obstacle()
            item.name = "offline_tower"
            item.center.x = 0.0
            item.center.y = 0.0
            item.center.z = 1.5
            item.radius = obstacle_radius
            item.height = 3.0
            obstacle_array.obstacles.append(item)
        start_x, goal_x = self._fixture_endpoints()
        start = self._scene_pose(start_x, 0.0, 0.0, stamp)
        goal = self._scene_pose(goal_x, 0.0, 0.0, stamp)
        self._obstacle_publisher.publish(obstacle_array)
        self._start_publisher.publish(start)
        self._goal_publisher.publish(goal)
        self.get_logger().info(
            f"Published offline fixture: {self._fixture}"
        )

    def _fixture_obstacle_radius(self) -> float | None:
        """Return the retained obstacle radius for this fixture."""
        if self._fixture in {
            "bspline-safe-open-space",
            "short-two-point-path",
            "three-point-path",
            "duplicate-control-point-path",
            "self-intersection-candidate",
        }:
            return None
        if self._fixture == "bspline-rejected-corner-cut":
            return 1.0
        return 0.20

    def _fixture_endpoints(self) -> tuple[float, float]:
        """Return Isaac-X endpoints that exercise the requested path size."""
        if self._fixture == "short-two-point-path":
            return -0.5, 0.5
        if self._fixture == "three-point-path":
            return -1.2, 1.2
        return -2.0, 2.0

    def _validate_edge_fixture_preflight(self) -> None:
        """Exercise control-only edges before the ROS scene transaction."""
        if self._fixture == "duplicate-control-point-path":
            controls = (
                Point3D(0.0, -1.0, -2.0),
                Point3D(0.0, -1.0, -2.0),
                Point3D(0.2, 0.0, -2.0),
                Point3D(0.0, 1.0, -2.0),
            )
            result = generate_bspline_candidate(
                controls,
                (),
                PlannerConfig(maximum_waypoint_spacing_m=2.0),
                BSplineConfig(
                    bspline_sample_spacing_m=0.2,
                    bspline_minimum_samples=2,
                    bspline_maximum_samples=100,
                ),
            )
            if not result.valid or result.control_point_count != 3:
                raise RuntimeError("duplicate-control preflight did not pass")
        if self._fixture == "self-intersection-candidate":
            crossing = (
                Point3D(0.0, 0.0, -2.0),
                Point3D(1.0, 1.0, -2.0),
                Point3D(0.0, 1.0, -2.0),
                Point3D(1.0, 0.0, -2.0),
            )
            validation, _, intersection = validate_bspline_candidate(
                crossing,
                (),
                PlannerConfig(maximum_waypoint_spacing_m=2.0),
                BSplineConfig(
                    bspline_sample_spacing_m=2.0,
                    bspline_minimum_samples=2,
                    bspline_maximum_curvature=100.0,
                ),
                crossing[0],
                crossing[-1],
            )
            if validation.valid or intersection != (0, 2):
                raise RuntimeError(
                    "self-intersection preflight did not reject"
                )

    @staticmethod
    def _scene_pose(x: float, y: float, z: float, stamp) -> PoseStamped:
        """Build one identity-orientation Isaac-world fixture pose."""
        message = PoseStamped()
        message.header.frame_id = ISAAC_WORLD_FRAME
        message.header.stamp = stamp
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.position.z = z
        message.pose.orientation.w = 1.0
        return message

    def _receive_raw(self, message: Path) -> None:
        self._raw = message
        self._maybe_finish()

    def _receive_simplified(self, message: Path) -> None:
        self._simplified = message
        self._maybe_finish()

    def _receive_candidate(self, message: Path) -> None:
        self._candidate = message
        self._maybe_finish()

    def _receive_bspline_valid(self, message: Bool) -> None:
        self._bspline_valid = message.data
        self._maybe_finish()

    def _receive_final(self, message: Path) -> None:
        self._final = message
        self._maybe_finish()

    def _receive_status(self, message: String) -> None:
        self._status = message.data
        self.get_logger().info(f"Planner status: {message.data}")
        if message.data.startswith("FAILURE|"):
            self._finish(1, message.data)
            return
        self._maybe_finish()

    def _maybe_finish(self) -> None:
        """Validate all received paths once the planner reports success."""
        if self._finished or not self._status.startswith("SUCCESS|"):
            return
        messages = self._raw, self._simplified, self._candidate, self._final
        if any(message is None for message in messages):
            return
        if self._bspline_valid is None:
            return
        if any(
            not message.poses
            for message in (self._raw, self._simplified, self._final)
        ):
            self._finish(1, "planner published an empty successful path")
            return
        if self._bspline_enabled and not self._candidate.poses:
            self._finish(1, "enabled planner did not publish a candidate")
            return
        for name, message in (
            ("raw", self._raw),
            ("simplified", self._simplified),
            ("candidate", self._candidate),
            ("final", self._final),
        ):
            if message.header.frame_id != PLANNING_FRAME:
                self._finish(1, f"{name} path frame is not px4_ned")
                return
        points = tuple(
            Point3D(
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            )
            for pose in self._final.poses
        )
        start_x, goal_x = self._fixture_endpoints()
        expected_start = Point3D(0.0, start_x, -2.0)
        expected_goal = Point3D(0.0, goal_x, -2.0)
        radius = self._fixture_obstacle_radius()
        obstacles = (
            (
                CircularObstacle(
                    "offline_tower",
                    Point3D(0.0, 0.0, -1.5),
                    radius,
                    3.0,
                ),
            )
            if radius is not None
            else ()
        )
        validation = validate_path(
            points,
            obstacles,
            PlannerConfig(),
            expected_start=expected_start,
            expected_goal=expected_goal,
        )
        if not validation.valid:
            self._finish(
                1,
                f"final path validation failed: {validation.reason}",
            )
            return
        if self._candidate.poses:
            candidate_start = self._candidate.poses[0].pose.position
            candidate_goal = self._candidate.poses[-1].pose.position
            candidate_endpoints = (
                Point3D(
                    candidate_start.x,
                    candidate_start.y,
                    candidate_start.z,
                ),
                Point3D(candidate_goal.x, candidate_goal.y, candidate_goal.z),
            )
            if candidate_endpoints != (expected_start, expected_goal):
                self._finish(1, "candidate did not preserve exact endpoints")
                return
        expected_valid, expected_source = self._expected_selection()
        if self._bspline_valid != expected_valid:
            self._finish(
                1,
                "unexpected bspline_valid for fixture "
                f"{self._fixture}: {self._bspline_valid}",
            )
            return
        fields = self._status_fields()
        expected_simplified_count = {
            "short-two-point-path": "2",
            "three-point-path": "3",
        }.get(self._fixture)
        if (
            expected_simplified_count is not None
            and fields.get("simplified_points") != expected_simplified_count
        ):
            self._finish(
                1,
                "unexpected simplified control count for fixture "
                f"{self._fixture}: {fields.get('simplified_points')}",
            )
            return
        if fields.get("final_source") != expected_source:
            self._finish(
                1,
                "unexpected final source for fixture "
                f"{self._fixture}: {fields.get('final_source')}",
            )
            return
        forbidden = [
            name
            for name, _ in self.get_topic_names_and_types()
            if name == "/fmu/in" or name.startswith("/fmu/in/")
        ]
        if forbidden:
            self._finish(1, f"forbidden PX4 topics detected: {forbidden}")
            return
        detail = (
            f"offline integration passed: raw={len(self._raw.poses)}, "
            f"simplified={len(self._simplified.poses)}, "
            f"final={len(self._final.poses)}, frame=px4_ned, "
            f"candidate={len(self._candidate.poses)}, "
            f"bspline_valid={str(self._bspline_valid).lower()}, "
            f"source={expected_source}, fixture={self._fixture}"
        )
        self._finish(0, detail)

    def _expected_selection(self) -> tuple[bool, str]:
        """Return the expected candidate validity and final source."""
        if not self._bspline_enabled or self._fixture == "bspline-disabled":
            return False, "ASTAR_SIMPLIFIED"
        rejected = {
            "bspline-rejected-corner-cut",
            "bspline-rejected-clearance",
            "curvature-limit-rejection",
        }
        if self._fixture in rejected:
            return False, "ASTAR_FALLBACK"
        return True, "BSPLINE"

    def _status_fields(self) -> dict[str, str]:
        """Parse the controlled key-value planning status."""
        fields = {}
        for field in self._status.split("|")[1:]:
            if "=" in field:
                key, value = field.split("=", 1)
                fields[key] = value
        return fields

    def _finish(self, exit_code: int, detail: str) -> None:
        """Set the process result and stop the finite harness."""
        if self._finished:
            return
        self._finished = True
        self.exit_code = exit_code
        if exit_code == 0:
            self.get_logger().info(detail)
        else:
            self.get_logger().error(detail)
        if rclpy.ok():
            rclpy.shutdown()


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


def offline_harness_main(args=None) -> int:
    """Run the finite offline fixture and return a shell-compatible code."""
    rclpy.init(args=args)
    node = OfflinePlannerHarness()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        exit_code = node.exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    main()
