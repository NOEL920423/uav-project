"""Explicitly sequence one ASTAR_EXPERT-controlled PX4 SITL flight."""

import importlib
import json
import math

from geometry_msgs.msg import PoseStamped

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from std_msgs.msg import Bool, String

from std_srvs.srv import SetBool

from uav_interfaces.msg import (
    ControlMuxStatus,
    ObstacleArray,
    Px4FlightStatus,
    Px4OutputGateStatus,
    Px4StreamStatus,
    TimedTrajectory,
    TrajectoryTrackingStatus,
)
from uav_interfaces.srv import (
    SendPx4VehicleCommand,
    SetControlSource,
    SetPx4FlightEnable,
    SetPx4OutputEnable,
    SetPx4StreamEnable,
)

from uav_navigation.trajectory_follower_node import (
    SET_TRACKING_ENABLE_SERVICE,
    TRACKING_STATUS_TOPIC,
    durable_qos,
    live_qos,
)

from uav_px4_control.control_mux_node import (
    MUX_STATUS_TOPIC,
    SET_SOURCE_SERVICE,
    control_qos,
)
from uav_px4_control.px4_flight_models import (
    FlightEvidence,
    Px4FlightConfig,
    altitude_above_ground,
    planner_status_allows_final_path,
    vehicle_command_was_accepted,
)
from uav_px4_control.px4_flight_state_machine import Px4FlightStateMachine
from uav_px4_control.px4_mapping_gate_node import (
    GATE_STATUS_TOPIC,
    SET_OUTPUT_ENABLE_SERVICE,
)
from uav_px4_control.px4_setpoint_streamer_node import (
    SET_STREAM_ENABLE_SERVICE,
    STREAM_STATUS_TOPIC,
    VEHICLE_ODOMETRY_TOPIC,
    VEHICLE_STATUS_TOPIC,
    px4_output_qos,
    sitl_process_matches,
    xrce_agent_detected,
)
from uav_px4_control.px4_vehicle_command_owner_node import (
    SEND_VEHICLE_COMMAND_SERVICE,
)


FLIGHT_STATUS_TOPIC = "/uav/px4/flight_status"
SET_FLIGHT_ENABLE_SERVICE = "/uav/px4/set_flight_enable"
VEHICLE_COMMAND_ACK_TOPIC = "/fmu/out/vehicle_command_ack"
VEHICLE_LAND_DETECTED_TOPIC = "/fmu/out/vehicle_land_detected"
PLANNER_STATUS_TOPIC = "/uav/planner/status"
BSPLINE_VALID_TOPIC = "/uav/planner/bspline_valid"
TRAJECTORY_TOPIC = "/uav/trajectory/candidate"
SCENE_OBSTACLES_TOPIC = "/uav/scene/obstacles"
SCENE_START_TOPIC = "/uav/scene/start"
SCENE_GOAL_TOPIC = "/uav/scene/goal"
EXTERNAL_SCENE_OBSTACLES_TOPIC = "/uav/isaac/scene/obstacles"
EXTERNAL_SCENE_START_TOPIC = "/uav/isaac/scene/start"
EXTERNAL_SCENE_GOAL_TOPIC = "/uav/isaac/scene/goal"
ISAAC_BRIDGE_STATUS_TOPIC = "/uav/isaac/bridge_status"
MSG_SUPERVISOR_DISABLED = (
    "SITL flight supervisor starts disabled; lifecycle commands "
    "use the sole VehicleCommand owner"
)


class Px4SitlFlightSupervisorNode(Node):
    """Own VehicleCommand while reusing the validated Phase 1-8 pipeline."""

    def __init__(self) -> None:
        """Create disabled command, evidence, service, and scene boundaries."""
        super().__init__("px4_sitl_flight_supervisor")
        message_module = importlib.import_module("px4_msgs.msg")
        self._VehicleCommand = message_module.VehicleCommand
        self._VehicleStatus = message_module.VehicleStatus
        self._VehicleOdometry = message_module.VehicleOdometry
        self._VehicleLandDetected = message_module.VehicleLandDetected
        self._VehicleCommandAck = message_module.VehicleCommandAck

        defaults = Px4FlightConfig()
        for name in defaults.__dataclass_fields__:
            self.declare_parameter(name, getattr(defaults, name))
        self.declare_parameter("takeoff_path_north_m", 0.30)
        self.declare_parameter("goal_north_m", 3.0)
        self.declare_parameter("goal_east_m", 0.5)
        self.declare_parameter("use_external_scene", False)
        self.declare_parameter("external_runtime_timeout_s", 0.50)
        self.declare_parameter("expected_sitl_process_fragment", (
            "/PX4-Autopilot/build/px4_sitl_default/bin/px4"
        ))
        self.config = Px4FlightConfig(**{
            name: self.get_parameter(name).value
            for name in defaults.__dataclass_fields__
        })
        self.machine = Px4FlightStateMachine(self.config)

        self._planner_path_valid = False
        self._planner_status = ""
        self._bspline_valid = False
        self._trajectory_valid = False
        self._trajectory_source = ""
        self._trajectory_goal: tuple[float, float, float] | None = None
        self._tracking_status: TrajectoryTrackingStatus | None = None
        self._mux_status: ControlMuxStatus | None = None
        self._gate_status: Px4OutputGateStatus | None = None
        self._stream_status: Px4StreamStatus | None = None
        self._last_stream_state = ""
        self._vehicle_status = None
        self._vehicle_status_receipt_s: float | None = None
        self._vehicle_odometry = None
        self._vehicle_odometry_receipt_s: float | None = None
        self._land_detected = None
        self._scene_kind = ""
        self._expected_goal: tuple[float, float, float] | None = None
        self._initial_north_east: tuple[float, float] | None = None
        self._ground_down_m: float | None = None
        self._last_command_ack = ""
        self._fatal_command_ack = ""
        self._vehicle_command_count = 0
        self._last_action_time: dict[str, float] = {}
        self._pending_services: dict[str, object] = {}
        self._landing_commanded = False
        self._external_obstacles: ObstacleArray | None = None
        self._external_start: PoseStamped | None = None
        self._external_goal: PoseStamped | None = None
        self._external_stamps: dict[str, tuple[int, int]] = {}
        self._external_scene_id = ""
        self._external_scene_revision = 0
        self._external_bridge_ready = False
        self._external_bridge_receipt_s: float | None = None

        qos = control_qos()
        self.create_subscription(
            String, PLANNER_STATUS_TOPIC, self._planner_callback, qos
        )
        self.create_subscription(
            Bool, BSPLINE_VALID_TOPIC, self._bspline_callback, durable_qos()
        )
        self.create_subscription(
            TimedTrajectory,
            TRAJECTORY_TOPIC,
            self._trajectory_callback,
            durable_qos(),
        )
        self.create_subscription(
            TrajectoryTrackingStatus,
            TRACKING_STATUS_TOPIC,
            self._tracking_callback,
            live_qos(),
        )
        self.create_subscription(
            ControlMuxStatus, MUX_STATUS_TOPIC, self._mux_callback, qos
        )
        self.create_subscription(
            Px4OutputGateStatus,
            GATE_STATUS_TOPIC,
            self._gate_callback,
            qos,
        )
        self.create_subscription(
            Px4StreamStatus,
            STREAM_STATUS_TOPIC,
            self._stream_callback,
            qos,
        )
        telemetry_qos = px4_output_qos()
        self.create_subscription(
            self._VehicleStatus,
            VEHICLE_STATUS_TOPIC,
            self._vehicle_status_callback,
            telemetry_qos,
        )
        self.create_subscription(
            self._VehicleOdometry,
            VEHICLE_ODOMETRY_TOPIC,
            self._vehicle_odometry_callback,
            telemetry_qos,
        )
        self.create_subscription(
            self._VehicleLandDetected,
            VEHICLE_LAND_DETECTED_TOPIC,
            self._land_callback,
            telemetry_qos,
        )
        self.create_subscription(
            self._VehicleCommandAck,
            VEHICLE_COMMAND_ACK_TOPIC,
            self._ack_callback,
            telemetry_qos,
        )
        self.create_subscription(
            ObstacleArray,
            EXTERNAL_SCENE_OBSTACLES_TOPIC,
            self._external_obstacles_callback,
            durable_qos(),
        )
        self.create_subscription(
            PoseStamped,
            EXTERNAL_SCENE_START_TOPIC,
            self._external_start_callback,
            durable_qos(),
        )
        self.create_subscription(
            PoseStamped,
            EXTERNAL_SCENE_GOAL_TOPIC,
            self._external_goal_callback,
            durable_qos(),
        )
        self.create_subscription(
            String,
            ISAAC_BRIDGE_STATUS_TOPIC,
            self._external_bridge_status_callback,
            qos,
        )

        self._obstacles_publisher = self.create_publisher(
            ObstacleArray, SCENE_OBSTACLES_TOPIC, durable_qos()
        )
        self._start_publisher = self.create_publisher(
            PoseStamped, SCENE_START_TOPIC, durable_qos()
        )
        self._goal_publisher = self.create_publisher(
            PoseStamped, SCENE_GOAL_TOPIC, durable_qos()
        )
        self._status_publisher = self.create_publisher(
            Px4FlightStatus, FLIGHT_STATUS_TOPIC, qos
        )
        self._flight_service = self.create_service(
            SetPx4FlightEnable,
            SET_FLIGHT_ENABLE_SERVICE,
            self._flight_enable_callback,
        )
        self._mux_client = self.create_client(
            SetControlSource, SET_SOURCE_SERVICE
        )
        self._gate_client = self.create_client(
            SetPx4OutputEnable, SET_OUTPUT_ENABLE_SERVICE
        )
        self._stream_client = self.create_client(
            SetPx4StreamEnable, SET_STREAM_ENABLE_SERVICE
        )
        self._tracking_client = self.create_client(
            SetBool, SET_TRACKING_ENABLE_SERVICE
        )
        self._vehicle_command_client = self.create_client(
            SendPx4VehicleCommand, SEND_VEHICLE_COMMAND_SERVICE
        )
        self._timer = self.create_timer(0.05, self._tick)
        self.get_logger().warning(MSG_SUPERVISOR_DISABLED)

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _flight_enable_callback(self, request, response):
        now = self._now_seconds()
        if request.enable and not self._sitl_guard_valid():
            response.accepted = False
            response.mission_enable_requested = False
            response.state = self.machine.state.value
            response.status_message = (
                "PX4 SITL/XRCE identity guard is not ready"
            )
            return response
        if request.enable and not self._external_environment_valid(now):
            response.accepted = False
            response.mission_enable_requested = False
            response.state = self.machine.state.value
            response.status_message = (
                "Isaac external scene/runtime is not ready"
            )
            return response
        accepted, message = self.machine.request_enable(request.enable, now)
        if accepted and request.enable:
            self._reset_mission_evidence()
            self._publish_takeoff_scene()
        response.accepted = accepted
        response.mission_enable_requested = (
            self.machine.mission_enable_requested
        )
        response.state = self.machine.state.value
        response.status_message = message
        return response

    def _reset_mission_evidence(self) -> None:
        self._scene_kind = ""
        self._expected_goal = None
        self._initial_north_east = None
        self._ground_down_m = None
        if self._vehicle_odometry is not None:
            down = float(self._vehicle_odometry.position[2])
            if math.isfinite(down):
                self._ground_down_m = down
        self._fatal_command_ack = ""
        self._last_command_ack = ""
        self._last_action_time.clear()
        self._pending_services.clear()
        self._landing_commanded = False

    def _planner_callback(self, message: String) -> None:
        self._planner_status = message.data
        self._planner_path_valid = message.data.startswith("SUCCESS|")

    def _bspline_callback(self, message: Bool) -> None:
        self._bspline_valid = bool(message.data)

    def _trajectory_callback(self, message: TimedTrajectory) -> None:
        self._trajectory_valid = bool(message.valid and message.points)
        self._trajectory_source = message.trajectory_source
        if message.points:
            point = message.points[-1].position
            self._trajectory_goal = (point.x, point.y, point.z)

    def _tracking_callback(self, message: TrajectoryTrackingStatus) -> None:
        self._tracking_status = message

    def _mux_callback(self, message: ControlMuxStatus) -> None:
        self._mux_status = message

    def _gate_callback(self, message: Px4OutputGateStatus) -> None:
        self._gate_status = message

    def _stream_callback(self, message: Px4StreamStatus) -> None:
        self._stream_status = message
        if message.state != self._last_stream_state:
            transition = (
                f"STREAM_TRANSITION {self._last_stream_state or 'NONE'}"
                f"->{message.state} reason={message.stop_reason} "
                f"rate={message.observed_rate_hz:.3f} "
                f"max_gap={message.maximum_publish_gap:.3f}"
            )
            # rclpy binds severity to a call site, so INFO/WARN need distinct
            # source lines when a stream later crosses into a fault state.
            if message.state.startswith(("STOPPED_", "LATCHED_")):
                self.get_logger().warning(transition)
            else:
                self.get_logger().info(transition)
            self._last_stream_state = message.state

    def _vehicle_status_callback(self, message) -> None:
        self._vehicle_status = message
        self._vehicle_status_receipt_s = self._now_seconds()

    def _vehicle_odometry_callback(self, message) -> None:
        self._vehicle_odometry = message
        self._vehicle_odometry_receipt_s = self._now_seconds()

    def _land_callback(self, message) -> None:
        self._land_detected = message

    @staticmethod
    def _message_stamp(message) -> tuple[int, int]:
        return message.header.stamp.sec, message.header.stamp.nanosec

    @staticmethod
    def _isaac_pose_valid(message: PoseStamped) -> bool:
        position = message.pose.position
        return bool(
            message.header.frame_id == "isaac_world"
            and all(math.isfinite(float(value)) for value in (
                position.x,
                position.y,
                position.z,
            ))
        )

    def _external_obstacles_callback(self, message: ObstacleArray) -> None:
        if message.header.frame_id != "isaac_world":
            self._external_obstacles = None
            return
        self._external_obstacles = message
        self._external_stamps["obstacles"] = self._message_stamp(message)

    def _external_start_callback(self, message: PoseStamped) -> None:
        if not self._isaac_pose_valid(message):
            self._external_start = None
            return
        self._external_start = message
        self._external_stamps["start"] = self._message_stamp(message)

    def _external_goal_callback(self, message: PoseStamped) -> None:
        if not self._isaac_pose_valid(message):
            self._external_goal = None
            return
        self._external_goal = message
        self._external_stamps["goal"] = self._message_stamp(message)

    def _external_bridge_status_callback(self, message: String) -> None:
        try:
            status = json.loads(message.data)
            ready = status.get("ready")
            scene_id = str(status.get("scene_id", "")).strip()
            revision = int(status.get("scene_revision", 0))
            if not isinstance(ready, bool) or not scene_id or revision <= 0:
                raise ValueError("invalid bridge status fields")
        except (TypeError, ValueError, json.JSONDecodeError):
            self._external_bridge_ready = False
            return
        self._external_bridge_ready = ready
        self._external_scene_id = scene_id
        self._external_scene_revision = revision
        self._external_bridge_receipt_s = self._now_seconds()

    def _uses_external_scene(self) -> bool:
        return bool(self.get_parameter("use_external_scene").value)

    def _external_environment_valid(self, now: float) -> bool:
        if not self._uses_external_scene():
            return True
        if (
            self._external_obstacles is None
            or self._external_start is None
            or self._external_goal is None
            or self._external_bridge_receipt_s is None
            or not self._external_bridge_ready
        ):
            return False
        stamps = set(self._external_stamps.values())
        if len(stamps) != 1 or len(self._external_stamps) != 3:
            return False
        timeout = float(
            self.get_parameter("external_runtime_timeout_s").value
        )
        return bool(
            math.isfinite(timeout)
            and timeout > 0.0
            and now - self._external_bridge_receipt_s <= timeout
        )

    def _ack_callback(self, message) -> None:
        watched = {
            self._VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            self._VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            self._VehicleCommand.VEHICLE_CMD_NAV_LAND,
        }
        if int(message.command) not in watched:
            return
        result_names = {
            self._VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED: "ACCEPTED",
            self._VehicleCommandAck.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED:
                "TEMPORARILY_REJECTED",
            self._VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED: "DENIED",
            self._VehicleCommandAck.VEHICLE_CMD_RESULT_UNSUPPORTED:
                "UNSUPPORTED",
            self._VehicleCommandAck.VEHICLE_CMD_RESULT_FAILED: "FAILED",
            self._VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS:
                "IN_PROGRESS",
            self._VehicleCommandAck.VEHICLE_CMD_RESULT_CANCELLED: "CANCELLED",
        }
        result = result_names.get(
            int(message.result), str(int(message.result))
        )
        self._last_command_ack = f"{int(message.command)}:{result}"
        fatal = {
            self._VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED,
            self._VehicleCommandAck.VEHICLE_CMD_RESULT_UNSUPPORTED,
            self._VehicleCommandAck.VEHICLE_CMD_RESULT_FAILED,
            self._VehicleCommandAck.VEHICLE_CMD_RESULT_CANCELLED,
        }
        if int(message.result) in fatal:
            self._fatal_command_ack = self._last_command_ack

    def _sitl_guard_valid(self) -> bool:
        fragment = str(
            self.get_parameter("expected_sitl_process_fragment").value
        )
        return sitl_process_matches(fragment) and xrce_agent_detected()

    def _goal_matches_expected(self) -> bool:
        if self._trajectory_goal is None or self._expected_goal is None:
            return False
        return math.dist(self._trajectory_goal, self._expected_goal) <= 0.05

    def _pipeline_ready(self) -> bool:
        return bool(
            planner_status_allows_final_path(
                self._planner_status, self._bspline_valid
            )
            and self._trajectory_valid
            and self._trajectory_source == "PHASE4_TIME_PARAMETERIZED"
            and self._goal_matches_expected()
        )

    def _evidence(self, now: float) -> FlightEvidence:
        status = self._vehicle_status
        odometry = self._vehicle_odometry
        armed = bool(
            status is not None
            and status.arming_state == status.ARMING_STATE_ARMED
        )
        offboard = bool(
            status is not None
            and status.nav_state == status.NAVIGATION_STATE_OFFBOARD
        )
        failsafe = bool(status is not None and status.failsafe)
        altitude = 0.0
        if odometry is not None and self._ground_down_m is not None:
            down = float(odometry.position[2])
            if math.isfinite(down):
                altitude = altitude_above_ground(
                    self._ground_down_m, down
                )
        telemetry_fresh = bool(
            self._vehicle_status_receipt_s is not None
            and self._vehicle_odometry_receipt_s is not None
            and now - self._vehicle_status_receipt_s <= 0.75
            and now - self._vehicle_odometry_receipt_s <= 0.25
        )
        mux = self._mux_status
        astar_selected = bool(
            mux is not None and mux.active_source == "ASTAR_EXPERT"
        )
        source_valid = bool(
            astar_selected
            and mux.selected_command_valid
            and not mux.hold_active
        )
        gate_safe = bool(
            self._gate_status is not None
            and self._gate_status.safe_to_forward
            and self._gate_status.state == "SAFE_TO_FORWARD"
        )
        gate_ready = bool(
            self._gate_status is not None
            and not self._gate_status.enable_requested
            and self._gate_status.state == "READY_DISABLED"
        )
        stream = self._stream_status
        stream_stable = bool(
            stream is not None
            and stream.streaming
            and stream.state == "STREAMING"
        )
        tracking = self._tracking_status
        tracking_active = bool(
            tracking is not None
            and tracking.state in {"TRACKING", "GOAL_SETTLING", "GOAL_HOLD"}
        )
        goal_reached = bool(
            tracking is not None and tracking.state == "GOAL_HOLD"
        )
        follower_valid = bool(
            tracking is not None and tracking.command_valid
        )
        distance = self._goal_distance()
        landed = bool(
            self._land_detected is not None and self._land_detected.landed
        )
        return FlightEvidence(
            pipeline_ready=(
                self._pipeline_ready() and self._scene_kind == "takeoff"
            ),
            mission_trajectory_ready=(
                self._pipeline_ready() and self._scene_kind == "mission"
            ),
            follower_command_valid=follower_valid,
            astar_selected=astar_selected,
            output_gate_ready=gate_ready,
            output_gate_safe=gate_safe,
            stream_stable=stream_stable,
            stream_rate_hz=(
                0.0 if stream is None else stream.observed_rate_hz
            ),
            offboard_active=offboard,
            vehicle_armed=armed,
            altitude_m=altitude,
            tracking_active=tracking_active,
            goal_reached=goal_reached,
            goal_distance_m=distance,
            landed=landed,
            telemetry_fresh=telemetry_fresh,
            source_valid=source_valid,
            failsafe=failsafe,
            fatal_command_ack=self._fatal_command_ack,
            environment_valid=self._external_environment_valid(now),
        )

    def _goal_distance(self) -> float:
        if self._vehicle_odometry is None or self._expected_goal is None:
            return math.inf
        position = tuple(
            float(value) for value in self._vehicle_odometry.position
        )
        if not all(math.isfinite(value) for value in position):
            return math.inf
        return math.dist(position, self._expected_goal)

    def _tick(self) -> None:
        now = self._now_seconds()
        if (
            self.machine.mission_enable_requested
            and not self._scene_kind
            and self._vehicle_odometry is not None
        ):
            self._publish_takeoff_scene()
        evidence = self._evidence(now)
        previous = self.machine.state
        decision = self.machine.step(now, evidence)
        if decision.state != previous:
            self.get_logger().info(
                f"FLIGHT_TRANSITION {previous.value}->{decision.state.value}"
            )
        for action in decision.actions:
            self._execute_action(action, now)
        self._status_publisher.publish(self._status_message(evidence))

    def _execute_action(self, action: str, now: float) -> None:
        if action == "SELECT_ASTAR":
            request = SetControlSource.Request()
            request.source = "ASTAR_EXPERT"
            self._request_service(action, self._mux_client, request, now)
        elif action == "SELECT_HOLD":
            request = SetControlSource.Request()
            request.source = "HOLD"
            self._request_service(action, self._mux_client, request, now)
        elif action in {"ENABLE_OUTPUT_GATE", "DISABLE_OUTPUT_GATE"}:
            request = SetPx4OutputEnable.Request()
            request.enable = action == "ENABLE_OUTPUT_GATE"
            self._request_service(action, self._gate_client, request, now)
        elif action in {"ENABLE_STREAM", "DISABLE_STREAM"}:
            request = SetPx4StreamEnable.Request()
            request.enable = action == "ENABLE_STREAM"
            self._request_service(action, self._stream_client, request, now)
        elif action in {"START_TRACKING", "STOP_TRACKING"}:
            request = SetBool.Request()
            request.data = action == "START_TRACKING"
            self._request_service(action, self._tracking_client, request, now)
        elif action == "PUBLISH_MISSION_SCENE":
            self._publish_mission_scene()
        elif action == "SEND_OFFBOARD":
            self._publish_vehicle_command(
                self._VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                now,
                param1=1.0,
                param2=6.0,
            )
        elif action == "SEND_ARM":
            self._publish_vehicle_command(
                self._VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                now,
                param1=1.0,
            )
        elif action == "SEND_LAND":
            land_command = self._VehicleCommand.VEHICLE_CMD_NAV_LAND
            if vehicle_command_was_accepted(
                self._last_command_ack, land_command
            ):
                return
            self._landing_commanded = True
            self._publish_vehicle_command(
                land_command,
                now,
            )

    def _request_service(self, key, client, request, now: float) -> bool:
        previous = self._last_action_time.get(key, -math.inf)
        if now - previous < self.config.command_retry_s:
            return False
        future = self._pending_services.get(key)
        if future is not None and not future.done():
            return False
        if not client.service_is_ready():
            return False
        self._last_action_time[key] = now
        future = client.call_async(request)
        self._pending_services[key] = future
        future.add_done_callback(
            lambda completed, action=key: self._service_done(action, completed)
        )
        return True

    def _service_done(self, action: str, future) -> None:
        try:
            response = future.result()
        except Exception as error:  # noqa: BLE001 - ROS future boundary
            self.get_logger().error(f"{action} service failed: {error}")
            return
        accepted = getattr(response, "accepted", None)
        if accepted is None:
            accepted = getattr(response, "success", False)
        message = getattr(response, "status_message", None)
        if message is None:
            message = getattr(response, "message", "")
        if accepted:
            self.get_logger().info(f"{action} accepted: {message}")
        else:
            self.get_logger().warning(f"{action} rejected: {message}")

    def _publish_vehicle_command(
        self,
        command: int,
        now: float,
        **parameters: float,
    ) -> None:
        key = f"vehicle_command_{command}"
        request = SendPx4VehicleCommand.Request()
        for index in range(1, 8):
            value = float(parameters.get(f"param{index}", 0.0))
            setattr(request, f"param{index}", value)
        request.command = int(command)
        requested = self._request_service(
            key, self._vehicle_command_client, request, now
        )
        if requested:
            self._vehicle_command_count += 1
            self.get_logger().info(f"VEHICLE_COMMAND command={command}")

    def _publish_takeoff_scene(self) -> None:
        if self._vehicle_odometry is None or self._scene_kind == "takeoff":
            return
        north = float(self._vehicle_odometry.position[0])
        east = float(self._vehicle_odometry.position[1])
        if not math.isfinite(north) or not math.isfinite(east):
            return
        if self._ground_down_m is None:
            down = float(self._vehicle_odometry.position[2])
            if not math.isfinite(down):
                return
            self._ground_down_m = down
        self._initial_north_east = (north, east)
        takeoff_north = north + float(
            self.get_parameter("takeoff_path_north_m").value
        )
        self._publish_scene("takeoff", north, east, takeoff_north, east)

    def _publish_mission_scene(self) -> None:
        if self._vehicle_odometry is None or self._scene_kind == "mission":
            return
        north = float(self._vehicle_odometry.position[0])
        east = float(self._vehicle_odometry.position[1])
        if self._uses_external_scene():
            self._publish_external_mission_scene(north, east)
            return
        goal_north = float(self.get_parameter("goal_north_m").value)
        goal_east = float(self.get_parameter("goal_east_m").value)
        self._publish_scene(
            "mission", north, east, goal_north, goal_east
        )

    def _publish_external_mission_scene(
        self, start_north: float, start_east: float
    ) -> None:
        now = self._now_seconds()
        if not self._external_environment_valid(now):
            return
        assert self._external_goal is not None
        assert self._external_obstacles is not None
        goal = self._external_goal.pose.position
        goal_north = float(goal.y)
        goal_east = float(goal.x)
        if self._ground_down_m is None:
            return
        target_down = -self.config.takeoff_altitude_m
        self._scene_kind = "mission"
        self._expected_goal = (goal_north, goal_east, target_down)
        self._planner_path_valid = False
        self._planner_status = ""
        self._bspline_valid = False
        self._trajectory_valid = False
        self._trajectory_goal = None
        stamp = self.get_clock().now().to_msg()
        obstacles = ObstacleArray()
        obstacles.header.stamp = stamp
        obstacles.header.frame_id = "isaac_world"
        obstacles.obstacles = self._external_obstacles.obstacles
        self._obstacles_publisher.publish(obstacles)
        self._start_publisher.publish(
            self._scene_pose(start_north, start_east, target_down, stamp)
        )
        self._goal_publisher.publish(
            self._scene_pose(goal_north, goal_east, target_down, stamp)
        )
        self.get_logger().info(
            "SCENE kind=mission source=isaac "
            f"id={self._external_scene_id} "
            f"revision={self._external_scene_revision} "
            f"obstacles={len(obstacles.obstacles)} "
            f"goal=({goal_north:.3f},{goal_east:.3f})"
        )

    def _publish_scene(
        self,
        kind: str,
        start_north: float,
        start_east: float,
        goal_north: float,
        goal_east: float,
    ) -> None:
        if self._ground_down_m is None:
            return
        target_down = -self.config.takeoff_altitude_m
        self._scene_kind = kind
        self._expected_goal = (
            goal_north,
            goal_east,
            target_down,
        )
        self._planner_path_valid = False
        self._planner_status = ""
        self._bspline_valid = False
        self._trajectory_valid = False
        self._trajectory_goal = None
        stamp = self.get_clock().now().to_msg()
        obstacles = ObstacleArray()
        obstacles.header.stamp = stamp
        obstacles.header.frame_id = "isaac_world"
        self._obstacles_publisher.publish(obstacles)
        self._start_publisher.publish(
            self._scene_pose(start_north, start_east, target_down, stamp)
        )
        self._goal_publisher.publish(
            self._scene_pose(goal_north, goal_east, target_down, stamp)
        )
        self.get_logger().info(
            f"SCENE kind={kind} start=({start_north:.3f},{start_east:.3f}) "
            f"goal=({goal_north:.3f},{goal_east:.3f})"
        )

    def _scene_pose(
        self, north: float, east: float, target_down: float, stamp
    ) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "isaac_world"
        pose.pose.position.x = east
        pose.pose.position.y = north
        pose.pose.position.z = -target_down
        pose.pose.orientation.w = 1.0
        return pose

    def _status_message(self, evidence: FlightEvidence) -> Px4FlightStatus:
        message = Px4FlightStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "px4_ned"
        message.state = self.machine.state.value
        message.mission_enable_requested = (
            self.machine.mission_enable_requested
        )
        message.planner_path_valid = self._planner_path_valid
        message.bspline_valid = self._bspline_valid
        message.trajectory_valid = self._trajectory_valid
        message.follower_command_valid = evidence.follower_command_valid
        message.astar_selected = evidence.astar_selected
        message.output_gate_safe = evidence.output_gate_safe
        message.stream_stable = evidence.stream_stable
        message.offboard_active = evidence.offboard_active
        message.vehicle_armed = evidence.vehicle_armed
        message.takeoff_altitude_reached = (
            evidence.altitude_m
            >= self.config.takeoff_altitude_m
            - self.config.takeoff_altitude_tolerance_m
        )
        message.tracking_active = evidence.tracking_active
        message.goal_reached = evidence.goal_reached
        message.landing_commanded = self._landing_commanded
        message.landed = evidence.landed
        if self._vehicle_odometry is not None:
            message.position_north_m = float(
                self._vehicle_odometry.position[0]
            )
            message.position_east_m = float(self._vehicle_odometry.position[1])
            message.position_down_m = float(self._vehicle_odometry.position[2])
        message.altitude_m = evidence.altitude_m
        message.goal_distance_m = evidence.goal_distance_m
        message.stream_rate_hz = evidence.stream_rate_hz
        message.vehicle_command_count = self._vehicle_command_count
        message.transition_count = self.machine.transition_count
        message.last_command_ack = self._last_command_ack
        message.failure_reason = self.machine.failure_reason
        environment_valid = str(
            self._external_environment_valid(self._now_seconds())
        ).lower()
        message.status_message = (
            f"scene={self._scene_kind}|trajectory_source="
            f"{self._trajectory_source or 'none'}|sitl_guard="
            f"{str(self._sitl_guard_valid()).lower()}|external_scene="
            f"{str(self._uses_external_scene()).lower()}|environment_valid="
            f"{environment_valid}"
        )
        return message


def main(args=None) -> int:
    """Run the explicitly enabled, SITL-guarded flight supervisor."""
    rclpy.init(args=args)
    node = Px4SitlFlightSupervisorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    main()
