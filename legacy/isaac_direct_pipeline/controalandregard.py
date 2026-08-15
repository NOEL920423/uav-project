"""
Isaac Sim / WebRTC PX4 drive-style keyboard controller + CSV demonstration logger.

Run this script INSIDE Isaac Sim:
    Window > Script Editor > paste this script > Run

After it starts, click the WebRTC viewport and control the PX4 drone directly
from the WebRTC window.

Controls:
    Up / Down      : hold to fly forward / backward
    Left / Right   : hold to steer left / right
    W / S          : optional cruise forward / backward toggle
    X              : neutral forward/backward cruise
    Q / E          : side-slip left / side-slip right
    U / J          : up / down
    Space          : brake / hover
    1 / 2 / 3      : slow / normal / fast
    G              : start / stop CSV recording
    M              : mark current demo as success
    F              : mark current demo as failure
    C              : toggle collision label
    H              : print help
    P              : print current status
    L              : land and stop controller

Notes:
    - This uses Omniverse/Isaac keyboard events through carb.input.
    - It reuses the PX4 MAVLink OFFBOARD flow: connect, warm-up setpoints,
      set OFFBOARD, arm, climb to initial altitude, then send body velocity.
    - CSV files are saved under ~/uav-project/uav_demo_logs by default.
    - The recorded command action is the actual smoothed body-frame velocity
      command sent to PX4: cmd_vx_body, cmd_vy_body, cmd_vz_body, cmd_yaw_rate.
"""

from __future__ import annotations

import csv
import math
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import carb
    import carb.input
    import omni.appwindow
    import omni.kit.app
except Exception as import_error:  # pragma: no cover - only meaningful inside Isaac Sim
    raise RuntimeError(
        "This script must be run inside Isaac Sim / Omniverse Kit, not regular Python."
    ) from import_error

try:
    from pymavlink import mavutil
except Exception as import_error:  # pragma: no cover - depends on Isaac Python environment
    mavutil = None
    _PYMAVLINK_IMPORT_ERROR = import_error
else:
    _PYMAVLINK_IMPORT_ERROR = None


# =============================================================================
# PX4 connection and flight settings
# =============================================================================

CONNECTION_STRINGS = [
    "udpout:127.0.0.1:14580",
    "udpout:127.0.0.1:18570",
    "udpin:0.0.0.0:14550",
    "udpin:0.0.0.0:14540",
]

TARGET_ALTITUDE_M = 2.0
TAKEOFF_TIMEOUT_S = 10.0
TAKEOFF_Z_TOLERANCE_M = 0.35

CONTROL_DT = 0.05  # 20 Hz target output rate.

SPEED_MODES = {
    "slow": {"xy": 0.35, "z": 0.25, "yaw": math.radians(25.0)},
    "normal": {"xy": 0.70, "z": 0.40, "yaw": math.radians(40.0)},
    "fast": {"xy": 1.10, "z": 0.60, "yaw": math.radians(65.0)},
}
DEFAULT_SPEED_MODE = "normal"

FORWARD_CRUISE_VALUE = 1.0
BACKWARD_CRUISE_VALUE = -0.65
TURN_FORWARD_SPEED_RATIO = 0.92
STEERING_AUTO_FORWARD = True
STEERING_FORWARD_ASSIST_RATIO = 0.55
STRAFE_SPEED_RATIO = 0.65

MAX_XY_ACCEL_MPS2 = 1.2
MAX_Z_ACCEL_MPS2 = 0.8
MAX_YAW_ACCEL_RADPS2 = math.radians(90.0)

HOVER_SECONDS_BEFORE_LAND = 0.6
HOVER_SECONDS_AFTER_LAND_COMMAND = 1.5

# =============================================================================
# Demonstration logging settings
# =============================================================================

ENABLE_LOGGING = True
AUTO_START_RECORDING_AFTER_TAKEOFF = True

LOG_DIR = str(Path.home() / "uav-project" / "uav_demo_logs")
LOG_PREFIX = "manual_demo"

# Your easy environment definition.
# Pads are on the ground. Flight target height is TARGET_ALTITUDE_M.
START_PAD_ISAAC = [0.0, 0.0, 0.0]
GOAL_PAD_ISAAC = [3.0, 5.0, 0.0]

# Used only for logging the relative obstacle vector.
# Update this to match the blue obstacle Transform > Position in Isaac Sim.
OBSTACLE_POS_ISAAC = [1.5, 2.5, 0.5]
OBSTACLE_RADIUS_M = 0.80

# Treat goal and obstacle as offsets from the UAV's local start position.
# This is usually safer in Pegasus/PX4 because the PX4 local origin may not
# perfectly match the Isaac world origin after spawn.
USE_RELATIVE_TO_START_POSITION = True

# Coordinate mapping used only for logging target/obstacle relative vectors.
# If you later discover that Isaac x/y and PX4 local NED x/y do not align,
# adjust these and collect new logs.
ISAAC_X_TO_NED_X_SIGN = 1.0
ISAAC_Y_TO_NED_Y_SIGN = 1.0
SWAP_XY_FOR_LOGGING = False

SUCCESS_RADIUS_XY_M = 0.50
SUCCESS_HEIGHT_TOLERANCE_M = 0.45

# MAVLink type masks.
# Position + yaw, ignore velocity, acceleration and yaw rate.
POSITION_YAW_TYPE_MASK = 2552
# Body-frame velocity + yaw rate, ignore position, acceleration and yaw angle.
BODY_VELOCITY_YAW_RATE_TYPE_MASK = 1479

PX4_MAIN_MODE = {
    1: "MANUAL",
    2: "ALTCTL",
    3: "POSCTL",
    4: "AUTO",
    5: "ACRO",
    6: "OFFBOARD",
    7: "STABILIZED",
    8: "RATTITUDE",
}

PX4_AUTO_SUB_MODE = {
    1: "READY",
    2: "TAKEOFF",
    3: "LOITER",
    4: "MISSION",
    5: "RTL",
    6: "LAND",
    7: "RTGS",
    8: "FOLLOW_TARGET",
    9: "PRECLAND",
}


def log(message: str) -> None:
    print(f"[PX4 WebRTC Keyboard Logger] {message}")
    try:
        carb.log_info(f"[PX4 WebRTC Keyboard Logger] {message}")
    except Exception:
        pass


def px4_custom_mode(main_mode: int, sub_mode: int = 0) -> int:
    return (sub_mode << 24) | (main_mode << 16)


def decode_px4_custom_mode(custom_mode: int) -> str:
    main_mode = (custom_mode >> 16) & 0xFF
    sub_mode = (custom_mode >> 24) & 0xFF
    main_name = PX4_MAIN_MODE.get(main_mode, f"UNKNOWN_MAIN_{main_mode}")
    if main_mode == 4:
        sub_name = PX4_AUTO_SUB_MODE.get(sub_mode, f"UNKNOWN_AUTO_SUB_{sub_mode}")
        return f"{main_name}.{sub_name}"
    return main_name


def approach(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(current + max_delta, target)
    if current > target:
        return max(current - max_delta, target)
    return current


def distance_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def isaac_offset_to_ned_offset(x_isaac: float, y_isaac: float, z_up: float) -> tuple[float, float, float]:
    if SWAP_XY_FOR_LOGGING:
        ned_x = ISAAC_Y_TO_NED_Y_SIGN * y_isaac
        ned_y = ISAAC_X_TO_NED_X_SIGN * x_isaac
    else:
        ned_x = ISAAC_X_TO_NED_X_SIGN * x_isaac
        ned_y = ISAAC_Y_TO_NED_Y_SIGN * y_isaac

    ned_z = -z_up
    return ned_x, ned_y, ned_z


def make_log_path() -> Path:
    log_dir = Path(LOG_DIR).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{LOG_PREFIX}_{stamp}.csv"


# Omniverse / carb.input may report arrow keys with slightly different names
# depending on version and platform. Normalize the common variants here.
KEY_ALIASES = {
    "UP": "ARROW_UP",
    "DOWN": "ARROW_DOWN",
    "LEFT": "ARROW_LEFT",
    "RIGHT": "ARROW_RIGHT",
    "KEY_UP": "ARROW_UP",
    "KEY_DOWN": "ARROW_DOWN",
    "KEY_LEFT": "ARROW_LEFT",
    "KEY_RIGHT": "ARROW_RIGHT",
    "ARROWUP": "ARROW_UP",
    "ARROWDOWN": "ARROW_DOWN",
    "ARROWLEFT": "ARROW_LEFT",
    "ARROWRIGHT": "ARROW_RIGHT",
    "UP_ARROW": "ARROW_UP",
    "DOWN_ARROW": "ARROW_DOWN",
    "LEFT_ARROW": "ARROW_LEFT",
    "RIGHT_ARROW": "ARROW_RIGHT",
}

MOTION_HOLD_KEYS = {
    "ARROW_UP",
    "ARROW_DOWN",
    "ARROW_LEFT",
    "ARROW_RIGHT",
    "A",
    "D",
    "U",
    "J",
    "Q",
    "E",
}


def canonical_key_name(key_name: str) -> str:
    return KEY_ALIASES.get(key_name.upper(), key_name.upper())


class IsaacWebRTCPX4DriveController:
    def __init__(self) -> None:
        self.master = None
        self.ready = False
        self.started = False
        self.stop_requested = False
        self.state_text = "created"

        self._input = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        if self._app_window is None:
            raise RuntimeError("No default Isaac/Kit app window was found.")
        self._keyboard = self._app_window.get_keyboard()
        self._keyboard_sub = None
        self._update_sub = None
        self._init_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self.active_keys: set[str] = set()
        self.speed_mode = DEFAULT_SPEED_MODE
        self.forward_cruise_cmd = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.yaw_rate = 0.0

        self.telemetry = {
            "pos_x": 0.0,
            "pos_y": 0.0,
            "pos_z": 0.0,
            "vel_x": 0.0,
            "vel_y": 0.0,
            "vel_z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "has_position": False,
            "has_attitude": False,
        }

        self.demo_origin_ned = (0.0, 0.0, -TARGET_ALTITUDE_M)
        self.goal_ned = (0.0, 0.0, -TARGET_ALTITUDE_M)
        self.obstacle_ned = (0.0, 0.0, 0.0)

        self.recording = False
        self.log_path: Optional[Path] = None
        self.log_file = None
        self.csv_writer = None
        self.recording_start_time = 0.0
        self.sample_index = 0
        self.manual_label = "unlabeled"
        self.manual_collision = False

        self._last_control_time = time.monotonic()
        self._last_status_time = 0.0
        self._last_command_print_time = 0.0

        if mavutil is not None:
            self.MAV_FRAME_BODY_NED = getattr(mavutil.mavlink, "MAV_FRAME_BODY_NED", 8)
        else:
            self.MAV_FRAME_BODY_NED = 8

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------
    def start(self) -> None:
        if mavutil is None:
            raise RuntimeError(
                "pymavlink is not installed in Isaac Sim's Python environment.\n"
                "Install it once with Isaac's python.sh, then run this script again.\n"
                f"Original import error: {_PYMAVLINK_IMPORT_ERROR}"
            )

        if self.started:
            log("Controller is already started.")
            return

        self.started = True
        self.stop_requested = False
        self.state_text = "starting"

        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="px4_webrtc_keyboard_drive_control_logger_update",
        )

        self._init_thread = threading.Thread(
            target=self._initialize_px4,
            name="PX4 WebRTC keyboard logger initialization",
            daemon=True,
        )
        self._init_thread.start()

        self.print_help()
        log("Controller started. Click the WebRTC viewport, then use the keyboard there.")

    def stop(self, land_vehicle: bool = True) -> None:
        self.stop_requested = True
        self.ready = False

        if self._keyboard_sub is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            except Exception:
                pass
            self._keyboard_sub = None

        if self._update_sub is not None:
            self._update_sub = None

        if land_vehicle and self.master is not None:
            try:
                self._send_hover_for_seconds(HOVER_SECONDS_BEFORE_LAND)
                self._land()
                time.sleep(HOVER_SECONDS_AFTER_LAND_COMMAND)
            except Exception as error:
                log(f"Stop/land warning: {error}")

        self._stop_recording()

        self.started = False
        self.state_text = "stopped"
        log("Controller stopped.")

    # ---------------------------------------------------------------------
    # PX4 / MAVLink setup
    # ---------------------------------------------------------------------
    def _initialize_px4(self) -> None:
        try:
            self.state_text = "connecting"
            master = self._connect_to_px4()
            if self.stop_requested:
                return

            self.master = master
            x_target = 0.0
            y_target = 0.0
            z_target = -TARGET_ALTITUDE_M
            yaw_target = 0.0

            self.state_text = "warmup"
            log("Sending warm-up position setpoints...")
            for _ in range(80):
                if self.stop_requested:
                    return
                self._send_gcs_heartbeat()
                self._send_position_setpoint(x_target, y_target, z_target, yaw_target)
                self._read_latest_telemetry_nonblocking()
                time.sleep(CONTROL_DT)

            self.state_text = "offboard"
            if not self._set_offboard_mode(x_target, y_target, z_target, yaw_target):
                self.state_text = "offboard_failed"
                log("OFFBOARD failed. Controller will not send velocity commands.")
                return

            self.state_text = "arming"
            if not self._arm(x_target, y_target, z_target, yaw_target, force=False):
                log("Normal arming failed. Trying force arm for simulator only...")
                if not self._arm(x_target, y_target, z_target, yaw_target, force=True):
                    self.state_text = "arming_failed"
                    log("Arming failed even with force arm.")
                    return

            self.state_text = "takeoff"
            self._wait_for_initial_altitude(z_target, yaw_target)

            with self._lock:
                self.active_keys.clear()
                self.forward_cruise_cmd = 0.0
                self.vx = self.vy = self.vz = self.yaw_rate = 0.0
                self._setup_demo_frame_locked()

            self.ready = True
            self.state_text = "ready"

            if ENABLE_LOGGING and AUTO_START_RECORDING_AFTER_TAKEOFF:
                self._start_recording()

            log("Ready. WebRTC keyboard control is active.")

        except Exception as error:
            self.state_text = "error"
            log(f"Initialization error: {error}")
            traceback.print_exc()

    def _connect_to_px4(self):
        for connection_string in CONNECTION_STRINGS:
            if self.stop_requested:
                raise RuntimeError("Stop requested during connection.")

            log(f"Trying {connection_string} ...")
            try:
                master = mavutil.mavlink_connection(connection_string)
            except OSError as error:
                log(f"Could not open {connection_string}: {error}")
                continue

            # Send a few GCS heartbeats so PX4 can discover this endpoint.
            for _ in range(20):
                master.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0,
                    0,
                    0,
                )
                time.sleep(0.1)

            heartbeat = master.wait_heartbeat(timeout=8)
            if heartbeat is not None:
                log("Heartbeat received.")
                log(f"Connected via: {connection_string}")
                log(f"Target system: {master.target_system}")
                log(f"Target component: {master.target_component}")
                return master

            log("No heartbeat on this port.")

        raise RuntimeError("No PX4 heartbeat received.")

    def _send_gcs_heartbeat(self) -> None:
        self.master.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            0,
        )

    def _send_position_setpoint(self, x_m: float, y_m: float, z_m: float, yaw_rad: float) -> None:
        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            POSITION_YAW_TYPE_MASK,
            float(x_m),
            float(y_m),
            float(z_m),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(yaw_rad),
            0.0,
        )

    def _send_velocity_setpoint(self, vx_mps: float, vy_mps: float, vz_mps: float, yaw_rate_radps: float) -> None:
        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.master.target_system,
            self.master.target_component,
            self.MAV_FRAME_BODY_NED,
            BODY_VELOCITY_YAW_RATE_TYPE_MASK,
            0.0,
            0.0,
            0.0,
            float(vx_mps),
            float(vy_mps),
            float(vz_mps),
            0.0,
            0.0,
            0.0,
            0.0,
            float(yaw_rate_radps),
        )

    def _set_offboard_mode(self, x_m: float, y_m: float, z_m: float, yaw_rad: float, timeout: float = 8.0) -> bool:
        offboard_custom_mode = px4_custom_mode(main_mode=6, sub_mode=0)
        log(f"Setting OFFBOARD using SET_MODE custom_mode={offboard_custom_mode}")

        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            offboard_custom_mode,
        )

        deadline = time.time() + timeout
        while time.time() < deadline and not self.stop_requested:
            self._send_gcs_heartbeat()
            self._send_position_setpoint(x_m, y_m, z_m, yaw_rad)
            self._read_latest_telemetry_nonblocking()

            message = self.master.recv_match(
                type=["HEARTBEAT", "STATUSTEXT", "COMMAND_ACK"],
                blocking=True,
                timeout=0.5,
            )
            self._print_message(message)

            if message is not None and message.get_type() == "HEARTBEAT":
                if decode_px4_custom_mode(message.custom_mode) == "OFFBOARD":
                    log("Mode changed to OFFBOARD.")
                    return True

        return False

    def _arm(
        self,
        x_m: float,
        y_m: float,
        z_m: float,
        yaw_rad: float,
        force: bool = False,
        timeout: float = 8.0,
    ) -> bool:
        log("Arming...")
        force_code = 21196.0 if force else 0.0
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1.0,
            force_code,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        deadline = time.time() + timeout
        while time.time() < deadline and not self.stop_requested:
            self._send_gcs_heartbeat()
            self._send_position_setpoint(x_m, y_m, z_m, yaw_rad)
            self._read_latest_telemetry_nonblocking()

            message = self.master.recv_match(
                type=["HEARTBEAT", "STATUSTEXT", "COMMAND_ACK"],
                blocking=True,
                timeout=0.5,
            )
            self._print_message(message)

            if message is not None and message.get_type() == "HEARTBEAT":
                armed = bool(message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                if armed:
                    log("Vehicle is armed.")
                    return True

        return False

    def _wait_for_initial_altitude(self, z_target: float, yaw_target: float) -> bool:
        log(f"Climbing to initial altitude: {-z_target:.1f} m ...")
        deadline = time.time() + TAKEOFF_TIMEOUT_S
        saw_local_position = False

        while time.time() < deadline and not self.stop_requested:
            self._send_gcs_heartbeat()
            self._send_position_setpoint(0.0, 0.0, z_target, yaw_target)
            self._read_latest_telemetry_nonblocking()

            with self._lock:
                saw_local_position = saw_local_position or self.telemetry["has_position"]
                current_z = self.telemetry["pos_z"]

            if saw_local_position and abs(current_z - z_target) <= TAKEOFF_Z_TOLERANCE_M:
                log(f"Initial altitude reached: z={current_z:.2f} m target={z_target:.2f} m")
                return True

            time.sleep(CONTROL_DT)

        if not saw_local_position:
            log("Warning: no LOCAL_POSITION_NED received during takeoff wait.")
        log("Initial altitude wait finished. Continuing with velocity control.")
        return False

    def _send_hover_for_seconds(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline:
            self._send_gcs_heartbeat()
            self._send_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
            self._read_latest_telemetry_nonblocking()
            time.sleep(CONTROL_DT)

    def _land(self) -> None:
        log("Landing...")
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0,
            0.0,
            0.0,
            0.0,
            math.nan,
            math.nan,
            math.nan,
            0.0,
        )

    # ---------------------------------------------------------------------
    # Telemetry and logging
    # ---------------------------------------------------------------------
    def _read_latest_telemetry_nonblocking(self) -> None:
        if self.master is None:
            return

        # Drain a limited number of messages per update so state does not lag.
        for _ in range(50):
            message = self.master.recv_match(
                type=["LOCAL_POSITION_NED", "ATTITUDE", "HEARTBEAT", "STATUSTEXT"],
                blocking=False,
            )
            if message is None:
                break

            msg_type = message.get_type()
            with self._lock:
                if msg_type == "LOCAL_POSITION_NED":
                    self.telemetry["pos_x"] = float(message.x)
                    self.telemetry["pos_y"] = float(message.y)
                    self.telemetry["pos_z"] = float(message.z)
                    self.telemetry["vel_x"] = float(message.vx)
                    self.telemetry["vel_y"] = float(message.vy)
                    self.telemetry["vel_z"] = float(message.vz)
                    self.telemetry["has_position"] = True
                elif msg_type == "ATTITUDE":
                    self.telemetry["roll"] = float(message.roll)
                    self.telemetry["pitch"] = float(message.pitch)
                    self.telemetry["yaw"] = float(message.yaw)
                    self.telemetry["has_attitude"] = True

            self._print_message_throttled(message)

    def _setup_demo_frame_locked(self) -> None:
        origin_x = self.telemetry["pos_x"] if USE_RELATIVE_TO_START_POSITION else 0.0
        origin_y = self.telemetry["pos_y"] if USE_RELATIVE_TO_START_POSITION else 0.0
        origin_z = self.telemetry["pos_z"] if self.telemetry["has_position"] else -TARGET_ALTITUDE_M

        # Keep the goal altitude at the current flight altitude.
        goal_dx, goal_dy, _ = isaac_offset_to_ned_offset(
            GOAL_PAD_ISAAC[0] - START_PAD_ISAAC[0],
            GOAL_PAD_ISAAC[1] - START_PAD_ISAAC[1],
            0.0,
        )

        obs_dx, obs_dy, obs_z = isaac_offset_to_ned_offset(
            OBSTACLE_POS_ISAAC[0] - START_PAD_ISAAC[0],
            OBSTACLE_POS_ISAAC[1] - START_PAD_ISAAC[1],
            OBSTACLE_POS_ISAAC[2] - START_PAD_ISAAC[2],
        )

        self.demo_origin_ned = (origin_x, origin_y, origin_z)
        self.goal_ned = (origin_x + goal_dx, origin_y + goal_dy, origin_z)
        self.obstacle_ned = (origin_x + obs_dx, origin_y + obs_dy, obs_z)

        log(
            "Demo frame set. "
            f"origin=({self.demo_origin_ned[0]:.2f}, {self.demo_origin_ned[1]:.2f}, {self.demo_origin_ned[2]:.2f}), "
            f"goal=({self.goal_ned[0]:.2f}, {self.goal_ned[1]:.2f}, {self.goal_ned[2]:.2f})"
        )

    def _start_recording(self) -> None:
        if not ENABLE_LOGGING:
            log("Logging is disabled.")
            return

        with self._lock:
            if self.recording:
                log(f"Already recording: {self.log_path}")
                return

            self.log_path = make_log_path()
            self.log_file = open(self.log_path, "w", newline="")
            fieldnames = [
                "time_wall",
                "demo_time",
                "sample_index",
                "state_text",
                "manual_label",
                "manual_collision",
                "success_auto",
                "pos_x_ned",
                "pos_y_ned",
                "pos_z_ned",
                "vel_x_ned",
                "vel_y_ned",
                "vel_z_ned",
                "roll",
                "pitch",
                "yaw",
                "origin_x_ned",
                "origin_y_ned",
                "origin_z_ned",
                "goal_x_ned",
                "goal_y_ned",
                "goal_z_ned",
                "rel_goal_x_ned",
                "rel_goal_y_ned",
                "rel_goal_z_ned",
                "distance_goal_xy",
                "distance_goal_3d",
                "obstacle_x_ned",
                "obstacle_y_ned",
                "obstacle_z_ned",
                "rel_obstacle_x_ned",
                "rel_obstacle_y_ned",
                "rel_obstacle_z_ned",
                "distance_obstacle_xy",
                "cmd_vx_body",
                "cmd_vy_body",
                "cmd_vz_body",
                "cmd_yaw_rate",
                "forward_cruise_cmd",
                "speed_mode",
                "active_keys",
                "has_motion_command",
            ]
            self.csv_writer = csv.DictWriter(self.log_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            self.recording_start_time = time.time()
            self.sample_index = 0
            self.manual_label = "unlabeled"
            self.manual_collision = False
            self.recording = True

        log(f"Recording started: {self.log_path}")

    def _stop_recording(self) -> None:
        with self._lock:
            if not self.recording and self.log_file is None:
                return

            path = self.log_path
            self.recording = False

            try:
                if self.log_file is not None:
                    self.log_file.flush()
                    self.log_file.close()
            except Exception as error:
                log(f"Logger close warning: {error}")
            finally:
                self.log_file = None
                self.csv_writer = None

        log(f"Recording stopped. Saved: {path}")

    def _toggle_recording(self) -> None:
        if self.recording:
            self._stop_recording()
        else:
            with self._lock:
                self._setup_demo_frame_locked()
            self._start_recording()

    def _mark_success(self) -> None:
        with self._lock:
            self.manual_label = "success"
        log("Manual label set to success.")

    def _mark_failure(self) -> None:
        with self._lock:
            self.manual_label = "failure"
        log("Manual label set to failure.")

    def _toggle_collision_label(self) -> None:
        with self._lock:
            self.manual_collision = not self.manual_collision
            state = self.manual_collision
        log(f"Manual collision label toggled to {state}.")

    def _write_log_row_locked(self, has_motion_command: bool) -> None:
        if not self.recording or self.csv_writer is None:
            return

        px = self.telemetry["pos_x"]
        py = self.telemetry["pos_y"]
        pz = self.telemetry["pos_z"]
        vx_ned = self.telemetry["vel_x"]
        vy_ned = self.telemetry["vel_y"]
        vz_ned = self.telemetry["vel_z"]
        roll = self.telemetry["roll"]
        pitch = self.telemetry["pitch"]
        yaw = self.telemetry["yaw"]

        origin_x, origin_y, origin_z = self.demo_origin_ned
        goal_x, goal_y, goal_z = self.goal_ned
        obs_x, obs_y, obs_z = self.obstacle_ned

        distance_goal_xy = distance_xy(px, py, goal_x, goal_y)
        distance_goal_3d = distance_3d(px, py, pz, goal_x, goal_y, goal_z)
        height_ok = abs(pz - goal_z) <= SUCCESS_HEIGHT_TOLERANCE_M
        success_auto = distance_goal_xy <= SUCCESS_RADIUS_XY_M and height_ok

        distance_obstacle_xy = distance_xy(px, py, obs_x, obs_y)

        self.csv_writer.writerow({
            "time_wall": time.time(),
            "demo_time": time.time() - self.recording_start_time,
            "sample_index": self.sample_index,
            "state_text": self.state_text,
            "manual_label": self.manual_label,
            "manual_collision": int(self.manual_collision),
            "success_auto": int(success_auto),
            "pos_x_ned": px,
            "pos_y_ned": py,
            "pos_z_ned": pz,
            "vel_x_ned": vx_ned,
            "vel_y_ned": vy_ned,
            "vel_z_ned": vz_ned,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "origin_x_ned": origin_x,
            "origin_y_ned": origin_y,
            "origin_z_ned": origin_z,
            "goal_x_ned": goal_x,
            "goal_y_ned": goal_y,
            "goal_z_ned": goal_z,
            "rel_goal_x_ned": goal_x - px,
            "rel_goal_y_ned": goal_y - py,
            "rel_goal_z_ned": goal_z - pz,
            "distance_goal_xy": distance_goal_xy,
            "distance_goal_3d": distance_goal_3d,
            "obstacle_x_ned": obs_x,
            "obstacle_y_ned": obs_y,
            "obstacle_z_ned": obs_z,
            "rel_obstacle_x_ned": obs_x - px,
            "rel_obstacle_y_ned": obs_y - py,
            "rel_obstacle_z_ned": obs_z - pz,
            "distance_obstacle_xy": distance_obstacle_xy,
            "cmd_vx_body": self.vx,
            "cmd_vy_body": self.vy,
            "cmd_vz_body": self.vz,
            "cmd_yaw_rate": self.yaw_rate,
            "forward_cruise_cmd": self.forward_cruise_cmd,
            "speed_mode": self.speed_mode,
            "active_keys": "+".join(sorted(self.active_keys)),
            "has_motion_command": int(has_motion_command),
        })

        self.sample_index += 1
        if self.sample_index % 20 == 0:
            self.log_file.flush()

    # ---------------------------------------------------------------------
    # Keyboard and update loop
    # ---------------------------------------------------------------------
    def _on_keyboard_event(self, event, *args, **kwargs):
        raw_key_name = getattr(event.input, "name", str(event.input)).upper()
        key_name = canonical_key_name(raw_key_name)
        event_type = event.type

        with self._lock:
            if event_type == carb.input.KeyboardEventType.KEY_PRESS:
                self._handle_key_press(key_name)
            elif event_type == carb.input.KeyboardEventType.KEY_RELEASE:
                self._handle_key_release(key_name)
            elif event_type == carb.input.KeyboardEventType.KEY_REPEAT:
                # W/S are cruise toggles, so repeats are ignored to avoid console spam.
                # A/D/Q/E/U/J are stateful through press/release already.
                pass

        # Consume handled events so viewport camera hotkeys do not fight the drone.
        return True

    def _handle_key_press(self, key_name: str) -> None:
        if key_name in MOTION_HOLD_KEYS:
            self.active_keys.add(key_name)
        elif key_name == "W":
            self.forward_cruise_cmd = FORWARD_CRUISE_VALUE
            log("Forward cruise enabled.")
        elif key_name == "S":
            self.forward_cruise_cmd = BACKWARD_CRUISE_VALUE
            log("Backward cruise enabled.")
        elif key_name == "X":
            self.forward_cruise_cmd = 0.0
            log("Forward/backward cruise neutral.")
        elif key_name == "SPACE":
            self._brake_locked()
        elif key_name == "KEY_1":
            self.speed_mode = "slow"
            log("Speed mode: slow")
        elif key_name == "KEY_2":
            self.speed_mode = "normal"
            log("Speed mode: normal")
        elif key_name == "KEY_3":
            self.speed_mode = "fast"
            log("Speed mode: fast")
        elif key_name == "G":
            self._toggle_recording()
        elif key_name == "M":
            self._mark_success()
        elif key_name == "F":
            self._mark_failure()
        elif key_name == "C":
            self._toggle_collision_label()
        elif key_name == "P":
            self.print_status()
        elif key_name == "H":
            self.print_help()
        elif key_name == "L":
            log("Land key pressed.")
            # Avoid doing blocking land work inside the keyboard callback.
            threading.Thread(target=self.stop, kwargs={"land_vehicle": True}, daemon=True).start()

    def _handle_key_release(self, key_name: str) -> None:
        if key_name in MOTION_HOLD_KEYS:
            self.active_keys.discard(key_name)

    def _brake_locked(self) -> None:
        self.active_keys.clear()
        self.forward_cruise_cmd = 0.0
        self.vx = self.vy = self.vz = self.yaw_rate = 0.0
        log("Brake / hover requested.")

    def _on_update(self, event) -> None:
        if self.stop_requested or not self.ready or self.master is None:
            return

        now = time.monotonic()
        dt = max(0.001, min(now - self._last_control_time, 0.15))
        if dt < CONTROL_DT:
            return
        self._last_control_time = now

        try:
            self._read_latest_telemetry_nonblocking()

            with self._lock:
                desired_vx, desired_vy, desired_vz, desired_yaw_rate = self._compute_desired_velocity_locked()
                self.vx = approach(self.vx, desired_vx, MAX_XY_ACCEL_MPS2 * dt)
                self.vy = approach(self.vy, desired_vy, MAX_XY_ACCEL_MPS2 * dt)
                self.vz = approach(self.vz, desired_vz, MAX_Z_ACCEL_MPS2 * dt)
                self.yaw_rate = approach(self.yaw_rate, desired_yaw_rate, MAX_YAW_ACCEL_RADPS2 * dt)

                vx, vy, vz, yaw_rate = self.vx, self.vy, self.vz, self.yaw_rate
                has_motion_command = bool(self.active_keys) or self.forward_cruise_cmd != 0.0

                self._write_log_row_locked(has_motion_command)

            self._send_gcs_heartbeat()
            self._send_velocity_setpoint(vx, vy, vz, yaw_rate)

            if has_motion_command and now - self._last_command_print_time > 0.7:
                log(
                    f"cmd vx={vx:+.2f}, vy={vy:+.2f}, vz={vz:+.2f}, "
                    f"yaw_rate={math.degrees(yaw_rate):+.1f} deg/s, "
                    f"mode={self.speed_mode}, cruise={self.forward_cruise_cmd:+.2f}, "
                    f"recording={self.recording}"
                )
                self._last_command_print_time = now
        except Exception as error:
            log(f"Update loop error: {error}")
            traceback.print_exc()
            self.ready = False

    def _compute_desired_velocity_locked(self) -> tuple[float, float, float, float]:
        limits = SPEED_MODES[self.speed_mode]
        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw_rate = 0.0

        steer_cmd = 0.0
        strafe_cmd = 0.0

        if "D" in self.active_keys or "ARROW_RIGHT" in self.active_keys:
            steer_cmd += 1.0
        if "A" in self.active_keys or "ARROW_LEFT" in self.active_keys:
            steer_cmd -= 1.0

        if "E" in self.active_keys:
            strafe_cmd += 1.0
        if "Q" in self.active_keys:
            strafe_cmd -= 1.0

        arrow_forward_pressed = "ARROW_UP" in self.active_keys
        arrow_backward_pressed = "ARROW_DOWN" in self.active_keys
        arrow_drive_cmd = 0.0
        if arrow_forward_pressed and not arrow_backward_pressed:
            arrow_drive_cmd = FORWARD_CRUISE_VALUE
        elif arrow_backward_pressed and not arrow_forward_pressed:
            arrow_drive_cmd = BACKWARD_CRUISE_VALUE

        drive_cmd = arrow_drive_cmd if arrow_drive_cmd != 0.0 else self.forward_cruise_cmd

        if drive_cmd != 0.0:
            vx = drive_cmd * limits["xy"]
            if steer_cmd != 0.0:
                vx *= TURN_FORWARD_SPEED_RATIO
        elif STEERING_AUTO_FORWARD and steer_cmd != 0.0:
            vx = STEERING_FORWARD_ASSIST_RATIO * limits["xy"]

        vy = strafe_cmd * limits["xy"] * STRAFE_SPEED_RATIO
        yaw_rate = steer_cmd * limits["yaw"]

        # Body NED convention: negative vz is up, positive vz is down.
        if "U" in self.active_keys:
            vz -= limits["z"]
        if "J" in self.active_keys:
            vz += limits["z"]

        return vx, vy, vz, yaw_rate

    # ---------------------------------------------------------------------
    # Console helpers
    # ---------------------------------------------------------------------
    def _print_message(self, message) -> None:
        if message is None:
            return
        msg_type = message.get_type()
        if msg_type == "HEARTBEAT":
            armed = bool(message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            decoded_mode = decode_px4_custom_mode(message.custom_mode)
            log(
                f"[HEARTBEAT] base_mode={message.base_mode}, "
                f"custom_mode={message.custom_mode}, decoded={decoded_mode}, "
                f"armed={armed}, system_status={message.system_status}"
            )
        elif msg_type == "STATUSTEXT":
            log(f"[STATUSTEXT] severity={message.severity}, text={message.text}")
        elif msg_type == "COMMAND_ACK":
            log(f"[COMMAND_ACK] command={message.command}, result={message.result}")
        elif msg_type == "LOCAL_POSITION_NED":
            log(
                f"[LOCAL_POSITION_NED] x={message.x:.2f}, y={message.y:.2f}, z={message.z:.2f}, "
                f"vx={message.vx:.2f}, vy={message.vy:.2f}, vz={message.vz:.2f}"
            )
        elif msg_type == "ATTITUDE":
            log(
                f"[ATTITUDE] roll={message.roll:.2f}, pitch={message.pitch:.2f}, yaw={message.yaw:.2f}"
            )

    def _print_message_throttled(self, message) -> None:
        if message is None:
            return
        wall_now = time.time()
        if message.get_type() == "STATUSTEXT":
            self._print_message(message)
        elif wall_now - self._last_status_time > 1.0:
            self._print_message(message)
            self._last_status_time = wall_now

    def print_status(self) -> None:
        with self._lock:
            pos = (
                self.telemetry["pos_x"],
                self.telemetry["pos_y"],
                self.telemetry["pos_z"],
            )
            goal = self.goal_ned
            dist_goal = distance_xy(pos[0], pos[1], goal[0], goal[1])

            log(
                f"Status={self.state_text}, ready={self.ready}, recording={self.recording}, "
                f"label={self.manual_label}, collision={self.manual_collision}, "
                f"pos=({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f}), "
                f"goal=({goal[0]:+.2f}, {goal[1]:+.2f}, {goal[2]:+.2f}), "
                f"dist_goal_xy={dist_goal:.2f}, "
                f"vx={self.vx:+.2f}, vy={self.vy:+.2f}, vz={self.vz:+.2f}, "
                f"yaw_rate={math.degrees(self.yaw_rate):+.1f} deg/s, "
                f"mode={self.speed_mode}, cruise={self.forward_cruise_cmd:+.2f}, "
                f"keys={sorted(self.active_keys)}, "
                f"log={self.log_path}"
            )

    def print_help(self) -> None:
        log("")
        log("WebRTC / Isaac keyboard controls:")
        log("  Up / Down    : hold to fly forward / backward")
        log("  Left / Right : hold to steer left / steer right")
        log("  W / S        : optional cruise forward / backward toggle")
        log("  X            : neutral forward/backward cruise")
        log("  Q / E        : side-slip left / side-slip right")
        log("  U / J        : up / down velocity")
        log("  Space        : brake now / hover")
        log("  1 / 2 / 3    : slow / normal / fast speed mode")
        log("  G            : start / stop CSV recording")
        log("  M            : mark current demo as success")
        log("  F            : mark current demo as failure")
        log("  C            : toggle collision label")
        log("  P            : print current status")
        log("  H            : show help")
        log("  L            : land and stop controller")
        log("Tip: click the WebRTC viewport first so Isaac receives keyboard focus.")
        log(f"CSV path: {LOG_DIR}")
        log("")


# Script Editor convenience: re-running this script stops the previous controller
# before creating a new one, so you do not stack multiple MAVLink controllers.
try:
    _px4_webrtc_keyboard_controller.stop(land_vehicle=False)  # type: ignore[name-defined]
except Exception:
    pass

_px4_webrtc_keyboard_controller = IsaacWebRTCPX4DriveController()
_px4_webrtc_keyboard_controller.start()
