"""
Isaac Sim Script Editor PX4 CSV action replay.

Purpose
-------
Replay a manual demonstration CSV recorded by:
    isaac_webrtc_px4_drive_control_arrow_keys_logger.py

Run this script INSIDE Isaac Sim:
    Window > Script Editor > paste this script > Run

What it does
------------
1. Reads a recorded CSV file.
2. Connects to PX4 through MAVLink.
3. Sends warm-up setpoints.
4. Switches to OFFBOARD.
5. Arms and climbs to TARGET_ALTITUDE_M.
6. Replays the recorded body-frame velocity commands:
       cmd_vx_body, cmd_vy_body, cmd_vz_body, cmd_yaw_rate

Important
---------
This is NOT behavior cloning yet.
This is only action replay:
    "At time t, send the same command that the human sent in the demo."

It will look best when:
    - The UAV starts from a similar pose.
    - The same environment is loaded.
    - The same takeoff altitude is used.
    - The CSV was recorded cleanly.

If the start position is different, action replay may drift.
"""

from __future__ import annotations

import csv
import math
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

try:
    import carb
    import omni.kit.app
except Exception as import_error:
    raise RuntimeError(
        "This script must be run inside Isaac Sim / Omniverse Kit Script Editor."
    ) from import_error

try:
    from pymavlink import mavutil
except Exception as import_error:
    mavutil = None
    _PYMAVLINK_IMPORT_ERROR = import_error
else:
    _PYMAVLINK_IMPORT_ERROR = None


# =============================================================================
# User settings
# =============================================================================

# Change this to the CSV you want to replay.
REPLAY_CSV_PATH = "/home/noel_614420090/uav-project/uav_demo_logs/manual_demo_20260603_143628.csv"

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

# Replay behavior.
REPLAY_SPEED_SCALE = 1.0       # 1.0 = original speed, 0.5 = slow motion, 2.0 = faster.
COMMAND_SCALE = 1.0            # 1.0 = original command strength.
SKIP_IDLE_AT_BEGINNING = True  # Skip early rows with near-zero command.
IDLE_COMMAND_EPS = 1e-3

HOVER_SECONDS_AFTER_REPLAY = 3.0
LAND_AFTER_REPLAY = False

HOVER_SECONDS_BEFORE_LAND = 0.6
HOVER_SECONDS_AFTER_LAND_COMMAND = 1.5

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


# =============================================================================
# Helpers
# =============================================================================

def log(message: str) -> None:
    print(f"[PX4 CSV Replay] {message}")
    try:
        carb.log_info(f"[PX4 CSV Replay] {message}")
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


def is_idle_command(row: dict) -> bool:
    values = [
        abs(row["cmd_vx_body"]),
        abs(row["cmd_vy_body"]),
        abs(row["cmd_vz_body"]),
        abs(row["cmd_yaw_rate"]),
    ]
    return max(values) <= IDLE_COMMAND_EPS


def load_replay_csv(csv_path: str) -> list[dict]:
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows = []
    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        required = ["demo_time", "cmd_vx_body", "cmd_vy_body", "cmd_vz_body", "cmd_yaw_rate"]
        for name in required:
            if name not in reader.fieldnames:
                raise ValueError(f"CSV is missing required column: {name}")

        for raw in reader:
            try:
                row = {
                    "demo_time": float(raw["demo_time"]),
                    "cmd_vx_body": float(raw["cmd_vx_body"]),
                    "cmd_vy_body": float(raw["cmd_vy_body"]),
                    "cmd_vz_body": float(raw["cmd_vz_body"]),
                    "cmd_yaw_rate": float(raw["cmd_yaw_rate"]),
                }

                # Optional fields for debug printing.
                for optional_name in [
                    "pos_x_ned", "pos_y_ned", "pos_z_ned",
                    "distance_goal_xy", "manual_label", "success_auto"
                ]:
                    if optional_name in raw:
                        row[optional_name] = raw[optional_name]

                rows.append(row)
            except Exception:
                # Ignore malformed rows instead of killing the replay.
                pass

    if not rows:
        raise ValueError(f"No valid replay rows in CSV: {path}")

    rows.sort(key=lambda x: x["demo_time"])

    if SKIP_IDLE_AT_BEGINNING:
        first_motion_index = 0
        for i, row in enumerate(rows):
            if not is_idle_command(row):
                first_motion_index = i
                break

        skipped = first_motion_index
        rows = rows[first_motion_index:]

        if rows:
            t0 = rows[0]["demo_time"]
            for row in rows:
                row["demo_time"] -= t0

        log(f"Skipped {skipped} idle rows at beginning.")

    log(f"Loaded {len(rows)} rows from: {path}")
    log(f"Replay duration: {rows[-1]['demo_time']:.2f} s")
    return rows


# =============================================================================
# Main replay controller
# =============================================================================

class IsaacScriptEditorPX4CSVReplayController:
    def __init__(self) -> None:
        if mavutil is None:
            raise RuntimeError(
                "pymavlink is not installed in Isaac Sim's Python environment.\n"
                "Install it into Isaac's Python environment, then run this script again.\n"
                f"Original import error: {_PYMAVLINK_IMPORT_ERROR}"
            )

        self.master = None
        self.started = False
        self.ready = False
        self.stop_requested = False
        self.finished = False
        self.state_text = "created"

        self._update_sub = None
        self._init_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self.MAV_FRAME_BODY_NED = getattr(mavutil.mavlink, "MAV_FRAME_BODY_NED", 8)

        self.rows = load_replay_csv(REPLAY_CSV_PATH)
        self.replay_start_time = 0.0
        self.current_index = 0

        self.current_cmd = {
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "yaw_rate": 0.0,
        }

        self.hover_start_time: Optional[float] = None
        self._last_update_time = time.monotonic()
        self._last_status_time = 0.0
        self._last_command_print_time = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self.started:
            log("Replay controller is already started.")
            return

        self.started = True
        self.stop_requested = False
        self.state_text = "starting"

        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="px4_csv_action_replay_update",
        )

        self._init_thread = threading.Thread(
            target=self._initialize_px4,
            name="PX4 CSV action replay initialization",
            daemon=True,
        )
        self._init_thread.start()

        log("=" * 72)
        log("CSV action replay controller started in Isaac Sim Script Editor.")
        log("Watch the UAV through WebRTC.")
        log("=" * 72)

    def stop(self, land_vehicle: bool = False) -> None:
        self.stop_requested = True
        self.ready = False

        if self._update_sub is not None:
            self._update_sub = None

        try:
            if self.master is not None:
                self._send_zero_velocity_for_seconds(0.8)
                if land_vehicle:
                    self._land()
        except Exception as error:
            log(f"Stop warning: {error}")

        self.started = False
        self.state_text = "stopped"
        log("Replay controller stopped.")

    # ------------------------------------------------------------------
    # PX4 setup
    # ------------------------------------------------------------------

    def _initialize_px4(self) -> None:
        try:
            self.state_text = "connecting"
            self.master = self._connect_to_px4()

            if self.stop_requested:
                return

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
                time.sleep(CONTROL_DT)

            self.state_text = "offboard"
            if not self._set_offboard_mode(x_target, y_target, z_target, yaw_target):
                self.state_text = "offboard_failed"
                log("OFFBOARD failed. Replay will not continue.")
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
                self.current_index = 0
                self.replay_start_time = time.monotonic()
                self.finished = False
                self.hover_start_time = None
                self.current_cmd = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}

            self.ready = True
            self.state_text = "replaying"
            log("Ready. CSV action replay is active.")

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

            message = self.master.recv_match(
                type=["LOCAL_POSITION_NED", "HEARTBEAT", "STATUSTEXT"],
                blocking=False,
            )
            if message is not None:
                if message.get_type() == "LOCAL_POSITION_NED":
                    saw_local_position = True
                    if abs(message.z - z_target) <= TAKEOFF_Z_TOLERANCE_M:
                        log(f"Initial altitude reached: z={message.z:.2f} m target={z_target:.2f} m")
                        return True
                self._print_message_throttled(message)

            time.sleep(CONTROL_DT)

        if not saw_local_position:
            log("Warning: no LOCAL_POSITION_NED received during takeoff wait.")
        log("Initial altitude wait finished. Continuing with CSV replay.")
        return False

    # ------------------------------------------------------------------
    # Replay update loop
    # ------------------------------------------------------------------

    def _on_update(self, event) -> None:
        if self.stop_requested or not self.ready or self.master is None:
            return

        now = time.monotonic()
        dt = max(0.001, min(now - self._last_update_time, 0.15))
        if dt < CONTROL_DT:
            return
        self._last_update_time = now

        try:
            self._send_gcs_heartbeat()

            with self._lock:
                if self.finished:
                    self._send_finished_command_locked()
                    return

                replay_time = (now - self.replay_start_time) * REPLAY_SPEED_SCALE

                # Move index forward to match replay time.
                while self.current_index + 1 < len(self.rows):
                    next_time = self.rows[self.current_index + 1]["demo_time"]
                    if next_time <= replay_time:
                        self.current_index += 1
                    else:
                        break

                row = self.rows[self.current_index]

                vx = COMMAND_SCALE * row["cmd_vx_body"]
                vy = COMMAND_SCALE * row["cmd_vy_body"]
                vz = COMMAND_SCALE * row["cmd_vz_body"]
                yaw_rate = COMMAND_SCALE * row["cmd_yaw_rate"]

                self.current_cmd = {
                    "vx": vx,
                    "vy": vy,
                    "vz": vz,
                    "yaw_rate": yaw_rate,
                }

                if self.current_index >= len(self.rows) - 1:
                    self.finished = True
                    self.hover_start_time = time.monotonic()
                    log("CSV replay finished. Hovering.")
                    self._send_finished_command_locked()
                    return

            self._send_velocity_setpoint(vx, vy, vz, yaw_rate)

            message = self.master.recv_match(
                type=["LOCAL_POSITION_NED", "HEARTBEAT", "STATUSTEXT"],
                blocking=False,
            )
            if message is not None:
                self._print_message_throttled(message)

            if time.time() - self._last_command_print_time > 0.7:
                log(
                    f"replay t={replay_time:.2f}s row={self.current_index}/{len(self.rows)-1} "
                    f"cmd vx={vx:+.2f}, vy={vy:+.2f}, vz={vz:+.2f}, "
                    f"yaw_rate={math.degrees(yaw_rate):+.1f} deg/s"
                )
                self._last_command_print_time = time.time()

        except Exception as error:
            log(f"Update loop error: {error}")
            traceback.print_exc()
            self.ready = False

    def _send_finished_command_locked(self) -> None:
        self._send_gcs_heartbeat()
        self._send_velocity_setpoint(0.0, 0.0, 0.0, 0.0)

        if self.hover_start_time is not None:
            if time.monotonic() - self.hover_start_time >= HOVER_SECONDS_AFTER_REPLAY:
                if LAND_AFTER_REPLAY:
                    self._land()
                log("Replay complete. Controller will keep sending hover setpoints.")
                self.hover_start_time = None

    def _send_zero_velocity_for_seconds(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while time.time() < deadline and self.master is not None:
            self._send_gcs_heartbeat()
            self._send_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
            time.sleep(CONTROL_DT)

    def _land(self) -> None:
        log("Sending LAND command...")
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

    # ------------------------------------------------------------------
    # Console helpers
    # ------------------------------------------------------------------

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
            log(
                f"Status={self.state_text}, ready={self.ready}, finished={self.finished}, "
                f"row={self.current_index}/{len(self.rows)-1}, "
                f"cmd=({self.current_cmd['vx']:+.2f}, {self.current_cmd['vy']:+.2f}, "
                f"{self.current_cmd['vz']:+.2f}, "
                f"{math.degrees(self.current_cmd['yaw_rate']):+.1f} deg/s)"
            )


# =============================================================================
# Script Editor convenience
# =============================================================================

# Re-running this script stops the previous replay controller first.
try:
    _px4_csv_replay_controller.stop(land_vehicle=False)  # type: ignore[name-defined]
except Exception:
    pass

_px4_csv_replay_controller = IsaacScriptEditorPX4CSVReplayController()
_px4_csv_replay_controller.start()

# Useful manual commands in Script Editor console after running:
#   _px4_csv_replay_controller.print_status()
#   _px4_csv_replay_controller.stop(land_vehicle=False)
#   _px4_csv_replay_controller.stop(land_vehicle=True)
