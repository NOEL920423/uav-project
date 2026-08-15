# 5.ros2_uav_pose_publisher_logger.py
# Run inside Isaac Sim with Isaac Sim VS Code Edition.
#
# Purpose:
#   Publish UAV pose from Isaac Sim to ROS2 and save pose data to CSV.
#
# ROS2 topic:
#   /isaac_uav/pose   geometry_msgs/msg/PoseStamped
#
# Stop:
#   builtins.stop_ros2_uav_pose_publisher_logger()

import builtins
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import omni
import omni.usd
import omni.kit.app
import omni.timeline
from pxr import UsdGeom


# ============================================================
# User settings
# ============================================================

UAV_BODY_PATH = "/World/quadrotor/body"

ROS_TOPIC_NAME = "/isaac_uav/pose"
ROS_NODE_NAME = "isaac_uav_pose_publisher"

PUBLISH_RATE_HZ = 10.0

LOG_DIR = str(Path.home() / "uav-project" / "ros2_uav_pose_logs")
CSV_PREFIX = "uav_pose_ros2"

PRINT_STATUS = True
PRINT_INTERVAL_S = 2.0


# ============================================================
# ROS2 import
# ============================================================

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    RCLPY_AVAILABLE = True
except Exception as exc:
    rclpy = None
    PoseStamped = None
    RCLPY_AVAILABLE = False
    RCLPY_IMPORT_ERROR = exc


# ============================================================
# Helpers
# ============================================================

def log(message):
    print(f"[ROS2UAVPose] {message}")


def warn(message):
    print(f"[ROS2UAVPose][Warning] {message}")


def get_stage():
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage found.")
    return stage


def prim_exists(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    return bool(prim and prim.IsValid())


def get_world_matrix(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)

    if not prim or not prim.IsValid():
        return None

    try:
        return omni.usd.get_world_transform_matrix(prim)
    except Exception:
        cache = UsdGeom.XformCache()
        return cache.GetLocalToWorldTransform(prim)


def extract_pose_from_matrix(matrix):
    translation = matrix.ExtractTranslation()

    x = float(translation[0])
    y = float(translation[1])
    z = float(translation[2])

    # Quaternion extraction from USD transform.
    try:
        rotation = matrix.ExtractRotation()
        quat = rotation.GetQuat()
        q_w = float(quat.GetReal())
        imag = quat.GetImaginary()
        q_x = float(imag[0])
        q_y = float(imag[1])
        q_z = float(imag[2])
    except Exception:
        q_x = 0.0
        q_y = 0.0
        q_z = 0.0
        q_w = 1.0

    return x, y, z, q_x, q_y, q_z, q_w


def make_stamp_from_sim_time(sim_time):
    if sim_time is None:
        now = time.time()
    else:
        now = float(sim_time)

    sec = int(now)
    nanosec = int((now - sec) * 1e9)
    return sec, nanosec


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_sim_time(event):
    event_time = getattr(event, "current_time", None)
    if event_time is not None:
        return float(event_time)
    try:
        return float(omni.timeline.get_timeline_interface().get_current_time())
    except Exception:
        return None


def sanitize_episode_id(episode_id):
    if episode_id is None:
        return ""
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(episode_id).strip()
    )


# ============================================================
# Main class
# ============================================================

class ROS2UAVPosePublisherLogger:
    def __init__(self, episode_id=None):
        self.stage = get_stage()
        self.episode_id = sanitize_episode_id(episode_id)
        self.subscription = None

        self.is_running = False
        self.start_wall_time = None
        self.last_publish_wall_time = 0.0
        self.last_print_wall_time = 0.0
        self.sample_index = 0

        self.node = None
        self.publisher = None

        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None

    def start(self):
        if not prim_exists(self.stage, UAV_BODY_PATH):
            raise RuntimeError(
                f"UAV body prim does not exist: {UAV_BODY_PATH}. "
                "Please start Pegasus / UAV scene first."
            )

        self.setup_csv()
        self.setup_ros2()

        self.start_wall_time = time.time()
        self.last_publish_wall_time = 0.0
        self.last_print_wall_time = 0.0
        self.sample_index = 0
        self.is_running = True

        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self.on_update,
            name="ROS2UAVPosePublisherLoggerUpdate",
        )

        log("Started.")
        log(f"UAV body path: {UAV_BODY_PATH}")
        log(f"ROS2 topic: {ROS_TOPIC_NAME}")
        log(f"CSV output: {self.csv_path}")
        log("Stop with: builtins.stop_ros2_uav_pose_publisher_logger()")

    def setup_csv(self):
        ensure_dir(LOG_DIR)

        if self.episode_id:
            filename = f"uav_pose_{self.episode_id}.csv"
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{CSV_PREFIX}_{stamp}.csv"
        self.csv_path = os.path.join(LOG_DIR, filename)

        self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")

        fieldnames = [
            "sample_index",
            "time_wall",
            "record_time",
            "sim_time",
            "ros_stamp_sec",
            "ros_stamp_nanosec",
            "uav_body_path",
            "x_isaac",
            "y_isaac",
            "z_isaac",
            "qx",
            "qy",
            "qz",
            "qw",
            "ros_topic",
            "publish_rate_hz",
        ]

        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_file.flush()

    def setup_ros2(self):
        if not RCLPY_AVAILABLE:
            warn("rclpy is not available in Isaac Sim Python.")
            warn(f"Import error: {RCLPY_IMPORT_ERROR}")
            warn("This script will save CSV only, but it will NOT publish ROS2 topic.")
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node(ROS_NODE_NAME)
        self.publisher = self.node.create_publisher(PoseStamped, ROS_TOPIC_NAME, 10)

        log("ROS2 publisher created.")

    def stop(self):
        self.is_running = False

        if self.subscription is not None:
            try:
                self.subscription.unsubscribe()
                log("Update subscription stopped.")
            except Exception as exc:
                warn(f"Failed to unsubscribe update callback: {exc}")
            self.subscription = None

        if self.node is not None:
            try:
                self.node.destroy_node()
                log("ROS2 node destroyed.")
            except Exception as exc:
                warn(f"Failed to destroy ROS2 node: {exc}")
            self.node = None
            self.publisher = None

        if self.csv_file is not None:
            try:
                self.csv_file.flush()
                self.csv_file.close()
                log(f"CSV closed: {self.csv_path}")
            except Exception as exc:
                warn(f"Failed to close CSV: {exc}")
            self.csv_file = None
            self.csv_writer = None

        log(f"Stopped. Samples: {self.sample_index}")

    def publish_ros2_pose(self, sec, nanosec, pose_values):
        if self.publisher is None:
            return

        x, y, z, qx, qy, qz, qw = pose_values

        msg = PoseStamped()
        msg.header.stamp.sec = int(sec)
        msg.header.stamp.nanosec = int(nanosec)
        msg.header.frame_id = "isaac_world"

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)

        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)

        self.publisher.publish(msg)

        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception:
            pass

    def write_csv_row(self, now_wall, record_time, sim_time, sec, nanosec, pose_values):
        x, y, z, qx, qy, qz, qw = pose_values

        self.csv_writer.writerow(
            {
                "sample_index": self.sample_index,
                "time_wall": now_wall,
                "record_time": record_time,
                "sim_time": "" if sim_time is None else float(sim_time),
                "ros_stamp_sec": int(sec),
                "ros_stamp_nanosec": int(nanosec),
                "uav_body_path": UAV_BODY_PATH,
                "x_isaac": x,
                "y_isaac": y,
                "z_isaac": z,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "ros_topic": ROS_TOPIC_NAME,
                "publish_rate_hz": PUBLISH_RATE_HZ,
            }
        )

        if self.sample_index % 20 == 0:
            self.csv_file.flush()

    def print_status_if_needed(self, now_wall, pose_values):
        if not PRINT_STATUS:
            return

        if now_wall - self.last_print_wall_time < PRINT_INTERVAL_S:
            return

        self.last_print_wall_time = now_wall

        x, y, z, qx, qy, qz, qw = pose_values

        log(
            f"publishing... samples={self.sample_index}, "
            f"pos_isaac=({x:+.3f}, {y:+.3f}, {z:+.3f}), "
            f"topic={ROS_TOPIC_NAME}"
        )

    def on_update(self, event):
        if not self.is_running:
            return

        now_wall = time.time()

        interval_s = 1.0 / float(PUBLISH_RATE_HZ)
        if now_wall - self.last_publish_wall_time < interval_s:
            return

        self.last_publish_wall_time = now_wall

        matrix = get_world_matrix(self.stage, UAV_BODY_PATH)

        if matrix is None:
            warn(f"Cannot read UAV body transform: {UAV_BODY_PATH}")
            return

        pose_values = extract_pose_from_matrix(matrix)

        sim_time = get_sim_time(event)
        sec, nanosec = make_stamp_from_sim_time(sim_time)

        record_time = now_wall - self.start_wall_time

        self.sample_index += 1

        self.publish_ros2_pose(sec, nanosec, pose_values)
        self.write_csv_row(now_wall, record_time, sim_time, sec, nanosec, pose_values)
        self.print_status_if_needed(now_wall, pose_values)


# ============================================================
# Public controls
# ============================================================

def stop_existing_ros2_uav_pose_publisher_logger():
    old = getattr(builtins, "_ros2_uav_pose_publisher_logger", None)

    if old is None:
        return

    try:
        old.stop()
        log("Stopped old pose publisher/logger.")
    except Exception as exc:
        warn(f"Failed to stop old pose publisher/logger: {exc}")

    try:
        delattr(builtins, "_ros2_uav_pose_publisher_logger")
    except Exception:
        pass


def start_ros2_uav_pose_publisher_logger(episode_id=None):
    stop_existing_ros2_uav_pose_publisher_logger()

    recorder = ROS2UAVPosePublisherLogger(episode_id=episode_id)
    builtins._ros2_uav_pose_publisher_logger = recorder
    recorder.start()
    return recorder


def stop_ros2_uav_pose_publisher_logger():
    recorder = getattr(builtins, "_ros2_uav_pose_publisher_logger", None)

    if recorder is None:
        log("No active pose publisher/logger.")
        return

    recorder.stop()

    try:
        delattr(builtins, "_ros2_uav_pose_publisher_logger")
    except Exception:
        pass


def print_ros2_uav_pose_publisher_logger_status():
    recorder = getattr(builtins, "_ros2_uav_pose_publisher_logger", None)

    if recorder is None:
        log("No active pose publisher/logger.")
        return

    log("Status:")
    log(f"  running: {recorder.is_running}")
    log(f"  samples: {recorder.sample_index}")
    log(f"  csv_path: {recorder.csv_path}")
    log(f"  ros_topic: {ROS_TOPIC_NAME}")


builtins.start_ros2_uav_pose_publisher_logger = start_ros2_uav_pose_publisher_logger
builtins.stop_ros2_uav_pose_publisher_logger = stop_ros2_uav_pose_publisher_logger
builtins.print_ros2_uav_pose_publisher_logger_status = print_ros2_uav_pose_publisher_logger_status


# Start immediately when this script is run. The episode manager provides the
# ID through builtins before the first load so camera and pose files match.
start_ros2_uav_pose_publisher_logger(
    getattr(builtins, "_uav_pose_episode_id", None)
)
