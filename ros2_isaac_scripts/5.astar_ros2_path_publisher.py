# px4_astar_lookahead_episode_runner_logger.py
#
# Ground-truth A* episode runner and logger for Isaac Sim + Pegasus + PX4.
#
# This version does NOT generate scene objects.
# Please run scene_episode_generator.py first.
#
# Features:
# - Read live obstacle transforms from /World/GeneratedEpisode/Obstacles
# - Read live red point / TargetDisk transform from /World/GeneratedEpisode/Target
# - Convert Isaac coordinates to PX4 local NED
# - Plan waypoints with a ground-truth 2D occupancy grid and A*
# - Validate simplified waypoint segments before flight
# - Add direct-path-biased A* cost to reduce unnecessary outer detours
# - Run PX4 OFFBOARD lookahead / carrot-chasing path following with velocity setpoints
# - Log mission data to CSV
#
# Keyboard:
# - H: print help
# - P: print current status
# - L: land and stop
# - R: TODO reset episode
#
# Publishes the validated A* path to ROS 2. No camera, OpenCV, or machine learning.

import json
import os
import csv
import heapq
import math
import time
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from pxr import Usd, UsdGeom, Gf, Sdf

import omni.usd
import omni.kit.app

import builtins

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Path as RosPath
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    ROS2_PATH_AVAILABLE = True
    ROS2_PATH_IMPORT_ERROR = None
except Exception as exc:
    rclpy = None
    PoseStamped = None
    RosPath = None
    DurabilityPolicy = None
    HistoryPolicy = None
    QoSProfile = None
    ReliabilityPolicy = None
    ROS2_PATH_AVAILABLE = False
    ROS2_PATH_IMPORT_ERROR = exc

try:
    import carb
    import omni.appwindow
except Exception:
    carb = None

try:
    from pymavlink import mavutil
except Exception as exc:
    raise RuntimeError(
        "pymavlink is not available in Isaac Sim Python. "
        "Please install pymavlink into Isaac Sim's Python environment."
    ) from exc


# ============================================================
# USER SETTINGS
# ============================================================

RUN_ON_PASTE = True

PUBLISH_ROS2_PATH_ONLY = True
ROS2_PATH_TOPIC = "/uav/planned_path"
ROS2_PATH_FRAME_ID = "px4_ned"
ROS2_PATH_PUBLISH_RATE_HZ = 1.0
ROS2_PATH_NODE_NAME = "isaac_astar_path_publisher"

EXPERIMENT_PRESET_NAME = "v4_astar_lookahead_debug_safety_envelope"

PX4_CONNECTION_STRING = "udpin:0.0.0.0:14550"

# Important coordinate setting from your current experiment.
# Isaac goal [3, 5, 0] should become PX4 NED approximately [5, 3, 0].
SWAP_XY = True

PX4_NED_OFFSET_X = 0.0
PX4_NED_OFFSET_Y = 0.0
PX4_NED_OFFSET_Z = 0.0

START_ISAAC = [0.0, 0.0, 0.0]
FINAL_GOAL_ISAAC = [3.0, 5.0, 0.0]

FLIGHT_ALTITUDE_M = 2.0

GENERATED_ROOT_PATH = "/World/GeneratedEpisode"

OBSTACLE_ROOT_PATH = f"{GENERATED_ROOT_PATH}/Obstacles"
TARGET_ROOT_PATH = f"{GENERATED_ROOT_PATH}/Target"
RED_POINT_PATH = f"{TARGET_ROOT_PATH}/RedPoint"
TARGET_DISK_PATH = f"{TARGET_ROOT_PATH}/TargetDisk"

# Read RedPoint first. TargetDisk is a fallback for older scene generator versions.
RED_POINT_CANDIDATE_PATHS = [RED_POINT_PATH, TARGET_DISK_PATH]

# Scene reading behavior.
# This version reads the currently visible USD Stage, not the old JSON/CSV record.
# It is designed for this workflow:
#   1. Generate random obstacles.
#   2. Manually move obstacles or TargetDisk in Isaac Sim.
#   3. Run this script and plan from the live edited Stage.
SCENE_READER_USE_LIVE_STAGE_TRANSFORMS = True
SCENE_READER_UPDATE_TICKS = 0

# When True, obstacles are read as top-level objects under /World/GeneratedEpisode/Obstacles.
# This prevents double-counting child mesh/cylinder prims if an obstacle is wrapped by an Xform.
READ_OBSTACLES_AS_LIVE_TOP_LEVEL_OBJECTS = True

# If you manually move TargetDisk, this lets the mission goal follow the moved disk.
# If you want to force the old fixed goal [3, 5, 0], set this to False.
USE_STAGE_TARGET_AS_FINAL_GOAL = True

# Optional extra obstacle roots for hand-made obstacle sets.
# Keep this conservative. Do not add broad paths like /World unless you filter names carefully.
EXTRA_OBSTACLE_ROOT_PATHS: List[str] = []

# Used only for fallback name filtering if needed.
OBSTACLE_NAME_PREFIXES = [
    "Obstacle_",
    "obstacle_",
    "CylinderObstacle_",
    "Building_",
    "building_",
]

# A* occupancy grid planning parameters.
GRID_RESOLUTION_M = 0.05
GRID_MARGIN_M = 2.0

UAV_SAFETY_RADIUS_M = 0.18
OBSTACLE_SAFETY_MARGIN_M = 0.13
MIN_SEGMENT_CLEARANCE_M = 0.07

# Safety radius naming aliases.
# The old names are kept for backward compatibility with your previous scripts.
# Conceptually this is configuration-space obstacle expansion:
# obstacle radius + UAV physical radius + static safety margin.
UAV_PHYSICAL_RADIUS_M = UAV_SAFETY_RADIUS_M
STATIC_SAFETY_MARGIN_M = OBSTACLE_SAFETY_MARGIN_M
SEGMENT_VALIDATION_MARGIN_M = MIN_SEGMENT_CLEARANCE_M

# Margins reserved for future data-driven tuning.
# GRID_DISCRETIZATION_MARGIN_M estimates the worst-case cell-center quantization error.
GRID_DISCRETIZATION_MARGIN_M = 0.5 * math.sqrt(2.0) * GRID_RESOLUTION_M
TRACKING_ERROR_MARGIN_M = 0.0
CONTROL_RESPONSE_MARGIN_M = 0.0
SENSOR_OR_STATE_UNCERTAINTY_M = 0.0

# Optional clearance-aware soft cost.
# Occupied cells are still forbidden. This only makes cells near the forbidden envelope
# slightly more expensive, so the path does not skim the orange/red disks for free.
USE_CLEARANCE_AWARE_COST_FIELD = True
SOFT_CLEARANCE_COST_RADIUS_M = 0.40
CLEARANCE_COST_WEIGHT = 0.25

PATH_SIMPLIFY_TOLERANCE_M = 0.05
MAX_WAYPOINT_SPACING_M = 1.30

# If the first A* path exists but fails the continuous segment safety check,
# the planner will rebuild the grid with this extra clearance and re-run A*.
REPLAN_WITH_SEGMENT_CLEARANCE_IF_NEEDED = True

# Softly prefer paths that stay closer to the start-goal line.
# This is not a safety override. Occupied cells and segment clearance still dominate.
USE_DIRECT_PATH_BIAS = True
DIRECT_PATH_BIAS_WEIGHT = 0.07

# If a grid cell around start/goal is occupied only because of grid rounding,
# the planner may search from/to the nearest free cell and then append the exact endpoint.
# If the exact start/goal is truly inside an inflated obstacle, the mission still stops.
ENDPOINT_FREE_SEARCH_RADIUS_M = 1.00
ENDPOINT_MIN_EXACT_CLEARANCE_M = 0.00

# PX4 control parameters.
CONTROL_RATE_HZ = 20.0
LOG_RATE_HZ = 20.0

WARMUP_SETPOINT_COUNT = 80
WARMUP_SETPOINT_RATE_HZ = 20.0

TAKEOFF_TIMEOUT_S = 20.0
MISSION_TIMEOUT_S = 120.0

# PX4 safety / confirmation settings.
# The UAV must actually be armed and must actually climb before the path mission starts.
WAIT_FOREVER_FOR_PX4_HEARTBEAT = True
PX4_HEARTBEAT_TIMEOUT_S = 180.0
PX4_HEARTBEAT_RETRY_PRINT_INTERVAL_S = 5.0
PX4_COMMAND_ACK_TIMEOUT_S = 5.0
PX4_ARM_CONFIRM_TIMEOUT_S = 8.0
PX4_MODE_CONFIRM_TIMEOUT_S = 5.0
REQUIRE_ARM_CONFIRMATION = True
REQUIRE_TAKEOFF_REACHED_BEFORE_MISSION = True
MIN_TAKEOFF_ALTITUDE_REACHED_M = 1.60

# PX4 custom mode fallback. Some pymavlink dialects report PX4 flightmode as
# UNKNOWN even when HEARTBEAT.custom_mode already equals OFFBOARD.
PX4_MAIN_MODE_OFFBOARD = 6
PX4_CUSTOM_MODE_OFFBOARD = PX4_MAIN_MODE_OFFBOARD << 16

# Lookahead path following / carrot chasing.
USE_LOOKAHEAD_PATH_FOLLOWING = True
PATH_LOOKAHEAD_M = 0.55
MIN_LOOKAHEAD_M = 0.35
MAX_LOOKAHEAD_M = 0.90

# Intermediate path points are allowed to be loose.
# The final goal remains precise.
INTERMEDIATE_WAYPOINT_REACHED_RADIUS_XY_M = 0.45
FINAL_WAYPOINT_REACHED_RADIUS_XY_M = 0.22
WAYPOINT_REACHED_RADIUS_Z_M = 0.30

# When lookahead is active, this prevents a target that would create an unsafe shortcut.
LOOKAHEAD_SEGMENT_CHECK_EXTRA_CLEARANCE_M = 0.07

# Runtime status print throttling.
STATUS_PRINT_INTERVAL_S = 1.0

KP_XY = 0.55
KP_Z = 0.85
KP_YAW = 1.20

MAX_SPEED_XY_MPS = 0.65
MIN_SPEED_XY_MPS = 0.25
MAX_SPEED_Z_MPS = 0.55
MAX_YAW_RATE_RADPS = 0.85

HOVER_MAX_SPEED_XY_MPS = 0.45
HOVER_MAX_SPEED_Z_MPS = 0.35

LAND_STOP_DELAY_S = 5.0

LOG_DIR = os.path.expanduser("~/uav-project/uav_episode_logs")

# Velocity command smoothing.
# This reduces twitchy motion when the lookahead target changes around narrow passages.
ENABLE_COMMAND_SMOOTHING = True
MAX_ACCEL_XY_MPS2 = 0.55
MAX_ACCEL_Z_MPS2 = 0.35
MAX_YAW_ACCEL_RADPS2 = 0.55

# Simple 2.5D planning: ignore obstacles that are clearly shorter than the flight altitude.
# The obstacle is ignored only if obstacle_top + vertical_clearance <= flight altitude.
ENABLE_OVERFLY_SHORT_OBSTACLES = True
OVERFLY_VERTICAL_CLEARANCE_M = 0.35

# Isaac Sim path visualization.
# Planned path is drawn before PX4 connection.
# IMPORTANT: Live executed trail drawing is disabled by default because USD stage edits
# from the PX4 mission thread can disturb Isaac Sim / Pegasus MAVLink polling.
DRAW_PATH_VISUALIZATION_IN_STAGE = True
PATH_VIS_ROOT = f"{GENERATED_ROOT_PATH}/DebugFlightPath"
# Keep the complete A* visualization prim in the stage, but hide it from the
# FPV/TOP render products by default. It can be toggled at runtime with:
# builtins.set_astar_path_visibility(True)
SHOW_PATH_VISUALIZATION = False

# DebugSafetyEnvelope visualization.
# These disks show the planner's forbidden region before PX4 connection.
# Orange disk: A* planning inflated radius.
# Red disk: continuous segment-validation forbidden radius.
# 畫在柱子上的安全範圍
DRAW_SAFETY_ENVELOPE = False # 安全範圍開關
SAFETY_ENVELOPE_ROOT = f"{GENERATED_ROOT_PATH}/DebugSafetyEnvelope"
DRAW_PLANNING_RADIUS = True
DRAW_VALIDATION_RADIUS = True
SAFETY_ENVELOPE_HEIGHT_M = 0.03
SAFETY_ENVELOPE_Z_OFFSET_M = 0.03
SAFETY_ENVELOPE_OPACITY = 0.28
SAFETY_ENVELOPE_TOO_NARROW_OPACITY = 0.45
DRAW_TOO_NARROW_GAP_LINES = True
TOO_NARROW_GAP_LINE_WIDTH_M = 0.035

# 畫在無人機上的安全範圍
DRAW_UAV_SAFETY_BUBBLE = True
UAV_BODY_PATH = "/World/quadrotor/body"
UAV_SAFETY_BUBBLE_ROOT = f"{UAV_BODY_PATH}/DebugSafetyBubble"
DRAW_UAV_PLANNING_BUBBLE = True
DRAW_UAV_VALIDATION_BUBBLE = True
UAV_SAFETY_BUBBLE_OPACITY = 0.28
UAV_SAFETY_BUBBLE_TOO_NARROW_OPACITY = 0.45

DRAW_RAW_ASTAR_PATH = False
DRAW_SIMPLIFIED_PATH = True
DRAW_WAYPOINT_MARKERS = True
DRAW_EXECUTED_TRAIL = False
PATH_VISUAL_Z_OFFSET_M = 0.04
PATH_LINE_WIDTH_M = 0.025
RAW_PATH_LINE_WIDTH_M = 0.015
EXECUTED_TRAIL_LINE_WIDTH_M = 0.035
WAYPOINT_MARKER_RADIUS_M = 0.035
EXECUTED_TRAIL_MIN_SPACING_M = 0.08
EXECUTED_TRAIL_MAX_POINTS = 1200

# Display colors. Values are RGB in 0.0~1.0.
COLOR_PLANNED_PATH = (0.0, 0.85, 1.0)
COLOR_RAW_PATH = (1.0, 0.65, 0.0)
COLOR_WAYPOINT_MARKER = (0.2, 1.0, 0.2)
COLOR_EXECUTED_TRAIL = (1.0, 0.1, 0.1)
COLOR_SAFETY_PLANNING_RADIUS = (1.0, 0.55, 0.0)
COLOR_SAFETY_VALIDATION_RADIUS = (1.0, 0.05, 0.05)
COLOR_SAFETY_TOO_NARROW_RADIUS = (0.75, 0.0, 0.0)
COLOR_TOO_NARROW_GAP_LINE = (1.0, 0.0, 0.0)


AUTO_LAND_AFTER_SUCCESS = True
AUTO_LAND_HOVER_SECONDS = 1.0

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ObstacleInfo:
    name: str
    prim_path: str
    world_position_isaac: List[float]
    bbox_center_isaac: List[float]
    bbox_size_isaac: List[float]
    estimated_radius_m: float
    center_ned: List[float]


@dataclass
class RedPointInfo:
    prim_path: str
    position_isaac: List[float]
    position_ned: List[float]


@dataclass
class Waypoint:
    label: str
    ned: List[float]


@dataclass
class AStarGridMap:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    resolution: float
    width: int
    height: int
    occupied: Dict[Tuple[int, int], str]
    occupied_cell_count: int
    obstacle_count: int
    extra_grid_clearance_m: float


@dataclass
class PlannerSummary:
    planner_type: str
    grid_resolution_m: float
    grid_min_x: float
    grid_max_x: float
    grid_min_y: float
    grid_max_y: float
    grid_width: int
    grid_height: int
    obstacle_count: int
    occupied_cell_count: int
    occupied_ratio: float
    astar_raw_path_point_count: int
    astar_waypoint_count: int
    path_is_safe: bool
    extra_grid_clearance_m: float
    use_direct_path_bias: bool
    direct_path_bias_weight: float
    use_lookahead_path_following: bool
    path_lookahead_m: float
    final_path_total_length_m: float


@dataclass
class PlannerResult:
    waypoints: List[Waypoint]
    raw_path_ned: List[List[float]]
    simplified_path_ned: List[List[float]]
    summary: PlannerSummary


@dataclass
class VehicleState:
    has_position: bool = False
    has_attitude: bool = False

    pos_x_ned: float = float("nan")
    pos_y_ned: float = float("nan")
    pos_z_ned: float = float("nan")

    vel_x_ned: float = float("nan")
    vel_y_ned: float = float("nan")
    vel_z_ned: float = float("nan")

    roll: float = float("nan")
    pitch: float = float("nan")
    yaw: float = 0.0


# ============================================================
# BASIC MATH UTILITIES
# ============================================================

def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def norm2(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def distance_xy(a: List[float], b: List[float]) -> float:
    return norm2(a[0] - b[0], a[1] - b[1])


def normalize_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def scale_xy_to_limit(vx: float, vy: float, max_speed: float) -> Tuple[float, float]:
    speed = norm2(vx, vy)

    if speed <= max_speed or speed < 1e-6:
        return vx, vy

    scale = max_speed / speed
    return vx * scale, vy * scale


def point_to_segment_distance_2d(
    point_xy: List[float],
    start_xy: List[float],
    end_xy: List[float],
) -> Tuple[float, float, List[float]]:
    px, py = point_xy
    ax, ay = start_xy
    bx, by = end_xy

    abx = bx - ax
    aby = by - ay
    ab_len2 = abx * abx + aby * aby

    if ab_len2 < 1e-9:
        closest = [ax, ay]
        return distance_xy(point_xy, closest), 0.0, closest

    t = ((px - ax) * abx + (py - ay) * aby) / ab_len2
    t_clamped = clamp(t, 0.0, 1.0)

    closest = [ax + t_clamped * abx, ay + t_clamped * aby]
    dist = distance_xy(point_xy, closest)

    return dist, t_clamped, closest


def format_vec(vec: List[float]) -> str:
    return "[" + ", ".join(f"{float(v): .3f}" for v in vec) + "]"


# ============================================================
# POLYLINE / LOOKAHEAD PATH UTILITIES
# ============================================================

def compute_polyline_lengths(path: List[List[float]]) -> Tuple[List[float], float]:
    """Return cumulative 2D arc length and total length for a NED polyline."""
    if not path:
        return [], 0.0

    cumulative = [0.0]
    total = 0.0

    for i in range(len(path) - 1):
        total += distance_xy(path[i], path[i + 1])
        cumulative.append(total)

    return cumulative, total


def sample_polyline_at_progress(
    path: List[List[float]],
    cumulative_lengths: List[float],
    target_progress_m: float,
) -> Tuple[List[float], int]:
    """Sample a point on a polyline by arc length progress."""
    if not path:
        raise RuntimeError("Cannot sample an empty path.")

    if len(path) == 1:
        return list(path[0]), 0

    target_progress_m = clamp(
        target_progress_m,
        0.0,
        cumulative_lengths[-1],
    )

    for i in range(len(path) - 1):
        seg_start_s = cumulative_lengths[i]
        seg_end_s = cumulative_lengths[i + 1]

        if target_progress_m <= seg_end_s or i == len(path) - 2:
            seg_len = max(1e-9, seg_end_s - seg_start_s)
            t = (target_progress_m - seg_start_s) / seg_len
            return interpolate_ned(path[i], path[i + 1], t), i

    return list(path[-1]), len(path) - 2


def find_closest_point_on_polyline(
    point_ned: List[float],
    path: List[List[float]],
    cumulative_lengths: List[float],
) -> Tuple[float, int, List[float], float]:
    """Return progress, segment index, closest point, and lateral distance."""
    if len(path) < 2:
        return 0.0, 0, list(path[0]), distance_xy(point_ned, path[0])

    best_progress = 0.0
    best_segment_index = 0
    best_closest = list(path[0])
    best_distance = float("inf")

    point_xy = [point_ned[0], point_ned[1]]

    for i in range(len(path) - 1):
        dist, t, closest_xy = point_to_segment_distance_2d(
            point_xy=point_xy,
            start_xy=[path[i][0], path[i][1]],
            end_xy=[path[i + 1][0], path[i + 1][1]],
        )

        if dist < best_distance:
            seg_len = cumulative_lengths[i + 1] - cumulative_lengths[i]
            best_progress = cumulative_lengths[i] + t * seg_len
            best_segment_index = i
            best_closest = [closest_xy[0], closest_xy[1], point_ned[2]]
            best_distance = dist

    return best_progress, best_segment_index, best_closest, best_distance


def compute_turn_angle_at_segment(path: List[List[float]], segment_index: int) -> float:
    """Return local path turn angle in radians near the active segment."""
    if len(path) < 3:
        return 0.0

    pivot_index = clamp(segment_index + 1, 1, len(path) - 2)
    pivot_index = int(pivot_index)

    a = path[pivot_index - 1]
    b = path[pivot_index]
    c = path[pivot_index + 1]

    v1x = b[0] - a[0]
    v1y = b[1] - a[1]
    v2x = c[0] - b[0]
    v2y = c[1] - b[1]

    n1 = norm2(v1x, v1y)
    n2 = norm2(v2x, v2y)

    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
    dot = clamp(dot, -1.0, 1.0)
    return math.acos(dot)


# ============================================================
# COORDINATE TRANSFORM
# ============================================================

def isaac_to_ned_position(isaac_xyz: List[float]) -> List[float]:
    """
    Convert Isaac world coordinates to PX4 local NED coordinates.

    Known mapping:
    Isaac goal [3, 5, 0] becomes PX4 NED approximately [5, 3, 0]
    when SWAP_XY=True.

    Isaac z-up becomes NED z-down.
    """
    ix = float(isaac_xyz[0])
    iy = float(isaac_xyz[1])
    iz = float(isaac_xyz[2])

    if SWAP_XY:
        nx = iy
        ny = ix
    else:
        nx = ix
        ny = iy

    nz = -iz

    return [
        nx + PX4_NED_OFFSET_X,
        ny + PX4_NED_OFFSET_Y,
        nz + PX4_NED_OFFSET_Z,
    ]


def isaac_ground_point_to_flight_ned(isaac_xyz: List[float]) -> List[float]:
    ned = isaac_to_ned_position(isaac_xyz)
    ned[2] = -FLIGHT_ALTITUDE_M + PX4_NED_OFFSET_Z
    return ned


def ned_to_isaac_position(ned_xyz: List[float]) -> List[float]:
    """Convert PX4 local NED coordinates back to Isaac world coordinates."""
    nx = float(ned_xyz[0]) - PX4_NED_OFFSET_X
    ny = float(ned_xyz[1]) - PX4_NED_OFFSET_Y
    nz = float(ned_xyz[2]) - PX4_NED_OFFSET_Z

    if SWAP_XY:
        ix = ny
        iy = nx
    else:
        ix = nx
        iy = ny

    iz = -nz
    return [ix, iy, iz]


# ============================================================
# SCENE READER
# ============================================================

def get_stage() -> Usd.Stage:
    stage = omni.usd.get_context().get_stage()

    if stage is None:
        raise RuntimeError("No active USD stage found.")

    return stage


def refresh_stage_after_manual_edits():
    """Give Isaac Sim a couple of UI/update ticks before reading live transforms.

    This helps when the user just dragged obstacle prims in the viewport and then
    immediately runs the planner. If the Kit update API is unavailable, reading the
    USD Stage still proceeds normally.
    """
    if not SCENE_READER_USE_LIVE_STAGE_TRANSFORMS:
        return

    ticks = max(0, int(SCENE_READER_UPDATE_TICKS))
    if ticks <= 0:
        return

    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        for _ in range(ticks):
            app.update()

        print(f"[SceneReader] Live Stage refresh completed: update_ticks={ticks}")

    except Exception as exc:
        print(f"[SceneReader] Warning: could not force Kit update ticks: {exc}")
        print("[SceneReader] Continuing with direct USD Stage read.")


def get_world_translation(prim) -> List[float]:
    xformable = UsdGeom.Xformable(prim)
    matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = matrix.ExtractTranslation()

    return [float(t[0]), float(t[1]), float(t[2])]


def get_bbox_purposes():
    purposes = [UsdGeom.Tokens.default_]

    for token_name in ["render", "proxy"]:
        token = getattr(UsdGeom.Tokens, token_name, None)
        if token is not None and token not in purposes:
            purposes.append(token)

    return purposes


def is_finite_vec(values: List[float]) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def compute_world_bbox(prim) -> Tuple[List[float], List[float]]:
    """Compute the live world-space bbox of a prim or its subtree.

    This intentionally computes a fresh BBoxCache each time so that manually moved
    obstacle positions are reflected in the planner.
    """
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        get_bbox_purposes(),
        False,
    )

    bound = bbox_cache.ComputeWorldBound(prim)
    aligned_box = bound.ComputeAlignedBox()

    min_v = aligned_box.GetMin()
    max_v = aligned_box.GetMax()

    center = (min_v + max_v) * 0.5
    size = max_v - min_v

    center_list = [float(center[0]), float(center[1]), float(center[2])]
    size_list = [
        abs(float(size[0])),
        abs(float(size[1])),
        abs(float(size[2])),
    ]

    if not is_finite_vec(center_list) or not is_finite_vec(size_list):
        raise RuntimeError(f"Invalid bbox for prim {prim.GetPath().pathString}")

    if max(size_list) < 1e-6:
        raise RuntimeError(f"Degenerate bbox for prim {prim.GetPath().pathString}")

    return center_list, size_list


def get_custom_string_attr(prim, attr_name: str) -> str:
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return ""

    try:
        value = attr.Get()
    except Exception:
        return ""

    if value is None:
        return ""

    return str(value)


def get_custom_float_attr(prim, attr_name: str) -> Optional[float]:
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return None
    try:
        value = float(attr.Get())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def prim_has_obstacle_metadata(prim) -> bool:
    return get_custom_string_attr(prim, "episode:object_type").lower() == "obstacle"


def prim_name_looks_like_obstacle(prim) -> bool:
    name = prim.GetName()
    return any(name.startswith(prefix) for prefix in OBSTACLE_NAME_PREFIXES)


def prim_has_boundable_descendant(prim) -> bool:
    if prim.IsA(UsdGeom.Boundable):
        return True

    for child in Usd.PrimRange(prim):
        if child.GetPath() == prim.GetPath():
            continue

        if child.IsA(UsdGeom.Boundable):
            return True

    return False


def should_include_obstacle_prim(prim) -> bool:
    if prim_has_obstacle_metadata(prim):
        return True

    if prim_name_looks_like_obstacle(prim):
        return True

    # Under the generated obstacle root, direct children are treated as obstacle objects
    # as long as they contain visible/boundable geometry.
    if prim_has_boundable_descendant(prim):
        return True

    return False


def collect_obstacle_prims_from_root(stage: Usd.Stage, root_path: str) -> List[Any]:
    root = stage.GetPrimAtPath(root_path)

    if not root or not root.IsValid():
        print(f"[SceneReader] No obstacle root found at {root_path}")
        return []

    candidates = []

    if READ_OBSTACLES_AS_LIVE_TOP_LEVEL_OBJECTS:
        for child in root.GetChildren():
            if not child or not child.IsValid():
                continue

            if should_include_obstacle_prim(child):
                candidates.append(child)

        return candidates

    for prim in Usd.PrimRange(root):
        if prim.GetPath().pathString == root_path:
            continue

        if not prim.IsValid():
            continue

        if prim.IsA(UsdGeom.Boundable) and should_include_obstacle_prim(prim):
            candidates.append(prim)

    return candidates


def make_obstacle_info_from_prim(prim) -> Optional[ObstacleInfo]:
    prim_path = prim.GetPath().pathString
    name = prim.GetName()

    try:
        world_position = get_world_translation(prim)
    except Exception:
        world_position = [float("nan"), float("nan"), float("nan")]

    try:
        bbox_center, bbox_size = compute_world_bbox(prim)
    except Exception as exc:
        print(f"[SceneReader] Warning: failed to compute live bbox for obstacle {prim_path}: {exc}")
        return None

    # Generated high-rises store their true yaw-invariant footprint
    # circumradius. Prefer that value so a rotated building remains safe without
    # the excessive inflation caused by taking the diagonal of its world AABB.
    # The bbox half-extent is retained as a live-stage/manual-scaling floor.
    metadata_radius = get_custom_float_attr(prim, "episode:radius")
    bbox_half_extent = 0.5 * max(float(bbox_size[0]), float(bbox_size[1]))
    if metadata_radius is not None and metadata_radius > 1e-6:
        estimated_radius = max(metadata_radius + 0.02, bbox_half_extent)
    else:
        estimated_radius = 0.5 * math.hypot(
            float(bbox_size[0]),
            float(bbox_size[1]),
        )

    if estimated_radius <= 1e-6:
        print(f"[SceneReader] Warning: obstacle {prim_path} has near-zero radius. Skipped.")
        return None

    center_ned = isaac_to_ned_position(bbox_center)

    return ObstacleInfo(
        name=name,
        prim_path=prim_path,
        world_position_isaac=world_position,
        bbox_center_isaac=bbox_center,
        bbox_size_isaac=bbox_size,
        estimated_radius_m=estimated_radius,
        center_ned=center_ned,
    )


def read_obstacles_from_stage() -> List[ObstacleInfo]:
    refresh_stage_after_manual_edits()

    stage = get_stage()
    root_paths = [OBSTACLE_ROOT_PATH] + list(EXTRA_OBSTACLE_ROOT_PATHS)

    obstacle_prims = []
    seen_paths = set()

    for root_path in root_paths:
        for prim in collect_obstacle_prims_from_root(stage, root_path):
            path = prim.GetPath().pathString
            if path in seen_paths:
                continue
            seen_paths.add(path)
            obstacle_prims.append(prim)

    obstacles = []

    for prim in obstacle_prims:
        obstacle = make_obstacle_info_from_prim(prim)
        if obstacle is not None:
            obstacles.append(obstacle)

    print(
        f"[SceneReader] Live obstacle read complete: "
        f"root_count={len(root_paths)}, obstacle_count={len(obstacles)}"
    )

    if obstacles:
        print("[SceneReader] These obstacle positions are from the CURRENT USD Stage, not old CSV/JSON.")

    return obstacles


def read_marker_position_live(selected_prim) -> Optional[List[float]]:
    try:
        bbox_center, _ = compute_world_bbox(selected_prim)
        return bbox_center
    except Exception:
        pass

    try:
        return get_world_translation(selected_prim)
    except Exception as exc:
        print(f"[SceneReader] Warning: failed to read marker transform: {exc}")
        return None


def read_red_point_from_stage() -> Optional[RedPointInfo]:
    stage = get_stage()

    selected_path = None
    selected_prim = None

    for candidate_path in RED_POINT_CANDIDATE_PATHS:
        prim = stage.GetPrimAtPath(candidate_path)

        if prim and prim.IsValid():
            selected_path = candidate_path
            selected_prim = prim
            break

    if selected_prim is None:
        print(
            "[SceneReader] No red point / target disk found. Tried: "
            + ", ".join(RED_POINT_CANDIDATE_PATHS)
        )
        return None

    position_isaac = read_marker_position_live(selected_prim)

    if position_isaac is None:
        return None

    position_ned = isaac_to_ned_position(position_isaac)

    print(
        f"[SceneReader] Live target marker read: path={selected_path}, "
        f"isaac={format_vec(position_isaac)}, ned={format_vec(position_ned)}"
    )

    return RedPointInfo(
        prim_path=selected_path,
        position_isaac=position_isaac,
        position_ned=position_ned,
    )


def resolve_final_goal_isaac(red_point: Optional[RedPointInfo]) -> List[float]:
    if USE_STAGE_TARGET_AS_FINAL_GOAL and red_point is not None:
        final_goal_isaac = [
            float(red_point.position_isaac[0]),
            float(red_point.position_isaac[1]),
            0.0,
        ]

        print(
            "[Main] USE_STAGE_TARGET_AS_FINAL_GOAL=True. "
            f"Using live target marker as final goal: {format_vec(final_goal_isaac)}"
        )
        return final_goal_isaac

    final_goal_isaac = list(FINAL_GOAL_ISAAC)
    print(
        "[Main] Using fixed FINAL_GOAL_ISAAC as final goal: "
        f"{format_vec(final_goal_isaac)}"
    )
    return final_goal_isaac


def print_scene_summary(obstacles: List[ObstacleInfo], red_point: Optional[RedPointInfo]):
    print("")
    print("========== Scene Ground Truth ==========")
    print(f"Obstacle count: {len(obstacles)}")

    for obs in obstacles:
        print(
            f"- {obs.name}: "
            f"path={obs.prim_path}, "
            f"center_isaac={format_vec(obs.bbox_center_isaac)}, "
            f"size={format_vec(obs.bbox_size_isaac)}, "
            f"radius={obs.estimated_radius_m:.3f}, "
            f"center_ned={format_vec(obs.center_ned)}"
        )

    if red_point is not None:
        print(
            f"RedPoint: "
            f"isaac={format_vec(red_point.position_isaac)}, "
            f"ned={format_vec(red_point.position_ned)}"
        )
    else:
        print("RedPoint: None")

    print("========================================")
    print("")


def print_planner_parameter_summary():
    print("")
    print("========== Planner Parameter Summary ==========")
    print(f"[Config] preset={EXPERIMENT_PRESET_NAME}")
    print(f"[Config] SCENE_READER_UPDATE_TICKS={SCENE_READER_UPDATE_TICKS}")
    print(f"[Config] GRID_RESOLUTION_M={GRID_RESOLUTION_M:.3f}")
    print(f"[Config] GRID_MARGIN_M={GRID_MARGIN_M:.3f}")
    print(f"[Config] UAV_SAFETY_RADIUS_M={UAV_SAFETY_RADIUS_M:.3f}")
    print(f"[Config] OBSTACLE_SAFETY_MARGIN_M={OBSTACLE_SAFETY_MARGIN_M:.3f}")
    print(f"[Config] MIN_SEGMENT_CLEARANCE_M={MIN_SEGMENT_CLEARANCE_M:.3f}")
    print(f"[Config] required_surface_gap_for_two_obstacles≈{2.0 * (UAV_SAFETY_RADIUS_M + OBSTACLE_SAFETY_MARGIN_M + MIN_SEGMENT_CLEARANCE_M):.3f} m")
    print(f"[Config] USE_DIRECT_PATH_BIAS={USE_DIRECT_PATH_BIAS}")
    print(f"[Config] DIRECT_PATH_BIAS_WEIGHT={DIRECT_PATH_BIAS_WEIGHT:.3f}")
    print(f"[Config] PATH_SIMPLIFY_TOLERANCE_M={PATH_SIMPLIFY_TOLERANCE_M:.3f}")
    print(f"[Config] MAX_WAYPOINT_SPACING_M={MAX_WAYPOINT_SPACING_M:.3f}")
    print(f"[Config] PATH_LOOKAHEAD_M={PATH_LOOKAHEAD_M:.3f}")
    print(f"[Config] LOOKAHEAD_SEGMENT_CHECK_EXTRA_CLEARANCE_M={LOOKAHEAD_SEGMENT_CHECK_EXTRA_CLEARANCE_M:.3f}")
    print(f"[Config] MAX_SPEED_XY_MPS={MAX_SPEED_XY_MPS:.3f}")
    print(f"[Config] WAIT_FOREVER_FOR_PX4_HEARTBEAT={WAIT_FOREVER_FOR_PX4_HEARTBEAT}")
    print(f"[Config] PX4_HEARTBEAT_TIMEOUT_S={PX4_HEARTBEAT_TIMEOUT_S:.1f}")
    print(f"[Config] REQUIRE_ARM_CONFIRMATION={REQUIRE_ARM_CONFIRMATION}")
    print(f"[Config] REQUIRE_TAKEOFF_REACHED_BEFORE_MISSION={REQUIRE_TAKEOFF_REACHED_BEFORE_MISSION}")
    print(f"[Config] MIN_TAKEOFF_ALTITUDE_REACHED_M={MIN_TAKEOFF_ALTITUDE_REACHED_M:.3f}")
    print(f"[Config] ENABLE_COMMAND_SMOOTHING={ENABLE_COMMAND_SMOOTHING}")
    print(f"[Config] MAX_ACCEL_XY_MPS2={MAX_ACCEL_XY_MPS2:.3f}")
    print(f"[Config] MAX_ACCEL_Z_MPS2={MAX_ACCEL_Z_MPS2:.3f}")
    print(f"[Config] MAX_YAW_ACCEL_RADPS2={MAX_YAW_ACCEL_RADPS2:.3f}")
    print(f"[Config] ENABLE_OVERFLY_SHORT_OBSTACLES={ENABLE_OVERFLY_SHORT_OBSTACLES}")
    print(f"[Config] OVERFLY_VERTICAL_CLEARANCE_M={OVERFLY_VERTICAL_CLEARANCE_M:.3f}")
    print(f"[Config] USE_CLEARANCE_AWARE_COST_FIELD={USE_CLEARANCE_AWARE_COST_FIELD}")
    print(f"[Config] SOFT_CLEARANCE_COST_RADIUS_M={SOFT_CLEARANCE_COST_RADIUS_M:.3f}")
    print(f"[Config] CLEARANCE_COST_WEIGHT={CLEARANCE_COST_WEIGHT:.3f}")
    print(f"[Config] GRID_DISCRETIZATION_MARGIN_M={GRID_DISCRETIZATION_MARGIN_M:.3f}")
    print(f"[Config] DRAW_PATH_VISUALIZATION_IN_STAGE={DRAW_PATH_VISUALIZATION_IN_STAGE}")
    print(f"[Config] DRAW_SAFETY_ENVELOPE={DRAW_SAFETY_ENVELOPE}")
    print(f"[Config] SAFETY_ENVELOPE_ROOT={SAFETY_ENVELOPE_ROOT}")
    print("[Config] planning_radius = obstacle_radius + UAV_SAFETY_RADIUS_M + OBSTACLE_SAFETY_MARGIN_M")
    print("[Config] validation_radius = planning_radius + MIN_SEGMENT_CLEARANCE_M")
    print("===============================================")
    print("")


def compute_obstacle_gap_diagnostics(
    obstacles: List[ObstacleInfo],
) -> List[Dict[str, Any]]:
    diagnostics: List[Dict[str, Any]] = []

    if len(obstacles) < 2:
        return diagnostics

    required_surface_gap = 2.0 * (
        UAV_SAFETY_RADIUS_M + OBSTACLE_SAFETY_MARGIN_M + MIN_SEGMENT_CLEARANCE_M
    )

    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            a = obstacles[i]
            b = obstacles[j]
            center_gap = distance_xy(a.center_ned, b.center_ned)
            surface_gap = center_gap - a.estimated_radius_m - b.estimated_radius_m
            pass_margin = surface_gap - required_surface_gap
            diagnostics.append(
                {
                    "pass_margin": pass_margin,
                    "surface_gap": surface_gap,
                    "center_gap": center_gap,
                    "required_surface_gap": required_surface_gap,
                    "name_a": a.name,
                    "name_b": b.name,
                    "obstacle_a": a,
                    "obstacle_b": b,
                    "status": "PASSABLE" if pass_margin >= 0.0 else "TOO_NARROW",
                }
            )

    diagnostics.sort(key=lambda item: item["pass_margin"])
    return diagnostics


def print_obstacle_gap_diagnostics(
    obstacles: List[ObstacleInfo],
    max_pairs: int = 12,
) -> List[Dict[str, Any]]:
    diagnostics = compute_obstacle_gap_diagnostics(obstacles)

    if len(obstacles) < 2:
        return diagnostics

    required_surface_gap = diagnostics[0]["required_surface_gap"] if diagnostics else 0.0

    print("")
    print("========== Obstacle Gap Diagnostics ==========")
    print(f"[Gap] required surface gap for a conservative pass: {required_surface_gap:.3f} m")
    print("[Gap] pass_margin = actual_surface_gap - required_surface_gap")
    print("[Gap] Negative pass_margin means the planner will treat this gap as too narrow.")
    print("[Gap] This is a geometry diagnostic. A* still also depends on grid resolution and path cost.")

    for item in diagnostics[:max_pairs]:
        print(
            f"[Gap] {item['name_a']} <-> {item['name_b']}: "
            f"surface_gap={item['surface_gap']:.3f} m, "
            f"center_gap={item['center_gap']:.3f} m, "
            f"required_surface_gap={item['required_surface_gap']:.3f} m, "
            f"pass_margin={item['pass_margin']:.3f} m, "
            f"status={item['status']}"
        )

    print("==============================================")
    print("")

    return diagnostics

def save_scene_summary_csv(
    episode_id: str,
    obstacles: List[ObstacleInfo],
    red_point: Optional[RedPointInfo],
) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)

    csv_path = os.path.join(LOG_DIR, f"scene_objects_{episode_id}.csv")

    fieldnames = [
        "episode_id",
        "object_type",
        "name",
        "prim_path",
        "isaac_x",
        "isaac_y",
        "isaac_z",
        "ned_x",
        "ned_y",
        "ned_z",
        "bbox_size_x",
        "bbox_size_y",
        "bbox_size_z",
        "estimated_radius_m",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for obs in obstacles:
            writer.writerow(
                {
                    "episode_id": episode_id,
                    "object_type": "obstacle",
                    "name": obs.name,
                    "prim_path": obs.prim_path,
                    "isaac_x": obs.bbox_center_isaac[0],
                    "isaac_y": obs.bbox_center_isaac[1],
                    "isaac_z": obs.bbox_center_isaac[2],
                    "ned_x": obs.center_ned[0],
                    "ned_y": obs.center_ned[1],
                    "ned_z": obs.center_ned[2],
                    "bbox_size_x": obs.bbox_size_isaac[0],
                    "bbox_size_y": obs.bbox_size_isaac[1],
                    "bbox_size_z": obs.bbox_size_isaac[2],
                    "estimated_radius_m": obs.estimated_radius_m,
                }
            )

        if red_point is not None:
            writer.writerow(
                {
                    "episode_id": episode_id,
                    "object_type": "red_point",
                    "name": "RedPoint",
                    "prim_path": red_point.prim_path,
                    "isaac_x": red_point.position_isaac[0],
                    "isaac_y": red_point.position_isaac[1],
                    "isaac_z": red_point.position_isaac[2],
                    "ned_x": red_point.position_ned[0],
                    "ned_y": red_point.position_ned[1],
                    "ned_z": red_point.position_ned[2],
                    "bbox_size_x": "",
                    "bbox_size_y": "",
                    "bbox_size_z": "",
                    "estimated_radius_m": "",
                }
            )

    print(f"[Logger] Scene summary CSV saved: {csv_path}")
    return csv_path

def save_ros2_waypoints_json(
    output_path: str,
    episode_id: str,
    start_ned: List[float],
    final_goal_ned: List[float],
    planning_result: PlannerResult,
) -> str:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    waypoints = [list(start_ned)]

    for waypoint in planning_result.waypoints:
        waypoints.append([
            float(waypoint.ned[0]),
            float(waypoint.ned[1]),
            float(waypoint.ned[2]),
        ])

    payload = {
        "episode_id": episode_id,
        "frame": "PX4_NED",
        "source": "isaac_astar_exporter",
        "flight_altitude_m": float(FLIGHT_ALTITUDE_M),
        "start_ned": [float(v) for v in start_ned],
        "final_goal_ned": [float(v) for v in final_goal_ned],
        "waypoint_count": len(waypoints),
        "waypoints": waypoints,
        "planner_summary": {
            "planner_type": planning_result.summary.planner_type,
            "grid_resolution_m": float(planning_result.summary.grid_resolution_m),
            "obstacle_count": int(planning_result.summary.obstacle_count),
            "astar_raw_path_point_count": int(planning_result.summary.astar_raw_path_point_count),
            "astar_waypoint_count": int(planning_result.summary.astar_waypoint_count),
            "path_is_safe": bool(planning_result.summary.path_is_safe),
            "final_path_total_length_m": float(planning_result.summary.final_path_total_length_m),
        },
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print("")
    print("========== ROS2 Waypoints Export ==========")
    print(f"[Exporter] Saved ROS2 waypoint JSON: {output_path}")
    print(f"[Exporter] waypoint_count={len(waypoints)}")
    print("[Exporter] frame=PX4_NED")

    for index, waypoint in enumerate(waypoints):
        print(
            f"[Exporter] waypoint {index:03d}: "
            f"[{waypoint[0]:+.3f}, {waypoint[1]:+.3f}, {waypoint[2]:+.3f}]"
        )

    print("==========================================")
    print("")

    return output_path



def build_ros2_path_waypoints(
    start_ned: List[float],
    planning_result: PlannerResult,
) -> List[List[float]]:
    """Build a de-duplicated PX4-NED path for ROS 2 publication."""
    candidates = [
        [float(start_ned[0]), float(start_ned[1]), float(start_ned[2])],
    ]

    for waypoint in planning_result.waypoints:
        candidates.append([
            float(waypoint.ned[0]),
            float(waypoint.ned[1]),
            float(waypoint.ned[2]),
        ])

    cleaned = []
    for waypoint in candidates:
        if cleaned:
            dx = waypoint[0] - cleaned[-1][0]
            dy = waypoint[1] - cleaned[-1][1]
            dz = waypoint[2] - cleaned[-1][2]
            if math.sqrt(dx * dx + dy * dy + dz * dz) < 1e-6:
                continue
        cleaned.append(waypoint)

    if len(cleaned) < 2:
        raise RuntimeError("The planned ROS2 path must contain at least two distinct waypoints.")

    return cleaned


class IsaacAStarRos2PathPublisher:
    """Keep the latest A* path alive as a transient-local ROS 2 Path publisher."""

    def __init__(
        self,
        waypoints_ned: List[List[float]],
        topic_name: str = ROS2_PATH_TOPIC,
        frame_id: str = ROS2_PATH_FRAME_ID,
        publish_rate_hz: float = ROS2_PATH_PUBLISH_RATE_HZ,
    ):
        if not ROS2_PATH_AVAILABLE:
            raise RuntimeError(
                "ROS2 nav_msgs/Path is unavailable in Isaac Sim Python. "
                f"Import error: {ROS2_PATH_IMPORT_ERROR}"
            )

        self.waypoints_ned = [list(map(float, waypoint)) for waypoint in waypoints_ned]
        self.topic_name = str(topic_name)
        self.frame_id = str(frame_id)
        self.publish_rate_hz = max(0.2, float(publish_rate_hz))
        self.publish_interval_s = 1.0 / self.publish_rate_hz

        self.node = None
        self.publisher = None
        self.update_subscription = None
        self.is_running = False
        self.last_publish_wall_time = 0.0
        self.publish_count = 0

    def start(self):
        previous = getattr(builtins, "_astar_ros2_path_publisher", None)
        if previous is not None and previous is not self:
            try:
                previous.stop()
            except Exception as exc:
                print(f"[ROS2PathPublisher] Warning: failed to stop previous publisher: {exc}")

        if not rclpy.ok():
            rclpy.init(args=None)

        self.node = rclpy.create_node(ROS2_PATH_NODE_NAME)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.node.create_publisher(RosPath, self.topic_name, qos)

        self.update_subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(
                self._on_update,
                name="IsaacAStarRos2PathPublisherUpdate",
            )
        )

        self.is_running = True
        self.last_publish_wall_time = 0.0
        self.publish_count = 0

        builtins._astar_ros2_path_publisher = self
        builtins.stop_astar_ros2_path_publisher = self.stop
        builtins.print_astar_ros2_path_status = self.print_status

        self.publish_path(force=True)

        print("")
        print("========== ROS2 A* Path Publisher ==========")
        print(f"[ROS2PathPublisher] topic={self.topic_name}")
        print(f"[ROS2PathPublisher] frame_id={self.frame_id}")
        print(f"[ROS2PathPublisher] waypoint_count={len(self.waypoints_ned)}")
        print(f"[ROS2PathPublisher] publish_rate_hz={self.publish_rate_hz:.2f}")
        print("[ROS2PathPublisher] Stop with: builtins.stop_astar_ros2_path_publisher()")
        print("============================================")
        print("")

    def build_message(self):
        message = RosPath()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id

        for waypoint in self.waypoints_ned:
            pose = PoseStamped()
            pose.header.stamp = message.header.stamp
            pose.header.frame_id = self.frame_id
            pose.pose.position.x = float(waypoint[0])
            pose.pose.position.y = float(waypoint[1])
            pose.pose.position.z = float(waypoint[2])
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)

        return message

    def publish_path(self, force=False):
        if not self.is_running or self.publisher is None or self.node is None:
            return

        now = time.time()
        if not force and now - self.last_publish_wall_time < self.publish_interval_s:
            return

        message = self.build_message()
        self.publisher.publish(message)
        self.last_publish_wall_time = now
        self.publish_count += 1

        if self.publish_count == 1 or self.publish_count % 5 == 0:
            print(
                f"[ROS2PathPublisher] Published path #{self.publish_count}: "
                f"poses={len(message.poses)}, topic={self.topic_name}"
            )

    def _on_update(self, _event):
        if not self.is_running:
            return

        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception as exc:
            print(f"[ROS2PathPublisher] Warning: rclpy.spin_once failed: {exc}")

        self.publish_path(force=False)

    def print_status(self):
        print(
            f"[ROS2PathPublisher] running={self.is_running}, "
            f"topic={self.topic_name}, frame={self.frame_id}, "
            f"waypoints={len(self.waypoints_ned)}, published={self.publish_count}"
        )

    def stop(self):
        self.is_running = False

        if self.update_subscription is not None:
            try:
                self.update_subscription.unsubscribe()
            except Exception:
                pass
            self.update_subscription = None

        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception as exc:
                print(f"[ROS2PathPublisher] Warning: failed to destroy node: {exc}")
            self.node = None
            self.publisher = None

        if getattr(builtins, "_astar_ros2_path_publisher", None) is self:
            builtins._astar_ros2_path_publisher = None

        print("[ROS2PathPublisher] Stopped.")


# ============================================================
# A* OCCUPANCY GRID PLANNER
# ============================================================

def obstacle_planning_radius(obstacle: ObstacleInfo) -> float:
    """Return the A* planning inflated radius.

    Why this formula:
    - The UAV is not a point. It has propellers and a physical footprint.
    - Expanding the obstacle lets the planner treat the UAV center as a point.
    - This is the standard configuration-space / Minkowski-sum idea.
    - The static margin covers tracking error, bbox uncertainty, small attitude wobble,
      and general safety reserve.
    """
    return (
        float(obstacle.estimated_radius_m)
        + float(UAV_SAFETY_RADIUS_M)
        + float(OBSTACLE_SAFETY_MARGIN_M)
    )


def obstacle_validation_radius(obstacle: ObstacleInfo) -> float:
    """Return the stricter radius for continuous segment validation."""
    return obstacle_planning_radius(obstacle) + float(MIN_SEGMENT_CLEARANCE_M)


def obstacle_effective_forbidden_radius(
    obstacle: ObstacleInfo,
    include_segment_validation_margin: bool = False,
) -> float:
    """Return a decomposed effective forbidden radius.

    This keeps your original parameter names working while documenting the future
    data-driven version. Later, TRACKING_ERROR_MARGIN_M can be estimated from CSV as
    actual distance from the UAV to the planned polyline.
    """
    radius = (
        float(obstacle.estimated_radius_m)
        + float(UAV_PHYSICAL_RADIUS_M)
        + float(STATIC_SAFETY_MARGIN_M)
        + float(TRACKING_ERROR_MARGIN_M)
        + float(GRID_DISCRETIZATION_MARGIN_M)
        + float(CONTROL_RESPONSE_MARGIN_M)
        + float(SENSOR_OR_STATE_UNCERTAINTY_M)
    )

    if include_segment_validation_margin:
        radius += float(SEGMENT_VALIDATION_MARGIN_M)

    return radius


def obstacle_inflated_radius(obstacle: ObstacleInfo) -> float:
    # Backward-compatible name used by the existing planner.
    return obstacle_planning_radius(obstacle)

def compute_grid_bounds(
    start_ned: List[float],
    goal_ned: List[float],
    obstacles: List[ObstacleInfo],
    extra_grid_clearance_m: float = 0.0,
) -> Tuple[float, float, float, float]:
    xs = [float(start_ned[0]), float(goal_ned[0])]
    ys = [float(start_ned[1]), float(goal_ned[1])]

    for obs in obstacles:
        radius = obstacle_inflated_radius(obs) + extra_grid_clearance_m
        xs.extend([obs.center_ned[0] - radius, obs.center_ned[0] + radius])
        ys.extend([obs.center_ned[1] - radius, obs.center_ned[1] + radius])

    min_x = min(xs) - GRID_MARGIN_M
    max_x = max(xs) + GRID_MARGIN_M
    min_y = min(ys) - GRID_MARGIN_M
    max_y = max(ys) + GRID_MARGIN_M

    return min_x, max_x, min_y, max_y


def ned_to_grid_index(ned_xy: List[float], grid_map: AStarGridMap) -> Tuple[int, int]:
    ix = int(round((float(ned_xy[0]) - grid_map.min_x) / grid_map.resolution))
    iy = int(round((float(ned_xy[1]) - grid_map.min_y) / grid_map.resolution))
    return ix, iy


def grid_index_to_ned(index: Tuple[int, int], grid_map: AStarGridMap) -> List[float]:
    ix, iy = index
    x = grid_map.min_x + float(ix) * grid_map.resolution
    y = grid_map.min_y + float(iy) * grid_map.resolution
    z = -FLIGHT_ALTITUDE_M + PX4_NED_OFFSET_Z
    return [x, y, z]


def is_grid_index_inside(index: Tuple[int, int], grid_map: AStarGridMap) -> bool:
    ix, iy = index
    return 0 <= ix < grid_map.width and 0 <= iy < grid_map.height


def build_occupancy_grid(
    start_ned: List[float],
    goal_ned: List[float],
    obstacles: List[ObstacleInfo],
    extra_grid_clearance_m: float = 0.0,
) -> AStarGridMap:
    min_x, max_x, min_y, max_y = compute_grid_bounds(
        start_ned=start_ned,
        goal_ned=goal_ned,
        obstacles=obstacles,
        extra_grid_clearance_m=extra_grid_clearance_m,
    )

    width = int(math.ceil((max_x - min_x) / GRID_RESOLUTION_M)) + 1
    height = int(math.ceil((max_y - min_y) / GRID_RESOLUTION_M)) + 1

    grid_map = AStarGridMap(
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        resolution=GRID_RESOLUTION_M,
        width=width,
        height=height,
        occupied={},
        occupied_cell_count=0,
        obstacle_count=len(obstacles),
        extra_grid_clearance_m=extra_grid_clearance_m,
    )

    for obs in obstacles:
        obstacle_xy = [obs.center_ned[0], obs.center_ned[1]]
        radius = obstacle_inflated_radius(obs) + extra_grid_clearance_m

        ix_min = int(math.floor((obstacle_xy[0] - radius - grid_map.min_x) / grid_map.resolution))
        ix_max = int(math.ceil((obstacle_xy[0] + radius - grid_map.min_x) / grid_map.resolution))
        iy_min = int(math.floor((obstacle_xy[1] - radius - grid_map.min_y) / grid_map.resolution))
        iy_max = int(math.ceil((obstacle_xy[1] + radius - grid_map.min_y) / grid_map.resolution))

        ix_min = max(0, ix_min)
        ix_max = min(grid_map.width - 1, ix_max)
        iy_min = max(0, iy_min)
        iy_max = min(grid_map.height - 1, iy_max)

        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                cell_ned = grid_index_to_ned((ix, iy), grid_map)
                if distance_xy(cell_ned, obstacle_xy) <= radius:
                    grid_map.occupied[(ix, iy)] = obs.name

    grid_map.occupied_cell_count = len(grid_map.occupied)
    return grid_map


def is_cell_occupied(index: Tuple[int, int], grid_map: AStarGridMap) -> bool:
    return index in grid_map.occupied


def point_clearance_to_inflated_obstacles(
    point_ned: List[float],
    obstacles: List[ObstacleInfo],
    extra_clearance_m: float = 0.0,
) -> Tuple[float, str]:
    """Return clearance from a point to the closest inflated obstacle boundary.

    Positive means the point is outside the inflated obstacle.
    Negative means the point is inside the inflated obstacle.
    """
    if not obstacles:
        return float("inf"), ""

    point_xy = [float(point_ned[0]), float(point_ned[1])]
    best_clearance = float("inf")
    best_name = ""

    for obs in obstacles:
        obs_xy = [obs.center_ned[0], obs.center_ned[1]]
        required_distance = obstacle_inflated_radius(obs) + extra_clearance_m
        clearance = distance_xy(point_xy, obs_xy) - required_distance

        if clearance < best_clearance:
            best_clearance = clearance
            best_name = obs.name

    return best_clearance, best_name


def find_nearest_free_grid_index(
    reference_index: Tuple[int, int],
    grid_map: AStarGridMap,
    max_search_radius_m: float,
) -> Optional[Tuple[int, int]]:
    """Find the nearest unoccupied grid cell around an occupied endpoint cell.

    This is only used to handle grid-rounding at exact start/goal points.
    It does not override the exact endpoint clearance safety check.
    """
    if is_grid_index_inside(reference_index, grid_map) and not is_cell_occupied(reference_index, grid_map):
        return reference_index

    max_steps = max(1, int(math.ceil(max_search_radius_m / grid_map.resolution)))
    best_index = None
    best_dist2 = float("inf")

    rx, ry = reference_index

    for dx in range(-max_steps, max_steps + 1):
        for dy in range(-max_steps, max_steps + 1):
            candidate = (rx + dx, ry + dy)

            if not is_grid_index_inside(candidate, grid_map):
                continue

            if is_cell_occupied(candidate, grid_map):
                continue

            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_index = candidate

    return best_index


def prepare_endpoint_index_for_astar(
    endpoint_name: str,
    endpoint_ned: List[float],
    endpoint_index: Tuple[int, int],
    grid_map: AStarGridMap,
    obstacles: List[ObstacleInfo],
    extra_grid_clearance_m: float,
) -> Optional[Tuple[int, int]]:
    """Return a usable grid index for start/goal, or None if unsafe.

    If the exact endpoint is safe but its rounded grid cell is occupied, this function
    searches for a nearby free cell. This avoids false failures caused by grid rounding.
    """
    exact_clearance, nearest_obstacle = point_clearance_to_inflated_obstacles(
        point_ned=endpoint_ned,
        obstacles=obstacles,
        extra_clearance_m=extra_grid_clearance_m,
    )

    print(
        f"[AStar] {endpoint_name} exact clearance to inflated obstacles "
        f"with extra_clearance={extra_grid_clearance_m:.3f} m: "
        f"{exact_clearance:.3f} m, nearest={nearest_obstacle}"
    )

    if not is_grid_index_inside(endpoint_index, grid_map):
        print(f"[AStar] ERROR: {endpoint_name} index is outside grid: {endpoint_index}")
        return None

    if not is_cell_occupied(endpoint_index, grid_map):
        return endpoint_index

    blocker = grid_map.occupied.get(endpoint_index, "unknown")

    if exact_clearance < ENDPOINT_MIN_EXACT_CLEARANCE_M:
        print(
            f"[AStar] ERROR: {endpoint_name} exact point is inside or too close to an inflated obstacle. "
            f"blocked_cell_by={blocker}, index={endpoint_index}, "
            f"exact_clearance={exact_clearance:.3f} m"
        )
        return None

    nearest_free = find_nearest_free_grid_index(
        reference_index=endpoint_index,
        grid_map=grid_map,
        max_search_radius_m=ENDPOINT_FREE_SEARCH_RADIUS_M,
    )

    if nearest_free is None:
        print(
            f"[AStar] ERROR: {endpoint_name} cell is occupied by {blocker}, "
            f"and no nearby free cell was found within {ENDPOINT_FREE_SEARCH_RADIUS_M:.3f} m."
        )
        return None

    nearest_free_ned = grid_index_to_ned(nearest_free, grid_map)
    print(
        f"[AStar] WARNING: {endpoint_name} rounded grid cell is occupied by {blocker}, "
        f"but exact point is clear. Using nearest free cell {nearest_free}, "
        f"ned={format_vec(nearest_free_ned)} for A* search."
    )

    return nearest_free


def astar_search_grid(
    grid_map: AStarGridMap,
    start_index: Tuple[int, int],
    goal_index: Tuple[int, int],
    start_ned: List[float],
    goal_ned: List[float],
    obstacles: List[ObstacleInfo],
) -> Optional[List[Tuple[int, int]]]:
    if not is_grid_index_inside(start_index, grid_map):
        print(f"[AStar] ERROR: start index is outside grid: {start_index}")
        return None

    if not is_grid_index_inside(goal_index, grid_map):
        print(f"[AStar] ERROR: goal index is outside grid: {goal_index}")
        return None

    if is_cell_occupied(start_index, grid_map):
        blocker = grid_map.occupied.get(start_index, "unknown")
        print(f"[AStar] ERROR: start cell is occupied by {blocker}. start_index={start_index}")
        return None

    if is_cell_occupied(goal_index, grid_map):
        blocker = grid_map.occupied.get(goal_index, "unknown")
        print(f"[AStar] ERROR: goal cell is occupied by {blocker}. goal_index={goal_index}")
        return None

    sqrt2 = math.sqrt(2.0)
    neighbors = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, sqrt2),
        (1, -1, sqrt2),
        (-1, 1, sqrt2),
        (-1, -1, sqrt2),
    ]

    def heuristic(index: Tuple[int, int]) -> float:
        return math.hypot(goal_index[0] - index[0], goal_index[1] - index[1])

    def direct_path_soft_cost(index: Tuple[int, int]) -> float:
        if not USE_DIRECT_PATH_BIAS or DIRECT_PATH_BIAS_WEIGHT <= 0.0:
            return 1.0

        candidate_ned = grid_index_to_ned(index, grid_map)
        distance_to_line, _, _ = point_to_segment_distance_2d(
            point_xy=[candidate_ned[0], candidate_ned[1]],
            start_xy=[start_ned[0], start_ned[1]],
            end_xy=[goal_ned[0], goal_ned[1]],
        )

        return 1.0 + DIRECT_PATH_BIAS_WEIGHT * distance_to_line

    def clearance_aware_soft_cost(index: Tuple[int, int]) -> float:
        if (
            not USE_CLEARANCE_AWARE_COST_FIELD
            or CLEARANCE_COST_WEIGHT <= 0.0
            or SOFT_CLEARANCE_COST_RADIUS_M <= 1e-6
        ):
            return 0.0

        candidate_ned = grid_index_to_ned(index, grid_map)
        total_extra_cost = 0.0

        for obs in obstacles:
            obs_xy = [obs.center_ned[0], obs.center_ned[1]]
            dist_to_center = distance_xy(candidate_ned, obs_xy)
            hard_radius = obstacle_inflated_radius(obs) + grid_map.extra_grid_clearance_m
            soft_radius = hard_radius + SOFT_CLEARANCE_COST_RADIUS_M

            if dist_to_center <= hard_radius:
                # This should already be filtered by occupied cells, but keep the
                # cost function defensive.
                return float("inf")

            if dist_to_center < soft_radius:
                normalized_nearness = (soft_radius - dist_to_center) / SOFT_CLEARANCE_COST_RADIUS_M
                total_extra_cost += CLEARANCE_COST_WEIGHT * normalized_nearness

        return total_extra_cost

    open_heap = []
    heapq.heappush(open_heap, (heuristic(start_index), 0.0, start_index))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start_index: 0.0}
    closed = set()

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        if current == goal_index:
            return reconstruct_grid_path(came_from, current)

        closed.add(current)

        for dx, dy, move_cost in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)

            if not is_grid_index_inside(neighbor, grid_map):
                continue

            if is_cell_occupied(neighbor, grid_map):
                continue

            soft_cost = direct_path_soft_cost(neighbor) + clearance_aware_soft_cost(neighbor)
            tentative_g = current_g + move_cost * soft_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(open_heap, (f_score, tentative_g, neighbor))

    return None

def reconstruct_grid_path(
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    current: Tuple[int, int],
) -> List[Tuple[int, int]]:
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def run_astar_on_grid(
    start_ned: List[float],
    goal_ned: List[float],
    obstacles: List[ObstacleInfo],
    extra_grid_clearance_m: float = 0.0,
) -> Tuple[Optional[List[List[float]]], AStarGridMap]:
    grid_map = build_occupancy_grid(
        start_ned=start_ned,
        goal_ned=goal_ned,
        obstacles=obstacles,
        extra_grid_clearance_m=extra_grid_clearance_m,
    )

    start_index = ned_to_grid_index(start_ned, grid_map)
    goal_index = ned_to_grid_index(goal_ned, grid_map)

    print("")
    print("========== A* Occupancy Grid ==========")
    print(
        f"[AStar] grid bounds: "
        f"x=[{grid_map.min_x:.3f}, {grid_map.max_x:.3f}], "
        f"y=[{grid_map.min_y:.3f}, {grid_map.max_y:.3f}]"
    )
    print(f"[AStar] grid size: width={grid_map.width}, height={grid_map.height}")
    print(f"[AStar] grid resolution: {grid_map.resolution:.3f} m")
    occupied_ratio = grid_map.occupied_cell_count / max(1, grid_map.width * grid_map.height)
    print(f"[AStar] obstacle count: {grid_map.obstacle_count}")
    print(f"[AStar] occupied cell count: {grid_map.occupied_cell_count}")
    print(f"[AStar] occupied ratio: {occupied_ratio:.4f}")
    print(f"[AStar] direct path bias enabled: {USE_DIRECT_PATH_BIAS}")
    print(f"[AStar] direct path bias weight: {DIRECT_PATH_BIAS_WEIGHT:.3f}")
    print(f"[AStar] clearance-aware cost enabled: {USE_CLEARANCE_AWARE_COST_FIELD}")
    print(f"[AStar] clearance cost weight: {CLEARANCE_COST_WEIGHT:.3f}")
    print(f"[AStar] soft clearance cost radius: {SOFT_CLEARANCE_COST_RADIUS_M:.3f} m")
    print(f"[AStar] extra grid clearance: {extra_grid_clearance_m:.3f} m")
    print(f"[AStar] start_index={start_index}, goal_index={goal_index}")

    search_start_index = prepare_endpoint_index_for_astar(
        endpoint_name="start",
        endpoint_ned=start_ned,
        endpoint_index=start_index,
        grid_map=grid_map,
        obstacles=obstacles,
        extra_grid_clearance_m=extra_grid_clearance_m,
    )

    search_goal_index = prepare_endpoint_index_for_astar(
        endpoint_name="goal",
        endpoint_ned=goal_ned,
        endpoint_index=goal_index,
        grid_map=grid_map,
        obstacles=obstacles,
        extra_grid_clearance_m=extra_grid_clearance_m,
    )

    if search_start_index is None or search_goal_index is None:
        print("[AStar] ERROR: start/goal endpoint check failed.")
        print("=======================================")
        print("")
        return None, grid_map

    grid_path = astar_search_grid(
        grid_map=grid_map,
        start_index=search_start_index,
        goal_index=search_goal_index,
        start_ned=start_ned,
        goal_ned=goal_ned,
        obstacles=obstacles,
    )

    if grid_path is None:
        print("[AStar] ERROR: no path found on this occupancy grid.")
        print("=======================================")
        print("")
        return None, grid_map

    raw_path_ned = [grid_index_to_ned(index, grid_map) for index in grid_path]

    # Preserve the exact mission endpoints instead of slightly rounded grid centers.
    if distance_xy(raw_path_ned[0], start_ned) > grid_map.resolution * 0.5:
        raw_path_ned.insert(0, list(start_ned))
    else:
        raw_path_ned[0] = list(start_ned)

    if distance_xy(raw_path_ned[-1], goal_ned) > grid_map.resolution * 0.5:
        raw_path_ned.append(list(goal_ned))
    else:
        raw_path_ned[-1] = list(goal_ned)

    print(f"[AStar] raw path length: {len(raw_path_ned)} point(s)")
    print("=======================================")
    print("")

    return raw_path_ned, grid_map


# ============================================================
# PATH SIMPLIFICATION AND VALIDATION
# ============================================================

def interpolate_ned(a: List[float], b: List[float], t: float) -> List[float]:
    return [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]


def rdp_simplify_path(path: List[List[float]], tolerance_m: float) -> List[List[float]]:
    if len(path) <= 2:
        return list(path)

    start = path[0]
    end = path[-1]

    max_distance = -1.0
    max_index = -1

    for i in range(1, len(path) - 1):
        distance, _, _ = point_to_segment_distance_2d(
            [path[i][0], path[i][1]],
            [start[0], start[1]],
            [end[0], end[1]],
        )

        if distance > max_distance:
            max_distance = distance
            max_index = i

    if max_distance > tolerance_m and max_index > 0:
        left = rdp_simplify_path(path[: max_index + 1], tolerance_m)
        right = rdp_simplify_path(path[max_index:], tolerance_m)
        return left[:-1] + right

    return [start, end]


def densify_path_by_spacing(path: List[List[float]], max_spacing_m: float) -> List[List[float]]:
    if len(path) <= 1:
        return list(path)

    dense_path = [list(path[0])]

    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        segment_length = distance_xy(a, b)
        steps = max(1, int(math.ceil(segment_length / max_spacing_m)))

        for step in range(1, steps + 1):
            t = float(step) / float(steps)
            dense_path.append(interpolate_ned(a, b, t))

    return dense_path


def segment_clearance_to_obstacle(
    start_ned: List[float],
    end_ned: List[float],
    obstacle: ObstacleInfo,
    extra_clearance_m: float,
) -> Tuple[float, float, List[float]]:
    obstacle_xy = [obstacle.center_ned[0], obstacle.center_ned[1]]
    start_xy = [start_ned[0], start_ned[1]]
    end_xy = [end_ned[0], end_ned[1]]

    distance_to_center, t, closest = point_to_segment_distance_2d(
        obstacle_xy,
        start_xy,
        end_xy,
    )

    required_distance = obstacle_inflated_radius(obstacle) + extra_clearance_m
    clearance = distance_to_center - required_distance

    return clearance, t, closest


def is_segment_safe(
    start_ned: List[float],
    end_ned: List[float],
    obstacles: List[ObstacleInfo],
    extra_clearance_m: float = MIN_SEGMENT_CLEARANCE_M,
) -> Tuple[bool, str]:
    for obs in obstacles:
        clearance, t, _ = segment_clearance_to_obstacle(
            start_ned=start_ned,
            end_ned=end_ned,
            obstacle=obs,
            extra_clearance_m=extra_clearance_m,
        )

        if clearance < 0.0:
            message = (
                f"segment unsafe near obstacle={obs.name}, "
                f"clearance={clearance:.3f} m, t={t:.3f}, "
                f"start={format_vec(start_ned)}, end={format_vec(end_ned)}"
            )
            return False, message

    return True, "safe"


def validate_path_segments(
    path: List[List[float]],
    obstacles: List[ObstacleInfo],
    extra_clearance_m: float = MIN_SEGMENT_CLEARANCE_M,
) -> Tuple[bool, str]:
    if len(path) <= 1:
        return True, "path has one point"

    for i in range(len(path) - 1):
        safe, message = is_segment_safe(
            start_ned=path[i],
            end_ned=path[i + 1],
            obstacles=obstacles,
            extra_clearance_m=extra_clearance_m,
        )

        if not safe:
            return False, f"segment_index={i}: {message}"

    return True, "all segments safe"


def greedy_safe_simplify_path(
    raw_path: List[List[float]],
    obstacles: List[ObstacleInfo],
    max_spacing_m: float,
    extra_clearance_m: float = MIN_SEGMENT_CLEARANCE_M,
) -> List[List[float]]:
    if len(raw_path) <= 2:
        return list(raw_path)

    simplified = [list(raw_path[0])]
    current_index = 0

    while current_index < len(raw_path) - 1:
        best_index = current_index + 1

        for candidate_index in range(current_index + 1, len(raw_path)):
            candidate = raw_path[candidate_index]

            if distance_xy(raw_path[current_index], candidate) > max_spacing_m:
                break

            safe, _ = is_segment_safe(
                start_ned=raw_path[current_index],
                end_ned=candidate,
                obstacles=obstacles,
                extra_clearance_m=extra_clearance_m,
            )

            if safe:
                best_index = candidate_index

        simplified.append(list(raw_path[best_index]))
        current_index = best_index

    return simplified


def simplify_and_validate_path(
    raw_path_ned: List[List[float]],
    obstacles: List[ObstacleInfo],
) -> Tuple[List[List[float]], bool, str]:
    if len(raw_path_ned) <= 2:
        candidate = densify_path_by_spacing(raw_path_ned, MAX_WAYPOINT_SPACING_M)
        safe, message = validate_path_segments(
            candidate,
            obstacles,
            extra_clearance_m=MIN_SEGMENT_CLEARANCE_M,
        )
        return candidate, safe, message

    rdp_path = rdp_simplify_path(raw_path_ned, PATH_SIMPLIFY_TOLERANCE_M)
    rdp_path = densify_path_by_spacing(rdp_path, MAX_WAYPOINT_SPACING_M)

    safe, message = validate_path_segments(
        rdp_path,
        obstacles,
        extra_clearance_m=MIN_SEGMENT_CLEARANCE_M,
    )

    if safe:
        return rdp_path, True, "RDP simplified path is safe"

    print(f"[Planner] RDP simplified path is unsafe: {message}")
    print("[Planner] Falling back to greedy safe simplification.")

    greedy_path = greedy_safe_simplify_path(
        raw_path=raw_path_ned,
        obstacles=obstacles,
        max_spacing_m=MAX_WAYPOINT_SPACING_M,
        extra_clearance_m=MIN_SEGMENT_CLEARANCE_M,
    )

    greedy_path = densify_path_by_spacing(greedy_path, MAX_WAYPOINT_SPACING_M)

    safe, message = validate_path_segments(
        greedy_path,
        obstacles,
        extra_clearance_m=MIN_SEGMENT_CLEARANCE_M,
    )

    if safe:
        return greedy_path, True, "greedy simplified path is safe"

    print(f"[Planner] Greedy simplified path is still unsafe: {message}")
    print("[Planner] Falling back to raw A* path for validation.")

    raw_dense = densify_path_by_spacing(raw_path_ned, MAX_WAYPOINT_SPACING_M)
    safe, message = validate_path_segments(
        raw_dense,
        obstacles,
        extra_clearance_m=MIN_SEGMENT_CLEARANCE_M,
    )

    return raw_dense, safe, message


def make_waypoints_from_path(path_ned: List[List[float]]) -> List[Waypoint]:
    if len(path_ned) < 2:
        raise RuntimeError("A* path must contain at least start and goal points.")

    waypoints = []
    mission_points = path_ned[1:]

    for i, point in enumerate(mission_points):
        point = list(point)
        point[2] = -FLIGHT_ALTITUDE_M + PX4_NED_OFFSET_Z

        if i == len(mission_points) - 1:
            label = "final_goal"
        else:
            label = f"astar_wp_{i:03d}"

        waypoints.append(Waypoint(label=label, ned=point))

    return waypoints


def print_final_waypoints(waypoints: List[Waypoint]):
    print("")
    print("========== Final A* Waypoints ==========")
    print(f"[Planner] simplified waypoint count: {len(waypoints)}")

    for i, wp in enumerate(waypoints):
        print(f"[Planner] waypoint {i:03d}: label={wp.label}, ned={format_vec(wp.ned)}")

    print("========================================")
    print("")


def build_planner_summary(
    grid_map: AStarGridMap,
    raw_path_ned: List[List[float]],
    waypoints: List[Waypoint],
    path_is_safe: bool,
    final_path_total_length_m: float,
) -> PlannerSummary:
    occupied_ratio = grid_map.occupied_cell_count / max(1, grid_map.width * grid_map.height)

    return PlannerSummary(
        planner_type="astar_grid_lookahead" if USE_LOOKAHEAD_PATH_FOLLOWING else "astar_grid",
        grid_resolution_m=grid_map.resolution,
        grid_min_x=grid_map.min_x,
        grid_max_x=grid_map.max_x,
        grid_min_y=grid_map.min_y,
        grid_max_y=grid_map.max_y,
        grid_width=grid_map.width,
        grid_height=grid_map.height,
        obstacle_count=grid_map.obstacle_count,
        occupied_cell_count=grid_map.occupied_cell_count,
        occupied_ratio=occupied_ratio,
        astar_raw_path_point_count=len(raw_path_ned),
        astar_waypoint_count=len(waypoints),
        path_is_safe=bool(path_is_safe),
        extra_grid_clearance_m=grid_map.extra_grid_clearance_m,
        use_direct_path_bias=bool(USE_DIRECT_PATH_BIAS),
        direct_path_bias_weight=float(DIRECT_PATH_BIAS_WEIGHT),
        use_lookahead_path_following=bool(USE_LOOKAHEAD_PATH_FOLLOWING),
        path_lookahead_m=float(PATH_LOOKAHEAD_M),
        final_path_total_length_m=float(final_path_total_length_m),
    )

def plan_waypoints_once(
    start_ned: List[float],
    goal_ned: List[float],
    obstacles: List[ObstacleInfo],
    extra_grid_clearance_m: float,
) -> Optional[PlannerResult]:
    raw_path_ned, grid_map = run_astar_on_grid(
        start_ned=start_ned,
        goal_ned=goal_ned,
        obstacles=obstacles,
        extra_grid_clearance_m=extra_grid_clearance_m,
    )

    if raw_path_ned is None:
        return None

    simplified_path_ned, path_is_safe, safety_message = simplify_and_validate_path(
        raw_path_ned=raw_path_ned,
        obstacles=obstacles,
    )

    waypoints = make_waypoints_from_path(simplified_path_ned)

    _, final_path_total_length_m = compute_polyline_lengths(simplified_path_ned)

    print(f"[Planner] path safety result: {path_is_safe}. {safety_message}")
    print(f"[Planner] final path total length: {final_path_total_length_m:.3f} m")
    print(f"[Planner] direct path bias weight: {DIRECT_PATH_BIAS_WEIGHT:.3f}")
    print(f"[Planner] clearance-aware cost enabled: {USE_CLEARANCE_AWARE_COST_FIELD}")
    print(f"[Planner] clearance cost weight: {CLEARANCE_COST_WEIGHT:.3f}")
    print(f"[Planner] lookahead enabled: {USE_LOOKAHEAD_PATH_FOLLOWING}")
    print(f"[Planner] path lookahead distance: {PATH_LOOKAHEAD_M:.3f} m")
    print_final_waypoints(waypoints)

    summary = build_planner_summary(
        grid_map=grid_map,
        raw_path_ned=raw_path_ned,
        waypoints=waypoints,
        path_is_safe=path_is_safe,
        final_path_total_length_m=final_path_total_length_m,
    )

    return PlannerResult(
        waypoints=waypoints,
        raw_path_ned=raw_path_ned,
        simplified_path_ned=simplified_path_ned,
        summary=summary,
    )


def plan_waypoints(
    start_ned: List[float],
    goal_ned: List[float],
    obstacles: List[ObstacleInfo],
) -> PlannerResult:
    print("")
    print("========== Ground Truth A* Planner ==========")
    print(f"[Planner] start_ned={format_vec(start_ned)}")
    print(f"[Planner] goal_ned={format_vec(goal_ned)}")
    print(f"[Planner] obstacle_count={len(obstacles)}")
    print(f"[Planner] GRID_RESOLUTION_M={GRID_RESOLUTION_M:.3f}")
    print(f"[Planner] UAV_SAFETY_RADIUS_M={UAV_SAFETY_RADIUS_M:.3f}")
    print(f"[Planner] OBSTACLE_SAFETY_MARGIN_M={OBSTACLE_SAFETY_MARGIN_M:.3f}")
    print(f"[Planner] MIN_SEGMENT_CLEARANCE_M={MIN_SEGMENT_CLEARANCE_M:.3f}")
    print(f"[Planner] USE_DIRECT_PATH_BIAS={USE_DIRECT_PATH_BIAS}")
    print(f"[Planner] DIRECT_PATH_BIAS_WEIGHT={DIRECT_PATH_BIAS_WEIGHT:.3f}")
    print(f"[Planner] USE_LOOKAHEAD_PATH_FOLLOWING={USE_LOOKAHEAD_PATH_FOLLOWING}")
    print(f"[Planner] PATH_LOOKAHEAD_M={PATH_LOOKAHEAD_M:.3f}")
    print("=============================================")
    print("")

    first_result = plan_waypoints_once(
        start_ned=start_ned,
        goal_ned=goal_ned,
        obstacles=obstacles,
        extra_grid_clearance_m=0.0,
    )

    if first_result is not None and first_result.summary.path_is_safe:
        return first_result

    if REPLAN_WITH_SEGMENT_CLEARANCE_IF_NEEDED:
        print("[Planner] First A* result was missing or unsafe.")
        print(
            "[Planner] Replanning with extra grid clearance equal to "
            f"MIN_SEGMENT_CLEARANCE_M={MIN_SEGMENT_CLEARANCE_M:.3f} m."
        )

        second_result = plan_waypoints_once(
            start_ned=start_ned,
            goal_ned=goal_ned,
            obstacles=obstacles,
            extra_grid_clearance_m=MIN_SEGMENT_CLEARANCE_M,
        )

        if second_result is not None and second_result.summary.path_is_safe:
            return second_result

    start_clearance, start_nearest = point_clearance_to_inflated_obstacles(
        start_ned, obstacles, extra_clearance_m=MIN_SEGMENT_CLEARANCE_M
    )
    goal_clearance, goal_nearest = point_clearance_to_inflated_obstacles(
        goal_ned, obstacles, extra_clearance_m=MIN_SEGMENT_CLEARANCE_M
    )

    raise RuntimeError(
        "A* planner could not produce a safe path. "
        "The UAV will not start the mission. "
        f"start_clearance_with_segment_margin={start_clearance:.3f} m near {start_nearest}; "
        f"goal_clearance_with_segment_margin={goal_clearance:.3f} m near {goal_nearest}. "
        "If start/goal clearance is negative, regenerate the scene with a larger DISK_SAFE_MARGIN. "
        "Otherwise try increasing GRID_MARGIN_M, reducing GRID_RESOLUTION_M, or adjusting safety margins."
    )


# ============================================================
# ISAAC SIM SAFETY ENVELOPE VISUALIZATION
# ============================================================

def set_prim_display_opacity(prim, opacity: float):
    try:
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayOpacityAttr([float(opacity)])
    except Exception:
        pass


def set_debug_custom_attr(prim, name: str, value: Any):
    try:
        if isinstance(value, bool):
            value_type = Sdf.ValueTypeNames.Bool
        elif isinstance(value, int):
            value_type = Sdf.ValueTypeNames.Int
        elif isinstance(value, float):
            value_type = Sdf.ValueTypeNames.Double
        else:
            value_type = Sdf.ValueTypeNames.String
            value = str(value)

        prim.CreateAttribute(f"debug:{name}", value_type, custom=True).Set(value)
    except Exception:
        pass


def create_debug_cylinder_disk(
    stage: Usd.Stage,
    path: str,
    radius_m: float,
    height_m: float,
    center_isaac_xy: List[float],
    z_offset_m: float,
    color_rgb: Tuple[float, float, float],
    opacity: float,
    metadata: Dict[str, Any],
):
    """Create a thin cylinder disk used as a top-down 2D safety envelope marker."""
    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cylinder.CreateRadiusAttr(float(radius_m))
    cylinder.CreateHeightAttr(float(height_m))

    prim = cylinder.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    z = float(z_offset_m) + 0.5 * float(height_m)
    xformable.AddTranslateOp().Set(
        Gf.Vec3d(float(center_isaac_xy[0]), float(center_isaac_xy[1]), z)
    )

    set_prim_display_color(prim, color_rgb)
    set_prim_display_opacity(prim, opacity)

    for key, value in metadata.items():
        set_debug_custom_attr(prim, key, value)

    return cylinder


def create_debug_gap_line(
    stage: Usd.Stage,
    line_path: str,
    obstacle_a: ObstacleInfo,
    obstacle_b: ObstacleInfo,
):
    point_a = [
        obstacle_a.bbox_center_isaac[0],
        obstacle_a.bbox_center_isaac[1],
        SAFETY_ENVELOPE_Z_OFFSET_M + SAFETY_ENVELOPE_HEIGHT_M + 0.06,
    ]
    point_b = [
        obstacle_b.bbox_center_isaac[0],
        obstacle_b.bbox_center_isaac[1],
        SAFETY_ENVELOPE_Z_OFFSET_M + SAFETY_ENVELOPE_HEIGHT_M + 0.06,
    ]

    create_or_update_curve(
        stage=stage,
        curve_path=line_path,
        points=[
            Gf.Vec3f(float(point_a[0]), float(point_a[1]), float(point_a[2])),
            Gf.Vec3f(float(point_b[0]), float(point_b[1]), float(point_b[2])),
        ],
        width_m=TOO_NARROW_GAP_LINE_WIDTH_M,
        color_rgb=COLOR_TOO_NARROW_GAP_LINE,
    )


def draw_safety_envelope_visualization(
    obstacles: List[ObstacleInfo],
    gap_diagnostics: Optional[List[Dict[str, Any]]] = None,
):
    """Draw the planner's forbidden regions in Isaac Sim before PX4 connection.

    /World/GeneratedEpisode/DebugSafetyEnvelope
    - PlanningRadius: orange disks for A* occupancy inflation.
    - ValidationRadius: red disks for stricter shortcut / segment validation.
    - TooNarrowGapLines: red lines between obstacle pairs whose validation envelopes
      leave negative pass_margin.
    """
    if not DRAW_SAFETY_ENVELOPE:
        print("[SafetyEnvelope] Disabled.")
        return

    try:
        stage = get_stage()
        delete_prim_if_exists(stage, SAFETY_ENVELOPE_ROOT)
        ensure_xform(stage, SAFETY_ENVELOPE_ROOT)

        planning_root = f"{SAFETY_ENVELOPE_ROOT}/PlanningRadius"
        validation_root = f"{SAFETY_ENVELOPE_ROOT}/ValidationRadius"
        gap_root = f"{SAFETY_ENVELOPE_ROOT}/TooNarrowGapLines"

        ensure_xform(stage, planning_root)
        ensure_xform(stage, validation_root)
        ensure_xform(stage, gap_root)

        gap_diagnostics = gap_diagnostics or compute_obstacle_gap_diagnostics(obstacles)
        too_narrow_names = set()

        for item in gap_diagnostics:
            if item.get("status") == "TOO_NARROW":
                too_narrow_names.add(item["name_a"])
                too_narrow_names.add(item["name_b"])

        print("")
        print("========== DebugSafetyEnvelope Visualization ==========")
        print(f"[SafetyEnvelope] root={SAFETY_ENVELOPE_ROOT}")
        print(f"[SafetyEnvelope] DRAW_PLANNING_RADIUS={DRAW_PLANNING_RADIUS}")
        print(f"[SafetyEnvelope] DRAW_VALIDATION_RADIUS={DRAW_VALIDATION_RADIUS}")
        print("[SafetyEnvelope] orange = A* planning inflated radius")
        print("[SafetyEnvelope] red = segment-validation forbidden radius")
        print("[SafetyEnvelope] darker red / red line = at least one TOO_NARROW pair")
        print(
            "[SafetyEnvelope] formula: planning_radius = obstacle_radius "
            "+ UAV_SAFETY_RADIUS_M + OBSTACLE_SAFETY_MARGIN_M"
        )
        print("[SafetyEnvelope] formula: validation_radius = planning_radius + MIN_SEGMENT_CLEARANCE_M")

        for index, obs in enumerate(obstacles):
            planning_radius = obstacle_planning_radius(obs)
            validation_radius = obstacle_validation_radius(obs)
            center_xy = [obs.bbox_center_isaac[0], obs.bbox_center_isaac[1]]
            is_too_narrow_related = obs.name in too_narrow_names

            if DRAW_VALIDATION_RADIUS:
                validation_color = (
                    COLOR_SAFETY_TOO_NARROW_RADIUS
                    if is_too_narrow_related
                    else COLOR_SAFETY_VALIDATION_RADIUS
                )
                validation_opacity = (
                    SAFETY_ENVELOPE_TOO_NARROW_OPACITY
                    if is_too_narrow_related
                    else SAFETY_ENVELOPE_OPACITY
                )

                create_debug_cylinder_disk(
                    stage=stage,
                    path=f"{validation_root}/{obs.name}_ValidationRadius",
                    radius_m=validation_radius,
                    height_m=SAFETY_ENVELOPE_HEIGHT_M,
                    center_isaac_xy=center_xy,
                    z_offset_m=SAFETY_ENVELOPE_Z_OFFSET_M,
                    color_rgb=validation_color,
                    opacity=validation_opacity,
                    metadata={
                        "object_type": "safety_envelope_validation_radius",
                        "obstacle_name": obs.name,
                        "source_obstacle_path": obs.prim_path,
                        "obstacle_radius_m": float(obs.estimated_radius_m),
                        "planning_radius_m": float(planning_radius),
                        "validation_radius_m": float(validation_radius),
                        "too_narrow_related": bool(is_too_narrow_related),
                    },
                )

            if DRAW_PLANNING_RADIUS:
                create_debug_cylinder_disk(
                    stage=stage,
                    path=f"{planning_root}/{obs.name}_PlanningRadius",
                    radius_m=planning_radius,
                    height_m=SAFETY_ENVELOPE_HEIGHT_M,
                    center_isaac_xy=center_xy,
                    z_offset_m=SAFETY_ENVELOPE_Z_OFFSET_M + SAFETY_ENVELOPE_HEIGHT_M + 0.01,
                    color_rgb=COLOR_SAFETY_PLANNING_RADIUS,
                    opacity=SAFETY_ENVELOPE_OPACITY,
                    metadata={
                        "object_type": "safety_envelope_planning_radius",
                        "obstacle_name": obs.name,
                        "source_obstacle_path": obs.prim_path,
                        "obstacle_radius_m": float(obs.estimated_radius_m),
                        "planning_radius_m": float(planning_radius),
                        "validation_radius_m": float(validation_radius),
                    },
                )

            print(
                f"[SafetyEnvelope] {obs.name}: "
                f"obstacle_radius={obs.estimated_radius_m:.3f} m, "
                f"planning_radius={planning_radius:.3f} m, "
                f"validation_radius={validation_radius:.3f} m"
            )

        if DRAW_TOO_NARROW_GAP_LINES:
            line_count = 0
            for item in gap_diagnostics:
                if item.get("status") != "TOO_NARROW":
                    continue

                obstacle_a = item["obstacle_a"]
                obstacle_b = item["obstacle_b"]
                create_debug_gap_line(
                    stage=stage,
                    line_path=f"{gap_root}/{obstacle_a.name}_to_{obstacle_b.name}",
                    obstacle_a=obstacle_a,
                    obstacle_b=obstacle_b,
                )
                line_count += 1

            print(f"[SafetyEnvelope] too-narrow gap line count={line_count}")

        print("=======================================================")
        print("")

    except Exception as exc:
        print(f"[SafetyEnvelope] Warning: could not draw safety envelope: {exc}")


# ============================================================
# ISAAC SIM PATH VISUALIZATION
# ============================================================

def delete_prim_if_exists(stage: Usd.Stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))


def ensure_xform(stage: Usd.Stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        return UsdGeom.Xform(prim)
    return UsdGeom.Xform.Define(stage, Sdf.Path(prim_path))


def set_prim_display_color(prim, color_rgb: Tuple[float, float, float]):
    try:
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr([
            Gf.Vec3f(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))
        ])
    except Exception:
        pass


def set_astar_path_visibility(visible: bool = True) -> bool:
    """Toggle the retained A* debug prim without deleting or rebuilding it."""
    stage = get_stage()
    prim = stage.GetPrimAtPath(PATH_VIS_ROOT)
    if not prim or not prim.IsValid():
        print(f"[Visualizer] Path root does not exist yet: {PATH_VIS_ROOT}")
        return False

    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()
    print(
        f"[Visualizer] A* path visibility={'visible' if visible else 'hidden'} "
        f"at {PATH_VIS_ROOT}"
    )
    return True


def ned_path_to_isaac_points(path_ned: List[List[float]], z_offset_m: float = 0.0) -> List[Gf.Vec3f]:
    points = []
    for point_ned in path_ned:
        isaac_point = ned_to_isaac_position(point_ned)
        isaac_point[2] += z_offset_m
        points.append(Gf.Vec3f(float(isaac_point[0]), float(isaac_point[1]), float(isaac_point[2])))
    return points


def create_or_update_curve(
    stage: Usd.Stage,
    curve_path: str,
    points: List[Gf.Vec3f],
    width_m: float,
    color_rgb: Tuple[float, float, float],
):
    if len(points) < 2:
        return None

    curve = UsdGeom.BasisCurves.Define(stage, Sdf.Path(curve_path))
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr(points)
    curve.CreateWidthsAttr([float(width_m)])
    set_prim_display_color(curve.GetPrim(), color_rgb)
    return curve


def create_path_marker_sphere(
    stage: Usd.Stage,
    sphere_path: str,
    point_ned: List[float],
    radius_m: float,
    color_rgb: Tuple[float, float, float],
):
    point_isaac = ned_to_isaac_position(point_ned)
    point_isaac[2] += PATH_VISUAL_Z_OFFSET_M

    sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(sphere_path))
    sphere.CreateRadiusAttr(float(radius_m))
    sphere.GetPrim().CreateAttribute("debug:object_type", Sdf.ValueTypeNames.String, custom=True).Set("path_marker")
    xformable = UsdGeom.Xformable(sphere.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(float(point_isaac[0]), float(point_isaac[1]), float(point_isaac[2])))
    set_prim_display_color(sphere.GetPrim(), color_rgb)
    return sphere


def draw_planned_path_visualization(
    raw_path_ned: List[List[float]],
    simplified_path_ned: List[List[float]],
    waypoints: List[Waypoint],
):
    """Draw the planned path in Isaac Sim as USD debug curves.

    Cyan curve: final validated path used by the lookahead follower.
    Orange curve: raw A* path, if DRAW_RAW_ASTAR_PATH=True.
    Green spheres: final waypoint markers, if DRAW_WAYPOINT_MARKERS=True.
    """
    if not DRAW_PATH_VISUALIZATION_IN_STAGE:
        return

    try:
        stage = get_stage()
        delete_prim_if_exists(stage, PATH_VIS_ROOT)
        ensure_xform(stage, PATH_VIS_ROOT)

        if DRAW_RAW_ASTAR_PATH and len(raw_path_ned) >= 2:
            raw_points = ned_path_to_isaac_points(raw_path_ned, PATH_VISUAL_Z_OFFSET_M + 0.03)
            create_or_update_curve(
                stage=stage,
                curve_path=f"{PATH_VIS_ROOT}/RawAStarPath",
                points=raw_points,
                width_m=RAW_PATH_LINE_WIDTH_M,
                color_rgb=COLOR_RAW_PATH,
            )

        if DRAW_SIMPLIFIED_PATH and len(simplified_path_ned) >= 2:
            planned_points = ned_path_to_isaac_points(simplified_path_ned, PATH_VISUAL_Z_OFFSET_M)
            create_or_update_curve(
                stage=stage,
                curve_path=f"{PATH_VIS_ROOT}/PlannedLookaheadPath",
                points=planned_points,
                width_m=PATH_LINE_WIDTH_M,
                color_rgb=COLOR_PLANNED_PATH,
            )

        if DRAW_WAYPOINT_MARKERS:
            ensure_xform(stage, f"{PATH_VIS_ROOT}/WaypointMarkers")
            for i, waypoint in enumerate(waypoints):
                create_path_marker_sphere(
                    stage=stage,
                    sphere_path=f"{PATH_VIS_ROOT}/WaypointMarkers/WP_{i:03d}",
                    point_ned=waypoint.ned,
                    radius_m=WAYPOINT_MARKER_RADIUS_M,
                    color_rgb=COLOR_WAYPOINT_MARKER,
                )

        # Visibility is inherited by every curve and waypoint marker below the
        # root, so both FPV and TOP cameras omit it while the data remains in USD.
        builtins.set_astar_path_visibility = set_astar_path_visibility
        set_astar_path_visibility(SHOW_PATH_VISUALIZATION)

        print(f"[Visualizer] Planned flight path drawn in Isaac Sim at: {PATH_VIS_ROOT}")
        print("[Visualizer] Cyan line = validated lookahead path. Green spheres = waypoint markers.")
        print("[Visualizer] Live red executed trail is disabled in v4.3 for PX4/Pegasus stability.")
        print("[Visualizer] Use draw_px4_mission_trail_from_csv.py after flight if you want the red actual trail.")
        print("[Visualizer] Toggle later with: builtins.set_astar_path_visibility(True/False)")

    except Exception as exc:
        print(f"[Visualizer] Warning: could not draw planned path visualization: {exc}")


def obstacle_can_be_overflown(obstacle: ObstacleInfo) -> bool:
    if not ENABLE_OVERFLY_SHORT_OBSTACLES:
        return False

    obstacle_top_isaac_z = float(obstacle.bbox_center_isaac[2]) + 0.5 * float(obstacle.bbox_size_isaac[2])
    flight_height_isaac_z = float(FLIGHT_ALTITUDE_M)
    return obstacle_top_isaac_z + OVERFLY_VERTICAL_CLEARANCE_M <= flight_height_isaac_z


def filter_obstacles_for_2p5d_planning(obstacles: List[ObstacleInfo]) -> List[ObstacleInfo]:
    if not ENABLE_OVERFLY_SHORT_OBSTACLES:
        print("[2.5D Planner] Overfly-short-obstacles disabled. All obstacles are avoided in XY.")
        return list(obstacles)

    planning_obstacles = []
    ignored_obstacles = []

    for obs in obstacles:
        obstacle_top_isaac_z = float(obs.bbox_center_isaac[2]) + 0.5 * float(obs.bbox_size_isaac[2])
        if obstacle_can_be_overflown(obs):
            ignored_obstacles.append((obs, obstacle_top_isaac_z))
        else:
            planning_obstacles.append(obs)

    print("")
    print("========== 2.5D Overfly Filter ==========")
    print(f"[2.5D Planner] ENABLE_OVERFLY_SHORT_OBSTACLES={ENABLE_OVERFLY_SHORT_OBSTACLES}")
    print(f"[2.5D Planner] FLIGHT_ALTITUDE_M={FLIGHT_ALTITUDE_M:.3f} m")
    print(f"[2.5D Planner] OVERFLY_VERTICAL_CLEARANCE_M={OVERFLY_VERTICAL_CLEARANCE_M:.3f} m")
    print(f"[2.5D Planner] total_obstacles={len(obstacles)}, avoided_in_xy={len(planning_obstacles)}, overflown={len(ignored_obstacles)}")

    for obs, top_z in ignored_obstacles:
        print(
            f"[2.5D Planner] overfly {obs.name}: "
            f"top_z={top_z:.3f} m, required_top<={FLIGHT_ALTITUDE_M - OVERFLY_VERTICAL_CLEARANCE_M:.3f} m"
        )

    print("=========================================")
    print("")
    return planning_obstacles


# ============================================================
# CSV LOGGER
# ============================================================

class CsvMissionLogger:
    def __init__(self, episode_id: str, planner_summary: PlannerSummary):
        os.makedirs(LOG_DIR, exist_ok=True)

        self.episode_id = episode_id
        self.planner_summary = planner_summary
        self.csv_path = os.path.join(LOG_DIR, f"mission_log_{episode_id}.csv")
        self.file = open(self.csv_path, "w", newline="")

        self.fieldnames = [
            "episode_id",
            "time_wall",
            "mission_time",
            "sample_index",
            "phase",
            "waypoint_index",
            "waypoint_label",

            "pos_x_ned",
            "pos_y_ned",
            "pos_z_ned",

            "vel_x_ned",
            "vel_y_ned",
            "vel_z_ned",

            "roll",
            "pitch",
            "yaw",

            "final_goal_x_ned",
            "final_goal_y_ned",
            "final_goal_z_ned",

            "active_target_x_ned",
            "active_target_y_ned",
            "active_target_z_ned",

            "cmd_vx_ned",
            "cmd_vy_ned",
            "cmd_vz_ned",
            "cmd_yaw_rate",

            "obstacle_count",
            "nearest_obstacle_name",
            "nearest_obstacle_x_ned",
            "nearest_obstacle_y_ned",
            "nearest_obstacle_radius_m",
            "distance_nearest_obstacle_xy",

            "red_point_x_isaac",
            "red_point_y_isaac",
            "red_point_z_isaac",
            "red_point_x_ned",
            "red_point_y_ned",
            "red_point_z_ned",

            "distance_final_goal_xy",
            "min_distance_to_obstacle_so_far",
            "path_length_so_far",
            "success_auto",

            "planner_type",
            "grid_resolution_m",
            "astar_raw_path_point_count",
            "astar_waypoint_count",
            "path_is_safe",
            "grid_min_x",
            "grid_max_x",
            "grid_min_y",
            "grid_max_y",
            "grid_width",
            "grid_height",
            "occupied_cell_count",
            "extra_grid_clearance_m",

            "occupied_ratio",
            "use_direct_path_bias",
            "direct_path_bias_weight",
            "use_lookahead_path_following",
            "lookahead_target_x_ned",
            "lookahead_target_y_ned",
            "lookahead_target_z_ned",
            "lookahead_distance_m",
            "path_following_mode",
            "active_path_segment_index",
            "path_progress_m",
            "remaining_path_length_m",
            "final_path_total_length_m",
        ]

        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()

        print(f"[Logger] Mission CSV created: {self.csv_path}")

    def write_row(self, row: Dict[str, Any]):
        self.writer.writerow(row)

    def flush(self):
        self.file.flush()

    def close(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass


# ============================================================
# PX4 RUNNER
# ============================================================

class Px4EpisodeRunner:
    def __init__(
        self,
        episode_id: str,
        obstacles: List[ObstacleInfo],
        red_point: Optional[RedPointInfo],
        waypoints: List[Waypoint],
        start_ned: List[float],
        final_goal_ned: List[float],
        path_points_ned: List[List[float]],
        planner_summary: PlannerSummary,
    ):
        self.episode_id = episode_id
        self.obstacles = obstacles
        self.red_point = red_point
        self.waypoints = waypoints
        self.start_ned = start_ned
        self.final_goal_ned = final_goal_ned
        self.path_points_ned = [list(point) for point in path_points_ned]
        self.path_cumulative_lengths, self.path_total_length_m = compute_polyline_lengths(self.path_points_ned)
        self.planner_summary = planner_summary

        self.master = None
        self.state = VehicleState()

        # PX4 status monitored from HEARTBEAT / COMMAND_ACK.
        self.px4_armed = False
        self.px4_base_mode = 0
        self.px4_custom_mode = 0
        self.px4_flightmode = "UNKNOWN"
        self.command_ack_results: Dict[int, int] = {}

        self.thread = None
        self.stop_requested = False
        self.land_requested = False
        self.land_sent = False
        self.land_command_time = None
        self.auto_land_start_time = None

        self.phase = "created"
        self.active_waypoint_index = -1
        self.active_waypoint_label = ""
        self.active_target_ned = list(start_ned)

        self.lookahead_target_ned = list(start_ned)
        self.lookahead_distance_m = 0.0
        self.path_following_mode = "waypoint"
        self.active_path_segment_index = 0
        self.path_progress_m = 0.0
        self.remaining_path_length_m = self.path_total_length_m
        self.last_status_print_wall = 0.0

        self.sample_index = 0
        self.success_auto = False

        self.path_length_so_far = 0.0
        self.min_distance_to_obstacle_so_far = float("inf")
        self.previous_position_xy = None

        self.mission_start_wall = None
        self.boot_wall = time.time()

        self.last_cmd = (0.0, 0.0, 0.0, 0.0)
        self.smoothed_cmd = [0.0, 0.0, 0.0, 0.0]
        self.last_command_smooth_time = time.time()

        self.executed_trail_points_isaac: List[Gf.Vec3f] = []
        self.executed_trail_last_ned: Optional[List[float]] = None

        self.logger = CsvMissionLogger(episode_id, planner_summary)

        self.keyboard = None
        self.keyboard_input = None
        self.keyboard_sub = None

    # ------------------------------
    # Public controls
    # ------------------------------

    def start(self):
        self.install_keyboard_handler()

        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

        print("[Runner] Mission thread started.")
        print("[Runner] Press H for help, P for status, L for emergency land.")

    def request_land(self):
        print("[Keyboard] Land requested.")
        self.land_requested = True

    def request_stop(self, send_land: bool = False):
        if send_land:
            self.request_land()
        else:
            print("[Runner] Stop requested without landing.")
            self.stop_requested = True

    def print_help(self):
        print("")
        print("========== Keyboard Help ==========")
        print("H: Print this help")
        print("P: Print current status")
        print("L: Emergency land and stop")
        print("R: TODO reset and start a new episode safely")
        print("===================================")
        print("")

    def print_status(self):
        print("")
        print("========== Current Status ==========")
        print(f"episode_id: {self.episode_id}")
        print(f"phase: {self.phase}")
        print(f"waypoint_index: {self.active_waypoint_index}")
        print(f"waypoint_label: {self.active_waypoint_label}")
        print(f"position_ned: {format_vec([self.state.pos_x_ned, self.state.pos_y_ned, self.state.pos_z_ned])}")
        print(f"velocity_ned: {format_vec([self.state.vel_x_ned, self.state.vel_y_ned, self.state.vel_z_ned])}")
        print(f"active_target_ned: {format_vec(self.active_target_ned)}")
        print(f"lookahead_target_ned: {format_vec(self.lookahead_target_ned)}")
        print(f"path_following_mode: {self.path_following_mode}")
        print(f"active_path_segment_index: {self.active_path_segment_index}")
        print(f"path_progress_m: {self.path_progress_m:.3f}")
        print(f"remaining_path_length_m: {self.remaining_path_length_m:.3f}")
        print(f"final_goal_ned: {format_vec(self.final_goal_ned)}")
        print(f"distance_final_goal_xy: {self.compute_distance_to_final_goal_xy():.3f}")
        print(f"path_length_so_far: {self.path_length_so_far:.3f}")
        print(f"min_distance_to_obstacle_so_far: {self.min_distance_to_obstacle_so_far:.3f}")
        print(f"success_auto: {self.success_auto}")
        print(f"planner_type: {self.planner_summary.planner_type}")
        print(f"astar_raw_path_point_count: {self.planner_summary.astar_raw_path_point_count}")
        print(f"astar_waypoint_count: {self.planner_summary.astar_waypoint_count}")
        print(f"path_is_safe: {self.planner_summary.path_is_safe}")
        print(f"direct_path_bias_weight: {self.planner_summary.direct_path_bias_weight:.3f}")
        print(f"lookahead_enabled: {self.planner_summary.use_lookahead_path_following}")
        print(f"final_path_total_length_m: {self.path_total_length_m:.3f}")
        print("====================================")
        print("")

    # ------------------------------
    # Keyboard
    # ------------------------------

    def install_keyboard_handler(self):
        if carb is None:
            print("[Keyboard] carb input is not available. Keyboard control disabled.")
            return

        try:
            app_window = omni.appwindow.get_default_app_window()

            if app_window is None:
                print("[Keyboard] No default app window found. Keyboard control disabled.")
                return

            self.keyboard = app_window.get_keyboard()
            self.keyboard_input = carb.input.acquire_input_interface()
            self.keyboard_sub = self.keyboard_input.subscribe_to_keyboard_events(
                self.keyboard,
                self.on_keyboard_event,
            )

            print("[Keyboard] Keyboard handler installed.")

        except Exception as exc:
            print(f"[Keyboard] Failed to install keyboard handler: {exc}")

    def on_keyboard_event(self, event):
        event_type = str(event.type).upper()

        if "PRESS" not in event_type:
            return True

        key_text = str(event.input).upper()

        if key_matches(key_text, ["H", "KEY_H"]):
            self.print_help()
        elif key_matches(key_text, ["P", "KEY_P"]):
            self.print_status()
        elif key_matches(key_text, ["L", "KEY_L"]):
            self.request_land()
        elif key_matches(key_text, ["R", "KEY_R"]):
            print("[Keyboard] R pressed. TODO: reset episode safely.")

        return True

    # ------------------------------
    # PX4 helpers
    # ------------------------------
    def save_ros2_waypoints_json(
        output_path: str,
        episode_id: str,
        start_ned: List[float],
        final_goal_ned: List[float],
        planning_result: PlannerResult,
    ) -> str:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Include the takeoff/starting altitude point first.
        # planning_result.waypoints usually excludes the exact start point.
        waypoints = [list(start_ned)]

        for waypoint in planning_result.waypoints:
            waypoints.append([
                float(waypoint.ned[0]),
                float(waypoint.ned[1]),
                float(waypoint.ned[2]),
            ])

        payload = {
            "episode_id": episode_id,
            "frame": "PX4_NED",
            "source": "isaac_astar_exporter",
            "flight_altitude_m": float(FLIGHT_ALTITUDE_M),
            "start_ned": [float(v) for v in start_ned],
            "final_goal_ned": [float(v) for v in final_goal_ned],
            "waypoint_count": len(waypoints),
            "waypoints": waypoints,
            "planner_summary": {
                "planner_type": planning_result.summary.planner_type,
                "grid_resolution_m": float(planning_result.summary.grid_resolution_m),
                "obstacle_count": int(planning_result.summary.obstacle_count),
                "astar_raw_path_point_count": int(planning_result.summary.astar_raw_path_point_count),
                "astar_waypoint_count": int(planning_result.summary.astar_waypoint_count),
                "path_is_safe": bool(planning_result.summary.path_is_safe),
                "final_path_total_length_m": float(planning_result.summary.final_path_total_length_m),
            },
        }

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

        print("")
        print("========== ROS2 Waypoints Export ==========")
        print(f"[Exporter] Saved ROS2 waypoint JSON: {output_path}")
        print(f"[Exporter] waypoint_count={len(waypoints)}")
        print(f"[Exporter] frame=PX4_NED")
        for index, waypoint in enumerate(waypoints):
            print(
                f"[Exporter] waypoint {index:03d}: "
                f"[{waypoint[0]:+.3f}, {waypoint[1]:+.3f}, {waypoint[2]:+.3f}]"
            )
        print("==========================================")
        print("")

        return output_path

    def target_system(self) -> int:
        if self.master is None:
            return 1

        system_id = int(getattr(self.master, "target_system", 1))

        if system_id <= 0:
            return 1

        return system_id

    def target_component(self) -> int:
        if self.master is None:
            return int(mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1)

        component_id = int(getattr(self.master, "target_component", 1))

        if component_id <= 0:
            return int(mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1)

        return component_id

    def update_heartbeat_state(self, msg):
        """Update local PX4 status from a HEARTBEAT message."""
        try:
            self.px4_base_mode = int(msg.base_mode)
            self.px4_custom_mode = int(msg.custom_mode)
            self.px4_armed = bool(
                self.px4_base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            self.px4_flightmode = str(getattr(self.master, "flightmode", "UNKNOWN"))
        except Exception:
            pass

    def mav_result_name(self, result_code: int) -> str:
        """Return a readable MAV_RESULT name without assuming every dialect has every enum.

        Some Isaac/PX4 pymavlink environments load the ardupilotmega dialect and do not
        expose MAV_RESULT_CANCELLED. Using getattr keeps ACK decoding from crashing.
        """
        result_code = int(result_code)
        names = {}

        for enum_name, label in [
            ("MAV_RESULT_ACCEPTED", "ACCEPTED"),
            ("MAV_RESULT_TEMPORARILY_REJECTED", "TEMPORARILY_REJECTED"),
            ("MAV_RESULT_DENIED", "DENIED"),
            ("MAV_RESULT_UNSUPPORTED", "UNSUPPORTED"),
            ("MAV_RESULT_FAILED", "FAILED"),
            ("MAV_RESULT_IN_PROGRESS", "IN_PROGRESS"),
            ("MAV_RESULT_CANCELLED", "CANCELLED"),
            ("MAV_RESULT_COMMAND_LONG_ONLY", "COMMAND_LONG_ONLY"),
            ("MAV_RESULT_COMMAND_INT_ONLY", "COMMAND_INT_ONLY"),
            ("MAV_RESULT_COMMAND_UNSUPPORTED_MAV_FRAME", "COMMAND_UNSUPPORTED_MAV_FRAME"),
        ]:
            enum_value = getattr(mavutil.mavlink, enum_name, None)
            if enum_value is not None:
                names[int(enum_value)] = label

        return names.get(result_code, f"UNKNOWN_{result_code}")

    def px4_custom_mode_is_offboard(self) -> bool:
        return int(self.px4_custom_mode) == int(PX4_CUSTOM_MODE_OFFBOARD)

    def wait_for_command_ack(self, command: int, timeout_s: float, label: str) -> Optional[int]:
        """Wait for COMMAND_ACK of a command. Returns MAV_RESULT code or None."""
        deadline = time.time() + timeout_s
        command = int(command)

        while time.time() < deadline:
            self.drain_telemetry()

            if command in self.command_ack_results:
                result = int(self.command_ack_results.pop(command))
                print(f"[PX4] ACK {label}: {self.mav_result_name(result)} ({result})")
                return result

            time.sleep(0.05)

        print(f"[PX4] Warning: no COMMAND_ACK received for {label} within {timeout_s:.1f}s.")
        return None

    def wait_until_armed(self, expected_armed: bool, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            self.drain_telemetry()

            if self.px4_armed == expected_armed:
                state_text = "armed" if expected_armed else "disarmed"
                print(f"[PX4] Vehicle is confirmed {state_text}.")
                return True

            time.sleep(0.05)

        print(
            f"[PX4] Warning: armed confirmation timeout. "
            f"expected={expected_armed}, current={self.px4_armed}, "
            f"flightmode={self.px4_flightmode}, base_mode={self.px4_base_mode}"
        )
        return False

    def wait_until_flightmode(self, expected_mode: str, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        expected_mode = expected_mode.upper()

        while time.time() < deadline:
            self.drain_telemetry()
            current_mode = str(self.px4_flightmode).upper()

            if expected_mode in current_mode:
                print(f"[PX4] Flight mode confirmed by pymavlink flightmode: {self.px4_flightmode}")
                return True

            if expected_mode == "OFFBOARD" and self.px4_custom_mode_is_offboard():
                print(
                    "[PX4] Flight mode confirmed by HEARTBEAT.custom_mode: "
                    f"OFFBOARD custom_mode={self.px4_custom_mode}"
                )
                self.px4_flightmode = "OFFBOARD"
                return True

            time.sleep(0.05)

        print(
            f"[PX4] Warning: flight mode confirmation timeout. "
            f"expected={expected_mode}, current={self.px4_flightmode}, "
            f"custom_mode={self.px4_custom_mode}, expected_custom_mode={PX4_CUSTOM_MODE_OFFBOARD}"
        )
        return False

    def send_command_long(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ):
        self.master.mav.command_long_send(
            self.target_system(),
            self.target_component(),
            int(command),
            0,
            float(param1),
            float(param2),
            float(param3),
            float(param4),
            float(param5),
            float(param6),
            float(param7),
        )

    # ------------------------------
    # Main mission
    # ------------------------------
    def start_front_camera_recording_if_available(self):
        start_recorder = getattr(builtins, "start_front_camera_png_recorder", None)

        if start_recorder is None:
            print("[VisionRecorder] Front camera recorder is not installed. Skip image recording.")
            return

        try:
            start_recorder(episode_id=self.episode_id)
            print(f"[VisionRecorder] Front camera recording started: episode_id={self.episode_id}")
        except Exception as exc:
            print(f"[VisionRecorder] Failed to start front camera recording: {exc}")

    def stop_front_camera_recording_if_available(self):
        stop_recorder = getattr(builtins, "stop_front_camera_png_recorder", None)

        if stop_recorder is None:
            return

        try:
            stop_recorder()
            print("[VisionRecorder] Front camera recording stopped.")
        except Exception as exc:
            print(f"[VisionRecorder] Failed to stop front camera recording: {exc}")

    def run(self):
        try:
            self.phase = "connect"
            self.connect_px4()

            self.phase = "request_streams"
            self.request_message_streams()

            self.phase = "wait_local_position"
            self.wait_for_local_position(timeout_s=10.0)

            self.phase = "warmup_setpoints"
            self.send_warmup_setpoints()

            self.phase = "set_offboard"
            self.set_offboard_mode()

            self.phase = "arm"
            self.arm_vehicle()

            print("[PX4] Sending post-arm climb setpoints ...")
            for _ in range(40):
                self.drain_telemetry()
                self.send_velocity_setpoint(0.0, 0.0, -0.4, 0.0)
                time.sleep(0.05)

            self.mission_start_wall = time.time()

            # Start FPV image recording only when the real flight mission starts.
            self.start_front_camera_recording_if_available()

            self.phase = "takeoff"
            self.run_control_loop()

        except Exception as exc:
            print(f"[Runner] ERROR: {exc}")

        finally:
            self.cleanup()

    def connect_px4(self):
        print(f"[PX4] Connecting: {PX4_CONNECTION_STRING}")
        print("[PX4] Planned path should already be visible in Isaac Sim.")
        print("[PX4] Waiting for PX4 heartbeat ...")
        print("[PX4] If this keeps waiting, start/restart Pegasus + PX4 now.")

        self.master = mavutil.mavlink_connection(
            PX4_CONNECTION_STRING,
            autoreconnect=True,
        )

        wait_start_wall = time.time()
        next_print_wall = wait_start_wall

        while not self.stop_requested:
            heartbeat = self.master.wait_heartbeat(timeout=2)

            if heartbeat is not None:
                self.update_heartbeat_state(heartbeat)
                print(
                    f"[PX4] Heartbeat received. "
                    f"system={self.master.target_system}, "
                    f"component={self.master.target_component}, "
                    f"armed={self.px4_armed}, "
                    f"flightmode={self.px4_flightmode}"
                )

                if int(self.master.target_component) <= 0:
                    print("[PX4] Heartbeat component is 0. Commands will target MAV_COMP_ID_AUTOPILOT1.")

                return

            elapsed = time.time() - wait_start_wall

            if time.time() >= next_print_wall:
                print(
                    f"[PX4] Still waiting for heartbeat on {PX4_CONNECTION_STRING}. "
                    f"elapsed={elapsed:.1f}s"
                )
                next_print_wall = time.time() + PX4_HEARTBEAT_RETRY_PRINT_INTERVAL_S

            if (not WAIT_FOREVER_FOR_PX4_HEARTBEAT) and elapsed >= PX4_HEARTBEAT_TIMEOUT_S:
                raise RuntimeError(
                    "Timeout while waiting for PX4 heartbeat. "
                    "The planned blue path was drawn, but PX4/Pegasus did not send MAVLink heartbeat. "
                    "Please start/restart PX4 and check UDP port 14550."
                )

        raise RuntimeError("Stop requested while waiting for PX4 heartbeat.")

    def request_message_streams(self):
        try:
            self.master.mav.request_data_stream_send(
                self.target_system(),
                self.target_component(),
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                int(CONTROL_RATE_HZ),
                1,
            )
        except Exception as exc:
            print(f"[PX4] Warning: request_data_stream failed: {exc}")

        self.request_message_interval("LOCAL_POSITION_NED", int(1e6 / CONTROL_RATE_HZ))
        self.request_message_interval("ATTITUDE", int(1e6 / CONTROL_RATE_HZ))

    def request_message_interval(self, message_name: str, interval_us: int):
        constant_name = f"MAVLINK_MSG_ID_{message_name}"
        msg_id = getattr(mavutil.mavlink, constant_name, None)

        if msg_id is None:
            return

        try:
            self.send_command_long(
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                float(msg_id),
                float(interval_us),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        except Exception as exc:
            print(f"[PX4] Warning: failed to set message interval for {message_name}: {exc}")

    def wait_for_local_position(self, timeout_s: float):
        print("[PX4] Waiting for LOCAL_POSITION_NED ...")

        deadline = time.time() + timeout_s

        while time.time() < deadline:
            self.drain_telemetry()

            if self.state.has_position:
                print(
                    f"[PX4] Local position received: "
                    f"{format_vec([self.state.pos_x_ned, self.state.pos_y_ned, self.state.pos_z_ned])}"
                )
                return

            time.sleep(0.05)

        raise RuntimeError("Timeout while waiting for LOCAL_POSITION_NED.")

    def send_warmup_setpoints(self):
        print("[PX4] Sending warm-up velocity setpoints ...")

        dt = 1.0 / WARMUP_SETPOINT_RATE_HZ

        for _ in range(WARMUP_SETPOINT_COUNT):
            self.drain_telemetry()
            self.send_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
            time.sleep(dt)

    def set_offboard_mode(self):
        print("[PX4] Setting OFFBOARD mode ...")

        mode_mapping = self.master.mode_mapping()

        if mode_mapping is None or "OFFBOARD" not in mode_mapping:
            raise RuntimeError("PX4 mode mapping does not contain OFFBOARD.")

        offboard_mode = mode_mapping["OFFBOARD"]

        print(f"[PX4] OFFBOARD raw mode value: {offboard_mode}")

        if isinstance(offboard_mode, tuple):
            # For PX4, pymavlink may return:
            # (base_mode, custom_main_mode, custom_sub_mode)
            base_mode = int(offboard_mode[0])
            custom_main_mode = int(offboard_mode[1]) if len(offboard_mode) > 1 else 6
            custom_sub_mode = int(offboard_mode[2]) if len(offboard_mode) > 2 else 0

            # PX4 custom_mode encoding:
            # main mode is stored at bits 16-23
            # sub mode is stored at bits 24-31
            custom_mode = (custom_main_mode << 16) | (custom_sub_mode << 24)

            self.master.mav.set_mode_send(
                self.target_system(),
                base_mode,
                custom_mode,
            )

            print(
                f"[PX4] OFFBOARD command sent by SET_MODE tuple. "
                f"base_mode={base_mode}, "
                f"custom_main_mode={custom_main_mode}, "
                f"custom_sub_mode={custom_sub_mode}, "
                f"custom_mode={custom_mode}"
            )

        else:
            mode_id = int(offboard_mode)

            self.master.mav.set_mode_send(
                self.target_system(),
                int(mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
                mode_id,
            )

            print(f"[PX4] OFFBOARD command sent by SET_MODE. mode_id={mode_id}")

        # SET_MODE usually does not return COMMAND_ACK. Confirm by HEARTBEAT when possible.
        self.wait_until_flightmode("OFFBOARD", PX4_MODE_CONFIRM_TIMEOUT_S)
        time.sleep(0.2)

    def arm_vehicle(self):
        print("[PX4] Arming vehicle ...")

        arm_command = mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM
        self.send_command_long(
            arm_command,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        result = self.wait_for_command_ack(
            command=arm_command,
            timeout_s=PX4_COMMAND_ACK_TIMEOUT_S,
            label="ARM",
        )

        if result is not None and result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            raise RuntimeError(
                "PX4 rejected ARM command. "
                f"result={self.mav_result_name(result)} ({result}). "
                "Check PX4 console for preflight/failsafe messages."
            )

        confirmed = self.wait_until_armed(True, PX4_ARM_CONFIRM_TIMEOUT_S)

        if REQUIRE_ARM_CONFIRMATION and not confirmed:
            raise RuntimeError(
                "Arm command was sent, but PX4 heartbeat never confirmed ARMED state. "
                "The UAV will not take off. Check PX4 console, safety/preflight checks, and Pegasus connection."
            )

        print("[PX4] Arm command accepted and vehicle armed.")

    def send_land_command(self):
        print("[PX4] Sending LAND command ...")

        self.send_command_long(
            mavutil.mavlink.MAV_CMD_NAV_LAND,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def run_control_loop(self):
        control_dt = 1.0 / CONTROL_RATE_HZ
        last_log_time = 0.0

        takeoff_target = list(self.start_ned)
        takeoff_target[2] = -FLIGHT_ALTITUDE_M + PX4_NED_OFFSET_Z

        takeoff_deadline = time.time() + TAKEOFF_TIMEOUT_S
        mission_deadline = time.time() + MISSION_TIMEOUT_S

        self.active_waypoint_index = -1
        self.active_waypoint_label = "takeoff"
        self.active_target_ned = list(takeoff_target)

        print(f"[Mission] Takeoff target NED: {format_vec(takeoff_target)}")

        while not self.stop_requested:
            loop_start = time.time()

            self.drain_telemetry()

            if self.land_requested:
                self.phase = "landing"
                self.active_waypoint_label = "landing"

                if not self.land_sent:
                    self.send_land_command()
                    self.land_sent = True
                    self.land_command_time = time.time()

                self.send_velocity_setpoint(0.0, 0.0, 0.0, 0.0)

                if self.land_command_time is not None:
                    if time.time() - self.land_command_time >= LAND_STOP_DELAY_S:
                        self.stop_requested = True

                self.update_metrics()
                last_log_time = self.log_if_needed(last_log_time)
                self.sleep_to_rate(loop_start, control_dt)
                continue

            if not self.state.has_position:
                self.send_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
                self.sleep_to_rate(loop_start, control_dt)
                continue

            if self.phase == "takeoff":
                self.active_target_ned = list(takeoff_target)

                cmd = self.compute_velocity_command(
                    target_ned=takeoff_target,
                    max_speed_xy=MAX_SPEED_XY_MPS,
                    max_speed_z=MAX_SPEED_Z_MPS,
                )

                self.send_control_velocity_setpoint(*cmd)

                if self.is_target_reached(takeoff_target):
                    print("[Mission] Takeoff reached. Starting waypoint mission.")
                    self.phase = "mission"
                    self.active_waypoint_index = 0
                    self.active_waypoint_label = self.waypoints[0].label
                    self.active_target_ned = list(self.waypoints[0].ned)

                elif time.time() > takeoff_deadline:
                    current_altitude_m = -float(self.state.pos_z_ned)
                    message = (
                        "Takeoff timeout. The UAV did not climb enough, so the waypoint mission will NOT start. "
                        f"current_altitude={current_altitude_m:.3f} m, "
                        f"required_altitude>={MIN_TAKEOFF_ALTITUDE_REACHED_M:.3f} m, "
                        f"target_ned={format_vec(takeoff_target)}, "
                        f"current_ned={format_vec([self.state.pos_x_ned, self.state.pos_y_ned, self.state.pos_z_ned])}, "
                        f"armed={self.px4_armed}, flightmode={self.px4_flightmode}. "
                        "This usually means PX4 is not actually armed/OFFBOARD, or Pegasus is not applying MAVLink setpoints."
                    )

                    if REQUIRE_TAKEOFF_REACHED_BEFORE_MISSION:
                        raise RuntimeError(message)

                    print("[Mission] Warning: " + message)
                    print("[Mission] REQUIRE_TAKEOFF_REACHED_BEFORE_MISSION=False, starting waypoint mission anyway.")
                    self.phase = "mission"
                    self.active_waypoint_index = 0
                    self.active_waypoint_label = self.waypoints[0].label
                    self.active_target_ned = list(self.waypoints[0].ned)

            elif self.phase == "mission":
                if time.time() > mission_deadline:
                    print("[Mission] Mission timeout. Switching to hover.")
                    self.phase = "hover"
                    self.active_waypoint_label = "timeout_hover"
                    self.active_target_ned = [
                        self.state.pos_x_ned,
                        self.state.pos_y_ned,
                        -FLIGHT_ALTITUDE_M + PX4_NED_OFFSET_Z,
                    ]

                else:
                    if USE_LOOKAHEAD_PATH_FOLLOWING:
                        self.path_following_mode = "lookahead"
                        cmd, final_reached = self.compute_lookahead_velocity_command()
                        self.send_control_velocity_setpoint(*cmd)

                        if time.time() - self.last_status_print_wall >= STATUS_PRINT_INTERVAL_S:
                            self.print_runtime_status()
                            self.last_status_print_wall = time.time()

                        if final_reached:
                            print("[Mission] Final goal reached by lookahead follower. Preparing auto landing.")
                            self.phase = "auto_land_wait"
                            self.success_auto = True
                            self.active_waypoint_label = "final_auto_land_wait"
                            self.active_target_ned = list(self.final_goal_ned)
                            self.lookahead_target_ned = list(self.final_goal_ned)
                            self.auto_land_start_time = time.time()

                    else:
                        self.path_following_mode = "waypoint"
                        wp = self.waypoints[self.active_waypoint_index]
                        self.active_waypoint_label = wp.label
                        self.active_target_ned = list(wp.ned)
                        self.lookahead_target_ned = list(wp.ned)

                        cmd = self.compute_velocity_command(
                            target_ned=self.active_target_ned,
                            max_speed_xy=MAX_SPEED_XY_MPS,
                            max_speed_z=MAX_SPEED_Z_MPS,
                        )

                        self.send_control_velocity_setpoint(*cmd)

                        is_final = self.active_waypoint_index == len(self.waypoints) - 1

                        if self.is_target_reached(self.active_target_ned, is_final=is_final):
                            print(
                                f"[Mission] Waypoint reached: "
                                f"index={self.active_waypoint_index}, "
                                f"label={wp.label}"
                            )

                            if is_final:
                                print("[Mission] Final goal reached. Preparing auto landing.")
                                self.phase = "auto_land_wait"
                                self.success_auto = True
                                self.active_waypoint_label = "final_auto_land_wait"
                                self.active_target_ned = list(self.final_goal_ned)
                                self.auto_land_start_time = time.time()
                            else:
                                self.active_waypoint_index += 1

            elif self.phase == "auto_land_wait":
                cmd = self.compute_velocity_command(
                    target_ned=self.active_target_ned,
                    max_speed_xy=HOVER_MAX_SPEED_XY_MPS,
                    max_speed_z=HOVER_MAX_SPEED_Z_MPS,
                )

                self.send_control_velocity_setpoint(*cmd)
                
                if self.auto_land_start_time is None:
                    self.auto_land_start_time = time.time()

                if time.time() - self.auto_land_start_time >= AUTO_LAND_HOVER_SECONDS:
                    self.stop_front_camera_recording_if_available()
                    print("[Mission] Auto landing now.")
                    self.land_requested = True
            else:
                self.send_velocity_setpoint(0.0, 0.0, 0.0, 0.0)

            self.update_metrics()
            last_log_time = self.log_if_needed(last_log_time)
            self.sleep_to_rate(loop_start, control_dt)

    def sleep_to_rate(self, loop_start: float, control_dt: float):
        elapsed = time.time() - loop_start
        sleep_time = max(0.0, control_dt - elapsed)
        time.sleep(sleep_time)

    def compute_lookahead_velocity_command(self) -> Tuple[Tuple[float, float, float, float], bool]:
        current_ned = [
            self.state.pos_x_ned,
            self.state.pos_y_ned,
            self.state.pos_z_ned,
        ]

        self.path_progress_m, self.active_path_segment_index, closest_point, lateral_error = (
            find_closest_point_on_polyline(
                point_ned=current_ned,
                path=self.path_points_ned,
                cumulative_lengths=self.path_cumulative_lengths,
            )
        )

        self.remaining_path_length_m = max(0.0, self.path_total_length_m - self.path_progress_m)

        base_lookahead = clamp(PATH_LOOKAHEAD_M, MIN_LOOKAHEAD_M, MAX_LOOKAHEAD_M)
        speed_xy = norm2(self.state.vel_x_ned, self.state.vel_y_ned)
        dynamic_lookahead = clamp(max(base_lookahead, speed_xy * 1.1), MIN_LOOKAHEAD_M, MAX_LOOKAHEAD_M)

        candidate_lookaheads = [
            dynamic_lookahead,
            max(MIN_LOOKAHEAD_M, dynamic_lookahead * 0.75),
            max(MIN_LOOKAHEAD_M, dynamic_lookahead * 0.50),
            MIN_LOOKAHEAD_M,
        ]

        selected_target = None
        selected_distance = candidate_lookaheads[-1]
        selected_segment_index = self.active_path_segment_index

        for lookahead_distance in candidate_lookaheads:
            target_progress = min(
                self.path_total_length_m,
                self.path_progress_m + lookahead_distance,
            )

            candidate_target, candidate_segment_index = sample_polyline_at_progress(
                path=self.path_points_ned,
                cumulative_lengths=self.path_cumulative_lengths,
                target_progress_m=target_progress,
            )
            candidate_target[2] = -FLIGHT_ALTITUDE_M + PX4_NED_OFFSET_Z

            # The planned polyline is already validated. This additional check prevents
            # the carrot target from causing an unsafe shortcut from the current UAV pose.
            safe, _message = is_segment_safe(
                start_ned=current_ned,
                end_ned=candidate_target,
                obstacles=self.obstacles,
                extra_clearance_m=LOOKAHEAD_SEGMENT_CHECK_EXTRA_CLEARANCE_M,
            )

            if safe:
                selected_target = candidate_target
                selected_distance = lookahead_distance
                selected_segment_index = candidate_segment_index
                break

        if selected_target is None:
            selected_target = list(closest_point)
            selected_target[2] = -FLIGHT_ALTITUDE_M + PX4_NED_OFFSET_Z
            selected_distance = 0.0

        # When close to the final goal, stop chasing a carrot and aim directly at the goal.
        if self.remaining_path_length_m <= max(0.80, PATH_LOOKAHEAD_M):
            selected_target = list(self.final_goal_ned)
            selected_distance = self.remaining_path_length_m
            selected_segment_index = max(0, len(self.path_points_ned) - 2)

        self.lookahead_target_ned = list(selected_target)
        self.active_target_ned = list(selected_target)
        self.lookahead_distance_m = float(selected_distance)
        self.active_path_segment_index = int(selected_segment_index)
        self.active_waypoint_index = int(selected_segment_index)
        self.active_waypoint_label = f"lookahead_seg_{self.active_path_segment_index:03d}"

        max_speed_xy = self.compute_dynamic_max_speed_xy()

        cmd = self.compute_velocity_command(
            target_ned=self.active_target_ned,
            max_speed_xy=max_speed_xy,
            max_speed_z=MAX_SPEED_Z_MPS,
        )

        final_reached = self.is_target_reached(self.final_goal_ned, is_final=True)
        return cmd, final_reached

    def compute_dynamic_max_speed_xy(self) -> float:
        final_distance = self.compute_distance_to_final_goal_xy()
        final_factor = 1.0

        if not math.isnan(final_distance):
            final_factor = clamp(final_distance / 1.50, 0.45, 1.0)

        clearance, _nearest_name = point_clearance_to_inflated_obstacles(
            point_ned=[self.state.pos_x_ned, self.state.pos_y_ned, self.state.pos_z_ned],
            obstacles=self.obstacles,
            extra_clearance_m=0.0,
        )

        if math.isinf(clearance):
            clearance_factor = 1.0
        else:
            clearance_factor = clamp(clearance / 0.80, 0.50, 1.0)

        turn_angle = compute_turn_angle_at_segment(
            path=self.path_points_ned,
            segment_index=self.active_path_segment_index,
        )
        turn_factor = clamp(1.0 - (turn_angle / math.pi) * 0.55, 0.55, 1.0)

        dynamic_max_speed = MAX_SPEED_XY_MPS * min(final_factor, clearance_factor, turn_factor)
        return clamp(dynamic_max_speed, MIN_SPEED_XY_MPS, MAX_SPEED_XY_MPS)

    def print_runtime_status(self):
        current_ned = [self.state.pos_x_ned, self.state.pos_y_ned, self.state.pos_z_ned]
        clearance, nearest_name = point_clearance_to_inflated_obstacles(
            point_ned=current_ned,
            obstacles=self.obstacles,
            extra_clearance_m=0.0,
        )

        clearance_text = "inf" if math.isinf(clearance) else f"{clearance:.3f}"

        print(
            "[Mission] "
            f"pos={format_vec(current_ned)} | "
            f"lookahead={format_vec(self.lookahead_target_ned)} | "
            f"seg={self.active_path_segment_index} | "
            f"progress={self.path_progress_m:.2f}/{self.path_total_length_m:.2f} m | "
            f"remaining={self.remaining_path_length_m:.2f} m | "
            f"clearance={clearance_text} m near {nearest_name} | "
            f"cmd=({self.last_cmd[0]:+.2f}, {self.last_cmd[1]:+.2f}, {self.last_cmd[2]:+.2f}, {self.last_cmd[3]:+.2f})"
        )

    def compute_velocity_command(
        self,
        target_ned: List[float],
        max_speed_xy: float,
        max_speed_z: float,
    ) -> Tuple[float, float, float, float]:
        ex = target_ned[0] - self.state.pos_x_ned
        ey = target_ned[1] - self.state.pos_y_ned
        ez = target_ned[2] - self.state.pos_z_ned

        cmd_vx = KP_XY * ex
        cmd_vy = KP_XY * ey
        cmd_vz = KP_Z * ez

        cmd_vx, cmd_vy = scale_xy_to_limit(cmd_vx, cmd_vy, max_speed_xy)
        cmd_vz = clamp(cmd_vz, -max_speed_z, max_speed_z)

        if norm2(ex, ey) > 0.2:
            desired_yaw = math.atan2(ey, ex)
        else:
            desired_yaw = self.state.yaw

        yaw_error = normalize_angle_rad(desired_yaw - self.state.yaw)
        cmd_yaw_rate = clamp(
            KP_YAW * yaw_error,
            -MAX_YAW_RATE_RADPS,
            MAX_YAW_RATE_RADPS,
        )

        if math.isnan(cmd_yaw_rate):
            cmd_yaw_rate = 0.0

        return cmd_vx, cmd_vy, cmd_vz, cmd_yaw_rate

    def reset_command_smoother(self):
        self.smoothed_cmd = [0.0, 0.0, 0.0, 0.0]
        self.last_command_smooth_time = time.time()

    def smooth_velocity_command(
        self,
        raw_cmd: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        if not ENABLE_COMMAND_SMOOTHING:
            return raw_cmd

        now = time.time()
        dt = max(1e-3, min(0.20, now - self.last_command_smooth_time))
        self.last_command_smooth_time = now

        raw_vx, raw_vy, raw_vz, raw_yaw_rate = raw_cmd
        old_vx, old_vy, old_vz, old_yaw_rate = self.smoothed_cmd

        max_delta_xy = MAX_ACCEL_XY_MPS2 * dt
        max_delta_z = MAX_ACCEL_Z_MPS2 * dt
        max_delta_yaw = MAX_YAW_ACCEL_RADPS2 * dt

        def limit_delta(new_value: float, old_value: float, max_delta: float) -> float:
            delta = new_value - old_value
            delta = clamp(delta, -max_delta, max_delta)
            return old_value + delta

        new_vx = limit_delta(raw_vx, old_vx, max_delta_xy)
        new_vy = limit_delta(raw_vy, old_vy, max_delta_xy)
        new_vz = limit_delta(raw_vz, old_vz, max_delta_z)
        new_yaw_rate = limit_delta(raw_yaw_rate, old_yaw_rate, max_delta_yaw)

        self.smoothed_cmd = [new_vx, new_vy, new_vz, new_yaw_rate]
        return new_vx, new_vy, new_vz, new_yaw_rate

    def send_control_velocity_setpoint(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float,
    ):
        cmd = self.smooth_velocity_command((vx, vy, vz, yaw_rate))
        self.send_velocity_setpoint(*cmd)

    def send_velocity_setpoint(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float,
    ):
        if self.master is None:
            return

        # Use velocity and yaw rate.
        # Ignore position, acceleration, and yaw.
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )

        time_boot_ms = int((time.time() - self.boot_wall) * 1000.0) & 0xFFFFFFFF

        self.master.mav.set_position_target_local_ned_send(
            time_boot_ms,
            self.target_system(),
            self.target_component(),
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            int(type_mask),
            0.0,
            0.0,
            0.0,
            float(vx),
            float(vy),
            float(vz),
            0.0,
            0.0,
            0.0,
            0.0,
            float(yaw_rate),
        )

        self.last_cmd = (
            float(vx),
            float(vy),
            float(vz),
            float(yaw_rate),
        )

    def drain_telemetry(self):
        if self.master is None:
            return

        while True:
            msg = self.master.recv_match(blocking=False)

            if msg is None:
                break

            msg_type = msg.get_type()

            if msg_type == "BAD_DATA":
                continue

            if msg_type == "HEARTBEAT":
                self.update_heartbeat_state(msg)

            elif msg_type == "COMMAND_ACK":
                try:
                    self.command_ack_results[int(msg.command)] = int(msg.result)
                except Exception:
                    pass

            elif msg_type == "LOCAL_POSITION_NED":
                self.state.has_position = True
                self.state.pos_x_ned = float(msg.x)
                self.state.pos_y_ned = float(msg.y)
                self.state.pos_z_ned = float(msg.z)
                self.state.vel_x_ned = float(msg.vx)
                self.state.vel_y_ned = float(msg.vy)
                self.state.vel_z_ned = float(msg.vz)

            elif msg_type == "ATTITUDE":
                self.state.has_attitude = True
                self.state.roll = float(msg.roll)
                self.state.pitch = float(msg.pitch)
                self.state.yaw = float(msg.yaw)

    def is_target_reached(self, target_ned: List[float], is_final: bool = False) -> bool:
        if not self.state.has_position:
            return False

        dx = target_ned[0] - self.state.pos_x_ned
        dy = target_ned[1] - self.state.pos_y_ned
        dz = target_ned[2] - self.state.pos_z_ned

        dist_xy = norm2(dx, dy)
        dist_z = abs(dz)

        radius_xy = (
            FINAL_WAYPOINT_REACHED_RADIUS_XY_M
            if is_final
            else INTERMEDIATE_WAYPOINT_REACHED_RADIUS_XY_M
        )

        return (
            dist_xy <= radius_xy
            and dist_z <= WAYPOINT_REACHED_RADIUS_Z_M
        )

    def update_executed_trail_visualization(self):
        if not DRAW_PATH_VISUALIZATION_IN_STAGE or not DRAW_EXECUTED_TRAIL:
            return

        if not self.state.has_position:
            return

        current_ned = [self.state.pos_x_ned, self.state.pos_y_ned, self.state.pos_z_ned]

        if self.executed_trail_last_ned is not None:
            if distance_xy(current_ned, self.executed_trail_last_ned) < EXECUTED_TRAIL_MIN_SPACING_M:
                return

        self.executed_trail_last_ned = list(current_ned)
        point_isaac = ned_to_isaac_position(current_ned)
        point_isaac[2] += PATH_VISUAL_Z_OFFSET_M + 0.08
        self.executed_trail_points_isaac.append(
            Gf.Vec3f(float(point_isaac[0]), float(point_isaac[1]), float(point_isaac[2]))
        )

        if len(self.executed_trail_points_isaac) > EXECUTED_TRAIL_MAX_POINTS:
            self.executed_trail_points_isaac = self.executed_trail_points_isaac[-EXECUTED_TRAIL_MAX_POINTS:]

        if len(self.executed_trail_points_isaac) < 2:
            return

        try:
            stage = get_stage()
            ensure_xform(stage, PATH_VIS_ROOT)
            create_or_update_curve(
                stage=stage,
                curve_path=f"{PATH_VIS_ROOT}/ExecutedTrail",
                points=self.executed_trail_points_isaac,
                width_m=EXECUTED_TRAIL_LINE_WIDTH_M,
                color_rgb=COLOR_EXECUTED_TRAIL,
            )
        except Exception as exc:
            # Do not stop the mission because of visualization issues.
            if self.sample_index % max(1, int(LOG_RATE_HZ * 5.0)) == 0:
                print(f"[Visualizer] Warning: could not update executed trail: {exc}")

    def update_metrics(self):
        if not self.state.has_position:
            return

        self.update_executed_trail_visualization()

        current_xy = [self.state.pos_x_ned, self.state.pos_y_ned]

        if self.previous_position_xy is not None:
            step = distance_xy(current_xy, self.previous_position_xy)

            if step < 3.0:
                self.path_length_so_far += step

        self.previous_position_xy = current_xy

        nearest = self.find_nearest_obstacle()

        if nearest is not None:
            _, _, _, clearance = nearest
            self.min_distance_to_obstacle_so_far = min(
                self.min_distance_to_obstacle_so_far,
                clearance,
            )

    def find_nearest_obstacle(self):
        if not self.state.has_position or not self.obstacles:
            return None

        uav_xy = [self.state.pos_x_ned, self.state.pos_y_ned]

        best = None
        best_clearance = float("inf")

        for obs in self.obstacles:
            obs_xy = [obs.center_ned[0], obs.center_ned[1]]
            center_distance = distance_xy(uav_xy, obs_xy)
            clearance = center_distance - obs.estimated_radius_m

            if clearance < best_clearance:
                best_clearance = clearance
                best = (obs, obs_xy, center_distance, clearance)

        return best

    def compute_distance_to_final_goal_xy(self) -> float:
        if not self.state.has_position:
            return float("nan")

        return distance_xy(
            [self.state.pos_x_ned, self.state.pos_y_ned],
            [self.final_goal_ned[0], self.final_goal_ned[1]],
        )

    def log_if_needed(self, last_log_time: float) -> float:
        now = time.time()

        if now - last_log_time >= 1.0 / LOG_RATE_HZ:
            self.log_row()
            return now

        return last_log_time

    def log_row(self):
        nearest = self.find_nearest_obstacle()

        if nearest is None:
            nearest_name = ""
            nearest_x = float("nan")
            nearest_y = float("nan")
            nearest_radius = float("nan")
            nearest_clearance = float("nan")
        else:
            obs, _, _, clearance = nearest
            nearest_name = obs.name
            nearest_x = obs.center_ned[0]
            nearest_y = obs.center_ned[1]
            nearest_radius = obs.estimated_radius_m
            nearest_clearance = clearance

        if self.red_point is None:
            red_isaac = [float("nan"), float("nan"), float("nan")]
            red_ned = [float("nan"), float("nan"), float("nan")]
        else:
            red_isaac = self.red_point.position_isaac
            red_ned = self.red_point.position_ned

        mission_time = 0.0

        if self.mission_start_wall is not None:
            mission_time = time.time() - self.mission_start_wall

        row = {
            "episode_id": self.episode_id,
            "time_wall": time.time(),
            "mission_time": mission_time,
            "sample_index": self.sample_index,
            "phase": self.phase,
            "waypoint_index": self.active_waypoint_index,
            "waypoint_label": self.active_waypoint_label,

            "pos_x_ned": self.state.pos_x_ned,
            "pos_y_ned": self.state.pos_y_ned,
            "pos_z_ned": self.state.pos_z_ned,

            "vel_x_ned": self.state.vel_x_ned,
            "vel_y_ned": self.state.vel_y_ned,
            "vel_z_ned": self.state.vel_z_ned,

            "roll": self.state.roll,
            "pitch": self.state.pitch,
            "yaw": self.state.yaw,

            "final_goal_x_ned": self.final_goal_ned[0],
            "final_goal_y_ned": self.final_goal_ned[1],
            "final_goal_z_ned": self.final_goal_ned[2],

            "active_target_x_ned": self.active_target_ned[0],
            "active_target_y_ned": self.active_target_ned[1],
            "active_target_z_ned": self.active_target_ned[2],

            "cmd_vx_ned": self.last_cmd[0],
            "cmd_vy_ned": self.last_cmd[1],
            "cmd_vz_ned": self.last_cmd[2],
            "cmd_yaw_rate": self.last_cmd[3],

            "obstacle_count": len(self.obstacles),
            "nearest_obstacle_name": nearest_name,
            "nearest_obstacle_x_ned": nearest_x,
            "nearest_obstacle_y_ned": nearest_y,
            "nearest_obstacle_radius_m": nearest_radius,
            "distance_nearest_obstacle_xy": nearest_clearance,

            "red_point_x_isaac": red_isaac[0],
            "red_point_y_isaac": red_isaac[1],
            "red_point_z_isaac": red_isaac[2],
            "red_point_x_ned": red_ned[0],
            "red_point_y_ned": red_ned[1],
            "red_point_z_ned": red_ned[2],

            "distance_final_goal_xy": self.compute_distance_to_final_goal_xy(),
            "min_distance_to_obstacle_so_far": self.min_distance_to_obstacle_so_far,
            "path_length_so_far": self.path_length_so_far,
            "success_auto": int(self.success_auto),

            "planner_type": self.planner_summary.planner_type,
            "grid_resolution_m": self.planner_summary.grid_resolution_m,
            "astar_raw_path_point_count": self.planner_summary.astar_raw_path_point_count,
            "astar_waypoint_count": self.planner_summary.astar_waypoint_count,
            "path_is_safe": int(self.planner_summary.path_is_safe),
            "grid_min_x": self.planner_summary.grid_min_x,
            "grid_max_x": self.planner_summary.grid_max_x,
            "grid_min_y": self.planner_summary.grid_min_y,
            "grid_max_y": self.planner_summary.grid_max_y,
            "grid_width": self.planner_summary.grid_width,
            "grid_height": self.planner_summary.grid_height,
            "occupied_cell_count": self.planner_summary.occupied_cell_count,
            "extra_grid_clearance_m": self.planner_summary.extra_grid_clearance_m,

            "occupied_ratio": self.planner_summary.occupied_ratio,
            "use_direct_path_bias": int(self.planner_summary.use_direct_path_bias),
            "direct_path_bias_weight": self.planner_summary.direct_path_bias_weight,
            "use_lookahead_path_following": int(self.planner_summary.use_lookahead_path_following),
            "lookahead_target_x_ned": self.lookahead_target_ned[0],
            "lookahead_target_y_ned": self.lookahead_target_ned[1],
            "lookahead_target_z_ned": self.lookahead_target_ned[2],
            "lookahead_distance_m": self.lookahead_distance_m,
            "path_following_mode": self.path_following_mode,
            "active_path_segment_index": self.active_path_segment_index,
            "path_progress_m": self.path_progress_m,
            "remaining_path_length_m": self.remaining_path_length_m,
            "final_path_total_length_m": self.path_total_length_m,
        }

        self.logger.write_row(row)
        self.sample_index += 1

        if self.sample_index % int(LOG_RATE_HZ) == 0:
            self.logger.flush()

    def cleanup(self):
        print("[Runner] Cleaning up ...")

        self.stop_front_camera_recording_if_available()

        try:
            self.send_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass

        try:
            self.logger.close()
            print(f"[Logger] Mission CSV closed: {self.logger.csv_path}")
        except Exception:
            pass

        self.phase = "stopped"
        print("[Runner] Stopped.")


# ============================================================
# KEYBOARD HELPERS
# ============================================================

def key_matches(key_text: str, candidates: List[str]) -> bool:
    normalized = key_text.upper()

    for candidate in candidates:
        candidate = candidate.upper()

        if normalized == candidate:
            return True

        if normalized.endswith("." + candidate):
            return True

        if normalized.endswith(candidate):
            return True

    return False


# ============================================================
# MAIN ENTRY
# ============================================================

def main():
    global PX4_EPISODE_RUNNER

    try:
        previous_runner = PX4_EPISODE_RUNNER
    except NameError:
        previous_runner = None

    if previous_runner is not None:
        try:
            print("[Main] Previous runner found. Requesting stop without landing.")
            previous_runner.request_stop(send_land=False)
        except Exception:
            pass

    episode_id = time.strftime("%Y%m%d_%H%M%S")

    print("")
    print("========== Isaac A* ROS2 Path Publisher ==========")
    print("[Main] This script plans in Isaac Sim and publishes nav_msgs/Path to ROS2.")
    print("[Main] Please run scene_episode_generator.py before this runner.")
    print(f"[Main] Obstacle root: {OBSTACLE_ROOT_PATH}")
    print(f"[Main] Red point path: {RED_POINT_PATH}")
    print(f"[Main] SCENE_READER_USE_LIVE_STAGE_TRANSFORMS={SCENE_READER_USE_LIVE_STAGE_TRANSFORMS}")
    print(f"[Main] READ_OBSTACLES_AS_LIVE_TOP_LEVEL_OBJECTS={READ_OBSTACLES_AS_LIVE_TOP_LEVEL_OBJECTS}")
    print(f"[Main] USE_STAGE_TARGET_AS_FINAL_GOAL={USE_STAGE_TARGET_AS_FINAL_GOAL}")
    print(f"[Main] EXPERIMENT_PRESET_NAME={EXPERIMENT_PRESET_NAME}")
    print("========================================================")
    print("")

    obstacles = read_obstacles_from_stage()

    if len(obstacles) == 0:
        raise RuntimeError(
            "No obstacles were loaded. Please run scene_episode_generator.py first "
            f"and make sure obstacles are created under {OBSTACLE_ROOT_PATH}."
        )

    red_point = read_red_point_from_stage()

    print_scene_summary(obstacles, red_point)
    print_planner_parameter_summary()
    all_gap_diagnostics = print_obstacle_gap_diagnostics(obstacles)
    planning_obstacles = filter_obstacles_for_2p5d_planning(obstacles)
    if len(planning_obstacles) != len(obstacles):
        planning_gap_diagnostics = print_obstacle_gap_diagnostics(planning_obstacles)
    else:
        planning_gap_diagnostics = all_gap_diagnostics
    draw_safety_envelope_visualization(planning_obstacles, planning_gap_diagnostics)
    save_scene_summary_csv(episode_id, obstacles, red_point)

    start_isaac = list(START_ISAAC)
    final_goal_isaac = resolve_final_goal_isaac(red_point)

    start_ned = isaac_ground_point_to_flight_ned(start_isaac)
    final_goal_ned = isaac_ground_point_to_flight_ned(final_goal_isaac)

    print(f"[Main] START_ISAAC={format_vec(start_isaac)}, start_ned={format_vec(start_ned)}")
    print(f"[Main] final_goal_isaac={format_vec(final_goal_isaac)}, final_goal_ned={format_vec(final_goal_ned)}")
    print(f"[Main] SWAP_XY={SWAP_XY}")

    try:
        planning_result = plan_waypoints(
            start_ned=start_ned,
            goal_ned=final_goal_ned,
            obstacles=planning_obstacles,
        )
    except RuntimeError as exc:
        print("")
        print("========== Planner Abort ==========")
        print("[Main] A safe A* path could not be produced.")
        print("[Main] The UAV will NOT connect to PX4 and will NOT start the mission.")
        print(f"[Main] Reason: {exc}")
        print("[Main] Check the Obstacle Gap Diagnostics above. If many gaps are TOO_NARROW, reduce safety margins or move obstacles farther apart.")
        print("===================================")
        print("")
        return

    if not planning_result.summary.path_is_safe:
        print("")
        print("========== Planner Abort ==========")
        print("[Main] Planner produced an unsafe path. Mission aborted before PX4 connection.")
        print("===================================")
        print("")
        return

    draw_planned_path_visualization(
        raw_path_ned=planning_result.raw_path_ned,
        simplified_path_ned=planning_result.simplified_path_ned,
        waypoints=planning_result.waypoints,
    )

    if PUBLISH_ROS2_PATH_ONLY:
        ros2_waypoints = build_ros2_path_waypoints(
            start_ned=start_ned,
            planning_result=planning_result,
        )
        publisher = IsaacAStarRos2PathPublisher(
            waypoints_ned=ros2_waypoints,
            topic_name=ROS2_PATH_TOPIC,
            frame_id=ROS2_PATH_FRAME_ID,
            publish_rate_hz=ROS2_PATH_PUBLISH_RATE_HZ,
        )
        publisher.start()
        print("[Main] PUBLISH_ROS2_PATH_ONLY=True. This script will NOT connect to PX4 directly.")
        print("[Main] Run: ros2 launch uav_px4_control astar_path_mission.launch.py")
        return
    
    PX4_EPISODE_RUNNER = Px4EpisodeRunner(
        episode_id=episode_id,
        obstacles=planning_obstacles,
        red_point=red_point,
        waypoints=planning_result.waypoints,
        start_ned=start_ned,
        final_goal_ned=final_goal_ned,
        path_points_ned=planning_result.simplified_path_ned,
        planner_summary=planning_result.summary,
    )

    PX4_EPISODE_RUNNER.start()


if RUN_ON_PASTE:
    main()
