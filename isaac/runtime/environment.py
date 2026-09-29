#!/usr/bin/env python3
"""Isaac 虛擬環境的唯一實作模組。

本檔集中管理障礙物與場景生成、顏色與材質、燈光，以及僅在
Isaac Sim 中使用的 USD、物理、相機和 ROS runtime bridge。
一般 Python 的資料集工具可安全匯入場景生成部分；Isaac 專屬
依賴只會在可用時載入。
"""

from __future__ import annotations

# RGB values are linear USD colors.  The light gray floor and navy obstacles
# provide both luminance and hue contrast in TOP and FPV images.
# 障礙物本體 RGB 顏色（0.0 到 1.0）。
OBSTACLE_COLOR = (0.03, 0.08, 0.18)
# 地板 RGB 顏色（0.0 到 1.0）。
FLOOR_COLOR = (0.60, 0.60, 0.60)
# 四面牆壁 RGB 顏色（0.0 到 1.0）。
WALL_COLOR = (0.45, 0.45, 0.45)
# 起點標記 RGB 顏色（0.0 到 1.0）。
START_MARKER_COLOR = (0.0, 0.3, 1.0)
# 終點標記 RGB 顏色（0.0 到 1.0）。
GOAL_MARKER_COLOR = (1.0, 0.0, 0.0)

MATERIAL_ROUGHNESS = 1.0
MATERIAL_METALLIC = 0.0
MATERIAL_SPECULAR_COLOR = (0.0, 0.0, 0.0)
MATERIAL_OPACITY = 1.0
MATERIAL_EMISSIVE_COLOR = (0.0, 0.0, 0.0)

# The Pegasus default environment contains a local 100000-intensity
# SphereLight.  Disable environment lights and replace them with one neutral,
# direction-independent DomeLight for the formal ML scene.
DISABLE_ENVIRONMENT_LIGHTS = True  # 是否關閉 Pegasus 原有環境光源。
DOME_LIGHT_INTENSITY = 800.0  # Dome 全域光照強度。
DOME_LIGHT_COLOR = (1.0, 1.0, 1.0)  # Dome 全域光照 RGB 顏色。
RTX_SHADOWS_ENABLED = False  # 是否啟用 RTX 陰影。
RTX_AMBIENT_OCCLUSION_ENABLED = False  # 是否啟用 RTX 環境遮蔽。

OBSTACLE_MATERIAL_NAME = "ObstacleMatte"
FLOOR_MATERIAL_NAME = "FloorMatte"
WALL_MATERIAL_NAME = "WallMatte"
START_MARKER_MATERIAL_NAME = "StartMarkerMatte"
GOAL_MARKER_MATERIAL_NAME = "GoalMarkerMatte"


def create_matte_material(stage, path: str, color):
    """Create one texture-free UsdPreviewSurface matte material."""
    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput(
        "diffuseColor", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(*map(float, color)))
    shader.CreateInput(
        "roughness", Sdf.ValueTypeNames.Float
    ).Set(float(MATERIAL_ROUGHNESS))
    shader.CreateInput(
        "metallic", Sdf.ValueTypeNames.Float
    ).Set(float(MATERIAL_METALLIC))
    shader.CreateInput(
        "useSpecularWorkflow", Sdf.ValueTypeNames.Int
    ).Set(1)
    shader.CreateInput(
        "specularColor", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(*map(float, MATERIAL_SPECULAR_COLOR)))
    shader.CreateInput(
        "opacity", Sdf.ValueTypeNames.Float
    ).Set(float(MATERIAL_OPACITY))
    shader.CreateInput(
        "emissiveColor", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(*map(float, MATERIAL_EMISSIVE_COLOR)))
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return material


def create_scene_materials(stage, scene_root: str):
    """Create the shared scene materials once below ``scene_root``."""
    materials_root = f"{scene_root}/Materials"
    return {
        "obstacle": create_matte_material(
            stage,
            f"{materials_root}/{OBSTACLE_MATERIAL_NAME}",
            OBSTACLE_COLOR,
        ),
        "floor": create_matte_material(
            stage,
            f"{materials_root}/{FLOOR_MATERIAL_NAME}",
            FLOOR_COLOR,
        ),
        "wall": create_matte_material(
            stage,
            f"{materials_root}/{WALL_MATERIAL_NAME}",
            WALL_COLOR,
        ),
        "start_marker": create_matte_material(
            stage,
            f"{materials_root}/{START_MARKER_MATERIAL_NAME}",
            START_MARKER_COLOR,
        ),
        "goal_marker": create_matte_material(
            stage,
            f"{materials_root}/{GOAL_MARKER_MATERIAL_NAME}",
            GOAL_MARKER_COLOR,
        ),
    }


def bind_material(prim, material) -> None:
    """Bind a shared USD material without changing geometry or physics APIs."""
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


import math
from pathlib import Path
import random
import re
import sys


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
from formal_expert_sensor_contract import TOP_RGB_COVERAGE_M

# ===== 可調整場景參數（修改此區即可改變新生成 episode 的場景） =====
# 是否強制在起點至終點直線上放置 blocker。
GUARANTEE_DIRECT_PATH_BLOCKERS = True
# 直線 blocker 數量；設為 0 可關閉直線障礙物。
DIRECT_PATH_BLOCKER_COUNT = 2
# 額外隨機障礙物數量；設為 0 即不生成隨機障礙物。
RANDOM_OBSTACLE_COUNT = 0
# 場景障礙物總數，由直線 blocker 與隨機障礙物數量自動相加。
NUM_OBSTACLES = DIRECT_PATH_BLOCKER_COUNT + RANDOM_OBSTACLE_COUNT
# 兩個直線 blocker 在起點到終點線段上的比例範圍。
DIRECT_PATH_BLOCKER_T_RANGES = ((0.34, 0.38), (0.62, 0.66))
# 直線 blocker 可離開中心線的最大橫向距離（公尺）。
DIRECT_PATH_BLOCKER_LATERAL_JITTER_M = 0.08
# 直線 blocker 的最小高度（公尺）。
DIRECT_PATH_BLOCKER_HEIGHT_MIN = 3.20
# 特大型 blocker 的索引（從 0 開始）。
SPECIAL_BLOCKER_INDEX = 0
# 特大型 blocker 的寬度基準（公尺）。
SPECIAL_BLOCKER_RADIUS_BASIS_WIDTH = 0.72
# 特大型 blocker 的深度基準（公尺）。
SPECIAL_BLOCKER_RADIUS_BASIS_DEPTH = 0.72
# 特大型 blocker 的高度（公尺）。
SPECIAL_BLOCKER_HEIGHT = 5.20
# 正式 episode 使用固定障礙物；episode seed 不影響障礙物幾何或外觀。
FIXED_OBSTACLE_DECORATION_SEED = 0
FIXED_OBSTACLE_LAYOUT = (
    {
        "x": 1.08,
        "y": 1.80,
        "radius_basis_width": 0.72,
        "radius_basis_depth": 0.72,
        "height": 5.20,
        "yaw_deg": 0.0,
        "placement_mode": "guaranteed_direct_path_blocker",
        "variant": "special_large",
    },
    {
        "x": 1.92,
        "y": 3.20,
        "radius_basis_width": 0.64,
        "radius_basis_depth": 0.64,
        "height": 4.00,
        "yaw_deg": 0.0,
        "placement_mode": "guaranteed_direct_path_blocker",
        "variant": "standard",
    },
)

# 可飛行區域與牆壁內側面的 X 座標範圍（公尺）。
X_MIN = -5.0
X_MAX = 5.0
# 可飛行區域與牆壁內側面的 Y 座標範圍（公尺）。
Y_MIN = -2.0
Y_MAX = 7.0
# 牆壁厚度（公尺）。
WALL_THICKNESS_M = 0.25
# 牆壁高度（公尺）。
WALL_HEIGHT_M = 5.50

# 無人機起點位置 x, y, z（公尺）。
START_POS = (0.0, 0.0, 0.0)
# 終點地面標記位置 x, y, z（公尺）。
TARGET_POS = (3.0, 5.0, 0.0)
# 飛行目標高度（公尺）。
FLIGHT_ALTITUDE_M = 1.5
# 起點與終點圓形標記半徑（公尺）。
DISK_RADIUS = 0.5

# 額外安全距離
DISK_SAFE_MARGIN = 1.0  # 起點與終點周圍禁止生成障礙物的額外距離（公尺）。
START_CLEAR_RADIUS = DISK_RADIUS + DISK_SAFE_MARGIN
TARGET_CLEAR_RADIUS = DISK_RADIUS + DISK_SAFE_MARGIN

RADIUS_BASIS_WIDTH_MIN = 0.46  # 隨機障礙物最小寬度基準（公尺）。
RADIUS_BASIS_WIDTH_MAX = 0.72  # 隨機障礙物最大寬度基準（公尺）。
RADIUS_BASIS_DEPTH_MIN = 0.46  # 隨機障礙物最小深度基準（公尺）。
RADIUS_BASIS_DEPTH_MAX = 0.72  # 隨機障礙物最大深度基準（公尺）。

# 障礙物高度最小值和最大值
CYLINDER_HEIGHT_MIN = 2.80  # 隨機障礙物最小高度（公尺）。
CYLINDER_HEIGHT_MAX = 5.20  # 隨機障礙物最大高度（公尺）。

BLOCKER_RADIUS_BASIS_WIDTH_MIN = 0.56  # 一般直線 blocker 最小寬度基準（公尺）。
BLOCKER_RADIUS_BASIS_DEPTH_MIN = 0.56  # 一般直線 blocker 最小深度基準（公尺）。
# These decoration samples remain in the RNG contract. Removing their draws
# changes later placement draws and therefore seed-to-geometry mapping.
OBSTACLE_YAW_MIN_DEG = -35.0  # 障礙物最小偏航角（度）。
OBSTACLE_YAW_MAX_DEG = 35.0  # 障礙物最大偏航角（度）。
SCENE_DECORATION_WINDOW_THICKNESS_M = 0.018
SCENE_DECORATION_WINDOW_HEIGHT_M = 0.16
SCENE_DECORATION_WINDOW_MARGIN_M = 0.08
SCENE_DECORATION_ROOF_HEIGHT_MIN = 0.10
SCENE_DECORATION_ROOF_HEIGHT_MAX = 0.28
SCENE_DECORATION_ANTENNA_HEIGHT_MIN = 0.25
SCENE_DECORATION_ANTENNA_HEIGHT_MAX = 0.55
# Retained only to consume the historical RNG draw; these values are never
# assigned to rendered obstacles now that visual color is centralized.
LEGACY_CYLINDER_COLOR_DRAWS = (
    (0.12, 0.18, 0.24),
    (0.20, 0.27, 0.31),
    (0.30, 0.31, 0.34),
    (0.27, 0.22, 0.20),
    (0.18, 0.22, 0.30),
)
SCENE_DECORATION_WINDOW_ON_COLORS = (
    (0.38, 0.72, 1.00),
    (0.62, 0.86, 1.00),
    (1.00, 0.78, 0.36),
)
SCENE_DECORATION_WINDOW_OFF_COLOR = (0.035, 0.055, 0.075)
SCENE_DECORATION_ROOF_STYLES = ("flat", "crown", "antenna")

# 兩個 cylinder 外緣間的最小實體距離，不是中心距離
MIN_OBSTACLE_GAP = 0.50  # 兩個圓柱表面間的最小間距（公尺）。

MAX_PLACEMENT_ATTEMPTS = 1000  # 每個障礙物允許的最大隨機放置嘗試次數。
RESET_POSITION_TOLERANCE_M = 0.50  # 重置位置距離起點的最大容許誤差（公尺）。
LIGHTING_CONTRACT = {
    "mode": "neutral_dome_only",
    "root": "/World/GeneratedEpisode/Lights",
    "dome": {
        "intensity": DOME_LIGHT_INTENSITY,
        "exposure": 0.0,
        "color": list(DOME_LIGHT_COLOR),
    },
}

# ===== 可調整相機、ROS 輸出與預覽參數 =====
# 無人機本體 USD Prim 路徑。
VEHICLE_BODY_PATH = "/World/quadrotor/body"
# 發布無人機姿態的 ROS topic。
POSE_TOPIC = "/isaac_uav/pose"
# 發布場景與 runtime 狀態的 ROS topic。
STATUS_TOPIC = "/uav/isaac/runtime_status"
# 場景資料使用的座標框架名稱。
FRAME_ID = "isaac_world"
# FPV RGB 影像輸出 topic。
CAMERA_TOPIC = "/uav/isaac/fpv/image/compressed"
# FPV 相機 USD 路徑。
CAMERA_PATH = "/World/RuntimeSensors/FPVCamera"
# Observer 相機 USD 路徑。
OBSERVER_CAMERA_PATH = "/World/RuntimeSensors/ObserverCamera"
# Observer RGB 影像輸出 topic。
OBSERVER_CAMERA_TOPIC = "/uav/isaac/observer/image/compressed"
# FPV 深度影像輸出 topic。
DEPTH_TOPIC = "/uav/isaac/fpv/depth/compressed"
# 場景切換命令接收 topic。
EPISODE_COMMAND_TOPIC = "/uav/isaac/episode_command"
# 正式生成場景的 USD 根路徑。
SCENE_ROOT = "/World/GeneratedEpisode"
# Bootstrap 預覽場景的 USD 根路徑。
BOOTSTRAP_SCENE_ROOT = "/World/BootstrapScene"
# 軌跡回放物件的 USD 根路徑。
REPLAY_ROOT = "/World/TrajectoryReplay"
# JPEG 影像品質（1 到 100）。
JPEG_QUALITY = 85
# FPV 深度影像發布週期（秒）。
DEPTH_PUBLISH_PERIOD_S = 0.20
# 深度相機最小有效距離（公尺）。
DEPTH_MIN_M = 0.05
# 深度相機最大有效距離（公尺）。
DEPTH_MAX_M = 30.0
# FPV 相機相對機體前方偏移（公尺）。
FPV_FORWARD_OFFSET_M = 0.45
# FPV 相機相對機體垂直偏移（公尺）。
FPV_HEIGHT_M = 0.12
# FPV 相機注視點前方距離（公尺）。
FPV_LOOK_AHEAD_M = 3.5
# FPV 相機注視點垂直偏移（公尺；負值向下）。
FPV_LOOK_DOWN_M = -0.8
# FPV 相機焦距。
FPV_FOCAL_LENGTH = 12.0
# FPV 相機水平感光面寬度。
FPV_HORIZONTAL_APERTURE = 28.0
# 非正式模式的 Observer 視角模式（TOP 或追蹤視角）。
OBSERVER_MODE = "TOP"
# 追蹤式 Observer 相機後方距離（公尺）。
OBSERVER_BACK_DISTANCE_M = 3.2
# 追蹤式 Observer 相機高度（公尺）。
OBSERVER_HEIGHT_M = 5.2
# 追蹤式 Observer 相機側向偏移（公尺）。
OBSERVER_SIDE_OFFSET_M = 2.2
# 追蹤式 Observer 相機前方注視距離（公尺）。
OBSERVER_LOOK_AHEAD_M = 2.5
# 追蹤式 Observer 相機注視高度偏移（公尺）。
OBSERVER_LOOK_AT_HEIGHT_M = -1.2
# TOP Observer 相機高度（公尺）。
OBSERVER_TOP_HEIGHT_M = 9.0
# TOP Observer 相機注視高度（公尺）。
OBSERVER_TOP_LOOK_AT_HEIGHT_M = 0.0
# Observer 相機焦距。
OBSERVER_FOCAL_LENGTH = 18.0
# Observer 相機水平感光面寬度。
OBSERVER_HORIZONTAL_APERTURE = 22.0
# 正式資料集 TOP 相機眼睛位置 x, y, z（公尺）。
FORMAL_OBSERVER_EYE = (0.0, 2.5, 15.0)
# 正式資料集 TOP 相機注視位置 x, y, z（公尺）。
FORMAL_OBSERVER_TARGET = (0.0, 2.5, 0.0)
# 正式資料集 TOP 相機向上方向 x, y, z。
FORMAL_OBSERVER_UP = (0.0, 1.0, 0.0)
# 正式資料集 TOP 相機覆蓋寬、高（公尺）。
FORMAL_OBSERVER_COVERAGE_M = TOP_RGB_COVERAGE_M
# FPV 與 Observer 相機裁切距離範圍（公尺）。
CAMERA_CLIPPING_RANGE = (0.05, 10000.0)
# 相機追蹤平滑係數。
CAMERA_SMOOTHING = 0.18
# Bootstrap 預覽模式的目標與正式 episode 共用相同位置。
GOAL = (TARGET_POS[0], TARGET_POS[1], FLIGHT_ALTITUDE_M)
# Bootstrap 預覽模式的預設障礙物清單。
OBSTACLES = ({
    "name": "BootstrapObstacle_001",
    "x": -1.5,
    "y": 1.5,
    "z": 1.25,
    "radius": 0.43,
    "height": 2.5,
},)


def _distance_2d(
    left_x: float, left_y: float, right_x: float, right_y: float
) -> float:
    return math.hypot(left_x - right_x, left_y - right_y)


def _point_to_direct_path_distance(x: float, y: float) -> float:
    start_x, start_y = START_POS[:2]
    target_x, target_y = TARGET_POS[:2]
    dx = target_x - start_x
    dy = target_y - start_y
    length_squared = dx * dx + dy * dy
    if length_squared < 1e-9:
        return _distance_2d(x, y, start_x, start_y)
    t = ((x - start_x) * dx + (y - start_y) * dy) / length_squared
    t = max(0.0, min(1.0, t))
    return _distance_2d(x, y, start_x + t * dx, start_y + t * dy)


def _is_valid_obstacle_position(
    x: float,
    y: float,
    radius: float,
    placed: list[dict],
) -> bool:
    if _distance_2d(x, y, *START_POS[:2]) < START_CLEAR_RADIUS + radius:
        return False
    if _distance_2d(x, y, *TARGET_POS[:2]) < TARGET_CLEAR_RADIUS + radius:
        return False
    return all(
        _distance_2d(x, y, item["x"], item["y"])
        >= radius + item["radius"] + MIN_OBSTACLE_GAP
        for item in placed
    )

"""
目前沒有獨立的 CYLINDER_RADIUS_MIN/MAX。
Radius 是由 radius basis width/depth 計算：
radius = 0.5 * math.hypot(radius_basis_width, radius_basis_depth)
"""
def _random_cylinder_spec(
    rng: random.Random, blocker: bool = False, special: bool = False
) -> dict:
    if special:
        radius_basis_width = SPECIAL_BLOCKER_RADIUS_BASIS_WIDTH
        radius_basis_depth = SPECIAL_BLOCKER_RADIUS_BASIS_DEPTH
        height = SPECIAL_BLOCKER_HEIGHT
    else:
        radius_basis_width = rng.uniform(
            BLOCKER_RADIUS_BASIS_WIDTH_MIN
            if blocker else RADIUS_BASIS_WIDTH_MIN,
            RADIUS_BASIS_WIDTH_MAX,
        )
        radius_basis_depth = rng.uniform(
            BLOCKER_RADIUS_BASIS_DEPTH_MIN
            if blocker else RADIUS_BASIS_DEPTH_MIN,
            RADIUS_BASIS_DEPTH_MAX,
        )
        height = rng.uniform(
            max(CYLINDER_HEIGHT_MIN, DIRECT_PATH_BLOCKER_HEIGHT_MIN)
            if blocker else CYLINDER_HEIGHT_MIN,
            CYLINDER_HEIGHT_MAX,
        )
    yaw_deg = rng.uniform(OBSTACLE_YAW_MIN_DEG, OBSTACLE_YAW_MAX_DEG)
    # Preserve this legacy draw so every seed keeps its exact geometry and
    # decoration RNG sequence even though obstacle color is now fixed.
    rng.choice(LEGACY_CYLINDER_COLOR_DRAWS)
    return {
        "shape": "cylinder",
        "radius_basis_width": radius_basis_width,
        "radius_basis_depth": radius_basis_depth,
        "height": height,
        "radius": 0.5 * math.hypot(
            radius_basis_width, radius_basis_depth
        ),
        "blocker_half_extent": 0.5 * min(
            radius_basis_width, radius_basis_depth
        ),
        "yaw_deg": yaw_deg,
        "color": list(OBSTACLE_COLOR),
        "window_on_color": list(rng.choice(SCENE_DECORATION_WINDOW_ON_COLORS)),
        "window_off_color": list(SCENE_DECORATION_WINDOW_OFF_COLOR),
        "roof_style": rng.choice(SCENE_DECORATION_ROOF_STYLES),
        "roof_height": rng.uniform(
            SCENE_DECORATION_ROOF_HEIGHT_MIN,
            SCENE_DECORATION_ROOF_HEIGHT_MAX,
        ),
        "antenna_height": rng.uniform(
            SCENE_DECORATION_ANTENNA_HEIGHT_MIN,
            SCENE_DECORATION_ANTENNA_HEIGHT_MAX,
        ),
        "collision": True, # 所有障礙物都要有碰撞
    }


def _window_contract(spec: dict, rng: random.Random) -> dict:
    row_count = max(5, min(11, int(spec["height"] / 0.44)))
    columns_x = max(
        2, min(3, int(spec["radius_basis_width"] / 0.22))
    )
    columns_y = max(
        2, min(3, int(spec["radius_basis_depth"] / 0.22))
    )
    window_count = row_count * 2 * (columns_x + columns_y)
    return {
        "row_count": row_count,
        "columns_x": columns_x,
        "columns_y": columns_y,
        "height_m": SCENE_DECORATION_WINDOW_HEIGHT_M,
        "thickness_m": SCENE_DECORATION_WINDOW_THICKNESS_M,
        "margin_m": SCENE_DECORATION_WINDOW_MARGIN_M,
        "on_pattern": [rng.random() < 0.72 for _ in range(window_count)],
    }

# 強制生成有擋住的障礙物(blocked)
def _generate_obstacles(rng: random.Random) -> list[dict]:
    """Build the fixed obstacle geometry and seed-specific building details."""
    placed: list[dict] = []
    if len(FIXED_OBSTACLE_LAYOUT) != NUM_OBSTACLES:
        raise RuntimeError(
            "fixed obstacle layout must define every configured obstacle"
        )

    for index, layout in enumerate(FIXED_OBSTACLE_LAYOUT):
        spec = _random_cylinder_spec(
            rng,
            blocker=True,
            special=index == SPECIAL_BLOCKER_INDEX,
        )
        width = float(layout["radius_basis_width"])
        depth = float(layout["radius_basis_depth"])
        height = float(layout["height"])
        radius = 0.5 * math.hypot(width, depth)
        half_extent = 0.5 * min(width, depth)
        x = float(layout["x"])
        y = float(layout["y"])
        if not (
            X_MIN + radius <= x <= X_MAX - radius
            and Y_MIN + radius <= y <= Y_MAX - radius
            and _is_valid_obstacle_position(x, y, radius, placed)
            and _point_to_direct_path_distance(x, y) <= half_extent
            and height >= DIRECT_PATH_BLOCKER_HEIGHT_MIN
        ):
            raise RuntimeError(f"invalid fixed obstacle layout entry {index}")
        spec.update({
            "x": x,
            "y": y,
            "z": 0.5 * height,
            "radius_basis_width": width,
            "radius_basis_depth": depth,
            "height": height,
            "radius": radius,
            "blocker_half_extent": half_extent,
            "yaw_deg": float(layout["yaw_deg"]),
            "placement_mode": layout["placement_mode"],
            "variant": layout["variant"],
        })
        placed.append(spec)

    physical_blockers = [
        item for item in placed
        if _point_to_direct_path_distance(item["x"], item["y"])
        <= item["blocker_half_extent"]
        and item["height"] >= DIRECT_PATH_BLOCKER_HEIGHT_MIN
    ]
    if len(physical_blockers) < DIRECT_PATH_BLOCKER_COUNT:
        raise RuntimeError("canonical direct-path blocker validation failed")

    for index, spec in enumerate(placed, start=1):
        spec["name"] = f"Obstacle_{index:03d}"
        spec["windows"] = _window_contract(spec, rng)
        spec["hierarchy"] = ["Body", "Windows", "Roof/Crown"]
        if spec["roof_style"] == "antenna":
            spec["hierarchy"].append("Roof/Antenna")
    return placed


def _blocked_goal_fixture() -> dict:
    radius = 0.85
    side = radius * math.sqrt(2.0)
    spec = {
        "name": "Obstacle_blocked_goal",
        "shape": "cylinder",
        "x": TARGET_POS[0],
        "y": TARGET_POS[1],
        "z": 1.5,
        "radius_basis_width": side,
        "radius_basis_depth": side,
        "height": 3.0,
        "radius": radius,
        "blocker_half_extent": 0.5 * side,
        "yaw_deg": 0.0,
        "color": list(OBSTACLE_COLOR),
        "window_on_color": list(SCENE_DECORATION_WINDOW_ON_COLORS[0]),
        "window_off_color": list(SCENE_DECORATION_WINDOW_OFF_COLOR),
        "roof_style": "flat",
        "roof_height": SCENE_DECORATION_ROOF_HEIGHT_MIN,
        "antenna_height": SCENE_DECORATION_ANTENNA_HEIGHT_MIN,
        "collision": True,
        "placement_mode": "blocked_goal_safe_failure",
        "fixture": "blocked_goal_safe_failure",
        "hierarchy": ["Body", "Windows", "Roof/Crown"],
    }
    spec["windows"] = _window_contract(spec, random.Random(0))
    return spec


def _collision_walls() -> list[dict]:
    """Return the four static walls enclosing the navigable mission area."""
    half_thickness = 0.5 * WALL_THICKNESS_M
    center_x = 0.5 * (X_MIN + X_MAX)
    center_y = 0.5 * (Y_MIN + Y_MAX)
    span_x = X_MAX - X_MIN + 2.0 * WALL_THICKNESS_M
    span_y = Y_MAX - Y_MIN
    return [
        {
            "name": "Wall_West",
            "position": [X_MIN - half_thickness, center_y, WALL_HEIGHT_M / 2.0],
            "size": [WALL_THICKNESS_M, span_y, WALL_HEIGHT_M],
        },
        {
            "name": "Wall_East",
            "position": [X_MAX + half_thickness, center_y, WALL_HEIGHT_M / 2.0],
            "size": [WALL_THICKNESS_M, span_y, WALL_HEIGHT_M],
        },
        {
            "name": "Wall_South",
            "position": [center_x, Y_MIN - half_thickness, WALL_HEIGHT_M / 2.0],
            "size": [span_x, WALL_THICKNESS_M, WALL_HEIGHT_M],
        },
        {
            "name": "Wall_North",
            "position": [center_x, Y_MAX + half_thickness, WALL_HEIGHT_M / 2.0],
            "size": [span_x, WALL_THICKNESS_M, WALL_HEIGHT_M],
        },
    ]


def generate_episode_scene(
    episode_id: str,
    seed: int,
    reset_east_m: float,
    reset_north_m: float,
    mode: str = "normal",
) -> dict:
    """Generate the configured obstacle distribution for one safe reset."""
    if not re.fullmatch(r"episode_[0-9]{6,}", episode_id):
        raise ValueError(
            "episode_id must use episode_ followed by at least six digits"
        )
    if mode not in {"normal", "blocked_goal"}:
        raise ValueError(f"unsupported scene mode: {mode}")
    reset = (float(reset_east_m), float(reset_north_m))
    if not all(math.isfinite(value) for value in reset):
        raise ValueError("scene reset pose must be finite")
    if _distance_2d(*reset, *START_POS[:2]) > RESET_POSITION_TOLERANCE_M:
        raise ValueError("vehicle reset pose is outside the canonical start margin")

    obstacles = _generate_obstacles(
        random.Random(FIXED_OBSTACLE_DECORATION_SEED)
    )
    direct_blocker_count = sum(
        item["placement_mode"] == "guaranteed_direct_path_blocker"
        for item in obstacles
    )
    if mode == "blocked_goal":
        obstacles.append(_blocked_goal_fixture())

    return {
        "episode_id": episode_id,
        "random_seed": int(seed),
        "generator": "canonical_cylinder_scene_generator_v1",
        "reference": (
            "legacy/isaac_ros2_episode_pipeline/2.scene_episode_generator.py"
        ),
        "mode": mode,
        "reset_kind": "full_isaac_pegasus_px4_restart",
        "observed_reset_pose": [reset[0], reset[1], 0.0],
        "start": list(START_POS),
        "target_marker": list(TARGET_POS),
        "goal": [TARGET_POS[0], TARGET_POS[1], FLIGHT_ALTITUDE_M],
        "obstacle_count": len(obstacles),
        "normal_obstacle_count": NUM_OBSTACLES,
        "direct_path_blocker_count": direct_blocker_count,
        "obstacles": obstacles,
        "walls": _collision_walls(),
        "wall_collision_bounds": {
            "east": [X_MIN, X_MAX],
            "north": [Y_MIN, Y_MAX],
        },
        "lighting": LIGHTING_CONTRACT,
        "placement_contract": {
            "area": {"x": [X_MIN, X_MAX], "y": [Y_MIN, Y_MAX]},
            "radius_basis_width_m": [
                RADIUS_BASIS_WIDTH_MIN, RADIUS_BASIS_WIDTH_MAX
            ],
            "radius_basis_depth_m": [
                RADIUS_BASIS_DEPTH_MIN, RADIUS_BASIS_DEPTH_MAX
            ],
            "height_m": [CYLINDER_HEIGHT_MIN, CYLINDER_HEIGHT_MAX],
            "yaw_deg": [OBSTACLE_YAW_MIN_DEG, OBSTACLE_YAW_MAX_DEG],
            "minimum_gap_m": MIN_OBSTACLE_GAP,
            "disk_radius_m": DISK_RADIUS,
            "disk_safe_margin_m": DISK_SAFE_MARGIN,
            "start_clear_radius_m": START_CLEAR_RADIUS,
            "target_clear_radius_m": TARGET_CLEAR_RADIUS,
            "guarantee_direct_path_blockers": (
                GUARANTEE_DIRECT_PATH_BLOCKERS
            ),
            "direct_path_blocker_count": DIRECT_PATH_BLOCKER_COUNT,
            "direct_path_blocker_t_ranges": [
                list(values) for values in DIRECT_PATH_BLOCKER_T_RANGES
            ],
            "direct_path_blocker_lateral_jitter_m": (
                DIRECT_PATH_BLOCKER_LATERAL_JITTER_M
            ),
            "direct_path_blocker_height_min_m": (
                DIRECT_PATH_BLOCKER_HEIGHT_MIN
            ),
        },
    }


# Isaac Sim and ROS 2 are optional for offline scene generation and validation.
try:
    import builtins
    import importlib.util
    from io import BytesIO
    import json
    import os
    import time

    from geometry_msgs.msg import PoseStamped
    import omni.kit.app
    import omni.timeline
    import omni.usd
    import rclpy
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics
    from formal_expert_sensor_contract import (
        FORMAL_RGB_PUBLISH_PERIOD_S,
        FPV_RGB_HEIGHT,
        FPV_RGB_WIDTH,
        LEGACY_OBSERVER_RGB_HEIGHT,
        LEGACY_OBSERVER_RGB_PUBLISH_PERIOD_S,
        LEGACY_OBSERVER_RGB_WIDTH,
        TOP_RGB_HEIGHT,
        TOP_RGB_MODE,
        TOP_RGB_PUBLISH_PERIOD_S,
        TOP_RGB_WIDTH,
    )
except ModuleNotFoundError:
    ISAAC_RUNTIME_AVAILABLE = False
else:
    ISAAC_RUNTIME_AVAILABLE = True


if ISAAC_RUNTIME_AVAILABLE:
    SCHEMA = "uav_isaac_runtime/v1"
    BOOTSTRAP_SCENE_ID = "bootstrap_fixed_scene_v1"
    SCENE_REVISION = 1
    PUBLISH_PERIOD_S = 0.05
    CAMERA_PUBLISH_PERIOD_S = FORMAL_RGB_PUBLISH_PERIOD_S
    CAMERA_WIDTH = FPV_RGB_WIDTH
    CAMERA_HEIGHT = FPV_RGB_HEIGHT
    MSG_WEBRTC_VIEWPORT = (
        "[IsaacRuntimeBridge] WebRTC viewport uses the fixed TOP camera."
    )

    def _world_pose(stage, prim_path):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        matrix = omni.usd.get_world_transform_matrix(prim)
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotation().GetQuat()
        imaginary = rotation.GetImaginary()
        values = (
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
            float(rotation.GetReal()),
        )
        return values if all(math.isfinite(value) for value in values) else None


    def _orthographic_aperture(stage, coverage_m):
        """Convert a world-space coverage in metres to USD camera aperture units."""
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
            raise RuntimeError("stage meters per unit must be finite and positive")
        return (
            float(coverage_m)
            / meters_per_unit
            / float(Gf.Camera.APERTURE_UNIT)
        )


    class IsaacRuntimeBridge:
        """Own one update callback and publish actual stage state at 20 Hz."""

        def __init__(self):
            self._stage = omni.usd.get_context().get_stage()
            if self._stage is None:
                raise RuntimeError("Isaac Sim has no active USD stage")
            self._owns_rclpy = not rclpy.ok()
            if self._owns_rclpy:
                rclpy.init(args=None)
            self._node = rclpy.create_node("isaac_runtime_bridge")
            # Load the pure diagnostic helper without importing a ROS overlay.
            helper = SCRIPT_ROOT.parents[1] / (
                "ros2_ws/src/uav_px4_control/uav_px4_control/diagnostics/__init__.py"
            )
            spec = importlib.util.spec_from_file_location("uav_timing", helper)
            timing_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(timing_module)
            self._timing = timing_module.TimingRecorder(self._node)
            self._pose_publisher = self._node.create_publisher(
                PoseStamped, POSE_TOPIC, 10
            )
            self._status_publisher = self._node.create_publisher(
                String, STATUS_TOPIC, 10
            )
            self._camera_enabled = (
                os.environ.get("UAV_FPV_CAMERA", "0") == "1"
            )
            self._formal_expert_sensors_enabled = (
                os.environ.get("UAV_EXPERT_SENSORS", "0") == "1"
            )
            self._expert_sensors_enabled = self._formal_expert_sensors_enabled
            self._observer_resolution = (
                (TOP_RGB_WIDTH, TOP_RGB_HEIGHT)
                if self._formal_expert_sensors_enabled
                else (LEGACY_OBSERVER_RGB_WIDTH, LEGACY_OBSERVER_RGB_HEIGHT)
            )
            self._observer_publish_period_s = (
                TOP_RGB_PUBLISH_PERIOD_S
                if self._formal_expert_sensors_enabled
                else LEGACY_OBSERVER_RGB_PUBLISH_PERIOD_S
            )
            self._observer_mode = (
                TOP_RGB_MODE
                if self._formal_expert_sensors_enabled
                else OBSERVER_MODE.lower()
            )
            self._camera_enabled = (
                self._camera_enabled or self._expert_sensors_enabled
            )
            self._scene_id = BOOTSTRAP_SCENE_ID
            self._scene_revision = SCENE_REVISION
            self._goal = GOAL
            self._obstacles = list(OBSTACLES)
            self._episode_id = ""
            self._random_seed = None
            self._scene_configuration = None
            self._scene_camera_boundary = None
            self._episode_command_error = ""
            self._replay = None
            self._replay_error = ""
            self._camera_publisher = None
            self._observer_camera_publisher = None
            self._depth_publisher = None
            self._camera_transform = None
            self._observer_camera_transform = None
            self._rgb_annotator = None
            self._observer_rgb_annotator = None
            self._depth_annotator = None
            self._render_product = None
            self._observer_render_product = None
            self._last_camera_publish_monotonic = 0.0
            self._last_observer_publish_monotonic = 0.0
            self._last_depth_publish_monotonic = 0.0
            self._camera_frame_count = 0
            self._observer_frame_count = 0
            self._depth_frame_count = 0
            self._camera_error = "disabled"
            self._observer_camera_error = "disabled"
            self._depth_error = "disabled"
            self._fpv_camera_position = None
            self._observer_camera_position = None
            self._observer_viewport_requested = (
                os.environ.get("UAV_OBSERVER_VIEWPORT", "0") == "1"
            )
            self._viewport_source = os.environ.get(
                "UAV_VIEWPORT_SOURCE", "top_rgb"
            ).strip().lower()
            self._observer_viewport_selected = False
            if self._camera_enabled:
                self._setup_camera()
            self._episode_command_subscription = self._node.create_subscription(
                String, EPISODE_COMMAND_TOPIC, self._episode_command_callback, 10
            )
            self._sequence = 0
            self._last_publish_monotonic = 0.0
            self._stopped = False
            self._subscription = (
                omni.kit.app.get_app()
                .get_update_event_stream()
                .create_subscription_to_pop(
                    self._on_update,
                    name="IsaacRuntimeBridgeUpdate",
                )
            )
            print(
                "[IsaacRuntimeBridge] Started: pose/status at 20 Hz, "
                f"FPV camera={'enabled' if self._camera_enabled else 'disabled'}, "
                "expert sensors="
                f"{'enabled' if self._expert_sensors_enabled else 'disabled'}"
            )

        def _setup_camera(self):
            import omni.replicator.core as rep

            existing = self._stage.GetPrimAtPath(CAMERA_PATH)
            if existing and existing.IsValid():
                self._stage.RemovePrim(CAMERA_PATH)
            camera = UsdGeom.Camera.Define(self._stage, CAMERA_PATH)
            camera.GetFocalLengthAttr().Set(FPV_FOCAL_LENGTH)
            camera.GetHorizontalApertureAttr().Set(FPV_HORIZONTAL_APERTURE)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(*CAMERA_CLIPPING_RANGE))
            self._camera_transform = UsdGeom.Xformable(
                camera.GetPrim()
            ).AddTransformOp()
            self._render_product = rep.create.render_product(
                CAMERA_PATH, (CAMERA_WIDTH, CAMERA_HEIGHT)
            )
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._rgb_annotator.attach([self._render_product])
            self._camera_publisher = self._node.create_publisher(
                CompressedImage, CAMERA_TOPIC, 10
            )
            self._camera_error = "warming"
            if self._expert_sensors_enabled:
                existing_observer = self._stage.GetPrimAtPath(
                    OBSERVER_CAMERA_PATH
                )
                if existing_observer and existing_observer.IsValid():
                    self._stage.RemovePrim(OBSERVER_CAMERA_PATH)
                observer_camera = UsdGeom.Camera.Define(
                    self._stage, OBSERVER_CAMERA_PATH
                )
                if self._formal_expert_sensors_enabled:
                    observer_camera.GetProjectionAttr().Set(
                        UsdGeom.Tokens.orthographic
                    )
                    observer_camera.GetHorizontalApertureAttr().Set(
                        _orthographic_aperture(
                            self._stage, FORMAL_OBSERVER_COVERAGE_M[0]
                        )
                    )
                    observer_camera.GetVerticalApertureAttr().Set(
                        _orthographic_aperture(
                            self._stage, FORMAL_OBSERVER_COVERAGE_M[1]
                        )
                    )
                else:
                    observer_camera.GetProjectionAttr().Set(
                        UsdGeom.Tokens.perspective
                    )
                    observer_camera.GetFocalLengthAttr().Set(
                        OBSERVER_FOCAL_LENGTH
                    )
                    observer_camera.GetHorizontalApertureAttr().Set(
                        OBSERVER_HORIZONTAL_APERTURE
                    )
                observer_camera.GetClippingRangeAttr().Set(
                    Gf.Vec2f(*CAMERA_CLIPPING_RANGE)
                )
                self._observer_camera_transform = UsdGeom.Xformable(
                    observer_camera.GetPrim()
                ).AddTransformOp()
                self._observer_render_product = rep.create.render_product(
                    OBSERVER_CAMERA_PATH, self._observer_resolution
                )
                self._observer_rgb_annotator = (
                    rep.AnnotatorRegistry.get_annotator("rgb")
                )
                self._observer_rgb_annotator.attach([
                    self._observer_render_product
                ])
                self._depth_annotator = rep.AnnotatorRegistry.get_annotator(
                    "distance_to_camera"
                )
                self._depth_annotator.attach([self._render_product])
                self._observer_camera_publisher = self._node.create_publisher(
                    CompressedImage, OBSERVER_CAMERA_TOPIC, 10
                )
                self._depth_publisher = self._node.create_publisher(
                    CompressedImage, DEPTH_TOPIC, 10
                )
                self._observer_camera_error = "warming"
                self._depth_error = "warming"
                if self._observer_viewport_requested:
                    self._select_observer_viewport()
            print(
                f"[IsaacRuntimeBridge] FPV render product: "
                f"{CAMERA_WIDTH}x{CAMERA_HEIGHT} JPEG quality {JPEG_QUALITY}"
            )

        def _select_observer_viewport(self):
            """Show the fixed observer camera without changing its render product."""
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is None:
                return False
            camera_path = (
                OBSERVER_CAMERA_PATH
                if self._viewport_source in {"top", "top_rgb"}
                else CAMERA_PATH
            )
            viewport.set_active_camera(camera_path)
            if not self._observer_viewport_selected:
                print(MSG_WEBRTC_VIEWPORT)
            self._observer_viewport_selected = True
            return True

        def _episode_command_callback(self, message):
            """Apply one seeded scene only while the vehicle is safely landed."""
            try:
                command = json.loads(message.data)
                if command.get("command") == "replay_trace":
                    self._start_trace_replay(command)
                    return
                if command.get("command") != "prepare_episode":
                    raise ValueError("unsupported episode command")
                episode_id = str(command["episode_id"])
                random_seed = int(command["random_seed"])
                mode = str(command.get("mode", "normal"))
                if (
                    self._episode_id == episode_id
                    and self._random_seed == random_seed
                    and isinstance(self._scene_configuration, dict)
                    and self._scene_configuration.get("mode") == mode
                ):
                    self._episode_command_error = ""
                    return
                pose = _world_pose(self._stage, VEHICLE_BODY_PATH)
                if pose is None or pose[2] > 0.25:
                    raise RuntimeError("vehicle must be landed before scene reset")
                scene = generate_episode_scene(
                    episode_id,
                    random_seed,
                    pose[0],
                    pose[1],
                    mode,
                )
                self._apply_scene(scene)
                self._scene_revision += 1
                self._scene_id = (
                    f"expert_{scene['episode_id']}_seed_{scene['random_seed']}"
                )
                self._episode_id = scene["episode_id"]
                self._random_seed = scene["random_seed"]
                self._goal = tuple(scene["goal"])
                self._obstacles = list(scene["obstacles"])
                self._scene_configuration = scene
                self._scene_camera_boundary = {
                    "fpv_rgb_frame_count": self._camera_frame_count,
                    "observer_rgb_frame_count": self._observer_frame_count,
                    "fpv_depth_frame_count": self._depth_frame_count,
                }
                self._episode_command_error = ""
                print(
                    f"[IsaacRuntimeBridge] Prepared {self._scene_id} "
                    f"revision={self._scene_revision} obstacles={len(self._obstacles)}"
                )
            except Exception as error:
                self._episode_command_error = f"{type(error).__name__}: {error}"
                print(f"[IsaacRuntimeBridge][ERROR] {self._episode_command_error}")

        def _start_trace_replay(self, command):
            """Animate an artifact trace as a visual-only ghost vehicle."""
            trace_path = Path(str(command["trace_path"])).expanduser().resolve()
            speed = float(command.get("playback_speed", 1.0))
            if not math.isfinite(speed) or speed <= 0.0:
                raise ValueError("playback_speed must be finite and positive")
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            if payload.get("schema") != "uav_bc_flight_trace/v1":
                raise ValueError("trace schema must be uav_bc_flight_trace/v1")
            samples = payload.get("samples")
            if not isinstance(samples, list) or len(samples) < 2:
                raise ValueError("trace must contain at least two samples")
            previous_time = -1.0
            normalized = []
            for source in samples:
                values = tuple(float(source[key]) for key in (
                    "time_s", "north_m", "east_m", "down_m", "yaw_rad"
                ))
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("trace samples must be finite")
                if values[0] < previous_time:
                    raise ValueError("trace sample times must be nondecreasing")
                previous_time = values[0]
                normalized.append(values)
            self._create_replay_prims(normalized)
            self._replay = {
                "trace_path": str(trace_path), "speed": speed,
                "samples": normalized, "started_monotonic": time.monotonic(),
                "state": "running", "duration_s": normalized[-1][0],
            }
            self._replay_error = ""
            self._episode_command_error = ""
            print(f"[IsaacRuntimeBridge] Replaying {trace_path} at {speed}x")

        def _create_replay_prims(self, samples):
            if self._stage.GetPrimAtPath(REPLAY_ROOT).IsValid():
                self._stage.RemovePrim(REPLAY_ROOT)
            UsdGeom.Xform.Define(self._stage, REPLAY_ROOT)
            ghost = UsdGeom.Capsule.Define(self._stage, f"{REPLAY_ROOT}/GhostUav")
            ghost.CreateRadiusAttr(0.18)
            ghost.CreateHeightAttr(0.20)
            ghost.CreateAxisAttr(UsdGeom.Tokens.z)
            self._set_display_color(ghost.GetPrim(), (1.0, 0.20, 0.05))
            self._replay_ghost_transform = UsdGeom.Xformable(
                ghost.GetPrim()
            ).AddTransformOp()
            line = UsdGeom.BasisCurves.Define(self._stage, f"{REPLAY_ROOT}/Path")
            line.CreateTypeAttr(UsdGeom.Tokens.linear)
            line.CreateCurveVertexCountsAttr([len(samples)])
            line.CreatePointsAttr([
                Gf.Vec3f(east, north, -down + 0.03)
                for _, north, east, down, _ in samples
            ])
            line.CreateWidthsAttr([0.035] * len(samples))
            self._set_display_color(line.GetPrim(), (1.0, 0.65, 0.05))

        def _update_replay(self, now_monotonic):
            if self._replay is None or self._replay["state"] != "running":
                return
            elapsed = now_monotonic - self._replay["started_monotonic"]
            replay_time = elapsed * self._replay["speed"]
            samples = self._replay["samples"]
            if replay_time >= samples[-1][0]:
                time_s, north, east, down, yaw = samples[-1]
                self._replay["state"] = "completed"
            else:
                right_index = next(index for index, sample in enumerate(samples)
                                   if sample[0] >= replay_time)
                left = samples[max(0, right_index - 1)]
                right = samples[right_index]
                fraction = 0.0 if right[0] == left[0] else (
                    (replay_time - left[0]) / (right[0] - left[0])
                )
                time_s = replay_time
                north, east, down, yaw = tuple(
                    left[index] + fraction * (right[index] - left[index])
                    for index in range(1, 5)
                )
            rotation = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), math.degrees(yaw))
            transform = Gf.Matrix4d(1.0)
            transform.SetRotate(rotation)
            transform.SetTranslateOnly(Gf.Vec3d(east, north, -down))
            self._replay_ghost_transform.Set(transform)

        def _apply_scene(self, scene):
            if self._stage.GetPrimAtPath(BOOTSTRAP_SCENE_ROOT).IsValid():
                self._stage.RemovePrim(BOOTSTRAP_SCENE_ROOT)
            if self._stage.GetPrimAtPath(SCENE_ROOT).IsValid():
                self._stage.RemovePrim(SCENE_ROOT)
            root = UsdGeom.Xform.Define(self._stage, SCENE_ROOT)
            root.GetPrim().SetCustomDataByKey("episode:id", scene["episode_id"])
            root.GetPrim().SetCustomDataByKey("episode:seed", scene["random_seed"])
            root.GetPrim().SetCustomDataByKey(
                "episode:generator", scene["generator"]
            )
            materials = create_scene_materials(self._stage, SCENE_ROOT)
            # ------------------------------------------------------------
            # Plain visual floor
            # ------------------------------------------------------------
            # 地板方形墊子的參數(僅外型，沒有碰撞)
            floor = self._create_box(
                f"{SCENE_ROOT}/PlainFloor",
                (100.0, 100.0, 0.01), # 地板尺寸
                (0.0, 0.0, 0.01), # 位置
                FLOOR_COLOR,
                collision=False,
            )
            bind_material(floor, materials["floor"])
            self._create_episode_lighting(scene["lighting"])
            UsdGeom.Xform.Define(self._stage, f"{SCENE_ROOT}/Walls")
            for source in scene["walls"]:
                wall = self._create_box(
                    f"{SCENE_ROOT}/Walls/{source['name']}",
                    source["size"],
                    source["position"],
                    WALL_COLOR,
                    collision=True,
                )
                bind_material(wall, materials["wall"])
                wall.SetCustomDataByKey("episode:shape", "box")
                wall.SetCustomDataByKey("episode:collision", True)
            UsdGeom.Xform.Define(
                self._stage,
                f"{SCENE_ROOT}/Obstacles",
            )

            for index, source in enumerate(
                scene["obstacles"],
                start=1,
            ):
                obstacle = self._create_cylinder_obstacle(
                    source,
                    index,
                    materials["obstacle"],
                )

                obstacle.SetCustomDataByKey(
                    "episode:shape",
                    "cylinder",
                )
                obstacle.SetCustomDataByKey(
                    "episode:radius",
                    float(source["radius"]),
                )
                obstacle.SetCustomDataByKey(
                    "episode:height",
                    float(source["height"]),
                )
            start = UsdGeom.Cylinder.Define(
                self._stage, f"{SCENE_ROOT}/Start/StartDisk"
            )
            start.CreateRadiusAttr(0.5)
            start.CreateHeightAttr(0.05)
            start.AddTranslateOp().Set(Gf.Vec3d(
                scene["start"][0], scene["start"][1], 0.025
            ))
            start.CreateDisplayColorAttr([Gf.Vec3f(*START_MARKER_COLOR)])
            bind_material(start.GetPrim(), materials["start_marker"])
            goal = UsdGeom.Cylinder.Define(
                self._stage, f"{SCENE_ROOT}/Target/TargetDisk"
            )
            goal.CreateRadiusAttr(0.5)
            goal.CreateHeightAttr(0.05)
            goal.AddTranslateOp().Set(Gf.Vec3d(
                scene["target_marker"][0], scene["target_marker"][1], 0.025
            ))
            goal.CreateDisplayColorAttr([Gf.Vec3f(*GOAL_MARKER_COLOR)])
            bind_material(goal.GetPrim(), materials["goal_marker"])

        def apply_gui_preview(self, settings):
            """Apply first-phase GUI settings to the active preview only."""
            color = tuple(float(value) for value in settings["obstacle_color"])
            material_shader = self._stage.GetPrimAtPath(
                f"{SCENE_ROOT}/Materials/{OBSTACLE_MATERIAL_NAME}/Shader"
            )
            if material_shader.IsValid():
                material_shader.GetAttribute("inputs:diffuseColor").Set(
                    Gf.Vec3f(*color)
                )
            dome = self._stage.GetPrimAtPath(f"{SCENE_ROOT}/Lights/Dome")
            if dome.IsValid(): dome.GetAttribute("inputs:intensity").Set(float(settings["dome_intensity"]))
            for source in self._obstacles:
                prim = self._stage.GetPrimAtPath(f"{SCENE_ROOT}/Obstacles/{source['name']}")
                if not prim.IsValid(): continue
                random_item = source.get("placement_mode") == "random"
                UsdGeom.Imageable(prim).MakeVisible() if not random_item or settings["random_obstacles_enabled"] else UsdGeom.Imageable(prim).MakeInvisible()
                self._set_display_color(prim, color)
                if source.get("placement_mode") == "guaranteed_direct_path_blocker":
                    height = float(settings["blocker_height"])
                    UsdGeom.Cylinder(prim).GetHeightAttr().Set(height)
                    prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(source["x"], source["y"], height * 0.5))

        @staticmethod
        def _set_prim_transform(prim, position, rotation_deg=None, scale=None):
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
            if rotation_deg is not None:
                xformable.AddRotateXYZOp().Set(
                    Gf.Vec3f(*map(float, rotation_deg))
                )
            if scale is not None:
                xformable.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))

        @staticmethod
        def _set_display_color(prim, color):
            UsdGeom.Gprim(prim).CreateDisplayColorAttr([
                Gf.Vec3f(*map(float, color))
            ])

        def _create_box(self, path, size, position, color, collision=False):
            cube = UsdGeom.Cube.Define(self._stage, path)
            cube.CreateSizeAttr(1.0)
            prim = cube.GetPrim()
            self._set_prim_transform(prim, position, scale=size)
            self._set_display_color(prim, color)
            if collision:
                UsdPhysics.CollisionAPI.Apply(prim)
            return prim

        def _create_cylinder_obstacle(self, source, index, material):
            name = str(source.get("name") or f"Obstacle_{index:03d}")
            path = f"{SCENE_ROOT}/Obstacles/{name}"
            cylinder = UsdGeom.Cylinder.Define(self._stage, path)
            cylinder.CreateRadiusAttr(float(source["radius"]))
            cylinder.CreateHeightAttr(float(source["height"]))
            prim = cylinder.GetPrim()
            self._set_prim_transform(
                prim,
                (source["x"], source["y"], source["z"]),
            )
            self._set_display_color(prim, OBSTACLE_COLOR)
            bind_material(prim, material)
            if bool(source["collision"]):
                UsdPhysics.CollisionAPI.Apply(prim)
            return prim

        def _create_episode_lighting(self, lighting):
            light_root = f"{SCENE_ROOT}/Lights"
            UsdGeom.Xform.Define(self._stage, light_root)
            dome_spec = lighting["dome"]
            dome = UsdLux.DomeLight.Define(self._stage, f"{light_root}/Dome")
            dome.CreateIntensityAttr(float(dome_spec["intensity"]))
            dome.CreateExposureAttr(float(dome_spec["exposure"]))
            dome.CreateColorAttr(Gf.Vec3f(*map(float, dome_spec["color"])))

        def _update_camera_pose(self):
            if self._camera_transform is None:
                return False
            prim = self._stage.GetPrimAtPath(VEHICLE_BODY_PATH)
            if not prim or not prim.IsValid():
                return False
            matrix = omni.usd.get_world_transform_matrix(prim)
            position = matrix.ExtractTranslation()
            forward = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
            forward = Gf.Vec3d(forward[0], forward[1], 0.0)
            length = math.hypot(float(forward[0]), float(forward[1]))
            if length <= 1e-6:
                return False
            direction = Gf.Vec3d(*(float(value) / length for value in forward))
            fpv_eye = Gf.Vec3d(
                position[0] + FPV_FORWARD_OFFSET_M * direction[0],
                position[1] + FPV_FORWARD_OFFSET_M * direction[1],
                position[2] + FPV_HEIGHT_M,
            )
            fpv_target = Gf.Vec3d(
                position[0] + FPV_LOOK_AHEAD_M * direction[0],
                position[1] + FPV_LOOK_AHEAD_M * direction[1],
                position[2] + FPV_LOOK_DOWN_M,
            )
            # FPV is a rigid body mount. World-space interpolation makes the eye
            # lag behind the current body pose/yaw while the look target does not,
            # which can put the UAV itself between eye and target during flight.
            self._fpv_camera_position = fpv_eye
            transform = Gf.Matrix4d().SetLookAt(
                self._fpv_camera_position,
                fpv_target,
                Gf.Vec3d(0.0, 0.0, 1.0),
            ).GetInverse()
            self._camera_transform.Set(transform)
            if self._observer_camera_transform is not None:
                if self._formal_expert_sensors_enabled:
                    observer_eye = Gf.Vec3d(*FORMAL_OBSERVER_EYE)
                    observer_target = Gf.Vec3d(*FORMAL_OBSERVER_TARGET)
                    observer_up = Gf.Vec3d(*FORMAL_OBSERVER_UP)
                    self._observer_camera_position = observer_eye
                elif OBSERVER_MODE == "TOP":
                    observer_eye = Gf.Vec3d(
                        position[0],
                        position[1],
                        position[2] + OBSERVER_TOP_HEIGHT_M,
                    )
                    observer_target = Gf.Vec3d(
                        position[0],
                        position[1],
                        position[2] + OBSERVER_TOP_LOOK_AT_HEIGHT_M,
                    )
                    observer_up = Gf.Vec3d(0.0, 1.0, 0.0)
                else:
                    right = Gf.Vec3d(direction[1], -direction[0], 0.0)
                    observer_eye = Gf.Vec3d(
                        position[0] - direction[0] * OBSERVER_BACK_DISTANCE_M
                        + right[0] * OBSERVER_SIDE_OFFSET_M,
                        position[1] - direction[1] * OBSERVER_BACK_DISTANCE_M
                        + right[1] * OBSERVER_SIDE_OFFSET_M,
                        position[2] + OBSERVER_HEIGHT_M,
                    )
                    observer_target = Gf.Vec3d(
                        position[0] + direction[0] * OBSERVER_LOOK_AHEAD_M,
                        position[1] + direction[1] * OBSERVER_LOOK_AHEAD_M,
                        position[2] + OBSERVER_LOOK_AT_HEIGHT_M,
                    )
                    observer_up = Gf.Vec3d(0.0, 0.0, 1.0)
                if not self._formal_expert_sensors_enabled:
                    self._observer_camera_position = self._smooth_position(
                        self._observer_camera_position, observer_eye
                    )
                observer_transform = Gf.Matrix4d().SetLookAt(
                    self._observer_camera_position,
                    observer_target,
                    observer_up,
                ).GetInverse()
                self._observer_camera_transform.Set(observer_transform)
            return True

        @staticmethod
        def _smooth_position(current, target):
            if current is None:
                return target
            return Gf.Vec3d(*(
                current[index] * (1.0 - CAMERA_SMOOTHING)
                + target[index] * CAMERA_SMOOTHING
                for index in range(3)
            ))

        @staticmethod
        def _jpeg_message(
            data, stamp, frame_id, expected_size=(CAMERA_WIDTH, CAMERA_HEIGHT)
        ):
            import numpy as np
            from PIL import Image

            if isinstance(data, dict):
                data = data.get("data")
            if data is None or getattr(data, "size", 0) == 0:
                raise RuntimeError("RGB annotator has no frame")
            rgb = np.asarray(data)[..., :3]
            width, height = expected_size
            if rgb.shape != (height, width, 3):
                raise RuntimeError(f"unexpected RGB shape {rgb.shape}")
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            stream = BytesIO()
            Image.fromarray(rgb, mode="RGB").save(
                stream, format="JPEG", quality=JPEG_QUALITY
            )
            message = CompressedImage()
            message.header.stamp = stamp
            message.header.frame_id = frame_id
            message.format = "jpeg; rgb8"
            message.data = stream.getvalue()
            return message

        def _publish_camera(self, stamp, now_monotonic):
            if (
                not self._camera_enabled
                or now_monotonic - self._last_camera_publish_monotonic
                < CAMERA_PUBLISH_PERIOD_S
            ):
                return
            self._last_camera_publish_monotonic = now_monotonic
            try:
                pose_started_ns = time.monotonic_ns()
                if not self._update_camera_pose():
                    raise RuntimeError("vehicle pose unavailable")
                pose_updated_ns = time.monotonic_ns()
                data = self._rgb_annotator.get_data()
                read_completed_ns = time.monotonic_ns()
                message = self._jpeg_message(
                    data, stamp, "isaac_fpv_optical"
                )
                encoded_ns = time.monotonic_ns()
                frame_sequence = self._camera_frame_count + 1
                self._timing.record(
                    "image_read", topic=CAMERA_TOPIC, frame_sequence=frame_sequence,
                    sequence_kind="publication_not_render_frame",
                    pose_update_started_ns=pose_started_ns,
                    pose_updated_ns=pose_updated_ns,
                    read_completed_ns=read_completed_ns, encoded_ns=encoded_ns,
                    read_ms=(read_completed_ns - pose_updated_ns) / 1e6,
                    encode_ms=(encoded_ns - read_completed_ns) / 1e6,
                    read_encode_ms=(encoded_ns - pose_updated_ns) / 1e6,
                    capture_time_known=False, render_completed_ns=None,
                    render_frame_id=None, pose_applies_to_read_frame=None,
                )
                self._timing.publish(
                    CAMERA_TOPIC, message, frame_sequence=frame_sequence,
                    read_started_ns=pose_updated_ns, encoded_ns=encoded_ns,
                    capture_time_known=False,
                )
                self._camera_publisher.publish(message)
                self._camera_frame_count += 1
                self._camera_error = ""
            except Exception as error:
                self._camera_error = f"{type(error).__name__}: {error}"

            if (
                self._expert_sensors_enabled
                and now_monotonic - self._last_observer_publish_monotonic
                >= self._observer_publish_period_s
            ):
                self._last_observer_publish_monotonic = now_monotonic
                try:
                    started_ns = time.monotonic_ns()
                    message = self._jpeg_message(
                        self._observer_rgb_annotator.get_data(),
                        stamp,
                        "isaac_observer_optical",
                        self._observer_resolution,
                    )
                    self._timing.record(
                        "image_read", topic=OBSERVER_CAMERA_TOPIC,
                        read_encode_ms=(time.monotonic_ns() - started_ns) / 1e6,
                        capture_time_known=False,
                    )
                    self._timing.publish(OBSERVER_CAMERA_TOPIC, message)
                    self._observer_camera_publisher.publish(message)
                    self._observer_frame_count += 1
                    self._observer_camera_error = ""
                except Exception as error:
                    self._observer_camera_error = (
                        f"{type(error).__name__}: {error}"
                    )

            if (
                self._expert_sensors_enabled
                and now_monotonic - self._last_depth_publish_monotonic
                >= DEPTH_PUBLISH_PERIOD_S
            ):
                self._last_depth_publish_monotonic = now_monotonic
                try:
                    import numpy as np
                    from PIL import Image

                    depth = self._depth_annotator.get_data()
                    if isinstance(depth, dict):
                        depth = depth.get("data")
                    depth = np.asarray(depth, dtype=np.float32).squeeze()
                    if depth.shape != (CAMERA_HEIGHT, CAMERA_WIDTH):
                        raise RuntimeError(f"unexpected depth shape {depth.shape}")
                    valid = np.isfinite(depth) & (depth >= DEPTH_MIN_M)
                    depth_mm = np.zeros(depth.shape, dtype=np.uint16)
                    depth_mm[valid] = np.rint(
                        np.clip(depth[valid], DEPTH_MIN_M, DEPTH_MAX_M) * 1000.0
                    ).astype(np.uint16)
                    stream = BytesIO()
                    Image.fromarray(depth_mm, mode="I;16").save(stream, format="PNG")
                    message = CompressedImage()
                    message.header.stamp = stamp
                    message.header.frame_id = "isaac_fpv_optical"
                    message.format = (
                        "png; 16UC1; unit=millimeter; range=50..30000; invalid=0"
                    )
                    message.data = stream.getvalue()
                    self._depth_publisher.publish(message)
                    self._depth_frame_count += 1
                    self._depth_error = ""
                except Exception as error:
                    self._depth_error = f"{type(error).__name__}: {error}"

        def _on_update(self, _event):
            if self._stopped:
                return
            if (
                self._observer_viewport_requested
                and not self._observer_viewport_selected
            ):
                self._select_observer_viewport()
            rclpy.spin_once(self._node, timeout_sec=0.0)
            now_monotonic = time.monotonic()
            self._update_replay(now_monotonic)
            if now_monotonic - self._last_publish_monotonic < PUBLISH_PERIOD_S:
                return
            self._last_publish_monotonic = now_monotonic
            timeline_playing = bool(
                omni.timeline.get_timeline_interface().is_playing()
            )
            prim = self._stage.GetPrimAtPath(VEHICLE_BODY_PATH)
            prim_valid = bool(prim and prim.IsValid())
            pose = _world_pose(self._stage, VEHICLE_BODY_PATH) if prim_valid else None
            pose_valid = pose is not None
            if pose_valid:
                message = PoseStamped()
                stamp = self._node.get_clock().now().to_msg()
                message.header.stamp = stamp
                message.header.frame_id = FRAME_ID
                message.pose.position.x = pose[0]
                message.pose.position.y = pose[1]
                message.pose.position.z = pose[2]
                message.pose.orientation.x = pose[3]
                message.pose.orientation.y = pose[4]
                message.pose.orientation.z = pose[5]
                message.pose.orientation.w = pose[6]
                self._pose_publisher.publish(message)
                self._publish_camera(stamp, now_monotonic)
            self._sequence += 1
            status = String()
            status.data = json.dumps({
                "schema": SCHEMA,
                "sequence": self._sequence,
                "scene_id": self._scene_id,
                "scene_revision": self._scene_revision,
                "timeline_playing": timeline_playing,
                "prim_valid": prim_valid,
                "pose_valid": pose_valid,
                "vehicle_prim_path": VEHICLE_BODY_PATH,
                "goal": list(self._goal),
                "obstacles": list(self._obstacles),
                "episode_id": self._episode_id,
                "random_seed": self._random_seed,
                "scene_configuration": self._scene_configuration,
                "scene_camera_boundary": self._scene_camera_boundary,
                "runtime_generation": int(getattr(
                    builtins, "_isaac_uav_runtime_generation", 0
                )),
                "episode_command_error": self._episode_command_error,
                "replay": None if self._replay is None else {
                    "trace_path": self._replay["trace_path"],
                    "speed": self._replay["speed"],
                    "state": self._replay["state"],
                    "duration_s": self._replay["duration_s"],
                    "error": self._replay_error,
                },
                "fpv_rgb_enabled": self._camera_enabled,
                "fpv_rgb_ready": self._camera_frame_count > 0,
                "fpv_rgb_frame_count": self._camera_frame_count,
                "fpv_rgb_error": self._camera_error,
                "observer_rgb_enabled": self._expert_sensors_enabled,
                "observer_rgb_ready": self._observer_frame_count > 0,
                "observer_rgb_frame_count": self._observer_frame_count,
                "observer_rgb_error": self._observer_camera_error,
                "observer_mode": self._observer_mode,
                "fpv_depth_enabled": self._expert_sensors_enabled,
                "fpv_depth_ready": self._depth_frame_count > 0,
                "fpv_depth_frame_count": self._depth_frame_count,
                "fpv_depth_error": self._depth_error,
            }, sort_keys=True, separators=(",", ":"))
            self._status_publisher.publish(status)

        def stop(self):
            """Stop callbacks and ROS resources deterministically."""
            if self._stopped:
                return
            self._stopped = True
            if self._subscription is not None:
                self._subscription.unsubscribe()
                self._subscription = None
            if self._rgb_annotator is not None:
                self._rgb_annotator.detach()
                self._rgb_annotator = None
            if self._observer_rgb_annotator is not None:
                self._observer_rgb_annotator.detach()
                self._observer_rgb_annotator = None
            if self._depth_annotator is not None:
                self._depth_annotator.detach()
                self._depth_annotator = None
            self._timing.close()
            self._node.destroy_node()
            if self._owns_rclpy and rclpy.ok():
                rclpy.shutdown()
            print("[IsaacRuntimeBridge] Stopped.")


    def stop_isaac_runtime_bridge():
        """Public Script Editor cleanup hook."""
        bridge = getattr(builtins, "_isaac_runtime_bridge", None)
        if bridge is not None:
            bridge.stop()
            builtins._isaac_runtime_bridge = None



def start_isaac_runtime_bridge() -> None:
    """Start the embedded bridge when this file is executed by Isaac Sim."""
    if not ISAAC_RUNTIME_AVAILABLE:
        raise RuntimeError("Isaac Sim and ROS 2 dependencies are unavailable")
    stop_isaac_runtime_bridge()
    builtins._isaac_runtime_bridge = IsaacRuntimeBridge()
    builtins.stop_isaac_runtime_bridge = stop_isaac_runtime_bridge
    from GUI import show_environment_gui
    show_environment_gui(builtins._isaac_runtime_bridge)


if __name__ == "__isaac_runtime_bridge__":
    start_isaac_runtime_bridge()
