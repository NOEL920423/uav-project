"""Deterministic canonical cylinder scenes without Isaac dependencies."""

from __future__ import annotations

import math
import random
import re

# 障礙物數量
NUM_OBSTACLES = 10

# 有效的障礙物參數
GUARANTEE_DIRECT_PATH_BLOCKERS = True
DIRECT_PATH_BLOCKER_COUNT = 2
DIRECT_PATH_BLOCKER_T_RANGES = ((0.34, 0.38), (0.62, 0.66))
DIRECT_PATH_BLOCKER_LATERAL_JITTER_M = 0.08
DIRECT_PATH_BLOCKER_HEIGHT_MIN = 3.20

# 障礙物生成範圍
X_MIN = -5.0
X_MAX = 5.0
Y_MIN = -2.0
Y_MAX = 7.0

START_POS = (0.0, 0.0, 0.0)
TARGET_POS = (3.0, 5.0, 0.0)
FLIGHT_ALTITUDE_M = 1.5
DISK_RADIUS = 0.5

# 額外安全距離
DISK_SAFE_MARGIN = 1.0
START_CLEAR_RADIUS = DISK_RADIUS + DISK_SAFE_MARGIN
TARGET_CLEAR_RADIUS = DISK_RADIUS + DISK_SAFE_MARGIN

RADIUS_BASIS_WIDTH_MIN = 0.46
RADIUS_BASIS_WIDTH_MAX = 0.72
RADIUS_BASIS_DEPTH_MIN = 0.46
RADIUS_BASIS_DEPTH_MAX = 0.72

# 障礙物高度最小值和最大值
CYLINDER_HEIGHT_MIN = 2.80
CYLINDER_HEIGHT_MAX = 5.20

BLOCKER_RADIUS_BASIS_WIDTH_MIN = 0.56
BLOCKER_RADIUS_BASIS_DEPTH_MIN = 0.56
# These decoration samples remain in the RNG contract. Removing their draws
# changes later placement draws and therefore seed-to-geometry mapping.
OBSTACLE_YAW_MIN_DEG = -35.0
OBSTACLE_YAW_MAX_DEG = 35.0
SCENE_DECORATION_WINDOW_THICKNESS_M = 0.018
SCENE_DECORATION_WINDOW_HEIGHT_M = 0.16
SCENE_DECORATION_WINDOW_MARGIN_M = 0.08
SCENE_DECORATION_ROOF_HEIGHT_MIN = 0.10
SCENE_DECORATION_ROOF_HEIGHT_MAX = 0.28
SCENE_DECORATION_ANTENNA_HEIGHT_MIN = 0.25
SCENE_DECORATION_ANTENNA_HEIGHT_MAX = 0.55
CYLINDER_COLORS = (
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
MIN_OBSTACLE_GAP = 0.50

MAX_PLACEMENT_ATTEMPTS = 1000
RESET_POSITION_TOLERANCE_M = 0.50
LIGHTING_CONTRACT = {
    "mode": "exact_legacy",
    "root": "/World/GeneratedEpisode/Lights",
    "dome": {
        "intensity": 300.0,
        "exposure": 0.0,
        "color": [0.92, 0.96, 1.0],
    },
    "key": {
        "intensity": 1300.0,
        "angle_deg": 4.0,
        "rotation_deg": [315.0, 0.0, 35.0],
        "color": [1.0, 0.96, 0.90],
    },
    "fill": {
        "intensity": 650.0,
        "angle_deg": 6.0,
        "rotation_deg": [300.0, 0.0, 215.0],
        "color": [0.84, 0.91, 1.0],
    },
}


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
def _random_cylinder_spec(rng: random.Random, blocker: bool = False) -> dict:
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
        "yaw_deg": rng.uniform(OBSTACLE_YAW_MIN_DEG, OBSTACLE_YAW_MAX_DEG),
        "color": list(rng.choice(CYLINDER_COLORS)),
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
    """Run the canonical obstacle rejection sampler and random call order."""
    placed: list[dict] = []
    blocker_count = 0
    if GUARANTEE_DIRECT_PATH_BLOCKERS:
        blocker_count = min(
            DIRECT_PATH_BLOCKER_COUNT,
            NUM_OBSTACLES,
            len(DIRECT_PATH_BLOCKER_T_RANGES),
        )

    start_x, start_y = START_POS[:2]
    target_x, target_y = TARGET_POS[:2]
    line_dx = target_x - start_x
    line_dy = target_y - start_y
    line_length = math.hypot(line_dx, line_dy)
    if blocker_count and line_length < 1e-6:
        raise RuntimeError("cannot place blockers when start and target coincide")
    normal_x = -line_dy / max(line_length, 1e-6)
    normal_y = line_dx / max(line_length, 1e-6)

    for blocker_index in range(blocker_count):
        t_min, t_max = DIRECT_PATH_BLOCKER_T_RANGES[blocker_index]
        for _attempt in range(MAX_PLACEMENT_ATTEMPTS):
            spec = _random_cylinder_spec(rng, blocker=True)
            radius = spec["radius"]
            t = rng.uniform(t_min, t_max)
            lateral_limit = min(
                DIRECT_PATH_BLOCKER_LATERAL_JITTER_M,
                spec["blocker_half_extent"] * 0.70,
            )
            lateral = rng.uniform(-lateral_limit, lateral_limit)
            x = start_x + t * line_dx + lateral * normal_x
            y = start_y + t * line_dy + lateral * normal_y
            inside_bounds = (
                X_MIN + radius <= x <= X_MAX - radius
                and Y_MIN + radius <= y <= Y_MAX - radius
            )
            blocks_direct_path = (
                _point_to_direct_path_distance(x, y)
                <= spec["blocker_half_extent"]
            )
            if not (
                inside_bounds
                and blocks_direct_path
                and _is_valid_obstacle_position(x, y, radius, placed)
            ):
                continue
            spec.update({
                "x": x,
                "y": y,
                "z": 0.5 * spec["height"],
                "placement_mode": "guaranteed_direct_path_blocker",
            })
            placed.append(spec)
            break
        else:
            raise RuntimeError(
                "could not place a canonical guaranteed direct-path blocker"
            )

    for _index in range(NUM_OBSTACLES - blocker_count):
        for _attempt in range(MAX_PLACEMENT_ATTEMPTS):
            spec = _random_cylinder_spec(rng)
            radius = spec["radius"]
            x = rng.uniform(X_MIN + radius, X_MAX - radius)
            y = rng.uniform(Y_MIN + radius, Y_MAX - radius)
            if not _is_valid_obstacle_position(x, y, radius, placed):
                continue
            spec.update({
                "x": x,
                "y": y,
                "z": 0.5 * spec["height"],
                "placement_mode": "random",
            })
            placed.append(spec)
            break
        else:
            raise RuntimeError(
                f"could not place canonical obstacle {len(placed) + 1}"
            )

    physical_blockers = [
        item for item in placed
        if _point_to_direct_path_distance(item["x"], item["y"])
        <= item["blocker_half_extent"]
        and item["height"] >= DIRECT_PATH_BLOCKER_HEIGHT_MIN
    ]
    if len(physical_blockers) < blocker_count:
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
        "color": list(CYLINDER_COLORS[0]),
        "window_on_color": list(SCENE_DECORATION_WINDOW_ON_COLORS[0]),
        "window_off_color": list(SCENE_DECORATION_WINDOW_OFF_COLOR),
        "roof_style": "flat",
        "roof_height": SCENE_DECORATION_ROOF_HEIGHT_MIN,
        "antenna_height": SCENE_DECORATION_ANTENNA_HEIGHT_MIN,
        "collision": True,
        "placement_mode": "phase10b_safe_failure",
        "fixture": "phase10b_safe_failure",
        "hierarchy": ["Body", "Windows", "Roof/Crown"],
    }
    spec["windows"] = _window_contract(spec, random.Random(0))
    return spec


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

    obstacles = _generate_obstacles(random.Random(int(seed)))
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
