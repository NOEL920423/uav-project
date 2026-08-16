"""Deterministic canonical high-rise scenes without Isaac dependencies."""

from __future__ import annotations

import math
import random
import re


NUM_OBSTACLES = 8
GUARANTEE_DIRECT_PATH_BLOCKERS = True
DIRECT_PATH_BLOCKER_COUNT = 2
DIRECT_PATH_BLOCKER_T_RANGES = ((0.34, 0.38), (0.62, 0.66))
DIRECT_PATH_BLOCKER_LATERAL_JITTER_M = 0.08
DIRECT_PATH_BLOCKER_HEIGHT_MIN = 3.20
X_MIN = -2.0
X_MAX = 5.0
Y_MIN = -1.0
Y_MAX = 7.0
START_POS = (0.0, 0.0, 0.0)
TARGET_POS = (3.0, 5.0, 0.0)
FLIGHT_ALTITUDE_M = 1.5
DISK_RADIUS = 0.5
DISK_SAFE_MARGIN = 1.0
START_CLEAR_RADIUS = DISK_RADIUS + DISK_SAFE_MARGIN
TARGET_CLEAR_RADIUS = DISK_RADIUS + DISK_SAFE_MARGIN
BUILDING_WIDTH_MIN = 0.46
BUILDING_WIDTH_MAX = 0.72
BUILDING_DEPTH_MIN = 0.46
BUILDING_DEPTH_MAX = 0.72
BUILDING_HEIGHT_MIN = 2.80
BUILDING_HEIGHT_MAX = 5.20
BLOCKER_BUILDING_WIDTH_MIN = 0.56
BLOCKER_BUILDING_DEPTH_MIN = 0.56
BUILDING_YAW_MIN_DEG = -35.0
BUILDING_YAW_MAX_DEG = 35.0
BUILDING_WINDOW_THICKNESS_M = 0.018
BUILDING_WINDOW_HEIGHT_M = 0.16
BUILDING_WINDOW_MARGIN_M = 0.08
BUILDING_ROOF_HEIGHT_MIN = 0.10
BUILDING_ROOF_HEIGHT_MAX = 0.28
BUILDING_ANTENNA_HEIGHT_MIN = 0.25
BUILDING_ANTENNA_HEIGHT_MAX = 0.55
BUILDING_FACADE_COLORS = (
    (0.12, 0.18, 0.24),
    (0.20, 0.27, 0.31),
    (0.30, 0.31, 0.34),
    (0.27, 0.22, 0.20),
    (0.18, 0.22, 0.30),
)
BUILDING_WINDOW_ON_COLORS = (
    (0.38, 0.72, 1.00),
    (0.62, 0.86, 1.00),
    (1.00, 0.78, 0.36),
)
BUILDING_WINDOW_OFF_COLOR = (0.035, 0.055, 0.075)
BUILDING_ROOF_STYLES = ("flat", "crown", "antenna")
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


def _random_building_spec(rng: random.Random, blocker: bool = False) -> dict:
    width = rng.uniform(
        BLOCKER_BUILDING_WIDTH_MIN if blocker else BUILDING_WIDTH_MIN,
        BUILDING_WIDTH_MAX,
    )
    depth = rng.uniform(
        BLOCKER_BUILDING_DEPTH_MIN if blocker else BUILDING_DEPTH_MIN,
        BUILDING_DEPTH_MAX,
    )
    height = rng.uniform(
        max(BUILDING_HEIGHT_MIN, DIRECT_PATH_BLOCKER_HEIGHT_MIN)
        if blocker else BUILDING_HEIGHT_MIN,
        BUILDING_HEIGHT_MAX,
    )
    return {
        "shape": "high_rise_building",
        "width": width,
        "depth": depth,
        "height": height,
        "radius": 0.5 * math.hypot(width, depth),
        "blocker_half_extent": 0.5 * min(width, depth),
        "yaw_deg": rng.uniform(BUILDING_YAW_MIN_DEG, BUILDING_YAW_MAX_DEG),
        "facade_color": list(rng.choice(BUILDING_FACADE_COLORS)),
        "window_on_color": list(rng.choice(BUILDING_WINDOW_ON_COLORS)),
        "window_off_color": list(BUILDING_WINDOW_OFF_COLOR),
        "roof_style": rng.choice(BUILDING_ROOF_STYLES),
        "roof_height": rng.uniform(
            BUILDING_ROOF_HEIGHT_MIN, BUILDING_ROOF_HEIGHT_MAX
        ),
        "antenna_height": rng.uniform(
            BUILDING_ANTENNA_HEIGHT_MIN, BUILDING_ANTENNA_HEIGHT_MAX
        ),
        "collision": True,
    }


def _window_contract(spec: dict, rng: random.Random) -> dict:
    row_count = max(5, min(11, int(spec["height"] / 0.44)))
    columns_x = max(2, min(3, int(spec["width"] / 0.22)))
    columns_y = max(2, min(3, int(spec["depth"] / 0.22)))
    window_count = row_count * 2 * (columns_x + columns_y)
    return {
        "row_count": row_count,
        "columns_x": columns_x,
        "columns_y": columns_y,
        "height_m": BUILDING_WINDOW_HEIGHT_M,
        "thickness_m": BUILDING_WINDOW_THICKNESS_M,
        "margin_m": BUILDING_WINDOW_MARGIN_M,
        "on_pattern": [rng.random() < 0.72 for _ in range(window_count)],
    }


def _generate_obstacles(rng: random.Random) -> list[dict]:
    """Port the canonical building rejection sampler and random call order."""
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
            spec = _random_building_spec(rng, blocker=True)
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
            spec = _random_building_spec(rng)
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
                f"could not place canonical building {len(placed) + 1}"
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
        spec["name"] = f"Building_{index:03d}"
        spec["windows"] = _window_contract(spec, rng)
        spec["hierarchy"] = ["Body", "Windows", "Roof/Crown"]
        if spec["roof_style"] == "antenna":
            spec["hierarchy"].append("Roof/Antenna")
    return placed


def _blocked_goal_fixture() -> dict:
    radius = 0.85
    side = radius * math.sqrt(2.0)
    spec = {
        "name": "Building_blocked_goal",
        "shape": "high_rise_building",
        "x": TARGET_POS[0],
        "y": TARGET_POS[1],
        "z": 1.5,
        "width": side,
        "depth": side,
        "height": 3.0,
        "radius": radius,
        "blocker_half_extent": 0.5 * side,
        "yaw_deg": 0.0,
        "facade_color": list(BUILDING_FACADE_COLORS[0]),
        "window_on_color": list(BUILDING_WINDOW_ON_COLORS[0]),
        "window_off_color": list(BUILDING_WINDOW_OFF_COLOR),
        "roof_style": "flat",
        "roof_height": BUILDING_ROOF_HEIGHT_MIN,
        "antenna_height": BUILDING_ANTENNA_HEIGHT_MIN,
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
    """Generate the canonical eight-building distribution for one safe reset."""
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
        "generator": "canonical_highrise_scene_generator_v1",
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
            "width_m": [BUILDING_WIDTH_MIN, BUILDING_WIDTH_MAX],
            "depth_m": [BUILDING_DEPTH_MIN, BUILDING_DEPTH_MAX],
            "height_m": [BUILDING_HEIGHT_MIN, BUILDING_HEIGHT_MAX],
            "yaw_deg": [BUILDING_YAW_MIN_DEG, BUILDING_YAW_MAX_DEG],
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
