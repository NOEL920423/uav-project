# scene_episode_generator.py
# Generate one UAV obstacle scene episode in Isaac Sim.
# Features:
# 1. Clear old generated obstacles
# 2. Create new obstacles
# 3. Create red target point
# 4. Set colors
# 5. Set collision
# 6. Record every generated object position to JSON and CSV

import os
import csv
import json
import math
import random
from datetime import datetime

from pxr import UsdGeom, Gf, Sdf, UsdPhysics, UsdLux

try:
    import omni.usd
except ImportError:
    raise RuntimeError(
        "This script must be executed inside Isaac Sim, "
        "for example from Script Editor or with isaacsim.sh --exec."
    )


# ============================================================
# User configuration
# ============================================================

GENERATED_ROOT = "/World/GeneratedEpisode"

# This folder stores episode records.
LOG_DIR = "/home/noel_614420090/uav-project/uav_demo_logs/scene_episodes"

# Random seed.
# Set to None if you want a different scene every time.
RANDOM_SEED = None
# RANDOM_SEED = 42

NUM_OBSTACLES = 8

# Every random episode first places obstacles that physically intersect the
# direct start-to-target corridor. The remaining obstacles are still sampled
# from the full scene, so the layout changes while A* is always forced to plan
# a meaningful detour.
GUARANTEE_DIRECT_PATH_BLOCKERS = True
DIRECT_PATH_BLOCKER_COUNT = 2
DIRECT_PATH_BLOCKER_T_RANGES = ((0.34, 0.38), (0.62, 0.66))
DIRECT_PATH_BLOCKER_LATERAL_JITTER_M = 0.08
# Flight altitude is 2.0 m and the planner permits 0.35 m vertical clearance.
# A guaranteed blocker must therefore be taller than 2.35 m.
DIRECT_PATH_BLOCKER_HEIGHT_MIN = 3.20

# Scene generation area.
# Unit is meter if your Isaac stage uses normal meter scale.
X_MIN = -2.0
X_MAX = 5.0
Y_MIN = -1.0
Y_MAX = 7.0

# Start and target positions.
START_POS = (0.0, 0.0, 0.0)
TARGET_POS = (3.0, 5.0, 0.0)

# Disk settings.
# If the disk size is [1.0, 1.0, 1.0] in diameter sense,
# then radius should be 0.5.
DISK_RADIUS = 0.5
DISK_HEIGHT = 0.05

CREATE_START_MARKER = True
CREATE_TARGET_MARKER = True

START_MARKER_RADIUS = DISK_RADIUS
START_MARKER_HEIGHT = DISK_HEIGHT
START_MARKER_COLOR = (0.0, 0.3, 1.0)
START_MARKER_HAS_COLLISION = False

TARGET_RADIUS = DISK_RADIUS
TARGET_HEIGHT = DISK_HEIGHT
TARGET_COLOR = (1.0, 0.0, 0.0)
TARGET_HAS_COLLISION = False

# Keep obstacle away from start and target disks.
# For the A* baseline, this margin must be larger than the planner safety envelope:
# UAV_SAFETY_RADIUS_M + OBSTACLE_SAFETY_MARGIN_M + MIN_SEGMENT_CLEARANCE_M + buffer.
# With the current A* runner values, 1.0 is a safer default than the older 0.5.
DISK_SAFE_MARGIN = 1.0
START_CLEAR_RADIUS = DISK_RADIUS + DISK_SAFE_MARGIN
TARGET_CLEAR_RADIUS = DISK_RADIUS + DISK_SAFE_MARGIN

# Random high-rise geometry. Dimensions are intentionally compact relative to
# the 7 x 8 m test area, while height is always above the 2 m flight layer.
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

# Obstacle spacing.
# A slightly larger gap makes the A* baseline less likely to create impossible mazes.
MIN_OBSTACLE_GAP = 0.50

# Red target disk is disabled.
# We do not create a red target object anymore.

# Laser spot settings.
CREATE_LASER_SPOT = False

# None means randomly choose one obstacle.
# If you want to fix it on a specific obstacle, use 1, 2, 3...
LASER_SPOT_OBSTACLE_INDEX = None

# The laser spot is a very thin red disk attached to the obstacle surface.
LASER_SPOT_RADIUS = 0.08
LASER_SPOT_THICKNESS = 0.01

# 0.65 means the spot is placed at 65% of the obstacle height.
LASER_SPOT_HEIGHT_RATIO = 0.65

# Push the spot slightly outside the obstacle surface to avoid z-fighting.
LASER_SPOT_SURFACE_OFFSET = 0.015

LASER_SPOT_COLOR = (1.0, 0.0, 0.0)
LASER_SPOT_HAS_COLLISION = False

# Optional small red light near the spot.
CREATE_LASER_SPOT_LIGHT = False
LASER_SPOT_LIGHT_INTENSITY = 250.0
LASER_SPOT_LIGHT_RADIUS = 0.05

# Colors.
COLOR_OBSTACLE = (0.20, 0.24, 0.28)
COLOR_TARGET = (1.0, 0.0, 0.0)
COLOR_ROOT = (0.2, 0.2, 0.2)

# Dataset lighting.  The default environment is intentionally dark, so every
# generated episode owns a neutral dome light plus two broad directional
# lights.  Keeping the lights below GENERATED_ROOT makes cleanup deterministic.
CREATE_EPISODE_LIGHTING = True
DOME_LIGHT_INTENSITY = 300.0
DOME_LIGHT_EXPOSURE = 0.0
DOME_LIGHT_COLOR = (0.92, 0.96, 1.0)
KEY_LIGHT_INTENSITY = 1300.0
KEY_LIGHT_ANGLE_DEG = 4.0
KEY_LIGHT_ROTATION_DEG = (315.0, 0.0, 35.0)
KEY_LIGHT_COLOR = (1.0, 0.96, 0.90)
FILL_LIGHT_INTENSITY = 650.0
FILL_LIGHT_ANGLE_DEG = 6.0
FILL_LIGHT_ROTATION_DEG = (300.0, 0.0, 215.0)
FILL_LIGHT_COLOR = (0.84, 0.91, 1.0)

# If True, also creates a visible start marker.
CREATE_START_MARKER = True
START_MARKER_RADIUS = 0.5
START_MARKER_HEIGHT = 0.05
START_MARKER_COLOR = (0.0, 0.3, 1.0)
START_MARKER_HAS_COLLISION = False

# If you previously created old obstacles directly under /World with these names,
# you may add their prefixes here.
# Be careful: do not put broad paths like "/World/Cylinder".
EXTRA_DELETE_PREFIXES = [
    # "/World/Obstacle_",
    # "/World/RedPoint",
]


# ============================================================
# Basic USD helpers
# ============================================================

def get_stage():
    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage found. Please open your Isaac Sim scene first.")
    return stage


def delete_prim_if_exists(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if prim and prim.IsValid():
        stage.RemovePrim(Sdf.Path(prim_path))


def clear_old_generated_scene(stage):
    delete_prim_if_exists(stage, GENERATED_ROOT)

    for prefix in EXTRA_DELETE_PREFIXES:
        paths_to_delete = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.startswith(prefix):
                paths_to_delete.append(path)

        for path in sorted(paths_to_delete, reverse=True):
            delete_prim_if_exists(stage, path)


def ensure_xform(stage, path):
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        return UsdGeom.Xform(prim)

    return UsdGeom.Xform.Define(stage, Sdf.Path(path))


def set_transform(prim, position, rotation_deg=None, scale=None):
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    if rotation_deg is not None:
        rotate_op = xformable.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(float(rotation_deg[0]), float(rotation_deg[1]), float(rotation_deg[2])))

    if scale is not None:
        scale_op = xformable.AddScaleOp()
        scale_op.Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))


def set_display_color(prim, color_rgb):
    gprim = UsdGeom.Gprim(prim)
    gprim.CreateDisplayColorAttr([Gf.Vec3f(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))])


def apply_collision(prim):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)


def remove_collision_if_exists(prim):
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        prim.RemoveAPI(UsdPhysics.CollisionAPI)


def set_custom_metadata(prim, data):
    for key, value in data.items():
        attr_name = f"episode:{key}"

        if isinstance(value, bool):
            value_type = Sdf.ValueTypeNames.Bool
        elif isinstance(value, int):
            value_type = Sdf.ValueTypeNames.Int
        elif isinstance(value, float):
            value_type = Sdf.ValueTypeNames.Double
        else:
            value_type = Sdf.ValueTypeNames.String
            value = str(value)

        attr = prim.CreateAttribute(attr_name, value_type, custom=True)
        attr.Set(value)


def ensure_physics_scene(stage):
    possible_paths = [
        "/World/physicsScene",
        "/World/PhysicsScene",
        "/physicsScene",
    ]

    for path in possible_paths:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            return str(prim.GetPath())

    physics_scene_path = "/World/physicsScene"
    UsdPhysics.Scene.Define(stage, Sdf.Path(physics_scene_path))
    return physics_scene_path


def create_episode_lighting(stage):
    """Create reproducible, camera-visible lighting for the generated scene."""
    lights_root = f"{GENERATED_ROOT}/Lights"
    ensure_xform(stage, lights_root)

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path(f"{lights_root}/Dome"))
    dome.CreateIntensityAttr(float(DOME_LIGHT_INTENSITY))
    dome.CreateExposureAttr(float(DOME_LIGHT_EXPOSURE))
    dome.CreateColorAttr(Gf.Vec3f(*[float(v) for v in DOME_LIGHT_COLOR]))

    key = UsdLux.DistantLight.Define(stage, Sdf.Path(f"{lights_root}/Key"))
    key.CreateIntensityAttr(float(KEY_LIGHT_INTENSITY))
    key.CreateAngleAttr(float(KEY_LIGHT_ANGLE_DEG))
    key.CreateColorAttr(Gf.Vec3f(*[float(v) for v in KEY_LIGHT_COLOR]))
    set_transform(
        key.GetPrim(),
        position=(0.0, 0.0, 8.0),
        rotation_deg=KEY_LIGHT_ROTATION_DEG,
    )

    fill = UsdLux.DistantLight.Define(stage, Sdf.Path(f"{lights_root}/Fill"))
    fill.CreateIntensityAttr(float(FILL_LIGHT_INTENSITY))
    fill.CreateAngleAttr(float(FILL_LIGHT_ANGLE_DEG))
    fill.CreateColorAttr(Gf.Vec3f(*[float(v) for v in FILL_LIGHT_COLOR]))
    set_transform(
        fill.GetPrim(),
        position=(0.0, 0.0, 8.0),
        rotation_deg=FILL_LIGHT_ROTATION_DEG,
    )

    print(
        "[SceneLighting] Created dataset lights: "
        f"dome={DOME_LIGHT_INTENSITY:.0f}, "
        f"key={KEY_LIGHT_INTENSITY:.0f}, fill={FILL_LIGHT_INTENSITY:.0f}"
    )


# ============================================================
# Object creation
# ============================================================

def create_cylinder(
    stage,
    path,
    radius,
    height,
    position,
    color,
    collision=True,
    object_type="cylinder",
    rotation_deg=None,
):
    cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(path))
    cylinder.CreateRadiusAttr(float(radius))
    cylinder.CreateHeightAttr(float(height))

    prim = cylinder.GetPrim()
    set_transform(prim, position, rotation_deg=rotation_deg)
    set_display_color(prim, color)

    if collision:
        apply_collision(prim)
    else:
        remove_collision_if_exists(prim)

    set_custom_metadata(
        prim,
        {
            "object_type": object_type,
            "radius": float(radius),
            "height": float(height),
            "collision": bool(collision),
        },
    )

    return prim


def create_box(stage, path, size_xyz, position, color, collision=False):
    cube = UsdGeom.Cube.Define(stage, Sdf.Path(path))
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    set_transform(prim, position=position, scale=size_xyz)
    set_display_color(prim, color)
    if collision:
        apply_collision(prim)
    else:
        remove_collision_if_exists(prim)
    return prim


def create_building_windows(
    stage,
    building_path,
    width,
    depth,
    height,
    row_count,
    columns_x,
    columns_y,
    window_on_color,
):
    """Create lightweight four-sided window grids beneath a building Xform."""
    window_root = f"{building_path}/Windows"
    ensure_xform(stage, window_root)
    thickness = float(BUILDING_WINDOW_THICKNESS_M)
    window_height = min(float(BUILDING_WINDOW_HEIGHT_M), height / (row_count + 2))
    usable_height = max(window_height, height - 2.0 * BUILDING_WINDOW_MARGIN_M)
    row_spacing = usable_height / max(1, row_count)
    window_width_x = max(0.07, (width - 2.0 * BUILDING_WINDOW_MARGIN_M) / max(1, columns_x) * 0.62)
    window_width_y = max(0.07, (depth - 2.0 * BUILDING_WINDOW_MARGIN_M) / max(1, columns_y) * 0.62)

    for row in range(row_count):
        z = BUILDING_WINDOW_MARGIN_M + (row + 0.5) * row_spacing

        for column in range(columns_x):
            x = -0.5 * width + (column + 0.5) * width / columns_x
            for face_name, y in (("North", 0.5 * depth + 0.5 * thickness), ("South", -0.5 * depth - 0.5 * thickness)):
                color = window_on_color if random.random() < 0.72 else BUILDING_WINDOW_OFF_COLOR
                create_box(
                    stage,
                    f"{window_root}/{face_name}_R{row:02d}_C{column:02d}",
                    (window_width_x, thickness, window_height),
                    (x, y, z),
                    color,
                    collision=False,
                )

        for column in range(columns_y):
            y = -0.5 * depth + (column + 0.5) * depth / columns_y
            for face_name, x in (("East", 0.5 * width + 0.5 * thickness), ("West", -0.5 * width - 0.5 * thickness)):
                color = window_on_color if random.random() < 0.72 else BUILDING_WINDOW_OFF_COLOR
                create_box(
                    stage,
                    f"{window_root}/{face_name}_R{row:02d}_C{column:02d}",
                    (thickness, window_width_y, window_height),
                    (x, y, z),
                    color,
                    collision=False,
                )


def create_obstacle(
    stage,
    index,
    x,
    y,
    width,
    depth,
    height,
    yaw_deg,
    facade_color,
    window_color,
    roof_style,
    roof_height,
    antenna_height,
):
    """Create a randomized, collidable high-rise building obstacle."""
    name = f"Building_{index:03d}"
    path = f"{GENERATED_ROOT}/Obstacles/{name}"
    root = ensure_xform(stage, path)
    root_prim = root.GetPrim()
    set_transform(root_prim, position=(x, y, 0.0), rotation_deg=(0.0, 0.0, yaw_deg))

    planning_radius = 0.5 * math.hypot(width, depth)
    set_custom_metadata(
        root_prim,
        {
            "object_type": "obstacle",
            "shape": "high_rise_building",
            "width": float(width),
            "depth": float(depth),
            "height": float(height),
            "radius": float(planning_radius),
            "collision": True,
            "roof_style": roof_style,
        },
    )

    create_box(
        stage,
        f"{path}/Body",
        (width, depth, height),
        (0.0, 0.0, 0.5 * height),
        facade_color,
        collision=True,
    )

    row_count = max(5, min(11, int(height / 0.44)))
    columns_x = max(2, min(3, int(width / 0.22)))
    columns_y = max(2, min(3, int(depth / 0.22)))
    create_building_windows(
        stage=stage,
        building_path=path,
        width=width,
        depth=depth,
        height=height,
        row_count=row_count,
        columns_x=columns_x,
        columns_y=columns_y,
        window_on_color=window_color,
    )

    roof_color = tuple(max(0.02, float(value) * 0.65) for value in facade_color)
    crown_width = width * (0.82 if roof_style == "flat" else 0.58)
    crown_depth = depth * (0.82 if roof_style == "flat" else 0.58)
    ensure_xform(stage, f"{path}/Roof")
    create_box(
        stage,
        f"{path}/Roof/Crown",
        (crown_width, crown_depth, roof_height),
        (0.0, 0.0, height + 0.5 * roof_height),
        roof_color,
        collision=False,
    )

    if roof_style == "antenna":
        create_cylinder(
            stage=stage,
            path=f"{path}/Roof/Antenna",
            radius=max(0.018, min(width, depth) * 0.045),
            height=antenna_height,
            position=(0.0, 0.0, height + roof_height + 0.5 * antenna_height),
            color=(0.12, 0.12, 0.14),
            collision=False,
            object_type="building_antenna",
        )

    record = {
        "name": name,
        "path": path,
        "type": "obstacle",
        "shape": "high_rise_building",
        "x": float(x),
        "y": float(y),
        "z": float(0.5 * height),
        "radius": float(planning_radius),
        "width": float(width),
        "depth": float(depth),
        "height": float(height),
        "yaw_deg": float(yaw_deg),
        "roof_style": str(roof_style),
        "collision": True,
        "color_r": float(facade_color[0]),
        "color_g": float(facade_color[1]),
        "color_b": float(facade_color[2]),
    }

    return root_prim, record


def create_target_marker(stage):
    path = f"{GENERATED_ROOT}/Target/TargetDisk"
    x, y, _ = TARGET_POS
    z = TARGET_HEIGHT / 2.0

    prim = create_cylinder(
        stage=stage,
        path=path,
        radius=TARGET_RADIUS,
        height=TARGET_HEIGHT,
        position=(x, y, z),
        color=TARGET_COLOR,
        collision=TARGET_HAS_COLLISION,
        object_type="target",
    )

    record = {
        "name": "TargetDisk",
        "path": path,
        "type": "target",
        "shape": "cylinder",
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "radius": float(TARGET_RADIUS),
        "height": float(TARGET_HEIGHT),
        "collision": bool(TARGET_HAS_COLLISION),
        "color_r": float(TARGET_COLOR[0]),
        "color_g": float(TARGET_COLOR[1]),
        "color_b": float(TARGET_COLOR[2]),
    }

    return prim, record


def create_start_marker(stage):
    path = f"{GENERATED_ROOT}/Start/StartDisk"
    x, y, _ = START_POS
    z = START_MARKER_HEIGHT / 2.0

    prim = create_cylinder(
        stage=stage,
        path=path,
        radius=START_MARKER_RADIUS,
        height=START_MARKER_HEIGHT,
        position=(x, y, z),
        color=START_MARKER_COLOR,
        collision=START_MARKER_HAS_COLLISION,
        object_type="start",
    )

    record = {
        "name": "StartDisk",
        "path": path,
        "type": "start",
        "shape": "cylinder",
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "radius": float(START_MARKER_RADIUS),
        "height": float(START_MARKER_HEIGHT),
        "collision": bool(START_MARKER_HAS_COLLISION),
        "color_r": float(START_MARKER_COLOR[0]),
        "color_g": float(START_MARKER_COLOR[1]),
        "color_b": float(START_MARKER_COLOR[2]),
    }

    return prim, record


def select_laser_obstacle(obstacle_records):
    if not obstacle_records:
        return None

    if LASER_SPOT_OBSTACLE_INDEX is None:
        return random.choice(obstacle_records)

    index = int(LASER_SPOT_OBSTACLE_INDEX) - 1
    index = max(0, min(index, len(obstacle_records) - 1))
    return obstacle_records[index]


def create_laser_spot_on_obstacle(stage, obstacle_record):
    obstacle_x = float(obstacle_record["x"])
    obstacle_y = float(obstacle_record["y"])
    obstacle_radius = float(obstacle_record["radius"])
    obstacle_height = float(obstacle_record["height"])

    # Make the laser spot face the UAV start position.
    # This is useful because the camera is more likely to see the red spot.
    direction_x = START_POS[0] - obstacle_x
    direction_y = START_POS[1] - obstacle_y
    direction_length = math.sqrt(direction_x ** 2 + direction_y ** 2)

    if direction_length < 1e-6:
        normal_x = 1.0
        normal_y = 0.0
    else:
        normal_x = direction_x / direction_length
        normal_y = direction_y / direction_length

    laser_x = obstacle_x + normal_x * (
        obstacle_radius + LASER_SPOT_THICKNESS * 0.5 + LASER_SPOT_SURFACE_OFFSET
    )
    laser_y = obstacle_y + normal_y * (
        obstacle_radius + LASER_SPOT_THICKNESS * 0.5 + LASER_SPOT_SURFACE_OFFSET
    )
    laser_z = obstacle_height * LASER_SPOT_HEIGHT_RATIO

    # The default cylinder axis is along Z.
    # Rotate it so the thin disk sticks to the side surface of the cylinder.
    angle_deg = math.degrees(math.atan2(normal_y, normal_x))
    rotation_deg = (0.0, 90.0, angle_deg)

    path = f"{GENERATED_ROOT}/LaserSpot/LaserSpot"

    prim = create_cylinder(
        stage=stage,
        path=path,
        radius=LASER_SPOT_RADIUS,
        height=LASER_SPOT_THICKNESS,
        position=(laser_x, laser_y, laser_z),
        color=LASER_SPOT_COLOR,
        collision=LASER_SPOT_HAS_COLLISION,
        object_type="laser_spot",
        rotation_deg=rotation_deg,
    )

    if CREATE_LASER_SPOT_LIGHT:
        light_path = f"{GENERATED_ROOT}/LaserSpot/LaserSpotLight"
        light = UsdLux.SphereLight.Define(stage, Sdf.Path(light_path))
        light.CreateRadiusAttr(float(LASER_SPOT_LIGHT_RADIUS))
        light.CreateIntensityAttr(float(LASER_SPOT_LIGHT_INTENSITY))
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.0, 0.0))

        light_prim = light.GetPrim()
        set_transform(light_prim, (laser_x, laser_y, laser_z))

    set_custom_metadata(
        prim,
        {
            "object_type": "laser_spot",
            "attached_obstacle": obstacle_record["name"],
            "radius": float(LASER_SPOT_RADIUS),
            "height": float(LASER_SPOT_THICKNESS),
            "collision": bool(LASER_SPOT_HAS_COLLISION),
        },
    )

    record = {
        "name": "LaserSpot",
        "path": path,
        "type": "laser_spot",
        "shape": "thin_cylinder_disk",
        "x": float(laser_x),
        "y": float(laser_y),
        "z": float(laser_z),
        "radius": float(LASER_SPOT_RADIUS),
        "height": float(LASER_SPOT_THICKNESS),
        "collision": bool(LASER_SPOT_HAS_COLLISION),
        "color_r": float(LASER_SPOT_COLOR[0]),
        "color_g": float(LASER_SPOT_COLOR[1]),
        "color_b": float(LASER_SPOT_COLOR[2]),
    }

    return prim, record

# ============================================================
# Random placement
# ============================================================

def distance_2d(a_x, a_y, b_x, b_y):
    return math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)


def point_to_direct_path_distance(x, y):
    start_x, start_y = float(START_POS[0]), float(START_POS[1])
    target_x, target_y = float(TARGET_POS[0]), float(TARGET_POS[1])
    dx = target_x - start_x
    dy = target_y - start_y
    length_squared = dx * dx + dy * dy
    if length_squared < 1e-9:
        return distance_2d(x, y, start_x, start_y)
    t = ((x - start_x) * dx + (y - start_y) * dy) / length_squared
    t = max(0.0, min(1.0, t))
    closest_x = start_x + t * dx
    closest_y = start_y + t * dy
    return distance_2d(x, y, closest_x, closest_y)


def is_valid_obstacle_position(x, y, radius, placed_obstacles):
    start_dist = distance_2d(x, y, START_POS[0], START_POS[1])
    if start_dist < START_CLEAR_RADIUS + radius:
        return False

    target_dist = distance_2d(x, y, TARGET_POS[0], TARGET_POS[1])
    if target_dist < TARGET_CLEAR_RADIUS + radius:
        return False

    for item in placed_obstacles:
        other_x = item["x"]
        other_y = item["y"]
        other_radius = item["radius"]

        min_dist = radius + other_radius + MIN_OBSTACLE_GAP
        if distance_2d(x, y, other_x, other_y) < min_dist:
            return False

    return True


def random_building_spec(blocker=False):
    width = random.uniform(
        BLOCKER_BUILDING_WIDTH_MIN if blocker else BUILDING_WIDTH_MIN,
        BUILDING_WIDTH_MAX,
    )
    depth = random.uniform(
        BLOCKER_BUILDING_DEPTH_MIN if blocker else BUILDING_DEPTH_MIN,
        BUILDING_DEPTH_MAX,
    )
    height = random.uniform(
        max(BUILDING_HEIGHT_MIN, DIRECT_PATH_BLOCKER_HEIGHT_MIN) if blocker else BUILDING_HEIGHT_MIN,
        BUILDING_HEIGHT_MAX,
    )
    return {
        "width": width,
        "depth": depth,
        "height": height,
        "radius": 0.5 * math.hypot(width, depth),
        "blocker_half_extent": 0.5 * min(width, depth),
        "yaw_deg": random.uniform(BUILDING_YAW_MIN_DEG, BUILDING_YAW_MAX_DEG),
        "facade_color": random.choice(BUILDING_FACADE_COLORS),
        "window_color": random.choice(BUILDING_WINDOW_ON_COLORS),
        "roof_style": random.choice(BUILDING_ROOF_STYLES),
        "roof_height": random.uniform(BUILDING_ROOF_HEIGHT_MIN, BUILDING_ROOF_HEIGHT_MAX),
        "antenna_height": random.uniform(BUILDING_ANTENNA_HEIGHT_MIN, BUILDING_ANTENNA_HEIGHT_MAX),
    }


def generate_obstacle_specs(num_obstacles):
    placed = []
    max_attempts = 1000

    blocker_count = 0
    if GUARANTEE_DIRECT_PATH_BLOCKERS:
        blocker_count = min(
            max(0, int(DIRECT_PATH_BLOCKER_COUNT)),
            max(0, int(num_obstacles)),
            len(DIRECT_PATH_BLOCKER_T_RANGES),
        )

    start_x, start_y = float(START_POS[0]), float(START_POS[1])
    target_x, target_y = float(TARGET_POS[0]), float(TARGET_POS[1])
    line_dx = target_x - start_x
    line_dy = target_y - start_y
    line_length = math.hypot(line_dx, line_dy)
    if blocker_count and line_length < 1e-6:
        raise RuntimeError("Cannot guarantee direct-path blockers when start and target coincide.")

    # Unit normal to the direct route. Small random lateral jitter retains
    # scene diversity while the route still crosses the building's inscribed
    # footprint circle, independent of building yaw.
    normal_x = -line_dy / max(line_length, 1e-6)
    normal_y = line_dx / max(line_length, 1e-6)

    for blocker_index in range(blocker_count):
        success = False
        t_min, t_max = DIRECT_PATH_BLOCKER_T_RANGES[blocker_index]

        for _attempt in range(max_attempts):
            spec = random_building_spec(blocker=True)
            radius = spec["radius"]
            t = random.uniform(float(t_min), float(t_max))
            lateral_limit = min(
                DIRECT_PATH_BLOCKER_LATERAL_JITTER_M,
                spec["blocker_half_extent"] * 0.70,
            )
            lateral = random.uniform(-lateral_limit, lateral_limit)
            x = start_x + t * line_dx + lateral * normal_x
            y = start_y + t * line_dy + lateral * normal_y

            inside_bounds = (
                X_MIN + radius <= x <= X_MAX - radius
                and Y_MIN + radius <= y <= Y_MAX - radius
            )
            blocks_direct_path = (
                point_to_direct_path_distance(x, y)
                <= spec["blocker_half_extent"]
            )

            if (
                inside_bounds
                and blocks_direct_path
                and is_valid_obstacle_position(x, y, radius, placed)
            ):
                spec.update(
                    {
                        "x": x,
                        "y": y,
                        "placement_mode": "guaranteed_direct_path_blocker",
                    }
                )
                placed.append(spec)
                success = True
                break

        if not success:
            raise RuntimeError(
                "Could not place a guaranteed direct-path blocker. "
                "Check the corridor ranges, clear radii, and obstacle spacing."
            )

    for _ in range(max(0, int(num_obstacles) - blocker_count)):
        success = False

        for _attempt in range(max_attempts):
            spec = random_building_spec(blocker=False)
            radius = spec["radius"]

            x = random.uniform(X_MIN + radius, X_MAX - radius)
            y = random.uniform(Y_MIN + radius, Y_MAX - radius)

            if is_valid_obstacle_position(x, y, radius, placed):
                spec.update(
                    {
                        "x": x,
                        "y": y,
                        "placement_mode": "random",
                    }
                )
                placed.append(spec)
                success = True
                break

        if not success:
            print("[Warning] Could not place one obstacle. Try using a larger area or fewer obstacles.")
            break

    physical_blockers = [
        item for item in placed
        if point_to_direct_path_distance(item["x"], item["y"])
        <= item["blocker_half_extent"]
        and item["height"] >= DIRECT_PATH_BLOCKER_HEIGHT_MIN
    ]
    if GUARANTEE_DIRECT_PATH_BLOCKERS and len(physical_blockers) < blocker_count:
        raise RuntimeError(
            "Generated scene failed the direct-path blocker validation: "
            f"expected={blocker_count}, actual={len(physical_blockers)}"
        )

    print(
        "[SceneConstraint] direct_path_blockers="
        f"{len(physical_blockers)} required={blocker_count}; "
        "the physical start-to-target corridor is blocked by high-rise buildings."
    )
    return placed


# ============================================================
# Logging
# ============================================================

def save_episode_records(episode_id, seed, records):
    os.makedirs(LOG_DIR, exist_ok=True)

    json_path = os.path.join(LOG_DIR, f"{episode_id}.json")
    csv_path = os.path.join(LOG_DIR, f"{episode_id}.csv")

    payload = {
        "episode_id": episode_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "generated_root": GENERATED_ROOT,
        "start_pos": START_POS,
        "target_pos": TARGET_POS,
        "num_objects": len(records),
        "objects": records,
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    fieldnames = [
        "name",
        "path",
        "type",
        "shape",
        "x",
        "y",
        "z",
        "radius",
        "width",
        "depth",
        "height",
        "yaw_deg",
        "roof_style",
        "collision",
        "color_r",
        "color_g",
        "color_b",
        "placement_mode",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    return json_path, csv_path


def print_episode_summary(episode_id, records, json_path, csv_path):
    print("")
    print("=" * 70)
    print(f"Scene episode generated: {episode_id}")
    print(f"Generated root: {GENERATED_ROOT}")
    print(f"Object count: {len(records)}")
    print("-" * 70)

    for record in records:
        print(
            f"{record['name']:>14s} | "
            f"{record['type']:>8s} | "
            f"pos=({record['x']:+.3f}, {record['y']:+.3f}, {record['z']:+.3f}) | "
            f"r={record['radius']:.3f}, h={record['height']:.3f} | "
            f"collision={record['collision']}"
        )

    print("-" * 70)
    print(f"JSON saved to: {json_path}")
    print(f"CSV  saved to: {csv_path}")
    print("=" * 70)
    print("")


# ============================================================
# Main
# ============================================================

def generate_scene_episode():
    stage = get_stage()

    if RANDOM_SEED is None:
        seed = random.randint(0, 999999)
    else:
        seed = int(RANDOM_SEED)

    random.seed(seed)

    episode_id = f"scene_episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed_{seed}"

    ensure_physics_scene(stage)

    clear_old_generated_scene(stage)

    root = ensure_xform(stage, GENERATED_ROOT)
    set_display_color(root.GetPrim(), COLOR_ROOT)

    if CREATE_EPISODE_LIGHTING:
        create_episode_lighting(stage)

    ensure_xform(stage, f"{GENERATED_ROOT}/Obstacles")
    ensure_xform(stage, f"{GENERATED_ROOT}/Start")
    ensure_xform(stage, f"{GENERATED_ROOT}/Target")

    if CREATE_START_MARKER:
        ensure_xform(stage, f"{GENERATED_ROOT}/Start")

    records = []

    if CREATE_START_MARKER:
        _start_prim, start_record = create_start_marker(stage)
        records.append(start_record)

    if CREATE_TARGET_MARKER:
        _target_prim, target_record = create_target_marker(stage)
        records.append(target_record)

    obstacle_specs = generate_obstacle_specs(NUM_OBSTACLES)

    for index, spec in enumerate(obstacle_specs, start=1):
        _prim, record = create_obstacle(
            stage=stage,
            index=index,
            x=spec["x"],
            y=spec["y"],
            width=spec["width"],
            depth=spec["depth"],
            height=spec["height"],
            yaw_deg=spec["yaw_deg"],
            facade_color=spec["facade_color"],
            window_color=spec["window_color"],
            roof_style=spec["roof_style"],
            roof_height=spec["roof_height"],
            antenna_height=spec["antenna_height"],
        )
        record["placement_mode"] = spec.get("placement_mode", "random")
        records.append(record)
    #     obstacle_records.append(record)

    # if CREATE_LASER_SPOT and obstacle_records:
    #     laser_obstacle = select_laser_obstacle(obstacle_records)
    #     _laser_prim, laser_record = create_laser_spot_on_obstacle(stage, laser_obstacle)
        # records.append(laser_record)

    # Do not create the old red target marker anymore.
    # _target_prim, target_record = create_target_marker(stage)
    # records.append(target_record)

    json_path, csv_path = save_episode_records(
        episode_id=episode_id,
        seed=seed,
        records=records,
    )

    print_episode_summary(
        episode_id=episode_id,
        records=records,
        json_path=json_path,
        csv_path=csv_path,
    )

    # Do not call Save() here because the current stage may be an anonymous layer.
    # The generated object positions are already saved to JSON and CSV.
    # Save the USD manually from Isaac Sim if needed.
    # omni.usd.get_context().get_stage().GetRootLayer().Save()

    return records


if __name__ == "__main__":
    generate_scene_episode()
