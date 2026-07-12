# scene_redpoint_episode_generator.py
# Generate one UAV obstacle scene episode with a red laser spot on an obstacle in Isaac Sim.
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

# Obstacle size range.
OBSTACLE_RADIUS_MIN = 0.18
OBSTACLE_RADIUS_MAX = 0.35
OBSTACLE_HEIGHT_MIN = 1.5
OBSTACLE_HEIGHT_MAX = 3.2

# Obstacle spacing.
# A slightly larger gap makes the A* baseline less likely to create impossible mazes.
MIN_OBSTACLE_GAP = 0.50

# Red target disk is disabled.
# We do not create a red target object anymore.

# Laser spot settings.
CREATE_LASER_SPOT = True

# None means randomly choose one obstacle.
# If you want to fix it on a specific obstacle, use 1, 2, 3...
LASER_SPOT_OBSTACLE_INDEX = None

# The laser spot is a very thin red disk attached to the obstacle surface.
LASER_SPOT_RADIUS = 0.08
LASER_SPOT_THICKNESS = 0.01

# 0.65 means the spot is placed at 65% of the obstacle height.
LASER_SPOT_HEIGHT_RATIO = 0.65
LASER_SPOT_HEIGHT_RATIO_MIN = 0.30
LASER_SPOT_HEIGHT_RATIO_MAX = 0.85
LASER_SPOT_RANDOM_SURFACE_ANGLE = False
LASER_SPOT_FACE_START = True
LASER_SPOT_ROOT = f"{GENERATED_ROOT}/LaserSpot"
LASER_SPOT_PATH = f"{LASER_SPOT_ROOT}/LaserSpot"

# Push the spot slightly outside the obstacle surface to avoid z-fighting.
LASER_SPOT_SURFACE_OFFSET = 0.015

LASER_SPOT_COLOR = (1.0, 0.0, 0.0)
LASER_SPOT_HAS_COLLISION = False

# Optional small red light near the spot.
CREATE_LASER_SPOT_LIGHT = False
LASER_SPOT_LIGHT_INTENSITY = 250.0
LASER_SPOT_LIGHT_RADIUS = 0.05

# Colors.
COLOR_OBSTACLE = (0.15, 0.15, 0.15)
COLOR_TARGET = (1.0, 0.0, 0.0)
COLOR_ROOT = (0.2, 0.2, 0.2)

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


def create_obstacle(stage, index, x, y, radius, height):
    path = f"{GENERATED_ROOT}/Obstacles/Obstacle_{index:03d}"
    z = height / 2.0

    prim = create_cylinder(
        stage=stage,
        path=path,
        radius=radius,
        height=height,
        position=(x, y, z),
        color=COLOR_OBSTACLE,
        collision=True,
        object_type="obstacle",
    )

    record = {
        "name": f"Obstacle_{index:03d}",
        "path": path,
        "type": "obstacle",
        "shape": "cylinder",
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "radius": float(radius),
        "height": float(height),
        "collision": True,
        "color_r": float(COLOR_OBSTACLE[0]),
        "color_g": float(COLOR_OBSTACLE[1]),
        "color_b": float(COLOR_OBSTACLE[2]),
    }

    return prim, record


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
    obstacle_z = float(obstacle_record["z"])
    obstacle_radius = float(obstacle_record["radius"])
    obstacle_height = float(obstacle_record["height"])

    height_ratio = random.uniform(
        float(LASER_SPOT_HEIGHT_RATIO_MIN),
        float(LASER_SPOT_HEIGHT_RATIO_MAX),
    )

    if LASER_SPOT_RANDOM_SURFACE_ANGLE:
        surface_angle_rad = random.uniform(-math.pi, math.pi)
        normal_x = math.cos(surface_angle_rad)
        normal_y = math.sin(surface_angle_rad)
    elif LASER_SPOT_FACE_START:
        # Make the laser spot face the UAV start position.
        # This is useful because the first camera baseline is more likely to see the red spot.
        direction_x = START_POS[0] - obstacle_x
        direction_y = START_POS[1] - obstacle_y
        direction_length = math.sqrt(direction_x ** 2 + direction_y ** 2)

        if direction_length < 1e-6:
            normal_x = 1.0
            normal_y = 0.0
        else:
            normal_x = direction_x / direction_length
            normal_y = direction_y / direction_length
    else:
        surface_angle_rad = math.radians(0.0)
        normal_x = math.cos(surface_angle_rad)
        normal_y = math.sin(surface_angle_rad)

    laser_x = obstacle_x + normal_x * (
        obstacle_radius + LASER_SPOT_THICKNESS * 0.5 + LASER_SPOT_SURFACE_OFFSET
    )
    laser_y = obstacle_y + normal_y * (
        obstacle_radius + LASER_SPOT_THICKNESS * 0.5 + LASER_SPOT_SURFACE_OFFSET
    )
    laser_z = obstacle_height * height_ratio

    # The default cylinder axis is along Z.
    # Rotate it so the thin disk sticks to the side surface of the cylinder.
    angle_deg = math.degrees(math.atan2(normal_y, normal_x))
    rotation_deg = (0.0, 90.0, angle_deg)

    ensure_xform(stage, LASER_SPOT_ROOT)
    path = LASER_SPOT_PATH

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
        light_path = f"{LASER_SPOT_ROOT}/LaserSpotLight"
        light = UsdLux.SphereLight.Define(stage, Sdf.Path(light_path))
        light.CreateRadiusAttr(float(LASER_SPOT_LIGHT_RADIUS))
        light.CreateIntensityAttr(float(LASER_SPOT_LIGHT_INTENSITY))
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.0, 0.0))

        light_prim = light.GetPrim()
        set_transform(light_prim, (laser_x, laser_y, laser_z))

    metadata = {
        "object_type": "laser_spot",
        "attached_obstacle": obstacle_record["name"],
        "attached_obstacle_path": obstacle_record["path"],
        "attached_obstacle_center_x": float(obstacle_x),
        "attached_obstacle_center_y": float(obstacle_y),
        "attached_obstacle_center_z": float(obstacle_z),
        "attached_obstacle_radius": float(obstacle_radius),
        "attached_obstacle_height": float(obstacle_height),
        "surface_normal_x": float(normal_x),
        "surface_normal_y": float(normal_y),
        "surface_normal_z": 0.0,
        "height_ratio": float(height_ratio),
        "radius": float(LASER_SPOT_RADIUS),
        "height": float(LASER_SPOT_THICKNESS),
        "collision": bool(LASER_SPOT_HAS_COLLISION),
    }
    set_custom_metadata(prim, metadata)

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
        "attached_obstacle": obstacle_record["name"],
        "attached_obstacle_path": obstacle_record["path"],
        "attached_obstacle_center_x": float(obstacle_x),
        "attached_obstacle_center_y": float(obstacle_y),
        "attached_obstacle_center_z": float(obstacle_z),
        "attached_obstacle_radius": float(obstacle_radius),
        "attached_obstacle_height": float(obstacle_height),
        "surface_normal_x": float(normal_x),
        "surface_normal_y": float(normal_y),
        "surface_normal_z": 0.0,
        "height_ratio": float(height_ratio),
    }

    return prim, record

# ============================================================
# Random placement
# ============================================================

def distance_2d(a_x, a_y, b_x, b_y):
    return math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) ** 2)


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


def generate_obstacle_specs(num_obstacles):
    placed = []
    max_attempts = 1000

    for _ in range(num_obstacles):
        success = False

        for _attempt in range(max_attempts):
            radius = random.uniform(OBSTACLE_RADIUS_MIN, OBSTACLE_RADIUS_MAX)
            height = random.uniform(OBSTACLE_HEIGHT_MIN, OBSTACLE_HEIGHT_MAX)

            x = random.uniform(X_MIN + radius, X_MAX - radius)
            y = random.uniform(Y_MIN + radius, Y_MAX - radius)

            if is_valid_obstacle_position(x, y, radius, placed):
                placed.append(
                    {
                        "x": x,
                        "y": y,
                        "radius": radius,
                        "height": height,
                    }
                )
                success = True
                break

        if not success:
            print("[Warning] Could not place one obstacle. Try using a larger area or fewer obstacles.")
            break

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
        "height",
        "collision",
        "color_r",
        "color_g",
        "color_b",
        "attached_obstacle",
        "attached_obstacle_path",
        "attached_obstacle_center_x",
        "attached_obstacle_center_y",
        "attached_obstacle_center_z",
        "attached_obstacle_radius",
        "attached_obstacle_height",
        "surface_normal_x",
        "surface_normal_y",
        "surface_normal_z",
        "height_ratio",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
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

        if record.get("type") == "laser_spot":
            print(
                f"{'':>14s} | "
                f"attached={record.get('attached_obstacle', '')}, "
                f"normal=({float(record.get('surface_normal_x', 0.0)):+.3f}, "
                f"{float(record.get('surface_normal_y', 0.0)):+.3f}), "
                f"height_ratio={float(record.get('height_ratio', 0.0)):.3f}"
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

    ensure_xform(stage, f"{GENERATED_ROOT}/Obstacles")
    ensure_xform(stage, f"{GENERATED_ROOT}/Start")
    ensure_xform(stage, f"{GENERATED_ROOT}/Target")
    ensure_xform(stage, LASER_SPOT_ROOT)

    if CREATE_START_MARKER:
        ensure_xform(stage, f"{GENERATED_ROOT}/Start")

    records = []
    obstacle_records = []

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
            radius=spec["radius"],
            height=spec["height"],
        )
        records.append(record)
        obstacle_records.append(record)

    if CREATE_LASER_SPOT and obstacle_records:
        laser_obstacle = select_laser_obstacle(obstacle_records)
        _laser_prim, laser_record = create_laser_spot_on_obstacle(stage, laser_obstacle)
        records.append(laser_record)

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