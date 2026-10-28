"""Offline geometry and reproducibility checks for the v5 center corridor."""

import copy
import json
import math

import pytest

from isaac.runtime import environment


def assert_corridor_geometry(scene):
    """Check the corridor independently using projections onto its axes."""
    profile = scene["scene_profile"]
    p = profile["parameters"]
    start_x, start_y = p["START_POS"][:2]
    dx = p["TARGET_POS"][0] - start_x
    dy = p["TARGET_POS"][1] - start_y
    length = math.hypot(dx, dy)
    half_width = profile["obstacle_placement"]["width_m"] / 2.0
    obstacles = scene["obstacles"]
    assert len(obstacles) == 2
    for obstacle in obstacles:
        x, y, radius = (obstacle[key] for key in ("x", "y", "radius"))
        t = ((x - start_x) * dx + (y - start_y) * dy) / length**2
        lateral = (-(x - start_x) * dy + (y - start_y) * dx) / length
        assert 0.0 <= t <= 1.0
        assert abs(lateral) <= half_width + 1e-12
        assert p["X_MIN"] + radius <= x <= p["X_MAX"] - radius
        assert p["Y_MIN"] + radius <= y <= p["Y_MAX"] - radius
        assert math.hypot(x - start_x, y - start_y) >= p["START_CLEAR_RADIUS"] + radius
        assert math.hypot(x - p["TARGET_POS"][0], y - p["TARGET_POS"][1]) >= p["TARGET_CLEAR_RADIUS"] + radius
        assert p["RADIUS_BASIS_WIDTH_MIN"] <= obstacle["radius_basis_width"] <= p["RADIUS_BASIS_WIDTH_MAX"]
        assert p["RADIUS_BASIS_DEPTH_MIN"] <= obstacle["radius_basis_depth"] <= p["RADIUS_BASIS_DEPTH_MAX"]
        assert p["CYLINDER_HEIGHT_MIN"] <= obstacle["height"] <= p["CYLINDER_HEIGHT_MAX"]
        assert radius == pytest.approx(0.5 * math.hypot(
            obstacle["radius_basis_width"], obstacle["radius_basis_depth"],
        ))
        assert obstacle["z"] == pytest.approx(obstacle["height"] / 2.0)
        assert obstacle["placement_mode"] == "random"
        assert obstacle["windows"]["on_pattern"]
    left, right = obstacles
    assert math.hypot(left["x"] - right["x"], left["y"] - right["y"]) >= (
        left["radius"] + right["radius"] + p["MIN_OBSTACLE_GAP"]
    )


@pytest.mark.parametrize("seed", range(103001, 103101))
def test_corridor_preserves_geometry_and_clearances_across_seeds(seed):
    scene = environment.generate_episode_scene(
        "episode_000001", seed, 0.0, 0.0, scene_version="v5",
    )
    assert_corridor_geometry(scene)
    assert scene["direct_path_blocker_count"] == 0
    assert scene["environment_parameters"]["obstacle_generation"]["random_obstacle_count"] == 2
    assert scene["placement_contract"]["obstacle_placement"] == {
        "mode": "direct_path_corridor", "width_m": 1.0, "constraint": "center",
        "longitudinal_t_range": [0.0, 1.0], "lateral_offset_m": [-0.5, 0.5],
    }


def test_corridor_seed_and_saved_profile_reproduce_geometry(tmp_path):
    profile = environment.load_scene_profile("v5")
    snapshot = tmp_path / "scene_parameters.json"
    snapshot.write_text(json.dumps(environment.scene_profile_record(profile)))
    left = environment.generate_episode_scene(
        "episode_000001", 103001, 0.0, 0.0, scene_version="v5",
    )
    right = environment.generate_episode_scene(
        "episode_000001", 103001, 0.0, 0.0, scene_version="v5",
        scene_profile=environment.load_scene_profile("v5", snapshot),
    )
    assert left == right
    changed = environment.generate_episode_scene(
        "episode_000001", 103002, 0.0, 0.0, scene_version="v5",
    )
    for index in range(2):
        assert left["obstacles"][index]["x"] != changed["obstacles"][index]["x"]
        assert left["obstacles"][index]["radius"] != changed["obstacles"][index]["radius"]
        assert left["obstacles"][index]["height"] != changed["obstacles"][index]["height"]


def test_both_centers_share_the_full_width_and_longitudinal_range():
    offsets = [[], []]
    orders = set()
    body_extends_outside = False
    length = math.hypot(3.0, 5.0)
    for seed in range(103001, 103101):
        scene = environment.generate_episode_scene(
            "episode_000001", seed, 0.0, 0.0, scene_version="v5",
        )
        positions = []
        for index, obstacle in enumerate(scene["obstacles"]):
            lateral = (-5.0 * obstacle["x"] + 3.0 * obstacle["y"]) / length
            offsets[index].append(lateral)
            positions.append((3.0 * obstacle["x"] + 5.0 * obstacle["y"]) / length**2)
            body_extends_outside |= abs(lateral) + obstacle["radius"] > 0.5
        orders.add(positions[0] < positions[1])
    assert orders == {True, False}
    for values in offsets:
        assert min(values) < -0.4
        assert max(values) > 0.4
    assert body_extends_outside


@pytest.mark.parametrize("start, target", [
    ((-3.0, 2.0, 0.0), (4.0, 2.0, 0.0)),
    ((0.0, -1.0, 0.0), (0.0, 6.0, 0.0)),
    ((4.0, 5.0, 0.0), (-2.0, 0.0, 0.0)),
])
def test_corridor_follows_translated_and_rotated_endpoints(start, target):
    profile = environment.load_scene_profile("v5")
    profile["parameters"].update(START_POS=start, TARGET_POS=target)
    scene = environment.generate_episode_scene(
        "episode_000001", 103001, *start[:2], scene_version="v5", scene_profile=profile,
    )
    assert_corridor_geometry(scene)


@pytest.mark.parametrize("width", [0.0, -1.0, True, "1.0"])
def test_invalid_corridor_width_is_rejected(width):
    profile = environment.load_scene_profile("v5")
    profile["obstacle_placement"]["width_m"] = width
    with pytest.raises(ValueError, match="invalid obstacle center corridor"):
        environment.scene_parameters(profile)


def test_v5_requires_center_corridor_and_random_obstacles():
    original = environment.load_scene_profile("v5")
    profile = copy.deepcopy(original)
    del profile["obstacle_placement"]
    with pytest.raises(ValueError, match="invalid obstacle center corridor"):
        environment.scene_parameters(profile)
    profile = copy.deepcopy(original)
    profile["obstacle_placement"]["constraint"] = "body"
    with pytest.raises(ValueError, match="invalid obstacle center corridor"):
        environment.scene_parameters(profile)
    profile = copy.deepcopy(original)
    profile["parameters"]["GUARANTEE_DIRECT_PATH_BLOCKERS"] = True
    with pytest.raises(ValueError, match="only seeded random obstacles"):
        environment.scene_parameters(profile)


def test_impossible_corridor_layout_fails_with_bounded_attempts():
    profile = environment.load_scene_profile("v5")
    profile["parameters"].update(MIN_OBSTACLE_GAP=10.0, MAX_PLACEMENT_ATTEMPTS=20)
    with pytest.raises(RuntimeError, match="after 20 layout attempts"):
        environment.generate_episode_scene(
            "episode_000001", 103001, 0.0, 0.0, scene_version="v5", scene_profile=profile,
        )


def test_corridor_rejects_coincident_endpoints():
    profile = environment.load_scene_profile("v5")
    profile["parameters"]["TARGET_POS"] = profile["parameters"]["START_POS"]
    with pytest.raises(RuntimeError, match="start and target coincide"):
        environment.generate_episode_scene(
            "episode_000001", 103001, 0.0, 0.0, scene_version="v5", scene_profile=profile,
        )
