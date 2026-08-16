"""Pure deterministic tests for the canonical high-rise scene distribution."""

import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "isaac" / "runtime" / "episode_scene.py"
SPEC = importlib.util.spec_from_file_location("episode_scene", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_seed_reproduces_canonical_eight_building_scene():
    """The same reset pose and seed produce byte-equal canonical metadata."""
    left = MODULE.generate_episode_scene("episode_000001", 102001, 0.0, 0.0)
    right = MODULE.generate_episode_scene("episode_000001", 102001, 0.0, 0.0)
    assert left == right
    assert left["generator"] == "canonical_highrise_scene_generator_v1"
    assert left["start"] == [0.0, 0.0, 0.0]
    assert left["target_marker"] == [3.0, 5.0, 0.0]
    assert left["goal"] == [3.0, 5.0, 1.5]
    assert len(left["obstacles"]) == 8
    assert {item["shape"] for item in left["obstacles"]} == {
        "high_rise_building"
    }
    assert left["direct_path_blocker_count"] == 2
    assert left["obstacles"][0]["x"] == pytest.approx(1.0852854015267037)
    assert left["obstacles"][0]["placement_mode"] == (
        "guaranteed_direct_path_blocker"
    )
    assert left["lighting"]["mode"] == "exact_legacy"
    assert left["lighting"]["dome"]["intensity"] == 300.0
    assert left["lighting"]["key"]["intensity"] == 1300.0
    assert left["lighting"]["fill"]["intensity"] == 650.0


def test_three_qa_seeds_are_distinct_and_obey_canonical_constraints():
    """QA seeds preserve buildings, blockers, bounds, disks, and spacing."""
    scenes = [
        MODULE.generate_episode_scene(
            f"episode_{index:06d}", 102000 + index, 0.0, 0.0
        )
        for index in range(1, 4)
    ]
    layouts = {
        tuple((item["x"], item["y"]) for item in scene["obstacles"])
        for scene in scenes
    }
    assert len(layouts) == 3
    for scene in scenes:
        obstacles = scene["obstacles"]
        assert scene["normal_obstacle_count"] == 8
        assert scene["direct_path_blocker_count"] == 2
        assert len(obstacles) == 8
        for item in obstacles:
            radius = item["radius"]
            assert 0.46 <= item["width"] <= 0.72
            assert 0.46 <= item["depth"] <= 0.72
            assert 2.8 <= item["height"] <= 5.2
            assert -35.0 <= item["yaw_deg"] <= 35.0
            assert radius == pytest.approx(
                0.5 * math.hypot(item["width"], item["depth"])
            )
            assert -2.0 + radius <= item["x"] <= 5.0 - radius
            assert -1.0 + radius <= item["y"] <= 7.0 - radius
            assert math.hypot(item["x"], item["y"]) >= 1.5 + radius
            assert math.hypot(item["x"] - 3.0, item["y"] - 5.0) >= (
                1.5 + radius
            )
        for index, left in enumerate(obstacles):
            for right in obstacles[index + 1:]:
                assert math.hypot(
                    left["x"] - right["x"], left["y"] - right["y"]
                ) >= left["radius"] + right["radius"] + 0.50
        blockers = [
            item for item in obstacles
            if item["placement_mode"] == "guaranteed_direct_path_blocker"
        ]
        assert len(blockers) == 2
        assert all(item["height"] >= 3.2 for item in blockers)
        assert all(
            MODULE._point_to_direct_path_distance(item["x"], item["y"])
            <= item["blocker_half_extent"]
            for item in blockers
        )
        assert all(item["windows"]["on_pattern"] for item in obstacles)
        assert all("Body" in item["hierarchy"] for item in obstacles)
        assert all("Windows" in item["hierarchy"] for item in obstacles)
        assert all("Roof/Crown" in item["hierarchy"] for item in obstacles)


def test_blocked_goal_preserves_eight_normal_obstacles_plus_fixture():
    """The Phase 10B safe-failure mode remains available and explicit."""
    scene = MODULE.generate_episode_scene(
        "episode_000005", 101005, 0.0, 0.0, "blocked_goal"
    )
    assert scene["normal_obstacle_count"] == 8
    assert scene["obstacle_count"] == 9
    blocker = scene["obstacles"][-1]
    assert (blocker["x"], blocker["y"]) == (3.0, 5.0)


def test_reset_pose_must_remain_near_canonical_start():
    """Scene recovery fails closed if PX4/Pegasus did not reset at the start."""
    with pytest.raises(ValueError, match="outside the canonical start margin"):
        MODULE.generate_episode_scene("episode_000001", 102001, 1.0, 1.0)
