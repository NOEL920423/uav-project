"""Pure deterministic tests for the canonical expert cylinder scene."""

import importlib.util
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "isaac" / "runtime" / "environment.py"
SPEC = importlib.util.spec_from_file_location("environment", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_seed_reproduces_configured_obstacle_scene():
    """The same reset pose and seed produce byte-equal canonical metadata."""
    left = MODULE.generate_episode_scene("episode_000001", 102001, 0.0, 0.0)
    right = MODULE.generate_episode_scene("episode_000001", 102001, 0.0, 0.0)
    assert left == right
    assert left["generator"] == "canonical_cylinder_scene_generator_v1"
    assert left["start"] == [0.0, 0.0, 0.0]
    assert left["target_marker"] == [3.0, 5.0, 0.0]
    assert left["goal"] == [3.0, 5.0, 1.5]
    assert len(left["obstacles"]) == MODULE.NUM_OBSTACLES
    assert MODULE.NUM_OBSTACLES == (
        MODULE.DIRECT_PATH_BLOCKER_COUNT
        + MODULE.RANDOM_OBSTACLE_COUNT
    )
    assert sum(
        item["placement_mode"] == "random" for item in left["obstacles"]
    ) == MODULE.RANDOM_OBSTACLE_COUNT
    assert {item["shape"] for item in left["obstacles"]} == {
        "cylinder"
    }
    assert left["direct_path_blocker_count"] == 2
    assert left["obstacles"][0]["placement_mode"] == (
        "guaranteed_direct_path_blocker"
    )
    assert left["obstacles"][0]["variant"] == "special_large"
    assert left["obstacles"][0]["radius_basis_width"] == pytest.approx(0.72)
    assert left["obstacles"][0]["radius_basis_depth"] == pytest.approx(0.72)
    assert left["obstacles"][0]["height"] == pytest.approx(5.2)
    assert len(left["walls"]) == 4
    assert all(wall["name"].startswith("Wall_") for wall in left["walls"])
    assert left["lighting"]["mode"] == "neutral_dome_only"
    assert left["lighting"]["dome"]["intensity"] == 800.0
    assert "key" not in left["lighting"]
    assert "fill" not in left["lighting"]


def test_three_qa_seeds_share_fixed_geometry_and_obey_constraints():
    """QA seeds keep geometry fixed while preserving scene constraints."""
    scenes = [
        MODULE.generate_episode_scene(
            f"episode_{index:06d}", 102000 + index, 0.0, 0.0
        )
        for index in range(1, 4)
    ]
    geometry = lambda scene: tuple(
        tuple(
            item[key]
            for key in (
                "x", "y", "z", "radius_basis_width",
                "radius_basis_depth", "height", "yaw_deg",
            )
        )
        for item in scene["obstacles"]
    )
    assert geometry(scenes[0]) == geometry(scenes[1]) == geometry(scenes[2])
    for scene in scenes:
        obstacles = scene["obstacles"]
        area = scene["placement_contract"]["area"]
        assert area == {
            "x": [MODULE.X_MIN, MODULE.X_MAX],
            "y": [MODULE.Y_MIN, MODULE.Y_MAX],
        }
        x_min, x_max = area["x"]
        y_min, y_max = area["y"]
        assert scene["normal_obstacle_count"] == MODULE.NUM_OBSTACLES
        assert scene["direct_path_blocker_count"] == 2
        assert len(obstacles) == MODULE.NUM_OBSTACLES
        for item in obstacles:
            radius = item["radius"]
            assert 0.46 <= item["radius_basis_width"] <= 0.72
            assert 0.46 <= item["radius_basis_depth"] <= 0.72
            assert 2.8 <= item["height"] <= 5.2
            assert -35.0 <= item["yaw_deg"] <= 35.0
            assert radius == pytest.approx(
                0.5 * math.hypot(
                    item["radius_basis_width"],
                    item["radius_basis_depth"],
                )
            )
            assert x_min + radius <= item["x"] <= x_max - radius
            assert y_min + radius <= item["y"] <= y_max - radius
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


def test_blocked_goal_preserves_normal_obstacles_plus_fixture():
    """The blocked-goal safe-failure mode remains available and explicit."""
    scene = MODULE.generate_episode_scene(
        "episode_000005", 101005, 0.0, 0.0, "blocked_goal"
    )
    assert scene["normal_obstacle_count"] == MODULE.NUM_OBSTACLES
    assert scene["obstacle_count"] == MODULE.NUM_OBSTACLES + 1
    blocker = scene["obstacles"][-1]
    assert (blocker["x"], blocker["y"]) == (3.0, 5.0)


def test_reset_pose_must_remain_near_canonical_start():
    """Scene recovery fails closed if PX4/Pegasus did not reset at the start."""
    with pytest.raises(ValueError, match="outside the canonical start margin"):
        MODULE.generate_episode_scene("episode_000001", 102001, 1.0, 1.0)


@pytest.mark.parametrize("version, expected_digest", [
    ("v3", "f9d92fb5399ff01006e064cc6a80a7f1012589c1183dc491ed7775329f7af74b"),
    ("v4", "f49b448ed6308c3967c7fa7cd3b0645df7fd0f5a7675ee1e56b96162c3254170"),
])
def test_scene_matches_historical_dataset_obstacles(version, expected_digest):
    """Golden digests come from the original episode_000001 records."""
    scene = MODULE.generate_episode_scene("episode_000001", 103001, 0.0, 0.0, scene_version=version)
    digest = hashlib.sha256(json.dumps(
        scene["obstacles"], sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    assert digest == expected_digest


def test_versions_use_frozen_parameters_and_distinct_seed_contracts(monkeypatch):
    monkeypatch.setattr(MODULE, "FLOOR_COLOR", (0.1, 0.2, 0.3))
    scenes = {
        version: [MODULE.generate_episode_scene(
            "episode_000001", seed, 0.0, 0.0, scene_version=version,
        ) for seed in (103001, 103002)] for version in ("v3", "v4")
    }
    assert scenes["v3"][0]["obstacles"] != scenes["v3"][1]["obstacles"]
    assert scenes["v4"][0]["obstacles"] == scenes["v4"][1]["obstacles"]
    assert scenes["v3"][0]["environment_parameters"]["arena"]["floor"]["color_linear_rgb"] == [0.6, 0.6, 0.6]
    assert MODULE.FLOOR_COLOR == (0.1, 0.2, 0.3)


def test_scene_ack_requires_selected_version_and_profile_digest():
    """Run the actual ROS callback offline to reject stale acknowledgments."""
    source = ROOT / "ros2_ws/src/uav_data_recorder/uav_data_recorder/episode_scene_client.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    callback = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_status")
    scope = {"json": json, "String": SimpleNamespace}
    exec(compile(ast.Module(body=[callback], type_ignores=[]), str(source), "exec"), scope)
    finishes = []
    logger = SimpleNamespace(debug=lambda value: None, info=lambda value: None)
    scene = MODULE.generate_episode_scene("episode_000001", 103001, 0.0, 0.0, scene_version="v3")
    client = SimpleNamespace(
        episode_id=scene["episode_id"], random_seed=103001, scene_mode="normal",
        scene_version="v3", scene_profile_sha256=scene["scene_profile_sha256"],
        minimum_scene_revision=0, expected_runtime_generation=1,
        get_logger=lambda: logger, _finish=lambda *args: finishes.append(args),
    )
    status = dict(
        episode_id=scene["episode_id"], random_seed=103001, scene_configuration=scene,
        timeline_playing=True, pose_valid=True, scene_revision=1, runtime_generation=1,
    )
    for version, digest in (("v4", scene["scene_profile_sha256"]), ("v3", "wrong-digest")):
        bad = {**status, "scene_configuration": {**scene, "scene_version": version, "scene_profile_sha256": digest}}
        scope["_status"](client, SimpleNamespace(data=json.dumps(bad)))
        assert finishes == []
    scope["_status"](client, SimpleNamespace(data=json.dumps(status)))
    assert finishes[0][0] == 0


def test_bridge_reapplies_same_episode_when_scene_version_changes():
    """A duplicate episode/seed must not reuse another version's scene."""
    cls = next(node for node in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(node, ast.ClassDef) and node.name == "IsaacRuntimeBridge")
    callback = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_episode_command_callback")
    scope = {**vars(MODULE), "_world_pose": lambda *args: (0.0, 0.0, 0.0)}
    exec(compile(ast.Module(body=[callback], type_ignores=[]), str(SOURCE), "exec"), scope)
    applied = []
    bridge = SimpleNamespace(
        _episode_id=None, _random_seed=None, _scene_configuration=None,
        _stage=None, _scene_revision=0, _camera_frame_count=1,
        _observer_frame_count=2, _depth_frame_count=3,
        _apply_scene=lambda scene: applied.append(scene),
    )
    for version in ("v3", "v4", "v4"):
        request = dict(command="prepare_episode", episode_id="episode_000001", random_seed=103001, scene_version=version)
        scope["_episode_command_callback"](bridge, SimpleNamespace(data=json.dumps(request)))
        assert bridge._episode_command_error == ""
    assert [scene["scene_version"] for scene in applied] == ["v3", "v4"]
    assert bridge._scene_revision == 2
