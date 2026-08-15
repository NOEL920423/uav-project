"""Pure deterministic tests for Phase 10B seeded scene generation."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "isaac" / "runtime" / "episode_scene.py"
SPEC = importlib.util.spec_from_file_location("episode_scene", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_seed_and_start_fully_reproduce_scene():
    """The same recorded reset pose and seed produce byte-equal metadata."""
    left = MODULE.generate_episode_scene("episode_000001", 101001, 0.5, 3.0)
    right = MODULE.generate_episode_scene("episode_000001", 101001, 0.5, 3.0)
    assert left == right
    assert left["random_seed"] == 101001
    assert left["reset_kind"] == "full_isaac_pegasus_px4_restart"


def test_pilot_seeds_generate_distinct_reachable_scene_descriptions():
    """Normal pilot seeds vary goals and keep bounded relative distances."""
    scenes = [
        MODULE.generate_episode_scene(
            f"episode_{index:06d}", 101000 + index, 0.0, 0.0
        )
        for index in range(1, 11)
    ]
    assert len({tuple(scene["goal"]) for scene in scenes}) == 10
    assert all(2.6 <= scene["distance_m"] <= 3.3 for scene in scenes)
    assert all(len(scene["obstacles"]) == 2 for scene in scenes)


def test_blocked_goal_mode_is_explicit_and_reproducible():
    """The pilot's safe failure fixture places one obstacle on its goal."""
    scene = MODULE.generate_episode_scene(
        "episode_000005", 101005, 1.0, -2.0, "blocked_goal"
    )
    blocker = scene["obstacles"][-1]
    assert (blocker["x"], blocker["y"]) == pytest.approx(scene["goal"][:2])
    assert scene["mode"] == "blocked_goal"
