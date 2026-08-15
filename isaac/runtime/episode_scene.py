"""Deterministic Phase 10B scene descriptions without Isaac dependencies."""

from __future__ import annotations

import math
import random


def generate_episode_scene(
    episode_id: str,
    seed: int,
    start_east_m: float,
    start_north_m: float,
    mode: str = "normal",
) -> dict:
    """Generate a bounded scene from a seed and the recorded landed pose."""
    if not episode_id.startswith("episode_"):
        raise ValueError("episode_id must use the episode_NNNNNN form")
    if mode not in {"normal", "blocked_goal"}:
        raise ValueError(f"unsupported scene mode: {mode}")
    values = (float(start_east_m), float(start_north_m))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("scene start must be finite")

    rng = random.Random(int(seed))
    heading_rad = rng.uniform(-math.pi, math.pi)
    distance_m = rng.uniform(2.6, 3.3)
    direction = (math.cos(heading_rad), math.sin(heading_rad))
    lateral = (-direction[1], direction[0])
    goal = (
        values[0] + distance_m * direction[0],
        values[1] + distance_m * direction[1],
        1.5,
    )
    side = -1.0 if rng.random() < 0.5 else 1.0
    obstacles = [
        {
            "name": f"Obstacle_{episode_id}_01",
            "x": values[0] + 0.52 * distance_m * direction[0]
            + side * 0.68 * lateral[0],
            "y": values[1] + 0.52 * distance_m * direction[1]
            + side * 0.68 * lateral[1],
            "z": 1.25,
            "radius": 0.43,
            "height": 2.5,
        },
        {
            "name": f"Obstacle_{episode_id}_02",
            "x": values[0] + 0.28 * distance_m * direction[0]
            - side * 1.25 * lateral[0],
            "y": values[1] + 0.28 * distance_m * direction[1]
            - side * 1.25 * lateral[1],
            "z": 1.0,
            "radius": 0.34,
            "height": 2.0,
        },
    ]
    if mode == "blocked_goal":
        obstacles.append({
            "name": f"Obstacle_{episode_id}_blocked_goal",
            "x": goal[0],
            "y": goal[1],
            "z": 1.5,
            "radius": 0.85,
            "height": 3.0,
        })
    return {
        "episode_id": episode_id,
        "random_seed": int(seed),
        "generator": "phase10b_seeded_relative_scene_v1",
        "mode": mode,
        "reset_kind": "full_isaac_pegasus_px4_restart",
        "start": [values[0], values[1], 0.0],
        "goal": list(goal),
        "heading_rad": heading_rad,
        "distance_m": distance_m,
        "obstacles": obstacles,
    }
