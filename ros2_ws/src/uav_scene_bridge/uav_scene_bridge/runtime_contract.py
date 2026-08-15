"""Validate the narrow JSON contract emitted from Isaac Sim."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


RUNTIME_SCHEMA = "uav_isaac_runtime/v1"
ISAAC_FRAME = "isaac_world"


@dataclass(frozen=True, slots=True)
class RuntimeObstacle:
    """One circular planning envelope measured in Isaac world metres."""

    name: str
    x: float
    y: float
    z: float
    radius: float
    height: float


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """One validated Isaac heartbeat and deterministic scene description."""

    sequence: int
    scene_id: str
    scene_revision: int
    timeline_playing: bool
    prim_valid: bool
    pose_valid: bool
    vehicle_prim_path: str
    goal: tuple[float, float, float]
    obstacles: tuple[RuntimeObstacle, ...]

    @property
    def ready(self) -> bool:
        """Return whether Isaac reports every runtime prerequisite healthy."""
        return self.timeline_playing and self.prim_valid and self.pose_valid


def _finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive(value, label: str) -> float:
    number = _finite(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _nonempty(value, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must be nonempty")
    return text


def _strict_bool(value, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be bool")
    return value


def parse_runtime_snapshot(payload: str) -> RuntimeSnapshot:
    """Parse and reject malformed, non-finite, or ambiguous Isaac status."""
    try:
        data = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"runtime status is not valid JSON: {error}"
        ) from error
    if not isinstance(data, dict):
        raise ValueError("runtime status must be a JSON object")
    if data.get("schema") != RUNTIME_SCHEMA:
        raise ValueError(f"runtime schema must be {RUNTIME_SCHEMA}")

    sequence = int(data.get("sequence", -1))
    revision = int(data.get("scene_revision", 0))
    if sequence < 0:
        raise ValueError("sequence must be nonnegative")
    if revision <= 0:
        raise ValueError("scene_revision must be positive")

    raw_goal = data.get("goal")
    if not isinstance(raw_goal, list) or len(raw_goal) != 3:
        raise ValueError("goal must contain exactly three coordinates")
    goal = tuple(
        _finite(value, f"goal[{index}]")
        for index, value in enumerate(raw_goal)
    )

    raw_obstacles = data.get("obstacles")
    if not isinstance(raw_obstacles, list):
        raise ValueError("obstacles must be a list")
    obstacles = []
    names = set()
    for index, raw in enumerate(raw_obstacles):
        if not isinstance(raw, dict):
            raise ValueError(f"obstacles[{index}] must be an object")
        name = _nonempty(
            raw.get("name", ""), f"obstacles[{index}].name"
        )
        if name in names:
            raise ValueError(f"duplicate obstacle name: {name}")
        names.add(name)
        obstacles.append(RuntimeObstacle(
            name=name,
            x=_finite(raw.get("x"), f"{name}.x"),
            y=_finite(raw.get("y"), f"{name}.y"),
            z=_finite(raw.get("z"), f"{name}.z"),
            radius=_positive(raw.get("radius"), f"{name}.radius"),
            height=_positive(raw.get("height"), f"{name}.height"),
        ))

    return RuntimeSnapshot(
        sequence=sequence,
        scene_id=_nonempty(data.get("scene_id", ""), "scene_id"),
        scene_revision=revision,
        timeline_playing=_strict_bool(
            data.get("timeline_playing"), "timeline_playing"
        ),
        prim_valid=_strict_bool(data.get("prim_valid"), "prim_valid"),
        pose_valid=_strict_bool(data.get("pose_valid"), "pose_valid"),
        vehicle_prim_path=_nonempty(
            data.get("vehicle_prim_path", ""), "vehicle_prim_path"
        ),
        goal=goal,
        obstacles=tuple(obstacles),
    )
