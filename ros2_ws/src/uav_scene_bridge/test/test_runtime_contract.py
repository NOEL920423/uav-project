"""Unit tests for the Isaac runtime JSON trust boundary."""

import json

import pytest

from uav_scene_bridge.runtime_contract import parse_runtime_snapshot


def _payload():
    return {
        "schema": "uav_isaac_runtime/v1",
        "sequence": 4,
        "scene_id": "bootstrap_fixed_scene_v1",
        "scene_revision": 1,
        "timeline_playing": True,
        "prim_valid": True,
        "pose_valid": True,
        "vehicle_prim_path": "/World/quadrotor/body",
        "goal": [0.5, 3.0, 1.5],
        "obstacles": [{
            "name": "BootstrapObstacle_001",
            "x": -1.5,
            "y": 1.5,
            "z": 1.25,
            "radius": 0.4,
            "height": 2.5,
        }],
    }


def test_accepts_ready_finite_runtime_snapshot():
    """A complete heartbeat exposes the deterministic scene exactly once."""
    snapshot = parse_runtime_snapshot(json.dumps(_payload()))
    assert snapshot.ready
    assert snapshot.goal == (0.5, 3.0, 1.5)
    assert snapshot.obstacles[0].name == "BootstrapObstacle_001"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema", "unknown"),
        ("sequence", -1),
        ("scene_revision", 0),
        ("timeline_playing", 1),
        ("goal", [0.0, float("nan"), 1.5]),
        ("obstacles", "not-a-list"),
    ],
)
def test_rejects_malformed_or_ambiguous_runtime_state(key, value):
    """Invalid schema, health, coordinates, or collection types fail closed."""
    payload = _payload()
    payload[key] = value
    with pytest.raises(ValueError):
        parse_runtime_snapshot(json.dumps(payload))


def test_unready_runtime_is_valid_but_not_ready():
    """A paused timeline parses for diagnosis but cannot authorize a scene."""
    payload = _payload()
    payload["timeline_playing"] = False
    assert not parse_runtime_snapshot(json.dumps(payload)).ready


def test_duplicate_obstacle_names_are_rejected():
    """Ambiguous obstacle identity cannot cross the bridge."""
    payload = _payload()
    payload["obstacles"].append(dict(payload["obstacles"][0]))
    with pytest.raises(ValueError, match="duplicate obstacle"):
        parse_runtime_snapshot(json.dumps(payload))
