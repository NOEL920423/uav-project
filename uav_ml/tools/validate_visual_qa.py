"""Validate the three-flight canonical high-rise Phase 10C visual QA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_SEEDS = {102001, 102002, 102003}
CAPTURE_PHASES = ("start", "mid_flight", "near_goal")
EXPECTED_LIGHTING = {
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


def _image_statistics(path: Path, expected_format: str) -> dict:
    with Image.open(path) as image:
        if image.size != (320, 180) or image.format != expected_format:
            raise ValueError(f"invalid image contract: {path}")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    luminance = rgb.astype(np.float32).mean(axis=2)
    return {
        "mean_luminance": float(luminance.mean()),
        "p01_luminance": float(np.percentile(luminance, 1.0)),
        "p99_luminance": float(np.percentile(luminance, 99.0)),
        "dynamic_range": int(rgb.max()) - int(rgb.min()),
        "dark_fraction_below_8": float((luminance < 8.0).mean()),
        "overexposed_fraction_above_247": float((luminance > 247.0).mean()),
    }


def _validate_buildings(directory: Path, scene: dict) -> tuple[list, int]:
    obstacles = scene.get("obstacles", [])
    if len(obstacles) != 8 or scene.get("normal_obstacle_count") != 8:
        raise ValueError(f"{directory.name}: expected exactly eight buildings")
    blockers = 0
    for building in obstacles:
        if building.get("shape") != "high_rise_building":
            raise ValueError(f"{directory.name}: non-high-rise obstacle")
        width = float(building["width"])
        depth = float(building["depth"])
        height = float(building["height"])
        yaw = float(building["yaw_deg"])
        radius = float(building["radius"])
        if not (
            0.46 <= width <= 0.72
            and 0.46 <= depth <= 0.72
            and 2.80 <= height <= 5.20
            and -35.0 <= yaw <= 35.0
        ):
            raise ValueError(f"{directory.name}: building dimensions invalid")
        if not math.isclose(radius, 0.5 * math.hypot(width, depth)):
            raise ValueError(f"{directory.name}: planner radius is not canonical")
        hierarchy = set(building.get("hierarchy", []))
        if not {"Body", "Windows", "Roof/Crown"}.issubset(hierarchy):
            raise ValueError(f"{directory.name}: building hierarchy incomplete")
        if building.get("roof_style") not in {"flat", "crown", "antenna"}:
            raise ValueError(f"{directory.name}: roof style invalid")
        windows = building.get("windows", {})
        expected_windows = int(windows.get("row_count", 0)) * 2 * (
            int(windows.get("columns_x", 0))
            + int(windows.get("columns_y", 0))
        )
        if len(windows.get("on_pattern", [])) != expected_windows:
            raise ValueError(f"{directory.name}: facade window layout incomplete")
        if building.get("placement_mode") == "guaranteed_direct_path_blocker":
            blockers += 1
    if blockers != 2 or scene.get("direct_path_blocker_count") != 2:
        raise ValueError(f"{directory.name}: expected two guaranteed blockers")
    return obstacles, blockers


def _validate_capture(directory: Path, phase: str, capture: dict) -> dict:
    status = capture.get("flight_status") or {}
    if not (
        status.get("state") == "TRACKING"
        and status.get("tracking_active") is True
        and status.get("astar_selected") is True
        and float(status.get("altitude_m", 0.0)) >= 1.0
    ):
        raise ValueError(f"{directory.name}: {phase} was not in ASTAR tracking")
    span = float(capture.get("stream_capture_span_s", math.inf))
    if not math.isfinite(span) or span > 0.65:
        raise ValueError(f"{directory.name}: {phase} streams too far apart")
    images = capture.get("images", {})
    fpv_path = directory / images["fpv_rgb"]["path"]
    observer_path = directory / images["observer_rgb"]["path"]
    depth_path = directory / images["fpv_depth_raw"]["path"]
    preview_path = directory / images["fpv_depth_preview"]["path"]
    fpv = _image_statistics(fpv_path, "JPEG")
    observer = _image_statistics(observer_path, "JPEG")
    for label, statistics in (("FPV", fpv), ("Observer", observer)):
        if statistics["dynamic_range"] < 32:
            raise ValueError(f"{directory.name}: {phase} {label} lacks range")
        if statistics["dark_fraction_below_8"] > 0.70:
            raise ValueError(f"{directory.name}: {phase} {label} mostly black")
        if statistics["overexposed_fraction_above_247"] > 0.30:
            raise ValueError(f"{directory.name}: {phase} {label} overexposed")
    with Image.open(depth_path) as image:
        if image.size != (320, 180) or image.format != "PNG":
            raise ValueError(f"{directory.name}: {phase} raw depth invalid")
        depth = np.asarray(image)
    if depth.dtype != np.uint16 or not np.any(depth > 0) or depth.max() > 30000:
        raise ValueError(f"{directory.name}: {phase} depth contract invalid")
    with Image.open(preview_path) as image:
        if image.size != (320, 180) or image.format != "PNG":
            raise ValueError(f"{directory.name}: {phase} preview invalid")
        preview = np.asarray(image.convert("L"), dtype=np.uint8)
    if int(preview.max()) - int(preview.min()) < 32:
        raise ValueError(f"{directory.name}: {phase} depth preview lacks range")
    valid_depth = depth[depth > 0]
    return {
        "flight_status": status,
        "stream_capture_span_s": span,
        "fpv_rgb": fpv,
        "observer_rgb": observer,
        "depth": {
            "encoding": "PNG uint16 millimetres",
            "minimum_valid_mm": int(valid_depth.min()),
            "maximum_valid_mm": int(valid_depth.max()),
            "valid_fraction": float((depth > 0).mean()),
            "preview_dynamic_range": int(preview.max()) - int(preview.min()),
        },
    }


def validate(root: Path, write_result: bool = True) -> dict:
    """Validate canonical scene, cameras, path detour, and terminal safety."""
    root = root.resolve()
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if len(directories) != 3:
        raise ValueError(f"expected exactly three QA scenes, found {len(directories)}")
    seeds = set()
    layouts = set()
    scenes = []
    for directory in directories:
        metadata = json.loads(
            (directory / "scene_metadata.json").read_text(encoding="utf-8")
        )
        scene = metadata.get("scene_configuration")
        if not isinstance(scene, dict):
            raise ValueError(f"{directory.name}: scene metadata missing")
        if scene.get("generator") != "canonical_highrise_scene_generator_v1":
            raise ValueError(f"{directory.name}: wrong scene generator")
        if scene.get("lighting") != EXPECTED_LIGHTING:
            raise ValueError(f"{directory.name}: lighting is not exact legacy")
        obstacles, blocker_count = _validate_buildings(directory, scene)
        seed = int(metadata["random_seed"])
        layout = tuple((item["x"], item["y"]) for item in obstacles)
        if seed in seeds or layout in layouts:
            raise ValueError("QA seeds and layouts must be unique")
        seeds.add(seed)
        layouts.add(layout)
        camera = metadata.get("camera_contract", {})
        if camera.get("fpv", {}).get("look_down_m") != -0.8:
            raise ValueError(f"{directory.name}: FPV look-down override missing")
        if camera.get("fpv", {}).get("position_smoothing") != (
            "disabled_rigid_body_mount"
        ):
            raise ValueError(f"{directory.name}: FPV is not a rigid mount")
        if camera.get("observer", {}).get("mode") != "TOP":
            raise ValueError(f"{directory.name}: Observer is not canonical TOP")
        planner_path = metadata.get("planner_path") or {}
        detour_ratio = float(planner_path.get("detour_ratio", 0.0))
        detour_distance = float(
            planner_path.get("detour_distance_xy_m", 0.0)
        )
        if detour_ratio <= 1.02 or detour_distance <= 0.10:
            raise ValueError(f"{directory.name}: A* path lacks meaningful detour")
        captures = metadata.get("captures", {})
        if set(captures) != set(CAPTURE_PHASES):
            raise ValueError(f"{directory.name}: missing flight capture phase")
        capture_results = {
            phase: _validate_capture(directory, phase, captures[phase])
            for phase in CAPTURE_PHASES
        }
        evidence = json.loads(
            (directory / "flight_evidence.json").read_text(encoding="utf-8")
        )
        final_px4 = evidence.get("final_px4") or {}
        first_evidence = evidence.get("first_evidence") or {}
        if not (
            evidence.get("success") is True
            and final_px4.get("landed") is True
            and final_px4.get("arming_state") == 1
            and final_px4.get("failsafe") is False
            and not evidence.get("stream_faults")
            and "final_disarmed" in first_evidence
            and "goal_reached" in first_evidence
        ):
            raise ValueError(f"{directory.name}: flight did not end safely")
        scenes.append({
            "episode_id": metadata["episode_id"],
            "random_seed": seed,
            "obstacle_count": len(obstacles),
            "direct_path_blocker_count": blocker_count,
            "planner_path": planner_path,
            "captures": capture_results,
            "flight_success": True,
            "safe_terminal_state": {
                "landed": True,
                "disarmed": True,
                "failsafe": False,
            },
        })
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"unexpected QA seeds: {sorted(seeds)}")
    result = {
        "valid": True,
        "phase": "10C",
        "purpose": "three-phase visual QA; not training data",
        "scene_count": len(scenes),
        "random_seeds": sorted(seeds),
        "obstacle_layouts_unique": len(layouts),
        "buildings_per_scene": 8,
        "direct_path_blockers_per_scene": 2,
        "scene_generator": "canonical_highrise_scene_generator_v1",
        "lighting": EXPECTED_LIGHTING,
        "fpv_geometry": (
            "legacy BODY_AXIS +X, effective look-down -0.8 m, rigid mount"
        ),
        "observer_geometry": "legacy Episode Manager TOP",
        "all_paths_have_meaningful_detours": True,
        "all_flights_successful_and_safely_landed": True,
        "scenes": scenes,
    }
    if write_result:
        (root / "visual_qa_manifest.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="artifacts/visual_qa/phase10c_highrise_rigid_fpv",
    )
    args = parser.parse_args()
    print(json.dumps(validate(Path(args.root)), indent=2))


if __name__ == "__main__":
    main()
