"""Tests for live BC checkpoint, observation, and action contracts."""

import json
import math
from pathlib import Path
import unittest
import tempfile
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from uav_ml.inference.bc_flight import (
    body_action_to_ned,
    build_state8,
    canonical_image_source,
    freshness_error,
    load_checkpoint_payload,
    resolve_checkpoint,
    validate_live_image,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class TimingReportTests(unittest.TestCase):
    def test_ulog_capture_rejects_unrelated_process(self):
        import os
        from uav_ml.tools.bc_flight_evaluation import ManagedFlightRuntime

        runtime = ManagedFlightRuntime(Path('.'), Path('.'), False, 'cpu', Path('.'), 'top_rgb', 180)
        runtime._isaac = SimpleNamespace(pid=os.getppid())
        self.assertTrue(runtime._owns_px4_process(os.getpid()))
        runtime._isaac = SimpleNamespace(pid=999999999)
        self.assertFalse(runtime._owns_px4_process(os.getpid()))

    def test_interrupted_evaluation_finalizes_saved_results(self):
        from uav_ml.tools.bc_flight_evaluation import _finalize_evaluation

        result = {
            "schema": "uav_bc_flight_result/v1",
            "episode": 1,
            "terminal_reason": "timeout",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode_dir = root / "episode_000001"
            episode_dir.mkdir()
            (episode_dir / "result.json").write_text(json.dumps(result))
            _finalize_evaluation(
                root,
                [],
                image_source="top_rgb",
                identity=SimpleNamespace(
                    checkpoint_path="checkpoint.pt", checkpoint_sha256="digest"
                ),
            )
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["episodes"], [result])
            self.assertTrue(summary["plots"]["summary"])

    def test_join_uses_observed_monotonic_stamps_not_source_clock(self):
        from scripts.diagnostics.summarize_bc_startup import write_timing_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "timing").mkdir()
            events = [
                {"node": "sender", "event": "publish", "mono_ns": 1000000,
                 "topic": "image", "key": "header:999999:0"},
                {"node": "receiver", "event": "receive", "mono_ns": 4000000,
                 "topic": "image", "key": "header:999999:0"},
                {"node": "receiver", "event": "receive", "mono_ns": 6000000,
                 "topic": "px4", "key": "px4_wire:9"},
            ]
            text = "\n".join(json.dumps({"schema": "uav_timing/v1", **event})
                             for event in events)
            (root / "timing/events.jsonl").write_text(text + "\n{truncated\n")
            summary = write_timing_report(root)
            self.assertEqual(summary["statistics"][0]["max_ms"], 3.0)
            self.assertEqual(summary["malformed_records"], 1)
            self.assertEqual(summary["unmatched_receives"], {"receiver｜px4": 1})
            self.assertIn("尚未證明", (root / "timing_summary.md").read_text())
            self.assertEqual(summary["ulog"], None)

    def test_empty_historical_log_does_not_invent_measurements(self):
        from scripts.diagnostics.summarize_bc_startup import write_timing_report

        with tempfile.TemporaryDirectory() as directory:
            summary = write_timing_report(Path(directory))
            self.assertEqual(summary["statistics"], [])
            self.assertEqual(summary["event_count"], 0)

    def test_open_ulog_survives_temporary_rootfs_removal(self):
        from uav_ml.tools.bc_flight_evaluation import ManagedFlightRuntime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "temporary.ulg"
            source.write_bytes(b"ULog-test-evidence")
            runtime = ManagedFlightRuntime(root, root, False, "cpu", root, "top_rgb", 180)
            runtime._ulog_handles[str(source)] = source.open("rb")
            source.unlink()
            destination = root / "saved"
            destination.mkdir()
            runtime._write_ulog_snapshot(destination)
            self.assertEqual((destination / source.name).read_bytes(), b"ULog-test-evidence")


class BcFlightContractTests(unittest.TestCase):
    def _top_checkpoint(self) -> Path:
        pointer = REPOSITORY_ROOT / (
            "artifacts/experiments/bc/bc_expert_cylinder_v1/top/latest.json"
        )
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        return Path(payload["best_checkpoint"])

    def test_legacy_top_is_canonical_top_rgb(self) -> None:
        self.assertEqual(canonical_image_source("top"), "top_rgb")
        self.assertEqual(canonical_image_source("top_rgb"), "top_rgb")

    def test_top_checkpoint_rejects_requested_fpv(self) -> None:
        with self.assertRaisesRegex(ValueError, "image source mismatch"):
            load_checkpoint_payload(
                self._top_checkpoint(), "fpv_rgb", torch.device("cpu")
            )

    def test_checkpoint_accepts_path_relative_to_bc_experiments(self) -> None:
        checkpoint = self._top_checkpoint().resolve()
        short_path = checkpoint.relative_to(
            REPOSITORY_ROOT / "artifacts/experiments/bc"
        )
        self.assertEqual(
            resolve_checkpoint(REPOSITORY_ROOT, short_path), checkpoint
        )

    def test_state8_matches_body_contract(self) -> None:
        state = build_state8(
            velocity_north_mps=0.4,
            velocity_east_mps=0.2,
            position_north_m=0.0,
            position_east_m=0.0,
            goal_north_m=3.0,
            goal_east_m=4.0,
            yaw_ned_rad=0.0,
            previous_normalized_action=(0.1, -0.2, 0.3),
        )
        np.testing.assert_allclose(
            state,
            [0.4, 0.2, 0.6, 0.8, 0.5, 0.1, -0.2, 0.3],
            atol=1e-6,
        )

    def test_body_action_converts_to_ned(self) -> None:
        north, east, down, yaw_rate = body_action_to_ned(
            (1.0, 1.0, -0.5), math.pi / 2.0
        )
        self.assertAlmostEqual(north, -0.8)
        self.assertAlmostEqual(east, 1.0)
        self.assertEqual(down, 0.0)
        self.assertAlmostEqual(yaw_rate, -0.5)

    def test_stale_image_and_odometry_fail_closed(self) -> None:
        self.assertEqual(
            freshness_error(2.0, 1.0, 1.9, True, 0.35, 0.25),
            "stale_top_rgb",
        )
        self.assertEqual(
            freshness_error(2.0, 1.9, 1.0, True, 0.35, 0.25),
            "stale_odometry",
        )
        self.assertIsNone(
            freshness_error(2.0, 1.9, 1.9, True, 0.35, 0.25)
        )

    def test_live_top_resolution_is_exact(self) -> None:
        """Reject a legacy observer resolution under the TOP RGB label."""
        stream = BytesIO()
        Image.new("RGB", (320, 180)).save(stream, format="JPEG")
        with self.assertRaisesRegex(ValueError, "must be 640x360"):
            validate_live_image(stream.getvalue(), "top_rgb")
        stream = BytesIO()
        Image.new("RGB", (640, 360)).save(stream, format="JPEG")
        validate_live_image(stream.getvalue(), "top_rgb")

    def test_live_fpv_rgb_and_depth_contracts_are_exact(self) -> None:
        stream = BytesIO()
        Image.new("RGB", (320, 180)).save(stream, format="JPEG")
        validate_live_image(stream.getvalue(), "fpv_rgb")

        stream = BytesIO()
        depth = np.full((180, 320), 1000, dtype=np.uint16)
        Image.fromarray(depth, mode="I;16").save(stream, format="PNG")
        validate_live_image(stream.getvalue(), "fpv_depth")

        stream = BytesIO()
        Image.new("L", (320, 180)).save(stream, format="PNG")
        with self.assertRaisesRegex(ValueError, "uint16"):
            validate_live_image(stream.getvalue(), "fpv_depth")

    def test_freshness_reason_tracks_requested_source(self) -> None:
        self.assertEqual(
            freshness_error(2.0, None, 1.9, True, 0.35, 0.25, "fpv_depth"),
            "waiting_for_fpv_depth",
        )
        self.assertEqual(
            freshness_error(2.0, 1.0, 1.9, True, 0.35, 0.25, "fpv_rgb"),
            "stale_fpv_rgb",
        )


if __name__ == "__main__":
    unittest.main()
