"""Tests for live BC checkpoint, observation, and action contracts."""

import json
import math
from pathlib import Path
import unittest
from io import BytesIO

import numpy as np
from PIL import Image
import torch

from uav_ml.inference.bc_flight import (
    body_action_to_ned,
    build_state8,
    canonical_image_source,
    freshness_error,
    load_checkpoint_payload,
    validate_live_image,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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


if __name__ == "__main__":
    unittest.main()
