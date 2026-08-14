"""Regression tests for RGB episode classification and clean splitting."""

import unittest
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from uav_ml.datasets.rgb_episode_dataset import RgbEpisodeDataset
from uav_ml.inference import RgbEncoderInference
from uav_ml.models import RgbAutoencoderV0
from uav_ml.tools.rgb_episode_audit import EpisodeAudit, _classify, _episode_split


def _audit(episode_id: str, environment: str) -> EpisodeAudit:
    return EpisodeAudit(
        episode_id=episode_id,
        category="astar_expert",
        environment=environment,
        eligible_autoencoder=True,
        eligible_expert_bc=True,
        manifest_rows=1,
        fpv_files=1,
        top_files=1,
        pose_rows=1,
        duration_sim_s=1.0,
        displacement_m=1.0,
        fpv_dimensions="960x540:RGB=1",
        top_dimensions="960x540:RGB=1",
        corrupt_images=0,
        missing_images=0,
        orphan_images=0,
        duplicate_consecutive_fpv=0,
        duplicate_consecutive_top=0,
        size_bytes=1,
        issue_count=0,
        status="valid",
        issues=[],
    )


class RgbEpisodeAuditTests(unittest.TestCase):
    def test_source_and_environment_classification(self) -> None:
        self.assertEqual(
            _classify("dual_camera_episode_bc_natural_astar_20260720_191355"),
            ("astar_expert", "natural"),
        )
        self.assertEqual(
            _classify("dual_camera_episode_bc_forced_astar_bc_20260720_202554"),
            ("policy_rollout", "forced"),
        )
        self.assertEqual(
            _classify("dual_camera_episode_city_bc_20260721_093339"),
            ("policy_rollout", "city"),
        )

    def test_split_is_episode_level_deterministic_and_stratified(self) -> None:
        audits = [
            _audit(f"episode_{environment}_{index}", environment)
            for environment in ("baseline", "city")
            for index in range(4)
        ]
        first = _episode_split(audits, seed=123)
        second = _episode_split(audits, seed=123)
        self.assertEqual(first, second)
        all_ids = [episode_id for values in first.values() for episode_id in values]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(set(all_ids), {audit.episode_id for audit in audits})
        for split_name in ("validation", "test"):
            environments = {episode_id.split("_")[1] for episode_id in first[split_name]}
            self.assertEqual(environments, {"baseline", "city"})

    def test_rgb_dataset_and_autoencoder_tensor_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="uav-rgb-ae-test-") as temporary:
            root = Path(temporary)
            episode_id = "dual_camera_episode_city_astar_20260811_000000"
            episode_dir = root / episode_id
            fpv_dir = episode_dir / "images" / "fpv"
            fpv_dir.mkdir(parents=True)
            image_path = fpv_dir / "frame_000001.png"
            pixels = np.zeros((54, 96, 3), dtype=np.uint8)
            pixels[..., 0] = 128
            Image.fromarray(pixels).save(image_path)
            with (episode_dir / "camera_frames.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "episode_id",
                        "frame_index",
                        "sim_time",
                        "fpv_image_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "episode_id": episode_id,
                        "frame_index": 1,
                        "sim_time": 1.0,
                        "fpv_image_path": str(image_path),
                    }
                )
            split_path = root / "split.json"
            split_path.write_text(
                json.dumps(
                    {
                        "seed": 1,
                        "unit": "episode",
                        "camera": "fpv",
                        "splits": {
                            "train": [episode_id],
                            "validation": [episode_id],
                            "test": [episode_id],
                        },
                    }
                ),
                encoding="utf-8",
            )
            dataset = RgbEpisodeDataset(root, split_path, "train")
            image = dataset[0]["image"]
            self.assertEqual(tuple(image.shape), (3, 72, 128))
            self.assertGreaterEqual(float(image.min()), 0.0)
            self.assertLessEqual(float(image.max()), 1.0)

            model = RgbAutoencoderV0()
            reconstruction, latent = model(image.unsqueeze(0))
            self.assertEqual(tuple(reconstruction.shape), (1, 3, 72, 128))
            self.assertEqual(tuple(latent.shape), (1, 64))
            self.assertTrue(torch.isfinite(reconstruction).all())
            self.assertTrue(torch.isfinite(latent).all())
            loss = torch.nn.functional.mse_loss(reconstruction, image.unsqueeze(0))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))

            checkpoint = root / "autoencoder.pt"
            torch.save(
                {
                    "model_class": "RgbAutoencoderV0",
                    "model_config": model.config.to_dict(),
                    "model_state": model.state_dict(),
                    "metadata": {"test": True},
                },
                checkpoint,
            )
            encoder = RgbEncoderInference(checkpoint, device="cpu")
            latent_array = encoder.encode(pixels)
            self.assertEqual(latent_array.shape, (64,))
            self.assertTrue(np.isfinite(latent_array).all())


if __name__ == "__main__":
    unittest.main()
