"""ROS/Isaac-independent regression tests for ML Phase 1."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from uav_ml.contracts import DEFAULT_CONTRACT
from uav_ml.datasets.dataset import BcEpisodeDataset
from uav_ml.datasets.split import split_episode_ids
from uav_ml.datasets.synthetic_expert import generate_synthetic_dataset
from uav_ml.datasets.validation import validate_dataset
from uav_ml.inference import BcPolicyInference
from uav_ml.models import BcPolicyV0
from uav_ml.training.checkpoint import load_checkpoint, save_checkpoint
from uav_ml.training.normalization import (
    TorchNormalizer,
    compute_normalization,
)


class MlPhase1Test(unittest.TestCase):
    """Exercise dataset, model, training step, checkpoint, and inference."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="uav-ml-test-")
        cls.dataset_path = Path(cls.temporary.name) / "dataset"
        generate_synthetic_dataset(
            cls.dataset_path,
            episodes=3,
            maximum_steps=8,
            seed=41,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_episode_split_is_deterministic_and_disjoint(self) -> None:
        ids = [f"episode_{index}" for index in range(10)]
        first = split_episode_ids(ids, seed=9)
        second = split_episode_ids(list(reversed(ids)), seed=9)
        self.assertEqual(first, second)
        self.assertFalse(set(first[0]) & set(first[1]))

    def test_dataset_shapes_sync_finite_and_deterministic(self) -> None:
        statistics = validate_dataset(self.dataset_path)
        self.assertEqual(statistics["episodes"], 3)
        first = BcEpisodeDataset(self.dataset_path, "train")
        second = BcEpisodeDataset(self.dataset_path, "train")
        self.assertEqual(len(first), len(second))
        for name, expected in (
            ("depth", (1, 64, 64)),
            ("velocity", (3,)),
            ("goal_direction", (3,)),
            ("action", (4,)),
        ):
            self.assertEqual(tuple(first[0][name].shape), expected)
            self.assertTrue(torch.isfinite(first[0][name]).all())
            self.assertTrue(torch.equal(first[0][name], second[0][name]))

    def test_normalization_uses_train_and_is_finite(self) -> None:
        train_dataset = BcEpisodeDataset(self.dataset_path, "train")
        stats = compute_normalization(train_dataset)
        self.assertEqual(stats.source_split, "train")
        self.assertTrue(np.isfinite(np.asarray(stats.action_std)).all())
        with self.assertRaises(ValueError):
            compute_normalization(BcEpisodeDataset(self.dataset_path, "validation"))

    def test_model_forward_batch_and_optimizer_step(self) -> None:
        model = BcPolicyV0()
        depth = torch.ones(3, 1, 64, 64)
        velocity = torch.zeros(3, 3)
        goal = torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1)
        target = torch.zeros(3, 4)
        output = model(depth, velocity, goal)
        self.assertEqual(tuple(output.shape), (3, 4))
        self.assertTrue(torch.isfinite(output).all())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss = torch.nn.functional.mse_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))

    def test_checkpoint_reload_inference_and_action_clipping(self) -> None:
        dataset = BcEpisodeDataset(self.dataset_path, "train")
        stats = compute_normalization(dataset)
        model = BcPolicyV0()
        checkpoint = Path(self.temporary.name) / "test_checkpoint.pt"
        save_checkpoint(checkpoint, model, stats, {"test": True})
        reloaded, reloaded_stats, _ = load_checkpoint(checkpoint)
        self.assertEqual(reloaded.parameter_count, model.parameter_count)
        self.assertEqual(reloaded_stats, stats)
        sample = dataset[0]
        policy = BcPolicyInference(str(checkpoint))
        first = policy.predict(
            sample["depth"].numpy(),
            sample["velocity"].numpy(),
            sample["goal_direction"].numpy(),
        )
        second = policy.predict(
            sample["depth"].numpy(),
            sample["velocity"].numpy(),
            sample["goal_direction"].numpy(),
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertLessEqual(
            np.linalg.norm(first[:3]),
            DEFAULT_CONTRACT.action_total_limit_mps + 1e-6,
        )
        self.assertLessEqual(
            abs(first[3]), DEFAULT_CONTRACT.yaw_rate_limit_radps + 1e-6
        )

    def test_normalized_one_step_is_finite(self) -> None:
        dataset = BcEpisodeDataset(self.dataset_path, "train")
        stats = compute_normalization(dataset)
        normalizer = TorchNormalizer(stats, torch.device("cpu"))
        batch = dataset[0]
        depth, velocity, goal = normalizer.observation(
            batch["depth"].unsqueeze(0),
            batch["velocity"].unsqueeze(0),
            batch["goal_direction"].unsqueeze(0),
        )
        target = normalizer.normalize_action(batch["action"].unsqueeze(0))
        model = BcPolicyV0()
        loss = torch.nn.functional.mse_loss(model(depth, velocity, goal), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

