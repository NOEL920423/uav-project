"""Regression tests for the formal BC training and evaluation tools."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from uav_ml.models import RgbAutoencoderV0
from uav_ml.datasets.expert_image_dataset import (
    preprocess_expert_image,
    select_episode_images,
)
from uav_ml.tools.bc_baseline import (
    DEFAULT_SEED,
    STATE_FIELDS,
    TARGET_FIELDS,
    TrainingConfig,
    audit_dataset,
    create_episode_split,
    train_baseline,
)
from uav_ml.tools.bc_evaluation import (
    BcPolicyRuntime,
    run_closed_loop_evaluation,
    unseen_evaluation_seeds,
)
from uav_ml.train_autoencoder import train as train_autoencoder


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_fixture(
    root: Path, episode_count: int = 10, encoder_source: str = "fpv_rgb"
) -> tuple[Path, Path]:
    dataset = root / "dataset"
    dataset.mkdir()
    encoder = RgbAutoencoderV0()
    encoder_path = root / "encoder.pt"
    torch.save(
        {
            "format_version": "rgb_autoencoder_checkpoint_v0.1",
            "model_class": "RgbAutoencoderV0",
            "model_config": encoder.config.to_dict(),
            "model_state": encoder.state_dict(),
            "metadata": {"fixture": True, "image_source": encoder_source},
        },
        encoder_path,
    )
    encoder_hash = _sha256(encoder_path)
    entries = []
    for index in range(1, episode_count + 1):
        episode_id = f"episode_{index:06d}"
        success = index != 3
        entries.append({
            "episode_id": episode_id,
            "seed": 1000 + index,
            "status": "complete" if success else "failed",
            "success": success,
        })
        if not success:
            continue
        episode_dir = dataset / episode_id
        image_dir = episode_dir / "fpv_rgb"
        image_dir.mkdir(parents=True)
        image_path = image_dir / "frame_000001.jpg"
        pixels = np.full((18, 32, 3), 20 * index, dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(image_path)
        top_dir = episode_dir / "observer_rgb"
        depth_dir = episode_dir / "fpv_depth"
        top_dir.mkdir()
        depth_dir.mkdir()
        top_path = top_dir / "frame_000001.jpg"
        depth_path = depth_dir / "frame_000001.png"
        Image.fromarray(pixels[:, ::-1].copy(), mode="RGB").save(top_path)
        Image.fromarray(
            np.full((18, 32), 1000 + index, dtype=np.uint16)
        ).save(depth_path)
        _write_json(episode_dir / "episode.json", {
            "episode_id": episode_id,
            "status": "complete",
            "success": True,
            "available_sensor_streams": {
                "runtime_status": {
                    "phase10c_observer_mode": "fixed_global_top"
                }
            },
        })
        _write_json(episode_dir / "validation.json", {
            "episode_id": episode_id,
            "valid": True,
            "episode_success": True,
            "sample_count": 1,
            "autoencoder_checkpoint_sha256": encoder_hash,
        })
        row = {
            **{name: 0.01 * index for name in STATE_FIELDS},
            **{name: 0.02 * index for name in TARGET_FIELDS},
            "image_path": str(image_path.relative_to(dataset)),
            "episode_id": episode_id,
            "sample_id": "1",
            "image_timestamp_s": "1.0",
            "state_timestamp_s": "1.0",
            "expert_action_timestamp_s": "1.0",
            "state_image_error_s": "0.0",
            "expert_action_image_error_s": "0.0",
        }
        with (episode_dir / "samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        auxiliary = {
            "episode_id": episode_id,
            "sample_id": "1",
            "primary_image_timestamp_s": "1.0",
            "observer_rgb_available": "True",
            "observer_rgb_timestamp_s": "1.0",
            "observer_rgb_error_s": "0.0",
            "observer_rgb_path": str(top_path.relative_to(dataset)),
            "observer_rgb_status": "matched",
            "fpv_depth_available": "True",
            "fpv_depth_timestamp_s": "1.0",
            "fpv_depth_error_s": "0.0",
            "fpv_depth_path": str(depth_path.relative_to(dataset)),
            "fpv_depth_status": "matched",
        }
        with (episode_dir / "auxiliary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(auxiliary))
            writer.writeheader()
            writer.writerow(auxiliary)
    _write_json(dataset / "collection_manifest.json", {
        "status": "complete",
        "episodes": entries,
    })
    _write_json(dataset / "collection_validation.json", {"valid": True})
    return dataset, encoder_path


class MockClosedLoopEnvironment:
    """Small deterministic policy-only environment fixture."""

    def __init__(self) -> None:
        self.seed = 0
        self.step_index = 0
        self.closed = False

    def _observation(self) -> dict:
        state = np.zeros(8, dtype=np.float32)
        state[4] = max(0.0, (2.0 - 0.5 * self.step_index) / 10.0)
        return {
            "rgb": np.zeros((72, 128, 3), dtype=np.uint8),
            "state": state,
            "position_xy": np.asarray([0.1 * self.step_index, 0.0]),
        }

    def reset(self, *, seed: int) -> tuple[dict, dict]:
        self.seed = seed
        self.step_index = 0
        return self._observation(), {
            "goal_distance_m": 2.0,
            "position_xy": np.zeros(2),
        }

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        self.step_index += 1
        self.last_action = np.asarray(action)
        done = self.step_index == 2
        success = done and self.seed % 3 == 0
        collision = done and self.seed % 3 == 1
        timeout = done and self.seed % 3 == 2
        observation = self._observation()
        return observation, 0.0, success or collision, timeout, {
            "success": success,
            "collision": collision,
            "out_of_bounds": False,
            "timeout": timeout,
            "goal_distance_m": 1.0,
            "position_xy": observation["position_xy"],
            "sim_time_s": 0.2,
        }

    def close(self) -> None:
        self.closed = True


class BcBaselineTests(unittest.TestCase):
    def test_audit_excludes_failures_and_split_has_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bc-baseline-audit-") as temporary:
            dataset, encoder = _make_fixture(Path(temporary))
            audit = audit_dataset(dataset, encoder)
            self.assertEqual(audit["total_episodes"], 10)
            self.assertEqual(audit["usable_episodes"], 9)
            self.assertEqual(audit["excluded_episodes"], 1)
            self.assertEqual(
                audit["exclusion_reasons"], {"episode_not_successful": 1}
            )
            first = create_episode_split(audit, DEFAULT_SEED)
            second = create_episode_split(audit, DEFAULT_SEED)
            self.assertEqual(first, second)
            groups = list(first["splits"].values())
            members = [item for group in groups for item in group]
            self.assertEqual(len(members), len(set(members)))
            self.assertNotIn("episode_000003", members)
            self.assertEqual(first["episode_counts"], {
                "train": 7,
                "validation": 1,
                "test": 1,
            })
            corrupt_image = dataset.joinpath(
                "episode_000004", "fpv_rgb", "frame_000001.jpg"
            )
            corrupt_image.unlink()
            corrupt_audit = audit_dataset(dataset, encoder)
            self.assertEqual(corrupt_audit["usable_episodes"], 8)
            self.assertEqual(corrupt_audit["excluded_episodes"], 2)
            self.assertEqual(corrupt_audit["exclusion_reasons"], {
                "episode_not_successful": 1,
                "invalid_or_corrupt": 1,
            })

    def test_image_sources_share_split_and_alignment_failure_is_loud(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bc-source-selection-") as temporary:
            root = Path(temporary)
            dataset, encoder = _make_fixture(root)
            manifests = []
            for source in ("fpv_rgb", "top", "fpv_depth"):
                audit = audit_dataset(dataset, encoder, source)
                manifests.append(create_episode_split(audit, 77)["splits"])
                selected = select_episode_images(
                    dataset, "episode_000001", source
                )
                tensor = preprocess_expert_image(
                    selected[0]["image_path"], source
                )
                self.assertEqual(tuple(tensor.shape), (3, 72, 128))
                self.assertGreaterEqual(float(tensor.min()), 0.0)
                self.assertLessEqual(float(tensor.max()), 1.0)
            self.assertEqual(manifests[0], manifests[1])
            self.assertEqual(manifests[1], manifests[2])

            auxiliary_path = dataset / "episode_000001" / "auxiliary.csv"
            with auxiliary_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["primary_image_timestamp_s"] = "2.0"
            with auxiliary_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "failed loudly"):
                audit_dataset(dataset, encoder, "top")

    def test_tiny_training_writes_reloadable_checkpoints_metrics_and_plots(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bc-baseline-train-") as temporary:
            root = Path(temporary)
            dataset, encoder = _make_fixture(root)
            output = root / "run"
            summary = train_baseline(
                dataset,
                encoder,
                output,
                TrainingConfig(
                    epochs=2,
                    batch_size=4,
                    learning_rate=1e-3,
                    early_stopping_patience=2,
                    seed=7,
                ),
                "cpu",
                encode_batch_size=4,
            )
            for name in (
                "best.pt",
                "last.pt",
                "dataset_audit.json",
                "split_manifest.json",
                "training_config.json",
                "training_history.csv",
                "metrics.json",
                "summary.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(len(summary["plots"]), 3)
            self.assertTrue(all(Path(path).is_file() for path in summary["plots"]))
            checkpoint = torch.load(
                output / "best.pt", map_location="cpu", weights_only=False
            )
            self.assertTrue(checkpoint["encoder_frozen"])
            self.assertIsInstance(checkpoint["observation_mean"], torch.Tensor)
            self.assertIsInstance(checkpoint["observation_std"], torch.Tensor)
            self.assertEqual(checkpoint["model_config"]["observation_dimension"], 72)
            self.assertEqual(checkpoint["model_config"]["action_dimension"], 3)
            self.assertEqual(checkpoint["image_source"], "fpv_rgb")
            training_config = json.loads(
                (output / "training_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(training_config["dataset_name"], "dataset")
            self.assertEqual(training_config["dataset_path"], str(dataset.resolve()))
            self.assertFalse(training_config["tensorboard_enabled"])
            self.assertEqual(training_config["maximum_epochs_requested"], 2)
            self.assertEqual(training_config["actual_epochs_trained"], 2)
            self.assertEqual(
                training_config["best_epoch"], summary["best_epoch"]
            )
            self.assertFalse(training_config["early_stopping_triggered"])
            self.assertTrue((output / "tensorboard").is_dir())
            self.assertTrue(any((output / "tensorboard").iterdir()))
            accumulator = EventAccumulator(str(output / "tensorboard"))
            accumulator.Reload()
            scalar_tags = set(accumulator.Tags()["scalars"])
            self.assertTrue({
                "bc/train_action_loss",
                "bc/validation_action_loss",
                "bc/test_forward_rmse",
                "bc/test_right_rmse",
                "bc/test_yaw_rate_rmse",
            }.issubset(scalar_tags))
            runtime = BcPolicyRuntime(output / "best.pt", torch.device("cpu"))
            self.assertFalse(any(
                parameter.requires_grad for parameter in runtime.encoder.parameters()
            ))
            action = runtime.act({
                "rgb": np.zeros((72, 128, 3), dtype=np.uint8),
                "state": np.zeros(8, dtype=np.float32),
            })
            self.assertEqual(action.shape, (3,))
            self.assertTrue(np.isfinite(action).all())

    def test_tiny_top_autoencoder_uses_reconstruction_loss_and_tensorboard(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ae-top-train-") as temporary:
            root = Path(temporary)
            dataset, _ = _make_fixture(root, encoder_source="top")
            output = root / "ae-run"
            summary = train_autoencoder(
                dataset_root=dataset,
                split_file=None,
                output_dir=output,
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                latent_dimension=64,
                device_name="cpu",
                seed=7,
                workers=0,
                image_source="top",
                image_log_interval=1,
            )
            self.assertEqual(summary["image_source"], "top")
            self.assertEqual(summary["encoder_architecture"], "RgbAutoencoderV0")
            self.assertEqual(summary["latent_dimension"], 64)
            self.assertEqual(summary["dataset_name"], "dataset")
            self.assertFalse(summary["tensorboard_enabled"])
            self.assertEqual(summary["maximum_epochs_requested"], 1)
            self.assertEqual(summary["actual_epochs_trained"], 1)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertFalse(summary["early_stopping_triggered"])
            self.assertEqual(summary["run_status"], "completed")
            for name in (
                "training_config.json",
                "dataset_audit.json",
                "split_manifest.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertTrue((output / "reconstruction_loss_curves.png").is_file())
            accumulator = EventAccumulator(str(output / "tensorboard"))
            accumulator.Reload()
            self.assertIn(
                "ae/train_reconstruction_loss", accumulator.Tags()["scalars"]
            )
            self.assertIn(
                "ae/validation_reconstruction_loss",
                accumulator.Tags()["scalars"],
            )
            self.assertIn(
                "ae/top/validation_original_vs_reconstructed",
                accumulator.Tags()["images"],
            )

    def test_training_interrupt_closes_ae_and_bc_summary_writers(self) -> None:
        with tempfile.TemporaryDirectory(prefix="training-interrupt-") as temporary:
            root = Path(temporary)
            dataset, encoder = _make_fixture(root)
            with mock.patch(
                "uav_ml.train_autoencoder.SummaryWriter"
            ) as ae_writer_class, mock.patch(
                "uav_ml.train_autoencoder._epoch",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    train_autoencoder(
                        dataset_root=dataset,
                        split_file=None,
                        output_dir=root / "ae-interrupted",
                        epochs=2,
                        batch_size=4,
                        workers=0,
                        image_source="top",
                    )
                ae_writer_class.return_value.flush.assert_called()
                ae_writer_class.return_value.close.assert_called()

            with mock.patch(
                "uav_ml.tools.bc_baseline.SummaryWriter"
            ) as bc_writer_class, mock.patch(
                "uav_ml.tools.bc_baseline._run_loader",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    train_baseline(
                        dataset,
                        encoder,
                        root / "bc-interrupted",
                        TrainingConfig(epochs=2, batch_size=4),
                        "cpu",
                        encode_batch_size=4,
                    )
                bc_writer_class.return_value.flush.assert_called()
                bc_writer_class.return_value.close.assert_called()

    def test_autoencoder_early_stopping_records_maximum_actual_and_best(self) -> None:
        def metrics(mse: float) -> dict[str, float]:
            return {"mse": mse, "mae": mse, "psnr_db": 10.0}

        with tempfile.TemporaryDirectory(prefix="ae-early-stop-") as temporary:
            root = Path(temporary)
            dataset, _ = _make_fixture(root)
            sequence = [
                metrics(0.3), metrics(0.2), metrics(0.1),
                metrics(0.3), metrics(0.2), metrics(0.2),
                metrics(0.15), metrics(0.1),
            ]
            with mock.patch(
                "uav_ml.train_autoencoder._epoch", side_effect=sequence
            ):
                summary = train_autoencoder(
                    dataset_root=dataset,
                    split_file=None,
                    output_dir=root / "ae-early-stop",
                    epochs=5,
                    batch_size=4,
                    workers=0,
                    image_source="top",
                    early_stopping_patience=1,
                )
            self.assertEqual(summary["maximum_epochs_requested"], 5)
            self.assertEqual(summary["actual_epochs_trained"], 2)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertTrue(summary["early_stopping_triggered"])

    def test_closed_loop_records_measured_metrics_and_policy_ownership(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bc-baseline-eval-") as temporary:
            root = Path(temporary)
            dataset, _ = _make_fixture(root)
            seeds = unseen_evaluation_seeds(dataset, 3, seed_base=2000)
            self.assertEqual(seeds, [2000, 2001, 2002])
            checkpoint = root / "best.pt"
            encoder = root / "encoder-proof.pt"
            checkpoint.write_bytes(b"checkpoint")
            encoder.write_bytes(b"encoder")
            environment = MockClosedLoopEnvironment()
            output = root / "evaluation"
            result = run_closed_loop_evaluation(
                environment=environment,
                policy=lambda observation: np.zeros(3, dtype=np.float32),
                seeds=seeds,
                output_dir=output,
                checkpoint=checkpoint,
                checkpoint_sha256=_sha256(checkpoint),
                encoder_checkpoint=encoder,
                encoder_sha256=_sha256(encoder),
                dataset_root=dataset,
                progress_interval_s=999.0,
            )
            self.assertTrue(environment.closed)
            self.assertEqual(result["control_source"], "BC_POLICY")
            self.assertEqual(result["expert_action_calls"], 0)
            self.assertFalse(result["action_blending"])
            self.assertEqual(result["aggregate"]["attempted_episodes"], 3)
            self.assertEqual(result["aggregate"]["successful_episodes"], 1)
            self.assertEqual(result["aggregate"]["collision_count"], 1)
            self.assertEqual(result["aggregate"]["timeout_count"], 1)
            self.assertGreater(result["records"][0]["path_length_m"], 0.0)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue(all(Path(path).is_file() for path in result["plots"]))

    def test_cli_help_is_available_without_training_or_isaac_startup(self) -> None:
        for command in (("bc-train", "--help"), ("bc-eval", "--help")):
            completed = subprocess.run(
                [str(REPOSITORY_ROOT / "uav"), *command],
                cwd=REPOSITORY_ROOT,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("usage:", completed.stdout)


if __name__ == "__main__":
    unittest.main()
