"""Focused CPU tests for single-episode BC diagnostics and artifact isolation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from test_bc_baseline import _make_fixture, _sha256, _write_json
from uav_ml.datasets.expert_image_dataset import IMAGE_PREPROCESSING
from uav_ml.inference.bc_flight import resolve_checkpoint
from uav_ml.inference.rgb_encoder import load_frozen_encoder
from uav_ml.models import LatentBcPolicy
from uav_ml.tools.bc_baseline import _episode_arrays, _run_loader
from uav_ml.tools.bc_overfit import (
    EVALUATION_SCOPE,
    FORMAT_VERSION,
    MILESTONE_EPOCHS,
    OverfitConfig,
    _parser,
    audit_single_episode,
    main,
    offline_action_metrics,
    train_single_episode,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _diagnostic_fixture(root: Path) -> tuple[Path, Path]:
    """Extend the existing BC fixture with three aligned retained pairs."""
    dataset, encoder = _make_fixture(root, episode_count=2, encoder_source="top")
    payload = torch.load(encoder, map_location="cpu", weights_only=False)
    payload["metadata"].update(dataset_root=str(dataset), image_preprocessing=IMAGE_PREPROCESSING["top"])
    torch.save(payload, encoder)
    _write_json(root / "summary.json", {
        "run_status": "completed", "dataset_root": str(dataset), "image_source": "top",
        "encoder_architecture": "RgbAutoencoderV0", "image_preprocessing": IMAGE_PREPROCESSING["top"],
        "latent_dimension": 64,
    })
    episode_dir = dataset / "episode_000001"
    episode = json.loads((episode_dir / "episode.json").read_text())
    episode.update(sample_count=3, terminal_flight_status={"state": "COMPLETE"}, accumulated_flight_evidence={
        name: True for name in ("goal_reached", "landing_commanded", "landed_after_landing_command", "terminal_complete")
    })
    _write_json(episode_dir / "episode.json", episode)
    validation = json.loads((episode_dir / "validation.json").read_text())
    validation["sample_count"] = 3
    _write_json(episode_dir / "validation.json", validation)
    with (episode_dir / "samples.csv").open(newline="") as stream:
        sample_template = next(csv.DictReader(stream))
    with (episode_dir / "auxiliary.csv").open(newline="") as stream:
        auxiliary_template = next(csv.DictReader(stream))
    samples, auxiliary = [], []
    for index, forward in enumerate((0.5, -0.5, 0.005), start=1):
        timestamp = 1.0 + 0.2 * index
        row = dict(sample_template, sample_id=str(index), image_timestamp_s=str(timestamp),
                   state_timestamp_s=str(timestamp + 0.005), expert_action_timestamp_s=str(timestamp + 0.009),
                   previous_action_timestamp_s=str(timestamp - 0.011), state_image_error_s="0.005",
                   expert_action_image_error_s="0.009", expert_action_forward=str(forward),
                   expert_action_right=str(-0.1 * index), expert_action_yaw_rate=str(0.05 * index),
                   body_velocity_forward_mps=str(0.1 * index), previous_action_yaw_rate="0.0",
                   mission_phase="TRACKING", success="True", failure="")
        joined = dict(auxiliary_template, sample_id=str(index), primary_image_timestamp_s=str(timestamp),
                      observer_rgb_timestamp_s=str(timestamp), fpv_depth_timestamp_s=str(timestamp))
        for source, path_field, destination in (
            ("fpv_rgb", "image_path", row), ("observer_rgb", "observer_rgb_path", joined)
        ):
            path = episode_dir / source / f"frame_{index:06d}.jpg"
            Image.fromarray(np.full((18, 32, 3), 40 * index, dtype=np.uint8)).save(path)
            destination[path_field] = str(path.relative_to(dataset))
        samples.append(row)
        auxiliary.append(joined)
    _write_csv(episode_dir / "samples.csv", samples)
    _write_csv(episode_dir / "auxiliary.csv", auxiliary)
    return dataset, encoder


def _output(root: Path, name: str) -> Path:
    return root / "experiments/diagnostics/bc_single_episode" / name


class BcOverfitTests(unittest.TestCase):
    def test_selection_uses_all_requested_pairs_without_requiring_three_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, _ = _diagnostic_fixture(root)
            audit = audit_single_episode(dataset, "episode_000001")
            self.assertEqual(audit["sample_count"], 3)
            self.assertEqual(audit["sample_ids"], [1, 2, 3])
            self.assertEqual(audit["evaluation_scope"], EVALUATION_SCOPE)
            self.assertFalse(any("episode_000002" in path for path in audit["files_sha256"]))
            collection_path = dataset / "collection_manifest.json"
            collection = json.loads(collection_path.read_text())
            collection["episodes"] = collection["episodes"][:1]
            _write_json(collection_path, collection)
            self.assertEqual(audit_single_episode(dataset, "episode_000001")["sample_count"], 3)
            for episode_id in ("episode_000002", "../episode_000001"):
                with self.subTest(episode_id=episode_id), self.assertRaises(ValueError):
                    audit_single_episode(dataset, episode_id)

    def test_bad_samples_alignment_and_corrupt_top_images_fail_loudly(self) -> None:
        for failure in ("nan_state", "timestamp", "identity", "top_alignment", "corrupt_top"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                dataset, _ = _diagnostic_fixture(Path(temporary))
                episode_dir = dataset / "episode_000001"
                with (episode_dir / "samples.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                with (episode_dir / "auxiliary.csv").open(newline="") as stream:
                    auxiliary = list(csv.DictReader(stream))
                if failure == "nan_state":
                    rows[0]["body_velocity_forward_mps"] = "nan"
                elif failure == "timestamp":
                    rows[0]["state_timestamp_s"] = "100.0"
                elif failure == "identity":
                    rows[1]["sample_id"] = "1"
                elif failure == "top_alignment":
                    auxiliary[0]["observer_rgb_timestamp_s"] = str(float(rows[0]["image_timestamp_s"]) + 0.002)
                    auxiliary[0]["observer_rgb_error_s"] = "0.002"
                else:
                    (dataset / auxiliary[0]["observer_rgb_path"]).write_bytes(b"invalid JPEG")
                _write_csv(episode_dir / "samples.csv", rows)
                _write_csv(episode_dir / "auxiliary.csv", auxiliary)
                with self.assertRaises((ValueError, OSError)):
                    audit_single_episode(dataset, "episode_000001")

    def test_success_evidence_and_retained_counts_are_required(self) -> None:
        for failure in ("failed_episode", "invalid_validation", "missing_goal", "sample_count"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                dataset, _ = _diagnostic_fixture(Path(temporary))
                episode_dir = dataset / "episode_000001"
                episode = json.loads((episode_dir / "episode.json").read_text())
                if failure == "failed_episode":
                    episode.update(status="failed", success=False)
                elif failure == "invalid_validation":
                    _write_json(episode_dir / "validation.json", {"episode_id": "episode_000001", "valid": False})
                elif failure == "missing_goal":
                    episode["accumulated_flight_evidence"]["goal_reached"] = False
                else:
                    episode["sample_count"] = 2
                _write_json(episode_dir / "episode.json", episode)
                with self.assertRaises(ValueError):
                    audit_single_episode(dataset, "episode_000001")

    def test_encoder_provenance_and_expected_sample_count_fail_before_output(self) -> None:
        for failure in ("wrong_source", "wrong_latent", "wrong_count"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset, encoder = _diagnostic_fixture(root)
                payload = torch.load(encoder, map_location="cpu", weights_only=False)
                if failure == "wrong_source":
                    payload["metadata"]["image_source"] = "fpv_rgb"
                elif failure == "wrong_latent":
                    payload["model_config"]["latent_dimension"] = 32
                torch.save(payload, encoder)
                output = _output(root, "invalid")
                with self.assertRaises(ValueError):
                    train_single_episode(dataset, "episode_000001", encoder, output,
                                         OverfitConfig(epochs=2, expected_samples=24 if failure == "wrong_count" else 3), project_root=root)
                self.assertFalse(output.exists())

    def test_offline_metrics_units_maximum_error_and_near_zero_sign_mask(self) -> None:
        target = torch.tensor([[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [0.005, 0, 0], [0, 0, 0]])
        predicted = torch.tensor([[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [-0.005, 0.5, 0.5], [0, 0, 0]])
        metrics = offline_action_metrics(predicted, target, 0.01)
        self.assertAlmostEqual(metrics["training_mse"], float((predicted.double() - target.double()).square().mean()))
        self.assertEqual(metrics["normalized"]["per_action"]["v_forward"]["maximum_absolute_error"], 1.0)
        self.assertEqual(metrics["forward_sign"], {
            "epsilon_normalized": 0.01, "eligible_samples": 2, "excluded_samples": 2, "correct_samples": 1, "accuracy": 0.5,
        })
        for name, scale in zip(("v_forward", "v_right", "yaw_rate"), (1.0, 0.8, 1.0)):
            for metric in ("mae", "rmse", "maximum_absolute_error"):
                self.assertAlmostEqual(metrics["physical"][name][metric], metrics["normalized"]["per_action"][name][metric] * scale)
        zeros = offline_action_metrics(torch.ones(2, 3), torch.zeros(2, 3), 0.01)
        self.assertIsNone(zeros["forward_sign"]["accuracy"])
        with self.assertRaises(ValueError):
            offline_action_metrics(torch.full((2, 3), float("nan")), torch.zeros(2, 3), 0.01)

    def test_short_training_is_reproducible_frozen_and_preserves_formal_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, encoder = _diagnostic_fixture(root)
            formal = root / "experiments/bc/dataset/top/run_formal"
            formal.mkdir(parents=True)
            (formal / "best.pt").write_bytes(b"original formal checkpoint")
            _write_json(formal / "summary.json", {"run_status": "completed"})
            pointer = {"format_version": "bc_baseline_v1.0", "updated_utc": "2026-10-13T00:00:00Z", "best_checkpoint": str(formal / "best.pt")}
            _write_json(root / "experiments/bc/latest.json", pointer)
            _write_json(formal.parent / "latest.json", pointer)
            _write_json(root / "experiments/autoencoder/latest.json", {"sentinel": True})
            before = {path: _sha256(path) for path in root.rglob("*") if path.is_file()}
            frozen_models = []

            def capture_encoder(path, device):
                model, payload = load_frozen_encoder(path, device)
                frozen_models.append((model, {name: value.clone() for name, value in model.state_dict().items()}))
                return model, payload

            deterministic_before = torch.are_deterministic_algorithms_enabled()
            with mock.patch("uav_ml.tools.bc_overfit.load_frozen_encoder", side_effect=capture_encoder), mock.patch(
                "uav_ml.tools.bc_overfit.MILESTONE_EPOCHS", (1, 2)
            ):
                summaries = [train_single_episode(dataset, "episode_000001", encoder, _output(root, name),
                                                  OverfitConfig(epochs=2, batch_size=2, expected_samples=3), project_root=root)
                             for name in ("first", "second")]
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic_before)
            for path, digest in before.items():
                self.assertEqual(_sha256(path), digest, str(path))
            self.assertEqual(resolve_checkpoint(root), formal / "best.pt")
            self.assertFalse(list((root / "experiments/diagnostics").rglob("latest.json")))
            for model, snapshot in frozen_models:
                self.assertFalse(model.training)
                self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()))
                self.assertTrue(all(torch.equal(model.state_dict()[name], value) for name, value in snapshot.items()))
            first, second = (Path(summary["output_dir"]) for summary in summaries)
            self.assertEqual((first / "training_history.csv").read_bytes(), (second / "training_history.csv").read_bytes())
            best = torch.load(first / "best.pt", map_location="cpu", weights_only=False)
            repeated = torch.load(second / "best.pt", map_location="cpu", weights_only=False)
            self.assertTrue(all(torch.equal(value, repeated["model_state"][name]) for name, value in best["model_state"].items()))
            self.assertEqual(best["format_version"], FORMAT_VERSION)
            self.assertNotIn("validation_metrics", best)
            self.assertEqual(tuple(best["observation_mean"].shape), (72,))
            self.assertAlmostEqual(float(best["observation_std"][-1]), 1e-6)
            with (first / "training_history.csv").open(newline="") as stream:
                history = list(csv.DictReader(stream))
            self.assertEqual(summaries[0]["best_training_mse"], min(float(row["train_mse"]) for row in history))
            self.assertEqual(summaries[0]["epochs_completed"], 2)
            self.assertEqual(len(summaries[0]["plots"]), 5)
            for path in summaries[0]["plots"]:
                with Image.open(path) as image:
                    image.verify()
            for name in ("experiment_config.json", "episode_audit.json", "observation_normalization.json", "offline_metrics.json",
                         "action_predictions.csv", "best.pt", "last.pt", "summary.json", "epoch_0001_metrics.json", "epoch_0002_metrics.json"):
                self.assertTrue((first / name).is_file(), name)
            with (first / "action_predictions.csv").open(newline="") as stream:
                predictions = list(csv.DictReader(stream))
            self.assertEqual([int(row["sample_id"]) for row in predictions], [1, 2, 3])
            observation, target = _episode_arrays(dataset, "episode_000001", frozen_models[0][0], torch.device("cpu"), 128, "top")
            reloaded = LatentBcPolicy()
            reloaded.load_state_dict(best["model_state"])
            with torch.no_grad():
                expected = reloaded((torch.from_numpy(observation) - best["observation_mean"]) / best["observation_std"]).numpy()
            actual = np.array([[float(row[f"{name}_predicted_normalized"]) for name in ("v_forward", "v_right", "yaw_rate")] for row in predictions])
            np.testing.assert_allclose(actual, expected, atol=1e-7)
            np.testing.assert_allclose([float(row["v_forward_expert_normalized"]) for row in predictions], target[:, 0])

    def test_checkpoint_selection_uses_post_epoch_same_set_mse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, encoder = _diagnostic_fixture(root)
            offsets = iter((0.2, 0.05, 0.1, 0.05))

            def controlled_evaluation(model, loader, device, optimizer=None):
                if optimizer is not None:
                    return _run_loader(model, loader, device, optimizer)
                targets = torch.cat([batch[1] for batch in loader])
                prediction = targets + next(offsets)
                return {}, prediction, targets

            with mock.patch("uav_ml.tools.bc_overfit._run_loader", side_effect=controlled_evaluation):
                summary = train_single_episode(dataset, "episode_000001", encoder, _output(root, "selection"),
                                               OverfitConfig(epochs=3), project_root=root)
            self.assertEqual(summary["best_epoch"], 2)
            best = torch.load(summary["best_checkpoint"], map_location="cpu", weights_only=False)
            last = torch.load(summary["last_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(best["epoch"], 2)
            self.assertEqual(last["epoch"], 3)
            self.assertLess(summary["best_training_mse"], summary["last_training_mse"])

    def test_existing_output_and_formal_directories_are_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, encoder = _diagnostic_fixture(root)
            existing = _output(root, "existing")
            existing.mkdir(parents=True)
            sentinel = existing / "best.pt"
            sentinel.write_bytes(b"keep existing checkpoint")
            formal = root / "experiments/bc"
            formal.mkdir(parents=True)
            linked = _output(root, "linked")
            linked.symlink_to(formal, target_is_directory=True)
            for output in (existing, formal / "run", dataset / "run", linked / "run"):
                with self.subTest(output=output), self.assertRaises((ValueError, FileExistsError)):
                    train_single_episode(dataset, "episode_000001", encoder, output, OverfitConfig(epochs=1), project_root=root)
            self.assertEqual(sentinel.read_bytes(), b"keep existing checkpoint")
            self.assertEqual(list(formal.iterdir()), [])

    def test_config_limits_and_cli_defaults_match_the_requested_experiment(self) -> None:
        for change in ({"epochs": 0}, {"epochs": 1001}, {"batch_size": 0}, {"learning_rate": float("nan")},
                       {"seed": 2**32}, {"forward_sign_epsilon": float("nan")}, {"expected_samples": 0}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                OverfitConfig(**change).validate()
        args = _parser().parse_args(["--episode-id", "episode_000001", "--encoder", "best.pt", "--expected-samples", "24"])
        self.assertEqual((args.dataset, args.epochs, args.batch_size, args.learning_rate, args.seed, args.device),
                         ("cylinder_v4", 1000, 64, 0.001, 614420090, "cpu"))
        self.assertEqual(MILESTONE_EPOCHS, (100, 300, 500, 1000))
        self.assertNotIn("--initialize-policy", _parser().format_help())
        self.assertNotIn("--patience", _parser().format_help())
        with tempfile.TemporaryDirectory() as temporary, mock.patch("sys.argv", [
            "bc-overfit", "--dataset", temporary, "--episode-id", "episode_000001", "--encoder", "best.pt", "--expected-samples", "24",
        ]), mock.patch("uav_ml.tools.bc_overfit.train_single_episode", return_value={"output_dir": "mock-run", "best_training_mse": 0.01}) as train:
            self.assertEqual(main(), 0)
            self.assertEqual(train.call_args.args[4], OverfitConfig(expected_samples=24))

    def test_shell_cli_help_routes_to_diagnostic_without_training(self) -> None:
        result = subprocess.run([str(REPOSITORY_ROOT / "uav"), "bc-overfit", "--help"], cwd=REPOSITORY_ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("./uav bc-overfit", result.stdout)
        self.assertIn("--episode-id", result.stdout)


if __name__ == "__main__":
    unittest.main()
