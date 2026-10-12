"""Regression tests for shared dataset and TensorBoard CLI helpers."""

from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

import torch

from uav_ml.datasets.expert_image_dataset import IMAGE_PREPROCESSING
from uav_ml.models import RgbAutoencoderV0

from uav_ml.tools.expert_collect import (
    _parser as expert_parser,
    main as expert_main,
)
from uav_ml.tools.bc_baseline import _parser as bc_parser, main as bc_main
from uav_ml.tools.training_cli import (
    TensorBoardServer,
    autoencoder_latest_path,
    experiment_run_directory,
    publish_autoencoder_latest,
    resolve_dataset,
    select_autoencoder_checkpoint,
)
from uav_ml.train_autoencoder import (
    _parser as autoencoder_parser,
    main as autoencoder_main,
)


class _FakeProcess:
    def __init__(self) -> None:
        self.pid = 12345
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return None if not self.terminated and not self.killed else 0

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout=None) -> int:
        self.waited = True
        return 0


class TrainingCliTests(unittest.TestCase):
    def test_resnet_training_uses_requested_dataset_and_its_own_experiment_path(self) -> None:
        defaults = bc_parser(resnet_training=True).parse_args([])
        self.assertEqual(defaults.encoder_type, "resnet18_trainable")
        self.assertNotIn("--encoder-type", bc_parser(resnet_training=True).format_help())
        with tempfile.TemporaryDirectory(prefix="resnet-cli-") as temporary:
            dataset = resolve_dataset(temporary, must_exist=True)
            with mock.patch("sys.argv", [
                "resnet-train", "--dataset", temporary, "--epochs", "3",
                "--no-tensorboard", "--encoder-type", "resnet18_trainable",
            ]), mock.patch("uav_ml.tools.bc_baseline.TensorBoardServer"), mock.patch(
                "uav_ml.tools.bc_baseline.train_baseline", side_effect=KeyboardInterrupt,
            ) as train:
                self.assertEqual(bc_main(), 130)
            self.assertEqual(train.call_args.args[0], dataset.path)
            self.assertIsNone(train.call_args.args[1])
            self.assertEqual(train.call_args.args[2].parts[-4], "resnet")
            self.assertEqual(train.call_args.args[3].epochs, 3)
            self.assertEqual(train.call_args.kwargs["encoder_type"], "resnet18_trainable")

    def test_resnet_autoencoder_cli_keeps_original_defaults_and_separate_latest_index(self) -> None:
        self.assertEqual(autoencoder_parser().parse_args([]).encoder_backbone, "cnn")
        args = autoencoder_parser().parse_args(["--encoder-backbone", "resnet18"])
        self.assertEqual(args.encoder_backbone, "resnet18")
        self.assertEqual(args.latent_dimension, 64)
        with tempfile.TemporaryDirectory(prefix="resnet-ae-cli-") as temporary:
            dataset = resolve_dataset(temporary, must_exist=True)
            with mock.patch(
                "sys.argv", ["ae-train", "--dataset", temporary,
                             "--encoder-backbone", "resnet18", "--no-tensorboard"],
            ), mock.patch("uav_ml.train_autoencoder.TensorBoardServer"), mock.patch(
                "uav_ml.train_autoencoder.train", side_effect=KeyboardInterrupt,
            ) as train:
                self.assertEqual(autoencoder_main(), 130)
            options = train.call_args.kwargs
            self.assertEqual(options["encoder_backbone"], "resnet18")
            self.assertEqual(options["output_dir"].parent.name, "resnet18")
            original_index = autoencoder_latest_path(dataset, "top")
            self.assertEqual(
                options["latest_index_path"],
                original_index.parent / "resnet18" / "latest.json",
            )
            self.assertFalse(original_index.exists())

    def test_ae_and_bc_accept_the_same_episode_subset_arguments(self) -> None:
        arguments = [
            "--episodes", "50", "--train-episodes", "30",
            "--validation-episodes", "10", "--test-episodes", "10",
        ]
        for parser_factory in (autoencoder_parser, bc_parser):
            with self.subTest(parser=parser_factory.__module__):
                args = parser_factory().parse_args(arguments)
                self.assertEqual(
                    (args.episodes, args.train_episodes, args.validation_episodes, args.test_episodes),
                    (50, 30, 10, 10),
                )
                defaults = parser_factory().parse_args([])
                self.assertIsNone(defaults.episodes)
                self.assertIsNone(defaults.train_episodes)
                self.assertIsNone(defaults.validation_episodes)
                self.assertIsNone(defaults.test_episodes)

    def test_short_and_explicit_dataset_paths_resolve_identically(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dataset-resolution-") as temporary:
            project = Path(temporary)
            expected = project / "datasets" / "bc_expert_cube"
            expected.mkdir(parents=True)
            short = resolve_dataset(
                "bc_expert_cube", must_exist=True, project_root=project
            )
            relative = resolve_dataset(
                "datasets/bc_expert_cube",
                must_exist=True,
                project_root=project,
            )
            absolute = resolve_dataset(
                expected, must_exist=True, project_root=project
            )
            legacy = project / "uav_vision_dataset"
            legacy.mkdir()
            explicit_dot = resolve_dataset(
                "./uav_vision_dataset", must_exist=True, project_root=project
            )
            self.assertEqual(short.path, expected.resolve())
            self.assertEqual(short, relative)
            self.assertEqual(relative, absolute)
            self.assertEqual(short.name, "bc_expert_cube")
            self.assertEqual(explicit_dot.path, legacy.resolve())
            run = experiment_run_directory(
                "autoencoder", short, "top", "STAMP", project_root=project
            )
            self.assertEqual(
                run,
                project / "experiments" / "autoencoder"
                / "bc_expert_cube" / "top" / "run_STAMP",
            )

    def test_collection_allows_new_name_but_training_requires_existing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="dataset-existence-") as temporary:
            project = Path(temporary)
            location = resolve_dataset(
                "new_collection", must_exist=False, project_root=project
            )
            self.assertEqual(
                location.path,
                (project / "datasets" / "new_collection").resolve(),
            )
            self.assertFalse(location.path.exists())
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                resolve_dataset(
                    "new_collection", must_exist=True, project_root=project
                )

    @mock.patch("uav_ml.tools.expert_collect.ExpertCollector")
    @mock.patch(
        "sys.argv",
        ["expert-collect", "--episodes", "1", "--dataset", "bc_expert_cube"],
    )
    def test_expert_collect_passes_resolved_new_dataset_to_collector(
        self, collector_class
    ) -> None:
        collector_class.return_value.run.return_value = {
            "valid": True,
            "status": "fixture",
        }
        self.assertEqual(expert_main(), 0)
        dataset_root = collector_class.call_args.kwargs["dataset_root"]
        repository_root = Path(__file__).resolve().parents[2]
        self.assertEqual(
            dataset_root,
            repository_root / "datasets" / "bc_expert_cube",
        )

    def test_legacy_dataset_root_remains_an_explicit_path_alias(self) -> None:
        args = autoencoder_parser().parse_args([
            "--dataset-root", "uav_vision_dataset", "--no-tensorboard"
        ])
        self.assertEqual(args.dataset_root, "uav_vision_dataset")
        self.assertIsNone(args.dataset)
        self.assertFalse(args.tensorboard)
        collection = expert_parser().parse_args([
            "--episodes", "100", "--dataset", "bc_expert_cube"
        ])
        self.assertEqual(collection.dataset, "bc_expert_cube")
        ae = autoencoder_parser().parse_args(["--dataset", "bc_expert_cube"])
        bc = bc_parser().parse_args(["--dataset", "bc_expert_cube"])
        self.assertEqual(ae.dataset, bc.dataset)
        self.assertTrue(ae.tensorboard)
        self.assertTrue(bc.tensorboard)
        self.assertEqual(ae.image_source, "top")
        self.assertEqual(bc.image_source, "top")
        self.assertIsNone(ae.output_dir)
        self.assertIsNone(bc.encoder)
        self.assertIsNone(bc.output)

    def test_matching_autoencoder_uses_only_completed_provenance_index(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ae-selection-") as temporary:
            project = Path(temporary)
            dataset_path = project / "datasets" / "cube"
            dataset_path.mkdir(parents=True)
            dataset = resolve_dataset(
                "cube", must_exist=True, project_root=project
            )
            run = experiment_run_directory(
                "autoencoder", dataset, "top", "STAMP", project_root=project
            )
            run.mkdir(parents=True)
            model = RgbAutoencoderV0()
            checkpoint = run / "best.pt"
            metadata = {
                "dataset_name": "cube",
                "dataset_root": str(dataset.path),
                "image_source": "top",
                "image_preprocessing": IMAGE_PREPROCESSING["top"],
                "encoder_architecture": "RgbAutoencoderV0",
                "latent_dimension": 64,
            }
            torch.save({
                "model_class": "RgbAutoencoderV0",
                "model_config": model.config.to_dict(),
                "model_state": model.state_dict(),
                "metadata": metadata,
            }, checkpoint)
            summary_path = run / "summary.json"
            summary = {
                **metadata,
                "run_status": "completed",
                "checkpoint_reload_verified": True,
                "artifacts": {
                    "best_checkpoint": str(checkpoint.resolve()),
                    "summary": str(summary_path.resolve()),
                },
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            index_path = autoencoder_latest_path(
                dataset, "top", project_root=project
            )
            publish_autoencoder_latest(index_path, summary)
            selection = select_autoencoder_checkpoint(
                dataset,
                "top",
                IMAGE_PREPROCESSING["top"],
                explicit=None,
                project_root=project,
            )
            self.assertTrue(selection.automatic)
            self.assertEqual(selection.checkpoint, checkpoint.resolve())
            explicit = select_autoencoder_checkpoint(
                dataset,
                "top",
                IMAGE_PREPROCESSING["top"],
                explicit=checkpoint,
                project_root=project,
            )
            self.assertFalse(explicit.automatic)
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                select_autoencoder_checkpoint(
                    dataset,
                    "fpv_rgb",
                    IMAGE_PREPROCESSING["fpv_rgb"],
                    explicit=checkpoint,
                    project_root=project,
                )
            wrong_path = project / "datasets" / "other"
            wrong_path.mkdir()
            wrong_dataset = resolve_dataset(
                "other", must_exist=True, project_root=project
            )
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                select_autoencoder_checkpoint(
                    wrong_dataset,
                    "top",
                    IMAGE_PREPROCESSING["top"],
                    explicit=checkpoint,
                    project_root=project,
                )

    @mock.patch(
        "sys.argv",
        [
            "ae-train",
            "--dataset",
            "definitely_missing_training_dataset",
            "--no-tensorboard",
        ],
    )
    def test_nonexistent_autoencoder_dataset_fails_before_training(self) -> None:
        self.assertEqual(autoencoder_main(), 1)

    @mock.patch("uav_ml.tools.training_cli.subprocess.Popen")
    def test_no_tensorboard_does_not_start_subprocess(self, popen) -> None:
        with tempfile.TemporaryDirectory(prefix="tensorboard-disabled-") as temporary:
            with TensorBoardServer(
                Path(temporary) / "tensorboard", enabled=False, port=6006
            ) as server:
                server.wait_until_interrupted()
            popen.assert_not_called()

    @mock.patch("uav_ml.tools.training_cli._port_available", return_value=True)
    @mock.patch("uav_ml.tools.training_cli.shutil.which", return_value="/bin/tensorboard")
    @mock.patch("uav_ml.tools.training_cli.subprocess.Popen")
    def test_tensorboard_subprocess_is_owned_and_reaped(
        self, popen, which, port_available
    ) -> None:
        process = _FakeProcess()
        popen.return_value = process
        with tempfile.TemporaryDirectory(prefix="tensorboard-lifecycle-") as temporary:
            event_file = Path(temporary) / "tensorboard" / "events.fixture"
            with TensorBoardServer(
                Path(temporary) / "tensorboard", enabled=True, port=6007
            ) as server:
                self.assertIs(server.process, process)
                event_file.write_text("event", encoding="utf-8")
            self.assertTrue(process.terminated)
            self.assertTrue(process.waited)
            self.assertFalse(process.killed)
            self.assertTrue(event_file.is_file())

    @mock.patch(
        "uav_ml.tools.training_cli.time.sleep", side_effect=KeyboardInterrupt
    )
    @mock.patch("uav_ml.tools.training_cli._port_available", return_value=True)
    @mock.patch("uav_ml.tools.training_cli.shutil.which", return_value="/bin/tensorboard")
    @mock.patch("uav_ml.tools.training_cli.subprocess.Popen")
    def test_completed_training_wait_exits_cleanly_on_ctrl_c(
        self, popen, which, port_available, sleep
    ) -> None:
        process = _FakeProcess()
        popen.return_value = process
        with tempfile.TemporaryDirectory(prefix="tensorboard-wait-") as temporary:
            with TensorBoardServer(
                Path(temporary) / "tensorboard", enabled=True, port=6008
            ) as server:
                server.wait_until_interrupted()
            self.assertTrue(process.terminated)
            self.assertTrue(process.waited)

    @mock.patch("uav_ml.tools.training_cli._port_available", return_value=False)
    @mock.patch("uav_ml.tools.training_cli.subprocess.Popen")
    def test_occupied_port_is_not_killed_or_reused(self, popen, port_available) -> None:
        with tempfile.TemporaryDirectory(prefix="tensorboard-port-") as temporary:
            with TensorBoardServer(
                Path(temporary) / "tensorboard", enabled=True, port=6006
            ) as server:
                server.wait_until_interrupted()
            popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
