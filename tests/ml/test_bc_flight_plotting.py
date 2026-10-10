"""Regression tests for formal BC artifact plotting."""

import json
from datetime import datetime
from io import StringIO
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image
from uav_ml.tools import bc_flight_plotting as plotting
from uav_ml.tools.bc_flight_plotting import (
    PLOT_FILENAMES,
    create_grouped_trajectory_plots,
    create_summary_plots,
    create_training_loss_artifacts,
    generate_evaluation_plots,
    main,
)
from matplotlib import colors, pyplot as plt
from matplotlib.patches import Rectangle


class BcFlightPlottingTests(unittest.TestCase):
    def _result(self, episode: int, reason: str) -> dict:
        return {
            "schema": "uav_bc_flight_result/v1",
            "episode": episode,
            "seed": 900000 + episode - 1,
            "image_source": "top_rgb",
            "success": reason == "success",
            "collision": reason == "collision",
            "timeout": reason == "timeout",
            "out_of_bounds": reason == "out_of_bounds",
            "terminal_reason": reason,
            "minimum_goal_distance_m": 0.2 + episode,
            "final_goal_distance_m": 0.4 + episode,
            "path_length_m": 2.0 + episode,
            "episode_duration_s": 10.0 + episode,
            "trace_file": "trajectory_trace.json",
        }

    def _trace(self, episode: int) -> dict:
        return {
            "schema": "uav_bc_flight_trace/v1",
            "episode": episode,
            "seed": 900000 + episode - 1,
            "image_source": "top_rgb",
            "coordinate_frame": "px4_ned",
            "action_contract": (
                "normalized_body_forward_right_yaw_v1.0"
            ),
            "start": {"north_m": 0.0, "east_m": 0.0},
            "goal": {"north_m": 5.0, "east_m": 3.0},
            "obstacles": [{
                "name": "Obstacle_001",
                "north_m": 2.0,
                "east_m": 1.0,
                "radius_m": 0.4,
            }],
            "uav_radius_m": 0.25,
            "collision_clearance_threshold_m": 0.0,
            "samples": [{
                "sample": index,
                "time_s": float(index) * 0.1,
                "inference_step": index,
                "north_m": float(index) * 0.1,
                "east_m": float(index) * 0.05,
                "down_m": -1.5,
                "yaw_rad": 0.1 * index,
                "goal_distance_m": 5.8 - 0.1 * index,
                "obstacle_clearance_m": 1.0 - 0.1 * index,
                "action_forward": 0.8,
                "action_right": 0.1 * index,
                "action_yaw_rate": -0.05 * index,
                "command_north_mps": 0.8,
                "command_east_mps": 0.1,
            } for index in range(5)],
        }

    def test_generates_episode_and_summary_plots(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            records = [
                self._result(1, "collision"),
                self._result(2, "success"),
            ]
            for result in records:
                episode = int(result["episode"])
                episode_dir = run_dir / f"episode_{episode:06d}"
                episode_dir.mkdir()
                (episode_dir / "result.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
                (episode_dir / "trajectory_trace.json").write_text(
                    json.dumps(self._trace(episode)), encoding="utf-8"
                )
            plots = generate_evaluation_plots(run_dir)
            self.assertEqual(set(plots["episodes"]), {
                "episode_000001", "episode_000002"
            })
            self.assertTrue(all(
                Path(path).is_file()
                for paths in plots["episodes"].values()
                for path in paths
            ))
            self.assertTrue(all(
                Path(path).is_file() for path in plots["summary"]
            ))
            episode_names = {
                Path(path).name
                for path in plots["episodes"]["episode_000001"]
            }
            self.assertEqual(episode_names, {
                "trajectory_1.png",
            })
            self.assertEqual(
                {path.name for path in (run_dir / "trajectory_plots").iterdir()},
                {"trajectory_1.png", "trajectory_2.png", "trajectory_001-004.png"},
            )
            self.assertEqual(
                [Path(path).name for path in plots["trajectory_groups"]],
                ["trajectory_001-004.png"],
            )
            for episode in (1, 2):
                self.assertFalse((run_dir / f"episode_{episode:06d}" / "plots").exists())
            self.assertEqual({Path(path).name for path in plots["summary"]}, {
                PLOT_FILENAMES["outcomes"], PLOT_FILENAMES["status"],
            })

    def test_missing_trace_does_not_create_fake_episode_plots(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            episode_dir = run_dir / "episode_000001"
            episode_dir.mkdir()
            result = self._result(1, "collision")
            (episode_dir / "result.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            plots = generate_evaluation_plots(run_dir)
            self.assertEqual(plots["episodes"], {})
            self.assertEqual(plots["trajectory_groups"], [])
            self.assertEqual({Path(path).name for path in plots["summary"]}, {
                PLOT_FILENAMES["outcomes"], PLOT_FILENAMES["status"],
            })

    def test_twenty_flights_form_five_ordered_groups_with_distinct_lines(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            records = [self._result(episode, "success") for episode in range(20, 0, -1)]
            for result in records:
                episode = result["episode"]
                episode_dir = run_dir / f"episode_{episode:06d}"
                episode_dir.mkdir()
                trace = self._trace(episode)
                # Duplicate decisions and invalid coordinates must not add points.
                trace["samples"].append({"east_m": 99, "north_m": 99, "inference_step": 0})
                trace["samples"].append({"east_m": None, "north_m": 1})
                (episode_dir / "trajectory_trace.json").write_text(json.dumps(trace))
            captured = []

            def capture(figure, path):
                lines = figure.axes[0].lines
                captured.append((
                    [line.get_label() for line in lines],
                    [line.get_color() for line in lines],
                    [list(line.get_xdata()) for line in lines],
                ))
                plt.close(figure)
                return str(path)

            with patch.object(plotting, "_save", side_effect=capture):
                paths = create_grouped_trajectory_plots(run_dir, records)
            self.assertEqual([Path(path).name for path in paths], [
                "trajectory_001-004.png", "trajectory_005-008.png",
                "trajectory_009-012.png", "trajectory_013-016.png",
                "trajectory_017-020.png",
            ])
            for group_index, (labels, line_colors, coordinates) in enumerate(captured):
                self.assertEqual(labels, [
                    f"Episode {episode:03d}: Success"
                    for episode in range(group_index * 4 + 1, group_index * 4 + 5)
                ])
                self.assertEqual(len(set(line_colors)), 4)
                self.assertTrue(all(
                    colors.to_rgb(color)[1] > max(colors.to_rgb(color)[::2])
                    for color in line_colors
                ))
                for points in coordinates:
                    self.assertEqual(len(points), 5)
                    for actual, expected in zip(points, [0, 0.05, 0.1, 0.15, 0.2]):
                        self.assertAlmostEqual(actual, expected)

    def test_grouped_lines_use_outcome_colors_and_control_issue_dashes(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            reasons = ("timeout", "success", "collision", "runtime_failure")
            records = [self._result(episode, reason)
                       for episode, reason in enumerate(reasons, 1)]
            for result in records:
                episode = result["episode"]
                episode_dir = run_dir / f"episode_{episode:06d}"
                episode_dir.mkdir()
                (episode_dir / "trajectory_trace.json").write_text(
                    json.dumps(self._trace(episode))
                )
            captured = []

            def capture(figure, path):
                axis = figure.axes[0]
                captured.extend((colors.to_rgb(line.get_color()), line.get_linestyle())
                                for line in axis.lines)
                # Shared walls, obstacle, collision envelope, start, and goal are drawn once.
                self.assertEqual(len(axis.patches), 8)
                walls = [patch for patch in axis.patches if isinstance(patch, Rectangle)]
                self.assertEqual(len(walls), 4)
                self.assertEqual(
                    [(wall.get_x(), wall.get_y(), wall.get_width(), wall.get_height())
                     for wall in walls],
                    [(-5.25, -2.0, 0.25, 9.0), (5.0, -2.0, 0.25, 9.0),
                     (-5.25, -2.25, 10.5, 0.25), (-5.25, 7.0, 10.5, 0.25)],
                )
                self.assertEqual(sum(wall.get_label() == "Arena walls" for wall in walls), 1)
                figure.canvas.draw()
                self.assertTrue(axis.get_autoscalex_on())
                self.assertTrue(axis.get_autoscaley_on())
                self.assertLess(axis.get_xlim()[0], -5.25)
                self.assertGreater(axis.get_xlim()[1], 5.25)
                self.assertLess(axis.get_ylim()[0], -2.25)
                self.assertGreater(axis.get_ylim()[1], 7.25)
                plt.close(figure)
                return str(path)

            with patch.object(plotting, "_save", side_effect=capture):
                create_grouped_trajectory_plots(run_dir, records)
            blue, green, red, purple = [color for color, _ in captured]
            self.assertGreater(blue[2], max(blue[:2]))
            self.assertGreater(green[1], max(green[::2]))
            self.assertGreater(red[0], max(red[1:]))
            self.assertGreater(min(purple[::2]), purple[1])
            self.assertEqual([style for _, style in captured], ["-", "-", "-", "--"])

    def test_missing_or_empty_traces_keep_original_episode_groups(self) -> None:
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            records = [self._result(episode, "timeout") for episode in (1, 2, 3, 5)]
            for episode in (1, 3, 5):
                episode_dir = run_dir / f"episode_{episode:06d}"
                episode_dir.mkdir()
                trace = self._trace(episode)
                if episode == 3:
                    trace["samples"] = []
                (episode_dir / "trajectory_trace.json").write_text(json.dumps(trace))
            captured = []

            def capture(figure, path):
                captured.append([line.get_label() for line in figure.axes[0].lines])
                plt.close(figure)
                return str(path)

            with patch.object(plotting, "_save", side_effect=capture):
                paths = create_grouped_trajectory_plots(run_dir, records)
            self.assertEqual([Path(path).name for path in paths], [
                "trajectory_001-004.png", "trajectory_005-008.png",
            ])
            self.assertEqual(captured, [["Episode 001: Timeout"], ["Episode 005: Timeout"]])

    def test_summary_displays_three_outcomes_and_integer_counts(self) -> None:
        with TemporaryDirectory() as temporary:
            records = [self._result(episode, reason) for episode, reason in enumerate(
                ("success", "timeout", "collision", "out_of_bounds", "runtime_failure"), 1
            )]
            captured = []

            def capture(figure, path):
                if path.name == "summary.png":
                    figure.canvas.draw()
                    axis = figure.axes[0]
                    captured.append(True)
                    self.assertEqual([label.get_text() for label in axis.get_xticklabels()],
                                     ["Success", "Timeout", "Collision"])
                    self.assertEqual([bar.get_height() for bar in axis.patches], [1, 1, 1])
                    self.assertTrue(all(float(tick).is_integer() for tick in axis.get_yticks()))
                    self.assertTrue(all("." not in label.get_text()
                                        for label in axis.get_yticklabels()))
                    self.assertEqual([text.get_text() for text in axis.texts], ["1 (20.0%)"] * 3)
                    self.assertEqual(axis.title.get_fontsize(), plotting.SUMMARY_TITLE_FONT_SIZE)
                    self.assertEqual(axis.yaxis.label.get_fontsize(), plotting.SUMMARY_LABEL_FONT_SIZE)
                    self.assertTrue(all(text.get_fontsize() == plotting.SUMMARY_COUNT_FONT_SIZE
                                        for text in axis.texts))
                    self.assertTrue(all(label.get_fontsize() == plotting.SUMMARY_TICK_FONT_SIZE
                                        for label in axis.get_xticklabels() + axis.get_yticklabels()))
                plt.close(figure)
                return str(path)

            with patch.object(plotting, "_save", side_effect=capture):
                create_summary_plots(Path(temporary), records)
            self.assertEqual(captured, [True])

    def test_offline_replot_preserves_recorded_test_duration(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps({"total_test_duration_s": 1234.5}))
            with patch.object(plotting, "generate_evaluation_plots", return_value={"episodes": {}}):
                self.assertEqual(main([str(root)]), 0)
            self.assertEqual(json.loads(summary_path.read_text()), {
                "total_test_duration_s": 1234.5, "plots": {"episodes": {}},
            })


class TrainingLossArtifactTests(unittest.TestCase):
    def _training_model(self, root: Path, name: str, architecture: str, duration: float) -> Path:
        directory = root / name
        directory.mkdir(parents=True)
        checkpoint = directory / "selected.pt"
        parameters = {
            "epochs": 20, "learning_rate": 0.0007, "batch_size": 8,
            "early_stopping_patience": 4, "min_epochs": 2, "seed": 7,
        }
        torch.save({
            "model_class": architecture, "epoch": 1, "training_config": parameters,
        }, checkpoint)
        (directory / "training_config.json").write_text(json.dumps({
            **parameters, "actual_epochs_trained": 3, "early_stopping_triggered": True,
            "training_duration_s": duration,
        }))
        (directory / "summary.json").write_text(json.dumps({
            "best_epoch": 3, "train_loss": 999, "training_duration_s": duration,
        }))
        history_name = "training_history.csv" if architecture == "LatentBcPolicy" else "history.csv"
        (directory / history_name).write_text(
            "epoch,train_mse,validation_mse\n1,3,1\n2,1,3\n3,2,0.5\n"
        )
        curve = directory / (
            "plots/loss_curves.png" if architecture == "LatentBcPolicy"
            else "reconstruction_loss_curves.png"
        )
        curve.parent.mkdir(exist_ok=True)
        Image.new("RGB", (8, 8), "green").save(curve)
        return checkpoint

    def test_exports_selected_bc_and_ae_curves_with_independent_best_epochs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc = self._training_model(root, "bc", "LatentBcPolicy", 12.5)
            ae = self._training_model(root, "ae", "RgbAutoencoderV0", 23.5)
            output = root / "evaluation"
            report = create_training_loss_artifacts(output, bc, ae, "RgbAutoencoderV0")
            saved = json.loads((output / "loss/loss_summary.json").read_text())
            self.assertEqual(saved, report)
            for name, checkpoint, duration in (("bc", bc, 12.5), ("ae", ae, 23.5)):
                info = report["models"][name]
                self.assertEqual(info["status"], "complete")
                self.assertEqual(info["checkpoint_path"], str(checkpoint))
                self.assertEqual(info["checkpoint_epoch"], 1)
                self.assertEqual(info["best_train_epoch"], 2)
                self.assertEqual(info["best_train_loss"], 1)
                self.assertEqual(info["best_validation_epoch"], 3)
                self.assertEqual(info["best_validation_loss"], 0.5)
                self.assertEqual(info["training_duration_s"], duration)
                self.assertEqual(info["parameters"], {
                    "maximum_epochs": 20, "learning_rate": 0.0007, "batch_size": 8,
                    "early_stopping_patience": 4, "minimum_epochs": 2, "seed": 7,
                })
                source = checkpoint.parent / (
                    "plots/loss_curves.png" if name == "bc" else "reconstruction_loss_curves.png"
                )
                self.assertEqual((output / info["loss_curve"]).read_bytes(), source.read_bytes())

    def test_missing_curve_is_regenerated_and_legacy_duration_is_explicit(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc = self._training_model(root, "bc", "LatentBcPolicy", 12.5)
            ae = self._training_model(root, "ae", "MultiImageAutoencoderV0", 23.5)
            (ae.parent / "reconstruction_loss_curves.png").unlink()
            for name in ("summary.json", "training_config.json"):
                path = ae.parent / name
                data = json.loads(path.read_text())
                del data["training_duration_s"]
                path.write_text(json.dumps(data))
            output = root / "evaluation"
            report = create_training_loss_artifacts(output, bc, ae, "MultiImageAutoencoderV0")
            info = report["models"]["ae"]
            self.assertEqual(info["status"], "partial")
            self.assertIsNone(info["training_duration_s"])
            self.assertEqual(info["missing_fields"], ["training_duration_s"])
            with Image.open(output / "loss/ae.png") as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 8)

    def test_frozen_resnet_does_not_inherit_bc_training_or_invent_losses(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc = self._training_model(root, "bc", "LatentBcPolicy", 12.5)
            encoder = bc.parent / "resnet18_encoder.pt"
            torch.save({"model_class": "ResNet18Encoder", "metadata": {"weights": "IMAGENET1K_V1"}}, encoder)
            output = root / "evaluation"
            report = create_training_loss_artifacts(output, bc, encoder, "ResNet18Encoder")
            info = report["models"]["resnet18"]
            self.assertEqual(info["status"], "pretrained_frozen")
            self.assertEqual(info["parameters"], {"weights": "IMAGENET1K_V1", "frozen": True})
            for field in ("best_train_epoch", "best_validation_epoch", "training_duration_s", "loss_curve"):
                self.assertIsNone(info[field])
            self.assertEqual({path.name for path in (output / "loss").iterdir()}, {"bc.png", "loss_summary.json"})

    def test_bad_model_history_preserves_original_error_and_other_model_evidence(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bc = self._training_model(root, "bc", "LatentBcPolicy", 12.5)
            ae = self._training_model(root, "ae", "RgbAutoencoderV0", 23.5)
            (ae.parent / "history.csv").write_text("epoch,train_mse,validation_mse\nbroken,1,2\n")
            stderr = StringIO()
            with redirect_stderr(stderr):
                report = create_training_loss_artifacts(root / "evaluation", bc, ae, "RgbAutoencoderV0")
            self.assertEqual(report["models"]["bc"]["status"], "complete")
            self.assertEqual(report["models"]["ae"]["status"], "error")
            self.assertIn("invalid literal for int()", stderr.getvalue())

    def test_default_directory_uses_loaded_encoder_taipei_time_and_bc_nas_category(self) -> None:
        from uav_ml.tools import bc_flight_evaluation as evaluation

        for architecture, label in (
            ("RgbAutoencoderV0", "AE"), ("MultiImageAutoencoderV0", "AE"),
            ("ResNet18Encoder", "ResNet18"),
        ):
            with self.subTest(architecture=architecture), TemporaryDirectory() as temporary:
                root = Path(temporary)
                bc = self._training_model(root, "misleading_resnet_folder", "LatentBcPolicy", 12.5)
                if label == "AE":
                    encoder = self._training_model(root, "encoder", architecture, 23.5)
                else:
                    encoder = bc.parent / "encoder.pt"
                    torch.save({"model_class": architecture, "metadata": {"weights": "IMAGENET1K_V1"}}, encoder)
                identity = SimpleNamespace(
                    checkpoint_path=str(bc), checkpoint_sha256="digest",
                    image_source="top_rgb",
                    encoder_path=str(encoder), image_sources=("top_rgb",),
                )
                policy = SimpleNamespace(identity=identity, encoder=type(architecture, (), {})())
                runtime = Mock()
                runtime.run_episode.return_value = {
                    "schema": "uav_bc_flight_result/v1", "episode": 1, "terminal_reason": "success",
                }
                output = root / "evaluations/bc" / f"{label}_1010T1530Y2026"
                with (
                    patch.object(evaluation, "__file__", str(root / "uav_ml/tools/bc_flight_evaluation.py")),
                    patch.object(evaluation, "_LAST_ARTIFACT_PATH", None),
                    patch.object(evaluation, "ENABLE_BC_FLIGHT_RECORDING", False),
                    patch.object(evaluation, "BcFlightPolicy", return_value=policy),
                    patch.object(evaluation, "ManagedFlightRuntime", return_value=runtime),
                    patch.object(evaluation, "datetime") as clock,
                    patch.object(plotting, "generate_evaluation_plots", return_value={}),
                    patch("uav_ml.tools.artifact_publish.publish_artifact_tree", return_value=root / "nas" / output.name) as publish,
                ):
                    clock.now.return_value = datetime(2026, 10, 10, 15, 30)
                    self.assertEqual(evaluation.main(["--checkpoint", str(bc), "--nas-root", str(root / "nas")]), 0)
                    self.assertEqual(str(clock.now.call_args.args[0]), "Asia/Taipei")
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["feature_extractor"], label)
                self.assertEqual(summary["encoder_architecture"], architecture)
                self.assertEqual(summary["encoder_checkpoint"], str(encoder))
                self.assertEqual(summary["loss_summary_file"], "loss/loss_summary.json")
                self.assertTrue((output / summary["loss_summary_file"]).is_file())
                publish.assert_called_once_with(output, root / "nas", "evaluations/bc", delete_local=True)


if __name__ == "__main__":
    unittest.main()
