"""Offline regression tests for the formal expert collection tool."""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from isaac.runtime.episode_scene import NUM_OBSTACLES, generate_episode_scene
from isaac.runtime.formal_expert_sensor_contract import (
    FORMAL_RGB_NOMINAL_RATE_HZ,
    FPV_RGB_HEIGHT,
    FPV_RGB_WIDTH,
    LEGACY_OBSERVER_RGB_HEIGHT,
    LEGACY_OBSERVER_RGB_WIDTH,
    TOP_RGB_ALIGNMENT_TOLERANCE_S,
    TOP_RGB_HEIGHT,
    TOP_RGB_MODE,
    TOP_RGB_WIDTH,
)
from uav_ml.tools.expert_collect import (
    AUXILIARY_FIELDS,
    CollectionManifestStore,
    DryRunBackend,
    EpisodeOutcome,
    ExpertCollector,
    ProgressDisplay,
    _parser,
    default_max_attempts,
)
from uav_ml.tools.expert_visual_qa import create_contact_sheet
from uav_ml.tools.validate_expert_batch import _validate_auxiliary
from uav_ml.tools.validate_expert_collection import (
    validate_episode_metadata,
    validate_cylinder_scene,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class InterruptingBackend(DryRunBackend):
    """Interrupt exactly once at the requested episode."""

    def __init__(self, interrupt_index: int) -> None:
        self.interrupt_index = interrupt_index
        self.seen: list[tuple[str, int]] = []

    def run_episode(self, **kwargs) -> int:
        self.seen.append((kwargs["episode_id"], kwargs["seed"]))
        if kwargs["index"] == self.interrupt_index:
            raise KeyboardInterrupt
        return super().run_episode(**kwargs)


class InfrastructureFailureBackend(DryRunBackend):
    """Fail from the managed runtime boundary."""

    def run_episode(self, **kwargs) -> int:
        raise RuntimeError("runtime readiness lost")


class ExpertCollectionToolTest(unittest.TestCase):
    """Exercise manifest safety, resume, scene freeze, progress, and QA."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _collector(
        self,
        dataset: Path,
        backend,
        *,
        resume: bool = False,
        episodes: int = 3,
        fixed_seed: int | None = None,
        max_attempts: int | None = None,
    ) -> ExpertCollector:
        return ExpertCollector(
            repository_root=REPOSITORY_ROOT,
            dataset_root=dataset,
            episodes=episodes,
            fixed_seed=fixed_seed,
            max_attempts=max_attempts,
            resume=resume,
            backend=backend,
            runtime_root=(
                self.root / ("resume-runtime" if resume else "runtime")
            ),
        )

    def test_dry_run_builds_unique_seeded_plan_and_finishes(self) -> None:
        dataset = self.root / "dataset"
        with contextlib.redirect_stdout(io.StringIO()):
            result = self._collector(dataset, DryRunBackend()).run()
        self.assertTrue(result["valid"])
        manifest = json.loads(
            (dataset / "collection_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["target_episodes"], 3)
        self.assertEqual(manifest["max_attempts"], 5)
        self.assertEqual(
            manifest["collection_runs"][0]["requested_accepted_episodes"], 3
        )
        self.assertEqual(
            [entry["seed"] for entry in manifest["episodes"]],
            [103001, 103002, 103003],
        )
        self.assertTrue(all(
            entry["status"] == "complete" for entry in manifest["episodes"]
        ))

    def test_fixed_seed_reaches_scene_generator_and_reproduces_scene(self) -> None:
        generated_scenes = []
        backends = []
        original = generate_episode_scene

        def generate(*args, **kwargs):
            scene = original(*args, **kwargs)
            generated_scenes.append(scene)
            return scene

        with mock.patch(
            "uav_ml.tools.expert_collect.generate_episode_scene",
            side_effect=generate,
        ), contextlib.redirect_stdout(io.StringIO()):
            for index in range(2):
                backend = InterruptingBackend(interrupt_index=99)
                backends.append(backend)
                self._collector(
                    self.root / f"fixed-seed-{index}",
                    backend,
                    episodes=1,
                    fixed_seed=103009,
                ).run()

        self.assertEqual(generated_scenes[0], generated_scenes[1])
        self.assertEqual(
            [backend.seen for backend in backends],
            [[("episode_000001", 103009)], [("episode_000001", 103009)]],
        )
        manifest = json.loads(
            (self.root / "fixed-seed-0" / "collection_manifest.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["fixed_seed"], 103009)
        self.assertEqual(manifest["max_attempts"], 1)
        self.assertEqual(manifest["episodes"][0]["seed"], 103009)

    def test_failed_fixed_seed_does_not_advance_to_another_seed(self) -> None:
        dataset = self.root / "failed-fixed-seed"
        backend = InterruptingBackend(interrupt_index=99)
        collector = self._collector(
            dataset,
            backend,
            episodes=1,
            fixed_seed=103009,
        )
        rejected = EpisodeOutcome(
            False, 0, 0, "collision produced black images", 0,
            "collision_tracking", {"valid": True, "episode_success": False},
        )
        with mock.patch.object(
            collector, "_dry_run_outcome", return_value=rejected
        ), contextlib.redirect_stdout(io.StringIO()):
            result = collector.run()

        self.assertFalse(result["valid"])
        self.assertTrue(result["dataset_incomplete"])
        self.assertEqual(backend.seen, [("episode_000001", 103009)])
        self.assertEqual(result["summary"]["attempted"], 1)
        self.assertEqual(result["summary"]["accepted"], 0)

    def test_fixed_seed_cli_requires_one_episode(self) -> None:
        args = _parser().parse_args([
            "--episodes", "1", "--seed", "103009",
        ])
        self.assertEqual(args.seed, 103009)
        with self.assertRaisesRegex(ValueError, "requires --episodes 1"):
            self._collector(
                self.root / "invalid-fixed-seed",
                DryRunBackend(),
                episodes=2,
                fixed_seed=103009,
            )

    def test_resume_keeps_completed_episode_and_reuses_no_seed(self) -> None:
        dataset = self.root / "dataset"
        interrupted = InterruptingBackend(interrupt_index=2)
        with self.assertRaises(KeyboardInterrupt), contextlib.redirect_stdout(
            io.StringIO()
        ):
            self._collector(dataset, interrupted).run()
        before = json.loads(
            (dataset / "collection_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(before["episodes"][0]["status"], "complete")
        self.assertEqual(before["episodes"][1]["status"], "interrupted")

        resumed = InterruptingBackend(interrupt_index=99)
        with contextlib.redirect_stdout(io.StringIO()):
            self._collector(dataset, resumed, resume=True).run()
        self.assertEqual(
            resumed.seen,
            [("episode_000002", 103002), ("episode_000003", 103003)],
        )
        after = json.loads(
            (dataset / "collection_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(after["status"], "complete")
        seeds = {entry["seed"] for entry in after["episodes"]}
        self.assertEqual(len(seeds), 3)

    def test_completed_collection_extends_dataset_wide_accepted_target(self) -> None:
        dataset = self.root / "dataset"
        with contextlib.redirect_stdout(io.StringIO()):
            self._collector(dataset, DryRunBackend()).run()
        before = json.loads(
            (dataset / "collection_manifest.json").read_text(encoding="utf-8")
        )
        first_three = [dict(entry) for entry in before["episodes"]]

        extended = InterruptingBackend(interrupt_index=99)
        with contextlib.redirect_stdout(io.StringIO()):
            self._collector(
                dataset, extended, resume=True, episodes=5
            ).run()
        self.assertEqual(
            extended.seen,
            [("episode_000004", 103004), ("episode_000005", 103005)],
        )
        after = json.loads(
            (dataset / "collection_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(after["requested_accepted_episodes"], 5)
        self.assertEqual(after["episodes"][:3], first_three)
        self.assertEqual(
            [entry["seed"] for entry in after["episodes"]],
            [103001, 103002, 103003, 103004, 103005],
        )
        self.assertEqual(
            [run["requested_episodes"] for run in after["collection_runs"]],
            [3, 5],
        )

    def test_completed_collection_appends_without_resume_flag(self) -> None:
        dataset = self.root / "dataset"
        with contextlib.redirect_stdout(io.StringIO()):
            self._collector(dataset, DryRunBackend()).run()

        appended = InterruptingBackend(interrupt_index=99)
        with contextlib.redirect_stdout(io.StringIO()):
            result = self._collector(
                dataset, appended, episodes=4
            ).run()

        self.assertEqual(appended.seen, [("episode_000004", 103004)])
        self.assertEqual(result["episode_count"], 4)

    def test_new_collection_never_overwrites_existing_directory(self) -> None:
        dataset = self.root / "dataset"
        dataset.mkdir()
        store = CollectionManifestStore(dataset)
        with self.assertRaises(FileExistsError):
            store.create(3, 103000)

    def test_invalid_scene_is_recorded_and_collection_continues(self) -> None:
        dataset = self.root / "dataset"
        original = generate_episode_scene

        def generate(episode_id, seed, reset_east_m, reset_north_m):
            if episode_id == "episode_000002":
                raise ValueError("fixture invalid scene")
            return original(
                episode_id, seed, reset_east_m, reset_north_m
            )

        with mock.patch(
            "uav_ml.tools.expert_collect.generate_episode_scene",
            side_effect=generate,
        ), contextlib.redirect_stdout(io.StringIO()):
            result = self._collector(dataset, DryRunBackend()).run()
        self.assertEqual(result["successful_episodes"], 3)
        self.assertEqual(result["failed_episodes"], 1)
        manifest = json.loads(
            (dataset / "collection_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["episodes"][1]["status"], "rejected")
        self.assertIn(
            "invalid_scene", manifest["episodes"][1]["terminal_reason"]
        )
        self.assertEqual(manifest["episodes"][2]["status"], "complete")
        self.assertEqual(manifest["episodes"][3]["seed"], 103004)
        self.assertEqual(manifest["episodes"][3]["status"], "complete")
        self.assertEqual(manifest["accepted_episode_ids"], [
            "episode_000001", "episode_000003", "episode_000004"
        ])
        rejection = json.loads(
            (dataset / "rejected_attempts" / "attempt_000002.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(rejection["seed"], 103002)
        self.assertEqual(rejection["failure_category"], "blocked_scene")
        self.assertIn("validation_result", rejection)
        self.assertIn("log_path", rejection)

    def test_invalid_scene_uses_formal_fpv_storage_directory(self) -> None:
        dataset = self.root / "dataset"
        backend = mock.Mock(produces_dataset=True)
        collector = self._collector(dataset, backend, episodes=1)
        with mock.patch(
            "uav_ml.tools.expert_collect.validate_collection_episode",
            return_value={"sample_count": 0},
        ):
            collector._record_invalid_scene(
                "episode_000001",
                103001,
                ValueError("fixture invalid scene"),
                None,
            )
        episode = dataset / "episode_000001"
        self.assertTrue((episode / "fpv_rgb").is_dir())
        self.assertFalse((episode / "images").exists())
        with (episode / "auxiliary.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            self.assertIn("observer_rgb_path", csv.DictReader(stream).fieldnames)

    def test_infrastructure_failure_leaves_resumable_manifest(self) -> None:
        dataset = self.root / "dataset"
        with self.assertRaisesRegex(RuntimeError, "readiness lost"), \
                contextlib.redirect_stdout(io.StringIO()):
            self._collector(dataset, InfrastructureFailureBackend()).run()
        manifest = json.loads(
            (dataset / "collection_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["status"], "stopped_infrastructure_failure"
        )
        self.assertEqual(
            manifest["episodes"][0]["status"], "infrastructure_failure"
        )
        summary = json.loads(
            (dataset / "collection_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["infrastructure_failures"], 1)

    def test_episode_failure_is_rejected_and_next_seed_is_attempted(self) -> None:
        dataset = self.root / "dataset"
        backend = InterruptingBackend(interrupt_index=99)
        collector = self._collector(
            dataset, backend, episodes=2, max_attempts=3
        )
        rejected = EpisodeOutcome(
            False, 17, 4, "tracking error caused collision", 0,
            "collision_tracking", {"valid": True, "episode_success": False},
        )
        accepted = EpisodeOutcome(
            True, 42, 2, "goal_reached_and_landed", 0,
            None, {"valid": True, "episode_success": True},
        )
        with mock.patch.object(
            collector,
            "_dry_run_outcome",
            side_effect=[rejected, accepted, accepted],
        ), contextlib.redirect_stdout(io.StringIO()):
            result = collector.run()
        self.assertTrue(result["valid"])
        self.assertEqual([seed for _, seed in backend.seen], [
            103001, 103002, 103003
        ])
        manifest = json.loads(
            (dataset / "collection_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["episodes"][0]["status"], "rejected")
        self.assertEqual(manifest["episodes"][0]["accepted_samples"], 0)
        self.assertEqual(manifest["accepted_episode_ids"], [
            "episode_000002", "episode_000003"
        ])
        evidence = json.loads(
            (dataset / "rejected_attempts" / "attempt_000001.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(evidence["failure_reason"], rejected.terminal_reason)

        (dataset / "dataset_manifest.json").write_text(json.dumps({
            "episodes": ["episode_000001", "episode_000002", "episode_000003"],
            "episode_count": 3,
            "sample_count": 999,
        }), encoding="utf-8")
        collector._reconcile_dataset_manifest()
        accepted_manifest = json.loads(
            (dataset / "dataset_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(accepted_manifest["episodes"], [
            "episode_000002", "episode_000003"
        ])
        self.assertEqual(accepted_manifest["episode_count"], 2)

    def test_max_attempts_returns_clear_incomplete_result(self) -> None:
        dataset = self.root / "dataset"
        backend = InterruptingBackend(interrupt_index=99)
        collector = self._collector(
            dataset, backend, episodes=3, max_attempts=3
        )
        rejected = EpisodeOutcome(
            False, 0, 0, "goal not reached", 0, "flight_failure",
            {"valid": True, "episode_success": False},
        )
        with mock.patch.object(
            collector, "_dry_run_outcome", return_value=rejected
        ), contextlib.redirect_stdout(io.StringIO()):
            result = collector.run()
        self.assertFalse(result["valid"])
        self.assertTrue(result["dataset_incomplete"])
        self.assertEqual(result["summary"]["attempted"], 3)
        self.assertEqual(result["summary"]["accepted"], 0)
        self.assertEqual(len(backend.seen), 3)

    def test_default_max_attempts_is_ceiling_of_one_point_five_times(self) -> None:
        self.assertEqual(default_max_attempts(1), 2)
        self.assertEqual(default_max_attempts(3), 5)
        self.assertEqual(default_max_attempts(100), 150)

    def test_image_validation_failure_is_rejectable_episode_outcome(self) -> None:
        dataset = self.root / "dataset"
        episode = dataset / "episode_000001"
        episode.mkdir(parents=True)
        (episode / "flight_evidence.json").write_text(
            "{}", encoding="utf-8"
        )
        collector = self._collector(dataset, DryRunBackend(), episodes=1)
        finalized = {
            "success": True,
            "terminal_reason": "goal_reached_and_landed",
            "rejected_sample_count": 0,
            "episode_disk_usage_bytes": 12,
        }
        with mock.patch(
            "uav_ml.tools.finalize_expert_episode.finalize",
            return_value=finalized,
        ), mock.patch(
            "uav_ml.tools.expert_collect.validate_collection_episode",
            side_effect=ValueError("FPV image is blank/dark"),
        ):
            outcome = collector._finalize_real_episode(
                "episode_000001", command_status=0
            )
        self.assertFalse(outcome.success)
        self.assertEqual(outcome.failure_category, "image_qa")
        self.assertIn("dataset_validation", outcome.terminal_reason)

    def test_visual_qa_data_failure_is_rejectable_but_internal_failure_raises(
        self,
    ) -> None:
        dataset = self.root / "dataset"
        collector = self._collector(dataset, DryRunBackend(), episodes=1)
        collector.store.create(1, 103000)
        with mock.patch(
            "uav_ml.tools.expert_collect.VISUAL_QA_INTERVAL", 1
        ), mock.patch(
            "uav_ml.tools.expert_collect.create_contact_sheet",
            side_effect=ValueError("fixture image QA failure"),
        ):
            error = collector._visual_qa(1, "episode_000001")
        self.assertEqual(error, "fixture image QA failure")

        with mock.patch(
            "uav_ml.tools.expert_collect.VISUAL_QA_INTERVAL", 1
        ), mock.patch(
            "uav_ml.tools.expert_collect.create_contact_sheet",
            side_effect=RuntimeError("fixture internal failure"),
        ), self.assertRaisesRegex(RuntimeError, "internal failure"):
            collector._visual_qa(1, "episode_000001")

    def test_resume_count_mismatch_does_not_corrupt_manifest(self) -> None:
        dataset = self.root / "dataset"
        store = CollectionManifestStore(dataset)
        store.create(3, 103000)
        store.set_collection_state("interrupted")
        with self.assertRaisesRegex(ValueError, "cannot reduce unfinished"):
            self._collector(
                dataset, DryRunBackend(), resume=True, episodes=2
            ).run()
        manifest = json.loads(
            (dataset / "collection_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["target_episodes"], 3)
        self.assertEqual(manifest["status"], "interrupted")
        self.assertEqual(len(manifest["episodes"]), 0)

    def test_resume_finishes_interrupted_validation_without_append(self) -> None:
        dataset = self.root / "dataset"
        with contextlib.redirect_stdout(io.StringIO()):
            self._collector(dataset, DryRunBackend()).run()
        store = CollectionManifestStore(dataset)
        store.data = json.loads(
            store.path.read_text(encoding="utf-8")
        )
        store.set_collection_state("interrupted", "validation interrupted")

        resumed = InterruptingBackend(interrupt_index=99)
        with contextlib.redirect_stdout(io.StringIO()):
            result = self._collector(
                dataset, resumed, resume=True, episodes=3
            ).run()

        self.assertEqual(resumed.seen, [])
        self.assertEqual(result["episode_count"], 3)
        manifest = json.loads(store.path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["target_episodes"], 3)
        self.assertEqual(len(manifest["collection_runs"]), 2)
        self.assertEqual(manifest["status"], "complete")

    def test_completed_legacy_manifest_is_migrated_before_append(self) -> None:
        dataset = self.root / "dataset"
        with contextlib.redirect_stdout(io.StringIO()):
            self._collector(dataset, DryRunBackend()).run()
        path = dataset / "collection_manifest.json"
        legacy = json.loads(path.read_text(encoding="utf-8"))
        legacy.pop("collection_runs")
        legacy.pop("active_run_number")
        legacy.pop("manifest_version")
        legacy.pop("requested_accepted_episodes")
        legacy["tool_version"] = "expert_collection_v1.0"
        legacy["status"] = "stopped_infrastructure_failure"
        path.write_text(json.dumps(legacy), encoding="utf-8")

        appended = InterruptingBackend(interrupt_index=99)
        with contextlib.redirect_stdout(io.StringIO()):
            self._collector(
                dataset, appended, resume=True, episodes=5
            ).run()

        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["target_episodes"], 5)
        self.assertEqual(appended.seen, [
            ("episode_000004", 103004),
            ("episode_000005", 103005),
        ])
        self.assertEqual(
            [run["requested_episodes"] for run in manifest["collection_runs"]],
            [5],
        )

    def test_scene_validator_enforces_frozen_cylinder_contract(self) -> None:
        scene = generate_episode_scene("episode_000001", 103001, 0.0, 0.0)
        result = validate_cylinder_scene(scene, "episode_000001", 103001)
        self.assertEqual(result["obstacle_count"], NUM_OBSTACLES)
        self.assertEqual(result["direct_path_blocker_count"], 2)
        wrong_count = json.loads(json.dumps(scene))
        wrong_count["obstacles"].pop()
        wrong_count["obstacle_count"] -= 1
        with self.assertRaisesRegex(
            ValueError, f"expected exactly {NUM_OBSTACLES} obstacles"
        ):
            validate_cylinder_scene(wrong_count, "episode_000001", 103001)
        changed = json.loads(json.dumps(scene))
        changed["obstacles"][0]["radius_basis_width"] = 0.9
        with self.assertRaisesRegex(ValueError, "out of range"):
            validate_cylinder_scene(changed, "episode_000001", 103001)

    def test_episode_id_has_no_six_digit_ceiling(self) -> None:
        scene = generate_episode_scene(
            "episode_1000000", 1103000, 0.0, 0.0
        )
        self.assertEqual(scene["episode_id"], "episode_1000000")

    def test_success_metadata_requires_paths_and_stream_rates(self) -> None:
        episode_id = "episode_000001"
        scene = generate_episode_scene(episode_id, 103001, 0.0, 0.0)
        episode = {
            "episode_id": episode_id,
            "random_seed": 103001,
            "scene_configuration": scene,
            "terminal_reason": "goal_reached_and_landed",
            "flight_duration_s": 42.0,
            "astar_path_information": {
                "validated_path": {
                    "point_count": 5,
                    "path_length_xy_m": 7.0,
                },
                "planner_status": "success",
            },
            "available_sensor_streams": {
                "fpv_rgb": {
                    "received": 210,
                    "observed_rate_hz": FORMAL_RGB_NOMINAL_RATE_HZ,
                },
                "fpv_depth": {"received": 209, "observed_rate_hz": 5.0},
                "observer_rgb": {"received": 84, "observed_rate_hz": 2.0},
            },
        }
        validation = {"episode_id": episode_id, "episode_success": True}
        result = validate_episode_metadata(episode, validation)
        self.assertEqual(result["stream_rates_hz"]["observer_rgb"], 2.0)
        formal = json.loads(json.dumps(episode))
        formal["available_sensor_streams"]["observer_rgb"][
            "observed_rate_hz"
        ] = FORMAL_RGB_NOMINAL_RATE_HZ
        formal["available_sensor_streams"]["observer_rgb"]["matched"] = 42
        formal["available_sensor_streams"]["runtime_status"] = {
            "phase10c_observer_mode": TOP_RGB_MODE,
        }
        validation["sample_count"] = 42
        result = validate_episode_metadata(formal, validation)
        self.assertEqual(
            result["stream_rates_hz"]["observer_rgb"],
            FORMAL_RGB_NOMINAL_RATE_HZ,
        )
        with self.assertRaisesRegex(ValueError, "observer_rgb"):
            validate_episode_metadata(formal | {
                "available_sensor_streams": {
                    **formal["available_sensor_streams"],
                    "observer_rgb": {
                        "received": 84,
                        "observed_rate_hz": 2.0,
                        "matched": 42,
                    },
                },
            }, validation)
        episode["available_sensor_streams"]["fpv_depth"][
            "observed_rate_hz"
        ] = 0.0
        with self.assertRaisesRegex(ValueError, "fpv_depth"):
            validate_episode_metadata(episode, validation)

    def test_progress_contains_required_live_fields(self) -> None:
        display = ProgressDisplay(100, started_monotonic=1.0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            display.render(
                index=37,
                seed=103037,
                state="TRACKING",
                success_count=35,
                failure_count=1,
                current_samples=42,
                current_rejected=113,
                total_samples=1426,
                dataset_bytes=48_300_000,
                force=True,
            )
        rendered = output.getvalue()
        for expected in (
            "Accepted: 35 / 100",
            "Attempts: 37",
            "seed     : 103037",
            "state    : TRACKING",
            "accepted : 35",
            "rejected : 1",
            "samples  : 42",
            "rejected : 113",
            "Elapsed",
            "ETA",
        ):
            self.assertIn(expected, rendered)

    def test_contact_sheet_contains_all_three_stream_rows(self) -> None:
        dataset = self.root / "dataset"
        episode = dataset / "episode_000020"
        for directory in ("fpv_rgb", "top_rgb", "fpv_depth"):
            (episode / directory).mkdir(parents=True, exist_ok=True)
        (episode / "episode.json").write_text(
            json.dumps({"success": True, "sample_count": 3}),
            encoding="utf-8",
        )
        samples = []
        auxiliary = []
        for index in range(1, 4):
            rgb = np.full((180, 320, 3), 30 * index, dtype=np.uint8)
            depth = np.full((180, 320), 1000 * index, dtype=np.uint16)
            image_relative = (
                f"episode_000020/fpv_rgb/frame_{index:06d}.jpg"
            )
            observer_relative = (
                f"episode_000020/top_rgb/frame_{index:06d}.jpg"
            )
            depth_relative = (
                f"episode_000020/fpv_depth/frame_{index:06d}.png"
            )
            Image.fromarray(rgb).save(dataset / image_relative, quality=85)
            Image.fromarray(rgb).save(dataset / observer_relative, quality=85)
            Image.fromarray(depth).save(dataset / depth_relative)
            samples.append({
                "episode_id": "episode_000020",
                "sample_id": index,
                "image_path": image_relative,
            })
            auxiliary.append({
                "episode_id": "episode_000020",
                "sample_id": index,
                "primary_image_timestamp_s": float(index),
                "observer_rgb_available": True,
                "observer_rgb_timestamp_s": float(index),
                "observer_rgb_error_s": 0.0,
                "observer_rgb_path": observer_relative,
                "observer_rgb_status": "matched",
                "fpv_depth_available": True,
                "fpv_depth_timestamp_s": float(index),
                "fpv_depth_error_s": 0.0,
                "fpv_depth_path": depth_relative,
                "fpv_depth_status": "matched",
            })
        for path, rows in (
            (episode / "samples.csv", samples),
            (episode / "auxiliary.csv", auxiliary),
        ):
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        result = create_contact_sheet(dataset, 20)
        availability = _validate_auxiliary(dataset, "episode_000020")
        output = dataset / result["contact_sheet"]
        self.assertTrue(output.is_file())
        with Image.open(output) as image:
            self.assertEqual(image.size, (960, 630))
        self.assertTrue(all(result["source_paths"]["fpv_depth"]))
        self.assertEqual(result["source_paths"]["fpv_rgb"], [
            f"episode_000020/fpv_rgb/frame_{index:06d}.jpg"
            for index in range(1, 4)
        ])
        self.assertEqual(result["source_paths"]["observer_rgb"], [
            f"episode_000020/top_rgb/frame_{index:06d}.jpg"
            for index in range(1, 4)
        ])
        self.assertNotIn("top_rgb", result["source_paths"])
        self.assertEqual(availability["observer_rgb_available"], 3)
        self.assertEqual(availability["fpv_depth_available"], 3)

    def test_formal_top_auxiliary_is_complete_aligned_and_not_reused(
        self,
    ) -> None:
        dataset = self.root / "dataset"
        episode_id = "episode_000021"
        episode = dataset / episode_id
        (episode / "top_rgb").mkdir(parents=True)
        (episode / "episode.json").write_text(json.dumps({
            "available_sensor_streams": {
                "runtime_status": {
                    "phase10c_observer_mode": TOP_RGB_MODE,
                },
            },
        }), encoding="utf-8")
        primary_timestamps = (1.0, 1.1)
        samples = [
            {
                "episode_id": episode_id,
                "sample_id": index,
                "image_timestamp_s": timestamp,
            }
            for index, timestamp in enumerate(primary_timestamps, start=1)
        ]
        auxiliary = []
        for index, timestamp in enumerate(primary_timestamps, start=1):
            relative = f"{episode_id}/top_rgb/frame_{index:06d}.jpg"
            image = np.full(
                (TOP_RGB_HEIGHT, TOP_RGB_WIDTH, 3),
                20 * index,
                dtype=np.uint8,
            )
            Image.fromarray(image).save(dataset / relative, quality=85)
            auxiliary.append({
                "episode_id": episode_id,
                "sample_id": index,
                "primary_image_timestamp_s": timestamp,
                "observer_rgb_available": True,
                "observer_rgb_timestamp_s": timestamp,
                "observer_rgb_error_s": 0.0,
                "observer_rgb_path": relative,
                "observer_rgb_status": "matched",
                "fpv_depth_available": False,
                "fpv_depth_timestamp_s": "",
                "fpv_depth_error_s": "",
                "fpv_depth_path": "",
                "fpv_depth_status": "stream_unavailable",
            })
        with (episode / "samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        with (episode / "auxiliary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=AUXILIARY_FIELDS)
            writer.writeheader()
            writer.writerows(auxiliary)

        availability = _validate_auxiliary(dataset, episode_id)
        self.assertEqual(availability["observer_rgb_available"], 2)

        samples[1]["image_timestamp_s"] = 1.0005
        auxiliary[1]["primary_image_timestamp_s"] = 1.0005
        auxiliary[1]["observer_rgb_timestamp_s"] = 1.0
        auxiliary[1]["observer_rgb_error_s"] = 0.0005
        with (episode / "samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        with (episode / "auxiliary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=AUXILIARY_FIELDS)
            writer.writeheader()
            writer.writerows(auxiliary)
        with self.assertRaisesRegex(ValueError, "timestamp was reused"):
            _validate_auxiliary(dataset, episode_id)

        auxiliary[1]["observer_rgb_available"] = False
        auxiliary[1]["observer_rgb_timestamp_s"] = ""
        auxiliary[1]["observer_rgb_error_s"] = ""
        auxiliary[1]["observer_rgb_path"] = ""
        auxiliary[1]["observer_rgb_status"] = "stream_unavailable"
        with (episode / "auxiliary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=AUXILIARY_FIELDS)
            writer.writeheader()
            writer.writerows(auxiliary)
        with self.assertRaisesRegex(ValueError, "formal TOP RGB is missing"):
            _validate_auxiliary(dataset, episode_id)

    def test_legacy_auxiliary_keeps_resolution_and_reuse_compatibility(
        self,
    ) -> None:
        dataset = self.root / "dataset"
        episode_id = "episode_000022"
        episode = dataset / episode_id
        (episode / "top_rgb").mkdir(parents=True)
        (episode / "episode.json").write_text(json.dumps({
            "available_sensor_streams": {
                "runtime_status": {"phase10c_observer_mode": "top"},
            },
        }), encoding="utf-8")
        samples = [
            {"episode_id": episode_id, "sample_id": 1,
             "image_timestamp_s": 1.0},
            {"episode_id": episode_id, "sample_id": 2,
             "image_timestamp_s": 1.2},
        ]
        image = np.full(
            (LEGACY_OBSERVER_RGB_HEIGHT, LEGACY_OBSERVER_RGB_WIDTH, 3),
            40,
            dtype=np.uint8,
        )
        auxiliary = []
        for index, sample in enumerate(samples, start=1):
            relative = f"{episode_id}/top_rgb/frame_{index:06d}.jpg"
            Image.fromarray(image).save(dataset / relative, quality=85)
            auxiliary.append({
                "episode_id": episode_id,
                "sample_id": index,
                "primary_image_timestamp_s": sample["image_timestamp_s"],
                "observer_rgb_available": True,
                "observer_rgb_timestamp_s": 1.0,
                "observer_rgb_error_s": abs(
                    sample["image_timestamp_s"] - 1.0
                ),
                "observer_rgb_path": relative,
                "observer_rgb_status": "matched",
                "fpv_depth_available": False,
                "fpv_depth_timestamp_s": "",
                "fpv_depth_error_s": "",
                "fpv_depth_path": "",
                "fpv_depth_status": "stream_unavailable",
            })
        for path, rows in (
            (episode / "samples.csv", samples),
            (episode / "auxiliary.csv", auxiliary),
        ):
            with path.open("w", newline="", encoding="utf-8") as stream:
                fields = (
                    list(rows[0]) if path.name == "samples.csv"
                    else AUXILIARY_FIELDS
                )
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        availability = _validate_auxiliary(dataset, episode_id)
        self.assertEqual(availability["observer_rgb_available"], 2)
        self.assertEqual(
            (FPV_RGB_WIDTH, FPV_RGB_HEIGHT),
            (LEGACY_OBSERVER_RGB_WIDTH, LEGACY_OBSERVER_RGB_HEIGHT),
        )
        self.assertLess(TOP_RGB_ALIGNMENT_TOLERANCE_S, 0.35)


if __name__ == "__main__":
    unittest.main()
