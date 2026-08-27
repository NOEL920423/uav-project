"""Regression tests for formal BC artifact plotting."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from uav_ml.tools.bc_flight_plotting import (
    PLOT_FILENAMES,
    generate_evaluation_plots,
)


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
                PLOT_FILENAMES["trajectory"],
                PLOT_FILENAMES["goal_distance"],
                PLOT_FILENAMES["action"],
                PLOT_FILENAMES["clearance"],
            })
            self.assertEqual(len(plots["summary"]), 6)

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
            self.assertEqual(len(plots["summary"]), 6)


if __name__ == "__main__":
    unittest.main()
