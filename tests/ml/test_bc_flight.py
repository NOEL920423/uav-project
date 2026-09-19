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
    def test_reader_summary_writes_chinese_control_questions_and_run_overview(self):
        from scripts.diagnostics.summarize_bc_startup import (
            write_reader_summary,
            write_run_reader_summary,
        )
        image = "/uav/isaac/fpv/image/compressed"
        summary = {
            "statistics": [
                {"segment": f"接收間隔｜bc_policy｜{image}", "mean_ms": 200.0, "p95_ms": 220.0, "max_ms": 230.0},
                {"segment": "影像接收至動作首次 PX4 發布", "mean_ms": 80.0, "p95_ms": 100.0, "max_ms": 120.0},
                {"segment": "PX4 發布時影像接收年齡（含重送）", "mean_ms": 160.0, "p95_ms": 250.0, "max_ms": 290.0},
                {"segment": "timer 執行間隔｜bc_policy", "mean_ms": 50.0, "p95_ms": 50.1, "max_ms": 300.0},
            ],
            "action_lineage": {"bc_new_actions": 8, "bc_repeats": 24},
            "ulog_inspection": {"files": [{"topics": [
                {"topic": "trajectory_setpoint"}, {"topic": "vehicle_local_position"},
            ]}]},
        }
        result = {"terminal_reason": "collision", "image_source": "fpv_rgb", "seed": 7}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode_000001"
            episode.mkdir()
            reader = write_reader_summary(episode, summary, result)
            (episode / "result.json").write_text(json.dumps(result))
            complete = {**summary, "reader_summary": reader}
            (episode / "timing_summary.json").write_text(json.dumps(complete))
            overview = write_run_reader_summary(root)
            text = (episode / "control_summary.md").read_text()
            run_text = (root / "closed_loop_control_summary.md").read_text()
        self.assertEqual(reader["image_rate_hz"], 5.0)
        self.assertEqual(reader["image_to_px4_p95_ms"], 100.0)
        self.assertEqual(reader["image_age_p95_ms"], 250.0)
        self.assertEqual(reader["maximum_timer_node"], "bc_policy")
        self.assertIn("影像接收 → 該動作首次送往 PX4", text)
        self.assertIn("下一個要驗證的假說", text)
        self.assertIn("碰撞", run_text)
        self.assertEqual(overview["outcomes"], {"碰撞": 1})

    @staticmethod
    def _diagnostics_module():
        import importlib.util
        path = REPOSITORY_ROOT / "ros2_ws/src/uav_px4_control/uav_px4_control/diagnostics/__init__.py"
        spec = importlib.util.spec_from_file_location("test_timing_helper", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_publish_api_timing_preserves_exception_and_excludes_record_writes(self):
        from unittest.mock import patch
        module = self._diagnostics_module()
        recorder = module.TimingRecorder.__new__(module.TimingRecorder)
        recorder.stream = True
        events = []
        recorder.record = lambda event, **fields: events.append((event, fields))
        error = RuntimeError("publisher failed")

        def fail(message):
            raise error

        publisher = SimpleNamespace(topic_name="/command", publish=fail)
        with patch.object(module.time, "monotonic_ns", side_effect=[1000000, 301000000]), patch.object(
            module.time, "thread_time_ns", side_effect=[1000000, 2000000]
        ):
            with self.assertRaises(RuntimeError) as caught:
                recorder.send(publisher, SimpleNamespace())
        self.assertIs(caught.exception, error)
        self.assertEqual([event for event, _ in events], ["publish_api_start", "publish_api_end"])
        self.assertEqual(events[-1][1]["duration_ms"], 300.0)
        self.assertEqual(events[-1][1]["thread_cpu_ms"], 1.0)
        self.assertFalse(events[-1][1]["succeeded"])
        recorder.stream = None
        sent = []
        self.assertIsNone(recorder.send(SimpleNamespace(publish=sent.append), 42))
        self.assertEqual(sent, [42])
        self.assertEqual(len(events), 2)

    def test_consumed_refs_do_not_change_when_new_input_arrives(self):
        from unittest.mock import patch
        module = self._diagnostics_module()
        node = SimpleNamespace(
            get_name=lambda: "test", get_clock=lambda: SimpleNamespace(
                now=lambda: SimpleNamespace(nanoseconds=123)),
            get_logger=lambda: SimpleNamespace(error=self.fail),
        )
        message = lambda stamp: SimpleNamespace(header=SimpleNamespace(
            stamp=SimpleNamespace(sec=stamp, nanosec=0)))
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"UAV_TIMING_DIR": directory, "UAV_TIMING_SCHED_SECONDS": "0"}
        ):
            recorder = module.TimingRecorder(node)
            recorder.receive("image", message(1))
            recorder.consume(active_source="BC_POLICY")
            recorder.receive("image", message(2))
            recorder.publish("command", message(3))
            recorder.close()
            rows = [json.loads(line) for line in next(Path(directory).glob("*.jsonl")).read_text().splitlines()]
        published = rows[-1]
        self.assertEqual(published["input_refs"]["image"]["key"], "header:1:0")
        self.assertEqual(published["consume_context"]["active_source"], "BC_POLICY")

    def test_callback_exception_preserves_original_and_records_exit(self):
        module = self._diagnostics_module()
        rows = []
        owner = SimpleNamespace(_timing=SimpleNamespace(
            stream=True, record=lambda event, **fields: rows.append((event, fields))))

        @module.timed_callback
        def failing(self):
            raise RuntimeError("original callback failure")

        with self.assertRaisesRegex(RuntimeError, "original callback failure"):
            failing(owner)
        self.assertEqual([row[0] for row in rows], ["callback_start", "callback_end"])
        self.assertGreaterEqual(rows[-1][1]["duration_ms"], 0)

    def test_scheduler_unavailable_is_explicit_and_stops_sampling(self):
        from unittest.mock import patch
        module = self._diagnostics_module()
        rows, errors = [], []
        recorder = module.TimingRecorder.__new__(module.TimingRecorder)
        recorder.scheduler_previous = None
        recorder.scheduler_until_ns = 123
        recorder.node = SimpleNamespace(get_logger=lambda: SimpleNamespace(error=errors.append))
        recorder.record = lambda event, **fields: rows.append((event, fields))
        with patch.object(Path, "read_text", side_effect=PermissionError("proc denied")):
            recorder._scheduler_sample(1)
        self.assertEqual(rows[-1][0], "scheduler_unavailable")
        self.assertIn("proc denied", errors[0])
        self.assertEqual(recorder.scheduler_until_ns, 0)

    def test_scheduler_deltas_keep_thread_wait_separate_from_process_io(self):
        from unittest.mock import patch
        import threading
        module = self._diagnostics_module()
        rows = []
        recorder = module.TimingRecorder.__new__(module.TimingRecorder)
        counters = {"thread_run_ns": 10, "thread_runqueue_wait_ns": 20,
                    "thread_timeslices": 1, "thread_block_io_ticks": 2,
                    "process_read_bytes": 100, "process_write_bytes": 200}
        recorder.scheduler_previous = (1, counters, threading.get_native_id())
        recorder.record = lambda event, **fields: rows.append((event, fields))
        fields = ["0"] * 40
        fields[39] = "5"
        with patch.object(Path, "read_text", side_effect=[
            "30 70 3", "123 (thread name) " + " ".join(fields),
            "read_bytes: 110\nwrite_bytes: 250\n",
        ]):
            recorder._scheduler_sample(1000001)
        delta = rows[0][1]["delta"]
        self.assertEqual(delta["thread_runqueue_wait_ns"], 50)
        self.assertEqual(delta["thread_block_io_ticks"], 3)
        self.assertEqual(delta["process_write_bytes"], 50)
        self.assertEqual(rows[0][1]["interval_ms"], 1)

    def test_fpv_timing_marks_capture_unknown_and_records_publication(self):
        import ast
        source = ast.parse((REPOSITORY_ROOT / "isaac/runtime/runtime_bridge.py").read_text())
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "IsaacRuntimeBridge")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_publish_camera")
        import time
        scope = {"time": time, "CAMERA_PUBLISH_PERIOD_S": 0.2, "CAMERA_TOPIC": "fpv"}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "runtime_bridge.py", "exec"), scope)
        events = []
        sent = []
        owner = SimpleNamespace(
            _camera_enabled=True, _last_camera_publish_monotonic=0,
            _camera_frame_count=0, _expert_sensors_enabled=False,
            _update_camera_pose=lambda: True,
            _rgb_annotator=SimpleNamespace(get_data=lambda: b"pixels"),
            _jpeg_message=lambda *args: b"jpeg",
            _camera_publisher=SimpleNamespace(publish=sent.append),
            _timing=SimpleNamespace(
                record=lambda event, **fields: events.append((event, fields)),
                publish=lambda topic, message, **fields: events.append((topic, fields))),
        )
        scope["_publish_camera"](owner, None, 1.0)
        self.assertEqual(sent, [b"jpeg"])
        self.assertFalse(events[0][1]["capture_time_known"])
        self.assertIsNone(events[0][1]["render_frame_id"])
        self.assertEqual(events[1][1]["frame_sequence"], 1)
        self.assertGreaterEqual(events[0][1]["encode_ms"], 0)

    def test_lineage_tracks_repeats_and_excludes_hold_and_unsafe_gate(self):
        from scripts.diagnostics.summarize_bc_startup import timing_summary
        bc = "/uav/control/bc_command"
        selected = "/uav/control/selected_command"
        candidate = "/uav/px4/setpoint_candidate"
        output = "/fmu/in/trajectory_setpoint"
        image = "/uav/isaac/fpv/image/compressed"
        action = {"action_id": 1, "image_topic": image,
                  "observation_inputs": {image: {"key": "img", "receive_ns": 1000000}},
                  "inference_completed_ns": 2000000}
        rows = []

        def pub(node, topic, key, at, **fields):
            rows.append({"schema": "uav_timing/v1", "node": node, "event": "publish",
                         "topic": topic, "key": key, "mono_ns": at * 1000000, **fields})

        pub("bc", bc, "bc1", 3, action_origin=action, repeated_action=False)
        pub("mux", selected, "m1", 4, input_refs={bc: {"key": "bc1"}}, consume_context={"active_source": "BC_POLICY"})
        pub("gate", candidate, "g1", 5, input_refs={selected: {"key": "m1"}}, consume_context={"state": "SAFE_TO_FORWARD"})
        pub("stream", output, "s1", 6, input_refs={candidate: {"key": "g1"}})
        pub("stream", output, "s2", 56, input_refs={candidate: {"key": "g1"}})
        pub("mux", selected, "hold", 60, input_refs={bc: {"key": "bc1"}}, consume_context={"active_source": "HOLD"})
        pub("gate", candidate, "g2", 61, input_refs={selected: {"key": "hold"}}, consume_context={"state": "SAFE_TO_FORWARD"})
        pub("stream", output, "s3", 62, input_refs={candidate: {"key": "g2"}})
        pub("gate", candidate, "g3", 63, input_refs={selected: {"key": "m1"}}, consume_context={"state": "FAULT_LATCHED"})
        pub("stream", output, "s4", 64, input_refs={candidate: {"key": "g3"}})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "timing").mkdir()
            (root / "timing/events.jsonl").write_text("\n".join(map(json.dumps, rows)))
            summary = timing_summary(root)
        metrics = {row["segment"]: row for row in summary["statistics"]}
        ages = metrics["PX4 發布時影像接收年齡（含重送）"]
        self.assertEqual((ages["count"], ages["mean_ms"], ages["max_ms"]), (2, 30, 55))
        self.assertEqual(metrics["影像接收至動作首次 PX4 發布"]["count"], 1)
        self.assertEqual(summary["action_lineage"]["stream_unlinked_or_non_bc"], 2)

    def test_ulog_inventory_preserves_dropout_and_exports_local_clock(self):
        from unittest.mock import patch
        from scripts.diagnostics.summarize_bc_startup import inspect_ulog
        log = SimpleNamespace(
            dropouts=[SimpleNamespace(timestamp=140000, duration=352)],
            data_list=[SimpleNamespace(name="trajectory_setpoint", multi_id=0,
                data={"timestamp": [1000, 2000], "velocity[0]": [1.0, 2.0]})])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "px4_ulog").mkdir()
            (root / "px4_ulog/test.ulg").touch()
            with patch.dict("sys.modules", {"pyulog": SimpleNamespace(ULog=lambda *a, **kw: log)}):
                report = inspect_ulog(root, export=True)
            entry = report["files"][0]
            self.assertEqual(entry["dropouts"][0]["duration_ms"], 352)
            self.assertIn("timesync_status", entry["missing_topics"])
            self.assertEqual(report["clock_alignment"], "unavailable")
            self.assertIn("1000,1.0", (root / entry["topics"][0]["csv"]).read_text())

    def test_ulog_parse_error_is_not_reported_as_a_healthy_log(self):
        from unittest.mock import Mock, patch
        from contextlib import redirect_stderr
        from io import StringIO
        from scripts.diagnostics.summarize_bc_startup import inspect_ulog
        stderr = StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "px4_ulog").mkdir()
            (root / "px4_ulog/broken.ulg").touch()
            fake = SimpleNamespace(ULog=Mock(side_effect=ValueError("bad header")))
            with patch.dict("sys.modules", {"pyulog": fake}), redirect_stderr(stderr):
                report = inspect_ulog(root)
        self.assertEqual(report["files"][0]["error"], "ValueError: bad header")
        self.assertIn("bad header", stderr.getvalue())

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
            self.assertTrue((root / "closed_loop_control_summary.md").is_file())

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
