"""Tests for live BC checkpoint, observation, and action contracts."""

import json
import math
import shutil
import subprocess
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
from uav_ml.inference.bc_flight_contract import LIVE_IMAGE_SIZES


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
        source = ast.parse((REPOSITORY_ROOT / "isaac/runtime/environment.py").read_text())
        cls = next(node for node in ast.walk(source) if isinstance(node, ast.ClassDef) and node.name == "IsaacRuntimeBridge")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_publish_camera")
        import time
        scope = {"time": time, "CAMERA_PUBLISH_PERIOD_S": 0.2, "CAMERA_TOPIC": "fpv"}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "environment.py", "exec"), scope)
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
                diagnostics_enabled=True,
            )
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["episodes"], [result])
            self.assertTrue(summary["plots"]["summary"])
            self.assertTrue((root / "closed_loop_control_summary.md").is_file())

    def test_evaluation_duration_includes_initialization_and_cleanup_on_failure(self):
        from unittest.mock import Mock, patch
        from uav_ml.tools import bc_flight_evaluation as evaluation

        for startup_failure in (False, True):
            with self.subTest(startup_failure=startup_failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "checkpoint.pt"
                checkpoint.touch()
                output = root / "run"
                clock = [100.0]
                identity = SimpleNamespace(
                    checkpoint_path=str(checkpoint), checkpoint_sha256="digest",
                    image_source="top_rgb",
                    image_sources=("top_rgb",),
                    encoder_path=str(root / "encoder.pt"),
                )
                result = {
                    "schema": "uav_bc_flight_result/v1", "episode": 1,
                    "terminal_reason": "timeout", "episode_duration_s": 11.0,
                }

                def load_policy(*args):
                    clock[0] += 7.0
                    return SimpleNamespace(
                        identity=identity, encoder=type("RgbAutoencoderV0", (), {})(),
                    )

                def advance(seconds):
                    clock[0] += seconds

                def start(*args):
                    advance(20.0)
                    if startup_failure:
                        raise RuntimeError("startup fault")

                def run_episode(*args):
                    advance(11.0)
                    return result

                runtime = Mock()
                runtime.preflight.side_effect = lambda: advance(2.0)
                runtime.start.side_effect = start
                runtime.run_episode.side_effect = run_episode
                runtime.cleanup.side_effect = lambda: advance(5.0)
                with (
                    patch.object(evaluation, "ENABLE_BC_FLIGHT_RECORDING", False),
                    patch.object(evaluation, "_LAST_ARTIFACT_PATH", None),
                    patch.object(evaluation, "BcFlightPolicy", side_effect=load_policy),
                    patch.object(evaluation, "ManagedFlightRuntime", return_value=runtime),
                    patch.object(evaluation.time, "monotonic", side_effect=lambda: clock[0]),
                    patch("uav_ml.tools.bc_flight_plotting.generate_evaluation_plots", return_value={}),
                    patch("uav_ml.tools.bc_flight_plotting.create_training_loss_artifacts", return_value={}),
                ):
                    arguments = ["--checkpoint", str(checkpoint), "--output", str(output)]
                    if startup_failure:
                        with self.assertRaisesRegex(RuntimeError, "startup fault"):
                            evaluation.main(arguments)
                    else:
                        self.assertEqual(evaluation.main(arguments), 0)
                runtime.cleanup.assert_called_once()
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["total_test_duration_s"], 34.0 if startup_failure else 45.0)
                self.assertEqual(summary["episodes"], [] if startup_failure else [result])
                self.assertEqual(summary["feature_extractor"], "AE")
                self.assertEqual(summary["loss_summary_file"], "loss/loss_summary.json")

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


class BcFlightDiagnosticsOptionTests(unittest.TestCase):
    def test_diagnosis_option_controls_reports_without_disabling_video(self):
        from unittest.mock import Mock, patch
        from uav_ml.tools import bc_flight_evaluation as evaluation

        for diagnosis in (False, True):
            with self.subTest(diagnosis=diagnosis), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "checkpoint.pt"
                checkpoint.touch()
                output = root / "run"
                result = {
                    "schema": "uav_bc_flight_result/v1", "episode": 1,
                    "terminal_reason": "collision",
                }
                policy = SimpleNamespace(
                    identity=SimpleNamespace(
                        checkpoint_path=str(checkpoint), checkpoint_sha256="digest",
                        image_source="top_rgb",
                        image_sources=("top_rgb",), encoder_path=str(root / "encoder.pt"),
                    ),
                    encoder=type("RgbAutoencoderV0", (), {})(),
                )
                runtime = Mock()
                runtime.run_episode.return_value = result
                with (
                    patch.object(evaluation, "_LAST_ARTIFACT_PATH", None),
                    patch.object(evaluation, "BcFlightPolicy", return_value=policy),
                    patch.object(evaluation, "ManagedFlightRuntime", return_value=runtime) as factory,
                    patch.object(evaluation.shutil, "which", return_value="/usr/bin/ffmpeg"),
                    patch.object(evaluation, "_finalize_episode_video", return_value=result) as video,
                    patch("uav_ml.tools.bc_flight_plotting.generate_evaluation_plots", return_value={}),
                    patch("uav_ml.tools.bc_flight_plotting.create_training_loss_artifacts", return_value={}),
                    patch("scripts.diagnostics.summarize_bc_startup.write_run_reader_summary") as report,
                ):
                    arguments = ["--checkpoint", str(checkpoint), "--output", str(output)]
                    if diagnosis:
                        arguments.append("--diagnosis")
                    self.assertEqual(evaluation.main(arguments), 0)
                self.assertEqual(factory.call_args.kwargs["diagnostics_enabled"], diagnosis)
                self.assertEqual(report.call_count, int(diagnosis))
                video.assert_called_once_with(output / "episode_000001/result.json")
                runtime.cleanup.assert_called_once()
                self.assertEqual(json.loads((output / "summary.json").read_text())["episodes"], [result])

    def test_runtime_gates_timing_and_logger_despite_inherited_environment(self):
        import os
        from unittest.mock import patch
        from uav_ml.tools import bc_flight_evaluation as evaluation

        for diagnosis in (False, True):
            with self.subTest(diagnosis=diagnosis), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                options = {"diagnostics_enabled": True} if diagnosis else {}
                runtime = evaluation.ManagedFlightRuntime(
                    root, root, False, "cpu", root, "top_rgb", 180, **options,
                )
                with (
                    patch.dict(os.environ, {
                        "UAV_TIMING_DIR": str(root / "inherited"),
                        "UAV_TIMING_SCHED_SECONDS": "30",
                        "UAV_BC_FLIGHT_DIAGNOSTICS": "0",
                    }),
                    patch.object(evaluation.subprocess, "Popen") as popen,
                    patch.object(evaluation.threading, "Thread") as thread,
                    patch.object(runtime, "_attach_streams"),
                    patch.object(evaluation, "_stop_process"),
                    patch("scripts.diagnostics.summarize_bc_startup.write_timing_report") as report,
                ):
                    runtime.start(root)
                    isaac_environment = popen.call_args.kwargs["env"]
                    self.assertEqual(isaac_environment["UAV_BC_VIDEO_TOP"], "1")
                    popen.return_value.wait.return_value = 0
                    self.assertEqual(runtime._uav(["expert-runtime-wait"], root / "ready.log"), 0)
                    ros_environment = popen.call_args.kwargs["env"]
                    for environment in (isaac_environment, ros_environment):
                        self.assertEqual(environment["UAV_BC_FLIGHT_DIAGNOSTICS"], "1")
                        if diagnosis:
                            self.assertEqual(environment["UAV_TIMING_DIR"], str(root / "timing"))
                            self.assertEqual(environment["UAV_TIMING_SCHED_SECONDS"], "30")
                        else:
                            self.assertNotIn("UAV_TIMING_DIR", environment)
                            self.assertNotIn("UAV_TIMING_SCHED_SECONDS", environment)
                    self.assertIn("param set SDLOG_MODE -1", (root / "px4-rc.params").read_text())
                    self.assertEqual(isaac_environment["PATH"].split(os.pathsep)[0], str(root))
                    if diagnosis:
                        self.assertEqual(thread.call_args.kwargs["target"], runtime._capture_ulog)
                        thread.return_value.start.assert_called_once()
                    else:
                        thread.assert_not_called()
                        runtime.save_ulog(root, abnormal=True)
                    runtime.cleanup()
                    self.assertEqual(report.call_count, int(diagnosis))
                    if not diagnosis:
                        self.assertFalse((root / "timing").exists())
                        self.assertFalse((root / "px4_ulog").exists())

    def test_replay_passes_the_diagnosis_option(self):
        from unittest.mock import Mock, patch
        from uav_ml.tools import bc_flight_evaluation as evaluation

        for diagnosis in (False, True):
            with self.subTest(diagnosis=diagnosis), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trace = root / "trace.json"
                trace.write_text(json.dumps({
                    "schema": "uav_bc_flight_trace/v1", "episode": 1, "seed": 7,
                    "samples": [{"time_s": 0.0}, {"time_s": 1.0}],
                }))
                runtime = Mock()
                runtime._uav.return_value = 0
                with (
                    patch.object(evaluation, "_LAST_ARTIFACT_PATH", None),
                    patch.object(evaluation, "ManagedFlightRuntime", return_value=runtime) as factory,
                ):
                    arguments = ["--replay-trace", str(trace), "--output", str(root / "replay")]
                    if diagnosis:
                        arguments.append("--diagnosis")
                    self.assertEqual(evaluation.main(arguments), 0)
                self.assertEqual(factory.call_args.args[-1], diagnosis)
                runtime.replay_trace.assert_called_once()
                runtime.cleanup.assert_called_once()


class BcCheckpointSelectionTests(unittest.TestCase):
    def _pointer(self, root, relative, updated, *, status="completed"):
        directory = root / "artifacts/experiments/bc" / relative
        run = directory / "run"
        run.mkdir(parents=True)
        checkpoint = run / "best.pt"
        checkpoint.touch()
        (run / "summary.json").write_text(json.dumps({"run_status": status}))
        pointer = directory / "latest.json"
        pointer.write_text(json.dumps({
            "format_version": "bc_baseline_v1.0", "updated_utc": updated,
            "best_checkpoint": str(checkpoint),
        }))
        return checkpoint, pointer

    def test_default_selects_latest_completed_across_datasets_and_encoders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, older = self._pointer(root, "old/top", "2026-09-10T04:30:00Z")
            newest, _ = self._pointer(root, "today/top/resnet18", "2026-10-10T08:30:00Z")
            older.touch()
            self.assertEqual(resolve_checkpoint(root), newest)

    def test_unusable_pointers_are_reported_and_incomplete_runs_are_skipped(self):
        from contextlib import redirect_stderr
        from io import StringIO

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good, _ = self._pointer(root, "valid/top", "2026-10-10T08:00:00Z")
            self._pointer(root, "unfinished/top", "2026-10-10T09:00:00Z", status="running")
            missing, _ = self._pointer(root, "missing/top", "2026-10-10T10:00:00Z")
            missing.unlink()
            _, malformed = self._pointer(root, "malformed/top", "2026-10-10T11:00:00Z")
            malformed.write_text("{")
            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(resolve_checkpoint(root), good)
            self.assertIn("BC training is not completed", stderr.getvalue())
            self.assertIn("BC checkpoint is missing", stderr.getvalue())
            self.assertIn("Expecting property name", stderr.getvalue())

    def test_explicit_checkpoint_still_overrides_the_newest_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chosen, _ = self._pointer(root, "chosen/top", "2026-09-10T04:00:00Z")
            self._pointer(root, "newer/top/resnet18", "2026-10-10T08:00:00Z")
            self.assertEqual(resolve_checkpoint(root, chosen), chosen)
            self.assertEqual(resolve_checkpoint(root, Path("chosen/top/run/best.pt")), chosen)

    def test_no_completed_checkpoint_has_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "No completed BC checkpoint"):
                resolve_checkpoint(Path(directory))

    def test_unspecified_image_source_uses_checkpoint_sources_in_order(self):
        from uav_ml.inference import bc_flight as flight

        cases = (
            ("RgbAutoencoderV0", ("top",)),
            ("ResNet18Encoder", ("fpv_rgb",)),
            ("MultiImageAutoencoderV0", ("fpv_rgb", "fpv_depth")),
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            for architecture, sources in cases:
                with self.subTest(architecture=architecture):
                    dimension = 512 if architecture == "ResNet18Encoder" else 64
                    preprocessing = {source: flight.IMAGE_PREPROCESSING[source] for source in sources}
                    if len(sources) == 1:
                        preprocessing = preprocessing[sources[0]]
                    torch.save({
                        "format_version": flight.BC_CHECKPOINT_FORMAT,
                        "model_class": flight.BC_MODEL_CLASS,
                        "encoder_architecture": architecture, "encoder_frozen": True,
                        "image_sources": list(sources), "latent_dimension": dimension,
                        "observation_contract": f"latent{dimension}_plus_body_state8_v1.0",
                        "action_contract": flight.BC_ACTION_CONTRACT,
                        "image_preprocessing": preprocessing,
                        "encoder_preprocessing": (
                            flight.resnet_preprocessing_for_sources(sources)
                            if architecture == "ResNet18Encoder" else preprocessing
                        ),
                        "physical_action_limits": flight.BC_ACTION_LIMITS,
                    }, checkpoint)
                    payload = load_checkpoint_payload(checkpoint, None, torch.device("cpu"))
                    self.assertEqual(payload["_runtime_image_sources"], [
                        canonical_image_source(source) for source in sources
                    ])
                    with self.assertRaisesRegex(ValueError, "primary image source does not match"):
                        load_checkpoint_payload(checkpoint, "fpv_depth", torch.device("cpu"))


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
        width, height = LIVE_IMAGE_SIZES["top_rgb"]
        stream = BytesIO()
        Image.new("RGB", (width + 1, height + 1)).save(stream, format="JPEG")
        with self.assertRaisesRegex(
            ValueError, f"must be {width}x{height}"
        ):
            validate_live_image(stream.getvalue(), "top_rgb")
        stream = BytesIO()
        Image.new("RGB", (width, height)).save(stream, format="JPEG")
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


class PolicyInputVideoTests(unittest.TestCase):
    @staticmethod
    def _png(color, size=(320, 180)):
        stream = BytesIO()
        Image.new("RGB", size, color).save(stream, format="PNG")
        return stream.getvalue()

    def test_capture_preserves_both_images_and_the_normalized_action(self):
        from uav_ml.tools.bc_flight_video import capture_policy_input_frame, _load_frames

        primary = self._png((190, 40, 30))
        top = self._png((20, 150, 60), (640, 640))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_policy_input_frame(
                root, image_bytes=primary, image_source="fpv_rgb",
                inference_count=7, timestamp_ns=1000,
                action_body=(0.5, 0.25, -0.1), top_image_bytes=top,
                top_time_delta_ns=100,
            )
            source, frames = _load_frames(root)
            frame = frames[0]
            self.assertEqual(source, "fpv_rgb")
            self.assertEqual((root / "frames" / frame.image_filename).read_bytes(), primary)
            self.assertEqual((root / "frames" / frame.top_image_filename).read_bytes(), top)
            self.assertEqual(frame.action_body, (0.5, 0.25, -0.1))
            self.assertEqual(frame.top_time_delta_ns, 100)

    def test_dashboard_shows_both_views_and_scales_physical_commands(self):
        from unittest.mock import patch
        from PIL import ImageDraw
        from uav_ml.tools.bc_flight_video import PolicyInputFrame, _annotate_frame

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary, top, output = root / "fpv.png", root / "top.png", root / "dashboard.png"
            primary.write_bytes(self._png((190, 40, 30)))
            top.write_bytes(self._png((20, 150, 60), (640, 640)))
            frame = PolicyInputFrame(7, 1000, primary.name, (0.5, 0.25, -0.1))
            with patch.object(ImageDraw.ImageDraw, "text", autospec=True,
                              side_effect=ImageDraw.ImageDraw.text) as text:
                _annotate_frame(primary, output, frame, image_source="fpv_rgb", top_source=top)
            labels = [call.args[2] for call in text.call_args_list]
            self.assertIn("MODEL INPUT / FPV RGB", labels)
            self.assertIn("+0.50 m/s", labels)
            self.assertIn("+0.20 m/s", labels)
            self.assertIn("-0.10 rad/s", labels)
            with Image.open(output) as image:
                self.assertEqual(image.size, (2560, 1440))
                self.assertEqual(image.getpixel((788, 784)), (190, 40, 30))
                self.assertEqual(image.getpixel((2052, 568)), (20, 150, 60))

    def test_pointer_handles_backward_left_and_yaw_only(self):
        from uav_ml.tools.bc_flight_video import PolicyInputFrame, _annotate_frame

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.png", root / "dashboard.png"
            source.write_bytes(self._png((80, 80, 80)))
            for action in ((0.5, 0.25, 0.0), (-0.5, -0.25, 0.0), (0.0, 0.0, 0.5)):
                with self.subTest(action=action):
                    frame = PolicyInputFrame(1, 0, source.name, action)
                    _annotate_frame(source, output, frame, image_source="fpv_rgb")
                    with Image.open(output) as image:
                        pixels = np.asarray(image)
                    y, x = np.where(np.all(pixels == (255, 220, 0), axis=2))
                    if action[0] == 0.0:
                        self.assertEqual(len(x), 0)
                    elif action[0] > 0.0:
                        self.assertGreater(x.mean(), 2360)
                        self.assertLess(y.mean(), 1238)
                    else:
                        self.assertLess(x.mean(), 2360)
                        self.assertGreater(y.mean(), 1238)

    def test_legacy_frames_reuse_top_or_show_missing_observer(self):
        from uav_ml.tools.bc_flight_video import _load_frames, _annotate_frame

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frames").mkdir()
            source = root / "frames/input.png"
            source.write_bytes(self._png((80, 80, 80)))
            for image_source in ("top_rgb", "fpv_rgb"):
                row = {"image_source": image_source, "image_filename": "input.png",
                       "action_body": [0, 0, 0], "inference_count": 1, "timestamp_ns": 1}
                (root / "frames.jsonl").write_text(json.dumps(row) + "\n")
                _, frames = _load_frames(root)
                self.assertEqual(frames[0].top_image_filename,
                                 "input.png" if image_source == "top_rgb" else None)
                _annotate_frame(source, root / "dashboard.png", frames[0],
                                image_source=image_source,
                                top_source=source if image_source == "top_rgb" else None)

    def test_depth_display_keeps_metric_range_and_original_file(self):
        from uav_ml.tools.bc_flight_video import PolicyInputFrame, _annotate_frame

        depth = np.full((180, 320), 1000, dtype=np.uint16)
        depth[:, 160:] = 20000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "depth.png", root / "dashboard.png"
            Image.fromarray(depth).save(source)
            original = source.read_bytes()
            _annotate_frame(source, output, PolicyInputFrame(1, 0, source.name, (0, 0, 0)),
                            image_source="fpv_depth")
            self.assertEqual(source.read_bytes(), original)
            with Image.open(output) as image:
                self.assertEqual(image.getpixel((400, 784)), (8, 8, 8))
                self.assertEqual(image.getpixel((1200, 784)), (170, 170, 170))

    def test_high_resolution_observer_keeps_standard_top_input_bytes(self):
        from uav_ml.tools.bc_flight_video import capture_policy_input_frame, _load_frames

        primary = self._png((190, 40, 30), (640, 640))
        top = self._png((20, 150, 60), (1024, 1024))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_policy_input_frame(
                root, image_bytes=primary, image_source="top_rgb",
                inference_count=1, timestamp_ns=1000, action_body=(0, 0, 0),
                top_image_bytes=top, top_time_delta_ns=0,
            )
            _, frames = _load_frames(root)
            frame = frames[0]
            self.assertNotEqual(frame.image_filename, frame.top_image_filename)
            self.assertEqual((root / "frames" / frame.image_filename).read_bytes(), primary)
            self.assertEqual((root / "frames" / frame.top_image_filename).read_bytes(), top)

    def test_direction_pointer_has_antialiased_edges(self):
        from uav_ml.tools.bc_flight_video import PolicyInputFrame, _annotate_frame

        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.png", Path(directory) / "dashboard.png"
            source.write_bytes(self._png((80, 80, 80)))
            _annotate_frame(source, output, PolicyInputFrame(1, 0, source.name, (0.5, 0.25, 0)))
            with Image.open(output) as image:
                pointer = np.asarray(image.crop((2230, 1108, 2490, 1370)))
            yellow_edges = ((pointer[..., 0] > 150) & (pointer[..., 0] < 254)
                            & (pointer[..., 1] > 100) & (pointer[..., 2] < 45))
            self.assertGreater(np.count_nonzero(yellow_edges), 0)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg and ffprobe are required")
    def test_real_mp4_encoding_has_dashboard_resolution_and_top_metadata(self):
        from uav_ml.tools.bc_flight_video import capture_policy_input_frame, render_policy_input_video

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in (1, 2):
                capture_policy_input_frame(
                    root, image_bytes=self._png((190, 40, 30)), image_source="fpv_rgb",
                    inference_count=index, timestamp_ns=index * 200_000_000,
                    action_body=(0.5, 0.25, 0.1),
                    top_image_bytes=self._png((20, 150, 60), (640, 640)),
                    top_time_delta_ns=0,
                )
            metadata = render_policy_input_video(root, root / "episode_000001.mp4")
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                 str(root / "episode_000001.mp4")],
                capture_output=True, text=True, check=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual((stream["width"], stream["height"]), (2560, 1440))
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual(metadata["frame_count"], 2)
            self.assertEqual(metadata["top_frame_count"], 2)
            self.assertEqual(metadata["action_units"], ["m/s", "m/s", "rad/s"])
            self.assertEqual(metadata["h264_crf"], 16)
            self.assertEqual(metadata["hud_supersampling"], 2)


if __name__ == "__main__":
    unittest.main()
