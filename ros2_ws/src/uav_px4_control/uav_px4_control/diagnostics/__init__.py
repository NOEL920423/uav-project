"""Optional local timing evidence shared by existing ROS adapters."""

import atexit
from functools import wraps
import json
import os
import threading
import time
from pathlib import Path


def message_key(message):
    """Keep source clocks opaque; a stamp is not automatically capture time."""
    if hasattr(message, "header"):
        stamp = message.header.stamp
        return f"header:{stamp.sec}:{stamp.nanosec}"
    if hasattr(message, "timestamp"):
        return f"px4_wire:{message.timestamp}"
    return None


def timed_callback(callback):
    """Record callback wall/CPU duration, including exceptional exits."""
    @wraps(callback)
    def measured(self, *args, **kwargs):
        recorder = self._timing
        if getattr(recorder, "stream", None) is None:
            return callback(self, *args, **kwargs)
        start = time.monotonic_ns()
        cpu = time.thread_time_ns()
        recorder.record("callback_start", callback=callback.__name__, started_ns=start)
        try:
            return callback(self, *args, **kwargs)
        finally:
            end = time.monotonic_ns()
            recorder.record(
                "callback_end", callback=callback.__name__, started_ns=start,
                duration_ms=(end - start) / 1e6,
                thread_cpu_ms=(time.thread_time_ns() - cpu) / 1e6,
            )
    return measured


class TimingRecorder:
    """Buffer bounded per-node JSONL records without changing ROS messages."""

    def __init__(self, node):
        """Enable file output only when the evaluator supplies a directory."""
        self.node = node
        self.inputs = {}
        self.sequence = 0
        self.last_tick = None
        self.stream = None
        self.consumed_inputs = {}
        self.consume_context = {}
        self.scheduler_previous = None
        self.scheduler_until_ns = 0
        directory = os.environ.get("UAV_TIMING_DIR")
        if directory:
            try:
                path = Path(directory)
                path.mkdir(parents=True, exist_ok=True)
                self.stream = (path / f"{node.get_name()}_{os.getpid()}.jsonl").open(
                    "w", encoding="utf-8"
                )
                atexit.register(self.close)
                self.record("clock", clock="host_monotonic_ns", pid=os.getpid())
                # Bounded, opt-in scheduler evidence for a targeted run.
                seconds = float(os.environ.get("UAV_TIMING_SCHED_SECONDS", "0"))
                if seconds > 0:
                    self.scheduler_until_ns = time.monotonic_ns() + int(seconds * 1e9)
            except (OSError, ValueError, OverflowError) as error:
                node.get_logger().error(f"TIMING_ERROR: {error}")

    def record(self, event, **fields):
        """Append one event with explicitly named host and ROS clocks."""
        if self.stream is None:
            return None
        now = time.monotonic_ns()
        self.sequence += 1
        payload = {
            "schema": "uav_timing/v1", "node": self.node.get_name(),
            "event": event, "sequence": self.sequence, "mono_ns": now,
            "ros_ns": self.node.get_clock().now().nanoseconds,
            "wall_ns": time.time_ns(), **fields,
            "pid": os.getpid(), "tid": threading.get_native_id(),
        }
        try:
            self.stream.write(json.dumps(payload, allow_nan=False) + "\n")
            if self.sequence % 32 == 0 or event in {"fault", "close"}:
                self.stream.flush()
        except (OSError, ValueError) as error:
            self.node.get_logger().error(f"TIMING_ERROR: {error}")
            self.close()
        return now

    def receive(self, topic, message):
        """Associate a received source key with its local callback time."""
        now = time.monotonic_ns()
        previous = self.inputs.get(topic)
        entry = {"key": message_key(message), "receive_ns": now}
        self.inputs[topic] = entry
        self.record(
            "receive", topic=topic, key=entry["key"],
            receive_gap_ms=None if previous is None else (
                now - previous["receive_ns"]
            ) / 1e6,
        )

    def publish(self, topic, message, **fields):
        """Record immediately before the publisher API call."""
        self.record(
            "publish", topic=topic, key=message_key(message),
            input_refs=self.consumed_inputs, consume_context=self.consume_context,
            **fields,
        )

    def send(self, publisher, message):
        """Measure the publish API only, preserving exceptions and ROS semantics."""
        if self.stream is None:
            return publisher.publish(message)
        topic = publisher.topic_name
        key = message_key(message)
        self.record("publish_api_start", topic=topic, key=key)
        # Exclude diagnostic file writes from the measured API duration.
        start = time.monotonic_ns()
        cpu = time.thread_time_ns()
        succeeded = False
        try:
            result = publisher.publish(message)
            succeeded = True
            return result
        finally:
            end = time.monotonic_ns()
            cpu_ms = (time.thread_time_ns() - cpu) / 1e6
            self.record(
                "publish_api_end", topic=topic, key=key, started_ns=start,
                duration_ms=(end - start) / 1e6, thread_cpu_ms=cpu_ms,
                succeeded=succeeded,
            )

    def tick(self):
        """Measure actual timer callback spacing, including scheduling jitter."""
        now = time.monotonic_ns()
        interval = None if self.last_tick is None else (now - self.last_tick) / 1e6
        self.last_tick = now
        self.record("tick", timer_interval_ms=interval)
        if now < self.scheduler_until_ns:
            self._scheduler_sample(now)

    def _scheduler_sample(self, now):
        """Read Linux counters; unavailable accounting must remain unknown."""
        task = Path(f"/proc/self/task/{threading.get_native_id()}")
        try:
            if self.scheduler_previous is None:
                accounting = {}
                for name in ("sched_schedstats", "task_delayacct"):
                    try:
                        accounting[name] = Path(f"/proc/sys/kernel/{name}").read_text().strip()
                    except OSError:
                        accounting[name] = None
                self.record("scheduler_accounting", settings=accounting)
            running, waiting, slices = map(int, (task / "schedstat").read_text().split())
            stat = (task / "stat").read_text().rsplit(")", 1)[1].split()
            io = dict(line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())
            current = {
                "thread_run_ns": running, "thread_runqueue_wait_ns": waiting,
                "thread_timeslices": slices, "thread_block_io_ticks": int(stat[39]),
                "process_read_bytes": int(io["read_bytes"]),
                "process_write_bytes": int(io["write_bytes"]),
            }
            previous = self.scheduler_previous
            delta = ({key: current[key] - previous[1][key] for key in current}
                     if previous and previous[2] == threading.get_native_id() else None)
            self.record(
                "scheduler", counters=current, delta=delta,
                interval_ms=(now - previous[0]) / 1e6 if delta is not None else None,
                block_io_tick_hz=os.sysconf("SC_CLK_TCK"),
                accounting_note="Zero counters do not prove accounting is enabled; process I/O is not thread I/O wait.",
            )
            self.scheduler_previous = (now, current, threading.get_native_id())
        except (OSError, ValueError, KeyError, IndexError) as error:
            self.record("scheduler_unavailable", reason=str(error))
            self.node.get_logger().error(f"TIMING_SCHED_ERROR: {error}")
            self.scheduler_until_ns = 0

    def consume(self, **fields):
        """Snapshot cached inputs; inactive sources may also be present."""
        now = time.monotonic_ns()
        self.consumed_inputs = {
            topic: dict(entry) for topic, entry in self.inputs.items()
        }
        self.consume_context = dict(fields)
        self.record("consume", inputs={
            topic: {**entry, "residence_ms": (now - entry["receive_ns"]) / 1e6}
            for topic, entry in self.inputs.items()
        }, **fields)

    def close(self):
        """Flush remaining records on normal process exit."""
        if self.stream is not None:
            stream, self.stream = self.stream, None
            try:
                stream.close()
            except OSError as error:
                self.node.get_logger().error(f"TIMING_ERROR: {error}")
