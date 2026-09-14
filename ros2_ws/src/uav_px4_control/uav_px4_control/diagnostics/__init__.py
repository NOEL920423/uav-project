"""Optional local timing evidence shared by existing ROS adapters."""

import atexit
import json
import os
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


class TimingRecorder:
    """Buffer bounded per-node JSONL records without changing ROS messages."""

    def __init__(self, node):
        """Enable file output only when the evaluator supplies a directory."""
        self.node = node
        self.inputs = {}
        self.sequence = 0
        self.last_tick = None
        self.stream = None
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
            except OSError as error:
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
        self.record("publish", topic=topic, key=message_key(message), **fields)

    def tick(self):
        """Measure actual timer callback spacing, including scheduling jitter."""
        now = time.monotonic_ns()
        interval = None if self.last_tick is None else (now - self.last_tick) / 1e6
        self.last_tick = now
        self.record("tick", timer_interval_ms=interval)

    def consume(self, **fields):
        """Snapshot cached inputs; inactive sources may also be present."""
        now = time.monotonic_ns()
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
