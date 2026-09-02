"""Prove live DDS telemetry belongs to an expected PX4 generation."""

from __future__ import annotations

import importlib
import json
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


STATUS_TOPIC = "/fmu/out/vehicle_status"
ODOMETRY_TOPIC = "/fmu/out/vehicle_odometry"
LAND_TOPIC = "/fmu/out/vehicle_land_detected"


def process_start_ticks(pid: int) -> int | None:
    """Return Linux /proc start ticks, accounting for the comm field."""
    try:
        data = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields_after_comm = data[data.rfind(")") + 2:].split()
        return int(fields_after_comm[19])
    except (OSError, ValueError, IndexError):
        return None


def process_identity(pid: int) -> dict:
    """Capture stable, read-only evidence for one PX4 process."""
    try:
        executable = os.readlink(f"/proc/{pid}/exe")
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
            b"\0", b" "
        ).decode(errors="replace").strip()
    except OSError:
        executable = ""
        command = ""
    return {
        "pid": pid,
        "start_ticks": process_start_ticks(pid),
        "executable": executable,
        "command": command,
    }


class Px4GenerationProbe(Node):
    """Require changing telemetry and a stable expected PID."""

    def __init__(self) -> None:
        """Create subscriptions and configure the expected generation."""
        super().__init__("px4_generation_probe")
        self.declare_parameter("expected_pid", 0)
        self.declare_parameter("expected_start_ticks", 0)
        self.declare_parameter("generation", 0)
        self.declare_parameter("timeout_s", 45.0)
        self.declare_parameter("minimum_samples", 5)
        self.declare_parameter("minimum_span_s", 0.5)
        self.declare_parameter("evidence_path", "/tmp/px4_generation.json")
        self.expected_pid = int(self.get_parameter("expected_pid").value)
        self.expected_start_ticks = int(
            self.get_parameter("expected_start_ticks").value
        )
        self.generation = int(self.get_parameter("generation").value)
        self.timeout_s = float(self.get_parameter("timeout_s").value)
        self.minimum_samples = int(
            self.get_parameter("minimum_samples").value
        )
        self.minimum_span_s = float(
            self.get_parameter("minimum_span_s").value
        )
        self.evidence_path = Path(
            str(self.get_parameter("evidence_path").value)
        )
        if self.expected_pid <= 0 or self.expected_start_ticks <= 0:
            raise ValueError(
                "expected PX4 PID and start ticks must be positive"
            )

        messages = importlib.import_module("px4_msgs.msg")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.started_monotonic = time.monotonic()
        self.status_samples: list[tuple[int, float]] = []
        self.odometry_samples: list[tuple[int, float]] = []
        self.land_samples: list[tuple[int, float]] = []
        self.latest_status = None
        self.latest_land = None
        self.finished = False
        self.exit_code = 1
        self.create_subscription(
            messages.VehicleStatus, STATUS_TOPIC, self._status, qos
        )
        self.create_subscription(
            messages.VehicleOdometry, ODOMETRY_TOPIC, self._odometry, qos
        )
        self.create_subscription(
            messages.VehicleLandDetected, LAND_TOPIC, self._land, qos
        )
        self.create_timer(0.10, self._tick)

    @staticmethod
    def _append_sample(target, message) -> None:
        timestamp = int(getattr(message, "timestamp", 0))
        if timestamp <= 0:
            return
        if not target or timestamp != target[-1][0]:
            target.append((timestamp, time.monotonic()))

    def _status(self, message) -> None:
        self.latest_status = message
        self._append_sample(self.status_samples, message)

    def _odometry(self, message) -> None:
        self._append_sample(self.odometry_samples, message)

    def _land(self, message) -> None:
        self.latest_land = message
        self._append_sample(self.land_samples, message)

    def _process_matches(self) -> bool:
        identity = process_identity(self.expected_pid)
        return bool(
            identity["start_ticks"] == self.expected_start_ticks
            and identity["executable"].endswith(
                "/build/px4_sitl_default/bin/px4"
            )
            and " -i 0 -d" in identity["command"]
        )

    def _stream_is_fresh(self, samples) -> bool:
        if len(samples) < self.minimum_samples:
            return False
        timestamps = [sample[0] for sample in samples]
        receive_times = [sample[1] for sample in samples]
        return bool(
            timestamps[-1] > timestamps[0]
            and all(right > left for left, right in zip(
                timestamps, timestamps[1:]
            ))
            and receive_times[-1] - receive_times[0] >= self.minimum_span_s
            and receive_times[0] >= self.started_monotonic
        )

    def _safe_state(self) -> bool:
        status = self.latest_status
        land = self.latest_land
        return bool(
            status is not None
            and land is not None
            and land.landed
            and status.arming_state != status.ARMING_STATE_ARMED
            and not status.failsafe
        )

    def _endpoint_gids(self, topic: str) -> list[str]:
        result = []
        for endpoint in self.get_publishers_info_by_topic(topic):
            try:
                result.append(bytes(endpoint.endpoint_gid).hex())
            except (TypeError, ValueError):
                result.append(str(endpoint.endpoint_gid))
        return result

    @staticmethod
    def _sample_payload(samples) -> dict:
        return {
            "count": len(samples),
            "first_timestamp": samples[0][0] if samples else None,
            "last_timestamp": samples[-1][0] if samples else None,
            "receive_span_s": (
                samples[-1][1] - samples[0][1] if len(samples) > 1 else 0.0
            ),
        }

    def _payload(self, success: bool, reason: str) -> dict:
        return {
            "success": success,
            "failure_reason": "" if success else reason,
            "generation": self.generation,
            "probe_started_monotonic": self.started_monotonic,
            "process": process_identity(self.expected_pid),
            "expected_start_ticks": self.expected_start_ticks,
            "process_generation_matches": self._process_matches(),
            "status": self._sample_payload(self.status_samples),
            "odometry": self._sample_payload(self.odometry_samples),
            "land": self._sample_payload(self.land_samples),
            "safe_landed_disarmed": self._safe_state(),
            "status_endpoint_gids": self._endpoint_gids(STATUS_TOPIC),
            "odometry_endpoint_gids": self._endpoint_gids(ODOMETRY_TOPIC),
        }

    def _tick(self) -> None:
        if self.finished:
            return
        if not self._process_matches():
            self._finish(False, "expected PX4 process generation disappeared")
            return
        if (
            self._stream_is_fresh(self.status_samples)
            and self._stream_is_fresh(self.odometry_samples)
            and self._safe_state()
        ):
            self._finish(True, "")
            return
        if time.monotonic() - self.started_monotonic > self.timeout_s:
            self._finish(False, "fresh landed/disarmed telemetry timed out")

    def _finish(self, success: bool, reason: str) -> None:
        if self.finished:
            return
        self.finished = True
        self.exit_code = 0 if success else 1
        payload = self._payload(success, reason)
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        log = self.get_logger().info if success else self.get_logger().error
        log(
            "PX4_GENERATION_RESULT "
            f"success={str(success).lower()} generation={self.generation} "
            f"pid={self.expected_pid} evidence={self.evidence_path}"
        )
        rclpy.shutdown()


def main(args=None) -> int:
    """Run the PX4 process-generation readiness probe."""
    rclpy.init(args=args)
    node = Px4GenerationProbe()
    try:
        rclpy.spin(node)
    finally:
        code = node.exit_code
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
