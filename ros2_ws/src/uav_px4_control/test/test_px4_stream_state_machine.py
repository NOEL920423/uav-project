"""Fail-closed regression suite for the Phase 8 stream state machine."""

from dataclasses import replace

import pytest

from uav_px4_control.px4_stream_models import (
    Px4StreamConfig,
    Px4StreamState,
    StreamCandidate,
    StreamGateEvidence,
    StreamReadiness,
    StreamTelemetry,
)
from uav_px4_control.px4_stream_state_machine import Px4StreamStateMachine


def config(**overrides):
    """Use short prestream thresholds while preserving safety timeouts."""
    values = {
        "minimum_prestream_duration_s": 0.10,
        "minimum_prestream_messages": 2,
    }
    values.update(overrides)
    return Px4StreamConfig(**values)


def candidate(now=10.0, timestamp_us=10_000_000, **overrides):
    """Create one healthy candidate."""
    values = {
        "receipt_time_s": now,
        "timestamp_us": timestamp_us,
        "velocity_ned_mps": (0.0, 0.0, 0.0),
        "yaw_rate_ned_radps": 0.0,
    }
    values.update(overrides)
    return StreamCandidate(**values)


def readiness(now=10.0, current_candidate=None, **overrides):
    """Create complete disarmed, non-OFFBOARD, no-failsafe evidence."""
    values = {
        "sitl_guard_valid": True,
        "dds_ready": True,
        "gate": StreamGateEvidence(
            bool_receipt_time_s=now,
            bool_safe_to_forward=True,
            status_receipt_time_s=now,
            status_safe_to_forward=True,
            status_state="SAFE_TO_FORWARD",
        ),
        "candidate": current_candidate or candidate(now),
        "telemetry": StreamTelemetry(
            oldest_receipt_time_s=now,
            newest_timestamp_us=int(now * 1e6),
            vehicle_armed=False,
            offboard_active=False,
            failsafe=False,
            odometry_valid=True,
        ),
    }
    values.update(overrides)
    return StreamReadiness(**values)


def prime(machine, now=10.0):
    """Establish the required three-message monotonic heartbeat window."""
    for index in range(3):
        message = candidate(
            now=now - 0.10 + index * 0.05,
            timestamp_us=10_000_000 + index,
        )
        machine.observe_candidate(message)
    return candidate(now=now, timestamp_us=10_000_002)


def begin_stream(machine, now=10.0):
    """Enable, enter ready, then publish the first message pair."""
    current = prime(machine, now)
    assert machine.request_enable(True)[0]
    result = machine.step(now, readiness(now, current), int(now * 1e6))
    assert result.state == Px4StreamState.PRESTREAM_READY
    assert result.should_publish is False
    later = now + 0.05
    current = candidate(later, 10_000_003)
    machine.observe_candidate(current)
    result = machine.step(later, readiness(later, current), int(later * 1e6))
    assert result.state == Px4StreamState.PRESTREAMING
    assert result.should_publish is True
    return result, later, current


def test_startup_and_healthy_gate_remain_disabled():
    """Neither connectivity nor Phase 7 health bypasses explicit enable."""
    machine = Px4StreamStateMachine(config())
    current = prime(machine)
    result = machine.step(10.0, readiness(10.0, current), 10_000_000)
    assert result.state == Px4StreamState.STREAM_DISABLED
    assert result.streaming is False
    assert result.trajectory_setpoint_count == 0


def test_explicit_enable_prestream_stream_and_disable():
    """Publish after readiness and stop immediately on explicit disable."""
    machine = Px4StreamStateMachine(config())
    _, now, _ = begin_stream(machine)
    for index in range(1, 4):
        tick = now + index * 0.05
        current = candidate(tick, 10_000_003 + index)
        machine.observe_candidate(current)
        result = machine.step(
            tick,
            readiness(tick, current),
            int(tick * 1e6),
        )
    assert result.state == Px4StreamState.STREAMING
    assert machine.request_enable(False)[0]
    result = machine.step(
        tick + 0.05,
        readiness(tick + 0.05, current),
        int((tick + 0.05) * 1e6),
    )
    assert result.state == Px4StreamState.STREAM_DISABLED
    assert result.should_publish is False


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("gate_false", Px4StreamState.STOPPED_GATE_FALSE),
        ("candidate_stale", Px4StreamState.STOPPED_STALE_CANDIDATE),
        ("gate_stale", Px4StreamState.STOPPED_STALE_GATE),
        ("telemetry_stale", Px4StreamState.STOPPED_STALE_TELEMETRY),
        ("failsafe", Px4StreamState.STOPPED_FAILSAFE),
        ("armed", Px4StreamState.STOPPED_ARMED),
        ("offboard", Px4StreamState.STOPPED_OFFBOARD_ACTIVE),
        ("dds_loss", Px4StreamState.STOPPED_DDS_LOSS),
        ("invalid_mapping", Px4StreamState.STOPPED_INVALID_MAPPING),
    ],
)
def test_runtime_faults_stop_both_publications_and_latch(mutation, expected):
    """Every runtime safety loss revokes publication authority atomically."""
    machine = Px4StreamStateMachine(config())
    _, now, current = begin_stream(machine)
    tick = now + 0.05
    current = candidate(tick, 10_000_004)
    machine.observe_candidate(current)
    evidence = readiness(tick, current)
    if mutation == "gate_false":
        evidence = replace(
            evidence,
            gate=replace(
                evidence.gate,
                bool_safe_to_forward=False,
                status_safe_to_forward=False,
                status_state="READY_DISABLED",
            ),
        )
    elif mutation == "candidate_stale":
        evidence = replace(
            evidence,
            candidate=replace(current, receipt_time_s=tick - 0.30),
        )
    elif mutation == "gate_stale":
        evidence = replace(
            evidence,
            gate=replace(
                evidence.gate,
                bool_receipt_time_s=tick - 0.60,
                status_receipt_time_s=tick - 0.60,
            ),
        )
    elif mutation == "telemetry_stale":
        evidence = replace(
            evidence,
            telemetry=replace(
                evidence.telemetry,
                oldest_receipt_time_s=tick - 0.60,
            ),
        )
    elif mutation == "failsafe":
        evidence = replace(
            evidence,
            telemetry=replace(evidence.telemetry, failsafe=True),
        )
    elif mutation == "armed":
        evidence = replace(
            evidence,
            telemetry=replace(evidence.telemetry, vehicle_armed=True),
        )
    elif mutation == "offboard":
        evidence = replace(
            evidence,
            telemetry=replace(evidence.telemetry, offboard_active=True),
        )
    elif mutation == "dds_loss":
        evidence = replace(evidence, dds_ready=False)
    elif mutation == "invalid_mapping":
        evidence = replace(
            evidence,
            candidate=replace(current, frame_id="map"),
        )
    result = machine.step(tick, evidence, int(tick * 1e6))
    assert result.state == expected
    assert result.should_publish is False
    assert machine.fault_latched is True
    result = machine.step(
        tick + 0.01,
        readiness(tick + 0.01, current),
        int((tick + 0.01) * 1e6),
    )
    assert result.state == Px4StreamState.LATCHED_STREAM_FAULT
    assert "latched after:" in result.stop_reason
    assert "disable/reset is required" in result.stop_reason


def test_gate_bool_status_disagreement_fails_closed():
    """Treat disagreement between the two Phase 7 gate channels as a fault."""
    machine = Px4StreamStateMachine(config())
    _, now, current = begin_stream(machine)
    tick = now + 0.05
    evidence = readiness(tick, current)
    evidence = replace(
        evidence,
        gate=replace(evidence.gate, bool_safe_to_forward=False),
    )
    result = machine.step(tick, evidence, int(tick * 1e6))
    assert result.state == Px4StreamState.STOPPED_GATE_FALSE
    assert result.should_publish is False


def test_time_regression_and_nonmonotonic_output_stop_stream():
    """Latch both ROS/candidate regression and repeated outgoing timestamps."""
    machine = Px4StreamStateMachine(config())
    _, now, current = begin_stream(machine)
    result = machine.step(
        now - 0.01,
        readiness(now - 0.01, current),
        int((now - 0.01) * 1e6),
    )
    assert result.state == Px4StreamState.STOPPED_TIME_JUMP

    machine = Px4StreamStateMachine(config())
    _, now, current = begin_stream(machine)
    tick = now + 0.05
    current = candidate(tick, 10_000_004)
    machine.observe_candidate(current)
    result = machine.step(tick, readiness(tick, current), int(now * 1e6))
    assert result.state == Px4StreamState.STOPPED_TIME_JUMP


def test_publish_gap_stops_before_next_message_pair():
    """Do not publish a recovery pair after the maximum gap is exceeded."""
    machine = Px4StreamStateMachine(config(maximum_publish_gap_s=0.20))
    _, now, _ = begin_stream(machine)
    tick = now + 0.21
    current = candidate(tick, 10_000_004)
    machine.observe_candidate(current)
    result = machine.step(tick, readiness(tick, current), int(tick * 1e6))
    assert result.state == Px4StreamState.STOPPED_PUBLISH_GAP
    assert result.should_publish is False
    assert result.dropped_cycle_count >= 3


def test_single_scheduler_delay_warns_through_drop_count_but_keeps_streaming():
    """A sub-limit scheduler delay is measured without loosening freshness."""
    machine = Px4StreamStateMachine(config(maximum_publish_gap_s=0.20))
    _, now, _ = begin_stream(machine)
    tick = now + 0.11
    current = candidate(tick, 10_000_004)
    machine.observe_candidate(current)
    result = machine.step(tick, readiness(tick, current), int(tick * 1e6))
    assert result.should_publish is True
    assert result.dropped_cycle_count >= 1
    assert machine.maximum_observed_gap_s == pytest.approx(0.11)


def test_startup_delay_uses_heartbeat_readiness_not_sleep():
    """Remain waiting until three distinct current candidate updates exist."""
    machine = Px4StreamStateMachine(config())
    assert machine.request_enable(True)[0]
    first = candidate(100.0, 100_000_000)
    machine.observe_candidate(first)
    result = machine.step(100.0, readiness(100.0, first), 100_000_000)
    assert result.state == Px4StreamState.WAITING_CANDIDATE
    for index in (1, 2):
        tick = 100.0 + index * 0.05
        current = candidate(tick, 100_000_000 + index)
        machine.observe_candidate(current)
    result = machine.step(tick, readiness(tick, current), int(tick * 1e6))
    assert result.state == Px4StreamState.PRESTREAM_READY


def test_fault_recovery_requires_disable_repair_and_explicit_enable():
    """Healthy data alone cannot clear a latch; explicit recovery can."""
    machine = Px4StreamStateMachine(config())
    _, now, current = begin_stream(machine)
    tick = now + 0.05
    failed = readiness(tick, current, dds_ready=False)
    machine.step(tick, failed, int(tick * 1e6))
    assert machine.request_enable(True)[0] is False
    assert machine.request_enable(False)[0] is True
    current = prime(machine, tick + 0.10)
    assert machine.request_enable(True)[0] is True
    result = machine.step(
        tick + 0.10,
        readiness(tick + 0.10, current),
        int((tick + 0.10) * 1e6),
    )
    assert result.state == Px4StreamState.PRESTREAM_READY
    for index in (1, 2):
        recovery_tick = tick + 0.10 + index * 0.05
        current = candidate(
            recovery_tick,
            10_000_002 + index,
        )
        machine.observe_candidate(current)
        result = machine.step(
            recovery_tick,
            readiness(recovery_tick, current),
            int(recovery_tick * 1e6),
        )
    assert result.trajectory_setpoint_count == 3
    assert machine.observed_rate_hz == pytest.approx(20.0)
