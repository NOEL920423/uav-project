"""Deterministic, PX4-independent Phase 8 stream regression fixtures."""

from dataclasses import replace

from uav_px4_control.px4_stream_models import (
    Px4StreamConfig,
    Px4StreamState,
    StreamCandidate,
    StreamGateEvidence,
    StreamReadiness,
    StreamTelemetry,
)
from uav_px4_control.px4_stream_state_machine import Px4StreamStateMachine


def _config() -> Px4StreamConfig:
    return Px4StreamConfig(
        minimum_prestream_duration_s=0.10,
        minimum_prestream_messages=2,
    )


def _candidate(now: float, sequence: int = 0) -> StreamCandidate:
    return StreamCandidate(
        receipt_time_s=now,
        timestamp_us=10_000_000 + sequence,
        velocity_ned_mps=(0.0, 0.0, 0.0),
        yaw_rate_ned_radps=0.0,
    )


def _readiness(now: float, current: StreamCandidate) -> StreamReadiness:
    return StreamReadiness(
        sitl_guard_valid=True,
        dds_ready=True,
        gate=StreamGateEvidence(
            bool_receipt_time_s=now,
            bool_safe_to_forward=True,
            status_receipt_time_s=now,
            status_safe_to_forward=True,
            status_state="SAFE_TO_FORWARD",
        ),
        candidate=current,
        telemetry=StreamTelemetry(
            oldest_receipt_time_s=now,
            newest_timestamp_us=int(now * 1e6),
            vehicle_armed=False,
            offboard_active=False,
            failsafe=False,
            odometry_valid=True,
        ),
    )


def _prime(machine: Px4StreamStateMachine, now: float = 10.0):
    current = None
    for index in range(3):
        current = _candidate(now - 0.10 + 0.05 * index, index)
        machine.observe_candidate(current)
    return current


def _begin(machine: Px4StreamStateMachine, now: float = 10.0):
    current = _prime(machine, now)
    machine.request_enable(True)
    machine.step(now, _readiness(now, current), int(now * 1e6))
    tick = now + 0.05
    current = _candidate(tick, 3)
    machine.observe_candidate(current)
    result = machine.step(tick, _readiness(tick, current), int(tick * 1e6))
    return result, tick, current


def run_stream_offline_fixtures() -> list[tuple[str, bool, str]]:
    """Run the required twenty named synthetic fixtures."""
    rows: list[tuple[str, bool, str]] = []

    machine = Px4StreamStateMachine(_config())
    current = _prime(machine)
    result = machine.step(10.0, _readiness(10.0, current), 10_000_000)
    rows.append((
        "startup-disabled",
        result.state == Px4StreamState.STREAM_DISABLED,
        result.state.value,
    ))
    rows.append((
        "gate-healthy-stream-disabled",
        result.trajectory_setpoint_count == 0,
        result.state.value,
    ))

    machine = Px4StreamStateMachine(_config())
    result, now, current = _begin(machine)
    rows.append((
        "explicit-stream-enable",
        result.should_publish,
        result.state.value,
    ))

    mutations = (
        (
            "candidate-stale",
            lambda item, tick: replace(
                item,
                candidate=replace(item.candidate, receipt_time_s=tick - 0.30),
            ),
            Px4StreamState.STOPPED_STALE_CANDIDATE,
        ),
        (
            "gate-false",
            lambda item, tick: replace(
                item,
                gate=replace(
                    item.gate,
                    bool_safe_to_forward=False,
                    status_safe_to_forward=False,
                    status_state="READY_DISABLED",
                ),
            ),
            Px4StreamState.STOPPED_GATE_FALSE,
        ),
        (
            "gate-status-stale",
            lambda item, tick: replace(
                item,
                gate=replace(
                    item.gate,
                    bool_receipt_time_s=tick - 0.60,
                    status_receipt_time_s=tick - 0.60,
                ),
            ),
            Px4StreamState.STOPPED_STALE_GATE,
        ),
        (
            "telemetry-stale",
            lambda item, tick: replace(
                item,
                telemetry=replace(
                    item.telemetry,
                    oldest_receipt_time_s=tick - 0.60,
                ),
            ),
            Px4StreamState.STOPPED_STALE_TELEMETRY,
        ),
        (
            "failsafe-true",
            lambda item, tick: replace(
                item,
                telemetry=replace(item.telemetry, failsafe=True),
            ),
            Px4StreamState.STOPPED_FAILSAFE,
        ),
        (
            "unexpected-armed",
            lambda item, tick: replace(
                item,
                telemetry=replace(item.telemetry, vehicle_armed=True),
            ),
            Px4StreamState.STOPPED_ARMED,
        ),
        (
            "unexpected-offboard-active",
            lambda item, tick: replace(
                item,
                telemetry=replace(item.telemetry, offboard_active=True),
            ),
            Px4StreamState.STOPPED_OFFBOARD_ACTIVE,
        ),
        (
            "invalid-mapping",
            lambda item, tick: replace(
                item,
                candidate=replace(item.candidate, frame_id="map"),
            ),
            Px4StreamState.STOPPED_INVALID_MAPPING,
        ),
    )
    for name, mutate, expected in mutations:
        machine = Px4StreamStateMachine(_config())
        _, now, _ = _begin(machine)
        tick = now + 0.05
        current = _candidate(tick, 4)
        machine.observe_candidate(current)
        evidence = mutate(_readiness(tick, current), tick)
        result = machine.step(tick, evidence, int(tick * 1e6))
        rows.append((
            name,
            result.state == expected and not result.should_publish,
            result.state.value,
        ))

    machine = Px4StreamStateMachine(_config())
    _, now, current = _begin(machine)
    result = machine.step(
        now - 0.01,
        _readiness(now, current),
        int(now * 1e6),
    )
    rows.append((
        "time-regression",
        result.state == Px4StreamState.STOPPED_TIME_JUMP,
        result.state.value,
    ))

    machine = Px4StreamStateMachine(_config())
    _, now, _ = _begin(machine)
    tick = now + 0.21
    current = _candidate(tick, 4)
    machine.observe_candidate(current)
    result = machine.step(
        tick,
        _readiness(tick, current),
        int(tick * 1e6),
    )
    rows.append((
        "publish-gap-fault",
        result.state == Px4StreamState.STOPPED_PUBLISH_GAP,
        result.state.value,
    ))

    machine = Px4StreamStateMachine(_config())
    _, now, current = _begin(machine)
    machine.request_enable(False)
    result = machine.step(
        now + 0.01,
        _readiness(now + 0.01, current),
        int((now + 0.01) * 1e6),
    )
    rows.append((
        "explicit-disable",
        result.state == Px4StreamState.STREAM_DISABLED,
        result.state.value,
    ))

    machine = Px4StreamStateMachine(_config())
    _, now, current = _begin(machine)
    failed = replace(_readiness(now + 0.05, current), dds_ready=False)
    machine.step(now + 0.05, failed, int((now + 0.05) * 1e6))
    result = machine.step(
        now + 0.06,
        _readiness(now + 0.06, current),
        int((now + 0.06) * 1e6),
    )
    rows.append((
        "latched-fault",
        result.state == Px4StreamState.LATCHED_STREAM_FAULT,
        result.state.value,
    ))
    machine.request_enable(False)
    current = _prime(machine, now + 0.20)
    machine.request_enable(True)
    result = machine.step(
        now + 0.20,
        _readiness(now + 0.20, current),
        int((now + 0.20) * 1e6),
    )
    rows.append((
        "explicit-reset-recovery",
        result.state == Px4StreamState.PRESTREAM_READY,
        result.state.value,
    ))

    machine = Px4StreamStateMachine(_config())
    machine.request_enable(True)
    first = _candidate(100.0, 0)
    machine.observe_candidate(first)
    result = machine.step(100.0, _readiness(100.0, first), 100_000_000)
    rows.append((
        "startup-scheduler-delay",
        result.state == Px4StreamState.WAITING_CANDIDATE,
        result.state.value,
    ))
    current = first
    for index in (1, 2):
        current = _candidate(100.0 + index * 0.05, index)
        machine.observe_candidate(current)
    result = machine.step(100.10, _readiness(100.10, current), 100_100_000)
    rows.append((
        "stable-heartbeat-readiness",
        result.state == Px4StreamState.PRESTREAM_READY,
        result.state.value,
    ))

    machine = Px4StreamStateMachine(_config())
    _, now, _ = _begin(machine)
    tick = now + 0.11
    current = _candidate(tick, 4)
    machine.observe_candidate(current)
    result = machine.step(
        tick,
        _readiness(tick, current),
        int(tick * 1e6),
    )
    rows.append((
        "single-delayed-timer-cycle",
        result.should_publish and result.dropped_cycle_count >= 1,
        result.state.value,
    ))
    for index in range(1, 4):
        tick += 0.10
        current = _candidate(tick, 4 + index)
        machine.observe_candidate(current)
        result = machine.step(
            tick,
            _readiness(tick, current),
            int(tick * 1e6),
        )
    rows.append((
        "repeated-timing-delay",
        result.should_publish and result.dropped_cycle_count >= 4,
        result.state.value,
    ))

    return rows


def main() -> int:
    """Print concise fixture evidence and return nonzero on any failure."""
    rows = run_stream_offline_fixtures()
    for name, passed, state in rows:
        print(f"{name}: {'PASS' if passed else 'FAIL'} state={state}")
    passed = sum(1 for _, result, _ in rows if result)
    print(f"px4 stream offline fixtures: {passed}/{len(rows)} PASS")
    if passed != len(rows):
        return 1
    print("px4 stream offline integration passed:")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
