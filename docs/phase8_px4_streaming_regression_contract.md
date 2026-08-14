# Phase 8 PX4 streaming regression contract

## Offline-first gate

The default suite imports no `px4_msgs` and starts no PX4, XRCE, simulator, or
live publisher. `./uav px4-stream-offline-check` runs the pure adapter, typed
state machine, ownership AST checks, and this deterministic matrix:

| Fixture | Required result |
|---|---|
| startup-disabled | `STREAM_DISABLED`, zero pairs |
| gate-healthy-stream-disabled | explicit enable still required |
| explicit-stream-enable | `PRESTREAM_READY -> PRESTREAMING` |
| candidate-stale | `STOPPED_STALE_CANDIDATE` |
| gate-false | `STOPPED_GATE_FALSE` |
| gate-status-stale | `STOPPED_STALE_GATE` |
| telemetry-stale | `STOPPED_STALE_TELEMETRY` |
| failsafe-true | `STOPPED_FAILSAFE` |
| unexpected-armed | `STOPPED_ARMED` |
| unexpected-offboard-active | `STOPPED_OFFBOARD_ACTIVE` |
| invalid-mapping | `STOPPED_INVALID_MAPPING` |
| time-regression | `STOPPED_TIME_JUMP` |
| publish-gap-fault | `STOPPED_PUBLISH_GAP` |
| explicit-disable | immediate `STREAM_DISABLED` |
| latched-fault | healthy data produces `LATCHED_STREAM_FAULT` |
| explicit-reset-recovery | repaired evidence reaches `PRESTREAM_READY` |
| startup-scheduler-delay | waits for heartbeats, no sleep assumption |
| stable-heartbeat-readiness | three current monotonic updates required |
| single-delayed-timer-cycle | warning/drop count, below failure limit |
| repeated-timing-delay | measured drops while every gap stays below limit |

DDS loss is additionally covered by the unit state matrix and must select
`STOPPED_DDS_LOSS`, stop both outputs, and latch.

## Static ownership contract

AST tests resolve all runtime `create_publisher` calls associated with
`/fmu/in/*`. The only permitted owner is
`px4_setpoint_streamer_node.py`, and the resolved set must equal the two-topic
allowlist. The runtime AST must not construct or publish `VehicleCommand`, mode
change, arm/disarm, takeoff, or land commands. Phase 8 launch files must not
start legacy direct publishers.

## Live acceptance sequence

Only after the full offline gate and five consecutive mux checks pass:

1. start PX4 SITL and Micro XRCE-DDS separately;
2. pass the read-only doctor;
3. observe the Phase 8 streamer at `STREAM_DISABLED`;
4. select a zero ASTAR candidate through the real mux;
5. explicitly enable the Phase 7 gate;
6. observe Phase 7 `SAFE_TO_FORWARD` while Phase 8 remains disabled;
7. explicitly enable Phase 8;
8. verify at least 40 message pairs, timing, timestamps, and exact mapping;
9. verify PX4 remains disarmed, OFFBOARD inactive, and not in failsafe;
10. explicitly disable and verify both publication counts stop.

Optional 0.10 m/s north/east/down and 0.10 rad/s yaw-rate fixtures repeat this
sequence separately only after zero succeeds. They are mapping diagnostics, not
flight tests.
