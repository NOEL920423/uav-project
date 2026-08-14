# Phase 8 PX4 QoS and timestamp findings

## Audited local contract

- PX4 checkout: v1.14.3 source at commit
  `1dacb4cdef2d7145754fc788fa8dc482eed74b40` (pre-existing dirty files were
  recorded and not modified by this project).
- `px4_msgs`: v1.14.0 at commit
  `ffb6e80e1c17e5714395611a020c282a87af8fa4`, clean.
- The six Phase 8 message definitions are compatible with the audited PX4
  checkout: `TrajectorySetpoint`, `OffboardControlMode`, `VehicleStatus`,
  `VehicleControlMode`, `VehicleOdometry`, and `FailsafeFlags`.

The external `px4_msgs` source was supplied as a read-only additional colcon
base path. It was not copied, vendored, or modified. The local, pre-existing
dirty `dds_topics.yaml` exposes both allowed PX4 inputs and all four required
outputs; Phase 8 did not change it.

## Live endpoint and QoS evidence

The live graph on 2026-08-10 reported one PX4 subscriber for each allowed
input and one PX4 publisher for each required telemetry output:

| Direction/topic | Count | Reliability | Durability | Graph history/depth |
|---|---:|---|---|---|
| input `/fmu/in/trajectory_setpoint` subscriber | 1 | `BEST_EFFORT` | `VOLATILE` | `UNKNOWN` / 0 |
| input `/fmu/in/offboard_control_mode` subscriber | 1 | `BEST_EFFORT` | `VOLATILE` | `UNKNOWN` / 0 |
| each required `/fmu/out/*` publisher | 1 | `BEST_EFFORT` | `TRANSIENT_LOCAL` | `UNKNOWN` / 0 |

The sole tracked Phase 8 publisher offered `BEST_EFFORT`,
`TRANSIENT_LOCAL`, keep-last depth 1 for both inputs. Fast DDS reported the
bare-DDS PX4 endpoint history/depth as unknown/zero, but reliability and
durability were compatible and both topics were received by PX4. There was no
publisher on `/fmu/in/vehicle_command`.

Raw live receipt measurements before choosing the live-only telemetry timeout
were approximately:

| Telemetry | Observed rate | Largest observed interval |
|---|---:|---:|
| `vehicle_status` | 1.870 Hz | 0.542 s |
| `vehicle_control_mode` | 1.878 Hz | 0.536 s |
| `vehicle_odometry` | about 120 Hz | below the low-rate topics |
| `failsafe_flags` | 1.742 Hz | 0.575 s |

Consequently the live launch overrides only `telemetry_timeout_s` from the
typed/offline default 0.50 s to 0.75 s. Candidate timeout remains 0.25 s,
gate-status timeout remains 0.50 s, the Phase 7 limits are unchanged, and the
0.20 s publish-gap fault remains unchanged. An earlier live run correctly
stopped after 11 pairs when the 0.50 s telemetry timeout was crossed; this was
not hidden by retrying.

## Timestamp and timing evidence

For every output pair, one ROS node-clock sample is converted as:

```text
timestamp_us = ros_time_nanoseconds // 1000
```

The exact same positive timestamp is assigned to `TrajectorySetpoint` and
`OffboardControlMode`. The state machine separately validates Phase 7
candidate timestamp monotonicity and outgoing strict monotonicity. Zero,
backward, or repeated outgoing timestamps stop output and latch a fault. Wall
clock is not read directly.

The final zero live run produced 41 matched pairs. Its first and last
`TrajectorySetpoint.timestamp` values were `1786355203010268` and
`1786355205010320`; all samples were strictly increasing. The observed rate
was 20.000656 Hz, minimum interval 0.047295 s, maximum interval 0.052220 s,
and RMS jitter from the 0.050 s target was 0.001016 s. The maximum stayed far
below the 0.20 s safety limit, so no cycle was dropped.

## DDS restart observation

Stopping the XRCE Agent made the streamer report `dds_ready=false`, stop both
topics, and latch. Restarting the Agent restored telemetry and endpoint
discovery but did not resume streaming. During discovery convergence, one
read-only doctor invocation observed the trajectory subscriber temporarily at
zero while the mode subscriber was already one and therefore failed safe. A
subsequent detailed graph observation showed both at one. The doctor now waits
for three consecutive complete process/telemetry/endpoint snapshots before its
final validation; it does not relax any acceptance condition.
