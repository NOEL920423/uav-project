# Phase 8 live PX4 SITL setpoint streaming design

## Scope and hard boundary

Phase 8 is a SITL-only ROS 2 message-boundary verification. It does not request
OFFBOARD, arm, disarm, take off, land, start Isaac Sim, or control a real
vehicle.

```text
PHASE 8 LIVE PX4 INPUT ALLOWLIST

/fmu/in/trajectory_setpoint
/fmu/in/offboard_control_mode

PHASE 8 FORBIDDEN

/fmu/in/vehicle_command
OFFBOARD MODE REQUEST
ARM
DISARM
TAKEOFF
LAND
ISAAC SIM
REAL VEHICLE
```

`uav_px4_control.px4_setpoint_streamer_node` is the sole tracked owner of both
allowed publishers. It consumes only the validated Phase 7 boundary:

```text
selected_command -> Phase 7 mapper -> setpoint_candidate
                 -> Phase 7 output gate -> safe_to_forward
                 -> Phase 8 streamer -> the two allowed PX4 inputs
```

It never subscribes to individual A*, joystick, or NavRL candidates. Legacy
direct publishers are neither changed nor launched.

## Exact mapping

The selected command is already `px4_ned`. North, east, signed down, and NED
yaw rate are copied without another frame conversion:

| `TrajectorySetpoint` field | Phase 8 value |
|---|---|
| `timestamp` | ROS node nanoseconds divided by 1000 |
| `position` | `[NaN, NaN, NaN]` |
| `velocity` | `[north, east, down]` |
| `acceleration` | `[NaN, NaN, NaN]` |
| `jerk` | `[NaN, NaN, NaN]` |
| `yaw` | `NaN` |
| `yawspeed` | candidate NED yaw rate |

`OffboardControlMode.timestamp` uses the same tick timestamp. Its flags are
exactly `position=false`, `velocity=true`, `acceleration=false`,
`attitude=false`, `body_rate=false`, and `actuator=false`. This declares a
velocity setpoint type; it is not an OFFBOARD mode request.

## Independent gates and state machine

Startup is `STREAM_DISABLED`. Publication requires all of the following on the
same cycle:

- explicit `/uav/px4/set_stream_enable` enable;
- Phase 7 boolean and detailed status agreement at `SAFE_TO_FORWARD`;
- at least three distinct, current, monotonic candidate heartbeats;
- fresh `VehicleStatus`, `VehicleControlMode`, `VehicleOdometry`, and
  `FailsafeFlags`;
- disarmed, OFFBOARD inactive, no failsafe, and valid NED odometry;
- one or more DDS subscribers for each allowed input;
- an observable local PX4 SITL process matching the configured build identity.

Waiting states do not publish. Once all evidence is stable, the node reports
`PRESTREAM_READY`, then publishes both messages as an atomic pair in
`PRESTREAMING`. It reports `STREAMING` only after at least 2.0 seconds and 40
pairs at the configured 20 Hz. No mode command follows that transition.

Runtime loss selects a typed `STOPPED_*` state and stops both publications.
Faults include gate false/disagreement, candidate/gate/telemetry staleness,
DDS loss, failsafe, unexpected armed or OFFBOARD state, invalid mapping,
backward/non-monotonic timestamps, and a publish gap above 0.20 seconds.
Faults latch. Recovery is exactly disable/reset, repair, establish fresh
readiness again, and explicitly enable. Fresh data alone never resumes output.

## Timing and stop semantics

The state machine uses receipt time for freshness and ROS node time for outgoing
microseconds. Candidate and outgoing timestamps must be monotonic. It tracks
counts, achieved rate, largest interval, and inferred dropped cycles. A
sub-limit scheduling delay is measured as a timing warning; a real maximum-gap
violation revokes output before another pair is published.

Message counts and dropped-cycle totals are node-lifetime cumulative. The
reported achieved rate is scoped to the current continuous stream session, so
an explicit disable/recovery interval cannot dilute the diagnostic rate.

Disabled or faulted means no `TrajectorySetpoint` and no
`OffboardControlMode` publication. Phase 8 deliberately does not send a zero
velocity rescue message because it never intentionally enters OFFBOARD flight.
In-flight command semantics are a later, separately reviewed phase.

## Local dependency resolution

The normal seven-package workspace remains PX4-independent. `px4_msgs` is an
execution-only dependency loaded by live nodes at construction time. For live
checks only, the clean v1.14.0 source at
`/home/noel_614420090/uav_ros2_ws/src/px4_msgs` is added as a read-only external
colcon base path and built into this project's `ros2_ws/install`. The legacy
workspace overlay is never sourced, its source is not modified, and no message
repository is copied or vendored into Git.
