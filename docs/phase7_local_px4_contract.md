# Phase 7 Local PX4 Contract Audit

## Scope and versions

This is a read-only audit performed before Phase 7 implementation. The legacy
workspace and firmware checkout were not sourced, built, or modified.

- PX4 firmware: `v1.14.3-dirty`, commit
  `1dacb4cdef2d7145754fc788fa8dc482eed74b40`. Existing unrelated changes are
  present in `Tools/setup/requirements.txt` and
  `src/modules/uxrce_dds_client/dds_topics.yaml`, plus untracked local files.
- `px4_msgs`: `v1.14.0`, commit
  `ffb6e80e1c17e5714395611a020c282a87af8fa4`, clean.
- The audited `TrajectorySetpoint`, `OffboardControlMode`, `VehicleCommand`,
  `VehicleStatus`, `VehicleOdometry`, `FailsafeFlags`, and
  `VehicleControlMode` definitions are byte-equivalent between these two
  checkouts.

## Exact setpoint contract

Local `TrajectorySetpoint` has `uint64 timestamp`, four `float32[3]` arrays
named `position`, `velocity`, `acceleration`, and `jerk`, plus `float32 yaw`
and `float32 yawspeed`. It is explicitly a NED-frame PID-position-controller
input. Units are m, m/s, m/s², m/s³, rad, and rad/s. The message says NaN means
that state is not controlled; jerk is logging-only.

Phase 7 therefore represents velocity-only mapping as follows:

| Local field | Phase 7 value |
|---|---|
| `position[3]` | all NaN |
| `velocity[3]` | selected north/east/down, unchanged |
| `acceleration[3]` | all NaN |
| `jerk[3]` | all NaN |
| `yaw` | NaN |
| `yawspeed` | selected NED yaw rate |

Firmware `PositionControl::empty_trajectory_setpoint` initializes every vector,
yaw, and yawspeed to NaN. During update, a finite yawspeed is retained and an
unset yaw falls back to current vehicle yaw. This justifies independent
yaw-rate mapping without inventing absolute yaw or integrating yaw rate.

`OffboardControlMode` contains timestamp and the exact booleans `position`,
`velocity`, `acceleration`, `attitude`, `body_rate`, and `actuator`. The local
commander gives the first true field priority. A future velocity adapter would
set only `velocity=true`; Phase 7 does not create or publish this message.

## State evidence

`VehicleStatus` exposes `arming_state` (`INIT=0`, `STANDBY=1`, `ARMED=2`, and
error/shutdown states), `nav_state` (`OFFBOARD=14`), `failsafe`,
`failure_detector_status`, and `pre_flight_checks_pass`.
`VehicleControlMode` separately exposes `flag_armed` and
`flag_control_offboard_enabled`. `FailsafeFlags` exposes local position and
velocity invalidity plus `offboard_control_signal_lost` and other failures.

`VehicleOdometry` timestamps are microseconds since system start. Pose frame
`1` is NED; velocity frame `1` is NED. Position/velocity entries use NaN for
invalid data and quaternion validity is indicated by its first element.
Synthetic Phase 7 telemetry models only evidence derivable from these fields.

## Timestamp and bridge finding

All audited uORB-style timestamps are `uint64` microseconds since system start.
The local `uxrce_dds_client` performs agent time synchronization and maintains
a nanosecond session offset derived from its microsecond `Timesync`. Legacy ROS
2 publishers use ROS clock nanoseconds divided by 1000. Phase 7 mirrors that
candidate convention with exact integer `nanoseconds // 1000` truncation, checks
uint64 overflow and monotonicity, and does not claim boot-relative correctness
without a live bridge. No wall-clock or PX4 process is used.

## Pre-Offboard legacy behavior

The legacy ROS 2 lookahead follower runs at 20 Hz and configures `prestream_s`
as 2.0 seconds before requesting OFFBOARD/ARM. Another legacy follower streams
the first 40 ticks at 20 Hz, then retries mode/arm every 10 ticks through tick
120. The direct MAVLink controller uses 80 warm-up velocity setpoints at 20 Hz
(4 seconds). These are local implementation policies. Firmware instead checks
that a recent `OffboardControlMode` has a selected control level and, for
velocity mode, valid local velocity; freshness is governed by `COM_OF_LOSS_T`.
Phase 7 models readiness only and sends no prestream, OFFBOARD, ARM, or command.

## Compatibility and known mismatch

The message definitions currently match exactly despite the firmware checkout
being dirty and tagged three patch releases beyond `px4_msgs`. Legacy code owns
real `/fmu/in/offboard_control_mode`, `/fmu/in/trajectory_setpoint`, and
`/fmu/in/vehicle_command` publishers and sometimes retries OFFBOARD/ARM. That
behavior is deliberately not migrated: Phase 7 stops at diagnostic
`safe_to_forward` and has no real PX4 publisher.
