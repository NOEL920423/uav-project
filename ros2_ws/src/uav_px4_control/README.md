# uav_px4_control

Phase 6 implements a deterministic offline control-source multiplexer. Phase 7
adds the validated diagnostic PX4 candidate and first fail-closed gate. Phase 8
adds a separately enabled, SITL-only streamer whose sole live ownership is:

- `/fmu/in/trajectory_setpoint`
- `/fmu/in/offboard_control_mode`

It has no `VehicleCommand`, mode-change, arm/disarm, takeoff/land, simulator
startup, Isaac Sim, real-vehicle, or flight behavior.

## Contract

Candidate inputs are `/uav/control/astar_command`,
`/uav/control/joystick_command`, `/uav/control/navrl_command`, and
`/uav/control/hold_command`. Canonical source identifiers are exactly `HOLD`,
`ASTAR_EXPERT`, `HUMAN_JOYSTICK`, and `NAVRL_POLICY`.

The mux exclusively publishes:

- `/uav/control/selected_command` (`geometry_msgs/msg/TwistStamped`)
- `/uav/control/source` (`std_msgs/msg/String`)
- `/uav/control/mux_status` (`uav_interfaces/msg/ControlMuxStatus`)

Selection requests use `/uav/control/set_source`
(`uav_interfaces/srv/SetControlSource`). Output stamps use mux ROS time.
Freshness uses receipt time. Movement-to-movement handoffs pass through an
exact-zero HOLD barrier, and selected-source faults latch HOLD until an
explicit healthy request. The internal exact-zero HOLD remains available even
if the external HOLD candidate is absent or invalid.

Configuration is installed from `config/control_mux.yaml`. The 19 mux contract
parameters plus `use_sim_time` cover publish rate, four source timeouts,
handoff/dwell timing, selected velocity/acceleration/yaw bounds, frame and
timestamp gates, fresh-before-switch, fault latching, and HOLD epsilon.

## Offline verification

```bash
./uav mux-check
./uav mux-safety-check
./uav control-stack-check
./uav px4-map-check
./uav px4-gate-check
./uav px4-boundary-check
```

The first two commands use synthetic candidates and a finite monitor. The last
connects the real scene/planner/parameterizer/follower pipeline through the mux
to the deterministic kinematic plant. These are non-flight tests and provide
no vehicle-dynamics, hardware joystick, NavRL runtime, Isaac Sim, or flight
evidence. The Phase 7 checks stop at `/uav/px4/safe_to_forward`, including an
injected synthetic failsafe and explicit latch recovery.

The future PX4 output adapter, if separately authorized and safety-gated, must
remain the sole owner of actual `/fmu/in/*` publications. It is not part of
Phase 7.

## Phase 8 stream contract

`px4_setpoint_streamer_node` subscribes only to the Phase 7 candidate,
`safe_to_forward`, detailed gate status, and four read-only PX4 telemetry
topics. It requires explicit stream enable plus Phase 7 approval, stable
candidate heartbeats, a local SITL guard, DDS subscribers, fresh telemetry,
disarmed state, OFFBOARD inactive, and no failsafe. It starts disabled and
stops both publications on any fault. Faults latch until explicit
disable/reset, repair, and re-enable.

```bash
./uav px4-stream-offline-check
./uav px4-sitl-doctor
./uav px4-sitl-stream-check
```

The first command is PX4-independent. The latter two require an already-running
local PX4 SITL and XRCE Agent; the doctor is read-only. See
`docs/phase8_live_px4_streaming_design.md` for dependency, mapping, timing, and
recovery details.

## Guarded SITL flight milestone

The flight milestone reuses the Phase 8 streamer and adds one explicitly
enabled supervisor as the sole `/fmu/in/vehicle_command` owner. It sequences
A*, B-spline/timed trajectory, follower, `ASTAR_EXPERT` mux selection, gate,
prestream, OFFBOARD, arm, takeoff, mission tracking, goal, and PX4 AUTO_LAND.

```bash
UAV_OFFLINE_TIMEOUT_SECONDS=150 ./uav px4-sitl-flight-check
```

This command flies an already-running local PX4 SITL. It starts with the
read-only SITL doctor and emits a finite JSON evidence record under
`run_logs/`. See `docs/px4_sitl_flight_milestone.md` for the exact safety
contract, accepted timeline, and limitations.
