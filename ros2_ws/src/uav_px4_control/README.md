# uav_px4_control

Phase 6 implements a deterministic offline control-source multiplexer. Despite
the package name, there is still no PX4 adapter, `px4_msgs` dependency,
OFFBOARD/arming logic, simulator startup, or `/fmu/in/*` publisher.

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
```

The first two commands use synthetic candidates and a finite monitor. The last
connects the real scene/planner/parameterizer/follower pipeline through the mux
to the deterministic kinematic plant. These are non-flight tests and provide
no PX4 setpoint-mapping, vehicle-dynamics, hardware joystick, NavRL runtime,
Isaac Sim, or flight evidence.

The future PX4 output adapter, if separately authorized and safety-gated, must
remain the sole owner of actual `/fmu/in/*` publications. It is not part of
Phase 6.
