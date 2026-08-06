# Offline Control-Source Arbitration Comparison

## Method and scope

The table is rendered from the 24 deterministic pure fixtures in
`uav_px4_control.control_mux_fixtures`. Every scenario executes the source
registry, receipt-time freshness gates, arbitration state machine, ordered
selected-command limits, and independent selected-command validator. HOLD
cycles are exact zero. Candidate age is based on node-clock receipt time.

The final row exercises a follower-shaped A* command sequence through the mux
and terminal HOLD contract. The live ROS `control-stack-check` separately
validates the actual Phase 5 follower, mux-selected output, and offline plant.

| Fixture | Requested | Sequence | Service | Switch HOLD (s) | Max age (s) | Max speed (m/s) | Max yaw (rad/s) | HOLD cycles | Transitions | Latched | Recovery | Expected | Observed |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| startup-hold-no-source | HOLD | HOLD_STARTUP | none | 0.000 | 0.000 | 0.000 | 0.000 | 1 | 0 | no | n/a | HOLD_STARTUP | HOLD_STARTUP |
| select-fresh-astar | ASTAR_EXPERT | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT | accepted | 0.000 | 0.020 | 0.060 | 0.000 | 1 | 1 | no | n/a | ACTIVE_ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT |
| astar-active-stale | ASTAR_EXPERT | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_STALE_SOURCE | accepted | 0.000 | 0.020 | 0.060 | 0.000 | 2 | 2 | yes | explicit request | HOLD_STALE_SOURCE | HOLD_STALE_SOURCE |
| stale-fault-latch | ASTAR_EXPERT | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_STALE_SOURCE -> HOLD_LATCHED_FAULT | accepted | 0.000 | 0.020 | 0.060 | 0.000 | 3 | 2 | yes | explicit request | HOLD_LATCHED_FAULT | HOLD_LATCHED_FAULT |
| explicit-astar-recovery | ASTAR_EXPERT | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_STALE_SOURCE -> HOLD_LATCHED_FAULT -> ACTIVE_ASTAR_EXPERT | accepted/accepted | 0.000 | 0.220 | 0.060 | 0.000 | 3 | 3 | yes | explicit request | ACTIVE_ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT |
| astar-to-joystick | HUMAN_JOYSTICK | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_SWITCH_BARRIER -> ACTIVE_HUMAN_JOYSTICK | accepted/accepted | 0.100 | 0.120 | 0.135 | 0.000 | 2 | 3 | no | n/a | ACTIVE_HUMAN_JOYSTICK | ACTIVE_HUMAN_JOYSTICK |
| switch-barrier-observed | HUMAN_JOYSTICK | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_SWITCH_BARRIER | accepted/accepted | 0.100 | 0.100 | 0.060 | 0.000 | 4 | 2 | no | n/a | HOLD_SWITCH_BARRIER | HOLD_SWITCH_BARRIER |
| target-stale-during-handoff | HUMAN_JOYSTICK | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_SWITCH_BARRIER -> HOLD_STALE_SOURCE | accepted/accepted | 0.100 | 0.020 | 0.060 | 0.000 | 3 | 2 | yes | explicit request | HOLD_STALE_SOURCE | HOLD_STALE_SOURCE |
| joystick-to-navrl | NAVRL_POLICY | ACTIVE_HUMAN_JOYSTICK -> HOLD_SWITCH_BARRIER -> ACTIVE_NAVRL_POLICY | accepted/accepted | 0.100 | 0.110 | 0.105 | 0.000 | 1 | 3 | no | n/a | ACTIVE_NAVRL_POLICY | ACTIVE_NAVRL_POLICY |
| unknown-source-rejected | ASTAR | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_INVALID_SOURCE | accepted/rejected | 0.000 | 0.020 | 0.060 | 0.000 | 2 | 2 | yes | explicit request | HOLD_INVALID_SOURCE | HOLD_INVALID_SOURCE |
| selected-wrong-frame | ASTAR_EXPERT | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_WRONG_FRAME | accepted | 0.000 | 0.020 | 0.060 | 0.000 | 2 | 2 | yes | explicit request | HOLD_WRONG_FRAME | HOLD_WRONG_FRAME |
| selected-nonfinite | ASTAR_EXPERT | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_INVALID_COMMAND | accepted | 0.000 | 0.020 | 0.060 | 0.000 | 2 | 2 | yes | explicit request | HOLD_INVALID_COMMAND | HOLD_INVALID_COMMAND |
| selected-excessive-speed | ASTAR_EXPERT | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_INVALID_COMMAND | accepted | 0.000 | 0.020 | 0.060 | 0.000 | 2 | 2 | yes | explicit request | HOLD_INVALID_COMMAND | HOLD_INVALID_COMMAND |
| nonmonotonic-candidate-stamp | ASTAR_EXPERT | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_INVALID_COMMAND | accepted | 0.000 | 0.020 | 0.060 | 0.000 | 2 | 2 | yes | explicit request | HOLD_INVALID_COMMAND | HOLD_INVALID_COMMAND |
| backward-node-time | ASTAR_EXPERT | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_TIME_JUMP | accepted | 0.000 | 0.020 | 0.060 | 0.000 | 2 | 2 | yes | explicit request | HOLD_TIME_JUMP | HOLD_TIME_JUMP |
| minimum-source-dwell | HUMAN_JOYSTICK | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT | accepted/rejected | 0.000 | 0.050 | 0.105 | 0.000 | 1 | 1 | no | n/a | ACTIVE_ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT |
| duplicate-source-request | ASTAR_EXPERT | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT | accepted/accepted | 0.000 | 0.040 | 0.090 | 0.000 | 1 | 1 | no | n/a | ACTIVE_ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT |
| explicit-hold | HOLD | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_REQUESTED | accepted/accepted | 0.000 | 0.020 | 0.060 | 0.000 | 2 | 2 | no | n/a | HOLD_REQUESTED | HOLD_REQUESTED |
| invalid-external-hold | HOLD | HOLD_STARTUP | none | 0.000 | 0.000 | 0.000 | 0.000 | 1 | 0 | no | n/a | HOLD_STARTUP | HOLD_STARTUP |
| simultaneous-source-exclusivity | ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT | accepted | 0.000 | 0.020 | 0.060 | 0.000 | 0 | 1 | no | n/a | ACTIVE_ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT |
| unselected-stale-isolation | ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT | accepted | 0.000 | 0.020 | 0.375 | 0.000 | 0 | 1 | no | n/a | ACTIVE_ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT |
| selected-acceleration-limiter | ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT | accepted | 0.000 | 0.020 | 0.030 | 0.000 | 0 | 1 | no | n/a | ACTIVE_ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT |
| selected-yaw-acceleration-limiter | ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT | accepted | 0.000 | 0.020 | 0.090 | 0.040 | 0 | 1 | no | n/a | ACTIVE_ASTAR_EXPERT | ACTIVE_ASTAR_EXPERT |
| follower-mux-plant-terminal | ASTAR_EXPERT | HOLD_STARTUP -> ACTIVE_ASTAR_EXPERT -> HOLD_REQUESTED | accepted/accepted | 0.000 | 0.001 | 0.600 | 0.100 | 2 | 2 | no | n/a | GOAL_HOLD | GOAL_HOLD |

## Interpretation and limitations

The worst selected values in these fixtures remain below `2.0 m/s` and
`1.5 rad/s`; derivative-specific tests independently verify `1.5 m/s²` and
`2.0 rad/s²`. Immediate exact-zero fail-closed HOLD is a safety override and
is excluded from ordinary movement derivative measurements.

These are deterministic offline arbitration measurements. They are not PX4
setpoint-mapping validation, a hardware joystick test, a NavRL
policy/runtime/model test, Isaac Sim or vehicle-dynamics simulation, real
disturbance-rejection evidence, or flight validation.
