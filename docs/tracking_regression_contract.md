# Phase 5 offline tracking regression contract

## Invariants

Every accepted trajectory has frame `px4_ned`, at least two finite points,
strictly increasing relative timestamps, and a matching fresh true validity
sample. Receipt time plus configured delay defines the epoch. Duplicate
identical trajectories do not reset it.

Every normal command is finite, uses `px4_ned`, has zero `angular.x/y`, obeys
total, horizontal, vertical, acceleration, yaw-rate, and yaw-acceleration
limits, and passes the independent validator. Every non-normal output is an
exact zero HOLD with a nonempty reason. No Phase 5 graph contains a publisher
whose topic begins `/fmu/in/`.

Terminal success requires final reference time and continuously satisfied
position, measured-speed, and wrapped-yaw tolerances for the complete settle
interval. Timeout without settling is a deterministic rejection/HOLD, not a
success.

## Deterministic fixtures

| # | Fixture | Expected result |
|---:|---|---|
| 1 | straight trajectory | tracking success and `GOAL_HOLD` |
| 2 | accepted Phase 3 B-spline trajectory | tracking success and `GOAL_HOLD` |
| 3 | A* fallback trajectory | tracking success and `GOAL_HOLD` |
| 4 | sharp but dynamically valid trajectory | bounded tracking success |
| 5 | start-position offset | feedback convergence and success |
| 6 | constant horizontal disturbance | bounded tracking success with measured error |
| 7 | duplicate trajectory message | accepted once; epoch unchanged |
| 8 | stale odometry | `HOLD_STALE_ODOMETRY` within configured timeout |
| 9 | stale trajectory validity | `HOLD_STALE_TRAJECTORY` within configured timeout |
| 10 | invalid validity flag | HOLD/rejection with invalid-validity reason |
| 11 | wrong odometry frame | `HOLD_INVALID_FRAME` |
| 12 | non-finite odometry | `HOLD_INVALID_COMMAND` with state diagnostic |
| 13 | backward time jump | `HOLD_TIME_JUMP`; history cleared; fresh sync required |
| 14 | command speed saturation | valid bounded command and speed saturation flag |
| 15 | command acceleration saturation | valid bounded command and acceleration flag |
| 16 | excessive tracking error | `HOLD_TRACKING_ERROR` |
| 17 | successful goal settling | full settle interval then `GOAL_HOLD` |
| 18 | terminal not reached | `TERMINAL_NOT_REACHED` and HOLD |
| 19 | yaw wrap crossing | shortest wrapped yaw feedback and bounded success |
| 20 | invalid command rejection | independent validator rejects and selects `HOLD_INVALID_COMMAND` |

## Required pure coverage

- sampler exact endpoints, before/inside/after behavior, every interpolated
  field, unwrapped-yaw interpolation, invalid timestamps, and non-finite data
- feedback sign and feedforward composition in NED
- each bound independently, combined ordered bounds, saturation reporting,
  HOLD construction, invalid command rejection, and validator independence
- missing/false/stale inputs, wrong frames, non-finite state, excessive error,
  backward time, equal time, and fresh synchronization after a jump
- terminal settling, tolerance break/reset, and terminal timeout
- fixed-step plant determinism, first-order response, acceleration limiting,
  disturbance, and default-disabled deterministic noise
- independent RMSE/max/rate/saturation/HOLD/stale/settling/completion metrics

## Required ROS graph coverage

The direct graph consists only of a fixed trajectory publisher, follower,
offline plant, and finite monitor. It verifies the four output topics, exact
frames, candidate Twist semantics, continuous bounded commands, state
progression, metrics, and absence of `/fmu/in/*`. Safety graph fixtures cover
at least stale odometry and invalid trajectory handling. The full graph adds
the fixed scene, A*, selected B-spline-or-fallback path, Phase 4
parameterizer, follower, plant, and monitor, and must end in `GOAL_HOLD`.

Required wrapper gates are:

```bash
./uav tracking-check
./uav tracking-safety-check
./uav full-pipeline-check
```

All existing Phase 2-4 unit, launch, wrapper, interface, import, and legacy
hash regressions remain mandatory. Generated outputs, logs, caches, and model
artifacts remain untracked.
