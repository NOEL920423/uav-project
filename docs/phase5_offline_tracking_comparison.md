# Offline Closed-Loop Tracking Comparison

## Method and scope

These results come from `uav_navigation.tracking_comparison` using the pure
Phase 5 follower, independent command validator, and deterministic 0.02 s
fixed-step first-order kinematic plant. Each source path is processed by the
Phase 4 parameterizer before tracking. Noise is disabled. Every row completed
in exact-zero `GOAL_HOLD` after continuously meeting all goal gates for 0.50 s.

This is not PX4, Isaac Sim, Pegasus, a vehicle-dynamics model, or real flight.
The NED velocity/yaw-rate candidate has not been mapped to a PX4 setpoint.
The disturbance row demonstrates deterministic behavior for one synthetic
constant disturbance; it is not proof of general disturbance rejection or
robustness.

## Position and tracking error metrics

| Fixture | Source | Points | Trajectory / simulation (s) | Position RMSE H/V (m) | Total pos. RMSE (m) | Max / terminal pos. error (m) | Velocity RMSE (m/s) | Yaw RMSE / max (rad) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| straight-trajectory | direct straight path | 3 | 2.309401 / 3.960000 | 0.221931 / 0.000000 | 0.221931 | 0.388565 / 0.091504 | 0.300666 | 0.000000 / 0.000000 |
| phase3-bspline-accepted | accepted Phase 3 B-spline-shaped path | 5 | 2.414285 / 3.680000 | 0.159178 / 0.000000 | 0.159178 | 0.224366 / 0.093548 | 0.236817 | 0.071139 / 0.180595 |
| astar-fallback | validated A* fallback path | 4 | 4.253235 / 5.200000 | 0.235327 / 0.000000 | 0.235327 | 0.348337 / 0.092533 | 0.158625 | 0.489456 / 1.295633 |
| sharp-dynamically-valid | sharp valid path | 4 | 3.073482 / 3.900000 | 0.130146 / 0.000000 | 0.130146 | 0.193377 / 0.066005 | 0.069727 | 0.648192 / 1.308560 |
| start-position-offset | straight path, 0.8 m offset | 3 | 2.309401 / 3.980000 | 0.436322 / 0.000000 | 0.436322 | 0.792633 / 0.092732 | 0.425882 | 0.000000 / 0.000000 |
| constant-horizontal-disturbance | straight path, synthetic crosswind | 3 | 2.309401 / 4.240000 | 0.230079 / 0.000000 | 0.230079 | 0.392804 / 0.118032 | 0.298154 | 0.000000 / 0.000000 |
| command-speed-saturation | long straight speed-demand path | 3 | 2.309401 / 3.860000 | 1.218889 / 0.000000 | 1.218889 | 1.871244 / 0.058310 | 0.689449 | 0.000000 / 0.000000 |
| yaw-wrap-crossing | unwrapped yaw across pi | 4 | 2.886788 / 4.820000 | 0.257554 / 0.000000 | 0.257554 | 0.388568 / 0.091444 | 0.289618 | 1.562690 / 3.132607 |

## Command, state, and completion metrics

| Fixture | Max speed / accel (m/s, m/s²) | Max yaw rate / accel (rad/s, rad/s²) | Saturations | HOLD cycles | Stale latency (s) | Settling (s) | Completion | Rejection / HOLD reason |
|---|---:|---:|---:|---:|---:|---:|---|---|
| straight-trajectory | 1.832985 / 1.500000 | 0.000000 / 0.000000 | 134 | 6 | 0.000000* | 0.500000 | GOAL_HOLD | none; terminal goal HOLD |
| phase3-bspline-accepted | 1.594934 / 1.500000 | 0.650601 / 2.000000 | 127 | 6 | 0.000000* | 0.500000 | GOAL_HOLD | none; terminal goal HOLD |
| astar-fallback | 0.775018 / 1.040705 | 1.500000 / 2.000000 | 149 | 6 | 0.000000* | 0.500000 | GOAL_HOLD | none; terminal goal HOLD |
| sharp-dynamically-valid | 0.620042 / 1.300143 | 1.500000 / 2.000000 | 163 | 6 | 0.000000* | 0.500000 | GOAL_HOLD | none; terminal goal HOLD |
| start-position-offset | 1.985406 / 1.500000 | 0.000000 / 0.000000 | 166 | 6 | 0.000000* | 0.500000 | GOAL_HOLD | none; terminal goal HOLD |
| constant-horizontal-disturbance | 1.834586 / 1.500000 | 0.000000 / 0.000000 | 135 | 6 | 0.000000* | 0.500000 | GOAL_HOLD | none; terminal goal HOLD |
| command-speed-saturation | 2.000000 / 1.500000 | 0.000000 / 0.000000 | 203 | 6 | 0.000000* | 0.500000 | GOAL_HOLD | none; terminal goal HOLD |
| yaw-wrap-crossing | 2.000000 / 1.500000 | 1.500000 / 2.000000 | 308 | 6 | 0.000000* | 0.500000 | GOAL_HOLD | none; terminal goal HOLD |

`*` No stale input is injected in these eight success comparisons, so the
accumulator retains its zero not-observed value. The separate stale-odometry
ROS safety fixture verifies timeout detection and exact-zero
`HOLD_STALE_ODOMETRY` behavior.

Command acceleration and yaw-acceleration maxima cover ordinary controller
candidates. Immediate exact-zero fail-closed HOLD transitions are safety
overrides: they remain included in HOLD-cycle counts but are excluded from the
normal candidate derivative maxima. The following normal candidate, if any,
is still measured relative to the previously published HOLD.

## Live ROS integration observation

The complete live offline graph processed the fixed scene through A*, selected
the validated 55-point B-spline, produced an 8.066690 s timed trajectory, then
ran the follower and plant to `GOAL_HOLD`. Its independent monitor observed
about 0.0391 m position RMSE and 0.0657 m maximum position error. That separate
integration observation is not substituted into the pure comparison rows.
