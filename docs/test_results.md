# Test results

## Phase 6.1 mux freshness synchronization stabilization

Date: 2026-08-10 (Asia/Taipei)

The Phase 8 baseline failure was investigated without starting Phase 8 or any
PX4/simulator process. The original mux monitor logged
`ACTIVE_ASTAR_EXPERT`, then a genuine receipt-time stale/latch transition, but
timed out at `WAIT_ASTAR` because asynchronous service-response polling had
lost the earlier ACTIVE event. The offline monitor now requires three recent,
monotonic candidate heartbeats, submits source requests only when the service
call is actually started, and retains ACTIVE observations across response
ordering. Production `astar_timeout_s=0.25 s` and fail-closed behavior are
unchanged.

```text
focused mux registry/state/ROS contract tests: 45 passed
complete workspace: 7 packages, 217 tests, 0 errors, 0 failures, 0 skipped
nominal mux-check stability: 10/10 passed
intentional stale-source stability: 3/3 passed
control-stack-check: PASS, GOAL_HOLD
Phase 2 through Phase 7 wrapper regression: PASS
/fmu/in/*: absent
```

Two nominal stability runs captured startup ASTAR gaps of `0.367503 s` and
`0.367946 s`. Selection remained in startup HOLD until sustained traffic was
established, then completed A* → joystick → NavRL → HOLD without an accidental
stale transition. The three intentional stale fixtures recorded
`0.639714-0.640463 s` gaps and every run still entered stale HOLD, latched, and
required explicit recovery. See `phase6_1_mux_timing_investigation.md` for the
timeline and evidence.

## Phase 6 offline control-source arbitration

Date: 2026-08-06 (Asia/Taipei)

Phase 6 adds a pure source registry, freshness monitor, fail-closed arbitration
state machine, ordered selected-command limits, independent validator, ROS 2
mux/service/status adapter, and finite offline harnesses. All 24 deterministic
fixtures reached their expected terminal state. The focused
`uav_px4_control` suite reported 62 passed tests; the complete seven-package
workspace reported 159 tests, 0 errors, 0 failures, and 0 skipped.

```text
./uav mux-check
  ASTAR_EXPERT -> HOLD barrier -> HUMAN_JOYSTICK -> HOLD barrier
  -> NAVRL_POLICY -> HOLD: PASS
./uav mux-safety-check
  selected source stale -> latched HOLD -> explicit recovery: PASS
  invalid external HOLD -> internal exact-zero HOLD remains available: PASS
./uav control-stack-check
  scene -> A* -> 55-point B-spline -> 8.066690 s trajectory
  -> follower -> mux -> plant -> GOAL_HOLD: PASS
```

The live control-stack monitor observed 430 movement cycles with one transition
to `ACTIVE_ASTAR_EXPERT`. Nominal mux validation observed 10 HOLD barrier
cycles, nonzero movement output, and six state transitions. Selected output stayed
within 2.0 m/s and 1.5 rad/s; pure derivative fixtures also enforce
1.5 m/s² and 2.0 rad/s². No `/fmu/in/*` topic existed.

These results validate offline arbitration contracts only. They do not validate
PX4 setpoint mapping, OFFBOARD, arming, joystick hardware, NavRL inference,
Isaac Sim, vehicle dynamics, or flight. See
`phase6_control_mux_comparison.md` for the 24-fixture table.

## Phase 5 offline closed-loop trajectory tracking

Date: 2026-08-06 (Asia/Taipei)

Phase 5 adds a pure deterministic sampler/controller/validator/plant/metrics
stack, its ROS 2 adapter and status interface, two finite offline tracking
graphs, and a complete scene-to-tracking graph. The direct nominal graph ended
in `GOAL_HOLD`; the stale-odometry graph selected exact-zero
`HOLD_STALE_ODOMETRY`; and the complete graph accepted a 55-point Phase 3
B-spline, parameterized it to 8.066690 s, and ended in `GOAL_HOLD` with
approximately 0.0391 m position RMSE and 0.0657 m maximum position error.

Focused package validation reported 99 tests, 0 errors, 0 failures, and
0 skipped. Coverage includes all 20 contract fixtures, interpolation fields
and endpoints, NED feedback and feedforward, ordered bounds, independent
command rejection, every input HOLD gate, time jumps, continuous goal
settling, deterministic plant behavior, metrics, ROS graph contracts, and
absence of `/fmu/in/*`.

```text
./uav tracking-check
  fixed trajectory -> follower -> plant -> GOAL_HOLD: PASS
./uav tracking-safety-check
  stale odometry -> exact-zero HOLD_STALE_ODOMETRY: PASS
./uav tracking-safety-check fixture:=invalid-validity-flag
  false validity -> exact-zero WAITING_VALIDITY: PASS
./uav tracking-safety-check fixture:=wrong-odometry-frame
  map odometry -> exact-zero HOLD_INVALID_FRAME: PASS
./uav tracking-safety-check fixture:=excessive-tracking-error
  3.001697 m maximum error -> HOLD_TRACKING_ERROR: PASS
./uav full-pipeline-check
  scene -> A* -> B-spline -> timed trajectory -> follower -> plant: PASS
```

All numerical results are offline kinematic fixture measurements. They are not
PX4 setpoint-mapping validation, UAV dynamics simulation, Isaac Sim results,
real-flight evidence, or a proof of disturbance rejection. See
`phase5_offline_tracking_comparison.md` for the eight-fixture table.

## Phase 2 A* migration

Date: 2026-08-06 (Asia/Taipei)

Environment:

- ROS 2 Jazzy from `/opt/ros/jazzy`
- system Python 3.12.3
- branch `feature/ros2-astar-bspline-integration`
- no legacy workspace sourced

### Package build and tests

The incremental pre-commit validation completed with:

```text
colcon build --symlink-install --packages-select uav_navigation
Summary: 1 package finished

colcon test --packages-select uav_navigation
27 pytest items passed

colcon test-result --verbose
Summary: 37 tests, 0 errors, 0 failures, 0 skipped
```

Coverage includes coordinate basis/inverse/offset boundaries, geometry,
planning and validation radii, 2.5D overflight thresholds, deterministic A*,
endpoint rejection/recovery, wide and narrow gaps, bounded no-path, exact
endpoints, safe simplification fallback, geometric metrics, flake8, and pep257.

The final clean whole-workspace result is recorded below.

### Offline ROS 2 integration

Command:

```bash
source /opt/ros/jazzy/setup.bash
source /home/noel_614420090/uav-project/ros2_ws/install/setup.bash
ros2 launch uav_navigation astar_planner_offline.launch.py
```

Observed result:

```text
Published fixed offline obstacle/start/goal
SUCCESS|method=rdp|raw_points=81|simplified_points=6|final_points=6
|length_m=4.538611|minimum_physical_clearance_m=0.460000|fallback=none
offline integration passed: raw=81, simplified=6, final=6, frame=px4_ned
```

The harness independently checked the converted exact endpoints `(0, -2, -2)`
and `(0, 2, -2)`, every final segment against the validation envelope, all
three nonempty path messages, and the absence of `/fmu/in/*`. The fixture exit
triggered a clean launch shutdown of the persistent planner.

Representative durable final-path echo:

```text
header.frame_id: px4_ned
first position: {x: 0.0, y: -2.0, z: -2.0}
last position:  {x: 0.0, y:  2.0, z: -2.0}
pose count: 6
identity orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
```

Topic types observed while the planner remained alive:

```text
/uav/scene/obstacles [uav_interfaces/msg/ObstacleArray]
/uav/scene/start [geometry_msgs/msg/PoseStamped]
/uav/scene/goal [geometry_msgs/msg/PoseStamped]
/uav/planner/path_raw [nav_msgs/msg/Path]
/uav/planner/path_simplified [nav_msgs/msg/Path]
/uav/planner/path [nav_msgs/msg/Path]
/uav/planner/status [std_msgs/msg/String]
```

`ros2 topic list | grep '^/fmu/in/'` returned no lines. No PX4, Isaac Sim,
Pegasus, or XRCE process was launched by the offline graph. A pre-existing
behavior-cloning inference worker (PIDs 153294/153301, started 2026-07-21) uses
an Isaac-distribution Python executable but is unrelated to this launch; its
start time predates this Phase 2 test by 16 days.

### Final clean validation

Existing `build/`, `install/`, and `log/` directories were moved without
deletion to `/tmp/uav-phase2-build.hT2Fi2`. An `env -i` shell used only
`/usr/bin:/bin`, ROS 2 Jazzy, `LANG=C.UTF-8`, and system Python 3.12.3.

```text
colcon build --symlink-install
Summary: 7 packages finished [8.45s]

colcon test
Summary: 7 packages finished [2.27s]

colcon test-result --verbose
Summary: 37 tests, 0 errors, 0 failures, 0 skipped
```

The offline launch was repeated from that isolated clean build and returned the
same successful 81/6/6 path result. Static scans found no ROS, Isaac, PX4, or
MAVLink import in the seven pure modules, no direct PX4 publisher, and no
tracked generated/build/cache/model artifact. The legacy workspace source hash
before and after Phase 2 was identical:

```text
9bb394c0e4e5616f0857ce61e5067971a5931ae5c32a0e33ef3d96af40b94beb
```

## Phase 4 deterministic trajectory parameterization

Final verification on 2026-08-06 used ROS 2 Jazzy and `/usr/bin/python3`.
`./uav build`, `./uav test`, and the independent `./uav verify` rebuild/test
cycle all passed for seven packages. The suite reported 77 tests, 0 errors,
0 failures, and 0 skipped. Coverage includes pure configuration, velocity/time
profiles, acceleration/lateral/jerk constraints, global scaling, NED yaw and
unwrap behavior, exact geometry/altitude preservation, structured rejection,
and a live ROS graph candidate/validity/status contract test.

The standalone harness passed all twelve deterministic fixtures. Accepted
fixtures were straight-line, Phase 3 B-spline-shaped path, sharp bend, high
curvature, adjacent duplicate cleanup, two-point, yaw wrap, and jerk scaling.
The one-point, non-finite, impossible scaling-budget, and wrong-frame fixtures
were rejected as expected. Every fixture also republished an identical input
and observed zero duplicate recomputations.

Final command results:

```text
./uav offline-check enable_bspline:=true
  source=BSPLINE, raw=81, simplified=6, candidate=55, final=55: PASS
./uav offline-check enable_bspline:=false
  source=ASTAR_SIMPLIFIED, raw=81, simplified=6, final=6: PASS
./uav trajectory-check
  straight-line valid=true, points=3, duration=6.000000 s: PASS
./uav pipeline-check
  scene -> A* -> accepted B-spline -> 55-point timed trajectory: PASS
./uav trajectory-check fixture:=impossible-config-rejection
  valid=false, maximum jerk diagnostic at point 0: PASS
./uav topics
  only /parameter_events and /rosout; no /fmu/in/*: PASS
```

The global-scaling examples measured scale 2.765436 for high curvature and
5.572329 for the strict jerk/yaw fixture. The accepted Phase 3 pipeline path
measured 8.066690 s duration, 2.151213 scale, 0.852011 m/s maximum speed,
2.991018 m/s³ maximum jerk, and 0.492743 rad/s maximum yaw rate. These are
offline feasibility measurements, not flight, controller, PX4, tracking, or
disturbance validation.

## Phase 3 validated B-spline candidate

Date: 2026-08-06 (Asia/Taipei)

The pure implementation uses system Python 3.12.3 and adds no NumPy, SciPy, or
other runtime dependency. Package and full-workspace results after the Phase 3
changes were:

```text
./uav build --packages-select uav_navigation
Summary: 1 package finished

./uav test --packages-select uav_navigation
Summary: 63 tests, 0 errors, 0 failures, 0 skipped

./uav verify
Summary: 7 packages finished (build)
Summary: 7 packages finished (test)
Summary: 63 tests, 0 errors, 0 failures, 0 skipped
VERIFY SUCCESS: build, tests, interfaces, imports, and safety checks passed
```

Coverage adds clamped basis/De Boor evaluation, effective-degree reduction,
exact endpoints, duplicate removal, arc-length resampling, sample bounds,
continuous clearance and bounds, straight/circular curvature, self-intersection,
planner selection/fallback, determinism, six-scene comparison, and ROS fixture
contracts. Existing Phase 2 regressions remain in the same suite.

Accepted ROS fixture:

```text
./uav offline-check enable_bspline:=true \
  fixture:=bspline-safe-single-obstacle
SUCCESS|astar_success=true|bspline_valid=true|bspline_selected=true
|final_source=BSPLINE|raw_points=81|simplified_points=6
|candidate_points=55|final_points=55|length_m=4.243945
|minimum_physical_clearance_m=0.419106
|maximum_curvature_inverse_m=0.749852
OFFLINE CHECK SUCCESS
```

Rejected ROS fixture, with overall success because the validated A* baseline
remained available:

```text
./uav offline-check enable_bspline:=true \
  fixture:=bspline-rejected-corner-cut
SUCCESS|astar_success=true|bspline_valid=false|bspline_selected=false
|final_source=ASTAR_FALLBACK|raw_points=81|simplified_points=6
|candidate_points=64|final_points=6
|bspline_rejection=segment 19 intersects obstacle offline_tower
OFFLINE CHECK SUCCESS
```

The disabled fixture returned `ASTAR_SIMPLIFIED`. Strict-clearance and
strict-curvature fixtures returned `ASTAR_FALLBACK`. Two-point, three-point,
duplicate-control-point, and self-intersection-preflight fixtures all completed
successfully. Every launch reported `SAFE: no /fmu/in/* topics detected`.

The installed `geometric_path_comparison` entry point completed six fixed
scenes: four candidates accepted as `BSPLINE`, while strict clearance and
curvature gates rejected the candidate and selected `ASTAR_FALLBACK`. Full
metrics and limitations are recorded in `phase3_geometric_comparison.md`.

The canonical legacy hash (pruning nested `.git` directories) remained exactly
the same before and after Phase 3, and that workspace was never sourced or
built:

```text
9bb394c0e4e5616f0857ce61e5067971a5931ae5c32a0e33ef3d96af40b94beb
```

## Phase 8 offline PX4 stream boundary

The first package-level Phase 8 validation retained the normal seven-package,
PX4-independent build. `uav_interfaces` and `uav_px4_control` reported 252
tests, 0 errors, 0 failures, and 0 skipped after adding 35 stream tests to the
217-test baseline. `./uav px4-stream-offline-check` independently reported all
35 adapter/state/ownership tests and all 20 required fixtures passing, followed
by `SAFE: no /fmu/in/* topics detected`.

No PX4, XRCE, Isaac Sim, OFFBOARD, arming, or live publisher was started for
that result. The required five-run pre-PX4 mux gate subsequently passed 5/5.

The live PX4 v1.14.3 SIH boundary then passed the zero fixture and all four
0.10 mapping fixtures. The final timing-instrumented zero run received 41
matched `TrajectorySetpoint`/`OffboardControlMode` pairs at 20.000656 Hz, with
0.047295 s minimum interval, 0.052220 s maximum interval, 0.001016 s RMS
jitter, strict timestamp monotonicity, and no dropped cycle. Gate false, stale
upstream candidate, and XRCE Agent loss each stopped publication, latched, and
required explicit repair/reset/re-enable. PX4 stayed disarmed and outside
OFFBOARD, and no `VehicleCommand` publisher existed. Detailed pass and failed
attempt evidence is retained in `phase8_sitl_test_results.md`.
