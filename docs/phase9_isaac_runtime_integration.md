# Phase 9 Isaac Sim runtime integration

## Scope and result

Phase 9 starts from the verified flight milestone commit
`2620fe58a8a1469c3d5fd9ec3a9139d82a7c06fa` on the new branch
`feature/isaac-runtime-integration`. The milestone branch itself was not
modified.

The accepted run on 2026-08-15 connected the existing A*, B-spline, timed
trajectory, follower, mux, output gate, and Phase 8 PX4 streamer to a Pegasus
Iris in Isaac Sim. The actual Isaac stage UAV took off, followed the ROS 2
trajectory, entered the goal tolerance, landed under `NAV_LAND`, and disarmed.
The evidence monitor returned exit code zero and `success=true`.

This phase does not add or modify BC, PPO, autoencoders, training, object
detection, cameras, dataset collection, reward logic, or learned-policy
runtime.

## Audit and reused components

The implementation deliberately reuses these repository components:

- `7.isaac_uav_bootstrap.py` for Pegasus environment, Iris construction,
  PX4 MAVLink backend, timeline startup, and PX4 auto-launch;
- the existing Isaac stage transform technique used by the legacy episode
  manager, reduced to a pose-only runtime adapter;
- `uav_scene_bridge` as the simulator access boundary;
- the typed `ObstacleArray` and `PoseStamped` planner scene contract;
- the existing A* planner, B-spline smoother, trajectory parameterizer,
  follower, `ASTAR_EXPERT` mux, PX4 output gate, flight supervisor, and
  single-owner Phase 8 setpoint streamer;
- the established Isaac-world-to-PX4 convention: Isaac `(x, y, z)` maps to
  planner `(east, north, up)` and PX4 local `(north, east, down)` at the
  supervisor boundary.

The missing interface was a narrow, current-state runtime boundary. The
legacy episode manager combined reset, camera, recording, and dataset
lifecycle and was therefore outside this phase. Phase 9 adds only:

1. an in-Isaac publisher of the actual UAV pose and a monotonic runtime/scene
   heartbeat, using standard ROS 2 messages;
2. an external bridge that validates that contract and republishes one
   coherent typed scene snapshot;
3. opt-in supervisor and monitor support for external scene freshness and
   actual Isaac motion evidence.

All Phase 9 behavior is opt-in. The Phase 8 defaults remain
`use_external_scene=false` and `require_isaac_evidence=false`.

## Runtime architecture

```text
Isaac stage /World/quadrotor/body
  -> /isaac_uav/pose + /uav/isaac/runtime_status
  -> fail-closed scene_bridge
  -> typed Isaac start / goal / obstacle snapshot
  -> existing A* -> B-spline -> timed trajectory -> follower
  -> ASTAR_EXPERT mux -> output gate -> Phase 8 PX4 streamer
  -> PX4 OFFBOARD / Pegasus dynamics -> Isaac stage UAV
```

The deterministic validation scene is `phase9_fixed_scene_v1`, revision 1.
It contains a collidable obstacle at Isaac `(-1.5, 1.5, 1.25)` and a goal at
Isaac `(0.5, 3.0, 1.5)`. The obstacle is intentionally off the direct flight
corridor, so this milestone validates scene transport without turning Phase 9
into an obstacle-avoidance tuning exercise.

The scene bridge starts disabled and becomes ready only when the pose and
runtime heartbeat are valid and no more than 0.5 seconds old. Scene messages
share one stamp and use reliable transient-local QoS. Mission enable is
rejected until this coherent external scene is ready. During a mission, stale
or invalid runtime evidence is passed to the existing state machine as an
environment fault.

Pegasus launches PX4 with a different command shape from the Phase 8 SIH
fixture. The streamer now recognizes only the audited instance-zero Pegasus
shape: the expected PX4 binary, the absolute PX4 `rcS`, `-i 0 -d`, and
`PX4_SIM_MODEL=gazebo-classic_iris`. Other local processes still fail the
identity gate.

## Reproduction commands

The accepted environment used Isaac Sim `5.1.0-rc.19+main.0.aa503a9b.local`
and PX4 commit `1dacb4cdef2d7145754fc788fa8dc482eed74b40`.

Build once from the repository:

```bash
cd /home/noel_614420090/uav-project
./uav build
```

Start the XRCE Agent in terminal 1:

```bash
MicroXRCEAgent udp4 -p 8888
```

Start Isaac Sim, the deterministic scene, Pegasus Iris, PX4, and the runtime
adapter in terminal 2:

```bash
cd /home/noel_614420090/isaacsim/_build/linux-x86_64/release
env -u DISPLAY -u WAYLAND_DISPLAY \
  ./isaac-sim.streaming.sh --livestream 2 \
  --exec /home/noel_614420090/uav-project/ros2_isaac_scripts/7.isaac_uav_bootstrap.py
```

After the Isaac bootstrap reports that the runtime bridge has started, run the
guarded acceptance in terminal 3:

```bash
cd /home/noel_614420090/uav-project
UAV_OFFLINE_TIMEOUT_SECONDS=180 ./uav isaac-runtime-flight-check
```

This command includes a five-second discovery interval, requires Isaac
evidence, and intentionally requests OFFBOARD, arm, flight, and landing. It
must never be used against a real vehicle. Stop terminal 2 and then terminal 1
with `Ctrl-C` after the mission has reported `COMPLETE`; the accepted shutdown
left no Isaac, PX4, or XRCE process running.

## Accepted runtime evidence

The machine-readable local evidence is
`run_logs/px4-sitl-flight_20260815T114011Z.json`. Runtime logs remain
gitignored; the relevant observations are preserved here.

| Elapsed (s) | Independently observed evidence |
|---:|---|
| 0.152 | Isaac runtime and coherent scene ready |
| 1.899 | Isaac pose rate evidence established |
| 2.072 | A* path, B-spline, and timed trajectory valid |
| 2.123 | mux selected `ASTAR_EXPERT` |
| 2.223 | PX4 output gate safe |
| 4.323 | setpoint stream reached 20 Hz |
| 4.344 | PX4 confirmed OFFBOARD |
| 4.380 | PX4 confirmed armed |
| 11.785 | actual Isaac UAV exceeded the 1.25 m takeoff threshold |
| 14.486 | actual Isaac UAV moved 0.5 m during mission tracking |
| 16.587 | actual Isaac UAV entered the 0.40 m goal tolerance |
| 18.523 | follower reported goal hold and landing was commanded |
| 22.567 | PX4 and actual Isaac pose independently confirmed landing |
| 24.760 | PX4 disarmed and mission state became `COMPLETE` |

Additional measurements:

- 396 Isaac pose samples; maximum observed pose rate 19.747 Hz;
- actual Isaac maximum altitude 1.562 m;
- actual Isaac tracking displacement 2.998 m;
- minimum actual Isaac-to-goal distance 0.114 m;
- final Isaac pose `(0.531, 3.171, 0.062)`, back at ground height;
- PX4 setpoint maximum observed rate 20.005 Hz, maximum gap 0.0503 s,
  zero dropped cycles, and no stream fault;
- PX4 maximum altitude 1.549 m and minimum goal distance 0.024 m;
- exactly the expected mode, arm, and land commands were accepted;
- final PX4 state: landed, disarmed, failsafe false; streamer disabled.

## Fail-closed fault evidence

The bridge process was terminated by exact PID while the second validation
run was in `TRACKING`. Local evidence is
`run_logs/px4-sitl-flight_20260815T114931Z.json`.

- `TRACKING` was observed at 15.328 s.
- The stale 0.5-second runtime contract caused `LANDING` at 16.327 s.
- The supervisor stopped tracking, disabled the output gate, disabled the
  streamer, and sent `NAV_LAND`; command 21 was accepted.
- PX4 and Isaac landing were independently observed at 20.208 s.
- The terminal state was intentionally `FAILED` at 22.327 s, with the vehicle
  landed and disarmed, PX4 failsafe false, and the stream at 0 Hz.

This proves the fault is not reported as mission success and does not keep
publishing stale setpoints. No timeout, freshness threshold, safety latch, or
failure gate was weakened to produce the successful run.

## Tests

Commands executed after all runtime processes were stopped:

```bash
./uav test
pytest -q tests/test_isaac_runtime_contract.py
./uav verify
git diff --check
bash -n uav
python3 -m compileall -q \
  ros2_ws/src/uav_px4_control/uav_px4_control \
  ros2_ws/src/uav_scene_bridge/uav_scene_bridge
```

Results:

- `./uav test`: 7 packages, 281 tests, 0 errors, 0 failures, 0 skipped;
- root Isaac contract tests: 4 passed;
- `./uav verify`: build, the same 281 tests, interfaces, import boundaries,
  generated-file checks, and `/fmu/in/*` safety checks all passed;
- focused `uav_px4_control`: 173 tests passed;
- focused `uav_scene_bridge`: 11 tests passed;
- shell syntax, Python byte-compilation, and diff whitespace checks passed.

The ament flake8 runs emitted the existing Python `fork()` warning in the
navigation and PX4 packages; they did not produce test failures.

## Changed areas

- Isaac bootstrap and new pose/runtime adapter;
- scene bridge runtime schema, validation, typed publications, tests, and
  dependencies;
- opt-in external-scene launch/configuration and supervisor inputs;
- Isaac-aware runtime acceptance monitor;
- exact Pegasus PX4 process identity and tests;
- state-machine environment-fault landing behavior and regression tests;
- the `isaac-runtime-flight-check` wrapper command;
- this report.

## Remaining risks

- This is one deterministic open scene and one Pegasus Iris; it does not yet
  establish performance across arbitrary USD scenes, vehicles, or clutter.
- The Isaac update loop produced 17--20 Hz under the measured headless load.
  Acceptance requires at least 15 Hz for actual-pose evidence, while the
  safety-critical PX4 setpoint stream remains at 20 Hz.
- A five-second startup delay was required for DDS discovery on this host.
  Slower machines may need an explicit readiness-driven launcher in a future
  phase; flight enable still fails closed if discovery is incomplete.
- One earlier exploratory run latched a prestream scheduling fault and did not
  arm. A clean retry passed with zero stream faults; the latch was retained.
- The external PX4 checkout contained pre-existing local changes and runtime
  artifacts. Phase 9 did not modify or commit that repository.
- This milestone is simulation evidence only. It does not establish real
  airframe, sensor, timing, networking, or hardware safety.
