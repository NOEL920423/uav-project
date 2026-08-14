# ROS 2 PX4 SITL flight milestone

## Scope and result

This milestone adds only the orchestration and evidence boundaries required to
fly the existing ROS 2 A*/B-spline stack in PX4 SITL. It does not add or change
BC, PPO, the autoencoder, learned-policy runtime, Isaac bridges, camera
bridges, or episode management.

The accepted run completed on 2026-08-14 with PX4 SIH and the XRCE Agent. The
machine-observed result is retained locally at
`run_logs/px4-sitl-flight_20260814T121231Z.json` (runtime logs are deliberately
gitignored). The monitor returned exit code zero and `success=true`; final PX4
evidence was disarmed, landed, and failsafe false.

## Reused Phase 8 baseline

The milestone merged the validated Phase 8 history without rewriting it:

- `855d33b` — single-owner PX4 setpoint streamer
- `9379ebd` — fail-closed SITL streaming gate
- `bb224d5` — PX4 streaming and DDS regression checks
- `6ccf97a` — Phase 8 SITL boundary documentation

The Phase 8 branch and the milestone's source branch share base `036791d`.
The Phase 8 streamer remains the sole owner of trajectory setpoint and
offboard-control-mode publications. The flight supervisor is the sole owner of
`/fmu/in/vehicle_command`.

## Flight-only additions

`px4_sitl_flight_supervisor_node` starts disabled and sequences a finite
mission only after `SetPx4FlightEnable` is explicitly called. It consumes
planner, trajectory, follower, mux, gate, streamer, command-ACK, vehicle
status, odometry, and land-detector evidence. The 1.5 m flight altitude is a
milestone configuration shared by the planner and supervisor.

The existing follower gains an explicit start/stop service so it can publish a
legal exact-zero prestart command for prestream, then begin the accepted
trajectory epoch only after PX4 is OFFBOARD and armed. Repeated start requests
are idempotent. A validated NED odometry bridge supplies live PX4 state to the
unchanged tracking controller.

The existing output gate still defaults to locking the vehicle-state
signature. The flight-only YAML disables that signature lock because OFFBOARD
and arming are intentional state transitions; every other validity,
freshness, source, telemetry, failsafe, latch, and explicit recovery gate
remains active. In-flight source, telemetry, gate, streamer, DDS, stale, or
invalid failures transition toward controlled landing.

After goal evidence, follower, setpoint stream, and output gate are explicitly
stopped before one accepted `NAV_LAND` command hands control to PX4 AUTO_LAND.
Completion still requires PX4 `landed=true` and disarmed state; altitude or a
timer cannot claim landing.

## Commands

Start the local XRCE Agent and PX4 SIH in separate terminals:

```bash
MicroXRCEAgent udp4 -p 8888
ninja -C /home/noel_614420090/PX4-Autopilot/build/px4_sitl_default \
  sihsim_quadx
```

Then run the guarded acceptance command:

```bash
UAV_OFFLINE_TIMEOUT_SECONDS=150 ./uav px4-sitl-flight-check
```

The wrapper first runs the read-only SITL doctor and refuses to proceed unless
PX4 is local SITL, XRCE/DDS is ready, the vehicle is disarmed and outside
OFFBOARD, no failsafe exists, and there is no competing VehicleCommand owner.
This command intentionally requests OFFBOARD, arm, flight, and landing; it is
not a real-vehicle command.

Offline regression commands used for this milestone:

```bash
./uav verify
./uav px4-stream-offline-check
./uav px4-gate-check
./uav mux-safety-check
./uav ml-test
```

## Accepted flight timeline

Times are seconds after the monitor started:

| Time | Evidence |
|---:|---|
| 2.02 | Explicit mission enabled; takeoff scene published |
| 2.07 | A* path, B-spline candidate, and timed trajectory valid |
| 2.12 | Mux selected `ASTAR_EXPERT` |
| 2.22 | PX4 output gate safe |
| 4.37 | Stable setpoint stream observed at 20.000 Hz |
| 4.38 | PX4 confirmed OFFBOARD |
| 4.42 | PX4 confirmed armed; commands 176 and 400 accepted |
| 4.62 | Trajectory follower in tracking; takeoff in progress |
| 9.92 | Configured takeoff threshold reached; mission replan began |
| 10.12 | Mission trajectory tracking active |
| 21.47 | Goal reached at 0.167 m; `NAV_LAND` accepted |
| 21.52 | PX4 AUTO_LAND active; setpoint stream stopped |
| 31.22 | PX4 reported landed and disarmed; state `COMPLETE` |

Peak observed stream rate was 20.006 Hz, maximum measured altitude above the
enable-time local datum was 2.066 m, and minimum goal distance was 0.152 m
against the 0.35 m tolerance. Exactly three vehicle commands were published:
OFFBOARD mode, arm, and land. No PX4 failsafe was observed.

## Regression and safety evidence

- `./uav verify`: 7 packages; 266 tests, zero errors/failures/skips.
- Phase 8 stream tests: 35/35 tests and 20/20 fixtures passed.
- PX4 gate: 23/23 pure tests plus safe/fault/latch/recovery launch passed.
- Mux stale-source latch and explicit recovery launch passed.
- ML regression: 14/14 passed; no ML source or artifact was changed here.

The successful flight proves the requested SITL architecture and evidence
chain. It does not prove real-airframe dynamics, real sensors, hardware timing,
or generalize beyond the tested local PX4 SIH/XRCE configuration.
