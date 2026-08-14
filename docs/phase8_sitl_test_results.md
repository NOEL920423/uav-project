# Phase 8 PX4 SITL test results

## Classification and environment

Phase 8 is a SITL-only ROS 2 setpoint-boundary verification. It did not request
OFFBOARD, arm, disarm, take off, land, start Isaac Sim/Pegasus, target a real
vehicle, or perform flight testing. The vehicle remained `DISARMED` and
OFFBOARD remained inactive for every live run.

The audited versions were PX4 v1.14.3-dirty at
`1dacb4cdef2d7145754fc788fa8dc482eed74b40` and clean `px4_msgs` v1.14.0 at
`ffb6e80e1c17e5714395611a020c282a87af8fa4`. Phase 8 did not modify either
external checkout. Live processes were started separately in `tmux astar`:

```bash
MicroXRCEAgent udp4 -p 8888
ninja -C /home/noel_614420090/PX4-Autopilot/build/px4_sitl_default sihsim_quadx
```

The local SIH target is PX4's built-in SITL simulation. Isaac Sim and Gazebo
were not started.

## Offline prerequisites

The complete offline gate passed before PX4 was started: doctor, build, test,
verify, mux, mux safety, control stack, PX4 map, gate, boundary, and stream
offline checks. The normal workspace contained seven packages and 252 tests,
with 0 errors, 0 failures, and 0 skipped. The Phase 8 subset added 35 tests to
the 217-test baseline, and all 20 deterministic stream fixtures passed.

The pre-PX4 mux stability loop passed 5/5. The first boundary run transiently
timed out at `WAIT_RECOVERED_SAFE`; the immediate diagnostic showed a startup
synchronization race, not a safety-limit violation, and the following run
passed. The result is retained here rather than being omitted.

## Doctor and graph evidence

The live read-only doctor verified:

- PX4 SITL and `MicroXRCEAgent udp4 -p 8888` processes present;
- one publisher on each of the four required telemetry topics;
- one subscriber on each allowed PX4 input;
- compatible `BEST_EFFORT` endpoint QoS;
- no `/fmu/in/vehicle_command` publisher;
- vehicle disarmed, OFFBOARD inactive, and no critical failsafe.

`offboard_control_signal_lost=true` was observed before streaming. PX4 v1.14.3
uses that as expected prestream evidence, so it is reported separately and is
not treated as a global failsafe while OFFBOARD is inactive. Unexpected
OFFBOARD activation remains an independent latched stop condition.

## Live mapping and timing

The zero fixture ran first. Each later fixture was a separate run and never
exceeded the authorized diagnostic magnitude of 0.10.

| Fixture | Trajectory/mode count | Rate | Maximum gap | Result |
|---|---:|---:|---:|---|
| zero, initial successful run | 42 / 42 | 20.016719 Hz | 0.051988 s | pass |
| north +0.10 m/s | 41 / 41 | 19.998463 Hz | 0.051954 s | pass |
| east +0.10 m/s | 42 / 42 | 19.997511 Hz | 0.052121 s | pass |
| down +0.10 m/s | 42 / 42 | 20.005118 Hz | 0.051992 s | pass |
| yaw-rate +0.10 rad/s | 43 / 43 | 20.010861 Hz | 0.051850 s | pass |
| zero, final timing-instrumented run | 41 / 41 | 20.000656 Hz | 0.052220 s | pass |

The final zero run measured minimum interval 0.047295 s and RMS jitter
0.001016 s. It used strictly increasing timestamps from `1786355203010268` to
`1786355205010320`. Every fixture verified NED identity mapping, NaN unused
position/acceleration/jerk/yaw fields, yaw-rate-only `yawspeed`, and
velocity-only `OffboardControlMode`. Each began disabled, required Phase 7 and
Phase 8 enables, reached streaming, explicitly disabled, and proved publication
stopped.

## Live fail-closed and recovery evidence

- **Gate false:** a healthy zero stream reached count 328. Disabling the Phase
  7 gate latched `LATCHED_STREAM_FAULT`; two later observations stayed at 328,
  and a direct re-enable request was rejected. Explicit stream reset, stable
  Phase 7 readiness, gate enable, and stream enable restored publication.
- **Stale upstream candidate:** pausing only the synthetic A* candidate made
  the mux select `HOLD` and report invalid selection, Phase 7 latched, and Phase
  8 stopped at count 646. Resuming the publisher alone did not recover output;
  explicit mux selection and both gate resets were required.
- **DDS loss:** interrupting the XRCE Agent set `dds_ready=false`, latched the
  streamer, and stopped at count 903. Restarting the Agent restored graph and
  telemetry but a direct stream enable was rejected. Explicit reset/recovery
  restored streaming and increased the count to 991.
- **Telemetry stale:** with the initial live 0.50 s telemetry setting, measured
  1.7--1.9 Hz PX4 status/flags traffic crossed the limit and correctly stopped
  after 11 pairs. The live-only timeout was then evidence-adjusted to 0.75 s;
  all offline stale-telemetry fixtures still fail closed.
- **Failsafe, unexpected armed/OFFBOARD, time regression, invalid mapping, and
  publish gap:** deterministic offline fixtures select the corresponding
  `STOPPED_*` state, publish no further pair, latch the original reason, and
  require explicit recovery. These unsafe vehicle states were not injected
  into live PX4.

## Failed attempts and residual observations

The live campaign retained the following diagnostics:

- `none_iris` waited for an external simulator on TCP 4560, so it was stopped
  without streaming. The exact locally audited `sihsim_quadx` target was used.
- One mistyped launch argument was rejected by `ros2 launch` before nodes
  started or any publisher existed.
- The first zero attempts exposed the expected prestream
  `offboard_control_signal_lost` circular gate, a Phase 7 startup readiness
  race, and the too-short 0.50 s live telemetry timeout. The fixes separate
  critical failsafe from prestream signal loss, require three consecutive
  `READY_DISABLED` heartbeats, and use the measured 0.75 s live-only timeout.
- After XRCE restart, one doctor invocation safely failed on a transient zero
  trajectory subscriber count; detailed graph inspection then showed both
  subscribers. Doctor readiness now requires three consecutive complete graph
  snapshots.
- Two isolated SIGINT teardowns produced an `rclpy`/`px4_msgs` conversion
  exception after the monitor had already completed successfully; later and
  final teardowns were clean. This remains a non-flight shutdown-race risk.
- PX4 SIH reported intermittent preflight estimator warnings while disarmed.
  No mode or arm request was sent and no flight occurred.

The final live graph was stopped cleanly. Generated PX4 logs and ROS logs were
not added to the project repository.
