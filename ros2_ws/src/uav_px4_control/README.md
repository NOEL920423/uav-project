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

## Physical USB joystick input

`control/joystick_control_node.py` reads one Linux `/dev/input/js*` device
directly through the nonblocking joydev API. It adds one node, reuses
`/uav/control/joystick_command` and `/uav/vehicle/odometry`, and needs no `/joy`
topic, extra driver node, or Python input dependency. The joydev event format
is documented by the
[Linux kernel](https://www.kernel.org/doc/html/latest/input/joydev/joystick-api.html).
Keep this hardware input separate from mux arbitration and synthetic fixtures.

For a RadioMaster Pocket, connect a data-capable cable to its top USB port,
power on the radio, select USB Joystick, and turn off RF in a simulator model.
The bottom USB port is for charging. USB axis order follows the radio model;
the defaults match the measured Pocket channels in this project: A1 forward,
A0 right, A3 yaw, and A4 (SA) deadman. Stick up/right is positive.

Build only the changed package, then inspect the actual USB channels:

```bash
./uav build --packages-select uav_px4_control
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
export PATH=/usr/bin:/bin:$PATH
ls -l /dev/input/by-id/*joystick /dev/input/js*
ros2 run uav_px4_control joystick_control_node --ros-args \
  -p device:=/dev/input/js0
```

Inspection is the default (`enable_control=false`): it shows an initial axis
table, then reports only changed axes/buttons at most once per second. Values
use two decimal places; unchanged inputs and inactive buttons stay quiet.
`A2=+0.52` means raw axis 2 is at +0.52; `B0=pressed` means button 0 is active.
It publishes no candidates. Move one stick or switch at a time to identify
its index and sign. Prefer a stable `by-id/*-joystick` device
path when available. If no `js*` exists, check USB Joystick mode, the data
cable, and the Linux `joydev` driver. Device open/read errors retain the
original exception. Grant access to the identified joystick through the
host's input-device permissions; do not run the ROS stack as root.

After verifying the mapping, enable candidate input explicitly:

```bash
ros2 run uav_px4_control joystick_control_node --ros-args \
  -p device:=/dev/input/js0 -p enable_control:=true \
  -p axis_indices:='[1, 0, 3]' \
  -p axis_signs:='[1.0, 1.0, 1.0]' \
  -p deadman_axis:=4
```

`axis_indices`, `axis_signs`, `axis_centers`, and `axis_scales` are three-entry
arrays ordered `[forward, right, yaw]`. Centers use raw normalized joystick
coordinates; scales multiply the centered input. Default deadzone is 0.05.
Movement limits reuse the BC contract: 1.0 m/s forward, 0.8 m/s right, and
1.0 rad/s yaw. The node rotates body velocities using fresh `px4_ned` odometry
and publishes at 50 Hz, including when sticks remain unchanged. Vertical
velocity defaults to zero. The human flight launch enables
`altitude_hold_m=1.5`: bounded proportional NED vertical feedback, with
the ground datum captured from the first valid odometry after node startup.
Start the node on the ground; restarting it in flight changes that datum.
A2 (left vertical stick) is unused in this first fixed-altitude mode.

The deadman defaults to raw axis 4 greater than 0.5. Configure the radio model
to expose a switch as that axis, or use `deadman_axis:=-1` and
`deadman_button:=<observed index>` for a USB button. These are mutually
exclusive. At startup and after a fault, release the deadman with centered
sticks, then activate it while sticks are still centered. With the deadman
released and sticks centered, the log must show `Control: ready`; after
activating, it shows
`publishing HUMAN_JOYSTICK candidates`. Parameters are read at startup;
restart the node to apply changes.

On deadman release, invalid/stale odometry, USB resynchronization, or a publish
loop gap over 0.20 s, an active node sends one zero command then stops
heartbeats. The existing mux selects exact-zero latched HOLD once its joystick
timeout expires (default 0.20 s after the last candidate). USB removal closes
the device and requires a node restart; it never silently reconnects or
switches devices. Stationary sticks are not a disconnect: joydev reports
changes only, and an empty nonblocking event queue is normal.

### Human flight in Isaac Sim

The existing `px4_sitl_flight.launch.py` now supports `human_control:=true`.
It starts the physical joystick node and the existing PX4 safety chain, and
omits the A* planner, parameterizer, follower, and automatic mission monitor.
The supervisor starts disabled. It owns takeoff and landing requests through
the existing VehicleCommand owner; the joystick still publishes candidates
only. No additional topic or production Python module is needed.

Human mode displays a `FLIGHT CONTROL` panel when its status changes: USB
connection, PX4 pose readiness, Isaac readiness, SA position, flight stage,
and one `NEXT` action. `PILOT CONTROL` means the supervisor handed control
to the sticks; a connected USB alone does not mean the vehicle can fly.
The panel observes the existing flight status topic and does not arm or
select a source. Starting with SA at +1 deliberately blocks input until
the operator first sets SA to -1 with centered sticks, then back to +1.
Missing telemetry or simulator evidence is displayed as waiting.
Periodic `PX4_TELEMETRY_DIAG` and `BC_STARTUP_DIAG` dictionaries and raw
axis updates are disabled only in this mode; fault summaries, warnings,
errors and raw tracebacks remain visible. Full telemetry and transition
diagnostics are still recorded by the existing timing recorder when
`UAV_TIMING_DIR` is supplied.
Standalone joystick inspection and the default A* launch retain their
diagnostic output.

After the Windows SSH tunnel and Linux USB/IP attach are active, stop any
standalone joystick inspection node. Use three Linux terminals:

1. Start the existing Isaac/PX4/XRCE runtime and keep its terminal open:

   ```bash
   cd ~/uav-project
   ./uav expert-scene-preview --seed 0
   ```

   Connect through the usual WebRTC viewer and wait for scene readiness.
   Despite the preview command's name, it supplies the runtime; terminal 2
   supplies the separate human flight graph. Skip this command if the same
   bootstrap runtime and XRCE agent are already running.

2. Start human control while the vehicle is on the ground:

   ```bash
   rosuav
   ros2 launch uav_px4_control px4_sitl_flight.launch.py \
     human_control:=true use_external_scene:=true
   ```

   `rosuav` is the local shell function in `~/.bashrc`: it selects system
   Python, sources Jazzy and `install/local_setup.bash`, and enters `ros2_ws`.
   A new shell without that function can source the underlay and workspace
   explicitly instead. Keep both sticks centered, set SA to its -1 position
   until `NEXT` asks you to move SA to +1, then switch SA to +1. Wait for
   `NEXT: READY: send the takeoff request in terminal 3`.

3. Explicitly request takeoff; keep the sticks centered during startup:

   ```bash
   rosuav
   ros2 service call /uav/px4/set_flight_enable \
     uav_interfaces/srv/SetPx4FlightEnable '{enable: true}'
   ```

   A rejected request explains which readiness condition is missing and
   does not arm the vehicle. Once accepted, the supervisor selects
   `FLIGHT_LIFECYCLE`, prestreams, requests OFFBOARD/arm, climbs toward 1.5 m,
   then selects `HUMAN_JOYSTICK` through the mux's existing zero barrier.
   Wait for `Flight: PILOT CONTROL` before moving
   the sticks. Right stick: forward/back and sideways relative to vehicle
   heading. Left horizontal stick: yaw. Centered sticks request zero horizontal
   speed while vertical feedback maintains the altitude target.

Switch SA back to -1 to end input and request landing. A detected USB read
failure, stale human candidate, stale telemetry, or a stopped safety chain
also requests landing and ends the flight; switching SA back on does not
resume an airborne flight. Keep the runtime and flight launch alive until
PX4 has landed and disarmed. An explicit landing request is also available:

```bash
ros2 service call /uav/px4/set_flight_enable \
  uav_interfaces/srv/SetPx4FlightEnable '{enable: false}'
```

The human launch enables `latch_after_stop=true`: even a brief SA release
ends input until the flight launch is restarted. After landing/disarming,
restart terminal 2 and repeat the neutral-SA/takeoff sequence for another
flight; the Isaac runtime can stay open.

The 0.20 s candidate watchdog detects missing ROS input publications. It
cannot guarantee a 0.20 s response to a broken SSH/USB-IP transport: joydev
keeps unchanged input values until the kernel reports a device error.
Transport timeout and recovery still need verification on the actual setup.
The node reads the device independently of terminal or Isaac viewer focus.

Human dataset recording remains pending. The existing recorder reads A*
commands and rejects human samples, so the launch explicitly rejects
`human_control:=true record_expert_dataset:=true`. Do not start another
physical or synthetic joystick publisher alongside this launch. The new
mode has targeted automated coverage; an actual Isaac flight is still
required to confirm PX4/physics behavior and altitude tracking.

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
