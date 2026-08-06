# Phase 6 offline control-source multiplexer design

## Scope and safety boundary

Phase 6 adds a deterministic ROS 2 control-source multiplexer and safety
arbitration contract. It is offline, non-flight software. It does not start or
control PX4, Isaac Sim, Pegasus, Micro XRCE-DDS, a joystick device, or a NavRL
model. It does not implement OFFBOARD, arming, takeoff, landing, or publish any
`/fmu/in/*` topic.

The command layers are deliberately distinct:

```text
source candidate -> mux-selected command -> future PX4 output
```

Phase 6 implements only the first two layers. A
`/uav/control/selected_command` is still a ROS-level offline candidate; it is
not a PX4 setpoint and is not evidence of vehicle safety or flight readiness.

## ROS graph and ownership

The mux consumes reliable, volatile `geometry_msgs/msg/TwistStamped`
candidates:

- `/uav/control/astar_command` — `ASTAR_EXPERT`
- `/uav/control/joystick_command` — `HUMAN_JOYSTICK`
- `/uav/control/navrl_command` — `NAVRL_POLICY`
- `/uav/control/hold_command` — optional external `HOLD` diagnostic input

It exclusively publishes:

- `/uav/control/selected_command` — selected or internal zero command
- `/uav/control/source` — active canonical source identifier
- `/uav/control/mux_status` — structured arbitration and freshness status

Selection uses `/uav/control/set_source` with
`uav_interfaces/srv/SetControlSource`. The exact identifiers are `HOLD`,
`ASTAR_EXPERT`, `HUMAN_JOYSTICK`, and `NAVRL_POLICY`; aliases and unknown
values are rejected. Startup and every fail-closed outcome select internal
`HOLD`. Only `control_mux_node` may publish the selected-command topic.

All command frames are exactly `px4_ned`. `linear.x/y/z` mean north/east/down
velocity in m/s, `angular.z` means NED yaw rate in rad/s, and `angular.x/y`
must be zero. The mux performs no coordinate transformation.

## Pure computation boundary

The ROS-independent modules live in `uav_px4_control`:

- `control_source_models.py` defines immutable configuration, commands,
  records, states, diagnostics, responses, and cycle results.
- `control_source_registry.py` owns candidate receipt history, source-specific
  timeouts, static candidate validation, and health classification.
- `control_mux.py` owns deterministic selection, dwell, barriers, latching,
  rate limiting, and fail-closed transitions.
- `selected_command_validator.py` independently validates the final selected
  command and the internal fallback HOLD.

These modules import no ROS, PX4, Isaac, Pegasus, XRCE, joystick, camera,
recorder, or ML runtime. `control_mux_node.py` is the only ROS adapter.

## Candidate receipt and freshness

Each source record retains its latest command, node-clock receipt time,
publisher stamp, frame, finite/static-valid decision, reason, and update
count. Freshness is computed from node-clock receipt age, never from publisher
stamp alone. Timeouts are source-specific.

A source is healthy only when it has been received and its latest command:

- is within its configured receipt timeout;
- uses `px4_ned` when wrong-frame rejection is enabled;
- has finite stamp and all six twist components;
- has exact-zero `angular.x/y` within the HOLD epsilon;
- obeys total, horizontal, vertical, and yaw-rate limits;
- has a strictly increasing publisher stamp when monotonic stamps are
  required; and
- for `HOLD`, has magnitude no greater than the HOLD epsilon.

Equal or older publisher stamps are rejected and do not refresh receipt-time
health. A malformed update makes that source unhealthy but does not interrupt
an unrelated active source. If the malformed or stale source is active, the
mux immediately emits internal zero HOLD and never automatically fails over to
another movement source.

## Selection and arbitration

Startup uses `requested_source=HOLD`, `active_source=HOLD`, and
`HOLD_STARTUP`. A `HOLD` request is always accepted immediately, clears any
pending transition, resets selected-command rate history, and produces exact
zero `HOLD_REQUESTED` output.

A movement-source request is accepted only if the identifier is canonical,
the source has a fresh valid candidate, the minimum dwell period has elapsed,
and a fresh pre-switch command exists when configured. Rejection is explicit
in the service response. Unknown requests fail closed to `HOLD_INVALID_SOURCE`.
Requests for the already-active source are idempotent and do not increment the
transition count.

Switching from one movement source to a different movement source follows this
sequence:

```text
current source -> accepted request -> exact-zero HOLD_SWITCH_BARRIER
               -> revalidate target -> target source
```

The barrier lasts at least `switch_hold_duration_s`. There is no blending or
interpolation. If the target becomes stale or invalid during the barrier, the
transition is cancelled and the mux remains fail-closed HOLD. The target must
be explicitly requested again after valid fresh data is available.

Movement commands are bounded in order by horizontal, vertical, total speed,
linear acceleration, yaw rate, and yaw acceleration. Acceleration histories
never cross source ownership. A barrier or explicit/fault HOLD resets the
movement source history to exact zero, so a newly activated source ramps from
the safe selected HOLD under the configured rate bounds.

## Fault latching and time behavior

With `latch_hold_after_fault=true`, active-source staleness, wrong frame,
non-finite data, non-monotonic stamp, selected-validator rejection, backward
ROS time, or an internal inconsistency latches HOLD. Fresh messages alone
cannot restore movement. Recovery requires fresh valid data followed by an
explicit valid `set_source` request.

A backward node-clock jump clears candidate freshness evidence, transition
state, dwell history, and rate history, then enters `HOLD_TIME_JUMP`. Equal
time does not advance command-rate history. All runtime timestamps use the
node ROS clock, respecting `use_sim_time`; wall/monotonic time is limited to
finite harness deadlines.

## Independent selected-command validation

The validator is separate from the registry and mux command generation. For
every output it independently checks:

- canonical and active-source-consistent ownership;
- exact `px4_ned` frame and finite fields;
- exact-zero `angular.x/y`;
- total, horizontal, vertical, and yaw-rate bounds;
- acceleration and yaw-acceleration against the previous independently valid
  selected command;
- strictly increasing output timestamps;
- exact-zero magnitude and nonempty reason for HOLD; and
- HOLD/active-state consistency.

Diagnostics identify constraint, source, measured value, limit, cycle,
timestamp, and reason. A candidate validation failure is never published as a
movement command: the mux constructs a fresh internal zero HOLD and validates
that fallback independently. Failure of the internally constructed HOLD is an
internal inconsistency and no unsafe candidate may be forwarded.

The selected `TwistStamped.header.stamp` is generated from current mux ROS
time. Candidate publisher stamps and receipt ages remain separate registry
diagnostics.

## Status and conservative defaults

`ControlMuxStatus` reports requested/active source, selected validity, HOLD
state/reason, switch progress and remaining time, selected-source age, selected
speed/yaw rate, transition count, healthy/stale source lists, and a concise
state/diagnostic message. Publication is fixed at 50 Hz with reliable,
volatile, keep-last QoS.

The 19 configuration fields use the locked defaults: `HOLD`, 50 Hz; source
timeouts `0.25/0.20/0.20/0.50 s`; switch HOLD `0.10 s`; dwell `0.20 s`;
speed limits `2.0/2.0/1.0 m/s`; acceleration `1.5 m/s^2`; yaw rate
`1.5 rad/s`; yaw acceleration `2.0 rad/s^2`; wrong-frame, monotonic stamp,
fresh-before-switch, and fault latch enabled; HOLD epsilon `1e-9`. All numeric
configuration is finite and nonnegative, with rates and limits positive and
component limits no greater than the total speed limit.

## Offline validation graph

Synthetic candidate publishers support fixed, varying, stale, wrong-frame,
non-finite, excessive, non-monotonic, delayed, and shutdown behavior. They do
not read hardware or load a policy. The existing deterministic first-order
kinematic plant gains a configurable command topic; Phase 5 keeps
`astar_command`, while the Phase 6 graph uses only `selected_command`.

The direct mux graph validates source selection and handoff. The safety graph
validates stale/invalid/time/dwell/latch behavior. The full graph is:

```text
fixed scene -> A* -> B-spline/fallback -> timed trajectory -> follower
            -> astar candidate -> mux -> selected command -> offline plant
```

Every graph is finite, checks exact source ownership and command bounds, and
rejects any `/fmu/in/*` topic. The plant remains a deterministic kinematic
fixture, not UAV dynamics.
