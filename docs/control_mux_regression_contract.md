# Phase 6 control mux regression contract

## Safety invariants

Every published selected command has a mux-clock stamp, frame `px4_ned`,
finite fields, zero `angular.x/y`, and passes the independent selected-command
validator. Normal movement obeys total/horizontal/vertical speed,
acceleration, yaw-rate, and yaw-acceleration limits. Every non-movement output
is exact zero with `active_source=HOLD` and a nonempty reason.

Only the explicitly active movement source can be forwarded. Healthy
unselected candidates cannot affect the output, and an unhealthy unselected
candidate cannot interrupt the active source. No fault causes automatic
movement-source failover. A latched fault requires fresh data and an explicit
valid service request before movement resumes.

Movement-to-movement selection includes at least the configured exact-zero
switch barrier. The target is revalidated after the barrier. No graph or
module publishes or remaps a selected command to `/fmu/in/*`.

## Deterministic fixtures

| # | Fixture | Expected service/state/output sequence |
|---:|---|---|
| 1 | startup with no source | `HOLD_STARTUP`, exact zero |
| 2 | select fresh A* | accepted, `ACTIVE_ASTAR_EXPERT`, bounded A* only |
| 3 | active A* becomes stale | `HOLD_STALE_SOURCE`, exact zero |
| 4 | stale fault latch | fresh candidate alone remains `HOLD_LATCHED_FAULT` |
| 5 | explicit A* recovery | accepted fresh request, active A* resumes |
| 6 | A* to joystick | accepted, A* -> barrier -> joystick |
| 7 | switch barrier observed | exact zero for complete configured duration |
| 8 | target stale during handoff | barrier -> fail-closed stale HOLD |
| 9 | joystick to NavRL | accepted, joystick -> barrier -> NavRL |
| 10 | unknown source | rejected, `HOLD_INVALID_SOURCE`, exact zero |
| 11 | selected wrong frame | `HOLD_WRONG_FRAME`, latched exact zero |
| 12 | selected non-finite | `HOLD_INVALID_COMMAND`, latched exact zero |
| 13 | excessive selected speed | rejected/fail-closed bounded contract |
| 14 | non-monotonic candidate stamp | `HOLD_INVALID_COMMAND`, latched zero |
| 15 | backward node time | histories cleared, `HOLD_TIME_JUMP` |
| 16 | minimum dwell | early request rejected without mixed ownership |
| 17 | duplicate active request | idempotent, no transition increment |
| 18 | explicit HOLD | immediately accepted exact `HOLD_REQUESTED` |
| 19 | invalid external HOLD | internal HOLD remains available and valid |
| 20 | simultaneous sources | only selected source is forwarded |
| 21 | unselected source stale | active healthy source remains uninterrupted |
| 22 | selected acceleration limit | output delta obeys `1.5 m/s^2` |
| 23 | selected yaw-acceleration limit | output delta obeys `2.0 rad/s^2` |
| 24 | follower through mux and plant | A* candidate only, selected owner only, terminal `GOAL_HOLD` |

Each fixture records the request/response, requested and active source,
command sequence, HOLD reason/cycles, source age, transition count, latch and
recovery behavior, expected terminal state, and observed terminal state.

## Required pure coverage

- exact source identifiers, topic mapping, timeout lookup, and unknown-source
  rejection;
- configuration finiteness, positivity, coherence, and boolean typing;
- never-received, fresh, boundary-age, stale, wrong-frame, non-finite,
  excessive, and non-monotonic candidate classification;
- startup HOLD, explicit HOLD, direct activation, idempotent request, dwell
  rejection, movement switch barrier, target revalidation, and cancelled
  handoff;
- active fault latching, fresh-data non-recovery, explicit recovery, backward
  time reset, and unselected-source isolation;
- horizontal, vertical, total, acceleration, yaw-rate, and yaw-acceleration
  bounds with source-history reset;
- independent validator ownership, frame, finiteness, angular x/y, every
  bound, timestamp, HOLD magnitude/reason, and active-state consistency;
- independent fallback HOLD validation and structured diagnostics; and
- all 24 deterministic fixtures plus comparison-table generation.

## Required ROS graph coverage

The direct finite graph contains only synthetic candidate publishers, the mux,
and an independent monitor. It observes startup HOLD, successful A* selection,
a complete movement-to-movement zero barrier, exact source ownership, all
three output topics, service response fields, and no `/fmu/in/*` publisher.
Before requesting a movement source, the monitor must observe at least three
recent candidate arrivals with strictly increasing stamps and independently
confirm that the mux reports the source healthy. Submitted service requests and
observed ACTIVE events are sticky synchronization evidence: a later status
sample cannot erase an activation that arrived before the asynchronous service
response. Nominal graphs fail immediately on any stale-source or latched HOLD.

The safety graph covers at least active-source staleness, fault latch,
fresh-data non-recovery, explicit recovery, wrong frame, non-finite input,
excessive input, non-monotonic publisher stamps, target staleness during
handoff, external-HOLD failure, and fail-closed internal HOLD.

The control-stack graph adds only the existing offline scene, planner,
trajectory parameterizer, A* follower, selected-command mux, deterministic
plant, and finite monitor. The plant subscribes to selected command only. It
must observe A* candidate publication, accepted `ASTAR_EXPERT` selection,
selected-command ownership, bounded continuous commands, tracking progression,
and terminal `GOAL_HOLD`.

Required wrapper gates are:

```bash
./uav mux-check
./uav mux-safety-check
./uav control-stack-check
```

The complete Phase 2–5 wrappers and the whole-workspace test suite remain
mandatory. Static scans must prove that the follower owns only
`astar_command`, synthetic joystick and NavRL fixtures own only their candidate
topics, the safety fixture owns only `hold_command`, only the mux owns
`selected_command`, pure modules contain no forbidden runtime import, and
tracked output contains no build, log, cache, bag, model, or dataset artifact.

## Completion boundary

Passing this regression contract establishes deterministic offline ROS-level
source arbitration only. It does not validate PX4 setpoint mapping, OFFBOARD,
arming, a hardware joystick, a NavRL policy/runtime/model, Isaac Sim, vehicle
dynamics, real disturbance rejection, or flight.
