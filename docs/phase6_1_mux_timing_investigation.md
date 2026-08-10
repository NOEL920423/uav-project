# Phase 6.1 mux timing investigation

## Scope and preserved evidence

This investigation is limited to the offline Phase 6 mux regression. It does
not start PX4, XRCE-DDS, Isaac Sim, Pegasus, OFFBOARD, arming, flight, or any
`/fmu/in/*` publisher. The original log remains unchanged at
`run_logs/mux-check_20260807T091620Z.log`.

The failing run started the ASTAR synthetic publisher at ROS time
`1786094180.858040739`. The monitor observed:

| ROS time | Observation |
|---:|---|
| `1786094180.886008512` | `HOLD_STARTUP`, active `HOLD` |
| `1786094180.915282504` | `ACTIVE_ASTAR_EXPERT` |
| `1786094181.256321977` | `HOLD_STALE_SOURCE` |
| `1786094181.274800934` | `HOLD_LATCHED_FAULT` |
| `1786094188.849890177` | timeout at monitor stage `WAIT_ASTAR` |

The ACTIVE-to-stale interval was about `0.341 s`, greater than the locked
`astar_timeout_s=0.25 s`. The stale and latched states were therefore the
intended production safety response. The defect was that the test monitor had
already logged ACTIVE in its status callback but had not yet consumed the
asynchronous service future. A later timer cycle advanced from
`REQUEST_ASTAR` to `WAIT_ASTAR` after the ACTIVE sample had been overwritten.
The monitor consequently reported that activation was never observed even
though its own log proved otherwise.

## Contract and code findings

- The synthetic publisher creates a new `TwistStamped` every `0.04 s`, refreshes
  `header.stamp`, and does not suppress identical velocity payloads.
- `ControlMuxNode` records its own ROS-clock receipt time in every candidate
  callback.
- `ControlSourceRegistry.update` replaces the receipt time and increments
  `update_count` for every arrival. Payload equality is not considered.
- Freshness is computed only from node receipt age. Publisher stamps are an
  independent strict-monotonicity validity gate.
- The wrapper has no retry. It runs the launch under a 20-second outer timeout,
  requires the monitor success marker, and separately scans for `/fmu/in/*`.
- The monitor used a 50 ms timer to poll an asynchronous service future and
  stored only the latest status sample. It could therefore miss an ACTIVE event
  that arrived between service execution and response processing.
- The monitor also advanced to a request stage even when `_request` had not
  submitted a request because the service was not yet ready.

## Root-cause classification

Classification: **F, multiple interacting causes**, with **D, monitor/test
synchronization race**, as the correctable software cause.

The original log proves that the mux received no valid ASTAR heartbeat for more
than 250 ms after activation, but it cannot distinguish publisher scheduling
from DDS delivery scheduling. A controlled 12-second observation on 2026-08-10
measured `24.998 Hz`, `0.039-0.041 s` intervals, and strictly increasing stamps,
so there is no evidence of a persistent publisher, stamp, duplicate-payload, or
receipt-bookkeeping defect. The original startup gap exposed the monitor race;
it did not justify weakening the safety timeout.

## Minimal fix

The offline monitor now:

1. independently observes candidate arrivals;
2. requires at least three recent arrivals with increasing stamps before a
   movement-source request;
3. advances stages only when a service request was actually submitted;
4. retains observed ACTIVE events across asynchronous service-response timing;
5. waits for target candidate readiness before each handoff; and
6. immediately fails a nominal/control-stack fixture on unexpected stale or
   latched HOLD, with candidate count, maximum observed gap, and stamp health in
   the summary.

Three 40 ms heartbeats provide startup continuity evidence without using a
fixed sleep. The 120 ms readiness-age bound is less than half the 250 ms ASTAR
timeout. Production `astar_timeout_s`, receipt-time logic, stale HOLD, fault
latch, internal exact-zero HOLD, and explicit recovery are unchanged.

## Verification results

The focused pure/static subset reported `45 passed`. The complete seven-package
workspace reported `217 tests, 0 errors, 0 failures, 0 skipped`.

The required nominal stability gate passed 10/10 consecutive runs. Every run
observed A* activation, both movement-source HOLD barriers, joystick and NavRL
activation, explicit terminal HOLD, monotonic ASTAR stamps, and the success
marker. No run entered `HOLD_STALE_SOURCE` or `HOLD_LATCHED_FAULT`.

Eight nominal runs had a maximum ASTAR inter-arrival gap between `0.040537 s`
and `0.041197 s`. Two runs captured startup gaps of `0.367503 s` and
`0.367946 s`, both greater than the production timeout. In both cases the new
readiness condition correctly withheld source selection until three recent
heartbeats had arrived. A* then remained healthy while selected and the nominal
sequence completed. This reproduces the original timing class without hiding
it or weakening stale detection.

The intentional stale-source gate passed 3/3 consecutive runs. Each run
observed `ACTIVE_ASTAR_EXPERT -> HOLD_STALE_SOURCE -> HOLD_LATCHED_FAULT`, kept
movement disabled when traffic resumed, and recovered only after the explicit
source request. The recorded intentional gaps were `0.639714-0.640463 s`.

The single post-fix checks and complete regression all passed:

- `mux-check`, `mux-safety-check`, and `control-stack-check`;
- `doctor`, `build`, `test`, and `verify`;
- every Phase 2 through Phase 7 offline wrapper listed in the Phase 6.1 task;
- the final graph contained only `/parameter_events` and `/rosout`; and
- every wrapper scan reported `SAFE: no /fmu/in/* topics detected`.

The legacy source-tree hash remained
`9bb394c0e4e5616f0857ce61e5067971a5931ae5c32a0e33ef3d96af40b94beb`.
