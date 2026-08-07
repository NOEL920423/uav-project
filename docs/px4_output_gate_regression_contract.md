# PX4 output gate regression contract

Phase 7 stops at a diagnostic permission boundary. `safe_to_forward=true` is
evidence that a future publisher *could* consume the current candidate; it does
not publish or authorize a PX4 uORB/ROS input topic.

## Required evidence

- a fresh, independently validated velocity-only candidate;
- a fresh Phase 6 mux status which is valid, not in HOLD, and names the same
  canonical active source;
- fresh synthetic PX4 telemetry with monotonically nondecreasing timestamps;
- connected, acceptable standby/armed and navigation values;
- passing preflight evidence, valid local position/velocity/odometry, and NED
  pose/velocity frames;
- no failsafe and no offboard-control-signal-loss evidence;
- an explicit output-enable request followed by one complete healthy cycle.

Disarmed (`ARMING_STATE_STANDBY`) and OFFBOARD-inactive telemetry can be healthy
for this diagnostic gate. That is intentional: Phase 7 does not choose arming
or mode-switch sequencing. It only determines whether a candidate is safe to
hand to a future, separately reviewed Phase 8 publisher/sequencer.

## Fail-closed and recovery rules

The startup state is `OUTPUT_DISABLED`. Missing inputs use explicit `WAITING_*`
states. Stale command/telemetry, invalid mapping, mux HOLD, failsafe, backward
time, or an unexpected vehicle-state change immediately makes
`safe_to_forward=false`. When an enable request was active, the fault latches.
Fresh data cannot clear the latch. Recovery is exactly:

1. request disable/reset;
2. repair and revalidate all evidence;
3. request enable;
4. complete one additional healthy gate cycle.

There is no automatic enable, retry, mode switch, arm request, setpoint
streaming, or `/fmu/in/*` publisher in this contract.
