# Phase 5 offline trajectory tracking design

## Scope and safety boundary

Phase 5 evaluates a ROS-level trajectory-follower candidate in a deterministic
closed loop. It is offline and non-flight. It does not start or control PX4,
Isaac Sim, Pegasus, Micro XRCE-DDS, cameras, a control-source mux, OFFBOARD,
arming, takeoff, landing, or a real vehicle. In particular,
`/uav/control/astar_command` is never remapped or forwarded to `/fmu/in/*`.

The command is only a candidate contract. Its NED fields have not been proven
to be a PX4 setpoint mapping, and a zero candidate is not evidence that a real
vehicle would hold position.

## Data flow and module boundary

```text
TimedTrajectory + Bool validity + px4_ned Odometry
  -> deterministic reference sampler
  -> feedforward plus position/velocity/yaw feedback
  -> ordered command bounds
  -> independent tracking-command validator
  -> /uav/control/astar_command (TwistStamped candidate)
  -> deterministic first-order kinematic plant
  -> /uav/vehicle/odometry
  -> state machine and Offline Closed-Loop Tracking Metrics
```

The pure modules are `tracking_models.py`, `trajectory_sampler.py`,
`trajectory_tracker.py`, `tracking_validator.py`, `tracking_metrics.py`, and
`offline_kinematic_plant.py`. They import neither ROS nor simulator/PX4 APIs.
`trajectory_follower_node.py` is the ROS adapter. The existing Phase 4
`TrajectoryPoint` representation and unwrapped NED yaw convention are reused.

## Frames, time, and topics

All trajectory, odometry, reference, command, and status headers use
`px4_ned`. Candidate `TwistStamped` semantics are:

- `linear.x/y/z`: north/east/down velocity in m/s
- `angular.z`: NED yaw-rate in rad/s
- `angular.x/y`: exactly zero

The follower subscribes to `/uav/trajectory/candidate`,
`/uav/trajectory/valid`, and `/uav/vehicle/odometry`. It publishes
`/uav/control/astar_command`, `/uav/control/astar_reference_pose`,
`/uav/control/astar_reference_twist`, and
`/uav/control/astar_tracking_status`.

`TimedTrajectory.header.stamp` identifies receipt/publication; it is not an
absolute start time. A newly accepted non-duplicate trajectory records node
receipt time and defines:

```text
tracking_epoch = receipt_time + trajectory_start_delay_s
trajectory_time = current_ros_time - tracking_epoch
```

Before the epoch the state is `PRESTART_HOLD`. A future supervisor may replace
this receipt-relative rule with an explicit synchronized start contract.

Backward ROS time clears command history and all synchronization evidence,
enters `HOLD_TIME_JUMP`, and requires a fresh trajectory, validity, and
odometry before tracking can resume. Equal time never advances rate history.

## Reference sampling and tracking law

Sampling is deterministic binary search over strictly increasing
`time_from_start`. Time below zero returns the exact first point with a
pre-start flag; time at or beyond the duration returns the exact final point
with a terminal flag. Interior samples linearly interpolate position,
velocity, acceleration, jerk, arc length, curvature, unwrapped yaw, yaw rate,
and yaw acceleration. Non-finite fields or invalid timestamps are rejected.

For a finite measured state, the unbounded candidate is:

```text
v_cmd = v_ref + position_kp * (p_ref - p)
                  + velocity_kd * (v_ref - v)
yaw_rate_cmd = yaw_rate_ref + yaw_kp * wrap(yaw_ref - yaw)
```

No roll, pitch, attitude, thrust, motor, or PX4 command is calculated.
Bounding order is fixed: reject non-finite; horizontal speed; vertical speed;
total speed; linear acceleration versus the preceding valid command; yaw
rate; yaw acceleration; then independent validation. Every clamp sets a
specific saturation flag and both unsaturated and selected commands remain in
the structured result.

## HOLD and state machine

HOLD is exactly zero linear velocity and zero angular velocity. A specific
reason accompanies every HOLD cycle. The explicit states are:

```text
WAITING_TRAJECTORY  WAITING_VALIDITY  WAITING_ODOMETRY  PRESTART_HOLD
TRACKING            GOAL_SETTLING    GOAL_HOLD
HOLD_STALE_TRAJECTORY  HOLD_STALE_ODOMETRY  HOLD_INVALID_FRAME
HOLD_TIME_JUMP      HOLD_TRACKING_ERROR     HOLD_INVALID_COMMAND
TERMINAL_NOT_REACHED
```

Input gates are evaluated before tracking. Missing trajectory, validity, or
odometry selects the matching waiting state; false/stale validity, stale
odometry, wrong frame, non-finite state, backward time, excessive tracking
error, and a failed independent validation select HOLD. The follower does not
recover or replan.

After reference time reaches the end, position, measured speed, and wrapped
yaw error must remain within their respective tolerances for
`goal_settle_time_s`. The state progresses `TRACKING -> GOAL_SETTLING ->
GOAL_HOLD`. If this does not happen within `maximum_terminal_wait_s`, the
state becomes `TERMINAL_NOT_REACHED` and remains HOLD.

## Independent validation

The validator is separate from command generation. It checks frame, finite
fields, total/horizontal/vertical speed, yaw rate, acceleration and yaw
acceleration relative to the previous independently valid command, monotonic
timestamps, stale state indicators, HOLD magnitude, and state/HOLD
consistency. Diagnostics carry the constraint, measured value, limit,
timestamp, and cycle index. Only a validated selected command is published.

## Deterministic plant and metrics

The plant uses fixed-step integration with first-order velocity and yaw-rate
response, acceleration limiting, configurable initial state and constant
disturbance, and optional deterministic measurement noise disabled by default.
It is a kinematic test fixture, not a UAV dynamics simulator.

The accumulator reports Offline Closed-Loop Tracking Metrics: position,
horizontal, vertical, velocity, and yaw RMSE; maximum/terminal errors;
maximum command speed, acceleration, yaw rate, and yaw acceleration;
saturation and HOLD counts; stale-detection latency; terminal settling time;
and completion state. These are deterministic offline measurements, not real
flight performance.

Command derivative maxima cover ordinary controller candidates. An immediate
exact-zero fail-closed HOLD is a safety override, so that transition is counted
as a HOLD cycle but is not interpreted as a dynamically rate-limited tracking
command. The next ordinary candidate is still compared with the preceding
published HOLD.

## Conservative defaults

`TrackingConfig` validates all values as finite and coherent. Defaults are
`position_kp=1.0`, `velocity_kd=0.2`, `yaw_kp=1.5`, speed limits
`2.0/2.0/1.0 m/s` (total/horizontal/vertical), acceleration `1.5 m/s^2`, yaw
rate `1.5 rad/s`, yaw acceleration `2.0 rad/s^2`, odometry timeout `0.25 s`,
validity timeout `0.50 s`, start delay `0.10 s`, control period `0.02 s`, goal
tolerances `0.15 m`, `0.15 m/s`, `0.20 rad`, settle time `0.50 s`, maximum
tracking error `2.0 m`, terminal wait `2.0 s`, wrong-frame rejection and
required validity enabled, and HOLD epsilon `1e-9`. These are offline
engineering defaults, not calibrated real-UAV gains or limits.
