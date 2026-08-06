# Phase 4 Trajectory Parameterization Design

Phase 4 adds deterministic, offline time parameterization after the Phase 3
final path. It is non-flight software: it does not command a vehicle, implement
a controller, or publish PX4 input topics.

## Data flow and ownership

The only geometric input is `nav_msgs/msg/Path` on `/uav/planner/path`. The
trajectory node verifies the `px4_ned` frame, removes adjacent duplicate points,
and preserves every remaining position exactly. Pose timestamps and
orientations are intentionally ignored. The node publishes:

- `/uav/trajectory/candidate` (`uav_interfaces/msg/TimedTrajectory`)
- `/uav/trajectory/valid` (`std_msgs/msg/Bool`)
- `/uav/trajectory/status` (`std_msgs/msg/String`)

The trajectory candidate is downstream data only. No follower, controller,
arming, takeoff, landing, OFFBOARD transition, simulator, XRCE bridge, or
`/fmu/in/*` publisher is part of this phase.

## Pure computation boundary

`trajectory_models.py`, `trajectory_parameterizer.py`,
`trajectory_validator.py`, `yaw_profile.py`, and `trajectory_metrics.py` are
ROS-independent. Only `trajectory_parameterizer_node.py` converts ROS messages,
owns publishers/subscriptions, and runs deterministic offline harnesses.

The parameterizer returns a structured result rather than throwing for expected
input rejection. The independent validator receives the original cleaned path,
the generated trajectory, frame, and limits; it does not trust the
parameterizer's success flag.

## Deterministic algorithm

1. Validate the frame, configuration, finiteness, point count, and nonzero
   adjacent 3-D distances.
2. Compute strictly increasing 3-D cumulative arc length and pointwise planar
   curvature in NED north/east coordinates.
3. Bound speed by both `maximum_speed_mps` and
   `sqrt(maximum_lateral_acceleration_mps2 / max(abs(curvature), epsilon))`.
4. Apply forward acceleration and backward deceleration passes in squared-speed
   form, including configured start and end speeds.
5. Assign segment duration with `2 * ds / (v_i + v_j)`. A conservative
   acceleration-limited duration handles a zero denominator; every duration is
   also bounded by `minimum_segment_time_s`.
6. Derive NED tangent velocity, tangential plus curvature-induced acceleration,
   finite-difference jerk, unwrapped `atan2(east, north)` yaw, yaw rate, and yaw
   acceleration.
7. Recompute the full candidate under one global time scale until every dynamic
   limit passes or the configured iteration/scale budget is exhausted.
8. Run the independent validator. Only that result controls the published
   validity flag.

Under a global scale `s`, velocity and yaw rate scale as `1/s`, acceleration and
yaw acceleration as `1/s^2`, and jerk as `1/s^3`. Geometry, arc length, and
curvature never change.

## Failure and publication semantics

Every received non-duplicate path produces `/uav/trajectory/valid` and a
machine-readable status string. A finite structured candidate is published even
when dynamic validation rejects it, with `valid=false`; malformed geometry that
cannot form a finite trajectory produces no candidate. Repeated identical path
messages are ignored. Status includes result, source, point count, duration,
scale, and either maxima or the first diagnostic (`constraint`, point index,
measured value, and limit).

## Safety invariants

- Frame is exactly `px4_ned`.
- Output position sequence equals the cleaned input sequence.
- Arc length and time are finite and strictly increasing.
- Start/end speed constraints and all configured dynamic limits are independently
  checked.
- No quaternion orientation is used to derive yaw.
- No Phase 0 legacy source is edited, and no flight or simulator runtime is
  introduced.
