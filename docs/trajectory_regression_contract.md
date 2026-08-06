# Trajectory Regression Contract

This contract freezes the Phase 4 offline trajectory behavior.

## Inputs and outputs

- Input topic: `/uav/planner/path`, type `nav_msgs/msg/Path`, frame `px4_ned`.
- Candidate topic: `/uav/trajectory/candidate`, type
  `uav_interfaces/msg/TimedTrajectory`.
- Validity topic: `/uav/trajectory/valid`, type `std_msgs/msg/Bool`.
- Status topic: `/uav/trajectory/status`, type `std_msgs/msg/String`.
- Input pose timestamps and orientations have no semantic meaning.
- Adjacent duplicate positions are removed; all other positions and altitude are
  unchanged and retain order.

## Acceptance invariants

For every accepted candidate:

- there are at least two points and every numeric field is finite;
- time begins at zero and is strictly increasing;
- arc length begins at zero and is strictly increasing;
- geometry exactly equals the cleaned source path;
- speed, longitudinal acceleration/deceleration, lateral acceleration, jerk,
  yaw rate, and yaw acceleration do not exceed configured limits (within the
  documented numerical tolerance);
- requested zero/nonzero start and end speeds are respected;
- yaw is the continuous unwrapped NED heading `atan2(east, north)`;
- `valid=true` is published only after independent validation.

Rejections identify the failed constraint, point index, measured value, and
limit whenever those values exist. Wrong-frame, fewer-than-minimum-point, and
non-finite inputs are rejected deterministically. An identical path received
again does not trigger another computation or publication.

## Required deterministic fixtures

The standalone harness supports: `straight-line`, `phase3-bspline`,
`sharp-bend`, `high-curvature`, `duplicate-adjacent`, `two-point`,
`invalid-one-point`, `nonfinite`, `yaw-wrap`, `jerk-scaling`,
`impossible-config-rejection`, and `wrong-frame`.

`./uav trajectory-check` runs the standalone graph. `./uav pipeline-check` runs
the fixed Phase 3 scene through A*, optional B-spline selection, and Phase 4
parameterization. Both execute in a clean ROS 2 Jazzy environment, write an
ignored runtime log, scan for `/fmu/in/*`, forward launch arguments, and return
nonzero unless their expected result is observed.

## Non-goals

This phase contains no follower, controller, setpoint conversion, OFFBOARD or
arming logic, takeoff/landing behavior, simulator integration, camera/recorder,
NavRL, XRCE agent, or PX4 input publication. A valid candidate is analysis data,
not authorization to fly.
